//! World persistence: one directory holds one `redb` database per world
//! (task 2.3).
//!
//! Layout (see the crate README for the full contract):
//! - `meta`: `format_version`, `seed` and test bookkeeping.
//! - `block_states`: on-disk state id → canonical state string. Ids are
//!   stable across restarts even when the runtime registry orders states
//!   differently, so cubes decode to the same blocks.
//! - `cubes`: 13-byte position key → `[record schema][zstd(codec bytes)]`.
//!
//! A cube record is written by exactly one transaction, so a crash never
//! leaves a half-written cube or a half-written batch. [`Storage`] is
//! `Send + Sync`: loads run on any thread (generation workers), writes run
//! on the server's `jbm-save` thread through [`Storage::save_batch`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;

use jbm_core::block::{BlockRegistry, BlockStateId, StateInfo};
use jbm_core::coords::{CUBE_VOLUME, CubePos};
use jbm_core::cube::Cube;
use redb::{
    Database, Durability, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition,
};
use zstd::stream::decode_all;

/// `meta` table: small named blobs (`format_version`, `seed`, …).
const META: TableDefinition<'_, &str, &[u8]> = TableDefinition::new("meta");
/// On-disk block state id → canonical `namespace:path[prop=value,…]` string.
const BLOCK_STATES: TableDefinition<'_, u32, &str> = TableDefinition::new("block_states");
/// Position key (below) → cube record.
const CUBES: TableDefinition<'_, &[u8; 13], &[u8]> = TableDefinition::new("cubes");

/// Current on-disk format version; a database written by a newer format is
/// rejected with [`StorageError::VersionTooNew`] instead of misread.
const FORMAT_VERSION: u32 = 1;

/// Cube record schema byte stored in front of every value in `cubes`.
const RECORD_SCHEMA: u8 = 1;

/// zstd level for cube records.
const ZSTD_LEVEL: i32 = 3;

/// Ways persistence can fail. Typed so callers (and tests) can match on the
/// case instead of parsing strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    Io(String),
    /// The database file is locked by another holder; never waited on.
    Locked(PathBuf),
    /// A stored record could not be decoded. The record is left untouched —
    /// callers regenerate the cube instead.
    Corrupt {
        pos: CubePos,
        reason: String,
    },
    /// A `meta` entry (named here) has the wrong shape.
    CorruptMeta(&'static str),
    /// The database names a block state the running registry does not know.
    UnknownBlockState(String),
    /// The database was written by a newer format version.
    VersionTooNew,
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message) => write!(f, "storage I/O error: {message}"),
            Self::Locked(path) => write!(f, "world database {path:?} is locked by another holder"),
            Self::Corrupt { pos, reason } => write!(f, "corrupt record at {pos:?}: {reason}"),
            Self::CorruptMeta(key) => write!(f, "corrupt world meta entry {key:?}"),
            Self::UnknownBlockState(name) => write!(f, "unknown block state {name:?}"),
            Self::VersionTooNew => write!(f, "world database format is newer than this build"),
        }
    }
}

impl std::error::Error for StorageError {}

fn meta_corrupt(key: &'static str) -> StorageError {
    StorageError::CorruptMeta(key)
}

/// Upper bound on a cube's codec bytes (header, a full palette, and one
/// word per block — looser than any real width), so a corrupt frame header
/// cannot make decode allocate without limit.
const MAX_PLAIN_BYTES: usize = 16 + 4 * CUBE_VOLUME + 8 * CUBE_VOLUME;

fn io_err(error: impl std::fmt::Display) -> StorageError {
    StorageError::Io(error.to_string())
}

fn db_err(path: &Path, error: redb::DatabaseError) -> StorageError {
    match error {
        redb::DatabaseError::DatabaseAlreadyOpen => StorageError::Locked(path.to_owned()),
        redb::DatabaseError::UpgradeRequired(_) => StorageError::VersionTooNew,
        other => io_err(other),
    }
}

/// 13-byte cube key: `[kind][x][y][z]`, each coordinate an i32 big-endian
/// with the sign bit flipped so keys sort by coordinate value. `kind = 0` is
/// a LOD0 cube; 1..=255 are reserved (LOD cubes come in task 2.7).
fn cube_key(pos: CubePos) -> [u8; 13] {
    fn field(out: &mut [u8], v: i32) {
        let bytes = v.to_be_bytes();
        out.copy_from_slice(&bytes);
        out[0] ^= 0x80;
    }
    let mut key = [0u8; 13];
    field(&mut key[1..5], pos.x);
    field(&mut key[5..9], pos.y);
    field(&mut key[9..13], pos.z);
    key
}

/// Inverse of [`cube_key`] for kind-0 keys; `None` for reserved kinds.
fn cube_pos(key: &[u8; 13]) -> Option<CubePos> {
    fn field(bytes: &[u8]) -> i32 {
        let mut raw = [0u8; 4];
        raw.copy_from_slice(bytes);
        raw[0] ^= 0x80;
        i32::from_be_bytes(raw)
    }
    (key[0] == 0).then(|| CubePos {
        x: field(&key[1..5]),
        y: field(&key[5..9]),
        z: field(&key[9..13]),
    })
}

/// Canonical state string: `namespace:path[prop=value,…]` with properties in
/// declaration order, no brackets when the block has none.
fn canonical_state(info: &StateInfo) -> String {
    if info.properties.is_empty() {
        return info.name.clone();
    }
    let props = info
        .properties
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{}[{}]", info.name, props)
}

/// Test hooks shared between the [`Storage`] handle and the save thread (see
/// [`Storage::hold_saves`] and [`Storage::fail_next_save`]).
#[derive(Default)]
struct Hooks {
    state: Mutex<HookState>,
    released: Condvar,
}

#[derive(Default)]
struct HookState {
    held: bool,
    /// Saves allowed through while held (`release_one_save`).
    passes: u32,
    fail_next: bool,
}

impl Hooks {
    fn fail_next_save(&self) {
        self.state.lock().unwrap().fail_next = true;
    }

    fn hold(&self) {
        self.state.lock().unwrap().held = true;
    }

    fn release(&self) {
        let mut state = self.state.lock().unwrap();
        state.held = false;
        self.released.notify_all();
    }

    fn release_one(&self) {
        let mut state = self.state.lock().unwrap();
        state.passes += 1;
        self.released.notify_all();
    }

    /// Blocks while saves are held, then applies the one-shot fail injection.
    fn enter_save(&self) -> Result<(), StorageError> {
        let mut state = self.state.lock().unwrap();
        while state.held && state.passes == 0 {
            state = self.released.wait(state).unwrap();
        }
        if state.held {
            state.passes -= 1;
        }
        if state.fail_next {
            state.fail_next = false;
            return Err(StorageError::Io("injected save failure (test hook)".into()));
        }
        Ok(())
    }
}

/// An open world database. Cheap to share as `Arc<Storage>`; loads may run
/// on any thread, writes serialize through redb's single writer.
pub struct Storage {
    db: Database,
    path: PathBuf,
    seed: u64,
    /// Runtime state id → on-disk id, indexed by `BlockStateId.0`.
    runtime_to_disk: Vec<u32>,
    /// On-disk id → runtime state id.
    disk_to_runtime: HashMap<u32, BlockStateId>,
    hooks: Arc<Hooks>,
}

/// `redb::Database` is not `Debug`; summarize the handle instead.
impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("path", &self.path)
            .field("seed", &self.seed)
            .field("states", &self.runtime_to_disk.len())
            .finish_non_exhaustive()
    }
}

impl Storage {
    /// Opens (or creates) `<dir>/world.redb`, creating the directory if
    /// missing. On reopen the **stored** seed wins over `seed`; a mismatch is
    /// logged once. The effective seed is available as [`Storage::seed`], so
    /// the server builds its generator from it.
    ///
    /// The `block_states` remap tables are built in one transaction: states
    /// missing from the table are appended with fresh ids in the same
    /// transaction, and a stored name the registry lacks fails with
    /// [`StorageError::UnknownBlockState`].
    pub fn open(
        dir: impl AsRef<Path>,
        registry: &BlockRegistry,
        seed: u64,
    ) -> Result<Self, StorageError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(io_err)?;
        let path = dir.join("world.redb");
        let db = Database::create(&path).map_err(|e| db_err(&path, e))?;

        let mut txn = db.begin_write().map_err(io_err)?;
        txn.set_durability(Durability::Immediate).map_err(io_err)?;

        // meta: format_version and seed. A fresh database takes the config
        // seed; an existing one keeps what it was created with.
        let effective_seed = {
            let mut meta = txn.open_table(META).map_err(io_err)?;
            let stored_version: Option<u32> = meta
                .get("format_version")
                .map_err(io_err)?
                .map(|guard| {
                    <[u8; 4]>::try_from(guard.value())
                        .map(u32::from_le_bytes)
                        .map_err(|_| meta_corrupt("format_version"))
                })
                .transpose()?;
            match stored_version {
                None => {
                    meta.insert("format_version", FORMAT_VERSION.to_le_bytes().as_slice())
                        .map_err(io_err)?;
                    meta.insert("seed", seed.to_le_bytes().as_slice())
                        .map_err(io_err)?;
                    seed
                }
                Some(FORMAT_VERSION) => {
                    let stored = meta
                        .get("seed")
                        .map_err(io_err)?
                        .map(|guard| {
                            <[u8; 8]>::try_from(guard.value())
                                .map(u64::from_le_bytes)
                                .map_err(|_| meta_corrupt("seed"))
                        })
                        .transpose()?
                        .unwrap_or(seed);
                    if stored != seed {
                        eprintln!(
                            "jbm: stored world seed {stored} in {:?} overrides the configured {seed}",
                            path
                        );
                    }
                    stored
                }
                Some(_) => return Err(StorageError::VersionTooNew),
            }
        };

        // block_states: load the stored name→id pairs, then map every runtime
        // state (appending the ones never saved before) in this transaction.
        // Opening `cubes` here creates the table on a fresh database, so a
        // read before the first save is not "table does not exist".
        let (runtime_to_disk, disk_to_runtime) = {
            let _ = txn.open_table(CUBES).map_err(io_err)?;
            let mut states = txn.open_table(BLOCK_STATES).map_err(io_err)?;
            let mut stored: HashMap<String, u32> = HashMap::new();
            let mut next_disk = 0u32;
            for row in states.iter().map_err(io_err)? {
                let (disk, name) = row.map_err(io_err)?;
                let disk = disk.value();
                next_disk = next_disk.max(disk + 1);
                stored.insert(name.value().to_string(), disk);
            }
            let mut runtime_names: Vec<(String, BlockStateId)> = (0..registry.state_count())
                .map(|id| {
                    let state = BlockStateId(id);
                    (canonical_state(registry.info(state)), state)
                })
                .collect();
            let known: HashMap<&str, BlockStateId> = runtime_names
                .iter()
                .map(|(name, state)| (name.as_str(), *state))
                .collect();
            for name in stored.keys() {
                if !known.contains_key(name.as_str()) {
                    return Err(StorageError::UnknownBlockState(name.clone()));
                }
            }
            let mut runtime_to_disk = vec![0u32; registry.state_count() as usize];
            let mut disk_to_runtime = HashMap::new();
            for (name, state) in &mut runtime_names {
                let disk = match stored.get(name.as_str()) {
                    Some(&disk) => disk,
                    None => {
                        let disk = next_disk;
                        next_disk += 1;
                        states.insert(disk, name.as_str()).map_err(io_err)?;
                        disk
                    }
                };
                runtime_to_disk[state.0 as usize] = disk;
                disk_to_runtime.insert(disk, *state);
            }
            (runtime_to_disk, disk_to_runtime)
        };
        txn.commit().map_err(io_err)?;

        Ok(Self {
            db,
            path,
            seed: effective_seed,
            runtime_to_disk,
            disk_to_runtime,
            hooks: Arc::new(Hooks::default()),
        })
    }

    /// The seed this world generates with: the stored one when reopening,
    /// otherwise the seed it was created with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Path of the underlying `world.redb`.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Loads the stored cube at `pos`, or `None` when never saved. A damaged
    /// record returns [`StorageError::Corrupt`] — the caller regenerates.
    pub fn load(&self, pos: CubePos) -> Result<Option<Cube>, StorageError> {
        let txn = self.db.begin_read().map_err(io_err)?;
        let table = txn.open_table(CUBES).map_err(io_err)?;
        let Some(record) = table.get(&cube_key(pos)).map_err(io_err)? else {
            return Ok(None);
        };
        self.decode_record(pos, record.value()).map(Some)
    }

    /// Writes every cube in `batch` in one transaction committed with
    /// immediate durability: when this returns, the batch is on disk. A
    /// crash mid-commit leaves the previous state, never a half-written
    /// batch.
    pub fn save_batch(&self, batch: &[(CubePos, &Cube)]) -> Result<(), StorageError> {
        self.save_batch_with_meta(batch, &[])
    }

    /// [`Storage::save_batch`] plus `meta` writes inside the same
    /// transaction. Test hook for the crash test, which stamps `last_batch`
    /// atomically with its cubes; the engine itself has no meta writes after
    /// open.
    pub fn save_batch_with_meta(
        &self,
        batch: &[(CubePos, &Cube)],
        meta: &[(&str, &[u8])],
    ) -> Result<(), StorageError> {
        self.hooks.enter_save()?;
        let mut txn = self.db.begin_write().map_err(io_err)?;
        txn.set_durability(Durability::Immediate).map_err(io_err)?;
        let mut plain = Vec::new();
        let mut record = Vec::new();
        {
            let mut cubes = txn.open_table(CUBES).map_err(io_err)?;
            for &(pos, cube) in batch {
                plain.clear();
                cube.encode_with(|state| self.runtime_to_disk[state.0 as usize], &mut plain);
                record.clear();
                record.push(RECORD_SCHEMA);
                // Bulk compression records the content size in the frame
                // header, so decoding can size its buffer exactly.
                let frame = zstd::bulk::compress(&plain, ZSTD_LEVEL).map_err(io_err)?;
                record.extend_from_slice(&frame);
                cubes
                    .insert(&cube_key(pos), record.as_slice())
                    .map_err(io_err)?;
            }
        }
        if !meta.is_empty() {
            let mut meta_table = txn.open_table(META).map_err(io_err)?;
            for (key, value) in meta {
                meta_table.insert(*key, *value).map_err(io_err)?;
            }
        }
        txn.commit().map_err(io_err)
    }

    /// How many cube records the database holds.
    pub fn cube_count(&self) -> Result<u64, StorageError> {
        let txn = self.db.begin_read().map_err(io_err)?;
        txn.open_table(CUBES).map_err(io_err)?.len().map_err(io_err)
    }

    /// Reads a `meta` value. Diagnostic/test hook (the crash test reads
    /// `last_batch`).
    pub fn meta(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        let txn = self.db.begin_read().map_err(io_err)?;
        Ok(txn
            .open_table(META)
            .map_err(io_err)?
            .get(key)
            .map_err(io_err)?
            .map(|guard| guard.value().to_vec()))
    }

    /// Every stored cube position. Diagnostic/test hook: the crash test
    /// loads every stored cube after a kill.
    pub fn cube_positions(&self) -> Result<Vec<CubePos>, StorageError> {
        let txn = self.db.begin_read().map_err(io_err)?;
        let table = txn.open_table(CUBES).map_err(io_err)?;
        let mut positions = Vec::new();
        for row in table.iter().map_err(io_err)? {
            let (key, _) = row.map_err(io_err)?;
            if let Some(pos) = cube_pos(key.value()) {
                positions.push(pos);
            }
        }
        Ok(positions)
    }

    /// Test hook: the next `save_batch` on any thread returns a synthetic
    /// failure instead of touching the database.
    pub fn fail_next_save(&self) {
        self.hooks.fail_next_save();
    }

    /// Test hook: every `save_batch` blocks until `release_saves`, making
    /// "an edit arrives while a batch is in flight" deterministic.
    pub fn hold_saves(&self) {
        self.hooks.hold();
    }

    /// Releases [`Storage::hold_saves`].
    pub fn release_saves(&self) {
        self.hooks.release();
    }

    /// Test hook: while saves are held, lets exactly one more `save_batch`
    /// through, so a test can commit batches one at a time.
    pub fn release_one_save(&self) {
        self.hooks.release_one();
    }

    /// Decodes one `cubes` record value (as produced by [`Self::encode_record`]).
    pub(crate) fn decode_record(&self, pos: CubePos, record: &[u8]) -> Result<Cube, StorageError> {
        let corrupt = |reason: String| StorageError::Corrupt { pos, reason };
        let Some((&schema, frame)) = record.split_first() else {
            return Err(corrupt("empty record".into()));
        };
        if schema != RECORD_SCHEMA {
            return Err(corrupt(format!("record schema {schema}")));
        }
        // Frames carry their content size (bulk compression), so one exact
        // bulk decompress replaces a streaming decoder and its window
        // buffers (~10x faster on a surface cube). A frame without a size
        // (written by a streaming encoder) still decodes, the slow way.
        let plain = match zstd::zstd_safe::get_frame_content_size(frame) {
            Ok(Some(size)) if size <= MAX_PLAIN_BYTES as u64 => {
                zstd::bulk::decompress(frame, size as usize)
                    .map_err(|e| corrupt(format!("zstd: {e}")))?
            }
            Ok(Some(size)) => return Err(corrupt(format!("zstd content size {size}"))),
            Ok(None) => decode_all(frame).map_err(|e| corrupt(format!("zstd: {e}")))?,
            Err(_) => return Err(corrupt("zstd: bad frame header".into())),
        };
        let disk_to_runtime = &self.disk_to_runtime;
        Cube::decode_with(&plain, |disk| disk_to_runtime.get(&disk).copied())
            .map_err(|e| corrupt(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{BLOCK_STATES, CUBES, Storage, StorageError, cube_key, cube_pos};
    use jbm_core::block::builtin::Blocks;
    use jbm_core::block::{BlockDef, BlockRegistry, Property};
    use jbm_core::coords::{CubePos, LocalPos};
    use jbm_core::cube::Cube;
    use std::time::Duration;

    fn registry() -> BlockRegistry {
        let mut registry = BlockRegistry::new();
        Blocks::register(&mut registry);
        registry
    }

    /// The same builtin blocks and states, registered in a different order —
    /// the remap test's registry B.
    fn reordered_registry() -> BlockRegistry {
        let mut registry = BlockRegistry::new();
        let simple = |registry: &mut BlockRegistry, name: &str| {
            registry
                .register(BlockDef {
                    name: format!("base:{name}"),
                    properties: Vec::new(),
                    solid: true,
                    opaque: true,
                })
                .unwrap();
        };
        for name in [
            "leaves",
            "sand",
            "oak_planks",
            "stone",
            "glass",
            "dirt",
            "grass_block",
        ] {
            simple(&mut registry, name);
        }
        // Same axis values, different order: the canonical strings match A's.
        registry
            .register(BlockDef {
                name: "base:oak_log".into(),
                properties: vec![Property {
                    name: "axis".into(),
                    values: ["z", "x", "y"].map(String::from).into(),
                }],
                solid: true,
                opaque: true,
            })
            .unwrap();
        simple(&mut registry, "water");
        registry
    }

    fn storage_in(dir: &tempfile::TempDir) -> Storage {
        Storage::open(dir.path(), &registry(), 42).unwrap()
    }

    fn pos(x: i32, y: i32, z: i32) -> CubePos {
        CubePos { x, y, z }
    }

    fn edited_cube() -> Cube {
        let mut fresh = BlockRegistry::new();
        let blocks = Blocks::register(&mut fresh);
        let mut cube = Cube::filled(blocks.stone);
        cube.set(LocalPos::new_unchecked(1, 2, 3), blocks.glass);
        cube
    }

    #[test]
    fn storage_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Storage>();
    }

    #[test]
    fn save_then_load_equals_the_original() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = storage_in(&dir);
        let cube = edited_cube();
        let batch = [(pos(3, -2, 7), &cube)];
        storage.save_batch(&batch).unwrap();
        assert_eq!(storage.load(pos(3, -2, 7)).unwrap(), Some(cube));
        assert_eq!(storage.cube_count().unwrap(), 1);
        assert_eq!(storage.cube_positions().unwrap(), vec![pos(3, -2, 7)]);
    }

    #[test]
    fn missing_cube_returns_none() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = storage_in(&dir);
        assert_eq!(storage.load(pos(0, 0, 0)).unwrap(), None);
        assert_eq!(storage.cube_count().unwrap(), 0);
    }

    #[test]
    fn reopen_keeps_cubes_and_seed() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let storage = storage_in(&dir);
            let batch = [(pos(-4, 5, -6), &edited_cube())];
            storage.save_batch(&batch).unwrap();
        }
        // A different config seed: the stored one wins.
        let reopened = Storage::open(dir.path(), &registry(), 999).unwrap();
        assert_eq!(reopened.seed(), 42);
        assert_eq!(reopened.load(pos(-4, 5, -6)).unwrap(), Some(edited_cube()));
        assert_eq!(reopened.cube_count().unwrap(), 1);
    }

    #[test]
    fn second_open_returns_locked() {
        let dir = tempfile::TempDir::new().unwrap();
        let _first = storage_in(&dir);
        let second = Storage::open(dir.path(), &registry(), 42).unwrap_err();
        assert!(matches!(second, StorageError::Locked(_)), "{second}");
    }

    #[test]
    fn key_ordering_holds_across_negative_coordinates() {
        // Encoding property: a key's bytes order exactly like its (x, y, z).
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut coords = Vec::new();
        for _ in 0..500 {
            let mut next = || {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as i32) % 50
            };
            coords.push(CubePos {
                x: next(),
                y: next(),
                z: next(),
            });
        }
        let mut keys: Vec<([u8; 13], CubePos)> =
            coords.into_iter().map(|p| (cube_key(p), p)).collect();
        keys.sort_by_key(|k| k.0);
        for pair in keys.windows(2) {
            assert_eq!(pair[0].0.cmp(&pair[1].0), pair[0].1.cmp(&pair[1].1));
        }
        // Negative, zero and positive coordinates round trip.
        for p in [
            pos(i32::MIN, 0, i32::MAX),
            pos(-1, -1, -1),
            pos(0, 0, 0),
            pos(1, 2, 3),
        ] {
            assert_eq!(cube_pos(&cube_key(p)), Some(p));
        }
    }

    #[test]
    fn remap_survives_registry_reordering() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut a = BlockRegistry::new();
        let blocks_a = Blocks::register(&mut a);
        let mut cube = Cube::filled(blocks_a.stone);
        cube.set(LocalPos::new_unchecked(0, 0, 0), blocks_a.oak_log);
        cube.set(
            LocalPos::new_unchecked(1, 0, 0),
            jbm_core::block::BlockStateId::AIR,
        );
        {
            let storage = Storage::open(dir.path(), &a, 42).unwrap();
            let batch = [(pos(0, 1, 0), &cube)];
            storage.save_batch(&batch).unwrap();
        }
        // Registry B registers the same blocks in a different order.
        let b = reordered_registry();
        let storage = Storage::open(dir.path(), &b, 42).unwrap();
        let loaded = storage.load(pos(0, 1, 0)).unwrap().unwrap();
        let mut expected = Cube::filled(b.state("base:stone", &[]).unwrap());
        expected.set(
            LocalPos::new_unchecked(0, 0, 0),
            b.state("base:oak_log", &[("axis", "y")]).unwrap(),
        );
        expected.set(
            LocalPos::new_unchecked(1, 0, 0),
            jbm_core::block::BlockStateId::AIR,
        );
        assert_eq!(loaded, expected);
        // Same block names as registry A assigned, cell for cell.
        let name_of = |index: usize| {
            let info = b.info(loaded.get_index(index));
            if info.properties.is_empty() {
                info.name.clone()
            } else {
                format!(
                    "{}[{}]",
                    info.name,
                    info.properties
                        .iter()
                        .map(|(k, v)| format!("{k}={v}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        };
        assert_eq!(name_of(0), "base:oak_log[axis=y]");
        assert_eq!(name_of(1), "base:air");
        assert_eq!(name_of(2), "base:stone");
    }

    #[test]
    fn unknown_stored_state_fails_open() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let storage = storage_in(&dir);
            // Sneak a state the next registry will not know into the table.
            let txn = storage.db.begin_write().unwrap();
            {
                let mut states = txn.open_table(BLOCK_STATES).unwrap();
                states.insert(9001u32, "modded:fancy_block").unwrap();
            }
            txn.commit().unwrap();
        }
        let error = Storage::open(dir.path(), &registry(), 42).unwrap_err();
        assert_eq!(
            error,
            StorageError::UnknownBlockState("modded:fancy_block".into())
        );
    }

    #[test]
    fn corrupt_record_returns_corrupt_from_load() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let storage = storage_in(&dir);
            let cube = edited_cube();
            let batch = [(pos(1, 1, 1), &cube)];
            storage.save_batch(&batch).unwrap();
            // Overwrite the value through a raw redb write.
            let txn = storage.db.begin_write().unwrap();
            {
                let mut cubes = txn.open_table(CUBES).unwrap();
                cubes
                    .insert(&cube_key(pos(1, 1, 1)), [9u8, 1, 2, 3].as_slice())
                    .unwrap();
            }
            txn.commit().unwrap();
        }
        let reopened = storage_in(&dir);
        let error = reopened.load(pos(1, 1, 1)).unwrap_err();
        assert!(
            matches!(&error, StorageError::Corrupt { pos: at, .. } if *at == pos(1, 1, 1)),
            "{error}"
        );
    }

    #[test]
    fn fail_next_save_hook_synthesizes_a_failure() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = storage_in(&dir);
        storage.fail_next_save();
        let cube = edited_cube();
        let batch = [(pos(0, 0, 0), &cube)];
        let error = storage.save_batch(&batch).unwrap_err();
        assert!(matches!(error, StorageError::Io(_)), "{error}");
        // The hook is one-shot: the retry goes through.
        storage.save_batch(&batch).unwrap();
        assert_eq!(storage.load(pos(0, 0, 0)).unwrap(), Some(cube));
    }

    #[test]
    fn hold_and_release_saves() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = std::sync::Arc::new(storage_in(&dir));
        storage.hold_saves();
        let cube = edited_cube();
        let moved = cube.clone();
        let saver = {
            let storage = std::sync::Arc::clone(&storage);
            std::thread::spawn(move || {
                let batch = [(pos(2, 2, 2), &moved)];
                storage.save_batch(&batch)
            })
        };
        std::thread::sleep(Duration::from_millis(50));
        assert!(!saver.is_finished(), "save must be held");
        storage.release_saves();
        saver.join().unwrap().unwrap();
        assert_eq!(storage.load(pos(2, 2, 2)).unwrap(), Some(cube));
    }
}
