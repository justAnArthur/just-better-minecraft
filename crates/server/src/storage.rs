//! World persistence: one directory holds one `redb` database per world
//! (task 2.3).
//!
//! Layout (see the crate README for the full contract):
//! - `meta`: `format_version`, `seed`, `lod_fingerprint` and test
//!   bookkeeping.
//! - `block_states`: on-disk state id → canonical state string. Ids are
//!   stable across restarts even when the runtime registry orders states
//!   differently, so cubes decode to the same blocks.
//! - `cubes`: 13-byte position key → `[record schema][zstd(codec bytes)]`.
//!   Key kind 0 is a LOD0 cube (schema 1); kind *n* ≥ 1 is a LOD
//!   level-*n* cube (schema 3: cells plus column tops, task 2.14).
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
use std::sync::atomic::{AtomicBool, Ordering};

use jbm_core::block::{BlockRegistry, BlockStateId, StateInfo};
use jbm_core::coords::{CUBE_VOLUME, CubePos};
use jbm_core::cube::Cube;
use jbm_core::lod::{LodTops, RULE_VERSION};
use redb::{
    Database, Durability, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition,
};
use zstd::stream::decode_all;

/// One LOD record to save: `(level, pos, cube, column tops, real-octant
/// mask)`.
pub type LodRecord<'a> = (u8, CubePos, &'a Cube, &'a LodTops, u8);

/// `meta` table: small named blobs (`format_version`, `seed`, …).
const META: TableDefinition<'_, &str, &[u8]> = TableDefinition::new("meta");
/// On-disk block state id → canonical `namespace:path[prop=value,…]` string.
const BLOCK_STATES: TableDefinition<'_, u32, &str> = TableDefinition::new("block_states");
/// Player name → player record ([`PlayerRecord`]).
const PLAYERS: TableDefinition<'_, &str, &[u8]> = TableDefinition::new("players");
/// Position key (below) → cube record.
const CUBES: TableDefinition<'_, &[u8; 13], &[u8]> = TableDefinition::new("cubes");

/// Current on-disk format version; a database written by a newer format is
/// rejected with [`StorageError::VersionTooNew`] instead of misread. Stays
/// 1 with the LOD records of task 2.7: databases written before 2.7 hold no
/// key with kind ≥ 1, so they open unchanged.
const FORMAT_VERSION: u32 = 1;

/// Cube record schema byte in front of every LOD0 value in `cubes`.
const RECORD_SCHEMA: u8 = 1;

/// Record schema byte in front of every LOD value in `cubes`:
/// `[3][known: u8][tops][zstd frame of the cube codec bytes]`, where
/// `known` is the LOD cube's real-octant bitmask and `tops` its column
/// tops in the self-delimiting [`LodTops::encode`] form (task 2.14; schema
/// 2 had no tops). A record that fails to decode is logged by the caller
/// and treated as absent.
const LOD_RECORD_SCHEMA: u8 = 3;

/// `meta` key holding the worldgen fingerprint stored LOD was written
/// under: `hash(compiled graph fingerprint, seed, lod::RULE_VERSION)` as
/// u64 LE. A mismatch means a different worldgen — or a different
/// downsampling rule — produced the stored LOD, so it is ignored and
/// overwritten; LOD0 records are never deleted.
const LOD_FINGERPRINT_KEY: &str = "lod_fingerprint";

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
    /// A player record (named here) could not be decoded.
    CorruptPlayer(String),
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
            Self::CorruptPlayer(name) => write!(f, "corrupt player record {name:?}"),
            Self::UnknownBlockState(name) => write!(f, "unknown block state {name:?}"),
            Self::VersionTooNew => write!(f, "world database format is newer than this build"),
        }
    }
}

impl std::error::Error for StorageError {}

/// What the world remembers about a player between sessions: where they
/// stood, where they looked and whether they were flying. Velocity is not
/// kept; a returning player starts at rest, not on ground.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerRecord {
    /// Feet position.
    pub position: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub flying: bool,
}

/// Player record schema: `[schema u8 = 1][x y z: f64][yaw pitch: f32][flying u8]`,
/// little-endian, 34 bytes.
const PLAYER_SCHEMA: u8 = 1;
const PLAYER_RECORD_LEN: usize = 1 + 3 * 8 + 2 * 4 + 1;

impl PlayerRecord {
    fn encode(&self) -> [u8; PLAYER_RECORD_LEN] {
        let mut out = [0u8; PLAYER_RECORD_LEN];
        out[0] = PLAYER_SCHEMA;
        for (i, v) in self.position.iter().enumerate() {
            out[1 + i * 8..9 + i * 8].copy_from_slice(&v.to_le_bytes());
        }
        out[25..29].copy_from_slice(&self.yaw.to_le_bytes());
        out[29..33].copy_from_slice(&self.pitch.to_le_bytes());
        out[33] = u8::from(self.flying);
        out
    }

    /// `None` for anything but a well-formed record with finite values.
    fn decode(bytes: &[u8]) -> Option<PlayerRecord> {
        if bytes.len() != PLAYER_RECORD_LEN || bytes[0] != PLAYER_SCHEMA || bytes[33] > 1 {
            return None;
        }
        let f64_at = |i: usize| f64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let f32_at = |i: usize| f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let record = PlayerRecord {
            position: [f64_at(1), f64_at(9), f64_at(17)],
            yaw: f32_at(25),
            pitch: f32_at(29),
            flying: bytes[33] == 1,
        };
        let finite = record.position.iter().all(|v| v.is_finite())
            && record.yaw.is_finite()
            && record.pitch.is_finite();
        finite.then_some(record)
    }
}

fn meta_corrupt(key: &'static str) -> StorageError {
    StorageError::CorruptMeta(key)
}

/// The stored-LOD worldgen fingerprint: the compiled graph's fingerprint
/// mixed with the world seed (the same graph under a different seed is
/// different worldgen) and the LOD downsampling rule version (stored LOD
/// built under another rule is stale, task 2.14). `DefaultHasher` has
/// fixed keys, so the mix is stable across runs.
fn lod_fingerprint(graph_fingerprint: u64, seed: u64, rule_version: u32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    graph_fingerprint.hash(&mut hasher);
    seed.hash(&mut hasher);
    rule_version.hash(&mut hasher);
    hasher.finish()
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
/// a LOD0 cube; kind *n* ≥ 1 is a LOD level-*n* cube (task 2.7).
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

/// The LOD twin of [`cube_key`]: kind = `level` (the scheme 2.3 reserved).
fn lod_key(level: u8, pos: CubePos) -> [u8; 13] {
    let mut key = cube_key(pos);
    key[0] = level;
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
    /// Whether stored LOD records match the current worldgen fingerprint
    /// (absent until [`Storage::set_worldgen_fingerprint`] says otherwise).
    /// Interior-mutability: load paths take `&self`.
    lod_usable: AtomicBool,
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
            lod_usable: AtomicBool::new(false),
            hooks: Arc::new(Hooks::default()),
        })
    }

    /// Declares the compiled worldgen graph the server runs, after open
    /// (the stored seed is part of the fingerprint, and only open knows
    /// it). The stored fingerprint in `meta` is compared with
    /// `hash(graph_fingerprint, seed, RULE_VERSION)`: on a match, stored LOD records are
    /// usable; on a mismatch (different worldgen or rule wrote them) they are
    /// deleted in the same transaction that overwrites the stored
    /// fingerprint, so no stale record is ever read — not in this run, and
    /// not after a restart. LOD0 records are never deleted either way.
    /// Databases written before 2.7 have no fingerprint: it is simply
    /// written. The
    /// LOD rule version ([`RULE_VERSION`]) is part of the fingerprint, so
    /// LOD stored under an older downsampling rule is resampled.
    pub fn set_worldgen_fingerprint(&self, graph_fingerprint: u64) -> Result<(), StorageError> {
        self.set_lod_fingerprint(lod_fingerprint(graph_fingerprint, self.seed, RULE_VERSION))
    }

    fn set_lod_fingerprint(&self, expected: u64) -> Result<(), StorageError> {
        let stored = self.meta(LOD_FINGERPRINT_KEY)?;
        match stored.as_deref() {
            Some(bytes)
                if bytes.len() == 8
                    && u64::from_le_bytes(bytes.try_into().unwrap()) == expected =>
            {
                self.lod_usable.store(true, Ordering::Release);
            }
            _ => {
                let mut txn = self.db.begin_write().map_err(io_err)?;
                txn.set_durability(Durability::Immediate).map_err(io_err)?;
                txn.open_table(META)
                    .map_err(io_err)?
                    .insert(LOD_FINGERPRINT_KEY, expected.to_le_bytes().as_slice())
                    .map_err(io_err)?;
                if stored.is_some() {
                    // Records written under a different fingerprint are
                    // stale: drop them with the fingerprint change.
                    txn.open_table(CUBES)
                        .map_err(io_err)?
                        .retain(|key, _| key[0] == 0)
                        .map_err(io_err)?;
                }
                txn.commit().map_err(io_err)?;
                self.lod_usable.store(true, Ordering::Release);
            }
        }
        Ok(())
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

    /// Loads the stored LOD cube at `(level, pos)` with its column tops and
    /// real-octant mask, or `None` when never saved — or when the worldgen
    /// fingerprint does not match (stored LOD from a different worldgen or
    /// LOD rule is ignored). A damaged record returns
    /// [`StorageError::Corrupt`]; the caller logs it and samples the cube
    /// from worldgen instead. Never panics.
    pub fn load_lod(
        &self,
        level: u8,
        pos: CubePos,
    ) -> Result<Option<(Cube, LodTops, u8)>, StorageError> {
        if !self.lod_usable.load(Ordering::Acquire) {
            return Ok(None);
        }
        let txn = self.db.begin_read().map_err(io_err)?;
        let table = txn.open_table(CUBES).map_err(io_err)?;
        let Some(record) = table.get(&lod_key(level, pos)).map_err(io_err)? else {
            return Ok(None);
        };
        let bytes = record.value();
        let corrupt = |reason: String| StorageError::Corrupt { pos, reason };
        let Some((&schema, rest)) = bytes.split_first() else {
            return Err(corrupt("empty record".into()));
        };
        if schema != LOD_RECORD_SCHEMA {
            return Err(corrupt(format!("record schema {schema}")));
        }
        let Some((&known, rest)) = rest.split_first() else {
            return Err(corrupt("missing known mask".into()));
        };
        let (tops, used) = LodTops::decode(rest).map_err(|e| corrupt(format!("tops: {e}")))?;
        if tops.level() != level {
            return Err(corrupt(format!(
                "tops level {} in a level {level} record",
                tops.level()
            )));
        }
        let cube = self.decode_frame(&rest[used..]).map_err(corrupt)?;
        Ok(Some((cube, tops, known)))
    }

    /// Writes every cube in `batch` in one transaction committed with
    /// immediate durability: when this returns, the batch is on disk. A
    /// crash mid-commit leaves the previous state, never a half-written
    /// batch.
    pub fn save_batch(&self, batch: &[(CubePos, &Cube)]) -> Result<(), StorageError> {
        self.save_batch_with_meta(batch, &[])
    }

    /// LOD0 cubes, LOD cubes and player records in one transaction: an
    /// edit's LOD0 record and the LOD records derived from it commit
    /// together or not at all, so a crash never leaves stored LOD
    /// disagreeing with stored LOD0 edits (task 2.7).
    pub fn save_lod_batch(
        &self,
        lod0: &[(CubePos, &Cube)],
        lod: &[LodRecord<'_>],
        players: &[(String, PlayerRecord)],
    ) -> Result<(), StorageError> {
        self.save_batch_full(lod0, lod, players, &[])
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
        self.save_batch_full(batch, &[], &[], meta)
    }

    /// [`Storage::save_batch`] plus player records, all in the one
    /// transaction: a batch's cubes and the players that saw them commit
    /// together or not at all.
    pub fn save_batch_with_players(
        &self,
        batch: &[(CubePos, &Cube)],
        players: &[(String, PlayerRecord)],
    ) -> Result<(), StorageError> {
        self.save_batch_full(batch, &[], players, &[])
    }

    /// The player record stored under `name`, if any.
    pub fn load_player(&self, name: &str) -> Result<Option<PlayerRecord>, StorageError> {
        let txn = self.db.begin_read().map_err(io_err)?;
        let table = match txn.open_table(PLAYERS) {
            Ok(table) => table,
            // A world saved before any player record has no table yet.
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(error) => return Err(io_err(error)),
        };
        let Some(record) = table.get(name).map_err(io_err)? else {
            return Ok(None);
        };
        PlayerRecord::decode(record.value())
            .map(Some)
            .ok_or_else(|| StorageError::CorruptPlayer(name.to_owned()))
    }

    fn save_batch_full(
        &self,
        batch: &[(CubePos, &Cube)],
        lod: &[LodRecord<'_>],
        players: &[(String, PlayerRecord)],
        meta: &[(&str, &[u8])],
    ) -> Result<(), StorageError> {
        self.hooks.enter_save()?;
        let mut txn = self.db.begin_write().map_err(io_err)?;
        txn.set_durability(Durability::Immediate).map_err(io_err)?;
        let mut plain = Vec::new();
        let mut record = Vec::new();
        {
            let mut cubes = txn.open_table(CUBES).map_err(io_err)?;
            let encode = |cube: &Cube, plain: &mut Vec<u8>| {
                plain.clear();
                cube.encode_with(|state| self.runtime_to_disk[state.0 as usize], plain);
            };
            for &(pos, cube) in batch {
                encode(cube, &mut plain);
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
            for &(level, pos, cube, tops, known) in lod {
                encode(cube, &mut plain);
                record.clear();
                record.push(LOD_RECORD_SCHEMA);
                record.push(known);
                tops.encode(&mut record);
                let frame = zstd::bulk::compress(&plain, ZSTD_LEVEL).map_err(io_err)?;
                record.extend_from_slice(&frame);
                cubes
                    .insert(&lod_key(level, pos), record.as_slice())
                    .map_err(io_err)?;
            }
        }
        if !players.is_empty() {
            let mut table = txn.open_table(PLAYERS).map_err(io_err)?;
            for (name, record) in players {
                table
                    .insert(name.as_str(), record.encode().as_slice())
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
        self.decode_frame(frame).map_err(corrupt)
    }

    /// The zstd + codec half both record schemas share.
    fn decode_frame(&self, frame: &[u8]) -> Result<Cube, String> {
        // Frames carry their content size (bulk compression), so one exact
        // bulk decompress replaces a streaming decoder and its window
        // buffers (~10x faster on a surface cube). A frame without a size
        // (written by a streaming encoder) still decodes, the slow way.
        let plain = match zstd::zstd_safe::get_frame_content_size(frame) {
            Ok(Some(size)) if size <= MAX_PLAIN_BYTES as u64 => {
                zstd::bulk::decompress(frame, size as usize).map_err(|e| e.to_string())?
            }
            Ok(Some(size)) => return Err(format!("zstd content size {size}")),
            Ok(None) => decode_all(frame).map_err(|e| e.to_string())?,
            Err(_) => return Err("zstd: bad frame header".into()),
        };
        let disk_to_runtime = &self.disk_to_runtime;
        Cube::decode_with(&plain, |disk| disk_to_runtime.get(&disk).copied())
            .map_err(|e| e.to_string())
    }

    /// How many LOD records (key kind ≥ 1) the database holds.
    /// Diagnostic: the stored-LOD size report.
    pub fn lod_record_count(&self) -> Result<u64, StorageError> {
        Ok(self.lod_positions()?.len() as u64)
    }

    /// Every stored LOD key as `(level, position)`. Diagnostic/test hook:
    /// the crash test verifies every stored LOD record after a kill.
    pub fn lod_positions(&self) -> Result<Vec<(u8, CubePos)>, StorageError> {
        let txn = self.db.begin_read().map_err(io_err)?;
        let table = txn.open_table(CUBES).map_err(io_err)?;
        let mut out = Vec::new();
        for row in table.iter().map_err(io_err)? {
            let (key, _) = row.map_err(io_err)?;
            let key = key.value();
            if key[0] >= 1 {
                let field = |bytes: &[u8]| {
                    let mut raw = [0u8; 4];
                    raw.copy_from_slice(bytes);
                    raw[0] ^= 0x80;
                    i32::from_be_bytes(raw)
                };
                out.push((
                    key[0],
                    CubePos {
                        x: field(&key[1..5]),
                        y: field(&key[5..9]),
                        z: field(&key[9..13]),
                    },
                ));
            }
        }
        Ok(out)
    }

    /// Test hook: overwrites the LOD record at `(level, pos)` with
    /// undecodable garbage, as a torn write would leave behind.
    pub fn damage_lod_record(&self, level: u8, pos: CubePos) -> Result<(), StorageError> {
        let mut txn = self.db.begin_write().map_err(io_err)?;
        txn.set_durability(Durability::Immediate).map_err(io_err)?;
        txn.open_table(CUBES)
            .map_err(io_err)?
            .insert(&lod_key(level, pos), [9u8, 9, 9, 9].as_slice())
            .map_err(io_err)?;
        txn.commit().map_err(io_err)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_STATES, CUBES, LodTops, RULE_VERSION, Storage, StorageError, cube_key, cube_pos,
        lod_fingerprint,
    };
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
    fn player_records_round_trip_across_a_reopen() {
        use super::PlayerRecord;
        let dir = tempfile::TempDir::new().unwrap();
        let alice = PlayerRecord {
            position: [-1234.25, 77.0, 9.5e6],
            yaw: 1.25,
            pitch: -0.5,
            flying: true,
        };
        {
            let storage = storage_in(&dir);
            assert_eq!(storage.load_player("alice").unwrap(), None, "no table yet");
            storage
                .save_batch_with_players(&[], &[("alice".to_owned(), alice)])
                .unwrap();
            assert_eq!(storage.load_player("bob").unwrap(), None);
        }
        let reopened = storage_in(&dir);
        assert_eq!(reopened.load_player("alice").unwrap(), Some(alice));
    }

    #[test]
    fn a_corrupt_player_record_is_a_typed_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = storage_in(&dir);
        {
            let txn = storage.db.begin_write().unwrap();
            {
                let mut table = txn.open_table(super::PLAYERS).unwrap();
                table.insert("alice", [1u8, 2, 3].as_slice()).unwrap();
                let mut nan = super::PlayerRecord {
                    position: [0.0; 3],
                    yaw: 0.0,
                    pitch: 0.0,
                    flying: false,
                }
                .encode();
                nan[1..9].copy_from_slice(&f64::NAN.to_le_bytes());
                table.insert("bob", nan.as_slice()).unwrap();
            }
            txn.commit().unwrap();
        }
        for name in ["alice", "bob"] {
            assert_eq!(
                storage.load_player(name),
                Err(StorageError::CorruptPlayer(name.to_owned()))
            );
        }
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

    // ------------------------------------------------------------------
    // Stored LOD (task 2.7)

    fn lod_record(level: u8) -> (Cube, LodTops, u8) {
        let mut cube = edited_cube();
        cube.set(
            LocalPos::new_unchecked(9, 9, 9),
            jbm_core::block::BlockStateId(3),
        );
        // Mixed column tops, so the record carries the full table.
        let mut tops = LodTops::full(level);
        tops.set_half(0, 9, 9, 1);
        tops.set_half(1, 3, 30, 0);
        tops.set_half(1, 31, 0, 1);
        // A realistic mask: three real octants, five filled.
        (cube, tops, 0b0001_0110)
    }

    #[test]
    fn lod_records_round_trip_in_the_same_transaction_as_lod0() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = storage_in(&dir);
        storage
            .set_worldgen_fingerprint(0xABCD_EF01_2345_6789)
            .unwrap();
        let (cube, tops, known) = lod_record(2);
        let lod0 = edited_cube();
        let lod_batch = [(2u8, pos(1, 2, 3), &cube, &tops, known)];
        storage
            .save_lod_batch(&[(pos(0, 0, 0), &lod0)], &lod_batch, &[])
            .unwrap();
        let (loaded, loaded_tops, loaded_known) =
            storage.load_lod(2, pos(1, 2, 3)).unwrap().unwrap();
        assert_eq!(loaded, cube);
        assert_eq!(loaded_tops, tops);
        assert_eq!(loaded_known, known);
        // The counts see both kinds.
        assert_eq!(storage.cube_count().unwrap(), 2);
        assert_eq!(storage.lod_record_count().unwrap(), 1);
        assert_eq!(storage.lod_positions().unwrap(), vec![(2, pos(1, 2, 3))]);
        // LOD0 keys are untouched by LOD records at the same coordinates.
        let lod0_at_same_pos = edited_cube();
        storage
            .save_lod_batch(&[(pos(1, 2, 3), &lod0_at_same_pos)], &[], &[])
            .unwrap();
        assert_eq!(storage.load(pos(1, 2, 3)).unwrap(), Some(lod0_at_same_pos));
        let (loaded, ..) = storage.load_lod(2, pos(1, 2, 3)).unwrap().unwrap();
        assert_eq!(loaded, cube, "the LOD record survived");
    }

    #[test]
    fn lod_loads_are_absent_until_a_fingerprint_matches() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = storage_in(&dir);
        // No fingerprint yet: nothing is readable (a fresh 2.7 database).
        assert_eq!(storage.load_lod(1, pos(0, 0, 0)).unwrap(), None);
        let (cube, tops, known) = lod_record(1);
        // Without a matching fingerprint, saving is still allowed but the
        // record stays unreadable — as if absent.
        storage
            .save_lod_batch(&[], &[(1, pos(4, 5, 6), &cube, &tops, known)], &[])
            .unwrap();
        assert_eq!(
            storage.load_lod(1, pos(4, 5, 6)).unwrap(),
            None,
            "no fingerprint declared: records read as absent"
        );
        storage.set_worldgen_fingerprint(42).unwrap();
        let loaded = storage.load_lod(1, pos(4, 5, 6)).unwrap().unwrap();
        assert_eq!(loaded, (cube, tops, known));
        // A different worldgen fingerprint drops the stored records; LOD0
        // records stay.
        storage.set_worldgen_fingerprint(43).unwrap();
        assert_eq!(
            storage.load_lod(1, pos(4, 5, 6)).unwrap(),
            None,
            "stale fingerprint: records read as absent"
        );
    }

    #[test]
    fn a_corrupt_lod_record_is_a_typed_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = storage_in(&dir);
        storage.set_worldgen_fingerprint(7).unwrap();
        let (cube, tops, known) = lod_record(3);
        storage
            .save_lod_batch(&[], &[(3, pos(7, 7, 7), &cube, &tops, known)], &[])
            .unwrap();
        storage.damage_lod_record(3, pos(7, 7, 7)).unwrap();
        let error = storage.load_lod(3, pos(7, 7, 7)).unwrap_err();
        assert!(
            matches!(&error, StorageError::Corrupt { pos: at, .. } if *at == pos(7, 7, 7)),
            "{error}"
        );
        // A LOD0 record at the same position is unaffected.
        assert_eq!(storage.load(pos(7, 7, 7)).unwrap(), None);
    }

    #[test]
    fn lod_stored_under_another_rule_version_resamples() {
        let dir = tempfile::TempDir::new().unwrap();
        let graph = 0x005E_ED0F_6A4F;
        let (cube, tops, known) = lod_record(2);
        {
            // A world whose LOD was written under the previous downsampling
            // rule (same graph, same seed).
            let storage = storage_in(&dir);
            let old = lod_fingerprint(graph, storage.seed(), RULE_VERSION - 1);
            storage.set_lod_fingerprint(old).unwrap();
            storage
                .save_lod_batch(&[], &[(2, pos(1, 1, 1), &cube, &tops, known)], &[])
                .unwrap();
            assert!(storage.load_lod(2, pos(1, 1, 1)).unwrap().is_some());
        }
        // Reopened by this build: the rule version differs, so the stored
        // LOD is gone (the pyramid resamples it from worldgen) …
        let storage = storage_in(&dir);
        storage.set_worldgen_fingerprint(graph).unwrap();
        assert_eq!(storage.load_lod(2, pos(1, 1, 1)).unwrap(), None);
        assert_eq!(storage.lod_record_count().unwrap(), 0);
        // … and what this run saves is readable again.
        storage
            .save_lod_batch(&[], &[(2, pos(1, 1, 1), &cube, &tops, known)], &[])
            .unwrap();
        assert!(storage.load_lod(2, pos(1, 1, 1)).unwrap().is_some());
        drop(storage);
        let storage = storage_in(&dir);
        storage.set_worldgen_fingerprint(graph).unwrap();
        assert!(storage.load_lod(2, pos(1, 1, 1)).unwrap().is_some());
    }
}
