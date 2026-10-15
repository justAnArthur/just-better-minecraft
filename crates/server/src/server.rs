//! The authoritative server: connection handling and the fixed-order tick
//! phases from the design doc's "Simulation" section. Clients are talked to
//! only through [`ServerLink`]s.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use glam::DVec3;
use jbm_core::block::BlockStateId;
use jbm_core::coords::{BlockPos, CubePos};
use jbm_core::link::{LinkClosed, ServerLink};
use jbm_core::physics::{self, BlockShapes, MoveInput, PlayerState, SolidTable};
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, PlayerId, ServerMsg};
use jbm_core::raycast::REACH;

use crate::gen_pool::GenPool;
use crate::lod::LodPyramid;
use crate::storage::{PlayerRecord, Storage, StorageError};
use crate::streaming::{Offset, desired_offsets, in_keep, offset_pos};
use crate::terrain::Generator;
use crate::tick::{TICK_PERIOD, TICKS_PER_SECOND, TickClock};

/// Most inbound messages one connection may deliver in a single tick.
/// Leftovers wait for the next tick, so a flooding client cannot stall the
/// tick.
const MESSAGES_PER_CONNECTION_PER_TICK: usize = 256;

/// Queued [`MoveInput`]s per player; on overflow the oldest is dropped.
const INPUT_QUEUE_CAP: usize = 20;

/// Inputs stepped per player per tick in phase 4.
const INPUTS_PER_TICK: usize = 4;

/// Slack on the reach check, in blocks, so a player whose client-side ray
/// was within `REACH` when they clicked still passes validation a round
/// trip later.
const REACH_MARGIN: f64 = 1.0;

/// One queued outgoing block edit. `None` broadcasts to every player with
/// the cube loaded; `Some(id)` sends to that player only (a rejection
/// revert).
type QueuedChange = (BlockPos, BlockStateId, Option<PlayerId>);

/// The seed every default server generates its world with. Fixed so a
/// fresh install always starts on the same terrain.
pub const DEFAULT_SEED: u64 = 0x0055_EED1_5EED;

/// How long [`Server::flush`] waits for the save thread before giving up
/// and logging loudly. Generous: a flush is a commit-to-disk, fsync
/// included.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(30);

/// One snapshot batch handed to the save thread: cube clones with the edit
/// version they were taken at, LOD cube clones, and player records —
/// committed together, so LOD derived from an edit never lands on disk
/// without the LOD0 edit it came from.
#[derive(Default)]
struct SaveBatch {
    cubes: Vec<(CubePos, jbm_core::cube::Cube, u64)>,
    lod: Vec<(u8, CubePos, jbm_core::cube::Cube, u8)>,
    players: Vec<(String, PlayerRecord)>,
}

impl SaveBatch {
    fn is_empty(&self) -> bool {
        self.cubes.is_empty() && self.lod.is_empty() && self.players.is_empty()
    }
}

/// The save thread's reply to a batch.
enum SaveAck {
    /// The batch committed; these cube versions, LOD cubes and player
    /// records are now on disk.
    Committed(
        Vec<(CubePos, u64)>,
        Vec<(u8, CubePos)>,
        Vec<(String, PlayerRecord)>,
    ),
    /// The batch failed; these cubes stay dirty and retry next interval.
    Failed(Vec<CubePos>, Vec<(u8, CubePos)>),
}

/// Per-cube edit bookkeeping (persistence only): the edit version, bumped by
/// every accepted edit, and the last version known committed to disk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EditVersions {
    version: u64,
    saved_version: u64,
}

/// Tuning knobs for a [`Server`]. The crate README documents what each field
/// costs; the defaults fit one player streaming a view radius of 8 cubes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    /// World seed fed to the terrain generator.
    pub seed: u64,
    /// Whether cubes are streamed to players at all.
    pub streaming: bool,
    /// Horizontal streaming radius, in cubes (`dx² + dz² ≤ r²`).
    pub view_radius: u32,
    /// Vertical streaming radius, in cubes (`|dy| ≤ v`).
    pub vertical_radius: u32,
    /// Most `CubeData` messages sent to one player per tick.
    pub send_budget_per_tick: usize,
    /// Generation worker threads. `0` selects inline mode: the tick thread
    /// itself generates up to `gen_budget_per_tick` cubes per tick, which is
    /// deterministic — what tests and benches use.
    pub gen_threads: usize,
    /// Most cubes generated inline per tick (inline mode only).
    pub gen_budget_per_tick: usize,
    /// Cap on cubes queued, generating, or generated but not yet collected.
    pub max_in_flight: usize,
    /// World directory for persistence. `None` (the default) disables
    /// persistence: edited cubes stay pinned in memory forever, exactly as
    /// before task 2.3.
    pub world_dir: Option<PathBuf>,
    /// How often dirty cubes are snapshotted to the save thread.
    pub save_interval: Duration,
    /// Top LOD level the server keeps (4 keeps LOD1–LOD4). `0` disables
    /// the pyramid entirely: no allocation, no work per cube or edit.
    pub lod_levels: u8,
    /// Horizontal radius of every LOD ring, in cubes of that ring's level
    /// (task 2.6). Each level streams a full disk; the client draws the
    /// finest ready level per octant.
    pub lod_view_radius: u32,
    /// Vertical radius of every LOD ring, in cubes of that ring's level.
    pub lod_vertical_radius: u32,
    /// Most `LodCubeData` messages sent to one player per tick, re-sends
    /// of changed cubes included.
    pub lod_send_budget_per_tick: usize,
    /// Cap on LOD cubes requested from the pyramid but not yet ready
    /// ([`LodPyramid::in_flight`]).
    pub lod_max_in_flight: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        let parallelism = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self {
            seed: DEFAULT_SEED,
            streaming: true,
            view_radius: 8,
            vertical_radius: 4,
            send_budget_per_tick: 64,
            gen_threads: parallelism.saturating_sub(2).max(1),
            gen_budget_per_tick: 64,
            max_in_flight: 512,
            world_dir: None,
            save_interval: Duration::from_secs(5),
            lod_levels: 4,
            lod_view_radius: 8,
            lod_vertical_radius: 3,
            lod_send_budget_per_tick: 32,
            lod_max_in_flight: 256,
        }
    }
}

impl ServerConfig {
    /// Streaming off and no generation threads: a server that only speaks
    /// the handshake/ping/tick protocol. Tests and benches that assert
    /// exact message sequences use this so their meaning stays unchanged.
    /// The LOD pyramid is off too.
    pub fn protocol_only() -> Self {
        Self {
            streaming: false,
            gen_threads: 0,
            lod_levels: 0,
            ..Self::default()
        }
    }
}

/// The cube containing a world position given in block coordinates.
fn cube_containing(position: DVec3) -> CubePos {
    BlockPos {
        x: position.x.floor() as i64,
        y: position.y.floor() as i64,
        z: position.z.floor() as i64,
    }
    .split()
    .0
}

/// Whether the center of the block at `pos` lies within
/// `REACH + REACH_MARGIN` of the player's `eye`. The margin covers the
/// round trip between the client's ray and this validation.
fn within_reach(pos: BlockPos, eye: DVec3) -> bool {
    let dx = pos.x as f64 + 0.5 - eye.x;
    let dy = pos.y as f64 + 0.5 - eye.y;
    let dz = pos.z as f64 + 0.5 - eye.z;
    let limit = REACH + REACH_MARGIN;
    dx * dx + dy * dy + dz * dz <= limit * limit
}

/// Read-only view of a joined player, for tests and diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerView {
    pub id: PlayerId,
    pub name: String,
    /// The authoritative physics state.
    pub state: PlayerState,
    /// Seq of the last processed input (0 before any).
    pub last_seq: u64,
    /// `PlayerInput` messages received in total.
    pub input_count: u64,
}

/// State the server tracks for one joined player.
struct Player {
    id: PlayerId,
    name: String,
    /// The authoritative state; stepped in phase 4 from the spawn state.
    state: PlayerState,
    /// Received, not-yet-processed inputs in arrival order. Capped at
    /// [`INPUT_QUEUE_CAP`]; overflow drops the oldest.
    input_queue: VecDeque<(u64, MoveInput)>,
    /// Seq of the last processed input; anything ≤ this is ignored on
    /// arrival.
    last_seq: u64,
    input_count: u64,
    /// Set in phase 4 when at least one input was stepped; makes phase 5
    /// send a `PlayerState` before the `Tick`.
    pending_ack: bool,
    /// The cube the player is centered on (their state position). Drives
    /// the desired and keep sets.
    center: CubePos,
    /// Cubes this player has been sent and not yet had unloaded. A cube is
    /// never sent twice while it stays in this set.
    sent: HashSet<CubePos>,
    /// Index into the precomputed nearest-first desired offsets: the next
    /// cube that may still need sending. Reset whenever the center moves,
    /// so an idle player costs O(1) per tick.
    cursor: usize,
    /// Look direction from the newest accepted input (radians); persisted
    /// with the player and handed back in the next `Welcome`.
    yaw: f32,
    pitch: f32,
}

impl Player {
    fn record(&self) -> PlayerRecord {
        PlayerRecord {
            position: self.state.position.to_array(),
            yaw: self.yaw,
            pitch: self.pitch,
            flying: self.state.flying,
        }
    }
}

enum ConnectionState {
    /// Connected, but no valid `Hello` has arrived yet.
    Pending,
    /// Welcomed and playing. Boxed: the player state outgrew the pending
    /// variant several times over.
    Joined(Box<Player>),
}

struct Connection {
    link: ServerLink,
    state: ConnectionState,
}

impl Connection {
    fn joined_player(&self) -> Option<&Player> {
        match &self.state {
            ConnectionState::Pending => None,
            ConnectionState::Joined(player) => Some(player),
        }
    }
}

/// What to do with a connection after handling one of its messages.
enum Outcome {
    Keep,
    Drop,
}

/// The authoritative simulation. Add connections with
/// [`Server::add_connection`] and drive time with [`Server::update`] (or
/// run single ticks with [`Server::tick`] in tests). Not `Debug`: it holds
/// boxed link trait objects.
pub struct Server {
    clock: TickClock,
    /// Ticks run so far; the number sent in `Tick` and `Pong` messages.
    tick_number: u64,
    /// In join order; drives message-drain order.
    connections: Vec<Connection>,
    /// Next id to hand out. Ids start at 1 and count up.
    next_player_id: u32,
    config: ServerConfig,
    pool: GenPool,
    /// The generator the pool and the LOD pipeline sample from; kept for
    /// the test counters (`Server::lod_samples`).
    generator: Arc<Generator>,
    /// Generated cubes the server keeps in memory, keyed by position.
    world: HashMap<CubePos, jbm_core::cube::Cube>,
    /// Per-state solidity for physics; built once from the same registry
    /// the generator uses, so both sides of a collision agree.
    solid: SolidTable,
    /// Total registered states; ids ≥ this are unknown and rejected by
    /// place validation.
    state_count: u32,
    /// Cubes that hold an accepted edit. Used only without persistence;
    /// with a world directory the pin rule is "version > saved_version" on
    /// [`Server::versions`] instead, and this set stays empty.
    pinned: HashSet<CubePos>,
    /// Edit bookkeeping per cube, only while persistence is on.
    versions: HashMap<CubePos, EditVersions>,
    /// Edit versions handed to the save thread and not yet acknowledged.
    /// A cube listed here stays pinned until its batch commits.
    pending_save: HashMap<CubePos, u64>,
    /// LOD cubes handed to the save thread and not yet acknowledged
    /// (task 2.7); a failed batch puts them back dirty for the next one.
    lod_pending_save: HashSet<(u8, CubePos)>,
    /// Open world database; `None` when persistence is off.
    storage: Option<Arc<Storage>>,
    /// Sends snapshot batches to the save thread.
    save_tx: Option<Sender<SaveBatch>>,
    /// Replies from the save thread, drained at the start of phase 5.
    ack_rx: Receiver<SaveAck>,
    /// The `jbm-save` thread; `None` when persistence is off.
    save_thread: Option<JoinHandle<()>>,
    /// Simulated time accumulated since the last snapshot.
    time_since_save: Duration,
    /// Batches the save thread committed (test accessor backing).
    saved_batches: u64,
    /// Player records as last committed, by name; a player is re-saved
    /// only when their record differs.
    saved_players: HashMap<String, PlayerRecord>,
    /// Records of players who left since the last snapshot, saved with the
    /// next batch so a disconnect never loses their last position.
    departed: Vec<(String, PlayerRecord)>,
    /// Batches the save thread reported failed (test accessor backing).
    failed_batches: u64,
    /// Batches handed to the save thread; each gets exactly one answer.
    sent_batches: u64,
    /// Block edits handled in phase 1, flushed early in phase 5 before the
    /// streaming pass and the `Tick` message.
    pending_changes: Vec<QueuedChange>,
    /// Desired-set offsets for the configured radii, sorted nearest-first.
    /// Computed once per config.
    desired: Vec<Offset>,
    /// The spawn position handed out in every `Welcome`.
    spawn: [f64; 3],
    /// Whether world eviction should run this tick: set when anything might
    /// have made a cube unwanted (a player moved or left, results arrived).
    /// An idle server never scans.
    eviction_dirty: bool,
    /// Joined players counted by the previous streaming pass, to spot joins
    /// and disconnects cheaply.
    joined_count: usize,
    /// The LOD pyramid built from ingested and edited LOD0 cubes. Eviction
    /// never touches it: LOD cubes outlive their LOD0 children, so builds
    /// stay visible from far away.
    lod: LodPyramid,
}

impl Server {
    /// A server with the default [`ServerConfig`]. Not `const`: building it
    /// starts the generation pool.
    pub fn new() -> Self {
        Self::with_config(ServerConfig::default())
    }

    /// A server with an explicit config: terrain seed, streaming shape,
    /// generation threading and an optional world directory. Panics if the
    /// world storage cannot open (locked, corrupt, too new); use
    /// [`Server::try_with_config`] to handle that.
    pub fn with_config(config: ServerConfig) -> Self {
        Self::try_with_config(config).expect("opening the world's storage")
    }

    /// [`Server::with_config`] without the panic: storage opens first, so
    /// the **stored** seed feeds the generator on reopen, and any storage
    /// error surfaces to the caller.
    pub fn try_with_config(config: ServerConfig) -> Result<Self, StorageError> {
        let mut registry = jbm_core::block::BlockRegistry::new();
        let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
        let storage = match &config.world_dir {
            Some(dir) => Some(Arc::new(Storage::open(dir, &registry, config.seed)?)),
            None => None,
        };
        let seed = storage.as_ref().map_or(config.seed, |s| s.seed());
        let generator = Arc::new(Generator::new(seed, blocks));
        let pool = GenPool::new(config.gen_threads, Arc::clone(&generator), storage.clone());
        let desired = desired_offsets(config.view_radius, config.vertical_radius);
        let spawn = generator.spawn();
        let solid = SolidTable::new(&registry);
        let state_count = registry.state_count();
        // Stored LOD only counts under the worldgen it was written with:
        // a mismatching fingerprint makes every stored LOD record read as
        // absent (and the next saves overwrite them).
        if let Some(storage) = &storage
            && config.lod_levels > 0
            && let Err(error) =
                storage.set_worldgen_fingerprint(crate::worldgen::overworld().fingerprint())
        {
            return Err(error);
        }
        let lod = LodPyramid::with_workers(
            config.lod_levels,
            Arc::clone(&generator),
            storage.clone(),
            config.gen_threads,
            config.gen_budget_per_tick,
        );

        let (save_tx, save_rx) = std::sync::mpsc::channel::<SaveBatch>();
        let (ack_tx, ack_rx) = std::sync::mpsc::channel::<SaveAck>();
        let save_thread = storage.clone().map(|storage| {
            std::thread::Builder::new()
                .name("jbm-save".to_owned())
                .spawn(move || save_loop(storage, save_rx, ack_tx))
                .expect("spawning the jbm-save thread")
        });

        Ok(Self {
            clock: TickClock::new(),
            tick_number: 0,
            connections: Vec::new(),
            next_player_id: 1,
            config,
            pool,
            generator,
            world: HashMap::new(),
            solid,
            state_count,
            pinned: HashSet::new(),
            versions: HashMap::new(),
            pending_save: HashMap::new(),
            lod_pending_save: HashSet::new(),
            storage,
            save_tx: Some(save_tx),
            ack_rx,
            save_thread,
            time_since_save: Duration::ZERO,
            saved_batches: 0,
            saved_players: HashMap::new(),
            departed: Vec::new(),
            failed_batches: 0,
            sent_batches: 0,
            pending_changes: Vec::new(),
            desired,
            spawn,
            eviction_dirty: true,
            joined_count: 0,
            lod,
        })
    }

    /// Register a fresh connection. It stays pending (receiving nothing)
    /// until a valid `Hello` arrives in a tick's message phase.
    pub fn add_connection(&mut self, link: ServerLink) {
        self.connections.push(Connection {
            link,
            state: ConnectionState::Pending,
        });
    }

    /// Run exactly one tick through the phases, in the design doc's fixed
    /// order.
    pub fn tick(&mut self) {
        self.tick_number += 1;
        self.phase_messages();
        self.phase_scheduled_block_ticks();
        self.phase_block_update_queue();
        self.phase_ecs_systems();
        // Persistence runs at the start of phase 5's section: acknowledgements
        // from the save thread are applied first, then an expired
        // save_interval snapshots dirty cubes. The tick thread itself never
        // compresses, encodes or waits on disk I/O — the save thread does.
        self.phase_save();
        self.phase_light_and_outgoing();
    }

    /// Feed `elapsed` time into the tick clock, run every due tick and
    /// return how many ran.
    pub fn update(&mut self, elapsed: Duration) -> u32 {
        let due = self.clock.advance(elapsed);
        for _ in 0..due {
            self.tick();
        }
        due
    }

    /// Ticks run so far.
    pub const fn tick_count(&self) -> u64 {
        self.tick_number
    }

    /// Whole ticks dropped by the catch-up cap; see [`TickClock`].
    pub const fn dropped_ticks(&self) -> u64 {
        self.clock.dropped_ticks()
    }

    /// Time left before the next clock-due tick — what the real-time loop
    /// sleeps for. Only meaningful when time is fed in via
    /// [`Server::update`].
    pub fn until_next_tick(&self) -> Duration {
        self.clock.until_next_tick()
    }

    /// Joined players in join order, for tests and diagnostics.
    pub fn players(&self) -> impl Iterator<Item = PlayerView> + '_ {
        self.connections.iter().filter_map(|conn| {
            conn.joined_player().map(|player| PlayerView {
                id: player.id,
                name: player.name.clone(),
                state: player.state,
                last_seq: player.last_seq,
                input_count: player.input_count,
            })
        })
    }

    /// Cubes the server currently keeps in memory.
    pub fn world_cube_count(&self) -> usize {
        self.world.len()
    }

    /// The level-`level` LOD cube at `pos`, when the pyramid holds one
    /// (levels are 1-based; `lod_levels` bounds the top).
    pub fn lod_cube(&self, level: u8, pos: CubePos) -> Option<&jbm_core::cube::Cube> {
        self.lod.get(level, pos).map(|lc| &lc.cube)
    }

    /// The cube at `(level, pos)` when it is ready to stream — every
    /// octant real or filled (task 2.7). Test and streaming access.
    pub fn lod_ready(&self, level: u8, pos: CubePos) -> Option<&jbm_core::cube::Cube> {
        self.lod.ready(level, pos)
    }

    /// Forwards to [`LodPyramid::request`]: queue `(level, pos)` for
    /// load-or-sample. Test and streaming access.
    pub fn request_lod(&mut self, level: u8, pos: CubePos) {
        self.lod.request(level, pos);
    }

    /// LOD jobs queued, running or finished-but-uncollected.
    pub fn lod_in_flight(&self) -> usize {
        self.lod.in_flight()
    }

    /// How many `sample_lod` sampling runs the generator has done — the
    /// test hook that proves storage reloads do not sample.
    pub fn lod_samples(&self) -> u64 {
        self.generator.lod_sample_calls()
    }

    /// The known-octant mask of the level-`level` LOD cube at `pos`, when
    /// the pyramid holds one.
    pub fn lod_known(&self, level: u8, pos: CubePos) -> Option<u8> {
        self.lod.get(level, pos).map(|lc| lc.known)
    }

    /// LOD cubes held across all levels.
    pub fn lod_cube_count(&self) -> usize {
        self.lod.total_cubes()
    }

    /// Approximate heap bytes of the LOD pyramid (see
    /// [`LodPyramid::heap_bytes`]); diagnostic, used by the LOD bench's
    /// default-view report.
    pub fn lod_heap_bytes(&self) -> usize {
        self.lod.heap_bytes()
    }

    /// The whole pyramid, for tests and diagnostics.
    pub fn lod_pyramid(&self) -> &LodPyramid {
        &self.lod
    }

    /// Cubes queued for generation, generating, or generated but not yet
    /// collected.
    pub fn in_flight(&self) -> usize {
        self.pool.in_flight()
    }

    /// Cubes currently sent to (and not unloaded from) the player.
    pub fn sent_count(&self, id: PlayerId) -> usize {
        self.connections
            .iter()
            .filter_map(Connection::joined_player)
            .find(|player| player.id == id)
            .map_or(0, |player| player.sent.len())
    }

    /// Cubes pinned by an accepted edit (never evicted). Without
    /// persistence that is the edited set; with a world directory it is
    /// every cube whose edit version is not yet committed
    /// (`version > saved_version`), which includes cubes in a save batch
    /// that has not been acknowledged yet.
    pub fn pinned_count(&self) -> usize {
        match &self.storage {
            Some(_) => self
                .versions
                .values()
                .filter(|v| v.version > v.saved_version)
                .count(),
            None => self.pinned.len(),
        }
    }

    /// Cubes with uncommitted edits (same as [`Server::pinned_count`]).
    pub fn dirty_count(&self) -> usize {
        self.pinned_count()
    }

    /// Cubes whose snapshot is with the save thread, unacknowledged.
    pub fn pending_save_count(&self) -> usize {
        self.pending_save.len()
    }

    /// Save batches committed by the save thread so far.
    pub fn saved_batches(&self) -> u64 {
        self.saved_batches
    }

    /// Save batches the save thread reported failed so far (their cubes
    /// stayed pinned for a retry).
    pub fn failed_batches(&self) -> u64 {
        self.failed_batches
    }

    /// Batches handed to the save thread and not yet answered (committed or
    /// failed). Tests wait for 0 before holding saves.
    pub fn unanswered_batches(&self) -> u64 {
        self.sent_batches - self.saved_batches - self.failed_batches
    }

    /// The open world storage, when persistence is on. Test and diagnostic
    /// access (the crash tests read the database through it).
    pub fn storage(&self) -> Option<&Arc<Storage>> {
        self.storage.as_ref()
    }

    /// The authoritative block at `pos`, or `None` when its cube is not
    /// in the server world.
    pub fn block(&self, pos: BlockPos) -> Option<BlockStateId> {
        let (cube_pos, local) = pos.split();
        self.world.get(&cube_pos).map(|cube| cube.get(local))
    }

    /// Phase 1: apply player inputs and channel messages. Connections are
    /// drained in join order, each connection's messages in arrival order,
    /// at most [`MESSAGES_PER_CONNECTION_PER_TICK`] per connection.
    /// Returns how many messages it handled (shutdown settling loops on it).
    fn phase_messages(&mut self) -> usize {
        let mut handled = 0;
        let mut dropped: Vec<usize> = Vec::new();
        for idx in 0..self.connections.len() {
            let mut remove = false;
            for _ in 0..MESSAGES_PER_CONNECTION_PER_TICK {
                let msg = match self.connections[idx].link.try_recv() {
                    Ok(Some(msg)) => {
                        handled += 1;
                        msg
                    }
                    Ok(None) => break,
                    Err(LinkClosed) => {
                        remove = true;
                        break;
                    }
                };
                if matches!(self.handle_message(idx, msg), Outcome::Drop) {
                    remove = true;
                    break;
                }
            }
            if remove {
                dropped.push(idx);
            }
        }
        // Remove back-to-front so the collected indices stay valid.
        for idx in dropped.into_iter().rev() {
            self.remove_connection(idx);
        }
        handled
    }

    /// Before the shutdown flush: handle every message the players already
    /// sent (inputs and edits still in the links) and step every queued
    /// input, so the saved world and positions match what the players last
    /// saw rather than lagging by a tick or two. Bounded.
    fn settle_received_input(&mut self) {
        for _ in 0..64 {
            let handled = self.phase_messages();
            let Server {
                connections,
                world,
                solid,
                ..
            } = self;
            let shapes = ServerShapes { world, solid };
            for conn in connections {
                if let ConnectionState::Joined(player) = &mut conn.state {
                    step_inputs(player, &shapes, usize::MAX);
                }
            }
            if handled == 0 {
                break;
            }
        }
    }

    /// Handle one inbound message for the connection at `idx`. Sends to a
    /// closed link drop the connection instead of panicking.
    ///
    /// Block edits are handled here, in message order as they arrive,
    /// validated against the player's authoritative state as it stands
    /// right now (phase 4 has not run yet this tick).
    fn handle_message(&mut self, idx: usize, msg: ClientMsg) -> Outcome {
        // Edits need the whole server (world, other players); everything
        // else only needs the one connection.
        match msg {
            ClientMsg::BreakBlock { pos } if self.connections[idx].joined_player().is_some() => {
                self.handle_break(idx, pos)
            }
            ClientMsg::PlaceBlock { pos, state }
                if self.connections[idx].joined_player().is_some() =>
            {
                self.handle_place(idx, pos, state)
            }
            msg => self.handle_message_rest(idx, msg),
        }
    }

    /// The non-edit half of [`Server::handle_message`].
    fn handle_message_rest(&mut self, idx: usize, msg: ClientMsg) -> Outcome {
        let conn = &mut self.connections[idx];
        match (&mut conn.state, msg) {
            (
                ConnectionState::Pending,
                ClientMsg::Hello {
                    protocol_version,
                    name,
                },
            ) => {
                if protocol_version != PROTOCOL_VERSION {
                    let reason =
                        format!("protocol version {protocol_version}, expected {PROTOCOL_VERSION}");
                    // The link may already be gone; the connection is
                    // dropped either way.
                    let _ = conn.link.send(ServerMsg::Rejected { reason });
                    return Outcome::Drop;
                }
                let player_id = PlayerId(self.next_player_id);
                self.next_player_id += 1;
                // A returning player starts where they left (persistence);
                // anyone else, or a corrupt record, gets the world spawn.
                let saved = match self.storage.as_ref().map(|s| s.load_player(&name)) {
                    Some(Ok(record)) => record,
                    Some(Err(error)) => {
                        eprintln!("jbm: {error}; spawning {name:?} at the world spawn");
                        None
                    }
                    None => None,
                };
                let record = saved.unwrap_or(PlayerRecord {
                    position: self.spawn,
                    yaw: 0.0,
                    pitch: 0.0,
                    flying: false,
                });
                let spawn = record.position;
                conn.state = ConnectionState::Joined(Box::new(Player {
                    id: player_id,
                    name,
                    // The Welcome's spawn is the initial state: velocity 0,
                    // not on ground.
                    state: PlayerState {
                        position: DVec3::from(spawn),
                        velocity: DVec3::ZERO,
                        on_ground: false,
                        flying: record.flying,
                    },
                    input_queue: VecDeque::new(),
                    last_seq: 0,
                    input_count: 0,
                    pending_ack: false,
                    center: cube_containing(DVec3::from(spawn)),
                    sent: HashSet::new(),
                    cursor: 0,
                    yaw: record.yaw,
                    pitch: record.pitch,
                }));
                let welcome = ServerMsg::Welcome {
                    player_id,
                    server_tick: self.tick_number,
                    tick_rate: TICKS_PER_SECOND,
                    // The saved position, or two blocks over the terrain
                    // surface at column (0, 0).
                    spawn,
                    yaw: record.yaw,
                    pitch: record.pitch,
                    flying: record.flying,
                };
                if conn.link.send(welcome).is_err() {
                    return Outcome::Drop;
                }
                Outcome::Keep
            }
            (ConnectionState::Pending, _) => {
                let disconnected = ServerMsg::Disconnected {
                    reason: "expected Hello as the first message".to_owned(),
                };
                let _ = conn.link.send(disconnected);
                Outcome::Drop
            }
            // A second Hello while joined is a protocol violation.
            (ConnectionState::Joined(_), ClientMsg::Hello { .. }) => {
                let disconnected = ServerMsg::Disconnected {
                    reason: "unexpected Hello while joined".to_owned(),
                };
                let _ = conn.link.send(disconnected);
                Outcome::Drop
            }
            (ConnectionState::Joined(player), ClientMsg::PlayerInput { seq, input, pitch }) => {
                player.input_count += 1;
                // Old or duplicate seqs are ignored outright; the queue
                // keeps arrival order, which a sane client makes seq order.
                if seq > player.last_seq {
                    // Look is display state, kept only to persist it.
                    player.yaw = input.yaw;
                    player.pitch = pitch;
                    player.input_queue.push_back((seq, input));
                    if player.input_queue.len() > INPUT_QUEUE_CAP {
                        player.input_queue.pop_front();
                    }
                }
                Outcome::Keep
            }
            (ConnectionState::Joined(_), ClientMsg::Ping { nonce }) => {
                let pong = ServerMsg::Pong {
                    nonce,
                    server_tick: self.tick_number,
                };
                if conn.link.send(pong).is_err() {
                    return Outcome::Drop;
                }
                Outcome::Keep
            }
            (ConnectionState::Joined(_), ClientMsg::Disconnect) => Outcome::Drop,
            // Edits never reach this half: handle_message routes them to
            // their own handlers while joined, and the Pending arm above
            // catches them while pending.
            (
                ConnectionState::Joined(_),
                ClientMsg::BreakBlock { .. } | ClientMsg::PlaceBlock { .. },
            ) => Outcome::Keep,
        }
    }

    /// A `BreakBlock` from the joined player at `idx`. Validation: the
    /// cube is loaded, the block center is within `REACH + 1` of the
    /// player's eye (the margin covers latency), and the block is not
    /// air. Accepted edits set the block, pin the cube and queue a
    /// broadcast; rejections queue the authoritative state back to the
    /// sender only. Rejections never disconnect.
    fn handle_break(&mut self, idx: usize, pos: BlockPos) -> Outcome {
        let eye = self.connections[idx]
            .joined_player()
            .expect("checked joined in handle_message")
            .state
            .eye();
        let valid = self
            .block(pos)
            .is_some_and(|state| state != BlockStateId::AIR)
            && within_reach(pos, eye);
        if valid {
            self.apply_edit(pos, BlockStateId::AIR);
        } else {
            self.queue_revert(idx, pos);
        }
        Outcome::Keep
    }

    /// A `PlaceBlock` from the joined player at `idx`. Validation adds to
    /// the break rules: the target is air, `state` is a registered
    /// non-air state, and a solid state's unit box strictly overlaps no
    /// joined player's box (touching is allowed).
    fn handle_place(&mut self, idx: usize, pos: BlockPos, state: BlockStateId) -> Outcome {
        let eye = self.connections[idx]
            .joined_player()
            .expect("checked joined in handle_message")
            .state
            .eye();
        let valid = self.block(pos) == Some(BlockStateId::AIR)
            && within_reach(pos, eye)
            && state != BlockStateId::AIR
            && state.0 < self.state_count
            && (!self.solid.is_solid(state) || !self.overlaps_any_player(pos));
        if valid {
            self.apply_edit(pos, state);
        } else {
            self.queue_revert(idx, pos);
        }
        Outcome::Keep
    }

    /// Applies an accepted edit: set the block, propagate it into the LOD
    /// pyramid (one cell per level, stopping at the first unchanged
    /// level), pin the cube against eviction (bumping its edit version
    /// under persistence), and queue the broadcast.
    fn apply_edit(&mut self, pos: BlockPos, state: BlockStateId) {
        let (cube_pos, local) = pos.split();
        {
            let lod_on = self.config.lod_levels > 0;
            let Server { world, lod, .. } = self;
            if let Some(cube) = world.get_mut(&cube_pos) {
                cube.set(local, state);
                if lod_on {
                    lod.edit_lod0(pos, cube);
                }
            }
        }
        match &self.storage {
            Some(_) => {
                self.versions.entry(cube_pos).or_default().version += 1;
            }
            None => {
                self.pinned.insert(cube_pos);
            }
        }
        self.pending_changes.push((pos, state, None));
    }

    /// Queues the authoritative state of `pos` back to the rejecting
    /// sender only, which reverts their optimistic edit. Nothing is sent
    /// when there is no authoritative state (the cube is not in the world,
    /// so the player cannot have it in their sent set either).
    fn queue_revert(&mut self, idx: usize, pos: BlockPos) {
        let Some(state) = self.block(pos) else {
            return;
        };
        let id = self.connections[idx]
            .joined_player()
            .expect("checked joined in handle_message")
            .id;
        self.pending_changes.push((pos, state, Some(id)));
    }

    /// Whether a solid block at `pos` would strictly overlap any joined
    /// player's box.
    fn overlaps_any_player(&self, pos: BlockPos) -> bool {
        self.connections
            .iter()
            .filter_map(Connection::joined_player)
            .any(|player| {
                let aabb = player.state.aabb();
                physics::block_overlaps_box(pos, aabb.min, aabb.max)
            })
    }

    /// Phase 2: run scheduled block ticks that are due. Empty stub until
    /// block scheduling lands (Stage 2).
    fn phase_scheduled_block_ticks(&mut self) {}

    /// Phase 3: process the block update queue (neighbor changes) to a
    /// per-tick budget. Empty stub until block updates land (Stage 2).
    fn phase_block_update_queue(&mut self) {}

    /// Persistence slice of the tick, run right before phase 5: apply the
    /// save thread's acknowledgements, then every `save_interval` snapshot
    /// dirty cubes and LOD and hand them off. No-op without a world
    /// directory.
    fn phase_save(&mut self) {
        if self.storage.is_none() {
            return;
        }
        self.drain_save_acks();
        self.time_since_save += TICK_PERIOD;
        if self.time_since_save < self.config.save_interval {
            return;
        }
        self.time_since_save = Duration::ZERO;
        let mut batch = SaveBatch::default();
        for (pos, edit) in &self.versions {
            // Pinned cubes whose current version is not already in flight.
            if edit.version > edit.saved_version
                && self.pending_save.get(pos) != Some(&edit.version)
                && let Some(cube) = self.world.get(pos)
            {
                batch.cubes.push((*pos, cube.clone(), edit.version));
            }
        }
        self.collect_lod_dirty(&mut batch);
        batch.players = self.player_snapshot();
        self.send_save_batch(batch);
    }

    /// Snapshots every LOD cube the pyramid marked dirty (ready cubes whose
    /// content changed) and not already in flight.
    fn collect_lod_dirty(&mut self, batch: &mut SaveBatch) {
        for (level, pos) in self.lod.take_store_dirty() {
            if self.lod_pending_save.contains(&(level, pos)) {
                continue;
            }
            if let Some(lc) = self.lod.get(level, pos)
                && lc.ready()
            {
                batch.lod.push((level, pos, lc.cube.clone(), lc.known));
            }
        }
    }

    /// Records of every joined player plus everyone who left since the last
    /// snapshot, keeping only those that differ from what is on disk.
    fn player_snapshot(&mut self) -> Vec<(String, PlayerRecord)> {
        let mut records: HashMap<String, PlayerRecord> = self.departed.drain(..).collect();
        for conn in &self.connections {
            if let Some(player) = conn.joined_player() {
                records.insert(player.name.clone(), player.record());
            }
        }
        let mut changed: Vec<(String, PlayerRecord)> = records
            .into_iter()
            .filter(|(name, record)| self.saved_players.get(name) != Some(record))
            .collect();
        changed.sort_by(|a, b| a.0.cmp(&b.0));
        changed
    }

    /// Removes a connection, remembering a joined player's record so the
    /// next save keeps where they left.
    fn remove_connection(&mut self, idx: usize) {
        let mut conn = self.connections.remove(idx);
        if self.storage.is_some()
            && let ConnectionState::Joined(player) = &mut conn.state
        {
            // Inputs that arrived before the disconnect still count: the
            // player saw them applied, so the saved position includes them.
            let shapes = ServerShapes {
                world: &self.world,
                solid: &self.solid,
            };
            step_inputs(player, &shapes, usize::MAX);
            self.departed.push((player.name.clone(), player.record()));
        }
    }

    /// Applies every reply that has arrived from the save thread.
    fn drain_save_acks(&mut self) {
        while let Ok(ack) = self.ack_rx.try_recv() {
            self.apply_save_ack(ack);
        }
    }

    fn apply_save_ack(&mut self, ack: SaveAck) {
        match ack {
            SaveAck::Committed(acks, lod, players) => {
                self.saved_batches += 1;
                self.saved_players.extend(players);
                for (pos, version) in acks {
                    let edit = self.versions.entry(pos).or_default();
                    edit.saved_version = edit.saved_version.max(version);
                    if self
                        .pending_save
                        .get(&pos)
                        .is_some_and(|&p| p <= edit.saved_version)
                    {
                        self.pending_save.remove(&pos);
                    }
                }
                for key in lod {
                    self.lod_pending_save.remove(&key);
                }
            }
            // A failed batch is logged and retried next interval: the cubes
            // stay dirty (never evicted), so edits are never dropped
            // silently. The batch's LOD cubes go back to the dirty set the
            // same way.
            // Players are re-sent next interval: `saved_players` only moves
            // on a commit, so their records still differ.
            SaveAck::Failed(positions, lod) => {
                self.failed_batches += 1;
                eprintln!(
                    "jbm: saving a batch of {} cubes and {} LOD cubes failed; retrying next interval",
                    positions.len(),
                    lod.len()
                );
                for pos in positions {
                    self.pending_save.remove(&pos);
                }
                self.lod_pending_save.clear();
                self.lod.mark_store_dirty(&lod);
            }
        }
    }

    /// Hands a snapshot batch to the save thread and marks its versions
    /// in flight. A dead save thread is logged loudly — those edits are
    /// only protected by the retry on the next interval.
    fn send_save_batch(&mut self, batch: SaveBatch) {
        if batch.is_empty() {
            return;
        }
        for (pos, _, version) in &batch.cubes {
            self.pending_save.insert(*pos, *version);
        }
        self.lod_pending_save
            .extend(batch.lod.iter().map(|&(level, pos, _, _)| (level, pos)));
        match &self.save_tx {
            Some(tx) if tx.send(batch).is_ok() => self.sent_batches += 1,
            _ => eprintln!("jbm: the save thread is gone; unsaved edits may be lost"),
        }
    }

    /// Sends everything still dirty and blocks until the save thread has
    /// acknowledged it (or [`FLUSH_TIMEOUT`] passes, which is logged).
    /// Called by [`ServerHandle::stop`](crate::run::ServerHandle::stop)
    /// paths and by `Drop`, so every accepted edit is persisted by the time
    /// the server is gone.
    pub fn flush(&mut self) {
        if self.storage.is_none() {
            return;
        }
        self.drain_save_acks();
        let mut batch = SaveBatch::default();
        for (pos, edit) in &self.versions {
            if edit.version > edit.saved_version
                && let Some(cube) = self.world.get(pos)
            {
                batch.cubes.push((*pos, cube.clone(), edit.version));
            }
        }
        self.collect_lod_dirty(&mut batch);
        batch.players = self.player_snapshot();
        self.send_save_batch(batch);
        // The saver answers batches in order, so once every batch sent so
        // far is answered, this one (which may hold only player records,
        // with no pending cube to mark it) is too.
        let deadline = Instant::now() + FLUSH_TIMEOUT;
        while self.saved_batches + self.failed_batches < self.sent_batches {
            let timeout = deadline.saturating_duration_since(Instant::now());
            match self.ack_rx.recv_timeout(timeout) {
                Ok(ack) => self.apply_save_ack(ack),
                Err(RecvTimeoutError::Timeout) => {
                    eprintln!(
                        "jbm: flush timed out with {} cubes unsaved",
                        self.pending_save.len()
                    );
                    break;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    }

    /// Phase 4: run ECS systems — here, player physics. Per player, in
    /// join order, up to [`INPUTS_PER_TICK`] queued inputs are stepped
    /// through the shared [`physics::step`] against a [`BlockShapes`]
    /// view of the server's world (a missing cube reads as `None` for its
    /// blocks, so players freeze at the edge of loaded terrain).
    ///
    /// Input-driven: a tick with no queued input does not step the player
    /// at all — a stalled client freezes in place. Anti-cheat and input
    /// timeouts are a later task; the caps above are the only protection
    /// for now.
    fn phase_ecs_systems(&mut self) {
        let Server {
            connections,
            world,
            solid,
            ..
        } = self;
        let shapes = ServerShapes { world, solid };
        for conn in connections {
            let ConnectionState::Joined(player) = &mut conn.state else {
                continue;
            };
            step_inputs(player, &shapes, INPUTS_PER_TICK);
        }
    }

    /// Phase 5: recompute dirty light and send outgoing changes — here,
    /// the tick's queued `BlockChange`s, then the cube streaming pass,
    /// then per player the `PlayerState` (only in ticks where at least one
    /// input was processed) followed by the `Tick` message every joined
    /// player receives, so cubes, blocks and state always precede their
    /// tick. A player whose link has closed is removed — a failed send
    /// never panics.
    fn phase_light_and_outgoing(&mut self) {
        self.flush_block_changes();
        if self.config.streaming {
            self.phase_streaming();
        }
        let tick_msg = ServerMsg::Tick {
            tick: self.tick_number,
        };
        let mut dropped: Vec<usize> = Vec::new();
        for (idx, conn) in self.connections.iter_mut().enumerate() {
            let ConnectionState::Joined(player) = &mut conn.state else {
                continue;
            };
            if player.pending_ack {
                let state_msg = ServerMsg::PlayerState {
                    ack_seq: player.last_seq,
                    state: player.state,
                };
                if conn.link.send(state_msg).is_err() {
                    dropped.push(idx);
                    continue;
                }
                player.pending_ack = false;
            }
            if conn.link.send(tick_msg.clone()).is_err() {
                dropped.push(idx);
            }
        }
        for idx in dropped.into_iter().rev() {
            self.remove_connection(idx);
        }
    }

    /// Sends the tick's queued `BlockChange`s: broadcasts go to every
    /// player whose sent set contains the cube, reverts to the rejecting
    /// sender only (and only if they still have the cube sent). This runs
    /// before the streaming pass and the `Tick`, which guarantees the
    /// "edit shows on the client within one tick" rule. Ordering safety: a
    /// `CubeData` sent later in the same tick already contains the edit —
    /// phase 1 mutated the world cube before any cube is cloned for
    /// streaming — so a freshly streamed cube is never stale.
    fn flush_block_changes(&mut self) {
        let changes = std::mem::take(&mut self.pending_changes);
        if changes.is_empty() {
            return;
        }
        let mut dropped: Vec<usize> = Vec::new();
        for (pos, state, recipient) in changes {
            let cube = pos.split().0;
            for (idx, conn) in self.connections.iter_mut().enumerate() {
                let ConnectionState::Joined(player) = &mut conn.state else {
                    continue;
                };
                let theirs = recipient.is_none_or(|id| id == player.id);
                if theirs
                    && player.sent.contains(&cube)
                    && conn
                        .link
                        .send(ServerMsg::BlockChange { pos, state })
                        .is_err()
                {
                    dropped.push(idx);
                }
            }
        }
        for idx in dropped.into_iter().rev() {
            self.remove_connection(idx);
        }
    }

    /// The streaming pass, once per tick. Collects finished generation
    /// results, then per player (in join order): unloads cubes that left
    /// their keep set, sends newly in-range cubes nearest-first up to the
    /// send budget, and requests generation for missing ones within the
    /// in-flight cap. Finally evicts world cubes no player keeps.
    fn phase_streaming(&mut self) {
        // 1. Collect finished generation results into the world. Inline
        //    mode generates up to the budget first; threaded mode drains
        //    whatever the workers finished (bounded by max_in_flight).
        //    Collection never overwrites: an edited cube may already be
        //    back in the world (pinned), so the stale generated copy is
        //    dropped — and must not touch the LOD pyramid either.
        let mut collected = false;
        let lod_on = self.config.lod_levels > 0;
        let inline = self.config.gen_threads == 0;
        let budget = self.config.gen_budget_per_tick;
        let Server {
            world, lod, pool, ..
        } = self;
        if inline {
            for _ in 0..budget {
                match pool.try_recv() {
                    Some((pos, cube, octant)) => {
                        collected = true;
                        if let std::collections::hash_map::Entry::Vacant(slot) = world.entry(pos) {
                            let inserted = slot.insert(cube);
                            if lod_on {
                                lod.insert_lod0_octant(pos, inserted, Some(&octant));
                            }
                        }
                    }
                    None => break,
                }
            }
        } else {
            while let Some((pos, cube, octant)) = pool.try_recv() {
                collected = true;
                if let std::collections::hash_map::Entry::Vacant(slot) = world.entry(pos) {
                    let inserted = slot.insert(cube);
                    if lod_on {
                        lod.insert_lod0_octant(pos, inserted, Some(&octant));
                    }
                }
            }
        }

        // Background LOD work (2.7) lands after LOD0 ingest and before
        // streaming reads the pyramid.
        lod.poll();

        // A join or disconnect changes who keeps which cubes.
        let joined_now = self
            .connections
            .iter()
            .filter(|conn| conn.joined_player().is_some())
            .count();
        let dirty = self.eviction_dirty || collected || joined_now != self.joined_count;
        self.joined_count = joined_now;

        // 2. Per player, in join order. The split borrows (connection vs
        //    world/pool/config) are disjoint fields of `self`.
        let mut dropped: Vec<usize> = Vec::new();
        let mut eviction_dirty = dirty;
        for idx in 0..self.connections.len() {
            let conn = &mut self.connections[idx];
            let ConnectionState::Joined(player) = &mut conn.state else {
                continue;
            };
            if stream_player(
                &mut conn.link,
                player,
                &self.world,
                &mut self.pool,
                &self.config,
                &self.desired,
                &mut eviction_dirty,
            ) {
                dropped.push(idx);
            }
        }
        for idx in dropped.into_iter().rev() {
            self.remove_connection(idx);
        }

        // 3. Eviction: drop world cubes no player's keep set contains,
        //    keeping every pinned cube. Unedited cubes regenerate exactly;
        //    edited cubes under persistence are reloaded from disk once
        //    their edits are committed (version <= saved_version), so
        //    dropping those is lossless too. Only cubes with uncommitted
        //    edits — unsaved, or in a batch the save thread has not
        //    acknowledged yet — stay pinned forever. Without persistence
        //    (world_dir: None) every edited cube stays pinned, as before
        //    2.3. This replaces the 1.7 "pinned cubes grow without bound"
        //    rule.
        if eviction_dirty {
            let (r, v) = (self.config.view_radius, self.config.vertical_radius);
            let centers: Vec<CubePos> = self
                .connections
                .iter()
                .filter_map(|conn| conn.joined_player().map(|player| player.center))
                .collect();
            let Server {
                world,
                pinned,
                versions,
                storage,
                eviction_dirty,
                ..
            } = self;
            let is_pinned = |pos: &CubePos| match storage {
                Some(_) => versions
                    .get(pos)
                    .is_some_and(|edit| edit.version > edit.saved_version),
                None => pinned.contains(pos),
            };
            world
                .retain(|pos, _| is_pinned(pos) || centers.iter().any(|&c| in_keep(*pos, c, r, v)));
            // Evicted cubes take their edit bookkeeping with them: their
            // committed edits are on disk, and a reload starts fresh.
            versions.retain(|pos, _| world.contains_key(pos));
            *eviction_dirty = false;
        }
    }
}

/// Steps up to `max` of `player`'s queued inputs, in seq order, through the
/// shared physics.
fn step_inputs(player: &mut Player, shapes: &ServerShapes<'_>, max: usize) {
    let mut processed = 0;
    while processed < max {
        let Some((seq, input)) = player.input_queue.pop_front() else {
            break;
        };
        // Defensive: phase 1 already filters, but the cap rules can leave a
        // stale seq behind after a burst.
        if seq <= player.last_seq {
            continue;
        }
        player.state = physics::step(player.state, &input, shapes);
        player.last_seq = seq;
        player.pending_ack = true;
        processed += 1;
    }
}

/// A [`BlockShapes`] view of the server's world: a cube lookup per block,
/// no clones. A missing cube reads as `None` (not loaded), which freezes
/// players at the edge of generated terrain.
struct ServerShapes<'a> {
    world: &'a HashMap<CubePos, jbm_core::cube::Cube>,
    solid: &'a SolidTable,
}

impl BlockShapes for ServerShapes<'_> {
    fn solid(&self, pos: BlockPos) -> Option<bool> {
        let (cube_pos, local) = pos.split();
        self.world
            .get(&cube_pos)
            .map(|cube| self.solid.is_solid(cube.get(local)))
    }
}

/// One player's slice of the streaming pass. Returns `true` when the link
/// closed and the connection must be dropped.
///
/// Cost contract: with an unchanged center and everything sent, this is a
/// handful of map lookups (the cursor already sits at the end of the offset
/// list) — no scans of the desired or sent sets.
fn stream_player(
    link: &mut ServerLink,
    player: &mut Player,
    world: &HashMap<CubePos, jbm_core::cube::Cube>,
    pool: &mut GenPool,
    config: &ServerConfig,
    desired: &[Offset],
    eviction_dirty: &mut bool,
) -> bool {
    // Center: the cube containing the authoritative state position.
    let center = cube_containing(player.state.position);
    let (r, v) = (config.view_radius, config.vertical_radius);

    if center != player.center {
        player.center = center;
        // Everything must be re-walked nearest-first from the new center.
        player.cursor = 0;
        *eviction_dirty = true;
        // Unload what left the keep set (the desired set grown by one in
        // each radius — hysteresis against border flapping).
        let to_unload: Vec<CubePos> = player
            .sent
            .iter()
            .copied()
            .filter(|&pos| !in_keep(pos, center, r, v))
            .collect();
        for pos in to_unload {
            player.sent.remove(&pos);
            if link.send(ServerMsg::UnloadCube { pos }).is_err() {
                return true;
            }
        }
    }

    // Send walk, nearest-first from the cursor. Already-sent cubes advance
    // the cursor for free; a missing cube stops the walk (the order stays
    // strictly nearest-first) after being requested.
    let mut sends_left = config.send_budget_per_tick;
    let mut i = player.cursor;
    while i < desired.len() {
        let pos = offset_pos(center, desired[i]);
        if player.sent.contains(&pos) {
            i += 1;
            continue;
        }
        let Some(cube) = world.get(&pos) else {
            if pool.in_flight() < config.max_in_flight {
                pool.request(pos);
            }
            break;
        };
        if sends_left == 0 {
            break;
        }
        let msg = ServerMsg::CubeData {
            pos,
            cube: Box::new(cube.clone()),
        };
        if link.send(msg).is_err() {
            player.cursor = i;
            return true;
        }
        player.sent.insert(pos);
        sends_left -= 1;
        i += 1;
    }
    player.cursor = i;

    // Request top-up: keep the pool fed ahead of the send cursor without
    // sending out of order. Runs only while something is missing, so an
    // idle player never scans.
    if pool.in_flight() < config.max_in_flight {
        for &offset in &desired[i..] {
            let pos = offset_pos(center, offset);
            if player.sent.contains(&pos) || world.contains_key(&pos) {
                continue;
            }
            pool.request(pos);
            if pool.in_flight() >= config.max_in_flight {
                break;
            }
        }
    }
    false
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // Persist every accepted edit, then shut the saver down. A panic
        // unwinding through here still flushes what it can; a flush failure
        // is logged by `flush` itself. With persistence on, input already
        // received is applied first so nothing the player saw is lost.
        if self.storage.is_some() {
            self.settle_received_input();
        }
        self.flush();
        self.save_tx = None;
        if let Some(thread) = self.save_thread.take() {
            let _ = thread.join();
        }
    }
}

/// The `jbm-save` thread's loop: take a snapshot batch, commit it as one
/// transaction, reply with acknowledgements. A failed commit is reported,
/// never swallowed — the tick thread keeps those cubes pinned and retries.
fn save_loop(storage: Arc<Storage>, batches: Receiver<SaveBatch>, acks: Sender<SaveAck>) {
    while let Ok(batch) = batches.recv() {
        let refs: Vec<(CubePos, &jbm_core::cube::Cube)> = batch
            .cubes
            .iter()
            .map(|(pos, cube, _)| (*pos, cube))
            .collect();
        let lod_refs: Vec<(u8, CubePos, &jbm_core::cube::Cube, u8)> = batch
            .lod
            .iter()
            .map(|&(level, pos, ref cube, known)| (level, pos, cube, known))
            .collect();
        let lod_keys: Vec<(u8, CubePos)> = batch
            .lod
            .iter()
            .map(|&(level, pos, _, _)| (level, pos))
            .collect();
        let ack = match storage.save_lod_batch(&refs, &lod_refs, &batch.players) {
            Ok(()) => SaveAck::Committed(
                batch
                    .cubes
                    .iter()
                    .map(|(pos, _, version)| (*pos, *version))
                    .collect(),
                lod_keys,
                batch.players,
            ),
            Err(error) => {
                eprintln!("jbm: committing a save batch failed: {error}");
                SaveAck::Failed(refs.into_iter().map(|(pos, _)| pos).collect(), lod_keys)
            }
        };
        if acks.send(ack).is_err() {
            break; // the server is gone
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{INPUT_QUEUE_CAP, INPUTS_PER_TICK, Server, ServerConfig, cube_containing};
    use jbm_core::block::{BlockStateId, builtin::Blocks};
    use jbm_core::coords::{BlockPos, CubePos};
    use jbm_core::link::{ClientLink, in_process_pair};
    use jbm_core::physics::MoveInput;
    use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};

    /// Every unit test here asserts exact message sequences, so it runs the
    /// server with streaming off (no CubeData noise).
    fn protocol_server() -> Server {
        Server::with_config(ServerConfig::protocol_only())
    }

    /// Streaming with tiny radii and inline generation: what the physics
    /// tests use, so players stand on real generated terrain.
    fn physics_server() -> Server {
        Server::with_config(ServerConfig {
            streaming: true,
            view_radius: 1,
            vertical_radius: 1,
            send_budget_per_tick: 64,
            gen_threads: 0,
            gen_budget_per_tick: 64,
            max_in_flight: 512,
            ..ServerConfig::default()
        })
    }

    /// A server with one client whose `Hello` is queued but not yet
    /// processed.
    fn pending_hello() -> (Server, ClientLink) {
        pending_hello_on(protocol_server())
    }

    fn pending_hello_on(server: Server) -> (Server, ClientLink) {
        let mut server = server;
        let (mut client, server_end) = in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "alice".to_owned(),
            })
            .unwrap();
        (server, client)
    }

    fn input(seq: u64, input: MoveInput) -> ClientMsg {
        ClientMsg::PlayerInput {
            seq,
            input,
            pitch: 0.0,
        }
    }

    fn fly_up(toggle: bool) -> MoveInput {
        MoveInput {
            fly_toggle: toggle,
            up: 1,
            ..MoveInput::IDLE
        }
    }

    fn drain(client: &mut ClientLink) -> Vec<ServerMsg> {
        let mut messages = Vec::new();
        while let Ok(Some(msg)) = client.try_recv() {
            messages.push(msg);
        }
        messages
    }

    /// Joins, settles terrain loading, and returns the spawn state.
    fn joined_and_loaded(
        server: &mut Server,
        client: &mut ClientLink,
    ) -> jbm_core::physics::PlayerState {
        server.tick();
        let _ = drain(client);
        for _ in 0..10 {
            server.tick();
            let _ = drain(client);
        }
        server.players().next().unwrap().state
    }

    #[test]
    fn hello_is_welcomed_with_first_id_and_tick_rate() {
        let (mut server, mut client) = pending_hello();
        server.tick();
        assert!(matches!(
            drain(&mut client).as_slice(),
            [ServerMsg::Welcome {
                player_id,
                server_tick: 1,
                tick_rate: 20,
                ..
            }, ServerMsg::Tick { tick: 1 }] if player_id.0 == 1
        ));
        let players: Vec<_> = server.players().collect();
        assert_eq!(players.len(), 1);
        assert_eq!(players[0].name, "alice");
        assert_eq!(players[0].input_count, 0);
        assert_eq!(players[0].last_seq, 0);
        assert_eq!(players[0].state.velocity, glam::DVec3::ZERO);
        assert!(!players[0].state.flying);
        assert!(!players[0].state.on_ground);
    }

    #[test]
    fn ids_are_handed_out_in_join_order_starting_at_one() {
        let mut server = protocol_server();
        let ids: Vec<u32> = [1, 2, 3]
            .iter()
            .map(|&n| {
                let (mut client, server_end) = in_process_pair();
                server.add_connection(server_end);
                client
                    .send(ClientMsg::Hello {
                        protocol_version: PROTOCOL_VERSION,
                        name: format!("p{n}"),
                    })
                    .unwrap();
                server.tick();
                drain(&mut client);
                let id = server.players().last().unwrap().id;
                drop(client);
                server.tick(); // drop the connection again
                id.0
            })
            .collect();
        assert_eq!(ids, [1, 2, 3]);
    }

    #[test]
    fn wrong_protocol_version_is_rejected_and_dropped() {
        let mut server = protocol_server();
        let (mut client, server_end) = in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION + 1,
                name: "mallory".to_owned(),
            })
            .unwrap();

        server.tick();
        assert!(matches!(
            drain(&mut client).as_slice(),
            [ServerMsg::Rejected { .. }]
        ));
        server.tick();
        assert_eq!(server.players().count(), 0);
        assert!(drain(&mut client).is_empty());
    }

    #[test]
    fn non_hello_while_pending_is_disconnected_and_dropped() {
        let mut server = protocol_server();
        let (mut client, server_end) = in_process_pair();
        server.add_connection(server_end);
        client.send(ClientMsg::Ping { nonce: 1 }).unwrap();

        server.tick();
        assert!(matches!(
            drain(&mut client).as_slice(),
            [ServerMsg::Disconnected { .. }]
        ));
        server.tick();
        assert_eq!(server.players().count(), 0);
    }

    #[test]
    fn ping_is_answered_with_a_pong_stamped_by_the_tick() {
        let (mut server, mut client) = pending_hello();
        server.tick(); // join
        let _ = drain(&mut client);
        client.send(ClientMsg::Ping { nonce: 7 }).unwrap();
        server.tick();
        assert!(matches!(
            drain(&mut client).as_slice(),
            [
                ServerMsg::Pong {
                    nonce: 7,
                    server_tick: 2
                },
                ServerMsg::Tick { tick: 2 }
            ]
        ));
    }

    #[test]
    fn inputs_are_applied_in_seq_order_and_duplicates_ignored() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        let spawn = joined_and_loaded(&mut server, &mut client);

        // Fly up one tick, then hover; the duplicate and the old seq must
        // not move the player again.
        client.send(input(1, fly_up(true))).unwrap();
        client.send(input(2, MoveInput::IDLE)).unwrap();
        client.send(input(1, fly_up(true))).unwrap();
        client.send(input(2, MoveInput::IDLE)).unwrap();
        server.tick();
        let players: Vec<_> = server.players().collect();
        assert_eq!(players[0].input_count, 4, "every message is counted");
        assert_eq!(players[0].last_seq, 2);
        assert_eq!(players[0].state.position.y, spawn.position.y + 0.55);
        assert!(players[0].state.flying);

        // The queued duplicates were dropped at arrival, not processed
        // later: one more empty tick changes nothing.
        server.tick();
        let players: Vec<_> = server.players().collect();
        assert_eq!(players[0].last_seq, 2);
        assert_eq!(players[0].state.position.y, spawn.position.y + 0.55);
    }

    #[test]
    fn at_most_four_inputs_are_processed_per_tick() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        let spawn = joined_and_loaded(&mut server, &mut client);

        for seq in 1..=6u64 {
            client.send(input(seq, fly_up(seq == 1))).unwrap();
        }
        server.tick();
        let players: Vec<_> = server.players().collect();
        assert_eq!(players[0].last_seq, INPUTS_PER_TICK as u64);
        // Mirror the physics' repeated additions so the compare is bitwise.
        let mut expected = spawn.position.y;
        for _ in 0..INPUTS_PER_TICK {
            expected += 0.55;
        }
        assert_eq!(players[0].state.position.y, expected);

        // The remaining two process on the next tick.
        server.tick();
        let players: Vec<_> = server.players().collect();
        assert_eq!(players[0].last_seq, 6);
        for _ in 0..2 {
            expected += 0.55;
        }
        assert_eq!(players[0].state.position.y, expected);
    }

    #[test]
    fn queue_cap_drops_the_oldest_inputs() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        joined_and_loaded(&mut server, &mut client);

        let burst = (INPUT_QUEUE_CAP + 5) as u64;
        for seq in 1..=burst {
            client.send(input(seq, fly_up(seq == 1))).unwrap();
        }
        server.tick();
        let players: Vec<_> = server.players().collect();
        assert_eq!(players[0].input_count, burst);
        // The queue kept the newest 20 (seqs 6..=25) and the first tick
        // processed four of those: the oldest five, fly toggle included,
        // were dropped on arrival.
        assert_eq!(players[0].last_seq, 9);

        // Everything queued eventually processes, exactly once each.
        for _ in 0..10 {
            server.tick();
        }
        let players: Vec<_> = server.players().collect();
        assert_eq!(players[0].last_seq, burst);
    }

    #[test]
    fn player_state_precedes_tick_only_in_ticks_with_inputs() {
        let (mut server, mut client) = pending_hello();
        server.tick();
        let _ = drain(&mut client);

        // No input: only the Tick.
        server.tick();
        assert!(matches!(
            drain(&mut client).as_slice(),
            [ServerMsg::Tick { tick: 2 }]
        ));

        client.send(input(1, MoveInput::IDLE)).unwrap();
        server.tick();
        assert!(matches!(
            drain(&mut client).as_slice(),
            [
                ServerMsg::PlayerState { ack_seq: 1, .. },
                ServerMsg::Tick { tick: 3 }
            ]
        ));

        server.tick();
        assert!(matches!(
            drain(&mut client).as_slice(),
            [ServerMsg::Tick { tick: 4 }]
        ));
    }

    #[test]
    fn streaming_center_follows_the_authoritative_position() {
        let mut server = Server::with_config(ServerConfig {
            streaming: true,
            view_radius: 2,
            vertical_radius: 1,
            send_budget_per_tick: 64,
            gen_threads: 0,
            gen_budget_per_tick: 64,
            max_in_flight: 512,
            ..ServerConfig::default()
        });
        let (mut client, server_end) = in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "walker".to_owned(),
            })
            .unwrap();
        server.tick();
        let _ = drain(&mut client);
        for _ in 0..10 {
            server.tick();
            let _ = drain(&mut client);
        }
        let start = server.players().next().unwrap().state;
        let center0 = jbm_core::coords::BlockPos {
            x: start.position.x.floor() as i64,
            y: start.position.y.floor() as i64,
            z: start.position.z.floor() as i64,
        }
        .split()
        .0;

        // Fly well above the terrain, then east across a cube border. The
        // unloaded rule stalls movement while cubes stream in, so the loop
        // runs until the position actually crosses.
        let east = std::f32::consts::FRAC_PI_2;
        let mut seq = 0u64;
        let mut crossed = false;
        for _ in 0..4000 {
            let state = server.players().next().unwrap().state;
            let rise = state.position.y < start.position.y + 40.0;
            let move_input = if rise {
                MoveInput {
                    up: 1,
                    fly_toggle: !state.flying,
                    ..MoveInput::IDLE
                }
            } else {
                MoveInput {
                    forward: 1,
                    sprint: true,
                    yaw: east,
                    ..MoveInput::IDLE
                }
            };
            seq += 1;
            client.send(input(seq, move_input)).unwrap();
            server.tick();
            for msg in drain(&mut client) {
                if let ServerMsg::CubeData { pos, .. } = msg
                    && pos.x >= center0.x + 2
                    && pos.z == center0.z
                {
                    crossed = true;
                }
            }
            let state = server.players().next().unwrap().state;
            if !rise && state.position.x >= start.position.x + 33.0 {
                break;
            }
        }
        let state = server.players().next().unwrap().state;
        assert!(
            state.position.x >= start.position.x + 33.0,
            "never crossed the cube border: {state:?}"
        );
        assert!(
            crossed,
            "no cubes streamed east of the start — the center did not follow"
        );
    }

    #[test]
    fn disconnect_removes_the_player() {
        let (mut server, mut client) = pending_hello();
        server.tick();
        let _ = drain(&mut client);
        assert_eq!(server.players().count(), 1);

        client.send(ClientMsg::Disconnect).unwrap();
        server.tick();
        assert_eq!(server.players().count(), 0);
    }

    #[test]
    fn closed_link_removes_the_player_without_panicking() {
        let (mut server, client) = pending_hello();
        server.tick();
        drop(client);
        server.tick();
        assert_eq!(server.players().count(), 0);
    }

    #[test]
    fn messages_are_capped_at_256_per_connection_per_tick() {
        let (mut server, mut client) = pending_hello();
        for nonce in 0..300u64 {
            client.send(ClientMsg::Ping { nonce }).unwrap();
        }

        server.tick();
        let messages = drain(&mut client);
        // The Hello plus 255 pings fit under the cap; then the tick
        // message from phase 5.
        let pongs: Vec<u64> = messages
            .iter()
            .filter_map(|m| match m {
                ServerMsg::Pong { nonce, .. } => Some(*nonce),
                _ => None,
            })
            .collect();
        assert_eq!(pongs.len(), 255);
        assert_eq!(pongs, (0..255).collect::<Vec<_>>());
        assert!(matches!(messages.last(), Some(ServerMsg::Tick { .. })));

        // Leftovers wait for the next tick: 45 pings plus the tick message.
        server.tick();
        let messages = drain(&mut client);
        let pongs: Vec<u64> = messages
            .iter()
            .filter_map(|m| match m {
                ServerMsg::Pong { nonce, .. } => Some(*nonce),
                _ => None,
            })
            .collect();
        assert_eq!(pongs, (255..300).collect::<Vec<_>>());
        assert!(matches!(messages.last(), Some(ServerMsg::Tick { .. })));
    }

    #[test]
    fn every_joined_player_receives_a_tick_message_each_tick() {
        let mut server = protocol_server();
        let mut clients = Vec::new();
        for n in 0..2 {
            let (mut client, server_end) = in_process_pair();
            server.add_connection(server_end);
            client
                .send(ClientMsg::Hello {
                    protocol_version: PROTOCOL_VERSION,
                    name: format!("p{n}"),
                })
                .unwrap();
            clients.push(client);
        }
        server.tick();
        for client in &mut clients {
            let _ = drain(client); // welcome + first tick
        }
        server.tick();
        server.tick();
        for client in &mut clients {
            assert!(matches!(
                drain(client).as_slice(),
                [ServerMsg::Tick { tick: 2 }, ServerMsg::Tick { tick: 3 }]
            ));
        }
    }

    #[test]
    fn update_runs_due_ticks_and_reports_the_count() {
        let (mut server, _client) = pending_hello();
        assert_eq!(server.update(crate::tick::TICK_PERIOD), 1);
        assert_eq!(server.tick_count(), 1);
        assert_eq!(server.update(crate::tick::TICK_PERIOD * 3), 3);
        assert_eq!(server.tick_count(), 4);
    }

    // ------------------------------------------------------------------
    // Block edits (task 1.7)

    /// Joins two players on a streaming server and settles their cubes.
    /// Both stand at spawn (identical sent sets), and no inputs are ever
    /// sent, so their authoritative state stays exactly at spawn.
    fn two_settled_players() -> (Server, ClientLink, ClientLink) {
        let (mut server, mut a) = pending_hello_on(physics_server());
        let (mut b, server_end) = in_process_pair();
        server.add_connection(server_end);
        b.send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "bob".to_owned(),
        })
        .unwrap();
        for _ in 0..12 {
            server.tick();
            let _ = drain(&mut a);
            let _ = drain(&mut b);
        }
        assert_eq!(server.players().count(), 2);
        (server, a, b)
    }

    /// The topmost solid block in column (0, 0), from the spawn feet down.
    fn surface_block(server: &Server) -> BlockPos {
        let feet = server.players().next().unwrap().state.position;
        let mut y = feet.y.floor() as i64;
        loop {
            let solid = server
                .block(BlockPos { x: 0, y, z: 0 })
                .is_some_and(|state| state != BlockStateId::AIR);
            if solid {
                return BlockPos { x: 0, y, z: 0 };
            }
            y -= 1;
        }
    }

    #[test]
    fn break_and_place_apply_pin_and_broadcast_within_one_tick() {
        let (mut server, mut a, mut b) = two_settled_players();
        let ground = surface_block(&server);

        a.send(ClientMsg::BreakBlock { pos: ground }).unwrap();
        server.tick();
        // The edit applied and the cube is pinned.
        assert_eq!(server.block(ground), Some(BlockStateId::AIR));
        assert_eq!(server.pinned_count(), 1);
        // Both players (both have the cube loaded) learn it before Tick.
        for client in [&mut a, &mut b] {
            assert!(matches!(
                drain(client).as_slice(),
                [
                    ServerMsg::BlockChange {
                        pos,
                        state: BlockStateId::AIR
                    },
                    ServerMsg::Tick { .. }
                ] if *pos == ground
            ));
        }

        // Place stone back into the now-air cell.
        let stone = Blocks::register(&mut jbm_core::block::BlockRegistry::new()).stone;
        a.send(ClientMsg::PlaceBlock {
            pos: ground,
            state: stone,
        })
        .unwrap();
        server.tick();
        assert_eq!(server.block(ground), Some(stone));
        assert!(matches!(
            drain(&mut a).as_slice(),
            [
                ServerMsg::BlockChange { state, .. },
                ServerMsg::Tick { .. }
            ] if *state == stone
        ));
    }

    #[test]
    fn rejected_edits_revert_the_sender_only() {
        let (mut server, mut a, mut b) = two_settled_players();
        let ground = surface_block(&server);
        let ground_state = server.block(ground).unwrap();
        let blocks = Blocks::register(&mut jbm_core::block::BlockRegistry::new());
        let feet = server.players().next().unwrap().state.position;

        // A grab bag of rejections: (message, the state the revert must
        // carry). Out of reach targets a loaded cube 40 blocks east.
        let far = BlockPos {
            x: 40,
            y: ground.y,
            z: 0,
        };
        assert!(server.block(far).is_some(), "far cube is loaded");
        let cases: Vec<(ClientMsg, BlockStateId)> = vec![
            (
                ClientMsg::BreakBlock { pos: far },
                server.block(far).unwrap(),
            ),
            // Breaking air: the cell above the surface.
            (
                ClientMsg::BreakBlock {
                    pos: BlockPos {
                        x: 0,
                        y: feet.y.floor() as i64,
                        z: 0,
                    },
                },
                BlockStateId::AIR,
            ),
            // Placing into the solid ground.
            (
                ClientMsg::PlaceBlock {
                    pos: ground,
                    state: blocks.stone,
                },
                ground_state,
            ),
            // Placing into the player's own feet cell (strict overlap).
            (
                ClientMsg::PlaceBlock {
                    pos: BlockPos {
                        x: 0,
                        y: feet.y.floor() as i64,
                        z: 0,
                    },
                    state: blocks.stone,
                },
                BlockStateId::AIR,
            ),
            // Placing air and an unknown state.
            (
                ClientMsg::PlaceBlock {
                    pos: BlockPos {
                        x: 0,
                        y: feet.y.floor() as i64,
                        z: 0,
                    },
                    state: BlockStateId::AIR,
                },
                BlockStateId::AIR,
            ),
            (
                ClientMsg::PlaceBlock {
                    pos: BlockPos {
                        x: 0,
                        y: feet.y.floor() as i64,
                        z: 0,
                    },
                    state: BlockStateId(999_999),
                },
                BlockStateId::AIR,
            ),
        ];
        for (msg, revert_state) in cases {
            a.send(msg.clone()).unwrap();
            server.tick();
            assert!(
                matches!(
                    drain(&mut a).as_slice(),
                    [
                        ServerMsg::BlockChange { state, .. },
                        ServerMsg::Tick { .. }
                    ] if *state == revert_state
                ),
                "sender must be reverted for {msg:?}"
            );
            assert_eq!(
                drain(&mut b).as_slice(),
                &[ServerMsg::Tick {
                    tick: server.tick_count()
                }],
                "nobody else hears about a rejection: {msg:?}"
            );
            assert_eq!(server.pinned_count(), 0, "nothing was edited: {msg:?}");
        }

        // An edit into an unloaded cube is rejected too, but the player
        // cannot have that cube in their sent set (a sent cube is always
        // in the server world), so there is no revert to send.
        let unloaded = BlockPos {
            x: 0,
            y: 40_000,
            z: 0,
        };
        assert_eq!(server.block(unloaded), None);
        a.send(ClientMsg::BreakBlock { pos: unloaded }).unwrap();
        server.tick();
        assert_eq!(
            drain(&mut a).as_slice(),
            &[ServerMsg::Tick {
                tick: server.tick_count()
            }]
        );
        // The connection survives every rejection.
        assert_eq!(server.players().count(), 2);
    }

    #[test]
    fn edits_from_a_pending_connection_are_disconnected() {
        let mut server = physics_server();
        let (mut client, server_end) = in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::BreakBlock {
                pos: BlockPos { x: 0, y: 0, z: 0 },
            })
            .unwrap();
        server.tick();
        assert!(matches!(
            drain(&mut client).as_slice(),
            [ServerMsg::Disconnected { .. }]
        ));
        server.tick();
        assert_eq!(server.players().count(), 0);
    }

    /// Flies the (only) player toward `yaw`, first rising to `cruise_y`
    /// (well above any terrain) then sprint-flying horizontally, until
    /// `done` holds. Steps and drains every tick; bounded by `max_ticks`.
    fn fly_until(
        server: &mut Server,
        client: &mut ClientLink,
        yaw: f32,
        cruise_y: f64,
        seq: &mut u64,
        mut done: impl FnMut(&Server, &[ServerMsg]) -> bool,
        max_ticks: usize,
    ) -> bool {
        for _ in 0..max_ticks {
            let state = server.players().next().unwrap().state;
            let rising = state.position.y < cruise_y;
            let fly = MoveInput {
                up: rising.into(),
                forward: (!rising) as i8,
                sprint: true,
                fly_toggle: !state.flying,
                yaw,
                ..MoveInput::IDLE
            };
            *seq += 1;
            client.send(input(*seq, fly)).unwrap();
            server.tick();
            let messages = drain(client);
            if done(server, &messages) {
                return true;
            }
        }
        false
    }

    #[test]
    fn a_pinned_cube_survives_walking_away_and_back() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        let _ = joined_and_loaded(&mut server, &mut client);
        let ground = surface_block(&server);
        let pinned_cube = ground.split().0;

        client.send(ClientMsg::BreakBlock { pos: ground }).unwrap();
        server.tick();
        let _ = drain(&mut client);
        assert_eq!(server.block(ground), Some(BlockStateId::AIR));
        assert_eq!(server.pinned_count(), 1);

        // Fly east (yaw pi/2) until the pinned cube leaves the keep set
        // and is unloaded from the client.
        let mut seq = 0u64;
        let start = server.players().next().unwrap().state.position;
        let cruise = start.y + 40.0;
        let left = fly_until(
            &mut server,
            &mut client,
            std::f32::consts::FRAC_PI_2,
            cruise,
            &mut seq,
            |_, messages| {
                messages
                    .iter()
                    .any(|m| matches!(m, ServerMsg::UnloadCube { pos } if *pos == pinned_cube))
            },
            4000,
        );
        assert!(left, "the pinned cube was never unloaded");
        // Evicted world cubes: everything unpinned around spawn is gone,
        // the pinned cube stays.
        assert_eq!(server.block(ground), Some(BlockStateId::AIR));
        let beside = BlockPos {
            x: 0,
            y: ground.y,
            z: 32,
        };
        assert_eq!(
            server.block(beside),
            None,
            "unpinned neighbour cube was evicted"
        );
        assert_eq!(server.pinned_count(), 1);

        // Fly home (yaw -pi/2): the cube streams back with the edit.
        let returned = fly_until(
            &mut server,
            &mut client,
            -std::f32::consts::FRAC_PI_2,
            cruise,
            &mut seq,
            |_, messages| {
                messages
                    .iter()
                    .any(|m| matches!(m, ServerMsg::CubeData { pos, .. } if *pos == pinned_cube))
            },
            4000,
        );
        assert!(returned, "the pinned cube never streamed back");
        assert_eq!(server.block(ground), Some(BlockStateId::AIR));
    }

    #[test]
    fn world_cube_count_keeps_only_pinned_cubes_after_the_player_leaves() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        let _ = joined_and_loaded(&mut server, &mut client);
        let loaded = server.world_cube_count();
        assert!(loaded > 1);

        let ground = surface_block(&server);
        client.send(ClientMsg::BreakBlock { pos: ground }).unwrap();
        server.tick();
        let _ = drain(&mut client);

        client.send(ClientMsg::Disconnect).unwrap();
        for _ in 0..5 {
            server.tick();
        }
        drop(client);
        for _ in 0..5 {
            server.tick();
        }
        assert_eq!(server.players().count(), 0);
        assert_eq!(server.pinned_count(), 1);
        assert_eq!(
            server.world_cube_count(),
            1,
            "only the pinned cube survives the player leaving"
        );
    }

    #[test]
    fn collecting_generation_never_overwrites_an_edited_cube() {
        let (mut server, mut client) = pending_hello_on(physics_server());
        let _ = joined_and_loaded(&mut server, &mut client);
        let center = cube_containing(server.players().next().unwrap().state.position);
        // In the keep set (r + 1 = 2) but outside the desired set (r = 1):
        // streaming will keep it, but never requests or sends it.
        let target = CubePos {
            x: center.x + 2,
            y: center.y,
            z: center.z,
        };
        assert!(!server.world.contains_key(&target));

        // Request generation, then land an edited cube in the world before
        // the result is collected — the race `entry().or_insert` must win.
        server.pool.request(target);
        let blocks = Blocks::register(&mut jbm_core::block::BlockRegistry::new());
        server
            .world
            .insert(target, jbm_core::cube::Cube::filled(blocks.glass));
        let probe = target.block(jbm_core::coords::LocalPos::new_unchecked(1, 1, 1));
        server.tick();
        let _ = drain(&mut client);

        assert_eq!(
            server.block(probe),
            Some(blocks.glass),
            "generation overwrote the edited cube"
        );
    }
}
