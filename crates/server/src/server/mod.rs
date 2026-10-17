//! The authoritative server: connection handling and the fixed-order tick
//! phases from the design doc's "Simulation" section. Clients are talked to
//! only through [`ServerLink`]s.
//!
//! Module map: [`config`] (the knobs), [`edits`] (Break/Place validation
//! and application), [`persist`] (snapshot batches and the save thread),
//! [`stream`] (phase 5's streaming pass), [`stream_plan`] (the pure
//! offset/keep math) and [`stats`] (read-only views, including the
//! [`ServerStats`] snapshot).

mod config;
mod edits;
mod persist;
mod stats;
mod stream;
mod stream_plan;

pub use config::{DEFAULT_SEED, ServerConfig};
pub use stats::{LodStats, PlayerStats, SaveStats, ServerStats, SpanStats};

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use glam::DVec3;
use jbm_core::block::BlockStateId;
use jbm_core::coords::{BlockPos, CubePos};
use jbm_core::link::{LinkClosed, ServerLink};
use jbm_core::lod::CubeKey;
use jbm_core::physics::{self, BlockShapes, MoveInput, PlayerState, SolidTable};
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, PlayerId, ServerMsg};

use crate::gen_pool::GenPool;
use crate::lod::LodPyramid;
use crate::spans::SurfaceSpans;
use crate::storage::{PlayerRecord, Storage, StorageError};
use crate::terrain::Generator;
use crate::tick::{TICKS_PER_SECOND, TickClock};

use persist::{EditVersions, SaveAck, SaveBatch};
use stream_plan::{LodWalkEntry, Offset, level_center};

/// Most inbound messages one connection may deliver in a single tick.
/// Leftovers wait for the next tick, so a flooding client cannot stall the
/// tick.
const MESSAGES_PER_CONNECTION_PER_TICK: usize = 256;

/// Queued [`MoveInput`]s per player; on overflow the oldest is dropped.
const INPUT_QUEUE_CAP: usize = 20;

/// Inputs stepped per player per tick in phase 4.
const INPUTS_PER_TICK: usize = 4;

/// One queued outgoing block edit. `None` broadcasts to every player with
/// the cube loaded; `Some(id)` sends to that player only (a rejection
/// revert).
type QueuedChange = (BlockPos, BlockStateId, Option<PlayerId>);

/// The cube containing a world position given in block coordinates.
pub(crate) fn cube_containing(position: DVec3) -> CubePos {
    BlockPos {
        x: position.x.floor() as i64,
        y: position.y.floor() as i64,
        z: position.z.floor() as i64,
    }
    .split()
    .0
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

/// Per-level LOD ring streaming state for one player: the level's center
/// cube and the cubes already sent.
struct PlayerLodLevel {
    center: CubePos,
    sent: HashSet<CubePos>,
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
    /// The desired offsets (the camera window plus the surface-following
    /// span extras), sorted nearest-first. Rebuilt when the center moves
    /// or a span lands; empty until the first tick.
    desired: Vec<Offset>,
    /// The span version `desired` was built at.
    desired_version: u64,
    /// Index into the nearest-first desired offsets: the next cube that
    /// may still need sending. Reset whenever the center moves or the
    /// desired list is rebuilt, so an idle player costs O(1) per tick.
    cursor: usize,
    /// Look direction from the newest accepted input (radians); persisted
    /// with the player and handed back in the next `Welcome`.
    yaw: f32,
    pitch: f32,
    /// LOD ring state per level (`lod[l - 1]` is level `l`); empty when
    /// the pyramid is off (`lod_levels == 0`).
    lod: Vec<PlayerLodLevel>,
    /// This player's merged LOD walk (level 1's window plus its surface
    /// spans), sorted like the window-only walk. Rebuilt when any level's
    /// center moves or a span lands; empty until the first streaming
    /// tick.
    lod_walk: Vec<LodWalkEntry>,
    /// The span version `lod_walk` was built at.
    lod_walk_version: u64,
    /// Cursor into the merged LOD walk: every entry before it is sent.
    /// Reset whenever any level's center moves or the walk is rebuilt;
    /// while cubes are missing it sits at the first unsent entry.
    lod_cursor: usize,
    /// Queued re-sends: cubes the pyramid changed that this player
    /// already has. Drained ahead of the walk within the LOD budget, and
    /// persistent across ticks so nothing starves.
    lod_resend: VecDeque<CubeKey>,
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
    /// LOD cubes handed to the save thread and not yet acknowledged; a
    /// failed batch puts them back dirty for the next one.
    lod_pending_save: HashSet<CubeKey>,
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
    /// The surface span store: per-column surface height bounds, filled
    /// incrementally, feeding the surface-following vertical windows of
    /// LOD0 and the level-1 ring.
    spans: SurfaceSpans,
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
        let generator = Arc::new(Generator::with_preset(seed, blocks, config.worldgen));
        let pool = GenPool::new(config.gen_threads, Arc::clone(&generator), storage.clone());
        let spawn = generator.spawn();
        let solid = SolidTable::new(&registry);
        let state_count = registry.state_count();
        // Stored LOD only counts under the worldgen it was written with:
        // a mismatching fingerprint makes every stored LOD record read as
        // absent (and the next saves overwrite them).
        if let Some(storage) = &storage
            && config.lod_levels > 0
            && let Err(error) =
                storage.set_worldgen_fingerprint(config.worldgen.compiled().fingerprint())
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
                .spawn(move || persist::save_loop(storage, save_rx, ack_tx))
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
            spans: SurfaceSpans::new(),
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

    /// Test and bench access: teleport the player to `position`, bypassing
    /// physics. The next tick's streaming phases observe the moved center
    /// exactly as they would after a real teleport; no inputs are queued,
    /// so the state sits at `position` until the next call.
    pub fn teleport_player(&mut self, id: PlayerId, position: DVec3) {
        for conn in &mut self.connections {
            if let ConnectionState::Joined(player) = &mut conn.state
                && player.id == id
            {
                player.state.position = position;
                return;
            }
        }
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
                let lod_levels = self.config.lod_levels;
                let lod0_center = cube_containing(DVec3::from(spawn));
                let lod_r = self.config.lod_view_radius;
                let lod_v = self.config.lod_vertical_radius;
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
                    desired: Vec::new(),
                    // u64::MAX forces the first streaming tick to build
                    // the desired list (and feed the span store) even
                    // though a stationary player never moves a center.
                    desired_version: u64::MAX,
                    cursor: 0,
                    yaw: record.yaw,
                    pitch: record.pitch,
                    lod: (1..=lod_levels)
                        .map(|level| PlayerLodLevel {
                            center: level_center(lod0_center, level),
                            sent: HashSet::new(),
                        })
                        .collect(),
                    lod_walk: Vec::new(),
                    lod_walk_version: u64::MAX,
                    lod_cursor: 0,
                    lod_resend: VecDeque::new(),
                }));
                let _ = (lod_r, lod_v);
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

    /// Phase 2: run scheduled block ticks that are due. Empty stub until
    /// block scheduling lands (Stage 2).
    fn phase_scheduled_block_ticks(&mut self) {}

    /// Phase 3: process the block update queue (neighbor changes) to a
    /// per-tick budget. Empty stub until block updates land (Stage 2).
    fn phase_block_update_queue(&mut self) {}

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
        // A held save (the storage test hook) parks the save thread in
        // `enter_save` indefinitely; release it so the flush and the join
        // below can finish — shutdown must never hang on the hook. A
        // no-op when nothing is held.
        if let Some(storage) = &self.storage {
            storage.release_saves();
        }
        self.flush();
        self.save_tx = None;
        if let Some(thread) = self.save_thread.take() {
            let _ = thread.join();
        }
    }
}

/// Helpers shared by the per-submodule unit tests: small deterministic
/// servers, joined clients, and drain/fly loops.
#[cfg(test)]
pub(crate) mod test_util {
    use jbm_core::block::BlockStateId;
    use jbm_core::coords::BlockPos;
    use jbm_core::link::{ClientLink, in_process_pair};
    use jbm_core::physics::MoveInput;
    use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};

    use super::{Server, ServerConfig};

    /// Every protocol unit test asserts exact message sequences, so it runs
    /// the server with streaming off (no CubeData noise).
    pub(crate) fn protocol_server() -> Server {
        Server::with_config(ServerConfig::protocol_only())
    }

    /// Streaming with tiny radii and inline generation: what the physics
    /// and edit tests use, so players stand on real generated terrain.
    pub(crate) fn physics_server() -> Server {
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
    pub(crate) fn pending_hello() -> (Server, ClientLink) {
        pending_hello_on(protocol_server())
    }

    pub(crate) fn pending_hello_on(server: Server) -> (Server, ClientLink) {
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

    pub(crate) fn input(seq: u64, input: MoveInput) -> ClientMsg {
        ClientMsg::PlayerInput {
            seq,
            input,
            pitch: 0.0,
        }
    }

    pub(crate) fn fly_up(toggle: bool) -> MoveInput {
        MoveInput {
            fly_toggle: toggle,
            up: 1,
            ..MoveInput::IDLE
        }
    }

    pub(crate) fn drain(client: &mut ClientLink) -> Vec<ServerMsg> {
        let mut messages = Vec::new();
        while let Ok(Some(msg)) = client.try_recv() {
            messages.push(msg);
        }
        messages
    }

    /// A drain with LOD ring traffic filtered out, for asserting exact
    /// edit sequences. These servers keep their (default) LOD rings on, so
    /// `LodCubeData` legitimately interleaves with `BlockChange` for many
    /// ticks after a join; the edit contract itself — the change lands
    /// before `Tick` — is unchanged.
    pub(crate) fn drain_edits(client: &mut ClientLink) -> Vec<ServerMsg> {
        drain(client)
            .into_iter()
            .filter(|m| {
                !matches!(
                    m,
                    ServerMsg::LodCubeData { .. } | ServerMsg::UnloadLodCube { .. }
                )
            })
            .collect()
    }

    /// Joins, settles terrain loading, and returns the spawn state.
    pub(crate) fn joined_and_loaded(
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

    /// Joins two players on a streaming server and settles their cubes.
    /// Both stand at spawn (identical sent sets), and no inputs are ever
    /// sent, so their authoritative state stays exactly at spawn.
    pub(crate) fn two_settled_players() -> (Server, ClientLink, ClientLink) {
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
    pub(crate) fn surface_block(server: &Server) -> BlockPos {
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

    /// Flies the (only) player toward `yaw`, first rising to `cruise_y`
    /// (well above any terrain) then sprint-flying horizontally, until
    /// `done` holds. Steps and drains every tick; bounded by `max_ticks`.
    pub(crate) fn fly_until(
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
}

#[cfg(test)]
mod tests {
    use jbm_core::link::in_process_pair;
    use jbm_core::physics::MoveInput;
    use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};

    use super::test_util::*;
    use super::{INPUT_QUEUE_CAP, INPUTS_PER_TICK};

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
}
