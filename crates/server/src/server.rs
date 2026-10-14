//! The authoritative server: connection handling and the fixed-order tick
//! phases from the design doc's "Simulation" section. Clients are talked to
//! only through [`ServerLink`]s.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use jbm_core::coords::{BlockPos, CubePos};
use jbm_core::link::{LinkClosed, ServerLink};
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, PlayerId, ServerMsg};

use crate::gen_pool::GenPool;
use crate::streaming::{Offset, desired_offsets, in_keep, offset_pos};
use crate::terrain::Generator;
use crate::tick::{TICKS_PER_SECOND, TickClock};

/// Most inbound messages one connection may deliver in a single tick.
/// Leftovers wait for the next tick, so a flooding client cannot stall the
/// tick.
const MESSAGES_PER_CONNECTION_PER_TICK: usize = 256;

/// The seed every default server generates its world with. Fixed so a
/// fresh install always starts on the same terrain.
pub const DEFAULT_SEED: u64 = 0x0055_EED1_5EED;

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
        }
    }
}

impl ServerConfig {
    /// Streaming off and no generation threads: a server that only speaks
    /// the handshake/ping/tick protocol. Tests and benches that assert
    /// exact message sequences use this so their meaning stays unchanged.
    pub fn protocol_only() -> Self {
        Self {
            streaming: false,
            gen_threads: 0,
            ..Self::default()
        }
    }
}

/// The cube containing a world position given in block coordinates.
fn cube_containing(position: [f64; 3]) -> CubePos {
    BlockPos {
        x: position[0].floor() as i64,
        y: position[1].floor() as i64,
        z: position[2].floor() as i64,
    }
    .split()
    .0
}

/// The player's latest movement state as stored server-side. Placeholder
/// shape mirroring [`ClientMsg::PlayerInput`]; task 1.6 replaces it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerInput {
    pub client_tick: u64,
    pub position: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
}

/// Read-only view of a joined player, for tests and diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerView {
    pub id: PlayerId,
    pub name: String,
    pub latest_input: Option<PlayerInput>,
    pub input_count: u64,
}

/// State the server tracks for one joined player.
struct Player {
    id: PlayerId,
    name: String,
    latest_input: Option<PlayerInput>,
    input_count: u64,
    /// The cube the player is centered on (latest input position, else the
    /// spawn). Drives the desired and keep sets.
    center: CubePos,
    /// Cubes this player has been sent and not yet had unloaded. A cube is
    /// never sent twice while it stays in this set.
    sent: HashSet<CubePos>,
    /// Index into the precomputed nearest-first desired offsets: the next
    /// cube that may still need sending. Reset whenever the center moves,
    /// so an idle player costs O(1) per tick.
    cursor: usize,
}

enum ConnectionState {
    /// Connected, but no valid `Hello` has arrived yet.
    Pending,
    /// Welcomed and playing.
    Joined(Player),
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
    /// Generated cubes the server keeps in memory, keyed by position.
    world: HashMap<CubePos, jbm_core::cube::Cube>,
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
}

impl Server {
    /// A server with the default [`ServerConfig`]. Not `const`: building it
    /// starts the generation pool.
    pub fn new() -> Self {
        Self::with_config(ServerConfig::default())
    }

    /// A server with an explicit config: terrain seed, streaming shape and
    /// generation threading.
    pub fn with_config(config: ServerConfig) -> Self {
        let mut registry = jbm_core::block::BlockRegistry::new();
        let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
        let generator = Arc::new(Generator::new(config.seed, blocks));
        let pool = GenPool::new(config.gen_threads, Arc::clone(&generator));
        let desired = desired_offsets(config.view_radius, config.vertical_radius);
        let spawn = generator.spawn();
        Self {
            clock: TickClock::new(),
            tick_number: 0,
            connections: Vec::new(),
            next_player_id: 1,
            config,
            pool,
            world: HashMap::new(),
            desired,
            spawn,
            eviction_dirty: true,
            joined_count: 0,
        }
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
                latest_input: player.latest_input,
                input_count: player.input_count,
            })
        })
    }

    /// Cubes the server currently keeps in memory.
    pub fn world_cube_count(&self) -> usize {
        self.world.len()
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

    /// Phase 1: apply player inputs and channel messages. Connections are
    /// drained in join order, each connection's messages in arrival order,
    /// at most [`MESSAGES_PER_CONNECTION_PER_TICK`] per connection.
    fn phase_messages(&mut self) {
        let mut dropped: Vec<usize> = Vec::new();
        for idx in 0..self.connections.len() {
            let mut remove = false;
            for _ in 0..MESSAGES_PER_CONNECTION_PER_TICK {
                let msg = match self.connections[idx].link.try_recv() {
                    Ok(Some(msg)) => msg,
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
            self.connections.remove(idx);
        }
    }

    /// Handle one inbound message for the connection at `idx`. Sends to a
    /// closed link drop the connection instead of panicking.
    fn handle_message(&mut self, idx: usize, msg: ClientMsg) -> Outcome {
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
                let spawn = self.spawn;
                conn.state = ConnectionState::Joined(Player {
                    id: player_id,
                    name,
                    latest_input: None,
                    input_count: 0,
                    center: cube_containing(spawn),
                    sent: HashSet::new(),
                    cursor: 0,
                });
                let welcome = ServerMsg::Welcome {
                    player_id,
                    server_tick: self.tick_number,
                    tick_rate: TICKS_PER_SECOND,
                    // Two blocks over the terrain surface at column (0, 0).
                    spawn,
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
            (
                ConnectionState::Joined(player),
                ClientMsg::PlayerInput {
                    client_tick,
                    position,
                    yaw,
                    pitch,
                },
            ) => {
                player.latest_input = Some(PlayerInput {
                    client_tick,
                    position,
                    yaw,
                    pitch,
                });
                player.input_count += 1;
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
        }
    }

    /// Phase 2: run scheduled block ticks that are due. Empty stub until
    /// block scheduling lands (Stage 2).
    fn phase_scheduled_block_ticks(&mut self) {}

    /// Phase 3: process the block update queue (neighbor changes) to a
    /// per-tick budget. Empty stub until block updates land (Stage 2).
    fn phase_block_update_queue(&mut self) {}

    /// Phase 4: run ECS systems: physics, AI, block entities, mod systems.
    /// Empty stub until bevy_ecs lands (task 1.5+).
    fn phase_ecs_systems(&mut self) {}

    /// Phase 5: recompute dirty light and send outgoing changes — here, the
    /// cube streaming pass (before the `Tick` message every joined player
    /// receives, so cubes always precede their tick) followed by the `Tick`
    /// itself. A player whose link has closed is removed — a failed send
    /// never panics.
    fn phase_light_and_outgoing(&mut self) {
        if self.config.streaming {
            self.phase_streaming();
        }
        let tick_msg = ServerMsg::Tick {
            tick: self.tick_number,
        };
        let mut dropped: Vec<usize> = Vec::new();
        for (idx, conn) in self.connections.iter_mut().enumerate() {
            if conn.joined_player().is_some() && conn.link.send(tick_msg.clone()).is_err() {
                dropped.push(idx);
            }
        }
        for idx in dropped.into_iter().rev() {
            self.connections.remove(idx);
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
        let mut collected = false;
        if self.config.gen_threads == 0 {
            for _ in 0..self.config.gen_budget_per_tick {
                match self.pool.try_recv() {
                    Some((pos, cube)) => {
                        collected = true;
                        self.world.insert(pos, cube);
                    }
                    None => break,
                }
            }
        } else {
            while let Some((pos, cube)) = self.pool.try_recv() {
                collected = true;
                self.world.insert(pos, cube);
            }
        }

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
            self.connections.remove(idx);
        }

        // 3. Eviction: drop world cubes no player's keep set contains.
        //    Regeneration is exact, so dropping is lossless — but task 1.7
        //    must pin edited cubes here or player builds will be lost on
        //    walk-away until persistence lands in 2.3.
        if eviction_dirty {
            let (r, v) = (self.config.view_radius, self.config.vertical_radius);
            let centers: Vec<CubePos> = self
                .connections
                .iter()
                .filter_map(|conn| conn.joined_player().map(|player| player.center))
                .collect();
            self.world
                .retain(|pos, _| centers.iter().any(|&c| in_keep(*pos, c, r, v)));
            self.eviction_dirty = false;
        }
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
    // Center: the latest input position, else the spawn cube.
    let center = player
        .latest_input
        .map(|input| cube_containing(input.position))
        .unwrap_or(player.center);
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

#[cfg(test)]
mod tests {
    use super::{Server, ServerConfig};
    use jbm_core::link::{ClientLink, in_process_pair};
    use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};

    /// Every unit test here asserts exact message sequences, so it runs the
    /// server with streaming off (no CubeData noise).
    fn protocol_server() -> Server {
        Server::with_config(ServerConfig::protocol_only())
    }

    /// A server with one client whose `Hello` is queued but not yet
    /// processed.
    fn pending_hello() -> (Server, ClientLink) {
        let mut server = protocol_server();
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

    fn drain(client: &mut ClientLink) -> Vec<ServerMsg> {
        let mut messages = Vec::new();
        while let Ok(Some(msg)) = client.try_recv() {
            messages.push(msg);
        }
        messages
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
        assert_eq!(players[0].latest_input, None);
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
    fn player_input_is_stored_as_latest_and_counted() {
        let (mut server, mut client) = pending_hello();
        server.tick();
        let _ = drain(&mut client);

        client
            .send(ClientMsg::PlayerInput {
                client_tick: 5,
                position: [1.0, 64.0, 2.0],
                yaw: 90.0,
                pitch: -10.0,
            })
            .unwrap();
        server.tick();
        let players: Vec<_> = server.players().collect();
        assert_eq!(players[0].input_count, 1);
        let input = players[0].latest_input.unwrap();
        assert_eq!(input.client_tick, 5);
        assert_eq!(input.position, [1.0, 64.0, 2.0]);
        assert_eq!(input.yaw, 90.0);
        assert_eq!(input.pitch, -10.0);

        client
            .send(ClientMsg::PlayerInput {
                client_tick: 6,
                position: [3.0, 65.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
            })
            .unwrap();
        server.tick();
        let players: Vec<_> = server.players().collect();
        assert_eq!(players[0].input_count, 2);
        assert_eq!(players[0].latest_input.unwrap().client_tick, 6);
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
