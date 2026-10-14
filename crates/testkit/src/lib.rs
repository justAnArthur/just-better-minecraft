//! Test-only helpers: a headless server harness and fixtures. Never a
//! dependency of shipping code — crates use it as a dev-dependency.

use std::collections::HashSet;
use std::time::Duration;

use jbm_core::coords::CubePos;
use jbm_core::link::{ClientLink, LinkClosed};
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, PlayerId, ServerMsg};
use jbm_server::Server;
use jbm_server::server::ServerConfig;
use jbm_server::tick::TICK_PERIOD;

/// A headless server driven by a manual, deterministic time source.
///
/// Wraps a real [`Server`] and steps it tick by tick. Connect clients with
/// [`HeadlessServer::connect`] and assert on server state through
/// [`HeadlessServer::server`].
pub struct HeadlessServer {
    server: Server,
}

impl HeadlessServer {
    /// A protocol-only server (no cube streaming), so tests that assert
    /// exact message sequences see only handshake/ping/tick traffic.
    pub fn new() -> Self {
        Self::with_config(ServerConfig::protocol_only())
    }

    /// A headless server with an explicit config — e.g. streaming enabled
    /// with small radii and inline generation for deterministic cube tests.
    pub fn with_config(config: ServerConfig) -> Self {
        Self {
            server: Server::with_config(config),
        }
    }

    /// Advance exactly `ticks` simulation ticks.
    pub fn step(&mut self, ticks: u32) {
        for _ in 0..ticks {
            // One period per update call, so the catch-up cap never bites.
            self.server.update(TICK_PERIOD);
        }
    }

    /// Let `d` of simulated time pass, delivered to the server in
    /// tick-sized slices like a well-behaved frame loop would; the
    /// trailing sub-tick remainder is kept as credit towards the next
    /// tick.
    pub fn run_for(&mut self, d: Duration) {
        let mut remaining = d;
        while remaining >= TICK_PERIOD {
            self.server.update(TICK_PERIOD);
            remaining -= TICK_PERIOD;
        }
        if !remaining.is_zero() {
            self.server.update(remaining);
        }
    }

    /// Ticks simulated so far.
    pub fn tick(&self) -> u64 {
        self.server.tick_count()
    }

    /// The underlying server, for assertions.
    pub fn server(&self) -> &Server {
        &self.server
    }

    /// Step one tick at a time (draining every client each tick so their
    /// [`TestClient::loaded`] sets stay current) until `cond` holds, or
    /// `max_ticks` ticks have run. Returns whether the condition was met.
    /// Never loops unboundedly.
    pub fn step_until(
        &mut self,
        clients: &mut [&mut TestClient],
        mut cond: impl FnMut(&Server, &[&TestClient]) -> bool,
        max_ticks: u32,
    ) -> bool {
        for _ in 0..max_ticks {
            self.step(1);
            for client in clients.iter_mut() {
                client.drain();
            }
            let snapshot: Vec<&TestClient> = clients.iter().map(|c| &**c).collect();
            if cond(&self.server, &snapshot) {
                return true;
            }
        }
        false
    }

    /// Create an in-process client, hand the server end to the server and
    /// send `Hello`. The client is welcomed on the next
    /// [`step`](Self::step).
    pub fn connect(&mut self, name: &str) -> TestClient {
        let mut client = self.connect_raw();
        client.send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: name.to_owned(),
        });
        client
    }

    /// Like [`connect`](Self::connect), but without the automatic `Hello`
    /// — for protocol-level tests that drive the handshake themselves.
    pub fn connect_raw(&mut self) -> TestClient {
        let (client, server_end) = jbm_core::link::in_process_pair();
        self.server.add_connection(server_end);
        TestClient {
            link: client,
            player_id: None,
            loaded: HashSet::new(),
            cube_data_count: 0,
            unload_count: 0,
        }
    }
}

impl Default for HeadlessServer {
    fn default() -> Self {
        Self::new()
    }
}

/// A client end connected to a [`HeadlessServer`].
pub struct TestClient {
    link: ClientLink,
    player_id: Option<PlayerId>,
    /// Cubes this client currently has loaded, maintained while draining.
    loaded: HashSet<CubePos>,
    cube_data_count: u64,
    unload_count: u64,
}

impl TestClient {
    /// Send a message to the server.
    pub fn send(&mut self, msg: ClientMsg) {
        self.link.send(msg).expect("server link still open");
    }

    /// Take every queued server message, remembering the player id from a
    /// `Welcome` and maintaining the loaded-cube set from `CubeData` and
    /// `UnloadCube`. A duplicate `CubeData` for a cube that is already
    /// loaded, or an `UnloadCube` for one that is not, fails the test right
    /// here with the position. Returns what arrived before the link closed,
    /// if it did.
    pub fn drain(&mut self) -> Vec<ServerMsg> {
        let mut messages = Vec::new();
        loop {
            match self.link.try_recv() {
                Ok(Some(msg)) => {
                    match &msg {
                        ServerMsg::Welcome { player_id, .. } => {
                            self.player_id = Some(*player_id);
                        }
                        ServerMsg::CubeData { pos, .. } => {
                            if !self.loaded.insert(*pos) {
                                panic!("duplicate CubeData for already-loaded cube {pos:?}");
                            }
                            self.cube_data_count += 1;
                        }
                        ServerMsg::UnloadCube { pos } => {
                            if !self.loaded.remove(pos) {
                                panic!("UnloadCube for cube that is not loaded: {pos:?}");
                            }
                            self.unload_count += 1;
                        }
                        _ => {}
                    }
                    messages.push(msg);
                }
                Ok(None) => break,
                Err(LinkClosed) => break,
            }
        }
        messages
    }

    /// The player id learned from a drained `Welcome`, if any yet.
    pub fn player_id(&self) -> Option<PlayerId> {
        self.player_id
    }

    /// The set of cubes this client currently has loaded, as tracked from
    /// `CubeData`/`UnloadCube` while draining.
    pub fn loaded(&self) -> &HashSet<CubePos> {
        &self.loaded
    }

    /// How many `CubeData` messages have been received in total.
    pub fn cube_data_count(&self) -> u64 {
        self.cube_data_count
    }

    /// How many `UnloadCube` messages have been received in total — zero is
    /// the hysteresis assertion for a player oscillating on a border.
    pub fn unload_count(&self) -> u64 {
        self.unload_count
    }
}
