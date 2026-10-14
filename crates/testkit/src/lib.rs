//! Test-only helpers: a headless server harness and fixtures. Never a
//! dependency of shipping code — crates use it as a dev-dependency.

use std::time::Duration;

use jbm_core::link::{ClientLink, LinkClosed};
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, PlayerId, ServerMsg};
use jbm_server::Server;
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
    pub const fn new() -> Self {
        Self {
            server: Server::new(),
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
}

impl TestClient {
    /// Send a message to the server.
    pub fn send(&mut self, msg: ClientMsg) {
        self.link.send(msg).expect("server link still open");
    }

    /// Take every queued server message, remembering the player id from a
    /// `Welcome`. Returns what arrived before the link closed, if it did.
    pub fn drain(&mut self) -> Vec<ServerMsg> {
        let mut messages = Vec::new();
        loop {
            match self.link.try_recv() {
                Ok(Some(msg)) => {
                    if let ServerMsg::Welcome { player_id, .. } = &msg {
                        self.player_id = Some(*player_id);
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
}
