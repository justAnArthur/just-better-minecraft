//! The authoritative server: connection handling and the fixed-order tick
//! phases from the design doc's "Simulation" section. Clients are talked to
//! only through [`ServerLink`]s.

use std::time::Duration;

use jbm_core::link::{LinkClosed, ServerLink};
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, PlayerId, ServerMsg};

use crate::tick::{TICKS_PER_SECOND, TickClock};

/// Most inbound messages one connection may deliver in a single tick.
/// Leftovers wait for the next tick, so a flooding client cannot stall the
/// tick.
const MESSAGES_PER_CONNECTION_PER_TICK: usize = 256;

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
}

impl Server {
    pub const fn new() -> Self {
        Self {
            clock: TickClock::new(),
            tick_number: 0,
            connections: Vec::new(),
            next_player_id: 1,
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
                conn.state = ConnectionState::Joined(Player {
                    id: player_id,
                    name,
                    latest_input: None,
                    input_count: 0,
                });
                let welcome = ServerMsg::Welcome {
                    player_id,
                    server_tick: self.tick_number,
                    tick_rate: TICKS_PER_SECOND,
                    // Placeholder until terrain exists (task 1.5).
                    spawn: [0.0, 80.0, 0.0],
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

    /// Phase 5: recompute dirty light and send outgoing changes. The only
    /// outgoing change so far is the `Tick` message every joined player
    /// receives. A player whose link has closed is removed here — a failed
    /// send never panics.
    fn phase_light_and_outgoing(&mut self) {
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
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Server;
    use jbm_core::link::{ClientLink, in_process_pair};
    use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};

    /// A server with one client whose `Hello` is queued but not yet
    /// processed.
    fn pending_hello() -> (Server, ClientLink) {
        let mut server = Server::new();
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
        let mut server = Server::new();
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
        let mut server = Server::new();
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
        let mut server = Server::new();
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
        let mut server = Server::new();
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
