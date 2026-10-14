//! Protocol messages exchanged between client and server over a
//! [`crate::link::Link`].
//!
//! Rules for every message type here:
//!
//! - Messages are plain owned data: no `Arc`, no references, no trait
//!   objects. They must stay serializable later (a network transport will
//!   reuse them unchanged).
//! - All messages derive `Debug, Clone, PartialEq` so tests can assert on
//!   them directly.
//! - No serde yet; serialization arrives with the network transport.

use crate::coords::CubePos;
use crate::cube::Cube;

/// Version of the client/server protocol. A `Hello` carrying any other
/// version is rejected.
pub const PROTOCOL_VERSION: u32 = 2;

/// Identifies a joined player inside one server. Ids start at 1 and count up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerId(pub u32);

/// Messages the client sends to the server.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientMsg {
    /// First message on a fresh connection. A matching
    /// [`PROTOCOL_VERSION`] joins the player; anything else is rejected.
    Hello { protocol_version: u32, name: String },
    /// The client's latest movement state. Placeholder shape; task 1.6
    /// replaces it with real movement input.
    PlayerInput {
        client_tick: u64,
        position: [f64; 3],
        yaw: f32,
        pitch: f32,
    },
    /// Round-trip latency probe; answered by [`ServerMsg::Pong`].
    Ping { nonce: u64 },
    /// Polite goodbye; the server removes the player on the next tick.
    Disconnect,
}

/// Messages the server sends to clients.
#[derive(Debug, Clone, PartialEq)]
pub enum ServerMsg {
    /// A `Hello` was accepted; the connection is now a player.
    Welcome {
        player_id: PlayerId,
        server_tick: u64,
        tick_rate: u32,
        /// Where the player starts, in world block coordinates.
        spawn: [f64; 3],
    },
    /// A `Hello` was refused; the connection is dropped right after.
    Rejected { reason: String },
    /// Sent to every joined player at the end of each tick.
    Tick { tick: u64 },
    /// Answer to [`ClientMsg::Ping`], stamped with the current tick.
    Pong { nonce: u64, server_tick: u64 },
    /// The server dropped this connection.
    Disconnected { reason: String },
    /// A cube the player should now have loaded: newly in range, or
    /// replaced wholesale. Uniform cubes (all air, all stone) are cheap,
    /// so they are sent too: "loaded and empty" differs from "not loaded".
    /// Boxed so the enum stays small for the common small messages.
    CubeData { pos: CubePos, cube: Box<Cube> },
    /// The cube left the player's range; the client drops it and its mesh.
    UnloadCube { pos: CubePos },
}
