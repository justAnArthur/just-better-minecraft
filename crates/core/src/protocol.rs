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

use crate::block::BlockStateId;
use crate::coords::{BlockPos, CubePos};
use crate::cube::Cube;
use crate::lod::LodTops;
use crate::physics::{MoveInput, PlayerState};

/// Version of the client/server protocol. A `Hello` carrying any other
/// version is rejected.
pub const PROTOCOL_VERSION: u32 = 7;

/// Identifies a joined player inside one server. Ids start at 1 and count up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerId(pub u32);

/// Messages the client sends to the server.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientMsg {
    /// First message on a fresh connection. A matching
    /// [`PROTOCOL_VERSION`] joins the player; anything else is rejected.
    Hello { protocol_version: u32, name: String },
    /// One tick of movement input. `seq` starts at 1 and increments by 1
    /// per client tick; the server ignores seqs it has already processed.
    /// Pitch is display-only (and later use); physics ignores it.
    PlayerInput {
        seq: u64,
        input: MoveInput,
        pitch: f32,
    },
    /// Round-trip latency probe; answered by [`ServerMsg::Pong`].
    Ping { nonce: u64 },
    /// Break the block at `pos` (task 1.7). Sent after the client's
    /// optimistic edit; the server validates and reverts rejections.
    BreakBlock { pos: BlockPos },
    /// Place `state` into the air cell at `pos`. `pos` is the cell being
    /// filled — the client computes it as the hit block plus the hit
    /// face's normal.
    PlaceBlock { pos: BlockPos, state: BlockStateId },
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
        /// The spawn position: the player's feet and initial state —
        /// velocity 0, not on ground. A returning player gets where they
        /// last stood (persistence), a new one the world spawn.
        spawn: [f64; 3],
        /// Initial look direction (radians), as the player last sent it;
        /// 0 for a new player.
        yaw: f32,
        pitch: f32,
        /// Whether the player starts flying.
        flying: bool,
    },
    /// The authoritative player state after the server processed input
    /// `ack_seq`. Sent in phase 5 before `Tick`, only in ticks where at
    /// least one input was processed.
    PlayerState { ack_seq: u64, state: PlayerState },
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
    /// The authoritative state of one block (task 1.7): an accepted edit
    /// broadcast to every player with the cube loaded, or the revert of a
    /// rejected edit sent to its sender only. Sent in phase 5 before the
    /// streaming pass and the `Tick`, so edits land within one tick.
    BlockChange { pos: BlockPos, state: BlockStateId },
    /// A LOD cube (task 2.6) the player should now have: newly in one of
    /// their LOD rings, or changed (a re-send replaces the client's copy
    /// wholesale). `level` is 1-based; a level-`n` cube at `pos` covers
    /// blocks `pos * CUBE_SIZE * 2^n` onwards, each cell `2^n` blocks per
    /// side. Uniform cubes are sent too, as for `CubeData`. `tops` are
    /// the cube's column tops (v7, task 2.14): the fill height of each
    /// column's topmost filled cell, which the client's mesher puts the
    /// coarse surface at.
    LodCubeData {
        level: u8,
        pos: CubePos,
        cube: Box<Cube>,
        tops: LodTops,
    },
    /// The LOD cube left the player's rings; the client drops it and its
    /// mesh.
    UnloadLodCube { level: u8, pos: CubePos },
}
