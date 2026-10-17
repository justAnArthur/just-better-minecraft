//! Block edits: validation of `BreakBlock`/`PlaceBlock` against the
//! authoritative state, applying accepted edits (world, LOD propagation,
//! pinning) and queueing reverts for rejections.

use glam::DVec3;
use jbm_core::block::BlockStateId;
use jbm_core::coords::BlockPos;
use jbm_core::physics;
use jbm_core::protocol::PlayerId;
use jbm_core::raycast::REACH;

use super::{Outcome, Server};

/// Slack on the reach check, in blocks, so a player whose client-side ray
/// was within `REACH` when they clicked still passes validation a round
/// trip later.
const REACH_MARGIN: f64 = 1.0;

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

impl Server {
    /// A `BreakBlock` from the joined player at `idx`. Validation: the
    /// cube is loaded, the block center is within `REACH + 1` of the
    /// player's eye (the margin covers latency), and the block is not
    /// air. Accepted edits set the block, pin the cube and queue a
    /// broadcast; rejections queue the authoritative state back to the
    /// sender only. Rejections never disconnect.
    pub(super) fn handle_break(&mut self, idx: usize, pos: BlockPos) -> Outcome {
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
    pub(super) fn handle_place(
        &mut self,
        idx: usize,
        pos: BlockPos,
        state: BlockStateId,
    ) -> Outcome {
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
        let id: PlayerId = self.connections[idx]
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
            .filter_map(|conn| conn.joined_player())
            .any(|player| {
                let aabb = player.state.aabb();
                physics::block_overlaps_box(pos, aabb.min, aabb.max)
            })
    }
}

#[cfg(test)]
mod tests {
    use jbm_core::block::{BlockStateId, builtin::Blocks};
    use jbm_core::coords::BlockPos;
    use jbm_core::protocol::{ClientMsg, ServerMsg};

    use crate::server::test_util::{
        drain, drain_edits, physics_server, surface_block, two_settled_players,
    };

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
                drain_edits(client).as_slice(),
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
            drain_edits(&mut a).as_slice(),
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
                    drain_edits(&mut a).as_slice(),
                    [
                        ServerMsg::BlockChange { state, .. },
                        ServerMsg::Tick { .. }
                    ] if *state == revert_state
                ),
                "sender must be reverted for {msg:?}"
            );
            assert_eq!(
                drain_edits(&mut b).as_slice(),
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
            drain_edits(&mut a).as_slice(),
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
        let (mut client, server_end) = jbm_core::link::in_process_pair();
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
}
