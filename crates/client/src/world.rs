//! The client-side world: cubes received from the server, the dirty set
//! of cubes needing a re-mesh, and per-cube versions. Pure logic, no GPU;
//! the app drives it against a [`jbm_core::protocol::ServerMsg`] stream.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::DVec3;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
use jbm_core::coords::{BlockPos, CubePos};
use jbm_core::cube::Cube;
use jbm_core::physics::{BlockShapes, SolidTable};
use jbm_core::protocol::ServerMsg;

use crate::mesh::greedy::{MeshInput, neighbor_index};

/// What `apply` did with a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldChange {
    /// Nothing that touches cubes.
    None,
    /// A cube was inserted or replaced; it and its loaded neighbors need
    /// a re-mesh.
    Loaded(CubePos),
    /// A cube left the range; the app also drops its GPU mesh.
    Unloaded(CubePos),
}

/// Cubes the client has, plus re-mesh bookkeeping.
///
/// - `CubeData` dirties the cube and its 26 loaded neighbors (a new
///   neighbor changes border faces and AO).
/// - Every dirtying gives the cube a fresh version from one world-wide
///   counter that never resets, so a mesh job built from older data (even
///   one from before an unload and reload) can never match the current
///   version, whatever order the pool finishes jobs in.
/// - `UnloadCube` removes the cube. Neighbors are NOT re-dirtied: their
///   meshes stay valid (a missing neighbor culls faces), and task 2.x
///   revisits this when streaming unloads matter visually.
pub struct ClientWorld {
    cubes: HashMap<CubePos, Arc<Cube>>,
    versions: HashMap<CubePos, u32>,
    dirty: HashSet<CubePos>,
    next_version: u32,
    /// Per-state solidity for collision; `new` builds the builtin table so
    /// plain construction stays usable in tests.
    solid: SolidTable,
}

impl Default for ClientWorld {
    fn default() -> Self {
        ClientWorld::new()
    }
}

impl BlockShapes for ClientWorld {
    fn solid(&self, pos: BlockPos) -> Option<bool> {
        let (cube_pos, local) = pos.split();
        self.cubes
            .get(&cube_pos)
            .map(|cube| self.solid.is_solid(cube.get(local)))
    }
}

impl ClientWorld {
    pub fn new() -> ClientWorld {
        let mut registry = BlockRegistry::new();
        builtin::register(&mut registry);
        ClientWorld {
            cubes: HashMap::new(),
            versions: HashMap::new(),
            dirty: HashSet::new(),
            next_version: 0,
            solid: SolidTable::new(&registry),
        }
    }

    /// Applies one server message. Returns what changed so the app can
    /// drop GPU meshes for unloaded cubes.
    pub fn apply(&mut self, msg: &ServerMsg) -> WorldChange {
        match msg {
            ServerMsg::CubeData { pos, cube } => {
                self.cubes.insert(*pos, Arc::new(cube.as_ref().clone()));
                self.mark_dirty(*pos);
                // Loaded neighbors change border culling and AO.
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        for dx in -1..=1 {
                            if (dx, dy, dz) == (0, 0, 0) {
                                continue;
                            }
                            let neighbor = CubePos {
                                x: pos.x + dx,
                                y: pos.y + dy,
                                z: pos.z + dz,
                            };
                            if self.cubes.contains_key(&neighbor) {
                                self.mark_dirty(neighbor);
                            }
                        }
                    }
                }
                WorldChange::Loaded(*pos)
            }
            ServerMsg::UnloadCube { pos } => {
                self.cubes.remove(pos);
                self.versions.remove(pos);
                self.dirty.remove(pos);
                WorldChange::Unloaded(*pos)
            }
            _ => WorldChange::None,
        }
    }

    /// Marks a cube for re-meshing and gives it a fresh version, which
    /// invalidates any mesh job already in flight for it.
    fn mark_dirty(&mut self, pos: CubePos) {
        self.next_version = self.next_version.wrapping_add(1);
        self.versions.insert(pos, self.next_version);
        self.dirty.insert(pos);
    }

    /// Takes up to `budget` dirty cubes, nearest to `camera` first, and
    /// builds their mesh inputs (center plus the 26 loaded neighbors).
    /// Taken cubes leave the dirty set; results that arrive stale are
    /// dropped by the caller using the returned version.
    pub fn take_dirty(&mut self, budget: usize, camera: DVec3) -> Vec<(CubePos, u32, MeshInput)> {
        if self.dirty.is_empty() {
            return Vec::new();
        }
        let mut order: Vec<(CubePos, f64)> = self
            .dirty
            .iter()
            .map(|&pos| {
                let center = cube_center(pos);
                let d = center - camera;
                (pos, d.length_squared())
            })
            .collect();
        order.sort_by(|a, b| a.1.total_cmp(&b.1));
        let taken: Vec<CubePos> = order.into_iter().take(budget).map(|(pos, _)| pos).collect();
        let mut inputs = Vec::with_capacity(taken.len());
        for pos in taken {
            self.dirty.remove(&pos);
            let Some(center) = self.cubes.get(&pos).cloned() else {
                continue; // unloaded while queued
            };
            let mut neighbors = std::array::from_fn(|_| None);
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for dx in -1..=1 {
                        if (dx, dy, dz) == (0, 0, 0) {
                            continue;
                        }
                        let neighbor = CubePos {
                            x: pos.x + dx,
                            y: pos.y + dy,
                            z: pos.z + dz,
                        };
                        if let Some(cube) = self.cubes.get(&neighbor) {
                            neighbors[neighbor_index(dx, dy, dz)] = Some(Arc::clone(cube));
                        }
                    }
                }
            }
            let version = self.versions.get(&pos).copied().unwrap_or(0);
            inputs.push((pos, version, MeshInput { center, neighbors }));
        }
        inputs
    }

    /// The current version of a cube (0 if not loaded). A mesh result
    /// with a lower version is stale and gets dropped.
    pub fn version(&self, pos: CubePos) -> Option<u32> {
        self.versions.get(&pos).copied()
    }

    /// Number of loaded cubes.
    pub fn cube_count(&self) -> usize {
        self.cubes.len()
    }

    /// Cubes still waiting for a (re-)mesh.
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    pub fn cube(&self, pos: CubePos) -> Option<&Arc<Cube>> {
        self.cubes.get(&pos)
    }

    /// Test hook: removes a single block from the client's copy of the
    /// world only — the server never learns of it. Prediction tests use it
    /// to make the client's collision world disagree with the server's.
    #[cfg(feature = "testing")]
    pub fn remove_block(&mut self, pos: BlockPos) {
        let (cube_pos, local) = pos.split();
        let Some(cube) = self.cubes.get(&cube_pos) else {
            return;
        };
        let mut replacement = cube.as_ref().clone();
        replacement.set(local, BlockStateId::AIR);
        self.cubes.insert(cube_pos, Arc::new(replacement));
        self.mark_dirty(cube_pos);
    }
}

/// World-space center of a cube, for distance ordering.
fn cube_center(pos: CubePos) -> DVec3 {
    DVec3::new(
        f64::from(pos.x) * 32.0 + 16.0,
        f64::from(pos.y) * 32.0 + 16.0,
        f64::from(pos.z) * 32.0 + 16.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jbm_core::block::{BlockRegistry, builtin};
    use jbm_core::protocol::PlayerId;

    fn stone_cube() -> Box<Cube> {
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        Box::new(Cube::filled(blocks.stone))
    }

    fn cube_data(pos: CubePos) -> ServerMsg {
        ServerMsg::CubeData {
            pos,
            cube: stone_cube(),
        }
    }

    #[test]
    fn a_neighbor_arriving_redirties_the_cubes_next_to_it() {
        let mut world = ClientWorld::new();
        let origin = CubePos { x: 0, y: 0, z: 0 };
        world.apply(&cube_data(origin));
        assert_eq!(world.dirty_count(), 1);
        let _ = world.take_dirty(10, DVec3::ZERO);
        assert_eq!(world.dirty_count(), 0);

        // A face neighbor arriving must re-dirty the origin.
        let neighbor = CubePos { x: 1, y: 0, z: 0 };
        world.apply(&cube_data(neighbor));
        assert_eq!(world.dirty_count(), 2, "origin and the new cube");
        let dirty = world.take_dirty(10, DVec3::ZERO);
        let positions: Vec<CubePos> = dirty.iter().map(|(p, _, _)| *p).collect();
        assert!(positions.contains(&origin));
        assert!(positions.contains(&neighbor));
    }

    #[test]
    fn unload_removes_the_cube() {
        let mut world = ClientWorld::new();
        let pos = CubePos { x: 3, y: 1, z: 2 };
        assert_eq!(world.apply(&cube_data(pos)), WorldChange::Loaded(pos));
        assert_eq!(world.cube_count(), 1);
        assert_eq!(
            world.apply(&ServerMsg::UnloadCube { pos }),
            WorldChange::Unloaded(pos)
        );
        assert_eq!(world.cube_count(), 0);
        assert_eq!(world.version(pos), None);
        assert_eq!(world.dirty_count(), 0, "unloaded cube leaves the dirty set");
    }

    #[test]
    fn versions_increase_per_cube() {
        let mut world = ClientWorld::new();
        let pos = CubePos { x: 0, y: 0, z: 0 };
        world.apply(&cube_data(pos));
        assert_eq!(world.version(pos), Some(1));
        world.apply(&cube_data(pos));
        assert_eq!(world.version(pos), Some(2));
        // Other messages leave versions alone.
        world.apply(&ServerMsg::Tick { tick: 5 });
        assert_eq!(world.version(pos), Some(2));
    }

    #[test]
    fn neighbor_arrival_invalidates_in_flight_mesh() {
        let mut world = ClientWorld::new();
        let a = CubePos { x: 0, y: 0, z: 0 };
        let b = CubePos { x: 1, y: 0, z: 0 };
        world.apply(&cube_data(a));
        let (_, v_old, _) = world.take_dirty(1, DVec3::ZERO).remove(0);
        // `b` arrives while `a`'s first mesh job is still in flight.
        world.apply(&cube_data(b));
        let v_now = world.version(a).unwrap();
        assert_ne!(v_old, v_now, "the job built without b must be stale");
    }

    #[test]
    fn reload_never_reuses_a_version() {
        let mut world = ClientWorld::new();
        let pos = CubePos { x: 0, y: 0, z: 0 };
        world.apply(&cube_data(pos));
        let v_before = world.version(pos).unwrap();
        world.apply(&ServerMsg::UnloadCube { pos });
        world.apply(&cube_data(pos));
        assert!(world.version(pos).unwrap() > v_before);
    }

    #[test]
    fn take_dirty_orders_nearest_first_and_respects_budget() {
        let mut world = ClientWorld::new();
        let near = CubePos { x: 0, y: 0, z: 0 };
        let mid = CubePos { x: 5, y: 0, z: 0 };
        let far = CubePos { x: 40, y: 0, z: 0 };
        for pos in [far, mid, near] {
            world.apply(&cube_data(pos));
        }
        let camera = DVec3::new(16.0, 16.0, 16.0);
        let taken = world.take_dirty(10, camera);
        let order: Vec<CubePos> = taken.iter().map(|(p, _, _)| *p).collect();
        assert_eq!(order, vec![near, mid, far], "nearest first");
        assert_eq!(world.dirty_count(), 0);

        // Budget: re-dirty everything, take only one.
        for pos in [far, mid, near] {
            world.apply(&cube_data(pos));
        }
        let taken = world.take_dirty(1, camera);
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].0, near);
        assert_eq!(world.dirty_count(), 2);
    }

    #[test]
    fn mesh_inputs_carry_loaded_neighbors() {
        let mut world = ClientWorld::new();
        let a = CubePos { x: 0, y: 0, z: 0 };
        let b = CubePos { x: 1, y: 0, z: 0 };
        world.apply(&cube_data(a));
        world.apply(&cube_data(b));
        let taken = world.take_dirty(10, DVec3::ZERO);
        assert_eq!(taken.len(), 2);
        for (pos, _, input) in &taken {
            if *pos == a {
                assert!(input.neighbor(1, 0, 0).is_some(), "+X neighbor loaded");
                assert!(input.neighbor(-1, 0, 0).is_none(), "-X neighbor missing");
                assert!(
                    input.neighbor(1, 1, 0).is_none(),
                    "diagonal neighbor missing"
                );
            } else {
                assert!(input.neighbor(-1, 0, 0).is_some(), "-X neighbor loaded");
                assert!(input.neighbor(1, 0, 0).is_none());
            }
        }
    }

    #[test]
    fn unrelated_messages_change_nothing() {
        let mut world = ClientWorld::new();
        assert_eq!(
            world.apply(&ServerMsg::Welcome {
                player_id: PlayerId(1),
                server_tick: 0,
                tick_rate: 20,
                spawn: [0.0, 64.0, 0.0],
            }),
            WorldChange::None
        );
        assert_eq!(
            world.apply(&ServerMsg::Pong {
                nonce: 1,
                server_tick: 1
            }),
            WorldChange::None
        );
        assert_eq!(world.cube_count(), 0);
    }

    #[test]
    fn block_shapes_reads_loaded_air_and_missing() {
        use jbm_core::physics::BlockShapes;

        let mut world = ClientWorld::new();
        let origin = CubePos { x: 0, y: 0, z: 0 };
        // A stone floor cube below an air cube above it.
        world.apply(&ServerMsg::CubeData {
            pos: origin,
            cube: stone_cube(),
        });
        world.apply(&ServerMsg::CubeData {
            pos: CubePos { x: 0, y: 1, z: 0 },
            cube: Box::new(Cube::default()),
        });
        let solid = |x: i64, y: i64, z: i64| world.solid(BlockPos { x, y, z });
        assert_eq!(solid(3, 2, 7), Some(true), "stone inside the floor cube");
        assert_eq!(solid(0, 32, 0), Some(false), "air in the cube above");
        assert_eq!(solid(0, 64, 0), None, "an unloaded cube is None");
    }

    #[test]
    #[cfg(feature = "testing")]
    fn remove_block_guts_only_the_client_copy() {
        use jbm_core::physics::BlockShapes;

        let mut world = ClientWorld::new();
        let origin = CubePos { x: 0, y: 0, z: 0 };
        world.apply(&cube_data(origin));
        let at = BlockPos { x: 3, y: 2, z: 7 };
        assert_eq!(world.solid(at), Some(true));
        world.remove_block(at);
        assert_eq!(world.solid(at), Some(false), "the block is gone");
        assert_eq!(
            world.solid(BlockPos { x: 4, y: 2, z: 7 }),
            Some(true),
            "its neighbors stay"
        );
        assert_eq!(world.dirty_count(), 1, "the cube re-meshes");
    }
}
