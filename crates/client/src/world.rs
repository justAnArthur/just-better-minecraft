//! The client-side world: cubes received from the server, the dirty set
//! of cubes needing a re-mesh, and per-cube versions. Pure logic, no GPU;
//! the app drives it against a [`jbm_core::protocol::ServerMsg`] stream.
//!
//! LOD ring cubes (task 2.6) live beside the LOD0 cubes, keyed by
//! [`CubeKey`] so meshes, versions and dirt share one bookkeeping at every
//! level. Mesh inputs carry the 26 **same-level** neighbors.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::DVec3;
use jbm_core::block::{BlockRegistry, builtin};
use jbm_core::coords::{BlockPos, CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;
use jbm_core::physics::{BlockShapes, SolidTable};
use jbm_core::protocol::ServerMsg;

use crate::lod::CubeKey;
use crate::mesh::{MeshInput, neighbor_index};

/// What `apply` did with a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldChange {
    /// Nothing that touches cubes.
    None,
    /// A cube was inserted or replaced (any level); it and its loaded
    /// same-level neighbors need a re-mesh.
    Loaded(CubeKey),
    /// A cube left the range (any level); the app also drops its GPU mesh
    /// and coverage readiness.
    Unloaded(CubeKey),
    /// One block changed (`BlockChange`, or an optimistic edit): the cube
    /// and its border neighbors were dirtied with edit priority. The
    /// position is the block's, not the cube's.
    Edited(BlockPos),
}

/// Cubes the client has, plus re-mesh bookkeeping.
///
/// - `CubeData` and `LodCubeData` dirty the cube and its 26 loaded
///   same-level neighbors (a new neighbor changes border faces and AO).
/// - A block edit (`BlockChange` or an optimistic `set_block`) dirties the
///   block's cube plus every loaded neighbor cube the block touches
///   through faces, edges or corners (see [`ClientWorld::set_block`]) and
///   gives the edited cube priority: [`ClientWorld::take_dirty`] takes
///   priority cubes first, so edits re-mesh promptly even while streaming
///   keeps the queue busy.
/// - Every dirtying gives the cube a fresh version from one world-wide
///   counter that never resets, so a mesh job built from older data (even
///   one from before an unload and reload) can never match the current
///   version, whatever order the pool finishes jobs in.
/// - `UnloadCube` / `UnloadLodCube` remove the cube. Neighbors are NOT
///   re-dirtied: their meshes stay valid (a missing neighbor culls faces).
pub struct ClientWorld {
    /// Every loaded cube keyed by level and position.
    cubes: HashMap<CubeKey, Arc<Cube>>,
    versions: HashMap<CubeKey, u32>,
    dirty: HashSet<CubeKey>,
    /// Dirty cubes taken first by `take_dirty`: those holding a recent
    /// edit. Always a subset of `dirty`.
    priority: HashSet<CubeKey>,
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
            .get(&CubeKey::lod0(cube_pos))
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
            priority: HashSet::new(),
            next_version: 0,
            solid: SolidTable::new(&registry),
        }
    }

    /// Applies one server message. Returns what changed so the app can
    /// drop GPU meshes for unloaded cubes.
    pub fn apply(&mut self, msg: &ServerMsg) -> WorldChange {
        match msg {
            ServerMsg::CubeData { pos, cube } => {
                self.load(CubeKey::lod0(*pos), cube);
                WorldChange::Loaded(CubeKey::lod0(*pos))
            }
            ServerMsg::LodCubeData { level, pos, cube } => {
                let key = CubeKey {
                    level: *level,
                    pos: *pos,
                };
                self.load(key, cube);
                WorldChange::Loaded(key)
            }
            ServerMsg::BlockChange { pos, state } => {
                if self.set_block(*pos, *state) {
                    WorldChange::Edited(*pos)
                } else {
                    // The cube is not loaded: nothing to update. A later
                    // `CubeData` for it already contains the edit.
                    WorldChange::None
                }
            }
            ServerMsg::UnloadCube { pos } => {
                self.unload(CubeKey::lod0(*pos));
                WorldChange::Unloaded(CubeKey::lod0(*pos))
            }
            ServerMsg::UnloadLodCube { level, pos } => {
                let key = CubeKey {
                    level: *level,
                    pos: *pos,
                };
                self.unload(key);
                WorldChange::Unloaded(key)
            }
            _ => WorldChange::None,
        }
    }

    /// Inserts or replaces a cube at any level and dirties it plus its 26
    /// loaded same-level neighbors.
    fn load(&mut self, key: CubeKey, cube: &Cube) {
        self.cubes.insert(key, Arc::new(cube.clone()));
        self.mark_dirty(key);
        // Loaded neighbors change border culling and AO.
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if (dx, dy, dz) == (0, 0, 0) {
                        continue;
                    }
                    let neighbor = CubeKey {
                        level: key.level,
                        pos: CubePos {
                            x: key.pos.x + dx,
                            y: key.pos.y + dy,
                            z: key.pos.z + dz,
                        },
                    };
                    if self.cubes.contains_key(&neighbor) {
                        self.mark_dirty(neighbor);
                    }
                }
            }
        }
    }

    fn unload(&mut self, key: CubeKey) {
        self.cubes.remove(&key);
        self.versions.remove(&key);
        self.dirty.remove(&key);
        self.priority.remove(&key);
    }

    /// Marks a cube for re-meshing and gives it a fresh version, which
    /// invalidates any mesh job already in flight for it.
    fn mark_dirty(&mut self, key: CubeKey) {
        self.next_version = self.next_version.wrapping_add(1);
        self.versions.insert(key, self.next_version);
        self.dirty.insert(key);
    }

    /// [`mark_dirty`] plus edit priority: `take_dirty` takes this cube
    /// before any streaming-dirty cube.
    fn mark_dirty_priority(&mut self, key: CubeKey) {
        self.mark_dirty(key);
        self.priority.insert(key);
    }

    /// Sets one block (an optimistic edit, or an authoritative
    /// `BlockChange`). Returns `false` when the block's cube is not
    /// loaded — the edit is then ignored, and a later `CubeData` for the
    /// cube carries the authoritative content anyway.
    ///
    /// Dirties the block's cube plus every loaded neighbor cube the block
    /// touches through faces, edges or corners: a block at local x == 0
    /// dirties the -X neighbor, one at x == CUBE_SIZE - 1 the +X neighbor,
    /// and so on per axis, up to 7 neighbors for a corner block (a border
    /// block changes the neighbor's face culling and AO). Only loaded
    /// neighbors are dirtied, all at edit priority.
    ///
    /// Setting a block to the state it already has dirties nothing: the
    /// server's echo of an accepted optimistic edit costs no re-mesh.
    pub fn set_block(&mut self, pos: BlockPos, state: jbm_core::block::BlockStateId) -> bool {
        let (cube_pos, local) = pos.split();
        let key = CubeKey::lod0(cube_pos);
        let Some(cube) = self.cubes.get_mut(&key) else {
            return false;
        };
        if cube.get(local) == state {
            return true;
        }
        // Copies only when a mesh job still holds the old cube.
        Arc::make_mut(cube).set(local, state);
        self.dirty_block(cube_pos, local);
        true
    }

    /// The edit dirty rule from [`ClientWorld::set_block`], on an
    /// already-split position. The edited cube gets priority; its dirtied
    /// neighbors do not (they re-mesh right behind it in the normal
    /// nearest-first order).
    fn dirty_block(&mut self, cube_pos: CubePos, local: jbm_core::coords::LocalPos) {
        self.mark_dirty_priority(CubeKey::lod0(cube_pos));
        // Offsets to reach on each axis: none, or -1 / +1 on a border.
        let span = |v: u8| -> ([i32; 1], usize) {
            if v == 0 {
                ([-1], 1)
            } else if v == (CUBE_SIZE - 1) as u8 {
                ([1], 1)
            } else {
                ([0], 0)
            }
        };
        let ([dx], nx) = span(local.x);
        let ([dy], ny) = span(local.y);
        let ([dz], nz) = span(local.z);
        for i in 0..=nx {
            for j in 0..=ny {
                for k in 0..=nz {
                    if i + j + k == 0 {
                        continue; // the edited cube itself
                    }
                    let neighbor = CubePos {
                        x: cube_pos.x + if i == 0 { 0 } else { dx },
                        y: cube_pos.y + if j == 0 { 0 } else { dy },
                        z: cube_pos.z + if k == 0 { 0 } else { dz },
                    };
                    // Priority too: a border edit opens or closes faces
                    // in the neighbor's mesh, and a late neighbor shows a
                    // hole into the world.
                    if self.cubes.contains_key(&CubeKey::lod0(neighbor)) {
                        self.mark_dirty_priority(CubeKey::lod0(neighbor));
                    }
                }
            }
        }
    }

    /// Takes up to `budget` dirty cubes and builds their mesh inputs
    /// (center plus the 26 loaded same-level neighbors). Taken cubes leave
    /// the dirty set; results that arrive stale are dropped by the caller
    /// using the returned version.
    ///
    /// Order (task 2.6): LOD0 edit-priority cubes first, nearest to
    /// `camera`; then LOD0 streaming dirt, nearest first; then LOD ring
    /// cubes, nearest first by distance in blocks divided by 2^level.
    /// **Budget split:** when both classes have dirt, LOD cubes keep a
    /// guaranteed quarter of the budget (minimum 1 of any budget ≥ 1), so
    /// neither class starves under a saturated queue — LOD0 keeps the
    /// other three quarters, and a class with nothing dirty yields its
    /// whole share to the other.
    pub fn take_dirty(&mut self, budget: usize, camera: DVec3) -> Vec<(CubeKey, u32, MeshInput)> {
        if self.dirty.is_empty() || budget == 0 {
            return Vec::new();
        }
        let distance = |key: CubeKey| (cube_center(key) - camera).length_squared();
        let lod_dirt: Vec<CubeKey> = self
            .dirty
            .iter()
            .copied()
            .filter(|key| key.level > 0)
            .collect();
        let lod_quota = if lod_dirt.is_empty() {
            0
        } else {
            (budget / 4).max(1)
        };
        let lod0_budget = budget - lod_quota.min(budget);

        let mut ordered: Vec<CubeKey> = self.priority.iter().copied().collect();
        ordered.sort_by(|a, b| distance(*a).total_cmp(&distance(*b)));
        let mut lod0_rest: Vec<CubeKey> = self
            .dirty
            .iter()
            .copied()
            .filter(|key| key.level == 0 && !self.priority.contains(key))
            .collect();
        lod0_rest.sort_by(|a, b| distance(*a).total_cmp(&distance(*b)));
        ordered.extend(lod0_rest);
        ordered.truncate(lod0_budget);

        let mut lod_ordered = lod_dirt;
        lod_ordered.sort_by(|a, b| distance(*a).total_cmp(&distance(*b)));
        ordered.extend(lod_ordered);
        ordered.truncate(budget);

        let taken: Vec<CubeKey> = ordered;
        for key in &taken {
            self.dirty.remove(key);
            self.priority.remove(key);
        }
        let mut inputs = Vec::with_capacity(taken.len());
        for key in taken {
            let Some(center) = self.cubes.get(&key).cloned() else {
                continue; // unloaded while queued
            };
            let mut neighbors = std::array::from_fn(|_| None);
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for dx in -1..=1 {
                        if (dx, dy, dz) == (0, 0, 0) {
                            continue;
                        }
                        let neighbor = CubeKey {
                            level: key.level,
                            pos: CubePos {
                                x: key.pos.x + dx,
                                y: key.pos.y + dy,
                                z: key.pos.z + dz,
                            },
                        };
                        if let Some(cube) = self.cubes.get(&neighbor) {
                            neighbors[neighbor_index(dx, dy, dz)] = Some(Arc::clone(cube));
                        }
                    }
                }
            }
            let version = self.versions.get(&key).copied().unwrap_or(0);
            inputs.push((
                key,
                version,
                MeshInput {
                    center,
                    neighbors,
                    level: key.level,
                },
            ));
        }
        inputs
    }

    /// The current version of a cube (0 if not loaded). A mesh result
    /// with a lower version is stale and gets dropped.
    pub fn version(&self, key: CubeKey) -> Option<u32> {
        self.versions.get(&key).copied()
    }

    /// Number of loaded LOD0 cubes.
    pub fn cube_count(&self) -> usize {
        self.cubes.keys().filter(|key| key.level == 0).count()
    }

    /// Loaded cubes at LOD `level` (1-based; 0 = LOD0).
    pub fn lod_cube_count(&self, level: u8) -> usize {
        self.cubes.keys().filter(|key| key.level == level).count()
    }

    /// Highest LOD level with any loaded cube (0 when none).
    pub fn max_lod_level(&self) -> u8 {
        self.cubes.keys().map(|key| key.level).max().unwrap_or(0)
    }

    /// Cubes still waiting for a (re-)mesh, LOD0 and LOD together.
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    /// Dirty LOD (level >= 1) cubes only — the LOD mesh queue length.
    pub fn lod_dirty_count(&self) -> usize {
        self.dirty.iter().filter(|key| key.level > 0).count()
    }

    pub fn cube(&self, key: CubeKey) -> Option<&Arc<Cube>> {
        self.cubes.get(&key)
    }

    /// The block at `pos`, or `None` when its cube is not loaded.
    pub fn block(&self, pos: BlockPos) -> Option<jbm_core::block::BlockStateId> {
        let (cube_pos, local) = pos.split();
        self.cubes
            .get(&CubeKey::lod0(cube_pos))
            .map(|cube| cube.get(local))
    }

    /// Whether `state` has a collision shape (the builtin registry's
    /// table) — the place-overlap check for optimistic edits.
    pub fn is_solid_state(&self, state: jbm_core::block::BlockStateId) -> bool {
        self.solid.is_solid(state)
    }

    /// Test hook: removes a single block from the client's copy of the
    /// world only — the server never learns of it. Prediction tests use it
    /// to make the client's collision world disagree with the server's.
    #[cfg(feature = "testing")]
    pub fn remove_block(&mut self, pos: BlockPos) {
        self.set_block(pos, jbm_core::block::BlockStateId::AIR);
    }
}

/// World-space center of a cube at any level, for distance ordering
/// (LOD distances are in blocks, so finer levels sort first at equal
/// cube offsets).
fn cube_center(key: CubeKey) -> DVec3 {
    let size = key.block_size() as f64;
    DVec3::new(
        f64::from(key.pos.x) * size + size / 2.0,
        f64::from(key.pos.y) * size + size / 2.0,
        f64::from(key.pos.z) * size + size / 2.0,
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

    fn lod_cube_data(level: u8, pos: CubePos) -> ServerMsg {
        ServerMsg::LodCubeData {
            level,
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
        let positions: Vec<CubeKey> = dirty.iter().map(|(p, _, _)| *p).collect();
        assert!(positions.contains(&CubeKey::lod0(origin)));
        assert!(positions.contains(&CubeKey::lod0(neighbor)));
    }

    #[test]
    fn unload_removes_the_cube() {
        let mut world = ClientWorld::new();
        let pos = CubePos { x: 3, y: 1, z: 2 };
        assert_eq!(
            world.apply(&cube_data(pos)),
            WorldChange::Loaded(CubeKey::lod0(pos))
        );
        assert_eq!(world.cube_count(), 1);
        assert_eq!(
            world.apply(&ServerMsg::UnloadCube { pos }),
            WorldChange::Unloaded(CubeKey::lod0(pos))
        );
        assert_eq!(world.cube_count(), 0);
        assert_eq!(world.version(CubeKey::lod0(pos)), None);
        assert_eq!(world.dirty_count(), 0, "unloaded cube leaves the dirty set");
    }

    #[test]
    fn lod_cubes_live_beside_lod0_with_the_same_bookkeeping() {
        let mut world = ClientWorld::new();
        let pos = CubePos { x: -2, y: 1, z: 3 };
        let key = CubeKey { level: 2, pos };
        assert_eq!(
            world.apply(&lod_cube_data(2, pos)),
            WorldChange::Loaded(key)
        );
        assert_eq!(world.lod_cube_count(2), 1);
        assert_eq!(world.cube_count(), 0, "LOD cubes are not LOD0 cubes");
        assert_eq!(world.dirty_count(), 1);
        assert_eq!(world.version(key), Some(1));

        // A same-level neighbor dirties the first cube; a LOD0 cube at the
        // same coordinates does not.
        let neighbor = CubePos { x: -1, y: 1, z: 3 };
        world.apply(&lod_cube_data(2, neighbor));
        assert_eq!(world.dirty_count(), 2, "both level-2 cubes");
        let _ = world.take_dirty(usize::MAX, DVec3::ZERO);
        world.apply(&cube_data(pos));
        assert_eq!(world.dirty_count(), 1, "only the new LOD0 cube");

        // Replace: version bumps, still one cube.
        world.apply(&lod_cube_data(2, pos));
        assert_eq!(world.lod_cube_count(2), 2);
        assert_eq!(world.version(key), Some(5));

        // Unload removes exactly that level's cube.
        assert_eq!(
            world.apply(&ServerMsg::UnloadLodCube { level: 2, pos }),
            WorldChange::Unloaded(key)
        );
        assert_eq!(world.lod_cube_count(2), 1);
        assert_eq!(world.cube_count(), 1);
    }

    #[test]
    fn versions_increase_per_cube() {
        let mut world = ClientWorld::new();
        let pos = CubePos { x: 0, y: 0, z: 0 };
        world.apply(&cube_data(pos));
        assert_eq!(world.version(CubeKey::lod0(pos)), Some(1));
        world.apply(&cube_data(pos));
        assert_eq!(world.version(CubeKey::lod0(pos)), Some(2));
        // Other messages leave versions alone.
        world.apply(&ServerMsg::Tick { tick: 5 });
        assert_eq!(world.version(CubeKey::lod0(pos)), Some(2));
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
        let v_now = world.version(CubeKey::lod0(a)).unwrap();
        assert_ne!(v_old, v_now, "the job built without b must be stale");
    }

    #[test]
    fn reload_never_reuses_a_version() {
        let mut world = ClientWorld::new();
        let pos = CubePos { x: 0, y: 0, z: 0 };
        world.apply(&cube_data(pos));
        let v_before = world.version(CubeKey::lod0(pos)).unwrap();
        world.apply(&ServerMsg::UnloadCube { pos });
        world.apply(&cube_data(pos));
        assert!(world.version(CubeKey::lod0(pos)).unwrap() > v_before);
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
        let order: Vec<CubeKey> = taken.iter().map(|(p, _, _)| *p).collect();
        assert_eq!(
            order,
            vec![CubeKey::lod0(near), CubeKey::lod0(mid), CubeKey::lod0(far)],
            "nearest first"
        );
        assert_eq!(world.dirty_count(), 0);

        // Budget: re-dirty everything, take only one.
        for pos in [far, mid, near] {
            world.apply(&cube_data(pos));
        }
        let taken = world.take_dirty(1, camera);
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].0, CubeKey::lod0(near));
        assert_eq!(world.dirty_count(), 2);
    }

    #[test]
    fn lod0_comes_first_but_lod_keeps_a_quarter_of_the_budget() {
        let mut world = ClientWorld::new();
        // Six LOD0 cubes and six LOD2 cubes around the camera.
        for i in 0..6i32 {
            world.apply(&cube_data(CubePos { x: i, y: 0, z: 0 }));
            world.apply(&lod_cube_data(2, CubePos { x: i, y: 0, z: 0 }));
        }
        // Budget 8: LOD0 gets 6 (8 - 2), LOD gets its guaranteed 2.
        let taken = world.take_dirty(8, DVec3::ZERO);
        let lod0 = taken.iter().filter(|(k, _, _)| k.level == 0).count();
        let lod = taken.iter().filter(|(k, _, _)| k.level > 0).count();
        assert_eq!(lod0, 6);
        assert_eq!(lod, 2, "the LOD quarter is guaranteed");
        assert_eq!(world.dirty_count(), 4);

        // A saturated LOD0 queue never starves LOD out completely.
        let rest = world.take_dirty(usize::MAX, DVec3::ZERO);
        assert_eq!(rest.len(), 4);
        assert_eq!(world.lod_dirty_count(), 0);

        // With only LOD dirt, LOD gets the whole budget.
        for i in 0..4i32 {
            world.apply(&lod_cube_data(3, CubePos { x: i, y: 0, z: 0 }));
        }
        let taken = world.take_dirty(16, DVec3::ZERO);
        assert_eq!(taken.len(), 4);
        assert!(taken.iter().all(|(k, _, _)| k.level == 3));
    }

    #[test]
    fn lod_distance_is_measured_in_blocks_per_cell() {
        let mut world = ClientWorld::new();
        // A LOD0 cube 4 cubes out and a LOD2 cube 1 cube out: both 128
        // blocks from the camera's cube; the LOD2 cube wins the tie by
        // sort stability... instead put them at clearly different block
        // distances: LOD0 at offset 2 (64+ blocks), LOD2 at offset 2
        // (256 blocks).
        world.apply(&lod_cube_data(2, CubePos { x: 2, y: 0, z: 0 }));
        world.apply(&cube_data(CubePos { x: 2, y: 0, z: 0 }));
        let taken = world.take_dirty(2, DVec3::ZERO);
        assert_eq!(taken[0].0.level, 0, "the LOD0 cube is nearer in blocks");
        assert_eq!(taken[1].0.level, 2);
    }

    #[test]
    fn mesh_inputs_carry_same_level_neighbors_and_level() {
        let mut world = ClientWorld::new();
        let a = CubePos { x: 0, y: 0, z: 0 };
        let b = CubePos { x: 1, y: 0, z: 0 };
        world.apply(&cube_data(a));
        world.apply(&cube_data(b));
        world.apply(&lod_cube_data(1, a));
        world.apply(&lod_cube_data(1, b));
        let taken = world.take_dirty(10, DVec3::ZERO);
        assert_eq!(taken.len(), 4);
        for (key, _, input) in &taken {
            assert_eq!(input.level, key.level);
            if key.level == 0 && key.pos == a || key.level == 1 && key.pos == a {
                assert!(input.neighbor(1, 0, 0).is_some(), "+X neighbor loaded");
                assert!(input.neighbor(-1, 0, 0).is_none(), "-X neighbor missing");
                assert!(
                    input.neighbor(1, 1, 0).is_none(),
                    "diagonal neighbor missing"
                );
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
                yaw: 0.0,
                pitch: 0.0,
                flying: false,
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
        // A LOD cube overlapping must not leak into collision.
        world.apply(&lod_cube_data(1, origin));
        let solid = |x: i64, y: i64, z: i64| world.solid(BlockPos { x, y, z });
        assert_eq!(solid(3, 2, 7), Some(true), "stone inside the floor cube");
        let size = i64::from(CUBE_SIZE);
        assert_eq!(solid(0, size, 0), Some(false), "air in the cube above");
        assert_eq!(solid(0, 2 * size, 0), None, "an unloaded cube is None");
    }

    /// Loads a 3x3x3 grid of cubes, clears the dirty set and returns the
    /// world ready for an edit-dirty-rule assertion.
    fn settled_grid() -> ClientWorld {
        let mut world = ClientWorld::new();
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    world.apply(&cube_data(CubePos {
                        x: dx,
                        y: dy,
                        z: dz,
                    }));
                }
            }
        }
        let _ = world.take_dirty(usize::MAX, DVec3::ZERO);
        assert_eq!(world.dirty_count(), 0);
        world
    }

    #[test]
    fn block_edits_dirty_exactly_the_touched_cubes() {
        let last = (CUBE_SIZE - 1) as u8;
        let cases: [(u8, u8, u8, Vec<CubePos>); 4] = [
            // Interior: only the edited cube.
            (5, 5, 5, vec![CubePos { x: 0, y: 0, z: 0 }]),
            // Face block on -X: the -X neighbor too.
            (
                0,
                5,
                5,
                vec![CubePos { x: 0, y: 0, z: 0 }, CubePos { x: -1, y: 0, z: 0 }],
            ),
            // Edge block at (-X, -Y, +Z): two face neighbors, one edge.
            (
                0,
                0,
                last,
                vec![
                    CubePos { x: 0, y: 0, z: 0 },
                    CubePos { x: -1, y: 0, z: 0 },
                    CubePos { x: 0, y: -1, z: 0 },
                    CubePos { x: -1, y: -1, z: 0 },
                    CubePos { x: 0, y: 0, z: 1 },
                    CubePos { x: -1, y: 0, z: 1 },
                    CubePos { x: 0, y: -1, z: 1 },
                    CubePos { x: -1, y: -1, z: 1 },
                ],
            ),
            // Corner block: all 7 neighbors.
            (
                last,
                0,
                0,
                vec![
                    CubePos { x: 0, y: 0, z: 0 },
                    CubePos { x: 1, y: 0, z: 0 },
                    CubePos { x: 0, y: -1, z: 0 },
                    CubePos { x: 1, y: -1, z: 0 },
                    CubePos { x: 0, y: 0, z: -1 },
                    CubePos { x: 1, y: 0, z: -1 },
                    CubePos { x: 0, y: -1, z: -1 },
                    CubePos { x: 1, y: -1, z: -1 },
                ],
            ),
        ];
        for (x, y, z, expected) in cases {
            let mut world = settled_grid();
            let at = BlockPos {
                x: i64::from(x),
                y: i64::from(y),
                z: i64::from(z),
            };
            assert!(
                world.set_block(at, jbm_core::block::BlockStateId::AIR),
                "cube loaded"
            );
            let mut dirtied: Vec<CubeKey> = world
                .take_dirty(usize::MAX, DVec3::new(16.0, 16.0, 16.0))
                .into_iter()
                .map(|(key, _, _)| key)
                .collect();
            dirtied.sort_by_key(|k| (k.level, k.pos.x, k.pos.y, k.pos.z));
            let mut expected: Vec<CubeKey> = expected.into_iter().map(CubeKey::lod0).collect();
            expected.sort_by_key(|k| (k.level, k.pos.x, k.pos.y, k.pos.z));
            assert_eq!(dirtied, expected, "local ({x},{y},{z})");
            assert_eq!(world.dirty_count(), 0);
        }
    }

    #[test]
    fn an_edit_echo_dirties_nothing() {
        let mut world = settled_grid();
        let at = BlockPos { x: 5, y: 5, z: 5 };
        assert!(world.set_block(at, jbm_core::block::BlockStateId::AIR));
        let _ = world.take_dirty(usize::MAX, DVec3::ZERO);
        let version = world.version(CubeKey::lod0(CubePos { x: 0, y: 0, z: 0 }));
        let echo = ServerMsg::BlockChange {
            pos: at,
            state: jbm_core::block::BlockStateId::AIR,
        };
        assert_eq!(world.apply(&echo), WorldChange::Edited(at));
        assert_eq!(world.dirty_count(), 0);
        assert_eq!(
            world.version(CubeKey::lod0(CubePos { x: 0, y: 0, z: 0 })),
            version
        );
    }

    #[test]
    fn unloaded_neighbors_are_not_dirtied_by_an_edit() {
        let mut world = ClientWorld::new();
        // A lone cube: every neighbor is missing, so only it dirties.
        world.apply(&cube_data(CubePos { x: 7, y: 0, z: 0 }));
        let _ = world.take_dirty(usize::MAX, DVec3::ZERO);
        let corner =
            CubePos { x: 7, y: 0, z: 0 }.block(jbm_core::coords::LocalPos { x: 0, y: 0, z: 0 });
        assert!(world.set_block(corner, jbm_core::block::BlockStateId::AIR));
        assert_eq!(world.dirty_count(), 1);
    }

    #[test]
    fn block_change_sets_loaded_blocks_and_ignores_the_rest() {
        let mut world = ClientWorld::new();
        let origin = CubePos { x: 0, y: 0, z: 0 };
        world.apply(&cube_data(origin));
        let _ = world.take_dirty(usize::MAX, DVec3::ZERO);
        let at = BlockPos { x: 3, y: 2, z: 7 };
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        assert_eq!(
            world.apply(&ServerMsg::BlockChange {
                pos: at,
                state: blocks.dirt
            }),
            WorldChange::Edited(at)
        );
        assert_eq!(world.block(at), Some(blocks.dirt));
        assert_eq!(world.dirty_count(), 1, "interior block: just the cube");

        // An unloaded cube ignores the change entirely.
        let far = BlockPos {
            x: 1000,
            y: 0,
            z: 0,
        };
        assert_eq!(
            world.apply(&ServerMsg::BlockChange {
                pos: far,
                state: blocks.dirt
            }),
            WorldChange::None
        );
        assert_eq!(world.block(far), None);
        assert_eq!(world.dirty_count(), 1);
    }

    #[test]
    fn edited_cubes_jump_the_mesh_queue() {
        let mut world = ClientWorld::new();
        // Streaming dirt: three cubes, the far one nearest the camera.
        for pos in [
            CubePos { x: 40, y: 0, z: 0 },
            CubePos { x: 5, y: 0, z: 0 },
            CubePos { x: 0, y: 0, z: 0 },
        ] {
            world.apply(&cube_data(pos));
        }
        let camera = DVec3::new(1290.0, 16.0, 16.0); // near the x=40 cube
        let edited = BlockPos { x: 3, y: 2, z: 7 }; // inside cube (0,0,0)
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        assert!(world.set_block(edited, blocks.dirt));

        // The edited cube is taken first even though it is farthest.
        let taken = world.take_dirty(1, camera);
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].0, CubeKey::lod0(CubePos { x: 0, y: 0, z: 0 }));
        // Then normal nearest-first for the rest.
        let rest = world.take_dirty(usize::MAX, camera);
        let order: Vec<CubeKey> = rest.iter().map(|(p, _, _)| *p).collect();
        assert_eq!(
            order,
            vec![
                CubeKey::lod0(CubePos { x: 40, y: 0, z: 0 }),
                CubeKey::lod0(CubePos { x: 5, y: 0, z: 0 })
            ]
        );
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
