//! Block editing: the hotbar, the hold-to-repeat timers, and the pure
//! break/place decisions. Winit-free so the flow is unit-testable; the app
//! wires mouse buttons, raycasts and sends the protocol messages.

use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
use jbm_core::coords::BlockPos;
use jbm_core::physics::{PlayerAabb, block_overlaps_box};
use jbm_core::raycast::RayHit;

use crate::world::ClientWorld;

/// How often a held edit button repeats, in seconds. The first press acts
/// immediately.
pub const EDIT_REPEAT: f64 = 0.25;

/// The nine hotbar slots (task 1.7's fixed set): stone, dirt, grass_block,
/// sand, glass, oak_planks, oak_log, leaves and water. Water is placeable
/// but non-solid. Slots are selected with keys 1–9.
#[derive(Debug, Clone)]
pub struct Hotbar {
    states: [BlockStateId; 9],
    selected: usize,
}

impl Default for Hotbar {
    fn default() -> Self {
        Self::new()
    }
}

impl Hotbar {
    pub fn new() -> Hotbar {
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        Hotbar {
            states: [
                blocks.stone,
                blocks.dirt,
                blocks.grass_block,
                blocks.sand,
                blocks.glass,
                blocks.oak_planks,
                blocks.oak_log,
                blocks.leaves,
                blocks.water,
            ],
            selected: 0,
        }
    }

    /// Selects slot `i` (0-based; out-of-range is ignored).
    pub fn select(&mut self, i: usize) {
        if i < self.states.len() {
            self.selected = i;
        }
    }

    /// The selected slot, 0-based.
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// The state in the selected slot.
    pub fn state(&self) -> BlockStateId {
        self.states[self.selected]
    }
}

/// Hold-to-repeat timing for one edit button. Time is any monotonic
/// seconds counter the app feeds (its accumulated frame time). `press`
/// arms the button so the next check fires immediately; each fired action
/// pushes the next one a full [`EDIT_REPEAT`] away.
#[derive(Debug, Clone, PartialEq)]
pub struct Repeat {
    held: bool,
    next: f64,
}

impl Default for Repeat {
    fn default() -> Self {
        Self::new()
    }
}

impl Repeat {
    pub fn new() -> Repeat {
        Repeat {
            held: false,
            next: f64::INFINITY,
        }
    }

    /// Arms the button: the action fires on the next check.
    pub fn press(&mut self) {
        self.held = true;
        self.next = f64::NEG_INFINITY;
    }

    /// Marks the button released; checks stop firing until the next
    /// press.
    pub fn release(&mut self) {
        self.held = false;
        self.next = f64::INFINITY;
    }

    /// Press or release in one call (mouse button events).
    pub fn set_held(&mut self, held: bool) {
        if held {
            self.press();
        } else {
            self.release();
        }
    }

    /// Whether the button is held.
    pub fn is_held(&self) -> bool {
        self.held
    }

    /// Whether an action should fire at `now`.
    pub fn due(&self, now: f64) -> bool {
        self.held && now >= self.next
    }

    /// Records that the action fired at `now`, scheduling the next one.
    pub fn fired(&mut self, now: f64) {
        self.next = now + EDIT_REPEAT;
    }
}

/// Where a place goes: the cell in front of the hit face, when that cell
/// is loaded and air, and the block (if solid) would not strictly overlap
/// `player`'s box — the same predicate the server validates with, so an
/// optimistic place is only rejected on race.
pub fn place_target(
    hit: &RayHit,
    state: BlockStateId,
    world: &ClientWorld,
    player: &PlayerAabb,
) -> Option<BlockPos> {
    let n = hit.face.normal();
    let pos = BlockPos {
        x: hit.block.x + i64::from(n[0]),
        y: hit.block.y + i64::from(n[1]),
        z: hit.block.z + i64::from(n[2]),
    };
    if world.block(pos) != Some(BlockStateId::AIR) {
        return None; // not loaded, or not air
    }
    if world.is_solid_state(state) && block_overlaps_box(pos, player.min, player.max) {
        return None;
    }
    Some(pos)
}

#[cfg(test)]
mod tests {
    use super::{EDIT_REPEAT, Hotbar, Repeat, place_target};
    use glam::DVec3;
    use jbm_core::block::{BlockRegistry, builtin};
    use jbm_core::coords::BlockPos;
    use jbm_core::cube::Cube;
    use jbm_core::protocol::ServerMsg;
    use jbm_core::raycast::{Face, RayHit};

    fn blocks() -> builtin::Blocks {
        let mut registry = BlockRegistry::new();
        builtin::Blocks::register(&mut registry)
    }

    #[test]
    fn hotbar_holds_the_nine_builtin_states_in_order() {
        let blocks = blocks();
        let mut hotbar = Hotbar::new();
        for slot in 0..9 {
            hotbar.select(slot);
            assert_eq!(hotbar.selected(), slot);
            assert_eq!(
                hotbar.state(),
                [
                    blocks.stone,
                    blocks.dirt,
                    blocks.grass_block,
                    blocks.sand,
                    blocks.glass,
                    blocks.oak_planks,
                    blocks.oak_log,
                    blocks.leaves,
                    blocks.water
                ][slot]
            );
        }
        hotbar.select(9);
        assert_eq!(hotbar.selected(), 8, "out-of-range selection is ignored");
    }

    #[test]
    fn repeat_fires_immediately_then_every_interval() {
        let mut repeat = Repeat::new();
        assert!(!repeat.due(0.0), "idle never fires");
        repeat.press();
        assert!(repeat.due(0.0), "the first press acts immediately");
        repeat.fired(1.0);
        assert!(!repeat.due(1.24), "waiting out the interval");
        assert!(repeat.due(1.25), "an edit exactly at the interval fires");
        repeat.fired(1.25);
        assert!(repeat.due(1.25 + EDIT_REPEAT));

        repeat.release();
        assert!(!repeat.is_held());
        assert!(!repeat.due(100.0), "released stops firing");
        repeat.press();
        assert!(repeat.due(100.0), "re-press acts immediately again");
    }

    /// A world whose one cube is stone up to y = 3 and air above, plus a
    /// hit on the stone's top face and a player-box constructor.
    fn setup() -> (crate::world::ClientWorld, RayHit) {
        let mut world = crate::world::ClientWorld::new();
        let blocks = blocks();
        let cube = Cube::from_fn(|p| {
            if p.y <= 3 {
                blocks.stone
            } else {
                jbm_core::block::BlockStateId::AIR
            }
        });
        world.apply(&ServerMsg::CubeData {
            pos: jbm_core::coords::CubePos { x: 0, y: 0, z: 0 },
            cube: Box::new(cube),
        });
        let _ = world.take_dirty(usize::MAX, DVec3::ZERO);
        let hit = RayHit {
            block: BlockPos { x: 5, y: 3, z: 7 },
            face: Face::PosY,
            distance: 3.0,
        };
        (world, hit)
    }

    fn aabb(min: DVec3, max: DVec3) -> jbm_core::physics::PlayerAabb {
        jbm_core::physics::PlayerAabb { min, max }
    }

    #[test]
    fn place_target_goes_in_front_of_the_hit_face() {
        let blocks = blocks();
        let (world, hit) = setup();
        let away = aabb(
            DVec3::new(100.0, 100.0, 100.0),
            DVec3::new(101.0, 101.8, 101.0),
        );
        assert_eq!(
            place_target(&hit, blocks.stone, &world, &away),
            Some(BlockPos { x: 5, y: 4, z: 7 })
        );
    }

    #[test]
    fn place_target_rejects_non_air_and_unloaded_cells() {
        let blocks = blocks();
        let (world, hit) = setup();
        let away = aabb(
            DVec3::new(100.0, 100.0, 100.0),
            DVec3::new(101.0, 101.8, 101.0),
        );

        // Solid cell: the hit block itself is stone, so aiming at its -Y
        // face targets the stone below.
        let into_solid = RayHit {
            block: hit.block,
            face: Face::NegY,
            distance: 3.0,
        };
        assert_eq!(place_target(&into_solid, blocks.stone, &world, &away), None);

        // Unloaded cell: far outside the one loaded cube.
        let far = RayHit {
            block: BlockPos { x: 500, y: 3, z: 7 },
            face: Face::PosY,
            distance: 3.0,
        };
        assert_eq!(place_target(&far, blocks.stone, &world, &away), None);
    }

    #[test]
    fn place_target_rejects_overlap_with_the_predicted_player() {
        let blocks = blocks();
        let (world, hit) = setup();
        // The player box swallows the target cell (5, 4, 7).
        let overlapping = aabb(DVec3::new(4.8, 3.9, 6.8), DVec3::new(5.7, 5.9, 7.7));
        assert_eq!(place_target(&hit, blocks.stone, &world, &overlapping), None);
        // Touching is fine (no strict overlap): the box rests exactly on
        // the cell's top face (y = 5.0).
        let touching = aabb(DVec3::new(4.8, 5.0, 6.8), DVec3::new(5.7, 5.9, 7.7));
        assert_eq!(
            place_target(&hit, blocks.stone, &world, &touching),
            Some(BlockPos { x: 5, y: 4, z: 7 })
        );
        // Non-solid states (water) never fail the overlap check.
        assert_eq!(
            place_target(&hit, blocks.water, &world, &overlapping),
            Some(BlockPos { x: 5, y: 4, z: 7 })
        );
    }
}
