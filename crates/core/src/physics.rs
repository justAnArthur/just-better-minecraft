//! Shared player physics: one fixed tick of axis-aligned box collision
//! against full-block collision shapes. Pure logic on f64 — the client's
//! predictor and the authoritative server run the exact same [`step`], so
//! their results agree bit for bit on identical worlds.

use glam::DVec3;

use crate::block::{BlockRegistry, BlockStateId};
use crate::coords::BlockPos;

/// The fixed physics timestep in seconds. Physics always advances one tick
/// at a time; there is no variable dt anywhere.
pub const TICK_DT: f64 = 1.0 / 20.0;

/// Player box width (X and Z), feet at the bottom center.
pub const PLAYER_WIDTH: f64 = 0.6;
/// Half the player box width; the box is centered on `position.x/z`.
pub const HALF_WIDTH: f64 = PLAYER_WIDTH / 2.0;
/// Player box height.
pub const PLAYER_HEIGHT: f64 = 1.8;
/// Eye height above the feet.
pub const EYE_HEIGHT: f64 = 1.62;
/// Walking speed, blocks per second.
pub const WALK_SPEED: f64 = 4.3;
/// Sprinting speed, blocks per second.
pub const SPRINT_SPEED: f64 = 5.6;
/// Gravity, blocks per second squared.
pub const GRAVITY: f64 = 32.0;
/// Jump launch velocity; a jump clears just over 1.25 blocks.
pub const JUMP_VELOCITY: f64 = 9.0;
/// Terminal fall speed, blocks per second.
pub const TERMINAL_FALL_SPEED: f64 = 78.0;
/// Flying speed, blocks per second.
pub const FLY_SPEED: f64 = 11.0;
/// Flying sprint speed, blocks per second.
pub const FLY_SPRINT_SPEED: f64 = 22.0;

/// One tick of player input. In walk mode `up` is ignored and `jump`
/// jumps; in fly mode `up` moves up/down (Space/Shift) and `jump` is
/// ignored.
///
/// `fly_toggle` flips [`PlayerState::flying`] at the start of the step.
/// The server trusts it for now — Stage 1 is creative-only; permission
/// checks arrive with the anti-cheat task.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveInput {
    /// -1, 0 or 1 along the yaw direction.
    pub forward: i8,
    /// -1, 0 or 1 along the yaw-right direction.
    pub strafe: i8,
    /// -1, 0 or 1; fly mode only.
    pub up: i8,
    /// Jump request; walk mode only.
    pub jump: bool,
    pub sprint: bool,
    /// Flip flying at the start of this step (one edge per press).
    pub fly_toggle: bool,
    /// Horizontal look direction in radians; same convention as the
    /// client camera: yaw 0 looks toward -Z, positive yaw turns toward +X.
    pub yaw: f32,
}

impl MoveInput {
    /// An input with no keys held, looking toward -Z.
    pub const IDLE: MoveInput = MoveInput {
        forward: 0,
        strafe: 0,
        up: 0,
        jump: false,
        sprint: false,
        fly_toggle: false,
        yaw: 0.0,
    };
}

/// The player's collision box, min inclusive / max exclusive in spirit:
/// touching faces do not overlap (see [`step`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerAabb {
    pub min: DVec3,
    pub max: DVec3,
}

/// The player's dynamic state: `position` is the **feet** (bottom center
/// of the box).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerState {
    /// Feet position.
    pub position: DVec3,
    pub velocity: DVec3,
    pub on_ground: bool,
    pub flying: bool,
}

impl PlayerState {
    /// The player's collision box.
    pub fn aabb(&self) -> PlayerAabb {
        PlayerAabb {
            min: DVec3::new(
                self.position.x - HALF_WIDTH,
                self.position.y,
                self.position.z - HALF_WIDTH,
            ),
            max: DVec3::new(
                self.position.x + HALF_WIDTH,
                self.position.y + PLAYER_HEIGHT,
                self.position.z + HALF_WIDTH,
            ),
        }
    }

    /// The eye position, `position + EYE_HEIGHT`.
    pub fn eye(&self) -> DVec3 {
        DVec3::new(
            self.position.x,
            self.position.y + EYE_HEIGHT,
            self.position.z,
        )
    }
}

/// World query for collision: `Some(solid)` says whether the block at
/// `pos` has a collision shape, `None` says the block is not loaded.
pub trait BlockShapes {
    fn solid(&self, pos: BlockPos) -> Option<bool>;
}

/// Flat per-state solidity, indexed by [`BlockStateId`], built once from a
/// [`BlockRegistry`] so the hot path never touches the registry.
///
/// The collision shape of a state is a full unit box when `StateInfo`
/// says solid, else empty. Per-state collision shapes are a later task;
/// this lookup is the one place that decides shape, so it can grow.
#[derive(Debug, Clone)]
pub struct SolidTable(Vec<bool>);

impl SolidTable {
    /// Builds the table for every state in `registry`.
    pub fn new(registry: &BlockRegistry) -> SolidTable {
        SolidTable(
            (0..registry.state_count())
                .map(|id| registry.info(BlockStateId(id)).solid)
                .collect(),
        )
    }

    /// Whether `state` has a collision shape.
    pub fn is_solid(&self, state: BlockStateId) -> bool {
        self.0.get(state.0 as usize).copied().unwrap_or(false)
    }
}

/// Whether the unit block box at `pos` strictly overlaps the box spanned by
/// `min`/`max`. Touching faces do not overlap — the same boundary rule the
/// sweeps use. Shared by the server's place validation and the client's
/// optimistic place check, so both reject exactly the same placements.
pub fn block_overlaps_box(pos: BlockPos, min: DVec3, max: DVec3) -> bool {
    let (x, y, z) = (pos.x as f64, pos.y as f64, pos.z as f64);
    x < max.x && x + 1.0 > min.x && y < max.y && y + 1.0 > min.y && z < max.z && z + 1.0 > min.z
}

/// Blocks whose open interval `(b, b+1)` strictly overlaps the open
/// interval `(min, max)`: `floor(min)` through `ceil(max) - 1` inclusive.
///
/// Boundary rule: touching faces do not overlap. A box resting exactly on
/// `y = 64.0` gets the range `64..=65` on Y (the block below, 63, is
/// excluded), so it does not collide with the floor during X/Z sweeps —
/// and conversely the floor stops a falling box at exactly `64.0`. There
/// is deliberately no epsilon anywhere: an epsilon that let the box sink
/// into a block would make client and server disagree about resting
/// positions, and exact face coordinates keep them bit-identical.
fn block_span(min: f64, max: f64) -> (i64, i64) {
    (min.floor() as i64, max.ceil() as i64 - 1)
}

/// Clips a displacement along one axis against solid blocks.
///
/// `min`/`max` are the box's extent on the moving axis; the extents on
/// the other two axes are fixed during this sweep and folded into
/// `blocks_on`, which reports whether the column at moving-axis block `b`
/// holds any solid block the box would collide with. Returns the applied
/// displacement and whether movement was blocked.
fn clip_axis(min: f64, max: f64, d: f64, blocks_on: impl Fn(i64) -> bool) -> (f64, bool) {
    if d > 0.0 {
        let target = max + d;
        let mut lead = target;
        let mut blocked = false;
        // The leading face can only enter blocks it strictly crosses:
        // from ceil(max) (the block starting at the face when it rests
        // exactly on a boundary) up to ceil(target) - 1.
        for b in (max.ceil() as i64)..=(target.ceil() as i64 - 1) {
            if blocks_on(b) {
                lead = lead.min(b as f64);
                blocked = true;
            }
        }
        (lead - max, blocked)
    } else if d < 0.0 {
        let target = min + d;
        let mut lead = target;
        let mut blocked = false;
        // Symmetric on the min face: blocks strictly crossed going down,
        // floor(min) - 1 being the block whose top face the feet rest on.
        for b in (target.floor() as i64)..=(min.floor() as i64 - 1) {
            if blocks_on(b) {
                lead = lead.max((b + 1) as f64);
                blocked = true;
            }
        }
        (lead - min, blocked)
    } else {
        (0.0, false)
    }
}

/// Advances `state` by one fixed tick under `input`, against `world`.
///
/// Step order:
///
/// 1. Apply the fly toggle, then set the wish velocity: horizontal
///    direction from forward/strafe rotated by yaw, normalized so
///    diagonals are not faster, times the speed. Velocity is set
///    directly — no acceleration or friction yet, and air control equals
///    ground control. Walk mode jumps from the ground and subtracts
///    gravity (clamped to terminal speed); fly mode sets `velocity.y`
///    from `up` with no gravity. The vertical displacement integrates
///    gravity exactly across the tick (see below).
/// 2. Swept collision per axis, Y then X then Z: clip each axis
///    displacement against every solid block box the moving box would
///    cross, then move. A blocked axis zeroes that velocity component.
/// 3. `on_ground` is true when the Y sweep was blocked while moving down.
///
/// Unloaded blocks: if any block the swept box touches this tick is not
/// loaded (`None`), the step returns the state unchanged — no movement
/// and no gravity. A player freezes at the edge of loaded terrain
/// instead of falling into it. Client and server both use this rule,
/// which keeps them in agreement while cubes stream in.
///
/// Arithmetic: only `+ - * /`, `floor`, `ceil`, `min`, `max` and `sqrt`
/// on f64, so results are bit-exact across platforms. The one exception
/// is `yaw.sin_cos()`, computed once per step below — the only
/// platform-dependent operation; server correction absorbs any drift.
pub fn step(state: PlayerState, input: &MoveInput, world: &impl BlockShapes) -> PlayerState {
    let mut next = state;
    if input.fly_toggle {
        next.flying = !next.flying;
    }

    // The step's single sin_cos: everything else is exact f64 arithmetic.
    let (sy, cy) = input.yaw.sin_cos();
    let (sy, cy) = (f64::from(sy), f64::from(cy));
    // Yaw convention (same as the client camera): yaw 0 looks toward -Z,
    // positive yaw turns toward +X.
    let forward = (sy, -cy);
    let right = (cy, sy);

    let mut wish_x = forward.0 * f64::from(input.forward) + right.0 * f64::from(input.strafe);
    let mut wish_z = forward.1 * f64::from(input.forward) + right.1 * f64::from(input.strafe);
    let len = (wish_x * wish_x + wish_z * wish_z).sqrt();
    if len > 0.0 {
        wish_x /= len;
        wish_z /= len;
    }
    let speed = match (next.flying, input.sprint) {
        (true, true) => FLY_SPRINT_SPEED,
        (true, false) => FLY_SPEED,
        (false, true) => SPRINT_SPEED,
        (false, false) => WALK_SPEED,
    };
    next.velocity.x = wish_x * speed;
    next.velocity.z = wish_z * speed;
    // Vertical: fly sets the velocity directly; walk integrates gravity
    // across the tick. The tick's displacement is the exact ballistic
    // integral (the average of the pre- and post-gravity velocities) so a
    // jump's apex is v²/2g = 1.2656 blocks, as the constants table
    // promises — a plain Euler sum would overshoot to 1.5.
    let dy = if next.flying {
        next.velocity.y = f64::from(input.up) * FLY_SPEED;
        next.velocity.y * TICK_DT
    } else {
        if input.jump && next.on_ground {
            next.velocity.y = JUMP_VELOCITY;
        }
        let vy_before = next.velocity.y;
        next.velocity.y -= GRAVITY * TICK_DT;
        if next.velocity.y < -TERMINAL_FALL_SPEED {
            next.velocity.y = -TERMINAL_FALL_SPEED;
        }
        (vy_before + next.velocity.y) * 0.5 * TICK_DT
    };

    let dx = next.velocity.x * TICK_DT;
    let dz = next.velocity.z * TICK_DT;

    // Unloaded check over the swept box: the union of the current box and
    // where the unclipped displacement would take it.
    let union = (
        DVec3::new(
            (next.position.x - HALF_WIDTH).min(next.position.x - HALF_WIDTH + dx),
            next.position.y.min(next.position.y + dy),
            (next.position.z - HALF_WIDTH).min(next.position.z - HALF_WIDTH + dz),
        ),
        DVec3::new(
            (next.position.x + HALF_WIDTH).max(next.position.x + HALF_WIDTH + dx),
            (next.position.y + PLAYER_HEIGHT).max(next.position.y + PLAYER_HEIGHT + dy),
            (next.position.z + HALF_WIDTH).max(next.position.z + HALF_WIDTH + dz),
        ),
    );
    let (ux0, ux1) = block_span(union.0.x, union.1.x);
    let (uy0, uy1) = block_span(union.0.y, union.1.y);
    let (uz0, uz1) = block_span(union.0.z, union.1.z);
    for by in uy0..=uy1 {
        for bz in uz0..=uz1 {
            for bx in ux0..=ux1 {
                if world
                    .solid(BlockPos {
                        x: bx,
                        y: by,
                        z: bz,
                    })
                    .is_none()
                {
                    return state;
                }
            }
        }
    }

    // Y sweep. Static X/Z spans come from the current box (unchanged by a
    // Y move), so a box resting exactly on a floor does not collide with
    // the block below while sweeping X/Z later.
    let (x0, x1) = block_span(next.position.x - HALF_WIDTH, next.position.x + HALF_WIDTH);
    let (z0, z1) = block_span(next.position.z - HALF_WIDTH, next.position.z + HALF_WIDTH);
    let (applied, blocked) =
        clip_axis(next.position.y, next.position.y + PLAYER_HEIGHT, dy, |by| {
            (z0..=z1).any(|bz| {
                (x0..=x1).any(|bx| {
                    world.solid(BlockPos {
                        x: bx,
                        y: by,
                        z: bz,
                    }) == Some(true)
                })
            })
        });
    next.position.y += applied;
    let on_ground = blocked && dy < 0.0;
    if blocked {
        next.velocity.y = 0.0;
    }
    next.on_ground = on_ground;

    // X sweep, against the post-Y Y-span and the current Z-span.
    let (y0, y1) = block_span(next.position.y, next.position.y + PLAYER_HEIGHT);
    let (applied, blocked) = clip_axis(
        next.position.x - HALF_WIDTH,
        next.position.x + HALF_WIDTH,
        dx,
        |bx| {
            (z0..=z1).any(|bz| {
                (y0..=y1).any(|by| {
                    world.solid(BlockPos {
                        x: bx,
                        y: by,
                        z: bz,
                    }) == Some(true)
                })
            })
        },
    );
    next.position.x += applied;
    if blocked {
        next.velocity.x = 0.0;
    }

    // Z sweep, against the post-Y Y-span and the post-X X-span.
    let (x0, x1) = block_span(next.position.x - HALF_WIDTH, next.position.x + HALF_WIDTH);
    let (applied, blocked) = clip_axis(
        next.position.z - HALF_WIDTH,
        next.position.z + HALF_WIDTH,
        dz,
        |bz| {
            (x0..=x1).any(|bx| {
                (y0..=y1).any(|by| {
                    world.solid(BlockPos {
                        x: bx,
                        y: by,
                        z: bz,
                    }) == Some(true)
                })
            })
        },
    );
    next.position.z += applied;
    if blocked {
        next.velocity.z = 0.0;
    }

    next
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world whose blocks below `top` are solid and everything above is
    /// air; everything is loaded.
    struct Floor {
        top: i64,
    }

    impl BlockShapes for Floor {
        fn solid(&self, pos: BlockPos) -> Option<bool> {
            Some(pos.y < self.top)
        }
    }

    /// A single-block-thick floor at `y == layer`; air above and below.
    struct ThinFloor {
        layer: i64,
    }

    impl BlockShapes for ThinFloor {
        fn solid(&self, pos: BlockPos) -> Option<bool> {
            Some(pos.y == self.layer)
        }
    }

    fn on_floor(x: f64, y: f64, z: f64) -> PlayerState {
        PlayerState {
            position: DVec3::new(x, y, z),
            velocity: DVec3::ZERO,
            on_ground: true,
            flying: false,
        }
    }

    /// Floor at y <= 63 plus a wall from `x_min` eastward, two blocks
    /// higher than needed so walking and jumping both stay blocked.
    struct Wall {
        x_min: i64,
    }

    impl BlockShapes for Wall {
        fn solid(&self, pos: BlockPos) -> Option<bool> {
            Some(pos.y < 64 || (pos.x >= self.x_min && pos.y < 70))
        }
    }

    fn idle() -> MoveInput {
        MoveInput::IDLE
    }

    fn walking(yaw: f32) -> MoveInput {
        MoveInput {
            forward: 1,
            yaw,
            ..MoveInput::IDLE
        }
    }

    fn run_until(
        state: PlayerState,
        input: &MoveInput,
        world: &impl BlockShapes,
        max: usize,
        stop: impl Fn(PlayerState) -> bool,
    ) -> PlayerState {
        let mut s = state;
        for _ in 0..max {
            if stop(s) {
                return s;
            }
            s = step(s, input, world);
        }
        s
    }

    #[test]
    fn falling_from_ten_blocks_lands_exactly_on_the_floor() {
        let world = Floor { top: 64 };
        let start = PlayerState {
            position: DVec3::new(0.5, 74.0, 0.5),
            velocity: DVec3::ZERO,
            on_ground: false,
            flying: false,
        };
        let end = run_until(start, &idle(), &world, 100, |s| s.on_ground);
        assert!(end.on_ground);
        assert_eq!(end.position.y, 64.0, "feet exactly on the top face");
        assert_eq!(end.position.x, 0.5, "no horizontal drift");
        assert_eq!(end.position.z, 0.5);
        assert_eq!(end.velocity.y, 0.0);
    }

    #[test]
    fn terminal_fall_lands_on_a_single_block_floor_without_tunneling() {
        let world = ThinFloor { layer: 63 };
        let start = PlayerState {
            position: DVec3::new(0.5, 1063.0, 0.5),
            velocity: DVec3::ZERO,
            on_ground: false,
            flying: false,
        };
        let mut s = start;
        let mut landed = None;
        for _ in 0..2000 {
            if s.on_ground {
                landed = Some(s);
                break;
            }
            s = step(s, &idle(), &world);
            assert!(
                s.velocity.y >= -TERMINAL_FALL_SPEED,
                "clamped to terminal: {}",
                s.velocity.y
            );
        }
        let end = landed.expect("never landed — tunneled through the floor");
        assert_eq!(end.position.y, 64.0);
        assert!(end.on_ground);
    }

    #[test]
    fn walking_into_a_wall_stops_flush_and_slides_along_the_free_axis() {
        // Floor at y <= 63 plus a wall at x >= 5.
        let world = Wall { x_min: 5 };
        let mut s = on_floor(0.5, 64.0, 0.5);
        // East is +X: yaw pi/2.
        let east = walking(std::f32::consts::FRAC_PI_2);
        for _ in 0..40 {
            s = step(s, &east, &world);
        }
        assert_eq!(s.position.x + HALF_WIDTH, 5.0, "stops flush with the wall");
        assert!(s.on_ground);

        // Face -Z and walk: x stays pinned, z keeps moving (sliding).
        let north = walking(0.0);
        let z_before = s.position.z;
        for _ in 0..10 {
            s = step(s, &north, &world);
        }
        assert_eq!(s.position.x + HALF_WIDTH, 5.0, "still flush");
        assert!(s.position.z < z_before, "keeps sliding along the free axis");
    }

    #[test]
    fn corridor_corner_never_tunnels_through_diagonal_gaps() {
        // Two walls meeting at an edge: x >= 5 and z >= 5, two blocks high
        // over a floor. A diagonal run into the corner must not slip
        // through the gap where the walls meet.
        struct Corner;
        impl BlockShapes for Corner {
            fn solid(&self, pos: BlockPos) -> Option<bool> {
                Some(pos.y < 64 || ((pos.x >= 5 || pos.z >= 5) && (64..=65).contains(&pos.y)))
            }
        }
        let world = Corner;
        let mut s = on_floor(0.5, 64.0, 0.5);
        // A net diagonal toward (+X, +Z): yaw pi points forward at +Z and
        // strafe-left at +X.
        let diagonal = MoveInput {
            forward: 1,
            strafe: -1,
            yaw: std::f32::consts::PI,
            ..MoveInput::IDLE
        };
        for _ in 0..60 {
            s = step(s, &diagonal, &world);
            let b = s.aabb();
            let (x0, x1) = block_span(b.min.x, b.max.x);
            let (y0, y1) = block_span(b.min.y, b.max.y);
            let (z0, z1) = block_span(b.min.z, b.max.z);
            for y in y0..=y1 {
                for z in z0..=z1 {
                    for x in x0..=x1 {
                        if world.solid(BlockPos { x, y, z }) == Some(true) {
                            panic!("box overlaps solid block at ({x},{y},{z}): {s:?}");
                        }
                    }
                }
            }
        }
        // Blocked by the corner in both axes, flush on at least one.
        assert!(
            s.position.x + HALF_WIDTH <= 5.0 && s.position.z + HALF_WIDTH <= 5.0,
            "did not pass the corner: {s:?}"
        );
        assert!(
            s.position.x + HALF_WIDTH == 5.0 || s.position.z + HALF_WIDTH == 5.0,
            "must end flush on a wall: {s:?}"
        );
    }

    #[test]
    fn head_bump_clips_at_the_ceiling_and_zeroes_vertical_velocity() {
        // Floor at y <= 63, ceiling at y >= 66: two blocks of headroom.
        struct Ceiling;
        impl BlockShapes for Ceiling {
            fn solid(&self, pos: BlockPos) -> Option<bool> {
                Some(pos.y < 64 || pos.y >= 66)
            }
        }
        let world = Ceiling;
        let mut s = on_floor(0.5, 64.0, 0.5);
        let mut apex: f64 = 64.0;
        let mut bumped = false;
        let jump = MoveInput {
            jump: true,
            ..MoveInput::IDLE
        };
        for _ in 0..40 {
            s = step(s, &jump, &world);
            apex = apex.max(s.position.y);
            if s.position.y == 66.0 - PLAYER_HEIGHT && s.velocity.y == 0.0 {
                bumped = true;
            }
        }
        assert_eq!(apex, 66.0 - PLAYER_HEIGHT, "head clips at the ceiling");
        assert!(bumped, "velocity.y becomes 0 at the bump");
        // With jump released the player falls back and lands.
        let end = run_until(s, &idle(), &world, 100, |s| s.on_ground);
        assert!(end.on_ground, "falls back and lands");
        assert_eq!(end.position.y, 64.0);
    }

    #[test]
    fn jump_height_is_between_1_2_and_1_3_blocks() {
        let world = Floor { top: 64 };
        let mut s = on_floor(0.5, 64.0, 0.5);
        let jump = MoveInput {
            jump: true,
            ..MoveInput::IDLE
        };
        s = step(s, &jump, &world);
        let rising = s;
        assert!(!rising.on_ground, "airborne after jumping");
        // Jump held mid-air must not relaunch: velocity keeps integrating
        // gravity instead of snapping back to JUMP_VELOCITY.
        let s2 = step(rising, &jump, &world);
        assert_eq!(
            s2.velocity.y,
            JUMP_VELOCITY - GRAVITY * TICK_DT - GRAVITY * TICK_DT
        );

        let mut apex: f64 = 64.0;
        let mut s = rising;
        for _ in 0..100 {
            if s.on_ground {
                break;
            }
            apex = apex.max(s.position.y);
            s = step(s, &idle(), &world);
        }
        let height = apex - 64.0;
        assert!(
            (1.2..1.3).contains(&height),
            "jump height {height} not in 1.2..1.3"
        );
        assert_eq!(s.position.y, 64.0, "lands back on the floor");
    }

    #[test]
    fn fly_mode_has_no_gravity_and_moves_up_down_at_fly_speed() {
        let world = Floor { top: 64 };
        let mut s = on_floor(0.5, 100.0, 0.5);
        let toggle_up = MoveInput {
            fly_toggle: true,
            up: 1,
            ..MoveInput::IDLE
        };
        s = step(s, &toggle_up, &world);
        assert!(s.flying);
        assert_eq!(s.velocity.y, FLY_SPEED);
        assert_eq!(s.position.y, 100.0 + FLY_SPEED * TICK_DT);

        let hover = MoveInput::IDLE;
        for _ in 0..50 {
            s = step(s, &hover, &world);
        }
        assert_eq!(
            s.position.y,
            100.0 + FLY_SPEED * TICK_DT,
            "no gravity in fly"
        );
        assert_eq!(s.velocity.y, 0.0);
        assert!(!s.on_ground);

        let down = MoveInput {
            up: -1,
            ..MoveInput::IDLE
        };
        s = step(s, &down, &world);
        assert_eq!(s.velocity.y, -FLY_SPEED);
        assert_eq!(
            s.position.y,
            100.0 + FLY_SPEED * TICK_DT - FLY_SPEED * TICK_DT
        );
    }

    #[test]
    fn fly_mode_collides_with_blocks() {
        // Ceiling at y >= 66 blocks the rise.
        struct Ceiling;
        impl BlockShapes for Ceiling {
            fn solid(&self, pos: BlockPos) -> Option<bool> {
                Some(pos.y >= 66)
            }
        }
        let world = Ceiling;
        let mut s = PlayerState {
            position: DVec3::new(0.5, 64.0, 0.5),
            velocity: DVec3::ZERO,
            on_ground: false,
            flying: true,
        };
        let up = MoveInput {
            up: 1,
            ..MoveInput::IDLE
        };
        for _ in 0..20 {
            s = step(s, &up, &world);
        }
        assert_eq!(s.position.y, 66.0 - PLAYER_HEIGHT, "head clips at ceiling");
        assert_eq!(s.velocity.y, 0.0);
    }

    #[test]
    fn fly_toggle_flips_once_per_flagged_input() {
        let world = Floor { top: 64 };
        let mut s = on_floor(0.5, 64.0, 0.5);
        assert!(!s.flying);
        let flagged = MoveInput {
            fly_toggle: true,
            ..MoveInput::IDLE
        };
        s = step(s, &flagged, &world);
        assert!(s.flying, "first flagged input turns fly on");
        s = step(s, &flagged, &world);
        assert!(!s.flying, "the next flagged input turns it off again");
        s = step(s, &flagged, &world);
        assert!(s.flying);
        s = step(s, &idle(), &world);
        assert!(s.flying, "unflagged inputs leave the mode alone");
    }

    #[test]
    fn block_overlap_follows_the_touching_boundary_rule() {
        // A player box resting exactly on a block's top face.
        let min = DVec3::new(0.2, 64.0, 0.2);
        let max = DVec3::new(0.8, 65.8, 0.8);
        let below = BlockPos { x: 0, y: 63, z: 0 };
        assert!(
            !block_overlaps_box(below, min, max),
            "touching the top face is not an overlap"
        );
        // The cell the feet stand in, one up: overlaps on all axes.
        assert!(block_overlaps_box(BlockPos { x: 0, y: 64, z: 0 }, min, max));
        // Beside and above: misses on at least one axis each time.
        assert!(!block_overlaps_box(
            BlockPos { x: 1, y: 64, z: 0 },
            min,
            max
        ));
        assert!(!block_overlaps_box(
            BlockPos {
                x: -1,
                y: 66,
                z: -1
            },
            min,
            max
        ));
    }

    #[test]
    fn diagonal_speed_equals_straight_speed() {
        let world = Floor { top: 64 };
        let straight = step(
            on_floor(0.0, 64.0, 0.0),
            &walking(std::f32::consts::FRAC_PI_2),
            &world,
        );
        let diagonal = step(
            on_floor(0.0, 64.0, 0.0),
            &walking(std::f32::consts::FRAC_PI_4),
            &world,
        );
        let d = DVec3::new(diagonal.position.x, 0.0, diagonal.position.z).length();
        assert!(
            (straight.position.x - d).abs() < 1e-9,
            "straight {} vs diagonal {d}",
            straight.position.x
        );
    }

    #[test]
    fn unloaded_blocks_freeze_the_player_completely() {
        // Loaded below y = 64, unloaded above: falling is fine, but the
        // swept box reaches y >= 64 and the step must be a no-op.
        struct HalfLoaded;
        impl BlockShapes for HalfLoaded {
            fn solid(&self, pos: BlockPos) -> Option<bool> {
                (pos.y < 64).then_some(pos.y < 63)
            }
        }
        let world = HalfLoaded;
        let start = PlayerState {
            position: DVec3::new(0.5, 63.5, 0.5),
            velocity: DVec3::new(0.0, -20.0, 0.0),
            on_ground: false,
            flying: false,
        };
        let after = step(start, &idle(), &world);
        assert_eq!(after, start, "state returned unchanged, gravity included");

        // Walk mode too: the box spans y in 63.5..65.3, touching the
        // unloaded layer, so even wish velocity must not apply.
        let walk = step(start, &walking(0.0), &world);
        assert_eq!(walk, start);
    }

    #[test]
    fn scenarios_match_far_from_the_origin() {
        let offset = 10_000_000.0;
        let neg = -10_000_000.0;

        // Falling lands with feet exactly on the floor at both offsets.
        for base in [offset, neg] {
            let world = Floor { top: 64 };
            let start = PlayerState {
                position: DVec3::new(base, 74.0, base),
                velocity: DVec3::ZERO,
                on_ground: false,
                flying: false,
            };
            let end = run_until(start, &idle(), &world, 100, |s| s.on_ground);
            assert_eq!(end.position.y, 64.0);
            assert_eq!(end.position.x, base, "x unchanged at |x| = 1e7");
            assert_eq!(end.position.z, base);
        }

        // Wall flush, relative to the offset.
        for base in [offset, neg] {
            let world = Wall {
                x_min: base as i64 + 5,
            };
            let mut far = on_floor(base, 64.0, 0.5);
            let east = walking(std::f32::consts::FRAC_PI_2);
            for _ in 0..40 {
                far = step(far, &east, &world);
            }
            assert_eq!(
                far.position.x + HALF_WIDTH,
                base + 5.0,
                "flush with a wall at |x| = 1e7"
            );
        }

        // Jump apex is the same relative height.
        let world = Floor { top: 64 };
        let mut near = on_floor(0.5, 64.0, 0.5);
        let mut afar = on_floor(neg, 64.0, 0.5);
        let jump = MoveInput {
            jump: true,
            ..MoveInput::IDLE
        };
        near = step(near, &jump, &world);
        afar = step(afar, &jump, &world);
        let mut apex_near: f64 = 64.0;
        let mut apex_far: f64 = 64.0;
        for _ in 0..100 {
            if !near.on_ground {
                apex_near = apex_near.max(near.position.y);
                near = step(near, &idle(), &world);
            }
            if !afar.on_ground {
                apex_far = apex_far.max(afar.position.y);
                afar = step(afar, &idle(), &world);
            }
        }
        assert!((apex_near - apex_far).abs() < 1e-9);
    }

    /// A tiny inline LCG — deterministic, no dependency.
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }

        fn f32_unit(&mut self) -> f32 {
            (self.below(1 << 24) as f32) / ((1 << 24) as f32)
        }
    }

    /// Random-but-deterministic terrain: a heightmap hashed from (x, z).
    struct FuzzWorld {
        seed: u64,
    }

    impl FuzzWorld {
        fn height(&self, x: i64, z: i64) -> i64 {
            let mut h = self.seed ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            h ^= (z as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            h ^= h >> 33;
            h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            h ^= h >> 33;
            60 + (h % 9) as i64
        }
    }

    impl BlockShapes for FuzzWorld {
        fn solid(&self, pos: BlockPos) -> Option<bool> {
            Some(pos.y <= self.height(pos.x, pos.z))
        }
    }

    #[test]
    fn fuzz_two_thousand_steps_never_overlap_a_solid_block() {
        let world = FuzzWorld { seed: 0x00F0_2A11 };
        let mut rng = Lcg(0x5EED_1234);
        let mut s = PlayerState {
            position: DVec3::new(500.5, 200.0, -300.5),
            velocity: DVec3::ZERO,
            on_ground: false,
            flying: false,
        };
        for i in 0..2000 {
            let input = MoveInput {
                forward: rng.below(3) as i8 - 1,
                strafe: rng.below(3) as i8 - 1,
                up: rng.below(3) as i8 - 1,
                jump: rng.below(2) == 0,
                sprint: rng.below(2) == 0,
                fly_toggle: rng.below(50) == 0,
                yaw: rng.f32_unit() * std::f32::consts::TAU - std::f32::consts::PI,
            };
            s = step(s, &input, &world);
            let b = s.aabb();
            let (x0, x1) = block_span(b.min.x, b.max.x);
            let (y0, y1) = block_span(b.min.y, b.max.y);
            let (z0, z1) = block_span(b.min.z, b.max.z);
            for y in y0..=y1 {
                for z in z0..=z1 {
                    for x in x0..=x1 {
                        if world.solid(BlockPos { x, y, z }) == Some(true) {
                            panic!(
                                "step {i}: box {b:?} overlaps solid block ({x},{y},{z}), input {input:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Wall contacts land exactly on the face, never an ulp inside it (an
    /// ulp inside would put the wall column in the box's static span and
    /// snag the next fall), at small and large coordinates: fly into walls
    /// at random positions and yaws, then drop and land on the floor.
    #[test]
    fn wall_contacts_never_stick() {
        struct Room(i64);
        impl BlockShapes for Room {
            // Floor below y = 0; walls at x >= 5, x <= -6, z >= 7, z <= -8,
            // all shifted by the offset on X and Z.
            fn solid(&self, p: BlockPos) -> Option<bool> {
                let (x, z) = (p.x - self.0, p.z - self.0);
                Some(p.y < 0 || x >= 5 || x <= -6 || z >= 7 || z <= -8)
            }
        }
        for offset in [0_i64, 1_000, 123_457, -9_876_543, 10_000_000] {
            let o = offset as f64;
            let mut seed: u64 = 12345;
            let mut rnd = || {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (seed >> 11) as f64 / (1u64 << 53) as f64
            };
            for _ in 0..5000 {
                let mut state = PlayerState {
                    position: DVec3::new(
                        o - 4.7 + rnd() * 9.4,
                        3.0 + rnd() * 5.0,
                        o - 6.7 + rnd() * 13.4,
                    ),
                    velocity: DVec3::ZERO,
                    on_ground: false,
                    flying: true,
                };
                let yaw = (rnd() * std::f64::consts::TAU) as f32;
                let fly = MoveInput {
                    forward: 1,
                    sprint: true,
                    yaw,
                    ..MoveInput::IDLE
                };
                for _ in 0..20 {
                    state = step(state, &fly, &Room(offset));
                }
                let b = state.aabb();
                assert!(
                    b.max.x <= o + 5.0
                        && b.min.x >= o - 5.0
                        && b.max.z <= o + 7.0
                        && b.min.z >= o - 7.0,
                    "box past a wall: {b:?}"
                );
                let land = MoveInput {
                    fly_toggle: true,
                    yaw,
                    ..MoveInput::IDLE
                };
                state = step(state, &land, &Room(offset));
                let idle = MoveInput {
                    yaw,
                    ..MoveInput::IDLE
                };
                for _ in 0..60 {
                    state = step(state, &idle, &Room(offset));
                }
                assert_eq!(state.position.y, 0.0, "stuck on a wall at {state:?}");
                assert!(state.on_ground);
            }
        }
    }
}
