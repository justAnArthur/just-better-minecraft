//! The playground debug world (task 2.12, `JBM_WORLD=playground`): a
//! fixed, deterministic layout around spawn instead of noise — flat
//! plane, staircase, cliff, pillar, pit, water pool, checkerboard, ridge
//! and dome, all at fixed offsets. A density graph cannot express it
//! (a checkerboard of two materials, per-feature tops), so it is a small
//! pure Rust model behind the same [`Generator`](super::Generator)
//! interface: every method here is a pure function of `(x, z)`, so
//! `generate` and `sample_lod` agree by construction and the same seed
//! always produces the same world (the layout ignores the seed; a
//! different seed changes nothing, which the determinism tests pin).
//!
//! Layout (block coordinates, all features in the z band 0..=95 north of
//! the spawn plane; spawn sits at `(-16.5, 66.5, 48.5)` facing them):
//!
//! | Feature | x × z | Shape |
//! |---|---|---|
//! | plane | everywhere else | stone to y=64, grass top |
//! | checkerboard | [-64,-33] × [0,31] | stone/glass alternating per (x+z)&1 |
//! | cliff | [-32,-1] × [0,31] | 64-block vertical wall, top y=128 |
//! | staircase | [0,63] × [0,31] | 1-block steps, y=65..=96 |
//! | pillar | (64, 0) | 1×1, 128 tall, top y=192 |
//! | pit | [72,79] × [0,7] | 32 deep (floor y=32), 8 wide |
//! | pool + sand rim | [96,127] × [0,31] (+2 rim) | floor y=60, water to sea level 63 |
//! | ridge | [144,207] × [32,95] | 45° diagonal stairs, rising to y=127 |
//! | dome | [208,272] × [31,95] | hemisphere, radius 32, on the plane |

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;

use super::SEA_LEVEL;

/// The plane's surface height: grass at y=64 over stone.
pub(super) const PLANE_HEIGHT: i64 = 64;

/// One playground column: its top solid block's y and material, and
/// whether water fills the space above it up to [`SEA_LEVEL`].
pub(super) struct Column {
    pub height: i64,
    pub top: BlockStateId,
    pub water: bool,
}

/// The playground column at world `(x, z)`. Pure; the feature table
/// above is the spec.
pub(super) fn column(x: i64, z: i64, blocks: &Blocks) -> Column {
    // The pool's two-block sand rim, then the pool itself.
    if (94..=129).contains(&x) && (-2..=33).contains(&z) {
        if (96..=127).contains(&x) && (0..=31).contains(&z) {
            return Column {
                height: 60,
                top: blocks.sand,
                water: true,
            };
        }
        return Column {
            height: PLANE_HEIGHT,
            top: blocks.sand,
            water: false,
        };
    }
    // The checkerboard: two materials alternating per column — greedy
    // meshing's worst case.
    if (-64..=-33).contains(&x) && (0..=31).contains(&z) {
        return Column {
            height: PLANE_HEIGHT,
            top: if (x + z).rem_euclid(2) == 0 {
                blocks.stone
            } else {
                blocks.glass
            },
            water: false,
        };
    }
    // The cliff: a 64-block vertical wall.
    if (-32..=-1).contains(&x) && (0..=31).contains(&z) {
        return Column {
            height: PLANE_HEIGHT + 64,
            top: blocks.stone,
            water: false,
        };
    }
    // The staircase: 1-block steps, two blocks deep each, y 65..=96.
    if (0..=63).contains(&x) && (0..=31).contains(&z) {
        return Column {
            height: PLANE_HEIGHT + 1 + x / 2,
            top: blocks.stone,
            water: false,
        };
    }
    // The pillar: one column, 128 tall.
    if x == 64 && z == 0 {
        return Column {
            height: PLANE_HEIGHT + 128,
            top: blocks.stone,
            water: false,
        };
    }
    // The pit: 32 deep, 8 wide.
    if (72..=79).contains(&x) && (0..=7).contains(&z) {
        return Column {
            height: PLANE_HEIGHT - 32,
            top: blocks.stone,
            water: false,
        };
    }
    // The ridge: 45° stairs along the diagonal, capped 64 above the plane.
    if (144..=207).contains(&x) && (32..=95).contains(&z) {
        let rise = (x - 144).min(z - 32).min(64);
        return Column {
            height: PLANE_HEIGHT + rise,
            top: blocks.stone,
            water: false,
        };
    }
    // The dome: a hemisphere of radius 32 on the plane, centred (240, 63).
    if (208..=272).contains(&x) && (31..=95).contains(&z) {
        let dx = x - 240;
        let dz = z - 63;
        let r2 = dx * dx + dz * dz;
        if r2 <= 32 * 32 {
            return Column {
                height: PLANE_HEIGHT + isqrt(32 * 32 - r2),
                top: blocks.stone,
                water: false,
            };
        }
    }
    // The plane.
    Column {
        height: PLANE_HEIGHT,
        top: blocks.grass_block,
        water: false,
    }
}

/// The block at world `(x, y, z)`: the column's material on top, stone
/// under it, water over a wet column up to sea level, air above. Shared
/// by `generate` (per block) and `sample_lod` (per sample), so the two
/// agree at every level by construction.
pub(super) fn state(x: i64, y: i64, z: i64, blocks: &Blocks) -> BlockStateId {
    let c = column(x, z, blocks);
    if y > c.height {
        if c.water && y <= SEA_LEVEL {
            return blocks.water;
        }
        return BlockStateId::AIR;
    }
    if y == c.height {
        return c.top;
    }
    blocks.stone
}

/// Integer square root (the dome's height only; the argument is small).
fn isqrt(n: i64) -> i64 {
    let mut root = n;
    while root * root > n {
        root = (root + n / root) / 2;
    }
    root
}

/// The spawn point: on the plane south of the features, facing them (the
/// client's yaw 0 looks toward -Z). Two above the surface, like the
/// other presets.
pub(super) fn spawn() -> [f64; 3] {
    [-16.5, (PLANE_HEIGHT + 2) as f64, 48.5]
}

#[cfg(test)]
mod tests {
    use super::*;
    use jbm_core::block::BlockRegistry;

    fn blocks() -> Blocks {
        let mut registry = BlockRegistry::new();
        Blocks::register(&mut registry)
    }

    #[test]
    fn isqrt_matches_the_perfect_squares() {
        for n in 0..=1024i64 {
            let root = isqrt(n);
            assert!(
                root * root <= n && (root + 1) * (root + 1) > n,
                "isqrt({n})"
            );
        }
    }

    /// Every feature sits where the table says, with the promised height
    /// and material.
    #[test]
    fn features_exist_at_their_fixed_offsets() {
        let blocks = blocks();
        let plane = column(-16, 48, &blocks);
        assert_eq!((plane.height, plane.top), (64, blocks.grass_block));
        // Checkerboard: neighbours alternate stone and glass.
        let a = column(-64, 0, &blocks);
        let b = column(-63, 0, &blocks);
        assert_eq!(a.top, blocks.stone);
        assert_eq!(b.top, blocks.glass);
        // Cliff, staircase ends, pillar, pit.
        assert_eq!(column(-32, 0, &blocks).height, 128);
        assert_eq!(column(0, 0, &blocks).height, 65);
        assert_eq!(column(63, 0, &blocks).height, 96);
        assert_eq!(column(64, 0, &blocks).height, 192);
        assert_eq!(column(72, 0, &blocks).height, 32);
        // Pool: water to sea level over a sand floor, sand rim around.
        assert_eq!(state(100, 63, 10, &blocks), blocks.water);
        assert_eq!(state(100, 60, 10, &blocks), blocks.sand);
        assert_eq!(column(95, 10, &blocks).top, blocks.sand);
        // Ridge: rises along the diagonal (its last full step is 63).
        assert_eq!(column(144, 32, &blocks).height, 64);
        assert_eq!(column(176, 64, &blocks).height, 96);
        assert_eq!(column(207, 95, &blocks).height, 127);
        // Dome: full height at the centre, plane at the footprint's rim.
        assert_eq!(column(240, 63, &blocks).height, 96);
        assert_eq!(column(272, 63, &blocks).height, 64);
    }

    /// The column model's layering: top material, stone below, water over
    /// a wet column, air above.
    #[test]
    fn state_layers_top_stone_water_air() {
        let blocks = blocks();
        assert_eq!(state(-16, 64, 48, &blocks), blocks.grass_block);
        assert_eq!(state(-16, 63, 48, &blocks), blocks.stone);
        assert_eq!(state(-16, 0, 48, &blocks), blocks.stone);
        assert_eq!(state(-16, 65, 48, &blocks), BlockStateId::AIR);
        // The pool column: sand floor, water up to sea level, air above.
        assert_eq!(state(100, 59, 10, &blocks), blocks.stone);
        assert_eq!(state(100, 60, 10, &blocks), blocks.sand);
        assert_eq!(state(100, 61, 10, &blocks), blocks.water);
        assert_eq!(state(100, 64, 10, &blocks), BlockStateId::AIR);
    }
}
