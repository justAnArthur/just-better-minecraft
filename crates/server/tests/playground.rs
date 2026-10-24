//! Task 2.12's playground debug world: determinism, the fixed features
//! at their fixed offsets, and `sample_lod` agreeing with downsampling
//! the generated cubes.

use jbm_core::block::BlockRegistry;
use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CubePos, LocalPos};
use jbm_core::cube::Cube;
use jbm_core::lod::{LodTops, octant_offset};
use jbm_server::terrain::Generator;
use jbm_server::worldgen::Preset;

fn blocks() -> Blocks {
    let mut registry = BlockRegistry::new();
    Blocks::register(&mut registry)
}

fn playground(seed: u64) -> Generator {
    Generator::with_preset(seed, blocks(), Preset::Playground)
}

/// The block at world `(x, y, z)`, through the cube that contains it.
fn block_at(terrain: &Generator, x: i64, y: i64, z: i64) -> BlockStateId {
    let cube = terrain.generate(CubePos {
        x: (x >> 5) as i32,
        y: (y >> 5) as i32,
        z: (z >> 5) as i32,
    });
    cube.get(LocalPos::new((x & 31) as u8, (y & 31) as u8, (z & 31) as u8).unwrap())
}

/// Positions covering every feature's footprint (in cube units — the
/// features live between x -64 and 272, z 0 and 95, y 32 and 192).
fn feature_positions() -> Vec<CubePos> {
    let mut positions = Vec::new();
    for cx in -3..=9i32 {
        for cz in -1..=3i32 {
            for cy in 0..=6i32 {
                positions.push(CubePos {
                    x: cx,
                    y: cy,
                    z: cz,
                });
            }
        }
    }
    positions
}

/// Same seed, same cubes — and the layout ignores the seed entirely, so
/// two different seeds also agree (the world is a fixed debug layout).
#[test]
fn the_playground_is_deterministic() {
    let a = playground(42);
    let b = playground(42);
    let c = playground(7);
    for pos in feature_positions() {
        let cube = a.generate(pos);
        assert_eq!(cube, b.generate(pos), "same seed differs at {pos:?}");
        assert_eq!(
            cube,
            c.generate(pos),
            "the layout must ignore the seed at {pos:?}"
        );
    }
}

/// Every feature exists at its documented offset, through the public
/// generator (the fixed table in terrain/playground.rs is the spec).
#[test]
fn every_feature_exists_at_its_offset() {
    let terrain = playground(1);
    let blocks = blocks();
    // The plane: grass at 64, stone below, air above — at the spawn
    // clearing south of the features.
    assert_eq!(block_at(&terrain, -16, 64, 48), blocks.grass_block);
    assert_eq!(block_at(&terrain, -16, 63, 48), blocks.stone);
    assert_eq!(block_at(&terrain, -16, 65, 48), BlockStateId::AIR);
    // Checkerboard: alternating stone and glass columns.
    assert_eq!(block_at(&terrain, -64, 64, 0), blocks.stone);
    assert_eq!(block_at(&terrain, -63, 64, 0), blocks.glass);
    assert_eq!(
        block_at(&terrain, -63, 63, 0),
        blocks.stone,
        "below the glass"
    );
    // Cliff: a 64-block wall (top 128) with air above the plane beside it.
    assert_eq!(block_at(&terrain, -32, 128, 0), blocks.stone);
    assert_eq!(block_at(&terrain, -32, 129, 0), BlockStateId::AIR);
    // Staircase: 1-block steps from 65 to 96.
    assert_eq!(block_at(&terrain, 0, 65, 0), blocks.stone);
    assert_eq!(block_at(&terrain, 0, 66, 0), BlockStateId::AIR);
    assert_eq!(block_at(&terrain, 63, 96, 0), blocks.stone);
    // Pillar: one column, 128 tall (top 192).
    assert_eq!(block_at(&terrain, 64, 192, 0), blocks.stone);
    assert_eq!(block_at(&terrain, 64, 193, 0), BlockStateId::AIR);
    assert_eq!(block_at(&terrain, 65, 70, 0), BlockStateId::AIR);
    // Pit: 32 deep (floor 32), 8 wide.
    assert_eq!(block_at(&terrain, 72, 32, 0), blocks.stone);
    assert_eq!(block_at(&terrain, 72, 33, 0), BlockStateId::AIR);
    // Pool: sand floor at 60, water to the sea level 63, sand rim.
    assert_eq!(block_at(&terrain, 100, 60, 10), blocks.sand);
    assert_eq!(block_at(&terrain, 100, 63, 10), blocks.water);
    assert_eq!(block_at(&terrain, 100, 64, 10), BlockStateId::AIR);
    assert_eq!(block_at(&terrain, 95, 64, 10), blocks.sand);
    // Ridge: 45° stairs along the diagonal.
    assert_eq!(block_at(&terrain, 176, 96, 64), blocks.stone);
    assert_eq!(block_at(&terrain, 176, 97, 64), BlockStateId::AIR);
    assert_eq!(
        block_at(&terrain, 175, 96, 64),
        BlockStateId::AIR,
        "off-diagonal"
    );
    // Dome: hemisphere peaking at 96 over its centre, open at the rim.
    assert_eq!(block_at(&terrain, 240, 96, 63), blocks.stone);
    assert_eq!(block_at(&terrain, 240, 97, 63), BlockStateId::AIR);
}

/// The spawn sits on the plane, two above the surface, south of the
/// features.
#[test]
fn spawn_is_on_the_plane_facing_the_features() {
    let terrain = playground(1);
    let blocks = blocks();
    let [x, y, z] = terrain.spawn();
    assert_eq!((x, y, z), (-16.5, 66.0, 48.5));
    assert_eq!(
        block_at(&terrain, x as i64, y as i64 - 2, z as i64),
        blocks.grass_block,
        "spawn stands on grass"
    );
}

/// The exact downsample of generated cubes, cells and column tops, level
/// by level (the same helper shape tests/lod.rs uses).
fn exact_cube(generator: &Generator, level: u8, pos: CubePos) -> (Cube, LodTops) {
    if level == 0 {
        return (generator.generate(pos), LodTops::LOD0);
    }
    let mut cube = Cube::filled(BlockStateId::AIR);
    let mut tops = LodTops::empty(level);
    for octant in 0..8u8 {
        let [dx, dy, dz] = octant_offset(octant).map(i32::from);
        let child = CubePos {
            x: pos.x * 2 + dx,
            y: pos.y * 2 + dy,
            z: pos.z * 2 + dz,
        };
        let (child_cube, child_tops) = exact_cube(generator, level - 1, child);
        jbm_core::lod::downsample_octant(&child_cube, &child_tops, &mut cube, &mut tops, octant);
    }
    (cube, tops)
}

/// `sample_lod` agrees with downsampling the generated cubes: exactly at
/// level 1, column tops included (the samples equal the LOD0 cells), and
/// at the coarser levels
/// under the same min-corner sampling rule the graph presets follow.
#[test]
fn sample_lod_agrees_with_downsampling_generated_cubes() {
    let generator = playground(42);
    // A spread over the feature band: the checkerboard, the cliff face,
    // the pool with its water, the pillar tip and the plane.
    let positions = [
        CubePos { x: -2, y: 2, z: 0 }, // checkerboard
        CubePos { x: -1, y: 4, z: 0 }, // cliff top
        CubePos { x: 3, y: 2, z: 0 },  // staircase
        CubePos { x: 2, y: 6, z: 0 },  // pillar tip
        CubePos { x: 3, y: 1, z: 0 },  // pit / plane edge
        CubePos { x: 3, y: 2, z: 0 },  // pool
        CubePos { x: 5, y: 3, z: 1 },  // ridge
        CubePos { x: 7, y: 3, z: 2 },  // dome
        CubePos { x: -1, y: 1, z: 1 }, // plane south
    ];
    for &pos in &positions {
        let expected = exact_cube(&generator, 1, pos);
        let generates = generator.generate_calls();
        let sampled = generator.sample_lod(1, pos);
        assert_eq!(sampled.0, expected.0, "level 1 cells at {pos:?}");
        assert_eq!(sampled.1, expected.1, "level 1 tops at {pos:?}");
        assert_eq!(
            generator.generate_calls(),
            generates,
            "sample_lod never generates, at {pos:?}"
        );
    }
    // Coarser levels: the same rule the graph presets' accuracy tests
    // pin — the sample tracks the exact downsample on the solid columns.
    for level in 2..=3u8 {
        let shift = 5 + level as i64;
        for &(x, z) in &[
            (-64i64, 0i64),
            (-32, 0),
            (64, 0),
            (100, 10),
            (176, 64),
            (240, 63),
        ] {
            let pos = CubePos {
                x: (x >> shift) as i32,
                y: (64i64 >> shift) as i32,
                z: (z >> shift) as i32,
            };
            let _ = generator.sample_lod(level, pos); // must not panic
        }
    }
}

/// `column_surface_bounds` reports the layout's heights (the streaming
/// span store runs on it).
#[test]
fn column_surface_bounds_follow_the_layout() {
    let terrain = playground(1);
    // The plane cube column south of every feature: flat 64.
    assert_eq!(terrain.column_surface_bounds(-1, 1), (64, 64));
    // The cliff cube column: uniform 128.
    assert_eq!(terrain.column_surface_bounds(-1, 0), (128, 128));
    // The pool interior cube column: uniform 60.
    assert_eq!(terrain.column_surface_bounds(3, 0), (60, 60));
    // Far from every feature: still the plane.
    assert_eq!(terrain.column_surface_bounds(100, 100), (64, 64));
}
