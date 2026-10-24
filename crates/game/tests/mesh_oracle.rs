//! Task 2.13 B: the bitmask sweep against real generated cubes. The
//! client-crate oracle test pins it on synthetic cubes; this one meshes
//! real overworld and playground cubes (through the server's generator,
//! which the client crate cannot depend on) and asserts the same
//! bit-for-bit equality against the pre-2.13 reference sweep. Levels >= 1
//! mesh the server's worldgen-sampled LOD cubes with their column tops
//! (task 2.14), so partial tops and their step faces are covered.
#![cfg(not(feature = "cube16"))]

use std::sync::Arc;

use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::mesh::oracle::reference_mesh_padded;
use jbm_client::mesh::{self, MeshInput, MeshTops, SIDE_OFFSETS, neighbor_index};
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::CubePos;
use jbm_core::cube::Cube;
use jbm_core::lod::LodTops;
use jbm_server::DEFAULT_SEED;
use jbm_server::terrain::Generator;
use jbm_server::worldgen::Preset;

/// Meshes `center` with its real neighbors (and, at level >= 1, the
/// column tops of it and its side neighbors) from `cube_at` and compares
/// the sweep against the reference at `level`. Returns whether the input
/// had partial column tops.
fn check(
    appearance: &BlockAppearance,
    cube_at: &dyn Fn(CubePos) -> (Cube, LodTops),
    center: CubePos,
    level: u8,
) -> bool {
    let at = |dx: i32, dy: i32, dz: i32| {
        cube_at(CubePos {
            x: center.x + dx,
            y: center.y + dy,
            z: center.z + dz,
        })
    };
    let mut neighbors = std::array::from_fn(|_| None);
    for dy in -1i32..=1 {
        for dz in -1i32..=1 {
            for dx in -1i32..=1 {
                if (dx, dy, dz) == (0, 0, 0) {
                    continue;
                }
                neighbors[neighbor_index(dx, dy, dz)] = Some(Arc::new(at(dx, dy, dz).0));
            }
        }
    }
    let (center_cube, center_tops) = at(0, 0, 0);
    let tops = if level == 0 {
        MeshTops::default()
    } else {
        MeshTops {
            center: Some(Arc::new(center_tops)),
            sides: SIDE_OFFSETS.map(|(dx, dz)| Some(Arc::new(at(dx, 0, dz).1))),
        }
    };
    let input = MeshInput {
        center: Arc::new(center_cube),
        neighbors,
        level,
        tops,
    };
    let padded = mesh::build_padded(&input, appearance);
    let heights = mesh::build_padded_heights(&input, &padded);
    let got = mesh::greedy::mesh_padded(&padded, appearance, level, heights.as_ref());
    let expected = reference_mesh_padded(&padded, appearance, level, heights.as_ref());
    assert!(
        got.faces == expected.faces,
        "{center:?} (level {level}): face list drifted"
    );
    assert_eq!(got.ranges, expected.ranges, "{center:?}: ranges drifted");
    assert_eq!(got.skirts, expected.skirts, "{center:?}: skirts drifted");
    assert_eq!(got.lids, expected.lids, "{center:?}: lids drifted");
    heights.is_some()
}

#[test]
fn bitmask_sweep_matches_the_reference_on_real_cubes() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);

    // Overworld: a 5x5x3 block of real cubes around the spawn column, at
    // LOD0 (generated cubes) and levels 1..=2 (sampled LOD cubes).
    let overworld = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let spawn = overworld.spawn();
    let center0 = CubePos {
        x: (spawn[0] as i64 / 32) as i32,
        y: 2,
        z: (spawn[2] as i64 / 32) as i32,
    };
    for level in 0..=2u8 {
        let over_cube = |pos: CubePos| {
            if level == 0 {
                (overworld.generate(pos), LodTops::LOD0)
            } else {
                overworld.sample_lod(level, pos)
            }
        };
        let center = lod_center(center0, level);
        let mut partial = 0;
        for dy in -1..=1i32 {
            for dz in -2..=2i32 {
                for dx in -2..=2i32 {
                    partial += usize::from(check(
                        &appearance,
                        &over_cube,
                        CubePos {
                            x: center.x + dx,
                            y: center.y + dy,
                            z: center.z + dz,
                        },
                        level,
                    ));
                }
            }
        }
        assert_eq!(partial > 0, level > 0, "level {level}: partial-top cubes");
    }

    // Playground (task 2.12's fixed debug layout): the same block around
    // its spawn.
    let playground = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Playground);
    let pspawn = playground.spawn();
    let pcenter = CubePos {
        x: (pspawn[0] as i64 / 32) as i32,
        y: 2,
        z: (pspawn[2] as i64 / 32) as i32,
    };
    for level in 0..=2u8 {
        let play_cube = |pos: CubePos| {
            if level == 0 {
                (playground.generate(pos), LodTops::LOD0)
            } else {
                playground.sample_lod(level, pos)
            }
        };
        let center = lod_center(pcenter, level);
        let mut partial = 0;
        for dy in -1..=1i32 {
            for dz in -2..=2i32 {
                for dx in -2..=2i32 {
                    partial += usize::from(check(
                        &appearance,
                        &play_cube,
                        CubePos {
                            x: center.x + dx,
                            y: center.y + dy,
                            z: center.z + dz,
                        },
                        level,
                    ));
                }
            }
        }
        assert_eq!(partial > 0, level > 0, "level {level}: partial-top cubes");
    }
}

/// The level-`level` cube containing LOD0 cube `pos`.
fn lod_center(pos: CubePos, level: u8) -> CubePos {
    CubePos {
        x: pos.x >> level,
        y: pos.y >> level,
        z: pos.z >> level,
    }
}
