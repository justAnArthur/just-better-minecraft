//! Task 2.13 B: the bitmask sweep against real generated cubes. The
//! client-crate oracle test pins it on synthetic cubes; this one meshes
//! real overworld and playground cubes (through the server's generator,
//! which the client crate cannot depend on) and asserts the same
//! bit-for-bit equality against the verbatim pre-2.13 reference sweep.
#![cfg(not(feature = "cube16"))]

use std::sync::Arc;

use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::mesh::oracle::reference_mesh_padded;
use jbm_client::mesh::{self, MeshInput, neighbor_index};
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::CubePos;
use jbm_core::cube::Cube;
use jbm_server::DEFAULT_SEED;
use jbm_server::terrain::Generator;
use jbm_server::worldgen::Preset;

/// Meshes `center` with its real neighbors from `cube_at` and compares
/// the sweep against the reference at `level`.
fn check(
    appearance: &BlockAppearance,
    cube_at: &dyn Fn(CubePos) -> Cube,
    center: CubePos,
    level: u8,
) {
    let mut neighbors = std::array::from_fn(|_| None);
    for dy in -1i32..=1 {
        for dz in -1i32..=1 {
            for dx in -1i32..=1 {
                if (dx, dy, dz) == (0, 0, 0) {
                    continue;
                }
                neighbors[neighbor_index(dx, dy, dz)] = Some(Arc::new(cube_at(CubePos {
                    x: center.x + dx,
                    y: center.y + dy,
                    z: center.z + dz,
                })));
            }
        }
    }
    let input = MeshInput {
        center: Arc::new(cube_at(center)),
        neighbors,
        level,
    };
    let padded = mesh::build_padded(&input, appearance);
    let got = mesh::greedy::mesh_padded(&padded, appearance, level);
    let expected = reference_mesh_padded(&padded, appearance, level);
    assert!(
        got.faces == expected.faces,
        "{center:?} (level {level}): face list drifted"
    );
    assert_eq!(got.ranges, expected.ranges, "{center:?}: ranges drifted");
    assert_eq!(got.skirts, expected.skirts, "{center:?}: skirts drifted");
    assert_eq!(got.lids, expected.lids, "{center:?}: lids drifted");
}

#[test]
fn bitmask_sweep_matches_the_reference_on_real_cubes() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);

    // Overworld: a 5x5x3 block of real cubes around the spawn column, at
    // LOD0 and (the same downsampled cubes are not needed — the sweep's
    // level only switches the seam rules) at levels 0..=2.
    let overworld = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let spawn = overworld.spawn();
    let center0 = CubePos {
        x: (spawn[0] as i64 / 32) as i32,
        y: 2,
        z: (spawn[2] as i64 / 32) as i32,
    };
    let over_cube = |pos: CubePos| overworld.generate(pos);
    for level in 0..=2u8 {
        for dy in -1..=1i32 {
            for dz in -2..=2i32 {
                for dx in -2..=2i32 {
                    check(
                        &appearance,
                        &over_cube,
                        CubePos {
                            x: center0.x + dx,
                            y: center0.y + dy,
                            z: center0.z + dz,
                        },
                        level,
                    );
                }
            }
        }
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
    let play_cube = |pos: CubePos| playground.generate(pos);
    for level in 0..=2u8 {
        for dy in -1..=1i32 {
            for dz in -2..=2i32 {
                for dx in -2..=2i32 {
                    check(
                        &appearance,
                        &play_cube,
                        CubePos {
                            x: pcenter.x + dx,
                            y: pcenter.y + dy,
                            z: pcenter.z + dz,
                        },
                        level,
                    );
                }
            }
        }
    }
}
