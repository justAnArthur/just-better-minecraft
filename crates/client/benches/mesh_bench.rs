//! Criterion benches for the mesher, group `mesh`. Cases:
//!
//! - `terrain_like`: a heightmap-filled cube (stone below, dirt layers,
//!   grass top, height 4..28) with air above, stone neighbors below and
//!   air on the sides — the typical terrain cube.
//! - `checkerboard`: the worst case (16384 x 6 faces, no merging).
//! - `uniform_stone_air_neighbors`: the non-fast uniform path.
//! - `padded_build`: only the padded 34^3 build from 27 cubes.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::mesh::greedy::{self, MeshInput, neighbor_index};
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
use jbm_core::cube::Cube;

fn setup() -> (BlockAppearance, builtin::Blocks) {
    let mut registry = BlockRegistry::new();
    let blocks = builtin::Blocks::register(&mut registry);
    (BlockAppearance::new(&registry), blocks)
}

/// Deterministic heightmap: smooth-ish steps varying 4..28 inside the cube.
fn height_at(x: u8, z: u8) -> u8 {
    ((x / 3) * 2 + (z / 4) + 4).clamp(4, 28)
}

fn terrain_like_cube(blocks: &builtin::Blocks) -> Cube {
    Cube::from_fn(|p| {
        let height = height_at(p.x, p.z);
        if p.y > height {
            BlockStateId::AIR
        } else if p.y == height {
            blocks.grass_block
        } else if p.y + 3 >= height {
            blocks.dirt
        } else {
            blocks.stone
        }
    })
}

/// Neighbors for `center`: `filler` in every slot, or all missing.
fn uniform_neighbors(filler: Option<Arc<Cube>>) -> [Option<Arc<Cube>>; 26] {
    std::array::from_fn(|_| filler.clone())
}

/// 9 neighbors above/below at dy, air above and stone below; the 8 side
/// slots air too (matches the terrain_like case description).
fn terrain_neighbors(air: &Arc<Cube>, stone: &Arc<Cube>) -> [Option<Arc<Cube>>; 26] {
    let mut neighbors = std::array::from_fn(|_| None);
    for dy in -1..=1 {
        for dz in -1..=1 {
            for dx in -1..=1 {
                if (dx, dy, dz) == (0, 0, 0) {
                    continue;
                }
                let filler = if dy < 0 { stone } else { air };
                neighbors[neighbor_index(dx, dy, dz)] = Some(Arc::clone(filler));
            }
        }
    }
    neighbors
}

fn bench(c: &mut Criterion) {
    let (appearance, blocks) = setup();
    let air = Arc::new(Cube::default());
    let stone = Arc::new(Cube::filled(blocks.stone));

    let terrain_input = MeshInput {
        center: Arc::new(terrain_like_cube(&blocks)),
        neighbors: terrain_neighbors(&air, &stone),
    };
    let checker_input = MeshInput {
        center: Arc::new(Cube::from_fn(|p| {
            if (p.x + p.y + p.z) % 2 == 0 {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        })),
        neighbors: uniform_neighbors(Some(Arc::clone(&air))),
    };
    let uniform_input = MeshInput {
        center: Arc::new(Cube::filled(blocks.stone)),
        neighbors: uniform_neighbors(Some(Arc::clone(&air))),
    };
    let padded_input = MeshInput {
        center: Arc::new(terrain_like_cube(&blocks)),
        neighbors: uniform_neighbors(Some(Arc::clone(&stone))),
    };

    let mut group = c.benchmark_group("mesh");
    group.bench_function("terrain_like", |b| {
        b.iter(|| {
            black_box(greedy::mesh(
                black_box(&terrain_input),
                black_box(&appearance),
            ))
        })
    });
    group.bench_function("checkerboard", |b| {
        b.iter(|| {
            black_box(greedy::mesh(
                black_box(&checker_input),
                black_box(&appearance),
            ))
        })
    });
    group.bench_function("uniform_stone_air_neighbors", |b| {
        b.iter(|| {
            black_box(greedy::mesh(
                black_box(&uniform_input),
                black_box(&appearance),
            ))
        })
    });
    group.bench_function("padded_build", |b| {
        b.iter(|| {
            black_box(greedy::build_padded(
                black_box(&padded_input),
                black_box(&appearance),
            ))
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
