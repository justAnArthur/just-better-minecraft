//! Criterion benches for the mesher (group `mesh`) and the face-page
//! suballocator (group `suballoc`). Mesh cases:
//!
//! - `terrain_like`: a heightmap-filled cube (stone below, dirt layers,
//!   grass top, height 4..28) with air above, stone neighbors below and
//!   air on the sides — the typical terrain cube.
//! - `checkerboard`: the worst case (16384 x 6 faces, no merging).
//! - `uniform_stone_air_neighbors`: the non-fast uniform path.
//! - `padded_build`: only the padded 34^3 build from 27 cubes.
//! - `terrain_like_lod2`: the `terrain_like` cells meshed as a level-2
//!   ring cube (octant splits, skirts, lids and — task 2.14 — partial
//!   column tops, a different one in nearly every column: the worst case
//!   for merging).
//! - `terrain_like_lod2_full_tops`: the same cube with every column top
//!   full — the pre-2.14 workload, so the fill-height path's fixed cost
//!   shows on its own.
//!
//! Suballoc case:
//! - `churn_10k`: the seeded 10,000-operation alloc/free churn from the
//!   suballocator unit tests, without the invariant checks.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::mesh::{self, MeshInput, MeshTops, neighbor_index};
use jbm_client::render::suballoc;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
use jbm_core::cube::Cube;
use jbm_core::lod::LodTops;

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

/// Level-2 column tops for [`terrain_like_cube`]: each column's top cell
/// (the one holding its grass) gets a deterministic partial height
/// 1..=4; a lower half under a filled upper half is full.
fn terrain_like_tops() -> LodTops {
    let mut tops = LodTops::empty(2);
    let half = jbm_core::coords::CUBE_SIZE as u8 / 2;
    for z in 0..jbm_core::coords::CUBE_SIZE as u8 {
        for x in 0..jbm_core::coords::CUBE_SIZE as u8 {
            let h = 1 + (x * 7 + z * 3) % 4;
            if height_at(x, z) >= half {
                tops.set_half(0, x, z, 4);
                tops.set_half(1, x, z, h);
            } else {
                tops.set_half(0, x, z, h);
            }
        }
    }
    tops
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
        level: 0,
        center: Arc::new(terrain_like_cube(&blocks)),
        neighbors: terrain_neighbors(&air, &stone),
        tops: MeshTops::default(),
    };
    let checker_input = MeshInput {
        level: 0,
        center: Arc::new(Cube::from_fn(|p| {
            if (p.x + p.y + p.z) % 2 == 0 {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        })),
        neighbors: uniform_neighbors(Some(Arc::clone(&air))),
        tops: MeshTops::default(),
    };
    let uniform_input = MeshInput {
        level: 0,
        center: Arc::new(Cube::filled(blocks.stone)),
        neighbors: uniform_neighbors(Some(Arc::clone(&air))),
        tops: MeshTops::default(),
    };
    let padded_input = MeshInput {
        level: 0,
        center: Arc::new(terrain_like_cube(&blocks)),
        neighbors: uniform_neighbors(Some(Arc::clone(&stone))),
        tops: MeshTops::default(),
    };

    let lod2_input = MeshInput {
        level: 2,
        center: Arc::new(terrain_like_cube(&blocks)),
        neighbors: terrain_neighbors(&air, &stone),
        tops: MeshTops {
            center: Some(Arc::new(terrain_like_tops())),
            ..MeshTops::default()
        },
    };

    let mut group = c.benchmark_group("mesh");
    group.bench_function("terrain_like", |b| {
        b.iter(|| {
            black_box(mesh::mesh(
                black_box(&terrain_input),
                black_box(&appearance),
            ))
        })
    });
    group.bench_function("checkerboard", |b| {
        b.iter(|| {
            black_box(mesh::mesh(
                black_box(&checker_input),
                black_box(&appearance),
            ))
        })
    });
    group.bench_function("uniform_stone_air_neighbors", |b| {
        b.iter(|| {
            black_box(mesh::mesh(
                black_box(&uniform_input),
                black_box(&appearance),
            ))
        })
    });
    group.bench_function("terrain_like_lod2", |b| {
        b.iter(|| black_box(mesh::mesh(black_box(&lod2_input), black_box(&appearance))))
    });
    let lod2_full_input = MeshInput {
        level: 2,
        center: Arc::clone(&lod2_input.center),
        neighbors: lod2_input.neighbors.clone(),
        tops: MeshTops::default(),
    };
    group.bench_function("terrain_like_lod2_full_tops", |b| {
        b.iter(|| {
            black_box(mesh::mesh(
                black_box(&lod2_full_input),
                black_box(&appearance),
            ))
        })
    });
    group.bench_function("padded_build", |b| {
        b.iter(|| {
            black_box(mesh::build_padded(
                black_box(&padded_input),
                black_box(&appearance),
            ))
        })
    });
    group.finish();

    let mut group = c.benchmark_group("suballoc");
    group.bench_function("churn_10k", |b| {
        b.iter(|| suballoc::churn(black_box(4096), 0x0123_4567_89ab_cdef, 10_000, false))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
