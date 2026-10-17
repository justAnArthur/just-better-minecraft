//! Criterion benches for the worldgen graph generator: a 32×32 heightmap
//! patch, a cube intersecting the surface, a cube that hits the all-air
//! fast path, the reference (task 1.5) generator those are compared
//! against, grid sampling at several LOD steps, and compilation — on the
//! classic graph, plus the overworld's (task 2.8) heightmap, mountain and
//! lowland surface cubes and LOD sampling next to them.

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::CubePos;
use jbm_server::DEFAULT_SEED;
use jbm_server::terrain::Generator;
use jbm_server::worldgen::reference::ReferenceGenerator;
use jbm_server::worldgen::{self, GRID_VOLUME, Preset, compile};
use std::hint::black_box;

/// The classic preset: the pre-2.8 terrain every case here compared
/// against before task 2.8, so the numbers stay comparable.
fn generator() -> Generator {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    Generator::with_preset(DEFAULT_SEED, blocks, Preset::Classic)
}

/// The shipped overworld (task 2.8).
fn overworld_generator() -> Generator {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    Generator::new(DEFAULT_SEED, blocks)
}

fn reference_generator() -> ReferenceGenerator {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    ReferenceGenerator::new(DEFAULT_SEED, blocks)
}

/// A cube position whose column intersects the surface near the origin.
fn surface_pos(generator: &Generator) -> CubePos {
    let h = generator.surface_height(0, 0);
    CubePos {
        x: 0,
        y: (h >> 5) as i32,
        z: 0,
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("worldgen");

    group.bench_function("heightmap_32x32", |b| {
        let generator = generator();
        // surface_height bypasses the column cache, so this measures
        // computing a 32×32 patch of surface heights from scratch.
        b.iter(|| {
            let mut min = i64::MAX;
            for z in 0..32i64 {
                for x in 0..32i64 {
                    let h = generator.surface_height(x, z);
                    min = min.min(h);
                }
            }
            min
        })
    });

    group.bench_function("surface_cube", |b| {
        let generator = generator();
        // A cube whose column intersects the surface near the origin: the
        // mixed-path cost, column stage cached after the first iteration.
        let pos = surface_pos(&generator);
        b.iter(|| generator.generate(black_box(pos)))
    });

    group.bench_function("air_cube", |b| {
        let generator = generator();
        // Far above every possible surface: the uniform-air fast path.
        let pos = CubePos { x: 0, y: 100, z: 0 };
        b.iter(|| generator.generate(black_box(pos)))
    });

    group.bench_function("reference_surface_cube", |b| {
        let generator = reference_generator();
        // The task 1.5 generator on the same cube: the baseline the graph
        // path must stay within 2× of (task 2.1 spec).
        let h = generator.surface_height(0, 0);
        let pos = CubePos {
            x: 0,
            y: (h >> 5) as i32,
            z: 0,
        };
        b.iter(|| generator.generate(black_box(pos)))
    });

    group.bench_function("compile_overworld", |b| {
        b.iter(|| {
            let graph = worldgen::Graph::from_json(worldgen::OVERWORLD_JSON).unwrap();
            compile(&graph).unwrap()
        })
    });

    group.finish();

    let mut group = c.benchmark_group("worldgen/sample_grid");
    // One 32³ grid near the surface per step. Step 32 spans 1,024 blocks,
    // so every column is a cache miss — the worst case.
    for step in [1u32, 2, 4, 8, 32] {
        group.bench_function(format!("{step}"), |b| {
            let graph = Preset::Classic.compiled();
            let bottom = (generator().surface_height(0, 0) >> 5) * 32;
            let mut out = Box::new([0.0; GRID_VOLUME]);
            b.iter(|| {
                graph.sample_grid(
                    DEFAULT_SEED,
                    black_box([0, bottom, 0]),
                    black_box(step),
                    &mut out,
                )
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("worldgen/sample_lod");
    // Sampling the LOD pyramid's fill content straight from the graph
    // (task 2.7), next to `surface_cube`: each level's surface cube, plus
    // level 2's uniform-air fast path. Surface cases are asserted mixed.
    for level in [1u8, 2, 4] {
        group.bench_function(format!("{level}/surface"), |b| {
            let generator = generator();
            let shift = 5 + level as i64;
            let pos = CubePos {
                x: 0,
                y: (generator.surface_height(0, 0) >> shift) as i32,
                z: 0,
            };
            assert!(
                generator.sample_lod(level, pos).uniform().is_none(),
                "level {level} surface case must be mixed"
            );
            b.iter(|| generator.sample_lod(level, black_box(pos)))
        });
    }
    group.bench_function("2/air", |b| {
        let generator = generator();
        let pos = CubePos { x: 0, y: 100, z: 0 };
        assert_eq!(
            generator.sample_lod(2, pos).uniform(),
            Some(jbm_core::block::BlockStateId::AIR),
            "the air case must hit the fast path"
        );
        b.iter(|| generator.sample_lod(2, black_box(pos)))
    });
    group.finish();

    // The overworld (task 2.8) next to the classic cases: its heightmap,
    // a genuinely steep mountain cube and a lowland one, and LOD sampling.
    // Task 2.8's budget: the mountain surface cube must stay within 2× of
    // the classic `worldgen/surface_cube` in the same run.
    let mut group = c.benchmark_group("worldgen/overworld");

    group.bench_function("heightmap_32x32", |b| {
        let generator = overworld_generator();
        b.iter(|| {
            let mut min = i64::MAX;
            for z in 0..32i64 {
                for x in 0..32i64 {
                    min = min.min(generator.surface_height(x, z));
                }
            }
            min
        })
    });

    // Finds a cube column whose 32×32 surface heights span at least
    // `range` blocks, searching cube columns outward from the coarse peak.
    fn steep_pos(generator: &Generator, range: i64) -> CubePos {
        let (mut best_h, mut bx, mut bz) = (i64::MIN, 0i64, 0i64);
        for z in (-2048..2048).step_by(16) {
            for x in (-2048..2048).step_by(16) {
                let h = generator.surface_height(x, z);
                if h > best_h {
                    (best_h, bx, bz) = (h, x, z);
                }
            }
        }
        let (cx0, cz0) = ((bx >> 5) as i32, (bz >> 5) as i32);
        for r in 0..8i32 {
            for dz in -r..=r {
                for dx in -r..=r {
                    let (cx, cz) = (cx0 + dx, cz0 + dz);
                    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
                    for cz_off in 0..32i64 {
                        for cx_off in 0..32i64 {
                            let h = generator.surface_height(
                                i64::from(cx) * 32 + cx_off,
                                i64::from(cz) * 32 + cz_off,
                            );
                            lo = lo.min(h);
                            hi = hi.max(h);
                        }
                    }
                    if hi - lo >= range {
                        return CubePos {
                            x: cx,
                            y: (generator
                                .surface_height(i64::from(cx) * 32 + 16, i64::from(cz) * 32 + 16)
                                >> 5) as i32,
                            z: cz,
                        };
                    }
                }
            }
        }
        panic!("no cube column with a {range}-block height range near the peak");
    }

    group.bench_function("surface_cube_mountain", |b| {
        let generator = overworld_generator();
        let pos = steep_pos(&generator, 24);
        // The mountain case really is steep and mixed.
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for cz in 0..32i64 {
            for cx in 0..32i64 {
                let h = generator
                    .surface_height(i64::from(pos.x) * 32 + cx, i64::from(pos.z) * 32 + cz);
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        assert!(hi - lo >= 24, "mountain cube not steep: [{lo}, {hi}]");
        assert!(
            generator.generate(pos).uniform().is_none(),
            "mountain cube must be mixed"
        );
        b.iter(|| generator.generate(black_box(pos)))
    });

    group.bench_function("surface_cube_lowland", |b| {
        let generator = overworld_generator();
        // A gentle land cube near spawn: flat-ish, still mixed.
        let spawn = generator.spawn();
        let pos = CubePos {
            x: (spawn[0] as i64 >> 5) as i32,
            y: (generator.surface_height(spawn[0] as i64, spawn[2] as i64) >> 5) as i32,
            z: (spawn[2] as i64 >> 5) as i32,
        };
        assert!(
            generator.generate(pos).uniform().is_none(),
            "lowland cube must be mixed"
        );
        b.iter(|| generator.generate(black_box(pos)))
    });
    group.finish();

    let mut group = c.benchmark_group("worldgen/sample_lod_overworld");
    for level in [1u8, 2, 4] {
        group.bench_function(format!("{level}/surface"), |b| {
            let generator = overworld_generator();
            let pos = steep_pos(&generator, 24);
            let pos = CubePos {
                x: pos.x >> level,
                y: pos.y >> level,
                z: pos.z >> level,
            };
            assert!(
                generator.sample_lod(level, pos).uniform().is_none(),
                "level {level} surface case must be mixed"
            );
            b.iter(|| generator.sample_lod(level, black_box(pos)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
