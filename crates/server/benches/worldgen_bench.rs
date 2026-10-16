//! Criterion benches for the worldgen graph generator: a 32×32 heightmap
//! patch, a cube intersecting the surface, a cube that hits the all-air
//! fast path, the reference (task 1.5) generator those are compared
//! against, grid sampling at several LOD steps, and compilation.

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::CubePos;
use jbm_server::DEFAULT_SEED;
use jbm_server::terrain::Generator;
use jbm_server::worldgen::reference::ReferenceGenerator;
use jbm_server::worldgen::{self, GRID_VOLUME, compile};
use std::hint::black_box;

fn generator() -> Generator {
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
            let graph = worldgen::overworld();
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
}

criterion_group!(benches, bench);
criterion_main!(benches);
