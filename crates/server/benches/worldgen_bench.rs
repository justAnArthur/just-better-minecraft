//! Criterion benches for the terrain generator: a 32×32 heightmap patch, a
//! cube intersecting the surface, and a cube that hits the all-air fast
//! path.

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::CubePos;
use jbm_server::DEFAULT_SEED;
use jbm_server::terrain::Generator;

fn generator() -> Generator {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    Generator::new(DEFAULT_SEED, blocks)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("worldgen");

    group.bench_function("heightmap_32x32", |b| {
        let generator = generator();
        // surface_height bypasses the height cache, so this measures
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
        // mixed-path cost, heightmap cached after the first iteration.
        let h = generator.surface_height(0, 0);
        let pos = CubePos {
            x: 0,
            y: (h >> 5) as i32,
            z: 0,
        };
        b.iter(|| generator.generate(pos))
    });

    group.bench_function("air_cube", |b| {
        let generator = generator();
        // Far above every possible surface: the uniform-air fast path.
        let pos = CubePos { x: 0, y: 100, z: 0 };
        b.iter(|| generator.generate(pos))
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
