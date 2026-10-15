//! Criterion benches for world persistence (task 2.3): the cube codec plus
//! zstd on a generated surface cube, one committed 64-cube batch, and a hot
//! load.

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::CubePos;
use jbm_core::cube::Cube;
use jbm_server::DEFAULT_SEED;
use jbm_server::Generator;
use jbm_server::storage::Storage;

/// A cube whose column intersects the surface near the origin (same recipe
/// as `worldgen/surface_cube`): the mixed-path record.
fn surface_cube(generator: &Generator) -> Cube {
    let h = generator.surface_height(0, 0);
    generator.generate(CubePos {
        x: 0,
        y: (h >> 5) as i32,
        z: 0,
    })
}

fn bench(c: &mut Criterion) {
    let mut registry = BlockRegistry::new();
    Blocks::register(&mut registry);
    let generator = Generator::new(DEFAULT_SEED, Blocks::register(&mut BlockRegistry::new()));
    let cube = surface_cube(&generator);
    let dir = tempfile::TempDir::new().unwrap();
    let storage = Storage::open(dir.path(), &registry, DEFAULT_SEED).unwrap();

    // The record bytes the save path produces: codec bytes at zstd level 3.
    let record = {
        let mut plain = Vec::new();
        cube.encode_with(|state| state.0, &mut plain);
        zstd::bulk::compress(&plain, 3).unwrap()
    };

    let mut group = c.benchmark_group("storage");

    group.bench_function("encode_surface_cube", |b| {
        b.iter(|| {
            let mut plain = Vec::new();
            cube.encode_with(|state| state.0, &mut plain);
            zstd::bulk::compress(&plain, 3).unwrap().len()
        })
    });

    group.bench_function("decode_surface_cube", |b| {
        b.iter(|| {
            // Same path as `Storage::load`: exact-size bulk decompress.
            let size = zstd::zstd_safe::get_frame_content_size(&record)
                .unwrap()
                .unwrap();
            let plain = zstd::bulk::decompress(&record, size as usize).unwrap();
            Cube::decode_with(&plain, |id| Some(jbm_core::block::BlockStateId(id))).unwrap()
        })
    });

    // One committed transaction of 64 surface cubes (distinct generated
    // cubes so the batch is realistic), fsync included.
    let h = generator.surface_height(0, 0);
    let batch_cubes: Vec<(CubePos, Cube)> = (0..64)
        .map(|i| {
            let pos = CubePos {
                x: i % 8,
                y: (h >> 5) as i32 + (i / 8) % 2,
                z: i / 16,
            };
            (pos, generator.generate(pos))
        })
        .collect();
    group.bench_function("save_batch_64", |b| {
        b.iter(|| {
            let batch: Vec<(CubePos, &Cube)> =
                batch_cubes.iter().map(|(pos, cube)| (*pos, cube)).collect();
            storage.save_batch(&batch).unwrap();
            storage.cube_count().unwrap()
        })
    });

    // A hot read plus decode of a mixed (non-uniform) cube from the batch:
    // uniform cubes decode trivially and would hide the decode cost.
    let mixed = batch_cubes
        .iter()
        .find(|(_, cube)| cube.uniform().is_none())
        .map(|(pos, _)| *pos)
        .expect("the batch holds a surface cube");
    group.bench_function("load_cube", |b| {
        b.iter(|| storage.load(mixed).unwrap().expect("stored"))
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
