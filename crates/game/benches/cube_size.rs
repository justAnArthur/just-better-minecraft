//! Criterion bench for task 1.8: what the cube size costs. The `cube16`
//! feature of `jbm-core` selects 16³ instead of 32³; the bench names are the
//! same at both sizes. Region and layering live in `benches/common/mod.rs`.
//!
//! - `cube_size/mesh_region`: meshing every cube of the region serially
//!   (the region's total meshing cost; inputs are built outside the loop).
//! - `cube_size/remesh_edit`: meshing the one cube containing the block at
//!   `(64, h(64, 64), 64)` — the latency floor of a single block edit.
//!   Note: with 1.7's neighbor dirtying, an edit on a cube border re-meshes
//!   up to 8 cubes, not 1.
//! - `cube_size/build_region`: building the region's cubes with
//!   `Cube::from_fn`, as a proxy for generation cost per volume.

mod common;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_client::mesh;
use jbm_server::DEFAULT_SEED;

fn bench(c: &mut Criterion) {
    let region = common::Region::new(DEFAULT_SEED);
    let inputs: Vec<_> = region
        .region_positions
        .iter()
        .map(|&pos| region.mesh_input(pos))
        .collect();
    let edit = region.mesh_input(region.edit_cube());

    let mut group = c.benchmark_group("cube_size");
    group.bench_function("mesh_region", |b| {
        b.iter(|| {
            for input in &inputs {
                black_box(mesh::mesh(input, &region.appearance));
            }
        })
    });
    group.bench_function("remesh_edit", |b| {
        b.iter(|| black_box(mesh::mesh(black_box(&edit), &region.appearance)))
    });
    group.bench_function("build_region", |b| {
        b.iter(|| {
            let cubes: Vec<_> = region
                .region_positions
                .iter()
                .map(|&pos| region.build_cube(pos))
                .collect();
            black_box(cubes)
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
