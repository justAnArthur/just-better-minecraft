//! Criterion benches for `jbm-core`. Groups: `sample` (the Stage 0 skeleton
//! bench) and `cube` (task 1.1 storage cases). Group and function names read
//! as `<area>/<case>` in reports.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use jbm_core::block::BlockStateId;
use jbm_core::coords::CUBE_VOLUME;
use jbm_core::cube::Cube;

const CUBE: usize = 32;
const CELLS: usize = CUBE * CUBE * CUBE;

fn fill_cube() -> [u16; CELLS] {
    let mut cube = [0u16; CELLS];
    let mut state: u32 = 0x9e37_79b9;
    for cell in &mut cube {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *cell = (state >> 16) as u16;
    }
    cube
}

/// The 16 cycling states used by the 16-state cases (none of them air, so a
/// cube grown from air keeps a 17-entry palette).
fn states_16() -> [BlockStateId; 16] {
    core::array::from_fn(|i| BlockStateId(i as u32 + 1))
}

fn cube_16_states() -> Cube {
    Cube::from_fn(|pos| BlockStateId((pos.index() % 16) as u32 + 1))
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample");
    group.bench_function("fill_32cubed", |b| {
        b.iter(|| {
            let cube = fill_cube();
            let mut hasher = DefaultHasher::new();
            cube.hash(&mut hasher);
            hasher.finish()
        })
    });
    group.finish();
}

fn bench_cube(c: &mut Criterion) {
    let mut group = c.benchmark_group("cube");
    let states = states_16();

    group.bench_function("get_all_16_states", |b| {
        let cube = cube_16_states();
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..CUBE_VOLUME {
                sum += cube.get_index(i).0 as u64;
            }
            black_box(sum)
        })
    });

    group.bench_function("set_all_16_states", |b| {
        b.iter(|| {
            let mut cube = Cube::default();
            for i in 0..CUBE_VOLUME {
                cube.set_index(i, states[i % 16]);
            }
            cube
        })
    });

    group.bench_function("palette_growth_4096", |b| {
        b.iter(|| {
            let mut cube = Cube::default();
            for i in 0..4096 {
                cube.set_index(i, BlockStateId(i as u32 + 1));
            }
            cube
        })
    });

    let cube16 = cube_16_states();
    group.bench_function("iter_sum_16_states", |b| {
        b.iter(|| cube16.iter().map(|state| state.0 as u64).sum::<u64>())
    });

    group.bench_function("compact_4096_to_16", |b| {
        b.iter_batched(
            || {
                let mut cube = Cube::default();
                // Grow the palette to 4097 entries, then use only 16 states.
                for i in 0..4096 {
                    cube.set_index(i, BlockStateId(i as u32 + 1));
                }
                for i in 0..CUBE_VOLUME {
                    cube.set_index(i, states[i % 16]);
                }
                cube
            },
            |mut cube| {
                cube.compact();
                black_box(cube.palette().len());
                cube
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench, bench_cube);
criterion_main!(benches);
