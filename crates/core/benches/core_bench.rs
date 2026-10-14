//! Sample criterion bench for `jbm-core`: fill and hash a 32³ cube of
//! `u16` block ids. Exists to prove the bench skeleton; later tasks add
//! real cases next to it.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use criterion::{Criterion, criterion_group, criterion_main};

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

criterion_group!(benches, bench);
criterion_main!(benches);
