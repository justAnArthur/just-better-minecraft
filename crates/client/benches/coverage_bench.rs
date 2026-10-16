//! Criterion bench for the coverage rule's incremental update (task 2.6):
//! `coverage/toggle` measures one readiness change at level 1 under a full
//! ring set — every level-0 cube of a level-1 cube plus the rings above.

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_client::lod::{Coverage, CubeKey, MAX_COVERED_LEVEL};
use jbm_core::coords::CubePos;

/// One level-1 cube's eight level-0 children plus the ancestor chain to
/// the level cap — the smallest full ring set.
fn ring_set() -> Vec<CubeKey> {
    let parent = CubeKey {
        level: 1,
        pos: CubePos { x: 7, y: 2, z: -3 },
    };
    let mut keys = vec![parent];
    for octant in 0..8u8 {
        keys.push(parent.child(octant));
    }
    let mut k = parent;
    while k.level < MAX_COVERED_LEVEL {
        k = k.parent().0;
        keys.push(k);
    }
    keys
}

fn bench(c: &mut Criterion) {
    let keys = ring_set();
    let mut group = c.benchmark_group("coverage");
    group.bench_function("toggle", |b| {
        b.iter_batched(
            || {
                let mut coverage = Coverage::default();
                for &key in &keys {
                    coverage.set_ready(key, true);
                }
                coverage.take_changed();
                coverage
            },
            |mut coverage| {
                coverage.set_ready(keys[1], false);
                coverage.set_ready(keys[1], true);
                coverage.take_changed();
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
