//! Criterion bench for the server tick clock: 1000 frame-style
//! `advance(16 ms)` calls on a fresh clock, matching a 60 fps render loop.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_server::tick::TickClock;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_clock");
    group.bench_function("advance_1000", |b| {
        b.iter(|| {
            let mut clock = TickClock::new();
            for _ in 0..1000 {
                clock.advance(Duration::from_millis(16));
            }
            clock.tick_count()
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
