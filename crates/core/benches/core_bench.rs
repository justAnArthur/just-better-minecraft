//! Criterion benches for `jbm-core`. Groups: `sample` (the Stage 0 skeleton
//! bench) and `cube` (task 1.1 storage cases). Group and function names read
//! as `<area>/<case>` in reports.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use glam::DVec3;
use jbm_core::block::BlockStateId;
use jbm_core::coords::{BlockPos, CUBE_VOLUME};
use jbm_core::cube::Cube;
use jbm_core::physics::{self, BlockShapes, MoveInput, PlayerState};
use jbm_core::raycast::{REACH, raycast};

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
        // 4096 at 32³; an eighth of the cube at smaller sizes.
        let distinct = (CUBE_VOLUME / 8).min(4096);
        b.iter(|| {
            let mut cube = Cube::default();
            for i in 0..distinct {
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
        // 4096 distinct states (an eighth of the cube at smaller sizes),
        // then only 16 states left in use.
        let distinct = (CUBE_VOLUME / 8).min(4096);
        b.iter_batched(
            || {
                let mut cube = Cube::default();
                for i in 0..distinct {
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

/// An infinite flat floor at `top`, everything loaded.
struct Floor {
    top: i64,
}

impl BlockShapes for Floor {
    fn solid(&self, pos: BlockPos) -> Option<bool> {
        Some(pos.y < self.top)
    }
}

/// Loaded air everywhere: a fall that never lands.
struct Air;

impl BlockShapes for Air {
    fn solid(&self, _pos: BlockPos) -> Option<bool> {
        Some(false)
    }
}

fn bench_physics(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics");

    group.bench_function("step_walk_flat", |b| {
        let world = Floor { top: 64 };
        let input = MoveInput {
            forward: 1,
            ..MoveInput::IDLE
        };
        let mut state = PlayerState {
            position: DVec3::new(0.5, 64.0, 0.5),
            velocity: DVec3::ZERO,
            on_ground: true,
            flying: false,
        };
        b.iter(|| {
            state = physics::step(state, &input, &world);
            black_box(state)
        })
    });

    group.bench_function("step_fall_terminal", |b| {
        let world = Air;
        let input = MoveInput::IDLE;
        let mut state = PlayerState {
            position: DVec3::new(0.5, 0.0, 0.5),
            velocity: DVec3::new(0.0, -78.0, 0.0),
            on_ground: false,
            flying: false,
        };
        b.iter(|| {
            state = physics::step(state, &input, &world);
            black_box(state)
        })
    });

    group.finish();
}

fn bench_raycast(c: &mut Criterion) {
    let mut group = c.benchmark_group("raycast");

    group.bench_function("reach_5_diagonal", |b| {
        // Loaded air everywhere, so the ray walks its full diagonal reach —
        // the worst-case per-targeting cost.
        let origin = DVec3::new(0.5, 0.5, 0.5);
        let dir = DVec3::new(1.0, 1.0, 1.0);
        b.iter(|| raycast(origin, dir, REACH, |_| Some(false)))
    });

    group.finish();
}

criterion_group!(benches, bench, bench_cube, bench_physics, bench_raycast);
criterion_main!(benches);
