//! Surface spans (task 2.10): the min/max surface height over a cube
//! column's footprint, cached per column and filled incrementally on the
//! tick thread under a per-tick budget. Streaming unions the camera's
//! vertical window with these spans, so tall terrain near the player is
//! covered by fine LOD even when it sits far above or below the camera
//! cube — the task 2.10 fix for coarse cubes drawn next to the player.
//!
//! Cost model (measured on the overworld, Apple M5 Pro): one LOD0 cube
//! column's exact bounds reuse the generator's per-column height cache
//! and cost about one mixed cube's evaluation (~80 µs overworld, ~20 µs
//! classic); one level-1 column samples 64 standalone heights
//! (~120 µs overworld). With the fill budgets below that is ≤ ~2 ms per
//! tick while a view fills, zero once settled, and no tick ever stalls
//! on the initial ~150 ms worth of columns.

use std::collections::{HashMap, HashSet, VecDeque};

use jbm_core::coords::CUBE_SIZE;

use crate::terrain::Generator;

/// LOD0 cube columns whose exact bounds one tick may compute (~1 ms
/// overworld, ~0.25 ms classic). The keep square around a player is
/// (2·8+3)² = 1225 columns, so a fresh view fills its nearest columns in
/// a handful of ticks and the square in about eighty.
pub(crate) const LOD0_SPAN_BUDGET: usize = 12;

/// Level-1 columns whose sampled bounds one tick may compute (~0.7 ms
/// overworld). The level-1 keep square is 19² = 361 columns.
pub(crate) const L1_SPAN_BUDGET: usize = 6;

/// How many sampled heights one level-1 column takes (an 8×8 grid over
/// its 64×64-block footprint, stride 8). Exact min/max over a level-n
/// footprint costs (32·2^n)² height evaluations — 4k at level 1 and
/// 4.2M at level 6 — which is why rings sample: a stride-8 cell cannot
/// hide a peak taller than the local relief, and the span only decides
/// which of a handful of vertical cubes to stream.
const L1_SAMPLES_PER_SIDE: i64 = 8;

/// Cache caps: above these the span maps clear wholesale (the entries
/// are pure functions of the deterministic heightmap, so a clear only
/// costs re-evaluation, never a wrong answer).
const LOD0_CAP: usize = 200_000;
const L1_CAP: usize = 100_000;

/// A column queued for a span computation: `(level, column x, column z)`
/// in that level's cube-column units.
type SpanJob = (u8, i32, i32);

/// The span store (task 2.10). Owned by the server; every method runs on
/// the tick thread. `lod0` holds exact bounds per LOD0 cube column,
/// `l1` sampled bounds per level-1 column; both are deterministic per
/// `(seed, preset)`, so caching cannot change any answer.
pub(crate) struct SurfaceSpans {
    lod0: HashMap<(i32, i32), (i64, i64)>,
    l1: HashMap<(i32, i32), (i64, i64)>,
    /// Columns still to compute, nearest-first, deduplicated by `queued`.
    queue: VecDeque<SpanJob>,
    queued: HashSet<SpanJob>,
    /// Bumped whenever a span lands, so players rebuild their desired
    /// lists and ring walks lazily.
    version: u64,
    /// Test/diagnostic counters.
    pub computed_lod0: u64,
    pub computed_l1: u64,
}

impl SurfaceSpans {
    pub(crate) fn new() -> SurfaceSpans {
        SurfaceSpans {
            lod0: HashMap::new(),
            l1: HashMap::new(),
            queue: VecDeque::new(),
            queued: HashSet::new(),
            version: 0,
            computed_lod0: 0,
            computed_l1: 0,
        }
    }

    /// The current version; players compare it against the version their
    /// desired list / walk was built at.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// The exact surface bounds of LOD0 cube column `(cx, cz)`, when
    /// already computed.
    pub(crate) fn lod0(&self, cx: i32, cz: i32) -> Option<(i64, i64)> {
        self.lod0.get(&(cx, cz)).copied()
    }

    /// The sampled surface bounds of level-1 column `(cx, cz)`, when
    /// already computed.
    pub(crate) fn l1(&self, cx: i32, cz: i32) -> Option<(i64, i64)> {
        self.l1.get(&(cx, cz)).copied()
    }

    /// Queues every column of the LOD0 keep square `(2r+3)²` and the
    /// level-1 keep square around `lod0_center` (in that level's units)
    /// that is not computed or queued yet. Nearest-first, so a fresh view
    /// covers its disk before its hysteresis ring. Called when a player's
    /// center moves; cheap when everything is already there.
    pub(crate) fn ensure_around(&mut self, lod0_center: (i32, i32), r: u32, lod_levels: u8) {
        let mut wanted: Vec<SpanJob> = Vec::new();
        let ri = i64::from(r);
        // LOD0 keep square.
        for dx in -ri - 1..=ri + 1 {
            for dz in -ri - 1..=ri + 1 {
                let col = (0u8, lod0_center.0 + dx as i32, lod0_center.1 + dz as i32);
                if !self.lod0.contains_key(&(col.1, col.2)) && !self.queued.contains(&col) {
                    wanted.push(col);
                }
            }
        }
        // Level-1 keep square, in level-1 columns around the level-1
        // center containing `lod0_center`.
        if lod_levels >= 1 {
            let l1_center = (lod0_center.0.div_euclid(2), lod0_center.1.div_euclid(2));
            for dx in -ri - 1..=ri + 1 {
                for dz in -ri - 1..=ri + 1 {
                    let col = (1u8, l1_center.0 + dx as i32, l1_center.1 + dz as i32);
                    if !self.l1.contains_key(&(col.1, col.2)) && !self.queued.contains(&col) {
                        wanted.push(col);
                    }
                }
            }
        }
        // Nearest-first, and LOD0 columns before L1 (the sort key makes
        // the two classes contiguous, which `poll`'s early-out relies
        // on): the fine view matters most.
        let center = (i64::from(lod0_center.0), i64::from(lod0_center.1));
        let l1c = (center.0.div_euclid(2), center.1.div_euclid(2));
        wanted.sort_by_key(|&(level, x, z)| {
            let (cx, cz) = if level == 0 {
                (center.0, center.1)
            } else {
                (l1c.0, l1c.1)
            };
            let (dx, dz) = (i64::from(x) - cx, i64::from(z) - cz);
            (level, dx * dx + dz * dz, x, z)
        });
        for job in wanted {
            self.queued.insert(job);
            self.queue.push_back(job);
        }
    }

    /// Runs up to the per-tick budgets of queued span computations
    /// against `generator`. Called once per tick before streaming; a
    /// settled view computes nothing. A spent budget skips its class's
    /// remaining jobs (they go back to the front of the queue, order
    /// intact) so neither class starves behind the other.
    pub(crate) fn poll(&mut self, generator: &Generator) {
        let mut lod0_left = LOD0_SPAN_BUDGET;
        let mut l1_left = L1_SPAN_BUDGET;
        let mut deferred: Vec<SpanJob> = Vec::new();
        while let Some(job) = self.queue.pop_front() {
            if lod0_left == 0 && l1_left == 0 {
                deferred.push(job);
                continue;
            }
            match job {
                (0, cx, cz) if lod0_left > 0 => {
                    self.queued.remove(&job);
                    if !self.lod0.contains_key(&(cx, cz)) {
                        let bounds = generator.column_surface_bounds(cx, cz);
                        if self.lod0.len() >= LOD0_CAP {
                            self.lod0.clear();
                        }
                        self.lod0.insert((cx, cz), bounds);
                        self.computed_lod0 += 1;
                        self.version += 1;
                    }
                    lod0_left -= 1;
                }
                (1, cx, cz) if l1_left > 0 => {
                    self.queued.remove(&job);
                    if !self.l1.contains_key(&(cx, cz)) {
                        let bounds = l1_sampled_bounds(generator, cx, cz);
                        if self.l1.len() >= L1_CAP {
                            self.l1.clear();
                        }
                        self.l1.insert((cx, cz), bounds);
                        self.computed_l1 += 1;
                        self.version += 1;
                    }
                    l1_left -= 1;
                }
                _ => deferred.push(job),
            }
        }
        for job in deferred.into_iter().rev() {
            self.queue.push_front(job);
        }
    }

    /// Queued span jobs remaining (test/diagnostic).
    pub(crate) fn pending(&self) -> usize {
        self.queue.len()
    }
}

/// The sampled surface bounds of level-1 column `(cx, cz)`: min/max
/// [`Generator::surface_height`] over an 8×8 grid covering its 64×64
/// block footprint. Sampling keeps the cost at 64 evaluations a column
/// wherever exact min/max would cost thousands; a stride-8 cell cannot
/// hide more than the local relief, which the below-widening and the
/// `max_column_cubes` cap absorb.
fn l1_sampled_bounds(generator: &Generator, cx: i32, cz: i32) -> (i64, i64) {
    let side = i64::from(CUBE_SIZE) * 2;
    let stride = side / L1_SAMPLES_PER_SIDE;
    let base_x = i64::from(cx) * side;
    let base_z = i64::from(cz) * side;
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for i in 0..L1_SAMPLES_PER_SIDE {
        for j in 0..L1_SAMPLES_PER_SIDE {
            let h = generator.surface_height(base_x + i * stride, base_z + j * stride);
            min = min.min(h);
            max = max.max(h);
        }
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::SurfaceSpans;
    use crate::terrain::Generator;
    use jbm_core::block::{BlockRegistry, builtin};
    use jbm_core::coords::CUBE_SIZE;

    fn generator() -> Generator {
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        Generator::new(7, blocks)
    }

    /// The queue fills nearest-first, poll respects its budgets, results
    /// are exact at LOD0 and sampled-but-covering at level 1, and repeat
    /// `ensure_around` calls queue nothing new.
    #[test]
    fn fills_nearest_first_within_budget_and_caches() {
        let generator = generator();
        let mut spans = SurfaceSpans::new();
        spans.ensure_around((100, -100), 2, 4);
        let queued = spans.pending();
        assert_eq!(
            queued,
            (2 * 2 + 3usize).pow(2) + (2 * 2 + 3usize).pow(2),
            "the LOD0 and L1 keep squares, nothing else"
        );
        // Re-ensuring at the same center queues nothing new.
        spans.ensure_around((100, -100), 2, 4);
        assert_eq!(spans.pending(), queued);

        // The first queue entries are the center columns (LOD0 before
        // L1 at equal distance, by the sort key).
        // Poll with the full budget: everything lands (small radii).
        for _ in 0..100 {
            spans.poll(&generator);
            if spans.pending() == 0 {
                break;
            }
        }
        assert_eq!(spans.pending(), 0);
        assert_eq!(spans.computed_lod0 as usize, (2 * 2 + 3usize).pow(2));
        assert_eq!(spans.computed_l1 as usize, (2 * 2 + 3usize).pow(2));
        assert_eq!(spans.version(), u64::try_from(queued).unwrap());
        assert!(spans.lod0(100, -100).is_some());
        assert!(spans.l1(50, -50).is_some());

        // LOD0 bounds are exact: match a brute-force min/max of
        // surface_height over the column.
        let size = i64::from(CUBE_SIZE);
        let (lo, hi) = spans.lod0(100, -100).unwrap();
        let mut brute = (i64::MAX, i64::MIN);
        for x in 0..size {
            for z in 0..size {
                let h = generator.surface_height(100 * size + x, -100 * size + z);
                brute.0 = brute.0.min(h);
                brute.1 = brute.1.max(h);
            }
        }
        assert_eq!((lo, hi), brute);

        // L1 bounds cover the sampled points and stay sane (the overworld
        // relief); they are sampled, so only containment of the samples
        // is asserted.
        let (l1lo, l1hi) = spans.l1(50, -50).unwrap();
        let h_mid = generator.surface_height(50 * size * 2 + size, -50 * size * 2 + size);
        assert!(l1lo <= h_mid && h_mid <= l1hi);

        // A settled poll computes nothing.
        let before = spans.version();
        spans.poll(&generator);
        assert_eq!(spans.version(), before);
        assert_eq!(spans.pending(), 0);
    }

    /// Moving the center queues only the new strip, nearest-first.
    #[test]
    fn move_queues_only_the_new_strip() {
        let generator = generator();
        let mut spans = SurfaceSpans::new();
        spans.ensure_around((0, 0), 2, 2);
        while spans.pending() > 0 {
            spans.poll(&generator);
        }
        spans.ensure_around((2, 0), 2, 2);
        // The LOD0 square moved two cubes (dx -1..=5): two 7-column
        // strips; the L1 square moved one (dx -2..=4): one strip.
        assert_eq!(spans.pending(), 3 * 7);
        // One poll spends its whole LOD0 budget (12 of the 14 new LOD0
        // columns) and then 6 of the 7 L1 columns — neither class waits
        // behind the other.
        spans.poll(&generator);
        assert_eq!(spans.computed_lod0, 49 + 12);
        assert_eq!(spans.computed_l1, 49 + 6);
        assert_eq!(spans.pending(), 2 + 1);
    }
}
