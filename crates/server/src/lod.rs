//! The server's pyramid of LOD cubes (tasks 2.4 and 2.7). Levels are
//! 1-based: level *n* cells are 2ⁿ blocks per side, and a level-*n* cube
//! covers the same volume as a 2×2×2 group of level-(*n*−1) cubes.
//!
//! # The content model (2.7)
//!
//! The content of a LOD cube at level *n* and position *p* is defined per
//! octant *o*:
//!
//! - **Real**, when the child at level *n*−1 holds real data. At level 1
//!   that is a LOD0 cube that has been in the world, in its latest content;
//!   at level *n ≥ 2* it is a level-(*n*−1) child with at least one real
//!   octant. The octant is the downsample of the child's **full content**
//!   (worldgen-filled octants included — never placeholder air), and
//!   `known` bit *o* is set exactly for these octants.
//! - **Worldgen** otherwise: the octant of `Generator::sample_lod(n, p)`,
//!   installed by the fill pipeline. `filled` bit *o* marks these octants.
//!
//! A cube is **ready** when every octant holds content — real, worldgen or
//! loaded from storage: `known | filled == 0xFF`. [`LodPyramid::ready`]
//! returns `Some` only then, and [`LodPyramid::take_changed`] never reports
//! a cube that is not ready, so streaming (2.6) only ever sees complete
//! cubes.
//!
//! A real octant at level *n ≥ 2* is the downsample of its child's full
//! content, so it can be computed **only once the child is ready** —
//! computing it earlier would read the child's not-yet-filled octants as
//! placeholder air, which the model forbids. Whenever a cube gains or
//! changes a real octant while its child is not ready — an insert, an edit,
//! a load — the propagation is therefore **deferred**: the parent marks the
//! octant *pending* and internally requests the child's fill
//! ([`LodPyramid::request`]); when the child becomes ready, a pending
//! octant is (re)computed from the child's then-full content and the walk
//! continues upward. The pending bit lands on octants that are already
//! known too (a stored record loads with real octants while its children
//! do not exist yet): a **stale** octant keeps its last real content —
//! never placeholder air — until that same recompute replaces it, so an
//! edit that landed while the child was filling is never lost. Pending
//! octants that hold no content never make a cube ready.
//!
//! The model is a pure function of two inputs — the set of LOD0 cubes ever
//! seen with their latest content, and the worldgen — and
//! [`LodPyramid::rebuild`] is its reference implementation: given a
//! generator and the LOD0 cubes it produces every ready cube from scratch.
//! The incremental paths must equal it exactly, cell for cell and `known`
//! for `known`, after any order of inserts, edits, requests, fills and
//! storage loads.
//!
//! # The fill pipeline (2.7)
//!
//! [`LodPyramid::request`] queues a **load-or-sample** job: the stored
//! record when there is one, otherwise `Generator::sample_lod`. Jobs run
//! on the pyramid's own worker threads (`jbm-lod-N`, a separate small pool
//! so LOD work can never delay the LOD0 generation pool's workers) or, with
//! zero threads, inline in [`LodPyramid::poll`] — up to the configured
//! budget per call, which tests and benches use for determinism.
//! [`LodPyramid::poll`] merges finished jobs into the pyramid — real
//! octants always win over the job's content — and completes the deferred
//! propagations that became possible, recording every changed or newly
//! ready cube for [`LodPyramid::take_changed`]. The pyramid's own requests
//! (the deferred propagations' fills) are uncapped and excluded from
//! [`LodPyramid::in_flight`], so they never consume the streaming budget
//! that call reports.
//!
//! Eviction never touches the pyramid: LOD cubes outlive their LOD0
//! children, so player builds stay visible from far away. The pyramid
//! grows with the explored area; stored cubes could be evicted and
//! reloaded, but in-memory eviction is still open (see docs/progress.md).

use std::collections::{HashMap, HashSet, VecDeque};
use std::mem::size_of;
use std::panic;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};

use jbm_core::block::BlockStateId;
use jbm_core::coords::{BlockPos, CubePos};
use jbm_core::cube::Cube;
use jbm_core::lod::{CellBox, copy_octant, downsample_octant, downsample_region, parent};

use crate::storage::{Storage, StorageError};
use crate::terrain::Generator;

/// One cube of a LOD level plus its per-octant bookkeeping. Each octant
/// holds content in one of three states — `known`, `filled` or empty — and
/// the `pending` bit can additionally mark a real octant whose content is
/// not final:
///
/// - `known` — real: the downsample of its child's full content
///   (`known` bit set).
/// - `filled` — worldgen or stored content for a child with no real data.
/// - empty — nothing yet; a fill job can supply it.
/// - `pending` — real by definition (the child holds real data) but the
///   child is not ready yet, so the content cannot be computed from the
///   child's full content. On an empty octant this means "computed when
///   the child readies"; on a `known` octant (both bits set) it means
///   **stale**: the child changed while it was not ready — an edit, a
///   re-insert or a load landed under it — so the octant keeps its last
///   real content, never placeholder air, until the child readies and the
///   octant is recomputed in full. Either way a pending octant never takes
///   content from a fill job.
///
/// A cube is ready exactly when every octant is `known` or `filled`; a
/// stale octant keeps the cube ready with its last real content until the
/// recompute lands (and the change it brings is reported).
#[derive(Debug, Clone)]
pub struct LodCube {
    pub cube: Cube,
    pub known: u8,
    pub filled: u8,
    pub pending: u8,
}

impl LodCube {
    fn empty() -> Self {
        Self {
            cube: Cube::filled(BlockStateId::AIR),
            known: 0,
            filled: 0,
            pending: 0,
        }
    }

    /// Every octant holds content (real or filled): the cube may stream.
    pub fn ready(&self) -> bool {
        self.known | self.filled == 0xFF
    }
}

/// The **ready** cubes whose content changed in one operation, in level
/// order — the no-partial-cubes contract, so an insert or edit never lists
/// a cube whose octants are not all filled yet. This is the internal delta
/// of [`LodPyramid::insert_lod0`] and [`LodPyramid::edit_lod0`]; streaming
/// (2.6) consumes [`LodPyramid::take_changed`] instead, which reports a
/// cube when it becomes ready and on every later change — so a cube left
/// out here because it was not ready yet is still delivered when it
/// readies, and neither channel ever carries a partial cube.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LodChanges {
    pub cubes: Vec<(u8, CubePos)>,
}

impl LodChanges {
    pub fn is_empty(&self) -> bool {
        self.cubes.is_empty()
    }
}

/// One load-or-sample job and its result, as the pool carries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Job {
    level: u8,
    pos: CubePos,
}

enum JobResult {
    Done(Job, Cube, u8),
    /// A worker panicked on this job; `poll` resumes the panic on the tick
    /// thread so it never silently vanishes with the worker.
    Panicked(Job, Box<dyn std::any::Any + Send>),
}

/// Test hooks shared between the pyramid and its workers (see
/// [`LodPyramid::hold_workers`] and [`LodPyramid::fail_next_job`]).
#[derive(Default)]
struct Hooks {
    state: Mutex<HookState>,
    released: Condvar,
}

#[derive(Default)]
struct HookState {
    held: bool,
    fail_next: bool,
}

impl Hooks {
    /// Blocks the worker until `release_workers`, after the job was taken.
    /// Shutdown breaks the wait (the drop path wakes the condvar), so
    /// dropping the pyramid never hangs on a held worker.
    fn enter_job(&self, shutdown: &AtomicBool) {
        let mut state = self.state.lock().unwrap();
        while state.held && !shutdown.load(Ordering::Acquire) {
            state = self.released.wait(state).unwrap();
        }
    }

    /// Wakes every held worker without clearing `held`: the drop path, so
    /// workers re-check shutdown and exit instead of sleeping forever.
    fn wake_all(&self) {
        self.released.notify_all();
    }
}

/// The pyramid itself: levels `1..=max_levels`, each a map of cubes, plus
/// the fill pipeline that turns requests into ready cubes. A pyramid with
/// `max_levels == 0` is disabled — it allocates nothing, spawns nothing and
/// every operation is a no-op.
pub struct LodPyramid {
    /// `levels[l - 1]` holds level `l`; empty while disabled.
    levels: Vec<HashMap<CubePos, LodCube>>,
    /// Ready cubes whose content changed since the last
    /// [`LodPyramid::take_changed`]; streaming re-sends them.
    changed: HashSet<(u8, CubePos)>,
    /// Ready cubes whose content changed since the last
    /// [`LodPyramid::take_store_dirty`]; persistence re-saves them.
    store_dirty: HashSet<(u8, CubePos)>,
    /// The worldgen the fill pipeline samples from; `None` in a bare
    /// pyramid (no fills: requests are no-ops).
    generator: Option<Arc<Generator>>,
    /// Stored LOD, when persistence is on.
    storage: Option<Arc<Storage>>,
    /// Jobs queued, running or finished-but-uncollected. The flag marks the
    /// pyramid's own requests (pending propagations), which do not count
    /// against the streaming cap [`LodPyramid::in_flight`] reports.
    in_flight: HashMap<Job, bool>,
    /// How many internal (pending-propagation) jobs were queued in total —
    /// the load report number for the default view.
    internal_requests: u64,
    /// Pending inline-mode jobs, in request order.
    inline_queue: VecDeque<Job>,
    /// `None` in inline mode.
    job_tx: Option<Sender<Job>>,
    results_rx: Receiver<JobResult>,
    shutdown: Arc<AtomicBool>,
    hooks: Arc<Hooks>,
    threads: Vec<JoinHandle<()>>,
    /// How many jobs one inline `poll` may run.
    inline_budget: usize,
}

impl LodPyramid {
    /// A bare pyramid with no fill pipeline: inserts and edits work (the
    /// 2.4 paths), but `request` queues nothing and no octant is ever
    /// filled. The server (and the fill tests) use
    /// [`LodPyramid::with_workers`].
    pub fn new(max_levels: u8) -> Self {
        Self::with_parts(max_levels, None, None, 0, 0)
    }

    /// The pyramid the server builds: levels `1..=max_levels` with the
    /// load-or-sample pipeline behind them. `threads > 0` starts named
    /// `jbm-lod-N` workers; `threads == 0` selects inline mode, where
    /// `poll` runs up to `inline_budget` jobs synchronously per call — the
    /// deterministic mode tests and benches use. A separate pool from the
    /// LOD0 generation pool on purpose: no queue is shared, so LOD work
    /// can never delay a LOD0 generation worker.
    pub fn with_workers(
        max_levels: u8,
        generator: Arc<Generator>,
        storage: Option<Arc<Storage>>,
        threads: usize,
        inline_budget: usize,
    ) -> Self {
        Self::with_parts(max_levels, Some(generator), storage, threads, inline_budget)
    }

    fn with_parts(
        max_levels: u8,
        generator: Option<Arc<Generator>>,
        storage: Option<Arc<Storage>>,
        threads: usize,
        inline_budget: usize,
    ) -> Self {
        let (results_tx, results_rx) = std::sync::mpsc::channel();
        let mut pyramid = Self {
            levels: vec![HashMap::new(); max_levels as usize],
            changed: HashSet::new(),
            store_dirty: HashSet::new(),
            generator,
            storage,
            in_flight: HashMap::new(),
            internal_requests: 0,
            inline_queue: VecDeque::new(),
            job_tx: None,
            results_rx,
            shutdown: Arc::new(AtomicBool::new(false)),
            hooks: Arc::new(Hooks::default()),
            threads: Vec::with_capacity(threads),
            inline_budget,
        };
        if max_levels > 0 && threads > 0 {
            let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
            // `mpsc::Receiver` is not `Sync`, so workers wait on it through
            // a mutex — the same pattern as the generation pool. The lock is
            // held only around `recv`, never while sampling.
            let job_rx = Arc::new(Mutex::new(job_rx));
            for id in 0..threads {
                let Some(generator) = pyramid.generator.clone() else {
                    break;
                };
                let worker = lod_worker::Worker {
                    generator,
                    storage: pyramid.storage.clone(),
                    jobs: Arc::clone(&job_rx),
                    results: results_tx.clone(),
                    shutdown: Arc::clone(&pyramid.shutdown),
                    hooks: Arc::clone(&pyramid.hooks),
                };
                let handle = thread::Builder::new()
                    .name(format!("jbm-lod-{id}"))
                    .spawn(move || lod_worker::run(worker))
                    .expect("spawning a jbm-lod worker thread");
                pyramid.threads.push(handle);
            }
            pyramid.job_tx = Some(job_tx);
        }
        pyramid
    }

    /// The configured top level; 0 when the pyramid is disabled.
    pub fn max_levels(&self) -> u8 {
        self.levels.len() as u8
    }

    fn enabled(&self) -> bool {
        !self.levels.is_empty()
    }

    fn level_mut(&mut self, level: u8) -> &mut HashMap<CubePos, LodCube> {
        &mut self.levels[level as usize - 1]
    }

    fn entry(&mut self, level: u8, pos: CubePos) -> &mut LodCube {
        self.level_mut(level)
            .entry(pos)
            .or_insert_with(LodCube::empty)
    }

    /// Records that `(level, pos)` changed (`content`) or first became
    /// ready (`was_ready == false`), for `take_changed` (ready cubes only)
    /// and, under persistence, for `take_store_dirty`.
    fn touch(&mut self, level: u8, pos: CubePos, was_ready: bool, content: bool) {
        let Some(lc) = self.get(level, pos) else {
            return;
        };
        let now_ready = lc.ready();
        if !now_ready {
            return; // not-ready cubes are never reported
        }
        if content || !was_ready {
            self.changed.insert((level, pos));
            if self.storage.is_some() {
                self.store_dirty.insert((level, pos));
            }
        }
    }

    /// A LOD0 cube entered the world (generated or loaded): downsample it
    /// into its level-1 parent — created as uniform air if absent — set the
    /// known bit, then walk the change up ([`LodPyramid::propagate`]).
    /// When `half` is given it is the cube's precomputed level-1 octant
    /// (a generation worker's downsample, installed with
    /// [`copy_octant`]); without it the downsample runs here. Re-inserting
    /// an unchanged cube reports no changes.
    pub fn insert_lod0(&mut self, pos: CubePos, cube: &Cube) -> LodChanges {
        self.insert_lod0_octant(pos, cube, None)
    }

    /// [`LodPyramid::insert_lod0`] with a precomputed level-1 octant.
    pub fn insert_lod0_octant(
        &mut self,
        pos: CubePos,
        cube: &Cube,
        half: Option<&Cube>,
    ) -> LodChanges {
        if !self.enabled() {
            return LodChanges::default();
        }
        let (parent_pos, octant) = parent(pos);
        let was_ready = self.get(1, parent_pos).is_some_and(|lc| lc.ready());
        let region = {
            let lc = self.entry(1, parent_pos);
            let region = match half {
                Some(half) => copy_octant(half, &mut lc.cube, octant),
                None => downsample_octant(cube, &mut lc.cube, octant),
            };
            lc.known |= 1 << octant;
            region
        };
        let mut changes = LodChanges::default();
        if region.is_some() {
            changes.cubes.push((1, parent_pos));
        }
        self.touch(1, parent_pos, was_ready, region.is_some());
        changes
            .cubes
            .extend(self.propagate(1, parent_pos, region).cubes);
        // The no-partial-cubes contract: list ready cubes only. A cube
        // filtered out here is still reported by `take_changed` when it
        // becomes ready (`touch` above recorded nothing for it yet).
        changes
            .cubes
            .retain(|&(level, pos)| self.ready(level, pos).is_some());
        changes
    }

    /// One accepted edit of the LOD0 block at `pos` (in the post-edit
    /// `cube`): recompute the single affected cell at level 1 and continue
    /// up while levels change, each level costing one cell. The walk stops
    /// at a level whose cube is missing or whose octant is not real — and
    /// at a child that is not ready: the parent's octant is marked pending
    /// (a *stale* mark when the octant is already known, as after a storage
    /// load) and the child's fill is requested, so the pending propagation
    /// recomputes the octant in full — edit included — when the child
    /// readies. Edits in an area with no pyramid cubes report no changes.
    pub fn edit_lod0(&mut self, pos: BlockPos, cube: &Cube) -> LodChanges {
        if !self.enabled() {
            return LodChanges::default();
        }
        let mut changes = LodChanges::default();
        let (child_pos, local) = pos.split();
        let (parent_pos, octant) = parent(child_pos);
        let was_ready = self.get(1, parent_pos).is_some_and(|lc| lc.ready());
        let region = {
            let Some(lc) = self.level_mut(1).get_mut(&parent_pos) else {
                return changes;
            };
            downsample_region(cube, &mut lc.cube, octant, CellBox::single(local))
        };
        let Some(region) = region else {
            return changes; // the first level did not change
        };
        changes.cubes.push((1, parent_pos));
        self.touch(1, parent_pos, was_ready, true);
        // The region walk up: each level recomputes only the changed cells
        // of its octant, while the child is ready and the octant real.
        let mut child = parent_pos;
        let mut region = region;
        for level in 2..=self.max_levels() {
            let (pos, octant) = parent(child);
            let bit = 1 << octant;
            // Set when the child is not ready: after the level borrows end,
            // the loop requests the child's fill and stops — the pending
            // propagation picks the edit up when the child readies.
            let mut deferred = false;
            let walked = {
                let (below, above) = self.levels.split_at_mut(level as usize - 1);
                let (Some(child_lc), Some(lc)) = (
                    below[level as usize - 2].get(&child),
                    above[0].get_mut(&pos),
                ) else {
                    break;
                };
                if !child_lc.ready() {
                    // The edit cannot be recomputed from the child's full
                    // content yet. Mark the parent's octant pending — even
                    // when it is already known, so the ready branch of
                    // `propagate` recomputes it in full instead of taking
                    // the unchanged-region shortcut.
                    lc.pending |= bit;
                    deferred = true;
                    None
                } else if lc.known & bit == 0 {
                    break;
                } else {
                    let was_ready = lc.ready();
                    let changed = downsample_region(&child_lc.cube, &mut lc.cube, octant, region);
                    Some((was_ready, changed))
                }
            };
            if deferred {
                self.request_internal(level - 1, child);
                break;
            }
            let Some((was_ready, changed)) = walked else {
                break;
            };
            match changed {
                Some(boxed) => {
                    changes.cubes.push((level, pos));
                    self.touch(level, pos, was_ready, true);
                    child = pos;
                    region = boxed;
                }
                None => break,
            }
        }
        // As in `insert_lod0_octant`: ready cubes only, and `take_changed`
        // still delivers the rest when they ready.
        changes
            .cubes
            .retain(|&(level, pos)| self.ready(level, pos).is_some());
        changes
    }

    /// The upward walk after the cube at `(child_level, child_pos)` gained
    /// real data or changed content by `region` (its own cells; `None`
    /// when the content did not change): while the child holds real data
    /// and is ready, its parent's octant holds (or gains) real content —
    /// recomputed from the child's full content when the octant was not
    /// real yet or is stale (pending), otherwise only over `region`. A
    /// child that holds real data but is not ready defers: the parent's
    /// octant goes pending — a stale mark when it was already known — and
    /// the child's fill is requested internally. A child with no real data
    /// stops the walk — its parent's octant is worldgen, not real.
    fn propagate(
        &mut self,
        child_level: u8,
        child_pos: CubePos,
        region: Option<CellBox>,
    ) -> LodChanges {
        let mut changes = LodChanges::default();
        if !self.enabled() {
            return changes;
        }
        let mut child = child_pos;
        let mut level = child_level;
        let mut region = region;
        while level < self.max_levels() {
            let parent_level = level + 1;
            let (pos, octant) = parent(child);
            let bit = 1 << octant;
            let Some(child_lc) = self.get(level, child) else {
                break;
            };
            if child_lc.known == 0 {
                break; // nothing real above: the parent's octant is worldgen
            }
            if !child_lc.ready() {
                // Real by definition but not computable yet: defer, and
                // fill the child. The pending bit lands on known octants
                // too (a load or an earlier run made them real): such an
                // octant is stale — its content predates the child's
                // latest changes — and the ready branch below recomputes
                // it in full.
                self.entry(parent_level, pos).pending |= bit;
                self.request_internal(level, child);
                break;
            }
            // The child is borrowed, not cloned: this runs per insert and
            // per poll on the tick thread, and a clone would double the
            // memory traffic of every level's downsample.
            let (below, above) = self.levels.split_at_mut(level as usize);
            let (was_ready, was_known, changed) = {
                let child_lc = &below[level as usize - 1][&child];
                let lc = above[0].entry(pos).or_insert_with(LodCube::empty);
                let was_ready = lc.ready();
                let was_known = lc.known & bit != 0;
                // A pending bit on a known octant marks it stale: the child
                // changed while it was not ready, so never trust a partial
                // region here — recompute the octant in full.
                let stale = lc.pending & bit != 0;
                let changed = if was_known && !stale {
                    region.and_then(|region| {
                        downsample_region(&child_lc.cube, &mut lc.cube, octant, region)
                    })
                } else {
                    lc.pending &= !bit;
                    lc.filled &= !bit;
                    lc.known |= bit;
                    downsample_octant(&child_lc.cube, &mut lc.cube, octant)
                };
                (was_ready, was_known, changed)
            };
            if changed.is_some() {
                changes.cubes.push((parent_level, pos));
            }
            self.touch(parent_level, pos, was_ready, changed.is_some());
            match changed {
                Some(boxed) => region = Some(boxed),
                None => {
                    if was_known {
                        break; // already real and unchanged: done
                    }
                    // Real gained with identical content: ancestors may
                    // still need the known transition — keep walking.
                    region = None;
                }
            }
            child = pos;
            level = parent_level;
        }
        changes
    }

    /// Rebuilds every level from scratch, bottom up, from the given LOD0
    /// cubes **and the worldgen** — the reference implementation of the
    /// content model the incremental paths are tested against. Each level
    /// is built from the level below, then every cube's non-real octants
    /// are filled from `Generator::sample_lod`, so the result holds only
    /// ready cubes. Filling covers the axis-aligned box of positions that
    /// spans the LOD0 cubes' ancestors at each level.
    pub fn rebuild<'a>(
        max_levels: u8,
        generator: &Generator,
        lod0: impl Iterator<Item = (CubePos, &'a Cube)>,
    ) -> Self {
        let mut pyramid = Self::new(max_levels);
        if !pyramid.enabled() {
            return pyramid;
        }
        // Level 1: one downsample per LOD0 cube, and the span they cover.
        let mut span: Option<(CubePos, CubePos)> = None;
        for (pos, cube) in lod0 {
            let (parent_pos, octant) = parent(pos);
            let lc = pyramid.entry(1, parent_pos);
            downsample_octant(cube, &mut lc.cube, octant);
            lc.known |= 1 << octant;
            span = Some(match span {
                None => (parent_pos, parent_pos),
                Some((lo, hi)) => (
                    CubePos {
                        x: lo.x.min(parent_pos.x),
                        y: lo.y.min(parent_pos.y),
                        z: lo.z.min(parent_pos.z),
                    },
                    CubePos {
                        x: hi.x.max(parent_pos.x),
                        y: hi.y.max(parent_pos.y),
                        z: hi.z.max(parent_pos.z),
                    },
                ),
            });
        }
        let Some((mut min, mut max)) = span else {
            return pyramid;
        };
        pyramid.fill_box(1, min, max, generator);
        // Each level above: real octants from the (now ready) level below,
        // then fill the box of parents.
        for level in 2..=max_levels {
            let children: Vec<CubePos> = pyramid.levels[level as usize - 2]
                .iter()
                .filter(|(_, lc)| lc.known != 0)
                .map(|(&pos, _)| pos)
                .collect();
            for child in children {
                let (pos, octant) = parent(child);
                // Borrow the child through a level split rather than
                // cloning it (as `propagate` does).
                let (below, above) = pyramid.levels.split_at_mut(level as usize - 1);
                let child_lc = &below[level as usize - 2][&child];
                let lc = above[0].entry(pos).or_insert_with(LodCube::empty);
                downsample_octant(&child_lc.cube, &mut lc.cube, octant);
                lc.known |= 1 << octant;
            }
            (min, max) = (parent(min).0, parent(max).0);
            pyramid.fill_box(level, min, max, generator);
        }
        pyramid
    }

    /// Fills every cube of `level` in the box `min..=max` (per axis) that
    /// the pyramid holds, plus creates-and-fills the missing ones: the
    /// reference's worldgen half.
    fn fill_box(&mut self, level: u8, min: CubePos, max: CubePos, generator: &Generator) {
        let mut positions: Vec<CubePos> = self.levels[level as usize - 1].keys().copied().collect();
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    positions.push(CubePos { x, y, z });
                }
            }
        }
        positions.sort_unstable();
        positions.dedup();
        for pos in positions {
            let sample = generator.sample_lod(level, pos);
            let lc = self.entry(level, pos);
            fill_octants(lc, &sample, 0);
        }
    }

    // ---- Streaming interface (tasks 2.6 and 2.7) ----
    //
    // 2.6 streams through exactly these calls and `take_changed`; 2.7
    // implements them (worldgen fill, storage) behind the contract below.

    /// The cube at `(level, pos)` if it is ready to stream: every octant
    /// holds real or worldgen-filled content. A ready cube can still change
    /// later (an edit, a LOD0 cube replacing a worldgen guess); such
    /// changes are reported by [`LodPyramid::take_changed`].
    pub fn ready(&self, level: u8, pos: CubePos) -> Option<&Cube> {
        self.get(level, pos)
            .filter(|lc| lc.ready())
            .map(|lc| &lc.cube)
    }

    /// Ask for `(level, pos)` to become ready: a load-or-sample job is
    /// queued (deduplicated; the caller caps itself with
    /// [`LodPyramid::in_flight`]). A no-op for a cube already ready, a
    /// level outside `1..=max_levels`, a pyramid without a generator, or a
    /// cube whose only missing octants are pending (their content comes
    /// from the child, not from a job).
    pub fn request(&mut self, level: u8, pos: CubePos) {
        self.request_checked(level, pos, false);
    }

    fn request_checked(&mut self, level: u8, pos: CubePos, internal: bool) -> bool {
        if !self.enabled() || self.generator.is_none() {
            return false;
        }
        if level == 0 || level > self.max_levels() {
            return false;
        }
        if let Some(lc) = self.get(level, pos) {
            if lc.ready() {
                return false;
            }
            // Only octants with no content at all can come from a job.
            if lc.known | lc.pending | lc.filled == 0xFF {
                return false;
            }
        }
        let job = Job { level, pos };
        if self.in_flight.contains_key(&job) {
            return false;
        }
        self.in_flight.insert(job, internal);
        if internal {
            self.internal_requests += 1;
        }
        match &self.job_tx {
            Some(tx) => {
                let _ = tx.send(job);
            }
            None => self.inline_queue.push_back(job),
        }
        true
    }

    /// The pyramid's own requests (deferred propagations). They are uncapped
    /// and do not count in [`LodPyramid::in_flight`]: they are self-limiting
    /// (at most one job per pyramid cube, deduplicated) and correctness-
    /// bearing, so they must never consume — or wait behind — the streaming
    /// budget; a burst of LOD0 ingest during exploration would otherwise
    /// crowd streaming's own requests out of its cap for as long as it
    /// lasts.
    fn request_internal(&mut self, level: u8, pos: CubePos) {
        self.request_checked(level, pos, true);
    }

    /// Streaming's requests not yet ready — the number the caller caps with
    /// `ServerConfig::lod_max_in_flight`. The pyramid's own (internal,
    /// pending-propagation) requests are excluded on purpose, so they can
    /// never starve streaming's.
    pub fn in_flight(&self) -> usize {
        self.in_flight
            .values()
            .filter(|internal| !**internal)
            .count()
    }

    /// Internal (pending-propagation) requests queued in total. A load
    /// report number: how much fill work the pyramid asks of itself.
    pub fn internal_requests(&self) -> u64 {
        self.internal_requests
    }

    /// All jobs queued, running or finished-but-uncollected, internal ones
    /// included. The drain observable: zero means no fill work remains
    /// anywhere in the pipeline.
    pub fn in_flight_total(&self) -> usize {
        self.in_flight.len()
    }

    /// Collect finished background work (worldgen samples, storage loads)
    /// into the pyramid, once per tick by the server, after LOD0 results
    /// are collected and before streaming. In inline mode this runs up to
    /// `inline_budget` jobs synchronously; in threaded mode it drains every
    /// finished result. Cubes it makes ready or changes are reported by
    /// `take_changed`. A worker panic is re-raised here.
    pub fn poll(&mut self) {
        if !self.enabled() || self.generator.is_none() {
            return;
        }
        if self.threads.is_empty() {
            let mut done = 0;
            while done < self.inline_budget.max(1) {
                let Some(job) = self.inline_queue.pop_front() else {
                    break;
                };
                self.run_job(job);
                done += 1;
            }
        } else {
            loop {
                match self.results_rx.try_recv() {
                    Ok(JobResult::Done(job, cube, known)) => {
                        self.in_flight.remove(&job);
                        self.merge(job, &cube, known);
                    }
                    Ok(JobResult::Panicked(job, payload)) => {
                        self.in_flight.remove(&job);
                        panic::resume_unwind(payload);
                    }
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
                }
            }
        }
    }

    /// Runs one job inline (the deterministic mode) and merges it.
    fn run_job(&mut self, job: Job) {
        self.in_flight.remove(&job);
        let Some(generator) = self.generator.clone() else {
            return;
        };
        {
            let mut state = self.hooks.state.lock().unwrap();
            if state.fail_next {
                state.fail_next = false;
                panic!("injected LOD job failure (test hook)");
            }
        }
        let (cube, known) = load_or_sample(self.storage.as_ref(), &generator, job);
        self.merge(job, &cube, known);
    }

    /// Merges one job result into its cube: real octants win (memory's own
    /// real data, or the stored record's), pending octants wait for their
    /// child, everything else takes the job's content. Then the deferred
    /// propagations that became possible run.
    fn merge(&mut self, job: Job, result: &Cube, result_known: u8) {
        let was_ready = self.get(job.level, job.pos).is_some_and(|lc| lc.ready());
        {
            let lc = self.entry(job.level, job.pos);
            fill_octants(lc, result, result_known);
        }
        self.touch(job.level, job.pos, was_ready, true);
        self.propagate(job.level, job.pos, None);
    }

    /// Every cube whose content changed since the last call, sorted by
    /// `(level, x, y, z)` and deduplicated — ready cubes only, so
    /// streaming never sends a partial cube. Streaming re-sends these to
    /// players who already have them; cubes nobody has are simply dropped
    /// from the set.
    pub fn take_changed(&mut self) -> Vec<(u8, CubePos)> {
        let mut changed: Vec<(u8, CubePos)> = self.changed.drain().collect();
        changed.sort_by_key(|&(level, pos)| (level, pos.x, pos.y, pos.z));
        changed
    }

    /// Ready cubes whose content changed since the last call — the
    /// persistence half of [`LodPyramid::take_changed`], kept separate so
    /// draining one never starves the other. Sorted like `take_changed`.
    pub fn take_store_dirty(&mut self) -> Vec<(u8, CubePos)> {
        let mut dirty: Vec<(u8, CubePos)> = self.store_dirty.drain().collect();
        dirty.sort_by_key(|&(level, pos)| (level, pos.x, pos.y, pos.z));
        dirty
    }

    /// Re-marks keys as needing a save (a failed save batch retries next
    /// interval), mirroring the LOD0 pending-save rule.
    pub fn mark_store_dirty(&mut self, keys: &[(u8, CubePos)]) {
        if self.storage.is_some() {
            self.store_dirty.extend(keys.iter().copied());
        }
    }

    /// Test hook: workers that have taken a job block until
    /// [`LodPyramid::release_workers`], making "a LOD0 insert lands while
    /// a sample job is in flight" deterministic.
    pub fn hold_workers(&self) {
        self.hooks.state.lock().unwrap().held = true;
    }

    /// Releases [`LodPyramid::hold_workers`].
    pub fn release_workers(&self) {
        let mut state = self.hooks.state.lock().unwrap();
        state.held = false;
        self.hooks.released.notify_all();
    }

    /// Test hook: the next job run on a worker (or inline) panics; `poll`
    /// re-raises it on the tick thread.
    pub fn fail_next_job(&self) {
        self.hooks.state.lock().unwrap().fail_next = true;
    }

    /// The level-`level` cube at `pos`, if the pyramid holds one — ready or
    /// not. Levels are 1-based; 0 and anything above
    /// [`LodPyramid::max_levels`] give `None`.
    pub fn get(&self, level: u8, pos: CubePos) -> Option<&LodCube> {
        if level == 0 {
            return None;
        }
        self.levels.get(level as usize - 1)?.get(&pos)
    }

    /// Every cube of `level`, for tests and (later) streaming.
    pub fn iter(&self, level: u8) -> impl Iterator<Item = (&CubePos, &LodCube)> + '_ {
        let index = (level as usize).wrapping_sub(1);
        self.levels
            .get(index)
            .into_iter()
            .flat_map(|map| map.iter())
    }

    /// Cubes held at `level`; 0 outside `1..=max_levels`.
    pub fn len(&self, level: u8) -> usize {
        if level == 0 {
            return 0;
        }
        self.levels.get(level as usize - 1).map_or(0, HashMap::len)
    }

    /// Cubes held across all levels.
    pub fn total_cubes(&self) -> usize {
        self.levels.iter().map(|map| map.len()).sum()
    }

    /// Approximate heap bytes across all levels: every cube's
    /// [`Cube::heap_bytes`] plus its entry, plus map capacity. Allocator
    /// overhead is not counted.
    pub fn heap_bytes(&self) -> usize {
        self.levels
            .iter()
            .map(|map| {
                map.values()
                    .map(|lc| lc.cube.heap_bytes() + size_of::<LodCube>())
                    .sum::<usize>()
                    + map.capacity() * (size_of::<(CubePos, LodCube)>() + 1)
            })
            .sum()
    }
}

impl Drop for LodPyramid {
    fn drop(&mut self) {
        // Same contract as the generation pool: flag shutdown so workers
        // skip queued-but-unstarted jobs, close the queue, join everyone.
        // A worker blocked in `recv` wakes immediately; one held by the
        // test hook is woken by `wake_all` and re-checks shutdown, so a
        // held worker never hangs the drop either.
        self.shutdown.store(true, Ordering::Release);
        self.hooks.wake_all();
        self.job_tx = None;
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Fills `lc`'s non-real, non-pending octants from `result`: `result_known`
/// marks octants the stored record held as real (they become real here
/// too); the rest are worldgen-filled. Real octants in memory always win.
fn fill_octants(lc: &mut LodCube, result: &Cube, result_known: u8) {
    for octant in 0..8u8 {
        let bit = 1 << octant;
        if lc.known & bit != 0 || lc.pending & bit != 0 {
            continue;
        }
        let becomes_real = result_known & bit != 0;
        if !becomes_real && lc.filled & bit != 0 {
            continue;
        }
        copy_octant(result, &mut lc.cube, octant);
        if becomes_real {
            lc.known |= bit;
        } else {
            lc.filled |= bit;
        }
    }
}

/// The pipeline's load-or-sample step: a stored record wins (with its real
/// mask), a missing or unreadable one is sampled from worldgen. A corrupt
/// record is logged with its position and resampled — never a panic — and
/// the record on disk stays untouched until the cube is saved again.
fn load_or_sample(storage: Option<&Arc<Storage>>, generator: &Generator, job: Job) -> (Cube, u8) {
    if let Some(storage) = storage {
        match storage.load_lod(job.level, job.pos) {
            Ok(Some((cube, known))) => return (cube, known),
            Ok(None) => {}
            Err(StorageError::Corrupt { pos, reason }) => {
                eprintln!(
                    "jbm: corrupt LOD record at {pos:?} level {} ({reason}); resampling",
                    job.level
                );
            }
            Err(error) => {
                eprintln!(
                    "jbm: loading LOD cube {pos:?} level {} failed ({error}); resampling",
                    job.level,
                    pos = job.pos
                );
            }
        }
    }
    (generator.sample_lod(job.level, job.pos), 0)
}

/// The worker half of the fill pipeline. A separate module so the worker's
/// owned state is a plain struct, not fields of the tick-thread pyramid.
mod lod_worker {
    use super::{Hooks, Job, JobResult, load_or_sample};
    use crate::storage::Storage;
    use crate::terrain::Generator;
    use std::panic::{self, AssertUnwindSafe};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{Receiver, Sender};

    /// One worker's owned handles.
    pub struct Worker {
        pub generator: Arc<Generator>,
        pub storage: Option<Arc<Storage>>,
        pub jobs: Arc<Mutex<Receiver<Job>>>,
        pub results: Sender<JobResult>,
        pub shutdown: Arc<AtomicBool>,
        pub hooks: Arc<Hooks>,
    }

    /// The worker loop: take a job (the queue lock held only around
    /// `recv`), run it, report. A panic is caught and forwarded so `poll`
    /// can re-raise it on the tick thread; the worker carries on.
    pub fn run(worker: Worker) {
        loop {
            let job = worker.jobs.lock().unwrap().recv();
            let job = match job {
                Ok(job) => job,
                Err(_) => break, // queue closed: shutting down
            };
            if worker.shutdown.load(Ordering::Acquire) {
                break;
            }
            worker.hooks.enter_job(&worker.shutdown);
            let Worker {
                ref generator,
                ref storage,
                ..
            } = worker;
            match panic::catch_unwind(AssertUnwindSafe(|| {
                load_or_sample(storage.as_ref(), generator, job)
            })) {
                Ok((cube, known)) => {
                    let _ = worker.results.send(JobResult::Done(job, cube, known));
                }
                Err(payload) => {
                    let _ = worker.results.send(JobResult::Panicked(job, payload));
                }
            }
        }
    }
}
