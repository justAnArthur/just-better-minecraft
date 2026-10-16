//! The generation pool: cubes are generated off the tick thread by named
//! worker threads, or inline on the tick thread in deterministic mode.
//!
//! Ownership model: exactly one thread (the server's tick thread) calls
//! [`GenPool::request`] and [`GenPool::try_recv`]; the workers only take jobs
//! from the queue and push results onto the results channel. That keeps the
//! in-flight dedup set plain data with no locking.

use std::collections::{HashSet, VecDeque};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};

use jbm_core::block::BlockStateId;
use jbm_core::coords::CubePos;
use jbm_core::cube::Cube;
use jbm_core::lod::downsample_octant;
use jbm_core::lod::parent;

use crate::storage::{Storage, StorageError};
use crate::terrain::Generator;

/// One finished (or exploded) job, as a worker reports it. The second cube
/// is the LOD0 cube's precomputed level-1 octant (task 2.7): the worker
/// downsamples off the tick thread, into a uniform-air scratch cube at the
/// octant's own slot of its level-1 parent, so pyramid ingest on the tick
/// thread is a bulk copy (`copy_octant`), not a downsample.
enum JobResult {
    Done(CubePos, Cube, Cube),
    /// A worker panicked generating this cube; the payload is re-raised on
    /// the tick thread by `try_recv` so the panic never silently vanishes.
    Panicked(CubePos, Box<dyn std::any::Any + Send>),
}

/// The level-1 octant of a LOD0 cube, as the workers hand it back: a
/// uniform-air scratch cube with the downsample installed at the cube's
/// octant slot of its parent.
fn lod1_octant(pos: CubePos, cube: &Cube) -> Cube {
    let (_, octant) = parent(pos);
    let mut scratch = Cube::filled(BlockStateId::AIR);
    downsample_octant(cube, &mut scratch, octant);
    scratch
}

/// Cube generation pool. With `threads == 0` it runs in inline mode: requests
/// queue up and [`GenPool::try_recv`] generates synchronously on the calling
/// (tick) thread — the deterministic mode tests and benches use.
///
/// Dropping the pool closes the job queue and joins every worker thread. It
/// never detaches and never hangs: workers block only on `recv`, which
/// errors out the moment the queue's sender side is dropped, and a shutdown
/// flag tells them to skip queued-but-unstarted jobs, so the threads exit
/// promptly even with a deep backlog.
pub struct GenPool {
    generator: Arc<Generator>,
    /// When set, positions are loaded from disk first and generated only as
    /// a fallback (task 2.3).
    storage: Option<Arc<Storage>>,
    /// `None` in inline mode.
    queue_tx: Option<Sender<CubePos>>,
    shutdown: Arc<AtomicBool>,
    /// Pending inline-mode requests, in request order.
    inline_queue: VecDeque<CubePos>,
    results_rx: Receiver<JobResult>,
    /// Positions queued, being generated, or generated but not yet collected.
    in_flight: HashSet<CubePos>,
    threads: Vec<JoinHandle<()>>,
}

impl GenPool {
    /// Build a pool with `threads` workers named `jbm-gen-N`, or an inline
    /// pool when `threads == 0`. With `storage`, every request is
    /// load-or-generate: a stored cube is read from disk, everything else
    /// (including a corrupt record, which is logged and left untouched) is
    /// generated.
    pub fn new(threads: usize, generator: Arc<Generator>, storage: Option<Arc<Storage>>) -> Self {
        let (results_tx, results_rx) = std::sync::mpsc::channel();
        let mut pool = Self {
            generator,
            storage,
            queue_tx: None,
            shutdown: Arc::new(AtomicBool::new(false)),
            inline_queue: VecDeque::new(),
            results_rx,
            in_flight: HashSet::new(),
            threads: Vec::with_capacity(threads),
        };
        if threads > 0 {
            let (queue_tx, queue_rx) = std::sync::mpsc::channel::<CubePos>();
            // `mpsc::Receiver` is not `Sync`, so workers wait on it through
            // a mutex — the classic sharing pattern. The lock is held only
            // around `recv`, never while generating.
            let queue_rx = Arc::new(Mutex::new(queue_rx));
            for id in 0..threads {
                let worker_generator = Arc::clone(&pool.generator);
                let worker_queue = Arc::clone(&queue_rx);
                let worker_results = results_tx.clone();
                let worker_shutdown = Arc::clone(&pool.shutdown);
                let worker_storage = pool.storage.clone();
                let handle = thread::Builder::new()
                    .name(format!("jbm-gen-{id}"))
                    .spawn(move || {
                        worker(
                            worker_generator,
                            worker_storage,
                            worker_queue,
                            worker_results,
                            worker_shutdown,
                        )
                    })
                    .expect("spawning a jbm-gen worker thread");
                pool.threads.push(handle);
            }
            pool.queue_tx = Some(queue_tx);
        }
        pool
    }

    /// Queue `pos` for generation. Requests are deduplicated while in flight:
    /// a position that is queued, being generated, or generated but not yet
    /// collected is not queued again.
    pub fn request(&mut self, pos: CubePos) {
        if !self.in_flight.insert(pos) {
            return;
        }
        match &self.queue_tx {
            Some(tx) => {
                // Cannot fail: the pool owns the receiving side's lifetime.
                let _ = tx.send(pos);
            }
            None => self.inline_queue.push_back(pos),
        }
    }

    /// Take one finished result if one is ready: the position, the cube,
    /// and the cube's precomputed level-1 octant scratch. In inline mode
    /// this generates the next queued cube (and its octant) synchronously
    /// on the calling thread.
    ///
    /// If a worker panicked generating a cube, that panic is resumed here —
    /// panics surface on the tick thread instead of vanishing with the
    /// worker.
    pub fn try_recv(&mut self) -> Option<(CubePos, Cube, Cube)> {
        if self.threads.is_empty() {
            let pos = self.inline_queue.pop_front()?;
            let cube = load_or_generate(self.storage.as_ref(), &self.generator, pos);
            let octant = lod1_octant(pos, &cube);
            self.in_flight.remove(&pos);
            Some((pos, cube, octant))
        } else {
            match self.results_rx.try_recv() {
                Ok(JobResult::Done(pos, cube, octant)) => {
                    self.in_flight.remove(&pos);
                    Some((pos, cube, octant))
                }
                Ok(JobResult::Panicked(pos, payload)) => {
                    self.in_flight.remove(&pos);
                    panic::resume_unwind(payload);
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
            }
        }
    }

    /// How many cubes are queued, being generated, or generated but not yet
    /// collected.
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }
}

impl Drop for GenPool {
    fn drop(&mut self) {
        // Flag shutdown first so workers skip queued-but-unstarted jobs, then
        // close the queue: a worker blocked in `recv` wakes immediately and
        // exits without touching any lock.
        self.shutdown.store(true, Ordering::Release);
        self.queue_tx = None;
        for handle in self.threads.drain(..) {
            // A worker that somehow died from an uncaught panic is already
            // finished; join returns promptly either way.
            let _ = handle.join();
        }
    }
}

/// The pool's load-or-generate step: a stored cube wins, a missing one is
/// generated, and a corrupt record is logged (with its position) and
/// regenerated — the record on disk stays untouched unless the cube is
/// edited and saved again.
fn load_or_generate(
    storage: Option<&Arc<Storage>>,
    generator: &Generator,
    pos: CubePos,
) -> jbm_core::cube::Cube {
    if let Some(storage) = storage {
        match storage.load(pos) {
            Ok(Some(cube)) => return cube,
            Ok(None) => {}
            Err(StorageError::Corrupt { pos, reason }) => {
                eprintln!("jbm: corrupt record at {pos:?} ({reason}); regenerating");
            }
            Err(error) => {
                eprintln!("jbm: loading cube {pos:?} failed ({error}); regenerating");
            }
        }
    }
    generator.generate(pos)
}

/// One worker's loop: take a position, generate, report. A panic inside
/// generation is caught and forwarded as [`JobResult::Panicked`] so it can
/// surface on the tick thread; the worker then carries on with the next job.
fn worker(
    generator: Arc<Generator>,
    storage: Option<Arc<Storage>>,
    queue: Arc<Mutex<Receiver<CubePos>>>,
    results: Sender<JobResult>,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        // Wait for the next job with the queue lock held only around recv.
        let job = queue.lock().unwrap().recv();
        let pos = match job {
            Ok(pos) => pos,
            Err(_) => break, // queue closed: the pool is shutting down
        };
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        match panic::catch_unwind(AssertUnwindSafe(|| {
            let cube = load_or_generate(storage.as_ref(), &generator, pos);
            let octant = lod1_octant(pos, &cube);
            (cube, octant)
        })) {
            Ok((cube, octant)) => {
                // The pool outlives its workers, so this cannot fail.
                let _ = results.send(JobResult::Done(pos, cube, octant));
            }
            Err(payload) => {
                let _ = results.send(JobResult::Panicked(pos, payload));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cube, GenPool, Generator};
    use jbm_core::block::BlockRegistry;
    use jbm_core::block::builtin::Blocks;
    use jbm_core::coords::CubePos;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn generator() -> Arc<Generator> {
        let mut registry = BlockRegistry::new();
        let blocks = Blocks::register(&mut registry);
        Arc::new(Generator::new(42, blocks))
    }

    #[test]
    fn inline_mode_generates_on_the_calling_thread_in_request_order() {
        let terrain = generator();
        let mut pool = GenPool::new(0, terrain, None);
        let wanted = [
            CubePos { x: 0, y: 2, z: 0 },
            CubePos { x: 1, y: 2, z: 0 },
            CubePos { x: 0, y: 3, z: 0 },
        ];
        assert_eq!(pool.in_flight(), 0);
        for pos in wanted {
            pool.request(pos);
        }
        assert_eq!(pool.in_flight(), 3);
        // Duplicate requests while in flight are dropped.
        pool.request(wanted[0]);
        assert_eq!(pool.in_flight(), 3);
        let got: Vec<CubePos> = wanted
            .iter()
            .map(|_| pool.try_recv().expect("a queued cube").0)
            .collect();
        assert_eq!(got, wanted);
        assert_eq!(pool.in_flight(), 0);
        assert!(pool.try_recv().is_none());
    }

    #[test]
    fn threaded_pool_returns_generated_cubes() {
        let terrain = generator();
        let mut pool = GenPool::new(2, Arc::clone(&terrain), None);
        let wanted: Vec<CubePos> = (0..8).map(|i| CubePos { x: i, y: 2, z: -i }).collect();
        for pos in &wanted {
            pool.request(*pos);
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut got = std::collections::HashSet::new();
        while got.len() < wanted.len() {
            assert!(Instant::now() < deadline, "generation timed out");
            if let Some((pos, cube, octant)) = pool.try_recv() {
                assert_eq!(cube, terrain.generate(pos));
                // The precomputed level-1 octant is the cube's downsample
                // at its own slot of the parent.
                let (_, octant_index) = jbm_core::lod::parent(pos);
                let mut expected = Cube::filled(jbm_core::block::BlockStateId::AIR);
                jbm_core::lod::downsample_octant(&cube, &mut expected, octant_index);
                assert_eq!(octant, expected, "octant scratch for {pos:?}");
                got.insert(pos);
            }
        }
        assert_eq!(pool.in_flight(), 0);
    }

    #[test]
    fn drop_with_queued_jobs_returns_promptly() {
        let terrain = generator();
        let start = Instant::now();
        {
            let mut pool = GenPool::new(2, terrain, None);
            // Far more queued jobs than could be processed quickly.
            for i in 0..4096 {
                pool.request(CubePos { x: i, y: 2, z: 0 });
            }
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "pool drop with queued jobs took {elapsed:?}"
        );
    }
}
