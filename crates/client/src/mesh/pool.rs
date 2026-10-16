//! The mesh worker pool: named threads pulling mesh jobs off a shared
//! queue, results delivered through an mpsc the client polls without
//! blocking. See crates/client/README.md for the threading model and
//! shutdown semantics.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::lod::CubeKey;
use crate::mesh::appearance::BlockAppearance;
use crate::mesh::greedy::{self, MeshInput, MeshOutput};

/// How long a worker blocks on the job queue before re-checking for
/// shutdown. Bounded so dropping the pool joins every thread promptly even
/// while the queue mutex is held.
const QUEUE_POLL: Duration = Duration::from_millis(50);

/// A finished mesh, stamped with the cube's version at submit time. The
/// client drops any result whose version is older than the cube's latest.
pub struct MeshResult {
    pub key: CubeKey,
    pub version: u32,
    pub output: MeshOutput,
}

/// What a worker sends back: a result, or a panic it caught.
enum Outcome {
    Done(MeshResult),
    Panic(String),
}

enum Job {
    Mesh(Box<MeshJob>),
    /// Test-only: proves worker panics surface through `try_recv`.
    #[cfg(test)]
    Panic,
}

struct MeshJob {
    key: CubeKey,
    version: u32,
    input: MeshInput,
}

/// Fixed-size pool of mesh workers. Dropping it stops accepting work,
/// discards queued jobs, lets in-flight jobs finish and joins every
/// thread. Threads are never detached.
pub struct MeshPool {
    appearance: Arc<BlockAppearance>,
    jobs: Option<Sender<Job>>,
    stop: Arc<AtomicBool>,
    results: Receiver<Outcome>,
    workers: Vec<JoinHandle<()>>,
}

impl MeshPool {
    /// Creates a pool with `threads` workers (0 = default:
    /// `available_parallelism() - 2`, minimum 1). Threads are named
    /// `jbm-mesh-N`. `appearance` is shared read-only with every worker.
    pub fn new(appearance: Arc<BlockAppearance>, threads: usize) -> MeshPool {
        let threads = if threads == 0 {
            thread::available_parallelism()
                .map_or(1, |n| n.get())
                .saturating_sub(2)
                .max(1)
        } else {
            threads
        };
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let stop = Arc::new(AtomicBool::new(false));
        let (result_tx, result_rx) = mpsc::channel::<Outcome>();
        let workers = (0..threads)
            .map(|i| {
                let jobs = Arc::clone(&job_rx);
                let results = result_tx.clone();
                let appearance = Arc::clone(&appearance);
                let stop = Arc::clone(&stop);
                thread::Builder::new()
                    .name(format!("jbm-mesh-{i}"))
                    .spawn(move || worker_loop(jobs, results, appearance, stop))
                    .expect("spawning a jbm-mesh worker")
            })
            .collect();
        MeshPool {
            appearance,
            jobs: Some(job_tx),
            stop,
            results: result_rx,
            workers,
        }
    }

    /// Queues one mesh job at any level. Non-blocking; never detaches
    /// work.
    pub fn submit(&mut self, key: CubeKey, version: u32, input: MeshInput) {
        let sender = self
            .jobs
            .as_ref()
            .expect("the job queue lives until the pool is dropped");
        sender
            .send(Job::Mesh(Box::new(MeshJob {
                key,
                version,
                input,
            })))
            .expect("mesh workers live until the pool is dropped");
    }

    /// Takes the next finished mesh, or `None` without blocking. A worker
    /// panic is re-raised here with the worker's message instead of being
    /// silently swallowed.
    pub fn try_recv(&mut self) -> Option<MeshResult> {
        match self.results.try_recv() {
            Ok(Outcome::Done(result)) => Some(result),
            Ok(Outcome::Panic(message)) => panic!("mesh worker panicked: {message}"),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    /// Number of worker threads.
    pub fn threads(&self) -> usize {
        self.workers.len()
    }

    /// The shared appearance table (for callers that mesh directly).
    pub fn appearance(&self) -> &Arc<BlockAppearance> {
        &self.appearance
    }
}

impl Drop for MeshPool {
    fn drop(&mut self) {
        // Stop first so workers discard anything still queued (mpsc would
        // otherwise drain the backlog before reporting Disconnected), then
        // close the queue to wake the ones blocked on recv. An in-flight
        // job finishes; its send harmlessly fails once the results
        // receiver is gone.
        self.stop.store(true, Ordering::Release);
        self.jobs.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(
    jobs: Arc<Mutex<Receiver<Job>>>,
    results: Sender<Outcome>,
    appearance: Arc<BlockAppearance>,
    stop: Arc<AtomicBool>,
) {
    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let job = {
            // Holding the lock across recv_timeout is fine: it only
            // serializes idle waiting, never meshing.
            let guard = jobs.lock().expect("mesh job queue mutex poisoned");
            match guard.recv_timeout(QUEUE_POLL) {
                Ok(job) => Some(job),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        };
        let Some(job) = job else {
            continue;
        };
        if stop.load(Ordering::Acquire) {
            break; // pool dropped: discard without running
        }
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| run_job(job, &appearance)))
            .unwrap_or_else(|payload| Outcome::Panic(panic_message(&*payload)));
        if results.send(outcome).is_err() {
            // The pool is gone; stop.
            break;
        }
    }
}

fn run_job(job: Job, appearance: &BlockAppearance) -> Outcome {
    match job {
        Job::Mesh(job) => {
            let output = greedy::mesh(&job.input, appearance);
            Outcome::Done(MeshResult {
                key: job.key,
                version: job.version,
                output,
            })
        }
        #[cfg(test)]
        Job::Panic => panic!("injected worker panic"),
    }
}

/// Best-effort extraction of a panic message. Callers must pass `&*box`
/// (the pointee, not `&box`: a `&Box<dyn Any>` coerces to `&dyn Any` of
/// the *box*, and every downcast fails — a classic trap).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
    use jbm_core::coords::CubePos;
    use jbm_core::cube::Cube;

    fn appearance() -> Arc<BlockAppearance> {
        let mut registry = BlockRegistry::new();
        builtin::register(&mut registry);
        Arc::new(BlockAppearance::new(&registry))
    }

    /// A cheap but non-trivial cube: a diagonal plane of stone.
    fn demo_cube(seed: u64) -> Arc<Cube> {
        Arc::new(Cube::from_fn(|p| {
            let mixed = (p.index() as u64).wrapping_mul(seed).wrapping_add(seed);
            match mixed % 5 {
                0 => BlockStateId(1), // stone
                1 => BlockStateId(2), // dirt
                _ => BlockStateId::AIR,
            }
        }))
    }

    fn pos(i: i32) -> CubeKey {
        CubeKey::lod0(CubePos { x: i, y: 0, z: 0 })
    }

    #[test]
    fn thousand_jobs_come_back_identical() {
        let appearance = appearance();
        let mut pool = MeshPool::new(Arc::clone(&appearance), 4);
        let cubes: Vec<_> = (0..1000)
            .map(|i| (pos(i), demo_cube(i as u64 + 1)))
            .collect();
        for (i, (p, cube)) in cubes.iter().enumerate() {
            pool.submit(*p, i as u32 + 1, MeshInput::isolated(Arc::clone(cube)));
        }
        let mut results = std::collections::HashMap::new();
        while results.len() < 1000 {
            if let Some(result) = pool.try_recv() {
                // Versions are stamped per job; arrival order is not.
                assert!((1..=1000).contains(&result.version));
                results.insert(result.key, result.output);
            } else {
                thread::sleep(Duration::from_millis(1));
            }
        }
        for (p, cube) in &cubes {
            let expected = greedy::mesh(&MeshInput::isolated(Arc::clone(cube)), &appearance);
            assert_eq!(&results[p], &expected, "pool output differs at {p:?}");
        }
    }

    #[test]
    fn results_carry_their_version_so_stale_ones_are_droppable() {
        let appearance = appearance();
        let mut pool = MeshPool::new(Arc::clone(&appearance), 1);
        let cube = demo_cube(7);
        let p = pos(0);
        pool.submit(p, 3, MeshInput::isolated(Arc::clone(&cube)));
        pool.submit(p, 4, MeshInput::isolated(Arc::clone(&cube)));
        let mut seen = Vec::new();
        while seen.len() < 2 {
            if let Some(result) = pool.try_recv() {
                assert_eq!(result.key, p);
                seen.push(result.version);
            }
        }
        assert_eq!(seen, vec![3, 4]);
        // The client-side rule: anything older than the cube's latest
        // version is dropped, which the carried version makes possible.
        let latest = 4;
        assert!(seen.iter().all(|v| *v <= latest));
    }

    #[test]
    fn dropping_with_queued_jobs_returns_promptly() {
        let appearance = appearance();
        let mut pool = MeshPool::new(Arc::clone(&appearance), 2);
        for i in 0..500 {
            pool.submit(pos(i), 1, MeshInput::isolated(demo_cube(i as u64 + 1)));
        }
        let start = std::time::Instant::now();
        drop(pool);
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "drop took {elapsed:?} with jobs queued"
        );
    }

    #[test]
    fn worker_panics_surface_through_try_recv() {
        let appearance = appearance();
        let mut pool = MeshPool::new(Arc::clone(&appearance), 1);
        pool.jobs
            .as_ref()
            .unwrap()
            .send(Job::Panic)
            .expect("queue open");
        let outcome = loop {
            match pool.results.try_recv() {
                Ok(outcome) => break outcome,
                Err(TryRecvError::Empty) => thread::sleep(Duration::from_millis(1)),
                Err(TryRecvError::Disconnected) => panic!("worker died without reporting"),
            }
        };
        // The outcome carries the message; try_recv re-raises it.
        match outcome {
            Outcome::Panic(message) => assert!(message.contains("injected worker panic")),
            Outcome::Done(_) => panic!("expected a panic outcome"),
        }
        // And the worker survived for later jobs.
        pool.submit(pos(0), 1, MeshInput::isolated(demo_cube(1)));
        let mut got = None;
        while got.is_none() {
            if let Some(result) = pool.try_recv() {
                got = Some(result);
            }
        }
        assert_eq!(got.unwrap().key, pos(0));
    }

    #[test]
    fn default_thread_count_is_available_parallelism_minus_two() {
        let appearance = appearance();
        let expected = thread::available_parallelism()
            .map_or(1, |n| n.get())
            .saturating_sub(2)
            .max(1);
        assert_eq!(
            MeshPool::new(Arc::clone(&appearance), 0).threads(),
            expected
        );
        assert_eq!(MeshPool::new(Arc::clone(&appearance), 3).threads(), 3);
    }
}
