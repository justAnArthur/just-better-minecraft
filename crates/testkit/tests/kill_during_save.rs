//! Crash-atomicity tests (task 2.3): a child process is SIGKILLed mid-save
//! and the parent proves every committed transaction is all-or-nothing.
//!
//! The children (`kill_child_storage`, `kill_child_server`) are gated by
//! `JBM_KILL_SAVE_CHILD`; they loop until killed, so they do nothing when
//! the suite runs them normally. `Child::kill` is SIGKILL on unix, so the
//! parent tests are `#[cfg(unix)]` (CI runs macOS and Ubuntu).

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{BlockPos, CubePos, LocalPos};
use jbm_core::cube::Cube;
use jbm_core::protocol::ClientMsg;
use jbm_server::DEFAULT_SEED;
use jbm_server::server::{Server, ServerConfig};
use jbm_server::storage::Storage;
use jbm_server::terrain::Generator;
use jbm_testkit::HeadlessServer;

/// Environment variable selecting which child role the test binary runs.
const CHILD_VAR: &str = "JBM_KILL_SAVE_CHILD";
/// Where the child puts its world.
const DIR_VAR: &str = "JBM_KILL_WORLD_DIR";

/// Small local xorshift so the kill timing is seeded, not wall-clock lucky.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn builtin_registry() -> BlockRegistry {
    let mut registry = BlockRegistry::new();
    Blocks::register(&mut registry);
    registry
}

/// The 64 cubes the storage-level child rewrites every batch: a 4×4×4 block.
fn positions() -> Vec<CubePos> {
    let mut positions = Vec::new();
    for x in -2..2 {
        for y in 2..6 {
            for z in -2..2 {
                positions.push(CubePos { x, y, z });
            }
        }
    }
    positions
}

/// The cube written for batch `k` at `pos`: a pattern derived from `k`, the
/// position and the cell, shared verbatim by the child and the parent.
fn pattern_cube(k: u32, pos: CubePos) -> Cube {
    let mix = |a: u32, b: u32| a.wrapping_mul(0x9e37_79b9).wrapping_add(b);
    Cube::from_fn(|local: LocalPos| {
        let h = mix(k, (local.index() as u32).wrapping_mul(31));
        let h = mix(h, (pos.x as u32).wrapping_mul(7));
        let h = mix(h, (pos.z as u32).wrapping_mul(13));
        jbm_core::block::BlockStateId(1 + h % 11)
    })
}

/// Runs the test binary as the `role` child on `dir`, waits for its
/// "started" line, and returns the child.
fn spawn_child(role: &str, dir: &std::path::Path, test: &str) -> Child {
    let exe = std::env::current_exe().expect("current test binary");
    let mut child = Command::new(exe)
        .arg(test)
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_VAR, role)
        .env(DIR_VAR, dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawning the child test");
    // Wait for "started" so the kill lands during the save loop, not during
    // process spin-up. The reader runs on a thread so a chatty child cannot
    // block on a full pipe.
    {
        let stdout = child.stdout.take().expect("piped stdout");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            while let Ok(n) = reader.read_line(&mut line) {
                if n == 0 {
                    break;
                }
                if line.trim() == "started" {
                    let _ = tx.send(());
                    break;
                }
                line.clear();
            }
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while rx.try_recv().is_err() {
            assert!(Instant::now() < deadline, "child never reported startup");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    child
}

/// Kills the child after a seeded 20–400 ms wait.
fn kill_after(child: &mut Child, rng: &mut XorShift64) {
    let ms = 20 + rng.next() % 381;
    std::thread::sleep(Duration::from_millis(ms));
    child.kill().expect("SIGKILL the child");
    let _ = child.wait();
}

#[test]
#[cfg(unix)]
fn kill_during_storage_save_never_tears_a_batch() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut rng = XorShift64(0x6b69_6c6c_5354_4f52);
    let registry = builtin_registry();
    for round in 1..=10 {
        let mut child = spawn_child("storage", dir.path(), "kill_child_storage");
        kill_after(&mut child, &mut rng);

        // The database opens (no stuck lock from the killed process) and
        // every committed batch is complete and self-consistent.
        let storage = Storage::open(dir.path(), &registry, 42)
            .unwrap_or_else(|e| panic!("round {round}: reopen failed: {e}"));
        let k = storage
            .meta("last_batch")
            .expect("meta read")
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
            .expect("last_batch never written");
        assert!(k >= 1, "round {round}: no batch ever committed");
        for pos in positions() {
            let cube = storage
                .load(pos)
                .unwrap_or_else(|e| panic!("round {round}: cube {pos:?} failed to load: {e}"))
                .unwrap_or_else(|| panic!("round {round}: cube {pos:?} missing"));
            assert_eq!(
                cube,
                pattern_cube(k, pos),
                "round {round}: cube {pos:?} is not from batch {k}"
            );
        }
        assert_eq!(storage.cube_count().unwrap(), 64);
    }
}

/// The placement order both sides derive from the deterministic terrain:
/// cells above each column's surface, within reach of the spawn eye,
/// bottom-up in a fixed footprint sweep. Every cell is provably air and
/// in reach, so the server accepts every placement.
fn placement_sequence(generator: &Generator) -> Vec<BlockPos> {
    let h00 = generator.surface_height(0, 0);
    let eye = (0.5, h00 as f64 + 3.62, 0.5);
    let mut cells = Vec::new();
    for level in 1..=12i64 {
        for x in 1..=5i64 {
            for z in -3..=3i64 {
                let y = h00 + level;
                if y <= generator.surface_height(x, z) {
                    continue; // generated terrain, not air
                }
                let dx = x as f64 + 0.5 - eye.0;
                let dy = y as f64 + 0.5 - eye.1;
                let dz = z as f64 + 0.5 - eye.2;
                if dx * dx + dy * dy + dz * dz <= 36.0 {
                    cells.push(BlockPos { x, y, z });
                }
            }
        }
    }
    cells
}

fn placed_state() -> jbm_core::block::BlockStateId {
    builtin_registry()
        .state("base:glass", &[])
        .expect("glass is registered")
}

fn generator() -> Generator {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    Generator::new(DEFAULT_SEED, blocks)
}

#[test]
#[cfg(unix)]
fn kill_during_server_save_keeps_a_prefix_of_placements() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut rng = XorShift64(0x5352_5650_5245_4658);
    let mut child = spawn_child("server", dir.path(), "kill_child_server");
    kill_after(&mut child, &mut rng);

    // A fresh server on the same directory (no stuck lock)...
    let config = ServerConfig {
        world_dir: Some(dir.path().to_owned()),
        save_interval: Duration::from_millis(50),
        streaming: false,
        gen_threads: 0,
        ..ServerConfig::default()
    };
    let server = Server::try_with_config(config).expect("reopen the world");
    let storage = server.storage().expect("persistence was on");
    assert_eq!(storage.seed(), DEFAULT_SEED);

    // ...loads every stored cube without error...
    let stored = storage.cube_positions().unwrap();
    for pos in stored {
        storage
            .load(pos)
            .unwrap_or_else(|e| panic!("stored cube {pos:?} failed to load: {e}"))
            .unwrap_or_else(|| panic!("stored cube {pos:?} missing"));
    }

    // ...and the placed blocks form a prefix of the placement sequence:
    // block i present implies i is exactly one past the previous present.
    let glass = placed_state();
    let placements = placement_sequence(&generator());
    assert!(
        placements.len() > 100,
        "sequence too short to outlive a kill"
    );
    let mut last_present: Option<usize> = None;
    for (i, pos) in placements.iter().enumerate() {
        let present = storage
            .load(pos.split().0)
            .expect("load")
            .is_some_and(|cube| cube.get(pos.split().1) == glass);
        if present {
            let contiguous = match last_present {
                Some(last) => i == last + 1,
                None => i == 0,
            };
            assert!(
                contiguous,
                "placement {i} at {pos:?} is present but the sequence is torn \
                 (last present: {last_present:?})"
            );
            last_present = Some(i);
        }
    }
    assert!(
        last_present.is_some(),
        "no placement was ever committed before the kill"
    );
}

// ---------------------------------------------------------------------
// Children. They run only when the parent re-executes this binary with
// JBM_KILL_SAVE_CHILD set; otherwise they return immediately.

#[test]
fn kill_child_storage() {
    let Ok(role) = std::env::var(CHILD_VAR) else {
        return;
    };
    if role != "storage" {
        return;
    }
    let dir = std::env::var(DIR_VAR).expect("child gets the world dir");
    let registry = builtin_registry();
    let storage = Storage::open(&dir, &registry, 42).expect("child opens storage");
    println!("started");
    std::io::stdout().flush().unwrap();
    let mut k = 0u32;
    loop {
        k += 1;
        let cubes: Vec<Cube> = positions()
            .iter()
            .map(|&pos| pattern_cube(k, pos))
            .collect();
        let batch: Vec<(CubePos, &Cube)> = positions()
            .iter()
            .zip(&cubes)
            .map(|(&p, c)| (p, c))
            .collect();
        storage
            .save_batch_with_meta(&batch, &[("last_batch", k.to_le_bytes().as_slice())])
            .expect("child batch commits");
    }
}

#[test]
fn kill_child_server() {
    let Ok(role) = std::env::var(CHILD_VAR) else {
        return;
    };
    if role != "server" {
        return;
    }
    let dir = std::env::var(DIR_VAR).expect("child gets the world dir");
    let config = ServerConfig {
        world_dir: Some(std::path::PathBuf::from(dir)),
        save_interval: Duration::from_millis(50),
        streaming: true,
        view_radius: 1,
        vertical_radius: 1,
        send_budget_per_tick: 64,
        gen_threads: 0,
        gen_budget_per_tick: 64,
        max_in_flight: 512,
        ..ServerConfig::default()
    };
    let mut hs = HeadlessServer::with_config(config);
    let mut client = hs.connect("builder");
    hs.step(1);
    client.drain();
    // Settle terrain around spawn so placement targets are loaded cubes.
    for _ in 0..40 {
        hs.step(1);
        client.drain();
        if hs.server().world_cube_count() > 1 {
            break;
        }
    }
    println!("started");
    std::io::stdout().flush().unwrap();

    // One placement per paced tick along the shared deterministic sequence;
    // then keep ticking (and saving) until the kill lands.
    let glass = placed_state();
    for pos in placement_sequence(&generator()) {
        client.send(ClientMsg::PlaceBlock { pos, state: glass });
        hs.step(1);
        client.drain();
        // Real-time pacing: without it a 400 ms window runs thousands of
        // ticks and exhausts the sequence before the kill.
        std::thread::sleep(Duration::from_millis(5));
    }
    loop {
        hs.step(1);
        std::thread::sleep(Duration::from_millis(5));
    }
}
