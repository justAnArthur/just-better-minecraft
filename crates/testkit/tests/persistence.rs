//! Persistence integration tests (task 2.3): edits survive a restart, a
//! committed cube is unpinned and evicted, an edit during a save stays
//! pinned until the next batch, and a failed save is retried.

use std::time::Duration;

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::BlockPos;
use jbm_core::physics::MoveInput;
use jbm_core::protocol::{ClientMsg, ServerMsg};
use jbm_server::server::ServerConfig;
use jbm_testkit::{HeadlessServer, TestClient};
use tempfile::TempDir;

/// A streaming, inline-generation server on a temp world dir with a short
/// save interval (one batch per tick).
fn persistent_config(dir: &TempDir, save_interval: Duration) -> ServerConfig {
    ServerConfig {
        streaming: true,
        view_radius: 1,
        vertical_radius: 1,
        send_budget_per_tick: 64,
        gen_threads: 0,
        gen_budget_per_tick: 64,
        max_in_flight: 512,
        world_dir: Some(dir.path().to_owned()),
        save_interval,
        ..ServerConfig::default()
    }
}

fn glass() -> BlockStateId {
    let mut registry = jbm_core::block::BlockRegistry::new();
    Blocks::register(&mut registry).glass
}

/// Joins, settles terrain and returns (feet position, surface block).
fn settle(hs: &mut HeadlessServer, client: &mut TestClient) -> BlockPos {
    hs.step(1);
    client.drain();
    let mut surface = None;
    for _ in 0..20 {
        hs.step(1);
        client.drain();
        if !client.loaded().is_empty() {
            surface = Some(surface_block(hs));
            break;
        }
    }
    surface.expect("terrain never streamed in")
}

/// The topmost solid block in column (0, 0), from the spawn feet down.
fn surface_block(hs: &HeadlessServer) -> BlockPos {
    let feet = hs.server().players().next().unwrap().state.position;
    let mut y = feet.y.floor() as i64;
    loop {
        let solid = hs
            .server()
            .block(BlockPos { x: 0, y, z: 0 })
            .is_some_and(|state| state != BlockStateId::AIR);
        if solid {
            return BlockPos { x: 0, y, z: 0 };
        }
        y -= 1;
    }
}

/// Steps once and sleeps in real time until `cond` holds, bounded by
/// `tries`. Needed because HeadlessServer steps run in microseconds while
/// the save thread commits on real disk: simulated time alone can outrun
/// the acknowledgement by orders of magnitude.
fn wait_until(
    hs: &mut HeadlessServer,
    cond: impl Fn(&HeadlessServer) -> bool,
    tries: usize,
) -> bool {
    for _ in 0..tries {
        hs.step(1);
        if cond(hs) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

fn input(seq: u64, input: MoveInput) -> ClientMsg {
    ClientMsg::PlayerInput {
        seq,
        input,
        pitch: 0.0,
    }
}

/// Flies toward `yaw` (rising to `cruise_y` first), one input per tick,
/// until `cond` holds. Returns whether it did; always bounded.
fn fly_until(
    hs: &mut HeadlessServer,
    client: &mut TestClient,
    yaw: f32,
    cruise_y: f64,
    seq: &mut u64,
    mut cond: impl FnMut(&HeadlessServer, &TestClient) -> bool,
    max_ticks: usize,
) -> bool {
    for _ in 0..max_ticks {
        let state = hs.server().players().next().unwrap().state;
        let rising = state.position.y < cruise_y;
        let fly = MoveInput {
            up: rising.into(),
            forward: (!rising) as i8,
            sprint: true,
            fly_toggle: !state.flying,
            yaw,
            ..MoveInput::IDLE
        };
        *seq += 1;
        client.send(input(*seq, fly));
        hs.step(1);
        client.drain();
        if cond(hs, client) {
            return true;
        }
    }
    false
}

#[test]
fn edits_survive_a_restart() {
    let dir = TempDir::new().unwrap();
    let mut hs = HeadlessServer::with_config(persistent_config(&dir, Duration::from_millis(150)));
    let mut client = hs.connect("alice");
    let ground = settle(&mut hs, &mut client);
    let above = BlockPos {
        x: 1,
        y: ground.y + 1,
        z: 0,
    };

    // Break the surface block and place glass beside it (both land in the
    // spawn cube: one dirty cube, edit version 2).
    client.send(ClientMsg::BreakBlock { pos: ground });
    client.send(ClientMsg::PlaceBlock {
        pos: above,
        state: glass(),
    });
    hs.step(1);
    client.drain();
    assert_eq!(hs.server().block(ground), Some(BlockStateId::AIR));
    assert_eq!(hs.server().block(above), Some(glass()));
    assert_eq!(hs.server().dirty_count(), 1);

    // Step past the save interval and wait (in real time) for the commit.
    let committed = wait_until(&mut hs, |hs| hs.server().saved_batches() >= 1, 200);
    assert!(committed, "no batch was committed");
    let unpinned = wait_until(&mut hs, |hs| hs.server().dirty_count() == 0, 100);
    assert!(unpinned, "edits never committed");

    // Drop: flush runs, then the server joins its save thread.
    drop(hs);
    drop(client);

    // A new server on the same directory...
    let mut hs = HeadlessServer::with_config(persistent_config(&dir, Duration::from_millis(150)));
    let mut client = hs.connect("bob");
    hs.step(1);
    client.drain();

    // ...streams the edited cube in from disk, edit included.
    let cube_pos = ground.split().0;
    let mut got_it = false;
    for _ in 0..100 {
        if got_it {
            break;
        }
        hs.step(1);
        for msg in client.drain() {
            if let ServerMsg::CubeData { pos, cube } = msg
                && pos == cube_pos
            {
                let local = ground.split().1;
                assert_eq!(cube.get(local), BlockStateId::AIR);
                got_it = true;
            }
        }
        got_it = got_it || hs.server().block(ground) == Some(BlockStateId::AIR);
    }
    assert!(got_it, "the edited cube never streamed back");
    assert_eq!(hs.server().block(above), Some(glass()));
}

#[test]
fn edited_cube_is_unpinned_and_evicted_after_save() {
    let dir = TempDir::new().unwrap();
    let mut hs = HeadlessServer::with_config(persistent_config(&dir, Duration::from_millis(150)));
    let mut client = hs.connect("alice");
    let ground = settle(&mut hs, &mut client);
    let edited_cube = ground.split().0;

    client.send(ClientMsg::BreakBlock { pos: ground });
    hs.step(1);
    client.drain();
    assert_eq!(hs.server().pinned_count(), 1);

    // Past the save interval the batch commits and the cube is unpinned.
    let unpinned = wait_until(&mut hs, |hs| hs.server().pinned_count() == 0, 200);
    assert!(unpinned, "the edited cube was never unpinned");
    assert!(hs.server().world_cube_count() > 1, "terrain never loaded");

    // Walk away (east): the cube leaves the keep set and — no longer
    // pinned — is evicted, unlike the 1.7 rule.
    let mut seq = 0u64;
    let cruise = hs.server().players().next().unwrap().state.position.y + 40.0;
    let left = fly_until(
        &mut hs,
        &mut client,
        std::f32::consts::FRAC_PI_2,
        cruise,
        &mut seq,
        |hs, _| hs.server().block(ground).is_none(),
        4000,
    );
    assert!(left, "the edited cube was never evicted");
    assert_eq!(hs.server().pinned_count(), 0);
    // With the player far away, every world cube sits in their keep set:
    // 13 columns (dx^2+dz^2 <= 4) x 5 vertical (|dy| <= 2) = 65 at most.
    // The spawn cubes — the edited one included — are gone.
    assert!(
        hs.server().world_cube_count() <= 65,
        "world still holds {} cubes outside any keep set",
        hs.server().world_cube_count()
    );

    // Walk back: the cube streams in from disk with the edit, not
    // regenerated without it.
    let returned = fly_until(
        &mut hs,
        &mut client,
        -std::f32::consts::FRAC_PI_2,
        cruise,
        &mut seq,
        |_hs, client| client.loaded().contains(&edited_cube),
        4000,
    );
    assert!(returned, "the edited cube never streamed back");
    assert_eq!(hs.server().block(ground), Some(BlockStateId::AIR));
}

#[test]
fn an_edit_during_a_save_stays_pinned_until_the_next_batch() {
    let dir = TempDir::new().unwrap();
    let mut hs = HeadlessServer::with_config(persistent_config(&dir, Duration::from_millis(50)));
    let mut client = hs.connect("alice");
    let ground = settle(&mut hs, &mut client);

    // Hold the saver, edit (version 1), let the interval snapshot it.
    let held = hs.server().storage().unwrap().clone();
    held.hold_saves();
    client.send(ClientMsg::BreakBlock { pos: ground });
    hs.step(1);
    client.drain();
    assert_eq!(hs.server().dirty_count(), 1);
    let mut snapshotted = false;
    for _ in 0..10 {
        hs.step(1);
        if hs.server().pending_save_count() == 1 {
            snapshotted = true;
            break;
        }
    }
    assert!(snapshotted, "the batch was never handed to the saver");
    // Give the saver a moment to take the batch, so the release below
    // deterministically unblocks an in-flight save.
    std::thread::sleep(Duration::from_millis(10));
    assert_eq!(hs.server().saved_batches(), 0);

    // Edit again (version 2) while batch 1 is held mid-save.
    client.send(ClientMsg::PlaceBlock {
        pos: ground,
        state: glass(),
    });
    hs.step(1);
    client.drain();
    assert_eq!(hs.server().block(ground), Some(glass()));

    // Release: batch 1 commits and its ack arrives — but the cube is at
    // version 2, so it stays pinned until the second batch commits.
    held.release_saves();
    let acked = wait_until(&mut hs, |hs| hs.server().saved_batches() == 1, 200);
    assert!(acked, "the held batch never committed");
    assert_eq!(
        hs.server().pinned_count(),
        1,
        "the cube must stay pinned after an edit made during the save"
    );
    assert_eq!(hs.server().block(ground), Some(glass()));

    let settled = wait_until(&mut hs, |hs| hs.server().pinned_count() == 0, 200);
    assert!(settled, "the second batch never committed");
    // And the latest edit is what is on disk. Drop the held handle too —
    // it keeps the database file open until it goes.
    drop(held);
    drop(hs);
    let storage = jbm_server::Storage::open(
        dir.path(),
        &{
            let mut registry = jbm_core::block::BlockRegistry::new();
            Blocks::register(&mut registry);
            registry
        },
        jbm_server::server::DEFAULT_SEED,
    )
    .unwrap();
    let cube = storage.load(ground.split().0).unwrap().unwrap();
    assert_eq!(cube.get(ground.split().1), glass());
}

#[test]
fn a_failed_save_keeps_cubes_pinned_and_retries() {
    let dir = TempDir::new().unwrap();
    let mut hs = HeadlessServer::with_config(persistent_config(&dir, Duration::from_millis(150)));
    let mut client = hs.connect("alice");
    let ground = settle(&mut hs, &mut client);

    client.send(ClientMsg::BreakBlock { pos: ground });
    hs.step(1);
    client.drain();
    hs.server().storage().unwrap().fail_next_save();

    // The next batch fails: the cube stays pinned and dirty, uncommitted.
    let failed = wait_until(
        &mut hs,
        |hs| hs.server().pending_save_count() == 0 && hs.server().saved_batches() == 0,
        200,
    );
    assert!(failed, "the failing batch was never taken off the queue");
    assert_eq!(hs.server().saved_batches(), 0);
    assert_eq!(hs.server().dirty_count(), 1, "a failed save must not unpin");

    // The hook is one-shot: the next interval's retry commits.
    let retried = wait_until(&mut hs, |hs| hs.server().saved_batches() == 1, 200);
    assert!(retried, "the retry never committed");
    assert_eq!(hs.server().pinned_count(), 0);
    assert_eq!(
        hs.server()
            .storage()
            .unwrap()
            .load(ground.split().0)
            .unwrap()
            .unwrap()
            .get(ground.split().1),
        BlockStateId::AIR
    );
}
