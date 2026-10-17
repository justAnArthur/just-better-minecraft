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
use jbm_server::worldgen::Preset;
use jbm_testkit::{HeadlessServer, TestClient};
use tempfile::TempDir;

/// A streaming, inline-generation server on a temp world dir with a short
/// save interval (one batch per tick).
fn persistent_config(dir: &TempDir, save_interval: Duration) -> ServerConfig {
    ServerConfig {
        worldgen: Preset::Classic, // today's terrain, as before 2.8
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
/// Releases held saves when dropped. Declared after the server, so it drops
/// first while a failed assertion unwinds: the server's flush-on-drop then
/// never waits on a save thread blocked by `hold_saves`.
struct ReleaseOnDrop(std::sync::Arc<jbm_server::Storage>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release_saves();
    }
}

/// Steps until every batch sent so far is answered. These tests send no
/// movement input, so after the join-time player record nothing else
/// changes on its own: batch counts then move only for what the test does.
fn quiesce(hs: &mut HeadlessServer) {
    let quiet = wait_until(hs, |hs| hs.server().unanswered_batches() == 0, 400);
    assert!(quiet, "the server never went quiet");
}

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
    quiesce(&mut hs);
    let base = hs.server().saved_batches();
    let held = hs.server().storage().unwrap().clone();
    held.hold_saves();
    let release = ReleaseOnDrop(held.clone());
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
    assert_eq!(hs.server().saved_batches(), base);

    // Edit again (version 2) while batch 1 is held mid-save.
    client.send(ClientMsg::PlaceBlock {
        pos: ground,
        state: glass(),
    });
    hs.step(1);
    client.drain();
    assert_eq!(hs.server().block(ground), Some(glass()));

    // Let batch 1 (and only batch 1) commit: its ack arrives, but the cube
    // is at version 2, so it stays pinned until the second batch commits.
    // The version-2 batch stays held, so the count cannot skip past 1.
    held.release_one_save();
    let acked = wait_until(&mut hs, |hs| hs.server().saved_batches() > base, 400);
    assert!(acked, "the held batch never committed");
    assert_eq!(
        hs.server().pinned_count(),
        1,
        "the cube must stay pinned after an edit made during the save"
    );
    assert_eq!(hs.server().block(ground), Some(glass()));

    assert_eq!(hs.server().saved_batches(), base + 1);

    held.release_saves();
    let settled = wait_until(&mut hs, |hs| hs.server().pinned_count() == 0, 400);
    assert!(settled, "the second batch never committed");
    // And the latest edit is what is on disk. Drop the held handle too —
    // it keeps the database file open until it goes.
    drop(release);
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

    // Hold saves so the failure and the retry happen one batch at a time.
    quiesce(&mut hs);
    let base = hs.server().saved_batches();
    let storage = hs.server().storage().unwrap().clone();
    storage.hold_saves();
    let _release = ReleaseOnDrop(storage.clone());
    storage.fail_next_save();
    client.send(ClientMsg::BreakBlock { pos: ground });
    hs.step(1);
    client.drain();
    let snapshotted = wait_until(&mut hs, |hs| hs.server().pending_save_count() == 1, 400);
    assert!(snapshotted, "the batch was never handed to the saver");

    // The next batch fails: the cube stays pinned and dirty, uncommitted.
    // Later batches stay held, so nothing can commit behind it.
    storage.release_one_save();
    // Wait on the failure count, not `pending_save_count() == 0`: the tick
    // that applies the failure may already re-snapshot the retry.
    let failed = wait_until(&mut hs, |hs| hs.server().failed_batches() == 1, 400);
    assert!(failed, "the failing batch was never reported");
    assert_eq!(hs.server().saved_batches(), base);
    assert_eq!(hs.server().dirty_count(), 1, "a failed save must not unpin");

    // The hook is one-shot: the next interval's retry commits.
    storage.release_saves();
    let retried = wait_until(&mut hs, |hs| hs.server().saved_batches() > base, 400);
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

/// The game's shape: threaded generation, R = 8 / V = 4, an edit restored
/// from disk after a restart, then flown away from until evicted and back.
#[test]
fn a_restored_edit_survives_eviction_with_threaded_generation() {
    let dir = TempDir::new().unwrap();
    let config = || ServerConfig {
        worldgen: Preset::Classic, // today's terrain, as before 2.8
        gen_threads: 4,
        view_radius: 8,
        vertical_radius: 4,
        ..persistent_config(&dir, Duration::from_millis(150))
    };
    let warm = |hs: &mut HeadlessServer, client: &mut TestClient| {
        for _ in 0..300 {
            hs.step(1);
            client.drain();
            std::thread::sleep(Duration::from_millis(2));
        }
    };
    // Session 1: edit, quit.
    let mut hs = HeadlessServer::with_config(config());
    let mut client = hs.connect("alice");
    warm(&mut hs, &mut client);
    let ground = settle(&mut hs, &mut client);
    client.send(ClientMsg::BreakBlock { pos: ground });
    hs.step(1);
    client.drain();
    drop(hs);
    drop(client);
    // Session 2: edit is back; fly far away and back.
    let mut hs = HeadlessServer::with_config(config());
    let mut client = hs.connect("alice");
    warm(&mut hs, &mut client);
    assert_eq!(
        hs.server().block(ground),
        Some(BlockStateId::AIR),
        "edit not restored"
    );
    let edited_cube = ground.split().0;
    let mut seq = 0u64;
    let cruise = hs.server().players().next().unwrap().state.position.y + 40.0;
    let left = fly_until(
        &mut hs,
        &mut client,
        std::f32::consts::FRAC_PI_2,
        cruise,
        &mut seq,
        |hs, _| {
            std::thread::sleep(Duration::from_millis(1));
            hs.server().block(ground).is_none()
        },
        20000,
    );
    assert!(left, "never evicted");
    let returned = fly_until(
        &mut hs,
        &mut client,
        -std::f32::consts::FRAC_PI_2,
        cruise,
        &mut seq,
        |hs, client| {
            std::thread::sleep(Duration::from_millis(1));
            client.loaded().contains(&edited_cube) && hs.server().block(ground).is_some()
        },
        20000,
    );
    assert!(returned, "never came back");
    assert_eq!(
        hs.server().block(ground),
        Some(BlockStateId::AIR),
        "server lost the edit"
    );
}

/// Joins `name` and returns its `Welcome`: (spawn, yaw, pitch, flying).
fn join(hs: &mut HeadlessServer, name: &str) -> (TestClient, [f64; 3], f32, f32, bool) {
    let mut client = hs.connect(name);
    for _ in 0..5 {
        hs.step(1);
        for msg in client.drain() {
            if let ServerMsg::Welcome {
                spawn,
                yaw,
                pitch,
                flying,
                ..
            } = msg
            {
                return (client, spawn, yaw, pitch, flying);
            }
        }
    }
    panic!("no Welcome for {name}");
}

/// Flies `client` east for a while, then looks up and down a little.
/// Returns the server's final record for the player.
fn wander(hs: &mut HeadlessServer, client: &mut TestClient) -> ([f64; 3], f32, f32) {
    let mut seq = 0u64;
    let cruise = hs.server().players().next().unwrap().state.position.y + 10.0;
    let mut ticks = 0;
    fly_until(
        hs,
        client,
        std::f32::consts::FRAC_PI_2,
        cruise,
        &mut seq,
        |_, _| {
            ticks += 1;
            ticks > 60
        },
        100,
    );
    seq += 1;
    client.send(ClientMsg::PlayerInput {
        seq,
        input: MoveInput {
            yaw: 0.75,
            ..MoveInput::IDLE
        },
        pitch: -0.4,
    });
    hs.step(1);
    client.drain();
    let state = hs.server().players().next().unwrap().state;
    assert!(state.flying, "the wander should end in flight");
    (state.position.to_array(), 0.75, -0.4)
}

#[test]
fn a_player_returns_where_they_left_looking_the_same_way() {
    let dir = TempDir::new().unwrap();
    // An hour-long interval: only the shutdown flush saves.
    let config = || persistent_config(&dir, Duration::from_secs(3600));
    let mut hs = HeadlessServer::with_config(config());
    let (mut client, spawn, _, _, flying) = join(&mut hs, "alice");
    assert!(!flying);
    let (position, yaw, pitch) = wander(&mut hs, &mut client);
    assert_ne!(position, spawn, "the player never moved");
    drop(hs);
    drop(client);

    let mut hs = HeadlessServer::with_config(config());
    let (_alice, back, back_yaw, back_pitch, back_flying) = join(&mut hs, "alice");
    assert_eq!(back, position);
    assert_eq!((back_yaw, back_pitch, back_flying), (yaw, pitch, true));
    let state = hs.server().players().next().unwrap().state;
    assert_eq!(state.position.to_array(), position);
    assert!(state.flying);
    // Someone new still gets the world spawn.
    let (_bob, bob_spawn, bob_yaw, _, bob_flying) = join(&mut hs, "bob");
    assert_eq!(bob_spawn, spawn);
    assert_eq!((bob_yaw, bob_flying), (0.0, false));
}

#[test]
fn a_player_who_disconnects_first_is_still_remembered() {
    let dir = TempDir::new().unwrap();
    let config = || persistent_config(&dir, Duration::from_secs(3600));
    let mut hs = HeadlessServer::with_config(config());
    let (mut client, _, _, _, _) = join(&mut hs, "alice");
    let (position, yaw, pitch) = wander(&mut hs, &mut client);
    // The client goes away first; the server notices and drops the
    // connection, then shuts down later.
    drop(client);
    hs.step(3);
    assert_eq!(hs.server().players().count(), 0);
    drop(hs);

    let mut hs = HeadlessServer::with_config(config());
    let (_alice, back, back_yaw, back_pitch, back_flying) = join(&mut hs, "alice");
    assert_eq!(back, position);
    assert_eq!((back_yaw, back_pitch, back_flying), (yaw, pitch, true));
}

/// Sends `n` eastward flight inputs without stepping the server, so they
/// sit unprocessed (in the link, then the input queue) at shutdown.
fn queue_flight(client: &mut TestClient, first_seq: u64, n: u64) {
    for seq in first_seq..first_seq + n {
        client.send(ClientMsg::PlayerInput {
            seq,
            input: MoveInput {
                forward: 1,
                sprint: true,
                yaw: std::f32::consts::FRAC_PI_2,
                ..MoveInput::IDLE
            },
            pitch: 0.1,
        });
    }
}

/// Where the player ends up when the server applies every queued input
/// normally: the reference for the quit-right-away tests.
fn reference_position(n: u64) -> [f64; 3] {
    let dir = TempDir::new().unwrap();
    let mut hs = HeadlessServer::with_config(persistent_config(&dir, Duration::from_secs(3600)));
    let (mut client, _, _, _, _) = join(&mut hs, "alice");
    wander(&mut hs, &mut client);
    queue_flight(&mut client, 1000, n);
    hs.step(10);
    hs.server()
        .players()
        .next()
        .unwrap()
        .state
        .position
        .to_array()
}

#[test]
fn quitting_right_after_moving_keeps_the_last_inputs() {
    let expected = reference_position(6);
    for client_leaves_first in [false, true] {
        let dir = TempDir::new().unwrap();
        let config = || persistent_config(&dir, Duration::from_secs(3600));
        let mut hs = HeadlessServer::with_config(config());
        let (mut client, _, _, _, _) = join(&mut hs, "alice");
        let (before, _, _) = wander(&mut hs, &mut client);
        queue_flight(&mut client, 1000, 6);
        if client_leaves_first {
            // The server notices the closed link on its next tick, after
            // reading the inputs still buffered in it.
            drop(client);
            hs.step(1);
            drop(hs);
        } else {
            // The server stops before ever ticking those inputs.
            drop(hs);
            drop(client);
        }
        let mut hs = HeadlessServer::with_config(config());
        let (_alice, back, _, pitch, _) = join(&mut hs, "alice");
        assert_ne!(back, before, "client_leaves_first = {client_leaves_first}");
        assert_eq!(
            back, expected,
            "client_leaves_first = {client_leaves_first}"
        );
        assert_eq!(pitch, 0.1);
    }
}
