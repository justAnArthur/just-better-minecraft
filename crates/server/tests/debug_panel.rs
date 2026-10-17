//! Task 2.12's debug-panel bridge on the server side: the running
//! server's stats publication and the live view-radius change, through
//! the real threaded `run::spawn` loop the game uses.

use std::time::{Duration, Instant};

use jbm_core::link::{ClientLink, in_process_pair};
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
use jbm_server::run;
use jbm_server::server::{Server, ServerConfig};

/// Every wait is bounded by this.
const TIMEOUT: Duration = Duration::from_secs(20);

/// The deterministic small config these tests run: inline generation,
/// generous budgets so settling is about the radius change, not the
/// pipeline.
fn config(view_radius: u32) -> ServerConfig {
    ServerConfig {
        world_dir: None,
        streaming: true,
        view_radius,
        vertical_radius: 1,
        send_budget_per_tick: 128,
        gen_threads: 0,
        gen_budget_per_tick: 256,
        max_in_flight: 2048,
        lod_levels: 1,
        lod_view_radius: 2,
        lod_vertical_radius: 1,
        lod_send_budget_per_tick: 128,
        lod_max_in_flight: 512,
        ..ServerConfig::default()
    }
}

/// Sends a Hello and returns once the Welcome arrived.
fn join(client: &mut ClientLink) {
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "panel".to_owned(),
        })
        .unwrap();
    let deadline = Instant::now() + TIMEOUT;
    loop {
        assert!(Instant::now() < deadline, "timed out waiting for Welcome");
        match client.try_recv().unwrap() {
            Some(ServerMsg::Welcome { .. }) => return,
            Some(_) => {}
            None => std::thread::sleep(Duration::from_millis(2)),
        }
    }
}

/// Drains everything the client link holds.
fn drain(client: &mut ClientLink) -> Vec<ServerMsg> {
    let mut messages = Vec::new();
    while let Some(msg) = client.try_recv().unwrap() {
        messages.push(msg);
    }
    messages
}

/// One player's `(lod0_pending, sent_count)` from the latest published
/// snapshot, polling until a snapshot with a player exists.
fn player_counters(handle: &run::ServerHandle) -> (usize, usize) {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        assert!(
            Instant::now() < deadline,
            "no stats snapshot with a player was ever published"
        );
        if let Some(stats) = handle.stats()
            && let Some(player) = stats.per_player.first()
        {
            return (player.lod0_pending, player.sent_count);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Waits until the counters satisfy `keep` (over `(lod0_pending,
/// sent_count)` — snapshots publish every 0.5 s, so a just-applied
/// change must be waited past, not assumed) and returns the sent count.
fn wait_counters(
    handle: &run::ServerHandle,
    client: &mut ClientLink,
    keep: impl Fn(usize, usize) -> bool,
) -> usize {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        assert!(Instant::now() < deadline, "counters never met the wait");
        drain(client);
        let (pending, sent) = player_counters(handle);
        if keep(pending, sent) {
            return sent;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The running server publishes stats twice a second, and the tick in
/// them advances.
#[test]
fn stats_publish_within_a_second_and_tick_advances() {
    let handle = run::spawn(Server::with_config(config(2)));
    let first = {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            assert!(Instant::now() < deadline, "no snapshot within 2 s");
            if let Some(stats) = handle.stats() {
                break stats.tick;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    assert!(first >= 1, "a snapshot carries at least one tick");
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        assert!(Instant::now() < deadline, "the tick never advanced");
        if handle.stats().is_some_and(|stats| stats.tick > first) {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    handle.stop();
}

/// Growing the view radius 4 -> 8 streams more LOD0 cubes; shrinking
/// 8 -> 4 unloads what left the narrower keep set. The client link
/// observes the unloads.
#[test]
fn set_view_radius_grows_then_shrinks_the_streamed_view() {
    let handle = run::spawn(Server::with_config(config(4)));
    let (mut client, server_end) = in_process_pair();
    handle.connect(server_end);
    join(&mut client);
    drain(&mut client);

    let sent_at_4 = wait_counters(&handle, &mut client, |pending, sent| {
        pending == 0 && sent > 0
    });

    handle.set_view_radius(8, 2);
    let sent_at_8 = wait_counters(&handle, &mut client, |pending, sent| {
        pending == 0 && sent > sent_at_4
    });
    assert!(
        sent_at_8 > sent_at_4,
        "radius 8 streamed {sent_at_8}, not more than radius 4's {sent_at_4}"
    );

    handle.set_view_radius(4, 2);
    let mut saw_unload = false;
    let deadline = Instant::now() + TIMEOUT;
    let sent_back_at_4 = loop {
        assert!(Instant::now() < deadline, "shrink never settled");
        for msg in drain(&mut client) {
            saw_unload |= matches!(msg, ServerMsg::UnloadCube { .. });
        }
        let (pending, sent) = player_counters(&handle);
        if pending == 0 && sent > 0 && sent < sent_at_8 {
            break sent;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(saw_unload, "shrinking unloaded nothing on the client link");
    assert!(
        sent_back_at_4 < sent_at_8,
        "shrunk view holds {sent_back_at_4}, not fewer than {sent_at_8}"
    );
    handle.stop();
}

/// The clamps: LOD0 lands in 2..=32, the ring radius in 2..=16 —
/// direct on the server, no thread.
#[test]
fn set_view_radius_clamps_both_radii() {
    let mut server = Server::with_config(config(8));
    server.set_view_radius(0, 0);
    assert_eq!(server.config_view_radius(), (2, 2));
    server.set_view_radius(u32::MAX, u32::MAX);
    assert_eq!(server.config_view_radius(), (32, 16));
    server.set_view_radius(8, 8);
    assert_eq!(server.config_view_radius(), (8, 8));
}
