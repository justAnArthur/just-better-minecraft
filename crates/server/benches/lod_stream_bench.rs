//! Criterion benches for LOD ring streaming (task 2.6), inline mode:
//!
//! - `lod/stream_settled` — one streaming tick of a player whose rings are
//!   complete and who stands still: the idle floor, expected near zero
//!   work beyond the per-level center checks.
//! - `lod/stream_moving` — a player teleported one level-1 cube (64
//!   blocks) every iteration, alternating between two adjacent level-1
//!   cubes: every sample takes the level-1 center change, the walk
//!   cursor reset and the keep-set unloads. Real sprint-fly speed
//!   (~1.1 blocks/tick) crosses a 64-block boundary only about every 58
//!   ticks, so driving this with inputs would measure mostly idle ticks.

use criterion::{Criterion, criterion_group, criterion_main};
use glam::DVec3;
use jbm_core::coords::{BlockPos, CUBE_SIZE};
use jbm_core::link::in_process_pair;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION};
use jbm_server::server::{Server, ServerConfig};
use jbm_server::worldgen::Preset;

/// Side length of a level-1 cube in blocks: two LOD0 cubes.
const LEVEL1_BLOCKS: f64 = CUBE_SIZE as f64 * 2.0;

/// The level-1 cube containing `pos`, per axis (must match
/// `streaming::level_center`: `lod0.div_euclid(2)`).
fn level1_of(pos: DVec3) -> [i32; 3] {
    let lod0 = BlockPos {
        x: pos.x.floor() as i64,
        y: pos.y.floor() as i64,
        z: pos.z.floor() as i64,
    }
    .split()
    .0;
    [
        lod0.x.div_euclid(2),
        lod0.y.div_euclid(2),
        lod0.z.div_euclid(2),
    ]
}

fn lod_config() -> ServerConfig {
    ServerConfig {
        worldgen: Preset::Classic, // keeps the pre-2.8 terrain comparable
        streaming: true,
        view_radius: 8,
        vertical_radius: 4,
        send_budget_per_tick: 64,
        gen_threads: 0,
        gen_budget_per_tick: 512,
        max_in_flight: 4096,
        lod_levels: 4,
        lod_view_radius: 8,
        lod_vertical_radius: 3,
        lod_send_budget_per_tick: 4096,
        lod_max_in_flight: 4096,
        ..ServerConfig::default()
    }
}

/// Joins one player and ticks until the rings stop delivering.
fn settle(server: &mut Server, client: &mut jbm_core::link::ClientLink) {
    // On this branch the pyramid only fills around visited LOD0 terrain
    // (worldgen fill is 2.7), so the far rings stay pending forever;
    // settled here means no cube traffic for a stretch of ticks.
    let mut quiet_rounds = 0u32;
    for _ in 0..400 {
        server.tick();
        let mut quiet = true;
        while let Ok(Some(msg)) = client.try_recv() {
            quiet = quiet
                && !matches!(
                    msg,
                    jbm_core::protocol::ServerMsg::LodCubeData { .. }
                        | jbm_core::protocol::ServerMsg::CubeData { .. }
                );
        }
        if quiet {
            quiet_rounds += 1;
            if quiet_rounds >= 20 {
                return;
            }
        } else {
            quiet_rounds = 0;
        }
    }
    panic!("the rings never settled");
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("lod");

    group.bench_function("stream_settled", |b| {
        let mut server = Server::with_config(lod_config());
        let (mut client, server_end) = in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "bench".to_owned(),
            })
            .unwrap();
        settle(&mut server, &mut client);
        b.iter(|| {
            server.tick();
            while let Ok(Some(_)) = client.try_recv() {}
        });
    });

    group.bench_function("stream_moving", |b| {
        let mut server = Server::with_config(lod_config());
        let (mut client, server_end) = in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "bench".to_owned(),
            })
            .unwrap();
        settle(&mut server, &mut client);
        // Alternate between the spawn and a spot one level-1 cube east.
        // The teleport itself is O(1) bookkeeping, deliberately inside
        // the timed closure: the measured work is the tick that reacts
        // to it. No inputs are sent, so physics never steps and the
        // state sits exactly where each teleport puts it.
        let id = server.players().next().unwrap().id;
        let home = server.players().next().unwrap().state.position;
        let away = DVec3::new(home.x + LEVEL1_BLOCKS, home.y, home.z);
        // The sequence is period-2 over exactly these two positions, so
        // one comparison covers every consecutive pair.
        assert_ne!(
            level1_of(home),
            level1_of(away),
            "adjacent iterations must sit in different level-1 cubes"
        );
        let mut at_away = false;
        b.iter(|| {
            at_away = !at_away;
            server.teleport_player(id, if at_away { away } else { home });
            server.tick();
            while let Ok(Some(_)) = client.try_recv() {}
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
