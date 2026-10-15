//! Criterion benches for LOD ring streaming (task 2.6), inline mode:
//!
//! - `lod/stream_settled` — one streaming tick of a player whose rings are
//!   complete and who stands still: the idle floor, expected near zero
//!   work beyond the per-level center checks.
//! - `lod/stream_moving` — a player crossing a level-1 boundary every
//!   tick: the center moves, the walk cursor resets and the keep-set
//!   unload scan runs.

use criterion::{Criterion, criterion_group, criterion_main};
use jbm_core::link::in_process_pair;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION};
use jbm_server::server::{Server, ServerConfig};

fn lod_config() -> ServerConfig {
    ServerConfig {
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

fn fly_input(yaw: f32, up: bool, fly_toggle: bool) -> ClientMsg {
    ClientMsg::PlayerInput {
        seq: 0,
        input: jbm_core::physics::MoveInput {
            fly_toggle,
            up: up.into(),
            forward: (!up).into(),
            sprint: true,
            yaw,
            ..jbm_core::physics::MoveInput::IDLE
        },
        pitch: 0.0,
    }
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
        // Sprint-fly east, rising first so terrain never blocks: the
        // level-1 center crosses a boundary every second cube, well
        // within the loop.
        let mut seq = 0u64;
        let east = std::f32::consts::FRAC_PI_2;
        b.iter(|| {
            let flying = server.players().next().unwrap().state.flying;
            let rising = server.players().next().unwrap().state.position.y < 200.0;
            let mut msg = fly_input(east, rising, !flying);
            seq += 1;
            if let ClientMsg::PlayerInput {
                seq: s,
                input: _,
                pitch: _,
            } = &mut msg
            {
                *s = seq;
            }
            client.send(msg).unwrap();
            server.tick();
            while let Ok(Some(_)) = client.try_recv() {}
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
