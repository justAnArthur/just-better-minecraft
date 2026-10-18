//! Criterion benches for the server: the tick clock, a full `Server::tick`
//! with 8 connected clients (protocol-only, so it stays comparable across
//! stages), and one idle streaming tick with everything already sent.

use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use jbm_core::link::in_process_pair;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION};
use jbm_server::Server;
use jbm_server::server::ServerConfig;
use jbm_server::tick::TickClock;
use jbm_server::worldgen::Preset;

/// Drain every queued message from a client link (untimed helper).
trait TryRecvDrain {
    fn try_recv_drain(&mut self) -> Vec<jbm_core::protocol::ServerMsg>;
}

impl TryRecvDrain for jbm_core::link::ClientLink {
    fn try_recv_drain(&mut self) -> Vec<jbm_core::protocol::ServerMsg> {
        let mut messages = Vec::new();
        while let Ok(Some(msg)) = self.try_recv() {
            messages.push(msg);
        }
        messages
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_clock");
    group.bench_function("advance_1000", |b| {
        b.iter(|| {
            let mut clock = TickClock::new();
            for _ in 0..1000 {
                clock.advance(Duration::from_millis(16));
            }
            clock.tick_count()
        })
    });
    group.finish();

    let mut group = c.benchmark_group("server");
    group.bench_function("tick_8_clients", |b| {
        let mut server = Server::with_config(ServerConfig::protocol_only());
        let mut clients = Vec::new();
        for i in 0..8u64 {
            let (mut client, server_end) = in_process_pair();
            client
                .send(ClientMsg::Hello {
                    protocol_version: PROTOCOL_VERSION,
                    name: format!("bench-{i}"),
                })
                .unwrap();
            server.add_connection(server_end);
            clients.push(client);
        }
        // Process the hellos so all 8 clients are joined before measuring.
        server.tick();
        for client in clients.iter_mut() {
            client.try_recv_drain();
        }

        // Seqs must advance across measured ticks or the server ignores
        // the inputs as stale.
        let mut seq = 0u64;
        b.iter_batched(
            || {
                // Untimed: drain last tick's replies, then queue one
                // PlayerInput per client for the tick being measured.
                seq += 1;
                for client in clients.iter_mut() {
                    client.try_recv_drain();
                    client
                        .send(ClientMsg::PlayerInput {
                            seq,
                            input: jbm_core::physics::MoveInput::IDLE,
                            pitch: 0.0,
                        })
                        .unwrap();
                }
            },
            |_| server.tick(),
            BatchSize::PerIteration,
        );
    });

    group.bench_function("tick_streaming_idle", |b| {
        // One player fully loaded at R = 8, inline generation. Measures a
        // tick with nothing left to send: the idle-player floor. Pinned to
        // the pre-2.13 shape (task 2.13 D/E changed the defaults): R = 8
        // with the camera window everywhere (`camera_window_radius` ≥ r)
        // is the full ±V window this bench's expected count asserts, and
        // the old ring shape keeps it comparable to the task's baseline.
        let mut server = Server::with_config(ServerConfig {
            worldgen: Preset::Classic, // keeps the pre-2.8 terrain comparable
            gen_threads: 0,
            view_radius: 8,
            lod_view_radius: 8,
            lod_levels: 6,
            camera_window_radius: 8,
            ..ServerConfig::default()
        });
        let (mut client, server_end) = in_process_pair();
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "bench-idle".to_owned(),
            })
            .unwrap();
        server.add_connection(server_end);
        server.tick();
        let _ = client.try_recv_drain();

        // Load everything: with R = 8, V = 4 the desired set is 1773 cubes;
        // the inline budget of 64/tick settles in well under 60 ticks. Loop
        // until a full quiet pass produces no CubeData.
        let player_id = server.players().next().unwrap().id;
        let expected: usize = {
            let mut columns = 0usize;
            for dx in -8i64..=8 {
                for dz in -8i64..=8 {
                    if dx * dx + dz * dz <= 64 {
                        columns += 1;
                    }
                }
            }
            columns * 9 // dy in -4..=4
        };
        for _ in 0..200 {
            server.tick();
            let quiet = client
                .try_recv_drain()
                .into_iter()
                .all(|msg| !matches!(msg, jbm_core::protocol::ServerMsg::CubeData { .. }));
            if quiet && server.sent_count(player_id) == expected {
                break;
            }
        }
        assert_eq!(server.sent_count(player_id), expected);

        b.iter_batched(
            || {
                // Untimed: drain the Tick messages of the previous run.
                while client.try_recv().unwrap().is_some() {}
            },
            |_| server.tick(),
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
