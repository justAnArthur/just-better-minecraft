//! Criterion benches for the server: the tick clock, and a full
//! `Server::tick` with 8 connected clients.

use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use jbm_core::link::in_process_pair;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION};
use jbm_server::Server;
use jbm_server::tick::TickClock;

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
        let mut server = Server::new();
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

        b.iter_batched(
            || {
                // Untimed: drain last tick's replies, then queue one
                // PlayerInput per client for the tick being measured.
                for (i, client) in clients.iter_mut().enumerate() {
                    while client.try_recv().unwrap().is_some() {}
                    client
                        .send(ClientMsg::PlayerInput {
                            client_tick: i as u64,
                            position: [i as f64, 64.0, 0.0],
                            yaw: 0.0,
                            pitch: 0.0,
                        })
                        .unwrap();
                }
            },
            |_| server.tick(),
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
