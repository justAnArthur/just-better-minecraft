//! Headless streaming tests: one player (and sometimes two) ends up with
//! exactly the right set of cubes, the set follows the player, memory stays
//! bounded, and threaded generation matches inline.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{BlockPos, CubePos};
use jbm_core::protocol::{ClientMsg, ServerMsg};
use jbm_server::terrain::Generator;
use jbm_server::{DEFAULT_SEED, ServerConfig};
use jbm_testkit::{HeadlessServer, TestClient};

/// The spec's streaming test shape: R = 3, V = 2, budgets 16.
fn streaming_config(gen_threads: usize) -> ServerConfig {
    ServerConfig {
        seed: DEFAULT_SEED,
        streaming: true,
        view_radius: 3,
        vertical_radius: 2,
        send_budget_per_tick: 16,
        gen_threads,
        gen_budget_per_tick: 16,
        max_in_flight: 512,
    }
}

/// A generator over the same seed the server uses, to check cube contents.
fn reference_generator() -> Generator {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    Generator::new(DEFAULT_SEED, blocks)
}

/// The cube containing a world position in block coordinates.
fn cube_of(position: [f64; 3]) -> CubePos {
    BlockPos {
        x: position[0].floor() as i64,
        y: position[1].floor() as i64,
        z: position[2].floor() as i64,
    }
    .split()
    .0
}

/// A position centered on a cube's home block, at height `y`.
fn position_of(cube: CubePos, y: f64) -> [f64; 3] {
    [
        f64::from(cube.x * 32) + 0.5,
        y,
        f64::from(cube.z * 32) + 0.5,
    ]
}

/// The set of cubes within `(r, v)` of `center`, derived here from the spec
/// shape (`dx² + dz² ≤ r²`, `|dy| ≤ v`) rather than the server's helpers.
fn shape(center: CubePos, r: i64, v: i64) -> HashSet<CubePos> {
    let mut set = HashSet::new();
    for dx in -r..=r {
        for dz in -r..=r {
            if dx * dx + dz * dz <= r * r {
                for dy in -v..=v {
                    set.insert(CubePos {
                        x: center.x + dx as i32,
                        y: center.y + dy as i32,
                        z: center.z + dz as i32,
                    });
                }
            }
        }
    }
    set
}

/// The desired set around `center` for R = 3, V = 2.
fn desired(center: CubePos) -> HashSet<CubePos> {
    shape(center, 3, 2)
}

/// The keep set around `center`: same shape with R + 1, V + 1.
fn keep(center: CubePos) -> HashSet<CubePos> {
    shape(center, 4, 3)
}

fn input(client_tick: u64, position: [f64; 3]) -> ClientMsg {
    ClientMsg::PlayerInput {
        client_tick,
        position,
        yaw: 0.0,
        pitch: 0.0,
    }
}

/// The spawn from the Welcome that just arrived in the client's queue.
fn take_welcome(client: &mut TestClient) -> [f64; 3] {
    for msg in client.drain() {
        if let ServerMsg::Welcome { spawn, .. } = msg {
            return spawn;
        }
    }
    panic!("no Welcome in the client's queue");
}

/// Step (draining the client each tick) until `cond` holds, bounded by
/// `max_ticks` and a 2 s wall clock. Returns every `CubeData` received.
fn step_until_collected(
    server: &mut HeadlessServer,
    client: &mut TestClient,
    mut cond: impl FnMut(&TestClient) -> bool,
    max_ticks: u32,
) -> Vec<(CubePos, jbm_core::cube::Cube)> {
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut collected = Vec::new();
    for _ in 0..max_ticks {
        assert!(
            Instant::now() < deadline,
            "streaming did not settle within 2 s"
        );
        server.step(1);
        for msg in client.drain() {
            if let ServerMsg::CubeData { pos, cube } = msg {
                collected.push((pos, *cube));
            }
        }
        if cond(client) {
            return collected;
        }
    }
    panic!("streaming did not settle within {max_ticks} ticks");
}

#[test]
fn join_loads_exactly_the_desired_set_nearest_first() {
    let mut server = HeadlessServer::with_config(streaming_config(0));
    let mut client = server.connect("alice");
    server.step(1);
    let spawn = take_welcome(&mut client);
    let center = cube_of(spawn);
    assert_eq!((spawn[0], spawn[2]), (0.5, 0.5));
    let expected = desired(center);

    let collected =
        step_until_collected(&mut server, &mut client, |c| c.loaded() == &expected, 200);
    // The first cube received is the player's own center cube.
    assert_eq!(collected.first().expect("at least one cube").0, center);

    // Exactly the desired set, every cube matching the generator.
    assert_eq!(client.loaded(), &expected);
    assert_eq!(collected.len(), expected.len());
    let reference = reference_generator();
    for (pos, cube) in collected {
        assert_eq!(cube, reference.generate(pos), "cube content at {pos:?}");
    }
    // The server kept exactly those cubes and nothing is still in flight.
    assert_eq!(server.server().world_cube_count(), expected.len());
    assert_eq!(server.server().in_flight(), 0);
    assert_eq!(
        server.server().sent_count(client.player_id().unwrap()),
        expected.len()
    );
}

#[test]
fn moving_ten_cubes_reloads_the_new_set_and_unloads_the_rest() {
    let mut server = HeadlessServer::with_config(streaming_config(0));
    let mut client = server.connect("alice");
    server.step(1);
    let spawn = take_welcome(&mut client);
    let center = cube_of(spawn);
    let expected = desired(center);
    step_until_collected(&mut server, &mut client, |c| c.loaded() == &expected, 200);

    let new_center = CubePos {
        x: center.x + 10,
        y: center.y,
        z: center.z,
    };
    client.send(input(1, position_of(new_center, spawn[1])));
    let new_desired = desired(new_center);
    let new_keep = keep(new_center);
    step_until_collected(
        &mut server,
        &mut client,
        |c| c.loaded() == &new_desired,
        200,
    );

    assert_eq!(client.loaded(), &new_desired);
    // No cube outside the keep set is loaded...
    for pos in client.loaded() {
        assert!(
            new_keep.contains(pos),
            "loaded {pos:?} outside the keep set"
        );
    }
    // ...and the old cubes really were unloaded (they cannot silently stay).
    assert!(client.unload_count() > 0);
    // The server's world followed: old cubes evicted, new ones in.
    assert_eq!(server.server().world_cube_count(), new_desired.len());
}

#[test]
fn oscillating_on_a_border_never_unloads() {
    let mut server = HeadlessServer::with_config(streaming_config(0));
    let mut client = server.connect("alice");
    server.step(1);
    let spawn = take_welcome(&mut client);
    let center = cube_of(spawn);
    let expected = desired(center);
    step_until_collected(&mut server, &mut client, |c| c.loaded() == &expected, 200);

    let neighbor = CubePos {
        x: center.x + 1,
        y: center.y,
        z: center.z,
    };
    for i in 0..20u64 {
        let target = if i % 2 == 0 { neighbor } else { center };
        client.send(input(i, position_of(target, spawn[1])));
        server.step(1);
        let _ = client.drain();
    }
    assert_eq!(
        client.unload_count(),
        0,
        "hysteresis must stop border cubes from flapping"
    );
    // Whatever is loaded stays inside one of the two keep sets.
    let keep_union: HashSet<_> = keep(center).union(&keep(neighbor)).copied().collect();
    for pos in client.loaded() {
        assert!(
            keep_union.contains(pos),
            "loaded {pos:?} outside both keep sets"
        );
    }
}

#[test]
fn disconnect_evicts_the_whole_world() {
    let mut server = HeadlessServer::with_config(streaming_config(0));
    let mut client = server.connect("alice");
    server.step(1);
    let spawn = take_welcome(&mut client);
    let center = cube_of(spawn);
    let expected = desired(center);
    step_until_collected(&mut server, &mut client, |c| c.loaded() == &expected, 200);
    assert_eq!(server.server().world_cube_count(), expected.len());

    drop(client);
    server.step(2);
    assert_eq!(server.server().world_cube_count(), 0);
}

#[test]
fn two_players_far_apart_each_get_their_own_cubes() {
    let mut server = HeadlessServer::with_config(streaming_config(0));
    let mut a = server.connect("alice");
    let mut b = server.connect("bob");
    server.step(1);
    let spawn_a = take_welcome(&mut a);
    let spawn_b = take_welcome(&mut b);
    assert_eq!(spawn_a, spawn_b, "both spawn over column (0, 0)");
    let center_a = cube_of(spawn_a);
    let center_b = CubePos {
        x: center_a.x + 1000,
        y: center_a.y,
        z: center_a.z,
    };
    b.send(input(1, position_of(center_b, spawn_b[1])));

    let expected_a = desired(center_a);
    let expected_b = desired(center_b);
    let settled = server.step_until(
        &mut [&mut a, &mut b],
        |_, cs| cs[0].loaded() == &expected_a && cs[1].loaded() == &expected_b,
        200,
    );
    assert!(settled, "both players did not settle");
    assert_eq!(a.loaded(), &expected_a);
    assert_eq!(b.loaded(), &expected_b);
    assert!(
        a.loaded().is_disjoint(b.loaded()),
        "players 1000 cubes apart must not share cubes"
    );
}

#[test]
fn threaded_generation_matches_inline() {
    let mut server = HeadlessServer::with_config(streaming_config(2));
    let mut client = server.connect("alice");
    server.step(1);
    let spawn = take_welcome(&mut client);
    let center = cube_of(spawn);
    let expected = desired(center);
    let collected =
        step_until_collected(&mut server, &mut client, |c| c.loaded() == &expected, 2000);

    assert_eq!(client.loaded(), &expected);
    let reference = reference_generator();
    for (pos, cube) in collected {
        assert_eq!(cube, reference.generate(pos), "cube content at {pos:?}");
    }
    assert_eq!(server.server().world_cube_count(), expected.len());
    assert_eq!(server.server().in_flight(), 0);
}
