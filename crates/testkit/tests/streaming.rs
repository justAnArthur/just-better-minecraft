//! Headless streaming tests: one player (and sometimes two) ends up with
//! exactly the right set of cubes, the set follows the player, memory stays
//! bounded, and threaded generation matches inline.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{BlockPos, CUBE_SIZE, CubePos};
use jbm_core::physics::{MoveInput, PlayerState};
use jbm_core::protocol::{ClientMsg, ServerMsg};
use jbm_server::terrain::Generator;
use jbm_server::worldgen::Preset;
use jbm_server::{DEFAULT_SEED, ServerConfig};
use jbm_testkit::{HeadlessServer, TestClient};

/// The spec's streaming test shape: R = 3, V = 2, budgets 16.
fn streaming_config(gen_threads: usize) -> ServerConfig {
    ServerConfig {
        worldgen: Preset::Classic, // today's terrain, as before 2.8
        seed: DEFAULT_SEED,
        streaming: true,
        view_radius: 3,
        vertical_radius: 2,
        send_budget_per_tick: 16,
        gen_threads,
        gen_budget_per_tick: 16,
        max_in_flight: 512,
        world_dir: None,
        save_interval: std::time::Duration::from_secs(5),
        lod_levels: 4,
        max_column_cubes: 0,
        camera_window_radius: 4,
        lod_view_radius: 8,
        lod_vertical_radius: 3,
        lod_send_budget_per_tick: 32,
        lod_max_in_flight: 256,
    }
}

/// A generator over the same seed the server uses, to check cube contents.
fn reference_generator() -> Generator {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    Generator::with_preset(DEFAULT_SEED, blocks, Preset::Classic)
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

fn input(seq: u64, move_input: MoveInput) -> ClientMsg {
    ClientMsg::PlayerInput {
        seq,
        input: move_input,
        pitch: 0.0,
    }
}

/// The authoritative state of the `idx`-th joined player.
fn player_at(server: &HeadlessServer, idx: usize) -> PlayerState {
    server.server().players().nth(idx).unwrap().state
}

/// East is +X: yaw pi/2 in the physics' yaw convention.
const EAST: f32 = std::f32::consts::FRAC_PI_2;

/// A flight above the terrain (which peaks at y = 92).
fn cruise_input(state: &PlayerState, min_y: f64, target_x: f64) -> MoveInput {
    if !state.flying {
        MoveInput {
            fly_toggle: true,
            ..MoveInput::IDLE
        }
    } else if state.position.y < min_y {
        MoveInput {
            up: 1,
            ..MoveInput::IDLE
        }
    } else if state.position.x < target_x {
        MoveInput {
            forward: 1,
            sprint: true,
            yaw: EAST,
            ..MoveInput::IDLE
        }
    } else {
        MoveInput::IDLE
    }
}

/// Flies the player above the terrain, then east until `x >= target_x`
/// (bounded), one input per tick. The unloaded rule stalls movement while
/// cubes stream in, so this is a closed loop on the authoritative
/// position. Panics if the target is not reached in `max_ticks`.
fn fly_east_until(
    server: &mut HeadlessServer,
    client: &mut TestClient,
    player_idx: usize,
    seq: &mut u64,
    min_y: f64,
    target_x: f64,
    max_ticks: u32,
) -> PlayerState {
    for _ in 0..max_ticks {
        let state = player_at(server, player_idx);
        if state.flying && state.position.y >= min_y && state.position.x >= target_x {
            return state;
        }
        *seq += 1;
        client.send(input(*seq, cruise_input(&state, min_y, target_x)));
        server.step(1);
        let _ = client.drain();
    }
    panic!("player never reached x >= {target_x}");
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
/// `max_ticks` and a 10 s wall clock. While generation is in flight the
/// loop sleeps 1 ms per tick, so threaded mode is bounded by time rather
/// than by how fast ticks spin (slow CI runners need far more ticks than a
/// laptop before the gen threads catch up). Returns every `CubeData`
/// received.
fn step_until_collected(
    server: &mut HeadlessServer,
    client: &mut TestClient,
    mut cond: impl FnMut(&TestClient) -> bool,
    max_ticks: u32,
) -> Vec<(CubePos, jbm_core::cube::Cube)> {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut collected = Vec::new();
    for _ in 0..max_ticks {
        assert!(
            Instant::now() < deadline,
            "streaming did not settle within 10 s"
        );
        if server.server().in_flight() > 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
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

    // Fly ten cubes east: the center now follows the authoritative
    // position, so the move has to be a real flight above the terrain.
    let mut seq = 0u64;
    let state = fly_east_until(
        &mut server,
        &mut client,
        0,
        &mut seq,
        spawn[1] + 60.0,
        f64::from(center.x * CUBE_SIZE) + 321.0,
        4000,
    );
    let new_center = cube_of([state.position.x, state.position.y, state.position.z]);
    assert_eq!(new_center.x, center.x + 10);
    let new_desired = desired(new_center);
    let new_keep = keep(new_center);
    // A gradual flight leaves the path's keep-band cubes loaded (only a
    // teleport would drop them at once), so the assertions are the
    // hysteresis contract itself: everything of the new desired set is
    // loaded, nothing outside the new keep set is.
    step_until_collected(
        &mut server,
        &mut client,
        |c| c.loaded().is_superset(&new_desired),
        200,
    );

    assert!(client.loaded().is_superset(&new_desired));
    // No cube outside the keep set is loaded...
    for pos in client.loaded() {
        assert!(
            new_keep.contains(pos),
            "loaded {pos:?} outside the keep set"
        );
    }
    // ...and the old cubes really were unloaded (they cannot silently stay).
    assert!(
        client.unload_count() > 0,
        "the start's cubes left the keep set"
    );
    // The server's world followed: everything outside the keep set is
    // evicted, so the count is bounded by the keep set, not the flight.
    assert!(server.server().world_cube_count() <= keep(new_center).len());
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
    // Fly to the cube border at x = 32 (world), then closed-loop steer
    // back and forth across it: the center cube flips every crossing.
    let border = f64::from(center.x * CUBE_SIZE) + f64::from(CUBE_SIZE);
    let mut seq = 0u64;
    let state = fly_east_until(
        &mut server,
        &mut client,
        0,
        &mut seq,
        spawn[1] + 60.0,
        border - 1.1,
        4000,
    );
    assert!(state.flying);
    let mut crossings = 0u32;
    let mut was_east = false;
    for _ in 0..2000 {
        if crossings >= 20 {
            break;
        }
        let state = player_at(&server, 0);
        let move_input = if state.position.x < border {
            MoveInput {
                forward: 1,
                sprint: true,
                yaw: EAST,
                ..MoveInput::IDLE
            }
        } else {
            MoveInput {
                forward: 1,
                sprint: true,
                yaw: -EAST,
                ..MoveInput::IDLE
            }
        };
        let is_east = state.position.x >= border;
        if was_east != is_east {
            crossings += 1;
            was_east = is_east;
        }
        seq += 1;
        client.send(input(seq, move_input));
        server.step(1);
        let _ = client.drain();
    }
    assert_eq!(crossings, 20, "the loop must cross the border repeatedly");
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

    // Eight cubes east is the nearest distance whose desired sets (R = 3)
    // cannot share a cube; bob flies there.
    let center_b = CubePos {
        x: center_a.x + 8,
        y: center_a.y,
        z: center_a.z,
    };
    let mut seq = 0u64;
    let state = fly_east_until(
        &mut server,
        &mut b,
        1,
        &mut seq,
        spawn_b[1] + 60.0,
        f64::from(center_b.x * CUBE_SIZE) + 0.5,
        4000,
    );
    let center_b = cube_of([state.position.x, state.position.y, state.position.z]);
    assert_eq!(center_b.x, center_a.x + 8);

    let expected_a = desired(center_a);
    let expected_b = desired(center_b);
    // Bob's flight path leaves keep-band cubes loaded behind him; the
    // contract is that his new desired set fully arrives.
    let settled = server.step_until(
        &mut [&mut a, &mut b],
        |_, cs| cs[0].loaded() == &expected_a && cs[1].loaded().is_superset(&expected_b),
        200,
    );
    assert!(settled, "both players did not settle");
    assert_eq!(a.loaded(), &expected_a);
    assert!(b.loaded().is_superset(&expected_b));
    assert!(
        a.loaded().is_disjoint(b.loaded()),
        "players 8 cubes apart must not share cubes"
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
    let collected = step_until_collected(
        &mut server,
        &mut client,
        |c| c.loaded() == &expected,
        100_000,
    );

    assert_eq!(client.loaded(), &expected);
    let reference = reference_generator();
    for (pos, cube) in collected {
        assert_eq!(cube, reference.generate(pos), "cube content at {pos:?}");
    }
    assert_eq!(server.server().world_cube_count(), expected.len());
    assert_eq!(server.server().in_flight(), 0);
}
