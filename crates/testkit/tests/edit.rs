//! Block-edit integration (task 1.7): two players near each other on a
//! streaming server; one places a block; within the same tick both
//! receive the `BlockChange` before that tick's `Tick`, and a player far
//! away receives nothing.

use jbm_core::block::{BlockRegistry, builtin};
use jbm_core::physics::MoveInput;
use jbm_core::protocol::{ClientMsg, ServerMsg};
use jbm_server::{DEFAULT_SEED, ServerConfig};
use jbm_testkit::HeadlessServer;

/// The streaming shape of the spec: R = 3, V = 2, inline generation.
fn config() -> ServerConfig {
    ServerConfig {
        seed: DEFAULT_SEED,
        streaming: true,
        view_radius: 3,
        vertical_radius: 2,
        send_budget_per_tick: 64,
        gen_threads: 0,
        gen_budget_per_tick: 64,
        max_in_flight: 512,
        world_dir: None,
        save_interval: std::time::Duration::from_secs(5),
        lod_levels: 4,
        lod_view_radius: 8,
        lod_vertical_radius: 3,
        lod_send_budget_per_tick: 32,
        lod_max_in_flight: 256,
    }
}

/// Cubes in the desired set for R = 3, V = 2: 29 columns of 5.
const DESIRED_CUBES: usize = 29 * 5;

#[test]
fn a_place_reaches_every_holder_within_one_tick_and_nobody_else() {
    let mut server = HeadlessServer::with_config(config());
    let mut a = server.connect("alice");
    let mut b = server.connect("bob");
    let mut far = server.connect("carol");

    // Let all three fully load their spawn area.
    server.step(1);
    for client in [&mut a, &mut b, &mut far] {
        let _ = client.drain();
    }
    for _ in 0..200 {
        server.step(1);
        for client in [&mut a, &mut b, &mut far] {
            client.drain();
        }
        if a.loaded().len() == DESIRED_CUBES
            && b.loaded().len() == DESIRED_CUBES
            && far.loaded().len() == DESIRED_CUBES
        {
            break;
        }
    }
    assert_eq!(a.loaded().len(), DESIRED_CUBES, "alice never settled");
    assert_eq!(b.loaded().len(), DESIRED_CUBES, "bob never settled");
    assert_eq!(far.loaded().len(), DESIRED_CUBES, "carol never settled");

    // Fly carol far away: well beyond the keep radius (R + 1 = 4 cubes =
    // 128 blocks), cruising high above the terrain.
    let mut left = false;
    for seq in 1..=2000_u64 {
        let state = server
            .server()
            .players()
            .find(|p| p.name == "carol")
            .unwrap()
            .state;
        let input = MoveInput {
            fly_toggle: !state.flying,
            up: (state.position.y < 150.0).into(),
            forward: (state.position.y >= 150.0) as i8,
            sprint: true,
            yaw: std::f32::consts::FRAC_PI_2,
            ..MoveInput::IDLE
        };
        far.send(ClientMsg::PlayerInput {
            seq,
            input,
            pitch: 0.0,
        });
        server.step(1);
        far.drain();
        a.drain();
        b.drain();
        if state.position.x >= 160.0 {
            left = true;
            break;
        }
    }
    assert!(left, "carol never flew far enough away");
    let spawn_cube = a.loaded().iter().copied().min().unwrap();
    assert!(
        !far.loaded().contains(&spawn_cube),
        "carol still holds a spawn cube"
    );

    // The edit: alice places stone one cell over her feet column. The
    // cell is air, within reach of her (spawn) eye, and overlaps no box.
    let feet = server
        .server()
        .players()
        .find(|p| p.name == "alice")
        .unwrap()
        .state
        .position;
    let at = jbm_core::coords::BlockPos {
        x: 1,
        y: feet.y.floor() as i64 + 2,
        z: 0,
    };
    let mut registry = BlockRegistry::new();
    let stone = builtin::Blocks::register(&mut registry).stone;
    a.send(ClientMsg::PlaceBlock {
        pos: at,
        state: stone,
    });

    // Exactly one server tick: both alice and bob see the BlockChange
    // before that tick's Tick message.
    server.step(1);
    for (name, client) in [("alice", &mut a), ("bob", &mut b)] {
        let messages = client.drain();
        let change = messages
            .iter()
            .position(|m| matches!(m, ServerMsg::BlockChange { pos, state } if *pos == at && *state == stone))
            .unwrap_or_else(|| panic!("{name} never received the BlockChange: {messages:?}"));
        let tick = messages
            .iter()
            .position(|m| matches!(m, ServerMsg::Tick { .. }))
            .unwrap_or_else(|| panic!("{name} received no Tick: {messages:?}"));
        assert!(
            change < tick,
            "{name}'s BlockChange must precede the Tick: {messages:?}"
        );
    }
    // Carol, far away, hears nothing about it.
    for msg in far.drain() {
        assert!(
            !matches!(msg, ServerMsg::BlockChange { .. }),
            "carol received {msg:?}"
        );
    }

    // And the world changed.
    assert_eq!(server.server().block(at), Some(stone));
    assert_eq!(server.server().pinned_count(), 1);
}
