//! End-to-end prediction tests: a headless server with inline generation,
//! a real `ClientWorld` and a real `Predictor`, no GPU. The client sends
//! one scripted input per tick, the server steps, both sides run the same
//! shared physics — these tests prove convergence, zero corrections on an
//! identical world, and a correction when the client's world lies.

use glam::DVec3;
use jbm_client::predict::Predictor;
use jbm_client::world::ClientWorld;
use jbm_core::block::{BlockRegistry, builtin::Blocks};
use jbm_core::coords::BlockPos;
use jbm_core::physics::{MoveInput, PlayerState};
use jbm_core::protocol::ServerMsg;
use jbm_server::terrain::Generator;
use jbm_server::worldgen::Preset;
use jbm_server::{DEFAULT_SEED, ServerConfig};
use jbm_testkit::{HeadlessServer, TestClient};

/// The streaming shape of the spec: R = 3, V = 2, inline generation.
fn config() -> ServerConfig {
    ServerConfig {
        worldgen: Preset::Classic, // pin the pre-2.8 terrain: these scenes and their goldens predate it
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
        lod_levels: 0, // this scene predates LOD rings; 2.6 keeps it LOD0-only
        lod_view_radius: 8,
        lod_vertical_radius: 3,
        lod_send_budget_per_tick: 32,
        lod_max_in_flight: 256,
        ..ServerConfig::default()
    }
}

/// Cubes in the desired set for R = 3, V = 2: 29 columns of 5.
const DESIRED_CUBES: usize = 29 * 5;

const TAU: f32 = std::f32::consts::TAU;

/// The 200-tick script: walk, jump, turn, sprint, toggle fly, fly up,
/// toggle back and fall.
fn script() -> Vec<MoveInput> {
    let mut script = Vec::with_capacity(200);
    for t in 0..200 {
        let input = match t {
            // Walk north, jumping every fifth tick.
            0..40 => MoveInput {
                forward: 1,
                jump: t % 5 == 0,
                ..MoveInput::IDLE
            },
            // Turn east and sprint.
            40..70 => MoveInput {
                forward: 1,
                sprint: true,
                yaw: TAU / 4.0,
                ..MoveInput::IDLE
            },
            // Toggle fly, rise.
            70 => MoveInput {
                fly_toggle: true,
                up: 1,
                ..MoveInput::IDLE
            },
            71..90 => MoveInput {
                up: 1,
                ..MoveInput::IDLE
            },
            // Fly diagonally at sprint.
            90..130 => MoveInput {
                forward: 1,
                strafe: 1,
                sprint: true,
                yaw: -TAU / 8.0,
                ..MoveInput::IDLE
            },
            // Toggle back and fall, then walk it off.
            130 => MoveInput {
                fly_toggle: true,
                ..MoveInput::IDLE
            },
            131..165 => MoveInput {
                forward: 1,
                jump: t % 7 == 0,
                yaw: TAU / 4.0,
                ..MoveInput::IDLE
            },
            // Strafe west, stand still at the end.
            165..195 => MoveInput {
                strafe: -1,
                ..MoveInput::IDLE
            },
            _ => MoveInput::IDLE,
        };
        script.push(input);
    }
    script
}

/// One scripted client: its link, world and predictor, plus what the loop
/// learns along the way.
struct Predicted {
    client: TestClient,
    world: ClientWorld,
    predictor: Predictor,
    last_server_state: Option<PlayerState>,
    /// Whether the prediction ever sank below the last acked server state
    /// (the client predicting a fall the server does not see).
    predicted_below_server: bool,
}

/// Joins, waits for the whole desired set to stream in (no inputs — the
/// predictor starts from the Welcome's spawn).
fn loaded_client(server: &mut HeadlessServer) -> Predicted {
    let mut client = server.connect("player");
    let mut world = ClientWorld::new();
    let mut spawn = None;
    for _ in 0..200 {
        server.step(1);
        for msg in client.drain() {
            if let ServerMsg::Welcome { spawn: s, .. } = msg {
                spawn = Some(DVec3::from(s));
            }
            world.apply(&msg);
        }
        if client.loaded().len() == DESIRED_CUBES {
            break;
        }
    }
    assert_eq!(
        client.loaded().len(),
        DESIRED_CUBES,
        "streaming did not settle"
    );
    Predicted {
        client,
        world,
        predictor: Predictor::new(spawn.expect("no Welcome")),
        last_server_state: None,
        predicted_below_server: false,
    }
}

/// Steps the prediction one tick and sends the input, without stepping
/// the server.
fn client_tick(player: &mut Predicted, input: MoveInput) {
    let Predicted {
        client,
        world,
        predictor,
        last_server_state,
        predicted_below_server,
    } = player;
    let message = predictor.tick(input, world);
    if let Some(server_state) = last_server_state
        && predictor.state().position.y < server_state.position.y
    {
        *predicted_below_server = true;
    }
    client.send(message);
}

/// Applies every queued server message: cubes into the world, state acks
/// into the predictor.
fn drain(player: &mut Predicted) {
    let messages = player.client.drain();
    for msg in messages {
        if let ServerMsg::PlayerState { ack_seq, state } = &msg {
            let Predicted {
                world, predictor, ..
            } = player;
            predictor.on_server_state(*ack_seq, *state, world);
            player.last_server_state = Some(*state);
        }
        player.world.apply(&msg);
    }
}

#[test]
fn prediction_converges_with_zero_corrections() {
    let mut server = HeadlessServer::with_config(config());
    let mut player = loaded_client(&mut server);
    let corrections_before = player.predictor.corrections();

    for input in script() {
        client_tick(&mut player, input);
        server.step(1);
        drain(&mut player);
    }

    assert_eq!(
        player.predictor.corrections(),
        corrections_before,
        "zero corrections after the world finished loading"
    );
    let server_state = player.last_server_state.expect("no PlayerState seen");
    assert_eq!(
        player.predictor.state(),
        server_state,
        "client equals the server"
    );
}

#[test]
fn prediction_converges_under_latency() {
    let mut server = HeadlessServer::with_config(config());
    let mut player = loaded_client(&mut server);
    let corrections_before = player.predictor.corrections();

    for (t, input) in script().into_iter().enumerate() {
        // The client ticks every tick; the server only every second one,
        // so its input queue fills between server ticks.
        client_tick(&mut player, input);
        if t % 2 == 1 {
            server.step(1);
            drain(&mut player);
        }
    }
    // Let the server process the tail and drain everything.
    server.step(2);
    drain(&mut player);

    assert_eq!(
        player.predictor.corrections(),
        corrections_before,
        "corrections stay zero once loaded"
    );
    let server_state = player.last_server_state.expect("no PlayerState seen");
    assert_eq!(
        player.predictor.state(),
        server_state,
        "still converges exactly"
    );
}

#[test]
fn a_client_side_hole_forces_a_correction() {
    let mut server = HeadlessServer::with_config(config());
    let mut player = loaded_client(&mut server);
    let script = script();

    for (t, input) in script.iter().enumerate() {
        // At tick 50 the script walks on the ground: gut the support
        // blocks out of the CLIENT's world only.
        if t == 50 {
            let feet = player.predictor.state().position;
            let layer = feet.y.floor() as i64 - 1;
            for dz in -2..=2 {
                for dx in -2..=2 {
                    player.world.remove_block(BlockPos {
                        x: feet.x.floor() as i64 + dx,
                        y: layer,
                        z: feet.z.floor() as i64 + dz,
                    });
                }
            }
        }
        client_tick(&mut player, *input);
        server.step(1);
        drain(&mut player);
    }

    assert!(
        player.predictor.corrections() > 0,
        "the client's missing floor must cause corrections"
    );
    assert!(
        player.predicted_below_server,
        "the client predicted a fall at some point"
    );
    let server_state = player.last_server_state.expect("no PlayerState seen");
    assert_eq!(
        player.predictor.state(),
        server_state,
        "final state equals the server's"
    );
}

#[test]
fn nobody_falls_through_the_floor() {
    let mut server = HeadlessServer::with_config(config());
    let mut client = server.connect("player");
    server.step(1);
    let _ = client.drain(); // Welcome

    // Idle inputs: gravity does the work, the server stays authoritative.
    for seq in 1..=100u64 {
        client.send(jbm_core::protocol::ClientMsg::PlayerInput {
            seq,
            input: MoveInput::IDLE,
            pitch: 0.0,
        });
        server.step(1);
        let _ = client.drain();
    }

    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let surface =
        Generator::with_preset(DEFAULT_SEED, blocks, Preset::Classic).surface_height(0, 0);
    let state = server.server().players().next().unwrap().state;
    assert_eq!(
        state.position,
        DVec3::new(0.5, (surface + 1) as f64, 0.5),
        "feet exactly on the surface"
    );
    assert!(state.on_ground, "landed, not hovering or sunk");
}
