//! The real-server LOD fly test (task 2.6; task 2.10 added the low
//! valley leg and the lookdown stop). Spawn, fly 1.5 km straight and
//! then 90 degrees for 300 m more, 40 blocks above the terrain at
//! pitch -15 degrees, then 600 m low along a valley (20 over the
//! terrain, the cliffs towering over the camera), and finally stop
//! looking straight down from 400 blocks — through the real client
//! world, mesh pool and renderer against a threaded real server. On the 2.8 overworld the
//! altitude follows the terrain: the target is the highest surface in a
//! 512-block look-ahead plus 40, so the flight rides over the ridges
//! (a fixed spawn-relative altitude would wedge into the first range).
//! Every 100 blocks it waits
//! (bounded) for the client mesh queue — no dirty cubes, no in-flight
//! mesh jobs — to be clear, polled about once per 50 ms server tick;
//! server ring readiness is never checked (whatever arrived and meshed
//! by then is what the check sees). It then checks the frame below the
//! horizon margin. It also renders one frame *without* waiting at every
//! stop and checks the middle half of it, which at these distances
//! covers well inside 512 blocks of the camera. Screenshots land under
//! `target/fly/` for the orchestrator to inspect.
//!
//! The hole check tolerates pinholes: the LOD0 greedy mesher leaves
//! T-junctions whose edge interpolation differs by ulps, so isolated
//! one-to-two-pixel cracks can show the sky through terrain (see the
//! sub-pixel expansion comment in
//! `crates/client/src/render/shaders/terrain.wgsl` — LOD0 skips the
//! expansion to keep the pre-LOD goldens bit-stable). Sentinel pixels
//! must therefore form 4-connected components of at most 2 pixels, at
//! most 8 such pixels per frame; anything bigger — a ring-border crack
//! is a long strip — fails. Eliminating the T-junctions in the mesher
//! is a follow-up. The detector lives in `jbm_client::testing::holes`
//! (shared with the rings-scene test) and is unit-tested without a GPU.
#![cfg(not(feature = "cube16"))]

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::gpu::Gpu;
use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::render::renderer::{HORIZON_COLOR, ZENITH_COLOR};
use jbm_client::terrain_sync::TerrainSync;
use jbm_client::testing::OffscreenRig;
use jbm_client::testing::holes::{assert_only_pinholes, describe_holes, sentinel_components};
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::link::ClientLink;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
use jbm_server::DEFAULT_SEED;
use jbm_server::run::{self, ServerHandle};
use jbm_server::server::{Server, ServerConfig};
use jbm_server::terrain::Generator;
use jbm_server::worldgen::Preset;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const SENTINEL: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
/// The whole test must finish within this; a hang fails instead of
/// hanging the suite.
const TEST_TIMEOUT: Duration = Duration::from_secs(600);
/// Per-stop wait for the client mesh queue to clear.
const STOP_TIMEOUT: Duration = Duration::from_secs(20);

/// One leg of the fly course: yaw, distance, and whether it flies high
/// (ridge-riding, 40 over the tallest surface ahead) or low (task 2.10:
/// 20 over the terrain with a short look-ahead, so the camera threads
/// valleys with cliff walls around it — the surface-following-window
/// view).
#[derive(Clone, Copy)]
struct Leg {
    yaw: f64,
    distance: f64,
    low: bool,
}

/// The fly course: 1500 blocks at yaw 0 (toward -Z), then 300 at yaw
/// +90 degrees (toward +X), then 600 more low along whatever valley the
/// turn lands in — pitch -15 degrees throughout.
const COURSE: [Leg; 3] = [
    Leg {
        yaw: 0.0,
        distance: 1500.0,
        low: false,
    },
    Leg {
        yaw: std::f64::consts::FRAC_PI_2,
        distance: 300.0,
        low: false,
    },
    Leg {
        yaw: std::f64::consts::FRAC_PI_2,
        distance: 600.0,
        low: true,
    },
];

/// How far ahead (blocks) the terrain-following altitude target looks:
/// the flyer climbs at half its cruise speed, so a window of 512 blocks
/// buys ~250 blocks of climb before a ridge arrives — the overworld's
/// whole relief. Terrain is a heightmap (no overhangs, task 2.8), so a
/// full-`up` climb always escapes a wall even if it starts late.
const LOOKAHEAD: i64 = 512;

/// The low leg's shorter window: threading a valley needs the floor's
/// height, not the ridges' — the shorter target keeps the camera down
/// where the cliffs tower over it.
const LOW_LOOKAHEAD: i64 = 256;

/// The cruise altitude target under `(x, z)` looking along `yaw`: the
/// highest surface ahead (task 2.10: the highest within the low leg's
/// shorter window) plus the leg's clearance. The flight rides over the
/// ridges instead of wedging into them the way a fixed spawn-relative
/// altitude would on the 2.8 terrain.
fn cruise_target(generator: &Generator, x: f64, z: f64, leg: Leg) -> f64 {
    let (fx, fz) = (leg.yaw.sin(), -leg.yaw.cos());
    let (lookahead, clearance) = if leg.low {
        (LOW_LOOKAHEAD, 20.0)
    } else {
        (LOOKAHEAD, 40.0)
    };
    let mut highest = i64::MIN;
    for d in (0..=lookahead).step_by(32) {
        let (sx, sz) = (x + fx * d as f64, z + fz * d as f64);
        highest = highest.max(generator.surface_height(sx.round() as i64, sz.round() as i64));
    }
    (highest as f64) + clearance
}

/// The client half: the terrain pipeline and the offscreen rig, plus
/// the protocol state the test drives (authoritative position, flying
/// flag, ack seq).
struct FlyClient {
    sync: TerrainSync,
    /// The authoritative player position, from the last `PlayerState`.
    server_pos: Option<DVec3>,
    /// The authoritative flying flag, and the last acknowledged input
    /// seq (`PlayerState::ack_seq`).
    server_flying: bool,
    server_ack: u64,
    rig: OffscreenRig,
}

impl FlyClient {
    fn new(gpu: &Gpu) -> FlyClient {
        let mut registry = jbm_core::block::BlockRegistry::new();
        jbm_core::block::builtin::register(&mut registry);
        let appearance = Arc::new(BlockAppearance::new(&registry));
        FlyClient {
            sync: TerrainSync::new(appearance, 0),
            server_pos: None,
            server_flying: false,
            server_ack: 0,
            rig: OffscreenRig::new(gpu, WIDTH, HEIGHT),
        }
    }

    /// Whether to send a fly toggle with the next input. The physics
    /// step freezes the player — returning the pre-toggle state — while
    /// the spawn cube is still generating, so a one-shot toggle sent at
    /// join time is consumed and silently lost. Retrying is safe with
    /// at most one unacked toggle in flight: a new one goes out only
    /// after the previous was consumed (`ack_seq` caught up) and the
    /// authoritative state still reports walking.
    fn needs_fly_toggle(&self, toggle_seq: u64) -> bool {
        !self.server_flying && self.server_ack >= toggle_seq
    }

    /// Drains the link into the terrain pipeline, capturing the
    /// `Welcome`'s spawn. Returns a fatal reason when the link dies.
    fn pump_link(
        &mut self,
        gpu: &Gpu,
        client: &mut ClientLink,
        spawn: &mut Option<[f64; 3]>,
    ) -> Option<String> {
        loop {
            match client.try_recv() {
                Ok(Some(msg)) => match &msg {
                    ServerMsg::Rejected { reason } | ServerMsg::Disconnected { reason } => {
                        return Some(reason.clone());
                    }
                    ServerMsg::Welcome { spawn: s, .. } => {
                        *spawn = Some(*s);
                    }
                    ServerMsg::PlayerState { ack_seq, state } => {
                        self.server_pos = Some(state.position);
                        self.server_flying = state.flying;
                        self.server_ack = *ack_seq;
                    }
                    ServerMsg::CubeData { .. }
                    | ServerMsg::UnloadCube { .. }
                    | ServerMsg::LodCubeData { .. }
                    | ServerMsg::UnloadLodCube { .. }
                    | ServerMsg::BlockChange { .. } => {
                        self.sync.apply(gpu, &mut self.rig.terrain, &msg);
                    }
                    _ => {}
                },
                Ok(None) => return None,
                Err(err) => return Some(err.to_string()),
            }
        }
    }

    /// Submits and uploads meshes within the app budgets (16 submits,
    /// 32 uploads per frame).
    fn pump_meshing(&mut self, gpu: &Gpu, camera: &Camera) {
        self.sync.pump(gpu, &mut self.rig.terrain, camera, 16, 32);
    }

    /// One 50 ms server tick's worth of client frames: meshing pumped at
    /// about 60 fps, as the app does (one submit/upload budget per frame).
    fn frames_for_one_tick(&mut self, gpu: &Gpu, camera: &Camera) {
        for _ in 0..3 {
            self.pump_meshing(gpu, camera);
            std::thread::sleep(Duration::from_millis(16));
        }
    }

    fn queue_clear(&self) -> bool {
        self.sync.queue_clear()
    }

    /// Renders one frame with the sentinel sky and returns it.
    fn render(&mut self, gpu: &Gpu, camera: &Camera) -> image::RgbaImage {
        self.rig
            .render_with_sky(gpu, camera, Some((HORIZON_COLOR, ZENITH_COLOR, SENTINEL)))
    }
}

/// The row margin per camera: the sky's ~15 degree below-horizon fade
/// (pure ground color only below `asin(0.25)`) plus the rings' far-edge
/// drop angle, plus slack for the outer cubes' missing far walls. Rows
/// clamp into the frame first, so a steep-down camera (task 2.10's
/// lookdown stop, where every row is below the horizon) checks the whole
/// frame instead of underflowing its region.
fn margin_rows(camera: &Camera) -> u32 {
    let row_of = |angle: f64| -> f64 {
        let below_axis = angle + f64::from(camera.pitch);
        let ndc_y = -below_axis.tan() / (f64::from(camera.fov_y) / 2.0).tan();
        (0.5 - ndc_y * 0.5) * f64::from(HEIGHT)
    };
    let clamp_row = |row: f64| row.clamp(0.0, f64::from(HEIGHT));
    let horizon = clamp_row(row_of(0.0));
    let fade = clamp_row(row_of((0.25f64).asin()));
    // Task 2.10: the outermost ring sits ~16 km out. Task 2.13 E keeps
    // that reach with different numbers (16 cubes x 32 blocks x 2^5
    // instead of 8 x 32 x 2^6), so the extent is unchanged.
    let ring_extent = 8.0 * 256.0 * 8.0 + 512.0;
    let edge = clamp_row(row_of(((camera.position.y - 76.0) / ring_extent).atan()));
    ((fade - horizon).max(edge - horizon) + 8.0)
        .ceil()
        .clamp(0.0, f64::from(HEIGHT)) as u32
}

/// The checked region below the horizon margin: all columns from
/// `margin_rows(camera)` down.
fn below_margin_region(camera: &Camera) -> (u32, u32, u32, u32) {
    (0, margin_rows(camera), WIDTH, HEIGHT)
}

/// The checked region near the camera: the middle half of the frame,
/// which at pitch -15 degrees and these distances covers well inside
/// 512 blocks of the camera.
fn near_region() -> (u32, u32, u32, u32) {
    (
        WIDTH / 4,
        HEIGHT / 4,
        WIDTH - WIDTH / 4,
        HEIGHT - HEIGHT / 4,
    )
}

#[test]
#[ignore = "slow real-server flight; run with --ignored"]
fn fly_1p5km_over_real_rings_shows_no_holes() {
    let start = Instant::now();
    let Some(gpu) = jbm_client::testing::screenshot::gpu_or_skip() else {
        return;
    };

    // A real threaded server at game scale.
    let handle: ServerHandle = run::spawn(Server::with_config(ServerConfig {
        gen_threads: 4,
        ..ServerConfig::default()
    }));
    let (mut client, server_end) = jbm_core::link::in_process_pair();
    handle.connect(server_end);
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "flyer".to_owned(),
        })
        .unwrap();

    let mut fly = FlyClient::new(&gpu);

    // Join: send a climbing fly-toggle tick until the Welcome arrives,
    // then keep climbing to cruise altitude. The toggle is re-sent
    // until a PlayerState confirms flying: physics discards a whole
    // step (toggle included) while the player's own column is not
    // loaded yet, and worldgen races the first steps after spawn.
    // `toggle_seq` is the seq of the last toggle sent (0 = none yet);
    // `needs_fly_toggle` keeps at most one unacked toggle in flight so
    // retries can never flip flying back off.
    let mut seq = 0u64;
    let mut toggle_seq = 0u64;
    let mut spawn: Option<[f64; 3]> = None;
    while spawn.is_none() {
        seq += 1;
        let fly_toggle = fly.needs_fly_toggle(toggle_seq);
        if fly_toggle {
            toggle_seq = seq;
        }
        let input = jbm_core::physics::MoveInput {
            fly_toggle,
            up: 1,
            ..jbm_core::physics::MoveInput::IDLE
        };
        client
            .send(ClientMsg::PlayerInput {
                seq,
                input,
                pitch: -15.0_f32.to_radians(),
            })
            .unwrap();
        std::thread::sleep(Duration::from_millis(50));
        if let Some(reason) = fly.pump_link(&gpu, &mut client, &mut spawn) {
            panic!("link died before joining: {reason}");
        }
        assert!(start.elapsed() < TEST_TIMEOUT, "never joined");
    }
    let spawn_at = DVec3::from(spawn.unwrap());

    // The camera sits at the server's authoritative player position (the
    // eye above the feet), so the client renders exactly where the server
    // streams.
    const EYE: f64 = 1.62;
    let mut camera = Camera::new(
        spawn_at + DVec3::new(0.0, EYE, 0.0),
        70.0_f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
    );
    camera.set_pitch(-15.0_f32.to_radians());

    // Climb to cruise altitude: 40 blocks over the highest ridge the
    // first leg will meet (task 2.8's terrain-following target; the
    // classic terrain needed only spawn + 40). The toggle keeps being
    // re-sent (one unacked in flight) until the server confirms flying,
    // for the same lost-step race as above.
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let nav = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let cruise = |x: f64, z: f64, leg: Leg| cruise_target(&nav, x, z, leg);
    while fly
        .server_pos
        .is_none_or(|p| p.y < cruise(p.x, p.z, COURSE[0]))
    {
        seq += 1;
        let fly_toggle = fly.needs_fly_toggle(toggle_seq);
        if fly_toggle {
            toggle_seq = seq;
        }
        client
            .send(ClientMsg::PlayerInput {
                seq,
                input: jbm_core::physics::MoveInput {
                    fly_toggle,
                    up: 1,
                    ..jbm_core::physics::MoveInput::IDLE
                },
                pitch: -15.0_f32.to_radians(),
            })
            .unwrap();
        fly.frames_for_one_tick(&gpu, &camera);
        if let Some(reason) = fly.pump_link(&gpu, &mut client, &mut spawn) {
            panic!("link died while climbing: {reason}");
        }
        if let Some(p) = fly.server_pos {
            camera.position = p + DVec3::new(0.0, EYE, 0.0);
        }
        assert!(
            start.elapsed() < TEST_TIMEOUT,
            "never reached cruise altitude"
        );
    }

    // Settle the initial view: the queue stays clear for a full second.
    let settle_deadline = Instant::now() + Duration::from_secs(60);
    let mut clear_ticks = 0;
    while clear_ticks < 20 {
        fly.frames_for_one_tick(&gpu, &camera);
        if let Some(reason) = fly.pump_link(&gpu, &mut client, &mut spawn) {
            panic!("link died while settling: {reason}");
        }
        clear_ticks = if fly.queue_clear() {
            clear_ticks + 1
        } else {
            0
        };
        assert!(
            Instant::now() < settle_deadline,
            "initial view never settled"
        );
    }
    let settled_in = start.elapsed();
    let image = fly.render(&gpu, &camera);
    std::fs::create_dir_all("target/fly").ok();
    image.save("target/fly/start.png").ok();
    println!(
        "initial view settled after {:.1} s: {}",
        settled_in.as_secs_f64(),
        describe_holes(&sentinel_components(&image, below_margin_region(&camera)))
    );

    let mut course_flown = [0.0f64; 3];
    for (leg_index, &leg) in COURSE.iter().enumerate() {
        let yaw = leg.yaw as f32;
        camera.set_yaw(yaw);
        while course_flown[leg_index] < leg.distance {
            // 100 blocks at sprint-fly speed (22 blocks/s): about 5
            // seconds of 20 Hz ticks. `up` holds the terrain-following
            // target: full climb while below it, level above. The target
            // only rises (a look-ahead max), so the flight never descends
            // back into a ridge it just cleared.
            for _ in 0..100 {
                seq += 1;
                let up = fly
                    .server_pos
                    .map_or(0, |p| i8::from(p.y < cruise(p.x, p.z, leg)));
                client
                    .send(ClientMsg::PlayerInput {
                        seq,
                        input: jbm_core::physics::MoveInput {
                            forward: 1,
                            sprint: true,
                            yaw,
                            up,
                            ..jbm_core::physics::MoveInput::IDLE
                        },
                        pitch: -15.0_f32.to_radians(),
                    })
                    .unwrap();
                fly.frames_for_one_tick(&gpu, &camera);
                if let Some(reason) = fly.pump_link(&gpu, &mut client, &mut spawn) {
                    panic!("link died mid-flight: {reason}");
                }
                if let Some(p) = fly.server_pos {
                    camera.position = p + DVec3::new(0.0, EYE, 0.0);
                }
                assert!(start.elapsed() < TEST_TIMEOUT, "timeout mid-flight");
            }
            course_flown[leg_index] += 100.0;
            let flown_total: f64 = course_flown.iter().sum();
            let flown_real = fly.server_pos.map_or(0.0, |p| {
                (DVec3::new(p.x, 0.0, p.z) - DVec3::new(spawn_at.x, 0.0, spawn_at.z)).length()
            });

            // One frame WITHOUT waiting: holes near the camera are
            // asserted away; farther ones are only reported.
            let impatient = fly.render(&gpu, &camera);
            let unwaited_path = std::path::Path::new("target/fly")
                .join(format!("stop_{flown_total:.0}_unwaited.png"));
            impatient.save(&unwaited_path).ok();
            println!(
                "fly {flown_total:.0} blocks (server {flown_real:.0} from spawn, at {:.0?}): unwaited below margin: {}",
                camera.position,
                describe_holes(&sentinel_components(
                    &impatient,
                    below_margin_region(&camera)
                ))
            );
            assert_only_pinholes(
                &impatient,
                near_region(),
                &format!("fly {flown_total:.0} unwaited near camera"),
                &unwaited_path,
            );

            // Wait (bounded) for the client mesh queue only; the
            // server's rings are never polled for readiness.
            let deadline = Instant::now() + STOP_TIMEOUT;
            loop {
                fly.frames_for_one_tick(&gpu, &camera);
                if let Some(reason) = fly.pump_link(&gpu, &mut client, &mut spawn) {
                    panic!("link died at a stop: {reason}");
                }
                if fly.queue_clear() || Instant::now() > deadline {
                    break;
                }
            }
            // Drain remaining mesh work before rendering.
            for _ in 0..200 {
                fly.pump_meshing(&gpu, &camera);
                if fly.queue_clear() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let image = fly.render(&gpu, &camera);
            let path =
                std::path::Path::new("target/fly").join(format!("stop_{flown_total:.0}.png"));
            std::fs::create_dir_all(path.parent().unwrap()).ok();
            image.save(&path).ok();
            assert_only_pinholes(
                &image,
                below_margin_region(&camera),
                &format!(
                    "fly {flown_total:.0} below the horizon margin ({} rows), image {}",
                    margin_rows(&camera),
                    path.display()
                ),
                &path,
            );
            assert!(start.elapsed() < TEST_TIMEOUT, "timeout at a stop");
        }
    }

    // Task 2.10: one stop looking straight down from 400 blocks above
    // the ground — the play-test's "coarse blocks in the valley under a
    // flying player" view. The surface-following windows must have LOD0
    // (or its own level's finest) under the camera; the pinhole rule
    // covers the whole frame (at pitch -89° the horizon margin row
    // clamps to the top).
    let ground = fly.server_pos.map_or(0.0, |p| {
        nav.surface_height(p.x.floor() as i64, p.z.floor() as i64) as f64
    });
    while fly.server_pos.is_none_or(|p| p.y < ground + 400.0) {
        seq += 1;
        client
            .send(ClientMsg::PlayerInput {
                seq,
                input: jbm_core::physics::MoveInput {
                    up: 1,
                    ..jbm_core::physics::MoveInput::IDLE
                },
                pitch: -89.0_f32.to_radians(),
            })
            .unwrap();
        fly.frames_for_one_tick(&gpu, &camera);
        if let Some(reason) = fly.pump_link(&gpu, &mut client, &mut spawn) {
            panic!("link died while climbing to the lookdown: {reason}");
        }
        if let Some(p) = fly.server_pos {
            camera.position = p + DVec3::new(0.0, EYE, 0.0);
        }
        assert!(
            start.elapsed() < TEST_TIMEOUT,
            "never reached lookdown altitude"
        );
    }
    camera.set_pitch(-89.0_f32.to_radians());
    let deadline = Instant::now() + STOP_TIMEOUT;
    loop {
        fly.frames_for_one_tick(&gpu, &camera);
        if let Some(reason) = fly.pump_link(&gpu, &mut client, &mut spawn) {
            panic!("link died at the lookdown: {reason}");
        }
        if fly.queue_clear() || Instant::now() > deadline {
            break;
        }
    }
    for _ in 0..200 {
        fly.pump_meshing(&gpu, &camera);
        if fly.queue_clear() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let image = fly.render(&gpu, &camera);
    let path = std::path::Path::new("target/fly/lookdown_400.png");
    image.save(path).ok();
    assert_only_pinholes(
        &image,
        below_margin_region(&camera),
        "fly lookdown from 400 blocks",
        path,
    );

    handle.stop();
}
