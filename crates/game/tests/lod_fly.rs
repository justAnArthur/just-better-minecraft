//! The real-server LOD fly test (task 2.6). Spawn, fly 1.5 km straight
//! and then 90 degrees for 300 m more, 40 blocks above the terrain at
//! pitch -15 degrees, through the real client world, mesh pool and
//! renderer against a threaded real server. Every 100 blocks it waits
//! (bounded) for the server's rings to be ready and sent and the client
//! mesh queue to drain, then asserts **zero sentinel pixels** below the
//! horizon margin. It also renders one frame *without* waiting at every
//! stop, reports those counts, and asserts they are zero within 512
//! blocks of the camera. Screenshots land under `target/fly/` for the
//! orchestrator to inspect.
//!
//! Ignored until 2.7 merges: on this branch the pyramid only fills
//! around visited LOD0 terrain, so the far rings stay missing and the
//! zero-hole assertion cannot hold. Run with `cargo test -p jbm-game
//! --test lod_fly -- --ignored` once `stage-2/wave-c` is merged.
#![cfg(not(feature = "cube16"))]

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::gpu::Gpu;
use jbm_client::lod::Coverage;
use jbm_client::mesh::appearance::{BlockAppearance, builtin_textures};
use jbm_client::mesh::pool::MeshPool;
use jbm_client::render::chunks::TerrainPass;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{
    DepthTarget, FrameInputs, HORIZON_COLOR, Renderer, ZENITH_COLOR,
};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::world::ClientWorld;
use jbm_core::link::ClientLink;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
use jbm_server::run::{self, ServerHandle};
use jbm_server::server::{Server, ServerConfig};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const SENTINEL: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
/// The whole test must finish within this; a hang fails instead of
/// hanging the suite.
const TEST_TIMEOUT: Duration = Duration::from_secs(600);
/// Per-stop wait for the rings to complete.
const STOP_TIMEOUT: Duration = Duration::from_secs(20);

/// The fly course: 1500 blocks at yaw 0 (toward -Z), then 300 at yaw
/// +90 degrees (toward +X), 40 above the surface, pitch -15 degrees.
const COURSE: [(f32, f64); 2] = [(0.0, 1500.0), (std::f32::consts::FRAC_PI_2, 300.0)];

/// The client half: world, mesh pool, coverage, renderer.
struct FlyClient {
    world: ClientWorld,
    pool: MeshPool,
    coverage: Coverage,
    in_flight: usize,
    /// The authoritative player position, from the last `PlayerState`.
    server_pos: Option<DVec3>,
    renderer: Renderer,
    terrain: TerrainPass,
    target: OffscreenTarget,
    depth: DepthTarget,
}

impl FlyClient {
    fn new(gpu: &Gpu) -> FlyClient {
        let mut registry = jbm_core::block::BlockRegistry::new();
        jbm_core::block::builtin::register(&mut registry);
        let appearance = Arc::new(BlockAppearance::new(&registry));
        let target = OffscreenTarget::new(&gpu.device, WIDTH, HEIGHT);
        let depth = DepthTarget::new(&gpu.device, WIDTH, HEIGHT);
        let renderer = Renderer::new(gpu, target.format());
        let (layers, _) = builtin_textures();
        let textures = BlockTextures::new(gpu, &layers).expect("texture array");
        let terrain = TerrainPass::new(gpu, target.format(), renderer.uniform_layout(), &textures);
        FlyClient {
            world: ClientWorld::new(),
            pool: MeshPool::new(appearance, 0),
            coverage: Coverage::default(),
            in_flight: 0,
            server_pos: None,
            renderer,
            terrain,
            target,
            depth,
        }
    }

    /// Drains the link into the world, dropping GPU state for unloads
    /// and capturing the `Welcome`'s spawn. Returns a fatal reason when
    /// the link dies.
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
                    ServerMsg::PlayerState { state, .. } => {
                        self.server_pos = Some(state.position);
                    }
                    ServerMsg::CubeData { .. }
                    | ServerMsg::UnloadCube { .. }
                    | ServerMsg::LodCubeData { .. }
                    | ServerMsg::UnloadLodCube { .. }
                    | ServerMsg::BlockChange { .. } => {
                        if let jbm_client::world::WorldChange::Unloaded(key) =
                            self.world.apply(&msg)
                        {
                            self.terrain.remove(gpu, key);
                            self.coverage.set_ready(key, false);
                        }
                    }
                    _ => {}
                },
                Ok(None) => return None,
                Err(err) => return Some(err.to_string()),
            }
        }
    }

    /// Submits and uploads meshes within the app budgets.
    fn pump_meshing(&mut self, gpu: &Gpu, camera: &Camera) {
        let inputs = self.world.take_dirty(16, camera.position);
        self.in_flight += inputs.len();
        for (key, version, input) in inputs {
            self.pool.submit(key, version, input);
        }
        let mut uploads = 0;
        while uploads < 32 {
            let Some(result) = self.pool.try_recv() else {
                break;
            };
            self.in_flight -= 1;
            if self.world.version(result.key) != Some(result.version) {
                continue;
            }
            self.terrain.upload(gpu, result.key, &result.output);
            self.coverage.set_ready(result.key, true);
            uploads += 1;
        }
        for (key, drawn, mask) in self.coverage.take_changed() {
            self.terrain.set_draw_state(gpu, key, drawn, mask);
        }
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
        self.world.dirty_count() == 0 && self.in_flight == 0
    }

    /// Renders one frame with the sentinel sky and returns it.
    fn render(&mut self, gpu: &Gpu, camera: &Camera) -> image::RgbaImage {
        self.renderer
            .set_sky_colors(Some((HORIZON_COLOR, ZENITH_COLOR, SENTINEL)));
        self.renderer.render(
            gpu,
            FrameInputs {
                target: &self.target.create_view(),
                depth: &self.depth,
                camera,
                terrain: Some(&mut self.terrain),
                outline: None,
                overlay: None,
            },
        );
        let image = self.target.read_rgba(gpu);
        self.renderer.set_sky_colors(None);
        image
    }
}

/// Sentinel pixels below `min_row`.
fn sentinel_count_below(image: &image::RgbaImage, min_row: u32) -> usize {
    let mut count = 0;
    for y in min_row..image.height() {
        for x in 0..image.width() {
            let p = image.get_pixel(x, y);
            if p[0] >= 250 && p[1] <= 5 && p[2] >= 250 {
                count += 1;
            }
        }
    }
    count
}

/// The row margin per camera: the sky's ~15 degree below-horizon fade
/// (pure ground color only below `asin(0.25)`) plus the rings' far-edge
/// drop angle, plus slack for the outer cubes' missing far walls.
fn margin_rows(camera: &Camera) -> u32 {
    let row_of = |angle: f64| -> f64 {
        let below_axis = angle + f64::from(camera.pitch);
        let ndc_y = -below_axis.tan() / (f64::from(camera.fov_y) / 2.0).tan();
        (0.5 - ndc_y * 0.5) * f64::from(HEIGHT)
    };
    let horizon = row_of(0.0);
    let fade = row_of((0.25f64).asin());
    let ring_extent = 8.0 * 256.0 + 512.0;
    let edge = row_of(((camera.position.y - 76.0) / ring_extent).atan());
    ((fade - horizon).max(edge - horizon) + 8.0).ceil().max(0.0) as u32
}

/// Sentinel pixels near the camera: the middle half of the frame, which
/// at pitch -15 degrees and these distances covers well inside 512
/// blocks of the camera.
fn sentinels_near(image: &image::RgbaImage) -> usize {
    let mut count = 0;
    for y in HEIGHT / 4..HEIGHT - HEIGHT / 4 {
        for x in WIDTH / 4..WIDTH - WIDTH / 4 {
            let p = image.get_pixel(x, y);
            if p[0] >= 250 && p[1] <= 5 && p[2] >= 250 {
                count += 1;
            }
        }
    }
    count
}

#[test]
#[ignore = "needs 2.7 (worldgen LOD fill) merged; run with --ignored after stage-2/wave-c"]
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
    // then keep climbing to cruise altitude.
    let mut seq = 0u64;
    let mut flying_sent = false;
    let mut spawn: Option<[f64; 3]> = None;
    while spawn.is_none() {
        seq += 1;
        let input = jbm_core::physics::MoveInput {
            fly_toggle: !flying_sent,
            up: 1,
            ..jbm_core::physics::MoveInput::IDLE
        };
        flying_sent = true;
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

    // Climb to cruise altitude, 40 blocks over spawn.
    while fly.server_pos.is_none_or(|p| p.y < spawn_at.y + 40.0) {
        seq += 1;
        client
            .send(ClientMsg::PlayerInput {
                seq,
                input: jbm_core::physics::MoveInput {
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
        "initial view settled after {:.1} s: {} sentinel pixels below the margin",
        settled_in.as_secs_f64(),
        sentinel_count_below(&image, margin_rows(&camera))
    );

    let mut course_flown = [0.0f64; 2];
    for (leg_index, &(yaw, leg)) in COURSE.iter().enumerate() {
        camera.set_yaw(yaw);
        while course_flown[leg_index] < leg {
            // 100 blocks at sprint-fly speed (22 blocks/s): about 5
            // seconds of 20 Hz ticks.
            for _ in 0..100 {
                seq += 1;
                client
                    .send(ClientMsg::PlayerInput {
                        seq,
                        input: jbm_core::physics::MoveInput {
                            forward: 1,
                            sprint: true,
                            yaw,
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
            impatient
                .save(format!("target/fly/stop_{flown_total:.0}_unwaited.png"))
                .ok();
            let near = sentinels_near(&impatient);
            println!(
                "fly {flown_total:.0} blocks (server {flown_real:.0} from spawn, at {:.0?}): unwaited sentinels near camera: {near}, below margin: {}",
                camera.position,
                sentinel_count_below(&impatient, margin_rows(&camera))
            );
            assert_eq!(
                near, 0,
                "unwaited frame has holes within 512 blocks at {flown_total:.0}"
            );

            // Wait (bounded) for the rings and the mesh queue.
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
            let margin = margin_rows(&camera);
            let sentinels = sentinel_count_below(&image, margin);
            let path =
                std::path::Path::new("target/fly").join(format!("stop_{flown_total:.0}.png"));
            std::fs::create_dir_all(path.parent().unwrap()).ok();
            image.save(&path).ok();
            println!(
                "fly {flown_total:.0}: {sentinels} sentinel pixels below the margin ({margin} rows), image {}",
                path.display()
            );
            assert_eq!(
                sentinels,
                0,
                "holes below the horizon margin at {flown_total:.0} blocks (image {})",
                path.display()
            );
            assert!(start.elapsed() < TEST_TIMEOUT, "timeout at a stop");
        }
    }

    handle.stop();
}
