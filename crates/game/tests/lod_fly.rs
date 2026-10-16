//! The real-server LOD fly test (task 2.6). Spawn, fly 1.5 km straight
//! and then 90 degrees for 300 m more, 40 blocks above the terrain at
//! pitch -15 degrees, through the real client world, mesh pool and
//! renderer against a threaded real server. Every 100 blocks it waits
//! (bounded) for the server's rings to be ready and sent and the client
//! mesh queue to drain, then checks the frame below the horizon
//! margin. It also renders one frame *without* waiting at every stop
//! and checks the middle half of it, which at these distances covers
//! well inside 512 blocks of the camera. Screenshots land under
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
//! is a follow-up. The detector itself is plain code over a pixel
//! buffer and is unit-tested without a GPU.
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
    /// The authoritative flying flag, and the last acknowledged input
    /// seq (`PlayerState::ack_seq`).
    server_flying: bool,
    server_ack: u64,
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
            server_flying: false,
            server_ack: 0,
            renderer,
            terrain,
            target,
            depth,
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

/// A 4-connected component of sentinel pixels in a checked region.
struct HoleComponent {
    pixels: Vec<(u32, u32)>,
}

impl HoleComponent {
    fn size(&self) -> usize {
        self.pixels.len()
    }

    /// The min and max corners of the bounding box, in image pixels.
    fn bounds(&self) -> ((u32, u32), (u32, u32)) {
        let mut min = (u32::MAX, u32::MAX);
        let mut max = (0, 0);
        for &(x, y) in &self.pixels {
            min.0 = min.0.min(x);
            min.1 = min.1.min(y);
            max.0 = max.0.max(x);
            max.1 = max.1.max(y);
        }
        (min, max)
    }
}

/// A pinhole is at most this many pixels in one component.
const PINHOLE_MAX_PIXELS: usize = 2;
/// At most this many pinhole pixels per frame in the checked region.
const PINHOLE_MAX_TOTAL: usize = 8;
/// Highlight for offending components in debug images; must differ
/// from the magenta sentinel and the sky gradient.
const HOLE_HIGHLIGHT: [u8; 4] = [255, 255, 0, 255];

fn is_sentinel(p: &image::Rgba<u8>) -> bool {
    p[0] >= 250 && p[1] <= 5 && p[2] >= 250
}

/// The 4-connected components of sentinel pixels inside `region`
/// (`x0..x1` by `y0..y1`, end-exclusive).
fn sentinel_components(
    image: &image::RgbaImage,
    region: (u32, u32, u32, u32),
) -> Vec<HoleComponent> {
    let (x0, y0, x1, y1) = region;
    let stride = x1 - x0;
    let mut visited = vec![false; (stride * (y1 - y0)) as usize];
    let in_region = |x: i64, y: i64| {
        x >= i64::from(x0) && x < i64::from(x1) && y >= i64::from(y0) && y < i64::from(y1)
    };
    let mut components = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let entry = ((y - y0) * stride + (x - x0)) as usize;
            if visited[entry] || !is_sentinel(image.get_pixel(x, y)) {
                continue;
            }
            // Flood fill from (x, y).
            let mut pixels = Vec::new();
            let mut stack = vec![(x, y)];
            visited[entry] = true;
            while let Some((cx, cy)) = stack.pop() {
                pixels.push((cx, cy));
                for (dx, dy) in [(1_i64, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (i64::from(cx) + dx, i64::from(cy) + dy);
                    if !in_region(nx, ny) {
                        continue;
                    }
                    let (nx, ny) = (nx as u32, ny as u32);
                    let neighbor = ((ny - y0) * stride + (nx - x0)) as usize;
                    if !visited[neighbor] && is_sentinel(image.get_pixel(nx, ny)) {
                        visited[neighbor] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            components.push(HoleComponent { pixels });
        }
    }
    components
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

/// "N components, largest M px, T sentinel px" — the per-stop report.
fn describe_holes(components: &[HoleComponent]) -> String {
    let largest = components
        .iter()
        .map(HoleComponent::size)
        .max()
        .unwrap_or(0);
    let total: usize = components.iter().map(HoleComponent::size).sum();
    let count = components.len();
    format!("{count} components, largest {largest} px, {total} sentinel px")
}

/// Pixels in components small enough to be pinholes.
fn pinhole_total(components: &[HoleComponent]) -> usize {
    components
        .iter()
        .filter(|c| c.size() <= PINHOLE_MAX_PIXELS)
        .map(HoleComponent::size)
        .sum()
}

/// Whether every component is a pinhole and they fit the per-frame cap.
fn only_pinholes(components: &[HoleComponent]) -> bool {
    components.iter().all(|c| c.size() <= PINHOLE_MAX_PIXELS)
        && pinhole_total(components) <= PINHOLE_MAX_TOTAL
}

/// The per-stop hole check: sentinel pixels in `region` must be pinhole
/// components only, within the per-frame cap. Prints the stats; on
/// failure saves the frame at `image_path`, saves a debug image with
/// the offending components highlighted next to it, and panics with
/// their bounding boxes.
fn assert_only_pinholes(
    image: &image::RgbaImage,
    region: (u32, u32, u32, u32),
    label: &str,
    image_path: &std::path::Path,
) {
    let components = sentinel_components(image, region);
    println!("{label}: {}", describe_holes(&components));
    if only_pinholes(&components) {
        return;
    }
    let oversized = components
        .iter()
        .filter(|c| c.size() > PINHOLE_MAX_PIXELS)
        .count();
    // On a total-cap failure every component contributes; otherwise
    // only the oversized ones offend.
    let over_total = pinhole_total(&components) > PINHOLE_MAX_TOTAL;
    let offending: Vec<&HoleComponent> = if over_total {
        components.iter().collect()
    } else {
        components
            .iter()
            .filter(|c| c.size() > PINHOLE_MAX_PIXELS)
            .collect()
    };
    let mut debug = image.clone();
    for component in &offending {
        for &(x, y) in &component.pixels {
            debug.put_pixel(x, y, image::Rgba(HOLE_HIGHLIGHT));
        }
    }
    let debug_path = image_path.with_extension("holes.png");
    std::fs::create_dir_all(image_path.parent().unwrap()).ok();
    image.save(image_path).ok();
    debug.save(&debug_path).ok();
    let boxes: Vec<String> = offending
        .iter()
        .map(|c| {
            let ((min_x, min_y), (max_x, max_y)) = c.bounds();
            format!("({min_x},{min_y})..({max_x},{max_y}) {}px", c.size())
        })
        .collect();
    panic!(
        "{label}: holes beyond pinhole tolerance ({PINHOLE_MAX_PIXELS} px per \
         component, {PINHOLE_MAX_TOTAL} px per frame): {oversized} oversized \
         components, pinhole total {}, offending [{}]; images {} and {}",
        pinhole_total(&components),
        boxes.join(", "),
        image_path.display(),
        debug_path.display(),
    );
}

/// A black frame with sentinel pixels at the given coordinates, for the
/// detector's unit test.
fn synthetic_frame(sentinels: &[(u32, u32)]) -> image::RgbaImage {
    let mut image = image::RgbaImage::from_pixel(WIDTH, HEIGHT, image::Rgba([0, 0, 0, 255]));
    for &(x, y) in sentinels {
        image.put_pixel(x, y, image::Rgba([255, 0, 255, 255]));
    }
    image
}

#[test]
fn hole_detector_distinguishes_pinholes_from_holes() {
    let region = near_region();

    // Isolated single pixels (diagonals do not connect: 4-connected)
    // and a horizontal 2-pixel pair are pinholes.
    let pinholes = synthetic_frame(&[
        (WIDTH / 4 + 4, HEIGHT / 4 + 4),
        (WIDTH / 4 + 5, HEIGHT / 4 + 5),
        (WIDTH / 4 + 40, HEIGHT / 4 + 4),
        (WIDTH / 4 + 60, HEIGHT / 4 + 20),
        (WIDTH / 4 + 61, HEIGHT / 4 + 20),
    ]);
    let components = sentinel_components(&pinholes, region);
    assert_eq!(components.iter().map(HoleComponent::size).sum::<usize>(), 5);
    assert_eq!(components.len(), 4, "the pair merges, diagonals do not");
    assert!(only_pinholes(&components));

    // A 3-pixel L shape is too big for a pinhole.
    let l_shape = synthetic_frame(&[
        (WIDTH / 4 + 10, HEIGHT / 4 + 10),
        (WIDTH / 4 + 11, HEIGHT / 4 + 10),
        (WIDTH / 4 + 10, HEIGHT / 4 + 11),
    ]);
    let components = sentinel_components(&l_shape, region);
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].size(), 3);
    assert!(!only_pinholes(&components));

    // A 1x20 strip — the shape of a ring-border crack — fails.
    let strip: Vec<(u32, u32)> = (0..20)
        .map(|i| (WIDTH / 4 + 30, HEIGHT / 4 + 10 + i))
        .collect();
    let components = sentinel_components(&synthetic_frame(&strip), region);
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].size(), 20);
    assert_eq!(
        components[0].bounds(),
        (
            (WIDTH / 4 + 30, HEIGHT / 4 + 10),
            (WIDTH / 4 + 30, HEIGHT / 4 + 29)
        )
    );
    assert!(!only_pinholes(&components));

    // Nine scattered singles pass the component size but bust the
    // per-frame total of 8.
    let scattered: Vec<(u32, u32)> = (0..9)
        .map(|i| (WIDTH / 4 + 5 * i, HEIGHT / 4 + 50))
        .collect();
    let components = sentinel_components(&synthetic_frame(&scattered), region);
    assert_eq!(components.len(), 9);
    assert_eq!(pinhole_total(&components), 9);
    assert!(!only_pinholes(&components));
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

    // Climb to cruise altitude, 40 blocks over spawn. The toggle keeps
    // being re-sent (one unacked in flight) until the server confirms
    // flying, for the same lost-step race as above.
    while fly.server_pos.is_none_or(|p| p.y < spawn_at.y + 40.0) {
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

    handle.stop();
}
