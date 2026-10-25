//! Task 2.13 A: the far-view measurement harness (`far_metrics`). An
//! ignored test (run with `-- --ignored --nocapture`) that settles a real
//! threaded server on the overworld (`DEFAULT_SEED`) at a pinned camera,
//! then prints one aligned table per camera and `view_radius`: desired /
//! resident / meshed cubes per level, faces and triangles per level,
//! indirect commands and draw calls of one frame, the mesher's per-cube
//! timings over the settled LOD0 set (p50 / p99 / total — the same
//! `mesh()` the worker pool runs, timed inline so the numbers are
//! per-cube), the quad-split study (re-meshing with AO, then the texture
//! layer, dropped from the merge key), server settle time and memory, the
//! client's face-page bytes, and one frame's CPU cost (encode + GPU wait;
//! the GPU time per pass is `perf_views`'s job). Two cameras — spawn and a
//! deterministic mountain top. The baseline run pins the pre-2.13 rings
//! (`view_radius` 8 and 16, `lod_view_radius` 8, `lod_levels` 6); the
//! task 2.13 E `far_metrics_defaults` run measures the shipped
//! defaults, and `far_shots` saves the tinted before/after PNGs.
//! Everything the task's later steps are judged against comes from here.
//! Task 2.16 adds `perf_views`, the frame-rate harness: CPU and GPU
//! (timestamp) frame times of the user's view and the look-review cameras
//! at 2560×1440 and 5120×2880.
#![cfg(not(feature = "cube16"))]

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::gpu::Gpu;
use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::mesh::{self, MeshInput, MeshTops, neighbor_index};
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::settings::RenderSettings;
use jbm_client::render::terrain::TerrainPass;
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::render::timing::GpuTimings;
use jbm_client::terrain_sync::TerrainSync;
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::link::ClientLink;
use jbm_core::protocol::PlayerId;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
use jbm_server::DEFAULT_SEED;
use jbm_server::server::{Server, ServerConfig};
use jbm_server::terrain::Generator;
use jbm_server::worldgen::Preset;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FRAMES: usize = 50;
const WARMUP_FRAMES: usize = 3;
/// Whole-run bound; a hang fails instead of hanging the suite.
const TEST_TIMEOUT: Duration = Duration::from_secs(1800);

fn row(label: &str, value: impl std::fmt::Display) {
    println!("{label:<46}{value:>16}");
}

/// The tallest surface in a `side`-block sample square centered on the
/// spawn — task 2.10's deterministic "on a mountain" position.
fn mountain_top(generator: &Generator, side: i64) -> DVec3 {
    let start = generator.spawn();
    let mut best = (i64::MIN, 0i64, 0i64);
    for x in (-side..=side).step_by(64) {
        for z in (-side..=side).step_by(64) {
            let h = generator.surface_height(start[0] as i64 + x, start[2] as i64 + z);
            if h > best.0 {
                best = (h, start[0] as i64 + x, start[2] as i64 + z);
            }
        }
    }
    DVec3::new(
        best.1 as f64 + 0.5,
        (best.0 + 2) as f64,
        best.2 as f64 + 0.5,
    )
}

/// The client half: the protocol state, the terrain pipeline and its own
/// offscreen frame targets (so a frame can be timed without the readback
/// `OffscreenRig::render_with_sky` does).
struct MetricsClient {
    sync: TerrainSync,
    renderer: Renderer,
    terrain: TerrainPass,
    depth: DepthTarget,
    target: OffscreenTarget,
    appearance: Arc<BlockAppearance>,
    received_lod0: HashSet<CubePos>,
}

impl MetricsClient {
    fn new(gpu: &Gpu) -> MetricsClient {
        MetricsClient::with_settings(gpu, RenderSettings::default())
    }

    /// A client at `WIDTH`×`HEIGHT` whose renderer and terrain pass run
    /// with `settings` (the perf harness passes `RenderSettings::from_env`).
    fn with_settings(gpu: &Gpu, settings: RenderSettings) -> MetricsClient {
        let mut registry = jbm_core::block::BlockRegistry::new();
        jbm_core::block::builtin::register(&mut registry);
        let appearance = Arc::new(BlockAppearance::new(&registry));
        let target = OffscreenTarget::new(&gpu.device, WIDTH, HEIGHT);
        let depth = DepthTarget::with_samples(&gpu.device, WIDTH, HEIGHT, settings.msaa_samples);
        let mut renderer = Renderer::with_settings(gpu, target.format(), settings);
        renderer.set_viewport(WIDTH, HEIGHT);
        let textures = BlockTextures::builtin(gpu).expect("texture array");
        let terrain = TerrainPass::with_settings(
            gpu,
            target.format(),
            renderer.uniform_layout(),
            &textures,
            &settings,
        );
        MetricsClient {
            sync: TerrainSync::new(Arc::clone(&appearance), 0),
            renderer,
            terrain,
            depth,
            target,
            appearance,
            received_lod0: HashSet::new(),
        }
    }

    fn pump_link(&mut self, gpu: &Gpu, client: &mut ClientLink) -> Option<String> {
        loop {
            match client.try_recv() {
                Ok(Some(msg)) => match &msg {
                    ServerMsg::Rejected { reason } | ServerMsg::Disconnected { reason } => {
                        return Some(reason.clone());
                    }
                    ServerMsg::CubeData { pos, .. } => {
                        self.received_lod0.insert(*pos);
                        self.sync.apply(gpu, &mut self.terrain, &msg);
                    }
                    ServerMsg::UnloadCube { pos } => {
                        self.received_lod0.remove(pos);
                        self.sync.apply(gpu, &mut self.terrain, &msg);
                    }
                    ServerMsg::LodCubeData { .. }
                    | ServerMsg::UnloadLodCube { .. }
                    | ServerMsg::BlockChange { .. } => {
                        self.sync.apply(gpu, &mut self.terrain, &msg);
                    }
                    _ => {}
                },
                Ok(None) => return None,
                Err(err) => return Some(err.to_string()),
            }
        }
    }

    /// Renders one frame (sky + terrain) without reading it back.
    fn render_frame(&mut self, gpu: &Gpu, camera: &Camera) {
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
    }

    /// Recreates the color target at `width`×`height` and the depth
    /// target at the renderer's scene size for it (the render scale,
    /// task 2.16), as the app does.
    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        self.target = OffscreenTarget::new(&gpu.device, width, height);
        self.renderer.set_viewport(width, height);
        let (scene_width, scene_height) = self.renderer.scene_size();
        self.depth = DepthTarget::with_samples(
            &gpu.device,
            scene_width,
            scene_height,
            self.renderer.msaa_samples(),
        );
    }

    fn pump_meshing(&mut self, gpu: &Gpu, camera: &Camera) {
        self.sync.pump(gpu, &mut self.terrain, camera, 16, 32);
    }

    fn queue_clear(&self) -> bool {
        self.sync.queue_clear()
    }
}

/// Joins the player (bounded) and teleports them to `site`.
fn join_and_teleport(
    server: &mut Server,
    win: &mut MetricsClient,
    client: &mut ClientLink,
    gpu: &Gpu,
    site: DVec3,
    deadline: Instant,
) {
    let mut joined = false;
    while !joined {
        server.tick();
        client
            .send(ClientMsg::PlayerInput {
                seq: 1,
                input: jbm_core::physics::MoveInput::IDLE,
                pitch: 0.0,
            })
            .unwrap();
        std::thread::sleep(Duration::from_millis(20));
        if let Some(reason) = win.pump_link(gpu, client) {
            panic!("link died before joining: {reason}");
        }
        joined = server.players().next().is_some();
        assert!(Instant::now() < deadline, "never joined");
    }
    server.teleport_player(PlayerId(1), site);
}

/// Pumps until everything is quiet (spans filled, nothing pending or in
/// flight anywhere, the client mesh queue clear for a run of polls) and
/// returns how long that took.
fn settle(
    server: &mut Server,
    win: &mut MetricsClient,
    client: &mut ClientLink,
    gpu: &Gpu,
    camera: &Camera,
    deadline: Instant,
) -> f64 {
    let started = Instant::now();
    let id = PlayerId(1);
    let mut clear_polls = 0;
    while clear_polls < 10 && Instant::now() < deadline {
        server.update(Duration::from_millis(50));
        for _ in 0..3 {
            win.pump_meshing(gpu, camera);
            std::thread::sleep(Duration::from_millis(16));
        }
        if let Some(reason) = win.pump_link(gpu, client) {
            panic!("link died while settling: {reason}");
        }
        let (_, _, pending_spans, _) = server.span_stats();
        let quiet = win.queue_clear()
            && pending_spans == 0
            && server.lod0_pending(id) == 0
            && server.lod_ring_pending(id) == 0
            && server.in_flight() == 0;
        clear_polls = if quiet { clear_polls + 1 } else { 0 };
        assert!(Instant::now() < deadline, "never settled");
    }
    for _ in 0..200 {
        win.pump_meshing(gpu, camera);
        if win.queue_clear() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    started.elapsed().as_secs_f64()
}

/// One measurement run: settle at `site` with `config` and print the
/// table. `tag` names the rows.
fn measure_site(
    gpu: &Gpu,
    nav: &Generator,
    site: DVec3,
    config: ServerConfig,
    tag: &str,
    deadline: Instant,
) {
    println!(
        "\n=== {tag}: view_radius {} (lod_view_radius {}, lod_levels {}) ===",
        config.view_radius, config.lod_view_radius, config.lod_levels
    );
    let mut server = Server::with_config(config);
    let (mut client, server_end) = jbm_core::link::in_process_pair();
    server.add_connection(server_end);
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "far_metrics".to_owned(),
        })
        .unwrap();
    let mut win = MetricsClient::new(gpu);
    join_and_teleport(&mut server, &mut win, &mut client, gpu, site, deadline);

    const EYE: f64 = 1.62;
    let mut camera = Camera::new(
        site + DVec3::new(0.0, EYE, 0.0),
        70.0_f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
    );
    // Look back toward the spawn column, 15 degrees down: a deterministic
    // view that keeps terrain in frame at both sites.
    let spawn = DVec3::from(nav.spawn());
    let to_spawn = spawn - camera.position;
    camera.set_yaw(f32::atan2(to_spawn.x as f32, -to_spawn.z as f32));
    camera.set_pitch(-15.0_f32.to_radians());

    let settle_secs = settle(&mut server, &mut win, &mut client, gpu, &camera, deadline);
    let id = PlayerId(1);

    // Server side: settle time, desired (= sent, settled) per level,
    // resident memory. `sent_counts_per_level` is [L1..L6, LOD0].
    row("settle time (s)", format!("{settle_secs:.1}"));
    let sent_per_level = server.sent_counts_per_level(id);
    let lod0_sent = sent_per_level.last().copied().unwrap_or(0);
    row("LOD0 desired (= sent)", lod0_sent);
    row(
        "ring sent L1..L6",
        format!("{:?}", &sent_per_level[..sent_per_level.len() - 1]),
    );
    row("world cubes (server)", server.world_cube_count());
    row("world cube bytes (server)", server.world_heap_bytes());
    row(
        "lod cubes / heap bytes",
        format!("{} / {}", server.lod_cube_count(), server.lod_heap_bytes()),
    );

    // Client side: residency and coverage per level.
    row("LOD0 received (client)", win.received_lod0.len());
    for stats in win.terrain.level_stats() {
        println!(
            "  level {}: resident {:>6} drawn {:>6} faces {:>8} tris {:>8} skirts {:>5} held bytes {:>10}",
            stats.level,
            stats.resident,
            stats.drawn,
            stats.faces,
            stats.faces * 2,
            stats.skirts,
            stats.faces * 8
        );
    }

    // One frame's counters: commands, draws, per-level issued faces.
    for _ in 0..WARMUP_FRAMES {
        win.render_frame(gpu, &camera);
    }
    let cull = win.terrain.read_cull_stats_blocking(gpu);
    let stats = win.terrain.stats();
    row(
        "frame: commands / draw calls",
        format!("{} / {}", cull.commands, stats.draw_calls),
    );
    row(
        "frame: faces issued / tris",
        format!("{} / {}", cull.faces, cull.faces * 2),
    );
    row("frame: drawn per level", format!("{:?}", cull.level_drawn));
    row(
        "face pages / page bytes",
        format!("{} / {}", stats.pages, stats.page_bytes),
    );
    row(
        "used faces / meshed",
        format!("{} / {}", stats.used_faces, stats.meshed),
    );

    // Frame CPU cost: encode time and GPU wait over FRAMES frames (the
    // GPU time per pass is perf_views' job).
    let mut encode = Duration::ZERO;
    let mut wait = Duration::ZERO;
    let mut total = Duration::ZERO;
    for _ in 0..FRAMES {
        let start = Instant::now();
        let encode_start = Instant::now();
        win.render_frame(gpu, &camera);
        encode += encode_start.elapsed();
        let wait_start = Instant::now();
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll failed");
        wait += wait_start.elapsed();
        total += start.elapsed();
    }
    row(
        "frame CPU mean (encode / wait / all)",
        format!(
            "{:.3} / {:.3} / {:.3} ms",
            encode.as_secs_f64() * 1000.0 / FRAMES as f64,
            wait.as_secs_f64() * 1000.0 / FRAMES as f64,
            total.as_secs_f64() * 1000.0 / FRAMES as f64
        ),
    );

    // Mesher timings over the settled LOD0 set: the same mesh() the worker
    // pool runs, timed inline per cube (the pool itself does not time its
    // jobs), plus the quad-split study on the same inputs.
    let appearance = win.appearance.as_ref();
    let world = win.sync.world();
    let mut inputs = Vec::new();
    for pos in &win.received_lod0 {
        let key = jbm_client::lod::CubeKey::lod0(*pos);
        let Some(center) = world.cube(key) else {
            continue;
        };
        let mut neighbors = std::array::from_fn(|_| None);
        for dy in -1i32..=1 {
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    if (dx, dy, dz) == (0, 0, 0) {
                        continue;
                    }
                    let neighbor = jbm_client::lod::CubeKey::lod0(CubePos {
                        x: pos.x + dx,
                        y: pos.y + dy,
                        z: pos.z + dz,
                    });
                    if let Some(cube) = world.cube(neighbor) {
                        neighbors[neighbor_index(dx, dy, dz)] = Some(Arc::clone(cube));
                    }
                }
            }
        }
        inputs.push(MeshInput {
            center: Arc::clone(center),
            neighbors,
            level: 0,
            tops: MeshTops::default(),
        });
    }
    let mut timings: Vec<f64> = Vec::with_capacity(inputs.len());
    let total_start = Instant::now();
    let mut faces_real = 0usize;
    for input in &inputs {
        let start = Instant::now();
        faces_real += mesh::mesh(input, appearance).faces.len();
        timings.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let mesh_total = total_start.elapsed();
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| timings[(q * (timings.len() - 1) as f64).round() as usize];
    row(
        "mesh per cube p50 / p99 (us)",
        format!("{:.1} / {:.1}", p(0.5), p(0.99)),
    );
    row(
        "mesh total settled (s / faces)",
        format!("{:.2} / {}", mesh_total.as_secs_f64(), faces_real),
    );

    // Quad-split causes: AO dropped from the merge key, then the layer.
    let faces_no_ao = inputs
        .iter()
        .map(|input| mesh::mesh_split_study(input, appearance, true, false))
        .sum::<usize>();
    let faces_no_layer = inputs
        .iter()
        .map(|input| mesh::mesh_split_study(input, appearance, false, true))
        .sum::<usize>();
    row(
        "quads real / -AO / -layer",
        format!("{faces_real} / {faces_no_ao} / {faces_no_layer}"),
    );
    row(
        "tris real / -AO / -layer",
        format!(
            "{} / {} / {}",
            faces_real * 2,
            faces_no_ao * 2,
            faces_no_layer * 2
        ),
    );
}

/// The pre-2.13 ring shape the task A baseline was measured with:
/// `lod_view_radius` 8, `lod_levels` 6 ("the rings as today" then).
/// Pinned so the baseline test keeps its meaning after task 2.13 E
/// changed the defaults — with one caveat: at `view_radius` 16 the
/// 2.13 E ring-radius floor clamps `lod_view_radius` 8 up to 9, so the
/// vr16 baseline rows are no longer bit-reproducible; the recorded
/// table (target/2.13-far-baseline2.txt) is the authority for those.
fn baseline_rings(view_radius: u32) -> ServerConfig {
    ServerConfig {
        gen_threads: 4,
        view_radius,
        lod_view_radius: 8,
        lod_levels: 6,
        ..ServerConfig::default()
    }
}

/// Task 2.13 A's baseline table: spawn and the mountain top, each at
/// `view_radius` 8 and 16, the pre-2.13 rings (see [`baseline_rings`]).
/// Ignored (slow, GPU): run with
/// `cargo test -p jbm-game --test far_metrics -- --ignored --nocapture`.
#[test]
#[ignore = "slow real-server measurement; run with --ignored"]
fn far_metrics_baseline() {
    let start = Instant::now();
    let Some(gpu) = jbm_client::testing::screenshot::gpu_or_skip() else {
        return;
    };
    let mut registry = jbm_core::block::BlockRegistry::new();
    let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
    let nav = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let spawn = DVec3::from(nav.spawn());
    let mountain = mountain_top(&nav, 2048);
    println!("far_metrics: spawn {spawn:.0?}, mountain {mountain:.0?}, cube size {CUBE_SIZE}");

    for view_radius in [8u32, 16] {
        measure_site(
            &gpu,
            &nav,
            spawn,
            baseline_rings(view_radius),
            "spawn",
            start + TEST_TIMEOUT,
        );
    }
    for view_radius in [8u32, 16] {
        measure_site(
            &gpu,
            &nav,
            mountain,
            baseline_rings(view_radius),
            "mountain",
            start + TEST_TIMEOUT,
        );
    }
    println!(
        "\nfar_metrics done in {:.0} s",
        start.elapsed().as_secs_f64()
    );
}

/// Task 2.13 E: the same table at the shipped defaults — `view_radius`
/// 16, `lod_view_radius` 16, `lod_levels` 5 (LOD0 to 512 blocks, rings
/// at 1..16 km). Ignored (slow, GPU): run with
/// `cargo test -p jbm-game --test far_metrics far_metrics_defaults -- --ignored --nocapture`.
#[test]
#[ignore = "slow real-server measurement; run with --ignored"]
fn far_metrics_defaults() {
    let start = Instant::now();
    let Some(gpu) = jbm_client::testing::screenshot::gpu_or_skip() else {
        return;
    };
    let mut registry = jbm_core::block::BlockRegistry::new();
    let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
    let nav = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let spawn = DVec3::from(nav.spawn());
    let mountain = mountain_top(&nav, 2048);
    let defaults = ServerConfig {
        gen_threads: 4,
        ..ServerConfig::default()
    };
    assert_eq!(
        (
            defaults.view_radius,
            defaults.lod_view_radius,
            defaults.lod_levels
        ),
        (16, 16, 5),
        "the task 2.13 E defaults"
    );

    for (site, tag) in [(spawn, "spawn"), (mountain, "mountain")] {
        measure_site(
            &gpu,
            &nav,
            site,
            defaults.clone(),
            tag,
            start + TEST_TIMEOUT,
        );
    }
    println!(
        "\nfar_metrics (defaults) done in {:.0} s",
        start.elapsed().as_secs_f64()
    );
}

/// Task 2.13 E's deliverable shots: spawn and the mountain top, LOD tint
/// on, at the pre-E defaults (`view_radius` 8, `lod_view_radius` 8,
/// `lod_levels` 6) and at the shipped ones (16 / 16 / 5) — LOD0 visibly
/// reaches about 4x the area after. Saved under `target/2.13-shots/`.
/// Ignored (slow, GPU): run with
/// `cargo test -p jbm-game --test far_metrics far_shots -- --ignored --nocapture`.
#[test]
#[ignore = "slow real-server measurement; run with --ignored"]
fn far_shots() {
    let start = Instant::now();
    let Some(gpu) = jbm_client::testing::screenshot::gpu_or_skip() else {
        return;
    };
    let mut registry = jbm_core::block::BlockRegistry::new();
    let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
    let nav = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let spawn = DVec3::from(nav.spawn());
    let mountain = mountain_top(&nav, 2048);
    let before = ServerConfig {
        gen_threads: 4,
        view_radius: 8,
        lod_view_radius: 8,
        lod_levels: 6,
        ..ServerConfig::default()
    };
    let after = ServerConfig {
        gen_threads: 4,
        ..ServerConfig::default()
    };
    let shot_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target")
        .join("2.13-shots");
    std::fs::create_dir_all(&shot_dir).expect("shot dir");

    for (site, tag) in [(spawn, "spawn"), (mountain, "mountain")] {
        for (config, when) in [(before.clone(), "before"), (after.clone(), "after")] {
            let mut server = Server::with_config(config);
            let (mut client, server_end) = jbm_core::link::in_process_pair();
            server.add_connection(server_end);
            client
                .send(ClientMsg::Hello {
                    protocol_version: PROTOCOL_VERSION,
                    name: "far_shots".to_owned(),
                })
                .unwrap();
            let mut win = MetricsClient::new(&gpu);
            join_and_teleport(
                &mut server,
                &mut win,
                &mut client,
                &gpu,
                site,
                start + TEST_TIMEOUT,
            );

            const EYE: f64 = 1.62;
            let mut camera = Camera::new(
                site + DVec3::new(0.0, EYE, 0.0),
                70.0_f32.to_radians(),
                WIDTH as f32 / HEIGHT as f32,
            );
            let spawn_pos = DVec3::from(nav.spawn());
            let to_spawn = spawn_pos - camera.position;
            camera.set_yaw(f32::atan2(to_spawn.x as f32, -to_spawn.z as f32));
            camera.set_pitch(-15.0_f32.to_radians());

            settle(
                &mut server,
                &mut win,
                &mut client,
                &gpu,
                &camera,
                start + TEST_TIMEOUT,
            );
            win.renderer.set_fog(true);
            win.renderer.set_lod_tint(true);
            for _ in 0..WARMUP_FRAMES {
                win.render_frame(&gpu, &camera);
            }
            win.render_frame(&gpu, &camera);
            let image = win.target.read_rgba(&gpu);
            let path = shot_dir.join(format!("{tag}_{when}.png"));
            image
                .save(&path)
                .unwrap_or_else(|err| panic!("save {}: {err}", path.display()));
            println!("saved {}", path.display());
        }
    }
}

/// The look review's cameras: (tag, feet position, extra height, yaw
/// degrees, pitch degrees) — spawn at eye level, spawn from 60 blocks up,
/// the mountain top, the coast looking out to sea, and 300 blocks up
/// looking down at 30 degrees. Shared by `look_shots` and `perf_views`.
fn look_sites(nav: &Generator) -> [(&'static str, DVec3, f64, f32, f32); 5] {
    let side = i64::from(CUBE_SIZE);
    let spawn = DVec3::from(nav.spawn());
    let mountain = mountain_top(nav, 2048);
    // First all-ocean column along +x from spawn, camera 3 columns inland.
    let ocean_cx = (0..4000)
        .find(|&cx| nav.column_surface_bounds(cx, 0).1 < jbm_server::terrain::SEA_LEVEL)
        .expect("an ocean column");
    let coast_x = i64::from(ocean_cx - 3) * side;
    let coast = DVec3::new(
        coast_x as f64,
        (nav.surface_height(coast_x, 0)
            .max(jbm_server::terrain::SEA_LEVEL)
            + 1) as f64,
        0.0,
    );
    [
        ("spawn_eye", spawn, 0.0, 0.0, -5.0),
        ("spawn_60up", spawn, 60.0, 0.0, -15.0),
        ("mountain", mountain, 0.0, 180.0, -10.0),
        ("coast", coast, 4.0, 90.0, -5.0),
        ("high_300", spawn, 300.0, 45.0, -30.0),
    ]
}

/// The look review: the shipped defaults, fog on, tint off and on, from
/// the cameras a player actually sees — spawn at eye level, spawn from 60
/// blocks up, the mountain top, the coast looking out to sea, and 300
/// blocks up looking down at 30 degrees. Saved under `target/look-shots/`.
/// Run with
/// `cargo test --release -p jbm-game --test far_metrics look_shots -- --ignored --nocapture`;
/// `JBM_LOOK_SHOTS=coast,mountain` limits the run to the named shots.
#[test]
#[ignore = "slow real-server screenshots; run with --ignored"]
fn look_shots() {
    let start = Instant::now();
    let Some(gpu) = jbm_client::testing::screenshot::gpu_or_skip() else {
        return;
    };
    let mut registry = jbm_core::block::BlockRegistry::new();
    let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
    let nav = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let shot_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target")
        .join("look-shots");
    std::fs::create_dir_all(&shot_dir).expect("shot dir");

    // `JBM_LOOK_SHOTS=coast,mountain` renders a subset (look tuning).
    let only = std::env::var("JBM_LOOK_SHOTS").ok();
    for (tag, feet, up, yaw, pitch) in look_sites(&nav) {
        if only
            .as_deref()
            .is_some_and(|list| !list.split(',').any(|t| t.trim() == tag))
        {
            continue;
        }
        let mut server = Server::with_config(ServerConfig {
            gen_threads: 6,
            ..ServerConfig::default()
        });
        let (mut client, server_end) = jbm_core::link::in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "look".to_owned(),
            })
            .unwrap();
        let mut win = MetricsClient::new(&gpu);
        let site = feet + DVec3::new(0.0, up, 0.0);
        join_and_teleport(
            &mut server,
            &mut win,
            &mut client,
            &gpu,
            site,
            start + TEST_TIMEOUT,
        );
        let mut camera = Camera::new(
            site + DVec3::new(0.0, 1.62, 0.0),
            70.0_f32.to_radians(),
            WIDTH as f32 / HEIGHT as f32,
        );
        camera.set_yaw(yaw.to_radians());
        camera.set_pitch(pitch.to_radians());
        settle(
            &mut server,
            &mut win,
            &mut client,
            &gpu,
            &camera,
            start + TEST_TIMEOUT,
        );
        win.renderer.set_fog(true);
        for tint in [false, true] {
            win.renderer.set_lod_tint(tint);
            for _ in 0..WARMUP_FRAMES {
                win.render_frame(&gpu, &camera);
            }
            win.render_frame(&gpu, &camera);
            let image = win.target.read_rgba(&gpu);
            let name = if tint {
                format!("{tag}_tint.png")
            } else {
                format!("{tag}.png")
            };
            let path = shot_dir.join(name);
            image.save(&path).expect("save shot");
            println!(
                "saved {} ({:.0} s)",
                path.display(),
                start.elapsed().as_secs_f64()
            );
        }
    }
}

/// The user's view from the task 2.16 report (an M5 Pro, the game
/// maximized on a 5K display, 41 fps): hovering 233 blocks up over the
/// hills 4 km from spawn, looking west-south-west and down. The HUD's XYZ
/// is the eye; yaw and pitch in degrees.
const USER_VIEW_EYE: DVec3 = DVec3::new(57.60, 232.67, -4087.75);
const USER_VIEW_YAW: f32 = 257.7;
const USER_VIEW_PITCH: f32 = -18.8;

/// Frames timed per view and resolution, after the warm-up.
const PERF_FRAMES: usize = 60;
const PERF_WARMUP: usize = 10;

/// The p50 and p95 of `samples` (sorted in place).
fn p50_p95(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    let at = |q: f64| samples[(q * (samples.len() - 1) as f64).round() as usize];
    (at(0.5), at(0.95))
}

/// `JBM_PERF_RES=2560x1440,5120x2880@1,5120x2880` (the default): the
/// target sizes, each optionally at a fixed render scale (`@<scale>`;
/// without one the settings' scale applies, auto by default).
fn perf_resolutions() -> Vec<(u32, u32, Option<f32>)> {
    let list =
        std::env::var("JBM_PERF_RES").unwrap_or_else(|_| "2560x1440,5120x2880@1,5120x2880".into());
    list.split(',')
        .map(|item| {
            let (size, scale) = match item.trim().split_once('@') {
                Some((size, scale)) => (size, Some(scale.parse().expect("render scale"))),
                None => (item.trim(), None),
            };
            let (w, h) = size.split_once('x').expect("WxH");
            (w.parse().expect("width"), h.parse().expect("height"), scale)
        })
        .collect()
}

/// Task 2.16's perf harness: the user's view and the five look-review
/// cameras on the default world, settled with the shipped server
/// defaults, rendered offscreen without vsync at each resolution of
/// `JBM_PERF_RES` (default 2560×1440, and 5120×2880 natively and at the
/// default render scale). Per view and
/// resolution it prints the CPU time of `Renderer::render` (encode and
/// submit; the frame's own CPU work) and the GPU time from pass
/// timestamps (the frame span and each pass), p50 / p95 over
/// `PERF_FRAMES` frames, plus the frame's faces, commands and draw calls.
/// Rendering settings come from the environment like the game's
/// (`JBM_MSAA`, `JBM_GPU_CULL`, `JBM_NO_CLOUDS`, …), so a lever can be
/// measured without a rebuild; `JBM_PERF_VIEWS=user,coast` limits the
/// views, and `JBM_PERF_SHOTS=1` saves each view and size's base frame
/// under `target/perf-shots/` (to judge the render scale). Run with
/// `cargo test --release -p jbm-game --test far_metrics perf_views -- --ignored --nocapture`.
#[test]
#[ignore = "slow real-server measurement; run with --ignored"]
fn perf_views() {
    let start = Instant::now();
    let Some(gpu) = jbm_client::testing::screenshot::gpu_or_skip() else {
        return;
    };
    let mut registry = jbm_core::block::BlockRegistry::new();
    let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
    let nav = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let settings = RenderSettings::from_env();
    let info = gpu.adapter.get_info();
    println!(
        "perf_views: {} ({:?}), msaa {}, timestamps {}",
        info.name,
        info.backend,
        settings.msaa_samples,
        gpu.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    );

    // (tag, eye, yaw degrees, pitch degrees)
    let mut views = vec![("user", USER_VIEW_EYE, USER_VIEW_YAW, USER_VIEW_PITCH)];
    for (tag, feet, up, yaw, pitch) in look_sites(&nav) {
        views.push((tag, feet + DVec3::new(0.0, up + 1.62, 0.0), yaw, pitch));
    }
    let only = std::env::var("JBM_PERF_VIEWS").ok();
    let resolutions = perf_resolutions();
    let variants = perf_variants();
    let mut table = Vec::new();
    for (tag, eye, yaw, pitch) in views {
        if only
            .as_deref()
            .is_some_and(|list| !list.split(',').any(|t| t.trim() == tag))
        {
            continue;
        }
        let mut server = Server::with_config(ServerConfig {
            gen_threads: 6,
            ..ServerConfig::default()
        });
        let (mut client, server_end) = jbm_core::link::in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "perf".to_owned(),
            })
            .unwrap();
        let mut win = MetricsClient::with_settings(&gpu, settings);
        let feet = eye - DVec3::new(0.0, 1.62, 0.0);
        join_and_teleport(
            &mut server,
            &mut win,
            &mut client,
            &gpu,
            feet,
            start + TEST_TIMEOUT,
        );
        let mut camera = Camera::new(eye, 70.0_f32.to_radians(), WIDTH as f32 / HEIGHT as f32);
        camera.set_yaw(yaw.to_radians());
        camera.set_pitch(pitch.to_radians());
        let settle_secs = settle(
            &mut server,
            &mut win,
            &mut client,
            &gpu,
            &camera,
            start + TEST_TIMEOUT,
        );
        println!("\n{tag}: settled in {settle_secs:.1} s");
        let levels = win
            .terrain
            .level_stats()
            .iter()
            .map(|l| {
                format!(
                    "L{} {}/{} {}k",
                    l.level,
                    l.resident,
                    l.drawn,
                    l.faces / 1000
                )
            })
            .collect::<Vec<_>>()
            .join("  ");
        println!("  resident/drawn cubes, resident faces: {levels}");

        for &(width, height, scale) in &resolutions {
            let base = RenderSettings {
                render_scale: scale.unwrap_or(settings.render_scale),
                ..settings
            };
            win.renderer.set_settings(base);
            win.resize(&gpu, width, height);
            camera.aspect = width as f32 / height as f32;
            let size = match scale {
                Some(scale) => format!("{width}x{height}@{scale}"),
                None => format!("{width}x{height}"),
            };
            let (scene_width, scene_height) = win.renderer.scene_size();
            // Variants take turns frame by frame, so background load on
            // the machine hits them alike.
            let mut samples: Vec<VariantSamples> = variants
                .iter()
                .map(|&(name, _)| VariantSamples::new(name))
                .collect();
            for frame in 0..(PERF_WARMUP + PERF_FRAMES) * variants.len() {
                let index = frame % variants.len();
                win.renderer.set_settings((variants[index].1)(base));
                let begin = Instant::now();
                win.render_frame(&gpu, &camera);
                let cpu_ms = begin.elapsed().as_secs_f64() * 1000.0;
                let timings = win.renderer.read_gpu_timings_blocking(&gpu);
                if frame >= PERF_WARMUP * variants.len() {
                    samples[index].push(cpu_ms, timings);
                }
            }
            win.renderer.set_settings(base);
            win.render_frame(&gpu, &camera);
            if std::env::var_os("JBM_PERF_SHOTS").is_some_and(|v| v == "1") {
                let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../target/perf-shots");
                std::fs::create_dir_all(&dir).expect("shot dir");
                let path = dir.join(format!("{tag}_{size}.png"));
                win.target.read_rgba(&gpu).save(&path).expect("save shot");
                println!("saved {}", path.display());
            }
            for (sample, &(_, variant)) in samples.iter_mut().zip(&variants) {
                // Each variant's own counts: one more frame with it.
                win.renderer.set_settings(variant(base));
                win.render_frame(&gpu, &camera);
                let cull = win.terrain.read_cull_stats_blocking(&gpu);
                let stats = win.terrain.stats();
                let summary = sample.summary();
                let line = format!(
                    "{tag:<11}{size:>14} {:>10} {:<10}{}  faces {:>8}  cmds {:>6}  draws {:>4}  occluded {:>5}",
                    format!("{scene_width}x{scene_height}"),
                    sample.name,
                    summary,
                    cull.faces,
                    cull.commands,
                    stats.draw_calls,
                    cull.occluded,
                );
                println!("{line}");
                table.push(line);
            }
        }
    }
    println!(
        "\nperf_views ({}, msaa {}): ms over {PERF_FRAMES} frames per variant",
        win_path_note(),
        settings.msaa_samples
    );
    for line in &table {
        println!("{line}");
    }
    println!("perf_views done in {:.0} s", start.elapsed().as_secs_f64());
}

/// A settings variant of the perf harness: a name and the change it
/// makes to the base settings.
type Variant = (&'static str, fn(RenderSettings) -> RenderSettings);

/// The variants `JBM_PERF_VARIANTS` names (comma-separated; `base` alone
/// by default): live-switchable levers measured against the base in the
/// same run, frame by frame.
fn perf_variants() -> Vec<Variant> {
    let all: [Variant; 9] = [
        ("base", |s| s),
        ("no_clouds", |s| RenderSettings { clouds: false, ..s }),
        ("no_water", |s| RenderSettings { water: false, ..s }),
        ("no_sun", |s| RenderSettings {
            sun_lighting: false,
            ..s
        }),
        ("no_fog", |s| RenderSettings { fog: false, ..s }),
        // The cheapest terrain fragment shader there is: edges only.
        ("flat_fs", |s| RenderSettings {
            wireframe_only: true,
            ..s
        }),
        ("gpu_cull", |s| RenderSettings {
            prefer_gpu_cull: true,
            ..s
        }),
        ("cpu_cull", |s| RenderSettings {
            prefer_gpu_cull: false,
            force_cpu_cull: true,
            ..s
        }),
        // Task 2.13's split radius: cubes beyond 4 cubes draw all their
        // directions in one command (fewer commands, their back faces
        // reach the vertex shader).
        ("split_4", |s| RenderSettings {
            direction_split_radius: 4,
            ..s
        }),
    ];
    let list = std::env::var("JBM_PERF_VARIANTS").unwrap_or_else(|_| "base".into());
    list.split(',')
        .map(|name| {
            *all.iter()
                .find(|(n, _)| *n == name.trim())
                .unwrap_or_else(|| panic!("unknown perf variant {name}"))
        })
        .collect()
}

/// One variant's samples: CPU ms, GPU frame ms and GPU ms per pass.
struct VariantSamples {
    name: &'static str,
    cpu: Vec<f64>,
    gpu: Vec<f64>,
    passes: Vec<(&'static str, Vec<f64>)>,
}

impl VariantSamples {
    fn new(name: &'static str) -> VariantSamples {
        VariantSamples {
            name,
            cpu: Vec::new(),
            gpu: Vec::new(),
            passes: Vec::new(),
        }
    }

    fn push(&mut self, cpu_ms: f64, timings: Option<GpuTimings>) {
        self.cpu.push(cpu_ms);
        let Some(timings) = timings else {
            return;
        };
        self.gpu.push(timings.frame_ms);
        for (label, ms) in timings.passes {
            match self.passes.iter_mut().find(|(l, _)| *l == label) {
                Some((_, samples)) => samples.push(ms),
                None => self.passes.push((label, vec![ms])),
            }
        }
    }

    /// `cpu p50 / p95  gpu min / p50 / p95  [pass p50, …]`. The GPU min
    /// is the least disturbed by other work on the machine.
    fn summary(&mut self) -> String {
        let (cpu50, cpu95) = p50_p95(&mut self.cpu);
        let (gpu_min, gpu50, gpu95) = if self.gpu.is_empty() {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            let (p50, p95) = p50_p95(&mut self.gpu);
            (self.gpu[0], p50, p95)
        };
        let passes = self
            .passes
            .iter_mut()
            .map(|(label, samples)| format!("{label} {:.2}", p50_p95(samples).0))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "cpu {cpu50:5.2} / {cpu95:5.2}  gpu {gpu_min:6.2} / {gpu50:6.2} / {gpu95:6.2}  [{passes}]"
        )
    }
}

/// The draw path the environment selects, for the table's title.
fn win_path_note() -> &'static str {
    if std::env::var_os("JBM_NO_MDI").is_some_and(|v| v == "1") {
        "fallback"
    } else if std::env::var_os("JBM_GPU_CULL").is_some_and(|v| v == "1") {
        "gpu-cull"
    } else {
        "cpu-cull"
    }
}

/// Task 2.17's seam probe: cracks are pixels where the sky shows through
/// solid terrain. Each view and size of `JBM_PERF_RES` renders twice with
/// the whole sky overridden to two different colors (fog, water shading
/// and clouds off, so no terrain fragment reads the sky); a pixel that
/// differs between the two shows the sky in some sample. The sky itself
/// and the AA edges of silhouettes against it are one big component
/// touching the image border; everything else is a crack. Prints the
/// crack components per view and saves the frame with the cracks painted
/// red (and grown, so a single pixel shows) under `target/seam-probe/`.
/// `JBM_MSAA=1` makes every crack sample a full pixel. Run with
/// `cargo test --release -p jbm-game --test far_metrics seam_probe -- --ignored --nocapture`.
#[test]
#[ignore = "slow real-server measurement; run with --ignored"]
fn seam_probe() {
    let start = Instant::now();
    let Some(gpu) = jbm_client::testing::screenshot::gpu_or_skip() else {
        return;
    };
    let mut registry = jbm_core::block::BlockRegistry::new();
    let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
    let nav = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let settings = RenderSettings {
        fog: false,
        water: false,
        clouds: false,
        ..RenderSettings::from_env()
    };
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/seam-probe");
    std::fs::create_dir_all(&dir).expect("probe dir");

    let mut views = vec![("user", USER_VIEW_EYE, USER_VIEW_YAW, USER_VIEW_PITCH)];
    for (tag, feet, up, yaw, pitch) in look_sites(&nav) {
        views.push((tag, feet + DVec3::new(0.0, up + 1.62, 0.0), yaw, pitch));
    }
    let only = std::env::var("JBM_PERF_VIEWS").ok();
    let mut table = Vec::new();
    for (tag, eye, yaw, pitch) in views {
        if only
            .as_deref()
            .is_some_and(|list| !list.split(',').any(|t| t.trim() == tag))
        {
            continue;
        }
        let mut server = Server::with_config(ServerConfig {
            gen_threads: 6,
            ..ServerConfig::default()
        });
        let (mut client, server_end) = jbm_core::link::in_process_pair();
        server.add_connection(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "seams".to_owned(),
            })
            .unwrap();
        let mut win = MetricsClient::with_settings(&gpu, settings);
        let feet = eye - DVec3::new(0.0, 1.62, 0.0);
        join_and_teleport(
            &mut server,
            &mut win,
            &mut client,
            &gpu,
            feet,
            start + TEST_TIMEOUT,
        );
        let mut camera = Camera::new(eye, 70.0_f32.to_radians(), WIDTH as f32 / HEIGHT as f32);
        camera.set_yaw(yaw.to_radians());
        camera.set_pitch(pitch.to_radians());
        settle(
            &mut server,
            &mut win,
            &mut client,
            &gpu,
            &camera,
            start + TEST_TIMEOUT,
        );
        for (width, height, scale) in perf_resolutions() {
            win.renderer.set_settings(RenderSettings {
                render_scale: scale.unwrap_or(settings.render_scale),
                ..settings
            });
            win.resize(&gpu, width, height);
            camera.aspect = width as f32 / height as f32;
            let mut frames = Vec::new();
            for sky in [[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]] {
                win.renderer.set_sky_colors(Some((sky, sky, sky)));
                for _ in 0..WARMUP_FRAMES {
                    win.render_frame(&gpu, &camera);
                }
                win.render_frame(&gpu, &camera);
                frames.push(win.target.read_rgba(&gpu));
            }
            win.renderer.set_sky_colors(None);
            let cracks = crack_components(&frames[0], &frames[1]);
            let pixels: usize = cracks.iter().map(Vec::len).sum();
            let largest = cracks.iter().map(Vec::len).max().unwrap_or(0);
            let size = match scale {
                Some(scale) => format!("{width}x{height}@{scale}"),
                None => format!("{width}x{height}"),
            };
            let mut shot = frames[1].clone();
            for &(x, y) in cracks.iter().flatten() {
                for dy in -2..=2_i64 {
                    for dx in -2..=2_i64 {
                        let (px, py) = (i64::from(x) + dx, i64::from(y) + dy);
                        if px >= 0 && py >= 0 && px < i64::from(width) && py < i64::from(height) {
                            shot.put_pixel(px as u32, py as u32, image::Rgba([255, 0, 0, 255]));
                        }
                    }
                }
            }
            let path = dir.join(format!("{tag}_{size}.png"));
            shot.save(&path).expect("save probe");
            let line = format!(
                "{tag:<11}{size:>14}  crack components {:>5}  pixels {:>6}  largest {:>4}",
                cracks.len(),
                pixels,
                largest
            );
            println!("{line}  ({})", path.display());
            table.push(line);
        }
    }
    println!("\nseam_probe (msaa {}):", settings.msaa_samples);
    for line in &table {
        println!("{line}");
    }
}

/// The 8-connected components of pixels that differ between `a` and `b`
/// (more than 3 levels in any channel), without the ones touching the
/// image border (the sky and silhouettes against it).
fn crack_components(a: &image::RgbaImage, b: &image::RgbaImage) -> Vec<Vec<(u32, u32)>> {
    let (w, h) = a.dimensions();
    let differs = |x: u32, y: u32| {
        let (p, q) = (a.get_pixel(x, y).0, b.get_pixel(x, y).0);
        (0..3).any(|c| p[c].abs_diff(q[c]) > 3)
    };
    let mut seen = vec![false; (w * h) as usize];
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if seen[(y * w + x) as usize] || !differs(x, y) {
                continue;
            }
            seen[(y * w + x) as usize] = true;
            let mut stack = vec![(x, y)];
            let mut component = Vec::new();
            let mut border = false;
            while let Some((cx, cy)) = stack.pop() {
                component.push((cx, cy));
                border |= cx == 0 || cy == 0 || cx == w - 1 || cy == h - 1;
                for dy in -1..=1_i64 {
                    for dx in -1..=1_i64 {
                        let (nx, ny) = (i64::from(cx) + dx, i64::from(cy) + dy);
                        if nx < 0 || ny < 0 || nx >= i64::from(w) || ny >= i64::from(h) {
                            continue;
                        }
                        let (nx, ny) = (nx as u32, ny as u32);
                        let index = (ny * w + nx) as usize;
                        if !seen[index] && differs(nx, ny) {
                            seen[index] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            if !border {
                out.push(component);
            }
        }
    }
    out
}
