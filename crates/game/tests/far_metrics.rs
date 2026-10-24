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
//! timestamp queries are not enabled on the device, so there is no
//! separate GPU-time number). Two cameras — spawn and a deterministic
//! mountain top. The baseline run pins the pre-2.13 rings
//! (`view_radius` 8 and 16, `lod_view_radius` 8, `lod_levels` 6); the
//! task 2.13 E `far_metrics_defaults` run measures the shipped
//! defaults, and `far_shots` saves the tinted before/after PNGs.
//! Everything the task's later steps are judged against comes from here.
#![cfg(not(feature = "cube16"))]

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::gpu::Gpu;
use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::mesh::{self, MeshInput, neighbor_index};
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::terrain::TerrainPass;
use jbm_client::render::texture_array::BlockTextures;
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
        let mut registry = jbm_core::block::BlockRegistry::new();
        jbm_core::block::builtin::register(&mut registry);
        let appearance = Arc::new(BlockAppearance::new(&registry));
        let target = OffscreenTarget::new(&gpu.device, WIDTH, HEIGHT);
        let depth = DepthTarget::new(&gpu.device, WIDTH, HEIGHT);
        let mut renderer = Renderer::new(gpu, target.format());
        renderer.set_viewport(WIDTH, HEIGHT);
        let textures = BlockTextures::builtin(gpu).expect("texture array");
        let terrain = TerrainPass::new(gpu, target.format(), renderer.uniform_layout(), &textures);
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

    // Frame CPU cost: encode time and GPU wait over FRAMES frames. GPU
    // timestamp queries are not enabled on the device, so there is no
    // separate GPU-time number.
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
    row("GPU time", "n/a (no timestamp queries)");

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
    let side = i64::from(CUBE_SIZE);
    let spawn = DVec3::from(nav.spawn());
    let mountain = mountain_top(&nav, 2048);
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
    let shot_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target")
        .join("look-shots");
    std::fs::create_dir_all(&shot_dir).expect("shot dir");

    // (tag, feet position, extra height, yaw degrees, pitch degrees)
    let sites = [
        ("spawn_eye", spawn, 0.0, 0.0, -5.0),
        ("spawn_60up", spawn, 60.0, 0.0, -15.0),
        ("mountain", mountain, 0.0, 180.0, -10.0),
        ("coast", coast, 4.0, 90.0, -5.0),
        ("high_300", spawn, 300.0, 45.0, -30.0),
    ];
    // `JBM_LOOK_SHOTS=coast,mountain` renders a subset (look tuning).
    let only = std::env::var("JBM_LOOK_SHOTS").ok();
    for (tag, feet, up, yaw, pitch) in sites {
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
        camera.set_yaw((yaw as f32).to_radians());
        camera.set_pitch((pitch as f32).to_radians());
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
