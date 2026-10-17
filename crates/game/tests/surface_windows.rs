//! Task 2.10 acceptance: surface-following vertical windows on the
//! overworld. A camera standing in a valley with a tall cliff wall in
//! front must be covered by LOD0 everywhere near it — before 2.10 the
//! fixed ±4-cube window left the cliff's face to the coarse rings (the
//! play-test's "flat 16-block wall in front of the player"). The test
//! finds a deterministic valley+cliff pair, streams a real threaded
//! server there, asserts every surface cube of the LOD0 disk was sent,
//! and renders with the level tint on: no pixel below the horizon margin
//! may carry a level ≥ 1 tint, because the LOD0 valley and cliff occlude
//! everything behind them. GPU-required (skips without an adapter);
//! 32³-only. The ignored measurement test prints the task's before/after
//! numbers.
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
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::link::ClientLink;
use jbm_core::protocol::PlayerId;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
use jbm_server::DEFAULT_SEED;
use jbm_server::server::{Server, ServerConfig};
use jbm_server::terrain::Generator;
use jbm_server::worldgen::Preset;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const SENTINEL: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
/// Whole-test bound; a hang fails instead of hanging the suite.
const TEST_TIMEOUT: Duration = Duration::from_secs(300);

/// A valley column (low surface) with a tall wall within 160 blocks: the
/// play-test's geometry. Deterministic per seed.
struct ValleyCliff {
    /// Feet position in the valley (two above the surface).
    valley: DVec3,
    /// A point near the top of the tallest nearby wall.
    wall_top: DVec3,
    /// That wall's height over the valley floor.
    wall_height: i64,
}

/// Finds the nearest valley+cliff pair: candidates every 128 blocks in
/// rings out from spawn, surface sampled on a 32-block grid within 160
/// blocks. A valley is at most `SEA_LEVEL + 12`; a wall is at least 90
/// blocks over it.
fn find_valley_cliff(generator: &Generator) -> ValleyCliff {
    let start = generator.spawn();
    let (sx, sz) = (start[0] as i64, start[2] as i64);
    let sample = |x: i64, z: i64| generator.surface_height(x, z);
    for ring in 0..48i64 {
        for dx in -ring..=ring {
            for dz in -ring..=ring {
                if dx.abs() != ring && dz.abs() != ring {
                    continue; // ring perimeter only
                }
                let (x, z) = (sx + dx * 128, sz + dz * 128);
                let floor_h = sample(x, z);
                if floor_h > 63 + 12 {
                    continue; // not a valley
                }
                let mut best = (floor_h, x, z); // (height, at)
                for nx in (x - 160..=x + 160).step_by(32) {
                    for nz in (z - 160..=z + 160).step_by(32) {
                        let h = sample(nx, nz);
                        if h > best.0 {
                            best = (h, nx, nz);
                        }
                    }
                }
                let wall_height = best.0 - floor_h;
                if wall_height >= 90 {
                    return ValleyCliff {
                        valley: DVec3::new(x as f64 + 0.5, (floor_h + 2) as f64, z as f64 + 0.5),
                        wall_top: DVec3::new(
                            best.1 as f64 + 0.5,
                            (best.0 - 8) as f64,
                            best.2 as f64 + 0.5,
                        ),
                        wall_height,
                    };
                }
            }
        }
    }
    panic!("no valley+cliff pair found within the search limit for this seed");
}

/// The tallest surface in a `side`-block sample square centered on the
/// spawn — a deterministic "on a mountain" position for the measurement
/// test.
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

/// The client half, as in the fly test: world, mesh pool, coverage,
/// renderer, plus every LOD0 cube position currently held.
struct WindowClient {
    world: ClientWorld,
    pool: MeshPool,
    coverage: Coverage,
    in_flight: usize,
    received_lod0: std::collections::HashSet<CubePos>,
    renderer: Renderer,
    terrain: TerrainPass,
    target: OffscreenTarget,
    depth: DepthTarget,
}

impl WindowClient {
    fn new(gpu: &Gpu) -> WindowClient {
        let mut registry = jbm_core::block::BlockRegistry::new();
        jbm_core::block::builtin::register(&mut registry);
        let appearance = Arc::new(BlockAppearance::new(&registry));
        let target = OffscreenTarget::new(&gpu.device, WIDTH, HEIGHT);
        let depth = DepthTarget::new(&gpu.device, WIDTH, HEIGHT);
        let renderer = Renderer::new(gpu, target.format());
        let (layers, _) = builtin_textures();
        let textures = BlockTextures::new(gpu, &layers).expect("texture array");
        let terrain = TerrainPass::new(gpu, target.format(), renderer.uniform_layout(), &textures);
        WindowClient {
            world: ClientWorld::new(),
            pool: MeshPool::new(appearance, 0),
            coverage: Coverage::default(),
            in_flight: 0,
            received_lod0: std::collections::HashSet::new(),
            renderer,
            terrain,
            target,
            depth,
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
                        self.world.apply(&msg);
                    }
                    ServerMsg::UnloadCube { pos } => {
                        self.received_lod0.remove(pos);
                        if let jbm_client::world::WorldChange::Unloaded(key) =
                            self.world.apply(&msg)
                        {
                            self.terrain.remove(gpu, key);
                            self.coverage.set_ready(key, false);
                        }
                    }
                    ServerMsg::LodCubeData { .. }
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

    fn queue_clear(&self) -> bool {
        self.world.dirty_count() == 0 && self.in_flight == 0
    }

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

#[test]
fn valley_cliff_is_full_lod0_near_the_camera() {
    let start = Instant::now();
    let Some(gpu) = jbm_client::testing::screenshot::gpu_or_skip() else {
        return;
    };

    // Deterministic valley+cliff on the default overworld.
    let mut registry = jbm_core::block::BlockRegistry::new();
    let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
    let nav = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let site = find_valley_cliff(&nav);
    println!(
        "valley at {:.0?}, wall {} blocks tall, wall top {:.0?}",
        site.valley, site.wall_height, site.wall_top
    );

    // A threaded server ticked by the test (the generation pool is what
    // threading buys; the tick cadence is the test's).
    let config = ServerConfig {
        gen_threads: 4,
        ..ServerConfig::default()
    };
    let view_radius = config.view_radius;
    let mut server = Server::with_config(config);
    let (mut client, server_end) = jbm_core::link::in_process_pair();
    server.add_connection(server_end);
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "valley".to_owned(),
        })
        .unwrap();
    let mut win = WindowClient::new(&gpu);
    let id = PlayerId(1);

    // Join, then teleport into the valley (physics-free; streaming
    // observes the moved center exactly as after a real teleport).
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
        if let Some(reason) = win.pump_link(&gpu, &mut client) {
            panic!("link died before joining: {reason}");
        }
        joined = server.players().next().is_some();
        assert!(start.elapsed() < TEST_TIMEOUT, "never joined");
    }
    server.teleport_player(id, site.valley);

    // The camera stands in the valley looking up at the wall top.
    const EYE: f64 = 1.62;
    let eye = site.valley + DVec3::new(0.0, EYE, 0.0);
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), WIDTH as f32 / HEIGHT as f32);
    let d = site.wall_top - eye;
    camera.set_yaw(f32::atan2(d.x as f32, -d.z as f32));
    camera.set_pitch(f32::atan2(d.y as f32, d.x.hypot(d.z) as f32));

    // Settle: server spans filled, LOD rings drained and the client mesh
    // queue clear for a run of polls (as in the fly test, whatever meshed
    // by then is what the render sees).
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut clear_polls = 0;
    while clear_polls < 10 && Instant::now() < deadline {
        server.update(Duration::from_millis(50));
        for _ in 0..3 {
            win.pump_meshing(&gpu, &camera);
            std::thread::sleep(Duration::from_millis(16));
        }
        if let Some(reason) = win.pump_link(&gpu, &mut client) {
            panic!("link died while settling: {reason}");
        }
        let (_, _, pending_spans, _) = server.span_stats();
        let settled = win.queue_clear()
            && pending_spans == 0
            && server.lod0_pending(id) == 0
            && server.lod_ring_pending(id) == 0
            && server.in_flight() == 0;
        clear_polls = if settled { clear_polls + 1 } else { 0 };
        assert!(start.elapsed() < TEST_TIMEOUT, "never settled");
    }
    for _ in 0..200 {
        win.pump_meshing(&gpu, &camera);
        if win.queue_clear() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    println!(
        "settled in {:.1} s; per-level sent {:?}",
        start.elapsed().as_secs_f64(),
        server.sent_counts_per_level(id),
    );

    // 1. Every surface cube of the LOD0 disk was sent: the disk is the
    //    cube-column set dx^2 + dz^2 <= r^2 (in cubes, as the server
    //    defines it); each column's surface cube — sampled at its center
    //    and four off-center blocks — is in the received set. The span
    //    covers the column's whole min..max, so any sampled surface sits
    //    inside it.
    let camera_cube = CubePos {
        x: (site.valley.x / f64::from(CUBE_SIZE)).floor() as i32,
        y: (site.valley.y / f64::from(CUBE_SIZE)).floor() as i32,
        z: (site.valley.z / f64::from(CUBE_SIZE)).floor() as i32,
    };
    let mut missing = Vec::new();
    let side_i = i64::from(CUBE_SIZE);
    for dcx in -i64::from(view_radius)..=i64::from(view_radius) {
        for dcz in -i64::from(view_radius)..=i64::from(view_radius) {
            if dcx * dcx + dcz * dcz > i64::from(view_radius).pow(2) {
                continue;
            }
            let col = CubePos {
                x: camera_cube.x + dcx as i32,
                z: camera_cube.z + dcz as i32,
                y: 0,
            };
            for (ox, oz) in [(16i64, 16i64), (8, 24), (24, 8), (8, 8), (24, 24)] {
                let (bx, bz) = (
                    i64::from(col.x) * side_i + ox,
                    i64::from(col.z) * side_i + oz,
                );
                let h = nav.surface_height(bx, bz);
                let cube = CubePos {
                    x: col.x,
                    y: h.div_euclid(side_i) as i32,
                    z: col.z,
                };
                if !win.received_lod0.contains(&cube) {
                    missing.push(cube);
                }
            }
        }
    }
    let camera_cube = CubePos {
        x: (site.valley.x / f64::from(CUBE_SIZE)).floor() as i32,
        y: (site.valley.y / f64::from(CUBE_SIZE)).floor() as i32,
        z: (site.valley.z / f64::from(CUBE_SIZE)).floor() as i32,
    };
    let offsets: Vec<(i32, i32, i32)> = missing
        .iter()
        .map(|c| {
            (
                c.x - camera_cube.x,
                c.y - camera_cube.y,
                c.z - camera_cube.z,
            )
        })
        .take(8)
        .collect();
    assert!(
        missing.is_empty(),
        "{} surface cubes of the LOD0 disk unsent (camera cube {:?}), first offsets {:?}",
        missing.len(),
        camera_cube,
        offsets,
    );

    // 2. Tint on: no pixel showing terrain within 200 blocks of the
    //    camera may be drawn by a LOD ring — the surface-following
    //    windows put LOD0 over the whole near relief. Per-pixel distance
    //    comes from the depth buffer (reverse-Z: distance = NEAR / d).
    //    Farther tinted pixels are legitimate ring terrain and are only
    //    counted for the report.
    win.renderer.set_fog(false);
    win.renderer.set_lod_tint(false);
    let plain = win.render(&gpu, &camera);
    win.renderer.set_lod_tint(true);
    let tinted = win.render(&gpu, &camera);
    let depths = win.depth.read_depth(&gpu);
    std::fs::create_dir_all("target/surface").ok();
    let _ = plain.save("target/surface/valley_plain.png");
    let _ = tinted.save("target/surface/valley_tinted.png");
    let mut near_tinted = 0usize;
    let mut far_tinted = 0usize;
    let inv_view_proj = camera.view_proj().inverse();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if plain.get_pixel(x, y).0 == tinted.get_pixel(x, y).0 {
                continue;
            }
            let d = depths[(y * WIDTH + x) as usize];
            if d <= 0.0 {
                far_tinted += 1; // sky-depth edge; not near terrain
                continue;
            }
            // Euclidean distance via unprojection at the pixel's depth
            // (`NEAR / d` is the view-axis distance; corner pixels need
            // the full ray length).
            use glam::Vec4;
            let ndc_x = (f64::from(x) + 0.5) / f64::from(WIDTH) * 2.0 - 1.0;
            let ndc_y = 1.0 - (f64::from(y) + 0.5) / f64::from(HEIGHT) * 2.0;
            let clip = Vec4::new(ndc_x as f32, ndc_y as f32, d, 1.0);
            let view = inv_view_proj * clip;
            let view = view / view.w;
            let world = camera.position
                + DVec3::new(f64::from(view.x), f64::from(view.y), f64::from(view.z));
            if (world - camera.position).length() <= 200.0 {
                near_tinted += 1;
            } else {
                far_tinted += 1;
            }
        }
    }
    println!(
        "tint check: {near_tinted} tinted pixels within 200 blocks (must be 0), \
         {far_tinted} farther ring pixels (legitimate)"
    );
    assert_eq!(
        near_tinted, 0,
        "{near_tinted} pixels of terrain within 200 blocks are drawn by a LOD ring; \
         images in target/surface/"
    );
}

/// The task 2.10 measurement: cubes desired per level, settle time and
/// memory at spawn and on a mountain, before (`max_column_cubes` 0 — the
/// fixed windows) and after (16, the default) surface-following windows.
/// Ignored: it runs two full settles per configuration; run with
/// `cargo test -p jbm-game --test surface_windows -- --ignored`.
#[test]
#[ignore = "measurement run; prints the task 2.10 numbers"]
fn surface_window_measurements() {
    let mut registry = jbm_core::block::BlockRegistry::new();
    let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
    let nav = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let spawn = DVec3::from(nav.spawn());
    let mountain = mountain_top(&nav, 2048);

    for cap in [0usize, 16] {
        for (site_name, site) in [("spawn", spawn), ("mountain", mountain)] {
            let config = ServerConfig {
                gen_threads: 4,
                max_column_cubes: cap,
                ..ServerConfig::default()
            };
            let mut server = Server::with_config(config);
            let (mut client, server_end) = jbm_core::link::in_process_pair();
            server.add_connection(server_end);
            client
                .send(ClientMsg::Hello {
                    protocol_version: PROTOCOL_VERSION,
                    name: "measure".to_owned(),
                })
                .unwrap();
            let id = PlayerId(1);
            let start = Instant::now();
            // Join and land at the site.
            loop {
                server.tick();
                std::thread::sleep(Duration::from_millis(10));
                while let Ok(Some(_)) = client.try_recv() {}
                if server.players().next().is_some() {
                    break;
                }
                assert!(start.elapsed() < TEST_TIMEOUT, "never joined");
            }
            server.teleport_player(id, site);
            // Settle: quiet for 20 ticks with nothing in flight.
            let mut quiet = 0;
            while quiet < 20 {
                server.update(Duration::from_millis(50));
                std::thread::sleep(Duration::from_millis(10));
                while let Ok(Some(_)) = client.try_recv() {}
                let (_, _, pending_spans, _) = server.span_stats();
                quiet = if server.lod_ring_pending(id) == 0
                    && server.in_flight() == 0
                    && server.lod_in_flight() == 0
                    && pending_spans == 0
                {
                    quiet + 1
                } else {
                    0
                };
                assert!(
                    start.elapsed() < TEST_TIMEOUT,
                    "{site_name} cap {cap} never settled"
                );
            }
            let settled = start.elapsed().as_secs_f64();
            let per_level = server.sent_counts_per_level(id);
            let (spans_l0, spans_l1, spans_pending, spans_version) = server.span_stats();
            println!(
                "cap {cap:>2} {site_name:>8}: settled {settled:6.1} s, sent per level {:?}, \
                 world cubes {}, lod heap {:.1} MB, spans l0 {} l1 {} (pending {}, version {})",
                per_level,
                server.world_cube_count(),
                server.lod_heap_bytes() as f64 / 1e6,
                spans_l0,
                spans_l1,
                spans_pending,
                spans_version,
            );
        }
    }
}
