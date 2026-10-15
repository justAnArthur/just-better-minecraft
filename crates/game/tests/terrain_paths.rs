//! Draw-path tests (tasks 2.2 and 2.5) on the real streamed scenes: the
//! gpu-cull, cpu-cull and fallback paths must render pixel-identical images
//! (exact, not the golden tolerance), the two-phase GPU culling must never
//! let a cube pop in across a moving camera, a camera inside a carved
//! closed room must leave most of the frustum occlusion-culled, and the
//! open-surface view must stay nearly uncullled. GPU tests, skipped
//! without an adapter like the other screenshot tests. 32³-only: the
//! streamed shape and cameras mirror `terrain_screenshot.rs` /
//! `edit_screenshot.rs`.
#![cfg(not(feature = "cube16"))]

use std::ops::Range;

use glam::DVec3;
use image::RgbaImage;
use jbm_client::camera::Camera;
use jbm_client::mesh::appearance::{BlockAppearance, builtin_textures};
use jbm_client::mesh::greedy;
use jbm_client::render::chunks::{Frustum, TerrainPass, TerrainPath};
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::{assert_matches_golden, gpu_or_skip};
use jbm_client::world::ClientWorld;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
use jbm_core::coords::{BlockPos, CUBE_SIZE};
use jbm_core::protocol::{ClientMsg, ServerMsg};
use jbm_server::terrain::Generator;
use jbm_server::{DEFAULT_SEED, ServerConfig};
use jbm_testkit::HeadlessServer;

/// Cubes in the streaming shape for R = 3, V = 2: 29 columns of 5.
const DESIRED_CUBES: usize = 29 * 5;

/// A camera at `eye` looking at `look_at`.
fn looking_at(eye: DVec3, look_at: DVec3) -> Camera {
    let d = look_at - eye;
    let yaw = f32::atan2(d.x as f32, -d.z as f32);
    let pitch = f32::atan2(d.y as f32, d.x.hypot(d.z) as f32);
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), 1.0);
    camera.set_yaw(yaw);
    camera.set_pitch(pitch);
    camera
}

fn server_config() -> ServerConfig {
    ServerConfig {
        seed: DEFAULT_SEED,
        streaming: true,
        lod_levels: 0, // this scene predates LOD rings; 2.6 keeps it LOD0-only
        view_radius: 3,
        vertical_radius: 2,
        send_budget_per_tick: 64,
        gen_threads: 0,
        gen_budget_per_tick: 64,
        max_in_flight: 512,
        ..ServerConfig::default()
    }
}

/// Streams terrain until settled and meshes every cube. When `edits` is
/// given they are sent through the protocol first (the `edit_spawn` scene).
fn streamed_scene(
    server: &mut HeadlessServer,
    edits: &[ClientMsg],
) -> (
    ClientWorld,
    Vec<(jbm_client::lod::CubeKey, greedy::MeshOutput)>,
    DVec3,
) {
    let mut registry = BlockRegistry::new();
    builtin::Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);

    let mut client = server.connect("viewer");
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
    let spawn = spawn.expect("no Welcome");

    if !edits.is_empty() {
        for msg in edits {
            client.send(msg.clone());
        }
        server.step(1);
        for msg in client.drain() {
            world.apply(&msg);
        }
    }

    let dirty = world.take_dirty(usize::MAX, spawn);
    let meshes: Vec<(jbm_client::lod::CubeKey, greedy::MeshOutput)> = dirty
        .into_iter()
        .map(|(key, _, input)| (key, greedy::mesh(&input, &appearance)))
        .collect();
    (world, meshes, spawn)
}

/// The interior of the closed 8x5x8 room about 20 blocks under the surface
/// at (8, 8): 8 wide in x and z, 5 tall in y, ceiling 20 blocks below the
/// surface height there. Mid-cube, so the whole room is one mesh.
fn room_bounds() -> (Range<i64>, Range<i64>, Range<i64>) {
    let generator = Generator::new(
        DEFAULT_SEED,
        builtin::Blocks::register(&mut BlockRegistry::new()),
    );
    let ceiling = generator.surface_height(8, 8) - 20;
    (4..12, ceiling - 5..ceiling, 4..12)
}

/// Carves the room by applying `BlockChange` messages — the same edit path
/// real block edits take through the client world — and asserts the room is
/// closed (every shell block solid) before returning.
fn carve_room(world: &mut ClientWorld) {
    let (xs, ys, zs) = room_bounds();
    for y in ys.clone() {
        for z in zs.clone() {
            for x in xs.clone() {
                world.apply(&ServerMsg::BlockChange {
                    pos: BlockPos { x, y, z },
                    state: BlockStateId::AIR,
                });
            }
        }
    }
    // Closed: the one-block shell around the interior is entirely solid.
    for y in ys.start - 1..=ys.end {
        for z in zs.start - 1..=zs.end {
            for x in xs.start - 1..=xs.end {
                let inside = xs.contains(&x) && ys.contains(&y) && zs.contains(&z);
                let state = world.block(BlockPos { x, y, z });
                if inside {
                    assert_eq!(state, Some(BlockStateId::AIR), "room block ({x},{y},{z})");
                } else {
                    assert_ne!(
                        state,
                        Some(BlockStateId::AIR),
                        "room shell block ({x},{y},{z}) leaks"
                    );
                }
            }
        }
    }
}

/// A camera standing in the middle of the carved room, looking horizontally
/// (yaw 0 looks toward -Z).
fn room_camera() -> (Camera, DVec3) {
    let (_, ys, _) = room_bounds();
    let eye = DVec3::new(7.5, ys.start as f64 + 2.5, 7.5);
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), 1.0);
    camera.set_pitch(0.0);
    (camera, eye)
}

/// Re-meshes everything the world marked dirty and re-uploads it into the
/// pass (the re-mesh path: meta slots rewritten in place).
fn upload_dirty(gpu: &jbm_client::gpu::Gpu, world: &mut ClientWorld, terrain: &mut TerrainPass) {
    let mut registry = BlockRegistry::new();
    builtin::Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    let dirty = world.take_dirty(usize::MAX, DVec3::ZERO);
    for (pos, _, input) in dirty {
        terrain.upload(gpu, pos, &greedy::mesh(&input, &appearance));
    }
}

/// Asserts two same-size images are exactly pixel-identical.
fn assert_identical(a: &RgbaImage, b: &RgbaImage, what: &str) {
    assert_eq!(a.dimensions(), b.dimensions(), "{what}: size mismatch");
    let differing = a
        .pixels()
        .zip(b.pixels())
        .filter(|(a, b)| a.0 != b.0)
        .count();
    assert_eq!(differing, 0, "{what}: {differing} pixels differ");
}

/// One frame through `terrain` into `target`, returns the image.
fn render_frame(
    gpu: &jbm_client::gpu::Gpu,
    renderer: &mut Renderer,
    target: &OffscreenTarget,
    depth: &DepthTarget,
    terrain: &mut TerrainPass,
    camera: &Camera,
) -> RgbaImage {
    let view = target.create_view();
    renderer.render(
        gpu,
        FrameInputs {
            target: &view,
            depth,
            camera,
            terrain: Some(terrain),
            outline: None,
            overlay: None,
        },
    );
    target.read_rgba(gpu)
}

/// Builds a pass holding the meshes and renders the scene through every
/// available path, asserting all images are pixel-identical (the path
/// equality test). Prints which paths ran.
fn render_all_paths(
    gpu: &jbm_client::gpu::Gpu,
    meshes: &[(jbm_client::lod::CubeKey, greedy::MeshOutput)],
    camera: &Camera,
    golden: &str,
) {
    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(gpu, target.format());
    renderer.set_viewport(256, 256);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(gpu, target.format(), renderer.uniform_layout(), &textures);
    for (pos, mesh) in meshes {
        terrain.upload(gpu, *pos, mesh);
    }

    // Three frames so the two-phase state settles, then compare per path.
    // gpu-cull is opt-in since it stopped being the default: prefer it
    // explicitly and assert it really is active when the adapter allows it.
    let mut reference: Option<(TerrainPath, RgbaImage)> = None;
    let mut compared = 0;
    for path in [
        TerrainPath::GpuCull,
        TerrainPath::CpuCull,
        TerrainPath::Fallback,
    ] {
        terrain.set_prefer_gpu_cull(path == TerrainPath::GpuCull);
        terrain.set_force_cpu_cull(path == TerrainPath::CpuCull);
        terrain.set_force_fallback(path == TerrainPath::Fallback);
        let active = terrain.path();
        // A path the adapter (or JBM_NO_MDI=1) rules out collapses to the
        // next one down; every available path must really be active.
        let available = match path {
            TerrainPath::GpuCull => terrain.gpu_cull_available(),
            TerrainPath::CpuCull => terrain.uses_mdi(),
            TerrainPath::Fallback => true,
        };
        if available {
            assert_eq!(active, path, "the requested path must be active");
        } else {
            println!("path equality: {path:?} unavailable here, rendering {active:?} instead");
        }
        for _ in 0..3 {
            render_frame(gpu, &mut renderer, &target, &depth, &mut terrain, camera);
        }
        let image = render_frame(gpu, &mut renderer, &target, &depth, &mut terrain, camera);
        let st = terrain.read_cull_stats_blocking(gpu);
        println!(
            "PATHDBG {path:?} candidates {} frustum {} occluded {} drawn {} commands {} faces {} overflow {}",
            st.candidates,
            st.frustum_culled,
            st.occluded,
            st.drawn(),
            st.commands,
            st.faces,
            st.region_overflow
        );
        let _ = std::fs::create_dir_all("target");
        let _ = image.save(format!("target/pathdbg2_{golden}_{active:?}.png"));
        match &reference {
            None => reference = Some((active, image)),
            Some((reference_path, reference_image)) => {
                assert_identical(
                    reference_image,
                    &image,
                    &format!("{path:?} vs {reference_path:?}"),
                );
                compared += 1;
            }
        }
    }
    assert!(compared >= 1, "at least two paths must have been compared");

    // Without MDI only the fallback exists: say so and check it against
    // the golden instead (the dedicated screenshot tests cover the default
    // path's golden on capable adapters).
    let (first_path, first_image) = reference.expect("set above");
    if first_path == TerrainPath::Fallback {
        println!(
            "adapter lacks multi-draw indirect: checking the fallback against the golden only"
        );
        assert_matches_golden(golden, &first_image);
    }
}

#[test]
#[cfg(not(feature = "cube16"))] // streamed shape and camera are 32³-only
fn terrain_spawn_paths_render_pixel_identical() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut server = HeadlessServer::with_config(server_config());
    let (_, meshes, spawn) = streamed_scene(&mut server, &[]);

    // The terrain_spawn camera: above and behind spawn, looking down.
    let camera = looking_at(
        spawn + DVec3::new(-24.0, 20.0, -24.0),
        spawn + DVec3::new(24.0, -8.0, 24.0),
    );
    render_all_paths(&gpu, &meshes, &camera, "terrain_spawn");
}

#[test]
#[cfg(not(feature = "cube16"))] // streamed shape and camera are 32³-only
fn edit_spawn_paths_render_pixel_identical() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut server = HeadlessServer::with_config(server_config());
    let generator = Generator::new(
        DEFAULT_SEED,
        builtin::Blocks::register(&mut BlockRegistry::new()),
    );
    let surface = |x: i64, z: i64| generator.surface_height(x, z);

    // The edit_spawn edits (edit_screenshot.rs): a 3x3 pit two blocks deep
    // just north of spawn plus a glass-and-planks pillar beside it.
    let mut edits = Vec::new();
    for x in 0..=2 {
        for z in -3..=-1 {
            for y in [surface(x, z), surface(x, z) - 1] {
                edits.push(ClientMsg::BreakBlock {
                    pos: BlockPos { x, y, z },
                });
            }
        }
    }
    let (pillar_x, pillar_z) = (1, 0);
    let pillar_base = surface(pillar_x, pillar_z) + 1;
    let registry_states = builtin::Blocks::register(&mut BlockRegistry::new());
    for (i, state) in [
        registry_states.glass,
        registry_states.glass,
        registry_states.oak_planks,
        registry_states.oak_planks,
    ]
    .into_iter()
    .enumerate()
    {
        edits.push(ClientMsg::PlaceBlock {
            pos: BlockPos {
                x: pillar_x,
                y: pillar_base + i as i64,
                z: pillar_z,
            },
            state,
        });
    }
    let (_, meshes, _) = streamed_scene(&mut server, &edits);

    // The edit_spawn camera, without outline or overlay (those are not
    // part of the terrain pass under test).
    let base = surface(0, 0);
    let camera = looking_at(
        DVec3::new(6.0, base as f64 + 5.5, -1.0),
        DVec3::new(1.5, base as f64 - 2.0, -2.0),
    );
    render_all_paths(&gpu, &meshes, &camera, "edit_spawn");
}

/// The 60-frame camera path through the spawn region: turning on the spot
/// above spawn, flying forward and down into the carved room, looking
/// around inside it, and rising back out.
fn path_frame(index: usize, spawn: DVec3, room_eye: DVec3) -> Camera {
    let vantage = spawn + DVec3::new(-24.0, 20.0, -24.0);
    let t = index as f64;
    let (eye, yaw, pitch) = if index < 20 {
        // Turning on the spot at the vantage.
        let yaw = (t / 20.0) * std::f64::consts::FRAC_PI_2;
        (vantage, yaw, -0.35)
    } else if index < 40 {
        // Flying forward and down into the room (straight through the
        // ground; a render camera has no collision).
        let s = (t - 20.0) / 20.0;
        let eye = vantage.lerp(room_eye, s);
        (eye, 0.6 * (1.0 - s), -0.35 * (1.0 - s))
    } else if index < 50 {
        // Inside the closed room, turning.
        let yaw = ((t - 40.0) / 10.0) * std::f64::consts::PI;
        (room_eye, yaw, 0.0)
    } else {
        // Rising back out to the vantage.
        let s = (t - 50.0) / 10.0;
        let eye = room_eye.lerp(vantage, s);
        (eye, 0.0, -0.35 * s)
    };
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), 1.0);
    camera.set_yaw(yaw as f32);
    camera.set_pitch(pitch as f32);
    camera
}

#[test]
#[cfg(not(feature = "cube16"))] // streamed shape, room and cameras are 32³-only
fn nothing_pops_in_across_a_moving_camera() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut server = HeadlessServer::with_config(server_config());
    let (mut world, meshes, spawn) = streamed_scene(&mut server, &[]);
    carve_room(&mut world);

    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    // One pass object across every frame, so the two-phase state is real.
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    for (pos, mesh) in &meshes {
        terrain.upload(&gpu, *pos, mesh);
    }
    upload_dirty(&gpu, &mut world, &mut terrain);
    if !terrain.gpu_cull_available() {
        println!("skipping: adapter lacks the gpu-cull path");
        return;
    }
    // Opt into gpu-cull (no longer the default) and prove it is active.
    terrain.set_prefer_gpu_cull(true);
    assert_eq!(terrain.path(), TerrainPath::GpuCull);

    let (_, room_eye) = room_camera();
    let mut frames_with_occlusion = 0;
    for index in 0..60 {
        let camera = path_frame(index, spawn, room_eye);
        // The same pass, gpu-cull first (advancing the visibility bits),
        // then cpu-cull on the identical camera: the reference.
        let gpu_image = render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
        let stats = terrain.read_cull_stats_blocking(&gpu);
        assert_eq!(
            stats.region_overflow, 0,
            "frame {index}: the CPU frustum bound must cover every GPU emission"
        );
        if stats.occluded > 0 {
            frames_with_occlusion += 1;
        }
        terrain.set_force_cpu_cull(true);
        let cpu_image = render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
        terrain.set_force_cpu_cull(false);
        assert_identical(
            &gpu_image,
            &cpu_image,
            &format!("frame {index}: gpu-cull popped in or dropped geometry"),
        );
    }
    assert!(
        frames_with_occlusion > 0,
        "the camera path must exercise real occlusion (none of the 60 frames culled anything)"
    );
    println!("no-popping: {frames_with_occlusion} of 60 frames had occluded cubes");
}

#[test]
#[cfg(not(feature = "cube16"))] // streamed shape, room and camera are 32³-only
fn cave_culling_hides_the_buried_world() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut server = HeadlessServer::with_config(server_config());
    let (mut world, meshes, _) = streamed_scene(&mut server, &[]);
    carve_room(&mut world);

    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    for (pos, mesh) in &meshes {
        terrain.upload(&gpu, *pos, mesh);
    }
    upload_dirty(&gpu, &mut world, &mut terrain);
    if !terrain.gpu_cull_available() {
        println!("skipping: adapter lacks the gpu-cull path");
        return;
    }
    // Opt into gpu-cull (no longer the default) and prove it is active.
    terrain.set_prefer_gpu_cull(true);
    assert_eq!(terrain.path(), TerrainPath::GpuCull);

    // The camera stands in the closed room looking horizontally; three
    // frames so occlusion is in its steady state.
    let (camera, _) = room_camera();
    for _ in 0..2 {
        render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    }
    let gpu_image = render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    let stats = terrain.read_cull_stats_blocking(&gpu);
    assert_eq!(
        stats.region_overflow, 0,
        "the CPU frustum bound must cover every GPU emission"
    );
    let frustum_passing = stats.candidates - stats.frustum_culled;
    println!(
        "cave culling: candidates {} frustum-culled {} frustum-passing {} occluded {} drawn {}",
        stats.candidates,
        stats.frustum_culled,
        frustum_passing,
        stats.occluded,
        stats.drawn(),
    );
    assert!(frustum_passing > 0, "the room walls must be in the frustum");
    assert!(
        u64::from(stats.occluded) * 10 >= u64::from(frustum_passing) * 8,
        "at least 80% of the frustum-passing candidates must be occlusion-culled \
         inside a closed room ({} of {})",
        stats.occluded,
        frustum_passing,
    );

    // And the image still equals the cpu-cull reference.
    terrain.set_force_cpu_cull(true);
    let cpu_image = render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    terrain.set_force_cpu_cull(false);
    assert_identical(&gpu_image, &cpu_image, "cave: gpu-cull vs cpu-cull");
}

#[test]
#[cfg(not(feature = "cube16"))] // streamed shape and camera are 32³-only
fn open_surface_view_stays_uncullled() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut server = HeadlessServer::with_config(server_config());
    let (_, meshes, spawn) = streamed_scene(&mut server, &[]);

    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    for (pos, mesh) in &meshes {
        terrain.upload(&gpu, *pos, mesh);
    }
    if !terrain.gpu_cull_available() {
        println!("skipping: adapter lacks the gpu-cull path");
        return;
    }
    // Opt into gpu-cull (no longer the default) and prove it is active.
    terrain.set_prefer_gpu_cull(true);
    assert_eq!(terrain.path(), TerrainPath::GpuCull);

    // The terrain_spawn camera on the surface; two warm-up frames so the
    // occlusion state is steady.
    let camera = looking_at(
        spawn + DVec3::new(-24.0, 20.0, -24.0),
        spawn + DVec3::new(24.0, -8.0, 24.0),
    );
    for _ in 0..2 {
        render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    }
    let gpu_image = render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    let stats = terrain.read_cull_stats_blocking(&gpu);
    assert_eq!(
        stats.region_overflow, 0,
        "the CPU frustum bound must cover every GPU emission"
    );
    let frustum_passing = stats.candidates - stats.frustum_culled;
    println!(
        "open view: candidates {} frustum-culled {} frustum-passing {} occluded {} drawn {}",
        stats.candidates,
        stats.frustum_culled,
        frustum_passing,
        stats.occluded,
        stats.drawn(),
    );
    // From the open surface most cubes stay visible: occlusion must stay
    // well below the cave's 80%.
    assert!(
        u64::from(stats.occluded) * 2 < u64::from(frustum_passing),
        "occluded {} of frustum-passing {} on the open surface",
        stats.occluded,
        frustum_passing,
    );

    terrain.set_force_cpu_cull(true);
    let cpu_image = render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    terrain.set_force_cpu_cull(false);
    assert_identical(&gpu_image, &cpu_image, "open view: gpu-cull vs cpu-cull");
}

#[test]
#[cfg(not(feature = "cube16"))] // streamed shape and camera are 32³-only
fn odd_resize_keeps_the_paths_identical() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut server = HeadlessServer::with_config(server_config());
    let (_, meshes, spawn) = streamed_scene(&mut server, &[]);

    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let mut depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    for (pos, mesh) in &meshes {
        terrain.upload(&gpu, *pos, mesh);
    }
    if !terrain.gpu_cull_available() {
        println!("skipping: adapter lacks the gpu-cull path");
        return;
    }
    // Opt into gpu-cull (no longer the default) and prove it is active.
    terrain.set_prefer_gpu_cull(true);
    assert_eq!(terrain.path(), TerrainPath::GpuCull);

    let camera = looking_at(
        spawn + DVec3::new(-24.0, 20.0, -24.0),
        spawn + DVec3::new(24.0, -8.0, 24.0),
    );
    // One frame at 256x256 (this builds the pyramid), then the odd resize.
    render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);

    let target = OffscreenTarget::new(&gpu.device, 333, 217);
    depth.resize(&gpu.device, 333, 217);
    renderer.set_viewport(333, 217);
    let gpu_image = render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    terrain.set_force_cpu_cull(true);
    let cpu_image = render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    terrain.set_force_cpu_cull(false);
    assert_identical(&gpu_image, &cpu_image, "odd resize: gpu-cull vs cpu-cull");
}

#[test]
#[cfg(not(feature = "cube16"))] // streamed shape and camera are 32³-only
fn terrain_spawn_stats_show_direction_skipping() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut server = HeadlessServer::with_config(server_config());
    let (_, meshes, spawn) = streamed_scene(&mut server, &[]);

    let camera = looking_at(
        spawn + DVec3::new(-24.0, 20.0, -24.0),
        spawn + DVec3::new(24.0, -8.0, 24.0),
    );
    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    for (pos, mesh) in &meshes {
        terrain.upload(&gpu, *pos, mesh);
    }
    render_frame(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    // On gpu-cull the drawn/faces/commands counters come back from the GPU;
    // the blocking read makes them exact for this frame.
    let cull = terrain.read_cull_stats_blocking(&gpu);
    assert_eq!(
        cull.region_overflow, 0,
        "the CPU frustum bound must cover every GPU emission"
    );
    let stats = terrain.stats();
    assert!(stats.drawn > 0);
    assert_eq!(u32::try_from(stats.drawn).unwrap(), cull.drawn());

    if terrain.uses_mdi() {
        assert!(
            stats.draw_calls <= 2 * stats.pages as u64,
            "at most one multi_draw_indirect per non-empty page and phase: {stats:?}"
        );
    }
    assert!(
        stats.commands >= stats.drawn as u64,
        "every drawn cube issues at least one command: {stats:?}"
    );
    assert!(
        stats.commands <= 6 * stats.drawn as u64,
        "at most one command per cube and direction: {stats:?}"
    );

    // Directions really are skipped: the issued faces fall below the total
    // face count of the cubes that passed the frustum test. Reproduce the
    // visibility test with the public Frustum to know that set.
    let frustum = Frustum::from_view_proj(&camera.view_proj());
    let total_drawn_faces: u64 = meshes
        .iter()
        .filter(|(_, mesh)| !mesh.is_empty())
        .filter(|(key, _)| {
            let pos = key.pos;
            let world_min = DVec3::new(
                f64::from(pos.x) * f64::from(CUBE_SIZE),
                f64::from(pos.y) * f64::from(CUBE_SIZE),
                f64::from(pos.z) * f64::from(CUBE_SIZE),
            );
            let min = camera.relative(world_min);
            let max = min + glam::Vec3::splat(CUBE_SIZE as f32);
            frustum.intersects_aabb(min, max)
        })
        .map(|(_, mesh)| mesh.face_count() as u64)
        .sum();
    assert!(
        stats.faces < total_drawn_faces,
        "faces {} must skip directions below the drawn cubes' total {total_drawn_faces}",
        stats.faces
    );
}
