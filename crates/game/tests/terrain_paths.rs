//! Draw-path tests (task 2.2) on the real streamed scenes: the MDI path
//! and the direct-draw fallback must render pixel-identical images, and
//! the terrain stats must reflect per-direction skipping. GPU tests,
//! skipped without an adapter like the other screenshot tests. 32³-only:
//! the streamed shape and cameras mirror `terrain_screenshot.rs` /
//! `edit_screenshot.rs`.
#![cfg(not(feature = "cube16"))]

use glam::{DVec3, Vec3};
use jbm_client::camera::Camera;
use jbm_client::mesh::appearance::{BlockAppearance, builtin_textures};
use jbm_client::mesh::greedy;
use jbm_client::render::chunks::{Frustum, TerrainPass};
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, Renderer};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::{assert_matches_golden, gpu_or_skip};
use jbm_client::world::ClientWorld;
use jbm_core::block::{BlockRegistry, builtin};
use jbm_core::coords::{CUBE_SIZE, CubePos};
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
        view_radius: 3,
        vertical_radius: 2,
        send_budget_per_tick: 64,
        gen_threads: 0,
        gen_budget_per_tick: 64,
        max_in_flight: 512,
    }
}

/// Streams terrain until settled and meshes every cube. When `edits` is
/// given they are sent through the protocol first (the `edit_spawn` scene).
fn streamed_scene(
    server: &mut HeadlessServer,
    edits: &[ClientMsg],
) -> (ClientWorld, Vec<(CubePos, greedy::MeshOutput)>, DVec3) {
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
    let meshes: Vec<(CubePos, greedy::MeshOutput)> = dirty
        .into_iter()
        .map(|(pos, _, input)| (pos, greedy::mesh(&input, &appearance)))
        .collect();
    (world, meshes, spawn)
}

/// Renders the scene through both draw paths (automatic first, then the
/// forced fallback) and asserts the two images are pixel-identical — exact,
/// not the golden tolerance. When the adapter lacks MDI there is nothing to
/// compare, so the fallback image is checked against `golden` instead.
/// Returns whether MDI was used for the first render.
fn render_both_paths(
    gpu: &jbm_client::gpu::Gpu,
    meshes: &[(CubePos, greedy::MeshOutput)],
    camera: &Camera,
    golden: &str,
) -> bool {
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

    let view = target.create_view();
    renderer.render(
        gpu,
        &view,
        depth.view(),
        camera,
        None,
        Some(&mut terrain),
        None,
    );
    let mdi_image = target.read_rgba(gpu);
    let mdi = terrain.uses_mdi();

    terrain.set_force_fallback(true);
    renderer.render(
        gpu,
        &view,
        depth.view(),
        camera,
        None,
        Some(&mut terrain),
        None,
    );
    let fallback_image = target.read_rgba(gpu);

    if mdi {
        assert_eq!(
            mdi_image.dimensions(),
            fallback_image.dimensions(),
            "paths render the same target"
        );
        let differing = mdi_image
            .pixels()
            .zip(fallback_image.pixels())
            .filter(|(a, b)| a.0 != b.0)
            .count();
        assert_eq!(
            differing, 0,
            "MDI and fallback images must be pixel-identical"
        );
    } else {
        println!("adapter lacks multi-draw indirect: checking fallback against the golden only");
        assert_matches_golden(golden, &fallback_image);
    }
    mdi
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
    render_both_paths(&gpu, &meshes, &camera, "terrain_spawn");
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
    // just north of spawn plus a glass-and-planks pillar beside it. Reach
    // is measured from the spawn eye; the server player never moves.
    let mut edits = Vec::new();
    for x in 0..=2 {
        for z in -3..=-1 {
            for y in [surface(x, z), surface(x, z) - 1] {
                edits.push(ClientMsg::BreakBlock {
                    pos: jbm_core::coords::BlockPos { x, y, z },
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
            pos: jbm_core::coords::BlockPos {
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
    render_both_paths(&gpu, &meshes, &camera, "edit_spawn");
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
    renderer.render(
        &gpu,
        &target.create_view(),
        depth.view(),
        &camera,
        None,
        Some(&mut terrain),
        None,
    );
    let stats = terrain.stats();
    assert!(stats.drawn > 0);

    if terrain.uses_mdi() {
        assert!(
            stats.draw_calls <= stats.pages as u64,
            "one multi_draw_indirect per non-empty page: {stats:?}"
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
        .filter(|(pos, _)| {
            let world_min = DVec3::new(
                f64::from(pos.x) * f64::from(CUBE_SIZE),
                f64::from(pos.y) * f64::from(CUBE_SIZE),
                f64::from(pos.z) * f64::from(CUBE_SIZE),
            );
            let min = camera.relative(world_min);
            let max = min + Vec3::splat(CUBE_SIZE as f32);
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
