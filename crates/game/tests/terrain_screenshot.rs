//! End-to-end screenshot: the real server generates terrain and streams it
//! over an in-process link; the real client world applies the messages,
//! meshes every cube and renders the view from spawn. This is the one test
//! that crosses both halves, so it lives in the crate that depends on both.

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::mesh::appearance::{BlockAppearance, builtin_textures};
use jbm_client::mesh::greedy;
use jbm_client::render::chunks::TerrainPass;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::{assert_matches_golden, gpu_or_skip};
use jbm_client::world::ClientWorld;
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::protocol::ServerMsg;
use jbm_server::worldgen::Preset;
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

#[test]
#[cfg(not(feature = "cube16"))] // streaming shape and golden are 32³-only
fn streamed_terrain_matches_golden() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };

    // Inline generation keeps the streamed set deterministic.
    let mut server = HeadlessServer::with_config(ServerConfig {
        worldgen: Preset::Classic, // pin the pre-2.8 terrain: this golden predates it
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
    });
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

    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);

    let mut registry = BlockRegistry::new();
    Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    let dirty = world.take_dirty(usize::MAX, spawn);
    assert_eq!(dirty.len(), DESIRED_CUBES);
    for (pos, _, input) in dirty {
        terrain.upload(&gpu, pos, &greedy::mesh(&input, &appearance));
    }

    // Above and behind spawn, looking down across it.
    let camera = looking_at(
        spawn + DVec3::new(-24.0, 20.0, -24.0),
        spawn + DVec3::new(24.0, -8.0, 24.0),
    );
    renderer.render(
        &gpu,
        FrameInputs {
            target: &target.create_view(),
            depth: &depth,
            camera: &camera,
            terrain: Some(&mut terrain),
            outline: None,
            overlay: None,
        },
    );
    assert_matches_golden("terrain_spawn", &target.read_rgba(&gpu));
}
