//! End-to-end edit screenshot (task 1.7): a real server, a real client
//! world, and a scene built entirely through `BreakBlock`/`PlaceBlock`
//! messages — a 3x3 pit two blocks deep next to spawn with a glass and
//! plank pillar beside it. Renders with the target outline on one pit
//! block and the crosshair; golden `edit_spawn`.

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::mesh::appearance::{BlockAppearance, builtin_textures};
use jbm_client::mesh::greedy;
use jbm_client::render::chunks::TerrainPass;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::overlay::OverlayText;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::{assert_matches_golden, gpu_or_skip};
use jbm_client::world::ClientWorld;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
use jbm_core::coords::BlockPos;
use jbm_core::protocol::{ClientMsg, ServerMsg};
use jbm_server::terrain::Generator;
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
fn edited_spawn_scene_matches_golden() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };

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
    });
    let mut registry = BlockRegistry::new();
    let blocks = builtin::Blocks::register(&mut registry);
    let generator = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Classic);
    let surface = |x: i64, z: i64| generator.surface_height(x, z);

    let mut client = server.connect("builder");
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
    let base = surface(0, 0);

    // The 3x3 pit, two deep, just north of spawn: break each column's
    // surface block and the one under it. The server player stands at
    // spawn (no inputs are ever sent), so reach is measured from the
    // spawn eye.
    let eye = spawn + DVec3::new(0.0, 1.62, 0.0);
    let mut edits = Vec::new();
    for x in 0..=2 {
        for z in -3..=-1 {
            for y in [surface(x, z), surface(x, z) - 1] {
                let pos = BlockPos { x, y, z };
                let center = DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
                assert!(
                    (center - eye).length() <= 6.0,
                    "pit block {pos:?} out of the server's reach"
                );
                edits.push(ClientMsg::BreakBlock { pos });
            }
        }
    }
    // The pillar beside the pit: two glass, two planks, standing on the
    // surface in front of the pit's east edge.
    let pillar_x = 1;
    let pillar_z = 0;
    let pillar_base = surface(pillar_x, pillar_z) + 1;
    for (i, state) in [
        blocks.glass,
        blocks.glass,
        blocks.oak_planks,
        blocks.oak_planks,
    ]
    .into_iter()
    .enumerate()
    {
        let pos = BlockPos {
            x: pillar_x,
            y: pillar_base + i as i64,
            z: pillar_z,
        };
        let center = DVec3::new(pos.x as f64 + 0.5, pos.y as f64 + 0.5, pos.z as f64 + 0.5);
        assert!(
            (center - eye).length() <= 6.0,
            "pillar block {pos:?} out of the server's reach"
        );
        edits.push(ClientMsg::PlaceBlock { pos, state });
    }
    for msg in edits {
        client.send(msg);
    }
    server.step(1);
    let mut changes = 0;
    for msg in client.drain() {
        if let ServerMsg::BlockChange { .. } = msg {
            changes += 1;
        }
        world.apply(&msg);
    }
    assert_eq!(changes, 18 + 4, "every edit must be accepted");
    // Every edited block's cube is pinned (the pit and pillar span up to
    // three cubes here, across the z = 0 cube border and possibly a
    // vertical one).
    assert_eq!(server.server().pinned_count(), 3);
    // The pit floor is open and the pillar stands.
    assert_eq!(
        world.block(BlockPos {
            x: 1,
            y: surface(1, -2) - 1,
            z: -2
        }),
        Some(BlockStateId::AIR)
    );
    assert_eq!(
        world.block(BlockPos {
            x: pillar_x,
            y: pillar_base + 3,
            z: pillar_z
        }),
        Some(blocks.oak_planks)
    );

    // Mesh everything (edits first) and render.
    let appearance = BlockAppearance::new(&registry);
    let dirty = world.take_dirty(usize::MAX, spawn);
    assert!(!dirty.is_empty());
    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    for (pos, _, input) in dirty {
        terrain.upload(&gpu, pos, &greedy::mesh(&input, &appearance));
    }

    // A camera about 6 blocks east of the pit, looking steeply down into it
    // so the floor, walls and corner AO all show;
    // the outline marks one pit floor block, and the empty overlay turns
    // the crosshair on without text.
    let camera = looking_at(
        DVec3::new(6.0, base as f64 + 5.5, -1.0),
        DVec3::new(1.5, base as f64 - 2.0, -2.0),
    );
    // The pit floor block under the pit's center column: solid, so the
    // outline marks what a player aiming there would target.
    let outlined = BlockPos {
        x: 1,
        y: surface(1, -2) - 2,
        z: -2,
    };
    let overlay = OverlayText::new();
    renderer.render(
        &gpu,
        FrameInputs {
            target: &target.create_view(),
            depth: &depth,
            camera: &camera,
            terrain: Some(&mut terrain),
            outline: Some(outlined),
            overlay: Some(&overlay),
        },
    );
    let with_outline = target.read_rgba(&gpu);
    assert_matches_golden("edit_spawn", &with_outline);

    // The golden tolerance (0.5% of pixels) is larger than the whole
    // outline, so the golden alone cannot catch a missing or depth-failed
    // outline. Render again without it and require a visible difference.
    renderer.render(
        &gpu,
        FrameInputs {
            target: &target.create_view(),
            depth: &depth,
            camera: &camera,
            terrain: Some(&mut terrain),
            outline: None,
            overlay: Some(&overlay),
        },
    );
    let without = target.read_rgba(&gpu);
    let outline_pixels = with_outline
        .pixels()
        .zip(without.pixels())
        .filter(|(a, b)| a.0.iter().zip(b.0.iter()).any(|(x, y)| x.abs_diff(*y) > 24))
        .count();
    assert!(
        outline_pixels >= 20,
        "target outline barely visible: {outline_pixels} pixels differ"
    );
}
