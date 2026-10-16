//! Screenshot tests: 256x256 offscreen renders compared against committed
//! goldens. Each scene pins its camera so the image is reproducible.

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::overlay::OverlayText;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::triangle;
use jbm_client::testing::screenshot::{assert_matches_golden, gpu_or_skip};

/// The sky_horizon camera: at (0, 80, 0), yaw 0, pitch `pitch_deg`, fov 70.
fn sky_camera(pitch_deg: f32) -> Camera {
    let mut camera = Camera::new(DVec3::new(0.0, 80.0, 0.0), 70.0_f32.to_radians(), 1.0);
    camera.set_pitch(pitch_deg.to_radians());
    camera
}

fn render_and_compare(name: &str, camera: &Camera, overlay: Option<OverlayText>) {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    renderer.render(
        &gpu,
        FrameInputs {
            target: &target.create_view(),
            depth: &depth,
            camera,
            terrain: None,
            outline: None,
            overlay: overlay.as_ref(),
        },
    );
    let image = target.read_rgba(&gpu);
    assert_matches_golden(name, &image);
}

#[test]
fn triangle_matches_golden() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    triangle::draw_triangle(&gpu, &target);
    let image = target.read_rgba(&gpu);
    assert_matches_golden("triangle", &image);
}

#[test]
fn sky_horizon_matches_golden() {
    render_and_compare("sky_horizon", &sky_camera(0.0), None);
}

#[test]
fn sky_up_matches_golden() {
    render_and_compare("sky_up", &sky_camera(60.0), None);
}

#[test]
fn sky_down_matches_golden() {
    render_and_compare("sky_down", &sky_camera(-60.0), None);
}

#[test]
fn debug_overlay_matches_golden() {
    let mut overlay = OverlayText::new();
    overlay.push("FPS 60  16.67 ms");
    overlay.push("XYZ 0.00 80.00 0.00");
    render_and_compare("debug_overlay", &sky_camera(0.0), Some(overlay));
}

// ---------------------------------------------------------------------------
// Terrain scenes: built from Cube::from_fn, meshed synchronously (no pool),
// uploaded and rendered with pinned cameras. The scenes place blocks by hand
// at 32³-local coordinates and the goldens match them, so everything below
// only runs at the default cube size (16³ would need new scenes + goldens).
#[cfg(not(feature = "cube16"))]
use jbm_client::mesh::appearance::BlockAppearance;
#[cfg(not(feature = "cube16"))]
use jbm_client::mesh::greedy::{self, MeshInput};
#[cfg(not(feature = "cube16"))]
use jbm_client::render::chunks::TerrainPass;
#[cfg(not(feature = "cube16"))]
use jbm_client::render::texture_array::BlockTextures;
#[cfg(not(feature = "cube16"))]
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
#[cfg(not(feature = "cube16"))]
use jbm_core::coords::CubePos;
#[cfg(not(feature = "cube16"))]
use jbm_core::cube::Cube;
#[cfg(not(feature = "cube16"))]
use std::collections::HashMap;
#[cfg(not(feature = "cube16"))]
use std::sync::Arc;

#[cfg(not(feature = "cube16"))] // 32³-only golden
fn test_setup() -> (BlockRegistry, BlockAppearance, builtin::Blocks) {
    let mut registry = BlockRegistry::new();
    let blocks = builtin::Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    (registry, appearance, blocks)
}

/// Renders a scene of cubes against `golden_name`.
#[cfg(not(feature = "cube16"))] // 32³-only golden
fn render_scene(golden_name: &str, scene: &[(CubePos, Cube)], camera: &Camera) {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let (_, appearance, _) = test_setup();
    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    let (layers, _) = jbm_client::mesh::appearance::builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);

    let by_pos: HashMap<CubePos, Arc<Cube>> = scene
        .iter()
        .map(|(pos, cube)| (*pos, Arc::new(cube.clone())))
        .collect();
    for (pos, cube) in &by_pos {
        let mut input = MeshInput::isolated(Arc::clone(cube));
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if (dx, dy, dz) == (0, 0, 0) {
                        continue;
                    }
                    let neighbor = CubePos {
                        x: pos.x + dx,
                        y: pos.y + dy,
                        z: pos.z + dz,
                    };
                    if let Some(cube) = by_pos.get(&neighbor) {
                        input.neighbors[greedy::neighbor_index(dx, dy, dz)] =
                            Some(Arc::clone(cube));
                    }
                }
            }
        }
        let output = greedy::mesh(&input, &appearance);
        terrain.upload(&gpu, jbm_client::lod::CubeKey::lod0(*pos), &output);
    }

    renderer.render(
        &gpu,
        FrameInputs {
            target: &target.create_view(),
            depth: &depth,
            camera,
            terrain: Some(&mut terrain),
            outline: None,
            overlay: None,
        },
    );
    let image = target.read_rgba(&gpu);
    assert_matches_golden(golden_name, &image);
}

/// A camera at `eye` looking at `look_at`.
#[cfg(not(feature = "cube16"))] // 32³-only golden
fn looking_at(eye: (f64, f64, f64), look_at: (f64, f64, f64)) -> Camera {
    let eye = DVec3::from(eye);
    let look_at = DVec3::from(look_at);
    let d = look_at - eye;
    let yaw = f32::atan2(d.x as f32, -d.z as f32);
    let horizontal = (d.x * d.x + d.z * d.z).sqrt();
    let pitch = f32::atan2(d.y as f32, horizontal as f32);
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), 1.0);
    camera.set_yaw(yaw);
    camera.set_pitch(pitch);
    camera
}

/// The mesh_blocks scene: a 12x12 stone floor with one of each builtin
/// block standing on it in a row. Returns the cube holding it.
#[cfg(not(feature = "cube16"))] // 32³-only golden
fn mesh_blocks_scene(blocks: &builtin::Blocks) -> Cube {
    Cube::from_fn(|p| {
        if p.y == 0 && (2..14).contains(&p.x) && (2..14).contains(&p.z) {
            return blocks.stone;
        }
        if p.y == 1 && (3..12).contains(&p.x) && p.z == 7 {
            // One of each builtin block in a row.
            return match p.x {
                3 => blocks.stone,
                4 => blocks.dirt,
                5 => blocks.grass_block,
                6 => blocks.sand,
                7 => blocks.water,
                8 => blocks.glass,
                9 => blocks.oak_planks,
                10 => blocks.oak_log,
                _ => blocks.leaves,
            };
        }
        BlockStateId::AIR
    })
}

#[test]
#[cfg(not(feature = "cube16"))] // 32³-only golden
fn mesh_blocks_matches_golden() {
    let (_, _, blocks) = test_setup();
    let cube = mesh_blocks_scene(&blocks);
    // Camera above a corner, ~30 degrees down: three faces of each block.
    let camera = looking_at((12.5, 5.5, 14.5), (7.5, 1.5, 7.5));
    render_scene(
        "mesh_blocks",
        &[(CubePos { x: 0, y: 0, z: 0 }, cube)],
        &camera,
    );
}

/// The mesh_ao scene: a stepped pyramid whose inner step corners show AO
/// darkening. Local to one cube.
#[cfg(not(feature = "cube16"))] // 32³-only golden
fn mesh_ao_scene(blocks: &builtin::Blocks) -> Cube {
    Cube::from_fn(|p| {
        let step = p.y as i32;
        if step > 5 {
            return BlockStateId::AIR;
        }
        // Square of side 11 - 2*step, roughly centered on (15.5, 15.5).
        let side = 11 - 2 * step;
        let lo = 16 - (side + 1) / 2;
        let hi = lo + side - 1;
        if (lo..=hi).contains(&(p.x as i32)) && (lo..=hi).contains(&(p.z as i32)) {
            blocks.stone
        } else {
            BlockStateId::AIR
        }
    })
}

#[test]
#[cfg(not(feature = "cube16"))] // 32³-only golden
fn mesh_ao_matches_golden() {
    let (_, _, blocks) = test_setup();
    let cube = mesh_ao_scene(&blocks);
    let camera = looking_at((24.5, 11.0, 24.5), (16.0, 2.0, 16.0));
    render_scene("mesh_ao", &[(CubePos { x: 0, y: 0, z: 0 }, cube)], &camera);
}

#[test]
#[cfg(not(feature = "cube16"))] // 32³-only golden
fn mesh_cube_border_matches_golden() {
    let (_, _, blocks) = test_setup();
    // A building crossing the x=32 boundary between cubes (0,0,0) and
    // (1,0,0): blocks x 24..=41, y 0..=5, z 10..=21.
    let make = |base_x: i32| {
        Cube::from_fn(move |p| {
            let x = p.x as i32 + base_x;
            let y = p.y as i32;
            let z = p.z as i32;
            if (24..=41).contains(&x) && (0..=5).contains(&y) && (10..=21).contains(&z) {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        })
    };
    let scene = vec![
        (CubePos { x: 0, y: 0, z: 0 }, make(0)),
        (CubePos { x: 1, y: 0, z: 0 }, make(32)),
    ];
    let camera = looking_at((36.0, 14.0, 31.0), (32.0, 3.0, 16.0));
    render_scene("mesh_cube_border", &scene, &camera);
}

#[test]
#[cfg(not(feature = "cube16"))] // 32³-only golden
fn mesh_far_origin_matches_the_same_golden_as_mesh_ao() {
    let (_, _, blocks) = test_setup();
    let cube = mesh_ao_scene(&blocks);
    // The mesh_ao scene at cube (31_250_000, 2, 31_250_000), about 1e9
    // blocks out, with the camera offset by the same amount. Rendering
    // must be camera-relative, so the same golden matches.
    let far = 31_250_000i64;
    let scene = [(
        CubePos {
            x: far as i32,
            y: 2,
            z: far as i32,
        },
        cube,
    )];
    let world = DVec3::new(far as f64 * 32.0, 64.0, far as f64 * 32.0);
    let eye = DVec3::new(24.5, 11.0, 24.5) + world;
    let look_at = DVec3::new(16.0, 2.0, 16.0) + world;
    let d = look_at - eye;
    let yaw = f32::atan2(d.x as f32, -d.z as f32);
    let pitch = f32::atan2(d.y as f32, ((d.x * d.x + d.z * d.z).sqrt()) as f32);
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), 1.0);
    camera.set_yaw(yaw);
    camera.set_pitch(pitch);
    render_scene("mesh_ao", &scene, &camera);
}
