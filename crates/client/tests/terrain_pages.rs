//! GPU test for the paged face storage (task 2.2): heavy upload churn —
//! re-uploads that must reuse freed slots and removals that free them —
//! then the image is compared against a freshly built pass holding only the
//! final meshes. Pixel-identical output proves slot reuse does not corrupt
//! what gets drawn. Skipped without an adapter. Cube-size agnostic (the
//! tiny page size is derived from `CUBE_SIZE`).

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::mesh::{self, MeshInput};
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::terrain::TerrainPass;
use jbm_client::render::terrain::TerrainPath;
use jbm_client::render::terrain::worst_case_faces;
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
use jbm_core::coords::CubePos;
use jbm_core::cube::Cube;
use std::collections::HashMap;
use std::sync::Arc;

/// The smallest legal page: exactly one worst-case cube mesh, so a handful
/// of checkerboard cubes forces several pages.
fn tiny_page_faces() -> u32 {
    worst_case_faces()
}

/// The worst-case mesh: a checkerboard of stone in air. With missing
/// neighbors (isolated input) the border faces are culled, so it lands just
/// under a full page.
fn checkerboard(blocks: &builtin::Blocks) -> Cube {
    Cube::from_fn(|p| {
        if (p.x + p.y + p.z) % 2 == 0 {
            blocks.stone
        } else {
            BlockStateId::AIR
        }
    })
}

/// A small mesh: one stone block at the cube's center, six faces.
fn center_block(blocks: &builtin::Blocks) -> Cube {
    let mid = (jbm_core::coords::CUBE_SIZE / 2) as u8;
    Cube::from_fn(|p| {
        if p.x == mid && p.y == mid && p.z == mid {
            blocks.stone
        } else {
            BlockStateId::AIR
        }
    })
}

fn mesh_of(appearance: &BlockAppearance, cube: Cube) -> mesh::MeshOutput {
    mesh::mesh(&MeshInput::isolated(Arc::new(cube)), appearance)
}

fn render_passes_identically(
    gpu: &jbm_client::gpu::Gpu,
    churned: &mut TerrainPass,
    fresh_meshes: &[(CubePos, mesh::MeshOutput)],
    camera: &Camera,
    path: jbm_client::render::terrain::TerrainPath,
) {
    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(gpu, target.format());
    renderer.set_viewport(256, 256);
    let textures = BlockTextures::builtin(gpu).expect("texture array");

    churned.set_prefer_gpu_cull(path == TerrainPath::GpuCull);
    churned.set_force_cpu_cull(path == TerrainPath::CpuCull);
    churned.set_force_fallback(path == TerrainPath::Fallback);
    // Without MDI both force flags collapse to the fallback path; the
    // comparison is still valid there.
    assert!(
        churned.path() == path
            || (path == TerrainPath::CpuCull && churned.path() == TerrainPath::Fallback),
        "the pass must use the requested path, got {}",
        churned.path_name()
    );

    let view = target.create_view();
    renderer.render(
        gpu,
        FrameInputs {
            target: &view,
            depth: &depth,
            camera,
            terrain: Some(churned),
            outline: None,
            overlay: None,
        },
    );
    let churned_image = target.read_rgba(gpu);

    let mut fresh = TerrainPass::with_page_faces(
        gpu,
        target.format(),
        renderer.uniform_layout(),
        &textures,
        tiny_page_faces(),
    );
    fresh.set_prefer_gpu_cull(path == TerrainPath::GpuCull);
    fresh.set_force_cpu_cull(path == TerrainPath::CpuCull);
    fresh.set_force_fallback(path == TerrainPath::Fallback);
    for (pos, mesh) in fresh_meshes {
        fresh.upload(gpu, jbm_client::lod::CubeKey::lod0(*pos), mesh);
    }
    renderer.render(
        gpu,
        FrameInputs {
            target: &view,
            depth: &depth,
            camera,
            terrain: Some(&mut fresh),
            outline: None,
            overlay: None,
        },
    );
    let fresh_image = target.read_rgba(gpu);

    assert_eq!(
        churned_image.dimensions(),
        fresh_image.dimensions(),
        "both passes render the same target"
    );
    let differing = churned_image
        .pixels()
        .zip(fresh_image.pixels())
        .filter(|(a, b)| a.0 != b.0)
        .count();
    assert_eq!(
        differing, 0,
        "churned slot reuse must render pixel-identical to a fresh pass"
    );
}

#[test]
fn slot_reuse_after_churn_renders_like_a_fresh_pass() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut registry = BlockRegistry::new();
    let blocks = builtin::Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);

    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    let textures = BlockTextures::builtin(&gpu).expect("texture array");
    let mut terrain = TerrainPass::with_page_faces(
        &gpu,
        target.format(),
        renderer.uniform_layout(),
        &textures,
        tiny_page_faces(),
    );

    // 200 cubes in a 10x20 grid; four checkerboards each nearly fill a
    // page, so at least 3 pages exist no matter the cube size.
    let positions: Vec<CubePos> = (0..200)
        .map(|i| CubePos {
            x: i % 10,
            y: 0,
            z: i / 10,
        })
        .collect();
    let checker = mesh_of(&appearance, checkerboard(&blocks));
    let center = mesh_of(&appearance, center_block(&blocks));
    let mut meshes: HashMap<CubePos, mesh::MeshOutput> = HashMap::new();
    for (i, &pos) in positions.iter().enumerate() {
        let mesh = if i < 4 {
            checker.clone()
        } else {
            center.clone()
        };
        terrain.upload(&gpu, jbm_client::lod::CubeKey::lod0(pos), &mesh);
        meshes.insert(pos, mesh);
    }
    assert!(
        terrain.page_count() >= 3,
        "tiny pages must force several pages, got {}",
        terrain.page_count()
    );

    // Re-upload half with different meshes: the checkerboards shrink to
    // center blocks (freeing near-page-sized ranges) and the first batch of
    // center blocks becomes checkerboards, landing in the freed ranges.
    for (i, &pos) in positions.iter().enumerate().take(100) {
        let mesh = if i < 4 {
            center.clone()
        } else if i < 60 {
            checker.clone()
        } else {
            // A third shape so the re-upload really differs.
            mesh_of(&appearance, center_block(&blocks))
        };
        terrain.upload(&gpu, jbm_client::lod::CubeKey::lod0(pos), &mesh);
        meshes.insert(pos, mesh);
    }
    // Remove a quarter of the cubes.
    for &pos in positions.iter().skip(150) {
        terrain.remove(&gpu, jbm_client::lod::CubeKey::lod0(pos));
        meshes.remove(&pos);
    }
    terrain.check_page_invariants();

    // The grid spans 320 x 640 blocks; look down onto its center.
    let size = f64::from(jbm_core::coords::CUBE_SIZE);
    let center_world = DVec3::new(5.0 * size, 0.0, 10.0 * size);
    let eye = center_world + DVec3::new(0.0, 900.0, 500.0);
    let to_center = center_world - eye;
    let yaw = f32::atan2(to_center.x as f32, -to_center.z as f32);
    let pitch = f32::atan2(
        to_center.y as f32,
        (to_center.x * to_center.x + to_center.z * to_center.z).sqrt() as f32,
    );
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), 1.0);
    camera.set_yaw(yaw);
    camera.set_pitch(pitch);

    let mut fresh_meshes: Vec<(CubePos, mesh::MeshOutput)> = meshes.into_iter().collect();
    fresh_meshes.sort_unstable_by_key(|(pos, _)| *pos);
    // The churned pass renders identically to a fresh pass on cpu-cull and,
    // when the adapter allows it, on gpu-cull too (task 2.5: meta slots and
    // region bounds survive the churn; the two-phase state is real because
    // the same pass object renders the frame).
    render_passes_identically(
        &gpu,
        &mut terrain,
        &fresh_meshes,
        &camera,
        TerrainPath::CpuCull,
    );
    if terrain.gpu_cull_available() {
        render_passes_identically(
            &gpu,
            &mut terrain,
            &fresh_meshes,
            &camera,
            TerrainPath::GpuCull,
        );
    } else {
        println!("skipping gpu-cull churn comparison: adapter lacks the GPU cull path");
    }
    // The direct-draw fallback shares the cpu-cull commands, so it needs no
    // separate churn run.
}
