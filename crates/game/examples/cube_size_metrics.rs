//! Task 1.8 metrics: one aligned table describing the measured terrain
//! region (x, z in [0, 128), y in [32, 128); see `benches/common/mod.rs`)
//! at the compiled cube size. Run at both sizes and compare:
//!
//! ```bash
//! cargo run -p jbm-game --release --example cube_size_metrics
//! cargo run -p jbm-game --release --example cube_size_metrics --features cube16
//! ```

#[path = "../benches/common/mod.rs"]
mod common;

use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::mesh::face::PackedFace;
use jbm_client::mesh::greedy;
use jbm_client::render::chunks::TerrainPass;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, Renderer};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_core::block::BlockStateId;
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;
use jbm_server::DEFAULT_SEED;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FRAMES: usize = 50;

fn row(label: &str, value: impl std::fmt::Display) {
    println!("{label:<34}{value:>16}");
}

/// Number of `CubeData` messages needed to cover a 256-block-radius,
/// ±128-block-high view cylinder at this cube size, counted with the
/// server's streaming rule (`dx² + dz² ≤ R²`, `|dy| ≤ V`) computed locally.
fn streaming_cube_count() -> (i64, i64, u64) {
    let r = 256 / i64::from(CUBE_SIZE);
    let v = 128 / i64::from(CUBE_SIZE);
    let mut columns = 0u64;
    for dx in -r..=r {
        for dz in -r..=r {
            if dx * dx + dz * dz <= r * r {
                columns += u64::try_from(2 * v + 1).unwrap();
            }
        }
    }
    (r, v, columns)
}

fn main() {
    let region = common::Region::new(DEFAULT_SEED);
    let blocks = region.blocks();

    let meshes: Vec<(CubePos, greedy::MeshOutput)> = region
        .region_positions
        .iter()
        .map(|&pos| {
            let input = region.mesh_input(pos);
            (pos, greedy::mesh(&input, &region.appearance))
        })
        .collect();

    let uniform = meshes
        .iter()
        .filter(|(pos, _)| {
            matches!(
                region.cube(*pos).uniform(),
                Some(state) if state == BlockStateId::AIR || state == blocks.stone
            )
        })
        .count();
    let non_empty: Vec<&greedy::MeshOutput> = meshes
        .iter()
        .map(|(_, m)| m)
        .filter(|m| !m.is_empty())
        .collect();
    let total_faces: usize = non_empty.iter().map(|m| m.face_count()).sum();
    let mesh_bytes = total_faces * size_of::<PackedFace>();
    let (bytes_mean, bytes_max) = if non_empty.is_empty() {
        (0, 0)
    } else {
        let sizes: Vec<usize> = non_empty
            .iter()
            .map(|m| m.face_count() * size_of::<PackedFace>())
            .collect();
        let max = sizes.iter().copied().max().unwrap();
        (sizes.iter().sum::<usize>() / sizes.len(), max)
    };

    // Cube storage: the paletted cubes plus the HashMap entries pointing at
    // them. Per entry we estimate (slot size + 1 control byte) / 0.875 max
    // load factor — hashbrown rounds capacity to powers of two, so this is
    // an estimate, not an exact allocator figure.
    let cube_bytes: usize = region
        .region_positions
        .iter()
        .map(|&pos| region.cube(pos).heap_bytes() + size_of::<Cube>())
        .sum();
    let map_entries = region.region_positions.len();
    let map_overhead =
        (map_entries as f64 * (size_of::<(CubePos, Arc<Cube>)>() + 1) as f64 / 0.875) as usize;

    println!("Cube-size metrics: region x,z in [0,128), y in [32,128)");
    row("cube size (blocks)", CUBE_SIZE);
    row("region cubes", region.region_positions.len());
    row("uniform cubes (air/stone)", uniform);
    row("non-empty meshes (draw calls)", non_empty.len());
    row("total faces", total_faces);
    row("mesh bytes (8 per face)", mesh_bytes);
    row(
        "mesh bytes per mesh, mean/max",
        format!("{bytes_mean} / {bytes_max}"),
    );
    row("cube storage (heap + Cube)", format!("{cube_bytes} B"));
    row("  + HashMap overhead (est.)", format!("~{map_overhead} B"));
    let (r, v, messages) = streaming_cube_count();
    row(
        "streaming 256r +-128h cylinder",
        format!("R={r} V={v} -> {messages} CubeData"),
    );

    // GPU: every region mesh uploaded, one offscreen frame from above the
    // region center looking down 45 degrees, timed as the mean of 50 frames
    // each followed by a device poll wait.
    let Some(gpu) = gpu_or_skip() else {
        println!("GPU: none");
        return;
    };
    let target = OffscreenTarget::new(&gpu.device, WIDTH, HEIGHT);
    let depth = DepthTarget::new(&gpu.device, WIDTH, HEIGHT);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(WIDTH, HEIGHT);
    let (layers, _) = jbm_client::mesh::appearance::builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    for (pos, mesh) in &meshes {
        terrain.upload(&gpu, *pos, mesh);
    }

    // 45 degrees down: the camera sits as far behind the center as above it.
    let center = DVec3::new(64.0, 64.0, 64.0);
    let d = 220.0;
    let eye = center + DVec3::new(0.0, d, d);
    let to_center = center - eye;
    let yaw = f32::atan2(to_center.x as f32, -to_center.z as f32);
    let pitch = f32::atan2(
        to_center.y as f32,
        (to_center.x * to_center.x + to_center.z * to_center.z).sqrt() as f32,
    );
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), WIDTH as f32 / HEIGHT as f32);
    camera.set_yaw(yaw);
    camera.set_pitch(pitch);

    let view = target.create_view();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..FRAMES {
        let start = Instant::now();
        renderer.render(&gpu, &view, depth.view(), &camera, None, Some(&mut terrain));
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll failed");
        total += start.elapsed();
    }
    let mean_ms = total.as_secs_f64() * 1000.0 / FRAMES as f64;
    row(
        &format!("GPU frame {WIDTH}x720, 50 mean"),
        format!("{mean_ms:.3} ms"),
    );
}
