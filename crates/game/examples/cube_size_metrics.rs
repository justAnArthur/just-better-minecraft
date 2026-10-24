//! Task 1.8 + 2.5 metrics: one aligned table describing the measured
//! terrain region (x, z in a square centered on x = z = 64 — 128 blocks
//! per side by default, or `JBM_REGION_SIDE` for game-sized runs — and y
//! in [32, 128); see `benches/common/mod.rs`) at the compiled cube size,
//! then the per-path GPU draw numbers for the surface view and a camera
//! inside a carved closed room (task 2.5): cull counters, encoded indirect
//! command slots (zero-count tails included) and the mean frame time over
//! 50 frames. Run at both sizes and compare:
//!
//! ```bash
//! cargo run -p jbm-game --release --example cube_size_metrics
//! cargo run -p jbm-game --release --example cube_size_metrics --features cube16
//! JBM_REGION_SIDE=544 cargo run -p jbm-game --release --example cube_size_metrics
//! ```

#[path = "../benches/common/mod.rs"]
mod common;

use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::gpu::Gpu;
use jbm_client::mesh;
use jbm_client::mesh::face::PackedFace;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::terrain::{TerrainPass, TerrainPath};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_core::block::BlockStateId;
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;
use jbm_server::DEFAULT_SEED;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FRAMES: usize = 50;
const WARMUP_FRAMES: usize = 3;

fn row(label: &str, value: impl std::fmt::Display) {
    println!("{label:<44}{value:>16}");
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

/// Records, submits and waits out one frame.
struct Frames {
    gpu: Gpu,
    renderer: Renderer,
    view: wgpu::TextureView,
    depth: DepthTarget,
    encode: std::time::Duration,
    wait: std::time::Duration,
}

impl Frames {
    fn render(&mut self, terrain: &mut TerrainPass, camera: &Camera) {
        let encode_start = Instant::now();
        self.renderer.render(
            &self.gpu,
            FrameInputs {
                target: &self.view,
                depth: &self.depth,
                camera,
                terrain: Some(terrain),
                outline: None,
                overlay: None,
            },
        );
        self.encode += encode_start.elapsed();
        let wait_start = Instant::now();
        self.gpu
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll failed");
        self.wait += wait_start.elapsed();
    }
}

/// The per-path block: warm up, print the cull counters, command slots and
/// the mean frame time over `FRAMES` frames.
fn measure_path(label: &str, frames: &mut Frames, terrain: &mut TerrainPass, camera: &Camera) {
    for _ in 0..WARMUP_FRAMES {
        frames.render(terrain, camera);
    }
    let stats = terrain.read_cull_stats_blocking(&frames.gpu);
    assert_eq!(
        stats.region_overflow, 0,
        "the CPU frustum bound must cover every GPU emission"
    );
    row(
        &format!("{label} {}", terrain.path_name()),
        format!(
            "drawn {} / frustum {} / occluded {}",
            stats.drawn(),
            stats.frustum_culled,
            stats.occluded
        ),
    );
    row(
        &format!("{label} {} counters", terrain.path_name()),
        format!(
            "candidates {} commands {} faces {}",
            stats.candidates, stats.commands, stats.faces
        ),
    );
    if terrain.path() == TerrainPath::GpuCull {
        row(
            &format!("{label} {} region slots (2 phases)", terrain.path_name()),
            terrain.region_command_slots(camera, 4.0 * CUBE_SIZE as f32),
        );
    }
    row(
        &format!("{label} {} draw calls / pages", terrain.path_name()),
        format!("{} / {}", terrain.stats().draw_calls, terrain.stats().pages),
    );
    let mut total = std::time::Duration::ZERO;
    frames.encode = std::time::Duration::ZERO;
    frames.wait = std::time::Duration::ZERO;
    for _ in 0..FRAMES {
        let start = Instant::now();
        frames.render(terrain, camera);
        total += start.elapsed();
    }
    let mean_ms = total.as_secs_f64() * 1000.0 / FRAMES as f64;
    let encode_ms = frames.encode.as_secs_f64() * 1000.0 / FRAMES as f64;
    let wait_ms = frames.wait.as_secs_f64() * 1000.0 / FRAMES as f64;
    row(
        &format!("{label} {} frame mean ({FRAMES})", terrain.path_name()),
        format!("{mean_ms:.3} ms"),
    );
    row(
        &format!("{label} {} encode / poll-wait mean", terrain.path_name()),
        format!("{encode_ms:.3} / {wait_ms:.3} ms"),
    );
}

fn main() {
    let mut region = common::Region::new(DEFAULT_SEED);
    let blocks = region.blocks();

    let meshes: Vec<(CubePos, mesh::MeshOutput)> = region
        .region_positions
        .iter()
        .map(|&pos| {
            let input = region.mesh_input(pos);
            (pos, mesh::mesh(&input, &region.appearance))
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
    let non_empty: Vec<&mesh::MeshOutput> = meshes
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

    let (x_range, y_range) = region.block_bounds();
    println!(
        "Cube-size metrics: region x,z in [{},{}), y in [{},{})",
        x_range.start, x_range.end, y_range.start, y_range.end
    );
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

    // GPU: every region mesh uploaded into paged face storage, then
    // offscreen frames per draw path — the surface view (above the region
    // center looking down 45 degrees) and the cave view (inside the carved
    // closed room, task 2.5).
    let Some(gpu) = gpu_or_skip() else {
        println!("GPU: none");
        return;
    };
    let target = OffscreenTarget::new(&gpu.device, WIDTH, HEIGHT);
    let depth = DepthTarget::new(&gpu.device, WIDTH, HEIGHT);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(WIDTH, HEIGHT);
    let textures = BlockTextures::builtin(&gpu).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    for (pos, mesh) in &meshes {
        terrain.upload(&gpu, jbm_client::lod::CubeKey::lod0(*pos), mesh);
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
    let mut surface_camera = Camera::new(eye, 70.0_f32.to_radians(), WIDTH as f32 / HEIGHT as f32);
    surface_camera.set_yaw(yaw);
    surface_camera.set_pitch(pitch);

    // The cave scene: same region with the closed room carved under the
    // surface, camera standing in the room (carve_cave asserts the room is
    // closed and the camera sits in air).
    region.carve_cave();
    let cave_meshes: Vec<(CubePos, mesh::MeshOutput)> = region
        .region_positions
        .iter()
        .map(|&pos| {
            let input = region.mesh_input(pos);
            (pos, mesh::mesh(&input, &region.appearance))
        })
        .collect();
    let (_, ys, _) = region.cave_room();
    let mut cave_camera = Camera::new(
        DVec3::new(63.5, ys.start as f64 + 2.5, 63.5),
        70.0_f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
    );
    cave_camera.set_pitch(0.0);

    let mut frames = Frames {
        gpu,
        renderer,
        view: target.create_view(),
        depth,
        encode: std::time::Duration::ZERO,
        wait: std::time::Duration::ZERO,
    };

    // Surface view, every path that exists. gpu-cull is opt-in in the game
    // (JBM_GPU_CULL=1; it lost to cpu-cull on Metal), so select it
    // explicitly like a JBM_GPU_CULL=1 run would.
    if terrain.gpu_cull_available() {
        terrain.set_prefer_gpu_cull(true);
        assert_eq!(terrain.path(), TerrainPath::GpuCull);
        measure_path("surface", &mut frames, &mut terrain, &surface_camera);
        terrain.set_prefer_gpu_cull(false);
    } else {
        row("surface gpu-cull", "unavailable on this adapter");
    }
    terrain.set_force_cpu_cull(true);
    measure_path("surface", &mut frames, &mut terrain, &surface_camera);
    terrain.set_force_cpu_cull(false);
    terrain.set_force_fallback(true);
    measure_path("surface", &mut frames, &mut terrain, &surface_camera);
    terrain.set_force_fallback(false);

    // Cave view: cpu-cull reference, then the occlusion-culled path.
    let mut cave_pass = TerrainPass::new(
        &frames.gpu,
        target.format(),
        frames.renderer.uniform_layout(),
        &textures,
    );
    for (pos, mesh) in &cave_meshes {
        cave_pass.upload(&frames.gpu, jbm_client::lod::CubeKey::lod0(*pos), mesh);
    }
    cave_pass.set_force_cpu_cull(true);
    measure_path("cave", &mut frames, &mut cave_pass, &cave_camera);
    if cave_pass.gpu_cull_available() {
        cave_pass.set_force_cpu_cull(false);
        cave_pass.set_prefer_gpu_cull(true);
        assert_eq!(cave_pass.path(), TerrainPath::GpuCull);
        measure_path("cave", &mut frames, &mut cave_pass, &cave_camera);
        cave_pass.set_prefer_gpu_cull(false);
    } else {
        row("cave gpu-cull", "unavailable on this adapter");
    }

    // Task 2.6: the LOD rings scene at game scale, per path and per
    // level.
    println!("\nrings scene (task 2.6, default lod_view_radius, 6 levels since task 2.10)");
    let mut registry = jbm_core::block::BlockRegistry::new();
    let blocks = jbm_core::block::builtin::Blocks::register(&mut registry);
    let generator = jbm_server::terrain::Generator::with_preset(
        DEFAULT_SEED,
        blocks,
        jbm_server::worldgen::Preset::Classic,
    );
    let rings = common::rings::RingsScene::new(&generator, &blocks, &region.appearance);
    let mut coverage = jbm_client::lod::Coverage::default();
    let upload_rings =
        |frames: &mut Frames, coverage: &mut jbm_client::lod::Coverage| -> TerrainPass {
            let mut pass = TerrainPass::new(
                &frames.gpu,
                target.format(),
                frames.renderer.uniform_layout(),
                &textures,
            );
            for (key, mesh) in &rings.meshes {
                pass.upload(&frames.gpu, *key, mesh);
                coverage.set_ready(*key, true);
            }
            for (key, drawn, mask) in coverage.take_changed() {
                pass.set_draw_state(&frames.gpu, key, drawn, mask);
            }
            pass
        };
    let mut rings_pass = upload_rings(&mut frames, &mut coverage);
    for stats in rings_pass.level_stats() {
        println!(
            "  level {}: resident {:>6} drawn {:>6} masked {:>5} faces {:>8} skirts {:>5} face bytes {:>10}",
            stats.level,
            stats.resident,
            stats.drawn,
            stats.masked_octants,
            stats.faces,
            stats.skirts,
            stats.faces * 8
        );
    }
    rings_pass.set_force_cpu_cull(true);
    measure_path("rings", &mut frames, &mut rings_pass, &rings.camera);
    if rings_pass.gpu_cull_available() {
        rings_pass.set_force_cpu_cull(false);
        rings_pass.set_prefer_gpu_cull(true);
        assert_eq!(rings_pass.path(), TerrainPath::GpuCull);
        measure_path("rings", &mut frames, &mut rings_pass, &rings.camera);
        rings_pass.set_prefer_gpu_cull(false);
    } else {
        row("rings gpu-cull", "unavailable on this adapter");
    }
}
