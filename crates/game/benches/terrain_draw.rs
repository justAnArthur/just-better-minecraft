//! One recorded and submitted terrain frame of the cube-size region
//! (benches/common/mod.rs), followed by a device poll wait, through each
//! draw path (task 2.5):
//!
//! - `terrain_draw/gpu_cull`: compute frustum + Hi-Z occlusion culling,
//!   two-phase `multi_draw_indirect` (the default path where available).
//! - `terrain_draw/mdi`: the CPU-culled `multi_draw_indirect` path (2.2).
//! - `terrain_draw/fallback`: the direct-draw loop.
//! - `terrain_draw/cave/cpu_cull` and `terrain_draw/cave/gpu_cull`: a
//!   camera inside a carved closed room under the surface, where the
//!   Hi-Z occlusion culling earns its keep.
//!
//! Every bench warms up 3 frames first, so the two-phase occlusion state
//! is in its steady state before timing. Without a GPU adapter the bench
//! prints a skip line and passes, so the CI bench smoke run stays green.

use std::time::Instant;

use criterion::measurement::WallTime;
use criterion::{Criterion, criterion_group, criterion_main};
use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::gpu::Gpu;
use jbm_client::mesh::greedy;
use jbm_client::render::chunks::TerrainPass;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_core::coords::CubePos;
use jbm_server::DEFAULT_SEED;

mod common;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const WARMUP_FRAMES: usize = 3;

/// One uploadable scene: meshes plus its camera.
struct Scene {
    meshes: Vec<(CubePos, greedy::MeshOutput)>,
    camera: Camera,
}

/// The surface scene: the metrics-example camera, above the region center
/// looking down 45 degrees, as far behind it as above it.
fn surface_scene(region: &common::Region) -> Scene {
    let meshes = region_meshes(region);
    let center = DVec3::new(64.0, 64.0, 64.0);
    let d = 220.0;
    let eye = center + DVec3::new(0.0, d, d);
    let d_to_center = center - eye;
    let yaw = f32::atan2(d_to_center.x as f32, -d_to_center.z as f32);
    let pitch = f32::atan2(
        d_to_center.y as f32,
        (d_to_center.x * d_to_center.x + d_to_center.z * d_to_center.z).sqrt() as f32,
    );
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), WIDTH as f32 / HEIGHT as f32);
    camera.set_yaw(yaw);
    camera.set_pitch(pitch);
    Scene { meshes, camera }
}

/// The cave scene: the region with the closed room carved under the
/// surface, camera standing in the middle of the room looking horizontally.
/// `carve_cave` asserts the camera really is in the closed room.
fn cave_scene(region: &mut common::Region) -> Scene {
    region.carve_cave();
    let meshes = region_meshes(region);
    let (_, ys, _) = region.cave_room();
    let eye = DVec3::new(63.5, ys.start as f64 + 2.5, 63.5);
    let mut camera = Camera::new(eye, 70.0_f32.to_radians(), WIDTH as f32 / HEIGHT as f32);
    camera.set_pitch(0.0); // looking horizontally down the room's axis
    Scene { meshes, camera }
}

fn region_meshes(region: &common::Region) -> Vec<(CubePos, greedy::MeshOutput)> {
    region
        .region_positions
        .iter()
        .map(|&pos| {
            let input = region.mesh_input(pos);
            (pos, greedy::mesh(&input, &region.appearance))
        })
        .collect()
}

/// Records, submits and waits out one frame.
struct Frames {
    gpu: Gpu,
    renderer: Renderer,
    view: wgpu::TextureView,
    depth: DepthTarget,
}

impl Frames {
    fn render(&mut self, terrain: &mut TerrainPass, camera: &Camera) {
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
        self.gpu
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll failed");
    }

    /// Warm up so the occlusion state is steady, then time the frame.
    fn bench(
        &mut self,
        group: &mut criterion::BenchmarkGroup<'_, WallTime>,
        name: &str,
        terrain: &mut TerrainPass,
        camera: &Camera,
    ) {
        for _ in 0..WARMUP_FRAMES {
            self.render(terrain, camera);
        }
        let stats = terrain.read_cull_stats_blocking(&self.gpu);
        assert_eq!(
            stats.region_overflow, 0,
            "the CPU frustum bound must cover every GPU emission"
        );
        println!(
            "{name}: path {} candidates {} frustum-culled {} occluded {} drawn {} commands {} overflow {}",
            terrain.path_name(),
            stats.candidates,
            stats.frustum_culled,
            stats.occluded,
            stats.drawn(),
            stats.commands,
            stats.region_overflow,
        );
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    self.render(terrain, camera);
                    total += start.elapsed();
                }
                total
            })
        });
    }
}

fn bench(c: &mut Criterion) {
    let Some(gpu) = gpu_or_skip() else {
        println!("skipping: no GPU adapter");
        return;
    };

    let mut region = common::Region::new(DEFAULT_SEED);
    let surface = surface_scene(&region);
    let cave = cave_scene(&mut region);

    let target = OffscreenTarget::new(&gpu.device, WIDTH, HEIGHT);
    let depth = DepthTarget::new(&gpu.device, WIDTH, HEIGHT);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(WIDTH, HEIGHT);
    let (layers, _) = jbm_client::mesh::appearance::builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");

    // Every bench gets its own pass (each path forces its own flags); grab
    // what the passes need before `gpu`/`renderer` move into `frames`.
    // The layout clone is a ref-count bump and outlives the closure.
    let uniform_layout = renderer.uniform_layout().clone();
    let format = target.format();
    let make_pass = |gpu: &jbm_client::gpu::Gpu, meshes: &[(CubePos, greedy::MeshOutput)]| {
        let mut pass = TerrainPass::new(gpu, format, &uniform_layout, &textures);
        for (pos, mesh) in meshes {
            pass.upload(gpu, *pos, mesh);
        }
        pass
    };

    let mut frames = Frames {
        gpu,
        renderer,
        view: target.create_view(),
        depth,
    };

    let mut group = c.benchmark_group("terrain_draw");

    // The default path first: compute frustum + Hi-Z culling, two MDI
    // phases. gpu_cull_available covers MDI + compute + R32Float storage.
    let mut gpu_pass = make_pass(&frames.gpu, &surface.meshes);
    if gpu_pass.gpu_cull_available() {
        frames.bench(&mut group, "gpu_cull", &mut gpu_pass, &surface.camera);
    } else {
        println!("skipping terrain_draw/gpu_cull: adapter lacks the GPU cull path");
    }

    // 2.2's CPU-culled MDI path and the direct-draw fallback.
    let mut mdi_pass = make_pass(&frames.gpu, &surface.meshes);
    mdi_pass.set_force_cpu_cull(true);
    frames.bench(&mut group, "mdi", &mut mdi_pass, &surface.camera);

    let mut fallback_pass = make_pass(&frames.gpu, &surface.meshes);
    fallback_pass.set_force_fallback(true);
    frames.bench(&mut group, "fallback", &mut fallback_pass, &surface.camera);

    // The cave: a camera inside a closed room under the surface. The
    // occlusion culling should drop most of the frustum there.
    let mut cave_cpu = make_pass(&frames.gpu, &cave.meshes);
    cave_cpu.set_force_cpu_cull(true);
    frames.bench(&mut group, "cave/cpu_cull", &mut cave_cpu, &cave.camera);

    let mut cave_gpu = make_pass(&frames.gpu, &cave.meshes);
    if cave_gpu.gpu_cull_available() {
        frames.bench(&mut group, "cave/gpu_cull", &mut cave_gpu, &cave.camera);
    } else {
        println!("skipping terrain_draw/cave/gpu_cull: adapter lacks the GPU cull path");
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
