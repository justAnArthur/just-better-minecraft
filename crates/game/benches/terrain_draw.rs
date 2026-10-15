//! One recorded and submitted terrain frame of the cube-size region
//! (benches/common/mod.rs), followed by a device poll wait, through each
//! draw path: `terrain_draw/mdi` (multi-draw indirect) and
//! `terrain_draw/fallback` (the direct-draw loop). Without a GPU adapter
//! the bench prints a skip line and passes, so the CI bench smoke run
//! stays green.

use std::time::Instant;

use criterion::{Criterion, criterion_group, criterion_main};
use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::mesh::greedy;
use jbm_client::render::chunks::TerrainPass;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, Renderer};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_core::coords::CubePos;
use jbm_server::DEFAULT_SEED;

mod common;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn bench(c: &mut Criterion) {
    let Some(gpu) = gpu_or_skip() else {
        println!("skipping: no GPU adapter");
        return;
    };

    let region = common::Region::new(DEFAULT_SEED);
    let meshes: Vec<(CubePos, greedy::MeshOutput)> = region
        .region_positions
        .iter()
        .map(|&pos| {
            let input = region.mesh_input(pos);
            (pos, greedy::mesh(&input, &region.appearance))
        })
        .collect();

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

    // The metrics-example camera: above the region center looking down 45
    // degrees, as far behind it as above it.
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
    let mut render_frame = |terrain: &mut TerrainPass| {
        renderer.render(
            &gpu,
            &view,
            depth.view(),
            &camera,
            None,
            Some(terrain),
            None,
        );
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll failed");
    };

    let mut group = c.benchmark_group("terrain_draw");
    if terrain.uses_mdi() {
        group.bench_function("mdi", |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    render_frame(&mut terrain);
                    total += start.elapsed();
                }
                total
            })
        });
    } else {
        println!("skipping terrain_draw/mdi: adapter lacks multi-draw indirect");
    }
    terrain.set_force_fallback(true);
    group.bench_function("fallback", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                render_frame(&mut terrain);
                total += start.elapsed();
            }
            total
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
