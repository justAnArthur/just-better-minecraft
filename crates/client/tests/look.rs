//! The look (task 2.15): sky (sun disk, clouds, the below-horizon
//! override), water shading and MSAA, asserted on offscreen renders
//! directly rather than against goldens. Skipped without an adapter;
//! 32³-only (hand-placed scenes at 32³-local coordinates).

#![cfg(not(feature = "cube16"))]

use std::sync::Arc;

use glam::DVec3;
use image::RgbaImage;
use jbm_client::camera::Camera;
use jbm_client::gpu::Gpu;
use jbm_client::lod::CubeKey;
use jbm_client::mesh;
use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::render::renderer::sun_direction;
use jbm_client::render::settings::RenderSettings;
use jbm_client::testing::OffscreenRig;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin::Blocks};
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;

const SIZE: u32 = 128;
const FOV: f32 = 70.0_f32.to_radians();

fn rig(gpu: &Gpu, samples: u32) -> OffscreenRig {
    let settings = RenderSettings {
        msaa_samples: samples,
        ..RenderSettings::default()
    };
    OffscreenRig::with_settings(gpu, SIZE, SIZE, &settings)
}

fn render(rig: &mut OffscreenRig, gpu: &Gpu, camera: &Camera) -> RgbaImage {
    rig.render_with_sky(gpu, camera, None)
}

fn rgb(image: &RgbaImage, x: u32, y: u32) -> [u8; 3] {
    let p = image.get_pixel(x, y).0;
    [p[0], p[1], p[2]]
}

fn luminance(p: [u8; 3]) -> f32 {
    0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])
}

/// Sum of absolute channel differences.
fn distance(a: [u8; 3], b: [u8; 3]) -> i32 {
    (0..3)
        .map(|c| (i32::from(a[c]) - i32::from(b[c])).abs())
        .sum()
}

fn near(a: [u8; 3], b: [u8; 3], tolerance: i32) -> bool {
    (0..3).all(|c| (i32::from(a[c]) - i32::from(b[c])).abs() <= tolerance)
}

/// The sun disk, at the center of a view straight at the sun, is
/// brighter than the sky 20 degrees away from it (clouds off).
#[test]
fn sun_disk_is_brighter_than_the_sky_around_it() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut rig = rig(&gpu, 4);
    // Clouds off: a cloud beside the sun would be as bright as the disk.
    rig.renderer.settings_mut().clouds = false;
    let eye = DVec3::new(0.0, 80.0, 0.0);
    let sun = sun_direction().as_dvec3();
    let camera = Camera::looking_at(eye, eye + sun * 100.0, FOV, 1.0);
    let image = render(&mut rig, &gpu, &camera);
    let c = SIZE / 2;
    let disk = rgb(&image, c, c);
    // 20° off the sun along the screen's x axis (the half-FOV of 35°
    // spans SIZE / 2 pixels).
    let offset = ((20.0_f32.to_radians().tan() / (FOV / 2.0).tan()) * (SIZE / 2) as f32) as u32;
    for p in [rgb(&image, c + offset, c), rgb(&image, c - offset, c)] {
        assert!(
            luminance(disk) > luminance(p) + 25.0,
            "sun disk {disk:?} is not clearly brighter than the sky beside it {p:?}"
        );
    }
}

/// Clouds draw above the horizon only: a horizontal view renders every
/// below-horizon row byte-identically with clouds on and off, while a
/// view up into the sky does show clouds (the toggle changes pixels).
#[test]
fn clouds_never_draw_below_the_horizon() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut rig = rig(&gpu, 4);
    for yaw in [0.0_f32, 90.0, 180.0, 270.0] {
        let mut camera = Camera::new(DVec3::new(0.0, 80.0, 0.0), FOV, 1.0);
        camera.set_yaw(yaw.to_radians());
        camera.set_pitch(0.0);
        rig.renderer.settings_mut().clouds = true;
        let with = render(&mut rig, &gpu, &camera);
        rig.renderer.settings_mut().clouds = false;
        let without = render(&mut rig, &gpu, &camera);
        // Pitch 0: rows below the center look below the horizon; one row
        // of margin for the multisampled center row.
        for y in SIZE / 2 + 1..SIZE {
            for x in 0..SIZE {
                assert_eq!(
                    rgb(&with, x, y),
                    rgb(&without, x, y),
                    "yaw {yaw}: pixel ({x}, {y}) below the horizon changed with clouds"
                );
            }
        }
    }
    let mut camera = Camera::new(DVec3::new(0.0, 80.0, 0.0), FOV, 1.0);
    camera.set_pitch(60.0_f32.to_radians());
    rig.renderer.settings_mut().clouds = true;
    let with = render(&mut rig, &gpu, &camera);
    rig.renderer.settings_mut().clouds = false;
    let without = render(&mut rig, &gpu, &camera);
    let changed = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .filter(|&(x, y)| distance(rgb(&with, x, y), rgb(&without, x, y)) > 6)
        .count();
    assert!(
        changed > (SIZE * SIZE / 20) as usize,
        "only {changed} pixels show clouds looking up"
    );
}

/// The hole tests' sentinel: with the sky colors overridden, a view
/// fully below the horizon is exactly the override's ground color — no
/// sun glow, no clouds — at every pixel.
#[test]
fn below_horizon_override_is_exactly_the_override_color() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut rig = rig(&gpu, 4);
    // Toward the sun's azimuth, so its glow would show if it leaked.
    let sun = sun_direction();
    let mut camera = Camera::new(DVec3::new(0.0, 80.0, 0.0), FOV, 1.0);
    camera.set_yaw(sun.x.atan2(-sun.z));
    camera.set_pitch(-60.0_f32.to_radians());
    let magenta = [1.0, 0.0, 1.0, 1.0];
    let image = rig.render_with_sky(
        &gpu,
        &camera,
        Some(([0.5, 0.6, 0.7, 1.0], [0.2, 0.3, 0.8, 1.0], magenta)),
    );
    for (x, y, p) in image.enumerate_pixels() {
        assert_eq!(
            p.0,
            [255, 0, 255, 255],
            "pixel ({x}, {y}) is not the override ground color"
        );
    }
}

/// A sheet of water, its surface at y = 16, x in -64..64, z in 0..320.
fn upload_water_sheet(rig: &mut OffscreenRig, gpu: &Gpu) {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    let cube = Arc::new(Cube::from_fn(|p| {
        if (p.y as i32) < CUBE_SIZE / 2 {
            blocks.water
        } else {
            BlockStateId::AIR
        }
    }));
    for x in -2..2 {
        for z in 0..10 {
            let input = mesh::MeshInput::isolated_at_level(Arc::clone(&cube), 0);
            let output = mesh::mesh(&input, &appearance);
            rig.terrain
                .upload(gpu, CubeKey::lod0(CubePos { x, y: 0, z }), &output);
        }
    }
}

/// Water mirrors the sky at grazing angles (Fresnel): the water pixel at
/// the center of a grazing view is closer to the sky it reflects than the
/// water seen straight down is to its reflected sky. Away from the sun
/// (no glint), fog off (no haze), time 0.
#[test]
fn grazing_water_is_closer_to_the_sky_than_water_seen_from_above() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut rig = rig(&gpu, 4);
    rig.renderer.set_fog(false);
    upload_water_sheet(&mut rig, &gpu);

    // Returns (water pixel, sky pixel in the mirrored direction).
    let mut sample = |eye: DVec3, look_at: DVec3| {
        let camera = Camera::looking_at(eye, look_at, FOV, 1.0);
        let water = rgb(&render(&mut rig, &gpu, &camera), SIZE / 2, SIZE / 2);
        // The reflected ray at the center: the same view mirrored in the
        // water plane, seen from far above any terrain (the sheet ends
        // far below, so the center pixel is pure sky).
        let mirrored = DVec3::new(look_at.x - eye.x, eye.y - look_at.y, look_at.z - eye.z);
        let high = DVec3::new(eye.x, 5000.0, eye.z);
        let sky_camera = Camera::looking_at(high, high + mirrored, FOV, 1.0);
        let sky = rgb(&render(&mut rig, &gpu, &sky_camera), SIZE / 2, SIZE / 2);
        (water, sky)
    };
    // Grazing: 3 blocks over the surface, looking 150 blocks down the
    // sheet toward -z (away from the sun).
    let (graze_water, graze_sky) =
        sample(DVec3::new(0.0, 19.0, 300.0), DVec3::new(0.0, 16.0, 150.0));
    // From above: 40 blocks straight down (a hair off vertical).
    let (down_water, down_sky) =
        sample(DVec3::new(0.0, 56.0, 160.0), DVec3::new(0.0, 16.0, 159.95));
    let graze = distance(graze_water, graze_sky);
    let down = distance(down_water, down_sky);
    assert!(
        graze + 30 < down,
        "grazing water {graze_water:?} vs sky {graze_sky:?} (distance {graze}) is not \
         clearly closer than water from above {down_water:?} vs sky {down_sky:?} ({down})"
    );
}

/// One isolated LOD1 cube (64 blocks) at the origin holding a stone
/// box (blocks 8..56): flat-colored faces, so a silhouette is two colors at 1× and blends at 4×.
fn upload_lod1_cube(rig: &mut OffscreenRig, gpu: &Gpu) {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    // A stone box in cells 4..28 (blocks 8..56) with an air margin: a
    // filled edge would face the missing neighbors (read as opaque) and
    // be culled.
    let cube = Arc::new(Cube::from_fn(|p| {
        let inside = |v: i32| (4..28).contains(&v);
        if inside(p.x as i32) && inside(p.y as i32) && inside(p.z as i32) {
            blocks.stone
        } else {
            BlockStateId::AIR
        }
    }));
    let input = mesh::MeshInput::isolated_at_level(cube, 1);
    let output = mesh::mesh(&input, &appearance);
    let key = CubeKey {
        level: 1,
        pos: CubePos { x: 0, y: 0, z: 0 },
    };
    rig.terrain.upload(gpu, key, &output);
}

/// MSAA: the silhouette of a flat-colored cube against the sky. At 1×
/// every pixel is pure — the sky behind it or one of the flat face
/// colors; at 4× edge pixels blend the two.
#[test]
fn silhouette_edges_blend_at_4x_and_are_pure_at_1x() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let eye = DVec3::new(-60.0, 110.0, 170.0);
    let camera = Camera::looking_at(eye, DVec3::new(32.0, 32.0, 32.0), FOV, 1.0);
    let mut images = Vec::new();
    for samples in [1, 4] {
        let mut rig = rig(&gpu, samples);
        rig.renderer.set_fog(false);
        let sky = render(&mut rig, &gpu, &camera);
        upload_lod1_cube(&mut rig, &gpu);
        let scene = render(&mut rig, &gpu, &camera);
        images.push((sky, scene));
    }
    let (sky1, scene1) = &images[0];
    let (sky4, scene4) = &images[1];
    assert_eq!(sky1, sky4, "the sky alone is the same at 1x and 4x");

    let is_sky = |x: u32, y: u32, scene: &RgbaImage| near(rgb(scene, x, y), rgb(sky1, x, y), 1);
    // The face palette: non-sky 1× pixels whose 3x3 neighborhood is all
    // the same color (face interiors).
    let mut palette: Vec<[u8; 3]> = Vec::new();
    for y in 1..SIZE - 1 {
        for x in 1..SIZE - 1 {
            let p = rgb(scene1, x, y);
            if is_sky(x, y, scene1) {
                continue;
            }
            let uniform = (-1i32..=1).all(|dy| {
                (-1i32..=1).all(|dx| {
                    near(
                        rgb(scene1, (x as i32 + dx) as u32, (y as i32 + dy) as u32),
                        p,
                        1,
                    )
                })
            });
            if uniform && !palette.iter().any(|&q| near(q, p, 2)) {
                palette.push(p);
            }
        }
    }
    assert!(
        (2..=3).contains(&palette.len()),
        "expected 2-3 visible flat faces, palette {palette:?}"
    );
    let pure = |x: u32, y: u32, scene: &RgbaImage| {
        is_sky(x, y, scene) || palette.iter().any(|&q| near(rgb(scene, x, y), q, 2))
    };
    let mut blends = 0;
    for y in 0..SIZE {
        for x in 0..SIZE {
            assert!(
                pure(x, y, scene1),
                "1x pixel ({x}, {y}) {:?} is a blend",
                rgb(scene1, x, y)
            );
            if !pure(x, y, scene4) {
                blends += 1;
            }
        }
    }
    assert!(blends >= 40, "only {blends} blended edge pixels at 4x");
}
