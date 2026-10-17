//! Far-look tests (task 2.9): LOD cubes draw flat per-material average
//! colors, LOD0 fades its textures into the same averages with distance,
//! and distance fog dissolves far terrain into the sky color along the
//! view ray. Pure-logic pieces (the average computation, the uniform
//! words) are unit-tested next to the code; these render offscreen and
//! assert pixels directly, not against goldens. Skipped without an
//! adapter; 32³-only (the scenes are hand-placed at 32³-local
//! coordinates).

#![cfg(not(feature = "cube16"))]

use std::sync::Arc;

use glam::{DVec3, Vec4};
use image::RgbaImage;
use jbm_client::camera::Camera;
use jbm_client::lod::CubeKey;
use jbm_client::mesh;
use jbm_client::mesh::appearance::{
    BlockAppearance, LAYER_OAK_PLANKS, LAYER_STONE, builtin_textures,
};
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::terrain::{
    FADE_END_FRACTION, FADE_START_FRACTION, FOG_END_FRACTION, FOG_START_FRACTION, TerrainPass,
};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin::Blocks};
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;

const SIZE: u32 = 256;
const FOV: f32 = 70.0_f32.to_radians();

/// A camera at `eye` looking at `look_at` (square aspect for the tests).
fn looking_at(eye: DVec3, look_at: DVec3) -> Camera {
    let d = look_at - eye;
    let yaw = f32::atan2(d.x as f32, -d.z as f32);
    let pitch = f32::atan2(d.y as f32, d.x.hypot(d.z) as f32);
    let mut camera = Camera::new(eye, FOV, 1.0);
    camera.set_yaw(yaw);
    camera.set_pitch(pitch);
    camera
}

/// Where a world point lands on the square target, in pixels (the same
/// math the GPU runs: rotation-only view-projection of the
/// camera-relative position).
fn project(camera: &Camera, world: DVec3, size: u32) -> (f32, f32) {
    let rel = camera.relative(world);
    let clip = camera.view_proj() * Vec4::new(rel.x, rel.y, rel.z, 1.0);
    (
        (clip.x / clip.w + 1.0) * 0.5 * size as f32,
        (1.0 - clip.y / clip.w) * 0.5 * size as f32,
    )
}

/// sRGB byte decode/encode, the curve the texture array and the target
/// use (mirrors the helpers in render/texture_array.rs).
fn srgb_to_linear(c: u8) -> f32 {
    let v = f32::from(c) / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> u8 {
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// WGSL smoothstep, the exact curve the shader computes.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One shared render context: gpu, target, renderer, textures, terrain.
struct Ctx {
    gpu: jbm_client::gpu::Gpu,
    target: OffscreenTarget,
    depth: DepthTarget,
    renderer: Renderer,
    textures: BlockTextures,
    terrain: TerrainPass,
}

impl Ctx {
    fn new(gpu: jbm_client::gpu::Gpu, size: u32) -> Ctx {
        let target = OffscreenTarget::new(&gpu.device, size, size);
        let depth = DepthTarget::new(&gpu.device, size, size);
        let mut renderer = Renderer::new(&gpu, target.format());
        renderer.set_viewport(size, size);
        let (layers, _) = builtin_textures();
        let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
        let terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
        Ctx {
            gpu,
            target,
            depth,
            renderer,
            textures,
            terrain,
        }
    }

    fn render(&mut self, camera: &Camera) -> RgbaImage {
        self.renderer.render(
            &self.gpu,
            FrameInputs {
                target: &self.target.create_view(),
                depth: &self.depth,
                camera,
                terrain: Some(&mut self.terrain),
                outline: None,
                overlay: None,
            },
        );
        self.target.read_rgba(&self.gpu)
    }

    /// The sky alone (no terrain), for comparing fogged pixels against the
    /// sky behind them.
    fn render_sky(&mut self, camera: &Camera) -> RgbaImage {
        self.renderer.render(
            &self.gpu,
            FrameInputs {
                target: &self.target.create_view(),
                depth: &self.depth,
                camera,
                terrain: None,
                outline: None,
                overlay: None,
            },
        );
        self.target.read_rgba(&self.gpu)
    }
}

/// Uploads and meshes one cube of `state` below local row `top_local`
/// (air above), isolated, at `level`.
fn upload_isolated(
    ctx: &mut Ctx,
    appearance: &BlockAppearance,
    state: BlockStateId,
    top_local: i32,
    level: u8,
) {
    let cube = Cube::from_fn(|p| {
        if (p.y as i32) < top_local {
            state
        } else {
            BlockStateId::AIR
        }
    });
    let key = CubeKey {
        level,
        pos: CubePos { x: 0, y: 0, z: 0 },
    };
    let input = mesh::MeshInput::isolated_at_level(Arc::new(cube), level);
    let output = mesh::mesh(&input, appearance);
    ctx.terrain.upload(&ctx.gpu, key, &output);
}

/// Task 2.9: a single LOD1 cube of one material renders its top face in
/// the layer's flat average color (times the top shade and the corner AO;
/// the isolated cube's top corners are all open, so both factors are 1).
/// No fog, no golden — a direct per-pixel assertion within ±1 LSB.
#[test]
fn lod1_cube_top_face_is_the_flat_average_color() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    let mut ctx = Ctx::new(gpu, SIZE);
    ctx.renderer.set_fog(false);

    // A stone plateau: filled below local row 8 (16 blocks at level 1),
    // so the top face sits at world y = 16 spanning x, z in 0..64.
    upload_isolated(&mut ctx, &appearance, blocks.stone, 8, 1);

    // Almost straight down onto the plateau: the sides are edge-on, so
    // the projected top face fills its bounding box.
    let camera = looking_at(DVec3::new(32.0, 140.0, 32.05), DVec3::new(32.0, 16.0, 32.0));
    let image = ctx.render(&camera);

    // The plateau's top-face corners, projected; a box shrunk 15% stays
    // strictly inside the face.
    let corners = [
        DVec3::new(0.0, 16.0, 0.0),
        DVec3::new(64.0, 16.0, 0.0),
        DVec3::new(0.0, 16.0, 64.0),
        DVec3::new(64.0, 16.0, 64.0),
    ];
    let projected: Vec<(f32, f32)> = corners.iter().map(|&c| project(&camera, c, SIZE)).collect();
    let min_x = projected.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    let max_x = projected.iter().map(|p| p.0).fold(f32::MIN, f32::max);
    let min_y = projected.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    let max_y = projected.iter().map(|p| p.1).fold(f32::MIN, f32::max);
    let shrink = |a: f32, b: f32| {
        let margin = (b - a) * 0.15;
        ((a + margin).ceil() as i32, (b - margin).floor() as i32)
    };
    let (x0, x1) = shrink(min_x, max_x);
    let (y0, y1) = shrink(min_y, max_y);

    let avg = ctx.textures.averages()[LAYER_STONE as usize];
    let expected = [
        linear_to_srgb(avg[0]),
        linear_to_srgb(avg[1]),
        linear_to_srgb(avg[2]),
    ];
    let mut checked = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let pixel = image.get_pixel(x as u32, y as u32);
            for (c, (&got, &want)) in pixel.0.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (i16::from(got) - i16::from(want)).abs() <= 1,
                    "pixel ({x}, {y}) channel {c}: {got} vs flat average {want}"
                );
            }
            checked += 1;
        }
    }
    assert!(
        checked > 1000,
        "the shrunk top-face box held only {checked} pixels"
    );
}

/// Task 2.9: a long flat LOD0 grass strip running away from the camera.
/// Near pixels keep the textured look (byte-identical to a render with
/// the fade pushed out of range), far pixels are the flat average within
/// ±2 LSB, and the fade factor recovered per row follows the analytic
/// smoothstep monotonically.
#[test]
fn lod0_grass_strip_fades_into_the_flat_average() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    // The fade window sits at distances where a 256x256 target starves
    // the mip chain (every texture is its 1x1 mip, i.e. its own average,
    // so the fade is sub-LSB by construction); 1024x1024 keeps >= 2
    // pixels per block through the window, where the fade is measurable
    // (and matches what the real 1280x720 window shows).
    const FADE_SIZE: u32 = 1024;
    let mut ctx = Ctx::new(gpu, FADE_SIZE);
    ctx.renderer.set_fog(false);

    // A wide plank sheet: 8 cubes across (x -128..128, covering the whole
    // view) by 10 cubes deep (z 0..320), each cube filled below local row
    // 16 so the surface (y = 16) has air above it inside its own cube — a
    // filled top row would have its +Y faces culled (the missing cube
    // above reads as opaque). Every column of one screen row looks at the
    // plane under the same elevation, so one row is one fade factor — and
    // averaging the row beats the sRGB quantization a single pixel
    // suffers when its mip-blurred texel sits near the layer average. The
    // sheet is oak planks, not grass: grass top is pure per-texel noise,
    // so its mips converge to the layer average before the fade window
    // opens and the mid-transition is physically sub-LSB there; the plank
    // seams are low-frequency structure the fade visibly blends.
    // `far_color` is material-agnostic.
    for x in -4..4 {
        for z in 0..10 {
            let cube = Cube::from_fn(|p| {
                if (p.y as i32) < CUBE_SIZE / 2 {
                    blocks.oak_planks
                } else {
                    BlockStateId::AIR
                }
            });
            let key = CubeKey::lod0(CubePos { x, y: 0, z });
            let input = mesh::MeshInput::isolated_at_level(Arc::new(cube), 0);
            let output = mesh::mesh(&input, &appearance);
            ctx.terrain.upload(&ctx.gpu, key, &output);
        }
    }

    // Low over one end, looking down the sheet: the rows run from ~45
    // blocks (bottom of the screen) to ~291 blocks (the sheet's far end).
    // Low, so every block in the fade window still spans >= 6 pixels —
    // higher viewpoints mip the texture toward its average before the
    // window opens, and the fade's mid-transition becomes sub-LSB (true
    // for any per-block-tiled texture; the mechanism itself is exact,
    // verified by the window's endpoints and the curve rows).
    let surface_y = 16.0_f64;
    let eye = DVec3::new(16.0, surface_y + 30.0, 290.0);
    let camera = looking_at(eye, DVec3::new(16.0, surface_y, 60.0));

    // The textured reference: both fade and fog windows far out of range.
    ctx.renderer.set_view_distance(1.0e6, 1.0e6);
    let reference = ctx.render(&camera);
    // The faded render: an LOD0 radius of 138 blocks (fade 55.2..117.3),
    // fog off — the sheet's distances (44.6..291) span the whole window.
    let lod0_radius = 138.0_f32;
    ctx.renderer.set_view_distance(lod0_radius, 1.0e6);
    let faded = ctx.render(&camera);

    let avg = ctx.textures.averages()[LAYER_OAK_PLANKS as usize];
    let avg_encoded = [
        linear_to_srgb(avg[0]),
        linear_to_srgb(avg[1]),
        linear_to_srgb(avg[2]),
    ];
    let fade_start = FADE_START_FRACTION * lod0_radius;
    let fade_end = FADE_END_FRACTION * lod0_radius;

    // Walk the rows bottom-up (y = 0 is the top row, so bottom-up means
    // descending y): the distance grows monotonically until the rays
    // leave the sheet at the horizon.
    // Stay clear of the sheet's side walls at the screen edges: their
    // fragments sit just past the fade start and round differently.
    let x_lo = FADE_SIZE / 4;
    let x_hi = FADE_SIZE - FADE_SIZE / 4;
    let mut rows = 0;
    for y in (16..FADE_SIZE - 16).rev() {
        // A row's elevation = camera pitch + the row's offset from the
        // view axis (yaw is 0, so every column's ray shares it).
        let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / FADE_SIZE as f32;
        let elevation = camera.pitch + (ndc_y * (FOV / 2.0).tan()).atan();
        if elevation >= 0.0 {
            break; // at or above the horizon: sky, not the sheet
        }
        let dist = ((surface_y - eye.y) / f64::from(elevation.sin())) as f32;
        let hit_z = 290.0 - f64::from(dist * elevation.cos());
        if !(0.0..=320.0).contains(&hit_z) || !(40.0..=300.0).contains(&dist) {
            continue;
        }

        let expected_f = smoothstep(fade_start, fade_end, dist);
        // Per pixel: the faded byte must equal the encoding of the
        // reference texel mixed with the average at the row's analytic
        // fade factor. The reference contributes an already-encoded texel
        // (half a quantum off the true one), the mix shrinks that error by
        // (1 - f), and the encode adds half a quantum — ±2 covers it. This
        // pins every row of the transition to the curve, which also makes
        // the transition monotonic along the strip (each row matches its
        // own factor, and the factor grows with the distance).
        for x in x_lo..x_hi {
            let p_ref = reference.get_pixel(x, y).0;
            let p_fad = faded.get_pixel(x, y).0;
            if expected_f <= 0.0 {
                // Near: the textured look, unchanged by the fade window
                // being moved — the mix factor is 0, so any difference is
                // a rounding flip of at most one quantum.
                for c in 0..3 {
                    assert!(
                        (i16::from(p_ref[c]) - i16::from(p_fad[c])).abs() <= 1,
                        "row {y} (dist {dist:.1}) column {x} channel {c}: {} vs textured {}",
                        p_fad[c],
                        p_ref[c]
                    );
                }
            }
            if expected_f >= 1.0 {
                // Far: the flat average within ±2 LSB (sRGB encode
                // rounding).
                for c in 0..3 {
                    assert!(
                        (i16::from(p_fad[c]) - i16::from(avg_encoded[c])).abs() <= 2,
                        "row {y} (dist {dist:.1}) column {x} channel {c}: {} vs average {}",
                        p_fad[c],
                        avg_encoded[c]
                    );
                }
            }
            for c in 0..3 {
                let lin_ref = srgb_to_linear(p_ref[c]);
                let mixed = lin_ref + (avg[c] - lin_ref) * expected_f;
                let expected_byte = linear_to_srgb(mixed);
                assert!(
                    (i16::from(p_fad[c]) - i16::from(expected_byte)).abs() <= 2,
                    "row {y} (dist {dist:.1}) column {x} channel {c}: {} vs mixed {}",
                    p_fad[c],
                    expected_byte
                );
            }
        }
        rows += 1;
    }
    assert!(rows > 100, "only {rows} rows of the strip were checked");
}

/// Task 2.9: with fog on, a terrain pixel at or beyond the fog end equals
/// the sky color behind it (the pixel's ray is above the horizon, where
/// the fog's clamped gradient is the sky's own), and a near pixel is
/// byte-identical with fog off.
#[test]
fn fog_matches_the_sky_and_leaves_near_terrain_alone() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    let mut ctx = Ctx::new(gpu, SIZE);

    // A grass sheet with its surface at y = 48 (cube row 1 filled below
    // local row 16, so the surface has air above it inside its own cube),
    // running from z = 0 to 864 and seen from below the surface looking
    // slightly up the far end: the far fragment's ray points above the
    // horizon, so the sky behind it and the fog color are the same
    // gradient value.
    for k in 0..27i32 {
        let cube = Cube::from_fn(|p| {
            if (p.y as i32) < CUBE_SIZE / 2 {
                blocks.grass_block
            } else {
                BlockStateId::AIR
            }
        });
        let key = CubeKey::lod0(CubePos { x: 0, y: 1, z: k });
        let input = mesh::MeshInput::isolated_at_level(Arc::new(cube), 0);
        let output = mesh::mesh(&input, &appearance);
        ctx.terrain.upload(&ctx.gpu, key, &output);
    }
    let camera = looking_at(DVec3::new(16.0, 24.0, 900.0), DVec3::new(16.0, 48.0, 400.0));

    // A small furthest ring radius so the fog saturates inside the scene;
    // the LOD0 radius keeps the default fade scale.
    ctx.renderer.set_view_distance(256.0, 600.0);
    assert!(ctx.renderer.fog(), "fog is on by default");
    let fogged = ctx.render(&camera);
    let sky = ctx.render_sky(&camera);
    ctx.renderer.set_fog(false);
    let clear = ctx.render(&camera);

    let fog_start = FOG_START_FRACTION * 600.0;
    let fog_end = FOG_END_FRACTION * 600.0;

    // The far point: on the sheet, beyond the fog end. Its pixel equals
    // the sky behind it within ±1 LSB.
    let far_world = DVec3::new(16.0, 48.0, 0.0);
    let far_dist = (far_world - camera.position).length();
    assert!(
        far_dist >= f64::from(fog_end),
        "far point at {far_dist} < fog end {fog_end}"
    );
    let (fx, fy) = project(&camera, far_world, SIZE);
    let far = fogged.get_pixel(fx.round() as u32, fy.round() as u32);
    let behind = sky.get_pixel(fx.round() as u32, fy.round() as u32);
    for c in 0..3 {
        assert!(
            (i16::from(far.0[c]) - i16::from(behind.0[c])).abs() <= 1,
            "far pixel channel {c}: fogged {} vs sky behind {}",
            far.0[c],
            behind.0[c]
        );
    }

    // The near point: well inside both the fade and fog starts, so the
    // pixel is byte-identical with fog off.
    let near_world = DVec3::new(16.0, 48.0, 830.0);
    let near_dist = (near_world - camera.position).length();
    assert!(
        near_dist < f64::from(fog_start) && near_dist < f64::from(FADE_START_FRACTION * 256.0),
        "near point at {near_dist} is not inside both windows"
    );
    let (nx, ny) = project(&camera, near_world, SIZE);
    let with_fog = fogged.get_pixel(nx.round() as u32, ny.round() as u32);
    let without_fog = clear.get_pixel(nx.round() as u32, ny.round() as u32);
    assert_eq!(
        with_fog.0, without_fog.0,
        "the near pixel must be unchanged by fog"
    );
}
