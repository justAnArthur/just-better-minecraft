//! Task 2.12's debug views, rendered offscreen and asserted per pixel or
//! per counter: wireframe edges over shaded interiors, chunk borders in
//! the level palette, the exploded shift, hidden levels leaving holes,
//! frozen culling keeping the captured view, and the per-level counters
//! summing to the total. Every view runs on all three draw paths (the
//! terrain pass applies the settings per frame; the path is forced with
//! the test setters). Skipped without an adapter; 32³-only.

#![cfg(not(feature = "cube16"))]

use std::sync::Arc;

use glam::DVec3;
use image::RgbaImage;
use jbm_client::camera::Camera;
use jbm_client::gpu::Gpu;
use jbm_client::lod::CubeKey;
use jbm_client::mesh;
use jbm_client::mesh::appearance::{BlockAppearance, builtin_textures};
use jbm_client::render::lines::LineVertex;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::settings::RenderSettings;
use jbm_client::render::terrain::{TerrainPass, TerrainPath, lod_tint_color};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::terrain_sync::TerrainSync;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_client::world::ClientWorld;
use jbm_core::block::{BlockRegistry, builtin::Blocks};
use jbm_core::coords::CubePos;
use jbm_core::cube::Cube;
use jbm_core::protocol::ServerMsg;

const SIZE: u32 = 256;
const FOV: f32 = 70.0_f32.to_radians();

/// One shared render context, far_look's shape.
struct Ctx {
    gpu: Gpu,
    target: OffscreenTarget,
    depth: DepthTarget,
    renderer: Renderer,
    terrain: TerrainPass,
}

impl Ctx {
    fn new(gpu: Gpu) -> Ctx {
        let target = OffscreenTarget::new(&gpu.device, SIZE, SIZE);
        let depth = DepthTarget::new(&gpu.device, SIZE, SIZE);
        let mut renderer = Renderer::new(&gpu, target.format());
        renderer.set_viewport(SIZE, SIZE);
        let (layers, _) = builtin_textures();
        let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
        let terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
        Ctx {
            gpu,
            target,
            depth,
            renderer,
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

    /// The renderer's settings, mutated by each view's test.
    fn settings(&mut self) -> &mut RenderSettings {
        self.renderer.settings_mut()
    }

    /// The drawn count on the active path (the gpu path's readback is
    /// blocking, so tests see this frame's number).
    fn drawn(&mut self) -> u32 {
        self.terrain.read_cull_stats_blocking(&self.gpu).drawn()
    }

    /// Forces one of the three draw paths (skipped when unavailable) and
    /// returns whether it is active.
    fn force_path(&mut self, path: TerrainPath) -> bool {
        self.terrain
            .set_prefer_gpu_cull(path == TerrainPath::GpuCull);
        self.terrain
            .set_force_cpu_cull(path == TerrainPath::CpuCull);
        self.terrain
            .set_force_fallback(path == TerrainPath::Fallback);
        match path {
            TerrainPath::GpuCull => self.terrain.gpu_cull_available(),
            TerrainPath::CpuCull => self.terrain.uses_mdi(),
            TerrainPath::Fallback => true,
        }
    }
}

/// A cube of `state` below local row `top_local` (air above), meshed and
/// uploaded at `key`.
fn upload(
    ctx: &mut Ctx,
    appearance: &BlockAppearance,
    state: jbm_core::block::BlockStateId,
    top_local: i32,
    key: CubeKey,
) {
    let cube = Cube::from_fn(|p| {
        if (p.y as i32) < top_local {
            state
        } else {
            jbm_core::block::BlockStateId::AIR
        }
    });
    let input = mesh::MeshInput::isolated_at_level(Arc::new(cube), key.level);
    let output = mesh::mesh(&input, appearance);
    ctx.terrain.upload(&ctx.gpu, key, &output);
}

/// Where a world point lands on the square target, in pixels.
fn project(camera: &Camera, world: DVec3) -> (f32, f32) {
    let rel = camera.relative(world);
    let clip = camera.view_proj() * glam::Vec4::new(rel.x, rel.y, rel.z, 1.0);
    (
        (clip.x / clip.w + 1.0) * 0.5 * SIZE as f32,
        (1.0 - clip.y / clip.w) * 0.5 * SIZE as f32,
    )
}

/// The mean color of a small box around a pixel (the sample the
/// comparisons run on).
fn mean_color(image: &RgbaImage, (x, y): (f32, f32), r: i32) -> [f32; 3] {
    let (cx, cy) = (x as i64, y as i64);
    let mut sum = [0f64; 3];
    let mut n = 0u64;
    for dy in -i64::from(r)..=i64::from(r) {
        for dx in -i64::from(r)..=i64::from(r) {
            let (px, py) = (cx + dx, cy + dy);
            if (0..SIZE as i64).contains(&px) && (0..SIZE as i64).contains(&py) {
                let p = image.get_pixel(px as u32, py as u32);
                sum[0] += f64::from(p[0]);
                sum[1] += f64::from(p[1]);
                sum[2] += f64::from(p[2]);
                n += 1;
            }
        }
    }
    [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64].map(|v| v as f32)
}

/// Task 2.12's wireframe: dark edges along every triangle edge,
/// the shaded terrain untouched between them; wireframe-only flattens
/// everything to edges over near-black.
#[test]
fn wireframe_edges_over_untouched_interiors() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    for path in [
        TerrainPath::GpuCull,
        TerrainPath::CpuCull,
        TerrainPath::Fallback,
    ] {
        // A fresh adapter per path: Gpu is not Clone, and each path's
        // context wants its own buffers anyway.
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let mut ctx = Ctx::new(gpu);
        if !ctx.force_path(path) {
            continue;
        }
        ctx.settings().fog = false;
        // A stone plateau at level 1: one merged 64x64 top quad whose
        // border the wireframe draws (top face at world y = 16).
        upload(
            &mut ctx,
            &appearance,
            blocks.stone,
            8,
            CubeKey {
                level: 1,
                pos: CubePos { x: 0, y: 0, z: 0 },
            },
        );
        let camera = Camera::looking_at(
            DVec3::new(32.0, 140.0, 32.05),
            DVec3::new(32.0, 16.0, 32.0),
            FOV,
            1.0,
        );
        let plain = ctx.render(&camera);

        // Wireframe on: dark lines along every triangle edge. The
        // mesher splits this plateau's top into several merged quads, so
        // the assertions are aggregate: lines exist, and somewhere a
        // 12x12 window of interior keeps the plain color exactly.
        ctx.settings().wireframe = true;
        let wired = ctx.render(&camera);
        let (x0, y0) = project(&camera, DVec3::new(0.0, 16.0, 0.0));
        let (x1, y1) = project(&camera, DVec3::new(64.0, 16.0, 64.0));
        let (lo, hi) = (
            (x0.min(x1) as i32 + 6, y0.min(y1) as i32 + 6),
            (x0.max(x1) as i32 - 6, y0.max(y1) as i32 - 6),
        );
        // An edge pixel is clearly darker than the plain render there (the
        // lines are thin and blend 70% toward 0.05 linear, so antialiased
        // pixels never reach a fixed dark level).
        let is_dark = |image: &RgbaImage, x: i32, y: i32| {
            let sum = |p: &image::Rgba<u8>| p.0[..3].iter().map(|&c| i32::from(c)).sum::<i32>();
            let wired = sum(image.get_pixel(x as u32, y as u32));
            let base = sum(plain.get_pixel(x as u32, y as u32));
            base - wired > 30
        };
        let mut dark = 0;
        for y in lo.1..=hi.1 {
            for x in lo.0..=hi.0 {
                dark += u32::from(is_dark(&wired, x, y));
            }
        }
        assert!(dark > 60, "{path:?}: no wireframe edges ({dark} dark px)");
        // An interior window with no dark pixel, whose color matches the
        // plain render there.
        let mut found_clear = false;
        'clear: for y in lo.1..=hi.1 - 12 {
            for x in lo.0..=hi.0 - 12 {
                let mut clear = true;
                for dy in 0..12 {
                    for dx in 0..12 {
                        if is_dark(&wired, x + dx, y + dy) {
                            clear = false;
                            break;
                        }
                    }
                    if !clear {
                        break;
                    }
                }
                if clear {
                    let mid = ((x + 6) as f32, (y + 6) as f32);
                    let plain_mid = mean_color(&plain, mid, 2);
                    let wired_mid = mean_color(&wired, mid, 2);
                    for c in 0..3 {
                        assert!(
                            (plain_mid[c] - wired_mid[c]).abs() <= 3.0,
                            "{path:?}: interior changed with wireframe on: {plain_mid:?} -> {wired_mid:?}"
                        );
                    }
                    found_clear = true;
                    break 'clear;
                }
            }
        }
        assert!(
            found_clear,
            "{path:?}: wireframe lines cross every interior window"
        );

        // Wireframe-only: everything is gray — edges (bright), the flat
        // background (dark) and their antialiased blends — with both
        // extremes present and no textured color anywhere in the box.
        ctx.settings().wireframe_only = true;
        let flat = ctx.render(&camera);
        let mut bright = 0;
        let mut dark_px = 0;
        for y in lo.1..=hi.1 {
            for x in lo.0..=hi.0 {
                let p = flat.get_pixel(x as u32, y as u32);
                let (r, g, b) = (p[0], p[1], p[2]);
                let max = i32::from(r).max(i32::from(g)).max(i32::from(b));
                let min = i32::from(r).min(i32::from(g)).min(i32::from(b));
                assert!(
                    max - min <= 12,
                    "{path:?}: wireframe-only pixel carries color: {p:?}"
                );
                bright += u32::from(max > 200);
                // The flat fill 0.04 linear lands at sRGB ~56.
                dark_px += u32::from(max < 80);
            }
        }
        assert!(bright > 20, "{path:?}: no bright wireframe-only edges");
        assert!(dark_px > 20, "{path:?}: no flat background");
    }
}

/// Task 2.12's chunk borders: line boxes around resident cubes, colored
/// by the tint palette (white for LOD0, red for level 1).
#[test]
fn chunk_borders_draw_level_colored_boxes() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    for path in [
        TerrainPath::GpuCull,
        TerrainPath::CpuCull,
        TerrainPath::Fallback,
    ] {
        // A fresh adapter per path: Gpu is not Clone, and each path's
        // context wants its own buffers anyway.
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let mut ctx = Ctx::new(gpu);
        if !ctx.force_path(path) {
            continue;
        }
        ctx.settings().fog = false;
        // A LOD0 cube and a level-1 cube, side by side.
        let lod0 = CubeKey {
            level: 0,
            pos: CubePos { x: -2, y: 0, z: 0 },
        };
        let lod1 = CubeKey {
            level: 1,
            pos: CubePos { x: 1, y: 0, z: 0 },
        };
        upload(&mut ctx, &appearance, blocks.stone, 16, lod0);
        upload(&mut ctx, &appearance, blocks.stone, 8, lod1);
        let camera = Camera::looking_at(
            DVec3::new(16.0, 40.0, -160.0),
            DVec3::new(16.0, 24.0, 16.0),
            FOV,
            1.0,
        );

        // The border lines, exactly the way the app builds them: one
        // twelve-edge box per resident cube, tinted by level. The test's
        // alpha is 1.0 so the assertions read the pure palette colors.
        let (camera_cube, camera_frac) = jbm_client::render::terrain::camera_split(camera.position);
        let mut vertices = Vec::new();
        for key in ctx.terrain.resident_keys() {
            let min = jbm_client::render::terrain::cube_origin(key, camera_cube, camera_frac);
            let size = jbm_client::render::terrain::space::cube_size(key.level);
            let max = min + size;
            let [r, g, b] = lod_tint_color(key.level);
            push_box(&mut vertices, min, max, [r, g, b, 1.0]);
        }
        assert!(vertices.len() >= 48, "two boxes of 24 vertices each");

        let plain = ctx.render(&camera);
        ctx.renderer.set_debug_lines(&ctx.gpu, &vertices);
        let bordered = ctx.render(&camera);
        ctx.renderer.set_debug_lines(&ctx.gpu, &[]);

        // Pixels that changed carry the lines. The level-1 box's color is
        // red-dominant; LOD0's is white (all channels high together).
        let mut red = 0;
        let mut white = 0;
        for (a, b) in plain.pixels().zip(bordered.pixels()) {
            if a == b {
                continue;
            }
            let (r, g, bl) = (b[0], b[1], b[2]);
            if u32::from(r) > 140 && u32::from(g) < 90 && u32::from(bl) < 90 {
                red += 1;
            }
            if u32::from(r) > 150 && u32::from(g) > 150 && u32::from(bl) > 150 {
                white += 1;
            }
        }
        assert!(red > 10, "{path:?}: no red level-1 border ({red} px)");
        assert!(white > 10, "{path:?}: no white LOD0 border ({white} px)");
    }
}

/// One axis-aligned box's twelve edges (the app's chunk-border shape).
fn push_box(vertices: &mut Vec<LineVertex>, min: glam::Vec3, max: glam::Vec3, color: [f32; 4]) {
    let (a, b) = (min, max);
    let corners = [
        glam::Vec3::new(a.x, a.y, a.z),
        glam::Vec3::new(b.x, a.y, a.z),
        glam::Vec3::new(b.x, b.y, a.z),
        glam::Vec3::new(a.x, b.y, a.z),
        glam::Vec3::new(a.x, a.y, b.z),
        glam::Vec3::new(b.x, a.y, b.z),
        glam::Vec3::new(b.x, b.y, b.z),
        glam::Vec3::new(a.x, b.y, b.z),
    ];
    for i in 0..4 {
        segment(vertices, corners[i], corners[(i + 1) % 4], color);
        segment(vertices, corners[i + 4], corners[(i + 1) % 4 + 4], color);
        segment(vertices, corners[i], corners[i + 4], color);
    }
}

fn segment(vertices: &mut Vec<LineVertex>, from: glam::Vec3, to: glam::Vec3, color: [f32; 4]) {
    let vertex = |p: glam::Vec3| LineVertex {
        pos: [p.x, p.y, p.z, 0.0],
        color,
    };
    vertices.push(vertex(from));
    vertices.push(vertex(to));
}

/// Task 2.12's explode: a gap of 4 shifts every cube by gap × its
/// cube-grid offset — the cube at grid x=2 moves 4·2·32 = 256 blocks
/// east, leaving its old screen region to the sky.
#[test]
fn exploded_gap_shifts_a_known_cube() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    for path in [
        TerrainPath::GpuCull,
        TerrainPath::CpuCull,
        TerrainPath::Fallback,
    ] {
        // A fresh adapter per path: Gpu is not Clone, and each path's
        // context wants its own buffers anyway.
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let mut ctx = Ctx::new(gpu);
        if !ctx.force_path(path) {
            continue;
        }
        ctx.settings().fog = false;
        let key = CubeKey {
            level: 0,
            pos: CubePos { x: 2, y: 0, z: 0 },
        };
        upload(&mut ctx, &appearance, blocks.stone, 16, key);
        // The camera frames the cube's unexploded position (blocks
        // 64..96 in x, centered 80).
        let camera = Camera::looking_at(
            DVec3::new(-140.0, 90.0, 16.0),
            DVec3::new(80.0, 16.0, 16.0),
            FOV,
            1.0,
        );
        let plain = ctx.render(&camera);
        let old_center = mean_color(&plain, project(&camera, DVec3::new(80.0, 8.0, 16.0)), 3);
        assert!(
            old_center[0] > 40.0,
            "{path:?}: the cube is not visible before exploding: {old_center:?}"
        );

        // Exploded by 4: the cube's world origin becomes 64 + 4·2·32 =
        // 320, its center 336 — still inside the fog-less view.
        ctx.settings().explode = 4;
        let exploded = ctx.render(&camera);
        let vacated = mean_color(&exploded, project(&camera, DVec3::new(80.0, 8.0, 16.0)), 3);
        let moved = mean_color(&exploded, project(&camera, DVec3::new(336.0, 8.0, 16.0)), 3);
        assert!(
            vacated[2] > vacated[0] && vacated[2] > 150.0,
            "{path:?}: the old position still shows the cube: {vacated:?}"
        );
        assert!(
            moved[0] > 40.0 && (moved[2] - moved[0]).abs() < 25.0,
            "{path:?}: the shifted cube is not visible at its new position: {moved:?}"
        );
        // Nothing drew nothing: the drawn count survived the skip-cull
        // mode (one cube, one path).
        assert!(ctx.drawn() >= 1, "{path:?}: explode drew nothing");
    }
}

/// Task 2.12's shown-level mask: hiding level 0 removes exactly the
/// LOD0 pixels (with the tint on, LOD0 is white and level 1 red).
#[test]
fn hiding_level_zero_removes_lod0_pixels() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    for path in [
        TerrainPath::GpuCull,
        TerrainPath::CpuCull,
        TerrainPath::Fallback,
    ] {
        // A fresh adapter per path: Gpu is not Clone, and each path's
        // context wants its own buffers anyway.
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let mut ctx = Ctx::new(gpu);
        if !ctx.force_path(path) {
            continue;
        }
        ctx.settings().fog = false;
        ctx.settings().lod_tint = true;
        let lod0 = CubeKey {
            level: 0,
            pos: CubePos { x: -2, y: 0, z: 0 },
        };
        let lod1 = CubeKey {
            level: 1,
            pos: CubePos { x: 1, y: 0, z: 0 },
        };
        upload(&mut ctx, &appearance, blocks.stone, 16, lod0);
        upload(&mut ctx, &appearance, blocks.stone, 8, lod1);
        let camera = Camera::looking_at(
            DVec3::new(0.0, 100.0, 300.0),
            DVec3::new(0.0, 16.0, 16.0),
            FOV,
            1.0,
        );
        // LOD0 spans x -64..-32 (untinted white top); level 1 spans
        // x 32..96 (red top).
        let lod0_px = project(&camera, DVec3::new(-48.0, 16.0, 16.0));
        let lod1_px = project(&camera, DVec3::new(64.0, 16.0, 16.0));
        let full = ctx.render(&camera);
        let white = mean_color(&full, lod0_px, 3);
        let red = mean_color(&full, lod1_px, 3);
        assert!(
            white[0] > 120.0 && white[2] > 120.0,
            "{path:?}: LOD0 not white under tint: {white:?}"
        );
        assert!(
            red[0] > 100.0 && red[1] < 90.0,
            "{path:?}: level 1 not red under tint: {red:?}"
        );

        // Hide level 0: its region falls to the sky, level 1 stays.
        ctx.settings().level_mask = 0b10;
        let hidden = ctx.render(&camera);
        let vacated = mean_color(&hidden, lod0_px, 3);
        let stayed = mean_color(&hidden, lod1_px, 3);
        assert!(
            vacated[2] > vacated[0] && vacated[2] > 150.0,
            "{path:?}: the LOD0 region did not become sky: {vacated:?}"
        );
        assert!(
            (stayed[0] - red[0]).abs() < 10.0,
            "{path:?}: hiding level 0 changed level 1: {stayed:?} vs {red:?}"
        );
    }
}

/// Task 2.12's freeze culling: after a 180° rotation, the drawn count
/// matches the frozen view, not the new one — and switching the freeze
/// off follows the camera again.
#[test]
fn freeze_cull_keeps_the_frozen_view_after_rotation() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    for path in [
        TerrainPath::GpuCull,
        TerrainPath::CpuCull,
        TerrainPath::Fallback,
    ] {
        // A fresh adapter per path: Gpu is not Clone, and each path's
        // context wants its own buffers anyway.
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let mut ctx = Ctx::new(gpu);
        if !ctx.force_path(path) {
            continue;
        }
        ctx.settings().fog = false;
        // Two lone cubes east and west of a fixed camera position, tops
        // at y = 80; each 180°-apart view frames exactly one of them.
        for x in [-4i32, 4] {
            upload(
                &mut ctx,
                &appearance,
                blocks.stone,
                16,
                CubeKey {
                    level: 0,
                    pos: CubePos { x, y: 2, z: 0 },
                },
            );
        }
        let look_east = Camera::looking_at(
            DVec3::new(32.0, 60.0, 16.0),
            DVec3::new(144.0, 80.0, 16.0),
            FOV,
            1.0,
        );
        let look_west = Camera::looking_at(
            DVec3::new(32.0, 60.0, 16.0),
            DVec3::new(-80.0, 80.0, 16.0),
            FOV,
            1.0,
        );
        // Baselines: each view draws its own cube only.
        let _ = ctx.render(&look_east);
        let _ = ctx.render(&look_east);
        let drawn_east = ctx.drawn();
        let _ = ctx.render(&look_west);
        let _ = ctx.render(&look_west);
        let drawn_west = ctx.drawn();
        assert_eq!(drawn_east, 1, "{path:?}: the east view drew {drawn_east}");
        assert_eq!(drawn_west, 1, "{path:?}: the west view drew {drawn_west}");

        // Freeze while looking east, then rotate west: the count stays
        // the frozen view's.
        ctx.settings().freeze_cull = true;
        let _ = ctx.render(&look_east); // captures the frozen view
        let _ = ctx.render(&look_west);
        assert_eq!(
            ctx.drawn(),
            drawn_east,
            "{path:?}: the frozen count followed the rotation"
        );
        // And the frozen view's pixels really draw from the rotated
        // camera: the west cube is now ahead of the viewer, but the
        // cull never selected it — the frame stays the east cube's
        // colors over its (now sideways) screen position or sky.
        ctx.settings().freeze_cull = false;
        let _ = ctx.render(&look_west);
        assert_eq!(
            ctx.drawn(),
            drawn_west,
            "{path:?}: unfreezing did not restore the live view"
        );
    }
}

/// A real multi-level scene through the client world: a 2x2 LOD0 disk
/// with its level-1 parent, the rings scene's smallest case (fresh per
/// path — ClientWorld is not Clone).
fn two_level_world(blocks: &Blocks) -> ClientWorld {
    let mut world = ClientWorld::new();
    let mut children = Vec::new();
    for dx in 0..2i32 {
        for dz in 0..2i32 {
            let pos = CubePos { x: dx, y: 2, z: dz };
            let cube = Cube::from_fn(|p| {
                if (p.y as i64) < 10 {
                    blocks.stone
                } else {
                    jbm_core::block::BlockStateId::AIR
                }
            });
            world.apply(&ServerMsg::CubeData {
                pos,
                cube: Box::new(cube.clone()),
            });
            children.push((pos, cube));
        }
    }
    let mut parent_cube = Cube::default();
    let mut parent_tops = jbm_core::lod::LodTops::empty(1);
    for (pos, cube) in &children {
        let (_, octant) = jbm_core::lod::parent(*pos);
        jbm_core::lod::downsample_octant(
            cube,
            &jbm_core::lod::LodTops::LOD0,
            &mut parent_cube,
            &mut parent_tops,
            octant,
        );
    }
    world.apply(&ServerMsg::LodCubeData {
        level: 1,
        pos: CubePos { x: 0, y: 1, z: 0 },
        cube: Box::new(parent_cube),
        tops: parent_tops,
    });
    world
}

/// Task 2.12's per-level counters: the per-level drawn counts sum to the
/// frame's total on every path (the gpu path's numbers arrive through
/// the readback ring).
#[test]
fn per_level_drawn_counts_sum_to_the_total() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    for path in [
        TerrainPath::GpuCull,
        TerrainPath::CpuCull,
        TerrainPath::Fallback,
    ] {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let mut ctx = Ctx::new(gpu);
        if !ctx.force_path(path) {
            continue;
        }
        let mut sync =
            TerrainSync::with_world(two_level_world(&blocks), Arc::new(appearance.clone()), 0);
        sync.settle_sync(
            &ctx.gpu,
            &mut ctx.terrain,
            &appearance,
            DVec3::new(32.0, 60.0, 32.0),
        );
        let camera = Camera::looking_at(
            DVec3::new(32.0, 90.0, 140.0),
            DVec3::new(32.0, 40.0, 32.0),
            FOV,
            1.0,
        );
        // Two frames so the gpu readback lands, then compare.
        let _ = ctx.render(&camera);
        let _ = ctx.render(&camera);
        let total = ctx.drawn();
        assert!(total >= 1, "{path:?}: nothing drew");
        let levels = ctx.terrain.level_stats();
        let sum: u32 = levels.iter().map(|l| l.frame_drawn).sum();
        assert_eq!(
            sum, total,
            "{path:?}: per-level drawn {sum} != total {total} ({levels:?})"
        );
        let faces: u64 = levels.iter().map(|l| l.frame_faces).sum();
        assert_eq!(
            faces,
            ctx.terrain.stats().faces,
            "{path:?}: per-level faces disagree with the frame total"
        );
    }
}
