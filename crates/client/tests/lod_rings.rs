//! LOD rings scene (task 2.6), client-only and deterministic: a synthetic
//! heightfield fills levels 0..=3 around a camera exactly as rings would
//! arrive — level 0 and level 1 from real terrain (level 1 by the core
//! downsampling rule over the level-0 data, since its footprint is
//! small), levels 2 and 3 by sampling the heightfield at each level's
//! resolution with a solid-leaning corner-max rule (the worldgen path
//! 2.7 brings). The cubes go through the real client world, coverage
//! rule, mesher and renderer; eight cameras — close, zoomed and
//! steep-down views straight at the LOD0/LOD1 and LOD1/LOD2 borders and
//! a corner where three levels meet — must show only pinholes below
//! the horizon margin on every draw path (gpu-cull, cpu-cull and
//! fallback): sentinel pixels may only form 4-connected components of
//! at most 2 px each, at most 8 such px per frame (greedy-mesher
//! T-junction pinholes; any larger component is a real hole and
//! fails). The three draw paths must also render the scene
//! pixel-identically. Skipped without an adapter; 32³-only.

#![cfg(not(feature = "cube16"))]

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use glam::{DVec3, Vec3};
use image::RgbaImage;
use jbm_client::camera::Camera;
use jbm_client::lod::Coverage;
use jbm_client::lod::CubeKey;
use jbm_client::mesh::appearance::{BlockAppearance, builtin_textures};
use jbm_client::mesh::greedy;
use jbm_client::render::chunks::{TerrainPass, TerrainPath};
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::holes::{
    assert_only_pinholes, component_details, describe_holes, sentinel_components,
};
use jbm_client::testing::screenshot::{assert_matches_golden, gpu_or_skip};
use jbm_client::world::ClientWorld;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin::Blocks};
use jbm_core::coords::{BlockPos, CUBE_SIZE, CubePos, LocalPos};
use jbm_core::cube::Cube;
use jbm_core::lod::{downsample_octant, parent};
use jbm_core::protocol::ServerMsg;

const SIZE: u32 = 256;
const FOV: f32 = 70.0_f32.to_radians();

/// The linear sentinel: pure magenta. The sky below the horizon is set to
/// it, so any hole in the terrain shows it.
const SENTINEL: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// Ring radii per level, in that level's cubes. Level 0's disk is the
/// streamed world; each level above is a full disk twice as coarse.
const RING_RADIUS: [u32; 4] = [2, 3, 4, 5];
/// Vertical half-thickness of each ring, in that level's cubes.
const RING_VERTICAL: [u32; 4] = [2, 2, 1, 1];

/// The smooth synthetic heightfield: gentle sines, gradient well under
/// one block per block so the coarse (solid-leaning) samples sit above
/// the fine surface by less than the two-cell skirt cover.
fn height_at(x: i64, z: i64) -> i64 {
    let fx = x as f64;
    let fz = z as f64;
    (64.0 + 6.0 * (fx / 53.0).sin() + 4.0 * (fz / 37.0 + fx / 91.0).sin()).round() as i64
}

/// The scene's block bounds (for the height grid).
fn block_span() -> (Range<i64>, Range<i64>) {
    // Covers the level-3 ring (radius 5 cubes of 256 blocks) plus its
    // one-cube shell.
    let reach = (RING_RADIUS[3] as i64 + 2) * (i64::from(CUBE_SIZE) << 3);
    (-reach..reach, -reach..reach)
}

/// Precomputed heights over the scene's block span.
struct HeightGrid {
    min_x: i64,
    min_z: i64,
    stride: usize,
    heights: Vec<i16>,
}

impl HeightGrid {
    fn build() -> HeightGrid {
        let (xs, zs) = block_span();
        let stride = (zs.end - zs.start) as usize;
        let mut heights = Vec::with_capacity((xs.end - xs.start) as usize * stride);
        for x in xs.clone() {
            for z in zs.clone() {
                heights.push(height_at(x, z) as i16);
            }
        }
        HeightGrid {
            min_x: xs.start,
            min_z: zs.start,
            stride,
            heights,
        }
    }

    fn at(&self, x: i64, z: i64) -> i64 {
        let index = ((x - self.min_x) as usize) * self.stride + (z - self.min_z) as usize;
        i64::from(self.heights[index])
    }

    /// The solid-leaning level-`level` sample of the heightfield for the
    /// cell with block corner `(x, y, z)` (all multiples of the cell
    /// size): filled when the terrain pokes above the cell's top
    /// anywhere in its footprint. The max runs over the 3x3 grid at half
    /// the cell step — the corner points of the four child-level cells —
    /// so a coarser sample can never sit below the finer level's sample
    /// it replaces (that ordering is what keeps level seams skirt- and
    /// gap-free: the coarse surface hangs over or meets the fine one).
    fn cell_filled(&self, x: i64, y: i64, z: i64, level: u8) -> bool {
        let step = (1i64 << level).max(2);
        let half = step / 2;
        let mut max_height = i64::MIN;
        for i in 0..3 {
            for j in 0..3 {
                max_height = max_height.max(self.at(x + i * half, z + j * half));
            }
        }
        y < max_height
    }
}

/// Layering for the scene: grass on top, dirt below, stone at depth.
fn scene_state(blocks: &Blocks, y: i64, surface: i64) -> BlockStateId {
    if y == surface {
        blocks.grass_block
    } else if y >= surface - 3 {
        blocks.dirt
    } else {
        blocks.stone
    }
}

/// The exact level-0 cube at `pos`.
fn lod0_cube(grid: &HeightGrid, blocks: &Blocks, pos: CubePos) -> Cube {
    Cube::from_fn(|local: LocalPos| {
        let block = pos.block(local);
        let surface = grid.at(block.x, block.z);
        if block.y > surface {
            BlockStateId::AIR
        } else {
            scene_state(blocks, block.y, surface)
        }
    })
}

/// A level >= 2 cube sampled at its own resolution.
fn sampled_cube(grid: &HeightGrid, blocks: &Blocks, pos: CubePos, level: u8) -> Cube {
    let cell_blocks = 1i64 << level;
    let base_x = i64::from(pos.x) * i64::from(CUBE_SIZE) * cell_blocks;
    let base_y = i64::from(pos.y) * i64::from(CUBE_SIZE) * cell_blocks;
    let base_z = i64::from(pos.z) * i64::from(CUBE_SIZE) * cell_blocks;
    Cube::from_fn(|local: LocalPos| {
        let x = base_x + i64::from(local.x) * cell_blocks;
        let y = base_y + i64::from(local.y) * cell_blocks;
        let z = base_z + i64::from(local.z) * cell_blocks;
        if grid.cell_filled(x, y, z, level) {
            // Material: grass at the topmost filled cell of the column,
            // sampled at the cell's corner.
            let above_filled = grid.cell_filled(x, y + cell_blocks, z, level);
            if above_filled {
                blocks.stone
            } else {
                let corner = grid
                    .at(x, z)
                    .max(grid.at(x + cell_blocks - 1, z + cell_blocks - 1));
                scene_state(blocks, y, corner)
            }
        } else {
            BlockStateId::AIR
        }
    })
}

/// Every level-`level` cube position in that ring around the scene's
/// center, as the merged walk would visit them (any deterministic order
/// does for building; the walk order is what the server tests cover).
fn ring_positions(level: u8) -> Vec<CubePos> {
    let radius = RING_RADIUS[level as usize - 1] as i64;
    let vertical = RING_VERTICAL[level as usize - 1] as i64;
    // The level-n center containing the scene's LOD0 center cube.
    let lod0_center = CubePos { x: 0, y: 2, z: 0 };
    let mut center = lod0_center;
    for _ in 0..level {
        center = parent(center).0;
    }
    let mut positions = Vec::new();
    for dy in -vertical..=vertical {
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dz * dz <= radius * radius {
                    positions.push(CubePos {
                        x: center.x + dx as i32,
                        y: center.y + dy as i32,
                        z: center.z + dz as i32,
                    });
                }
            }
        }
    }
    positions
}

/// Builds the whole scene into a `ClientWorld` by applying protocol
/// messages in ring-arrival order (near before far, coarse before fine
/// at equal distance, like the server walk), plus the appearance table.
fn build_scene() -> (ClientWorld, BlockAppearance, Blocks) {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    let grid = HeightGrid::build();

    let mut world = ClientWorld::new();

    // Level 0: the exact terrain ring.
    let mut lod0_positions = Vec::new();
    for dx in -2i64..=2 {
        for dz in -2i64..=2 {
            if dx * dx + dz * dz <= 4 {
                for dy in -2i64..=2 {
                    lod0_positions.push(CubePos {
                        x: dx as i32,
                        y: 2 + dy as i32,
                        z: dz as i32,
                    });
                }
            }
        }
    }
    for &pos in &lod0_positions {
        let cube = lod0_cube(&grid, &blocks, pos);
        world.apply(&ServerMsg::CubeData {
            pos,
            cube: Box::new(cube),
        });
    }

    // Level 1: the real downsampling rule over the level-0 data — every
    // LOD0 cube in the level-1 ring's footprint, including the ones the
    // level-0 ring does not stream.
    let mut level1: HashMap<CubePos, Cube> = HashMap::new();
    for pos in ring_positions(1) {
        level1.insert(pos, Cube::default());
    }
    let mut children: HashMap<CubePos, Cube> = HashMap::new();
    for &pos in &lod0_positions {
        children.insert(pos, lod0_cube(&grid, &blocks, pos));
    }
    for pos in ring_positions(1) {
        // The footprint of one level-1 cube is the 8 children around
        // its own cube position; build any not already present.
        for dy in 0..2i32 {
            for dz in 0..2i32 {
                for dx in 0..2i32 {
                    let child = CubePos {
                        x: pos.x * 2 + dx,
                        y: pos.y * 2 + dy,
                        z: pos.z * 2 + dz,
                    };
                    children
                        .entry(child)
                        .or_insert_with(|| lod0_cube(&grid, &blocks, child));
                }
            }
        }
    }
    for pos in ring_positions(1) {
        let mut parent_cube = level1[&pos].clone();
        for dy in 0..2i32 {
            for dz in 0..2i32 {
                for dx in 0..2i32 {
                    let child = CubePos {
                        x: pos.x * 2 + dx,
                        y: pos.y * 2 + dy,
                        z: pos.z * 2 + dz,
                    };
                    let (_, octant) = parent(child);
                    downsample_octant(&children[&child], &mut parent_cube, octant);
                }
            }
        }
        world.apply(&ServerMsg::LodCubeData {
            level: 1,
            pos,
            cube: Box::new(parent_cube),
        });
    }

    // Levels 2 and 3: worldgen-style sampling at each level's resolution.
    for level in [2u8, 3] {
        for pos in ring_positions(level) {
            world.apply(&ServerMsg::LodCubeData {
                level,
                pos,
                cube: Box::new(sampled_cube(&grid, &blocks, pos, level)),
            });
        }
    }
    (world, appearance, blocks)
}

/// Meshes and uploads everything dirty, feeding the coverage rule, until
/// the world is clean. Returns the coverage for inspection.
fn settle(
    gpu: &jbm_client::gpu::Gpu,
    world: &mut ClientWorld,
    terrain: &mut TerrainPass,
    appearance: &BlockAppearance,
    camera: DVec3,
) -> Coverage {
    let mut coverage = Coverage::default();
    loop {
        let dirty = world.take_dirty(usize::MAX, camera);
        if dirty.is_empty() {
            break;
        }
        for (key, _version, input) in dirty {
            let output = greedy::mesh(&input, appearance);
            terrain.upload(gpu, key, &output);
            // An empty mesh counts as ready.
            coverage.set_ready(key, true);
        }
        for (key, drawn, mask) in coverage.take_changed() {
            terrain.set_draw_state(gpu, key, drawn, mask);
        }
    }
    assert_eq!(world.dirty_count(), 0, "the scene never settled");
    coverage
}

/// The screen row of a ray `angle` radians below the horizontal: for a
/// camera pitched `p`, the ray sits `angle + p` below the view axis, and
/// NDC y is exactly `-tan(angle + p) / tan(fov / 2)` for a pitch-only
/// rotation (yaw cancels), so the row follows without projecting a
/// point that may fall behind the camera.
fn row_of_angle(camera: &Camera, angle: f64, height: u32) -> f64 {
    let below_axis = angle + f64::from(camera.pitch);
    let ndc_y = -(below_axis.tan()) / (f64::from(camera.fov_y) / 2.0).tan();
    (0.5 - ndc_y * 0.5) * f64::from(height)
}

/// The checked region for the hole rule: all columns from the
/// horizon-margin row down. The margin covers everything the sky
/// legitimately shows below the horizon: the ~15 degree
/// horizon-to-ground fade (pure ground color only below `asin(0.25)`)
/// and the rings' far edge, whose drop angle is `atan(camera height
/// above the terrain / ring extent)`; plus two rows of rasterization
/// slop. Documented per camera in the test output.
fn below_margin_region(camera: &Camera, height: u32) -> (u32, u32, u32, u32) {
    let fade_rows = row_of_angle(camera, (0.25f64).asin(), height);
    let horizon_rows = row_of_angle(camera, 0.0, height);
    // The rings are a disk of this radius around the scene centre; a
    // corner ray can meet the disk edge as close as radius minus the
    // camera's horizontal offset from the centre, so the edge angle uses
    // that worst case (the sky beyond the disk is legitimate).
    let ring_radius = (RING_RADIUS[3] as f64 + 1.0) * f64::from(CUBE_SIZE) * 8.0;
    let center = DVec3::new(16.0, 0.0, 16.0);
    let horizontal = (camera.position - center).truncate().length();
    let nearest_edge = (ring_radius - horizontal).max(64.0);
    let terrain_top = f64::from(CUBE_SIZE) * 4.0; // heightfield tops ~y=76
    let edge_angle = ((camera.position.y - terrain_top) / nearest_edge).atan();
    let edge_rows = row_of_angle(camera, edge_angle, height);
    // The slack also covers the disk edge's own silhouette: the
    // outermost cubes have no far-side wall (their outward neighbors are
    // missing, so those faces are culled), and rays grazing just past
    // the edge peek under the terrain for a few rows.
    let margin = (fade_rows - horizon_rows).max(edge_rows - horizon_rows) + 8.0;
    let start_row = (horizon_rows + margin).ceil() as u32;
    (0, start_row.min(height), SIZE, height)
}

/// The eight cameras. `name`, camera, and whether this one gets a golden.
fn cameras() -> Vec<(&'static str, Camera, bool)> {
    let center = DVec3::new(16.0, f64::from(CUBE_SIZE) * 2.6, 16.0);
    let surface = DVec3::new(16.0, 68.0, 16.0);
    // 1. Close, slightly down, spanning a LOD0/LOD1 border ahead.
    let mut cams: Vec<(&'static str, Camera, bool)> = vec![(
        "close",
        looking_at(
            center + DVec3::new(24.0, 26.0, 24.0),
            surface + DVec3::new(-30.0, -6.0, -30.0),
        ),
        true,
    )];
    // 2. Zoomed (narrow fov) straight at the LOD0/LOD1 border (+X side:
    // the level-0 disk ends near x = 64..80).
    cams.push((
        "zoom_border01",
        looking_at(DVec3::new(-20.0, 96.0, 16.0), DVec3::new(70.0, 62.0, 16.0)).zoomed(0.45),
        false,
    ));
    // 3. Steep-down onto the LOD0/LOD1 border.
    cams.push((
        "down_border01",
        looking_at(DVec3::new(16.0, 130.0, 110.0), DVec3::new(60.0, 60.0, 30.0)),
        false,
    ));
    // 4. Straight at the LOD1/LOD2 border (level-1 disk ends near
    // x = 3*64 = 192..256 blocks).
    cams.push((
        "border12",
        looking_at(
            DVec3::new(60.0, 110.0, 60.0),
            DVec3::new(240.0, 60.0, -40.0),
        ),
        true,
    ));
    // 5. A corner where three levels meet: the diagonal from the center
    // crosses the LOD0/LOD1 and LOD1/LOD2 borders and the coarse grid's
    // corner columns; look down the diagonal from high up.
    cams.push((
        "corner3",
        looking_at(
            DVec3::new(-120.0, 200.0, -120.0),
            DVec3::new(280.0, 40.0, 280.0),
        ),
        true,
    ));
    // 6. Low toward the horizon across all rings.
    cams.push((
        "horizon",
        looking_at(
            DVec3::new(16.0, 92.0, 16.0),
            DVec3::new(600.0, 60.0, -500.0),
        ),
        false,
    ));
    // 7. High above, looking across the whole ring stack.
    cams.push((
        "high",
        looking_at(
            DVec3::new(-300.0, 420.0, -300.0),
            DVec3::new(80.0, 50.0, 80.0),
        ),
        false,
    ));
    // 8. From the rings' outer half back toward the centre, zoomed:
    // the coarse levels dominating the frame with fine rings at the
    // centre. (Stays inside the rings: a camera beyond the outermost
    // disk sees legitimate below-horizon sky everywhere near it.)
    cams.push((
        "outside_in",
        looking_at(DVec3::new(700.0, 190.0, 700.0), DVec3::new(0.0, 60.0, 0.0)).zoomed(0.5),
        false,
    ));
    cams
}

fn looking_at(eye: DVec3, look_at: DVec3) -> Camera {
    let d = look_at - eye;
    let yaw = f32::atan2(d.x as f32, -d.z as f32);
    let pitch = f32::atan2(d.y as f32, d.x.hypot(d.z) as f32);
    let mut camera = Camera::new(eye, FOV, 1.0);
    camera.set_yaw(yaw);
    camera.set_pitch(pitch);
    camera
}

/// Renders one frame with the sentinel sky and returns the image.
fn render_sentinel(
    gpu: &jbm_client::gpu::Gpu,
    renderer: &mut Renderer,
    target: &OffscreenTarget,
    depth: &DepthTarget,
    terrain: &mut TerrainPass,
    camera: &Camera,
) -> RgbaImage {
    // The ground color carries the sentinel; the horizon and zenith keep
    // their built-in colors so the fade math stays realistic.
    renderer.set_sky_colors(Some((
        jbm_client::render::renderer::HORIZON_COLOR,
        jbm_client::render::renderer::ZENITH_COLOR,
        SENTINEL,
    )));
    renderer.render(
        gpu,
        FrameInputs {
            target: &target.create_view(),
            depth,
            camera,
            terrain: Some(terrain),
            outline: None,
            overlay: None,
        },
    );
    let image = target.read_rgba(gpu);
    renderer.set_sky_colors(None);
    image
}

/// Where test artifacts go: the workspace `target/rings`.
fn artifact_dir() -> std::path::PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("target")
        })
        .join("rings")
}

#[test]
fn rings_scene_has_no_holes_below_the_horizon_margin() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };

    let (mut world, appearance, _blocks) = build_scene();
    let target = OffscreenTarget::new(&gpu.device, SIZE, SIZE);
    let depth = DepthTarget::new(&gpu.device, SIZE, SIZE);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(SIZE, SIZE);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);

    // Mesh from a central camera so distance ordering is realistic.
    settle(
        &gpu,
        &mut world,
        &mut terrain,
        &appearance,
        DVec3::new(16.0, 90.0, 16.0),
    );

    // Sanity: every level has drawn cubes (the coverage rule accepted
    // the rings).
    let stats = terrain.level_stats();
    assert!(stats.len() >= 4, "levels 0..=3 must be resident");
    for level in 0..=3u8 {
        let entry = &stats[level as usize];
        assert!(entry.resident > 0, "level {level} has no resident cubes");
        assert!(entry.drawn > 0, "level {level} has no drawn cubes");
        println!(
            "rings: level {level} resident {} drawn {} masked {} faces {} skirts {}",
            entry.resident, entry.drawn, entry.masked_octants, entry.faces, entry.skirts
        );
    }

    std::fs::create_dir_all(artifact_dir()).expect("create artifact dir");
    for (name, camera, golden) in cameras() {
        // The pinhole rule holds on every draw path, not just the
        // default one. Switch paths, let the two-phase state settle,
        // then check — same protocol as the pixel-identity test.
        for path in [
            TerrainPath::GpuCull,
            TerrainPath::CpuCull,
            TerrainPath::Fallback,
        ] {
            terrain.set_prefer_gpu_cull(path == TerrainPath::GpuCull);
            terrain.set_force_cpu_cull(path == TerrainPath::CpuCull);
            terrain.set_force_fallback(path == TerrainPath::Fallback);
            let available = match path {
                TerrainPath::GpuCull => terrain.gpu_cull_available(),
                TerrainPath::CpuCull => terrain.uses_mdi(),
                TerrainPath::Fallback => true,
            };
            if available {
                assert_eq!(terrain.path(), path, "rings {name}: {path:?} not active");
            }
            // Three frames so the two-phase state settles, then check.
            for _ in 0..3 {
                let _ =
                    render_sentinel(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
            }
            let image =
                render_sentinel(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
            let region = below_margin_region(&camera, image.height());
            let components = sentinel_components(&image, region);
            println!(
                "rings {name} {path:?} (active {:?}, {}): {} below margin row {}: {}",
                terrain.path(),
                if available {
                    "available"
                } else {
                    "unavailable"
                },
                describe_holes(&components),
                region.1,
                component_details(&components),
            );
            // Only pinholes: components of at most 2 px, at most 8 pinhole
            // px per frame. Anything larger — a missing skirt, a wrong
            // border shows as a long strip — fails. The pinholes
            // themselves are the greedy mesher's T-junction seams, where
            // a pixel center landing within ~1/100 px of a seam is
            // claimed by neither face.
            assert_only_pinholes(
                &image,
                region,
                &format!("rings {name} {path:?}"),
                &artifact_dir().join(format!("{name}_{}.png", path_stem(path))),
            );
        }
        // Back to the default draw path for the artifact and goldens.
        terrain.set_prefer_gpu_cull(false);
        terrain.set_force_cpu_cull(false);
        terrain.set_force_fallback(false);
        for _ in 0..3 {
            let _ = render_sentinel(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
        }
        let image = render_sentinel(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
        let path = artifact_dir().join(format!("{name}.png"));
        image.save(&path).expect("save rings image");
        if golden {
            assert_matches_golden(&format!("rings_{name}"), &image);
        }
    }
}

/// The three draw paths render the rings scene pixel-identically.
#[test]
fn rings_scene_paths_render_pixel_identical() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let (mut world, appearance, _blocks) = build_scene();
    let target = OffscreenTarget::new(&gpu.device, SIZE, SIZE);
    let depth = DepthTarget::new(&gpu.device, SIZE, SIZE);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(SIZE, SIZE);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    settle(
        &gpu,
        &mut world,
        &mut terrain,
        &appearance,
        DVec3::new(16.0, 90.0, 16.0),
    );

    let (_, camera, _) = cameras()[4].clone_name(); // corner3: three levels in view
    let mut reference: Option<(TerrainPath, RgbaImage)> = None;
    for path in [
        TerrainPath::GpuCull,
        TerrainPath::CpuCull,
        TerrainPath::Fallback,
    ] {
        terrain.set_prefer_gpu_cull(path == TerrainPath::GpuCull);
        terrain.set_force_cpu_cull(path == TerrainPath::CpuCull);
        terrain.set_force_fallback(path == TerrainPath::Fallback);
        let available = match path {
            TerrainPath::GpuCull => terrain.gpu_cull_available(),
            TerrainPath::CpuCull => terrain.uses_mdi(),
            TerrainPath::Fallback => true,
        };
        if available {
            assert_eq!(terrain.path(), path);
        }
        // Three frames so the two-phase state settles, then compare.
        for _ in 0..3 {
            let _ = render_sentinel(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
        }
        let image = render_sentinel(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
        match &reference {
            None => reference = Some((terrain.path(), image)),
            Some((reference_path, reference_image)) => {
                let differing = reference_image
                    .pixels()
                    .zip(image.pixels())
                    .filter(|(a, b)| a.0 != b.0)
                    .count();
                assert_eq!(
                    differing, 0,
                    "{path:?} vs {reference_path:?}: {differing} pixels differ"
                );
            }
        }
    }
}

/// A single LOD1 cube renders like its LOD0 pattern scaled x2, on every
/// path. The pattern is 2-aligned so the downsampling rule reproduces it
/// exactly at half resolution.
#[test]
fn a_lod1_cube_renders_like_its_pattern_scaled() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);

    // The pattern: a flat stone plateau — a simple shape whose greedy
    // faces and AO corners are exactly scale-invariant (AO comes from
    // same-level neighbors, so stepped terrain would legitimately shade
    // differently per level; a plateau has no such corners beyond the
    // rim, which scales exactly).
    let pattern = Cube::from_fn(|p| {
        if (p.y as i32) < 8 {
            blocks.stone
        } else {
            BlockStateId::AIR
        }
    });

    // The LOD1 parent of the cube at the origin: downsample the pattern
    // into octant 0 and its 7 air siblings into the rest, so the parent
    // holds exactly the pattern's coarse image.
    let mut lod1 = Cube::default();
    for octant in 0..8u8 {
        let child = if octant == 0 {
            pattern.clone()
        } else {
            Cube::default()
        };
        downsample_octant(&child, &mut lod1, octant);
    }

    let target = OffscreenTarget::new(&gpu.device, SIZE, SIZE);
    let depth = DepthTarget::new(&gpu.device, SIZE, SIZE);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(SIZE, SIZE);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");

    let mut render_at = |level: u8, cube: &Cube| -> RgbaImage {
        let mut terrain =
            TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
        let mut coverage = Coverage::default();
        let key = CubeKey {
            level,
            pos: CubePos { x: 0, y: 0, z: 0 },
        };
        let input = greedy::MeshInput::isolated_at_level(Arc::new(cube.clone()), level);
        let output = greedy::mesh(&input, &appearance);
        terrain.upload(&gpu, key, &output);
        coverage.set_ready(key, true);
        for (key, drawn, mask) in coverage.take_changed() {
            terrain.set_draw_state(&gpu, key, drawn, mask);
        }
        // Same camera for both levels: a level-1 cell is twice as wide
        // but covers the same blocks, so the coarse cube must land on
        // exactly the same pixels with the same per-block textures —
        // that is the x2 scale invariance under test.
        let camera = looking_at(DVec3::new(48.0, 52.0, 48.0), DVec3::new(14.0, 8.0, 14.0));
        let _ = level;
        renderer.render(
            &gpu,
            FrameInputs {
                target: &target.create_view(),
                depth: &depth,
                camera: &camera,
                terrain: Some(&mut terrain),
                outline: None,
                overlay: None,
            },
        );
        target.read_rgba(&gpu)
    };

    // The coarse image must reproduce the plateau exactly: filled below
    // 4 cells, air above.
    assert_eq!(lod1.get(LocalPos::new_unchecked(8, 3, 8)), blocks.stone);
    assert_eq!(
        lod1.get(LocalPos::new_unchecked(8, 4, 8)),
        BlockStateId::AIR
    );
    let lod0_image = render_at(0, &pattern);
    let lod1_image = render_at(1, &lod1);
    // Compare directly. The two renders use the same camera and the
    // same world region, so everything must match except the silhouette
    // edge: the two levels subdivide the same surface into different
    // faces, and the boundary pixel of a silhouette belongs to whichever
    // face's rasterization claims it — a couple of pixels along the
    // outline. Anything beyond that (a wrong scale, misplaced cells,
    // misaligned textures) fails the bound.
    let (width, height) = lod0_image.dimensions();
    assert_eq!(lod1_image.dimensions(), (width, height));
    let total = width as usize * height as usize;
    let mut differing = 0;
    for (a, b) in lod0_image.pixels().zip(lod1_image.pixels()) {
        if a.0.iter().zip(b.0.iter()).any(|(x, y)| x.abs_diff(*y) > 3) {
            differing += 1;
        }
    }
    assert!(
        differing * 40 <= total,
        "LOD1 does not render like the LOD0 pattern scaled x2: {differing} of {total} pixels differ (limit 2.5%)"
    );

    // Every path draws the LOD1 cube identically.
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    let mut coverage = Coverage::default();
    let key = CubeKey {
        level: 1,
        pos: CubePos { x: 0, y: 0, z: 0 },
    };
    let input = greedy::MeshInput::isolated_at_level(Arc::new(lod1.clone()), 1);
    let output = greedy::mesh(&input, &appearance);
    terrain.upload(&gpu, key, &output);
    coverage.set_ready(key, true);
    for (key, drawn, mask) in coverage.take_changed() {
        terrain.set_draw_state(&gpu, key, drawn, mask);
    }
    let camera = looking_at(DVec3::new(96.0, 104.0, 96.0), DVec3::new(28.0, 16.0, 28.0));
    let mut reference: Option<RgbaImage> = None;
    for path in [
        TerrainPath::GpuCull,
        TerrainPath::CpuCull,
        TerrainPath::Fallback,
    ] {
        terrain.set_prefer_gpu_cull(path == TerrainPath::GpuCull);
        terrain.set_force_cpu_cull(path == TerrainPath::CpuCull);
        terrain.set_force_fallback(path == TerrainPath::Fallback);
        for _ in 0..3 {
            renderer.render(
                &gpu,
                FrameInputs {
                    target: &target.create_view(),
                    depth: &depth,
                    camera: &camera,
                    terrain: Some(&mut terrain),
                    outline: None,
                    overlay: None,
                },
            );
        }
        let image = target.read_rgba(&gpu);
        if let Some(reference) = &reference {
            let differing = reference
                .pixels()
                .zip(image.pixels())
                .filter(|(a, b)| a.0 != b.0)
                .count();
            assert_eq!(differing, 0, "{path:?}: {differing} pixels differ");
        } else {
            reference = Some(image);
        }
    }
}

/// A masked octant shows the sentinel where its terrain was; a fully
/// masked cube emits no commands at all.
#[test]
fn masked_octants_hide_their_volume_and_full_masks_draw_nothing() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);

    // A stone table inside one LOD1 cube.
    let cube = Cube::from_fn(|p| {
        if (8..24).contains(&p.y) {
            blocks.stone
        } else {
            BlockStateId::AIR
        }
    });

    let target = OffscreenTarget::new(&gpu.device, SIZE, SIZE);
    let depth = DepthTarget::new(&gpu.device, SIZE, SIZE);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(SIZE, SIZE);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    let mut coverage = Coverage::default();
    let key = CubeKey {
        level: 1,
        pos: CubePos { x: 0, y: 0, z: 0 },
    };
    let input = greedy::MeshInput::isolated_at_level(Arc::new(cube), 1);
    let output = greedy::mesh(&input, &appearance);
    terrain.upload(&gpu, key, &output);
    coverage.set_ready(key, true);
    for (key, drawn, mask) in coverage.take_changed() {
        terrain.set_draw_state(&gpu, key, drawn, mask);
    }

    // Octant 0 (+x low, +z low, +y low) masked: the sentinel shows where
    // its quarter of the table was. The camera looks at the cube's
    // corner between octants.
    terrain.set_draw_state(&gpu, key, true, 0x01);
    let camera = looking_at(DVec3::new(96.0, 80.0, 96.0), DVec3::new(30.0, 30.0, 30.0));
    let image = render_sentinel(&gpu, &mut renderer, &target, &depth, &mut terrain, &camera);
    let components = sentinel_components(&image, below_margin_region(&camera, image.height()));
    assert!(
        !components.is_empty(),
        "a masked octant must reveal the sentinel"
    );

    // Fully masked: no commands on any path.
    for path in [
        TerrainPath::CpuCull,
        TerrainPath::GpuCull,
        TerrainPath::Fallback,
    ] {
        terrain.set_prefer_gpu_cull(path == TerrainPath::GpuCull);
        terrain.set_force_cpu_cull(path == TerrainPath::CpuCull);
        terrain.set_force_fallback(path == TerrainPath::Fallback);
        terrain.set_draw_state(&gpu, key, true, 0x00);
        for _ in 0..2 {
            renderer.render(
                &gpu,
                FrameInputs {
                    target: &target.create_view(),
                    depth: &depth,
                    camera: &camera,
                    terrain: Some(&mut terrain),
                    outline: None,
                    overlay: None,
                },
            );
        }
        terrain.set_draw_state(&gpu, key, true, 0xFF);
        // A few frames so the GPU counters (read back a frame or two
        // late) reflect the fully masked state.
        for _ in 0..4 {
            renderer.render(
                &gpu,
                FrameInputs {
                    target: &target.create_view(),
                    depth: &depth,
                    camera: &camera,
                    terrain: Some(&mut terrain),
                    outline: None,
                    overlay: None,
                },
            );
        }
        let stats = terrain.read_cull_stats_blocking(&gpu);

        assert_eq!(
            stats.commands, 0,
            "{path:?}: a fully masked cube emits commands"
        );
        assert_eq!(stats.drawn(), 0, "{path:?}: a fully masked cube is drawn");
    }
}

/// Small helper for the tuple-without-Clone camera.
trait ClonedName {
    fn clone_name(&self) -> (&'static str, Camera, bool);
}

impl ClonedName for (&'static str, Camera, bool) {
    fn clone_name(&self) -> (&'static str, Camera, bool) {
        (self.0, clone_camera(&self.1), self.2)
    }
}

fn clone_camera(camera: &Camera) -> Camera {
    let mut copy = Camera::new(camera.position, camera.fov_y, camera.aspect);
    copy.set_yaw(camera.yaw);
    copy.set_pitch(camera.pitch);
    copy
}

/// File-stem token for a draw path, for per-path artifact names.
fn path_stem(path: TerrainPath) -> &'static str {
    match path {
        TerrainPath::GpuCull => "gpucull",
        TerrainPath::CpuCull => "cpucull",
        TerrainPath::Fallback => "fallback",
    }
}

/// A zoomed variant of a camera (narrower fov): for the "zoomed" views.
trait Zoomed {
    fn zoomed(self, factor: f32) -> Camera;
}

impl Zoomed for Camera {
    fn zoomed(self, factor: f32) -> Camera {
        Camera {
            fov_y: self.fov_y * factor,
            ..self
        }
    }
}

/// Unused-import guards for helpers only some tests need.
#[allow(dead_code)]
fn touch(_: Vec3, _: &HeightGrid, _: BlockPos) {}
