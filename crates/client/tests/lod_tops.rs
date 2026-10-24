//! Height-exact LOD on screen (task 2.14): a level-2 column top that is
//! shorter than its cell renders at its exact block height — the
//! silhouette row matches the projection of `quad_corners`, the CPU
//! mirror of the shader's corner math — and the step between two partial
//! tops at the same cell row is closed (no sentinel ground shows
//! through). Skipped without an adapter; 32³-only.

#![cfg(not(feature = "cube16"))]

use std::sync::Arc;

use glam::{DVec3, Vec4};
use image::RgbaImage;
use jbm_client::camera::Camera;
use jbm_client::lod::{Coverage, CubeKey};
use jbm_client::mesh::appearance::{BlockAppearance, builtin_textures};
use jbm_client::mesh::face::{Direction, quad_corners, unpack};
use jbm_client::mesh::{self, MeshInput, MeshOutput};
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::terrain::TerrainPass;
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::testing::holes::is_sentinel;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin::Blocks};
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;
use jbm_core::lod::LodTops;

const SIZE: u32 = 256;
const FOV: f32 = 70.0_f32.to_radians();
const LEVEL: u8 = 2;
/// The linear sentinel: pure magenta.
const SENTINEL: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

/// A level-2 plateau over the whole cube, cells y <= 3 (blocks 0..16),
/// with column tops `h(x, z)` in the top cell row.
fn plateau(blocks: &Blocks, h: impl Fn(u8, u8) -> u8) -> (Cube, LodTops) {
    let cube = Cube::from_fn(|p| {
        if p.y <= 3 {
            blocks.stone
        } else {
            BlockStateId::AIR
        }
    });
    let mut tops = LodTops::empty(LEVEL);
    for z in 0..CUBE_SIZE as u8 {
        for x in 0..CUBE_SIZE as u8 {
            tops.set_half(0, x, z, h(x, z));
        }
    }
    tops.compact();
    (cube, tops)
}

/// Meshes the plateau at the origin with air around it (a missing cube
/// below).
fn mesh_plateau(appearance: &BlockAppearance, cube: Cube, tops: LodTops) -> MeshOutput {
    let air = Arc::new(Cube::default());
    let mut input = MeshInput::isolated_at_level(Arc::new(cube), LEVEL);
    for dy in 0..=1 {
        for dz in -1..=1 {
            for dx in -1..=1 {
                if (dx, dy, dz) != (0, 0, 0) {
                    input.neighbors[mesh::neighbor_index(dx, dy, dz)] = Some(Arc::clone(&air));
                }
            }
        }
    }
    input.tops.center = Some(Arc::new(tops));
    mesh::mesh(&input, appearance)
}

/// Renders one mesh as the level-2 cube at the origin with the given sky
/// colors (horizon, zenith, ground).
fn render(
    output: &MeshOutput,
    camera: &Camera,
    sky: ([f32; 4], [f32; 4], [f32; 4]),
) -> Option<RgbaImage> {
    let gpu = gpu_or_skip()?;
    let target = OffscreenTarget::new(&gpu.device, SIZE, SIZE);
    let depth = DepthTarget::new(&gpu.device, SIZE, SIZE);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(SIZE, SIZE);
    let (layers, _) = builtin_textures();
    let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
    let mut terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
    let mut coverage = Coverage::default();
    let key = CubeKey {
        level: LEVEL,
        pos: CubePos { x: 0, y: 0, z: 0 },
    };
    terrain.upload(&gpu, key, output);
    coverage.set_ready(key, true);
    for (key, drawn, mask) in coverage.take_changed() {
        terrain.set_draw_state(&gpu, key, drawn, mask);
    }
    renderer.set_sky_colors(Some(sky));
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
    Some(target.read_rgba(&gpu))
}

/// The image row a world point projects to.
fn project_row(camera: &Camera, world: DVec3) -> f32 {
    let rel = camera.relative(world);
    let clip = camera.view_proj() * Vec4::new(rel.x, rel.y, rel.z, 1.0);
    let ndc_y = clip.y / clip.w;
    (1.0 - ndc_y) * 0.5 * SIZE as f32
}

#[test]
fn a_partial_top_renders_at_its_exact_height() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    // Every column top is 3 of the cell's 4 blocks: the surface sits at
    // block 15, one block under the cell boundary.
    let (cube, tops) = plateau(&blocks, |_, _| 3);
    let output = mesh_plateau(&appearance, cube, tops);

    // The -Z wall's top edge, from the CPU mirror of the shader corners.
    let wall = output
        .faces
        .iter()
        .map(|&f| unpack(f))
        .find(|f| f.direction == Direction::NegZ)
        .expect("a -Z wall");
    let top_y = quad_corners(&wall, LEVEL)
        .iter()
        .map(|c| c[1])
        .max()
        .unwrap();
    assert_eq!(top_y, 15, "{wall:?}");

    // Eye below the top, facing the wall: the wall's top edge is the
    // silhouette against an all-sentinel sky.
    let camera = Camera::looking_at(
        DVec3::new(64.0, 5.0, -30.0),
        DVec3::new(64.0, 5.0, 64.0),
        FOV,
        1.0,
    );
    let Some(image) = render(&output, &camera, (SENTINEL, SENTINEL, SENTINEL)) else {
        return;
    };
    let column = SIZE / 2;
    let silhouette = (0..SIZE)
        .find(|&row| !is_sentinel(image.get_pixel(column, row)))
        .expect("the wall is on screen") as f32;
    let expected = project_row(&camera, DVec3::new(64.0, f64::from(top_y), 0.0));
    let cell_top = project_row(&camera, DVec3::new(64.0, 16.0, 0.0));
    assert!(
        expected - cell_top > 4.0,
        "one block must span several pixels ({cell_top} vs {expected})"
    );
    assert!(
        (silhouette - expected).abs() <= 1.0,
        "silhouette at row {silhouette}, the exact height projects to {expected} (the cell top to {cell_top})"
    );
}

#[test]
fn a_step_between_two_partial_tops_leaves_no_hole() {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    // Columns x < 10 top out 3 blocks into the cell row, the rest 1: a
    // 2-block step at block x = 40 inside one cell row, between two
    // opaque cells the ordinary rule culls (and off the octant plane,
    // where a skirt would cover it anyway).
    let (cube, tops) = plateau(&blocks, |x, _| if x < 10 { 3 } else { 1 });
    let output = mesh_plateau(&appearance, cube, tops);
    let steps = output
        .faces
        .iter()
        .filter(|&&f| unpack(f).inset_bottom > 0)
        .count();
    assert!(steps > 0, "the step has faces");

    // From the low side, looking down at the step; every ray below the
    // horizon lands on the plateau, so any sentinel pixel is a hole.
    let camera = Camera::looking_at(
        DVec3::new(52.0, 40.0, 64.0),
        DVec3::new(40.0, 14.0, 64.0),
        FOV,
        1.0,
    );
    let sky = (
        jbm_client::render::renderer::HORIZON_COLOR,
        jbm_client::render::renderer::ZENITH_COLOR,
        SENTINEL,
    );
    let Some(image) = render(&output, &camera, sky) else {
        return;
    };
    let holes = image.pixels().filter(|p| is_sentinel(p)).count();
    assert_eq!(holes, 0, "{holes} sentinel pixels around the step");

    // The check sees holes: without the step faces the gap shows the
    // sentinel ground.
    let mut open = output.clone();
    open.faces.retain(|&f| unpack(f).inset_bottom == 0);
    open.ranges = {
        let mut ranges: [std::ops::Range<u32>; 6] = std::array::from_fn(|_| 0..0);
        let mut start = 0u32;
        for (di, range) in ranges.iter_mut().enumerate() {
            let count = open
                .faces
                .iter()
                .filter(|&&f| unpack(f).direction as usize == di)
                .count() as u32;
            *range = start..start + count;
            start += count;
        }
        ranges
    };
    let image = render(&open, &camera, sky).expect("adapter found above");
    let holes = image.pixels().filter(|p| is_sentinel(p)).count();
    assert!(
        holes > 20,
        "removing the step must open a hole ({holes} px)"
    );
}
