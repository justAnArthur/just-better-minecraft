//! The LOD rings scene for task 2.6 benches and metrics (game scale;
//! task 2.10 widened it from four levels to the default six): level 0
//! and level 1 from the real generator (level 1 by the core downsampling
//! rule over level-0 data), levels 2..=5 sampled from the generator's
//! surface at each level's resolution with a solid-leaning max over the
//! child-level corner grid (the worldgen path 2.7 brings). Heights are
//! precomputed onto cached grids — one per-block grid over the level-0/1
//! footprint and coarser grids over the outer rings — so building the
//! scene costs millions of array reads, not hundreds of millions of
//! generator calls.
//! Everything is keyed by the client's `CubeKey` and pre-meshed with
//! same-level neighbors, ready to upload into a `TerrainPass`.
#![allow(dead_code)]
// Shared by terrain_draw and cube_size_metrics only; the other targets
// that compile this module (cube_size bench) use none of it.

use std::collections::HashMap;
use std::sync::Arc;

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::lod::CubeKey;
use jbm_client::mesh::{self, MeshInput, neighbor_index};
use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CUBE_SIZE, CubePos, LocalPos};
use jbm_core::cube::Cube;
use jbm_core::lod::{downsample_octant, octant_offset, parent};
use jbm_server::terrain::Generator;

/// Default `lod_view_radius`: every ring is a disk of this many cubes of
/// its own level.
const RING_RADIUS: i64 = 8;
/// Vertical half-thickness of the LOD rings, in that level's cubes
/// (default `lod_vertical_radius`).
const RING_VERTICAL: i64 = 3;

/// A cached height grid over a block span, sampled every `stride`
/// blocks (heights are read at stride multiples only).
struct HeightGrid {
    min: i64,
    stride: i64,
    side: usize,
    heights: Vec<i16>,
}

impl HeightGrid {
    fn build(generator: &Generator, min: i64, max: i64, stride: i64) -> HeightGrid {
        let side = ((max - min) / stride + 1) as usize;
        let mut heights = Vec::with_capacity(side * side);
        for i in 0..side as i64 {
            for j in 0..side as i64 {
                heights.push(generator.surface_height(min + i * stride, min + j * stride) as i16);
            }
        }
        HeightGrid {
            min,
            stride,
            side,
            heights,
        }
    }

    /// The height at `(x, z)`, which must be a stride multiple in span.
    fn at(&self, x: i64, z: i64) -> i64 {
        let i = ((x - self.min) / self.stride) as usize;
        let j = ((z - self.min) / self.stride) as usize;
        i64::from(self.heights[i * self.side + j])
    }

    /// The solid-leaning level-`level` sample: filled when the terrain
    /// pokes above the cell's top anywhere in its footprint. The max
    /// runs over the 3x3 grid at half the cell step — the corner points
    /// of the four child-level cells — so a coarser sample never sits
    /// below the finer level's sample it replaces (that ordering keeps
    /// level seams skirt- and gap-free).
    fn max_over_cell(&self, x: i64, z: i64, level: u8) -> i64 {
        let half = (1i64 << level) / 2;
        let mut best = i64::MIN;
        for i in 0..3 {
            for j in 0..3 {
                best = best.max(self.at(x + i * half, z + j * half));
            }
        }
        best
    }
}

/// Layering mirrored from the generator's surface stage (grass / dirt
/// / stone by depth, sand near sea level).
fn scene_state(blocks: &Blocks, y: i64, surface: i64) -> BlockStateId {
    if surface <= 61 {
        if y >= surface - 3 {
            blocks.sand
        } else {
            blocks.stone
        }
    } else if y == surface {
        blocks.grass_block
    } else if y >= surface - 3 {
        blocks.dirt
    } else {
        blocks.stone
    }
}

/// The rings scene: every cube pre-meshed, keyed for upload, plus a
/// camera looking down across the stack from game height.
pub struct RingsScene {
    pub meshes: Vec<(CubeKey, mesh::MeshOutput)>,
    pub camera: Camera,
    /// Per-level streamed cube count (levels 0..=5).
    pub resident: [usize; 6],
}

impl RingsScene {
    pub fn new(
        generator: &Generator,
        blocks: &Blocks,
        appearance: &jbm_client::mesh::appearance::BlockAppearance,
    ) -> RingsScene {
        let lod0_center = CubePos { x: 0, y: 2, z: 0 };
        let level_center = |level: u8| {
            let mut center = lod0_center;
            for _ in 0..level {
                center = parent(center).0;
            }
            center
        };
        let disk = |level: u8| {
            let center = level_center(level);
            let mut positions = Vec::new();
            for dy in -RING_VERTICAL..=RING_VERTICAL {
                for dz in -RING_RADIUS..=RING_RADIUS {
                    for dx in -RING_RADIUS..=RING_RADIUS {
                        if dx * dx + dz * dz <= RING_RADIUS * RING_RADIUS {
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
        };

        // Grid 1: per-block heights over the level-0/1 footprint (the
        // level-1 ring's children span ~ +-17 level-0 cubes around the
        // center cube).
        let block_span = (RING_RADIUS * 2 + 4) * i64::from(CUBE_SIZE) * 2;
        let fine = HeightGrid::build(generator, -block_span, block_span, 1);

        // Grid 2: heights over the middle rings (levels 2 and 3 sample at
        // multiples of 2 blocks), and grid 3 over the far rings (levels 4
        // and 5 sample at multiples of 8 — level 5's disk reaches
        // ±(8+2)·32·32 blocks, where a stride-2 grid would be ~200 MB).
        let mid_span = (RING_RADIUS + 2) * i64::from(CUBE_SIZE) * 8;
        let coarse = HeightGrid::build(generator, -mid_span, mid_span, 2);
        let far_span = (RING_RADIUS + 2) * i64::from(CUBE_SIZE) * 32;
        let far = HeightGrid::build(generator, -far_span, far_span, 8);

        // Level 0 cubes: exact cubes wherever the LOD0 ring and the
        // level-1 downsampling need them.
        let make_lod0 = |pos: CubePos| {
            let base = (
                i64::from(pos.x) * i64::from(CUBE_SIZE),
                i64::from(pos.y) * i64::from(CUBE_SIZE),
                i64::from(pos.z) * i64::from(CUBE_SIZE),
            );
            Arc::new(Cube::from_fn(|local: LocalPos| {
                let x = base.0 + i64::from(local.x);
                let y = base.1 + i64::from(local.y);
                let z = base.2 + i64::from(local.z);
                let h = fine.at(x, z);
                if y > h {
                    BlockStateId::AIR
                } else {
                    scene_state(blocks, y, h)
                }
            }))
        };
        let mut lod0: HashMap<CubePos, Arc<Cube>> = HashMap::new();
        for pos in disk(1) {
            for octant in 0..8u8 {
                let [dx, dy, dz] = octant_offset(octant);
                let child = CubePos {
                    x: pos.x * 2 + dx as i32,
                    y: pos.y * 2 + dy as i32,
                    z: pos.z * 2 + dz as i32,
                };
                lod0.entry(child).or_insert_with(|| make_lod0(child));
            }
        }

        // All cubes by level for meshing. Level 0 streams only its own
        // ring (default view_radius); the footprint cubes beyond it are
        // downsampling sources, not streamed cubes.
        let mut by_level: [Vec<(CubePos, Arc<Cube>)>; 6] = std::array::from_fn(|_| Vec::new());
        for pos in disk(0) {
            by_level[0].push((pos, Arc::clone(&lod0[&pos])));
        }
        for pos in disk(1) {
            let mut parent_cube = Cube::default();
            for octant in 0..8u8 {
                let [dx, dy, dz] = octant_offset(octant);
                let child = CubePos {
                    x: pos.x * 2 + dx as i32,
                    y: pos.y * 2 + dy as i32,
                    z: pos.z * 2 + dz as i32,
                };
                downsample_octant(&lod0[&child], &mut parent_cube, octant);
            }
            by_level[1].push((pos, Arc::new(parent_cube)));
        }
        for level in 2u8..=5 {
            let cell_blocks = 1i64 << level;
            let grid = if level <= 3 { &coarse } else { &far };
            for pos in disk(level) {
                let base = (
                    i64::from(pos.x) * i64::from(CUBE_SIZE) * cell_blocks,
                    i64::from(pos.y) * i64::from(CUBE_SIZE) * cell_blocks,
                    i64::from(pos.z) * i64::from(CUBE_SIZE) * cell_blocks,
                );
                let cube = Cube::from_fn(|local: LocalPos| {
                    let x = base.0 + i64::from(local.x) * cell_blocks;
                    let y = base.1 + i64::from(local.y) * cell_blocks;
                    let z = base.2 + i64::from(local.z) * cell_blocks;
                    let max_surface = grid.max_over_cell(x, z, level);
                    if y >= max_surface {
                        return BlockStateId::AIR;
                    }
                    let is_top = y + cell_blocks >= max_surface;
                    if is_top {
                        scene_state(blocks, y, max_surface)
                    } else {
                        blocks.stone
                    }
                });
                by_level[level as usize].push((pos, Arc::new(cube)));
            }
        }

        // Mesh everything with same-level neighbors.
        let mut index: [HashMap<CubePos, Arc<Cube>>; 6] = std::array::from_fn(|_| HashMap::new());
        for level in 0..6usize {
            for (pos, cube) in &by_level[level] {
                index[level].insert(*pos, Arc::clone(cube));
            }
        }
        let mut meshes = Vec::new();
        let mut resident = [0usize; 6];
        for level in 0..6u8 {
            for (pos, cube) in &by_level[level as usize] {
                let mut neighbors = std::array::from_fn(|_| None);
                for dy in -1..=1i32 {
                    for dz in -1..=1i32 {
                        for dx in -1..=1i32 {
                            if (dx, dy, dz) == (0, 0, 0) {
                                continue;
                            }
                            let neighbor = CubePos {
                                x: pos.x + dx,
                                y: pos.y + dy,
                                z: pos.z + dz,
                            };
                            if let Some(n) = index[level as usize].get(&neighbor) {
                                neighbors[neighbor_index(dx, dy, dz)] = Some(Arc::clone(n));
                            }
                        }
                    }
                }
                let input = MeshInput {
                    center: Arc::clone(cube),
                    neighbors,
                    level,
                };
                let key = CubeKey { level, pos: *pos };
                meshes.push((key, mesh::mesh(&input, appearance)));
                resident[level as usize] += 1;
            }
        }

        // The camera: game height above the center, looking down across
        // the stack.
        let eye = DVec3::new(16.0, 150.0, 380.0);
        let look = DVec3::new(60.0, 64.0, -60.0);
        let d = look - eye;
        let yaw = f32::atan2(d.x as f32, -d.z as f32);
        let pitch = f32::atan2(d.y as f32, d.x.hypot(d.z) as f32);
        let mut camera = Camera::new(eye, 70.0_f32.to_radians(), 1280.0 / 720.0);
        camera.set_yaw(yaw);
        camera.set_pitch(pitch);

        RingsScene {
            meshes,
            camera,
            resident,
        }
    }
}
