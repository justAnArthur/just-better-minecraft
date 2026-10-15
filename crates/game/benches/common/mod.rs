//! Shared terrain region for the task 1.8 cube-size bench and metrics
//! example. The region is x, z in [0, 128) and y in [32, 128) world blocks
//! (48 cubes at 32³, 384 at 16³), built locally with `Cube::from_fn` from
//! `Generator::surface_height` and the 1.5 layering, so the server's own
//! `generate` stays out of what gets measured. Cubes in the one-cube shell
//! around the region are built the same way, so border faces are realistic.

use std::collections::HashMap;
use std::sync::Arc;

use jbm_client::mesh::appearance::BlockAppearance;
use jbm_client::mesh::greedy::{MeshInput, neighbor_index};
use jbm_core::block::BlockRegistry;
use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{BlockPos, CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;
use jbm_server::Generator;

/// Sea level of the 1.5 layering: columns with h at or below it get sand.
const SAND_LEVEL: i64 = 61;

/// The 1.5 block layering, mirrored from the server's `block_at`: grass at
/// h, dirt from h-3 to h-1, stone below, sand where h <= 61, air above.
/// Per-column: only `h` and `y` matter.
fn state_at(blocks: &Blocks, h: i64, y: i64) -> BlockStateId {
    if y > h {
        BlockStateId::AIR
    } else if h <= SAND_LEVEL {
        if y >= h - 3 {
            blocks.sand
        } else {
            blocks.stone
        }
    } else if y == h {
        blocks.grass_block
    } else if y >= h - 3 {
        blocks.dirt
    } else {
        blocks.stone
    }
}

/// The measured region plus its neighbor shell, all built locally.
pub struct Region {
    generator: Generator,
    blocks: Blocks,
    pub appearance: BlockAppearance,
    /// Region cubes and the one-cube shell around them.
    cubes: HashMap<CubePos, Arc<Cube>>,
    /// Positions of the cubes inside the region (the shell is not
    /// included), sorted by `(y, z, x)`.
    pub region_positions: Vec<CubePos>,
}

impl Region {
    pub fn new(seed: u64) -> Region {
        let mut registry = BlockRegistry::new();
        let blocks = Blocks::register(&mut registry);
        let appearance = BlockAppearance::new(&registry);
        let generator = Generator::new(seed, blocks);

        // Cube range covering the block bounds, plus the one-cube shell.
        let size = i64::from(CUBE_SIZE);
        let first = [0i64, 32, 0].map(|v| v / size);
        let last = [128i64, 128, 128].map(|v| v / size);
        let mut region = Region {
            generator,
            blocks,
            appearance,
            cubes: HashMap::new(),
            region_positions: Vec::new(),
        };
        for y in (first[1] - 1)..=(last[1] - 1) {
            for z in (first[2] - 1)..=(last[2] - 1) {
                for x in (first[0] - 1)..=(last[0] - 1) {
                    let pos = CubePos {
                        x: x as i32,
                        y: y as i32,
                        z: z as i32,
                    };
                    let cube = Arc::new(region.build_cube(pos));
                    let in_region = (first[0]..last[0]).contains(&x)
                        && (first[1]..last[1]).contains(&y)
                        && (first[2]..last[2]).contains(&z);
                    if in_region {
                        region.region_positions.push(pos);
                    }
                    region.cubes.insert(pos, cube);
                }
            }
        }
        region
    }

    /// Surface height at world column `(x, z)`.
    // Bench-only; the metrics example does not call it.
    #[allow(dead_code)]
    pub fn surface_height(&self, x: i64, z: i64) -> i64 {
        self.generator.surface_height(x, z)
    }

    /// Builds one cube of the region or shell from the layering.
    pub fn build_cube(&self, pos: CubePos) -> Cube {
        let generator = &self.generator;
        let blocks = self.blocks;
        Cube::from_fn(|local| {
            let block = pos.block(local);
            let h = generator.surface_height(block.x, block.z);
            state_at(&blocks, h, block.y)
        })
    }

    /// The mesh input (center + 26 neighbors) for a region cube. Every
    /// neighbor exists: the shell covers them.
    pub fn mesh_input(&self, pos: CubePos) -> MeshInput {
        let mut neighbors = std::array::from_fn(|_| None);
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if (dx, dy, dz) == (0, 0, 0) {
                        continue;
                    }
                    let neighbor = CubePos {
                        x: pos.x + dx,
                        y: pos.y + dy,
                        z: pos.z + dz,
                    };
                    if let Some(cube) = self.cubes.get(&neighbor) {
                        neighbors[neighbor_index(dx, dy, dz)] = Some(Arc::clone(cube));
                    }
                }
            }
        }
        MeshInput {
            center: Arc::clone(&self.cubes[&pos]),
            neighbors,
        }
    }

    /// The cube containing world block `(64, h(64, 64), 64)` — the latency
    /// floor of a block edit there.
    // Bench-only; the metrics example does not call it.
    #[allow(dead_code)]
    pub fn edit_cube(&self) -> CubePos {
        let h = self.generator.surface_height(64, 64);
        BlockPos { x: 64, y: h, z: 64 }.split().0
    }

    // Example-only; the bench builds mesh inputs directly.
    #[allow(dead_code)]
    pub fn cube(&self, pos: CubePos) -> &Arc<Cube> {
        &self.cubes[&pos]
    }

    // Example-only; the bench builds mesh inputs directly.
    #[allow(dead_code)]
    pub fn blocks(&self) -> Blocks {
        self.blocks
    }
}
