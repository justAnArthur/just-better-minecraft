//! Shared terrain region for the task 1.8 cube-size bench and metrics
//! example. The region spans y in [32, 128) world blocks and x, z in a
//! square centered on x = z = 64: 128 blocks per side by default
//! (`[0, 128)`, 48 cubes at 32³, 384 at 16³), or the positive multiple of
//! the cube size given by `JBM_REGION_SIDE` so the benches and metrics can
//! run game-sized regions (e.g. `JBM_REGION_SIDE=544` for 17x3x17 cubes at
//! 32³). It is built locally with `Cube::from_fn` from
//! `Generator::surface_height` and the 1.5 layering, so the server's own
//! `generate` stays out of what gets measured. Cubes in the one-cube shell
//! around the region are built the same way, so border faces are realistic.

use std::collections::HashMap;
use std::ops::Range;
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
    /// The x (and z) world-block range of the region.
    // Read only by `block_bounds`, which only the metrics example calls.
    #[allow(dead_code)]
    x_range: Range<i64>,
}

/// The horizontal side of the region, in world blocks: 128 by default, or
/// `JBM_REGION_SIDE` (a positive multiple of the cube size) so the metrics
/// example and benches can run game-sized regions.
fn region_side() -> i64 {
    let Some(value) = std::env::var_os("JBM_REGION_SIDE") else {
        return 128;
    };
    let side = value
        .to_str()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or_else(|| panic!("JBM_REGION_SIDE must be an integer, got {value:?}"));
    assert!(
        side > 0 && side % i64::from(CUBE_SIZE) == 0,
        "JBM_REGION_SIDE must be a positive multiple of the cube size ({CUBE_SIZE}), got {side}"
    );
    side
}

impl Region {
    pub fn new(seed: u64) -> Region {
        let mut registry = BlockRegistry::new();
        let blocks = Blocks::register(&mut registry);
        let appearance = BlockAppearance::new(&registry);
        let generator = Generator::new(seed, blocks);

        // Cube range covering the block bounds, plus the one-cube shell.
        // The horizontal range is cube-aligned and centered (to the cube)
        // on x = z = 64: 128 blocks per side reproduce [0, 128) exactly,
        // and any positive multiple of the cube size keeps whole cubes.
        let size = i64::from(CUBE_SIZE);
        let cubes_per_axis = region_side() / size;
        let first_x = 64i64.div_euclid(size) - cubes_per_axis / 2;
        let x_range = first_x * size..(first_x + cubes_per_axis) * size;
        let first = [first_x, 32 / size, first_x];
        let last = [
            first_x + cubes_per_axis,
            128 / size,
            first_x + cubes_per_axis,
        ];
        let mut region = Region {
            generator,
            blocks,
            appearance,
            cubes: HashMap::new(),
            region_positions: Vec::new(),
            x_range,
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

    /// The region's world-block bounds: x and z span the same range
    /// (128 wide by default, `JBM_REGION_SIDE` wide when set, centered on
    /// x = z = 64), y spans [32, 128).
    // Example-only; the benches do not print the bounds.
    #[allow(dead_code)]
    pub fn block_bounds(&self) -> (Range<i64>, Range<i64>) {
        (self.x_range.clone(), 32..128)
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

    /// The closed 8x5x8 air room about 20 blocks under the surface at the
    /// region center (task 2.5 cave scenes): interior block bounds.
    // Task 2.5 cave scenes only; the cube_size bench does not use them.
    #[allow(dead_code)]
    pub fn cave_room(&self) -> (Range<i64>, Range<i64>, Range<i64>) {
        let ceiling = self.generator.surface_height(64, 64) - 20;
        (60..68, ceiling - 5..ceiling, 60..68)
    }

    /// Carves the room into the region and its shell, rebuilding every
    /// cube that overlaps it. Panics unless the one-block shell around the
    /// interior is solid stone afterwards (the room is closed).
    // Task 2.5 cave scenes only; the cube_size bench does not use them.
    #[allow(dead_code)]
    pub fn carve_cave(&mut self) {
        let room = self.cave_room();
        let (rx, ry, rz) = &room;
        let size = i64::from(CUBE_SIZE);
        let first = [
            rx.start / size - 1,
            ry.start / size - 1,
            rz.start / size - 1,
        ];
        let last = [rx.end / size + 1, ry.end / size + 1, rz.end / size + 1];
        for y in first[1]..=last[1] {
            for z in first[2]..=last[2] {
                for x in first[0]..=last[0] {
                    let pos = CubePos {
                        x: x as i32,
                        y: y as i32,
                        z: z as i32,
                    };
                    let generator = &self.generator;
                    let blocks = self.blocks;
                    let cube = Cube::from_fn(|local| {
                        let block = pos.block(local);
                        if rx.contains(&block.x) && ry.contains(&block.y) && rz.contains(&block.z) {
                            BlockStateId::AIR
                        } else {
                            let h = generator.surface_height(block.x, block.z);
                            state_at(&blocks, h, block.y)
                        }
                    });
                    self.cubes.insert(pos, Arc::new(cube));
                }
            }
        }
        // The room is closed: every shell block around the interior is
        // solid (stone at that depth), and the interior is air.
        for y in ry.start - 1..=ry.end {
            for z in rz.start - 1..=rz.end {
                for x in rx.start - 1..=rx.end {
                    let inside = rx.contains(&x) && ry.contains(&y) && rz.contains(&z);
                    let pos = BlockPos { x, y, z };
                    let (cube_pos, local) = pos.split();
                    let state = self.cubes[&cube_pos].get(local);
                    if inside {
                        assert_eq!(state, BlockStateId::AIR, "room block {pos:?}");
                    } else {
                        assert_ne!(state, BlockStateId::AIR, "room shell {pos:?} leaks");
                    }
                }
            }
        }
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
