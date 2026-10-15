//! Terrain from the declarative worldgen graph: a [`Generator`] that runs
//! the compiled shipped graph ([`crate::worldgen::overworld`]) and turns
//! its density into 32³ cubes. The graph format, node table and
//! determinism rule are documented in the crate README's "Worldgen graph"
//! section; the module docs of [`crate::worldgen`] restate the rule.
//!
//! Terrain shape is data now: `final_density >= 0` means solid, and the
//! surface stage below decides which solid block a cell becomes. The task
//! 1.5 code this replaces lives on as [`crate::worldgen::reference`] for
//! the equivalence tests and the comparison bench.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;

use crate::worldgen::{self, CompiledGraph};

/// Sea-adjacent level: a solid cell whose surface sits at or below it gets
/// sand instead of grass and dirt. There is no water yet (later tasks);
/// this only changes the surface blocks.
const SAND_LEVEL: i64 = 61;

/// Documented height bounds of the shipped graph: `64 ± 28`, so `[36, 92]`.
/// Every column of the world stays inside this range by construction (the
/// noise deviation is clamped before rounding); the graph's own interval
/// bounds prove it (see the worldgen equivalence tests).
pub const MIN_HEIGHT: i64 = 36;
pub const MAX_HEIGHT: i64 = 92;

/// How many cells above a cube the surface stage may need to look at: `d`
/// is capped at 4, so cells up to `y + 4` decide the top rows' blocks.
const SURFACE_LOOKAHEAD: usize = 4;

/// The block a solid cell becomes, given `d`, the number of consecutive
/// solid cells directly above it (capped at [`SURFACE_LOOKAHEAD`]).
///
/// Equivalence with the heightmap layering of
/// `worldgen::reference::column_block(y, h)`: for a pure heightmap
/// density, solid means `y <= h`, so the run above a solid cell is
/// `h - y` (when at most 4) and the surface of the run is `y + d = h`.
/// Then:
///
/// - `d <= 3 && y + d <= 61` is `h <= 61 && y >= h - 3`: sand, exactly the
///   beach/seabed branch;
/// - else `d == 0` is `y == h` with `h > 61`: grass;
/// - else `d <= 3` is `h - 3 <= y < h` with `h > 61`: dirt;
/// - else (`d == 4`, i.e. `y <= h - 4`): stone.
///
/// Air cells (`y > h`) are not solid, so they stay air. The 3D-graph case
/// (overhangs) is the natural generalization of the same rules.
#[inline]
fn surface_block(solid: bool, d: u8, y: i64, blocks: &Blocks) -> BlockStateId {
    if !solid {
        BlockStateId::AIR
    } else if d <= 3 && y + i64::from(d) <= SAND_LEVEL {
        blocks.sand
    } else if d == 0 {
        blocks.grass_block
    } else if d <= 3 {
        blocks.dirt
    } else {
        blocks.stone
    }
}

/// How many `(cx, cz)` columns the generator may cache. Bounded, so memory
/// stays flat whatever order requests arrive in.
const COLUMN_CACHE_CAPACITY: usize = 64;

/// The per-column data cached for one cube column — today's heightmap
/// cache, generalized. `Bits` (banded graphs, the shipped one included)
/// holds one solidity bit per band row, `ceil(rows / 64)` words per column
/// in `z * 32 + x` order; `Live` (graphs whose band the interval bounds
/// cannot prove) holds the live column-stage values and keeps evaluating
/// rows on demand.
enum ColumnData {
    Bits(Arc<[u64]>),
    Live(Arc<[f64]>),
}

/// Solidity bit `row` of a band's word slice.
#[inline]
fn bit_at(bits: &[u64], row: usize) -> bool {
    bits[row / 64] & (1 << (row % 64)) != 0
}

/// Generates terrain cubes from a compiled density graph. Output is a pure
/// function of `(seed, graph, pos)`, whatever order cubes are requested in
/// and however often; the per-column cache only ever stores results of
/// that same pure function, so it cannot change any answer.
pub struct Generator {
    seed: u64,
    blocks: Blocks,
    graph: Arc<CompiledGraph>,
    column_cache: Mutex<HashMap<(i32, i32), ColumnData>>,
}

impl Generator {
    /// A generator running the shipped overworld graph.
    pub fn new(seed: u64, blocks: Blocks) -> Self {
        Self {
            seed,
            blocks,
            graph: worldgen::overworld(),
            column_cache: Mutex::new(HashMap::new()),
        }
    }

    /// A generator running any compiled graph, for tests and (from Stage 3)
    /// mods.
    pub fn from_graph(seed: u64, blocks: Blocks, graph: &CompiledGraph) -> Self {
        Self {
            seed,
            blocks,
            graph: Arc::new(graph.clone()),
            column_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Surface height (block y of the grass/sand layer) at world column
    /// `(x, z)`: the highest solid y, found from the column's xz stage plus
    /// a scan over y. Banded graphs (the shipped one included) scan their
    /// own provable band, so the answer is exact; for the shipped graph it
    /// is always within [`MIN_HEIGHT`, `MAX_HEIGHT`]. Unbanded graphs scan
    /// `[MIN_HEIGHT, MAX_HEIGHT]` only, and yield `MIN_HEIGHT - 1` when
    /// nothing in that range is solid.
    pub fn surface_height(&self, x: i64, z: i64) -> i64 {
        let graph = self.graph.as_ref();
        match graph.band() {
            Some(band) => {
                let mut regs = vec![0.0; graph.register_count()];
                let mut bits = vec![0u64; band.rows().div_ceil(64)];
                self.eval_band_bits(graph, band, x, z, &mut regs, &mut bits);
                // Rows at or above `high` are provably air and rows at or
                // below `low` provably solid, so the graph's own band bounds
                // the scan (not the shipped graph's height constants).
                for y in (band.low() + 1..band.high()).rev() {
                    if bit_at(&bits, (y - band.low() - 1) as usize) {
                        return y;
                    }
                }
                band.low()
            }
            None => {
                let mut regs = vec![0.0; graph.register_count()];
                graph.eval_column(self.seed, x, z, &mut regs);
                for y in (MIN_HEIGHT..=MAX_HEIGHT).rev() {
                    if graph.eval_cell_density(self.seed, x, y, z, &mut regs) >= 0.0 {
                        return y;
                    }
                }
                MIN_HEIGHT - 1
            }
        }
    }

    /// The spawn position over column `(0, 0)`: centered on the block, two
    /// above the surface.
    pub fn spawn(&self) -> [f64; 3] {
        [0.5, (self.surface_height(0, 0) + 2) as f64, 0.5]
    }

    /// Evaluates the band's uncertainty bits for one column: the column
    /// stage once, then one cell-stage evaluation per band row.
    fn eval_band_bits(
        &self,
        graph: &CompiledGraph,
        band: &worldgen::Band,
        x: i64,
        z: i64,
        regs: &mut [f64],
        bits: &mut [u64],
    ) {
        graph.eval_column(self.seed, x, z, regs);
        for row in 0..band.rows() {
            let y = band.low() + 1 + row as i64;
            let solid = graph.eval_cell_density(self.seed, x, y, z, regs) >= 0.0;
            if solid {
                bits[row / 64] |= 1 << (row % 64);
            }
        }
    }

    /// The cached per-column data of the 32² columns covered by cube column
    /// `(cx, cz)`, computed at most once thanks to the bounded cache.
    /// Consecutive cubes of one column (streaming works bottom-up-ish per
    /// column) hit the cache, so noise never runs twice per column.
    fn column_data(&self, cx: i32, cz: i32) -> ColumnData {
        if let Some(data) = self.column_cache.lock().unwrap().get(&(cx, cz)) {
            return match data {
                ColumnData::Bits(bits) => ColumnData::Bits(Arc::clone(bits)),
                ColumnData::Live(live) => ColumnData::Live(Arc::clone(live)),
            };
        }
        let graph = self.graph.as_ref();
        let base_x = i64::from(cx) * i64::from(CUBE_SIZE);
        let base_z = i64::from(cz) * i64::from(CUBE_SIZE);
        let data = match graph.band() {
            Some(band) => {
                let words = band.rows().div_ceil(64);
                let mut regs = vec![0.0; graph.register_count()];
                let mut column = vec![0u64; words];
                let mut bits = vec![0u64; (CUBE_SIZE * CUBE_SIZE) as usize * words];
                for z in 0..CUBE_SIZE {
                    for x in 0..CUBE_SIZE {
                        column.iter_mut().for_each(|w| *w = 0);
                        self.eval_band_bits(
                            graph,
                            band,
                            base_x + i64::from(x),
                            base_z + i64::from(z),
                            &mut regs,
                            &mut column,
                        );
                        let at = ((z * CUBE_SIZE + x) as usize) * words;
                        bits[at..at + words].copy_from_slice(&column);
                    }
                }
                ColumnData::Bits(Arc::from(bits))
            }
            None => {
                let live = graph.live_xz();
                let mut regs = vec![0.0; graph.register_count()];
                let mut stage = Vec::with_capacity((CUBE_SIZE * CUBE_SIZE) as usize * live.len());
                for z in 0..CUBE_SIZE {
                    for x in 0..CUBE_SIZE {
                        graph.eval_column(
                            self.seed,
                            base_x + i64::from(x),
                            base_z + i64::from(z),
                            &mut regs,
                        );
                        for &reg in live {
                            stage.push(regs[reg as usize]);
                        }
                    }
                }
                ColumnData::Live(Arc::from(stage))
            }
        };
        let mut cache = self.column_cache.lock().unwrap();
        if cache.len() >= COLUMN_CACHE_CAPACITY {
            // Stay bounded: drop the whole (small) working set rather than
            // growing without limit; streaming asks by column, so it refills.
            cache.clear();
        }
        let shared = match &data {
            ColumnData::Bits(bits) => ColumnData::Bits(Arc::clone(bits)),
            ColumnData::Live(live) => ColumnData::Live(Arc::clone(live)),
        };
        cache.insert((cx, cz), shared);
        data
    }

    /// The terrain cube at `pos`. Fast paths from the graph's interval
    /// bounds (seed-independent): a cube whose density maximum is below 0 is
    /// uniform air; one whose minimum — over the cube extended 4 cells up,
    /// so every surface-stage run is provably saturated — is at or above 0
    /// is uniform stone. Mixed cubes evaluate per column for the cube's rows
    /// plus [`SURFACE_LOOKAHEAD`] above (never peeking at neighbour cubes),
    /// then build in a single [`Cube::from_fn`] pass, so there is no
    /// per-block `set` palette churn.
    pub fn generate(&self, pos: CubePos) -> Cube {
        let graph = self.graph.as_ref();
        let bottom = i64::from(pos.y) * i64::from(CUBE_SIZE);
        let top = bottom + i64::from(CUBE_SIZE) - 1;
        let base_x = i64::from(pos.x) * i64::from(CUBE_SIZE);
        let base_z = i64::from(pos.z) * i64::from(CUBE_SIZE);
        let (stone_min, _) = graph.bounds(
            [base_x, bottom, base_z],
            [
                base_x + i64::from(CUBE_SIZE) - 1,
                top + SURFACE_LOOKAHEAD as i64,
                base_z + i64::from(CUBE_SIZE) - 1,
            ],
        );
        if stone_min >= 0.0 {
            return Cube::filled(self.blocks.stone);
        }
        let (_, air_max) = graph.bounds(
            [base_x, bottom, base_z],
            [
                base_x + i64::from(CUBE_SIZE) - 1,
                top,
                base_z + i64::from(CUBE_SIZE) - 1,
            ],
        );
        if air_max < 0.0 {
            return Cube::default();
        }

        // Mixed path. Per column: one fused downward pass over the cube's
        // rows plus the lookahead. `d` tracks the run of consecutive solid
        // cells above the current row, capped at [`SURFACE_LOOKAHEAD`]; the
        // pass starts above the cube's top, where the cap makes the unknown
        // run beyond the lookahead invisible. Cube cells are filled in
        // `(y·32 + z)·32 + x` index order — the order `from_fn` reads — so
        // the final pass is a plain lookup.
        let data = self.column_data(pos.x, pos.z);
        let rows = CUBE_SIZE as usize + SURFACE_LOOKAHEAD;
        let mut flat =
            vec![BlockStateId::AIR; CUBE_SIZE as usize * CUBE_SIZE as usize * CUBE_SIZE as usize];
        let blocks = self.blocks;
        let mut fill_column = |x: i32, z: i32, solid: &mut dyn FnMut(i64) -> bool| {
            let mut d: u8 = 0;
            for j in (0..rows).rev() {
                let y = bottom + j as i64;
                let is_solid = solid(y);
                if j < CUBE_SIZE as usize {
                    let index =
                        (j * CUBE_SIZE as usize + z as usize) * CUBE_SIZE as usize + x as usize;
                    flat[index] = surface_block(is_solid, d, y, &blocks);
                }
                d = if is_solid {
                    (d + 1).min(SURFACE_LOOKAHEAD as u8)
                } else {
                    0
                };
            }
        };
        match (&data, graph.band()) {
            (ColumnData::Bits(bits), Some(band)) => {
                let words = band.rows().div_ceil(64);
                for z in 0..CUBE_SIZE {
                    for x in 0..CUBE_SIZE {
                        let at = ((z * CUBE_SIZE + x) as usize) * words;
                        let column = &bits[at..at + words];
                        fill_column(x, z, &mut |y| {
                            if y <= band.low() {
                                true
                            } else if y >= band.high() {
                                false
                            } else {
                                bit_at(column, (y - band.low() - 1) as usize)
                            }
                        });
                    }
                }
            }
            (ColumnData::Live(live), None) => {
                let n_live = graph.live_xz().len();
                let mut regs = vec![0.0; graph.register_count()];
                for z in 0..CUBE_SIZE {
                    for x in 0..CUBE_SIZE {
                        let at = ((z * CUBE_SIZE + x) as usize) * n_live;
                        graph.restore_column(&mut regs, &live[at..at + n_live]);
                        fill_column(x, z, &mut |y| {
                            graph.eval_cell_density(
                                self.seed,
                                base_x + i64::from(x),
                                y,
                                base_z + i64::from(z),
                                &mut regs,
                            ) >= 0.0
                        });
                    }
                }
            }
            _ => unreachable!("column data and band always agree"),
        }
        let mut cells = flat.into_iter();
        Cube::from_fn(move |_| cells.next().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::{Generator, MAX_HEIGHT, MIN_HEIGHT};
    use jbm_core::block::builtin::Blocks;
    use jbm_core::block::{BlockRegistry, BlockStateId};
    use jbm_core::coords::{CubePos, LocalPos};
    use jbm_core::cube::Cube;
    use std::collections::HashMap;

    fn generator(seed: u64) -> Generator {
        let mut registry = BlockRegistry::new();
        let blocks = Blocks::register(&mut registry);
        Generator::new(seed, blocks)
    }

    fn state_ids() -> Blocks {
        let mut registry = BlockRegistry::new();
        Blocks::register(&mut registry)
    }

    #[test]
    fn generating_the_same_cube_twice_gives_equal_cubes() {
        let terrain = generator(42);
        for pos in [
            CubePos { x: 0, y: 2, z: 0 },
            CubePos { x: -3, y: -1, z: 7 },
            CubePos {
                x: 100,
                y: 0,
                z: -100,
            },
        ] {
            assert_eq!(terrain.generate(pos), terrain.generate(pos));
        }
    }

    #[test]
    fn generation_is_order_independent() {
        let terrain = generator(7);
        let mut positions = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    positions.push(CubePos { x, y, z });
                }
            }
        }
        let forward: HashMap<_, _> = positions
            .iter()
            .map(|&p| (p, terrain.generate(p)))
            .collect();
        let backward: HashMap<_, _> = positions
            .iter()
            .rev()
            .map(|&p| (p, terrain.generate(p)))
            .collect();
        // A third order: the first half reversed, then the rest forward.
        let interleaved: HashMap<_, _> = positions
            .iter()
            .take(14)
            .rev()
            .chain(positions.iter().skip(14))
            .map(|&p| (p, terrain.generate(p)))
            .collect();
        assert_eq!(forward.len(), positions.len());
        assert_eq!(backward.len(), positions.len());
        assert_eq!(interleaved.len(), positions.len());
        assert_eq!(forward, backward);
        assert_eq!(forward, interleaved);
    }

    #[test]
    fn different_seeds_give_different_terrain() {
        let a = generator(1);
        let b = generator(2);
        let differing = (-64..64)
            .filter(|&x| a.surface_height(x, x) != b.surface_height(x, x))
            .count();
        assert!(
            differing > 100,
            "seeds 1 and 2 look identical ({differing} differing columns)"
        );
    }

    /// The block at world `(x, y, z)`, through the cube that contains it.
    fn block_at(terrain: &Generator, x: i64, y: i64, z: i64) -> BlockStateId {
        let cube = terrain.generate(CubePos {
            x: (x >> 5) as i32,
            y: (y >> 5) as i32,
            z: (z >> 5) as i32,
        });
        cube.get(LocalPos::new((x & 31) as u8, (y & 31) as u8, (z & 31) as u8).unwrap())
    }

    #[test]
    fn column_layering_is_correct_at_a_sample_column() {
        let terrain = generator(42);
        let blocks = state_ids();
        // Find one grass and one sand column within a bounded window.
        let mut grass = None;
        let mut sand = None;
        'search: for r in 0..4096i64 {
            for &(dx, dz) in &[(r, 0), (-r, 3), (r / 2, -r), (-r / 4, r)] {
                let h = terrain.surface_height(dx, dz);
                if h > 62 && grass.is_none() {
                    grass = Some((dx, dz, h));
                }
                if h <= 61 && sand.is_none() {
                    sand = Some((dx, dz, h));
                }
                if grass.is_some() && sand.is_some() {
                    break 'search;
                }
            }
        }
        let (gx, gz, gh) = grass.expect("a grass column within 4096 blocks of origin");
        let (sx, sz, sh) = sand.expect("a sand column within 4096 blocks of origin");

        // Grass column: air above, grass at h, dirt h-3..h-1, stone below.
        assert_eq!(block_at(&terrain, gx, gh + 1, gz), BlockStateId::AIR);
        assert_eq!(block_at(&terrain, gx, gh, gz), blocks.grass_block);
        assert_eq!(block_at(&terrain, gx, gh - 1, gz), blocks.dirt);
        assert_eq!(block_at(&terrain, gx, gh - 3, gz), blocks.dirt);
        assert_eq!(block_at(&terrain, gx, gh - 4, gz), blocks.stone);
        assert_eq!(block_at(&terrain, gx, gh - 40, gz), blocks.stone);
        // Sand column: sand replaces the grass and dirt layers.
        assert_eq!(block_at(&terrain, sx, sh, sz), blocks.sand);
        assert_eq!(block_at(&terrain, sx, sh - 3, sz), blocks.sand);
        assert_eq!(block_at(&terrain, sx, sh - 4, sz), blocks.stone);
    }

    #[test]
    fn far_above_is_uniform_air_and_far_below_uniform_stone() {
        let terrain = generator(42);
        let air = terrain.generate(CubePos {
            x: 3,
            y: 100,
            z: -2,
        });
        assert_eq!(air, Cube::default());
        let stone = terrain.generate(CubePos {
            x: 3,
            y: -100,
            z: -2,
        });
        assert_eq!(stone, Cube::filled(state_ids().stone));
    }

    #[test]
    fn no_nan_or_extreme_values_at_a_billion() {
        let terrain = generator(42);
        for &(x, z) in &[
            (1_000_000_000i64, 1_000_000_000i64),
            (-1_000_000_000, 1_000_000_000),
            (1_000_000_000, -1_000_000_000),
            (-1_000_000_000, -1_000_000_000),
        ] {
            let h = terrain.surface_height(x, z);
            assert!(
                (MIN_HEIGHT..=MAX_HEIGHT).contains(&h),
                "h = {h} at ({x}, {z})"
            );
            // Generating a whole cube at the far coordinate must not blow
            // up either; heights inside it stay in range.
            let _ = terrain.generate(CubePos {
                x: (x >> 5) as i32,
                y: 2,
                z: (z >> 5) as i32,
            });
        }
    }

    #[test]
    fn heights_stay_in_the_documented_range_over_a_1024_square() {
        let terrain = generator(42);
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        // Every second block of a 1024×1024 area centred on the origin.
        for x in (-512..512).step_by(2) {
            for z in (-512..512).step_by(2) {
                let h = terrain.surface_height(x, z);
                assert!(
                    (MIN_HEIGHT..=MAX_HEIGHT).contains(&h),
                    "h = {h} at ({x}, {z})"
                );
                min = min.min(h);
                max = max.max(h);
            }
        }
        // The range must not be degenerate: hills actually happen.
        assert!(max - min >= 10, "suspiciously flat: [{min}, {max}]");
    }

    #[test]
    fn spawn_sits_two_blocks_over_the_surface() {
        let terrain = generator(42);
        let [x, y, z] = terrain.spawn();
        assert_eq!((x, z), (0.5, 0.5));
        assert_eq!(y, terrain.surface_height(0, 0) as f64 + 2.0);
    }

    #[test]
    fn vertically_adjacent_cubes_share_the_column_stage() {
        use std::sync::atomic::Ordering;
        // A private compiled graph: the shipped one is shared, and other
        // tests' evaluations would race this counter.
        let graph = crate::worldgen::compile(
            &crate::worldgen::Graph::from_json(crate::worldgen::OVERWORLD_JSON).unwrap(),
        )
        .unwrap();
        let terrain = Generator::from_graph(42, state_ids(), &graph);
        let pos = CubePos { x: 0, y: 2, z: 0 };
        let _ = terrain.generate(pos);
        let after_first = graph.column_evals.load(Ordering::Relaxed);
        // The same column again: the cache answers, no noise reruns.
        let _ = terrain.generate(CubePos { y: 3, ..pos });
        assert_eq!(graph.column_evals.load(Ordering::Relaxed), after_first);
    }
}
