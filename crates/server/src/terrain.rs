//! Terrain from the declarative worldgen graph: a [`Generator`] that runs
//! a compiled shipped graph ([`crate::worldgen::Preset`]) and turns its
//! density into 32³ cubes. The graph format, node table and determinism
//! rule are documented in the crate README's "Worldgen graph" section; the
//! module docs of [`crate::worldgen`] restate the rule.
//!
//! Terrain shape is data now: `final_density >= 0` means solid, and the
//! surface stage below decides which solid block a cell becomes. The task
//! 1.5 code this replaces lives on as [`crate::worldgen::reference`] for
//! the equivalence tests and the comparison bench.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;
use jbm_core::lod::downsample_cell;

use crate::worldgen::{self, CompiledGraph, Preset};

/// Sea level (task 2.8): air cells at or below it become `base:water` in
/// the surface stage, in both `generate` and `sample_lod`. Water is
/// non-solid, so physics passes through it; there is no swimming or flow.
/// Only the `Overworld` preset has a sea (see [`Preset::sea_level`]) —
/// `Classic` is frozen bit-for-bit and never gains water.
///
/// The water's top face sits at `SEA_LEVEL + 1 = 64`, a multiple of 16:
/// every LOD cell boundary up to level 4 falls on it, so the downsampled
/// sea is exactly as high at every level. (At 64 the top face sat at 65,
/// and each level's lean-solid rounding lifted it, 66, 68, 72, 80 — a
/// visible step on the ocean at every ring border.)
pub const SEA_LEVEL: i64 = 63;

/// The classic terrain's sand line: a solid cell whose surface sits at or
/// below it gets sand instead of grass and dirt. Frozen at 61 with the
/// classic preset.
const SAND_LEVEL: i64 = 61;

/// The overworld's beach line: columns whose surface is at most one block
/// above the water (`SEA_LEVEL + 1`), and the seabed, are sand, so coasts
/// show a sand rim above the waterline.
const BEACH_LEVEL: i64 = SEA_LEVEL + 1;

/// Above this height the surface is bare stone: no soil survives the
/// altitude (snow needs a block and a texture and is a follow-up).
const STONE_LEVEL: i64 = SEA_LEVEL + 180;

/// A column whose surface steps this many blocks (or more) to any of its
/// four immediate neighbours counts as a cliff and surfaces as stone.
const STEEP_SLOPE: i64 = 4;

/// The gentle-slope bar the spawn search requires: no neighbour step
/// larger than this.
const GENTLE_SLOPE: i64 = 2;

/// Documented height bounds of the shipped overworld graph, derived from
/// its interval bounds (the worldgen tests assert the match): every column
/// of the world stays inside this range, deep ocean floor to peak. The
/// classic graph's own, much narrower range `[36, 92]` sits inside it.
pub const MIN_HEIGHT: i64 = 23;
pub const MAX_HEIGHT: i64 = 408;

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

/// The surface material of one column of a sea-style (overworld) world,
/// decided from that column's own surface height `h` and its steepest
/// neighbour step `slope` — local per column, so it is cheap and
/// LOD-consistent (task 2.8):
///
/// - **stone** when `h >= STONE_LEVEL` (bare-rock peaks) or
///   `slope >= STEEP_SLOPE` (cliffs: soil cannot hold);
/// - **sand** when `h <= BEACH_LEVEL` (a beach rim one block above the
///   water, and the seabed);
/// - **grass** over dirt otherwise.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TopMaterial {
    Grass,
    Sand,
    Stone,
}

fn column_material(h: i64, slope: i64) -> TopMaterial {
    if h >= STONE_LEVEL || slope >= STEEP_SLOPE {
        TopMaterial::Stone
    } else if h <= BEACH_LEVEL {
        TopMaterial::Sand
    } else {
        TopMaterial::Grass
    }
}

/// The sea-style counterpart of [`surface_block`]: the column's material
/// decides its surface belt (cells within 3 of the surface), stone below,
/// and air at or below `sea` becomes water. `d` semantics as
/// [`surface_block`].
#[inline]
fn surface_block_sea(
    solid: bool,
    d: u8,
    y: i64,
    top: TopMaterial,
    sea: i64,
    blocks: &Blocks,
) -> BlockStateId {
    if !solid {
        if y <= sea {
            blocks.water
        } else {
            BlockStateId::AIR
        }
    } else if d >= SURFACE_LOOKAHEAD as u8 {
        blocks.stone
    } else {
        match top {
            TopMaterial::Sand => blocks.sand,
            TopMaterial::Grass if d == 0 => blocks.grass_block,
            TopMaterial::Grass => blocks.dirt,
            TopMaterial::Stone => blocks.stone,
        }
    }
}

/// How many `(cx, cz)` columns the generator may cache. Bounded, so memory
/// stays flat whatever order requests arrive in.
const COLUMN_CACHE_CAPACITY: usize = 64;

/// The per-column data cached for one cube column — today's heightmap
/// cache, generalized. `Bits` (banded graphs, the shipped ones included)
/// holds one solidity bit per band row, `ceil(rows / 64)` words per column
/// in `z * 32 + x` order, plus the column's surface height (the highest
/// solid row, `band.low()` when none — the material and slope rules read
/// it without rescanning); `Live` (graphs whose band the interval bounds
/// cannot prove) holds the live column-stage values and keeps evaluating
/// rows on demand.
enum ColumnData {
    Bits {
        bits: Arc<[u64]>,
        heights: Arc<[i64]>,
    },
    Live(Arc<[f64]>),
}

/// Solidity bit `row` of a band's word slice.
#[inline]
fn bit_at(bits: &[u64], row: usize) -> bool {
    bits[row / 64] & (1 << (row % 64)) != 0
}

/// How far out from the origin the spawn search may spiral. 4096 rings
/// cover an 8192-block square — several continents at the overworld's
/// scale — so a sea-style world always offers land well within it.
const SPAWN_SEARCH_RINGS: i64 = 4096;

/// The fixed spawn search order (see [`Generator::spawn`]): rings of
/// Chebyshev radius 0, 1, 2, … around the origin, each ring walked from
/// its north-west corner along the north edge, down the east edge, back
/// along the south edge and up the west edge. Pure order, no state, so the
/// same seed always spawns in the same place.
fn spawn_spiral() -> impl Iterator<Item = (i64, i64)> {
    (0..SPAWN_SEARCH_RINGS).flat_map(|r| {
        let mut ring = Vec::with_capacity(if r == 0 { 1 } else { (8 * r) as usize });
        for dx in -r..=r {
            ring.push((dx, -r)); // north edge, west → east
        }
        for dz in (-r + 1)..=r {
            ring.push((r, dz)); // east edge, north → south
        }
        for x in (-r..r).rev() {
            ring.push((x, r)); // south edge, east → west
        }
        for z in ((-r + 1)..r).rev() {
            ring.push((-r, z)); // west edge, south → north
        }
        ring.into_iter()
    })
}

/// Generates terrain cubes from a compiled density graph. Output is a pure
/// function of `(seed, graph, pos)`, whatever order cubes are requested in
/// and however often; the per-column cache only ever stores results of
/// that same pure function, so it cannot change any answer.
pub struct Generator {
    seed: u64,
    blocks: Blocks,
    graph: Arc<CompiledGraph>,
    /// The preset's sea level: `Some(level)` switches the surface stage to
    /// the overworld style (water below the sea, per-column materials).
    /// `None` (the classic preset, custom graphs) keeps the pre-2.8
    /// surface rules bit-for-bit.
    sea_level: Option<i64>,
    column_cache: Mutex<HashMap<(i32, i32), ColumnData>>,
    /// Test hooks (task 2.7): how many `generate` and how many
    /// `sample_lod` mixed-path calls ran. Fast paths count nowhere — the
    /// LOD tests use `lod_sample_calls` to prove a fast path or a storage
    /// reload did no sampling, and `generate_calls` to prove `sample_lod`
    /// never generates LOD0 cubes.
    generate_calls: AtomicU64,
    lod_sample_calls: AtomicU64,
}

impl Generator {
    /// A generator running the shipped overworld graph (task 2.8:
    /// continents, mountain ranges, valleys, sea).
    pub fn new(seed: u64, blocks: Blocks) -> Self {
        Self::with_preset(seed, blocks, Preset::Overworld)
    }

    /// A generator running one of the shipped graph presets.
    pub fn with_preset(seed: u64, blocks: Blocks, preset: Preset) -> Self {
        Self {
            seed,
            blocks,
            graph: preset.compiled(),
            sea_level: preset.sea_level(),
            column_cache: Mutex::new(HashMap::new()),
            generate_calls: AtomicU64::new(0),
            lod_sample_calls: AtomicU64::new(0),
        }
    }

    /// A generator running any compiled graph, for tests and (from Stage 3)
    /// mods. Custom graphs get no sea — the pre-2.8 surface rules — so
    /// tests and mods decide explicitly when they want water.
    pub fn from_graph(seed: u64, blocks: Blocks, graph: &CompiledGraph) -> Self {
        Self {
            seed,
            blocks,
            graph: Arc::new(graph.clone()),
            sea_level: None,
            column_cache: Mutex::new(HashMap::new()),
            generate_calls: AtomicU64::new(0),
            lod_sample_calls: AtomicU64::new(0),
        }
    }

    /// Test hook: `generate` calls so far.
    pub fn generate_calls(&self) -> u64 {
        self.generate_calls.load(Ordering::Relaxed)
    }

    /// Test hook: `sample_lod` calls that ran the sampling path (fast paths
    /// excluded) so far.
    pub fn lod_sample_calls(&self) -> u64 {
        self.lod_sample_calls.load(Ordering::Relaxed)
    }

    /// Surface height (block y of the grass/sand layer) at world column
    /// `(x, z)`: the highest solid y, found from the column's xz stage plus
    /// a scan over y. Banded graphs (the shipped ones included) scan their
    /// own provable band, so the answer is exact; for the overworld it is
    /// always within [`MIN_HEIGHT`, `MAX_HEIGHT`] (water above it is a
    /// surface-stage concern, not a density one). Unbanded graphs scan
    /// `[MIN_HEIGHT, MAX_HEIGHT]` only, and yield `MIN_HEIGHT - 1` when
    /// nothing in that range is solid.
    pub fn surface_height(&self, x: i64, z: i64) -> i64 {
        let graph = self.graph.as_ref();
        match graph.band() {
            Some(band) => {
                let mut regs = vec![0.0; graph.register_count()];
                let mut bits = vec![0u64; band.rows().div_ceil(64)];
                self.eval_band_bits(graph, band, x, z, &mut regs, &mut bits)
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

    /// The spawn position: centered on the block, two above the surface.
    /// Sea-style worlds (the overworld) search outward from `(0, 0)` in a
    /// fixed spiral — ring by ring, each ring in a fixed order — for the
    /// first land column at or above [`SEA_LEVEL`] whose steepest
    /// neighbour step is at most [`GENTLE_SLOPE`], so a fresh world never
    /// starts underwater or on a cliff. Deterministic per seed. Worlds
    /// without a sea (classic, custom graphs) keep spawning over
    /// `(0, 0)` exactly as before.
    pub fn spawn(&self) -> [f64; 3] {
        let Some(sea) = self.sea_level else {
            return [0.5, (self.surface_height(0, 0) + 2) as f64, 0.5];
        };
        for (x, z) in spawn_spiral() {
            let h = self.surface_height(x, z);
            if h < sea {
                continue;
            }
            let gentle = [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)]
                .into_iter()
                .all(|(nx, nz)| (self.surface_height(nx, nz) - h).abs() <= GENTLE_SLOPE);
            if gentle {
                return [x as f64 + 0.5, (h + 2) as f64, z as f64 + 0.5];
            }
        }
        panic!("no land column found for spawn within the search limit");
    }

    /// Evaluates the band's uncertainty bits for one column — the column
    /// stage once, then one cell-stage evaluation per band row — and
    /// returns the column's surface height (the highest solid row, or the
    /// band's provably-solid floor when none).
    fn eval_band_bits(
        &self,
        graph: &CompiledGraph,
        band: &worldgen::Band,
        x: i64,
        z: i64,
        regs: &mut [f64],
        bits: &mut [u64],
    ) -> i64 {
        graph.eval_column(self.seed, x, z, regs);
        let mut top = band.low();
        for row in 0..band.rows() {
            let y = band.low() + 1 + row as i64;
            let solid = graph.eval_cell_density(self.seed, x, y, z, regs) >= 0.0;
            if solid {
                bits[row / 64] |= 1 << (row % 64);
                top = y;
            }
        }
        top
    }

    /// The cached per-column data of the 32² columns covered by cube column
    /// `(cx, cz)`, computed at most once thanks to the bounded cache.
    /// Consecutive cubes of one column (streaming works bottom-up-ish per
    /// column) hit the cache, so noise never runs twice per column.
    fn column_data(&self, cx: i32, cz: i32) -> ColumnData {
        if let Some(data) = self.column_cache.lock().unwrap().get(&(cx, cz)) {
            return match data {
                ColumnData::Bits { bits, heights } => ColumnData::Bits {
                    bits: Arc::clone(bits),
                    heights: Arc::clone(heights),
                },
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
                let mut heights = vec![0i64; (CUBE_SIZE * CUBE_SIZE) as usize];
                for z in 0..CUBE_SIZE {
                    for x in 0..CUBE_SIZE {
                        column.iter_mut().for_each(|w| *w = 0);
                        let top = self.eval_band_bits(
                            graph,
                            band,
                            base_x + i64::from(x),
                            base_z + i64::from(z),
                            &mut regs,
                            &mut column,
                        );
                        let at = (z * CUBE_SIZE + x) as usize;
                        bits[at * words..(at + 1) * words].copy_from_slice(&column);
                        heights[at] = top;
                    }
                }
                ColumnData::Bits {
                    bits: Arc::from(bits),
                    heights: Arc::from(heights),
                }
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
            ColumnData::Bits { bits, heights } => ColumnData::Bits {
                bits: Arc::clone(bits),
                heights: Arc::clone(heights),
            },
            ColumnData::Live(live) => ColumnData::Live(Arc::clone(live)),
        };
        cache.insert((cx, cz), shared);
        data
    }

    /// The 34×34 grid of surface heights covering cube column `(cx, cz)`
    /// plus the one-block border ring the slope rule needs. Interior cells
    /// come from `column_data`'s cached heights (no extra evaluation); the
    /// ring from the four neighbouring cube columns' data — the bounded
    /// cache absorbs them. Returns `None` for unbanded graphs (no slope
    /// rule there).
    fn heights_with_border(&self, cx: i32, cz: i32) -> Option<Vec<i64>> {
        self.graph.band()?;
        let side = CUBE_SIZE as usize;
        let mut grid = vec![0i64; (side + 2) * (side + 2)];
        let mut put = |gx: usize, gz: usize, h: i64| grid[gz * (side + 2) + gx] = h;
        let heights_of = |cx: i32, cz: i32| -> Arc<[i64]> {
            match self.column_data(cx, cz) {
                ColumnData::Bits { heights, .. } => heights,
                ColumnData::Live(_) => unreachable!("banded graphs cache bits"),
            }
        };
        let own = heights_of(cx, cz);
        for z in 0..side {
            for x in 0..side {
                put(x + 1, z + 1, own[z * side + x]);
            }
        }
        let west = heights_of(cx - 1, cz);
        let east = heights_of(cx + 1, cz);
        let north = heights_of(cx, cz - 1);
        let south = heights_of(cx, cz + 1);
        for i in 0..side {
            put(0, i + 1, west[i * side + (side - 1)]);
            put(side + 1, i + 1, east[i * side]);
            // The north neighbour's south row is its local z = 31 row.
            put(i + 1, 0, north[(side - 1) * side + i]);
            put(i + 1, side + 1, south[i]);
        }
        Some(grid)
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
        self.generate_calls.fetch_add(1, Ordering::Relaxed);
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
        //
        // Sea-style worlds (the overworld) decide the column's surface
        // material first — from its own height and steepest neighbour step
        // in the bordered height grid — and fill water over submerged air;
        // everything else keeps the pre-2.8 block rules bit-for-bit.
        let data = self.column_data(pos.x, pos.z);
        let rows = CUBE_SIZE as usize + SURFACE_LOOKAHEAD;
        let mut flat =
            vec![BlockStateId::AIR; CUBE_SIZE as usize * CUBE_SIZE as usize * CUBE_SIZE as usize];
        let blocks = self.blocks;
        let sea = self.sea_level;
        let materials: Option<Vec<TopMaterial>> = sea.map(|_| {
            let grid = self
                .heights_with_border(pos.x, pos.z)
                .expect("banded graph");
            let side = CUBE_SIZE as usize;
            let mut materials = Vec::with_capacity(side * side);
            for z in 0..side {
                for x in 0..side {
                    let (gx, gz) = (x + 1, z + 1);
                    let h = grid[gz * (side + 2) + gx];
                    let slope = (h - grid[gz * (side + 2) + gx - 1])
                        .abs()
                        .max((h - grid[gz * (side + 2) + gx + 1]).abs())
                        .max((h - grid[(gz - 1) * (side + 2) + gx]).abs())
                        .max((h - grid[(gz + 1) * (side + 2) + gx]).abs());
                    materials.push(column_material(h, slope));
                }
            }
            materials
        });
        let material_at = |x: usize, z: usize| -> Option<TopMaterial> {
            materials.as_ref().map(|m| m[z * CUBE_SIZE as usize + x])
        };
        let mut fill_column = |x: i32, z: i32, solid: &mut dyn FnMut(i64) -> bool| {
            let top = material_at(x as usize, z as usize);
            let mut d: u8 = 0;
            for j in (0..rows).rev() {
                let y = bottom + j as i64;
                let is_solid = solid(y);
                if j < CUBE_SIZE as usize {
                    let index =
                        (j * CUBE_SIZE as usize + z as usize) * CUBE_SIZE as usize + x as usize;
                    flat[index] = match (sea, top) {
                        (Some(sea), Some(top)) => {
                            surface_block_sea(is_solid, d, y, top, sea, &blocks)
                        }
                        _ => surface_block(is_solid, d, y, &blocks),
                    };
                }
                d = if is_solid {
                    (d + 1).min(SURFACE_LOOKAHEAD as u8)
                } else {
                    0
                };
            }
        };
        match (&data, graph.band()) {
            (ColumnData::Bits { bits, .. }, Some(band)) => {
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

    /// A LOD cube at `level ≥ 1` and `pos`, sampled straight from the
    /// density graph at that level's resolution — no LOD0 cube is generated
    /// and the LOD0 world is never consulted (task 2.7).
    ///
    /// Density is sampled on the level-(*n*−1) grid: 2·`CUBE_SIZE` points
    /// per axis at step 2^(*n*−1), each at the minimum corner of its child
    /// cell, plus the surface lookahead above. The surface stage runs in
    /// **block units**: a sample's run `d` is its consecutive solid samples
    /// above times 2^(*n*−1), capped at [`SURFACE_LOOKAHEAD`], and its `y`
    /// is its block y — so at coarse levels the dirt band vanishes and the
    /// top stays grass or sand. One refinement: the **top solid sample**
    /// carries its column's true surface material — the surface is refined
    /// upward from it (at most 2^(*n*−1)−1 extra rows per column) so the
    /// sand/grass belt stays one block wide instead of one sample wide; at
    /// level 1 the refinement is empty and the sample is the surface. Each
    /// 2×2×2 group of samples then becomes one cell through
    /// [`downsample_cell`], the same rule as the pyramid.
    ///
    /// **Level 1 is exact**: the samples equal the generated LOD0 cells
    /// (asserted by the tests), so the cube equals `downsample_octant` of
    /// the eight generated LOD0 cubes. Higher levels are an approximation
    /// bounded by the accuracy tests.
    ///
    /// Fast paths from `bounds`, as `generate`: uniform air or uniform
    /// stone without touching the graph (zero samples — the fast-path tests
    /// count them through [`Generator::lod_sample_calls`]). Banded graphs
    /// (the shipped one included) evaluate the column stage once per sample
    /// column and take the band's provable solid/air rows for free.
    pub fn sample_lod(&self, level: u8, pos: CubePos) -> Cube {
        assert!(level >= 1, "sample_lod is for levels >= 1");
        let graph = self.graph.as_ref();
        let step = 1i64 << (level as u32 - 1); // 2^(n-1) blocks between samples
        let side = i64::from(CUBE_SIZE) << level as u32; // blocks per axis
        let base = [
            i64::from(pos.x) * side,
            i64::from(pos.y) * side,
            i64::from(pos.z) * side,
        ];
        let top = base[1] + side - 1;
        let max = [base[0] + side - 1, top, base[2] + side - 1];
        if graph
            .bounds(base, [max[0], top + SURFACE_LOOKAHEAD as i64, max[2]])
            .0
            >= 0.0
        {
            return Cube::filled(self.blocks.stone);
        }
        if graph.bounds(base, max).1 < 0.0 {
            return Cube::default();
        }
        self.lod_sample_calls.fetch_add(1, Ordering::Relaxed);

        // Sampling runs per column, exactly `generate`'s mixed path but
        // stepping `step` blocks: rows are the sample j of the column, plus
        // `lookahead` samples above the cube so top rows see the same
        // saturated runs `generate` would. Every column's solidity rows are
        // evaluated once and kept, with the column's true surface (the top
        // solid sample refined upward, at most `step - 1` rows), so the
        // fill never re-evaluates density.
        //
        // Sea-style worlds (the overworld) first collect every sample
        // column's surface over the grid plus a one-stride border ring, so
        // the slope rule sees the neighbours `generate`'s per-block rule
        // would approximate, then decide per-column materials and fill
        // water over submerged air — exactly `generate`'s styling.
        let per_axis = 2 * CUBE_SIZE as usize; // samples per axis
        let lookahead = SURFACE_LOOKAHEAD.div_ceil(step as usize);
        let rows = per_axis + lookahead;
        let mut states = vec![BlockStateId::AIR; per_axis * per_axis * per_axis];
        let blocks = self.blocks;
        // Sea styling needs per-column surface heights; only banded graphs
        // get it (the overworld always is; custom graphs are sea-less).
        let sea = self.sea_level.filter(|_| graph.band().is_some());
        let band = graph.band();
        let mut regs = vec![0.0; graph.register_count()];

        // Evaluates the sample rows of world column `(wx, wz)` into
        // `is_solid` and returns the column's true surface height. When no
        // sample in the window is solid the surface sits outside it; the
        // density `surface(x, z) − y` is monotone in `y` for the shipped
        // (heightmap) graphs, so a binary search over the band finds the
        // exact surface in a handful of evaluations — the slope rule needs
        // the true height of every neighbour column, or `generate`'s and
        // `sample_lod`'s materials would disagree at level 1.
        let mut eval_rows = |wx: i64, wz: i64, is_solid: &mut Vec<bool>| -> i64 {
            graph.eval_column(self.seed, wx, wz, &mut regs);
            is_solid.clear();
            is_solid.resize(rows, false);
            let mut solid_at = |y: i64| match band {
                Some(b) if y <= b.low() => true,
                Some(b) if y >= b.high() => false,
                _ => graph.eval_cell_density(self.seed, wx, y, wz, &mut regs) >= 0.0,
            };
            for j in (0..rows).rev() {
                is_solid[j] = solid_at(base[1] + step * j as i64);
            }
            match (0..rows).rev().find(|&j| is_solid[j]) {
                Some(top) => {
                    let top_y = base[1] + step * top as i64;
                    let mut h = top_y;
                    while h + 1 < top_y + step && solid_at(h + 1) {
                        h += 1;
                    }
                    h
                }
                None => match band {
                    None => MIN_HEIGHT - 1,
                    Some(b) => {
                        let window_top = base[1] + step * rows as i64 - 1;
                        if solid_at(window_top) {
                            // Surface above the window: least provably-air
                            // row above it, minus one.
                            let (mut lo, mut hi) = (window_top, b.high());
                            while lo + 1 < hi {
                                let mid = lo.midpoint(hi);
                                if solid_at(mid) {
                                    lo = mid;
                                } else {
                                    hi = mid;
                                }
                            }
                            lo
                        } else {
                            // Surface below the window: greatest solid row
                            // under it (`b.low()` is provably solid).
                            let (mut lo, mut hi) = (b.low(), base[1]);
                            while lo + 1 < hi {
                                let mid = lo.midpoint(hi);
                                if solid_at(mid) {
                                    lo = mid;
                                } else {
                                    hi = mid;
                                }
                            }
                            lo
                        }
                    }
                },
            }
        };

        let mut columns: Vec<Vec<bool>> = vec![vec![false; rows]; per_axis * per_axis];
        let mut true_surfaces = vec![0i64; per_axis * per_axis];
        // The extended surface grid (interior + one-stride ring), sea only.
        let extended = per_axis + 2;
        let mut surfaces = vec![0i64; extended * extended];
        if sea.is_some() {
            let mut scratch = Vec::new();
            for z in 0..extended {
                for x in 0..extended {
                    let interior = (1..=per_axis).contains(&x) && (1..=per_axis).contains(&z);
                    let (wx, wz) = (
                        base[0] + step * (x as i64 - 1),
                        base[2] + step * (z as i64 - 1),
                    );
                    let h = if interior {
                        let column = &mut columns[(z - 1) * per_axis + (x - 1)];
                        eval_rows(wx, wz, column)
                    } else {
                        eval_rows(wx, wz, &mut scratch)
                    };
                    surfaces[z * extended + x] = h;
                    if interior {
                        true_surfaces[(z - 1) * per_axis + (x - 1)] = h;
                    }
                }
            }
        } else {
            for z in 0..per_axis {
                for x in 0..per_axis {
                    let (wx, wz) = (base[0] + step * x as i64, base[2] + step * z as i64);
                    let column = &mut columns[z * per_axis + x];
                    let h = eval_rows(wx, wz, column);
                    true_surfaces[z * per_axis + x] = h;
                }
            }
        }

        let material_at = |x: usize, z: usize| -> TopMaterial {
            if sea.is_none() {
                return TopMaterial::Grass;
            }
            let (gx, gz) = (x + 1, z + 1);
            let h = surfaces[gz * extended + gx];
            // The slope bar scales with the sample stride so a level-*n*
            // cube classifies the same terrain `generate`'s per-block rule
            // would: a cliff is a [`STEEP_SLOPE`]-block step per block, not
            // per sample. At `step` 1 (level 1) this is exactly
            // `generate`'s rule, which the level-1 exactness test asserts.
            let bar = STEEP_SLOPE * step;
            let slope = (h - surfaces[gz * extended + gx - 1])
                .abs()
                .max((h - surfaces[gz * extended + gx + 1]).abs())
                .max((h - surfaces[(gz - 1) * extended + gx]).abs())
                .max((h - surfaces[(gz + 1) * extended + gx]).abs());
            if h >= STONE_LEVEL || slope >= bar {
                TopMaterial::Stone
            } else if h <= BEACH_LEVEL {
                TopMaterial::Sand
            } else {
                TopMaterial::Grass
            }
        };

        for z in 0..per_axis {
            for x in 0..per_axis {
                let is_solid = &columns[z * per_axis + x];
                let true_surface = true_surfaces[z * per_axis + x];
                let top = material_at(x, z);
                let top_sample = (0..rows).rev().find(|&j| is_solid[j]);
                let mut d: u8 = 0; // consecutive solid samples above, unscaled
                for j in (0..rows).rev() {
                    let y = base[1] + step * j as i64;
                    let solid_j = is_solid[j];
                    if j < per_axis {
                        // d in block units, as the surface stage defines it;
                        // the top solid sample carries the true surface's
                        // material via its `block_y`.
                        let d_blocks = ((d as i64) * step).min(SURFACE_LOOKAHEAD as i64) as u8;
                        let at = (j * per_axis + z) * per_axis + x;
                        let block_y = if Some(j) == top_sample {
                            true_surface
                        } else {
                            y
                        };
                        states[at] = match sea {
                            Some(sea) => {
                                surface_block_sea(solid_j, d_blocks, block_y, top, sea, &blocks)
                            }
                            None => surface_block(solid_j, d_blocks, block_y, &blocks),
                        };
                    }
                    d = if solid_j {
                        (d + 1).min(SURFACE_LOOKAHEAD as u8)
                    } else {
                        0
                    };
                }
            }
        }
        // Each cell's eight children are the samples at 2·cell + (dx, dy, dz)
        // — the same gather `downsample_octant` does, on the doubled grid.
        Cube::from_fn(|p| {
            let (px, py, pz) = (p.x as usize, p.y as usize, p.z as usize);
            let mut children = [BlockStateId::AIR; 8];
            for dy in 0..2usize {
                for dz in 0..2usize {
                    for dx in 0..2usize {
                        let at = ((2 * py + dy) * per_axis + 2 * pz + dz) * per_axis + 2 * px + dx;
                        children[dx + 2 * dz + 4 * dy] = states[at];
                    }
                }
            }
            downsample_cell(children)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Generator, MAX_HEIGHT, MIN_HEIGHT};
    use crate::worldgen::Preset;
    use jbm_core::block::builtin::Blocks;
    use jbm_core::block::{BlockRegistry, BlockStateId};
    use jbm_core::coords::{CubePos, LocalPos};
    use jbm_core::cube::Cube;
    use std::collections::HashMap;

    /// The classic preset: these tests assert the pre-2.8 terrain's exact
    /// layering and heights, so they pin it.
    fn generator(seed: u64) -> Generator {
        let mut registry = BlockRegistry::new();
        let blocks = Blocks::register(&mut registry);
        Generator::with_preset(seed, blocks, Preset::Classic)
    }

    fn overworld(seed: u64) -> Generator {
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
        let terrain = overworld(42);
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
        let terrain = overworld(42);
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
        // The range must not be degenerate: hills (and likely water) happen
        // even in one square kilometre; the 8 km shape statistics test in
        // tests/overworld.rs asserts the full continental relief.
        assert!(max - min >= 100, "suspiciously flat: [{min}, {max}]");
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
