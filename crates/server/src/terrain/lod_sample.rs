//! LOD sampling: whole LOD cubes straight from the density graph. The
//! three phases — sample columns, style, reduce — are the seam a later
//! height-exact LOD rule replaces.

use std::sync::atomic::Ordering;

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;
use jbm_core::lod::downsample_cell;

use super::Generator;
use super::surface::{
    SURFACE_LOOKAHEAD, TopMaterial, column_material, max_neighbor_step, style_column,
};
use crate::worldgen::{self, CompiledGraph};

impl Generator {
    /// A LOD cube at `level ≥ 1` and `pos`, sampled straight from the
    /// density graph at that level's resolution — no LOD0 cube is
    /// generated and the LOD0 world is never consulted.
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
        if self.playground {
            return self.sample_lod_playground(level, pos);
        }
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
        // Sea styling needs per-column surface heights; only banded graphs
        // get it (the overworld always is; custom graphs are sea-less).
        let sea = self.sea_level.filter(|_| graph.band().is_some());
        let mut sampler = ColumnSampler {
            seed: self.seed,
            graph,
            band: graph.band(),
            base,
            step,
            rows: per_axis + lookahead,
            regs: vec![0.0; graph.register_count()],
        };

        // Phase 1: sample every column. Phase 2: style them. Phase 3:
        // reduce each 2×2×2 sample group to one cell.
        let sampled = sample_columns(&mut sampler, sea.is_some(), per_axis);
        let states = style_columns(&sampler, &sampled, sea, &self.blocks, per_axis);
        reduce_to_cube(&states, per_axis)
    }

    /// The playground's LOD sample (task 2.12): the same three phases as
    /// the graph path, but "sampling" reads the fixed layout's block at
    /// each sample point — the very function `generate` fills LOD0 with —
    /// so level 1 is exact by construction and coarser levels follow the
    /// same min-corner sampling rule.
    fn sample_lod_playground(&self, level: u8, pos: CubePos) -> Cube {
        self.lod_sample_calls.fetch_add(1, Ordering::Relaxed);
        let step = 1i64 << (level as u32 - 1);
        let side = i64::from(CUBE_SIZE) << level as u32;
        let base = [
            i64::from(pos.x) * side,
            i64::from(pos.y) * side,
            i64::from(pos.z) * side,
        ];
        let per_axis = 2 * CUBE_SIZE as usize;
        let blocks = self.blocks;
        let mut states = vec![BlockStateId::AIR; per_axis * per_axis * per_axis];
        for j in 0..per_axis {
            let y = base[1] + step * j as i64;
            for z in 0..per_axis {
                let wz = base[2] + step * z as i64;
                for x in 0..per_axis {
                    let wx = base[0] + step * x as i64;
                    states[(j * per_axis + z) * per_axis + x] =
                        super::playground::state(wx, y, wz, &blocks);
                }
            }
        }
        reduce_to_cube(&states, per_axis)
    }
}

/// Phase 1's context: the graph and its band, the sample window's
/// geometry, and the scratch register array reused across every column
/// evaluation.
struct ColumnSampler<'a> {
    seed: u64,
    graph: &'a CompiledGraph,
    band: Option<&'a worldgen::Band>,
    /// The sample window's min corner; sample `(x, z, j)` sits at world
    /// `(base[0] + step·x, base[1] + step·j, base[2] + step·z)`.
    base: [i64; 3],
    step: i64,
    rows: usize,
    regs: Vec<f64>,
}

impl ColumnSampler<'_> {
    /// Evaluates the sample rows of world column `(wx, wz)` into
    /// `is_solid` and returns the column's true surface height. When no
    /// sample in the window is solid the surface sits outside it; the
    /// density `surface(x, z) − y` is monotone in `y` for the shipped
    /// (heightmap) graphs, so a binary search over the band finds the
    /// exact surface in a handful of evaluations — the slope rule needs
    /// the true height of every neighbour column, or `generate`'s and
    /// `sample_lod`'s materials would disagree at level 1.
    fn eval_rows(&mut self, wx: i64, wz: i64, is_solid: &mut Vec<bool>) -> i64 {
        let (seed, graph, band, base, step, rows) = (
            self.seed, self.graph, self.band, self.base, self.step, self.rows,
        );
        graph.eval_column(seed, wx, wz, &mut self.regs);
        is_solid.clear();
        is_solid.resize(rows, false);
        let mut solid_at = |y: i64| match band {
            Some(b) if y <= b.low() => true,
            Some(b) if y >= b.high() => false,
            _ => graph.eval_cell_density(seed, wx, y, wz, &mut self.regs) >= 0.0,
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
                None => super::MIN_HEIGHT - 1,
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
    }
}

/// Phase 1's output: every interior sample column's solidity rows and true
/// surface, plus — sea only — the bordered surface grid the material rule
/// reads (interior + one-stride ring).
struct SampledColumns {
    columns: Vec<Vec<bool>>,
    true_surfaces: Vec<i64>,
    /// `(per_axis + 2)²` heights, row-major; empty without a sea.
    surfaces: Vec<i64>,
}

/// **Phase 1 — sample columns.** With a sea, the bordered surface grid
/// forces one extra evaluation per border column; without one, only the
/// interior columns run.
fn sample_columns(sampler: &mut ColumnSampler, sea: bool, per_axis: usize) -> SampledColumns {
    let mut columns: Vec<Vec<bool>> = vec![vec![false; sampler.rows]; per_axis * per_axis];
    let mut true_surfaces = vec![0i64; per_axis * per_axis];
    if !sea {
        for z in 0..per_axis {
            for x in 0..per_axis {
                let (wx, wz) = (
                    sampler.base[0] + sampler.step * x as i64,
                    sampler.base[2] + sampler.step * z as i64,
                );
                let column = &mut columns[z * per_axis + x];
                true_surfaces[z * per_axis + x] = sampler.eval_rows(wx, wz, column);
            }
        }
        return SampledColumns {
            columns,
            true_surfaces,
            surfaces: Vec::new(),
        };
    }
    // The extended surface grid (interior + one-stride ring).
    let extended = per_axis + 2;
    let mut surfaces = vec![0i64; extended * extended];
    let mut scratch = Vec::new();
    for z in 0..extended {
        for x in 0..extended {
            let interior = (1..=per_axis).contains(&x) && (1..=per_axis).contains(&z);
            let (wx, wz) = (
                sampler.base[0] + sampler.step * (x as i64 - 1),
                sampler.base[2] + sampler.step * (z as i64 - 1),
            );
            let h = if interior {
                let column = &mut columns[(z - 1) * per_axis + (x - 1)];
                sampler.eval_rows(wx, wz, column)
            } else {
                sampler.eval_rows(wx, wz, &mut scratch)
            };
            surfaces[z * extended + x] = h;
            if interior {
                true_surfaces[(z - 1) * per_axis + (x - 1)] = h;
            }
        }
    }
    SampledColumns {
        columns,
        true_surfaces,
        surfaces,
    }
}

/// **Phase 2 — style.** Per column: the material from its height and
/// steepest neighbour step in the bordered grid (grass without a sea),
/// then one shared [`style_column`] pass whose top solid sample carries
/// the column's true surface. Returns the styled sample states, row-major
/// `(j, z, x)`.
fn style_columns(
    sampler: &ColumnSampler,
    sampled: &SampledColumns,
    sea: Option<i64>,
    blocks: &Blocks,
    per_axis: usize,
) -> Vec<BlockStateId> {
    let extended = per_axis + 2;
    let mut states = vec![BlockStateId::AIR; per_axis * per_axis * per_axis];
    let material_at = |x: usize, z: usize| -> TopMaterial {
        if sea.is_none() {
            return TopMaterial::Grass;
        }
        let (gx, gz) = (x + 1, z + 1);
        let h = sampled.surfaces[gz * extended + gx];
        let slope = max_neighbor_step(&sampled.surfaces, extended, gx, gz);
        column_material(h, slope, sampler.step)
    };
    let mut styled = vec![BlockStateId::AIR; per_axis];
    for z in 0..per_axis {
        for x in 0..per_axis {
            let is_solid = &sampled.columns[z * per_axis + x];
            let true_surface = sampled.true_surfaces[z * per_axis + x];
            let top = material_at(x, z);
            style_column(
                &mut styled,
                &mut |j| is_solid[j],
                sampler.rows,
                sampler.base[1],
                sampler.step,
                Some(true_surface),
                sea,
                sea.map(|_| top),
                blocks,
            );
            for (j, &state) in styled.iter().enumerate() {
                states[(j * per_axis + z) * per_axis + x] = state;
            }
        }
    }
    states
}

/// **Phase 3 — reduce.** Each cell's eight children are the samples at
/// `2·cell + (dx, dy, dz)` — the same gather `downsample_octant` does, on
/// the doubled grid. Shared with the playground's sampling path.
pub(super) fn reduce_to_cube(states: &[BlockStateId], per_axis: usize) -> Cube {
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
