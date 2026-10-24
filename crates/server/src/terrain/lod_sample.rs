//! LOD sampling: whole LOD cubes straight from the density graph. The
//! three phases — sample columns, style, reduce — and since task 2.14 the
//! reduce also produces the cube's column tops from each sampled
//! column's true surface.

use std::sync::atomic::Ordering;

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CUBE_SIZE, CubePos};
use jbm_core::cube::Cube;
use jbm_core::lod::{LodTops, downsample_cell};

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
    /// **Column tops** (task 2.14): each sample is a level-(*n*−1) child
    /// cell whose fill height comes from its column's surface `S` —
    /// `clamp(S − bottom + 1, 1, 2^(n−1))` for a solid sample (full below
    /// the surface, partial in the cell holding it), full for water (the
    /// sea surface sits on a cell boundary at every level) — and the tops
    /// reduce with the pyramid's recurrence (see [`reduce_to_cube`]).
    ///
    /// **Footprint envelope** (levels ≥ 2): a child cell covers a
    /// `2^(n−1)`-wide footprint, but its sample sees only the footprint's
    /// min corner — and real data's top is the footprint's *highest*
    /// block. So each sample column stands in for an estimate of its
    /// footprint's highest surface — the bilinear interpolation of its own
    /// and the next samples' true surfaces (+x, +z, +xz), maximized over
    /// the footprint (a corner) and rounded up — and is styled as if its
    /// surface were that height: the rows up to it count as solid. A
    /// parent cell's footprint thus reads a 3 × 3 grid of true surfaces
    /// instead of 2 × 2, which keeps the coarse top within a block of the
    /// footprint's highest block on smooth terrain instead of leaning
    /// under it. The sea worlds' bordered surface grid already holds the
    /// far ring; other graphs evaluate it (one extra row and column of
    /// surfaces). Level 1 has one-column footprints, so the envelope is
    /// the column itself and level 1 stays exact.
    ///
    /// **Level 1 is exact**: the samples equal the generated LOD0 cells
    /// (asserted by the tests), so the cube and its tops equal
    /// `downsample_octant` of the eight generated LOD0 cubes. Higher levels
    /// are an approximation bounded by the accuracy tests: the sample grid
    /// can miss a peak between samples.
    ///
    /// Fast paths from `bounds`, as `generate`: uniform air or uniform
    /// stone without touching the graph (zero samples — the fast-path tests
    /// count them through [`Generator::lod_sample_calls`]). Banded graphs
    /// (the shipped one included) evaluate the column stage once per sample
    /// column and take the band's provable solid/air rows for free.
    pub fn sample_lod(&self, level: u8, pos: CubePos) -> (Cube, LodTops) {
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
            // Solid past the lookahead: every column's surface is above
            // the cube, so every top is full.
            return (Cube::filled(self.blocks.stone), LodTops::full(level));
        }
        if graph.bounds(base, max).1 < 0.0 {
            return (Cube::default(), LodTops::empty(level));
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
        let grid = SampleGrid {
            states: &states,
            per_axis,
            surfaces: &sampled.envelope,
            base_y: base[1],
            step,
            water: self.blocks.water,
        };
        reduce_to_cube(&grid)
    }

    /// The playground's LOD sample (task 2.12): the same three phases as
    /// the graph path, but "sampling" reads the fixed layout's block at
    /// each sample point — the very function `generate` fills LOD0 with —
    /// so level 1 is exact by construction and coarser levels follow the
    /// same min-corner sampling rule.
    fn sample_lod_playground(&self, level: u8, pos: CubePos) -> (Cube, LodTops) {
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
        // The layout is a heightmap: each sample column's true surface is
        // its column's top solid block — over the grid plus the far ring
        // the footprint envelope reads.
        let ring = per_axis + 1;
        let mut ringed = vec![0i64; ring * ring];
        for z in 0..ring {
            let wz = base[2] + step * z as i64;
            for x in 0..ring {
                let wx = base[0] + step * x as i64;
                ringed[z * ring + x] = super::playground::column(wx, wz, &blocks).height;
            }
        }
        let envelope = footprint_envelope(&ringed, ring, 0, per_axis, step);
        for z in 0..per_axis {
            let wz = base[2] + step * z as i64;
            for x in 0..per_axis {
                let wx = base[0] + step * x as i64;
                let (own, raised) = (ringed[z * ring + x], envelope[z * per_axis + x]);
                let top = super::playground::state(wx, own, wz, &blocks);
                for j in 0..per_axis {
                    let y = base[1] + step * j as i64;
                    states[(j * per_axis + z) * per_axis + x] = if y > own && y <= raised {
                        top // raised to the footprint envelope
                    } else {
                        super::playground::state(wx, y, wz, &blocks)
                    };
                }
            }
        }
        reduce_to_cube(&SampleGrid {
            states: &states,
            per_axis,
            surfaces: &envelope,
            base_y: base[1],
            step,
            water: blocks.water,
        })
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
/// surface, its footprint envelope, plus — sea only — the bordered surface
/// grid the material rule reads (interior + one-stride ring).
struct SampledColumns {
    columns: Vec<Vec<bool>>,
    true_surfaces: Vec<i64>,
    /// `per_axis²` footprint envelopes (see [`footprint_envelope`]),
    /// row-major `(z, x)`: what the column is styled and topped as.
    envelope: Vec<i64>,
    /// `(per_axis + 2)²` heights, row-major; empty without a sea.
    surfaces: Vec<i64>,
}

/// The footprint envelope of every interior sample column: the bilinear
/// interpolation of the true surfaces at its own sample and the next ones
/// in +x, +z and +xz, at the four corners of its child cell's footprint
/// (offsets 0 and `step − 1`; a bilinear patch peaks at a corner), the max
/// rounded up. `ringed` is a `stride`-wide surface grid holding interior
/// column `(x, z)` at `(x + offset, z + offset)` and at least one more row
/// and column past the interior. At `step == 1` the footprint is the
/// column itself, so the envelope is the true surface (level 1 stays
/// exact).
fn footprint_envelope(
    ringed: &[i64],
    stride: usize,
    offset: usize,
    per_axis: usize,
    step: i64,
) -> Vec<i64> {
    let mut envelope = Vec::with_capacity(per_axis * per_axis);
    for z in 0..per_axis {
        for x in 0..per_axis {
            let at = |dx: usize, dz: usize| ringed[(z + offset + dz) * stride + x + offset + dx];
            if step == 1 {
                envelope.push(at(0, 0));
                continue;
            }
            let (s00, s10, s01, s11) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
            let mut top = s00;
            for (a, b) in [(step - 1, 0), (0, step - 1), (step - 1, step - 1)] {
                // step² · bilinear(a / step, b / step), rounded up.
                let scaled = s00 * (step - a) * (step - b)
                    + s10 * a * (step - b)
                    + s01 * (step - a) * b
                    + s11 * a * b;
                top = top.max(-(-scaled).div_euclid(step * step));
            }
            envelope.push(top);
        }
    }
    envelope
}

/// **Phase 1 — sample columns.** With a sea, the bordered surface grid
/// forces one extra evaluation per border column; without one, only the
/// interior columns run.
fn sample_columns(sampler: &mut ColumnSampler, sea: bool, per_axis: usize) -> SampledColumns {
    let mut columns: Vec<Vec<bool>> = vec![vec![false; sampler.rows]; per_axis * per_axis];
    let mut true_surfaces = vec![0i64; per_axis * per_axis];
    if !sea {
        // The interior plus — for the footprint envelope at levels ≥ 2 —
        // the far row and column.
        let ring = if sampler.step == 1 {
            per_axis
        } else {
            per_axis + 1
        };
        let mut ringed = vec![0i64; ring * ring];
        let mut scratch = Vec::new();
        for z in 0..ring {
            for x in 0..ring {
                let (wx, wz) = (
                    sampler.base[0] + sampler.step * x as i64,
                    sampler.base[2] + sampler.step * z as i64,
                );
                ringed[z * ring + x] = if x < per_axis && z < per_axis {
                    let h = sampler.eval_rows(wx, wz, &mut columns[z * per_axis + x]);
                    true_surfaces[z * per_axis + x] = h;
                    h
                } else {
                    sampler.eval_rows(wx, wz, &mut scratch)
                };
            }
        }
        return SampledColumns {
            columns,
            envelope: footprint_envelope(&ringed, ring, 0, per_axis, sampler.step),
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
        envelope: footprint_envelope(&surfaces, extended, 1, per_axis, sampler.step),
        true_surfaces,
        surfaces,
    }
}

/// **Phase 2 — style.** Per column: the material from its height and
/// steepest neighbour step in the bordered grid (grass without a sea),
/// then one shared [`style_column`] pass over the column raised to its
/// footprint envelope (rows above the true surface up to the envelope
/// count as solid), whose top solid sample carries the envelope height.
/// Returns the styled sample states, row-major `(j, z, x)`.
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
            let envelope = sampled.envelope[z * per_axis + x];
            let top = material_at(x, z);
            let (bottom, step) = (sampler.base[1], sampler.step);
            style_column(
                &mut styled,
                &mut |j| {
                    let y = bottom + step * j as i64;
                    is_solid[j] || (y > true_surface && y <= envelope)
                },
                sampler.rows,
                bottom,
                step,
                Some(envelope),
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

/// Phase 3's input: the styled sample states (row-major `(j, z, x)`),
/// each sample column's true surface (block y of its top solid block),
/// and the grid's geometry.
pub(super) struct SampleGrid<'a> {
    pub states: &'a [BlockStateId],
    pub per_axis: usize,
    /// `per_axis²` column surfaces (the footprint envelopes), row-major
    /// `(z, x)`.
    pub surfaces: &'a [i64],
    /// Block y of sample row 0.
    pub base_y: i64,
    /// Blocks between samples: the child cell size `2^(n−1)`.
    pub step: i64,
    /// Water fills whole child cells (the sea surface is cell-aligned).
    pub water: BlockStateId,
}

impl SampleGrid<'_> {
    fn state(&self, x: usize, j: usize, z: usize) -> BlockStateId {
        self.states[(j * self.per_axis + z) * self.per_axis + x]
    }

    /// The fill height of filled sample `(x, j, z)` as a child cell: water
    /// is full; a solid sample reaches its column's true surface, clamped
    /// to the cell (full below the surface).
    fn child_height(&self, x: usize, j: usize, z: usize) -> u8 {
        if self.state(x, j, z) == self.water {
            return self.step as u8;
        }
        let bottom = self.base_y + self.step * j as i64;
        (self.surfaces[z * self.per_axis + x] - bottom + 1).clamp(1, self.step) as u8
    }
}

/// **Phase 3 — reduce.** Each cell's eight children are the samples at
/// `2·cell + (dx, dy, dz)` — the same gather `downsample_octant` does, on
/// the doubled grid. The tops follow the pyramid's recurrence per octant
/// half (each half's 32 sample rows play one child cube): a half-column's
/// top cell is its topmost filled cell, and its height is the highest
/// `dy · step + h` over that cell's filled children, `h` from
/// [`SampleGrid::child_height`]. Shared with the playground's sampling
/// path.
pub(super) fn reduce_to_cube(grid: &SampleGrid) -> (Cube, LodTops) {
    let per_axis = grid.per_axis;
    let cube = Cube::from_fn(|p| {
        let (px, py, pz) = (p.x as usize, p.y as usize, p.z as usize);
        let mut children = [BlockStateId::AIR; 8];
        for dy in 0..2usize {
            for dz in 0..2usize {
                for dx in 0..2usize {
                    children[dx + 2 * dz + 4 * dy] =
                        grid.state(2 * px + dx, 2 * py + dy, 2 * pz + dz);
                }
            }
        }
        downsample_cell(children)
    });
    let level = grid.step.trailing_zeros() as u8 + 1;
    let mut tops = LodTops::empty(level);
    if cube.uniform() == Some(BlockStateId::AIR) {
        return (cube, tops);
    }
    let rows = per_axis / 2; // sample rows per octant half
    for z in 0..per_axis / 2 {
        for x in 0..per_axis / 2 {
            for half in 0..2usize {
                // The topmost filled sample row of each child column in
                // this half, then the parent row holding the highest one.
                let mut best: Option<(usize, u8)> = None; // (row j, height)
                for dz in 0..2usize {
                    for dx in 0..2usize {
                        let (cx, cz) = (2 * x + dx, 2 * z + dz);
                        let Some(j) = (half * rows..(half + 1) * rows)
                            .rev()
                            .find(|&j| grid.state(cx, j, cz) != BlockStateId::AIR)
                        else {
                            continue;
                        };
                        let h = (j & 1) as u8 * grid.step as u8 + grid.child_height(cx, j, cz);
                        best = Some(match best {
                            Some((bj, bh)) if bj / 2 > j / 2 => (bj, bh),
                            Some((bj, bh)) if bj / 2 == j / 2 => (bj.max(j), bh.max(h)),
                            _ => (j, h),
                        });
                    }
                }
                if let Some((_, h)) = best {
                    tops.set_half(half as u8, x as u8, z as u8, h);
                }
            }
        }
    }
    tops.compact();
    (cube, tops)
}
