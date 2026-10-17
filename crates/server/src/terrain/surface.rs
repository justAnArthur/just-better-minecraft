//! The surface stage: which block a solid cell becomes. One set of rules
//! serves both callers — `generate` (one sample per block, stride 1) and
//! `sample_lod` (samples spaced `step` blocks apart, `d` and the slope bar
//! scaled back into block units) — so a level-*n* cube classifies the
//! terrain the per-block rule would, and level 1 is exact.

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;

use super::SEA_LEVEL;

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

/// How many cells above a cube the surface stage may need to look at: `d`
/// is capped at 4, so cells up to `y + 4` decide the top rows' blocks.
pub(super) const SURFACE_LOOKAHEAD: usize = 4;

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
/// LOD-consistent:
///
/// - **stone** when `h >= STONE_LEVEL` (bare-rock peaks) or the slope
///   reaches the cliff bar (soil cannot hold);
/// - **sand** when `h <= BEACH_LEVEL` (a beach rim one block above the
///   water, and the seabed);
/// - **grass** over dirt otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TopMaterial {
    Grass,
    Sand,
    Stone,
}

/// The cliff bar scales with the sample `step` so a level-*n* cube
/// classifies the same terrain the per-block rule would: a cliff is a
/// [`STEEP_SLOPE`]-block step per block, not per sample. At `step` 1
/// (level 1) this is exactly `generate`'s rule, which the level-1
/// exactness test asserts.
pub(super) fn column_material(h: i64, slope: i64, step: i64) -> TopMaterial {
    if h >= STONE_LEVEL || slope >= STEEP_SLOPE * step {
        TopMaterial::Stone
    } else if h <= BEACH_LEVEL {
        TopMaterial::Sand
    } else {
        TopMaterial::Grass
    }
}

/// The steepest step from grid cell `(gx, gz)` to its four immediate
/// neighbours, on a `stride`-wide row-major height grid that carries a
/// one-cell border ring (both callers sample interior cells, so the
/// neighbours always exist).
pub(super) fn max_neighbor_step(grid: &[i64], stride: usize, gx: usize, gz: usize) -> i64 {
    let h = grid[gz * stride + gx];
    (h - grid[gz * stride + gx - 1])
        .abs()
        .max((h - grid[gz * stride + gx + 1]).abs())
        .max((h - grid[(gz - 1) * stride + gx]).abs())
        .max((h - grid[(gz + 1) * stride + gx]).abs())
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

/// The one downward surface-stage column loop both generators share.
/// Walks the column's `rows` samples top down (sample `j` at world y
/// `bottom + j · step`, `solid(j)` answering its solidity), tracking `d`,
/// the run of consecutive solid samples above the current row — in
/// samples, capped at [`SURFACE_LOOKAHEAD`] — and styles each of the
/// cube's own rows (the first `out.len()` samples) through
/// [`surface_block`] / [`surface_block_sea`]. `d` is rescaled into block
/// units by the stride, so at `step` 1 the states are exactly the
/// per-block rule's.
///
/// The **top solid sample** carries `true_surface` as its block y — the
/// column's surface refined upward from it (at most `step − 1` extra rows
/// per column), so the sand/grass belt stays one block wide instead of
/// one sample wide; `None` styles every row at its own y, which at
/// `step` 1 is the same thing (the refinement is empty). Sea styling
/// (`sea` and `top` both `Some`) fills water over submerged air; without
/// a sea the plain rules run.
#[allow(clippy::too_many_arguments)] // one explicit parameter per rule input beats a params struct at this size
pub(super) fn style_column(
    out: &mut [BlockStateId],
    solid: &mut dyn FnMut(usize) -> bool,
    rows: usize,
    bottom: i64,
    step: i64,
    true_surface: Option<i64>,
    sea: Option<i64>,
    top: Option<TopMaterial>,
    blocks: &Blocks,
) {
    let side = out.len();
    let mut d: u8 = 0; // consecutive solid samples above, unscaled
    let mut seen_solid = false; // the top solid sample has not passed yet
    for j in (0..rows).rev() {
        let y = bottom + step * j as i64;
        let is_solid = solid(j);
        if j < side {
            // d in block units, as the surface stage defines it.
            let d_blocks = ((i64::from(d) * step).min(SURFACE_LOOKAHEAD as i64)) as u8;
            let block_y = if is_solid && !seen_solid {
                true_surface.unwrap_or(y)
            } else {
                y
            };
            out[j] = match (sea, top) {
                (Some(sea), Some(top)) => {
                    surface_block_sea(is_solid, d_blocks, block_y, top, sea, blocks)
                }
                _ => surface_block(is_solid, d_blocks, block_y, blocks),
            };
        }
        seen_solid |= is_solid;
        d = if is_solid {
            (d + 1).min(SURFACE_LOOKAHEAD as u8)
        } else {
            0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BEACH_LEVEL, STEEP_SLOPE, STONE_LEVEL, TopMaterial, column_material, max_neighbor_step,
        style_column,
    };
    use jbm_core::block::BlockRegistry;
    use jbm_core::block::BlockStateId;
    use jbm_core::block::builtin::Blocks;

    fn blocks() -> Blocks {
        let mut registry = BlockRegistry::new();
        Blocks::register(&mut registry)
    }

    #[test]
    fn the_slope_bar_scales_with_the_stride() {
        // At step 1 the bar sits at STEEP_SLOPE exactly; each doubling of
        // the stride doubles it, so a level-n cube classifies the same
        // terrain the per-block rule would.
        assert_eq!(column_material(70, STEEP_SLOPE - 1, 1), TopMaterial::Grass);
        assert_eq!(column_material(70, STEEP_SLOPE, 1), TopMaterial::Stone);
        assert_eq!(
            column_material(70, STEEP_SLOPE * 2 - 1, 2),
            TopMaterial::Grass
        );
        assert_eq!(column_material(70, STEEP_SLOPE * 2, 2), TopMaterial::Stone);
        // Height bars do not scale.
        assert_eq!(column_material(STONE_LEVEL, 0, 4), TopMaterial::Stone);
        assert_eq!(column_material(BEACH_LEVEL, 0, 4), TopMaterial::Sand);
    }

    #[test]
    fn max_neighbor_step_reads_the_four_neighbours() {
        // A 3×3 grid (stride 3) with the center at 10 and its four
        // neighbours at 15 (north), 10 (south), 10 (west) and 10 (east):
        // the steepest step is the 5 to the north cell.
        let grid = [12, 15, 9, 10, 10, 10, 8, 10, 10];
        assert_eq!(max_neighbor_step(&grid, 3, 1, 1), 5);
    }

    /// One solid run of `height` cells at the bottom, air above, sampled
    /// at `step`-block stride from `bottom`; styled without a sea.
    fn styled_run(
        height: usize,
        rows: usize,
        bottom: i64,
        step: i64,
        blocks: &Blocks,
    ) -> Vec<BlockStateId> {
        let mut out = vec![BlockStateId::AIR; height];
        let mut solid = |j: usize| j < height;
        style_column(
            &mut out, &mut solid, rows, bottom, step, None, None, None, blocks,
        );
        out
    }

    #[test]
    fn stride_one_styles_the_classic_layering() {
        let blocks = blocks();
        // Surface at 70 (above the sand line): grass at 70, dirt 67..=69,
        // stone below.
        let out = styled_run(40, 44, 31, 1, &blocks);
        assert_eq!(out[70 - 31], blocks.grass_block);
        assert_eq!(out[69 - 31], blocks.dirt);
        assert_eq!(out[67 - 31], blocks.dirt);
        assert_eq!(out[66 - 31], blocks.stone);
        // A sand column: surface at 61, the sand line — sand down to
        // y = 58 (h − 3), stone below.
        let out = styled_run(32, 36, 30, 1, &blocks);
        assert_eq!(out[61 - 30], blocks.sand);
        assert_eq!(out[58 - 30], blocks.sand);
        assert_eq!(out[57 - 30], blocks.stone);
    }

    #[test]
    fn a_carried_true_surface_is_invisible_at_stride_one() {
        let blocks = blocks();
        // The same column styled with and without its true surface: at
        // step 1 the refinement is provably empty (the carried height is
        // the top solid sample's own y), so the two must agree. A carried
        // height that disagreed would leak — which is exactly why the
        // level-1 sampler never carries one.
        let mut plain = vec![BlockStateId::AIR; 8];
        let mut solid = |j: usize| j < 6;
        style_column(&mut plain, &mut solid, 10, 40, 1, None, None, None, &blocks);
        let mut carried = vec![BlockStateId::AIR; 8];
        let mut solid = |j: usize| j < 6;
        style_column(
            &mut carried,
            &mut solid,
            10,
            40,
            1,
            Some(45),
            None,
            None,
            &blocks,
        );
        assert_eq!(plain, carried);
    }
}
