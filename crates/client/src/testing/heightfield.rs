//! The heightfield-to-cube scene helpers the rings tests and benches
//! share: a cached height grid, the solid-leaning coarse sample over a
//! cell, and the surface layering rule. Each caller keeps its own
//! parameters (span, stride, sand level), so the cubes it builds are
//! exactly the ones it built before.

use std::ops::Range;

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;

/// A cached height grid: `height(x, z)` sampled every `step` blocks over
/// a rectangular span, so building a scene costs array reads, not
/// generator calls.
pub struct HeightGrid {
    min_x: i64,
    min_z: i64,
    /// Blocks between samples.
    step: i64,
    /// Samples per x row.
    side_z: usize,
    heights: Vec<i16>,
}

impl HeightGrid {
    /// Samples `height` over the half-open block ranges `xs` × `zs`, at
    /// every `step` blocks (step 1 caches every column). Both ranges must
    /// span a whole number of steps and be sampled at the same stride the
    /// readers will query.
    pub fn build(
        xs: Range<i64>,
        zs: Range<i64>,
        step: i64,
        height: impl Fn(i64, i64) -> i64,
    ) -> Self {
        assert!(
            (xs.end - xs.start) % step == 0 && (zs.end - zs.start) % step == 0,
            "grid span must be a whole number of {step}-block steps"
        );
        let side_z = ((zs.end - zs.start) / step) as usize;
        let mut heights = Vec::with_capacity(((xs.end - xs.start) / step) as usize * side_z);
        let min_x = xs.start;
        let min_z = zs.start;
        for x in xs.step_by(step as usize) {
            for z in zs.clone().step_by(step as usize) {
                heights.push(height(x, z) as i16);
            }
        }
        HeightGrid {
            min_x,
            min_z,
            step,
            side_z,
            heights,
        }
    }

    /// The sampled height at `(x, z)`, which must be a sample point (a
    /// multiple of `step` from the span's corner).
    pub fn at(&self, x: i64, z: i64) -> i64 {
        let i = ((x - self.min_x) / self.step) as usize;
        let j = ((z - self.min_z) / self.step) as usize;
        i64::from(self.heights[i * self.side_z + j])
    }

    /// The solid-leaning level-`level` sample of the heightfield for the
    /// cell with block corner `(x, z)` (a multiple of the cell size): the
    /// max height over the 3×3 grid at half the cell step — the corner
    /// points of the four child-level cells — so a coarser sample can
    /// never sit below the finer level's sample it replaces (that
    /// ordering keeps level seams skirt- and gap-free). A cell is filled
    /// at height `y` iff `y < max_over_cell(x, z, level)`.
    pub fn max_over_cell(&self, x: i64, z: i64, level: u8) -> i64 {
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

/// The surface layering: grass at the surface, dirt for the three blocks
/// below, stone at depth — and sand instead where `sand_level` is given
/// and the surface sits at or below it. `y > surface` is air, but the
/// caller decides that (the sampled scenes check the solid-leaning cell
/// fill first, so they can call this with a corner height below `y`).
pub fn layer_state(blocks: &Blocks, y: i64, surface: i64, sand_level: Option<i64>) -> BlockStateId {
    if sand_level.is_some_and(|level| surface <= level) {
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
