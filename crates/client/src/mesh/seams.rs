//! The seam rules, the crack hiders at level seams between LOD levels:
//! [`skirt`] for vertical seams (side directions), [`lid`] for horizontal
//! ones (+Y). Both apply only at level >= 1; level 0 output never
//! contains a seam face. The sweep in `super::greedy` consults them from
//! its culling branch.

use jbm_core::coords::CUBE_SIZE;

use crate::mesh::face::Direction;
use crate::mesh::input::{F_AIR, FLAGS_SHIFT, PAD, PAD_Y, PAD_Z};

/// Marks a face key emitted under the skirt rule (bit 30; bit 31 is the
/// non-zero flag).
pub(crate) const SKIRT_KEY_BIT: u32 = 1 << 30;

/// Marks a face key emitted under the lid rule (bit 6: above the 4 light
/// bits, below the AO byte at bit 7, so no packed field reads it).
pub(crate) const LID_KEY_BIT: u32 = 1 << 6;

/// The skirt rule, applied at level >= 1 in the side directions +-X/+-Z.
/// A skirt stands in for a coarse cell's side wall at a level seam,
/// hiding the crack where the neighboring volume is drawn by finer cubes
/// whose surface sits lower (the downsampling rule leans solid, so a
/// coarse surface can be up to 2^n - 1 blocks above the fine one):
///
/// ```text
///      fine side (level n-1)      coarse side (level n, this cube)
///                   air
///        ..........______..........______
///        grass    grass    ||  grass(grass)   <- top faces
///        dirt     dirt     ||  dirt          <- coarse cell c, filled
///        dirt     dirt     ||  dirt             the cell across the plane
///        stone    stone    ||  stone            is filled (or its cube is
///                                                 missing) -> normal rule
///                                                 would cull c's side face
/// ```
///
/// Emit c's face in direction d even though the cell across d is filled,
/// when ALL of these hold:
///
/// - c is filled;
/// - the cell across d lies in a different octant or a different cube (c
///   sits at the octant plane or the cube border along d);
/// - that cell is filled, or is in a missing neighbor cube;
/// - c is one of the top two filled cells of its column run: the cell
///   above c is air, or the cell two above is. At the cube's top row the
///   two-above cell is outside the padded array and counts as filled.
///
/// The face hangs down 2 cells worth of cover, enough for the 2^n - 1 the
/// rule leans solid. When the octant across is drawn by the same coarse
/// cube (no finer level arrived there), the skirt sits inside terrain and
/// is hidden by the neighbor's top face. Skirt faces are ordinary faces:
/// same AO, same texture, greedy-merged within the octant rule.
pub(crate) fn skirt(pad: &[u32], a_index: usize, direction: Direction) -> bool {
    let size = CUBE_SIZE as usize;
    let half = size / 2;
    let x = (a_index % PAD) as i32 - 1;
    let z = ((a_index / PAD_Z) % PAD) as i32 - 1;
    let y = (a_index / PAD_Y) as i32 - 1;
    // The cell across d lies in a different octant or a different cube.
    let boundary = match direction {
        Direction::PosX => x == half as i32 - 1 || x == size as i32 - 1,
        Direction::NegX => x == 0 || x == half as i32,
        Direction::PosZ => z == half as i32 - 1 || z == size as i32 - 1,
        Direction::NegZ => z == 0 || z == half as i32,
        Direction::PosY | Direction::NegY => false, // sides only
    };
    if !boundary {
        return false;
    }
    // One of the top two filled cells of the column run: air directly
    // above, or air two above (the top row's two-above cell is outside the
    // padded array and counts as filled).
    let air = |index: usize| pad[index] & (F_AIR << FLAGS_SHIFT) != 0;
    let above = a_index + PAD_Y;
    if air(above) {
        return true;
    }
    let two_above = a_index + 2 * PAD_Y;
    (y as usize) < size - 1 && air(two_above)
}

/// The lid rule, applied at level >= 1 in direction +Y: the skirt rule's
/// counterpart for horizontal seams. A cube whose upper octants (or the
/// cube above) are drawn by finer cubes would otherwise cull its top
/// faces at the seam against its OWN coarse cells across it, though the
/// finer cubes drawn there may be air at that height. Looking down a
/// slope that crosses the seam, the ray then meets no face at all: a
/// line of holes along the contour at the seam height.
///
/// Emit c's +Y face even though the cell above is filled, when c is
/// filled, the cell above is filled (or in a missing neighbor cube), and c
/// sits at y = CUBE_SIZE / 2 - 1 (the octant plane) or y = CUBE_SIZE - 1
/// (the cube border). There is no column restriction (unlike skirts): on
/// a cliff a finer column can be air at the seam however high the coarse
/// column above it reaches. Where the volume above is solid at the seam
/// (the same coarse cube's unmasked octant, or a finer cube that is
/// solid there) the lid lies inside terrain and is hidden; greedy merging
/// folds a buried lid plane into a few quads per octant.
pub(crate) fn lid(a_index: usize) -> bool {
    let size = CUBE_SIZE as usize;
    let y = a_index / PAD_Y - 1;
    y == size / 2 - 1 || y == size - 1
}
