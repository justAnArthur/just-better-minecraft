//! The greedy sweep over the padded cell view built in `super::input`:
//! merges coplanar faces with identical (layer, AO, light) into maximal
//! rectangles per direction. Rules are documented in
//! crates/client/README.md.
//!
//! The same sweep serves every LOD level: cells are cells whatever their
//! world size, so the face format is unchanged. One rule switches on for
//! level >= 1 (LOD0 output is bit-for-bit what it was):
//!
//! - **Octant-split merging.** Greedy rectangles never cross the cell
//!   planes x, y or z = `CUBE_SIZE / 2`, so every face lies in exactly one
//!   octant of its cube — which the renderer's octant masking needs. (In
//!   mask space the halves stay `[0, half)` and `[half, size)` however the
//!   u/v strides are signed, because the mask-to-world map is affine with
//!   slope +-1 per axis.)
//!
//! Two more level >= 1 rules — skirts and lids, the crack hiders at
//! level seams — live in `super::seams`.

use jbm_core::block::BlockStateId;
use jbm_core::coords::CUBE_SIZE;

use crate::mesh::appearance::BlockAppearance;
use crate::mesh::face::{DIRECTIONS, Direction, FaceFields, pack};
use crate::mesh::input::{F_AIR, F_OPAQUE_AO, F_OPAQUE_CULL, FLAGS_SHIFT, ORIGIN, PAD_Y, PAD_Z};
use crate::mesh::seams::{LID_KEY_BIT, SKIRT_KEY_BIT, lid, skirt};

use crate::mesh::input::MeshOutput;

/// Padded-index stride of an axis vector like (0, 0, -1).
#[inline]
pub(crate) fn stride(axis: [i32; 3]) -> isize {
    axis[0] as isize + PAD_Y as isize * axis[1] as isize + PAD_Z as isize * axis[2] as isize
}

/// Sweeps the six directions and greedily merges coplanar faces with
/// identical (layer, AO, light) into maximal rectangles. At level >= 1,
/// rectangles never cross an octant plane (see the module docs) and faces
/// covered by the skirt or lid rules are emitted with their key bit set.
pub fn mesh_padded(pad: &[u32], appearance: &BlockAppearance, level: u8) -> MeshOutput {
    sweep(pad, appearance, level, u32::MAX)
}

/// [`mesh_padded`] with merge-key bits masked out: every face key is
/// ANDed with `key_keep` before merging, so dropping the AO byte or the
/// layer field lets quads merge across them. Test-only measurement hook
/// (task 2.13 A's quad-split study); the packed layer/AO fields of the
/// result are garbage — only the face count is meaningful.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn sweep_key_masked(
    pad: &[u32],
    appearance: &BlockAppearance,
    level: u8,
    key_keep: u32,
) -> MeshOutput {
    sweep(pad, appearance, level, key_keep)
}

/// The mask of merge-key bits the AO byte occupies (bits 7..15).
#[cfg(any(test, feature = "testing"))]
pub(crate) const KEY_AO_BITS: u32 = 0xFF << 7;
/// The mask of merge-key bits the texture layer occupies (bits 15..30,
/// below the skirt bit).
#[cfg(any(test, feature = "testing"))]
pub(crate) const KEY_LAYER_BITS: u32 = 0x7FFF << 15;

/// The bitmask sweep (task 2.13 B). Instead of a per-cell key pass over
/// every slice, three walks over the padded array (one per axis) first
/// build the **boundary planes**: for each of the six directions, per
/// slice `k` and row `vi`, a `u32` of the u-cells whose A/B pair differs
/// — the only cells that can ever be visible (equal pairs are culled, and
/// so are air-A and opaque-B pairs, both tested per candidate). The sweep
/// then walks the set bits only, and the greedy merge runs over the set
/// bits of the key mask. A volume too noisy for that to pay off (the
/// pair-density along an axis rises above half — a checkerboard) is
/// detected while the planes are built; that axis's two directions sweep
/// every slice cell by cell instead, exactly like the reference. The
/// output — the same `PackedFace` list, order included, at every level —
/// is pinned bit for bit against `oracle::reference_mesh_padded`, the
/// verbatim pre-2.13 sweep.
fn sweep(pad: &[u32], appearance: &BlockAppearance, level: u8, key_keep: u32) -> MeshOutput {
    let mut output = MeshOutput {
        faces: Vec::new(),
        ranges: std::array::from_fn(|_| 0u32..0),
        skirts: 0,
        lids: 0,
    };
    let size = CUBE_SIZE as usize;
    let half = size / 2;
    let last = size - 1;
    let mut mask = [0u32; MASK_LEN];
    let mut bits = [0u32; CUBE_SIZE as usize];

    // The boundary planes, `[direction][k][vi]` with bit ui in that
    // direction's own mask layout (the u axis of a negative-stride
    // direction walks the world axis backwards, so its bits are mirrored
    // — each walk sets both directions' bits at once). `dense` marks the
    // axes whose walk aborted (see `build_differ_planes`).
    let mut differ = [[[0u32; CUBE_SIZE as usize]; CUBE_SIZE as usize]; 6];
    let dense = build_differ_planes(pad, &mut differ);

    for (dir_index, &direction) in DIRECTIONS.iter().enumerate() {
        let n = stride(direction.normal());
        let u = stride(direction.u_axis());
        let v = stride(direction.v_axis());
        // k is A's coordinate along the normal axis (always +1 per step);
        // only B = A + n moves along the signed normal.
        let n_axis = n.abs();
        // Walk u and v from the block corner where both strides are
        // positive, so negative-stride axes still cover blocks
        // 0..CUBE_SIZE - 1.
        let last_i = CUBE_SIZE as isize - 1;
        let u_shift = if u < 0 { -last_i * u } else { 0 };
        let v_shift = if v < 0 { -last_i * v } else { 0 };
        let corner = ORIGIN as isize + u_shift + v_shift;

        // layer-of-kind lookup for this direction (kind = state id + 1).
        let mut layer_of_kind = vec![0u16; appearance.len() + 1];
        for state in 0..appearance.len() {
            layer_of_kind[state + 1] = appearance.get(BlockStateId(state as u32)).layers[dir_index];
        }

        let side = matches!(
            direction,
            Direction::PosX | Direction::NegX | Direction::PosZ | Direction::NegZ
        );
        // A dense axis never got its planes: both of its directions take
        // the per-cell pass on every slice.
        let dense_dir = match direction {
            Direction::PosX | Direction::NegX => dense[0],
            Direction::PosY | Direction::NegY => dense[1],
            Direction::PosZ | Direction::NegZ => dense[2],
        };
        // The mask (u0, v0) cell of slice k is the A cell at cube
        // coordinates `cell_of(k, v0, u0)` — an affine map, because the
        // u/v/n frames are signed unit axes (the module docs). Decoding
        // the owner's padded index back through % and / gives the same
        // numbers; this spells them out per direction.
        let cell_of = |k: u32, v0: u32, u0: u32| -> (u32, u32, u32) {
            match direction {
                Direction::PosX => (k, v0, last as u32 - u0),
                Direction::NegX => (k, v0, u0),
                Direction::PosY => (v0, k, u0),
                Direction::NegY => (u0, k, v0),
                Direction::PosZ => (u0, v0, k),
                Direction::NegZ => (last as u32 - u0, v0, k),
            }
        };

        let start = output.faces.len() as u32;
        for (k, plane) in differ[dir_index].iter().enumerate() {
            let a_row = corner + k as isize * n_axis;
            // Seam rules only exist at level >= 1 and only on the slices
            // the boundary condition can hold (the slice axis IS the
            // boundary axis; `skirt`'s octant/cube planes per direction —
            // the negative directions boundary at 0 and half, the
            // positive ones at half-1 and size-1). Those slices — and
            // every slice of a dense direction — take the full per-cell
            // pass: the seam rules read culled cells too, and a dense
            // volume's planes cost more than the cells they skip.
            let seam_slice = level > 0
                && match direction {
                    Direction::PosX | Direction::PosY | Direction::PosZ => {
                        k == half - 1 || k == size - 1
                    }
                    Direction::NegX | Direction::NegZ => k == 0 || k == half,
                    Direction::NegY => false,
                };
            let full_pass = seam_slice || dense_dir;

            // Build the slice's visibility bits and keys. `mask` is all
            // zero here: the previous slice's merge consumed (and zeroed)
            // every key it found, so rows without candidates need no
            // clearing.
            let mut row_seeds = 0u32;
            for vi in 0..size {
                let a_base = a_row + vi as isize * v;
                if full_pass {
                    // The full per-cell pass, culled branch included: the
                    // seam rules read culled cells too. Off the seam
                    // slices (a dense direction's ordinary slices) the
                    // rules cannot fire — `skirt`/`lid` are position-
                    // gated — so this equals the reference sweep.
                    let mut row_bits = 0u32;
                    let mut a_i = a_base;
                    for ui in 0..size {
                        let b_i = a_i + n;
                        let a = pad[a_i as usize];
                        let b = pad[b_i as usize];
                        let a_air = a & (F_AIR << FLAGS_SHIFT) != 0;
                        let b_cull = b & (F_OPAQUE_CULL << FLAGS_SHIFT) != 0;
                        // Same state on both sides culls: opaque blocks
                        // via b_cull, and non-opaque ones too — water,
                        // glass and cutout leaves draw no faces between
                        // equal neighbors (an ocean once emitted every
                        // internal water face).
                        let culled = a_air || b_cull || a == b;
                        let key = if culled {
                            if level > 0
                                && side
                                && !a_air
                                && b_cull
                                && skirt(pad, a_i as usize, direction)
                            {
                                (1 << 31)
                                    | SKIRT_KEY_BIT
                                    | (u32::from(layer_of_kind[(a & 0xFFFF) as usize]) << 15)
                                    | (corner_ao(pad, b_i, u, v) << 7)
                                    | LIGHT
                            } else if level > 0
                                && direction == Direction::PosY
                                && !a_air
                                && b_cull
                                && lid(a_i as usize)
                            {
                                (1 << 31)
                                    | LID_KEY_BIT
                                    | (u32::from(layer_of_kind[(a & 0xFFFF) as usize]) << 15)
                                    | (corner_ao(pad, b_i, u, v) << 7)
                                    | LIGHT
                            } else {
                                0
                            }
                        } else {
                            let layer = u32::from(layer_of_kind[(a & 0xFFFF) as usize]);
                            let ao = corner_ao(pad, b_i, u, v);
                            (1 << 31) | (layer << 15) | (ao << 7) | LIGHT
                        };
                        if key != 0 {
                            mask[vi * size + ui] = key & key_keep;
                            row_bits |= 1 << ui;
                        }
                        a_i += u;
                    }
                    bits[vi] = row_bits;
                    row_seeds |= row_bits;
                } else {
                    // Candidates only: the boundary plane's set bits. A
                    // differing pair is visible unless A is air or B culls
                    // (equal pairs never made it here).
                    let mut candidates = plane[vi];
                    let mut row_bits = 0u32;
                    while candidates != 0 {
                        let ui = candidates.trailing_zeros() as usize;
                        candidates &= candidates - 1;
                        let a_i = a_base + ui as isize * u;
                        let b_i = a_i + n;
                        let a = pad[a_i as usize];
                        let b = pad[b_i as usize];
                        if a & (F_AIR << FLAGS_SHIFT) != 0
                            || b & (F_OPAQUE_CULL << FLAGS_SHIFT) != 0
                        {
                            continue;
                        }
                        let layer = u32::from(layer_of_kind[(a & 0xFFFF) as usize]);
                        let ao = corner_ao(pad, b_i, u, v);
                        let key = (1 << 31) | (layer << 15) | (ao << 7) | LIGHT;
                        mask[vi * size + ui] = key & key_keep;
                        row_bits |= 1 << ui;
                    }
                    bits[vi] = row_bits;
                    row_seeds |= row_bits;
                }
            }

            // Greedy merge over the set bits only: extend along u first,
            // then along v. At level >= 1 the extents stop at the octant
            // plane (mask indices [0, half) and [half, size) are the two
            // world halves whatever the stride signs — see the module
            // docs). Seeds walk (v0, u0) ascending exactly like the
            // reference's double loop; a consumed cell's bit may still be
            // set, so the mask is re-checked.
            if row_seeds != 0 {
                output
                    .faces
                    .reserve(bits.iter().map(|b| b.count_ones() as usize).sum());
                for v0 in 0..size {
                    let mut row_bits = bits[v0];
                    while row_bits != 0 {
                        let u0 = row_bits.trailing_zeros() as usize;
                        row_bits &= row_bits - 1;
                        let key = mask[v0 * size + u0];
                        if key == 0 {
                            continue; // consumed by an earlier rectangle
                        }
                        let u_limit = if level > 0 && u0 < half { half } else { size };
                        let mut w = 1usize;
                        while u0 + w < u_limit && mask[v0 * size + u0 + w] == key {
                            w += 1;
                        }
                        let v_limit = if level > 0 && v0 < half { half } else { size };
                        let mut h = 1usize;
                        'extend: while v0 + h < v_limit {
                            let row = (v0 + h) * size;
                            for ui in u0..u0 + w {
                                if mask[row + ui] != key {
                                    break 'extend;
                                }
                            }
                            h += 1;
                        }
                        for vv in v0..v0 + h {
                            for uu in u0..u0 + w {
                                mask[vv * size + uu] = 0;
                            }
                        }

                        // Owner block: the mask corner (u0, v0) of this
                        // slice, decoded by the affine cell map.
                        let (x, y, z) = cell_of(k as u32, v0 as u32, u0 as u32);
                        let ao8 = (key >> 7) & 0xFF;
                        output.faces.push(pack(&FaceFields {
                            x,
                            y,
                            z,
                            width: w as u32,
                            height: h as u32,
                            direction,
                            // 15 bits: bit 30 of the key is SKIRT_KEY_BIT,
                            // not part of the layer (reading 16 bits gave
                            // skirt faces layer + 32768: the texture array
                            // clamped it, the 2.9 per-layer color buffer
                            // read out of bounds).
                            layer: (key >> 15) & LAYER_KEY_MASK,
                            ao: [ao8 & 3, (ao8 >> 2) & 3, (ao8 >> 4) & 3, (ao8 >> 6) & 3],
                            light: key & 0xF,
                        }));
                        output.skirts += u32::from(key & SKIRT_KEY_BIT != 0);
                        output.lids += u32::from(key & LID_KEY_BIT != 0);
                    }
                }
            }
        }
        output.ranges[dir_index] = start..output.faces.len() as u32;
    }
    output
}

/// Pairs examined before the density abort may fire. Four rows' worth:
/// enough that a locally noisy start cannot flip a sparse volume, few
/// enough that a dense one wastes almost nothing.
const DENSE_MIN_PAIRS: u32 = 128;

/// Fills the boundary planes: one walk per axis over the padded array,
/// comparing every consecutive cell pair along that axis (the pairs the
/// two directions of the axis test) and setting, for each differing pair
/// at slice `k`, the candidate bit in both directions' layouts — the
/// positive-u direction's bit is the world-axis cell, the negative-u
/// direction's is its mirror (its u axis walks the world axis backwards;
/// see `cell_of`). Returns, per axis, whether the walk aborted as
/// **dense**: once at least [`DENSE_MIN_PAIRS`] pairs have been examined,
/// an axis whose differing pairs outnumber the equal ones stops building
/// planes and its directions fall back to the per-cell sweep — on noise
/// (a checkerboard: every pair differs) the planes cost more than the
/// cell tests they could ever save, while terrain stays at a few percent
/// and never trips this. Cube-size agnostic below 33 (a row must fit a
/// u32).
///
/// Every row is folded first (`fold |= a ^ b`, branch-free and
/// vectorizable) so uniform stretches — terrain interiors, uniform
/// cubes — never reach the per-pair bit extraction, which is itself
/// branchless: a mostly-uniform row with a few unpredictable boundaries
/// would otherwise pay a mispredict per cell.
// Each walk writes one differing pair into two differently-shaped plane
// layouts at scattered indices; range loops spell that out best.
#[allow(clippy::needless_range_loop)]
fn build_differ_planes(
    pad: &[u32],
    differ: &mut [[[u32; CUBE_SIZE as usize]; CUBE_SIZE as usize]; 6],
) -> [bool; 3] {
    let size = CUBE_SIZE as usize;
    let last = size - 1;
    let mut dense = [false; 3];

    // Pair j along an axis compares cube cells (j - 1, j); j = 0 touches
    // the left shell and j = size the right one. A positive direction's
    // slice k reads pair k + 1 (A = cube k, B = cube k + 1); a negative
    // direction's reads pair k (B = cube k - 1, the shell at k = 0). The
    // padded array is x-major, so every walk's pairs run as contiguous
    // x rows.
    //
    // x axis: serves +X (vi = y, ui = last - z) and -X (vi = y, ui = z).
    // A row's 33 pairs (shells included) read run[j] vs run[j + 1] from
    // one contiguous 34-word run starting at the left shell. The shells
    // are tested apart from the interior: on terrain the shells differ
    // wherever the row is below the surface while the interior stays
    // uniform, and a combined fold would push every such row through the
    // per-pair pass for nothing.
    let mut pairs = 0u32;
    let mut diffs = 0u32;
    'x: for y in 0..size {
        for z in 0..size {
            let run = &pad[ORIGIN + PAD_Y * y + PAD_Z * z - 1..][..size + 2];
            let bit_posx = 1u32 << (last - z);
            let bit_negx = 1u32 << z;
            let mut d = 0u64; // 33 pairs need more than a u32
            if run[0] != run[1] {
                d |= 1; // left shell, pair j = 0
            }
            if run[size + 1] != run[size] {
                d |= 1 << size; // right shell, pair j = size
            }
            let mut d_int = 0u64; // interior pairs j = 1..=size - 1
            let mut fold = 0u32;
            for t in 1..size {
                fold |= run[t + 1] ^ run[t];
            }
            if fold != 0 {
                for t in 1..size {
                    if run[t + 1] != run[t] {
                        d_int |= 1 << t;
                    }
                }
            }
            d |= d_int;
            // Density counts the interior only: the shell rows of a
            // boundary (a uniform cube facing air) are 100% dense by
            // design, and two planes of them say nothing about the
            // volume.
            pairs += size as u32 - 1;
            diffs += d_int.count_ones();
            if pairs >= DENSE_MIN_PAIRS && diffs * 2 > pairs {
                dense[0] = true;
                break 'x;
            }
            while d != 0 {
                let t = d.trailing_zeros() as usize;
                d &= d - 1;
                if t >= 1 {
                    differ[0][t - 1][y] |= bit_posx;
                }
                if t < size {
                    differ[1][t][y] |= bit_negx;
                }
            }
        }
    }
    // y axis: serves +Y (vi = x, ui = z) and -Y (vi = z, ui = x). Cube
    // (0, j, z) and its pair below, both contiguous in x. The -Y plane
    // takes the row's differ pattern word-wise (its layout is [k][z]
    // with bit x); the +Y plane is transposed, so its bits scatter.
    let mut pairs = 0u32;
    let mut diffs = 0u32;
    'y: for j in 0..=size {
        let interior_j = j >= 1 && j < size; // shell rows never count (above)
        let plane = ORIGIN + j * PAD_Y;
        for z in 0..size {
            let top = plane + PAD_Z * z;
            let below = top - PAD_Y;
            let mut fold = 0u32;
            for x in 0..size {
                fold |= pad[top + x] ^ pad[below + x];
            }
            if fold == 0 {
                continue;
            }
            let mut bits = 0u32;
            for x in 0..size {
                bits |= u32::from(pad[top + x] != pad[below + x]) << x;
            }
            if interior_j {
                pairs += size as u32;
                diffs += bits.count_ones();
                if pairs >= DENSE_MIN_PAIRS && diffs * 2 > pairs {
                    dense[1] = true;
                    break 'y;
                }
            }
            if j < size {
                differ[3][j][z] = bits;
            }
            let bit_posy = 1u32 << z;
            while bits != 0 {
                let x = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if j >= 1 {
                    differ[2][j - 1][x] |= bit_posy;
                }
            }
        }
    }
    // z axis: serves +Z (vi = y, ui = x) and -Z (vi = y, ui = last - x).
    // Cube (0, y, j) and its pair behind, both contiguous in x. The +Z
    // plane takes the pattern word-wise, the -Z plane its bit-mirror.
    let mut pairs = 0u32;
    let mut diffs = 0u32;
    'z: for j in 0..=size {
        let interior_j = j >= 1 && j < size; // shell rows never count (above)
        let plane = ORIGIN + j * PAD_Z;
        for y in 0..size {
            let top = plane + PAD_Y * y;
            let below = top - PAD_Z;
            let mut fold = 0u32;
            for x in 0..size {
                fold |= pad[top + x] ^ pad[below + x];
            }
            if fold == 0 {
                continue;
            }
            let mut bits = 0u32;
            for x in 0..size {
                bits |= u32::from(pad[top + x] != pad[below + x]) << x;
            }
            if interior_j {
                pairs += size as u32;
                diffs += bits.count_ones();
                if pairs >= DENSE_MIN_PAIRS && diffs * 2 > pairs {
                    dense[2] = true;
                    break 'z;
                }
            }
            if j >= 1 {
                differ[4][j - 1][y] = bits;
            }
            if j < size {
                // Mirror bit x to bit last - x (reverse_bits alone only
                // maps correctly at size 32).
                differ[5][j][y] = bits.reverse_bits() >> (32 - size as u32);
            }
        }
    }
    dense
}

/// The texture layer field of a face key: bits 15..30, below
/// SKIRT_KEY_BIT in `super::seams`.
pub(crate) const LAYER_KEY_MASK: u32 = 0x7FFF;

/// Light is always 15 until lighting exists.
pub(crate) const LIGHT: u32 = 15;

/// Cells in one u/v slice mask (`CUBE_SIZE²`).
pub(crate) const MASK_LEN: usize = (CUBE_SIZE as usize) * (CUBE_SIZE as usize);

/// AO of one unit face whose front block sits at padded index `front`,
/// with face strides `u` and `v`. Returns the four corner values packed
/// 2 bits each, c0..c3 in (u, v) corner order: 0 dark ..= 3 open.
#[inline]
pub(crate) fn corner_ao(pad: &[u32], front: isize, u: isize, v: isize) -> u32 {
    let occludes = |i: isize| pad[i as usize] & (F_OPAQUE_AO << FLAGS_SHIFT) != 0;
    let side_u_neg = occludes(front - u);
    let side_u_pos = occludes(front + u);
    let side_v_neg = occludes(front - v);
    let side_v_pos = occludes(front + v);
    let corner_neg_neg = occludes(front - u - v);
    let corner_pos_neg = occludes(front + u - v);
    let corner_pos_pos = occludes(front + u + v);
    let corner_neg_pos = occludes(front - u + v);
    let value = |side1: bool, side2: bool, corner: bool| -> u32 {
        if side1 && side2 {
            0
        } else {
            3 - (u32::from(side1) + u32::from(side2) + u32::from(corner))
        }
    };
    let c0 = value(side_u_neg, side_v_neg, corner_neg_neg);
    let c1 = value(side_u_pos, side_v_neg, corner_pos_neg);
    let c2 = value(side_u_pos, side_v_pos, corner_pos_pos);
    let c3 = value(side_u_neg, side_v_pos, corner_neg_pos);
    c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::mesh::face::{Direction, unpack};
    use crate::mesh::oracle::{legacy_mesh_padded, naive_unit_faces};
    use crate::mesh::{MeshInput, build_padded, mesh, neighbor_index};
    use jbm_core::block::builtin;
    use jbm_core::coords::CUBE_VOLUME;
    use jbm_core::cube::Cube;

    fn setup() -> (
        jbm_core::block::BlockRegistry,
        BlockAppearance,
        builtin::Blocks,
    ) {
        let mut registry = jbm_core::block::BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        let appearance = BlockAppearance::new(&registry);
        (registry, appearance, blocks)
    }

    fn mesh_isolated(cube: Cube, appearance: &BlockAppearance) -> MeshOutput {
        mesh(&MeshInput::isolated(Arc::new(cube)), appearance)
    }

    /// A mesh input whose 26 neighbors are all air.
    fn mesh_air_surrounded(cube: Cube, appearance: &BlockAppearance) -> MeshOutput {
        let air = Arc::new(Cube::default());
        let mut input = MeshInput::isolated(Arc::new(cube));
        for neighbor in &mut input.neighbors {
            *neighbor = Some(Arc::clone(&air));
        }
        mesh(&input, appearance)
    }

    /// Faces of one direction from the output.
    fn faces_of(output: &MeshOutput, direction: Direction) -> Vec<FaceFields> {
        let range = output.ranges[direction as usize].clone();
        output.faces[range.start as usize..range.end as usize]
            .iter()
            .map(|&f| unpack(f))
            .filter(|f| f.direction == direction)
            .collect()
    }

    #[test]
    fn single_stone_block_gives_six_unit_faces() {
        let (_, appearance, blocks) = setup();
        let cube = Cube::from_fn(|p| {
            if p.x == 0 && p.y == 0 && p.z == 0 {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        });
        let output = mesh_air_surrounded(cube, &appearance);
        assert_eq!(output.face_count(), 6);
        for direction in DIRECTIONS {
            let faces = faces_of(&output, direction);
            assert_eq!(faces.len(), 1, "{direction:?}");
            assert_eq!((faces[0].width, faces[0].height), (1, 1));
            assert_eq!((faces[0].x, faces[0].y, faces[0].z), (0, 0, 0));
            assert_eq!(faces[0].layer, crate::mesh::appearance::LAYER_STONE as u32);
            assert_eq!(faces[0].light, 15);
        }
    }

    #[test]
    fn two_by_two_stone_merges_to_2x2_faces() {
        let (_, appearance, blocks) = setup();
        let cube = Cube::from_fn(|p| {
            if p.x < 2 && p.y < 2 && p.z < 2 {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        });
        let output = mesh_air_surrounded(cube, &appearance);
        assert_eq!(output.face_count(), 6);
        for direction in DIRECTIONS {
            let faces = faces_of(&output, direction);
            assert_eq!(faces.len(), 1, "{direction:?}");
            assert_eq!((faces[0].width, faces[0].height), (2, 2));
        }
    }

    #[test]
    fn full_stone_cube_with_missing_neighbors_is_invisible() {
        let (_, appearance, blocks) = setup();
        let output = mesh_isolated(Cube::filled(blocks.stone), &appearance);
        assert_eq!(output.face_count(), 0);
    }

    #[test]
    fn full_stone_cube_with_air_neighbors_gives_six_max_faces() {
        let (_, appearance, blocks) = setup();
        let air = Arc::new(Cube::default());
        let mut input = MeshInput::isolated(Arc::new(Cube::filled(blocks.stone)));
        for i in 0..26 {
            input.neighbors[i] = Some(Arc::clone(&air));
        }
        let output = mesh(&input, &appearance);
        assert_eq!(output.face_count(), 6);
        for direction in DIRECTIONS {
            let faces = faces_of(&output, direction);
            assert_eq!(faces.len(), 1, "{direction:?}");
            assert_eq!(
                (faces[0].width, faces[0].height),
                (CUBE_SIZE as u32, CUBE_SIZE as u32)
            );
            // Every face is fully open: AO 3 on all corners.
            assert_eq!(faces[0].ao, [3, 3, 3, 3]);
        }
    }

    #[test]
    fn checkerboard_worst_case_is_half_volume_times_6_faces() {
        let (_, appearance, blocks) = setup();
        let cube = Cube::from_fn(|p| {
            if (p.x + p.y + p.z) % 2 == 0 {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        });
        let air = Arc::new(Cube::default());
        let mut input = MeshInput::isolated(Arc::new(cube));
        for i in 0..26 {
            input.neighbors[i] = Some(Arc::clone(&air));
        }
        let output = mesh(&input, &appearance);
        assert_eq!(output.face_count(), CUBE_VOLUME / 2 * 6);
        for direction in DIRECTIONS {
            assert_eq!(output.ranges[direction as usize].len(), CUBE_VOLUME / 2);
        }
    }

    #[test]
    fn glass_next_to_glass_culls_but_stone_next_to_glass_shows() {
        let (_, appearance, blocks) = setup();
        // One floor row: air, glass x1..=4, stone x5..=8, air. Glass and
        // stone touch at x=4 | x=5.
        let cube = Cube::from_fn(|p| {
            if p.y != 0 || p.z != CUBE_SIZE as u8 / 2 {
                return BlockStateId::AIR;
            }
            match p.x {
                1..=4 => blocks.glass,
                5..=8 => blocks.stone,
                _ => BlockStateId::AIR,
            }
        });
        let output = mesh_isolated(cube, &appearance);

        // Stone shows its face toward the (non-opaque) glass.
        let stone_nx: Vec<FaceFields> = faces_of(&output, Direction::NegX)
            .into_iter()
            .filter(|f| f.layer == crate::mesh::appearance::LAYER_STONE as u32)
            .collect();
        assert!(
            stone_nx
                .iter()
                .any(|f| f.x == 5 && f.z == CUBE_SIZE as u32 / 2 && f.y == 0),
            "stone face toward glass exists, got {stone_nx:?}"
        );
        // Glass shows no face toward the opaque stone...
        let glass_px: Vec<FaceFields> = faces_of(&output, Direction::PosX)
            .into_iter()
            .filter(|f| f.layer == crate::mesh::appearance::LAYER_GLASS as u32)
            .collect();
        assert!(
            glass_px.is_empty(),
            "glass face toward stone culled, got {glass_px:?}"
        );
        // ...and no glass-glass interior faces: only the outer end (x=1)
        // faces air with -X.
        let glass_nx: Vec<FaceFields> = faces_of(&output, Direction::NegX)
            .into_iter()
            .filter(|f| f.layer == crate::mesh::appearance::LAYER_GLASS as u32)
            .collect();
        assert!(
            glass_nx.iter().all(|f| f.x == 1),
            "only the outer glass block shows -X, got {:?}",
            glass_nx.iter().map(|f| f.x).collect::<Vec<_>>()
        );
        // Glass tops still render (cutout, not culled by air).
        assert!(
            faces_of(&output, Direction::PosY)
                .iter()
                .any(|f| f.layer == crate::mesh::appearance::LAYER_GLASS as u32),
            "glass top face exists"
        );
    }

    #[test]
    fn l_shaped_corner_shows_darkened_ao() {
        let (_, appearance, blocks) = setup();
        // An L on the floor: one arm along +X, one along +Z, meeting at
        // the corner block. The inner corner where the arms meet must show
        // AO darkening on the faces alongside it.
        let cube = Cube::from_fn(|p| {
            if p.y != 0 {
                return BlockStateId::AIR;
            }
            if (p.x < 6 && p.z == 0) || (p.z < 6 && p.x == 0) {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        });
        let output = mesh_isolated(cube, &appearance);
        // The +Z face of the x-arm block next to the z-arm (block (1,0,0))
        // has a corner occluded by the z-arm block at (0,0,1).
        let side = faces_of(&output, Direction::PosZ);
        let darkened: Vec<&FaceFields> = side
            .iter()
            .filter(|f| f.ao.iter().any(|&a| a < 3))
            .collect();
        assert!(
            darkened.iter().any(|f| f.x == 1 && f.z == 0 && f.y == 0),
            "the arm-junction face is darkened, got {darkened:?}"
        );
        // The z-arm's own +Z faces (further out) are fully open.
        assert!(
            side.iter()
                .any(|f| f.x == 0 && f.z >= 2 && f.ao == [3, 3, 3, 3]),
            "open faces away from the corner, got {side:?}"
        );
    }

    #[test]
    fn border_block_reads_neighbor_cube_for_culling_and_ao() {
        let (_, appearance, blocks) = setup();
        // Center: one stone block at the +X border, air elsewhere.
        let center = Arc::new(Cube::from_fn(|p| {
            if p.y == 0 && p.z == 0 && p.x == CUBE_SIZE as u8 - 1 {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        }));
        // Neighbor (+1, 0, 0): solid stone at its local (0, 0, 0) (right
        // against our border block) and at (0, 0, 1) (for AO).
        let neighbor = Arc::new(Cube::from_fn(|p| {
            if p.y == 0 && p.z <= 1 && p.x == 0 {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        }));
        let mut input = MeshInput::isolated(Arc::clone(&center));
        input.neighbors[neighbor_index(1, 0, 0)] = Some(neighbor);

        let output = mesh(&input, &appearance);
        // No +X face: the neighbor block against ours is opaque.
        assert_eq!(faces_of(&output, Direction::PosX).len(), 0);
        // The +Z face of our border block exists and its corners read the
        // neighbor: the neighbor stone at (CUBE_SIZE, 0, 1) world occludes.
        let pos_z = faces_of(&output, Direction::PosZ);
        assert_eq!(pos_z.len(), 1);
        assert!(
            pos_z[0].ao.iter().any(|&a| a < 3),
            "AO reads the neighbor cube, got {:?}",
            pos_z[0].ao
        );
    }

    #[test]
    fn uniform_air_is_fast_pathed_without_neighbors() {
        let (_, appearance, _) = setup();
        let output = mesh_isolated(Cube::default(), &appearance);
        assert_eq!(output.face_count(), 0);
    }

    /// Local xorshift, so tests never need the `rand` crate.
    struct XorShift64(u64);

    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    // ------------------------------------------------------------------
    // LOD levels

    /// A mesh input at `level` with air above and beside, a missing cube
    /// below (like a real terrain cube whose -Y neighbor never shows).
    fn mesh_air_surrounded_at(cube: Cube, appearance: &BlockAppearance, level: u8) -> MeshOutput {
        let air = Arc::new(Cube::default());
        let mut input = MeshInput::isolated_at_level(Arc::new(cube), level);
        for i in 0..26 {
            if i != neighbor_index(0, -1, 0) {
                input.neighbors[i] = Some(Arc::clone(&air));
            }
        }
        mesh(&input, appearance)
    }

    /// Every cell a face covers, as cube-local cell coordinates.
    fn face_cells(f: &FaceFields) -> Vec<(i32, i32, i32)> {
        let u = f.direction.u_axis();
        let v = f.direction.v_axis();
        let mut cells = Vec::new();
        for j in 0..f.height as i32 {
            for i in 0..f.width as i32 {
                cells.push((
                    f.x as i32 + i * u[0] + j * v[0],
                    f.y as i32 + i * u[1] + j * v[1],
                    f.z as i32 + i * u[2] + j * v[2],
                ));
            }
        }
        cells
    }

    #[test]
    fn lod_faces_never_cross_an_octant_plane() {
        let (_, appearance, blocks) = setup();
        let half = CUBE_SIZE / 2;
        for case in 0..8u64 {
            // Random-terrain cube (seeded), meshed at level 2.
            let cube = Cube::from_fn(|p| {
                let mut h = case ^ (p.index() as u64).wrapping_mul(0x9e37_79b9);
                h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                h ^= h >> 31;
                match h % 10 {
                    0..=2 => blocks.stone,
                    3 => blocks.dirt,
                    _ => BlockStateId::AIR,
                }
            });
            let output = mesh_air_surrounded_at(cube, &appearance, 2);
            assert!(!output.faces.is_empty(), "case {case}: terrain has faces");
            for &face in &output.faces {
                let f = unpack(face);
                let cells = face_cells(&f);
                for &(x, y, z) in &cells {
                    for &(ox, oy, oz) in &cells {
                        assert_eq!(
                            (ox < half, oy < half, oz < half),
                            (x < half, y < half, z < half),
                            "face {f:?} crosses an octant plane"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lod0_output_is_byte_identical_to_the_pre_change_mesher() {
        let (_, appearance, blocks) = setup();
        let states = [BlockStateId::AIR, blocks.stone, blocks.dirt, blocks.glass];
        let mut rng = XorShift64(0x1234_5678_9abc_def0);
        for case in 0..20u64 {
            let fill = |seed: u64| {
                Cube::from_fn(move |p| {
                    let mut h = seed ^ (p.index() as u64).wrapping_mul(0x9e37_79b9);
                    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    h ^= h >> 31;
                    states[(h % 100) as usize % states.len()]
                })
            };
            let center = Arc::new(fill(case.wrapping_mul(0x1234_5678) ^ rng.next()));
            let mut input = MeshInput::isolated(Arc::clone(&center));
            for i in 0..26 {
                if rng.next().is_multiple_of(3) {
                    let cube = if rng.next().is_multiple_of(2) {
                        Cube::filled(blocks.stone)
                    } else {
                        fill(rng.next())
                    };
                    input.neighbors[i] = Some(Arc::new(cube));
                }
            }
            let padded = build_padded(&input, &appearance);
            let got = mesh_padded(&padded, &appearance, 0);
            let expected = legacy_mesh_padded(&padded, &appearance);
            assert_eq!(
                got.faces, expected.faces,
                "case {case}: LOD0 output drifted"
            );
            assert_eq!(got.ranges, expected.ranges, "case {case}: ranges drifted");
            assert_eq!(got.skirts, 0, "no skirts at level 0");
        }
    }

    /// Faces of `direction` covering cell `(x, y, z)`, if any.
    fn faces_covering(
        output: &MeshOutput,
        direction: Direction,
        x: i32,
        y: i32,
        z: i32,
    ) -> Vec<FaceFields> {
        output
            .faces
            .iter()
            .map(|&f| unpack(f))
            .filter(|f| f.direction == direction && face_cells(f).contains(&(x, y, z)))
            .collect()
    }

    #[test]
    fn a_step_across_an_octant_plane_emits_the_skirt_two_cells_deep() {
        let (_, appearance, blocks) = setup();
        let half = CUBE_SIZE / 2;
        // Left half surface at cell y = 10, right half one cell lower at
        // y = 9: the surface drops by one cell across the octant plane.
        let cube = Cube::from_fn(|p| {
            let surface = if (p.x as i32) < half { 10 } else { 9 };
            if (p.y as i32) <= surface {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        });
        let output = mesh_air_surrounded_at(cube, &appearance, 1);
        // The higher side's border cell (x = half - 1) skirts toward +X,
        // two cells deep (the top two filled cells of its column).
        for y in [9, 10] {
            assert!(
                !faces_covering(&output, Direction::PosX, half - 1, y, 5).is_empty(),
                "skirt missing at x = {}, y = {y}",
                half - 1
            );
        }
        // Below the two-cell band: no skirt (the rule keeps skirts near
        // the surface).
        assert!(faces_covering(&output, Direction::PosX, half - 1, 8, 5).is_empty());
        assert!(faces_covering(&output, Direction::PosX, half - 1, 3, 5).is_empty());
        assert!(output.skirts > 0);
        // Skirt faces carry the block's real texture layer: the skirt key
        // bit must not leak into the layer field (it once read as layer +
        // 32768, out of bounds for the per-layer color buffer).
        let stone_layer = appearance.get(blocks.stone).layers[0] as u32;
        for f in faces_covering(&output, Direction::PosX, half - 1, 10, 5) {
            assert_eq!(f.layer, stone_layer, "skirt face layer");
        }
        // Interior columns (away from the plane) never skirt toward +X.
        assert!(faces_covering(&output, Direction::PosX, half - 3, 10, 5).is_empty());
    }

    #[test]
    fn a_flat_surface_across_the_plane_emits_hidden_skirts() {
        let (_, appearance, blocks) = setup();
        let half = CUBE_SIZE / 2;
        let cube = Cube::from_fn(|p| {
            if (p.y as i32) <= 10 {
                blocks.stone
            } else {
                BlockStateId::AIR
            }
        });
        let output = mesh_air_surrounded_at(cube, &appearance, 1);
        // The skirt faces exist at the top two cells of the border
        // columns on both sides of the plane — they sit flush inside the
        // neighbor's terrain, under its top face at y = 10.
        for (x, dir) in [(half - 1, Direction::PosX), (half, Direction::NegX)] {
            for y in [9, 10] {
                assert!(
                    !faces_covering(&output, dir, x, y, 5).is_empty(),
                    "skirt missing at x = {x}, y = {y}"
                );
            }
        }
        // An interior column of the flat surface has no +X faces at all:
        // the neighbor cell is filled stone.
        assert!(faces_covering(&output, Direction::PosX, half - 3, 10, 5).is_empty());
    }

    #[test]
    fn a_missing_neighbor_cube_at_the_cube_border_gets_a_skirt() {
        let (_, appearance, blocks) = setup();
        let last = CUBE_SIZE - 1;
        // Isolated input: every neighbor cube is missing. A surface
        // column at the +X border skirts into the missing cube.
        let make = || {
            Cube::from_fn(|p| {
                if (p.y as i32) <= 10 {
                    blocks.stone
                } else {
                    BlockStateId::AIR
                }
            })
        };
        let input = MeshInput::isolated_at_level(Arc::new(make()), 1);
        let output = mesh(&input, &appearance);
        for y in [9, 10] {
            assert!(
                !faces_covering(&output, Direction::PosX, last, y, 5).is_empty(),
                "border skirt missing at y = {y}"
            );
        }
        // Level 0 has no skirts: the border column's +X faces toward the
        // missing cube stay culled.
        let input = MeshInput::isolated_at_level(Arc::new(make()), 0);
        let output = mesh(&input, &appearance);
        assert!(faces_covering(&output, Direction::PosX, last, 10, 5).is_empty());
    }

    #[test]
    fn a_body_of_water_meshes_only_its_outer_faces() {
        let (_, appearance, blocks) = setup();
        // An 8x8x8 block of water in air: equal non-opaque neighbors cull,
        // so the mesh is its six outer sides, one merged quad each — not
        // the 3 x 7 x 64 internal faces it once also emitted. Glass too.
        for state in [blocks.water, blocks.glass] {
            let cube = Cube::from_fn(|p| {
                if p.x < 8 && p.y < 8 && p.z < 8 {
                    state
                } else {
                    BlockStateId::AIR
                }
            });
            let output = mesh_air_surrounded(cube, &appearance);
            assert_eq!(output.faces.len(), 6, "{state:?}: outer faces only");
        }
    }

    #[test]
    fn terrain_through_the_octant_plane_and_cube_top_gets_lids() {
        let (_, appearance, blocks) = setup();
        let half = CUBE_SIZE / 2;
        let last = CUBE_SIZE - 1;
        // A slope that crosses the y octant plane: columns x < 8 reach the
        // cube's top row (into the missing cube above), the rest end at
        // y = half + 3, so the plane cells under it are buried.
        let make = || {
            Cube::from_fn(|p| {
                let surface = if (p.x as i32) < 8 { last } else { half + 3 };
                if (p.y as i32) <= surface {
                    blocks.stone
                } else {
                    BlockStateId::AIR
                }
            })
        };
        let input = MeshInput::isolated_at_level(Arc::new(make()), 1);
        let output = mesh(&input, &appearance);
        // A lid at the octant plane (y = half - 1) in every column: the
        // upper octants may be drawn by finer cubes that are air there.
        for x in [2, 12, 20] {
            assert!(
                !faces_covering(&output, Direction::PosY, x, half - 1, 5).is_empty(),
                "plane lid missing at x = {x}"
            );
        }
        // A lid at the cube's top row under the missing cube above.
        assert!(!faces_covering(&output, Direction::PosY, 2, last, 5).is_empty());
        // No lid in the interior (between the planes).
        assert!(faces_covering(&output, Direction::PosY, 2, half + 2, 5).is_empty());
        assert!(output.lids > 0);
        // Buried lid planes merge: a few quads per octant (AO at the
        // slope's foot splits some), not one per cell (2 x 32 x 32).
        assert!(output.lids <= 32, "lids {} did not merge", output.lids);
        // Level 0 has no lids: LOD0 output stays the original mesher's.
        let input = MeshInput::isolated_at_level(Arc::new(make()), 0);
        let output = mesh(&input, &appearance);
        assert_eq!(output.lids, 0);
        assert!(faces_covering(&output, Direction::PosY, 2, half - 1, 5).is_empty());
    }

    #[test]
    fn greedy_output_rasterizes_to_the_naive_faces() {
        let (_, appearance, blocks) = setup();
        let states = [BlockStateId::AIR, blocks.stone, blocks.dirt, blocks.glass];
        let mut rng = XorShift64(0x0123_4567_89ab_cdef);
        for case in 0..200u64 {
            let fill = |seed: u64| {
                Cube::from_fn(move |p| {
                    let mut h = seed ^ (p.index() as u64).wrapping_mul(0x9e37_79b9);
                    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    h ^= h >> 31;
                    states[(h % 100) as usize % states.len()]
                })
            };
            let center = Arc::new(fill(case.wrapping_mul(0x1234_5678) ^ rng.next()));
            let mut input = MeshInput::isolated(Arc::clone(&center));
            // A random subset of neighbors, some uniform, some noisy.
            for i in 0..26 {
                if rng.next().is_multiple_of(3) {
                    let cube = if rng.next().is_multiple_of(2) {
                        Cube::filled(blocks.stone)
                    } else {
                        fill(rng.next())
                    };
                    input.neighbors[i] = Some(Arc::new(cube));
                }
            }

            let output = mesh(&input, &appearance);
            // Rasterize merged faces back to unit faces.
            let mut rasterized = Vec::new();
            for &face in &output.faces {
                let f = unpack(face);
                let ua = f.direction.u_axis();
                let va = f.direction.v_axis();
                for j in 0..f.height as i32 {
                    for i in 0..f.width as i32 {
                        let x = f.x as i32 + i * ua[0] + j * va[0];
                        let y = f.y as i32 + i * ua[1] + j * va[1];
                        let z = f.z as i32 + i * ua[2] + j * va[2];
                        rasterized.push((
                            x as u32,
                            y as u32,
                            z as u32,
                            f.direction as u8,
                            f.layer as u16,
                            f.ao,
                        ));
                    }
                }
            }
            // No two output faces overlap: rasterization is injective.
            let mut unique = rasterized.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(
                unique.len(),
                rasterized.len(),
                "case {case}: overlapping faces after merge"
            );

            let mut expected = naive_unit_faces(&input, &appearance);
            expected.sort_unstable();
            let mut got = rasterized;
            got.sort_unstable();
            assert_eq!(
                got, expected,
                "case {case}: greedy rasterization differs from the naive mesher"
            );

            // Ranges group faces by direction.
            for (di, range) in output.ranges.iter().enumerate() {
                for &face in &output.faces[range.start as usize..range.end as usize] {
                    assert_eq!(unpack(face).direction as usize, di);
                }
            }
        }
    }

    /// Task 2.13 B: the bitmask sweep must equal the verbatim pre-2.13
    /// sweep bit for bit at EVERY level — same PackedFace list, order
    /// included, plus ranges, skirts and lids. 2,100 seeded cases over
    /// mixed opaque / water / glass / leaves cubes with random neighbor
    /// patterns (missing, uniform, noisy) at levels 0..=3, so the octant
    /// splits, skirts and lids all fire.
    #[test]
    fn bitmask_sweep_matches_the_reference_at_every_level() {
        let (_, appearance, blocks) = setup();
        let states = [
            BlockStateId::AIR,
            blocks.stone,
            blocks.dirt,
            blocks.glass,
            blocks.water,
            blocks.leaves,
        ];
        let mut rng = XorShift64(0x2a13_0d15_eed5_1713);
        for case in 0..2100u64 {
            let fill = |seed: u64| {
                Cube::from_fn(move |p| {
                    let mut h = seed ^ (p.index() as u64).wrapping_mul(0x9e37_79b9);
                    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    h ^= h >> 31;
                    states[(h % 100) as usize % states.len()]
                })
            };
            let center = Arc::new(fill(case.wrapping_mul(0x5173_171d) ^ rng.next()));
            let mut input = MeshInput::isolated_at_level(
                Arc::clone(&center),
                (case % 4) as u8, // levels 0..=3
            );
            for i in 0..26 {
                let roll = rng.next() % 4;
                if roll == 0 {
                    continue; // missing neighbor
                }
                let cube = if roll == 1 {
                    Cube::filled(match rng.next() % 4 {
                        0 => blocks.stone,
                        1 => blocks.water,
                        2 => blocks.glass,
                        _ => BlockStateId::AIR,
                    })
                } else {
                    fill(rng.next())
                };
                input.neighbors[i] = Some(Arc::new(cube));
            }
            let padded = build_padded(&input, &appearance);
            let got = mesh_padded(&padded, &appearance, input.level);
            let expected =
                crate::mesh::oracle::reference_mesh_padded(&padded, &appearance, input.level);
            assert!(
                got.faces == expected.faces,
                "case {case} (level {}): face list drifted",
                input.level
            );
            assert_eq!(got.ranges, expected.ranges, "case {case}: ranges drifted");
            assert_eq!(got.skirts, expected.skirts, "case {case}: skirts drifted");
            assert_eq!(got.lids, expected.lids, "case {case}: lids drifted");
        }
    }

    /// Task 2.13 B follow-up: the noisy corpus above now mostly lands in
    /// the dense per-cell fallback (random fills differ everywhere), so
    /// the sparse candidate path — the one real terrain takes — needs its
    /// own oracle coverage, seam slices and all. Smooth heightmap cubes
    /// (a few percent of pairs differ) at levels 0..=3 with mixed
    /// neighbor kinds, checked bit for bit like the noisy corpus.
    #[test]
    fn sparse_terrain_matches_the_reference_with_seams() {
        let (_, appearance, blocks) = setup();
        let height_at = |x: u8, z: u8, seed: u64| -> u8 {
            let mut h = seed ^ (u64::from(x) << 8) ^ (u64::from(z) << 16);
            h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            h ^= h >> 31;
            // Smooth-ish: base gradient plus a wobble, 4..=29 like the
            // bench's terrain_like.
            (4 + (x / 3) * 2 + (z / 4) + (h % 3) as u8).clamp(4, 29)
        };
        let mut rng = XorShift64(0x00c0_ffee_2026);
        for case in 0..600u64 {
            let seed = case.wrapping_mul(0x2f13_171d) ^ rng.next();
            let center = Arc::new(Cube::from_fn(move |p| {
                let height = height_at(p.x, p.z, seed);
                if p.y > height {
                    BlockStateId::AIR
                } else if p.y == height {
                    blocks.grass_block
                } else if p.y + 3 >= height {
                    blocks.dirt
                } else {
                    blocks.stone
                }
            }));
            // Neighboring terrain offset by one seed step: same shape,
            // heights shifted a little — so octant planes and cube
            // borders see real steps (skirts, lids).
            let neighbor_cube = |seed: u64| {
                Arc::new(Cube::from_fn(move |p| {
                    let height = height_at(p.x, p.z, seed);
                    if p.y > height {
                        BlockStateId::AIR
                    } else if p.y + 3 >= height {
                        blocks.dirt
                    } else {
                        blocks.stone
                    }
                }))
            };
            let mut input = MeshInput::isolated_at_level(
                Arc::clone(&center),
                (case % 4) as u8, // levels 0..=3
            );
            for i in 0..26 {
                let roll = rng.next() % 4;
                if roll == 0 {
                    continue; // missing neighbor: skirts into it
                }
                let cube = if roll == 1 {
                    Arc::new(Cube::filled(blocks.stone))
                } else {
                    neighbor_cube(seed ^ rng.next())
                };
                input.neighbors[i] = Some(cube);
            }
            let padded = build_padded(&input, &appearance);
            let got = mesh_padded(&padded, &appearance, input.level);
            let expected =
                crate::mesh::oracle::reference_mesh_padded(&padded, &appearance, input.level);
            assert!(
                got.faces == expected.faces,
                "case {case} (level {}): sparse-path face list drifted",
                input.level
            );
            assert_eq!(got.ranges, expected.ranges, "case {case}: ranges drifted");
            assert_eq!(got.skirts, expected.skirts, "case {case}: skirts drifted");
            assert_eq!(got.lids, expected.lids, "case {case}: lids drifted");
        }
    }

    /// The affine owner decode: every direction's (x, y, z) matches the
    /// index arithmetic the reference uses, over a slice-full of seeds.
    /// (Covered implicitly by the case test above, but this pins the map
    /// itself for every direction at a glance.)
    #[test]
    fn affine_cell_map_matches_the_reference_index_decode() {
        let (_, appearance, blocks) = setup();
        let cube = Cube::from_fn(|p| {
            let mut h = (p.index() as u64).wrapping_mul(0x9e37_79b9);
            h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            h ^= h >> 31;
            match h % 12 {
                0..=3 => blocks.stone,
                4 => blocks.glass,
                5 => blocks.water,
                _ => BlockStateId::AIR,
            }
        });
        let air = Arc::new(Cube::default());
        let mut input = MeshInput::isolated(Arc::new(cube));
        for i in 0..26 {
            input.neighbors[i] = Some(Arc::clone(&air));
        }
        let padded = build_padded(&input, &appearance);
        let got = mesh_padded(&padded, &appearance, 0);
        let expected = crate::mesh::oracle::reference_mesh_padded(&padded, &appearance, 0);
        // Byte equality of the packed faces is the claim; the owners ride
        // in word 0.
        for (g, e) in got.faces.iter().zip(&expected.faces) {
            assert_eq!(g.0, e.0);
        }
        assert_eq!(got.faces.len(), expected.faces.len());
    }
}
