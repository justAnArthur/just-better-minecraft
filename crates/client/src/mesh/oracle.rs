//! Test-only reference meshers the greedy output is checked against, and
//! the seam any future mesher is verified through: a new sweep must
//! rasterize to the same naive unit faces and keep level 0 byte-identical
//! to the legacy sweep.

use jbm_core::block::BlockStateId;
use jbm_core::coords::CUBE_SIZE;
use jbm_core::cube::Cube;

use crate::mesh::appearance::BlockAppearance;
use crate::mesh::face::{DIRECTIONS, FaceFields, pack};
use crate::mesh::greedy::{LIGHT, MASK_LEN, corner_ao, stride};
use crate::mesh::input::{
    F_AIR, F_OPAQUE_CULL, FLAGS_SHIFT, MeshInput, MeshOutput, ORIGIN, PAD, PAD_Y, PAD_Z,
};

/// A deliberately independent reference mesher: one face per block,
/// reading cubes directly instead of any padded array.
pub(crate) fn naive_unit_faces(
    input: &MeshInput,
    appearance: &BlockAppearance,
) -> Vec<(u32, u32, u32, u8, u16, [u32; 4])> {
    let mut out = Vec::new();
    let state_at = |x: i32, y: i32, z: i32| -> Option<BlockStateId> {
        let clamp = |c: i32| {
            if c < 0 {
                -1
            } else if c >= CUBE_SIZE {
                1
            } else {
                0
            }
        };
        let (dx, dy, dz) = (clamp(x), clamp(y), clamp(z));
        let cube: &Cube = if (dx, dy, dz) == (0, 0, 0) {
            &input.center
        } else {
            input.neighbor(dx, dy, dz)?.as_ref()
        };
        Some(cube.get(jbm_core::coords::LocalPos {
            x: (x - dx * CUBE_SIZE) as u8,
            y: (y - dy * CUBE_SIZE) as u8,
            z: (z - dz * CUBE_SIZE) as u8,
        }))
    };
    for y in 0..CUBE_SIZE {
        for z in 0..CUBE_SIZE {
            for x in 0..CUBE_SIZE {
                let a = state_at(x, y, z).expect("center covers its local range");
                let app_a = appearance.get(a);
                if app_a.air {
                    continue;
                }
                for (di, direction) in DIRECTIONS.iter().enumerate() {
                    let n = direction.normal();
                    let (bx, by, bz) = (x + n[0], y + n[1], z + n[2]);
                    let b_opaque = match state_at(bx, by, bz) {
                        Some(state) => appearance.get(state).opaque,
                        // Missing cube: opaque for culling.
                        None => true,
                    };
                    if b_opaque {
                        continue;
                    }
                    if let Some(b) = state_at(bx, by, bz)
                        && b == a
                        && app_a.cutout
                    {
                        continue;
                    }
                    // AO, written independently of corner_ao: explicit
                    // sign arithmetic per corner.
                    let ua = direction.u_axis();
                    let va = direction.v_axis();
                    let offset_cell = |su: i32, sv: i32| -> bool {
                        let ox = bx + su * ua[0] + sv * va[0];
                        let oy = by + su * ua[1] + sv * va[1];
                        let oz = bz + su * ua[2] + sv * va[2];
                        match state_at(ox, oy, oz) {
                            Some(state) => appearance.get(state).opaque,
                            None => false,
                        }
                    };
                    let mut ao = [0u32; 4];
                    for (corner, (su, sv)) in
                        [(-1, -1), (1, -1), (1, 1), (-1, 1)].into_iter().enumerate()
                    {
                        let side1 = offset_cell(su, 0);
                        let side2 = offset_cell(0, sv);
                        let corner_cell = offset_cell(su, sv);
                        ao[corner] = if side1 && side2 {
                            0
                        } else {
                            3 - (u32::from(side1) + u32::from(side2) + u32::from(corner_cell))
                        };
                    }
                    out.push((x as u32, y as u32, z as u32, di as u8, app_a.layers[di], ao));
                }
            }
        }
    }
    out
}

/// The original greedy sweep, copied verbatim (modulo the skirts
/// counter): no octant splits, no skirts, no lids. Level 0 output must
/// match it byte for byte.
pub(crate) fn legacy_mesh_padded(pad: &[u32], appearance: &BlockAppearance) -> MeshOutput {
    let mut output = MeshOutput {
        faces: Vec::new(),
        ranges: std::array::from_fn(|_| 0u32..0),
        skirts: 0,
        lids: 0,
    };
    let size = CUBE_SIZE as usize;
    let mut mask = [0u32; MASK_LEN];
    for (dir_index, &direction) in DIRECTIONS.iter().enumerate() {
        let n = stride(direction.normal());
        let u = stride(direction.u_axis());
        let v = stride(direction.v_axis());
        let n_axis = n.abs();
        let last = CUBE_SIZE as isize - 1;
        let u_shift = if u < 0 { -last * u } else { 0 };
        let v_shift = if v < 0 { -last * v } else { 0 };
        let corner = ORIGIN as isize + u_shift + v_shift;
        let mut layer_of_kind = vec![0u16; appearance.len() + 1];
        for state in 0..appearance.len() {
            layer_of_kind[state + 1] = appearance.get(BlockStateId(state as u32)).layers[dir_index];
        }
        let start = output.faces.len() as u32;
        for k in 0..CUBE_SIZE as isize {
            let a_row = corner + k * n_axis;
            let b_row = a_row + n;
            for vi in 0..CUBE_SIZE as isize {
                for ui in 0..CUBE_SIZE as isize {
                    let a = pad[(a_row + ui * u + vi * v) as usize];
                    let b = pad[(b_row + ui * u + vi * v) as usize];
                    // Same-state culling for non-opaque blocks is updated
                    // here too: the reference pins everything else about
                    // the original sweep.
                    let culled = a & (F_AIR << FLAGS_SHIFT) != 0
                        || b & (F_OPAQUE_CULL << FLAGS_SHIFT) != 0
                        || a == b;
                    let key = if culled {
                        0
                    } else {
                        let layer = u32::from(layer_of_kind[(a & 0xFFFF) as usize]);
                        let ao = corner_ao(pad, b_row + ui * u + vi * v, u, v);
                        (1 << 31) | (layer << 15) | (ao << 7) | LIGHT
                    };
                    mask[(vi * CUBE_SIZE as isize + ui) as usize] = key;
                }
            }
            for v0 in 0..size {
                for u0 in 0..size {
                    let key = mask[v0 * size + u0];
                    if key == 0 {
                        continue;
                    }
                    let mut w = 1usize;
                    while u0 + w < size && mask[v0 * size + u0 + w] == key {
                        w += 1;
                    }
                    let mut h = 1usize;
                    'extend: while v0 + h < size {
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
                    let owner = (a_row + u0 as isize * u + v0 as isize * v) as usize;
                    let x = (owner % PAD) as u32 - 1;
                    let z = ((owner / PAD_Z) % PAD) as u32 - 1;
                    let y = (owner / PAD_Y) as u32 - 1;
                    let ao8 = (key >> 7) & 0xFF;
                    output.faces.push(pack(&FaceFields {
                        x,
                        y,
                        z,
                        width: w as u32,
                        height: h as u32,
                        direction,
                        layer: (key >> 15) & 0xFFFF,
                        ao: [ao8 & 3, (ao8 >> 2) & 3, (ao8 >> 4) & 3, (ao8 >> 6) & 3],
                        light: key & 0xF,
                    }));
                }
            }
        }
        output.ranges[dir_index] = start..output.faces.len() as u32;
    }
    output.skirts = 0;
    output
}
