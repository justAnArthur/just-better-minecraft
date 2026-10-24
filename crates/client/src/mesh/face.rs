//! The 8-byte packed face format shared by the mesher (Rust) and the
//! terrain shader (WGSL, `render/shaders/terrain.wgsl`). Both sides decode
//! the same bits; keep this comment block and the one in terrain.wgsl in
//! sync.
//!
//! A face is a rectangle of `width x height` blocks owned by the block at
//! cube-local `(x, y, z)`, facing one of the six axis directions. Width
//! runs along the face's `u` axis and height along its `v` axis:
//!
//! | direction | value | normal | u axis | v axis |
//! |---|---|---|---|---|
//! | +X | 0 | ( 1, 0, 0) | ( 0, 0, -1) | (0, 1, 0) |
//! | -X | 1 | (-1, 0, 0) | ( 0, 0,  1) | (0, 1, 0) |
//! | +Y | 2 | ( 0, 1, 0) | ( 0, 0,  1) | (1, 0, 0) |
//! | -Y | 3 | ( 0,-1, 0) | ( 1, 0,  0) | (0, 0, 1) |
//! | +Z | 4 | ( 0, 0, 1) | ( 1, 0,  0) | (0, 1, 0) |
//! | -Z | 5 | ( 0, 0,-1) | (-1, 0,  0) | (0, 1, 0) |
//!
//! `cross(u, v)` equals the normal for every direction, so quad corners in
//! the order c0=(0,0), c1=(1,0), c2=(1,1), c3=(0,1) wind counter-clockwise
//! viewed from outside the block — the front face for back-face culling.
//! The AO corners are packed in that same c0..c3 order.
//!
//! **Insets** (task 2.14) move a level >= 1 face's edges by whole blocks
//! so a coarse column's surface sits at its true height: `inset_bottom`
//! raises a side face's lower edge, `inset_top` lowers a side face's upper
//! edge or a +Y face's plane. Both are 0 on level-0 faces and on -Y faces.

use bytemuck::{Pod, Zeroable};

use jbm_core::coords::CUBE_SIZE;

/// One packed face: 8 bytes, two `u32` words.
///
/// **Word 0:**
///
/// | bits | field |
/// |---|---|
/// | 0..5 | x (block that owns the face, cube-local) |
/// | 5..10 | y |
/// | 10..15 | z |
/// | 15..20 | width - 1 |
/// | 20..25 | height - 1 |
/// | 25..28 | direction (see the table above) |
/// | 28..32 | inset_bottom, low 4 bits |
///
/// **Word 1:**
///
/// | bits | field |
/// |---|---|
/// | 0..16 | texture layer |
/// | 16..18 | AO corner c0 (0..3) |
/// | 18..20 | AO corner c1 |
/// | 20..22 | AO corner c2 |
/// | 22..24 | AO corner c3 |
/// | 24..26 | inset_bottom, high 2 bits |
/// | 26..32 | inset_top |
///
/// Both insets are 6 bits, in **blocks** (not cells), 0..=63. Word 1 bits
/// 24..28 once held a light nibble that was always 15 and never read on
/// the GPU; task 2.14 repurposed them.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct PackedFace(pub u32, pub u32);

/// The six face directions. Discriminals are the wire values packed into
/// word 0, bits 25..28.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Direction {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

pub const DIRECTIONS: [Direction; 6] = [
    Direction::PosX,
    Direction::NegX,
    Direction::PosY,
    Direction::NegY,
    Direction::PosZ,
    Direction::NegZ,
];

impl Direction {
    /// Unit normal, cube-local axes.
    pub const fn normal(self) -> [i32; 3] {
        match self {
            Direction::PosX => [1, 0, 0],
            Direction::NegX => [-1, 0, 0],
            Direction::PosY => [0, 1, 0],
            Direction::NegY => [0, -1, 0],
            Direction::PosZ => [0, 0, 1],
            Direction::NegZ => [0, 0, -1],
        }
    }

    /// The u axis of the face's (u, v) frame; see the module table.
    pub const fn u_axis(self) -> [i32; 3] {
        match self {
            Direction::PosX => [0, 0, -1],
            Direction::NegX => [0, 0, 1],
            Direction::PosY => [0, 0, 1],
            Direction::NegY => [1, 0, 0],
            Direction::PosZ => [1, 0, 0],
            Direction::NegZ => [-1, 0, 0],
        }
    }

    /// The v axis of the face's (u, v) frame; see the module table.
    pub const fn v_axis(self) -> [i32; 3] {
        match self {
            Direction::PosX | Direction::NegX | Direction::PosZ | Direction::NegZ => [0, 1, 0],
            Direction::PosY => [1, 0, 0],
            Direction::NegY => [0, 0, 1],
        }
    }
}

/// The four quad corners c0..c3 of a face of a level-`level` cube, in
/// cube-local **block** units: the CPU mirror of the corner math in
/// `terrain.wgsl`'s `vs_main` (keep them identical). The quad starts at
/// the owner's corner one cell along the normal if it points positive,
/// and one cell along u or v if that axis points negative, then spans
/// `width` cells along u and `height` along v; everything scales by the
/// cell size `2^level`. The insets then move y by whole blocks: a side
/// face's lower corners rise by `inset_bottom` and its upper corners drop
/// by `inset_top`; a +Y face drops by `inset_top`.
pub fn quad_corners(f: &FaceFields, level: u8) -> [[i32; 3]; 4] {
    let n = f.direction.normal();
    let u = f.direction.u_axis();
    let v = f.direction.v_axis();
    let owner = [f.x as i32, f.y as i32, f.z as i32];
    let base: [i32; 3] =
        std::array::from_fn(|i| owner[i] + n[i].max(0) + (-u[i]).max(0) + (-v[i]).max(0));
    let (w, h) = (f.width as i32, f.height as i32);
    let scale = 1i32 << level;
    let (bottom, top) = (f.inset_bottom as i32, f.inset_top as i32);
    [(0, 0), (1, 0), (1, 1), (0, 1)].map(|(cu, cv)| {
        let mut corner: [i32; 3] =
            std::array::from_fn(|i| (base[i] + u[i] * cu * w + v[i] * cv * h) * scale);
        // v[1] is 1 on side faces only; n[1].max(0) is 1 on +Y only.
        corner[1] += v[1] * if cv == 0 { bottom } else { -top } - n[1].max(0) * top;
        corner
    })
}

/// The unpacked fields of a [`PackedFace`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceFields {
    /// Cube-local position of the block that owns the face, `0..CUBE_SIZE`.
    pub x: u32,
    pub y: u32,
    pub z: u32,
    /// Face extent along u, `1..=CUBE_SIZE`.
    pub width: u32,
    /// Face extent along v, `1..=CUBE_SIZE`.
    pub height: u32,
    pub direction: Direction,
    /// Texture array layer.
    pub layer: u32,
    /// AO per corner, c0..c3 order, 0 (dark) ..= 3 (open).
    pub ao: [u32; 4],
    /// Blocks the lower edge of a side face rises, 0..=63 (level >= 1).
    pub inset_bottom: u32,
    /// Blocks the upper edge of a side face, or the plane of a +Y face,
    /// drops, 0..=63 (level >= 1).
    pub inset_top: u32,
}

/// Packs fields into the two words. Debug-asserts the documented ranges so
/// a broken mesher fails loudly in tests instead of aliasing fields. The
/// 5-bit field layout is fixed whatever `CUBE_SIZE` is (16 fits); only the
/// asserts track the cube size.
pub fn pack(f: &FaceFields) -> PackedFace {
    let size = CUBE_SIZE as u32;
    debug_assert!(f.x < size && f.y < size && f.z < size, "owner out of cube");
    debug_assert!(
        f.width >= 1 && f.width <= size,
        "width out of 1..=CUBE_SIZE"
    );
    debug_assert!(
        f.height >= 1 && f.height <= size,
        "height out of 1..=CUBE_SIZE"
    );
    debug_assert!(f.layer <= u16::MAX as u32, "layer overflows 16 bits");
    debug_assert!(
        f.inset_bottom <= 63 && f.inset_top <= 63,
        "inset out of 0..=63"
    );
    let w0 = f.x
        | (f.y << 5)
        | (f.z << 10)
        | ((f.width - 1) << 15)
        | ((f.height - 1) << 20)
        | ((f.direction as u32) << 25)
        | ((f.inset_bottom & 15) << 28);
    let ao = f.ao;
    debug_assert!(ao.iter().all(|&a| a <= 3), "AO out of 0..=3");
    let w1 = f.layer
        | (ao[0] << 16)
        | (ao[1] << 18)
        | (ao[2] << 20)
        | (ao[3] << 22)
        | ((f.inset_bottom >> 4) << 24)
        | (f.inset_top << 26);
    PackedFace(w0, w1)
}

/// Inverse of [`pack`].
pub fn unpack(face: PackedFace) -> FaceFields {
    let (w0, w1) = (face.0, face.1);
    let direction = match (w0 >> 25) & 7 {
        0 => Direction::PosX,
        1 => Direction::NegX,
        2 => Direction::PosY,
        3 => Direction::NegY,
        4 => Direction::PosZ,
        5 => Direction::NegZ,
        other => panic!("packed face direction {other} is not 0..=5"),
    };
    FaceFields {
        x: w0 & 31,
        y: (w0 >> 5) & 31,
        z: (w0 >> 10) & 31,
        width: ((w0 >> 15) & 31) + 1,
        height: ((w0 >> 20) & 31) + 1,
        direction,
        layer: w1 & 0xFFFF,
        ao: [
            (w1 >> 16) & 3,
            (w1 >> 18) & 3,
            (w1 >> 20) & 3,
            (w1 >> 22) & 3,
        ],
        inset_bottom: (w0 >> 28) | (((w1 >> 24) & 3) << 4),
        inset_top: w1 >> 26,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_face_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<PackedFace>(), 8);
        assert_eq!(std::mem::align_of::<PackedFace>(), 4);
    }

    #[test]
    fn pack_unpack_round_trips_edge_values() {
        // Every field at its extreme (the cube-size bounds, not the 5-bit
        // field bounds), in combinations that exercise adjacent bit
        // boundaries.
        let last = (CUBE_SIZE - 1) as u32;
        let size = CUBE_SIZE as u32;
        let cases = [
            FaceFields {
                x: 0,
                y: 0,
                z: 0,
                width: 1,
                height: 1,
                direction: Direction::PosX,
                layer: 0,
                ao: [0, 0, 0, 0],
                inset_bottom: 0,
                inset_top: 0,
            },
            FaceFields {
                x: last,
                y: last,
                z: last,
                width: size,
                height: size,
                direction: Direction::NegZ,
                layer: u16::MAX as u32,
                ao: [3, 3, 3, 3],
                inset_bottom: 63,
                inset_top: 63,
            },
            FaceFields {
                x: last,
                y: 0,
                z: last,
                width: 1,
                height: size,
                direction: Direction::NegY,
                layer: 1,
                ao: [3, 0, 1, 2],
                inset_bottom: 17,
                inset_top: 0,
            },
            FaceFields {
                x: 1,
                y: last,
                z: 0,
                width: size,
                height: 1,
                direction: Direction::PosZ,
                layer: 10,
                ao: [2, 2, 0, 1],
                inset_bottom: 0,
                inset_top: 40,
            },
        ];
        for (i, fields) in cases.iter().enumerate() {
            let packed = pack(fields);
            // Zero insets (every level-0 face) leave bits 28..32 of both
            // words clear.
            if fields.inset_bottom == 0 && fields.inset_top == 0 {
                assert_eq!(packed.0 >> 28, 0, "case {i}: word 0 inset bits set");
                assert_eq!(packed.1 >> 24, 0, "case {i}: word 1 inset bits set");
            }
            assert_eq!(unpack(packed), *fields, "case {i} does not round-trip");
        }
        // All six directions survive the trip.
        for direction in DIRECTIONS {
            let fields = FaceFields {
                x: 5,
                y: 6,
                z: 7,
                width: 8,
                height: 9,
                direction,
                layer: 11,
                ao: [1, 2, 3, 0],
                inset_bottom: 0,
                inset_top: 0,
            };
            assert_eq!(unpack(pack(&fields)), fields, "{direction:?}");
        }
    }

    #[test]
    fn quad_corners_cover_exactly_the_owned_blocks() {
        // A w x h face starting at its owner must cover the union of the
        // unit faces of the blocks owner + i*u + j*v, on the face plane.
        for direction in DIRECTIONS {
            let (w, h) = (3u32, 2u32);
            let fields = FaceFields {
                x: 10,
                y: 10,
                z: 10,
                width: w,
                height: h,
                direction,
                layer: 0,
                ao: [3; 4],
                inset_bottom: 0,
                inset_top: 0,
            };
            let corners = quad_corners(&fields, 0);
            let n = direction.normal();
            let (u, v) = (direction.u_axis(), direction.v_axis());
            let mut lo = [i32::MAX; 3];
            let mut hi = [i32::MIN; 3];
            for j in 0..h as i32 {
                for i in 0..w as i32 {
                    for axis in 0..3 {
                        let block = 10 + i * u[axis] + j * v[axis];
                        // The unit face spans [block, block + 1] in-plane
                        // and sits at block + max(n, 0) along the normal.
                        let (a, b) = if n[axis] != 0 {
                            let p = block + n[axis].max(0);
                            (p, p)
                        } else {
                            (block, block + 1)
                        };
                        lo[axis] = lo[axis].min(a);
                        hi[axis] = hi[axis].max(b);
                    }
                }
            }
            for axis in 0..3 {
                let cmin = corners.iter().map(|c| c[axis]).min().unwrap();
                let cmax = corners.iter().map(|c| c[axis]).max().unwrap();
                assert_eq!(
                    (cmin, cmax),
                    (lo[axis], hi[axis]),
                    "{direction:?} axis {axis}"
                );
            }
        }
    }

    #[test]
    fn every_inset_pair_round_trips_beside_saturated_fields() {
        let last = (CUBE_SIZE - 1) as u32;
        for inset_bottom in 0..64 {
            for inset_top in 0..64 {
                let fields = FaceFields {
                    x: last,
                    y: last,
                    z: last,
                    width: CUBE_SIZE as u32,
                    height: CUBE_SIZE as u32,
                    direction: Direction::NegZ,
                    layer: u16::MAX as u32,
                    ao: [3; 4],
                    inset_bottom,
                    inset_top,
                };
                assert_eq!(unpack(pack(&fields)), fields);
            }
        }
    }

    #[test]
    fn quad_corners_apply_insets_in_blocks_after_the_level_scale() {
        // A level-2 face (4-block cells) owned by cell (3, 4, 5), 2 x 3
        // cells, bottom inset 1 block and top inset 2.
        let face = |direction| FaceFields {
            x: 3,
            y: 4,
            z: 5,
            width: 2,
            height: 3,
            direction,
            layer: 0,
            ao: [3; 4],
            inset_bottom: 1,
            inset_top: 2,
        };
        for direction in [
            Direction::PosX,
            Direction::NegX,
            Direction::PosZ,
            Direction::NegZ,
        ] {
            let ys = quad_corners(&face(direction), 2).map(|c| c[1]);
            // c0/c1 on the bottom edge (cell 4 = block 16, raised by 1),
            // c2/c3 on the top (cell 7 = block 28, lowered by 2).
            assert_eq!(ys, [17, 17, 26, 26], "{direction:?}");
        }
        // +Y: the plane at the top of cell 4 (block 20), lowered by 2.
        assert_eq!(
            quad_corners(&face(Direction::PosY), 2).map(|c| c[1]),
            [18; 4]
        );
        // In-plane coordinates only scale.
        let pos_y = quad_corners(&face(Direction::PosY), 2);
        assert_eq!(pos_y[0], [12, 18, 20]);
        assert_eq!(pos_y[2], [12 + 3 * 4, 18, 20 + 2 * 4]);
    }

    #[test]
    fn direction_frames_are_right_handed() {
        // cross(u, v) must equal the normal so quads wind CCW from outside.
        for direction in DIRECTIONS {
            let u = direction.u_axis();
            let v = direction.v_axis();
            let n = direction.normal();
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            assert_eq!(cross, n, "{direction:?}: cross(u, v) != normal");
        }
    }
}
