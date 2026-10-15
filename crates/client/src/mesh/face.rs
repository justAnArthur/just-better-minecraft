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
/// | 28..32 | spare, zero |
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
/// | 24..28 | light 0..15 (always 15 until lighting exists) |
/// | 28..32 | spare, zero |
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

/// The four quad corners c0..c3 of a face in cube-local block units: the
/// CPU mirror of the corner math in `terrain.wgsl`'s `vs_main` (keep them
/// identical). The quad starts at the owner's corner one block along the
/// normal if it points positive, and one block along u or v if that axis
/// points negative, then spans `width` along u and `height` along v.
pub fn quad_corners(f: &FaceFields) -> [[i32; 3]; 4] {
    let n = f.direction.normal();
    let u = f.direction.u_axis();
    let v = f.direction.v_axis();
    let owner = [f.x as i32, f.y as i32, f.z as i32];
    let base: [i32; 3] =
        std::array::from_fn(|i| owner[i] + n[i].max(0) + (-u[i]).max(0) + (-v[i]).max(0));
    let (w, h) = (f.width as i32, f.height as i32);
    [(0, 0), (1, 0), (1, 1), (0, 1)]
        .map(|(cu, cv)| std::array::from_fn(|i| base[i] + u[i] * cu * w + v[i] * cv * h))
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
    /// Block light, 0..=15. Always 15 until lighting exists.
    pub light: u32,
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
    debug_assert!(f.light <= 15, "light out of 0..=15");
    let w0 = f.x
        | (f.y << 5)
        | (f.z << 10)
        | ((f.width - 1) << 15)
        | ((f.height - 1) << 20)
        | ((f.direction as u32) << 25);
    let ao = f.ao;
    debug_assert!(ao.iter().all(|&a| a <= 3), "AO out of 0..=3");
    let w1 =
        f.layer | (ao[0] << 16) | (ao[1] << 18) | (ao[2] << 20) | (ao[3] << 22) | (f.light << 24);
    PackedFace(w0, w1)
}

/// Inverse of [`pack`]. Spare bits are ignored.
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
        light: (w1 >> 24) & 15,
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
                light: 0,
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
                light: 15,
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
                light: 7,
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
                light: 15,
            },
        ];
        for (i, fields) in cases.iter().enumerate() {
            let packed = pack(fields);
            // Spare bits stay zero.
            assert_eq!(packed.0 >> 28, 0, "case {i}: word 0 spare bits set");
            assert_eq!(packed.1 >> 28, 0, "case {i}: word 1 spare bits set");
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
                light: 15,
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
                light: 15,
            };
            let corners = quad_corners(&fields);
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
