//! Coordinate types for the unbounded grid of 32³ cubes.
//!
//! The world spans about ±68 billion blocks per axis: cube coordinates are
//! `i32` and block coordinates `i64`, because `i32::MIN * 32` overflows `i32`.
//! Conversions use floor arithmetic, so block -1 lives in cube -1 at local 31.

/// Side length of a cube, in blocks.
pub const CUBE_SIZE: i32 = 32;
/// Bits of local coordinate per axis (`log2(CUBE_SIZE)`).
pub const CUBE_BITS: u32 = 5;
/// Number of blocks in a cube (`CUBE_SIZE³`).
pub const CUBE_VOLUME: usize = 32768;

/// Which cube a block is in. Ordered by `(x, y, z)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CubePos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// A block position in world space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockPos {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

/// A block inside a cube; each axis is `0..32` (fields are `u8`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalPos {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl CubePos {
    /// World position of local block `local` inside this cube.
    pub fn block(self, local: LocalPos) -> BlockPos {
        BlockPos {
            x: i64::from(self.x) * i64::from(CUBE_SIZE) + i64::from(local.x),
            y: i64::from(self.y) * i64::from(CUBE_SIZE) + i64::from(local.y),
            z: i64::from(self.z) * i64::from(CUBE_SIZE) + i64::from(local.z),
        }
    }
}

impl BlockPos {
    /// Splits into the containing cube and the block's position inside it.
    /// Floor division: -1 splits to cube -1, local 31.
    pub fn split(self) -> (CubePos, LocalPos) {
        (
            CubePos {
                x: (self.x >> CUBE_BITS) as i32,
                y: (self.y >> CUBE_BITS) as i32,
                z: (self.z >> CUBE_BITS) as i32,
            },
            LocalPos {
                x: (self.x & 31) as u8,
                y: (self.y & 31) as u8,
                z: (self.z & 31) as u8,
            },
        )
    }
}

impl LocalPos {
    /// Returns the position if every axis is in `0..32`.
    pub fn new(x: u8, y: u8, z: u8) -> Option<Self> {
        if x < CUBE_SIZE as u8 && y < CUBE_SIZE as u8 && z < CUBE_SIZE as u8 {
            Some(Self { x, y, z })
        } else {
            None
        }
    }

    /// Same as [`LocalPos::new`], asserting validity in debug builds.
    pub fn new_unchecked(x: u8, y: u8, z: u8) -> Self {
        debug_assert!(
            x < CUBE_SIZE as u8 && y < CUBE_SIZE as u8 && z < CUBE_SIZE as u8,
            "local coordinate out of 0..32"
        );
        Self { x, y, z }
    }

    /// Linear index into cube arrays: `(y * 32 + z) * 32 + x`.
    /// x is fastest, then z, then y.
    pub fn index(self) -> usize {
        (self.y as usize * CUBE_SIZE as usize + self.z as usize) * CUBE_SIZE as usize
            + self.x as usize
    }

    /// Inverse of [`LocalPos::index`]. Debug-asserts `index < CUBE_VOLUME`.
    pub fn from_index(index: usize) -> Self {
        debug_assert!(index < CUBE_VOLUME, "index out of cube");
        Self {
            x: (index & 31) as u8,
            z: ((index >> CUBE_BITS) & 31) as u8,
            y: (index >> (CUBE_BITS * 2)) as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_cube_round_trip() {
        let cubes = [
            CubePos {
                x: i32::MIN,
                y: i32::MIN,
                z: i32::MIN,
            },
            CubePos {
                x: -1,
                y: -2,
                z: -3,
            },
            CubePos { x: 0, y: 0, z: 0 },
            CubePos { x: 1, y: 2, z: 3 },
            CubePos {
                x: i32::MAX,
                y: i32::MAX,
                z: i32::MAX,
            },
        ];
        let locals = [
            LocalPos { x: 0, y: 0, z: 0 },
            LocalPos {
                x: 31,
                y: 31,
                z: 31,
            },
            LocalPos { x: 17, y: 4, z: 9 },
        ];
        for cube in cubes {
            for local in locals {
                let (c, l) = cube.block(local).split();
                assert_eq!((c, l), (cube, local));
            }
        }
    }

    #[test]
    fn negative_one_splits_to_cube_minus_one() {
        let pos = BlockPos {
            x: -1,
            y: -33,
            z: -32,
        };
        let (cube, local) = pos.split();
        assert_eq!(
            (cube, local),
            (
                CubePos {
                    x: -1,
                    y: -2,
                    z: -1
                },
                LocalPos { x: 31, y: 31, z: 0 }
            )
        );
    }

    #[test]
    fn extreme_block_positions() {
        // i32::MIN * 32 overflows i32; the i64 math must survive it.
        let min = BlockPos {
            x: i64::from(i32::MIN) * 32,
            y: i64::from(i32::MIN) * 32 + 31,
            z: i64::from(i32::MAX) * 32,
        };
        let (cube, local) = min.split();
        assert_eq!(cube.x, i32::MIN);
        assert_eq!(cube.y, i32::MIN);
        assert_eq!(cube.z, i32::MAX);
        assert_eq!(local, LocalPos { x: 0, y: 31, z: 0 });
    }

    #[test]
    fn local_index_round_trip_all_cells() {
        for i in 0..CUBE_VOLUME {
            let pos = LocalPos::from_index(i);
            assert_eq!(pos.index(), i);
            assert_eq!(
                (pos.y as usize * 32 + pos.z as usize) * 32 + pos.x as usize,
                i
            );
        }
    }

    #[test]
    fn local_new_validates() {
        assert_eq!(LocalPos::new(0, 0, 0), Some(LocalPos { x: 0, y: 0, z: 0 }));
        assert_eq!(
            LocalPos::new(31, 31, 31),
            Some(LocalPos {
                x: 31,
                y: 31,
                z: 31
            })
        );
        assert_eq!(LocalPos::new(32, 0, 0), None);
        assert_eq!(LocalPos::new(0, 32, 0), None);
        assert_eq!(LocalPos::new(0, 0, 255), None);
    }
}
