//! The server's pyramid of LOD cubes (task 2.4), built from the real world
//! by the core's downsampling rule. Levels are 1-based: level *n* cells are
//! 2ⁿ blocks per side, and a level-*n* cube covers the same volume as a
//! 2×2×2 group of level-(*n*−1) cubes.
//!
//! Every LOD0 cube that enters the world is downsampled into its level-1
//! parent ([`LodPyramid::insert_lod0`]); every accepted edit updates the
//! one affected cell per level ([`LodPyramid::edit_lod0`]), stopping as
//! soon as a level does not change. Nothing here streams or renders — that
//! is 2.6 — and LOD sampled from worldgen is 2.7.
//!
//! Eviction never touches the pyramid: LOD cubes outlive their LOD0
//! children, so player builds stay visible from far away. The pyramid grows
//! with the explored area until 2.6 (clipmap rings) and 2.7 (stored LOD)
//! give it bounds.

use std::collections::HashMap;
use std::mem::size_of;

use jbm_core::block::BlockStateId;
use jbm_core::coords::{BlockPos, CubePos};
use jbm_core::cube::Cube;
use jbm_core::lod::{CellBox, downsample_octant, downsample_region, parent};

/// One cube of a LOD level plus which of its octants hold real data.
///
/// `known` is an octant bitmask: bit *o* is set once that octant's child
/// has been downsampled into this cube at least once. Octants never
/// written read as [`BlockStateId::AIR`] (the cube starts uniform air), so
/// **`known` says which octants hold real data, and nothing else may be
/// assumed about the unknown ones** — an unknown octant's air is
/// placeholder, not measured emptiness. 2.7 fills unknown octants from
/// worldgen.
///
/// A level-*n+1* octant is marked known as soon as its level-*n* child has
/// any known data, so an inserted cube marks the whole chain of ancestors.
/// A partially known child propagates its placeholder air for its own
/// unknown octants — a deliberate simplification for 2.7 to revisit.
#[derive(Debug, Clone)]
pub struct LodCube {
    pub cube: Cube,
    pub known: u8,
}

/// The cubes whose content changed in one operation, in level order.
/// 2.6 will stream exactly these.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LodChanges {
    pub cubes: Vec<(u8, CubePos)>,
}

impl LodChanges {
    pub fn is_empty(&self) -> bool {
        self.cubes.is_empty()
    }
}

/// The pyramid itself: levels `1..=max_levels`, each a map of cubes. A
/// pyramid with `max_levels == 0` is disabled — it allocates nothing and
/// every operation is a no-op.
pub struct LodPyramid {
    /// `levels[l - 1]` holds level `l`; empty while disabled.
    levels: Vec<HashMap<CubePos, LodCube>>,
}

/// Downsampling state carried up the levels during one operation: the
/// cube whose parent is next, and the box of its cells that changed
/// (`None` once a level reported no change — only known-marking
/// continues).
struct Cursor {
    child: CubePos,
    region: Option<CellBox>,
}

impl LodPyramid {
    pub fn new(max_levels: u8) -> Self {
        Self {
            levels: vec![HashMap::new(); max_levels as usize],
        }
    }

    /// The configured top level; 0 when the pyramid is disabled.
    pub fn max_levels(&self) -> u8 {
        self.levels.len() as u8
    }

    fn enabled(&self) -> bool {
        !self.levels.is_empty()
    }

    /// A LOD0 cube entered the world (generated or loaded): downsample it
    /// into its level-1 parent — created as uniform air if absent — set the
    /// known bit, then propagate the changed region up the levels until a
    /// level reports no change or the top is reached. Re-inserting an
    /// unchanged cube reports no changes.
    pub fn insert_lod0(&mut self, pos: CubePos, cube: &Cube) -> LodChanges {
        if !self.enabled() {
            return LodChanges::default();
        }
        let mut changes = LodChanges::default();
        let mut cursor = Cursor {
            child: pos,
            region: None, // level 1 always downsamples the whole child
        };
        for level in 1..=self.max_levels() {
            let (pos, octant) = parent(cursor.child);
            let (below, above) = self.levels.split_at_mut(level as usize - 1);
            let lod_cube = above[0].entry(pos).or_insert_with(|| LodCube {
                cube: Cube::filled(BlockStateId::AIR),
                known: 0,
            });
            let changed = if level == 1 {
                downsample_octant(cube, &mut lod_cube.cube, octant)
            } else {
                match cursor.region {
                    // The child was created by the previous level's pass.
                    Some(region) => {
                        let child = &below[level as usize - 2][&cursor.child];
                        downsample_region(&child.cube, &mut lod_cube.cube, octant, region)
                    }
                    None => None,
                }
            };
            if changed.is_some() {
                changes.cubes.push((level, pos));
            }
            lod_cube.known |= 1 << octant;
            cursor = Cursor {
                child: pos,
                region: changed,
            };
        }
        changes
    }

    /// One accepted edit of the LOD0 block at `pos` (in the post-edit
    /// `cube`): recompute the single affected cell at level 1 and continue
    /// up while levels change, each level costing one cell. Edits in an
    /// area with no pyramid cubes report no changes.
    pub fn edit_lod0(&mut self, pos: BlockPos, cube: &Cube) -> LodChanges {
        if !self.enabled() {
            return LodChanges::default();
        }
        let mut changes = LodChanges::default();
        let (child_pos, local) = pos.split();
        let mut cursor = Cursor {
            child: child_pos,
            region: Some(CellBox::single(local)),
        };
        for level in 1..=self.max_levels() {
            let (parent_pos, octant) = parent(cursor.child);
            let (below, above) = self.levels.split_at_mut(level as usize - 1);
            let Some(lod_cube) = above[0].get_mut(&parent_pos) else {
                break; // no chain here: the edit has nothing to update
            };
            let changed = if level == 1 {
                downsample_region(
                    cube,
                    &mut lod_cube.cube,
                    octant,
                    cursor.region.expect("level 1 starts with the edited cell"),
                )
            } else {
                let region = cursor
                    .region
                    .expect("an unchanged level breaks the loop below");
                match below[level as usize - 2].get(&cursor.child) {
                    Some(child) => {
                        downsample_region(&child.cube, &mut lod_cube.cube, octant, region)
                    }
                    None => break,
                }
            };
            if changed.is_some() {
                changes.cubes.push((level, parent_pos));
            }
            match changed {
                None => break,
                Some(region) => {
                    cursor = Cursor {
                        child: parent_pos,
                        region: Some(region),
                    }
                }
            }
        }
        changes
    }

    /// Rebuilds every level from scratch, bottom up, from the given LOD0
    /// cubes. The reference the incremental paths are tested against:
    /// inserting and editing must match this exactly, cell for cell and
    /// `known` for `known`.
    pub fn rebuild<'a>(max_levels: u8, lod0: impl Iterator<Item = (CubePos, &'a Cube)>) -> Self {
        let mut pyramid = Self::new(max_levels);
        if !pyramid.enabled() {
            return pyramid;
        }
        for (pos, cube) in lod0 {
            let (parent_pos, octant) = parent(pos);
            let lod_cube = pyramid.levels[0]
                .entry(parent_pos)
                .or_insert_with(|| LodCube {
                    cube: Cube::filled(BlockStateId::AIR),
                    known: 0,
                });
            downsample_octant(cube, &mut lod_cube.cube, octant);
            lod_cube.known |= 1 << octant;
        }
        // Each level above is built from the level below: every cube of
        // level l - 1 (they all carry known data) is downsampled into its
        // parent and marks its octant known.
        for level in 2..=max_levels {
            let children: Vec<CubePos> =
                pyramid.levels[level as usize - 2].keys().copied().collect();
            for child in children {
                let (parent_pos, octant) = parent(child);
                let (below, above) = pyramid.levels.split_at_mut(level as usize - 1);
                let lod_cube = above[0].entry(parent_pos).or_insert_with(|| LodCube {
                    cube: Cube::filled(BlockStateId::AIR),
                    known: 0,
                });
                let child_cube = &below[level as usize - 2][&child].cube;
                downsample_octant(child_cube, &mut lod_cube.cube, octant);
                lod_cube.known |= 1 << octant;
            }
        }
        pyramid
    }

    /// The level-`level` cube at `pos`, if the pyramid holds one. Levels
    /// are 1-based; 0 and anything above [`LodPyramid::max_levels`] give
    /// `None`.
    pub fn get(&self, level: u8, pos: CubePos) -> Option<&LodCube> {
        if level == 0 {
            return None;
        }
        self.levels.get(level as usize - 1)?.get(&pos)
    }

    /// Every cube of `level`, for tests and (later) streaming.
    pub fn iter(&self, level: u8) -> impl Iterator<Item = (&CubePos, &LodCube)> + '_ {
        let index = (level as usize).wrapping_sub(1);
        self.levels
            .get(index)
            .into_iter()
            .flat_map(|map| map.iter())
    }

    /// Cubes held at `level`; 0 outside `1..=max_levels`.
    pub fn len(&self, level: u8) -> usize {
        if level == 0 {
            return 0;
        }
        self.levels.get(level as usize - 1).map_or(0, HashMap::len)
    }

    /// Cubes held across all levels.
    pub fn total_cubes(&self) -> usize {
        self.levels.iter().map(|map| map.len()).sum()
    }

    /// Approximate heap bytes across all levels: every cube's
    /// [`Cube::heap_bytes`] plus its entry, plus map capacity. Allocator
    /// overhead is not counted.
    pub fn heap_bytes(&self) -> usize {
        self.levels
            .iter()
            .map(|map| {
                map.values()
                    .map(|lc| lc.cube.heap_bytes() + size_of::<LodCube>())
                    .sum::<usize>()
                    + map.capacity() * (size_of::<(CubePos, LodCube)>() + 1)
            })
            .sum()
    }
}
