//! Mesher input and the padded cell view. [`mesh`] is the mesher's entry
//! point: it takes one cube plus its 26 same-level neighbors
//! ([`MeshInput`]), flattens them into the padded `(CUBE_SIZE + 2)^3` cell
//! array ([`build_padded`]), and hands that to the greedy sweep in
//! `super::greedy`. Rules are documented in crates/client/README.md.

use std::ops::Range;
use std::sync::Arc;

use jbm_core::block::BlockStateId;
use jbm_core::coords::CUBE_SIZE;
use jbm_core::cube::Cube;
use jbm_core::lod::LodTops;

use crate::mesh::appearance::BlockAppearance;
use crate::mesh::face::PackedFace;

/// Side length of the padded array: the cube plus a one-block shell.
pub const PAD: usize = CUBE_SIZE as usize + 2;
/// Padded index stride of the y axis.
pub(crate) const PAD_Y: usize = PAD * PAD;
/// Padded index stride of the z axis.
pub(crate) const PAD_Z: usize = PAD;
/// Padded index of block (0, 0, 0).
pub(crate) const ORIGIN: usize = PAD_Y + PAD_Z + 1;
/// Cells in the padded array.
pub const PAD_VOLUME: usize = PAD * PAD * PAD;

// Flag bits stored in the high half of each padded cell.
pub(crate) const F_AIR: u32 = 1;
/// Occludes for face culling. Missing neighbor cubes have this set, so no
/// walls face unloaded space.
pub(crate) const F_OPAQUE_CULL: u32 = 2;
/// Occludes for ambient occlusion. Missing neighbor cubes do NOT have
/// this: unloaded space reads as open for AO.
pub(crate) const F_OPAQUE_AO: u32 = 4;
const F_CUTOUT: u32 = 8;
pub(crate) const FLAGS_SHIFT: u32 = 16;
/// The padded cell for "no cube here": culling-opaque, AO-transparent.
const MISSING_CELL: u32 = F_OPAQUE_CULL << FLAGS_SHIFT;

/// Index of the neighbor at offset `(dx, dy, dz)`, each in -1..=1, not all
/// zero, into [`MeshInput::neighbors`]. Order: y outer, then z, then x,
/// with the center (13) removed.
pub fn neighbor_index(dx: i32, dy: i32, dz: i32) -> usize {
    debug_assert!((-1..=1).contains(&dx) && (-1..=1).contains(&dy) && (-1..=1).contains(&dz));
    debug_assert!((dx, dy, dz) != (0, 0, 0));
    let raw = ((dy + 1) * 3 + (dz + 1)) * 3 + (dx + 1);
    if raw > 13 {
        (raw - 1) as usize
    } else {
        raw as usize
    }
}

/// Everything the mesher needs for one cube: its storage and its 26
/// neighbors. Built by [`crate::world::ClientWorld`]; shared `Arc`s so
/// worker threads never copy cube data.
pub struct MeshInput {
    pub center: Arc<Cube>,
    pub neighbors: [Option<Arc<Cube>>; 26],
    /// The LOD level being meshed: 0 for streamed world cubes, >= 1 for
    /// ring cubes (cells are 2^level blocks per side).
    pub level: u8,
    /// The column tops (task 2.14) of the center and its four side
    /// neighbors at level >= 1; empty at level 0.
    pub tops: MeshTops,
}

/// The column tops a level >= 1 mesh reads (task 2.14): the center cube's
/// and those of its four same-level side neighbors, whose border column
/// heights the side faces at the cube border need. `None` means every
/// column top is a full cell (and is all a level-0 input ever carries).
#[derive(Clone, Default)]
pub struct MeshTops {
    pub center: Option<Arc<LodTops>>,
    /// The side neighbors' tops in [`SIDE_OFFSETS`] order.
    pub sides: [Option<Arc<LodTops>>; 4],
}

/// The `(dx, dz)` offsets of the four side neighbors, the order of
/// [`MeshTops::sides`]: +X, -X, +Z, -Z.
pub const SIDE_OFFSETS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

impl MeshInput {
    /// A mesh input with no neighbors loaded (everything around is
    /// "missing": culling-opaque, AO-open) at level 0.
    pub fn isolated(center: Arc<Cube>) -> MeshInput {
        MeshInput {
            center,
            neighbors: std::array::from_fn(|_| None),
            level: 0,
            tops: MeshTops::default(),
        }
    }

    /// [`MeshInput::isolated`] at `level`.
    pub fn isolated_at_level(center: Arc<Cube>, level: u8) -> MeshInput {
        MeshInput {
            level,
            ..MeshInput::isolated(center)
        }
    }

    pub fn neighbor(&self, dx: i32, dy: i32, dz: i32) -> Option<&Arc<Cube>> {
        self.neighbors[neighbor_index(dx, dy, dz)].as_ref()
    }
}

/// The mesher output: faces grouped by direction so the renderer can skip
/// back-facing directions. `ranges[d]` indexes `faces`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshOutput {
    pub faces: Vec<PackedFace>,
    pub ranges: [Range<u32>; 6],
    /// Faces emitted under the skirt rule (level >= 1 side directions):
    /// the skirt overhead. They carry the skirt bit in their merge key, so
    /// a counted face never merged with a non-skirt one.
    pub skirts: u32,
    /// Faces emitted under the lid rule (level >= 1, +Y at an octant plane
    /// or the cube's top row). Like skirts they carry their own merge-key
    /// bit, so a counted face never merged with an ordinary one.
    pub lids: u32,
}

impl MeshOutput {
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }
}

/// Builds the padded 34^3 cell array from the center cube and its
/// neighbors. Each cell packs `(flags << 16) | kind`, where kind is the
/// state id + 1 (0 means "missing cube"). Exposed for the
/// `mesh/padded_build` bench.
pub fn build_padded(input: &MeshInput, appearance: &BlockAppearance) -> Vec<u32> {
    // state id -> cell encoding, built once per mesh. State ids are dense.
    let mut table = vec![MISSING_CELL; appearance.len()];
    for (state, cell) in table.iter_mut().enumerate() {
        let app = appearance.get(BlockStateId(state as u32));
        let mut flags = 0;
        if app.air {
            flags |= F_AIR;
        }
        if app.opaque {
            flags |= F_OPAQUE_CULL | F_OPAQUE_AO;
        }
        if app.cutout {
            flags |= F_CUTOUT;
        }
        *cell = (flags << FLAGS_SHIFT) | (state as u32 + 1);
    }
    let cell_of =
        |state: BlockStateId| table.get(state.0 as usize).copied().unwrap_or(MISSING_CELL);

    let mut cells = vec![MISSING_CELL; PAD_VOLUME];

    // Center: one 32-cell row per (y, z). Uniform centers fill; others
    // stream Cube::iter(), which is ~3x faster than per-block get.
    if let Some(state) = input.center.uniform() {
        let cell = cell_of(state);
        for y in 0..CUBE_SIZE as usize {
            for z in 0..CUBE_SIZE as usize {
                let base = ORIGIN + PAD_Y * y + PAD_Z * z;
                cells[base..base + CUBE_SIZE as usize].fill(cell);
            }
        }
    } else {
        let mut states = input.center.iter();
        for y in 0..CUBE_SIZE as usize {
            for z in 0..CUBE_SIZE as usize {
                let base = ORIGIN + PAD_Y * y + PAD_Z * z;
                for x in 0..CUBE_SIZE as usize {
                    let state = states.next().expect("cube covers CUBE_VOLUME cells");
                    cells[base + x] = cell_of(state);
                }
            }
        }
    }

    // Neighbor shells. A cube at offset o contributes the padded cells on
    // the matching face; its local coordinate is p - 32 * o per axis, so
    // the same formula covers face, edge and corner neighbors. Cells no
    // present neighbor covers stay MISSING.
    for dy in -1..=1 {
        for dz in -1..=1 {
            for dx in -1..=1 {
                if (dx, dy, dz) == (0, 0, 0) {
                    continue;
                }
                let Some(cube) = input.neighbor(dx, dy, dz) else {
                    continue;
                };
                let uniform_cell = cube.uniform().map(cell_of);
                let (ys, ye) = axis_span(dy);
                let (zs, ze) = axis_span(dz);
                let (xs, _xe) = axis_span(dx);
                // Padded x is fixed at -1 or CUBE_SIZE for side neighbors;
                // the matching local x in the neighbor cube is
                // CUBE_SIZE - 1 or 0.
                let px = (xs + 1) as usize; // padded column within the row
                let lx = (xs - CUBE_SIZE * dx) as usize; // local x in the neighbor
                for py in ys..=ye {
                    let y_base = PAD_Y * ((py + 1) as usize);
                    let ly = py - CUBE_SIZE * dy;
                    for pz in zs..=ze {
                        let z_base = y_base + PAD_Z * ((pz + 1) as usize);
                        let lz = pz - CUBE_SIZE * dz;
                        if let Some(cell) = uniform_cell {
                            if xs == 0 {
                                cells[z_base + 1..z_base + 1 + CUBE_SIZE as usize].fill(cell);
                            } else {
                                cells[z_base + px] = cell;
                            }
                            continue;
                        }
                        let row = ((ly * CUBE_SIZE + lz) * CUBE_SIZE) as usize;
                        if xs == 0 {
                            for x in 0..CUBE_SIZE as usize {
                                let state = cube.get_index(row + x);
                                cells[z_base + 1 + x] = cell_of(state);
                            }
                        } else {
                            let state = cube.get_index(row + lx);
                            cells[z_base + px] = cell_of(state);
                        }
                    }
                }
            }
        }
    }
    cells
}

/// Whether `tops` hold a column top below a full cell of `level`.
pub(crate) fn has_partial_top(tops: Option<&LodTops>, level: u8) -> bool {
    let full = 1u8 << level;
    tops.is_some_and(|tops| !matches!(tops.uniform(), Some(h) if h == 0 || h == full))
}

/// The fill heights (task 2.14) of the cells a level >= 1 sweep reads, in
/// blocks: every filled (or missing) cell is a full `2^level` except the
/// **partial column tops** — the topmost filled cell of a column of the
/// center or of a side neighbor whose top is lower than a full cell. At
/// most one cell per padded column, so this stores that cell's padded y
/// and height per column. Built by [`build_padded_heights`].
pub struct PaddedHeights {
    full: u8,
    /// Per padded column `x + PAD · z`: the padded y of its partial top,
    /// `u8::MAX` for none.
    top_y: [u8; PAD * PAD],
    /// The partial top's fill height, `1..full`.
    top_h: [u8; PAD * PAD],
}

impl PaddedHeights {
    /// A full cell's height, `2^level` blocks.
    pub fn full(&self) -> u8 {
        self.full
    }

    /// The fill height of the filled or missing cell at padded `index`.
    /// (Air cells have height 0; callers never ask for them.)
    #[inline]
    pub fn at(&self, index: usize) -> u8 {
        let column = index % PAD_Y;
        if usize::from(self.top_y[column]) == index / PAD_Y {
            self.top_h[column]
        } else {
            self.full
        }
    }

    /// The partial tops as `(padded column, padded y)` pairs.
    pub(crate) fn partial_tops(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.top_y
            .iter()
            .enumerate()
            .filter(|&(_, &y)| y != u8::MAX)
            .map(|(column, &y)| (column, usize::from(y)))
    }
}

/// The padded kind (state id + 1) of [`BlockStateId::AIR`].
const AIR_KIND: u32 = BlockStateId::AIR.0 + 1;

/// Builds the [`PaddedHeights`] of a level >= 1 input from its tops and
/// its padded cells (`pad` from [`build_padded`]). `None` when every cell
/// is full height — level 0, or no partial column top in the center or a
/// side neighbor — and the sweep then meshes exactly as before task 2.14.
/// A column's top cell is its topmost non-air cell within the cube's 32
/// rows (the LOD occupancy notion: any non-air state is filled).
pub fn build_padded_heights(input: &MeshInput, pad: &[u32]) -> Option<PaddedHeights> {
    let level = input.level;
    if level == 0 {
        return None;
    }
    let sources: [(Option<&LodTops>, (i32, i32)); 5] = [
        (input.tops.center.as_deref(), (0, 0)),
        (input.tops.sides[0].as_deref(), SIDE_OFFSETS[0]),
        (input.tops.sides[1].as_deref(), SIDE_OFFSETS[1]),
        (input.tops.sides[2].as_deref(), SIDE_OFFSETS[2]),
        (input.tops.sides[3].as_deref(), SIDE_OFFSETS[3]),
    ];
    if !sources
        .iter()
        .any(|&(tops, _)| has_partial_top(tops, level))
    {
        return None;
    }
    let full = 1u8 << level;
    let mut heights = PaddedHeights {
        full,
        top_y: [u8::MAX; PAD * PAD],
        top_h: [0; PAD * PAD],
    };
    let size = CUBE_SIZE;
    for (tops, (dx, dz)) in sources {
        let Some(tops) = tops.filter(|&tops| has_partial_top(Some(tops), level)) else {
            continue;
        };
        // The cube's columns as (local x, local z) -> padded column: the
        // center covers all of them, a side neighbor only its border row
        // facing the center.
        let (xs, zs) = (shell_span(dx), shell_span(dz));
        for z in zs.clone() {
            for x in xs.clone() {
                let h = tops.get(x as u8, z as u8);
                if h == 0 || h >= full {
                    continue;
                }
                let column = (x + size * dx + 1) as usize + PAD * (z + size * dz + 1) as usize;
                // The upper half's top hides the lower one's.
                let start = if tops.half_top(1, x as u8, z as u8) != 0 {
                    size - 1
                } else {
                    size / 2 - 1
                };
                let top = (0..=start).rev().find(|&y| {
                    let kind = pad[column + PAD_Y * (y + 1) as usize] & 0xFFFF;
                    kind != 0 && kind != AIR_KIND
                });
                if let Some(y) = top {
                    heights.top_y[column] = (y + 1) as u8;
                    heights.top_h[column] = h;
                }
            }
        }
    }
    Some(heights)
}

/// The local coordinates along one horizontal axis of the cube at offset
/// `offset` whose columns the padded array holds: all of them for the
/// center (0), the border row facing it for a side neighbor.
fn shell_span(offset: i32) -> Range<i32> {
    match offset {
        1 => 0..1,
        -1 => CUBE_SIZE - 1..CUBE_SIZE,
        _ => 0..CUBE_SIZE,
    }
}

/// Padded-coordinate span for one axis of a neighbor offset: a fixed shell
/// coordinate (−1 or `CUBE_SIZE`) for nonzero offsets, the full
/// `0..=CUBE_SIZE - 1` for zero.
fn axis_span(offset: i32) -> (i32, i32) {
    match offset {
        -1 => (-1, -1),
        1 => (CUBE_SIZE, CUBE_SIZE),
        _ => (0, CUBE_SIZE - 1),
    }
}
