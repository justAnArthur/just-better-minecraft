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

use crate::mesh::appearance::BlockAppearance;
use crate::mesh::face::PackedFace;
use crate::mesh::greedy::mesh_padded;

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
}

impl MeshInput {
    /// A mesh input with no neighbors loaded (everything around is
    /// "missing": culling-opaque, AO-open) at level 0.
    pub fn isolated(center: Arc<Cube>) -> MeshInput {
        MeshInput {
            center,
            neighbors: std::array::from_fn(|_| None),
            level: 0,
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

/// Meshes one cube. See the README for the face rule and AO rule.
pub fn mesh(input: &MeshInput, appearance: &BlockAppearance) -> MeshOutput {
    if let Some(state) = input.center.uniform() {
        let app = appearance.get(state);
        if app.air {
            // Uniform air owns no faces, whatever the neighbors are.
            return MeshOutput::default();
        }
        if app.opaque && enclosed_by_opaque(input, appearance) {
            // Fully surrounded by uniform opaque face neighbors: nothing
            // of this cube is visible.
            return MeshOutput::default();
        }
    }
    let padded = build_padded(input, appearance);
    mesh_padded(&padded, appearance, input.level)
}

/// Whether all six face neighbors are present, uniform and opaque.
fn enclosed_by_opaque(input: &MeshInput, appearance: &BlockAppearance) -> bool {
    const FACE_OFFSETS: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    FACE_OFFSETS.iter().all(|&(dx, dy, dz)| {
        input
            .neighbor(dx, dy, dz)
            .and_then(|cube| cube.uniform())
            .is_some_and(|state| appearance.get(state).opaque)
    })
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
