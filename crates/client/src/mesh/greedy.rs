//! The greedy mesher: pure logic, no GPU. Builds a padded
//! `(CUBE_SIZE + 2)^3` block view from a cube and its neighbors, then merges
//! coplanar faces into maximal rectangles per direction. Rules are documented
//! in crates/client/README.md.

use std::ops::Range;
use std::sync::Arc;

use jbm_core::block::BlockStateId;
use jbm_core::coords::CUBE_SIZE;
use jbm_core::cube::Cube;

use crate::mesh::appearance::BlockAppearance;
use crate::mesh::face::{DIRECTIONS, FaceFields, PackedFace, pack};

/// Side length of the padded array: the cube plus a one-block shell.
pub const PAD: usize = CUBE_SIZE as usize + 2;
/// Padded index stride of the y axis.
const PAD_Y: usize = PAD * PAD;
/// Padded index stride of the z axis.
const PAD_Z: usize = PAD;
/// Padded index of block (0, 0, 0).
const ORIGIN: usize = PAD_Y + PAD_Z + 1;
/// Cells in the padded array.
pub const PAD_VOLUME: usize = PAD * PAD * PAD;

// Flag bits stored in the high half of each padded cell.
const F_AIR: u32 = 1;
/// Occludes for face culling. Missing neighbor cubes have this set, so no
/// walls face unloaded space.
const F_OPAQUE_CULL: u32 = 2;
/// Occludes for ambient occlusion. Missing neighbor cubes do NOT have
/// this: unloaded space reads as open for AO.
const F_OPAQUE_AO: u32 = 4;
const F_CUTOUT: u32 = 8;
const FLAGS_SHIFT: u32 = 16;
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
}

impl MeshInput {
    /// A mesh input with no neighbors loaded (everything around is
    /// "missing": culling-opaque, AO-open).
    pub fn isolated(center: Arc<Cube>) -> MeshInput {
        MeshInput {
            center,
            neighbors: std::array::from_fn(|_| None),
        }
    }

    pub fn neighbor(&self, dx: i32, dy: i32, dz: i32) -> Option<&Arc<Cube>> {
        self.neighbors[neighbor_index(dx, dy, dz)].as_ref()
    }
}

/// The mesher output: faces grouped by direction so the renderer can skip
/// back-facing directions (task 2.2). `ranges[d]` indexes `faces`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshOutput {
    pub faces: Vec<PackedFace>,
    pub ranges: [Range<u32>; 6],
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
    mesh_padded(&padded, appearance)
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

/// Padded-index stride of an axis vector like (0, 0, -1).
#[inline]
fn stride(axis: [i32; 3]) -> isize {
    axis[0] as isize + PAD_Y as isize * axis[1] as isize + PAD_Z as isize * axis[2] as isize
}

/// Sweeps the six directions and greedily merges coplanar faces with
/// identical (layer, AO, light) into maximal rectangles.
fn mesh_padded(pad: &[u32], appearance: &BlockAppearance) -> MeshOutput {
    let mut output = MeshOutput {
        faces: Vec::new(),
        ranges: std::array::from_fn(|_| 0u32..0),
    };
    let size = CUBE_SIZE as usize;
    let mut mask = [0u32; MASK_LEN];

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
        let last = CUBE_SIZE as isize - 1;
        let u_shift = if u < 0 { -last * u } else { 0 };
        let v_shift = if v < 0 { -last * v } else { 0 };
        let corner = ORIGIN as isize + u_shift + v_shift;

        // layer-of-kind lookup for this direction (kind = state id + 1).
        let mut layer_of_kind = vec![0u16; appearance.len() + 1];
        for state in 0..appearance.len() {
            layer_of_kind[state + 1] = appearance.get(BlockStateId(state as u32)).layers[dir_index];
        }

        let start = output.faces.len() as u32;
        for k in 0..CUBE_SIZE as isize {
            let a_row = corner + k * n_axis;
            let b_row = a_row + n;

            // Build the mask of face keys for this slice.
            for vi in 0..CUBE_SIZE as isize {
                for ui in 0..CUBE_SIZE as isize {
                    let a = pad[(a_row + ui * u + vi * v) as usize];
                    let b = pad[(b_row + ui * u + vi * v) as usize];
                    let culled = a & (F_AIR << FLAGS_SHIFT) != 0
                        || b & (F_OPAQUE_CULL << FLAGS_SHIFT) != 0
                        || (a == b && a & (F_CUTOUT << FLAGS_SHIFT) != 0);
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

            // Greedy merge: extend along u first, then along v.
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

                    // Owner block: the mask corner (u0, v0) of this slice.
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
    output
}

/// Light is always 15 until lighting exists (task 2.x).
const LIGHT: u32 = 15;

/// Cells in one u/v slice mask (`CUBE_SIZE²`).
const MASK_LEN: usize = (CUBE_SIZE as usize) * (CUBE_SIZE as usize);

/// AO of one unit face whose front block sits at padded index `front`,
/// with face strides `u` and `v`. Returns the four corner values packed
/// 2 bits each, c0..c3 in (u, v) corner order: 0 dark ..= 3 open.
#[inline]
fn corner_ao(pad: &[u32], front: isize, u: isize, v: isize) -> u32 {
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
    use crate::mesh::face::{Direction, unpack};
    use jbm_core::block::builtin;
    use jbm_core::coords::CUBE_VOLUME;

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

    /// A deliberately independent reference mesher: one face per block,
    /// reading cubes directly instead of any padded array.
    fn naive_unit_faces(
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
}
