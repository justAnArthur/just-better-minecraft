//! The LOD downsampling rule and the cube-level downsampling helpers
//! (task 2.4). Pure logic: no I/O, no threads, no registry data.
//!
//! A level-*n+1* cell covers a 2×2×2 group of level-*n* cells, its
//! "children", indexed `c = dx + 2·dz + 4·dy` — the same axis order as
//! [`LocalPos::index`](crate::coords::LocalPos::index). The rule for one
//! cell is [`downsample_cell`]; the cube-level helpers move between a child
//! cube and one octant of its parent.
//!
//! # The rule
//!
//! **Occupancy.** A child is *filled* when it is not air (water and every
//! other non-air state count; block properties are never consulted). The
//! cell is filled when **at least 2** of the 8 children are filled;
//! otherwise it is air. A one-block pillar (2 of 8) and a one-block wall
//! (4 of 8) therefore survive at every level, while a single floating
//! block or torch-sized feature vanishes at LOD1 — the expected behavior.
//! The rule leans solid: surfaces can rise by up to 2ⁿ − 1 blocks and
//! tunnels narrower than 2ⁿ can close. Downward skirts (2.6) hide the
//! first; the second is invisible from LOD distance.
//!
//! **Material** for a filled cell is "what you see from above": the
//! candidates are the topmost filled child of each of the four `(dx, dz)`
//! columns of the group (a column with no filled child gives none); pick
//! the most common candidate state. Ties go to the state appearing at the
//! highest `dy`, then to the smallest [`BlockStateId`].
//!
//! Worked example — grass stays on top: children with dirt in the lower
//! `dy` layer and grass in the upper layer (a one-step slope, or flat
//! ground aligned either way with the group) give one grass candidate per
//! column, so the cell is grass. Even when one column's grass sits above
//! the group (dirt at both its `dy`, grass only in the other three
//! columns), grass wins 3 candidates to dirt's 1.
//!
//! Worked example — a wall keeps its material: a vertical planks wall one
//! block thick standing on stone ground fills both `dy` of its `dx`
//! column, so both its `(dx, dz)` columns nominate planks (the topmost
//! filled child is planks), while the neighbouring `dx` columns nominate
//! stone. Planks wins the candidate count 2 to 2 only if the ground
//! columns also see stone — whenever the wall's own columns are the
//! majority or win the height tie-break, planks is the answer; in the
//! spec's arrangement (air above the ground beside the wall) the ground
//! columns nominate stone but the wall columns' planks candidates sit at
//! the higher `dy`, so planks wins the tie.
//!
//! **Locality.** The rule looks only at the eight children, never at
//! cells outside the group, so one edit changes at most one cell per
//! level — that locality is what makes edit propagation exact. Do not
//! "improve" the rule by consulting neighbours; a later task may refine
//! it with registry data, but it must stay local.

use crate::block::BlockStateId;
use crate::coords::{CUBE_SIZE, CubePos, LocalPos};
use crate::cube::Cube;

/// Children per parent cell.
pub const CHILDREN: usize = 8;

/// The downsampling rule for one cell: a pure function of the eight child
/// states, indexed `c = dx + 2·dz + 4·dy`. See the module docs for the rule
/// and its rationale.
pub fn downsample_cell(children: [BlockStateId; CHILDREN]) -> BlockStateId {
    let mut filled = 0;
    for &state in &children {
        if state != BlockStateId::AIR {
            filled += 1;
        }
    }
    if filled < 2 {
        return BlockStateId::AIR;
    }

    // The topmost filled child of each (dx, dz) column: at most four
    // candidates, as (state, dy).
    let mut candidates = [(BlockStateId::AIR, 0u8); 4];
    let mut count = 0;
    for dz in 0..2usize {
        for dx in 0..2usize {
            for dy in (0..2usize).rev() {
                let state = children[dx + 2 * dz + 4 * dy];
                if state != BlockStateId::AIR {
                    candidates[count] = (state, dy as u8);
                    count += 1;
                    break;
                }
            }
        }
    }
    debug_assert!(count >= 1, "filled >= 2 gives at least one candidate");

    // Most common candidate; ties to the highest dy the state appears at,
    // then to the smallest state id. A fixed scan over at most four
    // entries — no hashmap, so no iteration-order dependence.
    let rank = |(state, _): (BlockStateId, u8)| {
        let mut occurrences = 0u32;
        let mut top = 0u8;
        for &(candidate, dy) in &candidates[..count] {
            if candidate == state {
                occurrences += 1;
                top = top.max(dy);
            }
        }
        (occurrences, top, state)
    };
    let mut best = rank(candidates[0]);
    for &candidate in &candidates[1..count] {
        let ranked = rank(candidate);
        let better = ranked.0 > best.0
            || (ranked.0 == best.0 && ranked.1 > best.1)
            || (ranked.0 == best.0 && ranked.1 == best.1 && ranked.2 < best.2);
        if better {
            best = ranked;
        }
    }
    best.2
}

/// The parent cube of `pos` and which octant of it `pos` fills. Octants are
/// indexed `dx + 2·dz + 4·dy` over the child position `rem_euclid(2)` per
/// axis; the parent position is `div_euclid(2)` per axis, so cubes at
/// negative coordinates nest exactly like positive ones.
pub fn parent(pos: CubePos) -> (CubePos, u8) {
    let [dx, dy, dz] = octant_parts(pos);
    (
        CubePos {
            x: pos.x.div_euclid(2),
            y: pos.y.div_euclid(2),
            z: pos.z.div_euclid(2),
        },
        dx + 2 * dz + 4 * dy,
    )
}

/// `rem_euclid(2)` of each axis of `pos`, as `[dx, dy, dz]`.
fn octant_parts(pos: CubePos) -> [u8; 3] {
    [
        pos.x.rem_euclid(2) as u8,
        pos.y.rem_euclid(2) as u8,
        pos.z.rem_euclid(2) as u8,
    ]
}

/// The `[dx, dy, dz]` offset of octant `o` within its parent, the inverse
/// of the octant half of [`parent`].
pub fn octant_offset(octant: u8) -> [u8; 3] {
    [octant & 1, octant >> 2, (octant >> 1) & 1]
}

/// A box of cells: inclusive `min`, exclusive `max`, each axis in
/// `0..CUBE_SIZE`. Small and `Copy` so propagation can hand it up levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellBox {
    pub min: [u8; 3],
    pub max: [u8; 3],
}

impl CellBox {
    /// The box holding exactly one cell.
    pub fn single(pos: LocalPos) -> Self {
        Self {
            min: [pos.x, pos.y, pos.z],
            max: [pos.x + 1, pos.y + 1, pos.z + 1],
        }
    }

    /// The smallest box holding every cell of either input.
    pub fn union(self, other: Self) -> Self {
        let mut min = self.min;
        let mut max = self.max;
        for a in 0..3 {
            min[a] = min[a].min(other.min[a]);
            max[a] = max[a].max(other.max[a]);
        }
        Self { min, max }
    }
}

/// Recomputes every parent cell whose 2×2×2 group of child cells overlaps
/// `region` (a box of **child** cells), writing into `octant` of `parent`.
/// Returns the box of **parent** cells that actually changed, or `None`
/// when none did — that box is the next level's input, so propagation stays
/// proportional to what changed. [`Cube::compact`] runs after a write, so a
/// parent that became uniform (or shed palette entries) goes back to its
/// cheapest storage; that cost is part of this call and covered by the
/// `lod/edit_cell` bench.
pub fn downsample_region(
    child: &Cube,
    parent: &mut Cube,
    octant: u8,
    region: CellBox,
) -> Option<CellBox> {
    let half = CUBE_SIZE as usize / 2;
    let [ox, oy, oz] = octant_offset(octant).map(|v| v as usize * half);

    // Parent cells (octant-local) whose group {2k, 2k+1} overlaps
    // [min, max): k from floor((min-1)/2) to ceil(max/2) - 1.
    let first = |min: u8| (min as usize).saturating_sub(1) >> 1;
    let last = |max: u8| (max as usize + 1) >> 1; // exclusive

    let mut changed: Option<CellBox> = None;
    let mut children = [BlockStateId::AIR; CHILDREN];
    for ky in first(region.min[1])..last(region.max[1]) {
        for kz in first(region.min[2])..last(region.max[2]) {
            for kx in first(region.min[0])..last(region.max[0]) {
                for dy in 0..2usize {
                    for dz in 0..2usize {
                        for dx in 0..2usize {
                            children[dx + 2 * dz + 4 * dy] = child.get(LocalPos::new_unchecked(
                                (2 * kx + dx) as u8,
                                (2 * ky + dy) as u8,
                                (2 * kz + dz) as u8,
                            ));
                        }
                    }
                }
                // The parent cell's octant-local coordinates are the group
                // indices k themselves, not the loop ordinal.
                let pos =
                    LocalPos::new_unchecked((ox + kx) as u8, (oy + ky) as u8, (oz + kz) as u8);
                let down = downsample_cell(children);
                if parent.get(pos) != down {
                    parent.set(pos, down);
                    let cell = CellBox::single(pos);
                    changed = Some(match changed {
                        None => cell,
                        Some(boxed) => boxed.union(cell),
                    });
                }
            }
        }
    }
    if changed.is_some() {
        parent.compact();
    }
    changed
}

/// [`downsample_region`] over the whole child cube, with two fast paths: a
/// uniform child fills its octant with its own state (downsampling eight
/// equal states yields that state — asserted by the tests), and a parent
/// that is already uniform and equal to that fill is left untouched, so it
/// is never unpacked.
pub fn downsample_octant(child: &Cube, parent: &mut Cube, octant: u8) -> Option<CellBox> {
    if let Some(state) = child.uniform() {
        if parent.uniform() == Some(state) {
            return None;
        }
        let half = CUBE_SIZE as usize / 2;
        let [ox, oy, oz] = octant_offset(octant).map(|v| v as usize * half);
        let mut changed: Option<CellBox> = None;
        for y in oy..oy + half {
            for z in oz..oz + half {
                for x in ox..ox + half {
                    let pos = LocalPos::new_unchecked(x as u8, y as u8, z as u8);
                    if parent.get(pos) != state {
                        parent.set(pos, state);
                        let cell = CellBox::single(pos);
                        changed = Some(match changed {
                            None => cell,
                            Some(boxed) => boxed.union(cell),
                        });
                    }
                }
            }
        }
        if changed.is_some() {
            parent.compact();
        }
        changed
    } else {
        downsample_region(
            child,
            parent,
            octant,
            CellBox {
                min: [0, 0, 0],
                max: [CUBE_SIZE as u8; 3],
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: BlockStateId = BlockStateId::AIR;
    const STONE: BlockStateId = BlockStateId(1);
    const DIRT: BlockStateId = BlockStateId(2);
    const GRASS: BlockStateId = BlockStateId(3);
    const PLANKS: BlockStateId = BlockStateId(4);

    /// `children[dx + 2·dz + 4·dy]` from a per-cell closure taking
    /// `(dx, dy, dz)`.
    fn gather(mut f: impl FnMut(usize, usize, usize) -> BlockStateId) -> [BlockStateId; 8] {
        let mut children = [AIR; 8];
        for dy in 0..2usize {
            for dz in 0..2usize {
                for dx in 0..2usize {
                    children[dx + 2 * dz + 4 * dy] = f(dx, dy, dz);
                }
            }
        }
        children
    }

    /// Small local xorshift, so tests never need the `rand` crate.
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
    // Occupancy

    #[test]
    fn occupancy_threshold_is_two() {
        assert_eq!(downsample_cell([AIR; 8]), AIR);
        // Every single-filled child: air.
        for filled in 0..8 {
            let mut children = [AIR; 8];
            children[filled] = STONE;
            assert_eq!(downsample_cell(children), AIR, "filled at {filled}");
        }
        // Every pair of filled children: filled. The material is the pair's
        // vote — one candidate each, so the higher dy (then the smaller id)
        // decides.
        for a in 0..8 {
            for b in (a + 1)..8 {
                let mut children = [AIR; 8];
                children[a] = STONE;
                children[b] = DIRT;
                let (dy_a, dy_b) = (a >> 2, b >> 2);
                let expected = if dy_a > dy_b || (dy_a == dy_b && STONE < DIRT) {
                    STONE
                } else {
                    DIRT
                };
                assert_eq!(downsample_cell(children), expected, "filled at {a} and {b}");
            }
        }
    }

    #[test]
    fn eight_equal_states_downsample_to_that_state() {
        for state in [AIR, STONE, BlockStateId(777)] {
            assert_eq!(downsample_cell([state; 8]), state);
        }
    }

    // ------------------------------------------------------------------
    // Surface material

    #[test]
    fn grass_over_dirt_all_filled_is_grass() {
        // Grass everywhere in the upper dy layer, dirt below.
        let children = gather(|_dx, dy, _dz| if dy == 1 { GRASS } else { DIRT });
        assert_eq!(downsample_cell(children), GRASS);
    }

    #[test]
    fn grass_bottom_layer_with_air_above_is_grass() {
        // A grass shelf one block below the group top: the columns'
        // topmost filled child is grass.
        let children = gather(|_dx, dy, _dz| if dy == 0 { GRASS } else { AIR });
        assert_eq!(downsample_cell(children), GRASS);
    }

    #[test]
    fn one_step_slope_is_grass() {
        // A one-step slope: the dz = 0 columns stand one block higher —
        // grass at dy 1 over dirt — while the dz = 1 columns are grass at
        // dy 0 with air above. Every column's topmost filled child is
        // grass, so dirt never becomes a candidate.
        let children = gather(|_dx, dy, dz| match (dy, dz) {
            (1, 0) => GRASS,
            (0, 0) => DIRT,
            (1, 1) => AIR,
            (0, 1) => GRASS,
            _ => unreachable!("dy and dz are 0 or 1"),
        });
        assert_eq!(downsample_cell(children), GRASS);
    }

    #[test]
    fn three_grass_columns_beat_one_buried_dirt_column() {
        // Three columns grass-topped; the fourth has dirt at both dy (its
        // grass sits above the group). Grass takes the candidate vote 3 to
        // the dirt column's 1.
        let children = gather(|dx, dy, dz| {
            if dx == 1 && dz == 1 {
                DIRT // the buried column, dirt at both dy
            } else if dy == 1 {
                GRASS
            } else {
                DIRT
            }
        });
        assert_eq!(downsample_cell(children), GRASS);
    }

    // ------------------------------------------------------------------
    // Wall material

    #[test]
    fn wall_keeps_its_material_over_ground() {
        // A planks wall one block thick at dx = 0 (both dy, both dz);
        // beside it (dx = 1) stone ground at dy = 0 with air above. The
        // wall's two columns nominate planks at dy 1, the ground's two
        // nominate stone at dy 0: a 2–2 count tie broken by height.
        let children = gather(|dx, dy, _dz| match (dx, dy) {
            (0, _) => PLANKS,
            (1, 0) => STONE,
            (1, 1) => AIR,
            _ => unreachable!("dx is 0 or 1"),
        });
        assert_eq!(downsample_cell(children), PLANKS);
    }

    // ------------------------------------------------------------------
    // Tie-breaks

    #[test]
    fn count_tie_goes_to_the_higher_dy() {
        // Stone tops two columns at dy 0; planks tops two columns at dy 1.
        // The 2–2 count tie goes to the state at the higher dy.
        let children = gather(|dx, dy, _dz| match (dx, dy) {
            (0, 0) => STONE,
            (1, 1) => PLANKS,
            _ if dx > 1 => unreachable!("dx is 0 or 1"),
            _ => AIR,
        });
        assert_eq!(downsample_cell(children), PLANKS);
    }

    #[test]
    fn count_and_height_tie_goes_to_the_smaller_state_id() {
        // Dirt and stone each top one column, both at dy 1; a second filled
        // child keeps the cell non-air. Stone (id 1 < dirt's 2) wins.
        let children = gather(|dx, dy, dz| match (dx, dy, dz) {
            (0, 1, 0) => DIRT,
            (0, 0, 0) => DIRT,
            (1, 1, 1) => STONE,
            _ => AIR,
        });
        assert_eq!(downsample_cell(children), STONE);
    }

    // ------------------------------------------------------------------
    // Reference check

    #[test]
    fn random_tuples_match_a_plain_reference() {
        /// The rule written out plainly, independently of the
        /// implementation above: occupancy, then a counted vote over the
        /// column tops with explicit tie-breaks.
        fn reference(children: [BlockStateId; 8]) -> BlockStateId {
            let filled = children.iter().filter(|&&s| s != AIR).count();
            if filled < 2 {
                return AIR;
            }
            let mut candidates: Vec<(BlockStateId, usize)> = Vec::new();
            for dz in 0..2usize {
                for dx in 0..2usize {
                    for dy in (0..2usize).rev() {
                        if children[dx + 2 * dz + 4 * dy] != AIR {
                            candidates.push((children[dx + 2 * dz + 4 * dy], dy));
                            break;
                        }
                    }
                }
            }
            let mut votes: Vec<(u32, usize, BlockStateId)> = Vec::new();
            for &(state, _) in &candidates {
                if votes.iter().any(|&(_, _, s)| s == state) {
                    continue;
                }
                let mine: Vec<(BlockStateId, usize)> = candidates
                    .iter()
                    .copied()
                    .filter(|&(s, _)| s == state)
                    .collect();
                let count = mine.len() as u32;
                let top = mine.iter().map(|&(_, dy)| dy).max().unwrap();
                votes.push((count, top, state));
            }
            votes.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
            votes[0].2
        }

        let states = [AIR, STONE, DIRT, GRASS];
        let mut rng = XorShift64(0x2f6e_2b1a_9d4c_7e55);
        for _ in 0..10_000 {
            let children = core::array::from_fn(|_| states[(rng.next() % 4) as usize]);
            assert_eq!(downsample_cell(children), reference(children));
        }
    }

    // ------------------------------------------------------------------
    // Octants and regions

    #[test]
    fn parent_and_octant_round_trip() {
        for x in [-3, -2, -1, 0, 1, 2, 3] {
            for y in [-3, -2, -1, 0, 1, 2, 3] {
                for z in [-3, -2, -1, 0, 1, 2, 3] {
                    let pos = CubePos { x, y, z };
                    let (parent_pos, octant) = parent(pos);
                    assert_eq!(parent_pos.x * 2 + octant_offset(octant)[0] as i32, x);
                    assert_eq!(parent_pos.y * 2 + octant_offset(octant)[1] as i32, y);
                    assert_eq!(parent_pos.z * 2 + octant_offset(octant)[2] as i32, z);
                    // div_euclid / rem_euclid, spelled out for the negatives.
                    assert_eq!(parent_pos.x, x.div_euclid(2));
                    assert_eq!(parent_pos.y, y.div_euclid(2));
                    assert_eq!(parent_pos.z, z.div_euclid(2));
                }
            }
        }
    }

    #[test]
    fn cell_box_single_and_union() {
        let a = CellBox::single(LocalPos::new_unchecked(1, 2, 3));
        assert_eq!(a.min, [1, 2, 3]);
        assert_eq!(a.max, [2, 3, 4]);
        let b = CellBox::single(LocalPos::new_unchecked(5, 0, 7));
        let u = a.union(b);
        assert_eq!(u.min, [1, 0, 3]);
        assert_eq!(u.max, [6, 3, 8]);
        assert_eq!(a.union(a), a);
    }

    /// A cube whose octant-downsample is checked cell by cell against a
    /// direct application of `downsample_cell`.
    fn check_octant(child: &Cube, octant: u8) -> (Cube, Option<CellBox>) {
        let mut parent = Cube::filled(AIR);
        let changed = downsample_octant(child, &mut parent, octant);
        let half = (CUBE_SIZE / 2) as usize;
        let [ox, oy, oz] = octant_offset(octant).map(|v| v as usize * half);
        for y in 0..half {
            for z in 0..half {
                for x in 0..half {
                    let mut children = [AIR; 8];
                    for dy in 0..2usize {
                        for dz in 0..2usize {
                            for dx in 0..2usize {
                                children[dx + 2 * dz + 4 * dy] =
                                    child.get(LocalPos::new_unchecked(
                                        (2 * x + dx) as u8,
                                        (2 * y + dy) as u8,
                                        (2 * z + dz) as u8,
                                    ));
                            }
                        }
                    }
                    let expected = downsample_cell(children);
                    let got = parent.get(LocalPos::new_unchecked(
                        (ox + x) as u8,
                        (oy + y) as u8,
                        (oz + z) as u8,
                    ));
                    assert_eq!(got, expected, "octant {octant} cell ({x},{y},{z})");
                }
            }
        }
        (parent, changed)
    }

    #[test]
    fn octant_downsample_matches_the_rule_per_cell() {
        // A mixed cube with thin features, plus every octant.
        let size = CUBE_SIZE as usize;
        let child = Cube::from_fn(|p| {
            let (x, y, z) = (p.x as usize, p.y as usize, p.z as usize);
            // A 1-thick wall, a floating block, and layered ground.
            if x == size / 2 && z < size / 2 {
                PLANKS
            } else if (x, y, z) == (1, size - 1, 1) {
                STONE // the single floating block: vanishes
            } else if y < size / 4 {
                if y == size / 4 - 1 { GRASS } else { DIRT }
            } else {
                AIR
            }
        });
        assert!(child.uniform().is_none());
        for octant in 0..8u8 {
            check_octant(&child, octant);
        }
    }

    #[test]
    fn uniform_child_fills_its_octant_and_equal_parent_is_untouched() {
        let child = Cube::filled(STONE);
        let mut parent = Cube::filled(AIR);
        let changed = downsample_octant(&child, &mut parent, 5);
        assert!(changed.is_some());
        let half = CUBE_SIZE as usize / 2;
        let [ox, oy, oz] = octant_offset(5).map(|v| v as usize * half);
        for y in 0..CUBE_SIZE as usize {
            for z in 0..CUBE_SIZE as usize {
                for x in 0..CUBE_SIZE as usize {
                    let in_octant = (ox..ox + half).contains(&x)
                        && (oy..oy + half).contains(&y)
                        && (oz..oz + half).contains(&z);
                    let expected = if in_octant { STONE } else { AIR };
                    assert_eq!(
                        parent.get(LocalPos::new_unchecked(x as u8, y as u8, z as u8)),
                        expected
                    );
                }
            }
        }
        // The changed box is exactly the octant.
        let got = changed.unwrap();
        assert_eq!(got.min, [ox as u8, oy as u8, oz as u8]);
        assert_eq!(
            got.max,
            [(ox + half) as u8, (oy + half) as u8, (oz + half) as u8]
        );

        // A parent already uniform and equal is never unpacked: no change,
        // and no index array appears.
        let mut equal = Cube::filled(STONE);
        assert_eq!(downsample_octant(&child, &mut equal, 5), None);
        assert_eq!(equal.uniform(), Some(STONE));
        assert_eq!(equal.bits_per_entry(), 0);

        // Uniform air into uniform air: nothing.
        let mut air_parent = Cube::filled(AIR);
        assert_eq!(
            downsample_octant(&Cube::filled(AIR), &mut air_parent, 0),
            None
        );
    }

    #[test]
    fn downsample_region_returns_only_changed_cells() {
        // Parent already holds the octant-downsample of the child; a region
        // that changes nothing reports None and leaves the cube untouched.
        let child = Cube::from_fn(|p| {
            let y = p.y as usize;
            let third = CUBE_SIZE as usize / 3;
            if y < third {
                STONE
            } else if y < 2 * third {
                DIRT
            } else {
                AIR
            }
        });
        let mut parent = Cube::filled(AIR);
        downsample_octant(&child, &mut parent, 0);
        let before = parent.clone();

        let untouched = downsample_region(
            &child,
            &mut parent,
            0,
            CellBox {
                min: [0, 0, 0],
                max: [4, 4, 4],
            },
        );
        assert_eq!(untouched, None);
        assert_eq!(parent, before);

        // Edit one child cell so its parent cell flips: the region covers
        // it, and the returned box is that single parent cell.
        let mut edited = child.clone();
        let local = LocalPos::new_unchecked(5, 0, 5);
        edited.set(local, PLANKS);
        let group = LocalPos::new_unchecked(5 / 2, 0, 5 / 2);
        let changed = downsample_region(
            &edited,
            &mut parent,
            0,
            CellBox {
                min: [5, 0, 5],
                max: [6, 1, 6],
            },
        );
        // Stone group (8 stone) losing one cell keeps stone at the top of
        // every column... unless the flip matters; either way the reported
        // box must be exactly the recomputed cell when it changed.
        match changed {
            None => assert_eq!(parent.get(group), downsample_cell([STONE; 8])),
            Some(boxed) => {
                assert_eq!(boxed, CellBox::single(group));
                let children = gather(|dx, dy, dz| {
                    if (dx, dy, dz) == (1, 0, 1) {
                        PLANKS
                    } else {
                        STONE
                    }
                });
                assert_eq!(parent.get(group), downsample_cell(children));
            }
        }
        // And the parent satisfies its invariants after compact.
        parent.check_invariants();
    }

    #[test]
    fn parent_goes_back_to_uniform_after_an_airing_edit() {
        // A single-cell child inside uniform air: its parent cell is air, so
        // a region downsample over it leaves a uniform-air parent compact.
        let child = Cube::from_fn(|p| {
            if (p.x, p.y, p.z) == (0, 0, 0) {
                STONE
            } else {
                AIR
            }
        });
        let mut parent = Cube::filled(AIR);
        // Whole-octant pass writes nothing but air (1 of 8 filled): the
        // parent stays uniform and untouched.
        assert_eq!(downsample_octant(&child, &mut parent, 0), None);
        assert_eq!(parent.uniform(), Some(AIR));

        // Two filled children in one group, then removed again: the parent
        // was unpacked by the first pass and must compact back to uniform.
        let mut two = child.clone();
        two.set(LocalPos::new_unchecked(1, 0, 0), STONE);
        assert_eq!(
            downsample_octant(&two, &mut parent, 0),
            Some(CellBox::single(LocalPos::new_unchecked(0, 0, 0)))
        );
        assert_eq!(parent.uniform(), None);
        assert_eq!(
            downsample_octant(&child, &mut parent, 0),
            Some(CellBox::single(LocalPos::new_unchecked(0, 0, 0)))
        );
        assert_eq!(parent.uniform(), Some(AIR));
        parent.check_invariants();
    }
}
