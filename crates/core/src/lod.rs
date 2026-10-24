//! The LOD downsampling rule and the cube-level downsampling helpers
//! (tasks 2.4 and 2.14). Pure logic: no I/O, no threads, no registry data.
//!
//! A level-*n+1* cell covers a 2×2×2 group of level-*n* cells, its
//! "children", indexed `c = dx + 2·dz + 4·dy` — the same axis order as
//! [`LocalPos::index`](crate::coords::LocalPos::index). The rule for one
//! cell is [`downsample_cell`]; the cube-level helpers move between a child
//! cube and one octant of its parent.
//!
//! # The rule
//!
//! **Occupancy** (task 2.14). A child is *filled* when it is not air
//! (water and every other non-air state count; block properties are never
//! consulted). The cell is filled when **at least 1** of the 8 children is
//! filled; otherwise it is air. The coarse volume is therefore the union
//! of the fine volume rounded out to cells: thin walls, pillars and single
//! floating blocks all survive, and the coarse surface never sits below
//! the fine one (so downward skirts keep hiding seams). How *high* a
//! filled cell's surface is comes from its column top ([`LodTops`]), not
//! from the cell boundary, so the union does not make hills swell.
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
//! # Column tops (task 2.14)
//!
//! Every filled cell has a **fill height** `h ∈ 1..=2ⁿ` (level *n*): the
//! blocks from the cell's bottom to the top of its highest filled block.
//! At level 1 `h = 2` when any upper child (`dy = 1`) is filled, else 1;
//! at level *n* `h = max over filled children of (dy·2ⁿ⁻¹ + h_child)`.
//! A cube stores `h` only for the **topmost filled cell of each column**
//! ([`LodTops`]); every other filled cell counts as full (`h = 2ⁿ`),
//! including `h_child` for child cells that are not their child cube's
//! column top. That approximation flattens overhang undersides and
//! seabeds under water (the water cell is the column top); it is local and
//! deterministic and invisible on the shipped terrain. On real data the
//! top of a column is exact: the coarse top block is the highest filled
//! block of its footprint.
//!
//! [`LodTops`] keeps the value per column **per octant half** (the lower
//! and upper 16 rows), so the tops of an octant are a pure function of the
//! one child cube (or sampled cube) that fills it — exactly like the
//! cells. The column top of the whole cube is the upper half's value when
//! the upper half holds a filled cell, else the lower half's
//! ([`LodTops::get`]).
//!
//! **Locality.** The rule looks only at the eight children (and their
//! cube's column tops), never at cells outside the group, so one edit
//! changes at most one cell and one column top per level — that locality
//! is what makes edit propagation exact. Do not "improve" the rule by
//! consulting neighbours; a later task may refine it with registry data,
//! but it must stay local. Changing the rule bumps [`RULE_VERSION`].

use crate::block::BlockStateId;
use crate::coords::{BlockPos, CUBE_BITS, CUBE_SIZE, CubePos, LocalPos};
use crate::cube::Cube;

/// Children per parent cell.
pub const CHILDREN: usize = 8;

/// The version of the downsampling rule (occupancy, material and column
/// tops). Stored LOD is fingerprinted with it, so a world saved under an
/// older rule resamples instead of loading stale records. Version 2 is
/// task 2.14's 1-of-8 occupancy with column tops.
pub const RULE_VERSION: u32 = 2;

/// Columns of one cube: `CUBE_SIZE²`, indexed `x + CUBE_SIZE · z`.
pub const COLUMNS: usize = (CUBE_SIZE as usize) * (CUBE_SIZE as usize);

/// The far-look view distances in blocks, `(LOD0 disk radius, outermost
/// LOD ring radius)`: `view_radius · CUBE_SIZE` and `lod_view_radius ·
/// CUBE_SIZE · 2^lod_levels`. The one rule shared by the server config,
/// the game's view setup and the client's debug panel (task 2.12).
pub fn ring_radii(view_radius: u32, lod_view_radius: u32, lod_levels: u8) -> (f32, f32) {
    let cube = CUBE_SIZE as f32;
    (
        view_radius as f32 * cube,
        lod_view_radius as f32 * cube * (1u32 << lod_levels) as f32,
    )
}

/// The ring-radius floor (task 2.13 E): `lod_view_radius` must reach at
/// least `view_radius / 2 + 1`, so level 1 (`lod_view_radius · CUBE_SIZE
/// · 2` blocks) starts beyond the LOD0 disk (`view_radius · CUBE_SIZE`)
/// by at least one L1 cube — otherwise L1 draws nothing and LOD0's edge
/// meets a hole. The server config, the game and the debug panel all
/// clamp through this one rule.
pub fn min_lod_view_radius(view_radius: u32) -> u32 {
    view_radius / 2 + 1
}

/// A cube at a LOD level. Level 0 is the streamed world itself ("LOD0");
/// a level-*n* cube at `pos` covers blocks `pos * CUBE_SIZE * 2^n` onwards
/// and each of its cells is `2^n` blocks per side. One key type names
/// meshes, mesh jobs, GPU residency and coverage entries at every level,
/// on both sides of the link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CubeKey {
    pub level: u8,
    pub pos: CubePos,
}

impl CubeKey {
    /// The LOD0 cube at `pos`.
    pub fn lod0(pos: CubePos) -> CubeKey {
        CubeKey { level: 0, pos }
    }

    /// The parent cube one level up, and which octant of it this key fills
    /// (`dx + 2·dz + 4·dy`, as in [`parent`]). Callers with a level cap
    /// (the client's coverage tree stops at a fixed level) guard it
    /// themselves; the math itself is total.
    pub fn parent(&self) -> (CubeKey, u8) {
        let (pos, octant) = parent(self.pos);
        (
            CubeKey {
                level: self.level + 1,
                pos,
            },
            octant,
        )
    }

    /// The child cube filling `octant` of this one, one level down.
    pub fn child(&self, octant: u8) -> CubeKey {
        debug_assert!(self.level > 0, "level 0 has no children");
        let [dx, dy, dz] = octant_offset(octant);
        CubeKey {
            level: self.level - 1,
            pos: CubePos {
                x: self.pos.x * 2 + dx as i32,
                y: self.pos.y * 2 + dy as i32,
                z: self.pos.z * 2 + dz as i32,
            },
        }
    }

    /// The world-space side length of one cube at this level, in blocks:
    /// `CUBE_SIZE * 2^level`.
    pub fn block_size(&self) -> u64 {
        u64::from(CUBE_SIZE as u32) << self.level
    }

    /// The world-space corner block of this cube — its minimum block
    /// coordinate on every axis.
    pub fn min_block(&self) -> BlockPos {
        debug_assert!(self.level < 32, "level shifts past any i64 coordinate");
        let size = i64::from(CUBE_SIZE) << self.level;
        BlockPos {
            x: i64::from(self.pos.x) * size,
            y: i64::from(self.pos.y) * size,
            z: i64::from(self.pos.z) * size,
        }
    }

    /// The level-`level` cube containing block `pos` — floor arithmetic,
    /// so block -1 lands in cube -1 exactly as [`BlockPos::split`] at
    /// level 0. The inverse of [`CubeKey::min_block`] on the cube's span.
    pub fn containing(level: u8, pos: BlockPos) -> CubeKey {
        debug_assert!(level < 32, "level shifts past any i64 coordinate");
        let shift = CUBE_BITS + level as u32;
        CubeKey {
            level,
            pos: CubePos {
                x: (pos.x >> shift) as i32,
                y: (pos.y >> shift) as i32,
                z: (pos.z >> shift) as i32,
            },
        }
    }
}

/// The downsampling rule for one cell: a pure function of the eight child
/// states, indexed `c = dx + 2·dz + 4·dy`. See the module docs for the rule
/// and its rationale.
pub fn downsample_cell(children: [BlockStateId; CHILDREN]) -> BlockStateId {
    // Eight equal states (the bulk of any real cube) give that state.
    if children.iter().all(|&state| state == children[0]) {
        return children[0];
    }
    // Occupancy: any filled child fills the cell. (Eight air children
    // took the early return above.)

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
    debug_assert!(count >= 1, "a filled child gives at least one candidate");

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

/// The column tops of one LOD cube (task 2.14): for each column `(x, z)`
/// the fill height `h` of its topmost filled cell, in blocks
/// (`1..=2^level`; 0 for a column with no filled cell). See the module
/// docs for the rule.
///
/// The value is kept **per octant half** — the lower (`y < CUBE_SIZE/2`)
/// and upper rows separately — so an octant's tops depend only on the
/// cube that fills that octant: [`downsample_region`], [`downsample_octant`]
/// and [`copy_octant`] each write exactly their octant's quarter of the
/// columns of one half. [`LodTops::get`] combines the halves into the
/// cube's column top. Uniform tops (all empty, all full, a flat partial
/// plane) have no array at all, like a uniform [`Cube`].
///
/// [`LodTops::LOD0`] stands in for a level-0 child: every block is one
/// full block (`get` returns 1 everywhere — callers only read it where a
/// block is filled).
#[derive(Clone)]
pub struct LodTops {
    level: u8,
    data: TopsData,
}

#[derive(Clone)]
enum TopsData {
    Uniform(u8),
    /// `[half][z][x]`: index `half · COLUMNS + x + CUBE_SIZE · z`.
    Mixed(Box<[u8; 2 * COLUMNS]>),
}

/// Codec version byte of [`LodTops::encode`].
const TOPS_CODEC_VERSION: u8 = 1;

impl LodTops {
    /// The tops of a level-0 cube: one block per cell, so every filled cell
    /// is full at height 1.
    pub const LOD0: LodTops = LodTops {
        level: 0,
        data: TopsData::Uniform(1),
    };

    /// Every column empty (uniform air).
    pub fn empty(level: u8) -> Self {
        Self::uniform_at(level, 0)
    }

    /// Every column topped by a full cell (`h = 2^level`).
    pub fn full(level: u8) -> Self {
        Self::uniform_at(level, 1 << level)
    }

    /// Every column (both halves) at height `h`.
    pub fn uniform_at(level: u8, h: u8) -> Self {
        debug_assert!(level <= 7, "a fill height must fit a u8");
        debug_assert!(
            u32::from(h) <= 1 << level,
            "height {h} beyond level {level}"
        );
        Self {
            level,
            data: TopsData::Uniform(h),
        }
    }

    /// The LOD level these tops belong to.
    pub fn level(&self) -> u8 {
        self.level
    }

    /// A full cell's height in blocks: `2^level`.
    pub fn full_height(&self) -> u8 {
        1 << self.level
    }

    /// The single value of uniform tops, `None` when they vary.
    pub fn uniform(&self) -> Option<u8> {
        match &self.data {
            TopsData::Uniform(h) => Some(*h),
            TopsData::Mixed(_) => None,
        }
    }

    fn index(half: u8, x: u8, z: u8) -> usize {
        debug_assert!(half < 2 && (x as i32) < CUBE_SIZE && (z as i32) < CUBE_SIZE);
        half as usize * COLUMNS + x as usize + CUBE_SIZE as usize * z as usize
    }

    /// The cube's column top at `(x, z)`: the fill height of the topmost
    /// filled cell of the column, 0 when the column is empty.
    pub fn get(&self, x: u8, z: u8) -> u8 {
        match &self.data {
            TopsData::Uniform(h) => *h,
            TopsData::Mixed(values) => {
                let upper = values[Self::index(1, x, z)];
                if upper != 0 {
                    upper
                } else {
                    values[Self::index(0, x, z)]
                }
            }
        }
    }

    /// The top of column `(x, z)` within one octant half (`0` = the lower
    /// `CUBE_SIZE / 2` rows, `1` = the upper ones).
    pub fn half_top(&self, half: u8, x: u8, z: u8) -> u8 {
        match &self.data {
            TopsData::Uniform(h) => *h,
            TopsData::Mixed(values) => values[Self::index(half, x, z)],
        }
    }

    /// Sets one half-column's top and returns the previous value. Setting
    /// the value already there never unpacks uniform tops.
    pub fn set_half(&mut self, half: u8, x: u8, z: u8, h: u8) -> u8 {
        debug_assert!(
            u32::from(h) <= 1 << self.level,
            "height {h} beyond level {}",
            self.level
        );
        let index = Self::index(half, x, z);
        match &mut self.data {
            TopsData::Uniform(old) => {
                let old = *old;
                if old != h {
                    let mut values = Box::new([old; 2 * COLUMNS]);
                    values[index] = h;
                    self.data = TopsData::Mixed(values);
                }
                old
            }
            TopsData::Mixed(values) => std::mem::replace(&mut values[index], h),
        }
    }

    /// Returns mixed tops whose values are all equal to the uniform form.
    pub fn compact(&mut self) {
        if let TopsData::Mixed(values) = &self.data {
            let first = values[0];
            if values.iter().all(|&h| h == first) {
                self.data = TopsData::Uniform(first);
            }
        }
    }

    /// Heap bytes held: 0 when uniform.
    pub fn heap_bytes(&self) -> usize {
        match self.data {
            TopsData::Uniform(_) => 0,
            TopsData::Mixed(_) => 2 * COLUMNS,
        }
    }

    /// Whether these tops agree with `cube`'s cells: a half-column holds a
    /// non-zero top exactly when it holds a filled cell. A cheap invariant
    /// check for tests and debug assertions.
    pub fn consistent_with(&self, cube: &Cube) -> bool {
        let half = CUBE_SIZE as u8 / 2;
        for dy in 0..2u8 {
            for z in 0..CUBE_SIZE as u8 {
                for x in 0..CUBE_SIZE as u8 {
                    let filled = (dy * half..dy * half + half)
                        .any(|y| cube.get(LocalPos::new_unchecked(x, y, z)) != BlockStateId::AIR);
                    let h = self.half_top(dy, x, z);
                    if filled != (h != 0) || u32::from(h) > 1 << self.level {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Serializes the tops: `[codec version][level][kind]` then the
    /// uniform value (kind 0) or the `2 · COLUMNS` half-column values
    /// (kind 1). The storage layer zstd-compresses it with the cube.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.push(TOPS_CODEC_VERSION);
        out.push(self.level);
        match &self.data {
            TopsData::Uniform(h) => {
                out.push(0);
                out.push(*h);
            }
            TopsData::Mixed(values) => {
                out.push(1);
                out.extend_from_slice(&values[..]);
            }
        }
    }

    /// Inverse of [`LodTops::encode`]: validates the version, the level
    /// (`1..=7`), the exact length and every height against the level.
    /// Returns the tops and the bytes it consumed. Never panics.
    pub fn decode(bytes: &[u8]) -> Result<(LodTops, usize), String> {
        let [version, level, kind, rest @ ..] = bytes else {
            return Err("truncated tops header".into());
        };
        if *version != TOPS_CODEC_VERSION {
            return Err(format!("tops codec version {version}"));
        }
        if !(1..=7).contains(level) {
            return Err(format!("tops level {level}"));
        }
        let full = 1u32 << level;
        let check = |h: u8| {
            if u32::from(h) > full {
                Err(format!("tops height {h} beyond level {level}"))
            } else {
                Ok(())
            }
        };
        match kind {
            0 => {
                let Some(&h) = rest.first() else {
                    return Err("truncated uniform tops".into());
                };
                check(h)?;
                Ok((LodTops::uniform_at(*level, h), 4))
            }
            1 => {
                if rest.len() < 2 * COLUMNS {
                    return Err("truncated tops array".into());
                }
                let mut values = Box::new([0u8; 2 * COLUMNS]);
                values.copy_from_slice(&rest[..2 * COLUMNS]);
                for &h in values.iter() {
                    check(h)?;
                }
                let mut tops = LodTops {
                    level: *level,
                    data: TopsData::Mixed(values),
                };
                tops.compact();
                Ok((tops, 3 + 2 * COLUMNS))
            }
            other => Err(format!("tops kind {other}")),
        }
    }
}

impl PartialEq for LodTops {
    /// Content equality: the level and every half-column value, whatever
    /// the storage form.
    fn eq(&self, other: &Self) -> bool {
        if self.level != other.level {
            return false;
        }
        match (&self.data, &other.data) {
            (TopsData::Uniform(a), TopsData::Uniform(b)) => a == b,
            (TopsData::Mixed(a), TopsData::Mixed(b)) => a == b,
            (TopsData::Uniform(h), TopsData::Mixed(values))
            | (TopsData::Mixed(values), TopsData::Uniform(h)) => values.iter().all(|v| v == h),
        }
    }
}

impl Eq for LodTops {}

impl std::fmt::Debug for LodTops {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            TopsData::Uniform(h) => write!(f, "LodTops(L{}, uniform {h})", self.level),
            TopsData::Mixed(values) => {
                let (min, max) = values
                    .iter()
                    .fold((u8::MAX, 0u8), |(lo, hi), &h| (lo.min(h), hi.max(h)));
                write!(f, "LodTops(L{}, mixed {min}..={max})", self.level)
            }
        }
    }
}

/// Recomputes every parent cell whose 2×2×2 group of child cells overlaps
/// `region` (a box of **child** cells), writing into `octant` of `parent`,
/// and the octant's column tops (`parent_tops`, from the child's cells and
/// `child_tops`; [`LodTops::LOD0`] for a level-0 child). Returns the box of
/// **parent** cells that actually changed — a cell's state, or the height
/// of a column-top cell — or `None` when none did; that box is the next
/// level's input, so propagation stays proportional to what changed.
/// `region` must cover every child cell whose state or column-top height
/// changed. [`Cube::compact`] runs after a write, so a parent that became
/// uniform (or shed palette entries) goes back to its cheapest storage;
/// that cost is part of this call and covered by the `lod/edit_cell`
/// bench.
pub fn downsample_region(
    child: &Cube,
    child_tops: &LodTops,
    parent: &mut Cube,
    parent_tops: &mut LodTops,
    octant: u8,
    region: CellBox,
) -> Option<CellBox> {
    debug_assert_eq!(parent_tops.level(), child_tops.level() + 1);
    // Parent cells (octant-local) whose group {2k, 2k+1} overlaps
    // [min, max): k from floor(min / 2) to ceil(max / 2), exclusive.
    let first = region.min.map(|m| m as usize / 2);
    let last = region.max.map(|m| (m as usize).div_ceil(2));
    let groups = (last[0] - first[0]) * (last[1] - first[1]) * (last[2] - first[2]);
    let target = Target {
        child_tops,
        parent,
        parent_tops,
        octant,
    };
    // Large regions (whole-octant ingest) unpack the child once with the
    // sequential iterator, which is several times cheaper per cell than a
    // random `get`; small ones (edits) read the few cells they need.
    if groups >= UNPACK_GROUPS {
        let cells: Vec<BlockStateId> = child.iter().collect();
        downsample_groups(|i| cells[i], target, first, last)
    } else {
        downsample_groups(|i| child.get_index(i), target, first, last)
    }
}

/// Region size (in parent cells) from which [`downsample_region`] unpacks
/// the whole child instead of reading cells one by one.
const UNPACK_GROUPS: usize = 512;

/// What one octant downsample writes: the parent's cells and tops, from
/// the child's tops.
struct Target<'a> {
    child_tops: &'a LodTops,
    parent: &'a mut Cube,
    parent_tops: &'a mut LodTops,
    octant: u8,
}

/// Grows `changed` by one cell.
fn mark(changed: &mut Option<CellBox>, pos: LocalPos) {
    let cell = CellBox::single(pos);
    *changed = Some(match *changed {
        None => cell,
        Some(boxed) => boxed.union(cell),
    });
}

/// The column-top height of parent cell group `(kx, ky, kz)` (octant-local
/// group indices = child cell halves): the §2 recurrence over its filled
/// children, `max(dy · child_full + child_tops(x, z))`. Valid for the
/// topmost filled cell of a parent column, whose filled children are all
/// their child cube's column tops (everything above them is air).
fn group_top(
    read: &impl Fn(usize) -> BlockStateId,
    child_tops: &LodTops,
    [kx, ky, kz]: [usize; 3],
) -> u8 {
    let child_full = u16::from(child_tops.full_height());
    let mut h = 0u16;
    for dy in 0..2usize {
        for dz in 0..2usize {
            for dx in 0..2usize {
                let (cx, cy, cz) = (2 * kx + dx, 2 * ky + dy, 2 * kz + dz);
                let index = LocalPos::new_unchecked(cx as u8, cy as u8, cz as u8).index();
                if read(index) != BlockStateId::AIR {
                    let top = u16::from(child_tops.get(cx as u8, cz as u8));
                    h = h.max(dy as u16 * child_full + top);
                }
            }
        }
    }
    debug_assert!(h <= 2 * child_full);
    h as u8
}

/// The shared loop of [`downsample_region`]: `read` returns the child state
/// at a linear [`LocalPos::index`], `first..last` are the octant-local
/// parent cells to recompute. Cells first, then the column tops of every
/// column the region touches.
fn downsample_groups(
    read: impl Fn(usize) -> BlockStateId,
    target: Target<'_>,
    first: [usize; 3],
    last: [usize; 3],
) -> Option<CellBox> {
    let Target {
        child_tops,
        parent,
        parent_tops,
        octant,
    } = target;
    let half = CUBE_SIZE as usize / 2;
    let offset = octant_offset(octant);
    let [ox, oy, oz] = offset.map(|v| v as usize * half);
    let mut changed: Option<CellBox> = None;
    let mut children = [BlockStateId::AIR; CHILDREN];
    // The topmost filled recomputed row of each octant column, by
    // `kx + half · kz`; `u8::MAX` while none.
    let mut top_row = [u8::MAX; COLUMNS / 4];
    for ky in first[1]..last[1] {
        for kz in first[2]..last[2] {
            for kx in first[0]..last[0] {
                for dy in 0..2usize {
                    for dz in 0..2usize {
                        for dx in 0..2usize {
                            let index = LocalPos::new_unchecked(
                                (2 * kx + dx) as u8,
                                (2 * ky + dy) as u8,
                                (2 * kz + dz) as u8,
                            )
                            .index();
                            children[dx + 2 * dz + 4 * dy] = read(index);
                        }
                    }
                }
                // The parent cell's octant-local coordinates are the group
                // indices k themselves, not the loop ordinal.
                let pos =
                    LocalPos::new_unchecked((ox + kx) as u8, (oy + ky) as u8, (oz + kz) as u8);
                let down = downsample_cell(children);
                if down != BlockStateId::AIR {
                    top_row[kx + half * kz] = ky as u8;
                }
                if parent.set(pos, down) != down {
                    mark(&mut changed, pos);
                }
            }
        }
    }

    // Column tops of the touched columns (this octant's half only).
    for kz in first[2]..last[2] {
        for kx in first[0]..last[0] {
            let (px, pz) = ((ox + kx) as u8, (oz + kz) as u8);
            let at = |ky: usize| LocalPos::new_unchecked(px, (oy + ky) as u8, pz);
            // A filled row above the region keeps the column top — and its
            // height, whose children the region never touched — as it was.
            if (last[1]..half).any(|ky| parent.get(at(ky)) != BlockStateId::AIR) {
                continue;
            }
            let top = match top_row[kx + half * kz] {
                u8::MAX => (0..first[1])
                    .rev()
                    .find(|&ky| parent.get(at(ky)) != BlockStateId::AIR),
                row => Some(row as usize),
            };
            let h = top.map_or(0, |ky| group_top(&read, child_tops, [kx, ky, kz]));
            if parent_tops.set_half(offset[1], px, pz, h) != h
                && let Some(ky) = top
            {
                mark(&mut changed, at(ky));
            }
        }
    }
    if changed.is_some() {
        parent.compact();
        parent_tops.compact();
    }
    changed
}

/// [`downsample_region`] over the whole child cube, with two fast paths: a
/// uniform child fills its octant with its own state (downsampling eight
/// equal states yields that state — asserted by the tests) and its tops
/// from the child's, and a parent that is already uniform and equal to
/// that fill — cells and tops — is left untouched, so it is never
/// unpacked.
pub fn downsample_octant(
    child: &Cube,
    child_tops: &LodTops,
    parent: &mut Cube,
    parent_tops: &mut LodTops,
    octant: u8,
) -> Option<CellBox> {
    debug_assert_eq!(parent_tops.level(), child_tops.level() + 1);
    let Some(state) = child.uniform() else {
        return downsample_region(
            child,
            child_tops,
            parent,
            parent_tops,
            octant,
            CellBox {
                min: [0, 0, 0],
                max: [CUBE_SIZE as u8; 3],
            },
        );
    };
    let filled = state != BlockStateId::AIR;
    let child_full = child_tops.full_height();
    // A uniform filled child's every column tops at its top row, so the
    // parent's top row tops every octant column at child_full + the
    // highest of the four child column tops below it.
    let column_top = |kx: usize, kz: usize| -> u8 {
        if !filled {
            return 0;
        }
        let mut top = 0;
        for dz in 0..2 {
            for dx in 0..2 {
                top = top.max(child_tops.get((2 * kx + dx) as u8, (2 * kz + dz) as u8));
            }
        }
        child_full + top
    };
    let uniform_top = match child_tops.uniform() {
        _ if !filled => Some(0),
        Some(top) => Some(child_full + top),
        None => None,
    };
    if parent.uniform() == Some(state)
        && uniform_top.is_some()
        && parent_tops.uniform() == uniform_top
    {
        return None;
    }
    let half = CUBE_SIZE as usize / 2;
    let offset = octant_offset(octant);
    let [ox, oy, oz] = offset.map(|v| v as usize * half);
    let mut changed: Option<CellBox> = None;
    for y in oy..oy + half {
        for z in oz..oz + half {
            for x in ox..ox + half {
                let pos = LocalPos::new_unchecked(x as u8, y as u8, z as u8);
                if parent.get(pos) != state {
                    parent.set(pos, state);
                    mark(&mut changed, pos);
                }
            }
        }
    }
    for kz in 0..half {
        for kx in 0..half {
            let (px, pz) = ((ox + kx) as u8, (oz + kz) as u8);
            let h = uniform_top.unwrap_or_else(|| column_top(kx, kz));
            if parent_tops.set_half(offset[1], px, pz, h) != h && filled {
                mark(
                    &mut changed,
                    LocalPos::new_unchecked(px, (oy + half - 1) as u8, pz),
                );
            }
        }
    }
    if changed.is_some() {
        parent.compact();
        parent_tops.compact();
    }
    changed
}

/// Copies one octant's cells and column tops from `src` / `src_tops` to
/// `parent` / `parent_tops` (same octant of both), returning the box of
/// `parent` cells that actually changed — state or column-top height —
/// or `None`. This is the tick-thread half of the precomputed-octant
/// ingest (task 2.7): a generation worker downsamples a LOD0 cube into a
/// scratch cube's octant off-thread ([`downsample_octant`]), and the tick
/// thread installs it with this copy — no downsampling on the tick thread.
/// The fill pipeline installs sampled octants the same way.
pub fn copy_octant(
    src: &Cube,
    src_tops: &LodTops,
    parent: &mut Cube,
    parent_tops: &mut LodTops,
    octant: u8,
) -> Option<CellBox> {
    debug_assert_eq!(parent_tops.level(), src_tops.level());
    let half = CUBE_SIZE as usize / 2;
    let offset = octant_offset(octant);
    let [ox, oy, oz] = offset.map(|v| v as usize * half);
    let mut changed: Option<CellBox> = None;
    // A uniform source fills its octant with one state, the same shortcut
    // `downsample_octant` takes for uniform children.
    let uniform = src.uniform();
    let src_at = |pos: LocalPos| match uniform {
        Some(state) => state,
        None => src.get(pos),
    };
    for y in oy..oy + half {
        for z in oz..oz + half {
            for x in ox..ox + half {
                let pos = LocalPos::new_unchecked(x as u8, y as u8, z as u8);
                let state = src_at(pos);
                if parent.set(pos, state) != state {
                    mark(&mut changed, pos);
                }
            }
        }
    }
    let same_uniform_tops =
        src_tops.uniform().is_some() && src_tops.uniform() == parent_tops.uniform();
    if !same_uniform_tops {
        for z in oz..oz + half {
            for x in ox..ox + half {
                let (x, z) = (x as u8, z as u8);
                let h = src_tops.half_top(offset[1], x, z);
                if parent_tops.set_half(offset[1], x, z, h) != h && h != 0 {
                    // The column top cell: the octant column's topmost
                    // filled cell.
                    if let Some(y) = (oy..oy + half).rev().find(|&y| {
                        src_at(LocalPos::new_unchecked(x, y as u8, z)) != BlockStateId::AIR
                    }) {
                        mark(&mut changed, LocalPos::new_unchecked(x, y as u8, z));
                    }
                }
            }
        }
    }
    if changed.is_some() {
        parent.compact();
        parent_tops.compact();
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const AIR: BlockStateId = BlockStateId::AIR;
    const STONE: BlockStateId = BlockStateId(1);
    const DIRT: BlockStateId = BlockStateId(2);
    const GRASS: BlockStateId = BlockStateId(3);
    const PLANKS: BlockStateId = BlockStateId(4);

    /// Task 2.13 E: the floor keeps L1 beyond LOD0 at the defaults and at
    /// every slider combination.
    #[test]
    fn min_lod_view_radius_keeps_l1_beyond_lod0() {
        let cube = CUBE_SIZE as f32;
        let check = |view: u32, lod: u32| {
            let lod0 = view as f32 * cube;
            let l1 = lod as f32 * cube * 2.0;
            assert!(l1 > lod0, "L1 {l1} not beyond LOD0 {lod0}");
        };
        check(16, min_lod_view_radius(16));
        for view in 2..=32u32 {
            check(view, min_lod_view_radius(view));
        }
        // The floor itself is the smallest radius that satisfies it.
        assert_eq!(min_lod_view_radius(16), 9);
        assert_eq!(min_lod_view_radius(32), 17);
        assert_eq!(min_lod_view_radius(8), 5);
    }

    /// The one far-look radius rule (task 2.12): the LOD0 disk in blocks,
    /// and the outermost ring scaled by `2^lod_levels`. The shipped
    /// defaults (8, 8, 6) give 8 cubes and 512 cubes.
    #[test]
    fn ring_radii_follow_the_level_count() {
        let cube = CUBE_SIZE as f32;
        assert_eq!(ring_radii(8, 8, 6), (8.0 * cube, 8.0 * cube * 64.0));
        assert_eq!(ring_radii(4, 2, 0), (4.0 * cube, 2.0 * cube));
        assert_eq!(ring_radii(32, 16, 3).1, 16.0 * cube * 8.0);
    }

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
    fn occupancy_threshold_is_one() {
        assert_eq!(downsample_cell([AIR; 8]), AIR);
        // Every single filled child fills the cell with its own state.
        for filled in 0..8 {
            let mut children = [AIR; 8];
            children[filled] = STONE;
            assert_eq!(downsample_cell(children), STONE, "filled at {filled}");
        }
        // Every pair of filled children: filled. The material is the pair's
        // vote — one candidate each when they sit in different columns, so
        // the higher dy (then the smaller id) decides; in one column the
        // upper child is the only candidate.
        for a in 0..8 {
            for b in (a + 1)..8 {
                let mut children = [AIR; 8];
                children[a] = STONE;
                children[b] = DIRT;
                let (dy_a, dy_b) = (a >> 2, b >> 2);
                let expected = if a & 3 == b & 3 {
                    DIRT // same column: b is the upper child
                } else if dy_a > dy_b || (dy_a == dy_b && STONE < DIRT) {
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
            if filled < 1 {
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

    fn key(level: u8, x: i32, y: i32, z: i32) -> CubeKey {
        CubeKey {
            level,
            pos: CubePos { x, y, z },
        }
    }

    #[test]
    fn cube_key_parent_child_round_trip_at_negative_coordinates() {
        for (level, x, y, z) in [
            (1u8, -3i32, -2i32, -1i32),
            (2, -1, 0, 1),
            (3, 7, -8, 9),
            (0, -5, 5, -6),
        ] {
            let k = key(level, x, y, z);
            let (parent, octant) = k.parent();
            assert_eq!(parent.child(octant), k);
            assert_eq!(parent.level, level + 1);
            assert_eq!(parent.pos.x, x.div_euclid(2));
        }
        // Level 0 children of a level-1 cube tile its volume exactly.
        let parent = key(1, -2, 3, -4);
        let mut seen = HashSet::new();
        for octant in 0..8u8 {
            let child = parent.child(octant);
            assert_eq!(child.level, 0);
            assert!(seen.insert(child), "octants overlap");
            let (back, o) = child.parent();
            assert_eq!((back, o), (parent, octant));
        }
    }

    #[test]
    fn cube_key_block_size_scales_with_level() {
        assert_eq!(key(0, 0, 0, 0).block_size(), u64::from(CUBE_SIZE as u32));
        assert_eq!(
            key(1, 0, 0, 0).block_size(),
            u64::from(CUBE_SIZE as u32) * 2
        );
        assert_eq!(
            key(4, 0, 0, 0).block_size(),
            u64::from(CUBE_SIZE as u32) * 16
        );
    }

    /// `min_block` and `containing` are inverses over the cube's whole
    /// block span, negative coordinates included, and `containing(0, p)`
    /// agrees with `BlockPos::split`.
    #[test]
    fn cube_key_min_block_and_containing_round_trip() {
        for (level, x, y, z) in [(0u8, 0i32, 0, 0), (1, -1, 2, -3), (3, 5, -6, 7)] {
            let k = key(level, x, y, z);
            let min = k.min_block();
            let size = k.block_size() as i64;
            assert_eq!(CubeKey::containing(level, min), k);
            assert_eq!(
                CubeKey::containing(
                    level,
                    BlockPos {
                        x: min.x + size - 1,
                        y: min.y + size - 1,
                        z: min.z + size - 1,
                    }
                ),
                k
            );
            // One block past the span on any axis belongs to a neighbour.
            assert_ne!(
                CubeKey::containing(
                    level,
                    BlockPos {
                        x: min.x + size,
                        y: min.y,
                        z: min.z,
                    }
                ),
                k
            );
        }
        // Level 0 is exactly `BlockPos::split`.
        for (x, y, z) in [(0i64, 1, -1), (-33, 32, 31), (64, -65, 96)] {
            let pos = BlockPos { x, y, z };
            let (split_pos, _) = pos.split();
            assert_eq!(
                CubeKey::containing(0, pos),
                CubeKey {
                    level: 0,
                    pos: split_pos
                }
            );
        }
    }

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

    /// A LOD0 child's octant downsample into a fresh level-1 parent:
    /// cells and tops.
    fn down0(child: &Cube, octant: u8) -> (Cube, LodTops, Option<CellBox>) {
        let mut parent = Cube::filled(AIR);
        let mut tops = LodTops::empty(1);
        let changed = downsample_octant(child, &LodTops::LOD0, &mut parent, &mut tops, octant);
        (parent, tops, changed)
    }

    /// A cube whose octant-downsample is checked cell by cell against a
    /// direct application of `downsample_cell`.
    fn check_octant(child: &Cube, octant: u8) -> (Cube, Option<CellBox>) {
        let (parent, tops, changed) = down0(child, octant);
        assert!(tops.consistent_with(&parent));
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
                STONE // the single floating block: survives
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

    /// Task 2.14: a single floating block survives LOD1 as one cell of its
    /// own state, at height 1 (lower child) or 2 (upper child).
    #[test]
    fn a_single_floating_block_survives_as_one_short_cell() {
        let size = CUBE_SIZE as u8;
        for (y, height) in [(size - 1, 2u8), (size - 2, 1), (4, 1), (5, 2)] {
            let child = Cube::from_fn(|p| {
                if (p.x, p.y, p.z) == (3, y, 6) {
                    STONE
                } else {
                    AIR
                }
            });
            let (parent, tops, changed) = down0(&child, 0);
            let cell = LocalPos::new_unchecked(1, y / 2, 3);
            assert_eq!(parent.get(cell), STONE, "block at y {y}");
            assert_eq!(changed, Some(CellBox::single(cell)));
            let filled = parent.iter().filter(|&s| s != AIR).count();
            assert_eq!(filled, 1, "exactly one coarse cell");
            assert_eq!(tops.get(1, 3), height, "block at y {y}");
            assert_eq!(tops.get(0, 0), 0);
            assert!(tops.consistent_with(&parent));
        }
    }

    /// Thin features survive every level: a one-block wall and a 1×1
    /// pillar, taken up four levels through uniform-air siblings.
    #[test]
    fn thin_walls_and_pillars_survive_every_level() {
        let child = Cube::from_fn(|p| {
            if p.x == 9 || (p.x, p.z) == (20, 20) {
                PLANKS
            } else {
                AIR
            }
        });
        let mut cube = child;
        let mut tops = LodTops::LOD0;
        let (mut wall_x, mut pillar) = (9usize, 20usize);
        for level in 1..=4u8 {
            let mut parent = Cube::filled(AIR);
            let mut parent_tops = LodTops::empty(level);
            downsample_octant(&cube, &tops, &mut parent, &mut parent_tops, 0);
            (wall_x, pillar) = (wall_x / 2, pillar / 2);
            // The content shrinks into the origin octant level by level.
            let extent = CUBE_SIZE as usize >> level;
            for y in 0..extent {
                for z in 0..extent {
                    let wall = LocalPos::new_unchecked(wall_x as u8, y as u8, z as u8);
                    assert_eq!(parent.get(wall), PLANKS, "L{level} wall at {wall:?}");
                }
                let column = LocalPos::new_unchecked(pillar as u8, y as u8, pillar as u8);
                assert_eq!(parent.get(column), PLANKS, "L{level} pillar");
            }
            assert_eq!(parent_tops.get(wall_x as u8, 0), 1 << level, "full height");
            (cube, tops) = (parent, parent_tops);
        }
    }

    /// The §2 recurrence on hand-built children: the top cell's height is
    /// max(dy · child_full + child_top) over its filled children.
    #[test]
    fn fill_height_recurrence() {
        // A level-1 child whose column (0, 0) tops at y 1 with h 1 and
        // column (1, 0) tops at y 0 with h 2: the level-2 group (0, 0, 0)
        // takes max(1·2 + 1, 0·2 + 2) = 3.
        let child = Cube::from_fn(|p| match (p.x, p.y, p.z) {
            (0, 0..=1, 0) | (1, 0, 0) => STONE,
            _ => AIR,
        });
        let mut child_tops = LodTops::empty(1);
        child_tops.set_half(0, 0, 0, 1);
        child_tops.set_half(0, 1, 0, 2);
        assert!(child_tops.consistent_with(&child));
        let mut parent = Cube::filled(AIR);
        let mut tops = LodTops::empty(2);
        let changed = downsample_octant(&child, &child_tops, &mut parent, &mut tops, 0);
        assert_eq!(
            changed,
            Some(CellBox::single(LocalPos::new_unchecked(0, 0, 0)))
        );
        assert_eq!(tops.get(0, 0), 3);
        // Only the upper child: 2 + 1 = 3; only the lower one at full
        // height: 2. The level-0 case: 2 with an upper block, else 1.
        let upper_only = Cube::from_fn(|p| {
            if (p.x, p.y, p.z) == (0, 1, 0) {
                STONE
            } else {
                AIR
            }
        });
        let mut tops = LodTops::empty(2);
        downsample_octant(
            &upper_only,
            &child_tops,
            &mut Cube::filled(AIR),
            &mut tops,
            0,
        );
        assert_eq!(tops.get(0, 0), 3);
        let lower_only = Cube::from_fn(|p| {
            if (p.x, p.y, p.z) == (1, 0, 0) {
                STONE
            } else {
                AIR
            }
        });
        let mut tops = LodTops::empty(2);
        downsample_octant(
            &lower_only,
            &child_tops,
            &mut Cube::filled(AIR),
            &mut tops,
            0,
        );
        assert_eq!(tops.get(0, 0), 2);
        // A uniform child: every parent column tops at child_full + the
        // highest child top under it.
        let mut partial = LodTops::uniform_at(1, 2);
        for (x, z) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            partial.set_half(1, x, z, 1);
        }
        let mut tops = LodTops::empty(2);
        downsample_octant(
            &Cube::filled(STONE),
            &partial,
            &mut Cube::filled(AIR),
            &mut tops,
            7,
        );
        let half = CUBE_SIZE as u8 / 2;
        assert_eq!(
            tops.get(half, half),
            3,
            "1 + 2: all four child columns short"
        );
        assert_eq!(tops.get(half + 1, half), 4);
        assert_eq!(
            tops.half_top(0, half, half),
            0,
            "the lower half is not this octant's"
        );
    }

    #[test]
    fn uniform_child_fills_its_octant_and_equal_parent_is_untouched() {
        let child = Cube::filled(STONE);
        let mut parent = Cube::filled(AIR);
        let mut tops = LodTops::empty(1);
        let changed = downsample_octant(&child, &LodTops::LOD0, &mut parent, &mut tops, 5);
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
        assert!(tops.consistent_with(&parent));
        assert_eq!(tops.get(ox as u8, oz as u8), 2, "full at level 1");
        // The changed box is exactly the octant.
        let got = changed.unwrap();
        assert_eq!(got.min, [ox as u8, oy as u8, oz as u8]);
        assert_eq!(
            got.max,
            [(ox + half) as u8, (oy + half) as u8, (oz + half) as u8]
        );

        // A parent already uniform and equal — cells and tops — is never
        // unpacked: no change, and no index array appears.
        let mut equal = Cube::filled(STONE);
        let mut equal_tops = LodTops::full(1);
        assert_eq!(
            downsample_octant(&child, &LodTops::LOD0, &mut equal, &mut equal_tops, 5),
            None
        );
        assert_eq!(equal.uniform(), Some(STONE));
        assert_eq!(equal.bits_per_entry(), 0);
        assert_eq!(equal_tops.heap_bytes(), 0);

        // Uniform air into uniform air: nothing.
        let mut air_parent = Cube::filled(AIR);
        let mut air_tops = LodTops::empty(1);
        assert_eq!(
            downsample_octant(
                &Cube::filled(AIR),
                &LodTops::LOD0,
                &mut air_parent,
                &mut air_tops,
                0
            ),
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
        let (mut parent, mut tops, _) = down0(&child, 0);
        let before = (parent.clone(), tops.clone());
        let lod0 = &LodTops::LOD0;

        let untouched = downsample_region(
            &child,
            lod0,
            &mut parent,
            &mut tops,
            0,
            CellBox {
                min: [0, 0, 0],
                max: [4, 4, 4],
            },
        );
        assert_eq!(untouched, None);
        assert_eq!((parent.clone(), tops.clone()), before);

        // Edit one child cell so its parent cell flips: the region covers
        // it, and the returned box is that single parent cell.
        let mut edited = child.clone();
        let local = LocalPos::new_unchecked(5, 0, 5);
        edited.set(local, PLANKS);
        let group = LocalPos::new_unchecked(5 / 2, 0, 5 / 2);
        let changed = downsample_region(
            &edited,
            lod0,
            &mut parent,
            &mut tops,
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
        assert!(tops.consistent_with(&parent));

        // Lowering a column top by one block changes no cell state but the
        // column's height: the box is the top cell.
        let top_y = (2 * (CUBE_SIZE as usize / 3) - 1) as u8;
        assert_eq!(top_y % 2, 1, "the top block is an upper child");
        let mut lowered = child.clone();
        for (x, z) in [(6, 8), (7, 8), (6, 9), (7, 9)] {
            lowered.set(LocalPos::new_unchecked(x, top_y, z), AIR);
        }
        let cell = LocalPos::new_unchecked(3, top_y / 2, 4);
        assert_eq!(tops.get(3, 4), 2);
        let changed = downsample_region(
            &lowered,
            lod0,
            &mut parent,
            &mut tops,
            0,
            CellBox {
                min: [6, top_y, 8],
                max: [8, top_y + 1, 10],
            },
        );
        assert_eq!(changed, Some(CellBox::single(cell)));
        assert_eq!(tops.get(3, 4), 1);
        assert_eq!(parent.get(cell), DIRT);
    }

    #[test]
    fn parent_goes_back_to_uniform_after_an_airing_edit() {
        // A single-cell child inside uniform air: its parent cell fills
        // (1 of 8), and removing the block again compacts the parent and
        // its tops back to uniform air.
        let child = Cube::from_fn(|p| {
            if (p.x, p.y, p.z) == (0, 0, 0) {
                STONE
            } else {
                AIR
            }
        });
        let lod0 = &LodTops::LOD0;
        let mut parent = Cube::filled(AIR);
        let mut tops = LodTops::empty(1);
        let origin = Some(CellBox::single(LocalPos::new_unchecked(0, 0, 0)));
        assert_eq!(
            downsample_octant(&child, lod0, &mut parent, &mut tops, 0),
            origin
        );
        assert_eq!(parent.uniform(), None);
        assert_eq!(tops.uniform(), None);
        // A second block in the same group raises the cell's top only.
        let mut two = child.clone();
        two.set(LocalPos::new_unchecked(1, 1, 0), STONE);
        assert_eq!(
            downsample_octant(&two, lod0, &mut parent, &mut tops, 0),
            origin
        );
        assert_eq!(tops.get(0, 0), 2);
        assert_eq!(
            downsample_octant(&Cube::filled(AIR), lod0, &mut parent, &mut tops, 0),
            origin
        );
        assert_eq!(parent.uniform(), Some(AIR));
        assert_eq!(tops.uniform(), Some(0));
        parent.check_invariants();
    }

    #[test]
    fn copy_octant_installs_a_precomputed_octant() {
        // A mixed child downsamples into a scratch cube's octant off-thread;
        // copy_octant installs it into the real parent, changed cells only.
        let child = Cube::from_fn(|p| {
            let y = p.y as usize;
            if y < 8 {
                if y == 7 { GRASS } else { DIRT }
            } else {
                AIR
            }
        });
        let (scratch, scratch_tops, _) = down0(&child, 3);
        let mut parent = Cube::filled(AIR);
        let mut tops = LodTops::empty(1);
        let changed = copy_octant(&scratch, &scratch_tops, &mut parent, &mut tops, 3)
            .expect("air gains content");
        let half = CUBE_SIZE as usize / 2;
        let [ox, oy, oz] = octant_offset(3).map(|v| v as usize * half);
        assert_eq!(changed.min, [ox as u8, oy as u8, oz as u8]);
        // The child is filled below y = 8, so only the octant's bottom
        // four parent rows gain content; the air above does not change.
        assert_eq!(
            changed.max,
            [(ox + half) as u8, (oy + 4) as u8, (oz + half) as u8]
        );
        // Outside the octant stays air; inside equals the scratch octant.
        for y in 0..CUBE_SIZE as usize {
            for z in 0..CUBE_SIZE as usize {
                for x in 0..CUBE_SIZE as usize {
                    let pos = LocalPos::new_unchecked(x as u8, y as u8, z as u8);
                    let inside = (ox..ox + half).contains(&x)
                        && (oy..oy + half).contains(&y)
                        && (oz..oz + half).contains(&z);
                    assert_eq!(parent.get(pos), if inside { scratch.get(pos) } else { AIR });
                }
            }
        }
        assert_eq!(tops, scratch_tops);
        // Re-copying the same octant reports no change.
        assert_eq!(
            copy_octant(&scratch, &scratch_tops, &mut parent, &mut tops, 3),
            None
        );
        parent.check_invariants();

        // A tops-only difference is a change at the column's top cell.
        let mut taller = scratch_tops.clone();
        taller.set_half(0, ox as u8, oz as u8, 1);
        let top_cell = LocalPos::new_unchecked(ox as u8, (oy + 3) as u8, oz as u8);
        assert_eq!(
            copy_octant(&scratch, &taller, &mut parent, &mut tops, 3),
            Some(CellBox::single(top_cell))
        );

        // A uniform source fills its octant in bulk.
        let stone_src = Cube::filled(STONE);
        let full = LodTops::full(1);
        let mut air_parent = Cube::filled(AIR);
        assert!(
            copy_octant(
                &stone_src,
                &full,
                &mut air_parent,
                &mut LodTops::empty(1),
                0
            )
            .is_some()
        );
        assert!(
            copy_octant(
                &stone_src,
                &full,
                &mut Cube::filled(STONE),
                &mut full.clone(),
                0
            )
            .is_none()
        );
    }

    #[test]
    fn tops_codec_round_trips_and_rejects_garbage() {
        let mut mixed = LodTops::full(3);
        mixed.set_half(1, 4, 5, 3);
        mixed.set_half(0, 31 % CUBE_SIZE as u8, 0, 0);
        for tops in [LodTops::empty(1), LodTops::full(6), mixed] {
            let mut bytes = Vec::new();
            tops.encode(&mut bytes);
            let (back, used) = LodTops::decode(&bytes).unwrap();
            assert_eq!(back, tops);
            assert_eq!(used, bytes.len());
            for cut in 0..bytes.len() {
                assert!(LodTops::decode(&bytes[..cut]).is_err(), "cut at {cut}");
            }
        }
        // A height beyond the level and a bad level are rejected.
        assert!(LodTops::decode(&[TOPS_CODEC_VERSION, 1, 0, 3]).is_err());
        assert!(LodTops::decode(&[TOPS_CODEC_VERSION, 0, 0, 1]).is_err());
        assert!(LodTops::decode(&[TOPS_CODEC_VERSION + 1, 1, 0, 1]).is_err());
        assert!(LodTops::decode(&[TOPS_CODEC_VERSION, 1, 2, 1]).is_err());
    }

    // ------------------------------------------------------------------
    // Brute force: tops against LOD0 blocks, up to level 3

    /// A seeded block world spanning one level-`MAX` cube from the origin:
    /// random ground heights crossing a LOD0 cube border, overhangs,
    /// floating blocks and holes, plus uniform cubes above and below.
    struct World {
        side: usize,
        filled: Vec<bool>,
    }

    impl World {
        const MAX: u8 = 3;

        fn new(seed: u64) -> World {
            let size = CUBE_SIZE as usize;
            let side = size << Self::MAX;
            let mut rng = XorShift64(seed);
            let mut filled = vec![false; side * side * side];
            // Only the first `mixed` cube columns carry features; the rest
            // stay air above a flat floor, so most LOD0 cubes are uniform.
            let mixed = (4 * size).min(side);
            let at = |x: usize, y: usize, z: usize| (y * side + z) * side + x;
            for z in 0..side {
                for x in 0..side {
                    let ground = if x < mixed && z < mixed {
                        size / 2 + (rng.next() % size as u64) as usize
                    } else {
                        size - 1
                    };
                    for y in 0..=ground {
                        filled[at(x, y, z)] = true;
                    }
                }
            }
            // Holes, overhangs and floating blocks in the mixed area.
            for _ in 0..4000 {
                let x = (rng.next() % mixed as u64) as usize;
                let z = (rng.next() % mixed as u64) as usize;
                let y = size / 4 + (rng.next() % (2 * size) as u64) as usize;
                filled[at(x, y, z)] = !rng.next().is_multiple_of(3);
            }
            World { side, filled }
        }

        fn get(&self, x: usize, y: usize, z: usize) -> bool {
            self.filled[(y * self.side + z) * self.side + x]
        }

        fn lod0(&self, cx: usize, cy: usize, cz: usize) -> Cube {
            let size = CUBE_SIZE as usize;
            Cube::from_fn(|p| {
                let (x, y, z) = (
                    cx * size + p.x as usize,
                    cy * size + p.y as usize,
                    cz * size + p.z as usize,
                );
                if self.get(x, y, z) { STONE } else { AIR }
            })
        }

        /// The level-`level` cube at `pos` built bottom-up through
        /// `downsample_octant`.
        fn build(&self, level: u8, pos: [usize; 3]) -> (Cube, LodTops) {
            if level == 0 {
                return (self.lod0(pos[0], pos[1], pos[2]), LodTops::LOD0);
            }
            let mut cube = Cube::filled(AIR);
            let mut tops = LodTops::empty(level);
            for octant in 0..8u8 {
                let [dx, dy, dz] = octant_offset(octant).map(usize::from);
                let child_pos = [2 * pos[0] + dx, 2 * pos[1] + dy, 2 * pos[2] + dz];
                let (child, child_tops) = self.build(level - 1, child_pos);
                downsample_octant(&child, &child_tops, &mut cube, &mut tops, octant);
            }
            (cube, tops)
        }

        /// Checks a level-`level` cube at `pos` against the blocks: every
        /// cell is filled iff a block of its footprint is, and every
        /// half-column's top is the highest filled block of that half's
        /// footprint, measured from its cell's bottom.
        fn check(&self, level: u8, pos: [usize; 3], cube: &Cube, tops: &LodTops) {
            let size = CUBE_SIZE as usize;
            let cell = 1usize << level;
            let span = size * cell;
            let [bx, by, bz] = pos.map(|p| p * span);
            for z in 0..size {
                for x in 0..size {
                    // Highest filled block per cell row of the column.
                    let mut row_top = vec![None::<usize>; size];
                    for (y, top) in row_top.iter_mut().enumerate() {
                        for fy in (0..cell).rev() {
                            let hit = (0..cell).any(|fz| {
                                (0..cell).any(|fx| {
                                    self.get(
                                        bx + x * cell + fx,
                                        by + y * cell + fy,
                                        bz + z * cell + fz,
                                    )
                                })
                            });
                            if hit {
                                *top = Some(fy + 1);
                                break;
                            }
                        }
                        let got = cube.get(LocalPos::new_unchecked(x as u8, y as u8, z as u8));
                        assert_eq!(
                            got != AIR,
                            top.is_some(),
                            "L{level} {pos:?} cell ({x},{y},{z}) occupancy"
                        );
                    }
                    for half in 0..2usize {
                        let rows = half * size / 2..(half + 1) * size / 2;
                        let expected = rows.rev().find_map(|y| row_top[y]).map_or(0, |h| h as u8);
                        assert_eq!(
                            tops.half_top(half as u8, x as u8, z as u8),
                            expected,
                            "L{level} {pos:?} column ({x},{z}) half {half}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tops_match_a_brute_force_reference_up_to_level_3() {
        let world = World::new(0x7095_2b14_d00d_0001);
        for level in 1..=World::MAX {
            let cubes = 1usize << (World::MAX - level);
            for cz in 0..cubes.min(2) {
                for cy in 0..cubes.min(2) {
                    for cx in 0..cubes.min(2) {
                        let pos = [cx, cy, cz];
                        let (cube, tops) = world.build(level, pos);
                        assert!(tops.consistent_with(&cube));
                        world.check(level, pos, &cube, &tops);
                    }
                }
            }
        }
    }

    /// Incremental edits through `downsample_region` — each level fed the
    /// box the level below returned — equal a rebuild from scratch, tops
    /// included, and both match the brute-force reference.
    #[test]
    fn region_edits_keep_tops_exact_up_to_level_3() {
        let size = CUBE_SIZE as usize;
        let mut world = World::new(0x7095_2b14_d00d_0002);
        // The pyramid over the level-3 cube at the origin, kept per level.
        let mut levels: Vec<std::collections::HashMap<[usize; 3], (Cube, LodTops)>> =
            vec![std::collections::HashMap::new(); World::MAX as usize + 1];
        for level in 0..=World::MAX {
            let cubes = 1usize << (World::MAX - level);
            for cz in 0..cubes {
                for cy in 0..cubes {
                    for cx in 0..cubes {
                        let pos = [cx, cy, cz];
                        let built = world.build(level, pos);
                        levels[level as usize].insert(pos, built);
                    }
                }
            }
        }
        let mut rng = XorShift64(0x00ed_17ed_17ed);
        let mixed = 4 * size;
        for _ in 0..300 {
            let x = (rng.next() % mixed as u64) as usize;
            let z = (rng.next() % mixed as u64) as usize;
            let y = (rng.next() % (2 * size) as u64) as usize;
            let state = rng.next().is_multiple_of(2);
            world.filled[(y * world.side + z) * world.side + x] = state;
            let mut pos = [x / size, y / size, z / size];
            let local =
                LocalPos::new_unchecked((x % size) as u8, (y % size) as u8, (z % size) as u8);
            let lod0 = world.lod0(pos[0], pos[1], pos[2]);
            levels[0].insert(pos, (lod0, LodTops::LOD0));
            let mut region = Some(CellBox::single(local));
            for level in 1..=World::MAX as usize {
                let Some(boxed) = region else { break };
                let parent_pos = pos.map(|p| p / 2);
                let octant = ((pos[0] % 2) + 2 * (pos[2] % 2) + 4 * (pos[1] % 2)) as u8;
                let (child, child_tops) = levels[level - 1][&pos].clone();
                let (parent, parent_tops) = levels[level].get_mut(&parent_pos).unwrap();
                region = downsample_region(&child, &child_tops, parent, parent_tops, octant, boxed);
                pos = parent_pos;
            }
        }
        for level in 1..=World::MAX {
            for (&pos, (cube, tops)) in &levels[level as usize] {
                let (fresh, fresh_tops) = world.build(level, pos);
                assert_eq!(cube, &fresh, "L{level} {pos:?} cells");
                assert_eq!(tops, &fresh_tops, "L{level} {pos:?} tops");
                if pos.iter().all(|&p| p < 2) {
                    world.check(level, pos, cube, tops);
                }
            }
        }
    }
}
