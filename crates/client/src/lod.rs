//! Client-side LOD state (task 2.6): the cube key every level shares and
//! the coverage rule that decides which level draws which volume. Pure
//! logic, no GPU — the renderer asks [`Coverage`] what to draw, the world
//! stores the cubes themselves keyed by [`CubeKey`].

use std::collections::{HashMap, HashSet, VecDeque};

use jbm_core::coords::CubePos;
use jbm_core::lod::{octant_offset, parent};

/// The highest level [`Coverage`] tracks. Ancestor chains stop here: a
/// ready cube at the cap draws under the "no ready ancestor" clause, and a
/// level above it that arrives later still finds it (its own downward walk
/// reaches the entry). Far above any `lod_levels` a server configures.
pub const MAX_COVERED_LEVEL: u8 = 16;

/// A cube at a LOD level. Level 0 is the streamed world itself ("LOD0");
/// a level-*n* cube at `pos` covers blocks `pos * CUBE_SIZE * 2^n` onwards
/// and each of its cells is `2^n` blocks per side. One key type names
/// meshes, mesh jobs, GPU residency and coverage entries at every level.
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
    /// (`dx + 2·dz + 4·dy`, as in `jbm_core::lod::parent`).
    pub fn parent(&self) -> (CubeKey, u8) {
        debug_assert!(self.level < MAX_COVERED_LEVEL, "no parent above the cap");
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
        u64::from(jbm_core::coords::CUBE_SIZE as u32) << self.level
    }
}

/// One tracked cube: whether it is ready, whether it draws, and which of
/// its octants a ready child replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct CoverEntry {
    ready: bool,
    drawn: bool,
    /// Bit *o* set once the level-(*n*−1) child at octant *o* is ready.
    /// Meaningful for drawn cubes at level ≥ 1; level 0 keeps it zero.
    mask: u8,
}

/// The coverage rule: every volume is drawn by exactly one cube — the
/// finest ready one along its chain of ancestors.
///
/// - `drawn(c)` holds iff `ready(c)`, and either `drawn(parent(c))`, or no
///   ancestor of *c* is ready.
/// - A drawn cube at level ≥ 1 **masks** octant *o* when the child at *o*
///   is ready (a ready child of a drawn cube is itself drawn, so the child
///   covers that volume instead). A drawn cube draws exactly its unmasked
///   octants; a fully masked one draws nothing.
///
/// Why that is exactly one per volume: along a point's chain of cubes, the
/// topmost ready one *T* is drawn (no ancestor of *T* is ready). Drawing
/// passes down while the chain stays ready and stops at the first cube
/// that is not ready, whose parent draws the point — every step hands the
/// point to exactly one child, so no cube is skipped and none is doubled.
/// Points contained in no ready cube are drawn by nothing.
///
/// Readiness comes from the caller: a cube is ready when it is loaded and
/// its current mesh is uploaded (an empty mesh counts; a re-mesh in flight
/// keeps the old mesh, so the cube stays ready). [`Coverage::set_ready`]
/// updates incrementally: a change of `(n, pos)` touches its parent's
/// mask, its own state and its descendants, and nothing else — a frame
/// with no readiness change does no work at all (assert `work()`).
///
/// The transient this rule allows is accepted: a coarse cube briefly
/// covers a finer one when the cube between them arrives late, typically
/// for a moment at startup.
#[derive(Debug, Default)]
pub struct Coverage {
    /// Every ready cube plus the ancestors connecting them (placeholders
    /// with `ready = false`), so the tree is walkable both ways.
    entries: HashMap<CubeKey, CoverEntry>,
    /// Draw-state changes since the last [`Coverage::take_changed`], oldest
    /// first: (key, drawn, mask) for ready cubes.
    pending: VecDeque<(CubeKey, bool, u8)>,
    /// Entries touched by updates since the last [`Coverage::reset_work`].
    work: u64,
}

impl Coverage {
    /// Marks `(level, pos)` ready (or not). `ready = true` for an already
    /// ready cube is a cheap no-op — a re-mesh does not lose readiness.
    pub fn set_ready(&mut self, key: CubeKey, ready: bool) {
        if ready {
            self.make_chain(key);
        }
        match self.entries.get(&key) {
            Some(entry) if entry.ready == ready => return,
            None if !ready => return,
            _ => {}
        }
        self.work += 1;

        self.entries.entry(key).or_default().ready = ready;

        // The parent's mask: this cube's readiness replaces (or frees) one
        // of its octants.
        if key.level < MAX_COVERED_LEVEL {
            let (parent_key, octant) = key.parent();
            let mask_changed = self.entries.get_mut(&parent_key).is_some_and(|parent| {
                let mask = if ready {
                    parent.mask | 1 << octant
                } else {
                    parent.mask & !(1 << octant)
                };
                let changed = parent.mask != mask;
                parent.mask = mask;
                changed
            });
            if mask_changed
                && self
                    .entries
                    .get(&parent_key)
                    .is_some_and(|parent| parent.ready)
            {
                self.note(parent_key);
            }
        }

        // Own drawn state, then the descendants: below a ready key the "no
        // ready ancestor" clause is dead, below a not-ready one it revives.
        let (drawn_parent, ready_above) = self.context(key);
        let new_drawn = ready && (drawn_parent || !ready_above);
        let drawn_changed = self.entries.get_mut(&key).is_some_and(|entry| {
            let changed = entry.drawn != new_drawn;
            entry.drawn = new_drawn;
            changed
        });
        if drawn_changed && ready {
            self.note(key);
        }
        if key.level > 0 {
            let child_ready_above = if ready { true } else { ready_above };
            for octant in 0..8u8 {
                self.refresh(key.child(octant), new_drawn, child_ready_above);
            }
        }
    }

    /// `(drawn(parent), any ancestor of `key` ready)` — the two clauses of
    /// the drawn rule, read off the ancestor chain.
    fn context(&self, key: CubeKey) -> (bool, bool) {
        if key.level >= MAX_COVERED_LEVEL {
            return (false, false);
        }
        let (parent_key, _) = key.parent();
        let drawn_parent = self.entries.get(&parent_key).is_some_and(|e| e.drawn);
        let mut ready_above = false;
        let mut k = parent_key;
        while let Some(entry) = self.entries.get(&k) {
            ready_above |= entry.ready;
            if k.level >= MAX_COVERED_LEVEL {
                break;
            }
            let (next, _) = k.parent();
            k = next;
        }
        (drawn_parent, ready_above)
    }
    /// Recomputes one subtree after a readiness change above it:
    /// `drawn = ready && (parent_drawn || no ready ancestor)`, recursively.
    fn refresh(&mut self, key: CubeKey, parent_drawn: bool, ready_above: bool) {
        let Some(entry) = self.entries.get_mut(&key) else {
            return; // no entry: nothing ready below either
        };
        self.work += 1;
        let new_drawn = entry.ready && (parent_drawn || !ready_above);
        let child_ready_above = ready_above || entry.ready;
        let state_changed = entry.drawn != new_drawn;
        entry.drawn = new_drawn;
        if state_changed && entry.ready {
            self.note(key);
        }
        if key.level > 0 {
            for octant in 0..8u8 {
                self.refresh(key.child(octant), new_drawn, child_ready_above);
            }
        }
    }

    /// Creates not-ready placeholder entries for the ancestors of `key` up
    /// to the first existing one (entries above an existing entry always
    /// exist — they were chained when it was), so the tree stays walkable.
    fn make_chain(&mut self, key: CubeKey) {
        let mut k = key;
        while k.level < MAX_COVERED_LEVEL {
            let (parent_key, _) = k.parent();
            if self.entries.contains_key(&parent_key) {
                break;
            }
            self.entries.insert(parent_key, CoverEntry::default());
            k = parent_key;
        }
    }

    fn note(&mut self, key: CubeKey) {
        let entry = self.entries[&key];
        self.pending.push_back((key, entry.drawn, entry.mask));
    }

    /// The state of a ready cube: `(drawn, octant mask)`. `None` when the
    /// cube is not tracked as ready.
    pub fn state(&self, key: CubeKey) -> Option<(bool, u8)> {
        self.entries
            .get(&key)
            .filter(|entry| entry.ready)
            .map(|entry| (entry.drawn, entry.mask))
    }

    /// Every ready cube's `(key, drawn, mask)`, for the renderer, the
    /// overlay and tests. Iteration order is unspecified.
    pub fn iter_states(&self) -> impl Iterator<Item = (CubeKey, bool, u8)> + '_ {
        self.entries
            .iter()
            .filter(|(_, entry)| entry.ready)
            .map(|(key, entry)| (*key, entry.drawn, entry.mask))
    }

    /// Draw-state changes since the last call, oldest first. The caller
    /// rewrites renderer metadata for exactly these.
    pub fn take_changed(&mut self) -> Vec<(CubeKey, bool, u8)> {
        self.pending.drain(..).collect()
    }

    /// Entries touched by readiness changes since the last reset (the
    /// "no change, no work" hook).
    pub fn work(&self) -> u64 {
        self.work
    }

    pub fn reset_work(&mut self) {
        self.work = 0;
    }

    /// Drops everything (test hook).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.pending.clear();
        self.work = 0;
    }

    /// The reference computation, straight from the definition, for tests:
    /// for every ready cube, `drawn` and its octant mask.
    pub fn from_scratch(ready: &HashSet<CubeKey>) -> HashMap<CubeKey, (bool, u8)> {
        // drawn(c) = ready(c) && (drawn(parent(c)) || no ancestor of c is
        // ready), memoized up the chain. The recursion terminates: each
        // step either finds a not-ready ancestor (whose own drawn is false
        // or comes from the no-ancestor clause above it) or climbs.
        fn drawn(ready: &HashSet<CubeKey>, memo: &mut HashMap<CubeKey, bool>, c: CubeKey) -> bool {
            if !ready.contains(&c) {
                return false;
            }
            if let Some(&value) = memo.get(&c) {
                return value;
            }
            // Any ready ancestor? Walk up; a missing entry means nothing
            // above is tracked.
            let mut ready_above = false;
            let mut k = c;
            while k.level < MAX_COVERED_LEVEL {
                let (parent_key, _) = k.parent();
                if ready.contains(&parent_key) {
                    ready_above = true;
                    break;
                }
                k = parent_key;
            }
            let value = if ready_above {
                if c.level >= MAX_COVERED_LEVEL {
                    false
                } else {
                    drawn(ready, memo, c.parent().0)
                }
            } else {
                true
            };
            memo.insert(c, value);
            value
        }

        let mut memo = HashMap::new();
        let mut out = HashMap::with_capacity(ready.len());
        for &key in ready {
            let drawn_key = drawn(ready, &mut memo, key);
            let mut mask = 0u8;
            if key.level > 0 {
                for octant in 0..8u8 {
                    if ready.contains(&key.child(octant)) {
                        mask |= 1 << octant;
                    }
                }
            }
            out.insert(key, (drawn_key, mask));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jbm_core::coords::CUBE_SIZE;

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

    fn key(level: u8, x: i32, y: i32, z: i32) -> CubeKey {
        CubeKey {
            level,
            pos: CubePos { x, y, z },
        }
    }

    #[test]
    fn parent_child_round_trip_at_negative_coordinates() {
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
    fn block_size_scales_with_level() {
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

    /// A ready cube with nothing above draws; a ready ancestor takes over
    /// for its not-ready descendants' volume only when the chain is ready.
    #[test]
    fn single_cube_draws_and_masks_by_ready_children() {
        let mut coverage = Coverage::default();
        let c = key(1, -1, 2, -3);
        coverage.set_ready(c, true);
        assert_eq!(coverage.state(c), Some((true, 0)), "no ancestor: drawn");

        // One ready child masks its octant; the parent still draws the rest.
        let child = c.child(3);
        coverage.set_ready(child, true);
        let child_state = coverage.state(child).expect("child tracked");
        let (drawn, mask) = coverage.state(c).expect("parent tracked");
        assert!(drawn, "the parent has no ready ancestor, so it draws");
        assert_eq!(mask, 1 << 3, "octant 3 is masked by its ready child");
        assert!(child_state.0, "a ready child of a drawn cube is drawn");

        // All eight octants ready: the parent is fully masked.
        for octant in 0..8u8 {
            coverage.set_ready(c.child(octant), true);
        }
        assert_eq!(
            coverage.state(c),
            Some((true, 0xFF)),
            "drawn, but every octant is replaced"
        );
    }

    #[test]
    fn a_coarse_cube_covers_a_fine_one_when_the_between_level_is_missing() {
        let mut coverage = Coverage::default();
        let fine = key(0, 4, 4, 4);
        let mid = fine.parent().0; // level 1
        let coarse = mid.parent().0; // level 2

        // The fine cube draws alone.
        coverage.set_ready(fine, true);
        assert_eq!(coverage.state(fine), Some((true, 0)));

        // The coarse level arrives without the middle one: it draws and the
        // fine cube stops drawing — the accepted transient.
        coverage.set_ready(coarse, true);
        assert_eq!(coverage.state(coarse), Some((true, 0)));
        assert_eq!(coverage.state(fine), Some((false, 0)), "covered from above");

        // The middle level arrives: it draws, masks the fine cube's octant
        // of the coarse cube, and the fine cube draws again.
        coverage.set_ready(mid, true);
        assert_eq!(coverage.state(coarse), Some((true, 1 << mid.parent().1)));
        assert_eq!(coverage.state(mid), Some((true, 1 << fine.parent().1)));
        assert_eq!(coverage.state(fine), Some((true, 0)));
    }

    #[test]
    fn unloading_a_child_unmasks_its_parent() {
        let mut coverage = Coverage::default();
        let parent = key(1, 0, 0, 0);
        let child = parent.child(5);
        coverage.set_ready(parent, true);
        coverage.set_ready(child, true);
        assert_eq!(coverage.state(parent), Some((true, 1 << 5)));

        coverage.set_ready(child, false);
        assert_eq!(coverage.state(parent), Some((true, 0)));
        assert_eq!(coverage.state(child), None, "not tracked as ready");
    }

    #[test]
    fn set_ready_is_idempotent_and_does_no_work_the_second_time() {
        let mut coverage = Coverage::default();
        let k = key(2, 1, -1, 1);
        coverage.set_ready(k, true);
        coverage.take_changed();
        coverage.reset_work();
        coverage.set_ready(k, true);
        assert_eq!(coverage.work(), 0, "no readiness change, no work");
        assert!(coverage.take_changed().is_empty());
    }

    /// The property: over seeded random readiness sets on a small region
    /// with negative coordinates and levels 0..=4, every LOD0 cube is
    /// covered by exactly one drawn cube when any cube in its chain is
    /// ready, and by none otherwise. Incremental and from-scratch agree.
    #[test]
    fn every_volume_is_drawn_by_exactly_one_cube() {
        // Region: 4x2x4 LOD0 cubes at negative coordinates, nested inside
        // a single level-4 cube (x in [-8,-4): -8/8 = -1, -4/8 = -1).
        const X0: i32 = -8;
        const Y0: i32 = -4;
        const Z0: i32 = -16;
        const LEVELS: u8 = 4;
        let region: Vec<CubePos> = {
            let mut v = Vec::new();
            for y in Y0..Y0 + 2 {
                for z in Z0..Z0 + 4 {
                    for x in X0..X0 + 4 {
                        v.push(CubePos { x, y, z });
                    }
                }
            }
            v
        };
        // Every cube at levels 0..=4 that contains a region cell.
        let mut candidates: Vec<CubeKey> = Vec::new();
        for level in 0..=LEVELS {
            let mut seen = HashSet::new();
            for &pos in &region {
                let mut p = pos;
                for _ in 0..level {
                    p = parent(p).0;
                }
                if seen.insert(p) {
                    candidates.push(CubeKey { level, pos: p });
                }
            }
        }

        let mut rng = XorShift64(0x6c6f_6420_322e_3621);
        for case in 0..10_000u64 {
            // A random readiness set: each candidate ready with p ~ 1/2,
            // seeded per case so failures reproduce.
            let mut ready = HashSet::new();
            for (i, &k) in candidates.iter().enumerate() {
                let bit = rng.next() ^ case.wrapping_mul(0x9e37_79b9).wrapping_add(i as u64);
                if bit & 1 == 1 {
                    ready.insert(k);
                }
            }

            let mut coverage = Coverage::default();
            for &k in &ready {
                coverage.set_ready(k, true);
            }

            // From-scratch reference and incremental must agree cell for
            // cell (drawn and mask).
            let scratch = Coverage::from_scratch(&ready);
            for (k, state) in &scratch {
                assert_eq!(
                    coverage.state(*k),
                    Some(*state),
                    "case {case}: incremental differs at {k:?}"
                );
            }
            assert_eq!(
                coverage.iter_states().count(),
                scratch.len(),
                "case {case}: tracked ready set differs"
            );

            // The property: per region LOD0 cube, exactly one cube RENDERS
            // it when any cube in its chain is ready, none otherwise. The
            // renderer is the lowest drawn cube of the chain — every drawn
            // cube above it has this point's octant masked (its chain child
            // is ready), so it draws nothing here.
            for &pos in &region {
                let mut chain = Vec::new();
                let mut p = pos;
                for level in 0..=LEVELS {
                    chain.push(CubeKey { level, pos: p });
                    if level < LEVELS {
                        p = parent(p).0;
                    }
                }
                let renderers: Vec<CubeKey> = chain
                    .iter()
                    .enumerate()
                    .filter(|(i, k)| {
                        // Drawn, and the chain child below is not ready:
                        // this cube's octant for the point is unmasked.
                        let drawn = coverage.state(**k).is_some_and(|(d, _)| d);
                        let child_not_ready = match i.checked_sub(1).and_then(|j| chain.get(j)) {
                            Some(&below) => !ready.contains(&below),
                            None => true, // level 0: no children at all
                        };
                        drawn && child_not_ready
                    })
                    .map(|(_, k)| *k)
                    .collect();
                let any_ready = chain.iter().any(|k| ready.contains(k));
                if any_ready {
                    assert_eq!(
                        renderers.len(),
                        1,
                        "case {case}: {renderers:?} render {pos:?} (ready: {ready:?})"
                    );
                    // Every drawn cube strictly above the renderer masks
                    // the octant the point sits in.
                    let (_, renderer) = (renderers[0].level, renderers[0]);
                    for k in chain.iter().filter(|k| k.level > renderer.level) {
                        if let Some((true, mask)) = coverage.state(*k) {
                            let octant = chain
                                .iter()
                                .find(|c| c.level == k.level - 1)
                                .map(|child| child.parent().1)
                                .unwrap_or(0);
                            assert_ne!(
                                mask & (1 << octant),
                                0,
                                "case {case}: {k:?} draws over the renderer of {pos:?}"
                            );
                        }
                    }
                } else {
                    assert!(
                        renderers.is_empty(),
                        "case {case}: {renderers:?} render {pos:?} with nothing ready"
                    );
                }
            }
        }
    }

    /// Incremental equals from-scratch after every random toggle, and a
    /// no-change frame does zero work.
    #[test]
    fn random_toggles_match_from_scratch_and_idle_does_no_work() {
        const LEVELS: u8 = 4;
        // A little ring-like set: all cubes at levels 0..=3 over a 4x2x4
        // negative region.
        let mut candidates: Vec<CubeKey> = Vec::new();
        for level in 0..=LEVELS {
            let mut seen = HashSet::new();
            for y in -4..-2i32 {
                for z in -16..-12i32 {
                    for x in -8..-4i32 {
                        let mut p = CubePos { x, y, z };
                        for _ in 0..level {
                            p = parent(p).0;
                        }
                        if seen.insert(p) {
                            candidates.push(CubeKey { level, pos: p });
                        }
                    }
                }
            }
        }

        let mut rng = XorShift64(0x746f_6767_6c65_7332);
        let mut coverage = Coverage::default();
        let mut ready: HashSet<CubeKey> = HashSet::new();
        for step in 0..2_000u32 {
            let pick = candidates[(rng.next() as usize) % candidates.len()];
            let make_ready = rng.next() & 1 == 1;
            if make_ready {
                ready.insert(pick);
            } else {
                ready.remove(&pick);
            }
            coverage.set_ready(pick, make_ready);

            let scratch = Coverage::from_scratch(&ready);
            for (k, state) in &scratch {
                assert_eq!(
                    coverage.state(*k),
                    Some(*state),
                    "step {step}: {k:?} differs"
                );
            }
            // Every tracked ready cube matches, and nothing extra is ready.
            assert_eq!(coverage.iter_states().count(), scratch.len());
        }

        // No change: no work, no pending metadata updates.
        coverage.reset_work();
        coverage.take_changed();
        let before = coverage.iter_states().count();
        assert_eq!(coverage.work(), 0);
        assert!(coverage.take_changed().is_empty());
        assert_eq!(coverage.iter_states().count(), before);
    }
}
