//! CPU-side suballocator for one GPU face page: pure logic, no wgpu. A page
//! is `capacity` units where a unit is one 8-byte `PackedFace`.
//! `alloc` is first-fit over an ordered free list; `free` coalesces with
//! both neighbors, so the list stays sorted, disjoint and non-adjacent.

use std::ops::Range;

/// Free-list suballocator over one fixed-capacity page.
#[derive(Debug, Clone)]
pub struct Suballocator {
    capacity: u32,
    /// Sorted, disjoint, never-adjacent free ranges.
    free: Vec<Range<u32>>,
}

impl Suballocator {
    /// A page with everything free. Capacity must be nonzero (a page holds
    /// at least one worst-case cube mesh).
    pub fn new(capacity: u32) -> Suballocator {
        debug_assert!(capacity > 0, "a page must hold at least one face");
        Suballocator {
            capacity,
            // One Range<u32> element: the whole page free, not a Vec of u32.
            #[allow(clippy::single_range_in_vec_init)]
            free: vec![0..capacity],
        }
    }

    /// First-fit: the earliest free range that fits. `alloc(0)` is never
    /// called — empty meshes allocate nothing — so it is a debug assert.
    pub fn alloc(&mut self, len: u32) -> Option<Range<u32>> {
        debug_assert!(
            len > 0,
            "alloc(0) is a caller bug; empty meshes allocate nothing"
        );
        for (i, range) in self.free.iter().enumerate() {
            if range.len() >= len as usize {
                let start = range.start;
                if range.len() == len as usize {
                    self.free.remove(i);
                } else {
                    self.free[i] = start + len..range.end;
                }
                return Some(start..start + len);
            }
        }
        None
    }

    /// Returns a range previously handed out by [`alloc`](Self::alloc) and
    /// not yet freed. Double-freeing, or freeing a range that overlaps the
    /// free list at all, is a debug assert: it is a caller bug.
    pub fn free(&mut self, range: Range<u32>) {
        debug_assert!(
            range.start < range.end,
            "an empty range was never allocated"
        );
        debug_assert!(range.end <= self.capacity, "freed range out of bounds");
        let mut pos = self.free.partition_point(|r| r.end <= range.start);
        debug_assert!(
            pos == self.free.len() || self.free[pos].start >= range.end,
            "double free or freeing an unallocated range: {range:?} overlaps {:?}",
            self.free[pos]
        );
        let (mut start, mut end) = (range.start, range.end);
        if pos > 0 && self.free[pos - 1].end == start {
            start = self.free[pos - 1].start;
            self.free.remove(pos - 1);
            pos -= 1;
        }
        if pos < self.free.len() && self.free[pos].start == end {
            end = self.free[pos].end;
            self.free.remove(pos);
        }
        self.free.insert(pos, start..end);
    }

    /// Units currently handed out.
    pub fn used(&self) -> u32 {
        self.capacity - self.free.iter().map(|r| r.len() as u32).sum::<u32>()
    }

    /// The largest contiguous free block, 0 when the page is full.
    pub fn largest_free(&self) -> u32 {
        self.free.iter().map(|r| r.len() as u32).max().unwrap_or(0)
    }

    /// The free list itself, for tests and stats.
    pub fn free_ranges(&self) -> &[Range<u32>] {
        &self.free
    }

    /// Checks the invariants: free ranges sorted, disjoint, non-adjacent
    /// (always coalesced) and in bounds; used plus free equals capacity.
    /// Used by tests after every operation.
    pub fn check_invariants(&self) {
        let mut free_total = 0u64;
        for pair in self.free.windows(2) {
            assert!(
                pair[0].end < pair[1].start,
                "free list must be sorted, disjoint and coalesced: {:?}",
                self.free
            );
        }
        for range in &self.free {
            assert!(range.start < range.end, "empty free range {range:?}");
            assert!(
                range.end <= self.capacity,
                "free range {range:?} out of bounds"
            );
            free_total += range.len() as u64;
        }
        assert_eq!(
            free_total + u64::from(self.used()),
            u64::from(self.capacity),
            "used + free must equal capacity"
        );
    }
}

/// Local xorshift, so tests and benches never need the `rand` crate.
#[cfg(any(test, feature = "testing"))]
pub(crate) struct XorShift64(pub u64);

#[cfg(any(test, feature = "testing"))]
impl XorShift64 {
    pub(crate) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// The seeded 10,000-operation churn shared by the unit test (with
/// invariant checks against a bitmap model) and the `suballoc/churn_10k`
/// bench (without checks). Every fourth op frees a random live range, the
/// rest allocate a random length of 1..=128 units.
#[cfg(any(test, feature = "testing"))]
pub fn churn(capacity: u32, seed: u64, ops: usize, check: bool) {
    let mut rng = XorShift64(seed);
    let mut sub = Suballocator::new(capacity);
    let mut live: Vec<Range<u32>> = Vec::new();
    for _ in 0..ops {
        if !live.is_empty() && rng.next().is_multiple_of(4) {
            let i = (rng.next() as usize) % live.len();
            sub.free(live.remove(i));
        } else {
            let len = 1 + (rng.next() % 128) as u32;
            if let Some(range) = sub.alloc(len) {
                live.push(range);
            }
        }
        if check {
            sub.check_invariants();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts the free list as (start, end) pairs — plain tuples keep the
    /// `single_range_in_vec_init` lint away from `&[range]` literals.
    fn expect_free(sub: &Suballocator, expected: &[(u32, u32)]) {
        let got: Vec<(u32, u32)> = sub.free_ranges().iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(got, expected.to_vec());
    }

    #[test]
    fn fill_exactly_then_free_returns_one_block() {
        let mut sub = Suballocator::new(10);
        let a = sub.alloc(10).expect("whole page fits");
        assert_eq!(a, 0..10);
        assert_eq!(sub.largest_free(), 0);
        assert!(sub.alloc(1).is_none());
        sub.free(a);
        expect_free(&sub, &[(0, 10)]);
        sub.check_invariants();
    }

    #[test]
    fn freeing_the_middle_leaves_two_coalesced_holes() {
        let mut sub = Suballocator::new(12);
        let a = sub.alloc(4).unwrap();
        let b = sub.alloc(4).unwrap();
        let c = sub.alloc(4).unwrap();
        assert_eq!((a.clone(), b.clone(), c.clone()), (0..4, 4..8, 8..12));
        sub.free(b);
        expect_free(&sub, &[(4, 8)]);
        sub.check_invariants();
    }

    #[test]
    fn free_coalesces_left_right_and_both() {
        // Left: freeing a range adjacent to a later hole merges right.
        let mut sub = Suballocator::new(12);
        let a = sub.alloc(4).unwrap();
        let b = sub.alloc(4).unwrap();
        let c = sub.alloc(4).unwrap();
        sub.free(c); // hole at 8..12
        sub.free(b); // left-adjacent: merges into it
        expect_free(&sub, &[(4, 12)]);
        sub.free(a); // left-adjacent again: the whole page
        expect_free(&sub, &[(0, 12)]);

        // Right: freeing a range adjacent to an earlier hole merges left.
        let mut sub = Suballocator::new(8);
        let a = sub.alloc(4).unwrap();
        let b = sub.alloc(4).unwrap();
        sub.free(a); // hole at 0..4
        sub.free(b); // right-adjacent: merges into it
        expect_free(&sub, &[(0, 8)]);
        sub.check_invariants();
    }

    #[test]
    fn first_fit_picks_the_earliest_fitting_hole() {
        let mut sub = Suballocator::new(20);
        let a = sub.alloc(5).unwrap();
        let b = sub.alloc(5).unwrap();
        let c = sub.alloc(5).unwrap();
        let d = sub.alloc(5).unwrap();
        sub.free(b); // hole at 5..10
        sub.free(d); // hole at 15..20
        // A 3-unit request fits the first hole, not the later one.
        assert_eq!(sub.alloc(3), Some(5..8));
        // A request too big for the 2-unit remainder of the first hole but
        // fine for the second takes the second hole.
        assert_eq!(sub.alloc(4), Some(15..19));
        assert_eq!(a, 0..5);
        assert_eq!(c, 10..15);
        sub.check_invariants();
    }

    #[test]
    fn fragmented_space_fails_larger_requests() {
        let mut sub = Suballocator::new(30);
        for i in 0..6 {
            sub.alloc(5).unwrap_or_else(|| panic!("alloc {i}"));
        }
        // Free every other block: 15 free units in 3-unit holes.
        sub.free(0..5);
        sub.free(10..15);
        sub.free(20..25);
        assert_eq!(sub.largest_free(), 5);
        assert!(
            sub.alloc(6).is_none(),
            "fragmented space must not satisfy 6"
        );
        assert_eq!(sub.alloc(5), Some(0..5), "exact hole still works");
        sub.check_invariants();
    }

    #[test]
    fn used_tracks_allocations() {
        let mut sub = Suballocator::new(100);
        assert_eq!(sub.used(), 0);
        let a = sub.alloc(30).unwrap();
        assert_eq!(sub.used(), 30);
        sub.free(a);
        assert_eq!(sub.used(), 0);
    }

    // debug_assert-based tests only exist where debug assertions fire
    // (`cargo bench -- --test` compiles the lib tests without them).
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "double free")]
    fn double_free_is_a_debug_assert() {
        let mut sub = Suballocator::new(10);
        let a = sub.alloc(4).unwrap();
        sub.free(a.clone());
        sub.free(a); // debug builds must panic
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "double free")]
    fn freeing_unallocated_range_is_a_debug_assert() {
        let mut sub = Suballocator::new(10);
        let _ = sub.alloc(4).unwrap();
        sub.free(5..8); // inside the free block: unallocated
    }

    #[test]
    fn churn_10k_passes_invariants() {
        churn(4096, 0x0123_4567_89ab_cdef, 10_000, true);
    }

    #[test]
    fn random_churn_matches_a_bitmap_model() {
        // A simple bitmap is the model; the suballocator must always agree.
        let capacity = 1024usize;
        let mut rng = XorShift64(0xfeed_face_cafe_babe);
        let mut sub = Suballocator::new(capacity as u32);
        let mut bitmap = vec![true; capacity];
        let mut live: Vec<Range<u32>> = Vec::new();

        let model_alloc = |bitmap: &mut Vec<bool>, len: usize| -> Option<usize> {
            let mut run = 0usize;
            for (i, &is_free) in bitmap.iter().enumerate() {
                run = if is_free { run + 1 } else { 0 };
                if run == len {
                    bitmap[i + 1 - len..=i].fill(false);
                    return Some(i + 1 - len);
                }
            }
            None
        };
        let model_free = |bitmap: &mut Vec<bool>, range: &Range<u32>| {
            bitmap[range.start as usize..range.end as usize].fill(true);
        };

        for op in 0..10_000u64 {
            if !live.is_empty() && rng.next().is_multiple_of(4) {
                let i = (rng.next() as usize) % live.len();
                let range = live.remove(i);
                model_free(&mut bitmap, &range);
                sub.free(range);
            } else {
                let len = 1 + (rng.next() % 128) as usize;
                match model_alloc(&mut bitmap, len) {
                    Some(start) => live.push(sub.alloc(len as u32).unwrap_or_else(|| {
                        panic!(
                            "op {op}: model allocated {start}..{} but suballocator failed",
                            start + len
                        )
                    })),
                    None => assert!(
                        sub.alloc(len as u32).is_none(),
                        "op {op}: suballocator allocated but model did not"
                    ),
                }
                // The models pick the same block.
                if let Some(range) = live.last() {
                    assert_eq!(
                        bitmap[range.start as usize..range.end as usize],
                        vec![false; range.len()],
                        "op {op}: range {range:?} not marked used in the model"
                    );
                }
            }
            sub.check_invariants();
            // The free lists agree exactly.
            let model_free_list: Vec<Range<u32>> = {
                let mut ranges = Vec::new();
                let mut run: Option<u32> = None;
                for (i, &is_free) in bitmap.iter().enumerate() {
                    if is_free {
                        run = Some(run.unwrap_or(i as u32));
                    } else if let Some(start) = run.take() {
                        ranges.push(start..i as u32);
                    }
                }
                if let Some(start) = run {
                    ranges.push(start..capacity as u32);
                }
                ranges
            };
            assert_eq!(sub.free_ranges(), model_free_list.as_slice(), "op {op}");
        }
    }
}
