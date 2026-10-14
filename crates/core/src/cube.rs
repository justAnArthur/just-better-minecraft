//! Paletted, bit-packed storage for one 32³ cube of block states.
//!
//! Layout: a palette of distinct [`BlockStateId`]s plus palette indices
//! bit-packed into `u64` words. Entries never straddle a word:
//! `entries_per_word = 64 / bits`, padding bits are unused. Bits per entry are
//! `max(1, ceil(log2(palette_len)))`, at most 15 (a palette can never exceed
//! the 32768 cells). A uniform cube (one palette entry) has no index array at
//! all — an all-air cube costs only its palette.
//!
//! Array order is [`LocalPos::index`](crate::coords::LocalPos::index):
//! x fastest, then z, then y.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::mem::size_of;

use crate::block::BlockStateId;
use crate::coords::{CUBE_VOLUME, LocalPos};

/// Palettes at or below this size are searched linearly; larger ones get the
/// reverse map. Measured by the `cube/palette_growth_4096` bench.
const LINEAR_SCAN_LIMIT: usize = 16;

/// FxHash-style hasher: no seeding, no DoS resistance. Fine for local `u32`
/// keys that come from our own data, and much faster than SipHash.
#[derive(Clone, Copy, Default)]
struct FastHasher(u64);

const FAST_SEED: u64 = 0x517c_c1b7_2722_0a95;

impl FastHasher {
    #[inline]
    fn add(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(FAST_SEED);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.add(u64::from(byte));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }
}

#[derive(Clone, Copy, Default)]
struct FastHashBuild;

impl BuildHasher for FastHashBuild {
    type Hasher = FastHasher;

    fn build_hasher(&self) -> FastHasher {
        FastHasher(0)
    }
}

type ReverseMap = HashMap<BlockStateId, u16, FastHashBuild>;

/// Bits per entry for a palette of `palette_len` distinct states; 0 while
/// uniform. `palette_len` never exceeds `CUBE_VOLUME`, so this stays ≤ 15.
#[inline]
fn bits_for(palette_len: usize) -> u32 {
    if palette_len <= 1 {
        0
    } else {
        usize::BITS - (palette_len - 1).leading_zeros()
    }
}

#[inline]
fn entries_per_word(bits: u32) -> usize {
    64 / bits as usize
}

#[inline]
fn word_count(bits: u32) -> usize {
    CUBE_VOLUME.div_ceil(entries_per_word(bits))
}

/// One 32³ cube of block states, palette-compressed.
///
/// Invariants (checked by [`Cube::check_invariants`] in tests):
/// - palette entries are unique and at most `CUBE_VOLUME` of them;
/// - the index array is empty exactly when the palette has one entry;
/// - every index in the array is less than `palette.len()`.
#[derive(Clone)]
pub struct Cube {
    palette: Vec<BlockStateId>,
    /// `state -> palette index`, maintained only while
    /// `palette.len() > LINEAR_SCAN_LIMIT`.
    reverse: ReverseMap,
    /// Bit-packed palette indices, `word_count(bits)` words; empty iff uniform.
    data: Vec<u64>,
    /// Bits per entry; 0 while uniform.
    bits: u32,
}

impl Cube {
    /// A cube where every block is `state`. Costs only its palette entry.
    pub fn filled(state: BlockStateId) -> Self {
        Self {
            palette: vec![state],
            reverse: ReverseMap::default(),
            data: Vec::new(),
            bits: 0,
        }
    }

    /// Overwrites the whole cube with `state`, dropping the index array.
    pub fn fill(&mut self, state: BlockStateId) {
        self.palette = vec![state];
        self.reverse = ReverseMap::default();
        self.data = Vec::new();
        self.bits = 0;
    }

    /// Builds a cube from per-position states, evaluated in index order.
    /// Collapses to a uniform cube if only one distinct state is produced.
    pub fn from_fn(mut f: impl FnMut(LocalPos) -> BlockStateId) -> Self {
        let mut assign: ReverseMap = ReverseMap::default();
        let mut palette: Vec<BlockStateId> = Vec::new();
        let mut indices: Vec<u16> = Vec::with_capacity(CUBE_VOLUME);
        for index in 0..CUBE_VOLUME {
            let state = f(LocalPos::from_index(index));
            let next = palette.len() as u16;
            let assigned = *assign.entry(state).or_insert(next);
            if assigned == next {
                palette.push(state);
            }
            indices.push(assigned);
        }
        if palette.len() == 1 {
            return Self::filled(palette[0]);
        }
        let bits = bits_for(palette.len());
        let epw = entries_per_word(bits);
        let mut data = vec![0u64; word_count(bits)];
        for (index, &pi) in indices.iter().enumerate() {
            data[index / epw] |= u64::from(pi) << ((index % epw) * bits as usize);
        }
        let mut reverse = ReverseMap::default();
        if palette.len() > LINEAR_SCAN_LIMIT {
            for (i, &state) in palette.iter().enumerate() {
                reverse.insert(state, i as u16);
            }
        }
        Self {
            palette,
            reverse,
            data,
            bits,
        }
    }

    #[inline]
    pub fn get(&self, pos: LocalPos) -> BlockStateId {
        self.get_index(pos.index())
    }

    /// Reads by linear index (`0..CUBE_VOLUME`). O(1).
    #[inline]
    pub fn get_index(&self, index: usize) -> BlockStateId {
        debug_assert!(index < CUBE_VOLUME, "index out of cube");
        if self.data.is_empty() {
            return self.palette[0];
        }
        let epw = entries_per_word(self.bits);
        let bit = (index % epw) * self.bits as usize;
        let pi = (self.data[index / epw] >> bit) & ((1u64 << self.bits) - 1);
        self.palette[pi as usize]
    }

    #[inline]
    pub fn set(&mut self, pos: LocalPos, state: BlockStateId) -> BlockStateId {
        self.set_index(pos.index(), state)
    }

    /// Writes by linear index and returns the previous state. Setting the
    /// state a block already has leaves the palette untouched.
    pub fn set_index(&mut self, index: usize, state: BlockStateId) -> BlockStateId {
        debug_assert!(index < CUBE_VOLUME, "index out of cube");
        let old = self.get_index(index);
        if old == state {
            return old;
        }
        let pi = self.find_or_insert(state);
        debug_assert!(self.bits > 0);
        let epw = entries_per_word(self.bits);
        let bit = (index % epw) * self.bits as usize;
        let mask = ((1u64 << self.bits) - 1) << bit;
        let word = &mut self.data[index / epw];
        *word = (*word & !mask) | (u64::from(pi) << bit);
        old
    }

    /// The single state of a uniform cube, `None` otherwise.
    pub fn uniform(&self) -> Option<BlockStateId> {
        self.palette
            .first()
            .copied()
            .filter(|_| self.data.is_empty())
    }

    pub fn palette(&self) -> &[BlockStateId] {
        &self.palette
    }

    /// Bits per packed entry; 0 for a uniform cube.
    pub fn bits_per_entry(&self) -> u32 {
        self.bits
    }

    /// Bytes owned on the heap (palette + packed array + reverse map,
    /// by capacity). Approximate: allocator overhead and hashmap control
    /// bytes are not counted. Input to the 1.8 memory bench.
    pub fn heap_bytes(&self) -> usize {
        self.palette.capacity() * size_of::<BlockStateId>()
            + self.data.capacity() * size_of::<u64>()
            + self.reverse.capacity() * (size_of::<(BlockStateId, u16)>() + 1)
    }

    /// Iterates all `CUBE_VOLUME` states in index order. Caches one packed
    /// word at a time so only the palette lookup is a checked index.
    pub fn iter(&self) -> CubeIter<'_> {
        let uniform = if self.data.is_empty() {
            Some(self.palette[0])
        } else {
            None
        };
        CubeIter {
            palette: &self.palette,
            data: &self.data,
            bits: self.bits,
            mask: if self.bits == 0 {
                0
            } else {
                (1u64 << self.bits) - 1
            },
            epw: if self.bits == 0 {
                0
            } else {
                entries_per_word(self.bits) as u32
            },
            word: 0,
            word_index: 0,
            left_in_word: 0,
            remaining: CUBE_VOLUME,
            uniform,
        }
    }

    /// Drops unused palette entries, remaps indices and shrinks to the
    /// minimal width, going back to no array when a single state remains.
    ///
    /// Not run automatically by `set`: callers run it after bulk edits or
    /// before saving.
    pub fn compact(&mut self) {
        if self.data.is_empty() {
            return; // already minimal
        }
        let mut used = vec![false; self.palette.len()];
        let palette_len = self.palette.len();
        self.for_each_index(|pi| used[pi] = true);
        let mut remap = vec![0u16; palette_len];
        let mut new_palette = Vec::with_capacity(used.iter().filter(|u| **u).count());
        for (old, is_used) in used.iter().enumerate() {
            if *is_used {
                remap[old] = new_palette.len() as u16;
                new_palette.push(self.palette[old]);
            }
        }
        self.palette = new_palette;
        if self.palette.len() == 1 {
            self.data = Vec::new();
            self.bits = 0;
            self.reverse = ReverseMap::default();
            return;
        }
        let new_bits = bits_for(self.palette.len());
        self.repack(new_bits, Some(&remap));
        self.bits = new_bits;
        if self.palette.len() > LINEAR_SCAN_LIMIT {
            self.rebuild_reverse();
        } else {
            self.reverse = ReverseMap::default();
        }
    }

    /// Palette index for `state`, appending it (and repacking to a wider
    /// layout) when absent.
    fn find_or_insert(&mut self, state: BlockStateId) -> u16 {
        if self.palette.len() <= LINEAR_SCAN_LIMIT {
            if let Some(i) = self.palette.iter().position(|&s| s == state) {
                return i as u16;
            }
        } else if let Some(&i) = self.reverse.get(&state) {
            return i;
        }

        self.palette.push(state);
        let new_len = self.palette.len();
        if new_len == 2 {
            // Uniform -> packed: all-zero entries point at the old state.
            self.bits = 1;
            self.data = vec![0u64; word_count(1)];
        } else {
            let new_bits = bits_for(new_len);
            if new_bits != self.bits {
                self.repack(new_bits, None);
                self.bits = new_bits;
            }
        }
        if new_len > LINEAR_SCAN_LIMIT {
            if self.reverse.is_empty() {
                self.rebuild_reverse();
            } else {
                self.reverse.insert(state, (new_len - 1) as u16);
            }
        }
        (new_len - 1) as u16
    }

    /// Repacks every entry at `new_bits`, optionally remapping old palette
    /// indices through `remap` (used by `compact`).
    fn repack(&mut self, new_bits: u32, remap: Option<&[u16]>) {
        debug_assert!(self.bits > 0 && new_bits > 0);
        let old_bits = self.bits;
        let old_mask = (1u64 << old_bits) - 1;
        let old_epw = entries_per_word(old_bits);
        let new_epw = entries_per_word(new_bits);
        let mut new_data = Vec::with_capacity(word_count(new_bits));
        let mut word = 0u64;
        let mut filled = 0usize;
        let mut written = 0usize;
        'outer: for &old_word in &self.data {
            for e in 0..old_epw {
                if written == CUBE_VOLUME {
                    break 'outer;
                }
                let old_pi = ((old_word >> (e * old_bits as usize)) & old_mask) as usize;
                let pi = match remap {
                    Some(remap) => remap[old_pi] as usize,
                    None => old_pi,
                };
                word |= (pi as u64) << (filled * new_bits as usize);
                filled += 1;
                if filled == new_epw {
                    new_data.push(word);
                    word = 0;
                    filled = 0;
                }
                written += 1;
            }
        }
        if filled > 0 {
            new_data.push(word);
        }
        self.data = new_data;
    }

    /// Calls `f` with the raw palette index of every cell, in index order.
    fn for_each_index(&self, mut f: impl FnMut(usize)) {
        if self.data.is_empty() {
            return;
        }
        let bits = self.bits;
        let mask = (1u64 << bits) - 1;
        let epw = entries_per_word(bits);
        let mut n = 0usize;
        'outer: for &word in &self.data {
            for e in 0..epw {
                if n == CUBE_VOLUME {
                    break 'outer;
                }
                f(((word >> (e * bits as usize)) & mask) as usize);
                n += 1;
            }
        }
    }

    fn rebuild_reverse(&mut self) {
        self.reverse.clear();
        for (i, &state) in self.palette.iter().enumerate() {
            self.reverse.insert(state, i as u16);
        }
    }
}

impl Default for Cube {
    /// An all-air cube.
    fn default() -> Self {
        Self::filled(BlockStateId::AIR)
    }
}

/// Iterator over all states of a cube, in index order. See [`Cube::iter`].
pub struct CubeIter<'a> {
    palette: &'a [BlockStateId],
    data: &'a [u64],
    bits: u32,
    mask: u64,
    epw: u32,
    /// Current word, already shifted past consumed entries.
    word: u64,
    word_index: usize,
    left_in_word: u32,
    remaining: usize,
    /// Set while the cube is uniform: just repeat this state.
    uniform: Option<BlockStateId>,
}

impl Iterator for CubeIter<'_> {
    type Item = BlockStateId;

    #[inline]
    fn next(&mut self) -> Option<BlockStateId> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        if let Some(state) = self.uniform {
            return Some(state);
        }
        if self.left_in_word == 0 {
            // Safe by construction: entries never straddle words and the
            // array holds every cell.
            self.word = self.data[self.word_index];
            self.word_index += 1;
            self.left_in_word = self.epw;
        }
        self.left_in_word -= 1;
        let pi = (self.word & self.mask) as usize;
        self.word >>= self.bits;
        Some(self.palette[pi])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for CubeIter<'_> {}

#[cfg(test)]
impl Cube {
    /// Asserts every documented invariant of the storage layout.
    pub fn check_invariants(&self) {
        assert!(!self.palette.is_empty(), "empty palette");
        assert!(
            self.palette.len() <= CUBE_VOLUME,
            "palette larger than the cube"
        );
        let mut sorted = self.palette.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), self.palette.len(), "duplicate palette entry");
        assert_eq!(self.bits, bits_for(self.palette.len()), "stale width");

        if self.palette.len() == 1 {
            assert!(self.data.is_empty(), "uniform cube with an index array");
            assert_eq!(self.uniform(), Some(self.palette[0]));
        } else {
            assert!(!self.data.is_empty(), "non-uniform cube without an array");
            assert_eq!(self.data.len(), word_count(self.bits), "wrong word count");
            assert_eq!(self.uniform(), None);
            let len = self.palette.len();
            let mut n = 0;
            self.for_each_index(|pi| {
                assert!(pi < len, "index {pi} out of palette");
                n += 1;
            });
            assert_eq!(n, CUBE_VOLUME, "array does not cover the cube");
        }

        if self.palette.len() > LINEAR_SCAN_LIMIT {
            assert_eq!(self.reverse.len(), self.palette.len(), "stale reverse map");
            for (i, &state) in self.palette.iter().enumerate() {
                assert_eq!(
                    self.reverse.get(&state),
                    Some(&(i as u16)),
                    "reverse map disagrees with palette"
                );
            }
        } else {
            assert!(
                self.reverse.is_empty(),
                "reverse map kept below the scan limit"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn uniform_cube_has_no_array() {
        let stone = BlockStateId(7);
        let cube = Cube::filled(stone);
        assert_eq!(cube.uniform(), Some(stone));
        assert_eq!(cube.palette(), &[stone]);
        assert_eq!(cube.bits_per_entry(), 0);
        assert!(cube.heap_bytes() < 64, "heap_bytes: {}", cube.heap_bytes());
        assert_eq!(cube.get_index(0), stone);
        assert_eq!(cube.get_index(CUBE_VOLUME - 1), stone);
        assert_eq!(cube.iter().count(), CUBE_VOLUME);
        assert!(cube.iter().all(|s| s == stone));
        cube.check_invariants();

        let air = Cube::default();
        assert_eq!(air.uniform(), Some(BlockStateId::AIR));
        air.check_invariants();
    }

    #[test]
    fn growth_through_thresholds() {
        let mut cube = Cube::default();
        let mut written: Vec<(usize, BlockStateId)> = Vec::new();
        // Palette lengths after each phase: 2, 3, 5, 17, 257, 4097 (with air).
        for (target, expected_bits) in [
            (1usize, 1u32),
            (2, 2),
            (4, 3),
            (16, 5),
            (256, 9),
            (4096, 13),
        ] {
            while cube.palette().len() <= target {
                let id = BlockStateId(written.len() as u32 + 1);
                let index = written.len();
                assert_eq!(cube.set_index(index, id), BlockStateId::AIR);
                written.push((index, id));
            }
            assert_eq!(cube.palette().len(), target + 1);
            assert_eq!(cube.bits_per_entry(), expected_bits);
            for &(index, id) in &written {
                assert_eq!(cube.get_index(index), id);
            }
            cube.check_invariants();
        }
        assert_eq!(cube.palette().len(), 4097);
    }

    #[test]
    fn set_returns_old_and_same_state_is_free() {
        let mut cube = Cube::filled(BlockStateId(1));
        assert_eq!(cube.set_index(5, BlockStateId(2)), BlockStateId(1));
        let palette_before = cube.palette().to_vec();
        let words_before = cube.heap_bytes();
        assert_eq!(cube.set_index(5, BlockStateId(2)), BlockStateId(2));
        assert_eq!(cube.set_index(6, BlockStateId(1)), BlockStateId(1));
        assert_eq!(cube.palette(), palette_before.as_slice());
        assert!(cube.heap_bytes() <= words_before);
        cube.check_invariants();
    }

    #[test]
    fn compact_shrinks_palette_and_width() {
        let mut cube = Cube::default();
        // 4096 distinct states, then all cells overwritten with 16 states.
        for i in 0..4096 {
            cube.set_index(i, BlockStateId(i as u32 + 1));
        }
        let mut reference = vec![BlockStateId::AIR; CUBE_VOLUME];
        for (i, cell) in reference.iter_mut().enumerate() {
            let state = BlockStateId((i % 16) as u32 + 1);
            cube.set_index(i, state);
            *cell = state;
        }
        assert!(cube.palette().len() > 4096);
        cube.compact();
        assert!(
            cube.palette().len() <= 17,
            "palette: {}",
            cube.palette().len()
        );
        assert!(cube.bits_per_entry() <= 5);
        cube.check_invariants();
        for (i, &expected) in reference.iter().enumerate() {
            assert_eq!(cube.get_index(i), expected);
        }

        // Down to a single state: back to no array.
        cube.fill(BlockStateId(9));
        cube.set_index(0, BlockStateId(10));
        cube.set_index(0, BlockStateId(9));
        cube.compact();
        assert_eq!(cube.uniform(), Some(BlockStateId(9)));
        cube.check_invariants();
    }

    #[test]
    fn randomized_sets_match_reference() {
        let mut rng = XorShift64(0x2545_f491_4f6c_dd1d);
        let mut cube = Cube::default();
        let mut reference = vec![BlockStateId::AIR; CUBE_VOLUME];
        for op in 0..50_000 {
            let index = (rng.next() % CUBE_VOLUME as u64) as usize;
            let state = BlockStateId((rng.next() % 4096) as u32);
            assert_eq!(cube.set_index(index, state), reference[index]);
            reference[index] = state;
            if op % 1000 == 999 {
                cube.check_invariants();
            }
        }
        for (i, &expected) in reference.iter().enumerate() {
            assert_eq!(cube.get_index(i), expected);
        }
        for (state, &expected) in cube.iter().zip(&reference) {
            assert_eq!(state, expected);
        }
        cube.check_invariants();
    }

    #[test]
    fn worst_case_32768_states() {
        let cube = Cube::from_fn(|pos| BlockStateId(pos.index() as u32));
        assert_eq!(cube.palette().len(), CUBE_VOLUME);
        assert_eq!(cube.palette()[CUBE_VOLUME - 1], BlockStateId(32767));
        assert_eq!(cube.bits_per_entry(), 15);
        cube.check_invariants();
        for i in 0..CUBE_VOLUME {
            assert_eq!(cube.get_index(i), BlockStateId(i as u32));
        }
    }

    #[test]
    fn from_fn_visits_index_order_and_collapses_to_uniform() {
        let mut visited = 0usize;
        let cube = Cube::from_fn(|pos| {
            assert_eq!(pos.index(), visited);
            visited += 1;
            BlockStateId::AIR
        });
        assert_eq!(visited, CUBE_VOLUME);
        assert_eq!(cube.uniform(), Some(BlockStateId::AIR));
        assert_eq!(cube.bits_per_entry(), 0);
        assert_eq!(cube.palette().len(), 1);
        cube.check_invariants();
    }

    #[test]
    fn fill_resets_to_uniform() {
        let mut cube = Cube::from_fn(|pos| BlockStateId(pos.index() as u32 % 100));
        cube.fill(BlockStateId(5));
        assert_eq!(cube.uniform(), Some(BlockStateId(5)));
        assert_eq!(cube.bits_per_entry(), 0);
        assert!(cube.heap_bytes() < 64);
        cube.check_invariants();
    }
}
