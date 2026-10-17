//! CPU-side page and slot bookkeeping: what each resident cube holds
//! (page range, meta slot, direction ranges, coverage state), the free
//! list of meta slots, and the per-page region-bound running total. Pure
//! logic, no wgpu — the churn unit test drives it directly.

use std::collections::HashMap;
use std::ops::Range;

use glam::{IVec3, Vec3};

use crate::lod::CubeKey;
use crate::render::suballoc::Suballocator;

use super::space::{Frustum, cube_origin, cube_size, visible_directions};

/// One cube's uploaded mesh. Empty meshes keep an entry (so the world
/// knows the cube is meshed) but allocate no page range and hold no meta
/// slot. Non-empty meshes record their page, face range and meta slot so a
/// re-mesh can free exactly what they had and rewrite the slot in place.
pub(super) struct CubeMesh {
    pub(super) face_count: u32,
    /// Page index and face range within that page's buffer.
    pub(super) alloc: Option<(usize, Range<u32>)>,
    /// The cube's slot in the meta buffer (`None` for empty meshes).
    pub(super) slot: Option<u32>,
    /// Per-direction face ranges within the cube's faces
    /// (`MeshOutput::ranges`), so drawing can skip back-facing directions.
    pub(super) ranges: [Range<u32>; 6],
    /// Faces emitted under the skirt rule (level >= 1), from the mesher.
    pub(super) skirts: u32,
    /// The coverage state riding in the meta flags: whether the cube
    /// draws at all, and which octants a ready child replaces. Kept so a
    /// re-mesh rewrites its slot with the same state.
    pub(super) drawn: bool,
    pub(super) mask: u8,
}

/// Non-empty direction ranges of a mesh: the per-cube share of the
/// per-page command-region bound.
fn non_empty_dirs(ranges: &[Range<u32>; 6]) -> u32 {
    ranges.iter().filter(|r| !r.is_empty()).count() as u32
}

/// CPU-side mesh bookkeeping shared by every path: cube entries, meta
/// slots (free list, high-water mark) and the per-page region bound (a
/// running total of resident cubes' non-empty direction counts). Pure
/// logic, no wgpu — the churn unit test drives it directly.
#[derive(Default)]
pub(super) struct MeshBookkeeping {
    pub(super) meshes: HashMap<CubeKey, CubeMesh>,
    /// Slots handed out (high-water mark); slots below it are either in
    /// use or on the free list.
    pub(super) meta_len: u32,
    meta_free: Vec<u32>,
    /// Per-page running total of non-empty direction counts. Pages are
    /// never freed, so this only grows along `pages`.
    pub(super) page_dir_counts: Vec<u32>,
}

/// What `release` freed for a cube being re-meshed or removed. The
/// coverage state comes back only when the cube had a mesh entry.
pub(super) struct Released {
    pub(super) page: Option<usize>,
    pub(super) range: Option<Range<u32>>,
    pub(super) slot: Option<u32>,
    pub(super) drawn: Option<bool>,
    pub(super) mask: Option<u8>,
}

/// One GPU face page: a storage buffer, its suballocator and its bind group.
/// Pages are never freed or compacted — fragmentation grows until a later
/// compaction task; note this as a known limit.
pub(super) struct FacePage {
    pub(super) buffer: wgpu::Buffer,
    pub(super) suballoc: Suballocator,
    pub(super) bind_group: wgpu::BindGroup,
}

impl MeshBookkeeping {
    pub(super) fn page_added(&mut self) {
        self.page_dir_counts.push(0);
    }

    /// Drops the cube's previous allocation and region-bound share,
    /// returning what the caller must free on the GPU. The direction
    /// count leaves the page's total here already, so `commit` can add it
    /// to whichever page the new mesh lands in.
    pub(super) fn release(&mut self, key: &CubeKey) -> Released {
        let Some(mesh) = self.meshes.get(key) else {
            return Released {
                page: None,
                range: None,
                slot: None,
                drawn: None,
                mask: None,
            };
        };
        let (page, range) = match &mesh.alloc {
            Some((page, range)) => (Some(*page), Some(range.clone())),
            None => (None, None),
        };
        if let Some(page) = page {
            self.page_dir_counts[page] -= non_empty_dirs(&mesh.ranges);
        }
        Released {
            page,
            range,
            slot: mesh.slot,
            drawn: Some(mesh.drawn),
            mask: Some(mesh.mask),
        }
    }

    /// A free meta slot, or the next never-used one (the caller grows the
    /// GPU buffer when the returned slot reaches its capacity).
    pub(super) fn alloc_meta_slot(&mut self) -> u32 {
        if let Some(slot) = self.meta_free.pop() {
            return slot;
        }
        let slot = self.meta_len;
        self.meta_len += 1;
        slot
    }

    pub(super) fn free_meta_slot(&mut self, slot: u32) {
        self.meta_free.push(slot);
    }

    /// Records a non-empty mesh at `(page, range)`, reusing `slot` when
    /// the cube had one (a re-mesh rewrites its slot in place). Returns
    /// the slot the caller must write the meta to. `skirts`/`drawn`/`mask`
    /// carry the mesher's skirt count and the cube's coverage state.
    // The nine parameters are one upload's worth of bookkeeping; a bundling
    // struct would only be unpacked again here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn commit(
        &mut self,
        key: CubeKey,
        page: usize,
        range: Range<u32>,
        ranges: [Range<u32>; 6],
        slot: Option<u32>,
        skirts: u32,
        drawn: bool,
        mask: u8,
    ) -> u32 {
        let slot = slot.unwrap_or_else(|| self.alloc_meta_slot());
        self.page_dir_counts[page] += non_empty_dirs(&ranges);
        self.meshes.insert(
            key,
            CubeMesh {
                face_count: range.len() as u32,
                alloc: Some((page, range)),
                slot: Some(slot),
                ranges,
                skirts,
                drawn,
                mask,
            },
        );
        slot
    }

    /// An empty mesh: entry kept, no allocation, no slot. `drawn`/`mask`
    /// carry the cube's coverage state over, exactly as a non-empty
    /// re-mesh does: resetting them here once let a cube the coverage
    /// rule had hidden draw again after an empty-then-non-empty re-mesh
    /// (two levels drawing the same water, z-fighting).
    pub(super) fn commit_empty(&mut self, key: CubeKey, drawn: bool, mask: u8) {
        self.meshes.insert(
            key,
            CubeMesh {
                face_count: 0,
                alloc: None,
                slot: None,
                ranges: std::array::from_fn(|_| 0u32..0),
                skirts: 0,
                drawn,
                mask,
            },
        );
    }

    pub(super) fn total_dirs(&self) -> u32 {
        self.page_dir_counts.iter().sum()
    }

    /// The per-frame CPU frustum bound on each page's command region: for
    /// every resident mesh that passes the same frustum test and
    /// `visible_directions` rule the cpu-cull path uses (same origin
    /// formula), the count of non-empty, non-skipped directions, summed
    /// per page. Metal has no `draw_indirect_count` and emulates
    /// `multi_draw_indirect` as a CPU loop of one indirect draw per
    /// encoded command, so the encoded draw count must come from the CPU —
    /// sizing the regions to what survives the frustum (instead of every
    /// resident cube) keeps the zero-tail encode cost proportional to what
    /// is actually visible.
    ///
    /// The GPU still runs its own frustum, occlusion and direction tests;
    /// this only sizes the regions. The bound is >= what the GPU emits
    /// because both sides use identical math (the same view-projection
    /// rows, the same origin formula, the same comparisons): every command
    /// the GPU writes is for a frustum-passing cube and a direction this
    /// function counted. The shader's capacity clamp plus the
    /// `region_overflow` counter (asserted zero by the tests) guard the
    /// day that drifts.
    pub(super) fn frustum_page_bounds(
        &self,
        camera_cube: IVec3,
        camera_frac: Vec3,
        frustum: &Frustum,
    ) -> Vec<u32> {
        let mut bounds = vec![0u32; self.page_dir_counts.len()];
        for (key, mesh) in &self.meshes {
            let Some((page, _)) = &mesh.alloc else {
                continue;
            };
            // Cubes the coverage rule does not draw — fully masked ones
            // included — emit nothing on any path, so they stay out of
            // the bound too.
            if !mesh.drawn || mesh.mask == 0xFF {
                continue;
            }
            let min = cube_origin(*key, camera_cube, camera_frac);
            let max = min + cube_size(key.level);
            if !frustum.intersects_aabb(min, max) {
                continue;
            }
            let dirs = visible_directions(min, max);
            let kept = mesh
                .ranges
                .iter()
                .enumerate()
                .filter(|(d, range)| !range.is_empty() && dirs[*d])
                .count() as u32;
            bounds[*page] += kept;
        }
        bounds
    }

    /// The bound recounted from the meshes: the invariant the churn test
    /// checks after every operation.
    #[cfg(any(test, feature = "testing"))]
    pub(super) fn recount(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.page_dir_counts.len()];
        for mesh in self.meshes.values() {
            if let Some((page, _)) = &mesh.alloc {
                counts[*page] += non_empty_dirs(&mesh.ranges);
            }
        }
        counts
    }

    /// Slot and bound invariants (test-only): every non-empty mesh holds a
    /// slot, slots are unique, the free list holds only unused slots in
    /// range, and the running totals equal the recount.
    #[cfg(any(test, feature = "testing"))]
    pub(super) fn check_invariants(&self) {
        let mut used = std::collections::HashSet::new();
        for (pos, mesh) in &self.meshes {
            if mesh.face_count > 0 || mesh.alloc.is_some() {
                let slot = mesh
                    .slot
                    .unwrap_or_else(|| panic!("{pos:?} holds no meta slot"));
                assert!(slot < self.meta_len, "{pos:?} slot {slot} out of range");
                assert!(used.insert(slot), "slot {slot} used twice");
            } else {
                assert!(mesh.slot.is_none(), "an empty mesh must hold no slot");
            }
        }
        for &slot in &self.meta_free {
            assert!(slot < self.meta_len, "free slot {slot} out of range");
            assert!(!used.contains(&slot), "free slot {slot} is in use");
        }
        assert_eq!(
            self.page_dir_counts,
            self.recount(),
            "per-page region bound drifted from the recount"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::render::terrain::space::camera_split;
    use glam::DVec3;
    use jbm_core::coords::CubePos;

    fn key(x: i32, y: i32, z: i32, level: u8) -> CubeKey {
        CubeKey {
            level,
            pos: CubePos { x, y, z },
        }
    }

    /// The frustum bound is exactly the cpu-cull path's command count: for
    /// a camera, the per-page bounds count the (frustum-passing cube,
    /// non-empty non-skipped direction) pairs, and never exceed the static
    /// per-page totals.
    #[test]
    fn frustum_bound_counts_visible_directions_per_page() {
        // Camera in the middle of cube (0,0,0), yaw 0 (looking toward -Z),
        // every direction range non-empty in every mesh.
        let cam = Camera::new(DVec3::splat(16.0), 70.0_f32.to_radians(), 1.0);
        let (cube, frac) = camera_split(DVec3::splat(16.0));
        let frustum = Frustum::from_view_proj(&cam.view_proj());
        let six_ranges = || [0u32..3, 0u32..3, 0u32..3, 0u32..3, 0u32..3, 0u32..3];
        let mut book = MeshBookkeeping::default();
        book.page_added();
        book.page_added();
        // Ahead, fully in the (+,+,+) octant relative to the camera: only
        // -X, -Y, +Z face it. Page 0.
        book.commit(key(1, 1, -2, 0), 0, 0..18, six_ranges(), None, 0, true, 0);
        // Far outside the 70-degree fov on +X, ahead in z: dropped. Page 0.
        book.commit(key(12, 0, -1, 0), 0, 18..36, six_ranges(), None, 0, true, 0);
        // Behind the camera (yaw 0). Page 1.
        book.commit(key(0, 0, 2, 0), 1, 36..54, six_ranges(), None, 0, true, 0);
        // The cube containing the camera: all six directions, in any
        // frustum. Page 1.
        book.commit(key(0, 0, 0, 0), 1, 54..72, six_ranges(), None, 0, true, 0);
        let bounds = book.frustum_page_bounds(cube, frac, &frustum);
        assert_eq!(bounds, vec![3, 6]);
        assert!(
            bounds
                .iter()
                .zip(&book.page_dir_counts)
                .all(|(&bound, &total)| bound <= total),
            "the frustum bound must not exceed the static bound"
        );
        book.check_invariants();

        // Looking the other way (yaw pi): the ahead and out-of-fov cubes
        // leave the frustum, the behind cube enters it (five kept
        // directions — it straddles the camera's x and y) and the camera
        // cube keeps its six.
        let mut turned = Camera::new(DVec3::splat(16.0), 70.0_f32.to_radians(), 1.0);
        turned.set_yaw(std::f32::consts::PI);
        let bounds =
            book.frustum_page_bounds(cube, frac, &Frustum::from_view_proj(&turned.view_proj()));
        assert_eq!(bounds, vec![0, 11]);

        // An empty direction range is never counted, even when visible.
        let mut sparse = six_ranges();
        sparse[0] = 0u32..0;
        sparse[2] = 0u32..0;
        let mut book = MeshBookkeeping::default();
        book.page_added();
        book.commit(key(0, 0, 0, 0), 0, 0..6, sparse, None, 0, true, 0);
        let bounds =
            book.frustum_page_bounds(cube, frac, &Frustum::from_view_proj(&cam.view_proj()));
        assert_eq!(bounds, vec![4], "only the four non-empty directions count");
    }

    #[test]
    fn non_empty_dirs_counts_only_non_empty_ranges() {
        let mut ranges: [Range<u32>; 6] = std::array::from_fn(|d| (d as u32)..(d as u32 + 2));
        ranges[2] = 0..0;
        ranges[5] = 0..0;
        assert_eq!(non_empty_dirs(&ranges), 4);
    }

    /// The seeded churn for the per-page region bound and the meta slots:
    /// upload, re-mesh (same page, moving page, to empty) and remove
    /// against a recount after every check.
    #[test]
    fn region_bound_and_slots_stay_correct_across_churn() {
        let mut rng = XorShift64(0x5eed_c0de_0000_0001);
        let mut book = MeshBookkeeping::default();
        for _ in 0..4 {
            book.page_added();
        }
        let mut live: Vec<CubeKey> = Vec::new();
        let mut next_pos = 0;
        for _ in 0..10_000 {
            let op = rng.next() % 4;
            if op == 0 || live.is_empty() {
                // Upload a new cube.
                let pos = key(next_pos, 0, 0, 0);
                next_pos += 1;
                let ranges = random_ranges(&mut rng);
                let page = (rng.next() % 4) as usize;
                let start = (rng.next() % 1024) as u32;
                let len: u32 = ranges.iter().map(|r| r.len() as u32).sum();
                book.commit(pos, page, start..start + len, ranges, None, 0, true, 0);
                live.push(pos);
            } else if op == 1 {
                // Remove.
                let i = (rng.next() as usize) % live.len();
                let pos = live.swap_remove(i);
                let released = book.release(&pos);
                if let Some(slot) = released.slot {
                    book.free_meta_slot(slot);
                }
                book.meshes.remove(&pos);
            } else {
                // Re-mesh: release then commit, sometimes moving page and
                // sometimes emptying out.
                let i = (rng.next() as usize) % live.len();
                let pos = live[i];
                let released = book.release(&pos);
                if let Some(slot) = released.slot {
                    book.free_meta_slot(slot);
                }
                if rng.next().is_multiple_of(8) {
                    book.commit_empty(pos, true, 0);
                    live.swap_remove(i);
                } else {
                    let ranges = random_ranges(&mut rng);
                    let page = (rng.next() % 4) as usize;
                    let start = (rng.next() % 1024) as u32;
                    let len: u32 = ranges.iter().map(|r| r.len() as u32).sum();
                    let slot = book.alloc_meta_slot();
                    book.commit(
                        pos,
                        page,
                        start..start + len,
                        ranges,
                        Some(slot),
                        0,
                        true,
                        0,
                    );
                }
            }
            if rng.next().is_multiple_of(64) {
                book.check_invariants();
            }
        }
        book.check_invariants();
        // The slot free list plus the live slots partition the high-water
        // mark, so no slot leaks.
        let live_slots: std::collections::HashSet<u32> =
            book.meshes.values().filter_map(|m| m.slot).collect();
        assert_eq!(
            live_slots.len() + book.meta_free.len(),
            book.meta_len as usize,
            "every handed-out slot is live or free"
        );
    }

    /// 0..6 non-empty direction ranges of random lengths.
    fn random_ranges(rng: &mut XorShift64) -> [Range<u32>; 6] {
        let mut ranges = std::array::from_fn(|_| 0u32..0);
        let dirs = 1 + rng.next() % 6;
        for (d, range) in ranges.iter_mut().enumerate() {
            if (d as u64) < dirs {
                let start = (rng.next() % 32) as u32;
                *range = start..start + 1 + (rng.next() % 5) as u32;
            }
        }
        ranges
    }

    /// The seeded xorshift the suballocator churn uses.
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
}
