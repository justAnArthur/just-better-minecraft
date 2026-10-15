//! The terrain pass: all cube meshes live in a few large face "pages"
//! (storage buffers) managed by CPU-side suballocators (render/suballoc.rs),
//! one persistent `cube_meta` slot per resident non-empty cube, and three
//! draw paths:
//!
//! - **gpu-cull** (default where available): a compute pass tests every
//!   meta slot against the frustum and a Hi-Z occlusion pyramid
//!   (render/hiz.rs) and writes the indirect commands into per-page
//!   regions; two render phases draw them. The CPU's only per-cube work is
//!   sizing the regions: a frustum pass over the resident meta slots that
//!   bounds what the GPU can emit (Metal's multi_draw_indirect is a CPU
//!   loop over the encoded commands, so the encoded count must come from
//!   the CPU — see `MeshBookkeeping::frustum_page_bounds`).
//! - **cpu-cull**: the 2.2 CPU frustum and direction tests, using the same
//!   meta slots and the shared origin formula, drawn with one
//!   multi_draw_indirect per page.
//! - **fallback**: the cpu-cull commands issued as direct draws, for
//!   hardware without indirect support.
//!
//! See crates/client/README.md.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, IVec3, Mat4, Vec3, Vec4};
use jbm_core::coords::{CUBE_SIZE, CubePos};

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::mesh::face::PackedFace;
use crate::mesh::greedy::MeshOutput;
use crate::render::hiz::HizPyramid;
use crate::render::renderer::{DEPTH_FORMAT, DepthTarget};
use crate::render::suballoc::Suballocator;
use crate::render::texture_array::BlockTextures;

/// Bytes per face (`PackedFace`).
const FACE_BYTES: u64 = std::mem::size_of::<PackedFace>() as u64;

/// Bytes per indirect command (`wgpu::util::DrawIndirectArgs`).
const CMD_BYTES: u64 = 16;

/// Default page size: 4M faces (32 MiB). Clamped to the device limits at
/// construction, and never below one worst-case cube mesh.
const DEFAULT_PAGE_FACES: u32 = 4 * 1024 * 1024;

/// Meta slots preallocated before the meta buffer is grown.
const META_CAPACITY: u32 = 4096;

/// Indirect commands preallocated before the CPU indirect buffer is grown.
const INDIRECT_CAPACITY: u64 = 1024;

/// u32 words in the GPU stats buffer: 7 counters plus the region-overflow
/// counter (word 7, asserted zero by tests).
const STATS_WORDS: usize = 8;
const STATS_BYTES: u64 = STATS_WORDS as u64 * 4;

/// The most faces one cube's mesh can hold: the checkerboard worst case,
/// half the cube's blocks facing all six ways.
pub fn worst_case_faces() -> u32 {
    ((CUBE_SIZE as u64).pow(3) * 6 / 2) as u32
}

/// Meta flag: the slot holds a resident cube (a freed slot reads inactive
/// and the cull thread exits early).
pub const META_ACTIVE: u32 = 1;
/// Meta flag: the cube was visible last frame (two-phase culling state,
/// written by the late cull, reset on upload so new and re-meshed cubes
/// are picked up by the late pass of the same frame).
pub const META_VISIBLE: u32 = 2;

/// One cube's uploaded mesh, mirrored on the GPU as `CubeMeta`
/// (shaders/terrain.wgsl, shaders/cull.wgsl — keep all three in sync).
///
/// Layout: every field is a 16-byte group, so Rust `#[repr(C)]` and the
/// WGSL struct agree at 16-byte alignment with no hidden padding; the
/// layout test below pins it. 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CubeMeta {
    /// x, y, z = the cube coordinates; w = the page index.
    pub pos_page: [i32; 4],
    /// x = the face range start in the page; y = flags (META_*); z, w spare.
    pub start_flags: [u32; 4],
    /// The six per-direction (offset, count) pairs from
    /// `MeshOutput::ranges`, offsets relative to the cube's face range:
    /// direction d's pair is `dirs[d / 2][(d % 2) * 2 ..][..2]`.
    pub dirs: [[u32; 4]; 3],
}

impl CubeMeta {
    fn zeroed() -> CubeMeta {
        CubeMeta {
            pos_page: [0; 4],
            start_flags: [0; 4],
            dirs: [[0; 4]; 3],
        }
    }
}

/// The terrain-only per-frame uniform (group 2, binding 1). The two values
/// behind the shared origin formula; sky, outline and overlay share only
/// `FrameUniforms`, so they are untouched by it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TerrainUniforms {
    /// xyz = `floor(camera / CUBE_SIZE)` as vec3<i32>.
    pub camera_cube: [i32; 4],
    /// xyz = `camera − camera_cube · CUBE_SIZE` (f64 on the CPU, rounded
    /// to f32); w = CUBE_SIZE, so the shaders are cube-size agnostic.
    pub camera_frac: [f32; 4],
}

/// The cull shader's per-frame uniform (group 1, binding 0 of cull.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CullParams {
    pub camera_cube: [i32; 4],
    pub camera_frac: [f32; 4],
    /// x, y = depth target size; z = the pyramid's max mip level;
    /// w = meta slots in use (the dispatch bound).
    pub screen: [u32; 4],
    /// x = the late half's base, in command elements; y = page count;
    /// z = command capacity, in elements; w = CUBE_SIZE.
    pub regions: [u32; 4],
}

/// GPU culling counters, filled by cull.wgsl and read back one or two
/// frames late (see `TerrainPass::cull_stats`). On the cpu-cull and
/// fallback paths the same fields are filled CPU-side, immediately.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CullStats {
    /// Active slots this frame.
    pub candidates: u32,
    /// Active slots the frustum test rejected.
    pub frustum_culled: u32,
    /// Frustum-passing slots the Hi-Z test rejected.
    pub occluded: u32,
    /// Slots drawn in the early phase (were visible last frame).
    pub drawn_early: u32,
    /// Slots drawn in the late phase (newly visible).
    pub drawn_late: u32,
    /// Indirect commands emitted (early + late).
    pub commands: u32,
    /// Faces issued, counting only the non-skipped directions.
    pub faces: u32,
    /// Commands the shader dropped because their command-region slot was
    /// at or beyond the buffer capacity: a bound/capacity bug (the CPU
    /// frustum bound must be >= what the GPU emits). Always zero; the
    /// tests assert it.
    pub region_overflow: u32,
}

impl CullStats {
    fn from_words(words: &[u32]) -> CullStats {
        CullStats {
            candidates: words[0],
            frustum_culled: words[1],
            occluded: words[2],
            drawn_early: words[3],
            drawn_late: words[4],
            commands: words[5],
            faces: words[6],
            region_overflow: words[7],
        }
    }

    /// The drawn count for the overlay: early + late.
    pub fn drawn(&self) -> u32 {
        self.drawn_early + self.drawn_late
    }
}

/// One cube's uploaded mesh. Empty meshes keep an entry (so the world
/// knows the cube is meshed) but allocate no page range and hold no meta
/// slot. Non-empty meshes record their page, face range and meta slot so a
/// re-mesh can free exactly what they had and rewrite the slot in place.
struct CubeMesh {
    face_count: u32,
    /// Page index and face range within that page's buffer.
    alloc: Option<(usize, Range<u32>)>,
    /// The cube's slot in the meta buffer (`None` for empty meshes).
    slot: Option<u32>,
    /// Per-direction face ranges within the cube's faces
    /// (`MeshOutput::ranges`), so drawing can skip back-facing directions.
    ranges: [Range<u32>; 6],
}

/// The meta image of one mesh (`pos` is kept in the map key).
fn meta_for(pos: CubePos, page: usize, start: u32, ranges: &[Range<u32>; 6]) -> CubeMeta {
    let mut dirs = [[0u32; 4]; 3];
    for (d, range) in ranges.iter().enumerate() {
        dirs[d / 2][(d % 2) * 2] = range.start;
        dirs[d / 2][(d % 2) * 2 + 1] = range.len() as u32;
    }
    CubeMeta {
        pos_page: [pos.x, pos.y, pos.z, page as i32],
        start_flags: [start, META_ACTIVE, 0, 0],
        dirs,
    }
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
struct MeshBookkeeping {
    meshes: HashMap<CubePos, CubeMesh>,
    /// Slots handed out (high-water mark); slots below it are either in
    /// use or on the free list.
    meta_len: u32,
    meta_free: Vec<u32>,
    /// Per-page running total of non-empty direction counts. Pages are
    /// never freed, so this only grows along `pages`.
    page_dir_counts: Vec<u32>,
}

/// What `release` freed for a cube being re-meshed or removed.
struct Released {
    page: Option<usize>,
    range: Option<Range<u32>>,
    slot: Option<u32>,
}

impl MeshBookkeeping {
    fn page_added(&mut self) {
        self.page_dir_counts.push(0);
    }

    /// Drops the cube's previous allocation and region-bound share,
    /// returning what the caller must free on the GPU. The direction
    /// count leaves the page's total here already, so `commit` can add it
    /// to whichever page the new mesh lands in.
    fn release(&mut self, pos: &CubePos) -> Released {
        let Some(mesh) = self.meshes.get(pos) else {
            return Released {
                page: None,
                range: None,
                slot: None,
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
        }
    }

    /// A free meta slot, or the next never-used one (the caller grows the
    /// GPU buffer when the returned slot reaches its capacity).
    fn alloc_meta_slot(&mut self) -> u32 {
        if let Some(slot) = self.meta_free.pop() {
            return slot;
        }
        let slot = self.meta_len;
        self.meta_len += 1;
        slot
    }

    fn free_meta_slot(&mut self, slot: u32) {
        self.meta_free.push(slot);
    }

    /// Records a non-empty mesh at `(page, range)`, reusing `slot` when
    /// the cube had one (a re-mesh rewrites its slot in place). Returns
    /// the slot the caller must write the meta to.
    fn commit(
        &mut self,
        pos: CubePos,
        page: usize,
        range: Range<u32>,
        ranges: [Range<u32>; 6],
        slot: Option<u32>,
    ) -> u32 {
        let slot = slot.unwrap_or_else(|| self.alloc_meta_slot());
        self.page_dir_counts[page] += non_empty_dirs(&ranges);
        self.meshes.insert(
            pos,
            CubeMesh {
                face_count: range.len() as u32,
                alloc: Some((page, range)),
                slot: Some(slot),
                ranges,
            },
        );
        slot
    }

    /// An empty mesh: entry kept, no allocation, no slot.
    fn commit_empty(&mut self, pos: CubePos) {
        self.meshes.insert(
            pos,
            CubeMesh {
                face_count: 0,
                alloc: None,
                slot: None,
                ranges: std::array::from_fn(|_| 0u32..0),
            },
        );
    }

    fn total_dirs(&self) -> u32 {
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
    fn frustum_page_bounds(
        &self,
        camera_cube: IVec3,
        camera_frac: Vec3,
        frustum: &Frustum,
    ) -> Vec<u32> {
        let mut bounds = vec![0u32; self.page_dir_counts.len()];
        for (pos, mesh) in &self.meshes {
            let Some((page, _)) = &mesh.alloc else {
                continue;
            };
            let min = cube_origin(*pos, camera_cube, camera_frac);
            let max = min + Vec3::splat(CUBE_SIZE as f32);
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
    fn recount(&self) -> Vec<u32> {
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
    fn check_invariants(&self) {
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

/// One GPU face page: a storage buffer, its suballocator and its bind group.
/// Pages are never freed or compacted in this task — fragmentation grows
/// until a later compaction task; note this as a known limit.
struct FacePage {
    buffer: wgpu::Buffer,
    suballoc: Suballocator,
    bind_group: wgpu::BindGroup,
}

/// Camera-relative frustum for culling: five planes extracted from the
/// rotation-only view-projection (the far plane is at infinity with the
/// reverse-Z infinite-far projection). A point `p` is inside when
/// `plane . p >= 0` for every plane.
#[derive(Debug, Clone)]
pub struct Frustum {
    planes: [Vec4; 5],
}

impl Frustum {
    /// Extracts the left / right / bottom / top / near planes (Gribb &
    /// Hartmann row combinations) from a camera-relative view-projection.
    pub fn from_view_proj(view_proj: &Mat4) -> Frustum {
        let row = |r: usize| {
            Vec4::new(
                view_proj.x_axis[r],
                view_proj.y_axis[r],
                view_proj.z_axis[r],
                view_proj.w_axis[r],
            )
        };
        let row0 = row(0);
        let row1 = row(1);
        let row2 = row(2);
        let row3 = row(3);
        Frustum {
            planes: [
                row3 + row0, // left
                row3 - row0, // right
                row3 + row1, // bottom
                row3 - row1, // top
                row3 - row2, // near (reverse-Z: z' <= w')
            ],
        }
    }

    /// Whether any part of the axis-aligned box may be visible.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Support point farthest along the plane normal.
            let px = if plane.x >= 0.0 { max.x } else { min.x };
            let py = if plane.y >= 0.0 { max.y } else { min.y };
            let pz = if plane.z >= 0.0 { max.z } else { min.z };
            if plane.x * px + plane.y * py + plane.z * pz + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Which face directions of a cube with camera-relative AABB
/// `[rel_min, rel_max]` can contain front-facing faces. Order matches
/// `mesh::face::Direction`: +X, -X, +Y, -Y, +Z, -Z.
///
/// Per axis, one line each (the camera sits at the origin):
/// - `+X` is skipped when `rel_min.x >= 0`: every +X face plane is then at
///   `x >= rel_min.x >= 0 = camera.x`, and a +X plane is front-facing only
///   when the camera is strictly on its positive side — so all are back-
///   facing (or edge-on) and back-face culling would discard them anyway.
/// - `-X` is skipped when `rel_max.x <= 0`: every -X plane is at
///   `x <= rel_max.x <= 0`, so the camera is never on its negative side.
/// - Y and Z identically.
///
/// The image cannot change: the skipped faces were already discarded by the
/// pipeline's back-face culling.
pub fn visible_directions(rel_min: Vec3, rel_max: Vec3) -> [bool; 6] {
    [
        rel_min.x < 0.0,
        rel_max.x > 0.0,
        rel_min.y < 0.0,
        rel_max.y > 0.0,
        rel_min.z < 0.0,
        rel_max.z > 0.0,
    ]
}

/// The camera split behind the one origin formula: `camera_cube` is
/// `floor(camera / CUBE_SIZE)` per axis and `camera_frac` is
/// `camera − camera_cube · CUBE_SIZE`, computed in f64 and then rounded to
/// f32.
pub fn camera_split(camera: DVec3) -> (IVec3, Vec3) {
    let size = f64::from(CUBE_SIZE);
    let cube = IVec3::new(
        (camera.x / size).floor() as i32,
        (camera.y / size).floor() as i32,
        (camera.z / size).floor() as i32,
    );
    let frac = Vec3::new(
        (camera.x - f64::from(cube.x) * size) as f32,
        (camera.y - f64::from(cube.y) * size) as f32,
        (camera.z - f64::from(cube.z) * size) as f32,
    );
    (cube, frac)
}

/// The one origin formula, shared by the CPU paths here, `vs_main` in
/// shaders/terrain.wgsl and `cull.wgsl`:
/// `origin = (pos − camera_cube) · CUBE_SIZE − camera_frac`. The integer
/// part is exact (small integers in f32); the only rounding is the final
/// f32 subtract, and the shaders perform the very same two steps, so the
/// CPU and GPU cull on the same numbers.
pub fn cube_origin(pos: CubePos, camera_cube: IVec3, camera_frac: Vec3) -> Vec3 {
    let int_part = Vec3::new(
        (pos.x - camera_cube.x) as f32,
        (pos.y - camera_cube.y) as f32,
        (pos.z - camera_cube.z) as f32,
    ) * CUBE_SIZE as f32;
    int_part - camera_frac
}

/// Per-frame terrain draw statistics, for the overlay and the metrics
/// example.
#[derive(Debug, Clone, Copy, Default)]
pub struct TerrainStats {
    /// Cubes with an uploaded mesh (empty or not).
    pub meshed: usize,
    /// Cubes passing the frustum test with at least one face. On gpu-cull
    /// this (and `faces`, `commands`) comes from the GPU counters and is
    /// one or two frames late.
    pub drawn: usize,
    /// Faces issued this frame, counting only the non-skipped directions.
    pub faces: u64,
    /// Indirect commands (MDI) or fallback draws issued this frame; one per
    /// visible cube and non-empty, non-skipped face direction.
    pub commands: u64,
    /// Render-pass draw API calls: one per non-empty page and phase with
    /// MDI, one per command on the fallback path.
    pub draw_calls: u64,
    /// Face pages allocated so far (pages are never freed).
    pub pages: usize,
    /// GPU bytes across all pages.
    pub page_bytes: u64,
    /// Faces currently allocated across all pages.
    pub used_faces: u64,
}

/// A ring of three MAP_READ buffers the GPU stats are copied into, one per
/// frame; completed maps are read without blocking, so the visible stats
/// are one or two frames late.
struct StatsRing {
    buffers: Vec<wgpu::Buffer>,
    slots: Vec<SlotState>,
    cursor: usize,
    last_copied: Option<usize>,
    latest: CullStats,
}

enum SlotState {
    /// Idle: no copy since the last read.
    Free,
    /// A copy was recorded this frame. The map cannot be armed until that
    /// submit is on the queue, so the next frame (or the blocking read)
    /// arms it — a buffer with a pending map may not appear in a submit.
    Copied,
    /// map_async armed; waiting for the callback.
    Pending(std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>),
}

impl StatsRing {
    fn new(device: &wgpu::Device) -> StatsRing {
        let buffers = (0..3)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jbm cull stats readback"),
                    size: STATS_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        StatsRing {
            buffers,
            slots: vec![SlotState::Free, SlotState::Free, SlotState::Free],
            cursor: 0,
            last_copied: None,
            latest: CullStats::default(),
        }
    }

    /// One frame's start: complete what maps are done, then arm the maps
    /// for the slots copied last frame (their submit is on the queue by
    /// now, so arming is legal and the buffer never meets a submit while
    /// mapped).
    fn begin_frame(&mut self, gpu: &Gpu) {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        for i in 0..self.slots.len() {
            if let SlotState::Pending(receiver) = &self.slots[i] {
                match receiver.try_recv() {
                    Ok(Ok(())) => {
                        self.latest = Self::read(&self.buffers[i]);
                        self.buffers[i].unmap();
                        self.slots[i] = SlotState::Free;
                    }
                    Ok(Err(err)) => {
                        eprintln!("cull stats map failed: {err}");
                        self.buffers[i].unmap();
                        self.slots[i] = SlotState::Free;
                    }
                    Err(_) => {} // map not complete yet; retry next frame
                }
            }
            if matches!(self.slots[i], SlotState::Copied) {
                let (sender, receiver) = std::sync::mpsc::channel();
                self.buffers[i].map_async(wgpu::MapMode::Read, .., move |result| {
                    let _ = sender.send(result);
                });
                self.slots[i] = SlotState::Pending(receiver);
            }
        }
    }

    fn read(buffer: &wgpu::Buffer) -> CullStats {
        let data = buffer
            .slice(..)
            .get_mapped_range()
            .expect("stats buffer was just mapped");
        let mut words = [0u32; STATS_WORDS];
        for (i, word) in words.iter_mut().enumerate() {
            *word = u32::from_le_bytes([
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ]);
        }
        drop(data);
        CullStats::from_words(&words)
    }

    /// The next free slot for this frame's copy, if any (a still-mapped
    /// ring is skipped and the stats simply lag longer).
    fn take_slot(&mut self) -> Option<usize> {
        for i in 0..self.slots.len() {
            let slot = (self.cursor + i) % self.slots.len();
            if matches!(self.slots[slot], SlotState::Free) {
                self.cursor = (slot + 1) % self.slots.len();
                self.last_copied = Some(slot);
                self.slots[slot] = SlotState::Copied;
                return Some(slot);
            }
        }
        None
    }

    /// Waits for the most recent copied slot and reads it (tests, benches
    /// and the metrics example). The caller must have submitted the frame.
    fn read_blocking(&mut self, gpu: &Gpu) -> CullStats {
        let Some(slot) = self.last_copied else {
            return self.latest;
        };
        // Arm the map now if the frame that copied the slot has not been
        // followed by another frame yet (arming after the submit is legal);
        // then wait it out.
        if matches!(self.slots[slot], SlotState::Copied) {
            let (sender, receiver) = std::sync::mpsc::channel();
            self.buffers[slot].map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
            self.slots[slot] = SlotState::Pending(receiver);
        }
        if let SlotState::Pending(receiver) =
            std::mem::replace(&mut self.slots[slot], SlotState::Free)
        {
            let poll = gpu.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            assert!(poll.is_ok(), "device poll failed: {poll:?}");
            receiver
                .recv()
                .expect("map callback dropped without sending")
                .expect("stats buffer mapping failed");
            self.latest = Self::read(&self.buffers[slot]);
            self.buffers[slot].unmap();
        }
        self.latest
    }
}

/// Which of the three terrain paths a frame uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainPath {
    /// Compute frustum + Hi-Z occlusion culling writing the indirect
    /// commands; two draw phases.
    GpuCull,
    /// CPU frustum and directions over the same meta slots; one
    /// multi_draw_indirect per page.
    CpuCull,
    /// The cpu-cull commands issued as direct draws.
    Fallback,
}

/// Draws the terrain: one pipeline, one shared bind group, one page bind
/// group per page and one `multi_draw_indirect` (or a loop of direct draws)
/// per page.
pub struct TerrainPass {
    device: wgpu::Device,
    pipeline: wgpu::RenderPipeline,
    faces_layout: wgpu::BindGroupLayout,
    shared_layout: wgpu::BindGroupLayout,
    shared_bind_group: wgpu::BindGroup,
    /// Kept so the bind group can be rebuilt when the meta buffer grows;
    /// ref-counted clones.
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    pages: Vec<FacePage>,
    /// Faces per page (every page is the same size).
    page_faces: u32,
    book: MeshBookkeeping,
    /// The persistent cube metadata storage, read by the vertex shader
    /// (group 2, read-only) and the cull shader (group 1, read-write).
    meta_buffer: wgpu::Buffer,
    meta_capacity: u32,
    /// The terrain-only uniform (origin formula).
    terrain_uniforms: wgpu::Buffer,
    /// CPU-path indirect buffer (packed per page; grown like 2.2).
    indirect_buffer: wgpu::Buffer,
    indirect_capacity: u64,
    /// gpu-cull resources.
    cull_layout: wgpu::BindGroupLayout,
    cull_bind_group: wgpu::BindGroup,
    cull_early_pipeline: wgpu::ComputePipeline,
    cull_late_pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    commands_buffer: wgpu::Buffer,
    /// Command capacity in elements (each half holds `total_dirs`).
    commands_capacity: u32,
    region_base_buffer: wgpu::Buffer,
    region_counter_buffer: wgpu::Buffer,
    stats_buffer: wgpu::Buffer,
    stats_ring: StatsRing,
    hiz: Option<HizPyramid>,
    /// The 1x1 view the cull bind group binds before the first gpu-cull
    /// frame creates the real pyramid (uploads grow the region buffers,
    /// which rebuild the bind group, long before any cull dispatch).
    pyramid_placeholder: wgpu::TextureView,
    last_cull: CullStats,
    /// Whether multi-draw indirect is usable here: the adapter has
    /// `DownlevelFlags::INDIRECT_EXECUTION`, the device was created with
    /// `Features::INDIRECT_FIRST_INSTANCE`, and `JBM_NO_MDI` is unset.
    mdi: bool,
    /// Whether the gpu-cull path is usable: MDI plus
    /// `DownlevelFlags::COMPUTE_SHADERS` and R32Float storage textures,
    /// with `JBM_NO_GPU_CULL` unset.
    gpu_capable: bool,
    /// Forced cpu-cull for tests and benches (`set_force_cpu_cull`).
    force_cpu_cull: bool,
    /// Forced fallback for tests and benches (`set_force_fallback`).
    force_fallback: bool,
    stats: TerrainStats,
}

impl TerrainPass {
    pub fn new(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        textures: &BlockTextures,
    ) -> TerrainPass {
        let limits = gpu.device.limits();
        let max_faces = (limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / FACE_BYTES) as u32;
        let page_faces = DEFAULT_PAGE_FACES.min(max_faces).max(worst_case_faces());
        let pass = Self::build(gpu, color_format, uniform_layout, textures, page_faces);
        println!("terrain draw path: {}", pass.path_name());
        pass
    }

    /// Test-only constructor with an explicit (tiny) page size. The page
    /// must still hold at least one worst-case cube mesh, so a mesh larger
    /// than a whole page is impossible.
    #[cfg(any(test, feature = "testing"))]
    pub fn with_page_faces(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        textures: &BlockTextures,
        page_faces: u32,
    ) -> TerrainPass {
        assert!(
            page_faces >= worst_case_faces(),
            "a page must hold one worst-case cube mesh ({} faces)",
            worst_case_faces()
        );
        Self::build(gpu, color_format, uniform_layout, textures, page_faces)
    }

    fn build(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        textures: &BlockTextures,
        page_faces: u32,
    ) -> TerrainPass {
        let device = &gpu.device;

        // Group 1, per page: the faces storage buffer.
        let faces_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jbm terrain faces layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Group 2, shared: meta, the terrain uniform, textures, sampler.
        let shared_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jbm terrain shared layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let make_meta_buffer = |device: &wgpu::Device, slots: u32| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jbm terrain cube meta"),
                size: u64::from(slots) * std::mem::size_of::<CubeMeta>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let meta_buffer = make_meta_buffer(device, META_CAPACITY);
        let texture_view = textures.view().clone();
        let sampler = textures.sampler().clone();
        let terrain_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain uniforms"),
            size: std::mem::size_of::<TerrainUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shared_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jbm terrain shared"),
            layout: &shared_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: terrain_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(textures.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(textures.sampler()),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm terrain"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jbm terrain layout"),
            bind_group_layouts: &[
                Some(uniform_layout),
                Some(&faces_layout),
                Some(&shared_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jbm terrain pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[], // vertex pulling: no vertex buffers
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // The cull pass: group 0 is the frame uniforms (view_proj), group 1
        // the cull resources (see shaders/cull.wgsl).
        let cull_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jbm cull layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm cull"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cull.wgsl").into()),
        });
        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jbm cull layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&cull_layout)],
            immediate_size: 0,
        });
        let make_cull_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jbm cull pipeline"),
                layout: Some(&cull_pipeline_layout),
                module: &cull_shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let cull_early_pipeline = make_cull_pipeline("cull_early");
        let cull_late_pipeline = make_cull_pipeline("cull_late");

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm cull params"),
            size: std::mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let commands_capacity = 256u32;
        // STORAGE: the cull shader appends; INDIRECT: the draws; COPY_DST:
        // the per-frame clear_buffer (zero tails).
        let commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain gpu commands"),
            size: u64::from(commands_capacity) * CMD_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One u32 per page (grown on page creation). COPY_DST serves both
        // the write_buffer updates (base) and the per-frame clear_buffer
        // reset (counters).
        let make_region_buffer = |label: &'static str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let region_base_buffer = make_region_buffer("jbm region base");
        let region_counter_buffer = make_region_buffer("jbm region counters");
        let stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm cull stats"),
            size: STATS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats_ring = StatsRing::new(device);

        // A 1x1 placeholder pyramid view so the cull bind group always has
        // a texture bound; the real pyramid replaces it on the first
        // gpu-cull frame (before any cull dispatch happens).
        let placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jbm hiz placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let placeholder_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());

        let downlevel = gpu.adapter.get_downlevel_capabilities();
        let mdi = downlevel
            .flags
            .contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            && gpu
                .device
                .features()
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            && !std::env::var_os("JBM_NO_MDI").is_some_and(|v| v == "1");
        // R32Float write-only storage textures carry the pyramid; check
        // the adapter instead of trusting the format table.
        let r32f_storage = gpu
            .adapter
            .get_texture_format_features(wgpu::TextureFormat::R32Float)
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY);
        let gpu_capable = mdi
            && r32f_storage
            && downlevel
                .flags
                .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            && !std::env::var_os("JBM_NO_GPU_CULL").is_some_and(|v| v == "1");

        let cull_bind_group = Self::make_cull_bind_group(
            device,
            &cull_layout,
            &meta_buffer,
            &params_buffer,
            &commands_buffer,
            &region_base_buffer,
            &region_counter_buffer,
            &stats_buffer,
            &placeholder_view,
        );

        TerrainPass {
            device: gpu.device.clone(),
            pipeline,
            faces_layout,
            shared_layout,
            shared_bind_group,
            texture_view,
            sampler,
            pages: Vec::new(),
            page_faces,
            book: MeshBookkeeping::default(),
            meta_buffer,
            meta_capacity: META_CAPACITY,
            terrain_uniforms,
            indirect_buffer: Self::make_indirect_buffer(device, INDIRECT_CAPACITY),
            indirect_capacity: INDIRECT_CAPACITY,
            cull_layout,
            cull_bind_group,
            cull_early_pipeline,
            cull_late_pipeline,
            params_buffer,
            commands_buffer,
            commands_capacity,
            region_base_buffer,
            region_counter_buffer,
            stats_buffer,
            stats_ring,
            hiz: None,
            pyramid_placeholder: placeholder_view,
            last_cull: CullStats::default(),
            mdi,
            gpu_capable,
            force_cpu_cull: false,
            force_fallback: false,
            stats: TerrainStats::default(),
        }
    }

    /// The cull bind group: everything but the frame uniforms (group 0).
    /// Rebuilt whenever any of its resources is recreated (meta growth,
    /// command growth, region growth, pyramid recreation).
    // The seven parameters are the bind group's own entries; a bundling
    // struct would only be unwrapped again here.
    #[allow(clippy::too_many_arguments)]
    fn make_cull_bind_group(
        device: &wgpu::Device,
        cull_layout: &wgpu::BindGroupLayout,
        meta_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        commands_buffer: &wgpu::Buffer,
        region_base_buffer: &wgpu::Buffer,
        region_counter_buffer: &wgpu::Buffer,
        stats_buffer: &wgpu::Buffer,
        pyramid: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jbm cull"),
            layout: cull_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: commands_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: region_base_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: region_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: stats_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(pyramid),
                },
            ],
        })
    }

    fn rebuild_cull_bind_group(&mut self) {
        let pyramid = self
            .hiz
            .as_ref()
            .map_or(&self.pyramid_placeholder, |h| h.view());
        self.cull_bind_group = Self::make_cull_bind_group(
            &self.device,
            &self.cull_layout,
            &self.meta_buffer,
            &self.params_buffer,
            &self.commands_buffer,
            &self.region_base_buffer,
            &self.region_counter_buffer,
            &self.stats_buffer,
            pyramid,
        );
    }

    fn make_indirect_buffer(device: &wgpu::Device, commands: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain indirect"),
            size: commands * CMD_BYTES,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_page(&self, gpu: &Gpu) -> FacePage {
        let device = &gpu.device;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain face page"),
            size: u64::from(self.page_faces) * FACE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jbm terrain face page"),
            layout: &self.faces_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        FacePage {
            buffer,
            suballoc: Suballocator::new(self.page_faces),
            bind_group,
        }
    }

    /// First page that fits, else a new one (plus grown region buffers,
    /// which rebuild the cull bind group and need the pyramid — so the
    /// pyramid placeholder must still be alive; the first page is created
    /// by the first upload, long before the first draw). A mesh larger
    /// than a whole page is impossible: the constructor asserts pages hold
    /// at least one worst-case cube mesh.
    fn alloc_slot(&mut self, gpu: &Gpu, len: u32) -> (usize, Range<u32>) {
        let page = match self
            .pages
            .iter()
            .position(|p| p.suballoc.largest_free() >= len)
        {
            Some(page) => page,
            None => {
                let page = self.pages.len();
                let new_page = self.make_page(gpu);
                self.pages.push(new_page);
                self.book.page_added();
                self.grow_region_buffers();
                page
            }
        };
        let range = self.pages[page]
            .suballoc
            .alloc(len)
            .expect("a page chosen by largest_free must fit the mesh");
        (page, range)
    }

    /// The region buffers are one u32 per page; they grow when a page is
    /// added (pages are never freed).
    fn grow_region_buffers(&mut self) {
        let needed = (self.pages.len().max(4) as u64).next_power_of_two() * 4;
        if self.region_base_buffer.size() >= needed {
            return;
        }
        let make = |label: &'static str| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: needed,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        self.region_base_buffer = make("jbm region base");
        self.region_counter_buffer = make("jbm region counters");
        self.rebuild_cull_bind_group();
    }

    /// Replaces the mesh of one cube. A re-mesh frees the cube's old range,
    /// rewrites its meta slot in place (with the visible bit reset, so the
    /// late cull picks the new geometry up in the same frame) and never
    /// creates a buffer (except when no page fits, which adds one page). An
    /// empty mesh allocates nothing and holds no slot. The app enforces
    /// the per-frame upload budget before calling this.
    pub fn upload(&mut self, gpu: &Gpu, pos: CubePos, output: &MeshOutput) {
        let released = self.book.release(&pos);
        if let (Some(page), Some(range)) = (released.page, released.range) {
            self.pages[page].suballoc.free(range);
        }
        if output.is_empty() {
            // Keep the entry (the cube is meshed) but hold no allocation
            // and no slot.
            if let Some(slot) = released.slot {
                self.zero_meta_slot(gpu, slot);
                self.book.free_meta_slot(slot);
            }
            self.book.commit_empty(pos);
            return;
        }
        let len = output.faces.len() as u32;
        let (page, range) = self.alloc_slot(gpu, len);
        // Freed ranges are safe to reuse immediately: per the wgpu 30
        // `Queue::write_buffer` docs, "Calls to `write_buffer()` do *not*
        // submit the transfer to the GPU immediately. They begin GPU
        // execution only on the next call to `Queue::submit()`, just before
        // the explicitly submitted commands." Every earlier frame's draws
        // were in submits that already enqueued their own copies of the
        // data, so this write cannot race them.
        gpu.queue.write_buffer(
            &self.pages[page].buffer,
            u64::from(range.start) * FACE_BYTES,
            bytemuck::cast_slice(&output.faces),
        );
        let slot = self.book.commit(
            pos,
            page,
            range.clone(),
            output.ranges.clone(),
            released.slot,
        );
        if slot >= self.meta_capacity {
            self.grow_meta_buffer(gpu);
        }
        gpu.queue.write_buffer(
            &self.meta_buffer,
            u64::from(slot) * std::mem::size_of::<CubeMeta>() as u64,
            bytemuck::bytes_of(&meta_for(pos, page, range.start, &output.ranges)),
        );
    }

    /// Drops a cube's mesh (UnloadCube), freeing its page range and meta
    /// slot. Pages themselves are never freed in this task.
    pub fn remove(&mut self, gpu: &Gpu, pos: CubePos) {
        let released = self.book.release(&pos);
        if let (Some(page), Some(range)) = (released.page, released.range) {
            self.pages[page].suballoc.free(range);
        }
        if let Some(slot) = released.slot {
            self.zero_meta_slot(gpu, slot);
            self.book.free_meta_slot(slot);
        }
        self.book.meshes.remove(&pos);
    }

    /// A freed slot must read inactive: the cull threads dispatch over the
    /// whole high-water mark.
    fn zero_meta_slot(&self, gpu: &Gpu, slot: u32) {
        gpu.queue.write_buffer(
            &self.meta_buffer,
            u64::from(slot) * std::mem::size_of::<CubeMeta>() as u64,
            bytemuck::bytes_of(&CubeMeta::zeroed()),
        );
    }

    /// Grows the meta buffer (rare): a new buffer, a new cull bind group,
    /// a new shared bind group, and every live meta rewritten into it
    /// (each cube keeps its slot).
    fn grow_meta_buffer(&mut self, gpu: &Gpu) {
        self.meta_capacity *= 2;
        self.meta_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain cube meta"),
            size: u64::from(self.meta_capacity) * std::mem::size_of::<CubeMeta>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut metas = vec![CubeMeta::zeroed(); self.meta_capacity as usize];
        for (pos, mesh) in &self.book.meshes {
            let Some(slot) = mesh.slot else { continue };
            let (page, range) = mesh
                .alloc
                .as_ref()
                .expect("a mesh with a slot holds an allocation");
            metas[slot as usize] = meta_for(*pos, *page, range.start, &mesh.ranges);
        }
        gpu.queue
            .write_buffer(&self.meta_buffer, 0, bytemuck::cast_slice(&metas));
        self.rebuild_shared_bind_group(gpu);
        self.rebuild_cull_bind_group();
    }

    fn rebuild_shared_bind_group(&mut self, gpu: &Gpu) {
        self.shared_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jbm terrain shared"),
            layout: &self.shared_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.terrain_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    pub fn stats(&self) -> TerrainStats {
        self.stats
    }

    /// Whether the next draw issues multi_draw_indirect calls (the gpu-cull
    /// and cpu-cull paths), as opposed to the direct-draw fallback.
    pub fn uses_mdi(&self) -> bool {
        self.active_path() != TerrainPath::Fallback
    }

    /// Whether the hardware and environment allow the gpu-cull path at all
    /// (ignoring the forced-path setters).
    pub fn gpu_cull_available(&self) -> bool {
        self.gpu_capable
    }

    /// The path the next draw uses.
    pub fn path(&self) -> TerrainPath {
        self.active_path()
    }

    /// The path name shown in the overlay and logs.
    pub fn path_name(&self) -> &'static str {
        match self.active_path() {
            TerrainPath::GpuCull => "gpu-cull",
            TerrainPath::CpuCull => "cpu-cull",
            TerrainPath::Fallback => "fallback",
        }
    }

    fn active_path(&self) -> TerrainPath {
        if !self.mdi || self.force_fallback {
            TerrainPath::Fallback
        } else if self.force_cpu_cull || !self.gpu_capable {
            TerrainPath::CpuCull
        } else {
            TerrainPath::GpuCull
        }
    }

    /// Forces the cpu-cull path (or restores the automatic choice), for
    /// tests and benches. Falls back to direct draws when the adapter
    /// lacks MDI.
    pub fn set_force_cpu_cull(&mut self, force: bool) {
        self.force_cpu_cull = force;
    }

    /// Forces the direct-draw fallback path (or restores the automatic
    /// choice), for tests and benches. All paths share the same buffers
    /// and commands, so the image is identical.
    pub fn set_force_fallback(&mut self, force: bool) {
        self.force_fallback = force;
    }

    /// The culling counters for the overlay. On gpu-cull these are the GPU
    /// atomics read back through a 3-deep ring, so they are one or two
    /// frames late; on the CPU paths they are exact and immediate.
    pub fn cull_stats(&self) -> CullStats {
        match self.active_path() {
            TerrainPath::GpuCull => self.stats_ring.latest,
            _ => self.last_cull,
        }
    }

    /// Waits for the current frame's GPU counters and returns them (tests,
    /// benches and the metrics example); the GPU-derived fields of
    /// [`TerrainStats`](Self::stats) follow. On the CPU paths it returns the
    /// immediate counters.
    pub fn read_cull_stats_blocking(&mut self, gpu: &Gpu) -> CullStats {
        if self.active_path() == TerrainPath::GpuCull {
            self.last_cull = self.stats_ring.read_blocking(gpu);
            self.stats.drawn = self.last_cull.drawn() as usize;
            self.stats.faces = u64::from(self.last_cull.faces);
            self.stats.commands = u64::from(self.last_cull.commands);
        }
        self.last_cull
    }

    /// Runs [`Suballocator::check_invariants`] on every page, plus the
    /// meta slot and region bound invariants. Test-only.
    #[cfg(any(test, feature = "testing"))]
    pub fn check_page_invariants(&self) {
        for page in &self.pages {
            page.suballoc.check_invariants();
        }
        self.book.check_invariants();
    }

    /// The per-page command-region bound (test-only; the churn test checks
    /// it against a recount).
    #[cfg(any(test, feature = "testing"))]
    pub fn page_command_bounds(&self) -> Vec<u32> {
        self.book.page_dir_counts.clone()
    }

    /// Command-region slots encoded across both phases for `camera`,
    /// including the zero-count tails (the metrics example): twice the
    /// CPU frustum bound the frame draws with.
    pub fn region_command_slots(&self, camera: &Camera) -> u64 {
        let (camera_cube, camera_frac) = camera_split(camera.position);
        let frustum = Frustum::from_view_proj(&camera.view_proj());
        let total: u32 = self
            .book
            .frustum_page_bounds(camera_cube, camera_frac, &frustum)
            .iter()
            .sum();
        u64::from(total) * 2
    }

    /// Total faces across all uploaded meshes.
    pub fn total_faces(&self) -> u64 {
        self.book
            .meshes
            .values()
            .map(|m| u64::from(m.face_count))
            .sum()
    }

    pub fn meshed_count(&self) -> usize {
        self.book.meshes.len()
    }

    /// Face pages allocated so far (pages are never freed).
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Records the terrain draw on the chosen path. See the path table in
    /// the README; `draw_gpu_cull` and `draw_cpu_cull` below.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &DepthTarget,
        camera: &Camera,
        uniforms: &wgpu::BindGroup,
    ) {
        match self.active_path() {
            TerrainPath::GpuCull => {
                self.draw_gpu_cull(gpu, encoder, target, depth, camera, uniforms)
            }
            TerrainPath::CpuCull | TerrainPath::Fallback => {
                self.draw_cpu_cull(gpu, encoder, target, depth, camera, uniforms)
            }
        }
    }

    /// The gpu-cull frame: the GPU tests, emits and draws; the CPU's only
    /// per-cube work is the frustum pass that sizes the command regions.
    /// Phase order: early cull/draw, then Hi-Z build and late cull in one
    /// compute pass, then the late draw.
    fn draw_gpu_cull(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &DepthTarget,
        camera: &Camera,
        uniforms: &wgpu::BindGroup,
    ) {
        // Complete any stats readback from earlier frames (non-blocking)
        // and arm the maps for the slots copied last frame.
        self.stats_ring.begin_frame(gpu);

        let resident_dirs = self.book.total_dirs();
        let mut stats = self.page_stats(TerrainStats {
            meshed: self.book.meshes.len(),
            drawn: 0,
            faces: 0,
            commands: 0,
            draw_calls: 0,
            pages: self.pages.len(),
            page_bytes: (self.pages.len() as u64) * u64::from(self.page_faces) * FACE_BYTES,
            used_faces: 0,
        });
        if resident_dirs == 0 {
            self.stats = stats;
            self.last_cull = CullStats::default();
            return;
        }

        // The shared origin uniform (also read by the vertex shader).
        let (camera_cube, camera_frac) = camera_split(camera.position);
        let terrain_uniforms = TerrainUniforms {
            camera_cube: [camera_cube.x, camera_cube.y, camera_cube.z, 0],
            camera_frac: [
                camera_frac.x,
                camera_frac.y,
                camera_frac.z,
                CUBE_SIZE as f32,
            ],
        };
        gpu.queue.write_buffer(
            &self.terrain_uniforms,
            0,
            bytemuck::bytes_of(&terrain_uniforms),
        );

        // The pyramid is recreated together with the depth target (size or
        // generation change).
        let (width, height, generation) = depth.description();
        if self
            .hiz
            .as_ref()
            .is_none_or(|h| !h.matches_depth(width, height, generation))
        {
            self.hiz = Some(HizPyramid::new(
                gpu,
                depth.view(),
                width,
                height,
                generation,
            ));
            self.rebuild_cull_bind_group();
        }

        // Grow the command buffer when the bound outgrew it.
        let frustum = Frustum::from_view_proj(&camera.view_proj());
        let page_bounds = self
            .book
            .frustum_page_bounds(camera_cube, camera_frac, &frustum);
        let mut region_base = Vec::with_capacity(page_bounds.len());
        let mut bound_total = 0u32;
        for &bound in &page_bounds {
            region_base.push(bound_total);
            bound_total += bound;
        }
        if 2 * bound_total > self.commands_capacity {
            self.commands_capacity = (2 * bound_total).next_power_of_two();
            self.commands_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jbm terrain gpu commands"),
                size: u64::from(self.commands_capacity) * CMD_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.rebuild_cull_bind_group();
        }

        // region_base: the prefix sum over pages of this frame's CPU
        // frustum bound, rewritten each frame (a tiny buffer; the CPU
        // already knows the bound).
        let max_level = self.hiz.as_ref().expect("created above").max_level();
        gpu.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&CullParams {
                camera_cube: terrain_uniforms.camera_cube,
                camera_frac: terrain_uniforms.camera_frac,
                screen: [width, height, max_level, self.book.meta_len],
                regions: [
                    bound_total,
                    self.pages.len() as u32,
                    self.commands_capacity,
                    CUBE_SIZE as u32,
                ],
            }),
        );
        gpu.queue.write_buffer(
            &self.region_base_buffer,
            0,
            bytemuck::cast_slice(&region_base),
        );

        // Zero the counters, the stats and the command regions so a
        // region's unused tail is zero-count draws. Only the prefix this
        // frame's draws read needs clearing; slots past it are never
        // referenced. clear_buffer requires COPY_DST (see the buffer
        // usages above).
        encoder.clear_buffer(&self.region_counter_buffer, 0, None);
        encoder.clear_buffer(&self.stats_buffer, 0, None);
        if bound_total > 0 {
            encoder.clear_buffer(
                &self.commands_buffer,
                0,
                Some(u64::from(bound_total) * 2 * CMD_BYTES),
            );
        }

        // 1. Early cull: what was visible last frame and still passes the
        //    frustum.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jbm terrain early cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_early_pipeline);
            pass.set_bind_group(0, uniforms, &[]);
            pass.set_bind_group(1, &self.cull_bind_group, &[]);
            pass.dispatch_workgroups(self.book.meta_len.div_ceil(64), 1, 1);
        }

        // 2. Early draw.
        let mut draw_calls = 0u64;
        {
            let mut pass = self.begin_pass(encoder, target, depth.view());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, uniforms, &[]);
            pass.set_bind_group(2, &self.shared_bind_group, &[]);
            draw_calls += self.draw_regions(&mut pass, &region_base, &page_bounds, 0);
        }

        // 3+4. The Hi-Z pyramid from the depth the early draw left, then
        //    the late cull against it, in one compute pass: dispatches
        //    within a pass are ordered, and each dispatch is its own usage
        //    scope (WebGPU spec §3.4.4: "In a compute pass, each dispatch
        //    command is one usage scope"), so wgpu validates the mip k ->
        //    mip k+1 write->read transitions per dispatch and inserts the
        //    barrier between them; the pyramid dispatches ending in
        //    storage writes then the cull dispatch sampling the whole
        //    pyramid transition the same way.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jbm terrain hiz + late cull"),
                timestamp_writes: None,
            });
            self.hiz.as_ref().expect("created above").record(&mut pass);
            pass.set_pipeline(&self.cull_late_pipeline);
            pass.set_bind_group(0, uniforms, &[]);
            pass.set_bind_group(1, &self.cull_bind_group, &[]);
            pass.dispatch_workgroups(self.book.meta_len.div_ceil(64), 1, 1);
        }

        // 5. Late draw.
        {
            let mut pass = self.begin_pass(encoder, target, depth.view());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, uniforms, &[]);
            pass.set_bind_group(2, &self.shared_bind_group, &[]);
            draw_calls += self.draw_regions(&mut pass, &region_base, &page_bounds, bound_total);
        }

        // 6. The counters for the overlay, one or two frames late.
        if let Some(slot) = self.stats_ring.take_slot() {
            encoder.copy_buffer_to_buffer(
                &self.stats_buffer,
                0,
                &self.stats_ring.buffers[slot],
                0,
                STATS_BYTES,
            );
        }

        let cull = self.stats_ring.latest;
        stats.drawn = cull.drawn() as usize;
        stats.faces = u64::from(cull.faces);
        stats.commands = u64::from(cull.commands);
        stats.draw_calls = draw_calls;
        self.stats = stats;
        self.last_cull = cull;
    }

    /// Begins one terrain render pass; every phase loads the sky's color
    /// and depth.
    fn begin_pass<'a>(
        &self,
        encoder: &'a mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("jbm terrain pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// One multi_draw_indirect per page over its region of `phase_base`.
    /// The count is this frame's CPU frustum bound for the page: slots the
    /// shader did not fill were cleared to zero-count draws, which are
    /// no-ops everywhere (the reason `multi_draw_indirect_count` is not
    /// used; Metal lacks it, and its MDI is a CPU loop over the encoded
    /// count, so the count must come from the CPU anyway).
    fn draw_regions(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        region_base: &[u32],
        page_bounds: &[u32],
        phase_base: u32,
    ) -> u64 {
        let mut calls = 0u64;
        for (page, &count) in page_bounds.iter().enumerate() {
            if count == 0 {
                continue;
            }
            pass.set_bind_group(1, &self.pages[page].bind_group, &[]);
            let offset = u64::from(phase_base + region_base[page]) * CMD_BYTES;
            pass.multi_draw_indirect(&self.commands_buffer, offset, count);
            calls += 1;
        }
        calls
    }

    /// The cpu-cull frame (also the fallback path's culling): 2.2's CPU
    /// frustum and direction tests over the same meta slots, with the
    /// shared origin formula, so the CPU and GPU cull on the same numbers.
    fn draw_cpu_cull(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &DepthTarget,
        camera: &Camera,
        uniforms: &wgpu::BindGroup,
    ) {
        let (camera_cube, camera_frac) = camera_split(camera.position);
        gpu.queue.write_buffer(
            &self.terrain_uniforms,
            0,
            bytemuck::bytes_of(&TerrainUniforms {
                camera_cube: [camera_cube.x, camera_cube.y, camera_cube.z, 0],
                camera_frac: [
                    camera_frac.x,
                    camera_frac.y,
                    camera_frac.z,
                    CUBE_SIZE as f32,
                ],
            }),
        );

        let frustum = Frustum::from_view_proj(&camera.view_proj());
        // Deterministic order: HashMap iteration is randomized.
        let mut visible: Vec<(CubePos, Vec3, &CubeMesh)> = self
            .book
            .meshes
            .iter()
            .filter(|(_, mesh)| mesh.face_count > 0)
            .filter_map(|(pos, mesh)| {
                let min = cube_origin(*pos, camera_cube, camera_frac);
                let max = min + Vec3::splat(CUBE_SIZE as f32);
                frustum
                    .intersects_aabb(min, max)
                    .then_some((*pos, min, mesh))
            })
            .collect();
        visible.sort_unstable_by_key(|(pos, _, _)| *pos);

        // The CPU-side mirror of the GPU counters: every non-empty mesh is
        // a candidate, the frustum test culls the rest (there is no
        // occlusion culling on this path).
        let candidates = self
            .book
            .meshes
            .values()
            .filter(|m| m.face_count > 0)
            .count() as u32;

        let mut stats = self.page_stats(TerrainStats {
            meshed: self.book.meshes.len(),
            drawn: visible.len(),
            faces: 0,
            commands: 0,
            draw_calls: 0,
            pages: self.pages.len(),
            page_bytes: (self.pages.len() as u64) * u64::from(self.page_faces) * FACE_BYTES,
            used_faces: 0,
        });

        // One command per visible cube and face direction with a non-empty,
        // non-skipped range, grouped by page.
        let mut per_page: Vec<Vec<wgpu::util::DrawIndirectArgs>> =
            self.pages.iter().map(|_| Vec::new()).collect();
        let mut faces = 0u64;
        for (_, rel_min, mesh) in &visible {
            let Some((page, alloc_range)) = mesh.alloc.as_ref() else {
                continue;
            };
            let dirs = visible_directions(*rel_min, *rel_min + Vec3::splat(CUBE_SIZE as f32));
            for (dir, dir_range) in mesh.ranges.iter().enumerate() {
                if dir_range.is_empty() || !dirs[dir] {
                    continue;
                }
                per_page[*page].push(wgpu::util::DrawIndirectArgs {
                    vertex_count: dir_range.len() as u32 * 6,
                    instance_count: 1,
                    first_vertex: (alloc_range.start + dir_range.start) * 6,
                    first_instance: mesh.slot.expect("a drawn cube holds a meta slot"),
                });
                faces += dir_range.len() as u64;
            }
        }
        let mut commands: Vec<wgpu::util::DrawIndirectArgs> = Vec::new();
        // (page, command count) for every page that draws this frame.
        let mut page_spans: Vec<(usize, u32)> = Vec::new();
        for (page, cmds) in per_page.iter().enumerate() {
            if cmds.is_empty() {
                continue;
            }
            page_spans.push((page, cmds.len() as u32));
            commands.extend_from_slice(cmds);
        }
        stats.faces = faces;
        stats.commands = commands.len() as u64;

        self.last_cull = CullStats {
            candidates,
            frustum_culled: candidates - visible.len() as u32,
            occluded: 0,
            drawn_early: visible.len() as u32,
            drawn_late: 0,
            commands: commands.len() as u32,
            faces: faces as u32,
            region_overflow: 0,
        };

        if commands.is_empty() {
            self.stats = stats;
            return;
        }
        if commands.len() as u64 > self.indirect_capacity {
            // Grow (rare), like the pages.
            let capacity = (commands.len() as u64).next_power_of_two();
            self.indirect_buffer = Self::make_indirect_buffer(&gpu.device, capacity);
            self.indirect_capacity = capacity;
        }
        gpu.queue
            .write_buffer(&self.indirect_buffer, 0, bytemuck::cast_slice(&commands));

        let mut pass = self.begin_pass(encoder, target, depth.view());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, uniforms, &[]);
        pass.set_bind_group(2, &self.shared_bind_group, &[]);
        let mut draw_calls = 0u64;
        if self.uses_mdi() {
            let mut offset = 0u64;
            for &(page, count) in &page_spans {
                pass.set_bind_group(1, &self.pages[page].bind_group, &[]);
                pass.multi_draw_indirect(&self.indirect_buffer, offset * CMD_BYTES, count);
                offset += u64::from(count);
                draw_calls += 1;
            }
        } else {
            // Fallback: the same commands and buffers as plain direct draws.
            // Direct draws take a first instance without the feature.
            let mut cursor = 0usize;
            for &(page, count) in &page_spans {
                pass.set_bind_group(1, &self.pages[page].bind_group, &[]);
                for command in &commands[cursor..cursor + count as usize] {
                    let first_vertex = command.first_vertex;
                    let first_instance = command.first_instance;
                    pass.draw(
                        first_vertex..first_vertex + command.vertex_count,
                        first_instance..first_instance + command.instance_count,
                    );
                }
                cursor += count as usize;
                draw_calls += u64::from(count);
            }
        }
        drop(pass);

        stats.draw_calls = draw_calls;
        self.stats = stats;
    }

    /// Fills the page-derived field (`used_faces`) of a stats snapshot.
    fn page_stats(&self, mut stats: TerrainStats) -> TerrainStats {
        stats.used_faces = self
            .pages
            .iter()
            .map(|p| u64::from(p.suballoc.used()))
            .sum();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0)
    }

    #[test]
    fn aabb_in_front_is_visible() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        // Directly ahead of the camera (yaw 0 looks toward -Z).
        assert!(frustum.intersects_aabb(Vec3::new(-4.0, -4.0, -36.0), Vec3::new(4.0, 4.0, -4.0)));
    }

    #[test]
    fn aabb_behind_the_camera_is_culled() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        assert!(!frustum.intersects_aabb(Vec3::new(-4.0, -4.0, 4.0), Vec3::new(4.0, 4.0, 36.0)));
    }

    #[test]
    fn aabb_far_to_the_side_is_culled() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        // Well outside the 70-degree fov on the +X side, ahead in z.
        assert!(
            !frustum.intersects_aabb(Vec3::new(200.0, -8.0, -60.0), Vec3::new(232.0, 8.0, -28.0))
        );
        // ... and the mirror on -X.
        assert!(!frustum.intersects_aabb(
            Vec3::new(-232.0, -8.0, -60.0),
            Vec3::new(-200.0, 8.0, -28.0)
        ));
    }

    #[test]
    fn aabb_straddling_the_near_plane_stays_visible() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        // Crossing the near plane (and the camera itself) from well in
        // front to behind counts as visible (conservative).
        assert!(frustum.intersects_aabb(Vec3::new(-1.0, -1.0, -20.0), Vec3::new(1.0, 1.0, 20.0)));
    }

    #[test]
    fn cube_containing_the_camera_shows_all_six_directions() {
        assert!(
            visible_directions(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0))
                .iter()
                .all(|&visible| visible)
        );
        // A face plane running exactly through the camera is edge-on: the
        // cube touching the origin on a corner shows only the three sides
        // facing it.
        assert_eq!(
            visible_directions(Vec3::ZERO, Vec3::splat(1.0)),
            [false, true, false, true, false, true]
        );
        assert_eq!(
            visible_directions(Vec3::splat(-1.0), Vec3::ZERO),
            [true, false, true, false, true, false]
        );
    }

    #[test]
    fn cube_in_front_and_behind_the_camera() {
        // In front (yaw 0 looks toward -Z): -Z faces away.
        assert_eq!(
            visible_directions(Vec3::new(-1.0, -1.0, -5.0), Vec3::new(1.0, 1.0, -1.0)),
            [true, true, true, true, true, false]
        );
        // Behind: +Z faces away.
        assert_eq!(
            visible_directions(Vec3::new(-1.0, -1.0, 2.0), Vec3::new(1.0, 1.0, 6.0)),
            [true, true, true, true, false, true]
        );
    }

    #[test]
    fn cube_straddling_each_axis() {
        // X only: +X and -X both stay; the cube is above and in front, so
        // +Y and -Z face away.
        assert_eq!(
            visible_directions(Vec3::new(-1.0, 2.0, -5.0), Vec3::new(1.0, 4.0, -2.0)),
            [true, true, false, true, true, false]
        );
        // Y only.
        assert_eq!(
            visible_directions(Vec3::new(2.0, -1.0, -5.0), Vec3::new(4.0, 1.0, -2.0)),
            [false, true, true, true, true, false]
        );
        // Z only (cube behind on x).
        assert_eq!(
            visible_directions(Vec3::new(2.0, 2.0, -1.0), Vec3::new(4.0, 4.0, 1.0)),
            [false, true, false, true, true, true]
        );
        // Fully in the (+, +, +) octant: only the camera-facing sides show.
        assert_eq!(
            visible_directions(Vec3::new(2.0, 2.0, 2.0), Vec3::new(4.0, 4.0, 4.0)),
            [false, true, false, true, false, true]
        );
    }

    #[test]
    fn worst_case_faces_matches_the_checkerboard_bound() {
        let volume = (CUBE_SIZE as u64).pow(3);
        assert_eq!(u64::from(worst_case_faces()), 6 * volume / 2);
    }

    /// One f32 ulp (relative), with an absolute floor near zero.
    fn ulp(x: f32) -> f32 {
        f32::EPSILON * x.abs().max(1.0)
    }

    /// The origin formula against `Camera::relative` (f64 subtraction),
    /// within one ulp of the result, for one cube near the camera.
    fn check_origin(camera_pos: DVec3, pos: CubePos) {
        let camera = Camera::new(camera_pos, 70.0_f32.to_radians(), 1.0);
        let (cube, frac) = camera_split(camera_pos);
        let origin = cube_origin(pos, cube, frac);
        let world_min = DVec3::new(
            f64::from(pos.x) * f64::from(CUBE_SIZE),
            f64::from(pos.y) * f64::from(CUBE_SIZE),
            f64::from(pos.z) * f64::from(CUBE_SIZE),
        );
        let reference = camera.relative(world_min);
        for i in 0..3 {
            assert!(
                (origin[i] - reference[i]).abs() <= ulp(reference[i]),
                "camera {camera_pos:?} cube {pos:?}: origin {origin:?} vs reference {reference:?}"
            );
        }
    }

    #[test]
    fn origin_formula_matches_camera_relative() {
        // Near the origin, near negative borders, and around +-1e6 blocks,
        // on and off exact cube borders (including negative).
        for base in [
            DVec3::ZERO,
            DVec3::new(-1.0e6, 999_999.75, -1.0e6 + 16.5),
            DVec3::new(1.0e6 + 0.125, -1.0e6, 1.0e6 - 8.0),
        ] {
            for delta in [0.0, f64::from(CUBE_SIZE) / 2.0, f64::from(CUBE_SIZE)] {
                let camera = base + DVec3::new(delta, -delta, delta);
                // The cube containing the camera and its neighbors.
                let (cube, _) = camera_split(camera);
                for dx in -1..=1i32 {
                    for dy in -1..=1i32 {
                        for dz in -1..=1i32 {
                            check_origin(
                                camera,
                                CubePos {
                                    x: cube.x + dx,
                                    y: cube.y + dy,
                                    z: cube.z + dz,
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn camera_cube_floor_handles_negative_cameras() {
        // Exactly on a negative cube border: floor, not truncate.
        let (cube, frac) = camera_split(DVec3::new(-64.0, -32.0, -32.0));
        assert_eq!(cube, IVec3::new(-2, -1, -1));
        assert_eq!(frac, Vec3::ZERO);
        // One block before that border, and a negative in-cube position.
        let (cube, frac) = camera_split(DVec3::new(-64.5, 0.0, -1.0));
        assert_eq!(cube, IVec3::new(-3, 0, -1));
        assert_eq!(frac, Vec3::new(31.5, 0.0, 31.0));
        let (cube, frac) = camera_split(DVec3::new(-64.5, 0.0, 33.0));
        assert_eq!(cube, IVec3::new(-3, 0, 1));
        assert_eq!(frac, Vec3::new(31.5, 0.0, 1.0));
    }

    #[test]
    fn cube_meta_layout_matches_wgsl() {
        // The WGSL mirror (terrain.wgsl / cull.wgsl) is vec4i + vec4u +
        // array<vec4u, 3>: 80 bytes, members at 0 / 16 / 32.
        assert_eq!(std::mem::size_of::<CubeMeta>(), 80);
        assert_eq!(std::mem::offset_of!(CubeMeta, pos_page), 0);
        assert_eq!(std::mem::offset_of!(CubeMeta, start_flags), 16);
        assert_eq!(std::mem::offset_of!(CubeMeta, dirs), 32);
        assert!(
            std::mem::size_of::<CubeMeta>().is_multiple_of(16),
            "no tail padding: the storage element stride is the struct size"
        );
    }

    #[test]
    fn terrain_uniforms_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<TerrainUniforms>(), 32);
        assert_eq!(std::mem::offset_of!(TerrainUniforms, camera_cube), 0);
        assert_eq!(std::mem::offset_of!(TerrainUniforms, camera_frac), 16);
    }

    #[test]
    fn cull_params_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<CullParams>(), 64);
        assert_eq!(std::mem::offset_of!(CullParams, camera_cube), 0);
        assert_eq!(std::mem::offset_of!(CullParams, camera_frac), 16);
        assert_eq!(std::mem::offset_of!(CullParams, screen), 32);
        assert_eq!(std::mem::offset_of!(CullParams, regions), 48);
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
        book.commit(CubePos { x: 1, y: 1, z: -2 }, 0, 0..18, six_ranges(), None);
        // Far outside the 70-degree fov on +X, ahead in z: dropped. Page 0.
        book.commit(
            CubePos { x: 12, y: 0, z: -1 },
            0,
            18..36,
            six_ranges(),
            None,
        );
        // Behind the camera (yaw 0). Page 1.
        book.commit(CubePos { x: 0, y: 0, z: 2 }, 1, 36..54, six_ranges(), None);
        // The cube containing the camera: all six directions, in any
        // frustum. Page 1.
        book.commit(CubePos { x: 0, y: 0, z: 0 }, 1, 54..72, six_ranges(), None);
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
        book.commit(CubePos { x: 0, y: 0, z: 0 }, 0, 0..6, sparse, None);
        let bounds =
            book.frustum_page_bounds(cube, frac, &Frustum::from_view_proj(&cam.view_proj()));
        assert_eq!(bounds, vec![4], "only the four non-empty directions count");
    }

    #[test]
    fn meta_for_packs_direction_pairs() {
        let mut ranges: [Range<u32>; 6] =
            std::array::from_fn(|d| (d as u32 * 3)..(d as u32 * 3 + 3));
        ranges[1] = 0..0; // -X empty
        let pos = CubePos { x: -4, y: 5, z: -6 };
        let meta = meta_for(pos, 2, 100, &ranges);
        assert_eq!(meta.pos_page, [-4, 5, -6, 2]);
        assert_eq!(meta.start_flags, [100, META_ACTIVE, 0, 0]);
        // Direction d's (offset, count) pair is dirs[d / 2][(d % 2) * 2..].
        for (d, range) in ranges.iter().enumerate() {
            let (offset, count) = (
                meta.dirs[d / 2][(d % 2) * 2],
                meta.dirs[d / 2][(d % 2) * 2 + 1],
            );
            assert_eq!(offset, range.start);
            assert_eq!(count, range.len() as u32);
        }
        assert_eq!(non_empty_dirs(&ranges), 5);
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
        let mut live: Vec<CubePos> = Vec::new();
        let mut next_pos = 0;
        for _ in 0..10_000 {
            let op = rng.next() % 4;
            if op == 0 || live.is_empty() {
                // Upload a new cube.
                let pos = CubePos {
                    x: next_pos,
                    y: 0,
                    z: 0,
                };
                next_pos += 1;
                let ranges = random_ranges(&mut rng);
                let page = (rng.next() % 4) as usize;
                let start = (rng.next() % 1024) as u32;
                let len: u32 = ranges.iter().map(|r| r.len() as u32).sum();
                book.commit(pos, page, start..start + len, ranges, None);
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
                    book.commit_empty(pos);
                    live.swap_remove(i);
                } else {
                    let ranges = random_ranges(&mut rng);
                    let page = (rng.next() % 4) as usize;
                    let start = (rng.next() % 1024) as u32;
                    let len: u32 = ranges.iter().map(|r| r.len() as u32).sum();
                    let slot = book.alloc_meta_slot();
                    book.commit(pos, page, start..start + len, ranges, Some(slot));
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
