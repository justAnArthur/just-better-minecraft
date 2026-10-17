//! The terrain pass: all cube meshes live in a few large face "pages"
//! (storage buffers) managed by CPU-side suballocators (render/suballoc.rs),
//! one persistent `cube_meta` slot per resident non-empty cube, and three
//! draw paths:
//!
//! - **gpu-cull** (opt-in where available): a compute pass tests every
//!   meta slot against the frustum and a Hi-Z occlusion pyramid
//!   (render/hiz.rs) and writes the indirect commands into per-page
//!   regions; two render phases draw them. The CPU's only per-cube work is
//!   sizing the regions: a frustum pass over the resident meta slots that
//!   bounds what the GPU can emit (Metal's multi_draw_indirect is a CPU
//!   loop over the encoded commands, so the encoded count must come from
//!   the CPU — see `MeshBookkeeping::frustum_page_bounds`).
//! - **cpu-cull** (default): the CPU frustum and direction tests, using
//!   the same meta slots and the shared origin formula, drawn with one
//!   multi_draw_indirect per page.
//! - **fallback**: the cpu-cull commands issued as direct draws, for
//!   hardware without indirect support.
//!
//! The pass is split by concern: [`layout`] (the GPU layout contract),
//! [`space`] (camera and frustum math), [`pages`] (CPU-side bookkeeping),
//! [`stats`] (counters and the readback ring), [`pipelines`] (construction
//! and bind groups), [`cpu_cull`] and [`gpu_cull`] (the draw paths).
//! See crates/client/README.md.

pub mod cpu_cull;
pub mod gpu_cull;
pub mod layout;
pub mod pages;
pub mod pipelines;
pub mod space;
pub mod stats;

pub use layout::{
    AO_SOFTEN, CubeMeta, CullParams, DEFAULT_FURTHEST_RING_RADIUS, DEFAULT_LOD0_RING_RADIUS,
    FADE_END_FRACTION, FADE_START_FRACTION, FOG_END_FRACTION, FOG_START_FRACTION, LOD_TINT_COLORS,
    META_ACTIVE, META_LEVEL_SHIFT, META_MASK_SHIFT, META_VISIBLE, TerrainUniforms, lod_tint_color,
    unpack_meta_flags,
};
pub use space::{Frustum, camera_split, cube_origin, visible_directions};
pub use stats::{CullStats, LevelStats, TerrainStats};

use std::ops::Range;

use glam::{DVec3, Mat4};
use jbm_core::coords::CUBE_SIZE;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::lod::CubeKey;
use crate::mesh::MeshOutput;
use crate::render::hiz::HizPyramid;
use crate::render::renderer::DepthTarget;
use crate::render::settings::RenderSettings;
use crate::render::texture_array::BlockTextures;

use layout::{debug_uniform_words, far_uniform_words, meta_flags, meta_for, tint_uniform_words};
use pages::{FacePage, MeshBookkeeping};
use pipelines::FACE_BYTES;
use stats::StatsRing;

/// The most faces one cube's mesh can hold: the checkerboard worst case,
/// half the cube's blocks facing all six ways.
pub fn worst_case_faces() -> u32 {
    ((CUBE_SIZE as u64).pow(3) * 6 / 2) as u32
}

/// The freeze-cull view's captured state (task 2.12): the camera position
/// and rotation-only view-projection from the moment the debug view
/// switched on. The cull keeps testing against these while the real
/// camera moves; `frozen_view` hands them to the app so it can draw the
/// frozen frustum as lines.
#[derive(Debug, Clone, Copy)]
struct Frozen {
    position: DVec3,
    view_proj: Mat4,
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

/// The path-selection rule, pure so it can be unit-tested exhaustively:
/// no MDI (or a forced fallback) means direct draws; otherwise cpu-cull is
/// the default and gpu-cull needs a gpu-capable adapter **and** an explicit
/// preference (`JBM_GPU_CULL=1` / `set_prefer_gpu_cull`) — on Metal it
/// measured slower than cpu-cull in every scene (README, "Three paths").
/// The force flags win over the preference.
fn select_path(
    mdi: bool,
    gpu_capable: bool,
    prefer_gpu: bool,
    force_cpu: bool,
    force_fallback: bool,
) -> TerrainPath {
    if !mdi || force_fallback {
        TerrainPath::Fallback
    } else if force_cpu || !gpu_capable || !prefer_gpu {
        TerrainPath::CpuCull
    } else {
        TerrainPath::GpuCull
    }
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
    /// The per-layer average colors, kept for the shared bind group
    /// rebuilds.
    averages_buffer: wgpu::Buffer,
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
    /// CPU-path indirect buffer (packed per page; grown like the face
    /// pages).
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
    /// The freeze-cull view's uniform (group 1, binding 7 of cull.wgsl).
    freeze_buffer: wgpu::Buffer,
    /// The frozen camera the cull keeps using while `freeze_cull` is on;
    /// captured on the setting's rising edge, released when it goes off.
    frozen: Option<Frozen>,
    hiz: Option<HizPyramid>,
    /// The 1x1 view the cull bind group binds before the first gpu-cull
    /// frame creates the real pyramid (uploads grow the region buffers,
    /// which rebuild the bind group, long before any cull dispatch).
    pyramid_placeholder: wgpu::TextureView,
    last_cull: CullStats,
    /// Whether multi-draw indirect is usable here: the adapter has
    /// `DownlevelFlags::INDIRECT_EXECUTION` and the device was created
    /// with `Features::INDIRECT_FIRST_INSTANCE`.
    mdi: bool,
    /// Whether the gpu-cull path is usable: MDI plus
    /// `DownlevelFlags::COMPUTE_SHADERS` and R32Float storage textures.
    gpu_capable: bool,
    /// Whether gpu-cull is preferred over the cpu-cull default (from
    /// `RenderSettings` at construction, or `set_prefer_gpu_cull`).
    prefer_gpu_cull: bool,
    /// Forced cpu-cull for tests and benches (`set_force_cpu_cull`).
    force_cpu_cull: bool,
    /// Forced fallback for tests and benches (`set_force_fallback`).
    force_fallback: bool,
    stats: TerrainStats,
}

impl TerrainPass {
    /// Creates the pass with the default page size and settings (the
    /// cpu-cull default path, fog on, no tint). The app constructs with
    /// [`TerrainPass::with_settings`] so the environment-derived
    /// `RenderSettings` reach the path choice.
    pub fn new(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        textures: &BlockTextures,
    ) -> TerrainPass {
        TerrainPass::with_settings(
            gpu,
            color_format,
            uniform_layout,
            textures,
            &RenderSettings::default(),
        )
    }

    /// Creates the pass seeded from `settings`: the path preference and
    /// force flags (the environment half of the choice — the hardware
    /// half is probed here).
    pub fn with_settings(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        textures: &BlockTextures,
        settings: &RenderSettings,
    ) -> TerrainPass {
        let pass = TerrainPass::build(gpu, color_format, uniform_layout, textures, settings);
        println!("terrain draw path: {}", pass.path_name());
        if pass.gpu_capable && !pass.prefer_gpu_cull {
            println!("gpu-cull available (JBM_GPU_CULL=1)");
        }
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
        TerrainPass::build_with_page_faces(
            gpu,
            color_format,
            uniform_layout,
            textures,
            page_faces,
            &RenderSettings::default(),
        )
    }

    /// Replaces the mesh of one cube. A re-mesh frees the cube's old range,
    /// rewrites its meta slot in place (with the visible bit reset, so the
    /// late cull picks the new geometry up in the same frame) and never
    /// creates a buffer (except when no page fits, which adds one page). An
    /// empty mesh allocates nothing and holds no slot. The app enforces
    /// the per-frame upload budget before calling this.
    pub fn upload(&mut self, gpu: &Gpu, key: CubeKey, output: &MeshOutput) {
        let released = self.book.release(&key);
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
            self.book.commit_empty(
                key,
                released.drawn.unwrap_or(true),
                released.mask.unwrap_or(0),
            );
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
            key,
            page,
            range.clone(),
            output.ranges.clone(),
            released.slot,
            output.skirts,
            // A fresh mesh draws (the default for callers without the
            // coverage rule); a re-mesh keeps its coverage state.
            released.drawn.unwrap_or(true),
            released.mask.unwrap_or(0),
        );
        if slot >= self.meta_capacity {
            self.grow_meta_buffer(gpu);
        }
        let mesh = &self.book.meshes[&key];
        gpu.queue.write_buffer(
            &self.meta_buffer,
            u64::from(slot) * std::mem::size_of::<layout::CubeMeta>() as u64,
            bytemuck::bytes_of(&meta_for(
                key,
                page,
                range.start,
                &output.ranges,
                output.skirts,
                mesh.drawn,
                mesh.mask,
            )),
        );
    }

    /// Drops a cube's mesh (UnloadCube / UnloadLodCube), freeing its page
    /// range and meta slot. Pages themselves are never freed.
    pub fn remove(&mut self, gpu: &Gpu, key: CubeKey) {
        let released = self.book.release(&key);
        if let (Some(page), Some(range)) = (released.page, released.range) {
            self.pages[page].suballoc.free(range);
        }
        if let Some(slot) = released.slot {
            self.zero_meta_slot(gpu, slot);
            self.book.free_meta_slot(slot);
        }
        self.book.meshes.remove(&key);
    }

    /// Applies the coverage rule's verdict for one cube: whether it draws
    /// at all, and which octants a ready child replaces. Rewrites the
    /// meta slot's flag word only when the state actually changed —
    /// never per frame. No-op for cubes without a slot (empty meshes draw
    /// nothing anyway).
    pub fn set_draw_state(&mut self, gpu: &Gpu, key: CubeKey, drawn: bool, mask: u8) {
        let Some(mesh) = self.book.meshes.get_mut(&key) else {
            return;
        };
        if mesh.drawn == drawn && mesh.mask == mask {
            return;
        }
        mesh.drawn = drawn;
        mesh.mask = mask;
        let Some(slot) = mesh.slot else {
            return;
        };
        let (page, start, ranges, skirts) = match &mesh.alloc {
            Some((page, range)) => (
                Some(*page),
                Some(range.start),
                mesh.ranges.clone(),
                mesh.skirts,
            ),
            None => (None, None, mesh.ranges.clone(), mesh.skirts),
        };
        if let (Some(page), Some(start)) = (page, start) {
            let meta = meta_for(key, page, start, &ranges, skirts, drawn, mask);
            gpu.queue.write_buffer(
                &self.meta_buffer,
                u64::from(slot) * std::mem::size_of::<layout::CubeMeta>() as u64,
                bytemuck::bytes_of(&meta),
            );
        } else {
            // No allocation (should not happen for a slot holder): write
            // the flag word alone.
            let flags = meta_flags(drawn, key.level, mask);
            gpu.queue.write_buffer(
                &self.meta_buffer,
                u64::from(slot) * std::mem::size_of::<layout::CubeMeta>() as u64 + 16,
                bytemuck::bytes_of(&flags),
            );
        }
    }

    /// A freed slot must read inactive: the cull threads dispatch over the
    /// whole high-water mark.
    fn zero_meta_slot(&self, gpu: &Gpu, slot: u32) {
        gpu.queue.write_buffer(
            &self.meta_buffer,
            u64::from(slot) * std::mem::size_of::<layout::CubeMeta>() as u64,
            bytemuck::bytes_of(&layout::CubeMeta::zeroed()),
        );
    }

    /// Grows the meta buffer (rare): a new buffer, a new cull bind group,
    /// a new shared bind group, and every live meta rewritten into it
    /// (each cube keeps its slot).
    fn grow_meta_buffer(&mut self, gpu: &Gpu) {
        self.meta_capacity *= 2;
        self.meta_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain cube meta"),
            size: u64::from(self.meta_capacity) * std::mem::size_of::<layout::CubeMeta>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut metas = vec![layout::CubeMeta::zeroed(); self.meta_capacity as usize];
        for (key, mesh) in &self.book.meshes {
            let Some(slot) = mesh.slot else { continue };
            let (page, range) = mesh
                .alloc
                .as_ref()
                .expect("a mesh with a slot holds an allocation");
            metas[slot as usize] = meta_for(
                *key,
                *page,
                range.start,
                &mesh.ranges,
                mesh.skirts,
                mesh.drawn,
                mesh.mask,
            );
        }
        gpu.queue
            .write_buffer(&self.meta_buffer, 0, bytemuck::cast_slice(&metas));
        self.rebuild_shared_bind_group(gpu);
        self.rebuild_cull_bind_group();
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
            suballoc: crate::render::suballoc::Suballocator::new(self.page_faces),
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

    /// The per-frame terrain uniform for `camera` (the one object both
    /// draw paths write); the view distances, fog, tint and the task 2.12
    /// debug views come from the renderer-owned `RenderSettings`.
    fn terrain_uniforms_for(&self, camera: &Camera, settings: &RenderSettings) -> TerrainUniforms {
        let (camera_cube, camera_frac) = camera_split(camera.position);
        TerrainUniforms {
            camera_cube: [camera_cube.x, camera_cube.y, camera_cube.z, 0],
            camera_frac: [
                camera_frac.x,
                camera_frac.y,
                camera_frac.z,
                CUBE_SIZE as f32,
            ],
            far: far_uniform_words(settings.far_radii, settings.fog),
            tint: tint_uniform_words(settings.lod_tint),
            debug: debug_uniform_words(settings),
        }
    }

    /// Captures or releases the freeze-cull view on the setting's edges
    /// (task 2.12), and writes the frozen uniform the cull shader reads.
    fn update_frozen(&mut self, gpu: &Gpu, camera: &Camera, settings: &RenderSettings) {
        if settings.freeze_cull && self.frozen.is_none() {
            self.frozen = Some(Frozen {
                position: camera.position,
                view_proj: camera.view_proj(),
            });
        }
        if !settings.freeze_cull {
            self.frozen = None;
        }
        if let Some(frozen) = &self.frozen {
            let (camera_cube, camera_frac) = camera_split(frozen.position);
            let freeze = layout::FreezeParams {
                view_proj: frozen.view_proj.to_cols_array_2d(),
                camera_cube: [camera_cube.x, camera_cube.y, camera_cube.z, 0],
                camera_frac: [
                    camera_frac.x,
                    camera_frac.y,
                    camera_frac.z,
                    CUBE_SIZE as f32,
                ],
            };
            gpu.queue
                .write_buffer(&self.freeze_buffer, 0, bytemuck::bytes_of(&freeze));
        }
    }

    /// The frozen camera the cull keeps testing against while the
    /// freeze-cull view is on (task 2.12): its world position and
    /// rotation-only view-projection, for drawing the frozen frustum.
    pub fn frozen_view(&self) -> Option<(DVec3, Mat4)> {
        self.frozen
            .map(|frozen| (frozen.position, frozen.view_proj))
    }

    /// The resident cubes the coverage rule draws (task 2.12's chunk
    /// borders iterate them): one key per non-empty, drawn, not fully
    /// masked mesh.
    pub fn resident_keys(&self) -> Vec<CubeKey> {
        self.book
            .meshes
            .iter()
            .filter(|(_, mesh)| {
                mesh.face_count > 0 && mesh.drawn && mesh.mask != 0xFF && mesh.alloc.is_some()
            })
            .map(|(key, _)| *key)
            .collect()
    }

    pub fn stats(&self) -> TerrainStats {
        self.stats
    }

    /// Per-level residency, coverage and face counts, level-ascending,
    /// with the cull's per-level drawn and face counters merged in (task
    /// 2.12; on the gpu-cull path those lag one or two frames like the
    /// rest of `CullStats`).
    pub fn level_stats(&self) -> Vec<LevelStats> {
        let mut levels: Vec<LevelStats> = Vec::new();
        for (key, mesh) in &self.book.meshes {
            let level = key.level as usize;
            if levels.len() <= level {
                levels.resize_with(level + 1, LevelStats::default);
            }
            let entry = &mut levels[level];
            entry.level = key.level;
            entry.resident += 1;
            if mesh.drawn && mesh.mask != 0xFF {
                entry.drawn += 1;
                entry.masked_octants += mesh.mask.count_ones();
            }
            entry.faces += u64::from(mesh.face_count);
            entry.skirts += mesh.skirts;
        }
        for (level, entry) in levels.iter_mut().enumerate() {
            let index = level.min(self.last_cull.level_drawn.len() - 1);
            entry.frame_drawn = self.last_cull.level_drawn[index];
            entry.frame_faces = u64::from(self.last_cull.level_faces[index]);
        }
        levels
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
        select_path(
            self.mdi,
            self.gpu_capable,
            self.prefer_gpu_cull,
            self.force_cpu_cull,
            self.force_fallback,
        )
    }

    /// Forces the cpu-cull path (or restores the automatic choice), for
    /// tests and benches. Falls back to direct draws when the adapter
    /// lacks MDI.
    pub fn set_force_cpu_cull(&mut self, force: bool) {
        self.force_cpu_cull = force;
    }

    /// Prefers the gpu-cull path over the cpu-cull default (or restores
    /// the default), mirroring `JBM_GPU_CULL=1` for tests, benches and
    /// the metrics example. Ignored when the adapter is not gpu-capable.
    pub fn set_prefer_gpu_cull(&mut self, prefer: bool) {
        self.prefer_gpu_cull = prefer;
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
    /// the README; `draw_gpu_cull` and `draw_cpu_cull` in the path
    /// modules. `settings` is the renderer-owned frame state (fog, level
    /// tint, the task 2.12 debug views) — its path flags are re-applied
    /// every draw, so the panel's force checkboxes switch the path live.
    // The seven parameters are one frame's draw context; a bundling
    // struct would only be unpacked again in each path.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &DepthTarget,
        camera: &Camera,
        uniforms: &wgpu::BindGroup,
        settings: &RenderSettings,
    ) {
        // Live path switching (task 2.12): the settings' path flags win
        // over the cached ones every frame. The set_force_* setters stay
        // as the test and bench override for exactly the frames between
        // two draws.
        self.prefer_gpu_cull = settings.prefer_gpu_cull;
        self.force_cpu_cull = settings.force_cpu_cull;
        self.force_fallback = settings.force_fallback;
        self.update_frozen(gpu, camera, settings);
        match self.active_path() {
            TerrainPath::GpuCull => {
                self.draw_gpu_cull(gpu, encoder, target, depth, camera, uniforms, settings)
            }
            TerrainPath::CpuCull | TerrainPath::Fallback => {
                self.draw_cpu_cull(gpu, encoder, target, depth, camera, uniforms, settings)
            }
        }
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
    use crate::mesh::face::PackedFace;

    #[test]
    fn path_selection_rules() {
        use TerrainPath::{CpuCull, Fallback, GpuCull};
        // Exhaustive over the five booleans, with the expected path derived
        // from the documented rule, not from select_path itself: no MDI (or
        // a forced fallback) is the fallback; otherwise cpu-cull unless the
        // adapter is gpu-capable, gpu-cull is preferred and nothing forces
        // another path.
        for mdi in [false, true] {
            for gpu_capable in [false, true] {
                for prefer_gpu in [false, true] {
                    for force_cpu in [false, true] {
                        for force_fallback in [false, true] {
                            let expected = if !mdi || force_fallback {
                                Fallback
                            } else if gpu_capable && prefer_gpu && !force_cpu {
                                GpuCull
                            } else {
                                CpuCull
                            };
                            assert_eq!(
                                select_path(
                                    mdi,
                                    gpu_capable,
                                    prefer_gpu,
                                    force_cpu,
                                    force_fallback
                                ),
                                expected,
                                "({mdi}, {gpu_capable}, {prefer_gpu}, {force_cpu}, {force_fallback})",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn worst_case_faces_matches_the_checkerboard_bound() {
        let volume = (CUBE_SIZE as u64).pow(3);
        assert_eq!(u64::from(worst_case_faces()), 6 * volume / 2);
    }

    /// The byte sizes the page math and the indirect offsets are built
    /// on: one `PackedFace` per face, one `DrawIndirectArgs` per command.
    #[test]
    fn face_and_command_byte_sizes() {
        assert_eq!(FACE_BYTES, std::mem::size_of::<PackedFace>() as u64);
        assert_eq!(
            pipelines::CMD_BYTES,
            std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64
        );
    }
}
