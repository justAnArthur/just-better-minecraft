//! The terrain pass: all cube meshes live in a few large face "pages"
//! (storage buffers) managed by CPU-side suballocators (render/suballoc.rs),
//! one camera-relative origins buffer rewritten each frame, CPU frustum
//! culling of cube AABBs plus per-direction back-face skipping, and one
//! multi-draw indirect per page (or a direct-draw fallback loop on hardware
//! without indirect support). See crates/client/README.md.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Mat4, Vec3, Vec4};
use jbm_core::coords::{CUBE_SIZE, CubePos};

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::mesh::face::PackedFace;
use crate::mesh::greedy::MeshOutput;
use crate::render::renderer::DEPTH_FORMAT;
use crate::render::suballoc::Suballocator;
use crate::render::texture_array::BlockTextures;

/// Bytes per face (`PackedFace`).
const FACE_BYTES: u64 = std::mem::size_of::<PackedFace>() as u64;

/// Default page size: 4M faces (32 MiB). Clamped to the device limits at
/// construction, and never below one worst-case cube mesh.
const DEFAULT_PAGE_FACES: u32 = 4 * 1024 * 1024;

/// The most faces one cube's mesh can hold: the checkerboard worst case,
/// half the cube's blocks facing all six ways.
pub fn worst_case_faces() -> u32 {
    ((CUBE_SIZE as u64).pow(3) * 6 / 2) as u32
}

/// One cube's uploaded mesh. Empty meshes keep an entry (so the world knows
/// the cube is meshed) but allocate nothing. Non-empty meshes record their
/// page and face range so a re-mesh can free exactly what they had.
struct CubeMesh {
    face_count: u32,
    /// Page index and face range within that page's buffer.
    alloc: Option<(usize, Range<u32>)>,
    /// Per-direction face ranges within the cube's faces
    /// (`MeshOutput::ranges`), so drawing can skip back-facing directions.
    ranges: [Range<u32>; 6],
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

/// Per-frame terrain draw statistics, for the overlay and the metrics
/// example.
#[derive(Debug, Clone, Copy, Default)]
pub struct TerrainStats {
    /// Cubes with an uploaded mesh (empty or not).
    pub meshed: usize,
    /// Cubes passing the frustum test with at least one face.
    pub drawn: usize,
    /// Faces issued this frame, counting only the non-skipped directions.
    pub faces: u64,
    /// Indirect commands (MDI) or fallback draws issued this frame; one per
    /// visible cube and non-empty, non-skipped face direction.
    pub commands: u64,
    /// Render-pass draw API calls: one per non-empty page with MDI, one per
    /// command on the fallback path.
    pub draw_calls: u64,
    /// Face pages allocated so far (pages are never freed).
    pub pages: usize,
    /// GPU bytes across all pages.
    pub page_bytes: u64,
    /// Faces currently allocated across all pages.
    pub used_faces: u64,
}

/// Camera-relative cube origin, one per visible cube slot. `w` is padding
/// to the storage buffer's 16-byte element.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CubeOrigin {
    pos: [f32; 4],
}

/// Slots preallocated in the origins buffer. Grown (buffer recreated) only
/// when more cubes than this are simultaneously visible.
const ORIGIN_CAPACITY: u64 = 4096;

/// Indirect commands preallocated before the indirect buffer is grown.
const INDIRECT_CAPACITY: u64 = 1024;

/// Draws the terrain: one pipeline, one shared bind group, one page bind
/// group per page and one `multi_draw_indirect` (or a loop of direct draws)
/// per page.
pub struct TerrainPass {
    pipeline: wgpu::RenderPipeline,
    faces_layout: wgpu::BindGroupLayout,
    shared_layout: wgpu::BindGroupLayout,
    shared_bind_group: wgpu::BindGroup,
    /// Kept so the bind group can be rebuilt when the origins buffer
    /// grows; ref-counted clones.
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    origins_buffer: wgpu::Buffer,
    origins_capacity: u64,
    pages: Vec<FacePage>,
    /// Faces per page (every page is the same size).
    page_faces: u32,
    meshes: HashMap<CubePos, CubeMesh>,
    indirect_buffer: wgpu::Buffer,
    indirect_capacity: u64,
    /// Whether multi-draw indirect is usable here: the adapter has
    /// `DownlevelFlags::INDIRECT_EXECUTION`, the device was created with
    /// `Features::INDIRECT_FIRST_INSTANCE`, and `JBM_NO_MDI` is unset.
    mdi: bool,
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
        println!(
            "terrain draw path: {}",
            if pass.uses_mdi() {
                "multi-draw indirect"
            } else {
                "fallback (direct draws)"
            }
        );
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

        // Group 2, shared: origins, textures, sampler.
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
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let origins_buffer = Self::make_origins_buffer(device, ORIGIN_CAPACITY);
        let texture_view = textures.view().clone();
        let sampler = textures.sampler().clone();
        let shared_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jbm terrain shared"),
            layout: &shared_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: origins_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(textures.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
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

        let downlevel = gpu
            .adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION);
        let feature = gpu
            .device
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        let env_allows = !std::env::var_os("JBM_NO_MDI").is_some_and(|v| v == "1");

        TerrainPass {
            pipeline,
            faces_layout,
            shared_layout,
            shared_bind_group,
            texture_view,
            sampler,
            origins_buffer,
            origins_capacity: ORIGIN_CAPACITY,
            pages: Vec::new(),
            page_faces,
            meshes: HashMap::new(),
            indirect_buffer: Self::make_indirect_buffer(device, INDIRECT_CAPACITY),
            indirect_capacity: INDIRECT_CAPACITY,
            mdi: downlevel && feature && env_allows,
            force_fallback: false,
            stats: TerrainStats::default(),
        }
    }

    fn make_origins_buffer(device: &wgpu::Device, slots: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain origins"),
            size: slots * std::mem::size_of::<CubeOrigin>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_indirect_buffer(device: &wgpu::Device, commands: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain indirect"),
            size: commands * std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64,
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

    /// First page that fits, else a new one. A mesh larger than a whole
    /// page is impossible: the constructor asserts pages hold at least one
    /// worst-case cube mesh.
    fn alloc_slot(&mut self, gpu: &Gpu, len: u32) -> (usize, Range<u32>) {
        let page = match self
            .pages
            .iter()
            .position(|p| p.suballoc.largest_free() >= len)
        {
            Some(page) => page,
            None => {
                self.pages.push(self.make_page(gpu));
                self.pages.len() - 1
            }
        };
        let range = self.pages[page]
            .suballoc
            .alloc(len)
            .expect("a page chosen by largest_free must fit the mesh");
        (page, range)
    }

    /// Replaces the mesh of one cube. A re-mesh frees the cube's old range
    /// and replaces the slot; it never creates a buffer (except when no
    /// page fits, which adds one page). An empty mesh allocates nothing.
    /// The app enforces the per-frame upload budget before calling this.
    pub fn upload(&mut self, gpu: &Gpu, pos: CubePos, output: &MeshOutput) {
        if let Some((page, range)) = self.meshes.get(&pos).and_then(|m| m.alloc.clone()) {
            self.pages[page].suballoc.free(range);
        }
        if output.is_empty() {
            // Keep the entry (the cube is meshed) but hold no allocation.
            self.meshes.insert(
                pos,
                CubeMesh {
                    face_count: 0,
                    alloc: None,
                    ranges: std::array::from_fn(|_| 0u32..0),
                },
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
        self.meshes.insert(
            pos,
            CubeMesh {
                face_count: len,
                alloc: Some((page, range)),
                ranges: output.ranges.clone(),
            },
        );
    }

    /// Drops a cube's mesh (UnloadCube), freeing its page range. Pages
    /// themselves are never freed in this task.
    pub fn remove(&mut self, pos: CubePos) {
        if let Some(mesh) = self.meshes.remove(&pos)
            && let Some((page, range)) = mesh.alloc
        {
            self.pages[page].suballoc.free(range);
        }
    }

    pub fn stats(&self) -> TerrainStats {
        self.stats
    }

    /// Whether the next draw uses multi-draw indirect (as opposed to the
    /// direct-draw fallback). Reflects `set_force_fallback`.
    pub fn uses_mdi(&self) -> bool {
        self.mdi && !self.force_fallback
    }

    /// Forces the direct-draw fallback path (or restores the automatic
    /// choice), for tests and benches. Both paths share the same buffers
    /// and commands, so the image is identical.
    pub fn set_force_fallback(&mut self, force: bool) {
        self.force_fallback = force;
    }

    /// Runs [`Suballocator::check_invariants`] on every page. Test-only.
    #[cfg(any(test, feature = "testing"))]
    pub fn check_page_invariants(&self) {
        for page in &self.pages {
            page.suballoc.check_invariants();
        }
    }

    /// Total faces across all uploaded meshes.
    pub fn total_faces(&self) -> u64 {
        self.meshes.values().map(|m| u64::from(m.face_count)).sum()
    }

    pub fn meshed_count(&self) -> usize {
        self.meshes.len()
    }

    /// Face pages allocated so far (pages are never freed).
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    fn rel_min(pos: CubePos, camera: &Camera) -> Vec3 {
        let world_min = DVec3::new(
            f64::from(pos.x) * f64::from(CUBE_SIZE),
            f64::from(pos.y) * f64::from(CUBE_SIZE),
            f64::from(pos.z) * f64::from(CUBE_SIZE),
        );
        camera.relative(world_min)
    }

    /// Records the terrain draw: frustum-culled cubes, camera-relative
    /// origins rewritten into the shared buffer, one indirect command per
    /// visible cube and non-empty, non-skipped face direction grouped by
    /// page, then one `multi_draw_indirect` per page (or the same commands
    /// as a loop of direct draws).
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        camera: &Camera,
        uniforms: &wgpu::BindGroup,
    ) {
        let frustum = Frustum::from_view_proj(&camera.view_proj());
        // Deterministic order: HashMap iteration is randomized.
        let mut visible: Vec<(CubePos, Vec3, &CubeMesh)> = self
            .meshes
            .iter()
            .filter(|(_, mesh)| mesh.face_count > 0)
            .filter_map(|(pos, mesh)| {
                let min = Self::rel_min(*pos, camera);
                let max = min + Vec3::splat(CUBE_SIZE as f32);
                frustum
                    .intersects_aabb(min, max)
                    .then_some((*pos, min, mesh))
            })
            .collect();
        visible.sort_unstable_by_key(|(pos, _, _)| *pos);

        let slots = visible.len() as u64;
        if slots == 0 {
            self.stats = self.page_stats(TerrainStats {
                meshed: self.meshes.len(),
                drawn: 0,
                faces: 0,
                commands: 0,
                draw_calls: 0,
                pages: self.pages.len(),
                page_bytes: (self.pages.len() as u64) * u64::from(self.page_faces) * FACE_BYTES,
                used_faces: 0,
            });
            return;
        }
        if slots > self.origins_capacity {
            // Grow (rare): one new buffer plus bind group, still no
            // per-frame churn in steady state.
            let capacity = slots.next_power_of_two();
            self.origins_buffer = Self::make_origins_buffer(&gpu.device, capacity);
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jbm terrain shared"),
                layout: &self.shared_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.origins_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.shared_bind_group = bind_group;
            self.origins_capacity = capacity;
        }

        // Camera-relative origins, one per slot, rewritten every frame.
        let mut origins = Vec::with_capacity(visible.len());
        for (pos, _, _) in &visible {
            let relative = Self::rel_min(*pos, camera);
            origins.push(CubeOrigin {
                pos: [relative.x, relative.y, relative.z, 0.0],
            });
        }
        gpu.queue
            .write_buffer(&self.origins_buffer, 0, bytemuck::cast_slice(&origins));

        // One command per visible cube and face direction with a non-empty,
        // non-skipped range, grouped by page (multi_draw_indirect walks one
        // contiguous command block per page).
        let mut per_page: Vec<Vec<wgpu::util::DrawIndirectArgs>> =
            self.pages.iter().map(|_| Vec::new()).collect();
        let mut faces = 0u64;
        for (slot, (_, rel_min, mesh)) in visible.iter().enumerate() {
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
                    first_instance: slot as u32,
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
        if commands.len() as u64 > self.indirect_capacity {
            // Grow (rare), like the origins buffer.
            let capacity = (commands.len() as u64).next_power_of_two();
            self.indirect_buffer = Self::make_indirect_buffer(&gpu.device, capacity);
            self.indirect_capacity = capacity;
        }
        gpu.queue
            .write_buffer(&self.indirect_buffer, 0, bytemuck::cast_slice(&commands));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, uniforms, &[]);
        pass.set_bind_group(2, &self.shared_bind_group, &[]);
        let mut draw_calls = 0u64;
        if self.uses_mdi() {
            let args = std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64;
            let mut offset = 0u64;
            for &(page, count) in &page_spans {
                pass.set_bind_group(1, &self.pages[page].bind_group, &[]);
                pass.multi_draw_indirect(&self.indirect_buffer, offset * args, count);
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

        self.stats = self.page_stats(TerrainStats {
            meshed: self.meshes.len(),
            drawn: visible.len(),
            faces,
            commands: commands.len() as u64,
            draw_calls,
            pages: self.pages.len(),
            page_bytes: (self.pages.len() as u64) * u64::from(self.page_faces) * FACE_BYTES,
            used_faces: 0,
        });
    }

    /// Fills the page-derived fields (`used_faces`) of a stats snapshot.
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
}
