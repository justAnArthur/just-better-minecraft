//! Pipeline and bind-group construction for the terrain pass: the render
//! pipeline (with the sky-gradient snippet concatenated ahead of the
//! terrain shader so the fog evaluates the identical function the sky pass
//! shows), the two cull compute pipelines, the shared and cull bind
//! groups, and their rebuilds when a bound resource is recreated.

use crate::gpu::Gpu;
use crate::mesh::face::PackedFace;
use crate::render::renderer::DEPTH_FORMAT;
use crate::render::settings::RenderSettings;
use crate::render::texture_array::BlockTextures;

use super::TerrainPass;
use super::layout::{CubeMeta, CullParams, FreezeParams, TerrainUniforms};
use super::pages::MeshBookkeeping;
use super::stats::{CullStats, STATS_BYTES, StatsRing};
use super::worst_case_faces;

/// Bytes per face (`PackedFace`).
pub(super) const FACE_BYTES: u64 = std::mem::size_of::<PackedFace>() as u64;

/// Bytes per indirect command (`wgpu::util::DrawIndirectArgs`).
pub(super) const CMD_BYTES: u64 = 16;

/// Default page size: 4M faces (32 MiB). Clamped to the device limits at
/// construction, and never below one worst-case cube mesh.
const DEFAULT_PAGE_FACES: u32 = 4 * 1024 * 1024;

/// Meta slots preallocated before the meta buffer is grown.
const META_CAPACITY: u32 = 4096;

/// Indirect commands preallocated before the CPU indirect buffer is grown.
const INDIRECT_CAPACITY: u64 = 1024;

/// The initial gpu-cull command capacity, in command elements; grown with
/// the bound.
const COMMANDS_CAPACITY: u32 = 256;

impl TerrainPass {
    /// Builds everything the pass needs with the default page size.
    pub(super) fn build(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        textures: &BlockTextures,
        settings: &RenderSettings,
    ) -> TerrainPass {
        let limits = gpu.device.limits();
        let max_faces = (limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / FACE_BYTES) as u32;
        let page_faces = DEFAULT_PAGE_FACES.min(max_faces).max(worst_case_faces());
        TerrainPass::build_with_page_faces(
            gpu,
            color_format,
            uniform_layout,
            textures,
            page_faces,
            settings,
        )
    }

    /// Builds everything the pass needs with an explicit page size; the
    /// page must hold at least one worst-case cube mesh.
    pub(super) fn build_with_page_faces(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        textures: &BlockTextures,
        page_faces: u32,
        settings: &RenderSettings,
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
                    // VERTEX: the origin formula; FRAGMENT: the far-look
                    // fade and fog windows.
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                wgpu::BindGroupLayoutEntry {
                    // Per-layer average colors (the flat far look).
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Per-layer materials (task 2.15: water, far-side lean).
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: textures.averages_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: textures.materials_buffer().as_entire_binding(),
                },
            ],
        });

        // The sky gradient snippet is concatenated ahead of the terrain
        // text so the fog evaluates the identical function the sky pass
        // shows (WGSL has no includes). It has no dependencies, so leading
        // with it is valid.
        let terrain_source = format!(
            "{}{}",
            include_str!("../shaders/sky_gradient.wgsl"),
            include_str!("../shaders/terrain.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm terrain"),
            source: wgpu::ShaderSource::Wgsl(terrain_source.into()),
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
            // The renderer's MSAA count (task 2.15). Cutout layers keep
            // the alpha-test discard rather than alpha-to-coverage: the
            // builtin cutout texels are exactly 0 or 255, so the discard
            // is the same at 1× and 4× and keeps texel edges crisp.
            multisample: wgpu::MultisampleState {
                count: settings.msaa_samples,
                ..Default::default()
            },
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
                wgpu::BindGroupLayoutEntry {
                    // The freeze-cull view (task 2.12): the frozen
                    // view-projection and camera split.
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm cull"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cull.wgsl").into()),
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
        let commands_capacity = COMMANDS_CAPACITY;
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
        let freeze_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm cull freeze view"),
            size: std::mem::size_of::<FreezeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
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
                .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);

        let cull_bind_group = TerrainPass::make_cull_bind_group(
            device,
            &cull_layout,
            &meta_buffer,
            &params_buffer,
            &commands_buffer,
            &region_base_buffer,
            &region_counter_buffer,
            &stats_buffer,
            &placeholder_view,
            &freeze_buffer,
        );

        TerrainPass {
            device: gpu.device.clone(),
            pipeline,
            faces_layout,
            shared_layout,
            shared_bind_group,
            texture_view,
            sampler,
            // The per-layer average colors, kept for the shared bind group
            // rebuilds.
            averages_buffer: textures.averages_buffer().clone(),
            materials_buffer: textures.materials_buffer().clone(),
            msaa_samples: settings.msaa_samples,
            pages: Vec::new(),
            page_faces,
            book: MeshBookkeeping::default(),
            meta_buffer,
            meta_capacity: META_CAPACITY,
            terrain_uniforms,
            indirect_buffer: TerrainPass::make_indirect_buffer(device, INDIRECT_CAPACITY),
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
            freeze_buffer,
            frozen: None,
            hiz: None,
            pyramid_placeholder: placeholder_view,
            last_cull: CullStats::default(),
            mdi,
            gpu_capable,
            prefer_gpu_cull: settings.prefer_gpu_cull,
            force_cpu_cull: settings.force_cpu_cull,
            force_fallback: settings.force_fallback,
            applied_paths: [
                settings.prefer_gpu_cull,
                settings.force_cpu_cull,
                settings.force_fallback,
            ],
            stats: Default::default(),
        }
    }

    /// The cull bind group: everything but the frame uniforms (group 0).
    /// Rebuilt whenever any of its resources is recreated (meta growth,
    /// command growth, region growth, pyramid recreation).
    // The ten parameters are the bind group's own entries; a bundling
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
        freeze_buffer: &wgpu::Buffer,
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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: freeze_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub(super) fn rebuild_cull_bind_group(&mut self) {
        let pyramid = self
            .hiz
            .as_ref()
            .map_or(&self.pyramid_placeholder, |h| h.view());
        self.cull_bind_group = TerrainPass::make_cull_bind_group(
            &self.device,
            &self.cull_layout,
            &self.meta_buffer,
            &self.params_buffer,
            &self.commands_buffer,
            &self.region_base_buffer,
            &self.region_counter_buffer,
            &self.stats_buffer,
            pyramid,
            &self.freeze_buffer,
        );
    }

    pub(super) fn make_indirect_buffer(device: &wgpu::Device, commands: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain indirect"),
            size: commands * CMD_BYTES,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(super) fn rebuild_shared_bind_group(&mut self, gpu: &Gpu) {
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.averages_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.materials_buffer.as_entire_binding(),
                },
            ],
        });
    }
}
