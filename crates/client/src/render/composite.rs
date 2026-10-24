//! The render-scale composite (task 2.16): while the scene renders below
//! the window's resolution (`RenderSettings::render_scale`), it resolves
//! into a 1× scene color texture of its own size, and this pass upscales
//! that into the window with a Catmull-Rom filter (`composite.wgsl`).
//! The overlay then draws in the same pass at native resolution. The
//! renderer owns the pass; at scale 1 it stands down and the scene
//! resolves straight into the frame's target, as before.

use crate::gpu::Gpu;

/// The upscale pipeline and the scene color texture it samples.
pub struct CompositePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    color_format: wgpu::TextureFormat,
    /// The scene color texture and its bind group, created on the first
    /// scaled frame and on every scene resize.
    scene: Option<SceneColor>,
}

struct SceneColor {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

impl CompositePass {
    /// A composite into `color_format` targets.
    pub fn new(gpu: &Gpu, color_format: wgpu::TextureFormat) -> CompositePass {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm composite"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jbm composite layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jbm composite pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jbm composite pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
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
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        // Bilinear taps (the shader builds the bicubic from them),
        // clamped so the border texels do not wrap.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jbm composite sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        CompositePass {
            pipeline,
            layout,
            sampler,
            color_format,
            scene: None,
        }
    }

    /// The 1× scene color texture of `size`, (re)created when the size
    /// changed: the scene pass's resolve target (or its color attachment
    /// at 1× MSAA).
    pub fn scene_view(&mut self, device: &wgpu::Device, size: (u32, u32)) -> &wgpu::TextureView {
        if self.scene.as_ref().is_none_or(|s| s.size != size) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("jbm scene color"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jbm composite bind group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.scene = Some(SceneColor {
                view,
                bind_group,
                size,
            });
        }
        &self.scene.as_ref().expect("created above").view
    }

    /// Draws the upscaled scene over the whole target. Call after
    /// [`CompositePass::scene_view`] in the same frame.
    pub fn record(&self, pass: &mut wgpu::RenderPass<'_>) {
        let scene = self
            .scene
            .as_ref()
            .expect("scene_view creates the scene color first");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
