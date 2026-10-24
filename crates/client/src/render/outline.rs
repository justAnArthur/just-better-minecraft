//! The target outline: a line-list box around the block under the
//! crosshair. Dark (60% black), depth-tested against the terrain so
//! hidden edges disappear, and expanded slightly so it never z-fights
//! with the block faces it hugs.

use glam::DVec3;

use jbm_core::coords::BlockPos;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::render::renderer::DEPTH_FORMAT;

/// How far the box sits outside the block, in blocks.
const EXPAND: f64 = 0.002;

/// The 12 box edges as pairs of corner indices (see `CORNERS`).
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0), // bottom
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4), // top
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // verticals
];

/// Draws the outline. One pipeline, one small vertex buffer rewritten per
/// frame; positions are computed camera-relative in f64 on the CPU (world
/// coordinates never reach f32).
pub struct OutlinePass {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
}

impl OutlinePass {
    pub fn new(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        samples: u32,
    ) -> OutlinePass {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm outline"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/outline.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jbm outline layout"),
            bind_group_layouts: &[Some(uniform_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jbm outline pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            // Drawn into the renderer's multisampled scene target (task 2.15).
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm outline vertices"),
            size: 24 * 12,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        OutlinePass { pipeline, vertices }
    }

    /// Writes the outline of `block` (camera-relative) for this frame's
    /// [`OutlinePass::record`].
    pub fn prepare(&mut self, gpu: &Gpu, camera: &Camera, block: BlockPos) {
        // The 8 corners of the expanded box, camera-relative.
        let min = DVec3::new(
            block.x as f64 - EXPAND,
            block.y as f64 - EXPAND,
            block.z as f64 - EXPAND,
        );
        let max = DVec3::new(
            block.x as f64 + 1.0 + EXPAND,
            block.y as f64 + 1.0 + EXPAND,
            block.z as f64 + 1.0 + EXPAND,
        );
        let corners: [[f32; 3]; 8] = std::array::from_fn(|i| {
            let corner = DVec3::new(
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            );
            camera.relative(corner).to_array()
        });
        let mut data = [0f32; 24 * 3];
        for (edge, (a, b)) in EDGES.iter().enumerate() {
            data[edge * 6..edge * 6 + 3].copy_from_slice(&corners[*a]);
            data[edge * 6 + 3..edge * 6 + 6].copy_from_slice(&corners[*b]);
        }
        gpu.queue
            .write_buffer(&self.vertices, 0, bytemuck::cast_slice(&data));
    }

    /// Records the outline into the renderer's scene pass (task 2.16: no
    /// pass of its own), depth-tested against the terrain.
    pub fn record(&self, pass: &mut wgpu::RenderPass<'_>, uniforms: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, uniforms, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw(0..24, 0..1);
    }
}
