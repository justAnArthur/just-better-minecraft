//! The debug line pass (task 2.12): one small LineList pipeline the app
//! feeds CPU-built, camera-relative vertices — the chunk-border boxes
//! around resident cubes and the frozen cull frustum. Depth-tested
//! against the terrain so hidden borders hide; one pixel wide (WebGPU
//! has no wide lines). The renderer owns the pass and draws it after the
//! terrain when any vertices are set; offscreen rigs that never set any
//! see no change.

use bytemuck::{Pod, Zeroable};

use crate::gpu::Gpu;
use crate::render::renderer::DEPTH_FORMAT;

/// One line endpoint: a camera-relative position (the same space the
/// terrain renders in) and a linear RGBA color. Pairs of vertices form
/// lines. Mirrored as the WGSL vertex layout — the test at the bottom
/// pins it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    /// xyz = the endpoint, camera-relative in blocks; w spare.
    pub pos: [f32; 4],
    /// Linear RGBA.
    pub color: [f32; 4],
}

const LINES_WGSL: &str = r#"
struct FrameUniforms {
    view_proj: mat4x4f,
};
struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
};
@group(0) @binding(0) var<uniform> u: FrameUniforms;

@vertex
fn vs_main(@location(0) pos: vec4f, @location(1) color: vec4f) -> VsOut {
    var out: VsOut;
    out.pos = u.view_proj * vec4f(pos.xyz, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    return in.color;
}
"#;

/// The line pass: one pipeline over a growable vertex buffer. Empty by
/// default; `set_vertices` uploads one frame's lines and the renderer's
/// draw skips the pass entirely while the buffer is empty.
pub struct LinesPass {
    pipeline: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
    /// Vertices currently in the buffer (lines = half that).
    count: u32,
    /// Vertices the buffer holds.
    capacity: u32,
}

const INITIAL_CAPACITY: u32 = 4096;

impl LinesPass {
    /// Creates the pass for `color_format`, sharing the renderer's frame
    /// uniform layout (binding 0, `view_proj`).
    pub fn new(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        samples: u32,
    ) -> LinesPass {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm debug lines"),
            source: wgpu::ShaderSource::Wgsl(LINES_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jbm debug lines layout"),
            bind_group_layouts: &[Some(uniform_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jbm debug lines pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // pos: location 0; color: location 1.
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
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
                ..Default::default()
            },
            // Depth-tested against the terrain (Greater, reverse-Z) but
            // never writing: the lines annotate geometry, they are not
            // geometry.
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
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm debug lines"),
            size: u64::from(INITIAL_CAPACITY) * std::mem::size_of::<LineVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        LinesPass {
            pipeline,
            buffer,
            count: 0,
            capacity: INITIAL_CAPACITY,
        }
    }

    /// Replaces the frame's lines. An empty slice simply empties the pass
    /// (the draw skips it); a full buffer grows with the input.
    pub fn set_vertices(&mut self, gpu: &Gpu, vertices: &[LineVertex]) {
        self.count = vertices.len() as u32;
        if self.count == 0 {
            return;
        }
        if self.count > self.capacity {
            self.capacity = self.count.next_power_of_two();
            self.buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jbm debug lines"),
                size: u64::from(self.capacity) * std::mem::size_of::<LineVertex>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        gpu.queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(vertices));
    }

    /// Vertices currently set (test hook).
    pub fn vertex_count(&self) -> u32 {
        self.count
    }

    /// Records the lines into the renderer's scene pass (task 2.16: no
    /// pass of their own). The caller skips it while empty.
    pub(super) fn record(&self, pass: &mut wgpu::RenderPass<'_>, uniforms: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, uniforms, &[]);
        pass.set_vertex_buffer(0, self.buffer.slice(..));
        pass.draw(0..self.count, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WGSL vertex layout: two Float32x4 attributes, 32 bytes apart.
    #[test]
    fn line_vertex_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<LineVertex>(), 32);
        assert_eq!(std::mem::offset_of!(LineVertex, pos), 0);
        assert_eq!(std::mem::offset_of!(LineVertex, color), 16);
    }
}
