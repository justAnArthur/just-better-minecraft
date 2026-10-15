//! The frame renderer: the per-frame uniform buffer, the sky pass, the
//! terrain pass, and dispatch into the debug overlay.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use jbm_core::coords::BlockPos;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::render::chunks::TerrainPass;
use crate::render::outline::OutlinePass;
use crate::render::overlay::{OverlayRenderer, OverlayText};

/// Depth buffer format. Reverse-Z (cleared to 0.0, `Greater` compare) with
/// an infinite far plane keeps precision near the camera for Stage 2's
/// kilometre view distances.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Sky gradient colors (linear RGB; the sRGB target converts on store).
/// Documented in crates/client/README.md.
pub const HORIZON_COLOR: [f32; 4] = [0.70, 0.82, 0.94, 1.0];
pub const ZENITH_COLOR: [f32; 4] = [0.15, 0.40, 0.82, 1.0];
pub const GROUND_HAZE_COLOR: [f32; 4] = [0.34, 0.37, 0.43, 1.0];

/// Per-frame GPU constants.
///
/// Mirrored as `struct FrameUniforms` in `shaders/sky.wgsl` and
/// `shaders/overlay.wgsl`. Layout rule: every field is 16-byte aligned
/// (mat4 or padded vec4), so Rust `#[repr(C)]` and WGSL agree with no
/// hidden padding — enforced by `frame_uniforms_match_wgsl_layout`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FrameUniforms {
    /// Rotation-only view-projection (camera-relative rendering).
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    /// Camera world position split into f32 hi/lo halves: hi + lo ≈ position.
    pub camera_pos_hi: [f32; 4],
    pub camera_pos_lo: [f32; 4],
    pub horizon_color: [f32; 4],
    pub zenith_color: [f32; 4],
    pub ground_color: [f32; 4],
    /// x, y = target size in pixels (overlay needs pixel→NDC). z, w unused.
    pub viewport: [f32; 4],
}

/// A depth attachment matching [`DEPTH_FORMAT`], recreated on resize.
pub struct DepthTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl DepthTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> DepthTarget {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jbm depth target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        DepthTarget {
            texture,
            view,
            width,
            height,
        }
    }

    /// Recreates the texture if the size changed; does nothing otherwise.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        *self = DepthTarget::new(device, width, height);
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.format()
    }
}

/// Owns the pipelines and buffers needed to draw a frame (sky + terrain +
/// overlay) into a target of one specific color format. Create one per format: the
/// window surface (usually `Bgra8UnormSrgb`) and the offscreen test target
/// (`Rgba8UnormSrgb`) differ.
pub struct Renderer {
    uniform_buffer: wgpu::Buffer,
    uniform_layout: wgpu::BindGroupLayout,
    uniform_bind_group: wgpu::BindGroup,
    sky_pipeline: wgpu::RenderPipeline,
    overlay: OverlayRenderer,
    outline: OutlinePass,
    viewport: (u32, u32),
}

impl Renderer {
    pub fn new(gpu: &Gpu, color_format: wgpu::TextureFormat) -> Renderer {
        let device = &gpu.device;

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm frame uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jbm frame uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jbm frame uniforms"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm sky"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sky.wgsl").into()),
        });
        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jbm sky layout"),
            bind_group_layouts: &[Some(&uniform_layout)],
            immediate_size: 0,
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jbm sky pipeline"),
            layout: Some(&sky_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let overlay = OverlayRenderer::new(gpu, color_format, &uniform_layout);
        let outline = OutlinePass::new(gpu, color_format, &uniform_layout);

        Renderer {
            uniform_buffer,
            uniform_layout,
            uniform_bind_group,
            sky_pipeline,
            overlay,
            outline,
            viewport: (1, 1),
        }
    }

    /// The frame-uniform bind group layout (binding 0 in every pipeline),
    /// for passes created outside the renderer (terrain, tests).
    pub fn uniform_layout(&self) -> &wgpu::BindGroupLayout {
        &self.uniform_layout
    }

    /// Sets the target size in pixels (used by the overlay's pixel-space
    /// quads). Call after resize and before the first render.
    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.viewport = (width, height);
    }

    pub fn viewport(&self) -> (u32, u32) {
        self.viewport
    }

    fn build_uniforms(&self, camera: &Camera) -> FrameUniforms {
        let view_proj = camera.view_proj();
        let hi = camera.position.as_vec3();
        let lo = (camera.position - hi.as_dvec3()).as_vec3();
        FrameUniforms {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            camera_pos_hi: [hi.x, hi.y, hi.z, 0.0],
            camera_pos_lo: [lo.x, lo.y, lo.z, 0.0],
            horizon_color: HORIZON_COLOR,
            zenith_color: ZENITH_COLOR,
            ground_color: GROUND_HAZE_COLOR,
            viewport: [self.viewport.0 as f32, self.viewport.1 as f32, 0.0, 0.0],
        }
    }

    /// Draws one frame: the sky pass (which also clears depth to 0.0),
    /// then the terrain pass, then the target outline (`outline`, task
    /// 1.7), then the debug overlay — crosshair included — on top.
    #[allow(clippy::too_many_arguments)] // one optional input per pass; bundle into a frame struct when the next pass lands
    pub fn render(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        camera: &Camera,
        overlay: Option<&OverlayText>,
        terrain: Option<&mut TerrainPass>,
        outline: Option<BlockPos>,
    ) {
        let uniforms = self.build_uniforms(camera);
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jbm frame"),
            });

        // Sky pass: fullscreen triangle; also clears depth to 0.0 (far in
        // reverse-Z) so later passes can depth-test against it.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("jbm sky pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        if let Some(terrain) = terrain {
            terrain.draw(
                gpu,
                &mut encoder,
                target,
                depth,
                camera,
                &self.uniform_bind_group,
            );
        }

        if let Some(block) = outline {
            self.outline.draw(
                gpu,
                &mut encoder,
                target,
                depth,
                &self.uniform_bind_group,
                camera,
                block,
            );
        }

        if let Some(text) = overlay {
            self.overlay.draw(
                gpu,
                &mut encoder,
                target,
                &self.uniform_bind_group,
                self.viewport,
                text,
            );
        }

        gpu.queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WGSL mirror in shaders/sky.wgsl / shaders/overlay.wgsl is
    /// 2 × mat4x4f + 6 × vec4f = 224 bytes, every field 16-byte aligned.
    #[test]
    fn frame_uniforms_match_wgsl_layout() {
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 224);
        assert_eq!(std::mem::align_of::<FrameUniforms>(), 16);
        assert_eq!(std::mem::offset_of!(FrameUniforms, view_proj), 0);
        assert_eq!(std::mem::offset_of!(FrameUniforms, inv_view_proj), 64);
        assert_eq!(std::mem::offset_of!(FrameUniforms, camera_pos_hi), 128);
        assert_eq!(std::mem::offset_of!(FrameUniforms, camera_pos_lo), 144);
        assert_eq!(std::mem::offset_of!(FrameUniforms, horizon_color), 160);
        assert_eq!(std::mem::offset_of!(FrameUniforms, zenith_color), 176);
        assert_eq!(std::mem::offset_of!(FrameUniforms, ground_color), 192);
        assert_eq!(std::mem::offset_of!(FrameUniforms, viewport), 208);
        assert!(
            std::mem::size_of::<FrameUniforms>().is_multiple_of(16),
            "no tail padding"
        );
    }
}
