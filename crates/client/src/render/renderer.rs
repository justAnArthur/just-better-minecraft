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
/// `TEXTURE_BINDING` lets the Hi-Z fill pass (task 2.5) sample it.
pub struct DepthTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    /// Bumped on every recreation, so the terrain pass can rebuild its
    /// Hi-Z fill bind group even when the size did not change.
    generation: u32,
}

impl DepthTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> DepthTarget {
        static NEXT_GENERATION: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        DepthTarget {
            texture,
            view,
            width,
            height,
            // Globally unique, so two different targets of the same size
            // never match (the Hi-Z fill bind group must follow the exact
            // texture, not just its size).
            generation: NEXT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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

    /// Size and recreation generation: the key the terrain pass matches
    /// its Hi-Z pyramid against.
    pub fn description(&self) -> (u32, u32, u32) {
        (self.width, self.height, self.generation)
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.format()
    }
}

/// The inputs of one frame, bundled (task 2.5): the render target and its
/// depth attachment, the camera, and one optional input per pass — the
/// terrain pass, the target outline and the debug overlay.
pub struct FrameInputs<'a> {
    pub target: &'a wgpu::TextureView,
    pub depth: &'a DepthTarget,
    pub camera: &'a Camera,
    pub terrain: Option<&'a mut TerrainPass>,
    pub outline: Option<BlockPos>,
    pub overlay: Option<&'a OverlayText>,
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
    /// Sky color overrides (task 2.6 hole tests): linear RGB per slot.
    sky_override: Option<([f32; 4], [f32; 4], [f32; 4])>,
    /// Whether terrain distance fog applies (task 2.9); `JBM_NO_FOG=1` at
    /// construction, later `set_fog`. Propagated to the terrain pass every
    /// frame — the flag rides in the terrain uniform.
    fog: bool,
    /// Whether the level tint applies (task 2.10); set by the app's F4
    /// toggle / `JBM_LOD_TINT=1` at startup, propagated to the terrain
    /// pass every frame like the fog flag.
    lod_tint: bool,
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
                // COMPUTE too: the terrain cull pass (task 2.5) reads
                // view_proj on the GPU.
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
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

        // The sky gradient snippet is concatenated ahead of the sky text so
        // the terrain pass's fog can evaluate the identical function (WGSL
        // has no includes; task 2.9). It has no dependencies, so leading
        // with it is valid.
        let sky_source = format!(
            "{}{}",
            include_str!("shaders/sky_gradient.wgsl"),
            include_str!("shaders/sky.wgsl")
        );
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm sky"),
            source: wgpu::ShaderSource::Wgsl(sky_source.into()),
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
            sky_override: None,
            fog: !std::env::var_os("JBM_NO_FOG").is_some_and(|v| v == "1"),
            lod_tint: false,
        }
    }

    /// Enables or disables terrain distance fog (task 2.9). Off via
    /// `JBM_NO_FOG=1` at construction; screenshot tests use this to isolate
    /// geometry from fog. Fog never touches the sky pass, so the sentinel
    /// hole tests keep working with it on: a hole still shows the sentinel
    /// sky, while fogged terrain dissolves into the horizon color.
    pub fn set_fog(&mut self, enabled: bool) {
        self.fog = enabled;
    }

    /// Whether distance fog is on (task 2.9).
    pub fn fog(&self) -> bool {
        self.fog
    }

    /// Enables or disables the level tint (task 2.10), the debug view
    /// that multiplies every fragment by a fixed color per LOD level.
    /// The app's F4 key toggles it; `JBM_LOD_TINT=1` starts it on.
    /// Propagates to the terrain pass every frame like the fog flag.
    pub fn set_lod_tint(&mut self, enabled: bool) {
        self.lod_tint = enabled;
    }

    /// Whether the level tint is on (task 2.10).
    pub fn lod_tint(&self) -> bool {
        self.lod_tint
    }

    /// Overrides the sky colors (horizon, zenith, ground haze; linear
    /// RGB). Test hook for hole detection: set the **ground** color —
    /// the sky below the horizon — to a sentinel such as pure magenta
    /// and count sentinel pixels; any hole in the terrain below the
    /// horizon shows it. Pass `None` to restore the built-in colors.
    pub fn set_sky_colors(&mut self, colors: Option<([f32; 4], [f32; 4], [f32; 4])>) {
        self.sky_override = colors;
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
        let (horizon, zenith, ground) =
            self.sky_override
                .unwrap_or((HORIZON_COLOR, ZENITH_COLOR, GROUND_HAZE_COLOR));
        FrameUniforms {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            camera_pos_hi: [hi.x, hi.y, hi.z, 0.0],
            camera_pos_lo: [lo.x, lo.y, lo.z, 0.0],
            horizon_color: horizon,
            zenith_color: zenith,
            ground_color: ground,
            viewport: [self.viewport.0 as f32, self.viewport.1 as f32, 0.0, 0.0],
        }
    }

    /// Draws one frame: the sky pass (which also clears depth to 0.0),
    /// then the terrain pass, then the target outline (`outline`, task
    /// 1.7), then the debug overlay — crosshair included — on top.
    pub fn render(&mut self, gpu: &Gpu, frame: FrameInputs<'_>) {
        let uniforms = self.build_uniforms(frame.camera);
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
                    view: frame.target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: frame.depth.view(),
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

        if let Some(terrain) = frame.terrain {
            // The fog on/off hook rides along each frame (task 2.9): the
            // terrain pass bakes it into its per-frame uniform. The level
            // tint flag (task 2.10) rides the same way.
            terrain.set_fog(self.fog);
            terrain.set_lod_tint(self.lod_tint);
            terrain.draw(
                gpu,
                &mut encoder,
                frame.target,
                frame.depth,
                frame.camera,
                &self.uniform_bind_group,
            );
        }

        if let Some(block) = frame.outline {
            self.outline.draw(
                gpu,
                &mut encoder,
                frame.target,
                frame.depth.view(),
                &self.uniform_bind_group,
                frame.camera,
                block,
            );
        }

        if let Some(text) = frame.overlay {
            self.overlay.draw(
                gpu,
                &mut encoder,
                frame.target,
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
