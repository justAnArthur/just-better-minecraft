//! The frame renderer: the per-frame uniform buffer, the sky pass, the
//! terrain pass, and dispatch into the debug overlay.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use jbm_core::coords::BlockPos;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::render::lines::{LineVertex, LinesPass};
use crate::render::outline::OutlinePass;
use crate::render::overlay::{OverlayRenderer, OverlayText};
use crate::render::settings::RenderSettings;
use crate::render::terrain::TerrainPass;
use crate::render::timing::{GpuTimer, GpuTimings};

/// Depth buffer format. Reverse-Z (cleared to 0.0, `Greater` compare) with
/// an infinite far plane keeps precision near the camera for Stage 2's
/// kilometre view distances.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Sky gradient colors (linear RGB; the sRGB target converts on store).
/// Documented in crates/client/README.md.
pub const HORIZON_COLOR: [f32; 4] = [0.70, 0.82, 0.94, 1.0];
pub const ZENITH_COLOR: [f32; 4] = [0.15, 0.40, 0.82, 1.0];
pub const GROUND_HAZE_COLOR: [f32; 4] = [0.34, 0.37, 0.43, 1.0];

/// The fixed sun (task 2.15; no day cycle yet): a mid-morning sun
/// [`SUN_ELEVATION_DEGREES`] above the horizon, its horizontal direction
/// [`SUN_AZIMUTH_XZ`] (toward the sun, not axis-aligned so all four block
/// sides read differently). [`sun_direction`] combines them.
pub const SUN_ELEVATION_DEGREES: f32 = 36.0;
pub const SUN_AZIMUTH_XZ: [f32; 2] = [-0.80, 0.60];
/// Sun radiance (linear RGB): warm white. A sunlit top face receives
/// `SUN_COLOR · sin(elevation)` plus the sky ambient.
pub const SUN_COLOR: [f32; 3] = [1.00, 0.85, 0.62];
/// The hemisphere ambient (linear RGB): sky blue from straight above,
/// a dimmer warm ground bounce from straight below, blended by the
/// normal's y.
pub const AMBIENT_SKY: [f32; 3] = [0.32, 0.40, 0.54];
pub const AMBIENT_GROUND: [f32; 3] = [0.21, 0.20, 0.19];

/// The unit vector toward the sun (task 2.15), mirrored by the shaders
/// through `FrameUniforms::sun_dir`.
pub fn sun_direction() -> Vec3 {
    let (sin_e, cos_e) = SUN_ELEVATION_DEGREES.to_radians().sin_cos();
    let h = glam::Vec2::from(SUN_AZIMUTH_XZ).normalize();
    Vec3::new(h.x * cos_e, sin_e, h.y * cos_e)
}

/// The period, in blocks, of every world-anchored procedural pattern
/// (clouds, water ripples; task 2.15). The camera position reaches the
/// shaders reduced modulo this in f64 (`FrameUniforms::camera_wrap`), so
/// the patterns stay precise at any world coordinate; every pattern
/// tiles exactly with this period, so the wrap never shows a seam.
pub const WRAP_PERIOD: f64 = 65536.0;

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
    /// Task 2.15. xyz = the unit vector toward the sun; w = the animation
    /// time in seconds (cloud drift, water ripples; 0 in tests).
    pub sun_dir: [f32; 4],
    /// rgb = the sun's radiance (linear); w = sun lighting on (1.0) or
    /// off (0.0, the pre-2.15 flat look).
    pub sun_color: [f32; 4],
    /// rgb = the sky-hemisphere ambient (linear); w = clouds on / off.
    pub ambient_sky: [f32; 4],
    /// rgb = the ground-bounce ambient (linear); w = water shading on /
    /// off.
    pub ambient_ground: [f32; 4],
    /// xyz = the camera position modulo [`WRAP_PERIOD`] (reduced in f64):
    /// the origin of the world-anchored procedural patterns. w spare.
    pub camera_wrap: [f32; 4],
}

/// A depth attachment matching [`DEPTH_FORMAT`], recreated on resize.
/// `TEXTURE_BINDING` lets the Hi-Z fill pass (task 2.5) sample it. Its
/// sample count must match the renderer's (task 2.15: 4× MSAA by default,
/// [`DepthTarget::with_samples`] for 1×).
pub struct DepthTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    samples: u32,
    /// Bumped on every recreation, so the terrain pass can rebuild its
    /// Hi-Z fill bind group even when the size did not change.
    generation: u32,
}

impl DepthTarget {
    /// A depth target at the default MSAA sample count
    /// ([`DEFAULT_MSAA_SAMPLES`](crate::render::settings::DEFAULT_MSAA_SAMPLES)),
    /// matching `Renderer::new`.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> DepthTarget {
        DepthTarget::with_samples(
            device,
            width,
            height,
            crate::render::settings::DEFAULT_MSAA_SAMPLES,
        )
    }

    /// A depth target with `samples` samples per pixel (1 or 4). Only a
    /// single-sample target can be read back (`read_depth`).
    pub fn with_samples(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        samples: u32,
    ) -> DepthTarget {
        static NEXT_GENERATION: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jbm depth target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            // COPY_SRC (the read_depth hook) only where a copy is legal:
            // multisampled textures cannot be copied to a buffer.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | if samples == 1 {
                    wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::empty()
                },
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        DepthTarget {
            texture,
            view,
            width,
            height,
            samples,
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
        *self = DepthTarget::with_samples(device, width, height, self.samples);
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The last frame's per-pixel depths, row-major, `width * height`
    /// f32s (task 2.10's test hook; reverse-Z, so
    /// distance = [`crate::camera::NEAR`] / depth). Blocking: encodes one
    /// texture→buffer copy of the whole target and maps it. Single-sample
    /// targets only (a multisampled texture cannot be copied out); tests
    /// that read depth render at 1×.
    pub fn read_depth(&self, gpu: &Gpu) -> Vec<f32> {
        assert_eq!(
            self.samples, 1,
            "read_depth needs a single-sample depth target (DepthTarget::with_samples(.., 1))"
        );
        let bytes_per_pixel = 4;
        let unaligned = bytes_per_pixel * self.width;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.max(bytes_per_pixel);
        let padded = unaligned.div_ceil(alignment) * alignment;

        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm depth readback"),
            size: u64::from(padded) * u64::from(self.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jbm depth readback"),
            });
        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        let poll = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        assert!(poll.is_ok(), "device poll failed: {poll:?}");
        receiver
            .recv()
            .expect("map callback dropped without sending")
            .expect("buffer mapping failed");
        let data = buffer
            .slice(..)
            .get_mapped_range()
            .expect("range was just mapped for reading");
        let mut depths = Vec::with_capacity((self.width * self.height) as usize);
        let row_bytes = unaligned as usize;
        for y in 0..self.height as usize {
            let row = &data[y * padded as usize..y * padded as usize + row_bytes];
            depths.extend(
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        drop(data);
        buffer.unmap();
        depths
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

    /// Samples per pixel (1, or 4 for MSAA).
    pub fn samples(&self) -> u32 {
        self.samples
    }
}

/// The multisampled color attachment (task 2.15) the sky, terrain, debug
/// lines and outline render into; resolved into the frame's target before
/// the overlay draws at 1×. Owned by the renderer (one color format) and
/// recreated whenever the depth target's size changes.
struct MsaaColor {
    view: wgpu::TextureView,
    size: (u32, u32),
}

impl MsaaColor {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        (width, height): (u32, u32),
        samples: u32,
    ) -> MsaaColor {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jbm msaa color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        MsaaColor {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            size: (width, height),
        }
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
    /// The task 2.12 debug lines (chunk borders, the frozen frustum);
    /// skipped while no vertices are set.
    lines: LinesPass,
    viewport: (u32, u32),
    /// Sky color overrides (hole tests): linear RGB per slot.
    sky_override: Option<([f32; 4], [f32; 4], [f32; 4])>,
    /// The animation time in seconds (cloud drift, water ripples); the
    /// app advances it, tests leave it at 0.
    time: f32,
    color_format: wgpu::TextureFormat,
    /// The multisampled color attachment while `msaa_samples > 1`,
    /// created on the first frame and on every resize.
    msaa_color: Option<MsaaColor>,
    /// The frame-level settings (fog, level tint, far-look radii, path
    /// flags) the renderer owns; passed to the terrain pass with every
    /// draw. Seeded from `RenderSettings::from_env()` by the app through
    /// `set_settings`; defaults otherwise.
    settings: RenderSettings,
    /// The pass timer (task 2.16), when the device has timestamp queries.
    timer: Option<GpuTimer>,
}

impl Renderer {
    /// A renderer with the default settings (4× MSAA).
    pub fn new(gpu: &Gpu, color_format: wgpu::TextureFormat) -> Renderer {
        Renderer::with_settings(gpu, color_format, RenderSettings::default())
    }

    /// A renderer seeded with `settings`; its pipelines are built for
    /// `settings.msaa_samples`, so depth targets passed to `render` must
    /// have that many samples.
    pub fn with_settings(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        settings: RenderSettings,
    ) -> Renderer {
        let device = &gpu.device;
        let samples = settings.msaa_samples;
        let multisample = wgpu::MultisampleState {
            count: samples,
            ..Default::default()
        };

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
                depth_compare: Some(wgpu::CompareFunction::Equal),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample,
            multiview_mask: None,
            cache: None,
        });

        // The overlay draws after the MSAA resolve, at 1×.
        let overlay = OverlayRenderer::new(gpu, color_format, &uniform_layout);
        let outline = OutlinePass::new(gpu, color_format, &uniform_layout, samples);
        let lines = LinesPass::new(gpu, color_format, &uniform_layout, samples);

        Renderer {
            uniform_buffer,
            uniform_layout,
            uniform_bind_group,
            sky_pipeline,
            overlay,
            outline,
            lines,
            viewport: (1, 1),
            sky_override: None,
            time: 0.0,
            color_format,
            msaa_color: None,
            settings,
            timer: GpuTimer::new(gpu),
        }
    }

    /// Replaces the renderer-owned settings wholesale, except the MSAA
    /// sample count: the pipelines were built for the construction-time
    /// count (`Renderer::with_settings`), which therefore stays.
    pub fn set_settings(&mut self, settings: RenderSettings) {
        self.settings = RenderSettings {
            msaa_samples: self.settings.msaa_samples,
            ..settings
        };
    }

    /// The newest GPU pass times (task 2.16), one or two frames late;
    /// `None` without timestamp queries or before the first readback.
    pub fn gpu_timings(&self) -> Option<&GpuTimings> {
        self.timer.as_ref().and_then(GpuTimer::latest)
    }

    /// Waits for the submitted frames and returns the newest GPU pass
    /// times (the perf harness).
    pub fn read_gpu_timings_blocking(&mut self, gpu: &Gpu) -> Option<GpuTimings> {
        self.timer.as_mut().and_then(|t| t.read_blocking(gpu))
    }

    /// The MSAA sample count the pipelines were built for.
    pub fn msaa_samples(&self) -> u32 {
        self.settings.msaa_samples
    }

    /// Sets the animation time in seconds (task 2.15): cloud drift and
    /// water ripples follow it. The app advances it every frame; tests
    /// leave it at 0 so images stay deterministic.
    pub fn set_time(&mut self, seconds: f32) {
        self.time = seconds;
    }

    /// The renderer-owned settings the terrain draws run with.
    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    /// Mutable access for the debug panel (task 2.12): it edits the
    /// settings; the renderer applies them next frame.
    pub fn settings_mut(&mut self) -> &mut RenderSettings {
        &mut self.settings
    }

    /// Enables or disables terrain distance fog. Off via `JBM_NO_FOG=1`
    /// (through `RenderSettings::from_env`); screenshot tests use this to
    /// isolate geometry from fog. Fog never touches the sky pass, so the
    /// sentinel hole tests keep working with it on: a hole still shows
    /// the sentinel sky, while fogged terrain dissolves into the horizon
    /// color.
    pub fn set_fog(&mut self, enabled: bool) {
        self.settings.fog = enabled;
    }

    /// Whether distance fog is on.
    pub fn fog(&self) -> bool {
        self.settings.fog
    }

    /// Enables or disables the level tint, the debug view that
    /// multiplies every fragment by a fixed color per LOD level. The
    /// app's F4 key toggles it; `JBM_LOD_TINT=1` starts it on.
    pub fn set_lod_tint(&mut self, enabled: bool) {
        self.settings.lod_tint = enabled;
    }

    /// Whether the level tint is on.
    pub fn lod_tint(&self) -> bool {
        self.settings.lod_tint
    }

    /// Sets the far-look view distances: the LOD0 ring radius and the
    /// furthest ring radius, both in blocks. The LOD0 radius sets where
    /// LOD0's textures fade into the flat average colors
    /// (`FADE_START_FRACTION`..`FADE_END_FRACTION` of it, so both sides
    /// of the LOD0/LOD1 border are flat); the furthest radius sets the
    /// distance-fog window (`FOG_START_FRACTION`..`FOG_END_FRACTION` of
    /// it, so the outermost ring's edge is never visible). Defaults match
    /// the `ServerConfig` defaults; the game passes its own values.
    pub fn set_view_distance(&mut self, lod0_ring_radius: f32, furthest_ring_radius: f32) {
        self.settings.far_radii = [lod0_ring_radius, furthest_ring_radius];
    }

    /// Replaces the frame's debug lines (task 2.12): the chunk-border
    /// boxes and the frozen cull frustum, camera-relative. The app
    /// rebuilds the list whenever its debug views need it; an empty list
    /// stands the pass down (offscreen tests that never call this see no
    /// change).
    pub fn set_debug_lines(&mut self, gpu: &Gpu, vertices: &[LineVertex]) {
        self.lines.set_vertices(gpu, vertices);
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
        let sun = sun_direction();
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        let wrap = camera.position.map(|c| c.rem_euclid(WRAP_PERIOD)).as_vec3();
        FrameUniforms {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            camera_pos_hi: [hi.x, hi.y, hi.z, 0.0],
            camera_pos_lo: [lo.x, lo.y, lo.z, 0.0],
            horizon_color: horizon,
            zenith_color: zenith,
            ground_color: ground,
            viewport: [self.viewport.0 as f32, self.viewport.1 as f32, 0.0, 0.0],
            sun_dir: [sun.x, sun.y, sun.z, self.time],
            sun_color: [
                SUN_COLOR[0],
                SUN_COLOR[1],
                SUN_COLOR[2],
                flag(self.settings.sun_lighting),
            ],
            ambient_sky: [
                AMBIENT_SKY[0],
                AMBIENT_SKY[1],
                AMBIENT_SKY[2],
                flag(self.settings.clouds),
            ],
            ambient_ground: [
                AMBIENT_GROUND[0],
                AMBIENT_GROUND[1],
                AMBIENT_GROUND[2],
                flag(self.settings.water),
            ],
            camera_wrap: [wrap.x, wrap.y, wrap.z, 0.0],
        }
    }

    /// Draws one frame: a clear (depth to 0.0), the terrain pass, the sky
    /// pass over the pixels the terrain left uncovered, the debug lines,
    /// the target outline (`outline`, task 1.7), then the debug overlay — crosshair included — on top. With
    /// MSAA (task 2.15) everything before the overlay renders into the
    /// renderer's multisampled color target and `frame.depth` (which must
    /// have the same sample count), resolved into `frame.target` before
    /// the overlay.
    pub fn render(&mut self, gpu: &Gpu, frame: FrameInputs<'_>) {
        let samples = self.settings.msaa_samples;
        assert_eq!(
            frame.depth.samples(),
            samples,
            "the depth target's sample count must match the renderer's MSAA setting"
        );
        if samples > 1
            && self
                .msaa_color
                .as_ref()
                .is_none_or(|m| m.size != frame.depth.size())
        {
            self.msaa_color = Some(MsaaColor::new(
                &gpu.device,
                self.color_format,
                frame.depth.size(),
                samples,
            ));
        }
        // The scene's color attachment: the multisampled target, or the
        // frame's own target at 1×.
        let msaa = samples > 1;
        let scene_target = match &self.msaa_color {
            Some(color) if msaa => &color.view,
            _ => frame.target,
        };
        let uniforms = self.build_uniforms(frame.camera);
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jbm frame"),
            });
        if let Some(timer) = &mut self.timer {
            timer.begin_frame(gpu);
        }
        let timer = self.timer.as_ref();

        // The clear: color, and depth to 0.0 (far in reverse-Z) so the
        // terrain can depth-test against it. Without terrain the sky pass
        // clears for itself.
        let terrain_drawn = frame.terrain.is_some();
        if let Some(terrain) = frame.terrain {
            drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("jbm clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_target,
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
                timestamp_writes: timer.and_then(|t| t.render_writes("clear")),
                occlusion_query_set: None,
                multiview_mask: None,
            }));
            // The frame settings (fog, level tint) ride along each draw:
            // the terrain pass bakes them into its per-frame uniform.
            terrain.draw(
                gpu,
                &mut encoder,
                scene_target,
                frame.depth,
                frame.camera,
                &self.uniform_bind_group,
                &self.settings,
                timer,
            );
        }

        // Sky pass (task 2.15: after the terrain): a fullscreen triangle at
        // the far plane, depth-tested Equal against the cleared depth, so
        // it shades only the pixels no terrain covered — the clouds' noise
        // is never evaluated under the terrain (a cave frame used to pay
        // for a full screen of it).
        {
            let (color_load, depth_load) = if terrain_drawn {
                (wgpu::LoadOp::Load, wgpu::LoadOp::Load)
            } else {
                (
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    wgpu::LoadOp::Clear(0.0),
                )
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("jbm sky pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: color_load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: frame.depth.view(),
                    depth_ops: Some(wgpu::Operations {
                        load: depth_load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timer.and_then(|t| t.render_writes("sky")),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // The task 2.12 debug lines, over the terrain (depth-tested
        // against it) and under the outline and overlay.
        if self.lines.vertex_count() > 0 {
            self.lines.draw(
                &mut encoder,
                scene_target,
                frame.depth.view(),
                &self.uniform_bind_group,
            );
        }

        if let Some(block) = frame.outline {
            self.outline.draw(
                gpu,
                &mut encoder,
                scene_target,
                frame.depth.view(),
                &self.uniform_bind_group,
                frame.camera,
                block,
            );
        }

        // The MSAA resolve (task 2.15): an empty pass whose only job is
        // resolving the multisampled scene into the frame's target, so
        // the overlay (and egui after it) draw at 1× on top.
        if msaa {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("jbm msaa resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_target,
                    depth_slice: None,
                    resolve_target: Some(frame.target),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timer.and_then(|t| t.render_writes("resolve")),
                occlusion_query_set: None,
                multiview_mask: None,
            });
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

        if let Some(timer) = &mut self.timer {
            timer.end_frame(&mut encoder);
        }
        gpu.queue.submit([encoder.finish()]);
        if let Some(timer) = &mut self.timer {
            timer.after_submit(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WGSL mirror in shaders/sky.wgsl / shaders/overlay.wgsl /
    /// shaders/outline.wgsl / shaders/terrain.wgsl is 2 × mat4x4f +
    /// 11 × vec4f = 304 bytes, every field 16-byte aligned.
    #[test]
    fn frame_uniforms_match_wgsl_layout() {
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 304);
        assert_eq!(std::mem::align_of::<FrameUniforms>(), 16);
        assert_eq!(std::mem::offset_of!(FrameUniforms, view_proj), 0);
        assert_eq!(std::mem::offset_of!(FrameUniforms, inv_view_proj), 64);
        assert_eq!(std::mem::offset_of!(FrameUniforms, camera_pos_hi), 128);
        assert_eq!(std::mem::offset_of!(FrameUniforms, camera_pos_lo), 144);
        assert_eq!(std::mem::offset_of!(FrameUniforms, horizon_color), 160);
        assert_eq!(std::mem::offset_of!(FrameUniforms, zenith_color), 176);
        assert_eq!(std::mem::offset_of!(FrameUniforms, ground_color), 192);
        assert_eq!(std::mem::offset_of!(FrameUniforms, viewport), 208);
        assert_eq!(std::mem::offset_of!(FrameUniforms, sun_dir), 224);
        assert_eq!(std::mem::offset_of!(FrameUniforms, sun_color), 240);
        assert_eq!(std::mem::offset_of!(FrameUniforms, ambient_sky), 256);
        assert_eq!(std::mem::offset_of!(FrameUniforms, ambient_ground), 272);
        assert_eq!(std::mem::offset_of!(FrameUniforms, camera_wrap), 288);
        assert!(
            std::mem::size_of::<FrameUniforms>().is_multiple_of(16),
            "no tail padding"
        );
    }

    /// The fixed sun is a mid-morning sun: 35–55° up, not axis-aligned
    /// (every side face gets a different share of it).
    #[test]
    fn sun_is_mid_morning_and_not_axis_aligned() {
        let sun = sun_direction();
        assert!((sun.length() - 1.0).abs() < 1e-5);
        let elevation = sun.y.asin().to_degrees();
        assert!((35.0..=55.0).contains(&elevation), "{elevation}");
        let sides = [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z].map(|n| n.dot(sun));
        for i in 0..4 {
            for j in i + 1..4 {
                assert!((sides[i] - sides[j]).abs() > 0.05, "{sides:?}");
            }
        }
    }
}
