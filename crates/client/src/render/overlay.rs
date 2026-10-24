//! Debug overlay: font8x8 text as instanced screen-space quads, plus the
//! pure-logic FrameStats that feeds its FPS line.

use std::collections::VecDeque;
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use font8x8::UnicodeFonts;

use crate::gpu::Gpu;

/// Glyph atlas covers printable ASCII.
const FIRST_CHAR: u8 = 32;
const CHAR_COUNT: u32 = 95;
const GLYPH_SIZE: u32 = 8;
const ATLAS_WIDTH: u32 = GLYPH_SIZE * CHAR_COUNT;

/// Text to draw: one opaque line each. Filled by the app layer.
#[derive(Debug, Default)]
pub struct OverlayText {
    pub lines: Vec<String>,
}

impl OverlayText {
    pub fn new() -> OverlayText {
        OverlayText::default()
    }

    pub fn push(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }
}

/// Averages frame time over a sliding window (0.5 s by default). Pure
/// logic; unit-tested with synthetic frame times. The window holds frames
/// that ended within the last `window` seconds. A second, count-based
/// window of the last [`SPREAD_WINDOW`] frames feeds the debug panel's
/// average / p99 / worst line.
#[derive(Debug)]
pub struct FrameStats {
    window: f64,
    /// (frame end time, duration) in seconds; end times are cumulative.
    samples: VecDeque<(f64, f64)>,
    /// The last `SPREAD_WINDOW` durations, in seconds.
    spread: VecDeque<f64>,
    now: f64,
}

/// How many recent frames the debug panel's frame-time spread covers
/// (task 2.12): two seconds at 60 fps.
const SPREAD_WINDOW: usize = 120;

impl Default for FrameStats {
    fn default() -> Self {
        FrameStats::new(0.5)
    }
}

impl FrameStats {
    pub fn new(window_secs: f64) -> FrameStats {
        FrameStats {
            window: window_secs,
            samples: VecDeque::new(),
            spread: VecDeque::new(),
            now: 0.0,
        }
    }

    /// Records one frame's duration and drops samples that ended before
    /// the window started.
    pub fn push(&mut self, frame_time: Duration) {
        let duration = frame_time.as_secs_f64();
        self.now += duration;
        self.samples.push_back((self.now, duration));
        self.spread.push_back(duration);
        if self.spread.len() > SPREAD_WINDOW {
            self.spread.pop_front();
        }
        let cutoff = self.now - self.window;
        while let Some(&(end, _)) = self.samples.front() {
            if end < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// The frame-time spread the debug panel prints: `(average, p99,
    /// worst)` in milliseconds over the last 120 frames (zeros with no
    /// samples yet). The p99 index is `(len - 1) · 0.99` rounded, so it
    /// lands on the worst sample only from 100 frames on.
    pub fn frame_ms_spread(&self) -> (f64, f64, f64) {
        if self.spread.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut sorted: Vec<f64> = self.spread.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let average = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let p99_index = ((sorted.len() as f64 - 1.0) * 0.99).round() as usize;
        (
            average * 1000.0,
            sorted[p99_index.min(sorted.len() - 1)] * 1000.0,
            sorted[sorted.len() - 1] * 1000.0,
        )
    }

    fn total(&self) -> f64 {
        self.samples.iter().map(|&(_, d)| d).sum()
    }

    /// Average frames per second over the window (0.0 with no samples).
    pub fn fps(&self) -> f64 {
        let total = self.total();
        if total <= 0.0 {
            0.0
        } else {
            self.samples.len() as f64 / total
        }
    }

    /// Average frame duration in milliseconds (0.0 with no samples).
    pub fn frame_ms(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.total() * 1000.0 / self.samples.len() as f64
        }
    }
}

/// One instanced screen-space quad: a rectangle in pixels with atlas UVs
/// and an RGBA color. Mirrored as `QuadInstance` in shaders/overlay.wgsl;
/// both are 48 bytes with no hidden padding (the vec4f at offset 32 makes
/// WGSL's 16-byte struct alignment agree with Rust's `#[repr(C)]`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QuadInstance {
    /// Top-left corner in pixels.
    pub pos: [f32; 2],
    /// Size in pixels.
    pub size: [f32; 2],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub color: [f32; 4],
}

/// Builds the R8 glyph atlas for ASCII 32..127 from font8x8 (public
/// domain), rows padded to the 256-byte copy alignment.
fn glyph_atlas_pixels() -> Vec<u8> {
    let padded_row = (ATLAS_WIDTH as usize).next_multiple_of(256);
    let mut data = vec![0u8; padded_row * GLYPH_SIZE as usize];
    for char_index in 0..CHAR_COUNT {
        let ch = (FIRST_CHAR + char_index as u8) as char;
        let Some(glyph) = font8x8::BASIC_FONTS.get(ch) else {
            continue;
        };
        for y in 0..GLYPH_SIZE {
            for x in 0..GLYPH_SIZE {
                // Bit x of row y is pixel (x, y); bit 0 is the leftmost pixel.
                if glyph[y as usize] & (1 << x) != 0 {
                    let px = char_index * GLYPH_SIZE + x;
                    data[y as usize * padded_row + px as usize] = 255;
                }
            }
        }
    }
    data
}

/// Draws `OverlayText`: instanced glyph quads sampled from a font8x8
/// atlas, each line backed by a semi-transparent dark box, with alpha
/// blending on.
pub struct OverlayRenderer {
    atlas_bind_group: wgpu::BindGroup,
    glyph_pipeline: wgpu::RenderPipeline,
    solid_pipeline: wgpu::RenderPipeline,
    instance_buffer: wgpu::Buffer,
    capacity: u64,
    /// The prepared frame's (solid quads, all quads) instance counts.
    counts: (u32, u32),
    /// Integer glyph scale: 2 renders 8x8 glyphs as 16x16 px.
    pub scale: u32,
}

/// Screen layout in pixels.
const MARGIN: u32 = 4;
const PAD: u32 = 2;

const BG_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 0.55];
const TEXT_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// Crosshair geometry in pixels: white arms over a dark outline, so it
/// reads on both sky and terrain.
const CROSS_ARM: f32 = 6.0;
const CROSS_THICK: f32 = 2.0;
const CROSS_OUTLINE: f32 = 1.0;
const CROSS_DARK: [f32; 4] = [0.0, 0.0, 0.0, 0.7];
const CROSS_WHITE: [f32; 4] = [1.0, 1.0, 1.0, 0.9];

impl OverlayRenderer {
    pub fn new(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
    ) -> OverlayRenderer {
        let device = &gpu.device;
        let queue = &gpu.queue;

        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jbm glyph atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_WIDTH,
                height: GLYPH_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let padded_row = (ATLAS_WIDTH as usize).next_multiple_of(256) as u32;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &glyph_atlas_pixels(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: ATLAS_WIDTH,
                height: GLYPH_SIZE,
                depth_or_array_layers: 1,
            },
        );
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jbm glyph sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let atlas_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jbm glyph atlas layout"),
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
        let atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jbm glyph atlas"),
            layout: &atlas_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jbm overlay layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&atlas_layout)],
            immediate_size: 0,
        });
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 24,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 4,
                },
            ],
        };
        let make_pipeline = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("jbm overlay pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(instance_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let solid_pipeline = make_pipeline("fs_solid");
        let glyph_pipeline = make_pipeline("fs_glyph");

        let capacity = 1024u64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm overlay instances"),
            size: capacity * std::mem::size_of::<QuadInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        OverlayRenderer {
            atlas_bind_group,
            glyph_pipeline,
            solid_pipeline,
            instance_buffer,
            capacity,
            counts: (0, 0),
            scale: 2,
        }
    }

    /// Lays out `text` (plus the crosshair: a small centered white cross
    /// with a dark outline, visible on both sky and terrain) and writes
    /// the instances for this frame's [`OverlayRenderer::record`].
    /// `viewport` is the target size in pixels (crosshair centering).
    pub fn prepare(&mut self, gpu: &Gpu, viewport: (u32, u32), text: &OverlayText) {
        let (boxes, glyphs) = self.layout(text);
        // Solid quads: line boxes then the crosshair (dark bars under
        // white arms). Glyph quads follow.
        let mut instances = boxes;
        let solid_count = instances.len() as u32 + 4;
        instances.extend(crosshair_quads(viewport));
        instances.extend(glyphs);
        let instance_count = instances.len() as u32;
        self.counts = (solid_count, instance_count);
        if instance_count == 0 {
            return;
        }

        let bytes = bytemuck::cast_slice(&instances);
        if instance_count as u64 > self.capacity {
            self.capacity = (instance_count as u64).next_power_of_two();
            self.instance_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jbm overlay instances"),
                size: self.capacity * std::mem::size_of::<QuadInstance>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        gpu.queue.write_buffer(&self.instance_buffer, 0, bytes);
    }

    /// Records the prepared overlay into `pass`, a single-sample pass over
    /// the frame target (task 2.16: the renderer's present pass, after
    /// the scene), drawn over what is there.
    pub fn record(&self, pass: &mut wgpu::RenderPass<'_>, uniforms: &wgpu::BindGroup) {
        let (solid_count, instance_count) = self.counts;
        if instance_count == 0 {
            return;
        }
        pass.set_bind_group(0, uniforms, &[]);
        // Both pipelines share this layout, so the atlas bind group is set
        // even for the solid quads (which do not sample it).
        pass.set_bind_group(1, &self.atlas_bind_group, &[]);
        pass.set_vertex_buffer(0, self.instance_buffer.slice(..));

        pass.set_pipeline(&self.solid_pipeline);
        pass.draw(0..6, 0..solid_count);
        pass.set_pipeline(&self.glyph_pipeline);
        pass.draw(0..6, solid_count..instance_count);
    }

    /// Turns lines into background-box instances and glyph instances, all
    /// in pixels (top-left origin). The caller interleaves the crosshair
    /// quads between the two halves.
    fn layout(&self, text: &OverlayText) -> (Vec<QuadInstance>, Vec<QuadInstance>) {
        let scale = self.scale;
        let glyph_w = GLYPH_SIZE * scale;
        let glyph_h = GLYPH_SIZE * scale;
        let mut boxes = Vec::new();
        let mut glyphs = Vec::new();
        for (line_index, line) in text.lines.iter().enumerate() {
            let y = MARGIN + line_index as u32 * (glyph_h + 2 * PAD);
            let chars: Vec<char> = line
                .chars()
                .filter(|c| c.is_ascii_graphic() || *c == ' ')
                .collect();
            if chars.is_empty() {
                continue;
            }
            let text_w = chars.len() as u32 * glyph_w;
            boxes.push(QuadInstance {
                pos: [(MARGIN - PAD) as f32, (y - PAD) as f32],
                size: [(text_w + 2 * PAD) as f32, (glyph_h + 2 * PAD) as f32],
                uv_min: [0.0, 0.0],
                uv_max: [0.0, 0.0],
                color: BG_COLOR,
            });
            for (char_index, ch) in chars.iter().enumerate() {
                let column = *ch as u8 - FIRST_CHAR;
                let x0 = MARGIN + char_index as u32 * glyph_w;
                glyphs.push(QuadInstance {
                    pos: [x0 as f32, y as f32],
                    size: [glyph_w as f32, glyph_h as f32],
                    uv_min: [
                        (column as u32 * GLYPH_SIZE) as f32 / ATLAS_WIDTH as f32,
                        0.0,
                    ],
                    uv_max: [
                        ((column as u32 + 1) * GLYPH_SIZE) as f32 / ATLAS_WIDTH as f32,
                        1.0,
                    ],
                    color: TEXT_COLOR,
                });
            }
        }
        (boxes, glyphs)
    }
}

/// The crosshair as four solid quads: two dark outline bars drawn under
/// two white arms, centered on the target.
fn crosshair_quads(viewport: (u32, u32)) -> [QuadInstance; 4] {
    let (cx, cy) = (viewport.0 as f32 / 2.0, viewport.1 as f32 / 2.0);
    let quad = |x: f32, y: f32, w: f32, h: f32, color: [f32; 4]| QuadInstance {
        pos: [x, y],
        size: [w, h],
        uv_min: [0.0, 0.0],
        uv_max: [0.0, 0.0],
        color,
    };
    let o = CROSS_OUTLINE;
    [
        // Dark outline: each bar padded by `o` on every side.
        quad(
            cx - CROSS_ARM - o,
            cy - CROSS_THICK / 2.0 - o,
            2.0 * (CROSS_ARM + o),
            CROSS_THICK + 2.0 * o,
            CROSS_DARK,
        ),
        quad(
            cx - CROSS_THICK / 2.0 - o,
            cy - CROSS_ARM - o,
            CROSS_THICK + 2.0 * o,
            2.0 * (CROSS_ARM + o),
            CROSS_DARK,
        ),
        // White arms on top.
        quad(
            cx - CROSS_ARM,
            cy - CROSS_THICK / 2.0,
            2.0 * CROSS_ARM,
            CROSS_THICK,
            CROSS_WHITE,
        ),
        quad(
            cx - CROSS_THICK / 2.0,
            cy - CROSS_ARM,
            CROSS_THICK,
            2.0 * CROSS_ARM,
            CROSS_WHITE,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_average_sixty_fps() {
        let mut stats = FrameStats::new(0.5);
        for _ in 0..30 {
            stats.push(Duration::from_secs_f64(1.0 / 60.0));
        }
        assert!((stats.fps() - 60.0).abs() < 0.1, "{}", stats.fps());
        assert!(
            (stats.frame_ms() - 1000.0 / 60.0).abs() < 0.01,
            "{}",
            stats.frame_ms()
        );
    }

    #[test]
    fn stats_window_drops_old_frames() {
        let mut stats = FrameStats::new(0.5);
        for _ in 0..60 {
            stats.push(Duration::from_secs_f64(1.0 / 1000.0));
        }
        // 25 slow frames fill exactly 0.5 s and evict every fast frame.
        for _ in 0..25 {
            stats.push(Duration::from_secs_f64(0.02));
        }
        assert!((stats.fps() - 50.0).abs() < 0.1, "{}", stats.fps());
        assert!(
            (stats.frame_ms() - 20.0).abs() < 0.01,
            "{}",
            stats.frame_ms()
        );
    }

    #[test]
    fn stats_empty_is_zero() {
        let stats = FrameStats::default();
        assert_eq!(stats.fps(), 0.0);
        assert_eq!(stats.frame_ms(), 0.0);
        assert_eq!(stats.frame_ms_spread(), (0.0, 0.0, 0.0));
    }

    /// The debug panel's spread (task 2.12): average, p99 and worst over
    /// the last 120 frames in milliseconds, and only the last 120 — an
    /// ancient 5 s frame must age out of all three.
    #[test]
    fn spread_reports_avg_p99_worst_of_the_last_120_frames() {
        let mut stats = FrameStats::default();
        for _ in 0..120 {
            stats.push(Duration::from_secs_f64(1.0 / 60.0));
        }
        // One huge frame, then fill past the window so it ages out.
        stats.push(Duration::from_secs_f64(5.0));
        for _ in 0..120 {
            stats.push(Duration::from_secs_f64(1.0 / 60.0));
        }
        let (avg, p99, worst) = stats.frame_ms_spread();
        let sixth = 1000.0 / 60.0;
        assert!((avg - sixth).abs() < 0.01, "avg {avg}");
        assert!((p99 - sixth).abs() < 0.01, "p99 {p99}");
        assert!((worst - sixth).abs() < 0.01, "worst {worst}");
        // Without the fill, four huge frames sit in the top 2%: p99 and
        // worst both reach them, and the average moves toward them.
        let mut stats = FrameStats::default();
        for _ in 0..200 {
            stats.push(Duration::from_secs_f64(0.010));
        }
        for _ in 0..4 {
            stats.push(Duration::from_secs_f64(0.050));
        }
        let (avg, p99, worst) = stats.frame_ms_spread();
        // The window holds the last 120 frames only: 116 fast + 4 slow.
        assert!(
            (avg - (116.0 * 10.0 + 4.0 * 50.0) / 120.0).abs() < 0.01,
            "avg {avg}"
        );
        assert!((worst - 50.0).abs() < 1e-9, "worst {worst}");
        assert!((p99 - 50.0).abs() < 1e-9, "p99 {p99}");
    }

    #[test]
    fn quad_instance_layout_is_packed() {
        assert_eq!(std::mem::size_of::<QuadInstance>(), 48);
        assert_eq!(std::mem::offset_of!(QuadInstance, color), 32);
    }

    #[test]
    fn crosshair_quads_center_a_white_cross_over_its_dark_outline() {
        let quads = crosshair_quads((256, 128));
        // Dark bars first, white arms after; all centered at (128, 64).
        for (i, quad) in quads.iter().enumerate() {
            let cx = quad.pos[0] + quad.size[0] / 2.0;
            let cy = quad.pos[1] + quad.size[1] / 2.0;
            assert_eq!((cx, cy), (128.0, 64.0), "quad {i} centered");
        }
        assert_eq!(quads[0].color, super::CROSS_DARK);
        assert_eq!(quads[1].color, super::CROSS_DARK);
        assert_eq!(quads[2].color, super::CROSS_WHITE);
        assert_eq!(quads[3].color, super::CROSS_WHITE);
        // The outline is strictly larger than the arm it sits under.
        assert!(quads[0].size[0] > quads[2].size[0] && quads[0].size[1] > quads[2].size[1]);
        assert!(quads[1].size[0] > quads[3].size[0] && quads[1].size[1] > quads[3].size[1]);
    }

    #[test]
    fn atlas_encodes_glyph_bits() {
        let atlas = glyph_atlas_pixels();
        let row = (ATLAS_WIDTH as usize).next_multiple_of(256);
        let px =
            |c: u8, x: usize, y: usize| atlas[y * row + (c as usize - FIRST_CHAR as usize) * 8 + x];
        // Space is blank.
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(px(b' ', x, y), 0);
            }
        }
        // 'A' apex: BASIC_FONTS['A'][0] == 0x0C, so pixels (2,0) and (3,0)
        // are set and (0,0) is clear.
        assert_eq!(px(b'A', 2, 0), 255);
        assert_eq!(px(b'A', 3, 0), 255);
        assert_eq!(px(b'A', 0, 0), 0);
        // The last atlas char, '~', contributes at least one pixel.
        assert!(atlas.contains(&255));
    }
}
