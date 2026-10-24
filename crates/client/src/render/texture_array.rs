//! The block texture array: one `D2Array` sRGB texture with CPU-generated
//! mipmaps, plus generated placeholder layers, the per-layer average
//! colors (task 2.9) and the per-layer material table (task 2.15).

use bytemuck::{Pod, Zeroable};
use image::{Rgba, RgbaImage};

use crate::gpu::Gpu;

/// Why a texture array could not be created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextureError {
    /// `layers` was empty.
    Empty,
    /// A layer's dimensions differed from the first layer's.
    MismatchedSizes {
        first: (u32, u32),
        layer: usize,
        got: (u32, u32),
    },
    /// Layers must be square powers of two.
    NotSquarePowerOfTwo((u32, u32)),
    /// More layers than the adapter's `max_texture_array_layers`.
    TooManyLayers { requested: u32, max: u32 },
}

impl std::fmt::Display for TextureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextureError::Empty => write!(f, "texture array needs at least one layer"),
            TextureError::MismatchedSizes { first, layer, got } => write!(
                f,
                "layer {layer} is {got:?} but layer 0 is {first:?}: all layers must match"
            ),
            TextureError::NotSquarePowerOfTwo(size) => {
                write!(f, "layers must be square powers of two, got {size:?}")
            }
            TextureError::TooManyLayers { requested, max } => {
                write!(f, "{requested} layers exceed the adapter limit of {max}")
            }
        }
    }
}

impl std::error::Error for TextureError {}

/// Per-layer material properties (task 2.15), the terrain shader's way to
/// tell layers apart without hard-coding layer numbers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LayerMaterial {
    /// Water: the fragment shader draws Fresnel sky reflection, sun glint
    /// and ripples instead of the texture.
    pub water: bool,
    /// The far-side lean `(layer, strength)`: this layer's flat far color
    /// (the LOD0 fade and every LOD level) leans toward `layer`'s average
    /// by `strength` (0..=1) times the fade. For layers whose top band
    /// differs from the body — grass_side leans toward grass_top — so
    /// distant terrace sides do not read as brown contour stripes.
    pub far_side: Option<(u16, f32)>,
}

/// The material table of the builtin layers, by layer name: water is
/// water, grass_side leans toward grass_top far away. Unknown names get
/// the plain default.
pub fn builtin_materials(names: &[&str]) -> Vec<LayerMaterial> {
    let layer_of = |wanted: &str| {
        names
            .iter()
            .position(|&name| name == wanted)
            .map(|i| i as u16)
    };
    names
        .iter()
        .map(|&name| match name {
            "water" => LayerMaterial {
                water: true,
                far_side: None,
            },
            "grass_side" => LayerMaterial {
                water: false,
                far_side: layer_of("grass_top").map(|top| (top, GRASS_SIDE_FAR_LEAN)),
            },
            _ => LayerMaterial::default(),
        })
        .collect()
}

/// How far grass_side's flat far color leans toward grass_top (task 2.15
/// E): most of the way, so terrace sides read as darker grass (the
/// lighting still separates them) instead of dirt stripes.
pub const GRASS_SIDE_FAR_LEAN: f32 = 0.75;

/// The GPU image of one [`LayerMaterial`] (terrain group 2, binding 5;
/// mirrored as `LayerMaterial` in shaders/terrain.wgsl — the layout test
/// pins it). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct GpuLayerMaterial {
    /// x = water (1.0) or not (0.0); y = the far-side lean strength; z, w
    /// spare.
    pub flags: [f32; 4],
    /// rgb = the linear color the far side leans toward (the target
    /// layer's average); w spare.
    pub far_side: [f32; 4],
}

/// The GPU material table from the specs and the layer averages; layers
/// past the end of `materials` get the default.
fn gpu_materials(materials: &[LayerMaterial], averages: &[[f32; 4]]) -> Vec<GpuLayerMaterial> {
    (0..averages.len())
        .map(|layer| {
            let material = materials.get(layer).copied().unwrap_or_default();
            let (color, strength) = match material.far_side {
                Some((target, strength)) if (target as usize) < averages.len() => {
                    (averages[target as usize], strength.clamp(0.0, 1.0))
                }
                _ => (averages[layer], 0.0),
            };
            GpuLayerMaterial {
                flags: [if material.water { 1.0 } else { 0.0 }, strength, 0.0, 0.0],
                far_side: [color[0], color[1], color[2], 0.0],
            }
        })
        .collect()
}

/// A block texture array: all layers one square power-of-two size, full
/// mip chain generated on the CPU, nearest filtering with linear mips.
/// Also carries each layer's average color (task 2.9's far look) as a CPU
/// table and a storage buffer the terrain pass binds for flat far colors,
/// and the per-layer material table (task 2.15) as a second buffer.
pub struct BlockTextures {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    layers: u32,
    /// Per-layer average color, linear RGB + mean alpha (task 2.9).
    averages: Vec<[f32; 4]>,
    /// The same averages as a storage buffer (one vec4f per layer), bound
    /// by the terrain pass in group 2.
    averages_buffer: wgpu::Buffer,
    /// The per-layer material table (task 2.15), CPU mirror and buffer.
    materials: Vec<GpuLayerMaterial>,
    materials_buffer: wgpu::Buffer,
}

impl BlockTextures {
    /// The builtin placeholder layers with their material table — what
    /// the app and every scene test draw with.
    pub fn builtin(gpu: &Gpu) -> Result<BlockTextures, TextureError> {
        let (layers, names) = crate::mesh::appearance::builtin_textures();
        BlockTextures::with_materials(gpu, &layers, &builtin_materials(&names))
    }

    /// Uploads `layers` with every layer's material at the default (no
    /// water, no far-side lean). See [`BlockTextures::with_materials`].
    pub fn new(gpu: &Gpu, layers: &[RgbaImage]) -> Result<BlockTextures, TextureError> {
        BlockTextures::with_materials(gpu, layers, &[])
    }

    /// Uploads `layers` (all the same square power-of-two size) as a mip
    /// chain, with `materials[i]` describing layer `i` (missing entries
    /// default). Mipmaps are a CPU 2x2 box filter in linear space. Errors
    /// instead of panicking on bad input or adapter limits.
    pub fn with_materials(
        gpu: &Gpu,
        layers: &[RgbaImage],
        materials: &[LayerMaterial],
    ) -> Result<BlockTextures, TextureError> {
        let first = layers
            .first()
            .map(|l| l.dimensions())
            .ok_or(TextureError::Empty)?;
        let (width, height) = first;
        if width == 0 || width != height || !width.is_power_of_two() {
            return Err(TextureError::NotSquarePowerOfTwo(first));
        }
        for (layer, image) in layers.iter().enumerate() {
            if image.dimensions() != first {
                return Err(TextureError::MismatchedSizes {
                    first,
                    layer,
                    got: image.dimensions(),
                });
            }
        }
        let count = layers.len() as u32;
        let max = gpu.adapter.limits().max_texture_array_layers;
        if count > max {
            return Err(TextureError::TooManyLayers {
                requested: count,
                max,
            });
        }

        // The full chain: width, width/2, ..., 1.
        let mip_count = width.trailing_zeros() + 1;
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jbm block textures"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: count,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // mip_chain[mip][layer]
        let mut mips: Vec<Vec<RgbaImage>> = vec![layers.to_vec()];
        for level in 1..mip_count {
            let previous = &mips[level as usize - 1];
            mips.push(previous.iter().map(downsample_2x2).collect());
        }
        for (level, level_images) in mips.iter().enumerate() {
            let mip_size = width >> level;
            for (layer, image) in level_images.iter().enumerate() {
                write_layer_mip(gpu, &texture, image, layer as u32, level as u32, mip_size);
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jbm block texture sampler"),
            // Nearest texel filtering keeps block art crisp; linear mipmap
            // filtering blends between chain levels.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            // Greedy faces span many blocks and tile the texture per
            // block: UVs run past 1.0 and must wrap, with the sampler
            // (not fract() in the shader) doing it so mip derivatives
            // stay correct across tile seams.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        // Task 2.9: each layer's average color, for flat far-terrain
        // shading. Computed in linear space from the base layer (not the
        // 1x1 mip, which is 8-bit quantized), alpha-weighted so cutout
        // holes in leaves and glass do not darken it.
        let averages: Vec<[f32; 4]> = layers.iter().map(layer_average).collect();
        let averages_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm layer averages"),
            size: (count as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&averages_buffer, 0, bytemuck::cast_slice(&averages));
        let materials = gpu_materials(materials, &averages);
        let materials_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm layer materials"),
            size: (count as u64) * std::mem::size_of::<GpuLayerMaterial>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&materials_buffer, 0, bytemuck::cast_slice(&materials));
        Ok(BlockTextures {
            texture,
            view,
            sampler,
            layers: count,
            averages,
            averages_buffer,
            materials,
            materials_buffer,
        })
    }

    /// The per-layer GPU material table (task 2.15) — the CPU mirror of
    /// the materials buffer, for tests.
    pub fn materials(&self) -> &[GpuLayerMaterial] {
        &self.materials
    }

    /// The material table as a storage buffer (one `GpuLayerMaterial`
    /// per layer), bound by the terrain pass in group 2 (binding 5).
    pub fn materials_buffer(&self) -> &wgpu::Buffer {
        &self.materials_buffer
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    pub fn layer_count(&self) -> u32 {
        self.layers
    }

    /// Per-layer average colors (task 2.9), linear RGB with the mean alpha
    /// in w — the CPU mirror of the storage buffer, for tests.
    pub fn averages(&self) -> &[[f32; 4]] {
        &self.averages
    }

    /// The averages as a storage buffer (one vec4f per layer), bound by the
    /// terrain pass in group 2.
    pub fn averages_buffer(&self) -> &wgpu::Buffer {
        &self.averages_buffer
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.format()
    }
}

/// One layer's average color (task 2.9): linear RGB averaged from the base
/// layer, each pixel weighted by its alpha so cutout holes (leaves, glass)
/// contribute nothing, plus the mean alpha in w. A fully transparent layer
/// falls back to the unweighted mean — there is nothing to weight by, and
/// such layers are never sampled where their color shows.
fn layer_average(image: &RgbaImage) -> [f32; 4] {
    let mut sum = [0.0f64; 3];
    let mut weight = 0.0f64;
    let mut alpha_sum = 0.0f64;
    for pixel in image.pixels() {
        let alpha = f64::from(pixel.0[3]) / 255.0;
        let linear = [
            f64::from(srgb_to_linear(pixel.0[0])),
            f64::from(srgb_to_linear(pixel.0[1])),
            f64::from(srgb_to_linear(pixel.0[2])),
        ];
        for c in 0..3 {
            sum[c] += linear[c] * alpha;
        }
        weight += alpha;
        alpha_sum += alpha;
    }
    let n = (image.width() * image.height()) as f64;
    let mean_alpha = alpha_sum / n;
    if weight <= f64::EPSILON * n {
        let mut plain = [0.0f64; 3];
        for pixel in image.pixels() {
            for (c, acc) in plain.iter_mut().enumerate() {
                *acc += f64::from(srgb_to_linear(pixel.0[c]));
            }
        }
        return [
            (plain[0] / n) as f32,
            (plain[1] / n) as f32,
            (plain[2] / n) as f32,
            mean_alpha as f32,
        ];
    }
    [
        (sum[0] / weight) as f32,
        (sum[1] / weight) as f32,
        (sum[2] / weight) as f32,
        mean_alpha as f32,
    ]
}

/// Uploads one mip level of one layer, padding rows to the copy alignment.
fn write_layer_mip(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    image: &RgbaImage,
    layer: u32,
    mip: u32,
    size: u32,
) {
    let row_bytes =
        (size as usize * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
    let mut data = Vec::with_capacity(row_bytes * size as usize);
    for y in 0..size as usize {
        let row = &image.as_raw()[y * size as usize * 4..(y + 1) * size as usize * 4];
        data.extend_from_slice(row);
        data.resize(data.len() + row_bytes - row.len(), 0);
    }
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: mip,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row_bytes as u32),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
}

/// Halves the image with a 2x2 box filter computed in linear space
/// (sRGB -> linear -> average -> sRGB) for RGB. Alpha is already linear,
/// so it is averaged directly without the sRGB curve.
fn downsample_2x2(src: &RgbaImage) -> RgbaImage {
    let (w, h) = src.dimensions();
    let mut out = RgbaImage::new(w / 2, h / 2);
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            let mut linear = [0.0f32; 3];
            let mut alpha_sum = 0u32;
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let p = src.get_pixel(x * 2 + dx, y * 2 + dy).0;
                for c in 0..3 {
                    linear[c] += srgb_to_linear(p[c]);
                }
                alpha_sum += u32::from(p[3]);
            }
            let mut pixel = [0u8; 4];
            for c in 0..3 {
                pixel[c] = linear_to_srgb(linear[c] / 4.0);
            }
            // Linear average of alpha, rounded to nearest.
            pixel[3] = ((alpha_sum + 2) / 4) as u8;
            out.put_pixel(x, y, Rgba(pixel));
        }
    }
    out
}

fn srgb_to_linear(c: u8) -> f32 {
    let v = f32::from(c) / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> u8 {
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Generated 16x16 test layers: distinct hues with a 1-pixel darker
/// border and a white diagonal so orientation is visible. No assets.
pub fn placeholder_layers(n: u32) -> Vec<RgbaImage> {
    (0..n)
        .map(|i| {
            let hue = i as f32 / n as f32 * 360.0;
            let [r, g, b] = hsv_to_rgb(hue, 0.75, 1.0);
            let mut img = RgbaImage::from_pixel(16, 16, Rgba([r, g, b, 255]));
            // 1-pixel darker border.
            let dark = [r / 2, g / 2, b / 2, 255];
            for x in 0..16 {
                img.put_pixel(x, 0, Rgba(dark));
                img.put_pixel(x, 15, Rgba(dark));
            }
            for y in 0..16 {
                img.put_pixel(0, y, Rgba(dark));
                img.put_pixel(15, y, Rgba(dark));
            }
            // White main diagonal.
            for i in 0..16 {
                img.put_pixel(i, i, Rgba([255, 255, 255, 255]));
            }
            img
        })
        .collect()
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let hp = (h % 360.0) / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Copies one layer/mip back to the CPU as a tightly packed image.
    fn read_layer_mip(gpu: &Gpu, textures: &BlockTextures, layer: u32, mip: u32) -> RgbaImage {
        let size = 16u32 >> mip;
        let row_bytes =
            (size as usize * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm texture array readback"),
            size: (row_bytes * size as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &textures.texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes as u32),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = sender.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll failed");
        receiver
            .recv()
            .expect("map callback dropped")
            .expect("map failed");
        let data = buffer
            .slice(..)
            .get_mapped_range()
            .expect("mapped")
            .to_vec();
        buffer.unmap();
        let mut pixels = Vec::with_capacity((size * size * 4) as usize);
        for row in 0..size as usize {
            let start = row * row_bytes;
            pixels.extend_from_slice(&data[start..start + size as usize * 4]);
        }
        RgbaImage::from_raw(size, size, pixels).expect("exact size")
    }

    #[test]
    fn mip0_roundtrips_exactly() {
        let Some(gpu) = crate::testing::screenshot::gpu_or_skip() else {
            return;
        };
        let layers = placeholder_layers(4);
        let textures = BlockTextures::new(&gpu, &layers).expect("array creation");
        assert_eq!(textures.layer_count(), 4);
        let read = read_layer_mip(&gpu, &textures, 2, 0);
        assert_eq!(read.dimensions(), (16, 16));
        assert_eq!(read, layers[2]);
    }

    #[test]
    fn mip1_matches_cpu_box_filter() {
        let Some(gpu) = crate::testing::screenshot::gpu_or_skip() else {
            return;
        };
        let layers = placeholder_layers(4);
        let textures = BlockTextures::new(&gpu, &layers).expect("array creation");
        let read = read_layer_mip(&gpu, &textures, 2, 1);
        let expected = downsample_2x2(&layers[2]);
        for (a, e) in read.pixels().zip(expected.pixels()) {
            for c in 0..4 {
                assert!(
                    (i16::from(a.0[c]) - i16::from(e.0[c])).abs() <= 1,
                    "channel {c}: {} vs {}",
                    a.0[c],
                    e.0[c]
                );
            }
        }
    }

    #[test]
    fn mismatched_sizes_error() {
        let Some(gpu) = crate::testing::screenshot::gpu_or_skip() else {
            return;
        };
        let mut layers = placeholder_layers(2);
        layers.push(RgbaImage::new(8, 8));
        let err = BlockTextures::new(&gpu, &layers)
            .err()
            .expect("creation should fail");
        assert_eq!(
            err,
            TextureError::MismatchedSizes {
                first: (16, 16),
                layer: 2,
                got: (8, 8)
            }
        );
    }

    #[test]
    fn non_square_or_non_power_of_two_errors() {
        let Some(gpu) = crate::testing::screenshot::gpu_or_skip() else {
            return;
        };
        let bad = vec![RgbaImage::new(12, 12)];
        assert_eq!(
            BlockTextures::new(&gpu, &bad)
                .err()
                .expect("creation should fail"),
            TextureError::NotSquarePowerOfTwo((12, 12))
        );
    }

    #[test]
    fn mip1_of_half_alpha_layer_averages_alpha_linearly() {
        // A 16x16 layer half alpha 0, half alpha 255. The 2x2 blocks that
        // straddle the odd-column split average to 128 +- 1; pure blocks
        // stay 0 or 255. Going through the sRGB curve instead of a linear
        // average used to skew the mixed blocks toward 0.
        let mut layer = RgbaImage::from_pixel(16, 16, Rgba([200, 200, 200, 255]));
        for y in 0..16 {
            for x in 0..9 {
                layer.put_pixel(x, y, Rgba([200, 200, 200, 0]));
            }
        }
        let mip1 = downsample_2x2(&layer);
        assert_eq!(mip1.dimensions(), (8, 8));
        let mut mixed = 0;
        for (x, y, pixel) in mip1.enumerate_pixels() {
            let alpha = i16::from(pixel.0[3]);
            // Column 4 covers source columns 8 and 9, one of each half.
            if x == 4 {
                assert!(
                    (alpha - 128).abs() <= 1,
                    "mixed block at ({x}, {y}) has alpha {alpha}, not 128 +- 1"
                );
                mixed += 1;
            } else {
                assert!(
                    alpha == 0 || alpha == 255,
                    "pure block at ({x}, {y}) has alpha {alpha}"
                );
            }
        }
        assert_eq!(mixed, 8, "one mixed column of blocks");
    }

    #[test]
    fn placeholders_are_marked() {
        let layers = placeholder_layers(3);
        assert_eq!(layers.len(), 3);
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(layer.dimensions(), (16, 16));
            // Distinct hues at an off-diagonal interior pixel.
            assert_ne!(
                layers[i].get_pixel(4, 5),
                layers[(i + 1) % 3].get_pixel(4, 5)
            );
            // White diagonal mark.
            assert_eq!(layer.get_pixel(7, 7), &Rgba([255, 255, 255, 255]));
            // Darker border.
            let inner = layer.get_pixel(4, 4).0;
            let border = layer.get_pixel(0, 4).0;
            assert!(border[0] < inner[0] || border[1] < inner[1] || border[2] < inner[2]);
        }
    }

    /// Task 2.9: a 2x2 layer of known colors averages to the linear-space
    /// mean of the four sRGB-decoded primaries.
    #[test]
    fn layer_average_of_known_2x2_is_the_linear_mean() {
        let mut layer = RgbaImage::new(2, 2);
        layer.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        layer.put_pixel(1, 0, Rgba([0, 255, 0, 255]));
        layer.put_pixel(0, 1, Rgba([0, 0, 255, 255]));
        layer.put_pixel(1, 1, Rgba([255, 255, 255, 255]));
        // Every channel is full-scale in exactly two of the four pixels
        // (its primary and white), so each linear mean is half the decoded
        // full scale. The expectation goes through the same sRGB decode as
        // the implementation, so it follows the curve rather than assuming
        // srgb(255) == 1.0.
        let half = f64::from(srgb_to_linear(255)) / 2.0;
        let avg = layer_average(&layer);
        for (c, channel) in avg.iter().enumerate().take(3) {
            assert!(
                (f64::from(*channel) - half).abs() < 1e-6,
                "channel {c}: {channel} vs {half}"
            );
        }
        assert!((avg[3] - 1.0).abs() < 1e-6, "mean alpha of opaque layer");
    }

    /// Task 2.9: transparent pixels do not darken the average — a cutout
    /// layer averages only what it shows. The black transparent pixels
    /// would halve every channel under an unweighted mean.
    #[test]
    fn layer_average_ignores_transparent_pixels() {
        let mut layer = RgbaImage::new(2, 2);
        layer.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        layer.put_pixel(1, 0, Rgba([255, 0, 0, 255]));
        layer.put_pixel(0, 1, Rgba([0, 0, 0, 0]));
        layer.put_pixel(1, 1, Rgba([0, 0, 0, 0]));
        let avg = layer_average(&layer);
        let red = f64::from(srgb_to_linear(255));
        assert!((f64::from(avg[0]) - red).abs() < 1e-6, "r = {}", avg[0]);
        assert!(
            avg[1] < 1e-6 && avg[2] < 1e-6,
            "g/b = {}/{}",
            avg[1],
            avg[2]
        );
        assert!((avg[3] - 0.5).abs() < 1e-6, "mean alpha = {}", avg[3]);
    }

    /// A fully transparent layer has nothing to weight by: the unweighted
    /// linear mean, so the far color of such a layer is still defined.
    #[test]
    fn layer_average_of_fully_transparent_layer_falls_back() {
        let mut layer = RgbaImage::new(2, 2);
        layer.put_pixel(0, 0, Rgba([255, 0, 0, 0]));
        layer.put_pixel(1, 0, Rgba([0, 255, 0, 0]));
        layer.put_pixel(0, 1, Rgba([0, 0, 255, 0]));
        layer.put_pixel(1, 1, Rgba([0, 0, 0, 0]));
        let avg = layer_average(&layer);
        // Each channel is full-scale in exactly one of the four pixels,
        // so every unweighted mean is a quarter of the decoded scale.
        let quarter = f64::from(srgb_to_linear(255)) / 4.0;
        for (c, channel) in avg.iter().enumerate().take(3) {
            assert!(
                (f64::from(*channel) - quarter).abs() < 1e-6,
                "channel {c}: {channel} vs {quarter}"
            );
        }
        assert!(avg[3] < 1e-6);
    }

    /// The builtin layers have usable averages: cutout layers (glass,
    /// leaves) stay bright, grass top is green, and every average is a
    /// finite color in [0, 1].
    #[test]
    fn builtin_layer_averages_are_sane() {
        let (layers, _) = crate::mesh::appearance::builtin_textures();
        for layer in &layers {
            let avg = layer_average(layer);
            for c in 0..4 {
                assert!(
                    avg[c].is_finite() && (0.0..=1.0).contains(&avg[c]),
                    "{avg:?}"
                );
            }
        }
        let grass = layer_average(&layers[2]); // grass_top
        assert!(grass[1] > grass[0] && grass[1] > grass[2], "{grass:?}");
        let leaves = layer_average(&layers[10]); // leaves (cutout)
        assert!(
            leaves[1] > 0.05,
            "cutout holes must not blacken: {leaves:?}"
        );
    }

    /// The material table's GPU layout matches the WGSL struct
    /// (vec4f + vec4f, 32 bytes), and the builtin table marks exactly
    /// water as water and leans grass_side toward grass_top's average.
    #[test]
    fn builtin_materials_mark_water_and_the_grass_side_lean() {
        assert_eq!(std::mem::size_of::<GpuLayerMaterial>(), 32);
        assert_eq!(std::mem::offset_of!(GpuLayerMaterial, far_side), 16);
        let (layers, names) = crate::mesh::appearance::builtin_textures();
        let materials = builtin_materials(&names);
        let averages: Vec<[f32; 4]> = layers.iter().map(layer_average).collect();
        let gpu = gpu_materials(&materials, &averages);
        assert_eq!(gpu.len(), layers.len());
        for (i, name) in names.iter().enumerate() {
            let water = *name == "water";
            assert_eq!(gpu[i].flags[0] == 1.0, water, "{name}");
            if *name == "grass_side" {
                let top = names.iter().position(|&n| n == "grass_top").unwrap();
                assert_eq!(gpu[i].flags[1], GRASS_SIDE_FAR_LEAN);
                assert_eq!(&gpu[i].far_side[..3], &averages[top][..3]);
            } else {
                assert_eq!(gpu[i].flags[1], 0.0, "{name} leans nowhere");
            }
        }
        // Missing specs default; an out-of-range lean target is ignored.
        let odd = gpu_materials(
            &[LayerMaterial {
                water: false,
                far_side: Some((99, 1.0)),
            }],
            &averages[..2],
        );
        assert_eq!(odd[0].flags, [0.0; 4]);
        assert_eq!(odd[1].flags, [0.0; 4]);
    }
}
