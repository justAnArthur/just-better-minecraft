//! The block texture array: one `D2Array` sRGB texture with CPU-generated
//! mipmaps, plus generated placeholder layers. Nothing samples it yet; the
//! Stage 1 mesher (task 1.4) will.

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

/// A block texture array: all layers one square power-of-two size, full
/// mip chain generated on the CPU, nearest filtering with linear mips.
pub struct BlockTextures {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    layers: u32,
}

impl BlockTextures {
    /// Uploads `layers` (all the same square power-of-two size) as a mip
    /// chain. Mipmaps are a CPU 2x2 box filter in linear space. Errors
    /// instead of panicking on bad input or adapter limits.
    pub fn new(gpu: &Gpu, layers: &[RgbaImage]) -> Result<BlockTextures, TextureError> {
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
            ..Default::default()
        });
        Ok(BlockTextures {
            texture,
            view,
            sampler,
            layers: count,
        })
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

    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.format()
    }
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
/// (sRGB -> linear -> average -> sRGB).
fn downsample_2x2(src: &RgbaImage) -> RgbaImage {
    let (w, h) = src.dimensions();
    let mut out = RgbaImage::new(w / 2, h / 2);
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            let mut linear = [0.0f32; 4];
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let p = src.get_pixel(x * 2 + dx, y * 2 + dy).0;
                for c in 0..4 {
                    linear[c] += srgb_to_linear(p[c]);
                }
            }
            let mut pixel = [0u8; 4];
            for c in 0..4 {
                pixel[c] = linear_to_srgb(linear[c] / 4.0);
            }
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
}
