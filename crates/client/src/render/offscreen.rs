//! An offscreen render target that can be read back as an RGBA image.

use crate::gpu::Gpu;
use image::RgbaImage;

/// Owns a single-sample `Rgba8UnormSrgb` texture usable as a render
/// attachment and copy source.
pub struct OffscreenTarget {
    texture: wgpu::Texture,
    width: u32,
    height: u32,
}

impl OffscreenTarget {
    /// Creates a target of the given size on `device`.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> OffscreenTarget {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jbm offscreen target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        OffscreenTarget {
            texture,
            width,
            height,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.format()
    }

    /// A default view of the texture, suitable as a render-pass color
    /// attachment.
    pub fn create_view(&self) -> wgpu::TextureView {
        self.texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Copies the texture to a staging buffer, maps it, strips row
    /// padding and returns the pixels as an image.
    pub fn read_rgba(&self, gpu: &Gpu) -> RgbaImage {
        let bytes_per_pixel = 4;
        let unaligned = bytes_per_pixel * self.width;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.max(bytes_per_pixel);
        let padded = unaligned.div_ceil(alignment) * alignment;

        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm offscreen readback"),
            size: u64::from(padded) * u64::from(self.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jbm offscreen readback"),
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
        // Wait for the submitted copy and the mapping to complete. A poll
        // failure must panic in release builds too (with the error), not
        // hang on recv() waiting for a callback that will never fire.
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
        let mut pixels = Vec::with_capacity((unaligned * self.height) as usize);
        for row in 0..self.height {
            let start = (row * padded) as usize;
            pixels.extend_from_slice(&data[start..start + unaligned as usize]);
        }
        drop(data);
        buffer.unmap();

        RgbaImage::from_raw(self.width, self.height, pixels)
            .expect("row-stripped buffer is exactly width * height * 4 bytes")
    }
}
