//! The hierarchical-Z depth pyramid (task 2.5): an `R32Float` texture with
//! a full mip chain over the depth target. Mip 0 is a compute copy of the
//! depth target; mip k+1 is a 2x2 **min** reduction of mip k, because in
//! reverse-Z the smaller depth is the farther surface and occlusion needs
//! the farthest one. The occlusion cull in `shaders/cull.wgsl` samples the
//! pyramid. See `render/chunks.rs` for how it is used per frame.

use crate::gpu::Gpu;

/// The pyramid texture plus everything needed to rebuild it each frame.
/// Recreated when the depth target is (see `matches_depth`). The views
/// keep the texture alive; it is not stored directly.
pub struct HizPyramid {
    /// All mips in one view, sampled by the cull shader (`textureLoad` with
    /// an explicit level).
    full_view: wgpu::TextureView,
    fill_pipeline: wgpu::ComputePipeline,
    fill_bind_group: wgpu::BindGroup,
    reduce_pipeline: wgpu::ComputePipeline,
    /// One bind group per reduction step: mip k sampled, mip k+1 stored.
    reduce_bind_groups: Vec<wgpu::BindGroup>,
    /// Mip sizes from mip 0 up.
    sizes: Vec<(u32, u32)>,
    generation: u32,
}

/// One reduction step, the exact rule `shaders/hiz.wgsl` implements:
/// `dst(x, y)` is the min over `src` in `[2x ..= xe] x [2y ..= ye]`, where
/// the last dst texel of each axis takes everything remaining
/// (`xe = src_w - 1`), and every other dst texel takes its 2-texel pair:
///
/// ```text
/// src (5 wide)          dst (2 wide)    odd edge: the extra last
/// x 0 1 | 2 3 4         x 0   1         column 4 folds into dst 1
/// dst 0 covers {0,1}, dst 1 (last) covers {2,3,4}
/// ```
///
/// The fold makes every source texel contribute exactly once, so no texel
/// of mip k is ever skipped (the classic Hi-Z bug) and every mip texel is
/// the min over exactly the mip-0 texels its footprint covers.
/// Floor-halved sizes keep the chain inside wgpu's mip limit
/// (`floor(log2(max(w, h))) + 1`).
pub fn reduce_step(src: &[f32], src_w: u32, src_h: u32) -> (Vec<f32>, u32, u32) {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);
    let mut dst = vec![0.0f32; (dst_w * dst_h) as usize];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let xe = if x + 1 == dst_w { src_w - 1 } else { 2 * x + 1 };
            let ye = if y + 1 == dst_h { src_h - 1 } else { 2 * y + 1 };
            let mut m = f32::INFINITY;
            for sy in 2 * y..=ye {
                for sx in 2 * x..=xe {
                    m = m.min(src[sy as usize * src_w as usize + sx as usize]);
                }
            }
            dst[y as usize * dst_w as usize + x as usize] = m;
        }
    }
    (dst, dst_w, dst_h)
}

impl HizPyramid {
    /// Builds a pyramid for a depth target of `width` x `height` and the
    /// bind group that copies the depth into mip 0. `depth_view` must be a
    /// single-mip view of the depth texture.
    pub fn new(
        gpu: &Gpu,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        generation: u32,
    ) -> HizPyramid {
        let device = &gpu.device;

        // One layout shared by the fill and every reduction bind group:
        // a sampled source, a write-only R32Float storage destination and
        // a (src w, src h, dst w, dst h) uniform. Binding mip k as sampled
        // and mip k+1 as storage is legal: wgpu tracks usage per
        // subresource, and two different mips never overlap.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jbm hiz layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        // Depth32Float and R32Float both sample as
                        // non-filterable float.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm hiz"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hiz.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jbm hiz layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("jbm hiz pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let fill_pipeline = make_pipeline("fill_mip0");
        let reduce_pipeline = make_pipeline("reduce_mip");

        // Full mip chain down to 1x1, floor-halved like wgpu's own mip
        // sizing (at most floor(log2(max(w, h))) + 1 levels, which the
        // device enforces).
        let mut sizes = Vec::new();
        let (mut w, mut h) = (width, height);
        loop {
            sizes.push((w, h));
            if w == 1 && h == 1 {
                break;
            }
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        let mip_count = sizes.len() as u32;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jbm hiz pyramid"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mip_view = |mip: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("jbm hiz mip"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };

        let make_bind_group = |src: &wgpu::TextureView,
                               dst: &wgpu::TextureView,
                               (sw, sh, dw, dh): (u32, u32, u32, u32)|
         -> wgpu::BindGroup {
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jbm hiz sizes"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue
                .write_buffer(&uniform, 0, bytemuck::bytes_of(&[sw, sh, dw, dh]));
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jbm hiz step"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            })
        };

        let fill_bind_group =
            make_bind_group(depth_view, &mip_view(0), (width, height, width, height));
        let reduce_bind_groups = (0..mip_count - 1)
            .map(|mip| {
                let (sw, sh) = sizes[mip as usize];
                let (dw, dh) = sizes[mip as usize + 1];
                make_bind_group(&mip_view(mip), &mip_view(mip + 1), (sw, sh, dw, dh))
            })
            .collect();

        HizPyramid {
            full_view,
            fill_pipeline,
            fill_bind_group,
            reduce_pipeline,
            reduce_bind_groups,
            sizes,
            generation,
        }
    }

    /// Whether this pyramid matches a depth target of the given size and
    /// generation (a depth texture recreated at the same size still needs a
    /// new fill bind group, so the generation is part of the key).
    pub fn matches_depth(&self, width: u32, height: u32, generation: u32) -> bool {
        self.sizes.first() == Some(&(width, height)) && self.generation == generation
    }

    /// The view with every mip, for the cull shader.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.full_view
    }

    /// The highest mip level (1x1); `textureLoad` levels are clamped to it.
    pub fn max_level(&self) -> u32 {
        self.sizes.len() as u32 - 1
    }

    /// Records the rebuild: the mip-0 fill, then one dispatch per reduction
    /// step. Every dispatch is its own compute pass so wgpu inserts the
    /// write->read barriers between mips.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let (w, h) = self.sizes[0];
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jbm hiz fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fill_pipeline);
            pass.set_bind_group(0, &self.fill_bind_group, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        for (i, bind_group) in self.reduce_bind_groups.iter().enumerate() {
            let (dw, dh) = self.sizes[i + 1];
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jbm hiz reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduce_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seeded xorshift the suballocator churn already uses.
    struct XorShift64(u64);
    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Builds the full pyramid with `reduce_step` and asserts every mip
    /// texel equals the min over exactly the mip-0 texels its footprint
    /// covers. Footprints propagate alongside the mips (the odd-edge fold
    /// makes an edge texel's footprint wider than 2^k), and one level's
    /// footprints must tile mip 0 exactly — no mip-0 texel skipped (the
    /// classic Hi-Z bug) and none counted twice.
    fn check_min_over_footprint(width: u32, height: u32, seed: u64) {
        let mut rng = XorShift64(seed);
        let mut mip0 = vec![0.0f32; (width * height) as usize];
        for d in &mut mip0 {
            *d = (rng.next() % 1000) as f32 / 7.0;
        }

        // Per texel: (data, w, h, footprints), each footprint the mip-0
        // span (x0, x1, y0, y1).
        let mut w = width;
        let mut h = height;
        let mut data = mip0.clone();
        let mut foot: Vec<(u32, u32, u32, u32)> = (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                (x, x + 1, y, y + 1)
            })
            .collect();
        loop {
            for ty in 0..h {
                for tx in 0..w {
                    let (x0, x1, y0, y1) = foot[(ty * w + tx) as usize];
                    let mut expected = f32::INFINITY;
                    for sy in y0..y1 {
                        for sx in x0..x1 {
                            expected =
                                expected.min(mip0[sy as usize * width as usize + sx as usize]);
                        }
                    }
                    assert_eq!(
                        data[(ty * w + tx) as usize],
                        expected,
                        "mip texel ({tx}, {ty}) of {width}x{height}"
                    );
                }
            }
            // The footprints tile mip 0 exactly.
            let mut covered = vec![false; (width * height) as usize];
            for ty in 0..h {
                for tx in 0..w {
                    let (x0, x1, y0, y1) = foot[(ty * w + tx) as usize];
                    for sy in y0..y1 {
                        for sx in x0..x1 {
                            assert!(
                                !covered[(sy * width + sx) as usize],
                                "mip texel ({tx}, {ty}) of {width}x{height} overlaps a neighbor"
                            );
                            covered[(sy * width + sx) as usize] = true;
                        }
                    }
                }
            }
            assert!(
                covered.iter().all(|&c| c),
                "every mip-0 texel of {width}x{height} must be covered"
            );
            if w == 1 && h == 1 {
                break;
            }
            // Propagate: dst texel t covers src texels [2t ..= xe] per
            // axis (the last one takes the remainder), so its footprint
            // spans the child footprints of exactly those texels.
            let xe = |t: u32, dst: u32| if t + 1 == dst { w - 1 } else { 2 * t + 1 };
            let ye = |t: u32, dst: u32| if t + 1 == dst { h - 1 } else { 2 * t + 1 };
            let src_foot = std::mem::take(&mut foot);
            let (dst, dw, dh) = reduce_step(&data, w, h);
            for ty in 0..dh {
                for tx in 0..dw {
                    let x0 = src_foot[(ty * w + 2 * tx) as usize].0;
                    let x1 = src_foot[(ty * w + xe(tx, dw)) as usize].1;
                    let y0 = src_foot[(2 * ty * w) as usize].2;
                    let y1 = src_foot[(ye(ty, dh) * w) as usize].3;
                    foot.push((x0, x1, y0, y1));
                }
            }
            data = dst;
            w = dw;
            h = dh;
        }
    }

    #[test]
    fn pyramid_is_the_min_over_mip0_footprints() {
        check_min_over_footprint(1, 1, 1);
        check_min_over_footprint(3, 5, 2);
        check_min_over_footprint(17, 9, 3);
        check_min_over_footprint(640, 361, 4);
    }

    #[test]
    fn odd_edge_folds_the_extra_texel_in() {
        // src 5 wide, dst 2 wide: dst 1 covers {2, 3, 4} (the odd remainder
        // folds into the last texel), so a lone low value anywhere past the
        // first pair must survive into dst 1.
        let src = vec![10.0, 10.0, 10.0, 10.0, 2.0];
        let (dst, w, _) = reduce_step(&src, 5, 1);
        assert_eq!(w, 2);
        assert_eq!(dst, vec![10.0, 2.0]);
        let src = vec![10.0, 10.0, 10.0, 2.0, 10.0];
        let (dst, _, _) = reduce_step(&src, 5, 1);
        assert_eq!(dst, vec![10.0, 2.0]);
        // An even size pairs exactly, with no fold.
        let src = vec![10.0, 10.0, 10.0, 2.0];
        let (dst, w, _) = reduce_step(&src, 4, 1);
        assert_eq!(w, 2);
        assert_eq!(dst, vec![10.0, 2.0]);
        // A 1-wide source collapses to one texel covering it.
        let src = vec![2.0];
        let (dst, w, _) = reduce_step(&src, 1, 1);
        assert_eq!((w, dst.len()), (1, 1));
        assert_eq!(dst, vec![2.0]);
    }
}
