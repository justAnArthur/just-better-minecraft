//! The terrain pass: per-cube face storage buffers (vertex pulling), one
//! camera-relative origins buffer rewritten each frame, and CPU frustum
//! culling of cube AABBs. See crates/client/README.md.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Mat4, Vec3, Vec4};
use jbm_core::coords::CubePos;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::mesh::greedy::MeshOutput;
use crate::render::renderer::DEPTH_FORMAT;
use crate::render::texture_array::BlockTextures;

/// One cube's uploaded mesh. Empty meshes keep an entry (so the world knows
/// the cube is meshed) but allocate nothing. The faces buffer is owned by
/// the bind group (wgpu resources are internally ref-counted).
struct CubeMesh {
    face_count: u32,
    bind_group: Option<wgpu::BindGroup>,
}

/// Camera-relative frustum for culling: five planes extracted from the
/// rotation-only view-projection (the far plane is at infinity with the
/// reverse-Z infinite-far projection). A point `p` is inside when
/// `plane . p >= 0` for every plane.
#[derive(Debug, Clone)]
pub struct Frustum {
    planes: [Vec4; 5],
}

impl Frustum {
    /// Extracts the left / right / bottom / top / near planes (Gribb &
    /// Hartmann row combinations) from a camera-relative view-projection.
    pub fn from_view_proj(view_proj: &Mat4) -> Frustum {
        let row = |r: usize| {
            Vec4::new(
                view_proj.x_axis[r],
                view_proj.y_axis[r],
                view_proj.z_axis[r],
                view_proj.w_axis[r],
            )
        };
        let row0 = row(0);
        let row1 = row(1);
        let row2 = row(2);
        let row3 = row(3);
        Frustum {
            planes: [
                row3 + row0, // left
                row3 - row0, // right
                row3 + row1, // bottom
                row3 - row1, // top
                row3 - row2, // near (reverse-Z: z' <= w')
            ],
        }
    }

    /// Whether any part of the axis-aligned box may be visible.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Support point farthest along the plane normal.
            let px = if plane.x >= 0.0 { max.x } else { min.x };
            let py = if plane.y >= 0.0 { max.y } else { min.y };
            let pz = if plane.z >= 0.0 { max.z } else { min.z };
            if plane.x * px + plane.y * py + plane.z * pz + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Per-frame terrain draw statistics, for the overlay.
#[derive(Debug, Clone, Copy, Default)]
pub struct TerrainStats {
    /// Cubes with an uploaded mesh (empty or not).
    pub meshed: usize,
    /// Cubes passing the frustum test with at least one face.
    pub drawn: usize,
    /// Faces issued this frame.
    pub faces: u64,
    /// Draw calls issued this frame.
    pub draws: u64,
}

/// Camera-relative cube origin, one per visible cube slot. `w` is padding
/// to the storage buffer's 16-byte element.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CubeOrigin {
    pos: [f32; 4],
}

/// Slots preallocated in the origins buffer. Grown (buffer recreated) only
/// when more cubes than this are simultaneously visible.
const ORIGIN_CAPACITY: u64 = 4096;

/// Draws the terrain: one pipeline, one shared bind group, one per-cube
/// bind group and one draw per visible cube.
pub struct TerrainPass {
    pipeline: wgpu::RenderPipeline,
    faces_layout: wgpu::BindGroupLayout,
    shared_layout: wgpu::BindGroupLayout,
    shared_bind_group: wgpu::BindGroup,
    /// Kept so the bind group can be rebuilt when the origins buffer
    /// grows; ref-counted clones.
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    origins_buffer: wgpu::Buffer,
    origins_capacity: u64,
    meshes: HashMap<CubePos, CubeMesh>,
    stats: TerrainStats,
}

impl TerrainPass {
    pub fn new(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        uniform_layout: &wgpu::BindGroupLayout,
        textures: &BlockTextures,
    ) -> TerrainPass {
        let device = &gpu.device;

        // Group 1, per cube: the faces storage buffer.
        let faces_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jbm terrain faces layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Group 2, shared: origins, textures, sampler.
        let shared_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jbm terrain shared layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let origins_buffer = Self::make_origins_buffer(device, ORIGIN_CAPACITY);
        let texture_view = textures.view().clone();
        let sampler = textures.sampler().clone();
        let shared_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jbm terrain shared"),
            layout: &shared_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: origins_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(textures.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(textures.sampler()),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jbm terrain"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jbm terrain layout"),
            bind_group_layouts: &[
                Some(uniform_layout),
                Some(&faces_layout),
                Some(&shared_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jbm terrain pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[], // vertex pulling: no vertex buffers
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        TerrainPass {
            pipeline,
            faces_layout,
            shared_layout,
            shared_bind_group,
            texture_view,
            sampler,
            origins_buffer,
            origins_capacity: ORIGIN_CAPACITY,
            meshes: HashMap::new(),
            stats: TerrainStats::default(),
        }
    }

    fn make_origins_buffer(device: &wgpu::Device, slots: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm terrain origins"),
            size: slots * std::mem::size_of::<CubeOrigin>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Replaces the mesh of one cube. An empty mesh allocates nothing.
    /// The app enforces the per-frame upload budget before calling this.
    pub fn upload(&mut self, gpu: &Gpu, pos: CubePos, output: &MeshOutput) {
        if output.is_empty() {
            // Keep the entry (the cube is meshed) but drop GPU memory.
            self.meshes.insert(
                pos,
                CubeMesh {
                    face_count: 0,
                    bind_group: None,
                },
            );
            return;
        }
        let device = &gpu.device;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jbm cube faces"),
            size: output.faces.len() as u64
                * std::mem::size_of::<crate::mesh::face::PackedFace>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&output.faces));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jbm cube faces"),
            layout: &self.faces_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        self.meshes.insert(
            pos,
            CubeMesh {
                face_count: output.faces.len() as u32,
                bind_group: Some(bind_group),
            },
        );
    }

    /// Drops a cube's mesh (UnloadCube). Its buffers free when the GPU is
    /// done with them.
    pub fn remove(&mut self, pos: CubePos) {
        self.meshes.remove(&pos);
    }

    pub fn stats(&self) -> TerrainStats {
        self.stats
    }

    /// Total faces across all uploaded meshes.
    pub fn total_faces(&self) -> u64 {
        self.meshes.values().map(|m| u64::from(m.face_count)).sum()
    }

    pub fn meshed_count(&self) -> usize {
        self.meshes.len()
    }

    /// Records the terrain draw: frustum-culled cubes, camera-relative
    /// origins rewritten into the shared buffer, then one draw per cube.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        camera: &Camera,
        uniforms: &wgpu::BindGroup,
    ) {
        let frustum = Frustum::from_view_proj(&camera.view_proj());
        // Deterministic order: HashMap iteration is randomized.
        let mut visible: Vec<(CubePos, &CubeMesh)> = self
            .meshes
            .iter()
            .filter(|(_, mesh)| mesh.face_count > 0)
            .filter(|(pos, _)| {
                let world_min = DVec3::new(
                    f64::from(pos.x) * 32.0,
                    f64::from(pos.y) * 32.0,
                    f64::from(pos.z) * 32.0,
                );
                let min = camera.relative(world_min);
                let max = min + Vec3::splat(32.0);
                frustum.intersects_aabb(min, max)
            })
            .map(|(pos, mesh)| (*pos, mesh))
            .collect();
        visible.sort_unstable_by_key(|(pos, _)| *pos);

        let slots = visible.len() as u64;
        if slots == 0 {
            self.stats = TerrainStats {
                meshed: self.meshes.len(),
                drawn: 0,
                faces: 0,
                draws: 0,
            };
            return;
        }
        if slots > self.origins_capacity {
            // Grow (rare): one new buffer plus bind group, still no
            // per-frame churn in steady state.
            let capacity = slots.next_power_of_two();
            self.origins_buffer = Self::make_origins_buffer(&gpu.device, capacity);
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jbm terrain shared"),
                layout: &self.shared_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.origins_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.shared_bind_group = bind_group;
            self.origins_capacity = capacity;
        }

        // Camera-relative origins, one per slot, rewritten every frame.
        let mut origins = Vec::with_capacity(visible.len());
        for (pos, _) in &visible {
            let world_min = DVec3::new(
                f64::from(pos.x) * 32.0,
                f64::from(pos.y) * 32.0,
                f64::from(pos.z) * 32.0,
            );
            let relative = camera.relative(world_min);
            origins.push(CubeOrigin {
                pos: [relative.x, relative.y, relative.z, 0.0],
            });
        }
        gpu.queue
            .write_buffer(&self.origins_buffer, 0, bytemuck::cast_slice(&origins));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("jbm terrain pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, uniforms, &[]);
        pass.set_bind_group(2, &self.shared_bind_group, &[]);
        let mut faces = 0u64;
        let mut draws = 0u64;
        for (slot, (_, mesh)) in visible.iter().enumerate() {
            let Some(bind_group) = mesh.bind_group.as_ref() else {
                continue;
            };
            pass.set_bind_group(1, bind_group, &[]);
            pass.draw(0..mesh.face_count * 6, slot as u32..slot as u32 + 1);
            faces += u64::from(mesh.face_count);
            draws += 1;
        }
        drop(pass);

        self.stats = TerrainStats {
            meshed: self.meshes.len(),
            drawn: visible.len(),
            faces,
            draws,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0)
    }

    #[test]
    fn aabb_in_front_is_visible() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        // Directly ahead of the camera (yaw 0 looks toward -Z).
        assert!(frustum.intersects_aabb(Vec3::new(-4.0, -4.0, -36.0), Vec3::new(4.0, 4.0, -4.0)));
    }

    #[test]
    fn aabb_behind_the_camera_is_culled() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        assert!(!frustum.intersects_aabb(Vec3::new(-4.0, -4.0, 4.0), Vec3::new(4.0, 4.0, 36.0)));
    }

    #[test]
    fn aabb_far_to_the_side_is_culled() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        // Well outside the 70-degree fov on the +X side, ahead in z.
        assert!(
            !frustum.intersects_aabb(Vec3::new(200.0, -8.0, -60.0), Vec3::new(232.0, 8.0, -28.0))
        );
        // ... and the mirror on -X.
        assert!(!frustum.intersects_aabb(
            Vec3::new(-232.0, -8.0, -60.0),
            Vec3::new(-200.0, 8.0, -28.0)
        ));
    }

    #[test]
    fn aabb_straddling_the_near_plane_stays_visible() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        // Crossing the near plane (and the camera itself) from well in
        // front to behind counts as visible (conservative).
        assert!(frustum.intersects_aabb(Vec3::new(-1.0, -1.0, -20.0), Vec3::new(1.0, 1.0, 20.0)));
    }
}
