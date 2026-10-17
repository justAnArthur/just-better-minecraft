//! The cpu-cull draw path (and the fallback path's culling): the CPU
//! frustum and direction tests over the meta slots, with the shared
//! origin formula, so the CPU and GPU cull on the same numbers.

use glam::Vec3;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::lod::CubeKey;
use crate::render::renderer::DepthTarget;
use crate::render::settings::RenderSettings;

use super::TerrainPass;
use super::pages::CubeMesh;
use super::pipelines::{CMD_BYTES, FACE_BYTES};
use super::space::{Frustum, camera_split, cube_origin, cube_size, visible_directions};
use super::stats::{CullStats, TerrainStats};

impl TerrainPass {
    /// The cpu-cull frame (also the fallback path's culling): the CPU
    /// frustum and direction tests over the same meta slots, with the
    /// shared origin formula, so the CPU and GPU cull on the same numbers.
    // The eight parameters are one frame's draw context; a bundling struct
    // would only be unpacked again here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_cpu_cull(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &DepthTarget,
        camera: &Camera,
        uniforms: &wgpu::BindGroup,
        settings: &RenderSettings,
    ) {
        let (camera_cube, camera_frac) = camera_split(camera.position);
        let terrain_uniforms = self.terrain_uniforms_for(camera, settings);
        gpu.queue.write_buffer(
            &self.terrain_uniforms,
            0,
            bytemuck::bytes_of(&terrain_uniforms),
        );

        let frustum = Frustum::from_view_proj(&camera.view_proj());
        // Deterministic order: HashMap iteration is randomized.
        let mut visible: Vec<(CubeKey, Vec3, &CubeMesh)> = self
            .book
            .meshes
            .iter()
            // Coverage: undrawn cubes (a finer level replaces them) and
            // fully masked ones emit nothing on any path.
            .filter(|(_, mesh)| mesh.face_count > 0 && mesh.drawn && mesh.mask != 0xFF)
            .filter_map(|(key, mesh)| {
                let min = cube_origin(*key, camera_cube, camera_frac);
                let max = min + cube_size(key.level);
                frustum
                    .intersects_aabb(min, max)
                    .then_some((*key, min, mesh))
            })
            .collect();
        visible.sort_unstable_by_key(|(key, _, _)| *key);

        // The CPU-side mirror of the GPU counters: every drawn non-empty
        // mesh is a candidate, the frustum test culls the rest (there is
        // no occlusion culling on this path).
        let candidates = self
            .book
            .meshes
            .values()
            .filter(|m| m.face_count > 0 && m.drawn && m.mask != 0xFF)
            .count() as u32;

        let mut stats = self.page_stats(TerrainStats {
            meshed: self.book.meshes.len(),
            drawn: visible.len(),
            faces: 0,
            commands: 0,
            draw_calls: 0,
            pages: self.pages.len(),
            page_bytes: (self.pages.len() as u64) * u64::from(self.page_faces) * FACE_BYTES,
            used_faces: 0,
        });

        // One command per visible cube and face direction with a non-empty,
        // non-skipped range, grouped by page.
        let mut per_page: Vec<Vec<wgpu::util::DrawIndirectArgs>> =
            self.pages.iter().map(|_| Vec::new()).collect();
        let mut faces = 0u64;
        for (key, rel_min, mesh) in &visible {
            let Some((page, alloc_range)) = mesh.alloc.as_ref() else {
                continue;
            };
            let dirs = visible_directions(*rel_min, *rel_min + cube_size(key.level));
            for (dir, dir_range) in mesh.ranges.iter().enumerate() {
                if dir_range.is_empty() || !dirs[dir] {
                    continue;
                }
                per_page[*page].push(wgpu::util::DrawIndirectArgs {
                    vertex_count: dir_range.len() as u32 * 6,
                    instance_count: 1,
                    first_vertex: (alloc_range.start + dir_range.start) * 6,
                    first_instance: mesh.slot.expect("a drawn cube holds a meta slot"),
                });
                faces += dir_range.len() as u64;
            }
        }
        let mut commands: Vec<wgpu::util::DrawIndirectArgs> = Vec::new();
        // (page, command count) for every page that draws this frame.
        let mut page_spans: Vec<(usize, u32)> = Vec::new();
        for (page, cmds) in per_page.iter().enumerate() {
            if cmds.is_empty() {
                continue;
            }
            page_spans.push((page, cmds.len() as u32));
            commands.extend_from_slice(cmds);
        }
        stats.faces = faces;
        stats.commands = commands.len() as u64;

        self.last_cull = CullStats {
            candidates,
            frustum_culled: candidates - visible.len() as u32,
            occluded: 0,
            drawn_early: visible.len() as u32,
            drawn_late: 0,
            commands: commands.len() as u32,
            faces: faces as u32,
            region_overflow: 0,
        };

        if commands.is_empty() {
            self.stats = stats;
            return;
        }
        if commands.len() as u64 > self.indirect_capacity {
            // Grow (rare), like the pages.
            let capacity = (commands.len() as u64).next_power_of_two();
            self.indirect_buffer = TerrainPass::make_indirect_buffer(&gpu.device, capacity);
            self.indirect_capacity = capacity;
        }
        gpu.queue
            .write_buffer(&self.indirect_buffer, 0, bytemuck::cast_slice(&commands));

        let mut pass = self.begin_pass(encoder, target, depth.view());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, uniforms, &[]);
        pass.set_bind_group(2, &self.shared_bind_group, &[]);
        let mut draw_calls = 0u64;
        if self.uses_mdi() {
            let mut offset = 0u64;
            for &(page, count) in &page_spans {
                pass.set_bind_group(1, &self.pages[page].bind_group, &[]);
                pass.multi_draw_indirect(&self.indirect_buffer, offset * CMD_BYTES, count);
                offset += u64::from(count);
                draw_calls += 1;
            }
        } else {
            // Fallback: the same commands and buffers as plain direct draws.
            // Direct draws take a first instance without the feature.
            let mut cursor = 0usize;
            for &(page, count) in &page_spans {
                pass.set_bind_group(1, &self.pages[page].bind_group, &[]);
                for command in &commands[cursor..cursor + count as usize] {
                    let first_vertex = command.first_vertex;
                    let first_instance = command.first_instance;
                    pass.draw(
                        first_vertex..first_vertex + command.vertex_count,
                        first_instance..first_instance + command.instance_count,
                    );
                }
                cursor += count as usize;
                draw_calls += u64::from(count);
            }
        }
        drop(pass);

        stats.draw_calls = draw_calls;
        self.stats = stats;
    }
}
