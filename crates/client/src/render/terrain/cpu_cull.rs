//! The cpu-cull draw path (and the fallback path's culling): the CPU
//! frustum and direction tests over the meta slots, with the shared
//! origin formula, so the CPU and GPU cull on the same numbers. The
//! task 2.12 debug views apply here exactly as they do in cull.wgsl:
//! hidden levels never walk, the freeze view keeps the captured frustum
//! and camera split, and the explode view skips both tests (and the
//! direction skip, whose bound the mode replaces with the static
//! per-page totals).

use std::ops::Range;

use glam::Vec3;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::lod::CubeKey;
use crate::render::renderer::DepthTarget;
use crate::render::settings::RenderSettings;
use crate::render::timing::GpuTimer;

use super::TerrainPass;
use super::pages::CubeMesh;
use super::pipelines::{CMD_BYTES, FACE_BYTES};
use super::space::{
    Frustum, aabb_distance_squared, camera_split, cube_origin, cube_size, direction_split_blocks,
    visible_directions,
};
use super::stats::{CullStats, TerrainStats};

/// The culling basis of the frame (task 2.12): the camera split and
/// frustum the tests run against — the live camera, or the frozen one
/// while the freeze-cull view is on.
pub(super) struct CullBasis {
    pub(super) camera_cube: glam::IVec3,
    pub(super) camera_frac: Vec3,
    pub(super) frustum: Frustum,
}

/// The indirect commands one visible cube contributes (task 2.13 C),
/// pure so the far/near rule can be unit-tested without a GPU. Beyond
/// `split_radius_blocks` from the camera the cube's whole face range
/// draws in one command — back-face culling drops exactly the faces
/// direction skipping dropped, so the pixels cannot change. Nearer cubes
/// keep one command per non-empty, non-skipped direction (skipped only
/// when not exploding: a slid cube can straddle the camera). Returns the
/// face count the commands issue.
// The eight parameters are one cube's draw context; a bundling struct
// would only be unpacked again here.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_cube_commands(
    out: &mut Vec<wgpu::util::DrawIndirectArgs>,
    face_start: u32,
    ranges: &[Range<u32>; 6],
    slot: u32,
    rel_min: Vec3,
    rel_max: Vec3,
    exploding: bool,
    split_radius_blocks: f32,
) -> u32 {
    let radius_squared = split_radius_blocks * split_radius_blocks;
    if aabb_distance_squared(rel_min, rel_max) > radius_squared {
        // Far: one command over the cube's whole, direction-contiguous
        // face range.
        let total: u32 = ranges.iter().map(|r| r.len() as u32).sum();
        if total == 0 {
            return 0;
        }
        out.push(wgpu::util::DrawIndirectArgs {
            vertex_count: total * 6,
            instance_count: 1,
            first_vertex: face_start * 6,
            first_instance: slot,
        });
        return total;
    }
    let dirs = visible_directions(rel_min, rel_max);
    let mut faces = 0u32;
    for (dir, dir_range) in ranges.iter().enumerate() {
        if dir_range.is_empty() || (!exploding && !dirs[dir]) {
            continue;
        }
        out.push(wgpu::util::DrawIndirectArgs {
            vertex_count: dir_range.len() as u32 * 6,
            instance_count: 1,
            first_vertex: (face_start + dir_range.start) * 6,
            first_instance: slot,
        });
        faces += dir_range.len() as u32;
    }
    faces
}

impl TerrainPass {
    /// The culling basis of the frame (task 2.12): the live camera split
    /// and frustum, or the frozen view's while `freeze_cull` is on.
    pub(super) fn cull_basis(&self, camera: &Camera) -> CullBasis {
        match &self.frozen {
            Some(frozen) => CullBasis {
                camera_cube: camera_split(frozen.position).0,
                camera_frac: camera_split(frozen.position).1,
                frustum: Frustum::from_view_proj(&frozen.view_proj),
            },
            None => CullBasis {
                camera_cube: camera_split(camera.position).0,
                camera_frac: camera_split(camera.position).1,
                frustum: Frustum::from_view_proj(&camera.view_proj()),
            },
        }
    }

    /// Whether a level draws under the settings' shown-level mask
    /// (task 2.12). Levels beyond 6 clamp to the last bit.
    fn level_shown(settings: &RenderSettings, level: u8) -> bool {
        settings.level_mask & (1 << (level.min(6))) != 0
    }

    /// The cpu-cull frame (also the fallback path's culling): the CPU
    /// frustum and direction tests over the same meta slots, with the
    /// shared origin formula, so the CPU and GPU cull on the same numbers.
    // The nine parameters are one frame's draw context; a bundling struct
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
        timer: Option<&GpuTimer>,
    ) {
        let terrain_uniforms = self.terrain_uniforms_for(camera, settings);
        gpu.queue.write_buffer(
            &self.terrain_uniforms,
            0,
            bytemuck::bytes_of(&terrain_uniforms),
        );

        // The culling basis (task 2.12): frozen while the freeze view is
        // on — the terrain uniform above still carries the real camera,
        // so the frozen set renders from wherever the camera really is.
        let basis = self.cull_basis(camera);
        let exploding = settings.explode > 0;

        // Deterministic order: HashMap iteration is randomized. The
        // explode view skips the frustum test (the moved cubes sit
        // outside every conservative bound); hidden levels never draw.
        let mut visible: Vec<(CubeKey, Vec3, &CubeMesh)> = self
            .book
            .meshes
            .iter()
            // Coverage: undrawn cubes (a finer level replaces them) and
            // fully masked ones emit nothing on any path.
            .filter(|(key, mesh)| {
                mesh.face_count > 0
                    && mesh.drawn
                    && mesh.mask != 0xFF
                    && Self::level_shown(settings, key.level)
            })
            .filter_map(|(key, mesh)| {
                let min = cube_origin(*key, basis.camera_cube, basis.camera_frac);
                let max = min + cube_size(key.level);
                if exploding || basis.frustum.intersects_aabb(min, max) {
                    Some((*key, min, mesh))
                } else {
                    None
                }
            })
            .collect();
        visible.sort_unstable_by_key(|(key, _, _)| *key);

        // The CPU-side mirror of the GPU counters: every drawn non-empty
        // mesh at a shown level is a candidate, the frustum test culls the
        // rest (there is no occlusion culling on this path; the explode
        // view culls nothing).
        let candidates = self
            .book
            .meshes
            .iter()
            .filter(|(key, m)| {
                m.face_count > 0
                    && m.drawn
                    && m.mask != 0xFF
                    && Self::level_shown(settings, key.level)
            })
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

        // One command per visible cube and face direction with a non-empty
        // range — skipped only when back-facing, which the explode view
        // forgoes (a cube can slide across the camera) — grouped by page.
        let mut per_page: Vec<Vec<wgpu::util::DrawIndirectArgs>> =
            self.pages.iter().map(|_| Vec::new()).collect();
        let mut faces = 0u64;
        let mut level_drawn = [0u32; 7];
        let mut level_faces = [0u32; 7];
        let split_radius_blocks = direction_split_blocks(settings.direction_split_radius);
        for (key, rel_min, mesh) in &visible {
            let Some((page, alloc_range)) = mesh.alloc.as_ref() else {
                continue;
            };
            let level_index = (key.level as usize).min(6);
            level_drawn[level_index] += 1;
            let issued = push_cube_commands(
                &mut per_page[*page],
                alloc_range.start,
                &mesh.ranges,
                mesh.slot.expect("a drawn cube holds a meta slot"),
                *rel_min,
                *rel_min + cube_size(key.level),
                exploding,
                split_radius_blocks,
            );
            faces += u64::from(issued);
            level_faces[level_index] += issued;
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
            level_drawn,
            level_faces,
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

        let mut pass = self.begin_pass(encoder, target, depth.view(), timer);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges_of(counts: [u32; 6]) -> [Range<u32>; 6] {
        let mut start = 0u32;
        std::array::from_fn(|d| {
            let range = start..start + counts[d];
            start += counts[d];
            range
        })
    }

    fn args_of(cmd: &wgpu::util::DrawIndirectArgs) -> (u32, u32, u32) {
        (cmd.vertex_count, cmd.first_vertex, cmd.first_instance)
    }

    /// Task 2.13 C's rule: a far cube gets ONE command covering its whole
    /// (direction-contiguous) face range; a near cube gets one per
    /// non-empty, non-skipped direction.
    #[test]
    fn far_cube_gets_one_command_near_cube_gets_per_direction() {
        let ranges = ranges_of([3, 0, 5, 2, 0, 4]);
        // Far: the whole cube sits 200 blocks out on +X (the split radius
        // is 128 blocks).
        let far_min = Vec3::new(200.0, -16.0, -16.0);
        let far_max = far_min + Vec3::splat(32.0);
        let mut commands = Vec::new();
        let faces = push_cube_commands(
            &mut commands,
            100,
            &ranges,
            7,
            far_min,
            far_max,
            false,
            128.0,
        );
        assert_eq!(commands.len(), 1, "one command for the far cube");
        assert_eq!(args_of(&commands[0]), (14 * 6, 100 * 6, 7));
        assert_eq!(faces, 14);

        // Near, in the (+, +, +) octant of the camera: only the
        // camera-facing directions — and only their non-empty ranges
        // (-X is empty here, so -Y at offset 8 and -Z at 10 remain).
        let near_min = Vec3::new(16.0, 16.0, 16.0);
        let near_max = near_min + Vec3::splat(32.0);
        let mut commands = Vec::new();
        let faces = push_cube_commands(
            &mut commands,
            100,
            &ranges,
            7,
            near_min,
            near_max,
            false,
            128.0,
        );
        assert_eq!(
            commands.iter().map(args_of).collect::<Vec<_>>(),
            vec![(2 * 6, 108 * 6, 7), (4 * 6, 110 * 6, 7)]
        );
        assert_eq!(faces, 6);

        // A cube straddling the camera is near even at radius 0 (distance
        // zero): all six non-empty directions.
        let mut commands = Vec::new();
        push_cube_commands(
            &mut commands,
            0,
            &ranges,
            0,
            Vec3::splat(-16.0),
            Vec3::splat(16.0),
            false,
            0.0,
        );
        assert_eq!(commands.len(), 4, "the non-empty directions only");

        // At radius 0 anything strictly outside the camera is far: one
        // command even for the straddling-ish cube one block out.
        let mut commands = Vec::new();
        let faces = push_cube_commands(
            &mut commands,
            50,
            &ranges,
            3,
            Vec3::new(1.0, -16.0, -16.0),
            Vec3::new(33.0, 16.0, 16.0),
            false,
            0.0,
        );
        assert_eq!(commands.len(), 1);
        assert_eq!(faces, 14);

        // Exploding keeps per-direction emission for near cubes (a slid
        // cube can straddle the camera; the direction test is off).
        let mut commands = Vec::new();
        push_cube_commands(
            &mut commands,
            100,
            &ranges,
            7,
            near_min,
            near_max,
            true,
            128.0,
        );
        assert_eq!(commands.len(), 4, "every non-empty direction");
    }
}
