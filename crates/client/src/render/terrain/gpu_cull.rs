//! The gpu-cull draw path: a compute pass tests every meta slot against
//! the frustum and the Hi-Z occlusion pyramid and writes the indirect
//! commands into per-page regions; two render phases draw them. The CPU's
//! only per-cube work is the frustum pass that sizes the regions.

use jbm_core::coords::CUBE_SIZE;

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::render::hiz::HizPyramid;
use crate::render::renderer::DepthTarget;
use crate::render::settings::RenderSettings;
use crate::render::timing::GpuTimer;

use super::TerrainPass;
use super::layout::{CullParams, cull_debug_words};
use super::pipelines::{CMD_BYTES, FACE_BYTES};
use super::space::direction_split_blocks;
use super::stats::{CullStats, STATS_BYTES, TerrainStats};

impl TerrainPass {
    /// The gpu-cull frame: the GPU tests, emits and draws; the CPU's only
    /// per-cube work is the frustum pass that sizes the command regions.
    /// Phase order: early cull/draw, then Hi-Z build and late cull in one
    /// compute pass, then the late draw.
    // The nine parameters are one frame's draw context; a bundling struct
    // would only be unpacked again here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_gpu_cull(
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
        // Complete any stats readback from earlier frames (non-blocking)
        // and arm the maps for the slots copied last frame.
        self.stats_ring.begin_frame(gpu);

        let resident_dirs = self.book.total_dirs();
        let mut stats = self.page_stats(TerrainStats {
            meshed: self.book.meshes.len(),
            drawn: 0,
            faces: 0,
            commands: 0,
            draw_calls: 0,
            pages: self.pages.len(),
            page_bytes: (self.pages.len() as u64) * u64::from(self.page_faces) * FACE_BYTES,
            used_faces: 0,
        });
        if resident_dirs == 0 {
            self.stats = stats;
            self.last_cull = CullStats::default();
            return;
        }

        // The shared origin uniform (also read by the vertex shader).
        let terrain_uniforms = self.terrain_uniforms_for(camera, settings);
        gpu.queue.write_buffer(
            &self.terrain_uniforms,
            0,
            bytemuck::bytes_of(&terrain_uniforms),
        );

        // The pyramid is recreated together with the depth target (size or
        // generation change).
        let (width, height, generation) = depth.description();
        if self
            .hiz
            .as_ref()
            .is_none_or(|h| !h.matches_depth(width, height, generation))
        {
            self.hiz = Some(HizPyramid::new(
                gpu,
                depth.view(),
                width,
                height,
                generation,
                depth.samples(),
            ));
            self.rebuild_cull_bind_group();
        }

        // Grow the command buffer when the bound outgrew it. The bound
        // follows the culling basis of the frame (task 2.12): the frozen
        // view's split and frustum while freeze-cull is on, and — while
        // exploded — the static per-page non-empty-direction totals, since
        // the shader then emits every direction of every active cube.
        let basis = self.cull_basis(camera);
        let page_bounds = if settings.explode > 0 {
            self.book.page_dir_counts.clone()
        } else {
            self.book.frustum_page_bounds(
                basis.camera_cube,
                basis.camera_frac,
                &basis.frustum,
                direction_split_blocks(settings.direction_split_radius),
            )
        };
        let mut region_base = Vec::with_capacity(page_bounds.len());
        let mut bound_total = 0u32;
        for &bound in &page_bounds {
            region_base.push(bound_total);
            bound_total += bound;
        }
        if 2 * bound_total > self.commands_capacity {
            self.commands_capacity = (2 * bound_total).next_power_of_two();
            self.commands_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jbm terrain gpu commands"),
                size: u64::from(self.commands_capacity) * CMD_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.rebuild_cull_bind_group();
        }

        // region_base: the prefix sum over pages of this frame's CPU
        // frustum bound, rewritten each frame (a tiny buffer; the CPU
        // already knows the bound).
        let max_level = self.hiz.as_ref().expect("created above").max_level();
        gpu.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&CullParams {
                camera_cube: terrain_uniforms.camera_cube,
                camera_frac: terrain_uniforms.camera_frac,
                screen: [width, height, max_level, self.book.meta_len],
                regions: [
                    bound_total,
                    self.pages.len() as u32,
                    self.commands_capacity,
                    CUBE_SIZE as u32,
                ],
                debug: cull_debug_words(settings),
                split: [
                    direction_split_blocks(settings.direction_split_radius),
                    0.0,
                    0.0,
                    0.0,
                ],
            }),
        );
        gpu.queue.write_buffer(
            &self.region_base_buffer,
            0,
            bytemuck::cast_slice(&region_base),
        );

        // Zero the counters, the stats and the command regions so a
        // region's unused tail is zero-count draws. Only the prefix this
        // frame's draws read needs clearing; slots past it are never
        // referenced. clear_buffer requires COPY_DST (see the buffer
        // usages above).
        encoder.clear_buffer(&self.region_counter_buffer, 0, None);
        encoder.clear_buffer(&self.stats_buffer, 0, None);
        if bound_total > 0 {
            encoder.clear_buffer(
                &self.commands_buffer,
                0,
                Some(u64::from(bound_total) * 2 * CMD_BYTES),
            );
        }

        // 1. Early cull: what was visible last frame and still passes the
        //    frustum.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jbm terrain early cull"),
                timestamp_writes: timer.and_then(|t| t.compute_writes("cull")),
            });
            pass.set_pipeline(&self.cull_early_pipeline);
            pass.set_bind_group(0, uniforms, &[]);
            pass.set_bind_group(1, &self.cull_bind_group, &[]);
            pass.dispatch_workgroups(self.book.meta_len.div_ceil(64), 1, 1);
        }

        // 2. Early draw.
        let mut draw_calls = 0u64;
        {
            let mut pass = self.begin_pass(encoder, target, depth.view(), timer);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, uniforms, &[]);
            pass.set_bind_group(2, &self.shared_bind_group, &[]);
            draw_calls += self.draw_regions(&mut pass, &region_base, &page_bounds, 0);
        }

        // 3+4. The Hi-Z pyramid from the depth the early draw left, then
        //    the late cull against it, in one compute pass: dispatches
        //    within a pass are ordered, and each dispatch is its own usage
        //    scope (WebGPU spec §3.4.4: "In a compute pass, each dispatch
        //    command is one usage scope"), so wgpu validates the mip k ->
        //    mip k+1 write->read transitions per dispatch and inserts the
        //    barrier between them; the pyramid dispatches ending in
        //    storage writes then the cull dispatch sampling the whole
        //    pyramid transition the same way.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jbm terrain hiz + late cull"),
                timestamp_writes: timer.and_then(|t| t.compute_writes("hiz+cull")),
            });
            self.hiz.as_ref().expect("created above").record(&mut pass);
            pass.set_pipeline(&self.cull_late_pipeline);
            pass.set_bind_group(0, uniforms, &[]);
            pass.set_bind_group(1, &self.cull_bind_group, &[]);
            pass.dispatch_workgroups(self.book.meta_len.div_ceil(64), 1, 1);
        }

        // 5. Late draw.
        {
            let mut pass = self.begin_pass(encoder, target, depth.view(), timer);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, uniforms, &[]);
            pass.set_bind_group(2, &self.shared_bind_group, &[]);
            draw_calls += self.draw_regions(&mut pass, &region_base, &page_bounds, bound_total);
        }

        // 6. The counters for the overlay, one or two frames late.
        if let Some(slot) = self.stats_ring.take_slot() {
            encoder.copy_buffer_to_buffer(
                &self.stats_buffer,
                0,
                &self.stats_ring.buffers[slot],
                0,
                STATS_BYTES,
            );
        }

        let cull = self.stats_ring.latest;
        stats.drawn = cull.drawn() as usize;
        stats.faces = u64::from(cull.faces);
        stats.commands = u64::from(cull.commands);
        stats.draw_calls = draw_calls;
        self.stats = stats;
        self.last_cull = cull;
    }

    /// One multi_draw_indirect per page over its region of `phase_base`.
    /// The count is this frame's CPU frustum bound for the page: slots the
    /// shader did not fill were cleared to zero-count draws, which are
    /// no-ops everywhere (the reason `multi_draw_indirect_count` is not
    /// used; Metal lacks it, and its MDI is a CPU loop over the encoded
    /// count, so the count must come from the CPU anyway).
    fn draw_regions(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        region_base: &[u32],
        page_bounds: &[u32],
        phase_base: u32,
    ) -> u64 {
        let mut calls = 0u64;
        for (page, &count) in page_bounds.iter().enumerate() {
            if count == 0 {
                continue;
            }
            pass.set_bind_group(1, &self.pages[page].bind_group, &[]);
            let offset = u64::from(phase_base + region_base[page]) * CMD_BYTES;
            pass.multi_draw_indirect(&self.commands_buffer, offset, count);
            calls += 1;
        }
        calls
    }
}
