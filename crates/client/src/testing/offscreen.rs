//! The offscreen test rig: renderer, terrain pass, color target and
//! depth target in one struct, so GPU tests keep only their protocol
//! state.

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::mesh::appearance::builtin_textures;
use crate::render::offscreen::OffscreenTarget;
use crate::render::renderer::{DepthTarget, FrameInputs, Renderer};
use crate::render::terrain::TerrainPass;
use crate::render::texture_array::BlockTextures;

/// A ready-to-render offscreen scene half: the frame renderer, the
/// terrain pass, an `Rgba8UnormSrgb` target and a depth target, all one
/// color format. `render_with_sky` draws sky + terrain and reads the
/// frame back; `read_depth` returns the last frame's reverse-Z depths.
pub struct OffscreenRig {
    pub renderer: Renderer,
    pub terrain: TerrainPass,
    pub depth: DepthTarget,
    target: OffscreenTarget,
}

impl OffscreenRig {
    /// Builds everything a frame needs at `width` x `height` with the
    /// builtin block textures and default render settings.
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> OffscreenRig {
        let target = OffscreenTarget::new(&gpu.device, width, height);
        let depth = DepthTarget::new(&gpu.device, width, height);
        let renderer = Renderer::new(gpu, target.format());
        let (layers, _) = builtin_textures();
        let textures = BlockTextures::new(gpu, &layers).expect("texture array");
        let terrain = TerrainPass::new(gpu, target.format(), renderer.uniform_layout(), &textures);
        OffscreenRig {
            renderer,
            terrain,
            depth,
            target,
        }
    }

    /// Renders one frame (sky + terrain, no outline or overlay) with the
    /// given sky-color override — tests pass the sentinel ground color so
    /// holes read as sentinel — and reads the color target back. The
    /// override applies to this frame only.
    pub fn render_with_sky(
        &mut self,
        gpu: &Gpu,
        camera: &Camera,
        sky: Option<([f32; 4], [f32; 4], [f32; 4])>,
    ) -> image::RgbaImage {
        self.renderer.set_sky_colors(sky);
        self.renderer.render(
            gpu,
            FrameInputs {
                target: &self.target.create_view(),
                depth: &self.depth,
                camera,
                terrain: Some(&mut self.terrain),
                outline: None,
                overlay: None,
            },
        );
        let image = self.target.read_rgba(gpu);
        self.renderer.set_sky_colors(None);
        image
    }

    /// The depth target's contents after the last render (reverse-Z:
    /// bigger is nearer, 0.0 is the far plane / sky).
    pub fn read_depth(&self, gpu: &Gpu) -> Vec<f32> {
        self.depth.read_depth(gpu)
    }
}
