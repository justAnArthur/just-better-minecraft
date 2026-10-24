//! The offscreen test rig: renderer, terrain pass, color target and
//! depth target in one struct, so GPU tests keep only their protocol
//! state.

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::render::offscreen::OffscreenTarget;
use crate::render::renderer::{DepthTarget, FrameInputs, Renderer};
use crate::render::settings::RenderSettings;
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
    /// builtin block textures and default render settings (4× MSAA).
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> OffscreenRig {
        OffscreenRig::with_settings(gpu, width, height, &RenderSettings::default())
    }

    /// As [`OffscreenRig::new`] at 1× MSAA: for `read_depth` and the
    /// hole tests, whose exact sentinel check needs single-sample pixels
    /// (at 4× a sliver hole resolves to a blend the check would miss).
    pub fn one_sample(gpu: &Gpu, width: u32, height: u32) -> OffscreenRig {
        let settings = RenderSettings {
            msaa_samples: 1,
            ..RenderSettings::default()
        };
        OffscreenRig::with_settings(gpu, width, height, &settings)
    }

    /// As [`OffscreenRig::new`] with explicit settings — the sample count
    /// (`settings.msaa_samples`) is fixed here for the rig's lifetime;
    /// `read_depth` needs a 1× rig.
    pub fn with_settings(
        gpu: &Gpu,
        width: u32,
        height: u32,
        settings: &RenderSettings,
    ) -> OffscreenRig {
        let target = OffscreenTarget::new(&gpu.device, width, height);
        let depth = DepthTarget::with_samples(&gpu.device, width, height, settings.msaa_samples);
        let renderer = Renderer::with_settings(gpu, target.format(), *settings);
        let textures = BlockTextures::builtin(gpu).expect("texture array");
        let terrain = TerrainPass::with_settings(
            gpu,
            target.format(),
            renderer.uniform_layout(),
            &textures,
            settings,
        );
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
    /// bigger is nearer, 0.0 is the far plane / sky). Needs a 1× rig
    /// ([`OffscreenRig::with_settings`] with `msaa_samples` 1).
    pub fn read_depth(&self, gpu: &Gpu) -> Vec<f32> {
        self.depth.read_depth(gpu)
    }
}
