//! The egui debug panel (task 2.12): F3 toggles one `egui::Window`
//! ("Debug") drawn last, over sky, terrain, the crosshair and the hotbar,
//! into the same surface view. While it is open the cursor is released and
//! egui owns the mouse and keyboard ([`game_keys_enabled`] decides when a
//! game key still applies); the bitmap overlay stands down and its lines
//! move into the panel's Stats section instead. Offscreen tests never
//! construct an [`EguiPass`] — it needs a window, they do not.
//!
//! Split: this module owns the winit/wgpu plumbing ([`EguiPass`], the
//! input routing), [`panel`] owns the widgets. The panel edits
//! [`RenderSettings`](crate::render::settings::RenderSettings) and plain
//! state; it holds no rendering logic of its own.

pub mod panel;

use std::error::Error;

use egui_wgpu::Renderer as EguiWgpuRenderer;
use egui_winit::State as EguiWinitState;
use winit::event::WindowEvent;
use winit::window::Window;

use crate::gpu::Gpu;

/// Hooks the game fills in so the panel can show server numbers and steer
/// the server's view radius without the client ever depending on
/// `jbm-server` (the link carries only protocol messages). Plain closures
/// over `ServerStats` formatted in the game crate: `server_lines` returns
/// label/value rows, `set_view_radius` forwards a new
/// `(view_radius, lod_view_radius)` pair in cubes.
pub struct DebugHooks {
    pub server_lines: Box<dyn Fn() -> Vec<(String, String)> + Send>,
    pub set_view_radius: Box<dyn Fn(u32, u32) + Send>,
}

/// Whether a game key event reaches [`InputState`](crate::input::InputState):
/// while the panel is open, egui owns the keyboard whenever it wants it
/// (a text box has focus); otherwise the game keys work exactly as with
/// the panel closed. Pure, so the routing is unit-testable without a
/// window.
pub(crate) fn game_keys_enabled(panel_open: bool, egui_wants_keyboard: bool) -> bool {
    !(panel_open && egui_wants_keyboard)
}

/// The egui integration for one window: the winit state (event translation,
/// raw input), the wgpu renderer, and the shared context. One per app;
/// created when the window is, never in offscreen tests.
pub struct EguiPass {
    pub context: egui::Context,
    state: EguiWinitState,
    renderer: EguiWgpuRenderer,
}

impl EguiPass {
    /// Creates the pass for `window`'s surface format. `color_format` is
    /// the surface's format the panel renders into.
    pub fn new(
        gpu: &Gpu,
        window: &Window,
        color_format: wgpu::TextureFormat,
    ) -> Result<EguiPass, Box<dyn Error>> {
        let context = egui::Context::default();
        let state = EguiWinitState::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer = EguiWgpuRenderer::new(
            &gpu.device,
            color_format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                ..egui_wgpu::RendererOptions::default()
            },
        );
        Ok(EguiPass {
            context,
            state,
            renderer,
        })
    }

    /// Whether egui currently wants the keyboard (a widget has text
    /// focus). The app stops applying game keys while this is true and
    /// the panel is open.
    pub fn wants_keyboard_input(&self) -> bool {
        self.context.egui_wants_keyboard_input()
    }

    /// Feeds one window event to egui. Call for every event while the
    /// panel is open (keyboard, mouse, wheel, resize); returns whether
    /// egui consumed it, for callers that want the stronger per-event
    /// signal on top of [`EguiPass::wants_keyboard_input`].
    pub fn on_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.state.on_window_event(window, event);
        if response.repaint {
            window.request_redraw();
        }
        response.consumed
    }

    /// Starts one frame: takes egui's raw input from the winit state.
    /// The caller builds its widgets on [`EguiPass::context`], then calls
    /// [`EguiPass::draw`].
    pub fn begin_frame(&mut self, window: &Window) {
        let input = self.state.take_egui_input(window);
        self.context.begin_pass(input);
    }

    /// Ends the frame and draws the panel into `view`, over everything
    /// the renderer put there (its own encoder and submit; the caller
    /// presents afterwards). Own pass with `Load`, so the terrain stays.
    pub fn draw(&mut self, gpu: &Gpu, window: &Window, view: &wgpu::TextureView) {
        let full_output = self.context.end_pass();
        self.state
            .handle_platform_output(window, full_output.platform_output);
        let pixels_per_point = full_output.pixels_per_point;
        let clipped = self
            .context
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let (width, height) = {
            let size = window.inner_size();
            (size.width, size.height)
        };
        let descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };
        for (tex_id, image_deltas) in &full_output.textures_delta.set {
            for image_delta in image_deltas {
                self.renderer
                    .update_texture(&gpu.device, &gpu.queue, *tex_id, image_delta);
            }
        }
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jbm egui pass"),
            });
        // Buffer uploads first: `render` takes a `'static` pass (callbacks
        // may reference outside resources; wgpu keeps them alive itself),
        // hence forget_lifetime — after it the encoder is no longer usable.
        let user_cmd_bufs = self.renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &clipped,
            &descriptor,
        );
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("jbm egui render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.renderer
            .render(&mut pass.forget_lifetime(), &clipped, &descriptor);
        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        gpu.queue
            .submit(std::iter::once(encoder.finish()).chain(user_cmd_bufs));
    }
}

#[cfg(test)]
mod tests {
    use super::game_keys_enabled;

    /// The panel routes the keyboard: while it is open and egui wants the
    /// keyboard (a text box has focus), game keys must not apply; with the
    /// panel closed — or open over a non-text widget — they apply exactly
    /// as before.
    #[test]
    fn game_keys_follow_the_panel_and_egui_focus() {
        assert!(game_keys_enabled(false, false));
        assert!(game_keys_enabled(false, true), "closed beats egui");
        assert!(game_keys_enabled(true, false), "open but egui idle");
        assert!(!game_keys_enabled(true, true), "egui owns the keyboard");
    }

    /// egui 0.37 ships no fonts unless `default_fonts` is on; without
    /// them every glyph in the panel renders as a replacement box.
    #[test]
    fn egui_has_its_default_fonts() {
        assert!(!egui::FontDefinitions::default().font_data.is_empty());
    }
}
