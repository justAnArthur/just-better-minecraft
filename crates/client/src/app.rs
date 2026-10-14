//! Window, surface and the frame loop: fly camera over the sky with the
//! debug overlay on top.

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec3;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::input::{FlyController, InputState, Key as MoveKey};
use crate::render::overlay::{FrameStats, OverlayText};
use crate::render::renderer::{DepthTarget, Renderer};

/// Frame dt is clamped so a pause (window drag, debugger) does not
/// teleport the camera.
const MAX_DT: f32 = 0.25;

/// Spawns the game window and runs the frame loop until the user quits.
pub fn run() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Everything that needs the window to exist first.
struct Windowed {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    gpu: Gpu,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    depth: DepthTarget,
}

struct App {
    windowed: Option<Windowed>,
    camera: Camera,
    input: InputState,
    controller: FlyController,
    stats: FrameStats,
    cursor_grabbed: bool,
    last_frame: Option<Instant>,
}

impl App {
    fn new() -> App {
        App {
            windowed: None,
            camera: Camera::new(
                DVec3::new(0.0, 80.0, 0.0),
                70.0_f32.to_radians(),
                16.0 / 9.0,
            ),
            input: InputState::new(),
            controller: FlyController::default(),
            stats: FrameStats::default(),
            cursor_grabbed: false,
            last_frame: None,
        }
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Box<dyn Error>> {
        let window = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title("just-better-minecraft")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
            )?,
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(window.clone()),
        ));
        let surface = instance.create_surface(window.clone())?;
        let gpu = Gpu::for_surface(instance, &surface)?;

        let caps = surface.get_capabilities(&gpu.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let present_mode = if std::env::var_os("JBM_VSYNC").is_some_and(|v| v == "0") {
            wgpu::PresentMode::AutoNoVsync
        } else {
            wgpu::PresentMode::AutoVsync
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        surface.configure(&gpu.device, &config);

        let mut renderer = Renderer::new(&gpu, format);
        let depth = DepthTarget::new(&gpu.device, config.width, config.height);
        renderer.set_viewport(config.width, config.height);
        self.camera.aspect = config.width as f32 / config.height as f32;

        window.request_redraw();
        self.windowed = Some(Windowed {
            window,
            surface,
            gpu,
            config,
            renderer,
            depth,
        });
        Ok(())
    }

    fn reconfigure(w: &mut Windowed) {
        w.surface.configure(&w.gpu.device, &w.config);
        w.depth
            .resize(&w.gpu.device, w.config.width, w.config.height);
        w.renderer.set_viewport(w.config.width, w.config.height);
    }

    fn frame(&mut self, w: &mut Windowed) {
        let now = Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map(|t| now - t)
            .unwrap_or(Duration::ZERO);
        self.stats.push(dt);
        let dt = dt.as_secs_f32().min(MAX_DT);

        self.controller.update(&mut self.camera, &self.input, dt);

        let mut overlay = OverlayText::new();
        let p = self.camera.position;
        overlay.push(format!(
            "FPS {:5.0}  {:6.2} ms",
            self.stats.fps(),
            self.stats.frame_ms()
        ));
        overlay.push(format!("XYZ {:.2} {:.2} {:.2}", p.x, p.y, p.z));
        overlay.push(format!(
            "CUBE {} {} {}",
            (p.x / 32.0).floor() as i64,
            (p.y / 32.0).floor() as i64,
            (p.z / 32.0).floor() as i64
        ));
        overlay.push(format!(
            "YAW {:.1} PITCH {:.1}",
            self.camera.yaw.to_degrees(),
            self.camera.pitch.to_degrees()
        ));

        let frame = match w.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                Self::reconfigure(w);
                return;
            }
            wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => {
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("skipping frame: surface validation error");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        w.renderer
            .render(&w.gpu, &view, w.depth.view(), &self.camera, Some(&overlay));
        w.gpu.queue.present(frame);
        w.window.request_redraw();
    }
}

/// Tries to grab the cursor (Locked, falling back to Confined) and hide
/// it. Returns whether a grab mode succeeded.
fn grab_cursor(window: &Window) -> bool {
    let locked = window.set_cursor_grab(CursorGrabMode::Locked);
    let confined = match &locked {
        Ok(()) => Ok(()),
        Err(_) => window.set_cursor_grab(CursorGrabMode::Confined),
    };
    if confined.is_ok() {
        window.set_cursor_visible(false);
        true
    } else {
        false
    }
}

fn release_cursor(window: &Window) {
    let _ = window.set_cursor_grab(CursorGrabMode::None);
    window.set_cursor_visible(true);
}

/// Maps a winit logical key to a movement key: WASD to move, Space up,
/// Shift down, Ctrl sprint.
fn map_key(key: &Key) -> Option<MoveKey> {
    match key {
        Key::Character(c) => match c.to_lowercase().as_str() {
            "w" => Some(MoveKey::Forward),
            "s" => Some(MoveKey::Back),
            "a" => Some(MoveKey::Left),
            "d" => Some(MoveKey::Right),
            _ => None,
        },
        Key::Named(NamedKey::Space) => Some(MoveKey::Up),
        Key::Named(NamedKey::Shift) => Some(MoveKey::Down),
        Key::Named(NamedKey::Control) => Some(MoveKey::Sprint),
        _ => None,
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.windowed.is_some() {
            return;
        }
        if let Err(err) = self.init(event_loop) {
            eprintln!("failed to start: {err}");
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(w) = self.windowed.as_mut() else {
            return;
        };
        if w.window.id() != window_id {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                // Ignore 0x0 while minimized.
                if size.width > 0 && size.height > 0 {
                    w.config.width = size.width;
                    w.config.height = size.height;
                    self.camera.aspect = size.width as f32 / size.height as f32;
                    Self::reconfigure(w);
                }
            }
            WindowEvent::Focused(false) => {
                // Do not keep moving when the window loses focus mid-press.
                self.input.clear_keys();
                if self.cursor_grabbed {
                    release_cursor(&w.window);
                    self.cursor_grabbed = false;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match &event.logical_key {
                    Key::Named(NamedKey::Escape) if pressed => {
                        if self.cursor_grabbed {
                            release_cursor(&w.window);
                            self.cursor_grabbed = false;
                        } else {
                            event_loop.exit();
                        }
                    }
                    key => {
                        if let Some(mk) = map_key(key) {
                            self.input.set_key(mk, pressed);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed
                    && button == MouseButton::Left
                    && !self.cursor_grabbed
                {
                    self.cursor_grabbed = grab_cursor(&w.window);
                }
            }
            WindowEvent::RedrawRequested => {
                let mut w = match self.windowed.take() {
                    Some(w) => w,
                    None => return,
                };
                self.frame(&mut w);
                self.windowed = Some(w);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.cursor_grabbed
        {
            self.input
                .accumulate_mouse(glam::Vec2::new(delta.0 as f32, delta.1 as f32));
        }
    }
}
