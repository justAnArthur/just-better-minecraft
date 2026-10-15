//! Window, surface and the frame loop: the client runtime. Drains the
//! server link into a [`ClientWorld`], re-meshes dirty cubes on a
//! [`MeshPool`], uploads finished meshes within a per-frame budget, and
//! renders sky + terrain + debug overlay.

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec3;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
use jbm_core::coords::CUBE_SIZE;
use jbm_core::link::ClientLink;
use jbm_core::physics::TICK_DT;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
use jbm_core::raycast::{REACH, RayHit, raycast};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use crate::camera::Camera;
use crate::edit::{Hotbar, Repeat, place_target};
use crate::gpu::Gpu;
use crate::input::{InputState, Key as MoveKey, apply_mouse_look, move_input};
use crate::mesh::appearance::BlockAppearance;
use crate::mesh::pool::MeshPool;
use crate::predict::Predictor;
use crate::render::chunks::TerrainPass;
use crate::render::overlay::{FrameStats, OverlayText};
use crate::render::renderer::{DepthTarget, Renderer};
use crate::render::texture_array::BlockTextures;
use crate::world::ClientWorld;

/// Frame dt is clamped so a pause (window drag, debugger) does not
/// teleport the camera — and, with the catch-up cap below, never
/// accumulates more than one clamped frame's worth of sim ticks.
const MAX_DT: f32 = 0.25;

/// The fixed simulation period: 20 Hz, matching the physics tick.
const SIM_PERIOD: Duration = Duration::from_millis(1000 / 20);

/// Sim ticks run per frame at most; the rest of the accumulator is
/// dropped (a stalled frame must not spiral into a catch-up burst).
const MAX_SIM_TICKS_PER_FRAME: u32 = 5;

/// Dirty cubes handed to the mesh pool per frame.
const SUBMIT_BUDGET: usize = 16;

/// Finished meshes uploaded to the GPU per frame, so streaming never
/// hitches a frame. Leftover results stay queued for the next frame.
const UPLOAD_BUDGET: usize = 32;

/// Spawns the game window and runs the frame loop until the user quits
/// (or the server link dies). `link` is the client's end of a connection;
/// `main` wires it to an in-process server.
pub fn run(link: ClientLink) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::new(link);
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
    terrain: TerrainPass,
}

struct App {
    windowed: Option<Windowed>,
    camera: Camera,
    input: InputState,
    stats: FrameStats,
    cursor_grabbed: bool,
    /// Set while the OS reports the window fully hidden; the redraw loop
    /// pauses until it is visible again.
    occluded: bool,
    last_frame: Option<Instant>,
    /// Leftover simulation time: whole ticks are consumed by the fixed
    /// 20 Hz sim loop below, the fraction becomes the render alpha.
    accumulator: Duration,
    /// The predicted player. `None` until the `Welcome` arrives — the
    /// sim does not run before it.
    predictor: Option<Predictor>,
    /// Server link; drained every frame without blocking.
    link: ClientLink,
    world: ClientWorld,
    pool: MeshPool,
    /// Mesh jobs submitted but not yet received back.
    in_flight: usize,
    joined: bool,
    /// Set when the server refused, disconnected or the link closed; the
    /// loop exits cleanly after the current frame.
    fatal: Option<String>,
    /// Block editing (task 1.7): hold-to-repeat timers per button and the
    /// 9-slot hotbar. `frame_time` is the monotonic seconds counter the
    /// timers run on.
    edit_break: Repeat,
    edit_place: Repeat,
    hotbar: Hotbar,
    frame_time: f64,
    /// The registry, kept for state-name lookups in the overlay.
    registry: Arc<BlockRegistry>,
}

impl App {
    fn new(link: ClientLink) -> App {
        let mut registry = BlockRegistry::new();
        builtin::register(&mut registry);
        let appearance = Arc::new(BlockAppearance::new(&registry));
        let mut link = link;
        if let Err(err) = link.send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "player".to_owned(),
        }) {
            eprintln!("cannot reach the server: {err}");
        }
        App {
            windowed: None,
            camera: Camera::new(
                DVec3::new(0.0, 80.0, 0.0),
                70.0_f32.to_radians(),
                16.0 / 9.0,
            ),
            input: InputState::new(),
            stats: FrameStats::default(),
            cursor_grabbed: false,
            occluded: false,
            last_frame: None,
            accumulator: Duration::ZERO,
            predictor: None,
            link,
            world: ClientWorld::new(),
            pool: MeshPool::new(appearance, 0),
            in_flight: 0,
            joined: false,
            fatal: None,
            edit_break: Repeat::new(),
            edit_place: Repeat::new(),
            hotbar: Hotbar::new(),
            frame_time: 0.0,
            registry: Arc::new(registry),
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

        let (layers, _) = crate::mesh::appearance::builtin_textures();
        let textures = BlockTextures::new(&gpu, &layers)?;
        let terrain = TerrainPass::new(&gpu, format, renderer.uniform_layout(), &textures);

        window.request_redraw();
        self.windowed = Some(Windowed {
            window,
            surface,
            gpu,
            config,
            renderer,
            depth,
            terrain,
        });
        Ok(())
    }

    fn reconfigure(w: &mut Windowed) {
        w.surface.configure(&w.gpu.device, &w.config);
        w.depth
            .resize(&w.gpu.device, w.config.width, w.config.height);
        w.renderer.set_viewport(w.config.width, w.config.height);
    }

    /// Drains every queued server message (bounded work per message, no
    /// cap on count) and applies it. Sets `fatal` on terminal errors.
    fn pump_link(&mut self, w: &mut Windowed) {
        loop {
            match self.link.try_recv() {
                Ok(Some(msg)) => match &msg {
                    ServerMsg::Welcome { spawn, .. } => {
                        // The spawn is the feet position; the predictor
                        // interpolates the eye for rendering.
                        self.predictor = Some(Predictor::new(DVec3::from(*spawn)));
                        self.joined = true;
                    }
                    ServerMsg::PlayerState { ack_seq, state } => {
                        if let Some(predictor) = &mut self.predictor {
                            predictor.on_server_state(*ack_seq, *state, &self.world);
                        }
                    }
                    ServerMsg::Rejected { reason } => {
                        eprintln!("server rejected the connection: {reason}");
                        self.fatal = Some(reason.clone());
                        return;
                    }
                    ServerMsg::Disconnected { reason } => {
                        eprintln!("server disconnected: {reason}");
                        self.fatal = Some(reason.clone());
                        return;
                    }
                    ServerMsg::CubeData { .. }
                    | ServerMsg::UnloadCube { .. }
                    | ServerMsg::BlockChange { .. } => {
                        if let crate::world::WorldChange::Unloaded(pos) = self.world.apply(&msg) {
                            w.terrain.remove(pos);
                        }
                    }
                    ServerMsg::Tick { .. } | ServerMsg::Pong { .. } => {}
                },
                Ok(None) => return,
                Err(err) => {
                    eprintln!("server link closed: {err}");
                    self.fatal = Some(err.to_string());
                    return;
                }
            }
        }
    }

    /// One fixed 20 Hz simulation tick: sample the keys into a
    /// `MoveInput`, step the prediction, send it. Does nothing before the
    /// `Welcome` (there is no predictor yet).
    fn sim_tick(&mut self) {
        let App {
            predictor,
            input,
            camera,
            world,
            link,
            fatal,
            ..
        } = self;
        let Some(predictor) = predictor else {
            return;
        };
        let fly_toggle = input.take_fly_toggle();
        let mut message = predictor.tick(move_input(input, camera.yaw, fly_toggle), world);
        // Physics ignores pitch; the camera's goes along for the ride.
        if let ClientMsg::PlayerInput { pitch, .. } = &mut message {
            *pitch = camera.pitch;
        }
        if let Err(err) = link.send(message) {
            eprintln!("cannot send input, server link closed: {err}");
            *fatal = Some(err.to_string());
        }
    }

    /// The block under the crosshair: a ray from the predicted eye along
    /// the camera forward vector, up to `REACH`, against the client world
    /// (`StateInfo::solid` decides what is targetable; unloaded space
    /// stops the ray). `None` while not joined or the cursor is free.
    fn target_block(&self) -> Option<RayHit> {
        if !self.joined || !self.cursor_grabbed {
            return None;
        }
        let dir = DVec3::from(self.camera.forward());
        raycast(self.camera.position, dir, REACH, |pos| {
            self.world
                .block(pos)
                .map(|state| self.world.is_solid_state(state))
        })
    }

    /// Fires due edit buttons against `target`: the optimistic edit lands
    /// in the world immediately (dirty with priority) and the message goes
    /// to the server, which validates and reverts on rejection.
    fn pump_edits(&mut self, target: Option<RayHit>) {
        let break_due = self.edit_break.due(self.frame_time);
        if break_due {
            self.edit_break.fired(self.frame_time);
        }
        let place_due = self.edit_place.due(self.frame_time);
        if place_due {
            self.edit_place.fired(self.frame_time);
        }
        if break_due && let Some(hit) = target {
            self.world.set_block(hit.block, BlockStateId::AIR);
            self.send_edit(ClientMsg::BreakBlock { pos: hit.block });
        }
        if place_due
            && let Some(hit) = target
            && let Some(aabb) = self.predictor.as_ref().map(|p| p.state().aabb())
        {
            let state = self.hotbar.state();
            if let Some(pos) = place_target(&hit, state, &self.world, &aabb) {
                self.world.set_block(pos, state);
                self.send_edit(ClientMsg::PlaceBlock { pos, state });
            }
        }
    }

    /// Sends an edit message; a closed link ends the loop.
    fn send_edit(&mut self, msg: ClientMsg) {
        if let Err(err) = self.link.send(msg) {
            eprintln!("cannot send edit, server link closed: {err}");
            self.fatal = Some(err.to_string());
        }
    }

    /// Submits dirty cubes to the pool and receives finished meshes,
    /// uploading at most [`UPLOAD_BUDGET`] of them. Stale results (the
    /// cube changed again) are dropped.
    fn pump_meshing(&mut self, w: &mut Windowed) {
        let inputs = self.world.take_dirty(SUBMIT_BUDGET, self.camera.position);
        self.in_flight += inputs.len();
        for (pos, version, input) in inputs {
            self.pool.submit(pos, version, input);
        }

        let mut uploads = 0;
        while uploads < UPLOAD_BUDGET {
            let Some(result) = self.pool.try_recv() else {
                break;
            };
            self.in_flight -= 1;
            if self.world.version(result.pos) != Some(result.version) {
                continue; // stale: a newer version exists and is queued
            }
            w.terrain.upload(&w.gpu, result.pos, &result.output);
            uploads += 1;
        }
    }

    fn frame(&mut self, w: &mut Windowed) {
        let now = Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map(|t| now - t)
            .unwrap_or(Duration::ZERO);
        self.stats.push(dt);
        let dt = Duration::from_secs_f32(dt.as_secs_f32().min(MAX_DT));
        self.frame_time += dt.as_secs_f64();

        // Mouse look stays per frame; movement is sampled per sim tick.
        apply_mouse_look(&mut self.camera, &self.input);

        // Fixed 20 Hz sim: run every due tick (at most the catch-up cap,
        // dropping the leftover accumulator), then render between the
        // previous and current tick at the leftover fraction.
        self.accumulator += dt;
        let mut ticks = 0;
        while self.accumulator >= SIM_PERIOD && ticks < MAX_SIM_TICKS_PER_FRAME {
            self.sim_tick();
            self.accumulator -= SIM_PERIOD;
            ticks += 1;
        }
        if ticks == MAX_SIM_TICKS_PER_FRAME {
            self.accumulator = Duration::ZERO;
        }
        let alpha = self.accumulator.as_secs_f64() / TICK_DT;
        if let Some(predictor) = &self.predictor {
            self.camera.position = predictor.render_position(alpha);
        }

        self.pump_link(w);
        let target = self.target_block();
        self.pump_edits(target);
        self.pump_meshing(w);

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
            (p.x / f64::from(CUBE_SIZE)).floor() as i64,
            (p.y / f64::from(CUBE_SIZE)).floor() as i64,
            (p.z / f64::from(CUBE_SIZE)).floor() as i64
        ));
        overlay.push(format!(
            "YAW {:.1} PITCH {:.1}",
            self.camera.yaw.to_degrees(),
            self.camera.pitch.to_degrees()
        ));
        if let Some(hit) = target
            && let Some(state) = self.world.block(hit.block)
        {
            overlay.push(format!(
                "TARGET {} {} {} {}",
                hit.block.x,
                hit.block.y,
                hit.block.z,
                self.registry.info(state).name
            ));
        } else {
            overlay.push("TARGET ---".to_string());
        }
        overlay.push(format!(
            "HOTBAR {}",
            self.registry.info(self.hotbar.state()).name
        ));
        if let Some(predictor) = &self.predictor {
            let state = predictor.state();
            overlay.push(format!(
                "MODE {} GROUND {}",
                if state.flying { "FLY" } else { "WALK" },
                if state.on_ground { "YES" } else { "NO" }
            ));
            overlay.push(format!(
                "VEL {:.2} {:.2} {:.2}",
                state.velocity.x, state.velocity.y, state.velocity.z
            ));
            overlay.push(format!(
                "PENDING {} CORRECTIONS {}",
                predictor.pending_len(),
                predictor.corrections()
            ));
        } else {
            overlay.push("MODE --- (not joined)".to_string());
        }
        let terrain_stats = w.terrain.stats();
        overlay.push(format!(
            "CUBES {} MESHED {} PENDING {}",
            self.world.cube_count(),
            w.terrain.meshed_count(),
            self.world.dirty_count() + self.in_flight
        ));
        overlay.push(format!(
            "FACES {} DRAWS {}",
            w.terrain.total_faces(),
            terrain_stats.draws
        ));

        // Every skipped frame still asks for the next one: the first frames
        // on macOS can come back Occluded before the window is shown, and
        // returning without a redraw request would stop the loop for good.
        let frame = match w.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                Self::reconfigure(w);
                w.window.request_redraw();
                return;
            }
            wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => {
                if !self.occluded {
                    w.window.request_redraw();
                }
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("skipping frame: surface validation error");
                w.window.request_redraw();
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        w.renderer.render(
            &w.gpu,
            &view,
            w.depth.view(),
            &self.camera,
            Some(&overlay),
            Some(&mut w.terrain),
            target.map(|hit| hit.block),
        );
        w.gpu.queue.present(frame);
        if !self.occluded {
            w.window.request_redraw();
        }
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
                // Do not keep moving or editing when the window loses
                // focus mid-press: the release event never arrives.
                self.input.clear_keys();
                self.edit_break.set_held(false);
                self.edit_place.set_held(false);
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
                            self.edit_break.set_held(false);
                            self.edit_place.set_held(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                    key => {
                        // F toggles flying: one edge per press, consumed by
                        // the next sim tick.
                        if pressed && matches!(key, Key::Character(c) if c.to_lowercase() == "f") {
                            self.input.press_fly_toggle();
                        }
                        // Keys 1-9 select a hotbar slot (task 1.7).
                        if pressed
                            && let Key::Character(c) = key
                            && let Some(slot) = c
                                .as_str()
                                .chars()
                                .next()
                                .and_then(|ch| ch.to_digit(10).filter(|d| (1..=9).contains(d)))
                        {
                            self.hotbar.select(slot as usize - 1);
                        }
                        if let Some(mk) = map_key(key) {
                            self.input.set_key(mk, pressed);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    // While the cursor is free, the first left click grabs
                    // it and does nothing else.
                    MouseButton::Left if !self.cursor_grabbed => {
                        if pressed {
                            self.cursor_grabbed = grab_cursor(&w.window);
                        }
                    }
                    MouseButton::Left => self.edit_break.set_held(pressed),
                    MouseButton::Right => self.edit_place.set_held(pressed),
                    _ => {}
                }
            }
            WindowEvent::Occluded(occluded) => {
                // Pause the loop while fully hidden (minimized, other
                // space) instead of spinning on Occluded surfaces.
                self.occluded = occluded;
                if !occluded {
                    w.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let mut w = match self.windowed.take() {
                    Some(w) => w,
                    None => return,
                };
                self.frame(&mut w);
                self.windowed = Some(w);
                if self.fatal.is_some() {
                    event_loop.exit();
                }
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
