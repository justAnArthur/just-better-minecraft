//! Window, surface and the frame loop: the client runtime. Drains the
//! server link through a [`TerrainSync`] (world, mesh pool, coverage,
//! uploads), and renders sky + terrain + debug overlay.

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec3;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
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
use crate::debug_ui::DebugHooks;
use crate::debug_ui::EguiPass;
use crate::debug_ui::game_keys_enabled;
use crate::debug_ui::panel::WorldSetup;
use crate::edit::{Hotbar, Repeat, place_target};
use crate::gpu::Gpu;
use crate::hud::{HudPlayer, HudSnapshot, lines as hud_lines};
use crate::input::{InputState, Key as MoveKey, apply_mouse_look, move_input};
use crate::mesh::appearance::BlockAppearance;
use crate::predict::Predictor;
use crate::render::overlay::{FrameStats, OverlayText};
use crate::render::renderer::{DepthTarget, FrameInputs, Renderer};
use crate::render::settings::RenderSettings;
use crate::render::terrain::TerrainPass;
use crate::render::texture_array::BlockTextures;
use crate::terrain_sync::TerrainSync;

/// Frame dt is clamped so a pause (window drag, debugger) does not
/// teleport the camera — and, with the catch-up cap below, never
/// accumulates more than one clamped frame's worth of sim ticks.
/// The animation clock's wrap, in seconds (task 2.15): a few hours, so the
/// f32 clock keeps millisecond precision; clouds jump once per wrap.
const LOOK_CLOCK_WRAP: f32 = 4.0 * 3600.0;

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
/// `main` wires it to an in-process server. Uses the default view
/// distances; [`run_with_view`] takes explicit ones.
pub fn run(link: ClientLink) -> Result<(), Box<dyn Error>> {
    run_with_view(link, ViewSetup::default())
}

/// The view distances the far look (task 2.9) keys on: ServerConfig-like
/// values the client cannot learn from the link itself. The LOD0 ring
/// radius sets where LOD0's textures fade into the flat average colors;
/// the furthest ring radius sets the distance-fog window (see
/// `Renderer::set_view_distance`). `debug` (task 2.12) is the optional
/// server bridge the debug panel's World section uses; the client stays
/// free of server types — the game fills the closures.
pub struct ViewSetup {
    /// The server's LOD0 streaming radius, in cubes (`view_radius`).
    pub view_radius: u32,
    /// The server's ring radius per LOD level, in cubes
    /// (`lod_view_radius`).
    pub lod_view_radius: u32,
    /// The server's LOD level count (`lod_levels`).
    pub lod_levels: u8,
    /// The debug panel's server hooks, when the caller has a server to
    /// talk to (the game's in-process one).
    pub debug: Option<DebugHooks>,
}

impl ViewSetup {
    /// The documented defaults, matching `ServerConfig`: `view_radius` 8,
    /// `lod_view_radius` 8, 6 LOD levels.
    pub fn default_radii() -> ViewSetup {
        ViewSetup {
            view_radius: 8,
            lod_view_radius: 8,
            lod_levels: 6,
            debug: None,
        }
    }

    /// The far-look radii in blocks, `(LOD0 disk, outermost ring)`, by
    /// [`jbm_core::lod::ring_radii`].
    pub fn ring_radii(&self) -> (f32, f32) {
        jbm_core::lod::ring_radii(self.view_radius, self.lod_view_radius, self.lod_levels)
    }
}

impl Default for ViewSetup {
    fn default() -> ViewSetup {
        ViewSetup::default_radii()
    }
}

/// [`run`] with explicit view distances (task 2.9): the game knows its
/// `ServerConfig` and passes the ring radii the fog and fade scale on.
pub fn run_with_view(link: ClientLink, view: ViewSetup) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::new(link);
    app.view = view;
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
    /// The egui debug panel's plumbing; created with the window, so
    /// offscreen tests never touch it.
    egui: EguiPass,
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
    /// The animation clock in seconds (task 2.15: cloud drift, water
    /// ripples), wrapped at `LOOK_CLOCK_WRAP` so f32 stays precise.
    look_clock: f32,
    /// Leftover simulation time: whole ticks are consumed by the fixed
    /// 20 Hz sim loop below, the fraction becomes the render alpha.
    accumulator: Duration,
    /// The predicted player. `None` until the `Welcome` arrives — the
    /// sim does not run before it.
    predictor: Option<Predictor>,
    /// Server link; drained every frame without blocking.
    link: ClientLink,
    /// The terrain pipeline: world, mesh pool, coverage, in-flight jobs
    /// (see [`crate::terrain_sync`]).
    sync: TerrainSync,
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
    /// `JBM_AUTOFLY=<blocks>` autopilot (task 2.6 gate): fly straight for
    /// that many blocks after spawn, then exit with a frame-time summary.
    autofly: Option<Autofly>,
    /// View distances for the far look (task 2.9), applied to the terrain
    /// pass at window init.
    view: ViewSetup,
    /// Every frame's duration (seconds), for the autopilot summary.
    frame_times: Vec<f64>,
    /// When the last presented frame rendered; spans skipped loops.
    last_present: Option<Instant>,
    /// Whether the egui debug panel is open (task 2.12, F3). While it is,
    /// the cursor is released, egui owns the mouse and keyboard (game
    /// keys apply only while it does not want the keyboard), and the
    /// bitmap overlay stands down for the panel's Stats section.
    panel_open: bool,
    /// The World section's editable view distances (cube units).
    panel_world: Option<WorldSetup>,
    /// The debug panel's server bridge, when the caller passed one.
    debug_hooks: Option<DebugHooks>,
    /// Server label/value rows for the panel, refreshed a few times per
    /// second (the hooks' `server_lines` clone is not free).
    server_lines: Vec<(String, String)>,
    /// When the panel's server rows were last refreshed.
    server_lines_at: Option<Instant>,
}

/// How long the autopilot waits before re-sending a fly toggle that the
/// server has not confirmed. A frozen physics step (spawn cube still
/// generating) consumes the toggle without applying it, so the first
/// attempt right after join is routinely lost; at this spacing at most
/// one toggle is ever in flight, which keeps the flip parity safe.
const AUTOFLY_RETRY_TOGGLE: Duration = Duration::from_secs(1);

/// The `JBM_AUTOFLY` autopilot state machine.
struct Autofly {
    /// Blocks to fly after reaching cruise altitude.
    distance: f64,
    /// Whether the player has been seen flying (the run is underway).
    toggled: bool,
    /// The spawn position the cruise altitude is measured from.
    origin: DVec3,
    /// Blocks flown horizontally so far.
    flown: f64,
    /// When the last fly toggle was sent.
    last_toggle: Instant,
}

impl Autofly {
    fn from_env() -> Option<Autofly> {
        let value = std::env::var_os("JBM_AUTOFLY")?;
        let distance = value
            .to_str()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or_else(|| panic!("JBM_AUTOFLY must be a number, got {value:?}"));
        Some(Autofly {
            distance,
            toggled: false,
            origin: DVec3::ZERO,
            flown: 0.0,
            last_toggle: Instant::now(),
        })
    }

    /// Whether this tick should send another fly toggle: the player has
    /// never been seen flying, and enough time has passed that the
    /// previous toggle can no longer be in flight.
    fn wants_toggle(&self, since_last_attempt: Duration) -> bool {
        !self.toggled && since_last_attempt >= AUTOFLY_RETRY_TOGGLE
    }
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
            look_clock: 0.0,
            accumulator: Duration::ZERO,
            predictor: None,
            link,
            sync: TerrainSync::new(appearance, 0),
            joined: false,
            fatal: None,
            edit_break: Repeat::new(),
            edit_place: Repeat::new(),
            hotbar: Hotbar::new(),
            frame_time: 0.0,
            registry: Arc::new(registry),
            autofly: Autofly::from_env(),
            view: ViewSetup::default(),
            frame_times: Vec::new(),
            last_present: None,
            panel_open: false,
            panel_world: None,
            debug_hooks: None,
            server_lines: Vec::new(),
            server_lines_at: None,
        }
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Box<dyn Error>> {
        let debug = self.view.debug.take();
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

        // The one environment read for rendering: fog, the level tint,
        // the gpu-cull preference and the fallback force all come from
        // here, and the view's ring radii ride along (the caller knows
        // them; the client cannot learn them from the link).
        let (lod0_ring_radius, furthest_ring_radius) = self.view.ring_radii();
        let settings = RenderSettings {
            far_radii: [lod0_ring_radius, furthest_ring_radius],
            ..RenderSettings::from_env()
        };
        // MSAA (task 2.15) is fixed here: the pipelines and the depth
        // target are built for the startup sample count.
        let mut renderer = Renderer::with_settings(&gpu, format, settings);
        let depth = DepthTarget::with_samples(
            &gpu.device,
            config.width,
            config.height,
            settings.msaa_samples,
        );
        renderer.set_viewport(config.width, config.height);
        self.camera.aspect = config.width as f32 / config.height as f32;

        let textures = BlockTextures::builtin(&gpu)?;
        let terrain = TerrainPass::with_settings(
            &gpu,
            format,
            renderer.uniform_layout(),
            &textures,
            &settings,
        );
        let egui = EguiPass::new(&gpu, &window, format)?;
        // The panel's World sliders start at the view distances the caller
        // passed.
        self.panel_world = Some(WorldSetup {
            view_radius: self.view.view_radius,
            lod_view_radius: self.view.lod_view_radius,
            lod_levels: self.view.lod_levels,
        });
        self.debug_hooks = debug;

        window.request_redraw();
        self.windowed = Some(Windowed {
            window,
            surface,
            gpu,
            config,
            renderer,
            depth,
            terrain,
            egui,
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
                    ServerMsg::Welcome {
                        spawn,
                        yaw,
                        pitch,
                        flying,
                        ..
                    } => {
                        // The spawn is the feet position; the predictor
                        // interpolates the eye for rendering. A returning
                        // player also gets their saved look and flight.
                        self.predictor =
                            Some(Predictor::new(DVec3::from(*spawn)).with_flying(*flying));
                        self.camera.yaw = *yaw;
                        self.camera.pitch = *pitch;
                        self.joined = true;
                        if let Some(autofly) = &mut self.autofly {
                            autofly.origin = DVec3::from(*spawn);
                        }
                    }
                    ServerMsg::PlayerState { ack_seq, state } => {
                        if let Some(predictor) = &mut self.predictor {
                            predictor.on_server_state(*ack_seq, *state, self.sync.world());
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
                    | ServerMsg::LodCubeData { .. }
                    | ServerMsg::UnloadLodCube { .. }
                    | ServerMsg::BlockChange { .. } => {
                        self.sync.apply(&w.gpu, &mut w.terrain, &msg);
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
    ///
    /// The `JBM_AUTOFLY` autopilot replaces the sampled keys with its own
    /// `MoveInput` — through this same path, so prediction and the server
    /// both see a real player.
    fn sim_tick(&mut self) {
        let App {
            predictor,
            input,
            camera,
            sync,
            link,
            fatal,
            autofly,
            joined,
            ..
        } = self;
        let Some(predictor) = predictor else {
            return;
        };
        let mut fly_toggle = input.take_fly_toggle();
        let mut stepped = move_input(input, camera.yaw, fly_toggle);
        fly_toggle = false; // consumed by whichever input steps below
        if *joined && let Some(autofly) = autofly.as_mut() {
            // Climb to 40 blocks above spawn, then straight ahead at
            // sprint speed (yaw 0: toward -Z).
            let state = predictor.state();
            if state.flying {
                autofly.toggled = true;
            } else if autofly.wants_toggle(autofly.last_toggle.elapsed()) {
                fly_toggle = true;
                autofly.last_toggle = Instant::now();
            }
            let rising = state.position.y < autofly.origin.y + 40.0;
            stepped = jbm_core::physics::MoveInput {
                up: rising.into(),
                forward: (!rising).into(),
                sprint: true,
                fly_toggle,
                ..jbm_core::physics::MoveInput::IDLE
            };
            let delta = state.position - autofly.origin;
            autofly.flown = delta.x.hypot(delta.z);
        }
        let mut message = predictor.tick(stepped, sync.world());
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
        let world = self.sync.world();
        raycast(self.camera.position, dir, REACH, |pos| {
            world.block(pos).map(|state| world.is_solid_state(state))
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
            self.sync
                .world_mut()
                .set_block(hit.block, BlockStateId::AIR);
            self.send_edit(ClientMsg::BreakBlock { pos: hit.block });
        }
        if place_due
            && let Some(hit) = target
            && let Some(aabb) = self.predictor.as_ref().map(|p| p.state().aabb())
        {
            let state = self.hotbar.state();
            if let Some(pos) = place_target(&hit, state, self.sync.world(), &aabb) {
                self.sync.world_mut().set_block(pos, state);
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

    /// Submits dirty cubes (LOD0 and LOD, one budget) to the pool and
    /// receives finished meshes, uploading at most [`UPLOAD_BUDGET`] of
    /// them; see [`TerrainSync::pump`].
    fn pump_meshing(&mut self, w: &mut Windowed) {
        self.sync.pump(
            &w.gpu,
            &mut w.terrain,
            &self.camera,
            SUBMIT_BUDGET,
            UPLOAD_BUDGET,
        );
    }

    fn frame(&mut self, w: &mut Windowed) {
        let now = Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map(|t| now - t)
            .unwrap_or(Duration::ZERO);
        // Present-to-present time for the autopilot summary: `dt` above
        // spans only since the previous loop iteration, which under an
        // occluded window is a skipped (unpresented) frame.
        let present_dt = self
            .last_present
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

        // One snapshot of everything the overlay prints; `hud::lines`
        // formats it (the egui debug panel will read the same snapshot).
        let snapshot = HudSnapshot {
            fps: self.stats.fps(),
            frame_ms: self.stats.frame_ms(),
            position: self.camera.position,
            yaw: self.camera.yaw,
            pitch: self.camera.pitch,
            target: target.and_then(|hit| {
                self.sync
                    .world()
                    .block(hit.block)
                    .map(|state| (hit.block, self.registry.info(state).name.clone()))
            }),
            hotbar: self.registry.info(self.hotbar.state()).name.clone(),
            player: self.predictor.as_ref().map(|predictor| {
                let state = predictor.state();
                HudPlayer {
                    flying: state.flying,
                    on_ground: state.on_ground,
                    velocity: state.velocity,
                    pending: predictor.pending_len(),
                    corrections: predictor.corrections(),
                }
            }),
            cubes: self.sync.world().cube_count(),
            meshed: w.terrain.meshed_count(),
            pending: self.sync.world().dirty_count() + self.sync.in_flight(),
            terrain: w.terrain.stats(),
            path_name: w.terrain.path_name(),
            cull: w.terrain.cull_stats(),
            levels: w.terrain.level_stats(),
            // The queue length is the LOD dirty set plus the level >= 1
            // share of the in-flight mesh jobs (tracked by `TerrainSync`).
            lod_queue: self.sync.world().lod_dirty_count() + self.sync.in_flight_lod(),
            lod_tint: w.renderer.lod_tint(),
        };
        let mut overlay = OverlayText::new();
        if self.panel_open {
            // The panel's Stats section prints these lines instead of the
            // bitmap overlay (task 2.12): the two can never disagree.
            let refresh = self
                .server_lines_at
                .is_none_or(|at| at.elapsed() >= Duration::from_millis(500));
            if refresh && let Some(hooks) = &self.debug_hooks {
                self.server_lines = (hooks.server_lines)();
                self.server_lines_at = Some(Instant::now());
            }
            w.egui.begin_frame(&w.window);
            let world = self
                .panel_world
                .as_mut()
                .expect("panel world set at window init");
            crate::debug_ui::panel::show(
                &w.egui.context,
                &snapshot,
                &self.stats,
                w.renderer.settings_mut(),
                world,
                self.debug_hooks.as_ref(),
                &self.server_lines,
            );
        } else {
            for line in hud_lines(&snapshot) {
                overlay.push(line);
            }
        }

        // The autopilot finishes once its distance is flown: print the
        // frame-time summary and end the loop after this frame.
        if let Some(autofly) = &self.autofly
            && self.joined
            && autofly.flown >= autofly.distance
            && autofly.toggled
        {
            self.autofly = None;
            print_frame_time_summary(&self.frame_times);
            self.fatal = Some("autofly complete".to_owned());
        }

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
        // The task 2.12 debug lines (chunk borders, the frozen frustum),
        // rebuilt each frame from the settings and the resident set.
        let debug_lines = self.build_debug_lines(&w.terrain, &self.camera, w.renderer.settings());
        w.renderer.set_debug_lines(&w.gpu, &debug_lines);
        self.look_clock = (self.look_clock + dt.as_secs_f32()) % LOOK_CLOCK_WRAP;
        w.renderer.set_time(self.look_clock);
        w.renderer.render(
            &w.gpu,
            FrameInputs {
                target: &view,
                depth: &w.depth,
                camera: &self.camera,
                terrain: Some(&mut w.terrain),
                outline: target.map(|hit| hit.block),
                overlay: (!self.panel_open).then_some(&overlay),
            },
        );
        // The egui panel is drawn last, over sky, terrain, the crosshair
        // and the hotbar, into the same surface view (its own encoder and
        // submit, still before the present).
        if self.panel_open {
            w.egui.draw(&w.gpu, &w.window, &view);
        }
        w.gpu.queue.present(frame);
        // Only presented frames count for the autopilot summary —
        // Occluded-surface skips return before this point.
        self.frame_times.push(present_dt.as_secs_f64());
        if !self.occluded {
            w.window.request_redraw();
        }
    }

    /// The frame's debug lines (task 2.12), camera-relative like the
    /// terrain: the chunk-border boxes while that view is on, and the
    /// frozen cull frustum while freeze-cull is on. Empty (and the line
    /// pass stands down) when neither view is active.
    fn build_debug_lines(
        &self,
        terrain: &TerrainPass,
        camera: &Camera,
        settings: &crate::render::settings::RenderSettings,
    ) -> Vec<crate::render::lines::LineVertex> {
        let mut vertices = Vec::new();
        if settings.chunk_borders {
            self.push_chunk_borders(&mut vertices, terrain, camera);
        }
        if settings.freeze_cull {
            self.push_frozen_frustum(&mut vertices, terrain, camera);
        }
        vertices
    }

    /// Chunk-border boxes: resident drawn cubes within a small radius per
    /// level, each level's boxes at that level's cube size and tinted
    /// with the level's palette color, so LOD0 shows white 16-block boxes
    /// and coarse levels their own larger boxes in their own color.
    fn push_chunk_borders(
        &self,
        vertices: &mut Vec<crate::render::lines::LineVertex>,
        terrain: &TerrainPass,
        camera: &Camera,
    ) {
        /// How many cubes of each level around the camera draw borders.
        const BORDER_RADIUS: i64 = 3;
        let (camera_cube, camera_frac) = crate::render::terrain::camera_split(camera.position);
        for key in terrain.resident_keys() {
            // The cube's own level units: its position is already in them.
            let stride = (jbm_core::coords::CUBE_SIZE as i64) << key.level;
            let center = [
                (camera.position.x / stride as f64).floor() as i64,
                (camera.position.y / stride as f64).floor() as i64,
                (camera.position.z / stride as f64).floor() as i64,
            ];
            let distance = [
                i64::from(key.pos.x) - center[0],
                i64::from(key.pos.y) - center[1],
                i64::from(key.pos.z) - center[2],
            ]
            .iter()
            .map(|&d| d.abs())
            .max()
            .unwrap_or(i64::MAX);
            if distance > BORDER_RADIUS {
                continue;
            }
            let min = crate::render::terrain::cube_origin(key, camera_cube, camera_frac);
            let size = crate::render::terrain::space::cube_size(key.level);
            let max = min + size;
            let [r, g, b] = crate::render::terrain::lod_tint_color(key.level);
            let color = [r, g, b, 0.8];
            push_box_edges(vertices, min, max, color);
        }
    }

    /// The frozen cull frustum as its corner box: near plane to a far
    /// cutoff (the projection's far plane is at infinity, so the lines
    /// stop at a fixed distance), re-based from the frozen camera's space
    /// into the real camera's, where the renderer draws.
    fn push_frozen_frustum(
        &self,
        vertices: &mut Vec<crate::render::lines::LineVertex>,
        terrain: &TerrainPass,
        camera: &Camera,
    ) {
        /// The far cutoff of the drawn frustum, in blocks (the real far
        /// plane is at infinity).
        const FAR_CUTOFF_DEPTH: f32 = 1.0 / 2048.0;
        let Some((frozen_pos, view_proj)) = terrain.frozen_view() else {
            return;
        };
        let inverse = view_proj.inverse();
        // NDC depth 1 = the near plane; reverse-Z depth d sits at
        // NEAR / d blocks (the camera constant rides in the projection).
        let rebased = |ndc: glam::Vec4| -> glam::Vec3 {
            let unprojected = inverse * ndc;
            let rel_frozen = unprojected.truncate() / unprojected.w;
            let shift = (frozen_pos - camera.position).as_vec3();
            rel_frozen + shift
        };
        let near: [glam::Vec3; 4] = [
            rebased(glam::Vec4::new(-1.0, -1.0, 1.0, 1.0)),
            rebased(glam::Vec4::new(1.0, -1.0, 1.0, 1.0)),
            rebased(glam::Vec4::new(1.0, 1.0, 1.0, 1.0)),
            rebased(glam::Vec4::new(-1.0, 1.0, 1.0, 1.0)),
        ];
        let far: [glam::Vec3; 4] = [
            rebased(glam::Vec4::new(-1.0, -1.0, FAR_CUTOFF_DEPTH, 1.0)),
            rebased(glam::Vec4::new(1.0, -1.0, FAR_CUTOFF_DEPTH, 1.0)),
            rebased(glam::Vec4::new(1.0, 1.0, FAR_CUTOFF_DEPTH, 1.0)),
            rebased(glam::Vec4::new(-1.0, 1.0, FAR_CUTOFF_DEPTH, 1.0)),
        ];
        let color = [1.0, 0.25, 0.2, 0.9];
        for ring in [near, far] {
            for i in 0..4 {
                push_segment(vertices, ring[i], ring[(i + 1) % 4], color);
            }
        }
        for i in 0..4 {
            push_segment(vertices, near[i], far[i], color);
        }
    }
}

/// One box's twelve edges, axis-aligned `[min, max]`.
fn push_box_edges(
    vertices: &mut Vec<crate::render::lines::LineVertex>,
    min: glam::Vec3,
    max: glam::Vec3,
    color: [f32; 4],
) {
    let (a, b) = (min, max);
    let corners = [
        glam::Vec3::new(a.x, a.y, a.z),
        glam::Vec3::new(b.x, a.y, a.z),
        glam::Vec3::new(b.x, b.y, a.z),
        glam::Vec3::new(a.x, b.y, a.z),
        glam::Vec3::new(a.x, a.y, b.z),
        glam::Vec3::new(b.x, a.y, b.z),
        glam::Vec3::new(b.x, b.y, b.z),
        glam::Vec3::new(a.x, b.y, b.z),
    ];
    // Bottom rect, top rect, then the four pillars.
    for i in 0..4 {
        push_segment(vertices, corners[i], corners[(i + 1) % 4], color);
        push_segment(vertices, corners[i + 4], corners[(i + 1) % 4 + 4], color);
        push_segment(vertices, corners[i], corners[i + 4], color);
    }
}

/// One line segment, two vertices.
fn push_segment(
    vertices: &mut Vec<crate::render::lines::LineVertex>,
    from: glam::Vec3,
    to: glam::Vec3,
    color: [f32; 4],
) {
    let vertex = |p: glam::Vec3| crate::render::lines::LineVertex {
        pos: [p.x, p.y, p.z, 0.0],
        color,
    };
    vertices.push(vertex(from));
    vertices.push(vertex(to));
}

/// Prints the autopilot's frame-time summary: count, mean, p50, p95, p99,
/// max, and the frames over 33 ms and 50 ms.
fn print_frame_time_summary(times: &[f64]) {
    println!("{}", frame_time_summary(times));
}

/// The autopilot's frame-time summary line: count, mean, p50, p95, p99,
/// max in milliseconds, and the frames over 33 ms and 50 ms. `times` are
/// in seconds.
fn frame_time_summary(times: &[f64]) -> String {
    if times.is_empty() {
        return "autofly: no frames recorded".to_owned();
    }
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let ms = |seconds: f64| format!("{:.2} ms", seconds * 1000.0);
    let percentile = |p: f64| {
        let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    };
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let over = |limit: f64| times.iter().filter(|&&t| t > limit).count();
    format!(
        "autofly: {} frames, mean {}, p50 {}, p95 {}, p99 {}, max {}, over 33 ms {}, over 50 ms {}",
        times.len(),
        ms(mean),
        ms(percentile(0.50)),
        ms(percentile(0.95)),
        ms(percentile(0.99)),
        ms(sorted[sorted.len() - 1]),
        over(0.033),
        over(0.050),
    )
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
        // While the debug panel is open, egui sees the input events first
        // (task 2.12); the game's own arms below then stand down for the
        // events egui owns. F3 and Escape are pre-empted before egui so
        // the panel can always be toggled and closed.
        if self.panel_open {
            match &event {
                WindowEvent::KeyboardInput { .. }
                | WindowEvent::CursorMoved { .. }
                | WindowEvent::MouseInput { .. }
                | WindowEvent::MouseWheel { .. }
                | WindowEvent::Resized { .. }
                | WindowEvent::Focused(..) => {
                    w.egui.on_event(&w.window, &event);
                }
                _ => {}
            }
        }
        match event {
            WindowEvent::CloseRequested => {
                // Closed mid-flight: still print what the autopilot has,
                // marked partial, so an early close is not silent.
                if let Some(autofly) = self.autofly.take() {
                    println!(
                        "autofly: stopped early at {:.0} of {:.0} blocks",
                        autofly.flown, autofly.distance
                    );
                    print_frame_time_summary(&self.frame_times);
                }
                event_loop.exit();
            }
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
                        if self.panel_open {
                            // The panel is open and the cursor already
                            // free: Escape closes the panel, never the
                            // game.
                            self.panel_open = false;
                        } else if self.cursor_grabbed {
                            release_cursor(&w.window);
                            self.cursor_grabbed = false;
                            self.edit_break.set_held(false);
                            self.edit_place.set_held(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                    // F3 toggles the debug panel (task 2.12). Opening it
                    // releases the cursor and stands the game input down;
                    // closing it leaves the cursor free for the usual
                    // click-to-grab.
                    key if pressed && matches!(key, Key::Named(NamedKey::F3)) => {
                        self.panel_open = !self.panel_open;
                        if self.panel_open {
                            if self.cursor_grabbed {
                                release_cursor(&w.window);
                                self.cursor_grabbed = false;
                            }
                            self.input.clear_keys();
                            self.edit_break.set_held(false);
                            self.edit_place.set_held(false);
                        }
                    }
                    key => {
                        // Game keys apply unless the open panel's egui
                        // wants the keyboard (a text box has focus).
                        // Releases always apply, so a key held when egui
                        // took the keyboard does not stay stuck down.
                        if !game_keys_enabled(self.panel_open, w.egui.wants_keyboard_input()) {
                            if !pressed && let Some(mk) = map_key(key) {
                                self.input.set_key(mk, false);
                            }
                            return;
                        }
                        // F toggles flying: one edge per press, consumed by
                        // the next sim tick.
                        if pressed && matches!(key, Key::Character(c) if c.to_lowercase() == "f") {
                            self.input.press_fly_toggle();
                        }
                        // F4 toggles the level tint debug view (task 2.10):
                        // every fragment multiplied by a fixed color per LOD
                        // level, on every draw path.
                        if pressed && matches!(key, Key::Named(NamedKey::F4)) {
                            w.renderer.set_lod_tint(!w.renderer.lod_tint());
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
                // The open panel owns the mouse: no grabbing, no breaking
                // or placing (egui already saw the event above).
                if self.panel_open {
                    return;
                }
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
                // space) instead of spinning on Occluded surfaces —
                // except under the autopilot, which must finish
                // unattended (a gate run's window may never be shown).
                self.occluded = occluded && self.autofly.is_none();
                if !self.occluded {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The autopilot re-sends the fly toggle until flying is observed:
    /// a toggle consumed by a frozen physics step (spawn cube still
    /// generating) is otherwise lost forever and the run never leaves
    /// the ground. It must retry only after the retry window, and never
    /// once flying has been seen.
    #[test]
    fn autofly_retries_the_fly_toggle_until_flying() {
        let mut autofly = Autofly {
            distance: 100.0,
            toggled: false,
            origin: DVec3::ZERO,
            flown: 0.0,
            last_toggle: Instant::now(),
        };
        // Fresh attempt: too soon to retry.
        assert!(!autofly.wants_toggle(Duration::ZERO));
        // The attempt is old: retry (the toggle may have been lost).
        assert!(autofly.wants_toggle(AUTOFLY_RETRY_TOGGLE));
        assert!(autofly.wants_toggle(AUTOFLY_RETRY_TOGGLE * 10));
        // Flying observed: never again.
        autofly.toggled = true;
        assert!(!autofly.wants_toggle(AUTOFLY_RETRY_TOGGLE * 10));
    }

    /// Frame times are recorded in seconds but reported in milliseconds:
    /// 0.010, 0.012 and 0.040 s must print 10/12/40 ms, not 0.01 ms.
    /// The percentile index is `(len - 1) * p` rounded, so p50 picks the
    /// middle sample and p95/p99 round up to the 40 ms one; the over-33
    /// and over-50 counts compare seconds.
    #[test]
    fn frame_time_summary_reports_milliseconds() {
        assert_eq!(
            frame_time_summary(&[0.010, 0.012, 0.040]),
            "autofly: 3 frames, mean 20.67 ms, p50 12.00 ms, p95 40.00 ms, \
             p99 40.00 ms, max 40.00 ms, over 33 ms 1, over 50 ms 0"
        );
        assert_eq!(frame_time_summary(&[]), "autofly: no frames recorded");
    }
}
