//! The Debug window's widgets (task 2.12). The panel edits
//! [`RenderSettings`] and plain state — it holds no rendering logic; the
//! renderer and the terrain pass read what it wrote. Stats are read-only,
//! refreshed every frame from the same [`HudSnapshot`] the bitmap overlay
//! prints.

use crate::hud::HudSnapshot;
use crate::render::overlay::FrameStats;
use crate::render::settings::RenderSettings;

/// The World section's editable half: the server view distances in cubes
/// (`view_radius`, `lod_view_radius`), applied live through the game's
/// [`DebugHooks`](super::DebugHooks) when they change.
pub(crate) struct WorldSetup {
    pub view_radius: u32,
    pub lod_view_radius: u32,
}

/// Draws the Debug window. `hooks` is `None` when the game passed no
/// server bridge (tests, plain `app::run`): the World section then only
/// reports the client-side radii.
pub(crate) fn show(
    ctx: &egui::Context,
    snapshot: &HudSnapshot,
    stats: &FrameStats,
    settings: &mut RenderSettings,
    world: &mut WorldSetup,
    hooks: Option<&super::DebugHooks>,
    server_lines: &[(String, String)],
) {
    egui::Window::new("Debug")
        .default_open(true)
        .show(ctx, |ui| {
            stats_section(ui, snapshot, stats);
            views_section(ui, settings);
            world_section(ui, world, settings, hooks, server_lines);
        });
}

/// Stats (read-only): frame timing, the same lines the bitmap overlay
/// prints, and the frame-time spread over the last 120 frames.
fn stats_section(ui: &mut egui::Ui, snapshot: &HudSnapshot, stats: &FrameStats) {
    egui::CollapsingHeader::new("Stats")
        .default_open(true)
        .show(ui, |ui| {
            let (avg, p99, worst) = stats.frame_ms_spread();
            ui.label(format!(
                "FPS {:5.0}  frame {:6.2} ms  (120f avg {:.2}  p99 {:.2}  worst {:.2} ms)",
                snapshot.fps, snapshot.frame_ms, avg, p99, worst
            ));
            ui.separator();
            for line in crate::hud::lines(snapshot) {
                ui.monospace(line);
            }
        });
}

/// Views: the debug toggles. Each checkbox edits `RenderSettings` — the
/// renderer applies it next frame.
fn views_section(ui: &mut egui::Ui, settings: &mut RenderSettings) {
    egui::CollapsingHeader::new("Views").show(ui, |ui| {
        ui.checkbox(&mut settings.fog, "distance fog");
        ui.checkbox(&mut settings.lod_tint, "level tint (F4)");
    });
}

/// World: view radius sliders (applied live to the server when hooks
/// exist) and the read-only server configuration line. Changing a slider
/// also moves the renderer's far-look radii so the fade and fog windows
/// follow the new view distance.
fn world_section(
    ui: &mut egui::Ui,
    world: &mut WorldSetup,
    settings: &mut RenderSettings,
    hooks: Option<&super::DebugHooks>,
    server_lines: &[(String, String)],
) {
    const RING_LEVELS: f32 = 64.0; // the shipped 6 LOD levels
    egui::CollapsingHeader::new("World").show(ui, |ui| {
        let Some(hooks) = hooks else {
            ui.weak("no server bridge (plain client run)");
            return;
        };
        let mut applied = false;
        applied |= drag_u32(
            ui,
            "view radius (LOD0 cubes)",
            &mut world.view_radius,
            2..=32,
        );
        applied |= drag_u32(
            ui,
            "ring radius (cubes per level)",
            &mut world.lod_view_radius,
            2..=16,
        );
        if applied {
            (hooks.set_view_radius)(world.view_radius, world.lod_view_radius);
            settings.far_radii = [
                world.view_radius as f32 * jbm_core::coords::CUBE_SIZE as f32,
                world.lod_view_radius as f32 * jbm_core::coords::CUBE_SIZE as f32 * RING_LEVELS,
            ];
        }
        for (label, value) in server_lines {
            ui.monospace(format!("{label}: {value}"));
        }
    });
}

/// One integer drag with its label; reports whether the value changed
/// this frame.
fn drag_u32(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut u32,
    range: std::ops::RangeInclusive<u32>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui.add(egui::DragValue::new(value).range(range)).changed();
    });
    changed
}
