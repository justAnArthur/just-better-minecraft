//! The Debug window's widgets (task 2.12). The panel edits
//! [`RenderSettings`] and plain state — it holds no rendering logic; the
//! renderer and the terrain pass read what it wrote. Stats are read-only,
//! refreshed every frame from the same [`HudSnapshot`] the bitmap overlay
//! prints.

use crate::hud::HudSnapshot;
use crate::render::overlay::FrameStats;
use crate::render::settings::{LOD_LEVELS, RenderSettings};
use crate::render::terrain::LevelStats;

/// The World section's editable half: the server view distances in cubes
/// (`view_radius`, `lod_view_radius`), applied live through the game's
/// [`DebugHooks`](super::DebugHooks) when they change.
pub(crate) struct WorldSetup {
    pub view_radius: u32,
    pub lod_view_radius: u32,
    /// Fixed for the run: the ring radius scales by `2^lod_levels`.
    pub lod_levels: u8,
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
/// prints, the frame-time spread over the last 120 frames, and the
/// per-LOD-level table (resident / drawn this frame / faces issued).
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
            ui.separator();
            // Terrain totals: triangles are two per issued face, and the
            // page accounting shows the GPU memory the meshes hold.
            ui.monospace(format!(
                "TRIS {} PAGES {} ({:.1} MiB) USED_FACES {}",
                snapshot.terrain.faces * 2,
                snapshot.terrain.pages,
                snapshot.terrain.page_bytes as f64 / (1024.0 * 1024.0),
                snapshot.terrain.used_faces,
            ));
            ui.monospace(format!(
                "LEVEL  RESIDENT  DRAWN  FACES   (drawn counts {})",
                if snapshot.path_name == "gpu-cull" {
                    "GPU readback, 1-2 frames late"
                } else {
                    "exact"
                }
            ));
            egui::Grid::new("levels")
                .num_columns(4)
                .striped(true)
                .show(ui, |ui| {
                    for entry in per_level_rows(&snapshot.levels) {
                        ui.monospace(entry[0].clone());
                        ui.monospace(entry[1].clone());
                        ui.monospace(entry[2].clone());
                        ui.monospace(entry[3].clone());
                        ui.end_row();
                    }
                });
        });
}

/// The per-level table rows: level, resident cubes, cubes the cull drew
/// this frame, faces issued this frame.
fn per_level_rows(levels: &[LevelStats]) -> Vec<[String; 4]> {
    levels
        .iter()
        .map(|stats| {
            [
                format!("L{}", stats.level),
                stats.resident.to_string(),
                stats.frame_drawn.to_string(),
                stats.frame_faces.to_string(),
            ]
        })
        .collect()
}

/// Views: the debug toggles. Each control edits `RenderSettings` — the
/// renderer and the terrain pass apply it next frame, on every draw path
/// (the path flags switch the path live).
fn views_section(ui: &mut egui::Ui, settings: &mut RenderSettings) {
    egui::CollapsingHeader::new("Views").show(ui, |ui| {
        ui.checkbox(&mut settings.wireframe, "wireframe (JBM_WIREFRAME)");
        ui.add_enabled(
            settings.wireframe,
            egui::Checkbox::new(&mut settings.wireframe_only, "wireframe only"),
        );
        ui.checkbox(
            &mut settings.chunk_borders,
            "chunk borders (JBM_CHUNK_BORDERS)",
        );
        ui.checkbox(&mut settings.lod_tint, "level tint (F4, JBM_LOD_TINT)");
        ui.checkbox(
            &mut settings.freeze_cull,
            "freeze culling (JBM_FREEZE_CULL)",
        );
        ui.separator();
        // The explode slider: 0 is off; while nonzero the cull skips its
        // frustum test (the moved cubes sit outside every bound).
        let mut explode = settings.explode as i32;
        ui.horizontal(|ui| {
            ui.label("exploded (JBM_EXPLODE)");
            ui.add(
                egui::DragValue::new(&mut explode)
                    .range(0..=8)
                    .suffix(" gap"),
            );
        });
        settings.explode = explode.clamp(0, 8) as u32;
        ui.separator();
        ui.label("levels (JBM_LEVELS): a hidden level leaves a hole");
        ui.horizontal_wrapped(|ui| {
            for level in 0..LOD_LEVELS {
                let mut shown = settings.level_mask & (1 << level) != 0;
                let label = format!("L{level}");
                if ui.toggle_value(&mut shown, label).changed() {
                    settings.level_mask ^= 1 << level;
                }
            }
        });
        ui.separator();
        ui.checkbox(&mut settings.fog, "distance fog");
        ui.checkbox(&mut settings.sun_lighting, "sun lighting (JBM_NO_SUN)");
        ui.checkbox(&mut settings.clouds, "clouds (JBM_NO_CLOUDS)");
        ui.checkbox(&mut settings.water, "water shading (JBM_NO_WATER)");
        // Read-only: the pipelines are built for the startup sample count.
        ui.label(format!(
            "MSAA {}x (JBM_MSAA, at startup)",
            settings.msaa_samples
        ));
        ui.checkbox(
            &mut settings.prefer_gpu_cull,
            "prefer gpu cull (JBM_GPU_CULL)",
        );
        ui.checkbox(&mut settings.force_cpu_cull, "force cpu cull");
        ui.checkbox(&mut settings.force_fallback, "force fallback (JBM_NO_MDI)");
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
            world.lod_view_radius = clamped_ring_radius(world.view_radius, world.lod_view_radius);
            (hooks.set_view_radius)(world.view_radius, world.lod_view_radius);
            let (lod0, furthest) = jbm_core::lod::ring_radii(
                world.view_radius,
                world.lod_view_radius,
                world.lod_levels,
            );
            settings.far_radii = [lod0, furthest];
        }
        for (label, value) in server_lines {
            ui.monospace(format!("{label}: {value}"));
        }
    });
}

/// Task 2.13 E's floor for the World sliders: a drag that left
/// `lod_view_radius` below `view_radius / 2 + 1` would put L1 inside
/// the LOD0 disk, so the value rises to the floor — the same clamp the
/// server applies, mirrored so the slider shows what will stream.
pub(crate) fn clamped_ring_radius(view_radius: u32, lod_view_radius: u32) -> u32 {
    lod_view_radius.max(jbm_core::lod::min_lod_view_radius(view_radius))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-level rows carry the cull's frame counters, not the
    /// residency-only ones.
    #[test]
    fn per_level_rows_show_frame_counters() {
        let levels = vec![
            LevelStats {
                level: 0,
                resident: 120,
                drawn: 120,
                frame_drawn: 61,
                faces: 90_000,
                frame_faces: 30_000,
                ..LevelStats::default()
            },
            LevelStats {
                level: 1,
                resident: 40,
                drawn: 12,
                frame_drawn: 3,
                faces: 2_000,
                frame_faces: 400,
                ..LevelStats::default()
            },
        ];
        let rows = per_level_rows(&levels);
        assert_eq!(rows[0], ["L0", "120", "61", "30000"]);
        assert_eq!(rows[1], ["L1", "40", "3", "400"]);
    }

    /// Task 2.13 E: the ring-radius slider can never leave L1 inside
    /// the LOD0 disk — the floor the server also enforces.
    #[test]
    fn ring_radius_slider_never_starts_l1_inside_lod0() {
        // The shipped defaults already satisfy the floor.
        assert_eq!(clamped_ring_radius(16, 16), 16);
        // A low drag rises to view_radius / 2 + 1.
        assert_eq!(clamped_ring_radius(16, 2), 9);
        assert_eq!(clamped_ring_radius(8, 4), 5);
        // The slider's 2..=16 top is not enough at view_radius 32.
        assert_eq!(clamped_ring_radius(32, 16), 17);
        // Higher values pass through untouched.
        assert_eq!(clamped_ring_radius(4, 16), 16);
    }
}
