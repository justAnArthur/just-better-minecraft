//! The debug overlay's text: one function from a per-frame stats
//! snapshot to the lines drawn. The app fills the snapshot once per
//! frame; the egui debug panel reads the same snapshot, so the two
//! displays can never disagree.

use glam::DVec3;
use jbm_core::coords::{BlockPos, CUBE_SIZE};

use crate::render::terrain::{CullStats, LevelStats, TerrainStats};

/// The predicted player's share of the overlay (the prediction lines).
#[derive(Debug, Clone, Copy)]
pub struct HudPlayer {
    pub flying: bool,
    pub on_ground: bool,
    pub velocity: DVec3,
    /// Unacknowledged inputs.
    pub pending: usize,
    /// Prediction corrections so far.
    pub corrections: u64,
}

/// One frame's worth of everything the debug overlay prints. Filled by
/// the app; `lines` formats it.
#[derive(Debug, Clone)]
pub struct HudSnapshot {
    /// Sliding-window FPS and frame time.
    pub fps: f64,
    pub frame_ms: f64,
    pub position: DVec3,
    /// Yaw / pitch in radians.
    pub yaw: f32,
    pub pitch: f32,
    /// The crosshair target (block position + state name), if any.
    pub target: Option<(BlockPos, String)>,
    /// The selected hotbar state's name.
    pub hotbar: String,
    /// The predicted player; `None` before the `Welcome` arrives.
    pub player: Option<HudPlayer>,
    /// Cubes held by the client world.
    pub cubes: usize,
    /// Cubes with an uploaded mesh.
    pub meshed: usize,
    /// Dirty cubes plus in-flight mesh jobs.
    pub pending: usize,
    pub terrain: TerrainStats,
    /// The active draw path's name (cpu-cull / gpu-cull / fallback).
    pub path_name: &'static str,
    /// The cull counters (one or two frames late on the gpu-cull path).
    pub cull: CullStats,
    /// Per-LOD-level residency and coverage stats.
    pub levels: Vec<LevelStats>,
    /// The LOD mesh queue length (dirty + in-flight level >= 1 jobs).
    pub lod_queue: usize,
    /// Whether the level tint debug view is on.
    pub lod_tint: bool,
    /// The debug views off their defaults
    /// (`RenderSettings::debug_view_names`); none at the defaults.
    pub views: Vec<String>,
}

/// Formats the snapshot into the overlay lines, in draw order.
pub fn lines(snap: &HudSnapshot) -> Vec<String> {
    let mut lines = Vec::new();
    let p = snap.position;
    lines.push(format!("FPS {:5.0}  {:6.2} ms", snap.fps, snap.frame_ms));
    lines.push(format!("XYZ {:.2} {:.2} {:.2}", p.x, p.y, p.z));
    lines.push(format!(
        "CUBE {} {} {}",
        (p.x / f64::from(CUBE_SIZE)).floor() as i64,
        (p.y / f64::from(CUBE_SIZE)).floor() as i64,
        (p.z / f64::from(CUBE_SIZE)).floor() as i64
    ));
    lines.push(format!(
        "YAW {:.1} PITCH {:.1}",
        snap.yaw.to_degrees(),
        snap.pitch.to_degrees()
    ));
    if let Some((block, name)) = &snap.target {
        lines.push(format!(
            "TARGET {} {} {} {}",
            block.x, block.y, block.z, name
        ));
    } else {
        lines.push("TARGET ---".to_string());
    }
    lines.push(format!("HOTBAR {}", snap.hotbar));
    if let Some(player) = &snap.player {
        lines.push(format!(
            "MODE {} GROUND {}",
            if player.flying { "FLY" } else { "WALK" },
            if player.on_ground { "YES" } else { "NO" }
        ));
        lines.push(format!(
            "VEL {:.2} {:.2} {:.2}",
            player.velocity.x, player.velocity.y, player.velocity.z
        ));
        lines.push(format!(
            "PENDING {} CORRECTIONS {}",
            player.pending, player.corrections
        ));
    } else {
        lines.push("MODE --- (not joined)".to_string());
    }
    lines.push(format!(
        "CUBES {} MESHED {} PENDING {}",
        snap.cubes, snap.meshed, snap.pending
    ));
    lines.push(format!(
        "FACES {} PAGES {} CMDS {} DRAWS {} {}",
        snap.terrain.faces,
        snap.terrain.pages,
        snap.terrain.commands,
        snap.terrain.draw_calls,
        snap.path_name
    ));
    // The cull counters are read back through a 3-deep ring, so this
    // line is one or two frames late.
    lines.push(format!(
        "CULL DRAWN {} FRUSTUM {} OCCLUDED {}",
        snap.cull.drawn(),
        snap.cull.frustum_culled,
        snap.cull.occluded
    ));
    // Per LOD level: resident cubes, drawn cubes, masked octants — plus
    // the LOD mesh queue length.
    let mut lod = String::from("LOD");
    for stats in &snap.levels {
        if stats.level == 0 {
            continue;
        }
        lod.push_str(&format!(
            " L{} {}/{} M{}",
            stats.level, stats.resident, stats.drawn, stats.masked_octants
        ));
    }
    lod.push_str(&format!(" Q {}", snap.lod_queue));
    lines.push(lod);
    // Any debug view that changes the picture is named, so a view left on
    // with the panel closed (the wireframe's edge lines) cannot pass for
    // a rendering fault.
    if !snap.views.is_empty() {
        lines.push(format!("VIEWS {}", snap.views.join(" ")));
    }
    // While the level tint is on, the legend names each level's color.
    if snap.lod_tint {
        lines.push(
            concat!(
                "LOD TINT L1 2x2 red L2 4x4 yellow L3 8x8 green ",
                "L4 16x16 cyan L5 32x32 blue L6 64x64 purple",
            )
            .to_string(),
        );
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A snapshot with every line's input populated: two LOD levels,
    /// a target, a joined player and the tint on.
    fn full() -> HudSnapshot {
        HudSnapshot {
            fps: 1234.5,
            frame_ms: 0.81,
            position: DVec3::new(-1.5, 65.25, 4096.125),
            yaw: 1.0,
            pitch: -0.5,
            target: Some((BlockPos { x: 7, y: 64, z: -9 }, "grass_block".to_string())),
            hotbar: "stone".to_string(),
            player: Some(HudPlayer {
                flying: true,
                on_ground: false,
                velocity: DVec3::new(22.0, 0.0, -1.25),
                pending: 3,
                corrections: 1,
            }),
            cubes: 145,
            meshed: 140,
            pending: 12,
            terrain: TerrainStats {
                faces: 123_456,
                pages: 4,
                commands: 928,
                draw_calls: 8,
                ..TerrainStats::default()
            },
            path_name: "cpu-cull",
            cull: CullStats {
                drawn_early: 900,
                drawn_late: 28,
                frustum_culled: 40,
                occluded: 7,
                ..CullStats::default()
            },
            levels: vec![
                LevelStats {
                    level: 0,
                    resident: 140,
                    drawn: 29,
                    ..LevelStats::default()
                },
                LevelStats {
                    level: 1,
                    resident: 60,
                    drawn: 52,
                    masked_octants: 11,
                    ..LevelStats::default()
                },
            ],
            lod_queue: 5,
            lod_tint: true,
            views: vec!["WIREFRAME".to_string(), "NO-FOG".to_string()],
        }
    }

    /// Every line, byte-for-byte as the overlay drew it before the
    /// formatting moved here — the format strings are the contract.
    #[test]
    fn full_snapshot_formats_to_the_overlay_lines() {
        assert_eq!(
            lines(&full()),
            vec![
                "FPS  1234    0.81 ms",
                "XYZ -1.50 65.25 4096.12",
                "CUBE -1 2 128",
                "YAW 57.3 PITCH -28.6",
                "TARGET 7 64 -9 grass_block",
                "HOTBAR stone",
                "MODE FLY GROUND NO",
                "VEL 22.00 0.00 -1.25",
                "PENDING 3 CORRECTIONS 1",
                "CUBES 145 MESHED 140 PENDING 12",
                "FACES 123456 PAGES 4 CMDS 928 DRAWS 8 cpu-cull",
                "CULL DRAWN 928 FRUSTUM 40 OCCLUDED 7",
                "LOD L1 60/52 M11 Q 5",
                "VIEWS WIREFRAME NO-FOG",
                "LOD TINT L1 2x2 red L2 4x4 yellow L3 8x8 green L4 16x16 cyan L5 32x32 blue L6 64x64 purple",
            ]
        );
    }

    /// Before the Welcome, and with no target, the player block is one
    /// placeholder line and TARGET shows the dash; level 0 alone leaves
    /// a bare `LOD ... Q 0` line, no views line and no tint legend.
    #[test]
    fn minimal_snapshot_marks_the_missing_player_and_target() {
        let mut snap = full();
        snap.player = None;
        snap.target = None;
        snap.levels.truncate(1);
        snap.lod_queue = 0;
        snap.lod_tint = false;
        snap.views.clear();
        let lines = lines(&snap);
        assert_eq!(lines[4], "TARGET ---");
        assert_eq!(lines[6], "MODE --- (not joined)");
        assert_eq!(lines[lines.len() - 1], "LOD Q 0");
        assert_eq!(lines.len(), 11);
    }
}
