//! Renderer-owned frame settings: one struct for the toggles and view
//! distances the renderer and the terrain pass must agree on — fog, the
//! level tint, the far-look radii, the terrain draw-path flags and the
//! task 2.12 debug views. The renderer owns the one live copy and hands
//! `&RenderSettings` to `TerrainPass::prepare` each frame; the app reads the
//! environment once at startup through [`RenderSettings::from_env`] (the
//! only place the JBM_* rendering variables are read). The debug panel
//! edits the same fields live.

use crate::render::terrain::{DEFAULT_FURTHEST_RING_RADIUS, DEFAULT_LOD0_RING_RADIUS};

/// How many LOD levels exist (levels 0..=6): the width of `level_mask`.
pub const LOD_LEVELS: u32 = 7;

/// Every level shown (the default level mask).
pub const ALL_LEVELS: u8 = 0x7F;

/// The frame-level rendering settings.
///
/// - `fog` / `lod_tint` / `far_radii` are per-frame visual state: the
///   renderer passes them to the terrain pass with every draw.
/// - `prefer_gpu_cull` / `force_cpu_cull` / `force_fallback` seed the
///   terrain pass's path choice; the pass re-reads them from the settings
///   every draw (task 2.12), so the panel's checkboxes switch the path
///   live. The pass's own `set_force_*` setters remain the test and bench
///   override.
/// - The debug-view fields (task 2.12) ride the same path: the terrain
///   pass bakes them into its per-frame uniforms and cull params.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSettings {
    /// Whether terrain distance fog applies; off via `JBM_NO_FOG=1`.
    /// Screenshot tests switch it to isolate geometry from fog. Fog never
    /// touches the sky pass, so the sentinel hole tests keep working with
    /// it on.
    pub fog: bool,
    /// Whether the level tint debug view applies (every fragment
    /// multiplied by a fixed color per LOD level); on via F4 or
    /// `JBM_LOD_TINT=1`.
    pub lod_tint: bool,
    /// Far-look view distances in blocks: `[LOD0 ring radius, furthest
    /// ring radius]`. `Renderer::set_view_distance` sets them.
    pub far_radii: [f32; 2],
    /// Prefer the gpu-cull path over the cpu-cull default; on via
    /// `JBM_GPU_CULL=1`. Ignored when the adapter is not gpu-capable.
    pub prefer_gpu_cull: bool,
    /// Force the cpu-cull path (no environment variable; tests, benches,
    /// the panel's checkbox).
    pub force_cpu_cull: bool,
    /// Force the direct-draw fallback; on via `JBM_NO_MDI=1`.
    pub force_fallback: bool,
    /// The wireframe debug view (task 2.12): the terrain fragment shader
    /// darkens every triangle edge (quad outlines and diagonals); on via
    /// `JBM_WIREFRAME=1`.
    pub wireframe: bool,
    /// Wireframe only (task 2.12): the edges over a flat background, no
    /// textures, shade, fog or tint; on via `JBM_WIREFRAME_ONLY=1`
    /// (implies `wireframe`).
    pub wireframe_only: bool,
    /// The chunk-border debug view (task 2.12): the renderer's line pass
    /// draws resident cube boundaries, colored by the tint palette; on via
    /// `JBM_CHUNK_BORDERS=1`.
    pub chunk_borders: bool,
    /// The exploded-cube debug view's gap in blocks (task 2.12): the
    /// vertex shader shifts every cube by `gap × its cube-grid offset`,
    /// spreading the world apart. 0 = off; `JBM_EXPLODE=<gap>` seeds it.
    /// While nonzero, both cull paths skip frustum culling (the moved
    /// cubes sit outside every conservative bound).
    pub explode: u32,
    /// Which LOD levels draw, one bit per level 0..=6 (task 2.12): a
    /// hidden level leaves a hole. `JBM_LEVELS=0,1,3` shows exactly those
    /// levels; the panel's per-level checkboxes edit the bits.
    pub level_mask: u8,
    /// The freeze-culling debug view (task 2.12): the cull keeps using the
    /// frustum and position captured when it was switched on while the
    /// camera moves freely, so what draws no longer follows the view; on
    /// via `JBM_FREEZE_CULL=1`.
    pub freeze_cull: bool,
    /// The direction-split radius, in cubes (task 2.13 C): cubes whose
    /// AABB is farther than this from the camera draw all their face
    /// directions in one indirect command — back-face culling drops the
    /// same faces direction skipping dropped, so the image cannot change.
    /// Nearer cubes keep one command per visible direction. 0 draws one
    /// command everywhere, the default (`u32::MAX`) splits everywhere;
    /// `JBM_DIRECTION_SPLIT=<cubes>` seeds it.
    pub direction_split_radius: u32,
    /// Sun and sky lighting (task 2.15): a fixed mid-morning sun plus a
    /// hemisphere ambient light every terrain fragment. Off restores the
    /// pre-2.15 flat look (fixed per-direction multipliers), for
    /// comparison; off via `JBM_NO_SUN=1`.
    pub sun_lighting: bool,
    /// The procedural cloud layer in the sky (task 2.15); off via
    /// `JBM_NO_CLOUDS=1`.
    pub clouds: bool,
    /// Water shading (task 2.15): Fresnel sky reflection, sun glint and
    /// ripples on water faces; off draws water as its flat texture like
    /// any block. Off via `JBM_NO_WATER=1`.
    pub water: bool,
    /// MSAA sample count (task 2.15): 4 (default) or 1, via
    /// `JBM_MSAA=1|4`. Applied when the renderer and terrain pass are
    /// created — pipelines and targets are built for it — so changing it
    /// later has no effect (the panel shows it read-only).
    pub msaa_samples: u32,
    /// The scene's render scale (task 2.16): the terrain, sky, lines and
    /// outline render at this fraction of the window's pixels per axis
    /// and are upscaled to it (the overlay and egui stay native). 0 is
    /// auto ([`effective_render_scale`]: native up to
    /// [`AUTO_SCALE_BUDGET`] pixels, scaled down to it above); otherwise
    /// [`MIN_RENDER_SCALE`]..=1. `JBM_RENDER_SCALE=auto|<fraction>`; the
    /// panel changes it live.
    pub render_scale: f32,
}

impl Default for RenderSettings {
    fn default() -> RenderSettings {
        RenderSettings {
            fog: true,
            lod_tint: false,
            far_radii: [DEFAULT_LOD0_RING_RADIUS, DEFAULT_FURTHEST_RING_RADIUS],
            prefer_gpu_cull: false,
            force_cpu_cull: false,
            force_fallback: false,
            wireframe: false,
            wireframe_only: false,
            chunk_borders: false,
            explode: 0,
            level_mask: ALL_LEVELS,
            freeze_cull: false,
            direction_split_radius: DEFAULT_DIRECTION_SPLIT_RADIUS,
            sun_lighting: true,
            clouds: true,
            water: true,
            msaa_samples: DEFAULT_MSAA_SAMPLES,
            render_scale: AUTO_RENDER_SCALE,
        }
    }
}

impl RenderSettings {
    /// The debug views and look toggles that are off their defaults, by
    /// name, for the HUD (task 2.17). The wireframe overlay draws a dark
    /// line along every triangle edge; left on from the panel with the
    /// panel closed, it reads as broken rendering, so the HUD names it
    /// (and every other non-default view) while it is on. The level tint
    /// has its own legend line.
    pub fn debug_view_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if self.wireframe_only {
            names.push("WIREFRAME-ONLY".to_owned());
        } else if self.wireframe {
            names.push("WIREFRAME".to_owned());
        }
        if self.chunk_borders {
            names.push("CHUNK-BORDERS".to_owned());
        }
        if self.explode > 0 {
            names.push(format!("EXPLODE {}", self.explode));
        }
        if self.level_mask != ALL_LEVELS {
            let shown: Vec<String> = (0..LOD_LEVELS)
                .filter(|level| self.level_mask >> level & 1 != 0)
                .map(|level| level.to_string())
                .collect();
            names.push(format!("LEVELS {}", shown.join(",")));
        }
        if self.freeze_cull {
            names.push("FREEZE-CULL".to_owned());
        }
        for (on, name) in [
            (self.fog, "NO-FOG"),
            (self.sun_lighting, "NO-SUN"),
            (self.clouds, "NO-CLOUDS"),
            (self.water, "NO-WATER"),
        ] {
            if !on {
                names.push(name.to_owned());
            }
        }
        names
    }
}

/// `RenderSettings::render_scale`'s auto value (task 2.16).
pub const AUTO_RENDER_SCALE: f32 = 0.0;

/// The lowest render scale the setting accepts.
pub const MIN_RENDER_SCALE: f32 = 0.25;

/// The scene pixels the auto render scale allows (task 2.16): 2560×1600.
/// Measured with `far_metrics::perf_views` on an M5 Pro, the 4× MSAA
/// scene costs ~4–7 ms at 2560×1440 and ~13–23 ms at 5120×2880, nearly
/// all of it per pixel. Windows up to this budget (every 1440p and
/// smaller display, the test rigs) render natively; above it — a 5K
/// display, a Retina laptop panel — the scene is scaled to the budget,
/// which on a 2× HiDPI screen is about one scene pixel per point.
pub const AUTO_SCALE_BUDGET: u32 = 2560 * 1600;

/// The render scale applied to a `viewport`-sized window: `render_scale`
/// clamped to [`MIN_RENDER_SCALE`]..=1, or for auto the largest scale
/// that fits [`AUTO_SCALE_BUDGET`], capped at 1.
pub fn effective_render_scale(render_scale: f32, (width, height): (u32, u32)) -> f32 {
    if render_scale > 0.0 {
        return render_scale.clamp(MIN_RENDER_SCALE, 1.0);
    }
    let pixels = f64::from(width) * f64::from(height);
    (f64::from(AUTO_SCALE_BUDGET) / pixels.max(1.0))
        .sqrt()
        .min(1.0) as f32
}

/// The scene's pixel size for a `viewport`-sized window at
/// `render_scale` ([`effective_render_scale`]); the viewport itself at
/// scale 1, never below 1×1.
pub fn scene_size(render_scale: f32, viewport: (u32, u32)) -> (u32, u32) {
    let scale = effective_render_scale(render_scale, viewport);
    if scale >= 1.0 {
        return viewport;
    }
    let axis = |pixels: u32| ((pixels as f32 * scale).round() as u32).max(1);
    (axis(viewport.0), axis(viewport.1))
}

/// Maps a `JBM_RENDER_SCALE` value to the setting: a fraction in
/// (0, 1] (clamped to [`MIN_RENDER_SCALE`]), or auto for anything else,
/// including unset and "auto".
fn render_scale_from_env(value: Option<&str>) -> f32 {
    match value.map(str::trim).and_then(|v| v.parse::<f32>().ok()) {
        Some(scale) if scale > 0.0 => scale.clamp(MIN_RENDER_SCALE, 1.0),
        _ => AUTO_RENDER_SCALE,
    }
}

/// The default MSAA sample count (task 2.15). `JBM_MSAA=1` switches it
/// off; 4 is the only other count every wgpu backend guarantees for
/// render-attachment formats.
pub const DEFAULT_MSAA_SAMPLES: u32 = 4;

/// Maps a `JBM_MSAA` value to a supported sample count: "1" is 1, "4"
/// (or anything else, including unset) the default.
fn msaa_from_env(value: Option<&str>) -> u32 {
    match value.map(str::trim) {
        Some("1") => 1,
        _ => DEFAULT_MSAA_SAMPLES,
    }
}

/// The default direction-split radius (task 2.13 C), in cubes: none —
/// every cube draws one command per visible direction. Task 2.13 set 4
/// cubes to collapse the far bulk to one command per cube; task 2.16
/// measured the cost of that: the far cubes' back faces (~35 % of a
/// 2 M-face view) then reach the vertex shader, which is the geometry
/// floor of the frame. Splitting everywhere roughly doubles the commands
/// (~3.5 k → ~6.6 k, +0.2 ms CPU) and takes 0.3–0.7 ms off the GPU.
pub const DEFAULT_DIRECTION_SPLIT_RADIUS: u32 = u32::MAX;

impl RenderSettings {
    /// Reads the rendering environment variables: `JBM_NO_FOG`,
    /// `JBM_LOD_TINT`, `JBM_GPU_CULL` and `JBM_NO_MDI` as `NAME=1`;
    /// `JBM_WIREFRAME`, `JBM_WIREFRAME_ONLY`, `JBM_CHUNK_BORDERS` and
    /// `JBM_FREEZE_CULL` as `NAME=1`; `JBM_EXPLODE=<gap>` (any value,
    /// clamped to 0..=8) and `JBM_LEVELS=<comma-separated level list>` as
    /// the shown-levels mask; task 2.15's `JBM_NO_SUN`, `JBM_NO_CLOUDS`
    /// and `JBM_NO_WATER` as `NAME=1` and `JBM_MSAA=1|4`; task 2.16's
    /// `JBM_RENDER_SCALE=auto|<fraction>`. The app calls this once at
    /// startup and never reads them again.
    pub fn from_env() -> RenderSettings {
        let flag = |name: &str| std::env::var_os(name).is_some_and(|v| v == "1");
        let explode = std::env::var("JBM_EXPLODE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(0)
            .min(8);
        let level_mask = level_mask_from_list(&std::env::var("JBM_LEVELS").unwrap_or_default());
        let direction_split = std::env::var("JBM_DIRECTION_SPLIT")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(DEFAULT_DIRECTION_SPLIT_RADIUS);
        RenderSettings::from_flags(
            DebugFlags {
                no_fog: flag("JBM_NO_FOG"),
                lod_tint: flag("JBM_LOD_TINT"),
                prefer_gpu_cull: flag("JBM_GPU_CULL"),
                no_mdi: flag("JBM_NO_MDI"),
                wireframe: flag("JBM_WIREFRAME"),
                wireframe_only: flag("JBM_WIREFRAME_ONLY"),
                chunk_borders: flag("JBM_CHUNK_BORDERS"),
                freeze_cull: flag("JBM_FREEZE_CULL"),
                no_sun: flag("JBM_NO_SUN"),
                no_clouds: flag("JBM_NO_CLOUDS"),
                no_water: flag("JBM_NO_WATER"),
            },
            explode,
            level_mask,
            direction_split,
        )
        .with_msaa(msaa_from_env(std::env::var("JBM_MSAA").ok().as_deref()))
        .with_render_scale(render_scale_from_env(
            std::env::var("JBM_RENDER_SCALE").ok().as_deref(),
        ))
    }

    /// The same settings at `render_scale`.
    fn with_render_scale(self, render_scale: f32) -> RenderSettings {
        RenderSettings {
            render_scale,
            ..self
        }
    }

    /// The same settings with `samples` MSAA samples.
    fn with_msaa(self, samples: u32) -> RenderSettings {
        RenderSettings {
            msaa_samples: samples,
            ..self
        }
    }

    /// The environment mapping, pure so it can be unit-tested.
    fn from_flags(
        flags: DebugFlags,
        explode: u32,
        level_mask: u8,
        direction_split_radius: u32,
    ) -> RenderSettings {
        RenderSettings {
            fog: !flags.no_fog,
            lod_tint: flags.lod_tint,
            prefer_gpu_cull: flags.prefer_gpu_cull,
            force_fallback: flags.no_mdi,
            wireframe: flags.wireframe || flags.wireframe_only,
            wireframe_only: flags.wireframe_only,
            chunk_borders: flags.chunk_borders,
            explode,
            level_mask,
            freeze_cull: flags.freeze_cull,
            direction_split_radius,
            sun_lighting: !flags.no_sun,
            clouds: !flags.no_clouds,
            water: !flags.no_water,
            ..RenderSettings::default()
        }
    }
}

/// The boolean half of the environment mapping.
#[derive(Debug, Clone, Copy)]
struct DebugFlags {
    no_fog: bool,
    lod_tint: bool,
    prefer_gpu_cull: bool,
    no_mdi: bool,
    wireframe: bool,
    wireframe_only: bool,
    chunk_borders: bool,
    freeze_cull: bool,
    no_sun: bool,
    no_clouds: bool,
    no_water: bool,
}

/// Parses `JBM_LEVELS`'s comma-separated level list ("0,1,3") into the
/// bit mask. Empty or unparsable input keeps every level; entries outside
/// 0..=6 are ignored.
fn level_mask_from_list(list: &str) -> u8 {
    let trimmed = list.trim();
    if trimmed.is_empty() {
        return ALL_LEVELS;
    }
    let mut mask = 0u8;
    let mut any = false;
    for part in trimmed.split(',') {
        if let Ok(level) = part.trim().parse::<u32>()
            && level < LOD_LEVELS
        {
            mask |= 1 << level;
            any = true;
        }
    }
    if any { mask } else { ALL_LEVELS }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags() -> DebugFlags {
        DebugFlags {
            no_fog: false,
            lod_tint: false,
            prefer_gpu_cull: false,
            no_mdi: false,
            wireframe: false,
            wireframe_only: false,
            chunk_borders: false,
            freeze_cull: false,
            no_sun: false,
            no_clouds: false,
            no_water: false,
        }
    }

    /// The HUD names no view at the defaults, and each non-default
    /// debug view and look toggle once (task 2.17).
    #[test]
    fn debug_view_names_list_the_non_default_views() {
        assert!(RenderSettings::default().debug_view_names().is_empty());
        let settings = RenderSettings {
            wireframe: true,
            explode: 2,
            level_mask: 0b101,
            water: false,
            ..RenderSettings::default()
        };
        assert_eq!(
            settings.debug_view_names(),
            vec!["WIREFRAME", "EXPLODE 2", "LEVELS 0,2", "NO-WATER"]
        );
        let only = RenderSettings {
            wireframe: true,
            wireframe_only: true,
            ..RenderSettings::default()
        };
        assert_eq!(only.debug_view_names(), vec!["WIREFRAME-ONLY"]);
    }

    #[test]
    fn defaults_match_the_documented_choices() {
        let settings = RenderSettings::default();
        assert!(settings.fog, "fog is on by default");
        assert!(!settings.lod_tint);
        assert!(!settings.prefer_gpu_cull, "cpu-cull is the default path");
        assert!(!settings.force_cpu_cull);
        assert!(!settings.force_fallback);
        assert_eq!(settings.level_mask, ALL_LEVELS, "every level shows");
        assert!(!settings.wireframe);
        assert!(!settings.wireframe_only);
        assert!(!settings.chunk_borders);
        assert_eq!(settings.explode, 0);
        assert!(!settings.freeze_cull);
        assert_eq!(
            settings.direction_split_radius, DEFAULT_DIRECTION_SPLIT_RADIUS,
            "per-direction everywhere by default"
        );
        assert_eq!(
            settings.far_radii,
            [DEFAULT_LOD0_RING_RADIUS, DEFAULT_FURTHEST_RING_RADIUS]
        );
        assert!(settings.sun_lighting && settings.clouds && settings.water);
        assert_eq!(settings.msaa_samples, 4, "4x MSAA by default");
        assert_eq!(settings.render_scale, AUTO_RENDER_SCALE, "auto by default");
    }

    /// Task 2.16: auto keeps every window up to the budget native and
    /// scales larger ones down to it; a fixed scale is clamped.
    #[test]
    fn render_scale_fits_the_budget() {
        let auto = AUTO_RENDER_SCALE;
        for native in [(1280, 720), (2560, 1440), (2560, 1600)] {
            assert_eq!(scene_size(auto, native), native, "{native:?} stays native");
        }
        let (w, h) = scene_size(auto, (5120, 2880));
        assert!(w * h <= AUTO_SCALE_BUDGET + w + h, "5K fits the budget");
        assert!(w * h >= AUTO_SCALE_BUDGET - w - h, "and fills it");
        assert!(
            (w as f32 / h as f32 - 16.0 / 9.0).abs() < 0.01,
            "the aspect is kept"
        );
        let (w, h) = scene_size(auto, (3024, 1964));
        assert!(w < 3024 && w * h <= AUTO_SCALE_BUDGET + w + h);
        assert_eq!(scene_size(0.5, (5120, 2880)), (2560, 1440));
        assert_eq!(scene_size(1.0, (5120, 2880)), (5120, 2880));
        assert_eq!(scene_size(0.01, (400, 200)), (100, 50), "clamped to 0.25");
        assert_eq!(scene_size(auto, (0, 0)), (0, 0));
        assert_eq!(render_scale_from_env(Some(" 0.5 ")), 0.5);
        assert_eq!(render_scale_from_env(Some("2")), 1.0);
        assert_eq!(render_scale_from_env(Some("0.1")), MIN_RENDER_SCALE);
        assert_eq!(render_scale_from_env(Some("auto")), AUTO_RENDER_SCALE);
        assert_eq!(render_scale_from_env(Some("0")), AUTO_RENDER_SCALE);
        assert_eq!(render_scale_from_env(None), AUTO_RENDER_SCALE);
    }

    /// Task 2.15's look switches: each `JBM_NO_*` flips exactly its own
    /// field, and `JBM_MSAA` accepts 1 and 4 (anything else is 4).
    #[test]
    fn look_env_flags_map_to_settings() {
        let base = || RenderSettings::from_flags(flags(), 0, ALL_LEVELS, 4);
        let mut f = flags();
        f.no_sun = true;
        let settings = RenderSettings::from_flags(f, 0, ALL_LEVELS, 4);
        assert_eq!(
            settings,
            RenderSettings {
                sun_lighting: false,
                ..base()
            }
        );
        let mut f = flags();
        f.no_clouds = true;
        assert!(!RenderSettings::from_flags(f, 0, ALL_LEVELS, 4).clouds);
        let mut f = flags();
        f.no_water = true;
        assert!(!RenderSettings::from_flags(f, 0, ALL_LEVELS, 4).water);
        assert_eq!(msaa_from_env(Some("1")), 1);
        assert_eq!(msaa_from_env(Some(" 4 ")), 4);
        assert_eq!(msaa_from_env(Some("8")), 4, "unsupported counts fall back");
        assert_eq!(msaa_from_env(None), 4);
        assert_eq!(base().with_msaa(1).msaa_samples, 1);
    }

    #[test]
    fn env_flags_map_to_settings() {
        // Nothing set: the defaults.
        let settings =
            RenderSettings::from_flags(flags(), 0, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS);
        assert_eq!(settings, RenderSettings::default());
        // Each variable flips exactly its own field.
        let mut f = flags();
        f.no_fog = true;
        let settings = RenderSettings::from_flags(f, 0, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS);
        assert!(!settings.fog);
        assert!(!settings.lod_tint);
        assert!(!settings.prefer_gpu_cull);
        assert!(!settings.force_fallback);
        let mut f = flags();
        f.lod_tint = true;
        assert!(
            RenderSettings::from_flags(f, 0, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS).lod_tint
        );
        let mut f = flags();
        f.prefer_gpu_cull = true;
        let settings = RenderSettings::from_flags(f, 0, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS);
        assert!(settings.prefer_gpu_cull);
        assert!(!settings.force_fallback);
        // JBM_NO_MDI forces the fallback (the same path a missing-MDI
        // adapter selects).
        let mut f = flags();
        f.no_mdi = true;
        f.prefer_gpu_cull = true;
        let settings = RenderSettings::from_flags(f, 0, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS);
        assert!(settings.force_fallback);
        assert!(settings.prefer_gpu_cull, "the flags are independent");
    }

    /// The task 2.12 debug-view variables: each flips its own field, and
    /// `JBM_WIREFRAME_ONLY=1` implies the wireframe.
    #[test]
    fn debug_env_flags_map_to_settings() {
        let mut f = flags();
        f.wireframe = true;
        let settings = RenderSettings::from_flags(f, 0, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS);
        assert!(settings.wireframe);
        assert!(!settings.wireframe_only);
        let mut f = flags();
        f.wireframe_only = true;
        let settings = RenderSettings::from_flags(f, 0, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS);
        assert!(settings.wireframe, "wireframe-only implies wireframe");
        assert!(settings.wireframe_only);
        let mut f = flags();
        f.chunk_borders = true;
        assert!(
            RenderSettings::from_flags(f, 0, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS)
                .chunk_borders
        );
        let mut f = flags();
        f.freeze_cull = true;
        assert!(
            RenderSettings::from_flags(f, 0, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS)
                .freeze_cull
        );
        let settings =
            RenderSettings::from_flags(flags(), 4, ALL_LEVELS, DEFAULT_DIRECTION_SPLIT_RADIUS);
        assert_eq!(settings.explode, 4);
    }

    /// `JBM_LEVELS` parses a comma-separated list into the mask; junk
    /// keeps every level rather than hiding the world.
    #[test]
    fn levels_list_parses_into_the_mask() {
        assert_eq!(level_mask_from_list("0,1,3"), 0b000_1011);
        assert_eq!(level_mask_from_list("6"), 0b100_0000);
        assert_eq!(level_mask_from_list(" 2 , 4 "), 0b01_0100);
        assert_eq!(level_mask_from_list(""), ALL_LEVELS);
        assert_eq!(level_mask_from_list("junk"), ALL_LEVELS);
        assert_eq!(level_mask_from_list("9,17"), ALL_LEVELS, "out of range");
        assert_eq!(level_mask_from_list("0,x,2"), 0b101, "junk entries skip");
    }
}
