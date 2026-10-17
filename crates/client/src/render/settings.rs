//! Renderer-owned frame settings: one struct for the toggles and view
//! distances the renderer and the terrain pass must agree on — fog, the
//! level tint, the far-look radii and the terrain draw-path flags. The
//! renderer owns the one live copy and hands `&RenderSettings` to
//! `TerrainPass::draw` each frame; the app reads the environment once at
//! startup through [`RenderSettings::from_env`] (the only place the
//! JBM_* rendering variables are read). The debug panel adds its toggles
//! as fields here.

use crate::render::terrain::{DEFAULT_FURTHEST_RING_RADIUS, DEFAULT_LOD0_RING_RADIUS};

/// The frame-level rendering settings.
///
/// - `fog` / `lod_tint` / `far_radii` are per-frame visual state: the
///   renderer passes them to the terrain pass with every draw.
/// - `prefer_gpu_cull` / `force_cpu_cull` / `force_fallback` seed the
///   terrain pass's path choice at construction (`TerrainPass::with_settings`);
///   the pass's own `set_force_*` setters remain the test and bench
///   override.
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
    /// Force the cpu-cull path (no environment variable; tests, benches).
    pub force_cpu_cull: bool,
    /// Force the direct-draw fallback; on via `JBM_NO_MDI=1`.
    pub force_fallback: bool,
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
        }
    }
}

impl RenderSettings {
    /// Reads the rendering environment variables, each checked as
    /// `NAME=1`: `JBM_NO_FOG` (fog off), `JBM_LOD_TINT` (tint on),
    /// `JBM_GPU_CULL` (prefer gpu-cull) and `JBM_NO_MDI` (force the
    /// direct-draw fallback). The app calls this once at startup and
    /// never reads them again.
    pub fn from_env() -> RenderSettings {
        let flag = |name: &str| std::env::var_os(name).is_some_and(|v| v == "1");
        RenderSettings::from_flags(
            flag("JBM_NO_FOG"),
            flag("JBM_LOD_TINT"),
            flag("JBM_GPU_CULL"),
            flag("JBM_NO_MDI"),
        )
    }

    /// The environment mapping, pure so it can be unit-tested.
    fn from_flags(
        no_fog: bool,
        lod_tint: bool,
        prefer_gpu_cull: bool,
        no_mdi: bool,
    ) -> RenderSettings {
        RenderSettings {
            fog: !no_fog,
            lod_tint,
            prefer_gpu_cull,
            force_fallback: no_mdi,
            ..RenderSettings::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_documented_choices() {
        let settings = RenderSettings::default();
        assert!(settings.fog, "fog is on by default");
        assert!(!settings.lod_tint);
        assert!(!settings.prefer_gpu_cull, "cpu-cull is the default path");
        assert!(!settings.force_cpu_cull);
        assert!(!settings.force_fallback);
        assert_eq!(
            settings.far_radii,
            [DEFAULT_LOD0_RING_RADIUS, DEFAULT_FURTHEST_RING_RADIUS]
        );
    }

    #[test]
    fn env_flags_map_to_settings() {
        // Nothing set: the defaults.
        let settings = RenderSettings::from_flags(false, false, false, false);
        assert_eq!(settings, RenderSettings::default());
        // Each variable flips exactly its own field.
        let settings = RenderSettings::from_flags(true, false, false, false);
        assert!(!settings.fog);
        assert!(!settings.lod_tint);
        assert!(!settings.prefer_gpu_cull);
        assert!(!settings.force_fallback);
        let settings = RenderSettings::from_flags(false, true, false, false);
        assert!(settings.fog);
        assert!(settings.lod_tint);
        let settings = RenderSettings::from_flags(false, false, true, false);
        assert!(settings.prefer_gpu_cull);
        assert!(!settings.force_fallback);
        // JBM_NO_MDI forces the fallback (the same path a missing-MDI
        // adapter selects).
        let settings = RenderSettings::from_flags(false, false, true, true);
        assert!(settings.force_fallback);
        assert!(settings.prefer_gpu_cull, "the flags are independent");
    }
}
