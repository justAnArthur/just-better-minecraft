//! [`ServerConfig`]: every tuning knob of a
//! [`Server`](super::Server) in one place. The crate README documents what
//! each field costs; the defaults fit one player streaming a view radius
//! of 8 cubes.

use std::path::PathBuf;
use std::time::Duration;

/// The seed every default server generates its world with. Fixed so a
/// fresh install always starts on the same terrain.
pub const DEFAULT_SEED: u64 = 0x0055_EED1_5EED;

/// Tuning knobs for a [`Server`](super::Server). The crate README documents
/// what each field costs; the defaults fit one player streaming a view
/// radius of 8 cubes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    /// World seed fed to the terrain generator.
    pub seed: u64,
    /// Which shipped worldgen graph generates the world. `Overworld` (the
    /// default) is the dramatic terrain with a sea; `Classic` is the older,
    /// flatter terrain, pinned by tests, benches and goldens that depend
    /// on it.
    pub worldgen: crate::worldgen::Preset,
    /// Whether cubes are streamed to players at all.
    pub streaming: bool,
    /// Horizontal streaming radius, in cubes (`dx² + dz² ≤ r²`). Task
    /// 2.13 E raised the default 8 -> 16: LOD0 (full detail) reaches
    /// ~512 blocks, and the coarse rings start only beyond it.
    pub view_radius: u32,
    /// Vertical streaming radius, in cubes (`|dy| ≤ v`).
    pub vertical_radius: u32,
    /// Most `CubeData` messages sent to one player per tick.
    pub send_budget_per_tick: usize,
    /// Generation worker threads. `0` selects inline mode: the tick thread
    /// itself generates up to `gen_budget_per_tick` cubes per tick, which is
    /// deterministic — what tests and benches use.
    pub gen_threads: usize,
    /// Most cubes generated inline per tick (inline mode only).
    pub gen_budget_per_tick: usize,
    /// Cap on cubes queued, generating, or generated but not yet collected.
    pub max_in_flight: usize,
    /// World directory for persistence. `None` (the default) disables
    /// persistence: edited cubes stay pinned in memory forever.
    pub world_dir: Option<PathBuf>,
    /// How often dirty cubes are snapshotted to the save thread.
    pub save_interval: Duration,
    /// Top LOD level the server keeps (5 since task 2.13 E: each ring is
    /// twice the previous — 1, 2, 4, 8, 16 km — with `lod_view_radius` 16;
    /// the same ~16 km far reach the old 6 levels had). `0` disables the
    /// pyramid entirely: no allocation, no work per cube or edit.
    pub lod_levels: u8,
    /// Horizontal radius of every LOD ring, in cubes of that ring's level.
    /// Each level streams a full disk; the client draws the finest ready
    /// level per octant. 16 since task 2.13 E: level *n* reaches
    /// `lod_view_radius · 32 · 2^n` blocks (1 km at L1 … 16 km at L5), so
    /// L1 starts beyond the 512-block LOD0 disk instead of inside it.
    pub lod_view_radius: u32,
    /// Vertical radius of every LOD ring, in cubes of that ring's level.
    pub lod_vertical_radius: u32,
    /// Most `LodCubeData` messages sent to one player per tick, re-sends
    /// of changed cubes included.
    pub lod_send_budget_per_tick: usize,
    /// Cap on LOD cubes requested from the pyramid but not yet ready
    /// ([`LodPyramid::in_flight`](crate::lod::LodPyramid::in_flight)).
    pub lod_max_in_flight: usize,
    /// Surface-following windows: the most cubes desired per column per
    /// level **beyond the camera's vertical window**, nearest to the camera
    /// first. 16 cubes of cliff face over a valley camera; flat terrain
    /// around the camera is never affected (its span sits inside the
    /// window).
    pub max_column_cubes: usize,
    /// The camera window radius (task 2.13 D), in cubes: the LOD0 and
    /// level-1 vertical windows (`vertical_radius`, `lod_vertical_radius`)
    /// apply only within this horizontal distance of the camera; beyond
    /// it a column streams only its surface span (`max_column_cubes`
    /// cubes nearest the camera, the extra cube below included). A
    /// digging or flying player keeps the full window around them; far
    /// columns at the player's depth are invisible in a cave-less world.
    /// Coarser rings keep their plain windows (no spans above level 1;
    /// their defaults already cover the shipped relief).
    pub camera_window_radius: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        let parallelism = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self {
            seed: DEFAULT_SEED,
            worldgen: crate::worldgen::Preset::Overworld,
            streaming: true,
            view_radius: 16,
            vertical_radius: 4,
            send_budget_per_tick: 64,
            gen_threads: parallelism.saturating_sub(2).max(1),
            gen_budget_per_tick: 64,
            max_in_flight: 512,
            world_dir: None,
            save_interval: Duration::from_secs(5),
            lod_levels: 5,
            lod_view_radius: 16,
            lod_vertical_radius: 3,
            lod_send_budget_per_tick: 32,
            lod_max_in_flight: 256,
            max_column_cubes: 16,
            camera_window_radius: 4,
        }
    }
}

impl ServerConfig {
    /// The ring-radius floor of task 2.13 E: `lod_view_radius` must reach
    /// at least `view_radius / 2 + 1`, so level 1 (`lod_view_radius · 64`
    /// blocks) starts beyond the LOD0 disk (`view_radius · 32` blocks) by
    /// at least one L1 cube — otherwise L1 draws nothing and LOD0's edge
    /// meets a hole. [`ServerConfig::clamped`] enforces it.
    pub fn min_lod_view_radius(view_radius: u32) -> u32 {
        jbm_core::lod::min_lod_view_radius(view_radius)
    }

    /// [`ServerConfig`] with the ring-radius floor enforced: raises
    /// `lod_view_radius` to `view_radius / 2 + 1` with a one-line warning
    /// when an override (environment, slider, test) left L1 inside LOD0.
    /// The game and `Server::with_config` both build their config through
    /// here, so the streamed rings and the client's far-look radii (which
    /// the game copies before the server exists) can never disagree.
    pub fn clamped(mut self) -> Self {
        let min = Self::min_lod_view_radius(self.view_radius);
        if self.lod_view_radius < min {
            eprintln!(
                "jbm: lod_view_radius {} would leave L1 inside LOD0 (view_radius {}); using {min}",
                self.lod_view_radius, self.view_radius
            );
            self.lod_view_radius = min;
        }
        self
    }

    /// Streaming off and no generation threads: a server that only speaks
    /// the handshake/ping/tick protocol. Tests and benches that assert
    /// exact message sequences use this so their meaning stays unchanged.
    /// The LOD pyramid is off too.
    pub fn protocol_only() -> Self {
        Self {
            streaming: false,
            gen_threads: 0,
            lod_levels: 0,
            ..Self::default()
        }
    }

    /// The far-look view distances in blocks: `(lod0 disk radius,
    /// outermost LOD ring radius)`, by [`jbm_core::lod::ring_radii`]. The
    /// client cannot learn these from the link, so the game, which built
    /// the config, passes the radii along at startup.
    pub fn ring_radii(&self) -> (f32, f32) {
        jbm_core::lod::ring_radii(self.view_radius, self.lod_view_radius, self.lod_levels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Task 2.13 E: the shipped defaults already satisfy the
    /// ring-radius floor, so `clamped` leaves them alone; an override
    /// that left L1 inside the LOD0 disk rises to the floor; nothing
    /// above it moves.
    #[test]
    fn clamped_enforces_the_ring_radius_floor() {
        let defaults = ServerConfig::default().clamped();
        assert_eq!(
            (
                defaults.view_radius,
                defaults.lod_view_radius,
                defaults.lod_levels
            ),
            (16, 16, 5),
            "the task 2.13 E defaults"
        );

        let low = ServerConfig {
            view_radius: 16,
            lod_view_radius: 8,
            ..ServerConfig::default()
        }
        .clamped();
        assert_eq!(low.lod_view_radius, 9, "16 / 2 + 1");
        assert_eq!(low.view_radius, 16, "only the ring radius moves");

        let high = ServerConfig {
            view_radius: 8,
            lod_view_radius: 16,
            ..ServerConfig::default()
        }
        .clamped();
        assert_eq!(high.lod_view_radius, 16, "above the floor is untouched");
    }

    /// The floor keeps level 1's ring (in blocks) strictly outside the
    /// LOD0 disk at every slider combination — a full L1 cube beyond
    /// at even view radii, half a cube at odd ones (integer division).
    #[test]
    fn floor_keeps_l1_beyond_lod0() {
        for view_radius in 2..=32u32 {
            let lod = ServerConfig::min_lod_view_radius(view_radius);
            let cube = u64::try_from(jbm_core::coords::CUBE_SIZE).unwrap();
            let lod0_blocks = u64::from(view_radius) * cube;
            let l1_blocks = u64::from(lod) * cube * 2;
            assert!(
                l1_blocks > lod0_blocks,
                "L1 at {l1_blocks} blocks is not beyond LOD0 at {lod0_blocks}"
            );
        }
    }
}
