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
    /// Horizontal streaming radius, in cubes (`dx² + dz² ≤ r²`).
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
    /// Top LOD level the server keeps (6 keeps LOD1–LOD6: L5 cells are 32
    /// blocks, L6 64, and the outer ring reaches ~16 km). `0` disables the
    /// pyramid entirely: no allocation, no work per cube or edit.
    pub lod_levels: u8,
    /// Horizontal radius of every LOD ring, in cubes of that ring's level.
    /// Each level streams a full disk; the client draws the finest ready
    /// level per octant.
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
}

impl Default for ServerConfig {
    fn default() -> Self {
        let parallelism = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self {
            seed: DEFAULT_SEED,
            worldgen: crate::worldgen::Preset::Overworld,
            streaming: true,
            view_radius: 8,
            vertical_radius: 4,
            send_budget_per_tick: 64,
            gen_threads: parallelism.saturating_sub(2).max(1),
            gen_budget_per_tick: 64,
            max_in_flight: 512,
            world_dir: None,
            save_interval: Duration::from_secs(5),
            lod_levels: 6,
            lod_view_radius: 8,
            lod_vertical_radius: 3,
            lod_send_budget_per_tick: 32,
            lod_max_in_flight: 256,
            max_column_cubes: 16,
        }
    }
}

impl ServerConfig {
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
