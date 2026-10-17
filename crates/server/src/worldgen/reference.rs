//! The original hand-written heightmap terrain, kept verbatim as the
//! reference implementation the graph generator is tested and benchmarked
//! against. Compiled only under the crate's `reference` cargo feature
//! (enabled for the crate's own tests and benches through a self
//! dev-dependency): production code runs the graph path, and the classic
//! preset — which pins the same terrain bit-for-bit as a shipped graph —
//! keeps the goldens meaningful even without this module.
//!
//! Heights are a pure function of `(seed, x, z)` built from the shared
//! noise primitives ([`super::noise`]) by `+`, `-`, `*`, `floor` and one
//! clamp — all bit-exact, so the same seed gives byte-identical terrain
//! everywhere.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CUBE_SIZE, CubePos, LocalPos};
use jbm_core::cube::Cube;

use super::noise::{fbm, noise2};

/// Sea-adjacent level: columns at or below it get sand instead of grass and
/// dirt. There is no water yet (later tasks); this only changes the surface
/// blocks.
const SAND_LEVEL: i64 = 61;

/// Base terrain height before any noise.
const BASE_HEIGHT: f64 = 64.0;

/// Amplitude of the rolling-hill noise, scaled so the observed |noise| of
/// about 0.7 reaches the full ±28 deviation.
const HILL_AMP: f64 = 40.0;
/// Hard clamp on the deviation, so heights provably stay within
/// [`crate::terrain::MIN_HEIGHT`, `crate::terrain::MAX_HEIGHT`] even at the noise's theoretical bound.
const MAX_DEV: f64 = 28.0;
/// Wavelength scale of the large rolling hills (5 fractal octaves).
const HILL_FREQ: f64 = 0.006;
/// Wavelength scale of the low-frequency "hilliness" mask.
const MASK_FREQ: f64 = 0.0009;
/// Octaves of the fractal hill noise.
const OCTAVES: u32 = 5;

/// Clamps `t` into `[0, 1]`. `clamp` is a pure comparison — bit-exact.
#[inline]
fn clamp01(t: f64) -> f64 {
    t.clamp(0.0, 1.0)
}

/// Terrain height at column `(x, z)`: `h = 64 + clamp(40 · hilliness ·
/// rolling, ±28)`, rounded with `floor(v + 0.5)` — exact `+` and `floor`
/// arithmetic. `hilliness` is a very low frequency mask that flattens some
/// regions and amplifies others; `rolling` is the 5-octave fractal.
pub fn height(seed: u64, x: i64, z: i64) -> i64 {
    let xf = x as f64;
    let zf = z as f64;
    let mask = noise2(seed ^ 0x0D1B_54A3_2D16_8ED5, xf * MASK_FREQ, zf * MASK_FREQ);
    let hilliness = clamp01(0.5 + mask * 1.4);
    let rolling = fbm(seed, xf * HILL_FREQ, zf * HILL_FREQ, OCTAVES);
    let dev = (HILL_AMP * hilliness * rolling).clamp(-MAX_DEV, MAX_DEV);
    (BASE_HEIGHT + (dev + 0.5).floor()) as i64
}

/// The block at world height `y` in a column whose surface sits at `h`.
/// Public so the surface-stage rewrite in `terrain.rs` can argue equivalence
/// against it.
#[inline]
pub fn column_block(y: i64, h: i64, blocks: &Blocks) -> BlockStateId {
    if y > h {
        BlockStateId::AIR
    } else if h <= SAND_LEVEL {
        // Beach and seabed columns: sand takes the grass and dirt layers.
        if y >= h - 3 {
            blocks.sand
        } else {
            blocks.stone
        }
    } else if y == h {
        blocks.grass_block
    } else if y >= h - 3 {
        blocks.dirt
    } else {
        blocks.stone
    }
}

/// How many `(cx, cz)` heightmaps the generator may cache. Bounded, so memory
/// stays flat whatever order requests arrive in.
const HEIGHT_CACHE_CAPACITY: usize = 64;

/// The 32² surface heights of one cube column, indexed `z * 32 + x` to match
/// the cube-local column of the same `(x, z)`.
type Heightmap = Arc<[i64; (CUBE_SIZE * CUBE_SIZE) as usize]>;

/// Generates terrain cubes. Output is a pure function of `(seed, pos)`,
/// whatever order cubes are requested in and however often; the heightmap
/// cache only ever stores results of that same pure function, so it cannot
/// change any answer.
pub struct ReferenceGenerator {
    seed: u64,
    blocks: Blocks,
    height_cache: Mutex<HashMap<(i32, i32), Heightmap>>,
}

impl ReferenceGenerator {
    pub fn new(seed: u64, blocks: Blocks) -> Self {
        Self {
            seed,
            blocks,
            height_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Surface height (block y of the grass/sand layer) at world column
    /// `(x, z)`. Always within [`crate::terrain::MIN_HEIGHT`, `crate::terrain::MAX_HEIGHT`].
    pub fn surface_height(&self, x: i64, z: i64) -> i64 {
        height(self.seed, x, z)
    }

    /// The spawn position over column `(0, 0)`: centered on the block, two
    /// above the surface.
    pub fn spawn(&self) -> [f64; 3] {
        [0.5, (self.surface_height(0, 0) + 2) as f64, 0.5]
    }

    /// The heights of the 32² columns covered by cube column `(cx, cz)`,
    /// computed at most once thanks to the bounded cache. Consecutive cubes
    /// of one column (streaming works bottom-up-ish per column) hit the
    /// cache.
    fn heightmap(&self, cx: i32, cz: i32) -> Heightmap {
        if let Some(map) = self.height_cache.lock().unwrap().get(&(cx, cz)) {
            return Arc::clone(map);
        }
        let mut map: Box<[i64; (CUBE_SIZE * CUBE_SIZE) as usize]> =
            Box::new([0; (CUBE_SIZE * CUBE_SIZE) as usize]);
        let base_x = i64::from(cx) * i64::from(CUBE_SIZE);
        let base_z = i64::from(cz) * i64::from(CUBE_SIZE);
        for z in 0..CUBE_SIZE {
            for x in 0..CUBE_SIZE {
                map[(z * CUBE_SIZE + x) as usize] =
                    height(self.seed, base_x + i64::from(x), base_z + i64::from(z));
            }
        }
        let map: Heightmap = Arc::from(map);
        let mut cache = self.height_cache.lock().unwrap();
        if cache.len() >= HEIGHT_CACHE_CAPACITY {
            // Stay bounded: drop the whole (small) working set rather than
            // growing without limit; streaming asks by column, so it refills.
            cache.clear();
        }
        cache.insert((cx, cz), Arc::clone(&map));
        map
    }

    /// The terrain cube at `pos`. Fast paths: a cube whose bottom is above
    /// the column-max height of its 32² area is uniform air, one whose top is
    /// below `min height - 4` is uniform stone — no per-block work at all.
    /// Mixed cubes are built in a single [`Cube::from_fn`] pass, so there is
    /// no per-block `set` palette churn.
    pub fn generate(&self, pos: CubePos) -> Cube {
        let heights = self.heightmap(pos.x, pos.z);
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        for &h in heights.iter() {
            if h < min {
                min = h;
            }
            if h > max {
                max = h;
            }
        }
        let bottom = i64::from(pos.y) * i64::from(CUBE_SIZE);
        let top = bottom + i64::from(CUBE_SIZE) - 1;
        if bottom > max {
            return Cube::default();
        }
        if top < min - 4 {
            return Cube::filled(self.blocks.stone);
        }
        let blocks = self.blocks;
        Cube::from_fn(move |local: LocalPos| {
            let h = heights[usize::from(local.z) * CUBE_SIZE as usize + usize::from(local.x)];
            column_block(bottom + i64::from(local.y), h, &blocks)
        })
    }
}
