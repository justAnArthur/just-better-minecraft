//! Task 2.8 acceptance for the new overworld: the graph's interval bounds
//! are sound and prove the documented height range, the world's shape
//! statistics hold at the 8 km scale, water fills the ocean at LOD0 and
//! LOD1, and the spawn search lands on gentle land for every seed.

use jbm_core::block::{BlockRegistry, BlockStateId, builtin::Blocks};
use jbm_core::coords::{CubePos, LocalPos};
use jbm_server::DEFAULT_SEED;
use jbm_server::terrain::{Generator, MAX_HEIGHT, MIN_HEIGHT, SEA_LEVEL};
use jbm_server::worldgen::{self, Preset};

fn blocks() -> Blocks {
    let mut registry = BlockRegistry::new();
    Blocks::register(&mut registry)
}

fn overworld(seed: u64) -> Generator {
    Generator::with_preset(seed, blocks(), Preset::Overworld)
}

/// Small local xorshift, so tests never need the `rand` crate.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// The block at world `(x, y, z)`, through the cube that contains it.
fn block_at(terrain: &Generator, x: i64, y: i64, z: i64) -> BlockStateId {
    let cube = terrain.generate(CubePos {
        x: (x >> 5) as i32,
        y: (y >> 5) as i32,
        z: (z >> 5) as i32,
    });
    cube.get(LocalPos::new((x & 31) as u8, (y & 31) as u8, (z & 31) as u8).unwrap())
}

#[test]
fn overworld_bounds_are_sound_over_100k_columns() {
    let compiled = worldgen::overworld();
    // 100,000 seeded single-block samples, |x|, |z| up to 2^30, each inside
    // its own one-block interval bound. y spans the band plus margin.
    let mut rng = XorShift64(0x5EED_2028_0000_0003);
    let mut worst_lo = f64::INFINITY;
    let mut worst_hi = f64::NEG_INFINITY;
    for _ in 0..100_000 {
        let at = [
            ((rng.next() & ((1 << 31) - 1)) as i64) - (1 << 30),
            (rng.next() % 512) as i64,
            ((rng.next() & ((1 << 31) - 1)) as i64) - (1 << 30),
        ];
        let density = compiled.sample(rng.next(), at);
        let (min, max) = compiled.bounds(at, at);
        assert!(
            density >= min && density <= max,
            "density {density} at {at:?} outside bounds [{min}, {max}]"
        );
        worst_lo = worst_lo.max(min);
        worst_hi = worst_hi.min(max);
    }
    // The proven height range, printed and cross-checked against the
    // documented constants and the band (which must exist and fit
    // MAX_BAND_ROWS = 512 rows for the heightmap fast paths to hold).
    let band = compiled.band().expect("the overworld graph is banded");
    println!(
        "overworld proven height range: band ({}, {}), {} rows; \
         documented [{MIN_HEIGHT}, {MAX_HEIGHT}]",
        band.low(),
        band.high(),
        band.rows()
    );
    assert_eq!((MIN_HEIGHT, MAX_HEIGHT), (band.low() + 1, band.high() - 1));
    assert!(band.rows() <= 512, "band too wide for the bit cache");
}

#[test]
fn overworld_shape_statistics_at_8km() {
    let terrain = overworld(DEFAULT_SEED);
    // A seeded 8 km × 8 km sample at 16-block spacing (500 × 500 columns).
    let n = 500usize;
    let mut heights = Vec::with_capacity(n * n);
    for zi in 0..n {
        for xi in 0..n {
            let x = (xi as i64 - n as i64 / 2) * 16;
            let z = (zi as i64 - n as i64 / 2) * 16;
            heights.push(terrain.surface_height(x, z));
        }
    }

    let ocean = heights.iter().filter(|&&h| h < SEA_LEVEL).count() as f64;
    let ocean_fraction = ocean / (n * n) as f64;
    let mut sorted = heights.clone();
    sorted.sort_unstable();
    let quantile = |q: f64| sorted[((q * (n * n - 1) as f64).round()) as usize];
    let p99 = quantile(0.99);
    let land: Vec<i64> = heights
        .iter()
        .copied()
        .filter(|&h| h >= SEA_LEVEL)
        .collect();
    let mut land_sorted = land.clone();
    land_sorted.sort_unstable();
    let land_median = land_sorted[land_sorted.len() / 2];

    // Connected mountain regions (8-connectivity) of columns at least
    // sea + 100 tall; the span is a region's bounding-box extent.
    let mountain = |h: i64| h >= SEA_LEVEL + 100;
    let mut visited = vec![false; n * n];
    let mut best_span = 0i64;
    let mut regions = 0usize;
    for start in 0..n * n {
        if visited[start] || !mountain(heights[start]) {
            continue;
        }
        regions += 1;
        let mut stack = vec![start];
        visited[start] = true;
        let (mut x0, mut x1, mut z0, mut z1) = (n, 0usize, n, 0usize);
        let mut size = 0usize;
        while let Some(at) = stack.pop() {
            let (x, z) = (at % n, at / n);
            size += 1;
            x0 = x0.min(x);
            x1 = x1.max(x);
            z0 = z0.min(z);
            z1 = z1.max(z);
            for dx in -1i64..=1 {
                for dz in -1i64..=1 {
                    let (nx, nz) = (x as i64 + dx, z as i64 + dz);
                    if (0..n as i64).contains(&nx)
                        && (0..n as i64).contains(&nz)
                        && !visited[nz as usize * n + nx as usize]
                        && mountain(heights[nz as usize * n + nx as usize])
                    {
                        visited[nz as usize * n + nx as usize] = true;
                        stack.push(nz as usize * n + nx as usize);
                    }
                }
            }
        }
        if size >= 4 {
            best_span = best_span.max(((x1 - x0 + 1).max(z1 - z0 + 1) as i64 - 1) * 16);
        }
    }

    println!(
        "overworld shape @ 8km, seed {DEFAULT_SEED:#x}: ocean fraction {ocean_fraction:.3}, \
         p50 {} p99 {p99} max {}, min {}; land median {land_median}; \
         {regions} mountain regions, best span {best_span} blocks",
        quantile(0.5),
        sorted[n * n - 1],
        sorted[0],
    );
    assert!(
        (0.20..=0.50).contains(&ocean_fraction),
        "ocean fraction {ocean_fraction:.3} outside [0.20, 0.50]"
    );
    assert!(
        p99 >= SEA_LEVEL + 200,
        "p99 height {p99} below sea level + 200"
    );
    assert!(
        (SEA_LEVEL + 5..=SEA_LEVEL + 60).contains(&land_median),
        "land median {land_median} outside [sea + 5, sea + 60]"
    );
    assert!(
        best_span >= 2000,
        "no mountain region spans 2 km (best {best_span})"
    );
    assert!(
        sorted[n * n - 1] <= MAX_HEIGHT && sorted[0] >= MIN_HEIGHT,
        "heights outside the documented range"
    );
}

#[test]
fn ocean_columns_fill_with_water_to_sea_level_at_lod0_and_lod1() {
    let terrain = overworld(DEFAULT_SEED);
    let water = blocks().water;
    // Find a genuine ocean column near the origin.
    let mut found = None;
    'search: for r in 0..2048i64 {
        for &(dx, dz) in &[(r, 0), (-r, 7), (r / 2, -r), (-r / 3, r)] {
            let h = terrain.surface_height(dx, dz);
            if h < SEA_LEVEL - 4 {
                found = Some((dx, dz, h));
                break 'search;
            }
        }
    }
    let (x, z, h) = found.expect("an ocean column within 2048 blocks");
    assert!(h < SEA_LEVEL, "column too shallow for the water belt");

    // LOD0: water from h + 1 to SEA_LEVEL, air above.
    assert_ne!(block_at(&terrain, x, h, z), BlockStateId::AIR);
    for y in (h + 1)..=SEA_LEVEL {
        assert_eq!(block_at(&terrain, x, y, z), water, "water at y = {y}");
    }
    for y in (SEA_LEVEL + 1)..=(SEA_LEVEL + 4) {
        assert_eq!(
            block_at(&terrain, x, y, z),
            BlockStateId::AIR,
            "air above the sea at y = {y}"
        );
    }

    // LOD1: the 2-block cell containing SEA_LEVEL is filled (water), the
    // one above is air — `sample_lod` applies the same rule.
    let lod = terrain.sample_lod(
        1,
        CubePos {
            x: (x >> 6) as i32,
            y: (SEA_LEVEL >> 6) as i32,
            z: (z >> 6) as i32,
        },
    );
    let local = |y: i64| {
        LocalPos::new(
            (x & 63) as u8 >> 1,
            (y & 63) as u8 >> 1,
            (z & 63) as u8 >> 1,
        )
    };
    assert_ne!(
        lod.get(local(SEA_LEVEL).unwrap()),
        BlockStateId::AIR,
        "LOD1 cell at sea level is filled"
    );
    assert_eq!(
        lod.get(local(SEA_LEVEL + 4).unwrap()),
        BlockStateId::AIR,
        "LOD1 cell above the sea is air"
    );
}

#[test]
fn spawn_lands_on_gentle_land_for_ten_seeds() {
    for seed in [
        DEFAULT_SEED,
        1,
        42,
        0x0BAD_C0DE,
        0xFFFF_FFFF_FFFF_FFFF,
        7,
        1234,
        999_999,
        0x5EED_5EED,
        77,
    ] {
        let terrain = overworld(seed);
        let [x, y, z] = terrain.spawn();
        let (xi, zi) = (x.floor() as i64, z.floor() as i64);
        let h = terrain.surface_height(xi, zi);
        assert!(h >= SEA_LEVEL, "seed {seed:#x}: spawn at sea floor");
        assert_eq!(y, (h + 2) as f64, "seed {seed:#x}: two above the surface");
        for (nx, nz) in [(xi - 1, zi), (xi + 1, zi), (xi, zi - 1), (xi, zi + 1)] {
            assert!(
                (terrain.surface_height(nx, nz) - h).abs() <= 2,
                "seed {seed:#x}: spawn slope steeper than the gentle bar"
            );
        }
        // Deterministic per seed.
        assert_eq!(
            terrain.spawn(),
            [x, y, z],
            "seed {seed:#x}: spawn not stable"
        );
    }
}
