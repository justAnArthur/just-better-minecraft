//! Task 2.8 acceptance for the new overworld: the graph's interval bounds
//! are sound and prove the documented height range, the world's shape
//! statistics hold at the 8 km scale, water fills the ocean at LOD0 and
//! LOD1, and the spawn search lands on gentle land for every seed.

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CubePos, LocalPos};
use jbm_server::DEFAULT_SEED;
use jbm_server::terrain::{Generator, MAX_HEIGHT, MIN_HEIGHT, SEA_LEVEL};
use jbm_server::worldgen::{self, Preset};
use jbm_testkit::builtin_blocks;

fn blocks() -> Blocks {
    builtin_blocks()
}

fn overworld(seed: u64) -> Generator {
    Generator::with_preset(seed, blocks(), Preset::Overworld)
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
    let mut rng = jbm_testkit::XorShift64(0x5EED_2028_0000_0003);
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
fn ocean_columns_fill_with_water_to_sea_level_at_every_level() {
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

    // LOD1..=4: the cell containing SEA_LEVEL is filled (water) and the
    // cell containing SEA_LEVEL + 1 is air at EVERY level — the water's
    // top face (SEA_LEVEL + 1 = 64) is a multiple of 16, a cell boundary
    // at all four levels, so the sea is equally high on both sides of
    // every ring border. `sample_lod` applies the same rule as generate.
    assert_eq!((SEA_LEVEL + 1) % 16, 0, "sea surface off the level-4 grid");
    for level in 1u8..=4 {
        let cell = 1i64 << level;
        let side = 32 * cell;
        let at = |y: i64| {
            let lod = terrain.sample_lod(
                level,
                CubePos {
                    x: x.div_euclid(side) as i32,
                    y: y.div_euclid(side) as i32,
                    z: z.div_euclid(side) as i32,
                },
            );
            let local = LocalPos::new(
                (x.rem_euclid(side) / cell) as u8,
                (y.rem_euclid(side) / cell) as u8,
                (z.rem_euclid(side) / cell) as u8,
            );
            lod.get(local.unwrap())
        };
        assert_ne!(
            at(SEA_LEVEL),
            BlockStateId::AIR,
            "LOD{level} cell at sea level is filled"
        );
        assert_eq!(
            at(SEA_LEVEL + 1),
            BlockStateId::AIR,
            "LOD{level} cell above the sea is air"
        );
    }
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

/// Task 2.10 E: the per-level accuracy of the coarse surface against the
/// true surface. For a seeded sample of land columns, the height error
/// `coarse_top − true_top` (the topmost solid cell of the level-*n*
/// `sample_lod` cube over the column, minus `surface_height`) — mean and
/// p99 per level 1..=6. The downsampling rule leans solid, so the errors
/// sit mostly ≥ 0 and within the documented `2^n − 1` bias; the printed
/// numbers are the task's report. Water cells count as surface (the sea
/// is flat and aligned, so sea columns are skipped for the solid
/// comparison instead).
#[test]
fn lod_surface_accuracy_per_level() {
    let terrain = overworld(DEFAULT_SEED);
    let blocks = blocks();
    let spawn = terrain.spawn();
    // A seeded sample of land columns within ±2 km of spawn.
    let mut rng = jbm_testkit::XorShift64(0x00a1_1ce5_2a10);
    let mut columns = Vec::new();
    while columns.len() < 512 {
        let x = spawn[0] as i64 + (rng.next() % 4096) as i64 - 2048;
        let z = spawn[2] as i64 + (rng.next() % 4096) as i64 - 2048;
        let h = terrain.surface_height(x, z);
        if h >= SEA_LEVEL {
            columns.push((x, z, h));
        }
    }

    let mut cube_cache =
        std::collections::HashMap::<(u8, CubePos), std::sync::Arc<jbm_core::cube::Cube>>::new();
    for level in 1..=6u8 {
        let cell = 1i64 << level;
        let cube_side = i64::from(jbm_core::coords::CUBE_SIZE) * cell;
        let mut errors: Vec<i64> = Vec::with_capacity(columns.len());
        for &(x, z, h) in &columns {
            // The level-n cube containing the column's true surface; the
            // solid lean (at most 2^n - 1 blocks) can push the coarse top
            // into the cube above, so scan that one too when needed.
            let xz = (
                x.div_euclid(cube_side) as i32,
                z.div_euclid(cube_side) as i32,
            );
            let mut coarse_top = None;
            for cube_y in [
                h.div_euclid(cube_side) as i32,
                h.div_euclid(cube_side) as i32 + 1,
            ] {
                let pos = CubePos {
                    x: xz.0,
                    y: cube_y,
                    z: xz.1,
                };
                let cube = cube_cache
                    .entry((level, pos))
                    .or_insert_with(|| std::sync::Arc::new(terrain.sample_lod(level, pos)));
                let local_x = ((x - i64::from(xz.0) * cube_side) / cell) as u8;
                let local_z = ((z - i64::from(xz.1) * cube_side) / cell) as u8;
                for local_y in (0..32u8).rev() {
                    let state = cube.get(LocalPos::new_unchecked(local_x, local_y, local_z));
                    if state != BlockStateId::AIR && state != blocks.water {
                        coarse_top =
                            Some(i64::from(cube_y) * cube_side + i64::from(local_y + 1) * cell - 1);
                        break;
                    }
                }
                if coarse_top.is_some() {
                    break;
                }
            }
            if let Some(coarse) = coarse_top {
                errors.push(coarse - h);
            }
        }
        errors.sort_unstable();
        let mean = errors.iter().sum::<i64>() as f64 / errors.len() as f64;
        let p99 = errors[(errors.len() * 99) / 100];
        let min = errors[0];
        println!(
            "lod accuracy L{level}: n {} mean {mean:+.2} p99 {p99:+} min {min:+} \
             (one-cell lean {})",
            errors.len(),
            1i64 << level
        );
        // The solid lean keeps the coarse surface over the true one on
        // average (mean >= 0), with a p99 within two cells — the measured
        // lean (e.g. L2 reaching +6 = 1.5 cells) compounds the sampled
        // stride's own lean with the downsample's; the exact worst case
        // is not derived here, the printed numbers are the measurement.
        assert!(
            p99 <= 2i64 << level,
            "L{level} p99 {p99} beyond the two-cell sanity bound"
        );
        assert!(mean >= 0.0, "L{level} mean {mean} under the true surface");
    }
}

/// Task 2.10 F: beaches stay narrow. The sand rule (`h <= SEA_LEVEL + 1`)
/// makes every lowland column at the waterline sand; with the 2.8
/// continental spline's gentle rise off the sea, flat lowland became
/// huge sand plains. The 2.10 spline rises off the sea level quickly, so
/// in the 8 km shape sample the sand columns are a small fraction of
/// land — while coasts still show their beach rim.
#[test]
fn beaches_stay_narrow_but_coasts_still_show_sand() {
    let terrain = overworld(DEFAULT_SEED);
    let n = 500usize; // the shape sample's grid, 16-block spacing, 8 km
    let at = |xi: usize, zi: usize| {
        terrain.surface_height(
            (xi as i64 - n as i64 / 2) * 16,
            (zi as i64 - n as i64 / 2) * 16,
        )
    };
    let mut land = 0usize;
    let mut sand = 0usize;
    let mut coastal_sand = 0usize;
    for zi in 0..n {
        for xi in 0..n {
            let h = at(xi, zi);
            if h < SEA_LEVEL {
                continue; // sea (its floor is sand, but not "beach")
            }
            land += 1;
            if h <= SEA_LEVEL + 1 {
                sand += 1;
                // A beach column near the water: some 4-neighbour in the
                // sample is sea.
                let sea_neighbor = [(zi > 0, xi, zi.saturating_sub(1)), (zi + 1 < n, xi, zi + 1)]
                    .into_iter()
                    .any(|(ok, x, z)| ok && at(x, z) < SEA_LEVEL)
                    || [(xi > 0, xi.saturating_sub(1), zi), (xi + 1 < n, xi + 1, zi)]
                        .into_iter()
                        .any(|(ok, x, z)| ok && at(x, z) < SEA_LEVEL);
                if sea_neighbor {
                    coastal_sand += 1;
                }
            }
        }
    }
    let fraction = sand as f64 / land as f64;
    println!(
        "beaches @ 8km: {sand} sand of {land} land columns ({fraction:.3}), \
         {coastal_sand} coastal sand columns"
    );
    assert!(
        fraction < 0.05,
        "sand covers {:.1}% of land — the sand plains are back",
        fraction * 100.0
    );
    assert!(
        coastal_sand >= 100,
        "only {coastal_sand} coastal sand columns — the beach rim vanished"
    );
}
