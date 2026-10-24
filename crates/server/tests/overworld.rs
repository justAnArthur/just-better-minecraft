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
            lod.0.get(local.unwrap())
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

/// Task 2.14 (restating 2.10 E): the per-level accuracy of the coarse
/// surface. For a seeded sample of 512 land columns, per level 1..=5, the
/// coarse top — the topmost filled cell over the column plus its column
/// top height — minus the **highest true surface over the column's
/// footprint** (the `2ⁿ × 2ⁿ` columns the coarse column covers):
/// - real data (the pyramid's downsample of generated LOD0 cubes): exactly
///   0 — the coarse top is the highest filled block in the footprint;
/// - worldgen-sampled data (`sample_lod`): within a block for ≥ 95 % of
///   columns at levels 1..=4, mean in [−1, 1] at every level. The sample
///   grid can miss a peak between samples: at level 5 the samples are 16
///   blocks apart and the footprint envelope (see `sample_lod`) only
///   interpolates between them, so ~10 % of hill columns miss their peak
///   by more than a block there — level 5 is held to ≥ 85 %.
///
/// The coarse top minus the footprint's *mean* true height is printed too.
#[test]
fn lod_surface_accuracy_per_level() {
    use jbm_core::cube::Cube;
    use jbm_core::lod::{LodTops, downsample_octant, octant_offset};
    use std::collections::HashMap;
    use std::sync::Arc;

    let terrain = overworld(DEFAULT_SEED);
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

    // True surfaces per LOD0 cube column: every footprint up to level 5
    // (32 × 32 blocks) lies inside its column's LOD0 cube column.
    let mut surfaces = HashMap::<(i64, i64), Vec<i64>>::new();
    for &(x, z, _) in &columns {
        let key = (x.div_euclid(32), z.div_euclid(32));
        surfaces.entry(key).or_insert_with(|| {
            let mut heights = Vec::with_capacity(32 * 32);
            for dz in 0..32 {
                for dx in 0..32 {
                    heights.push(terrain.surface_height(key.0 * 32 + dx, key.1 * 32 + dz));
                }
            }
            heights
        });
    }

    // Real data: the exact pyramid over one LOD0 cube column. A level-n
    // cube's cells and tops over a column depend only on the LOD0 cubes
    // under it (locality), so the octants off that column stay air.
    type Lod = Arc<(Cube, LodTops)>;
    type RealMemo = HashMap<(u8, CubePos, (i32, i32)), Lod>;
    fn real(
        terrain: &Generator,
        memo: &mut RealMemo,
        level: u8,
        pos: CubePos,
        column: (i32, i32),
    ) -> Lod {
        if let Some(lod) = memo.get(&(level, pos, column)) {
            return Arc::clone(lod);
        }
        let lod = if level == 0 {
            Arc::new((terrain.generate(pos), LodTops::LOD0))
        } else {
            let mut cube = Cube::filled(BlockStateId::AIR);
            let mut tops = LodTops::empty(level);
            for octant in 0..8u8 {
                let [dx, dy, dz] = octant_offset(octant).map(i32::from);
                let child = CubePos {
                    x: pos.x * 2 + dx,
                    y: pos.y * 2 + dy,
                    z: pos.z * 2 + dz,
                };
                let shift = level as u32 - 1;
                if column.0 >> shift != child.x || column.1 >> shift != child.z {
                    continue;
                }
                let child_lod = real(terrain, memo, level - 1, child, column);
                downsample_octant(&child_lod.0, &child_lod.1, &mut cube, &mut tops, octant);
            }
            Arc::new((cube, tops))
        };
        memo.insert((level, pos, column), Arc::clone(&lod));
        lod
    }
    // The coarse top (block y of the highest filled block) over world
    // column (x, z) at `level`, scanning the cubes from `above` down to
    // `below`.
    fn coarse_top(
        mut cube_at: impl FnMut(CubePos) -> Lod,
        level: u8,
        (x, z): (i64, i64),
        above: i64,
        below: i64,
    ) -> Option<i64> {
        let cell = 1i64 << level;
        let side = 32 * cell;
        let (cx, cz) = (x.div_euclid(side), z.div_euclid(side));
        let local_x = ((x - cx * side) / cell) as u8;
        let local_z = ((z - cz * side) / cell) as u8;
        for cube_y in (below.div_euclid(side) - 1..=above.div_euclid(side) + 1).rev() {
            let lod = cube_at(CubePos {
                x: cx as i32,
                y: cube_y as i32,
                z: cz as i32,
            });
            for local_y in (0..32u8).rev() {
                if lod
                    .0
                    .get(LocalPos::new_unchecked(local_x, local_y, local_z))
                    != BlockStateId::AIR
                {
                    let h = i64::from(lod.1.get(local_x, local_z));
                    return Some(cube_y * side + i64::from(local_y) * cell + h - 1);
                }
            }
        }
        None
    }

    let mut sampled_memo = HashMap::<(u8, CubePos), Lod>::new();
    for level in 1..=5u8 {
        let cell = 1i64 << level;
        let mut real_memo = RealMemo::new();
        let mut real_errors = Vec::with_capacity(columns.len());
        let mut sampled_errors = Vec::with_capacity(columns.len());
        let mut sampled_vs_mean = 0.0f64;
        let mut real_vs_mean = 0.0f64;
        for &(x, z, h) in &columns {
            let heights = &surfaces[&(x.div_euclid(32), z.div_euclid(32))];
            let (fx, fz) = (
                x.rem_euclid(32) & !(cell - 1),
                z.rem_euclid(32) & !(cell - 1),
            );
            let mut max = i64::MIN;
            let mut sum = 0i64;
            for dz in 0..cell {
                for dx in 0..cell {
                    let s = heights[((fz + dz) * 32 + fx + dx) as usize];
                    max = max.max(s);
                    sum += s;
                }
            }
            let mean = sum as f64 / (cell * cell) as f64;
            let column = (x.div_euclid(32) as i32, z.div_euclid(32) as i32);
            let real_top = coarse_top(
                |pos| real(&terrain, &mut real_memo, level, pos, column),
                level,
                (x, z),
                max,
                h,
            )
            .expect("a land column has a real coarse top");
            let sampled_top = coarse_top(
                |pos| {
                    Arc::clone(
                        sampled_memo
                            .entry((level, pos))
                            .or_insert_with(|| Arc::new(terrain.sample_lod(level, pos))),
                    )
                },
                level,
                (x, z),
                max,
                h,
            )
            .expect("a land column has a sampled coarse top");
            real_errors.push(real_top - max);
            sampled_errors.push(sampled_top - max);
            real_vs_mean += real_top as f64 - mean;
            sampled_vs_mean += sampled_top as f64 - mean;
        }
        let n = columns.len() as f64;
        sampled_errors.sort_unstable();
        let sampled_mean = sampled_errors.iter().sum::<i64>() as f64 / n;
        let within_one = sampled_errors.iter().filter(|e| e.abs() <= 1).count() as f64 / n;
        let worst_real = real_errors.iter().copied().max_by_key(|e| e.abs()).unwrap();
        println!(
            "lod accuracy L{level} (vs footprint max): real: worst {worst_real:+}; \
             sampled: n {} mean {sampled_mean:+.2} |err|<=1 {:.1}% p1 {:+} p99 {:+} \
             min {:+} max {:+}; vs footprint mean: real {:+.2} sampled {:+.2}",
            sampled_errors.len(),
            within_one * 100.0,
            sampled_errors[sampled_errors.len() / 100],
            sampled_errors[sampled_errors.len() * 99 / 100],
            sampled_errors[0],
            sampled_errors[sampled_errors.len() - 1],
            real_vs_mean / n,
            sampled_vs_mean / n,
        );
        assert_eq!(
            worst_real, 0,
            "L{level}: the real-data coarse top is not the footprint's highest block"
        );
        let bar = if level <= 4 { 0.95 } else { 0.85 };
        assert!(
            within_one >= bar,
            "L{level}: only {:.1}% of sampled columns within a block",
            within_one * 100.0
        );
        assert!(
            (-1.0..=1.0).contains(&sampled_mean),
            "L{level}: sampled mean error {sampled_mean:+.2} outside [-1, 1]"
        );
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
