//! Pure helpers for cube streaming: the desired offset list and the
//! desired/keep set predicates. The tick-thread driving logic lives in
//! [`crate::server::stream`]; keeping the math here makes it directly
//! testable.

use jbm_core::coords::CubePos;

/// One cube offset from a player's center cube.
pub(crate) type Offset = (i32, i32, i32);

/// The position at `offset` from `center`.
pub(crate) fn offset_pos(center: CubePos, offset: Offset) -> CubePos {
    CubePos {
        x: center.x + offset.0,
        y: center.y + offset.1,
        z: center.z + offset.2,
    }
}

/// Every cube offset within horizontal radius `r` (dx² + dz² ≤ r²) and
/// vertical radius `v` (|dy| ≤ v) of a center cube, sorted nearest-first by
/// `dx² + dy² + dz²` with ties broken by `(x, y, z)` so the order is fully
/// deterministic. The list depends only on the config, so the server
/// precomputes it once.
pub(crate) fn desired_offsets(r: u32, v: u32) -> Vec<Offset> {
    let r2 = u64::from(r) * u64::from(r);
    let mut offsets = Vec::new();
    let ri = i64::from(r);
    let vi = i64::from(v);
    for dx in -ri..=ri {
        for dz in -ri..=ri {
            if dx * dx + dz * dz <= r2 as i64 {
                for dy in -vi..=vi {
                    offsets.push((dx as i32, dy as i32, dz as i32));
                }
            }
        }
    }
    offsets.sort_by_key(|&(x, y, z)| {
        let d =
            i64::from(x) * i64::from(x) + i64::from(y) * i64::from(y) + i64::from(z) * i64::from(z);
        (d, x, y, z)
    });
    offsets
}

/// Is `pos` inside the desired set of a player centered on `center`
/// (radii `r`, `v`)?
pub(crate) fn in_desired(pos: CubePos, center: CubePos, r: u32, v: u32) -> bool {
    let dx = i64::from(pos.x - center.x);
    let dz = i64::from(pos.z - center.z);
    let dy = i64::from(pos.y - center.y);
    let r2 = i64::from(r) * i64::from(r);
    dx * dx + dz * dz <= r2 && dy.abs() <= i64::from(v)
}

/// Is `pos` inside the keep set of a player centered on `center`? Same shape
/// as the desired set with radius + 1: the hysteresis band that stops a
/// player standing on a border from making cubes flap in and out.
pub(crate) fn in_keep(pos: CubePos, center: CubePos, r: u32, v: u32) -> bool {
    in_desired(pos, center, r + 1, v + 1)
}

/// One entry of the merged LOD ring walk: a level and the offset from that
/// level's center cube.
pub(crate) type LodWalkEntry = (u8, Offset);

/// The vertical cube indices a column's surface spans, from its surface
/// height bounds: `[floor(min / side) - 1, floor(max / side)]` — widened by
/// one cube below so the surface cube's −Y neighbour is there for meshing
/// and skirts. `side` is the cube side in blocks at the level in question
/// (`CUBE_SIZE · 2^level`).
pub(crate) fn span_cubes(span: (i64, i64), side: i64) -> (i64, i64) {
    (span.0.div_euclid(side) - 1, span.1.div_euclid(side))
}

/// The `dy` offsets a column adds to the desired set beyond the camera
/// window: the span cubes outside `[center_y - v, center_y + v]`, at most
/// `cap` of them, nearest to the camera's `dy` first (ties to the lower
/// cube — deterministic). Returned in that nearest-first order. `None`
/// means the column's span is not known yet: window only.
pub(crate) fn column_extra_dys(
    span: Option<(i64, i64)>,
    side: i64,
    center_y: i64,
    v: u32,
    cap: usize,
) -> Vec<i32> {
    let Some(span) = span else {
        return Vec::new();
    };
    let (lo, hi) = span_cubes(span, side);
    let window_lo = center_y - i64::from(v);
    let window_hi = center_y + i64::from(v);
    let mut extras: Vec<i32> = (lo..=hi)
        .filter(|&y| y < window_lo || y > window_hi)
        .map(|y| (y - center_y) as i32)
        .collect();
    if extras.len() > cap {
        extras.sort_by_key(|&dy| (i64::from(dy).abs(), i64::from(dy)));
        extras.truncate(cap);
    }
    extras
}

/// Is `dy` one of the offsets [`column_extra_dys`] would keep? The
/// keep-set form of the question, for unload and eviction checks: window
/// `v`, span extras trimmed to `cap`, both one looser than the desired
/// form (the same +1 hysteresis the radii get).
#[cfg(test)]
pub(crate) fn extra_dy_keeps(
    span: Option<(i64, i64)>,
    side: i64,
    center_y: i64,
    v: u32,
    cap: usize,
    dy: i32,
) -> bool {
    let ady = i64::from(dy).abs();
    if ady <= i64::from(v) {
        return true;
    }
    let Some(span) = span else {
        return false;
    };
    let (lo, hi) = span_cubes(span, side);
    let y = center_y + i64::from(dy);
    if y < lo || y > hi {
        return false;
    }
    // Within the span but outside the window: kept only while among the
    // `cap` nearest extras (recompute the cut exactly as the desired
    // builder trims, with the loosened window/cap).
    let extras = (lo..=hi)
        .filter(|&yy| (yy - center_y).abs() > i64::from(v))
        .map(|yy| yy - center_y);
    let mut sorted: Vec<i64> = extras.collect();
    sorted.sort_by_key(|&d| (d.abs(), d));
    match sorted.iter().position(|&d| d == i64::from(dy)) {
        Some(index) => index < cap,
        None => false,
    }
}

/// The horizontal offsets of a disk `dx² + dz² ≤ r²`, sorted nearest-first
/// by `(dx² + dz², dx, dz)` — the column order the surface-following
/// builders below walk.
fn disk_columns(r: u32) -> Vec<(i32, i32)> {
    let r2 = u64::from(r) * u64::from(r);
    let ri = i64::from(r);
    let mut columns = Vec::new();
    for dx in -ri..=ri {
        for dz in -ri..=ri {
            if dx * dx + dz * dz <= r2 as i64 {
                columns.push((dx as i32, dz as i32));
            }
        }
    }
    columns.sort_by_key(|&(x, z)| {
        (
            i64::from(x) * i64::from(x) + i64::from(z) * i64::from(z),
            x,
            z,
        )
    });
    columns
}

/// The `dy` offsets of a column's surface span — the whole span (widened
/// one cube below), trimmed to `cap` cubes nearest the camera's `dy`
/// (ties to the lower cube — deterministic). Task 2.13 D: beyond the
/// camera window radius this is ALL a far column desires.
pub(crate) fn column_span_dys(
    span: Option<(i64, i64)>,
    side: i64,
    center_y: i64,
    cap: usize,
) -> Vec<i32> {
    let Some(span) = span else {
        return Vec::new();
    };
    let (lo, hi) = span_cubes(span, side);
    let mut dys: Vec<i32> = (lo..=hi).map(|y| (y - center_y) as i32).collect();
    if dys.len() > cap {
        dys.sort_by_key(|&dy| (i64::from(dy).abs(), i64::from(dy)));
        dys.truncate(cap);
    }
    dys
}

/// The keep-set form of [`column_span_dys`] (task 2.13 D): is `dy` one of
/// the span cubes a far column would keep, at the loosened `cap` (the +1
/// hysteresis the radii get)?
pub(crate) fn span_dy_keeps(
    span: Option<(i64, i64)>,
    side: i64,
    center_y: i64,
    cap: usize,
    dy: i32,
) -> bool {
    let Some(span) = span else {
        return false;
    };
    let (lo, hi) = span_cubes(span, side);
    let y = center_y + i64::from(dy);
    if y < lo || y > hi {
        return false;
    }
    let dys = (lo..=hi).map(|yy| yy - center_y);
    let mut sorted: Vec<i64> = dys.collect();
    sorted.sort_by_key(|&d| (d.abs(), d));
    match sorted.iter().position(|&d| d == i64::from(dy)) {
        Some(index) => index < cap,
        None => false,
    }
}

/// The desired offsets with surface-following windows and the task 2.13 D
/// camera window radius: within `window_r` cubes horizontally a column
/// desires the full camera window (`dy` in `±v`) plus its surface-span
/// extras beyond the window (the 2.10 rule, trimmed to `cap` nearest);
/// beyond `window_r` — far columns, whose depth band at the player's
/// level no cave-less world can show — a column desires ONLY its surface
/// span (the same widened-below span, `column_span_dys`, trimmed to
/// `cap`), and nothing until the span is known. `window_r >= r` is the
/// pre-2.13 shape. The union is sorted exactly like the window-only
/// list: nearest-first by `dx² + dy² + dz²`, ties by `(x, y, z)`.
pub(crate) fn desired_offsets_surface(
    r: u32,
    v: u32,
    cap: usize,
    center_y: i64,
    window_r: u32,
    span_of: impl Fn(i32, i32) -> Option<(i64, i64)>,
) -> Vec<Offset> {
    let side = i64::from(jbm_core::coords::CUBE_SIZE);
    let window_r2 = u64::from(window_r) * u64::from(window_r);
    let mut offsets = desired_offsets(window_r.min(r), v);
    for &(dx, dz) in &disk_columns(r) {
        let near =
            (i64::from(dx) * i64::from(dx) + i64::from(dz) * i64::from(dz)) as u64 <= window_r2;
        let span = span_of(dx, dz);
        if span.is_none() {
            continue;
        }
        if near {
            for dy in column_extra_dys(span, side, center_y, v, cap) {
                offsets.push((dx, dy, dz));
            }
        } else {
            for dy in column_span_dys(span, side, center_y, cap) {
                offsets.push((dx, dy, dz));
            }
        }
    }
    offsets.sort_by_key(|&(x, y, z)| {
        let d =
            i64::from(x) * i64::from(x) + i64::from(y) * i64::from(y) + i64::from(z) * i64::from(z);
        (d, x, y, z)
    });
    offsets
}

/// The merged LOD ring walk with surface-following windows. Level 1
/// unions its camera window with the column's surface span (`span1_of`,
/// in level-1 cubes' columns; `None` while unknown). `center1_y` is the
/// level-1 center cube's absolute y. Levels ≥ 2 keep the plain window:
/// with the default `lod_vertical_radius` 3 a level-*n* window spans
/// `3 · CUBE_SIZE · 2^n ≥ 384` blocks from the player's level cube, which
/// covers the whole shipped relief (`MIN_HEIGHT..=MAX_HEIGHT`) from any
/// camera height — only level 1's ±192 blocks can miss it. Sorted exactly
/// like the window-only walk.
pub(crate) fn lod_walk_surface(
    lod_levels: u8,
    r: u32,
    v: u32,
    cap: usize,
    center1_y: i64,
    window_r: u32,
    span1_of: impl Fn(i32, i32) -> Option<(i64, i64)>,
) -> Vec<LodWalkEntry> {
    let side = i64::from(jbm_core::coords::CUBE_SIZE) * 2;
    let window_r2 = u64::from(window_r) * u64::from(window_r);
    let mut walk = Vec::new();
    for level in 1..=lod_levels {
        for &(dx, dz) in &disk_columns(r) {
            // Task 2.13 D: the camera window applies only within the
            // camera window radius — level 1's far columns keep only
            // their sampled span (coarser levels have no spans; their
            // default windows already cover the whole shipped relief).
            let near = level > 1
                || (i64::from(dx) * i64::from(dx) + i64::from(dz) * i64::from(dz)) as u64
                    <= window_r2;
            if near {
                for dy in -i64::from(v)..=i64::from(v) {
                    walk.push((level, (dx, dy as i32, dz)));
                }
            }
            if level == 1 {
                let span = span1_of(dx, dz);
                let dys = if near {
                    column_extra_dys(span, side, center1_y, v, cap)
                } else {
                    column_span_dys(span, side, center1_y, cap)
                };
                for dy in dys {
                    walk.push((level, (dx, dy, dz)));
                }
            }
        }
    }
    walk.sort_by_key(|&(level, (x, y, z))| {
        let d =
            i64::from(x) * i64::from(x) + i64::from(y) * i64::from(y) + i64::from(z) * i64::from(z);
        (d, std::cmp::Reverse(level), x, y, z)
    });
    walk
}

/// The level-`level` cube (level >= 1) containing the LOD0 cube `lod0`:
/// `div_euclid(2^level)` per axis — `CubeKey::parent` applied `level`
/// times, each halving — so negative coordinates nest exactly. (Task
/// 2.13's far_metrics caught this dividing by only `2^(level-1)`: the
/// level-1 ring then streamed around the raw LOD0 center, which near the
/// origin still overlapped the parents the client's coverage rule needs
/// but anywhere farther left LOD0 coverage-dead with a coarse level
/// drawing right next to the player.)
pub(crate) fn level_center(lod0: CubePos, level: u8) -> CubePos {
    debug_assert!(level >= 1);
    let shift = 1i32 << level;
    CubePos {
        x: lod0.x.div_euclid(shift),
        y: lod0.y.div_euclid(shift),
        z: lod0.z.div_euclid(shift),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LodWalkEntry, column_extra_dys, desired_offsets, desired_offsets_surface, extra_dy_keeps,
        in_desired, in_keep, lod_walk_surface, offset_pos, span_cubes,
    };
    use jbm_core::coords::{CUBE_SIZE, CubePos};

    const CENTER: CubePos = CubePos { x: -5, y: 2, z: 9 };

    /// The merged walk over `(level, offset)` for LOD ring streaming with
    /// plain vertical windows — the reference shape the surface-following
    /// walk above extends and these tests pin.
    fn lod_walk(lod_levels: u8, r: u32, v: u32) -> Vec<LodWalkEntry> {
        let mut walk = Vec::new();
        for level in 1..=lod_levels {
            for offset in desired_offsets(r, v) {
                walk.push((level, offset));
            }
        }
        walk.sort_by_key(|&(level, (x, y, z))| {
            let d = i64::from(x) * i64::from(x)
                + i64::from(y) * i64::from(y)
                + i64::from(z) * i64::from(z);
            (d, std::cmp::Reverse(level), x, y, z)
        });
        walk
    }

    fn d2(o: (i32, i32, i32)) -> i64 {
        i64::from(o.0).pow(2) + i64::from(o.1).pow(2) + i64::from(o.2).pow(2)
    }

    #[test]
    fn offsets_are_sorted_nearest_first_with_deterministic_ties() {
        let offsets = desired_offsets(8, 4);
        assert_eq!(offsets.first(), Some(&(0, 0, 0)));
        let mut prev: Option<(i64, i32, i32, i32)> = None;
        for &(x, y, z) in &offsets {
            let key = (d2((x, y, z)), x, y, z);
            if let Some(prev) = prev {
                // Full keys are unique (offsets are), so strict ordering.
                assert!(key > prev, "order broke at {key:?} after {prev:?}");
            }
            prev = Some(key);
        }
    }

    #[test]
    fn offsets_are_unique_and_match_the_shape() {
        let (r, v) = (3u32, 2u32);
        let offsets = desired_offsets(r, v);
        let mut seen = std::collections::HashSet::new();
        for &o in &offsets {
            assert!(seen.insert(o), "duplicate offset {o:?}");
            assert!(in_desired(offset_pos(CENTER, o), CENTER, r, v));
        }
        // Every offset the shape admits is present: brute-force the box.
        let mut expected = 0usize;
        for dx in -3i64..=3 {
            for dz in -3i64..=3 {
                if dx * dx + dz * dz <= 9 {
                    expected += 5; // dy in -2..=2
                }
            }
        }
        assert_eq!(offsets.len(), expected);
    }

    #[test]
    fn lod_walk_orders_by_distance_then_coarse_first() {
        let walk = lod_walk(3, 4, 2);
        // Full disks at every level.
        let disk = desired_offsets(4, 2).len();
        assert_eq!(walk.len(), 3 * disk);
        // Fully ordered by (distance, level descending, x, y, z).
        let mut prev: Option<(i64, u8, i32, i32, i32)> = None;
        for &(level, (x, y, z)) in &walk {
            let d = i64::from(x).pow(2) + i64::from(y).pow(2) + i64::from(z).pow(2);
            let key = (d, level, x, y, z);
            if let Some(prev) = prev {
                // Level descending: encode by negating for the compare.
                let prev_ordered = (prev.0, std::cmp::Reverse(prev.1), prev.2, prev.3, prev.4);
                let key_ordered = (key.0, std::cmp::Reverse(key.1), key.2, key.3, key.4);
                assert!(
                    key_ordered > prev_ordered,
                    "order broke at {key:?} after {prev:?}"
                );
            }
            prev = Some(key);
        }
        // Equal distance, different levels: the coarser cube comes first.
        let index_of = |level: u8| {
            walk.iter()
                .position(|&(l, (x, y, z))| l == level && (x, y, z) == (2, 0, 0))
                .unwrap()
        };
        assert!(index_of(3) < index_of(2));
        assert!(index_of(2) < index_of(1));
    }

    #[test]
    fn level_centers_nest_at_negative_coordinates() {
        use super::level_center;
        let lod0 = CubePos { x: -7, y: 3, z: -5 };
        assert_eq!(level_center(lod0, 1), CubePos { x: -4, y: 1, z: -3 });
        assert_eq!(level_center(lod0, 2), CubePos { x: -2, y: 0, z: -2 });
        assert_eq!(level_center(lod0, 3), CubePos { x: -1, y: 0, z: -1 });
        // Every level-n cube maps back onto its LOD0 children.
        for level in 1..=4u8 {
            let center = level_center(lod0, level);
            let back = 1i32 << level;
            assert!(center.x * back <= lod0.x && lod0.x < (center.x + 1) * back);
            assert!(center.y * back <= lod0.y && lod0.y < (center.y + 1) * back);
            assert!(center.z * back <= lod0.z && lod0.z < (center.z + 1) * back);
        }
    }

    #[test]
    fn keep_set_is_one_larger_in_both_radii() {
        let (r, v) = (3u32, 2u32);
        // On the desired boundary: in desired, in keep.
        assert!(in_desired(offset_pos(CENTER, (3, 0, 0)), CENTER, r, v));
        assert!(in_keep(offset_pos(CENTER, (3, 0, 0)), CENTER, r, v));
        // One step out horizontally: not desired, still kept.
        assert!(!in_desired(offset_pos(CENTER, (4, 0, 0)), CENTER, r, v));
        assert!(in_keep(offset_pos(CENTER, (4, 0, 0)), CENTER, r, v));
        // On the keep boundary: still kept (≤ r² with r = 4).
        assert!(in_keep(offset_pos(CENTER, (4, 0, 0)), CENTER, r, v)); // 16 = 4²
        assert!(in_keep(offset_pos(CENTER, (3, 0, 2)), CENTER, r, v)); // 9+4 = 13 ≤ 16
        assert!(!in_keep(offset_pos(CENTER, (4, 0, 1)), CENTER, r, v)); // 16+1 > 16
        assert!(!in_keep(offset_pos(CENTER, (5, 0, 0)), CENTER, r, v)); // 25 > 16
        // Vertical hysteresis: dy = v+1 kept, dy = v+2 dropped.
        assert!(!in_desired(offset_pos(CENTER, (0, 3, 0)), CENTER, r, v));
        assert!(in_keep(offset_pos(CENTER, (0, 3, 0)), CENTER, r, v));
        assert!(!in_keep(offset_pos(CENTER, (0, 4, 0)), CENTER, r, v));
        // The center itself and negatives work the same.
        assert!(in_desired(CENTER, CENTER, r, v));
        assert!(in_keep(offset_pos(CENTER, (-3, -2, -2)), CENTER, r, v)); // 9+4 ≤ 16
        assert!(!in_keep(offset_pos(CENTER, (-3, -2, -3)), CENTER, r, v)); // 9+9 > 16
    }

    /// A column's span cubes are the surface height bounds in cube units,
    /// widened one cube below.
    #[test]
    fn span_cubes_floor_and_widen_below() {
        let side = i64::from(CUBE_SIZE);
        // Heights 100..300 at LOD0: cubes 3..9, widened to 2..9.
        assert_eq!(span_cubes((100, 300), side), (100 / side - 1, 300 / side));
        // Exactly on a cube boundary: the boundary cube is the top.
        assert_eq!(span_cubes((64, 64), side), (2 - 1, 2));
        // Negative heights (unbanded test graphs) floor toward -inf.
        assert_eq!(
            span_cubes((-5, 30), side),
            ((-5i64).div_euclid(side) - 1, 30i64.div_euclid(side))
        );
        assert!((-5i64).div_euclid(side) < 0);
    }

    /// Extras are the span cubes outside the window, trimmed to the cap,
    /// nearest the camera first; ties to the lower cube.
    #[test]
    fn column_extra_dys_trim_to_cap_nearest_first() {
        let side = i64::from(CUBE_SIZE);
        // Camera in a valley (cube 1, window dy -2..2 → cubes -1..3), a
        // cliff overhead spanning cubes 1..12: extras are cubes 4..12,
        // dys 3..11.
        let span = (side * 2, side * 12 + 5);
        assert_eq!(
            column_extra_dys(Some(span), side, 1, 2, 16),
            (3..=11).collect::<Vec<_>>()
        );
        // A flying camera (cube 8, window cubes 6..10) inside the same
        // cliff span: cubes 1..5 hang below (dy -7..-3) and the cubes
        // above the window (11, 12) poke out as dy 3, 4.
        assert_eq!(
            column_extra_dys(Some(span), side, 8, 2, 16),
            [-7, -6, -5, -4, -3, 3, 4]
        );
        // Cap 2 keeps the two nearest extras (dy 1, 2 here — above the
        // window, nearest the camera cube 1).
        assert_eq!(column_extra_dys(Some(span), side, 1, 0, 2), vec![1, 2]);
        // Tie at equal distance: the lower cube first.
        assert_eq!(column_extra_dys(Some(span), side, 6, 0, 2), vec![-1, 1]);
        // Unknown span: nothing beyond the window.
        assert!(column_extra_dys(None, side, 1, 2, 16).is_empty());
    }

    /// The keep form of the extras question, one looser in window and cap
    /// (the +1 hysteresis).
    #[test]
    fn extra_dy_keeps_is_desired_loosened_by_one() {
        let side = i64::from(CUBE_SIZE);
        let span = (side * 2, side * 12 + 5);
        // Camera cube 1, desired window v=0 cap=2: dy 1, 2 are the extras.
        // Keep (v=1, cap=3): dy -1..1 window-kept; dy 2, 3, 4 kept as the
        // three nearest extras; dy 5 not.
        for dy in [-1, 0, 1, 2, 3, 4] {
            assert!(
                extra_dy_keeps(Some(span), side, 1, 1, 3, dy),
                "dy {dy} should be kept"
            );
        }
        assert!(!extra_dy_keeps(Some(span), side, 1, 1, 3, 5));
        assert!(!extra_dy_keeps(Some(span), side, 1, 1, 3, -2));
        // No span: window only.
        assert!(extra_dy_keeps(None, side, 1, 1, 3, 1));
        assert!(!extra_dy_keeps(None, side, 1, 1, 3, 2));
    }

    /// The surface-following desired list is the window list plus
    /// per-column extras, still nearest-first with deterministic ties, and
    /// a span inside the window changes nothing.
    #[test]
    fn desired_offsets_surface_adds_span_extras_sorted() {
        let (r, v, cap) = (2u32, 1u32, 16usize);
        let side = i64::from(CUBE_SIZE);
        // A cliff over one column only: heights spanning cubes 2..6.
        let cliff = (side * 2, side * 6);
        let span_of = |dx: i32, _dz: i32| {
            if dx == 1 { Some(cliff) } else { None }
        };
        let offsets = desired_offsets_surface(r, v, cap, 0, r, span_of);
        // Sorted nearest-first with unique full keys.
        let d2 = |o: (i32, i32, i32)| {
            i64::from(o.0).pow(2) + i64::from(o.1).pow(2) + i64::from(o.2).pow(2)
        };
        let mut prev: Option<(i64, i32, i32, i32)> = None;
        for &(x, y, z) in &offsets {
            let key = (d2((x, y, z)), x, y, z);
            if let Some(prev) = prev {
                assert!(key > prev, "order broke at {key:?} after {prev:?}");
            }
            prev = Some(key);
        }
        // The cliff's extras are all there (window dy -1..1 around center
        // cube 0, span cubes 1..6 → extras dy 2..6).
        for dy in 2..=6i32 {
            assert!(
                offsets.contains(&(1, dy, 0)),
                "cliff extra (1, {dy}, 0) missing"
            );
        }
        // Columns without spans keep exactly the window list.
        let plain = desired_offsets(r, v);
        assert!(
            plain
                .iter()
                .all(|&o| o.0 != 1 || o.2 != 0 || offsets.contains(&o))
        );
        assert!(offsets.len() > plain.len());
        // A span inside the window adds nothing: flat land at cubes 0..1.
        let flat = |_: i32, _: i32| Some((0i64, side));
        assert_eq!(
            desired_offsets_surface(r, v, cap, 0, r, flat).len(),
            plain.len(),
            "a span inside the window must not change the desired set"
        );
        // The span's cube indices are absolute: a camera at cube 8 over
        // the same cliff (span cubes 1..6, window cubes 7..9) gets the
        // span as offsets dy -7..-2 — the flying-over-a-valley case that
        // first exposed a relative/absolute mix-up.
        let high = desired_offsets_surface(r, v, cap, 8, r, span_of);
        for dy in -7..=-2i32 {
            assert!(
                high.contains(&(1, dy, 0)),
                "camera cube 8: cliff extra (1, {dy}, 0) missing"
            );
        }
        assert!(!high.contains(&(1, 2, 0)), "no extras above the window");
        // The cliff spans the three disk columns with dx == 1 (dz -1..=1).
        assert_eq!(high.len(), plain.len() + 3 * 6);
    }

    /// Level 1 gets span extras, coarser levels do not, and the merged
    /// order stays (distance, coarse first, x, y, z).
    #[test]
    fn lod_walk_surface_extends_level_1_only() {
        let (r, v, cap) = (2u32, 1u32, 16usize);
        let side = i64::from(CUBE_SIZE);
        let span = (side, side * 13); // heights 32..416: level-1 cubes -1..6
        let span1 = |_: i32, _: i32| Some(span);
        let walk = lod_walk_surface(3, r, v, cap, 0, r, span1);
        // Fully ordered like the window-only walk.
        let mut prev: Option<(i64, std::cmp::Reverse<u8>, i32, i32, i32)> = None;
        for &(level, (x, y, z)) in &walk {
            let d = i64::from(x).pow(2) + i64::from(y).pow(2) + i64::from(z).pow(2);
            let key = (d, std::cmp::Reverse(level), x, y, z);
            if let Some(prev) = prev {
                assert!(key > prev, "order broke at {key:?} after {prev:?}");
            }
            prev = Some(key);
        }
        // Level 1 gained extras beyond dy -1..1: the span reaches
        // level-1 cube 6, so dys 2..6 are in (level-1 cubes are 64
        // blocks; heights 32..416 span cubes -1..6).
        let level1: std::collections::HashSet<(i32, i32, i32)> = walk
            .iter()
            .filter(|&&(l, _)| l == 1)
            .map(|&(_, o)| o)
            .collect();
        for dy in 2..=6i32 {
            assert!(
                level1.contains(&(0, dy, 0)),
                "level 1 extra dy {dy} missing"
            );
        }
        // Levels 2 and 3 keep exactly their window disks.
        for level in [2u8, 3] {
            let count = walk.iter().filter(|&&(l, _)| l == level).count();
            assert_eq!(count, desired_offsets(r, v).len());
        }
    }

    /// Task 2.13 D: beyond the camera window radius a column desires only
    /// its surface span (cap-trimmed, widened one cube below); within it
    /// the full camera window plus extras, exactly the pre-2.13 shape.
    #[test]
    fn beyond_the_camera_window_radius_only_span_cubes_are_desired() {
        let (r, v, cap) = (4u32, 3u32, 16usize);
        let side = i64::from(CUBE_SIZE);
        // Flat land whose surface sits at cube 9 (span cubes 8..9 —
        // widened one below) over the whole disk.
        let span = (side * 9, side * 9);
        let flat = |_: i32, _: i32| Some(span);
        let window_r = 2u32;
        let offsets = desired_offsets_surface(r, v, cap, 0, window_r, flat);

        // Within window_r: the full window (dy -v..=v), span inside it.
        for dy in -3i32..=3 {
            assert!(offsets.contains(&(1, dy, 0)), "near window dy {dy} missing");
        }
        // The near window columns are exactly the disk of window_r; check
        // a column on the window_r boundary circle is near and one just
        // outside (still inside r) is far.
        assert!(
            offsets.contains(&(2, 0, 0)),
            "the window_r boundary column is near"
        );
        assert!(
            !offsets.contains(&(3, 3, 0)),
            "a far column's window cube must not be desired"
        );
        assert!(!offsets.contains(&(3, -3, 0)));
        assert!(!offsets.contains(&(3, 3, 1)));
        // ... but the far column's span cubes are: dy 8 and 9 (the widened
        // span), trimmed to the cap of 16 — both fit.
        for dy in [8i32, 9] {
            assert!(offsets.contains(&(3, dy, 0)), "far span dy {dy} missing");
            assert!(
                offsets.contains(&(3, dy, 1)),
                "far span dy {dy} missing (dz 1)"
            );
        }
        // Nothing else of that far column.
        assert!(!offsets.contains(&(3, 0, 0)));
        assert!(!offsets.contains(&(3, 7, 0)));
        assert!(!offsets.contains(&(3, 10, 0)));

        // An unknown span desires nothing beyond the window columns: far
        // columns wait for their span.
        let none = |_: i32, _: i32| None;
        let window_only = desired_offsets_surface(r, v, cap, 0, window_r, none);
        let plain_window = desired_offsets(window_r, v);
        assert_eq!(window_only, plain_window);

        // The cap trims the far span to the nearest cubes: a tall cliff
        // (span cubes 2..30) at cap 4 keeps dys 8..11 around camera 0...
        // no — nearest four to 0 are 2, 3, 4, 5 (the span's own bottom).
        let cliff = (side * 2, side * 30);
        let cliffed = |dx: i32, _dz: i32| {
            if dx == 3 { Some(cliff) } else { None }
        };
        let offsets = desired_offsets_surface(r, v, 2, 0, window_r, cliffed);
        // The span widens one cube below: cubes 1..30, cap 2 keeps dys 1, 2.
        for dy in 1..=2i32 {
            assert!(
                offsets.contains(&(3, dy, 0)),
                "cap keeps the nearest span cubes"
            );
        }
        assert!(!offsets.contains(&(3, 3, 0)), "beyond the cap");
        assert!(!offsets.contains(&(3, 6, 0)), "beyond the cap");
    }

    /// Task 2.13 D: a player at any depth keeps the full camera window
    /// around them — the digging case. A camera 100 blocks down (cube -4
    /// at 32-block cubes) still desires dy -v..=v of the near columns.
    #[test]
    fn a_deep_camera_still_gets_the_full_window_near_it() {
        let (r, v, cap) = (8u32, 4u32, 16usize);
        let side = i64::from(CUBE_SIZE);
        let span = (side * 9, side * 9); // surface far above the digger
        let flat = |_: i32, _: i32| Some(span);
        // Camera at cube -4 (about 100 blocks under the surface).
        let offsets = desired_offsets_surface(r, v, cap, -4, 4, flat);
        for dy in -4i32..=4 {
            assert!(
                offsets.contains(&(0, dy, 0)),
                "the digging player's window dy {dy} missing"
            );
            assert!(
                offsets.contains(&(3, dy, 0)),
                "within window_r dy {dy} missing"
            );
        }
        // The far columns carry the surface span as dys relative to the
        // deep camera: 9 - (-4) = 13 (and 12, widened below).
        assert!(offsets.contains(&(5, 12, 0)));
        assert!(offsets.contains(&(5, 13, 0)));
        assert!(!offsets.contains(&(5, 0, 0)));
    }

    /// Task 2.13 D: the level-1 walk mirrors the rule — near columns keep
    /// the window, far columns only their span; coarser levels always do.
    #[test]
    fn lod_walk_far_columns_keep_only_their_span() {
        let (r, v, cap) = (4u32, 3u32, 16usize);
        let side = i64::from(CUBE_SIZE);
        // Level-1 span cubes 4..5 (heights 256..383-ish), side 64.
        let span = (side * 2 * 4, side * 2 * 4);
        let span1 = |_: i32, _: i32| Some(span);
        let walk = lod_walk_surface(2, r, v, cap, 0, 2, span1);
        let level1: std::collections::HashSet<(i32, i32, i32)> = walk
            .iter()
            .filter(|&&(l, _)| l == 1)
            .map(|&(_, o)| o)
            .collect();
        // Near column (0, 0): the full window.
        for dy in -3i32..=3 {
            assert!(level1.contains(&(0, dy, 0)), "near window dy {dy} missing");
        }
        // Far column (3, 0): only the span dys 3 and 4 (the level-1 side
        // is 64 blocks; heights 256..256 span cubes 3..4, widened below).
        assert!(level1.contains(&(3, 3, 0)));
        assert!(level1.contains(&(3, 4, 0)));
        assert!(!level1.contains(&(3, 0, 0)), "no far window at level 1");
        assert!(!level1.contains(&(3, -3, 0)));
        // Level 2 keeps its plain window everywhere.
        let count = walk.iter().filter(|&&(l, _)| l == 2).count();
        assert_eq!(count, desired_offsets(r, v).len());
    }
}
