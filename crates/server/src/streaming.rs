//! Pure helpers for cube streaming: the desired offset list and the
//! desired/keep set predicates. The tick-thread driving logic lives in
//! [`crate::server`] phase 5; keeping the math here makes it directly
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

/// One entry of the merged LOD ring walk (task 2.6): a level and the
/// offset from that level's center cube.
pub(crate) type LodWalkEntry = (u8, Offset);

/// The merged walk over `(level, offset)` for LOD ring streaming with
/// plain vertical windows — the pre-2.10 shape, kept as the reference the
/// surface-following walk (below) extends and the streaming tests pin.
#[cfg(test)]
pub(crate) fn lod_walk(lod_levels: u8, r: u32, v: u32) -> Vec<LodWalkEntry> {
    let mut walk = Vec::new();
    for level in 1..=lod_levels {
        for offset in desired_offsets(r, v) {
            walk.push((level, offset));
        }
    }
    walk.sort_by_key(|&(level, (x, y, z))| {
        let d =
            i64::from(x) * i64::from(x) + i64::from(y) * i64::from(y) + i64::from(z) * i64::from(z);
        (d, std::cmp::Reverse(level), x, y, z)
    });
    walk
}

/// The vertical cube indices a column's surface spans, from its surface
/// height bounds (task 2.10): `[floor(min / side) - 1, floor(max / side)]`
/// — widened by one cube below so the surface cube's −Y neighbour is
/// there for meshing and skirts. `side` is the cube side in blocks at
/// the level in question (`CUBE_SIZE · 2^level`).
pub(crate) fn span_cubes(span: (i64, i64), side: i64) -> (i64, i64) {
    (span.0.div_euclid(side) - 1, span.1.div_euclid(side))
}

/// The `dy` offsets a column adds to the desired set beyond the camera
/// window (task 2.10): the span cubes outside `[center_y - v, center_y +
/// v]`, at most `cap` of them, nearest to the camera's `dy` first (ties
/// to the lower cube — deterministic). Returned in that nearest-first
/// order. `None` means the column's span is not known yet: window only.
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

/// Is `dy` one of the offsets [`column_extra_dys`] would keep (task
/// 2.10)? The keep-set form of the question, for unload and eviction
/// checks: window `v`, span extras trimmed to `cap`, both one looser
/// than the desired form (the same +1 hysteresis the radii get).
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

/// The desired offsets with surface-following windows (task 2.10): every
/// camera-window offset of [`desired_offsets`] plus, per column of the
/// disk, the surface-span extras `span_of` supplies (the column's surface
/// height bounds, `None` while that column's span has not been computed
/// yet — window only until then), each column trimmed to `cap` extras
/// nearest the camera. The union is sorted exactly like the window-only
/// list: nearest-first by `dx² + dy² + dz²`, ties by `(x, y, z)`.
pub(crate) fn desired_offsets_surface(
    r: u32,
    v: u32,
    cap: usize,
    span_of: impl Fn(i32, i32) -> Option<(i64, i64)>,
) -> Vec<Offset> {
    let side = i64::from(jbm_core::coords::CUBE_SIZE);
    let mut offsets = desired_offsets(r, v);
    for &(dx, dz) in &disk_columns(r) {
        let span = span_of(dx, dz);
        if span.is_none() {
            continue;
        }
        for dy in column_extra_dys(span, side, 0, v, cap) {
            offsets.push((dx, dy, dz));
        }
    }
    offsets.sort_by_key(|&(x, y, z)| {
        let d =
            i64::from(x) * i64::from(x) + i64::from(y) * i64::from(y) + i64::from(z) * i64::from(z);
        (d, x, y, z)
    });
    offsets
}

/// The merged LOD ring walk with surface-following windows (task 2.10).
/// Level 1 unions its camera window with the column's surface span
/// (`span1_of`, in level-1 cubes' columns; `None` while unknown). Levels
/// ≥ 2 keep the plain window: with the default `lod_vertical_radius` 3 a
/// level-*n* window spans `3 · CUBE_SIZE · 2^n ≥ 384` blocks from the
/// player's level cube, which covers the whole shipped relief
/// (`MIN_HEIGHT..=MAX_HEIGHT`) from any camera height — only level 1's
/// ±192 blocks can miss it. Sorted exactly like the window-only walk.
pub(crate) fn lod_walk_surface(
    lod_levels: u8,
    r: u32,
    v: u32,
    cap: usize,
    span1_of: impl Fn(i32, i32) -> Option<(i64, i64)>,
) -> Vec<LodWalkEntry> {
    let side = i64::from(jbm_core::coords::CUBE_SIZE) * 2;
    let mut walk = Vec::new();
    for level in 1..=lod_levels {
        for &(dx, dz) in &disk_columns(r) {
            for dy in -i64::from(v)..=i64::from(v) {
                walk.push((level, (dx, dy as i32, dz)));
            }
            if level == 1 {
                let span = span1_of(dx, dz);
                for dy in column_extra_dys(span, side, 0, v, cap) {
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
/// `div_euclid(2^(level-1))` per axis, so negative coordinates nest
/// exactly.
pub(crate) fn level_center(lod0: CubePos, level: u8) -> CubePos {
    debug_assert!(level >= 1);
    let shift = 1i32 << (level - 1);
    CubePos {
        x: lod0.x.div_euclid(shift),
        y: lod0.y.div_euclid(shift),
        z: lod0.z.div_euclid(shift),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        column_extra_dys, desired_offsets, desired_offsets_surface, extra_dy_keeps, in_desired,
        in_keep, lod_walk_surface, offset_pos, span_cubes,
    };
    use jbm_core::coords::{CUBE_SIZE, CubePos};

    const CENTER: CubePos = CubePos { x: -5, y: 2, z: 9 };

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
        let walk = super::lod_walk(3, 4, 2);
        // Full disks at every level.
        let disk = super::desired_offsets(4, 2).len();
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
        assert_eq!(level_center(lod0, 1), lod0);
        assert_eq!(level_center(lod0, 2), CubePos { x: -4, y: 1, z: -3 });
        assert_eq!(level_center(lod0, 3), CubePos { x: -2, y: 0, z: -2 });
        // Every level-n cube maps back onto its LOD0 children.
        for level in 1..=4u8 {
            let center = level_center(lod0, level);
            let back = 1i32 << (level - 1);
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

    /// Task 2.10: a column's span cubes are the surface height bounds in
    /// cube units, widened one cube below.
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

    /// Task 2.10: extras are the span cubes outside the window, trimmed
    /// to the cap, nearest the camera first; ties to the lower cube.
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

    /// Task 2.10: the keep form of the extras question, one looser in
    /// window and cap (the +1 hysteresis).
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

    /// Task 2.10: the surface-following desired list is the window list
    /// plus per-column extras, still nearest-first with deterministic
    /// ties, and a span inside the window changes nothing.
    #[test]
    fn desired_offsets_surface_adds_span_extras_sorted() {
        let (r, v, cap) = (2u32, 1u32, 16usize);
        let side = i64::from(CUBE_SIZE);
        // A cliff over one column only: heights spanning cubes 2..6.
        let cliff = (side * 2, side * 6);
        let span_of = |dx: i32, _dz: i32| {
            if dx == 1 { Some(cliff) } else { None }
        };
        let offsets = desired_offsets_surface(r, v, cap, span_of);
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
            desired_offsets_surface(r, v, cap, flat).len(),
            plain.len(),
            "a span inside the window must not change the desired set"
        );
    }

    /// Task 2.10: level 1 gets span extras, coarser levels do not, and
    /// the merged order stays (distance, coarse first, x, y, z).
    #[test]
    fn lod_walk_surface_extends_level_1_only() {
        let (r, v, cap) = (2u32, 1u32, 16usize);
        let side = i64::from(CUBE_SIZE);
        let span = (side, side * 13); // heights 32..416: level-1 cubes -1..6
        let span1 = |_: i32, _: i32| Some(span);
        let walk = lod_walk_surface(3, r, v, cap, span1);
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
            assert_eq!(count, super::desired_offsets(r, v).len());
        }
    }
}
