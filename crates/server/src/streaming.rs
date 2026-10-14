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

#[cfg(test)]
mod tests {
    use super::{desired_offsets, in_desired, in_keep, offset_pos};
    use jbm_core::coords::CubePos;

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
}
