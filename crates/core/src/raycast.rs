//! Voxel raycast (Amanatides–Woo DDA): the shared targeting primitive
//! behind the crosshair. Pure logic over a `hit` closure, so any world
//! representation can drive it — the client passes its `ClientWorld`, the
//! tests pass closures.

use glam::DVec3;

use crate::coords::BlockPos;

/// How far a player can target and edit blocks.
pub const REACH: f64 = 5.0;

/// One of a block's six faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Face {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl Face {
    /// The face's outward unit normal.
    pub fn normal(self) -> [i8; 3] {
        match self {
            Face::PosX => [1, 0, 0],
            Face::NegX => [-1, 0, 0],
            Face::PosY => [0, 1, 0],
            Face::NegY => [0, -1, 0],
            Face::PosZ => [0, 0, 1],
            Face::NegZ => [0, 0, -1],
        }
    }
}

/// A ray hit: the block, the face the ray entered through, and the
/// Euclidean distance from the origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub block: BlockPos,
    pub face: Face,
    pub distance: f64,
}

/// The face a ray travelling along `dir` would have entered its start
/// voxel through: the one opposite the direction's dominant axis (ties
/// break to the lowest axis).
fn start_face(dir: DVec3) -> Face {
    let d = [dir.x, dir.y, dir.z];
    let abs = [d[0].abs(), d[1].abs(), d[2].abs()];
    let axis = if abs[0] >= abs[1] && abs[0] >= abs[2] {
        0
    } else if abs[1] >= abs[2] {
        1
    } else {
        2
    };
    crossed_face(axis, if d[axis] >= 0.0 { 1 } else { -1 })
}

/// The face entered when crossing `axis`'s boundary moving along `step`:
/// stepping +X enters the next voxel through its −X face, and so on.
fn crossed_face(axis: usize, step: i64) -> Face {
    match (axis, step == 1) {
        (0, true) => Face::NegX,
        (0, false) => Face::PosX,
        (1, true) => Face::NegY,
        (1, false) => Face::PosY,
        _ => {
            debug_assert!(axis == 2, "axis out of range");
            if step == 1 { Face::NegZ } else { Face::PosZ }
        }
    }
}

/// Traces the ray `origin + t · dir` through the block grid, one voxel at
/// a time (Amanatides–Woo), for at most `max_dist` Euclidean distance.
///
/// `hit` classifies each voxel: `Some(true)` targetable, `Some(false)`
/// pass-through, `None` not loaded. The ray stops at `None` and returns
/// `None` — never targeting through unloaded space. The targetable
/// predicate is the caller's; the client uses `StateInfo::solid` via
/// [`crate::physics::SolidTable`], so air and water pass through
/// (per-state outline shapes are a later task).
///
/// `dir` may be any nonzero finite vector (it is normalized internally, so
/// `distance` is Euclidean). Zero components never divide: their axis
/// crossing times are infinite. When the origin starts inside a targetable
/// block, the hit is that block at distance 0 with the face given by
/// [`start_face`]'s rule.
pub fn raycast(
    origin: DVec3,
    dir: DVec3,
    max_dist: f64,
    mut hit: impl FnMut(BlockPos) -> Option<bool>,
) -> Option<RayHit> {
    let length = dir.length();
    if length == 0.0 || !length.is_finite() || max_dist.is_nan() || max_dist < 0.0 {
        return None;
    }
    let dir = dir / length;
    let d = [dir.x, dir.y, dir.z];

    let mut voxel = [
        origin.x.floor() as i64,
        origin.y.floor() as i64,
        origin.z.floor() as i64,
    ];
    // The start voxel is queried first, so an origin inside a targetable
    // block hits immediately at distance 0.
    let start = BlockPos {
        x: voxel[0],
        y: voxel[1],
        z: voxel[2],
    };
    match hit(start) {
        Some(true) => {
            return Some(RayHit {
                block: start,
                face: start_face(dir),
                distance: 0.0,
            });
        }
        Some(false) => {}
        None => return None,
    }

    // Per-axis crossing state. A zero direction component leaves its
    // `t_max` at infinity: the ray never crosses that axis (1/0 = inf
    // without a division by zero ever being evaluated).
    let o = [origin.x, origin.y, origin.z];
    let mut step = [0i64; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    for axis in 0..3 {
        if d[axis] > 0.0 {
            step[axis] = 1;
            t_max[axis] = (voxel[axis] as f64 + 1.0 - o[axis]) / d[axis];
            t_delta[axis] = 1.0 / d[axis];
        } else if d[axis] < 0.0 {
            step[axis] = -1;
            t_max[axis] = (voxel[axis] as f64 - o[axis]) / d[axis];
            t_delta[axis] = 1.0 / -d[axis];
        }
    }

    loop {
        // The axis whose next boundary is nearest (ties to the lowest
        // axis, keeping the walk deterministic).
        let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            0
        } else if t_max[1] <= t_max[2] {
            1
        } else {
            2
        };
        let t = t_max[axis];
        if t > max_dist {
            return None;
        }
        voxel[axis] += step[axis];
        t_max[axis] += t_delta[axis];
        let pos = BlockPos {
            x: voxel[0],
            y: voxel[1],
            z: voxel[2],
        };
        match hit(pos) {
            Some(true) => {
                return Some(RayHit {
                    block: pos,
                    face: crossed_face(axis, step[axis]),
                    distance: t,
                });
            }
            Some(false) => {}
            None => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Face, REACH, raycast};
    use crate::coords::BlockPos;
    use glam::DVec3;

    fn pos(x: i64, y: i64, z: i64) -> BlockPos {
        BlockPos { x, y, z }
    }

    /// Every voxel loaded; only `solid` is targetable.
    fn world(solid: &[BlockPos]) -> impl Fn(BlockPos) -> Option<bool> + '_ {
        move |p| Some(solid.contains(&p))
    }

    /// Loaded air everywhere.
    fn air() -> impl Fn(BlockPos) -> Option<bool> {
        |_| Some(false)
    }

    #[test]
    fn axis_aligned_hits_on_each_face() {
        let cases = [
            // (origin, dir, target, expected face, expected distance)
            (
                DVec3::new(-1.5, 0.5, 0.5),
                DVec3::new(1.0, 0.0, 0.0),
                pos(2, 0, 0),
                Face::NegX,
                3.5,
            ),
            (
                DVec3::new(3.5, 0.5, 0.5),
                DVec3::new(-1.0, 0.0, 0.0),
                pos(0, 0, 0),
                Face::PosX,
                2.5,
            ),
            (
                DVec3::new(0.5, -2.5, 0.5),
                DVec3::new(0.0, 1.0, 0.0),
                pos(0, 2, 0),
                Face::NegY,
                4.5,
            ),
            (
                DVec3::new(0.5, 3.5, 0.5),
                DVec3::new(0.0, -1.0, 0.0),
                pos(0, 0, 0),
                Face::PosY,
                2.5,
            ),
            (
                DVec3::new(0.5, 0.5, -1.5),
                DVec3::new(0.0, 0.0, 1.0),
                pos(0, 0, 2),
                Face::NegZ,
                3.5,
            ),
            (
                DVec3::new(0.5, 0.5, 3.5),
                DVec3::new(0.0, 0.0, -1.0),
                pos(0, 0, 0),
                Face::PosZ,
                2.5,
            ),
        ];
        for (origin, dir, target, face, distance) in cases {
            let hit = raycast(origin, dir, REACH, world(&[target])).expect("must hit");
            assert_eq!(hit.block, target, "from {origin} along {dir}");
            assert_eq!(hit.face, face);
            assert_eq!(hit.distance, distance);
        }
    }

    #[test]
    fn diagonal_walks_the_expected_voxel_sequence() {
        let mut visited = Vec::new();
        let hit = |p: BlockPos| -> Option<bool> {
            visited.push(p);
            // Unloaded just past the 3x3x3 box, so the walk is bounded.
            if (0..=2).contains(&p.x) && (0..=2).contains(&p.y) && (0..=2).contains(&p.z) {
                Some(false)
            } else {
                None
            }
        };
        assert_eq!(
            raycast(
                DVec3::new(0.5, 0.5, 0.5),
                DVec3::new(1.0, 1.0, 1.0),
                10.0,
                hit
            ),
            None,
            "the ray stops at the unloaded voxel"
        );
        assert_eq!(
            visited,
            vec![
                pos(0, 0, 0),
                pos(1, 0, 0),
                pos(1, 1, 0),
                pos(1, 1, 1),
                pos(2, 1, 1),
                pos(2, 2, 1),
                pos(2, 2, 2),
                pos(3, 2, 2), // first unloaded voxel, where it stops
            ]
        );
    }

    #[test]
    fn a_ray_stops_at_an_unloaded_voxel() {
        let hit = |p: BlockPos| (p.x < 2).then_some(false);
        assert_eq!(
            raycast(
                DVec3::new(0.5, 0.5, 0.5),
                DVec3::new(1.0, 0.0, 0.0),
                10.0,
                hit
            ),
            None
        );
    }

    #[test]
    fn origin_exactly_on_a_grid_corner() {
        // Moving away from the corner: the start voxel is the (+,+,+) one.
        let mut visited = Vec::new();
        let hit = |p: BlockPos| -> Option<bool> {
            visited.push(p);
            Some(p == pos(2, 3, 4))
        };
        let hit_result = raycast(
            DVec3::new(1.0, 2.0, 3.0),
            DVec3::new(1.0, 1.0, 1.0),
            10.0,
            hit,
        )
        .expect("must hit");
        assert_eq!(hit_result.block, pos(2, 3, 4));
        assert_eq!(
            hit_result.face,
            Face::NegZ,
            "the z boundary is crossed last"
        );
        assert!((hit_result.distance - 3.0f64.sqrt()).abs() < 1e-12);
        assert_eq!(
            visited[..4],
            [pos(1, 2, 3), pos(2, 2, 3), pos(2, 3, 3), pos(2, 3, 4)][..]
        );

        // Moving into the corner: the first crossings are at distance 0,
        // one axis at a time, and a solid voxel there is hit immediately.
        let hit = |p: BlockPos| Some(p == pos(0, 2, 3));
        let hit_result = raycast(
            DVec3::new(1.0, 2.0, 3.0),
            DVec3::new(-1.0, -1.0, -1.0),
            10.0,
            hit,
        )
        .expect("must hit");
        assert_eq!(hit_result.block, pos(0, 2, 3));
        assert_eq!(hit_result.face, Face::PosX);
        assert_eq!(hit_result.distance, 0.0);
    }

    #[test]
    fn zero_direction_components_never_divide() {
        // Purely vertical: x and z stay put.
        let hit = raycast(
            DVec3::new(7.5, 0.5, -3.5),
            DVec3::new(0.0, 1.0, 0.0),
            10.0,
            world(&[pos(7, 4, -4)]),
        )
        .expect("must hit");
        assert_eq!(hit.block, pos(7, 4, -4));
        assert_eq!(hit.face, Face::NegY);
        assert_eq!(hit.distance, 3.5);

        // One zero component on a diagonal.
        let hit = raycast(
            DVec3::new(0.5, 0.5, 0.5),
            DVec3::new(1.0, 0.0, 1.0),
            10.0,
            world(&[pos(2, 0, 2)]),
        )
        .expect("must hit");
        assert_eq!(hit.block, pos(2, 0, 2));
        assert_eq!(hit.face, Face::NegZ, "the z boundary is crossed last");

        // The zero vector has no direction at all.
        assert_eq!(
            raycast(DVec3::new(0.5, 0.5, 0.5), DVec3::ZERO, 10.0, air()),
            None
        );
    }

    #[test]
    fn negative_and_far_from_the_origin() {
        for base in [-100_i64, -10_000_000, 10_000_000] {
            let (o, target) = (base as f64 + 0.5, pos(base - 4, 20, -51));
            let hit = raycast(
                DVec3::new(o, 20.5, -50.5),
                DVec3::new(-1.0, 0.0, 0.0),
                10.0,
                world(&[target]),
            )
            .expect("must hit");
            assert_eq!(hit.block, target);
            assert_eq!(hit.face, Face::PosX);
            assert_eq!(hit.distance, 3.5, "at base {base}");
        }
    }

    #[test]
    fn max_dist_is_respected_including_an_exact_hit() {
        let origin = DVec3::new(0.5, 0.5, 0.5);
        let dir = DVec3::new(0.0, 0.0, -1.0);
        let target = pos(0, 0, -3); // entered through its near face, z = -2.0
        assert!(raycast(origin, dir, 2.5, world(&[target])).is_some());
        assert_eq!(raycast(origin, dir, 2.5 - 1e-9, world(&[target])), None);
        // The start voxel counts even with no budget to walk.
        let here = pos(0, 0, 0);
        assert_eq!(
            raycast(origin, dir, 0.0, world(&[here])).map(|h| h.distance),
            Some(0.0)
        );
    }

    #[test]
    fn origin_inside_a_solid_block_hits_at_distance_zero() {
        let origin = DVec3::new(5.5, 5.5, 5.5);
        let solid = [pos(5, 5, 5)];
        for (dir, face) in [
            (DVec3::new(1.0, 0.2, 0.1), Face::NegX),
            (DVec3::new(0.1, -1.0, 0.3), Face::PosY),
            (DVec3::new(-0.4, 0.1, -0.9), Face::PosZ),
        ] {
            let hit = raycast(origin, dir, REACH, world(&solid)).expect("must hit");
            assert_eq!(hit.block, pos(5, 5, 5));
            assert_eq!(hit.face, face, "opposite the dominant axis of {dir}");
            assert_eq!(hit.distance, 0.0);
        }
    }

    #[test]
    fn non_unit_directions_measure_euclidean_distance() {
        let hit = raycast(
            DVec3::new(0.5, 0.5, 0.5),
            DVec3::new(5.0, 0.0, 0.0),
            3.5,
            world(&[pos(4, 0, 0)]),
        )
        .expect("must hit");
        assert_eq!(hit.block, pos(4, 0, 0));
        assert_eq!(hit.distance, 3.5);
    }
}
