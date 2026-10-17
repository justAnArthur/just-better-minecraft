//! Camera-relative space math shared by the CPU and GPU culling: the
//! camera split behind the one origin formula, the frustum, and the face
//! directions a cube can still show front faces in. The GPU mirrors of
//! these rules live in shaders/terrain.wgsl and shaders/cull.wgsl; both
//! sides must cull on the same numbers.

use glam::{DVec3, IVec3, Mat4, Vec3, Vec4};
use jbm_core::coords::CUBE_SIZE;

use crate::lod::CubeKey;

/// Camera-relative frustum for culling: five planes extracted from the
/// rotation-only view-projection (the far plane is at infinity with the
/// reverse-Z infinite-far projection). A point `p` is inside when
/// `plane . p >= 0` for every plane.
#[derive(Debug, Clone)]
pub struct Frustum {
    planes: [Vec4; 5],
}

impl Frustum {
    /// Extracts the left / right / bottom / top / near planes (Gribb &
    /// Hartmann row combinations) from a camera-relative view-projection.
    pub fn from_view_proj(view_proj: &Mat4) -> Frustum {
        let row = |r: usize| {
            Vec4::new(
                view_proj.x_axis[r],
                view_proj.y_axis[r],
                view_proj.z_axis[r],
                view_proj.w_axis[r],
            )
        };
        let row0 = row(0);
        let row1 = row(1);
        let row2 = row(2);
        let row3 = row(3);
        Frustum {
            planes: [
                row3 + row0, // left
                row3 - row0, // right
                row3 + row1, // bottom
                row3 - row1, // top
                row3 - row2, // near (reverse-Z: z' <= w')
            ],
        }
    }

    /// Whether any part of the axis-aligned box may be visible.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Support point farthest along the plane normal.
            let px = if plane.x >= 0.0 { max.x } else { min.x };
            let py = if plane.y >= 0.0 { max.y } else { min.y };
            let pz = if plane.z >= 0.0 { max.z } else { min.z };
            if plane.x * px + plane.y * py + plane.z * pz + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Which face directions of a cube with camera-relative AABB
/// `[rel_min, rel_max]` can contain front-facing faces. Order matches
/// `mesh::face::Direction`: +X, -X, +Y, -Y, +Z, -Z.
///
/// Per axis, one line each (the camera sits at the origin):
/// - `+X` is skipped when `rel_min.x >= 0`: every +X face plane is then at
///   `x >= rel_min.x >= 0 = camera.x`, and a +X plane is front-facing only
///   when the camera is strictly on its positive side — so all are back-
///   facing (or edge-on) and back-face culling would discard them anyway.
/// - `-X` is skipped when `rel_max.x <= 0`: every -X plane is at
///   `x <= rel_max.x <= 0`, so the camera is never on its negative side.
/// - Y and Z identically.
///
/// The image cannot change: the skipped faces were already discarded by the
/// pipeline's back-face culling.
pub fn visible_directions(rel_min: Vec3, rel_max: Vec3) -> [bool; 6] {
    [
        rel_min.x < 0.0,
        rel_max.x > 0.0,
        rel_min.y < 0.0,
        rel_max.y > 0.0,
        rel_min.z < 0.0,
        rel_max.z > 0.0,
    ]
}

/// The camera split behind the one origin formula: `camera_cube` is
/// `floor(camera / CUBE_SIZE)` per axis and `camera_frac` is
/// `camera − camera_cube · CUBE_SIZE`, computed in f64 and then rounded to
/// f32.
pub fn camera_split(camera: DVec3) -> (IVec3, Vec3) {
    let size = f64::from(CUBE_SIZE);
    let cube = IVec3::new(
        (camera.x / size).floor() as i32,
        (camera.y / size).floor() as i32,
        (camera.z / size).floor() as i32,
    );
    let frac = Vec3::new(
        (camera.x - f64::from(cube.x) * size) as f32,
        (camera.y - f64::from(cube.y) * size) as f32,
        (camera.z - f64::from(cube.z) * size) as f32,
    );
    (cube, frac)
}

/// The one origin formula, scaled by the cube's level and shared by the
/// CPU paths here, `vs_main` in shaders/terrain.wgsl and `cull.wgsl`:
/// `origin = (pos·2^level − camera_cube) · CUBE_SIZE − camera_frac`. A
/// level-*n* cube at `pos` covers blocks `pos * CUBE_SIZE * 2^n` onwards,
/// so the integer part stays exact (the product is computed in i64; for
/// any position a server can stream, `pos·2^n` is asserted to fit i32 —
/// coordinates beyond that are unreachable at f64 camera precision); the
/// only rounding is the final f32 subtract, and the shaders perform the
/// very same steps, so the CPU and GPU cull on the same numbers.
pub fn cube_origin(key: CubeKey, camera_cube: IVec3, camera_frac: Vec3) -> Vec3 {
    let scale = 1i64 << key.level;
    let int = Vec3::new(
        scaled_cube(key.pos.x, scale, camera_cube.x),
        scaled_cube(key.pos.y, scale, camera_cube.y),
        scaled_cube(key.pos.z, scale, camera_cube.z),
    );
    int * CUBE_SIZE as f32 - camera_frac
}

/// One axis of [`cube_origin`]: `(pos·2^level − camera_cube)` as f32,
/// asserted to be an i32-range integer for streamable positions.
fn scaled_cube(pos: i32, scale: i64, camera_cube: i32) -> f32 {
    let scaled = i64::from(pos) * scale;
    debug_assert!(
        scaled <= i64::from(i32::MAX) && scaled >= i64::from(i32::MIN),
        "pos·2^level overflows i32: {pos}·{scale}"
    );
    (scaled - i64::from(camera_cube)) as f32
}

/// The camera-relative side length of a level-`level` cube.
pub fn cube_size(level: u8) -> Vec3 {
    Vec3::splat(CUBE_SIZE as f32 * (1u32 << level) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;

    fn camera() -> Camera {
        Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0)
    }

    #[test]
    fn aabb_in_front_is_visible() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        // Directly ahead of the camera (yaw 0 looks toward -Z).
        assert!(frustum.intersects_aabb(Vec3::new(-4.0, -4.0, -36.0), Vec3::new(4.0, 4.0, -4.0)));
    }

    #[test]
    fn aabb_behind_the_camera_is_culled() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        assert!(!frustum.intersects_aabb(Vec3::new(-4.0, -4.0, 4.0), Vec3::new(4.0, 4.0, 36.0)));
    }

    #[test]
    fn aabb_far_to_the_side_is_culled() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        // Well outside the 70-degree fov on the +X side, ahead in z.
        assert!(
            !frustum.intersects_aabb(Vec3::new(200.0, -8.0, -60.0), Vec3::new(232.0, 8.0, -28.0))
        );
        // ... and the mirror on -X.
        assert!(!frustum.intersects_aabb(
            Vec3::new(-232.0, -8.0, -60.0),
            Vec3::new(-200.0, 8.0, -28.0)
        ));
    }

    #[test]
    fn aabb_straddling_the_near_plane_stays_visible() {
        let frustum = Frustum::from_view_proj(&camera().view_proj());
        // Crossing the near plane (and the camera itself) from well in
        // front to behind counts as visible (conservative).
        assert!(frustum.intersects_aabb(Vec3::new(-1.0, -1.0, -20.0), Vec3::new(1.0, 1.0, 20.0)));
    }

    #[test]
    fn cube_containing_the_camera_shows_all_six_directions() {
        assert!(
            visible_directions(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0))
                .iter()
                .all(|&visible| visible)
        );
        // A face plane running exactly through the camera is edge-on: the
        // cube touching the origin on a corner shows only the three sides
        // facing it.
        assert_eq!(
            visible_directions(Vec3::ZERO, Vec3::splat(1.0)),
            [false, true, false, true, false, true]
        );
        assert_eq!(
            visible_directions(Vec3::splat(-1.0), Vec3::ZERO),
            [true, false, true, false, true, false]
        );
    }

    #[test]
    fn cube_in_front_and_behind_the_camera() {
        // In front (yaw 0 looks toward -Z): -Z faces away.
        assert_eq!(
            visible_directions(Vec3::new(-1.0, -1.0, -5.0), Vec3::new(1.0, 1.0, -1.0)),
            [true, true, true, true, true, false]
        );
        // Behind: +Z faces away.
        assert_eq!(
            visible_directions(Vec3::new(-1.0, -1.0, 2.0), Vec3::new(1.0, 1.0, 6.0)),
            [true, true, true, true, false, true]
        );
    }

    #[test]
    fn cube_straddling_each_axis() {
        // X only: +X and -X both stay; the cube is above and in front, so
        // +Y and -Z face away.
        assert_eq!(
            visible_directions(Vec3::new(-1.0, 2.0, -5.0), Vec3::new(1.0, 4.0, -2.0)),
            [true, true, false, true, true, false]
        );
        // Y only.
        assert_eq!(
            visible_directions(Vec3::new(2.0, -1.0, -5.0), Vec3::new(4.0, 1.0, -2.0)),
            [false, true, true, true, true, false]
        );
        // Z only (cube behind on x).
        assert_eq!(
            visible_directions(Vec3::new(2.0, 2.0, -1.0), Vec3::new(4.0, 4.0, 1.0)),
            [false, true, false, true, true, true]
        );
        // Fully in the (+, +, +) octant: only the camera-facing sides show.
        assert_eq!(
            visible_directions(Vec3::new(2.0, 2.0, 2.0), Vec3::new(4.0, 4.0, 4.0)),
            [false, true, false, true, false, true]
        );
    }

    /// One f32 ulp (relative), with an absolute floor near zero.
    fn ulp(x: f32) -> f32 {
        f32::EPSILON * x.abs().max(1.0)
    }

    /// The origin formula against `Camera::relative` (f64 subtraction),
    /// within one ulp of the result, for one cube near the camera. The
    /// cube's level scales its world position by 2^level.
    fn check_origin(camera_pos: DVec3, k: CubeKey) {
        let camera = Camera::new(camera_pos, 70.0_f32.to_radians(), 1.0);
        let (cube, frac) = camera_split(camera_pos);
        let origin = cube_origin(k, cube, frac);
        let scale = f64::from(CUBE_SIZE) * f64::from(1u32 << k.level);
        let world_min = DVec3::new(
            f64::from(k.pos.x) * scale,
            f64::from(k.pos.y) * scale,
            f64::from(k.pos.z) * scale,
        );
        let reference = camera.relative(world_min);
        for i in 0..3 {
            assert!(
                (origin[i] - reference[i]).abs() <= ulp(reference[i]),
                "camera {camera_pos:?} cube {k:?}: origin {origin:?} vs reference {reference:?}"
            );
        }
    }

    #[test]
    fn origin_formula_matches_camera_relative() {
        // Near the origin, near negative borders, and around +-1e6 blocks,
        // on and off exact cube borders (including negative).
        for base in [
            DVec3::ZERO,
            DVec3::new(-1.0e6, 999_999.75, -1.0e6 + 16.5),
            DVec3::new(1.0e6 + 0.125, -1.0e6, 1.0e6 - 8.0),
        ] {
            for delta in [0.0, f64::from(CUBE_SIZE) / 2.0, f64::from(CUBE_SIZE)] {
                let camera = base + DVec3::new(delta, -delta, delta);
                // The cube containing the camera and its neighbors.
                let (cube, _) = camera_split(camera);
                for dx in -1..=1i32 {
                    for dy in -1..=1i32 {
                        for dz in -1..=1i32 {
                            check_origin(
                                camera,
                                CubeKey {
                                    level: 0,
                                    pos: jbm_core::coords::CubePos {
                                        x: cube.x + dx,
                                        y: cube.y + dy,
                                        z: cube.z + dz,
                                    },
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn camera_cube_floor_handles_negative_cameras() {
        // Exactly on a negative cube border: floor, not truncate.
        let (cube, frac) = camera_split(DVec3::new(-64.0, -32.0, -32.0));
        assert_eq!(cube, IVec3::new(-2, -1, -1));
        assert_eq!(frac, Vec3::ZERO);
        // One block before that border, and a negative in-cube position.
        let (cube, frac) = camera_split(DVec3::new(-64.5, 0.0, -1.0));
        assert_eq!(cube, IVec3::new(-3, 0, -1));
        assert_eq!(frac, Vec3::new(31.5, 0.0, 31.0));
        let (cube, frac) = camera_split(DVec3::new(-64.5, 0.0, 33.0));
        assert_eq!(cube, IVec3::new(-3, 0, 1));
        assert_eq!(frac, Vec3::new(31.5, 0.0, 1.0));
    }
}
