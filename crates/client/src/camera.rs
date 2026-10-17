//! The fly camera: f64 world position, yaw/pitch orientation, and a
//! reverse-Z infinite-far projection built for camera-relative rendering.

use glam::{DVec3, Mat4, Vec3, Vec4};

/// Near plane distance in world units.
pub const NEAR: f32 = 0.05;

/// Pitch is clamped to just under straight up/down so the view matrix
/// never degenerates.
pub const MAX_PITCH: f32 = 89.0_f32.to_radians();

/// A first-person camera.
///
/// Invariants (see crates/client/README.md):
/// - `position` is f64 because world coordinates reach ~6.8e10.
/// - Rendering is camera-relative: [`Camera::view_proj`] places the camera
///   at the origin (rotation only) and geometry is offset with
///   [`Camera::relative`] before it is drawn.
/// - The projection is reverse-Z with an infinite far plane: near maps to
///   depth 1.0, the far plane maps to 0.0, and the depth attachment is
///   cleared to 0.0 with a `Greater` compare.
pub struct Camera {
    /// World-space position in f64 (block coordinates can reach 6.8e10).
    pub position: DVec3,
    /// Horizontal rotation in radians. Yaw 0 looks toward -Z; yaw increases
    /// toward +X (turning right).
    pub yaw: f32,
    /// Vertical rotation in radians, clamped to +-MAX_PITCH.
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Width / height of the target.
    pub aspect: f32,
}

impl Camera {
    pub fn new(position: DVec3, fov_y: f32, aspect: f32) -> Camera {
        Camera {
            position,
            yaw: 0.0,
            pitch: 0.0,
            fov_y,
            aspect,
        }
    }

    /// Sets yaw without clamping (full turns are fine).
    pub fn set_yaw(&mut self, yaw: f32) {
        self.yaw = yaw;
    }

    /// Sets pitch, clamped to ±89°.
    pub fn set_pitch(&mut self, pitch: f32) {
        self.pitch = pitch.clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// A camera at `eye` looking toward `look_at`, with the given fov and
    /// aspect — the fixed-camera constructor tests and the playground use.
    /// Yaw 0 looks toward -Z; pitch is clamped like [`Camera::set_pitch`].
    pub fn looking_at(eye: DVec3, look_at: DVec3, fov_y: f32, aspect: f32) -> Camera {
        let d = look_at - eye;
        let yaw = f32::atan2(d.x as f32, -d.z as f32);
        let pitch = f32::atan2(d.y as f32, d.x.hypot(d.z) as f32);
        let mut camera = Camera::new(eye, fov_y, aspect);
        camera.set_yaw(yaw);
        camera.set_pitch(pitch);
        camera
    }

    /// Unit look direction: yaw 0 pitch 0 is (0, 0, -1).
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, -cy * cp)
    }

    /// Unit right direction, horizontal (yaw plane).
    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, sy)
    }

    /// Rotation-only view matrix: the camera sits at the origin and the
    /// world is shifted to it (camera-relative rendering).
    pub fn view(&self) -> Mat4 {
        let f = self.forward();
        let r = self.right();
        let u = r.cross(f);
        Mat4::from_cols(
            Vec4::new(r.x, u.x, -f.x, 0.0),
            Vec4::new(r.y, u.y, -f.y, 0.0),
            Vec4::new(r.z, u.z, -f.z, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Reverse-Z perspective projection with an infinite far plane.
    /// Maps view z = -NEAR to depth 1.0 and z = -inf to 0.0.
    pub fn projection(&self) -> Mat4 {
        let f = 1.0 / (self.fov_y * 0.5).tan();
        Mat4::from_cols(
            Vec4::new(f / self.aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, NEAR, 0.0),
        )
    }

    /// Rotation-only view-projection matrix (f32; camera-relative).
    pub fn view_proj(&self) -> Mat4 {
        self.projection() * self.view()
    }

    /// A world-space point as camera-relative f32 coordinates:
    /// `(world - position)` computed in f64, then cast. Keeps sub-block
    /// precision even at world coordinates around 1e9.
    pub fn relative(&self, world: DVec3) -> Vec3 {
        let d = world - self.position;
        Vec3::new(d.x as f32, d.y as f32, d.z as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0)
    }

    #[test]
    fn yaw_zero_looks_toward_negative_z() {
        let c = camera();
        assert!(c.forward().abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), 1e-6));
        assert!(c.right().abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn quarter_yaw_turns_toward_positive_x() {
        let mut c = camera();
        c.set_yaw(90.0_f32.to_radians());
        assert!(c.forward().abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-6));
        assert!(c.right().abs_diff_eq(Vec3::new(0.0, 0.0, 1.0), 1e-6));
    }

    #[test]
    fn pitch_up_tilts_forward() {
        let mut c = camera();
        c.set_pitch(45.0_f32.to_radians());
        let s = (45.0_f32.to_radians()).sin();
        assert!(c.forward().abs_diff_eq(Vec3::new(0.0, s, -s), 1e-6));
    }

    #[test]
    fn pitch_is_clamped() {
        let mut c = camera();
        c.set_pitch(4.0);
        assert_eq!(c.pitch, MAX_PITCH);
        c.set_pitch(-4.0);
        assert_eq!(c.pitch, -MAX_PITCH);
        c.set_pitch(0.5);
        assert_eq!(c.pitch, 0.5);
    }

    #[test]
    fn point_straight_ahead_projects_to_screen_center() {
        let c = camera();
        // A point 10 units straight ahead of the origin camera.
        let ndc = c.view_proj().project_point3(Vec3::new(0.0, 0.0, -10.0));
        assert!(ndc.x.abs() < 1e-5, "x not centered: {}", ndc.x);
        assert!(ndc.y.abs() < 1e-5, "y not centered: {}", ndc.y);
        // Reverse-Z: depth = NEAR / distance = 0.05 / 10.
        assert!((ndc.z - NEAR / 10.0).abs() < 1e-5, "depth: {}", ndc.z);
    }

    #[test]
    fn nearer_point_gets_larger_depth() {
        let c = camera();
        let vp = c.view_proj();
        // Depth = NEAR / distance: 1.0 at the near plane itself, 0.005 at
        // 10 units, tending to 0.0 at infinity.
        let at_near = vp.project_point3(Vec3::new(0.0, 0.0, -NEAR)).z;
        let near = vp.project_point3(Vec3::new(0.0, 0.0, -0.1)).z;
        let far = vp.project_point3(Vec3::new(0.0, 0.0, -10.0)).z;
        assert!((at_near - 1.0).abs() < 1e-5, "near-plane depth: {at_near}");
        assert!((near - 0.5).abs() < 1e-5, "depth at 0.1: {near}");
        assert!((far - NEAR / 10.0).abs() < 1e-5, "depth at 10: {far}");
        assert!(near > far);
    }

    #[test]
    fn relative_keeps_sub_millimetre_precision_at_billions() {
        let mut c = camera();
        c.position = DVec3::new(1.0e9, 0.0, 0.0);
        let rel = c.relative(DVec3::new(1.0e9 + 1.0e-3, 0.0, 0.0));
        assert!((rel.x - 1.0e-3).abs() < 1.0e-6, "rel.x: {}", rel.x);
        assert!(rel.y.abs() < 1e-9 && rel.z.abs() < 1e-9);
    }

    #[test]
    fn looking_at_aims_forward_at_the_target() {
        // Straight down -Z: yaw 0, pitch 0.
        let c = Camera::looking_at(
            DVec3::ZERO,
            DVec3::new(0.0, 0.0, -10.0),
            70.0_f32.to_radians(),
            1.0,
        );
        assert!(c.forward().abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), 1e-6));
        // From +X toward the origin: yaw 90°, and a downward pitch when
        // the target sits lower.
        let c = Camera::looking_at(
            DVec3::new(10.0, 10.0, 0.0),
            DVec3::ZERO,
            70.0_f32.to_radians(),
            1.0,
        );
        assert!((c.yaw - 90.0_f32.to_radians()).abs() < 1e-5);
        assert!((c.pitch - (-45.0_f32.to_radians())).abs() < 1e-5);
    }
}
