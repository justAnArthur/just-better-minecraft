//! Fly-camera input: a winit-independent key set plus accumulated mouse
//! motion, and the controller that turns them into camera motion.

use std::cell::Cell;

use glam::{DVec3, Vec2};

use crate::camera::Camera;

/// A movement input, decoupled from winit types so the logic stays
/// unit-testable. The app layer maps physical keys onto these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Forward,
    Back,
    Left,
    Right,
    Up,
    Down,
    Sprint,
}

impl Key {
    fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

/// Held movement keys and accumulated, not-yet-consumed mouse motion.
///
/// Mouse deltas live in a `Cell` so [`FlyController::update`] can consume
/// them through a shared reference, matching the app's event flow.
#[derive(Debug, Default)]
pub struct InputState {
    keys: u8,
    mouse_delta: Cell<Vec2>,
}

impl InputState {
    pub fn new() -> InputState {
        InputState::default()
    }

    /// Marks `key` held or released.
    pub fn set_key(&mut self, key: Key, held: bool) {
        if held {
            self.keys |= key.bit();
        } else {
            self.keys &= !key.bit();
        }
    }

    pub fn is_held(&self, key: Key) -> bool {
        self.keys & key.bit() != 0
    }

    /// Releases every key (used when the window loses focus).
    pub fn clear_keys(&mut self) {
        self.keys = 0;
    }

    /// Adds raw mouse motion (pixels) to the pending delta.
    pub fn accumulate_mouse(&self, delta: Vec2) {
        self.mouse_delta.set(self.mouse_delta.get() + delta);
    }

    /// Returns the pending mouse delta and resets it to zero.
    pub fn take_mouse_delta(&self) -> Vec2 {
        let delta = self.mouse_delta.get();
        self.mouse_delta.set(Vec2::ZERO);
        delta
    }
}

/// Fly movement tuning: speed in blocks per second, a sprint multiplier
/// and mouse sensitivity in radians per pixel.
#[derive(Debug, Clone, Copy)]
pub struct FlyController {
    pub speed: f64,
    pub sprint_multiplier: f32,
    pub sensitivity: f32,
}

impl Default for FlyController {
    fn default() -> Self {
        FlyController {
            speed: 20.0,
            sprint_multiplier: 4.0,
            sensitivity: 0.002,
        }
    }
}

impl FlyController {
    /// Applies look and movement for one frame of `dt` seconds.
    ///
    /// Horizontal movement follows yaw only; Up/Down move along world Y;
    /// the combined direction is normalized so diagonals are not faster.
    /// Mouse deltas in `input` are consumed.
    pub fn update(&self, camera: &mut Camera, input: &InputState, dt: f32) {
        let delta = input.take_mouse_delta();
        camera.set_yaw(camera.yaw + delta.x * self.sensitivity);
        camera.set_pitch(camera.pitch - delta.y * self.sensitivity);

        let yaw = camera.yaw as f64;
        let forward = DVec3::new(yaw.sin(), 0.0, -yaw.cos());
        let right = DVec3::new(yaw.cos(), 0.0, yaw.sin());

        let mut move_dir = DVec3::ZERO;
        if input.is_held(Key::Forward) {
            move_dir += forward;
        }
        if input.is_held(Key::Back) {
            move_dir -= forward;
        }
        if input.is_held(Key::Right) {
            move_dir += right;
        }
        if input.is_held(Key::Left) {
            move_dir -= right;
        }
        if input.is_held(Key::Up) {
            move_dir.y += 1.0;
        }
        if input.is_held(Key::Down) {
            move_dir.y -= 1.0;
        }
        if move_dir != DVec3::ZERO {
            move_dir = move_dir.normalize();
        }
        let speed = self.speed
            * if input.is_held(Key::Sprint) {
                f64::from(self.sprint_multiplier)
            } else {
                1.0
            };
        camera.position += move_dir * speed * f64::from(dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> FlyController {
        FlyController {
            speed: 10.0,
            sprint_multiplier: 4.0,
            sensitivity: 0.1,
        }
    }

    #[test]
    fn forward_one_second_moves_speed_toward_negative_z() {
        let mut camera = Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0);
        let mut input = InputState::new();
        input.set_key(Key::Forward, true);
        controller().update(&mut camera, &input, 1.0);
        assert!(
            (camera.position.z + 10.0).abs() < 1e-6,
            "{:?}",
            camera.position
        );
        assert!(camera.position.x.abs() < 1e-6);
        assert!(camera.position.y.abs() < 1e-6);
    }

    #[test]
    fn diagonal_movement_keeps_the_same_speed() {
        let mut camera = Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0);
        let mut input = InputState::new();
        input.set_key(Key::Forward, true);
        input.set_key(Key::Right, true);
        controller().update(&mut camera, &input, 1.0);
        assert!(
            (camera.position.length() - 10.0).abs() < 1e-6,
            "{:?}",
            camera.position
        );
        let d = (45.0_f64).to_radians();
        assert!((camera.position.x - 10.0 * d.sin()).abs() < 1e-6);
        assert!((camera.position.z + 10.0 * d.cos()).abs() < 1e-6);
    }

    #[test]
    fn up_moves_along_world_y() {
        let mut camera = Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0);
        camera.set_yaw(90.0_f32.to_radians());
        let mut input = InputState::new();
        input.set_key(Key::Up, true);
        controller().update(&mut camera, &input, 1.0);
        assert!((camera.position.y - 10.0).abs() < 1e-6);
        assert!(camera.position.x.abs() < 1e-6 && camera.position.z.abs() < 1e-6);
    }

    #[test]
    fn sprint_multiplies_speed() {
        let mut camera = Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0);
        let mut input = InputState::new();
        input.set_key(Key::Forward, true);
        input.set_key(Key::Sprint, true);
        controller().update(&mut camera, &input, 1.0);
        assert!(
            (camera.position.z + 40.0).abs() < 1e-6,
            "{:?}",
            camera.position
        );
    }

    #[test]
    fn mouse_delta_rotates_and_is_consumed() {
        let mut camera = Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0);
        let input = InputState::new();
        input.accumulate_mouse(Vec2::new(5.0, 0.0));
        controller().update(&mut camera, &input, 0.0);
        assert!((camera.yaw - 0.5).abs() < 1e-6, "yaw: {}", camera.yaw);
        // The delta was consumed: a second update with no motion changes nothing.
        let yaw = camera.yaw;
        controller().update(&mut camera, &input, 0.0);
        assert_eq!(camera.yaw, yaw);
    }

    #[test]
    fn mouse_up_increases_pitch() {
        let mut camera = Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0);
        let input = InputState::new();
        input.accumulate_mouse(Vec2::new(0.0, -2.0));
        controller().update(&mut camera, &input, 0.0);
        assert!((camera.pitch - 0.2).abs() < 1e-6, "pitch: {}", camera.pitch);
    }
}
