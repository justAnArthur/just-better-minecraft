//! Input: a winit-independent key set, accumulated mouse motion, and the
//! mapping from held keys onto the physics [`MoveInput`]. Mouse look is
//! applied per frame; movement is sampled per simulation tick.

use std::cell::Cell;

use glam::Vec2;

use jbm_core::physics::MoveInput;

use crate::camera::Camera;

/// Mouse sensitivity in radians per pixel.
pub const MOUSE_SENSITIVITY: f32 = 0.002;

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
/// Mouse deltas and the fly-toggle edge live in `Cell`s so the app's event
/// flow can set them through a shared reference.
#[derive(Debug, Default)]
pub struct InputState {
    keys: u8,
    mouse_delta: Cell<Vec2>,
    fly_toggle: Cell<bool>,
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

    /// Records one F press (one edge per press).
    pub fn press_fly_toggle(&self) {
        self.fly_toggle.set(true);
    }

    /// Returns whether F was pressed since the last call, clearing the
    /// edge. Consumed by the next simulation tick.
    pub fn take_fly_toggle(&self) -> bool {
        let pressed = self.fly_toggle.get();
        self.fly_toggle.set(false);
        pressed
    }
}

/// Applies pending mouse motion to the camera's yaw and pitch. Called once
/// per frame; the delta is consumed.
pub fn apply_mouse_look(camera: &mut Camera, input: &InputState) {
    let delta = input.take_mouse_delta();
    camera.set_yaw(camera.yaw + delta.x * MOUSE_SENSITIVITY);
    camera.set_pitch(camera.pitch - delta.y * MOUSE_SENSITIVITY);
}

/// Builds the tick's [`MoveInput`] from held keys, the camera yaw and an
/// F-press edge: WASD move, Space jumps (walk) or flies up, Shift flies
/// down, Ctrl sprints.
pub fn move_input(state: &InputState, yaw: f32, fly_toggle: bool) -> MoveInput {
    let axis = |plus: Key, minus: Key| state.is_held(plus) as i8 - state.is_held(minus) as i8;
    MoveInput {
        forward: axis(Key::Forward, Key::Back),
        strafe: axis(Key::Right, Key::Left),
        up: axis(Key::Up, Key::Down),
        jump: state.is_held(Key::Up),
        sprint: state.is_held(Key::Sprint),
        fly_toggle,
        yaw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn keys_map_onto_move_input_axes() {
        let mut input = InputState::new();
        input.set_key(Key::Forward, true);
        input.set_key(Key::Left, true);
        input.set_key(Key::Sprint, true);
        let m = move_input(&input, 0.25, false);
        assert_eq!(m.forward, 1);
        assert_eq!(m.strafe, -1);
        assert_eq!(m.up, 0);
        assert!(!m.jump);
        assert!(m.sprint);
        assert!(!m.fly_toggle);
        assert_eq!(m.yaw, 0.25);
    }

    #[test]
    fn space_and_shift_drive_the_vertical_axis_and_jump() {
        let mut input = InputState::new();
        input.set_key(Key::Up, true);
        assert_eq!(move_input(&input, 0.0, false).up, 1);
        assert!(move_input(&input, 0.0, false).jump);

        input.set_key(Key::Up, false);
        input.set_key(Key::Down, true);
        let m = move_input(&input, 0.0, false);
        assert_eq!(m.up, -1);
        assert!(!m.jump, "jump follows Space only");
    }

    #[test]
    fn opposing_keys_cancel() {
        let mut input = InputState::new();
        input.set_key(Key::Forward, true);
        input.set_key(Key::Back, true);
        assert_eq!(move_input(&input, 0.0, false).forward, 0);
    }

    #[test]
    fn fly_toggle_edge_is_consumed_once() {
        let input = InputState::new();
        input.press_fly_toggle();
        input.press_fly_toggle();
        assert!(move_input(&input, 0.0, input.take_fly_toggle()).fly_toggle);
        assert!(
            !input.take_fly_toggle(),
            "two presses before one tick still toggle once"
        );
    }

    #[test]
    fn mouse_delta_rotates_and_is_consumed() {
        let mut camera = Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0);
        let input = InputState::new();
        input.accumulate_mouse(Vec2::new(5.0, 0.0));
        apply_mouse_look(&mut camera, &input);
        assert!((camera.yaw - 5.0 * MOUSE_SENSITIVITY).abs() < 1e-6);
        // The delta was consumed: a second update with no motion changes nothing.
        let yaw = camera.yaw;
        apply_mouse_look(&mut camera, &input);
        assert_eq!(camera.yaw, yaw);
    }

    #[test]
    fn mouse_up_increases_pitch() {
        let mut camera = Camera::new(DVec3::ZERO, 70.0_f32.to_radians(), 1.0);
        let input = InputState::new();
        input.accumulate_mouse(Vec2::new(0.0, -2.0));
        apply_mouse_look(&mut camera, &input);
        assert!(
            (camera.pitch - 2.0 * MOUSE_SENSITIVITY).abs() < 1e-6,
            "pitch: {}",
            camera.pitch
        );
    }
}
