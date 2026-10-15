//! Client-side prediction: steps the shared physics ahead of the server,
//! remembers every unacknowledged input, and replays them over each
//! authoritative [`ServerMsg::PlayerState`]. Pure logic, no winit — the
//! app drives it from the frame loop.

use std::collections::VecDeque;

use glam::DVec3;
use jbm_core::physics::{self, BlockShapes, MoveInput, PlayerState};
use jbm_core::protocol::ClientMsg;

/// Unacknowledged inputs kept for replay; past that the oldest is dropped
/// and counted (the server's next state resynchronizes anyway).
const PENDING_CAP: usize = 128;

/// The client's predicted player.
///
/// - [`Predictor::tick`] steps the prediction and returns the `PlayerInput`
///   to send. `pitch` in the returned message is a placeholder the app
///   overwrites with the camera pitch (physics ignores it).
/// - [`Predictor::on_server_state`] drops acknowledged inputs, re-simulates
///   from the server state through the remaining pending inputs **in seq
///   order**, and counts a correction when the result differs from the
///   prediction.
pub struct Predictor {
    state: PlayerState,
    /// The previous tick's state, for render interpolation.
    prev: PlayerState,
    /// Seq of the next input; starts at 1.
    next_seq: u64,
    pending: VecDeque<(u64, MoveInput)>,
    corrections: u64,
    dropped: u64,
}

impl Predictor {
    /// A predictor starting at the spawn feet position: velocity 0, not on
    /// ground, walking.
    pub fn new(spawn: DVec3) -> Predictor {
        Predictor {
            state: PlayerState {
                position: spawn,
                velocity: DVec3::ZERO,
                on_ground: false,
                flying: false,
            },
            prev: PlayerState {
                position: spawn,
                velocity: DVec3::ZERO,
                on_ground: false,
                flying: false,
            },
            next_seq: 1,
            pending: VecDeque::new(),
            corrections: 0,
            dropped: 0,
        }
    }

    /// Steps the prediction one tick under `input` and returns the
    /// `PlayerInput` to send for it.
    pub fn tick(&mut self, input: MoveInput, world: &impl BlockShapes) -> ClientMsg {
        self.prev = self.state;
        self.state = physics::step(self.state, &input, world);
        let seq = self.next_seq;
        self.next_seq += 1;
        self.pending.push_back((seq, input));
        if self.pending.len() > PENDING_CAP {
            self.pending.pop_front();
            self.dropped += 1;
        }
        ClientMsg::PlayerInput {
            seq,
            input,
            pitch: 0.0,
        }
    }

    /// Applies an authoritative state: trims acknowledged inputs, replays
    /// the rest on top of it, and counts a correction if the replay
    /// disagrees with the prediction.
    ///
    /// An agreeing ack (the normal case, every tick) must not touch the
    /// render interpolation: resetting `prev` there made the camera jump
    /// to the newest tick on every ack, a 20 Hz stutter. A correction
    /// moves both endpoints by the correction offset, so the rendered
    /// motion keeps its shape and only the error itself shows as a snap.
    /// Smoothing that snap is a later task.
    pub fn on_server_state(&mut self, ack_seq: u64, state: PlayerState, world: &impl BlockShapes) {
        while self.pending.front().is_some_and(|(seq, _)| *seq <= ack_seq) {
            self.pending.pop_front();
        }
        let mut replayed = state;
        for (_, input) in &self.pending {
            replayed = physics::step(replayed, input, world);
        }
        if replayed != self.state {
            self.corrections += 1;
            self.prev.position += replayed.position - self.state.position;
            self.state = replayed;
        }
    }

    /// The current predicted state.
    pub fn state(&self) -> PlayerState {
        self.state
    }

    /// The eye position interpolated between the previous and current
    /// tick, for rendering at sim fraction `alpha` in 0..=1.
    pub fn render_position(&self, alpha: f64) -> DVec3 {
        self.prev.eye().lerp(self.state.eye(), alpha)
    }

    /// Total corrections so far.
    pub fn corrections(&self) -> u64 {
        self.corrections
    }

    /// Inputs sent but not yet acknowledged.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Inputs dropped for exceeding [`PENDING_CAP`].
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jbm_core::coords::BlockPos;

    /// A solid floor below `top`, all loaded.
    struct Floor {
        top: i64,
    }

    impl BlockShapes for Floor {
        fn solid(&self, pos: BlockPos) -> Option<bool> {
            Some(pos.y < self.top)
        }
    }

    fn walking_forward() -> MoveInput {
        MoveInput {
            forward: 1,
            ..MoveInput::IDLE
        }
    }

    #[test]
    fn a_matching_server_state_costs_no_correction_and_trims_pending() {
        let world = Floor { top: 64 };
        let mut predictor = Predictor::new(DVec3::new(0.5, 64.0, 0.5));

        // The server echoes back exactly what the client predicted.
        let mut predicted = Vec::new();
        for _ in 0..5 {
            let _ = predictor.tick(walking_forward(), &world);
            predicted.push(predictor.state());
        }
        for (ack, state) in (1u64..).zip(predicted) {
            predictor.on_server_state(ack, state, &world);
        }
        assert_eq!(predictor.corrections(), 0);
        assert_eq!(predictor.pending_len(), 0);
    }

    #[test]
    fn a_one_block_server_error_corrects_once_to_the_replayed_state() {
        let world = Floor { top: 64 };
        let mut predictor = Predictor::new(DVec3::new(0.5, 64.0, 0.5));

        for _ in 0..3 {
            let _ = predictor.tick(walking_forward(), &world);
        }
        // The server is one block ahead after acking the second input.
        let mut server = predictor.state();
        server.position.y += 1.0;
        predictor.on_server_state(2, server, &world);

        assert_eq!(predictor.corrections(), 1);
        // The result is the server state replayed through the remaining
        // pending input (seq 3), not the raw server state.
        let expected = physics::step(server, &walking_forward(), &world);
        assert_eq!(predictor.state(), expected);
        assert_eq!(predictor.pending_len(), 1);
    }

    #[test]
    fn replay_applies_pending_inputs_in_seq_order() {
        let world = Floor { top: 64 };
        let mut predictor = Predictor::new(DVec3::new(0.5, 64.0, 0.5));

        // Order matters: A toggles fly and rises, B toggles back off.
        // A then B lands 0.47 above the floor; B then A ends below the
        // start. The predictor must produce the first.
        let a = MoveInput {
            fly_toggle: true,
            up: 1,
            ..MoveInput::IDLE
        };
        let b = MoveInput {
            fly_toggle: true,
            ..MoveInput::IDLE
        };
        let _ = predictor.tick(a, &world);
        let _ = predictor.tick(b, &world);

        // Nothing acknowledged: the server state is the spawn state.
        let server = Predictor::new(DVec3::new(0.5, 64.0, 0.5)).state();
        predictor.on_server_state(0, server, &world);

        let forward = physics::step(server, &a, &world);
        let expected = physics::step(forward, &b, &world);
        let reversed = physics::step(physics::step(server, &b, &world), &a, &world);
        assert_ne!(expected, reversed, "the two orders must differ");
        assert_eq!(predictor.state(), expected, "replayed in seq order");
    }

    #[test]
    fn the_pending_cap_holds() {
        let world = Floor { top: 64 };
        let mut predictor = Predictor::new(DVec3::new(0.5, 64.0, 0.5));
        for _ in 0..200 {
            let _ = predictor.tick(MoveInput::IDLE, &world);
        }
        assert_eq!(predictor.pending_len(), 128);
        assert_eq!(predictor.dropped(), 72);
    }

    #[test]
    fn render_position_interpolates_the_eye() {
        let world = Floor { top: 64 };
        let mut predictor = Predictor::new(DVec3::new(0.5, 64.0, 0.5));
        let before = predictor.render_position(0.5);
        let _ = predictor.tick(walking_forward(), &world);
        let after = predictor.render_position(0.5);
        assert!(after.z < before.z, "walking forward moves -Z");
        assert_eq!(
            predictor.render_position(0.0).y,
            64.0 + jbm_core::physics::EYE_HEIGHT
        );
        // At alpha 1 the eye sits at the current state's eye.
        let m = predictor.state();
        assert_eq!(predictor.render_position(1.0), m.eye());
    }

    #[test]
    fn an_agreeing_ack_leaves_render_interpolation_alone() {
        let floor = Floor { top: 64 };
        let mut predictor = Predictor::new(DVec3::new(0.5, 64.0, 0.5));
        let walk = MoveInput {
            forward: 1,
            ..MoveInput::IDLE
        };
        // A server that agrees: it steps the same inputs from the same start.
        let mut server = predictor.state();
        for _ in 0..3 {
            predictor.tick(walk, &floor);
            server = physics::step(server, &walk, &floor);
        }
        let before = predictor.render_position(0.5);
        predictor.on_server_state(2, physics_state_after(2, walk, &floor), &floor);
        assert_eq!(predictor.corrections(), 0);
        assert_eq!(predictor.state(), server);
        assert_eq!(predictor.render_position(0.5), before);
    }

    #[test]
    fn a_correction_shifts_both_interpolation_endpoints() {
        let floor = Floor { top: 64 };
        let mut predictor = Predictor::new(DVec3::new(0.5, 64.0, 0.5));
        let walk = MoveInput {
            forward: 1,
            ..MoveInput::IDLE
        };
        predictor.tick(walk, &floor);
        predictor.tick(walk, &floor);
        let before_mid = predictor.render_position(0.5);
        let before_end = predictor.render_position(1.0);
        // The server says tick 1 ended one block further along +X.
        let mut off = physics_state_after(1, walk, &floor);
        off.position.x += 1.0;
        predictor.on_server_state(1, off, &floor);
        assert_eq!(predictor.corrections(), 1);
        let shift = predictor.render_position(1.0) - before_end;
        assert!((shift - DVec3::X).length() < 1e-9, "{shift:?}");
        let mid_shift = predictor.render_position(0.5) - before_mid;
        assert!((mid_shift - DVec3::X).length() < 1e-9, "{mid_shift:?}");
    }

    /// The state after `n` steps of `input` from the predictor's spawn.
    fn physics_state_after(n: usize, input: MoveInput, world: &Floor) -> PlayerState {
        let mut state = Predictor::new(DVec3::new(0.5, 64.0, 0.5)).state();
        for _ in 0..n {
            state = physics::step(state, &input, world);
        }
        state
    }
}
