//! Test-only helpers: a headless server harness and fixtures. Never a
//! dependency of shipping code — crates use it as a dev-dependency.

use std::time::Duration;

use jbm_server::tick::{TICKS_PER_SECOND, TickClock};

const TICK_PERIOD: Duration = Duration::from_millis(1000 / TICKS_PER_SECOND as u64);

/// A headless server driven by a manual, deterministic time source.
///
/// This is the harness later stages extend with a world, a fake client
/// channel and replay recording; for now it only advances the tick clock.
#[derive(Debug, Clone, Default)]
pub struct HeadlessServer {
    clock: TickClock,
}

impl HeadlessServer {
    pub const fn new() -> Self {
        Self {
            clock: TickClock::new(),
        }
    }

    /// Advance exactly `ticks` simulation ticks.
    pub fn step(&mut self, ticks: u32) {
        for _ in 0..ticks {
            // One period per advance call, so the catch-up cap never bites.
            self.clock.advance(TICK_PERIOD);
        }
    }

    /// Let `d` of simulated time pass, delivered to the clock in tick-sized
    /// slices like a well-behaved frame loop would; the trailing sub-tick
    /// remainder is kept as credit towards the next tick.
    pub fn run_for(&mut self, d: Duration) {
        let mut remaining = d;
        while remaining >= TICK_PERIOD {
            self.clock.advance(TICK_PERIOD);
            remaining -= TICK_PERIOD;
        }
        if !remaining.is_zero() {
            self.clock.advance(remaining);
        }
    }

    /// Ticks simulated so far.
    pub fn tick(&self) -> u64 {
        self.clock.tick_count()
    }
}
