//! Fixed-rate tick clock for the simulation loop.
//!
//! Pure logic: the caller feeds in elapsed time (from any source, real or
//! simulated) and the clock answers how many whole ticks are due. No threads,
//! no sleeping, no real clock reads.

use std::time::Duration;

/// Simulation rate: 20 ticks per second.
pub const TICKS_PER_SECOND: u32 = 20;

/// Most whole ticks a single [`TickClock::advance`] can return. A longer
/// stall drops the excess instead of catching up, so a slow frame cannot
/// start a spiral of death; dropped ticks stay visible in
/// [`TickClock::dropped_ticks`].
pub const MAX_CATCH_UP_TICKS: u32 = 10;

const TICK_PERIOD: Duration = Duration::from_millis(1000 / TICKS_PER_SECOND as u64);

/// Accumulates elapsed time and hands out whole ticks at
/// [`TICKS_PER_SECOND`].
#[derive(Debug, Clone)]
pub struct TickClock {
    tick_count: u64,
    dropped_ticks: u64,
    /// Elapsed time towards the next tick; always < one tick period.
    leftover: Duration,
}

impl TickClock {
    pub const fn new() -> Self {
        Self {
            tick_count: 0,
            dropped_ticks: 0,
            leftover: Duration::ZERO,
        }
    }

    /// Ticks handed out so far.
    pub const fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Whole ticks dropped by the [`MAX_CATCH_UP_TICKS`] cap.
    pub const fn dropped_ticks(&self) -> u64 {
        self.dropped_ticks
    }

    /// Feed `elapsed` time into the clock and return how many whole ticks
    /// are due now: at most [`MAX_CATCH_UP_TICKS`]. Excess whole ticks are
    /// dropped and counted; only the sub-tick remainder is kept.
    pub fn advance(&mut self, elapsed: Duration) -> u32 {
        self.leftover = self.leftover.saturating_add(elapsed);
        let period = TICK_PERIOD.as_nanos();
        let due = self.leftover.as_nanos() / period;
        let run = due.min(MAX_CATCH_UP_TICKS as u128) as u32;
        let dropped = u64::try_from(due - run as u128).unwrap_or(u64::MAX);
        self.dropped_ticks = self.dropped_ticks.saturating_add(dropped);
        self.tick_count += run as u64;
        self.leftover = Duration::from_nanos((self.leftover.as_nanos() % period) as u64);
        run
    }
}

impl Default for TickClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{TICK_PERIOD, TICKS_PER_SECOND, TickClock};
    use std::time::Duration;

    #[test]
    fn tick_period_matches_rate() {
        assert_eq!(TICK_PERIOD, Duration::from_secs(1) / TICKS_PER_SECOND);
        assert_eq!(TICK_PERIOD, Duration::from_millis(50));
    }

    #[test]
    fn fifty_ms_is_exactly_one_tick() {
        let mut clock = TickClock::new();
        assert_eq!(clock.advance(Duration::from_millis(50)), 1);
        assert_eq!(clock.tick_count(), 1);
        assert_eq!(clock.dropped_ticks(), 0);
    }

    #[test]
    fn sub_period_time_accumulates() {
        let mut clock = TickClock::new();
        assert_eq!(clock.advance(Duration::from_millis(49)), 0);
        assert_eq!(clock.tick_count(), 0);
        assert_eq!(clock.advance(Duration::from_millis(1)), 1);
        assert_eq!(clock.tick_count(), 1);
    }

    #[test]
    fn one_second_is_capped_at_ten_ticks() {
        let mut clock = TickClock::new();
        assert_eq!(clock.advance(Duration::from_secs(1)), 10);
        assert_eq!(clock.tick_count(), 10);
        assert_eq!(clock.dropped_ticks(), 10);
        // The sub-tick remainder is zero, so the next whole tick is 50 ms away.
        assert_eq!(clock.advance(Duration::from_millis(49)), 0);
    }

    #[test]
    fn zero_duration_ticks_nothing() {
        let mut clock = TickClock::new();
        assert_eq!(clock.advance(Duration::ZERO), 0);
        assert_eq!(clock.tick_count(), 0);
        assert_eq!(clock.dropped_ticks(), 0);
    }
}
