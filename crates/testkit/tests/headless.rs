use std::time::Duration;

use jbm_testkit::HeadlessServer;

#[test]
fn step_advances_exactly_that_many_ticks() {
    let mut server = HeadlessServer::new();
    server.step(100);
    assert_eq!(server.tick(), 100);
}

#[test]
fn run_for_five_seconds_in_fifty_ms_increments_yields_exactly_100_ticks() {
    let mut server = HeadlessServer::new();
    for _ in 0..100 {
        server.run_for(Duration::from_millis(50));
    }
    assert_eq!(server.tick(), 100);
}

#[test]
fn run_for_one_shot_five_seconds_yields_the_same_100_ticks() {
    let mut server = HeadlessServer::new();
    server.run_for(Duration::from_secs(5));
    assert_eq!(server.tick(), 100);
}

#[test]
fn sub_tick_remainders_accumulate_across_run_for_calls() {
    let mut server = HeadlessServer::new();
    for _ in 0..2 {
        server.run_for(Duration::from_millis(25));
    }
    assert_eq!(server.tick(), 1);
}
