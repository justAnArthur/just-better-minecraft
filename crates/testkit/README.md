# jbm-testkit

Test-only helpers: a headless server harness and shared fixtures.

**Public API so far:** `HeadlessServer` — a server driven by a manual,
deterministic time source. This is the harness later stages extend with a
world, a fake client channel and replay recording; for now it wraps the
`jbm-server` tick clock.

- `HeadlessServer::new()`, `step(ticks)` advances exactly that many ticks,
  `run_for(d)` lets `d` of simulated time pass (delivered in tick-sized
  slices, so the catch-up cap never trips), `tick()` reports the count.

## Writing a headless test

Put integration tests in `tests/` (see `tests/headless.rs`):

```rust
use std::time::Duration;
use jbm_testkit::HeadlessServer;

let mut server = HeadlessServer::new();
server.step(100);                            // exact tick counts
server.run_for(Duration::from_secs(5));      // or time-driven
assert_eq!(server.tick(), 200);
```

Other crates add `jbm-testkit` as a dev-dependency only. Prefer `step` when
a test cares about exact tick numbers, `run_for` when it cares about
durations; both are deterministic.

**Invariants**

- Never a runtime dependency of any shipping crate; other crates pull it in
  only as a dev-dependency.
- May depend on `jbm-core` and `jbm-server`, but never on `jbm-client` (client
  tests add it as their own dev-dependency if needed).
