# jbm-server

The authoritative simulation: world generation, lighting, storage, and the
server side of the protocol.

**Public API so far:** `tick` — the fixed-rate simulation clock.

- `TICKS_PER_SECOND: u32 = 20`, `MAX_CATCH_UP_TICKS: u32 = 10`.
- `TickClock::new()` / `Default`, `tick_count()`, `dropped_ticks()`, and
  `advance(elapsed) -> u32`, which accumulates elapsed time and returns the
  whole ticks due, capped at `MAX_CATCH_UP_TICKS`. Excess whole ticks are
  dropped (no spiral of death) and counted in `dropped_ticks()`; only the
  sub-tick remainder is kept. Pure logic: no threads, no sleeping, no real
  clock reads — the caller feeds in time.

## Benchmarks

Criterion benches live in `benches/`, registered in `Cargo.toml`:

```toml
[[bench]]
name = "tick_bench"
harness = false
```

To add a case, add a function to `benches/tick_bench.rs` inside a
`benchmark_group` named for the area, with the case as the function name, so
reports read `<area>/<case>` (e.g. `tick_clock/advance_1000`). `criterion` is
a dev-dependency via `[workspace.dependencies]` only — never a runtime dep.

```bash
cargo bench -p jbm-server -- --test   # run each bench once (what CI calls)
cargo bench -p jbm-server             # full criterion run with numbers
```

**Invariants**

- Depends only on `jbm-core`; never on `jbm-client`. Both sides communicate
  exclusively through `jbm-core` protocol messages.
- Hot paths (lighting, worldgen inner loops) never call into mods.
