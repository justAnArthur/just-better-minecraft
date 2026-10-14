# bench-1 Baselines after Stage 1 wave A

**Goal:** idle-machine baseline numbers for the benches added in 1.1 and 1.3.

Run:
- `cargo bench -p jbm-core --bench core_bench 2>&1 | tail -60`
- `cargo bench -p jbm-server --bench tick_bench 2>&1 | tail -40`

Benches: `sample/fill_32cubed`, `cube/get_all_16_states`, `cube/set_all_16_states`, `cube/palette_growth_4096`, `cube/iter_sum_16_states`, `cube/compact_4096_to_16`, `tick_clock/advance_1000`, `server/tick_8_clients`. Previous numbers: docs/progress.md "Benchmarks" (only fill_32cubed and advance_1000 have one; the rest are new baselines).
