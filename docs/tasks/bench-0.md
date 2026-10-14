# bench-0 Baseline bench run (Stage 0 gate)

**Goal:** check the bench brief works end to end and record baseline numbers.

Run:
- `cargo bench -p jbm-core --bench core_bench 2>&1 | tail -40`
- `cargo bench -p jbm-server --bench tick_bench 2>&1 | tail -40`

Benches: `sample/fill_32cubed`, `tick_clock/advance_1000`. Previous numbers: see docs/progress.md "Benchmarks" (fill_32cubed has none yet).
