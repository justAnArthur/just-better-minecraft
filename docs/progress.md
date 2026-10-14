# Progress

## Next up

Stage 0 gate: CI green on the wave B PR, then the bench brief run (0.4's last check), then merge and /clear. Next stage: Stage 1 wave A (1.1, 1.2, 1.3). Remote: github.com/justAnArthur/just-better-minecraft (private).

## Benchmarks

| bench | median | machine | date |
|---|---|---|---|
| sample/fill_32cubed | 118.16 µs | Apple M5 Pro (bench-0) | 2026-10-14 |
| tick_clock/advance_1000 | 4.41 µs | Apple M5 Pro (bench-0) | 2026-10-14 |

Baseline = bench-0 run on an otherwise idle machine. Compare only runs made without parallel builds (the implementer's 3.67 µs ran alongside other builds; ignore it).

## Decisions

- 2026-10-14: Crate packages are prefixed `jbm-` (a package named `core` would clash with Rust's `core`). Dirs stay `crates/core` etc.
- 2026-10-14: Workers run through `scripts/delegate.sh` (the `glm` CLI directly, in the background) rather than the haiku relay agent, so long Rust builds aren't cut off by the relay's 10-minute Bash timeout.
- 2026-10-14: Screenshot tests compare with a tolerance and skip when no GPU adapter exists, unless `JBM_REQUIRE_GPU=1` (set in CI on Linux with Mesa lavapipe).

## Known issues

- client/render/offscreen.rs: device poll failure is only a debug_assert; in release a failed poll could hang `recv()`. Fix when readback is used outside tests.
- testkit re-derives TICK_PERIOD instead of reusing a server constant. Export one canonical constant in 1.3.
- First release/bench build is slow (wgpu + thin LTO + codegen-units=1). Revisit profile if CI bench-smoke gets too slow.

## Log

### 2026-10-14
- Repo created. 0.3 done: CLAUDE.md, docs/design.md, docs/plan.md, docs/progress.md. 0.4 drafted: docs/briefs/, scripts/delegate.sh (validated by running 0.1 through it).
- 0.1 merged: Cargo workspace (core, server, client, modhost, game, testkit), toolchain 1.98.1, edition 2024. Implementer + reviewer briefs worked end to end (reports in format), so the 0.4 brief check passed for implementer/reviewer; bench brief gets exercised by 0.6.
- Wave B merged on branch stage-0/wave-b: 0.2 CI workflow, 0.5 screenshot harness (wgpu 30.0.1, golden triangle checked by orchestrator), 0.6 TickClock + HeadlessServer + criterion 0.8.2. All reviewers: MERGE.
- Repo: github.com/justAnArthur/just-better-minecraft (private).
- 0.4 done: bench brief exercised via bench-0 (report in format). Baseline numbers recorded above.
