# Progress

## Next up

Stage 1 · Voxel sandbox, wave B: 1.4 greedy mesher + worker pool + packed faces + vertex pulling (client), 1.5 noise terrain + cube streaming around the player (server). Wave B also wires the server thread into `jbm-game` (client connects over `in_process_pair`).

## Benchmarks

| bench | median | machine | date |
|---|---|---|---|
| sample/fill_32cubed | 105.76 µs | Apple M5 Pro (bench-1) | 2026-10-14 |
| tick_clock/advance_1000 | 4.14 µs | Apple M5 Pro (bench-1) | 2026-10-14 |
| cube/get_all_16_states | 91.66 µs | Apple M5 Pro (bench-1) | 2026-10-14 |
| cube/set_all_16_states | 493.68 µs | Apple M5 Pro (bench-1) | 2026-10-14 |
| cube/palette_growth_4096 | 626.19 µs | Apple M5 Pro (bench-1) | 2026-10-14 |
| cube/iter_sum_16_states | 32.29 µs | Apple M5 Pro (bench-1) | 2026-10-14 |
| cube/compact_4096_to_16 | 96.57 µs | Apple M5 Pro (bench-1) | 2026-10-14 |
| server/tick_8_clients | 292.53 ns | Apple M5 Pro (bench-1) | 2026-10-14 |

Baselines are bench worker runs on an otherwise idle machine. Run-to-run noise on this Mac is ±10–15%; treat smaller deltas as noise. `iter()` is ~3× faster than per-block `get`, so the mesher should iterate. Compare only runs made without parallel builds (the implementer's 3.67 µs ran alongside other builds; ignore it).

## Decisions

- 2026-10-14: Reverse-Z infinite projection (Depth32Float, clear 0, compare Greater) and camera-relative rendering (camera pos f64, geometry offset in f64 then cast) are renderer invariants, chosen for Stage 2 view distances.
- 2026-10-14: Cube palette lookup: linear scan up to 16 entries, local Fx-style HashMap above. `Cube::compact()` is explicit (after bulk edits / before save), never automatic on `set`.
- 2026-10-14: Client↔server boundary is `jbm_core::link::Link<Out, In>`; in-process uses `std::sync::mpsc`. Server thread is owned by `ServerHandle` (join on drop, panics surface in `stop`).

- 2026-10-14: Crate packages are prefixed `jbm-` (a package named `core` would clash with Rust's `core`). Dirs stay `crates/core` etc.
- 2026-10-14: Workers run through `scripts/delegate.sh` (the `glm` CLI directly, in the background) rather than the haiku relay agent, so long Rust builds aren't cut off by the relay's 10-minute Bash timeout.
- 2026-10-14: Screenshot tests compare with a tolerance and skip when no GPU adapter exists, unless `JBM_REQUIRE_GPU=1` (set in CI on Linux with Mesa lavapipe).

## Known issues

- client/render/texture_array.rs: mip box filter applies the sRGB curve to alpha too. Fix in 1.4 before translucent textures (filter alpha linearly).
- client/render/overlay.rs: `layout` and `glyph_count` duplicate the char filter (keep in sync or return both counts); `FrameStats` grows unbounded on repeated zero-dt frames (unreachable in app); `render` trusts `set_viewport`.
- server/run.rs: stop/drop can wait up to one tick (≤50 ms) while the thread sleeps. Fine until shutdown latency matters.
- Windowed app path has no automated test (workers' sandbox can't open a window). The orchestrator launches `cargo run -p jbm-game` before merging any wave that touches `client/src/app.rs`; this caught a redraw loop that stalled on macOS.
- First release/bench build is slow (wgpu + thin LTO + codegen-units=1). Revisit profile if CI bench-smoke gets too slow.

## Log

### 2026-10-14
- Repo created. 0.3 done: CLAUDE.md, docs/design.md, docs/plan.md, docs/progress.md. 0.4 drafted: docs/briefs/, scripts/delegate.sh (validated by running 0.1 through it).
- 0.1 merged: Cargo workspace (core, server, client, modhost, game, testkit), toolchain 1.98.1, edition 2024. Implementer + reviewer briefs worked end to end (reports in format), so the 0.4 brief check passed for implementer/reviewer; bench brief gets exercised by 0.6.
- Wave B merged on branch stage-0/wave-b: 0.2 CI workflow, 0.5 screenshot harness (wgpu 30.0.1, golden triangle checked by orchestrator), 0.6 TickClock + HeadlessServer + criterion 0.8.2. All reviewers: MERGE.
- Repo: github.com/justAnArthur/just-better-minecraft (private).
- 0.4 done: bench brief exercised via bench-0 (report in format). Baseline numbers recorded above.
- **Stage 0 gate passed.** PR #1 merged with CI green on macOS and Ubuntu; the triangle golden (made on Metal) matches on Mesa lavapipe in CI, so screenshot tolerances work across backends.
- Stage 1 wave A merged on branch stage-1/wave-a: 1.1 cube storage + palette + block registry, 1.2 window/fly camera/sky/overlay/texture array (goldens checked by orchestrator; binary launched and ran clean), 1.3 protocol + Link + Server tick phases + real-time server thread + 100-tick headless test. All reviewers: MERGE. 92 tests green.
- Fix during review: redraw loop stopped after a skipped (Occluded) first frame, leaving a blank window on macOS; now every skipped frame requests the next, and the loop pauses on `WindowEvent::Occluded(true)`. User confirmed the sky renders.
- bench-1 baselines recorded above. Repo setting "Allow auto-merge" turned on (user approved).
- **Stage 1 wave A done.** PR #2 merged with CI green on macOS and Ubuntu (new sky goldens match on lavapipe).
