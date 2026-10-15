# Progress

## Next up

Stage 1 · Voxel sandbox, wave C: 1.6 collision + walk/fly + client prediction with server correction, 1.7 raycast + break/place + re-mesh on change (must pin edited cubes against server eviction), 1.8 16³ vs 32³ bench.

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
| server/tick_streaming_idle | 53.63 ns | Apple M5 Pro (orchestrator; tick_8_clients 299 ns in the same run) | 2026-10-14 |
| mesh/terrain_like | 220.77 µs | Apple M5 Pro (bench-2, idle) | 2026-10-15 |
| mesh/checkerboard | 697.34 µs | Apple M5 Pro (bench-2, idle) | 2026-10-15 |
| mesh/uniform_stone_air_neighbors | 218.93 µs | Apple M5 Pro (bench-2, idle) | 2026-10-15 |
| mesh/padded_build | 20.21 µs | Apple M5 Pro (bench-2, idle) | 2026-10-15 |
| worldgen/heightmap_32x32 | 20.22 µs | Apple M5 Pro (bench-2, idle) | 2026-10-15 |
| worldgen/surface_cube | 92.40 µs | Apple M5 Pro (bench-2, idle) | 2026-10-15 |
| worldgen/air_cube | 146.92 ns | Apple M5 Pro (bench-2, idle) | 2026-10-15 |

Baselines are bench worker runs on an otherwise idle machine. Run-to-run noise on this Mac is ±10–15%; treat smaller deltas as noise. `iter()` is ~3× faster than per-block `get`, so the mesher should iterate. Compare only runs made without parallel builds (the implementer's 3.67 µs ran alongside other builds; ignore it).

## Decisions

- 2026-10-14: Reverse-Z infinite projection (Depth32Float, clear 0, compare Greater) and camera-relative rendering (camera pos f64, geometry offset in f64 then cast) are renderer invariants, chosen for Stage 2 view distances.
- 2026-10-14: Cube palette lookup: linear scan up to 16 entries, local Fx-style HashMap above. `Cube::compact()` is explicit (after bulk edits / before save), never automatic on `set`.
- 2026-10-14: Client↔server boundary is `jbm_core::link::Link<Out, In>`; in-process uses `std::sync::mpsc`. Server thread is owned by `ServerHandle` (join on drop, panics surface in `stop`).

- 2026-10-14: Packed face anchor: a face is stored at its first block along u/v; the quad starts at that block's corner, offset +1 along the normal if positive and +1 along u/v if negative. `mesh::face::quad_corners` mirrors the shader and is tested against the mesher's rasterization.
- 2026-10-14: Mesh versions come from one client-wide counter bumped on every dirtying (not per CubeData), so a mesh job built from older neighbor data is always stale. Missing neighbor cubes count as opaque for face culling, air for AO.
- 2026-10-14: Server streams a cylinder (R=8 horizontal, V=4 vertical) nearest-first with an R+1/V+1 keep set for hysteresis; cubes no player keeps are evicted (regeneration is exact until edits exist).

- 2026-10-14: Crate packages are prefixed `jbm-` (a package named `core` would clash with Rust's `core`). Dirs stay `crates/core` etc.
- 2026-10-14: Workers run through `scripts/delegate.sh` (the `glm` CLI directly, in the background) rather than the haiku relay agent, so long Rust builds aren't cut off by the relay's 10-minute Bash timeout.
- 2026-10-14: Screenshot tests compare with a tolerance and skip when no GPU adapter exists, unless `JBM_REQUIRE_GPU=1` (set in CI on Linux with Mesa lavapipe).

- 2026-10-15: **Cube size stays 32³** (task 1.8). Same 128×96×128 terrain region, M5 Pro under light load:

  | | 32³ | 16³ |
  |---|---|---|
  | mesh whole region | 5.65 ms | 8.70 ms (+54%) |
  | re-mesh one edited cube | 236 µs | 55 µs |
  | build region (gen proxy) | 67 ms | 79 ms |
  | draw calls (region) | 24 | 82 |
  | cubes to stream, 256-block view | 1,773 | 13,549 |
  | cube memory (region) | 240 KB | 132 KB + 8× map entries |
  | GPU frame (region, 720p) | 0.71 ms | 0.92 ms |

  16³ wins only single-edit latency and storage of mostly-uniform space. A 236 µs re-mesh, even ×8 for a corner edit, is far inside a frame on the worker pool. Every per-cube cost scales 8× at 16³: messages, map entries, draws, and later LOD nodes and redb keys. The `cube16` feature stays for re-measuring, but CI does not build it, so expect it to rot.

## Known issues

- client/render/texture_array.rs: mip box filter applies the sRGB curve to alpha too. Fix in 1.4 before translucent textures (filter alpha linearly).
- client/render/overlay.rs: `layout` and `glyph_count` duplicate the char filter (keep in sync or return both counts); `FrameStats` grows unbounded on repeated zero-dt frames (unreachable in app); `render` trusts `set_viewport`.
- server/run.rs: stop/drop can wait up to one tick (≤50 ms) while the thread sleeps. Fine until shutdown latency matters.
- Windowed app path has no automated test (workers' sandbox can't open a window). The orchestrator launches `cargo run -p jbm-game` before merging any wave that touches `client/src/app.rs`; this caught a redraw loop that stalled on macOS.
- Server eviction drops cubes no player keeps; 1.7 must pin edited cubes until persistence (2.3), or edits are lost.
- Reviewers passed 1.4 with three visible bugs (one-block face offset for +X/−Z, stale mesh race, magenta grass side). The orchestrator must keep inspecting goldens zoomed in, and golden cameras must be close enough to show per-block detail.
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
- Stage 1 wave B merged on branch stage-1/wave-b: 1.4 greedy mesher + mesh pool + packed faces + vertex pulling + client world + game wiring, 1.5 noise terrain + gen pool + cube streaming. Both reviewers: MERGE. Orchestrator fixed during review: quad corner offset for negative-u faces (sky through step corners), stale mesh versions on neighbor arrival, missing grass_side texture; moved golden cameras closer. Protocol v2 (CubeData, UnloadCube, Welcome.spawn) and content `PartialEq` on Cube landed on main first so both tasks shared it.
- User played the merged build: terrain renders and streams.
- Wave B follow-ups: workspace crates build at opt-level 1 in dev (client unit tests 32 s → 4 s, workspace suite ~8 s); end-to-end terrain screenshot (`crates/game/tests/terrain_screenshot.rs`, golden `terrain_spawn`, inspected: terraces, grass sides, sand, no holes). Mesh/worldgen baselines deferred: machine never idle.
- bench-2: mesh/worldgen baselines from an idle run (user ran them with other apps closed). terrain_like 221 µs, inside the ≤300 µs target; the earlier 2× numbers were load.
- 1.8 merged on stage-1/wave-c: `cube16` feature (cube-size literals now derive from `CUBE_SIZE`), `cube_size` bench and `cube_size_metrics` example in jbm-game. Decision: keep 32³ (see Decisions).
- 1.6 merged on stage-1/wave-c: shared player physics in core (swept AABB, Y-X-Z, exact face contact, freeze at unloaded blocks, exact ballistic integration so a jump apex is v²/2g = 1.27), protocol v3 (`PlayerInput { seq, input }`, `PlayerState { ack_seq, state }`), input-driven server stepping (≤4 inputs per tick, queue 20), client `Predictor` with replay on ack and 20 Hz fixed sim plus render interpolation. Reviewer: one finding (an orchestrator probe left in the tree). The orchestrator added a wall-contact test at offsets up to 10⁷. physics/step_walk_flat ≈12.5 ns.
