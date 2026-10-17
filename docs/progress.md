# Progress

## Next up

Stage 2 · Far view, wave E (redesign after the wave D play-test, in this order): 2.11 code review and restructuring (done); 2.12 playground with an egui debug panel (done); 2.13 meshing optimization and a larger full-detail radius so LOD draws only far away; 2.14 height-exact Distant-Horizons-style LOD. Then the Stage 2 gate (fly across kilometres without hitches, you approve the look). Wave D merged (2.8 overworld, 2.9 far look, 2.10 LOD on tall terrain).

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
| worldgen/surface_cube (graph) | 173.8 µs vs reference 135.2 µs (1.29×) | Apple M5 Pro (2.1 reviewer, parallel builds) | 2026-10-15 |
| worldgen/sample_grid/{1..32} | ≈310 µs | Apple M5 Pro (2.1 implementer, parallel builds) | 2026-10-15 |
| storage/encode_surface_cube | 9.37 µs | Apple M5 Pro (orchestrator, light load) | 2026-10-15 |
| storage/decode_surface_cube | 30.15 µs | Apple M5 Pro (orchestrator, light load) | 2026-10-15 |
| storage/save_batch_64 | 4.00 ms | Apple M5 Pro (orchestrator, light load) | 2026-10-15 |
| storage/load_cube | 23.83 µs | Apple M5 Pro (orchestrator, light load) | 2026-10-15 |
| suballoc/churn_10k | ≈308 µs | Apple M5 Pro (2.2 implementer) | 2026-10-15 |
| terrain_draw/mdi, /fallback | ≈370 µs, ≈329 µs | Apple M5 Pro (2.2 implementer) | 2026-10-15 |
| lod/insert_lod0/surface | ≈102 µs (worldgen/surface_cube 160 µs in the same run) | Apple M5 Pro (orchestrator, heavy load) | 2026-10-15 |
| lod/edit (one real change, 4 levels) | ≈13.6 µs | Apple M5 Pro (orchestrator, heavy load) | 2026-10-15 |
| lod/downsample_octant/surface | ≈80 µs | Apple M5 Pro (orchestrator, heavy load) | 2026-10-15 |
| terrain_draw/gpu_cull, /mdi (cpu-cull), /fallback | ≈380, ≈248, ≈211 µs | Apple M5 Pro (2.5 implementer) | 2026-10-15 |
| terrain_draw/cave/gpu_cull, /cave/cpu_cull | ≈386, ≈257 µs | Apple M5 Pro (2.5 implementer) | 2026-10-15 |

Baselines are bench worker runs on an otherwise idle machine. Run-to-run noise on this Mac is ±10–15%; treat smaller deltas as noise. `iter()` is ~3× faster than per-block `get`, so the mesher should iterate. Compare only runs made without parallel builds (the implementer's 3.67 µs ran alongside other builds; ignore it). The wave A worldgen rows ran during three parallel worker builds: the graph/reference ratio is meaningful, the absolute numbers are not (idle baseline surface_cube was 92 µs). Re-run worldgen idle before 2.7.

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

- 2026-10-15: Terrain is data: a JSON density graph (`crates/server/assets/worldgen/overworld.json`) compiled to a flat register program with constant folding, xz-only instructions run once per column, interval bounds for fast paths and a per-column "band" solidity cache. Solid means density ≥ 0; grass/dirt/sand come from a code surface stage (depth below air). It reproduces the 1.5 terrain bit for bit; the old code stays as `worldgen::reference` for the equivalence tests until a later stage deletes it.
- 2026-10-15: Terrain draws from shared 32 MiB face pages with a first-fit suballocator and one `multi_draw_indirect` per page; one indirect command per visible cube and face direction, back-facing directions skipped (a ≥1-block margin keeps f32 rounding safe). Fallback (`JBM_NO_MDI=1`, or no INDIRECT_EXECUTION / INDIRECT_FIRST_INSTANCE) issues the same commands as direct draws; images are pixel-identical. Region draw calls 24 → 1.
- 2026-10-15: LOD downsampling: a 2×2×2 group is filled when at least 2 of 8 children are non-air; its material is the most common among the topmost filled child of each of the 4 columns (ties: higher dy, then smaller id). The rule is local to the group, so an edit changes at most one cell per level and incremental propagation equals a full rebuild exactly. The server keeps levels 1–4 (`lod_levels`) in memory, fed by every generated/loaded cube and every edit; LOD cubes outlive their LOD0 children.
- 2026-10-15: GPU frustum + two-phase Hi-Z occlusion culling exists and is tested pixel-identical to CPU culling on every frame of a moving camera, but **cpu-cull is the default**; gpu-cull is opt-in (`JBM_GPU_CULL=1`). On Metal it was slower in every measured scene (game-sized region: surface 0.80–1.07 vs 0.43–0.61 ms, cave 0.41–0.71 vs 0.27–0.46 ms): `multi_draw_indirect` is a CPU loop there, `draw_indirect_count` is missing, and the per-frame Hi-Z rebuild costs ~0.18 ms. Re-measure in 2.6 with LOD rings; levers in the client README.
- 2026-10-15: Worlds persist in `saves/default/world.redb` (`JBM_WORLD_DIR`, `JBM_NO_SAVE=1`). Only edited cubes are stored (zstd'd palette + words, on-disk id table for remapping); unedited cubes regenerate. A `jbm-save` thread commits one transaction per batch every 5 s and on drop; cubes stay pinned until their latest edit version is committed, then evict normally. The gen pool is load-or-generate. Players (by name) are saved with each batch and on disconnect, and rejoin at their saved position, look and flight (protocol v5 `Welcome { spawn, yaw, pitch, flying }`).

## Known issues

- client/render/texture_array.rs: mip box filter applies the sRGB curve to alpha too. Fix in 1.4 before translucent textures (filter alpha linearly).
- client/render/overlay.rs: `layout` and `glyph_count` duplicate the char filter (keep in sync or return both counts); `FrameStats` grows unbounded on repeated zero-dt frames (unreachable in app); `render` trusts `set_viewport`.
- server/run.rs: stop/drop can wait up to one tick (≤50 ms) while the thread sleeps. Fine until shutdown latency matters.
- Windowed app path has no automated test (workers' sandbox can't open a window). The orchestrator launches `cargo run -p jbm-game` before merging any wave that touches `client/src/app.rs`; this caught a redraw loop that stalled on macOS.
- Without a world dir (`world_dir: None`, all tests by default) edited cubes stay pinned forever, as before 2.3. With persistence they unpin once saved.
- server: `Drop for Server` joins `jbm-save` after `flush`'s 30 s timeout, so a hung commit can still block shutdown. Bound the join if it ever matters.
- client: face pages are never freed or compacted; fragmentation grows with churn until a compaction task (2.6 or later).
- LOD pyramid has no in-memory eviction; it grows with the explored area. Stored LOD (2.7) means evicted cubes could be dropped and reloaded on request, but nothing evicts yet.
- LOD0 greedy-mesh T-junction pinholes: 1–6 single sky pixels per frame (terrain.wgsl skips the edge expansion at LOD0 to keep goldens and path identity exact). Hole tests tolerate them (fly: components ≤ 2 px, ≤ 8 px per frame; synthetic rings scene: ≤ 3 / 12). Eliminating T-junctions in the mesher is a follow-up; it changes LOD0 meshes and goldens.
- A LOD ring boundary crossing costs ~7.3 ms on the tick thread (`lod/stream_moving`: center change, cursor reset, keep-set unloads), about every 3 s at sprint speed. Fine within the 50 ms tick; make it incremental if ticks get tight.
- `sample_lod` for a mixed cube is ~22× `surface_cube` (per-cell density in the band instead of the band bit path). The default rings still fill in ~1 s threaded; move it to the band bit evaluator if fill time matters.
- Stored LOD loads serialize on one redb read transaction per cube (~160 µs), so a reload fills the rings in ~0.9 s, barely faster than sampling. Batch loads in one read transaction to make reload really instant.
- `physics::step` returns the pre-toggle state when a block in the swept box is unloaded, so a fly toggle sent while the spawn cube generates is consumed and lost. The autopilot and the fly test retry until the server confirms; real players just press again. Fix in physics (apply toggles even when movement is frozen).
- LOD ingest runs on the tick thread: ~0.7× the generation cost of a surface cube (~88 µs against 124 µs; the level-1 downsample itself moved to the gen pool workers in 2.7).
- gpu-cull rebuilds a full-resolution Hi-Z pyramid every frame; start it at half resolution if gpu-cull becomes the default.
- `worldgen::reference` (the 1.5 noise code) is kept only for equivalence tests; delete it once LOD sampling (2.7) has its own coverage.
- Golden tolerance (0.5% of pixels differing) is larger than thin features such as the target outline (~40–140 px). Tests for thin features must also assert directly: `edit_screenshot` renders with and without the outline and requires a difference.
- Reviewers passed 1.4 with three visible bugs (one-block face offset for +X/−Z, stale mesh race, magenta grass side). The orchestrator must keep inspecting goldens zoomed in, and golden cameras must be close enough to show per-block detail.
- LOD surfaces lean high: coarse top minus true top averages +0.6 (L1), +1.7 (L2), +4.0 (L3), +9.7 (L4), +19.8 (L5), +44 (L6) blocks, p99 about 1.5 cells (2.10 measurement). The downsampling rule (2 of 8 children) is what the skirts rely on; 2.14 replaces it with height-exact columns.
- Level ≥ 1 quads overlap by 1/512 block at convex edges (the T-junction expansion), so draw paths may disagree on an exact edge pixel; the rings path-identity test allows 2.
- Ring surface spans are exact for LOD0, sampled 8×8 for L1 and skipped for L2+ (their camera windows cover the overworld's height range).
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
- 1.7 merged on stage-1/wave-c: raycast (DDA, `REACH = 5`), protocol v4 (`BreakBlock`, `PlaceBlock`, `BlockChange`), server validation (reach + 1 margin, air rules, player overlap) with revert-to-sender, `BlockChange` flushed before streaming and `Tick` (same-tick delivery), pinned edited cubes, `or_insert` for generation results. Client: optimistic edits, priority re-mesh queue, hold-to-repeat, 1–9 hotbar, crosshair, depth-tested target outline, golden `edit_spawn`. raycast/reach_5_diagonal ≈ 58 ns. The GLM worker hit its Z.ai usage limit before finishing, so the orchestrator finished it:
  - fixed the outline: it was invisible (40% black on dirt), and its edges z-fought into dashes; fix is a vertex pull toward the camera and 60% alpha;
  - moved the golden camera closer;
  - added the outline visibility check;
  - made touched border neighbors share edit priority;
  - skipped re-meshing on a no-op `BlockChange` (the server's echo of an accepted edit);
  - `Arc::make_mut` in `set_block`;
  - released held edit buttons on focus loss and Esc;
  - fixed clippy errors and wrote the READMEs.
- Stage 1 gate passed: the user played the wave C build (walk, fly, break, place). Movement stuttered at 20 Hz because every server ack reset the render interpolation; fixed in fix/interp-snap (agreeing acks leave it alone, corrections shift both endpoints). The user confirmed it is smooth.

### 2026-10-15
- Stage 2 wave A specs written (docs/tasks/2.1–2.3). All three GLM implementers ran in parallel on branch stage-2/wave-a; no Z.ai limit hit. Reviewers: MERGE ×3.
- 2.2 merged: paged face buffers + suballocator, multi-draw indirect with per-direction commands and back-face direction skipping, direct-draw fallback, path-equality and slot-reuse GPU tests, `terrain_draw` bench. Orchestrator also fixed a client test that failed under `cube16` (literal 32).
- 2.1 merged: density graph format + compiler + evaluator, shipped overworld graph, exact equivalence with the 1.5 code (3 seeds, 512² + far columns), `sample_grid` at LOD steps, goldens untouched. The worker caught a spec error (`y − 70` is solid above 70 under "≥ 0 is solid"). Orchestrator fixed `surface_height` scanning the shipped graph's range instead of the graph's own band, with a test.
- 2.3 merged: redb storage + cube codec in core + background saver + load-or-generate, kill-during-save tests (storage-level atomicity over 10 SIGKILL rounds, server-level prefix property). Orchestrator: decode builds the cube from the stored words instead of re-packing (load 211 → 24 µs), bulk zstd frames, typed corrupt-meta error, fixed the load bench (it measured a uniform cube), resolved merge conflicts with 2.1/2.2, and made the held-save tests deterministic after they flaked under the parallel workspace run (one attempted fix hung the test binary; now a `release_one_save` hook, a `failed_batches` counter and a release guard).
- Play-test of wave A: no holes, edits survive restart and flying away and back. Position and look were not kept (player data was out of 2.3's scope); the orchestrator added player records (`players` table, protocol v5 `Welcome` with look and flight, departures captured on disconnect) with storage and end-to-end tests, plus a regression test for an edit restored from disk surviving eviction with threaded generation.
- Stage 2 wave B specs written (docs/tasks/2.4, 2.5). Both GLM implementers hit the Z.ai 5-hour limit mid-task and resumed from their uncommitted work after the reset; the last 2.5 round hit the weekly limit and the orchestrator finished it.
- 2.4 merged: LOD rule in core (`lod.rs`: `downsample_cell`, octant/region API), `LodPyramid` in the server (levels 1–4, `known` octant mask, ingest on collect, one-cell-per-level edit path), thin-wall/trunk/terrain tests, incremental-equals-rebuild over 2,000 edits ×10 seeds. Orchestrator: octant downsample ~2× faster (unpack once, all-equal fast path, `Cube::compact` skips the repack when nothing is unused), both edit benches had measured no-op edits (~100 ns) and now force a real change (~13 µs), fixed `codec_rejects_bad_headers` under `cube16` (broken since 2.3). design.md's downsampling paragraph now states the rule.
- 2.5 merged: meta slots + one origin formula for all paths, Hi-Z pyramid with odd-size folding, two-phase compute culling with per-page regions, CPU frustum bound on region sizes, `Renderer::render(gpu, FrameInputs)`. Tests: three-path pixel equality, 60-frame no-popping, cave ≥80% occluded (93%), odd resize, churn. Orchestrator measured a game-sized region, found gpu-cull slower on Metal, and made cpu-cull the default (see Decisions); fixed the path-equality test under `JBM_NO_MDI=1`.
- Play-test of wave B: both the default and `JBM_GPU_CULL=1` render normally, no pop-in.
- CI caught a gpu-cull bug Metal hid: on lavapipe `textureDimensions(pyramid, level)` reported a top mip larger than 1×1, so the Hi-Z clamp let reads run past the mip, returned 0 (infinitely far) and kept occluded cubes (7–10 of 14 instead of 13). Reproduced in a local Ubuntu container (colima + mesa-vulkan-drivers); the cull shader now derives mip sizes from the depth size (`max(1, size >> level)`). Lesson: run GPU tests on lavapipe locally before pushing shader changes; Metal is more forgiving.

### 2026-10-16
- Stage 2 wave C specs written (docs/tasks/2.6, 2.7) after the orchestrator committed the shared interface (protocol v6 LOD messages, ring config, `LodPyramid` streaming API stubs) so both could run in parallel. Both implementers hit the Z.ai 5-hour limit; the orchestrator fixed 2.7's failing tests and lint in between, and both resumed.
- 2.7 merged: `Generator::sample_lod` (level 1 exactly equals the downsample of generated LOD0), the content model (real octants from the downsample of the child's full content, worldgen octants otherwise, `ready` only for complete cubes, deferred pending propagation), `jbm-lod-N` fill workers with LOD0 priority and inline mode, stored LOD in redb (key kind = level, record schema 2 with the known mask, worldgen fingerprint). Review found an edit lost under a not-ready child after a reload (fixed, with a storage-backed property test) and a shutdown hang when a save was held (`Drop` now releases held saves); orchestrator found lost LOD saves while a save was in flight and a per-level cube clone.
- 2.6 merged: per-level ring streaming (full disks, merged walk coarse-first at ties, re-sends first, keep-set unloads), the client LOD world, the exactly-once coverage rule (10k-case property, incremental == from scratch), octant-split meshing with skirts, level-scaled rendering on all three paths, `JBM_AUTOFLY` autopilot. The real-server fly test (1.5 km + 300 m) passes all 18 stops. Its first failures were harness bugs (meshing 3× slower than the app, camera off the server position, a lost fly toggle) plus LOD0 T-junction pinholes, now tolerated (see Known issues). Several GLM sessions edited the 2.6 worktree at once after a launcher retried; one owner session finished it.
- Play-test of wave C: no holes, cracks or flicker at ring borders. Autopilot over 2 km: ~10 ms mean, max ~40 ms, one frame over 33 ms, none over 50 ms. The user found the terrain too flat to show the LOD and the textured far LOD noisy, and chose wave D (2.8 dramatic worldgen, 2.9 Distant-Horizons-style far look) before closing Stage 2.
- Commits now follow scoped Conventional Commits (`type(scope): subject`, crate scopes); release automation reads them on main (PR #12). Wave PRs squash-merge.

### 2026-10-17
- Wave D: both GLM implementers hit the 5-hour limit, were resumed, then hung for ~7 h: a headless `claude -p` never wakes from a background task. The orchestrator finished 2.8/2.9 itself. Briefs now forbid background commands; `glm-watch <dir>` streams a GLM session's thinking and tool calls live.
- Found by the real fly test over the new mountains: slopes crossing a horizontal LOD seam showed a contour line of holes (lids added to the mesher); the sea surface stepped up at every ring border (sea surface moved to y = 64, a multiple of 16); skirt faces carried layer + 32768 (the skirt key bit leaked into the layer field; harmless until 2.9's per-layer color buffer read out of bounds — black on lavapipe).
- 2.10 after the play-test: coarse LOD was drawn next to the player because LOD0 streamed a fixed ±128-block window; windows now follow the terrain surface. Six rings (~16 km), F4 level tint, narrow beaches (~3% sand).
- Play-test fixes: a cube reloaded under a ready ancestor drew again (coverage did not re-report an unchanged "hidden" verdict; two levels z-fought on water); faces between equal non-opaque blocks (water, glass) were all meshed.
- The user does not approve the look yet: LOD too close, coarse shapes inaccurate, wants a playground with debug views and a code cleanup. Wave D merged as a checkpoint; wave E (2.11–2.14) is the redesign.
- 2.11 restructuring (no behavior change), merged so far: f mesher split; a worldgen noise module, `reference` behind a feature, terrain split with one shared surface/LOD column rule; b `server.rs` split into seven submodules; c `render/chunks.rs` split into `render/terrain/*` plus `RenderSettings` (fog, tint, radii and draw-path flags in one place, the renderer owns the radii). Goldens and bench medians unchanged (terrain_draw A/B on one machine within noise). d `TerrainSync` (client streaming state out of `app.rs`), `HudSnapshot`, `testing::OffscreenRig`; e shared helpers (`CubeKey` in core, `ring_radii`, `Camera::looking_at`, testkit RNG and block fixtures, test heightfield). 2.11 complete.
- 2.12 playground merged: egui 0.37 debug panel on F3 (dependency exception; a wgpu upgrade now waits for a matching egui), stats with triangles and per-level resident/drawn/faces, live views (wireframe, wireframe only, chunk borders, exploded cubes, level tint, per-level mask, freeze culling with the frozen frustum drawn, fog, force CPU cull, force fallback) each with an env var, live server stats and view-radius sliders through `DebugHooks`, and `JBM_WORLD=playground` (a fixed debug world that never saves). One ring-radius rule, `jbm_core::lod::ring_radii`. Every view tested on all three draw paths; goldens unchanged; terrain_draw medians no worse (the before-run was under build load). Screenshots in `target/2.12-shots/` of the task worktree.
