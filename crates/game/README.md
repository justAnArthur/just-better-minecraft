# jbm-game

The `jbm` binary: starts the client and an in-process server talking over
a channel (see docs/design.md, "Architecture overview").

**Wiring (task 1.4):** `main` spawns the server with
`jbm_server::run::spawn(Server::new())` (dedicated `"jbm-server"` thread),
builds an `in_process_pair()` link, hands the server end to the thread via
`handle.connect(server_end)`, and runs
`jbm_client::app::run_with_view(client_end, view)` (task 2.9; see
"Far-look wiring" below)
— the window, the link polling, the mesh pool and the terrain renderer.
When the window closes (or the server drops the link), `app::run` returns
and `handle.stop()` joins the server thread. The server thread is never
detached; stopping it propagates a server panic rather than swallowing it.

`Server::new()` streams generated terrain around the player (task 1.5); it
runs the default `Overworld` worldgen preset (task 2.8's continents,
mountain ranges and sea). Every test, bench and example here that depends
on the pre-2.8 terrain pins `ServerConfig { worldgen: Preset::Classic }`
(`terrain_screenshot`, `edit_screenshot`, `edit_e2e`, `prediction`,
`terrain_paths`, the draw benches and the cube-size metrics) so their
goldens and height assumptions keep their meaning; `lod_fly` alone uses
the new `Overworld`, as the real-flight check.

**Far-look wiring (task 2.9):** `main` derives a
`jbm_client::app::ViewSetup` from its `ServerConfig` before the config
moves into the server, via `ServerConfig::ring_radii()` (the LOD0 ring
radius and the furthest ring radius, in blocks), and runs the client
with `app::run_with_view`. The
client cannot learn these from the link; its terrain pass scales the
texture→flat fade and the distance-fog windows on them (see the client
README, "The far look").

**World persistence (task 2.3):** the server saves into `saves/default`
under the current directory. `JBM_WORLD_DIR=<path>` overrides the
directory, and `JBM_NO_SAVE=1` disables persistence entirely (nothing is
written; edits last only for the session). A storage error — the world is
locked by another process, or was written by a newer format — prints one
line to stderr and exits with status 1 instead of panicking. Quitting the
window stops the server thread, which flushes every accepted edit to disk
before the process exits; `/saves` is git-ignored.

**End-to-end screenshot:** `tests/terrain_screenshot.rs` is the one test
that crosses both halves. A `HeadlessServer` (inline generation, R = 3,
V = 2) streams to a `TestClient`; every message goes into a real
`ClientWorld`, each cube is meshed and the view from spawn is rendered
offscreen against the `terrain_spawn` golden (stored with the client's
goldens in `crates/client/tests/golden/`). It skips without a GPU, like
the client's screenshot tests, and is 32³-only (cfg-gated off under the
`cube16` feature).

**Edit tests (task 1.7):** `tests/edit_e2e.rs` (no GPU) breaks a surface
block through the protocol and checks that after one tick the client
world has air there, the dirtied cubes match the touch rule, and edit
cubes jump the mesh queue. `tests/edit_screenshot.rs` builds a 3×3,
2-deep pit and a glass and plank pillar next to spawn, all through
server messages, then renders it with the target outline and crosshair
against the `edit_spawn` golden. The golden tolerance (0.5% of pixels)
is larger than the whole outline, so the test also renders without the
outline and requires at least 20 pixels to differ.

**Cube-size benchmark (task 1.8):** `benches/cube_size.rs` (group
`cube_size`) measures the 128×96×128-block terrain region built in
`benches/common/mod.rs` from `Generator::surface_height` and the 1.5
layering (`jbm_client::testing::heightfield::layer_state`, air kept
local) — no server `generate` involved. The region's horizontal side is
`JBM_REGION_SIDE` (blocks, centered on x = z = 64, a positive multiple of
the cube size) when set — e.g. `JBM_REGION_SIDE=544` runs the benches and
metrics over a game-sized 17×3×17-cube region at 32³ — and 128 otherwise;
the metrics example's header prints the bounds actually used:

- `mesh_region` — meshing every region cube serially;
- `remesh_edit` — meshing the one cube holding block (64, h(64,64), 64),
  the single-edit latency floor (a border edit re-meshes up to 8 cubes via
  neighbor dirtying);
- `build_region` — building the region's cubes with `Cube::from_fn`.

**Terrain draw benchmark (tasks 2.2 + 2.5):** `benches/terrain_draw.rs`
(group `terrain_draw`, same region, GPU required) records and submits one
offscreen frame followed by a device poll wait, through each draw path,
after 3 warm-up frames so the two-phase occlusion state is steady:
`gpu_cull` (compute frustum + Hi-Z culling, two-phase MDI; skipped with a
printed line when the adapter lacks it), `mdi` (2.2's CPU-culled
multi-draw indirect), `fallback` (the forced direct-draw loop),
`cave/cpu_cull` and `cave/gpu_cull` — a camera inside a closed 8×5×8 room
carved ~20 blocks under the surface (`Region::carve_cave` in
`benches/common/mod.rs` asserts in setup that the camera sits in air and
the room is closed), where the occlusion culling earns its keep. Without
a GPU adapter the bench prints a skip line and passes, so the CI bench
smoke run stays green.

The benches and the metrics example select every path explicitly, so
they measure gpu-cull even though the game defaults to cpu-cull. To
play on gpu-cull, run `JBM_GPU_CULL=1 cargo run -p jbm-game --release`
(the overlay names the active path).

**LOD rings (task 2.6).** `terrain_draw` adds `rings/cpu_cull` and
`rings/gpu_cull`: the rings scene at game scale (default
`lod_view_radius`, 6 levels since task 2.10) built in
`benches/common/rings.rs` —
levels 0–1 from the real generator (level 1 by the core downsampling
rule over level-0 data), levels 2–5 sampled from the generator's
surface at each level's resolution (the shared
`jbm_client::testing::heightfield` grid and `max_over_cell`), every
cube pre-meshed with
same-level neighbors and uploaded through the real coverage rule.
Setup asserts every level has resident **and** drawn cubes (the
coverage rule accepted the rings). `examples/cube_size_metrics.rs`
prints the same scene per path (cull counts, frame time) and per level
(resident, drawn, masked octants, faces, skirt faces, face-page bytes).
Measured on Apple M5 Pro (Metal), frame mean including the GPU wait:
`rings/cpu_cull` ≈ 621 µs vs `rings/gpu_cull` ≈ 948 µs — at ring scale
too the extra passes cost more than the occlusion saves, so cpu-cull
stays the default.

**Autopilot (task 2.6 gate).** `JBM_AUTOFLY=<blocks> cargo run -p
jbm-game --release` turns the app into an autopilot: once spawned it
toggles flying, climbs 40 blocks above spawn, flies straight at sprint
speed for `<blocks>` blocks through the normal `PlayerInput` path
(prediction and the server both see a real player), then exits,
printing a frame-time summary (count, mean, p50, p95, p99, max, frames
over 33 ms and over 50 ms).

**Surface-following windows (task 2.10).**
`tests/surface_windows.rs` covers the 2.10 streaming fix on the real
overworld: it deterministically finds a valley with a ≥90-block wall
within 160 blocks (the play-test's geometry), streams a threaded server
there and asserts (1) every surface cube of the LOD0 disk — sampled
five blocks per cube column — was sent, and (2) with the client's level
tint on, no pixel whose depth places it within 200 blocks of the camera
is drawn by a LOD ring (per-pixel distance via `DepthTarget::read_depth`,
reverse-Z `distance = NEAR / depth`); farther ring pixels are legitimate
and only counted. The ignored `surface_window_measurements` test prints
the task's before/after numbers (cubes sent per level, settle time,
memory) at spawn and on a mountain with `max_column_cubes` 0 (the
pre-2.10 fixed windows) and 16 (the default).

**LOD fly test (task 2.6).** `tests/lod_fly.rs` (GPU, `#[ignore]`d —
slow real-server flight; run with `cargo test -p jbm-game --test
lod_fly -- --ignored`) flies a real client against a threaded real
server: 1.5 km straight, then a 90° turn for 300 m, 40 blocks above the
terrain at pitch −15°, then (task 2.10) 600 m low along a valley
(20 over the terrain, cliffs towering over the camera) and one final
stop looking straight down from 400 blocks. Every 100 blocks it waits
(20 s cap) for the
client mesh queue to be clear for a run of ticks (server ring readiness
is never checked — whatever arrived and meshed by then is what the
check sees), then checks the sentinel pixels below the horizon
margin; it also
renders one frame *without* waiting at every stop and checks the middle
half of it, well inside 512 blocks of the camera (farther counts are
only reported). The check tolerates pinholes — 4-connected sentinel
components of at most 2 pixels, at most 8 such pixels per frame —
because the LOD0 greedy mesher's T-junctions leave isolated one-pixel
cracks (documented in
`crates/client/src/render/shaders/terrain.wgsl`, whose sub-pixel
expansion skips LOD0 to keep the pre-LOD goldens bit-stable). Anything
bigger, e.g. the strip of a ring-border crack, fails, with a debug
image highlighting the offending components; eliminating the
T-junctions in the mesher is a follow-up. A plain unit test covers the
detector on synthetic frames, no GPU needed. Screenshots land under
`target/fly/`. The whole run is bounded by a 10-minute timeout that
fails the test instead of hanging it.

`examples/cube_size_metrics.rs` prints the matching metrics table (cube
count, uniform cubes, the old one-draw-per-non-empty-cube number, faces,
mesh bytes, cube storage + a documented HashMap overhead estimate,
streaming-granularity message count) plus, when an adapter exists, a GPU
section per path **and scene** (surface view and the cave room): the cull
counters (`drawn / frustum-culled / occluded`, `candidates / commands /
faces`), the encoded indirect command-region slots including the
zero-count tails (gpu-cull, both phases), draw calls / pages, the mean
frame time over 50 frames after a 3-frame warm-up, and that mean split
into CPU encode+submit vs the device-poll wait (where the GPU-side costs
— the Hi-Z rebuild above all — show up). Both run at 32³
and, with `--features cube16`, at 16³:

```bash
cargo bench -p jbm-game --bench cube_size [--features cube16]
cargo bench -p jbm-game --bench terrain_draw
cargo run -p jbm-game --release --example cube_size_metrics [--features cube16]
JBM_REGION_SIDE=544 cargo bench -p jbm-game --bench terrain_draw
JBM_REGION_SIDE=544 cargo run -p jbm-game --release --example cube_size_metrics
```

**Draw-path tests (tasks 2.2 + 2.5):** `tests/terrain_paths.rs` renders
the streamed `terrain_spawn` and `edit_spawn` scenes through the gpu-cull,
cpu-cull and fallback paths and requires **pixel-identical** images
(exact, not the golden tolerance; without adapter MDI it checks the
fallback against the golden instead), and asserts the terrain stats on
`terrain_spawn` (draw calls ≤ 2·pages on MDI, one to six commands per
drawn cube, faces below the drawn cubes' total — directions really are
skipped). Task 2.5 adds: a 60-frame moving camera (turning on the spot,
flying down into a carved closed room, looking around inside, rising
back out) that must render gpu-cull ≡ cpu-cull **at every frame** with
the same pass object (two-phase state is real) and with `occluded > 0`
on some frames — the anti-popping test; a cave test (camera in the
closed room, 3 frames, ≥80% of frustum-passing cubes occlusion-culled,
image equal to cpu-cull); an open-surface test (`occluded` stays small,
image equal to cpu-cull); and an odd-resize test (333×217 keeps the
paths identical). Every blocking stats read also asserts
`region_overflow == 0`: the CPU frustum bound must cover every command
the GPU emits. All print their cull counts. The room is carved by
applying `BlockChange` messages through the client world — the same edit
path real block edits take.

**Invariants**

- The only executable crate in the workspace.
- Depends on client, server, and core; client and server never depend on each
  other.
