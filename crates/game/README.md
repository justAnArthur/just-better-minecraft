# jbm-game

The `jbm` binary: starts the client and an in-process server talking over
a channel (see docs/design.md, "Architecture overview").

**Wiring (task 1.4):** `main` spawns the server with
`jbm_server::run::spawn(Server::new())` (dedicated `"jbm-server"` thread),
builds an `in_process_pair()` link, hands the server end to the thread via
`handle.connect(server_end)`, and runs `jbm_client::app::run(client_end)`
— the window, the link polling, the mesh pool and the terrain renderer.
When the window closes (or the server drops the link), `app::run` returns
and `handle.stop()` joins the server thread. The server thread is never
detached; stopping it propagates a server panic rather than swallowing it.

`Server::new()` streams generated terrain around the player (task 1.5).

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
layering — no server `generate` involved:

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

`examples/cube_size_metrics.rs` prints the matching metrics table (cube
count, uniform cubes, the old one-draw-per-non-empty-cube number, faces,
mesh bytes, cube storage + a documented HashMap overhead estimate,
streaming-granularity message count) plus, when an adapter exists, a GPU
section per path **and scene** (surface view and the cave room): the cull
counters (`drawn / frustum-culled / occluded`, `candidates / commands /
faces`), the encoded indirect command-region slots including the
zero-count tails (gpu-cull, both phases), draw calls / pages, and the
mean frame time over 50 frames after a 3-frame warm-up. Both run at 32³
and, with `--features cube16`, at 16³:

```bash
cargo bench -p jbm-game --bench cube_size [--features cube16]
cargo bench -p jbm-game --bench terrain_draw
cargo run -p jbm-game --release --example cube_size_metrics [--features cube16]
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
paths identical). All print their cull counts. The room is carved by
applying `BlockChange` messages through the client world — the same edit
path real block edits take.

**Invariants**

- The only executable crate in the workspace.
- Depends on client, server, and core; client and server never depend on each
  other.
