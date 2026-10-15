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

`examples/cube_size_metrics.rs` prints the matching metrics table (cube
count, uniform cubes, draw calls, faces, mesh bytes, cube storage + a
documented HashMap overhead estimate, streaming-granularity message count,
and a GPU frame time when an adapter exists). Both run at 32³ and, with
`--features cube16`, at 16³:

```bash
cargo bench -p jbm-game --bench cube_size [--features cube16]
cargo run -p jbm-game --release --example cube_size_metrics [--features cube16]
```

**Invariants**

- The only executable crate in the workspace.
- Depends on client, server, and core; client and server never depend on each
  other.
