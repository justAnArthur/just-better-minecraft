# jbm-client

The player-facing side: renderer, UI, input, audio, and client-side
prediction. Talks to the server only through `jbm-core` protocol messages.

**Public API**

- `gpu::Gpu` / `gpu::GpuError` — GPU acquisition. `Gpu::headless()` for
  tests (no surface, never panics); `Gpu::for_surface(instance, &surface)`
  for the window.
- `camera::Camera` — f64 position, yaw/pitch (pitch clamped to ±89°),
  reverse-Z infinite-far projection, camera-relative `view_proj()` and
  `relative()`.
- `input::InputState` — winit-independent key set, mouse delta
  accumulation and the F-key fly-toggle edge; `input::move_input` maps
  held keys onto the physics `MoveInput`, `input::apply_mouse_look`
  applies the per-frame look.
- `mesh::appearance` — the flat per-state appearance table plus the
  procedural builtin 16×16 placeholder textures (`builtin_textures`).
- `mesh::face` — the 8-byte `PackedFace` format (see below).
- `mesh::greedy` — the mesher (`mesh()`, `MeshInput`, `MeshOutput`) and
  `build_padded()` (bench-only).
- `mesh::pool::MeshPool` — the mesh worker pool.
- `world::ClientWorld` — cubes from the server plus dirty/version state.
- `render::renderer::Renderer` — the frame renderer (sky + terrain +
  overlay dispatch). One `Renderer` per color format; works against any
  target, so the window surface and the offscreen test target share the
  code.
- `render::renderer::DepthTarget` — `Depth32Float` attachment with resize.
- `render::chunks::TerrainPass` — the terrain pass: per-cube face storage
  buffers, camera-relative origins, CPU frustum culling.
- `render::overlay` — debug overlay (`OverlayText`, `OverlayRenderer`,
  `FrameStats`).
- `render::texture_array::BlockTextures` + `placeholder_layers(n)` —
  block texture array with CPU-generated mipmaps (repeat addressing so
  merged faces tile per block).
- `render::offscreen::OffscreenTarget` — an `Rgba8UnormSrgb` render
  target with `read_rgba(&Gpu) -> image::RgbaImage` readback.
- `render::triangle` — the minimal test pipeline.
- `predict::Predictor` — client-side prediction of the shared physics
  (see below).
- `app::run(link)` — opens the window and runs the frame loop against a
  server link.
- `testing::screenshot` (only with `--features testing` or in unit tests)
  — golden-image comparison helpers, see below.

## Invariants

- Depends only on `jbm-core`; must NOT depend on `jbm-server` (enforced by
  `cargo tree -p jbm-client` in CI-style checks).
- Renderer changes need screenshot tests.
- Perf-sensitive code needs a bench (`benches/mesh_bench.rs`).

## Meshing (task 1.4)

### Block appearance table

`BlockAppearance` is built once from the `BlockRegistry` and the builtin
blocks, then shared read-only (`Arc`) with the mesh workers. It is a flat
`Vec` indexed by `BlockStateId.0`; the hot path never calls
`registry.info()` or does string lookups. Per state it holds flags
(air / opaque / cutout) and six texture layers, one per face direction.
`builtin_textures()` generates the eleven 16×16 placeholder layers
(stone, dirt, grass_top, grass_side, sand, water, glass, oak_planks,
oak_log_side, oak_log_top, leaves) with a deterministic hash; glass and
leaves use alpha exactly 0 or 255 for cutout.

**Water is opaque blue for now; the translucent water pass is a later
task.** (Non-opaque in the registry, so it is not culled against — this
makes water bodies render their internal faces until translucency lands.)

### Packed face format (8 bytes)

`PackedFace(u32, u32)`; the bit layout lives in one comment block in
`mesh/face.rs` mirrored by `render/shaders/terrain.wgsl` — keep the two in
sync. Word 0: x (5), y (5), z (5), width−1 (5), height−1 (5), direction
(3), spare. Word 1: texture layer (16), AO 2 bits × 4 corners, light 4
bits (always 15 until lighting exists), spare.

Width runs along the face's **u** axis, height along its **v** axis:

| dir | value | normal | u axis | v axis |
|---|---|---|---|---|
| +X | 0 | ( 1,0,0) | ( 0,0,−1) | (0,1,0) |
| −X | 1 | (−1,0,0) | ( 0,0, 1) | (0,1,0) |
| +Y | 2 | ( 0,1,0) | ( 0,0, 1) | (1,0,0) |
| −Y | 3 | ( 0,−1,0)| ( 1,0, 0) | (0,0,1) |
| +Z | 4 | ( 0,0,1) | ( 1,0, 0) | (0,1,0) |
| −Z | 5 | ( 0,0,−1)| (−1,0, 0) | (0,1,0) |

`cross(u, v) = normal`, so quad corners c0=(0,0), c1=(1,0), c2=(1,1),
c3=(0,1) wind counter-clockwise viewed from outside (back-face culling
is on). AO corners are packed in c0..c3 order. The vertex shader picks
the triangle diagonal along the darker AO pair to avoid interpolation
anisotropy.

### Face rule

A face of block A toward neighbor B is emitted when A is not air, B is
not opaque, and not (A == B and A is cutout) — so glass-glass and
leaves-leaves interiors are culled. A **missing neighbor cube counts as
opaque** for this rule (no walls facing unloaded space) and as **air for
AO** (unloaded space reads as open). Uniform fast paths: uniform air
meshes to nothing without building anything; a uniform opaque cube whose
six face neighbors are all present, uniform and opaque meshes to nothing.

### AO rule

Per corner, 0 (dark) ..= 3 (open), classic side1/side2/corner:
`if side1 && side2 { 0 } else { 3 − (side1 + side2 + corner) }`, sampled
from the eight cells around the block in front of the face. Opaque blocks
occlude; cutout and air do not.

### Greedy merge

The mesher first builds a padded 34³ array (center via `Cube::iter()`,
which is ~3× faster than per-block `get`; neighbor shells via bounded
loops — uniform neighbors fill, non-uniform use `get_index` on just the
~1k border cells). Per direction and slice, coplanar faces with identical
(texture layer, AO pattern, light) merge into maximal rectangles: extend
along u first, then along v. Faces never span outside the cube. Output is
grouped by direction (`ranges`) so 2.2 can skip back-facing directions.

**Bench medians (this Mac, single-threaded):** `terrain_like` 244 µs
(target ≤ 300 µs), `checkerboard` 769 µs, `uniform_stone_air_neighbors`
233 µs, `padded_build` 22 µs.

### Worker pool

`MeshPool::new(appearance, threads)` (0 = `available_parallelism() − 2`,
minimum 1) spawns named threads `jbm-mesh-N`. The job queue is a
`Mutex<mpsc::Receiver>`; workers `recv_timeout` (50 ms) so shutdown is
prompt, and only ever hold the lock while waiting — never while meshing.
Results come back on an mpsc drained with non-blocking `try_recv()`.
Each job carries the cube's version at submit time. Dropping the pool
sets a stop flag, closes the queue (queued jobs are **discarded**, not
drained), lets in-flight jobs finish and joins every thread; nothing
detaches. A worker panic is caught and re-raised by the next `try_recv()`
with the panic message.

### World / dirty / version flow

`ClientWorld` holds `HashMap<CubePos, Arc<Cube>>`, a dirty set and
per-cube versions. `CubeData` inserts the cube, bumps its version and
dirties it and its 26 loaded neighbors (a new neighbor changes border
culling and AO). `UnloadCube` removes the cube — neighbors are *not*
re-dirtied (a missing neighbor culls faces, so their meshes stay valid).
`take_dirty(budget, camera)` returns the nearest dirty cubes first with
their `MeshInput` and version. A mesh result older than the cube's
current version is stale and dropped (a newer version implies a newer
dirty entry).

### Terrain rendering

Per meshed cube: one storage buffer of `PackedFace`s, one bind group, one
draw of `face_count × 6` vertices — no vertex buffer, no index buffer
(vertex pulling; the shader expands each face into a quad). Empty meshes
allocate nothing. Cube origins are camera-relative: each frame
`cube_min_world − camera.position` is computed in f64, cast to f32 and
written into **one** origins storage buffer (rewritten per frame, grown
only when visible cubes exceed capacity); the draw uses
`first_instance = slot` so `instance_index` selects the origin. Absolute
world coordinates never reach f32.

**Upload budget:** at most 32 mesh uploads per frame (leftover results
stay queued), plus at most 16 dirty-cube submissions — streaming never
hitches a frame. CPU frustum culling tests each cube AABB against the
camera-relative frustum (5 planes; far plane is at infinity).

Terrain pipeline: depth `Greater` with reverse-Z, back-face culling,
samples `BlockTextures` (nearest, linear mips, repeat addressing so
merged faces tile per block), `discard` below alpha 0.5, directional
shading (top 1.0, ±Z 0.8, ±X 0.6, bottom 0.5) times the AO factor
(0.4/0.6/0.8/1.0).

### App runtime

`app::run(link)` sends `Hello`, then polls the link every frame without
blocking. `Welcome` creates the `Predictor` at the spawn feet position
(the sim does not run before it); `Rejected`, `Disconnected` or a closed
link print the reason to stderr and exit the loop cleanly. Each frame
applies mouse look, runs a fixed 20 Hz sim clock with an accumulator (at
most 5 catch-up ticks per frame; the leftover accumulator is dropped so a
stall never spirals), sets the camera position to
`predictor.render_position(alpha)` where `alpha` is the accumulator
fraction, drains the link into `ClientWorld` and the `Predictor`
(`PlayerState` → `on_server_state`), submits dirty cubes, receives and
uploads meshes within the budgets above. The overlay adds
`CUBES <loaded> MESHED <n> PENDING <dirty + in flight>`,
`FACES <total> DRAWS <n>`, and the prediction lines
`MODE <WALK/FLY> GROUND <YES/NO>`, `VEL <x y z>` and
`PENDING <n> CORRECTIONS <n>`.

## Prediction (task 1.6)

`predict::Predictor` runs the shared `jbm_core::physics::step` ahead of
the server:

- **Tick:** each sim tick steps the prediction, stores the input with
  its seq (starting at 1) and returns the `PlayerInput` to send. Pending
  (unacknowledged) inputs are capped at 128; past that the oldest is
  dropped and counted — the server's next state resynchronizes anyway.
- **Correction:** `on_server_state(ack_seq, state)` drops pending inputs
  with seq ≤ ack, re-simulates from the authoritative state through the
  remaining pending inputs **in seq order**, and counts a correction when
  the replay differs from the prediction (exact `PartialEq`). The
  replayed state wins; corrections **snap** — the render interpolation's
  previous state jumps too. Smoothing the snap is a later task.
- **Interpolation:** `render_position(alpha)` returns the eye position
  lerped between the previous and current tick states, so rendering at
  any frame rate stays smooth between 20 Hz sim ticks.

The client's `ClientWorld` implements `jbm_core::physics::BlockShapes`
(its `SolidTable` is built from the builtin registry in `new`), so
prediction collides against exactly the cubes the server sent — the
unloaded rule freezes both sides identically at streaming edges. End-to
end convergence (and the client-world-lies correction path) is proven by
`crates/game/tests/prediction.rs`.

### Reverse-Z rendering

The depth buffer is `Depth32Float` cleared to **0.0** with a **`Greater`**
compare; the projection maps the near plane (0.05) to depth 1.0 and has an
**infinite far plane** (depth → 0.0). Most precision sits near the camera,
which Stage 2's kilometre view distances need. Every pipeline must draw
with `Greater`, never `Less`, and never clear depth to 1.0.

### Camera-relative rendering

World coordinates reach ~6.8·10¹⁰, beyond f32 precision. `Camera::position`
is f64; `view_proj()` is built with the camera at the origin (rotation
only). Geometry is offset by `(world_pos - camera.position)` in f64, cast
to f32 via `Camera::relative`, then drawn against that matrix. Never bake
the camera translation into the view matrix. The `mesh_far_origin`
screenshot test renders the same scene ~10⁹ blocks out and must match the
same golden, proving this works.

### Uniform layout rule

`FrameUniforms` (`render/renderer.rs`) is `#[repr(C)]` and mirrored as
`struct FrameUniforms` in `shaders/sky.wgsl`, `shaders/overlay.wgsl` and
`shaders/terrain.wgsl`. Every field is a `mat4x4f` or padded `vec4f`, so
Rust and WGSL agree at 16-byte alignment with no hidden padding;
`frame_uniforms_match_wgsl_layout` asserts offsets and total size
(224 bytes). When adding a field, pad it to 16 bytes and update the test
and all mirrors. The spare `viewport` field (x, y = target size in px)
feeds the overlay's pixel→NDC math.

## Sky colors

Linear-RGB constants in `renderer.rs`, converted to sRGB by the target:

| Constant | Linear | ≈ sRGB hex |
| --- | --- | --- |
| `ZENITH_COLOR` | 0.15, 0.40, 0.82 | `#6caaea` |
| `HORIZON_COLOR` | 0.70, 0.82, 0.94 | `#daeaf8` |
| `GROUND_HAZE_COLOR` | 0.34, 0.37, 0.43 | `#9ea4af` |

Above the horizon the color rises from horizon pale to zenith blue with
`pow(dir.y, 0.45)`; below it fades to ground haze over ~15°.

## Controls

| Input | Action |
| --- | --- |
| W / A / S / D | Move (yaw-plane) |
| Space | Jump (walk) / fly up |
| Shift | Fly down |
| Ctrl | Sprint (walk 5.6, fly 22 blocks/s) |
| F | Toggle fly (one edge per press) |
| Mouse | Look (while cursor grabbed) |
| Left click | Grab + hide cursor (Locked, falls back to Confined) |
| Esc | Release cursor; a second Esc quits |

`JBM_VSYNC=0` switches the surface from `AutoVsync` to `AutoNoVsync`.

## Debug overlay

`render/overlay.rs` rasterizes `font8x8` (public domain) ASCII 32..127
into an R8 atlas once, then draws each line as instanced screen-space
quads (integer scale, 2× by default) with alpha blending: white text on a
semi-transparent dark box per line. `FrameStats` averages FPS / frame ms
over a sliding 0.5 s window (pure logic, unit-tested). The app fills the
lines `FPS`, `XYZ` (2 decimals), `CUBE` (32-block cube coordinates),
`YAW`/`PITCH` in degrees, `CUBES`/`MESHED`/`PENDING` and
`FACES`/`DRAWS`.

## Screenshot tests

A screenshot test renders offscreen with wgpu and compares the PNG
against a committed golden in `crates/client/tests/golden/<name>.png`.
See `tests/screenshot.rs` for the pattern:

```rust
let Some(gpu) = gpu_or_skip() else { return; };   // skips without an adapter, panics if JBM_REQUIRE_GPU=1
let target = OffscreenTarget::new(&gpu.device, 256, 256);
// ... draw ...
let image = target.read_rgba(&gpu);
assert_matches_golden("triangle", &image);
```

- **Add a test:** render, pick a unique `name`, run once with
  `JBM_UPDATE_GOLDEN=1`, commit the new golden next to the test.
- **Update a golden** after an intended renderer change: rerun the test
  with `JBM_UPDATE_GOLDEN=1` and commit the new PNG.
- **On failure** the harness writes `<target>/screenshots/<name>.actual.png`
  and `<name>.diff.png` (differing pixels magenta, rest dimmed grayscale)
  and panics with the differing-pixel counts and both paths.
- Comparison tolerates ≤3 per-channel delta per pixel and ≤0.5% differing
  pixels, so Metal, Vulkan and lavapipe rasterizers all pass against one
  golden; a size mismatch always fails.
- A missing golden panics and writes `<name>.actual.png` next to where
  the golden belongs.

Current goldens: `triangle` (Stage 0), `sky_horizon` / `sky_up` /
`sky_down` (sky gradient from pinned cameras), `debug_overlay`
(sky_horizon + fixed overlay text), `mesh_blocks` (builtin blocks on a
stone floor; glass and leaves cutout visible), `mesh_ao` (stepped
pyramid with AO in the inner corners), `mesh_cube_border` (a building
crossing the boundary between two cubes — no seam),
`mesh_far_origin` (the `mesh_ao` scene ~10⁹ blocks out; matches the same
`mesh_ao` golden, proving camera-relative rendering).
