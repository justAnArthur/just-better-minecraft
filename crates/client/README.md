# jbm-client

The player-facing side: renderer, UI, input, audio, and client-side
prediction. Talks to the server only through `jbm-core` protocol messages.

**Cube size:** all mesher and renderer dimensions derive from
`jbm_core::coords::CUBE_SIZE`, so the crate compiles and its unit tests
pass under `jbm-core`'s `cube16` feature (16³ cubes; see jbm-core's
README). The terrain-scene goldens are hand-placed at 32³-local
coordinates and cfg-gated off under the crate's own `cube16` passthrough;
`app.rs` camera-cube math is 32³-only (out of task 1.8's scope).

**Public API**

- `gpu::Gpu` / `gpu::GpuError` — GPU acquisition. `Gpu::headless()` for
  tests (no surface, never panics); `Gpu::for_surface(instance, &surface)`
  for the window.
- `camera::Camera` — f64 position, yaw/pitch (pitch clamped to ±89°),
  reverse-Z infinite-far projection, camera-relative `view_proj()` and
  `relative()`; `Camera::looking_at(eye, look_at, fov, aspect)` is the
  fixed-camera constructor the tests and the playground share.
- `input::InputState` — winit-independent key set, mouse delta
  accumulation and the F-key fly-toggle edge; `input::move_input` maps
  held keys onto the physics `MoveInput`, `input::apply_mouse_look`
  applies the per-frame look.
- `mesh::appearance` — the flat per-state appearance table plus the
  procedural builtin 16×16 placeholder textures (`builtin_textures`).
- `mesh::face` — the 8-byte `PackedFace` format (see below).
- `mesh` — the mesher's public API: `mesh()` (the entry point, with the
  uniform-cube shortcuts), `MeshInput`, `MeshOutput`, `neighbor_index()`
  and `build_padded()` (bench-only).
- `mesh::input` — the padded `(CUBE_SIZE + 2)^3` cell view and the
  `MeshInput`/`MeshOutput` types.
- `mesh::greedy` — the greedy sweep over the padded view (`mesh_padded`).
- `mesh::seams` — the level-seam crack hiders (skirts and lids, level
  >= 1 only).
- `mesh::oracle` (test-only) — the reference meshers the greedy output
  is checked against.
- `mesh::pool::MeshPool` — the mesh worker pool.
- `world::ClientWorld` — cubes from the server plus dirty/version state.
- `terrain_sync::TerrainSync` — the world → mesh pool → coverage →
  terrain-upload pipeline in one place: it owns the `ClientWorld`, the
  `MeshPool`, the `Coverage` and the in-flight mesh-job count.
  `apply(gpu, terrain, msg)` feeds one server message in (an unload also
  drops the GPU mesh and coverage readiness), `pump(gpu, terrain,
  camera, submit_budget, upload_budget)` submits dirty cubes and
  uploads finished meshes within the caller's budgets (the app's and the
  GPU tests' 16 submits / 32 uploads), `settle_sync(gpu, terrain,
  appearance, camera)` meshes everything dirty inline (no pool round
  trip) until the world is clean — for deterministic test scenes — and
  `queue_clear()` reports an empty mesh queue. `in_flight_lod()` is the
  level ≥ 1 share of the in-flight jobs (the overlay's LOD queue
  length).
- `render::renderer::Renderer` — the frame renderer (sky + terrain +
  outline + overlay dispatch), taking its per-frame inputs as one
  `render::renderer::FrameInputs` struct (target, depth, camera, terrain,
  outline, overlay). One `Renderer` per color format; works against any
  target, so the window surface and the offscreen test target share the
  code.
- `render::renderer::DepthTarget` — `Depth32Float` attachment with resize;
  `TEXTURE_BINDING` usage so the Hi-Z pass can sample it.
- `render::terrain::TerrainPass` — the terrain pass: paged face storage,
  persistent meta slots, and three draw paths — GPU compute culling
  (frustum + Hi-Z occlusion, two-phase), CPU culling with
  `multi_draw_indirect`, or a direct-draw fallback (see "Terrain
  rendering" below). Split by concern under `render/terrain/`:
  `mod.rs` (the public API), `layout.rs` (the GPU layout contract),
  `space.rs` (camera/frustum math), `pages.rs` (CPU bookkeeping),
  `stats.rs` (counters and the readback ring), `pipelines.rs`
  (construction and bind groups), `cpu_cull.rs` / `gpu_cull.rs` (the draw
  paths).
- `render::settings::RenderSettings` — the frame-level settings (fog,
  level tint, far-look radii, draw-path flags) the renderer owns and
  passes to `TerrainPass::draw`; `from_env()` is the one read of the
  `JBM_NO_FOG` / `JBM_LOD_TINT` / `JBM_GPU_CULL` / `JBM_NO_MDI`
  variables, done by the app at startup.
- `render::hiz::HizPyramid` — the hierarchical-Z depth pyramid the GPU
  cull tests occlusion against.
- `render::suballoc::Suballocator` — the CPU-side free-list suballocator
  behind the pages (pure logic, no wgpu).
- `render::overlay` — debug overlay (`OverlayText`, `OverlayRenderer`,
  `FrameStats`).
- `lod` — the client's LOD state (task 2.6): the coverage rule
  (`Coverage`) and `core::lod::CubeKey` re-exported as
  `jbm_client::lod::CubeKey`, the one path every call site uses.
- `render::texture_array::BlockTextures` + `placeholder_layers(n)` —
  block texture array with CPU-generated mipmaps (repeat addressing so
  merged faces tile per block), plus each layer's average color (task
  2.9) as a CPU table (`averages()`) and a storage buffer
  (`averages_buffer()`) the terrain pass binds.
- `render::offscreen::OffscreenTarget` — an `Rgba8UnormSrgb` render
  target with `read_rgba(&Gpu) -> image::RgbaImage` readback.
- `render::triangle` — the minimal test pipeline.
- `predict::Predictor` — client-side prediction of the shared physics
  (see below).
- `hud` — the debug overlay's text: `HudSnapshot` (one frame's stats:
  fps, camera, target, player, terrain/cull/level counters, LOD queue,
  tint flag) and `hud::lines(&snapshot) -> Vec<String>`. The app fills
  the snapshot once per frame; the 2.12 egui debug panel reads the same
  struct.
- `app::run(link)` — opens the window and runs the frame loop against a
  server link, with the default view distances; `app::run_with_view`
  takes a `ViewSetup` (the ring radii the far look keys on — the game
  derives it from its `ServerConfig`). `JBM_AUTOFLY=<blocks>` (task 2.6
  gate) turns the app into an autopilot: once spawned it toggles flying,
  climbs 40 blocks above spawn, flies straight at sprint speed for
  `<blocks>` blocks through the normal `PlayerInput` path (prediction
  and the server both see a real player), then exits printing a
  frame-time summary (count, mean, p50, p95, p99, max, frames over
  33 ms and 50 ms). The autopilot ignores the window-occlusion pause so
  an unattended gate run finishes even when its window is never shown.
- `testing::screenshot` (only with `--features testing` or in unit tests)
  — golden-image comparison helpers, see below.
- `testing::OffscreenRig` (same gate) — the offscreen render half for GPU
  tests: renderer, terrain pass, color target and depth in one struct,
  with `render_with_sky(gpu, camera, sky) -> RgbaImage` (sky-color
  override for this frame only) and `read_depth(gpu)`. Tests keep only
  their protocol state and drive a `TerrainSync` against `rig.terrain`.

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

The mesher first builds a padded `(CUBE_SIZE + 2)³` array (center via
`Cube::iter()`, which is ~3× faster than per-block `get`; neighbor shells
via bounded loops — uniform neighbors fill, non-uniform use `get_index` on
just the border cells). Per direction and slice, coplanar faces with identical
(texture layer, AO pattern, light) merge into maximal rectangles: extend
along u first, then along v. Faces never span outside the cube. Output is
grouped by direction (`ranges`) so 2.2 can skip back-facing directions.

**Bench medians (Apple M5 Pro, idle, single-threaded):** `terrain_like`
221 µs (target ≤ 300 µs), `checkerboard` 697 µs,
`uniform_stone_air_neighbors` 219 µs, `padded_build` 20 µs.

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

**Edits (task 1.7):** `set_block` (optimistic) and `BlockChange`
(authoritative) set one block and dirty its cube plus every loaded
neighbor the block touches through a face, edge or corner (x == 0 dirties
−X, x == CUBE_SIZE−1 dirties +X, and so on: up to 7 neighbors). All of
those go into a priority set that `take_dirty` drains first, nearest
first, before streaming dirt; a late neighbor would show a hole. Setting
a block to its current state dirties nothing, so the server's echo of an
accepted edit costs no re-mesh. `Arc::make_mut` copies the cube only
while a mesh job still holds the old one.

**Edit flow:** each frame the app raycasts from the camera (the rendered
eye, so the crosshair matches what is drawn) up to `REACH` against
`ClientWorld`. Left button breaks the target: optimistic air, then
`BreakBlock`. Right button places the hotbar block at target + face
normal when that cell is loaded, air, and (for solid states) clear of
the predicted player box: optimistic set, then `PlaceBlock`. Held
buttons repeat every 0.25 s; the first press acts at once. A rejected
edit comes back as a `BlockChange` with the real state and is reverted
like any other change. Hotbar, repeat timers and the place decision live
in `edit.rs`, winit-free and unit-tested.

### The LOD world and the coverage rule (task 2.6)

`ClientWorld` stores LOD ring cubes beside the LOD0 cubes, keyed by
`lod::CubeKey { level, pos }` — mesh jobs, mesh results, versions,
dirt and GPU residency all use that one key (LOD0 is level 0).
`LodCubeData` inserts or replaces the cube and dirties it and its 26
loaded **same-level** neighbors; `UnloadLodCube` removes it. Mesh
scheduling keeps LOD0 priority (edits first, then nearest), takes LOD0
streaming dirt nearest-first, then LOD dirt nearest-first by distance in
blocks divided by 2^level — one budget, split 3:1 in LOD0's favor when
both have dirt (a guaranteed quarter for LOD, minimum one of any budget;
each class yields its whole share when idle), so neither starves.

**The coverage rule** (`lod::Coverage`): every volume is drawn by
exactly one cube — the finest ready one along its chain of ancestors.
`drawn(c)` holds iff `ready(c)`, and either `drawn(parent(c))`, or no
ancestor of *c* is ready. A drawn cube at level ≥ 1 **masks** octant *o*
when the child at *o* is ready (that child is then itself drawn and
covers the octant); a drawn cube draws exactly its unmasked octants, and
a fully masked one draws nothing (it reads inactive to the cull pass;
the vertex shader also collapses masked faces to degenerate quads as a
second line of defense). *Why exactly one:* along a point's chain, the
topmost ready cube *T* is drawn (nothing above it is ready); drawing
passes down while the chain stays ready and stops at the first
not-ready cube, whose parent draws the point — every step hands the
point to exactly one child, so nothing is skipped or doubled. Points in
no ready cube are drawn by nothing. Readiness comes from the app: a
cube is ready when loaded and its mesh is uploaded (empty meshes count;
a re-mesh in flight keeps the old mesh). Updates are incremental — a
readiness change touches the parent's mask, the cube itself and its
descendants, nothing else — and a frame with no change does no work
(`Coverage::work()`). The accepted transient: a coarse cube briefly
covers a finer one when the cube between them arrives late, typically
for a moment at startup.

### Octant-split meshing and skirts (task 2.6)

The same greedy mesher serves every level; cells are cells whatever
their world size. Two rules switch on for level ≥ 1 (LOD0 output is
byte-for-byte the pre-2.6 mesher's, pinned by a test against a copied
reference):

- **Octant-split merging.** Greedy rectangles never cross the cell
  planes x/y/z = `CUBE_SIZE / 2`, so every face lies in exactly one
  octant, which masking needs. (In mask space the halves stay
  `[0, half)`/`[half, size)` however the u/v strides are signed.)
- **Skirts** (±X/±Z only), the crack hider at level seams. Emit cell
  *c*'s face in direction *d* even though the cell across is filled
  when: *c* is filled; the cell across lies in a different octant or a
  different cube; that cell is filled or in a missing neighbor cube;
  and *c* is one of the top two filled cells of its column run (air
  above, or air two above; at the cube's top row the two-above look
  reads as filled). The faces stand in for the coarse cell's side when
  the volume across is drawn by finer cubes whose surface sits lower
  (the downsampling rule leans solid, so the coarse surface can be up
  to 2^n − 1 blocks above the fine one; two coarse cells of skirt cover
  that). When the octant across is drawn by the same coarse cube the
  skirt sits inside terrain, hidden by the neighbor's top face. Skirt
  faces are ordinary faces (same AO from same-level neighbors, same
  texture, greedy-merged within the octant rule); `MeshOutput::skirts`
  counts them for the overhead report.
- **Lids** (+Y only, wave D), the skirts' counterpart for horizontal
  seams. Emit cell *c*'s top face even though the cell above is filled
  when *c* is filled and sits at y = `CUBE_SIZE / 2 − 1` (the octant
  plane) or y = `CUBE_SIZE − 1` (the cube border). The upper octants or
  the cube above may be drawn by finer cubes that are air at that height;
  without the lid a slope crossing the seam shows a contour line of holes
  (the lod_fly test over the 2.8 mountains found it). No column
  restriction, since a cliff can leave a finer column air at the seam
  however high the coarse one reaches; buried lids are hidden in terrain
  and greedy-merge to a few quads per octant. `MeshOutput::lids` counts
  them.

The mesher also feeds the renderer a per-face skirt bit and counts
skirt faces; the render expands level ≥ 1 quads by ~1/512 block
(sub-pixel) to close T-junction hairlines that greedy merging leaves
where a merged quad's edge abuts smaller quads — an invisible overlap
between coplanar same-texture neighbours. LOD0 skips the expansion to
keep its goldens bit-stable; its pinholes are single pixels, bounded by
the rings hole tests.

### Rendering LOD cubes

- **CubeMeta flags** (`start_flags.y`): bit 0 active (drawn by the
  coverage rule; freed slots and fully masked cubes read inactive), bit
  1 visible-last-frame, bits 8–12 the LOD level, bits 16–24 the octant
  mask. The 80-byte layout is pinned by the layout test; `set_draw_state`
  rewrites a slot only when its level, mask or mesh changes.
- **One scaled origin formula**, shared exactly by the shader, the cull
  shader and `cube_origin` on the CPU:
  `origin = (pos·2^level − camera_cube) · CUBE_SIZE − camera_frac`; cell
  offsets multiply by 2^level, and `pos·2^level` is asserted to stay in
  i32 for streamable positions. UVs stay one tile per **block**
  (multiplied by 2^level), so textures line up across ring borders.
- **Masked octants** collapse to degenerate quads (no fragments), and a
  fully masked cube emits no commands on any path. AABBs are
  `CUBE_SIZE·2^level` wide everywhere (frustum, direction skipping,
  Hi-Z).
- The three paths stay pixel-identical with LOD scenes (proven by the
  rings tests); the 2.6 re-measurement of gpu-cull at ring scale
  (cpu-cull ≈ 621 µs vs gpu-cull ≈ 948 µs, Apple M5 Pro / Metal) is in
  the game README.

### The far look: flat material colors and distance fog (task 2.9)

The far view reads as a smooth landscape (the Distant Horizons look),
not as noisy repeated block textures, and the LOD0/LOD1 switch is
invisible:

- **Per-layer average colors.** At texture load, `BlockTextures`
  computes each layer's average **in linear space** from the base mip,
  weighting every pixel by its alpha so cutout holes (leaves, glass) do
  not darken it; a fully transparent layer falls back to the
  unweighted mean. The averages live as a CPU table (`averages()`, for
  tests) and one storage buffer of `vec4f` per layer bound in the
  terrain group (binding 4). Grass top and grass side are different
  layers, so a hillside reads green on top and earthy on the sides.
- **`far_color(tex, layer, dist, level)`** (documented in
  `terrain.wgsl`) returns the base color before shade and fog:
  level ≥ 1 draws the layer's average, flat — no texture look, no
  cutout discard (a LOD cell is solid by the downsampling rule; the
  sample itself still happens because `textureSample` must stay in
  uniform control flow, and `mix(x, y, 1.0)` drops its value). Level 0
  mixes `mix(texture, average, fade(dist))` where `fade` is a
  smoothstep over **40%..85% of the LOD0 ring radius** — by the
  LOD0/LOD1 border (85%) both sides are flat, so the border cannot
  show. The cutout discard still applies while `fade < 1`.
- **AO softening:** AO contrast scales with cell size and reads blotchy
  far away, so coarse levels pull the corner AO factor toward fully
  open by a per-level factor (`AO_SOFTEN` = 1.0 / 0.6 / 0.45 / 0.3,
  clamped at the last entry). Level 0 keeps the exact factor, so
  pre-2.9 LOD0 goldens stay bit-stable. Directional face shading stays
  on every level.
- **Distance fog.** Every terrain fragment blends into the sky color
  along its own view ray with the ray's y clamped just above the
  horizon (`max(dir.y, 1e-6)`): the same gradient the sky pass paints
  behind it, so far rings dissolve into the sky with no visible band
  and the outermost ring's edge is never visible. The factor is a
  smoothstep over **35%..98% of the furthest ring radius** (98% keeps
  the edge inside the fog). WGSL has no includes, so the gradient
  lives once in `shaders/sky_gradient.wgsl` and pipeline creation
  concatenates that snippet ahead of both `sky.wgsl` and `terrain.wgsl`
  — the two can never drift.
- **Where the radii come from:** the client cannot learn them from the
  link, so the game derives a `ViewSetup` from its `ServerConfig`
  (`view_radius · CUBE_SIZE` and `lod_view_radius · CUBE_SIZE ·
  2^lod_levels`) and `app::run_with_view` feeds it into its one
  `RenderSettings::from_env()` read; the renderer-owned settings hand
  them to the terrain pass every frame (`Renderer::set_view_distance`
  changes them). Defaults match the `ServerConfig` defaults
  (`DEFAULT_LOD0_RING_RADIUS`, `DEFAULT_FURTHEST_RING_RADIUS`).
- **Test hooks:** `Renderer::set_fog(bool)` (also `JBM_NO_FOG=1`, read
  once by the app through `RenderSettings::from_env`) switches the fog
  off entirely so screenshot tests can isolate geometry from fog; the
  fade always stays. Fog paints terrain fragments only — never the sky
  pass — so the sentinel hole tests keep working with fog on: a hole
  still shows the sentinel sky.
- **Tuning:** every window is a fraction of a radius; the fractions are
  the consts in `render/terrain/layout.rs` (`FADE_START_FRACTION`,
  `FADE_END_FRACTION`, `FOG_START_FRACTION`, `FOG_END_FRACTION`,
  `AO_SOFTEN`) mirrored as WGSL consts in `terrain.wgsl` — keep the two
  in sync (tests compute expectations from the Rust ones).

Every draw path shares the one fragment shader, so cpu-cull, gpu-cull
and fallback stay pixel-identical under the far look (the rings
path-equality tests cover it).

### Level tint debug view (task 2.10)

**F4** (or `JBM_LOD_TINT=1` at startup) multiplies every fragment by a
fixed color per LOD level — one uniform flag (`TerrainUniforms.tint`),
applied before shade and fog, identical on all three draw paths because
they share the one fragment shader. LOD0 stays white; the others follow
`LOD_TINT_COLORS` in `render/terrain/layout.rs` (mirrored by `lod_tint` in
`terrain.wgsl`): L1 (2×2×2 cells) red, L2 (4×4) yellow, L3 (8×8) green,
L4 (16×16) cyan, L5 (32×32) blue, L6 (64×64) purple — never pure
magenta, the hole-test sentinel. While it is on, the overlay shows the
legend (`LOD TINT L1 2x2 red L2 4x4 yellow …`). Use it to see which
level draws where; the rings test `level_tint_marks_each_drawn_level`
pins that each level's color appears exactly where the coverage rule
draws that level.

### Terrain rendering

All cube meshes live in a few large face **pages**: one `STORAGE |
COPY_DST` buffer of 8-byte `PackedFace`s (default 4M faces = 32 MiB,
clamped to `max_storage_buffer_binding_size` / `max_buffer_size`, never
below one worst-case cube mesh — `6 · CUBE_SIZE³ / 2` faces) plus one
CPU-side `Suballocator` and one bind group per page. No vertex buffer, no
index buffer (vertex pulling; the shader expands each face into a quad).
A re-mesh frees the cube's old range and allocates a new slot
(first-fit over the pages, new page only when none fits) — it never
creates a buffer per cube. Empty meshes keep an entry with no allocation.
Freed ranges are reused immediately: wgpu's `Queue::write_buffer` docs
promise staged writes "begin GPU execution only on the next call to
`Queue::submit()`, just before the explicitly submitted commands", so a
this-frame write cannot race earlier frames' draws. The cube's six
per-direction sub-ranges from `MeshOutput::ranges` ride along so drawing
can skip directions.

**Known limit:** pages are never freed or compacted in this task;
fragmentation grows until a later compaction task (2.6).

**Meta slots (task 2.5):** every resident cube with a non-empty mesh
holds a slot in one persistent `cube_meta` storage buffer (80-byte
`CubeMeta`: cube position + page, face-range start + flags, the six
per-direction `(offset, count)` pairs; mirrored in `terrain.wgsl` and
`cull.wgsl`, pinned by a layout test). Slots come from a free list;
upload writes one, `remove` zeroes it, a re-mesh rewrites it in place
with the visible bit reset. **`first_instance` is the meta slot on every
path** — nothing per-cube is rewritten per frame.

**The one origin formula:** the vertex shader computes each cube's
camera-relative origin itself: `origin = (pos − camera_cube) · CUBE_SIZE
− camera_frac`, where `camera_cube = floor(camera / CUBE_SIZE)` and
`camera_frac = camera − camera_cube · CUBE_SIZE` (f64 on the CPU, rounded
to f32 once). The integer part is exact; the only rounding is the final
f32 subtract, so no absolute world coordinate reaches f32. Both values
ride in a terrain-only uniform (group 2 — sky, outline and overlay share
just `FrameUniforms`), and the CPU paths use the same Rust function
(`cube_origin`), so CPU and GPU cull on the same numbers.

### Three paths (task 2.5)

| Path | When | Culling | Draw |
|---|---|---|---|
| `cpu-cull` (default) | MDI available | 2.2's CPU frustum + directions over the same meta slots | `multi_draw_indirect` per page |
| `gpu-cull` (opt-in) | MDI **and** `DownlevelFlags::COMPUTE_SHADERS` + R32Float storage textures, **and** `JBM_GPU_CULL=1` or `set_prefer_gpu_cull(true)` | compute: frustum, Hi-Z, directions | `multi_draw_indirect` per page, two phases |
| `fallback` | no MDI, or `JBM_NO_MDI=1`, or `set_force_fallback(true)` | as `cpu-cull` | direct draws |

The path is chosen once at construction (`select_path`, a pure,
exhaustively tested function) and logged once; a gpu-capable adapter
also logs "gpu-cull available (JBM_GPU_CULL=1)". `set_prefer_gpu_cull`,
`set_force_cpu_cull` and `set_force_fallback` exist for tests and
benches; the force flags win over the preference. `path()` /
`path_name()` report the choice.

**Why gpu-cull is opt-in.** On the only hardware measured so far (Apple
Silicon, Metal) it is slower than cpu-cull in every scene, caves
included. Game-sized region (`JBM_REGION_SIDE=544`: 867 cubes, 430
meshes, 1280×720), frame mean including the GPU wait:

| Scene | gpu-cull | cpu-cull | gpu-cull commands / encoded slots | cpu-cull commands |
|---|---|---|---|---|
| surface | 0.80–1.07 ms | 0.43–0.61 ms | 928 / 1856 | 928 |
| cave (closed room) | 0.41–0.71 ms | 0.27–0.46 ms | 15 / 640 | 320 |

Where the time goes: on Metal `multi_draw_indirect` is a CPU loop of
one indirect draw per command and `draw_indirect_count` is missing, so
every region slot is encoded even when the GPU wrote a zero-count draw
(the CPU frustum bound keeps that to 2× the cpu-cull count, one region
per phase); the per-frame Hi-Z rebuild at 1280×720 costs about 0.18 ms;
the second render pass and the cull dispatches are the rest. In these
scenes the faces saved by occlusion cost less to raster than that.
cpu-cull stays the default until gpu-cull wins a measurement. Levers,
in order: re-measure with the LOD rings of 2.6 (far more geometry);
start the pyramid at half resolution, reducing straight from the depth
target; `multi_draw_indirect_count` on Vulkan/DX12.

**gpu-cull, per frame** (the CPU's only per-cube work is sizing the
command regions — see "Per-page regions and zero tails" below):

1. **Early cull** (compute): one thread per meta slot; slots that were
   visible last frame and pass the frustum test (the same plane test as
   the CPU `Frustum`, ported to WGSL with the same comparisons) emit
   their commands into the **early** half.
2. **Early draw**: render pass 1 draws the early regions, loading color
   and depth after the sky.
3. **Hi-Z build + late cull** (one compute pass): the pyramid is rebuilt
   from the depth the early draw left, then every active slot is tested
   against the frustum and the pyramid; `visible = frustum && !occluded`
   is written back for next frame, and slots not emitted early emit into
   the **late** half.
4. **Late draw**: render pass 2 draws the late regions.

Drawing what was visible last frame first means the pyramid already
contains the scene the player saw, so nothing pops in when the camera
moves; caves under the player cost almost nothing from the second frame
of a view on. New and re-meshed cubes start with `visible = 0`, so the
late pass picks them up the same frame (a fresh pass renders correctly
on its first frame; occlusion takes effect from the second frame).
Command order inside a region is nondeterministic (atomics), which is
fine for opaque, depth-tested terrain — the path-equality tests prove
the images are pixel-identical.

**The Hi-Z pyramid** (`render/hiz.rs` + `shaders/hiz.wgsl`): an
`R32Float` texture with a full floor-halved mip chain over the depth
target (mip 0 is a compute copy — depth textures are sampleable, not
storable). Every reduction is a 2×2 **min**, because in reverse-Z the
smaller depth is the farther surface and occlusion needs the farthest
one. Reduction rule (mirrored by the unit-tested `reduce_step`): each
dst texel is the min over its 2-texel pair of src texels, and the
**last** dst texel of an axis takes the remainder — so with an odd 5-texel
source, `dst 0` covers `{0,1}` and `dst 1` covers `{2,3,4}`. The fold
means no mip-0 texel is ever skipped (the classic Hi-Z bug with
floor-sized mips); every mip texel is the min over exactly the mip-0
texels its footprint covers (unit-tested). Mip k is bound as sampled and
mip k+1 as write-only storage in one dispatch — legal because wgpu
tracks usage per subresource. The whole chain is one compute pass: one
fill dispatch plus one reduce dispatch per mip, then the late cull rides
the same pass. That is sound because dispatches within a compute pass
are ordered and each dispatch is its own usage scope (WebGPU spec §3.4.4
"Synchronization": "In a compute pass, each dispatch command
(`dispatchWorkgroups()` or `dispatchWorkgroupsIndirect()`) is one usage
scope"), so wgpu inserts the write→read barrier at each mip transition.
Recreated together with the depth target.

**Occlusion test** (in `cull.wgsl`, conservative — it may keep an
occluded cube but never drops a visible one): project the 8 AABB
corners; if any corner has `w ≤` the near plane or the AABB contains
the camera, the cube is visible. Otherwise take the screen rectangle in
mip-0 texels (expanded a texel) and the cube's nearest depth (`max z/w`
in reverse-Z), pick the mip level where the rect spans at most 2×2
texels, read the covering texels (clamped) — their min is the farthest
occluder — and call the cube occluded iff `nearest < farthest`.

**Per-page regions and zero tails:** GPU-emitted commands go into
per-page regions of one command buffer (early and late halves). The
shader appends with `atomicAdd` on a per-page counter, and each frame
the CPU writes `region_base[page]` — the prefix sum of that frame's
**CPU frustum bound**: a walk over the resident meta slots running the
same frustum test and `visible_directions` rule as the cpu-cull path
(same origin formula, same view-projection rows), counting each
frustum-passing cube's non-empty, non-skipped directions per page. The
GPU still does all the deciding (frustum, Hi-Z occlusion, directions);
the bound only sizes the regions — and since both sides use identical
math, the bound is always ≥ what the GPU emits. The counters and the
frame's command prefix are `clear_buffer`ed first, so a region's unused
tail is **zero-count draws**: each page is drawn with one
`multi_draw_indirect(region_base, bound)` and the tail draws are no-ops
everywhere. The bound exists because Metal has no
`draw_indirect_count` and emulates `multi_draw_indirect` as a CPU loop
of one indirect draw per *encoded* command — the encoded count must
come from the CPU, and sizing it to the frustum-passing set (instead of
every resident cube) keeps the encode loop proportional to what is
visible. If the GPU ever emitted past the buffer capacity anyway, the
shader clamps the write and counts it in the `region_overflow` stat,
asserted zero by the path tests. A per-page running total (updated on
upload and remove) still exists as the static worst-case bound and
drives the command buffer's growth. `multi_draw_indirect_count` would
skip the tails entirely but Metal lacks it; adopting it on Vulkan/DX12
is a later optimization.

**Drawing** (cpu-cull and fallback): CPU frustum culling tests each
camera-relative cube AABB (from the same origin formula) against 5
planes, then, in sorted `CubePos` order, one indirect command is emitted
per visible cube **and face direction** with a non-empty range, skipping
back-facing directions (`visible_directions`: +X is skipped when the
whole cube is at `rel_min.x ≥ 0` — every +X face plane is then at or
beyond the camera's x and faces away — and symmetrically for −X, ±Y,
±Z; back-face culling discarded those faces anyway, so the image is
unchanged; the GPU path ports the same rule). Each command carries
`first_vertex = (page range start + direction range start) · 6` and
`first_instance = slot`; `vertex_index` includes `first_vertex` for
non-indexed draws, so `faces[vertex / 6]` still indexes page-local faces.

MDI is used when the adapter reports `DownlevelFlags::INDIRECT_EXECUTION`
and the device was created with `Features::INDIRECT_FIRST_INSTANCE`
(requested in `gpu.rs` only when the adapter supports it — device
creation never fails over it): **one draw call per page (and phase)**
instead of one per cube. The fallback issues the same commands as
`pass.draw(first_vertex..first_vertex + count, slot..slot + 1)` in a
loop; direct draws accept a first instance without the feature. Same
image (proven pixel-identical by tests).

**Stats and their latency:** `CullStats` (`candidates`, `frustum_culled`,
`occluded`, `drawn_early`, `drawn_late`, `commands`, `faces`,
`region_overflow`) is filled
by GPU atomics on the gpu-cull path and copied each frame into a ring of
3 `MAP_READ` buffers whose maps complete without blocking — so
`cull_stats()` (the overlay's `CULL DRAWN n FRUSTUM n OCCLUDED n` line)
is **one or two frames late**. On the CPU paths the same fields are
exact and immediate. `read_cull_stats_blocking(gpu)` waits out the
current frame's counters (tests, benches, the metrics example);
`region_overflow` counts commands the capacity clamp dropped (a bound
bug — zero, asserted by the path tests).

**`TerrainStats`:** `meshed`, `drawn`, `faces` (only non-skipped
directions), `commands` (indirect commands or fallback draws — one per
visible cube and kept direction; on gpu-cull from the GPU counters, one
or two frames late), `draw_calls` (render-pass draw API calls: 1 per
non-empty page and phase with MDI), `pages`, `page_bytes`, `used_faces`
(the CPU-known fields stay exact on every path).

**Upload budget:** at most 32 mesh uploads per frame (leftover results
stay queued), plus at most 16 dirty-cube submissions — streaming never
hitches a frame.

Terrain pipeline: depth `Greater` with reverse-Z, back-face culling,
samples `BlockTextures` (nearest, linear mips, repeat addressing so
merged faces tile per block), `discard` below alpha 0.5, directional
shading (top 1.0, ±Z 0.8, ±X 0.6, bottom 0.5) times the AO factor
(0.4/0.6/0.8/1.0). The far look (task 2.9, see above) replaces the
texel with the layer average on level ≥ 1, fades level 0's texel into
it with distance, and fogs every fragment toward the horizon sky color.

### App runtime

`app::run(link)` sends `Hello`, then polls the link every frame without
blocking. `Welcome` creates the `Predictor` at the spawn feet position
(the sim does not run before it); `Rejected`, `Disconnected` or a closed
link print the reason to stderr and exit the loop cleanly. Each frame
applies mouse look, runs a fixed 20 Hz sim clock with an accumulator (at
most 5 catch-up ticks per frame; the leftover accumulator is dropped so a
stall never spirals), sets the camera position to
`predictor.render_position(alpha)` where `alpha` is the accumulator
fraction, drains the link through a `TerrainSync` (cube messages via
`apply`) and into the `Predictor` (`PlayerState` → `on_server_state`),
then pumps the pipeline within the budgets above. The overlay comes from
a `hud::HudSnapshot` filled once per frame:
`CUBES <loaded> MESHED <n> PENDING <dirty + in flight>`,
`FACES <drawn, after direction skipping> PAGES <p> CMDS <c> DRAWS <d>
<path name>`, `CULL DRAWN <n> FRUSTUM <n> OCCLUDED <n>` (the GPU
counters, one or two frames late), and the prediction lines
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
| F4 | Toggle the LOD level tint debug view (task 2.10) |
| Mouse | Look (while cursor grabbed) |
| Left click | Grab + hide cursor (Locked, falls back to Confined); once grabbed, break the target block (hold to repeat every 0.25 s) |
| Right click | Place the hotbar block against the targeted face (hold to repeat) |
| 1–9 | Hotbar: stone, dirt, grass_block, sand, glass, oak_planks, oak_log, leaves, water |
| Esc | Release cursor; a second Esc quits |

`JBM_VSYNC=0` switches the surface from `AutoVsync` to `AutoNoVsync`.

## Debug overlay

`render/overlay.rs` rasterizes `font8x8` (public domain) ASCII 32..127
into an R8 atlas once, then draws each line as instanced screen-space
quads (integer scale, 2× by default) with alpha blending: white text on a
semi-transparent dark box per line. `FrameStats` averages FPS / frame ms
over a sliding 0.5 s window (pure logic, unit-tested). The app fills a
`hud::HudSnapshot` once per frame and `hud::lines` formats it into the
lines `FPS`, `XYZ` (2 decimals), `CUBE` (32-block cube coordinates),
`YAW`/`PITCH` in degrees, `CUBES`/`MESHED`/`PENDING`,
`FACES`/`PAGES`/`CMDS`/`DRAWS` (+ draw path), `TARGET` (block position
and state name) and `HOTBAR`.
Whenever an overlay is drawn, a centered crosshair (white with a dark
outline) is drawn too.

**Target outline:** `render/outline.rs` draws a line-list box around the
targeted block, expanded by 0.002, depth-tested, 60% black. Lines get no
depth bias, so the shader pulls each vertex 0.2% of its distance toward
the camera; without that the edges z-fought into dashes against
coplanar neighbor faces.

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
`mesh_ao` golden, proving camera-relative rendering), and the task 2.6
rings goldens `rings_close` / `rings_border12` / `rings_corner3`.

`tests/lod_rings.rs` (task 2.6) has no golden of its own for most
cameras: it builds a synthetic heightfield's rings (levels 0..=3, level
1 by the real downsampling rule over level-0 data, levels 2–3 sampled
at their own resolution with a child-corner-dominating max), feeds them
through the real client world, coverage rule, mesher and renderer, and
asserts **zero sentinel pixels** below a per-camera horizon margin
(computed from the sky's ~15° below-horizon fade and the rings' far-edge
drop angle) for eight cameras — close, zoomed and steep-down views at
the LOD0/LOD1 and LOD1/LOD2 borders, a corner where three levels meet,
horizon, high and outside-in views — with a small documented allowance
for pre-existing single-pixel T-junction pinholes. The three draw paths
must render the scene pixel-identically; a LOD1 cube renders like its
LOD0 pattern at the same camera (the ×2 scale invariance), and masking
hides exactly the masked volume with fully masked cubes emitting no
commands.

`tests/far_look.rs` (task 2.9) has no goldens either: it asserts
pixels directly. A single LOD1 cube's lit top face is the layer's flat
average within ±1 LSB (times shade and AO); a long LOD0 strip keeps the
textured look near, reaches the flat average far (±2 LSB), and every
transition row matches the analytic smoothstep at its distance; with
fog on, a fragment beyond the fog end equals the sky behind it (±1 LSB)
while a near pixel is byte-identical with fog off. The border test in
`tests/lod_rings.rs` (`lod0_lod1_border_is_invisible_with_flat_colors`)
checks 16×16 windows either side of the LOD0/LOD1 border for
`zoom_border01` and `down_border01`: mean luminance within 3%, fog off.

`tests/terrain_pages.rs` (task 2.2, extended by 2.5) has no golden of
its own: it churns 200 cubes through tiny face pages (re-uploads that
reuse freed slots, removals) and requires the result to be
**pixel-identical** to a freshly built pass holding only the final
meshes, on the cpu-cull path and — when the adapter allows — the gpu-cull
path, with every page's suballocator and the meta-slot / region-bound
invariants passing `check_invariants`. Cube-size agnostic.

Path-level GPU tests live in `crates/game/tests/terrain_paths.rs` (task
2.5): the three paths must render `terrain_spawn` and `edit_spawn`
pixel-identically, a 60-frame moving camera (through a carved room and
back out) must never differ between gpu-cull and cpu-cull, a camera in
the closed room must leave ≥80% of frustum-passing cubes
occlusion-culled, the open surface view stays nearly uncullled, and an
odd resize (333×217) keeps the paths identical.
