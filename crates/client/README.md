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
- `input::InputState` / `input::FlyController` — winit-independent key
  set + mouse delta accumulation, and the fly movement that consumes them.
- `render::renderer::Renderer` — the frame renderer (sky pass + overlay
  dispatch). One `Renderer` per color format; works against any target,
  so the window surface and the offscreen test target share the code.
- `render::renderer::DepthTarget` — `Depth32Float` attachment with resize.
- `render::overlay` — debug overlay (`OverlayText`, `OverlayRenderer`,
  `FrameStats`).
- `render::texture_array::BlockTextures` + `placeholder_layers(n)` —
  block texture array with CPU-generated mipmaps (unsampled until 1.4).
- `render::offscreen::OffscreenTarget` — an `Rgba8UnormSrgb` render
  target with `read_rgba(&Gpu) -> image::RgbaImage` readback.
- `render::triangle` — the minimal test pipeline.
- `app::run()` — opens the window and runs the frame loop.
- `testing::screenshot` (only with `--features testing` or in unit tests)
  — golden-image comparison helpers, see below.

## Invariants

- Depends only on `jbm-core`; must NOT depend on `jbm-server` (enforced by
  `cargo tree -p jbm-client` in CI-style checks).
- Renderer changes need screenshot tests.

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
the camera translation into the view matrix.

### Uniform layout rule

`FrameUniforms` (`render/renderer.rs`) is `#[repr(C)]` and mirrored as
`struct FrameUniforms` in `shaders/sky.wgsl` and `shaders/overlay.wgsl`.
Every field is a `mat4x4f` or padded `vec4f`, so Rust and WGSL agree at
16-byte alignment with no hidden padding; `frame_uniforms_match_wgsl_layout`
asserts offsets and total size (224 bytes). When adding a field, pad it to
16 bytes and update the test and both mirrors. The spare `viewport` field
(x, y = target size in px) feeds the overlay's pixel→NDC math.

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
| Space / Shift | Move up / down (world Y) |
| Ctrl | Sprint (×4) |
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
lines `FPS`, `XYZ` (2 decimals), `CUBE` (32-block cube coordinates) and
`YAW`/`PITCH` in degrees.

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
(sky_horizon + fixed overlay text).
