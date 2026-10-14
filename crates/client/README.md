# jbm-client

The player-facing side: renderer, UI, input, audio, and client-side
prediction.

**Public API**

- `gpu::Gpu` / `gpu::GpuError` — headless instance/adapter/device/queue
  acquisition (`Gpu::headless()`, no surface, no panics).
- `render::offscreen::OffscreenTarget` — an `Rgba8UnormSrgb` render
  target with `read_rgba(&Gpu) -> image::RgbaImage` readback (row padding
  stripped).
- `render::triangle` — the minimal test pipeline: one
  vertex-index-driven triangle (no vertex buffers) on the fixed
  `CLEAR_COLOR` dark gray. First stop for every renderer feature.
- `testing::screenshot` (only with `--features testing` or in unit
  tests) — golden-image comparison helpers, see below.

**Invariants**

- Depends only on `jbm-core`; must NOT depend on `jbm-server` (enforced by
  `cargo tree -p jbm-client` in CI-style checks).
- Renderer changes need screenshot tests.

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
