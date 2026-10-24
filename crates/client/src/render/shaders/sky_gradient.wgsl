// The sky gradient and the sun glow, shared verbatim by sky.wgsl (the sky
// pass) and terrain.wgsl (the haze, task 2.9 / 2.15; water reflections,
// task 2.15). WGSL has no includes, so pipeline creation concatenates this
// snippet ahead of both shader texts — the two can never drift. The colors
// are parameters so each caller passes its own values (both shaders read
// the same FrameUniforms fields).
//
// Above the horizon the color rises from the pale horizon into the zenith
// with pow(dir.y, 0.45); below it fades into the darker ground-haze color
// over ~15 degrees (t reaches 1 at dir.y = -0.25).
fn sky_gradient(dir: vec3f, horizon: vec3f, zenith: vec3f, ground: vec3f) -> vec3f {
    if (dir.y >= 0.0) {
        let t = pow(min(dir.y, 1.0), 0.45);
        return mix(horizon, zenith, t);
    }
    let t = min(-dir.y * 4.0, 1.0);
    return mix(horizon, ground, t);
}

// The sun's glow (task 2.15): a soft, warm halo around the sun direction
// `sun`, added to the sky and — through the haze — to fogged terrain in
// the sun's direction, so distant hills under the sun melt into the same
// glow the sky shows behind them. A wide faint lobe (the bright, warm sky
// around a mid-morning sun) plus a tight one (the corona); the sun disk
// itself is sky-only (sky.wgsl).
//
// Below the horizon the glow fades out with the gradient's own ground
// blend (`below` = that blend's t, 0 above the horizon), so a pixel fully
// in the ground haze is exactly the ground color — the hole tests'
// sentinel stays exact.
const SUN_GLOW_WIDE: f32 = 0.16;
const SUN_GLOW_WIDE_POWER: f32 = 5.0;
const SUN_GLOW_TIGHT: f32 = 0.45;
const SUN_GLOW_TIGHT_POWER: f32 = 90.0;

fn sun_glow(dir: vec3f, sun: vec3f, sun_color: vec3f) -> vec3f {
    let c = max(dot(dir, sun), 0.0);
    let glow = SUN_GLOW_WIDE * pow(c, SUN_GLOW_WIDE_POWER)
        + SUN_GLOW_TIGHT * pow(c, SUN_GLOW_TIGHT_POWER);
    let below = min(max(-dir.y, 0.0) * 4.0, 1.0);
    return sun_color * glow * (1.0 - below);
}
