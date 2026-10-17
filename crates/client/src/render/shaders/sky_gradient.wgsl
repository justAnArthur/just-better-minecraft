// The sky gradient, shared verbatim by sky.wgsl (the sky pass) and
// terrain.wgsl (distance fog, task 2.9). WGSL has no includes, so pipeline
// creation concatenates this snippet ahead of both shader texts — the two
// can never drift. The colors are parameters so each caller passes its own
// values (both shaders read the same FrameUniforms fields).
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
