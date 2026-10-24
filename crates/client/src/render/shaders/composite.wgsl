// Render-scale composite (task 2.16): upscales the scene, rendered at a
// fraction of the window's pixels, to the window with a Catmull-Rom
// filter — sharper than bilinear, so block edges and nearest-magnified
// textures stay crisp. Nine bilinear taps evaluate the 4×4 bicubic
// kernel: each axis folds its two middle weights into one tap at their
// weighted offset.

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    // One triangle oversized to cover the screen.
    let p = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var out: VsOut;
    out.pos = vec4f(p[i], 0.0, 1.0);
    out.uv = vec2f(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
    return out;
}

fn tap(uv: vec2f, weight: f32) -> vec3f {
    return textureSampleLevel(scene, scene_sampler, uv, 0.0).rgb * weight;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let size = vec2f(textureDimensions(scene));
    let pos = in.uv * size;
    // The texel center at or left of/above the sample, and the offset
    // from it.
    let center = floor(pos - 0.5) + 0.5;
    let f = pos - center;
    // Catmull-Rom weights of the four texels per axis.
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let uv0 = (center - 1.0) / size;
    let uv12 = (center + w2 / w12) / size;
    let uv3 = (center + 2.0) / size;

    var color = tap(vec2f(uv0.x, uv0.y), w0.x * w0.y);
    color += tap(vec2f(uv12.x, uv0.y), w12.x * w0.y);
    color += tap(vec2f(uv3.x, uv0.y), w3.x * w0.y);
    color += tap(vec2f(uv0.x, uv12.y), w0.x * w12.y);
    color += tap(vec2f(uv12.x, uv12.y), w12.x * w12.y);
    color += tap(vec2f(uv3.x, uv12.y), w3.x * w12.y);
    color += tap(vec2f(uv0.x, uv3.y), w0.x * w3.y);
    color += tap(vec2f(uv12.x, uv3.y), w12.x * w3.y);
    color += tap(vec2f(uv3.x, uv3.y), w3.x * w3.y);
    // The negative lobes can overshoot at hard edges.
    return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
