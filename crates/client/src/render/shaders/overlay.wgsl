// Debug overlay: instanced screen-space quads. One vertex entry, two
// fragment entries: solid color boxes (fs_solid) and font8x8 glyphs
// sampled from an R8 atlas (fs_glyph).

struct FrameUniforms {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    camera_pos_hi: vec4f,
    camera_pos_lo: vec4f,
    horizon_color: vec4f,
    zenith_color: vec4f,
    ground_color: vec4f,
    viewport: vec4f,
    // Task 2.15 (see FrameUniforms in render/renderer.rs).
    sun_dir: vec4f,        // xyz toward the sun, w = time (s)
    sun_color: vec4f,      // rgb sun radiance, w = sun lighting on
    ambient_sky: vec4f,    // rgb sky ambient, w = clouds on
    ambient_ground: vec4f, // rgb ground bounce, w = water shading on
    camera_wrap: vec4f,    // xyz camera position mod WRAP_PERIOD
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;

@group(1) @binding(0) var atlas: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct QuadInstance {
    @location(0) pos: vec2f,
    @location(1) size: vec2f,
    @location(2) uv_min: vec2f,
    @location(3) uv_max: vec2f,
    @location(4) color: vec4f,
};

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32, inst: QuadInstance) -> VsOut {
    // Six vertices = two triangles covering the quad; no index buffer.
    let corners = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
        vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
    );
    let c = corners[i];
    // Pixel-space (top-left origin) to NDC.
    let px = inst.pos + c * inst.size;
    let ndc = vec2f(
        px.x / u.viewport.x * 2.0 - 1.0,
        1.0 - px.y / u.viewport.y * 2.0,
    );
    var out: VsOut;
    out.pos = vec4f(ndc, 1.0, 1.0);
    out.uv = mix(inst.uv_min, inst.uv_max, c);
    out.color = inst.color;
    return out;
}

@fragment
fn fs_glyph(in: VsOut) -> @location(0) vec4f {
    let mask = textureSample(atlas, atlas_sampler, in.uv).r;
    return vec4f(in.color.rgb, in.color.a * mask);
}

@fragment
fn fs_solid(in: VsOut) -> @location(0) vec4f {
    return in.color;
}
