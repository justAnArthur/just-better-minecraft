// Sky pass: a fullscreen triangle with no vertex buffers. The fragment
// shader reconstructs the view ray from the inverse view-projection and
// picks a color from the vertical angle.

struct FrameUniforms {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    camera_pos_hi: vec4f,
    camera_pos_lo: vec4f,
    horizon_color: vec4f,
    zenith_color: vec4f,
    ground_color: vec4f,
    viewport: vec4f,
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;

struct VsOut {
    @builtin(position) pos: vec4f,
    // Clip-space x/y of this pixel, also its NDC after /w (w == 1).
    @location(0) ndc: vec2f,
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
    out.pos = vec4f(p[i], 1.0, 1.0);
    out.ndc = p[i];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // A point on the near plane along this pixel's view ray; the matrix
    // is rotation-only (camera-relative), so normalizing it gives the
    // world-space ray direction.
    let near = u.inv_view_proj * vec4f(in.ndc, 1.0, 1.0);
    let dir = normalize(near.xyz / near.w);
    // The gradient itself lives in sky_gradient.wgsl, concatenated ahead
    // of this text at pipeline creation (task 2.9): terrain.wgsl's fog
    // evaluates the very same function.
    return vec4f(
        sky_gradient(dir, u.horizon_color.rgb, u.zenith_color.rgb, u.ground_color.rgb),
        1.0,
    );
}
