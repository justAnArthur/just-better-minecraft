// The target outline: a depth-tested line-list box around the targeted
// block. Positions are camera-relative (see render/outline.rs); the color
// is a fixed dark, 60% black (40% vanished against dirt at golden size).

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
};

@vertex
fn vs_main(@location(0) p: vec3f) -> VsOut {
    var out: VsOut;
    // Depth bias does not apply to line topologies, and the box edges are
    // coplanar (within the 0.002 expansion) with neighboring block faces,
    // so the edges z-fought into dashes. Pull each vertex 0.2% of its
    // distance toward the camera instead (positions are camera-relative):
    // 1 cm at the 5-block reach, invisible as an offset.
    out.pos = u.view_proj * vec4f(p * 0.998, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 0.0, 0.6);
}
