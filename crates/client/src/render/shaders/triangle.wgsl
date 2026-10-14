// One triangle, no vertex buffer: positions and colors are selected with
// vertex_index. Keeps the first offscreen pipeline free of any buffers.

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let positions = array<vec2f, 3>(
        vec2f(0.0, 0.75),
        vec2f(-0.75, -0.75),
        vec2f(0.75, -0.75),
    );
    let colors = array<vec3f, 3>(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, 0.0, 1.0),
    );
    var out: VsOut;
    out.pos = vec4f(positions[i], 0.0, 1.0);
    out.color = colors[i];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
