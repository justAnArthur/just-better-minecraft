// Terrain pass: vertex pulling. Each cube owns a storage buffer of packed
// faces; the vertex shader expands face (vertex_index / 6) into a quad and
// the fragment shader samples the block texture array. The bit layout
// mirrors the comment block in crates/client/src/mesh/face.rs — keep the
// two files in sync.
//
// Bind groups:
//   0: shared frame uniforms (same layout as sky.wgsl)
//   1: per-cube faces storage buffer (array<vec2u>, one vec2 per face)
//   2: shared terrain resources: origins, textures, sampler
//
// Each draw covers one cube: face_count * 6 vertices, one instance, with
// first_instance set to the cube's slot in the origins buffer. Origins are
// camera-relative (world cube corner - camera position, computed in f64 on
// the CPU), so no absolute world coordinate ever reaches f32.

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
@group(1) @binding(0) var<storage, read> faces: array<vec2u>;
@group(2) @binding(0) var<storage, read> origins: array<vec4f>;
@group(2) @binding(1) var block_textures: texture_2d_array<f32>;
@group(2) @binding(2) var block_sampler: sampler;

// Per-direction frames, identical to the table in mesh/face.rs:
// direction -> (normal, u axis, v axis), with cross(u, v) = normal.
//   0 +X: u=(0,0,-1) v=(0,1,0)   1 -X: u=(0,0,1)  v=(0,1,0)
//   2 +Y: u=(0,0,1)  v=(1,0,0)   3 -Y: u=(1,0,0) v=(0,0,1)
//   4 +Z: u=(1,0,0)  v=(0,1,0)   5 -Z: u=(-1,0,0) v=(0,1,0)
fn normals() -> array<vec3i, 6> {
    return array<vec3i, 6>(
        vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0),
        vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1),
    );
}
fn u_axes() -> array<vec3i, 6> {
    return array<vec3i, 6>(
        vec3i(0, 0, -1), vec3i(0, 0, 1), vec3i(0, 0, 1),
        vec3i(1, 0, 0), vec3i(1, 0, 0), vec3i(-1, 0, 0),
    );
}
fn v_axes() -> array<vec3i, 6> {
    return array<vec3i, 6>(
        vec3i(0, 1, 0), vec3i(0, 1, 0), vec3i(1, 0, 0),
        vec3i(0, 0, 1), vec3i(0, 1, 0), vec3i(0, 1, 0),
    );
}

// Directional shading per face: top 1.0, +/-Z 0.8, +/-X 0.6, bottom 0.5.
fn face_shade() -> array<f32, 6> {
    return array<f32, 6>(0.6, 0.6, 1.0, 0.5, 0.8, 0.8);
}

struct VsOut {
    @builtin(position) pos: vec4f,
    // UV in whole-texture tiles: one tile per block.
    @location(0) uv: vec2f,
    @location(1) @interpolate(flat) layer: u32,
    // Directional shade times the corner's AO factor, interpolated.
    @location(2) shade: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) origin_index: u32,
) -> VsOut {
    let face = faces[vertex / 6u];

    // Word 0: x | y<<5 | z<<10 | (width-1)<<15 | (height-1)<<20 | dir<<25.
    let bx = f32(face.x & 31u);
    let by = f32((face.x >> 5u) & 31u);
    let bz = f32((face.x >> 10u) & 31u);
    let width = f32(((face.x >> 15u) & 31u) + 1u);
    let height = f32(((face.x >> 20u) & 31u) + 1u);
    let direction = (face.x >> 25u) & 7u;

    // Word 1: layer | ao c0..c3 (2 bits each) | light<<24.
    let layer = (face.y & 0xFFFFu);
    let ao0 = (face.y >> 16u) & 3u;
    let ao1 = (face.y >> 18u) & 3u;
    let ao2 = (face.y >> 20u) & 3u;
    let ao3 = (face.y >> 22u) & 3u;

    // Quad corners c0..c3 and the AO table use the same slot order, so
    // corner_table indexes both. The diagonal is picked by AO: split
    // along the darker pair to avoid interpolation anisotropy.
    var corners = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
        vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
    );
    var corner_ao = array<u32, 6>(ao0, ao1, ao2, ao0, ao2, ao3);
    if (ao0 + ao2) > (ao1 + ao3) {
        corners = array<vec2f, 6>(
            vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
            vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 0.0),
        );
        corner_ao = array<u32, 6>(ao1, ao2, ao3, ao1, ao3, ao0);
    }
    let slot = vertex % 6u;
    let c = corners[slot];

    let n = normals()[direction];
    let ua = u_axes()[direction];
    let va = v_axes()[direction];
    // The face plane sits one block along the normal when it points
    // positive; at the block itself when negative.
    let block = vec3i(i32(bx), i32(by), i32(bz)) + max(n, vec3i(0));
    let offset = ua * i32(c.x * width) + va * i32(c.y * height);
    let relative = origins[origin_index].xyz + vec3f(block + offset);

    // v is world-up on side faces and texture v grows downward, so the
    // image's top row lands at the block's top.
    let uv = vec2f(c.x * width, (1.0 - c.y) * height);

    let ao_factor = array<f32, 4>(0.4, 0.6, 0.8, 1.0);
    var out: VsOut;
    out.pos = u.view_proj * vec4f(relative, 1.0);
    out.uv = uv;
    out.layer = layer;
    out.shade = face_shade()[direction] * ao_factor[corner_ao[slot]];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // wgpu 30 follows the WGSL spec where the array layer is a separate
    // argument (not packed into coords.z).
    let tex = textureSample(block_textures, block_sampler, in.uv, in.layer);
    // Cutout: glass, leaves and holes in textures discard below 0.5.
    if (tex.a < 0.5) {
        discard;
    }
    return vec4f(tex.rgb * in.shade, 1.0);
}
