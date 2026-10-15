// Terrain pass: vertex pulling. All cube meshes share a few large face
// "pages" (storage buffers; render/suballoc.rs hands out face ranges) and
// group 1 binds the page the current draw reads from. The vertex shader
// expands face (vertex_index / 6) into a quad and the fragment shader
// samples the block texture array. The bit layout mirrors the comment
// block in crates/client/src/mesh/face.rs — keep the two files in sync.
//
// Bind groups:
//   0: shared frame uniforms (same layout as sky.wgsl)
//   1: the face page's storage buffer (array<vec2u>, one vec2 per face)
//   2: shared terrain resources: cube meta, the terrain-only uniform,
//      textures, sampler
//
// Each draw covers one cube and one face direction: vertex_count faces * 6
// vertices, one instance, with first_vertex = (page range start + direction
// range start) * 6 and first_instance = the cube's slot in the persistent
// cube_meta buffer (on every path; the GPU cull writes the commands with
// the same slot). vertex_index includes first_vertex — WebGPU sets
// vertexIndex to firstVertex + i for a non-indexed draw (WGSL
// @builtin(vertex_index)), and wgpu passes the draw range start straight
// through — so faces[vertex / 6] indexes page-local faces. instance_index
// likewise includes first_instance, selecting the cube's meta entry.
//
// The vertex shader computes the camera-relative origin itself (see
// cube_origin below): the integer cube-grid part is exact in f32 and the
// only rounding is the final subtract of the camera fraction, so CPU and
// GPU cull on the same numbers and no absolute world coordinate ever
// reaches f32.

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

// Mirrored by `CubeMeta` in render/chunks.rs (80 bytes, 16-aligned; layout
// test there). Written once per upload/re-meshesh or coverage change,
// never per frame.
struct CubeMeta {
    pos_page: vec4i,        // xyz = cube coordinates, w = page index
    // x = face range start in the page; y = flags: bit 0 active (the
    // coverage rule draws this cube), bit 1 visible-last-frame, bits
    // 8..12 = LOD level, bits 16..24 = octant mask (bit o set when a
    // ready child replaces octant o); z = skirt face count, w spare.
    start_flags: vec4u,
    dirs: array<vec4u, 3>,  // six per-direction (offset, count) pairs
};

// Mirrored by `TerrainUniforms` in render/chunks.rs (32 bytes). Terrain
// only: sky, outline and overlay share just FrameUniforms.
struct TerrainUniforms {
    camera_cube: vec4i, // xyz = floor(camera / CUBE_SIZE)
    camera_frac: vec4f, // xyz = camera - camera_cube * CUBE_SIZE (f32); w = CUBE_SIZE
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;
@group(1) @binding(0) var<storage, read> faces: array<vec2u>;
@group(2) @binding(0) var<storage, read> cube_meta: array<CubeMeta>;
@group(2) @binding(1) var<uniform> terrain: TerrainUniforms;
@group(2) @binding(2) var block_textures: texture_2d_array<f32>;
@group(2) @binding(3) var block_sampler: sampler;

// The one origin formula, scaled by the cube's LOD level and shared by
// this shader, cull.wgsl and `cube_origin` in render/chunks.rs:
// origin = (pos * 2^level - camera_cube) * CUBE_SIZE - camera_frac
// (CUBE_SIZE rides in camera_frac.w, so the shader is cube-size
// agnostic). A level-n cube at pos covers blocks pos*CUBE_SIZE*2^n
// onwards. The integer product stays inside i32 for any position a
// server can stream (the CPU mirror asserts it), and the only rounding
// is the final f32 subtract — one rounding total, and the CPU path
// computes the very same value.
fn cube_origin(pos: vec3i, level: u32) -> vec3f {
    // pos * 2^level stays inside i32 for any position a server can
    // stream (the CPU mirror debug-asserts it).
    let scaled = pos * vec3i(i32(1u << level)) - terrain.camera_cube.xyz;
    let int_part = vec3f(scaled) * terrain.camera_frac.w;
    return int_part - terrain.camera_frac.xyz;
}

// The cell scale of a level: cell offsets and UVs multiply by 2^level,
// since one cell spans 2^level blocks.
fn cell_scale(level: u32) -> f32 {
    return f32(1u << level);
}

// The in-plane quad inset factor (see vs_main): 1 - 2^-16.
const EDGE_INSET_BLOCKS: f32 = 0.0;

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
    @builtin(instance_index) slot: u32,
) -> VsOut {
    let face = faces[vertex / 6u];
    let m = cube_meta[slot];

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
    let corner_index = vertex % 6u;
    // The cube's LOD level (meta flags, bits 8..12) scales every cell to
    // 2^level blocks (task 2.6).
    let level = (m.start_flags.y >> 8u) & 15u;
    let scale = cell_scale(level);
    let c = corners[corner_index];

    let n = normals()[direction];
    let ua = u_axes()[direction];
    let va = v_axes()[direction];
    // The quad starts at the owning block's corner where the face plane
    // and the u/v walk begin: one block along the normal when it points
    // positive, and one block along u or v when that axis points negative
    // (the owner is the first block along u/v, so a -Z walk starts at its
    // +Z side). Mirrored by `quad_corners` in mesh/face.rs.
    let block = vec3i(i32(bx), i32(by), i32(bz)) + max(n, vec3i(0))
        + max(-ua, vec3i(0)) + max(-va, vec3i(0));
    let offset = ua * i32(c.x * width) + va * i32(c.y * height);

    let relative = cube_origin(m.pos_page.xyz, level)
        + vec3f(block + offset) * scale;

    // v is world-up on side faces and texture v grows downward, so the
    // image's top row lands at the block's top. One texture tile per
    // BLOCK (not per cell), so tiles line up across ring borders.
    let uv = vec2f(c.x * width, (1.0 - c.y) * height) * scale;

    // Octant masking (level >= 1): a ready child replaces this octant, so
    // the cube's faces here must produce no fragments. Collapsing every
    // vertex of the face to one point makes all triangles degenerate.
    // Octants are indexed in CELL units: dx + 2*dz + 4*dy over the cell
    // halves (CUBE_SIZE / 2 rides in camera_frac.w / 2).
    let half_cell = u32(terrain.camera_frac.w / 2.0);
    let octant = u32(bx >= f32(half_cell))
        + 2u * u32(bz >= f32(half_cell))
        + 4u * u32(by >= f32(half_cell));
    if (level > 0u && ((m.start_flags.y >> 16u) >> octant & 1u) != 0u) {
        var out: VsOut;
        out.pos = vec4f(0.0, 0.0, 0.0, 1.0);
        out.uv = vec2f(0.0);
        out.layer = layer;
        out.shade = 0.0;
        return out;
    }

    let ao_factor = array<f32, 4>(0.4, 0.6, 0.8, 1.0);
    var out: VsOut;
    out.pos = u.view_proj * vec4f(relative, 1.0);
    out.uv = uv;
    out.layer = layer;
    out.shade = face_shade()[direction] * ao_factor[corner_ao[corner_index]];
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
