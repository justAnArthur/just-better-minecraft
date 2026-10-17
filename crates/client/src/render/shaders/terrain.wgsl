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
//      textures, sampler, per-layer average colors (task 2.9)
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

// Mirrored by `TerrainUniforms` in render/chunks.rs (64 bytes). Terrain
// only: sky, outline and overlay share just FrameUniforms.
struct TerrainUniforms {
    camera_cube: vec4i, // xyz = floor(camera / CUBE_SIZE)
    camera_frac: vec4f, // xyz = camera - camera_cube * CUBE_SIZE (f32); w = CUBE_SIZE
    // Task 2.9 far look. x = the LOD0 ring radius in blocks (sets the
    // texture-to-flat fade window); y = the furthest ring radius in blocks
    // (sets the fog window); z = fog on (1.0) or off (0.0, the
    // Renderer::set_fog / JBM_NO_FOG hook); w spare.
    far: vec4f,
    // Task 2.10 level tint. x = tint on (1.0) or off (0.0, the F4 /
    // JBM_LOD_TINT hook); y, z, w spare.
    tint: vec4f,
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;
@group(1) @binding(0) var<storage, read> faces: array<vec2u>;
@group(2) @binding(0) var<storage, read> cube_meta: array<CubeMeta>;
@group(2) @binding(1) var<uniform> terrain: TerrainUniforms;
@group(2) @binding(2) var block_textures: texture_2d_array<f32>;
@group(2) @binding(3) var block_sampler: sampler;
// One vec4f per texture layer: the alpha-weighted linear average color
// (task 2.9), computed by BlockTextures at load.
@group(2) @binding(4) var<storage, read> layer_avg: array<vec4f>;

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

// The outward quad expansion, in blocks (see vs_main).
const EDGE_EXPAND_BLOCKS: f32 = 1.0 / 512.0;

// The cell scale of a level: cell offsets and UVs multiply by 2^level,
// since one cell spans 2^level blocks.
fn cell_scale(level: u32) -> f32 {
    return f32(1u << level);
}

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

// ---------------------------------------------------------------------------
// Task 2.9: the far look — flat material colors and distance fog.
//
// far_color(layer, dist, level) is the base color before directional shade
// and fog:
//   - level >= 1 (a LOD cube): the layer's average color, flat. No cutout
//     discard — a LOD cell is solid by the downsampling rule.
//   - level 0: mix(texture, average, fade(dist)) where fade is a
//     smoothstep from FADE_START_FRACTION to FADE_END_FRACTION of the
//     LOD0 ring radius (terrain.far.x). By the LOD0/LOD1 border (85% of
//     the radius) LOD0 is flat-colored too, so both sides of the border
//     match. The cutout discard still applies while fade < 1, i.e. while
//     any of the texture still shows.
//
// Distance fog then blends every terrain fragment into the sky color
// along its view ray at the horizon (sky_gradient below, y clamped just
// above 0): a smoothstep from FOG_START_FRACTION to FOG_END_FRACTION of
// the furthest ring radius (terrain.far.y), reaching 1 just inside the
// outermost ring so its edge is never visible. terrain.far.z switches the
// fog off entirely (Renderer::set_fog / JBM_NO_FOG; the fade stays).
//
// The fractions are mirrored by the FADE_* / FOG_* constants in
// render/chunks.rs (tests compute expected values from them) — keep the
// two in sync.
const FADE_START_FRACTION: f32 = 0.40;
const FADE_END_FRACTION: f32 = 0.85;
const FOG_START_FRACTION: f32 = 0.35;
const FOG_END_FRACTION: f32 = 0.98;
// The fog color's view direction clamps y to just above zero (never a
// negative pow argument, and the fogged terrain meets the horizon color
// the sky shows at the horizon line).
const FOG_DIR_MIN_Y: f32 = 1e-6;

fn fade(dist: f32) -> f32 {
    return smoothstep(
        FADE_START_FRACTION * terrain.far.x,
        FADE_END_FRACTION * terrain.far.x,
        dist,
    );
}

fn fog_factor(dist: f32) -> f32 {
    return smoothstep(
        FOG_START_FRACTION * terrain.far.y,
        FOG_END_FRACTION * terrain.far.y,
        dist,
    ) * terrain.far.z;
}

// The base color of a fragment before shade and fog (see the block
// comment above). `tex` is the already-sampled texture (the sample itself
// must stay in uniform control flow).
fn far_color(tex: vec4f, layer: u32, dist: f32, level: u32) -> vec3f {
    let avg = layer_avg[layer].rgb;
    if (level >= 1u) {
        return avg;
    }
    return mix(tex.rgb, avg, fade(dist));
}

// AO softening per LOD level (task 2.9): AO contrast scales with cell
// size and reads blotchy far away, so coarse levels pull the AO factor
// toward fully open. Level 0 keeps the exact factor (pre-2.9 goldens stay
// bit-stable). Mirrored by AO_SOFTEN in render/chunks.rs — keep in sync.
fn ao_soften(level: u32) -> f32 {
    if (level == 0u) {
        return 1.0;
    }
    if (level == 1u) {
        return 0.6;
    }
    if (level == 2u) {
        return 0.45;
    }
    return 0.3;
}

// The level tint (task 2.10), mirrored by LOD_TINT_COLORS in
// render/chunks.rs — keep in sync. One fixed color per level so a debug
// view shows which level draws where; level 0 stays white. The last entry
// is purple, deliberately not pure magenta (the hole sentinel).
fn lod_tint(level: u32) -> vec3f {
    let colors = array<vec3f, 7>(
        vec3f(1.0, 1.0, 1.0),   // LOD0: untinted
        vec3f(1.0, 0.0, 0.0),   // L1 2x2 red
        vec3f(1.0, 1.0, 0.0),   // L2 4x4 yellow
        vec3f(0.0, 1.0, 0.0),   // L3 8x8 green
        vec3f(0.0, 1.0, 1.0),   // L4 16x16 cyan
        vec3f(0.0, 0.0, 1.0),   // L5 32x32 blue
        vec3f(0.55, 0.15, 1.0), // L6 64x64 purple
    );
    return colors[min(level, 6u)];
}

struct VsOut {
    @builtin(position) pos: vec4f,
    // UV in whole-texture tiles: one tile per block.
    @location(0) uv: vec2f,
    @location(1) @interpolate(flat) layer: u32,
    // Directional shade times the corner's (level-softened) AO factor,
    // interpolated.
    @location(2) shade: f32,
    // The camera-relative position (task 2.9): its length is the fragment
    // distance the fade and fog smoothsteps run on. Interpolation is
    // exact for positions (affine in clip space via perspective-correct
    // varying interpolation) and the length is taken per fragment.
    @location(3) rel: vec3f,
    // The cube's LOD level (task 2.9), flat: far_color switches on it.
    @location(4) @interpolate(flat) level: u32,
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

    // The cube's LOD level (meta flags, bits 8..12) scales every cell
    // to 2^level blocks (task 2.6).
    let level = (m.start_flags.y >> 8u) & 15u;
    let scale = cell_scale(level);
    let relative = cube_origin(m.pos_page.xyz, level)
        + vec3f(block + offset) * scale;
    // Sub-pixel expansion (level >= 1 only): greedy merging leaves
    // T-junctions where a merged quad's edge abuts smaller quads, and the
    // two edges' interpolation can differ by ulps — a hairline crack the
    // eye sees straight through (there is no face behind terrain).
    // Pushing every corner ~1/512 block outward closes those cracks with
    // an overlap that is invisible: coplanar neighbours carry the same
    // texture, and the step-edge slivers are far below a pixel. LOD0
    // skips it to keep the pre-LOD goldens bit-stable; its pinholes are
    // single pixels and bounded by the hole tests instead.
    let expand_blocks = select(0.0, EDGE_EXPAND_BLOCKS, level > 0u);
    let expand = vec3f(ua) * ((c.x - 0.5) * 2.0 * expand_blocks)
        + vec3f(va) * ((c.y - 0.5) * 2.0 * expand_blocks);
    let relative_expanded = relative + expand;

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
        out.rel = vec3f(0.0);
        out.level = 0u;
        return out;
    }

    let ao_factor = array<f32, 4>(0.4, 0.6, 0.8, 1.0);
    // AO softening (task 2.9): pull the corner's AO factor toward fully
    // open by the level's factor — see ao_soften above.
    let ao = mix(1.0, ao_factor[corner_ao[corner_index]], ao_soften(level));
    var out: VsOut;
    out.pos = u.view_proj * vec4f(relative_expanded, 1.0);
    out.uv = uv;
    out.layer = layer;
    out.shade = face_shade()[direction] * ao;
    out.rel = relative_expanded;
    out.level = level;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // wgpu 30 follows the WGSL spec where the array layer is a separate
    // argument (not packed into coords.z). The sample must stay in uniform
    // control flow (WGSL requires it for textureSample), and the LOD level
    // arrives as a flat varying the uniformity analysis cannot prove
    // constant — so the level >= 1 path fetches the texel it then ignores;
    // far_color drops its value exactly (mix(x, y, 1.0) == y).
    let tex = textureSample(block_textures, block_sampler, in.uv, in.layer);
    let dist = length(in.rel);
    var base = far_color(tex, in.layer, dist, in.level);
    // Level tint (task 2.10): the debug view that shows which level draws
    // where, before shade and fog. One uniform flag, shared by every draw
    // path (they share this fragment shader).
    if (terrain.tint.x > 0.5) {
        base = base * lod_tint(in.level);
    }
    // Cutout: glass, leaves and holes in textures discard below 0.5 —
    // only while any texture still shows (level 0, not fully faded; a LOD
    // cell is solid by the downsampling rule). The alpha test leads so
    // opaque fragments (almost all terrain) never evaluate the fade.
    if (tex.a < 0.5 && in.level == 0u && fade(dist) < 1.0) {
        discard;
    }
    var color = base * in.shade;
    // Distance fog (task 2.9): blend into the sky color along this
    // fragment's view ray at the horizon — the same gradient the sky pass
    // paints behind the terrain — so the far rings dissolve into it with
    // no visible band. See the far-look block comment above. The
    // direction reuses the already-computed distance instead of a second
    // length() (normalize).
    let f = fog_factor(dist);
    if (f > 0.0) {
        let dir = in.rel / dist;
        let fog_color = sky_gradient(
            vec3f(dir.x, max(dir.y, FOG_DIR_MIN_Y), dir.z),
            u.horizon_color.rgb,
            u.zenith_color.rgb,
            u.ground_color.rgb,
        );
        color = mix(color, fog_color, f);
    }
    return vec4f(color, 1.0);
}
