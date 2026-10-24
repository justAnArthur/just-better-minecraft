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
//      textures, sampler, per-layer average colors (task 2.9), per-layer
//      materials (task 2.15)
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
    // Task 2.15 (see FrameUniforms in render/renderer.rs).
    sun_dir: vec4f,        // xyz toward the sun, w = time (s)
    sun_color: vec4f,      // rgb sun radiance, w = sun lighting on
    ambient_sky: vec4f,    // rgb sky ambient, w = clouds on
    ambient_ground: vec4f, // rgb ground bounce, w = water shading on
    camera_wrap: vec4f,    // xyz camera position mod WRAP_PERIOD
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

// Mirrored by `TerrainUniforms` in render/chunks.rs (80 bytes). Terrain
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
    // Task 2.12 debug views. x = wireframe, y = wireframe-only,
    // z = explode gap in blocks (0 = off), w spare.
    debug: vec4f,
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

// Per-layer material (task 2.15), mirrored by `GpuLayerMaterial` in
// render/texture_array.rs (32 bytes; layout test there). Water and the
// far-side lean are decided per layer here, never by layer number.
struct LayerMaterial {
    // x = water (1.0) or not; y = the far-side lean strength (0..1);
    // z, w spare.
    flags: vec4f,
    // rgb = the linear color this layer's flat far color leans toward
    // (for grass_side: grass_top's average); w spare.
    far_side: vec4f,
};
@group(2) @binding(5) var<storage, read> layer_material: array<LayerMaterial>;

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
// The haze (task 2.15, replacing 2.9's plain distance fog) then blends
// every terrain fragment toward the sky along its view ray — see the
// haze block below. Its hard end is a smoothstep from FOG_START_FRACTION
// to FOG_END_FRACTION of the furthest ring radius (terrain.far.y),
// reaching 1 just inside the outermost ring so its edge is never visible.
// terrain.far.z switches it off entirely (Renderer::set_fog / JBM_NO_FOG;
// the fade stays).
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
    // The far-side lean (task 2.15): a layer whose top band differs from
    // its body (grass_side) leans its flat color toward the band's
    // material by the fade, so mid-distance terrace sides read as shaded
    // grass, not brown contour stripes. Level >= 1 is fully faded, so
    // the LOD0/L1 border still matches exactly.
    let material = layer_material[layer];
    let f = select(fade(dist), 1.0, level >= 1u);
    let avg = mix(layer_avg[layer].rgb, material.far_side.rgb, material.flags.y * f);
    if (level >= 1u) {
        return avg;
    }
    return mix(tex.rgb, avg, f);
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
    // The corner's (level-softened) AO factor, interpolated (task 2.15:
    // the fragment shader lights with it; it used to arrive premultiplied
    // by a fixed per-direction shade).
    @location(2) ao: f32,
    // The camera-relative position (task 2.9): its length is the fragment
    // distance the fade and fog smoothsteps run on. Interpolation is
    // exact for positions (affine in clip space via perspective-correct
    // varying interpolation) and the length is taken per fragment.
    @location(3) rel: vec3f,
    // The cube's LOD level (task 2.9), flat: far_color switches on it.
    @location(4) @interpolate(flat) level: u32,
    // Barycentric coordinates within the triangle (task 2.12): one
    // component is 1 at each of the triangle's corners. The wireframe
    // view draws a fragment as edge where the smallest one is within a
    // pixel or so of zero, measured in screen space (fwidth).
    @location(5) bary: vec3f,
    // The face direction (task 2.15), flat: the fragment shader's normal.
    @location(6) @interpolate(flat) dir: u32,
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
    var relative = cube_origin(m.pos_page.xyz, level)
        + vec3f(block + offset) * scale;
    // The exploded-cube debug view (task 2.12): shift every cube by the
    // gap times its own cube-grid offset, spreading the world apart
    // radially from the world origin. A pure world-space translation, so
    // it applies the same in camera-relative space. The cull skips its
    // frustum test while this is on (the moved cubes sit outside every
    // conservative bound) — see cull.wgsl.
    if (terrain.debug.z > 0.0) {
        let cube_offset = vec3f(m.pos_page.xyz * i32(1u << level)) * terrain.camera_frac.w;
        relative += terrain.debug.z * cube_offset;
    }
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
        out.ao = 0.0;
        out.dir = 0u;
        out.rel = vec3f(0.0);
        out.level = 0u;
        out.bary = vec3f(1.0);
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
    out.ao = ao;
    out.dir = direction;
    out.rel = relative_expanded;
    out.level = level;
    out.bary = array<vec3f, 3>(
        vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0),
    )[corner_index % 3u];
    return out;
}

// The wireframe debug view's edge test (task 2.12): coverage in 0..1 of
// the triangle's edges, about one pixel wide whatever the distance or
// the triangle's size, antialiased over one pixel. Screen-space
// derivatives, so it must run in uniform control flow: fs_main calls it
// only behind the uniform debug flags, before anything can discard.
const WIRE_PIXELS: f32 = 0.5;

fn wireframe_edge(in: VsOut) -> f32 {
    let pixels = in.bary / max(fwidth(in.bary), vec3f(1e-6));
    let nearest = min(pixels.x, min(pixels.y, pixels.z));
    return 1.0 - smoothstep(WIRE_PIXELS - 0.5, WIRE_PIXELS + 0.5, nearest);
}

// ---------------------------------------------------------------------------
// Task 2.15: sun and sky light, water, and the haze.
//
// Lighting. A fixed sun (u.sun_dir, u.sun_color) and a hemisphere ambient
// (u.ambient_sky from above, u.ambient_ground from below, blended by the
// normal's y). The sky is brighter on the sun's side, so the ambient
// leans by AMBIENT_SUN_SIDE toward faces turned to the sun's azimuth —
// with the sun off-axis that gives all four sides of a block their own
// value even in shadow. Per fragment:
//   albedo * (ambient(n) * ao + sun * max(dot(n, L), 0) * mix(1, ao, SUN_AO))
// where n is the face normal and ao the (level-softened) corner factor:
// AO darkens the ambient fully and the direct sun by SUN_AO. Every level
// lights the same way, so LOD0 and L1+ match at the border. With
// u.sun_color.w == 0 (lighting off) the pre-2.15 flat look returns: the
// fixed per-direction multipliers (top 1.0, +/-Z 0.8, +/-X 0.6, bottom
// 0.5) times AO, as a formula of the normal.
const AMBIENT_SUN_SIDE: f32 = 0.10;
const SUN_AO: f32 = 0.5;

fn face_normal(dir: u32) -> vec3f {
    return vec3f(normals()[dir]);
}

fn ambient(n: vec3f) -> vec3f {
    let hemi = mix(u.ambient_ground.rgb, u.ambient_sky.rgb, 0.5 + 0.5 * n.y);
    let sun_side = normalize(u.sun_dir.xz + vec2f(1e-4, 0.0));
    return hemi * (1.0 + AMBIENT_SUN_SIDE * dot(n.xz, sun_side));
}

fn lighting(n: vec3f, ao: f32) -> vec3f {
    if (u.sun_color.w < 0.5) {
        let up_down = select(0.5, 1.0, n.y > 0.0);
        return vec3f(dot(abs(n), vec3f(0.6, up_down, 0.8)) * ao);
    }
    let direct = max(dot(n, u.sun_dir.xyz), 0.0);
    return ambient(n) * ao + u.sun_color.rgb * direct * mix(1.0, ao, SUN_AO);
}

// The sky seen along `dir` as the terrain shading knows it: the shared
// gradient plus the sun's glow (no disk, no clouds — the haze meets the
// sky at and just above the horizon, where clouds have thinned to
// nothing). y is clamped just above the horizon: fogged terrain meets
// the horizon color the sky shows at the horizon line, and reflections
// never look below it.
fn sky_along(dir: vec3f) -> vec3f {
    let d = vec3f(dir.x, max(dir.y, FOG_DIR_MIN_Y), dir.z);
    return sky_gradient(d, u.horizon_color.rgb, u.zenith_color.rgb, u.ground_color.rgb)
        + sun_glow(d, u.sun_dir.xyz, u.sun_color.rgb);
}

// Water (task 2.15). Still an opaque block in the opaque pass; the
// fragment shader makes it read as water:
//   - a deep-water base color, lit like any face;
//   - a Schlick Fresnel blend toward the sky reflected about the surface
//     normal (sky_along: the shared gradient and sun glow), so water
//     mirrors the sky at grazing angles and shows its depth looking down;
//   - a sun glint (a tight specular lobe around the reflected sun);
//   - procedural ripple normals on up-facing faces: a few sine waves over
//     the world position (camera_wrap + rel; every wave vector is a
//     whole number of cycles per WRAP_PERIOD, so the pattern tiles the
//     wrap exactly), animated by the time uniform (0 in tests). Short
//     ripples fade out with distance (RIPPLE_FADE_*) and a long, gentle
//     swell further out (SWELL_FADE_*), before either could alias, so
//     far water is a calm mirror. The same function shades water at every
//     level, from the same world position, so the LOD0/L1 seam cannot
//     show.
const WATER_DEEP: vec3f = vec3f(0.018, 0.080, 0.20);
const WATER_F0: f32 = 0.035;
const WATER_REFLECT_MAX: f32 = 0.85;
const WATER_GLINT: f32 = 2.2;
const WATER_GLINT_POWER: f32 = 600.0;
const RIPPLE_STRENGTH: f32 = 0.06;
const RIPPLE_FADE_START: f32 = 24.0;
const RIPPLE_FADE_END: f32 = 220.0;
// The swell: three long, gentle waves (~45..64 blocks) that keep the
// open water from reading as a flat plate where the ripples have faded;
// gone before they would alias (SWELL_FADE_*).
const SWELL_STRENGTH: f32 = 0.028;
const SWELL_FADE_START: f32 = 160.0;
const SWELL_FADE_END: f32 = 1400.0;

// One ripple wave's slope: the gradient direction of
// sin(2*pi*(k . p / WRAP + speed * t)) scaled to unit steepness (every
// wave as steep as the others, whatever its length). k is in whole cycles
// per WRAP_PERIOD (65536 blocks); the phase is reduced with fract before
// the cos, so large k . p stays precise.
fn ripple_slope(p: vec2f, k: vec2f, speed: f32) -> vec2f {
    let phase = fract(dot(k, p) / 65536.0 + speed * u.sun_dir.w);
    return cos(phase * 6.2831853) * normalize(k);
}

fn water_normal(n: vec3f, rel: vec3f, dist: f32) -> vec3f {
    if (n.y < 0.5) {
        return n;
    }
    let ripple = RIPPLE_STRENGTH * (1.0 - smoothstep(RIPPLE_FADE_START, RIPPLE_FADE_END, dist));
    let swell = SWELL_STRENGTH * (1.0 - smoothstep(SWELL_FADE_START, SWELL_FADE_END, dist));
    if (ripple + swell <= 0.0) {
        return n;
    }
    // camera_wrap is already reduced mod WRAP_PERIOD and rel is small
    // where the waves show, so p stays precise.
    let p = u.camera_wrap.xz + rel.xz;
    var slope = vec2f(0.0);
    if (ripple > 0.0) {
        // Five ripples, wavelengths ~3.6..9.6 blocks, in five directions.
        var r = ripple_slope(p, vec2f(6400.0, 2368.0), 0.11);
        r += ripple_slope(p, vec2f(-3904.0, 9600.0), 0.15);
        r += ripple_slope(p, vec2f(12160.0, -6080.0), 0.19);
        r += ripple_slope(p, vec2f(2368.0, 17920.0), 0.23);
        r += ripple_slope(p, vec2f(-15040.0, -9024.0), 0.17);
        slope += r * ripple;
    }
    var w = ripple_slope(p, vec2f(896.0, 512.0), 0.021);
    w += ripple_slope(p, vec2f(-640.0, 1088.0), 0.027);
    w += ripple_slope(p, vec2f(1408.0, -320.0), 0.033);
    slope += w * swell;
    return normalize(vec3f(-slope.x, 1.0, -slope.y));
}

fn water_color(n_face: vec3f, rel: vec3f, dist: f32, ao: f32) -> vec3f {
    let n = water_normal(n_face, rel, dist);
    let view = -rel / dist; // toward the eye
    let cos_v = clamp(dot(n, view), 0.0, 1.0);
    let fresnel = WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - cos_v, 5.0);
    let r = reflect(-view, n);
    let reflected = sky_along(r);
    let deep = WATER_DEEP * lighting(n_face, ao);
    var color = mix(deep, reflected, min(fresnel, WATER_REFLECT_MAX));
    let glint = pow(max(dot(r, u.sun_dir.xyz), 0.0), WATER_GLINT_POWER);
    color += u.sun_color.rgb * glint * WATER_GLINT * select(0.0, 1.0, r.y > 0.0);
    return color;
}

// The haze: height-aware aerial perspective (task 2.15). Light from the
// fragment is attenuated (extinction) and replaced by sky light scattered
// in along the ray (in-scatter: sky_along, so the sun's glow shows in the
// sun's direction), in proportion to the ray's optical depth through an
// exponential height fog: density HAZE_DENSITY at HAZE_BASE_Y (about sea
// level), falling by e every HAZE_SCALE_HEIGHT blocks up — thick in
// valleys, thin from altitude. The optical depth of a straight ray from
// height y0 rising dy over its length d integrates in closed form:
//   tau = HAZE_DENSITY * exp(-(y0 - base) / H) * d * (1 - exp(-dy / H)) / (dy / H)
// The haze amount is 1 - exp(-tau), combined with the hard end (the
// FOG_* smoothstep of the furthest ring radius) as 1 - (1 - h)(1 - end),
// so the outermost ring always vanishes into the sky behind it.
const HAZE_DENSITY: f32 = 1.6e-4;
const HAZE_BASE_Y: f32 = 63.0;
const HAZE_SCALE_HEIGHT: f32 = 160.0;

fn haze_amount(rel: vec3f, dist: f32) -> f32 {
    let camera_y = u.camera_pos_hi.y + u.camera_pos_lo.y;
    let h0 = clamp((camera_y - HAZE_BASE_Y) / HAZE_SCALE_HEIGHT, -2.0, 30.0);
    let k = clamp(rel.y / HAZE_SCALE_HEIGHT, -30.0, 30.0);
    var integral = 1.0;
    if (abs(k) > 1e-3) {
        integral = (1.0 - exp(-k)) / k;
    }
    let tau = HAZE_DENSITY * exp(-h0) * dist * integral;
    let haze = 1.0 - exp(-tau);
    let end = fog_factor(dist);
    return (1.0 - (1.0 - haze) * (1.0 - end)) * terrain.far.z;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // Wireframe-only (task 2.12): the edges over a flat background — no
    // textures, shade, tint or fog. Bright lines on near-black, so the
    // pure geometry reads at any distance.
    var edge = 0.0;
    if (terrain.debug.x > 0.5 || terrain.debug.y > 0.5) {
        edge = wireframe_edge(in);
    }
    if (terrain.debug.y > 0.5) {
        return vec4f(mix(vec3f(0.04), vec3f(0.95), edge), 1.0);
    }
    // wgpu 30 follows the WGSL spec where the array layer is a separate
    // argument (not packed into coords.z). The sample must stay in uniform
    // control flow (WGSL requires it for textureSample), and the LOD level
    // arrives as a flat varying the uniformity analysis cannot prove
    // constant — so the level >= 1 path fetches the texel it then ignores;
    // far_color drops its value exactly (mix(x, y, 1.0) == y).
    let tex = textureSample(block_textures, block_sampler, in.uv, in.layer);
    let dist = length(in.rel);
    // Cutout: glass, leaves and holes in textures discard below 0.5 —
    // only while any texture still shows (level 0, not fully faded; a LOD
    // cell is solid by the downsampling rule). The alpha test leads so
    // opaque fragments (almost all terrain) never evaluate the fade.
    if (tex.a < 0.5 && in.level == 0u && fade(dist) < 1.0) {
        discard;
    }
    let n = face_normal(in.dir);
    var color: vec3f;
    if (layer_material[in.layer].flags.x > 0.5 && u.ambient_ground.w > 0.5) {
        color = water_color(n, in.rel, dist, in.ao);
    } else {
        color = far_color(tex, in.layer, dist, in.level) * lighting(n, in.ao);
    }
    // Level tint (task 2.10): the debug view that shows which level draws
    // where, before fog. One uniform flag, shared by every draw path
    // (they share this fragment shader).
    if (terrain.tint.x > 0.5) {
        color = color * lod_tint(in.level);
    }
    // The haze (see above): toward the sky along this fragment's ray. The
    // direction reuses the already-computed distance instead of a second
    // length() (normalize).
    let h = haze_amount(in.rel, dist);
    if (h > 0.0) {
        color = mix(color, sky_along(in.rel / dist), h);
    }
    // Wireframe (task 2.12): dark lines over the shaded terrain, drawn
    // last so they read through fog.
    if (terrain.debug.x > 0.5) {
        color = mix(color, vec3f(0.05), 0.7 * edge);
    }
    return vec4f(color, 1.0);
}
