// GPU frustum + Hi-Z occlusion culling (task 2.5). One thread per meta
// slot; inactive slots exit. Two dispatches per frame:
//
//   cull_early (before render pass 1): slots that were visible last frame
//     and pass the frustum test emit their commands into the early half.
//     Drawing what was visible last frame first means the Hi-Z pyramid
//     built from that depth already contains the scene the player saw, so
//     nothing pops in when the camera moves.
//
//   cull_late (after the Hi-Z build, before render pass 2): every active
//     slot is tested against the frustum and the pyramid; visible = frustum
//     && !occluded is written back for next frame, and slots that were not
//     emitted early emit into the late half.
//
// Command emission: one command per non-empty, non-back-facing direction
// (the visible_directions rule of render/chunks.rs, ported with the same
// comparisons), appended into per-page regions with an atomicAdd on a
// per-page counter. Each frame the CPU sizes every region with the same
// frustum test and visible_directions rule (over the meta slots it
// tracks), so the bound is >= what this shader emits into it — Metal has
// no draw_indirect_count and emulates multi_draw_indirect as a CPU loop
// of one indirect draw per encoded command, so the encoded count has to
// come from the CPU anyway, and sizing to the frustum keeps the zero-tail
// encode cost proportional to what is visible. The CPU also clears the
// counters and zeroes the command prefix first, so a region's unused tail
// is zero-count draws: multi_draw_indirect is issued for the region's
// full bound and the zero-tail draws are no-ops. Should the GPU ever emit
// past the buffer capacity, the idx < capacity clamp below drops the
// write and counts it in the region_overflow stat (asserted zero by the
// tests). Command order inside a region is nondeterministic, which is
// fine for opaque depth-tested terrain.

struct FrameUniforms {
    view_proj: mat4x4f,
};

// Mirrored by `CubeMeta` in render/chunks.rs (80 bytes, 16-aligned; layout
// test there). dirs holds the six per-direction (offset, count) pairs,
// offsets relative to the cube's face range start.
struct CubeMeta {
    pos_page: vec4i,        // xyz = cube coordinates, w = page index
    start_flags: vec4u,     // x = face range start in the page, y = flags
    dirs: array<vec4u, 3>,
};

const ACTIVE: u32 = 1u;
const VISIBLE: u32 = 2u;

// Mirrors wgpu::util::DrawIndirectArgs (16 bytes).
struct DrawCmd {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32, // the meta slot, on every path
};

// Mirrored by `CullParams` in render/chunks.rs (64 bytes).
struct CullParams {
    camera_cube: vec4i, // xyz = floor(camera / CUBE_SIZE)
    camera_frac: vec4f, // xyz = camera - camera_cube * CUBE_SIZE
    screen: vec4u,      // x,y = depth size; z = max mip level; w = meta len
    regions: vec4u,     // x = late base (elements); y = page count; z = command capacity (elements); w = CUBE_SIZE
}

// Stats buffer indices; the overlay reads these one or two frames late.
const STAT_CANDIDATES: u32 = 0u;
const STAT_FRUSTUM_CULLED: u32 = 1u;
const STAT_OCCLUDED: u32 = 2u;
const STAT_DRAWN_EARLY: u32 = 3u;
const STAT_DRAWN_LATE: u32 = 4u;
const STAT_COMMANDS: u32 = 5u;
const STAT_FACES: u32 = 6u;
// Commands dropped by the capacity clamp: a bound bug (must stay zero).
const STAT_REGION_OVERFLOW: u32 = 7u;

// The projection's near plane (camera.rs NEAR); w of a projected point is
// -z_view, so w <= NEAR means at or behind the near plane.
const NEAR: f32 = 0.05;

@group(0) @binding(0) var<uniform> u: FrameUniforms;
@group(1) @binding(0) var<uniform> params: CullParams;
@group(1) @binding(1) var<storage, read_write> cube_meta: array<CubeMeta>;
@group(1) @binding(2) var<storage, read_write> commands: array<DrawCmd>;
@group(1) @binding(3) var<storage, read> region_base: array<u32>;
@group(1) @binding(4) var<storage, read_write> region_counter: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
@group(1) @binding(6) var pyramid: texture_2d<f32>;

// The one origin formula, identical to the vertex shader in terrain.wgsl
// and to `cube_origin` in render/chunks.rs: the integer part is exact, the
// only rounding is the final f32 subtract, so CPU and GPU cull on the same
// numbers and no absolute world coordinate ever reaches f32.
fn cube_origin(pos: vec3i) -> vec3f {
    let int_part = vec3f(pos - params.camera_cube.xyz) * f32(params.regions.w);
    return int_part - params.camera_frac.xyz;
}

// The plane test of `Frustum::intersects_aabb` (render/chunks.rs), ported
// with the same comparisons: Gribb-Hartmann rows of the rotation-only
// view-projection (the reverse-Z far plane is at infinity, so five planes).
// Row r of a column-major mat4x4f (glam stores columns).
fn matrix_row(m: mat4x4f, r: u32) -> vec4f {
    return vec4f(m[0][r], m[1][r], m[2][r], m[3][r]);
}

fn frustum_visible(rel_min: vec3f, rel_max: vec3f) -> bool {
    let m = u.view_proj;
    let r0 = matrix_row(m, 0u);
    let r1 = matrix_row(m, 1u);
    let r2 = matrix_row(m, 2u);
    let r3 = matrix_row(m, 3u);
    let planes = array<vec4f, 5>(r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2);
    for (var i = 0u; i < 5u; i++) {
        let plane = planes[i];
        // Support point farthest along the plane normal.
        let px = select(rel_min.x, rel_max.x, plane.x >= 0.0);
        let py = select(rel_min.y, rel_max.y, plane.y >= 0.0);
        let pz = select(rel_min.z, rel_max.z, plane.z >= 0.0);
        if (plane.x * px + plane.y * py + plane.z * pz + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

// `visible_directions` ported with the same comparisons (the camera sits
// at the origin): +X is skipped when the whole cube is at rel_min.x >= 0
// (every +X face plane is then at or beyond the camera and faces away),
// -X when rel_max.x <= 0, and likewise for Y and Z. Back-face culling
// discarded those faces anyway, so skipping them cannot change the image.
// Order matches mesh::face::Direction: +X, -X, +Y, -Y, +Z, -Z.
struct VisibleDirs {
    px: bool,
    nx: bool,
    py: bool,
    ny: bool,
    pz: bool,
    nz: bool,
}

fn visible_dirs(rel_min: vec3f, rel_max: vec3f) -> VisibleDirs {
    var v: VisibleDirs;
    v.px = rel_min.x < 0.0;
    v.nx = rel_max.x > 0.0;
    v.py = rel_min.y < 0.0;
    v.ny = rel_max.y > 0.0;
    v.pz = rel_min.z < 0.0;
    v.nz = rel_max.z > 0.0;
    return v;
}

fn dir_visible(dirs: VisibleDirs, d: u32) -> bool {
    switch d {
        case 0u: { return dirs.px; }
        case 1u: { return dirs.nx; }
        case 2u: { return dirs.py; }
        case 3u: { return dirs.ny; }
        case 4u: { return dirs.pz; }
        case 5u: { return dirs.nz; }
        default: { return false; }
    }
}

// Appends one command per non-empty, non-skipped direction into this
// phase's region for the cube's page. idx is bounds-checked against the
// command capacity: the CPU's frustum bound (region_base) is provably
// large enough for this shader's identical-math emission, so a drop here
// is a bound bug — it is counted in region_overflow for the tests to
// catch instead of corrupting memory.
fn emit(m: CubeMeta, slot: u32, phase_base: u32, rel_min: vec3f, rel_max: vec3f) {
    let page = u32(m.pos_page.w);
    let dirs = visible_dirs(rel_min, rel_max);
    let start = m.start_flags.x;
    for (var d = 0u; d < 6u; d++) {
        let off = m.dirs[d / 2u][(d % 2u) * 2u];
        let cnt = m.dirs[d / 2u][(d % 2u) * 2u + 1u];
        if (cnt == 0u || !dir_visible(dirs, d)) {
            continue;
        }
        let idx = phase_base + region_base[page] + atomicAdd(&region_counter[page], 1u);
        if (idx < params.regions.z) {
            commands[idx] = DrawCmd(cnt * 6u, 1u, (start + off) * 6u, slot);
        } else {
            atomicAdd(&stats[STAT_REGION_OVERFLOW], 1u);
        }
        atomicAdd(&stats[STAT_COMMANDS], 1u);
        atomicAdd(&stats[STAT_FACES], cnt);
    }
}

// The conservative Hi-Z occlusion test. Conservative means it may keep a
// cube that is actually hidden (every step errs toward keeping), but must
// never drop one that could contribute a pixel.
fn occluded(rel_min: vec3f, rel_max: vec3f) -> bool {
    // The AABB contains the camera: some face is visible.
    if (rel_min.x <= 0.0 && rel_max.x >= 0.0 && rel_min.y <= 0.0 && rel_max.y >= 0.0
        && rel_min.z <= 0.0 && rel_max.z >= 0.0)
    {
        return false;
    }

    // Project the 8 corners. reverse-Z: depth 1 at the near plane, 0 at
    // infinity, so the *max* z/w is the cube's *nearest* point.
    var nearest = -2.0;
    var min_px = vec2f(1e9);
    var max_px = vec2f(-1e9);
    for (var i = 0u; i < 8u; i++) {
        let corner = vec3f(
            select(rel_min.x, rel_max.x, (i & 1u) == 1u),
            select(rel_min.y, rel_max.y, (i & 2u) == 2u),
            select(rel_min.z, rel_max.z, (i & 4u) == 4u),
        );
        let p = u.view_proj * vec4f(corner, 1.0);
        // w = -z_view; w <= NEAR means at or behind the near plane, where
        // the projection is undefined: keep the cube.
        if (p.w <= NEAR) {
            return false;
        }
        let ndc = vec2f(p.x / p.w, p.y / p.w);
        // mip-0 texels run top-left origin: ndc.y +1 is texture row 0.
        let px = vec2f(
            (ndc.x * 0.5 + 0.5) * f32(params.screen.x),
            (0.5 - ndc.y * 0.5) * f32(params.screen.y),
        );
        min_px = min(min_px, px);
        max_px = max(max_px, px);
        nearest = max(nearest, p.z / p.w);
    }

    // Expand by one texel: the rect is f32 and the rasterizer may cover
    // slightly more. A larger rect only adds texels, and more texels can
    // only *lower* the occluder depth, which keeps more cubes.
    let lo = floor(min_px) - vec2f(1.0);
    let hi = ceil(max_px) + vec2f(1.0);

    // Pick the mip level at which the rect spans at most 2x2 texels.
    let span = max(max(hi.x - lo.x, hi.y - lo.y), 1.0);
    let level = u32(clamp(floor(log2(span)), 0.0, f32(params.screen.z)));

    // The texels covering the rect at that level, clamped to the texture.
    let scale = f32(1u << level);
    // The mip size comes from the depth size, not textureDimensions: the
    // pyramid's floor-halved chain (render/hiz.rs) is exactly
    // max(1, size >> level), and lavapipe's textureDimensions(t, level)
    // reported a mip larger than 1x1 at the top level, so the clamp let
    // reads run past the mip, returned 0 (= infinitely far in reverse-Z)
    // and silently disabled occlusion for those cubes.
    let mip_size = max(vec2i(1), vec2i(params.screen.xy >> vec2u(level)));
    let mlo = clamp(vec2i(floor(lo / scale)), vec2i(0), mip_size - vec2i(1));
    let mhi = clamp(vec2i(floor(hi / scale)), vec2i(0), mip_size - vec2i(1));

    // The farthest occluder depth is the min over those texels (reverse-Z:
    // the min of a 2x2 block is the farthest surface in it, all the way up
    // the pyramid).
    var farthest = 2.0;
    for (var y = mlo.y; y <= mhi.y; y++) {
        for (var x = mlo.x; x <= mhi.x; x++) {
            farthest = min(farthest, textureLoad(pyramid, vec2i(x, y), i32(level)).r);
        }
    }

    // Occluded iff the cube's nearest point is still farther than the
    // farthest occluder: every pixel of the cube then fails the Greater
    // depth test against something already drawn. Strict < keeps cubes
    // exactly flush with an occluder (their own coplanar surfaces).
    return nearest < farthest;
}

@compute @workgroup_size(64)
fn cull_early(@builtin(global_invocation_id) gid: vec3u) {
    let slot = gid.x;
    if (slot >= params.screen.w) {
        return;
    }
    let m = cube_meta[slot];
    if ((m.start_flags.y & ACTIVE) == 0u) {
        return;
    }
    let rel_min = cube_origin(m.pos_page.xyz);
    let rel_max = rel_min + vec3f(f32(params.regions.w));
    // Draw what was visible last frame; the early phase never touches the
    // visible bit (the late phase needs last frame's value).
    if ((m.start_flags.y & VISIBLE) != 0u && frustum_visible(rel_min, rel_max)) {
        atomicAdd(&stats[STAT_DRAWN_EARLY], 1u);
        emit(m, slot, 0u, rel_min, rel_max);
    }
}

@compute @workgroup_size(64)
fn cull_late(@builtin(global_invocation_id) gid: vec3u) {
    let slot = gid.x;
    if (slot >= params.screen.w) {
        return;
    }
    let m = cube_meta[slot];
    if ((m.start_flags.y & ACTIVE) == 0u) {
        return;
    }
    atomicAdd(&stats[STAT_CANDIDATES], 1u);
    let rel_min = cube_origin(m.pos_page.xyz);
    let rel_max = rel_min + vec3f(f32(params.regions.w));
    var visible = true;
    if (!frustum_visible(rel_min, rel_max)) {
        atomicAdd(&stats[STAT_FRUSTUM_CULLED], 1u);
        visible = false;
    } else if (occluded(rel_min, rel_max)) {
        atomicAdd(&stats[STAT_OCCLUDED], 1u);
        visible = false;
    }
    if (!visible) {
        cube_meta[slot].start_flags.y = m.start_flags.y & ~VISIBLE;
        return;
    }
    cube_meta[slot].start_flags.y = m.start_flags.y | VISIBLE;
    // Emitted early this frame iff it was visible last frame and passed
    // the same frustum test (deterministic), so only the rest draw late.
    if ((m.start_flags.y & VISIBLE) == 0u) {
        atomicAdd(&stats[STAT_DRAWN_LATE], 1u);
        emit(m, slot, params.regions.x, rel_min, rel_max);
    }
}
