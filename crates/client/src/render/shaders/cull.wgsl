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
// offsets relative to the cube's face range start. start_flags.y: bit 0
// active (the coverage rule draws this cube), bit 1 visible-last-frame,
// bits 8..12 = LOD level, bits 16..24 = octant mask.
struct CubeMeta {
    pos_page: vec4i,        // xyz = cube coordinates, w = page index
    start_flags: vec4u,     // x = face range start in the page, y = flags
    dirs: array<vec4u, 3>,
};

// The LOD level of a slot (bits 8..12 of the flags word).
fn level_of(m: CubeMeta) -> u32 {
    return (m.start_flags.y >> 8u) & 15u;
}

const ACTIVE: u32 = 1u;
const VISIBLE: u32 = 2u;

// Mirrors wgpu::util::DrawIndirectArgs (16 bytes).
struct DrawCmd {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32, // the meta slot, on every path
};

// Mirrored by `CullParams` in render/chunks.rs (80 bytes).
struct CullParams {
    camera_cube: vec4i, // xyz = floor(camera / CUBE_SIZE)
    camera_frac: vec4f, // xyz = camera - camera_cube * CUBE_SIZE
    screen: vec4u,      // x,y = depth size; z = max mip level; w = meta len
    regions: vec4u,     // x = late base (elements); y = page count; z = command capacity (elements); w = CUBE_SIZE
    // Task 2.12 debug views. x = shown-level mask (bit L = level L
    // draws); y = flags: bit 0 explode, bit 1 freeze; z, w spare.
    debug: vec4u,
    // The direction-split radius in blocks (task 2.13 C): x = the radius;
    // y, z, w spare.
    split: vec4f,
}

// The freeze-cull view (task 2.12), mirrored by `FreezeParams` in
// render/chunks.rs (96 bytes, binding 7): the view-projection and camera
// split captured when the view switched on. While the freeze flag is set
// the cull tests against this instead of the live camera, so what draws
// stops following the real view.
struct FreezeParams {
    view_proj: mat4x4f,
    camera_cube: vec4i,
    camera_frac: vec4f,
};

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
// Per-level counters (task 2.12): level L's drawn count at 8 + L and its
// face count at 15 + L, levels 0..=6 (the last entry collects anything
// beyond, which the shipped 7-level pyramid never produces).
fn stat_level_drawn(level: u32) -> u32 {
    return 8u + min(level, 6u);
}
fn stat_level_faces(level: u32) -> u32 {
    return 15u + min(level, 6u);
}

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
@group(1) @binding(7) var<uniform> freeze: FreezeParams;

// The task 2.12 debug-view flags (params.debug.y).
fn exploding() -> bool {
    return (params.debug.y & 1u) != 0u;
}
fn frozen() -> bool {
    return (params.debug.y & 2u) != 0u;
}

// Whether a level draws under the shown-level mask (task 2.12): a hidden
// level leaves a hole — the cube emits nothing and never becomes
// "visible", so switching it back on starts it in the late phase.
fn level_shown(level: u32) -> bool {
    return (params.debug.x >> min(level, 6u) & 1u) != 0u;
}

// The view the cull tests against: the frozen one while the freeze view
// is on (the freeze uniform then also carries the matching camera split),
// the live camera otherwise.
fn cull_view_proj() -> mat4x4f {
    if (frozen()) {
        return freeze.view_proj;
    }
    return u.view_proj;
}

// The one origin formula, scaled by the cube's level and identical to
// the vertex shader in terrain.wgsl and `cube_origin` in
// render/chunks.rs: origin = (pos*2^level - camera_cube) * CUBE_SIZE -
// camera_frac. The integer product stays inside i32 for any streamable
// position (the CPU mirror asserts it), the only rounding is the final
// f32 subtract, so CPU and GPU cull on the same numbers and no absolute
// world coordinate ever reaches f32. While frozen (task 2.12) the split
// comes from the freeze uniform, so the AABBs land in the frozen view's
// space where the frozen frustum expects them.
fn cube_origin(pos: vec3i, level: u32) -> vec3f {
    // pos * 2^level stays inside i32 for any streamable position.
    let cube = select(params.camera_cube.xyz, freeze.camera_cube.xyz, frozen());
    let frac = select(params.camera_frac.xyz, freeze.camera_frac.xyz, frozen());
    let scaled = pos * vec3i(i32(1u << level)) - cube;
    let int_part = vec3f(scaled) * f32(params.regions.w);
    return int_part - frac;
}

// A level-n cube is CUBE_SIZE * 2^n wide: the AABB half of the cull
// tests.
fn cube_extent(level: u32) -> f32 {
    return f32(params.regions.w) * f32(1u << level);
}

// The plane test of `Frustum::intersects_aabb` (render/chunks.rs), ported
// with the same comparisons: Gribb-Hartmann rows of the rotation-only
// view-projection (the reverse-Z far plane is at infinity, so five planes).
// Row r of a column-major mat4x4f (glam stores columns).
fn matrix_row(m: mat4x4f, r: u32) -> vec4f {
    return vec4f(m[0][r], m[1][r], m[2][r], m[3][r]);
}

fn frustum_visible(rel_min: vec3f, rel_max: vec3f) -> bool {
    let m = cull_view_proj();
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

// The direction-split rule (task 2.13 C): beyond `split.x` blocks the
// cube's whole, direction-contiguous face range draws in ONE command —
// back-face culling drops exactly the faces direction skipping dropped,
// so the pixels cannot change. The distance is to the closest point of
// the camera-relative AABB (the camera sits at the origin), computed
// with explicit multiplies and adds left to right — no dot(), which a
// backend may contract into fused multiply-adds while the Rust side
// (`aabb_distance_squared` in space.rs) would not; the CPU bound and
// this shader must verdict identically.
fn cube_far(rel_min: vec3f, rel_max: vec3f) -> bool {
    let cx = min(max(0.0, rel_min.x), rel_max.x);
    let cy = min(max(0.0, rel_min.y), rel_max.y);
    let cz = min(max(0.0, rel_min.z), rel_max.z);
    let r2 = params.split.x * params.split.x;
    return cx * cx + cy * cy + cz * cz > r2;
}

// Appends one command per non-empty, non-skipped direction into this
// phase's region for the cube's page — or, beyond the direction-split
// radius, one command over the cube's whole face range (the ranges are
// direction-contiguous, so the whole range is one span). idx is
// bounds-checked against the command capacity: the CPU's frustum bound
// (region_base) is provably large enough for this shader's
// identical-math emission, so a drop here is a bound bug — it is
// counted in region_overflow for the tests to catch instead of
// corrupting memory.
fn emit(m: CubeMeta, slot: u32, phase_base: u32, rel_min: vec3f, rel_max: vec3f) {
    // A fully masked drawn cube emits nothing (the coverage rule: all
    // eight octants are replaced by ready children).
    if ((m.start_flags.y >> 16u) & 0xFFu) == 0xFFu {
        return;
    }
    let level = level_of(m);
    // A hidden level emits nothing (task 2.12): the shown-level mask.
    if (!level_shown(level)) {
        return;
    }
    let page = u32(m.pos_page.w);
    let dirs = visible_dirs(rel_min, rel_max);
    let start = m.start_flags.x;
    if (cube_far(rel_min, rel_max)) {
        var total = 0u;
        for (var d = 0u; d < 6u; d++) {
            total = total + m.dirs[d / 2u][(d % 2u) * 2u + 1u];
        }
        if (total == 0u) {
            return;
        }
        let idx = phase_base + region_base[page] + atomicAdd(&region_counter[page], 1u);
        if (idx < params.regions.z) {
            commands[idx] = DrawCmd(total * 6u, 1u, start * 6u, slot);
        } else {
            atomicAdd(&stats[STAT_REGION_OVERFLOW], 1u);
        }
        atomicAdd(&stats[STAT_COMMANDS], 1u);
        atomicAdd(&stats[STAT_FACES], total);
        atomicAdd(&stats[stat_level_faces(level)], total);
        return;
    }
    for (var d = 0u; d < 6u; d++) {
        let off = m.dirs[d / 2u][(d % 2u) * 2u];
        let cnt = m.dirs[d / 2u][(d % 2u) * 2u + 1u];
        // While exploded (task 2.12) the direction test is skipped too:
        // a cube can slide across the camera, and the CPU bound in that
        // mode is the static per-page non-empty-direction total.
        if (cnt == 0u || (!exploding() && !dir_visible(dirs, d))) {
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
        atomicAdd(&stats[stat_level_faces(level)], cnt);
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
    let level = level_of(m);
    // A hidden level emits nothing (task 2.12), in either phase.
    if (!level_shown(level)) {
        return;
    }
    let rel_min = cube_origin(m.pos_page.xyz, level);
    let rel_max = rel_min + vec3f(cube_extent(level));
    // Draw what was visible last frame; the early phase never touches the
    // visible bit (the late phase needs last frame's value). Exploded
    // (task 2.12) skips the frustum test entirely.
    let passes = exploding() || frustum_visible(rel_min, rel_max);
    if ((m.start_flags.y & VISIBLE) != 0u && passes) {
        atomicAdd(&stats[STAT_DRAWN_EARLY], 1u);
        atomicAdd(&stats[stat_level_drawn(level)], 1u);
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
    let level = level_of(m);
    if (!level_shown(level)) {
        // Hidden levels never draw, so they are never visible either.
        cube_meta[slot].start_flags.y = m.start_flags.y & ~VISIBLE;
        return;
    }
    let rel_min = cube_origin(m.pos_page.xyz, level);
    let rel_max = rel_min + vec3f(cube_extent(level));
    var visible = true;
    // Exploded skips both tests (the moved geometry trusts no bound);
    // frozen skips the occlusion test (the pyramid holds the real view's
    // depth, not the frozen one) but keeps the frozen frustum.
    if (!exploding() && !frustum_visible(rel_min, rel_max)) {
        atomicAdd(&stats[STAT_FRUSTUM_CULLED], 1u);
        visible = false;
    } else if (!exploding() && !frozen() && occluded(rel_min, rel_max)) {
        atomicAdd(&stats[STAT_OCCLUDED], 1u);
        visible = false;
    }
    if (!visible) {
        cube_meta[slot].start_flags.y = m.start_flags.y & ~VISIBLE;
        return;
    }
    cube_meta[slot].start_flags.y = m.start_flags.y | VISIBLE;
    // Emitted early this frame iff it was visible last frame and passed
    // the same frustum test (deterministic — under explode every active
    // cube passes, so the split still partitions them), so only the rest
    // draw late.
    if ((m.start_flags.y & VISIBLE) == 0u) {
        atomicAdd(&stats[STAT_DRAWN_LATE], 1u);
        atomicAdd(&stats[stat_level_drawn(level)], 1u);
        emit(m, slot, params.regions.x, rel_min, rel_max);
    }
}
