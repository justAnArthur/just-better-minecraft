// The hierarchical-Z pyramid reduction (task 2.5). The depth target is
// reverse-Z (near = 1.0, far = 0.0), so the *smaller* depth is the
// *farther* surface, and occlusion culling needs the farthest one: every
// reduction is a 2x2 **min**.
//
// fill_mip0: one thread per texel copies the depth target into mip 0 of
// the R32Float pyramid (a plain compute copy; the depth texture is only
// sampleable, not storable).
//
// reduce_mip: mip k is bound as sampled (binding 0), mip k+1 as a
// write-only R32Float storage texture (binding 1). Binding one mip as
// sampled and another as storage in one dispatch is legal: usage is
// tracked per subresource and different mips never overlap.
//
// The reduction rule (mirrored by `reduce_step` in render/hiz.rs — keep
// the two in sync): dst sizes are floor-halved (wgpu's own mip sizing),
//
//   dst(x, y) = min over src in [2x ..= xe] x [2y ..= ye],
//   xe = 2x + 1 for every dst texel but the last of an axis, which takes
//   everything remaining (xe = src_w - 1), and likewise ye.
//
// Odd sizes (the classic Hi-Z bug): the extra last row/column of src
// folds into the last dst texel instead of being skipped:
//
//   src  (5 wide):  0  1 | 2  3  4
//   dst  (2 wide): [0  1] [2  3  4]      <- 4 folds into dst 1
//
// So every src texel contributes to exactly one dst texel and every mip
// texel is the min over exactly the mip-0 texels its footprint covers
// (unit-tested in render/hiz.rs).

struct Sizes {
    src: vec2u,
    dst: vec2u,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> sizes: Sizes;

@compute @workgroup_size(8, 8)
fn fill_mip0(@builtin(global_invocation_id) gid: vec3u) {
    let p = vec2u(gid.xy);
    if (p.x >= sizes.dst.x || p.y >= sizes.dst.y) {
        return;
    }
    let depth = textureLoad(src_tex, vec2i(p), 0).r;
    textureStore(dst_tex, vec2i(p), vec4f(depth, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn reduce_mip(@builtin(global_invocation_id) gid: vec3u) {
    let d = vec2u(gid.xy);
    if (d.x >= sizes.dst.x || d.y >= sizes.dst.y) {
        return;
    }
    // 2x2 min; the last dst texel of each axis folds in the odd remainder
    // (see above), so the sampled range spans up to 3 texels per axis and
    // no src texel is skipped.
    let sx0 = d.x * 2u;
    let sx1 = select(d.x * 2u + 1u, sizes.src.x - 1u, d.x + 1u == sizes.dst.x);
    let sy0 = d.y * 2u;
    let sy1 = select(d.y * 2u + 1u, sizes.src.y - 1u, d.y + 1u == sizes.dst.y);
    var m = 2.0; // reverse-Z: everything is below 2.0
    for (var j = 0u; j < 3u; j++) {
        let sy = min(sy0 + j, sy1);
        for (var i = 0u; i < 3u; i++) {
            let sx = min(sx0 + i, sx1);
            m = min(m, textureLoad(src_tex, vec2i(i32(sx), i32(sy)), 0).r);
        }
    }
    textureStore(dst_tex, vec2i(d), vec4f(m, 0.0, 0.0, 0.0));
}
