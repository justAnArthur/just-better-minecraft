// The Hi-Z mip-0 fill for a multisampled depth target (task 2.15's MSAA):
// the hiz.wgsl fill_mip0 rule, except each texel takes the **min over its
// samples**. Reverse-Z, so the min is the farthest sample: at a
// silhouette pixel half covered by a near occluder the pyramid keeps the
// far background, and the occlusion test stays conservative (it may keep
// an occluded cube, never drop a visible one). A separate module because
// the source binding's type (texture_depth_multisampled_2d) differs from
// the single-sample fill's; the Sizes uniform and the destination match
// hiz.wgsl.

struct Sizes {
    src: vec2u,
    dst: vec2u,
};

@group(0) @binding(0) var src_ms: texture_depth_multisampled_2d;
@group(0) @binding(1) var dst_tex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> sizes: Sizes;

@compute @workgroup_size(8, 8)
fn fill_mip0_ms(@builtin(global_invocation_id) gid: vec3u) {
    let p = vec2u(gid.xy);
    if (p.x >= sizes.dst.x || p.y >= sizes.dst.y) {
        return;
    }
    let count = textureNumSamples(src_ms);
    var depth = 2.0; // reverse-Z: every depth is below 2.0
    for (var s = 0u; s < count; s++) {
        depth = min(depth, textureLoad(src_ms, vec2i(p), i32(s)));
    }
    textureStore(dst_tex, vec2i(p), vec4f(depth, 0.0, 0.0, 0.0));
}
