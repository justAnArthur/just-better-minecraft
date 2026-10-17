//! The GPU-side layout contract of the terrain pass: the `CubeMeta` slot
//! record, the terrain-only per-frame uniform, the cull-shader params, the
//! meta flag bits and the far-look constants. Every struct here is
//! `#[repr(C)]` and mirrored as a WGSL struct (shaders/terrain.wgsl,
//! shaders/cull.wgsl — keep all copies in sync); the tests at the bottom
//! pin each layout, so Rust and the shaders cannot drift apart silently.

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use jbm_core::coords::CUBE_SIZE;

use crate::lod::CubeKey;

/// Meta flag: the slot holds a resident cube that the coverage rule draws
/// (a freed slot or an undrawn cube reads inactive and the cull thread
/// exits early; the coverage rule clears it for cubes a finer level
/// replaces).
pub const META_ACTIVE: u32 = 1;
/// Meta flag: the cube was visible last frame (two-phase culling state,
/// written by the late cull, reset on upload so new and re-meshed cubes
/// are picked up by the late pass of the same frame).
pub const META_VISIBLE: u32 = 2;
/// Meta bits 8..12 of `start_flags.y`: the cube's LOD level (0 = LOD0),
/// scaling its origin and cell offsets by 2^level.
pub const META_LEVEL_SHIFT: u32 = 8;
/// Meta bits 16..24 of `start_flags.y`: the octant mask — bit *o* set
/// when a ready child replaces octant *o*, so the vertex shader collapses
/// faces whose owning cell sits in a masked octant.
pub const META_MASK_SHIFT: u32 = 16;

/// Packs the level and octant mask into the flag word. `drawn && mask ==
/// 0xFF` — a cube whose every octant a ready child replaces — reads
/// inactive, so the cull pass skips it like any undrawn cube (its mesh,
/// if any, is fully collapsed by the vertex shader anyway).
pub(super) fn meta_flags(drawn: bool, level: u8, mask: u8) -> u32 {
    let mut flags = u32::from(level) << META_LEVEL_SHIFT | u32::from(mask) << META_MASK_SHIFT;
    if drawn && mask != 0xFF {
        flags |= META_ACTIVE;
    }
    flags
}

/// Unpacks `(drawn-active, level, octant mask)` from the flag word.
pub fn unpack_meta_flags(flags: u32) -> (bool, u8, u8) {
    (
        flags & META_ACTIVE != 0,
        ((flags >> META_LEVEL_SHIFT) & 0xF) as u8,
        (flags >> META_MASK_SHIFT) as u8,
    )
}

/// One cube's uploaded mesh, mirrored on the GPU as `CubeMeta`
/// (shaders/terrain.wgsl, shaders/cull.wgsl — keep all three in sync).
///
/// Layout: every field is a 16-byte group, so Rust `#[repr(C)]` and the
/// WGSL struct agree at 16-byte alignment with no hidden padding; the
/// layout test below pins it. 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CubeMeta {
    /// x, y, z = the cube coordinates; w = the page index.
    pub pos_page: [i32; 4],
    /// x = the face range start in the page; y = flags (META_ACTIVE,
    /// META_VISIBLE, level << 8, octant mask << 16); z = skirt faces, w
    /// spare.
    pub start_flags: [u32; 4],
    /// The six per-direction (offset, count) pairs from
    /// `MeshOutput::ranges`, offsets relative to the cube's face range:
    /// direction d's pair is `dirs[d / 2][(d % 2) * 2 ..][..2]`.
    pub dirs: [[u32; 4]; 3],
}

impl CubeMeta {
    pub(super) fn zeroed() -> CubeMeta {
        CubeMeta {
            pos_page: [0; 4],
            start_flags: [0; 4],
            dirs: [[0; 4]; 3],
        }
    }
}

/// The terrain-only per-frame uniform (group 2, binding 1). The two values
/// behind the shared origin formula; sky, outline and overlay share only
/// `FrameUniforms`, so they are untouched by it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TerrainUniforms {
    /// xyz = `floor(camera / CUBE_SIZE)` as vec3<i32>.
    pub camera_cube: [i32; 4],
    /// xyz = `camera − camera_cube · CUBE_SIZE` (f64 on the CPU, rounded
    /// to f32); w = CUBE_SIZE, so the shaders are cube-size agnostic.
    pub camera_frac: [f32; 4],
    /// The far look. x = the LOD0 ring radius in blocks (the
    /// texture→flat fade window is a fraction of it); y = the furthest
    /// ring radius in blocks (the fog window is a fraction of it);
    /// z = fog on (1.0) / off (0.0 — `RenderSettings` / `JBM_NO_FOG`);
    /// w spare.
    pub far: [f32; 4],
    /// The level tint. x = tint on (1.0) / off (0.0 — F4 /
    /// `JBM_LOD_TINT=1`); the fragment shader multiplies each fragment by
    /// [`LOD_TINT_COLORS`] at its cube's level. y, z, w spare.
    pub tint: [f32; 4],
    /// The task 2.12 debug views. x = wireframe on (1.0) / off; y =
    /// wireframe-only; z = the explode gap in blocks (0 = off); the
    /// vertex shader shifts every cube by `z × its cube-grid offset` and
    /// the fragment shader draws the quad edges. w spare.
    pub debug: [f32; 4],
}

/// The level tint colors, linear RGB, indexed by LOD level; level 0 is
/// white (untinted). One fixed color per level so a screenshot shows at a
/// glance which level draws where: L1 (2×2×2 cells) red, L2 (4×4) yellow,
/// L3 (8×8) green, L4 (16×16) cyan, L5 (32×32) blue, L6 (64×64) purple.
/// Never pure magenta at any level: that is the hole sentinel the tests
/// look for. Levels above the table clamp to the last entry.
pub const LOD_TINT_COLORS: [[f32; 3]; 7] = [
    [1.0, 1.0, 1.0],   // LOD0: untinted
    [1.0, 0.0, 0.0],   // L1 2x2 red
    [1.0, 1.0, 0.0],   // L2 4x4 yellow
    [0.0, 1.0, 0.0],   // L3 8x8 green
    [0.0, 1.0, 1.0],   // L4 16x16 cyan
    [0.0, 0.0, 1.0],   // L5 32x32 blue
    [0.55, 0.15, 1.0], // L6 64x64 purple (magenta-free)
];

/// The tint color of `level` (clamped at the last entry).
pub fn lod_tint_color(level: u8) -> [f32; 3] {
    LOD_TINT_COLORS[(level as usize).min(LOD_TINT_COLORS.len() - 1)]
}

/// Fractions of the ring radii the far look keys on, mirrored as WGSL
/// consts in `shaders/terrain.wgsl` — keep the two in sync. The fade
/// covers 40%..85% of the LOD0 ring radius (flat on both sides of the
/// LOD0/LOD1 border); the fog starts at 35% of the furthest ring radius
/// and saturates at 98%, just inside the outermost ring.
pub const FADE_START_FRACTION: f32 = 0.40;
pub const FADE_END_FRACTION: f32 = 0.85;
pub const FOG_START_FRACTION: f32 = 0.35;
pub const FOG_END_FRACTION: f32 = 0.98;

/// AO softening per LOD level, mirrored by `ao_soften` in
/// `shaders/terrain.wgsl` — keep the two in sync. Level 0 keeps the exact
/// AO factor (the LOD0 goldens stay bit-stable); coarser levels pull the
/// factor toward fully open, because AO contrast scales with cell size and
/// reads blotchy far away. Levels beyond the table clamp to the last entry.
pub const AO_SOFTEN: [f32; 4] = [1.0, 0.6, 0.45, 0.3];

/// Default view distances for the far look, matching the `ServerConfig`
/// defaults (`view_radius` 8, `lod_levels` 6, `lod_view_radius` 8): the
/// LOD0 ring radius and the furthest (level-6) ring radius, in blocks.
pub const DEFAULT_LOD0_RING_RADIUS: f32 = 8.0 * CUBE_SIZE as f32;
pub const DEFAULT_FURTHEST_RING_RADIUS: f32 = 8.0 * CUBE_SIZE as f32 * 64.0;

/// The `far` words of [`TerrainUniforms`] from the radii and fog flag.
pub(super) fn far_uniform_words(far_radii: [f32; 2], fog: bool) -> [f32; 4] {
    [far_radii[0], far_radii[1], if fog { 1.0 } else { 0.0 }, 0.0]
}

/// The `tint` words of [`TerrainUniforms`] from the tint flag.
pub(super) fn tint_uniform_words(tint: bool) -> [f32; 4] {
    [if tint { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0]
}

/// The cull shader's per-frame uniform (group 1, binding 0 of cull.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CullParams {
    pub camera_cube: [i32; 4],
    pub camera_frac: [f32; 4],
    /// x, y = depth target size; z = the pyramid's max mip level;
    /// w = meta slots in use (the dispatch bound).
    pub screen: [u32; 4],
    /// x = the late half's base, in command elements; y = page count;
    /// z = command capacity, in elements; w = CUBE_SIZE.
    pub regions: [u32; 4],
    /// The task 2.12 debug views, as seen by the cull shader. x = the
    /// shown-level mask (bit L = level L draws; a hidden level emits
    /// nothing); y = flags — bit 0 explode (skip the frustum, occlusion
    /// and direction tests: every active cube emits every non-empty
    /// direction, matching the CPU's `page_dir_counts` bound), bit 1
    /// freeze (cull against the frozen view in the `freeze` uniform —
    /// binding 7 — instead of the live camera). z, w spare.
    pub debug: [u32; 4],
}

/// The freeze-cull view captured when the debug view switches on (group 1,
/// binding 7 of cull.wgsl): the view-projection and camera split the cull
/// keeps using while the real camera moves. 96 bytes, no tail padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FreezeParams {
    /// The frozen rotation-only view-projection (camera-relative).
    pub view_proj: [[f32; 4]; 4],
    /// The frozen `camera_cube` (xyz; w spare).
    pub camera_cube: [i32; 4],
    /// The frozen `camera_frac` (xyz; w = CUBE_SIZE, as in
    /// [`TerrainUniforms`]).
    pub camera_frac: [f32; 4],
}

/// The debug words of [`TerrainUniforms`] from the settings.
pub(super) fn debug_uniform_words(settings: &crate::render::settings::RenderSettings) -> [f32; 4] {
    [
        if settings.wireframe { 1.0 } else { 0.0 },
        if settings.wireframe_only { 1.0 } else { 0.0 },
        settings.explode as f32,
        0.0,
    ]
}

/// The debug words of [`CullParams`] from the settings.
pub(super) fn cull_debug_words(settings: &crate::render::settings::RenderSettings) -> [u32; 4] {
    [
        u32::from(settings.level_mask),
        u32::from(settings.explode > 0) | u32::from(settings.freeze_cull) << 1,
        0,
        0,
    ]
}

/// The meta image of one mesh (`key` is kept in the map key). The flags
/// carry the cube's level and coverage state (`drawn`, `mask`); `visible`
/// starts cleared so the late cull picks new geometry up the same frame.
pub(super) fn meta_for(
    key: CubeKey,
    page: usize,
    start: u32,
    ranges: &[Range<u32>; 6],
    skirts: u32,
    drawn: bool,
    mask: u8,
) -> CubeMeta {
    let mut dirs = [[0u32; 4]; 3];
    for (d, range) in ranges.iter().enumerate() {
        dirs[d / 2][(d % 2) * 2] = range.start;
        dirs[d / 2][(d % 2) * 2 + 1] = range.len() as u32;
    }
    CubeMeta {
        pos_page: [key.pos.x, key.pos.y, key.pos.z, page as i32],
        start_flags: [start, meta_flags(drawn, key.level, mask), skirts, 0],
        dirs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jbm_core::coords::CubePos;

    fn key(x: i32, y: i32, z: i32, level: u8) -> CubeKey {
        CubeKey {
            level,
            pos: CubePos { x, y, z },
        }
    }

    #[test]
    fn cube_meta_layout_matches_wgsl() {
        // The WGSL mirror (terrain.wgsl / cull.wgsl) is vec4i + vec4u +
        // array<vec4u, 3>: 80 bytes, members at 0 / 16 / 32.
        assert_eq!(std::mem::size_of::<CubeMeta>(), 80);
        assert_eq!(std::mem::offset_of!(CubeMeta, pos_page), 0);
        assert_eq!(std::mem::offset_of!(CubeMeta, start_flags), 16);
        assert_eq!(std::mem::offset_of!(CubeMeta, dirs), 32);
        assert!(
            std::mem::size_of::<CubeMeta>().is_multiple_of(16),
            "no tail padding: the storage element stride is the struct size"
        );
    }

    #[test]
    fn terrain_uniforms_layout_matches_wgsl() {
        // The WGSL mirror (terrain.wgsl) is vec4i + 4 × vec4f = 80 bytes,
        // every field 16-byte aligned.
        assert_eq!(std::mem::size_of::<TerrainUniforms>(), 80);
        assert_eq!(std::mem::offset_of!(TerrainUniforms, camera_cube), 0);
        assert_eq!(std::mem::offset_of!(TerrainUniforms, camera_frac), 16);
        assert_eq!(std::mem::offset_of!(TerrainUniforms, far), 32);
        assert_eq!(std::mem::offset_of!(TerrainUniforms, tint), 48);
        assert_eq!(std::mem::offset_of!(TerrainUniforms, debug), 64);
        assert!(
            std::mem::size_of::<TerrainUniforms>().is_multiple_of(16),
            "no tail padding"
        );
    }

    #[test]
    fn cull_params_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<CullParams>(), 80);
        assert_eq!(std::mem::offset_of!(CullParams, camera_cube), 0);
        assert_eq!(std::mem::offset_of!(CullParams, camera_frac), 16);
        assert_eq!(std::mem::offset_of!(CullParams, screen), 32);
        assert_eq!(std::mem::offset_of!(CullParams, regions), 48);
        assert_eq!(std::mem::offset_of!(CullParams, debug), 64);
    }

    #[test]
    fn freeze_params_layout_matches_wgsl() {
        // The WGSL mirror (cull.wgsl binding 7) is mat4x4f + vec4i +
        // vec4f + vec4u = 96 bytes.
        assert_eq!(std::mem::size_of::<FreezeParams>(), 96);
        assert_eq!(std::mem::offset_of!(FreezeParams, view_proj), 0);
        assert_eq!(std::mem::offset_of!(FreezeParams, camera_cube), 64);
        assert_eq!(std::mem::offset_of!(FreezeParams, camera_frac), 80);
    }

    /// The debug-view words follow the settings exactly: the terrain
    /// uniform carries the wireframe flags and explode gap, the cull
    /// params the level mask and the explode/freeze flag bits.
    #[test]
    fn debug_words_follow_the_settings() {
        use crate::render::settings::RenderSettings;
        let mut settings = RenderSettings::default();
        assert_eq!(debug_uniform_words(&settings), [0.0; 4]);
        assert_eq!(cull_debug_words(&settings), [0x7F, 0, 0, 0]);
        settings.wireframe = true;
        settings.explode = 5;
        settings.level_mask = 0b101;
        settings.freeze_cull = true;
        assert_eq!(debug_uniform_words(&settings), [1.0, 0.0, 5.0, 0.0]);
        assert_eq!(cull_debug_words(&settings), [0b101, 0b11, 0, 0]);
        settings.wireframe_only = true;
        assert_eq!(debug_uniform_words(&settings), [1.0, 1.0, 5.0, 0.0]);
    }

    /// The level tint colors: LOD0 white, every level's zero-channel
    /// pattern unique (so a pixel identifies its level), no two adjacent
    /// levels sharing a channel that stays, and never pure magenta (the
    /// hole sentinel) anywhere.
    #[test]
    fn lod_tint_colors_identify_levels_and_avoid_magenta() {
        assert_eq!(lod_tint_color(0), [1.0, 1.0, 1.0], "LOD0 untinted");
        for level in 1..=6u8 {
            let color = lod_tint_color(level);
            assert!(
                color != [1.0, 0.0, 1.0],
                "level {level} must never tint pure magenta (the hole sentinel)"
            );
        }
        // The zero-channel signature of each color is unique: red
        // (r only), yellow (r,g), green (g only), cyan (g,b), blue (b
        // only), purple (r reduced, b full).
        let signatures: Vec<[bool; 3]> = (1..=6u8)
            .map(|level| lod_tint_color(level).map(|c| c > 0.25))
            .collect();
        for i in 0..signatures.len() {
            for j in i + 1..signatures.len() {
                assert_ne!(
                    signatures[i],
                    signatures[j],
                    "levels {} and {} are indistinguishable",
                    i + 1,
                    j + 1
                );
            }
        }
        // Above the table clamps to purple, still magenta-free.
        assert_eq!(lod_tint_color(7), lod_tint_color(6));
        assert_eq!(tint_uniform_words(true), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(tint_uniform_words(false), [0.0; 4]);
    }

    /// The far-look uniform words follow the radii and fog flag exactly.
    #[test]
    fn far_uniform_words_follow_the_fog_flag() {
        assert_eq!(
            far_uniform_words([256.0, 4096.0], true),
            [256.0, 4096.0, 1.0, 0.0]
        );
        assert_eq!(
            far_uniform_words([256.0, 4096.0], false),
            [256.0, 4096.0, 0.0, 0.0]
        );
        assert_eq!(
            far_uniform_words([64.0, 1280.0], false),
            [64.0, 1280.0, 0.0, 0.0]
        );
        // The defaults follow the ServerConfig defaults' shape at any cube
        // size: 8 cubes for the LOD0 ring, and the level-6 disk (64x
        // coarser) for the furthest.
        assert_eq!(DEFAULT_LOD0_RING_RADIUS, 8.0 * CUBE_SIZE as f32);
        assert_eq!(
            DEFAULT_FURTHEST_RING_RADIUS,
            DEFAULT_LOD0_RING_RADIUS * 64.0
        );
    }

    #[test]
    fn meta_for_packs_direction_pairs_and_level_flags() {
        let mut ranges: [Range<u32>; 6] =
            std::array::from_fn(|d| (d as u32 * 3)..(d as u32 * 3 + 3));
        ranges[1] = 0..0; // -X empty
        let meta = meta_for(key(-4, 5, -6, 3), 2, 100, &ranges, 7, true, 0x05);
        assert_eq!(meta.pos_page, [-4, 5, -6, 2]);
        assert_eq!(
            meta.start_flags,
            [
                100,
                META_ACTIVE | (3 << META_LEVEL_SHIFT) | (0x05 << META_MASK_SHIFT),
                7,
                0
            ]
        );
        assert_eq!(
            unpack_meta_flags(meta.start_flags[1]),
            (true, 3, 0x05),
            "flags round-trip"
        );
        // An undrawn cube reads inactive to the cull shaders.
        let undrawn = meta_for(key(0, 0, 0, 1), 0, 0, &ranges, 0, false, 0xFF);
        assert_eq!(unpack_meta_flags(undrawn.start_flags[1]), (false, 1, 0xFF));
        // Direction d's (offset, count) pair is dirs[d / 2][(d % 2) * 2..].
        for (d, range) in ranges.iter().enumerate() {
            let (offset, count) = (
                meta.dirs[d / 2][(d % 2) * 2],
                meta.dirs[d / 2][(d % 2) * 2 + 1],
            );
            assert_eq!(offset, range.start);
            assert_eq!(count, range.len() as u32);
        }
    }
}
