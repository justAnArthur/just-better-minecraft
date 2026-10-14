//! The flat per-state appearance table the mesher reads instead of the
//! registry, plus the procedural builtin placeholder textures.
//!
//! [`BlockAppearance`] is built once from the [`BlockRegistry`] and the
//! builtin blocks, then shared read-only (`Arc`) with the mesh workers: the
//! hot path indexes a flat `Vec` and never calls `registry.info()` or does
//! string lookups.

use image::{Rgba, RgbaImage};
use jbm_core::block::{BlockRegistry, BlockStateId};

/// Texture array layers produced by [`builtin_textures`], in order. The
/// values are what `FaceFields::layer` carries and what the terrain shader
/// uses as the `texture_2d_array` index.
pub const LAYER_STONE: u16 = 0;
pub const LAYER_DIRT: u16 = 1;
pub const LAYER_GRASS_TOP: u16 = 2;
pub const LAYER_GRASS_SIDE: u16 = 3;
pub const LAYER_SAND: u16 = 4;
pub const LAYER_WATER: u16 = 5;
pub const LAYER_GLASS: u16 = 6;
pub const LAYER_OAK_PLANKS: u16 = 7;
pub const LAYER_OAK_LOG_SIDE: u16 = 8;
pub const LAYER_OAK_LOG_TOP: u16 = 9;
pub const LAYER_LEAVES: u16 = 10;
/// Number of builtin texture layers.
pub const LAYER_COUNT: u16 = 11;

/// Appearance of one block state. Flags come from the registry; layers
/// index the texture array per face direction, in
/// [`Direction`](super::face::Direction) order: +X, -X, +Y, -Y, +Z, -Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateAppearance {
    pub air: bool,
    pub opaque: bool,
    /// Cutout blocks render with an alpha threshold and cull faces
    /// against themselves (glass next to glass).
    pub cutout: bool,
    pub layers: [u16; 6],
}

/// Flat table of per-state appearance, indexed by `BlockStateId.0`.
#[derive(Debug, Clone)]
pub struct BlockAppearance {
    states: Vec<StateAppearance>,
}

impl BlockAppearance {
    /// Builds the table for every state in `registry`, mapping the builtin
    /// `base:` blocks to their texture layers. Unknown states get layer 0
    /// on every face. Panics on an out-of-range id (they are dense by
    /// construction).
    pub fn new(registry: &BlockRegistry) -> BlockAppearance {
        let solid_layers = |layer: u16| [layer; 6];
        let states = (0..registry.state_count())
            .map(|id| {
                let info = registry.info(BlockStateId(id));
                let (cutout, layers) = match info.name.as_str() {
                    "base:air" => (false, solid_layers(LAYER_STONE)),
                    "base:stone" => (false, solid_layers(LAYER_STONE)),
                    "base:dirt" => (false, solid_layers(LAYER_DIRT)),
                    "base:sand" => (false, solid_layers(LAYER_SAND)),
                    "base:water" => (false, solid_layers(LAYER_WATER)),
                    "base:oak_planks" => (false, solid_layers(LAYER_OAK_PLANKS)),
                    // grass: green top, dirt bottom, side strip on the walls.
                    "base:grass_block" => (
                        false,
                        [
                            LAYER_GRASS_SIDE,
                            LAYER_GRASS_SIDE,
                            LAYER_GRASS_TOP,
                            LAYER_DIRT,
                            LAYER_GRASS_SIDE,
                            LAYER_GRASS_SIDE,
                        ],
                    ),
                    "base:glass" => (true, solid_layers(LAYER_GLASS)),
                    "base:leaves" => (true, solid_layers(LAYER_LEAVES)),
                    // log: rings on the two faces along the axis, bark
                    // elsewhere, for each of the three axis states.
                    "base:oak_log" => {
                        let axis = info
                            .properties
                            .iter()
                            .find(|(name, _)| name == "axis")
                            .map(|(_, value)| value.as_str())
                            .unwrap_or("y");
                        let mut layers = [LAYER_OAK_LOG_SIDE; 6];
                        let ring_faces: &[usize] = match axis {
                            "x" => &[0, 1],
                            "z" => &[4, 5],
                            _ => &[2, 3],
                        };
                        for &face in ring_faces {
                            layers[face] = LAYER_OAK_LOG_TOP;
                        }
                        (false, layers)
                    }
                    _ => (false, solid_layers(LAYER_STONE)),
                };
                StateAppearance {
                    air: info.name == "base:air", // one string compare per state, at startup only
                    opaque: info.opaque,
                    cutout,
                    layers,
                }
            })
            .collect();
        BlockAppearance { states }
    }

    /// Appearance of one state. Panics on an out-of-range id.
    pub fn get(&self, state: BlockStateId) -> &StateAppearance {
        &self.states[state.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

/// Small deterministic hash for per-pixel texture noise. Stable across
/// runs and platforms so goldens never churn.
fn noise(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = 0x811c_9dc5_u32 ^ seed;
    for byte in x.to_le_bytes().iter().chain(&y.to_le_bytes()) {
        h ^= u32::from(*byte);
        h = h.wrapping_mul(0x0100_0193);
    }
    // Final avalanche so nearby (x, y) do not correlate.
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_f491);
    h ^= h >> 13;
    h
}

/// Adjusts a base color by noise in `-spread..=spread` per channel.
fn shade(base: [u8; 3], spread: i32, n: u32) -> Rgba<u8> {
    let d = (n % (2 * spread as u32 + 1)) as i32 - spread;
    Rgba([
        (base[0] as i32 + d).clamp(0, 255) as u8,
        (base[1] as i32 + d).clamp(0, 255) as u8,
        (base[2] as i32 + d).clamp(0, 255) as u8,
        255,
    ])
}

/// The builtin placeholder textures: 16x16 procedural layers, one per
/// [`LAYER_*`](self) constant, returned with their names in the same
/// order. Colors are chosen so each block is recognizable in screenshots.
/// Glass and leaves use alpha exactly 0 or 255 (cutout).
pub fn builtin_textures() -> (Vec<RgbaImage>, Vec<&'static str>) {
    const NAMES: [&str; LAYER_COUNT as usize] = [
        "stone",
        "dirt",
        "grass_top",
        "grass_side",
        "sand",
        "water",
        "glass",
        "oak_planks",
        "oak_log_side",
        "oak_log_top",
        "leaves",
    ];
    let layers = NAMES
        .iter()
        .enumerate()
        .map(|(layer, &name)| make_layer(name, layer as u32))
        .collect();
    (layers, NAMES.to_vec())
}

fn make_layer(name: &str, layer: u32) -> RgbaImage {
    let mut img = RgbaImage::new(16, 16);
    for y in 0..16u32 {
        for x in 0..16u32 {
            let n = noise(x, y, layer);
            let pixel = match name {
                "stone" => shade([128, 128, 128], 14, n),
                "dirt" => shade([121, 85, 58], 12, n),
                "grass_top" => shade([94, 157, 52], 14, n),
                "sand" => shade([219, 207, 163], 8, n),
                // Opaque blue for now; translucency is a later task.
                "water" => shade([47, 93, 197], 8, n),
                "oak_log_side" => {
                    // Vertical bark stripes.
                    let stripe = (x + (n & 1)).is_multiple_of(4);
                    let base = if stripe { [80, 62, 37] } else { [107, 84, 52] };
                    shade(base, 6, n)
                }
                "oak_log_top" => {
                    // Concentric rings around the center.
                    let dx = x as i32 - 8;
                    let dy = y as i32 - 8;
                    let r = ((dx * dx + dy * dy) as f32).sqrt();
                    let ring = (r / 2.5).floor() as i32 % 2 == 0;
                    let base = if ring { [151, 121, 76] } else { [120, 94, 58] };
                    shade(base, 5, n)
                }
                "oak_planks" => {
                    // Horizontal plank seams every 4 px with staggered
                    // vertical joints.
                    let seam = y % 4 == 3 || (x + 8 * (y / 4)) % 8 == 7;
                    let base = if seam { [125, 96, 58] } else { [162, 130, 78] };
                    shade(base, 7, n)
                }
                "glass" => {
                    // Transparent pane: opaque 1 px border plus two short
                    // diagonal streaks; everything else alpha 0.
                    let border = x == 0 || x == 15 || y == 0 || y == 15;
                    let streak =
                        (x == y && (3..12).contains(&x)) || (x + y == 21 && (11..16).contains(&x));
                    if border || streak {
                        Rgba([222, 239, 245, 255])
                    } else {
                        Rgba([190, 219, 231, 0])
                    }
                }
                "leaves" => {
                    // Dense foliage with scattered holes (alpha 0) and
                    // darker leaf clusters.
                    let hole = n.is_multiple_of(7);
                    if hole {
                        Rgba([0, 0, 0, 0])
                    } else if n.is_multiple_of(5) {
                        shade([41, 84, 24], 6, n)
                    } else {
                        shade([60, 118, 35], 12, n)
                    }
                }
                _ => shade([255, 0, 255], 0, n),
            };
            img.put_pixel(x, y, pixel);
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;
    use jbm_core::block::builtin;

    #[test]
    fn covers_every_state_with_registry_flags() {
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        let table = BlockAppearance::new(&registry);
        assert_eq!(table.len(), registry.state_count() as usize);
        assert!(table.get(BlockStateId::AIR).air);
        assert!(table.get(blocks.stone).opaque);
        assert!(!table.get(blocks.stone).cutout);
        let water = table.get(blocks.water);
        assert!(!water.opaque && !water.cutout && !water.air);
        for &state in &[blocks.glass, blocks.leaves] {
            let info = table.get(state);
            assert!(info.cutout, "should be cutout");
            assert!(!info.opaque, "cutout blocks are not opaque");
        }
    }

    #[test]
    fn grass_block_faces_map_top_side_bottom() {
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        let table = BlockAppearance::new(&registry);
        let grass = table.get(blocks.grass_block);
        // +X, -X, +Y, -Y, +Z, -Z.
        assert_eq!(
            grass.layers,
            [
                LAYER_GRASS_SIDE,
                LAYER_GRASS_SIDE,
                LAYER_GRASS_TOP,
                LAYER_DIRT,
                LAYER_GRASS_SIDE,
                LAYER_GRASS_SIDE,
            ]
        );
    }

    #[test]
    fn oak_log_rings_follow_the_axis_property() {
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        let table = BlockAppearance::new(&registry);
        for (axis, ring_faces) in [("y", [2usize, 3]), ("x", [0, 1]), ("z", [4, 5])] {
            let state = registry
                .state("base:oak_log", &[("axis", axis)])
                .expect("axis state exists");
            let layers = table.get(state).layers;
            for (face, &layer) in layers.iter().enumerate() {
                let expected = if ring_faces.contains(&face) {
                    LAYER_OAK_LOG_TOP
                } else {
                    LAYER_OAK_LOG_SIDE
                };
                assert_eq!(layer, expected, "axis {axis}, face {face}");
            }
        }
        // The default state is axis y.
        assert_eq!(
            table.get(blocks.oak_log).layers[2],
            LAYER_OAK_LOG_TOP,
            "default oak_log is axis y: +Y shows rings"
        );
    }

    #[test]
    fn builtin_textures_layers_match_names() {
        let (layers, names) = builtin_textures();
        assert_eq!(layers.len(), LAYER_COUNT as usize);
        assert_eq!(names.len(), LAYER_COUNT as usize);
        assert_eq!(names[0], "stone");
        assert_eq!(names[LAYER_LEAVES as usize], "leaves");
        for layer in &layers {
            assert_eq!(layer.dimensions(), (16, 16));
        }
        // Deterministic: two calls agree pixel for pixel.
        let (again, _) = builtin_textures();
        assert_eq!(layers, again);
    }

    #[test]
    fn cutout_layers_use_binary_alpha() {
        let (layers, names) = builtin_textures();
        for &cutout in &["glass", "leaves"] {
            let layer = &layers[names.iter().position(|&n| n == cutout).unwrap()];
            let mut zeros = 0;
            let mut ones = 0;
            for pixel in layer.pixels() {
                assert!(
                    pixel.0[3] == 0 || pixel.0[3] == 255,
                    "alpha must be 0 or 255"
                );
                if pixel.0[3] == 0 {
                    zeros += 1;
                } else {
                    ones += 1;
                }
            }
            assert!(zeros > 0 && ones > 0, "{cutout} needs both alphas");
        }
    }

    #[test]
    fn layers_are_visually_distinct() {
        let (layers, _) = builtin_textures();
        for (i, a) in layers.iter().enumerate() {
            for (j, b) in layers.iter().enumerate().skip(i + 1) {
                assert_ne!(a, b, "layers {i} and {j} are identical");
            }
        }
    }
}
