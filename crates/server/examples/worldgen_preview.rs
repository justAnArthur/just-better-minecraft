//! Writes a top-down preview PNG of the overworld: an N-kilometre square
//! coloured by surface height and water, with hillshading so ridges and
//! valleys read at a glance (task 2.8; the orchestrator inspects these to
//! tune the terrain).
//!
//! ```text
//! cargo run -p jbm-server --example worldgen_preview [-- <seed> [<km>]]
//! ```
//!
//! Defaults: the default server seed, 8 km, written to
//! `target/worldgen_preview/overworld_<seed>_<km>km.png`. One pixel per
//! `step` blocks, `step` chosen so the image stays near 2000 px per side.

use std::path::PathBuf;

use image::{Rgb, RgbImage};
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_server::DEFAULT_SEED;
use jbm_server::terrain::{Generator, MAX_HEIGHT, MIN_HEIGHT, SEA_LEVEL};
use jbm_server::worldgen::Preset;

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|a| u64::from_str_radix(a.trim_start_matches("0x"), 16).ok())
        .unwrap_or(DEFAULT_SEED);
    let km: i64 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(8);
    let extent = 1000 * km;
    let step = (extent / 2000).max(1);
    let side = (extent / step) as u32;

    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let terrain = Generator::with_preset(seed, blocks, Preset::Overworld);

    let started = std::time::Instant::now();
    // One height per pixel; one row of extra samples on every side feeds
    // the hillshade's gradient.
    let mut heights = vec![0i64; ((side + 2) * (side + 2)) as usize];
    for z in 0..(side + 2) as i64 {
        for x in 0..(side + 2) as i64 {
            let wx = (x - 1) * step - extent / 2;
            let wz = (z - 1) * step - extent / 2;
            heights[(z * (side as i64 + 2) + x) as usize] = terrain.surface_height(wx, wz);
        }
    }
    println!(
        "sampled {} columns in {:?}",
        (side + 2) * (side + 2),
        started.elapsed()
    );

    let mut image = RgbImage::new(side, side);
    for z in 0..side as i64 {
        for x in 0..side as i64 {
            let at = |x: i64, z: i64| heights[((z + 1) * (side as i64 + 2) + x + 1) as usize];
            let h = at(x, z);
            let rgb = if h < SEA_LEVEL {
                // Water: deeper is darker blue; the shade band stays visible.
                let depth = (SEA_LEVEL - h).min(40) as f32 / 40.0;
                Rgb([
                    (40.0 - 20.0 * depth) as u8,
                    (90.0 - 50.0 * depth) as u8,
                    (170.0 - 70.0 * depth) as u8,
                ])
            } else {
                // Land hypsometric ramp: beach sand, lowland green, hill
                // olive, bare rock gray, a light cap on the highest peaks.
                let t = (h - SEA_LEVEL) as f32 / (MAX_HEIGHT - SEA_LEVEL) as f32;
                let t = t.clamp(0.0, 1.0);
                let ramp = |stops: &[(f32, [f32; 3])]| -> [f32; 3] {
                    let (mut lo, mut hi) = (stops[0], stops[stops.len() - 1]);
                    for w in stops.windows(2) {
                        if t >= w[0].0 {
                            (lo, hi) = (w[0], w[1]);
                        }
                    }
                    let f = if hi.0 > lo.0 {
                        ((t - lo.0) / (hi.0 - lo.0)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    [
                        lo.1[0] + f * (hi.1[0] - lo.1[0]),
                        lo.1[1] + f * (hi.1[1] - lo.1[1]),
                        lo.1[2] + f * (hi.1[2] - lo.1[2]),
                    ]
                };
                let c = ramp(&[
                    (0.00, [194.0, 178.0, 128.0]),
                    (0.03, [96.0, 152.0, 64.0]),
                    (0.25, [72.0, 120.0, 56.0]),
                    (0.45, [124.0, 116.0, 84.0]),
                    (0.60, [136.0, 132.0, 128.0]),
                    (0.80, [176.0, 174.0, 172.0]),
                    (1.00, [236.0, 236.0, 238.0]),
                ]);
                Rgb([c[0] as u8, c[1] as u8, c[2] as u8])
            };
            // Hillshade: the surface gradient against a north-west light.
            let dx = (at(x + 1, z) - at(x - 1, z)) as f32 / step as f32;
            let dz = (at(x, z + 1) - at(x, z - 1)) as f32 / step as f32;
            let shade = 1.0 + (dx + dz).clamp(-2.0, 2.0) * 0.22;
            let shade = shade.clamp(0.55, 1.45);
            let [r, g, b] = rgb.0;
            image.put_pixel(
                x as u32,
                z as u32,
                Rgb([
                    (r as f32 * shade).min(255.0) as u8,
                    (g as f32 * shade).min(255.0) as u8,
                    (b as f32 * shade).min(255.0) as u8,
                ]),
            );
        }
    }

    let out =
        PathBuf::from("target/worldgen_preview").join(format!("overworld_{seed:#x}_{km}km.png"));
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create the preview directory");
    }
    image.save(&out).expect("write the preview PNG");
    println!(
        "wrote {} ({}×{}, height range [{}, {}])",
        out.display(),
        side,
        side,
        MIN_HEIGHT,
        MAX_HEIGHT
    );
}
