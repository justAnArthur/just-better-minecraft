//! 4-connected-component hole detection for sentinel-sky frame checks.
//!
//! A test renders with a magenta ground sentinel so terrain holes show
//! the sky through the terrain. The rule (shared by the rings-scene and
//! real-server fly tests): sentinel pixels in the checked region must be
//! *pinholes* — 4-connected components of at most [`PINHOLE_MAX_PIXELS`]
//! pixels, at most [`PINHOLE_MAX_TOTAL`] such pixels per frame. Anything
//! bigger is a real hole (a missing skirt or a ring-border crack is a
//! long strip) and fails. The pinholes come from the LOD0 greedy mesher's
//! T-junctions, whose edge interpolation differs by ulps (see the
//! sub-pixel expansion comment in `crates/client/src/render/shaders/
//! terrain.wgsl` — LOD0 skips the expansion to keep the pre-LOD goldens
//! bit-stable); eliminating them in the mesher is a follow-up. The
//! detector is plain code over a pixel buffer and is unit-tested without
//! a GPU.

use image::{Rgba, RgbaImage};

/// A 4-connected component of sentinel pixels in a checked region.
pub struct HoleComponent {
    pixels: Vec<(u32, u32)>,
}

impl HoleComponent {
    pub fn size(&self) -> usize {
        self.pixels.len()
    }

    /// The min and max corners of the bounding box, in image pixels.
    pub fn bounds(&self) -> ((u32, u32), (u32, u32)) {
        let mut min = (u32::MAX, u32::MAX);
        let mut max = (0, 0);
        for &(x, y) in &self.pixels {
            min.0 = min.0.min(x);
            min.1 = min.1.min(y);
            max.0 = max.0.max(x);
            max.1 = max.1.max(y);
        }
        (min, max)
    }
}

/// A pinhole is at most this many pixels in one component.
pub const PINHOLE_MAX_PIXELS: usize = 2;
/// At most this many pinhole pixels per frame in the checked region.
pub const PINHOLE_MAX_TOTAL: usize = 8;
/// Highlight for offending components in debug images; must differ
/// from the magenta sentinel and the sky gradient.
const HOLE_HIGHLIGHT: [u8; 4] = [255, 255, 0, 255];

pub fn is_sentinel(p: &Rgba<u8>) -> bool {
    p[0] >= 250 && p[1] <= 5 && p[2] >= 250
}

/// The 4-connected components of sentinel pixels inside `region`
/// (`x0..x1` by `y0..y1`, end-exclusive).
pub fn sentinel_components(image: &RgbaImage, region: (u32, u32, u32, u32)) -> Vec<HoleComponent> {
    let (x0, y0, x1, y1) = region;
    let stride = x1 - x0;
    let mut visited = vec![false; (stride * (y1 - y0)) as usize];
    let in_region = |x: i64, y: i64| {
        x >= i64::from(x0) && x < i64::from(x1) && y >= i64::from(y0) && y < i64::from(y1)
    };
    let mut components = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let entry = ((y - y0) * stride + (x - x0)) as usize;
            if visited[entry] || !is_sentinel(image.get_pixel(x, y)) {
                continue;
            }
            // Flood fill from (x, y).
            let mut pixels = Vec::new();
            let mut stack = vec![(x, y)];
            visited[entry] = true;
            while let Some((cx, cy)) = stack.pop() {
                pixels.push((cx, cy));
                for (dx, dy) in [(1_i64, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (i64::from(cx) + dx, i64::from(cy) + dy);
                    if !in_region(nx, ny) {
                        continue;
                    }
                    let (nx, ny) = (nx as u32, ny as u32);
                    let neighbor = ((ny - y0) * stride + (nx - x0)) as usize;
                    if !visited[neighbor] && is_sentinel(image.get_pixel(nx, ny)) {
                        visited[neighbor] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            components.push(HoleComponent { pixels });
        }
    }
    components
}

/// "N components, largest M px, T sentinel px" — the per-frame report.
pub fn describe_holes(components: &[HoleComponent]) -> String {
    let largest = components
        .iter()
        .map(HoleComponent::size)
        .max()
        .unwrap_or(0);
    let total: usize = components.iter().map(HoleComponent::size).sum();
    let count = components.len();
    format!("{count} components, largest {largest} px, {total} sentinel px")
}

/// Every component with its size and bounding box, for diagnostics:
/// "2 px (10,20)..(11,20), 1 px (40,50)..(40,50)".
pub fn component_details(components: &[HoleComponent]) -> String {
    if components.is_empty() {
        return "none".to_owned();
    }
    let parts: Vec<String> = components
        .iter()
        .map(|c| {
            let ((min_x, min_y), (max_x, max_y)) = c.bounds();
            format!("{} px ({min_x},{min_y})..({max_x},{max_y})", c.size())
        })
        .collect();
    parts.join(", ")
}

/// A pinhole tolerance: components of at most `max_pixels` pixels, at most
/// `max_total` such pixels per frame.
#[derive(Debug, Clone, Copy)]
pub struct PinholeRule {
    pub max_pixels: usize,
    pub max_total: usize,
}

impl PinholeRule {
    /// The default rule ([`PINHOLE_MAX_PIXELS`], [`PINHOLE_MAX_TOTAL`]),
    /// used by the real-server fly test.
    pub const STRICT: PinholeRule = PinholeRule {
        max_pixels: PINHOLE_MAX_PIXELS,
        max_total: PINHOLE_MAX_TOTAL,
    };
}

/// Pixels in components small enough to be pinholes under `rule`.
pub fn pinhole_total_with(components: &[HoleComponent], rule: PinholeRule) -> usize {
    components
        .iter()
        .filter(|c| c.size() <= rule.max_pixels)
        .map(HoleComponent::size)
        .sum()
}

/// [`pinhole_total_with`] under [`PinholeRule::STRICT`].
pub fn pinhole_total(components: &[HoleComponent]) -> usize {
    pinhole_total_with(components, PinholeRule::STRICT)
}

/// Whether every component is a pinhole under `rule` and they fit its
/// per-frame cap.
pub fn only_pinholes_with(components: &[HoleComponent], rule: PinholeRule) -> bool {
    components.iter().all(|c| c.size() <= rule.max_pixels)
        && pinhole_total_with(components, rule) <= rule.max_total
}

/// [`only_pinholes_with`] under [`PinholeRule::STRICT`].
pub fn only_pinholes(components: &[HoleComponent]) -> bool {
    only_pinholes_with(components, PinholeRule::STRICT)
}

/// [`assert_only_pinholes_with`] under [`PinholeRule::STRICT`].
pub fn assert_only_pinholes(
    image: &RgbaImage,
    region: (u32, u32, u32, u32),
    label: &str,
    image_path: &std::path::Path,
) {
    assert_only_pinholes_with(image, region, label, image_path, PinholeRule::STRICT);
}

/// The frame hole check: sentinel pixels in `region` must be pinhole
/// components only, within `rule`'s per-frame cap. Prints the stats; on
/// failure saves the frame at `image_path`, saves a debug image with
/// the offending components highlighted next to it, and panics with
/// their bounding boxes.
pub fn assert_only_pinholes_with(
    image: &RgbaImage,
    region: (u32, u32, u32, u32),
    label: &str,
    image_path: &std::path::Path,
    rule: PinholeRule,
) {
    let components = sentinel_components(image, region);
    println!("{label}: {}", describe_holes(&components));
    if only_pinholes_with(&components, rule) {
        return;
    }
    let oversized = components
        .iter()
        .filter(|c| c.size() > rule.max_pixels)
        .count();
    // On a total-cap failure every component contributes; otherwise
    // only the oversized ones offend.
    let over_total = pinhole_total_with(&components, rule) > rule.max_total;
    let offending: Vec<&HoleComponent> = if over_total {
        components.iter().collect()
    } else {
        components
            .iter()
            .filter(|c| c.size() > rule.max_pixels)
            .collect()
    };
    let mut debug = image.clone();
    for component in &offending {
        for &(x, y) in &component.pixels {
            debug.put_pixel(x, y, Rgba(HOLE_HIGHLIGHT));
        }
    }
    let debug_path = image_path.with_extension("holes.png");
    std::fs::create_dir_all(image_path.parent().unwrap()).ok();
    image.save(image_path).ok();
    debug.save(&debug_path).ok();
    let boxes: Vec<String> = offending
        .iter()
        .map(|c| {
            let ((min_x, min_y), (max_x, max_y)) = c.bounds();
            format!("({min_x},{min_y})..({max_x},{max_y}) {}px", c.size())
        })
        .collect();
    panic!(
        "{label}: holes beyond pinhole tolerance ({} px per component, {} px \
         per frame): {oversized} oversized components, pinhole total {}, \
         offending [{}]; images {} and {}",
        rule.max_pixels,
        rule.max_total,
        pinhole_total_with(&components, rule),
        boxes.join(", "),
        image_path.display(),
        debug_path.display(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 360;

    /// The checked region for the synthetic frames: the middle half.
    fn region() -> (u32, u32, u32, u32) {
        (
            WIDTH / 4,
            HEIGHT / 4,
            WIDTH - WIDTH / 4,
            HEIGHT - HEIGHT / 4,
        )
    }

    /// A black frame with sentinel pixels at the given coordinates, for
    /// the detector's unit test.
    fn synthetic_frame(sentinels: &[(u32, u32)]) -> RgbaImage {
        let mut image = RgbaImage::from_pixel(WIDTH, HEIGHT, Rgba([0, 0, 0, 255]));
        for &(x, y) in sentinels {
            image.put_pixel(x, y, Rgba([255, 0, 255, 255]));
        }
        image
    }

    #[test]
    fn hole_detector_distinguishes_pinholes_from_holes() {
        let region = region();

        // Isolated single pixels (diagonals do not connect: 4-connected)
        // and a horizontal 2-pixel pair are pinholes.
        let pinholes = synthetic_frame(&[
            (WIDTH / 4 + 4, HEIGHT / 4 + 4),
            (WIDTH / 4 + 5, HEIGHT / 4 + 5),
            (WIDTH / 4 + 40, HEIGHT / 4 + 4),
            (WIDTH / 4 + 60, HEIGHT / 4 + 20),
            (WIDTH / 4 + 61, HEIGHT / 4 + 20),
        ]);
        let components = sentinel_components(&pinholes, region);
        assert_eq!(components.iter().map(HoleComponent::size).sum::<usize>(), 5);
        assert_eq!(components.len(), 4, "the pair merges, diagonals do not");
        assert!(only_pinholes(&components));

        // A 3-pixel L shape is too big for a pinhole.
        let l_shape = synthetic_frame(&[
            (WIDTH / 4 + 10, HEIGHT / 4 + 10),
            (WIDTH / 4 + 11, HEIGHT / 4 + 10),
            (WIDTH / 4 + 10, HEIGHT / 4 + 11),
        ]);
        let components = sentinel_components(&l_shape, region);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].size(), 3);
        assert!(!only_pinholes(&components));

        // A 1x20 strip — the shape of a ring-border crack — fails.
        let strip: Vec<(u32, u32)> = (0..20)
            .map(|i| (WIDTH / 4 + 30, HEIGHT / 4 + 10 + i))
            .collect();
        let components = sentinel_components(&synthetic_frame(&strip), region);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].size(), 20);
        assert_eq!(
            components[0].bounds(),
            (
                (WIDTH / 4 + 30, HEIGHT / 4 + 10),
                (WIDTH / 4 + 30, HEIGHT / 4 + 29)
            )
        );
        assert!(!only_pinholes(&components));

        // Nine scattered singles pass the component size but bust the
        // per-frame total of 8.
        let scattered: Vec<(u32, u32)> = (0..9)
            .map(|i| (WIDTH / 4 + 5 * i, HEIGHT / 4 + 50))
            .collect();
        let components = sentinel_components(&synthetic_frame(&scattered), region);
        assert_eq!(components.len(), 9);
        assert_eq!(pinhole_total(&components), 9);
        assert!(!only_pinholes(&components));
    }
}
