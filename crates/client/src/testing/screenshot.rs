//! Golden-image comparison for screenshot tests.
//!
//! A test renders offscreen, calls [`assert_matches_golden`] with a name,
//! and the rendered image is compared against a committed PNG under
//! `crates/client/tests/golden/`. Set `JBM_UPDATE_GOLDEN=1` to regenerate
//! goldens instead of comparing.

use image::{Rgba, RgbaImage};
use std::path::PathBuf;

use crate::gpu::Gpu;

/// A pixel counts as identical when every channel differs by at most this
/// much. Absorbs small rasterizer and blending differences between the
/// Metal, Vulkan and lavapipe backends.
const MAX_CHANNEL_DELTA: u8 = 3;

/// Up to this fraction of pixels may differ before the test fails. Absorbs
/// edge-coverage differences between hardware rasterizers; large-area
/// regressions still fail loudly.
const MAX_DIFFERING_FRACTION: f64 = 0.005;

/// Where committed goldens live: `crates/client/tests/golden/<name>.png`.
fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.png"))
}

/// Where failure images go: `<target dir>/screenshots/<name>.*.png`.
fn failure_dir() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("target")
        })
        .join("screenshots")
}

fn save(image: &RgbaImage, path: &std::path::Path) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("cannot create screenshot dir {}: {e}", parent.display()));
    }
    image
        .save(path)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}

/// Asserts `actual` matches the committed golden for `name`.
///
/// With `JBM_UPDATE_GOLDEN=1` in the environment, (over)writes the golden
/// and returns. A missing golden panics after writing `<name>.actual.png`
/// next to where it would live. A comparison failure writes
/// `target/screenshots/<name>.actual.png` and `<name>.diff.png` before
/// panicking.
pub fn assert_matches_golden(name: &str, actual: &RgbaImage) {
    let golden_path = golden_path(name);
    if std::env::var_os("JBM_UPDATE_GOLDEN").is_some_and(|v| v == "1") {
        save(actual, &golden_path);
        return;
    }
    if !golden_path.is_file() {
        let actual_path = golden_path.with_file_name(format!("{name}.actual.png"));
        save(actual, &actual_path);
        panic!(
            "golden image {} is missing (wrote {}); run the test with \
             JBM_UPDATE_GOLDEN=1 and commit the golden",
            golden_path.display(),
            actual_path.display()
        );
    }
    let golden = image::open(&golden_path)
        .unwrap_or_else(|e| panic!("cannot read golden {}: {e}", golden_path.display()))
        .to_rgba8();
    assert_similar(name, &golden, actual);
}

/// Panics unless `actual` is close enough to `golden`.
fn assert_similar(name: &str, golden: &RgbaImage, actual: &RgbaImage) {
    if golden.dimensions() != actual.dimensions() {
        let actual_path = failure_dir().join(format!("{name}.actual.png"));
        save(actual, &actual_path);
        panic!(
            "screenshot {name}: size mismatch, golden is {}x{} but actual is \
             {}x{}; actual written to {}",
            golden.width(),
            golden.height(),
            actual.width(),
            actual.height(),
            actual_path.display()
        );
    }

    let (width, height) = actual.dimensions();
    let total = width as usize * height as usize;
    let differing = golden
        .pixels()
        .zip(actual.pixels())
        .filter(|(g, a)| differs(g, a))
        .count();
    let fraction = differing as f64 / total as f64;
    if fraction > MAX_DIFFERING_FRACTION {
        let dir = failure_dir();
        let actual_path = dir.join(format!("{name}.actual.png"));
        let diff_path = dir.join(format!("{name}.diff.png"));
        save(actual, &actual_path);
        save(&diff_image(golden, actual), &diff_path);
        panic!(
            "screenshot {name}: {differing} of {total} pixels ({:.3}%) differ by more \
             than {MAX_CHANNEL_DELTA} per channel (limit {:.3}%); actual: {}, diff: {}",
            fraction * 100.0,
            MAX_DIFFERING_FRACTION * 100.0,
            actual_path.display(),
            diff_path.display()
        );
    }
}

fn differs(g: &Rgba<u8>, a: &Rgba<u8>) -> bool {
    g.0.iter()
        .zip(a.0.iter())
        .any(|(g, a)| g.abs_diff(*a) > MAX_CHANNEL_DELTA)
}

/// Differing pixels in magenta, everything else dimmed grayscale.
fn diff_image(golden: &RgbaImage, actual: &RgbaImage) -> RgbaImage {
    let mut diff = RgbaImage::new(actual.width(), actual.height());
    for (g, (a, d)) in golden.pixels().zip(actual.pixels().zip(diff.pixels_mut())) {
        if differs(g, a) {
            *d = Rgba([255, 0, 255, 255]);
        } else {
            let [r, gr, b, _] = a.0;
            let gray = (u32::from(r) * 299 + u32::from(gr) * 587 + u32::from(b) * 114) / 1000 / 4;
            let gray = gray as u8;
            *d = Rgba([gray, gray, gray, 255]);
        }
    }
    diff
}

/// Returns a headless GPU, or `None` after printing "skipping: no GPU
/// adapter" when the machine has no usable adapter. Panics instead when
/// `JBM_REQUIRE_GPU=1` is set, so environments that must have a GPU fail
/// loudly rather than silently skipping.
pub fn gpu_or_skip() -> Option<Gpu> {
    match Gpu::headless() {
        Ok(gpu) => Some(gpu),
        Err(err) => {
            if std::env::var_os("JBM_REQUIRE_GPU").is_some_and(|v| v == "1") {
                panic!("JBM_REQUIRE_GPU=1 but no GPU adapter is available: {err}");
            }
            println!("skipping: no GPU adapter");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, channel: [u8; 4]) -> RgbaImage {
        RgbaImage::from_pixel(width, height, Rgba(channel))
    }

    #[test]
    fn identical_images_pass() {
        let image = solid(256, 256, [10, 20, 30, 255]);
        assert_similar("unit_identical", &image, &image);
    }

    #[test]
    fn one_pixel_off_by_10_passes() {
        let golden = solid(256, 256, [100, 100, 100, 255]);
        let mut actual = golden.clone();
        actual.put_pixel(128, 128, Rgba([110, 100, 100, 255]));
        assert_similar("unit_one_pixel", &golden, &actual);
    }

    #[test]
    #[should_panic(expected = "pixels (")]
    fn one_percent_of_pixels_off_fails() {
        let golden = solid(256, 256, [100, 100, 100, 255]);
        let mut actual = golden.clone();
        // 656 of 65536 pixels = 1.001%, above the 0.5% limit.
        for y in 0..8 {
            for x in 0..82 {
                actual.put_pixel(x, y, Rgba([110, 100, 100, 255]));
            }
        }
        assert_similar("unit_one_percent", &golden, &actual);
    }

    #[test]
    #[should_panic(expected = "size mismatch")]
    fn size_mismatch_fails() {
        let golden = solid(256, 256, [10, 20, 30, 255]);
        let actual = solid(128, 128, [10, 20, 30, 255]);
        assert_similar("unit_size_mismatch", &golden, &actual);
    }
}
