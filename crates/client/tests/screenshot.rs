//! Screenshot test: renders the triangle pipeline offscreen at 256x256 and
//! compares against the committed golden.

use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::triangle;
use jbm_client::testing::screenshot::{assert_matches_golden, gpu_or_skip};

#[test]
fn triangle_matches_golden() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    triangle::draw_triangle(&gpu, &target);
    let image = target.read_rgba(&gpu);
    assert_matches_golden("triangle", &image);
}
