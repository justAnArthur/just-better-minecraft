//! Screenshot tests: 256x256 offscreen renders compared against committed
//! goldens. Each scene pins its camera so the image is reproducible.

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::overlay::OverlayText;
use jbm_client::render::renderer::{DepthTarget, Renderer};
use jbm_client::render::triangle;
use jbm_client::testing::screenshot::{assert_matches_golden, gpu_or_skip};

/// The sky_horizon camera: at (0, 80, 0), yaw 0, pitch `pitch_deg`, fov 70.
fn sky_camera(pitch_deg: f32) -> Camera {
    let mut camera = Camera::new(DVec3::new(0.0, 80.0, 0.0), 70.0_f32.to_radians(), 1.0);
    camera.set_pitch(pitch_deg.to_radians());
    camera
}

fn render_and_compare(name: &str, camera: &Camera, overlay: Option<OverlayText>) {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let target = OffscreenTarget::new(&gpu.device, 256, 256);
    let depth = DepthTarget::new(&gpu.device, 256, 256);
    let mut renderer = Renderer::new(&gpu, target.format());
    renderer.set_viewport(256, 256);
    renderer.render(
        &gpu,
        &target.create_view(),
        depth.view(),
        camera,
        overlay.as_ref(),
    );
    let image = target.read_rgba(&gpu);
    assert_matches_golden(name, &image);
}

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

#[test]
fn sky_horizon_matches_golden() {
    render_and_compare("sky_horizon", &sky_camera(0.0), None);
}

#[test]
fn sky_up_matches_golden() {
    render_and_compare("sky_up", &sky_camera(60.0), None);
}

#[test]
fn sky_down_matches_golden() {
    render_and_compare("sky_down", &sky_camera(-60.0), None);
}

#[test]
fn debug_overlay_matches_golden() {
    let mut overlay = OverlayText::new();
    overlay.push("FPS 60  16.67 ms");
    overlay.push("XYZ 0.00 80.00 0.00");
    render_and_compare("debug_overlay", &sky_camera(0.0), Some(overlay));
}
