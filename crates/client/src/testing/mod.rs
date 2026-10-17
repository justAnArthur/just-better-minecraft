//! Test-only helpers: golden-image comparison, hole detection for
//! screenshot tests, and the offscreen render rig.

pub mod holes;
pub mod offscreen;
pub mod screenshot;

pub use offscreen::OffscreenRig;
