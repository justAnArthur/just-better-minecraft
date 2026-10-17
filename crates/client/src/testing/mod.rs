//! Test-only helpers: golden-image comparison, hole detection for
//! screenshot tests, the offscreen render rig, and the shared
//! heightfield scene fixtures.

pub mod heightfield;
pub mod holes;
pub mod offscreen;
pub mod screenshot;

pub use offscreen::OffscreenRig;
