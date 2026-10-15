//! The player-facing side: renderer, UI, input, audio, and client-side
//! prediction. Talks to the server only through `jbm-core` protocol
//! messages.

pub mod app;
pub mod camera;
pub mod edit;
pub mod gpu;
pub mod input;
pub mod mesh;
pub mod predict;
pub mod render;
pub mod world;

#[cfg(any(test, feature = "testing"))]
pub mod testing;
