//! The player-facing side: renderer, UI, input, audio, and client-side
//! prediction. Talks to the server only through `jbm-core` protocol
//! messages.

pub mod app;
pub mod camera;
pub mod edit;
pub mod gpu;
pub mod hud;
pub mod input;
pub mod lod;
pub mod mesh;
pub mod predict;
pub mod render;
pub mod terrain_sync;
pub mod world;

#[cfg(any(test, feature = "testing"))]
pub mod testing;
