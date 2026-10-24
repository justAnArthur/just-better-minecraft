//! Rendering: offscreen targets, the frame renderer (sky + terrain +
//! outline + overlay), the render-scale composite, the renderer-owned
//! frame settings, the terrain pass (paged face storage, suballocator,
//! meta slots, GPU/CPU culling, multi-draw indirect), the Hi-Z pyramid,
//! the block texture array and the GPU pass timer.

pub mod composite;
pub mod hiz;
pub mod lines;
pub mod offscreen;
pub mod outline;
pub mod overlay;
pub mod renderer;
pub mod settings;
pub mod suballoc;
pub mod terrain;
pub mod texture_array;
pub mod timing;
pub mod triangle;
