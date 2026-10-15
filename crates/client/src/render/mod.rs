//! Rendering: offscreen targets, the frame renderer (sky + terrain +
//! outline + overlay), the terrain pass (paged face storage, suballocator,
//! meta slots, GPU/CPU culling, multi-draw indirect), the Hi-Z pyramid,
//! and the block texture array.

pub mod chunks;
pub mod hiz;
pub mod offscreen;
pub mod outline;
pub mod overlay;
pub mod renderer;
pub mod suballoc;
pub mod texture_array;
pub mod triangle;
