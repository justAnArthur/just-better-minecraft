//! Turning cubes into GPU meshes: the block appearance table, the packed
//! face format, the greedy mesher, and the worker pool that runs it.
//! See crates/client/README.md for the face format and meshing rules.

pub mod appearance;
pub mod face;
pub mod greedy;
pub mod pool;
