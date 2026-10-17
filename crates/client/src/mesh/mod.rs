//! Turning cubes into GPU meshes: the block appearance table, the packed
//! face format, the greedy mesher, and the worker pool that runs it.
//! See crates/client/README.md for the face format and meshing rules.
//!
//! The mesher itself is split by phase: [`input`] builds the padded cell
//! view (and owns the `mesh` entry point), [`greedy`] sweeps it, and
//! [`seams`] holds the level-seam crack-hider rules. [`oracle`] holds the
//! test-only reference meshers.

pub mod appearance;
pub mod face;
pub mod greedy;
pub mod input;
pub mod pool;
pub mod seams;

/// Test-only reference meshers the greedy output is checked against.
#[cfg(test)]
pub(crate) mod oracle;
