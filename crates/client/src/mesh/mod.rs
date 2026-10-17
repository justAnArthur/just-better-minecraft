//! Turning cubes into GPU meshes: the block appearance table, the packed
//! face format, the greedy mesher, and the worker pool that runs it.
//! See crates/client/README.md for the face format and meshing rules.
//!
//! The mesher is split by phase: [`mesh`] (here) is the entry point,
//! [`input`] builds the padded cell view, [`greedy`] sweeps it, and
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

pub use input::{MeshInput, MeshOutput, build_padded, neighbor_index};

use appearance::BlockAppearance;
use greedy::mesh_padded;

/// Meshes one cube. See the README for the face rule and AO rule.
pub fn mesh(input: &MeshInput, appearance: &BlockAppearance) -> MeshOutput {
    if let Some(state) = input.center.uniform() {
        let app = appearance.get(state);
        if app.air {
            // Uniform air owns no faces, whatever the neighbors are.
            return MeshOutput::default();
        }
        if app.opaque && enclosed_by_opaque(input, appearance) {
            // Fully surrounded by uniform opaque face neighbors: nothing
            // of this cube is visible.
            return MeshOutput::default();
        }
    }
    let padded = build_padded(input, appearance);
    mesh_padded(&padded, appearance, input.level)
}

/// Whether all six face neighbors are present, uniform and opaque.
fn enclosed_by_opaque(input: &MeshInput, appearance: &BlockAppearance) -> bool {
    const FACE_OFFSETS: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    FACE_OFFSETS.iter().all(|&(dx, dy, dz)| {
        input
            .neighbor(dx, dy, dz)
            .and_then(|cube| cube.uniform())
            .is_some_and(|state| appearance.get(state).opaque)
    })
}
