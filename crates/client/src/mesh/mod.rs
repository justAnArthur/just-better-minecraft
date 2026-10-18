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

/// Task 2.13 A's quad-split study: mesh `input` with merge-key fields
/// dropped and return the face count. `ignore_ao` clears the AO byte from
/// the merge key, `ignore_layer` the texture-layer field, so quads merge
/// across them; the counts against the real mesher show what AO and the
/// per-direction layers cost in triangles. Test/bench only.
#[cfg(any(test, feature = "testing"))]
pub fn mesh_split_study(
    input: &MeshInput,
    appearance: &BlockAppearance,
    ignore_ao: bool,
    ignore_layer: bool,
) -> usize {
    let mut keep = u32::MAX;
    if ignore_ao {
        keep &= !greedy::KEY_AO_BITS;
    }
    if ignore_layer {
        keep &= !greedy::KEY_LAYER_BITS;
    }
    let padded = build_padded(input, appearance);
    greedy::sweep_key_masked(&padded, appearance, input.level, keep)
        .faces
        .len()
}

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
