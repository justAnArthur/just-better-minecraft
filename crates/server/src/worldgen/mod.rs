//! Declarative worldgen: a JSON graph of density functions, compiled once
//! into a flat instruction list and evaluated natively at any resolution.
//! See the "Worldgen graph" section of the crate README for the format, the
//! node table and how to add a node.
//!
//! **Determinism rule** (carried over from the original `terrain.rs`):
//! every node is a pure function of its inputs built only from IEEE-754
//! `+`, `-`, `*`, `/` and `floor` on `f64`, plus comparisons and integer
//! hashing to pick gradients from a fixed table. Division and floor are
//! IEEE-exact, and multiplication by a power of two (the `fbm2` frequency
//! ladder) only shifts an exponent. Every one of those operations is
//! bit-exact on every platform, so the same `(seed, graph)` gives identical
//! density values everywhere. No `sin`/`cos`/`exp`/`powf` in any node.
//!
//! **Operand association matters.** Float addition and multiplication are
//! commutative but not associative, so the *shape* of the expression tree
//! fixes the bits: the shipped graph computes `(40 · hilliness) · rolling`,
//! not `40 · (hilliness · rolling)`. When editing
//! `assets/worldgen/overworld.json`, keep the tree shape of today's
//! reference terrain (`worldgen::reference`) or the output will drift off
//! by one ulp and the equivalence tests will catch it.

use std::sync::{Arc, OnceLock};

mod graph;
#[doc(hidden)]
pub mod reference;

pub use graph::{
    Band, CompiledGraph, GRID_SIDE, GRID_VOLUME, Graph, GraphError, Node, NodeValue, Outputs, Salt,
    compile,
};

/// The shipped overworld density graph, expressed exactly as the reference
/// terrain of task 1.5 (`worldgen::reference`): see the association warning
/// above before editing it.
pub const OVERWORLD_JSON: &str = include_str!("../../assets/worldgen/overworld.json");

/// The compiled shipped graph. Compiled once per process; a broken shipped
/// graph is a bug caught by the worldgen tests, so loading it panics.
pub fn overworld() -> Arc<CompiledGraph> {
    static OVERWORLD: OnceLock<Arc<CompiledGraph>> = OnceLock::new();
    OVERWORLD
        .get_or_init(|| {
            let graph = Graph::from_json(OVERWORLD_JSON)
                .expect("the shipped overworld graph parses (tested)");
            Arc::new(compile(&graph).expect("the shipped overworld graph compiles (tested)"))
        })
        .clone()
}
