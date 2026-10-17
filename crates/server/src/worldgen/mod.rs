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
//! fixes the bits: the classic graph computes `(40 · hilliness) · rolling`,
//! not `40 · (hilliness · rolling)`. When editing either shipped graph
//! (`assets/worldgen/overworld.json`, `assets/worldgen/classic.json`),
//! keep operand nesting exactly as written or the output will drift off by
//! one ulp and the equivalence tests will catch it. The classic graph is
//! frozen bit-for-bit against `worldgen::reference`; the overworld's
//! reference values are its own nodes' reference implementations.

use std::sync::{Arc, OnceLock};

mod graph;
mod noise;
/// The hand-written heightmap terrain the graph generator is tested and
/// benchmarked against. Test-only: enabled by the crate's `reference`
/// cargo feature, which the crate's own tests and benches turn on through
/// a self dev-dependency (the pattern `jbm-client` uses for `testing`).
#[cfg(feature = "reference")]
pub mod reference;

pub use graph::{
    Band, CompiledGraph, GRID_SIDE, GRID_VOLUME, Graph, GraphError, Node, NodeValue, Outputs, Salt,
    compile,
};

/// The shipped overworld density graph (task 2.8): continents, mountain
/// ranges and valleys at kilometre scale — see the crate README's
/// "Worldgen" section for the layer-by-layer description. See the
/// association warning above before editing it.
pub const OVERWORLD_JSON: &str = include_str!("../../assets/worldgen/overworld.json");

/// The task 1.5/2.1 terrain, frozen bit-for-bit. Tests, benches and
/// examples that depend on that terrain pin [`Preset::Classic`]; old
/// worlds' LOD0 records around edits keep generating from it too whenever
/// their fingerprint matches.
pub const CLASSIC_JSON: &str = include_str!("../../assets/worldgen/classic.json");

/// Which shipped worldgen a server generates with ([`crate::server::
/// ServerConfig::worldgen`]). `Overworld` is the default; `Classic` is the
/// pre-2.8 terrain, kept so tests and goldens that depend on it keep their
/// meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Overworld,
    Classic,
}

impl Preset {
    /// The preset's JSON text.
    pub fn json(self) -> &'static str {
        match self {
            Preset::Overworld => OVERWORLD_JSON,
            Preset::Classic => CLASSIC_JSON,
        }
    }

    /// The preset's compiled graph, compiled once per process. A broken
    /// shipped graph is a bug caught by the worldgen tests, so loading
    /// panics.
    pub fn compiled(self) -> Arc<CompiledGraph> {
        match self {
            Preset::Overworld => overworld(),
            Preset::Classic => classic(),
        }
    }

    /// The preset's sea level: air at or below it becomes water (task 2.8).
    /// `None` for `Classic` — it never had water, and its tests and goldens
    /// stay meaningful only if it never gains any.
    pub fn sea_level(self) -> Option<i64> {
        match self {
            Preset::Overworld => Some(crate::terrain::SEA_LEVEL),
            Preset::Classic => None,
        }
    }
}

/// The compiled shipped overworld graph. Compiled once per process; a
/// broken shipped graph is a bug caught by the worldgen tests, so loading
/// it panics.
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

/// The compiled classic graph (see [`CLASSIC_JSON`]).
pub fn classic() -> Arc<CompiledGraph> {
    static CLASSIC: OnceLock<Arc<CompiledGraph>> = OnceLock::new();
    CLASSIC
        .get_or_init(|| {
            let graph = Graph::from_json(CLASSIC_JSON).expect("the classic graph parses (tested)");
            Arc::new(compile(&graph).expect("the classic graph compiles (tested)"))
        })
        .clone()
}
