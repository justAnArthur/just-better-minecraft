# jbm-core

Shared types for the whole engine: cubes, palettes, block states, registries,
and the protocol messages the client and server exchange.

**Public API so far:** `ENGINE_NAME` — the engine's canonical name string.

## Benchmarks

Criterion benches live in `benches/`, registered in `Cargo.toml`:

```toml
[[bench]]
name = "core_bench"
harness = false
```

To add a case, add a function to `benches/core_bench.rs` inside a
`benchmark_group` named for the area, with the case as the function name, so
reports read `<area>/<case>` (e.g. `sample/fill_32cubed`). `criterion` is a
dev-dependency via `[workspace.dependencies]` only — never a runtime dep.

```bash
cargo bench -p jbm-core -- --test   # run each bench once (what CI calls)
cargo bench -p jbm-core             # full criterion run with numbers
```

**Invariants**

- No dependencies on any other jbm crate; everything else may depend on this.
- Contains types only; no I/O, no threads, no engine logic.
- Never depends on `jbm-client` or `jbm-server` (both depend on *this* crate).
