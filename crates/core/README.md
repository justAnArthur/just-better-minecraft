# jbm-core

Shared types for the whole engine: cubes, palettes, block states, registries,
and the protocol messages the client and server exchange.

**Public API so far:** `ENGINE_NAME` — the engine's canonical name string;
the `protocol` and `link` modules below.

## Protocol and links

`protocol` holds the message enums the client and server exchange:
`PROTOCOL_VERSION`, `PlayerId`, `ClientMsg` (`Hello`, `PlayerInput`,
`Ping`, `Disconnect`) and `ServerMsg` (`Welcome`, `Rejected`, `Tick`,
`Pong`, `Disconnected`). Messages are plain owned data — no `Arc`, no
references, no trait objects — all derive `Debug, Clone, PartialEq`, and
stay serializable for a future network transport (no serde yet).

`link` is the transport boundary: `trait Link<Out, In>: Send` with
non-blocking `try_recv` (returns `Ok(None)` when empty) and
`send`/`try_recv` returning `LinkClosed` only once the other end is gone
and the queue has drained. `ClientLink`/`ServerLink` are boxed trait
aliases for the two ends, and `in_process_pair()` builds a connected pair
on `std::sync::mpsc`. A network socket can implement `Link` later without
either side changing.

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
