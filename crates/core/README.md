# jbm-core

Shared types for the whole engine: cubes, palettes, block states, registries,
and the protocol messages the client and server exchange.

**Public API so far:** `ENGINE_NAME` — the engine's canonical name string.

**Invariants**

- No dependencies on any other jbm crate; everything else may depend on this.
- Contains types only; no I/O, no threads, no engine logic.
- Never depends on `jbm-client` or `jbm-server` (both depend on *this* crate).
