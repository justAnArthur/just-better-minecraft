# jbm-modhost

Wasmtime embedding and WIT bindings (both sides), used by the client and the
server to run WASM mods.

**Public API so far:** empty (Stage 0 scaffolding).

**Invariants**

- Depends only on `jbm-core`.
- The only crate allowed to link wasmtime; no other crate touches mod
  execution.
- Hot paths (meshing, collision, light) never call into mods — mods are
  consulted through this host only at defined extension points.
