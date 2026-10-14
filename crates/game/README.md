# jbm-game

The `jbm` binary: starts the client and an in-process server talking over a
channel (see docs/design.md, "Architecture overview").

**Public API so far:** a `main` that prints the engine name and version and
exits 0. Stage 0 scaffolding; nothing starts yet.

**Invariants**

- The only executable crate in the workspace.
- Depends on client, server, and core; client and server never depend on each
  other.
