# jbm-client

The player-facing side: renderer, UI, input, audio, and client-side
prediction.

**Public API so far:** empty (Stage 0 scaffolding).

**Invariants**

- Depends only on `jbm-core`; must NOT depend on `jbm-server` (enforced by
  `cargo tree -p jbm-client` in CI-style checks).
- Renderer changes need screenshot tests.
