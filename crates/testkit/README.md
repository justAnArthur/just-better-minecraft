# jbm-testkit

Test-only helpers: a headless server harness and shared fixtures.

**Public API so far:** empty (Stage 0 scaffolding).

**Invariants**

- Never a runtime dependency of any shipping crate; other crates pull it in
  only as a dev-dependency.
- May depend on `jbm-core` and `jbm-server`, but never on `jbm-client` (client
  tests add it as their own dev-dependency if needed).
