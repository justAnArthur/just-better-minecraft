# jbm-server

The authoritative simulation: world generation, lighting, storage, and the
server side of the protocol.

**Public API so far:** empty (Stage 0 scaffolding).

**Invariants**

- Depends only on `jbm-core`; never on `jbm-client`. Both sides communicate
  exclusively through `jbm-core` protocol messages.
- Hot paths (lighting, worldgen inner loops) never call into mods.
