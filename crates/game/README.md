# jbm-game

The `jbm` binary: starts the client and an in-process server talking over a
channel (see docs/design.md, "Architecture overview").

**Public API so far:** `main` calls `jbm_client::app::run()`, which opens
the game window (fly camera over the sky with a debug overlay). The
in-process server (task 1.3) and its wiring arrive in Stage 1 wave B.

**Invariants**

- The only executable crate in the workspace.
- Depends on client, server, and core; client and server never depend on each
  other.
