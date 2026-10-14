# jbm-testkit

Test-only helpers: a headless server harness and shared fixtures.

**Public API so far:** `HeadlessServer` — a real `jbm_server::Server`
driven by a manual, deterministic time source, plus in-process
`TestClient`s connected to it.

- `HeadlessServer::new()` is protocol-only (no cube streaming), so tests
  that assert exact message sequences see only handshake/ping/tick
  traffic. `HeadlessServer::with_config(ServerConfig)` takes an explicit
  config — e.g. streaming enabled with small radii and inline generation
  (`gen_threads = 0`) for deterministic cube tests.
- `step(ticks)` advances exactly that many real server ticks, `run_for(d)`
  lets `d` of simulated time pass (delivered in tick-sized slices, so the
  catch-up cap never trips), `tick()` reports the count, `server(&self)`
  exposes the `Server` for assertions.
- `step_until(clients, cond, max_ticks)` steps one tick at a time,
  draining every client each tick (keeping their `loaded()` sets current),
  until `cond(&Server, &[&TestClient])` holds or the tick budget runs out.
  Returns whether the condition was met. Bounded — never a naked loop.
- `connect(name)` creates an in-process link pair, hands the server end to
  the server and sends `Hello`; the client is welcomed on the next
  `step(1)`. `connect_raw()` does the same without the `Hello`, for
  protocol-level tests that drive the handshake themselves (e.g. a wrong
  protocol version).
- `TestClient`: `send(ClientMsg)`, `drain() -> Vec<ServerMsg>` (remembers
  the `Welcome`'s id) and `player_id() -> Option<PlayerId>`. `drain` also
  maintains the client's loaded-cube set from `CubeData`/`UnloadCube`: a
  duplicate `CubeData` for an already-loaded cube, or an `UnloadCube` for
  an unloaded one, panics with the position. Read it back with
  `loaded() -> &HashSet<CubePos>`; `cube_data_count()` and
  `unload_count()` are totals for assertions (zero unloads is the
  hysteresis check).

## Writing a headless test

Put integration tests in `tests/` (see `tests/headless.rs` for the clock,
`tests/protocol.rs` for the client-server protocol exchange, and
`tests/streaming.rs` for cube streaming):

```rust
use jbm_core::protocol::{ClientMsg, ServerMsg};
use jbm_testkit::HeadlessServer;

let mut server = HeadlessServer::new();
let mut client = server.connect("alice");

server.step(1);                            // client is welcomed
assert!(matches!(client.drain().first(), Some(ServerMsg::Welcome { .. })));

server.step(100);                          // exact tick counts
assert_eq!(server.tick(), 101);
assert_eq!(server.server().players().count(), 1);
```

Other crates add `jbm-testkit` as a dev-dependency only. Prefer `step`
when a test cares about exact tick numbers, `run_for` when it cares about
durations; both are deterministic.

**Invariants**

- Never a runtime dependency of any shipping crate; other crates pull it
  in only as a dev-dependency.
- May depend on `jbm-core` and `jbm-server`, but never on `jbm-client`
  (client tests add it as their own dev-dependency if needed).
