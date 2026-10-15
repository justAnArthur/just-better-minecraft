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

**One caveat for persistence tests (task 2.3):** a HeadlessServer's ticks
run in microseconds while the `jbm-save` thread commits on real disk. A
test waiting for a batch's acknowledgement must sleep in real time between
steps (see `wait_until` in `tests/persistence.rs`) — simulated time alone
can outrun the commit by orders of magnitude.

## Persistence tests

`tests/persistence.rs` (task 2.3): edits survive a drop-and-restart (the
edited cube streams back in from disk), an edited cube is unpinned and
evicted once its batch commits (then reloads with the edit on return), an
edit made while a batch is held mid-save (`Storage::hold_saves`) keeps the
cube pinned until the *second* batch commits, and a failed save
(`Storage::fail_next_save`) keeps cubes pinned and retries next interval.
Each uses a temp world dir, inline generation and a one-tick save interval.

`tests/kill_during_save.rs` proves crash atomicity by re-executing its own
binary as a child (`--exact kill_child_* --nocapture`, gated by
`JBM_KILL_SAVE_CHILD`) and SIGKILLing it mid-save (unix only):

- **Storage level:** the child rewrites the same 64 pattern cubes and
  stamps `last_batch = k` in one transaction, forever. The parent kills it
  after a seeded 20–400 ms, ten rounds on the same directory, and checks
  the database reopens (no stuck lock), all 64 cubes decode, every cube
  shows the same `k`, and `k == last_batch` — no torn batch.
- **Server level:** the child runs a real server (50 ms save interval)
  with one client placing glass blocks along a sequence derived from the
  deterministic terrain (every target provably air and in reach, so every
  placement is accepted). The parent kills it, reopens the world, and
  checks every stored cube loads and the placed blocks are a prefix of
  the sequence.

**Invariants**

- Never a runtime dependency of any shipping crate; other crates pull it
  in only as a dev-dependency.
- May depend on `jbm-core` and `jbm-server`, but never on `jbm-client`
  (client tests add it as their own dev-dependency if needed).
