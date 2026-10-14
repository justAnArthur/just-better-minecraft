# jbm-server

The authoritative simulation: world generation, lighting, storage, and the
server side of the protocol.

## Tick clock (`tick`)

- `TICKS_PER_SECOND: u32 = 20`, `TICK_PERIOD` (the one canonical tick
  period — never re-derive it), `MAX_CATCH_UP_TICKS: u32 = 10`.
- `TickClock::new()` / `Default`, `tick_count()`, `dropped_ticks()`,
  `until_next_tick()`, and `advance(elapsed) -> u32`, which accumulates
  elapsed time and returns the whole ticks due, capped at
  `MAX_CATCH_UP_TICKS`. Excess whole ticks are dropped (no spiral of
  death) and counted in `dropped_ticks()`; only the sub-tick remainder is
  kept. `until_next_tick()` is how long the real-time loop sleeps.
  Pure logic: no threads, no sleeping, no real clock reads — the caller
  feeds in time.

## Server (`server`)

`Server::new()`, `add_connection(ServerLink)`, `tick()` runs exactly one
tick, `update(elapsed) -> u32` feeds the clock and runs every due tick.
Read access for tests: `tick_count()`, `dropped_ticks()`,
`players() -> impl Iterator<Item = PlayerView>` (id, name, latest input,
input count).

A connection is pending until a valid `Hello` (matching
`jbm-core`'s `PROTOCOL_VERSION`), then joined with a fresh `PlayerId`
starting at 1. Wrong version → `Rejected` + drop; a non-`Hello` while
pending → `Disconnected` + drop; `Ping` → `Pong`; `PlayerInput` → stored
as latest input and counted; `Disconnect` or a closed link → the player is
removed. Sends to closed links remove the player and never panic.

**Tick phases** (fixed order, per `docs/design.md` "Simulation"):

1. Messages — drain every connection, join order, arrival order, capped
   at 256 messages per connection per tick (leftovers wait).
2. Scheduled block ticks — stub until Stage 2.
3. Block update queue — stub until Stage 2.
4. ECS systems — stub until bevy_ecs (task 1.5+).
5. Light and outgoing changes — send `Tick { tick }` to every joined
   player.

## Run loop (`run`)

`run::spawn(Server) -> ServerHandle` starts one thread named
`"jbm-server"` that loops: measure elapsed time with `Instant`, call
`server.update(elapsed)`, sleep `until_next_tick()`. New connections
arrive via `ServerHandle::connect(&self, ServerLink)` (an mpsc the loop
drains at the start of each tick). `ServerHandle::stop(self) -> Server`
sets a stop flag, joins the thread and returns the `Server` for
inspection; a thread panic propagates through `stop` instead of being
swallowed. Dropping a handle also stops and joins the thread — it never
detaches.

**Threading model:** one server thread owns the `Server`; tests talk to
it only through links (or by `stop()` returning the `Server`). The tick
loop itself is single-threaded; region-parallel tick work arrives in a
later stage.

## Benchmarks

Criterion benches live in `benches/`, registered in `Cargo.toml`:

```toml
[[bench]]
name = "tick_bench"
harness = false
```

To add a case, add a function to `benches/tick_bench.rs` inside a
`benchmark_group` named for the area, with the case as the function name,
so reports read `<area>/<case>` (e.g. `tick_clock/advance_1000`).
`criterion` is a dev-dependency via `[workspace.dependencies]` only —
never a runtime dep.

```bash
cargo bench -p jbm-server -- --test   # run each bench once (what CI calls)
cargo bench -p jbm-server             # full criterion run with numbers
```

Current cases: `tick_clock/advance_1000` and `server/tick_8_clients`
(one `Server::tick()` with 8 joined in-process clients, each sending one
`PlayerInput`; replies drained outside the timed section).

**Invariants**

- Depends only on `jbm-core`; never on `jbm-client`. Both sides
  communicate exclusively through `jbm-core` protocol messages.
- Hot paths (lighting, worldgen inner loops) never call into mods.
