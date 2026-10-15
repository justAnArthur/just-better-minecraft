# jbm-server

The authoritative simulation: world generation, lighting, storage, and the
server side of the protocol.

## Terrain (`terrain`)

Temporary noise terrain (task 2.1 replaces it with the declarative density
graph). `Generator::new(seed, Blocks)` produces cubes;
`generate(CubePos) -> Cube` is a pure function of `(seed, pos)`, and
`surface_height(x, z) -> i64` is public (the spawn point is
`[0.5, h(0,0) + 2.0, 0.5]`).

- **Layering per column with surface height `h`:** grass_block at `h`, dirt
  from `h-3` to `h-1`, stone below; sand replaces grass and dirt where
  `h ≤ 61`. No water, trees or caves yet.
- **Height range:** `h = 64 + clamp(40 · hilliness · rolling, ±28)`, so
  heights provably stay in `[36, 92]`. `rolling` is a 5-octave fractal
  (±28 hills); `hilliness` is a very low frequency mask that leaves some
  areas flat and others mountainous.
- **Determinism rule:** only IEEE-754 `+ - * floor` on `f64` plus integer
  hashing into a fixed 256-entry gradient table — no `sin`/`cos`/`exp`,
  which are not bit-exact across platforms. The same seed gives identical
  terrain everywhere, well past |x| = 10⁹.
- **Fast paths:** a cube above the column-max height of its 32² area is a
  uniform air cube; one whose top is below `min height - 4` is uniform
  stone. The 32² heightmap of a column is computed once (bounded cache of
  64 columns) and shared by all `cy` values. Mixed cubes are built in one
  `Cube::from_fn` pass — never per-block `set`.

## Generation pool (`gen_pool`)

`GenPool::new(threads, Arc<Generator>)` — named workers `jbm-gen-N`, or
inline mode with `threads == 0` (requests queue; `try_recv` generates on
the calling thread — the deterministic mode tests and benches use).

- `request(pos)` / `try_recv() -> Option<(CubePos, Cube)>`. Requests are
  deduplicated while in flight (queued, generating, or done-but-uncollected).
- **Shutdown:** dropping the pool flags shutdown, closes the queue and
  joins every thread; queued-but-unstarted jobs are discarded, so drop
  returns promptly and never detaches.
- A panic inside generation is caught and forwarded; `try_recv` resumes it
  on the tick thread so it never silently vanishes with a worker.

## Server (`server`)

`Server::new()` (default config), `Server::with_config(ServerConfig)`,
`add_connection(ServerLink)`, `tick()` runs exactly one tick, `update(elapsed)
-> u32` feeds the clock and runs every due tick. Read access for tests:
`tick_count()`, `dropped_ticks()`, `players()`, `world_cube_count()`,
`in_flight()`, `sent_count(PlayerId)`.

`ServerConfig` (all fields public):

| field | default | meaning |
| --- | --- | --- |
| `seed: u64` | `DEFAULT_SEED` | terrain seed |
| `streaming: bool` | `true` | cube streaming on/off |
| `view_radius: u32` | 8 | horizontal radius, cubes |
| `vertical_radius: u32` | 4 | vertical radius, cubes |
| `send_budget_per_tick: usize` | 64 | CubeData per player per tick |
| `gen_threads: usize` | `available_parallelism() - 2`, min 1 | 0 = inline mode |
| `gen_budget_per_tick: usize` | 64 | inline generation per tick |
| `max_in_flight: usize` | 512 | queued + generating + uncollected |

`ServerConfig::protocol_only()` turns streaming off with no generation
threads — what exact-message-sequence tests and benches use.

A connection is pending until a valid `Hello` (matching `PROTOCOL_VERSION`),
then joined with a fresh `PlayerId` starting at 1. `Welcome.spawn` is the
terrain spawn over column (0, 0) — the player's initial physics state:
velocity 0, not on ground, walking. Wrong version → `Rejected` + drop; a
non-`Hello` while pending → `Disconnected` + drop; `Ping` → `Pong`;
`PlayerInput` → queued for phase 4; `Disconnect` or a closed link → the
player is removed. Sends to closed links remove the player and never
panic.

## Player physics and input processing (task 1.6)

Each joined player holds the authoritative `PlayerState` (from spawn), a
queue of received inputs and the last processed seq. Phase 1 pushes
`PlayerInput`s onto the queue: an input whose seq is ≤ the last processed
one is ignored; the queue is capped at **20**, dropping the oldest on
overflow. Phase 4 steps, per player in join order, up to **4** queued
inputs per tick through the shared `jbm_core::physics::step`, against a
`BlockShapes` view of the server's world (a missing cube reads as `None`
— one cube lookup per block, no clones; the flat `SolidTable` decides
shapes). Physics is **input-driven**: a tick with no queued input does
not step the player, so a stalled client freezes in place. Anti-cheat
and input timeouts are a later task; the caps above are the only
protection for now.

Phase 5 sends `PlayerState { ack_seq, state }` — the state after
processing input `ack_seq` — before the `Tick`, only in ticks where at
least one input was processed. The streaming center is the cube
containing the authoritative position. `PlayerView` (what `players()`
returns) exposes `state`, `last_seq` and `input_count`.

**Tick phases** (fixed order, per `docs/design.md` "Simulation"):

1. Messages — drain every connection, join order, arrival order, capped
   at 256 messages per connection per tick (leftovers wait).
2. Scheduled block ticks — stub until Stage 2.
3. Block update queue — stub until Stage 2.
4. ECS systems — player physics, as described above.
5. Light and outgoing changes — the streaming pass (below), then each
   player's `PlayerState` (if any input was processed), then `Tick` for
   every joined player (so cubes and state always precede their tick).

## Cube streaming (phase 5)

For a player whose center cube is `c` (the cube containing their
authoritative `PlayerState` position):

- **Desired set:** every cube with `dx² + dz² ≤ R²` and `|dy| ≤ V`. The
  offset list is precomputed once per config, sorted nearest-first by
  `dx² + dy² + dz²` with ties broken by `(x, y, z)`.
- **Keep set:** the same shape with `R + 1`, `V + 1`. A cube is unloaded
  only when it leaves the keep set — the hysteresis band that stops a
  player standing on a border from making cubes flap.

Each tick: collect finished generation results (inline mode generates up
to `gen_budget_per_tick` first), then per player, in join order:

1. If the center cube moved: send `UnloadCube` for every sent cube outside
   the keep set, and reset the send cursor.
2. Walk the desired offsets nearest-first from the cursor: send `CubeData`
   for generated, unsent cubes up to `send_budget_per_tick`; a cube is
   never sent twice while it stays loaded. The walk pauses at the first
   not-yet-generated cube (order stays strictly nearest-first).
3. Request generation for missing cubes (also ahead of the cursor, keeping
   the pool fed) while `in_flight() < max_in_flight`.

**Idle cost:** with an unchanged center and everything sent, one player
costs a few hash lookups per tick — the cursor sits at the end of the
offset list, so nothing is scanned. Unload scans and world eviction run
only when something changed (a player moved or left, results arrived).

**Eviction:** cubes are dropped from server memory when no player's keep
set contains them. Regeneration is exact, so this is lossless — for now.
**Task 1.7 must pin edited cubes (or they will be lost on walk-away) until
persistence lands in 2.3.**

**Disconnect:** the player's sent-set is freed and its cubes are evicted
on the next tick unless another player keeps them.

## Run loop (`run`)

`run::spawn(Server) -> ServerHandle` starts one thread named
`"jbm-server"` that loops: measure elapsed time with `Instant`, call
`server.update(elapsed)`, sleep `until_next_tick()`. New connections
arrive via `ServerHandle::connect(&self, ServerLink)`. `ServerHandle::stop(self)
-> Server` sets a stop flag, joins the thread and returns the `Server` for
inspection; a thread panic propagates through `stop` instead of being
swallowed. Dropping a handle also stops and joins the thread — it never
detaches. (Dropping the `Server` also drops its `GenPool`, which joins the
generation threads.)

**Threading model:** one server thread owns the `Server`; tests talk to
it only through links (or by `stop()` returning the `Server`). The tick
loop itself is single-threaded; generation runs on the `GenPool` threads,
never on the tick thread (inline test mode excepted).

## Benchmarks

Criterion benches live in `benches/`, registered in `Cargo.toml`:

```toml
[[bench]]
name = "tick_bench"
harness = false
```

To add a case, add a function inside a `benchmark_group` named for the
area, with the case as the function name, so reports read `<area>/<case>`
(e.g. `tick_clock/advance_1000`). `criterion` is a dev-dependency via
`[workspace.dependencies]` only — never a runtime dep.

```bash
cargo bench -p jbm-server -- --test   # run each bench once (what CI calls)
cargo bench -p jbm-server             # full criterion run with numbers
```

Current cases:

- `tick_clock/advance_1000`
- `server/tick_8_clients` — one tick, 8 joined clients, `protocol_only()`
  so it stays comparable across stages.
- `server/tick_streaming_idle` — one tick, one player fully loaded at
  R = 8 (1773 cubes), inline mode, nothing left to send: the idle floor.
- `worldgen/heightmap_32x32` — a 32×32 patch of surface heights.
- `worldgen/surface_cube` — a cube intersecting the surface (mixed path,
  heightmap cached).
- `worldgen/air_cube` — a cube far above the surface (fast path).

**Invariants**

- Depends only on `jbm-core` (plus `glam` for the player state's `DVec3`);
  never on `jbm-client`. Both sides communicate exclusively through
  `jbm-core` protocol messages.
- Hot paths (lighting, worldgen inner loops) never call into mods.
