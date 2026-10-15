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

`GenPool::new(threads, Arc<Generator>, Option<Arc<Storage>>)` — named
workers `jbm-gen-N`, or inline mode with `threads == 0` (requests queue;
`try_recv` generates on the calling thread — the deterministic mode tests
and benches use).

- `request(pos)` / `try_recv() -> Option<(CubePos, Cube)>`. Requests are
  deduplicated while in flight (queued, generating, or done-but-uncollected).
- **Load-or-generate** (task 2.3): when a `Storage` is attached, every
  request reads the cube from disk first and generates only when it was
  never saved. A `Corrupt` record is logged with its position and
  regenerated; the record on disk stays untouched unless the cube is edited
  and saved again.
- **Shutdown:** dropping the pool flags shutdown, closes the queue and
  joins every thread; queued-but-unstarted jobs are discarded, so drop
  returns promptly and never detaches.
- A panic inside generation is caught and forwarded; `try_recv` resumes it
  on the tick thread so it never silently vanishes with a worker.

## Persistence (`storage`, task 2.3)

A world is one directory holding one `redb` database, `world.redb`.
`Storage::open(dir, &BlockRegistry, seed)` creates the directory and
database when missing, and is `Send + Sync` (shared as `Arc<Storage>`:
loads run on the generation threads, writes on the saver thread). A second
open of the same file fails with the typed `StorageError::Locked` — it
never corrupts the database or waits.

**Tables** (one transaction per write):

| table | key → value |
| --- | --- |
| `meta` | `&str → &[u8]`: `format_version` (`u32` = 1) and `seed` (`u64`) |
| `block_states` | on-disk `u32` → canonical `namespace:path[prop=value,…]` (properties in declaration order) |
| `cubes` | 13-byte position key → `[record schema u8 = 1][zstd level-3 frame of the cube codec bytes]` |

The **cube key** is `[kind: u8][x][y][z]`, each coordinate an i32
big-endian with the sign bit flipped, so keys sort by coordinate value.
`kind = 0` is a LOD0 cube; 1..=255 are reserved for LOD cubes (task 2.7).

**State remapping:** runtime `BlockStateId`s are re-assigned every start;
on-disk ids are stable. On open, both remap tables are built in one
transaction: stored names the registry maps are reused, states missing
from the table are appended with fresh ids, and a stored name the registry
lacks fails open with `StorageError::UnknownBlockState` (unknown-block
placeholders come later). A database written by a newer format version
fails with `StorageError::VersionTooNew`.

**Seed:** on reopen the stored seed wins over the configured one (mismatch
logged once); `Storage::seed()` returns it so `Server::try_with_config`
builds the generator from it.

**API:** `load(pos)` (missing → `None`, damaged record → `Corrupt`),
`save_batch(&[(CubePos, &Cube)])` — one transaction committed with redb's
immediate durability, so when it returns the batch is on disk — plus
`cube_count()` and the diagnostic/test hooks `meta`, `cube_positions` and
`save_batch_with_meta` (the crash test stamps `last_batch` atomically with
its cubes).

### Dirty tracking and pinning

Each world cube carries an **edit version** (`u64`), bumped by every
accepted edit, and a **saved version** — the last version known committed
to disk. With persistence on, a cube is **pinned while
`version > saved_version`**: that covers both never-saved edits and cubes
in a save batch the saver has not acknowledged yet. Without persistence
(`world_dir: None`) the pin is the edited set, exactly as before 2.3, and
pinned cubes still grow without bound.

**Eviction** replaces the 1.7 rule: a cube is dropped when no player's
keep set contains it *unless it is pinned*. Unedited cubes regenerate
exactly; edited cubes reload from disk once their edits are committed —
both lossless. Only uncommitted edits pin a cube forever.

### The `jbm-save` thread and its guarantees

One background thread owns the write side. Every `save_interval` the tick
thread takes a **snapshot** — a clone of each pinned cube whose current
version has not already been sent — and hands it off as one batch; the
saver encodes, compresses and commits one transaction per batch, then
replies with `(pos, version)` acknowledgements. The tick thread applies
acks at the start of phase 5: `saved_version = max(saved_version, acked)`,
so an edit made after a snapshot keeps the cube pinned until the next
batch. The tick thread never compresses, encodes or waits on disk I/O
during normal ticks.

**Failure modes:** a failed commit is logged and its cubes stay pinned and
retry next interval — edits are never dropped silently. A dead saver
thread is logged loudly. `Server::flush()` sends everything still dirty
and blocks for its ack (bounded by 30 s, logged on timeout); `Drop for
Server` flushes, closes the channel and joins the saver, so
`ServerHandle::stop` and dropping a handle persist all edits.

**Test hooks** (on `Storage`): `hold_saves`/`release_saves` block the
saver mid-batch to make "edit during save" deterministic;
`fail_next_save` injects one synthetic commit failure. Test accessors on
`Server`: `dirty_count()`, `pending_save_count()`, `saved_batches()`,
`storage()`.

## Server (`server`)

`Server::new()` (default config), `Server::with_config(ServerConfig)`
(panics if the world storage cannot open; `Server::try_with_config` returns
the `StorageError` instead, and opens storage *first* so the stored seed
feeds the generator), `add_connection(ServerLink)`, `tick()` runs exactly
one tick, `update(elapsed) -> u32` feeds the clock and runs every due tick. Read access for tests:
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
| `world_dir: Option<PathBuf>` | `None` | persistence; `None` = today's behavior |
| `save_interval: Duration` | 5 s | snapshot cadence for the save thread |

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
set contains them, except pinned cubes (see "Dirty tracking and pinning"
under Persistence). Generation results never overwrite a cube already in
the world (`entry().or_insert`), so an edit cannot be clobbered by a late
generation result.

## Block edits (task 1.7)

`BreakBlock` and `PlaceBlock` are handled in phase 1, in message order,
against the player's authoritative `PlayerState` at that moment. An edit
is accepted only when:

- the block's cube is in the server world;
- the block center is within `REACH + 1.0` of the player's eye (the
  margin covers latency);
- break: the block is not air;
- place: the target is air, the state is registered and not air, and a
  solid state's unit box does not strictly overlap any joined player.

An accepted edit sets the block, pins its cube and queues a `BlockChange`
for everyone. A rejected one queues a `BlockChange` with the current
state to the sender only, reverting their optimistic edit; rejections
never disconnect. In phase 5, queued changes go out before the streaming
pass and before `Tick`, to each player whose sent set holds the cube, so
an edit reaches every nearby client within the tick it was made. A cube
first sent later in the same tick already contains the edit, so it is
never stale. Test reads: `pinned_count()`, `block(pos)`.

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
detaches. (Dropping the `Server` also runs `flush()`, persisting every
accepted edit, then drops its `GenPool`, which joins the generation
threads, and the `jbm-save` thread.)

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
- `storage/encode_surface_cube` ≈ 13 µs — codec plus zstd level 3.
- `storage/decode_surface_cube` ≈ 207 µs — zstd plus codec rebuild.
- `storage/save_batch_64` ≈ 4.7 ms — one committed 64-cube transaction,
  fsync included.
- `storage/load_cube` ≈ 1 µs — a hot read plus decode. On this machine a
  surface cube's record is 199 bytes against 8.2 kB of heap storage (air
  and uniform-stone records are 18 bytes), and one flush-sized commit of
  1,000 cubes takes ≈ 24 ms.

**Invariants**

- Depends only on `jbm-core` (plus `glam` for the player state's `DVec3`);
  never on `jbm-client`. Both sides communicate exclusively through
  `jbm-core` protocol messages.
- Hot paths (lighting, worldgen inner loops) never call into mods.
