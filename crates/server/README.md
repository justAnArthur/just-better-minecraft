# jbm-server

The authoritative simulation: world generation, lighting, storage, and the
server side of the protocol.

## Terrain (`terrain`)

`Generator::new(seed, Blocks)` produces cubes by evaluating the compiled
shipped worldgen graph (below); `generate(CubePos) -> Cube` is a pure
function of `(seed, pos)`, and `surface_height(x, z) -> i64` is public (the
spawn point is `[0.5, h(0,0) + 2.0, 0.5]`). `Generator::from_graph(seed,
Blocks, &CompiledGraph)` runs any graph instead — for tests now, mods from
Stage 3. `MIN_HEIGHT = 36` and `MAX_HEIGHT = 92` stay the documented bounds
of the shipped graph.

- **Layering per column with surface height `h`:** grass_block at `h`, dirt
  from `h-3` to `h-1`, stone below; sand replaces grass and dirt where
  `h ≤ 61`. No water, trees or caves yet.
- **Fast paths:** interval bounds over the cube decide uniform air (density
  max < 0) and uniform stone (density min ≥ 0 over the cube extended 4 cells
  up, so every surface run is provably saturated). Mixed cubes evaluate per
  column in one downward pass and build in one `Cube::from_fn` sweep —
  never per-block `set`. A bounded per-column cache (64 cube columns) keeps
  the evaluated surface band so vertically adjacent cubes cost almost
  nothing; it is a pure-function cache and cannot change answers.
- The task 1.5 noise terrain lives on as `worldgen::reference`, kept only
  for the equivalence tests and the comparison bench.

## Worldgen graph (`worldgen`)

Terrain shape is data: a JSON graph of density functions, compiled once by
`compile(&Graph) -> CompiledGraph` into a flat, topologically ordered
instruction list over a register array — no recursion, no allocation per
sample. A cell is **solid** when `final_density ≥ 0`. The shipped graph is
`assets/worldgen/overworld.json` (`worldgen::OVERWORLD_JSON`, also compiled
once per process by `worldgen::overworld()`); it re-expresses the task 1.5
terrain bit-identically.

**Format** (`Graph::from_json`, version 1): a `nodes` map of named,
shareable subexpressions (compiled once per sample however often they are
referenced) and an `outputs` map that must contain `final_density`. An
operand is a JSON number (constant), `{"ref": "name"}` (named node), or an
inline node object. Every load error is a typed `GraphError` naming the
offending node path (`nodes.hills.a`, `outputs.final_density`): unknown
type, missing field, dangling ref, cycle, bad spline, unknown version,
missing output. Loading never panics on bad input.

**Node table** (all `f64`):

| node | fields | notes |
| --- | --- | --- |
| `x`, `y`, `z` | — | block coordinates of the sample |
| `add`, `sub`, `mul`, `min`, `max` | `a`, `b` | |
| `neg`, `abs`, `floor` | `a` | |
| `clamp` | `input`, `min`, `max` | `f64::clamp` semantics; `min > max` is a compile error when constant |
| `noise2` | `salt`, `x_scale`, `z_scale` | the reference 2D gradient noise at `(x·x_scale, z·z_scale)`, seeded `world_seed ^ salt`; range ±√2/2 |
| `fbm2` | `salt`, `scale`, `octaves` | the reference fractal sum at `(x·scale, z·scale)`, normalized; range ±√2/2 |
| `spline` | `input`, `points` | piecewise linear over `[[x, y], …]`, clamped at both ends; points strictly increasing in `x`, at least two |

Salts are full `u64`s written as hex strings (`"0x0D1B54A32D168ED5"`) so
values above 2⁵³ round-trip through JSON exactly.

**Determinism and the association rule.** Nodes use only IEEE-754
`+ - * /`, `floor` and comparisons, plus integer hashing into the fixed
256-entry gradient table — every one of those is bit-exact on every
platform, so the same `(seed, graph)` gives identical terrain everywhere.
No `sin`/`cos`/`exp`/`powf` anywhere. Float arithmetic is commutative but
**not associative**: the expression tree's shape fixes the bits, so keep
operand nesting exactly as written when editing `overworld.json` — it
computes `(40 · hilliness) · rolling`, not `40 · (hilliness · rolling)`.
The equivalence tests catch any drift.

**Compilation stages.** `compile` folds constants, then splits
instructions by dimension analysis: constants run once, column-stage (xz)
instructions once per `(x, z)` column, cell-stage (xyz) instructions once
per block. The shipped graph's only cell-stage work is `y` and `- y`, so 3D
evaluation stays near heightmap cost. `CompiledGraph::bounds(min, max)`
gives sound interval bounds over an inclusive box (noise contributes its
documented ±√2/2); the cube fast paths above are built on it.

**Surface stage** (code, not graph, for now): for a solid cell at `y`, let
`d` be the number of consecutive solid cells directly above it, capped at
4. If `d ≤ 3` and the surface `y + d` is at or below 61 (`SAND_LEVEL`):
sand; else if `d = 0`: grass; else if `d ≤ 3`: dirt; else stone. Computing
`d` for the top rows evaluates up to 4 cells above the cube — never a
neighbour cube. For a pure heightmap density this is exactly the old
`column_block`; the equivalence argument is a comment on
`terrain::surface_block`.

**Per-column cache.** Rows the interval bounds cannot decide form a band
(the shipped graph's is `y ∈ (36, 93)`); the generator caches one solidity
bit per band row per column — today's heightmap cache, generalized — so
vertically adjacent cubes reuse it instead of re-running noise. Graphs
whose band cannot be proven fall back to caching live column-stage values
and evaluating rows on demand.

**`sample_grid` layout** (for later LOD tasks): `CompiledGraph::
sample_grid(seed, origin, step, out)` fills a 32³ grid of densities at
`origin + step·(i, j, k)` over `(x = i, y = j, z = k)`, laid out as
`(k·32 + j)·32 + i`, `step` a power of two in `1..=64`. The column stage
runs once per grid column; step 1 at a cube origin gives exactly the
densities `generate`'s solidity decisions use.

**Adding a node:** add the variant to `Node` (with its serde shape), parse
it in `parse_node` (allow-list its fields), add an `Op` variant plus its
`operands`/`remap` arms, evaluate it in `eval_op`, give it an interval rule
in `interval_op` (sound, not tight), extend the error tests, and add its
row to the table above. Anything beyond `+ - * / floor` and comparisons
breaks the determinism rule — don't.

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
| `players` | player name → `[schema u8 = 1][x y z: f64][yaw pitch: f32][flying u8]` (34 bytes, little-endian) |

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

### Players

A returning player (matched by the `Hello` name) joins where they left:
`Welcome` carries their saved feet position, look `yaw`/`pitch` and
`flying` (protocol v5), and their server state starts there at rest. A new
name, or a record that fails to decode (`StorageError::CorruptPlayer`,
logged), gets the world spawn. The server keeps each player's look from
their newest accepted `PlayerInput`. Every snapshot also carries the
records of joined players and of players who left since the last one
(`remove_connection` captures them, so a disconnect never loses the last
position), written only when they differ from what was last committed and
in the same transaction as that batch's cubes. `flush` saves them too and
waits until every batch sent has been answered. Nothing a player saw is
lost to timing: a disconnecting player's queued inputs are stepped before
their record is taken, and on shutdown (`Drop`, with persistence on) the
server first handles every message already in the links — inputs and
edits — and steps all queued inputs, then flushes.

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
| `lod_levels: u8` | 4 | top LOD level kept; 0 disables the pyramid |

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

## LOD ring streaming (phase 5, task 2.6)

Per joined player and per level *n* in `1..=lod_levels`, the server keeps
a **level center** (the level-*n* cube containing the player), a **sent
set** and the walk state, and streams a full disk of
`lod_view_radius`/`lod_vertical_radius` level-*n* cubes around the
center — full disks, not annuli: the client needs the coarse cube under
a finer one while the finer one loads, and to fill octants the finer
level does not cover yet. That costs about a quarter more cubes per
level than an annulus would (`disk(r) / (disk(r) - disk(r/2))`; at
r = 8, v = 3: 1407 disk vs 1064 annulus cubes, +32%).

- **Order.** One merged walk over `(level, offset)`, precomputed once
  from the config: sorted by the offset's squared distance in level
  units, then level **descending** (coarse first at equal distance),
  then `(x, y, z)`.
- **Sending.** A ready cube is sent as `LodCubeData` and recorded in
  the sent set. A not-ready cube is requested (while
  `in_flight < lod_max_in_flight`) and **skipped** — the walk does not
  stop at gaps, because the client's coverage rule makes any arrival
  order hole-free. At most `lod_send_budget_per_tick` LOD messages per
  player per tick, re-sends included.
- **Re-sends.** Once per tick, before the per-player pass, the pyramid's
  changed set is drained (`take_changed`); every changed cube a player
  has in its sent set is queued. Queued re-sends go first within the
  budget and the queue persists across ticks, so an edit's LOD change
  reaches every holder within one tick whenever the budget allows, and
  nothing starves under saturation.
- **Unloading.** When a level's center moves, `UnloadLodCube` goes out
  for every sent cube that left that level's keep set (the disk shape at
  +1 in both radii — the same hysteresis as LOD0), and the walk cursor
  resets.
- **Cost contract.** A settled player whose rings are complete costs a
  few center comparisons per level per tick — the cursor sits at the end
  of the walk, so nothing is scanned. While cubes are missing the
  per-tick scan is bounded by the skip window (`LOD_SKIP_WINDOW` = 512
  entries past the cursor) plus the send budget; the cursor only
  advances over sent entries, so a permanent gap in the pyramid costs a
  bounded look-ahead every tick, never a full rescan (a level-center
  move resets it, like LOD0).
- **Off switches.** No LOD messages at all when `streaming` is false,
  `lod_levels == 0`, or the connection has not joined.

LOD cubes are never evicted from the pyramid (2.7's known issue).
`Server::lod_ring_pending(id)` reports how many desired ring cubes are
unsent (test/fly-test hook).

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

## LOD pyramid (`lod`, task 2.4)

`LodPyramid` keeps levels `1..=lod_levels`, each a map of `LodCube { cube:
Cube, known: u8 }`, built from the real world by the core's
`jbm_core::lod` downsampling rule. A level-*n* cube covers the same
volume as a 2×2×2 group of level-(*n*−1) cubes; nothing is streamed or
rendered yet (2.6), and LOD from worldgen sampling is 2.7.

- **`known`** is an octant bitmask: bit *o* is set once that octant's
  child has been downsampled into the cube at least once. Octants never
  written read as air, so `known` says which octants hold real data and
  **nothing else may be assumed about the unknown ones** — their air is
  placeholder, not measured emptiness. An inserted cube marks its whole
  ancestor chain known; a partially known child propagates its
  placeholder air for its own unknown octants (a deliberate
  simplification for 2.7 to revisit).
- **Ingest:** `insert_lod0(pos, &Cube) -> LodChanges` downsamples a cube
  entering the world into its level-1 parent (created as uniform air if
  absent) and propagates the changed region up until a level reports no
  change. The server calls it in the streaming pass only when
  `entry().or_insert` actually inserts — a stale generated copy dropped
  because an edited cube is present never touches the pyramid.
- **Edits:** `edit_lod0(BlockPos, &Cube) -> LodChanges` recomputes the
  one affected cell per level (`downsample_region` on a single cell) and
  stops at the first unchanged level. The server calls it from
  `apply_edit` after setting the block.
- **`LodChanges`** lists the `(level, CubePos)` cubes whose content
  changed, in level order — 2.6 will stream exactly these.
- **Eviction never touches the pyramid.** LOD cubes outlive their LOD0
  children, so player builds stay visible from far away; re-inserting a
  regenerated or reloaded cube reports no changes. The cost: the pyramid
  grows with the explored area until 2.6 (clipmap rings) and 2.7 (stored
  LOD) give it bounds.
- `LodPyramid::rebuild(max_levels, lod0)` rebuilds every level bottom-up
  from scratch — the reference the incremental paths are tested against
  (they must match cell for cell and `known` for `known`). Accessors:
  `get(level, pos)`, `len(level)`, `total_cubes()`, `heap_bytes()`.

`lod_levels: u8` (default 4 = LOD1–LOD4) tops the pyramid; **0 disables
it entirely** — no allocation, no per-cube or per-edit work
(`protocol_only()` uses 0). `insert_lod0` runs on the tick thread for
every collected cube; its cost is bounded by the `lod/insert_lod0/*`
and `lod/edit` benches in `benches/lod_bench.rs`. Server read accessors:
`lod_cube(level, pos)`, `lod_known(level, pos)`, `lod_cube_count()`,
`lod_heap_bytes()`.

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
  per-column cache warm).
- `worldgen/air_cube` — a cube far above the surface (fast path).
- `worldgen/reference_surface_cube` — the task 1.5 generator on the same
  cube; the graph path must stay within 2× of it (task 2.1 budget).
- `worldgen/compile_overworld` — parsing and compiling the shipped graph.
- `worldgen/sample_grid/{1,2,4,8,32}` — one 32³ density grid near the
  surface per LOD step; step 32 spans 1,024 blocks (every column a cache
  miss, the worst case).
- `storage/encode_surface_cube` ≈ 9 µs — codec plus zstd level 3 (bulk
  frames, which record their content size).
- `storage/decode_surface_cube` ≈ 30 µs — exact-size zstd decompress plus
  codec validation.
- `storage/save_batch_64` ≈ 4.0 ms — one committed 64-cube transaction,
  fsync included.
- `storage/load_cube` ≈ 24 µs — a hot read plus decode of a mixed cube. On this machine a
  surface cube's record is 199 bytes against 8.2 kB of heap storage (air
  and uniform-stone records are 18 bytes), and one flush-sized commit of
  1,000 cubes takes ≈ 24 ms.
- `lod/insert_lod0/{surface,air}` (task 2.4, `lod_bench.rs`) — a
  generated surface cube (mixed, asserted) or an air cube inserted into
  an existing 4-level pyramid.
- `lod/edit` — one edit propagated through all levels.
- `lod/stream_settled` (task 2.6, `lod_stream_bench.rs`) — one streaming
  tick of a player whose rings have settled at game scale: the idle
  floor.
- `lod/stream_moving` — a player sprint-flying across level-1 boundaries
  every second cube: center moves, cursor resets and keep-set unloads.

**Invariants**

- Depends only on `jbm-core` (plus `glam` for the player state's `DVec3`,
  and `serde`/`serde_json` for the worldgen graph format); never on
  `jbm-client`. Both sides communicate exclusively through `jbm-core`
  protocol messages.
- Hot paths (lighting, worldgen inner loops) never call into mods.
