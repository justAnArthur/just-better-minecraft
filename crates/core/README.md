# jbm-core

Shared types for the whole engine: coordinates, block states and the block
registry, paletted cube storage, the shared player physics, and the
protocol messages the client and server exchange.

## Modules

### `coords` — positions across the ±2³¹-cube world

- `CUBE_SIZE = 32`, `CUBE_BITS = 5`, `CUBE_VOLUME = 32768`.
- `CubePos { x, y, z: i32 }` — which cube a block is in; totally ordered by
  `(x, y, z)`.
- `BlockPos { x, y, z: i64 }` — world block position. `i64` because
  `i32::MIN * 32` overflows `i32`.
- `LocalPos` — a block inside a cube, `0..32` on each axis (fields are `u8`).
  `LocalPos::new -> Option<Self>` validates; `new_unchecked` only
  debug-asserts.
- **Index order:** `LocalPos::index() = (y * 32 + z) * 32 + x` — x fastest,
  then z, then y. Every cube array in the engine uses this order;
  `LocalPos::from_index` inverts it.
- Conversions are floor arithmetic (`>> 5` and `& 31` on `i64`), so block -1
  splits to cube -1, local 31: `BlockPos::split`, `CubePos::block`.

### `block` — block states and the registry

- `BlockStateId(pub u32)` interns a `(block, properties)` pair;
  `BlockStateId::AIR` is id `0`.
- `BlockRegistry` is built once at startup (`BlockRegistry::new()` registers
  `base:air` as state 0, not solid, not opaque) and then shared read-only —
  no interior mutability. All lookups (`block`, `default_state`, `state`,
  `info`, `with_property`) are O(1) in the number of states; `info` indexes a
  flat `Vec<StateInfo>` that carries the per-state `solid`/`opaque` flags.
- `register(BlockDef)` assigns every combination of property values a
  consecutive run of state ids (last property varies fastest). The first
  value of each property is the default. Errors: duplicate or malformed name
  (exactly one `:`, non-empty `[a-z0-9_.]` parts), property with no values,
  duplicate property names, total state count over `u32::MAX`.
- `block::builtin` registers the Stage 1 `base:` blocks (stone, dirt,
  grass_block, sand, water, glass, oak_planks, oak_log with `axis=y|x|z`,
  leaves); `builtin::Blocks` hands out their default state ids. Stage 3 moves
  them into a mod.

### `cube` — paletted, bit-packed storage

- A `Cube` is 32³ block states as a palette (`Vec<BlockStateId>`, unique
  entries) plus palette indices bit-packed into `Vec<u64>` words.
- **Packing rules:** entries never straddle a word —
  `entries_per_word = 64 / bits` with the tail bits of each word unused;
  `bits = max(1, ceil(log2(palette_len)))`, at most 15 (a palette can never
  exceed the 32768 cells). A uniform cube (single palette entry) has **no
  index array at all**; `bits_per_entry()` is 0 and `heap_bytes()` is a few
  bytes.
- **Growth:** setting a state absent from the palette appends it; when the
  palette outgrows the width, all entries repack at the new width. Setting a
  block to the state it already has does not touch the palette. Palette
  lookup is a linear scan up to 16 entries, then a `HashMap` reverse map with
  a local FxHash-style hasher (exercised by `cube/palette_growth_4096`).
- **`compact()` contract:** drops unused palette entries, remaps indices,
  shrinks to the minimal width and returns to no array when one state
  remains. It is *not* called automatically by `set` — callers run it after
  bulk edits or before saving.
- `iter()` yields all `CUBE_VOLUME` states in index order (one cached word at
  a time); `from_fn` builds a cube from per-position states in index order
  and collapses to uniform on a single distinct state.
- `PartialEq`/`Eq` compare content (every block), not storage: palette
  order, width and stale palette entries don't matter. `Debug` prints the
  palette and width only.
- Invariants (unique palette entries; array empty iff uniform; every index
  `< palette.len()`) are asserted by `#[cfg(test)] Cube::check_invariants`,
  which tests call after operations.

### `physics` — the one shared player step (task 1.6)

Units are blocks and seconds, f64. The step is fixed:
`TICK_DT = 1/20`, one tick at a time, no variable dt. The client's
predictor and the authoritative server call the same pure
`physics::step(state, input, world) -> PlayerState`, so identical worlds
give bit-identical results.

**Constants** (all public, all documented here):

| constant | value |
| --- | --- |
| player box | 0.6 × 1.8 × 0.6, feet at the bottom center |
| `EYE_HEIGHT` | 1.62 |
| walk speed | 4.3 |
| sprint speed | 5.6 |
| gravity | 32 blocks/s² |
| jump velocity | 9.0 (a jump clears just over 1.25 blocks) |
| terminal fall speed | 78 |
| fly speed | 11 |
| fly sprint speed | 22 |

**Types:** `MoveInput` (forward/strafe/up as i8, jump, sprint, fly_toggle,
yaw — the same yaw convention as the camera: 0 looks toward −Z, positive
turns toward +X), `PlayerState` (feet position, velocity, on_ground,
flying; `aabb()` and `eye()`), `trait BlockShapes` (`solid(pos) ->
Option<bool>`, `None` = block not loaded) and `SolidTable` (a flat
`Vec<bool>` by `BlockStateId` built from a `BlockRegistry`, so the hot
path never touches the registry — and the one place a state's collision
shape is decided, a full unit box when solid for now).

**Step order:**

1. `fly_toggle` flips `flying`. The server trusts the flag — Stage 1 is
   creative-only; permission checks come with anti-cheat.
2. Wish velocity: horizontal direction from forward/strafe rotated by
   yaw, normalized so diagonals are not faster, times the speed
   (fly/sprint variants above). Velocity is set directly — no
   acceleration or friction yet, air control equals ground control. Walk
   mode jumps from the ground and subtracts gravity, clamped to terminal
   speed; the vertical displacement is gravity's exact integral across
   the tick (the average of the pre/post-gravity velocities), so a jump's
   apex is v²/2g = 1.2656 blocks. Fly mode sets `velocity.y = up × fly
   speed`, no gravity.
3. Swept collision per axis, **Y then X then Z**: clip the axis
   displacement against every solid block the moving box would cross,
   then move; a blocked axis zeroes that velocity component.
   `on_ground` is true when the Y sweep was blocked while moving down.

**Boundary rule:** touching faces do not overlap. The blocks a box
strictly overlaps on an axis are exactly `floor(min) ..= ceil(max) - 1`;
a box resting exactly on `y = 64.0` spans Y blocks 64..=65 and never
collides with block 63 during the X/Z sweeps, and a falling box stops
with its feet at exactly `64.0` (bitwise — tests assert it). There is
deliberately **no epsilon**: one that let the box sink into a block would
make client and server disagree about resting positions.

**Unloaded rule:** if any block the swept box touches this tick is not
loaded (`None`), the step returns the state unchanged — no movement, no
gravity, no fly toggle. A player freezes at the edge of loaded terrain
instead of falling into it; client and server both apply this, keeping
them in agreement while cubes stream in.

**Determinism:** only `+ - * / floor ceil min max sqrt` on f64, so steps
are bit-exact across platforms. The one exception is `yaw.sin_cos()`,
computed once per step — the only platform-dependent operation; server
correction absorbs any drift.

## Protocol and links

`protocol` holds the message enums the client and server exchange
(`PROTOCOL_VERSION = 3`): `ClientMsg` (`Hello`, `PlayerInput { seq,
input, pitch }`, `Ping`, `Disconnect`) and `ServerMsg` (`Welcome` with
the spawn feet position — the initial state: velocity 0, not on ground,
walking — `PlayerState { ack_seq, state }` sent before `Tick` in ticks
where the server processed at least one input, `Rejected`, `Tick`,
`Pong`, `Disconnected`, `CubeData { pos, cube }`, `UnloadCube { pos }`).
`PlayerInput.seq` starts at 1 and increments per client tick; the server
ignores seqs ≤ the last processed one. Pitch is display-only; physics
ignores it. Messages are plain owned data — no `Arc`, no references, no
trait objects — all derive `Debug, Clone, PartialEq`, and stay
serializable for a future network transport (no serde yet).

`link` is the transport boundary: `trait Link<Out, In>: Send` with
non-blocking `try_recv` (returns `Ok(None)` when empty) and
`send`/`try_recv` returning `LinkClosed` only once the other end is gone
and the queue has drained. `ClientLink`/`ServerLink` are boxed trait
aliases for the two ends, and `in_process_pair()` builds a connected pair
on `std::sync::mpsc`. A network socket can implement `Link` later without
either side changing.

## Benchmarks

Criterion benches live in `benches/`, registered in `Cargo.toml`:

```toml
[[bench]]
name = "core_bench"
harness = false
```

To add a case, add a function to `benches/core_bench.rs` inside a
`benchmark_group` named for the area, with the case as the function name, so
reports read `<area>/<case>` (e.g. `sample/fill_32cubed`, `cube/get_all_16_states`).
Current cases include `physics/step_walk_flat` (median ≈ 12 ns on this
Mac against a trivial flat-floor world) and `physics/step_fall_terminal`
(≈ 13 ns, fall at terminal speed over loaded air).
`criterion` is a dev-dependency via `[workspace.dependencies]` only — never a
runtime dep.

```bash
cargo bench -p jbm-core -- --test   # run each bench once (what CI calls)
cargo bench -p jbm-core             # full criterion run with numbers
```

**Invariants**

- No dependencies on any other jbm crate; everything else may depend on this.
- Contains types only; no I/O, no threads, no engine logic.
- Never depends on `jbm-client` or `jbm-server` (both depend on *this* crate).
