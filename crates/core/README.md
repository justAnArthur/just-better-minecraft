# jbm-core

Shared types for the whole engine: coordinates, block states and the block
registry, paletted cube storage, and the protocol messages the client and
server exchange.

## The `cube16` feature

`cube16` is a compile-time benchmark feature (task 1.8) that shrinks cubes
from 32³ to 16³: `CUBE_BITS` becomes 4 and `CUBE_SIZE`, `CUBE_VOLUME` and
every mask/shift in `coords` derive from it. Everything in this crate and
in the client mesher derives from those constants, so both compile and their
unit tests pass at either size. Limits:

- **The server is not supported under `cube16`.** It compiles, but its
  behavior (streaming radii, terrain assumptions, tests) is only defined at
  32³. Do not run the game binary or the end-to-end tests with it.
- CI never enables it. Only the task 1.8 bench
  (`cargo bench -p jbm-game --bench cube_size`) and example
  (`cargo run -p jbm-game --release --example cube_size_metrics`) use it,
  via the passthrough features on `jbm-client` and `jbm-game`.
- Client goldens and hand-placed scene tests are 32³-only and cfg-gated off.
- Changing the default cube size is an orchestrator decision, recorded in
  docs/progress.md; this feature only measures the trade-off.

## Modules

### `coords` — positions across the ±2³¹-cube world

- `CUBE_BITS = 5` (4 under `cube16`), `CUBE_SIZE = 1 << CUBE_BITS`,
  `CUBE_VOLUME = CUBE_SIZE³`, `CUBE_MASK = CUBE_SIZE - 1`.
- `CubePos { x, y, z: i32 }` — which cube a block is in; totally ordered by
  `(x, y, z)`.
- `BlockPos { x, y, z: i64 }` — world block position. `i64` because
  `i32::MIN * CUBE_SIZE` overflows `i32`.
- `LocalPos` — a block inside a cube, `0..CUBE_SIZE` on each axis (fields
  are `u8`). `LocalPos::new -> Option<Self>` validates; `new_unchecked` only
  debug-asserts.
- **Index order:** `LocalPos::index() = (y * CUBE_SIZE + z) * CUBE_SIZE + x`
  — x fastest, then z, then y. Every cube array in the engine uses this
  order; `LocalPos::from_index` inverts it.
- Conversions are floor arithmetic (`>> CUBE_BITS` and `& CUBE_MASK` on
  `i64`), so block -1 splits to cube -1, local `CUBE_SIZE - 1`:
  `BlockPos::split`, `CubePos::block`.

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

- A `Cube` is `CUBE_SIZE³` block states as a palette (`Vec<BlockStateId>`,
  unique entries) plus palette indices bit-packed into `Vec<u64>` words.
- **Packing rules:** entries never straddle a word —
  `entries_per_word = 64 / bits` with the tail bits of each word unused;
  `bits = max(1, ceil(log2(palette_len)))`, at most `ceil(log2(CUBE_VOLUME))`
  (a palette can never exceed the cube's cells). A uniform cube (single
  palette entry) has **no index array at all**; `bits_per_entry()` is 0 and
  `heap_bytes()` is a few bytes.
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
## Protocol and links

`protocol` holds the message enums the client and server exchange:
`PROTOCOL_VERSION`, `PlayerId`, `ClientMsg` (`Hello`, `PlayerInput`,
`Ping`, `Disconnect`) and `ServerMsg` (`Welcome` with the spawn position,
`Rejected`, `Tick`, `Pong`, `Disconnected`, `CubeData { pos, cube }`,
`UnloadCube { pos }`). Messages are plain owned data — no `Arc`, no
references, no trait objects — all derive `Debug, Clone, PartialEq`, and
stay serializable for a future network transport (no serde yet).

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
