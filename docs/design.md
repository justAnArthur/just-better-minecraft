# just-better-minecraft — Design Doc

> Exported from the design doc artifact (Design tab), 2026-10-14. The artifact is the source of truth for design; update this file when it changes.
> Source: https://claude.ai/code/artifact/17c52715-5320-490f-b429-e2d028b74657

## Vision and goals

We are building a fast, mod-first voxel engine that fixes Minecraft's structural pain points, not a 1.8 clone. The engine knows nothing about redstone, the Nether or zombies: all game content ships as mods, and "the game" is a modpack.

v1 differentiators:

1. **Huge view distance.** A built-in LOD system shows kilometres of terrain while simulating only what is near.
2. **Unlimited height.** Cubic chunks; no build ceiling or floor.
3. **Multithreaded simulation.** One busy farm never stalls the whole world.
4. **Mod-first, client and server.** One sandboxed WASM mod system on both sides, with a stable API and automatic sync of mods to clients.

Non-goals for v1:

- Feature parity with any Minecraft version, or protocol/save compatibility with it.
- Mojang code or assets. Clean-room implementation, own or openly licensed art.
- Multiplayer UX (server browser, accounts). The architecture supports it; the product ships later.

## Key decisions

| Area | Decision | Why |
| --- | --- | --- |
| Language | Rust (stable) | Speed without GC pauses; safe parallelism for meshing, worldgen and ticking |
| Graphics | wgpu (Vulkan, Metal, DX12) | Native Metal on macOS, no MoltenVK; near raw-Vulkan speed for this workload with far less code |
| Windowing / input | winit | Standard pairing with wgpu |
| Entities | `bevy_ecs` used standalone (not the full Bevy engine) | Parallel system scheduling without Bevy's renderer and release churn |
| Mods | WASM components (`wasmtime`, WIT interfaces) on client and server, written in Rust or AssemblyScript | Sandboxed, language-agnostic, versionable ABI; safe to auto-download |
| Process model | Client and server split internally from day one; singleplayer runs an in-process server | Multiplayer later without a rewrite; forces clean mod boundaries |
| Multiplayer | Not in v1 | The transport is a channel in-process now; swapped for a network socket later |
| IDs | Namespaced strings (`redstone:wire`) mapped to per-world numeric palettes | No ID conflicts between mods |
| Storage | `redb` (embedded, crash-safe key-value store) | No corrupted region files |

## Architecture overview

The engine is a client and a server sharing one `core` crate; mods plug into a mod host on each side and never touch the hot paths.

In v1 both sides run in one process and talk over a channel. Multiplayer later swaps the channel for a network socket without changing either side.

- `core`: shared types (cubes, palettes, block states, registries, protocol messages)
- `server`: simulation, worldgen, lighting, storage
- `client`: renderer, UI, input, audio, prediction
- `modhost`: wasmtime embedding and WIT bindings, used by both sides
- `game`: the binary that starts the client and an in-process server
- `mods/*`: the vanilla mods, each its own crate compiled to WASM

```
            game (binary)
           /             \
     client  <-channel->  server
   (render, UI,         (sim, worldgen,
    input, predict)      light, storage)
        |    \           /    |
        |     \-- core --/    |
        |                     |
     modhost (client)    modhost (server)
        |                     |
     client.wasm          server.wasm   <- mods/*
```

## World representation

The world is an unbounded 3D grid of 32×32×32-block cubes. There is no height limit, and only the cubes near the player are loaded.

- **Cube coordinates** are `i32` triples, so the world spans about ±68 billion blocks on every axis. Block positions within a cube are 5-bit local coordinates.
- **Storage per cube** is a palette plus a bit-packed index array (1–15 bits per block depending on palette size). An all-air or all-stone cube stores just its single palette entry, with no array.
- **Block states** are `(block id, properties)`, interned to a global `u32` state id at startup. Properties come from the mod's declaration (for example `facing`, `powered`).
- **IDs** are namespaced strings in mods and on disk. Each world keeps its own string → numeric table, so adding or removing a mod never shuffles IDs.
- **Block entities** (chests, furnaces) live in the ECS, keyed by block position. They are not stored in the cube array.
- **Dimensions** are separate `World` instances, each with its own generator, sky rules, gravity and time. The engine only knows "a world"; mods define dimensions.
- **Why 32³:** fewer cubes and fewer draw calls than 16³, and still small enough to re-mesh quickly after an edit. This is a starting point to benchmark, not a fixed rule.

## LOD system

LOD is a pyramid of the same 32³ cube format: at level *n*, each cell covers 2ⁿ blocks per side. One storage format, mesher and streaming path serve every level.

| Level | Block size of one cell | Area one cube covers |
| --- | --- | --- |
| LOD0 | 1 block | 32 blocks |
| LOD1 | 2 blocks | 64 blocks |
| LOD2 | 4 blocks | 128 blocks |
| LOD3 | 8 blocks | 256 blocks |
| LOD*n* | 2ⁿ blocks | 32·2ⁿ blocks |

**Clipmap.** Rings of LOD levels surround the camera, each ring twice as far and twice as coarse as the one inside it. When the camera crosses a cube boundary, only the cubes at the ring edges change.

**Two sources of LOD data:**

1. **From worldgen, for terrain nobody has visited.** Worldgen is written as density and biome functions that can be sampled at any resolution. Sampling them every 2ⁿ blocks gives a LOD cube directly, without generating the full-detail terrain underneath. Distant mountains appear in seconds.
2. **From the real world, for terrain that exists.** When a LOD0 cube is generated or edited, it is downsampled into its LOD1 parent, then up the pyramid. Player builds stay visible from far away, and they override the worldgen guess.

**Worldgen is declarative.** Terrain shape is a graph of density functions (noise, splines, math) declared as data. The engine evaluates it natively at any resolution, which makes LOD sampling possible and keeps worldgen equally fast whatever language a mod uses. Code runs only for later stages such as trees, ores and structures.

**Downsampling rule.** Each 2×2×2 group becomes one cell, as a pure function of the eight child states (so one edit changes at most one cell per level, and propagation is exact). A child is filled when it is not air; the cell is filled when **at least 2 of 8** children are filled, otherwise it is air. Thin features such as walls and tree trunks then survive every level instead of vanishing, while a single floating block or torch-sized feature disappears. The rule leans solid — surfaces can rise by up to 2ⁿ − 1 blocks (skirts hide that) and tunnels narrower than 2ⁿ can close (invisible from LOD distance). A filled cell's material is "what you see from above": the candidates are the topmost filled child of each of the four `(dx, dz)` columns of the group; pick the most common candidate, ties going to the state at the highest `dy`, then to the smallest state id. Grass therefore stays on top of a surface however it lines up with the group, and a wall keeps its own material. Water and every other non-air state count as filled; block properties are not consulted. A later refinement may use registry data, but must stay local to the eight children.

**Structures from mods** appear in LOD only once they are generated. Mods can optionally provide a cheap "LOD stamp" for their structures so a village's silhouette shows before anyone visits it.

**Seams** between levels are hidden with skirts: short downward faces on the edges of coarser cubes. That is enough for blocky terrain; no Transvoxel-style stitching.

**Simulation never runs in LOD.** LOD cubes are read-only rendering data. Simulation distance (around 8–16 cubes) and view distance are set separately.

**Later option:** raymarch the farthest rings instead of rasterizing them. Not for v1.

## Rendering

Rendering is GPU-driven: the CPU builds compact meshes off the main thread, and the GPU decides what to draw.

- **Meshing** runs on a worker pool. Greedy meshing merges coplanar faces of the same block. Each face is packed into 8 bytes (position, size, direction, texture index, AO, light), and the vertex shader unpacks it. There is no vertex buffer per vertex.
- **Block models.** Full cubes use the fast path. Non-cube shapes (stairs, slabs, plants, wires) come from declarative models that mods ship (a JSON-like list of boxes and textures). The engine meshes them natively, so mods never run during meshing.
- **Buffers.** All cube meshes live in a few large GPU buffers with a suballocator. A cube's re-mesh replaces its slot instead of creating a new buffer.
- **Drawing.** One indirect draw command per visible cube and face direction, issued with multi-draw indirect. Back-facing directions are skipped per cube.
- **Culling.** Frustum culling and hierarchical-Z occlusion culling run in a compute shader that writes the indirect commands. Caves under the player cost almost nothing.
- **Textures.** One texture array holds all block textures, built at load time from every mod's assets.
- **Fallback.** Hardware without multi-draw indirect gets a plain loop of draw calls. Slower, same output.

**Lighting** has two separate layers:

| Layer | Purpose | How |
| --- | --- | --- |
| Gameplay light | Mob spawning, crop growth, mod logic | Integer 0–15 block and sky light, flood-filled on the server per cube |
| Visual light | What the player sees | Uses the gameplay values plus smooth AO, colored block light, and later screen-space or voxel GI |

Sky light with unlimited height is the hard case. Each column keeps a sky heightmap of the highest opaque block seen so far. Unloaded cubes above the heightmap count as open sky; flood fill corrects it as cubes load.

## Simulation

The server ticks at a fixed 20 Hz, in parallel across regions, and only does work that something asked for.

**Tick phases** (each in a fixed order, so results are deterministic):

1. Apply player inputs and network or channel messages.
2. Run scheduled block ticks that are due.
3. Process the block update queue (neighbor changes), to a per-tick budget.
4. Run ECS systems: physics, AI, block entities, mod systems.
5. Recompute dirty light; send changes to clients.

**Parallelism.** Loaded cubes are grouped into regions (for example 4×4×4 cubes). Independent regions are ticked in parallel on worker threads; work that crosses a region border is queued and applied at a sync point. This is the same idea as the Folia server fork. ECS systems that touch disjoint data also run in parallel through `bevy_ecs` scheduling. Chunk generation, meshing and lighting always run on worker threads, never on the tick thread.

**No random polling of everything.** Minecraft's random ticks (crop growth, leaf decay) stay, but only for blocks that registered for them. Block entities sleep until an event or a scheduled tick wakes them. An idle furnace costs nothing.

**Catch-up simulation.** When a cube loads, mods receive the time elapsed since it was saved. A wheat field or a furnace computes its result in one step, so unloaded farms still progress without being simulated.

**Physics.** Axis-aligned box collision against per-block-state collision shapes, swept per tick. The player is predicted on the client and corrected by the server, even in singleplayer, so multiplayer later behaves the same.

## Mod system

A mod is a package of declarative content plus optional WASM code for the server, the client, or both. It talks to the engine only through versioned WIT interfaces, never by patching engine code.

**Package layout**

```
redstone/
  mod.toml          # id, version, dependencies (semver), API version
  content/          # blocks, items, recipes, models, loot tables (data)
  assets/           # textures, sounds, lang files
  server.wasm       # optional: server-side behavior
  client.wasm       # optional: client-side behavior
```

**Rule: content is data, behavior is code.** A block's shape, textures, hardness, light emission, collision box and drops are declared in `content/`. The engine meshes, lights and collides it natively. WASM runs only for behavior.

| Side | Mods can | Engine API (WIT worlds) |
| --- | --- | --- |
| Server | Register blocks, items, entities, dimensions; handle events; scheduled and random ticks; worldgen stages; commands; custom saved data | `world`, `registry`, `events`, `scheduler`, `worldgen`, `entities`, `storage` |
| Client | Custom block/entity renderers (as GPU-buffer builders, not raw wgpu), particles, UI screens and HUD, keybinds, sounds | `render`, `ui`, `input`, `audio`, `particles` |
| Both | Send typed messages between client and server halves of the same mod | `net` |

**Mod languages.** Two languages compile to the same WASM components and use the same WIT interfaces.

| Language | Use for | Speed vs native Rust (approximate) |
| --- | --- | --- |
| Rust | Heavy logic: redstone simulation, custom worldgen stages, anything looping over many blocks per tick | ~1.2–2× slower |
| AssemblyScript (TypeScript-like syntax) | Most gameplay: block and item behavior, events, commands, crafting, UI screens, mob decisions | ~1.5–3× slower |

There is no official WIT binding generator for AssemblyScript, so we write one that turns our WIT files into AssemblyScript bindings. Full TypeScript (through `jco componentize` or an embedded V8) can come later behind the same API if mod authors need npm or real TS.

**Performance at the WASM boundary.** Calls cost far more than a Rust function call, so the API works in batches:

- Worldgen receives a whole cube or column and returns a filled buffer.
- Block access is by region views (`read_region`, `write_region`), not one block per call.
- Events are delivered in batches per tick.
- Hot paths (meshing, collision, light) never call into mods.

**Stability.** The WIT API is versioned (`engine:world@1.2`). Old mods keep working through adapter shims when the API moves to a new major version. A mod declares the API versions it needs; the loader refuses mismatches with a clear message instead of a crash.

**Sync to clients.** A client joining a world (local or later remote) receives the server's mod list with hashes. Missing client halves and assets are downloaded automatically. This is safe because WASM is sandboxed: no file system or network access except through engine APIs.

**Load order and conflicts.** Dependencies form a graph and load in topological order. Two mods may extend the same block through declared extension points (for example `on_neighbor_change` hooks), never by overwriting each other.

**Hot reload** of `content/` and `.wasm` in development builds, so mod authors (and agents) iterate without restarting.

## Persistence

A world is one directory with a `redb` database per dimension, written in transactions so a crash never leaves a half-written cube.

- **Keys:** `(kind, cube x, y, z)` for cubes, LOD cubes, block entities and entities; separate tables for player data, mod data and the ID palette.
- **Values:** cube palette and index array, compressed with zstd. LOD cubes are stored too, so far terrain loads instantly next session.
- **Saving:** dirty cubes are written in batches on a background thread every few seconds and on exit.
- **Schemas:** every stored record carries a schema version. The engine migrates its own records; each mod ships migrations for its own data. No central data-fixer like Minecraft's DataFixerUpper.
- **Removed mods:** blocks from a missing mod are kept as "unknown block" placeholders that preserve the original data, so re-adding the mod restores them.
- **Format is internal.** No compatibility with Minecraft's Anvil format; an importer can be a separate tool later if wanted.

## Vanilla as mods

The default game is a modpack of small, independent mods. Each one can be removed, replaced or forked, and each is also the reference example for the API it exercises.

| Mod | Contents | Depends on | API it proves |
| --- | --- | --- | --- |
| `base` | Air, stone, dirt, grass, wood, basic tools, player inventory, hotbar, block breaking and placing | — | Registries, declarative blocks and items, UI |
| `survival` | Health, hunger, damage, death and respawn, day/night cycle | `base` | Entity components, events |
| `crafting` | Crafting grid, recipes, furnace and smelting | `base` | Block entities, scheduled ticks, catch-up sim |
| `worldgen.overworld` | Terrain, biomes, caves, ores, trees | `base` | Worldgen stages, LOD sampling |
| `structures` | Villages, dungeons, strongholds-style structures | `worldgen.overworld` | Structure placement, LOD stamps |
| `mobs.passive`, `mobs.hostile` | Animals and monsters, spawning, AI | `survival` | Entities, pathfinding, client renderers |
| `redstone` | Wire, torches, repeaters, pistons, buttons, levers | `base` | Neighbor updates, deterministic ordering, circuit compilation |
| `dimension.nether`, `dimension.end` | Dimensions, portals, their blocks and mobs | `base`, `worldgen.*` | Multiple worlds, sky rules, travel between worlds |
| `farming` | Crops, growth, animals breeding | `survival` | Random ticks, catch-up sim |

The engine ships with no content at all. Running it with no mods gives an empty void and a free-flying camera, which is also the first milestone's test harness.

## Roadmap and how agents build it

Five milestones, each ending in something you can run and playtest; multiplayer comes after M5.

- **M1 Voxel sandbox** — walk, break, place in noise terrain.
- **M2 Far view** — LOD pyramid, clipmap, persistence.
- **M3 Mod host** — WIT v0, wasmtime, content moved into mods.
- **M4 Survival core** — light, entities, UI, survival/crafting/mobs/farming mods.
- **M5 Depth and API freeze** — region-parallel ticking, dimensions, redstone, structures, API 1.0.

M3 is the turning point: from then on, every new feature is a mod, which tests the API as it grows.

**How agents build it**

- **You own the boundaries:** crate layout, WIT files and this doc. Agents implement behind them and propose changes to them, never silently.
- **Small tasks with a spec:** each milestone splits into tasks of one subsystem each, with a short spec paragraph and acceptance tests.
- **Agents verify their own work:** headless server tests with deterministic replays; screenshot tests (golden images) for the renderer so agents can see visual bugs; benchmarks (`criterion`) for meshing, worldgen and ticking.
- **Parallel work:** independent crates (renderer vs worldgen vs mod host) are built by separate agents in separate git worktrees.
- **You playtest every gate.** Game feel, controls and art direction stay human calls.

Stage-by-stage execution with subagents and context management: see [plan.md](plan.md).

## Risks and open questions

| Risk | Impact | Mitigation |
| --- | --- | --- |
| WASM call overhead makes mods slow | Laggy mod-heavy worlds | Batch APIs from day one; benchmark `redstone` and worldgen as mods in M3 before freezing the API |
| Client-side mod API is large (rendering, UI) | Slow progress, unstable API | Ship UI and particles first; custom renderers produce buffers, never touch wgpu directly |
| Sky light with unlimited height | Wrong lighting at cube borders, visible bugs | Column heightmap plus incremental flood fill; dedicated test worlds |
| LOD from worldgen disagrees with real terrain | Pop-in when walking closer | Same density functions at every resolution; real data overrides the guess |
| Region-parallel ticking causes nondeterminism | Redstone and duplication bugs | Fixed phase order, border work queued to sync points, replay tests |
| Scope creep toward "better at everything" | Never ships | v1 = the four differentiators; everything else is a mod later |
| Agent-written code drifts architecturally | Hard-to-change codebase | Human-owned crate boundaries and WIT files; agents implement behind them |

**Open questions**

- [ ] Full TypeScript later: jco componentize or embedded V8? Decide after profiling AssemblyScript mods.
- [ ] Exact cube size: 32³ vs 16³, decided by meshing and memory benchmarks in M1.
- [ ] Art direction and texture resolution (16×16 vs higher), which affects the texture array and LOD look.
