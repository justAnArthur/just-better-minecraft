# Build plan

> Exported from the design doc artifact (Build plan tab), 2026-10-14. This file is the working checklist; the orchestrator ticks tasks here after each wave.

The project is built in six stages by one orchestrator session that delegates every task to subagents and keeps all state in repo files, so `/compact` and `/clear` never lose anything.

## How it works

One orchestrator session (Claude Opus, in the repo) runs each stage. It writes task specs, starts subagents, merges and keeps the progress files; subagents do the reading and coding in their own contexts.

Per-task loop:

```
orchestrator writes docs/tasks/<id>.md
  -> creates worktree ../jbm-wt/<id> on branch task/<id>
  -> glm-worker implementer (brief: docs/briefs/implementer.md) -> report
  -> glm-worker reviewer, fresh context (brief: docs/briefs/reviewer.md) -> verdict
       fix  -> new implementer with the findings
       merge -> orchestrator checks screenshots / risky diffs, merges, updates progress.md
after each wave: /compact     after each stage gate: /clear
```

Tasks in the same wave run in parallel, each implementer in its own git worktree. Only reports of 200 words or less come back to the orchestrator.

## Repo memory files

Rule: anything that must survive a `/clear` lives in a file. A fresh session reads `CLAUDE.md` and `docs/progress.md` and knows where it is.

| File | Holds | Updated by | When |
| --- | --- | --- | --- |
| `CLAUDE.md` | Build and test commands, crate map, coding rules, "read docs/progress.md first" | Orchestrator | Rarely |
| `docs/design.md` | The Design tab, exported | You | On design changes |
| `docs/plan.md` | This file: stages, waves, task checklist | Orchestrator | After each wave |
| `docs/progress.md` | "Next up" at the top, then a dated log: merged tasks, decisions, measured numbers, known issues | Orchestrator | After each task |
| `docs/tasks/<id>.md` | One spec per task: goal, crate, interfaces it may touch, acceptance tests, out of scope | Orchestrator | Before delegating |
| `docs/briefs/*.md` | GLM brief templates (implementer, reviewer, bench) | Orchestrator | Rarely |
| `crates/*/README.md` | Crate purpose, public API, invariants | Implementer | With the code |
| `wit/*.wit` | Mod API, human-owned from Stage 3 | You + orchestrator | Reviewed changes only |

## Subagents

| Role | Runs on | How | Job | Returns (≤ 200 words) |
| --- | --- | --- | --- | --- |
| Orchestrator | Claude Opus | This session | Task specs, merges, progress files, screenshot checks, final check of risky changes, talks to you | — |
| Implementer | GLM-5.3 | `glm-worker` in its own git worktree | Implements one task spec: code, tests, crate README | Branch, files touched, tests added, open issues |
| Reviewer | GLM-5.3 | `glm-worker`, fresh context | Reviews a branch against its spec and the design doc; runs tests and clippy | Verdict (merge / fix), up to 5 findings with file:line |
| Bench | GLM-5.3 | `glm-worker` | Runs `criterion` benches, compares with the last numbers in progress.md | A small table of numbers and the delta |

The orchestrator still checks itself before merging: screenshot tests (it reads the PNGs), and changes to `wit/`, `unsafe` code, concurrency or GPU memory.

## Context rules

- `/compact` after every wave, `/clear` after every stage. You type both; the orchestrator says when.
- Subagent reports are capped at 200 words. Anything longer goes into a file.
- The orchestrator never reads whole source files; it reads crate READMEs or asks `Explore`.
- Build and test output is filtered (`cargo test 2>&1 | tail -30`).
- One stage per session.

## Stage 0 · Setup

| Task | Who | Done when |
| --- | --- | --- |
| [x] 0.1 Cargo workspace: `core`, `server`, `client`, `modhost`, `game`; pinned toolchain; `rustfmt`, `clippy` | implementer | `cargo build` and `cargo clippy -D warnings` pass |
| [ ] 0.2 CI on GitHub Actions: build, test, clippy on macOS and Linux | implementer | Green run on the first PR |
| [x] 0.3 `CLAUDE.md`, `docs/design.md`, `docs/plan.md`, `docs/progress.md` | orchestrator | A fresh session can summarize the project from them |
| [ ] 0.4 GLM brief templates (implementer, reviewer, bench) in `docs/briefs/` | orchestrator | A dummy task runs through implementer and reviewer and returns reports in format |
| [ ] 0.5 Offscreen screenshot test harness (wgpu renders to PNG, compared with golden images) | implementer | A test renders a triangle and matches its golden image |
| [ ] 0.6 Headless server test helpers and `criterion` bench skeleton | implementer | One sample test and one sample bench run in CI |

**Gate:** CI green; screenshot tests and the bench brief work end to end. Then `/clear`.

## Stage 1 · Voxel sandbox (M1)

Blocks are hardcoded in Rust for now.

| Wave | Task | Crate | Done when |
| --- | --- | --- | --- |
| A | [ ] 1.1 Cube storage, palette compression, block state registry | `core` | Unit tests + bench for get/set and palette growth |
| A | [ ] 1.2 Window, wgpu setup, fly camera, texture array, debug overlay (FPS, position) | `client` | Screenshot test of an empty sky scene |
| A | [ ] 1.3 20 Hz tick loop, in-process channel, protocol message types | `server`, `core` | Headless test: client and server exchange messages for 100 ticks |
| B | [ ] 1.4 Greedy mesher on a worker pool, packed 8-byte faces, vertex pulling | `client` | Golden images for sample cubes; meshing bench |
| B | [ ] 1.5 Simple noise terrain, cube streaming around the player | `server` | Headless test streams cubes in a radius; screenshot of terrain |
| C | [ ] 1.6 Collision, walk and fly, client prediction with server correction | `client`, `server` | Tests: no falling through floors, prediction converges |
| C | [ ] 1.7 Raycast, break and place blocks, re-mesh on change | `client`, `server` | Test: edit on server shows on client within one tick |
| C | [ ] 1.8 Benchmark 16³ vs 32³ cubes (meshing time, memory, draw calls) | bench | Decision recorded in progress.md |

**Gate:** you play it: walk, break, place. Then `/clear`.

## Stage 2 · Far view (M2)

| Wave | Task | Crate | Done when |
| --- | --- | --- | --- |
| A | [ ] 2.1 Declarative worldgen: density-function graph format and native evaluator; replaces 1.5's noise | `server` | Same terrain from graph as from code; eval bench at several resolutions |
| A | [ ] 2.2 GPU buffer suballocator, multi-draw indirect, fallback path | `client` | Screenshot tests unchanged; draw-call count drops in bench |
| A | [ ] 2.3 `redb` persistence of cubes, zstd, background saving | `server` | Kill-during-save test reloads a valid world |
| B | [ ] 2.4 LOD pyramid: downsampling rule, edit propagation up the levels | `core`, `server` | Tests: thin walls survive LOD1–2; edit shows in parent LOD |
| B | [ ] 2.5 GPU frustum and Hi-Z occlusion culling in compute | `client` | Golden images match; bench shows culled counts in caves |
| C | [ ] 2.6 Clipmap streaming of LOD rings, skirts at seams | `client`, `server` | Fly test: no holes at ring borders (screenshots) |
| C | [ ] 2.7 LOD sampled straight from worldgen for unvisited terrain; stored LOD in redb | `server` | Far terrain appears without generating full detail; reload shows it instantly |

**Gate:** fly across kilometres without frame hitches; bench records frame times. Then `/clear`.

## Stage 3 · Mod host (M3)

The API stage. Wave A is drafted by the orchestrator and reviewed by you before any implementer starts.

| Wave | Task | Crate | Done when |
| --- | --- | --- | --- |
| A | [ ] 3.1 WIT files v0: `world`, `registry`, `events`, `scheduler`, `worldgen` (server); `render`, `ui`, `input` (client) | `wit/` | You approve them |
| A | [ ] 3.2 `wasmtime` embedding on the server, mod loading from `mod.toml`, dependency order | `modhost`, `server` | A Rust test mod registers a block and handles an event |
| B | [ ] 3.3 Declarative content loader: blocks, items, models, textures from `content/` and `assets/` | `modhost`, `core`, `client` | A data-only mod adds a block that renders and collides |
| B | [ ] 3.4 AssemblyScript binding generator from WIT, plus a template mod | `tools/as-bindgen` | An AssemblyScript mod passes the same test as the Rust one |
| B | [ ] 3.5 Client-side mod host: UI and input APIs | `modhost`, `client` | A client mod draws a HUD element |
| C | [ ] 3.6 Move all hardcoded content into `base` and `worldgen.overworld` mods | `mods/` | Engine runs with zero built-in content |
| C | [ ] 3.7 Hot reload of `content/` and `.wasm` in dev builds | `modhost` | Edit a texture or handler, see it change without restart |
| C | [ ] 3.8 WASM boundary benchmark: batched vs per-block calls | bench | Numbers in progress.md; API changed if batching is not enough |

**Gate:** the engine has no content of its own, and both mod languages work. Then `/clear`.

## Stage 4 · Survival core (M4)

Engine services in wave A, then content mods in AssemblyScript in wave B.

| Wave | Task | Crate | Done when |
| --- | --- | --- | --- |
| A | [ ] 4.1 Gameplay light (flood fill, sky heightmap) and visual light (smooth AO, colored light) | `server`, `client` | Light tests at cube borders; golden images |
| A | [ ] 4.2 Entities in ECS, entity sync, pathfinding service for mods | `server`, `core` | Headless test: an entity walks a maze |
| A | [ ] 4.3 UI framework: inventory grid, slots, drag and drop, mod-defined screens | `client`, `modhost` | Screenshot tests of inventory and a mod screen |
| B | [ ] 4.4 `survival` mod: health, hunger, damage, death, day/night | `mods/` | Headless tests for each rule |
| B | [ ] 4.5 `crafting` mod: recipes, crafting grid, furnace as block entity | `mods/` | Furnace smelts with catch-up after reload |
| B | [ ] 4.6 `mobs.passive`, `mobs.hostile`: spawning, AI, client renderers | `mods/` | Mobs spawn by light level and path to the player |
| B | [ ] 4.7 `farming` mod: crops, random ticks, catch-up growth | `mods/` | Crops grow while their cube is unloaded |

**Gate:** you play the full loop: spawn, gather, craft, fight, save, reload. Then `/clear`.

## Stage 5 · Depth and API freeze (M5)

| Wave | Task | Crate | Done when |
| --- | --- | --- | --- |
| A | [ ] 5.1 Region-parallel ticking with border queues and sync points | `server` | Replay test: same result single- and multi-threaded |
| A | [ ] 5.2 Multiple worlds per server, dimension API, moving entities between worlds | `server`, `wit/` | A test mod adds a flat dimension and a portal |
| B | [ ] 5.3 `redstone` mod in Rust: wire, torches, repeaters, pistons; deterministic updates | `mods/` | Circuit test suite; tick bench of a large circuit |
| B | [ ] 5.4 `dimension.nether`, `dimension.end` mods | `mods/` | Travel there and back; both generate and render LOD |
| B | [ ] 5.5 `structures` mod with LOD stamps | `mods/` | Villages visible in LOD before visiting |
| C | [ ] 5.6 Mod API 1.0: freeze WIT, version it, write mod author docs | `wit/`, `docs/` | Docs build; every API has an example mod |
| C | [ ] 5.7 Mod-heavy world benchmark (all vanilla mods plus stress mods) | bench | Tick and frame times recorded |

**Gate:** someone (or a fresh agent with only the docs) builds a working mod. Then multiplayer planning.

## Kickoff prompts

Copy these into the orchestrator session. Replace `N` and the wave letter.

**Start a stage** (right after `/clear`):

```
Read CLAUDE.md, docs/progress.md and the Stage N section of docs/plan.md.
Write task specs for wave A into docs/tasks/ and show me the list.
Then run one glm-worker implementer per task in parallel, each in its own worktree.
When one returns, run a glm-worker reviewer on its branch. Merge what passes,
send failures back to a new implementer with the findings.
Update docs/progress.md after each merge, and tell me when the wave is done.
```

**Next wave:**

```
Wave A is merged. Start wave B the same way.
```

**Compact after a wave:**

```
/compact Keep: current stage and wave, tasks done and open, decisions made this stage, anything I asked to remember. Drop: diffs, test output, finished task details.
```

**Gate check:**

```
Stage N gate: run the bench glm-worker, check the screenshots yourself, compare results with the gate in docs/plan.md, update docs/progress.md, and tell me exactly what to playtest.
```

**Resume anytime:**

```
Read CLAUDE.md and docs/progress.md. Tell me where we are and what is next.
```
