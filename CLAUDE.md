# just-better-minecraft

A fast, mod-first voxel engine in Rust. The engine has no content; everything (blocks, mobs, redstone, dimensions) ships as WASM mods. Design: `docs/design.md`. Stages and task checklist: `docs/plan.md`.

**Read `docs/progress.md` first.** Its "Next up" section says where the project is.

## Commands

```bash
cargo build --workspace
cargo test --workspace 2>&1 | tail -30
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -30
cargo fmt --all --check
cargo bench --workspace -- --test          # smoke-run every bench once
JBM_UPDATE_GOLDEN=1 cargo test -p jbm-client --test screenshot   # regenerate golden images
```

Always filter build and test output (`| tail -30`) before it reaches an agent context.

## Crate map

| Dir | Package | Purpose |
| --- | --- | --- |
| `crates/core` | `jbm-core` | Shared types: cubes, palettes, block states, registries, protocol messages |
| `crates/server` | `jbm-server` | Simulation, worldgen, lighting, storage |
| `crates/client` | `jbm-client` | Renderer, UI, input, audio, prediction |
| `crates/modhost` | `jbm-modhost` | wasmtime embedding and WIT bindings (both sides) |
| `crates/game` | `jbm-game` | Binary: starts the client and an in-process server |
| `crates/testkit` | `jbm-testkit` | Test-only helpers: headless server harness, fixtures |
| `wit/` | — | Mod API (human-owned from Stage 3) |
| `mods/` | — | Vanilla content mods (from Stage 3) |

Each crate has a `README.md` with purpose, public API and invariants. Read that instead of the source.

## Coding rules

- Rust edition 2024, toolchain pinned in `rust-toolchain.toml`. Dependency versions live in `[workspace.dependencies]`; crates use `dep.workspace = true`.
- `unsafe_code` is denied workspace-wide. A crate that truly needs it opts out in its own `Cargo.toml` with a comment saying why; the orchestrator reviews every such change.
- Clippy clean with `-D warnings`. No `#[allow]` without a one-line reason comment.
- `client` and `server` never depend on each other; they share only `core` and talk through protocol messages.
- Hot paths (meshing, collision, light) never call into mods.
- No Mojang code or assets. Clean-room only.
- Tests: unit tests next to code; integration tests in `crates/*/tests/`. Renderer changes need screenshot tests; perf-sensitive code needs a `criterion` bench.
- Git: one branch per task, `task/<id>`. Commit messages are plain imperative sentences. **Never add `Co-Authored-By` trailers, "Generated with" lines or any AI/agent attribution** to commits or PRs.

## How work is done (orchestrator + GLM workers)

- The orchestrator (Claude) writes task specs in `docs/tasks/<id>.md`, delegates, reviews risky diffs, merges, and updates `docs/progress.md` and the checklist in `docs/plan.md`.
- Workers are GLM-5.3 agents run with `scripts/delegate.sh <role> <task-id>` (roles: `implementer`, `reviewer`, `bench`). It builds the brief from `docs/briefs/<role>.md`, runs the `glm` CLI in the task's worktree (`../jbm-wt/<id>`, branch `task/<id>`), and writes the report to `.agents/reports/<id>-<role>.md` (full log in `.agents/logs/`).
- Reports are ≤ 200 words. The orchestrator reads reports, not logs, unless something failed.
- Before merging, the orchestrator itself checks: screenshot PNGs, and any change to `wit/`, `unsafe`, concurrency or GPU memory management.
- `/compact` after each wave, `/clear` after each stage gate.
