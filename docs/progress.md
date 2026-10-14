# Progress

## Next up

Stage 0 · Setup. Wave A: 0.1 (workspace). Wave B: 0.2 (CI), 0.5 (screenshot harness), 0.6 (testkit + benches), in parallel after 0.1 merges.

## Benchmarks

None yet.

## Decisions

- 2026-10-14: Crate packages are prefixed `jbm-` (a package named `core` would clash with Rust's `core`). Dirs stay `crates/core` etc.
- 2026-10-14: Workers run through `scripts/delegate.sh` (the `glm` CLI directly, in the background) rather than the haiku relay agent, so long Rust builds aren't cut off by the relay's 10-minute Bash timeout.
- 2026-10-14: Screenshot tests compare with a tolerance and skip when no GPU adapter exists, unless `JBM_REQUIRE_GPU=1` (set in CI on Linux with Mesa lavapipe).

## Known issues

None yet.

## Log

### 2026-10-14
- Repo created. 0.3 done: CLAUDE.md, docs/design.md, docs/plan.md, docs/progress.md. 0.4 drafted: docs/briefs/, scripts/delegate.sh (validated by running 0.1 through it).
