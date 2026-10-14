# just-better-minecraft

A fast, mod-first voxel engine in Rust. The engine ships no content: blocks,
mobs, redstone, and dimensions are all WASM mods loaded through a mod host on
each side of a client/server split. See [docs/design.md](docs/design.md) for
the architecture and [CLAUDE.md](CLAUDE.md) for how the project is built and
how work is organized.

## Build and test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
cargo run -p jbm-game
```
