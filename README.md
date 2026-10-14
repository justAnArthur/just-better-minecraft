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

## CI

[GitHub Actions](.github/workflows/ci.yml) runs on every push to `main` and
every pull request: `cargo fmt --check` and `cargo clippy -D warnings` on
Linux, `cargo test --workspace` on Linux and macOS, and a one-shot
`cargo bench -- --test` smoke run on Linux. The Linux test job installs
Mesa's lavapipe software Vulkan driver and sets `JBM_REQUIRE_GPU=1` so
screenshot tests must run; on macOS the variable stays unset and those
tests skip when no usable GPU is present. Failed test runs upload
`target/screenshots/` as an artifact.
