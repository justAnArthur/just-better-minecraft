//! The `jbm` binary: starts the client and an in-process server.

fn main() {
    // jbm-client and jbm-server are linked in so the binary pulls in both
    // sides; nothing starts them yet (Stage 0 is an empty workspace).
    println!("{} v{}", jbm_core::ENGINE_NAME, env!("CARGO_PKG_VERSION"));
}
