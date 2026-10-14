//! The `jbm` binary: starts the client window. The in-process server
//! (task 1.3) and its wiring arrive in Stage 1 wave B.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    jbm_client::app::run()
}
