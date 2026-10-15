//! The `jbm` binary: starts an in-process server on its own thread,
//! connects the client to it over an in-process link, and runs the client
//! window. The server thread is joined on exit — never detached.

use std::path::PathBuf;

use jbm_core::link::in_process_pair;
use jbm_server::run;
use jbm_server::server::{Server, ServerConfig};

/// The world directory the game saves to, unless `JBM_WORLD_DIR` overrides
/// it. `saves/default` under the current directory.
fn world_dir() -> Option<PathBuf> {
    if std::env::var("JBM_NO_SAVE").is_ok_and(|v| v == "1") {
        return None;
    }
    Some(match std::env::var("JBM_WORLD_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => PathBuf::from("saves/default"),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ServerConfig {
        world_dir: world_dir(),
        ..ServerConfig::default()
    };
    // A storage error (locked world, newer format, …) is a one-line exit,
    // not a panic.
    let server = match Server::try_with_config(config) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("jbm: cannot open the world: {error}");
            std::process::exit(1);
        }
    };
    let handle = run::spawn(server);
    let (client_end, server_end) = in_process_pair();
    handle.connect(server_end);
    let result = jbm_client::app::run(client_end);
    // The window closed (or the link died); stop and join the server. That
    // flushes every accepted edit to disk before the process exits.
    handle.stop();
    result
}
