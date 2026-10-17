//! The `jbm` binary: starts an in-process server on its own thread,
//! connects the client to it over an in-process link, and runs the client
//! window. The server thread is joined on exit — never detached.

use std::path::PathBuf;

use jbm_client::app::ViewSetup;
use jbm_core::link::in_process_pair;
use jbm_server::run;
use jbm_server::server::{Server, ServerConfig};

/// The far-look view distances (task 2.9) derived from the server config:
/// the LOD0 streaming disk radius and the outermost LOD ring's radius, in
/// blocks. The client cannot learn these from the link, so the game —
/// which built the config — passes them along.
fn view_setup(config: &ServerConfig) -> ViewSetup {
    let (lod0_ring_radius, furthest_ring_radius) = config.ring_radii();
    ViewSetup {
        lod0_ring_radius,
        furthest_ring_radius,
    }
}

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
    // The far-look view distances, read off the config before it moves
    // into the server (task 2.9).
    let view = view_setup(&config);
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
    let result = jbm_client::app::run_with_view(client_end, view);
    // The window closed (or the link died); stop and join the server. That
    // flushes every accepted edit to disk before the process exits.
    handle.stop();
    result
}
