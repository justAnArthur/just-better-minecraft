//! The `jbm` binary: starts an in-process server on its own thread,
//! connects the client to it over an in-process link, and runs the client
//! window. The server thread is joined on exit — never detached.

use jbm_core::link::in_process_pair;
use jbm_server::run;
use jbm_server::server::Server;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = run::spawn(Server::new());
    let (client_end, server_end) = in_process_pair();
    handle.connect(server_end);
    let result = jbm_client::app::run(client_end);
    // The window closed (or the link died); stop and join the server.
    handle.stop();
    result
}
