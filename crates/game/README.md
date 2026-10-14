# jbm-game

The `jbm` binary: starts the client and an in-process server talking over
a channel (see docs/design.md, "Architecture overview").

**Wiring (task 1.4):** `main` spawns the server with
`jbm_server::run::spawn(Server::new())` (dedicated `"jbm-server"` thread),
builds an `in_process_pair()` link, hands the server end to the thread via
`handle.connect(server_end)`, and runs `jbm_client::app::run(client_end)`
— the window, the link polling, the mesh pool and the terrain renderer.
When the window closes (or the server drops the link), `app::run` returns
and `handle.stop()` joins the server thread. The server thread is never
detached; stopping it propagates a server panic rather than swallowing it.

Until task 1.5 (terrain generation and cube streaming) merges,
`Server::new()` sends no cubes, so the client flies over empty sky.

**Invariants**

- The only executable crate in the workspace.
- Depends on client, server, and core; client and server never depend on each
  other.
