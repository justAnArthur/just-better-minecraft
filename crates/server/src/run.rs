//! The real-time server loop: one dedicated thread ticking a [`Server`] at
//! [`TICK_PERIOD`](crate::tick::TICK_PERIOD) cadence.

use std::panic;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use jbm_core::link::ServerLink;

use crate::server::Server;

/// Handle to a server running on its own thread. Dropping it stops and
/// joins the thread — a handle never detaches.
pub struct ServerHandle {
    new_links: mpsc::Sender<ServerLink>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Server>>,
}

/// Start `server` on a dedicated thread named `"jbm-server"` and return a
/// handle to it.
pub fn spawn(server: Server) -> ServerHandle {
    let (new_links_tx, new_links_rx) = mpsc::channel::<ServerLink>();
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let thread = thread::Builder::new()
        .name("jbm-server".to_owned())
        .spawn(move || drive(server, new_links_rx, thread_stop))
        .expect("spawning the jbm-server thread");
    ServerHandle {
        new_links: new_links_tx,
        stop,
        thread: Some(thread),
    }
}

/// The loop the server thread runs: measure elapsed time with `Instant`,
/// drain new connections, run due ticks, then sleep until the next one.
fn drive(
    mut server: Server,
    new_links: mpsc::Receiver<ServerLink>,
    stop: Arc<AtomicBool>,
) -> Server {
    let mut last = Instant::now();
    while !stop.load(Ordering::Acquire) {
        let now = Instant::now();
        let elapsed = now - last;
        last = now;
        while let Ok(link) = new_links.try_recv() {
            server.add_connection(link);
        }
        server.update(elapsed);
        thread::sleep(server.until_next_tick());
    }
    server
}

impl ServerHandle {
    /// Hand a new connection to the server thread, which registers it at
    /// the start of its next tick. Silently ignored only if the thread is
    /// already gone (it panicked); [`ServerHandle::stop`] surfaces that.
    pub fn connect(&self, link: ServerLink) {
        let _ = self.new_links.send(link);
    }

    /// Stop the server thread, join it and return the [`Server`] so tests
    /// can inspect it. If the thread panicked, the panic is propagated
    /// rather than swallowed.
    pub fn stop(mut self) -> Server {
        self.request_stop();
        let thread = self
            .thread
            .take()
            .expect("handle owns the thread until stopped");
        match thread.join() {
            Ok(server) => server,
            Err(panic_payload) => panic::resume_unwind(panic_payload),
        }
    }

    fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.request_stop();
        if let Some(thread) = self.thread.take() {
            // Join so the thread never outlives the handle, even on drop.
            // A panicked thread is quietly discarded here; `stop` is the
            // API that surfaces panics.
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Server, spawn};
    use jbm_core::link::{ClientLink, Link, LinkClosed, in_process_pair};
    use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
    use std::panic;
    use std::thread;
    use std::time::{Duration, Instant};

    /// Every wait in these tests is bounded by this.
    const TIMEOUT: Duration = Duration::from_secs(2);

    /// Receive the next server message, polling until it arrives or
    /// `timeout` passes. Panics on timeout — the server thread may have
    /// died, and an unbounded wait would hide that.
    fn recv_timeout(client: &mut ClientLink, timeout: Duration) -> Result<ServerMsg, LinkClosed> {
        let deadline = Instant::now() + timeout;
        loop {
            match client.try_recv() {
                Ok(Some(msg)) => return Ok(msg),
                Ok(None) => {
                    assert!(
                        Instant::now() < deadline,
                        "timed out waiting for a server message"
                    );
                    thread::sleep(Duration::from_millis(2));
                }
                Err(closed) => return Err(closed),
            }
        }
    }

    /// A link whose server-facing half explodes on the first send, to
    /// prove panics in the server thread surface through `stop`.
    struct PanicOnSendLink {
        hello_sent: bool,
    }

    impl Link<ServerMsg, ClientMsg> for PanicOnSendLink {
        fn send(&mut self, _msg: ServerMsg) -> Result<(), LinkClosed> {
            panic!("link send exploded")
        }

        fn try_recv(&mut self) -> Result<Option<ClientMsg>, LinkClosed> {
            if self.hello_sent {
                Ok(None)
            } else {
                self.hello_sent = true;
                Ok(Some(ClientMsg::Hello {
                    protocol_version: PROTOCOL_VERSION,
                    name: "boom".to_owned(),
                }))
            }
        }
    }

    /// Drain the link until the other end closes it. Panics on timeout —
    /// the server thread may have died, and an unbounded wait would hide
    /// that.
    fn wait_for_close(client: &mut ClientLink, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            match client.try_recv() {
                Err(LinkClosed) => return,
                Ok(Some(_)) => {}
                Ok(None) => {
                    assert!(
                        Instant::now() < deadline,
                        "timed out waiting for LinkClosed"
                    );
                    thread::sleep(Duration::from_millis(2));
                }
            }
        }
    }

    #[test]
    fn connected_client_is_welcomed() {
        let (mut client, server_end) = in_process_pair();
        let handle = spawn(Server::new());
        handle.connect(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "alice".to_owned(),
            })
            .unwrap();
        let msg = recv_timeout(&mut client, TIMEOUT).expect("server link open");
        assert!(matches!(msg, ServerMsg::Welcome { .. }));
        handle.stop();
    }

    #[test]
    fn three_hundred_ms_runs_a_handful_of_ticks() {
        let handle = spawn(Server::new());
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(300) {
            thread::sleep(Duration::from_millis(25));
        }
        let server = handle.stop();
        // 300 ms is 6 ticks at 20 Hz; the bounds stay loose so a loaded
        // CI machine still passes.
        let ticks = server.tick_count();
        assert!(
            (3..=12).contains(&ticks),
            "tick count {ticks} outside 3..=12"
        );
    }

    #[test]
    fn dropping_the_handle_stops_the_thread() {
        let (mut client, server_end) = in_process_pair();
        let handle = spawn(Server::new());
        handle.connect(server_end);
        drop(handle);
        // Drop joined the thread, so the Server and its links are gone:
        // the client end must observe LinkClosed once drained.
        wait_for_close(&mut client, TIMEOUT);
    }

    #[test]
    fn stop_propagates_a_server_thread_panic() {
        let (mut client, server_end) = in_process_pair();
        let handle = spawn(Server::new());
        handle.connect(server_end);
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "observer".to_owned(),
            })
            .unwrap();
        // The exploding link joins on a later tick; sending its Welcome
        // panics the server thread. The unwind drops the Server and every
        // connection with it, which the client end observes as a close.
        handle.connect(Box::new(PanicOnSendLink { hello_sent: false }));
        wait_for_close(&mut client, TIMEOUT);

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| handle.stop()));
        assert!(
            result.is_err(),
            "stop must propagate the server thread panic"
        );
    }
}
