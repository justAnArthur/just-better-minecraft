//! The transport boundary between client and server.
//!
//! A [`Link`] is one end of a bidirectional message pipe: the client's end
//! sends [`ClientMsg`](crate::protocol::ClientMsg)s and receives
//! [`ServerMsg`](crate::protocol::ServerMsg)s, and the server's end is the
//! mirror image. In v1 both ends live in one process
//! ([`in_process_pair`]); a network socket can implement the same trait
//! later without either side changing.
//!
//! Semantics:
//!
//! - `try_recv` never blocks: it returns `Ok(None)` when nothing is queued.
//! - [`LinkClosed`] is returned only once the other end has been dropped
//!   and nothing is left to read; messages that were already queued are
//!   still delivered first. The same goes for `send` once the other end is
//!   gone.
//!
//! Invariants: core spawns no threads and does no I/O here — the pair is
//! built on `std::sync::mpsc` and nothing else.

use std::sync::mpsc;

use crate::protocol::{ClientMsg, ServerMsg};

/// Error from a link whose other end is gone and whose queue has drained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkClosed;

impl std::fmt::Display for LinkClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("link closed by the other end")
    }
}

impl std::error::Error for LinkClosed {}

/// One end of a client-server message pipe. See the [module docs](self).
pub trait Link<Out, In>: Send {
    /// Queue `msg` for the other end. Errors only when the other end is
    /// gone.
    fn send(&mut self, msg: Out) -> Result<(), LinkClosed>;

    /// Take the next queued message, or `Ok(None)` if the queue is empty.
    /// Never blocks.
    fn try_recv(&mut self) -> Result<Option<In>, LinkClosed>;
}

/// The client's end of a link: sends `ClientMsg`, receives `ServerMsg`.
pub type ClientLink = Box<dyn Link<ClientMsg, ServerMsg>>;
/// The server's end of a link for one client: sends `ServerMsg`, receives
/// `ClientMsg`.
pub type ServerLink = Box<dyn Link<ServerMsg, ClientMsg>>;

/// One side of an in-process pair: a sender and the opposite side's queue.
struct LinkEnd<Out, In> {
    out: mpsc::Sender<Out>,
    inbox: mpsc::Receiver<In>,
}

impl<Out, In> Link<Out, In> for LinkEnd<Out, In>
where
    Out: Send,
    In: Send,
{
    fn send(&mut self, msg: Out) -> Result<(), LinkClosed> {
        self.out.send(msg).map_err(|_| LinkClosed)
    }

    fn try_recv(&mut self) -> Result<Option<In>, LinkClosed> {
        match self.inbox.try_recv() {
            Ok(msg) => Ok(Some(msg)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err(LinkClosed),
        }
    }
}

/// Create a connected pair of in-process links: the client's end and the
/// server's end. Unbounded, order-preserving, no threads spawned.
pub fn in_process_pair() -> (ClientLink, ServerLink) {
    let (client_to_server, server_inbox) = mpsc::channel();
    let (server_to_client, client_inbox) = mpsc::channel();
    let client: ClientLink = Box::new(LinkEnd {
        out: client_to_server,
        inbox: client_inbox,
    });
    let server: ServerLink = Box::new(LinkEnd {
        out: server_to_client,
        inbox: server_inbox,
    });
    (client, server)
}

#[cfg(test)]
mod tests {
    use super::{LinkClosed, in_process_pair};
    use crate::protocol::{ClientMsg, ServerMsg};
    use std::thread;

    #[test]
    fn order_is_preserved_in_both_directions() {
        let (mut client, mut server) = in_process_pair();
        for nonce in 0..3 {
            client.send(ClientMsg::Ping { nonce }).unwrap();
        }
        for nonce in 0..3 {
            assert_eq!(server.try_recv(), Ok(Some(ClientMsg::Ping { nonce })));
        }
        for tick in 0..3 {
            server.send(ServerMsg::Tick { tick }).unwrap();
        }
        for tick in 0..3 {
            assert_eq!(client.try_recv(), Ok(Some(ServerMsg::Tick { tick })));
        }
    }

    #[test]
    fn try_recv_on_an_empty_link_returns_none() {
        let (mut client, mut server) = in_process_pair();
        assert_eq!(client.try_recv(), Ok(None));
        assert_eq!(server.try_recv(), Ok(None));
    }

    #[test]
    fn dropping_one_end_closes_the_other_after_draining() {
        let (mut client, server) = in_process_pair();
        let mut server = server;
        server.send(ServerMsg::Tick { tick: 1 }).unwrap();
        server.send(ServerMsg::Tick { tick: 2 }).unwrap();
        drop(server);
        // Queued messages are still delivered first.
        assert_eq!(client.try_recv(), Ok(Some(ServerMsg::Tick { tick: 1 })));
        assert_eq!(client.try_recv(), Ok(Some(ServerMsg::Tick { tick: 2 })));
        assert_eq!(client.try_recv(), Err(LinkClosed));
        // And sends now fail too.
        assert_eq!(client.send(ClientMsg::Disconnect), Err(LinkClosed));
    }

    #[test]
    fn the_link_can_be_sent_across_a_thread() {
        let (client, server) = in_process_pair();
        let mut server = server;
        let handle = thread::spawn(move || {
            let mut client = client;
            client.send(ClientMsg::Ping { nonce: 42 }).unwrap();
            client
        });
        let mut client = handle.join().unwrap();
        assert_eq!(server.try_recv(), Ok(Some(ClientMsg::Ping { nonce: 42 })));
        // Still usable in both directions afterwards.
        server
            .send(ServerMsg::Pong {
                nonce: 42,
                server_tick: 0,
            })
            .unwrap();
        assert_eq!(
            client.try_recv(),
            Ok(Some(ServerMsg::Pong {
                nonce: 42,
                server_tick: 0
            }))
        );
    }
}
