//! The authoritative simulation: worldgen, lighting, storage, and the server
//! side of the protocol. Talks to the client only through `jbm-core`
//! protocol messages.

pub mod run;
pub mod server;
pub mod tick;

pub use run::ServerHandle;
pub use server::{PlayerInput, PlayerView, Server};
pub use tick::TickClock;
