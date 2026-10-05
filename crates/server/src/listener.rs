//! The listener SPI, ported from upstream `src/listener.ts`.
//!
//! Upstream's `src/index.ts` re-exports only this module's `ServerListener`;
//! the connection types its signature references live in the crate's private
//! connection module and re-export at the root so implementors can name
//! them.

use pi_chord::future::LocalBoxFuture;

use crate::connection::ByteConnectionAcceptor;
use crate::errors::Failure;

/// Supplies established byte connections after any required transport
/// authentication, upstream's `ServerListener`.
pub trait ServerListener {
    /// Starts listening and passes authorized connections to `accept`,
    /// upstream's `start`.
    ///
    /// # Errors
    /// Whatever the transport raises while starting; the server closes every
    /// listener it already started and abandons startup.
    fn start(&self, accept: ByteConnectionAcceptor) -> LocalBoxFuture<Result<(), Failure>>;

    /// Stops listening, upstream's `close`.
    ///
    /// # Errors
    /// Whatever the transport raises while closing; the server surfaces it
    /// through `close` and `closed`.
    fn close(&self) -> LocalBoxFuture<Result<(), Failure>>;
}
