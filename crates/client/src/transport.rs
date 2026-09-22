//! The pluggable byte-transport seam, ported from upstream
//! `src/transport.ts`.

use std::rc::Rc;

use pi_chord::future::LocalBoxFuture;

use crate::errors::ClientError;

/// The outbound half of one byte transport, upstream's `ByteTransport`.
///
/// Implementations deliver sends in invocation order and keep repeated
/// closes harmless.
pub trait ByteTransport {
    /// Sends one byte chunk. Calls must be delivered in invocation order.
    ///
    /// The future resolves once the transport accepted the chunk, not
    /// necessarily once the peer received it; a send failure is terminal for
    /// the connection.
    fn send(&self, chunk: Vec<u8>) -> LocalBoxFuture<Result<(), ClientError>>;

    /// Closes the transport. Implementations must make repeated calls
    /// harmless.
    fn close(&self);
}

/// Delivers one inbound chunk, the transport's `onData` handler.
pub type TransportDataHandler = Rc<dyn Fn(&[u8])>;

/// Reports one terminal transport failure, the transport's `onError`
/// handler.
pub type TransportErrorHandler = Rc<dyn Fn(&ClientError)>;

/// The inbound handlers a transport drives, upstream's
/// `ByteTransportHandlers`.
///
/// Handlers run on the single thread the client lives on; exactly one
/// terminal handler is expected per transport.
#[derive(Clone)]
pub struct ByteTransportHandlers {
    /// Delivers an arbitrary inbound byte chunk.
    pub on_data: TransportDataHandler,
    /// Reports an orderly terminal close.
    pub on_close: Rc<dyn Fn()>,
    /// Reports a terminal transport failure.
    pub on_error: TransportErrorHandler,
}

impl std::fmt::Debug for ByteTransportHandlers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ByteTransportHandlers")
            .finish_non_exhaustive()
    }
}

/// Creates a fresh connected, authenticated transport, upstream's
/// `ByteTransportFactory`.
pub type ByteTransportFactory =
    Rc<dyn Fn(ByteTransportHandlers) -> LocalBoxFuture<Result<Rc<dyn ByteTransport>, ClientError>>>;
