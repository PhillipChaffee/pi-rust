//! The connection SPI, ported from upstream `src/connection.ts`.
//!
//! Upstream's `ConnectionState` is a plain object the server mutates and the
//! handler closures capture; the port restates it as the granular-cell
//! [`ConnectionCore`] so independent fields stay independently borrowable
//! while the async paths run.
//!
//! The internals are crate-wide by design; `unreachable_pub` owns the
//! narrower `pub` surface the root re-exports carry.
#![allow(
    clippy::redundant_pub_crate,
    reason = "the internals are crate-wide by design; unreachable_pub owns the narrower `pub` surface the root re-exports carry"
)]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use pi_chord::future::LocalBoxFuture;
use pi_chord::services::state_codec::ServiceStateEncoder;
use pi_protocol::{ClientMessageDecoder, RpcTarget};

use crate::errors::Failure;

use pi_agent_core::harness::context::AbortController;

/// The inbound-chunk callback one handler drives, upstream's
/// `onData(chunk: Uint8Array)`.
pub type DataCallback = Rc<dyn Fn(&[u8])>;

/// The transport-closed callback one handler drives, upstream's `onClose()`.
pub type CloseCallback = Rc<dyn Fn()>;

/// The transport-failure callback one handler drives, upstream's
/// `onError(error: Error)`.
pub type ErrorCallback = Rc<dyn Fn(&Failure)>;

/// An established, authorized ordered byte connection, upstream's
/// `ByteConnection`.
pub trait ByteConnection {
    /// Whether the transport has observed closure, upstream's `closed`.
    fn closed(&self) -> bool;

    /// Write one chunk in arrival order, upstream's `send`.
    ///
    /// # Errors
    /// A closed transport, an exceeded pending-byte cap, or whatever the
    /// transport raises while writing.
    fn send(&self, chunk: Vec<u8>) -> LocalBoxFuture<Result<(), Failure>>;

    /// Close the transport after every pending write, optionally emitting
    /// `final_chunk` last, upstream's `close(finalChunk?)`.
    ///
    /// # Errors
    /// Whatever the transport raises while closing.
    fn close(&self, final_chunk: Option<Vec<u8>>) -> LocalBoxFuture<Result<(), Failure>>;
}

/// The callbacks one accepted connection drives, upstream's
/// `ByteConnectionHandler`.
pub struct ByteConnectionHandler {
    /// Delivers one ordered inbound chunk, upstream's `onData`.
    pub on_data: DataCallback,
    /// Runs once the transport closed, upstream's `onClose`.
    pub on_close: CloseCallback,
    /// Reports a transport failure, upstream's `onError`.
    pub on_error: ErrorCallback,
}

impl std::fmt::Debug for ByteConnectionHandler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ByteConnectionHandler").finish()
    }
}

/// Supplies established byte connections after any required transport
/// authentication, upstream's `ByteConnectionAcceptor`.
pub type ByteConnectionAcceptor = Rc<dyn Fn(Rc<dyn ByteConnection>) -> ByteConnectionHandler>;

/// The lifecycle stage one presentation connection sits in, upstream's
/// `ConnectionStage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConnectionStage {
    /// The client hello has not arrived.
    AwaitingHello,
    /// The handshake is in flight.
    Handshaking,
    /// The handshake completed and requests route.
    Ready,
    /// Failing: the terminal frame is queued behind pending output.
    Closing,
    /// Failed or shut down.
    Closed,
}

/// The identity token one presentation carries into the router, upstream's
/// `client: object` reference equality.
#[derive(Clone)]
pub(crate) struct ClientToken(Rc<()>);

impl ClientToken {
    pub(crate) fn new() -> Self {
        Self(Rc::new(()))
    }

    /// The map key derived from the identity; the token `Rc` is held by
    /// every mapped value, so the address cannot be reused while mapped.
    pub(crate) fn key(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }
}

/// One active request's cancellation surface and routed target, upstream's
/// `activeRequests` entry.
pub(crate) struct ActiveRequest {
    pub(crate) controller: AbortController,
    pub(crate) target: RpcTarget,
}

/// One presentation connection's live state, upstream's `ConnectionState`.
pub(crate) struct ConnectionCore {
    /// The router identity the session routes key on.
    pub(crate) token: ClientToken,
    /// The transport bytes flow over.
    pub(crate) connection: Rc<dyn ByteConnection>,
    /// The incremental client-message decoder, upstream's `decoder`.
    pub(crate) decoder: RefCell<ClientMessageDecoder>,
    /// The per-subscription state encoders, upstream's
    /// `serviceStateEncoders`.
    pub(crate) service_state_encoders: RefCell<HashMap<String, ServiceStateEncoder>>,
    /// The lifecycle stage, upstream's `stage`.
    pub(crate) stage: Cell<ConnectionStage>,
    /// Whether the presentation disconnected, upstream's `disconnected`.
    pub(crate) disconnected: Cell<bool>,
    /// The in-flight handshake's settle latch, upstream's `handshake`
    /// promise; queued messages wait behind it.
    pub(crate) handshake: RefCell<Option<Rc<crate::latch::Latch<()>>>>,
    /// The handshake timeout's task handle, upstream's `handshakeTimeout`;
    /// `None` between spawn and the state's publication.
    pub(crate) handshake_timeout: RefCell<Option<tokio::task::JoinHandle<()>>>,
    /// The server-scoped service attachment, upstream's `serverServices`.
    pub(crate) server_services:
        RefCell<Option<Rc<dyn crate::types::RoutedServerServiceAttachment>>>,
    /// The active requests keyed by correlation id, upstream's
    /// `activeRequests`.
    pub(crate) active_requests: RefCell<HashMap<String, Rc<ActiveRequest>>>,
}

impl ConnectionCore {
    /// Clears the handshake timeout, upstream's `clearTimeout(state.
    /// handshakeTimeout)`.
    pub(crate) fn clear_handshake_timeout(&self) {
        if let Some(handle) = self.handshake_timeout.borrow_mut().take() {
            handle.abort();
        }
    }
}

/// Whether the connection is beyond dispatch, upstream's
/// `isTerminalConnection`.
pub(crate) const fn is_terminal_connection(state: &ConnectionCore) -> bool {
    state.disconnected.get()
        || matches!(
            state.stage.get(),
            ConnectionStage::Closing | ConnectionStage::Closed
        )
}
