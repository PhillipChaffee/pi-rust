//! Rust port of `packages/server` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The transport-neutral server half of the remote-session wire: a
//! [`ServerListener`] supplies established [`ByteConnection`]s, the
//! [`Server`] handshakes each one (client hello → server hello or
//! `hello_error`), then routes requests — server-wide through the host's
//! service attachment, session-scoped through the router to a
//! host-provided [`ServerHost`] — and publishes attachment changes and
//! subscription updates out of band (`pi-protocol` owns the framing and the
//! envelope schemas; the service meaning of the opaque payloads belongs to
//! `pi-chord`). The client-side mirror lives in `pi-client`.
//!
//! Upstream is single-threaded Node: connection events, promise chaining,
//! and per-client operation serialization all run on one event loop in
//! arrival order. The port keeps that shape — the server is a
//! single-threaded object (`Rc`/`RefCell` internally, `!Send`) whose
//! listener tasks and per-client operation chains run as local tasks, so
//! every surface expects the caller's current-thread tokio runtime behind a
//! `LocalSet`. Four upstream contracts restate because JavaScript provides
//! them at runtime and Rust cannot:
//!
//! - Per-client promise chains (`runForClient`) become one driver task per
//!   client running queued operations FIFO; a settled operation never
//!   poisons the next, upstream's `previous.catch(() => {})`.
//! - Reused settled promises (a release latch, an in-flight open, the
//!   `closed` promise) become shared-latch waiters that hand late waiters the
//!   settled value.
//! - Upstream's error classes (`ServerError` and its five subclasses, chord's
//!   `RemoteServiceError`, protocol's `ProtocolValidationError`, and the
//!   `AggregateError` family) restate as one owned [`Failure`] taxonomy;
//!   identity-carrying opaque errors ride an `Rc` so verbatim rethrows
//!   (upstream's `rejects.toBe`) stay observable.
//! - Upstream's structural identity — the wire snapshot/update/call objects
//!   *are* their JSON — becomes the `pi-chord` serializers
//!   (`wire_snapshot_to_json`, `wire_update_to_json`,
//!   `service_call_to_json`) the response paths encode through.
//!
//! The Unix listener transport, socket-path derivation, and the composed
//! preset live in [`unix`], mirroring upstream's `./unix` subpath export;
//! the test doubles (host, harness, wire client) live in [`testing`],
//! mirroring upstream's shipped `./testing` subpath. Windows is out of scope
//! for this effort (map ticket "Decide the Rust stack"), so [`unix`]
//! compiles on Unix targets only.

#![forbid(unsafe_code)]

mod connection;
mod errors;
mod latch;
mod listener;
mod server;
mod session_router;
pub mod testing;
mod types;
#[cfg(unix)]
pub mod unix;

pub use connection::{
    ByteConnection, ByteConnectionAcceptor, ByteConnectionHandler, CloseCallback, DataCallback,
    ErrorCallback,
};
pub use errors::{Failure, INTERNAL_SERVER_ERROR_MESSAGE, ServerError, ServerOperationErrorCode};
pub use listener::ServerListener;
pub use server::Server;
pub use types::{
    ErrorObserver, HasSessionId, RoutedServerPresentation, RoutedServerServiceAttachment,
    RoutedServerServiceHost, RoutedSessionAttachment, RoutedSessionHandle, ServerHost,
    ServerOptions, ServicePublisher,
};
