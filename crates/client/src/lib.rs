//! Rust port of `packages/client` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The transport-neutral client half of the remote-session wire: it opens
//! a pluggable [`ByteTransport`](transport), performs the versioned
//! handshake, then correlates requests, cancellations, and service
//! subscriptions over framed bytes (`pi-protocol` owns the framing and the
//! envelope schemas; the service meaning of the opaque payloads belongs to
//! `pi-chord`). Server-side routing mirrors it in `pi-server`.
//!
//! Upstream is single-threaded Node: transport events, promise callbacks,
//! and subscriber delivery all run on one event loop in invocation order.
//! The port keeps that shape — the client is a single-threaded object
//! (`Rc`/`RefCell` internally, `!Send`) whose transport tasks run as local
//! tasks, so every surface expects the caller's current-thread tokio
//! runtime behind a `LocalSet`. Three upstream contracts are restated
//! because JavaScript provides them at runtime and Rust cannot:
//!
//! - Promise resolvers behind the handshake and every request collapse to
//!   [`tokio::sync::oneshot`] waiters; per-request `AbortSignal`
//!   listeners become a `select!` over the chord signal.
//! - Upstream's error classes (`ServerError`, `DisconnectedError`,
//!   `ClientDisposedError`, re-raised `ProtocolValidationError`, and
//!   cause-walking errno probes) restate as one owned taxonomy,
//!   [`errors::ClientError`], whose `source` chain carries what upstream's
//!   `Error.cause` chain carried.
//! - Upstream's promise-chained subscriber delivery serializes async
//!   listeners; the chord port fixed listeners as synchronous closures, so
//!   ordered delivery is the loop that calls them in decode order.
//!
//! The Unix transports and discovery live in [`unix`], mirroring
//! upstream's `./unix` subpath export; the crate root stays the single
//! import point, mirroring upstream's `src/index.ts` re-export list.

#![forbid(unsafe_code)]

pub mod client;
pub mod connection;
pub mod errors;
pub mod transport;
pub mod types;
#[cfg(unix)]
pub mod unix;

pub use client::{Client, ClientServiceTransport, create_client_service_transport};
pub use errors::{ClientError, is_error_code, to_disconnected_error};
pub use transport::{ByteTransport, ByteTransportFactory, ByteTransportHandlers};
pub use types::{
    AttachmentChangeListener, ClientOptions, ConnectionState, ConnectionStateChange,
    ConnectionStateListener, ListenerErrorHandler, ServiceSubscription, Unsubscribe,
};
