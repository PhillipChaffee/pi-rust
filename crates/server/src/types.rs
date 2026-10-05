//! The host-facing contract types, ported from upstream `src/types.ts`.
//!
//! `ready` is a crate-wide helper; this module's public surface is the
//! re-export list.
#![allow(
    clippy::redundant_pub_crate,
    reason = "`ready` is a crate-wide helper; this module's public surface is the re-export list"
)]

use std::rc::Rc;

use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::types::{JsonValue, ServiceCall, ServiceProviderUpdate};

use crate::errors::Failure;
use crate::listener::ServerListener;
use pi_agent_core::harness::context::Context;

/// The observer one error report reaches, upstream's `onError?: (error:
/// Error) => void` and its listener-side twin.
pub type ErrorObserver = Rc<dyn Fn(&Failure)>;

/// The construction options of a [`Server`](crate::Server), upstream's
/// `ServerOptions`.
pub struct ServerOptions {
    /// The transport listeners the server starts and closes, upstream's
    /// `listeners`.
    pub listeners: Vec<Rc<dyn ServerListener>>,
    /// Stable logical server identity supplied by the installation or
    /// profile, upstream's `serverId`; a canonical lowercase UUIDv4.
    pub server_id: String,
    /// The framed-byte ceiling every encoder and decoder shares; defaults to
    /// protocol's 16 MiB, upstream's `maxFrameLength`.
    pub max_frame_length: Option<usize>,
    /// The handshake budget before the connection fails; defaults to 5,000
    /// ms, upstream's `handshakeTimeoutMs`.
    pub handshake_timeout_ms: Option<u64>,
    /// Runs whenever the connected-presentation count changes, upstream's
    /// `onConnectionCountChanged`.
    pub on_connection_count_changed: Option<Rc<dyn Fn(usize)>>,
    /// Reports every error the server routes but does not answer, upstream's
    /// `onError`.
    pub on_error: Option<ErrorObserver>,
}

impl std::fmt::Debug for ServerOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ServerOptions")
            .field("listeners", &self.listeners.len())
            .field("server_id", &self.server_id)
            .field("max_frame_length", &self.max_frame_length)
            .field("handshake_timeout_ms", &self.handshake_timeout_ms)
            .finish_non_exhaustive()
    }
}

/// The publisher a service attachment invokes per provider update.
///
/// Upstream's `publish: (subscriptionId, update, context) => MaybePromise<void>`:
/// a failure rejects the invoking service call, upstream's promise rejection.
pub type ServicePublisher =
    Rc<dyn Fn(&str, &ServiceProviderUpdate, &Context) -> LocalBoxFuture<Result<(), Failure>>>;

/// One presentation connection's live capability for a hosted Session,
/// upstream's `RoutedSessionAttachment`.
pub trait RoutedSessionAttachment {
    /// Route one contract-agnostic service operation to the attached Session
    /// endpoint, upstream's `invokeService`; `None` is upstream's `undefined`
    /// return, while `Some(null)` is a JSON `null` result.
    ///
    /// # Errors
    /// Whatever the Session endpoint raises; the server classifies it.
    fn invoke_service(
        &self,
        call: ServiceCall,
        publish: ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>>;

    /// Release the capability, upstream's `release`.
    ///
    /// # Errors
    /// Whatever the release raises.
    fn release(&self, context: Context) -> LocalBoxFuture<Result<(), Failure>>;
}

/// Presentation-scoped routing capabilities available to server service
/// implementations, upstream's `RoutedServerPresentation`.
pub trait RoutedServerPresentation {
    /// Attach this presentation to one hosted Session, upstream's
    /// `attachSession`.
    ///
    /// # Errors
    /// `server_draining` while the server drains, or whatever the routing
    /// raises.
    fn attach_session(
        &self,
        session_id: &str,
        context: Context,
    ) -> LocalBoxFuture<Result<(), Failure>>;

    /// Detach this presentation from its Session, upstream's `detachSession`.
    ///
    /// # Errors
    /// Whatever the release raises.
    fn detach_session(&self, context: Context) -> LocalBoxFuture<Result<(), Failure>>;

    /// Release routed attachments and handles before the application deletes
    /// durable metadata, upstream's `prepareSessionRemoval`.
    ///
    /// # Errors
    /// Whatever the release or close raises.
    fn prepare_session_removal(
        &self,
        session_id: &str,
        context: Context,
    ) -> LocalBoxFuture<Result<(), Failure>>;
}

/// One connection's server-scoped service endpoint, upstream's
/// `RoutedServerServiceAttachment`.
pub trait RoutedServerServiceAttachment {
    /// Route one contract-agnostic service operation to the server-scoped
    /// endpoint, upstream's `invokeService`.
    ///
    /// # Errors
    /// Whatever the endpoint raises; the server classifies it.
    fn invoke_service(
        &self,
        call: ServiceCall,
        publish: ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>>;

    /// Release the endpoint, upstream's `release`.
    ///
    /// # Errors
    /// Whatever the release raises.
    fn release(&self, context: Context) -> LocalBoxFuture<Result<(), Failure>>;
}

/// The host capability that hands each presentation its server-scoped
/// endpoint, upstream's `RoutedServerServiceHost`.
pub trait RoutedServerServiceHost {
    /// Attach one presentation, upstream's `attachClient`.
    ///
    /// # Errors
    /// Whatever the host raises; the server classifies it.
    fn attach_client(
        &self,
        presentation: Rc<dyn RoutedServerPresentation>,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedServerServiceAttachment>, Failure>>;
}

/// A process-safe handle that acquires presentation-scoped Session
/// capabilities, upstream's `RoutedSessionHandle`.
pub trait RoutedSessionHandle {
    /// Acquire the presentation-scoped capability, upstream's `attachClient`.
    ///
    /// # Errors
    /// Whatever the handle raises; the server classifies it.
    fn attach_client(
        &self,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionAttachment>, Failure>>;

    /// Resolves with the unexpected-termination error, or `Some`/`None`
    /// settled after an expected close, upstream's `terminated?: Promise<Error
    /// | undefined>`; `None` models the absent field.
    fn terminated(&self) -> Option<LocalBoxFuture<Option<Failure>>>;

    /// Close the handle, upstream's `close`.
    ///
    /// # Errors
    /// Whatever the close raises.
    fn close(&self, context: Context) -> LocalBoxFuture<Result<(), Failure>>;
}

/// The metadata shape server routing reads, upstream's `TMetadata extends
/// SessionMetadata`: routing reads only the durable session id out of it, so
/// the port carries that one access as the bound.
pub trait HasSessionId {
    /// The durable session id, upstream's `metadata.id`.
    fn session_id(&self) -> &str;
}

/// Application capabilities used by server-wide management and Session
/// routing, upstream's `ServerHost<TMetadata>`.
pub trait ServerHost {
    /// The metadata the host resolves, upstream's `TMetadata`.
    type Metadata: HasSessionId + 'static;

    /// The server-scoped service endpoint factory, upstream's
    /// `serverServices`.
    fn server_services(&self) -> Rc<dyn RoutedServerServiceHost>;

    /// Resolve one durable Session ID, upstream's `resolveSession`; the `Rc`
    /// preserves object identity through `open_session`, upstream's
    /// same-object contract.
    ///
    /// # Errors
    /// A bounded routing error (`session_not_found`,
    /// `session_ambiguous`, ...) the server answers with its code.
    fn resolve_session(
        &self,
        session_id: &str,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<Self::Metadata>, Failure>>;

    /// Open one resolved Session, upstream's `openSession`.
    ///
    /// # Errors
    /// Whatever the host raises; the server classifies it.
    fn open_session(
        &self,
        metadata: Rc<Self::Metadata>,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionHandle>, Failure>>;
}

/// Boxes an immediately-available value into the crate's future shape, the
/// `Promise.resolve(value)` restatement host doubles use.
pub(crate) fn ready<T: 'static>(value: T) -> LocalBoxFuture<T> {
    boxed(std::future::ready(value))
}

/// The durable Session metadata satisfies the routing read directly.
impl HasSessionId for pi_agent_core::harness::session::types::SessionMetadata {
    fn session_id(&self) -> &str {
        &self.id
    }
}
