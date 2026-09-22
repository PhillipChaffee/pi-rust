//! The client's option and subscription vocabulary, ported from upstream
//! `src/types.ts`.

use std::rc::Rc;

use pi_chord::future::LocalBoxFuture;
use pi_chord::types::ServiceSubscriptionSnapshot;
use pi_protocol::{RpcTarget, SessionTarget};

use crate::client::{ClientCore, SubscriptionShared};
use crate::errors::ClientError;
use crate::transport::ByteTransportFactory;

/// The connection lifecycle state, upstream's `ConnectionState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// No usable transport; the resting state after any failure.
    Disconnected,
    /// A transport is opening and the client hello has not been answered
    /// yet.
    Connecting,
    /// The handshake completed.
    Connected,
}

impl ConnectionState {
    /// The name upstream reports in state-change messages, the spelling
    /// tests assert on.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disconnected => "disconnected",
            Self::Connecting => "connecting",
            Self::Connected => "connected",
        }
    }
}

/// One state transition report, upstream's `ConnectionStateChange`.
#[derive(Debug, Clone)]
pub struct ConnectionStateChange {
    /// The state the connection moved to.
    pub state: ConnectionState,
    /// Why the connection left `Connecting`/`Connected`; absent when the
    /// move into `Connecting` itself is reported.
    pub error: Option<ClientError>,
}

/// Removes one listener, upstream's `Unsubscribe`.
pub type Unsubscribe = Box<dyn Fn()>;

/// Reports subscriber failures without letting them corrupt client state,
/// upstream's `ListenerErrorHandler`.
pub type ListenerErrorHandler = Rc<dyn Fn(&ClientError)>;

/// Observes connection state transitions, upstream's
/// `(change: ConnectionStateChange) => void` listener.
pub type ConnectionStateListener = Rc<dyn Fn(&ConnectionStateChange)>;

/// Observes the selected session route, upstream's `AttachmentChangeListener`.
pub type AttachmentChangeListener = Rc<dyn Fn(Option<&SessionTarget>)>;

/// One live service subscription, upstream's `ServiceSubscription`.
///
/// Updates buffer until [`start`](Self::start) is called, so the caller can
/// install the snapshot first; every accessor reads the state the
/// subscription held when the server accepted it.
#[derive(Clone)]
pub struct ServiceSubscription {
    shared: Rc<SubscriptionShared>,
}

impl ServiceSubscription {
    pub(crate) const fn new(shared: Rc<SubscriptionShared>) -> Self {
        Self { shared }
    }

    /// The correlation id the subscription requests and cancels carry.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.shared.id
    }

    /// The routed target the subscription fences to.
    #[must_use]
    pub fn target(&self) -> &RpcTarget {
        &self.shared.target
    }

    /// The snapshot the server took when it accepted the subscription.
    #[must_use]
    pub fn snapshot(&self) -> &ServiceSubscriptionSnapshot {
        &self.shared.snapshot
    }

    /// Begins ordered update delivery after the caller has installed the
    /// snapshot. Repeat calls and calls after [`dispose`](Self::dispose)
    /// are no-ops.
    pub fn start(&self) {
        ClientCore::start_subscription(&self.shared.core, &self.shared.active);
    }

    /// Closes the subscription: drops queued updates and, while the
    /// client is still connected to the subscription's target, sends the
    /// unsubscribe call. The unsubscribe frame is sent before the returned
    /// future resolves.
    ///
    /// # Errors
    /// Upstream rejects when the unsubscribe call fails; queued updates are
    /// dropped either way.
    #[must_use]
    pub fn dispose(&self) -> LocalBoxFuture<Result<(), ClientError>> {
        ClientCore::dispose_subscription(&self.shared.core, &self.shared)
    }
}

impl std::fmt::Debug for ServiceSubscription {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ServiceSubscription")
            .field("id", &self.shared.id)
            .field("target", &self.shared.target)
            .finish_non_exhaustive()
    }
}

/// The client's construction options, upstream's `ClientOptions`.
pub struct ClientOptions {
    /// Creates the transport every connection attempt opens.
    pub transport_factory: ByteTransportFactory,
    /// Logical server identity expected at the physical endpoint.
    pub server_id: String,
    /// Maximum frame length; defaults to the protocol default.
    pub max_frame_length: Option<usize>,
    /// Reports subscriber failures without allowing them to corrupt client
    /// state.
    pub on_listener_error: Option<ListenerErrorHandler>,
}

impl std::fmt::Debug for ClientOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClientOptions")
            .field("server_id", &self.server_id)
            .field("max_frame_length", &self.max_frame_length)
            .finish_non_exhaustive()
    }
}
