//! The client, ported from upstream `src/client.ts`.
//!
//! Upstream drives the client through promise resolvers, per-request abort
//! listeners, and promise-chained subscriber delivery; the port restates
//! each on the single-threaded substrate the chord port already fixed:
//! oneshot waiters behind the handshake and request promises, the chord
//! [`AbortSignal`] behind the abort
//! listeners, and synchronous ordered listener calls where upstream's
//! promise chain serialized them.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use pi_chord::context::{AbortReason, AbortSignal, Context, background_context};
use pi_chord::errors::RemoteServiceErrorCode;
use pi_chord::errors::{ChordError, RemoteServiceError};
use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::services::state_codec::{ServiceStateDecoder, create_service_state_decoder};
use pi_chord::services::wire::{
    create_service_catalogue_call, create_service_subscribe_call, create_service_unsubscribe_call,
    object as json_object, parse_service_catalogue, parse_wire_service_provider_update,
    parse_wire_service_subscription_snapshot,
};
use pi_chord::types::{
    JsonValue, RemoteServiceTransport, ServiceCall, ServiceCatalogueEntry, ServiceMode,
    ServiceProviderListener, ServiceProviderUpdate, ServiceSubscription as ChordSubscription,
    ServiceSubscriptionSnapshot,
};
use pi_protocol::{
    CancelEnvelope, ClientMessage, DEFAULT_MAX_FRAME_LENGTH, FrameDecoderOptions,
    ProtocolValidationError, RequestEnvelope, ResponseEnvelope, RpcTarget, ServerHello,
    SessionTarget, is_server_id,
};
use tokio::sync::oneshot;

use crate::connection::{Connection, ConnectionOptions, ServerPayload, panic_message};
use crate::errors::ClientError;
use crate::types::{
    AttachmentChangeListener, ClientOptions, ConnectionState, ConnectionStateChange,
    ConnectionStateListener, ListenerErrorHandler, ServiceSubscription, Unsubscribe,
};

/// One active subscription's delivery state, upstream's
/// `ActiveServiceListener`.
pub(crate) struct ActiveServiceListener {
    /// The subscriber's update listener.
    pub(crate) listener: Rc<dyn Fn(&ServiceProviderUpdate)>,
    /// The per-subscription operation decoder, upstream's
    /// `createServiceStateDecoder()`.
    pub(crate) decoder: ServiceStateDecoder,
    /// Wire updates that arrived before the snapshot; decoded into
    /// [`Self::queued`] on hydration.
    pub(crate) queued_wire_updates: Vec<JsonValue>,
    /// Decoded updates held until the subscription starts.
    pub(crate) queued: Vec<ServiceProviderUpdate>,
    /// Whether the snapshot arrived and the wire queue drained.
    pub(crate) hydrated: bool,
    /// Whether delivery has begun.
    pub(crate) ready: bool,
}

/// Settles one pending request's caller future with the response,
/// applying the subscription transform when one is armed.
type PendingSettle = Box<dyn FnOnce(&Rc<ClientCore>, &ResponseEnvelope)>;

/// Settles one pending request's caller future with a failure, upstream's
/// `pending.reject`.
type PendingReject = Box<dyn FnOnce(&ClientError)>;

/// The oneshot sender one pending request settles through, held until
/// exactly one settle or reject consumes it.
type PendingSender<V> = Rc<RefCell<Option<oneshot::Sender<Result<V, ClientError>>>>>;

/// One correlated in-flight request, upstream's `PendingRequest` with its
/// resolve wrapper folded in: the settle closure runs the subscription
/// arm's decode at settlement time, and a transform failure fails the
/// connection.
pub(crate) struct PendingRequest {
    settle: PendingSettle,
    reject: PendingReject,
}

impl PendingRequest {
    /// Assembles the pending from its settle closure and the rejection
    /// side of its oneshot.
    fn new<V: 'static>(settle: PendingSettle, sender_cell: &PendingSender<V>) -> Self {
        Self {
            settle,
            reject: reject_closure(sender_cell),
        }
    }
}

/// The rejection side of one pending request's channel: takes the sender
/// once and settles the caller's future with the failure.
fn reject_closure<V: 'static>(sender_cell: &PendingSender<V>) -> Box<dyn FnOnce(&ClientError)> {
    let sender_cell = sender_cell.clone();
    Box::new(move |error: &ClientError| {
        if let Some(sender) = sender_cell.borrow_mut().take() {
            let _ = sender.send(Err(error.clone()));
        }
    })
}

/// The subscription handle's shared state, shared between the client's
/// listener table and the handle the caller holds.
pub(crate) struct SubscriptionShared {
    /// The correlation id, upstream's `service-<n>` sequence.
    pub(crate) id: String,
    /// The routed target.
    pub(crate) target: RpcTarget,
    /// The accepted snapshot.
    pub(crate) snapshot: ServiceSubscriptionSnapshot,
    /// The client core, for the unsubscribe call on dispose.
    pub(crate) core: Rc<ClientCore>,
    /// The live listener entry the client's table holds.
    pub(crate) active: Rc<RefCell<ActiveServiceListener>>,
    /// Whether dispose already ran.
    pub(crate) disposed: Cell<bool>,
}

/// The client's interior state, upstream's `Client` private fields.
pub(crate) struct ClientCore {
    /// The logical server identity every call fences to.
    pub(crate) server_id: String,
    /// Reports subscriber failures without letting them corrupt client
    /// state.
    pub(crate) on_listener_error: Option<ListenerErrorHandler>,
    pub(crate) connection: Connection,
    pub(crate) pending: RefCell<HashMap<String, PendingRequest>>,
    pub(crate) state_listeners: RefCell<Vec<ConnectionStateListener>>,
    pub(crate) attachment_listeners: RefCell<Vec<AttachmentChangeListener>>,
    pub(crate) service_listeners: RefCell<HashMap<String, Rc<RefCell<ActiveServiceListener>>>>,
    pub(crate) request_sequence: Cell<u64>,
    pub(crate) service_sequence: Cell<u64>,
    pub(crate) hello: RefCell<Option<ServerHello>>,
    pub(crate) attachment: RefCell<Option<SessionTarget>>,
    pub(crate) disposed: Cell<bool>,
}

impl ClientCore {
    fn handle_connection_state_change(core: &Rc<Self>, change: &ConnectionStateChange) {
        if change.state == ConnectionState::Disconnected {
            *core.hello.borrow_mut() = None;
            Self::set_attachment(core, None);
            let error = change
                .error
                .clone()
                .unwrap_or_else(ClientError::disconnected);
            Self::reject_pending(core, &error);
            core.service_listeners.borrow_mut().clear();
        }
        for listener in core.state_listeners.borrow().to_vec() {
            let reported =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listener(change)));
            if let Err(payload) = reported {
                Self::report_listener_error(core, &ClientError::other(panic_message(&*payload)));
            }
        }
    }

    fn set_attachment(core: &Rc<Self>, attachment: Option<&SessionTarget>) {
        let previous = core.attachment.borrow().clone();
        let changed = match (&previous, attachment) {
            (Some(previous), Some(current)) => {
                previous.server_id != current.server_id
                    || previous.session_id != current.session_id
                    || previous.attachment_id != current.attachment_id
            }
            (None, None) => false,
            (Some(_), None) | (None, Some(_)) => true,
        };
        if !changed {
            return;
        }
        *core.attachment.borrow_mut() = attachment.cloned();
        for listener in core.attachment_listeners.borrow().to_vec() {
            let reported =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listener(attachment)));
            if let Err(payload) = reported {
                Self::report_listener_error(core, &ClientError::other(panic_message(&*payload)));
            }
        }
    }

    fn reject_pending(core: &Rc<Self>, error: &ClientError) {
        let requests: Vec<PendingRequest> = core
            .pending
            .borrow_mut()
            .drain()
            .map(|(_, request)| request)
            .collect();
        for request in requests {
            (request.reject)(error);
        }
    }

    fn deliver_update(
        core: &Rc<Self>,
        active: &Rc<RefCell<ActiveServiceListener>>,
        update: &ServiceProviderUpdate,
    ) {
        let listener = active.borrow().listener.clone();
        let delivered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listener(update)));
        if let Err(payload) = delivered {
            Self::report_listener_error(core, &ClientError::other(panic_message(&*payload)));
        }
    }

    fn report_listener_error(core: &Rc<Self>, error: &ClientError) {
        let Some(handler) = core.on_listener_error.clone() else {
            return;
        };
        // Diagnostics cannot affect protocol or transport state.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler(error)));
    }

    fn target_is_current(core: &Rc<Self>, target: &RpcTarget) -> bool {
        match target {
            RpcTarget::Server(target) => core
                .hello
                .borrow()
                .as_ref()
                .is_some_and(|hello| hello.server_id.as_str() == target.server_id.as_str()),
            RpcTarget::Session(target) => {
                core.attachment.borrow().as_ref().is_some_and(|attachment| {
                    attachment.server_id == target.server_id
                        && attachment.session_id == target.session_id
                        && attachment.attachment_id == target.attachment_id
                })
            }
        }
    }

    pub(crate) fn start_subscription(core: &Rc<Self>, active: &Rc<RefCell<ActiveServiceListener>>) {
        let queued = {
            let mut listener = active.borrow_mut();
            if listener.ready {
                return;
            }
            listener.ready = true;
            std::mem::take(&mut listener.queued)
        };
        for update in &queued {
            Self::deliver_update(core, active, update);
        }
    }

    /// The unsubscribe frame is sent before the returned future resolves;
    /// the awaiter settles when the server answers or the client
    /// disconnects.
    pub(crate) fn dispose_subscription(
        core: &Rc<Self>,
        shared: &Rc<SubscriptionShared>,
    ) -> LocalBoxFuture<Result<(), ClientError>> {
        if shared.disposed.get() {
            return boxed(std::future::ready(Ok(())));
        }
        shared.disposed.set(true);
        if subscription_registered(core, &shared.id, &shared.active) {
            core.service_listeners.borrow_mut().remove(&shared.id);
        }
        let unsubscribe = if core.connection.state() == ConnectionState::Connected
            && Self::target_is_current(core, &shared.target)
        {
            request_plain(
                core,
                &shared.target,
                &create_service_unsubscribe_call(&shared.id),
                None,
            )
        } else {
            boxed(std::future::ready(Ok(None)))
        };
        let active = Rc::clone(&shared.active);
        boxed(async move {
            let result = unsubscribe.await.map(|_| ());
            {
                let mut listener = active.borrow_mut();
                listener.queued_wire_updates.clear();
                listener.queued.clear();
            }
            result
        })
    }

    fn handle_message(core: &Rc<Self>, payload: ServerPayload) {
        match payload {
            ServerPayload::Attachment(envelope) => {
                if let Some(target) = &envelope.attachment
                    && target.server_id.as_str() != core.server_id
                {
                    core.connection
                        .fail(&ClientError::Protocol(ProtocolValidationError::new(
                            "Attachment update belongs to another server",
                        )));
                    return;
                }
                Self::set_attachment(core, envelope.attachment.as_ref());
            }
            ServerPayload::ServiceEvent(envelope) => {
                let Some(active) = core
                    .service_listeners
                    .borrow()
                    .get(&envelope.subscription_id)
                    .cloned()
                else {
                    return;
                };
                let hydrated = active.borrow().hydrated;
                if !hydrated {
                    active
                        .borrow_mut()
                        .queued_wire_updates
                        .push(envelope.update);
                    return;
                }
                let update = match decode_service_update(&active, &envelope.update) {
                    Ok(update) => update,
                    Err(error) => {
                        core.connection.fail(&error);
                        return;
                    }
                };
                if active.borrow().ready {
                    Self::deliver_update(core, &active, &update);
                } else {
                    active.borrow_mut().queued.push(update);
                }
            }
            ServerPayload::Response(response) => {
                let pending = core.pending.borrow_mut().remove(response_id(&response));
                match pending {
                    Some(pending) => (pending.settle)(core, &response),
                    None => {
                        core.connection
                            .fail(&ClientError::Protocol(ProtocolValidationError::new(
                                "Response has no matching request",
                            )));
                    }
                }
            }
        }
    }
}

/// The client half of the remote-session surface, upstream's `Client`.
#[derive(Clone)]
pub struct Client {
    core: Rc<ClientCore>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Client")
            .field("server_id", &self.core.server_id)
            .field("connection_state", &self.connection_state())
            .field("attachment", &self.attachment())
            .field("disposed", &self.disposed())
            .finish()
    }
}

impl Client {
    /// Builds a client over the given options.
    ///
    /// # Errors
    /// Upstream throws a `TypeError` for a non-canonical `serverId` or a
    /// `maxFrameLength` outside the encodable range; the port returns the
    /// same failures.
    pub fn new(options: ClientOptions) -> Result<Self, ClientError> {
        if !is_server_id(&options.server_id) {
            return Err(ClientError::other(
                "serverId must be a canonical lowercase UUIDv4",
            ));
        }
        let max_frame_length = options.max_frame_length.unwrap_or(DEFAULT_MAX_FRAME_LENGTH);
        validate_max_frame_length(max_frame_length)?;
        let ClientOptions {
            transport_factory,
            server_id,
            max_frame_length: _,
            on_listener_error,
        } = options;
        let core = Rc::new_cyclic(|client: &std::rc::Weak<ClientCore>| {
            let handshake_client = client.clone();
            let message_client = client.clone();
            let state_client = client.clone();
            let connection_server_id = server_id.clone();
            ClientCore {
                server_id,
                on_listener_error,
                connection: Connection::new(ConnectionOptions {
                    transport_factory,
                    server_id: connection_server_id,
                    max_frame_length,
                    on_handshake: Box::new(move |hello| {
                        if let Some(core) = handshake_client.upgrade() {
                            *core.hello.borrow_mut() = Some(hello.clone());
                        }
                    }),
                    on_message: Box::new(move |payload| {
                        if let Some(core) = message_client.upgrade() {
                            ClientCore::handle_message(&core, payload);
                        }
                    }),
                    on_state_change: Box::new(move |change| {
                        if let Some(core) = state_client.upgrade() {
                            ClientCore::handle_connection_state_change(&core, change);
                        }
                    }),
                }),
                pending: RefCell::new(HashMap::new()),
                state_listeners: RefCell::new(Vec::new()),
                attachment_listeners: RefCell::new(Vec::new()),
                service_listeners: RefCell::new(HashMap::new()),
                request_sequence: Cell::new(0),
                service_sequence: Cell::new(0),
                hello: RefCell::new(None),
                attachment: RefCell::new(None),
                disposed: Cell::new(false),
            }
        });
        Ok(Self { core })
    }

    /// Whether [`dispose`](Self::dispose) has run.
    #[must_use]
    pub fn disposed(&self) -> bool {
        self.core.disposed.get()
    }

    /// The connection's current lifecycle state.
    #[must_use]
    pub fn connection_state(&self) -> ConnectionState {
        self.core.connection.state()
    }

    /// Whether the handshake completed and the transport is live.
    #[must_use]
    pub fn connected(&self) -> bool {
        self.connection_state() == ConnectionState::Connected
    }

    /// The logical server identity every call fences to.
    #[must_use]
    pub fn server_id(&self) -> &str {
        &self.core.server_id
    }

    /// The server hello the current handshake answered with, cleared on
    /// disconnect.
    #[must_use]
    pub fn hello(&self) -> Option<ServerHello> {
        self.core.hello.borrow().clone()
    }

    /// The selected session route, out-of-band updates included.
    #[must_use]
    pub fn attachment(&self) -> Option<SessionTarget> {
        self.core.attachment.borrow().clone()
    }

    /// Builds a client and connects it, disposing on failure, upstream's
    /// static `Client.connect(options)`.
    ///
    /// Upstream spells this `Client.connect(options)`; the instance method
    /// keeps the `connect` name, so the factory takes the `_new` suffix.
    #[must_use]
    pub fn connect_new(options: ClientOptions) -> LocalBoxFuture<Result<Self, ClientError>> {
        boxed(async move {
            let client = Self::new(options)?;
            client.connect().await?;
            Ok(client)
        })
    }

    /// Connects or reconnects, clearing the handshake state first.
    ///
    /// # Errors
    /// Upstream rejects through the handshake promise; the future carries
    /// the same failures.
    #[must_use]
    pub fn connect(&self) -> LocalBoxFuture<Result<ServerHello, ClientError>> {
        if self.core.disposed.get() {
            return boxed(std::future::ready(Err(ClientError::Disposed)));
        }
        *self.core.hello.borrow_mut() = None;
        self.core.connection.connect()
    }

    /// Reconnects, upstream's `reconnect()` alias for [`connect`](Self::connect).
    #[must_use]
    pub fn reconnect(&self) -> LocalBoxFuture<Result<ServerHello, ClientError>> {
        self.connect()
    }

    /// Disconnects with the default reason.
    pub fn disconnect(&self) {
        self.core.connection.disconnect();
    }

    /// Observes connection state changes.
    ///
    /// # Panics
    /// Upstream throws synchronously on a disposed client; the port panics
    /// with the same failure, which fan-outs recover through
    /// [`catch_unwind`](std::panic::catch_unwind).
    pub fn on_connection_state_change(&self, listener: ConnectionStateListener) -> Unsubscribe {
        self.assert_not_disposed();
        register_listener(&self.core, |core| &core.state_listeners, listener)
    }

    /// Observes the selected session route, out-of-band updates included.
    ///
    /// # Panics
    /// Upstream throws synchronously on a disposed client; the port panics
    /// with the same failure.
    pub fn on_attachment_change(&self, listener: AttachmentChangeListener) -> Unsubscribe {
        self.assert_not_disposed();
        register_listener(&self.core, |core| &core.attachment_listeners, listener)
    }

    /// Invokes one low-level protocol call against an explicit routed
    /// target. The frame is sent before the returned future resolves; the
    /// awaiter settles when the correlated response arrives.
    ///
    /// # Errors
    /// Upstream rejects with the server's bounded error on a failed call,
    /// the abort reason on cancellation, and disconnect or disposal
    /// failures otherwise.
    #[must_use]
    pub fn request(
        &self,
        target: &RpcTarget,
        call: &ServiceCall,
        signal: Option<&AbortSignal>,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, ClientError>> {
        request_plain(&self.core, target, call, signal.cloned().as_ref())
    }

    /// Fetches and validates the routed target's service catalogue.
    ///
    /// # Errors
    /// Upstream rejects with the call's failure, or fails the connection
    /// and rejects with a `ProtocolValidationError` when the answer is not
    /// a service catalogue.
    #[must_use]
    pub fn service_catalogue(
        &self,
        target: &RpcTarget,
        signal: Option<&AbortSignal>,
    ) -> LocalBoxFuture<Result<Vec<ServiceCatalogueEntry>, ClientError>> {
        let core = Rc::clone(&self.core);
        let request = request_plain(
            &core,
            target,
            &create_service_catalogue_call(),
            signal.cloned().as_ref(),
        );
        boxed(async move {
            let result = request.await?.unwrap_or(JsonValue::Null);
            match parse_service_catalogue(&result) {
                Ok(entries) => Ok(entries),
                Err(error) => {
                    let validation = validation_error(&error);
                    core.connection.fail(&validation);
                    Err(validation)
                }
            }
        })
    }

    /// Opens one service subscription, buffering updates until the
    /// snapshot arrives and the caller begins delivery.
    ///
    /// # Errors
    /// Upstream rejects with the subscribe call's failure, the abort
    /// reason, and disconnect or disposal failures; a subscription removed
    /// while its request was in flight rejects as disconnected.
    #[must_use]
    pub fn subscribe_service(
        &self,
        target: &RpcTarget,
        service_id: &str,
        mode: ServiceMode,
        listener: Rc<dyn Fn(&ServiceProviderUpdate)>,
        signal: Option<&AbortSignal>,
    ) -> LocalBoxFuture<Result<ServiceSubscription, ClientError>> {
        let core = &self.core;
        let subscription_id = format!("service-{}", core.service_sequence.get() + 1);
        core.service_sequence.set(core.service_sequence.get() + 1);
        let active = Rc::new(RefCell::new(ActiveServiceListener {
            listener,
            decoder: create_service_state_decoder(),
            queued_wire_updates: Vec::new(),
            queued: Vec::new(),
            hydrated: false,
            ready: false,
        }));
        core.service_listeners
            .borrow_mut()
            .insert(subscription_id.clone(), active.clone());
        let request = request_subscription(
            core,
            target,
            &create_service_subscribe_call(&subscription_id, service_id, mode),
            signal.cloned().as_ref(),
            Rc::clone(&active),
        );
        let core = Rc::clone(core);
        let target = target.clone();
        boxed(async move {
            match request.await {
                Err(error) => {
                    if subscription_registered(&core, &subscription_id, &active) {
                        core.service_listeners.borrow_mut().remove(&subscription_id);
                    }
                    Err(error)
                }
                Ok(snapshot) => {
                    if !subscription_registered(&core, &subscription_id, &active) {
                        return Err(ClientError::disconnected());
                    }
                    Ok(ServiceSubscription::new(Rc::new(SubscriptionShared {
                        id: subscription_id,
                        target,
                        snapshot,
                        core,
                        active,
                        disposed: Cell::new(false),
                    })))
                }
            }
        })
    }

    /// Disposes the client: rejects pending requests, disconnects, and
    /// clears every listener table. Repeat calls are no-ops.
    pub fn dispose(&self) {
        let core = &self.core;
        if core.disposed.get() {
            return;
        }
        core.disposed.set(true);
        ClientCore::reject_pending(core, &ClientError::Disposed);
        core.connection.fail(&ClientError::Disposed);
        *core.hello.borrow_mut() = None;
        ClientCore::set_attachment(core, None);
        core.state_listeners.borrow_mut().clear();
        core.attachment_listeners.borrow_mut().clear();
        core.service_listeners.borrow_mut().clear();
    }

    fn assert_not_disposed(&self) {
        assert!(!self.core.disposed.get(), "{}", ClientError::Disposed);
    }
}

/// The correlation id one response answers, upstream's `message.id`.
fn response_id(response: &ResponseEnvelope) -> &str {
    match response {
        ResponseEnvelope::Success(success) => &success.id,
        ResponseEnvelope::Failure(failure) => &failure.id,
    }
}

/// Whether the listener table still holds `active` under `id`.
fn subscription_registered(
    core: &Rc<ClientCore>,
    subscription_id: &str,
    active: &Rc<RefCell<ActiveServiceListener>>,
) -> bool {
    core.service_listeners
        .borrow()
        .get(subscription_id)
        .is_some_and(|current| Rc::ptr_eq(current, active))
}

/// Registers one listener and returns its unsubscribe closure, the
/// observer plumbing both listener tables share: the closure holds the
/// core weakly so a live subscription cannot keep the client alive, and
/// removes exactly the listener it was created for.
fn register_listener<T: ?Sized + 'static>(
    core: &Rc<ClientCore>,
    listeners: fn(&ClientCore) -> &RefCell<Vec<Rc<T>>>,
    listener: Rc<T>,
) -> Unsubscribe {
    listeners(core).borrow_mut().push(listener.clone());
    let core = Rc::downgrade(core);
    Box::new(move || {
        if let Some(core) = core.upgrade() {
            listeners(&core)
                .borrow_mut()
                .retain(|registered| !Rc::ptr_eq(registered, &listener));
        }
    })
}

/// Registers the request, encodes the frame, and sends it, upstream's
/// `#request` setup through the send.
///
/// Upstream rejects through the request's promise when the encode fails,
/// and throws synchronously when the send does; the port returns the send
/// failure directly and lets the encode rejection settle the caller's
/// future through its oneshot.
fn begin_request(
    core: &Rc<ClientCore>,
    target: &RpcTarget,
    call: &ServiceCall,
    signal: Option<&AbortSignal>,
    pending: PendingRequest,
) -> Result<String, ClientError> {
    if core.disposed.get() {
        return Err(ClientError::Disposed);
    }
    if core.connection.state() != ConnectionState::Connected {
        return Err(ClientError::disconnected());
    }
    if let Some(signal) = signal
        && signal.aborted()
    {
        return Err(abort_signal_error(signal));
    }
    let id = format!("request-{}", core.request_sequence.get() + 1);
    core.request_sequence.set(core.request_sequence.get() + 1);
    core.pending.borrow_mut().insert(id.clone(), pending);
    let frame = match encode_message(
        &ClientMessage::Request(RequestEnvelope {
            id: id.clone(),
            target: target.clone(),
            call: service_call_to_json(call),
        }),
        core.connection.max_frame_length(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            let error = ClientError::Protocol(error);
            if let Some(pending) = core.pending.borrow_mut().remove(&id) {
                (pending.reject)(&error);
            }
            return Ok(id);
        }
    };
    core.connection.send(frame)?;
    Ok(id)
}

/// Registers the pending request, sends its frame, and builds the caller's
/// await, the tail both request flows share: without a signal the future
/// awaits the correlated response, with one it races the response against
/// the abort and cancels the request.
///
/// The frame is sent before the returned future is even polled, upstream's
/// send-then-promise split.
fn request_start<V: 'static>(
    core: &Rc<ClientCore>,
    target: &RpcTarget,
    call: &ServiceCall,
    signal: Option<&AbortSignal>,
    settle: PendingSettle,
    sender_cell: &PendingSender<V>,
    receiver: oneshot::Receiver<Result<V, ClientError>>,
) -> LocalBoxFuture<Result<V, ClientError>> {
    let id = match begin_request(
        core,
        target,
        call,
        signal,
        PendingRequest::new(settle, sender_cell),
    ) {
        Ok(id) => id,
        Err(error) => return boxed(std::future::ready(Err(error))),
    };
    let core = Rc::clone(core);
    match signal.cloned() {
        None => boxed(async move {
            receiver
                .await
                .unwrap_or_else(|_| Err(ClientError::disconnected()))
        }),
        Some(signal) => {
            let target = target.clone();
            boxed(async move {
                tokio::select! {
                    settled = receiver => settled.unwrap_or_else(|_| Err(ClientError::disconnected())),
                    reason = signal.wait() => {
                        send_cancel(&core, &id, &target);
                        Err(abort_error(&reason))
                    }
                }
            })
        }
    }
}

/// Invokes one plain call and awaits its correlated response, upstream's
/// `#request` without a transform.
fn request_plain(
    core: &Rc<ClientCore>,
    target: &RpcTarget,
    call: &ServiceCall,
    signal: Option<&AbortSignal>,
) -> LocalBoxFuture<Result<Option<JsonValue>, ClientError>> {
    let (sender, receiver) = oneshot::channel::<Result<Option<JsonValue>, ClientError>>();
    let sender_cell: PendingSender<Option<JsonValue>> = Rc::new(RefCell::new(Some(sender)));
    let settle = Box::new({
        let sender = sender_cell.clone();
        move |_core: &Rc<ClientCore>, response: &ResponseEnvelope| {
            let settled = match response {
                ResponseEnvelope::Success(success) => Ok(success.result.clone()),
                ResponseEnvelope::Failure(failure) => {
                    Err(ClientError::Server(failure.error.clone()))
                }
            };
            if let Some(sender) = sender.borrow_mut().take() {
                let _ = sender.send(settled);
            }
        }
    });
    request_start(core, target, call, signal, settle, &sender_cell, receiver)
}

/// Invokes the subscribe control call; the response's snapshot decodes at
/// settlement and hydrates the buffered wire updates, upstream's transform
/// parameter.
fn request_subscription(
    core: &Rc<ClientCore>,
    target: &RpcTarget,
    call: &ServiceCall,
    signal: Option<&AbortSignal>,
    active: Rc<RefCell<ActiveServiceListener>>,
) -> LocalBoxFuture<Result<ServiceSubscriptionSnapshot, ClientError>> {
    let (sender, receiver) = oneshot::channel::<Result<ServiceSubscriptionSnapshot, ClientError>>();
    let sender_cell: PendingSender<ServiceSubscriptionSnapshot> =
        Rc::new(RefCell::new(Some(sender)));
    let settle = Box::new({
        let sender = sender_cell.clone();
        move |core: &Rc<ClientCore>, response: &ResponseEnvelope| {
            let Some(sender) = sender.borrow_mut().take() else {
                return;
            };
            match response {
                ResponseEnvelope::Success(success) => {
                    match hydrate_subscription(&active, success.result.as_ref()) {
                        Ok(snapshot) => {
                            let _ = sender.send(Ok(snapshot));
                        }
                        Err(error) => {
                            let _ = sender.send(Err(error.clone()));
                            core.connection.fail(&error);
                        }
                    }
                }
                ResponseEnvelope::Failure(failure) => {
                    let _ = sender.send(Err(ClientError::Server(failure.error.clone())));
                }
            }
        }
    });
    request_start(core, target, call, signal, settle, &sender_cell, receiver)
}

/// Sends the cancel envelope for one aborted request, upstream's
/// `sendCancel`: only once the request frame was sent and the connection
/// still stands, and a failure fails the connection.
fn send_cancel(core: &Rc<ClientCore>, id: &str, target: &RpcTarget) {
    if core.connection.state() != ConnectionState::Connected {
        return;
    }
    match encode_message(
        &ClientMessage::Cancel(CancelEnvelope {
            id: id.to_string(),
            target: target.clone(),
        }),
        core.connection.max_frame_length(),
    ) {
        Ok(frame) => {
            if let Err(error) = core.connection.send(frame) {
                core.connection.fail(&error);
            }
        }
        Err(error) => core.connection.fail(&validation_error(&error)),
    }
}

/// Encodes one client message against the shared frame bound.
fn encode_message(
    message: &ClientMessage,
    max_frame_length: usize,
) -> Result<Vec<u8>, ProtocolValidationError> {
    pi_protocol::encode_client_message(message, FrameDecoderOptions { max_frame_length })
}

/// The abort rejection a cancelled request settles with, upstream's
/// `abortError(signal)`: the caller's reason, or the standard abort error.
fn abort_error(reason: &AbortReason) -> ClientError {
    ClientError::other(reason.to_string())
}

/// The rejection a pre-aborted request settles with.
fn abort_signal_error(signal: &AbortSignal) -> ClientError {
    signal.reason().map_or_else(
        || ClientError::other("The operation was aborted"),
        |reason| abort_error(&reason),
    )
}

/// The typed failure a validation surface raises, upstream's re-raised
/// `ProtocolValidationError`.
fn validation_error(error: impl std::fmt::Display) -> ClientError {
    ClientError::Protocol(ProtocolValidationError::new(error.to_string()))
}

/// Restates one typed service call as the strict-JSON wire shape, the
/// `parseServiceCall(call)` conversion upstream embeds in the request
/// envelope.
fn service_call_to_json(call: &ServiceCall) -> JsonValue {
    let mut entries = vec![("serviceId", JsonValue::string(call.service_id.clone()))];
    if let Some(instance) = &call.instance {
        entries.push((
            "instance",
            json_object(vec![
                ("key", JsonValue::string(instance.key.clone())),
                ("generation", JsonValue::Number(instance.generation.into())),
            ]),
        ));
    }
    entries.push(("member", JsonValue::string(call.member.clone())));
    entries.push(("args", JsonValue::Array(call.args.clone())));
    json_object(entries)
}

/// Decodes the snapshot response and drains the buffered wire updates,
/// upstream's `subscribeService` transform.
fn hydrate_subscription(
    active: &Rc<RefCell<ActiveServiceListener>>,
    result: Option<&JsonValue>,
) -> Result<ServiceSubscriptionSnapshot, ClientError> {
    let mut listener = active.borrow_mut();
    let wire = parse_wire_service_subscription_snapshot(result.unwrap_or(&JsonValue::Null))
        .map_err(|error| validation_error(&error))?;
    let snapshot = listener
        .decoder
        .decode_snapshot(&wire)
        .map_err(|error| validation_error(&error))?;
    listener.hydrated = true;
    let queued_wire = std::mem::take(&mut listener.queued_wire_updates);
    for update in queued_wire {
        let parsed = parse_wire_service_provider_update(&update)
            .map_err(|error| validation_error(&error))?;
        let decoded = listener
            .decoder
            .decode_update(&parsed)
            .map_err(|error| validation_error(&error))?;
        listener.queued.push(decoded);
    }
    Ok(snapshot)
}

/// Decodes one post-hydration wire update, upstream's `service_update`
/// decode whose failure fails the connection.
fn decode_service_update(
    active: &Rc<RefCell<ActiveServiceListener>>,
    update: &JsonValue,
) -> Result<ServiceProviderUpdate, ClientError> {
    let mut listener = active.borrow_mut();
    let parsed =
        parse_wire_service_provider_update(update).map_err(|error| validation_error(&error))?;
    listener
        .decoder
        .decode_update(&parsed)
        .map_err(|error| validation_error(&error))
}

/// Adapts a lazily resolved routed client target to a chord service
/// transport, upstream's `createClientServiceTransport`.
///
/// Upstream spells the transport's `subscribe`/`invoke` rejections as-is;
/// the chord port's transport contract carries [`ChordError`], so the
/// adapter maps: a bounded server error whose code names a chord
/// remote-service failure maps onto it, everything else flattens to its
/// message.
#[derive(Clone)]
pub struct ClientServiceTransport {
    client: Client,
    get_target: Rc<dyn Fn() -> Option<RpcTarget>>,
}

impl std::fmt::Debug for ClientServiceTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClientServiceTransport")
            .field("client", &self.client)
            .finish_non_exhaustive()
    }
}

impl RemoteServiceTransport for ClientServiceTransport {
    fn invoke(
        &self,
        call: ServiceCall,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, ChordError>> {
        let Some(target) = (self.get_target)() else {
            return boxed(std::future::ready(Err(ChordError::message(
                "Remote service target is unavailable",
            ))));
        };
        let client = self.client.clone();
        boxed(async move {
            client
                .request(&target, &call, context.abort_signal().as_ref())
                .await
                .map_err(to_chord_error)
        })
    }

    fn subscribe(
        &self,
        service_id: String,
        mode: ServiceMode,
        listener: ServiceProviderListener,
        context: Context,
    ) -> LocalBoxFuture<Result<ChordSubscription, ChordError>> {
        let client = self.client.clone();
        let get_target = self.get_target.clone();
        boxed(async move {
            let Some(target) = get_target() else {
                return Err(ChordError::message("Remote service target is unavailable"));
            };
            let subscription = client
                .subscribe_service(
                    &target,
                    &service_id,
                    mode,
                    Rc::new(move |update| listener(update, &background_context())),
                    context.abort_signal().as_ref(),
                )
                .await
                .map_err(to_chord_error)?;
            Ok(ChordSubscription {
                snapshot: subscription.snapshot().clone(),
                activate: Box::new({
                    let subscription = subscription.clone();
                    move || {
                        subscription.start();
                        Ok(())
                    }
                }),
                close: Box::new({
                    let subscription = subscription.clone();
                    move |_context: Option<Context>| {
                        let dispose = subscription.dispose();
                        boxed(async move { dispose.await.map_err(to_chord_error) })
                    }
                }),
            })
        })
    }
}

/// Builds the transport adapter, upstream's `createClientServiceTransport`.
#[must_use]
pub fn create_client_service_transport(
    client: &Client,
    get_target: impl Fn() -> Option<RpcTarget> + 'static,
) -> ClientServiceTransport {
    ClientServiceTransport {
        client: client.clone(),
        get_target: Rc::new(get_target),
    }
}

/// Maps a client failure onto the chord error model, the rejection
/// conversion the chord port's transport contract requires.
fn to_chord_error(error: ClientError) -> ChordError {
    match error {
        ClientError::Server(failure) => match RemoteServiceErrorCode::parse(&failure.code) {
            Some(code) => ChordError::Remote(RemoteServiceError::new(code, failure.message)),
            None => ChordError::message(failure.message),
        },
        other => ChordError::message(other.to_string()),
    }
}

/// Validates one frame length, upstream's connection-constructor range
/// check.
///
/// # Errors
/// Upstream throws a `TypeError` outside `1..=u32::MAX`; the port returns
/// the same failure.
pub(crate) fn validate_max_frame_length(max_frame_length: usize) -> Result<(), ClientError> {
    if max_frame_length == 0 || max_frame_length > crate::connection::MAX_UINT32 {
        return Err(ClientError::other(format!(
            "Client maxFrameLength must be an integer between 1 and {}",
            crate::connection::MAX_UINT32
        )));
    }
    Ok(())
}
