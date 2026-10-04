//! The server endpoint, ported from upstream `src/server.ts`.
//!
//! Upstream mutates each connection's state object across awaits and queues
//! messages behind the in-flight handshake promise; the port restates the
//! queueing behind a [`Latch`] the handshake settles and keeps every borrow
//! short so the async paths cannot collide. The handler closures capture the
//! server core weakly — upstream's closures reference the server through the
//! event loop's reachability, which an `Rc` cycle would pin forever.
//!
//! `ServerCore` is the crate-wide core; the public surface is the
//! [`Server`] handle.
#![allow(
    clippy::redundant_pub_crate,
    reason = "ServerCore is the crate-wide core; the public surface is the Server handle"
)]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::time::Duration;

use pi_agent_core::harness::context::{
    Context, background_context, placeholder_context, with_cancel,
};
use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::services::state_codec::create_service_state_encoder;
use pi_chord::services::wire::{
    ServiceControlCall, decode_service_control_call, wire_snapshot_to_json, wire_update_to_json,
};
use pi_chord::services::{parse_service_call, parse_service_subscription_snapshot};
use pi_chord::types::{JsonValue, ServiceProviderUpdate};
use pi_protocol::{
    AttachmentEnvelope, CancelEnvelope, ClientHello, ClientMessage, ClientMessageDecoder,
    FrameDecoderOptions, ProtocolError, ProtocolValidationError, RequestEnvelope, ResponseEnvelope,
    ResponseFailure, RpcTarget, ServerHello, ServerHelloError, ServerId, ServerMessage,
    encode_server_message, is_supported_protocol_version,
};

use crate::connection::{
    ActiveRequest, ByteConnection, ByteConnectionAcceptor, ByteConnectionHandler, ClientToken,
    ConnectionCore, ConnectionStage, is_terminal_connection,
};
use crate::errors::{Failure, ServerError};
use crate::latch::CloseLatch as ServerCloseLatch;
use crate::latch::Latch;
use crate::listener::ServerListener;
use crate::session_router::SessionRouter;
use crate::types::{RoutedServerPresentation, ServerHost, ServerOptions, ServicePublisher, ready};

/// The handshake budget, upstream's `DEFAULT_HANDSHAKE_TIMEOUT_MS`.
const DEFAULT_HANDSHAKE_TIMEOUT_MS: u64 = 5_000;
/// The framed-byte ceiling, upstream's `MAX_UINT32` option guard.
const MAX_UINT32: usize = u32::MAX as usize;
/// The upper bound a timeout may take, upstream's `MAX_TIMER_DELAY_MS`
/// Node-timer bound restated for the tokio clock.
const MAX_TIMER_DELAY_MS: u64 = 2_147_483_647;

/// The transport-neutral server endpoint, upstream's `Server<TMetadata>`.
///
/// The generic parameter rides the host, upstream's `TMetadata extends
/// SessionMetadata`. The handle is cheaply cloneable; every surface expects
/// the caller's current-thread tokio runtime behind a `LocalSet`.
pub struct Server<H: ServerHost> {
    core: Rc<ServerCore<H>>,
}

impl<H: ServerHost> Clone for Server<H> {
    fn clone(&self) -> Self {
        Self {
            core: Rc::clone(&self.core),
        }
    }
}

impl<H: ServerHost> std::fmt::Debug for Server<H> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Server")
            .field("server_id", &self.core.server_id.as_str())
            .finish_non_exhaustive()
    }
}

impl<H: ServerHost + 'static> Server<H> {
    /// Builds a server over `host` and `options`, upstream's `new
    /// Server(host, options)`; the constructor throws for invalid options.
    ///
    /// # Errors
    /// A non-canonical `server_id`, or a `max_frame_length` /
    /// `handshake_timeout_ms` outside its range, upstream's `TypeError`s.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the host is the server's owned dependency, upstream's `new Server(host, options)"
    )]
    pub fn new(host: Rc<H>, options: ServerOptions) -> Result<Self, Failure> {
        let server_id = ServerId::new(&options.server_id)
            .ok_or_else(|| Failure::message("serverId must be a canonical lowercase UUIDv4"))?;
        let max_frame_length = options
            .max_frame_length
            .unwrap_or(pi_protocol::DEFAULT_MAX_FRAME_LENGTH);
        if max_frame_length == 0 || max_frame_length > MAX_UINT32 {
            return Err(Failure::message(format!(
                "Server maxFrameLength must be an integer between 1 and {MAX_UINT32}"
            )));
        }
        let handshake_timeout_ms = options
            .handshake_timeout_ms
            .unwrap_or(DEFAULT_HANDSHAKE_TIMEOUT_MS);
        if handshake_timeout_ms == 0 || handshake_timeout_ms > MAX_TIMER_DELAY_MS {
            return Err(Failure::message(format!(
                "Server handshakeTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}"
            )));
        }
        let core = Rc::new_cyclic(|core| ServerCore {
            sessions: SessionRouter::new(Rc::clone(&host), server_id.clone(), core.clone()),
            host: Rc::clone(&host),
            listeners: options.listeners,
            server_id: server_id.clone(),
            max_frame_length,
            handshake_timeout_ms,
            on_connection_count_changed: options.on_connection_count_changed,
            on_error: options.on_error,
            connections: RefCell::new(HashMap::new()),
            closing: Cell::new(false),
            started: Cell::new(false),
            start_done: RefCell::new(None),
            close_latch: RefCell::new(None),
            closed_latch: Rc::new(Latch::new()),
        });
        Ok(Self { core })
    }

    /// The logical server identity, upstream's `serverId`.
    #[must_use]
    pub fn server_id(&self) -> &str {
        self.core.server_id.as_str()
    }

    /// The registry identity the teardowns track, upstream's object
    /// identity.
    #[must_use]
    pub fn identity_token(&self) -> usize {
        Rc::as_ptr(&self.core) as usize
    }

    /// Starts every listener, upstream's `start`; the returned future
    /// resolves with the server handle.
    ///
    /// A second call before the first settles rejects with `already
    /// starting`; after a settle it rejects with `already started` or
    /// `closing or closed`.
    #[must_use]
    pub fn start(&self) -> LocalBoxFuture<Result<Self, Failure>>
    where
        Self: Sized,
    {
        if self.core.started.get() {
            return ready(Err(Failure::message("Server is already started")));
        }
        if self.core.start_done.borrow().is_some() {
            return ready(Err(Failure::message("Server is already starting")));
        }
        if self.core.closing.get() {
            return ready(Err(Failure::message("Server is closing or closed")));
        }
        let done = Rc::new(Latch::new());
        *self.core.start_done.borrow_mut() = Some(Rc::clone(&done));
        let core = Rc::clone(&self.core);
        let spawn_done = Rc::clone(&done);
        tokio::task::spawn_local(async move {
            let result = core.start_internal().await;
            spawn_done.settle(result);
        });
        let core = Rc::downgrade(&self.core);
        boxed(async move {
            match done.wait().await {
                Ok(()) => core
                    .upgrade()
                    .map(|core| Self { core })
                    .ok_or_else(|| Failure::message("server is gone")),
                Err(failure) => Err(failure),
            }
        })
    }

    /// Closes every listener and connection and drains routed Sessions,
    /// upstream's `close`; idempotent through the shared settle.
    #[must_use]
    pub fn close(&self) -> LocalBoxFuture<Result<(), Failure>> {
        if let Some(existing) = self.core.close_latch.borrow().clone() {
            return boxed(async move { existing.wait().await });
        }
        self.core.closing.set(true);
        let latch = Rc::new(Latch::new());
        *self.core.close_latch.borrow_mut() = Some(Rc::clone(&latch));
        let core = Rc::clone(&self.core);
        let spawn_latch = Rc::clone(&latch);
        tokio::task::spawn_local(async move {
            let result = core.close_internal().await;
            spawn_latch.settle(result);
        });
        boxed(async move { latch.wait().await })
    }

    /// The shutdown outcome, upstream's `closed`: resolves after shutdown,
    /// rejects when listener or routed-Session cleanup fails.
    #[must_use]
    pub fn closed(&self) -> LocalBoxFuture<Result<(), Failure>> {
        let core = Rc::clone(&self.core);
        boxed(async move { core.closed_latch.wait().await })
    }

    /// Adopts one established connection, upstream's `accept`; the returned
    /// handler is what the listener drives.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the connection joins the server's registry, upstream's by-value accept"
    )]
    pub fn accept(&self, connection: Rc<dyn ByteConnection>) -> ByteConnectionHandler {
        self.core.accept(connection)
    }
}

/// The shared server state, upstream's `Server` private fields.
pub(crate) struct ServerCore<H: ServerHost> {
    host: Rc<H>,
    listeners: Vec<Rc<dyn ServerListener>>,
    server_id: ServerId,
    max_frame_length: usize,
    handshake_timeout_ms: u64,
    on_connection_count_changed: Option<Rc<dyn Fn(usize)>>,
    on_error: Option<crate::types::ErrorObserver>,
    connections: RefCell<HashMap<usize, Rc<ConnectionCore>>>,
    pub(crate) sessions: SessionRouter<H>,
    pub(crate) closing: Cell<bool>,
    started: Cell<bool>,
    start_done: RefCell<Option<ServerCloseLatch>>,
    close_latch: RefCell<Option<ServerCloseLatch>>,
    closed_latch: ServerCloseLatch,
}

impl<H: ServerHost + 'static> ServerCore<H> {
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the connection joins the connection registry, upstream's by-value accept"
    )]
    fn accept(self: &Rc<Self>, connection: Rc<dyn ByteConnection>) -> ByteConnectionHandler {
        if self.closing.get() {
            let weak = Rc::downgrade(self);
            let closing_connection = Rc::clone(&connection);
            tokio::task::spawn_local(async move {
                if let Some(core) = weak.upgrade() {
                    core.close_connection(&closing_connection, None).await;
                }
            });
            let weak = Rc::downgrade(self);
            return ByteConnectionHandler {
                on_data: Rc::new(|_| {}),
                on_close: Rc::new(|| {}),
                on_error: Rc::new(move |error| {
                    if let Some(core) = weak.upgrade() {
                        core.report_error(error);
                    }
                }),
            };
        }
        let decoder = match ClientMessageDecoder::new(FrameDecoderOptions {
            max_frame_length: self.max_frame_length,
        }) {
            Ok(decoder) => decoder,
            // Construction validated the bound the decoder accepts, so this
            // arm is unreachable; an inert handler keeps the transport
            // honest without a panic.
            Err(error) => {
                self.report_error(&Failure::message(error.message()));
                return ByteConnectionHandler {
                    on_data: Rc::new(|_| {}),
                    on_close: Rc::new(|| {}),
                    on_error: Rc::new(|_| {}),
                };
            }
        };
        let state = Rc::new(ConnectionCore {
            token: ClientToken::new(),
            connection: Rc::clone(&connection),
            decoder: RefCell::new(decoder),
            service_state_encoders: RefCell::new(HashMap::new()),
            stage: Cell::new(ConnectionStage::AwaitingHello),
            disconnected: Cell::new(false),
            handshake: RefCell::new(None),
            handshake_timeout: RefCell::new(None),
            server_services: RefCell::new(None),
            active_requests: RefCell::new(HashMap::new()),
        });
        let timeout_task = {
            let core = Rc::downgrade(self);
            let state = Rc::clone(&state);
            let handshake_timeout_ms = self.handshake_timeout_ms;
            tokio::task::spawn_local(async move {
                tokio::time::sleep(Duration::from_millis(handshake_timeout_ms)).await;
                if let Some(core) = core.upgrade() {
                    core.fail_protocol(
                        &state,
                        ProtocolError {
                            code: "invalid_request".to_string(),
                            message: "Handshake timeout".to_string(),
                        },
                    )
                    .await;
                }
            })
        };
        *state.handshake_timeout.borrow_mut() = Some(timeout_task);
        self.connections
            .borrow_mut()
            .insert(state.token.key(), Rc::clone(&state));
        self.notify_connection_count_changed();
        let weak_for_data = Rc::downgrade(self);
        let state_for_data = Rc::clone(&state);
        let weak_for_close = Rc::downgrade(self);
        let state_for_close = Rc::clone(&state);
        let weak_for_error = Rc::downgrade(self);
        let state_for_error = Rc::clone(&state);
        let error_connection = Rc::clone(&connection);
        ByteConnectionHandler {
            on_data: Rc::new(move |chunk| {
                if let Some(core) = weak_for_data.upgrade() {
                    core.receive(&state_for_data, chunk);
                }
            }),
            on_close: Rc::new(move || {
                if let Some(core) = weak_for_close.upgrade() {
                    core.transport_closed(&state_for_close);
                }
            }),
            on_error: Rc::new(move |error| {
                if let Some(core) = weak_for_error.upgrade() {
                    core.report_error(error);
                    let connection = Rc::clone(&error_connection);
                    let state = Rc::clone(&state_for_error);
                    tokio::task::spawn_local(async move {
                        core.close_connection(&connection, None).await;
                        core.disconnect(&state);
                    });
                }
            }),
        }
    }

    async fn start_internal(self: &Rc<Self>) -> Result<(), Failure> {
        let mut started: Vec<Rc<dyn ServerListener>> = Vec::new();
        let outcome: Result<(), Failure> = async {
            for listener in &self.listeners {
                listener.start(self.acceptor()).await?;
                started.push(Rc::clone(listener));
            }
            self.started.set(true);
            Ok(())
        }
        .await;
        let result = match outcome {
            Ok(()) => Ok(()),
            Err(error) => {
                self.closing.set(true);
                let mut cleanup_errors: Vec<Failure> = Vec::new();
                let mut handles = Vec::with_capacity(started.len());
                for listener in &started {
                    let listener = Rc::clone(listener);
                    handles.push(tokio::task::spawn_local(
                        async move { listener.close().await },
                    ));
                }
                for handle in handles {
                    match handle.await {
                        Ok(Ok(())) => {}
                        Ok(Err(cleanup)) => cleanup_errors.push(cleanup),
                        Err(join) => cleanup_errors.push(Failure::message(join.to_string())),
                    }
                }
                if let Err(cleanup) = self.close_server_state().await {
                    cleanup_errors.push(cleanup);
                }
                if cleanup_errors.is_empty() {
                    self.settle_closed(Ok(()));
                    Err(error)
                } else {
                    let mut errors = vec![error];
                    errors.extend(cleanup_errors);
                    let failure = Failure::Aggregate {
                        message: "Server startup and cleanup failed".to_string(),
                        errors,
                    };
                    self.settle_closed(Err(failure.clone()));
                    Err(failure)
                }
            }
        };
        *self.start_done.borrow_mut() = None;
        result
    }

    fn acceptor(self: &Rc<Self>) -> ByteConnectionAcceptor {
        let weak = Rc::downgrade(self);
        Rc::new(move |connection| {
            weak.upgrade().map_or_else(
                || ByteConnectionHandler {
                    on_data: Rc::new(|_| {}),
                    on_close: Rc::new(|| {}),
                    on_error: Rc::new(|_| {}),
                },
                |core| core.accept(connection),
            )
        })
    }

    async fn close_internal(self: &Rc<Self>) -> Result<(), Failure> {
        let starting = self.start_done.borrow().clone();
        if let Some(start) = starting {
            let _ = start.wait().await;
        }
        let mut errors: Vec<Failure> = Vec::new();
        let mut handles = Vec::with_capacity(self.listeners.len());
        for listener in &self.listeners {
            let listener = Rc::clone(listener);
            handles.push(tokio::task::spawn_local(
                async move { listener.close().await },
            ));
        }
        for handle in handles {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => errors.push(error),
                Err(join) => errors.push(Failure::message(join.to_string())),
            }
        }
        if let Err(error) = self.close_server_state().await {
            errors.push(error);
        }
        self.started.set(false);
        if errors.is_empty() {
            self.settle_closed(Ok(()));
            return Ok(());
        }
        let failure = if errors.len() == 1 {
            errors.swap_remove(0)
        } else {
            Failure::Aggregate {
                message: "Server shutdown failed".to_string(),
                errors,
            }
        };
        self.settle_closed(Err(failure.clone()));
        Err(failure)
    }

    fn receive(self: &Rc<Self>, state: &Rc<ConnectionCore>, chunk: &[u8]) {
        if is_terminal_connection(state) {
            return;
        }
        let messages = match state.decoder.borrow_mut().push(chunk) {
            Ok(messages) => messages,
            Err(error) => {
                let core = Rc::clone(self);
                let state = Rc::clone(state);
                let failure = Failure::Validation(error);
                tokio::task::spawn_local(async move {
                    core.fail_protocol(&state, failure.wire_error()).await;
                });
                return;
            }
        };
        for message in messages {
            if is_terminal_connection(state) {
                return;
            }
            self.dispatch_message(state, message);
        }
    }

    fn dispatch_message(self: &Rc<Self>, state: &Rc<ConnectionCore>, message: ClientMessage) {
        if state.stage.get() == ConnectionStage::AwaitingHello {
            if !matches!(message, ClientMessage::Hello(_)) {
                let core = Rc::clone(self);
                let state = Rc::clone(state);
                tokio::task::spawn_local(async move {
                    core.fail_protocol(
                        &state,
                        ProtocolError {
                            code: "invalid_request".to_string(),
                            message: "The first client message must be hello".to_string(),
                        },
                    )
                    .await;
                });
                return;
            }
            let ClientMessage::Hello(hello) = message else {
                return;
            };
            state.stage.set(ConnectionStage::Handshaking);
            let done = Rc::new(Latch::new());
            *state.handshake.borrow_mut() = Some(Rc::clone(&done));
            let core = Rc::clone(self);
            let state = Rc::clone(state);
            tokio::task::spawn_local(async move {
                core.finish_handshake(&state, hello, &done).await;
            });
            return;
        }
        if matches!(message, ClientMessage::Hello(_)) {
            let core = Rc::clone(self);
            let state = Rc::clone(state);
            tokio::task::spawn_local(async move {
                core.fail_protocol(
                    &state,
                    ProtocolError {
                        code: "invalid_request".to_string(),
                        message: "hello may only be sent as the first message".to_string(),
                    },
                )
                .await;
            });
            return;
        }
        if state.stage.get() == ConnectionStage::Ready {
            match message {
                ClientMessage::Cancel(envelope) => self.handle_cancel(state, &envelope),
                ClientMessage::Request(envelope) => {
                    let core = Rc::clone(self);
                    let state = Rc::clone(state);
                    tokio::task::spawn_local(async move {
                        core.handle_request(&state, envelope).await;
                    });
                }
                ClientMessage::Hello(_) => {}
            }
            return;
        }
        if state.stage.get() != ConnectionStage::Handshaking {
            return;
        }
        let Some(handshake) = state.handshake.borrow().clone() else {
            return;
        };
        let core = Rc::clone(self);
        let state = Rc::clone(state);
        tokio::task::spawn_local(async move {
            handshake.wait().await;
            if state.stage.get() == ConnectionStage::Ready && !state.disconnected.get() {
                match message {
                    ClientMessage::Cancel(envelope) => core.handle_cancel(&state, &envelope),
                    ClientMessage::Request(envelope) => {
                        core.handle_request(&state, envelope).await;
                    }
                    ClientMessage::Hello(_) => {}
                }
            }
        });
    }

    async fn finish_handshake(
        self: &Rc<Self>,
        state: &Rc<ConnectionCore>,
        hello: ClientHello,
        done: &Latch<()>,
    ) {
        let outcome: Result<(), Failure> = async {
            if !is_supported_protocol_version(hello.version) {
                self.fail_protocol(
                    state,
                    ProtocolError {
                        code: "version".to_string(),
                        message: format!(
                            "Unsupported protocol version {}; expected {}",
                            hello.version.get(),
                            pi_protocol::PROTOCOL_VERSION
                        ),
                    },
                )
                .await;
                return Ok(());
            }
            if self.closing.get()
                || state.disconnected.get()
                || state.stage.get() != ConnectionStage::Handshaking
                || state.connection.closed()
            {
                return Ok(());
            }
            let presentation: Rc<dyn RoutedServerPresentation> = Rc::new(CorePresentation {
                core: Rc::downgrade(self),
                state: Rc::clone(state),
            });
            let services = match self
                .host
                .server_services()
                .attach_client(presentation, placeholder_context())
                .await
            {
                Ok(services) => services,
                Err(error) => return Err(error),
            };
            if self.closing.get()
                || state.disconnected.get()
                || state.stage.get() != ConnectionStage::Handshaking
                || state.connection.closed()
            {
                services.release(placeholder_context()).await?;
                return Ok(());
            }
            *state.server_services.borrow_mut() = Some(Rc::clone(&services));
            let sent = self
                .send_message(
                    state,
                    ServerMessage::Hello(ServerHello {
                        server_id: self.server_id.clone(),
                    }),
                )
                .await;
            if sent
                && !state.disconnected.get()
                && state.stage.get() == ConnectionStage::Handshaking
            {
                state.stage.set(ConnectionStage::Ready);
                state.clear_handshake_timeout();
            }
            Ok(())
        }
        .await;
        if let Err(error) = &outcome {
            let wire = self.to_protocol_error(error);
            self.fail_protocol(state, wire).await;
        }
        done.settle(());
    }

    fn handle_cancel(self: &Rc<Self>, state: &Rc<ConnectionCore>, envelope: &CancelEnvelope) {
        if target_server_id(&envelope.target).as_str() != self.server_id.as_str() {
            return;
        }
        let active = state.active_requests.borrow().get(&envelope.id).cloned();
        if let Some(active) = active
            && same_target(&active.target, &envelope.target)
        {
            active.controller.abort("RPC request cancelled");
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one upstream method ported 1:1; splitting it would scatter the response contract"
    )]
    async fn handle_request(
        self: &Rc<Self>,
        state: &Rc<ConnectionCore>,
        envelope: RequestEnvelope,
    ) {
        if state.active_requests.borrow().contains_key(&envelope.id) {
            self.send_message(
                state,
                ServerMessage::Response(ResponseEnvelope::Failure(ResponseFailure {
                    id: envelope.id.clone(),
                    error: ProtocolError {
                        code: "invalid_request".to_string(),
                        message: "Request ID is already active".to_string(),
                    },
                })),
            )
            .await;
            return;
        }
        let Ok(call) = parse_service_call(&envelope.call) else {
            self.send_message(
                state,
                ServerMessage::Response(ResponseEnvelope::Failure(ResponseFailure {
                    id: envelope.id.clone(),
                    error: ProtocolError {
                        code: "invalid_request".to_string(),
                        message: "Invalid service call".to_string(),
                    },
                })),
            )
            .await;
            return;
        };
        let (context, controller) = with_cancel(&placeholder_context());
        let active = Rc::new(ActiveRequest {
            controller,
            target: envelope.target.clone(),
        });
        state
            .active_requests
            .borrow_mut()
            .insert(envelope.id.clone(), Rc::clone(&active));
        let control = decode_service_control_call(&call);
        let subscribing = match &control {
            Some(ServiceControlCall::Subscribe {
                subscription_id, ..
            }) => Some(subscription_id.clone()),
            _ => None,
        };
        let pending_updates: Rc<RefCell<Vec<ServiceProviderUpdate>>> =
            Rc::new(RefCell::new(Vec::new()));
        let subscription_ready = Rc::new(Cell::new(subscribing.is_none()));
        let responded = Cell::new(false);
        let installed = Cell::new(false);
        let publish: ServicePublisher = {
            let core = Rc::downgrade(self);
            let state = Rc::clone(state);
            let subscribing = subscribing.clone();
            let pending_updates = Rc::clone(&pending_updates);
            let subscription_ready = Rc::clone(&subscription_ready);
            Rc::new(
                move |subscription_id: &str, update: &ServiceProviderUpdate, _: &Context| {
                    if let Some(subscribed) = &subscribing
                        && subscribed == subscription_id
                        && !subscription_ready.get()
                    {
                        pending_updates.borrow_mut().push(update.clone());
                        return ready(Ok(()));
                    }
                    let core = core.clone();
                    let state = Rc::clone(&state);
                    let subscription_id = subscription_id.to_string();
                    let update = update.clone();
                    boxed(async move {
                        match core.upgrade() {
                            Some(core) => {
                                core.send_service_update(&state, &subscription_id, &update)
                                    .await
                            }
                            None => Ok(()),
                        }
                    })
                },
            )
        };
        let is_session_target = matches!(envelope.target, RpcTarget::Session(_));
        let outcome: Result<(), Failure> = async {
            if target_server_id(&envelope.target).as_str() != self.server_id.as_str() {
                return Err(Failure::Server(ServerError::wrong_server()));
            }
            if let Some(subscription_id) = &subscribing
                && state
                    .service_state_encoders
                    .borrow()
                    .contains_key(subscription_id)
            {
                return Err(Failure::Validation(ProtocolValidationError::new(format!(
                    "Duplicate service subscription {subscription_id}"
                ))));
            }
            let mut result: Option<JsonValue> = if is_session_target {
                self.sessions
                    .execute_service_call(
                        &state.token,
                        envelope.target.clone(),
                        call,
                        publish,
                        context,
                    )
                    .await?
            } else {
                let services = state.server_services.borrow().clone();
                match services {
                    Some(services) => services.invoke_service(call, publish, context).await?,
                    None => {
                        // `call` is still owned here, upstream's un-moved
                        // throw site.
                        return Err(Failure::Validation(ProtocolValidationError::new(format!(
                            "Unknown service member {}.{}",
                            call.service_id, call.member
                        ))));
                    }
                }
            };
            if let Some(subscription_id) = &subscribing {
                let Some(value) = result else {
                    return Err(Failure::Validation(ProtocolValidationError::new(
                        "Service subscription did not return a snapshot",
                    )));
                };
                let mut encoder = create_service_state_encoder();
                let snapshot = parse_service_subscription_snapshot(&value)
                    .map_err(|error| Failure::Other(Rc::new(error)))?;
                let wire = encoder
                    .encode_snapshot(&snapshot)
                    .map_err(|error| Failure::Other(Rc::new(error)))?;
                result = Some(wire_snapshot_to_json(&wire));
                state
                    .service_state_encoders
                    .borrow_mut()
                    .insert(subscription_id.clone(), encoder);
                installed.set(true);
            } else if let Some(ServiceControlCall::Unsubscribe { subscription_id }) = &control {
                state
                    .service_state_encoders
                    .borrow_mut()
                    .remove(subscription_id);
            }
            self.send_message(
                state,
                match result {
                    Some(result) => ServerMessage::Response(ResponseEnvelope::Success(
                        pi_protocol::ResponseSuccess {
                            id: envelope.id.clone(),
                            result: Some(result),
                        },
                    )),
                    None => ServerMessage::Response(ResponseEnvelope::Success(
                        pi_protocol::ResponseSuccess {
                            id: envelope.id.clone(),
                            result: None,
                        },
                    )),
                },
            )
            .await;
            responded.set(true);
            if let Some(subscription_id) = &subscribing {
                while !pending_updates.borrow().is_empty() {
                    let update = pending_updates.borrow_mut().remove(0);
                    self.send_service_update(state, subscription_id, &update)
                        .await?;
                }
                subscription_ready.set(true);
            }
            Ok(())
        }
        .await;
        if let Err(error) = outcome {
            if let Some(subscription_id) = &subscribing
                && installed.get()
                && !responded.get()
            {
                state
                    .service_state_encoders
                    .borrow_mut()
                    .remove(subscription_id);
            }
            if responded.get() {
                self.report_error(&error);
                self.close_connection(&state.connection, None).await;
                self.disconnect(state);
            } else {
                let error = if active.controller.signal().aborted() {
                    ProtocolError {
                        code: "cancelled".to_string(),
                        message: "RPC request cancelled".to_string(),
                    }
                } else {
                    self.to_protocol_error(&error)
                };
                self.send_message(
                    state,
                    ServerMessage::Response(ResponseEnvelope::Failure(ResponseFailure {
                        id: envelope.id.clone(),
                        error,
                    })),
                )
                .await;
            }
        }
        let still_active = state
            .active_requests
            .borrow()
            .get(&envelope.id)
            .is_some_and(|current| Rc::ptr_eq(current, &active));
        if still_active {
            state.active_requests.borrow_mut().remove(&envelope.id);
        }
        self.sessions.clear_session_interest(&state.token);
    }

    fn transport_closed(self: &Rc<Self>, state: &Rc<ConnectionCore>) {
        if !state.disconnected.get() && state.stage.get() != ConnectionStage::Closing {
            match state.decoder.borrow_mut().end() {
                Ok(()) => {}
                Err(error) => self.report_error(&Failure::Validation(error)),
            }
        }
        self.disconnect(state);
    }

    fn disconnect(self: &Rc<Self>, state: &Rc<ConnectionCore>) {
        if state.disconnected.get() {
            return;
        }
        state.disconnected.set(true);
        state.stage.set(ConnectionStage::Closed);
        state.clear_handshake_timeout();
        for active in state.active_requests.borrow().values() {
            active.controller.abort("Client disconnected");
        }
        state.active_requests.borrow_mut().clear();
        state.service_state_encoders.borrow_mut().clear();
        if self
            .connections
            .borrow_mut()
            .remove(&state.token.key())
            .is_some()
        {
            self.notify_connection_count_changed();
        }
        let server_services = state.server_services.borrow_mut().take();
        let core = Rc::downgrade(self);
        let token = state.token.clone();
        tokio::task::spawn_local(async move {
            let Some(core) = core.upgrade() else {
                return;
            };
            let mut handles = vec![{
                let core = Rc::clone(&core);
                let token = token.clone();
                tokio::task::spawn_local(async move {
                    core.sessions
                        .disconnect(&token, placeholder_context())
                        .await
                })
            }];
            if let Some(services) = server_services {
                handles.push(tokio::task::spawn_local(async move {
                    services.release(placeholder_context()).await
                }));
            }
            for handle in handles {
                if let Ok(Err(error)) = handle.await {
                    core.report_error(&error);
                }
            }
        });
    }

    async fn send_service_update(
        self: &Rc<Self>,
        state: &Rc<ConnectionCore>,
        subscription_id: &str,
        update: &ServiceProviderUpdate,
    ) -> Result<(), Failure> {
        let encoded = {
            let mut encoders = state.service_state_encoders.borrow_mut();
            let Some(encoder) = encoders.get_mut(subscription_id) else {
                return Ok(());
            };
            encoder
                .encode_update(update)
                .map(|wire| wire_update_to_json(&wire))
                .map_err(|error| Failure::Other(Rc::new(error)))?
        };
        self.send_message(
            state,
            ServerMessage::ServiceEvent(pi_protocol::ServiceEventEnvelope {
                subscription_id: subscription_id.to_string(),
                update: encoded,
            }),
        )
        .await;
        Ok(())
    }

    async fn send_message(
        self: &Rc<Self>,
        state: &Rc<ConnectionCore>,
        message: ServerMessage,
    ) -> bool {
        if state.disconnected.get() || state.connection.closed() {
            return false;
        }
        let frame = match encode_server_message(
            &message,
            FrameDecoderOptions {
                max_frame_length: self.max_frame_length,
            },
        ) {
            Ok(frame) => frame,
            Err(error) => {
                self.report_error(&Failure::Validation(error));
                self.close_connection(&state.connection, None).await;
                self.disconnect(state);
                return false;
            }
        };
        match state.connection.send(frame).await {
            Ok(()) => true,
            Err(error) => {
                self.report_error(&error);
                self.close_connection(&state.connection, None).await;
                self.disconnect(state);
                false
            }
        }
    }

    async fn fail_protocol(self: &Rc<Self>, state: &Rc<ConnectionCore>, error: ProtocolError) {
        if state.disconnected.get()
            || matches!(
                state.stage.get(),
                ConnectionStage::Closing | ConnectionStage::Closed
            )
        {
            return;
        }
        state.stage.set(ConnectionStage::Closing);
        state.clear_handshake_timeout();
        let final_frame = match encode_server_message(
            &ServerMessage::HelloError(ServerHelloError { error }),
            FrameDecoderOptions {
                max_frame_length: self.max_frame_length,
            },
        ) {
            Ok(frame) => Some(frame),
            Err(encode_error) => {
                self.report_error(&Failure::Validation(encode_error));
                None
            }
        };
        self.close_connection(&state.connection, final_frame).await;
        self.disconnect(state);
    }

    async fn close_server_state(self: &Rc<Self>) -> Result<(), Failure> {
        let connections: Vec<Rc<ConnectionCore>> =
            self.connections.borrow().values().cloned().collect();
        for state in &connections {
            state.stage.set(ConnectionStage::Closing);
            state.clear_handshake_timeout();
        }
        let mut handles = Vec::with_capacity(connections.len());
        for state in &connections {
            let core = Rc::clone(self);
            let connection = Rc::clone(&state.connection);
            handles.push(tokio::task::spawn_local(async move {
                core.close_connection(&connection, None).await;
            }));
        }
        for handle in handles {
            let _ = handle.await;
        }
        for state in &connections {
            self.disconnect(state);
        }
        let cleanup = self.sessions.close(background_context()).await;
        self.connections.borrow_mut().clear();
        cleanup
    }

    async fn close_connection(
        self: &Rc<Self>,
        connection: &Rc<dyn ByteConnection>,
        final_chunk: Option<Vec<u8>>,
    ) {
        if let Err(error) = connection.close(final_chunk).await {
            self.report_error(&error);
        }
    }

    fn to_protocol_error(self: &Rc<Self>, error: &Failure) -> ProtocolError {
        match error {
            Failure::Server(_) | Failure::Remote(_) | Failure::Validation(_) => error.wire_error(),
            _ => {
                self.report_error(error);
                error.wire_error()
            }
        }
    }

    fn notify_connection_count_changed(self: &Rc<Self>) {
        if let Some(callback) = &self.on_connection_count_changed {
            let count = self.connections.borrow().len();
            callback(count);
        }
    }

    pub(crate) fn report_error(self: &Rc<Self>, error: &Failure) {
        if let Some(on_error) = &self.on_error {
            on_error(error);
        }
    }

    fn settle_closed(&self, result: Result<(), Failure>) {
        self.closed_latch.settle(result);
    }

    pub(crate) async fn publish_attachment(
        self: &Rc<Self>,
        client: &ClientToken,
        attachment: Option<pi_protocol::SessionTarget>,
        _context: &Context,
    ) {
        let state = self.connections.borrow().get(&client.key()).cloned();
        if let Some(state) = state {
            self.send_message(
                &state,
                ServerMessage::Attachment(AttachmentEnvelope { attachment }),
            )
            .await;
        }
    }
}

/// The presentation the handshake hands the host's service factory, routed
/// back into the owning server's Session router.
struct CorePresentation<H: ServerHost> {
    core: Weak<ServerCore<H>>,
    state: Rc<ConnectionCore>,
}

impl<H: ServerHost + 'static> RoutedServerPresentation for CorePresentation<H> {
    fn attach_session(
        &self,
        session_id: &str,
        context: Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        self.core.upgrade().map_or_else(
            || ready(Err(Failure::Server(ServerError::server_draining()))),
            |core| {
                // Recorded synchronously at the call, upstream's attach
                // request whose synchronous acquire-check joins the open
                // still in flight.
                core.sessions
                    .note_session_interest(&self.state.token, session_id);
                core.sessions
                    .attach_client(&self.state.token, session_id.to_string(), context)
            },
        )
    }

    fn detach_session(&self, context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        self.core.upgrade().map_or_else(
            || ready(Err(Failure::Server(ServerError::server_draining()))),
            |core| core.sessions.detach_client(&self.state.token, context),
        )
    }

    fn prepare_session_removal(
        &self,
        session_id: &str,
        context: Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        self.core.upgrade().map_or_else(
            || ready(Err(Failure::Server(ServerError::server_draining()))),
            |core| core.sessions.remove_session(session_id, context),
        )
    }
}

/// Whether two routed targets fence to the same server, session, and
/// attachment, upstream's `sameTarget`.
fn same_target(left: &RpcTarget, right: &RpcTarget) -> bool {
    match (left, right) {
        (RpcTarget::Server(left), RpcTarget::Server(right)) => left.server_id == right.server_id,
        (RpcTarget::Session(left), RpcTarget::Session(right)) => {
            left.server_id == right.server_id
                && left.session_id == right.session_id
                && left.attachment_id == right.attachment_id
        }
        _ => false,
    }
}

/// The routed target's server identity, shared by both target shapes.
const fn target_server_id(target: &RpcTarget) -> &ServerId {
    match target {
        RpcTarget::Server(target) => &target.server_id,
        RpcTarget::Session(target) => &target.server_id,
    }
}
#[allow(
    unused_imports,
    reason = "the alias keeps the SPI import surface honest"
)]
use crate::listener::ServerListener as _;
