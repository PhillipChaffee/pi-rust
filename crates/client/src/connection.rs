//! The connection state machine, ported from upstream `src/connection.ts`.
//!
//! One lifecycle at a time: a transport factory opens a byte transport, the
//! client hello goes out, and the first server message either completes the
//! handshake or fails the connection. Server data before the hello is sent,
//! out-of-order handshake messages, and framing or schema failures all
//! latch the connection into [`ConnectionState::Disconnected`].

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_chord::future::{LocalBoxFuture, boxed};
use pi_protocol::{
    ClientHello, ClientMessage, FrameDecoderOptions, PROTOCOL_VERSION, ProtocolValidationError,
    ResponseEnvelope, ServerHello, ServerMessage, ServerMessageDecoder, encode_client_message,
};
use tokio::sync::oneshot;

use crate::errors::{ClientError, to_disconnected_error};
use crate::transport::{ByteTransport, ByteTransportFactory, ByteTransportHandlers};
use crate::types::{ConnectionState, ConnectionStateChange};

/// The upper bound a `maxFrameLength` may take, upstream's `MAX_UINT32`
/// guard against frame headers that cannot be honored.
pub(crate) const MAX_UINT32: usize = u32::MAX as usize;

/// The messages a connected client processes, upstream's
/// `Exclude<ServerMessage, {type: hello | hello_error}>`: the connection
/// consumes the handshake shapes itself and hands the rest to its owner.
#[derive(Debug, Clone)]
pub(crate) enum ServerPayload {
    /// A routed call's answer.
    Response(ResponseEnvelope),
    /// An out-of-band subscription update.
    ServiceEvent(pi_protocol::ServiceEventEnvelope),
    /// An out-of-band attachment route update.
    Attachment(pi_protocol::AttachmentEnvelope),
}

/// The connection's construction options, upstream's `ConnectionOptions`.
///
/// Upstream validates `maxFrameLength` in the connection constructor; the
/// port hoists that validation to
/// [`validate_max_frame_length`](crate::client::validate_max_frame_length),
/// which the client constructor runs first, so this struct carries the
/// validated bound directly.
pub(crate) struct ConnectionOptions {
    /// Creates the transport every connection attempt opens.
    pub(crate) transport_factory: ByteTransportFactory,
    /// The logical server identity the handshake must answer with.
    pub(crate) server_id: String,
    /// The validated frame-length bound every encoder and decoder shares.
    pub(crate) max_frame_length: usize,
    /// Runs once the server hello validates; failures fail the connection.
    pub(crate) on_handshake: Box<dyn Fn(&ServerHello)>,
    /// Delivers post-handshake messages in decode order.
    pub(crate) on_message: Box<dyn Fn(ServerPayload)>,
    /// Reports every lifecycle transition.
    pub(crate) on_state_change: Box<dyn Fn(&ConnectionStateChange)>,
}

/// One connection's shared state. The [`Connection`] handle and every
/// transport handler reach the lifecycle through it; the handlers hold it
/// weakly so the owning client's drop ends the cycle.
pub(crate) struct ConnCore {
    pub(crate) factory: ByteTransportFactory,
    pub(crate) on_handshake: Box<dyn Fn(&ServerHello)>,
    pub(crate) on_message: Box<dyn Fn(ServerPayload)>,
    pub(crate) on_state_change: Box<dyn Fn(&ConnectionStateChange)>,
    pub(crate) server_id: String,
    pub(crate) max_frame_length: usize,
    pub(crate) lifecycle: RefCell<Lifecycle>,
    pub(crate) sequence: Cell<u64>,
}

/// The connection handle; the client owns one and forwards its surface
/// through the shared core.
#[derive(Clone)]
pub(crate) struct Connection {
    core: Rc<ConnCore>,
}

/// One connection attempt's live state, upstream's `ActiveConnection`
/// fields plus the lifecycle union.
pub(crate) enum Lifecycle {
    Disconnected,
    Connecting {
        id: u64,
        decoder: ServerMessageDecoder,
        handshake: Option<oneshot::Sender<Result<ServerHello, ClientError>>>,
        transport: Option<Rc<dyn ByteTransport>>,
    },
    Connected {
        id: u64,
        decoder: ServerMessageDecoder,
        handshake: Option<oneshot::Sender<Result<ServerHello, ClientError>>>,
        transport: Rc<dyn ByteTransport>,
    },
}

impl Connection {
    /// Builds a connection over the given options.
    pub(crate) fn new(options: ConnectionOptions) -> Self {
        Self {
            core: Rc::new(ConnCore {
                factory: options.transport_factory,
                on_handshake: options.on_handshake,
                on_message: options.on_message,
                on_state_change: options.on_state_change,
                server_id: options.server_id,
                max_frame_length: options.max_frame_length,
                lifecycle: RefCell::new(Lifecycle::Disconnected),
                sequence: Cell::new(0),
            }),
        }
    }

    /// The current lifecycle state.
    #[must_use]
    pub(crate) fn state(&self) -> ConnectionState {
        self.core.state()
    }

    /// The frame-length bound every encoder shares.
    #[must_use]
    pub(crate) fn max_frame_length(&self) -> usize {
        self.core.max_frame_length
    }

    /// Opens a transport and resolves with the server hello; see
    /// [`ConnCore::connect`].
    pub(crate) fn connect(&self) -> LocalBoxFuture<Result<ServerHello, ClientError>> {
        ConnCore::connect(&self.core)
    }

    /// Disconnects with the default reason; see [`ConnCore::disconnect`].
    pub(crate) fn disconnect(&self) {
        self.core.disconnect();
    }

    /// Fails and closes the connection; see [`ConnCore::fail`].
    pub(crate) fn fail(&self, error: &ClientError) {
        self.core.fail(error);
    }

    /// Sends one framed message; see [`ConnCore::send`].
    pub(crate) fn send(&self, frame: Vec<u8>) -> Result<(), ClientError> {
        ConnCore::send(&self.core, frame)
    }
}

impl ConnCore {
    /// The current lifecycle state.
    #[must_use]
    pub(crate) fn state(&self) -> ConnectionState {
        match &*self.lifecycle.borrow() {
            Lifecycle::Disconnected => ConnectionState::Disconnected,
            Lifecycle::Connecting { .. } => ConnectionState::Connecting,
            Lifecycle::Connected { .. } => ConnectionState::Connected,
        }
    }

    /// Opens a transport, sends the client hello, and resolves with the
    /// server hello.
    ///
    /// # Errors
    /// Upstream rejects through the handshake promise; the future carries
    /// the same failures: a rejected transport factory, a failed hello
    /// send, a server hello mismatch or error, and protocol or framing
    /// failures on the way.
    pub(crate) fn connect(core: &Rc<Self>) -> LocalBoxFuture<Result<ServerHello, ClientError>> {
        let current = core.state();
        if current != ConnectionState::Disconnected {
            return boxed(std::future::ready(Err(ClientError::disconnected_with(
                format!("Client is already {}", current.as_str()),
            ))));
        }
        let id = core.sequence.get() + 1;
        core.sequence.set(id);
        let decoder = match ServerMessageDecoder::new(FrameDecoderOptions {
            max_frame_length: core.max_frame_length,
        }) {
            Ok(decoder) => decoder,
            Err(error) => {
                return boxed(std::future::ready(Err(ClientError::other(
                    error.to_string(),
                ))));
            }
        };
        let (sender, receiver) = oneshot::channel();
        core.lifecycle.replace(Lifecycle::Connecting {
            id,
            decoder,
            handshake: Some(sender),
            transport: None,
        });
        (core.on_state_change)(&ConnectionStateChange {
            state: ConnectionState::Connecting,
            error: None,
        });
        let handlers = ByteTransportHandlers {
            on_data: {
                let core = Rc::downgrade(core);
                Rc::new(move |chunk| {
                    if let Some(core) = core.upgrade() {
                        core.handle_data(id, chunk);
                    }
                })
            },
            on_close: {
                let core = Rc::downgrade(core);
                Rc::new(move || {
                    if let Some(core) = core.upgrade()
                        && core.is_current(id)
                    {
                        core.handle_close();
                    }
                })
            },
            on_error: {
                let core = Rc::downgrade(core);
                Rc::new(move |error| {
                    if let Some(core) = core.upgrade()
                        && core.is_current(id)
                    {
                        core.fail_and_close(&to_disconnected_error(error));
                    }
                })
            },
        };
        tokio::task::spawn_local(Self::open_transport(Rc::clone(core), id, handlers));
        boxed(async move {
            receiver
                .await
                .unwrap_or_else(|_| Err(ClientError::disconnected()))
        })
    }

    /// Disconnects with the default reason, upstream's
    /// `disconnect(reason = "Client disconnected")`.
    pub(crate) fn disconnect(&self) {
        self.fail_and_close(&ClientError::disconnected_with("Client disconnected"));
    }

    /// Fails and closes the connection, upstream's `fail(error)`.
    pub(crate) fn fail(&self, error: &ClientError) {
        self.fail_and_close(error);
    }

    /// Sends one framed message on the connected transport.
    ///
    /// # Errors
    /// Upstream throws when the connection is not connected; the port
    /// returns the same disconnected failure.
    pub(crate) fn send(core: &Rc<Self>, frame: Vec<u8>) -> Result<(), ClientError> {
        let transport = match &*core.lifecycle.borrow() {
            Lifecycle::Connected { transport, .. } => transport.clone(),
            Lifecycle::Disconnected | Lifecycle::Connecting { .. } => {
                return Err(ClientError::disconnected());
            }
        };
        let sending = transport.send(frame);
        let core = Rc::downgrade(core);
        tokio::task::spawn_local(async move {
            if let Err(error) = sending.await {
                let Some(core) = core.upgrade() else {
                    return;
                };
                let same = {
                    let lifecycle = core.lifecycle.borrow();
                    match &*lifecycle {
                        Lifecycle::Connected {
                            transport: current, ..
                        } => Rc::ptr_eq(current, &transport),
                        Lifecycle::Disconnected | Lifecycle::Connecting { .. } => false,
                    }
                };
                if same {
                    core.fail_and_close(&to_disconnected_error(&error));
                }
            }
        });
        Ok(())
    }

    async fn open_transport(core: Rc<Self>, id: u64, handlers: ByteTransportHandlers) {
        let transport = match (core.factory)(handlers).await {
            Ok(transport) => transport,
            Err(error) => {
                if core.is_current(id) {
                    core.fail(&to_disconnected_error(&error));
                }
                return;
            }
        };
        let current = match &*core.lifecycle.borrow() {
            Lifecycle::Connecting { id: current, .. } => Some(*current),
            Lifecycle::Disconnected | Lifecycle::Connected { .. } => None,
        };
        if current != Some(id) {
            transport.close();
            return;
        }
        match &mut *core.lifecycle.borrow_mut() {
            Lifecycle::Connecting {
                transport: slot, ..
            } => *slot = Some(transport.clone()),
            Lifecycle::Disconnected | Lifecycle::Connected { .. } => {
                transport.close();
                return;
            }
        }
        let frame = match encode_client_message(
            &ClientMessage::Hello(ClientHello {
                version: PROTOCOL_VERSION.into(),
            }),
            FrameDecoderOptions {
                max_frame_length: core.max_frame_length,
            },
        ) {
            Ok(frame) => frame,
            Err(error) => {
                if core.is_current(id) {
                    core.fail_and_close(&to_disconnected_error(&ClientError::Protocol(error)));
                }
                return;
            }
        };
        if let Err(error) = transport.send(frame).await
            && core.is_current(id)
        {
            core.fail_and_close(&to_disconnected_error(&error));
        }
    }

    fn handle_data(&self, id: u64, chunk: &[u8]) {
        enum Data {
            Stale,
            BeforeHello,
            Messages(Vec<ServerMessage>),
            DecodeFailed(ProtocolValidationError),
        }
        let outcome = {
            let mut lifecycle = self.lifecycle.borrow_mut();
            match &mut *lifecycle {
                Lifecycle::Connecting {
                    id: current,
                    transport,
                    decoder,
                    ..
                } if *current == id => {
                    if transport.is_none() {
                        Data::BeforeHello
                    } else {
                        match decoder.push(chunk) {
                            Ok(messages) => Data::Messages(messages),
                            Err(error) => Data::DecodeFailed(error),
                        }
                    }
                }
                Lifecycle::Connected {
                    id: current,
                    decoder,
                    ..
                } if *current == id => match decoder.push(chunk) {
                    Ok(messages) => Data::Messages(messages),
                    Err(error) => Data::DecodeFailed(error),
                },
                _ => Data::Stale,
            }
        };
        match outcome {
            Data::Stale => {}
            Data::BeforeHello => {
                self.fail_and_close(&ClientError::Protocol(ProtocolValidationError::new(
                    "Received server data before the client hello was sent",
                )));
            }
            Data::DecodeFailed(error) => self.fail_and_close(&ClientError::Protocol(error)),
            Data::Messages(messages) => {
                for message in messages {
                    if self.state() == ConnectionState::Disconnected {
                        return;
                    }
                    self.handle_message(message);
                }
            }
        }
    }

    fn handle_message(&self, message: ServerMessage) {
        let connecting = matches!(&*self.lifecycle.borrow(), Lifecycle::Connecting { .. });
        if connecting {
            match &message {
                ServerMessage::HelloError(error) => {
                    self.fail_and_close(&ClientError::Server(error.error.clone()));
                }
                ServerMessage::Hello(hello) => {
                    let id = match &*self.lifecycle.borrow() {
                        Lifecycle::Connecting { id, .. } => *id,
                        _ => 0,
                    };
                    self.complete_handshake(id, &hello.clone());
                }
                ServerMessage::Response(_)
                | ServerMessage::ServiceEvent(_)
                | ServerMessage::Attachment(_) => {
                    self.fail_and_close(&ClientError::Protocol(ProtocolValidationError::new(
                        "Expected server hello as first message",
                    )));
                }
            }
            return;
        }
        if !matches!(&*self.lifecycle.borrow(), Lifecycle::Connected { .. }) {
            return;
        }
        match message {
            ServerMessage::Hello(_) | ServerMessage::HelloError(_) => {
                self.fail_and_close(&ClientError::Protocol(ProtocolValidationError::new(
                    "Unexpected handshake message",
                )));
            }
            ServerMessage::Response(envelope) => {
                (self.on_message)(ServerPayload::Response(envelope));
            }
            ServerMessage::ServiceEvent(envelope) => {
                (self.on_message)(ServerPayload::ServiceEvent(envelope));
            }
            ServerMessage::Attachment(envelope) => {
                (self.on_message)(ServerPayload::Attachment(envelope));
            }
        }
    }

    fn complete_handshake(&self, id: u64, hello: &ServerHello) {
        let transitioned = {
            let mut lifecycle = self.lifecycle.borrow_mut();
            match std::mem::replace(&mut *lifecycle, Lifecycle::Disconnected) {
                Lifecycle::Connecting {
                    id: connecting_id,
                    decoder,
                    handshake,
                    transport: Some(transport),
                } if connecting_id == id => {
                    *lifecycle = Lifecycle::Connected {
                        id: connecting_id,
                        decoder,
                        handshake,
                        transport,
                    };
                    true
                }
                other => {
                    *lifecycle = other;
                    false
                }
            }
        };
        if !transitioned {
            self.fail_and_close(&ClientError::Protocol(ProtocolValidationError::new(
                "Received server hello before the client hello was sent",
            )));
            return;
        }
        if self.server_id.as_str() != hello.server_id.as_str() {
            self.fail_and_close(&ClientError::Protocol(ProtocolValidationError::new(
                format!(
                    "Connected server \"{}\" does not match \"{}\"",
                    hello.server_id, self.server_id
                ),
            )));
            return;
        }
        let settled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (self.on_handshake)(hello);
        }));
        if let Err(payload) = settled {
            if self.is_current(id) {
                self.fail_and_close(&ClientError::other(panic_message(&*payload)));
            }
            return;
        }
        if !self.is_current(id) {
            return;
        }
        (self.on_state_change)(&ConnectionStateChange {
            state: ConnectionState::Connected,
            error: None,
        });
        if !self.is_current(id) {
            return;
        }
        let handshake = match &mut *self.lifecycle.borrow_mut() {
            Lifecycle::Connected { handshake, .. } => handshake.take(),
            Lifecycle::Disconnected | Lifecycle::Connecting { .. } => None,
        };
        if let Some(sender) = handshake {
            let _ = sender.send(Ok(hello.clone()));
        }
    }

    fn handle_close(&self) {
        let error = {
            let mut lifecycle = self.lifecycle.borrow_mut();
            match &mut *lifecycle {
                Lifecycle::Disconnected => return,
                Lifecycle::Connecting { decoder, .. } | Lifecycle::Connected { decoder, .. } => {
                    match decoder.end() {
                        Ok(()) => ClientError::disconnected_with("Byte transport closed"),
                        Err(error) => ClientError::Protocol(error),
                    }
                }
            }
        };
        self.fail(&error);
    }

    fn fail_and_close(&self, error: &ClientError) {
        let transport = match &*self.lifecycle.borrow() {
            Lifecycle::Disconnected => None,
            Lifecycle::Connecting { transport, .. } => transport.as_ref().map(Rc::clone),
            Lifecycle::Connected { transport, .. } => Some(transport.clone()),
        };
        self.fail_inner(error);
        if let Some(transport) = transport {
            transport.close();
        }
    }

    fn fail_inner(&self, error: &ClientError) {
        let handshake = {
            let mut lifecycle = self.lifecycle.borrow_mut();
            if matches!(&*lifecycle, Lifecycle::Disconnected) {
                return;
            }
            match std::mem::replace(&mut *lifecycle, Lifecycle::Disconnected) {
                Lifecycle::Connecting { handshake, .. }
                | Lifecycle::Connected { handshake, .. } => handshake,
                Lifecycle::Disconnected => None,
            }
        };
        if let Some(sender) = handshake {
            let _ = sender.send(Err(error.clone()));
        }
        (self.on_state_change)(&ConnectionStateChange {
            state: ConnectionState::Disconnected,
            error: Some(error.clone()),
        });
    }

    fn is_current(&self, id: u64) -> bool {
        match &*self.lifecycle.borrow() {
            Lifecycle::Disconnected => false,
            Lifecycle::Connecting { id: current, .. }
            | Lifecycle::Connected { id: current, .. } => *current == id,
        }
    }
}

/// Recovers a panic's message, upstream's `Error` instances that listeners
/// throw; the chord port keeps the same helper for its fan-outs.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "panic".to_string()
}
