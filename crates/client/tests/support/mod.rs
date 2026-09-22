//! Test fixtures shared by the client suites, porting upstream's
//! `test/support.ts` plus the local-task runner every suite drives the
//! single-threaded client through.

#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]
#![allow(
    clippy::unwrap_used,
    reason = "test fixtures unwrap invariants the case's assertions cover"
)]
#![allow(
    dead_code,
    reason = "the fixtures are shared across the suites; each test binary links its own slice"
)]
#![allow(
    unreachable_pub,
    reason = "the fixture module is shared across test binaries, never exported"
)]

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::rc::Rc;

use pi_client::{
    ByteTransport, ByteTransportHandlers, Client, ClientError, ConnectionStateChange,
    ServiceSubscription,
};
use pi_protocol::{
    AttachmentEnvelope, ClientMessage, ClientMessageDecoder, FrameDecoderOptions, ResponseEnvelope,
    ServerHello, ServerId, ServerMessage, encode_server_message,
};
use tokio::sync::oneshot;

use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::types::{JsonValue, RemoteServiceTransport, ServiceMode, ServiceProviderUpdate};

#[cfg(unix)]
pub mod unix;

/// The canonical server identity every fixture connects to, upstream's
/// suite constant.
pub const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

/// The routed server-wide target the suites call through, upstream's
/// `serverTarget`.
#[must_use]
pub fn server_target() -> pi_protocol::RpcTarget {
    pi_protocol::RpcTarget::Server(pi_protocol::ServerTarget {
        server_id: ServerId::new(SERVER_ID).unwrap(),
    })
}

/// Builds a routed session target over this suite's server identity.
#[must_use]
pub fn session_target(session_id: &str, attachment_id: &str) -> pi_protocol::RpcTarget {
    pi_protocol::RpcTarget::Session(pi_protocol::SessionTarget {
        server_id: ServerId::new(SERVER_ID).unwrap(),
        session_id: session_id.to_string(),
        attachment_id: attachment_id.to_string(),
    })
}

/// Restates one typed service call, the literal the request cases drive.
#[must_use]
pub fn service_call(
    service_id: &str,
    member: &str,
    args: Vec<JsonValue>,
) -> pi_chord::types::ServiceCall {
    pi_chord::types::ServiceCall {
        service_id: service_id.to_string(),
        instance: None,
        member: member.to_string(),
        args,
    }
}

/// Encodes one server message with the default frame options, the raw
/// frame the probe transports deliver.
#[must_use]
pub fn server_frame(message: &ServerMessage) -> Vec<u8> {
    encode_server_message(message, FrameDecoderOptions::default()).expect("the frame encodes")
}

/// The request envelope one recorded client message carries.
#[must_use]
pub const fn request_call_of(message: &ClientMessage) -> Option<&pi_protocol::RequestEnvelope> {
    match message {
        ClientMessage::Request(envelope) => Some(envelope),
        _ => None,
    }
}

/// The request envelope at `index` of the server's recorded messages, the
/// access the call-shape assertions run through.
#[must_use]
pub fn request_envelope(server: &MemoryByteServer, index: usize) -> pi_protocol::RequestEnvelope {
    let messages = server.messages();
    let Some(envelope) = request_call_of(&messages[index]) else {
        panic!("the request arrives");
    };
    envelope.clone()
}

/// Attaches a session through the session-management control surface,
/// upstream's `attachClient`.
pub async fn attach_client(client: &Client, server: &Rc<MemoryByteServer>, session_id: &str) {
    let expected = server.messages().len() + 1;
    let attaching = client.request(
        &server_target(),
        &service_call(
            "pi.session-management",
            "attach",
            vec![JsonValue::string(session_id)],
        ),
        None,
    );
    server.wait_for_messages(expected).await;
    let request_id = request_call_of(server.messages().last().expect("attach request arrives"))
        .unwrap_or_else(|| panic!("Missing attach request"))
        .id
        .clone();
    server.send(&ServerMessage::Attachment(AttachmentEnvelope {
        attachment: Some(pi_protocol::SessionTarget {
            server_id: ServerId::new(SERVER_ID).unwrap(),
            session_id: session_id.to_string(),
            attachment_id: format!("attachment-{session_id}"),
        }),
    }));
    send_success(server, &request_id, Some(JsonValue::Null));
    attaching.await.expect("attach resolves");
}

/// The `r`-form revision replace op the snapshot members publish,
/// upstream's short form.
#[must_use]
pub fn revision_replace_op(revision: i64) -> JsonValue {
    JsonValue::Array(vec![
        JsonValue::string("r"),
        pi_chord::services::wire::object(vec![("revision", JsonValue::Number(revision.into()))]),
    ])
}

/// The `s`-form revision set ops the state publications carry.
#[must_use]
pub fn revision_ops(revision: i64) -> Vec<JsonValue> {
    vec![JsonValue::Array(vec![
        JsonValue::string("s"),
        JsonValue::Array(vec![JsonValue::string("revision")]),
        JsonValue::Number(revision.into()),
    ])]
}

/// One state member of the wire snapshot the subscription cases fabricate.
#[must_use]
pub fn state_member(ops: Vec<JsonValue>) -> JsonValue {
    pi_chord::services::wire::object(vec![
        ("name", JsonValue::string("state")),
        ("kind", JsonValue::string("state")),
        ("sequence", JsonValue::Number(0i64.into())),
        ("ops", JsonValue::Array(ops)),
    ])
}

/// The snapshot response carrying `members`, the literal the subscription
/// cases answer with.
#[must_use]
pub fn snapshot_with_members(members: Vec<JsonValue>) -> JsonValue {
    let instances = JsonValue::Array(vec![pi_chord::services::wire::object(vec![(
        "members",
        JsonValue::Array(members),
    )])]);
    pi_chord::services::wire::object(vec![
        ("serviceId", JsonValue::string("pi.models")),
        ("mode", JsonValue::string("singleton")),
        ("instances", instances),
    ])
}

/// The snapshot response the subscription cases fabricate, the same
/// literal the buffering case decodes.
#[must_use]
pub fn snapshot_result() -> JsonValue {
    snapshot_with_members(vec![state_member(vec![revision_replace_op(0)])])
}

/// A state publication on the suite's subscription, upstream's update
/// envelope.
#[must_use]
pub fn state_update(sequence: u64, ops: Vec<JsonValue>) -> pi_protocol::ServiceEventEnvelope {
    state_wire_update(
        "service-1",
        "state",
        JsonValue::Number(i64::try_from(sequence).expect("sequence fits i64").into()),
        ops,
    )
}

/// A raw state wire update for `subscription_id`, the envelope the
/// undecodable-update cases deliver.
#[must_use]
pub fn state_wire_update(
    subscription_id: &str,
    member: &str,
    sequence: JsonValue,
    ops: Vec<JsonValue>,
) -> pi_protocol::ServiceEventEnvelope {
    pi_protocol::ServiceEventEnvelope {
        subscription_id: subscription_id.to_string(),
        update: pi_chord::services::wire::object(vec![
            ("type", JsonValue::string("state")),
            ("member", JsonValue::string(member)),
            ("sequence", sequence),
            ("ops", JsonValue::Array(ops)),
        ]),
    }
}

/// Sends the success response `id` with `result`, the settle the request
/// cases correlate through.
pub fn send_success(server: &Rc<MemoryByteServer>, id: &str, result: Option<JsonValue>) {
    server.send(&ServerMessage::Response(ResponseEnvelope::Success(
        pi_protocol::ResponseSuccess {
            id: id.to_string(),
            result,
        },
    )));
}

/// Sends the failure response `id` with the bounded server error.
pub fn send_failure(server: &Rc<MemoryByteServer>, id: &str, error: pi_protocol::ProtocolError) {
    server.send(&ServerMessage::Response(ResponseEnvelope::Failure(
        pi_protocol::ResponseFailure {
            id: id.to_string(),
            error,
        },
    )));
}

/// Assembles the suite's client options literal over a factory and
/// identity, with the default frame bound and no listener-error hook.
#[must_use]
pub fn client_options(
    transport_factory: pi_client::ByteTransportFactory,
    server_id: &str,
) -> pi_client::ClientOptions {
    client_options_with(transport_factory, server_id, None, None)
}

/// The options literal with explicit frame bound and listener-error hook,
/// for the cases that bind those surfaces.
#[must_use]
pub fn client_options_with(
    transport_factory: pi_client::ByteTransportFactory,
    server_id: &str,
    max_frame_length: Option<usize>,
    on_listener_error: Option<pi_client::ListenerErrorHandler>,
) -> pi_client::ClientOptions {
    pi_client::ClientOptions {
        transport_factory,
        server_id: server_id.to_string(),
        max_frame_length,
        on_listener_error,
    }
}

/// A no-op service update listener, upstream's noop subscriber literal.
#[must_use]
pub fn noop_service_listener() -> Rc<dyn Fn(&ServiceProviderUpdate)> {
    Rc::new(|_update: &ServiceProviderUpdate| {})
}

/// Opens one service subscription in a local task, the spawn every
/// subscription case drives; the handle settles with the subscribe
/// outcome.
pub fn open_subscription(
    client: &Client,
    target: &pi_protocol::RpcTarget,
    service_id: &str,
    listener: Rc<dyn Fn(&ServiceProviderUpdate)>,
    signal: Option<&pi_chord::context::AbortSignal>,
) -> tokio::task::JoinHandle<Result<ServiceSubscription, ClientError>> {
    tokio::task::spawn_local(client.subscribe_service(
        target,
        service_id,
        ServiceMode::Singleton,
        listener,
        signal.cloned().as_ref(),
    ))
}

/// Opens the suite's default subscription against the server target with a
/// no-op listener, the fixture most subscription cases spawn.
pub fn open_default_subscription(
    client: &Client,
    signal: Option<&pi_chord::context::AbortSignal>,
) -> tokio::task::JoinHandle<Result<ServiceSubscription, ClientError>> {
    open_subscription(
        client,
        &server_target(),
        "pi.models",
        noop_service_listener(),
        signal,
    )
}

/// Records every connection-state change the client reports, the recorder
/// the lifecycle cases assert against.
#[must_use]
pub fn record_state_changes(client: &Client) -> Rc<RefCell<Vec<ConnectionStateChange>>> {
    let changes: Rc<RefCell<Vec<ConnectionStateChange>>> = Rc::new(RefCell::new(Vec::new()));
    let listener = changes.clone();
    let _unsubscribe = client.on_connection_state_change(Rc::new(move |change| {
        listener.borrow_mut().push(change.clone());
    }));
    changes
}

/// Records every attachment change's session id, the recorder the
/// retarget cases assert against.
#[must_use]
pub fn record_attachment_sessions(client: &Client) -> Rc<RefCell<Vec<Option<String>>>> {
    let sessions: Rc<RefCell<Vec<Option<String>>>> = Rc::new(RefCell::new(Vec::new()));
    let listener = sessions.clone();
    let _unsubscribe = client.on_attachment_change(Rc::new(move |target| {
        listener
            .borrow_mut()
            .push(target.map(|route| route.session_id.clone()));
    }));
    sessions
}

/// Drives one future on the current-thread runtime behind a `LocalSet`,
/// the substrate the client's local tasks require.
pub fn run_local<T>(future: impl Future<Output = T>) -> T {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tokio::task::LocalSet::new().block_on(&runtime, future)
}

/// An in-memory byte server, upstream's `MemoryByteServer`: it records the
/// client's decoded messages, answers the handshake, and exposes the same
/// send/disconnect/error/test hooks the upstream fixture does.
pub struct MemoryByteServer {
    server_id: String,
    messages: RefCell<Vec<ClientMessage>>,
    client_close_count: Cell<usize>,
    handlers: RefCell<Option<ByteTransportHandlers>>,
    decoder: RefCell<Option<ClientMessageDecoder>>,
    waiters: RefCell<Vec<(usize, oneshot::Sender<()>)>>,
}

impl Default for MemoryByteServer {
    fn default() -> Self {
        Self::new(SERVER_ID)
    }
}

impl MemoryByteServer {
    #[must_use]
    pub fn new(server_id: &str) -> Self {
        Self {
            server_id: server_id.to_string(),
            messages: RefCell::new(Vec::new()),
            client_close_count: Cell::new(0),
            handlers: RefCell::new(None),
            decoder: RefCell::new(None),
            waiters: RefCell::new(Vec::new()),
        }
    }

    /// The client messages decoded so far, in arrival order.
    pub fn messages(&self) -> Vec<ClientMessage> {
        self.messages.borrow().clone()
    }

    /// The count of client transports that closed against this server.
    #[must_use]
    pub const fn client_close_count(&self) -> usize {
        self.client_close_count.get()
    }

    /// Opens one transport against this server, upstream's `connect`.
    pub fn connect(self: &Rc<Self>, handlers: ByteTransportHandlers) -> Rc<dyn ByteTransport> {
        *self.handlers.borrow_mut() = Some(handlers.clone());
        *self.decoder.borrow_mut() = Some(
            ClientMessageDecoder::new(FrameDecoderOptions::default())
                .expect("default frame options are valid"),
        );
        Rc::new(MemoryTransport {
            server: Rc::clone(self),
            handlers,
            closed: Rc::new(Cell::new(false)),
        })
    }

    /// Resolves once at least `count` client messages have arrived,
    /// upstream's `waitForMessages`.
    pub fn wait_for_messages(self: &Rc<Self>, count: usize) -> LocalBoxFuture<()> {
        if self.messages.borrow().len() >= count {
            return boxed(std::future::ready(()));
        }
        let (sender, receiver) = oneshot::channel();
        self.waiters.borrow_mut().push((count, sender));
        boxed(async move {
            let _ = receiver.await;
        })
    }

    /// Sends one server message over the live transport, upstream's `send`.
    pub fn send(self: &Rc<Self>, message: &ServerMessage) {
        let frame = encode_server_message(message, FrameDecoderOptions::default())
            .expect("message encodes");
        self.send_raw(&frame);
    }

    /// Delivers one raw byte chunk, upstream's `sendRaw`.
    pub fn send_raw(self: &Rc<Self>, chunk: &[u8]) {
        let handlers = self.handlers.borrow().clone();
        (handlers.expect("No client connection").on_data)(chunk);
    }

    /// Reports an orderly transport close, upstream's `disconnect`.
    pub fn disconnect(&self) {
        let handlers = self.handlers.borrow_mut().take();
        if let Some(handlers) = handlers {
            (handlers.on_close)();
        }
    }

    /// Reports a terminal transport failure, upstream's `error`.
    pub fn error(self: &Rc<Self>, error: &ClientError) {
        let handlers = self.handlers.borrow_mut().take();
        if let Some(handlers) = handlers {
            (handlers.on_error)(error);
        }
    }

    fn record(self: &Rc<Self>, message: ClientMessage) {
        self.messages.borrow_mut().push(message);
        let length = self.messages.borrow().len();
        let waiters = std::mem::take(&mut *self.waiters.borrow_mut());
        for (count, sender) in waiters {
            if length >= count {
                let _ = sender.send(());
            } else {
                self.waiters.borrow_mut().push((count, sender));
            }
        }
    }

    fn send_handshake(self: &Rc<Self>) {
        self.send(&ServerMessage::Hello(ServerHello {
            server_id: ServerId::new(&self.server_id).expect("fixture server ids are canonical"),
        }));
    }
}

/// The in-memory transport one [`MemoryByteServer`] hands the client,
/// upstream's fixture's returned `ByteTransport` object literal.
struct MemoryTransport {
    server: Rc<MemoryByteServer>,
    handlers: ByteTransportHandlers,
    closed: Rc<Cell<bool>>,
}

impl ByteTransport for MemoryTransport {
    fn send(&self, chunk: Vec<u8>) -> LocalBoxFuture<Result<(), ClientError>> {
        let decoded = self
            .server
            .decoder
            .borrow_mut()
            .as_mut()
            .expect("the server keeps a decoder per transport")
            .push(&chunk)
            .map_err(ClientError::Protocol);
        match decoded {
            Ok(messages) => {
                for message in messages {
                    let handshake = matches!(message, ClientMessage::Hello(_));
                    self.server.record(message);
                    if handshake {
                        self.server.send_handshake();
                    }
                }
                boxed(std::future::ready(Ok(())))
            }
            Err(error) => boxed(std::future::ready(Err(error))),
        }
    }

    fn close(&self) {
        if self.closed.get() {
            return;
        }
        self.closed.set(true);
        self.server
            .client_close_count
            .set(self.server.client_close_count.get() + 1);
        let same = self
            .server
            .handlers
            .borrow()
            .as_ref()
            .is_some_and(|current| Rc::ptr_eq(&current.on_data, &self.handlers.on_data));
        if same {
            *self.server.handlers.borrow_mut() = None;
        }
    }
}

/// Connects one client to the in-memory server, upstream's `connectClient`.
pub fn connect_client(
    server: &Rc<MemoryByteServer>,
) -> LocalBoxFuture<Result<Client, ClientError>> {
    connect_client_to(server, SERVER_ID)
}

/// Connects one client to the in-memory server, fencing to the expected
/// identity.
pub fn connect_client_to(
    server: &Rc<MemoryByteServer>,
    expected_server_id: &str,
) -> LocalBoxFuture<Result<Client, ClientError>> {
    let options = client_options(in_memory_factory(server), expected_server_id);
    Client::connect_new(options)
}

/// A transport that counts sends and closes, upstream's inline transport
/// object literals in the lifecycle tests.
pub struct CountingTransport {
    /// The send count, upstream's `sendCount`.
    pub send_count: Rc<Cell<usize>>,
    /// The close count, upstream's `closeCount`.
    pub close_count: Rc<Cell<usize>>,
    /// Runs once, at first send: the hook the handshake-error transport
    /// drives server data through.
    pub on_send: Option<Rc<dyn Fn()>>,
}

impl ByteTransport for CountingTransport {
    fn send(&self, _chunk: Vec<u8>) -> LocalBoxFuture<Result<(), ClientError>> {
        self.send_count.set(self.send_count.get() + 1);
        if let Some(on_send) = &self.on_send {
            on_send();
        }
        boxed(std::future::ready(Ok(())))
    }

    fn close(&self) {
        self.close_count.set(self.close_count.get() + 1);
    }
}

/// Builds a [`CountingTransport`] factory.
#[must_use]
pub fn counting_transport(
    send_count: Rc<Cell<usize>>,
    close_count: Rc<Cell<usize>>,
    on_send: Option<Rc<dyn Fn()>>,
) -> pi_client::ByteTransportFactory {
    Rc::new(move |_handlers| {
        let transport: Rc<dyn ByteTransport> = Rc::new(CountingTransport {
            send_count: Rc::clone(&send_count),
            close_count: Rc::clone(&close_count),
            on_send: on_send.clone(),
        });
        boxed(std::future::ready(Ok(transport)))
    })
}

/// A client connected through the in-memory server's transport factory.
#[must_use]
pub fn in_memory_factory(server: &Rc<MemoryByteServer>) -> pi_client::ByteTransportFactory {
    let server = Rc::clone(server);
    Rc::new(move |handlers| {
        let transport = server.connect(handlers);
        boxed(std::future::ready(Ok(transport)))
    })
}

/// The switches the gate-transport probes read back: whether a gated send
/// future ran to completion, and how many closes the transport saw.
pub struct GateHandle {
    /// The oneshot the gated sends await before resolving.
    pub gate: Rc<RefCell<Option<oneshot::Receiver<()>>>>,
    /// Set by each gated send future before it resolves with the failure.
    pub resolved: Rc<Cell<bool>>,
    /// Every close the transport received.
    pub close_count: Rc<Cell<usize>>,
    /// The transport handlers, for tests that deliver server data or close
    /// events directly.
    pub handlers: Rc<RefCell<Option<ByteTransportHandlers>>>,
}

/// A transport that answers the client hello on its first send, fails every
/// send from `fail_after` on, and (when a gate is armed) makes the first
/// gated send wait for the gate before reporting the failure, upstream's
/// inline lifecycle-probe transports with a delayed send result.
struct HandshakeGateTransport {
    handlers: ByteTransportHandlers,
    hello_frame: Vec<u8>,
    sends: Cell<usize>,
    fail_after: usize,
    handle: Rc<GateHandle>,
}

impl ByteTransport for HandshakeGateTransport {
    fn send(&self, _chunk: Vec<u8>) -> LocalBoxFuture<Result<(), ClientError>> {
        let ordinal = self.sends.get();
        self.sends.set(ordinal + 1);
        if ordinal < self.fail_after {
            // The handshake send succeeds and answers with the server hello,
            // so the connection completes like the in-memory server's does.
            (self.handlers.on_data)(&self.hello_frame);
            return boxed(std::future::ready(Ok(())));
        }
        let receiver = self.handle.gate.borrow_mut().take();
        let handle = Rc::clone(&self.handle);
        match receiver {
            None => {
                handle.resolved.set(true);
                boxed(std::future::ready(Err(ClientError::other("send exploded"))))
            }
            Some(receiver) => boxed(async move {
                let _ = receiver.await;
                handle.resolved.set(true);
                Err(ClientError::other("send exploded"))
            }),
        }
    }

    fn close(&self) {
        self.handle
            .close_count
            .set(self.handle.close_count.get() + 1);
    }
}

/// Builds a factory whose transports answer the handshake and then fail
/// sends per [`HandshakeGateTransport`]'s rules; the third element fires
/// the gate when one was armed.
#[must_use]
pub fn handshake_gate_factory(
    fail_after: usize,
    gate: bool,
) -> (
    pi_client::ByteTransportFactory,
    Rc<GateHandle>,
    Option<oneshot::Sender<()>>,
) {
    let hello = encode_server_message(
        &ServerMessage::Hello(ServerHello {
            server_id: ServerId::new(SERVER_ID).expect("fixture server ids are canonical"),
        }),
        FrameDecoderOptions::default(),
    )
    .expect("the handshake frame encodes");
    let (fire, receiver) = if gate {
        let (sender, receiver) = oneshot::channel();
        (Some(sender), Some(receiver))
    } else {
        (None, None)
    };
    let handle = Rc::new(GateHandle {
        gate: Rc::new(RefCell::new(receiver)),
        resolved: Rc::new(Cell::new(false)),
        close_count: Rc::new(Cell::new(0)),
        handlers: Rc::new(RefCell::new(None)),
    });
    let factory = {
        let handle = Rc::clone(&handle);
        Rc::new(move |handlers: ByteTransportHandlers| {
            *handle.handlers.borrow_mut() = Some(handlers.clone());
            let transport: Rc<dyn ByteTransport> = Rc::new(HandshakeGateTransport {
                handlers,
                hello_frame: hello.clone(),
                sends: Cell::new(0),
                fail_after,
                handle: Rc::clone(&handle),
            });
            boxed(std::future::ready(Ok(transport)))
        })
    };
    (factory, handle, fire)
}

/// Builds a factory whose every connection attempt fails, upstream's
/// rejected `ByteTransportFactory` the connect-failure cases drive.
#[must_use]
pub fn failing_factory(message: &'static str) -> pi_client::ByteTransportFactory {
    Rc::new(move |_handlers| boxed(std::future::ready(Err(ClientError::other(message)))))
}

/// The handlers a factory-armed client holds, the readback the
/// direct-delivery cases drive.
pub type ArmedHandlers = Rc<RefCell<Option<ByteTransportHandlers>>>;

/// Delivers one raw chunk through the armed handlers.
pub fn deliver_raw(handlers: &ArmedHandlers, chunk: &[u8]) {
    (handlers
        .borrow()
        .as_ref()
        .expect("the transport arms the handlers")
        .on_data)(chunk);
}

/// Reports the orderly close through the armed handlers.
pub fn deliver_close(handlers: &ArmedHandlers) {
    (handlers
        .borrow()
        .as_ref()
        .expect("the transport arms the handlers")
        .on_close)();
}

/// Wraps `factory` so the handlers the client arms are recorded, the
/// capture the direct-delivery cases read back.
#[must_use]
pub fn recording_factory(
    factory: pi_client::ByteTransportFactory,
) -> (pi_client::ByteTransportFactory, ArmedHandlers) {
    let armed: ArmedHandlers = Rc::new(RefCell::new(None));
    let wrapped = {
        let armed = Rc::clone(&armed);
        Rc::new(move |handlers: ByteTransportHandlers| {
            *armed.borrow_mut() = Some(handlers.clone());
            factory(handlers)
        })
    };
    (wrapped, armed)
}

/// A counting transport whose first send delivers `chunk` to the client,
/// the on-send probe the early-data and garbage-frame handshake cases
/// drive, with the armed handlers for direct deliveries afterwards.
pub struct DeliveringTransport {
    /// The factory the client options carry.
    pub factory: pi_client::ByteTransportFactory,
    /// The send count, upstream's `sendCount`.
    pub send_count: Rc<Cell<usize>>,
    /// The close count, upstream's `closeCount`.
    pub close_count: Rc<Cell<usize>>,
    /// The handlers the client armed, for direct deliveries afterwards.
    pub armed: ArmedHandlers,
}

/// Builds the delivering transport; the first send emits `chunk`.
#[must_use]
pub fn delivering_transport(chunk: Vec<u8>) -> DeliveringTransport {
    let send_count = Rc::new(Cell::new(0));
    let close_count = Rc::new(Cell::new(0));
    let armed: ArmedHandlers = Rc::new(RefCell::new(None));
    let on_send = {
        let armed = Rc::clone(&armed);
        Rc::new(move || {
            if let Some(handlers) = armed.borrow().as_ref() {
                (handlers.on_data)(&chunk);
            }
        })
    };
    let counting = counting_transport(
        Rc::clone(&send_count),
        Rc::clone(&close_count),
        Some(on_send),
    );
    let factory = {
        let armed = Rc::clone(&armed);
        let counting = Rc::clone(&counting);
        Rc::new(move |handlers: ByteTransportHandlers| {
            *armed.borrow_mut() = Some(handlers.clone());
            counting(handlers)
        })
    };
    DeliveringTransport {
        factory,
        send_count,
        close_count,
        armed,
    }
}

/// Invokes one chord call through the client transport in a local task,
/// the spawn the transport mapping cases assert against.
pub fn spawn_transport_invoke(
    transport: &pi_client::ClientServiceTransport,
    call: pi_chord::types::ServiceCall,
) -> tokio::task::JoinHandle<Result<Option<JsonValue>, pi_chord::errors::ChordError>> {
    tokio::task::spawn_local(transport.invoke(call, pi_chord::context::background_context()))
}

/// Opens one chord subscription through the client transport in a local
/// task, the spawn the transport mapping cases assert against.
pub fn spawn_transport_subscribe(
    transport: &pi_client::ClientServiceTransport,
) -> tokio::task::JoinHandle<
    Result<pi_chord::types::ServiceSubscription, pi_chord::errors::ChordError>,
> {
    tokio::task::spawn_local(transport.subscribe(
        "pi.models".to_string(),
        ServiceMode::Singleton,
        Rc::new(|_update, _context| {}),
        pi_chord::context::background_context(),
    ))
}

/// A no-op handler set, the transport callbacks the transport-only cases
/// connect with.
#[must_use]
pub fn noop_handlers() -> ByteTransportHandlers {
    ByteTransportHandlers {
        on_data: Rc::new(|_chunk: &[u8]| {}),
        on_close: Rc::new(|| {}),
        on_error: Rc::new(|_error: &ClientError| {}),
    }
}
