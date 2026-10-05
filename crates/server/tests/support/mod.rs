//! The shared fixtures of the pi-server suite: the in-memory loopback the
//! conformance and protocol cases drive, the server registry the teardown
//! closes, and the small value helpers the upstream cases inline.

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test fixtures settle invariants the driving cases' assertions cover; the restriction lints target production code"
)]
#![allow(
    dead_code,
    unreachable_pub,
    reason = "the fixture module is shared across test binaries, never exported"
)]

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::types::{JsonNumber, JsonObject, JsonValue, ServiceCall};
use pi_protocol::ResponseEnvelope;
use pi_server::RoutedServerServiceAttachment;
use pi_server::testing::{Deferred, OpenGate};
use pi_server::testing::{ProtocolTestClient, TestServerHost, WireChannel};
use pi_server::{
    ByteConnectionHandler, RoutedServerServiceHost, RoutedSessionAttachment, RoutedSessionHandle,
    ServerHost, ServerOptions, ServicePublisher,
};
use pi_server::{Failure, Server};

/// The canonical conformance server identity, upstream's `serverId` const.
pub const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

/// A fresh temporary socket path, upstream's
/// `mkdtemp(...) + join(dir, "server.sock")`.
pub fn temp_socket_path(prefix: &str) -> String {
    let directory = tempfile::tempdir().expect("the tempdir builds");
    let path = directory.path().join("server.sock");
    let path = path.to_string_lossy().into_owned();
    // The directory outlives the test through the leaked handle the caller
    // never sees; the socket path stays unique.
    std::mem::forget(directory);
    let _ = prefix;
    path
}

/// Boxes a predicate into the wire client's waiter shape, the coercion the
/// closures need at every `next` call.
pub fn predicate<F: Fn(&pi_protocol::ServerMessage) -> bool + 'static>(
    function: F,
) -> Rc<dyn Fn(&pi_protocol::ServerMessage) -> bool> {
    Rc::new(function)
}

/// Drives one future on a current-thread runtime behind a `LocalSet`, the
/// single-threaded contract the crate documents.
pub fn run_local<T>(future: impl Future<Output = T>) -> T {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tokio::task::LocalSet::new().block_on(&runtime, future)
}

/// The protocol version as the wire number, the `hello(version)` default and
/// its off-by-one rejection probe.
pub const fn version(value: f64) -> JsonNumber {
    JsonNumber::new(value).unwrap()
}

/// Retries a condition until it settles, upstream's `expect.poll`.
pub async fn poll_until(condition: impl Fn() -> bool) {
    for _ in 0..10_000 {
        if condition() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("the polled condition never settled");
}

/// The server registry the teardown closes, upstream's per-file `servers`
/// set.
#[derive(Clone, Default)]
pub struct Servers(Rc<RefCell<Vec<TrackedServer>>>);

struct TrackedServer {
    token: usize,
    close: LocalBoxFuture<()>,
}

impl Servers {
    /// Registers one server for the teardown, swallowing its close failure
    /// the way the upstream teardown never asserts on it.
    pub fn track<H: ServerHost + 'static>(&self, server: &Server<H>) {
        let token = server.identity_token();
        let closing = server.clone();
        self.0.borrow_mut().push(TrackedServer {
            token,
            close: boxed(async move {
                let _ = closing.close().await;
            }),
        });
    }

    /// Drops one registration, upstream's `servers.delete(server)`.
    pub fn forget<H: ServerHost + 'static>(&self, server: &Server<H>) {
        let token = server.identity_token();
        self.0.borrow_mut().retain(|tracked| tracked.token != token);
    }

    /// Closes every registered server, upstream's `afterEach`.
    pub async fn close_all(&self) {
        let tracked: Vec<TrackedServer> = self.0.borrow_mut().drain(..).collect();
        for tracked in tracked {
            tracked.close.await;
        }
    }
}

/// Builds one unstarted server over `host` with the conformance identity and
/// no listeners, upstream's `createServer(host, id?)`.
pub fn create_server(host: &Rc<TestServerHost>) -> Server<TestServerHost> {
    Server::new(
        Rc::clone(host),
        ServerOptions {
            listeners: Vec::new(),
            server_id: SERVER_ID.to_string(),
            max_frame_length: None,
            handshake_timeout_ms: None,
            on_connection_count_changed: None,
            on_error: None,
        },
    )
    .unwrap()
}

/// Builds one unstarted server over a generic host, the shape the inline
/// conformance hosts drive.
pub fn create_server_over<H: ServerHost + 'static>(
    host: Rc<H>,
    on_error: Option<pi_server::ErrorObserver>,
) -> Server<H> {
    Server::new(
        host,
        ServerOptions {
            listeners: Vec::new(),
            server_id: SERVER_ID.to_string(),
            max_frame_length: None,
            handshake_timeout_ms: None,
            on_connection_count_changed: None,
            on_error,
        },
    )
    .unwrap()
}

/// The loopback pair one conformance connection drives, upstream's
/// `connect(server)`: the server's frames feed the wire client, and the
/// client's channel feeds the server's handler.
pub fn connect(server: &Server<TestServerHost>) -> Rc<ProtocolTestClient> {
    connect_over(server)
}

/// The same loopback over a generic host.
pub fn connect_over<H: ServerHost + 'static>(server: &Server<H>) -> Rc<ProtocolTestClient> {
    let handler = Rc::new(RefCell::new(Option::<ByteConnectionHandler>::None));
    let closed = Rc::new(Cell::new(false));
    let client = Rc::new(RefCell::new(Option::<Rc<ProtocolTestClient>>::None));

    // The server-side view: frames the server sends feed the client decoder.
    let loopback_connection: Rc<dyn pi_server::ByteConnection> = {
        let client = Rc::clone(&client);
        let closed = Rc::clone(&closed);
        Rc::new(LoopbackConnection { client, closed })
    };
    let handler_value = server.accept(loopback_connection);
    *handler.borrow_mut() = Some(handler_value);

    // The client-side view: the client's channel feeds the server handler.
    let channel: Rc<dyn WireChannel> = {
        let handler = Rc::clone(&handler);
        let closed = Rc::clone(&closed);
        let client = Rc::clone(&client);
        Rc::new(LoopbackChannel {
            handler,
            closed,
            client,
        })
    };
    let wire_client = Rc::new(ProtocolTestClient::new(channel));
    *client.borrow_mut() = Some(Rc::clone(&wire_client));
    wire_client
}

/// The server-side loopback transport, upstream's `connection` object
/// literal.
struct LoopbackConnection {
    client: Rc<RefCell<Option<Rc<ProtocolTestClient>>>>,
    closed: Rc<Cell<bool>>,
}

impl pi_server::ByteConnection for LoopbackConnection {
    fn closed(&self) -> bool {
        self.closed.get()
    }

    fn send(&self, chunk: Vec<u8>) -> LocalBoxFuture<Result<(), Failure>> {
        if let Some(client) = self.client.borrow().as_ref() {
            client.receive(&chunk);
        }
        boxed(async { Ok(()) })
    }

    fn close(&self, final_chunk: Option<Vec<u8>>) -> LocalBoxFuture<Result<(), Failure>> {
        if let Some(final_chunk) = final_chunk
            && let Some(client) = self.client.borrow().as_ref()
        {
            client.receive(&final_chunk);
        }
        self.closed.set(true);
        if let Some(client) = self.client.borrow().as_ref() {
            client.mark_closed();
        }
        boxed(async { Ok(()) })
    }
}

/// The client-side loopback channel, upstream's `channel` object literal.
struct LoopbackChannel {
    handler: Rc<RefCell<Option<ByteConnectionHandler>>>,
    closed: Rc<Cell<bool>>,
    client: Rc<RefCell<Option<Rc<ProtocolTestClient>>>>,
}

impl WireChannel for LoopbackChannel {
    fn send(&self, chunk: Vec<u8>) -> LocalBoxFuture<()> {
        if let Some(handler) = self.handler.borrow().as_ref() {
            (handler.on_data)(&chunk);
        }
        boxed(async {})
    }

    fn send_fragmented(&self, chunk: Vec<u8>, split_at: usize) -> LocalBoxFuture<()> {
        let (head, tail) = chunk.split_at(split_at);
        if let Some(handler) = self.handler.borrow().as_ref() {
            (handler.on_data)(head);
            (handler.on_data)(tail);
        }
        boxed(async {})
    }

    fn close(&self) -> LocalBoxFuture<()> {
        if self.closed.get() {
            return boxed(async {});
        }
        self.closed.set(true);
        if let Some(handler) = self.handler.borrow().as_ref() {
            (handler.on_close)();
        }
        if let Some(client) = self.client.borrow().as_ref() {
            client.mark_closed();
        }
        boxed(async {})
    }
}

/// Builds `{ serviceId: "test.session", member, args }`, upstream's
/// `sessionCall`.
pub fn session_call(member: &str, args: Vec<JsonValue>) -> ServiceCall {
    ServiceCall {
        service_id: "test.session".to_string(),
        instance: None,
        member: member.to_string(),
        args,
    }
}

/// The server-addressed target the conformance calls fence to.
pub fn server_target() -> pi_protocol::RpcTarget {
    pi_protocol::RpcTarget::Server(pi_protocol::ServerTarget {
        server_id: pi_protocol::ServerId::new(SERVER_ID).unwrap(),
    })
}

/// The server-addressed target for one other server.
pub fn other_server_target() -> pi_protocol::RpcTarget {
    pi_protocol::RpcTarget::Server(pi_protocol::ServerTarget {
        server_id: pi_protocol::ServerId::new("00000000-0000-4000-8000-000000000002").unwrap(),
    })
}

/// Whether two failures are the same routed error, upstream's
/// `expect(errors).toContain(releaseError)` identity check: opaque failures
/// match by their `Rc` identity, the rest by value.
pub fn same_failure(left: &Failure, right: &Failure) -> bool {
    use Failure as F;
    match (left, right) {
        (F::Other(left), F::Other(right)) => Rc::ptr_eq(left, right),
        (
            F::Aggregate { message, errors },
            F::Aggregate {
                message: m2,
                errors: e2,
            },
        )
        | (
            F::Cleanup { message, errors },
            F::Cleanup {
                message: m2,
                errors: e2,
            },
        ) => message == m2 && errors.len() == e2.len(),
        (F::Server(left), F::Server(right)) => left == right,
        (F::Remote(left), F::Remote(right)) => left == right,
        (F::Validation(left), F::Validation(right)) => left == right,
        _ => false,
    }
}

/// A recorded `{"arbitrary": true}` call payload, upstream's object literal.
pub fn arbitrary_call() -> JsonValue {
    JsonValue::Object(JsonObject::from_entries(vec![(
        "arbitrary".to_string(),
        JsonValue::Bool(true),
    )]))
}

/// The counting listener the composition cases build, upstream's
/// `TestListener`.
pub struct TestListener {
    pub start_count: Cell<u32>,
    pub close_count: Cell<u32>,
    start_error: Option<Failure>,
    close_error: Option<Failure>,
}

impl TestListener {
    /// A listener that starts and closes cleanly.
    pub fn new(start_error: Option<Failure>) -> Rc<Self> {
        Rc::new(Self {
            start_count: Cell::new(0),
            close_count: Cell::new(0),
            start_error,
            close_error: None,
        })
    }

    /// A listener whose start and close carry their own failures.
    pub fn new_with_close_error(
        start_error: Option<Failure>,
        close_error: Option<Failure>,
    ) -> Rc<Self> {
        Rc::new(Self {
            start_count: Cell::new(0),
            close_count: Cell::new(0),
            start_error,
            close_error,
        })
    }
}

impl pi_server::ServerListener for TestListener {
    fn start(
        &self,
        _accept: pi_server::ByteConnectionAcceptor,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        self.start_count.set(self.start_count.get() + 1);
        self.start_error.as_ref().map_or_else(
            || boxed(async { Ok(()) }),
            |error| {
                let error = error.clone();
                boxed(async move { Err(error) })
            },
        )
    }

    fn close(&self) -> LocalBoxFuture<Result<(), Failure>> {
        self.close_count.set(self.close_count.get() + 1);
        self.close_error.as_ref().map_or_else(
            || boxed(async { Ok(()) }),
            |error| {
                let error = error.clone();
                boxed(async move { Err(error) })
            },
        )
    }
}

/// The one-line session-metadata fixture the inline hosts resolve.
pub fn metadata(id: &str) -> Rc<pi_agent_core::harness::session::types::SessionMetadata> {
    Rc::new(pi_agent_core::harness::session::types::SessionMetadata {
        id: id.to_string(),
        created_at: 1,
        storage_version: 1,
        cwd: None,
        parent_session_id: None,
        legacy_parent_session_path: None,
    })
}

/// The error recorder the observer-carrying cases drive, upstream's
/// `errors: Error[]` fixture.
pub fn error_recorder() -> (Rc<RefCell<Vec<Failure>>>, pi_server::ErrorObserver) {
    let errors = Rc::new(RefCell::new(Vec::<Failure>::new()));
    let observer: pi_server::ErrorObserver = {
        let errors = Rc::clone(&errors);
        Rc::new(move |error: &Failure| errors.borrow_mut().push(error.clone()))
    };
    (errors, observer)
}

/// The resolved-metadata future the inline test hosts share, upstream's
/// `resolveSession: async () => metadata`.
pub fn resolve_ok<T: Clone + 'static>(metadata: Rc<T>) -> LocalBoxFuture<Result<Rc<T>, Failure>> {
    boxed(async move { Ok(metadata) })
}

/// The unreachable-open future the hosts that never open a session carry,
/// upstream's `openSession: async () => { throw ... }` arms.
pub fn open_unreachable() -> LocalBoxFuture<Result<Rc<dyn RoutedSessionHandle>, Failure>> {
    boxed(async { Err(Failure::message("the host never opens a session")) })
}

/// The empty-lease future, upstream's `invokeService: async () => undefined`.
pub fn lease_no_result() -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
    boxed(async { Ok(None) })
}

/// The `null`-result lease future, upstream's `return null` attach arms.
pub fn lease_null_result() -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
    boxed(async { Ok(Some(JsonValue::Null)) })
}

/// The released-Ok future, upstream's `release() {}` arms.
pub fn release_ok() -> LocalBoxFuture<Result<(), Failure>> {
    boxed(async { Ok(()) })
}

/// The closed-Ok future, upstream's `close: async () => {}` arms.
pub fn close_ok() -> LocalBoxFuture<Result<(), Failure>> {
    boxed(async { Ok(()) })
}

/// The none-terminated surface, upstream's absent `terminated` field.
pub fn no_terminated() -> Option<LocalBoxFuture<Option<Failure>>> {
    None
}

/// The inline handle the custom hosts hand out, the shared shape behind the
/// per-case hooks.
pub struct InlineHandle {
    pub terminated: Option<Deferred<Option<Failure>>>,
    pub continue_acquiring: Option<Deferred<()>>,
    pub acquiring_entered: Option<Deferred<()>>,
    pub release_count: Rc<Cell<u32>>,
    pub on_attach: Option<Rc<dyn Fn()>>,
}

impl RoutedSessionHandle for InlineHandle {
    fn attach_client(
        &self,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionAttachment>, Failure>> {
        if let Some(entered) = &self.acquiring_entered {
            entered.resolve(());
        }
        let continue_acquiring = self.continue_acquiring.clone();
        let on_attach = self.on_attach.clone();
        let release_count = Rc::clone(&self.release_count);
        boxed(async move {
            if let Some(continue_acquiring) = continue_acquiring {
                continue_acquiring.promise().await;
            }
            if let Some(on_attach) = on_attach {
                on_attach();
            }
            let lease: Rc<dyn RoutedSessionAttachment> = Rc::new(InlineLease { release_count });
            Ok(lease)
        })
    }

    fn terminated(&self) -> Option<LocalBoxFuture<Option<Failure>>> {
        self.terminated.as_ref().map(Deferred::promise)
    }

    fn close(
        &self,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        close_ok()
    }
}

/// The inline lease the handles hand out.
pub struct InlineLease {
    pub release_count: Rc<Cell<u32>>,
}

impl RoutedSessionAttachment for InlineLease {
    fn invoke_service(
        &self,
        _call: ServiceCall,
        _publish: ServicePublisher,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        lease_no_result()
    }

    fn release(
        &self,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        self.release_count.set(self.release_count.get() + 1);
        release_ok()
    }
}

/// The handle factory one `InlineHost` opens with, upstream's `openSession`
/// body.
pub type HandleFactory = Rc<
    dyn Fn(
        Rc<pi_agent_core::harness::session::types::SessionMetadata>,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionHandle>, Failure>>,
>;

/// The generic inline host the boundary cases build, upstream's object
/// literals: it resolves `metadata` and hands out the handle the factory
/// returns.
pub struct InlineHost {
    pub services: Rc<dyn RoutedServerServiceHost>,
    pub metadata: Rc<pi_agent_core::harness::session::types::SessionMetadata>,
    /// The handle factory, upstream's `openSession` body.
    pub open: HandleFactory,
}

impl pi_server::HasSessionId for InlineHost {
    fn session_id(&self) -> &str {
        &self.metadata.id
    }
}

impl ServerHost for InlineHost {
    type Metadata = pi_agent_core::harness::session::types::SessionMetadata;

    fn server_services(&self) -> Rc<dyn RoutedServerServiceHost> {
        Rc::clone(&self.services)
    }

    fn resolve_session(
        &self,
        _session_id: &str,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<Rc<Self::Metadata>, Failure>> {
        resolve_ok(Rc::clone(&self.metadata))
    }

    fn open_session(
        &self,
        metadata: Rc<Self::Metadata>,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionHandle>, Failure>> {
        (self.open)(metadata)
    }
}

/// The generic inline lease factory's handle over one shared hook object.
pub struct HookHandle<H: SharedLeaseHooks + 'static> {
    pub hooks: Rc<H>,
}

impl<H: SharedLeaseHooks> RoutedSessionHandle for HookHandle<H> {
    fn attach_client(
        &self,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionAttachment>, Failure>> {
        let hooks = Rc::clone(&self.hooks);
        boxed(async move {
            Ok({
                let lease: Rc<dyn RoutedSessionAttachment> = hooks.lease();
                lease
            })
        })
    }

    fn terminated(&self) -> Option<LocalBoxFuture<Option<Failure>>> {
        self.hooks.terminated()
    }

    fn close(
        &self,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        close_ok()
    }
}

/// The hook surface the shared handle's leases derive from.
/// The [`SharedLeaseHooks`] over a fresh [`InlineLease`], the no-op arm.
pub struct EmptyHooks;

impl SharedLeaseHooks for EmptyHooks {
    fn lease(&self) -> Rc<dyn RoutedSessionAttachment> {
        Rc::new(InlineLease {
            release_count: Rc::new(Cell::new(0)),
        })
    }
}

pub trait SharedLeaseHooks {
    /// The lease one attach hands out, upstream's `attachClient` return.
    fn lease(&self) -> Rc<dyn RoutedSessionAttachment>;
    /// The termination surface, upstream's `terminated?`.
    fn terminated(&self) -> Option<LocalBoxFuture<Option<Failure>>> {
        None
    }
}

/// The handle factory over one shared hook object, the `openSession` body
/// the hook-driven cases share.
pub fn open_hooks<H: SharedLeaseHooks + 'static>(hooks: Rc<H>) -> HandleFactory {
    Rc::new(move |_| {
        let handle: Rc<dyn RoutedSessionHandle> = Rc::new(HookHandle {
            hooks: Rc::clone(&hooks),
        });
        boxed(async move { Ok(handle) })
    })
}

/// The handle factory over one fixed handle, the fixed-lease cases share.
pub fn open_fixed(handle: Rc<dyn RoutedSessionHandle>) -> HandleFactory {
    Rc::new(move |_| {
        let handle = Rc::clone(&handle);
        boxed(async move { Ok(handle) })
    })
}

/// The handle factory that always fails, upstream's `openSession` throw
/// arms.
pub fn open_failing(failure: Failure) -> HandleFactory {
    Rc::new(move |_| {
        let failure = failure.clone();
        boxed(async move { Err(failure) })
    })
}

/// The handle factory that never opens, upstream's `throw`ing
/// `openSession` arms.
pub fn open_unreachable_factory() -> HandleFactory {
    Rc::new(|_| open_unreachable())
}

/// The released-Ok lease impl body, the `release` arm the test leases share.
#[macro_export]
macro_rules! lease_release_ok {
    () => {
        fn release(
            &self,
            _context: pi_agent_core::harness::context::Context,
        ) -> LocalBoxFuture<Result<(), Failure>> {
            boxed(async { Ok(()) })
        }
    };
}

/// The greeted-case prelude: host, seed, server, tracked, connected client,
/// upstream's per-case handshake scaffolding.
pub async fn greeted_case() -> (
    Servers,
    Rc<TestServerHost>,
    Server<TestServerHost>,
    Rc<ProtocolTestClient>,
) {
    let servers = Servers::default();
    let host = Rc::new(TestServerHost::new());
    host.seed("session-1", None).await.unwrap();
    let server = create_server(&host);
    servers.track(&server);
    let client = connect(&server);
    client.hello(version(8.0)).await.unwrap();
    (servers, host, server, client)
}

/// The attached-client prelude the session-scoped cases share: seed, serve,
/// connect, greet, attach, and hand back the registry, server, client, and
/// harness, upstream's per-case attach scaffolding.
pub async fn attached_case() -> (
    Servers,
    Rc<TestServerHost>,
    Server<TestServerHost>,
    Rc<ProtocolTestClient>,
    Rc<TestHarnessFixture>,
) {
    let servers = Servers::default();
    let host = Rc::new(TestServerHost::new());
    host.seed("session-1", None).await.unwrap();
    let server = create_server(&host);
    servers.track(&server);
    let client = connect(&server);
    client.hello(version(8.0)).await.unwrap();
    client.attach(SERVER_ID, "session-1").await.unwrap();
    let harness = host.latest_harness("session-1");
    (servers, host, server, client, harness)
}

/// The harness alias the prelude returns.
pub type TestHarnessFixture = pi_server::testing::TestHarness;

/// The attachment id the latest recorded route named, upstream's
/// `latestAttachmentId` helper the conformance suite drives.
pub fn latest_attachment_id(client: &ProtocolTestClient, session_id: &str) -> String {
    client
        .messages()
        .iter()
        .rev()
        .find_map(|message| match message {
            pi_protocol::ServerMessage::Attachment(pi_protocol::AttachmentEnvelope {
                attachment: Some(target),
            }) if target.session_id == session_id => Some(target.attachment_id.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("Missing attachment for {session_id}"))
}

/// The hello-error probe the protocol cases share: connect, send `bytes`,
/// wait for the `hello_error`, assert the code, and await the close,
/// upstream's hostile-framed-input and first-message rejection shape.
pub async fn expect_hello_error(
    server: &Server<TestServerHost>,
    send: impl FnOnce(&Rc<ProtocolTestClient>) -> LocalBoxFuture<()>,
    expected_code: &str,
) {
    let client = connect(server);
    let delivered = send(&client);
    delivered.await;
    let error = client
        .next(predicate(|message: &pi_protocol::ServerMessage| {
            matches!(message, pi_protocol::ServerMessage::HelloError(_))
        }))
        .await
        .unwrap();
    let pi_protocol::ServerMessage::HelloError(envelope) = error else {
        panic!("expected hello_error");
    };
    assert_eq!(envelope.error.code, expected_code);
    client.wait_for_close().await;
}

/// The subscription-case prelude: the host over the control lease, the
/// published handle, the recorded calls, and the connected client, upstream's
/// per-subscription scaffolding.
#[allow(
    clippy::type_complexity,
    reason = "the tuple reads as one fixture bundle the cases destructure"
)]
pub async fn subscription_case() -> (
    Servers,
    Rc<ProtocolTestClient>,
    Rc<RefCell<Option<ServicePublisher>>>,
    Rc<RefCell<Vec<ServiceCall>>>,
) {
    let servers = Servers::default();
    let published = Rc::new(RefCell::new(None::<ServicePublisher>));
    let calls = Rc::new(RefCell::new(Vec::new()));
    let host = Rc::new(InlineHost {
        services: Rc::new(ControlLease {
            published: Rc::clone(&published),
            calls: Rc::clone(&calls),
        }),
        metadata: metadata("session-1"),
        open: open_hooks(Rc::new(EmptyHooks)),
    });
    let server = create_server_over(host, None);
    servers.track(&server);
    let client = connect_over(&server);
    client.hello(version(8.0)).await.unwrap();
    (servers, client, published, calls)
}

/// The lease that answers `$chord.service` control calls.
pub struct ControlLease {
    pub published: Rc<RefCell<Option<ServicePublisher>>>,
    pub calls: Rc<RefCell<Vec<ServiceCall>>>,
}

impl RoutedServerServiceHost for ControlLease {
    fn attach_client(
        &self,
        _presentation: Rc<dyn pi_server::RoutedServerPresentation>,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedServerServiceAttachment>, Failure>> {
        let lease: Rc<dyn RoutedServerServiceAttachment> = Rc::new(ControlServiceLease {
            published: Rc::clone(&self.published),
            calls: Rc::clone(&self.calls),
        });
        boxed(async move { Ok(lease) })
    }
}

struct ControlServiceLease {
    published: Rc<RefCell<Option<ServicePublisher>>>,
    calls: Rc<RefCell<Vec<ServiceCall>>>,
}

impl RoutedServerServiceAttachment for ControlServiceLease {
    fn invoke_service(
        &self,
        call: ServiceCall,
        publish: ServicePublisher,
        context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        let published = Rc::clone(&self.published);
        let calls = Rc::clone(&self.calls);
        boxed(async move {
            calls.borrow_mut().push(call.clone());
            if call.service_id == "$chord.service" && call.member == "subscribe" {
                let subscription_id = match call.args.first() {
                    Some(JsonValue::Str(subscription_id)) => subscription_id.clone(),
                    _ => return Err(Failure::message("no subscription id")),
                };
                // An update published before the snapshot buffers, upstream's
                // pendingUpdates.
                let buffered = pi_chord::types::ServiceProviderUpdate::Unavailable;
                publish(&subscription_id, &buffered, &context).await?;
                *published.borrow_mut() = Some(publish);
                let snapshot = JsonValue::Object(JsonObject::from_entries(vec![
                    (
                        "serviceId".to_string(),
                        JsonValue::Str("pi.models".to_string()),
                    ),
                    ("mode".to_string(), JsonValue::Str("singleton".to_string())),
                    ("instances".to_string(), JsonValue::Array(vec![])),
                ]));
                return Ok(Some(snapshot));
            }
            if call.service_id == "$chord.service" && call.member == "unsubscribe" {
                return Ok(Some(JsonValue::Null));
            }
            Err(Failure::message(format!(
                "Unsupported control call {}.{}",
                call.service_id, call.member
            )))
        })
    }

    fn release(
        &self,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        release_ok()
    }
}

/// The default Unix-server options over one path, upstream's
/// `createUnixServer(host, { serverId, path })`.
#[cfg(unix)]
pub fn unix_server_options(path: &str) -> pi_server::unix::UnixServerOptions {
    pi_server::unix::UnixServerOptions {
        server_id: SERVER_ID.to_string(),
        path: path.to_string(),
        mode: None,
        max_pending_bytes: None,
        graceful_close_timeout_ms: None,
        max_frame_length: None,
        handshake_timeout_ms: None,
        on_connection_count_changed: None,
        on_error: None,
    }
}

/// The default Unix-listener options over one path.
#[cfg(unix)]
pub fn unix_listener_options(path: &str) -> pi_server::unix::UnixListenerOptions {
    pi_server::unix::UnixListenerOptions {
        path: path.to_string(),
        mode: None,
        max_pending_bytes: None,
        graceful_close_timeout_ms: None,
        max_frame_length: None,
        on_error: None,
    }
}

/// Awaits a gated session-scoped call's completion and asserts the success
/// shape, upstream's `gate.release.resolve(); let response = ...` tail.
pub async fn settle_gated_call(
    gate: &OpenGate,
    calling: tokio::task::JoinHandle<Result<ResponseEnvelope, String>>,
) -> ResponseEnvelope {
    gate.release.resolve(());
    let response = calling.await.unwrap().unwrap();
    assert!(matches!(response, ResponseEnvelope::Success(_)));
    response
}

/// The seeded connected client case, before the handshake: the client is
/// connected but the hello has not gone out, upstream's connect-only
/// prologue the coalesced-frame cases drive.
pub async fn connected_case() -> (Servers, Server<TestServerHost>, Rc<ProtocolTestClient>) {
    let servers = Servers::default();
    let host = Rc::new(TestServerHost::new());
    host.seed("session-1", None).await.unwrap();
    let server = create_server(&host);
    servers.track(&server);
    let client = connect(&server);
    (servers, server, client)
}

/// Starts one gated session-scoped call and waits for it to enter the gate,
/// upstream's `gateNextServiceCall` + spawned `requestSessionService`
/// scaffolding the cancel cases share.
pub async fn start_gated_session_call(
    client: &Rc<ProtocolTestClient>,
    harness: &Rc<TestHarnessFixture>,
) -> (
    OpenGate,
    tokio::task::JoinHandle<Result<ResponseEnvelope, String>>,
) {
    let gate = harness.gate_next_service_call();
    let calling_client = Rc::clone(client);
    let calling = tokio::task::spawn_local(async move {
        calling_client
            .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
            .await
    });
    gate.entered.promise().await;
    (gate, calling)
}

/// One started Unix-transport server over a fresh host, upstream's
/// `createServer(listen({ path }))` double the transport cases drive.
pub fn create_unix_test_server(path: &str) -> Server<TestServerHost> {
    pi_server::unix::create_unix_server(Rc::new(TestServerHost::new()), unix_server_options(path))
        .expect("the unix server options are valid")
}

/// The inline-host case scaffold: the server over `services`/`open`, the
/// tracked server, and the hello'd client, upstream's inline host literal
/// plus handshake.
pub async fn inline_case(
    services: Rc<dyn RoutedServerServiceHost>,
    open: HandleFactory,
    on_error: Option<pi_server::ErrorObserver>,
) -> (Servers, Server<InlineHost>, Rc<ProtocolTestClient>) {
    let servers = Servers::default();
    let host = Rc::new(InlineHost {
        services,
        metadata: metadata("session-1"),
        open,
    });
    let server = create_server_over(host, on_error);
    servers.track(&server);
    let client = connect_over(&server);
    client.hello(version(8.0)).await.unwrap();
    (servers, server, client)
}

/// The server-scoped `pi.session-management` call body, upstream's
/// `{ serviceId: "pi.session-management", member, args }` literal.
pub fn server_management_call(member: &str, args: Vec<JsonValue>) -> ServiceCall {
    ServiceCall {
        service_id: "pi.session-management".to_string(),
        instance: None,
        member: member.to_string(),
        args,
    }
}

/// The service lease that answers nothing, the stand-in the presentation
/// fixtures share.
pub struct InertLease;

impl RoutedServerServiceAttachment for InertLease {
    fn invoke_service(
        &self,
        _call: ServiceCall,
        _publish: ServicePublisher,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        boxed(async move { Err(Failure::message("no test server services")) })
    }

    fn release(
        &self,
        _context: pi_agent_core::harness::context::Context,
    ) -> LocalBoxFuture<Result<(), Failure>> {
        boxed(async { Ok(()) })
    }
}

/// The inert lease behind the presentation fixtures' `attachClient` answer.
pub fn inert_service_lease() -> Rc<dyn RoutedServerServiceAttachment> {
    Rc::new(InertLease)
}

/// The singleton models-subscription request, upstream's
/// `subscribe("sub-1", "pi.models", "singleton")` call the wire cases share.
pub async fn subscribe_models_request(
    client: &Rc<ProtocolTestClient>,
) -> Result<ResponseEnvelope, String> {
    client
        .request_service(
            server_target(),
            pi_chord::services::wire::create_service_subscribe_call(
                "sub-1",
                "pi.models",
                pi_chord::types::ServiceMode::Singleton,
            ),
            None,
        )
        .await
}

/// The all-None test-server options, upstream's default `TestServerOptions`.
pub fn plain_test_options() -> pi_server::testing::TestServerOptions {
    pi_server::testing::TestServerOptions {
        listeners: None,
        host: None,
        server_id: None,
        max_frame_length: None,
        handshake_timeout_ms: None,
        on_error: None,
    }
}

/// The test-server options over explicit listeners, the rest default.
pub fn listener_test_options(
    listeners: Vec<Rc<dyn pi_server::ServerListener>>,
) -> pi_server::testing::TestServerOptions {
    pi_server::testing::TestServerOptions {
        listeners: Some(listeners),
        ..plain_test_options()
    }
}

/// The in-memory server pinned to a small frame ceiling or handshake budget,
/// upstream's `createServer({...overrides})` doubles the wire cases drive.
pub fn framed_server(
    host: Rc<TestServerHost>,
    max_frame_length: Option<usize>,
    handshake_timeout_ms: Option<u64>,
    on_error: Option<pi_server::ErrorObserver>,
) -> Server<TestServerHost> {
    Server::new(
        host,
        ServerOptions {
            listeners: Vec::new(),
            server_id: SERVER_ID.to_string(),
            max_frame_length,
            handshake_timeout_ms,
            on_connection_count_changed: None,
            on_error,
        },
    )
    .expect("the framed server options are valid")
}

/// Awaits the queued second hello's rejection and asserts its shape,
/// upstream's queued-message arm tail.
pub async fn expect_first_message_rejection(client: &Rc<ProtocolTestClient>) {
    let error = client
        .next(predicate(|message: &pi_protocol::ServerMessage| {
            matches!(message, pi_protocol::ServerMessage::HelloError(_))
        }))
        .await
        .expect("the queued hello answers hello_error");
    let pi_protocol::ServerMessage::HelloError(envelope) = error else {
        panic!("expected hello_error");
    };
    assert_eq!(envelope.error.code, "invalid_request");
    assert!(envelope.error.message.contains("first message"));
    client.wait_for_close().await;
}

/// Encodes several client messages back to back into one wire chunk,
/// upstream's coalesced-frame fixtures.
pub fn coalesced_client_wire(messages: &[pi_protocol::ClientMessage]) -> Option<Vec<u8>> {
    let mut wire = Vec::new();
    for message in messages {
        wire.extend_from_slice(
            &pi_protocol::encode_client_message(
                message,
                pi_protocol::FrameDecoderOptions::default(),
            )
            .ok()?,
        );
    }
    Some(wire)
}
