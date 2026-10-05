//! The test host and harness, ported from upstream `src/testing/host.ts`.
//!
//! The [`TestHarness`] answers service calls from a scripted queue and
//! exposes the counters the conformance cases assert on; [`TestServerHost`]
//! resolves sessions through an in-memory repository and opens harnesses,
//! with the same error and gate knobs upstream carries.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::harness::context::{Context, background_context};
use pi_agent_core::harness::session::memory::{MemorySessionRepo, MemorySessionRepoOptions};
use pi_agent_core::harness::session::types::{
    Session, SessionCreateOptions, SessionError, SessionMetadata, SessionRepo,
};
use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::types::{JsonObject, JsonValue, ServiceCall};

use crate::errors::{Failure, ServerError};
use crate::latch::Deferred;
use crate::types::{
    HasSessionId, RoutedServerPresentation, RoutedServerServiceAttachment, RoutedServerServiceHost,
    RoutedSessionAttachment, RoutedSessionHandle, ServerHost, ServicePublisher, ready,
};

/// Builds the `{"ok": true}` JSON the harness answers with, upstream's
/// `nextServiceResult` default.
#[must_use]
pub fn ok_result() -> Option<JsonValue> {
    Some(JsonValue::Object(JsonObject::from_entries(vec![(
        "ok".to_string(),
        JsonValue::Bool(true),
    )])))
}

/// The open/release pair one gate arms, upstream's `OpenGate`.
pub struct OpenGate {
    /// Resolves once the gated operation entered.
    pub entered: Deferred<()>,
    /// Resolves to let the gated operation continue.
    pub release: Deferred<()>,
}

impl std::fmt::Debug for OpenGate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("OpenGate").finish()
    }
}

/// The counters and scripted answers one harness carries, shared between the
/// handle and every lease it hands out, upstream's `TestHarness` fields.
struct HarnessCore {
    attached_clients: Cell<u32>,
    attachment_release_count: Cell<u32>,
    close_count: Cell<u32>,
    service_calls: RefCell<Vec<ServiceCall>>,
    fail_attachment_release: RefCell<Option<Failure>>,
    fail_close: RefCell<Option<Failure>>,
    next_service_error: RefCell<Option<Failure>>,
    next_service_result: RefCell<Option<JsonValue>>,
    next_close_gate: RefCell<Option<OpenGate>>,
    next_service_gate: RefCell<Option<OpenGate>>,
    closed: Deferred<()>,
    termination: Deferred<Option<Failure>>,
}

/// The scripted Session handle the conformance cases drive, upstream's
/// `TestHarness`.
pub struct TestHarness {
    core: Rc<HarnessCore>,
    session: Rc<dyn Session>,
}

impl std::fmt::Debug for TestHarness {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TestHarness").finish()
    }
}

impl TestHarness {
    /// Builds the harness over one opened Session, upstream's `new
    /// TestHarness(session)`.
    pub fn new(session: Rc<dyn Session>) -> Self {
        Self {
            core: Rc::new(HarnessCore {
                attached_clients: Cell::new(0),
                attachment_release_count: Cell::new(0),
                close_count: Cell::new(0),
                service_calls: RefCell::new(Vec::new()),
                fail_attachment_release: RefCell::new(None),
                fail_close: RefCell::new(None),
                next_service_error: RefCell::new(None),
                next_service_result: RefCell::new(ok_result()),
                next_close_gate: RefCell::new(None),
                next_service_gate: RefCell::new(None),
                closed: Deferred::new(),
                termination: Deferred::new(),
            }),
            session,
        }
    }

    /// The live attachment count, upstream's `attachedClients`.
    #[must_use]
    pub fn attached_clients(&self) -> u32 {
        self.core.attached_clients.get()
    }

    /// How many lease releases ran, upstream's `attachmentReleaseCount`.
    #[must_use]
    pub fn attachment_release_count(&self) -> u32 {
        self.core.attachment_release_count.get()
    }

    /// How many closes ran, upstream's `closeCount`.
    #[must_use]
    pub fn close_count(&self) -> u32 {
        self.core.close_count.get()
    }

    /// The recorded service calls, upstream's `serviceCalls`.
    #[must_use]
    pub fn service_calls(&self) -> Vec<ServiceCall> {
        self.core.service_calls.borrow().clone()
    }

    /// Arms the next lease release to fail, upstream's
    /// `failAttachmentRelease`.
    pub fn set_fail_attachment_release(&self, error: Option<Failure>) {
        *self.core.fail_attachment_release.borrow_mut() = error;
    }

    /// Arms the next close to fail, upstream's `failClose`.
    pub fn set_fail_close(&self, error: Option<Failure>) {
        *self.core.fail_close.borrow_mut() = error;
    }

    /// Arms the next service call to fail, upstream's `nextServiceError`.
    pub fn set_next_service_error(&self, error: Option<Failure>) {
        *self.core.next_service_error.borrow_mut() = error;
    }

    /// Scripts the next service call's result, upstream's
    /// `nextServiceResult`; `None` is upstream's `undefined`.
    pub fn set_next_service_result(&self, result: Option<JsonValue>) {
        *self.core.next_service_result.borrow_mut() = result;
    }

    /// Gates the next close, upstream's `gateNextClose`.
    #[must_use]
    pub fn gate_next_close(&self) -> OpenGate {
        let gate = OpenGate {
            entered: Deferred::new(),
            release: Deferred::new(),
        };
        *self.core.next_close_gate.borrow_mut() = Some(OpenGate {
            entered: gate.entered.clone(),
            release: gate.release.clone(),
        });
        gate
    }

    /// Gates the next service call, upstream's `gateNextServiceCall`.
    #[must_use]
    pub fn gate_next_service_call(&self) -> OpenGate {
        let gate = OpenGate {
            entered: Deferred::new(),
            release: Deferred::new(),
        };
        *self.core.next_service_gate.borrow_mut() = Some(OpenGate {
            entered: gate.entered.clone(),
            release: gate.release.clone(),
        });
        gate
    }

    /// Terminates the handle with `error` after closing the Session,
    /// upstream's `terminate`.
    pub async fn terminate(&self, error: Failure) {
        let _ = self.session.close(&background_context()).await;
        self.core.termination.resolve(Some(error));
    }

    /// Resolves once an expected close finished, upstream's `closed`.
    #[must_use]
    pub fn closed(&self) -> LocalBoxFuture<()> {
        self.core.closed.promise()
    }

    /// Resolves with the termination error, or `None` after an expected
    /// close, upstream's `terminated`.
    #[must_use]
    pub fn terminated(&self) -> LocalBoxFuture<Option<Failure>> {
        self.core.termination.promise()
    }
}

impl RoutedSessionHandle for TestHarness {
    fn attach_client(
        &self,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionAttachment>, Failure>> {
        self.core
            .attached_clients
            .set(self.core.attached_clients.get() + 1);
        let lease: Rc<dyn RoutedSessionAttachment> = Rc::new(TestLease {
            core: Rc::clone(&self.core),
            released: Cell::new(false),
        });
        ready(Ok(lease))
    }

    fn terminated(&self) -> Option<LocalBoxFuture<Option<Failure>>> {
        Some(self.core.termination.promise())
    }

    fn close(&self, context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        let core = Rc::clone(&self.core);
        let session = Rc::clone(&self.session);
        boxed(async move {
            core.close_count.set(core.close_count.get() + 1);
            let gate = core.next_close_gate.borrow_mut().take();
            if let Some(gate) = gate {
                gate.entered.resolve(());
                gate.release.promise().await;
            }
            if let Some(error) = core.fail_close.borrow_mut().take() {
                return Err(error);
            }
            session
                .close(&context)
                .await
                .map_err(|error: SessionError| Failure::other(Rc::new(error)))?;
            core.closed.resolve(());
            core.termination.resolve(None);
            Ok(())
        })
    }
}

/// One lease the harness handed out, upstream's `attachClient` return
/// object.
struct TestLease {
    core: Rc<HarnessCore>,
    released: Cell<bool>,
}

impl RoutedSessionAttachment for TestLease {
    fn invoke_service(
        &self,
        call: ServiceCall,
        _publish: ServicePublisher,
        _context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        let core = Rc::clone(&self.core);
        boxed(async move {
            core.service_calls.borrow_mut().push(call);
            if let Some(error) = core.next_service_error.borrow_mut().take() {
                return Err(error);
            }
            let gate = core.next_service_gate.borrow_mut().take();
            if let Some(gate) = gate {
                gate.entered.resolve(());
                gate.release.promise().await;
            }
            let result = core.next_service_result.borrow_mut().take();
            *core.next_service_result.borrow_mut() = ok_result();
            Ok(result)
        })
    }

    fn release(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        if self.released.get() {
            return ready(Ok(()));
        }
        self.core
            .attachment_release_count
            .set(self.core.attachment_release_count.get() + 1);
        if let Some(error) = self.core.fail_attachment_release.borrow().clone() {
            return ready(Err(error));
        }
        self.released.set(true);
        self.core
            .attached_clients
            .set(self.core.attached_clients.get() - 1);
        ready(Ok(()))
    }
}

/// The server-scoped service endpoint the test host hands every
/// presentation.
///
/// Upstream's `createTestServerServices`: it routes the session-management
/// attach/detach calls into the presentation and rejects everything else.
#[must_use]
pub fn create_test_server_services() -> Rc<dyn RoutedServerServiceHost> {
    Rc::new(TestServerServices)
}

struct TestServerServices;

impl RoutedServerServiceHost for TestServerServices {
    fn attach_client(
        &self,
        presentation: Rc<dyn RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedServerServiceAttachment>, Failure>> {
        let lease: Rc<dyn RoutedServerServiceAttachment> =
            Rc::new(TestServerServiceLease { presentation });
        ready(Ok(lease))
    }
}

struct TestServerServiceLease {
    presentation: Rc<dyn RoutedServerPresentation>,
}

impl RoutedServerServiceAttachment for TestServerServiceLease {
    fn invoke_service(
        &self,
        call: ServiceCall,
        _publish: ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        let presentation = Rc::clone(&self.presentation);
        boxed(async move {
            if call.service_id == "pi.session-management" && call.instance.is_none() {
                if call.member == "attach"
                    && call.args.len() == 1
                    && let Some(JsonValue::Str(session_id)) = call.args.first()
                {
                    presentation.attach_session(session_id, context).await?;
                    return Ok(Some(JsonValue::Null));
                }
                if call.member == "detach" && call.args.is_empty() {
                    presentation.detach_session(context).await?;
                    return Ok(Some(JsonValue::Null));
                }
            }
            Err(Failure::message(format!(
                "Unsupported test server service {}.{}",
                call.service_id, call.member
            )))
        })
    }

    fn release(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        ready(Ok(()))
    }
}

/// The routing state the host mutates across futures, shared through an
/// `Rc` so the trait methods can capture it.
struct TestHostState {
    harnesses: RefCell<HashMap<String, Vec<Rc<TestHarness>>>>,
    open_session_count: Cell<u32>,
    next_open_session_error: RefCell<Option<Failure>>,
    next_harness_close_error: RefCell<Option<Failure>>,
    next_open_session_gate: RefCell<Option<OpenGate>>,
}

/// The host the conformance cases drive, upstream's `TestServerHost`.
pub struct TestServerHost {
    repo: Arc<MemorySessionRepo>,
    services: Rc<dyn RoutedServerServiceHost>,
    state: Rc<TestHostState>,
}

impl std::fmt::Debug for TestServerHost {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TestServerHost").finish()
    }
}

impl Default for TestServerHost {
    fn default() -> Self {
        Self::new()
    }
}

impl TestServerHost {
    /// Builds the host over a fresh in-memory repository, upstream's `new
    /// TestServerHost()`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            repo: Arc::new(MemorySessionRepo::new(MemorySessionRepoOptions {
                now: Some(Arc::new(|| 1)),
            })),
            services: create_test_server_services(),
            state: Rc::new(TestHostState {
                harnesses: RefCell::new(HashMap::new()),
                open_session_count: Cell::new(0),
                next_open_session_error: RefCell::new(None),
                next_harness_close_error: RefCell::new(None),
                next_open_session_gate: RefCell::new(None),
            }),
        }
    }

    /// How many sessions hold harnesses, upstream's `harnesses.size`.
    #[must_use]
    pub fn harness_sessions(&self) -> usize {
        self.state.harnesses.borrow().len()
    }

    /// How many harnesses one session holds, upstream's
    /// `harnesses.get(id)?.length`.
    pub fn harnesses_for(&self, id: &str) -> usize {
        self.state.harnesses.borrow().get(id).map_or(0, Vec::len)
    }

    /// How many opens ran, upstream's `openSessionCount`.
    #[must_use]
    pub fn open_session_count(&self) -> u32 {
        self.state.open_session_count.get()
    }

    /// Arms the next open to fail, upstream's `nextOpenSessionError`.
    pub fn set_next_open_session_error(&self, error: Option<Failure>) {
        *self.state.next_open_session_error.borrow_mut() = error;
    }

    /// Arms the next harness's close to fail, upstream's
    /// `nextHarnessCloseError`.
    pub fn set_next_harness_close_error(&self, error: Option<Failure>) {
        *self.state.next_harness_close_error.borrow_mut() = error;
    }

    /// Gates the next open, upstream's `gateNextOpenSession`.
    #[must_use]
    pub fn gate_next_open_session(&self) -> OpenGate {
        let gate = OpenGate {
            entered: Deferred::new(),
            release: Deferred::new(),
        };
        *self.state.next_open_session_gate.borrow_mut() = Some(OpenGate {
            entered: gate.entered.clone(),
            release: gate.release.clone(),
        });
        gate
    }

    /// The most recently opened harness for one session, upstream's
    /// `latestHarness`.
    ///
    /// # Panics
    /// When the session holds no harness, upstream's `No harness for` throw.
    #[must_use]
    pub fn latest_harness(&self, id: &str) -> Rc<TestHarness> {
        let harnesses = self.state.harnesses.borrow();
        harnesses
            .get(id)
            .and_then(|stack| stack.last())
            .cloned()
            .unwrap_or_else(|| panic!("No harness for {id}"))
    }

    /// Creates one closed session and returns its metadata, upstream's
    /// `seed`.
    ///
    /// # Errors
    /// Whatever the repository raises.
    pub async fn seed(
        &self,
        id: &str,
        parent_session_id: Option<String>,
    ) -> Result<Rc<SessionMetadata>, Failure> {
        let context = background_context();
        let session = self
            .repo
            .create(
                SessionCreateOptions {
                    id: Some(id.to_string()),
                    parent_session_id,
                },
                &context,
            )
            .await
            .map_err(|error: SessionError| Failure::other(Rc::new(error)))?;
        let metadata = Rc::new(session.metadata().clone());
        session
            .close(&context)
            .await
            .map_err(|error: SessionError| Failure::other(Rc::new(error)))?;
        Ok(metadata)
    }
}

impl ServerHost for TestServerHost {
    type Metadata = SessionMetadata;

    fn server_services(&self) -> Rc<dyn RoutedServerServiceHost> {
        Rc::clone(&self.services)
    }

    fn resolve_session(
        &self,
        session_id: &str,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<SessionMetadata>, Failure>> {
        let repo = Arc::clone(&self.repo);
        let session_id = session_id.to_string();
        boxed(async move {
            let sessions = repo
                .list(&context)
                .await
                .map_err(|error: SessionError| Failure::other(Rc::new(error)))?;
            let mut matches: Vec<SessionMetadata> = sessions
                .into_iter()
                .filter(|metadata| metadata.id == session_id)
                .collect();
            match matches.len() {
                0 => Err(Failure::Server(ServerError::session_not_found(format!(
                    "Unknown session: {session_id}"
                )))),
                1 => Ok(Rc::new(matches.swap_remove(0))),
                _ => Err(Failure::Server(ServerError::session_ambiguous())),
            }
        })
    }

    fn open_session(
        &self,
        metadata: Rc<SessionMetadata>,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionHandle>, Failure>> {
        let state = Rc::clone(&self.state);
        let repo = Arc::clone(&self.repo);
        boxed(async move {
            state
                .open_session_count
                .set(state.open_session_count.get() + 1);
            let gate = state.next_open_session_gate.borrow_mut().take();
            if let Some(gate) = gate {
                gate.entered.resolve(());
                gate.release.promise().await;
            }
            let session: Rc<dyn Session> = match repo.open(&metadata, &context).await {
                Ok(session) => Rc::from(session),
                Err(error) => return Err(Failure::other(Rc::new(error))),
            };
            let result: Result<Rc<TestHarness>, Failure> = async {
                if let Some(error) = state.next_open_session_error.borrow_mut().take() {
                    return Err(error);
                }
                let harness = Rc::new(TestHarness::new(Rc::clone(&session)));
                if let Some(error) = state.next_harness_close_error.borrow_mut().take() {
                    harness.set_fail_close(Some(error));
                }
                state
                    .harnesses
                    .borrow_mut()
                    .entry(metadata.session_id().to_string())
                    .or_default()
                    .push(Rc::clone(&harness));
                Ok(harness)
            }
            .await;
            if result.is_err() {
                let _ = session.close(&context).await;
            }
            result.map(|harness| {
                let handle: Rc<dyn RoutedSessionHandle> = harness;
                handle
            })
        })
    }
}

impl TestServerHost {
    /// Opens the same seeded session twice against the raw repository, the
    /// reopen-rejection arm's fixture.
    ///
    /// # Errors
    /// The fixture's own mismatch report when the reopen shape differs from
    /// the repository contract.
    pub async fn repo_open_twice(&self, context: &Context) -> Result<(), Failure> {
        let metadata = self.resolve_session("session-1", context.clone()).await?;
        let first = self.repo.open(&metadata, context).await;
        let second = self.repo.open(&metadata, context).await;
        match (first, second) {
            (Ok(first_session), second) => {
                let _ = second;
                let _ = first_session.close(context).await;
                Err(Failure::message("the second open unexpectedly succeeded"))
            }
            (Err(first), second) => {
                let _ = second;
                let _ = first;
                Ok(())
            }
        }
    }
}
