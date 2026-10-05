//! The Session-protocol conformance suite, ported from upstream
//! `test/conformance.test.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test assertions panic at the failing case only; the restriction lints target production code"
)]

mod support;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_agent_core::harness::context::{Context, background_context};
use pi_agent_core::harness::session::types::SessionMetadata;
use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::types::{JsonValue, ServiceCall};
use pi_protocol::{
    ProtocolError, ResponseEnvelope, RpcTarget, ServerId, ServerMessage, SessionTarget,
};
use pi_server::testing::{Deferred, TestServerHost, create_test_server_services};
use pi_server::{
    Failure, RoutedServerPresentation, RoutedServerServiceAttachment, RoutedServerServiceHost,
    RoutedSessionHandle, ServerError, ServerHost, ServicePublisher,
};

use support::{
    SERVER_ID, Servers, arbitrary_call, connect, connect_over, create_server, create_server_over,
    poll_until, run_local, same_failure, server_target, session_call, version,
};

fn response_failure<'a>(message: &'a ServerMessage, id: &str) -> Option<&'a ProtocolError> {
    match message {
        ServerMessage::Response(ResponseEnvelope::Failure(failure)) if failure.id == id => {
            Some(&failure.error)
        }
        _ => None,
    }
}

fn error_code(message: &ServerMessage, id: &str) -> String {
    response_failure(message, id)
        .map_or_else(|| "<no failure>".to_string(), |error| error.code.clone())
}

/// The metadata-identity host, upstream's `attach passes concrete repository
/// metadata` case.
struct MetadataHost {
    services: Rc<dyn RoutedServerServiceHost>,
    metadata: Rc<BackendMetadata>,
    received: Rc<RefCell<Option<Rc<BackendMetadata>>>>,
}

#[derive(Debug)]
struct BackendMetadata {
    base: SessionMetadata,
    path: String,
    modified_at: i64,
}

impl pi_server::HasSessionId for BackendMetadata {
    fn session_id(&self) -> &str {
        &self.base.id
    }
}

impl ServerHost for MetadataHost {
    type Metadata = BackendMetadata;

    fn server_services(&self) -> Rc<dyn RoutedServerServiceHost> {
        Rc::clone(&self.services)
    }

    fn resolve_session(
        &self,
        _session_id: &str,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<BackendMetadata>, Failure>> {
        let metadata = Rc::clone(&self.metadata);
        boxed(async move { Ok(metadata) })
    }

    fn open_session(
        &self,
        metadata: Rc<BackendMetadata>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionHandle>, Failure>> {
        *self.received.borrow_mut() = Some(Rc::clone(&metadata));
        let handle: Rc<dyn RoutedSessionHandle> = Rc::new(support::InlineHandle {
            terminated: None,
            continue_acquiring: None,
            acquiring_entered: None,
            release_count: Rc::new(Cell::new(0)),
            on_attach: None,
        });
        boxed(async { Ok(handle) })
    }
}

/// The recording host, upstream's `routes opaque server services` case.
struct RecordingHost {
    backing: Rc<TestServerHost>,
    services: Rc<dyn RoutedServerServiceHost>,
}

impl ServerHost for RecordingHost {
    type Metadata = SessionMetadata;

    fn server_services(&self) -> Rc<dyn RoutedServerServiceHost> {
        Rc::clone(&self.services)
    }

    fn resolve_session(
        &self,
        session_id: &str,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<SessionMetadata>, Failure>> {
        self.backing.resolve_session(session_id, context)
    }

    fn open_session(
        &self,
        metadata: Rc<SessionMetadata>,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionHandle>, Failure>> {
        self.backing.open_session(metadata, context)
    }
}

struct RecordingServices {
    release_count: Rc<Cell<u32>>,
}

impl RoutedServerServiceHost for RecordingServices {
    fn attach_client(
        &self,
        presentation: Rc<dyn RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedServerServiceAttachment>, Failure>> {
        let lease: Rc<dyn RoutedServerServiceAttachment> = Rc::new(RecordingLease {
            presentation,
            release_count: Rc::clone(&self.release_count),
        });
        boxed(async { Ok(lease) })
    }
}

struct RecordingLease {
    presentation: Rc<dyn RoutedServerPresentation>,
    release_count: Rc<Cell<u32>>,
}

impl RoutedServerServiceAttachment for RecordingLease {
    fn invoke_service(
        &self,
        call: ServiceCall,
        _publish: ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        let presentation = Rc::clone(&self.presentation);
        boxed(async move {
            if call.service_id != "pi.session-management" {
                return Err(Failure::message("Unexpected service"));
            }
            if call.member == "attach"
                && let Some(JsonValue::Str(session_id)) = call.args.first()
            {
                presentation.attach_session(session_id, context).await?;
                return Ok(Some(JsonValue::Null));
            }
            if call.member == "detach" {
                presentation.detach_session(context).await?;
                return Ok(Some(JsonValue::Null));
            }
            Err(Failure::message("Unexpected service member"))
        })
    }

    fn release(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        self.release_count.set(self.release_count.get() + 1);
        boxed(async { Ok(()) })
    }
}

/// The ambiguous host, upstream's `rejects an ambiguous session ID` case.
struct AmbiguousHost {
    services: Rc<dyn RoutedServerServiceHost>,
}

impl ServerHost for AmbiguousHost {
    type Metadata = SessionMetadata;

    fn server_services(&self) -> Rc<dyn RoutedServerServiceHost> {
        Rc::clone(&self.services)
    }

    fn resolve_session(
        &self,
        _session_id: &str,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<SessionMetadata>, Failure>> {
        boxed(async { Err(Failure::Server(ServerError::session_ambiguous())) })
    }

    fn open_session(
        &self,
        _metadata: Rc<SessionMetadata>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn RoutedSessionHandle>, Failure>> {
        boxed(async {
            Err(Failure::message(
                "must not create a Harness for an ambiguous session",
            ))
        })
    }
}

/// The concurrent-termination host, upstream's `releases a lease acquired
/// concurrently with Harness termination` case.
type TerminatingHost = support::InlineHost;

/// The handle the terminating host opens, the shared deferreds wired once.
#[allow(
    clippy::needless_pass_by_value,
    reason = "the gates join the factory, upstream's host-held deferreds"
)]
fn terminating_open(gates: Rc<TerminatingGates>) -> support::HandleFactory {
    let handle: Rc<dyn RoutedSessionHandle> = Rc::new(support::InlineHandle {
        terminated: Some(gates.terminated.clone()),
        continue_acquiring: Some(gates.continue_acquiring.clone()),
        acquiring_entered: Some(gates.acquiring.clone()),
        release_count: Rc::clone(&gates.release_count),
        on_attach: None,
    });
    support::open_fixed(handle)
}

/// The shared termination gates the case drives, upstream's host-held
/// deferreds.
struct TerminatingGates {
    acquiring: Deferred<()>,
    continue_acquiring: Deferred<()>,
    terminated: Deferred<Option<Failure>>,
    release_count: Rc<Cell<u32>>,
}

fn inline_services() -> Rc<dyn RoutedServerServiceHost> {
    create_test_server_services()
}

/// Awaits two futures to completion, upstream's `Promise.all`.
async fn futures_join<A, B>(
    left: impl Future<Output = A>,
    right: impl Future<Output = B>,
) -> (A, B) {
    tokio::join!(left, right)
}

#[test]
fn handshake_identifies_the_logical_server_without_listing_sessions() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        let server = create_server(&host);
        servers.track(&server);
        let client = connect(&server);

        let hello = client.hello(version(8.0)).await.unwrap();
        assert!(
            matches!(hello, ServerMessage::Hello(answer) if answer.server_id.as_str() == SERVER_ID)
        );
        assert_eq!(host.harness_sessions(), 0);

        servers.close_all().await;
    });
}

#[test]
fn rejects_a_semantically_invalid_service_call_after_envelope_decoding() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        let server = create_server(&host);
        servers.track(&server);
        let client = connect(&server);
        client.hello(version(8.0)).await.unwrap();
        let response = client.next(support::predicate(|message: &ServerMessage| {
            response_failure(message, "invalid-call").is_some()
        }));
        client
            .send_message(&pi_protocol::ClientMessage::Request(
                pi_protocol::RequestEnvelope {
                    id: "invalid-call".to_string(),
                    target: server_target(),
                    call: arbitrary_call(),
                },
            ))
            .await;
        let response = response.await.unwrap();
        assert_eq!(error_code(&response, "invalid-call"), "invalid_request");

        servers.close_all().await;
    });
}

#[test]
fn attach_passes_concrete_repository_metadata_to_the_harness_host() {
    run_local(async {
        let servers = Servers::default();
        let metadata = Rc::new(BackendMetadata {
            base: SessionMetadata {
                id: "session-1".to_string(),
                created_at: 1,
                storage_version: 1,
                cwd: Some("/workspace".to_string()),
                parent_session_id: None,
                legacy_parent_session_path: None,
            },
            path: "/sessions/session-1.jsonl".to_string(),
            modified_at: 2,
        });
        let received = Rc::new(RefCell::new(None::<Rc<BackendMetadata>>));
        let host = Rc::new(MetadataHost {
            services: inline_services(),
            metadata: Rc::clone(&metadata),
            received: Rc::clone(&received),
        });
        let server = create_server_over(host, None);
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();

        let attached = client.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(attached, ResponseEnvelope::Success(_)));
        let received = received.borrow().clone().unwrap();
        assert!(Rc::ptr_eq(&received, &metadata));
        assert_eq!(received.path, metadata.path);
        assert_eq!(received.modified_at, metadata.modified_at);

        servers.close_all().await;
    });
}

#[test]
fn routes_opaque_server_services_and_publishes_attachment_changes_out_of_band() {
    run_local(async {
        let servers = Servers::default();
        let backing = Rc::new(TestServerHost::new());
        backing.seed("session-1", None).await.unwrap();
        let release_count = Rc::new(Cell::new(0));
        let host = Rc::new(RecordingHost {
            backing: Rc::clone(&backing),
            services: Rc::new(RecordingServices {
                release_count: Rc::clone(&release_count),
            }),
        });
        let server = create_server_over(host, None);
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();
        let attached = client.next(support::predicate(|message: &ServerMessage| {
            attachment_target(message).is_some_and(|target| target.session_id == "session-1")
        }));
        let attach = client.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(attach, ResponseEnvelope::Success(_)));
        let attached = attached.await.unwrap();
        let target = attachment_target(&attached).unwrap();
        assert_eq!(target.session_id, "session-1");
        assert!(!target.attachment_id.is_empty());
        assert_eq!(backing.latest_harness("session-1").attached_clients(), 1);

        let detached = client.next(support::predicate(|message: &ServerMessage| {
            matches!(message, ServerMessage::Attachment(env) if env.attachment.is_none())
        }));
        let detach = client
            .request_service(server_target(), session_call_detach(), None)
            .await
            .unwrap();
        assert!(matches!(detach, ResponseEnvelope::Success(_)));
        let detached = detached.await.unwrap();
        assert!(matches!(detached, ServerMessage::Attachment(env) if env.attachment.is_none()));
        assert_eq!(backing.latest_harness("session-1").attached_clients(), 0);
        client.close().await;
        poll_until(|| release_count.get() == 1).await;

        servers.close_all().await;
    });
}

fn session_call_detach() -> ServiceCall {
    ServiceCall {
        service_id: "pi.session-management".to_string(),
        instance: None,
        member: "detach".to_string(),
        args: Vec::new(),
    }
}

const fn attachment_target(message: &ServerMessage) -> Option<&SessionTarget> {
    match message {
        ServerMessage::Attachment(pi_protocol::AttachmentEnvelope {
            attachment: Some(target),
        }) => Some(target),
        _ => None,
    }
}

#[test]
fn permits_multiple_client_attachments_per_session() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        let server = create_server(&host);
        servers.track(&server);
        let first = connect(&server);
        let second = connect(&server);
        let (first_hello, second_hello) =
            futures_join(first.hello(version(8.0)), second.hello(version(8.0))).await;
        first_hello.unwrap();
        second_hello.unwrap();

        let attach = first.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(attach, ResponseEnvelope::Success(_)));
        let reattach = first.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(reattach, ResponseEnvelope::Success(_)));
        assert_eq!(host.latest_harness("session-1").attached_clients(), 1);
        let second_attach = second.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(second_attach, ResponseEnvelope::Success(_)));
        assert_eq!(host.harnesses_for("session-1"), 1);
        assert_eq!(host.latest_harness("session-1").attached_clients(), 2);

        first.close().await;
        poll_until(|| host.latest_harness("session-1").attached_clients() == 1).await;
        let reattach_second = second.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(reattach_second, ResponseEnvelope::Success(_)));

        servers.close_all().await;
    });
}

#[test]
fn clears_connection_ownership_when_attachment_release_fails() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        let (errors, observer) = support::error_recorder();
        let server = create_server_over(Rc::clone(&host), Some(observer));
        servers.track(&server);
        let first = connect(&server);
        let second = connect(&server);
        let (first_hello, second_hello) =
            futures_join(first.hello(version(8.0)), second.hello(version(8.0))).await;
        first_hello.unwrap();
        second_hello.unwrap();
        first.attach(SERVER_ID, "session-1").await.unwrap();
        let harness = host.latest_harness("session-1");
        let release_error = Failure::message("release failed");
        harness.set_fail_attachment_release(Some(release_error.clone()));

        first.close().await;
        poll_until(|| harness.attachment_release_count() == 1).await;
        poll_until(|| {
            errors
                .borrow()
                .iter()
                .any(|error| same_failure(error, &release_error))
        })
        .await;
        harness.set_fail_attachment_release(None);
        let second_attach = second.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(second_attach, ResponseEnvelope::Success(_)));

        servers.close_all().await;
    });
}

#[test]
fn requires_the_requesting_client_to_hold_the_targeted_session_attachment() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        let (first_seed, second_seed) =
            futures_join(host.seed("session-1", None), host.seed("session-2", None)).await;
        first_seed.unwrap();
        second_seed.unwrap();
        let server = create_server(&host);
        servers.track(&server);
        let attached = connect(&server);
        let unattached = connect(&server);
        let (attached_hello, unattached_hello) =
            futures_join(attached.hello(version(8.0)), unattached.hello(version(8.0))).await;
        attached_hello.unwrap();
        unattached_hello.unwrap();

        let rejected = unattached
            .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
            .await
            .unwrap();
        assert_eq!(
            error_code(&ServerMessage::Response(rejected), "request-1"),
            "session_not_attached"
        );
        attached.attach(SERVER_ID, "session-1").await.unwrap();
        let wrong_session = attached
            .request_session_service(SERVER_ID, "session-2", session_call("run", vec![]), None)
            .await
            .unwrap();
        assert_eq!(
            error_code(&ServerMessage::Response(wrong_session), "request-2"),
            "session_not_attached"
        );
        let answered = attached
            .request_session_service(
                SERVER_ID,
                "session-1",
                session_call("run", vec![JsonValue::Str("Hello".to_string())]),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            response_success(&ServerMessage::Response(answered), "request-3"),
            Some(&ok_result())
        );
        assert_eq!(
            host.latest_harness("session-1").service_calls(),
            vec![session_call(
                "run",
                vec![JsonValue::Str("Hello".to_string())]
            )]
        );

        servers.close_all().await;
    });
}

fn response_success<'a>(message: &'a ServerMessage, id: &str) -> Option<&'a Option<JsonValue>> {
    match message {
        ServerMessage::Response(ResponseEnvelope::Success(success)) if success.id == id => {
            Some(&success.result)
        }
        _ => None,
    }
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "the Option mirrors upstream's nullable result shape the assertions compare"
)]
fn ok_result() -> Option<JsonValue> {
    Some(JsonValue::Object(
        pi_chord::types::JsonObject::from_entries(vec![("ok".to_string(), JsonValue::Bool(true))]),
    ))
}

#[test]
fn rejects_a_stale_attachment_route_after_switching_sessions() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        let (first_seed, second_seed) =
            futures_join(host.seed("session-1", None), host.seed("session-2", None)).await;
        first_seed.unwrap();
        second_seed.unwrap();
        let server = create_server(&host);
        servers.track(&server);
        let client = connect(&server);
        client.hello(version(8.0)).await.unwrap();
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let first_attachment_id = support::latest_attachment_id(&client, "session-1");
        client.attach(SERVER_ID, "session-2").await.unwrap();

        let stale = client
            .request_service(
                RpcTarget::Session(SessionTarget {
                    server_id: ServerId::new(SERVER_ID).unwrap(),
                    session_id: "session-1".to_string(),
                    attachment_id: first_attachment_id,
                }),
                session_call("run", vec![JsonValue::Str("stale".to_string())]),
                None,
            )
            .await
            .unwrap();
        let ResponseEnvelope::Failure(failure) = stale else {
            panic!("the stale route fails");
        };
        assert_eq!(failure.error.code, "session_not_attached");
        assert!(host.latest_harness("session-1").service_calls().is_empty());

        servers.close_all().await;
    });
}

#[test]
fn preserves_opaque_service_results_and_bounds_adapter_defects() {
    run_local(async {
        let (servers, host, _server, client) = support::greeted_case().await;
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let harness = host.latest_harness("session-1");
        harness.set_next_service_result(Some(JsonValue::Object(
            pi_chord::types::JsonObject::from_entries(vec![
                ("accepted".to_string(), JsonValue::Bool(false)),
                ("reason".to_string(), JsonValue::Str("closed".to_string())),
            ]),
        )));
        let opaque = client
            .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
            .await
            .unwrap();
        assert_eq!(
            response_success(&ServerMessage::Response(opaque), "request-2"),
            Some(&Some(JsonValue::Object(
                pi_chord::types::JsonObject::from_entries(vec![
                    ("accepted".to_string(), JsonValue::Bool(false)),
                    ("reason".to_string(), JsonValue::Str("closed".to_string())),
                ]),
            )))
        );

        harness.set_next_service_error(Some(Failure::message("private adapter detail")));
        let defect = client
            .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
            .await
            .unwrap();
        let defect_message = ServerMessage::Response(defect);
        let error = response_failure(&defect_message, "request-3").unwrap();
        assert_eq!(error.code, "internal_error");
        assert_eq!(error.message, "Internal server error");

        servers.close_all().await;
    });
}

#[test]
fn admits_concurrent_service_calls_to_the_attached_session() {
    run_local(async {
        let (servers, host, _server, client) = support::greeted_case().await;
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let harness = host.latest_harness("session-1");
        let gate = harness.gate_next_service_call();
        let first_client = Rc::clone(&client);
        let first = tokio::task::spawn_local(async move {
            first_client
                .request_session_service(
                    SERVER_ID,
                    "session-1",
                    session_call("run", vec![JsonValue::Str("first".to_string())]),
                    None,
                )
                .await
        });
        gate.entered.promise().await;
        let second_client = Rc::clone(&client);
        let second = tokio::task::spawn_local(async move {
            second_client
                .request_session_service(
                    SERVER_ID,
                    "session-1",
                    session_call("run", vec![JsonValue::Str("second".to_string())]),
                    None,
                )
                .await
        });

        let second = second.await.unwrap().unwrap();
        assert!(matches!(second, ResponseEnvelope::Success(_)));
        assert_eq!(
            harness.service_calls(),
            vec![
                session_call("run", vec![JsonValue::Str("first".to_string())]),
                session_call("run", vec![JsonValue::Str("second".to_string())]),
            ]
        );
        gate.release.resolve(());
        let first = first.await.unwrap().unwrap();
        assert!(matches!(first, ResponseEnvelope::Success(_)));

        servers.close_all().await;
    });
}

#[test]
fn keeps_attachment_demand_until_an_accepted_service_call_settles_after_disconnect() {
    run_local(async {
        let (servers, host, _server, client) = support::greeted_case().await;
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let harness = host.latest_harness("session-1");
        let gate = harness.gate_next_service_call();
        let calling_client = Rc::clone(&client);
        let calling = tokio::task::spawn_local(async move {
            calling_client
                .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
                .await
        });
        gate.entered.promise().await;

        client.close().await;
        assert_eq!(harness.attached_clients(), 1);
        gate.release.resolve(());
        let calling = calling.await.unwrap();
        assert!(
            calling.is_err_and(|error| error.contains("closed")),
            "the closed wire rejects the waiting response"
        );
        poll_until(|| harness.attached_clients() == 0).await;

        servers.close_all().await;
    });
}

#[test]
fn rejects_requests_addressed_to_another_server_before_repository_access() {
    run_local(async {
        let (servers, host, _server, client) = support::greeted_case().await;

        let response = client
            .attach("00000000-0000-4000-8000-000000000002", "session-1")
            .await
            .unwrap();
        assert_eq!(
            error_code(&ServerMessage::Response(response), "request-1"),
            "wrong_server"
        );
        assert_eq!(host.harness_sessions(), 0);

        servers.close_all().await;
    });
}

#[test]
fn reports_an_unknown_session_without_creating_a_harness() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        let server = create_server(&host);
        servers.track(&server);
        let client = connect(&server);
        client.hello(version(8.0)).await.unwrap();

        let response = client.attach(SERVER_ID, "missing").await.unwrap();
        assert_eq!(
            error_code(&ServerMessage::Response(response), "request-1"),
            "session_not_found"
        );
        assert_eq!(host.harness_sessions(), 0);

        servers.close_all().await;
    });
}

#[test]
fn rejects_an_ambiguous_session_id_without_creating_a_harness() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(AmbiguousHost {
            services: inline_services(),
        });
        let server = create_server_over(host, None);
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();

        let response = client.attach(SERVER_ID, "duplicate").await.unwrap();
        assert_eq!(
            error_code(&ServerMessage::Response(response), "request-1"),
            "session_ambiguous"
        );

        servers.close_all().await;
    });
}

#[test]
fn invalidates_a_terminated_harness_handle_and_allows_a_later_attach() {
    run_local(async {
        let (servers, host, _server, client) = support::greeted_case().await;
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let first_harness = host.latest_harness("session-1");

        first_harness
            .terminate(Failure::message("worker crashed"))
            .await;
        first_harness.terminated().await;
        poll_until(|| first_harness.attached_clients() == 0).await;
        assert_eq!(first_harness.attachment_release_count(), 1);

        let reattach = client.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(reattach, ResponseEnvelope::Success(_)));
        assert_eq!(host.harnesses_for("session-1"), 2);

        servers.close_all().await;
    });
}

#[test]
fn connection_loss_releases_its_attachment_while_server_shutdown_closes_the_harness() {
    run_local(async {
        let (servers, host, server, client) = support::greeted_case().await;
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let harness = host.latest_harness("session-1");

        client.close().await;
        poll_until(|| harness.attached_clients() == 0).await;
        assert_eq!(harness.close_count(), 0);
        servers.forget(&server);
        server.close().await.unwrap();
        assert_eq!(harness.close_count(), 1);
    });
}

#[test]
fn releases_a_lease_acquired_concurrently_with_harness_termination() {
    run_local(async {
        let servers = Servers::default();
        let metadata = support::metadata("session-1");
        let release_count = Rc::new(Cell::new(0));
        let gates = Rc::new(TerminatingGates {
            acquiring: Deferred::new(),
            continue_acquiring: Deferred::new(),
            terminated: Deferred::new(),
            release_count: Rc::clone(&release_count),
        });
        let host = Rc::new(TerminatingHost {
            services: inline_services(),
            metadata: Rc::clone(&metadata),
            open: terminating_open(Rc::clone(&gates)),
        });
        let server = create_server_over(host, None);
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();
        let attach_client = Rc::clone(&client);
        let attach =
            tokio::task::spawn_local(
                async move { attach_client.attach(SERVER_ID, "session-1").await },
            );
        gates.acquiring.promise().await;

        gates
            .terminated
            .resolve(Some(Failure::message("worker crashed")));
        gates.continue_acquiring.resolve(());
        let attach = attach.await.unwrap().unwrap();
        assert_eq!(
            error_code(&ServerMessage::Response(attach), "request-1"),
            "server_draining"
        );
        assert_eq!(release_count.get(), 1);

        servers.close_all().await;
    });
}

#[test]
fn shares_a_harness_creation_failure_releases_the_session_and_allows_a_later_retry() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        host.set_next_open_session_error(Some(Failure::message("Harness creation failed")));
        let server = create_server(&host);
        servers.track(&server);
        let first = connect(&server);
        let second = connect(&server);
        let (first_hello, second_hello) =
            futures_join(first.hello(version(8.0)), second.hello(version(8.0))).await;
        first_hello.unwrap();
        second_hello.unwrap();
        let gate = host.gate_next_open_session();

        let first_client = Rc::clone(&first);
        let first_attach =
            tokio::task::spawn_local(
                async move { first_client.attach(SERVER_ID, "session-1").await },
            );
        gate.entered.promise().await;
        let second_client = Rc::clone(&second);
        let second_attach =
            tokio::task::spawn_local(
                async move { second_client.attach(SERVER_ID, "session-1").await },
            );
        gate.release.resolve(());

        let first_response = first_attach.await.unwrap().unwrap();
        let second_response = second_attach.await.unwrap().unwrap();
        // Each client counts its own correlation sequence, so the second
        // client's attach is also `request-1`.
        assert_eq!(
            error_code(&ServerMessage::Response(first_response), "request-1"),
            "internal_error"
        );
        assert_eq!(
            error_code(&ServerMessage::Response(second_response), "request-1"),
            "internal_error"
        );
        assert_eq!(host.open_session_count(), 1);

        let retry = first.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(retry, ResponseEnvelope::Success(_)));
        assert_eq!(host.open_session_count(), 2);
        assert_eq!(host.harnesses_for("session-1"), 1);

        servers.close_all().await;
    });
}

#[test]
fn closes_a_harness_acquired_while_server_shutdown_is_in_progress() {
    run_local(async {
        let (servers, host, server, client) = support::greeted_case().await;
        let gate = host.gate_next_open_session();
        let attach_client = Rc::clone(&client);
        let attach =
            tokio::task::spawn_local(
                async move { attach_client.attach(SERVER_ID, "session-1").await },
            );
        gate.entered.promise().await;
        servers.forget(&server);
        let closing = server.close();
        gate.release.resolve(());

        closing.await.unwrap();
        let attach = attach.await.unwrap();
        assert!(
            attach.is_err(),
            "the closed wire rejects the waiting response"
        );
        assert_eq!(host.latest_harness("session-1").close_count(), 1);
    });
}

#[test]
fn fails_shutdown_when_an_in_flight_acquisition_cannot_release_its_harness() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        let cleanup_error = Failure::message("close failed");
        host.set_next_harness_close_error(Some(cleanup_error));
        let server = create_server(&host);
        servers.track(&server);
        let client = connect(&server);
        client.hello(version(8.0)).await.unwrap();
        let gate = host.gate_next_open_session();
        let attach_client = Rc::clone(&client);
        let attach =
            tokio::task::spawn_local(
                async move { attach_client.attach(SERVER_ID, "session-1").await },
            );
        gate.entered.promise().await;

        servers.forget(&server);
        let closing = server.close();
        gate.release.resolve(());

        let close_error = closing.await.unwrap_err();
        assert!(
            close_error
                .to_string()
                .contains("Failed to close routed Sessions")
        );
        let closed = server.closed().await.unwrap_err();
        assert!(
            closed
                .to_string()
                .contains("Failed to close routed Sessions")
        );
        let attach = attach.await.unwrap();
        assert!(
            attach.is_err(),
            "the closed wire rejects the waiting response"
        );
        assert_eq!(host.latest_harness("session-1").close_count(), 1);

        host.latest_harness("session-1")
            .close(background_context())
            .await
            .unwrap();
    });
}
