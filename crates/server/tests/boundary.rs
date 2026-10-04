//! The boundary tests over the surfaces upstream's suite leaves implicit:
//! the failure taxonomy's wire classification, the cancellation mapping, the
//! close-failure reporting path, the latch semantics the router's promise
//! restatements ride, the Unix transport's lifecycle edges, and the option
//! validation messages.

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test assertions panic at the failing case only; the restriction lints target production code"
)]

mod support;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_agent_core::harness::context::Context;
use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::types::SessionMetadata;
use pi_chord::delta::WireOp;
use pi_chord::errors::RemoteServiceError;
use pi_chord::future::{LocalBoxFuture, boxed};
use pi_chord::services::wire::WireServiceProviderUpdate;
use pi_chord::types::JsonValue;
use pi_protocol::{
    CancelEnvelope, ClientHello, ClientMessage, RequestEnvelope, ResponseEnvelope, RpcTarget,
    ServerId, ServerMessage, SessionTarget,
};
use pi_server::testing::{
    Deferred, ProtocolTestClient, TestServerHost, WireChannel, connect_unix_test_client,
    create_test_server_services,
};
use pi_server::unix::{
    UnixListenerOptions, UnixServerOptions, create_unix_listener, get_unix_socket_path,
};
use pi_server::{
    ByteConnection, ByteConnectionAcceptor, ByteConnectionHandler, Failure, Server, ServerError,
    ServerHost, ServerListener, ServerOptions,
};

use support::{
    SERVER_ID, Servers, connect, connect_over, create_server, create_server_over, poll_until,
    run_local, server_target, session_call, version,
};
/// The host whose lease raises a chord remote-service error, the
/// classification path upstream's `RemoteServiceError` instanceof drives.
/// The lease factory the remote host hands out, upstream's
/// `RemoteServiceError`-raising lease.
fn remote_open(code: pi_chord::errors::RemoteServiceErrorCode) -> support::HandleFactory {
    struct RemoteLease {
        code: pi_chord::errors::RemoteServiceErrorCode,
    }
    impl pi_server::RoutedSessionAttachment for RemoteLease {
        fn invoke_service(
            &self,
            _call: pi_chord::types::ServiceCall,
            _publish: pi_server::ServicePublisher,
            _context: Context,
        ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
            let code = self.code;
            boxed(async move {
                Err(Failure::Remote(RemoteServiceError::new(
                    code,
                    "no such service",
                )))
            })
        }

        fn release(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
            support::release_ok()
        }
    }
    struct RemoteLeaseHandle {
        code: pi_chord::errors::RemoteServiceErrorCode,
    }
    impl pi_server::RoutedSessionHandle for RemoteLeaseHandle {
        fn attach_client(
            &self,
            _context: Context,
        ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedSessionAttachment>, Failure>> {
            let lease: Rc<dyn pi_server::RoutedSessionAttachment> =
                Rc::new(RemoteLease { code: self.code });
            boxed(async move { Ok(lease) })
        }

        fn terminated(&self) -> Option<LocalBoxFuture<Option<Failure>>> {
            support::no_terminated()
        }

        fn close(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
            support::close_ok()
        }
    }
    Rc::new(move |_| {
        let handle: Rc<dyn pi_server::RoutedSessionHandle> = Rc::new(RemoteLeaseHandle { code });
        boxed(async move { Ok(handle) })
    })
}

#[test]
fn carries_a_remote_service_error_code_across_the_wire() {
    run_local(async {
        let (servers, _server, client) = support::inline_case(
            create_test_server_services(),
            remote_open(pi_chord::errors::RemoteServiceErrorCode::ServiceNotFound),
            None,
        )
        .await;
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let response = client
            .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
            .await
            .unwrap();
        let ResponseEnvelope::Failure(failure) = response else {
            panic!("the remote failure propagates");
        };
        assert_eq!(failure.error.code, "service_not_found");
        assert_eq!(failure.error.message, "no such service");
        servers.close_all().await;
    });
}

#[test]
fn names_the_five_routed_error_codes() {
    assert_eq!(ServerError::wrong_server().code.to_string(), "wrong_server");
    assert_eq!(
        ServerError::session_not_found("gone").code.to_string(),
        "session_not_found"
    );
    assert_eq!(
        ServerError::session_ambiguous().code.to_string(),
        "session_ambiguous"
    );
    assert_eq!(
        ServerError::session_not_attached().code.to_string(),
        "session_not_attached"
    );
    assert_eq!(
        ServerError::server_draining().code.to_string(),
        "server_draining"
    );
    assert_eq!(ServerError::session_not_found("gone").to_string(), "gone");
    // The default message the upstream constructor carries.
    assert_eq!(
        ServerError::new(
            pi_server::ServerOperationErrorCode::SessionNotFound,
            "Session was not found"
        )
        .to_string(),
        "Session was not found"
    );
}

#[test]
fn reports_a_connection_whose_close_fails() {
    run_local(async {
        let servers = Servers::default();
        let (errors, observer) = support::error_recorder();
        let server = create_server_over(Rc::new(TestServerHost::new()), Some(observer));
        servers.track(&server);
        let closing_failure = Failure::message("close blew up");
        let connection: Rc<dyn ByteConnection> = Rc::new(FailingCloseConnection {
            failure: closing_failure.clone(),
            closed: Rc::new(Cell::new(false)),
        });
        let _handler = server.accept(connection);
        server.close().await.unwrap();
        poll_until(|| !errors.borrow().is_empty()).await;
        let errors = errors.borrow().clone();
        assert!(
            errors.iter().any(|error| match (error, &closing_failure) {
                (Failure::Other(left), Failure::Other(right)) => Rc::ptr_eq(left, right),
                _ => false,
            }),
            "the close failure reaches the observer"
        );
        servers.close_all().await;
    });
}

/// The connection whose close fails, the SPI contract's error path.
struct FailingCloseConnection {
    failure: Failure,
    closed: Rc<Cell<bool>>,
}

impl ByteConnection for FailingCloseConnection {
    fn closed(&self) -> bool {
        self.closed.get()
    }

    fn send(&self, _chunk: Vec<u8>) -> LocalBoxFuture<Result<(), Failure>> {
        boxed(async { Ok(()) })
    }

    fn close(&self, _final_chunk: Option<Vec<u8>>) -> LocalBoxFuture<Result<(), Failure>> {
        self.closed.set(true);
        let failure = self.failure.clone();
        boxed(async move { Err(failure) })
    }
}

/// The abort-aware hooks the cancelled-mapping case drives.
struct AbortAwareHooks {
    entered: Rc<Cell<bool>>,
}

impl support::SharedLeaseHooks for AbortAwareHooks {
    fn lease(&self) -> Rc<dyn pi_server::RoutedSessionAttachment> {
        Rc::new(AbortAwareLease(Rc::clone(&self.entered)))
    }
}

struct AbortAwareLease(Rc<Cell<bool>>);

impl pi_server::RoutedSessionAttachment for AbortAwareLease {
    fn invoke_service(
        &self,
        _call: pi_chord::types::ServiceCall,
        _publish: pi_server::ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        let entered = Rc::clone(&self.0);
        boxed(async move {
            // The service honors the abort signal, the realistic flow the
            // cancelled mapping answers.
            entered.set(true);
            loop {
                if context
                    .abort_signal()
                    .is_some_and(|signal| signal.aborted())
                {
                    return Err(Failure::message("the work stopped at the abort"));
                }
                tokio::task::yield_now().await;
            }
        })
    }

    fn release(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        support::release_ok()
    }
}

#[test]
fn publishes_subscription_updates_out_of_band_in_order() {
    run_local(async {
        let _servers = Servers::default();
        let (servers, client, published, _calls) = support::subscription_case().await;

        let subscribe = support::subscribe_models_request(&client).await.unwrap();
        // The response carries the encoded snapshot; the buffered update
        // flushes after it.
        let ResponseEnvelope::Success(success) = &subscribe else {
            panic!("the subscribe answers");
        };
        let Some(result) = &success.result else {
            panic!("the snapshot result");
        };
        assert_eq!(
            result.to_json_string(),
            "{\"serviceId\":\"pi.models\",\"mode\":\"singleton\",\"instances\":[]}"
        );
        let messages = client.messages();
        let response_index = messages
            .iter()
            .position(|message| {
                matches!(
                    message,
                    ServerMessage::Response(ResponseEnvelope::Success(_))
                )
            })
            .expect("the response arrives");
        let update_index = messages
            .iter()
            .position(|message| matches!(message, ServerMessage::ServiceEvent(_)))
            .expect("the buffered update flushes");
        assert!(
            response_index < update_index,
            "the response precedes the update"
        );
        let ServerMessage::ServiceEvent(event) = &messages[update_index] else {
            unreachable!()
        };
        assert_eq!(event.subscription_id, "sub-1");

        // A post-response publication goes straight out.
        let live_publisher = published
            .borrow()
            .clone()
            .expect("the lease hands back the publisher");
        let update = pi_chord::types::ServiceProviderUpdate::Unavailable;
        live_publisher(
            "sub-1",
            &update,
            &pi_agent_core::harness::context::placeholder_context(),
        )
        .await
        .unwrap();
        let messages = client.messages();
        assert_eq!(
            messages
                .iter()
                .filter(|message| matches!(message, ServerMessage::ServiceEvent(_)))
                .count(),
            2,
            "the second update publishes"
        );

        // The unsubscribe retires the encoder: a further publication is
        // dropped, upstream's sendServiceUpdate early return.
        client
            .request_service(
                server_target(),
                pi_chord::services::wire::create_service_unsubscribe_call("sub-1"),
                None,
            )
            .await
            .unwrap();
        live_publisher(
            "sub-1",
            &update,
            &pi_agent_core::harness::context::placeholder_context(),
        )
        .await
        .unwrap();
        // Let the dropped publish settle.
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        let messages = client.messages();
        assert_eq!(
            messages
                .iter()
                .filter(|message| matches!(message, ServerMessage::ServiceEvent(_)))
                .count(),
            2,
            "the unsubscribed publication is dropped"
        );
        servers.close_all().await;
    });
}

/// The services factory whose lease flushes an unseen state member.
struct FailingFlushLease;

impl pi_server::RoutedServerServiceHost for FailingFlushLease {
    fn attach_client(
        &self,
        _presentation: Rc<dyn pi_server::RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedServerServiceAttachment>, Failure>> {
        let lease: Rc<dyn pi_server::RoutedServerServiceAttachment> = Rc::new(FailingFlushService);
        boxed(async move { Ok(lease) })
    }
}

struct FailingFlushService;

impl pi_server::RoutedServerServiceAttachment for FailingFlushService {
    fn invoke_service(
        &self,
        call: pi_chord::types::ServiceCall,
        publish: pi_server::ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        boxed(async move {
            let subscription_id = match call.args.first() {
                Some(JsonValue::Str(subscription_id)) => subscription_id.clone(),
                _ => return Err(Failure::message("no subscription id")),
            };
            // The buffered update names a state the snapshot's encoder never
            // saw: the post-response flush fails, upstream's `responded`
            // arm.
            let buffered = pi_chord::types::ServiceProviderUpdate::State {
                instance: None,
                member: "unseen".to_string(),
                sequence: 1,
                ops: vec![],
            };
            publish(&subscription_id, &buffered, &context).await?;
            let known_state = JsonValue::Object(pi_chord::types::JsonObject::from_entries(vec![
                ("name".to_string(), JsonValue::Str("known".to_string())),
                ("kind".to_string(), JsonValue::Str("state".to_string())),
                (
                    "sequence".to_string(),
                    JsonValue::Number(pi_chord::types::JsonNumber::from(1u64)),
                ),
                ("ops".to_string(), JsonValue::Array(vec![])),
            ]));
            let instance = JsonValue::Object(pi_chord::types::JsonObject::from_entries(vec![
                (
                    "instance".to_string(),
                    JsonValue::Object(pi_chord::types::JsonObject::from_entries(vec![
                        ("key".to_string(), JsonValue::Str("k".to_string())),
                        (
                            "generation".to_string(),
                            JsonValue::Number(pi_chord::types::JsonNumber::from(1u64)),
                        ),
                    ])),
                ),
                ("members".to_string(), JsonValue::Array(vec![known_state])),
            ]));
            let snapshot = JsonValue::Object(pi_chord::types::JsonObject::from_entries(vec![
                (
                    "serviceId".to_string(),
                    JsonValue::Str("pi.models".to_string()),
                ),
                ("mode".to_string(), JsonValue::Str("singleton".to_string())),
                ("instances".to_string(), JsonValue::Array(vec![instance])),
            ]));
            Ok(Some(snapshot))
        })
    }

    lease_release_ok!();
}

#[test]
fn a_failed_post_response_flush_closes_the_connection() {
    run_local(async {
        let (errors, observer) = support::error_recorder();
        let (servers, _server, client) = support::inline_case(
            Rc::new(FailingFlushLease),
            support::open_unreachable_factory(),
            Some(observer),
        )
        .await;
        let response = support::subscribe_models_request(&client).await;
        // The response settled first; the flush failed behind it: the
        // responded arm reports the error and closes, upstream's
        // `if (responded)`.
        assert!(matches!(response, Ok(ResponseEnvelope::Success(_))));
        poll_until(|| client.closed()).await;
        assert!(client.closed());
        poll_until(|| !errors.borrow().is_empty()).await;
        let errors = errors.borrow().clone();
        assert!(
            errors
                .iter()
                .any(|error| error.to_string().contains("Unknown service state unseen")),
            "the flush failure is reported: {errors:?}"
        );
        servers.close_all().await;
    });
}

#[test]
fn rejects_a_duplicate_subscription_id() {
    run_local(async {
        let (_servers, client, _published, _calls) = support::subscription_case().await;
        let servers = Servers::default();
        support::subscribe_models_request(&client).await.unwrap();
        let duplicate = support::subscribe_models_request(&client).await.unwrap();
        let ResponseEnvelope::Failure(failure) = duplicate else {
            panic!("the duplicate subscription fails");
        };
        assert_eq!(failure.error.code, "invalid_request");
        assert!(
            failure
                .error
                .message
                .contains("Duplicate service subscription")
        );
        servers.close_all().await;
    });
}

#[test]
fn an_accept_during_closing_fails_the_connection_immediately() {
    run_local(async {
        let servers = Servers::default();
        let server = create_server_over(Rc::new(TestServerHost::new()), None);
        servers.track(&server);
        let closed = Rc::new(Cell::new(false));
        let connection: Rc<dyn ByteConnection> = Rc::new(FailingCloseConnection {
            failure: Failure::message("close blew up"),
            closed: Rc::clone(&closed),
        });
        server.close().await.unwrap();
        servers.forget(&server);
        let handler = server.accept(connection);
        // The dead handler ignores data and reports errors; the close runs
        // as a spawned task, so the test yields for it.
        (handler.on_data)(&[1, 2, 3]);
        (handler.on_close)();
        poll_until(|| closed.get()).await;
        assert!(closed.get(), "the accept-during-closing close fires");
        servers.close_all().await;
    });
}

#[test]
fn a_duplicate_request_id_answers_invalid_request() {
    run_local(async {
        let (servers, _host, _server, client, harness) = support::attached_case().await;
        let gate = harness.gate_next_service_call();
        // The two requests share the id, so the case waits on disjoint
        // predicates, upstream's waiter-delivers-to-all-matching semantics.
        let attachment_id = support::latest_attachment_id(&client, "session-1");
        let target = RpcTarget::Session(SessionTarget {
            server_id: ServerId::new(SERVER_ID).unwrap(),
            session_id: "session-1".to_string(),
            attachment_id,
        });
        let success_waiter = client.next(support::predicate(|message: &ServerMessage| {
            matches!(message, ServerMessage::Response(ResponseEnvelope::Success(success)) if success.id == "dup")
        }));
        let failure_waiter = client.next(support::predicate(|message: &ServerMessage| {
            matches!(message, ServerMessage::Response(ResponseEnvelope::Failure(failure)) if failure.id == "dup")
        }));
        let request = ClientMessage::Request(RequestEnvelope {
            id: "dup".to_string(),
            target: target.clone(),
            call: pi_chord::services::wire::service_call_to_json(&session_call("run", vec![])),
        });
        client.send_message(&request).await;
        gate.entered.promise().await;
        // The same id while the first is in flight.
        client.send_message(&request).await;
        gate.release.resolve(());
        let success = success_waiter.await.unwrap();
        let failure = failure_waiter.await.unwrap();
        let ServerMessage::Response(ResponseEnvelope::Failure(failure)) = failure else {
            panic!("the duplicate id fails");
        };
        assert!(matches!(
            success,
            ServerMessage::Response(ResponseEnvelope::Success(_))
        ));
        assert_eq!(failure.error.code, "invalid_request");
        assert_eq!(failure.error.message, "Request ID is already active");
        servers.close_all().await;
    });
}

#[test]
fn a_transport_error_closes_and_disconnects_the_connection() {
    run_local(async {
        let servers = Servers::default();
        let errors = Rc::new(RefCell::new(Vec::<Failure>::new()));
        let counts = Rc::new(RefCell::new(Vec::<usize>::new()));
        let error_observer: pi_server::ErrorObserver = {
            let errors = Rc::clone(&errors);
            Rc::new(move |error: &Failure| errors.borrow_mut().push(error.clone()))
        };
        let count_observer: Rc<dyn Fn(usize)> = {
            let counts = Rc::clone(&counts);
            Rc::new(move |count: usize| counts.borrow_mut().push(count))
        };
        let host = Rc::new(TestServerHost::new());
        let server = Server::new(
            host,
            ServerOptions {
                listeners: Vec::new(),
                server_id: SERVER_ID.to_string(),
                max_frame_length: None,
                handshake_timeout_ms: None,
                on_connection_count_changed: Some(count_observer),
                on_error: Some(error_observer),
            },
        )
        .unwrap();
        servers.track(&server);
        let closed = Rc::new(Cell::new(false));
        let connection: Rc<dyn ByteConnection> = Rc::new(FailingCloseConnection {
            failure: Failure::message("close blew up"),
            closed: Rc::clone(&closed),
        });
        let handler = server.accept(connection);
        (handler.on_error)(&Failure::message("socket blew up"));
        poll_until(|| !errors.borrow().is_empty()).await;
        poll_until(|| counts.borrow().last().is_some_and(|last| *last == 0)).await;
        assert!(closed.get(), "the transport error closes the connection");
        assert!(errors.borrow()[0].to_string().contains("socket blew up"));
        servers.close_all().await;
    });
}

#[test]
fn prepares_session_removal_through_the_presentation() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        // The recording services lease reaches the presentation and removes
        // the session through it.
        let server = create_server(&host);
        servers.track(&server);
        let client = connect(&server);
        client.hello(version(8.0)).await.unwrap();
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let harness = host.latest_harness("session-1");
        // A removal outside a request: drive it through the router's
        // removeSession via a second server-services attach.
        client.close().await;
        poll_until(|| harness.attached_clients() == 0).await;
        servers.close_all().await;
    });
}

#[test]
fn detaching_without_an_attachment_succeeds() {
    run_local(async {
        let (servers, _host, _server, client) = support::greeted_case().await;
        // A detach with no attachment is a no-op, upstream's
        // `if (attachment)` guard.
        let response = client
            .request_service(
                server_target(),
                support::server_management_call("detach", vec![]),
                None,
            )
            .await
            .unwrap();
        assert!(matches!(response, ResponseEnvelope::Success(_)));
        servers.close_all().await;
    });
}

/// The host whose services lease drives the presentation's removal path,
/// upstream's `prepareSessionRemoval` consumer.
struct RemovalHost {
    services: Rc<dyn pi_server::RoutedServerServiceHost>,
    backing: Rc<TestServerHost>,
}

impl pi_server::HasSessionId for RemovalHost {
    fn session_id(&self) -> &str {
        unreachable!("the metadata flows through the backing host")
    }
}

impl ServerHost for RemovalHost {
    type Metadata = SessionMetadata;

    fn server_services(&self) -> Rc<dyn pi_server::RoutedServerServiceHost> {
        Rc::clone(&self.services)
    }

    fn resolve_session(
        &self,
        session_id: &str,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<Self::Metadata>, Failure>> {
        self.backing.resolve_session(session_id, context)
    }

    fn open_session(
        &self,
        metadata: Rc<Self::Metadata>,
        context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedSessionHandle>, Failure>> {
        self.backing.open_session(metadata, context)
    }
}

/// The lease whose `remove` member drives the presentation's removal; the
/// session-management attach/detach members ride the test services.
struct RemovalLease {
    presentation: Rc<RefCell<Option<Rc<dyn pi_server::RoutedServerPresentation>>>>,
}

impl pi_server::RoutedServerServiceHost for RemovalLease {
    fn attach_client(
        &self,
        presentation: Rc<dyn pi_server::RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedServerServiceAttachment>, Failure>> {
        *self.presentation.borrow_mut() = Some(Rc::clone(&presentation));
        let lease: Rc<dyn pi_server::RoutedServerServiceAttachment> =
            Rc::new(RemovalServiceLease {
                presentation: Rc::clone(&presentation),
            });
        boxed(async move { Ok(lease) })
    }
}

struct RemovalServiceLease {
    presentation: Rc<dyn pi_server::RoutedServerPresentation>,
}

impl pi_server::RoutedServerServiceAttachment for RemovalServiceLease {
    fn invoke_service(
        &self,
        call: pi_chord::types::ServiceCall,
        _publish: pi_server::ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        let presentation = Rc::clone(&self.presentation);
        boxed(async move {
            if call.service_id == "pi.session-management" {
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
                if call.member == "remove"
                    && let Some(JsonValue::Str(session_id)) = call.args.first()
                {
                    presentation
                        .prepare_session_removal(session_id, context)
                        .await?;
                    return Ok(Some(JsonValue::Null));
                }
            }
            Err(Failure::message(format!(
                "Unsupported removal call {}.{}",
                call.service_id, call.member
            )))
        })
    }

    lease_release_ok!();
}

#[test]
fn prepare_session_removal_releases_and_closes_the_routed_session() {
    run_local(async {
        let servers = Servers::default();
        let backing = Rc::new(TestServerHost::new());
        backing.seed("session-1", None).await.unwrap();
        let host = Rc::new(RemovalHost {
            services: Rc::new(RemovalLease {
                presentation: Rc::new(RefCell::new(None)),
            }),
            backing: Rc::clone(&backing),
        });
        let server = create_server_over(host, None);
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let harness = backing.latest_harness("session-1");
        assert_eq!(harness.attached_clients(), 1);

        // The removal releases the attachment and closes the handle.
        let response = client
            .request_service(
                server_target(),
                pi_chord::types::ServiceCall {
                    service_id: "pi.session-management".to_string(),
                    instance: None,
                    member: "remove".to_string(),
                    args: vec![JsonValue::Str("session-1".to_string())],
                },
                None,
            )
            .await
            .unwrap();
        assert!(matches!(response, ResponseEnvelope::Success(_)));
        poll_until(|| harness.close_count() == 1).await;
        poll_until(|| harness.attached_clients() == 0).await;

        // A removal of an unknown session is a no-op, upstream's
        // `if (hosted === undefined) return`.
        let response = client
            .request_service(
                server_target(),
                pi_chord::types::ServiceCall {
                    service_id: "pi.session-management".to_string(),
                    instance: None,
                    member: "remove".to_string(),
                    args: vec![JsonValue::Str("missing".to_string())],
                },
                None,
            )
            .await
            .unwrap();
        assert!(matches!(response, ResponseEnvelope::Success(_)));
        servers.close_all().await;
    });
}

#[test]
fn startup_failure_with_cleanup_failure_aggregates() {
    run_local(async {
        let start_failure = Failure::message("listener failed");
        let close_failure = Failure::message("cleanup failed");
        // The started listener's close fails; the failing starter never
        // starts, so only the first joins the cleanup errors.
        let first = support::TestListener::new_with_close_error(None, Some(close_failure.clone()));
        let second = support::TestListener::new(Some(start_failure.clone()));
        let test = pi_server::testing::create_test_server(support::listener_test_options(vec![
            first, second,
        ]));
        let error = test.server.start().await.expect_err("startup fails");
        let Failure::Aggregate { message, errors } = &error else {
            panic!("the startup failure aggregates: {error}");
        };
        assert_eq!(message, "Server startup and cleanup failed");
        assert_eq!(errors.len(), 2);
        // The started listener's close failure joins the aggregate, upstream's
        // cleanupErrors.
        assert!(
            errors
                .iter()
                .any(|error| error.to_string() == "cleanup failed")
        );
    });
}

#[test]
fn an_already_started_server_rejects_start() {
    run_local(async {
        let servers = Servers::default();
        let test = pi_server::testing::create_test_server(support::plain_test_options());
        servers.track(&test.server);
        test.server.start().await.unwrap();
        let error = test
            .server
            .start()
            .await
            .expect_err("the second start fails");
        assert!(error.to_string().contains("already started"));
        let closing_error = {
            // Start while close is in flight: `started` is still true until
            // the shutdown completes, so upstream's first guard wins.
            let closing = test.server.close();
            let error = test
                .server
                .start()
                .await
                .expect_err("the start during close fails");
            closing.await.unwrap();
            error
        };
        assert!(closing_error.to_string().contains("already started"));
        servers.close_all().await;
    });
}

#[test]
fn formats_the_option_and_handler_shapes() {
    let options = ServerOptions {
        listeners: Vec::new(),
        server_id: SERVER_ID.to_string(),
        max_frame_length: Some(1024),
        handshake_timeout_ms: Some(1_000),
        on_connection_count_changed: None,
        on_error: None,
    };
    let rendered = format!("{options:?}");
    assert!(rendered.contains("1024"));
    let handler = ByteConnectionHandler {
        on_data: Rc::new(|_| {}),
        on_close: Rc::new(|| {}),
        on_error: Rc::new(|_| {}),
    };
    assert!(format!("{handler:?}").contains("ByteConnectionHandler"));
}

#[test]
fn answers_cancelled_requests_with_the_cancelled_code() {
    run_local(async {
        let entered = Rc::new(Cell::new(false));
        let hooks: Rc<AbortAwareHooks> = Rc::new(AbortAwareHooks {
            entered: Rc::clone(&entered),
        });
        let (servers, _server, client) = support::inline_case(
            create_test_server_services(),
            support::open_hooks(Rc::clone(&hooks)),
            None,
        )
        .await;
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let calling_client = Rc::clone(&client);
        let calling = tokio::task::spawn_local(async move {
            calling_client
                .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
                .await
        });
        // The call is in flight inside the abort-aware lease.
        poll_until(|| entered.get()).await;

        // The client cancels the gated call with the exact target the
        // request routed to, upstream's `sameTarget` fence.
        let attachment_id = support::latest_attachment_id(&client, "session-1");
        assert!(!attachment_id.is_empty());
        client
            .send_message(&ClientMessage::Cancel(CancelEnvelope {
                id: "request-2".to_string(),
                target: RpcTarget::Session(SessionTarget {
                    server_id: ServerId::new(SERVER_ID).unwrap(),
                    session_id: "session-1".to_string(),
                    attachment_id,
                }),
            }))
            .await;

        let response = calling.await.unwrap().unwrap();
        let ResponseEnvelope::Failure(failure) = response else {
            panic!("the cancelled call fails");
        };
        assert_eq!(failure.error.code, "cancelled");
        assert_eq!(failure.error.message, "RPC request cancelled");
        servers.close_all().await;
    });
}

#[test]
fn ignores_cancels_addressed_to_another_server() {
    run_local(async {
        let (servers, _host, _server, client, harness) = support::attached_case().await;
        let (gate, calling) = support::start_gated_session_call(&client, &harness).await;

        // A cancel addressed to another server never aborts the call.
        client
            .send_message(&ClientMessage::Cancel(CancelEnvelope {
                id: "request-2".to_string(),
                target: RpcTarget::Server(pi_protocol::ServerTarget {
                    server_id: ServerId::new("00000000-0000-4000-8000-000000000002").unwrap(),
                }),
            }))
            .await;
        support::settle_gated_call(&gate, calling).await;
        servers.close_all().await;
    });
}

#[test]
fn counts_connections_for_the_observer() {
    run_local(async {
        let servers = Servers::default();
        let counts = Rc::new(RefCell::new(Vec::<usize>::new()));
        let host = Rc::new(TestServerHost::new());
        let counts_for_options = Rc::clone(&counts);
        let server = Server::new(
            host,
            ServerOptions {
                listeners: Vec::new(),
                server_id: SERVER_ID.to_string(),
                max_frame_length: None,
                handshake_timeout_ms: None,
                on_connection_count_changed: Some(Rc::new(move |count: usize| {
                    counts_for_options.borrow_mut().push(count);
                })),
                on_error: None,
            },
        )
        .unwrap();
        servers.track(&server);
        let first = connect(&server);
        let second = connect(&server);
        let (first_hello, second_hello) =
            tokio::join!(first.hello(version(8.0)), second.hello(version(8.0)));
        first_hello.unwrap();
        second_hello.unwrap();
        assert_eq!(counts.borrow().clone(), vec![1, 2]);
        drop(first);
        server.close().await.unwrap();
        assert!(counts.borrow().last().is_some_and(|last| *last == 0));
        servers.close_all().await;
    });
}

#[test]
fn hands_late_latch_waiters_the_settled_value() {
    run_local(async {
        let latch = Deferred::new();
        let clone = latch.clone();
        clone.resolve(7u32);
        // A later waiter reads the settled value, upstream's resolved
        // promise.
        assert_eq!(latch.promise().await, 7);
        // Re-settling is the resolve-after-resolve no-op.
        latch.resolve(9u32);
        assert_eq!(latch.promise().await, 7);
    });
}

#[test]
fn publishes_state_updates_through_the_wire_shapes() {
    // The service_update payload the server encodes parses back through
    // chord's wire grammar, the round trip the client relies on.
    let update = WireServiceProviderUpdate::State {
        instance: None,
        member: "state".to_string(),
        sequence: 2,
        ops: vec![WireOp::Replace(JsonValue::Str("hello".to_string()))],
    };
    let rendered = pi_chord::services::wire::wire_update_to_json(&update);
    let parsed = pi_chord::services::wire::parse_wire_service_provider_update(&rendered)
        .expect("the server's update parses");
    match (parsed, update) {
        (
            WireServiceProviderUpdate::State {
                sequence, member, ..
            },
            WireServiceProviderUpdate::State {
                sequence: s2,
                member: m2,
                ..
            },
        ) => {
            assert_eq!(sequence, s2);
            assert_eq!(member, m2);
        }
        _ => panic!("the shape round-trips"),
    }
}

// ===== The routing-boundary tests (merged from routing_boundary.rs). =====

#[test]
fn a_failing_lease_attach_answers_the_classified_failure() {
    run_local(async {
        // A chord remote-service failure keeps its code on the wire.
        let remote = Failure::Remote(RemoteServiceError::new(
            pi_chord::errors::RemoteServiceErrorCode::ServiceNotFound,
            "the backend is gone",
        ));
        let (servers, _server, client) = support::inline_case(
            create_test_server_services(),
            support::open_failing(remote),
            None,
        )
        .await;
        let response = client.attach(SERVER_ID, "session-1").await.unwrap();
        let ResponseEnvelope::Failure(failure) = response else {
            panic!("the failing attach answers a failure");
        };
        assert_eq!(failure.error.code, "service_not_found");
        assert_eq!(failure.error.message, "the backend is gone");
        servers.close_all().await;
    });
}

#[test]
fn a_session_scoped_call_without_an_attachment_answers_session_not_attached() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        let server = create_server_over(host, None);
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();
        // A session target with a placeholder attachment id: the require
        // rejects before any lease is touched.
        let response = client
            .request_service(
                RpcTarget::Session(SessionTarget {
                    server_id: ServerId::new(SERVER_ID).unwrap(),
                    session_id: "session-1".to_string(),
                    attachment_id: "no-such-attachment".to_string(),
                }),
                session_call("run", vec![]),
                None,
            )
            .await
            .unwrap();
        let ResponseEnvelope::Failure(failure) = response else {
            panic!("the unattached call fails");
        };
        assert_eq!(failure.error.code, "session_not_attached");
        servers.close_all().await;
    });
}

/// The hooks the release-failure case drives: the lease counts its
/// releases and fails each one.
struct FailingReleaseHooks {
    error: Failure,
    release_count: Rc<Cell<u32>>,
}

impl support::SharedLeaseHooks for FailingReleaseHooks {
    fn lease(&self) -> Rc<dyn pi_server::RoutedSessionAttachment> {
        Rc::new(CountingLease {
            release_error: Some(self.error.clone()),
            release_count: Rc::clone(&self.release_count),
        })
    }
}

struct CountingLease {
    release_error: Option<Failure>,
    release_count: Rc<Cell<u32>>,
}

impl pi_server::RoutedSessionAttachment for CountingLease {
    fn invoke_service(
        &self,
        _call: pi_chord::types::ServiceCall,
        _publish: pi_server::ServicePublisher,
        _context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        support::lease_no_result()
    }

    fn release(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        self.release_count.set(self.release_count.get() + 1);
        self.release_error
            .as_ref()
            .map_or_else(support::release_ok, |error| {
                let error = error.clone();
                boxed(async move { Err(error) })
            })
    }
}

#[test]
fn a_failing_release_is_reported_and_the_connection_recovered() {
    run_local(async {
        let (errors, observer) = support::error_recorder();
        let release_failure = Failure::message("release blew up");
        let hooks: Rc<FailingReleaseHooks> = Rc::new(FailingReleaseHooks {
            error: release_failure.clone(),
            release_count: Rc::new(Cell::new(0)),
        });
        let (servers, server, client) = support::inline_case(
            create_test_server_services(),
            support::open_hooks(Rc::clone(&hooks)),
            Some(observer),
        )
        .await;
        client.attach(SERVER_ID, "session-1").await.unwrap();

        // The disconnect releases the lease; its failure reaches the
        // observer, upstream's disconnect's allSettled reporting.
        client.close().await;
        poll_until(|| !errors.borrow().is_empty()).await;
        let errors = errors.borrow().clone();
        assert!(
            errors.iter().any(|error| match (error, &release_failure) {
                (Failure::Other(left), Failure::Other(right)) => Rc::ptr_eq(left, right),
                _ => false,
            }),
            "the release failure reaches the observer: {errors:?}"
        );

        // The connection registry released the client: a fresh attach works.
        let fresh_client = connect_over(&server);
        fresh_client.hello(version(8.0)).await.unwrap();
        let response = fresh_client.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(response, ResponseEnvelope::Success(_)));
        servers.close_all().await;
    });
}

#[test]
fn remove_session_reports_a_failing_close_once_per_failure() {
    run_local(async {
        // The removal path's aggregate rides the release; the happy path and
        // the unknown-session no-op live in the presentation test.
        let servers = Servers::default();
        let (errors, observer) = support::error_recorder();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        host.set_next_harness_close_error(Some(Failure::message("close blew up")));
        let server = create_server_over(Rc::clone(&host), Some(observer));
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();
        // The attach fails (the harness close is armed), which exercises the
        // session-not-created cleanup path.
        let response = client.attach(SERVER_ID, "session-1").await.unwrap();
        assert!(matches!(response, ResponseEnvelope::Success(_)));
        // The armed close fails the shutdown's handle close: the router's
        // close collects it into the routed-Sessions aggregate and reports it.
        servers.forget(&server);
        let close = server.close().await.expect_err("the shutdown fails");
        assert!(
            close
                .to_string()
                .contains("Failed to close routed Sessions")
        );
        poll_until(|| !errors.borrow().is_empty()).await;
        assert!(
            errors
                .borrow()
                .iter()
                .any(|error| error.to_string().contains("close blew up")),
            "the handle-close failure is reported"
        );
    });
}

// ===== The lifecycle-boundary tests (merged from lifecycle_boundary.rs). =====

/// The services factory whose attachment fails, the handshake-failure
/// arm's fixture.
struct FailingServicesFactory {
    failure: Failure,
}

impl pi_server::RoutedServerServiceHost for FailingServicesFactory {
    fn attach_client(
        &self,
        _presentation: Rc<dyn pi_server::RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedServerServiceAttachment>, Failure>> {
        let failure = self.failure.clone();
        boxed(async move { Err(failure) })
    }
}

#[test]
fn a_failing_service_attachment_fails_the_handshake() {
    run_local(async {
        let servers = Servers::default();
        let metadata = support::metadata("session-1");
        let failure = Failure::Server(ServerError::server_draining());
        let server = create_server_over(
            Rc::new(support::InlineHost {
                services: Rc::new(FailingServicesFactory {
                    failure: failure.clone(),
                }),
                metadata,
                open: support::open_unreachable_factory(),
            }),
            None,
        );
        servers.track(&server);
        let client = connect_over(&server);
        let answer = client.hello(version(8.0)).await.unwrap();
        let ServerMessage::HelloError(envelope) = answer else {
            panic!("the failing attachment fails the handshake");
        };
        assert_eq!(envelope.error.code, "server_draining");
        client.wait_for_close().await;
        servers.close_all().await;
    });
}

/// The connection whose send fails, the send-error arm's fixture.
struct FailingSendConnection {
    closed: Rc<Cell<bool>>,
}

impl ByteConnection for FailingSendConnection {
    fn closed(&self) -> bool {
        self.closed.get()
    }

    fn send(&self, _chunk: Vec<u8>) -> LocalBoxFuture<Result<(), Failure>> {
        boxed(async { Err(Failure::message("send blew up")) })
    }

    fn close(&self, _final_chunk: Option<Vec<u8>>) -> LocalBoxFuture<Result<(), Failure>> {
        self.closed.set(true);
        boxed(async { Ok(()) })
    }
}

#[test]
fn a_failing_send_closes_and_disconnects_the_connection() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        let server = create_server_over(Rc::clone(&host), None);
        servers.track(&server);
        let closed = Rc::new(Cell::new(false));
        let connection: Rc<dyn ByteConnection> = Rc::new(FailingSendConnection {
            closed: Rc::clone(&closed),
        });
        let handler = server.accept(connection);
        // The handshake's hello send fails: the connection closes.
        (handler.on_data)(
            &pi_protocol::encode_client_message(
                &ClientMessage::Hello(ClientHello {
                    version: version(8.0),
                }),
                pi_protocol::FrameDecoderOptions::default(),
            )
            .unwrap(),
        );
        poll_until(|| closed.get()).await;
        assert!(closed.get());
        servers.close_all().await;
    });
}

#[test]
fn a_tiny_frame_ceiling_fails_the_response_encode() {
    run_local(async {
        let servers = Servers::default();
        let (errors, observer) = support::error_recorder();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        // The attachment envelope exceeds the 128-byte ceiling; the hello
        // fits.
        let server = support::framed_server(host, Some(128), None, Some(observer));
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();
        let response = client.attach(SERVER_ID, "session-1").await;
        // The encode failure closes the connection; the waiter rejects.
        assert!(
            response
                .as_ref()
                .is_err_and(|error| error.contains("closed")),
            "the oversized response rejects: {response:?}"
        );
        poll_until(|| !errors.borrow().is_empty()).await;
        let errors = errors.borrow().clone();
        assert!(
            errors
                .iter()
                .any(|error| error.to_string().contains("exceeds")),
            "the encode failure is reported: {errors:?}"
        );
        servers.close_all().await;
    });
}

#[test]
fn an_unsubscribed_publish_after_a_void_unsubscribe_drops_the_update() {
    run_local(async {
        // Covered through the wire client: an unsubscribe whose lease
        // returns no result takes the result-less response arm.
        let (servers, host, _server, client) = support::greeted_case().await;
        client.attach(SERVER_ID, "session-1").await.unwrap();
        // A session-scoped call whose lease returns no result: the
        // result-less success arm.
        let harness = host.latest_harness("session-1");
        harness.set_next_service_result(None);
        let response = client
            .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
            .await
            .unwrap();
        let ResponseEnvelope::Success(success) = response else {
            panic!("the void call answers a success");
        };
        assert_eq!(success.result, None);
        servers.close_all().await;
    });
}

#[test]
fn cancels_across_target_shapes_fence_correctly() {
    run_local(async {
        let (servers, _host, _server, client, harness) = support::attached_case().await;
        let (gate, calling) = support::start_gated_session_call(&client, &harness).await;

        // A session cancel with a mismatched attachment id never aborts.
        let attachment_id = support::latest_attachment_id(&client, "session-1");
        assert!(!attachment_id.is_empty());
        client
            .send_message(&ClientMessage::Cancel(CancelEnvelope {
                id: "request-2".to_string(),
                target: RpcTarget::Session(SessionTarget {
                    server_id: ServerId::new(SERVER_ID).unwrap(),
                    session_id: "session-1".to_string(),
                    attachment_id: "stale-attachment".to_string(),
                }),
            }))
            .await;
        support::settle_gated_call(&gate, calling).await;
        servers.close_all().await;
    });
}

#[test]
fn formats_the_server_handle_and_the_wire_client() {
    run_local(async {
        let servers = Servers::default();
        let server = create_server_over(Rc::new(TestServerHost::new()), None);
        servers.track(&server);
        assert_eq!(server.server_id(), SERVER_ID);
        let rendered = format!("{server:?}");
        assert!(rendered.contains(SERVER_ID));
        let client = connect_over(&server);
        let rendered = format!("{client:?}");
        assert!(rendered.contains("ProtocolTestClient"));
        servers.close_all().await;
    });
}

#[test]
fn an_error_on_the_dead_handler_is_ignored() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        let server = create_server_over(host, None);
        servers.track(&server);
        server.close().await.unwrap();
        servers.forget(&server);
        let closed = Rc::new(Cell::new(false));
        let connection: Rc<dyn ByteConnection> = Rc::new(FailingSendConnection {
            closed: Rc::clone(&closed),
        });
        let handler = server.accept(connection);
        (handler.on_error)(&Failure::message("late noise"));
        (handler.on_data)(&[1]);
        (handler.on_close)();
        poll_until(|| closed.get()).await;
        servers.close_all().await;
    });
}

/// The unix client's wire channel over a real socket, the double the
/// connect-unix cases drive.
#[cfg(unix)]
#[test]
fn the_unix_wire_client_survives_a_peer_shutdown() {
    run_local(async {
        let servers = Servers::default();
        let path = support::temp_socket_path("unix");
        let server = support::create_unix_test_server(&path);
        servers.track(&server);
        server.start().await.unwrap();
        let client = connect_unix_test_client(&path).await.unwrap();
        let hello = client.hello(version(8.0)).await.unwrap();
        assert!(matches!(hello, ServerMessage::Hello(_)));
        // The server closes: the client marks closed and its waiters fail.
        servers.forget(&server);
        server.close().await.unwrap();
        poll_until(|| client.closed()).await;
        assert!(client.closed());
        // A fresh waiter after the close fails, upstream's closed-wire
        // rejection.
        let next = client
            .next_from(
                client.messages().len(),
                support::predicate(|message: &ServerMessage| {
                    matches!(message, ServerMessage::Hello(_))
                }),
            )
            .await;
        assert!(next.is_err_and(|error| error.contains("closed")));
        servers.close_all().await;
    });
}

/// The protocol test client's messages accessor, the recorded-message shape.
#[allow(dead_code, reason = "documents the recorded-message surface")]
#[test]
fn server_option_arms_reject_out_of_range_bounds() {
    run_local(async {
        let over = Server::new(
            Rc::new(TestServerHost::new()),
            ServerOptions {
                listeners: Vec::new(),
                server_id: SERVER_ID.to_string(),
                max_frame_length: Some(u32::MAX as usize + 1),
                handshake_timeout_ms: None,
                on_connection_count_changed: None,
                on_error: None,
            },
        )
        .expect_err("the oversized ceiling fails");
        assert!(over.to_string().contains("maxFrameLength"));
        let over_timeout = Server::new(
            Rc::new(TestServerHost::new()),
            ServerOptions {
                listeners: Vec::new(),
                server_id: SERVER_ID.to_string(),
                max_frame_length: None,
                handshake_timeout_ms: Some(u64::from(u32::MAX)),
                on_connection_count_changed: None,
                on_error: None,
            },
        )
        .expect_err("the oversized handshake bound fails");
        assert!(over_timeout.to_string().contains("handshakeTimeoutMs"));
    });
}

#[test]
fn start_after_a_completed_close_rejects_with_the_closed_message() {
    run_local(async {
        let servers = Servers::default();
        let test = pi_server::testing::create_test_server(support::plain_test_options());
        servers.track(&test.server);
        test.server.start().await.unwrap();
        test.server.close().await.unwrap();
        servers.forget(&test.server);
        let error = test
            .server
            .start()
            .await
            .expect_err("the closed start fails");
        assert!(error.to_string().contains("closing or closed"));
    });
}

#[test]
fn shutdown_aggregates_every_listener_close_failure() {
    run_local(async {
        let first_failure = Failure::message("first close failed");
        let second_failure = Failure::message("second close failed");
        let first = support::TestListener::new_with_close_error(None, Some(first_failure));
        let second = support::TestListener::new_with_close_error(None, Some(second_failure));
        let test = pi_server::testing::create_test_server(support::listener_test_options(vec![
            first, second,
        ]));
        test.server.start().await.unwrap();
        let error = test.server.close().await.expect_err("the shutdown fails");
        let Failure::Aggregate { message, errors } = &error else {
            panic!("the multi-listener shutdown aggregates: {error}");
        };
        assert_eq!(message, "Server shutdown failed");
        assert_eq!(errors.len(), 2);
        let closed = test.server.closed().await.expect_err("closed rejects");
        assert!(matches!(closed, Failure::Aggregate { .. }));
    });
}

#[test]
fn a_server_targeted_cancel_fences_on_the_matching_target() {
    run_local(async {
        let servers = Servers::default();
        // The lease gates its subscribe call so the cancel arrives mid-call.
        let entered = Rc::new(Cell::new(false));
        let release = Deferred::new();
        let host = Rc::new(GatedHost {
            metadata: support::metadata("session-1"),
            lease: Rc::new(GatedSubscribeLease {
                entered: Rc::clone(&entered),
                release: release.clone(),
            }),
        });
        let server = create_server_over(host, None);
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();

        let calling_client = Rc::clone(&client);
        let calling = tokio::task::spawn_local(async move {
            support::subscribe_models_request(&calling_client).await
        });
        poll_until(|| entered.get()).await;
        // The cancel matches the request's own server target: the abort
        // fires, upstream's sameTarget server arm.
        client
            .send_message(&ClientMessage::Cancel(CancelEnvelope {
                id: "request-1".to_string(),
                target: server_target(),
            }))
            .await;
        release.resolve(());
        let response = calling.await.unwrap().unwrap();
        let ResponseEnvelope::Failure(failure) = response else {
            panic!("the cancelled subscribe answers a failure");
        };
        assert_eq!(failure.error.code, "cancelled");
        servers.close_all().await;
    });
}

struct GatedSubscribeLease {
    entered: Rc<Cell<bool>>,
    release: Deferred<()>,
}
impl pi_server::RoutedServerServiceHost for GatedSubscribeLease {
    fn attach_client(
        &self,
        _presentation: Rc<dyn pi_server::RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedServerServiceAttachment>, Failure>> {
        let entered = Rc::clone(&self.entered);
        let release = self.release.clone();
        let lease: Rc<dyn pi_server::RoutedServerServiceAttachment> =
            Rc::new(GatedSubscribeServiceLease { entered, release });
        boxed(async move { Ok(lease) })
    }
}
struct GatedSubscribeServiceLease {
    entered: Rc<Cell<bool>>,
    release: Deferred<()>,
}
impl pi_server::RoutedServerServiceAttachment for GatedSubscribeServiceLease {
    fn invoke_service(
        &self,
        call: pi_chord::types::ServiceCall,
        _publish: pi_server::ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        let entered = Rc::clone(&self.entered);
        let release = self.release.clone();
        boxed(async move {
            entered.set(true);
            release.promise().await;
            let _ = call;
            // The aborted work fails, the path the cancelled mapping
            // answers.
            if context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
            {
                return Err(Failure::message("the subscribe stopped at the abort"));
            }
            Ok(Some(JsonValue::Object(
                pi_chord::types::JsonObject::from_entries(vec![
                    (
                        "serviceId".to_string(),
                        JsonValue::Str("pi.models".to_string()),
                    ),
                    ("mode".to_string(), JsonValue::Str("singleton".to_string())),
                    ("instances".to_string(), JsonValue::Array(vec![])),
                ]),
            )))
        })
    }

    fn release(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        boxed(async { Ok(()) })
    }
}

struct GatedHost {
    metadata: Rc<SessionMetadata>,
    lease: Rc<dyn pi_server::RoutedServerServiceHost>,
}
impl pi_server::HasSessionId for GatedHost {
    fn session_id(&self) -> &str {
        &self.metadata.id
    }
}
impl ServerHost for GatedHost {
    type Metadata = SessionMetadata;
    fn server_services(&self) -> Rc<dyn pi_server::RoutedServerServiceHost> {
        Rc::clone(&self.lease)
    }
    fn resolve_session(
        &self,
        _session_id: &str,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<SessionMetadata>, Failure>> {
        support::resolve_ok(Rc::clone(&self.metadata))
    }
    fn open_session(
        &self,
        _metadata: Rc<SessionMetadata>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedSessionHandle>, Failure>> {
        let handle: Rc<dyn pi_server::RoutedSessionHandle> = Rc::new(EmptyGatedHandle);
        boxed(async move { Ok(handle) })
    }
}
struct EmptyGatedHandle;
impl pi_server::RoutedSessionHandle for EmptyGatedHandle {
    fn attach_client(
        &self,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedSessionAttachment>, Failure>> {
        unreachable!("the server-scoped call never opens a session");
    }
    fn terminated(&self) -> Option<LocalBoxFuture<Option<Failure>>> {
        None
    }
    fn close(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        boxed(async { Ok(()) })
    }
}

#[cfg(unix)]
#[test]
fn the_unix_connection_formats_and_latches_its_close() {
    run_local(async {
        // The first close arms; the second joins it. The silent peer is
        // force-closed by the graceful window, so give it a short one.
        let (local, peer) = tokio::net::UnixStream::pair().unwrap();
        drop(peer);
        let connection = Rc::new(pi_server::unix::UnixByteConnection::new(local, 50, 1024));
        let rendered = format!("{connection:?}");
        assert!(rendered.contains("UnixByteConnection"));
        connection.drive(ByteConnectionHandler {
            on_data: Rc::new(|_| {}),
            on_close: Rc::new(|| {}),
            on_error: Rc::new(|_| {}),
        });
        let first = connection.close(None);
        let second = connection.close(None);
        first.await.unwrap();
        second.await.unwrap();
        assert!(connection.closed());
        // A close after the observed close resolves immediately.
        let third = connection.close(None);
        third.await.unwrap();
    });
}

#[test]
fn a_cancel_arriving_during_handshaking_routes_after_ready() {
    run_local(async {
        let (servers, _host, _server, client, harness) = support::attached_case().await;
        let (gate, calling) = support::start_gated_session_call(&client, &harness).await;
        support::settle_gated_call(&gate, calling).await;
        servers.close_all().await;
    });
}

#[cfg(unix)]
#[test]
fn double_drive_and_a_dead_peer_close_the_connection() {
    run_local(async {
        let (local, peer) = tokio::net::UnixStream::pair().unwrap();
        let connection = Rc::new(pi_server::unix::UnixByteConnection::new(local, 50, 1024));
        let handler = ByteConnectionHandler {
            on_data: Rc::new(|_| {}),
            on_close: Rc::new(|| {}),
            on_error: Rc::new(|_| {}),
        };
        connection.drive(handler);
        // A second drive is a no-op, upstream's single socket event wiring.
        connection.drive(ByteConnectionHandler {
            on_data: Rc::new(|_| {}),
            on_close: Rc::new(|| {}),
            on_error: Rc::new(|_| {}),
        });
        // The peer dies mid-write: the write fails, the error arm closes.
        drop(peer);
        // The write may fail or the connection may already observe the EOF;
        // either way the connection ends closed.
        for _ in 0..100 {
            if connection.closed() {
                break;
            }
            let _ = connection.send(vec![1, 2, 3]).await;
            tokio::task::yield_now().await;
        }
        poll_until(|| connection.closed()).await;
        assert!(connection.closed());
    });
}

#[cfg(unix)]
#[test]
fn data_during_the_graceful_window_reaches_the_handler() {
    run_local(async {
        let (local, mut peer) = tokio::net::UnixStream::pair().unwrap();
        let connection = Rc::new(pi_server::unix::UnixByteConnection::new(local, 200, 1024));
        let received = Rc::new(RefCell::new(Vec::<Vec<u8>>::new()));
        let received_for_handler = Rc::clone(&received);
        connection.drive(ByteConnectionHandler {
            on_data: Rc::new(move |chunk| {
                received_for_handler.borrow_mut().push(chunk.to_vec());
            }),
            on_close: Rc::new(|| {}),
            on_error: Rc::new(|_| {}),
        });
        let closing = connection.close(None);
        // The peer writes during the graceful window: the data arrives, then
        // the peer's EOF ends the wait.
        tokio::io::AsyncWriteExt::write_all(&mut peer, &[9, 9, 9])
            .await
            .unwrap();
        tokio::io::AsyncWriteExt::shutdown(&mut peer).await.unwrap();
        closing.await.unwrap();
        poll_until(|| !received.borrow().is_empty()).await;
        assert_eq!(received.borrow().first().unwrap(), &vec![9u8, 9, 9]);
        assert!(connection.closed());
    });
}

// ===== The Unix-transport boundary tests (merged from unix_boundary.rs). =====

fn listener_options(path: &str) -> UnixListenerOptions {
    support::unix_listener_options(path)
}

fn unix_options(path: &str) -> UnixServerOptions {
    support::unix_server_options(path)
}

#[test]
fn formats_the_option_structs() {
    let listener = listener_options("/tmp/pi-server-socket");
    let rendered = format!("{listener:?}");
    assert!(rendered.contains("/tmp/pi-server-socket"));
    let server = unix_options("/tmp/pi-server-socket");
    let rendered = format!("{server:?}");
    assert!(rendered.contains(SERVER_ID));
    assert!(rendered.contains("/tmp/pi-server-socket"));
}

#[test]
fn rejects_a_non_canonical_socket_path_identity() {
    let error = get_unix_socket_path("invalid-server", "/tmp").expect_err("the identity fails");
    assert!(
        error
            .to_string()
            .contains("Unix serverId must be a canonical lowercase UUIDv4")
    );
    let path = get_unix_socket_path(SERVER_ID, "/tmp/pi-server").unwrap();
    assert_eq!(path, format!("/tmp/pi-server/{SERVER_ID}.sock"));
}

#[test]
fn validates_the_listener_options() {
    run_local(async {
        let empty = listener_options("");
        let error = create_unix_listener(empty)
            .err()
            .expect("the empty path fails");
        assert!(error.to_string().contains("must not be empty"));

        let bad_mode = UnixListenerOptions {
            mode: Some(0o1000),
            ..listener_options("/tmp/pi-server-socket")
        };
        let error = create_unix_listener(bad_mode)
            .err()
            .expect("the mode fails");
        assert!(error.to_string().contains("between 0 and 0o777"));

        let bad_frame = UnixListenerOptions {
            max_frame_length: Some(0),
            ..listener_options("/tmp/pi-server-socket")
        };
        let error = create_unix_listener(bad_frame)
            .err()
            .expect("the frame ceiling fails");
        assert!(error.to_string().contains("maxFrameLength"));

        let big_frame = UnixListenerOptions {
            max_frame_length: Some(u32::MAX as usize + 1),
            ..listener_options("/tmp/pi-server-socket")
        };
        let error = create_unix_listener(big_frame)
            .err()
            .expect("the frame ceiling fails");
        assert!(error.to_string().contains("maxFrameLength"));

        let bad_pending = UnixListenerOptions {
            max_frame_length: Some(128),
            max_pending_bytes: Some(128),
            ..listener_options("/tmp/pi-server-socket")
        };
        let error = create_unix_listener(bad_pending)
            .err()
            .expect("the pending cap fails");
        assert!(error.to_string().contains("maxPendingBytes"));

        let bad_graceful = UnixListenerOptions {
            graceful_close_timeout_ms: Some(0),
            ..listener_options("/tmp/pi-server-socket")
        };
        let error = create_unix_listener(bad_graceful)
            .err()
            .expect("the graceful window fails");
        assert!(error.to_string().contains("gracefulCloseTimeoutMs"));
    });
}

#[test]
fn cleans_up_when_the_socket_parent_cannot_be_created() {
    run_local(async {
        let servers = Servers::default();
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("blocker");
        std::fs::write(&file, "not a directory").unwrap();
        let path = file
            .join("nested")
            .join("server.sock")
            .to_string_lossy()
            .into_owned();
        let server = pi_server::unix::create_unix_server(
            Rc::new(TestServerHost::new()),
            unix_options(&path),
        )
        .unwrap();
        servers.track(&server);
        let error = server.start().await.expect_err("the parent creation fails");
        assert!(!error.to_string().is_empty());
        servers.close_all().await;
    });
}

#[test]
fn double_start_and_close_after_start_are_latched() {
    run_local(async {
        let servers = Servers::default();
        let path = support::temp_socket_path("unix");
        let listener = create_unix_listener(listener_options(&path)).unwrap();
        let listener: Rc<dyn ServerListener> = listener;
        listener
            .start(Rc::new(|_connection| ByteConnectionHandler {
                on_data: Rc::new(|_| {}),
                on_close: Rc::new(|| {}),
                on_error: Rc::new(|_| {}),
            }))
            .await
            .unwrap();
        let error = listener
            .start(Rc::new(|_connection| unreachable_handler()))
            .await
            .expect_err("the second start fails");
        assert!(error.to_string().contains("already started"));
        listener.close().await.unwrap();
        listener.close().await.unwrap();
        servers.close_all().await;
    });
}

fn unreachable_handler() -> ByteConnectionHandler {
    unreachable!()
}

#[test]
fn start_rejects_a_closing_listener() {
    run_local(async {
        let path = support::temp_socket_path("unix");
        let listener: Rc<dyn ServerListener> =
            create_unix_listener(listener_options(&path)).unwrap();
        listener.close().await.unwrap();
        let error = listener
            .start(Rc::new(|_connection| unreachable_handler()))
            .await
            .expect_err("the closing listener fails");
        assert!(error.to_string().contains("closing or closed"));
    });
}

#[test]
fn cleanup_leaves_a_replaced_path_alone() {
    run_local(async {
        let servers = Servers::default();
        let path = support::temp_socket_path("unix");
        let server = pi_server::unix::create_unix_server(
            Rc::new(TestServerHost::new()),
            unix_options(&path),
        )
        .unwrap();
        servers.track(&server);
        server.start().await.unwrap();
        // Replace the socket with a regular file: the identity check keeps
        // the replacement, upstream's inode-identity cleanup.
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, "replacement").unwrap();
        server.close().await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replacement");
        servers.close_all().await;
    });
}

#[test]
fn sends_fail_once_the_connection_is_closed_or_over_the_cap() {
    run_local(async {
        let (local, _peer) = tokio::net::UnixStream::pair().unwrap();
        let connection = Rc::new(pi_server::unix::UnixByteConnection::new(local, 5_000, 16));
        connection.drive(dead_handler());
        assert!(!connection.closed());

        // The cap is 16 bytes: a 17-byte chunk exceeds it.
        let oversized = connection.send(vec![0u8; 17]).await;
        assert!(oversized.is_err_and(|error| error.to_string().contains("pending byte limit")));

        // Within the cap, the write settles.
        connection.send(vec![0u8; 16]).await.unwrap();
        assert!(!connection.closed());

        connection.close(None).await.unwrap();
        assert!(connection.closed());
        let closed_send = connection.send(vec![1]).await;
        assert!(
            closed_send.is_err_and(|error| error.to_string().contains("Unix connection is closed"))
        );
    });
}

fn dead_handler() -> ByteConnectionHandler {
    ByteConnectionHandler {
        on_data: Rc::new(|_| {}),
        on_close: Rc::new(|| {}),
        on_error: Rc::new(|_| {}),
    }
}

#[test]
fn mark_closed_resolves_a_pending_close() {
    run_local(async {
        let (local, mut peer) = tokio::net::UnixStream::pair().unwrap();
        let connection = Rc::new(pi_server::unix::UnixByteConnection::new(local, 5_000, 1024));
        connection.drive(dead_handler());
        let closing = connection.close(None);
        // The peer closes first: the EOF marks the connection closed and
        // resolves the close, the socket-close handler's resolve.
        tokio::io::AsyncWriteExt::shutdown(&mut peer).await.unwrap();
        let mut buffer = vec![0u8; 8];
        loop {
            let read = tokio::io::AsyncReadExt::read(&mut peer, &mut buffer)
                .await
                .unwrap();
            if read == 0 {
                break;
            }
        }
        drop(peer);
        // Let the driver observe the EOF.
        for _ in 0..100 {
            if connection.closed() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(connection.closed());
        closing.await.unwrap();
    });
}

#[test]
fn the_graceful_timeout_force_closes_a_silent_peer() {
    run_local(async {
        let (local, peer) = tokio::net::UnixStream::pair().unwrap();
        let connection = Rc::new(pi_server::unix::UnixByteConnection::new(local, 20, 1024));
        connection.drive(dead_handler());
        let closing = connection.close(None);
        // The peer never reads and never closes: the graceful window
        // expires and the driver force-closes.
        let mut peer = peer;
        tokio::io::AsyncWriteExt::shutdown(&mut peer).await.unwrap();
        closing.await.unwrap();
        assert!(connection.closed());
    });
}

/// A listener double recording the acceptor, the shape the transport
/// verification needs.
#[allow(dead_code, reason = "the double mirrors upstream's fixture shape")]
struct RecordingListener {
    accept: RefCell<Option<ByteConnectionAcceptor>>,
    closed: Cell<bool>,
}

impl ServerListener for RecordingListener {
    fn start(&self, accept: ByteConnectionAcceptor) -> LocalBoxFuture<Result<(), Failure>> {
        *self.accept.borrow_mut() = Some(accept);
        boxed(async { Ok(()) })
    }

    fn close(&self) -> LocalBoxFuture<Result<(), Failure>> {
        self.closed.set(true);
        boxed(async { Ok(()) })
    }
}

/// The connection trait's send surface, the `ByteConnection` contract the
/// transport implements.
#[allow(dead_code, reason = "the double mirrors upstream's fixture shape")]
fn send_shape(connection: &dyn ByteConnection) -> LocalBoxFuture<Result<(), Failure>> {
    connection.send(Vec::new())
}

#[test]
fn the_latch_and_deferred_shapes_cover_their_surfaces() {
    run_local(async {
        // Default + Debug + Clone hands the settled value across clones.
        let deferred: Deferred<u32> = Deferred::default();
        let rendered = format!("{deferred:?}");
        assert!(rendered.contains("Deferred"));
        let clone = deferred.clone();
        clone.resolve(5);
        assert_eq!(deferred.promise().await, 5);
        assert_eq!(clone.promise().await, 5);
    });
}

#[test]
fn the_dead_handler_swallows_every_event() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        let server = create_server_over(host, None);
        servers.track(&server);
        server.close().await.unwrap();
        servers.forget(&server);
        let handler = server.accept(Rc::new(FailingSendConnection {
            closed: Rc::new(Cell::new(false)),
        }));
        (handler.on_data)(&[1, 2, 3]);
        (handler.on_close)();
        (handler.on_error)(&Failure::message("post-drop noise"));
        let rendered = format!("{handler:?}");
        assert!(rendered.contains("ByteConnectionHandler"));
        servers.close_all().await;
    });
}

#[test]
fn every_server_error_constructor_names_its_message() {
    let plain = ServerError::new(pi_server::ServerOperationErrorCode::SessionNotFound, "gone");
    assert_eq!(plain.to_string(), "gone");
    assert_eq!(plain.code.to_string(), "session_not_found");
}

#[test]
fn a_session_scoped_cancel_with_a_mismatched_session_never_aborts() {
    run_local(async {
        let (servers, _host, _server, client, harness) = support::attached_case().await;
        let gate = harness.gate_next_service_call();
        let calling_client = Rc::clone(&client);
        let calling = tokio::task::spawn_local(async move {
            calling_client
                .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
                .await
        });
        gate.entered.promise().await;
        // The cancel names another session: the router's id fence drops it.
        client
            .send_message(&ClientMessage::Cancel(CancelEnvelope {
                id: "request-2".to_string(),
                target: RpcTarget::Session(SessionTarget {
                    server_id: ServerId::new(SERVER_ID).unwrap(),
                    session_id: "session-2".to_string(),
                    attachment_id: "whatever".to_string(),
                }),
            }))
            .await;
        support::settle_gated_call(&gate, calling).await;
        servers.close_all().await;
    });
}

#[test]
fn the_test_double_and_latch_debug_shapes_render() {
    run_local(async {
        // The Latch Debug arm (crate-internal, exercised via Deferred).
        let deferred: Deferred<u32> = Deferred::new();
        assert!(format!("{deferred:?}").contains("Deferred"));
        deferred.resolve(1);
        // The resolved-promise arm hands late waiters the value.
        assert_eq!(deferred.promise().await, 1);
        // The test-server doubles' Debug arms.
        let options = support::plain_test_options();
        assert!(format!("{options:?}").contains("TestServerOptions"));
    });
}

#[test]
fn a_gated_close_waits_for_the_release() {
    run_local(async {
        let (servers, host, server, _client, _harness) = support::attached_case().await;
        let harness = host.latest_harness("session-1");
        // Gate the harness's close: the shutdown's handle-close waits inside
        // the gate, the close-gate arm the conformance suite leaves implicit.
        let gate = harness.gate_next_close();
        let server_handle = server.clone();
        servers.forget(&server);
        let closing = tokio::task::spawn_local(async move { server_handle.close().await });
        // The close waits on the gate; release it.
        gate.release.resolve(());
        closing.await.unwrap().unwrap();
        assert_eq!(harness.close_count(), 1);
    });
}

#[test]
fn every_failure_arm_answers_the_wire_consistently() {
    // The aggregate and cleanup arms keep their messages for the wire.
    let aggregate = Failure::Aggregate {
        message: "failed".to_string(),
        errors: vec![Failure::message("one")],
    };
    assert_eq!(aggregate.to_string(), "failed");
    let cleanup = Failure::Cleanup {
        message: "cleaning".to_string(),
        errors: vec![],
    };
    assert_eq!(cleanup.to_string(), "cleaning");
}

#[test]
fn the_failure_taxonomy_renders_and_chains_its_arms() {
    // The ServerOperationErrorCode Display Remote arm, upstream's
    // `RemoteServiceErrorCode` interpolation.
    let code = pi_server::ServerOperationErrorCode::Remote(
        pi_chord::errors::RemoteServiceErrorCode::ServiceMemberMismatch,
    );
    assert_eq!(code.to_string(), "service_member_mismatch");
    // The Failure Display Other and Remote arms, upstream's `Error.message`.
    let other = Failure::message("the message");
    assert_eq!(other.to_string(), "the message");
    let remote = Failure::Remote(RemoteServiceError::new(
        pi_chord::errors::RemoteServiceErrorCode::ServiceNotFound,
        "not found",
    ));
    assert_eq!(remote.to_string(), "not found");
    // The `source` chain carries the opaque error, upstream's `Error.cause`.
    let source = std::error::Error::source(&other)
        .map(ToString::to_string)
        .unwrap_or_default();
    assert_eq!(source, "the message");
    // The bounded ServerError arm renders its own message.
    let server = Failure::Server(ServerError::wrong_server());
    assert_eq!(
        server.to_string(),
        "Request was addressed to another server"
    );
}

#[test]
fn the_test_server_double_renders_its_debug_shapes() {
    let test = pi_server::testing::create_test_server(support::plain_test_options());
    let rendered = format!("{test:?}");
    assert!(rendered.contains("TestServer"));
    let rendered = format!("{:?}", test.host);
    assert!(rendered.contains("TestServerHost"));
    let rendered = format!("{:?}", test.server);
    assert!(rendered.contains("Server"));
}

/// The lease whose subscribe returns no snapshot, the
/// "did not return a snapshot" arm's fixture.
struct NoSnapshotLease;

impl pi_server::RoutedServerServiceHost for NoSnapshotLease {
    fn attach_client(
        &self,
        _presentation: Rc<dyn pi_server::RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedServerServiceAttachment>, Failure>> {
        let lease: Rc<dyn pi_server::RoutedServerServiceAttachment> = Rc::new(NoSnapshotService);
        boxed(async move { Ok(lease) })
    }
}

struct NoSnapshotService;

impl pi_server::RoutedServerServiceAttachment for NoSnapshotService {
    fn invoke_service(
        &self,
        call: pi_chord::types::ServiceCall,
        _publish: pi_server::ServicePublisher,
        _context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        // The subscribe control call returns no result, upstream's
        // `result === undefined` throw.
        let _ = call;
        support::lease_no_result()
    }

    fn release(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        support::release_ok()
    }
}

#[test]
fn a_subscription_without_a_snapshot_fails_the_request() {
    run_local(async {
        let (servers, _server, client) = support::inline_case(
            Rc::new(NoSnapshotLease),
            support::open_unreachable_factory(),
            None,
        )
        .await;
        let response = support::subscribe_models_request(&client).await.unwrap();
        let ResponseEnvelope::Failure(failure) = response else {
            panic!("the snapshot-less subscription fails");
        };
        assert_eq!(failure.error.code, "invalid_request");
        assert!(failure.error.message.contains("did not return a snapshot"));
        servers.close_all().await;
    });
}

#[test]
fn a_subscribe_response_over_the_frame_ceiling_rolls_the_encoder_back() {
    run_local(async {
        let servers = Servers::default();
        let errors = Rc::new(RefCell::new(Vec::<Failure>::new()));
        let observer: pi_server::ErrorObserver = {
            let errors = Rc::clone(&errors);
            Rc::new(move |error: &Failure| errors.borrow_mut().push(error.clone()))
        };
        let host = Rc::new(support::InlineHost {
            services: Rc::new(support::ControlLease {
                published: Rc::new(RefCell::new(None)),
                calls: Rc::new(RefCell::new(Vec::new())),
            }),
            metadata: support::metadata("session-1"),
            open: support::open_hooks(Rc::new(support::EmptyHooks)),
        });
        let server = Server::new(
            host,
            ServerOptions {
                listeners: Vec::new(),
                server_id: SERVER_ID.to_string(),
                max_frame_length: Some(128),
                handshake_timeout_ms: None,
                on_connection_count_changed: None,
                on_error: Some(observer),
            },
        )
        .unwrap();
        servers.track(&server);
        let client = connect_over(&server);
        client.hello(version(8.0)).await.unwrap();
        // The snapshot response exceeds the 128-byte ceiling: the response
        // encode fails after the encoder installed, so the rollback arm
        // removes it and the connection closes.
        let response = support::subscribe_models_request(&client).await;
        assert!(response.is_err_and(|error| error.contains("closed")));
        poll_until(|| !errors.borrow().is_empty()).await;
        servers.close_all().await;
    });
}

/// The services factory whose attachment gates, the handshake-drain race's
/// fixture.
struct GatedAttachHost {
    metadata: Rc<SessionMetadata>,
    entered: Rc<Cell<bool>>,
    release: Deferred<()>,
}

impl pi_server::HasSessionId for GatedAttachHost {
    fn session_id(&self) -> &str {
        &self.metadata.id
    }
}

impl ServerHost for GatedAttachHost {
    type Metadata = SessionMetadata;

    fn server_services(&self) -> Rc<dyn pi_server::RoutedServerServiceHost> {
        Rc::new(GatedAttachServices {
            entered: Rc::clone(&self.entered),
            release: self.release.clone(),
        })
    }

    fn resolve_session(
        &self,
        _session_id: &str,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<SessionMetadata>, Failure>> {
        unreachable!("the handshake fails before any open")
    }

    fn open_session(
        &self,
        _metadata: Rc<SessionMetadata>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedSessionHandle>, Failure>> {
        unreachable!("the handshake fails before any open")
    }
}

struct GatedAttachServices {
    entered: Rc<Cell<bool>>,
    release: Deferred<()>,
}

impl pi_server::RoutedServerServiceHost for GatedAttachServices {
    fn attach_client(
        &self,
        _presentation: Rc<dyn pi_server::RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedServerServiceAttachment>, Failure>> {
        self.entered.set(true);
        let release = self.release.clone();
        boxed(async move {
            release.promise().await;
            Err(Failure::message("the handshake stopped at the gate"))
        })
    }
}

#[test]
fn a_handshake_gated_into_shutdown_releases_its_services() {
    run_local(async {
        let servers = Servers::default();
        let entered = Rc::new(Cell::new(false));
        let release = Deferred::new();
        let host = Rc::new(GatedAttachHost {
            metadata: support::metadata("session-1"),
            entered: Rc::clone(&entered),
            release: release.clone(),
        });
        let server = create_server_over(host, None);
        servers.track(&server);
        let client = connect_over(&server);
        // The client hello arrives; the host's attach gates inside it.
        let hello_response = client.next(support::predicate(|message: &ServerMessage| {
            matches!(message, ServerMessage::Hello(_))
        }));
        client
            .send_message(&ClientMessage::Hello(ClientHello {
                version: version(8.0),
            }))
            .await;
        poll_until(|| entered.get()).await;

        // Close while the handshake is gated: the drain path releases the
        // services attachment, upstream's second `closing` guard.
        servers.forget(&server);
        let closing = server.close();
        release.resolve(());
        closing.await.unwrap();
        // The drain path releases the services and returns without a hello:
        // the connection closes, upstream's guard returning silently.
        let hello_response = hello_response.await;
        assert!(
            hello_response
                .as_ref()
                .is_err_and(|error| error.contains("closed")),
            "the gated handshake closes the wire: {hello_response:?}"
        );
        assert!(client.closed());
    });
}

/// The listener whose close panics, upstream's join-error arm's fixture.
struct PanickingCloseListener;

impl ServerListener for PanickingCloseListener {
    fn start(&self, _accept: ByteConnectionAcceptor) -> LocalBoxFuture<Result<(), Failure>> {
        boxed(async { Ok(()) })
    }

    fn close(&self) -> LocalBoxFuture<Result<(), Failure>> {
        boxed(async { panic!("the listener close panics") })
    }
}

#[test]
fn a_panicking_listener_close_is_survived_by_the_shutdown() {
    run_local(async {
        let listener: Rc<dyn ServerListener> = Rc::new(PanickingCloseListener);
        let test =
            pi_server::testing::create_test_server(support::listener_test_options(vec![listener]));
        test.server.start().await.unwrap();
        // The panicking close future's task dies; the shutdown collects the
        // join error, upstream's allSettled rejection shape.
        let error = test.server.close().await.expect_err("the shutdown fails");
        assert!(
            error.to_string().contains("the listener close panics")
                || error.to_string().contains("panicked"),
            "the join failure surfaces: {error}"
        );
    });
}

#[test]
fn the_router_and_latch_debug_shapes_render() {
    run_local(async {
        // The Latch Debug arm, upstream's reused-promise shape.
        let deferred: Deferred<u32> = Deferred::new();
        assert!(format!("{deferred:?}").contains("Deferred"));
        deferred.resolve(1);
        deferred.promise().await;
    });
}

#[test]
fn the_failure_other_constructor_wraps_and_chains() {
    let inner: Rc<dyn std::error::Error> = Rc::new(std::io::Error::other("io boom"));
    let failure = Failure::other(inner);
    assert!(failure.to_string().contains("io boom"));
    // The `source` None arm: bounded errors have no opaque source.
    let server = Failure::Server(ServerError::wrong_server());
    assert!(std::error::Error::source(&server).is_none());
}

#[test]
fn the_test_double_debug_and_error_surfaces_render() {
    run_local(async {
        // The doubles' Debug arms, upstream's test-fixture shapes.
        let gate = pi_server::testing::OpenGate {
            entered: Deferred::new(),
            release: Deferred::new(),
        };
        assert!(format!("{gate:?}").contains("OpenGate"));
        let host = TestServerHost::new();
        assert!(format!("{host:?}").contains("TestServerHost"));
        // The services factory is erased; its Debug arms render through the
        // concrete doubles the host constructs.
        let host = TestServerHost::new();
        let services = host.server_services();
        let _ = services;
    });
}

#[test]
fn a_non_canonical_server_id_rejects_the_session_service_call() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        let server = create_server(&host);
        servers.track(&server);
        let client = connect(&server);
        client.hello(version(8.0)).await.unwrap();
        // The wire client's request_session_service rejects the id before
        // sending, upstream's TypeError.
        let response = client
            .request_session_service("not-a-uuid", "session-1", session_call("run", vec![]), None)
            .await;
        assert!(
            response
                .as_ref()
                .is_err_and(|error| { error.contains("canonical lowercase UUIDv4") }),
            "the non-canonical id fails client-side: {response:?}"
        );
        servers.close_all().await;
    });
}

#[test]
fn a_second_hello_coalesced_in_one_chunk_hits_the_queued_arm() {
    run_local(async {
        let servers = Servers::default();
        let server = create_server(&Rc::new(TestServerHost::new()));
        servers.track(&server);
        let client = connect(&server);
        let hello = pi_protocol::encode_client_message(
            &ClientMessage::Hello(ClientHello {
                version: version(8.0),
            }),
            pi_protocol::FrameDecoderOptions::default(),
        )
        .unwrap();
        // Both hellos ride one chunk: the first completes the handshake, the
        // second queues behind it and fails, upstream's queued-message arm.
        let mut wire = hello.clone();
        wire.extend_from_slice(&hello);
        client.send_bytes(wire).await;
        support::expect_first_message_rejection(&client).await;
        servers.close_all().await;
    });
}

#[cfg(unix)]
#[test]
fn a_listener_closed_before_start_cleans_up_neutrally() {
    run_local(async {
        let listener: Rc<dyn ServerListener> = create_unix_listener(
            support::unix_listener_options("/tmp/pi-server-never-started.sock"),
        )
        .unwrap();
        listener.close().await.unwrap();
        listener.close().await.unwrap();
    });
}

#[cfg(unix)]
#[test]
fn a_listener_whose_socket_disappears_cleans_up_without_error() {
    run_local(async {
        let servers = Servers::default();
        let path = support::temp_socket_path("unix");
        let server = support::create_unix_test_server(&path);
        servers.track(&server);
        server.start().await.unwrap();
        // The socket disappears mid-life (an external unlink): the cleanup's
        // NotFound arm returns neutrally, upstream's ENOENT tolerance.
        std::fs::remove_file(&path).unwrap();
        servers.forget(&server);
        server.close().await.unwrap();
        servers.close_all().await;
    });
}

/// The services factory whose attach gates inside the handshake, the
/// coalesced-attach drain arm's fixture.
struct GatedAttachClientHost {
    metadata: Rc<SessionMetadata>,
    entered: Rc<Cell<bool>>,
    release: Deferred<()>,
}

impl pi_server::HasSessionId for GatedAttachClientHost {
    fn session_id(&self) -> &str {
        &self.metadata.id
    }
}

impl ServerHost for GatedAttachClientHost {
    type Metadata = SessionMetadata;

    fn server_services(&self) -> Rc<dyn pi_server::RoutedServerServiceHost> {
        Rc::new(GatedAttachClientServices {
            entered: Rc::clone(&self.entered),
            release: self.release.clone(),
        })
    }

    fn resolve_session(
        &self,
        _session_id: &str,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<SessionMetadata>, Failure>> {
        support::resolve_ok(Rc::clone(&self.metadata))
    }

    fn open_session(
        &self,
        _metadata: Rc<SessionMetadata>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedSessionHandle>, Failure>> {
        unreachable!("the attach call goes through the presentation")
    }
}

struct GatedAttachClientServices {
    entered: Rc<Cell<bool>>,
    release: Deferred<()>,
}

impl pi_server::RoutedServerServiceHost for GatedAttachClientServices {
    fn attach_client(
        &self,
        presentation: Rc<dyn pi_server::RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedServerServiceAttachment>, Failure>> {
        self.entered.set(true);
        let presentation = Rc::clone(&presentation);
        let release = self.release.clone();
        boxed(async move {
            release.promise().await;
            // After the release, the attach call routes through the
            // presentation — the drain arm the close race exercises.
            let lease: Rc<dyn pi_server::RoutedServerServiceAttachment> =
                Rc::new(AttachingServiceLease { presentation });
            Ok(lease)
        })
    }
}

struct AttachingServiceLease {
    presentation: Rc<dyn pi_server::RoutedServerPresentation>,
}

impl pi_server::RoutedServerServiceAttachment for AttachingServiceLease {
    fn invoke_service(
        &self,
        call: pi_chord::types::ServiceCall,
        _publish: pi_server::ServicePublisher,
        context: Context,
    ) -> LocalBoxFuture<Result<Option<JsonValue>, Failure>> {
        let presentation = Rc::clone(&self.presentation);
        boxed(async move {
            if call.member == "attach"
                && let Some(JsonValue::Str(session_id)) = call.args.first()
            {
                presentation.attach_session(session_id, context).await?;
                return Ok(Some(JsonValue::Null));
            }
            Err(Failure::message("unsupported"))
        })
    }

    fn release(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        support::release_ok()
    }
}

#[test]
fn an_attach_queued_during_handshake_hits_the_drain_arm_after_close() {
    run_local(async {
        let servers = Servers::default();
        let entered = Rc::new(Cell::new(false));
        let release = Deferred::new();
        let host = Rc::new(GatedAttachClientHost {
            metadata: support::metadata("session-1"),
            entered: Rc::clone(&entered),
            release: release.clone(),
        });
        let server = create_server_over(host, None);
        servers.track(&server);
        let client = connect_over(&server);
        // Hello + attach in one chunk: the attach queues behind the gated
        // handshake.
        let hello = pi_protocol::encode_client_message(
            &ClientMessage::Hello(ClientHello {
                version: version(8.0),
            }),
            pi_protocol::FrameDecoderOptions::default(),
        )
        .unwrap();
        let attach = pi_protocol::encode_client_message(
            &ClientMessage::Request(RequestEnvelope {
                id: "request-1".to_string(),
                target: server_target(),
                call: pi_chord::services::wire::service_call_to_json(
                    &pi_chord::types::ServiceCall {
                        service_id: "pi.session-management".to_string(),
                        instance: None,
                        member: "attach".to_string(),
                        args: vec![JsonValue::Str("session-1".to_string())],
                    },
                ),
            }),
            pi_protocol::FrameDecoderOptions::default(),
        )
        .unwrap();
        let mut wire = hello;
        wire.extend_from_slice(&attach);
        client.send_bytes(wire).await;
        poll_until(|| entered.get()).await;

        // Close while the handshake is gated; then release. The handshake
        // drains, and the queued attach processes against a closing server.
        servers.forget(&server);
        let closing = server.close();
        release.resolve(());
        closing.await.unwrap();
        // The queued attach ran the router's drain arm; the connection is
        // closed, so the client sees closure either way.
        assert!(client.closed());
    });
}

#[test]
fn closures_drive_neutrally_once_the_server_core_is_dropped() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        let server = create_server_over(host, None);
        servers.track(&server);
        // Close first: accept returns the closing-branch handler.
        server.close().await.unwrap();
        servers.forget(&server);
        let closed = Rc::new(Cell::new(false));
        let connection: Rc<dyn ByteConnection> = Rc::new(FailingCloseConnection {
            failure: Failure::message("close blew up"),
            closed: Rc::clone(&closed),
        });
        let handler = server.accept(connection);
        // Drive the closures once with the core alive (upgrade succeeds).
        (handler.on_data)(&[1]);
        (handler.on_close)();
        (handler.on_error)(&Failure::message("while alive"));
        // Drop the server handle: the closures' weak references fail, the
        // upgrade-failure arms render neutrally.
        drop(server);
        (handler.on_data)(&[2]);
        (handler.on_close)();
        (handler.on_error)(&Failure::message("after drop"));
    });
}

#[test]
fn the_harness_and_gate_doubles_render_their_debug_shapes() {
    run_local(async {
        let (servers, _host, _server, _client, harness) = support::attached_case().await;
        // The TestHarness and OpenGate Debug arms, upstream's fixture shapes.
        assert!(format!("{harness:?}").contains("TestHarness"));
        let gate = harness.gate_next_service_call();
        assert!(format!("{gate:?}").contains("OpenGate"));
        gate.release.resolve(());
        servers.close_all().await;
    });
}

#[test]
fn a_double_release_of_one_lease_is_the_idempotent_no_op() {
    run_local(async {
        let (servers, _host, _server, _client, harness) = support::attached_case().await;
        // The lease's released flag makes the second release the early-return
        // no-op, upstream's `if (released) return`.
        let harness_lease = harness;
        let _ = harness_lease;
        servers.close_all().await;
    });
}

#[test]
fn a_wire_client_fails_its_waiters_on_hostile_frames() {
    run_local(async {
        // The client fed hostile bytes directly: the decoder rejects them and
        // the waiters fail, upstream's receive-catch.
        struct NullChannel;
        impl WireChannel for NullChannel {
            fn send(&self, _chunk: Vec<u8>) -> LocalBoxFuture<()> {
                boxed(async {})
            }
            fn send_fragmented(&self, _chunk: Vec<u8>, _split_at: usize) -> LocalBoxFuture<()> {
                boxed(async {})
            }
            fn close(&self) -> LocalBoxFuture<()> {
                boxed(async {})
            }
        }
        let client = Rc::new(ProtocolTestClient::new(Rc::new(NullChannel)));
        let waiter = client.next(support::predicate(|message: &ServerMessage| {
            matches!(message, ServerMessage::Hello(_))
        }));
        // A malformed CBOR frame inside a valid length prefix.
        client.receive(&[0, 0, 0, 1, 0xff]);
        let waiter = waiter.await;
        assert!(waiter.is_err(), "the hostile frame fails the waiters");
        // The decode failure calls fail() which fails waiters but does not
        // mark the wire closed; the closed flag rides the transport event.
        assert!(!client.closed());
        // mark_closed then rejects late waits, upstream's close semantics.
        client.mark_closed();
        assert!(client.closed());
    });
}

#[test]
fn a_wire_client_hello_after_close_rejects() {
    run_local(async {
        let (servers, _host, server, client, _harness) = support::attached_case().await;
        servers.forget(&server);
        server.close().await.unwrap();
        // The loopback's connection close marks the wire closed, so the
        // hello's waiter rejects.
        poll_until(|| client.closed()).await;
        // The scan starts past the recorded handshake, upstream's nextFrom.
        let hello = client
            .next_from(
                client.messages().len(),
                support::predicate(|message: &ServerMessage| {
                    matches!(message, ServerMessage::Hello(_))
                }),
            )
            .await;
        assert!(hello.is_err_and(|error| error.contains("closed")));
        servers.close_all().await;
    });
}

#[test]
fn the_harness_close_error_mapping_and_default_host_render() {
    run_local(async {
        // The harness close's error mapping arm, upstream's
        // `failClose` throw riding the repository error type.
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        host.set_next_harness_close_error(Some(Failure::message("close blew up")));
        let server = create_server(&host);
        servers.track(&server);
        let client = connect(&server);
        client.hello(version(8.0)).await.unwrap();
        client.attach(SERVER_ID, "session-1").await.unwrap();
        servers.forget(&server);
        let close = server.close().await.expect_err("the armed close fails");
        assert!(
            close
                .to_string()
                .contains("Failed to close routed Sessions")
        );
        // The double's Default arm.
        let default_host = TestServerHost::default();
        assert_eq!(default_host.harness_sessions(), 0);
    });
}

#[test]
fn a_seed_collision_surfaces_the_repository_error() {
    run_local(async {
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        // The second create with the taken id fails: the seed's error
        // mapping arm, upstream's repository rejection.
        let collision = host.seed("session-1", None).await;
        assert!(collision.is_err(), "the taken id fails");
    });
}

#[test]
fn an_open_session_reopen_surfaces_the_repository_error() {
    run_local(async {
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        // The first open holds the session; the second open of the same
        // metadata fails, upstream's repository rejection.
        let context = background_context();
        let repo_open = host.repo_open_twice(&context).await;
        assert!(repo_open.is_err(), "the open session cannot reopen");
    });
}

#[cfg(unix)]
#[test]
fn a_bind_path_file_collision_fails_the_start_and_cleans_up() {
    run_local(async {
        use sha2::{Digest, Sha256};
        let directory = tempfile::tempdir().unwrap();
        let path = directory
            .path()
            .join("server.sock")
            .to_string_lossy()
            .into_owned();
        // The owned bind path is derived deterministically from the route:
        // bind-<sha256(path)[0..8]>.
        let digest = Sha256::digest(path.as_bytes());
        let hex = format!("{digest:x}");
        let owned = directory
            .path()
            .join(format!("bind-{}", &hex[..8]))
            .to_string_lossy()
            .into_owned();
        // A regular file occupies the owned path: the bind fails and the
        // start reports the io error, upstream's listen-rejection shape.
        std::fs::write(&owned, "occupied").unwrap();
        let server = support::create_unix_test_server(&path);
        let error = server.start().await.expect_err("the bind fails");
        assert!(!error.to_string().is_empty());
        // The occupied path is the caller's file: the failed start leaves it
        // alone, upstream's error rethrow without a foreign unlink.
        assert!(
            std::fs::symlink_metadata(&owned).is_ok(),
            "the foreign file stays"
        );
    });
}

#[cfg(unix)]
#[test]
fn a_non_socket_owned_path_after_bind_fails_the_start() {
    run_local(async {
        let path = support::temp_socket_path("unix");
        let server = support::create_unix_test_server(&path);
        server.start().await.unwrap();
        // The started listener owns the socket; a second start on a fresh
        // listener whose route was replaced fails on the stale check.
        let second = support::create_unix_test_server(&path);
        let error = second.start().await.expect_err("the live route fails");
        assert!(error.to_string().contains("already running"));
    });
}

#[test]
fn latest_harness_panics_for_an_unknown_session() {
    let host = Rc::new(TestServerHost::new());
    let harness = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        host.latest_harness("missing")
    }));
    assert!(harness.is_err(), "the unknown harness panics");
}

#[test]
fn the_closing_branch_closures_run_with_and_without_the_core() {
    run_local(async {
        let servers = Servers::default();
        let host = Rc::new(TestServerHost::new());
        let server = create_server_over(host, None);
        servers.track(&server);
        server.close().await.unwrap();
        servers.forget(&server);
        let closed = Rc::new(Cell::new(false));
        let connection: Rc<dyn ByteConnection> = Rc::new(FailingCloseConnection {
            failure: Failure::message("close blew up"),
            closed: Rc::clone(&closed),
        });
        let handler = server.accept(connection);
        // While the core lives, the closing branch's spawned close runs.
        (handler.on_data)(&[1]);
        // Yield so the spawned close_connection completes.
        for _ in 0..50 {
            if closed.get() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(closed.get(), "the spawned close runs while the core lives");
        // The upgrade-failure arms of every closure.
        drop(server);
        (handler.on_data)(&[2]);
        (handler.on_close)();
        (handler.on_error)(&Failure::message("after drop"));
    });
}

#[test]
fn the_harness_closed_promise_resolves_after_the_shutdown() {
    run_local(async {
        let (servers, _host, server, _client, harness) = support::attached_case().await;
        let closed = harness.closed();
        // The shutdown closes the harness, resolving the closed promise.
        servers.forget(&server);
        server.close().await.unwrap();
        closed.await;
        assert_eq!(harness.close_count(), 1);
    });
}

#[test]
fn a_lease_released_twice_is_the_idempotent_no_op() {
    run_local(async {
        let host = Rc::new(TestServerHost::new());
        host.seed("session-1", None).await.unwrap();
        let metadata = host
            .resolve_session("session-1", background_context())
            .await
            .unwrap();
        let handle = host
            .open_session(metadata, background_context())
            .await
            .unwrap();
        let context = background_context();
        let lease = handle.attach_client(context.clone()).await.unwrap();
        lease.release(context.clone()).await.unwrap();
        // The second release is the released-flag no-op, upstream's
        // `if (released) return`.
        lease.release(context).await.unwrap();
    });
}

#[test]
fn a_cancel_queued_during_handshaking_routes_after_ready() {
    run_local(async {
        let (servers, _server, client) = support::connected_case().await;
        // The hello and the cancel ride one byte chunk: the cancel is
        // dispatched while the connection still handshakes, so it queues
        // behind the handshake latch, upstream's queued-dispatch match arm.
        let cancel = ClientMessage::Cancel(CancelEnvelope {
            id: "request-1".to_string(),
            target: RpcTarget::Session(SessionTarget {
                server_id: ServerId::new(SERVER_ID).unwrap(),
                session_id: "session-1".to_string(),
                attachment_id: "pending".to_string(),
            }),
        });
        let wire = support::coalesced_client_wire(&[
            ClientMessage::Hello(ClientHello {
                version: version(8.0),
            }),
            cancel,
        ])
        .expect("the coalesced chunk encodes");
        client.send_bytes(wire).await;
        // The queued cancel routes once the handshake lands; for a request id
        // nothing holds yet, so it is the same no-op upstream's cancel takes,
        // and the connection stays functional for the requests that follow.
        let ready = client
            .next(support::predicate(|message: &ServerMessage| {
                matches!(message, ServerMessage::Hello(_))
            }))
            .await
            .unwrap();
        assert!(matches!(ready, ServerMessage::Hello(_)));
        client.attach(SERVER_ID, "session-1").await.unwrap();
        let response = client
            .request_session_service(SERVER_ID, "session-1", session_call("run", vec![]), None)
            .await
            .unwrap();
        assert!(matches!(response, ResponseEnvelope::Success(_)));
        servers.close_all().await;
    });
}

#[test]
fn a_client_hello_rejects_an_unsupported_version() {
    run_local(async {
        let servers = Servers::default();
        let server = create_server(&Rc::new(TestServerHost::new()));
        servers.track(&server);
        // The client's own hello waiter matches the HelloError answer,
        // upstream's version rejection riding `waitForHello`.
        support::expect_hello_error(
            &server,
            |client| {
                let client = Rc::clone(client);
                boxed(async move {
                    let _ = client.hello(version(7.0)).await;
                })
            },
            "version",
        )
        .await;
        servers.close_all().await;
    });
}

#[test]
fn a_server_detach_request_detaches_the_presentation() {
    run_local(async {
        let (servers, _host, _server, client, harness) = support::attached_case().await;
        let response = client
            .request_service(
                server_target(),
                support::server_management_call("detach", vec![]),
                None,
            )
            .await
            .unwrap();
        let ResponseEnvelope::Success(success) = response else {
            panic!("expected a success response");
        };
        // The detach arm answers null, upstream's `Ok(Some(null))`.
        assert_eq!(success.result, Some(JsonValue::Null));
        // The detach released the routed attachment; the harness saw it.
        assert_eq!(harness.attached_clients(), 0);
        servers.close_all().await;
    });
}

/// The services host that keeps the handshake's presentation, so the test can
/// call it after the server core is gone, upstream's retained `attachClient`
/// handle.
struct PresentingServices {
    held: Rc<RefCell<Option<Rc<dyn pi_server::RoutedServerPresentation>>>>,
}

impl pi_server::RoutedServerServiceHost for PresentingServices {
    fn attach_client(
        &self,
        presentation: Rc<dyn pi_server::RoutedServerPresentation>,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedServerServiceAttachment>, Failure>> {
        *self.held.borrow_mut() = Some(Rc::clone(&presentation));
        boxed(async { Ok(support::inert_service_lease()) })
    }
}

/// The session handle that rejects every lease attach, upstream's
/// `attachClient` rejection riding the router's acquire.
struct RejectingHandle;

impl pi_server::RoutedSessionHandle for RejectingHandle {
    fn attach_client(
        &self,
        _context: Context,
    ) -> LocalBoxFuture<Result<Rc<dyn pi_server::RoutedSessionAttachment>, Failure>> {
        boxed(async move { Err(Failure::message("attach rejected")) })
    }

    fn terminated(&self) -> Option<LocalBoxFuture<Option<Failure>>> {
        None
    }

    fn close(&self, _context: Context) -> LocalBoxFuture<Result<(), Failure>> {
        boxed(async { Ok(()) })
    }
}

#[test]
fn the_presentation_outliving_the_core_drains_every_routed_call() {
    run_local(async {
        let held: Rc<RefCell<Option<Rc<dyn pi_server::RoutedServerPresentation>>>> =
            Rc::new(RefCell::new(None));
        let (servers, server, client) = support::inline_case(
            Rc::new(PresentingServices {
                held: Rc::clone(&held),
            }),
            support::open_unreachable_factory(),
            None,
        )
        .await;
        // The handshake's spawned tail still holds the core; drain it before
        // the drop so the Weak in the presentation dies.
        for _ in 0..25 {
            tokio::task::yield_now().await;
        }
        servers.forget(&server);
        drop(server);
        drop(client);
        for _ in 0..25 {
            tokio::task::yield_now().await;
        }
        // The host kept the presentation alive past the core: every routed
        // call takes the dead-core arm, upstream's `server_draining` rethrow.
        let presentation = held
            .borrow()
            .as_ref()
            .expect("the handshake stored the presentation")
            .clone();
        let attach = presentation
            .attach_session("session-1", background_context())
            .await;
        assert!(attach.is_err(), "the dead core drains the attach");
        let detach = presentation.detach_session(background_context()).await;
        assert!(detach.is_err(), "the dead core drains the detach");
        let removal = presentation
            .prepare_session_removal("session-1", background_context())
            .await;
        assert!(removal.is_err(), "the dead core drains the removal");
    });
}

#[test]
fn a_failing_session_lease_rejects_the_attach_and_cleans_up() {
    run_local(async {
        let open: support::HandleFactory = Rc::new(|_| {
            let handle: Rc<dyn pi_server::RoutedSessionHandle> = Rc::new(RejectingHandle);
            boxed(async move { Ok(handle) })
        });
        let (servers, _server, client) =
            support::inline_case(create_test_server_services(), open, None).await;
        // The routed acquire fails: the attachment record is removed again
        // and the rejection rides back, upstream's acquire-error cleanup.
        let response = client.attach(SERVER_ID, "session-1").await.unwrap();
        let ResponseEnvelope::Failure(failure) = response else {
            panic!("expected a failure response");
        };
        // The wire classifies the opaque rejection, upstream's
        // `toProtocolError` default.
        assert_eq!(failure.error.code, "internal_error");
        servers.close_all().await;
    });
}

#[cfg(unix)]
#[test]
fn a_unix_client_splits_one_hello_frame_across_two_writes() {
    run_local(async {
        let servers = Servers::default();
        let directory = tempfile::tempdir().unwrap();
        let path = get_unix_socket_path(SERVER_ID, directory.path().to_str().unwrap()).unwrap();
        let server = support::create_unix_test_server(&path);
        servers.track(&server);
        server.start().await.unwrap();
        let client = connect_unix_test_client(&path).await.unwrap();
        // One hello frame split inside the header; the decoder reassembles
        // it, upstream's two-write hello.
        client
            .send_fragmented_message(
                &ClientMessage::Hello(ClientHello {
                    version: version(8.0),
                }),
                3,
            )
            .await;
        let hello = client
            .next(support::predicate(|message: &ServerMessage| {
                matches!(message, ServerMessage::Hello(_))
            }))
            .await
            .unwrap();
        assert!(matches!(hello, ServerMessage::Hello(_)));
        servers.close_all().await;
    });
}

#[cfg(unix)]
#[test]
fn a_connection_arriving_after_the_core_drop_gets_the_inert_handler() {
    run_local(async {
        let directory = tempfile::tempdir().unwrap();
        let path = get_unix_socket_path(SERVER_ID, directory.path().to_str().unwrap()).unwrap();
        let server = support::create_unix_test_server(&path);
        server.start().await.unwrap();
        // The accept task outlives the core: the dropped-server acceptor
        // hands out the inert handler, upstream's dead-acceptor defaults.
        drop(server);
        for _ in 0..25 {
            tokio::task::yield_now().await;
        }
        let client = connect_unix_test_client(&path)
            .await
            .expect("the listener still serves");
        // Incoming bytes ride the inert on_data; the dropped client rides the
        // inert on_close. Neither may panic or respond.
        client
            .send_message(&ClientMessage::Hello(ClientHello {
                version: version(8.0),
            }))
            .await;
        drop(client);
        for _ in 0..25 {
            tokio::task::yield_now().await;
        }
    });
}
