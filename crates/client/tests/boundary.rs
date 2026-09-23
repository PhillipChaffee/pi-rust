//! Boundary tests binding the branches the 1:1 ported suites leave open:
//! factory and encode failures on the connection lifecycle, listener-error
//! fan-outs, the service-transport error mapping, and the restated error
//! taxonomy. Upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

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
    clippy::indexing_slicing,
    reason = "the cases index messages the preceding waits guarantee"
)]
#![allow(
    clippy::missing_const_for_fn,
    reason = "test case bodies are deliberately free functions, not const-promotable code"
)]

mod support;

use std::cell::RefCell;
use std::rc::Rc;

use pi_chord::context::{background_context, with_cancel};
use pi_chord::errors::{ChordError, RemoteServiceErrorCode};
use pi_chord::future::boxed;
use pi_chord::types::{
    JsonValue, ServiceCall, ServiceCatalogueEntry, ServiceMode, ServiceProviderUpdate,
};
use pi_client::{
    ByteTransport, ByteTransportHandlers, Client, ClientError as ClientErrorKind, ConnectionState,
    create_client_service_transport, is_error_code, to_disconnected_error,
};
use pi_protocol::{
    AttachmentEnvelope, ClientMessage as ProtocolClientMessage, ProtocolError, ResponseEnvelope,
    RpcTarget, ServerHello, ServerId, ServerMessage,
};
use support::{
    MemoryByteServer, SERVER_ID, attach_client, client_options, client_options_with,
    connect_client, deliver_close, deliver_raw, delivering_transport, failing_factory,
    handshake_gate_factory, in_memory_factory, noop_handlers, open_default_subscription,
    open_subscription, record_attachment_sessions, record_state_changes, recording_factory,
    request_envelope, revision_ops, revision_replace_op, run_local, send_failure, send_success,
    server_frame, server_target, service_call, snapshot_result, snapshot_with_members,
    spawn_transport_invoke, spawn_transport_subscribe, state_member, state_update,
    state_wire_update,
};

fn instance_call(service_id: &str, member: &str, args: Vec<JsonValue>) -> ServiceCall {
    ServiceCall {
        service_id: service_id.to_string(),
        instance: Some(pi_chord::types::ServiceInstanceAddress {
            key: "instance-key".to_string(),
            generation: 3,
        }),
        member: member.to_string(),
        args,
    }
}

const OTHER_SERVER_ID: &str = "00000000-0000-4000-8000-000000000002";

/// The unknown-member update the undecodable-member cases deliver, once
/// post-hydration and once buffered ahead of the snapshot.
fn unknown_member_update() -> pi_protocol::ServiceEventEnvelope {
    state_wire_update(
        "service-1",
        "unknown-member",
        JsonValue::Number(1i64.into()),
        vec![],
    )
}

#[test]
fn spells_the_connection_state_names_upstream_reports() {
    assert_eq!(ConnectionState::Disconnected.as_str(), "disconnected");
    assert_eq!(ConnectionState::Connecting.as_str(), "connecting");
    assert_eq!(ConnectionState::Connected.as_str(), "connected");
}

#[test]
fn formats_debug_output_and_reports_disposal_state() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = Client::new(client_options(in_memory_factory(&server), SERVER_ID))
            .expect("the identity is canonical");
        assert_eq!(client.server_id(), SERVER_ID);
        assert!(!client.disposed());
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        let debug = format!("{client:?}");
        assert!(
            debug.contains("server_id") && debug.contains("connection_state"),
            "the client debug names its identity and lifecycle: {debug}"
        );
        let options_debug = format!(
            "{:?}",
            client_options(in_memory_factory(&server), SERVER_ID)
        );
        assert!(
            options_debug.contains("server_id") && options_debug.contains("max_frame_length"),
            "the options debug names its fields: {options_debug}"
        );

        client.dispose();
        client.dispose();
        assert!(client.disposed());
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        let reconnected = client.connect().await;
        assert_eq!(
            reconnected.expect_err("a disposed client rejects"),
            ClientErrorKind::Disposed
        );
    });
}

#[test]
fn rejects_max_frame_lengths_outside_the_encodable_range() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        for max_frame_length in [0, u32::MAX as usize + 1] {
            let error = Client::new(client_options_with(
                in_memory_factory(&server),
                SERVER_ID,
                Some(max_frame_length),
                None,
            ))
            .expect_err("an out-of-range frame bound fails");
            assert!(
                error.to_string().contains("between 1 and 4294967295"),
                "the error names the encodable range: {error}"
            );
        }
        let client = Client::new(client_options_with(
            in_memory_factory(&server),
            SERVER_ID,
            Some(u32::MAX as usize),
            None,
        ))
        .expect("the u32::MAX boundary is encodable");
        client.dispose();
    });
}

#[test]
fn fails_the_handshake_when_the_client_hello_cannot_be_encoded() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = Client::new(client_options_with(
            in_memory_factory(&server),
            SERVER_ID,
            Some(1),
            None,
        ))
        .expect("the frame bound itself is in range");
        let error = client.connect().await.expect_err("the hello cannot encode");
        assert!(
            matches!(error, ClientErrorKind::Disconnected { cause: Some(_), .. }),
            "the protocol failure wraps as disconnected: {error}"
        );
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        assert_eq!(server.client_close_count(), 1);
    });
}

#[test]
fn fails_the_connection_when_the_transport_factory_rejects() {
    run_local(async {
        let client = Client::new(client_options(failing_factory("boom"), SERVER_ID))
            .expect("the identity is canonical");
        let error = client
            .connect()
            .await
            .expect_err("a rejected factory fails");
        assert_eq!(error.to_string(), "boom");
        assert!(
            matches!(error, ClientErrorKind::Disconnected { cause: Some(_), .. }),
            "the factory failure wraps as disconnected: {error}"
        );
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn rejects_connect_attempts_while_connecting_or_connected() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = Client::new(client_options(in_memory_factory(&server), SERVER_ID))
            .expect("the identity is canonical");
        let first = client.connect();
        let error = client
            .connect()
            .await
            .expect_err("the second connect fails");
        assert_eq!(error.to_string(), "Client is already connecting");
        first.await.expect("the first handshake answers");
        let error = client.connect().await.expect_err("the reconnect fails");
        assert_eq!(error.to_string(), "Client is already connected");
        client.dispose();
    });
}

#[test]
fn fails_pending_requests_when_the_transport_send_fails() {
    run_local(async {
        let (factory, handle, _fire) = handshake_gate_factory(1, false);
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        client.connect().await.expect("the handshake answers");
        let pending = client.request(&server_target(), &service_call("test", "run", vec![]), None);
        let error = pending
            .await
            .expect_err("the send failure fails the request");
        assert_eq!(error.to_string(), "send exploded");
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        assert_eq!(handle.close_count.get(), 1);
    });
}

#[test]
fn ignores_a_send_failure_reported_after_the_client_is_dropped() {
    run_local(async {
        let (factory, handle, fire) = handshake_gate_factory(1, true);
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        client.connect().await.expect("the handshake answers");
        let pending = client.request(&server_target(), &service_call("test", "run", vec![]), None);
        drop(pending);
        drop(client);
        fire.expect("the gate arms")
            .send(())
            .expect("the gate fires");
        tokio::task::yield_now().await;
        assert!(
            handle.resolved.get(),
            "the spawned send task runs the failed send to completion after the client drops"
        );
    });
}

#[test]
fn ignores_a_send_failure_after_the_connection_already_failed() {
    run_local(async {
        let (factory, _handle, fire) = handshake_gate_factory(1, true);
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        client.connect().await.expect("the handshake answers");
        let changes = record_state_changes(&client);
        let pending = client.request(&server_target(), &service_call("test", "run", vec![]), None);
        client.disconnect();
        let error = pending.await.expect_err("the disconnect fails the request");
        assert_eq!(error.to_string(), "Client disconnected");
        fire.expect("the gate arms")
            .send(())
            .expect("the gate fires");
        tokio::task::yield_now().await;
        let recorded = changes.borrow().clone();
        let last = recorded.last().expect("the disconnect was reported");
        let error = last
            .error
            .clone()
            .expect("the disconnect carries its failure");
        assert_eq!(
            error.to_string(),
            "Client disconnected",
            "the late send failure must not re-fail the connection: {error}"
        );
    });
}

#[test]
fn passes_through_an_already_disconnected_transport_error() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let changes = record_state_changes(&client);
        server.error(&ClientErrorKind::disconnected());
        let recorded = changes.borrow().clone();
        let last = recorded.last().expect("the disconnect was reported");
        assert_eq!(last.state, ConnectionState::Disconnected);
        assert_eq!(last.error, Some(ClientErrorKind::disconnected()));
    });
}

#[test]
fn displays_and_sources_the_client_error_taxonomy() {
    let server_error = ClientErrorKind::Server(ProtocolError {
        code: "session_not_found".to_string(),
        message: "Unknown session".to_string(),
    });
    assert_eq!(server_error.to_string(), "Unknown session");
    assert!(std::error::Error::source(&server_error).is_none());
    assert_eq!(ClientErrorKind::Disposed.to_string(), "Client is disposed");

    let enoent = ClientErrorKind::from_io(&std::io::Error::from_raw_os_error(2));
    assert!(
        matches!(&enoent, ClientErrorKind::Other { code: Some(code), .. } if code == "ENOENT"),
        "the errno name is carried: {enoent}"
    );
    let epipe = ClientErrorKind::from_io(&std::io::Error::from_raw_os_error(32));
    assert!(
        matches!(&epipe, ClientErrorKind::Other { code: Some(code), .. } if code == "EPIPE"),
        "the errno name is carried: {epipe}"
    );
    // macOS and Linux number connection failures differently.
    #[cfg(target_os = "macos")]
    let (econnreset, etimedout) = (54, 60);
    #[cfg(target_os = "linux")]
    let (econnreset, etimedout) = (104, 110);
    let reset = ClientErrorKind::from_io(&std::io::Error::from_raw_os_error(econnreset));
    assert!(
        matches!(&reset, ClientErrorKind::Other { code: Some(code), .. } if code == "ECONNRESET"),
        "the errno name is carried: {reset}"
    );
    let timed_out = ClientErrorKind::from_io(&std::io::Error::from_raw_os_error(etimedout));
    assert!(
        matches!(&timed_out, ClientErrorKind::Other { code: Some(code), .. } if code == "ETIMEDOUT"),
        "the errno name is carried: {timed_out}"
    );
    let unnamed = ClientErrorKind::from_io(&std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "bad data",
    ));
    assert!(
        matches!(&unnamed, ClientErrorKind::Other { code: None, .. }),
        "an io failure without an errno carries no code: {unnamed}"
    );
    let unmapped = ClientErrorKind::from_io(&std::io::Error::from_raw_os_error(13));
    assert!(
        matches!(&unmapped, ClientErrorKind::Other { code: None, .. }),
        "an errno outside the table carries no code: {unmapped}"
    );

    let cause = ClientErrorKind::other_with_code("broken pipe", Some("EPIPE".to_string()));
    let wrapped = to_disconnected_error(&cause);
    assert_eq!(wrapped.to_string(), "broken pipe");
    let source = std::error::Error::source(&wrapped)
        .and_then(|source| source.downcast_ref::<ClientErrorKind>())
        .expect("the disconnected failure sources its cause");
    assert_eq!(*source, cause);
    assert!(is_error_code(&wrapped, "EPIPE"));
    assert!(!is_error_code(&wrapped, "ENOENT"));
    assert!(!is_error_code(&server_error, "EPIPE"));
}

#[test]
fn formats_the_byte_transport_handlers_debug_output() {
    let handlers = noop_handlers();
    assert!(
        format!("{handlers:?}").contains("ByteTransportHandlers"),
        "the handlers debug names the struct"
    );
}

#[test]
fn fetches_a_valid_service_catalogue_round_trip() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let catalogue = client.service_catalogue(&server_target(), None);
        server.wait_for_messages(2).await;
        let envelope = request_envelope(&server, 1);
        send_success(
            &server,
            &envelope.id,
            Some(JsonValue::Array(vec![pi_chord::services::wire::object(
                vec![
                    ("serviceId", JsonValue::string("pi.models")),
                    ("mode", JsonValue::string("singleton")),
                ],
            )])),
        );
        let entries = catalogue.await.expect("the catalogue validates");
        assert_eq!(
            entries,
            vec![ServiceCatalogueEntry {
                service_id: "pi.models".to_string(),
                mode: ServiceMode::Singleton,
            }]
        );
        client.dispose();
    });
}

#[test]
fn removes_the_subscription_listener_when_the_server_rejects_the_subscribe_call() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let opening = open_default_subscription(&client, None);
        server.wait_for_messages(2).await;
        send_failure(
            &server,
            "request-1",
            ProtocolError {
                code: "service_not_found".to_string(),
                message: "no such service".to_string(),
            },
        );
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("a rejected subscribe call fails");
        assert_eq!(
            error,
            ClientErrorKind::Server(ProtocolError {
                code: "service_not_found".to_string(),
                message: "no such service".to_string(),
            })
        );
        assert!(
            client.connected(),
            "a rejected call does not fail the connection"
        );
        client.dispose();
    });
}

#[test]
fn cleans_up_the_subscription_listener_when_the_request_is_disconnected_or_disposed() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        server.disconnect();
        let opening = open_default_subscription(&client, None);
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("a disconnected subscribe call fails");
        assert!(matches!(error, ClientErrorKind::Disconnected { .. }));

        client.dispose();
        let opening = open_default_subscription(&client, None);
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("a disposed subscribe call fails");
        assert_eq!(error, ClientErrorKind::Disposed);
    });
}

#[test]
fn rejects_the_subscription_when_it_is_removed_while_its_request_is_in_flight() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let opening = open_default_subscription(&client, None);
        server.wait_for_messages(2).await;
        send_success(&server, "request-1", Some(snapshot_result()));
        client.dispose();
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("a removed subscription rejects as disconnected");
        assert_eq!(error.to_string(), "Client is disconnected");
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn buffers_post_hydration_updates_until_the_subscription_starts() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let updates: Rc<RefCell<Vec<ServiceProviderUpdate>>> = Rc::new(RefCell::new(Vec::new()));
        let updates_listener = updates.clone();
        let opening = open_subscription(
            &client,
            &server_target(),
            "pi.models",
            Rc::new(move |update: &ServiceProviderUpdate| {
                updates_listener.borrow_mut().push(update.clone());
            }),
            None,
        );
        server.wait_for_messages(2).await;
        server.send(&ServerMessage::ServiceEvent(state_update(
            1,
            revision_ops(1),
        )));
        send_success(&server, "request-1", Some(snapshot_result()));
        let subscription = opening
            .await
            .expect("the subscription task ran")
            .expect("the subscription opens");
        assert_eq!(subscription.id(), "service-1");
        assert!(
            matches!(subscription.target(), RpcTarget::Server(target) if target.server_id.as_str() == SERVER_ID),
            "the subscription fences to its routed target"
        );
        let debug = format!("{subscription:?}");
        assert!(
            debug.contains("ServiceSubscription") && debug.contains("service-1"),
            "the subscription debug names the id: {debug}"
        );

        server.send(&ServerMessage::ServiceEvent(state_update(
            2,
            revision_ops(2),
        )));
        assert_eq!(updates.borrow().len(), 0);
        subscription.start();
        assert_eq!(updates.borrow().len(), 2, "both buffered updates deliver");
        subscription.start();
        assert_eq!(updates.borrow().len(), 2, "a repeat start delivers nothing");
        assert!(
            matches!(&updates.borrow()[0], ServiceProviderUpdate::State { .. }),
            "the buffered update decoded as a state publication"
        );

        let disposing = subscription.dispose();
        server.wait_for_messages(3).await;
        send_success(&server, "request-2", None);
        disposing.await.expect("the subscription closes");
        let repeat = subscription.dispose().await;
        assert!(repeat.is_ok(), "a repeat dispose is a no-op");
        subscription.start();
        assert_eq!(
            updates.borrow().len(),
            2,
            "starting a disposed subscription delivers nothing"
        );
        client.dispose();
    });
}

#[test]
fn fails_the_connection_when_a_hydrated_update_is_undecodable() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let opening = tokio::task::spawn_local(client.subscribe_service(
            &server_target(),
            "pi.models",
            ServiceMode::Singleton,
            Rc::new(|_update: &ServiceProviderUpdate| {}),
            None,
        ));
        server.wait_for_messages(2).await;
        server.send(&ServerMessage::Response(ResponseEnvelope::Success(
            pi_protocol::ResponseSuccess {
                id: "request-1".to_string(),
                result: Some(snapshot_result()),
            },
        )));
        opening
            .await
            .expect("the subscription task ran")
            .expect("the subscription opens");
        server.send(&ServerMessage::ServiceEvent(
            pi_protocol::ServiceEventEnvelope {
                subscription_id: "service-1".to_string(),
                update: pi_chord::services::wire::object(vec![
                    ("type", JsonValue::string("state")),
                    ("member", JsonValue::string("state")),
                    ("sequence", JsonValue::string("not-a-number")),
                    ("ops", JsonValue::Array(vec![])),
                ]),
            },
        ));
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn reports_panicking_listeners_through_the_listener_error_handler() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let reported: Rc<RefCell<Vec<ClientErrorKind>>> = Rc::new(RefCell::new(Vec::new()));
        let reported_listener = reported.clone();
        let client = Client::new(client_options_with(
            in_memory_factory(&server),
            SERVER_ID,
            None,
            Some(Rc::new(move |error: &ClientErrorKind| {
                reported_listener.borrow_mut().push(error.clone());
            })),
        ))
        .expect("the identity is canonical");
        client.connect().await.expect("the handshake");

        let _attachment_listener = client.on_attachment_change(Rc::new(|_target| {
            // A String payload exercises the panic-message recovery's
            // owned-string arm.
            panic!("{}", "attachment listener exploded");
        }));
        attach_client(&client, &server, "session-1").await;

        let opening = open_subscription(
            &client,
            &server_target(),
            "pi.models",
            Rc::new(|_update: &ServiceProviderUpdate| {
                panic!("service listener exploded");
            }),
            None,
        );
        server.wait_for_messages(3).await;
        // Buffer one update before start, so the panic surfaces through the
        // delivery path rather than an empty queue.
        server.send(&ServerMessage::ServiceEvent(state_update(
            1,
            revision_ops(1),
        )));
        send_success(&server, "request-2", Some(snapshot_result()));
        let subscription = opening
            .await
            .expect("the subscription task ran")
            .expect("the subscription opens");
        subscription.start();

        let _state_listener = client.on_connection_state_change(Rc::new(|_change| {
            // A payload that is neither a message string nor a String lands
            // on the panic fallback.
            std::panic::panic_any(0u8);
        }));
        server.disconnect();

        let reported = reported.borrow().clone();
        assert_eq!(
            reported.len(),
            4,
            "each panicking listener reports once per change: {reported:?}"
        );
        assert_eq!(reported[0].to_string(), "attachment listener exploded");
        assert_eq!(reported[1].to_string(), "service listener exploded");
        // The disconnect first clears the attachment, then reports the state
        // change, so the attachment listener fires once more before the
        // state listener's unnamed payload falls back to `panic`.
        assert_eq!(reported[2].to_string(), "attachment listener exploded");
        assert_eq!(reported[3].to_string(), "panic");
    });
}

#[test]
fn survives_a_panicking_listener_error_handler() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = Client::new(client_options_with(
            in_memory_factory(&server),
            SERVER_ID,
            None,
            Some(Rc::new(|_error: &ClientErrorKind| {
                panic!("error handler exploded");
            })),
        ))
        .expect("the identity is canonical");
        client.connect().await.expect("the handshake");
        let _state_listener = client.on_connection_state_change(Rc::new(|_change| {
            panic!("state listener exploded");
        }));
        server.disconnect();
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn removes_listeners_when_their_unsubscribe_handles_run() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let states: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let states_listener = states.clone();
        let drop_state = client.on_connection_state_change(Rc::new(move |change| {
            states_listener
                .borrow_mut()
                .push(change.state.as_str().to_string());
        }));
        drop_state();
        let sessions: Rc<RefCell<Vec<Option<String>>>> = Rc::new(RefCell::new(Vec::new()));
        let sessions_listener = sessions.clone();
        let drop_attachment = client.on_attachment_change(Rc::new(move |target| {
            sessions_listener
                .borrow_mut()
                .push(target.map(|route| route.session_id.clone()));
        }));
        drop_attachment();

        attach_client(&client, &server, "session-1").await;
        assert!(client.attachment().is_some());
        server.disconnect();
        assert_eq!(states.borrow().clone(), Vec::<String>::new());
        assert_eq!(sessions.borrow().clone(), Vec::<Option<String>>::new());
        // The unsubscribe handles stay callable after the client drops; the
        // weak lookup misses and removes nothing.
        drop(client);
        drop_state();
        drop_attachment();
    });
}

#[test]
fn disconnects_with_the_default_reason_and_rejects_new_requests() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        client.disconnect();
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        let error = client
            .request(&server_target(), &service_call("test", "run", vec![]), None)
            .await
            .expect_err("a request on a disconnected client rejects");
        assert_eq!(error.to_string(), "Client is disconnected");
    });
}

#[test]
fn fails_the_connection_on_an_attachment_update_from_another_server() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        server.send(&ServerMessage::Attachment(AttachmentEnvelope {
            attachment: Some(pi_protocol::SessionTarget {
                server_id: ServerId::new(OTHER_SERVER_ID).unwrap(),
                session_id: "session-1".to_string(),
                attachment_id: "attachment-session-1".to_string(),
            }),
        }));
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        assert!(client.attachment().is_none());
    });
}

#[test]
fn retargets_the_attachment_when_the_server_moves_the_session() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let sessions = record_attachment_sessions(&client);
        attach_client(&client, &server, "session-1").await;
        server.send(&ServerMessage::Attachment(AttachmentEnvelope {
            attachment: Some(pi_protocol::SessionTarget {
                server_id: ServerId::new(SERVER_ID).unwrap(),
                session_id: "session-2".to_string(),
                attachment_id: "attachment-session-2".to_string(),
            }),
        }));
        let attachment = client.attachment().expect("the session retargets");
        assert_eq!(attachment.session_id, "session-2");
        assert_eq!(attachment.attachment_id, "attachment-session-2");
        // Re-attaching the same session through a fresh attachment id also
        // counts as a change, so every compared field runs.
        server.send(&ServerMessage::Attachment(AttachmentEnvelope {
            attachment: Some(pi_protocol::SessionTarget {
                server_id: ServerId::new(SERVER_ID).unwrap(),
                session_id: "session-2".to_string(),
                attachment_id: "attachment-session-2b".to_string(),
            }),
        }));
        let attachment = client.attachment().expect("the attachment refreshes");
        assert_eq!(attachment.attachment_id, "attachment-session-2b");
        assert_eq!(
            sessions.borrow().clone(),
            vec![
                Some("session-1".to_string()),
                Some("session-2".to_string()),
                Some("session-2".to_string()),
            ]
        );
        client.dispose();
    });
}

#[test]
fn ignores_stale_server_messages_after_the_connection_fails() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let (factory, armed) = recording_factory(in_memory_factory(&server));
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        client.connect().await.expect("the handshake");

        // Two unmatched responses in one chunk: the first fails the
        // connection, the second hits the mid-batch state check.
        let unmatched = |id: &str| {
            server_frame(&ServerMessage::Response(ResponseEnvelope::Success(
                pi_protocol::ResponseSuccess {
                    id: id.to_string(),
                    result: None,
                },
            )))
        };
        let mut batch = unmatched("orphan-1");
        batch.extend(unmatched("orphan-2"));
        server.send_raw(&batch);
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);

        // Everything the transport still delivers is stale and ignored.
        deliver_raw(&armed, &unmatched("orphan-3"));
        deliver_close(&armed);
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn fails_the_connection_on_a_second_handshake_message() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        server.send(&ServerMessage::Hello(ServerHello {
            server_id: ServerId::new(SERVER_ID).unwrap(),
        }));
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn fails_the_handshake_when_a_response_arrives_before_the_server_hello() {
    run_local(async {
        let probe = delivering_transport(server_frame(&ServerMessage::Response(
            ResponseEnvelope::Success(pi_protocol::ResponseSuccess {
                id: "early".to_string(),
                result: None,
            }),
        )));
        let client = Client::new(client_options(probe.factory, SERVER_ID))
            .expect("the identity is canonical");
        let error = client.connect().await.expect_err("an early response fails");
        assert!(
            matches!(error, ClientErrorKind::Protocol(_))
                && error.to_string() == "Expected server hello as first message"
        );
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        assert_eq!(probe.send_count.get(), 1);
        assert_eq!(probe.close_count.get(), 1);
    });
}

#[test]
fn fails_the_handshake_on_garbage_during_the_handshake() {
    run_local(async {
        // A complete frame whose payload is not a server message: the
        // handshake decoder rejects it instead of buffering it.
        let probe = delivering_transport(vec![0, 0, 0, 1, 0]);
        let client = Client::new(client_options(probe.factory, SERVER_ID))
            .expect("the identity is canonical");
        let error = client
            .connect()
            .await
            .expect_err("garbage fails the handshake");
        assert!(matches!(error, ClientErrorKind::Protocol(_)));
        assert!(
            error
                .to_string()
                .contains("Invalid server protocol message"),
            "the garbage frame is reported: {error}"
        );
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        assert_eq!(probe.close_count.get(), 1);
    });
}

#[test]
fn rejects_an_oversized_request_without_sending_it() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = Client::new(client_options_with(
            in_memory_factory(&server),
            SERVER_ID,
            Some(100),
            None,
        ))
        .expect("the frame bound is in range");
        client.connect().await.expect("the handshake answers");
        // The oversized call carries a keyed instance, so the wire
        // restatement of `instance` runs before the encode fails on the
        // frame bound.
        let pending = client.request(
            &server_target(),
            &instance_call("test", "run", vec![JsonValue::string("x".repeat(200))]),
            None,
        );
        let error = pending.await.expect_err("an oversized request rejects");
        assert!(matches!(error, ClientErrorKind::Protocol(_)));
        assert!(
            client.connected(),
            "the oversized frame does not fail the connection"
        );
        assert_eq!(
            server.messages().len(),
            1,
            "only the hello reached the server"
        );
        client.dispose();
    });
}

#[test]
fn maps_client_failures_onto_chord_errors_through_the_service_transport() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let transport = create_client_service_transport(&client, || Some(server_target()));
        assert!(
            format!("{transport:?}").contains("ClientServiceTransport"),
            "the transport debug names the struct"
        );

        let pending = spawn_transport_invoke(&transport, service_call("test", "run", vec![]));
        server.wait_for_messages(2).await;
        let envelope = request_envelope(&server, 1);
        send_success(&server, &envelope.id, Some(JsonValue::string("done")));
        assert_eq!(
            pending
                .await
                .expect("the task ran")
                .expect("the invocation answers"),
            Some(JsonValue::string("done"))
        );

        let pending = spawn_transport_invoke(&transport, service_call("test", "run", vec![]));
        server.wait_for_messages(3).await;
        send_failure(
            &server,
            "request-2",
            ProtocolError {
                code: "service_not_found".to_string(),
                message: "boom".to_string(),
            },
        );
        let error = pending
            .await
            .expect("the task ran")
            .expect_err("a bounded server failure maps onto the chord error");
        assert!(
            matches!(&error, ChordError::Remote(service)
                if service.code == RemoteServiceErrorCode::ServiceNotFound && service.message == "boom"),
            "the chord error carries the code and message: {error}"
        );

        let pending = spawn_transport_invoke(&transport, service_call("test", "run", vec![]));
        server.wait_for_messages(4).await;
        send_failure(
            &server,
            "request-3",
            ProtocolError {
                code: "session_not_found".to_string(),
                message: "gone".to_string(),
            },
        );
        let error = pending
            .await
            .expect("the task ran")
            .expect_err("an unbounded server failure flattens to its message");
        assert_eq!(error.to_string(), "gone");

        client.dispose();
        let pending = spawn_transport_invoke(&transport, service_call("test", "run", vec![]));
        let error = pending
            .await
            .expect("the task ran")
            .expect_err("a disposed client rejects");
        assert_eq!(error.to_string(), "Client is disposed");
    });
}

#[test]
fn rejects_service_transport_calls_when_the_target_is_unavailable() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let transport = create_client_service_transport(&client, || None);
        let rejected = spawn_transport_invoke(&transport, service_call("test", "run", vec![]));
        let error = rejected
            .await
            .expect("the task ran")
            .expect_err("an unavailable target rejects invocations");
        assert_eq!(error.to_string(), "Remote service target is unavailable");

        let rejected = spawn_transport_subscribe(&transport);
        let error = rejected
            .await
            .expect("the task ran")
            .expect_err("an unavailable target rejects subscriptions");
        assert_eq!(error.to_string(), "Remote service target is unavailable");
        client.dispose();
    });
}

#[test]
fn cancels_one_subscription_request_without_disconnecting() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let (_context, controller) = with_cancel(&background_context());
        let signal = controller.signal();
        let opening = open_default_subscription(&client, Some(signal));
        server.wait_for_messages(2).await;
        controller.abort("stop this subscription");
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("an aborted subscription rejects");
        assert_eq!(error.to_string(), "stop this subscription");
        server.wait_for_messages(3).await;
        assert_eq!(
            server.messages()[2],
            ProtocolClientMessage::Cancel(pi_protocol::CancelEnvelope {
                id: "request-1".to_string(),
                target: server_target(),
            })
        );
        assert!(client.connected());
        client.dispose();
    });
}

#[test]
fn drops_listener_panics_without_a_listener_error_handler() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = Client::new(client_options(in_memory_factory(&server), SERVER_ID))
            .expect("the identity is canonical");
        client.connect().await.expect("the handshake");
        let _state_listener = client.on_connection_state_change(Rc::new(|_change| {
            panic!("state listener exploded");
        }));
        server.disconnect();
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn disposes_the_subscription_after_the_connection_dropped() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let opening = open_default_subscription(&client, None);
        server.wait_for_messages(2).await;
        send_success(&server, "request-1", Some(snapshot_result()));
        let subscription = opening
            .await
            .expect("the subscription task ran")
            .expect("the subscription opens");
        let messages = server.messages().len();
        server.disconnect();
        let disposed = subscription.dispose().await;
        assert!(
            disposed.is_ok(),
            "disposing after the connection dropped skips the unsubscribe call"
        );
        assert_eq!(
            server.messages().len(),
            messages,
            "no unsubscribe frame is sent for a dead connection"
        );
    });
}

#[test]
fn rejects_the_handshake_when_a_state_listener_disconnects() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = Client::new(client_options(in_memory_factory(&server), SERVER_ID))
            .expect("the identity is canonical");
        let disconnecting = client.clone();
        let _state_listener = client.on_connection_state_change(Rc::new(move |change| {
            if change.state == ConnectionState::Connected {
                disconnecting.disconnect();
            }
        }));
        let error = client
            .connect()
            .await
            .expect_err("the listener disconnects");
        assert_eq!(error.to_string(), "Client disconnected");
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn rejects_an_in_flight_subscription_when_the_connection_drops() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let opening = open_default_subscription(&client, None);
        server.wait_for_messages(2).await;
        server.disconnect();
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("the disconnect rejects the pending subscription");
        assert!(matches!(error, ClientErrorKind::Disconnected { .. }));
    });
}

#[test]
fn fails_the_subscription_hydration_when_a_buffered_update_is_undecodable() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let opening = open_default_subscription(&client, None);
        server.wait_for_messages(2).await;
        // A wire update that does not parse, buffered ahead of the snapshot.
        server.send(&ServerMessage::ServiceEvent(
            pi_protocol::ServiceEventEnvelope {
                subscription_id: "service-1".to_string(),
                update: JsonValue::Null,
            },
        ));
        send_success(&server, "request-1", Some(snapshot_result()));
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("hydration fails on the buffered update");
        assert!(matches!(error, ClientErrorKind::Protocol(_)));
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn fails_updates_that_target_an_unknown_state_member() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        // Post-hydration: the state codec has no entry for the member.
        let opening = open_default_subscription(&client, None);
        server.wait_for_messages(2).await;
        send_success(&server, "request-1", Some(snapshot_result()));
        opening
            .await
            .expect("the subscription task ran")
            .expect("the subscription opens");
        server.send(&ServerMessage::ServiceEvent(unknown_member_update()));
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);

        // The same unknown member, buffered ahead of the snapshot, fails
        // hydration's queued decode instead.
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let opening = open_default_subscription(&client, None);
        server.wait_for_messages(2).await;
        server.send(&ServerMessage::ServiceEvent(unknown_member_update()));
        server.wait_for_messages(2).await;
        server.send(&ServerMessage::ServiceEvent(
            pi_protocol::ServiceEventEnvelope {
                subscription_id: "service-1".to_string(),
                update: pi_chord::services::wire::object(vec![
                    ("type", JsonValue::string("state")),
                    ("member", JsonValue::string("unknown-member")),
                    ("sequence", JsonValue::Number(1i64.into())),
                    ("ops", JsonValue::Array(vec![])),
                ]),
            },
        ));
        server.send(&ServerMessage::Response(ResponseEnvelope::Success(
            pi_protocol::ResponseSuccess {
                id: "request-1".to_string(),
                result: Some(snapshot_result()),
            },
        )));
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("hydration fails on the buffered update");
        assert!(matches!(error, ClientErrorKind::Protocol(_)));
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn rejects_the_handshake_when_the_transport_closes_while_connecting() {
    run_local(async {
        // The hello send waits on the gate, so the connection is still
        // connecting when the transport reports its orderly close.
        let (factory, handle, _fire) = handshake_gate_factory(0, true);
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        let connecting = client.connect();
        tokio::task::yield_now().await;
        let handlers = handle
            .handlers
            .borrow()
            .as_ref()
            .expect("the transport arms the handlers")
            .clone();
        (handlers.on_close)();
        let error = connecting
            .await
            .expect_err("a close during the handshake rejects");
        assert_eq!(error.to_string(), "Byte transport closed");
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
        assert_eq!(handle.close_count.get(), 1);
    });
}

#[test]
fn tolerates_handler_deliveries_after_the_client_drops() {
    run_local(async {
        // The hello send waits on the gate, so the connection task — and
        // with it the connection core — outlives the dropped client.
        let (factory, handle, fire) = handshake_gate_factory(0, true);
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        let connecting = client.connect();
        // Let the open-transport task reach its parked hello send, so the
        // transport exists and the handlers are registered.
        tokio::task::yield_now().await;
        drop(connecting);
        drop(client);
        let handlers = handle
            .handlers
            .borrow()
            .as_ref()
            .expect("the transport arms the handlers")
            .clone();
        let hello = server_frame(&ServerMessage::Hello(ServerHello {
            server_id: ServerId::new(SERVER_ID).unwrap(),
        }));
        let orphan_response = server_frame(&ServerMessage::Response(ResponseEnvelope::Success(
            pi_protocol::ResponseSuccess {
                id: "orphan".to_string(),
                result: None,
            },
        )));
        // Completing the handshake inside the still-open connection reaches
        // every client fan-out through its weak lookup, which now misses.
        (handlers.on_data)(&hello);
        (handlers.on_data)(&orphan_response);
        (handlers.on_close)();
        assert_eq!(
            handle.close_count.get(),
            1,
            "the late close event closes the connection's transport"
        );
        // The parked hello send's gate resolves now; the open-transport task
        // returns without re-failing the dead connection, and the handlers
        // lose their last hold on the connection core.
        fire.expect("the gate arms")
            .send(())
            .expect("the gate fires");
        tokio::task::yield_now().await;
        (handlers.on_data)(&orphan_response);
        (handlers.on_close)();
        (handlers.on_error)(&ClientErrorKind::other("late"));
        assert_eq!(
            handle.close_count.get(),
            1,
            "deliveries after the connection ends are no-ops"
        );
    });
}

#[test]
fn reports_the_hello_send_failure_even_after_the_client_drops() {
    run_local(async {
        // The hello send itself waits for the gate, so the connection is
        // still connecting when the client drops.
        let (factory, handle, fire) = handshake_gate_factory(0, true);
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        let connecting = client.connect();
        drop(connecting);
        drop(client);
        fire.expect("the gate arms")
            .send(())
            .expect("the gate fires");
        tokio::task::yield_now().await;
        assert!(
            handle.resolved.get(),
            "the parked hello send runs to its failure after the client drops"
        );
    });
}

#[test]
fn leaves_a_rejected_factory_unreported_after_disposal() {
    run_local(async {
        let (fire, receiver) = tokio::sync::oneshot::channel::<()>();
        let factory = {
            let receiver = RefCell::new(Some(receiver));
            Rc::new(move |_handlers: ByteTransportHandlers| {
                let receiver = receiver.borrow_mut().take();
                let rejected: Result<Rc<dyn ByteTransport>, ClientErrorKind> =
                    Err(ClientErrorKind::other("boom"));
                boxed(async move {
                    if let Some(receiver) = receiver {
                        let _ = receiver.await;
                    }
                    rejected
                })
            })
        };
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        let connecting = client.connect();
        client.dispose();
        assert_eq!(
            connecting
                .await
                .expect_err("the dispose settles the handshake"),
            ClientErrorKind::Disposed
        );
        fire.send(()).expect("the gate fires");
        tokio::task::yield_now().await;
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn rejects_a_catalogue_request_that_outlives_the_connection() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let catalogue = client.service_catalogue(&server_target(), None);
        server.disconnect();
        let error = catalogue
            .await
            .expect_err("the disconnect rejects the pending catalogue");
        assert!(matches!(error, ClientErrorKind::Disconnected { .. }));
    });
}

#[test]
fn maps_disposed_subscription_attempts_onto_chord_errors() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let transport = create_client_service_transport(&client, || Some(server_target()));
        client.dispose();
        let rejected = spawn_transport_subscribe(&transport);
        let error = rejected
            .await
            .expect("the task ran")
            .expect_err("a disposed client rejects subscriptions");
        assert_eq!(error.to_string(), "Client is disposed");
    });
}

#[test]
fn fails_the_subscription_when_the_snapshot_repeats_a_state_member() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let opening = open_default_subscription(&client, None);
        server.wait_for_messages(2).await;
        // A duplicated state member so the wire parse rejects it.
        let member = state_member(vec![revision_replace_op(0)]);
        send_success(
            &server,
            "request-1",
            Some(snapshot_with_members(vec![member.clone(), member])),
        );
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("a duplicated state member fails hydration");
        assert!(matches!(error, ClientErrorKind::Protocol(_)));
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn fails_the_subscription_when_a_snapshot_op_is_malformed() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let opening = open_default_subscription(&client, None);
        server.wait_for_messages(2).await;
        // The wire short form needs a preceding path definition; the state
        // codec rejects it during hydration.
        let snapshot = snapshot_with_members(vec![state_member(vec![JsonValue::Array(vec![
            JsonValue::string("s"),
            JsonValue::Number(0i64.into()),
        ])])]);
        send_success(&server, "request-1", Some(snapshot));
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("a malformed snapshot op fails hydration");
        assert!(matches!(error, ClientErrorKind::Protocol(_)));
        assert_eq!(client.connection_state(), ConnectionState::Disconnected);
    });
}

#[test]
fn rejects_a_request_future_that_outlives_the_client() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let pending = client.request(&server_target(), &service_call("test", "run", vec![]), None);
        drop(client);
        let error = pending
            .await
            .expect_err("the dropped client closes the answer channel");
        assert_eq!(error.to_string(), "Client is disconnected");
    });
}
