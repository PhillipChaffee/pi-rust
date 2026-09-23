//! The client suite, ported from upstream `test/client.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

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
    reason = "the ported cases index messages the preceding waits guarantee"
)]
#![allow(
    clippy::too_many_lines,
    reason = "the buffering case is one upstream test; splitting it would scatter the assertions"
)]

mod support;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_chord::context::{background_context, with_cancel};
use pi_chord::delta::op::Op;
use pi_chord::delta::ops::Seg;
use pi_chord::future::boxed;
use pi_chord::types::{
    JsonValue, RemoteServiceTransport, ServiceMemberSnapshot, ServiceMode, ServiceProviderUpdate,
};
use pi_client::{Client, ClientError as ClientErrorKind, create_client_service_transport};
use pi_protocol::{
    AttachmentEnvelope, ClientMessage, ProtocolError, ResponseEnvelope, RpcTarget, ServerHello,
    ServerId, ServerMessage, encode_frame,
};
use support::{
    MemoryByteServer, SERVER_ID, attach_client, client_options, connect_client, connect_client_to,
    delivering_transport, in_memory_factory, noop_service_listener, open_subscription,
    record_attachment_sessions, request_envelope, revision_ops, run_local, send_failure,
    send_success, server_frame, server_target, service_call, session_target, snapshot_result,
    state_update,
};

fn object_field<'a>(value: &'a JsonValue, key: &str) -> Option<&'a JsonValue> {
    match value {
        JsonValue::Object(object) => object.get(key),
        _ => None,
    }
}

fn snapshot_value(path: Vec<&str>, value: JsonValue) -> Op {
    Op::Set {
        path: path
            .into_iter()
            .map(|key| Seg::Key(key.to_string()))
            .collect(),
        value,
    }
}

#[test]
fn requires_a_canonical_uuidv4_server_identity() {
    run_local(async {
        let options = client_options(
            in_memory_factory(&Rc::new(MemoryByteServer::new(SERVER_ID))),
            "invalid-server",
        );
        let error = Client::new(options).expect_err("a non-canonical server id fails");
        assert!(
            error.to_string().contains("serverId"),
            "the error names the server identity"
        );
    });
}

#[test]
fn connects_only_to_the_expected_logical_server() {
    run_local(async {
        let matching = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&matching).await.expect("the handshake");
        let hello = client.hello().expect("the handshake completed");
        assert_eq!(hello.server_id.as_str(), SERVER_ID);
        client.dispose();

        let wrong = Rc::new(MemoryByteServer::new(
            "00000000-0000-4000-8000-000000000002",
        ));
        let connected = connect_client_to(&wrong, SERVER_ID).await;
        assert!(matches!(connected, Err(ClientErrorKind::Protocol(_))));
    });
}

#[test]
fn updates_attachment_state_from_out_of_band_server_routing() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let changes = record_attachment_sessions(&client);

        attach_client(&client, &server, "session-1").await;
        let attachment = client.attachment().expect("the session attaches");
        assert_eq!(attachment.session_id, "session-1");
        assert_eq!(attachment.attachment_id, "attachment-session-1");
        let envelope = request_envelope(&server, 1);
        assert_eq!(envelope.target, server_target());
        assert_eq!(
            object_field(&envelope.call, "serviceId"),
            Some(&JsonValue::string("pi.session-management"))
        );
        assert_eq!(
            object_field(&envelope.call, "member"),
            Some(&JsonValue::string("attach"))
        );
        assert_eq!(
            object_field(&envelope.call, "args"),
            Some(&JsonValue::Array(vec![JsonValue::string("session-1")]))
        );

        server.send(&ServerMessage::Attachment(AttachmentEnvelope {
            attachment: None,
        }));
        assert!(client.attachment().is_none());
        assert_eq!(
            changes.borrow().clone(),
            vec![Some("session-1".to_string()), None]
        );
        client.dispose();
    });
}

#[test]
fn buffers_service_updates_until_the_subscription_snapshot_arrives() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        attach_client(&client, &server, "session-1").await;
        let transport = create_client_service_transport(&client, {
            let client = client.clone();
            move || client.attachment().map(RpcTarget::Session)
        });
        let updates: Rc<RefCell<Vec<ServiceProviderUpdate>>> = Rc::new(RefCell::new(Vec::new()));
        let updates_listener = updates.clone();
        let opening = tokio::task::spawn_local(transport.subscribe(
            "pi.models".to_string(),
            ServiceMode::Singleton,
            Rc::new(move |update, _context| updates_listener.borrow_mut().push(update.clone())),
            background_context(),
        ));
        server.wait_for_messages(3).await;
        let envelope = request_envelope(&server, 2);
        assert_eq!(
            envelope.target,
            session_target("session-1", "attachment-session-1")
        );
        assert_eq!(
            object_field(&envelope.call, "serviceId"),
            Some(&JsonValue::string("$chord.service"))
        );
        assert_eq!(
            object_field(&envelope.call, "member"),
            Some(&JsonValue::string("subscribe"))
        );
        assert_eq!(
            object_field(&envelope.call, "args"),
            Some(&JsonValue::Array(vec![
                JsonValue::string("service-1"),
                JsonValue::string("pi.models"),
                JsonValue::string("singleton"),
            ]))
        );
        server.send(&ServerMessage::ServiceEvent(state_update(
            1,
            revision_ops(1),
        )));
        assert_eq!(updates.borrow().len(), 0);
        send_success(&server, "request-2", Some(snapshot_result()));
        let subscription = opening
            .await
            .expect("the subscription opens")
            .expect("the subscription opens");
        assert_eq!(updates.borrow().len(), 0);
        server.send(&ServerMessage::ServiceEvent(
            pi_protocol::ServiceEventEnvelope {
                subscription_id: "closed-subscription".to_string(),
                update: pi_chord::services::wire::object(vec![
                    ("type", JsonValue::string("state")),
                    ("member", JsonValue::string("state")),
                    ("sequence", JsonValue::Number(99i64.into())),
                    (
                        "ops",
                        JsonValue::Array(vec![JsonValue::Array(vec![
                            JsonValue::string("s"),
                            JsonValue::Number(99i64.into()),
                            JsonValue::Number(99i64.into()),
                        ])]),
                    ),
                ]),
            },
        ));
        assert!(client.connected());
        assert_eq!(
            subscription.snapshot.instances[0].members,
            vec![ServiceMemberSnapshot::State {
                name: "state".to_string(),
                sequence: 0,
                ops: vec![Op::Replace(pi_chord::services::wire::object(vec![(
                    "revision",
                    JsonValue::Number(0i64.into())
                )]))],
            }]
        );
        (subscription.activate)().expect("activation succeeds");
        assert_eq!(
            updates
                .borrow()
                .iter()
                .map(|update| match update {
                    ServiceProviderUpdate::State { .. } => "state",
                    _ => "other",
                })
                .collect::<Vec<_>>(),
            vec!["state"]
        );
        server.send(&ServerMessage::ServiceEvent(state_update(
            2,
            revision_ops(2),
        )));
        server.send(&ServerMessage::ServiceEvent(state_update(
            3,
            vec![
                JsonValue::Array(vec![
                    JsonValue::string("#"),
                    JsonValue::Number(0i64.into()),
                    JsonValue::Array(vec![JsonValue::string("revision")]),
                ]),
                JsonValue::Array(vec![
                    JsonValue::string("s"),
                    JsonValue::Number(0i64.into()),
                    JsonValue::Number(3i64.into()),
                ]),
            ],
        )));
        assert_eq!(updates.borrow().len(), 3);
        assert_eq!(
            match &updates.borrow()[2] {
                ServiceProviderUpdate::State { ops, .. } => ops.clone(),
                _ => panic!("the third update is a state publication"),
            },
            vec![snapshot_value(
                vec!["revision"],
                JsonValue::Number(3i64.into()),
            )]
        );

        let disposing = (subscription.close)(Some(background_context()));
        server.wait_for_messages(4).await;
        let envelope = request_envelope(&server, 3);
        assert_eq!(
            object_field(&envelope.call, "serviceId"),
            Some(&JsonValue::string("$chord.service"))
        );
        assert_eq!(
            object_field(&envelope.call, "member"),
            Some(&JsonValue::string("unsubscribe"))
        );
        assert_eq!(
            object_field(&envelope.call, "args"),
            Some(&JsonValue::Array(vec![JsonValue::string("service-1")]))
        );
        send_success(&server, "request-3", None);
        disposing.await.expect("the subscription closes");
        client.dispose();
    });
}

#[test]
fn correlates_out_of_order_generic_service_responses() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let first = client.request(
            &server_target(),
            &service_call("test", "first", vec![]),
            None,
        );
        let second = client.request(
            &server_target(),
            &service_call("test", "second", vec![]),
            None,
        );
        server.wait_for_messages(3).await;
        send_success(&server, "request-2", Some(JsonValue::string("second")));
        send_success(&server, "request-1", Some(JsonValue::string("first")));
        assert_eq!(first.await, Ok(Some(JsonValue::string("first"))));
        assert_eq!(second.await, Ok(Some(JsonValue::string("second"))));
        client.dispose();
    });
}

#[test]
fn exposes_bounded_server_errors() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let pending = client.request(
            &server_target(),
            &service_call("test", "missing", vec![]),
            None,
        );
        server.wait_for_messages(2).await;
        send_failure(
            &server,
            "request-1",
            ProtocolError {
                code: "session_not_found".to_string(),
                message: "Unknown session".to_string(),
            },
        );
        let error = pending.await.expect_err("bounded server errors reject");
        assert_eq!(
            error,
            ClientErrorKind::Server(ProtocolError {
                code: "session_not_found".to_string(),
                message: "Unknown session".to_string(),
            })
        );
        client.dispose();
    });
}

#[test]
fn does_not_send_a_pre_aborted_untyped_rpc_request() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let (_context, controller) = with_cancel(&background_context());
        controller.abort("already cancelled");
        let signal = controller.signal();

        let error = client
            .request(
                &server_target(),
                &service_call("test", "noop", vec![]),
                Some(signal),
            )
            .await
            .expect_err("pre-aborted requests reject with the reason");
        assert_eq!(error.to_string(), "already cancelled");
        assert_eq!(server.messages().len(), 1);
        client.dispose();
    });
}

#[test]
fn cancels_one_untyped_rpc_request_without_disconnecting() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let (_context, controller) = with_cancel(&background_context());
        let signal = controller.signal();
        let pending = client.request(
            &server_target(),
            &service_call(
                "test",
                "mutate",
                vec![pi_chord::services::wire::object(vec![(
                    "value",
                    JsonValue::Number(42i64.into()),
                )])],
            ),
            Some(signal),
        );
        server.wait_for_messages(2).await;
        let envelope = request_envelope(&server, 1);
        assert_eq!(envelope.id, "request-1");
        assert_eq!(envelope.target, server_target());
        assert_eq!(
            object_field(&envelope.call, "serviceId"),
            Some(&JsonValue::string("test"))
        );
        assert_eq!(
            object_field(&envelope.call, "member"),
            Some(&JsonValue::string("mutate"))
        );
        assert_eq!(
            object_field(&envelope.call, "args"),
            Some(&JsonValue::Array(vec![pi_chord::services::wire::object(
                vec![("value", JsonValue::Number(42i64.into())),]
            )]))
        );

        controller.abort("stop this request");
        let error = pending.await.expect_err("aborted requests reject");
        assert_eq!(error.to_string(), "stop this request");
        server.wait_for_messages(3).await;
        let messages = server.messages();
        assert_eq!(
            messages[2],
            ClientMessage::Cancel(pi_protocol::CancelEnvelope {
                id: "request-1".to_string(),
                target: server_target(),
            })
        );
        send_failure(
            &server,
            "request-1",
            ProtocolError {
                code: "cancelled".to_string(),
                message: "cancelled".to_string(),
            },
        );
        assert!(client.connected());
        client.dispose();
    });
}

#[test]
fn rejects_pending_requests_after_disconnect_or_disposal() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let pending = client.request(
            &server_target(),
            &service_call("test", "pending", vec![]),
            None,
        );
        server.disconnect();
        assert!(matches!(
            pending.await,
            Err(ClientErrorKind::Disconnected { .. })
        ));
        client.dispose();
        let disposed = client.request(
            &server_target(),
            &service_call("test", "disposed", vec![]),
            None,
        );
        assert!(matches!(disposed.await, Err(ClientErrorKind::Disposed)));
    });
}

#[test]
fn rejects_server_data_delivered_before_the_client_hello_is_sent() {
    run_local(async {
        let send_count = Rc::new(Cell::new(0));
        let close_count = Rc::new(Cell::new(0));
        let handshake_frame = server_frame(&ServerMessage::Hello(ServerHello {
            server_id: ServerId::new(SERVER_ID).unwrap(),
        }));
        let factory = {
            let send_count = Rc::clone(&send_count);
            let close_count = Rc::clone(&close_count);
            Rc::new(move |handlers: pi_client::ByteTransportHandlers| {
                (handlers.on_data)(&handshake_frame);
                support::counting_transport(Rc::clone(&send_count), Rc::clone(&close_count), None)(
                    handlers,
                )
            })
        };
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        let connected = client.connect();
        let error = connected.await.expect_err("early server data fails");
        assert!(
            matches!(error, ClientErrorKind::Protocol(_))
                && error.to_string() == "Received server data before the client hello was sent"
        );
        assert_eq!(
            client.connection_state(),
            pi_client::ConnectionState::Disconnected
        );
        assert_eq!(send_count.get(), 0);
        assert_eq!(close_count.get(), 1);
    });
}

#[test]
fn rejects_typed_handshake_errors_and_closes_the_transport() {
    run_local(async {
        let probe = delivering_transport(server_frame(&ServerMessage::HelloError(
            pi_protocol::ServerHelloError {
                error: ProtocolError {
                    code: "version".to_string(),
                    message: "Unsupported protocol version".to_string(),
                },
            },
        )));
        let client = Client::new(client_options(probe.factory, SERVER_ID))
            .expect("the identity is canonical");
        let error = client.connect().await.expect_err("hello errors reject");
        assert_eq!(
            error,
            ClientErrorKind::Server(ProtocolError {
                code: "version".to_string(),
                message: "Unsupported protocol version".to_string(),
            })
        );
        assert_eq!(
            client.connection_state(),
            pi_client::ConnectionState::Disconnected
        );
        assert_eq!(probe.close_count.get(), 1);
    });
}

#[test]
fn rejects_pending_requests_and_reconnects_through_a_fresh_transport() {
    run_local(async {
        let first = Rc::new(MemoryByteServer::new(SERVER_ID));
        let second = Rc::new(MemoryByteServer::new(SERVER_ID));
        let connection = Rc::new(Cell::new(0usize));
        let factory = {
            let first = Rc::clone(&first);
            let second = Rc::clone(&second);
            let connection = Rc::clone(&connection);
            Rc::new(move |handlers| {
                let index = connection.get();
                connection.set(index + 1);
                let server = if index == 0 {
                    Rc::clone(&first)
                } else {
                    Rc::clone(&second)
                };
                let transport = server.connect(handlers);
                boxed(std::future::ready(Ok(transport)))
            })
        };
        let client =
            Client::new(client_options(factory, SERVER_ID)).expect("the identity is canonical");
        let states: Rc<RefCell<Vec<pi_client::ConnectionState>>> =
            Rc::new(RefCell::new(Vec::new()));
        let states_listener = states.clone();
        let _state_listener = client.on_connection_state_change(Rc::new(move |change| {
            states_listener.borrow_mut().push(change.state);
        }));
        client.connect().await.expect("the first handshake");
        attach_client(&client, &first, "session-1").await;
        let target = client.attachment().expect("the session attaches");
        let pending = client.request(
            &RpcTarget::Session(target),
            &service_call("test.session", "run", vec![]),
            None,
        );
        first.wait_for_messages(3).await;
        let envelope = request_envelope(&first, 2);
        assert_eq!(
            object_field(&envelope.call, "serviceId"),
            Some(&JsonValue::string("test.session"))
        );
        assert_eq!(
            object_field(&envelope.call, "member"),
            Some(&JsonValue::string("run"))
        );
        first.disconnect();

        assert!(matches!(
            pending.await,
            Err(ClientErrorKind::Disconnected { .. })
        ));
        let hello = client.reconnect().await.expect("the reconnect answers");
        assert_eq!(hello.server_id.as_str(), SERVER_ID);
        assert_eq!(connection.get(), 2);
        assert!(client.connected());
        assert_eq!(second.messages().len(), 1);
        assert_eq!(
            states.borrow().clone(),
            vec![
                pi_client::ConnectionState::Connecting,
                pi_client::ConnectionState::Connected,
                pi_client::ConnectionState::Disconnected,
                pi_client::ConnectionState::Connecting,
                pi_client::ConnectionState::Connected,
            ]
        );
        client.dispose();
    });
}

#[test]
fn reports_transport_failures_without_leaving_requests_pending() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let pending = client.request(
            &server_target(),
            &service_call("test", "pending", vec![]),
            None,
        );
        server.wait_for_messages(2).await;
        server.error(&ClientErrorKind::other("read failed"));

        let error = pending.await.expect_err("transport failures fail requests");
        assert!(matches!(error, ClientErrorKind::Disconnected { .. }));
        assert_eq!(error.to_string(), "read failed");
        let cause = std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<ClientErrorKind>())
            .expect("the cause carries the failure");
        assert_eq!(cause.to_string(), "read failed");
        assert_eq!(
            client.connection_state(),
            pi_client::ConnectionState::Disconnected
        );
    });
}

#[test]
fn disconnects_on_invalid_or_truncated_server_framing() {
    run_local(async {
        let invalid_server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let invalid_client = connect_client(&invalid_server)
            .await
            .expect("the handshake");
        let response_frame = server_frame(&ServerMessage::Response(ResponseEnvelope::Success(
            pi_protocol::ResponseSuccess {
                id: "unknown".to_string(),
                result: Some(JsonValue::Number(1i64.into())),
            },
        )));
        invalid_server.send_raw(&encode_frame(&response_frame).expect("the frame encodes"));
        assert_eq!(
            invalid_client.connection_state(),
            pi_client::ConnectionState::Disconnected
        );

        let truncated_server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let truncated_client = connect_client(&truncated_server)
            .await
            .expect("the handshake");
        let pending = truncated_client.request(
            &server_target(),
            &service_call("test", "pending", vec![]),
            None,
        );
        truncated_server.wait_for_messages(2).await;
        truncated_server.send_raw(&[0, 0, 0, 2, 1]);
        truncated_server.disconnect();

        let error = pending.await.expect_err("truncation fails the request");
        assert!(matches!(error, ClientErrorKind::Protocol(_)));
        assert!(
            error.to_string().to_lowercase().contains("truncated"),
            "the truncation is reported: {error}"
        );
        assert_eq!(
            truncated_client.connection_state(),
            pi_client::ConnectionState::Disconnected
        );
    });
}

#[test]
fn disconnects_when_a_response_has_no_matching_request() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        send_success(&server, "unknown-request", Some(JsonValue::Array(vec![])));
        assert_eq!(
            client.connection_state(),
            pi_client::ConnectionState::Disconnected
        );
        assert_eq!(server.client_close_count(), 1);
    });
}

/// Exercises the subscription transform's failure path: a snapshot answer
/// that is not a service subscription fails the connection and rejects with
/// the validation error (port-only boundary; upstream reaches the same path
/// through its transform catch).
#[test]
fn fails_the_connection_when_the_subscription_snapshot_is_invalid() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        attach_client(&client, &server, "session-1").await;
        let opening = open_subscription(
            &client,
            &session_target("session-1", "attachment-session-1"),
            "pi.models",
            noop_service_listener(),
            None,
        );
        server.wait_for_messages(3).await;
        send_success(&server, "request-2", Some(JsonValue::Null));
        let error = opening
            .await
            .expect("the subscription task ran")
            .expect_err("an invalid snapshot fails");
        assert!(matches!(error, ClientErrorKind::Protocol(_)));
        assert_eq!(
            client.connection_state(),
            pi_client::ConnectionState::Disconnected
        );
    });
}

/// Exercises the service catalogue surface upstream ships but its suite
/// does not drive (port-only boundary).
#[test]
fn fetches_and_validates_the_service_catalogue() {
    run_local(async {
        let server = Rc::new(MemoryByteServer::new(SERVER_ID));
        let client = connect_client(&server).await.expect("the handshake");
        let catalogue = client.service_catalogue(&server_target(), None);
        server.wait_for_messages(2).await;
        let envelope = request_envelope(&server, 1);
        assert_eq!(
            object_field(&envelope.call, "member"),
            Some(&JsonValue::string("catalogue"))
        );
        send_success(
            &server,
            &envelope.id,
            Some(pi_chord::services::wire::object(vec![(
                "ignored",
                JsonValue::Null,
            )])),
        );
        let error = catalogue.await.expect_err("a non-array answer is invalid");
        assert!(matches!(error, ClientErrorKind::Protocol(_)));
        assert_eq!(
            client.connection_state(),
            pi_client::ConnectionState::Disconnected
        );
    });
}
