//! The wire-protocol suite, ported from upstream `test/protocol.test.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

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

use pi_chord::future::boxed;
use pi_protocol::{
    ClientHello, ClientMessage, RequestEnvelope, ResponseEnvelope, RpcTarget, ServerId,
    ServerMessage, encode_client_message, encode_frame,
};
use pi_server::testing::TestServerHost;
use pi_server::{ByteConnection, ByteConnectionHandler, Failure, Server};

use support::{SERVER_ID, Servers, connect, predicate, run_local, version};

/// The hostile CBOR frame `0xff`, upstream's `encodeFrame(Uint8Array.of(0xff))`.
fn malformed_cbor() -> Vec<u8> {
    encode_frame(&[0xff]).unwrap()
}

/// A schema-invalid hello, upstream's `encodeFrame(encodeCbor({type: "hello",
/// version: 1, extra: true}))`.
fn schema_invalid_cbor() -> Vec<u8> {
    let value = pi_protocol::CborValue::map(vec![
        ("type", pi_protocol::CborValue::Text("hello".to_string())),
        ("version", pi_protocol::CborValue::Int(1)),
        ("extra", pi_protocol::CborValue::Bool(true)),
    ]);
    let encoded = pi_protocol::encode_cbor(&value, &pi_protocol::CborOptions::default())
        .expect("the hostile shape encodes");
    encode_frame(&encoded).unwrap()
}

/// The oversized frame `1, 0, 0, 1`: its big-endian header names
/// 0x01000001 bytes, one above the 16 MiB ceiling, so the decoder rejects
/// the frame before any payload.
fn oversized_frame() -> Vec<u8> {
    vec![1, 0, 0, 1]
}

#[test]
fn requires_hello_as_the_first_message() {
    run_local(async {
        let servers = Servers::default();
        let server = support::create_server(&Rc::new(TestServerHost::new()));
        servers.track(&server);
        support::expect_hello_error(
            &server,
            |client| {
                let client = Rc::clone(client);
                boxed(async move {
                    client
                        .send_message(&ClientMessage::Request(RequestEnvelope {
                            id: "request-1".to_string(),
                            target: server_target(),
                            call: session_directory_list(),
                        }))
                        .await;
                })
            },
            "invalid_request",
        )
        .await;
        servers.close_all().await;
    });
}

#[test]
fn rejects_unsupported_protocol_versions() {
    run_local(async {
        let servers = Servers::default();
        let server = support::create_server(&Rc::new(TestServerHost::new()));
        servers.track(&server);
        let client = connect(&server);
        let answer = client.hello(version(9.0)).await.unwrap();
        let ServerMessage::HelloError(envelope) = answer else {
            panic!("expected hello_error");
        };
        assert_eq!(envelope.error.code, "version");
        client.wait_for_close().await;

        servers.close_all().await;
    });
}

#[test]
fn accepts_fragmented_hello_and_request_frames() {
    run_local(async {
        let servers = Servers::default();
        let server = support::create_server(&Rc::new(TestServerHost::new()));
        servers.track(&server);
        let client = connect(&server);
        let hello_response = client.next(predicate(|message: &ServerMessage| {
            matches!(message, ServerMessage::Hello(_))
        }));
        client
            .send_fragmented_message(
                &ClientMessage::Hello(ClientHello {
                    version: version(8.0),
                }),
                split_point(&ClientMessage::Hello(ClientHello {
                    version: version(8.0),
                })),
            )
            .await;
        let hello_response = hello_response.await.unwrap();
        assert!(
            matches!(hello_response, ServerMessage::Hello(answer) if answer.server_id.as_str() == SERVER_ID)
        );

        let response = client.next(predicate(|message: &ServerMessage| {
            matches!(message, ServerMessage::Response(_))
        }));
        let request = ClientMessage::Request(RequestEnvelope {
            id: "request-1".to_string(),
            target: server_target(),
            call: session_directory_list(),
        });
        client
            .send_fragmented_message(&request, split_point(&request))
            .await;
        let response = response.await.unwrap();
        let ServerMessage::Response(ResponseEnvelope::Failure(failure)) = response else {
            panic!("expected a failure response");
        };
        assert_eq!(failure.error.code, "internal_error");

        servers.close_all().await;
    });
}

/// The split point one frame fragments at, upstream's
/// `Math.floor(frame.byteLength / 2)`.
fn split_point(message: &ClientMessage) -> usize {
    let frame = encode_client_message(message, pi_protocol::FrameDecoderOptions::default())
        .expect("the driving case sends a valid message");
    frame.len() / 2
}

fn server_target() -> RpcTarget {
    RpcTarget::Server(pi_protocol::ServerTarget {
        server_id: ServerId::new(SERVER_ID).unwrap(),
    })
}

fn session_directory_list() -> pi_chord::types::JsonValue {
    pi_chord::types::JsonValue::Object(pi_chord::types::JsonObject::from_entries(vec![
        (
            "serviceId".to_string(),
            pi_chord::types::JsonValue::Str("pi.session-directory".to_string()),
        ),
        (
            "member".to_string(),
            pi_chord::types::JsonValue::Str("list".to_string()),
        ),
        (
            "args".to_string(),
            pi_chord::types::JsonValue::Array(vec![]),
        ),
    ]))
}

#[test]
fn rejects_hostile_framed_input_malformed_cbor() {
    run_local(async {
        rejects_hostile(malformed_cbor()).await;
    });
}

#[test]
fn rejects_hostile_framed_input_schema_invalid_cbor() {
    run_local(async {
        rejects_hostile(schema_invalid_cbor()).await;
    });
}

#[test]
fn rejects_hostile_framed_input_oversized_frame() {
    run_local(async {
        rejects_hostile(oversized_frame()).await;
    });
}

async fn rejects_hostile(bytes: Vec<u8>) {
    let servers = Servers::default();
    let server = support::create_server(&Rc::new(TestServerHost::new()));
    servers.track(&server);
    support::expect_hello_error(
        &server,
        |client| {
            let client = Rc::clone(client);
            let bytes = bytes;
            boxed(async move { client.send_bytes(bytes).await })
        },
        "invalid_request",
    )
    .await;
    servers.close_all().await;
}

#[test]
fn rejects_a_second_hello_after_completing_the_handshake() {
    run_local(async {
        let servers = Servers::default();
        let server = support::create_server(&Rc::new(TestServerHost::new()));
        servers.track(&server);
        let client = connect(&server);
        client.hello(version(8.0)).await.unwrap();
        client
            .send_message(&ClientMessage::Hello(ClientHello {
                version: version(8.0),
            }))
            .await;
        support::expect_first_message_rejection(&client).await;

        servers.close_all().await;
    });
}

#[test]
fn processes_a_hello_and_request_coalesced_in_one_byte_chunk() {
    run_local(async {
        let servers = Servers::default();
        let server = support::create_server(&Rc::new(TestServerHost::new()));
        servers.track(&server);
        let client = connect(&server);
        let hello = encode_client_message(
            &ClientMessage::Hello(ClientHello {
                version: version(8.0),
            }),
            pi_protocol::FrameDecoderOptions::default(),
        )
        .unwrap();
        let request = encode_client_message(
            &ClientMessage::Request(RequestEnvelope {
                id: "request-1".to_string(),
                target: server_target(),
                call: session_directory_list(),
            }),
            pi_protocol::FrameDecoderOptions::default(),
        )
        .unwrap();
        let mut wire = hello.clone();
        wire.extend_from_slice(&request);

        client.send_bytes(wire).await;
        let hello_response = client
            .next(predicate(|message: &ServerMessage| {
                matches!(message, ServerMessage::Hello(_))
            }))
            .await
            .unwrap();
        assert!(matches!(hello_response, ServerMessage::Hello(_)));
        let response = client
            .next(predicate(|message: &ServerMessage| {
                matches!(message, ServerMessage::Response(_))
            }))
            .await
            .unwrap();
        let ServerMessage::Response(ResponseEnvelope::Failure(failure)) = response else {
            panic!("expected a failure response");
        };
        assert_eq!(failure.id, "request-1");
        assert_eq!(failure.error.code, "internal_error");

        servers.close_all().await;
    });
}

#[test]
fn reports_a_truncated_final_frame_when_the_peer_closes() {
    run_local(async {
        let servers = Servers::default();
        let errors = Rc::new(RefCell::new(Vec::<Failure>::new()));
        let errors_observer: pi_server::ErrorObserver = {
            let errors = Rc::clone(&errors);
            Rc::new(move |error: &Failure| errors.borrow_mut().push(error.clone()))
        };
        let server = create_server_with_errors(errors_observer);
        servers.track(&server);
        let closed = Rc::new(Cell::new(false));
        let connection: Rc<dyn ByteConnection> = {
            let closed = Rc::clone(&closed);
            Rc::new(SilentConnection { closed })
        };
        let handler: ByteConnectionHandler = server.accept(connection);
        (handler.on_data)(&[0, 0, 0, 2, 1]);
        (handler.on_close)();

        assert!(!closed.get());
        let errors = errors.borrow().clone();
        assert_eq!(errors.len(), 1);
        assert!(
            errors[0].to_string().to_lowercase().contains("truncated"),
            "the error names the truncation: {}",
            errors[0]
        );
        servers.close_all().await;
    });
}

/// A connection that accepts frames silently, upstream's fixture.
struct SilentConnection {
    closed: Rc<Cell<bool>>,
}

impl ByteConnection for SilentConnection {
    fn closed(&self) -> bool {
        self.closed.get()
    }

    fn send(&self, _chunk: Vec<u8>) -> pi_chord::future::LocalBoxFuture<Result<(), Failure>> {
        boxed(async { Ok(()) })
    }

    fn close(
        &self,
        _final_chunk: Option<Vec<u8>>,
    ) -> pi_chord::future::LocalBoxFuture<Result<(), Failure>> {
        self.closed.set(true);
        boxed(async { Ok(()) })
    }
}

fn create_server_with_errors(on_error: pi_server::ErrorObserver) -> Server<TestServerHost> {
    Server::new(
        Rc::new(TestServerHost::new()),
        pi_server::ServerOptions {
            listeners: Vec::new(),
            server_id: SERVER_ID.to_string(),
            max_frame_length: None,
            handshake_timeout_ms: None,
            on_connection_count_changed: None,
            on_error: Some(on_error),
        },
    )
    .unwrap()
}
