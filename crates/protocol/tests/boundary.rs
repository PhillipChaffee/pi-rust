//! Boundary tests binding the branches the 1:1 port leaves open: encode-side
//! arms the upstream suite drives through JS-only paths, the restated
//! rejection surface's remaining cases, and the defensive restatements the
//! port carries for fidelity. Upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]

use pi_protocol::{
    AttachmentEnvelope, CancelEnvelope, CborOptions, CborValue, ClientMessage,
    ClientMessageDecoder, DEFAULT_MAX_CBOR_BYTE_LENGTH, FrameDecoderOptions, ProtocolError,
    ResponseEnvelope, ResponseFailure, RpcTarget, ServerHelloError, ServerId, ServerMessage,
    ServerMessageDecoder, ServiceEventEnvelope, SessionTarget, decode_cbor, encode_cbor,
    encode_client_message, encode_server_message, is_server_id, parse_client_message,
    parse_server_message,
};

fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).expect("hex digit pair"))
        .collect()
}

const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

fn t(text: &str) -> CborValue {
    CborValue::Text(text.to_string())
}

fn map(entries: Vec<(&str, CborValue)>) -> CborValue {
    CborValue::map(entries)
}

fn server_id() -> ServerId {
    ServerId::new(SERVER_ID).expect("canonical test id")
}

#[test]
fn spells_the_server_id_back_and_checks_the_canonical_form() {
    assert_eq!(server_id().to_string(), SERVER_ID);
    assert!(is_server_id(SERVER_ID));
    assert!(!is_server_id("00000000-0000-4000-7000-000000000001"));
    assert!(!is_server_id("00000000-0000-4000-8000-0000000000010"));
}

#[test]
fn displays_the_codec_error_message_it_carries() {
    let error = parse_client_message(&CborValue::Null).expect_err("null is not a message");
    assert_eq!(error.to_string(), error.message());
    assert_eq!(error.message(), "Invalid client protocol message");
}

#[test]
fn encodes_every_server_message_variant_and_round_trips_it() {
    let messages = [
        ServerMessage::HelloError(ServerHelloError {
            error: ProtocolError {
                code: "wrong_server".to_string(),
                message: "safe".to_string(),
            },
        }),
        ServerMessage::Response(ResponseEnvelope::Failure(ResponseFailure {
            id: "request-1".to_string(),
            error: ProtocolError {
                code: "cancelled".to_string(),
                message: "safe".to_string(),
            },
        })),
        ServerMessage::ServiceEvent(ServiceEventEnvelope {
            subscription_id: "subscription-1".to_string(),
            update: pi_chord::types::JsonValue::Bool(true),
        }),
        ServerMessage::Attachment(AttachmentEnvelope {
            attachment: Some(SessionTarget {
                server_id: server_id(),
                session_id: "session-1".to_string(),
                attachment_id: "attachment-1".to_string(),
            }),
        }),
        ServerMessage::Attachment(AttachmentEnvelope { attachment: None }),
    ];
    for message in messages {
        let wire = encode_server_message(&message, FrameDecoderOptions::default())
            .expect("the variant encodes");
        let mut message_decoder =
            ServerMessageDecoder::new(FrameDecoderOptions::default()).expect("default options");
        let decoded = message_decoder.push(&wire).expect("clean stream");
        message_decoder.end().expect("clean stream");
        assert_eq!(decoded, vec![message]);
    }
}

#[test]
fn encodes_a_cancel_envelope_and_round_trips_it() {
    let cancel = ClientMessage::Cancel(CancelEnvelope {
        id: "request-1".to_string(),
        target: RpcTarget::Server(pi_protocol::ServerTarget {
            server_id: server_id(),
        }),
    });
    let wire = encode_client_message(&cancel, FrameDecoderOptions::default()).expect("encodes");
    let mut message_decoder =
        ClientMessageDecoder::new(FrameDecoderOptions::default()).expect("default options");
    let decoded = message_decoder.push(&wire).expect("clean stream");
    message_decoder.end().expect("clean stream");
    assert_eq!(decoded, vec![cancel]);
}

#[test]
fn round_trips_a_json_rich_opaque_payload_through_the_framed_codec() {
    // Every JSON arm of the payload conversion: null, booleans, numbers
    // riding both the integer and float64 paths, strings, arrays, objects.
    let call = pi_chord::types::JsonValue::Object(pi_chord::types::JsonObject::from_entries(vec![
        ("nil".to_string(), pi_chord::types::JsonValue::Null),
        ("yes".to_string(), pi_chord::types::JsonValue::Bool(true)),
        (
            "int".to_string(),
            pi_chord::types::JsonValue::Number(pi_chord::types::JsonNumber::from(2_i64)),
        ),
        (
            "float".to_string(),
            pi_chord::types::JsonValue::Number(
                pi_chord::types::JsonNumber::new(1.5).expect("finite"),
            ),
        ),
        (
            "text".to_string(),
            pi_chord::types::JsonValue::Str("s".to_string()),
        ),
        (
            "list".to_string(),
            pi_chord::types::JsonValue::Array(vec![pi_chord::types::JsonValue::Number(
                pi_chord::types::JsonNumber::from(0_i64),
            )]),
        ),
    ]));
    let request = ClientMessage::Request(pi_protocol::RequestEnvelope {
        id: "request-1".to_string(),
        target: RpcTarget::Server(pi_protocol::ServerTarget {
            server_id: server_id(),
        }),
        call,
    });
    let wire = encode_client_message(&request, FrameDecoderOptions::default()).expect("encodes");
    let mut message_decoder =
        ClientMessageDecoder::new(FrameDecoderOptions::default()).expect("default options");
    let decoded = message_decoder.push(&wire).expect("clean stream");
    message_decoder.end().expect("clean stream");
    assert_eq!(decoded, vec![request]);
}

#[test]
fn rejects_an_opaque_payload_nested_past_the_json_depth_cap() {
    // `isJsonValue`'s 512-level cap restates as the conversion's depth cap;
    // a hand-built payload deeper than the cap is not strict JSON.
    let mut deep = CborValue::Null;
    for _ in 0..=pi_chord::json::MAX_DEPTH {
        deep = CborValue::Array(vec![deep]);
    }
    let request = map(vec![
        ("type", t("request")),
        ("id", t("request-1")),
        ("target", map(vec![("serverId", t(SERVER_ID))])),
        ("call", deep),
    ]);
    assert!(parse_client_message(&request).is_err());
}

type RejectionCase = (&'static str, CborValue, fn(&CborValue) -> bool);

fn server_rejection_cases() -> Vec<RejectionCase> {
    vec![
        (
            "hello_error with an extra field",
            map(vec![
                ("type", t("hello_error")),
                ("error", map(vec![("code", t("x")), ("message", t("m"))])),
                ("extra", CborValue::Bool(true)),
            ]),
            |value| parse_server_message(value).is_err(),
        ),
        (
            "hello_error with an empty error code",
            map(vec![
                ("type", t("hello_error")),
                ("error", map(vec![("code", t("")), ("message", t("m"))])),
            ]),
            |value| parse_server_message(value).is_err(),
        ),
        (
            "server hello version as text",
            map(vec![
                ("type", t("hello")),
                ("version", t("8")),
                ("serverId", t(SERVER_ID)),
            ]),
            |value| parse_server_message(value).is_err(),
        ),
        (
            "server hello version 7",
            map(vec![
                ("type", t("hello")),
                ("version", CborValue::Int(7)),
                ("serverId", t(SERVER_ID)),
            ]),
            |value| parse_server_message(value).is_err(),
        ),
        (
            "failure response with an extra field",
            map(vec![
                ("type", t("response")),
                ("id", t("request-1")),
                ("ok", CborValue::Bool(false)),
                ("error", map(vec![("code", t("x")), ("message", t("m"))])),
                ("extra", CborValue::Bool(true)),
            ]),
            |value| parse_server_message(value).is_err(),
        ),
        (
            "service_update with an extra field",
            map(vec![
                ("type", t("service_update")),
                ("subscriptionId", t("subscription-1")),
                ("update", CborValue::Bool(true)),
                ("extra", CborValue::Bool(true)),
            ]),
            |value| parse_server_message(value).is_err(),
        ),
        (
            "attachment route that is neither a target nor null",
            map(vec![
                ("type", t("attachment")),
                ("attachment", CborValue::Bool(true)),
            ]),
            |value| parse_server_message(value).is_err(),
        ),
        (
            "error object carrying an extra field",
            map(vec![
                ("type", t("response")),
                ("id", t("request-1")),
                ("ok", CborValue::Bool(false)),
                (
                    "error",
                    map(vec![
                        ("code", t("x")),
                        ("message", t("m")),
                        ("extra", CborValue::Bool(true)),
                    ]),
                ),
            ]),
            |value| parse_server_message(value).is_err(),
        ),
        (
            "attachment envelope without its attachment field",
            map(vec![
                ("type", t("attachment")),
                ("extra", CborValue::Bool(true)),
            ]),
            |value| parse_server_message(value).is_err(),
        ),
    ]
}

fn client_rejection_cases() -> Vec<RejectionCase> {
    vec![
        (
            "request target missing its attachment id",
            map(vec![
                ("type", t("request")),
                ("id", t("request-1")),
                (
                    "target",
                    map(vec![
                        ("serverId", t(SERVER_ID)),
                        ("sessionId", t("session-1")),
                    ]),
                ),
                ("call", CborValue::Null),
            ]),
            |value| parse_client_message(value).is_err(),
        ),
        (
            "unknown client message type",
            map(vec![("type", t("unknown"))]),
            |value| parse_client_message(value).is_err(),
        ),
        (
            "client hello without a version",
            map(vec![("type", t("hello"))]),
            |value| parse_client_message(value).is_err(),
        ),
    ]
}

#[test]
fn rejects_the_remaining_schema_shapes() {
    let cases = client_rejection_cases()
        .into_iter()
        .chain(server_rejection_cases());
    for (label, value, rejects) in cases {
        assert!(rejects(&value), "rejects {label}");
    }
}

#[test]
fn decoder_rejects_indefinite_length_items_on_integer_arguments() {
    // Upstream's readArgument carries the indefinite-length rejection for
    // the uint/nint paths (`0x1f`, `0x3f`), which the fixture list leaves
    // untested; the port keeps the same rejection.
    for wire in ["1f", "3f"] {
        assert!(decode_cbor(&from_hex(wire), &CborOptions::default()).is_err());
    }
}

#[test]
fn encoder_enforces_the_string_and_container_limits_it_carries() {
    let byte_bounded = CborOptions {
        max_byte_length: 16,
        ..CborOptions::default()
    };
    let error = encode_cbor(
        &CborValue::Text("0123456789abcdef0".to_string()),
        &byte_bounded,
    )
    .expect_err("text over the byte limit");
    assert!(error.message().contains("text string length"));

    let bytes = CborValue::Bytes(vec![0; DEFAULT_MAX_CBOR_BYTE_LENGTH + 1]);
    let error = encode_cbor(&bytes, &CborOptions::default()).expect_err("bytes over the limit");
    assert!(error.message().contains("byte string length"));

    let container_bounded = CborOptions {
        max_container_length: 2,
        ..CborOptions::default()
    };
    let map = CborValue::Map(vec![
        ("a".to_string(), CborValue::Int(1)),
        ("b".to_string(), CborValue::Int(2)),
        ("c".to_string(), CborValue::Int(3)),
    ]);
    let error = encode_cbor(&map, &container_bounded).expect_err("map over the limit");
    assert!(error.message().contains("map length"));
}

#[test]
fn encoder_takes_the_negative_integer_path_for_negative_floats() {
    let encoded = encode_cbor(&CborValue::Float(-5.0), &CborOptions::default()).expect("encodes");
    assert_eq!(hex(&encoded), "24");
}

#[test]
fn resolves_option_limits_against_their_configured_ranges() {
    let cases = [
        (
            CborOptions {
                max_byte_length: u32::MAX as usize + 1,
                ..CborOptions::default()
            },
            "maxByteLength",
        ),
        (
            CborOptions {
                max_container_length: u32::MAX as usize + 1,
                ..CborOptions::default()
            },
            "maxContainerLength",
        ),
        (
            CborOptions {
                // The configured depth cap is 512 (upstream's MAX_CONFIGURED_DEPTH).
                max_depth: 513,
                ..CborOptions::default()
            },
            "maxDepth",
        ),
    ];
    for (options, name) in cases {
        let error =
            encode_cbor(&CborValue::Null, &options).expect_err(&format!("{name} over range"));
        assert!(
            error
                .message()
                .contains(&format!("{name} must be an integer between 0 and")),
            "{name}: {}",
            error.message()
        );
    }
}

#[test]
fn decoders_reject_the_configuration_range_at_construction() {
    let options = FrameDecoderOptions {
        max_frame_length: u32::MAX as usize + 1,
    };
    let client_error = ClientMessageDecoder::new(options).expect_err("past the u32 range");
    assert!(
        client_error
            .message()
            .contains("must be an integer between 0 and")
    );
    let server_error = ServerMessageDecoder::new(options).expect_err("past the u32 range");
    assert!(
        server_error
            .message()
            .contains("must be an integer between 0 and")
    );
}

#[test]
fn decoders_latch_the_failed_state_across_end() {
    let mut truncated = ServerMessageDecoder::new(FrameDecoderOptions::default())
        .expect("default options are valid");
    assert!(truncated.push(&[0x00, 0x00, 0x00, 0x02, 0x01]).is_ok());
    assert!(truncated.end().is_err());
    let error = truncated.end().expect_err("failed state latches");
    assert!(
        error
            .message()
            .contains("server message decoder has failed")
    );

    let mut broken = ClientMessageDecoder::new(FrameDecoderOptions::default())
        .expect("default options are valid");
    assert!(
        broken
            .push(&[0x00, 0x00, 0x00, 0x02, 0x01])
            .expect("payload stays partial")
            .is_empty()
    );
    assert!(broken.end().is_err());
    let error = broken.end().expect_err("failed state latches");
    assert!(
        error
            .message()
            .contains("client message decoder has failed")
    );
}

fn hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}
