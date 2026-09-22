//! The protocol validation suite, ported from upstream
//! `test/protocol.test.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]

use pi_chord::types::{JsonNumber, JsonObject, JsonValue};
use pi_protocol::{
    AttachmentEnvelope, CancelEnvelope, CborOptions, CborValue, ClientHello, ClientMessage,
    ClientMessageDecoder, FrameDecoder, FrameDecoderOptions, PROTOCOL_VERSION, ProtocolError,
    RequestEnvelope, ResponseEnvelope, ResponseFailure, ResponseSuccess, RpcTarget, ServerHello,
    ServerId, ServerMessage, ServerMessageDecoder, ServerTarget, ServiceEventEnvelope,
    SessionTarget, decode_cbor, encode_cbor, encode_client_message, encode_frame,
    encode_server_message, is_supported_protocol_version, parse_client_message,
    parse_server_message,
};

const SERVER_ID: &str = "00000000-0000-4000-8000-000000000001";

fn t(text: &str) -> CborValue {
    CborValue::Text(text.to_string())
}

fn map(entries: Vec<(&str, CborValue)>) -> CborValue {
    CborValue::map(entries)
}

fn js(text: &str) -> JsonValue {
    JsonValue::Str(text.to_string())
}

fn jo(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(JsonObject::from_entries(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    ))
}

const fn ja(items: Vec<JsonValue>) -> JsonValue {
    JsonValue::Array(items)
}

fn number(value: i64) -> JsonValue {
    JsonValue::Number(JsonNumber::from(value))
}

fn to_cbor(value: &JsonValue) -> CborValue {
    match value {
        JsonValue::Null => CborValue::Null,
        JsonValue::Bool(flag) => CborValue::Bool(*flag),
        JsonValue::Number(number) => CborValue::Float(number.get()),
        JsonValue::Str(text) => CborValue::Text(text.clone()),
        JsonValue::Array(items) => CborValue::Array(items.iter().map(to_cbor).collect()),
        JsonValue::Object(object) => CborValue::Map(
            object
                .iter()
                .map(|(key, value)| (key.to_string(), to_cbor(value)))
                .collect(),
        ),
    }
}

fn concatenate(chunks: &[&[u8]]) -> Vec<u8> {
    let mut result = Vec::with_capacity(chunks.iter().map(|chunk| chunk.len()).sum());
    for chunk in chunks {
        result.extend_from_slice(chunk);
    }
    result
}

fn options() -> FrameDecoderOptions {
    FrameDecoderOptions::default()
}

fn frame_decoder() -> FrameDecoder {
    FrameDecoder::new(FrameDecoderOptions::default()).expect("default options are valid")
}

fn server_id() -> ServerId {
    ServerId::new(SERVER_ID).expect("canonical test id")
}

fn hello_message() -> ClientMessage {
    ClientMessage::Hello(ClientHello {
        version: JsonNumber::from(PROTOCOL_VERSION.cast_signed()),
    })
}

fn server_hello_message() -> ServerMessage {
    ServerMessage::Hello(ServerHello {
        server_id: server_id(),
    })
}

fn models_call() -> JsonValue {
    jo(vec![
        ("serviceId", js("pi.models")),
        ("member", js("list")),
        ("args", ja(vec![])),
    ])
}

fn opaque_call() -> JsonValue {
    jo(vec![
        ("serviceId", js("application.custom")),
        (
            "instance",
            jo(vec![("key", js("instance-1")), ("generation", number(2))]),
        ),
        ("member", js("invoke")),
        (
            "args",
            ja(vec![
                jo(vec![("arbitrary", JsonValue::Bool(true))]),
                ja(vec![js("opaque")]),
            ]),
        ),
    ])
}

fn session_directory_call() -> JsonValue {
    jo(vec![
        ("serviceId", js("pi.session-directory")),
        ("member", js("list")),
        ("args", ja(vec![])),
    ])
}

fn session_request(call: &JsonValue) -> CborValue {
    map(vec![
        ("type", t("request")),
        ("id", t("request-1")),
        (
            "target",
            map(vec![
                ("serverId", t(SERVER_ID)),
                ("sessionId", t("session-1")),
                ("attachmentId", t("attachment-1")),
            ]),
        ),
        ("call", to_cbor(call)),
    ])
}

fn session_request_message(call: JsonValue) -> ClientMessage {
    ClientMessage::Request(RequestEnvelope {
        id: "request-1".to_string(),
        target: RpcTarget::Session(SessionTarget {
            server_id: server_id(),
            session_id: "session-1".to_string(),
            attachment_id: "attachment-1".to_string(),
        }),
        call,
    })
}

#[test]
fn negotiates_protocol_version_8() {
    assert_eq!(PROTOCOL_VERSION, 8);
    assert!(is_supported_protocol_version(JsonNumber::from(8_i64)));
    assert!(!is_supported_protocol_version(JsonNumber::from(7_i64)));
    assert!(!is_supported_protocol_version(
        JsonNumber::new(8.5).expect("finite")
    ));
}

#[test]
fn accepts_integer_client_hello_versions_for_negotiation() {
    for version in [
        0,
        PROTOCOL_VERSION.cast_signed(),
        PROTOCOL_VERSION.cast_signed() + 1,
    ] {
        let value = map(vec![
            ("type", t("hello")),
            ("version", CborValue::Int(version)),
        ]);
        let parsed = parse_client_message(&value).expect("integer versions negotiate");
        assert_eq!(
            parsed,
            ClientMessage::Hello(ClientHello {
                version: JsonNumber::from(version),
            })
        );
    }
}

#[test]
fn rejects_an_invalid_client_hello() {
    let cases = [
        map(vec![("type", t("hello")), ("version", t("8"))]),
        map(vec![
            ("type", t("hello")),
            ("version", CborValue::Float(8.5)),
        ]),
        map(vec![
            ("type", t("hello")),
            ("version", CborValue::Int(8)),
            ("extra", CborValue::Bool(true)),
        ]),
    ];
    for value in cases {
        assert!(parse_client_message(&value).is_err(), "rejects {value:?}");
    }
}

#[test]
fn rejects_non_canonical_uuidv4_server_ids() {
    for server_id in [
        "",
        "server-1",
        "00000000-0000-7000-8000-000000000001",
        "00000000-0000-4000-7000-000000000001",
        "00000000-0000-4000-8000-00000000000A",
    ] {
        let value = map(vec![
            ("type", t("request")),
            ("id", t("request-1")),
            ("target", map(vec![("serverId", t(server_id))])),
            ("call", to_cbor(&models_call())),
        ]);
        assert!(
            parse_client_message(&value).is_err(),
            "rejects non-canonical server id {server_id:?}"
        );
    }
}

#[test]
fn keeps_routed_request_and_event_payloads_opaque() {
    let parsed =
        parse_client_message(&session_request(&opaque_call())).expect("opaque call parses");
    assert_eq!(parsed, session_request_message(opaque_call()));

    // Strict JSON whose service meaning belongs to Chord parses as-is.
    let arbitrary = jo(vec![(
        "arbitrary",
        js("strict JSON whose service meaning belongs to Chord"),
    )]);
    let parsed =
        parse_client_message(&session_request(&arbitrary)).expect("arbitrary strict JSON parses");
    let ClientMessage::Request(RequestEnvelope { call, .. }) = &parsed else {
        panic!("a request parses");
    };
    assert_eq!(call, &arbitrary);

    let parsed = parse_server_message(&map(vec![
        ("type", t("service_update")),
        ("subscriptionId", t("subscription-1")),
        (
            "update",
            map(vec![("applicationDefined", CborValue::Bool(true))]),
        ),
    ]))
    .expect("opaque update parses");
    let ServerMessage::ServiceEvent(ServiceEventEnvelope { update, .. }) = &parsed else {
        panic!("a service_update parses");
    };
    assert_eq!(
        update
            .as_object()
            .and_then(|object| object.get("applicationDefined")),
        Some(&JsonValue::Bool(true))
    );
}

#[test]
fn rejects_non_json_opaque_payloads() {
    for (label, argument) in [
        ("byte array", CborValue::Bytes(vec![1])),
        ("non-finite number", CborValue::Float(f64::NAN)),
    ] {
        let call = map(vec![
            ("serviceId", t("application.custom")),
            ("member", t("invoke")),
            ("args", CborValue::Array(vec![argument.clone()])),
        ]);
        let request = map(vec![
            ("type", t("request")),
            ("id", t("request-1")),
            ("target", map(vec![("serverId", t(SERVER_ID))])),
            ("call", call),
        ]);
        assert!(
            parse_client_message(&request).is_err(),
            "client rejects a {label} payload"
        );
        let response = map(vec![
            ("type", t("response")),
            ("id", t("request-1")),
            ("ok", CborValue::Bool(true)),
            ("result", CborValue::Array(vec![argument])),
        ]);
        assert!(
            parse_server_message(&response).is_err(),
            "server rejects a {label} payload"
        );
    }
    // Upstream's remaining cases — an `undefined` property and a cycle —
    // are unrepresentable in the owned tree: there is no `undefined`
    // variant, and owned data cannot reference itself.
}

#[test]
fn validates_request_cancellation_envelopes() {
    let cancel = map(vec![
        ("type", t("cancel")),
        ("id", t("request-1")),
        ("target", map(vec![("serverId", t(SERVER_ID))])),
    ]);
    assert_eq!(
        parse_client_message(&cancel).expect("cancel parses"),
        ClientMessage::Cancel(CancelEnvelope {
            id: "request-1".to_string(),
            target: RpcTarget::Server(ServerTarget {
                server_id: server_id(),
            }),
        })
    );

    let empty_id = map(vec![
        ("type", t("cancel")),
        ("id", t("")),
        ("target", map(vec![("serverId", t(SERVER_ID))])),
    ]);
    assert!(parse_client_message(&empty_id).is_err());

    let extra = map(vec![
        ("type", t("cancel")),
        ("id", t("request-1")),
        ("target", map(vec![("serverId", t(SERVER_ID))])),
        ("extra", CborValue::Bool(true)),
    ]);
    assert!(parse_client_message(&extra).is_err());
}

#[test]
fn validates_attachment_route_updates() {
    let attached = map(vec![
        ("type", t("attachment")),
        (
            "attachment",
            map(vec![
                ("serverId", t(SERVER_ID)),
                ("sessionId", t("session-1")),
                ("attachmentId", t("attachment-1")),
            ]),
        ),
    ]);
    let detached = map(vec![
        ("type", t("attachment")),
        ("attachment", CborValue::Null),
    ]);
    assert_eq!(
        parse_server_message(&attached).expect("attached parses"),
        ServerMessage::Attachment(AttachmentEnvelope {
            attachment: Some(SessionTarget {
                server_id: server_id(),
                session_id: "session-1".to_string(),
                attachment_id: "attachment-1".to_string(),
            }),
        })
    );
    assert_eq!(
        parse_server_message(&detached).expect("detached parses"),
        ServerMessage::Attachment(AttachmentEnvelope { attachment: None })
    );

    let partial = map(vec![
        ("type", t("attachment")),
        ("attachment", map(vec![("sessionId", t("session-1"))])),
    ]);
    assert!(parse_server_message(&partial).is_err());
}

#[test]
fn rejects_malformed_request_boundaries() {
    let empty_id = map(vec![
        ("type", t("request")),
        ("id", t("")),
        ("target", map(vec![("serverId", t(SERVER_ID))])),
        ("call", to_cbor(&models_call())),
    ]);
    assert!(parse_client_message(&empty_id).is_err());

    let extra = map(vec![
        ("type", t("request")),
        ("id", t("request-1")),
        ("target", map(vec![("serverId", t(SERVER_ID))])),
        ("call", to_cbor(&models_call())),
        ("extra", CborValue::Bool(true)),
    ]);
    assert!(parse_client_message(&extra).is_err());
}

#[test]
fn accepts_a_successful_void_response_without_a_result_field() {
    let value = map(vec![
        ("type", t("response")),
        ("id", t("request-1")),
        ("ok", CborValue::Bool(true)),
    ]);
    assert_eq!(
        parse_server_message(&value).expect("void response parses"),
        ServerMessage::Response(ResponseEnvelope::Success(ResponseSuccess {
            id: "request-1".to_string(),
            result: None,
        }))
    );
}

#[test]
fn rejects_malformed_server_boundaries() {
    let invalid_hello = map(vec![
        ("type", t("hello")),
        ("version", CborValue::Int(PROTOCOL_VERSION.cast_signed())),
        ("serverId", t("server-1")),
    ]);
    assert!(parse_server_message(&invalid_hello).is_err());

    let extra_response = map(vec![
        ("type", t("response")),
        ("id", t("request-1")),
        ("ok", CborValue::Bool(true)),
        ("result", CborValue::Array(vec![])),
        ("extra", CborValue::Bool(true)),
    ]);
    assert!(parse_server_message(&extra_response).is_err());

    let empty_code = map(vec![
        ("type", t("response")),
        ("id", t("request-1")),
        ("ok", CborValue::Bool(false)),
        ("error", map(vec![("code", t("")), ("message", t("bad"))])),
    ]);
    assert!(parse_server_message(&empty_code).is_err());
}

#[test]
fn accepts_the_opaque_error_codes() {
    for code in [
        "wrong_server",
        "cancelled",
        "service_not_found",
        "application_error",
    ] {
        let value = map(vec![
            ("type", t("response")),
            ("id", t("request-1")),
            ("ok", CborValue::Bool(false)),
            (
                "error",
                map(vec![("code", t(code)), ("message", t("safe"))]),
            ),
        ]);
        assert_eq!(
            parse_server_message(&value).unwrap_or_else(|_| panic!("accepts {code}")),
            ServerMessage::Response(ResponseEnvelope::Failure(ResponseFailure {
                id: "request-1".to_string(),
                error: ProtocolError {
                    code: code.to_string(),
                    message: "safe".to_string(),
                },
            }))
        );
    }
}

#[test]
fn rejects_unknown_messages_and_fields() {
    let snapshot_hello = map(vec![
        ("type", t("hello")),
        ("version", CborValue::Int(PROTOCOL_VERSION.cast_signed())),
        ("serverId", t(SERVER_ID)),
        ("snapshot", CborValue::Map(vec![])),
    ]);
    assert!(parse_server_message(&snapshot_hello).is_err());

    let unknown = map(vec![
        ("type", t("unknown")),
        ("event", CborValue::Map(vec![])),
    ]);
    assert!(parse_server_message(&unknown).is_err());
}

#[test]
fn does_not_parse_json_strings_as_messages() {
    // Upstream feeds `JSON.stringify(clientHello)`; a decoded text string is
    // the same shape on the wire.
    assert!(
        parse_client_message(&CborValue::Text(
            "{\"type\":\"hello\",\"version\":8}".to_string()
        ))
        .is_err()
    );
    assert!(parse_server_message(&CborValue::Text(
        "{\"type\":\"hello\",\"version\":8,\"serverId\":\"00000000-0000-4000-8000-000000000001\"}"
            .to_string(),
    ))
    .is_err());
}

#[test]
fn encodes_complete_client_and_server_frames() {
    let client_wire = encode_client_message(&hello_message(), options()).expect("hello encodes");
    let mut frames = frame_decoder();
    let client_frames = frames.push(&client_wire).expect("clean stream");
    let value = decode_cbor(&client_frames[0], &CborOptions::default()).expect("frame decodes");
    assert_eq!(
        parse_client_message(&value).expect("parses"),
        hello_message()
    );

    let server_wire =
        encode_server_message(&server_hello_message(), options()).expect("server hello encodes");
    let mut frames = frame_decoder();
    let server_frames = frames.push(&server_wire).expect("clean stream");
    let value = decode_cbor(&server_frames[0], &CborOptions::default()).expect("frame decodes");
    assert_eq!(
        parse_server_message(&value).expect("parses"),
        server_hello_message()
    );
}

#[test]
fn enforces_outbound_frame_limits() {
    let bounded = FrameDecoderOptions {
        max_frame_length: 8,
    };
    let client_error =
        encode_client_message(&hello_message(), bounded).expect_err("over the frame limit");
    assert!(client_error.message().contains("Unable to encode client"));
    let server_error =
        encode_server_message(&server_hello_message(), bounded).expect_err("over the frame limit");
    assert!(server_error.message().contains("Unable to encode server"));
}

#[test]
fn incrementally_decodes_fragmented_and_coalesced_client_messages() {
    let messages = [
        hello_message(),
        session_request_message(session_directory_call()),
    ];
    let first = encode_client_message(&hello_message(), options()).expect("hello encodes");
    let second = encode_client_message(
        &session_request_message(session_directory_call()),
        options(),
    )
    .expect("request encodes");
    let wire = concatenate(&[&first, &second]);

    for split in 0..=wire.len() {
        let mut message_decoder = ClientMessageDecoder::new(options()).expect("default options");
        let mut decoded = message_decoder.push(&wire[..split]).expect("clean stream");
        decoded.extend(message_decoder.push(&wire[split..]).expect("clean stream"));
        message_decoder.end().expect("clean stream");
        assert_eq!(decoded, messages);
    }
}

#[test]
fn incrementally_decodes_fragmented_and_coalesced_server_messages() {
    let response = ServerMessage::Response(ResponseEnvelope::Success(ResponseSuccess {
        id: "request-1".to_string(),
        result: Some(JsonValue::Array(vec![])),
    }));
    let first =
        encode_server_message(&server_hello_message(), options()).expect("server hello encodes");
    let second = encode_server_message(&response, options()).expect("response encodes");
    let wire = concatenate(&[&first, &second]);

    let split = first.len() + second.len() / 2;
    let mut message_decoder = ServerMessageDecoder::new(options()).expect("default options");
    assert_eq!(
        message_decoder.push(&wire[..split]).expect("clean stream"),
        vec![server_hello_message()]
    );
    assert_eq!(
        message_decoder.push(&wire[split..]).expect("clean stream"),
        vec![response]
    );
    message_decoder.end().expect("clean stream");
}

#[test]
fn rejects_invalid_framed_input() {
    let schema_invalid = encode_frame(
        &encode_cbor(
            &map(vec![
                ("type", t("hello")),
                ("version", CborValue::Int(1)),
                ("extra", CborValue::Bool(true)),
            ]),
            &CborOptions::default(),
        )
        .expect("map encodes"),
    )
    .expect("frame encodes");
    let cases = [
        (
            "empty CBOR payload",
            encode_frame(&[]).expect("frame encodes"),
        ),
        (
            "malformed CBOR",
            encode_frame(&[0xff]).expect("frame encodes"),
        ),
        ("schema-invalid CBOR", schema_invalid),
    ];
    let clean = encode_client_message(&hello_message(), options()).expect("clean frame encodes");
    for (label, wire) in cases {
        let mut message_decoder = ClientMessageDecoder::new(options()).expect("default options");
        assert!(message_decoder.push(&wire).is_err(), "rejects {label}");
        let error = message_decoder
            .push(&clean)
            .expect_err("failed state latches");
        assert!(error.message().to_lowercase().contains("failed"), "{label}");
    }
}

#[test]
fn rejects_truncated_and_oversized_framing() {
    let mut truncated =
        ServerMessageDecoder::new(FrameDecoderOptions::default()).expect("default options");
    assert_eq!(
        truncated
            .push(&[0x00, 0x00, 0x00, 0x02, 0x01])
            .expect("payload stays partial"),
        Vec::<ServerMessage>::new()
    );
    assert!(truncated.end().is_err());

    let mut oversized = ClientMessageDecoder::new(FrameDecoderOptions {
        max_frame_length: 3,
    })
    .expect("bounded options");
    assert!(oversized.push(&[0x00, 0x00, 0x00, 0x04]).is_err());
}
