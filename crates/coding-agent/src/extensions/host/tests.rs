#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_protocol::{CborOptions, CborValue, FrameDecoder, FrameDecoderOptions, encode_cbor};
use serde_json::json;

use super::{
    EXTENSION_PROTOCOL_VERSION, ExtensionIdentity, Frame, cbor_to_json, decode_frame, encode_frame,
    json_to_cbor,
};

#[expect(
    clippy::too_many_lines,
    reason = "the fixture table is the wire's complete frame index"
)]
fn sample_frames() -> Vec<Frame> {
    vec![
        Frame::Hello {
            protocol_version: EXTENSION_PROTOCOL_VERSION,
            extension: ExtensionIdentity {
                name: "demo".to_string(),
                version: "0.1.0".to_string(),
            },
        },
        Frame::Welcome {
            protocol_version: EXTENSION_PROTOCOL_VERSION,
        },
        Frame::HelloError {
            code: "protocol_version".to_string(),
            message: "unsupported wire version".to_string(),
        },
        Frame::Ready,
        Frame::RegisterEvent {
            channel: "agent_start".to_string(),
        },
        Frame::RegisterTool {
            tool: json!({"name": "demo_tool", "parameters": {"type": "object"}}),
        },
        Frame::RegisterCommand {
            command: json!({"name": "demo"}),
        },
        Frame::RegisterFlag {
            flag: json!({"name": "demo", "type": "boolean"}),
        },
        Frame::RegisterProvider {
            provider: json!({"name": "demo"}),
        },
        Frame::CallEvent {
            call: 7,
            channel: "agent_start".to_string(),
            payload: json!({"step": 1}),
        },
        Frame::CallTool {
            call: 8,
            tool: "demo_tool".to_string(),
            tool_call_id: "call_1".to_string(),
            params: json!({"path": "README.md"}),
            context: json!({"cwd": "/tmp", "thinking_level": "off"}),
        },
        Frame::CallUpdate {
            call: 8,
            update: json!({"content": "partial"}),
        },
        Frame::CallResult {
            call: 8,
            result: json!({"ok": true}),
        },
        Frame::CallError {
            call: 8,
            code: "handler_failed".to_string(),
            message: "boom".to_string(),
            stack: Some("at demo".to_string()),
        },
        Frame::CallError {
            call: 9,
            code: "handler_failed".to_string(),
            message: "boom".to_string(),
            stack: None,
        },
        Frame::CallCommand {
            call: 10,
            name: "demo".to_string(),
            args: "--flag".to_string(),
        },
        Frame::CallProvider {
            call: 11,
            provider: "demo".to_string(),
            request: json!({"model": "demo-model"}),
        },
        Frame::StreamEvent {
            call: 11,
            event: json!({"type": "text_delta"}),
        },
        Frame::StreamEnd { call: 11 },
        Frame::ContextCall {
            call: 12,
            id: 1,
            method: "cwd".to_string(),
            params: None,
        },
        Frame::ContextCall {
            call: 12,
            id: 2,
            method: "models.list".to_string(),
            params: Some(json!({"provider": "demo"})),
        },
        Frame::ContextResult {
            call: 12,
            id: 1,
            value: json!("/tmp"),
        },
        Frame::ContextError {
            call: 12,
            id: 2,
            code: "unsupported".to_string(),
            message: "no ui bound".to_string(),
        },
        Frame::UiCall {
            call: 13,
            id: 3,
            method: "select".to_string(),
            params: Some(json!({"options": ["a", "b"]})),
        },
        Frame::UiResult {
            call: 13,
            id: 3,
            value: json!("a"),
        },
        Frame::UiError {
            call: 13,
            id: 4,
            code: "unknown_method".to_string(),
            message: "no such method".to_string(),
        },
        Frame::Signal {
            call: 14,
            signal: 5,
            aborted: true,
        },
    ]
}

fn payload_of(encoded: &[u8]) -> Vec<u8> {
    let mut decoder =
        FrameDecoder::new(FrameDecoderOptions::default()).expect("the default options are valid");
    let mut frames = decoder.push(encoded).expect("the frames split");
    assert_eq!(frames.len(), 1);
    frames.remove(0)
}

#[test]
fn every_frame_round_trips() {
    for frame in sample_frames() {
        let encoded = encode_frame(&frame).expect("the frame encodes");
        let payload = payload_of(&encoded);
        assert_eq!(
            decode_frame(&payload).expect("the frame decodes"),
            frame,
            "round trip for {frame:?}"
        );
    }
}

#[test]
fn protocol_version_is_one() {
    assert_eq!(EXTENSION_PROTOCOL_VERSION, 1);
}

fn raw_frame(value: &CborValue) -> Vec<u8> {
    encode_cbor(value, &CborOptions::default()).expect("the raw frame encodes")
}

#[test]
fn bridge_keeps_integers_integral() {
    assert_eq!(json_to_cbor(&json!(42)), CborValue::Int(42));
    assert_eq!(json_to_cbor(&json!(-7)), CborValue::Int(-7));
    assert_eq!(json_to_cbor(&json!(0)), CborValue::Int(0));
}

#[test]
fn bridge_carries_non_integral_numbers_as_floats() {
    assert_eq!(json_to_cbor(&json!(1.5)), CborValue::Float(1.5));
    assert_eq!(json_to_cbor(&json!(1.0)), CborValue::Float(1.0));
}

#[test]
fn bridge_sends_unsigned_values_past_i64_as_floats() {
    assert!(matches!(
        json_to_cbor(&json!(u64::MAX)),
        CborValue::Float(_)
    ));
}

#[test]
fn bridge_round_trips_nested_json() {
    let value = json!({"a": [1, 2.5, null, true, "x"], "nested": {"n": -3}});
    let cbor = json_to_cbor(&value);
    assert_eq!(cbor_to_json(&cbor).expect("the bridge returns JSON"), value);
}

#[test]
fn bridge_rejects_cbor_byte_strings() {
    assert!(cbor_to_json(&CborValue::Bytes(vec![1])).is_err());
}

#[test]
fn bridge_rejects_non_finite_floats() {
    assert!(cbor_to_json(&CborValue::Float(f64::NAN)).is_err());
    assert!(cbor_to_json(&CborValue::Float(f64::INFINITY)).is_err());
}

#[test]
fn decode_rejects_unknown_type() {
    let bytes = raw_frame(&CborValue::map(vec![(
        "type",
        CborValue::Text("nope".to_string()),
    )]));
    let error = decode_frame(&bytes).expect_err("the type is unknown");
    assert!(error.message().contains("unknown extension frame type"));
}

#[test]
fn decode_rejects_missing_type() {
    assert!(decode_frame(&raw_frame(&CborValue::Map(Vec::new()))).is_err());
}

#[test]
fn decode_rejects_missing_field() {
    let bytes = raw_frame(&CborValue::map(vec![
        ("type", CborValue::Text("call_event".to_string())),
        ("call", CborValue::Int(1)),
    ]));
    let error = decode_frame(&bytes).expect_err("the channel is missing");
    assert!(error.message().contains("channel"));
}

#[test]
fn decode_rejects_wrong_typed_field() {
    let bytes = raw_frame(&CborValue::map(vec![
        ("type", CborValue::Text("welcome".to_string())),
        ("protocol_version", CborValue::Text("1".to_string())),
    ]));
    let error = decode_frame(&bytes).expect_err("the version is a string");
    assert!(error.message().contains("protocol_version"));
}

#[test]
fn decode_rejects_wrong_typed_bool() {
    let bytes = raw_frame(&CborValue::map(vec![
        ("type", CborValue::Text("signal".to_string())),
        ("call", CborValue::Int(1)),
        ("signal", CborValue::Int(2)),
        ("aborted", CborValue::Text("yes".to_string())),
    ]));
    let error = decode_frame(&bytes).expect_err("aborted is a string");
    assert!(error.to_string().contains("aborted"));
}

#[test]
fn decode_rejects_wrong_typed_optional_field() {
    let bytes = raw_frame(&CborValue::map(vec![
        ("type", CborValue::Text("call_error".to_string())),
        ("call", CborValue::Int(1)),
        ("code", CborValue::Text("handler_failed".to_string())),
        ("message", CborValue::Text("boom".to_string())),
        ("stack", CborValue::Int(0)),
    ]));
    assert!(decode_frame(&bytes).is_err());
}

#[test]
fn decode_rejects_non_map_payload() {
    assert!(decode_frame(&raw_frame(&CborValue::Array(Vec::new()))).is_err());
}

#[test]
fn decode_rejects_non_json_payload() {
    let bytes = raw_frame(&CborValue::map(vec![
        ("type", CborValue::Text("call_event".to_string())),
        ("call", CborValue::Int(1)),
        ("channel", CborValue::Text("agent_start".to_string())),
        ("payload", CborValue::Bytes(vec![0])),
    ]));
    let error = decode_frame(&bytes).expect_err("the payload is a byte string");
    assert!(error.message().contains("byte string"));
}

#[test]
fn decode_rejects_trailing_data() {
    let mut bytes = raw_frame(&CborValue::map(vec![(
        "type",
        CborValue::Text("ready".to_string()),
    )]));
    bytes.push(0);
    assert!(decode_frame(&bytes).is_err());
}

#[test]
fn decode_rejects_duplicate_keys() {
    let bytes = b"\xa2\x64type\x67welcome\x64type\x67welcome";
    assert!(decode_frame(bytes).is_err());
}

#[test]
fn encode_rejects_integers_past_the_safe_range() {
    let frame = Frame::CallEvent {
        call: 1,
        channel: "agent_start".to_string(),
        payload: json!(i64::MAX),
    };
    assert!(encode_frame(&frame).is_err());
}

#[test]
fn encode_rejects_ids_past_the_wire_range() {
    assert!(encode_frame(&Frame::StreamEnd { call: u64::MAX }).is_err());
}
