//! [`decode_frame`]: one pi-protocol payload to a [`Frame`].
//!
//! The substrate's rejections are inherited; this layer rejects a non-map
//! payload, an unknown `type`, a missing or wrong-typed field, and a payload
//! the JSON bridge cannot represent.

use pi_protocol::{CborOptions, CborValue, decode_cbor};

use super::WireError;
use super::bridge::cbor_to_json;
use super::frame::{ExtensionIdentity, Frame};

/// Decodes one pi-protocol frame payload into a [`Frame`].
///
/// `payload` is the item a [`pi_protocol::FrameDecoder`] yields, without its
/// 4-byte length prefix.
///
/// # Errors
///
/// Returns [`WireError`] for the framing and CBOR rejections the substrate
/// carries, and for an unknown frame type, a missing or wrong-typed field,
/// or a payload that is not strict JSON.
pub fn decode_frame(payload: &[u8]) -> Result<Frame, WireError> {
    let value = decode_cbor(payload, &CborOptions::default())?;
    frame_from_cbor(&value)
}

#[expect(
    clippy::too_many_lines,
    reason = "the frame table is one match, the wire's complete index"
)]
pub(super) fn frame_from_cbor(value: &CborValue) -> Result<Frame, WireError> {
    let fields = as_fields(value, "envelope")?;
    let kind = required_text(fields, "type")?;
    match kind {
        "hello" => Ok(Frame::Hello {
            protocol_version: required_uint(fields, "protocol_version")?,
            extension: decode_identity(required(fields, "extension")?)?,
        }),
        "welcome" => Ok(Frame::Welcome {
            protocol_version: required_uint(fields, "protocol_version")?,
        }),
        "hello_error" => Ok(Frame::HelloError {
            code: required_text(fields, "code")?.to_string(),
            message: required_text(fields, "message")?.to_string(),
        }),
        "ready" => Ok(Frame::Ready),
        "register_event" => Ok(Frame::RegisterEvent {
            channel: required_text(fields, "channel")?.to_string(),
        }),
        "register_tool" => Ok(Frame::RegisterTool {
            tool: required_json(fields, "tool")?,
        }),
        "register_command" => Ok(Frame::RegisterCommand {
            command: required_json(fields, "command")?,
        }),
        "register_flag" => Ok(Frame::RegisterFlag {
            flag: required_json(fields, "flag")?,
        }),
        "register_provider" => Ok(Frame::RegisterProvider {
            provider: required_json(fields, "provider")?,
        }),
        "call_event" => Ok(Frame::CallEvent {
            call: required_uint(fields, "call")?,
            channel: required_text(fields, "channel")?.to_string(),
            payload: required_json(fields, "payload")?,
        }),
        "call_tool" => Ok(Frame::CallTool {
            call: required_uint(fields, "call")?,
            tool: required_text(fields, "tool")?.to_string(),
            tool_call_id: required_text(fields, "tool_call_id")?.to_string(),
            params: required_json(fields, "params")?,
            context: required_json(fields, "context")?,
        }),
        "call_update" => Ok(Frame::CallUpdate {
            call: required_uint(fields, "call")?,
            update: required_json(fields, "update")?,
        }),
        "call_result" => Ok(Frame::CallResult {
            call: required_uint(fields, "call")?,
            result: required_json(fields, "result")?,
        }),
        "call_error" => Ok(Frame::CallError {
            call: required_uint(fields, "call")?,
            code: required_text(fields, "code")?.to_string(),
            message: required_text(fields, "message")?.to_string(),
            stack: optional_text(fields, "stack")?,
        }),
        "call_command" => Ok(Frame::CallCommand {
            call: required_uint(fields, "call")?,
            name: required_text(fields, "name")?.to_string(),
            args: required_text(fields, "args")?.to_string(),
        }),
        "call_provider" => Ok(Frame::CallProvider {
            call: required_uint(fields, "call")?,
            provider: required_text(fields, "provider")?.to_string(),
            request: required_json(fields, "request")?,
        }),
        "stream_event" => Ok(Frame::StreamEvent {
            call: required_uint(fields, "call")?,
            event: required_json(fields, "event")?,
        }),
        "stream_end" => Ok(Frame::StreamEnd {
            call: required_uint(fields, "call")?,
        }),
        "context_call" => Ok(Frame::ContextCall {
            call: required_uint(fields, "call")?,
            id: required_uint(fields, "id")?,
            method: required_text(fields, "method")?.to_string(),
            params: optional_json(fields, "params")?,
        }),
        "context_result" => Ok(Frame::ContextResult {
            call: required_uint(fields, "call")?,
            id: required_uint(fields, "id")?,
            value: required_json(fields, "value")?,
        }),
        "context_error" => Ok(Frame::ContextError {
            call: required_uint(fields, "call")?,
            id: required_uint(fields, "id")?,
            code: required_text(fields, "code")?.to_string(),
            message: required_text(fields, "message")?.to_string(),
        }),
        "ui_call" => Ok(Frame::UiCall {
            call: required_uint(fields, "call")?,
            id: required_uint(fields, "id")?,
            method: required_text(fields, "method")?.to_string(),
            params: optional_json(fields, "params")?,
        }),
        "ui_result" => Ok(Frame::UiResult {
            call: required_uint(fields, "call")?,
            id: required_uint(fields, "id")?,
            value: required_json(fields, "value")?,
        }),
        "ui_error" => Ok(Frame::UiError {
            call: required_uint(fields, "call")?,
            id: required_uint(fields, "id")?,
            code: required_text(fields, "code")?.to_string(),
            message: required_text(fields, "message")?.to_string(),
        }),
        "signal" => Ok(Frame::Signal {
            call: required_uint(fields, "call")?,
            signal: required_uint(fields, "signal")?,
            aborted: required_bool(fields, "aborted")?,
        }),
        other => Err(WireError::new(format!(
            "unknown extension frame type {other:?}"
        ))),
    }
}

fn decode_identity(value: &CborValue) -> Result<ExtensionIdentity, WireError> {
    let fields = as_fields(value, "extension identity")?;
    Ok(ExtensionIdentity {
        name: required_text(fields, "name")?.to_string(),
        version: required_text(fields, "version")?.to_string(),
    })
}

fn as_fields<'a>(value: &'a CborValue, what: &str) -> Result<&'a [(String, CborValue)], WireError> {
    match value {
        CborValue::Map(entries) => Ok(entries),
        _ => Err(WireError::new(format!(
            "the extension {what} must be a CBOR map"
        ))),
    }
}

fn required<'a>(fields: &'a [(String, CborValue)], name: &str) -> Result<&'a CborValue, WireError> {
    fields
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
        .ok_or_else(|| WireError::new(format!("extension frame is missing the {name:?} field")))
}

fn optional<'a>(fields: &'a [(String, CborValue)], name: &str) -> Option<&'a CborValue> {
    fields
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}

fn required_text<'a>(fields: &'a [(String, CborValue)], name: &str) -> Result<&'a str, WireError> {
    match required(fields, name)? {
        CborValue::Text(value) => Ok(value),
        _ => Err(WireError::new(format!(
            "extension frame field {name:?} must be a text string"
        ))),
    }
}

fn optional_text(fields: &[(String, CborValue)], name: &str) -> Result<Option<String>, WireError> {
    optional(fields, name)
        .map(|value| match value {
            CborValue::Text(text) => Ok(text.clone()),
            _ => Err(WireError::new(format!(
                "extension frame field {name:?} must be a text string"
            ))),
        })
        .transpose()
}

fn required_uint(fields: &[(String, CborValue)], name: &str) -> Result<u64, WireError> {
    match required(fields, name)? {
        CborValue::Int(integer) => u64::try_from(*integer).map_err(|_| {
            WireError::new(format!(
                "extension frame field {name:?} must be a non-negative integer"
            ))
        }),
        _ => Err(WireError::new(format!(
            "extension frame field {name:?} must be a non-negative integer"
        ))),
    }
}

fn required_bool(fields: &[(String, CborValue)], name: &str) -> Result<bool, WireError> {
    match required(fields, name)? {
        CborValue::Bool(value) => Ok(*value),
        _ => Err(WireError::new(format!(
            "extension frame field {name:?} must be a boolean"
        ))),
    }
}

fn required_json(
    fields: &[(String, CborValue)],
    name: &str,
) -> Result<serde_json::Value, WireError> {
    cbor_to_json(required(fields, name)?)
}

fn optional_json(
    fields: &[(String, CborValue)],
    name: &str,
) -> Result<Option<serde_json::Value>, WireError> {
    optional(fields, name).map(cbor_to_json).transpose()
}
