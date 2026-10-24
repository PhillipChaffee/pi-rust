//! [`encode_frame`]: one [`Frame`] to a length-prefixed pi-protocol frame.
//!
//! The payload runs under pi-protocol's 16 MiB cap and rejection surface;
//! this layer adds the envelope's own range check on ids and the JSON
//! bridge's number mapping.

use pi_protocol::{CborOptions, CborValue, encode_cbor};

use super::WireError;
use super::bridge::json_to_cbor;
use super::frame::Frame;

/// Encodes one frame into a length-prefixed pi-protocol frame.
///
/// # Errors
///
/// Returns [`WireError`] when a field or payload number is outside the
/// codec's range, or when the payload exceeds pi-protocol's frame cap.
pub fn encode_frame(frame: &Frame) -> Result<Vec<u8>, WireError> {
    let value = frame_to_cbor(frame)?;
    let payload = encode_cbor(&value, &CborOptions::default())?;
    pi_protocol::encode_frame(&payload).map_err(|error| WireError::new(error.message()))
}

fn text(value: &str) -> CborValue {
    CborValue::Text(value.to_string())
}

fn uint(value: u64) -> Result<CborValue, WireError> {
    i64::try_from(value)
        .map(CborValue::Int)
        .map_err(|_| WireError::new(format!("extension frame id {value} exceeds the wire range")))
}

fn envelope(kind: &str, fields: Vec<(&str, CborValue)>) -> CborValue {
    let mut entries = Vec::with_capacity(fields.len() + 1);
    entries.push(("type".to_string(), text(kind)));
    entries.extend(
        fields
            .into_iter()
            .map(|(name, value)| (name.to_string(), value)),
    );
    CborValue::Map(entries)
}

fn scope_call(
    kind: &str,
    call: u64,
    id: u64,
    method: &str,
    params: Option<&serde_json::Value>,
) -> Result<CborValue, WireError> {
    let mut fields = vec![
        ("call", uint(call)?),
        ("id", uint(id)?),
        ("method", text(method)),
    ];
    if let Some(params) = params {
        fields.push(("params", json_to_cbor(params)));
    }
    Ok(envelope(kind, fields))
}

fn scope_value(
    kind: &str,
    call: u64,
    id: u64,
    value: &serde_json::Value,
) -> Result<CborValue, WireError> {
    Ok(envelope(
        kind,
        vec![
            ("call", uint(call)?),
            ("id", uint(id)?),
            ("value", json_to_cbor(value)),
        ],
    ))
}

fn scope_error(
    kind: &str,
    call: u64,
    id: u64,
    code: &str,
    message: &str,
) -> Result<CborValue, WireError> {
    Ok(envelope(
        kind,
        vec![
            ("call", uint(call)?),
            ("id", uint(id)?),
            ("code", text(code)),
            ("message", text(message)),
        ],
    ))
}

#[expect(
    clippy::too_many_lines,
    reason = "the frame table is one match, the wire's complete index"
)]
fn frame_to_cbor(frame: &Frame) -> Result<CborValue, WireError> {
    Ok(match frame {
        Frame::Hello {
            protocol_version,
            extension,
        } => envelope(
            "hello",
            vec![
                ("protocol_version", uint(*protocol_version)?),
                (
                    "extension",
                    CborValue::map(vec![
                        ("name", text(&extension.name)),
                        ("version", text(&extension.version)),
                    ]),
                ),
            ],
        ),
        Frame::Welcome { protocol_version } => envelope(
            "welcome",
            vec![("protocol_version", uint(*protocol_version)?)],
        ),
        Frame::HelloError { code, message } => envelope(
            "hello_error",
            vec![("code", text(code)), ("message", text(message))],
        ),
        Frame::Ready => envelope("ready", Vec::new()),
        Frame::RegisterEvent { channel } => {
            envelope("register_event", vec![("channel", text(channel))])
        }
        Frame::RegisterTool { tool } => {
            envelope("register_tool", vec![("tool", json_to_cbor(tool))])
        }
        Frame::RegisterCommand { command } => {
            envelope("register_command", vec![("command", json_to_cbor(command))])
        }
        Frame::RegisterFlag { flag } => {
            envelope("register_flag", vec![("flag", json_to_cbor(flag))])
        }
        Frame::RegisterProvider { provider } => envelope(
            "register_provider",
            vec![("provider", json_to_cbor(provider))],
        ),
        Frame::CallEvent {
            call,
            channel,
            payload,
        } => envelope(
            "call_event",
            vec![
                ("call", uint(*call)?),
                ("channel", text(channel)),
                ("payload", json_to_cbor(payload)),
            ],
        ),
        Frame::CallTool {
            call,
            tool,
            tool_call_id,
            params,
            context,
        } => envelope(
            "call_tool",
            vec![
                ("call", uint(*call)?),
                ("tool", text(tool)),
                ("tool_call_id", text(tool_call_id)),
                ("params", json_to_cbor(params)),
                ("context", json_to_cbor(context)),
            ],
        ),
        Frame::CallUpdate { call, update } => envelope(
            "call_update",
            vec![("call", uint(*call)?), ("update", json_to_cbor(update))],
        ),
        Frame::CallResult { call, result } => envelope(
            "call_result",
            vec![("call", uint(*call)?), ("result", json_to_cbor(result))],
        ),
        Frame::CallError {
            call,
            code,
            message,
            stack,
        } => {
            let mut fields = vec![
                ("call", uint(*call)?),
                ("code", text(code)),
                ("message", text(message)),
            ];
            if let Some(stack) = stack {
                fields.push(("stack", text(stack)));
            }
            envelope("call_error", fields)
        }
        Frame::CallCommand { call, name, args } => envelope(
            "call_command",
            vec![
                ("call", uint(*call)?),
                ("name", text(name)),
                ("args", text(args)),
            ],
        ),
        Frame::CallProvider {
            call,
            provider,
            request,
        } => envelope(
            "call_provider",
            vec![
                ("call", uint(*call)?),
                ("provider", text(provider)),
                ("request", json_to_cbor(request)),
            ],
        ),
        Frame::StreamEvent { call, event } => envelope(
            "stream_event",
            vec![("call", uint(*call)?), ("event", json_to_cbor(event))],
        ),
        Frame::StreamEnd { call } => envelope("stream_end", vec![("call", uint(*call)?)]),
        Frame::ContextCall {
            call,
            id,
            method,
            params,
        } => scope_call("context_call", *call, *id, method, params.as_ref())?,
        Frame::ContextResult { call, id, value } => {
            scope_value("context_result", *call, *id, value)?
        }
        Frame::ContextError {
            call,
            id,
            code,
            message,
        } => scope_error("context_error", *call, *id, code, message)?,
        Frame::UiCall {
            call,
            id,
            method,
            params,
        } => scope_call("ui_call", *call, *id, method, params.as_ref())?,
        Frame::UiResult { call, id, value } => scope_value("ui_result", *call, *id, value)?,
        Frame::UiError {
            call,
            id,
            code,
            message,
        } => scope_error("ui_error", *call, *id, code, message)?,
        Frame::Signal {
            call,
            signal,
            aborted,
        } => envelope(
            "signal",
            vec![
                ("call", uint(*call)?),
                ("signal", uint(*signal)?),
                ("aborted", CborValue::Bool(*aborted)),
            ],
        ),
    })
}
