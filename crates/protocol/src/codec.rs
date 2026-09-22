//! The validated message codec, ported from upstream `src/codec.ts`.
//!
//! Upstream validates envelope shapes with typebox `Check` and gates opaque
//! payloads with chord's `isJsonValue`; the port restates both walks over
//! the decoded CBOR tree. The `isJsonValue` walk becomes the
//! `json_from_cbor` conversion: a byte string is the one non-JSON value the
//! CBOR decoder can produce, non-finite numbers die at [`JsonNumber::new`],
//! and the depth cap is [`pi_chord::json::MAX_DEPTH`] — everything else the
//! gate polices (exotic prototypes, sparse arrays, cycles) is
//! unrepresentable in the owned tree.

use std::fmt;

use pi_chord::json::MAX_DEPTH;
use pi_chord::types::{JsonNumber, JsonObject, JsonValue};

use crate::cbor::{CborOptions, CborValue, decode_cbor, encode_cbor};
use crate::framing::{FrameDecoder, FrameDecoderOptions, FrameError, encode_frame};
use crate::protocol::{
    AttachmentEnvelope, CancelEnvelope, ClientHello, ClientMessage, PROTOCOL_VERSION,
    ProtocolError, ProtocolErrorCode, RequestEnvelope, ResponseEnvelope, ResponseFailure,
    ResponseSuccess, RpcTarget, ServerHello, ServerHelloError, ServerId, ServerMessage,
    ServerTarget, ServiceEventEnvelope, SessionTarget,
};

/// The error the validation and codec layers raise, ported from upstream's
/// `ProtocolValidationError`.
///
/// Upstream extends `Error` with `name = "ProtocolValidationError"`; the
/// distinct type is the same discrimination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolValidationError {
    message: String,
}

impl ProtocolValidationError {
    /// Builds the error from its message, the constructor surface upstream
    /// client code reaches for when it re-raises validation failures.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The error message, the upstream `Error.message` surface.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ProtocolValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ProtocolValidationError {}

/// Upstream bounds embedded codec errors at 500 characters before embedding
/// them in a codec message.
fn bounded_message(error: impl fmt::Display) -> String {
    let message = error.to_string();
    if message.chars().count() <= 500 {
        return message;
    }
    let mut bounded: String = message.chars().take(497).collect();
    bounded.push_str("...");
    bounded
}

/// Converts one decoded CBOR item to the strict-JSON value the envelope
/// schemas validate, the upstream `isJsonValue` gate restated over the wire
/// shape.
fn json_from_cbor(value: &CborValue, depth: usize) -> Option<JsonValue> {
    if depth > MAX_DEPTH {
        return None;
    }
    Some(match value {
        CborValue::Null => JsonValue::Null,
        CborValue::Bool(flag) => JsonValue::Bool(*flag),
        CborValue::Int(integer) => JsonValue::Number(JsonNumber::from(*integer)),
        CborValue::Float(number) => JsonValue::Number(JsonNumber::new(*number)?),
        CborValue::Text(text) => JsonValue::Str(text.clone()),
        CborValue::Bytes(_) => return None,
        CborValue::Array(items) => {
            let mut converted = Vec::with_capacity(items.len());
            for item in items {
                converted.push(json_from_cbor(item, depth + 1)?);
            }
            JsonValue::Array(converted)
        }
        CborValue::Map(entries) => {
            let mut converted = JsonObject::default();
            for (key, item) in entries {
                converted.set(key.clone(), json_from_cbor(item, depth + 1)?);
            }
            JsonValue::Object(converted)
        }
    })
}

/// Converts one owned JSON value to the CBOR tree the encoder writes.
///
/// Upstream's encoder picks the integer path when `Number.isInteger` holds
/// and the value is not `-0`; [`CborValue::Float`] carries the number
/// through, because the encoder applies that same rule to floats.
fn json_to_cbor(value: &JsonValue) -> CborValue {
    match value {
        JsonValue::Null => CborValue::Null,
        JsonValue::Bool(flag) => CborValue::Bool(*flag),
        JsonValue::Number(number) => CborValue::Float(number.get()),
        JsonValue::Str(text) => CborValue::Text(text.clone()),
        JsonValue::Array(items) => CborValue::Array(items.iter().map(json_to_cbor).collect()),
        JsonValue::Object(object) => CborValue::Map(
            object
                .iter()
                .map(|(key, value)| (key.to_string(), json_to_cbor(value)))
                .collect(),
        ),
    }
}

fn text(value: &str) -> CborValue {
    CborValue::Text(value.to_string())
}

fn protocol_error_to_value(error: &ProtocolError) -> CborValue {
    CborValue::map(vec![
        ("code", text(&error.code)),
        ("message", text(&error.message)),
    ])
}

fn rpc_target_to_value(target: &RpcTarget) -> CborValue {
    match target {
        RpcTarget::Server(target) => {
            CborValue::map(vec![("serverId", text(target.server_id.as_str()))])
        }
        RpcTarget::Session(target) => session_target_to_value(target),
    }
}

fn session_target_to_value(target: &SessionTarget) -> CborValue {
    CborValue::map(vec![
        ("serverId", text(target.server_id.as_str())),
        ("sessionId", text(&target.session_id)),
        ("attachmentId", text(&target.attachment_id)),
    ])
}

fn client_message_to_value(message: &ClientMessage) -> CborValue {
    match message {
        ClientMessage::Hello(hello) => CborValue::map(vec![
            ("type", text("hello")),
            ("version", CborValue::Float(hello.version.get())),
        ]),
        ClientMessage::Request(request) => CborValue::map(vec![
            ("type", text("request")),
            ("id", text(&request.id)),
            ("target", rpc_target_to_value(&request.target)),
            ("call", json_to_cbor(&request.call)),
        ]),
        ClientMessage::Cancel(cancel) => CborValue::map(vec![
            ("type", text("cancel")),
            ("id", text(&cancel.id)),
            ("target", rpc_target_to_value(&cancel.target)),
        ]),
    }
}

fn server_message_to_value(message: &ServerMessage) -> CborValue {
    match message {
        ServerMessage::Hello(hello) => {
            let version = PROTOCOL_VERSION.cast_signed();
            CborValue::map(vec![
                ("type", text("hello")),
                ("version", CborValue::Int(version)),
                ("serverId", text(hello.server_id.as_str())),
            ])
        }
        ServerMessage::HelloError(hello_error) => CborValue::map(vec![
            ("type", text("hello_error")),
            ("error", protocol_error_to_value(&hello_error.error)),
        ]),
        ServerMessage::Response(ResponseEnvelope::Success(success)) => {
            let mut entries = vec![
                ("type", text("response")),
                ("id", text(&success.id)),
                ("ok", CborValue::Bool(true)),
            ];
            if let Some(result) = &success.result {
                entries.push(("result", json_to_cbor(result)));
            }
            CborValue::map(entries)
        }
        ServerMessage::Response(ResponseEnvelope::Failure(failure)) => CborValue::map(vec![
            ("type", text("response")),
            ("id", text(&failure.id)),
            ("ok", CborValue::Bool(false)),
            ("error", protocol_error_to_value(&failure.error)),
        ]),
        ServerMessage::ServiceEvent(event) => CborValue::map(vec![
            ("type", text("service_update")),
            ("subscriptionId", text(&event.subscription_id)),
            ("update", json_to_cbor(&event.update)),
        ]),
        ServerMessage::Attachment(attachment) => CborValue::map(vec![
            ("type", text("attachment")),
            (
                "attachment",
                attachment
                    .attachment
                    .as_ref()
                    .map_or(CborValue::Null, session_target_to_value),
            ),
        ]),
    }
}

fn strict_keys(object: &JsonObject, expected: &[&str]) -> bool {
    object.len() == expected.len() && expected.iter().all(|key| object.contains_key(key))
}

fn string_field<'a>(object: &'a JsonObject, key: &str) -> Option<&'a str> {
    object.get(key)?.as_str()
}

fn non_empty_field<'a>(object: &'a JsonObject, key: &str) -> Option<&'a str> {
    let value = string_field(object, key)?;
    if value.is_empty() { None } else { Some(value) }
}

fn protocol_error_from_value(value: &JsonValue) -> Option<ProtocolError> {
    let object = value.as_object()?;
    if !strict_keys(object, &["code", "message"]) {
        return None;
    }
    Some(ProtocolError {
        code: ProtocolErrorCode::from(non_empty_field(object, "code")?),
        message: string_field(object, "message")?.to_string(),
    })
}

fn session_target_from_value(value: &JsonValue) -> Option<SessionTarget> {
    let object = value.as_object()?;
    if !strict_keys(object, &["serverId", "sessionId", "attachmentId"]) {
        return None;
    }
    Some(SessionTarget {
        server_id: ServerId::new(non_empty_field(object, "serverId")?)?,
        session_id: non_empty_field(object, "sessionId")?.to_string(),
        attachment_id: non_empty_field(object, "attachmentId")?.to_string(),
    })
}

fn rpc_target_from_value(value: &JsonValue) -> Option<RpcTarget> {
    let object = value.as_object()?;
    if strict_keys(object, &["serverId"]) {
        return Some(RpcTarget::Server(ServerTarget {
            server_id: ServerId::new(non_empty_field(object, "serverId")?)?,
        }));
    }
    Some(RpcTarget::Session(session_target_from_value(value)?))
}

/// The client hello's negotiating version, the upstream
/// `Type.Integer({ minimum: 0 })` check.
fn client_version(value: &JsonValue) -> Option<JsonNumber> {
    let JsonValue::Number(version) = value else {
        return None;
    };
    if version.get() < 0.0 || version.get().fract() != 0.0 {
        return None;
    }
    Some(*version)
}

fn validate_client_message(value: &JsonValue) -> Option<ClientMessage> {
    let object = value.as_object()?;
    match string_field(object, "type")? {
        "hello" => {
            if !strict_keys(object, &["type", "version"]) {
                return None;
            }
            Some(ClientMessage::Hello(ClientHello {
                version: client_version(object.get("version")?)?,
            }))
        }
        "request" => {
            if !strict_keys(object, &["type", "id", "target", "call"]) {
                return None;
            }
            Some(ClientMessage::Request(RequestEnvelope {
                id: non_empty_field(object, "id")?.to_string(),
                target: rpc_target_from_value(object.get("target")?)?,
                call: object.get("call")?.clone(),
            }))
        }
        "cancel" => {
            if !strict_keys(object, &["type", "id", "target"]) {
                return None;
            }
            Some(ClientMessage::Cancel(CancelEnvelope {
                id: non_empty_field(object, "id")?.to_string(),
                target: rpc_target_from_value(object.get("target")?)?,
            }))
        }
        _ => None,
    }
}

fn validate_server_message(value: &JsonValue) -> Option<ServerMessage> {
    let object = value.as_object()?;
    match string_field(object, "type")? {
        "hello" => {
            if !strict_keys(object, &["type", "version", "serverId"]) {
                return None;
            }
            let JsonValue::Number(version) = object.get("version")? else {
                return None;
            };
            #[allow(
                clippy::cast_precision_loss,
                clippy::float_cmp,
                reason = "the protocol version is exactly representable as f64; the schema compares literal equality, so a margin would be wrong"
            )]
            if version.get() != PROTOCOL_VERSION as f64 {
                return None;
            }
            Some(ServerMessage::Hello(ServerHello {
                server_id: ServerId::new(non_empty_field(object, "serverId")?)?,
            }))
        }
        "hello_error" => {
            if !strict_keys(object, &["type", "error"]) {
                return None;
            }
            Some(ServerMessage::HelloError(ServerHelloError {
                error: protocol_error_from_value(object.get("error")?)?,
            }))
        }
        "response" => {
            let flag = object.get("ok")?.as_bool()?;
            let id = non_empty_field(object, "id")?.to_string();
            if flag {
                // `ok: true` allows `result`, optionally, and nothing else.
                let with_result = strict_keys(object, &["type", "id", "ok", "result"]);
                if !with_result && !strict_keys(object, &["type", "id", "ok"]) {
                    return None;
                }
                Some(ServerMessage::Response(ResponseEnvelope::Success(
                    ResponseSuccess {
                        id,
                        result: object.get("result").cloned(),
                    },
                )))
            } else {
                if !strict_keys(object, &["type", "id", "ok", "error"]) {
                    return None;
                }
                Some(ServerMessage::Response(ResponseEnvelope::Failure(
                    ResponseFailure {
                        id,
                        error: protocol_error_from_value(object.get("error")?)?,
                    },
                )))
            }
        }
        "service_update" => {
            if !strict_keys(object, &["type", "subscriptionId", "update"]) {
                return None;
            }
            Some(ServerMessage::ServiceEvent(ServiceEventEnvelope {
                subscription_id: non_empty_field(object, "subscriptionId")?.to_string(),
                update: object.get("update")?.clone(),
            }))
        }
        "attachment" => {
            if !strict_keys(object, &["type", "attachment"]) {
                return None;
            }
            let attachment = match object.get("attachment")? {
                JsonValue::Null => None,
                target @ JsonValue::Object(_) => Some(session_target_from_value(target)?),
                _ => return None,
            };
            Some(ServerMessage::Attachment(AttachmentEnvelope { attachment }))
        }
        _ => None,
    }
}

fn parse_message<M>(
    value: &CborValue,
    validate: fn(&JsonValue) -> Option<M>,
    kind: &str,
) -> Result<M, ProtocolValidationError> {
    let invalid = || ProtocolValidationError::new(format!("Invalid {kind} protocol message"));
    let json = json_from_cbor(value, 0).ok_or_else(invalid)?;
    validate(&json).ok_or_else(invalid)
}

/// Validates one decoded client message against the envelope schemas.
///
/// # Errors
///
/// Returns [`ProtocolValidationError`] for any value outside the client
/// schema — unknown or missing fields, wrong literal types, empty ids,
/// non-canonical server ids, non-integer versions, or a payload that is not
/// strict JSON.
pub fn parse_client_message(value: &CborValue) -> Result<ClientMessage, ProtocolValidationError> {
    parse_message(value, validate_client_message, "client")
}

/// Validates one decoded server message against the envelope schemas.
///
/// # Errors
///
/// Returns [`ProtocolValidationError`] for any value outside the server
/// schema — unknown or missing fields, wrong literal types, empty ids or
/// error codes, a server id or version that fails the handshake shape, or a
/// payload that is not strict JSON.
pub fn parse_server_message(value: &CborValue) -> Result<ServerMessage, ProtocolValidationError> {
    parse_message(value, validate_server_message, "server")
}

fn cbor_limits(max_frame_length: usize) -> CborOptions {
    CborOptions {
        max_byte_length: max_frame_length,
        ..CborOptions::default()
    }
}

fn encode_protocol_message<M>(
    value: &CborValue,
    parse: fn(&CborValue) -> Result<M, ProtocolValidationError>,
    kind: &str,
    options: FrameDecoderOptions,
) -> Result<Vec<u8>, ProtocolValidationError> {
    // Upstream validates first and encodes the validated value; the built
    // tree is that value, so the same parse runs over it.
    parse(value)?;
    let unable = |error: &str| {
        ProtocolValidationError::new(format!("Unable to encode {kind} protocol message: {error}"))
    };
    let payload = encode_cbor(value, &cbor_limits(options.max_frame_length))
        .map_err(|error| unable(&bounded_message(error)))?;
    encode_frame(&payload).map_err(|error| unable(&bounded_message(error)))
}

/// Validates and encodes one complete length-prefixed client message.
///
/// # Errors
///
/// Returns [`ProtocolValidationError`] when the message fails the client
/// schema, or when the encoded bytes or frame exceed `options.max_frame_length`.
pub fn encode_client_message(
    message: &ClientMessage,
    options: FrameDecoderOptions,
) -> Result<Vec<u8>, ProtocolValidationError> {
    encode_protocol_message(
        &client_message_to_value(message),
        parse_client_message,
        "client",
        options,
    )
}

/// Validates and encodes one complete length-prefixed server message.
///
/// # Errors
///
/// Returns [`ProtocolValidationError`] when the message fails the server
/// schema, or when the encoded bytes or frame exceed `options.max_frame_length`.
pub fn encode_server_message(
    message: &ServerMessage,
    options: FrameDecoderOptions,
) -> Result<Vec<u8>, ProtocolValidationError> {
    encode_protocol_message(
        &server_message_to_value(message),
        parse_server_message,
        "server",
        options,
    )
}

#[derive(Debug)]
struct ValidatedMessageDecoder<M> {
    frames: FrameDecoder,
    failed: bool,
    kind: &'static str,
    max_frame_length: usize,
    parse: fn(&CborValue) -> Result<M, ProtocolValidationError>,
}

impl<M> ValidatedMessageDecoder<M> {
    fn new(
        kind: &'static str,
        parse: fn(&CborValue) -> Result<M, ProtocolValidationError>,
        options: FrameDecoderOptions,
    ) -> Result<Self, FrameError> {
        Ok(Self {
            frames: FrameDecoder::new(options)?,
            failed: false,
            kind,
            max_frame_length: options.max_frame_length,
            parse,
        })
    }

    fn push(&mut self, chunk: &[u8]) -> Result<Vec<M>, ProtocolValidationError> {
        if self.failed {
            return Err(ProtocolValidationError::new(format!(
                "{} message decoder has failed",
                self.kind
            )));
        }
        match self.push_inner(chunk) {
            Ok(messages) => Ok(messages),
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    fn push_inner(&mut self, chunk: &[u8]) -> Result<Vec<M>, ProtocolValidationError> {
        let frame_error = |error: &str| {
            ProtocolValidationError::new(format!("Invalid {} protocol frame: {error}", self.kind))
        };
        let frames = self
            .frames
            .push(chunk)
            .map_err(|error| frame_error(&bounded_message(error)))?;
        let mut messages = Vec::new();
        for frame in frames {
            let value = decode_cbor(&frame, &cbor_limits(self.max_frame_length))
                .map_err(|error| frame_error(&bounded_message(error)))?;
            messages.push((self.parse)(&value)?);
        }
        Ok(messages)
    }

    fn end(&mut self) -> Result<(), ProtocolValidationError> {
        if self.failed {
            return Err(ProtocolValidationError::new(format!(
                "{} message decoder has failed",
                self.kind
            )));
        }
        match self.frames.end() {
            Ok(()) => Ok(()),
            Err(error) => {
                self.failed = true;
                Err(ProtocolValidationError::new(format!(
                    "Invalid {} protocol framing: {}",
                    self.kind,
                    bounded_message(error)
                )))
            }
        }
    }
}

/// Incrementally decodes and validates framed client messages.
///
/// The decoder latches the failed state: after any [`push`](Self::push) or
/// [`end`](Self::end) error, every later call fails too.
#[derive(Debug)]
pub struct ClientMessageDecoder(ValidatedMessageDecoder<ClientMessage>);

impl ClientMessageDecoder {
    /// A decoder bounded by `options.max_frame_length`.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError`] when `max_frame_length` exceeds the unsigned
    /// 32-bit range; upstream propagates the frame decoder's `RangeError`
    /// unwrapped.
    pub fn new(options: FrameDecoderOptions) -> Result<Self, FrameError> {
        Ok(Self(ValidatedMessageDecoder::new(
            "client",
            parse_client_message,
            options,
        )?))
    }

    /// Feeds one chunk, returning the messages it completes in order.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolValidationError`] for framing, CBOR, or schema
    /// failures and latches the failed state.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<ClientMessage>, ProtocolValidationError> {
        self.0.push(chunk)
    }

    /// Ends the stream, rejecting truncated framing.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolValidationError`] when the decoder has failed or
    /// when a partial frame remains; those latches stay failed.
    pub fn end(&mut self) -> Result<(), ProtocolValidationError> {
        self.0.end()
    }
}

/// Incrementally decodes and validates framed server messages.
///
/// The decoder latches the failed state: after any [`push`](Self::push) or
/// [`end`](Self::end) error, every later call fails too.
#[derive(Debug)]
pub struct ServerMessageDecoder(ValidatedMessageDecoder<ServerMessage>);

impl ServerMessageDecoder {
    /// A decoder bounded by `options.max_frame_length`.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError`] when `max_frame_length` exceeds the unsigned
    /// 32-bit range; upstream propagates the frame decoder's `RangeError`
    /// unwrapped.
    pub fn new(options: FrameDecoderOptions) -> Result<Self, FrameError> {
        Ok(Self(ValidatedMessageDecoder::new(
            "server",
            parse_server_message,
            options,
        )?))
    }

    /// Feeds one chunk, returning the messages it completes in order.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolValidationError`] for framing, CBOR, or schema
    /// failures and latches the failed state.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<ServerMessage>, ProtocolValidationError> {
        self.0.push(chunk)
    }

    /// Ends the stream, rejecting truncated framing.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolValidationError`] when the decoder has failed or
    /// when a partial frame remains; those latches stay failed.
    pub fn end(&mut self) -> Result<(), ProtocolValidationError> {
        self.0.end()
    }
}

/// Whether the version is exactly [`PROTOCOL_VERSION`], the upstream
/// `isSupportedProtocolVersion` check.
///
/// Upstream takes any JavaScript number; the owned finite-number type makes
/// non-finite inputs unrepresentable while `8.5` stays false.
#[must_use]
#[allow(
    clippy::float_cmp,
    reason = "the schema compares literal equality against the protocol version, so a margin would be wrong"
)]
pub fn is_supported_protocol_version(version: JsonNumber) -> bool {
    #[allow(
        clippy::cast_precision_loss,
        reason = "the protocol version is exactly representable as f64"
    )]
    let protocol_version = PROTOCOL_VERSION as f64;
    version.get() == protocol_version
}
