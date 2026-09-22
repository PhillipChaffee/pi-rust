//! The routed envelope schemas, ported from upstream `src/protocol.ts`.
//!
//! Upstream builds the schemas with typebox and validates with `Check`; the
//! port restates each schema as an owned type whose fields carry the wire's
//! name in their documentation, with the validation surface living in
//! [`crate::codec::parse_client_message`] and
//! [`crate::codec::parse_server_message`]. The schemas deliberately
//! duplicate the agent-side wire vocabulary upstream; no shared module
//! deduplicates them ([ADR 0003](../../docs/adr/0003-crate-graph-and-porting-route.md)).

use std::fmt;

use pi_chord::types::{JsonNumber, JsonValue};

/// The protocol version a client hello negotiates with and a server hello
/// must answer exactly.
pub const PROTOCOL_VERSION: u64 = 8;

fn is_canonical_uuid_v4(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    let hex = |byte: u8| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f');
    bytes.iter().enumerate().all(|(index, byte)| match index {
        8 | 13 | 18 | 23 => *byte == b'-',
        14 => *byte == b'4',
        19 => matches!(*byte, b'8' | b'9' | b'a' | b'b'),
        _ => hex(*byte),
    })
}

/// The canonical lowercase UUIDv4 identifier a routed target fences to, the
/// upstream `ServerId` schema.
///
/// The pattern upstream compiles into the schema —
/// `^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$` —
/// is the constructor's only admission rule, so a non-canonical id is
/// unrepresentable once parsed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ServerId(String);

impl ServerId {
    /// Parses a canonical UUIDv4 server id, or [`None`] for any other text.
    #[must_use]
    pub fn new(value: &str) -> Option<Self> {
        if is_canonical_uuid_v4(value) {
            Some(Self(value.to_string()))
        } else {
            None
        }
    }

    /// The canonical spelling the wire carries.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ServerId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Whether the text is a canonical UUIDv4 server id, the upstream
/// `isServerId` check.
#[must_use]
pub fn is_server_id(value: &str) -> bool {
    is_canonical_uuid_v4(value)
}

/// The opaque error code string the wire carries, the upstream
/// `ProtocolErrorCode` alias for `string`.
pub type ProtocolErrorCode = String;

/// The error code and message a server reports, ported from upstream's
/// `ProtocolError`.
///
/// `code` is the opaque code the other side sends (`wrong_server`,
/// `cancelled`, `service_not_found`, `application_error`, ...); the
/// vocabulary belongs to chord, and the protocol only requires a non-empty
/// string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolError {
    /// The opaque error code the other side sends; the wire requires at
    /// least one character.
    pub code: ProtocolErrorCode,
    /// The human-readable message the other side sends.
    pub message: String,
}

/// A client's opening frame, which must be the first one sent; any protocol
/// version from `0` up negotiates here, and the server answers with its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientHello {
    /// The client's protocol version, a non-negative integer on the wire.
    pub version: JsonNumber,
}

/// A server-wide call, fenced to one logical server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerTarget {
    /// The logical server the call fences to.
    pub server_id: ServerId,
}

/// A session call, fenced to one logical server, durable session, and live
/// attachment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionTarget {
    /// The logical server the call fences to.
    pub server_id: ServerId,
    /// The durable session id the other side assigned.
    pub session_id: String,
    /// The live attachment id the other side assigned.
    pub attachment_id: String,
}

/// The routed target of a request or cancel, the upstream `RpcTarget`
/// union: a server-wide call or a session call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcTarget {
    /// A server-wide call.
    Server(ServerTarget),
    /// A session call.
    Session(SessionTarget),
}

/// A routed call envelope; the `call` payload stays opaque strict JSON whose
/// service meaning belongs to chord.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestEnvelope {
    /// The correlation id the client assigned; the wire requires at least
    /// one character.
    pub id: String,
    /// Where the call routes.
    pub target: RpcTarget,
    /// The opaque service call, strict JSON end to end.
    pub call: JsonValue,
}

/// A cancellation for a previously sent request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancelEnvelope {
    /// The correlation id of the request being cancelled; the wire requires
    /// at least one character.
    pub id: String,
    /// Where the cancelled call routes.
    pub target: RpcTarget,
}

/// One message a client may send, discriminated by the wire's `type` field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientMessage {
    /// The opening [`ClientHello`], which must be the first frame sent.
    Hello(ClientHello),
    /// A routed service call.
    Request(RequestEnvelope),
    /// A cancellation of an earlier request.
    Cancel(CancelEnvelope),
}

/// A server's answer to a matching-version client hello; the version is the
/// fixed [`PROTOCOL_VERSION`] literal on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerHello {
    /// The logical server that accepted the connection.
    pub server_id: ServerId,
}

/// A server's refusal of a handshake it cannot accept; the connection
/// closes after this frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerHelloError {
    /// Why the handshake failed.
    pub error: ProtocolError,
}

/// A successful routed call's answer, the `ok: true` wire shape; `result`
/// stays absent for a void response and carries opaque strict JSON
/// otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseSuccess {
    /// The correlation id of the answered request.
    pub id: String,
    /// The opaque result, absent when the call returned nothing.
    pub result: Option<JsonValue>,
}

/// A failed routed call's answer, the `ok: false` wire shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseFailure {
    /// The correlation id of the answered request.
    pub id: String,
    /// Why the call failed.
    pub error: ProtocolError,
}

/// The two response shapes, discriminated by the wire's `ok` flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseEnvelope {
    /// `ok: true`, with an optional opaque result.
    Success(ResponseSuccess),
    /// `ok: false`, with the server's error.
    Failure(ResponseFailure),
}

/// An out-of-band update to a subscription's replicated state; the update
/// payload stays opaque strict JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceEventEnvelope {
    /// The subscription the update belongs to; the wire requires at least
    /// one character.
    pub subscription_id: String,
    /// The opaque subscription update, strict JSON end to end.
    pub update: JsonValue,
}

/// An out-of-band update to this presentation's selected session route;
/// [`None`] detaches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachmentEnvelope {
    /// The session route now selected, or [`None`] when detached.
    pub attachment: Option<SessionTarget>,
}

/// One message a server may send, discriminated by the wire's `type` field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerMessage {
    /// The handshake answer with a matching server id.
    Hello(ServerHello),
    /// A handshake rejection; the connection closes after it.
    HelloError(ServerHelloError),
    /// A routed call's answer.
    Response(ResponseEnvelope),
    /// An out-of-band subscription update.
    ServiceEvent(ServiceEventEnvelope),
    /// An out-of-band attachment route update.
    Attachment(AttachmentEnvelope),
}
