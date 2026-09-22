//! Rust port of `packages/protocol` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The transport-neutral wire for remote pi sessions: a 4-byte big-endian
//! length-prefixed frame carries exactly one definite-length RFC 8949 CBOR
//! item ([`framing`]), and the framed payload validates as one of a small
//! set of routed envelope schemas ([`protocol`], [`codec`]). The hand-ported
//! codec ([`cbor`]) reproduces upstream's byte-exact encoding and full
//! rejection surface — indefinite lengths, tags, break markers,
//! float16/float32, non-finite numbers, `undefined`, duplicate keys, and
//! trailing data all fail — so the upstream hex vectors are the porting
//! oracle ([ADR 0002](../../docs/adr/0002-hand-ported-cbor-codec.md)).
//!
//! Envelope payloads stay opaque strict JSON whose service meaning belongs
//! to `pi_chord`; the protocol validates only that opaque values are strict
//! JSON. Upstream is transport-neutral: no sockets, no streams — the
//! decoders accept arbitrary chunk boundaries through an explicit
//! `push`/`end` buffer API, and this crate declares no async machinery.
//!
//! Two upstream contracts are restated because JavaScript provides them at
//! runtime and Rust cannot:
//!
//! - The JS value domain the codec moves (plain objects, arrays,
//!   `Uint8Array`, safe numbers) is the owned [`cbor::CborValue`] tree; the
//!   JS values upstream rejects at encode time have no variant to occupy.
//! - The typebox schemas restate as owned types whose constructors carry
//!   the validation surface ([`protocol::ServerId`]), with the
//!   schema walk in [`codec::parse_client_message`] and
//!   [`codec::parse_server_message`].
//!
//! The crate root stays the single import point, mirroring upstream's
//! `src/index.ts` re-export list.

pub mod cbor;
pub mod codec;
pub mod framing;
pub mod protocol;

pub use cbor::{
    CborError, CborOptions, CborValue, DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH, DEFAULT_MAX_CBOR_DEPTH, decode_cbor, encode_cbor,
};
pub use codec::{
    ClientMessageDecoder, ProtocolValidationError, ServerMessageDecoder, encode_client_message,
    encode_server_message, is_supported_protocol_version, parse_client_message,
    parse_server_message,
};
pub use framing::{
    DEFAULT_MAX_FRAME_LENGTH, FrameDecoder, FrameDecoderOptions, FrameError, encode_frame,
};
pub use protocol::{
    AttachmentEnvelope, CancelEnvelope, ClientHello, ClientMessage, PROTOCOL_VERSION,
    ProtocolError, ProtocolErrorCode, RequestEnvelope, ResponseEnvelope, ResponseFailure,
    ResponseSuccess, RpcTarget, ServerHello, ServerHelloError, ServerId, ServerMessage,
    ServerTarget, ServiceEventEnvelope, SessionTarget, is_server_id,
};
