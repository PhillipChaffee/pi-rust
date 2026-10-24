//! The out-of-process extension host wire, [ADR 0010]'s CBOR session.
//!
//! A third-party extension is a child process whose stdin/stdout carry
//! pi-protocol's framed CBOR; this module owns the frame vocabulary, the
//! codec, and the integer-preserving JSON bridge. Spawning and dispatch are
//! later slices (#170, #171); the domain-shape serialization is #169.
//!
//! [ADR 0010]: ../../../../../docs/adr/0010-extension-host-wire-protocol.md

use std::fmt;

use pi_protocol::CborError;

mod bridge;
mod codec;
mod decode;
mod frame;
#[cfg(test)]
mod tests;

pub use bridge::{cbor_to_json, json_to_cbor};
pub use codec::encode_frame;
pub use decode::decode_frame;
pub use frame::{EXTENSION_PROTOCOL_VERSION, ExtensionIdentity, Frame};

/// A wire-level failure: a malformed or unknown frame, or a payload the
/// JSON bridge cannot represent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireError {
    message: String,
}

impl WireError {
    /// Builds the error from its message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The failure message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for WireError {}

impl From<CborError> for WireError {
    fn from(error: CborError) -> Self {
        Self::new(error.message())
    }
}
