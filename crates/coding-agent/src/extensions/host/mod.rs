//! The out-of-process extension host wire, [ADR 0010]'s CBOR session.
//!
//! A third-party extension is a child process whose stdin/stdout carry
//! pi-protocol's framed CBOR; this module owns the frame vocabulary, the
//! codec, the integer-preserving JSON bridge, the payload mappings for the
//! domain shapes (#169), and the live session with its dispatch proxies
//! (#170). Spawning is the loader slice (#171).
//!
//! [ADR 0010]: ../../../../../docs/adr/0010-extension-host-wire-protocol.md

use std::fmt;

use pi_protocol::CborError;

mod bridge;
mod codec;
mod context;
mod decode;
mod frame;
mod link;
mod payload;
mod session;
pub mod spawn;
#[cfg(test)]
mod tests;

pub use bridge::{cbor_to_json, json_to_cbor};
pub use codec::encode_frame;
pub use context::{HostCallError, UiCallHandler};
pub use decode::decode_frame;
pub use frame::{EXTENSION_PROTOCOL_VERSION, ExtensionIdentity, Frame};
pub use link::Link;
pub use payload::descriptors::{
    CommandDeclaration, ContextSnapshot, FlagDeclaration, ToolDeclaration,
    command_declaration_from_wire, command_to_wire, context_snapshot_from_wire,
    context_snapshot_to_wire, flag_declaration_from_wire, flag_to_wire, provider_from_wire,
    provider_to_wire, tool_declaration_from_wire, tool_definition_to_wire,
};
pub use payload::events::{EVENT_CHANNELS, event_from_wire, event_to_wire};
pub use payload::results::{RESULT_CHANNELS, result_from_wire, result_to_wire};
pub use session::{HostConnectOptions, HostSession, SharedContext, shared_context};
pub use spawn::{ExtensionLoader, HANDSHAKE_TIMEOUT, SHUTDOWN_GRACE, SpawnError, SpawnOptions};

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
