//! The extension session's frame vocabulary, [ADR 0010]'s envelope types.
//!
//! Every frame is a CBOR map with a `type` text discriminator and the
//! fields ADR 0010 names. Payload fields carry upstream's TypeScript JSON
//! shape as [`Value`]: the event, result, and descriptor mappings land with
//! the serialization slice (#169), so this module owns only the envelope.
//!
//! [ADR 0010]: ../../../../../docs/adr/0010-extension-host-wire-protocol.md

use serde_json::Value;

/// The extension-session wire version, independent of the remote-session
/// `PROTOCOL_VERSION` ([ADR 0010]).
///
/// [ADR 0010]: ../../../../../docs/adr/0010-extension-host-wire-protocol.md
pub const EXTENSION_PROTOCOL_VERSION: u64 = 1;

/// The extension identity a child announces in its `hello`, ADR 0010's
/// `extension: { name, version }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionIdentity {
    /// The extension's package name.
    pub name: String,
    /// The extension's package version.
    pub version: String,
}

/// One frame on the extension session, ADR 0010's envelope union.
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "serde_json::Value has no Eq because JSON carries floats"
)]
#[derive(Clone, Debug, PartialEq)]
pub enum Frame {
    /// Child → host: open the session.
    Hello {
        /// The child's wire version.
        protocol_version: u64,
        /// The child's identity.
        extension: ExtensionIdentity,
    },
    /// Host → child: the handshake was accepted.
    Welcome {
        /// The host's wire version.
        protocol_version: u64,
    },
    /// Host → child: the handshake was refused; the host closes the session.
    HelloError {
        /// The refusal code.
        code: String,
        /// The human-readable refusal message.
        message: String,
    },
    /// Child → host: registration is complete and dispatch may start.
    Ready,
    /// Child → host: register one event handler.
    RegisterEvent {
        /// The event channel the handler subscribes to.
        channel: String,
    },
    /// Child → host: register one tool.
    RegisterTool {
        /// The declarative tool definition.
        tool: Value,
    },
    /// Child → host: register one command.
    RegisterCommand {
        /// The command descriptor.
        command: Value,
    },
    /// Child → host: register one CLI flag.
    RegisterFlag {
        /// The flag descriptor.
        flag: Value,
    },
    /// Child → host: register one model provider.
    RegisterProvider {
        /// The provider descriptor.
        provider: Value,
    },
    /// Host → child: dispatch one event handler.
    CallEvent {
        /// The host-assigned call id.
        call: u64,
        /// The event channel.
        channel: String,
        /// The event payload.
        payload: Value,
    },
    /// Host → child: execute one tool call.
    CallTool {
        /// The host-assigned call id.
        call: u64,
        /// The registered tool name.
        tool: String,
        /// The LLM tool-call id.
        tool_call_id: String,
        /// The validated tool arguments.
        params: Value,
        /// The serializable `ExtensionContext` snapshot.
        context: Value,
    },
    /// Child → host: report a tool's streaming update.
    CallUpdate {
        /// The host-assigned call id.
        call: u64,
        /// The tool update.
        update: Value,
    },
    /// Child → host: the call succeeded.
    CallResult {
        /// The host-assigned call id.
        call: u64,
        /// The handler result.
        result: Value,
    },
    /// Child → host: the call failed.
    CallError {
        /// The host-assigned call id.
        call: u64,
        /// The failure code.
        code: String,
        /// The human-readable failure message.
        message: String,
        /// The failure stack, when the child carries one.
        stack: Option<String>,
    },
    /// Host → child: invoke one registered command.
    CallCommand {
        /// The host-assigned call id.
        call: u64,
        /// The command name.
        name: String,
        /// The raw argument string.
        args: String,
    },
    /// Host → child: stream one provider response.
    CallProvider {
        /// The host-assigned call id.
        call: u64,
        /// The registered provider name.
        provider: String,
        /// The provider request.
        request: Value,
    },
    /// Child → host: one provider stream event.
    StreamEvent {
        /// The host-assigned call id.
        call: u64,
        /// The assistant message event.
        event: Value,
    },
    /// Child → host: the provider stream finished.
    StreamEnd {
        /// The host-assigned call id.
        call: u64,
    },
    /// Child → host: read host context inside a call scope.
    ContextCall {
        /// The host-assigned call id.
        call: u64,
        /// The caller-assigned request id.
        id: u64,
        /// The whitelisted context method.
        method: String,
        /// The method parameters, when it takes any.
        params: Option<Value>,
    },
    /// Host → child: the context read succeeded.
    ContextResult {
        /// The host-assigned call id.
        call: u64,
        /// The caller-assigned request id.
        id: u64,
        /// The method's return value.
        value: Value,
    },
    /// Host → child: the context read failed in scope.
    ContextError {
        /// The host-assigned call id.
        call: u64,
        /// The caller-assigned request id.
        id: u64,
        /// The failure code.
        code: String,
        /// The human-readable failure message.
        message: String,
    },
    /// Child → host: reach the UI inside a call scope.
    UiCall {
        /// The host-assigned call id.
        call: u64,
        /// The caller-assigned request id.
        id: u64,
        /// The whitelisted UI method.
        method: String,
        /// The method parameters, when it takes any.
        params: Option<Value>,
    },
    /// Host → child: the UI call succeeded.
    UiResult {
        /// The host-assigned call id.
        call: u64,
        /// The caller-assigned request id.
        id: u64,
        /// The method's return value.
        value: Value,
    },
    /// Host → child: the UI call failed in scope.
    UiError {
        /// The host-assigned call id.
        call: u64,
        /// The caller-assigned request id.
        id: u64,
        /// The failure code.
        code: String,
        /// The human-readable failure message.
        message: String,
    },
    /// Host → child: a signal inside the call scope was aborted.
    Signal {
        /// The host-assigned call id.
        call: u64,
        /// The host-assigned signal id.
        signal: u64,
        /// Whether the signal is aborted.
        aborted: bool,
    },
}
