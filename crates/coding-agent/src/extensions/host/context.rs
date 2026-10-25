//! The context-call whitelist: the reads a wire child reaches through
//! `context_call` inside a call scope ([ADR 0010]).
//!
//! Every method answers in scope — an unknown name is `unknown_method`, an
//! unbound UI is `unsupported`, and a stale context is `stale` — so
//! additive methods never fail the protocol. The values are the domain
//! accessors' own JSON shapes; the wire carries them without a second
//! vocabulary.
//!
//! [ADR 0010]: ../../../../../docs/adr/0010-extension-host-wire-protocol.md

use std::fmt;

use pi_agent_core::harness::context::AbortSignal;
use serde_json::{Value, json};

use crate::extensions::reasons::ExtensionMode;
use crate::extensions::runtime::HandlerContext;

mod domain;
mod options;

/// One in-scope call failure: the `context_error` / `ui_error` body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostCallError {
    code: String,
    message: String,
    stack: Option<String>,
}

impl HostCallError {
    /// Build a failure from its code and message.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            stack: None,
        }
    }

    /// Carry the child's failure stack.
    #[must_use]
    pub fn with_stack(mut self, stack: Option<String>) -> Self {
        self.stack = stack;
        self
    }

    /// The child-side stack, when the failure carried one.
    #[must_use]
    pub fn stack(&self) -> Option<&str> {
        self.stack.as_deref()
    }

    /// The `unknown_method` failure an unrecognized method answers with.
    #[must_use]
    pub fn unknown_method(method: &str) -> Self {
        Self::new(
            "unknown_method",
            format!("the extension context has no `{method}` method"),
        )
    }

    /// The `unsupported` failure an unbound surface answers with.
    #[must_use]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new("unsupported", message)
    }

    /// The failure code.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// The failure message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The `stale` restatement of a context accessor's refusal.
    pub(super) fn stale(message: String) -> Self {
        Self::new("stale", message)
    }

    /// The `invalid_params` failure for a malformed method argument.
    pub(super) fn invalid_params(message: impl Into<String>) -> Self {
        Self::new("invalid_params", message)
    }
}

impl fmt::Display for HostCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for HostCallError {}

/// The UI pass-through seam a `ui_call` reaches. The interactive surfaces
/// bind their own implementation (the interactive ticket); without one the
/// session answers `unsupported`.
pub trait UiCallHandler: Send + Sync {
    /// Run one UI method, answering with its value or an in-scope failure.
    fn call(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> pi_ai::types::BoxedFuture<'static, Result<Value, HostCallError>>;
}

/// Answer one `context_call` from the whitelist.
///
/// `has_ui` is the session's binding truth, not the context's carried
/// handle: ADR 0010 makes an unbound UI read as absent even when a stale
/// context handle survives.
///
/// # Errors
///
/// Returns an in-scope [`HostCallError`] for an unknown method, malformed
/// parameters, a stale context, or a failing accessor.
pub(super) fn call_context(
    context: &HandlerContext,
    method: &str,
    params: Option<&Value>,
    has_ui: bool,
    signal_id: &mut dyn FnMut(&AbortSignal) -> u64,
) -> Result<Value, HostCallError> {
    match method {
        "cwd" => context.cwd().map(Value::from).map_err(HostCallError::stale),
        "mode" => context
            .mode()
            .map(|mode| Value::from(mode_name(mode)))
            .map_err(HostCallError::stale),
        "has_ui" => Ok(Value::Bool(has_ui)),
        "model" => json_value(context.get_model().map_err(HostCallError::stale)?),
        "scoped_models" => options::scoped_models_to_wire(
            &context.get_scoped_models().map_err(HostCallError::stale)?,
        ),
        "thinking_level" => json_value(context.get_thinking_level().map_err(HostCallError::stale)?),
        "is_idle" => context
            .is_idle()
            .map(Value::Bool)
            .map_err(HostCallError::stale),
        "is_project_trusted" => context
            .is_project_trusted()
            .map(Value::Bool)
            .map_err(HostCallError::stale),
        "has_pending_messages" => context
            .has_pending_messages()
            .map(Value::Bool)
            .map_err(HostCallError::stale),
        "get_context_usage" => context
            .get_context_usage()
            .map_err(HostCallError::stale)
            .map(|usage| {
                usage.map_or(Value::Null, |usage| {
                    json!({
                        "tokens": usage.tokens,
                        "contextWindow": usage.context_window,
                        "percent": usage.percent,
                    })
                })
            }),
        "get_system_prompt" => context
            .get_system_prompt()
            .map(Value::from)
            .map_err(HostCallError::stale),
        "get_system_prompt_options" => Ok(options::system_prompt_options_to_wire(
            &context
                .get_system_prompt_options()
                .map_err(HostCallError::stale)?,
        )),
        "compact" => {
            let instructions = domain::optional_param_string(params, "customInstructions")?;
            context
                .compact(instructions)
                .map(|()| Value::Null)
                .map_err(HostCallError::stale)
        }
        "abort" => context
            .abort()
            .map(|()| Value::Null)
            .map_err(HostCallError::stale),
        "shutdown" => context
            .shutdown()
            .map(|()| Value::Null)
            .map_err(HostCallError::stale),
        "get_signal" => context
            .get_signal()
            .map_err(HostCallError::stale)
            .map(|signal| {
                signal.map_or_else(
                    || json!({ "signal": null }),
                    |signal| json!({ "signal": signal_id(&signal) }),
                )
            }),
        other if domain::is_domain_method(other) => domain::call_domain(context, other, params),
        other => Err(HostCallError::unknown_method(other)),
    }
}

fn json_value<T: serde::Serialize>(value: T) -> Result<Value, HostCallError> {
    serde_json::to_value(value)
        .map_err(|error| HostCallError::invalid_params(format!("the value is not JSON: {error}")))
}

const fn mode_name(mode: ExtensionMode) -> &'static str {
    match mode {
        ExtensionMode::Tui => "tui",
        ExtensionMode::Rpc => "rpc",
        ExtensionMode::Json => "json",
        ExtensionMode::Print => "print",
    }
}
