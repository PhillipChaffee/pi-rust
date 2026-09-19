//! Redacted assistant-message diagnostics, ported from
//! `packages/ai/src/utils/diagnostics.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! These helpers turn a thrown value into the redacted
//! [`AssistantMessageDiagnostic`] attached to assistant messages for failures
//! and recoveries. The data shape lives in [`crate::types`]; this module owns
//! the extraction helpers.

use crate::types::{AssistantMessage, AssistantMessageDiagnostic, DiagnosticErrorInfo};

/// The panic-free equivalent of upstream's thrown-value formatting: every
/// Rust error is `Display`-able, so the string, error, and arbitrary-value
/// branches of `formatThrownValue` collapse to one path.
#[must_use]
pub fn format_thrown_value(value: &dyn std::fmt::Display) -> String {
    value.to_string()
}

/// Extract the redacted error info from a thrown error.
///
/// Upstream reads `error.name`, `error.message`, `error.stack`, and the
/// SDK-specific `error.code` field. Rust errors carry their classification in
/// the type name and their message in `Display`; the stack capture follows
/// `RUST_BACKTRACE` like every other panic-path capture, and `code` stays
/// unset — provider adapters that know a code attach it themselves.
#[must_use]
pub fn extract_diagnostic_error<E: std::error::Error>(error: &E) -> DiagnosticErrorInfo {
    let name = type_short_name(std::any::type_name::<E>());
    let message = error.to_string();
    DiagnosticErrorInfo {
        name: Some(name.clone()),
        message: if message.is_empty() { name } else { message },
        stack: capture_stack(),
        code: None,
    }
}

/// Build a diagnostic stamped with the supplied timestamp, upstream's
/// `Date.now()` capture point.
#[must_use]
pub fn create_assistant_message_diagnostic<E: std::error::Error>(
    kind: &str,
    error: &E,
    details: Option<std::collections::BTreeMap<String, serde_json::Value>>,
    timestamp_ms: i64,
) -> AssistantMessageDiagnostic {
    AssistantMessageDiagnostic {
        kind: kind.to_owned(),
        timestamp: timestamp_ms,
        error: Some(extract_diagnostic_error(error)),
        details,
    }
}

/// Append a diagnostic to a message's diagnostics list, creating the list
/// when the message has none.
pub fn append_assistant_message_diagnostic(
    message: &mut AssistantMessage,
    diagnostic: AssistantMessageDiagnostic,
) {
    message
        .diagnostics
        .get_or_insert_with(Vec::new)
        .push(diagnostic);
}

fn type_short_name(type_path: &str) -> String {
    type_path.rsplit(':').next().unwrap_or(type_path).to_owned()
}

fn capture_stack() -> Option<String> {
    let backtrace = std::backtrace::Backtrace::capture();
    match backtrace.status() {
        std::backtrace::BacktraceStatus::Captured => Some(backtrace.to_string()),
        _ => None,
    }
}
