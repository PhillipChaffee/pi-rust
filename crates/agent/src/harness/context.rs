//! The harness invocation context, ported from upstream
//! `src/harness/context.ts`.
//!
//! Upstream re-exports chord's context machinery and attaches the telemetry
//! parent as an ordinary context value; the port re-exports chord's Rust
//! port of the same machinery unchanged and keeps the telemetry parent as a
//! context value under the same `pi.telemetryContext` key. The chord model
//! is an explicitly passed value object, not ambient state, so the harness
//! carries it as the `context` parameter of every operation — the seam the
//! map's harness-foundations ticket settles, with no divergence from the
//! chord crate's Rust model to record as an ADR.

pub use pi_chord::context::{
    AbortController, AbortReason, AbortSignal, AwaitWithContext, Context, ContextKey,
    WaitAborted, await_with_context, background_context, create_context_key, placeholder_context,
    with_abort_signal, with_cancel, with_context_value, without_abort_signal,
};

use std::sync::OnceLock;

use pi_telemetry::{NOOP_TELEMETRY_CONTEXT, TelemetryHandle};

/// The key the telemetry parent is stored under, upstream's
/// `TELEMETRY_CONTEXT_KEY` (`"pi.telemetryContext"`).
///
/// The value type is the dispatch handle per ADR 0004: telemetry crosses
/// crate boundaries erased, and chord's `Context` values are
/// `Any + Send + Sync`.
fn telemetry_context_key() -> &'static ContextKey<TelemetryHandle> {
    static KEY: OnceLock<ContextKey<TelemetryHandle>> = OnceLock::new();
    KEY.get_or_init(|| create_context_key("pi.telemetryContext"))
}

/// Return the telemetry parent attached to a context, or the shared no-op
/// parent.
#[must_use]
pub fn get_telemetry_context(context: &Context) -> TelemetryHandle {
    context
        .value(telemetry_context_key())
        .cloned()
        .unwrap_or_else(|| TelemetryHandle::new(NOOP_TELEMETRY_CONTEXT))
}

/// Derive a context whose telemetry children use the supplied parent or
/// active span.
#[must_use]
pub fn with_telemetry_context(telemetry_context: TelemetryHandle, context: &Context) -> Context {
    with_context_value(telemetry_context_key(), telemetry_context, context)
}

#[cfg(test)]
mod tests;