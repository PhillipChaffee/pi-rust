//! The shared inert context, ported from upstream `noop.ts`.
//!
//! Upstream freezes one shared span object and hands it to every body; in
//! Rust the span is a zero-sized type, so identity holds for every handle
//! and immutability is the default rather than a freeze call.

use std::future::Future;

use crate::{SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan};

/// The shared context used when an application does not provide one.
///
/// Bodies are admitted and their results preserved; nothing is recorded and
/// no payload is inspected.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTelemetryContext;

/// The one inert span the noop context hands out.
///
/// Child spans started from it return the same zero-sized handle, matching
/// upstream's "child spans return the same span" guarantee.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopSpan;

impl TelemetryContext for NoopTelemetryContext {
    type Span = NoopSpan;

    async fn start_span<T, E, Fut, F>(&self, _options: SpanOptions, body: F) -> Result<T, E>
    where
        F: FnOnce(NoopSpan) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        body(NoopSpan).await
    }
}

impl TelemetryContext for NoopSpan {
    type Span = Self;

    async fn start_span<T, E, Fut, F>(&self, _options: SpanOptions, body: F) -> Result<T, E>
    where
        F: FnOnce(Self) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        body(Self).await
    }
}

impl TelemetrySpan for NoopSpan {
    fn add_event(&self, _name: &str, _attributes: SpanAttributes) {}

    fn set_attributes(&self, _attributes: SpanAttributes) {}

    fn set_status(&self, _status: SpanStatus) {}
}

/// The shared noop context, mirroring upstream's frozen `NOOP_TELEMETRY_CONTEXT`.
pub const NOOP_TELEMETRY_CONTEXT: NoopTelemetryContext = NoopTelemetryContext;
