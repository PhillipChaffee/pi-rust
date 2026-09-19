//! Vendor-neutral callback-span telemetry contract, ported from
//! `packages/telemetry` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The contract is callback-managed: [`TelemetryContext::start_span`] hands a
//! span to a body closure and settles the span when that body's future
//! completes — there is no `end()`. Three reference pieces ship beside the
//! contract: the shared [`NOOP_TELEMETRY_CONTEXT`], the recording
//! [`InMemoryTelemetryContext`] for tests and local diagnostics, and the
//! adapter conformance suite under [`testing`]. On top of the contract, the
//! [`schema`] module provides serializable schema data plus a macro that
//! generates a compile-time-checked span vocabulary, and [`typed`] binds that
//! vocabulary to an explicit parent context.
//!
//! The port deliberately has no exporter, no ambient context, no timestamps,
//! and no backend dependency, mirroring upstream's "Security and
//! Portability" stance: contexts are passed explicitly, never via
//! thread-locals.
//!
//! Two upstream contracts are restated rather than copied, because Rust has
//! no runtime analogue:
//!
//! - Hostile (throwing) telemetry payloads cannot exist for owned plain
//!   data, so recording is infallible by construction; the conformance
//!   suite's Proxy-hostility cases are dropped (see [`testing`]).
//! - A panicking body is a programmer bug, not a telemetry outcome; spans
//!   map callback failures onto `Result` errors, and the automatic error
//!   status applies to `Err` returns. Error *details* (upstream reads
//!   `Error.name`/`Error.message`) need an error-trait seam the agent crate
//!   will introduce; until then the automatic status carries no details.

#![forbid(unsafe_code)]

pub mod dispatch;
pub mod memory;
pub mod noop;
pub mod schema;
pub mod testing;
pub mod typed;

// Upstream's `index.ts` re-exports the noop context and the in-memory
// adapter beside the contract; the crate root keeps that single import point.
pub use crate::dispatch::{
    BoxedSpanBody, DynSpanHandle, DynTelemetryContext, DynTelemetrySpan, ErasedSpanFuture,
    SpanBodyError, SpanBodyFailure, TelemetryHandle,
};
pub use crate::memory::{InMemoryTelemetryContext, RecordedTelemetryEvent, RecordedTelemetrySpan};
pub use crate::noop::{NOOP_TELEMETRY_CONTEXT, NoopSpan, NoopTelemetryContext};

use std::collections::BTreeMap;
use std::future::Future;

/// A single telemetry attribute value.
///
/// `f64` is its own variant because schema literals and recorded attributes
/// must round-trip integers and floats distinctly.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    /// A string value.
    Str(String),
    /// An integral value.
    Int(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
    /// A list of string values.
    StrArray(Vec<String>),
    /// A list of integral values.
    IntArray(Vec<i64>),
    /// A list of floating-point values.
    FloatArray(Vec<f64>),
    /// A list of boolean values.
    BoolArray(Vec<bool>),
}

impl From<&str> for AttributeValue {
    fn from(value: &str) -> Self {
        Self::Str(value.to_owned())
    }
}

impl From<String> for AttributeValue {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl From<i64> for AttributeValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for AttributeValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<bool> for AttributeValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<Vec<String>> for AttributeValue {
    fn from(value: Vec<String>) -> Self {
        Self::StrArray(value)
    }
}

impl From<Vec<i64>> for AttributeValue {
    fn from(value: Vec<i64>) -> Self {
        Self::IntArray(value)
    }
}

impl From<Vec<f64>> for AttributeValue {
    fn from(value: Vec<f64>) -> Self {
        Self::FloatArray(value)
    }
}

impl From<Vec<bool>> for AttributeValue {
    fn from(value: Vec<bool>) -> Self {
        Self::BoolArray(value)
    }
}

/// The name-to-value map of a span or event.
///
/// A [`None`] entry is the port of upstream's `undefined` value: it is legal
/// in the input map and every recording step skips it, so `None` never
/// overwrites an existing attribute.
pub type SpanAttributes = BTreeMap<String, Option<AttributeValue>>;

/// Options for starting a span.
#[derive(Clone, Debug)]
pub struct SpanOptions {
    /// The span name, as recorded by adapters.
    pub name: String,
    /// Attributes present at span start; `None` mirrors upstream's absent
    /// `attributes` field.
    pub attributes: Option<SpanAttributes>,
}

impl SpanOptions {
    /// Builds options for a span with no start attributes.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: None,
        }
    }

    /// Attaches start attributes, replacing any previous value.
    #[must_use]
    pub fn with_attributes(mut self, attributes: SpanAttributes) -> Self {
        self.attributes = Some(attributes);
        self
    }
}

/// The name and message pair attached to an explicit error status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanError {
    /// A short classification of the failure, upstream's `error.name`.
    pub name: String,
    /// A human-readable description, upstream's `error.message`.
    pub message: String,
}

/// The last status recorded for a span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpanStatus {
    /// The span's work completed without a recorded failure.
    Ok,
    /// The span failed; details are present only when recorded explicitly,
    /// mirroring upstream's optional `error` field.
    Error {
        /// The recorded failure details, when set explicitly.
        error: Option<SpanError>,
    },
}

/// Starts spans and hands them to a body future; every span is also a
/// context, because child spans start from a live span handle.
///
/// Settlement is callback-scoped: the span opens when
/// [`TelemetryContext::start_span`] is
/// called and settles when the body future completes — `Ok` settles as
/// success, `Err` settles with an automatic error status unless the body set
/// an explicit status first. A future that is never polled leaves its span
/// open, like an upstream promise that never settles; a body that panics is
/// a programmer bug and intentionally settles nothing.
pub trait TelemetryContext {
    /// The span handle type this context hands to bodies.
    type Span: TelemetrySpan;

    /// Registers a span synchronously and returns a future that runs `body`
    /// with it on first poll and settles the span on completion.
    ///
    /// Bodies are `Send` because the dispatch handle ([`TelemetryHandle`],
    /// [ADR 0004]) drives them across threads; a future owned by one thread
    /// cannot cross the erased layer.
    ///
    /// # Errors
    /// Propagates the body's `Err` value unchanged, after settling the span.
    fn start_span<T, E, Fut, F>(
        &self,
        options: SpanOptions,
        body: F,
    ) -> impl Future<Output = Result<T, E>> + Send
    where
        F: FnOnce(Self::Span) -> Fut + Send,
        Fut: Future<Output = Result<T, E>> + Send;
}

/// A live span handle: records events, attributes, and status, and starts
/// child spans that record as its children.
///
/// Handles are cheap clones sharing one underlying span; they stay usable
/// after settlement, where every call is inert (see the conformance suite).
pub trait TelemetrySpan: TelemetryContext<Span = Self> + Clone {
    /// Records an event on the span; inert once settled.
    fn add_event(&self, name: &str, attributes: SpanAttributes);

    /// Merges attributes into the span, later names winning; inert once
    /// settled.
    fn set_attributes(&self, attributes: SpanAttributes);

    /// Replaces the span status; the last explicit call wins and is
    /// preserved over the automatic error status; inert once settled.
    fn set_status(&self, status: SpanStatus);
}
