//! The typed span starter, porting upstream's `createTypedSpanStarter`.
//!
//! The starter binds one explicit parent to a schema vocabulary: span names
//! are generated marker types, so the attribute struct is selected by the
//! span marker itself, an unknown span name has no marker to pass, and a
//! union-valued name variable cannot be passed where one marker type is
//! expected — upstream's union-narrowing guarantee falls out of the type
//! system.

use std::future::Future;
use std::marker::PhantomData;

use crate::schema::{IntoSpanAttributes, SpanDefinition};
use crate::{SpanOptions, TelemetryContext, TelemetrySpan};

/// A typed view of one live span, bound to its generated vocabulary.
pub struct SchemaSpan<N: SpanDefinition, Raw: TelemetrySpan> {
    raw: Raw,
    _span: PhantomData<N>,
}

impl<N: SpanDefinition, Raw: TelemetrySpan> SchemaSpan<N, Raw> {
    /// Wraps a raw span handle in the typed view for one vocabulary entry.
    #[must_use]
    pub const fn new(raw: Raw) -> Self {
        Self {
            raw,
            _span: PhantomData,
        }
    }

    /// Records an event declared on this span; events of other spans do not
    /// type-check here.
    pub fn add_event<E>(&self, _event: E, attributes: E::Attributes)
    where
        E: crate::schema::EventDefinition + crate::schema::EventOf<N>,
    {
        self.raw
            .add_event(E::NAME, attributes.into_span_attributes());
    }

    /// Merges the span's typed end attributes; every entry is optional and
    /// absent entries are skipped.
    pub fn set_attributes(&self, attributes: N::End) {
        self.raw.set_attributes(attributes.into_span_attributes());
    }

    /// Replaces the span status through the raw handle.
    pub fn set_status(&self, status: crate::SpanStatus) {
        self.raw.set_status(status);
    }
}

impl<N: SpanDefinition, Raw: TelemetrySpan> std::fmt::Debug for SchemaSpan<N, Raw> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaSpan")
            .field("span", &N::NAME)
            .finish()
    }
}

/// Binds an explicit parent context to the span vocabulary of the schemas
/// named in [`create_typed_span_starter!`](crate::create_typed_span_starter).
///
/// Per-span overloads are type selected, so attributes always come from the
/// schema that owns the span. A union-valued span name must be narrowed
/// before the call — upstream's
/// union-narrowing guarantee:
///
/// ```compile_fail
/// use pi_telemetry::create_typed_span_starter;
/// use pi_telemetry::define_telemetry_schema;
/// use pi_telemetry::memory::InMemoryTelemetryContext;
///
/// define_telemetry_schema! {
///     /// The operation vocabulary.
///     pub schema starter_operation {
///         version: 1,
///         spans: {
///             /// The operation span.
///             operation => "Operation" {
///                 parents: [root_or_external],
///                 start_attributes: {},
///                 end_attributes: {},
///                 status: { default ok, error_when "The operation fails" },
///             },
///         },
///     }
/// }
///
/// define_telemetry_schema! {
///     /// The request vocabulary.
///     pub schema starter_request {
///         version: 1,
///         spans: {
///             /// The request span.
///             request => "Request" {
///                 parents: [spans "operation"],
///                 start_attributes: {},
///                 end_attributes: {},
///                 status: { default ok, error_when "The request fails" },
///             },
///         },
///     }
/// }
///
/// async fn union_valued_names_must_be_narrowed() {
///     let context = InMemoryTelemetryContext::new();
///     let starter = create_typed_span_starter!(context, starter_operation::Schema, starter_request::Schema);
///     // A union-valued name cannot preserve name and attribute correlation.
///     let name = if true {
///         starter_operation::operation::Span
///     } else {
///         starter_request::request::Span
///     };
///     let _ = starter.start_span(name, (), |_| async { Ok(()) });
/// }
/// ```
///
/// Attributes are selected from the schema that owns the span, so another
/// schema's attribute struct does not type-check:
///
/// ```compile_fail
/// use pi_telemetry::create_typed_span_starter;
/// use pi_telemetry::define_telemetry_schema;
/// use pi_telemetry::memory::InMemoryTelemetryContext;
///
/// define_telemetry_schema! {
///     /// The operation vocabulary with a kind attribute.
///     pub schema owning_schema {
///         version: 1,
///         spans: {
///             /// The operation span.
///             operation => "Operation" {
///                 parents: [root_or_external],
///                 start_attributes: {
///                     "kind" kind as Kind: string required values [Read: "read", Write: "write"] description: "Kind",
///                 },
///                 end_attributes: {},
///                 status: { default ok, error_when "The operation fails" },
///             },
///         },
///     }
/// }
///
/// define_telemetry_schema! {
///     /// The request vocabulary without attributes.
///     pub schema borrowing_schema {
///         version: 1,
///         spans: {
///             /// The request span.
///             request => "Request" {
///                 parents: [spans "operation"],
///                 start_attributes: {},
///                 end_attributes: {},
///                 status: { default ok, error_when "The request fails" },
///             },
///         },
///     }
/// }
///
/// async fn attributes_come_from_the_owning_schema() {
///     let context = InMemoryTelemetryContext::new();
///     let starter = create_typed_span_starter!(context, owning_schema::Schema, borrowing_schema::Schema);
///     let _ = starter.start_span(
///         borrowing_schema::request::Span,
///         owning_schema::operation::Start {
///             kind: owning_schema::operation::Kind::Read,
///         },
///         |_| async { Ok(()) },
///     );
/// }
/// ```
pub struct TypedSpanStarter<C: TelemetryContext> {
    parent: C,
}

impl<C: TelemetryContext> TypedSpanStarter<C> {
    /// Binds the starter to one parent context.
    #[must_use]
    pub const fn new(parent: C) -> Self {
        Self { parent }
    }

    /// Starts the span named by `marker`, selecting the typed start
    /// attributes from its vocabulary, and hands the body the typed span
    /// plus a child starter bound to the new span.
    ///
    /// # Errors
    /// Propagates the body's `Err` value unchanged, after span settlement.
    pub fn start_span<N, T, E, Fut, F>(
        &self,
        _span: N,
        attributes: N::Start,
        body: F,
    ) -> impl Future<Output = Result<T, E>> + Send
    where
        N: SpanDefinition,
        F: FnOnce(SchemaSpan<N, C::Span>, TypedSpanStarter<C::Span>) -> Fut + Send,
        Fut: Future<Output = Result<T, E>> + Send,
    {
        let options = SpanOptions::new(N::NAME).with_attributes(attributes.into_span_attributes());
        TelemetryContext::start_span(&self.parent, options, move |raw| {
            let typed = SchemaSpan::new(raw.clone());
            let child = TypedSpanStarter::new(raw);
            body(typed, child)
        })
    }
}

impl<C: TelemetryContext> std::fmt::Debug for TypedSpanStarter<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedSpanStarter").finish_non_exhaustive()
    }
}
