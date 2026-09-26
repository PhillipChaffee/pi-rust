//! Ports `packages/telemetry/test/telemetry.test.ts` one to one: the schema
//! layer's runtime behavior and the noop context. Upstream's compile-time
//! assertions (`expectTypeOf`, `@ts-expect-error`) port to the
//! `compile_fail` doctests on [`pi_telemetry::define_telemetry_schema`] and
//! [`pi_telemetry::typed::TypedSpanStarter`], plus the structural-equality
//! checks below.

use std::collections::BTreeMap;

use pi_telemetry::memory::InMemoryTelemetryContext;
use pi_telemetry::noop::NOOP_TELEMETRY_CONTEXT;
use pi_telemetry::schema::{
    IntoSpanAttributes, TelemetryAttributeDefinition, TelemetryAttributeMetadata,
    TelemetryAttributeType, TelemetryEventDefinition, TelemetryLiteral, TelemetryParentDefinition,
    TelemetrySchema, TelemetrySchemaDefinition, TelemetrySpanDefinition, TelemetryStatusDefinition,
};
use pi_telemetry::testing::Rejection;
use pi_telemetry::typed::TypedSpanStarter;
use pi_telemetry::{
    AttributeValue, SpanAttributes, SpanError, SpanOptions, SpanStatus, TelemetryContext,
    TelemetrySpan, create_typed_span_starter, define_telemetry_schema,
};

define_telemetry_schema! {
    /// The vocabulary of the test operation, ported from the schema test.
    pub schema test_operation {
        version: 1,
        spans: {
            /// The operation span.
            operation => "Test operation" {
                parents: [any],
                start_attributes: {
                    "kind" kind as Kind: string required values [Read: "read", Write: "write"] description: "Kind",
                },
                end_attributes: {},
                events: {
                    result => "Result" {
                        attributes: {
                            "outcome" outcome as Outcome: string required values [Ok: "ok", Error: "error"] description: "Outcome",
                        },
                    },
                },
                status: { default ok, error_when "The operation fails" },
            },
        },
    }
}

define_telemetry_schema! {
    /// The operation schema of the combined vocabulary.
    pub schema combined_operation {
        version: 1,
        spans: {
            operation => "Operation" {
                parents: [root_or_external],
                start_attributes: {
                    "kind" kind as Kind: string required values [Read: "read", Write: "write"] description: "Kind",
                },
                end_attributes: {},
                status: { default ok, error_when "The operation fails" },
            },
        },
    }
}

define_telemetry_schema! {
    /// The request schema of the combined vocabulary.
    pub schema combined_request {
        version: 3,
        spans: {
            request => "Request" {
                parents: [spans "operation"],
                start_attributes: {
                    "provider" provider: string required description: "Provider",
                },
                end_attributes: {
                    "response" response: string optional description: "Response kind",
                },
                status: { default ok, error_when "The request fails" },
            },
        },
    }
}

define_telemetry_schema! {
    /// A schema exercising every attribute kind, beyond upstream's suite.
    pub schema coverage_kinds {
        version: 1,
        spans: {
            kinds => "Every attribute kind" {
                parents: [any],
                start_attributes: {
                    "plain" plain: string required description: "Plain string",
                    "count" count: number required description: "Count",
                    "enabled" enabled: boolean required description: "Enabled",
                    "tags" tags: string_array optional description: "Tags",
                    "sizes" sizes: number_array optional description: "Sizes",
                    "flags" flags: boolean_array optional description: "Flags",
                },
                end_attributes: {
                    "note" note: string optional description: "Note",
                },
                status: { default ok, error_when "The kind exercise fails" },
            },
        },
    }
}

fn attributes(entries: Vec<(&str, Option<AttributeValue>)>) -> SpanAttributes {
    SpanAttributes::from_iter(
        entries
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value)),
    )
}

fn str_attributes(name: &str, value: &str) -> SpanAttributes {
    attributes(vec![(name, Some(AttributeValue::from(value)))])
}

/// Ports "preserves serializable definitions and infers exact attributes".
#[test]
fn preserves_definitions_and_infers_exact_attributes() {
    let schema = test_operation::Schema::definition();

    // The identity-and-JSON-serializability pair: upstream asserts the
    // helper returns the definition unchanged and that it stringifies; the
    // port asserts the generated data term equals the hand-built definition.
    let expected = TelemetrySchemaDefinition {
        version: 1,
        spans: BTreeMap::from([(
            "operation".to_owned(),
            TelemetrySpanDefinition {
                description: "Test operation".to_owned(),
                parents: TelemetryParentDefinition::Any,
                start_attributes: BTreeMap::from([(
                    "kind".to_owned(),
                    TelemetryAttributeDefinition {
                        metadata: TelemetryAttributeMetadata {
                            description: "Kind".to_owned(),
                            sensitive: false,
                            cardinality: None,
                        },
                        kind: TelemetryAttributeType::Str,
                        values: vec![
                            TelemetryLiteral::Str("read".to_owned()),
                            TelemetryLiteral::Str("write".to_owned()),
                        ],
                        element_values: Vec::new(),
                        examples: Vec::new(),
                        required: true,
                    },
                )]),
                end_attributes: BTreeMap::new(),
                events: BTreeMap::from([(
                    "result".to_owned(),
                    TelemetryEventDefinition {
                        description: "Result".to_owned(),
                        attributes: BTreeMap::from([(
                            "outcome".to_owned(),
                            TelemetryAttributeDefinition {
                                metadata: TelemetryAttributeMetadata {
                                    description: "Outcome".to_owned(),
                                    sensitive: false,
                                    cardinality: None,
                                },
                                kind: TelemetryAttributeType::Str,
                                values: vec![
                                    TelemetryLiteral::Str("ok".to_owned()),
                                    TelemetryLiteral::Str("error".to_owned()),
                                ],
                                element_values: Vec::new(),
                                examples: Vec::new(),
                                required: true,
                            },
                        )]),
                    },
                )]),
                status: TelemetryStatusDefinition {
                    error_when: "The operation fails".to_owned(),
                },
            },
        )]),
    };
    assert_eq!(schema, expected);

    // Upstream infers `{ kind: "read" | "write" }` at the type level; the
    // port constructs the generated struct and converts it.
    let start = test_operation::operation::Start {
        kind: test_operation::operation::Kind::Read,
    };
    assert_eq!(
        start.into_span_attributes(),
        attributes(vec![("kind", Some(AttributeValue::from("read")))]),
    );

    // The empty end schema converts to no attributes.
    assert_eq!(
        test_operation::operation::End {}.into_span_attributes(),
        SpanAttributes::new(),
    );
}

/// Ports "records the declared vocabulary at runtime", the runnable half of
/// upstream's compile-time-failures block.
#[tokio::test]
async fn records_declared_vocabulary() {
    let context = InMemoryTelemetryContext::new();
    let starter = create_typed_span_starter!(context.clone(), test_operation::Schema);
    let result = starter.start_span(
        test_operation::operation::Span,
        test_operation::operation::Start {
            kind: test_operation::operation::Kind::Write,
        },
        |span, _child| async move {
            span.add_event(
                test_operation::operation::result::Event,
                test_operation::operation::result::Attributes {
                    outcome: test_operation::operation::result::Outcome::Error,
                },
            );
            let _ = format!("{span:?}");
            Ok::<(), Rejection>(())
        },
    );
    assert_eq!(result.await, Ok(()));

    let spans = context.get_spans();
    assert_eq!(spans[0].events[0].name, "result");
    assert_eq!(
        spans[0].events[0].attributes,
        attributes(vec![("outcome", Some(AttributeValue::from("error")))]),
    );
}

/// Ports "combines schema vocabularies and binds child starters to their
/// parent spans".
#[tokio::test]
async fn combines_schema_vocabularies_and_binds_child_starters_to_their_parent_spans() {
    let context = InMemoryTelemetryContext::new();
    // Upstream also passes unreadable (Proxy) schemas to the starter; owned
    // values make that passivity static, so the case is dropped.
    let starter = create_typed_span_starter!(
        context.clone(),
        combined_operation::Schema,
        combined_request::Schema
    );

    let result = starter
        .start_span(
            combined_operation::operation::Span,
            combined_operation::operation::Start {
                kind: combined_operation::operation::Kind::Read,
            },
            |_operation, child| async move {
                child
                    .start_span(
                        combined_request::request::Span,
                        combined_request::request::Start {
                            provider: "example".to_owned(),
                        },
                        |request_span, _child| async move {
                            request_span.set_attributes(combined_request::request::End {
                                response: Some("cached".to_owned()),
                            });
                            request_span.set_status(SpanStatus::Ok);
                            Ok::<i64, Rejection>(42)
                        },
                    )
                    .await
            },
        )
        .await;
    assert_eq!(result, Ok(42));

    let spans = context.get_spans();
    let operation = find_span(&spans, "operation");
    let request = find_span(&spans, "request");
    assert_eq!(operation.parent_id, None);
    assert_eq!(request.parent_id, Some(operation.id));

    let sync_error = starter
        .start_span(
            combined_operation::operation::Span,
            combined_operation::operation::Start {
                kind: combined_operation::operation::Kind::Write,
            },
            |_span, _child| async { Err::<(), Rejection>(Rejection::SYNC) },
        )
        .await;
    assert_eq!(sync_error, Err(Rejection::SYNC));

    let async_error = starter
        .start_span(
            combined_request::request::Span,
            combined_request::request::Start {
                provider: "example".to_owned(),
            },
            |_span, _child| async {
                tokio::task::yield_now().await;
                Err::<(), Rejection>(Rejection::ASYNC)
            },
        )
        .await;
    assert_eq!(async_error, Err(Rejection::ASYNC));
}

/// Ports the `NOOP_TELEMETRY_CONTEXT` admission and shared-span case. Upstream
/// asserts `Object.isFrozen` and `child === span`; in Rust the noop span is
/// a zero-sized type, so identity and immutability are structural.
#[tokio::test]
async fn admits_callbacks_and_reuses_one_inert_span() {
    let admitted = &mut false;
    let result = NOOP_TELEMETRY_CONTEXT
        .start_span(SpanOptions::new("first"), |span| {
            let slot = &mut *admitted;
            async move {
                *slot = true;
                span.start_span(SpanOptions::new("child"), |_child_span| async {
                    Ok::<i64, Rejection>(42)
                })
                .await
            }
        })
        .await;
    assert_eq!(result, Ok(42));
    assert!(*admitted);
}

/// Ports the noop rejection-identity case.
#[tokio::test]
async fn noop_preserves_rejection_values() {
    let sync = NOOP_TELEMETRY_CONTEXT
        .start_span(SpanOptions::new("sync"), |_| async {
            Err::<(), Rejection>(Rejection::SYNC)
        })
        .await;
    assert_eq!(sync, Err(Rejection::SYNC));

    let async_result = NOOP_TELEMETRY_CONTEXT
        .start_span(SpanOptions::new("async"), |_| async {
            tokio::task::yield_now().await;
            Err::<(), Rejection>(Rejection::ASYNC)
        })
        .await;
    assert_eq!(async_result, Err(Rejection::ASYNC));
}

/// Ports the payload non-inspection case; upstream drives it with unreadable
/// (Proxy) payloads, which cannot exist for owned values.
#[tokio::test]
async fn noop_does_not_inspect_or_retain_telemetry_payloads() {
    let result = NOOP_TELEMETRY_CONTEXT
        .start_span(
            SpanOptions::new("operation")
                .with_attributes(str_attributes("secret", "prompt content")),
            |span| async move {
                span.add_event("event", str_attributes("secret", "content"));
                span.set_attributes(str_attributes("secret", "content"));
                span.set_status(SpanStatus::Error { error: None });
                Ok::<(), Rejection>(())
            },
        )
        .await;
    assert_eq!(result, Ok(()));
}

/// Exercises every attribute kind end to end, beyond upstream's suite, so
/// the generated conversions and data kinds stay covered.
#[test]
fn every_attribute_kind_converts_into_span_attributes() {
    let start = coverage_kinds::kinds::Start {
        plain: "value".to_owned(),
        count: 3,
        enabled: true,
        tags: Some(vec!["a".to_owned()]),
        sizes: Some(vec![1, 2]),
        flags: Some(vec![false]),
    };
    let mut expected = SpanAttributes::new();
    expected.insert("plain".to_owned(), Some(AttributeValue::from("value")));
    expected.insert("count".to_owned(), Some(AttributeValue::Int(3)));
    expected.insert("enabled".to_owned(), Some(AttributeValue::Bool(true)));
    expected.insert(
        "tags".to_owned(),
        Some(AttributeValue::from(vec!["a".to_owned()])),
    );
    expected.insert(
        "sizes".to_owned(),
        Some(AttributeValue::from(vec![1i64, 2])),
    );
    expected.insert("flags".to_owned(), Some(AttributeValue::from(vec![false])));
    assert_eq!(start.into_span_attributes(), expected);

    let end = coverage_kinds::kinds::End {
        note: Some("done".to_owned()),
    };
    let mut expected_end = SpanAttributes::new();
    expected_end.insert("note".to_owned(), Some(AttributeValue::from("done")));
    assert_eq!(end.into_span_attributes(), expected_end);

    assert_eq!(coverage_kinds::Schema::definition().version, 1);
}

/// Formats the reference types, pinning their `Debug` impls.
#[tokio::test]
async fn reference_types_format_debug() {
    assert_eq!(
        AttributeValue::from("text"),
        AttributeValue::Str("text".to_owned())
    );
    assert_eq!(
        AttributeValue::from("text".to_owned()),
        AttributeValue::Str("text".to_owned())
    );
    assert_eq!(AttributeValue::from(1i64), AttributeValue::Int(1));
    assert_eq!(AttributeValue::from(1.5f64), AttributeValue::Float(1.5));
    assert_eq!(AttributeValue::from(true), AttributeValue::Bool(true));
    assert_eq!(
        AttributeValue::from(vec!["a".to_owned()]),
        AttributeValue::StrArray(vec!["a".to_owned()])
    );
    assert_eq!(
        AttributeValue::from(vec![1i64]),
        AttributeValue::IntArray(vec![1])
    );
    assert_eq!(
        AttributeValue::from(vec![1.5f64]),
        AttributeValue::FloatArray(vec![1.5])
    );
    assert_eq!(
        AttributeValue::from(vec![true]),
        AttributeValue::BoolArray(vec![true])
    );
    let values = [
        AttributeValue::from("text"),
        AttributeValue::Int(1),
        AttributeValue::Float(1.5),
        AttributeValue::Bool(true),
        AttributeValue::StrArray(vec!["text".to_owned()]),
        AttributeValue::IntArray(vec![1]),
        AttributeValue::FloatArray(vec![1.5]),
        AttributeValue::BoolArray(vec![true]),
    ];
    for value in &values {
        assert!(!format!("{value:?}").is_empty());
    }
    assert!(!format!("{:?}", SpanOptions::new("formatted")).is_empty());
    assert!(!format!("{:?}", SpanStatus::Ok).is_empty());
    assert!(
        !format!(
            "{:?}",
            SpanError {
                name: "n".to_owned(),
                message: "m".to_owned()
            }
        )
        .is_empty()
    );
    assert!(!format!("{NOOP_TELEMETRY_CONTEXT:?}").is_empty());
    assert!(!format!("{:?}", pi_telemetry::noop::NoopSpan).is_empty());
    assert!(!format!("{:?}", Rejection::SYNC).is_empty());

    let context = InMemoryTelemetryContext::new();
    let escaped = &mut None;
    let formatted = context.start_span(SpanOptions::new("formatted"), |span| {
        let slot = &mut *escaped;
        async move {
            *slot = Some(span);
            Ok::<(), Rejection>(())
        }
    });
    assert_eq!(formatted.await, Ok(()));
    let handle = escaped.as_ref().map_or_else(
        || unreachable!("the formatting body captured the span handle"),
        Clone::clone,
    );
    assert!(!format!("{context:?}").is_empty());
    assert!(!format!("{handle:?}").is_empty());
    let spans = context.get_spans();
    assert!(!format!("{:?}", spans[0]).is_empty());
    assert!(!format!("{:?}", spans[0].events).is_empty());

    let starter = TypedSpanStarter::new(InMemoryTelemetryContext::new());
    assert!(!format!("{starter:?}").is_empty());
}

/// Exercises the compile-time duplicate rejection at runtime, so the const
/// uniqueness walk stays covered.
#[test]
fn span_name_uniqueness_walks_the_name_lists() {
    assert!(pi_telemetry::schema::span_names_unique(&[&[
        "read", "write"
    ]]));
    assert!(pi_telemetry::schema::span_names_unique(&[
        &["operation"],
        &["request"]
    ]));
    assert!(!pi_telemetry::schema::span_names_unique(&[
        &["request"],
        &["request"]
    ]));
}

fn find_span<'a>(
    spans: &'a [pi_telemetry::memory::RecordedTelemetrySpan],
    name: &str,
) -> &'a pi_telemetry::memory::RecordedTelemetrySpan {
    let span = spans.iter().find(|candidate| candidate.name == name);
    #[allow(
        clippy::option_if_let_else,
        reason = "the match reads clearer than a map_or_else with an identity closure"
    )]
    match span {
        Some(span) => span,
        None => unreachable!("no recorded span named {name}"),
    }
}

/// A span's optional wire-name literal sets the span name the schema data
/// and `SPAN_NAMES` carry, so dotted wire names a Rust identifier cannot
/// spell ride the declaration.
mod wire_named {
    pi_telemetry::define_telemetry_schema! {
        /// The vocabulary of one named span.
        pub schema wire_named {
            version: 1,
            spans: {
                /// The named span.
                operation as "pi.test.operation" => "Named operation" {
                    parents: [any],
                    start_attributes: {},
                    end_attributes: {},
                    status: { default ok, error_when "The operation fails" },
                },
            },
        }
    }

    #[test]
    fn the_wire_name_replaces_the_ident_in_names_and_data() {
        assert_eq!(
            <wire_named::Schema as pi_telemetry::schema::TelemetrySchema>::SPAN_NAMES,
            &["pi.test.operation"]
        );
        assert_eq!(
            <wire_named::operation::Span as pi_telemetry::schema::SpanDefinition>::NAME,
            "pi.test.operation"
        );
        let spans =
            <wire_named::Schema as pi_telemetry::schema::TelemetrySchema>::definition().spans;
        assert!(spans.contains_key("pi.test.operation"));
        assert!(!spans.contains_key("operation"));
    }
}
