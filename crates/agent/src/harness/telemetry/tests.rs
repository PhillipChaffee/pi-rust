//! The agent telemetry schema suite, ported from upstream
//! `test/harness/telemetry.test.ts`.
//!
//! The type-level attribute checks restate as the typed structs' compile
//! time shapes; the negative-type assertions (`@ts-expect-error`) have no
//! Rust slot because unknown attributes are compile errors the typed
//! starter itself rejects. The checked-in markdown reference render is a
//! repo script and rides its own ticket.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use pi_telemetry::schema::{IntoSpanAttributes, TelemetrySchema};
use pi_telemetry::{NOOP_TELEMETRY_CONTEXT, TelemetryHandle};

use crate::harness::context::{background_context, with_telemetry_context};
use crate::harness::telemetry::{harness_schema, ai_schema, start_ai_span, start_harness_span};

/// Both schemas carry upstream's span vocabularies in declaration order;
/// the composed schema list binds ai first, harness second.
#[test]
fn serializes_both_schemas_in_upstream_span_order() {
    assert_eq!(
        <ai_schema::Schema as TelemetrySchema>::SPAN_NAMES,
        &["pi.ai.request"]
    );
    assert_eq!(
        <harness_schema::Schema as TelemetrySchema>::SPAN_NAMES,
        &[
            "pi.harness.run",
            "pi.harness.compaction",
            "pi.harness.navigation",
            "pi.harness.checkpoint",
            "pi.harness.turn",
            "pi.harness.step",
            "pi.harness.tool",
            "pi.harness.hook",
            "pi.harness.sleep",
            "pi.harness.event_handler",
            "pi.session.write",
        ]
    );
    // The definition maps rebuild per call and carry the same names.
    let ai = <ai_schema::Schema as TelemetrySchema>::definition();
    let harness = <harness_schema::Schema as TelemetrySchema>::definition();
    assert_eq!(ai.version, 1);
    assert_eq!(harness.version, 1);
    assert_eq!(ai.spans.len(), 1);
    assert_eq!(harness.spans.len(), 11);
}

/// The composed typed starter starts a harness step with a nested AI
/// request child, upstream's `createTypedSpanStarter(NOOP, SCHEMAS)` test.
#[tokio::test]
async fn starts_ai_and_harness_spans_through_one_composed_typed_starter() {
    let starter = crate::agent_telemetry_starter!(NOOP_TELEMETRY_CONTEXT);
    let () = starter
        .start_span(
            harness_schema::step::Span,
            harness_schema::step::Start {
                lane_name: "main".to_owned(),
                operation_id: "operation".to_owned(),
                step_kind: harness_schema::step::StepKind::Assistant,
                step_attempt: 1,
                compaction_reason: None,
            },
            |step_span, start_child_span| async move {
                step_span.set_attributes(harness_schema::step::End {
                    step_outcome: Some(harness_schema::step::StepOutcome::Succeeded),
                });
                let () = start_child_span
                    .start_span(
                        ai_schema::request::Span,
                        ai_schema::request::Start {
                            operation: ai_schema::request::AiOperation::Stream,
                            provider: "provider".to_owned(),
                            model: "model".to_owned(),
                            api: "api".to_owned(),
                            streaming: true,
                            deferred: None,
                        },
                        |request_span, _child| async move {
                            request_span.set_attributes(ai_schema::request::End {
                                response_stop_reason: Some(
                                    ai_schema::request::AiStopReason::Stop,
                                ),
                                response_model: None,
                                response_id: None,
                                http_status_code: None,
                                usage_input_tokens: None,
                                usage_output_tokens: None,
                                usage_cache_read_tokens: None,
                                usage_cache_write_tokens: None,
                                usage_reasoning_tokens: None,
                                usage_total_tokens: None,
                                usage_cost: None,
                                stream_chunk_count: None,
                                stream_time_to_first_chunk_ms: None,
                                error_type: None,
                            });
                            Ok::<(), std::io::Error>(())
                        },
                    )
                    .await
                    .expect("the request span body always settles");
                Ok::<(), std::io::Error>(())
            },
        )
        .await
        .expect("the step span body always settles");
}

/// The harness helpers fetch the telemetry parent from the harness context
/// and hand the body the span plus a child context, upstream's `startAiSpan`
/// and `startHarnessSpan` tests.
#[tokio::test]
async fn starts_ai_and_harness_spans_through_the_harness_context() {
    let telemetry = TelemetryHandle::new(pi_telemetry::InMemoryTelemetryContext::default());
    let context = with_telemetry_context(telemetry.clone(), &background_context());
    let () = start_ai_span(
        ai_schema::request::Span,
        ai_schema::request::Start {
            operation: ai_schema::request::AiOperation::Stream,
            provider: "provider".to_owned(),
            model: "model".to_owned(),
            api: "api".to_owned(),
            streaming: true,
            deferred: None,
        },
        |span, _child_context| async move {
            span.set_attributes(ai_schema::request::End {
                response_model: None,
                response_id: None,
                response_stop_reason: Some(ai_schema::request::AiStopReason::ToolUse),
                http_status_code: None,
                usage_input_tokens: None,
                usage_output_tokens: None,
                usage_cache_read_tokens: None,
                usage_cache_write_tokens: None,
                usage_reasoning_tokens: None,
                usage_total_tokens: None,
                usage_cost: None,
                stream_chunk_count: None,
                stream_time_to_first_chunk_ms: None,
                error_type: None,
            }
            .into_span_attributes());
            Ok::<(), std::io::Error>(())
        },
        &context,
    )
    .await
    .expect("the AI span body always settles");

    let () = start_harness_span(
        harness_schema::run::Span,
        harness_schema::run::Start {
            session_id: "session".to_owned(),
            lane_name: "main".to_owned(),
            operation_id: "operation".to_owned(),
            operation_recovery: false,
            operation_kind: harness_schema::run::RunKind::Run,
        },
        |span, _child_context| async move {
            span.set_attributes(harness_schema::run::End {
                operation_outcome: Some(harness_schema::run::RunOutcome::Completed),
                error_code: None,
                error_type: None,
            }
            .into_span_attributes());
            Ok::<(), std::io::Error>(())
        },
        &context,
    )
    .await
    .expect("the harness span body always settles");
}

/// The session-write span's per-span literals carry the write counters,
/// upstream's `WriteStart`/`WriteEnd` fixture checks; the structs
/// themselves are the compile-time check.
#[test]
fn session_write_literals_carry_the_write_counts() {
    let write_start = harness_schema::session_write::Start {
        session_id: "session".to_owned(),
        lane_name: None,
        operation_id: None,
        session_item_count: 2,
        session_item_kinds: vec!["entry".to_owned(), "value".to_owned(), "list".to_owned()],
    };
    let write_end = harness_schema::session_write::End {
        session_first_seq: Some(1),
        session_last_seq: Some(2),
    };
    assert_eq!(write_start.session_item_count, 2);
    assert_eq!(write_end.session_last_seq, Some(2));
    let _ = write_start.into_span_attributes();
    let _ = write_end.into_span_attributes();
}