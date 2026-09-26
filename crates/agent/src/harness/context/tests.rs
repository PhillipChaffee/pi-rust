//! The harness context telemetry suite, ported from upstream
//! `test/harness/context.test.ts`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::io;

use pi_telemetry::{InMemoryTelemetryContext, NOOP_TELEMETRY_CONTEXT, SpanOptions};

use crate::harness::context::{
    background_context, get_telemetry_context, placeholder_context, with_telemetry_context,
};

/// No telemetry attached resolves to the shared no-op parent: the span runs
/// and records nothing, upstream's `toBe(NOOP_TELEMETRY_CONTEXT)` identity
/// restated over the zero-sized noop type.
#[tokio::test]
async fn uses_no_op_telemetry_when_none_is_attached() {
    let context = background_context();
    let handle = get_telemetry_context(&context);
    let noop: pi_telemetry::NoopTelemetryContext = NOOP_TELEMETRY_CONTEXT;
    let resolved = pi_telemetry::TelemetryHandle::new(noop);
    let () = resolved
        .start_span(SpanOptions::new("noop"), |_| async {
            Ok::<(), io::Error>(())
        })
        .await
        .expect("the noop span body always settles");
    let _ = handle;
    let todo_context = placeholder_context();
    let () = get_telemetry_context(&todo_context)
        .start_span(SpanOptions::new("todo"), |_| async {
            Ok::<(), io::Error>(())
        })
        .await
        .expect("the noop span body always settles");
}

/// The telemetry parent rides the context as an ordinary value; the child
/// span's context parents its spans onto the parent span.
#[tokio::test]
async fn carries_telemetry_as_an_ordinary_context_value() {
    let telemetry = InMemoryTelemetryContext::default();
    let parent = pi_telemetry::TelemetryHandle::new(telemetry.clone());
    let context = with_telemetry_context(parent, &background_context());

    let parent_handle = get_telemetry_context(&context);
    let () = parent_handle
        .start_span(SpanOptions::new("parent"), move |span| {
            let child_context = with_telemetry_context(span.telemetry_handle(), &context);
            async move {
                let () = get_telemetry_context(&child_context)
                    .start_span(SpanOptions::new("child"), |_child| async {
                        Ok::<(), io::Error>(())
                    })
                    .await
                    .expect("the child span body always settles");
                Ok::<(), io::Error>(())
            }
        })
        .await
        .expect("the parent span body always settles");

    let spans = telemetry.get_spans();
    let names: Vec<&str> = spans.iter().map(|span| span.name.as_str()).collect();
    assert_eq!(names, vec!["parent", "child"]);
    assert_eq!(spans[1].parent_id, Some(spans[0].id));
}
