//! Dispatch-handle behavior over the in-memory adapter: settlement, erased
//! failure details, child spans, and the typed helper's value transport.

use std::io;
use std::sync::{Arc, Mutex, MutexGuard};

use pi_telemetry::dispatch::{
    BoxedSpanBody, DynSpanHandle, ErasedSpanFuture, SpanBodyError, SpanBodyFailure, TelemetryHandle,
};
use pi_telemetry::{
    AttributeValue, InMemoryTelemetryContext, NOOP_TELEMETRY_CONTEXT, RecordedTelemetrySpan,
    SpanAttributes, SpanOptions, SpanStatus,
};

/// Locks a test mutex, recovering from poisoning like the adapters do.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn attributes(pairs: &[(&str, &str)]) -> SpanAttributes {
    let mut attributes = SpanAttributes::new();
    for (name, value) in pairs {
        attributes.insert(
            name.to_string(),
            Some(AttributeValue::Str(value.to_string())),
        );
    }
    attributes
}

fn only(spans: &[RecordedTelemetrySpan]) -> RecordedTelemetrySpan {
    assert_eq!(spans.len(), 1);
    spans[0].clone()
}

fn expect_ok<T, E>(outcome: Result<T, E>) -> T {
    #[allow(
        clippy::option_if_let_else,
        reason = "the match reads clearer than a map_or_else with an identity closure"
    )]
    match outcome {
        Ok(value) => value,
        Err(_) => unreachable!("expected an Ok outcome"),
    }
}

fn expect_err<T, E>(outcome: Result<T, E>) -> E {
    #[allow(
        clippy::option_if_let_else,
        reason = "the match reads clearer than a map_or_else with an identity closure"
    )]
    match outcome {
        Err(error) => error,
        Ok(_) => unreachable!("expected an Err outcome"),
    }
}

#[tokio::test]
async fn boxed_body_records_and_settles_ok() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let boxed: BoxedSpanBody = Box::new(|span: DynSpanHandle| {
        span.set_attributes(attributes(&[("model", "gpt")]));
        span.add_event("provider.request", SpanAttributes::new());
        let done: ErasedSpanFuture = Box::pin(async { Ok(()) });
        done
    });
    let outcome = handle
        .start_span_erased(SpanOptions::new("llm.request"), boxed)
        .await;
    expect_ok(outcome);
    let span = only(&adapter.get_spans());
    assert_eq!(span.name, "llm.request");
    assert!(span.parent_id.is_none());
    assert_eq!(span.attributes, attributes(&[("model", "gpt")]));
    assert_eq!(span.events.len(), 1);
    assert_eq!(span.events[0].name, "provider.request");
    assert_eq!(span.status, SpanStatus::Ok);
    assert!(span.settled);
}

#[tokio::test]
async fn erased_failure_keeps_name_and_message() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let boxed: BoxedSpanBody = Box::new(|_span: DynSpanHandle| {
        let failed: ErasedSpanFuture = Box::pin(async {
            Err(SpanBodyError {
                name: "RateLimitError".to_string(),
                message: "429 too many requests".to_string(),
            })
        });
        failed
    });
    let outcome = handle
        .start_span_erased(SpanOptions::new("llm.request"), boxed)
        .await;
    assert_eq!(
        outcome,
        Err(SpanBodyError {
            name: "RateLimitError".to_string(),
            message: "429 too many requests".to_string(),
        })
    );
    let span = only(&adapter.get_spans());
    assert_eq!(
        span.status,
        SpanStatus::Error {
            error: Some(pi_telemetry::SpanError {
                name: "RateLimitError".to_string(),
                message: "429 too many requests".to_string(),
            }),
        }
    );
}

#[tokio::test]
async fn typed_helper_returns_the_body_value() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let value = expect_ok(
        handle
            .start_span(
                SpanOptions::new("request"),
                |span: DynSpanHandle| async move {
                    span.add_event("seen", SpanAttributes::new());
                    Ok::<u32, io::Error>(42)
                },
            )
            .await,
    );
    assert_eq!(value, 42);
    let span = only(&adapter.get_spans());
    assert_eq!(span.name, "request");
    assert_eq!(span.events.len(), 1);
    assert_eq!(span.status, SpanStatus::Ok);
}

#[tokio::test]
async fn typed_helper_projects_errors_with_details() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let outcome: Result<u32, io::Error> = handle
        .start_span(SpanOptions::new("request"), |_span| async {
            Err(io::Error::other("connection reset"))
        })
        .await;
    let error = expect_err(outcome);
    assert_eq!(error.to_string(), "connection reset");
    let span = only(&adapter.get_spans());
    assert_eq!(
        span.status,
        SpanStatus::Error {
            error: Some(pi_telemetry::SpanError {
                name: std::any::type_name::<io::Error>().to_string(),
                message: "connection reset".to_string(),
            }),
        }
    );
}

#[tokio::test]
async fn child_spans_parent_through_the_erased_handle() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let boxed: BoxedSpanBody = Box::new(|span: DynSpanHandle| {
        let child_body: BoxedSpanBody = Box::new(|child: DynSpanHandle| {
            child.set_attributes(attributes(&[("child", "yes")]));
            let done: ErasedSpanFuture = Box::pin(async { Ok(()) });
            done
        });
        let child = span.start_span(SpanOptions::new("child"), child_body);
        let parent_done: ErasedSpanFuture = Box::pin(async move {
            expect_ok(child.await);
            Ok(())
        });
        parent_done
    });
    let outcome = handle
        .start_span_erased(SpanOptions::new("parent"), boxed)
        .await;
    expect_ok(outcome);
    let spans = adapter.get_spans();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].name, "parent");
    assert_eq!(spans[1].name, "child");
    assert_eq!(spans[1].parent_id, Some(spans[0].id));
    let parent_end = spans[0].end_sequence;
    let child_end = spans[1].end_sequence;
    assert!(child_end.zip(parent_end).is_some_and(|(c, p)| c < p));
}

#[tokio::test]
async fn noop_adapter_preserves_the_body_value() {
    let handle = TelemetryHandle::new(NOOP_TELEMETRY_CONTEXT);
    let value = expect_ok(
        handle
            .start_span(SpanOptions::new("request"), |_span| async {
                Ok::<_, io::Error>("kept")
            })
            .await,
    );
    assert_eq!(value, "kept");
}

#[tokio::test]
async fn explicit_status_survives_a_failed_body() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let outcome: Result<u32, io::Error> = handle
        .start_span(SpanOptions::new("request"), |span| async move {
            span.set_status(SpanStatus::Error {
                error: Some(pi_telemetry::SpanError {
                    name: "Explicit".to_string(),
                    message: "recorded before failing".to_string(),
                }),
            });
            Err(io::Error::other("later failure"))
        })
        .await;
    assert!(outcome.is_err());
    let span = only(&adapter.get_spans());
    assert_eq!(
        span.status,
        SpanStatus::Error {
            error: Some(pi_telemetry::SpanError {
                name: "Explicit".to_string(),
                message: "recorded before failing".to_string(),
            }),
        }
    );
}

/// A non-`std::error::Error` failure type to pin the manual
/// [`SpanBodyFailure`] path.
struct Unprojected;

impl SpanBodyFailure for Unprojected {
    fn span_body_failure(&self) -> SpanBodyError {
        SpanBodyError {
            name: "Unprojected".to_string(),
            message: "manual projection".to_string(),
        }
    }
}

#[tokio::test]
async fn manual_failure_projection_flows_through() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let outcome: Result<u32, Unprojected> = handle
        .start_span(SpanOptions::new("request"), |_span| async {
            Err(Unprojected)
        })
        .await;
    expect_err(outcome);
    let span = only(&adapter.get_spans());
    assert_eq!(
        span.status,
        SpanStatus::Error {
            error: Some(pi_telemetry::SpanError {
                name: "Unprojected".to_string(),
                message: "manual projection".to_string(),
            }),
        }
    );
}

/// The handle is `Clone` and clones share one adapter: recordings made through
/// one clone are visible to the other's snapshot reader.
#[tokio::test]
async fn cloned_handles_share_one_adapter() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let cloned = handle.clone();
    let outcome: Result<(), io::Error> = cloned
        .start_span(SpanOptions::new("shared"), |_span: DynSpanHandle| async {
            Ok(())
        })
        .await;
    expect_ok(outcome);
    assert_eq!(adapter.get_spans().len(), 1);
}

/// Children of settled spans record nowhere reachable; through the erased
/// layer the same delegation holds, so a late child body still runs.
#[tokio::test]
async fn child_of_settled_parent_runs_unobserved() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let escaped: Arc<Mutex<Option<DynSpanHandle>>> = Arc::new(Mutex::new(None));
    let escaped_for_body = Arc::clone(&escaped);
    let boxed: BoxedSpanBody = Box::new(move |span: DynSpanHandle| {
        *lock(&escaped_for_body) = Some(span);
        let done: ErasedSpanFuture = Box::pin(async { Ok(()) });
        done
    });
    let outcome = handle
        .start_span_erased(SpanOptions::new("settled.parent"), boxed)
        .await;
    expect_ok(outcome);
    // The root settled after its body; a child started from the settled
    // handle records into a throwaway state nothing can observe.
    let escaped_span = lock(&escaped).take();
    let Some(settled_span) = escaped_span else {
        unreachable!("the body must escape its span handle")
    };
    let child_body: BoxedSpanBody = Box::new(|_child: DynSpanHandle| {
        let done: ErasedSpanFuture = Box::pin(async { Ok(()) });
        done
    });
    let child = settled_span.start_span(SpanOptions::new("late.child"), child_body);
    expect_ok(child.await);
    assert_eq!(adapter.get_spans().len(), 1);
    assert_eq!(only(&adapter.get_spans()).name, "settled.parent");
}

#[tokio::test]
async fn child_failure_records_details_on_the_child() {
    let adapter = InMemoryTelemetryContext::new();
    let handle = TelemetryHandle::new(adapter.clone());
    let boxed: BoxedSpanBody = Box::new(|span: DynSpanHandle| {
        let child_body: BoxedSpanBody = Box::new(|_child: DynSpanHandle| {
            let failed: ErasedSpanFuture = Box::pin(async {
                Err(SpanBodyError {
                    name: "ChildBoom".to_string(),
                    message: "child failed".to_string(),
                })
            });
            failed
        });
        let child = span.start_span(SpanOptions::new("child"), child_body);
        let parent_done: ErasedSpanFuture = Box::pin(async move {
            // The child's failure settles the child; the parent body succeeds.
            let _ = child.await;
            Ok(())
        });
        parent_done
    });
    let outcome = handle
        .start_span_erased(SpanOptions::new("parent"), boxed)
        .await;
    expect_ok(outcome);
    let spans = adapter.get_spans();
    assert_eq!(spans.len(), 2);
    let child = &spans[1];
    assert_eq!(
        child.status,
        SpanStatus::Error {
            error: Some(pi_telemetry::SpanError {
                name: "ChildBoom".to_string(),
                message: "child failed".to_string(),
            }),
        }
    );
    assert_eq!(spans[0].status, SpanStatus::Ok, "the parent body succeeded");
}
