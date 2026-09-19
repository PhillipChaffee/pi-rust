//! The adapter conformance suite, ported from upstream's `./testing` subpath.
//!
//! [`create_telemetry_adapter_conformance`] builds runner-independent cases
//! from a fixture factory, so any adapter can be driven through the
//! callback-span contract by any test framework. The fixture couples to the
//! recorded-snapshot shape ([`RecordedTelemetrySpan`]), exactly as upstream
//! couples to its in-memory record type; other adapters expose an
//! equivalent snapshot reader.
//!
//! Three upstream case families are absent because Rust upholds their
//! contract statically:
//!
//! - *Unreadable payload passivity* (2 cases): upstream simulates hostile
//!   payloads whose property access throws, and asserts recording survives.
//!   Recording here takes owned plain data, so hostile payloads cannot
//!   exist; "recording never propagates payload failures" is a property of
//!   the API shape, not a runtime behavior to test.
//! - *Atomic failure of attribute/status calls* (2 cases): upstream asserts
//!   a failed merge leaves state untouched. There is no failing merge for
//!   owned plain data; the `None`-entry skip rule on [`SpanAttributes`] is
//!   the whole of the atomicity axis, covered by the recording case.
//!
//! The synchronous-admission case is adapted rather than dropped: upstream
//! asserts the callback runs before the promise resolves; Rust registers
//! the span synchronously and runs the body on first poll, so the case
//! asserts registration is observable before the future is awaited.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::memory::{RecordedTelemetryEvent, RecordedTelemetrySpan, lock};
use crate::{
    AttributeValue, SpanAttributes, SpanError, SpanOptions, SpanStatus, TelemetryContext,
    TelemetrySpan,
};

/// A rejection marker for the rejection-identity case: distinct values
/// stand in for upstream's distinct error objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rejection {
    kind: &'static str,
}

impl Rejection {
    /// Stands in for a synchronous throw.
    pub const SYNC: Self = Self { kind: "sync" };

    /// Stands in for an asynchronous rejection.
    pub const ASYNC: Self = Self { kind: "async" };
}

/// A fresh adapter instance and normalized snapshot reader owned by one
/// conformance case.
pub struct TelemetryAdapterFixture<C: TelemetryContext> {
    /// The adapter under test.
    pub context: C,
    /// Reads the adapter's recorded spans as detached snapshots.
    pub get_spans: Box<dyn Fn() -> Vec<RecordedTelemetrySpan>>,
}

impl<C: TelemetryContext> std::fmt::Debug for TelemetryAdapterFixture<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TelemetryAdapterFixture")
            .finish_non_exhaustive()
    }
}

/// A runner-independent conformance case, runnable by any test framework.
pub struct TelemetryAdapterConformanceCase {
    /// The contract area this case pins, upstream's group.
    pub group: &'static str,
    /// The one-sentence contract statement this case asserts.
    pub name: &'static str,
    runner: Box<dyn Fn() -> Pin<Box<dyn Future<Output = ()>>>>,
}

impl std::fmt::Debug for TelemetryAdapterConformanceCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TelemetryAdapterConformanceCase")
            .field("group", &self.group)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl TelemetryAdapterConformanceCase {
    /// Runs the case on a fresh adapter fixture.
    #[must_use]
    pub fn run(&self) -> Pin<Box<dyn Future<Output = ()>>> {
        (self.runner)()
    }
}

/// Creates the conformance cases for the callback telemetry adapter
/// contract, each running against a fresh fixture from `factory`.
///
/// # Panics
/// A case panics when its adapter breaks the contract it pins, exactly like
/// the assertion failure it replaces.
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "the case list mirrors upstream's conformance file one to one"
)]
pub fn create_telemetry_adapter_conformance<F, C>(
    factory: F,
) -> Vec<TelemetryAdapterConformanceCase>
where
    F: Fn() -> TelemetryAdapterFixture<C> + Clone + 'static,
    C: TelemetryContext + 'static,
    C::Span: Send,
{
    vec![
        case(
            factory.clone(),
            "callback lifecycle",
            "registers synchronously and preserves the result",
            |fixture| {
                Box::pin(async move {
                    let TelemetryAdapterFixture { context, get_spans } = fixture;
                    let future = context.start_span(SpanOptions::new("success"), |_| async {
                        Ok::<_, Rejection>(EXPECTED)
                    });
                    // Registration is synchronous: the unsettled span is
                    // observable before the body has run.
                    assert_eq!(get_spans().len(), 1);
                    let result = future.await;
                    assert_eq!(result, Ok(EXPECTED));
                    let spans = get_spans();
                    let span = find_span(&spans, "success");
                    assert_eq!(span.status, SpanStatus::Ok);
                    assert!(span.settled);
                })
            },
        ),
        case(
            factory.clone(),
            "callback lifecycle",
            "preserves synchronous and asynchronous rejection values",
            |fixture| {
                Box::pin(async move {
                    let TelemetryAdapterFixture { context, get_spans } = fixture;
                    let sync_result = context
                        .start_span(SpanOptions::new("sync-error"), |_| async {
                            Err::<(), Rejection>(Rejection::SYNC)
                        })
                        .await;
                    assert_eq!(sync_result, Err(Rejection::SYNC));

                    let async_result = context
                        .start_span(SpanOptions::new("async-error"), |_| async {
                            std::future::ready(()).await;
                            Err::<(), Rejection>(Rejection::ASYNC)
                        })
                        .await;
                    assert_eq!(async_result, Err(Rejection::ASYNC));

                    let empty_result = context
                        .start_span(SpanOptions::new("empty-error"), |_| async {
                            Err::<(), ()>(())
                        })
                        .await;
                    assert_eq!(empty_result, Err(()));

                    let spans = get_spans();
                    for name in ["sync-error", "async-error", "empty-error"] {
                        assert!(
                            matches!(find_span(&spans, name).status, SpanStatus::Error { .. }),
                            "expected {name} to settle with an error status"
                        );
                    }
                })
            },
        ),
        case(
            factory.clone(),
            "status",
            "uses last explicit status without automatic overwrite",
            |fixture| {
                Box::pin(async move {
                    let TelemetryAdapterFixture { context, get_spans } = fixture;
                    let last_status = context
                        .start_span(SpanOptions::new("last-status"), |span| async move {
                            span.set_status(SpanStatus::Error {
                                error: Some(error_detail("first")),
                            });
                            span.set_status(SpanStatus::Ok);
                            Ok::<(), Rejection>(())
                        })
                        .await;
                    assert_eq!(last_status, Ok(()));

                    let explicit_before_throw = context
                        .start_span(
                            SpanOptions::new("explicit-before-throw"),
                            |span| async move {
                                span.set_status(SpanStatus::Ok);
                                Err::<(), Rejection>(Rejection::SYNC)
                            },
                        )
                        .await;
                    assert_eq!(explicit_before_throw, Err(Rejection::SYNC));

                    let explicit_before_rejection = context
                        .start_span(
                            SpanOptions::new("explicit-before-rejection"),
                            |span| async move {
                                span.set_status(SpanStatus::Error {
                                    error: Some(error_detail("async failure")),
                                });
                                Err::<(), Rejection>(Rejection::ASYNC)
                            },
                        )
                        .await;
                    assert_eq!(explicit_before_rejection, Err(Rejection::ASYNC));

                    let expected_failure = context
                        .start_span(SpanOptions::new("expected-failure"), |span| async move {
                            span.set_status(SpanStatus::Error {
                                error: Some(error_detail("returned failure")),
                            });
                            Ok::<(), Rejection>(())
                        })
                        .await;
                    assert_eq!(expected_failure, Ok(()));

                    let spans = get_spans();
                    assert_eq!(find_span(&spans, "last-status").status, SpanStatus::Ok);
                    assert_eq!(
                        find_span(&spans, "explicit-before-throw").status,
                        SpanStatus::Ok
                    );
                    assert_eq!(
                        find_span(&spans, "explicit-before-rejection").status,
                        SpanStatus::Error {
                            error: Some(error_detail("async failure")),
                        }
                    );
                    assert_eq!(
                        find_span(&spans, "expected-failure").status,
                        SpanStatus::Error {
                            error: Some(error_detail("returned failure")),
                        }
                    );
                })
            },
        ),
        case(
            factory.clone(),
            "recording",
            "merges attributes and records ordered events",
            |fixture| {
                Box::pin(async move {
                    let TelemetryAdapterFixture { context, get_spans } = fixture;
                    let mut start = SpanAttributes::new();
                    start.insert("start".to_owned(), Some(AttributeValue::from("value")));
                    start.insert("overwrite".to_owned(), Some(AttributeValue::from("start")));
                    start.insert("ignored".to_owned(), None);
                    let recording = context
                        .start_span(
                            SpanOptions::new("recording").with_attributes(start),
                            |span| async move {
                                span.set_attributes({
                                    let mut attributes = SpanAttributes::new();
                                    attributes
                                        .insert("count".to_owned(), Some(AttributeValue::Int(1)));
                                    attributes.insert(
                                        "overwrite".to_owned(),
                                        Some(AttributeValue::from("middle")),
                                    );
                                    attributes
                                });
                                span.set_attributes({
                                    let mut attributes = SpanAttributes::new();
                                    attributes.insert("count".to_owned(), None);
                                    attributes.insert(
                                        "overwrite".to_owned(),
                                        Some(AttributeValue::from("end")),
                                    );
                                    attributes
                                });
                                span.add_event("first", {
                                    let mut attributes = SpanAttributes::new();
                                    attributes
                                        .insert("index".to_owned(), Some(AttributeValue::Int(1)));
                                    attributes.insert("ignored".to_owned(), None);
                                    attributes
                                });
                                span.add_event("second", {
                                    let mut attributes = SpanAttributes::new();
                                    attributes
                                        .insert("index".to_owned(), Some(AttributeValue::Int(2)));
                                    attributes
                                });
                                Ok::<(), Rejection>(())
                            },
                        )
                        .await;
                    assert_eq!(recording, Ok(()));

                    let recorded = get_spans();
                    let span = find_span(&recorded, "recording");
                    let mut expected_attributes = SpanAttributes::new();
                    expected_attributes
                        .insert("start".to_owned(), Some(AttributeValue::from("value")));
                    expected_attributes
                        .insert("overwrite".to_owned(), Some(AttributeValue::from("end")));
                    expected_attributes.insert("count".to_owned(), Some(AttributeValue::Int(1)));
                    assert_eq!(span.attributes, expected_attributes);
                    assert_eq!(
                        span.events,
                        vec![
                            event("first", int_attributes("index", 1)),
                            event("second", int_attributes("index", 2)),
                        ]
                    );
                })
            },
        ),
        case(
            factory.clone(),
            "inert calls",
            "makes calls after settlement inert",
            |fixture| {
                Box::pin(async move {
                    let TelemetryAdapterFixture { context, get_spans } = fixture;
                    let escaped = &mut None::<C::Span>;
                    let settled = context
                        .start_span(
                            SpanOptions::new("settled")
                                .with_attributes(str_attributes("value", "initial")),
                            |span| {
                                let slot = &mut *escaped;
                                async move {
                                    *slot = Some(span);
                                    Ok::<(), Rejection>(())
                                }
                            },
                        )
                        .await;
                    assert_eq!(settled, Ok(()));
                    let captured = escaped.as_ref().map_or_else(
                        || unreachable!("body must capture the settled span handle"),
                        C::Span::clone,
                    );

                    captured.set_attributes(str_attributes("value", "late"));
                    captured.add_event("late", bool_attributes("value", true));
                    captured.set_status(SpanStatus::Error { error: None });
                    let child_result = captured
                        .start_span(SpanOptions::new("late-child"), |_| async {
                            Ok::<i64, Rejection>(7)
                        })
                        .await;
                    assert_eq!(child_result, Ok(7));

                    let spans = get_spans();
                    assert_eq!(spans.len(), 1);
                    let recorded = &spans[0];
                    assert_eq!(recorded.attributes, str_attributes("value", "initial"));
                    assert!(recorded.events.is_empty());
                    assert_eq!(recorded.status, SpanStatus::Ok);
                })
            },
        ),
        case(
            factory,
            "parentage",
            "records nested and concurrent child relationships",
            |fixture| {
                Box::pin(async move {
                    let TelemetryAdapterFixture { context, get_spans } = fixture;
                    let (release, await_gate) = gate();
                    let parented = context
                        .start_span(SpanOptions::new("parent"), |parent| async move {
                            let first = parent.start_span(
                                SpanOptions::new("first-child"),
                                move |_| async move {
                                    await_gate.await;
                                    Ok::<(), Rejection>(())
                                },
                            );
                            let second = parent
                                .start_span(SpanOptions::new("second-child"), |_| async {
                                    Ok::<_, Rejection>("done")
                                });
                            assert_eq!(second.await, Ok("done"));
                            release.release();
                            first.await?;
                            Ok::<(), Rejection>(())
                        })
                        .await;
                    assert_eq!(parented, Ok(()));

                    let spans = get_spans();
                    let parent = find_span(&spans, "parent");
                    let first = find_span(&spans, "first-child");
                    let second = find_span(&spans, "second-child");
                    assert_eq!(parent.parent_id, None);
                    assert_eq!(first.parent_id, Some(parent.id));
                    assert_eq!(second.parent_id, Some(parent.id));
                    let parent_end = end_sequence(parent);
                    let first_end = end_sequence(first);
                    let second_end = end_sequence(second);
                    assert!(second_end < first_end);
                    assert!(first_end < parent_end);
                })
            },
        ),
    ]
}

/// A release gate for the parentage case: the first child's body awaits the
/// gate while the parent completes the second child, then releases it. The
/// adapter contract is runtime-free, so the gate is a std-only waker future
/// rather than a tokio channel.
struct GateState {
    released: bool,
    waker: Option<std::task::Waker>,
}

struct GateRelease {
    state: Arc<Mutex<GateState>>,
}

impl GateRelease {
    fn release(&self) {
        let mut state = lock(&self.state);
        state.released = true;
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }
}

struct GateAwait {
    state: Arc<Mutex<GateState>>,
}

impl Future for GateAwait {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<()> {
        let mut state = lock(&self.state);
        if state.released {
            return std::task::Poll::Ready(());
        }
        state.waker = Some(cx.waker().clone());
        std::task::Poll::Pending
    }
}

fn gate() -> (GateRelease, GateAwait) {
    let state = Arc::new(Mutex::new(GateState {
        released: false,
        waker: None,
    }));
    (
        GateRelease {
            state: Arc::clone(&state),
        },
        GateAwait { state },
    )
}

const EXPECTED: &str = "expected";

/// Builds one conformance case whose body runs against a fresh fixture.
fn case<C, F, B>(
    factory: F,
    group: &'static str,
    name: &'static str,
    body: B,
) -> TelemetryAdapterConformanceCase
where
    F: Fn() -> TelemetryAdapterFixture<C> + 'static,
    C: TelemetryContext + 'static,
    C::Span: Send,
    B: Fn(TelemetryAdapterFixture<C>) -> Pin<Box<dyn Future<Output = ()>>> + 'static,
{
    TelemetryAdapterConformanceCase {
        group,
        name,
        runner: Box::new(move || {
            let fixture = factory();
            body(fixture)
        }),
    }
}

/// Finds a recorded span by name; a missing span is a conformance-case bug.
#[allow(
    clippy::panic,
    reason = "a missing recorded span is a conformance-case bug, reported with the span name"
)]
fn find_span<'a>(spans: &'a [RecordedTelemetrySpan], name: &str) -> &'a RecordedTelemetrySpan {
    let span = spans.iter().find(|candidate| candidate.name == name);
    span.unwrap_or_else(|| panic!("expected recorded span {name}"))
}

/// Unwraps a settlement sequence, which every settled span carries.
#[allow(
    clippy::panic,
    reason = "a settled span without a sequence is a conformance-case bug"
)]
fn end_sequence(span: &RecordedTelemetrySpan) -> u64 {
    #[allow(
        clippy::option_if_let_else,
        reason = "the match reads clearer than a map_or_else with an identity closure"
    )]
    match span.end_sequence {
        Some(sequence) => sequence,
        None => panic!("expected settled span {} with end sequence", span.name),
    }
}

fn event(name: &str, attributes: SpanAttributes) -> RecordedTelemetryEvent {
    RecordedTelemetryEvent {
        name: name.to_owned(),
        attributes,
    }
}

fn int_attributes(name: &str, value: i64) -> SpanAttributes {
    let mut attributes = SpanAttributes::new();
    attributes.insert(name.to_owned(), Some(AttributeValue::Int(value)));
    attributes
}

fn str_attributes(name: &str, value: &str) -> SpanAttributes {
    let mut attributes = SpanAttributes::new();
    attributes.insert(name.to_owned(), Some(AttributeValue::from(value)));
    attributes
}

fn bool_attributes(name: &str, value: bool) -> SpanAttributes {
    let mut attributes = SpanAttributes::new();
    attributes.insert(name.to_owned(), Some(AttributeValue::Bool(value)));
    attributes
}

fn error_detail(message: &str) -> SpanError {
    SpanError {
        name: "Expected".to_owned(),
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_memory_fixture() -> TelemetryAdapterFixture<crate::memory::InMemoryTelemetryContext> {
        let context = crate::memory::InMemoryTelemetryContext::new();
        let reader = context.clone();
        TelemetryAdapterFixture {
            context,
            get_spans: Box::new(move || reader.get_spans()),
        }
    }

    /// Pins the conformance helper `Debug` impls.
    #[test]
    fn conformance_helpers_format_debug() {
        let fixture = in_memory_fixture();
        assert!(!format!("{fixture:?}").is_empty());
        let conformance = create_telemetry_adapter_conformance(in_memory_fixture);
        assert!(!format!("{:?}", conformance[0]).is_empty());
    }

    /// Exercises the gate's pending path, its waker release, and its ready
    /// poll, which the parentage case only reaches with concurrent adapters.
    #[test]
    fn gate_waits_then_releases() {
        let (release, wait) = gate();
        let mut wait = std::pin::pin!(wait);
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        assert!(matches!(
            Future::poll(wait.as_mut(), &mut cx),
            std::task::Poll::Pending
        ));
        release.release();
        assert!(matches!(
            Future::poll(wait.as_mut(), &mut cx),
            std::task::Poll::Ready(())
        ));
    }

    /// A missing recorded span is a conformance-case bug and must panic.
    #[test]
    #[should_panic(expected = "expected recorded span")]
    fn find_span_reports_missing_names() {
        find_span(&[], "missing");
    }

    /// A settled span without an end sequence is a conformance-case bug.
    #[test]
    #[should_panic(expected = "expected settled span")]
    fn end_sequence_reports_missing_sequences() {
        let span = RecordedTelemetrySpan {
            id: 1,
            parent_id: None,
            name: "unsettled".to_owned(),
            attributes: SpanAttributes::new(),
            events: Vec::new(),
            status: SpanStatus::Ok,
            settled: true,
            end_sequence: None,
        };
        let _ = end_sequence(&span);
    }
}
