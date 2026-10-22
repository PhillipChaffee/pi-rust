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
