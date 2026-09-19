//! Ports `packages/telemetry/test/conformance.test.ts` one to one, minus
//! the Proxy-hostility cases the Rust adapter contract upholds statically
//! (see [`pi_telemetry::testing`]).

use pi_telemetry::memory::InMemoryTelemetryContext;
use pi_telemetry::testing::{TelemetryAdapterFixture, create_telemetry_adapter_conformance};
use pi_telemetry::{AttributeValue, SpanAttributes, SpanOptions, TelemetryContext, TelemetrySpan};

fn in_memory_fixture() -> TelemetryAdapterFixture<InMemoryTelemetryContext> {
    let context = InMemoryTelemetryContext::new();
    let reader = context.clone();
    TelemetryAdapterFixture {
        context,
        get_spans: Box::new(move || reader.get_spans()),
    }
}

/// Runs every conformance case against the in-memory adapter, grouped as
/// upstream's suite groups them.
#[tokio::test]
async fn in_memory_telemetry_context_conformance() {
    let conformance = create_telemetry_adapter_conformance(in_memory_fixture);
    let mut groups: Vec<&str> = Vec::new();
    for case in &conformance {
        if !groups.contains(&case.group) {
            groups.push(case.group);
        }
    }
    for group in groups {
        for case in conformance.iter().filter(|case| case.group == group) {
            case.run().await;
        }
    }
}

/// Returns one array-of-strings attribute entry.
fn str_array(name: &str, values: &[&str]) -> (String, Option<AttributeValue>) {
    (
        name.to_owned(),
        Some(AttributeValue::StrArray(
            values.iter().map(|value| (*value).to_owned()).collect(),
        )),
    )
}

/// Ports the detached-snapshot immutability case.
#[tokio::test]
async fn returns_detached_snapshots_without_exposing_mutable_recording_state() {
    let context = InMemoryTelemetryContext::new();
    let mut start_attributes = SpanAttributes::new();
    let (tags_name, tags_value) = str_array("tags", &["initial"]);
    start_attributes.insert(tags_name, tags_value);

    let opened = context
        .start_span(
            SpanOptions::new("snapshot").with_attributes(start_attributes),
            |span| {
                let reader = context.clone();
                async move {
                    span.add_event("event", int_attributes("value", 1));
                    let open = reader.get_spans();
                    Ok::<_, ()>((open[0].settled, open[0].end_sequence))
                }
            },
        )
        .await;
    let Ok((open_settled, open_end_sequence)) = opened else {
        unreachable!("the snapshot body succeeds")
    };
    assert!(!open_settled);
    assert_eq!(open_end_sequence, None);

    let mut spans = context.get_spans();
    assert!(spans[0].settled);
    assert_eq!(spans[0].end_sequence, Some(1));

    // Mutating the detached snapshot must not reach the recording state.
    let (tags_name, tags_value) = str_array("tags", &["mutated"]);
    spans[0].attributes.insert(tags_name, tags_value);
    spans[0].events[0]
        .attributes
        .insert("value".to_owned(), Some(AttributeValue::Int(2)));

    let second = context.get_spans();
    let mut expected_attributes = SpanAttributes::new();
    let (tags_name, tags_value) = str_array("tags", &["initial"]);
    expected_attributes.insert(tags_name, tags_value);
    assert_eq!(second[0].attributes, expected_attributes);
    assert_eq!(
        second[0].events,
        vec![event("event", int_attributes("value", 1))]
    );
}

fn event(name: &str, attributes: SpanAttributes) -> pi_telemetry::memory::RecordedTelemetryEvent {
    pi_telemetry::memory::RecordedTelemetryEvent {
        name: name.to_owned(),
        attributes,
    }
}

fn int_attributes(name: &str, value: i64) -> SpanAttributes {
    let mut attributes = SpanAttributes::new();
    attributes.insert(name.to_owned(), Some(AttributeValue::Int(value)));
    attributes
}
