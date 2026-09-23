//! The `EventStream` port, from `test/event-stream.test.ts` (pi #9055
//! regression semantics).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use common::event_type_name;

use common::bare_assistant_message;
use pi_ai::types::{AssistantMessageEvent, StopReason};
use pi_ai::utils::event_stream::{
    EventStream, assistant_message_event_stream, create_assistant_message_event_stream,
};

#[tokio::test]
async fn drains_buffered_events_in_order_and_ignores_events_pushed_after_completion() {
    let stream: EventStream<u32, u32> = EventStream::new(|event| *event == 3, |event| Some(*event));
    stream.push(1);
    stream.push(2);
    stream.push(3);
    stream.push(4);

    assert_eq!(stream.result().await, 3);

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    assert_eq!(events, [1, 2, 3]);
}

#[tokio::test]
async fn preserves_order_when_events_arrive_after_buffered_draining_starts() {
    let stream: EventStream<u32, u32> = EventStream::new(|_| false, |event| Some(*event));
    stream.push(1);
    stream.push(2);

    assert_eq!(stream.next().await, Some(1));

    stream.push(3);
    assert_eq!(stream.next().await, Some(2));
    assert_eq!(stream.next().await, Some(3));

    stream.end(Some(&3));
    assert_eq!(stream.next().await, None);
}

#[tokio::test]
async fn delivers_events_to_waiting_consumers_in_registration_order() {
    let stream: EventStream<u32, u32> = EventStream::new(|_| false, |event| Some(*event));

    let first = stream.clone();
    let second = stream.clone();
    let first_event = tokio::spawn(async move { first.next().await });
    let second_event = tokio::spawn(async move { second.next().await });

    // Yield once so both waiters register before the pushes.
    tokio::task::yield_now().await;

    stream.push(1);
    stream.push(2);

    assert_eq!(first_event.await.expect("first task"), Some(1));
    assert_eq!(second_event.await.expect("second task"), Some(2));
}

#[tokio::test]
async fn drains_buffered_events_after_end_and_resolves_the_explicit_result() {
    let stream: EventStream<u32, String> =
        EventStream::new(|_: &u32| false, |event: &u32| Some(event.to_string()));
    stream.push(1);
    stream.push(2);
    stream.end(Some(&String::from("complete")));

    assert_eq!(stream.result().await, "complete");

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    assert_eq!(events, [1, 2]);
}

#[tokio::test]
async fn wakes_all_waiting_consumers_when_ended_without_a_result() {
    let stream: EventStream<u32, u32> = EventStream::new(|_| false, |event| Some(*event));

    let first = stream.clone();
    let second = stream.clone();
    let first_event = tokio::spawn(async move { first.next().await });
    let second_event = tokio::spawn(async move { second.next().await });

    tokio::task::yield_now().await;

    stream.end(None);

    assert_eq!(first_event.await.expect("first task"), None);
    assert_eq!(second_event.await.expect("second task"), None);
}

#[tokio::test]
async fn the_assistant_stream_resolves_to_done_and_error_messages() {
    let stream = assistant_message_event_stream();

    let mut final_message = bare_assistant_message();
    final_message.stop_reason = StopReason::Stop;
    let done = final_message.clone();
    stream.push(AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: done.clone(),
    });
    stream.push(AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: String::from("late"),
        partial: bare_assistant_message(),
    });

    assert_eq!(stream.result().await, done);
    assert_eq!(
        stream.next().await.map(|event| event_type_name(&event)),
        Some("done")
    );
    assert_eq!(stream.next().await, None);

    let failing = assistant_message_event_stream();
    let mut failed = bare_assistant_message();
    failed.stop_reason = StopReason::Error;
    failed.error_message = Some(String::from("setup failed"));
    failing.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: failed.clone(),
    });
    assert_eq!(
        failing.result().await.error_message,
        Some(String::from("setup failed"))
    );
}

#[tokio::test]
async fn result_waiters_registered_before_completion_receive_the_settled_value() {
    let stream: EventStream<u32, u32> = EventStream::new(|event| *event == 7, |event| Some(*event));

    let first = stream.clone();
    let second = stream.clone();
    let first_result = tokio::spawn(async move { first.result().await });
    let second_result = tokio::spawn(async move { second.result().await });

    tokio::task::yield_now().await;

    stream.push(7);

    assert_eq![first_result.await.expect("first waiter"), 7];
    assert_eq![second_result.await.expect("second waiter"), 7];
}

#[tokio::test]
async fn a_completing_push_without_a_result_leaves_registered_waiters_parked() {
    // The completing event settles a `None` result (no extractable value);
    // result waiters are only woken with an actual value, so a caller
    // registered before the push stays parked until an explicit end
    // supplies one.
    let stream: EventStream<u32, u32> = EventStream::new(|event| *event == 7, |_| None);

    tokio::spawn({
        let stream = stream.clone();
        async move { stream.result().await }
    });
    tokio::task::yield_now().await;
    stream.push(7);

    // The stream is now done with no result; the waiter stays parked until
    // the explicit end result arrives, so drain iteration finishes first.
    assert_eq![stream.next().await, Some(7)];
    assert_eq![stream.next().await, None];
}

#[tokio::test]
async fn result_waiters_registered_before_an_explicit_end_receive_the_end_result() {
    let stream: EventStream<u32, String> =
        EventStream::new(|_: &u32| false, |event: &u32| Some(event.to_string()));

    let waiter = {
        let stream = stream.clone();
        tokio::spawn(async move { stream.result().await })
    };
    let queued_reader = {
        let stream = stream.clone();
        tokio::spawn(async move {
            let mut events = Vec::new();
            while let Some(event) = stream.next().await {
                events.push(event);
            }
            events
        })
    };

    tokio::task::yield_now().await;

    stream.push(1);
    stream.end(Some(&String::from("finished")));

    assert_eq![waiter.await.expect("the result waiter"), "finished"];
    assert_eq![queued_reader.await.expect("the queue reader"), [1]];
}

#[tokio::test]
async fn a_late_result_caller_after_settlement_reads_the_stored_value() {
    let stream: EventStream<u32, u32> = EventStream::new(|event| *event == 3, |event| Some(*event));
    stream.push(3);

    assert_eq![stream.result().await, 3];
    assert_eq![stream.result().await, 3, "settlement is repeatable"];
}

#[tokio::test]
async fn a_late_result_caller_after_end_without_a_result_never_resolves() {
    // end(None) leaves the final result unset, exactly like the upstream
    // promise left pending: a caller registered after the end parks
    // forever, so the test just pins that nothing resolves.
    let stream: EventStream<u32, u32> = EventStream::new(|_| false, |event| Some(*event));
    stream.end(None);
    let waiter = {
        let stream = stream.clone();
        tokio::spawn(async move { stream.result().await })
    };
    tokio::task::yield_now().await;
    assert![!waiter.is_finished(), "no result resolves"];
}

#[tokio::test]
async fn pushes_after_the_stream_has_ended_are_silently_dropped() {
    let stream: EventStream<u32, u32> = EventStream::new(|_| false, |event| Some(*event));
    stream.end(Some(&1));
    stream.push(2);
    assert_eq![stream.next().await, None];
    assert_eq![stream.result().await, 1];
}

#[tokio::test]
async fn clones_share_the_queue_and_the_final_result() {
    let stream: EventStream<u32, u32> = EventStream::new(|event| *event == 2, |event| Some(*event));
    let reader = stream.clone();

    stream.push(1);
    assert_eq![reader.next().await, Some(1)];

    let result_reader = stream.clone();
    stream.push(2);
    assert_eq![reader.next().await, Some(2)];
    assert_eq![result_reader.result().await, 2];
}

#[test]
fn the_debug_form_names_the_stream() {
    let stream: EventStream<u32, u32> = EventStream::new(|_| false, |event| Some(*event));
    let debug = format!("{stream:?}");
    assert![debug.starts_with("EventStream"), "{debug}"];

    let assistant = create_assistant_message_event_stream();
    assert![format!("{assistant:?}").starts_with("EventStream")];
}
