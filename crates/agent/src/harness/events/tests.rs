//! Boundary tests for the harness event bus and watcher.
//!
//! Upstream has no unit file for `events.ts` — the bus and the buffered
//! watcher ride the runtime suites, which port with the runtime child.
//! These tests pin the boundary semantics this slice restates: subscription
//! delivery through the ordered tail, handler-failure isolation into
//! `handler_error` events, the watcher's buffer-then-start contract, and
//! the resnapshot barrier's drop-then-hold replay.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
#![expect(clippy::unwrap_used, reason = "tests unwrap the pinned outcomes")]

use std::sync::{Arc, Mutex};

use pi_ai::types::BoxedFuture;
use pi_chord::context::background_context;

use crate::harness::agent_harness::{HarnessEvent, HarnessEventPayload, HarnessEventType};
use crate::harness::events::{HarnessEventBus, ListenerError};

fn lane_event(run_id: &str) -> HarnessEvent {
    HarnessEvent::lane_scoped(
        "main",
        false,
        HarnessEventPayload::RunStart {
            run_id: run_id.to_owned(),
            started_at: 1,
        },
    )
    .expect("run_start is lane-scoped")
}

type Recorded = Arc<Mutex<Vec<HarnessEvent>>>;

fn listener(sink: Recorded) -> crate::harness::agent_harness::EventListener {
    Arc::new(move |event: &HarnessEvent, _context| {
        let sink = Arc::clone(&sink);
        let event = event.clone();
        {
            let boxed: BoxedFuture<'static, Result<(), ListenerError>> =
                Box::pin(async move {
            sink.lock().expect("received lock").push(event);
            Ok(())
        });
            boxed
        }
    })
}

fn handler_errors_of(event: &HarnessEvent) -> Option<String> {
    match &event.payload {
        HarnessEventPayload::HandlerError { error, .. } => Some(error.clone()),
        _ => None,
    }
}

#[tokio::test]
async fn subscriptions_receive_delivered_events_and_unsubscribe() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let received: Recorded = Arc::new(Mutex::new(Vec::new()));
    let subscription = bus
        .on(HarnessEventType::RunStart, listener(Arc::clone(&received)))
        .expect("subscribe");
    bus.emit(lane_event("run"), &context).await;
    assert_eq!(received.lock().expect("received lock").len(), 1);
    subscription.unsubscribe();
    bus.emit(lane_event("run-2"), &context).await;
    assert_eq!(received.lock().expect("received lock").len(), 1);
}

/// A listener failure isolates into a `handler_error` event delivered to
/// the surviving listeners.
#[tokio::test]
async fn listener_failures_isolate_into_handler_errors() {
    let bus = HarnessEventBus::new();
    let handler_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let observer_sink = Arc::clone(&handler_errors);
    bus.on(
        HarnessEventType::HandlerError,
        Arc::new(move |event: &HarnessEvent, _context| {
            let sink = Arc::clone(&observer_sink);
            let owned = event.clone();
            let boxed: BoxedFuture<'static, Result<(), ListenerError>> = Box::pin(async move {
                if let Some(error) = handler_errors_of(&owned) {
                    sink.lock().expect("handler error lock").push(error);
                }
                Ok(())
            });
            boxed
        }),
    )
    .expect("subscribe the error observer");
    let failing: crate::harness::agent_harness::EventListener =
        Arc::new(|_event, _context| Box::pin(async { Err("listener failed".into()) }));
    bus.on(HarnessEventType::RunStart, failing)
        .expect("subscribe the failing listener");
    bus.emit(lane_event("run"), &background_context()).await;
    assert_eq!(
        handler_errors.lock().expect("handler error lock").clone(),
        vec!["listener failed".to_owned()]
    );
}

/// Events emitted before a watcher's `start` buffer and replay on start;
/// after `unsubscribe`, pushes drop.
#[tokio::test]
async fn watchers_buffer_until_start_and_drop_after_unsubscribe() {
    let bus = HarnessEventBus::new();
    let watcher = bus
        .watch::<usize>(
            0,
            Arc::new(|_event| true),
            None,
        )
        .expect("watch");
    bus.emit(lane_event("run"), &background_context()).await;
    let delivered: Recorded = Arc::new(Mutex::new(Vec::new()));
    {
        let sink = Arc::clone(&delivered);
        watcher_start(
            &watcher,
            Arc::new(move |event: &HarnessEvent, _context| {
                let sink = Arc::clone(&sink);
                let event = event.clone();
                {
            let boxed: BoxedFuture<'static, Result<(), ListenerError>> =
                Box::pin(async move {
                    sink.lock().expect("delivered lock").push(event);
                    Ok(())
                });
            boxed
        }
            }),
        );
    }
    // The buffered event replays on start.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(delivered.lock().expect("delivered lock").len(), 1);
    watcher.unsubscribe();
    bus.emit(lane_event("run-2"), &background_context()).await;
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(delivered.lock().expect("delivered lock").len(), 1);
    assert_eq!(watcher.snapshot(), 0);
}

fn watcher_start(
    watcher: &Arc<crate::harness::events::BufferedEventWatcher<usize>>,
    listener: crate::harness::agent_harness::EventListener,
) {
    crate::harness::agent_harness::WatchHandle::start(watcher.as_ref(), listener);
}

/// `watch_from_snapshot` installs the captured snapshot and resnapshots
/// through the capture with a marked boundary; a failed capture
/// unsubscribes.
#[tokio::test]
async fn watch_from_snapshot_captures_and_resnapshots() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let captured: Arc<Mutex<u64>> = Arc::new(Mutex::new(7));
    let capture_source = Arc::clone(&captured);
    let watcher = bus
        .watch_from_snapshot(
            move |_capture_context| {
                let value = *capture_source.lock().expect("captured lock");
                async move { Ok::<u64, std::io::Error>(value) }
            },
            Arc::new(|_event| true),
            &context,
        )
        .await
        .expect("watch from snapshot");
    assert_eq!(watcher.snapshot(), 7);
    *captured.lock().expect("captured lock") = 9;
    let resnapshotted = crate::harness::agent_harness::WatchHandle::resnapshot(&*watcher, &context)
        .await
        .expect("resnapshot");
    assert_eq!(resnapshotted, 9);
    assert_eq!(watcher.snapshot(), 9);
}

/// A capture that never marks its boundary reports the contract violation
/// and replays the held events.
#[tokio::test]
async fn an_unmarked_resnapshot_boundary_reports_the_violation() {
    let bus = HarnessEventBus::new();
    let watcher = bus
        .watch(0_u64, Arc::new(|_event| true), None)
        .expect("watch");
    // No resnapshot callback installed: the watcher reports the
    // non-resnapshotting contract.
    let error = crate::harness::agent_harness::WatchHandle::resnapshot(&*watcher, &background_context())
        .await
        .expect_err("a watcher without a capture cannot resnapshot");
    assert!(error.to_string().contains("does not support resnapshot"));
}

/// A closed bus rejects subscriptions and watches with its close error.
#[tokio::test]
async fn a_closed_bus_rejects_subscriptions_and_watches() {
    let bus = HarnessEventBus::new();
    bus.close("closed".to_owned());
    let error = bus
        .on(HarnessEventType::RunStart, listener(Arc::new(Mutex::new(Vec::new()))))
        .expect_err("closed bus rejects subscriptions");
    assert_eq!(error, "closed");
    let error = bus
        .watch(0, Arc::new(|_event| true), None)
        .expect_err("closed bus rejects watches");
    assert_eq!(error, "closed");
}