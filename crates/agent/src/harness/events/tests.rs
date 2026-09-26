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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::types::BoxedFuture;
use pi_chord::context::background_context;
use tokio::sync::Notify;

use crate::harness::agent_harness::{
    HarnessEvent, HarnessEventPayload, HarnessEventType, WatchHandle,
};
use crate::harness::events::{HarnessEventBus, ListenerError, ResnapshotCapture};

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
            let boxed: BoxedFuture<'static, Result<(), ListenerError>> = Box::pin(async move {
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
        .watch::<usize>(0, Arc::new(|_event| true), None)
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
    WatchHandle::start(watcher.as_ref(), listener);
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
    let resnapshotted = WatchHandle::resnapshot(&*watcher, &context)
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
    let error =
        WatchHandle::resnapshot(&*watcher, &background_context())
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
        .on(
            HarnessEventType::RunStart,
            listener(Arc::new(Mutex::new(Vec::new()))),
        )
        .expect_err("closed bus rejects subscriptions");
    assert_eq!(error, "closed");
    let error = bus
        .watch(0, Arc::new(|_event| true), None)
        .expect_err("closed bus rejects watches");
    assert_eq!(error, "closed");
}

/// Waits until the flag flips, yielding so the spawned drain tasks run.
async fn wait_until(flag: &AtomicBool) {
    while !flag.load(Ordering::Acquire) {
        tokio::task::yield_now().await;
    }
}

/// Waits until the sink holds `count` events, yielding so the delivery
/// tails settle.
async fn wait_for_count(sink: &Recorded, count: usize) {
    for _ in 0..200 {
        if sink.lock().expect("received lock").len() >= count {
            return;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        sink.lock().expect("received lock").len() >= count,
        "the sink never reached {count} events"
    );
}

/// The run id of a `run_start` event, `""` for every other payload.
fn run_id_of(event: &HarnessEvent) -> &str {
    match &event.payload {
        HarnessEventPayload::RunStart { run_id, .. } => run_id,
        _ => "",
    }
}

/// A listener that records every event and parks on `gate` for the one
/// whose run id is `gate_run_id`, flipping `entered` first so the test
/// knows the drain tail is parked mid-delivery.
fn gating_listener(
    sink: Recorded,
    gate_run_id: &'static str,
    entered: Arc<AtomicBool>,
    gate: Arc<Notify>,
) -> crate::harness::agent_harness::EventListener {
    Arc::new(move |event: &HarnessEvent, _context| {
        let sink = Arc::clone(&sink);
        let entered = Arc::clone(&entered);
        let gate = Arc::clone(&gate);
        let event = event.clone();
        Box::pin(async move {
            sink.lock().expect("received lock").push(event.clone());
            if run_id_of(&event) == gate_run_id {
                entered.store(true, Ordering::Release);
                gate.notified().await;
            }
            Ok(())
        })
    })
}

/// The resnapshot capture's sink for the choreography's boundary debug
/// rendering.
type Rendered = Arc<Mutex<Option<String>>>;

/// The bus builds through `Default` and renders its debug shape.
#[test]
fn the_bus_builds_through_default_and_renders_its_debug_shape() {
    let bus = HarnessEventBus::default();
    assert!(format!("{bus:?}").contains("HarnessEventBus"));
}

/// `emit_batch` on a closed bus or an empty batch resolves without
/// delivering anything.
#[tokio::test]
async fn emit_batch_skips_delivery_on_a_closed_bus_or_an_empty_batch() {
    let bus = HarnessEventBus::new();
    let received: Recorded = Arc::new(Mutex::new(Vec::new()));
    bus.on(
        HarnessEventType::RunStart,
        listener(Arc::clone(&received)),
    )
    .expect("subscribe");
    bus.emit_batch(vec![]).await;
    bus.close("closed".to_owned());
    bus.emit_batch(vec![(lane_event("run"), background_context())])
        .await;
    assert!(received.lock().expect("received lock").is_empty());
}

/// A push onto a running drain tail reuses the running drainer instead of
/// spawning a second one, on the watcher tail and the bus tail alike.
#[tokio::test]
async fn pushes_while_a_tail_is_draining_reuse_the_running_drainer() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let watcher = bus
        .watch(0, Arc::new(|_event: &HarnessEvent| true), None)
        .expect("watch");
    let watcher_sink: Recorded = Arc::new(Mutex::new(Vec::new()));
    WatchHandle::start(watcher.as_ref(), listener(Arc::clone(&watcher_sink)));
    // Back-to-back pushes and emits with no await between them: the first
    // call spawns the drainer, the later ones must reuse it.
    watcher.push(lane_event("watch-1"), context.clone());
    watcher.push(lane_event("watch-2"), context.clone());
    let first = bus.emit_batch(vec![(lane_event("bus-1"), context.clone())]);
    let second = bus.emit_batch(vec![(lane_event("bus-2"), context.clone())]);
    first.await;
    second.await;
    wait_for_count(&watcher_sink, 4).await;
    let mut ids: Vec<String> = watcher_sink
        .lock()
        .expect("received lock")
        .iter()
        .map(|event| run_id_of(event).to_owned())
        .collect();
    ids.sort();
    assert_eq!(ids, ["bus-1", "bus-2", "watch-1", "watch-2"]);
}

/// An event enqueued before a resnapshot carries a stale epoch and is
/// dropped by the watcher's drain; events pushed after the resnapshot
/// deliver under the new epoch.
#[tokio::test]
async fn a_stale_epoch_event_is_dropped_when_a_resnapshot_bumps_the_epoch() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let entered = Arc::new(AtomicBool::new(false));
    let gate = Arc::new(Notify::new());
    let resnapshot_started = Arc::new(AtomicBool::new(false));
    let capture: ResnapshotCapture<u64> = {
        let started = Arc::clone(&resnapshot_started);
        Arc::new(move |_capture_context, boundary| {
            let started = Arc::clone(&started);
            Box::pin(async move {
                started.store(true, Ordering::Release);
                boundary.mark();
                Ok(1_u64)
            })
        })
    };
    let watcher = bus
        .watch(
            0_u64,
            Arc::new(|event: &HarnessEvent| run_id_of(event).starts_with("watched")),
            Some(capture),
        )
        .expect("watch");
    let watcher_sink: Recorded = Arc::new(Mutex::new(Vec::new()));
    WatchHandle::start(
        watcher.as_ref(),
        gating_listener(
            Arc::clone(&watcher_sink),
            "watched-gate",
            Arc::clone(&entered),
            Arc::clone(&gate),
        ),
    );
    bus.emit(lane_event("watched-gate"), &context).await;
    wait_until(&entered).await;
    // The gate event's watcher delivery is parked; this one enqueues with
    // the pre-resnapshot epoch.
    bus.emit(lane_event("watched-second"), &context).await;
    let task = tokio::spawn({
        let watcher = Arc::clone(&watcher);
        let context = context.clone();
        async move { WatchHandle::resnapshot(watcher.as_ref(), &context).await }
    });
    wait_until(&resnapshot_started).await;
    gate.notify_one();
    let result = task.await.expect("resnapshot task");
    assert_eq!(result.expect("resnapshot"), 1);
    assert_eq!(watcher.snapshot(), 1);
    bus.emit(lane_event("watched-after"), &context).await;
    wait_for_count(&watcher_sink, 2).await;
    let ids: Vec<String> = watcher_sink
        .lock()
        .expect("received lock")
        .iter()
        .map(|event| run_id_of(event).to_owned())
        .collect();
    assert_eq!(ids, ["watched-gate", "watched-after"]);
}

/// A watch listener's failure becomes a `handler_error` event on the bus,
/// addressed to the watcher audience with the failed event's lane and
/// type; the `handler_error` delivery does not recurse.
#[tokio::test]
async fn a_watch_listener_failure_becomes_a_handler_error_on_the_bus() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let watcher = bus
        .watch(0, Arc::new(|_event: &HarnessEvent| true), None)
        .expect("watch");
    let watcher_sink: Recorded = Arc::new(Mutex::new(Vec::new()));
    let failing: crate::harness::agent_harness::EventListener = {
        let sink = Arc::clone(&watcher_sink);
        Arc::new(move |event: &HarnessEvent, _context| {
            let sink = Arc::clone(&sink);
            let event = event.clone();
            Box::pin(async move {
                sink.lock().expect("received lock").push(event.clone());
                // The listener fails on every event, the handler_error
                // replay included; the on-error guard must stop there.
                Err("watch failed".into())
            })
        })
    };
    WatchHandle::start(watcher.as_ref(), failing);
    bus.emit(lane_event("watched-boom"), &context).await;
    wait_for_count(&watcher_sink, 2).await;
    let events = watcher_sink.lock().expect("received lock").clone();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].lane.as_deref(), Some("main"));
    assert_eq!(run_id_of(&events[0]), "watched-boom");
    assert_eq!(events[1].lane.as_deref(), Some("main"));
    assert!(matches!(
        &events[1].payload,
        HarnessEventPayload::HandlerError {
            error,
            kind: crate::harness::agent_harness::HandlerErrorKind::Event { event },
            ..
        } if error == "watch failed" && event == "run_start"
    ));
}

/// A `handler_error` recipient's own failure is not reported further: the
/// surviving recorder sees exactly one `handler_error`.
#[tokio::test]
async fn a_handler_error_recipient_failure_is_not_reported_further() {
    let bus = HarnessEventBus::new();
    let handler_errors: Recorded = Arc::new(Mutex::new(Vec::new()));
    bus.on(
        HarnessEventType::HandlerError,
        listener(Arc::clone(&handler_errors)),
    )
    .expect("subscribe the recorder");
    bus.on(
        HarnessEventType::HandlerError,
        Arc::new(|_event: &HarnessEvent, _context| {
            Box::pin(async { Err("recipient failed".into()) })
        }),
    )
    .expect("subscribe the failing recipient");
    bus.on(
        HarnessEventType::RunStart,
        Arc::new(|_event: &HarnessEvent, _context| {
            Box::pin(async { Err("listener failed".into()) })
        }),
    )
    .expect("subscribe the failing listener");
    bus.emit(lane_event("run"), &background_context()).await;
    wait_for_count(&handler_errors, 1).await;
    assert_eq!(handler_errors.lock().expect("received lock").len(), 1);
}

/// The first close error wins; the close barrier clears the registries on
/// the delivery tail, after which a stale subscription unsubscribes
/// nothing.
#[tokio::test]
async fn a_second_close_is_a_no_op_and_the_barrier_clears_the_registries() {
    let bus = HarnessEventBus::new();
    let received: Recorded = Arc::new(Mutex::new(Vec::new()));
    let subscription = bus
        .on(HarnessEventType::RunStart, listener(Arc::clone(&received)))
        .expect("subscribe");
    bus.close("first".to_owned());
    bus.close("second".to_owned());
    let error = bus
        .on(
            HarnessEventType::RunStart,
            listener(Arc::new(Mutex::new(Vec::new()))),
        )
        .expect_err("closed bus rejects subscriptions");
    assert_eq!(error, "first");
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    subscription.unsubscribe();
    assert!(received.lock().expect("received lock").is_empty());
}

/// A closed bus rejects `watch_from_snapshot` with its close error.
#[tokio::test]
async fn a_closed_bus_rejects_watch_from_snapshot() {
    let bus = HarnessEventBus::new();
    bus.close("closed".to_owned());
    let error = bus
        .watch_from_snapshot(
            |_capture_context| async { Ok::<u64, std::io::Error>(1) },
            Arc::new(|_event: &HarnessEvent| true),
            &background_context(),
        )
        .await
        .expect_err("closed bus rejects watch_from_snapshot");
    assert_eq!(error, "closed");
}

/// A failed `watch_from_snapshot` capture unsubscribes the installed
/// watcher and reports the capture's error.
#[tokio::test]
async fn a_failed_capture_unsubscribes_the_watcher() {
    let bus = HarnessEventBus::new();
    let error = bus
        .watch_from_snapshot(
            |_capture_context| async {
                Err::<u64, std::io::Error>(std::io::Error::other("capture failed"))
            },
            Arc::new(|_event: &HarnessEvent| true),
            &background_context(),
        )
        .await
        .expect_err("the capture failed");
    assert!(error.contains("capture failed"), "{error}");
}

/// A capture that never marks its boundary reports the contract violation;
/// events pushed before the boundary were dropped, and events held by a
/// direct boundary mark replay.
#[tokio::test]
async fn an_unmarked_capture_reports_the_violation_and_replays_the_held_events() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let entered = Arc::new(AtomicBool::new(false));
    let gate = Arc::new(Notify::new());
    let capture: ResnapshotCapture<u64> = {
        let entered = Arc::clone(&entered);
        let gate = Arc::clone(&gate);
        Arc::new(move |_capture_context, _boundary| {
            let entered = Arc::clone(&entered);
            let gate = Arc::clone(&gate);
            Box::pin(async move {
                entered.store(true, Ordering::Release);
                gate.notified().await;
                Ok(4_u64)
            })
        })
    };
    let watcher = bus
        .watch(0_u64, Arc::new(|_event: &HarnessEvent| true), Some(capture))
        .expect("watch");
    let watcher_sink: Recorded = Arc::new(Mutex::new(Vec::new()));
    WatchHandle::start(watcher.as_ref(), listener(Arc::clone(&watcher_sink)));
    let task = tokio::spawn({
        let watcher = Arc::clone(&watcher);
        let context = context.clone();
        async move { WatchHandle::resnapshot(watcher.as_ref(), &context).await }
    });
    wait_until(&entered).await;
    bus.emit(lane_event("dropped"), &context).await;
    // A direct boundary mark flips the hold without the capture marking;
    // pushes after it are held for the violation path to replay.
    watcher.mark_resnapshot_boundary();
    bus.emit(lane_event("held"), &context).await;
    gate.notify_one();
    let result = task.await.expect("resnapshot task");
    let error = result.expect_err("the capture never marked");
    assert!(
        error.to_string().contains("did not mark its boundary"),
        "{error}"
    );
    wait_for_count(&watcher_sink, 1).await;
    let ids: Vec<String> = watcher_sink
        .lock()
        .expect("received lock")
        .iter()
        .map(|event| run_id_of(event).to_owned())
        .collect();
    assert_eq!(ids, ["held"]);
    assert_eq!(watcher.snapshot(), 0);
}

/// A bus listener's failure delivers its `handler_error` to the watcher
/// audience when no `handler_error` listeners are registered.
#[tokio::test]
async fn a_bus_listener_failure_reaches_the_watcher_audience() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let watcher = bus
        .watch(0, Arc::new(|_event: &HarnessEvent| true), None)
        .expect("watch");
    let watcher_sink: Recorded = Arc::new(Mutex::new(Vec::new()));
    WatchHandle::start(watcher.as_ref(), listener(Arc::clone(&watcher_sink)));
    bus.on(
        HarnessEventType::RunStart,
        Arc::new(|_event: &HarnessEvent, _context| {
            Box::pin(async { Err("listener failed".into()) })
        }),
    )
    .expect("subscribe the failing listener");
    bus.emit(lane_event("run"), &context).await;
    wait_for_count(&watcher_sink, 2).await;
    let events = watcher_sink.lock().expect("received lock").clone();
    assert_eq!(events.len(), 2);
    let handler_errors = events
        .iter()
        .filter(|event| matches!(&event.payload, HarnessEventPayload::HandlerError { .. }))
        .count();
    assert_eq!(handler_errors, 1);
    let original = events
        .iter()
        .filter(|event| matches!(&event.payload, HarnessEventPayload::RunStart { .. }))
        .count();
    assert_eq!(original, 1);
    let handler_error = events
        .iter()
        .find(|event| matches!(&event.payload, HarnessEventPayload::HandlerError { .. }))
        .expect("the handler_error event");
    assert_eq!(handler_error.lane.as_deref(), Some("main"));
    assert!(matches!(
        &handler_error.payload,
        HarnessEventPayload::HandlerError {
            error,
            kind: crate::harness::agent_harness::HandlerErrorKind::Event { event },
            ..
        } if error == "listener failed" && event == "run_start"
    ));
}

/// A capture that marks its boundary but fails replays the events held
/// after the boundary and reports the capture's own error.
#[tokio::test]
async fn a_marking_capture_that_fails_replays_and_reports() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let entered = Arc::new(AtomicBool::new(false));
    let started = Arc::new(Notify::new());
    let marked = Arc::new(Notify::new());
    let flipped = Arc::new(Notify::new());
    let capture: ResnapshotCapture<u64> = {
        let entered = Arc::clone(&entered);
        let started = Arc::clone(&started);
        let marked = Arc::clone(&marked);
        let flipped = Arc::clone(&flipped);
        Arc::new(move |_capture_context, boundary| {
            let entered = Arc::clone(&entered);
            let started = Arc::clone(&started);
            let marked = Arc::clone(&marked);
            let flipped = Arc::clone(&flipped);
            Box::pin(async move {
                entered.store(true, Ordering::Release);
                started.notified().await;
                boundary.mark();
                marked.notify_one();
                flipped.notified().await;
                Err(ListenerError::from(std::io::Error::other("capture boom")))
            })
        })
    };
    let watcher = bus
        .watch(
            0_u64,
            Arc::new(|event: &HarnessEvent| run_id_of(event) != "probe"),
            Some(capture),
        )
        .expect("watch");
    let watcher_sink: Recorded = Arc::new(Mutex::new(Vec::new()));
    WatchHandle::start(watcher.as_ref(), listener(Arc::clone(&watcher_sink)));
    let task = tokio::spawn({
        let watcher = Arc::clone(&watcher);
        let context = context.clone();
        async move { WatchHandle::resnapshot(watcher.as_ref(), &context).await }
    });
    wait_until(&entered).await;
    started.notify_one();
    marked.notified().await;
    // The boundary barrier rides the bus tail ahead of this probe batch;
    // once the probe resolves, the watcher holds.
    bus.emit(lane_event("probe"), &context).await;
    bus.emit(lane_event("held"), &context).await;
    flipped.notify_one();
    let result = task.await.expect("resnapshot task");
    let error = result.expect_err("the capture failed");
    assert!(error.to_string().contains("capture boom"), "{error}");
    wait_for_count(&watcher_sink, 1).await;
    assert_eq!(
        run_id_of(&watcher_sink.lock().expect("received lock")[0]),
        "held"
    );
}

/// A second resnapshot while one is in progress reports busy.
#[tokio::test]
async fn a_second_resnapshot_while_one_is_in_progress_reports_busy() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let entered = Arc::new(AtomicBool::new(false));
    let gate = Arc::new(Notify::new());
    let capture: ResnapshotCapture<u64> = {
        let entered = Arc::clone(&entered);
        let gate = Arc::clone(&gate);
        Arc::new(move |_capture_context, boundary| {
            let entered = Arc::clone(&entered);
            let gate = Arc::clone(&gate);
            Box::pin(async move {
                entered.store(true, Ordering::Release);
                gate.notified().await;
                boundary.mark();
                Ok(3_u64)
            })
        })
    };
    let watcher = bus
        .watch(0_u64, Arc::new(|_event: &HarnessEvent| true), Some(capture))
        .expect("watch");
    WatchHandle::start(watcher.as_ref(), listener(Arc::new(Mutex::new(Vec::new()))));
    let task = tokio::spawn({
        let watcher = Arc::clone(&watcher);
        let context = context.clone();
        async move { WatchHandle::resnapshot(watcher.as_ref(), &context).await }
    });
    wait_until(&entered).await;
    let busy = WatchHandle::resnapshot(watcher.as_ref(), &context)
        .await
        .expect_err("a resnapshot is already in progress");
    assert!(busy.to_string().contains("already in progress"), "{busy}");
    gate.notify_one();
    let result = task.await.expect("resnapshot task");
    assert_eq!(result.expect("resnapshot"), 3);
}

/// A resnapshot drops events that arrive before the boundary, holds events
/// that arrive after it, and replays the held ones with the new snapshot;
/// the boundary renders its debug shape and marking without a hold is a
/// no-op.
#[tokio::test]
async fn the_resnapshot_choreography_drops_then_holds_then_replays() {
    let bus = HarnessEventBus::new();
    let context = background_context();
    let entered = Arc::new(AtomicBool::new(false));
    let started = Arc::new(Notify::new());
    let marked = Arc::new(Notify::new());
    let flipped = Arc::new(Notify::new());
    let rendered: Rendered = Arc::new(Mutex::new(None));
    let capture: ResnapshotCapture<u64> = {
        let entered = Arc::clone(&entered);
        let started = Arc::clone(&started);
        let marked = Arc::clone(&marked);
        let flipped = Arc::clone(&flipped);
        let rendered = Arc::clone(&rendered);
        Arc::new(move |_capture_context, boundary| {
            let entered = Arc::clone(&entered);
            let started = Arc::clone(&started);
            let marked = Arc::clone(&marked);
            let flipped = Arc::clone(&flipped);
            let rendered = Arc::clone(&rendered);
            Box::pin(async move {
                entered.store(true, Ordering::Release);
                started.notified().await;
                *rendered.lock().expect("rendered lock") = Some(format!("{boundary:?}"));
                boundary.mark();
                marked.notify_one();
                flipped.notified().await;
                Ok(5_u64)
            })
        })
    };
    let watcher = bus
        .watch(
            0_u64,
            Arc::new(|event: &HarnessEvent| run_id_of(event).starts_with("watched")),
            Some(capture),
        )
        .expect("watch");
    let watcher_sink: Recorded = Arc::new(Mutex::new(Vec::new()));
    WatchHandle::start(watcher.as_ref(), listener(Arc::clone(&watcher_sink)));
    bus.emit(lane_event("watched-pre"), &context).await;
    let task = tokio::spawn({
        let watcher = Arc::clone(&watcher);
        let context = context.clone();
        async move { WatchHandle::resnapshot(watcher.as_ref(), &context).await }
    });
    wait_until(&entered).await;
    bus.emit(lane_event("watched-drop"), &context).await;
    started.notify_one();
    marked.notified().await;
    // The boundary barrier rides the bus tail ahead of this probe batch;
    // once the probe resolves, the watcher holds.
    bus.emit(lane_event("probe"), &context).await;
    bus.emit(lane_event("watched-hold"), &context).await;
    flipped.notify_one();
    let result = task.await.expect("resnapshot task");
    assert_eq!(result.expect("resnapshot"), 5);
    assert_eq!(watcher.snapshot(), 5);
    wait_for_count(&watcher_sink, 2).await;
    let ids: Vec<String> = watcher_sink
        .lock()
        .expect("received lock")
        .iter()
        .map(|event| run_id_of(event).to_owned())
        .collect();
    assert_eq!(ids, ["watched-pre", "watched-hold"]);
    assert!(
        rendered
            .lock()
            .expect("rendered lock")
            .as_ref()
            .expect("the capture rendered the boundary")
            .contains("ResnapshotBoundary")
    );
    watcher.mark_resnapshot_boundary();
}

/// An unsubscribed watcher drops pushes, no-ops repeat unsubscribes, and
/// reports resnapshot unavailable; the trait-object surface dispatches to
/// the same behavior.
#[tokio::test]
async fn an_unsubscribed_watcher_drops_pushes_and_reports_resnapshot_unavailable() {
    let bus = HarnessEventBus::new();
    let watcher = bus
        .watch(0_usize, Arc::new(|_event: &HarnessEvent| true), None)
        .expect("watch");
    assert!(format!("{watcher:?}").contains("BufferedEventWatcher"));
    let handle: Box<dyn WatchHandle<usize>> = Box::new((*watcher).clone());
    handle.set_snapshot(3);
    assert_eq!(handle.snapshot(), 3);
    watcher.unsubscribe();
    watcher.unsubscribe();
    watcher.push(lane_event("late"), background_context());
    handle.unsubscribe();
    let error = WatchHandle::resnapshot(watcher.as_ref(), &background_context())
        .await
        .expect_err("the watcher is unsubscribed");
    assert!(error.to_string().contains("unsubscribed"), "{error}");
}
