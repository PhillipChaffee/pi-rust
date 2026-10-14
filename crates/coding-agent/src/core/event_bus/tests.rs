use super::*;
use serde_json::json;

#[tokio::test]
async fn emits_to_subscribers_on_their_channel() {
    let bus = EventBusController::new();
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&seen);
    let unsubscribe = bus.on(
        "greeting",
        Arc::new(move |data: Value| {
            let sink = Arc::clone(&sink);
            Box::pin(async move {
                sink.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(json!(data).to_string());
            })
        }),
    );
    bus.emit("greeting", &json!("hi"));
    bus.emit("other", &json!("ignored"));
    tokio::task::yield_now().await;
    assert_eq!(
        seen.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone(),
        vec!["\"hi\"".to_string()]
    );
    unsubscribe();
    bus.emit("greeting", &json!("again"));
    tokio::task::yield_now().await;
    assert_eq!(
        seen.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        1
    );
}

#[test]
fn clear_drops_every_listener() {
    let bus = EventBusController::new();
    let noop: EventHandler = Arc::new(|_| Box::pin(async {}));
    let _ = bus.on("a", noop);
    bus.clear();
    assert!(
        bus.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .listeners
            .is_empty()
    );
}

#[test]
fn bus_and_controller_debug_the_channel_names() {
    let bus = EventBusController::new();
    let noop: EventHandler = Arc::new(|_| Box::pin(async {}));
    let _ = bus.on("a", noop);
    assert!(format!("{:?}", bus.bus).contains('a'));
    assert!(format!("{bus:?}").contains('a'));
}

#[test]
fn unsubscribe_twice_is_harmless() {
    let bus = EventBusController::new();
    let noop: EventHandler = Arc::new(|_| Box::pin(async {}));
    let unsubscribe = bus.on("a", noop);
    unsubscribe();
    // The channel is gone; the second call finds no listeners.
    unsubscribe();
}
