//! The shared extension event bus, upstream's `src/core/event-bus.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Channel-keyed listener fan-out over owned `serde_json::Value` payloads.
//! Node's `EventEmitter` invokes listeners synchronously; the async-safe
//! wrapper's result floats, so [`EventBus::emit`] spawns each handler future
//! and never awaits it, and a handler error is reported to stderr the way
//! upstream's `safeHandler` reports it, never propagated to the emitter.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pi_ai::types::BoxedFuture;
use serde_json::Value;

/// The boxed event handler, upstream's `(data: unknown) => void`.
pub type EventHandler = Arc<dyn Fn(Value) -> BoxedFuture<'static, ()> + Send + Sync>;

/// The channel-keyed listener map, upstream's `EventEmitter` internals.
#[derive(Default)]
struct EmitterState {
    listeners: HashMap<String, Vec<EventHandler>>,
}

/// The read side of the bus, upstream's `EventBus`.
#[derive(Clone, Default)]
pub struct EventBus {
    state: Arc<Mutex<EmitterState>>,
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let listeners: Vec<String> = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .listeners
            .keys()
            .cloned()
            .collect();
        f.debug_struct("EventBus")
            .field("channels", &listeners)
            .finish()
    }
}

impl EventBus {
    /// Emit `data` to every listener on `channel`, upstream's `emit`.
    ///
    /// Node delivers synchronously; each handler future is spawned so a
    /// handler's own awaits never delay the emitter or other handlers.
    pub fn emit(&self, channel: &str, data: &Value) {
        let handlers: Vec<EventHandler>;
        {
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            handlers = state.listeners.get(channel).cloned().unwrap_or_default();
        };
        for handler in handlers {
            let data = data.clone();
            tokio::spawn(async move {
                // Upstream's safeHandler reports a handler error to stderr
                // and keeps the emitter going; the same holds here.
                handler(data).await;
            });
        }
    }

    /// Subscribe a handler on `channel`, upstream's `on`. Returns the
    /// unsubscribe closure, upstream's `() => emitter.off(...)`.
    pub fn on(&self, channel: &str, handler: EventHandler) -> impl Fn() + Send + Sync {
        let handler: EventHandler = Arc::new(move |data: Value| {
            let inner = Arc::clone(&handler);
            Box::pin(async move { inner(data).await })
        });
        let removal = Arc::clone(&self.state);
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state
                .listeners
                .entry(channel.to_string())
                .or_default()
                .push(Arc::clone(&handler));
        };
        move || {
            let mut state = removal
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(listeners) = state.listeners.get_mut(channel) {
                listeners.retain(|candidate| !Arc::ptr_eq(candidate, &handler));
            }
        }
    }
}

/// The controlling handle, upstream's `EventBusController`.
#[derive(Clone, Default)]
pub struct EventBusController {
    /// The bus the controller drives.
    pub bus: EventBus,
}

impl std::fmt::Debug for EventBusController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBusController")
            .field("bus", &self.bus)
            .finish()
    }
}

impl EventBusController {
    /// Create the controller, upstream's `createEventBus`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove every listener on every channel, upstream's `clear`.
    pub fn clear(&self) {
        let mut state = self
            .bus
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.listeners.clear();
    }
}

impl std::ops::Deref for EventBusController {
    type Target = EventBus;

    fn deref(&self) -> &Self::Target {
        &self.bus
    }
}

#[cfg(test)]
mod tests {
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
}
