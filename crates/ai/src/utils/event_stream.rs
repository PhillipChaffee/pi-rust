//! The generic event stream, ported from
//! `packages/ai/src/utils/event-stream.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (pi #9055 ordering semantics).
//!
//! A single-producer, many-consumer broadcast queue: `push` delivers each
//! event to the oldest waiting consumer or buffers it; iteration drains
//! buffered events in order; a completed stream drops later pushes silently;
//! `end` wakes every waiting consumer. The final result resolves from the
//! completing event or an explicit `end` result and is cloneable to any
//! number of waiters.
//!
//! Porting restatements: upstream builds the stream on hand-rolled
//! async-iterable waiter promises; here waiters are tokio oneshots. Multiple
//! Rust iterators share one consumer queue exactly like the TypeScript
//! generators do. The final-result waiters require `R: Clone` so late
//! `result()` callers observe the settled value like repeated `await`s of
//! the same promise do.

use std::sync::Mutex;

use tokio::sync::oneshot;

use crate::types::{AssistantMessage, AssistantMessageEvent};

/// What a waiting consumer wakes to: an event, or the end of the stream.
enum Waiter<T> {
    Item(T),
    Done,
}

type IsCompleteCallback<T> = Box<dyn Fn(&T) -> bool + Send + Sync>;
type ExtractResultCallback<T, R> = Box<dyn Fn(&T) -> Option<R> + Send + Sync>;

struct State<T, R> {
    queue: std::collections::VecDeque<T>,
    waiting: std::collections::VecDeque<oneshot::Sender<Waiter<T>>>,
    done: bool,
    result: Option<R>,
    result_waiters: Vec<oneshot::Sender<R>>,
    is_complete: IsCompleteCallback<T>,
    extract_result: ExtractResultCallback<T, R>,
}

/// A generic event stream: push events in, iterate them out, and await the
/// final result, upstream's `EventStream<T, R = T>`. Clones share the same
/// queue, waiters, and result — one stream, many handles.
pub struct EventStream<T, R> {
    state: std::sync::Arc<Mutex<State<T, R>>>,
}

impl<T, R> std::fmt::Debug for EventStream<T, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventStream").finish_non_exhaustive()
    }
}

impl<T, R: Clone> EventStream<T, R> {
    /// Create a stream. `is_complete` marks the event that settles the final
    /// result; `extract_result` pulls that result from the completing event
    /// and returns `None` only for events `is_complete` never selects.
    #[must_use]
    pub fn new(
        is_complete: impl Fn(&T) -> bool + Send + Sync + 'static,
        extract_result: impl Fn(&T) -> Option<R> + Send + Sync + 'static,
    ) -> Self {
        Self {
            state: std::sync::Arc::new(Mutex::new(State {
                queue: std::collections::VecDeque::new(),
                waiting: std::collections::VecDeque::new(),
                done: false,
                result: None,
                result_waiters: Vec::new(),
                is_complete: Box::new(is_complete),
                extract_result: Box::new(extract_result),
            })),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State<T, R>> {
        // Poison recovery is the Rust-native passivity statement: owned
        // plain data cannot throw on read, so a poisoned lock recovers and
        // keeps the stream usable.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Push an event. After the stream completes, later pushes are silently
    /// dropped. A completing event still reaches consumers: the final result
    /// settles first, then the event is delivered or queued like any other.
    pub fn push(&self, event: T) {
        let mut state = self.lock();
        if state.done {
            return;
        }
        let is_complete = (state.is_complete)(&event);
        let settled = if is_complete {
            (state.extract_result)(&event)
        } else {
            None
        };
        if is_complete {
            // The final result settles before the completing event reaches a
            // consumer, upstream's order.
            state.done = true;
            state.result.clone_from(&settled);
            for waiter in std::mem::take(&mut state.result_waiters) {
                if let Some(result) = &settled {
                    let _ = waiter.send(result.clone());
                }
            }
        }
        match state.waiting.pop_front() {
            Some(waiter) => {
                let _ = waiter.send(Waiter::Item(event));
            }
            None => state.queue.push_back(event),
        }
    }

    /// End the stream. An explicit result settles the final result for every
    /// waiter; without one the final result stays unset, exactly like the
    /// upstream promise left pending.
    pub fn end(&self, result: Option<&R>) {
        let mut state = self.lock();
        state.done = true;
        state.result = result.cloned();
        for waiter in std::mem::take(&mut state.result_waiters) {
            if let Some(result) = result {
                let _ = waiter.send(result.clone());
            }
        }
        while let Some(waiter) = state.waiting.pop_front() {
            let _ = waiter.send(Waiter::Done);
        }
    }

    /// Take the next event, buffering draining in order. [`None`] ends the
    /// iteration.
    ///
    /// Consumers register in call order and receive events in push order,
    /// so two concurrent consumers split the stream round-robin.
    pub async fn next(&self) -> Option<T> {
        let receiver = {
            let mut state = self.lock();
            if let Some(event) = state.queue.pop_front() {
                return Some(event);
            }
            if state.done {
                return None;
            }
            let (sender, receiver) = oneshot::channel();
            state.waiting.push_back(sender);
            receiver
        };
        match receiver.await {
            Ok(Waiter::Item(event)) => Some(event),
            Ok(Waiter::Done) | Err(_) => None,
        }
    }

    /// Await the final result, upstream's `result()` promise. Resolves when
    /// the completing event or `end` supplies a result; callers registered
    /// after settlement receive the stored value.
    pub async fn result(&self) -> R
    where
        R: Clone,
    {
        let receiver = {
            let mut state = self.lock();
            if let Some(result) = &state.result {
                return result.clone();
            }
            let (sender, receiver) = oneshot::channel();
            state.result_waiters.push(sender);
            receiver
        };
        receiver.await.unwrap_or_else(|_| Self::settlement_lost())
    }

    /// The failure a result waiter hits when every stream handle is gone:
    /// the sender only closes when the stream is dropped, so a live waiter
    /// implies a live stream and this path is a caller bug.
    #[expect(
        clippy::panic,
        reason = "awaiting result() after dropping the stream is a caller bug, not a runtime condition"
    )]
    fn settlement_lost() -> ! {
        panic!("event stream result channel closed without settlement")
    }
}

impl<T, R> Clone for EventStream<T, R> {
    fn clone(&self) -> Self {
        Self {
            state: std::sync::Arc::clone(&self.state),
        }
    }
}

/// One event of the assistant-message stream protocol paired with its
/// settlement: the stream resolves to the final [`AssistantMessage`] on
/// `done` and to the failing message on `error`.
pub type AssistantMessageEventStream = EventStream<AssistantMessageEvent, AssistantMessage>;

/// Create an [`AssistantMessageEventStream`], the factory upstream exposes
/// for extensions.
#[must_use]
pub fn create_assistant_message_event_stream() -> AssistantMessageEventStream {
    assistant_message_event_stream()
}

/// An [`EventStream`] over [`AssistantMessageEvent`] resolving to the final
/// [`AssistantMessage`]: the `done` event carries it, and the `error` event
/// carries the failing message.
#[must_use]
pub fn assistant_message_event_stream() -> AssistantMessageEventStream {
    EventStream::new(
        |event| {
            matches!(
                event,
                AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
            )
        },
        |event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
            AssistantMessageEvent::Error { error, .. } => Some(error.clone()),
            // Only `done` and `error` complete the stream; the other event
            // types never reach the extractor.
            _ => None,
        },
    )
}
