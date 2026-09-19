//! Tests for the context port, mirroring upstream `test/context.test.ts`.

use super::*;
use crate::test_support::some;
use std::future::poll_fn;
use std::task::Poll;

#[test]
fn provides_distinct_empty_root_contexts() {
    let key: ContextKey<String> = create_context_key("value");

    // The roots are distinct values; the port mints a fresh chain per
    // constructor call, so identity never aliases.
    assert_ne!(
        format!("{}", todo_context()),
        format!("{}", background_context())
    );
    assert!(todo_context().abort_signal().is_none());
    assert!(todo_context().value(&key).is_none());
    assert_eq!(
        format!("{}", background_context()),
        "[Context BACKGROUND_CONTEXT]"
    );
    assert_eq!(format!("{}", todo_context()), "[Context TODO_CONTEXT]");
}

#[test]
fn layers_typed_values_without_modifying_parents() {
    let background = background_context();
    let first_key: ContextKey<String> = create_context_key("first");
    let second_key: ContextKey<f64> = create_context_key("second");
    let first = with_context_value(&first_key, "one".to_string(), &background);
    let second = with_context_value(&second_key, 2.0, &first);
    let replaced = with_context_value(&first_key, "updated".to_string(), &second);

    assert!(background.value(&first_key).is_none());
    assert_eq!(first.value(&first_key), Some(&"one".to_string()));
    assert!(first.value(&second_key).is_none());
    assert_eq!(second.value(&first_key), Some(&"one".to_string()));
    assert_eq!(second.value(&second_key), Some(&2.0));
    assert_eq!(replaced.value(&first_key), Some(&"updated".to_string()));
    assert_eq!(second.value(&first_key), Some(&"one".to_string()));
    assert_eq!(
        format!("{replaced}"),
        "[Context BACKGROUND_CONTEXT].WithValue(first).WithValue(second).WithValue(first)"
    );
}

/// Polls a wait future once, the listener-fixture analogue.
fn poll_wait(wait: &mut WaitAborted) -> Poll<AbortReason> {
    let waker = Waker::noop();
    let mut cx = TaskContext::from_waker(waker);
    std::pin::pin!(wait).poll(&mut cx)
}

#[test]
fn inherits_parent_cancellation_and_isolates_child_cancellation() {
    let parent_controller = AbortController {
        signal: AbortSignal::own(),
    };
    let parent = with_abort_signal(parent_controller.signal().clone(), &background_context());
    let (child, child_cancel) = with_cancel(&parent);
    let (sibling, _sibling_cancel) = with_cancel(&parent);

    // The listener fixture becomes the waiter future itself.
    let mut child_wait = some(child.abort_signal()).wait();
    let mut sibling_wait = some(sibling.abort_signal()).wait();

    child_cancel.abort("child");
    let child_signal = some(child.abort_signal());
    assert!(child_signal.aborted());
    assert_eq!(
        child_signal.reason(),
        Some(AbortReason::Caller("child".to_string()))
    );
    assert!(!some(sibling.abort_signal()).aborted());
    assert!(!some(parent.abort_signal()).aborted());
    // The waiter resolves exactly once, on the first abort.
    assert_eq!(
        poll_wait(&mut child_wait),
        Poll::Ready(AbortReason::Caller("child".to_string()))
    );

    parent_controller.abort("parent");
    assert!(some(sibling.abort_signal()).aborted());
    assert_eq!(
        some(sibling.abort_signal()).reason(),
        Some(AbortReason::Caller("parent".to_string()))
    );
    // The sibling waiter was pending through the child's cancellation and
    // resolves with the parent's reason.
    assert_eq!(
        poll_wait(&mut sibling_wait),
        Poll::Ready(AbortReason::Caller("parent".to_string()))
    );
}

#[test]
fn masks_caller_cancellation_for_mandatory_cleanup() {
    let controller = AbortController {
        signal: AbortSignal::own(),
    };
    let key: ContextKey<String> = create_context_key("value");
    let context = with_context_value(
        &key,
        "preserved".to_string(),
        &with_abort_signal(controller.signal().clone(), &background_context()),
    );
    let cleanup = without_abort_signal(&context);

    controller.abort_without_reason();
    assert!(some(context.abort_signal()).aborted());
    assert!(cleanup.abort_signal().is_none());
    assert_eq!(cleanup.value(&key), Some(&"preserved".to_string()));
}

#[test]
fn stops_waiting_when_the_invocation_is_cancelled() {
    let controller = AbortController {
        signal: AbortSignal::own(),
    };
    let context = with_abort_signal(controller.signal().clone(), &background_context());
    // The work runs on its own shared state; cancellation drops only the
    // waiter, not the underlying work.
    let completed = Arc::new(Mutex::new(None::<String>));
    let work = {
        let completed = Arc::clone(&completed);
        poll_fn(move |_cx| {
            let Some(mut state) = completed.lock().ok() else {
                return Poll::Pending;
            };
            if let Some(value) = state.clone() {
                return Poll::Ready(value);
            }
            *state = Some("completed later".to_string());
            Poll::Pending
        })
    };
    let waiting = await_with_context(work, &context);
    let waker = Waker::noop();
    let mut cx = TaskContext::from_waker(waker);

    controller.abort("cancelled");
    let polled = std::pin::pin!(waiting).poll(&mut cx);
    let reason = match polled {
        Poll::Ready(Err(reason)) => Some(reason),
        _ => None,
    };
    assert_eq!(
        reason,
        Some(AbortReason::Caller("cancelled".to_string())),
        "cancellation resolves the waiter with the reason"
    );
    // The underlying work's own driver ran to completion independently.
    let stored = some(completed.lock().ok()).clone();
    assert_eq!(stored, Some("completed later".to_string()));
    // Without a signal, awaiting is a plain await.
    let settled = await_with_context(
        std::future::ready("completed".to_string()),
        &background_context(),
    );
    let output = match std::pin::pin!(settled).poll(&mut cx) {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    };
    assert_eq!(output, Some(Ok("completed".to_string())));
}
