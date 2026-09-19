//! The runtime-selected dispatch handle, the concrete type downstream crates
//! hold when a telemetry context crosses a crate boundary or sits in runtime
//! data — the ambient-context-bag path ([ADR 0004]).
//!
//! TypeScript erases interfaces naturally; Rust cannot store the generic
//! [`TelemetryContext`] (object-unsafe: its generic `start_span`) in a struct
//! field, and threading generics through every downstream signature is the
//! rejected alternative. ADR 0004 therefore pairs the raw trait — which stays
//! the statically-typed path — with this open, unsealed object-safe mirror and
//! a `Clone` handle of boxed futures. Every operation crossing the handle pays
//! one vtable hop plus one future allocation, the same dynamic dispatch
//! TypeScript performs on every interface call.
//!
//! Two erasure costs are contractual, not incidental:
//!
//! - A body handed to the handle owns its data (`'static`); borrowed state
//!   stays with statically-known sites on the raw generic trait.
//! - The body's `Err` value is projected onto the [`SpanBodyError`]
//!   name/message pair — the port of upstream's `Error.name`/`Error.message`
//!   read — and recorded as an explicit span status, so erased failures keep
//!   the details the raw trait's automatic status leaves off pending the
//!   error-trait seam.
//!
//! [ADR 0004]: ../../docs/adr/0004-open-telemetry-seam.md

use std::error::Error as StdError;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::memory::lock;
use crate::{SpanAttributes, SpanError, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan};

/// The erased `Err` projection of a span body: the `(name, message)` pair the
/// automatic error status records, upstream's `Error.name`/`Error.message`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanBodyError {
    /// A short classification of the failure.
    pub name: String,
    /// A human-readable description.
    pub message: String,
}

/// Projects a body's `Err` value onto the [`SpanBodyError`] pair the erased
/// layer records.
///
/// Implemented for every [`std::error::Error`] (type name + `Display`);
/// implement it for error types that are not `Error`s.
pub trait SpanBodyFailure {
    /// The `(name, message)` projection of this failure.
    fn span_body_failure(&self) -> SpanBodyError;
}

impl<E: StdError> SpanBodyFailure for E {
    fn span_body_failure(&self) -> SpanBodyError {
        SpanBodyError {
            name: std::any::type_name::<E>().to_owned(),
            message: self.to_string(),
        }
    }
}

/// The boxed future a span body resolves to: the erased outcome.
pub type ErasedSpanFuture = Pin<Box<dyn Future<Output = Result<(), SpanBodyError>> + Send>>;

/// The boxed body handed to the dispatch layer. It owns its data; borrowed
/// state stays on the raw generic trait.
pub type BoxedSpanBody = Box<dyn FnOnce(DynSpanHandle) -> ErasedSpanFuture + Send>;

/// Object-safe recording surface of a live span handle, handed to boxed
/// bodies.
///
/// Settlement and inert-after-settle semantics are the raw trait's; this is
/// the erased view of [`TelemetrySpan`].
pub trait DynTelemetrySpan: Send + Sync {
    /// The erased view of [`TelemetrySpan::add_event`].
    fn add_event(&self, name: &str, attributes: SpanAttributes);
    /// The erased view of [`TelemetrySpan::set_attributes`].
    fn set_attributes(&self, attributes: SpanAttributes);
    /// The erased view of [`TelemetrySpan::set_status`].
    fn set_status(&self, status: SpanStatus);
    /// Starts an erased child span whose settlement follows the boxed body.
    #[must_use]
    fn start_span(&self, options: SpanOptions, body: BoxedSpanBody) -> ErasedSpanFuture;
}

/// The clone-able erased span handle handed to boxed bodies; shares one
/// underlying span with every clone.
#[derive(Clone)]
pub struct DynSpanHandle(Arc<dyn DynTelemetrySpan>);

impl std::fmt::Debug for DynSpanHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynSpanHandle").finish_non_exhaustive()
    }
}

impl DynSpanHandle {
    /// Wraps a recording surface.
    #[must_use]
    pub fn new(inner: Arc<dyn DynTelemetrySpan>) -> Self {
        Self(inner)
    }

    /// Records an event; inert once settled.
    pub fn add_event(&self, name: &str, attributes: SpanAttributes) {
        self.0.add_event(name, attributes);
    }

    /// Merges attributes; inert once settled.
    pub fn set_attributes(&self, attributes: SpanAttributes) {
        self.0.set_attributes(attributes);
    }

    /// Replaces the status; inert once settled.
    pub fn set_status(&self, status: SpanStatus) {
        self.0.set_status(status);
    }

    /// Starts an erased child span whose settlement follows the boxed body.
    #[must_use]
    pub fn start_span(&self, options: SpanOptions, body: BoxedSpanBody) -> ErasedSpanFuture {
        self.0.start_span(options, body)
    }
}

/// Object-safe mirror of the span-start contract for runtime-selected
/// adapters.
///
/// The erased layer preserves the raw contract's settlement semantics and
/// records the erased failure's `(name, message)` pair as an explicit status,
/// which the automatic error status then preserves.
pub trait DynTelemetryContext: Send + Sync {
    /// Registers a span synchronously and returns a future that runs the
    /// boxed body and settles the span on its completion.
    fn start_span(&self, options: SpanOptions, body: BoxedSpanBody) -> ErasedSpanFuture;
}

/// The concrete `Clone` type downstream crates hold for a runtime-selected
/// telemetry context — the type in option bags and across crate boundaries.
///
/// Wrap any adapter with [`TelemetryHandle::new`]; the raw generic trait
/// remains available where the adapter type is statically known.
#[derive(Clone)]
pub struct TelemetryHandle(Arc<dyn DynTelemetryContext>);

impl std::fmt::Debug for TelemetryHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TelemetryHandle").finish_non_exhaustive()
    }
}

/// The explicit error status the erased layer records from a body failure, so
/// settlement preserves the `(name, message)` details over the automatic
/// status, which carries none.
fn erased_failure_status(failure: &SpanBodyError) -> SpanStatus {
    SpanStatus::Error {
        error: Some(SpanError {
            name: failure.name.clone(),
            message: failure.message.clone(),
        }),
    }
}

/// Records a failure's details as an explicit status unless the body already
/// set an explicit status itself — explicit calls win over the failure
/// projection, the raw contract's "last explicit call wins" rule.
fn settle_erased_failure<S: TelemetrySpan>(
    explicit_status: &AtomicBool,
    raw: &S,
    failure: &SpanBodyError,
) {
    if explicit_status.load(Ordering::Acquire) {
        return;
    }
    raw.set_status(erased_failure_status(failure));
}

impl TelemetryHandle {
    /// Wraps an adapter whose type is not statically known at the store site.
    /// The adapter and its spans must be `Send + Sync` because the handle
    /// crosses await points and threads; the adapter is shared behind an
    /// `Arc`, so no `Clone` bound is needed.
    pub fn new<C>(context: C) -> Self
    where
        C: TelemetryContext + Send + Sync + 'static,
        C::Span: TelemetrySpan + Send + Sync + 'static,
    {
        Self(Arc::new(ErasedContext {
            inner: Arc::new(context),
        }))
    }

    /// Starts a span with a fully erased body — the dispatch layer's own
    /// surface, for bodies that carry no typed value across the handle.
    /// Prefer [`TelemetryHandle::start_span`] for typed bodies.
    #[must_use]
    pub fn start_span_erased(&self, options: SpanOptions, body: BoxedSpanBody) -> ErasedSpanFuture {
        self.0.start_span(options, body)
    }

    /// Starts a span through the erased adapter and returns a future for the
    /// body's typed value. The body runs on first poll of the returned
    /// future; the span settles when it completes, `Ok` as success, `Err`
    /// with its [`SpanBodyFailure`] projection recorded as an explicit status.
    ///
    /// # Errors
    /// Propagates the body's `Err` value unchanged, after settling the span.
    ///
    /// # Panics
    /// Never, if adapters honor the settlement contract: the boxed body sends
    /// its typed outcome before the span settles, so the receiver is filled
    /// by the time settlement resolves. A broken adapter surfaces here.
    pub fn start_span<T, E, Fut, F>(
        &self,
        options: SpanOptions,
        body: F,
    ) -> impl Future<Output = Result<T, E>>
    where
        F: FnOnce(DynSpanHandle) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, E>> + Send + 'static,
        T: Send + 'static,
        E: SpanBodyFailure + Send + 'static,
    {
        let (sender, receiver) = once_channel();
        let boxed: BoxedSpanBody = Box::new(move |span| {
            Box::pin(async move {
                let outcome = body(span).await;
                let failure = outcome.as_ref().err().map(E::span_body_failure);
                sender.send(outcome);
                failure.map_or(Ok(()), Err)
            })
        });
        let settled = self.0.start_span(options, boxed);
        async move {
            let _ = settled.await;
            #[expect(
                clippy::expect_used,
                reason = "the boxed body sends its typed outcome before the span settles; \
                          a missing value is an adapter violating the contract, a programmer bug"
            )]
            receiver
                .await
                .expect("boxed body sends its typed outcome before the span settles")
        }
    }
}

/// The erased adapter: bridges the object-safe contract onto any adapter whose
/// span type is clone-able and `Send + Sync`.
struct ErasedContext<C> {
    inner: Arc<C>,
}

impl<C> DynTelemetryContext for ErasedContext<C>
where
    C: TelemetryContext + Send + Sync + 'static,
    C::Span: TelemetrySpan + Send + Sync + 'static,
{
    fn start_span(&self, options: SpanOptions, body: BoxedSpanBody) -> ErasedSpanFuture {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            inner
                .start_span(options, move |span: C::Span| async move {
                    let explicit_status = Arc::new(AtomicBool::new(false));
                    let erased_span = DynSpanHandle::new(Arc::new(ErasedSpan {
                        inner: Arc::new(span.clone()),
                        explicit_status: Arc::clone(&explicit_status),
                    }));
                    let outcome = body(erased_span).await;
                    if let Err(ref failure) = outcome {
                        settle_erased_failure(&explicit_status, &span, failure);
                    }
                    outcome
                })
                .await
        })
    }
}

/// The erased view of one adapter span, wrapping a clone-able generic span.
/// The flag records whether the body set an explicit status through the
/// erased handle, so the failure projection does not overwrite it.
struct ErasedSpan<S> {
    inner: Arc<S>,
    explicit_status: Arc<AtomicBool>,
}

impl<S> DynTelemetrySpan for ErasedSpan<S>
where
    S: TelemetrySpan + Send + Sync + 'static,
{
    fn add_event(&self, name: &str, attributes: SpanAttributes) {
        self.inner.add_event(name, attributes);
    }

    fn set_attributes(&self, attributes: SpanAttributes) {
        self.inner.set_attributes(attributes);
    }

    fn set_status(&self, status: SpanStatus) {
        self.explicit_status.store(true, Ordering::Release);
        self.inner.set_status(status);
    }

    fn start_span(&self, options: SpanOptions, body: BoxedSpanBody) -> ErasedSpanFuture {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            inner
                .start_span(options, move |child: S| async move {
                    let explicit_status = Arc::new(AtomicBool::new(false));
                    let erased_child = DynSpanHandle::new(Arc::new(Self {
                        inner: Arc::new(child.clone()),
                        explicit_status: Arc::clone(&explicit_status),
                    }));
                    let outcome = body(erased_child).await;
                    if let Err(ref failure) = outcome {
                        settle_erased_failure(&explicit_status, &child, failure);
                    }
                    outcome
                })
                .await
        })
    }
}

/// A std-only async oneshot, keeping the crate dependency-free; the typed
/// [`TelemetryHandle::start_span`] transports the body's value through one.
struct OnceShared<T> {
    slot: Mutex<Option<T>>,
    waker: Mutex<Option<Waker>>,
}

struct OnceSender<T> {
    shared: Arc<OnceShared<T>>,
}

struct OnceReceiver<T> {
    shared: Arc<OnceShared<T>>,
}

fn once_channel<T>() -> (OnceSender<T>, OnceReceiver<T>) {
    let shared = Arc::new(OnceShared {
        slot: Mutex::new(None),
        waker: Mutex::new(None),
    });
    (
        OnceSender {
            shared: Arc::clone(&shared),
        },
        OnceReceiver { shared },
    )
}

impl<T> OnceSender<T> {
    fn send(self, value: T) {
        *lock(&self.shared.slot) = Some(value);
        let waker = lock(&self.shared.waker).take();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

impl<T> OnceReceiver<T> {
    fn poll_value(&self, cx: &Context<'_>) -> Poll<Option<T>> {
        // The waker registers before the slot is read: a send between the two
        // steps finds the waker in place and wakes the task, and a value
        // stored earlier is taken below. No wake can be lost.
        *lock(&self.shared.waker) = Some(cx.waker().clone());
        lock(&self.shared.slot)
            .take()
            .map_or(Poll::Pending, |value| Poll::Ready(Some(value)))
    }
}

impl<T> Future for OnceReceiver<T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().poll_value(cx)
    }
}

#[cfg(test)]
mod once_channel_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingWake(AtomicUsize);

    impl std::task::Wake for CountingWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn send_before_poll_is_visible() {
        let (sender, receiver) = once_channel::<u8>();
        sender.send(3);
        let waker = Waker::noop();
        let cx = Context::from_waker(waker);
        assert_eq!(receiver.poll_value(&cx), Poll::Ready(Some(3)));
        assert_eq!(receiver.poll_value(&cx), Poll::Pending);
    }

    #[test]
    fn send_after_poll_registers_the_waker_and_wakes() {
        let (sender, receiver) = once_channel::<u8>();
        let wake_count = Arc::new(CountingWake(AtomicUsize::new(0)));
        let waker = Waker::from(wake_count.clone());
        let cx = Context::from_waker(&waker);
        assert_eq!(receiver.poll_value(&cx), Poll::Pending);
        assert_eq!(wake_count.0.load(Ordering::Relaxed), 0);
        sender.send(7);
        assert_eq!(wake_count.0.load(Ordering::Relaxed), 1);
        assert_eq!(receiver.poll_value(&cx), Poll::Ready(Some(7)));
    }

    #[test]
    fn receiver_future_polls_through_the_once_channel() {
        let (sender, receiver) = once_channel::<u32>();
        let polled = std::future::poll_fn(|cx| receiver.poll_value(cx));
        let mut polled = std::pin::pin!(polled);
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        assert!(matches!(Pin::new(&mut polled).poll(&mut cx), Poll::Pending));
        sender.send(9);
        assert_eq!(Pin::new(&mut polled).poll(&mut cx), Poll::Ready(Some(9)));
    }

    #[test]
    fn debug_impls_name_the_handle_without_state() {
        struct ProbeSpan;

        impl DynTelemetrySpan for ProbeSpan {
            fn add_event(&self, _name: &str, _attributes: SpanAttributes) {}
            fn set_attributes(&self, _attributes: SpanAttributes) {}
            fn set_status(&self, _status: SpanStatus) {}
            fn start_span(&self, _options: SpanOptions, _body: BoxedSpanBody) -> ErasedSpanFuture {
                unreachable!("probe never starts child spans")
            }
        }

        let span_debug = format!("{:?}", DynSpanHandle::new(Arc::new(ProbeSpan)));
        assert!(span_debug.contains("DynSpanHandle"));
        let handle_debug = format!(
            "{:?}",
            TelemetryHandle::new(crate::noop::NoopTelemetryContext)
        );
        assert!(handle_debug.contains("TelemetryHandle"));
    }
}
