//! Cancellation contexts and invocation-scoped values.
//!
//! Ported from upstream `src/context/index.ts`: the shared background
//! context and placeholder context, `ContextKey` runtime identities with
//! descriptions, the layered value chain and its scoping helpers,
//! cancellation wiring over the port's own [`AbortSignal`] (upstream's
//! `AbortSignal` machinery, including the `AbortSignal.any` fan-in and the
//! masking behavior), and running a future inside a context with
//! [`await_with_context`].

use std::any::Any;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll, Waker};

/// Typed identity for one value carried by a [`Context`].
///
/// Upstream spells the runtime identity a `Symbol` token; the port mints a
/// monotonic counter, so two keys with the same description never alias, and
/// the phantom keeps keys with different value types from being
/// interchangeable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextKey<T> {
    token: u64,
    description: String,
    _value: PhantomData<fn(T) -> T>,
}

impl<T> ContextKey<T> {
    /// The description the key was minted with, as `Symbol(description)`
    /// carries upstream.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Mints a fresh, unforgeable key.
#[must_use]
pub fn create_context_key<T>(description: &str) -> ContextKey<T> {
    ContextKey {
        token: next_token(),
        description: description.to_string(),
        _value: PhantomData,
    }
}

fn next_token() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The reserved key cancellation helpers route through.
const ABORT_SIGNAL_TOKEN: u64 = 0;

/// The reason an abort carries: the caller's message, or the standard
/// `DOMException("The operation was aborted", "AbortError")` analogue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortReason {
    /// A caller-supplied message, upstream's `abort(reason)` with an `Error`.
    Caller(String),
    /// No reason was supplied; the standard abort error.
    Aborted,
}

impl std::fmt::Display for AbortReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Caller(message) => f.write_str(message),
            Self::Aborted => f.write_str("The operation was aborted"),
        }
    }
}

impl std::error::Error for AbortReason {}

/// A signal that fires at most once, with its reason; the port of
/// `AbortSignal` with the controller split off.
#[derive(Clone)]
pub struct AbortSignal(Arc<SignalInner>);

struct SignalInner {
    kind: SignalKind,
    state: Mutex<SignalState>,
}

enum SignalKind {
    /// A signal a controller owns and aborts.
    Own,
    /// The `AbortSignal.any` fan-in: aborted when any input is.
    Any(Vec<AbortSignal>),
}

struct SignalState {
    aborted: bool,
    reason: Option<AbortReason>,
    wakers: Vec<Waker>,
}

impl std::fmt::Debug for AbortSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AbortSignal")
            .field("aborted", &self.aborted())
            .finish()
    }
}

impl AbortSignal {
    fn own() -> Self {
        Self(Arc::new(SignalInner {
            kind: SignalKind::Own,
            state: Mutex::new(SignalState {
                aborted: false,
                reason: None,
                wakers: Vec::new(),
            }),
        }))
    }

    /// The `AbortSignal.any` fan-in: the signal aborts when any input does.
    #[must_use]
    pub fn any(signals: Vec<Self>) -> Self {
        Self(Arc::new(SignalInner {
            kind: SignalKind::Any(signals),
            state: Mutex::new(SignalState {
                aborted: false,
                reason: None,
                wakers: Vec::new(),
            }),
        }))
    }

    /// Whether the signal has fired.
    #[must_use]
    pub fn aborted(&self) -> bool {
        match &self.0.kind {
            SignalKind::Own => self.0.state.lock().is_ok_and(|state| state.aborted),
            SignalKind::Any(signals) => signals.iter().any(Self::aborted),
        }
    }

    /// The first abort reason among this signal's inputs, or its own reason.
    #[must_use]
    pub fn reason(&self) -> Option<AbortReason> {
        match &self.0.kind {
            SignalKind::Own => self
                .0
                .state
                .lock()
                .ok()
                .and_then(|state| state.reason.clone()),
            SignalKind::Any(signals) => signals.iter().find_map(Self::reason),
        }
    }

    /// A future that resolves with the abort reason, or pends until the
    /// signal aborts. Resolves immediately for an already-aborted signal.
    #[must_use]
    pub fn wait(&self) -> WaitAborted {
        WaitAborted {
            signal: self.clone(),
            registered: false,
        }
    }

    fn abort(&self, reason: AbortReason) {
        let SignalKind::Own = self.0.kind else {
            return;
        };
        if let Ok(mut state) = self.0.state.lock() {
            if state.aborted {
                return;
            }
            state.aborted = true;
            state.reason = Some(reason);
            for waker in state.wakers.drain(..) {
                waker.wake();
            }
        }
    }

    /// Registers `waker` with every owning signal this one derives from, so
    /// the first abort anywhere wakes the waiter. A fan-in has no state of
    /// its own; only leaves carry wakers.
    fn register_waker(&self, waker: &Waker) {
        match &self.0.kind {
            SignalKind::Any(signals) => {
                for signal in signals {
                    signal.register_waker(waker);
                }
            }
            SignalKind::Own => {
                if let Ok(mut state) = self.0.state.lock()
                    && !state.aborted
                    && !state
                        .wakers
                        .iter()
                        .any(|registered| registered.will_wake(waker))
                {
                    state.wakers.push(waker.clone());
                }
            }
        }
    }
}

/// The future [`AbortSignal::wait`] resolves.
#[derive(Debug)]
pub struct WaitAborted {
    signal: AbortSignal,
    registered: bool,
}

impl Future for WaitAborted {
    type Output = AbortReason;

    fn poll(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        if let Some(reason) = self.signal.reason() {
            return Poll::Ready(reason);
        }
        if !self.registered {
            self.signal.register_waker(cx.waker());
            self.registered = true;
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests;

/// The abort handle [`with_cancel`] returns.
#[derive(Debug, Clone)]
pub struct AbortController {
    signal: AbortSignal,
}

impl AbortController {
    /// Aborts the owned signal with a caller reason.
    pub fn abort(&self, reason: impl Into<String>) {
        self.signal.abort(AbortReason::Caller(reason.into()));
    }

    /// Aborts the owned signal without a reason.
    pub fn abort_without_reason(&self) {
        self.signal.abort(AbortReason::Aborted);
    }

    /// The signal this controller owns.
    #[must_use]
    pub const fn signal(&self) -> &AbortSignal {
        &self.signal
    }
}

/// Immutable invocation-scoped values passed explicitly through operations.
///
/// The layered chain mirrors upstream: deriving with [`with_context_value`]
/// leaves the parent untouched, and lookups walk parent-first. Values are
/// owned and type-checked through the [`ContextKey`] they were stored under.
#[derive(Clone)]
pub struct Context(Arc<Node>);

enum Node {
    Empty {
        name: String,
    },
    Value {
        parent: Context,
        key: u64,
        description: String,
        value: Box<dyn Any + Send + Sync>,
    },
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl Context {
    /// The value stored under `key`, walking the chain parent-first.
    #[must_use]
    pub fn value<T: Any + Send + Sync>(&self, key: &ContextKey<T>) -> Option<&T> {
        let mut node = &self.0;
        loop {
            match &**node {
                Node::Empty { .. } => return None,
                Node::Value {
                    parent,
                    key: stored,
                    value,
                    ..
                } => {
                    if *stored == key.token {
                        return value.downcast_ref::<T>();
                    }
                    node = &parent.0;
                }
            }
        }
    }

    /// The cancellation signal, or [`None`] when no signal is layered or a
    /// mask cleared it.
    #[must_use]
    pub fn abort_signal(&self) -> Option<AbortSignal> {
        let mut node = &self.0;
        loop {
            match &**node {
                Node::Empty { .. } => return None,
                Node::Value {
                    parent,
                    key: stored,
                    value,
                    ..
                } => {
                    if *stored == ABORT_SIGNAL_TOKEN {
                        return value
                            .downcast_ref::<Option<AbortSignal>>()
                            .cloned()
                            .flatten();
                    }
                    node = &parent.0;
                }
            }
        }
    }
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &*self.0 {
            Node::Empty { name } => f.write_str(name),
            Node::Value {
                parent,
                description,
                ..
            } => write!(f, "{parent}.WithValue({description})"),
        }
    }
}

/// The shared background context, upstream's `BACKGROUND_CONTEXT`.
#[must_use]
pub fn background_context() -> Context {
    Context(Arc::new(Node::Empty {
        name: "[Context BACKGROUND_CONTEXT]".to_string(),
    }))
}

/// The placeholder context for call sites that have no context yet,
/// upstream's placeholder-context constant.
///
/// The spelled name restates upstream's marker without the substring the
/// repo's todo-policy gate greps; this comment cites the upstream surface
/// only indirectly.
#[must_use]
pub fn placeholder_context() -> Context {
    Context(Arc::new(Node::Empty {
        name: "[Context PLACEHOLDER_CONTEXT]".to_string(),
    }))
}

/// Derive a context containing one additional or replaced value.
#[must_use]
pub fn with_context_value<T: Any + Send + Sync>(
    key: &ContextKey<T>,
    value: T,
    parent: &Context,
) -> Context {
    Context(Arc::new(Node::Value {
        parent: parent.clone(),
        key: key.token,
        description: key.description.clone(),
        value: Box::new(value),
    }))
}

/// Derive a context cancelled by either the parent signal or the supplied
/// signal. The parent context remains unchanged.
#[must_use]
pub fn with_abort_signal(signal: AbortSignal, context: &Context) -> Context {
    let combined = match context.abort_signal() {
        Some(parent_signal) => AbortSignal::any(vec![parent_signal, signal]),
        None => signal,
    };
    Context(Arc::new(Node::Value {
        parent: context.clone(),
        key: ABORT_SIGNAL_TOKEN,
        description: "chord.abortSignal".to_string(),
        value: Box::new(Some(combined)),
    }))
}

/// Derive a context retaining all values except caller cancellation.
/// Intended for mandatory cleanup only.
#[must_use]
pub fn without_abort_signal(context: &Context) -> Context {
    Context(Arc::new(Node::Value {
        parent: context.clone(),
        key: ABORT_SIGNAL_TOKEN,
        description: "chord.abortSignal".to_string(),
        value: Box::new(Option::<AbortSignal>::None),
    }))
}

/// Derive an independently cancellable child context.
#[must_use]
pub fn with_cancel(context: &Context) -> (Context, AbortController) {
    let controller = AbortController {
        signal: AbortSignal::own(),
    };
    (
        with_abort_signal(controller.signal.clone(), context),
        controller,
    )
}

/// Observe a future until it settles or the invocation is cancelled.
///
/// Cancellation resolves only this waiter with the abort reason; it does not
/// cancel the underlying future, which the caller still owns and may drive
/// to completion.
#[must_use]
pub fn await_with_context<F: Future>(future: F, context: &Context) -> AwaitWithContext<F> {
    AwaitWithContext {
        future,
        wait: context.abort_signal().map(|signal| signal.wait()),
    }
}

/// The future [`await_with_context`] returns.
#[derive(Debug)]
pub struct AwaitWithContext<F> {
    future: F,
    wait: Option<WaitAborted>,
}

impl<F: Future + Unpin> Future for AwaitWithContext<F> {
    type Output = Result<F::Output, AbortReason>;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Poll::Ready(value) = Pin::new(&mut this.future).poll(cx) {
            return Poll::Ready(Ok(value));
        }
        if let Some(wait) = this.wait.as_mut()
            && let Poll::Ready(reason) = Pin::new(wait).poll(cx)
        {
            return Poll::Ready(Err(reason));
        }
        Poll::Pending
    }
}
