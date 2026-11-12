//! Concurrent interactive model-catalog refresh sharing, upstream
//! `src/modes/interactive/model-catalog-refresh.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! One all-catalog refresh runs per runtime while callers overlap; a caller
//! that stops waiting rejects with its own abort error without cancelling
//! the shared work, and the shared work is cancelled once the last caller
//! leaves. A later caller starts a fresh refresh.
//!
//! Port restatements:
//!
//! - Upstream keys the active refresh on the runtime object through a
//!   `WeakMap`. Rust has no object-keyed weak map, so the coordinator keys on
//!   [`ModelCatalogRuntime::catalog_identity`] and removes an entry when its
//!   last waiter leaves. [`ModelRuntime`] answers its shared core's address,
//!   which every handle clone resolves to the same value.
//! - The shared `Promise` is a detached [`tokio::spawn`] task publishing
//!   through a `watch` channel: a Rust future cannot be awaited by two
//!   callers, and the shared operation must outlive the first caller's
//!   future to keep upstream's "one caller stops waiting" behavior. The
//!   settled [`ModelsRefreshResult`] is handed out as `Arc` because it is
//!   not `Clone` and every caller observes the same result object upstream.
//! - `raceWithAbortSignal(active.promise, signal)` becomes the `select!`
//!   between the caller's [`CancellationToken`] and the shared completion;
//!   `signal.throwIfAborted()` becomes the early `is_cancelled` check.
//! - Upstream's `.finally` aborts the controller when the waiter count
//!   reaches zero and the shared promise then deletes the map entry. The
//!   port removes the entry, cancels the controller, and aborts the task in
//!   one step under the coordinator lock, so a caller either joins the
//!   entry before it is removed or starts a fresh refresh.
//! - A panicking refresh task has no upstream shape (the promise rejects
//!   with the thrown error); here it surfaces as
//!   [`ModelCatalogRefreshError::Failed`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use pi_ai::models::{ModelsRefreshOptions, ModelsRefreshResult};
use pi_ai::types::BoxedFuture;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::model_runtime::{ModelRuntime, ModelRuntimeCore};
use crate::utils::abort::AbortError;

/// The runtime slice the coordinator needs, upstream's
/// `Pick<ModelRuntime, "refresh">`.
///
/// [`Self::catalog_identity`] is the port's stand-in for the object identity
/// a `WeakMap` key would carry: equal values mean the same runtime, and
/// clones of one runtime handle answer the same value.
pub trait ModelCatalogRuntime {
    /// Refresh every provider catalog, upstream's `refresh`.
    fn refresh(&self, options: ModelsRefreshOptions) -> BoxedFuture<'_, ModelsRefreshResult>;

    /// The stable identity of the runtime the refresh belongs to.
    fn catalog_identity(&self) -> usize;
}

/// Refresh one runtime through its inherent method; the helper keeps the
/// trait implementation below from resolving `refresh` to itself.
fn refresh_runtime(
    model_runtime: &ModelRuntime,
    options: ModelsRefreshOptions,
) -> BoxedFuture<'_, ModelsRefreshResult> {
    // `ModelRuntime` derefs to its core; naming the core's method keeps the
    // call from resolving back to this trait's `refresh`.
    Box::pin(ModelRuntimeCore::refresh(model_runtime, options))
}

impl ModelCatalogRuntime for ModelRuntime {
    fn refresh(&self, options: ModelsRefreshOptions) -> BoxedFuture<'_, ModelsRefreshResult> {
        refresh_runtime(self, options)
    }

    fn catalog_identity(&self) -> usize {
        let core: &ModelRuntimeCore = self;
        std::ptr::from_ref(core).addr()
    }
}

/// Why a shared catalog refresh did not deliver a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelCatalogRefreshError {
    /// The caller's signal aborted before the refresh settled, upstream's
    /// `AbortError` from `raceWithAbortSignal`.
    Aborted,
    /// The shared refresh task ended without publishing a result.
    Failed,
}

impl ModelCatalogRefreshError {
    /// Whether the caller's own cancellation ended the wait.
    #[must_use]
    pub const fn is_aborted(self) -> bool {
        matches!(self, Self::Aborted)
    }
}

impl std::fmt::Display for ModelCatalogRefreshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Aborted => std::fmt::Display::fmt(&AbortError, f),
            Self::Failed => f.write_str("the shared model catalog refresh failed"),
        }
    }
}

impl std::error::Error for ModelCatalogRefreshError {}

/// The shared refresh state, upstream `ActiveModelCatalogRefresh`.
struct ActiveModelCatalogRefresh {
    /// Cancels the shared operation, upstream's `controller`.
    controller: CancellationToken,
    /// The published result every waiter watches.
    result: watch::Receiver<Option<Arc<ModelsRefreshResult>>>,
    /// The detached refresh task; aborting it restates cancellation of an
    /// abandoned promise.
    task: tokio::task::JoinHandle<()>,
    /// The callers still waiting, upstream's `waiters`.
    waiters: AtomicUsize,
}

/// The active refreshes keyed by runtime identity, upstream's
/// `activeByRuntime` `WeakMap`.
static ACTIVE_REFRESHES: LazyLock<Mutex<HashMap<usize, Arc<ActiveModelCatalogRefresh>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn active_refreshes()
-> std::sync::MutexGuard<'static, HashMap<usize, Arc<ActiveModelCatalogRefresh>>> {
    ACTIVE_REFRESHES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Join the runtime's active refresh or start one, incrementing the waiter
/// count, upstream's `activeByRuntime.get`/creation block plus `waiters++`.
fn acquire<R>(model_runtime: &R, identity: usize) -> Arc<ActiveModelCatalogRefresh>
where
    R: ModelCatalogRuntime + Clone + Send + Sync + 'static,
{
    let mut active = active_refreshes();
    if let Some(existing) = active.get(&identity) {
        existing.waiters.fetch_add(1, Ordering::SeqCst);
        return Arc::clone(existing);
    }

    let (result_tx, result_rx) = watch::channel(None);
    let controller = CancellationToken::new();
    let task_controller = controller.clone();
    let runtime = model_runtime.clone();
    let task = tokio::spawn(async move {
        let result = runtime
            .refresh(ModelsRefreshOptions {
                signal: Some(task_controller),
                ..ModelsRefreshOptions::default()
            })
            .await;
        let _ = result_tx.send(Some(Arc::new(result)));
    });
    let created = Arc::new(ActiveModelCatalogRefresh {
        controller,
        result: result_rx,
        task,
        waiters: AtomicUsize::new(1),
    });
    active.insert(identity, Arc::clone(&created));
    created
}

/// Decrement a waiter and, at zero, retire the entry, upstream's
/// `.finally` decrement plus the shared promise's deletion.
fn release(identity: usize, active: &Arc<ActiveModelCatalogRefresh>) {
    let retire = {
        let mut refreshes = active_refreshes();
        if active.waiters.fetch_sub(1, Ordering::SeqCst) != 1 {
            return;
        }
        let is_active = refreshes
            .get(&identity)
            .is_some_and(|entry| Arc::ptr_eq(entry, active));
        if is_active {
            refreshes.remove(&identity);
        }
        is_active
    };
    if retire {
        active.controller.cancel();
        active.task.abort();
    }
}

/// Decrements the waiter count on every exit path, including a dropped
/// caller future — the port's shape for a `.finally` attached when the
/// raced promise is created rather than when it is awaited.
struct WaiterGuard {
    identity: usize,
    active: Arc<ActiveModelCatalogRefresh>,
}

impl Drop for WaiterGuard {
    fn drop(&mut self) {
        release(self.identity, &self.active);
    }
}

/// Wait until the shared task publishes a result, or report that it ended
/// without doing so.
async fn await_result(
    mut result: watch::Receiver<Option<Arc<ModelsRefreshResult>>>,
) -> Result<Arc<ModelsRefreshResult>, ModelCatalogRefreshError> {
    loop {
        let published = result.borrow_and_update().clone();
        if let Some(value) = published {
            return Ok(value);
        }
        if result.changed().await.is_err() {
            return Err(ModelCatalogRefreshError::Failed);
        }
    }
}

/// Share concurrent interactive all-catalog refreshes while keeping each
/// caller's cancellation independent, upstream `refreshModelCatalogs`.
///
/// # Errors
/// [`ModelCatalogRefreshError::Aborted`] when `signal` aborts before the
/// shared refresh settles; [`ModelCatalogRefreshError::Failed`] when the
/// shared refresh task ends without a result.
pub async fn refresh_model_catalogs<R>(
    model_runtime: &R,
    signal: &CancellationToken,
) -> Result<Arc<ModelsRefreshResult>, ModelCatalogRefreshError>
where
    R: ModelCatalogRuntime + Clone + Send + Sync + 'static,
{
    if signal.is_cancelled() {
        return Err(ModelCatalogRefreshError::Aborted);
    }

    let identity = model_runtime.catalog_identity();
    let active = acquire(model_runtime, identity);
    let _waiter = WaiterGuard {
        identity,
        active: Arc::clone(&active),
    };

    tokio::select! {
        () = signal.cancelled() => Err(ModelCatalogRefreshError::Aborted),
        result = await_result(active.result.clone()) => result,
    }
}
