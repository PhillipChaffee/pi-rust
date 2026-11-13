//! The interactive catalog-refresh coordinator at pin `60e7e76`: the three
//! upstream `model-catalog-refresh.test.ts` cases plus early-abort and
//! re-settlement boundaries.

#![expect(
    clippy::expect_used,
    reason = "a shared result that failed unexpectedly fails the assertion by design"
)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pi_ai::models::{ModelsRefreshOptions, ModelsRefreshResult};
use pi_ai::types::BoxedFuture;
use pi_coding_agent::modes::interactive::model_catalog_refresh::{
    ModelCatalogRefreshError, ModelCatalogRuntime, refresh_model_catalogs,
};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

fn lock<T>(value: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    value.lock().unwrap_or_else(PoisonError::into_inner)
}

type RefreshFn =
    dyn Fn(ModelsRefreshOptions) -> BoxedFuture<'static, ModelsRefreshResult> + Send + Sync;

#[derive(Clone)]
struct FakeRuntime {
    id: usize,
    calls: Arc<AtomicUsize>,
    signals: Arc<Mutex<Vec<CancellationToken>>>,
    refresh: Arc<RefreshFn>,
}

impl FakeRuntime {
    fn new(
        refresh: impl Fn(ModelsRefreshOptions) -> BoxedFuture<'static, ModelsRefreshResult>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            calls: Arc::new(AtomicUsize::new(0)),
            signals: Arc::new(Mutex::new(Vec::new())),
            refresh: Arc::new(refresh),
        }
    }
}

impl ModelCatalogRuntime for FakeRuntime {
    fn refresh(&self, options: ModelsRefreshOptions) -> BoxedFuture<'_, ModelsRefreshResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(signal) = &options.signal {
            lock(&self.signals).push(signal.clone());
        }
        (self.refresh)(options)
    }

    fn catalog_identity(&self) -> usize {
        self.id
    }
}

/// A runtime whose refresh waits for the returned sender.
fn deferred_runtime() -> (FakeRuntime, oneshot::Sender<ModelsRefreshResult>) {
    let (release, receiver) = oneshot::channel();
    let receiver = Arc::new(Mutex::new(Some(receiver)));
    let runtime = FakeRuntime::new(move |_options| {
        let receiver = receiver
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        Box::pin(async move {
            match receiver {
                Some(receiver) => receiver.await.unwrap_or_default(),
                None => ModelsRefreshResult::default(),
            }
        })
    });
    (runtime, release)
}

/// A runtime whose refresh never settles and ignores its token.
fn pending_runtime() -> FakeRuntime {
    FakeRuntime::new(|_options| {
        Box::pin(async { std::future::pending::<ModelsRefreshResult>().await })
    })
}

async fn wait_until(mut condition: impl FnMut() -> bool) {
    let mut reached = false;
    for _ in 0..10_000 {
        if condition() {
            reached = true;
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(reached, "condition was not reached");
}

#[test]
fn model_runtime_satisfies_the_coordinator_bounds() {
    fn assert_bounds<T: ModelCatalogRuntime + Clone + Send + Sync + 'static>() {}
    assert_bounds::<pi_coding_agent::model_runtime::ModelRuntime>();
}

#[tokio::test]
async fn shares_one_runtime_refresh_between_concurrent_callers() {
    let (runtime, release) = deferred_runtime();
    let first = CancellationToken::new();
    let second = CancellationToken::new();
    let calls = Arc::clone(&runtime.calls);
    let releaser = async move {
        wait_until(|| calls.load(Ordering::SeqCst) == 1).await;
        let _ = release.send(ModelsRefreshResult::default());
    };

    let (first_result, second_result, ()) = tokio::join!(
        refresh_model_catalogs(&runtime, &first),
        refresh_model_catalogs(&runtime, &second),
        releaser,
    );

    let first_result = first_result.expect("first caller");
    let second_result = second_result.expect("second caller");
    assert!(Arc::ptr_eq(&first_result, &second_result));
    assert!(!first_result.aborted);
    assert!(first_result.errors.is_empty());
    assert_eq!(runtime.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn keeps_the_shared_refresh_alive_when_one_caller_stops_waiting() {
    let (runtime, release) = deferred_runtime();
    let first = CancellationToken::new();
    let second = CancellationToken::new();
    let gate_runtime = runtime.clone();
    let signals = Arc::clone(&runtime.signals);
    let first_for_gate = first.clone();
    let (first_done_sender, first_done) = oneshot::channel();

    let first_branch = async {
        let outcome = refresh_model_catalogs(&runtime, &first).await;
        assert!(matches!(outcome, Err(ModelCatalogRefreshError::Aborted)));
        let _ = first_done_sender.send(());
    };
    let second_branch = refresh_model_catalogs(&runtime, &second);
    let gate = async move {
        wait_until(|| gate_runtime.calls.load(Ordering::SeqCst) == 1).await;
        first_for_gate.cancel();
        let _ = first_done.await;
        {
            let recorded = lock(&signals);
            assert_eq!(recorded.len(), 1);
            let cancelled = recorded[0].is_cancelled();
            drop(recorded);
            assert!(!cancelled, "the shared refresh survives one caller leaving");
        }
        let _ = release.send(ModelsRefreshResult::default());
    };

    let ((), second_result, ()) = tokio::join!(first_branch, second_branch, gate);
    assert!(second_result.is_ok());
}

#[tokio::test]
async fn aborts_an_abandoned_refresh_and_allows_a_later_refresh_to_start() {
    let runtime = pending_runtime();
    let first = CancellationToken::new();
    let first_cancel = first.clone();
    let calls = Arc::clone(&runtime.calls);
    let cancel_first = async move {
        wait_until(|| calls.load(Ordering::SeqCst) == 1).await;
        first_cancel.cancel();
    };
    let (first_outcome, ()) = tokio::join!(refresh_model_catalogs(&runtime, &first), cancel_first);

    assert!(matches!(
        first_outcome,
        Err(ModelCatalogRefreshError::Aborted)
    ));
    assert!(lock(&runtime.signals)[0].is_cancelled());

    let second = CancellationToken::new();
    let second_cancel = second.clone();
    let calls = Arc::clone(&runtime.calls);
    let cancel_second = async move {
        wait_until(|| calls.load(Ordering::SeqCst) == 2).await;
        second_cancel.cancel();
    };
    let (second_outcome, ()) =
        tokio::join!(refresh_model_catalogs(&runtime, &second), cancel_second);

    assert_eq!(runtime.calls.load(Ordering::SeqCst), 2);
    assert!(matches!(
        second_outcome,
        Err(ModelCatalogRefreshError::Aborted)
    ));
}

#[tokio::test]
async fn rejects_an_already_aborted_signal_without_starting_a_refresh() {
    let runtime = pending_runtime();
    let signal = CancellationToken::new();
    signal.cancel();

    let outcome = refresh_model_catalogs(&runtime, &signal).await;

    assert_eq!(outcome.err(), Some(ModelCatalogRefreshError::Aborted));
    assert_eq!(runtime.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_settled_refresh_does_not_serve_a_later_caller() {
    let (runtime, release) = deferred_runtime();
    let first = CancellationToken::new();
    let calls = Arc::clone(&runtime.calls);
    let releaser = async move {
        wait_until(|| calls.load(Ordering::SeqCst) == 1).await;
        let _ = release.send(ModelsRefreshResult::default());
    };
    let (first_result, ()) = tokio::join!(refresh_model_catalogs(&runtime, &first), releaser);
    assert!(first_result.is_ok());

    let second = CancellationToken::new();
    let second_result = refresh_model_catalogs(&runtime, &second).await;

    assert!(second_result.is_ok());
    assert_eq!(
        runtime.calls.load(Ordering::SeqCst),
        2,
        "the retired entry starts a fresh refresh"
    );
}
