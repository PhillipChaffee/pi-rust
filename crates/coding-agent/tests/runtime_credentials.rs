//! Upstream `packages/coding-agent/test/runtime-credentials.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::runtime_credentials` (#120).
//!
//! Porting restatements this suite records:
//!
//! - The persistent-store doubles implement [`CredentialStore`] directly;
//!   `vi.spyOn(storage, "delete")` restates as a wrapping store that fails
//!   one delete with the abort failure.
//! - The signal-forwarding probe records the received cancellation token per
//!   call; the identity assertion restates as cancelling the caller's token
//!   after the calls and observing every recorded handle report cancelled —
//!   only handles cloned from that one token do.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::resolve::now_ms;
use pi_ai::auth::types::{
    ApiKeyCredential, AuthError, AuthOptions, AuthType, Credential, CredentialInfo,
    CredentialModifyFn, OAuthCredentials,
};
use pi_ai::types::BoxedFuture;
use pi_ai::utils::abort::AbortError;
use pi_coding_agent::auth_storage::{AuthStorage, AuthStorageData, InMemoryAuthStorageBackend};
use pi_coding_agent::runtime_credentials::RuntimeCredentials;
use tokio_util::sync::CancellationToken;

/// The boxed failure the double reports, the trait's error type: the helper's
/// return type carries the unsize coercion the call site cannot spell as a
/// trivial cast.
fn boxed_error(error: impl std::error::Error + Send + Sync + 'static) -> AuthError {
    Box::new(error)
}

/// The poisoned-mutex-tolerant lock the recording cell reads through.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The shared handle letting a test read the wrapped store's own view,
/// upstream keeping both `storage` and `credentials` in scope.
struct SharedStore(Arc<AuthStorage<InMemoryAuthStorageBackend>>);

impl CredentialStore for SharedStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        CredentialStore::read(&*self.0, provider_id, options)
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        CredentialStore::list(&*self.0, options)
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        CredentialStore::modify(&*self.0, provider_id, f, options)
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        CredentialStore::delete(&*self.0, provider_id, options)
    }
}

/// The store that records every received signal, upstream's `received`
/// array and the inline double pushing `options?.signal` into it.
struct RecordingStore {
    signals: Arc<Mutex<Vec<Option<CancellationToken>>>>,
}

impl RecordingStore {
    fn record(&self, options: Option<&AuthOptions>) {
        lock(&self.signals).push(options.and_then(|options| options.signal.clone()));
    }
}

impl CredentialStore for RecordingStore {
    fn read<'a>(
        &'a self,
        _provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        self.record(options);
        Box::pin(async move { Ok(None) })
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        self.record(options);
        Box::pin(async move { Ok(Vec::new()) })
    }

    fn modify<'a>(
        &'a self,
        _provider_id: &'a str,
        _f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        self.record(options);
        Box::pin(async move { Ok(None) })
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        self.record(options);
        Box::pin(async move { Ok(()) })
    }
}

/// The store whose first `delete` fails with the abort failure, upstream's
/// `vi.spyOn(storage, "delete").mockRejectedValueOnce(aborted)`.
struct AbortOnceDeleteStore {
    inner: Arc<dyn CredentialStore>,
    aborted: AtomicBool,
    deletes: Arc<AtomicUsize>,
}

impl CredentialStore for AbortOnceDeleteStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        self.inner.read(provider_id, options)
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        self.inner.list(options)
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        self.inner.modify(provider_id, f, options)
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        self.deletes.fetch_add(1, Ordering::SeqCst);
        let first_failure = !self.aborted.swap(true, Ordering::SeqCst);
        Box::pin(async move {
            if first_failure {
                return Err(boxed_error(AbortError));
            }
            self.inner.delete(provider_id, options).await
        })
    }
}

/// An api-key credential with the given key, the shape the read assertions pin.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the helper returns the store read's full Option shape so assertions compare it whole"
)]
fn api_key(key: &str) -> Option<Credential> {
    Some(Credential::ApiKey(ApiKeyCredential {
        key: Some(key.to_owned()),
        env: None,
    }))
}

/// Seed an in-memory store with one api-key credential, the setup the
/// masking and deletion tests share.
fn in_memory_store(provider_id: &str, key: &str) -> Arc<AuthStorage<InMemoryAuthStorageBackend>> {
    let mut data = AuthStorageData::new();
    data.insert(
        provider_id.to_owned(),
        serde_json::json!({"type": "api_key", "key": key}),
    );
    Arc::new(in_memory(&data))
}

/// The in-memory store, the call spelled with its backend parameter because
/// the generic constructor path cannot infer it.
fn in_memory(data: &AuthStorageData) -> AuthStorage<InMemoryAuthStorageBackend> {
    AuthStorage::<InMemoryAuthStorageBackend>::in_memory(data)
}

#[tokio::test]
async fn runtime_overrides_mask_stored_credentials_without_persisting() {
    let storage = in_memory_store("anthropic", "stored-key");
    let credentials = RuntimeCredentials::new(Arc::new(SharedStore(Arc::clone(&storage))));

    credentials.set_runtime_api_key("anthropic", "runtime-key".to_owned());
    let masked = credentials.read("anthropic", None).await.expect("read");
    assert_eq!(masked, api_key("runtime-key"));

    let stored = storage.read("anthropic", None).await.expect("stored read");
    assert_eq!(
        stored,
        api_key("stored-key"),
        "the override never persisted"
    );

    credentials.remove_runtime_api_key("anthropic");
    let restored = credentials
        .read("anthropic", None)
        .await
        .expect("restored read");
    assert_eq!(restored, api_key("stored-key"));
}

#[tokio::test]
async fn enumeration_merges_overrides_without_exposing_keys() {
    let oauth = Credential::OAuth(OAuthCredentials {
        refresh: "refresh".to_owned(),
        access: "access".to_owned(),
        expires: now_ms() + 60_000,
        extra: BTreeMap::new(),
    });
    let mut data = AuthStorageData::new();
    data.insert(
        "anthropic".to_owned(),
        serde_json::to_value(&oauth).expect("seed serializes"),
    );
    let storage = Arc::new(in_memory(&data));
    let credentials = RuntimeCredentials::new(Arc::new(SharedStore(Arc::clone(&storage))));
    credentials.set_runtime_api_key("anthropic", "runtime-key".to_owned());
    credentials.set_runtime_api_key("openai", "other-runtime-key".to_owned());

    let listed = credentials.list(None).await.expect("list");
    assert_eq!(
        listed,
        vec![
            CredentialInfo {
                provider_id: "anthropic".to_owned(),
                auth_type: AuthType::ApiKey,
            },
            CredentialInfo {
                provider_id: "openai".to_owned(),
                auth_type: AuthType::ApiKey,
            },
        ],
    );
}

#[tokio::test]
async fn forwards_operation_signals_to_the_persistent_store() {
    let signals = Arc::new(Mutex::new(Vec::new()));
    let credentials = RuntimeCredentials::new(Arc::new(RecordingStore {
        signals: Arc::clone(&signals),
    }));

    let token = CancellationToken::new();
    let options = AuthOptions {
        signal: Some(token.clone()),
    };
    credentials
        .read("anthropic", Some(&options))
        .await
        .expect("read");
    credentials.list(Some(&options)).await.expect("list");
    let modify: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(None) }));
    credentials
        .modify("anthropic", modify, Some(&options))
        .await
        .expect("modify");
    credentials
        .delete("anthropic", Some(&options))
        .await
        .expect("delete");

    let recorded = lock(&signals).clone();
    assert_eq!(
        recorded.len(),
        4,
        "every operation reached the persistent store"
    );
    token.cancel();
    assert!(
        recorded
            .iter()
            .all(|signal| signal.as_ref().is_some_and(CancellationToken::is_cancelled)),
        "all four calls saw the operation's token"
    );
}

#[tokio::test]
async fn keeps_a_runtime_override_when_persistent_deletion_is_cancelled() {
    let inner = in_memory_store("anthropic", "stored-key");
    let deletes = Arc::new(AtomicUsize::new(0));
    let credentials = RuntimeCredentials::new(Arc::new(AbortOnceDeleteStore {
        inner,
        aborted: AtomicBool::new(false),
        deletes: Arc::clone(&deletes),
    }));
    credentials.set_runtime_api_key("anthropic", "runtime-key".to_owned());

    let token = CancellationToken::new();
    let options = AuthOptions {
        signal: Some(token),
    };
    let outcome = credentials.delete("anthropic", Some(&options)).await;
    let error = outcome.expect_err("the cancelled deletion must fail");
    assert!(
        error.downcast_ref::<AbortError>().is_some(),
        "the deletion fails with the abort failure, got: {error}"
    );
    assert_eq!(
        deletes.load(Ordering::SeqCst),
        1,
        "the store saw one delete"
    );

    let read = credentials.read("anthropic", None).await.expect("read");
    assert_eq!(
        read,
        api_key("runtime-key"),
        "the override survives the failed delete"
    );
}

#[tokio::test]
async fn delete_clears_both_the_override_and_persisted_credential() {
    let storage = in_memory_store("anthropic", "stored-key");
    let credentials = RuntimeCredentials::new(Arc::new(SharedStore(Arc::clone(&storage))));
    credentials.set_runtime_api_key("anthropic", "runtime-key".to_owned());

    credentials.delete("anthropic", None).await.expect("delete");

    let read = credentials.read("anthropic", None).await.expect("read");
    assert_eq!(read, None);
    let listed = credentials.list(None).await.expect("list");
    assert_eq!(listed, Vec::<CredentialInfo>::new());
}
