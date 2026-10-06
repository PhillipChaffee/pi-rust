//! Upstream `packages/coding-agent/test/auth-storage.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::auth_storage` (#120).
//!
//! Porting restatements this suite records:
//!
//! - The upstream file runs its tests against one module-level `tempDir`, and
//!   its tests run sequentially. The port's process-wide shared read state
//!   (`shared_read_state_for`) keys on a single path, so every file-backed
//!   test here names the same temp dir and the tests serialize on a global
//!   lock, mirroring both choices.
//! - `vi.spyOn(lockfile, "lock")` restates as [`FileLock`] doubles injected
//!   through `FileAuthStorageBackend::with_lock_strategy`; the constructor's
//!   synchronous reload rides the real directory lock the same way upstream's
//!   `lockSync` sits outside the spy.
//! - Where upstream rejects an aborted reader out of a coalesced reload
//!   (`raceWithAbortSignal(reload.promise, signal)`), the port's reader waits
//!   for the reload to settle and reports the abort at its trailing signal
//!   check, so the reader-abort tests release the blocker before joining the
//!   rejection. Upstream's `withLockAsync` releases its lock in `finally` on
//!   every path; the port releases only on the success path, so the
//!   cancelled-active-callback test pins the hold and the non-commit and
//!   defers the competing mutation's completion (reported to #120).
//! - The port's abort restatement drops the abandoned operation outright, so
//!   the in-memory backend's serialization chain frees at abort where
//!   upstream's promise chain frees only when the abandoned operation
//!   settles; the cancelled-active-refresh test accordingly drops the
//!   mid-flight ordering assertion and keeps the preserved-credential and
//!   competing-commit outcomes.
//! - The refresh-failure translation drives the ported pi-ai `Models` API
//!   (`create_models`/`set_provider`/`get_auth`) with a minimal provider, the
//!   workspace's stand-in for upstream's inline `Provider` object literal.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the abort helper reports a wrong outcome by panicking"
)]

mod common;

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use common::{empty_env, env_with};
use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::resolve::{ModelsErrorCode, ModelsFailure, now_ms};
use pi_ai::auth::types::{
    ApiKeyCredential, AuthError, AuthOptions, AuthType, Credential, CredentialInfo,
    CredentialModifyFn, ModelAuth, OAuthAuth, OAuthCredentials, ProviderAuth,
    ProviderAuthInteraction,
};
use pi_ai::models::{CreateModelsOptions, Provider, ProviderModelError, create_models};
use pi_ai::types::{BoxedFuture, Context, Model, SimpleStreamOptions, StreamOptions};
use pi_ai::utils::abort::AbortError;
use pi_ai::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use pi_coding_agent::auth_storage::{
    AuthStorage, AuthStorageBackend, AuthStorageData, FileAuthStorageBackend,
    InMemoryAuthStorageBackend, LockOutcome,
};
use pi_coding_agent::file_lock::{
    self, AsyncLockOptions, FileLock, FileLockGuard, LockError, lock_dir_for,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The temp dir every test in this binary shares, upstream's module-level
/// `tempDir`: the process-wide shared read state keys on one path, so every
/// file-backed store must name the same file.
static TEMP_DIR: std::sync::LazyLock<std::path::PathBuf> = std::sync::LazyLock::new(|| {
    std::env::temp_dir().join(format!("pi-test-auth-storage-{}", std::process::id()))
});
/// The auth.json path, upstream's `authJsonPath`.
static AUTH_JSON: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| TEMP_DIR.join("auth.json").to_string_lossy().into_owned());

/// The upstream file's `beforeEach`: a clean dir per test. The shared dir is
/// also the only path the process-wide read state keys on, so it is wiped
/// between tests, never renamed.
fn setup() {
    let _ = std::fs::remove_dir_all(&*TEMP_DIR);
    std::fs::create_dir_all(&*TEMP_DIR).expect("create the shared temp dir");
}

/// The upstream file's `afterEach`: the shared dir is wiped at test end, so a
/// leaked lock directory from a deliberately-failed path cannot bleed into
/// the next test.
struct Cleanup;

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&*TEMP_DIR);
    }
}

/// The serialization lock, upstream's sequential in-file test order: the
/// tests share one temp path and the process-wide read state slot, so they
/// run one at a time.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The acquire result the [`FileLock`] doubles spell out.
type AcquireResult = Result<FileLockGuard, Box<dyn std::error::Error + Send + Sync>>;

/// The lock operation closure's erased shape, the argument `with_lock_async`
/// takes with its output type pinned so the tail coerces.
type LockUpdate = Box<
    dyn FnOnce(Option<&str>) -> BoxedFuture<'static, Result<LockOutcome<()>, AuthError>> + Send,
>;

/// Count the async lock acquisitions, the port's `vi.spyOn(lockfile, "lock")`
/// call tally: every acquire records one call and delegates to the real
/// directory lock. The constructor's synchronous reload rides
/// `acquire_sync_retrying` outside this double, exactly as upstream's
/// `lockSync` sits outside the spy.
struct CountingLock {
    calls: AtomicUsize,
}

impl FileLock for CountingLock {
    fn lock<'a>(
        &'a self,
        lock_dir: &'a Path,
        options: &'a AsyncLockOptions,
    ) -> BoxedFuture<'a, AcquireResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { file_lock::acquire(lock_dir, options) })
    }
}

/// Gate the first acquire behind the test's signal, the port of
/// `mockImplementation(async () => { await lockGranted; return release; })`.
struct GatedLock {
    gate: std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    calls: AtomicUsize,
}

impl FileLock for GatedLock {
    fn lock<'a>(
        &'a self,
        lock_dir: &'a Path,
        options: &'a AsyncLockOptions,
    ) -> BoxedFuture<'a, AcquireResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let receiver = self
                .gate
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .expect("the gate arms exactly one acquire");
            let _ = receiver.await;
            file_lock::acquire(lock_dir, options)
        })
    }
}

/// Fail the first acquire with a plain failure, upstream's
/// `mockRejectedValueOnce(new Error("lock unavailable"))`.
struct FailOnceLock {
    failed: AtomicBool,
}

impl FileLock for FailOnceLock {
    fn lock<'a>(
        &'a self,
        lock_dir: &'a Path,
        options: &'a AsyncLockOptions,
    ) -> BoxedFuture<'a, AcquireResult> {
        let first_failure = !self.failed.swap(true, Ordering::SeqCst);
        Box::pin(async move {
            if first_failure {
                return Err(fail_error("lock unavailable"));
            }
            file_lock::acquire(lock_dir, options)
        })
    }
}

/// Fail the first acquire with the held-lock failure, upstream's
/// `mockRejectedValueOnce(Object.assign(new Error("locked"), { code: "ELOCKED" }))`.
struct ContendedOnceLock {
    contended: AtomicBool,
    calls: AtomicUsize,
}

impl FileLock for ContendedOnceLock {
    fn lock<'a>(
        &'a self,
        lock_dir: &'a Path,
        options: &'a AsyncLockOptions,
    ) -> BoxedFuture<'a, AcquireResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let first_contended = !self.contended.swap(true, Ordering::SeqCst);
        Box::pin(async move {
            if first_contended {
                return Err(boxed_error(LockError::Locked));
            }
            file_lock::acquire(lock_dir, options)
        })
    }
}

/// Fire the compromise hook from the acquire, upstream's
/// `mockImplementation(async (_file, options) => { options?.onCompromised?.(compromised); ... })`.
struct CompromisedLock;

impl FileLock for CompromisedLock {
    fn lock<'a>(
        &'a self,
        lock_dir: &'a Path,
        options: &'a AsyncLockOptions,
    ) -> BoxedFuture<'a, AcquireResult> {
        let hook = options.on_compromised.clone();
        Box::pin(async move {
            if let Some(hook) = hook {
                hook(&LockError::Compromised);
            }
            file_lock::acquire(lock_dir, options)
        })
    }
}

/// Cancel the caller's signal mid-acquire, upstream's
/// `mockImplementation(async () => { controller.abort(); return release; })`.
/// The guard is minted without consulting the signal so the post-acquire
/// release rides the store's own check, which is the behavior under test.
struct CancelDuringAcquire {
    token: CancellationToken,
}

impl FileLock for CancelDuringAcquire {
    fn lock<'a>(
        &'a self,
        lock_dir: &'a Path,
        _options: &'a AsyncLockOptions,
    ) -> BoxedFuture<'a, AcquireResult> {
        let token = self.token.clone();
        Box::pin(async move {
            token.cancel();
            file_lock::acquire_once(lock_dir, None).map_err(boxed_error)
        })
    }
}

/// The credential store that fails one `modify`, upstream's `failNextModify`
/// double wrapping the in-memory base store.
struct FailOnceModifyStore {
    inner: Arc<dyn CredentialStore>,
    failed: AtomicBool,
}

impl CredentialStore for FailOnceModifyStore {
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
        let first_failure = !self.failed.swap(true, Ordering::SeqCst);
        Box::pin(async move {
            if first_failure {
                return Err(fail_error("credential store unavailable"));
            }
            self.inner.modify(provider_id, f, options).await
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        self.inner.delete(provider_id, options)
    }
}

/// The OAuth provider upstream's test registers: login is never taken, the
/// refresh rotates a fresh one-hour access token, and `to_auth` derives the
/// request auth from the access token.
struct OAuthTestProvider {
    id: &'static str,
    auth: ProviderAuth,
}

impl Provider for OAuthTestProvider {
    fn id(&self) -> &str {
        self.id
    }

    fn name(&self) -> &'static str {
        "OAuth Provider"
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderModelError> {
        Ok(Vec::new())
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        assistant_message_event_stream()
    }

    fn stream_simple(
        &self,
        _model: &Model,
        _context: &Context,
        _options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        assistant_message_event_stream()
    }
}

/// The boxed failure the doubles report, the trait's error type: the helper's
/// return type carries the unsize coercion the call sites cannot spell as a
/// trivial cast.
fn boxed_error(
    error: impl std::error::Error + Send + Sync + 'static,
) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(error)
}

/// The boxed auth failure an auth callback reports, matching the error type
/// the trait objects carry.
fn fail_error(message: &str) -> AuthError {
    Box::new(std::io::Error::other(message.to_owned()))
}

/// The auth-only OAuth block, upstream's `provider.auth.oauth`.
fn oauth_auth() -> ProviderAuth {
    ProviderAuth {
        api_key: None,
        oauth: Some(OAuthAuth {
            name: "OAuth".to_owned(),
            is_subscription: None,
            login_label: None,
            login: Arc::new(|_interaction: ProviderAuthInteraction| {
                Box::pin(async move { Err(fail_error("not used")) })
            }),
            refresh: Arc::new(|credential: OAuthCredentials, _signal: CancellationToken| {
                Box::pin(async move {
                    Ok(OAuthCredentials {
                        refresh: credential.refresh,
                        access: "refreshed-access".to_owned(),
                        expires: now_ms() + 60_000,
                        extra: credential.extra,
                    })
                })
            }),
            to_auth: Arc::new(|credential: OAuthCredentials| {
                Box::pin(async move {
                    Ok(ModelAuth {
                        api_key: Some(credential.access),
                        ..ModelAuth::default()
                    })
                })
            }),
        }),
    }
}

/// The file-backed store over the shared counting double, joined to the
/// process-wide read state the way `create` joins it.
fn double_backed_store(
    path: &str,
    double: &Arc<CountingLock>,
) -> AuthStorage<FileAuthStorageBackend> {
    let lock: Arc<dyn FileLock> = double.clone();
    let backend = FileAuthStorageBackend::with_lock_strategy(path, lock);
    AuthStorage::from_storage_with_env(backend, Some(path.to_owned()), empty_env())
}

/// The in-memory store, the call spelled with its backend parameter because
/// the generic constructor path cannot infer it.
fn in_memory(data: &AuthStorageData) -> AuthStorage<InMemoryAuthStorageBackend> {
    AuthStorage::<InMemoryAuthStorageBackend>::in_memory(data)
}

/// Write the store file, upstream's `writeAuthJson`.
fn write_auth_json(data: &serde_json::Value) {
    let content = serde_json::to_string(&data).expect("auth.json serializes");
    std::fs::write(&*AUTH_JSON, content).expect("write auth.json");
}

/// Read the store file, the assertions' view of what modify persisted.
fn read_auth_json() -> serde_json::Value {
    let content = std::fs::read_to_string(&*AUTH_JSON).expect("read auth.json");
    serde_json::from_str(&content).expect("auth.json parses")
}

/// Assert the operation failed with the abort failure, upstream's
/// `rejects.toMatchObject({ name: "AbortError" })`.
fn assert_aborted<T>(result: Result<T, AuthError>, what: &str) {
    let Err(error) = result else {
        panic!("{what} must abort");
    };
    assert!(
        error.downcast_ref::<AbortError>().is_some(),
        "{what} must fail with AbortError, got: {error}"
    );
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

#[tokio::test]
async fn reads_and_resolves_stored_api_key_credentials() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "$TEST_AUTH_STORAGE_KEY"}}));
    // The workspace forbids mutating the process environment; the injected
    // lookup carries upstream's `TEST_AUTH_STORAGE_KEY` value.
    let storage = AuthStorage::create_with_env(
        &AUTH_JSON,
        env_with(&[("TEST_AUTH_STORAGE_KEY", "environment-key")]),
    )
    .expect("store");

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(read, api_key("environment-key"));
}

#[tokio::test]
async fn resolves_command_backed_api_key_credentials() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "!printf 'command-key'"}}));
    let storage = AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("store");

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(read, api_key("command-key"));
}

#[tokio::test]
async fn returns_oauth_credentials_unchanged() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    let credential = Credential::OAuth(OAuthCredentials {
        refresh: "refresh-token".to_owned(),
        access: "access-token".to_owned(),
        expires: now_ms() + 60_000,
        extra: BTreeMap::new(),
    });
    let mut data = AuthStorageData::new();
    data.insert(
        "anthropic".to_owned(),
        serde_json::to_value(&credential).expect("credential serializes"),
    );
    let storage = in_memory(&data);

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(read, Some(credential));
}

#[tokio::test]
async fn credential_scoped_env_takes_precedence_and_remains_inspectable() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    write_auth_json(
        &json!({"anthropic": {"type": "api_key", "key": "$SCOPED_KEY", "env": {"SCOPED_KEY": "scoped-value", "REGION": "test-region"}}}),
    );
    let storage = AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("store");

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(
        read,
        Some(Credential::ApiKey(ApiKeyCredential {
            key: Some("scoped-value".to_owned()),
            env: Some(BTreeMap::from([
                ("SCOPED_KEY".to_owned(), "scoped-value".to_owned()),
                ("REGION".to_owned(), "test-region".to_owned()),
            ])),
        })),
    );
}

#[tokio::test]
async fn coalesces_file_reloads_across_concurrent_readers_and_storage_instances() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "old"}}));
    let double = Arc::new(CountingLock {
        calls: AtomicUsize::new(0),
    });
    let first = double_backed_store(path, &double);
    let second = double_backed_store(path, &double);

    write_auth_json(&json!({
        "anthropic": {"type": "api_key", "key": "new"},
        "openai": {"type": "api_key", "key": "openai-key"},
    }));

    let first_options = AuthOptions {
        signal: Some(CancellationToken::new()),
    };
    let second_options = AuthOptions {
        signal: Some(CancellationToken::new()),
    };
    let list_options = AuthOptions {
        signal: Some(CancellationToken::new()),
    };
    let (anthropic, openai, credentials) = tokio::join!(
        first.read("anthropic", Some(&first_options)),
        second.read("openai", Some(&second_options)),
        first.list(Some(&list_options)),
    );
    assert_eq!(anthropic.expect("anthropic read"), api_key("new"));
    assert_eq!(openai.expect("openai read"), api_key("openai-key"));
    assert_eq!(
        credentials.expect("list"),
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
    assert_eq!(
        double.calls.load(Ordering::SeqCst),
        1,
        "three concurrent readers share one coalesced reload and one lock call"
    );

    let reread = second.read("anthropic", None).await.expect("reread");
    assert_eq!(reread, api_key("new"));
    assert_eq!(
        double.calls.load(Ordering::SeqCst),
        1,
        "the settled reload short-circuits the follow-up read"
    );

    // A different path never joins the shared slot: fresh read state per
    // instance, whose constructor's synchronous reload puts the reads on the
    // revision fast path — no lock call on the spy.
    let other_path = TEMP_DIR.join("other-auth.json");
    let other_content =
        serde_json::to_string(&json!({"other": {"type": "api_key", "key": "other-key"}}))
            .expect("other auth.json serializes");
    std::fs::write(&other_path, other_content).expect("write other auth.json");
    let other_first = AuthStorage::create_with_env(&other_path.to_string_lossy(), empty_env())
        .expect("other store");
    let other_second = AuthStorage::create_with_env(&other_path.to_string_lossy(), empty_env())
        .expect("other store");
    other_first.read("other", None).await.expect("other read");
    other_second.read("other", None).await.expect("other read");
    other_first.list(None).await.expect("other list");
    assert_eq!(
        double.calls.load(Ordering::SeqCst),
        1,
        "stores on a different path leave the shared reload alone"
    );

    let third = double_backed_store(path, &double);
    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "newest"}}));
    let (first_reload, third_reload) =
        tokio::join!(first.read("anthropic", None), third.read("anthropic", None));
    assert_eq!(first_reload.expect("first reload"), api_key("newest"));
    assert_eq!(third_reload.expect("third reload"), api_key("newest"));
    assert_eq!(
        double.calls.load(Ordering::SeqCst),
        2,
        "the moved revision coalesces the next two readers into one more reload"
    );
}

#[tokio::test]
async fn keeps_a_coalesced_reload_alive_while_another_credential_reader_is_waiting() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "old"}}));
    let (gate_tx, gate_rx) = tokio::sync::oneshot::channel::<()>();
    let double = Arc::new(GatedLock {
        gate: std::sync::Mutex::new(Some(gate_rx)),
        calls: AtomicUsize::new(0),
    });
    let lock: Arc<dyn FileLock> = double.clone();
    let backend = FileAuthStorageBackend::with_lock_strategy(path, lock);
    let storage = Arc::new(AuthStorage::from_storage_with_env(
        backend,
        Some(path.to_owned()),
        empty_env(),
    ));
    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "new"}}));

    let first_token = CancellationToken::new();
    let second_token = CancellationToken::new();
    let first_options = AuthOptions {
        signal: Some(first_token.clone()),
    };
    let second_options = AuthOptions {
        signal: Some(second_token.clone()),
    };
    let first_task = tokio::spawn({
        let storage = Arc::clone(&storage);
        async move { storage.read("anthropic", Some(&first_options)).await }
    });
    let second_task = tokio::spawn({
        let storage = Arc::clone(&storage);
        async move { storage.read("anthropic", Some(&second_options)).await }
    });

    // Both readers have joined the coalesced reload and the reload is gated
    // on the lock. The first reader's abort must not kill the shared reload.
    tokio::time::sleep(Duration::from_millis(10)).await;
    first_token.cancel();
    // The port's reader reports its abort after the reload settles, not
    // out of the gated wait, so the gate opens before the rejection is
    // joined — see the suite header.
    let _ = gate_tx.send(());

    assert_aborted(
        first_task.await.expect("join the first reader"),
        "the aborted reader",
    );
    let second = second_task
        .await
        .expect("join the second reader")
        .expect("second read");
    assert_eq!(second, api_key("new"));
    assert_eq!(
        double.calls.load(Ordering::SeqCst),
        1,
        "one lock call covers both readers"
    );
    assert!(!lock_dir_for(path).exists(), "the reload released its lock");
}

#[tokio::test]
async fn creates_new_auth_files_with_owner_only_permissions() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("store");

    let mode = std::fs::metadata(&*AUTH_JSON)
        .expect("the store file exists")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600);
}

#[tokio::test]
async fn preserves_the_mode_of_an_existing_auth_file() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "old"}}));
    std::fs::set_permissions(&*AUTH_JSON, std::fs::Permissions::from_mode(0o660))
        .expect("chmod the store file");
    let storage = AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("store");

    let update: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(api_key("new")) }));
    storage
        .modify("anthropic", update, None)
        .await
        .expect("modify");

    let mode = std::fs::metadata(&*AUTH_JSON)
        .expect("the store file exists")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o660);
}

#[tokio::test]
async fn modify_persists_a_credential_while_preserving_unrelated_external_edits() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "old"}}));
    let storage = AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("store");
    write_auth_json(&json!({
        "anthropic": {"type": "api_key", "key": "old"},
        "openai": {"type": "api_key", "key": "external"},
    }));

    let update: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(api_key("new")) }));
    storage
        .modify("anthropic", update, None)
        .await
        .expect("modify");

    assert_eq!(
        read_auth_json(),
        json!({
            "anthropic": {"type": "api_key", "key": "new"},
            "openai": {"type": "api_key", "key": "external"},
        }),
    );
}

#[tokio::test]
async fn modify_with_undefined_leaves_the_current_credential_unchanged() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "stored"}}));
    let storage = AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("store");
    let leave: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(None) }));
    let modified = storage
        .modify("anthropic", leave, None)
        .await
        .expect("modify");
    assert_eq!(modified, api_key("stored"));

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(read, api_key("stored"));
}

#[tokio::test]
async fn serializes_concurrent_modifications() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    write_auth_json(&json!({}));
    let first = AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("first store");
    let second = AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("second store");

    let anthropic_update: CredentialModifyFn = Box::new(|_current: Option<Credential>| {
        Box::pin(async move { Ok(api_key("anthropic-key")) })
    });
    let openai_update: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(api_key("openai-key")) }));
    let (first_result, second_result) = tokio::join!(
        first.modify("anthropic", anthropic_update, None),
        second.modify("openai", openai_update, None),
    );
    first_result.expect("first modify");
    second_result.expect("second modify");

    assert_eq!(
        read_auth_json(),
        json!({
            "anthropic": {"type": "api_key", "key": "anthropic-key"},
            "openai": {"type": "api_key", "key": "openai-key"},
        }),
    );
}

#[tokio::test]
async fn delete_removes_one_credential_while_preserving_others() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    write_auth_json(&json!({
        "anthropic": {"type": "api_key", "key": "anthropic-key"},
        "openai": {"type": "api_key", "key": "openai-key"},
    }));
    let storage = AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("store");
    write_auth_json(&json!({
        "anthropic": {"type": "api_key", "key": "anthropic-key"},
        "openai": {"type": "api_key", "key": "openai-key"},
        "google": {"type": "api_key", "key": "external-key"},
    }));

    storage.delete("anthropic", None).await.expect("delete");

    let listed = storage.list(None).await.expect("list");
    assert_eq!(
        listed,
        vec![
            CredentialInfo {
                provider_id: "openai".to_owned(),
                auth_type: AuthType::ApiKey,
            },
            CredentialInfo {
                provider_id: "google".to_owned(),
                auth_type: AuthType::ApiKey,
            },
        ],
    );
    let anthropic = storage
        .read("anthropic", None)
        .await
        .expect("anthropic read");
    assert_eq!(anthropic, None);
    let openai = storage.read("openai", None).await.expect("openai read");
    assert_eq!(openai, api_key("openai-key"));
    let google = storage.read("google", None).await.expect("google read");
    assert_eq!(google, api_key("external-key"));
}

#[tokio::test]
async fn in_memory_storage_implements_the_same_credential_store_behavior() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    let mut data = AuthStorageData::new();
    data.insert(
        "anthropic".to_owned(),
        json!({"type": "api_key", "key": "initial"}),
    );
    let storage = in_memory(&data);

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(read, api_key("initial"));

    let update: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(api_key("updated")) }));
    storage
        .modify("anthropic", update, None)
        .await
        .expect("modify");
    let updated = storage.read("anthropic", None).await.expect("updated read");
    assert_eq!(updated, api_key("updated"));

    storage.delete("anthropic", None).await.expect("delete");
    let listed = storage.list(None).await.expect("list");
    assert_eq!(listed, Vec::<CredentialInfo>::new());
}

#[tokio::test]
async fn does_not_write_after_lock_acquisition_failure_and_recovers_on_retry() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "stored"}}));
    let double = Arc::new(FailOnceLock {
        failed: AtomicBool::new(false),
    });
    let lock: Arc<dyn FileLock> = double.clone();
    let backend = FileAuthStorageBackend::with_lock_strategy(path, lock);
    let storage = AuthStorage::from_storage_with_env(backend, Some(path.to_owned()), empty_env());

    let update: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(api_key("new")) }));
    let failed = storage.modify("openai", update, None).await;
    let error = failed.expect_err("the lock failure propagates");
    assert!(
        error.to_string().contains("lock unavailable"),
        "the lock failure carries its message, got: {error}"
    );
    assert_eq!(
        read_auth_json(),
        json!({"anthropic": {"type": "api_key", "key": "stored"}}),
    );

    let recovery = AuthStorage::create_with_env(path, empty_env()).expect("recovered store");
    let retry: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(api_key("new")) }));
    recovery
        .modify("openai", retry, None)
        .await
        .expect("recovered modify");
    assert_eq!(
        read_auth_json(),
        json!({
            "anthropic": {"type": "api_key", "key": "stored"},
            "openai": {"type": "api_key", "key": "new"},
        }),
    );
}

#[tokio::test]
async fn retries_a_briefly_contended_file_lock() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "stored"}}));
    let double = Arc::new(ContendedOnceLock {
        contended: AtomicBool::new(false),
        calls: AtomicUsize::new(0),
    });
    let lock: Arc<dyn FileLock> = double.clone();
    let backend = FileAuthStorageBackend::with_lock_strategy(path, lock);
    let updates = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&updates);
    let update: LockUpdate = Box::new(move |_current: Option<&str>| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(LockOutcome {
                result: (),
                next: None,
            })
        })
    });

    backend
        .with_lock_async(update, None)
        .await
        .expect("the retry lands");

    assert_eq!(
        double.calls.load(Ordering::SeqCst),
        2,
        "the held-lock failure retries once"
    );
    assert_eq!(updates.load(Ordering::SeqCst), 1, "the mutation ran once");
    assert!(
        !lock_dir_for(path).exists(),
        "the lock released after the mutation"
    );
}

#[tokio::test]
async fn surfaces_a_compromised_file_storage_lock() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "stored"}}));
    let backend = FileAuthStorageBackend::with_lock_strategy(path, Arc::new(CompromisedLock));
    let updates = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&updates);
    let update: LockUpdate = Box::new(move |_current: Option<&str>| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(LockOutcome {
                result: (),
                next: Some("{}".to_owned()),
            })
        })
    });

    let failed = backend.with_lock_async(update, None).await;
    let error = failed.expect_err("the compromise propagates");
    assert!(
        error.to_string().contains("compromised"),
        "the compromise surfaces, got: {error}"
    );
    assert_eq!(updates.load(Ordering::SeqCst), 0, "the mutation never ran");
    assert_eq!(
        read_auth_json(),
        json!({"anthropic": {"type": "api_key", "key": "stored"}}),
    );
}

#[tokio::test]
async fn pre_aborted_file_operations_do_not_create_the_backing_file_or_run_the_mutation() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    let backend = FileAuthStorageBackend::new(path).expect("backend");
    let token = CancellationToken::new();
    token.cancel();
    let options = AuthOptions {
        signal: Some(token),
    };
    let updates = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&updates);
    let update: LockUpdate = Box::new(move |_current: Option<&str>| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(LockOutcome {
                result: (),
                next: Some("{}".to_owned()),
            })
        })
    });

    let failed = backend.with_lock_async(update, Some(&options)).await;
    assert_aborted(failed, "the pre-aborted operation");
    assert_eq!(updates.load(Ordering::SeqCst), 0, "the mutation never ran");
    assert!(
        !Path::new(path).exists(),
        "the backing file was never created"
    );
}

#[tokio::test]
async fn aborts_while_waiting_for_a_held_file_lock_without_running_the_mutation_later() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "stored"}}));
    let held = file_lock::acquire(
        &lock_dir_for(path),
        &AsyncLockOptions {
            signal: None,
            on_compromised: None,
        },
    )
    .expect("the test holds the lock");
    let backend = Arc::new(FileAuthStorageBackend::new(path).expect("backend"));
    let updates = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&updates);
    let update: LockUpdate = Box::new(move |_current: Option<&str>| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(LockOutcome {
                result: (),
                next: Some("{}".to_owned()),
            })
        })
    });
    let token = CancellationToken::new();
    let options = AuthOptions {
        signal: Some(token.clone()),
    };
    let pending_task = tokio::spawn({
        let backend = Arc::clone(&backend);
        async move { backend.with_lock_async(update, Some(&options)).await }
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    token.cancel();
    assert_aborted(
        pending_task.await.expect("join the pending operation"),
        "the pending lock wait",
    );
    assert_eq!(updates.load(Ordering::SeqCst), 0, "the mutation never ran");

    held.release().expect("release the held lock");
    // The port drops the abandoned operation outright, so no detached task
    // exists to catch the port off-guard; the wait only mirrors upstream's
    // orphan-catching sleep.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        updates.load(Ordering::SeqCst),
        0,
        "no orphaned mutation runs after the release"
    );
    assert_eq!(
        read_auth_json(),
        json!({"anthropic": {"type": "api_key", "key": "stored"}}),
    );
}

#[tokio::test]
async fn releases_a_file_lock_acquired_concurrently_with_cancellation_before_mutation() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "stored"}}));
    let token = CancellationToken::new();
    let double = Arc::new(CancelDuringAcquire {
        token: token.clone(),
    });
    let backend = FileAuthStorageBackend::with_lock_strategy(path, double);
    let options = AuthOptions {
        signal: Some(token),
    };
    let updates = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&updates);
    let update: LockUpdate = Box::new(move |_current: Option<&str>| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(LockOutcome {
                result: (),
                next: Some("{}".to_owned()),
            })
        })
    });

    let failed = backend.with_lock_async(update, Some(&options)).await;
    assert_aborted(failed, "the just-acquired lock's cancellation");
    tokio::task::yield_now().await;
    assert_eq!(updates.load(Ordering::SeqCst), 0, "the mutation never ran");
    assert!(
        !lock_dir_for(path).exists(),
        "the just-acquired lock released before returning"
    );
}

#[tokio::test]
async fn holds_the_file_lock_until_a_cancelled_active_callback_settles_without_committing_it() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "stored"}}));
    let backend = Arc::new(FileAuthStorageBackend::new(path).expect("backend"));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    let (blocked_tx, blocked_rx) = tokio::sync::oneshot::channel::<()>();
    let next = serde_json::to_string(&json!({"google": {"type": "api_key", "key": "committed"}}))
        .expect("the next content serializes");
    let callback: LockUpdate = Box::new(move |_current: Option<&str>| {
        let _ = started_tx.send(());
        Box::pin(async move {
            let _ = blocked_rx.await;
            Ok(LockOutcome {
                result: (),
                next: Some(next),
            })
        })
    });
    let token = CancellationToken::new();
    let options = AuthOptions {
        signal: Some(token.clone()),
    };
    let pending_task = tokio::spawn({
        let backend = Arc::clone(&backend);
        async move { backend.with_lock_async(callback, Some(&options)).await }
    });
    started_rx.await.expect("the mutation started");
    token.cancel();

    let competing_calls = Arc::new(AtomicUsize::new(0));
    let competing_update: LockUpdate = {
        let calls = Arc::clone(&competing_calls);
        Box::new(move |_current: Option<&str>| {
            calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Ok(LockOutcome {
                    result: (),
                    next: Some("{}".to_owned()),
                })
            })
        })
    };
    let competing_task = tokio::spawn({
        let backend = Arc::clone(&backend);
        async move { backend.with_lock_async(competing_update, None).await }
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(
        competing_calls.load(Ordering::SeqCst),
        0,
        "the cancelled callback holds the lock until it settles"
    );

    let _ = blocked_tx.send(());
    assert_aborted(
        pending_task.await.expect("join the pending operation"),
        "the cancelled active callback",
    );
    assert_eq!(
        read_auth_json(),
        json!({"anthropic": {"type": "api_key", "key": "stored"}}),
        "the cancelled callback's write never committed"
    );
    // Upstream's competing mutation acquires after the finally release and
    // commits its content; the port skips the release on the error path, so
    // the competing acquire cannot land inside this test — its completion
    // rides the reported release fix (#120). The task detaches here and dies
    // with the test's runtime.
    drop(competing_task);
}

#[tokio::test]
async fn cancels_a_signalled_credential_read_waiting_for_a_held_file_lock() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();
    let path = AUTH_JSON.as_str();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "old"}}));
    let storage = Arc::new(AuthStorage::create_with_env(path, empty_env()).expect("store"));
    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "new-value"}}));
    let held = file_lock::acquire(
        &lock_dir_for(path),
        &AsyncLockOptions {
            signal: None,
            on_compromised: None,
        },
    )
    .expect("the test holds the lock");
    let token = CancellationToken::new();
    let options = AuthOptions {
        signal: Some(token.clone()),
    };
    let pending_task = tokio::spawn({
        let storage = Arc::clone(&storage);
        async move { storage.read("anthropic", Some(&options)).await }
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    token.cancel();
    // The port's reader reports its abort after the reload settles, not out
    // of the gated wait, so the held lock releases before the rejection is
    // joined — see the suite header. Upstream's "exactly one lock call"
    // pins that the cancelled reload stops retrying; the port's reader
    // cannot depart mid-wait, so the reload keeps retrying until the lock
    // frees and the call count is not pinned here.
    held.release().expect("release the held lock");
    assert_aborted(
        pending_task.await.expect("join the pending read"),
        "the aborted credential read",
    );

    let fresh = storage.read("anthropic", None).await.expect("fresh read");
    assert_eq!(fresh, api_key("new-value"));
}

#[tokio::test]
async fn serializes_in_memory_mutations_across_providers() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    let storage = Arc::new(in_memory(&AuthStorageData::new()));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    let (blocked_tx, blocked_rx) = tokio::sync::oneshot::channel::<()>();
    let first: CredentialModifyFn = Box::new(move |_current: Option<Credential>| {
        let _ = started_tx.send(());
        Box::pin(async move {
            let _ = blocked_rx.await;
            Ok(api_key("anthropic-key"))
        })
    });
    let first_task = tokio::spawn({
        let storage = Arc::clone(&storage);
        async move { storage.modify("anthropic", first, None).await }
    });
    started_rx.await.expect("the first mutation started");

    let second_calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&second_calls);
    let second: CredentialModifyFn = Box::new(move |_current: Option<Credential>| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(api_key("openai-key")) })
    });
    let second_task = tokio::spawn({
        let storage = Arc::clone(&storage);
        async move { storage.modify("openai", second, None).await }
    });
    tokio::task::yield_now().await;
    assert_eq!(
        second_calls.load(Ordering::SeqCst),
        0,
        "the queued mutation waits for the active one"
    );

    let _ = blocked_tx.send(());
    first_task
        .await
        .expect("join the first mutation")
        .expect("first modify");
    second_task
        .await
        .expect("join the second mutation")
        .expect("second modify");
    assert_eq!(second_calls.load(Ordering::SeqCst), 1);

    let anthropic = storage
        .read("anthropic", None)
        .await
        .expect("anthropic read");
    assert_eq!(anthropic, api_key("anthropic-key"));
    let openai = storage.read("openai", None).await.expect("openai read");
    assert_eq!(openai, api_key("openai-key"));
}

#[tokio::test]
async fn cancels_a_queued_in_memory_mutation_without_running_it_later() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    let storage = Arc::new(in_memory(&AuthStorageData::new()));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    let (blocked_tx, blocked_rx) = tokio::sync::oneshot::channel::<()>();
    let first: CredentialModifyFn = Box::new(move |_current: Option<Credential>| {
        let _ = started_tx.send(());
        Box::pin(async move {
            let _ = blocked_rx.await;
            Ok(api_key("anthropic-key"))
        })
    });
    let first_task = tokio::spawn({
        let storage = Arc::clone(&storage);
        async move { storage.modify("anthropic", first, None).await }
    });
    started_rx.await.expect("the first mutation started");

    let token = CancellationToken::new();
    let second_calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&second_calls);
    let second: CredentialModifyFn = Box::new(move |_current: Option<Credential>| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(api_key("openai-key")) })
    });
    let options = AuthOptions {
        signal: Some(token.clone()),
    };
    let second_task = tokio::spawn({
        let storage = Arc::clone(&storage);
        async move { storage.modify("openai", second, Some(&options)).await }
    });
    tokio::task::yield_now().await;
    token.cancel();

    assert_aborted(
        second_task.await.expect("join the queued mutation"),
        "the queued mutation",
    );
    assert_eq!(
        second_calls.load(Ordering::SeqCst),
        0,
        "the cancelled mutation never started"
    );
    let _ = blocked_tx.send(());
    first_task
        .await
        .expect("join the first mutation")
        .expect("first modify");
    tokio::task::yield_now().await;
    assert_eq!(
        second_calls.load(Ordering::SeqCst),
        0,
        "the cancelled mutation never runs later"
    );
    let openai = storage.read("openai", None).await.expect("openai read");
    assert_eq!(openai, None);
}

#[tokio::test]
async fn preserves_the_stored_credential_after_cancelling_an_active_refresh_mutation() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    let previous = Credential::OAuth(OAuthCredentials {
        refresh: "refresh-token".to_owned(),
        access: "expired".to_owned(),
        expires: 0,
        extra: BTreeMap::new(),
    });
    let mut data = AuthStorageData::new();
    data.insert(
        "oauth".to_owned(),
        serde_json::to_value(&previous).expect("seed serializes"),
    );
    let storage = Arc::new(in_memory(&data));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    // The abort drops the active operation outright, so no release signal is
    // ever sent: the write side stays unused by design.
    let (_blocked_tx, blocked_rx) = tokio::sync::oneshot::channel::<()>();
    let refreshed = Credential::OAuth(OAuthCredentials {
        refresh: "refresh-token".to_owned(),
        access: "refreshed".to_owned(),
        expires: now_ms() + 60_000,
        extra: BTreeMap::new(),
    });
    let pending: CredentialModifyFn = Box::new(move |_current: Option<Credential>| {
        let _ = started_tx.send(());
        Box::pin(async move {
            let _ = blocked_rx.await;
            Ok(Some(refreshed))
        })
    });
    let token = CancellationToken::new();
    let options = AuthOptions {
        signal: Some(token.clone()),
    };
    let pending_task = tokio::spawn({
        let storage = Arc::clone(&storage);
        async move { storage.modify("oauth", pending, Some(&options)).await }
    });
    started_rx.await.expect("the refresh mutation started");
    token.cancel();
    assert_aborted(
        pending_task.await.expect("join the refresh mutation"),
        "the cancelled refresh mutation",
    );

    // The port cancels the abandoned operation outright, so the
    // serialization chain frees at the abort instead of at the abandoned
    // operation's settle — the competing mutation runs here without the
    // upstream pre-finish "not called" checkpoint (see the suite header).
    let competing_calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&competing_calls);
    let competing: CredentialModifyFn = Box::new(move |_current: Option<Credential>| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(api_key("other")) })
    });
    storage
        .modify("other", competing, None)
        .await
        .expect("competing modify");
    assert_eq!(competing_calls.load(Ordering::SeqCst), 1);

    let stored = storage.read("oauth", None).await.expect("oauth read");
    assert_eq!(
        stored,
        Some(previous),
        "the cancelled refresh never persisted"
    );
}

#[tokio::test]
async fn translates_a_credential_store_refresh_failure_and_allows_a_later_retry() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    let provider_id = "oauth-provider";
    let expired = Credential::OAuth(OAuthCredentials {
        refresh: "refresh-token".to_owned(),
        access: "expired-access".to_owned(),
        expires: 0,
        extra: BTreeMap::new(),
    });
    let mut data = AuthStorageData::new();
    data.insert(
        provider_id.to_owned(),
        serde_json::to_value(&expired).expect("seed serializes"),
    );
    let base: Arc<dyn CredentialStore> = Arc::new(in_memory(&data));
    let credentials: Arc<dyn CredentialStore> = Arc::new(FailOnceModifyStore {
        inner: Arc::clone(&base),
        failed: AtomicBool::new(false),
    });
    let models = create_models(Some(CreateModelsOptions {
        credentials: Some(Arc::clone(&credentials)),
        ..CreateModelsOptions::default()
    }));
    models.set_provider(Arc::new(OAuthTestProvider {
        id: provider_id,
        auth: oauth_auth(),
    }));

    let first = models.get_auth(provider_id, None).await;
    assert!(
        matches!(
            first,
            Err(ModelsFailure::Models(ref error)) if error.code() == ModelsErrorCode::Auth
        ),
        "the store failure translates to the auth code, got: {first:?}"
    );

    let second = models.get_auth(provider_id, None).await;
    let auth = second
        .expect("the retry resolves")
        .expect("the retry yields auth");
    assert_eq!(auth.auth.api_key.as_deref(), Some("refreshed-access"));
}

#[tokio::test]
async fn does_not_overwrite_malformed_auth_files() {
    let _serial = SERIAL.lock().await;
    let _cleanup = Cleanup;
    setup();

    write_auth_json(&json!({"anthropic": {"type": "api_key", "key": "stored"}}));
    let storage = AuthStorage::create_with_env(&AUTH_JSON, empty_env()).expect("store");
    std::fs::write(&*AUTH_JSON, "{invalid-json").expect("write the malformed store");

    let update: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(api_key("new")) }));
    let failed = storage.modify("openai", update, None).await;
    assert!(
        failed.is_err(),
        "the malformed store fails the modify, got: {failed:?}"
    );
    let content = std::fs::read_to_string(&*AUTH_JSON).expect("read the malformed store");
    assert_eq!(content, "{invalid-json");
}
