//! Credential storage backed by auth.json, upstream's
//! `packages/coding-agent/src/core/auth-storage.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Provider auth orchestration belongs to ModelRuntime and pi-ai Models.
//!
//! Porting restatements this module records:
//!
//! - The stored data is a provider-id keyed record of raw JSON values
//!   (upstream's `AuthStorageData`), so a written store round-trips without
//!   revalidation — the upstream cast's semantics. A typed read converts one
//!   entry through pi-ai's [`Credential`] and reports a malformed entry as
//!   the read-only store's validation message, where the upstream cast
//!   deferred to the type system.
//! - The credential's unknown api-key fields upstream's spread preserves do
//!   not survive a Rust round-trip: serde drops fields outside
//!   [`ApiKeyCredential`], whose type
//!   upstream carries no index signature for either.
//! - The shared read state keyed on the file revision restates as one
//!   process-wide slot: the first file-backed store takes it, later stores
//!   on the same path share its snapshot and coalesced reload, stores on a
//!   different path get a fresh snapshot without disturbing the slot.
//! - The coalesced reload restates on a reader-counted single-flight slot
//!   instead of a detached promise: the first reader spawns the reload task,
//!   later readers wait on the slot's result, and the reload cancels when
//!   the last reader departs, upstream's `readers === 0 →
//!   controller.abort()`. The task settles the slot whatever the reload
//!   did, upstream's `void reload.promise.then(clear, clear)`.
//! - `proper-lockfile` restates on [`crate::file_lock`]: the sync `withLock`
//!   acquires behind the ten-attempt retry loop, the async `withLockAsync`
//!   behind the thirty-second deadline with exponential backoff and jitter,
//!   and the compromise hook fires from the release path the same way the
//!   dependency reports a vanished lock. The lock seam ([`FileLock`]) stands
//!   in for the tests' `vi.spyOn(lockfile, "lock")`.
//! - The environment the `$VAR`/`!cmd` key resolution reads is injectable
//!   ([`EnvLookup`]); the plain constructors read the process environment.

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::types::{
    ApiKeyCredential, AuthError, AuthOptions, Credential, CredentialInfo, CredentialModifyFn,
};
use pi_ai::types::BoxedFuture;
use pi_ai::utils::abort::AbortError;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::config::EnvLookup;
use crate::file_lock::{
    AsyncLockOptions, FileLock, FileLockGuard, LockError, MkdirLock, OnCompromised,
    acquire_sync_retrying, lock_dir_for, release_with_hook,
};
use crate::resolve_config_value::{is_command_config_value, resolve_config_value_with};
use crate::utils::abort::race_with_abort_signal;
use crate::utils::paths::{PathInputOptions, get_file_revision, normalize_path};
use crate::utils::text::strip_bom;

/// The contended-acquire deadline, upstream's `staleMs`.
const CONTENDED_DEADLINE: Duration = Duration::from_secs(30);
/// The contended-acquire backoff base cap in milliseconds, upstream's
/// `maxDelayMs / 2` bound.
const CONTENDED_MAX_DELAY_HALF_MS: u64 = 1_000;

/// One store failure, carrying the upstream message verbatim.
#[derive(Debug)]
struct StorageError(String);

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StorageError {}

/// The store's error shape, boxed at the auth error type.
fn auth_error(message: impl Into<String>) -> AuthError {
    Box::new(StorageError(message.into()))
}

/// The abort failure every early-exit reports, boxed at the auth error type.
fn aborted() -> AuthError {
    Box::new(AbortError)
}

/// Whether the signal has fired, the `signal?.throwIfAborted()` checks.
fn check_signal(signal: Option<&CancellationToken>) -> Result<(), AuthError> {
    if signal.is_some_and(CancellationToken::is_cancelled) {
        return Err(aborted());
    }
    Ok(())
}

/// The stored data shape, upstream's `AuthStorageData`: a provider-id keyed
/// record. Raw values, so a written store round-trips untouched.
pub type AuthStorageData = serde_json::Map<String, Value>;

/// Write the store file, upstream's `AUTH_FILE_WRITE_OPTIONS` writes: the
/// callers format the content, and `0o600` applies on creation only so
/// administrator-managed modes and ACLs remain intact.
fn write_auth_file(path: &str, content: &str) -> Result<(), AuthError> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .map_err(|error| auth_error(error.to_string()))?;
    file.write_all(content.as_bytes())
        .map_err(|error| auth_error(error.to_string()))
}

/// Create the parent directory, upstream's `ensureParentDir`: recursive with
/// mode `0o700` on the directories it creates. `std::fs::create_dir_all`
/// cannot carry the mode, so the recursion is explicit — the builder's mode
/// applies to every directory it creates.
fn ensure_parent_dir(path: &str) -> Result<(), AuthError> {
    let Some(parent) = Path::new(path).parent() else {
        return Ok(());
    };
    if parent.exists() {
        return Ok(());
    }
    mkdir_all_0700(parent).map_err(|error| auth_error(error.to_string()))
}

/// The recursive component-wise `mkdirSync(dir, { recursive: true,
/// mode: 0o700 })`: the builder's mode applies to every directory it creates.
fn mkdir_all_0700(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    if dir.is_dir() {
        return Ok(());
    }
    if let Some(parent) = dir.parent() {
        mkdir_all_0700(parent)?;
    }
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(dir)
        .or_else(|error| if dir.is_dir() { Ok(()) } else { Err(error) })
}

/// Create the store file when missing, upstream's `ensureFileExists`.
fn ensure_file_exists(path: &str) -> Result<(), AuthError> {
    if !Path::new(path).exists() {
        write_auth_file(path, "{}")?;
    }
    Ok(())
}

/// Parse the stored content, upstream's `parseStorageData`: missing content
/// is the empty record, anything else parses through the BOM strip. A parse
/// failure carries serde's message where upstream carried V8's.
fn parse_storage_data(content: Option<&str>) -> Result<AuthStorageData, AuthError> {
    let Some(content) = content else {
        return Ok(AuthStorageData::new());
    };
    if content.is_empty() {
        return Ok(AuthStorageData::new());
    }
    serde_json::from_str(strip_bom(content))
        .map_err(|error| auth_error(format!("Failed to read auth.json: {error}")))
}

/// Serialize the data the way upstream writes it: two-space pretty JSON.
fn serialize_storage_data(data: &AuthStorageData) -> String {
    serde_json::to_string_pretty(data).unwrap_or_else(|_| "{}".to_string())
}

/// Convert one stored entry to its credential, the typed read the upstream
/// cast left to the type system. A malformed entry reports the read-only
/// store's validation message; a stored `null` reads as no credential, the
/// falsy entry upstream's lookups produce.
fn entry_to_credential(provider_id: &str, value: &Value) -> Result<Option<Credential>, AuthError> {
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|_| {
            auth_error(format!(
                "Invalid auth.json credential for provider \"{provider_id}\""
            ))
        })
}

/// One credential entry serialized for storage.
fn credential_to_value(credential: &Credential) -> Value {
    serde_json::to_value(credential).unwrap_or(Value::Null)
}

/// Copy the borrowed content into the owned shape the boxed futures carry.
fn owned_content(content: Option<&str>) -> Option<String> {
    content.map(str::to_string)
}

// =============================================================================
// Shared file read state
// =============================================================================

/// One in-flight coalesced reload, upstream's `AuthFileReload`: the
/// cancellation the last departing reader arms, the reader count, the
/// settled result, and the waiters' notification.
struct ReloadSlot {
    cancel: CancellationToken,
    readers: std::sync::atomic::AtomicUsize,
    result: Mutex<Option<Result<AuthStorageData, String>>>,
    ready: tokio::sync::Notify,
}

impl ReloadSlot {
    /// The settled result, waiting for the reload task to publish it.
    async fn settle(&self) -> Result<AuthStorageData, String> {
        loop {
            let notified = self.ready.notified();
            tokio::pin!(notified);
            let settled = self
                .result
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            if let Some(settled) = settled {
                return settled;
            }
            notified.await;
        }
    }

    /// Depart one reader, upstream's finally: the last reader clears the
    /// current slot and arms the reload's cancellation.
    fn depart(slot: &Arc<Self>, read_state: &Mutex<ReadState>) {
        if slot
            .readers
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel)
            == 1
        {
            let should_clear = {
                let mut state = read_state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let current_matches = state
                    .reload
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, slot));
                if current_matches {
                    state.reload = None;
                }
                current_matches
            };
            if should_clear {
                slot.cancel.cancel();
            }
        }
    }
}

/// The per-path snapshot, upstream's `AuthFileReadState`.
#[derive(Default)]
struct ReadState {
    data: AuthStorageData,
    revision: Option<String>,
    reload: Option<Arc<ReloadSlot>>,
}

/// The process-wide slot's shape: the path it was taken under and the shared
/// snapshot.
type SharedReadStateSlot = Option<(String, Arc<Mutex<ReadState>>)>;

/// The one process-wide slot, upstream's `sharedAuthFileReadState`: the
/// first file-backed store takes it, same-path stores share its snapshot.
static SHARED_AUTH_FILE_READ_STATE: LazyLock<Mutex<SharedReadStateSlot>> =
    LazyLock::new(|| Mutex::new(None));

fn shared_read_state_for(auth_path: &str) -> Arc<Mutex<ReadState>> {
    let mut shared = SHARED_AUTH_FILE_READ_STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((taken_path, state)) = shared.as_ref() {
        if taken_path == auth_path {
            return Arc::clone(state);
        }
        return Arc::new(Mutex::new(ReadState::default()));
    }
    let state = Arc::new(Mutex::new(ReadState::default()));
    *shared = Some((auth_path.to_string(), Arc::clone(&state)));
    state
}

// =============================================================================
// Backends
// =============================================================================

/// The lock outcome, upstream's `LockResult<T>`: the operation's result and
/// the optional next file content.
#[derive(Debug)]
pub struct LockOutcome<T> {
    /// The operation's result, handed back to the caller.
    pub result: T,
    /// The next file content to write; `None` leaves the file untouched.
    pub next: Option<String>,
}

/// The storage surface behind an [`AuthStorage`], upstream's
/// `AuthStorageBackend`.
pub trait AuthStorageBackend: Send + Sync + 'static {
    /// The synchronous locked operation, upstream's `withLock`. `current` is
    /// the file content when present.
    ///
    /// # Errors
    /// A lock-acquisition, read, or write failure.
    fn with_lock<T>(
        &self,
        f: impl FnOnce(Option<&str>) -> LockOutcome<T> + Send,
    ) -> Result<T, AuthError>;

    /// The asynchronous locked operation, upstream's `withLockAsync`.
    ///
    /// # Errors
    /// A lock-acquisition, read, or write failure; an already-aborted or
    /// aborted-during-operation signal; or a compromised lock.
    fn with_lock_async<T>(
        &self,
        f: impl FnOnce(Option<&str>) -> BoxedFuture<'static, Result<LockOutcome<T>, AuthError>>
        + Send
        + 'static,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'static, Result<T, AuthError>>;
}

/// The contended acquire loop, upstream's `acquireLockAsync`: a thirty-second
/// deadline, exponential backoff with jitter on held-lock errors, the signal
/// checked before, between, and after every wait, and a just-acquired lock
/// released when the signal fires.
async fn acquire_contended(
    lock: &dyn FileLock,
    lock_dir: &Path,
    signal: Option<&CancellationToken>,
    on_compromised: Option<OnCompromised>,
) -> Result<FileLockGuard, AuthError> {
    let deadline = Instant::now() + CONTENDED_DEADLINE;
    let mut retry: u32 = 0;
    let options = AsyncLockOptions {
        signal: signal.cloned(),
        on_compromised,
    };
    loop {
        check_signal(signal)?;
        match lock.lock(lock_dir, &options).await {
            Ok(guard) => {
                if let Some(signal) = signal
                    && signal.is_cancelled()
                {
                    guard
                        .release()
                        .map_err(|error| auth_error(error.to_string()))?;
                    return Err(aborted());
                }
                return Ok(guard);
            }
            Err(error) => {
                check_signal(signal)?;
                let locked = error
                    .downcast_ref::<LockError>()
                    .is_some_and(LockError::is_locked);
                let remaining = deadline.saturating_duration_since(Instant::now());
                if !locked || remaining.is_zero() {
                    return Err(error);
                }
                let base_delay_ms =
                    std::cmp::min(10 * 2u64.saturating_pow(retry), CONTENDED_MAX_DELAY_HALF_MS);
                retry += 1;
                #[expect(
                    clippy::cast_precision_loss,
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the backoff math rides upstream's JS numbers: the base is at most 1000 and the product rounds into u64 range, and remaining is a positive duration"
                )]
                let jittered =
                    ((base_delay_ms as f64) * (1.0 + rand::random::<f64>())).round() as u64;
                let delay_ms = remaining.as_millis().min(u128::from(u64::MAX));
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "remaining is clamped into the u64 range immediately above"
                )]
                let delay = std::cmp::min(jittered, delay_ms as u64);
                match signal {
                    Some(signal) => {
                        tokio::select! {
                            () = signal.cancelled() => return Err(aborted()),
                            () = tokio::time::sleep(Duration::from_millis(delay)) => {}
                        }
                    }
                    None => tokio::time::sleep(Duration::from_millis(delay)).await,
                }
            }
        }
    }
}

/// The auth.json file backend, upstream's `FileAuthStorageBackend`.
pub struct FileAuthStorageBackend {
    auth_path: String,
    lock: Arc<dyn FileLock>,
}

impl std::fmt::Debug for FileAuthStorageBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileAuthStorageBackend")
            .field("auth_path", &self.auth_path)
            .finish_non_exhaustive()
    }
}

impl FileAuthStorageBackend {
    /// The backend over `auth_path`, upstream's constructor with its default
    /// path normalized.
    ///
    /// # Errors
    /// A `file://` path that does not convert to a local path.
    pub fn new(auth_path: &str) -> Result<Self, AuthError> {
        let normalized = normalize_path(auth_path, &PathInputOptions::default())
            .map_err(|error| auth_error(error.to_string()))?;
        Ok(Self {
            auth_path: normalized,
            lock: Arc::new(MkdirLock),
        })
    }

    /// The backend over a replaced lock strategy, the test seam standing in
    /// for upstream's `vi.spyOn(lockfile, "lock")`.
    #[must_use]
    pub fn with_lock_strategy(auth_path: &str, lock: Arc<dyn FileLock>) -> Self {
        Self {
            auth_path: auth_path.to_string(),
            lock,
        }
    }

    fn lock_dir(&self) -> PathBuf {
        lock_dir_for(&self.auth_path)
    }
}

impl AuthStorageBackend for FileAuthStorageBackend {
    fn with_lock<T>(
        &self,
        f: impl FnOnce(Option<&str>) -> LockOutcome<T> + Send,
    ) -> Result<T, AuthError> {
        ensure_parent_dir(&self.auth_path)?;
        ensure_file_exists(&self.auth_path)?;

        let guard = acquire_sync_retrying(&self.lock_dir())
            .map_err(|error| auth_error(error.to_string()))?;
        // Upstream's finally releases on every path, so the body computes
        // without early returns and the release tail always runs.
        let outcome: Result<T, AuthError> = (|| {
            let current = std::fs::read_to_string(&self.auth_path).ok();
            let outcome = f(current.as_deref());
            if let Some(next) = &outcome.next {
                write_auth_file(&self.auth_path, next)?;
            }
            Ok(outcome.result)
        })();
        // Upstream's finally: the release runs whatever happened, and its
        // error replaces the outcome the way a finally throw would.
        match guard.release() {
            Ok(()) => outcome,
            Err(error) => Err(auth_error(error.to_string())),
        }
    }

    fn with_lock_async<T>(
        &self,
        f: impl FnOnce(Option<&str>) -> BoxedFuture<'static, Result<LockOutcome<T>, AuthError>>
        + Send
        + 'static,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'static, Result<T, AuthError>> {
        let signal = options.and_then(|options| options.signal.clone());
        let auth_path = self.auth_path.clone();
        let lock_dir = self.lock_dir();
        let lock = Arc::clone(&self.lock);

        Box::pin(async move {
            check_signal(signal.as_ref())?;
            ensure_parent_dir(&auth_path)?;
            ensure_file_exists(&auth_path)?;

            // Upstream's onCompromised plumbing: the hook records the first
            // compromise and every throwIfCompromised site surfaces it.
            let compromised: Arc<Mutex<Option<AuthError>>> = Arc::new(Mutex::new(None));
            let hook: OnCompromised = {
                let flag = Arc::clone(&compromised);
                Arc::new(move |_error: &LockError| {
                    let mut slot = flag
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if slot.is_none() {
                        *slot = Some(auth_error("Auth storage lock was compromised"));
                    }
                })
            };
            let throw_if_compromised = || -> Result<(), AuthError> {
                let taken = compromised
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                if let Some(error) = taken {
                    return Err(error);
                }
                Ok(())
            };

            let guard = acquire_contended(&*lock, &lock_dir, signal.as_ref(), Some(hook)).await?;
            // Upstream's finally releases on every path, so the body computes
            // without early returns and the release tail always runs.
            let outcome: Result<T, AuthError> = async {
                throw_if_compromised()?;
                check_signal(signal.as_ref())?;
                let current = std::fs::read_to_string(&auth_path).ok();
                let outcome = f(current.as_deref()).await?;
                throw_if_compromised()?;
                check_signal(signal.as_ref())?;
                if let Some(next) = &outcome.next {
                    write_auth_file(&auth_path, next)?;
                }
                throw_if_compromised()?;
                Ok(outcome.result)
            }
            .await;
            // Upstream's finally: the release runs whatever happened, its
            // errors swallowed, the compromise hook still firing.
            release_with_hook(
                guard,
                &AsyncLockOptions {
                    signal: signal.clone(),
                    on_compromised: None,
                },
            );
            outcome
        })
    }
}

/// The in-memory backend's shared state: the value and the operation queue,
/// held behind an `Arc` so the boxed futures own their inputs.
#[derive(Debug, Default)]
struct InMemoryInner {
    value: Mutex<Option<String>>,
    chain: tokio::sync::Mutex<()>,
}

/// The in-memory backend, upstream's `InMemoryAuthStorageBackend`: the value
/// is the serialized form, async operations serialize through one queue.
#[derive(Debug, Default)]
pub struct InMemoryAuthStorageBackend {
    inner: Arc<InMemoryInner>,
}

impl InMemoryAuthStorageBackend {
    /// Seed the value with serialized content, upstream's `withLock` seeding.
    fn seed(&self, next: String) {
        *self
            .inner
            .value
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(next);
    }
}

impl AuthStorageBackend for InMemoryAuthStorageBackend {
    fn with_lock<T>(
        &self,
        f: impl FnOnce(Option<&str>) -> LockOutcome<T> + Send,
    ) -> Result<T, AuthError> {
        let current = self
            .inner
            .value
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let outcome = f(current.as_deref());
        if let Some(next) = outcome.next {
            *self
                .inner
                .value
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(next);
        }
        Ok(outcome.result)
    }

    fn with_lock_async<T>(
        &self,
        f: impl FnOnce(Option<&str>) -> BoxedFuture<'static, Result<LockOutcome<T>, AuthError>>
        + Send
        + 'static,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'static, Result<T, AuthError>> {
        let signal = options.and_then(|options| options.signal.clone());
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            // Upstream wraps the queued operation in raceWithAbortSignal: an
            // already-aborted signal rejects without queueing.
            check_signal(signal.as_ref())?;
            let race = race_with_abort_signal(
                async {
                    let _queued = inner.chain.lock().await;
                    // The queued operation's own signal check runs at the
                    // front of the queue, so a cancelled mutation never runs.
                    check_signal(signal.as_ref())?;
                    let current = inner
                        .value
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone();
                    let outcome = f(current.as_deref()).await?;
                    check_signal(signal.as_ref())?;
                    if let Some(next) = outcome.next {
                        *inner
                            .value
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(next);
                    }
                    Ok::<T, AuthError>(outcome.result)
                },
                signal.as_ref(),
            )
            .await;
            match race {
                Ok(result) => Ok(result),
                Err(pi_ai::utils::abort::RaceError::Aborted(_)) => Err(aborted()),
                Err(pi_ai::utils::abort::RaceError::Operation(error)) => Err(error),
            }
        })
    }
}

// =============================================================================
// AuthStorage
// =============================================================================

/// Credential storage backed by a JSON file, upstream's `AuthStorage`.
///
/// The write path is [`CredentialStore::modify`] and
/// [`CredentialStore::delete`]; reads serve the latest known snapshot,
/// reloading through a coalesced in-flight reload when the file revision
/// moved.
pub struct AuthStorage<B: AuthStorageBackend = FileAuthStorageBackend> {
    storage: Arc<B>,
    auth_path: Option<String>,
    env: EnvLookup,
    read_state: Arc<Mutex<ReadState>>,
}

impl<B: AuthStorageBackend> std::fmt::Debug for AuthStorage<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthStorage")
            .field("auth_path", &self.auth_path)
            .finish_non_exhaustive()
    }
}

impl AuthStorage<FileAuthStorageBackend> {
    /// The store over `auth_path`, upstream's `create`: the normalized path's
    /// file backend, the shared read state keyed on the path.
    ///
    /// # Errors
    /// A `file://` path that does not convert to a local path.
    pub fn create(auth_path: &str) -> Result<Self, AuthError> {
        Self::create_with_env(auth_path, crate::config::default_env_lookup())
    }

    /// [`create`](Self::create) over an injected environment lookup, the
    /// `$VAR`/`!cmd` resolution seam.
    ///
    /// # Errors
    /// A `file://` path that does not convert to a local path.
    pub fn create_with_env(auth_path: &str, env: EnvLookup) -> Result<Self, AuthError> {
        let normalized = normalize_path(auth_path, &PathInputOptions::default())
            .map_err(|error| auth_error(error.to_string()))?;
        let storage = FileAuthStorageBackend::new(&normalized)?;
        let read_state = shared_read_state_for(&normalized);
        let store = Self {
            storage: Arc::new(storage),
            auth_path: Some(normalized),
            env,
            read_state,
        };
        store.reload_unless_revision_matches();
        Ok(store)
    }
}

impl<B: AuthStorageBackend> AuthStorage<B> {
    /// The store over an arbitrary backend, upstream's `fromStorage`: no file
    /// path, so reads reload through the backend and never short-circuit.
    #[must_use]
    pub fn from_storage(storage: B) -> Self {
        Self::from_storage_with_env(storage, None, crate::config::default_env_lookup())
    }

    /// [`from_storage`](Self::from_storage) over an injected environment
    /// lookup.
    #[must_use]
    pub fn from_storage_with_env(storage: B, auth_path: Option<String>, env: EnvLookup) -> Self {
        let read_state = auth_path
            .as_deref()
            .map(shared_read_state_for)
            .unwrap_or_default();
        let store = Self {
            storage: Arc::new(storage),
            auth_path,
            env,
            read_state,
        };
        store.reload_unless_revision_matches();
        store
    }

    /// The in-memory store seeded with `data`, upstream's `inMemory`.
    #[must_use]
    pub fn in_memory(data: &AuthStorageData) -> AuthStorage<InMemoryAuthStorageBackend> {
        let storage = InMemoryAuthStorageBackend::default();
        let seeded = serde_json::to_string_pretty(data).unwrap_or_else(|_| "{}".to_string());
        storage.seed(seeded);
        AuthStorage::from_storage(storage)
    }

    /// The constructor tail, upstream's revision check and unconditional
    /// reload: a fresh snapshot reloads through the sync lock, a matching
    /// revision keeps the shared one.
    fn reload_unless_revision_matches(&self) {
        let Some(auth_path) = &self.auth_path else {
            self.reload();
            return;
        };
        let revision = get_file_revision(auth_path);
        let matches = revision.is_some() && {
            let state = self
                .read_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            revision == state.revision
        };
        if !matches {
            self.reload();
        }
    }

    fn update_read_state(&self, data: AuthStorageData, revision: Option<String>) {
        let mut state = self
            .read_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.data = data;
        state.revision = revision;
    }

    /// Reload credentials from storage, upstream's `reload`: every failure
    /// preserves the last valid in-memory snapshot.
    pub fn reload(&self) {
        let mut content: Option<String> = None;
        let mut revision: Option<String> = None;
        let outcome = self.storage.with_lock(|current| {
            content = owned_content(current);
            revision = self.auth_path.as_deref().and_then(get_file_revision);
            LockOutcome {
                result: (),
                next: None,
            }
        });
        if matches!(outcome, Ok(()))
            && let Ok(data) = parse_storage_data(content.as_deref())
        {
            self.update_read_state(data, revision);
        }
        // Preserve the last valid in-memory snapshot.
    }

    async fn reload_from_storage_async(
        &self,
        options: Option<&AuthOptions>,
    ) -> Result<AuthStorageData, AuthError> {
        let read_state = Arc::clone(&self.read_state);
        let auth_path = self.auth_path.clone();
        self.storage
            .with_lock_async(
                move |content| {
                    let content = owned_content(content);
                    let read_state = Arc::clone(&read_state);
                    Box::pin(async move {
                        let current_data = parse_storage_data(content.as_deref())?;
                        let revision = auth_path.as_deref().and_then(get_file_revision);
                        {
                            let mut state = read_state
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            state.data.clone_from(&current_data);
                            state.revision = revision;
                        }
                        Ok(LockOutcome {
                            result: current_data,
                            next: None,
                        })
                    })
                },
                options,
            )
            .await
    }

    /// The latest data, upstream's `readLatestData`: the in-memory backend
    /// reloads directly, the file backend short-circuits on a matching
    /// revision and otherwise joins the coalesced reload slot.
    async fn read_latest_data(
        &self,
        options: Option<&AuthOptions>,
    ) -> Result<AuthStorageData, AuthError> {
        let signal = options.and_then(|options| options.signal.as_ref());
        check_signal(signal)?;

        let Some(auth_path) = &self.auth_path else {
            let reload = self.reload_from_storage_async(options);
            return match reload.await {
                Ok(data) => Ok(data),
                // Upstream: without a signal, a failed reload resolves to the
                // current snapshot; with one, the failure propagates.
                Err(_) if signal.is_none() => Ok(self
                    .read_state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .data
                    .clone()),
                Err(error) => Err(error),
            };
        };

        let revision = get_file_revision(auth_path);
        if revision.is_some() {
            let state = self
                .read_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if revision == state.revision {
                return Ok(state.data.clone());
            }
        }

        // Join or start the coalesced reload, upstream's readLatestData slot
        // machinery.
        let slot = {
            let mut state = self
                .read_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.reload.is_none() {
                let slot = Arc::new(ReloadSlot {
                    cancel: CancellationToken::new(),
                    readers: std::sync::atomic::AtomicUsize::new(0),
                    result: Mutex::new(None),
                    ready: tokio::sync::Notify::new(),
                });
                state.reload = Some(Arc::clone(&slot));
                let job = ReloadJob {
                    storage: Arc::clone(&self.storage),
                    read_state: Arc::clone(&self.read_state),
                    auth_path: auth_path.clone(),
                    slot: Arc::clone(&slot),
                };
                tokio::spawn(job.run());
            }
            let Some(slot) = state.reload.as_ref() else {
                return Err(auth_error("reload slot vanished"));
            };
            let joined = Arc::clone(slot);
            joined
                .readers
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            drop(state);
            joined
        };

        // Upstream races the abort against the reload promise
        // (raceWithAbortSignal): a signalled reader rejects promptly and
        // departs, which arms the reload's cancellation when it is the last
        // reader.
        let settled = match signal {
            Some(signal) => {
                tokio::select! {
                    () = signal.cancelled() => {
                        ReloadSlot::depart(&slot, &self.read_state);
                        return Err(aborted());
                    }
                    settled = slot.settle() => settled,
                }
            }
            None => slot.settle().await,
        };
        ReloadSlot::depart(&slot, &self.read_state);

        match (signal, settled) {
            (_, Ok(data)) => Ok(data),
            (Some(_), Err(message)) => Err(auth_error(message)),
            (None, Err(_)) => Ok(self
                .read_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .data
                .clone()),
        }
    }

    /// Read one credential, upstream's `read`: api-key keys resolve through
    /// [`crate::resolve_config_value`], other credentials return as stored.
    ///
    /// # Errors
    /// A reload failure when a signal is supplied, a malformed stored entry,
    /// or an aborted signal.
    pub async fn read(
        &self,
        provider: &str,
        options: Option<&AuthOptions>,
    ) -> Result<Option<Credential>, AuthError> {
        let data = self.read_latest_data(options).await?;
        let credential = match data.get(provider) {
            Some(entry) => entry_to_credential(provider, entry)?,
            None => None,
        };
        check_signal(options.and_then(|options| options.signal.as_ref()))?;
        match credential {
            Some(Credential::ApiKey(api_key)) if api_key.key.is_some() => {
                let Some(key) = api_key.key.clone() else {
                    return Ok(Some(Credential::ApiKey(api_key)));
                };
                let resolved = resolve_config_value_with(&key, api_key.env.as_ref(), &self.env);
                Ok(Some(Credential::ApiKey(ApiKeyCredential {
                    key: resolved,
                    env: api_key.env,
                })))
            }
            other => Ok(other),
        }
    }

    /// List credential metadata, upstream's `list`: the stored types without
    /// resolving key values.
    ///
    /// # Errors
    /// A reload failure or an aborted signal.
    pub async fn list(
        &self,
        options: Option<&AuthOptions>,
    ) -> Result<Vec<CredentialInfo>, AuthError> {
        let data = self.read_latest_data(options).await?;
        check_signal(options.and_then(|options| options.signal.as_ref()))?;
        Ok(data
            .iter()
            .filter_map(|(provider_id, value)| {
                let credential: Credential = serde_json::from_value(value.clone()).ok()?;
                Some(CredentialInfo {
                    provider_id: provider_id.clone(),
                    auth_type: credential.auth_type(),
                })
            })
            .collect())
    }

    /// The serialized write path, upstream's `modify`: the caller's function
    /// sees the current stored credential; `None` leaves the entry and
    /// records the revision, a credential merges into the file and leaves
    /// the revision unset so the next read reloads.
    ///
    /// # Errors
    /// The function's failure, a storage failure, or an aborted signal — an
    /// aborted write is never applied.
    pub async fn modify(
        &self,
        provider: &str,
        f: CredentialModifyFn,
        options: Option<&AuthOptions>,
    ) -> Result<Option<Credential>, AuthError> {
        let bookkeeping = Arc::new(Mutex::new(ModifyBookkeeping {
            latest: self
                .read_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .data
                .clone(),
            revision: None,
        }));
        let provider = provider.to_string();
        let auth_path = self.auth_path.clone();
        let bookkeeping_for_closure = Arc::clone(&bookkeeping);
        let result = self
            .storage
            .with_lock_async(
                move |content| {
                    let content = owned_content(content);
                    let auth_path = auth_path.clone();
                    let bookkeeping = Arc::clone(&bookkeeping_for_closure);
                    Box::pin(async move {
                        let current_data = parse_storage_data(content.as_deref())?;
                        let current = match current_data.get(&provider) {
                            Some(value) => entry_to_credential(&provider, value)?,
                            None => None,
                        };
                        let next = f(current.clone()).await?;
                        match next {
                            None => {
                                // Upstream re-reads currentData[provider] for
                                // the result: the fn consumed its copy.
                                let result = match current_data.get(&provider) {
                                    Some(value) => entry_to_credential(&provider, value)?,
                                    None => None,
                                };
                                {
                                    let mut bookkeeping = bookkeeping
                                        .lock()
                                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                                    bookkeeping.latest.clone_from(&current_data);
                                    bookkeeping.revision =
                                        auth_path.as_deref().and_then(get_file_revision);
                                }
                                Ok(LockOutcome { result, next: None })
                            }
                            Some(next) => {
                                let mut merged = current_data;
                                merged.insert(provider.clone(), credential_to_value(&next));
                                bookkeeping
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                                    .latest
                                    .clone_from(&merged);
                                Ok(LockOutcome {
                                    result: Some(next),
                                    next: Some(serialize_storage_data(&merged)),
                                })
                            }
                        }
                    })
                },
                options,
            )
            .await;
        // Upstream: the await's failure skips the updateReadState entirely.
        if result.is_ok() {
            let bookkeeping = bookkeeping
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.update_read_state(bookkeeping.latest.clone(), bookkeeping.revision.clone());
        }
        result
    }

    /// Remove one credential, upstream's `delete`.
    ///
    /// # Errors
    /// A storage failure or an aborted signal.
    pub async fn delete(
        &self,
        provider: &str,
        options: Option<&AuthOptions>,
    ) -> Result<(), AuthError> {
        let latest = Arc::new(Mutex::new(
            self.read_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .data
                .clone(),
        ));
        let provider = provider.to_string();
        let latest_for_closure = Arc::clone(&latest);
        let result = self
            .storage
            .with_lock_async(
                move |content| {
                    let content = owned_content(content);
                    let latest = Arc::clone(&latest_for_closure);
                    Box::pin(async move {
                        let mut current_data = parse_storage_data(content.as_deref())?;
                        current_data.shift_remove(&provider);
                        latest
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .clone_from(&current_data);
                        Ok(LockOutcome {
                            result: (),
                            next: Some(serialize_storage_data(&current_data)),
                        })
                    })
                },
                options,
            )
            .await;
        // Upstream: the await's failure skips the updateReadState entirely.
        if matches!(result, Ok(())) {
            self.update_read_state(
                latest
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone(),
                None,
            );
        }
        result
    }
}

/// The write bookkeeping upstream's `latestData`/`revision` locals carry: the
/// closure records the post-operation snapshot, the caller publishes it to
/// the read state.
struct ModifyBookkeeping {
    latest: AuthStorageData,
    revision: Option<String>,
}

/// The coalesced reload's owned inputs, the spawned task carrying upstream's
/// detached reload promise: it publishes the slot's result whatever the
/// reload did and clears itself while it is still the current slot.
struct ReloadJob<B: AuthStorageBackend> {
    storage: Arc<B>,
    read_state: Arc<Mutex<ReadState>>,
    auth_path: String,
    slot: Arc<ReloadSlot>,
}

impl<B: AuthStorageBackend> ReloadJob<B> {
    async fn run(self) {
        let options = AuthOptions {
            signal: Some(self.slot.cancel.clone()),
        };
        let read_state = Arc::clone(&self.read_state);
        let auth_path = self.auth_path.clone();
        let result = self
            .storage
            .with_lock_async(
                move |content| {
                    let content = owned_content(content);
                    let read_state = Arc::clone(&read_state);
                    Box::pin(async move {
                        let current_data = parse_storage_data(content.as_deref())?;
                        let revision = get_file_revision(&auth_path);
                        {
                            let mut state = read_state
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            state.data.clone_from(&current_data);
                            state.revision = revision;
                        }
                        Ok(LockOutcome {
                            result: current_data,
                            next: None,
                        })
                    })
                },
                Some(&options),
            )
            .await;

        *self
            .slot
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(match result {
            Ok(data) => Ok(data),
            Err(error) => Err(error.to_string()),
        });
        self.slot.ready.notify_waiters();

        let mut state = self
            .read_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(current) = &state.reload
            && Arc::ptr_eq(current, &self.slot)
        {
            state.reload = None;
        }
    }
}

impl<B: AuthStorageBackend> CredentialStore for AuthStorage<B> {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(Self::read(self, provider_id, options))
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        Box::pin(Self::list(self, options))
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(Self::modify(self, provider_id, f, options))
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        Box::pin(Self::delete(self, provider_id, options))
    }
}

// =============================================================================
// ReadOnlyAuthStorage
// =============================================================================

/// The read-only view over an auth.json file, upstream's
/// `ReadOnlyAuthStorage`: loads and validates once, never writes.
pub struct ReadOnlyAuthStorage {
    auth_path: String,
    env: EnvLookup,
    data: Mutex<Option<AuthStorageData>>,
}

impl std::fmt::Debug for ReadOnlyAuthStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadOnlyAuthStorage")
            .field("auth_path", &self.auth_path)
            .finish_non_exhaustive()
    }
}

impl ReadOnlyAuthStorage {
    /// The reader over `auth_path`, upstream's constructor with its default
    /// path normalized.
    ///
    /// # Errors
    /// A `file://` path that does not convert to a local path.
    pub fn new(auth_path: &str) -> Result<Self, AuthError> {
        Self::new_with_env(auth_path, crate::config::default_env_lookup())
    }

    /// [`new`](Self::new) over an injected environment lookup.
    ///
    /// # Errors
    /// A `file://` path that does not convert to a local path.
    pub fn new_with_env(auth_path: &str, env: EnvLookup) -> Result<Self, AuthError> {
        let normalized = normalize_path(auth_path, &PathInputOptions::default())
            .map_err(|error| auth_error(error.to_string()))?;
        Ok(Self {
            auth_path: normalized,
            env,
            data: Mutex::new(None),
        })
    }

    /// Load and validate the file, upstream's `load`: missing file is the
    /// empty record; every entry must be a well-formed api-key or OAuth
    /// credential.
    fn load(&self) -> Result<AuthStorageData, AuthError> {
        let cached = self
            .data
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(data) = cached {
            return Ok(data);
        }
        let loaded = self.read_and_validate()?;
        *self
            .data
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(loaded.clone());
        Ok(loaded)
    }

    fn read_and_validate(&self) -> Result<AuthStorageData, AuthError> {
        let content = std::fs::read_to_string(&self.auth_path)
            .map_err(|error| auth_error(format!("Failed to read auth.json: {error}")))?;
        let parsed: Value = serde_json::from_str(strip_bom(&content))
            .map_err(|error| auth_error(format!("Failed to read auth.json: {error}")))?;
        let Value::Object(entries) = parsed else {
            return Err(auth_error("Invalid auth.json: expected an object"));
        };
        for (provider_id, value) in &entries {
            if serde_json::from_value::<Credential>(value.clone()).is_err() {
                return Err(auth_error(format!(
                    "Invalid auth.json credential for provider \"{provider_id}\""
                )));
            }
        }
        Ok(entries)
    }
}

impl CredentialStore for ReadOnlyAuthStorage {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(async move {
            check_signal(options.and_then(|options| options.signal.as_ref()))?;
            let data = self.load()?;
            let credential = match data.get(provider_id) {
                Some(entry) => entry_to_credential(provider_id, entry)?,
                None => None,
            };
            check_signal(options.and_then(|options| options.signal.as_ref()))?;
            match credential {
                Some(Credential::ApiKey(api_key)) if api_key.key.is_some() => {
                    let Some(key) = api_key.key.clone() else {
                        return Ok(Some(Credential::ApiKey(api_key)));
                    };
                    if is_command_config_value(&key) {
                        return Ok(Some(Credential::ApiKey(api_key)));
                    }
                    let resolved = resolve_config_value_with(&key, api_key.env.as_ref(), &self.env);
                    Ok(Some(Credential::ApiKey(ApiKeyCredential {
                        key: resolved,
                        env: api_key.env,
                    })))
                }
                other => Ok(other),
            }
        })
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        Box::pin(async move {
            check_signal(options.and_then(|options| options.signal.as_ref()))?;
            let data = self.load()?;
            check_signal(options.and_then(|options| options.signal.as_ref()))?;
            Ok(data
                .iter()
                .filter_map(|(provider_id, value)| {
                    let credential: Credential = serde_json::from_value(value.clone()).ok()?;
                    Some(CredentialInfo {
                        provider_id: provider_id.clone(),
                        auth_type: credential.auth_type(),
                    })
                })
                .collect())
        })
    }

    fn modify<'a>(
        &'a self,
        _provider_id: &'a str,
        _f: CredentialModifyFn,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(async move {
            Err(auth_error(
                "Read-only credential storage cannot modify auth.json",
            ))
        })
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        Box::pin(async move {
            Err(auth_error(
                "Read-only credential storage cannot modify auth.json",
            ))
        })
    }
}

// =============================================================================
// One-off read
// =============================================================================

/// One-off synchronous read of a stored credential from an auth.json file,
/// upstream's `readStoredCredential`.
///
/// No store is instantiated and no configured key value resolves; a missing
/// or malformed file reads as `None`.
#[must_use]
pub fn read_stored_credential(provider_id: &str, auth_path: &str) -> Option<Credential> {
    let normalized = normalize_path(auth_path, &PathInputOptions::default()).ok()?;
    let content = std::fs::read_to_string(normalized).ok()?;
    let data: AuthStorageData = serde_json::from_str(strip_bom(&content)).ok()?;
    data.get(provider_id)
        .and_then(|value| serde_json::from_value(value.clone()).ok())
}
