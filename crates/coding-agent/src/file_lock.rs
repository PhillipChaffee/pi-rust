//! The file-lock strategy, upstream's `proper-lockfile` dependency restated.
//!
//! Three stores share the dependency — `core/auth-storage.ts`, `core/trust-manager.ts`,
//! and `core/settings-manager.ts` — and their call sites split into two shapes:
//! a synchronous acquire behind a fixed retry loop (upstream's
//! `acquireLockSyncWithRetry` copies) and an asynchronous single-attempt acquire
//! with stale-lock recovery and a compromise hook (upstream's
//! `lockfile.lock(..., { retries: 0, stale, onCompromised })`).
//!
//! The restated lock is a `mkdir` on `<target>.lock` (the dependency's default
//! `lockfilePath`), created `0o700`; releasing removes the directory. An
//! `EEXIST` is the dependency's `ELOCKED`. Stale recovery follows the
//! dependency's rule: a lock directory whose mtime is older than the stale
//! window is removed and the acquire retried, so a crashed holder does not
//! wedge the store; the mtime is read through `symlink_metadata`, which does
//! not follow the directory entry itself.
//!
//! Upstream's `onCompromised` fires when the dependency discovers the lock
//! directory gone while it was believed held — at release, or at the periodic
//! verification its `update` machinery performs. The port detects the same
//! condition eagerly: [`FileLockGuard::check`] stats the directory so the
//! auth store's between-steps checks observe the compromise at the same points
//! upstream's `throwIfCompromised` calls run, and [`FileLockGuard::release`]
//! reports a vanished directory instead of failing silently.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use pi_ai::types::BoxedFuture;
use pi_ai::utils::abort::AbortError;
use tokio_util::sync::CancellationToken;

/// The sync retry budget, upstream's `maxAttempts`/`delayMs` copies.
const SYNC_MAX_ATTEMPTS: usize = 10;
const SYNC_RETRY_DELAY: Duration = Duration::from_millis(20);

/// Why an acquire failed, upstream's thrown `proper-lockfile` errors.
#[derive(Debug)]
pub enum LockError {
    /// The lock directory exists — the dependency's `ELOCKED`.
    Locked,
    /// Any other filesystem failure, the raw error carried.
    Io(std::io::Error),
    /// The lock directory vanished while held — the dependency's
    /// compromise condition.
    Compromised,
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Locked => write!(f, "locked"),
            Self::Io(error) => write!(f, "{error}"),
            Self::Compromised => write!(f, "lock was compromised"),
        }
    }
}

impl std::error::Error for LockError {}

impl LockError {
    /// Whether the failure is the retryable held-lock case, upstream's
    /// `code === "ELOCKED"` checks.
    #[must_use]
    pub const fn is_locked(&self) -> bool {
        matches!(self, Self::Locked)
    }
}

/// The compromise callback, upstream's `onCompromised` option.
pub type OnCompromised = Arc<dyn Fn(&LockError) + Send + Sync>;

/// An acquired lock directory; dropping without releasing leaks the lock, so
/// callers release explicitly the way upstream's `release()` is always
/// reached through `finally`.
#[derive(Debug)]
pub struct FileLockGuard {
    lock_dir: PathBuf,
}

impl FileLockGuard {
    /// The lock directory this guard owns.
    #[must_use]
    pub fn lock_dir(&self) -> &Path {
        &self.lock_dir
    }

    /// Whether the lock directory still exists, the between-steps ownership
    /// check the auth store runs where upstream's `throwIfCompromised` reads
    /// the hook flag.
    ///
    /// # Errors
    /// [`LockError::Compromised`] when the directory is gone.
    pub fn check(&self) -> Result<(), LockError> {
        if self.lock_dir.exists() {
            Ok(())
        } else {
            Err(LockError::Compromised)
        }
    }

    /// Remove the lock directory, upstream's `release()`.
    ///
    /// # Errors
    /// [`LockError::Compromised`] when the directory is already gone (the
    /// caller invokes the compromise hook itself), [`LockError::Io`] when the
    /// removal fails for another reason.
    pub fn release(self) -> Result<(), LockError> {
        match std::fs::remove_dir(&self.lock_dir) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(LockError::Compromised)
            }
            Err(error) => Err(LockError::Io(error)),
        }
    }
}

/// The lock directory for a target, upstream's `${path}.lock` default and the
/// trust store's explicit `lockfilePath` override — plain concatenation both.
#[must_use]
pub fn lock_dir_for(target: &str) -> PathBuf {
    PathBuf::from(format!("{target}.lock"))
}

/// One acquire attempt: `mkdir` the lock directory, honoring the stale
/// window, upstream's `lockfile.lock`/`lockSync` core.
///
/// An existing directory is [`LockError::Locked`], unless it is older than
/// `stale_ms`, in which case it is removed and the `mkdir` retried once — the
/// crashed-holder recovery the `stale` option buys.
///
/// # Errors
/// [`LockError::Locked`] when the directory is held and fresh, [`LockError::Io`]
/// for other filesystem failures.
pub fn acquire_once(lock_dir: &Path, stale_ms: Option<u64>) -> Result<FileLockGuard, LockError> {
    for _ in 0..2 {
        match std::fs::create_dir(lock_dir) {
            Ok(()) => {
                return Ok(FileLockGuard {
                    lock_dir: lock_dir.to_path_buf(),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let stale = stale_ms.is_some_and(|window| {
                    let mtime = std::fs::symlink_metadata(lock_dir)
                        .and_then(|meta| meta.modified())
                        .ok();
                    mtime.is_some_and(|mtime| {
                        SystemTime::now()
                            .duration_since(mtime)
                            .is_ok_and(|age| age >= Duration::from_millis(window))
                    })
                });
                if !stale {
                    return Err(LockError::Locked);
                }
                let _ = std::fs::remove_dir_all(lock_dir);
            }
            Err(error) => return Err(LockError::Io(error)),
        }
    }
    Err(LockError::Locked)
}

/// Acquire behind the fixed sync retry loop, upstream's three
/// `acquireLockSyncWithRetry` copies.
///
/// Ten attempts, twenty milliseconds between, only the held-lock case
/// retries, and every other error — or the last held-lock error on the tenth
/// attempt — propagates as-is.
///
/// # Errors
/// The final [`LockError`] after the budget is spent.
pub fn acquire_sync_retrying(lock_dir: &Path) -> Result<FileLockGuard, LockError> {
    let mut last: Option<LockError> = None;
    for attempt in 0..SYNC_MAX_ATTEMPTS {
        match acquire_once(lock_dir, None) {
            Ok(guard) => return Ok(guard),
            Err(error) if error.is_locked() && attempt + 1 < SYNC_MAX_ATTEMPTS => {
                last = Some(error);
                std::thread::sleep(SYNC_RETRY_DELAY);
            }
            Err(error) => return Err(error),
        }
    }
    Err(last.unwrap_or(LockError::Locked))
}

/// The options for an asynchronous acquire, upstream's
/// `{ signal, onCompromised }` plumbing around `lockfile.lock`.
#[derive(Clone)]
pub struct AsyncLockOptions {
    /// Cancellation while the lock is contended, upstream's `signal` — an
    /// aborted signal rejects without acquiring and releases a just-acquired
    /// lock before returning.
    pub signal: Option<CancellationToken>,
    /// The compromise hook, upstream's `onCompromised`.
    pub on_compromised: Option<OnCompromised>,
}

impl std::fmt::Debug for AsyncLockOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncLockOptions")
            .field("signal", &self.signal)
            .field(
                "on_compromised",
                &self.on_compromised.as_ref().map(|_| "()"),
            )
            .finish()
    }
}

/// The asynchronous acquire, upstream's `lockfile.lock` call shape: one
/// attempt with stale recovery, the signal checked before, during, and after.
///
/// The retry-with-backoff loop that upstream wraps around this acquire is the
/// auth store's, not the dependency's — it lives with [`crate::auth_storage`].
///
/// # Errors
/// [`LockError::Locked`] when the directory is held and fresh, [`LockError::Io`]
/// for other filesystem failures, or the abort failure once the signal fires.
pub fn acquire(
    lock_dir: &Path,
    options: &AsyncLockOptions,
) -> Result<FileLockGuard, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(signal) = &options.signal
        && signal.is_cancelled()
    {
        return Err(Box::new(AbortError));
    }
    let guard = match acquire_once(lock_dir, Some(30_000)) {
        Ok(guard) => guard,
        Err(error) => {
            if let Some(signal) = &options.signal
                && signal.is_cancelled()
            {
                return Err(Box::new(AbortError));
            }
            return Err(Box::new(error));
        }
    };
    if let Some(signal) = &options.signal
        && signal.is_cancelled()
    {
        let _ = guard.release();
        return Err(Box::new(AbortError));
    }
    Ok(guard)
}

/// Release through the compromise hook, the shared tail of every async
/// release site.
///
/// The hook fires when the release reports the lock vanished, and other
/// release failures are swallowed exactly where upstream wraps the release
/// in a bare catch.
pub fn release_with_hook(guard: FileLockGuard, options: &AsyncLockOptions) {
    if matches!(guard.release(), Err(LockError::Compromised))
        && let Some(hook) = &options.on_compromised
    {
        hook(&LockError::Compromised);
    }
}

/// The seam the auth store's async lock sites hang on, upstream's
/// `vi.spyOn(lockfile, "lock")` surface.
///
/// The tests inject doubles that fail, gate, or fire the compromise hook in
/// place of the real directory lock.
pub trait FileLock: Send + Sync {
    /// One asynchronous acquire on `lock_dir`.
    fn lock<'a>(
        &'a self,
        lock_dir: &'a Path,
        options: &'a AsyncLockOptions,
    ) -> BoxedFuture<'a, Result<FileLockGuard, Box<dyn std::error::Error + Send + Sync>>>;
}

/// The real directory lock, the default [`FileLock`].
#[derive(Debug, Default)]
pub struct MkdirLock;

impl FileLock for MkdirLock {
    fn lock<'a>(
        &'a self,
        lock_dir: &'a Path,
        options: &'a AsyncLockOptions,
    ) -> BoxedFuture<'a, Result<FileLockGuard, Box<dyn std::error::Error + Send + Sync>>> {
        Box::pin(async move { acquire(lock_dir, options) })
    }
}
