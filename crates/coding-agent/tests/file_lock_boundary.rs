//! Boundary tests binding the `file_lock` branches the 1:1 suites leave
//! untested, at pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759.
//!
//! The stale-window recovery runs by backdating the lock directory's mtime
//! with `File::set_modified` (futimens reaches directories) — the dependency's
//! `stale` option rides the same mtime read.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "a wrong release outcome is reported by panicking"
)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use pi_ai::utils::abort::AbortError;
use pi_coding_agent::file_lock::{
    AsyncLockOptions, FileLock, LockError, MkdirLock, OnCompromised, acquire, acquire_once,
    acquire_sync_retrying, lock_dir_for, release_with_hook,
};
use tokio_util::sync::CancellationToken;

fn temp_dir(prefix: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("temp dir")
}

/// Restore a directory's mode on drop, so a failing assertion cannot leave a
/// read-only tree behind for the temp-dir cleanup to trip over.
struct RestorePerms<'a> {
    path: &'a std::path::Path,
}

impl Drop for RestorePerms<'_> {
    fn drop(&mut self) {
        let _ = fs::set_permissions(self.path, fs::Permissions::from_mode(0o755));
    }
}

/// The compromise hook recording every fire, the assertions' view.
fn recording_hook(fired: Arc<Mutex<Vec<String>>>) -> OnCompromised {
    Arc::new(move |error: &LockError| {
        fired
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(error.to_string());
    })
}

#[test]
fn lock_error_display_strings_and_the_locked_predicate() {
    assert_eq!(LockError::Locked.to_string(), "locked");
    assert_eq!(LockError::Compromised.to_string(), "lock was compromised");
    let io = LockError::Io(std::io::Error::other("boom"));
    assert_eq!(io.to_string(), "boom");

    assert!(LockError::Locked.is_locked());
    assert!(!LockError::Compromised.is_locked());
    assert!(!LockError::Io(std::io::Error::other("x")).is_locked());
}

#[test]
fn lock_dir_for_appends_the_lock_suffix() {
    assert_eq!(
        lock_dir_for("/agent/auth.json"),
        std::path::PathBuf::from("/agent/auth.json.lock")
    );
}

#[test]
fn a_fresh_guard_checks_and_reports_a_vanished_lock() {
    let temp = temp_dir("pi-flock-check-");
    let dir = temp.path().join("target.lock");
    let guard = acquire_once(&dir, None).expect("acquire");
    assert_eq!(
        guard.lock_dir(),
        dir,
        "the guard names the directory it owns"
    );

    guard.check().expect("the lock is held");
    fs::remove_dir(&dir).expect("another process steals the lock");
    let error = guard.check().expect_err("the vanished lock must fail");
    assert!(matches!(error, LockError::Compromised));
}

#[test]
fn release_reports_a_vanished_lock_as_compromised() {
    let temp = temp_dir("pi-flock-vanished-");
    let dir = temp.path().join("target.lock");
    let guard = acquire_once(&dir, None).expect("acquire");
    fs::remove_dir(&dir).expect("the lock vanishes");

    let error = guard.release().expect_err("the vanished lock must fail");
    assert!(matches!(error, LockError::Compromised));
}

#[test]
fn release_reports_a_stray_file_as_an_io_error() {
    let temp = temp_dir("pi-flock-stray-");
    let dir = temp.path().join("target.lock");
    let guard = acquire_once(&dir, None).expect("acquire");
    fs::write(dir.join("stray"), "x").expect("a file inside the lock dir");

    let error = guard.release().expect_err("the non-empty dir must fail");
    match error {
        LockError::Io(error) => {
            assert_eq!(error.kind(), std::io::ErrorKind::DirectoryNotEmpty);
        }
        other => panic!("expected an io error, got: {other:?}"),
    }
}

#[test]
fn acquire_creates_an_owner_only_directory_and_release_removes_it() {
    let temp = temp_dir("pi-flock-mode-");
    let dir = temp.path().join("target.lock");

    let guard = acquire_once(&dir, None).expect("acquire");
    let mode = fs::metadata(&dir)
        .expect("lock metadata")
        .permissions()
        .mode();
    // The dependency's bare `fs.mkdir` leaves the mode to the process umask;
    // the only contract is that the owner keeps write access.
    assert_eq!(
        mode & 0o700,
        0o700,
        "the lock directory stays owner-writable"
    );
    guard.release().expect("release");
    assert!(!dir.exists(), "release removes the directory");
}

#[test]
fn a_fresh_lock_reports_locked() {
    let temp = temp_dir("pi-flock-fresh-");
    let dir = temp.path().join("target.lock");
    let _held = acquire_once(&dir, None).expect("the holder");

    let error = acquire_once(&dir, Some(30_000)).expect_err("the fresh lock must fail");
    assert!(matches!(error, LockError::Locked));
}

#[test]
fn acquire_once_without_a_stale_window_reports_locked() {
    let temp = temp_dir("pi-flock-nostale-");
    let dir = temp.path().join("target.lock");
    let _held = acquire_once(&dir, None).expect("the holder");

    let error = acquire_once(&dir, None).expect_err("the held lock must fail");
    assert!(matches!(error, LockError::Locked));
}

#[test]
fn a_stale_lock_is_removed_and_the_acquire_retried() {
    let temp = temp_dir("pi-flock-stale-");
    let dir = temp.path().join("target.lock");
    fs::create_dir(&dir).expect("seed the crashed holder's lock");
    {
        let handle = fs::File::open(&dir).expect("open the lock dir");
        handle
            .set_modified(SystemTime::now() - Duration::from_secs(60))
            .expect("backdate the lock mtime past the window");
    }

    let guard = acquire_once(&dir, Some(30_000)).expect("the stale lock recovers");
    assert!(dir.exists(), "the retry re-created the lock");
    guard.release().expect("release the recovered lock");
}

#[test]
fn sync_retrying_acquires_a_free_lock() {
    let temp = temp_dir("pi-flock-sync-free-");
    let dir = temp.path().join("target.lock");

    let guard = acquire_sync_retrying(&dir).expect("acquire");
    guard.release().expect("release");
}

#[test]
fn sync_retrying_exhausts_and_reports_locked() {
    let temp = temp_dir("pi-flock-sync-exhaust-");
    let dir = temp.path().join("target.lock");
    let _held = acquire_once(&dir, None).expect("the holder");

    let started = Instant::now();
    let error = acquire_sync_retrying(&dir).expect_err("the budget must exhaust");
    let elapsed = started.elapsed();
    assert!(matches!(error, LockError::Locked));
    assert!(
        elapsed >= Duration::from_millis(150),
        "nine retry sleeps ran, took: {elapsed:?}"
    );
}

#[test]
fn sync_retrying_propagates_non_locked_errors_immediately() {
    let temp = temp_dir("pi-flock-sync-io-");
    let parent = temp.path().join("ro");
    fs::create_dir(&parent).expect("parent dir");
    fs::set_permissions(&parent, fs::Permissions::from_mode(0o555)).expect("read-only the parent");
    let _restore = RestorePerms { path: &parent };

    let started = Instant::now();
    let error = acquire_sync_retrying(&parent.join("x.lock")).expect_err("the mkdir must fail");
    let elapsed = started.elapsed();
    assert!(
        matches!(error, LockError::Io(_)),
        "the mkdir failure is the io case, got: {error:?}"
    );
    assert!(
        elapsed < Duration::from_millis(100),
        "a non-locked failure never retries, took: {elapsed:?}"
    );
}

#[test]
fn acquire_reports_a_pre_aborted_signal() {
    let temp = temp_dir("pi-flock-preabort-");
    let dir = temp.path().join("target.lock");
    let token = CancellationToken::new();
    token.cancel();
    let options = AsyncLockOptions {
        signal: Some(token),
        on_compromised: None,
    };

    let error = acquire(&dir, &options).expect_err("the abort must fail fast");
    assert!(
        error.downcast_ref::<AbortError>().is_some(),
        "the abort failure rides the boxed error, got: {error}"
    );
    assert!(!dir.exists(), "nothing was acquired");
}

#[test]
fn acquire_reports_a_held_lock_without_a_signal() {
    let temp = temp_dir("pi-flock-held-");
    let dir = temp.path().join("target.lock");
    let _held = acquire_once(&dir, None).expect("the holder");

    let error = acquire(
        &dir,
        &AsyncLockOptions {
            signal: None,
            on_compromised: None,
        },
    )
    .expect_err("the held lock must fail");
    let error = error
        .downcast_ref::<LockError>()
        .expect("the lock error is boxed as-is");
    assert!(matches!(error, LockError::Locked));
}

#[test]
fn acquire_succeeds_with_a_live_signal() {
    let temp = temp_dir("pi-flock-live-");
    let dir = temp.path().join("target.lock");
    let options = AsyncLockOptions {
        signal: Some(CancellationToken::new()),
        on_compromised: None,
    };

    let guard = acquire(&dir, &options).expect("acquire");
    assert!(dir.exists());
    guard.release().expect("release");
}

#[tokio::test]
async fn mkdir_lock_serves_the_file_lock_trait() {
    let temp = temp_dir("pi-flock-trait-");
    let dir = temp.path().join("target.lock");
    let lock: Arc<dyn FileLock> = Arc::new(MkdirLock);

    let guard = lock
        .lock(
            &dir,
            &AsyncLockOptions {
                signal: None,
                on_compromised: None,
            },
        )
        .await
        .expect("the trait acquire lands");
    assert!(dir.exists());

    let error = lock
        .lock(
            &dir,
            &AsyncLockOptions {
                signal: None,
                on_compromised: None,
            },
        )
        .await
        .expect_err("the second acquire must fail");
    let error = error
        .downcast_ref::<LockError>()
        .expect("the held lock boxes the lock error");
    assert!(matches!(error, LockError::Locked));

    guard.release().expect("release");
}

#[test]
fn async_lock_options_debug_renders_the_fields() {
    let plain = format!(
        "{:?}",
        AsyncLockOptions {
            signal: None,
            on_compromised: None,
        }
    );
    assert!(plain.contains("signal: None"));
    assert!(plain.contains("on_compromised: None"));

    let full = format!(
        "{:?}",
        AsyncLockOptions {
            signal: Some(CancellationToken::new()),
            on_compromised: Some(recording_hook(Arc::new(Mutex::new(Vec::new())))),
        }
    );
    assert!(full.contains("signal: Some"));
    assert!(full.contains("on_compromised: Some(\"()\")"));
}

#[test]
fn release_with_hook_fires_only_for_a_compromised_lock() {
    // A vanished lock fires the hook with the compromise error.
    let temp = temp_dir("pi-flock-hook-compromised-");
    let dir = temp.path().join("target.lock");
    let fired = Arc::new(Mutex::new(Vec::<String>::new()));
    let guard = acquire_once(&dir, None).expect("acquire");
    fs::remove_dir(&dir).expect("the lock vanishes");
    release_with_hook(
        guard,
        &AsyncLockOptions {
            signal: None,
            on_compromised: Some(recording_hook(Arc::clone(&fired))),
        },
    );
    assert_eq!(
        fired
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone(),
        vec!["lock was compromised".to_string()],
    );

    // Any other release failure is swallowed silently.
    let temp = temp_dir("pi-flock-hook-io-");
    let dir = temp.path().join("target.lock");
    let fired = Arc::new(Mutex::new(Vec::<String>::new()));
    let guard = acquire_once(&dir, None).expect("acquire");
    fs::write(dir.join("stray"), "x").expect("a file inside the lock dir");
    release_with_hook(
        guard,
        &AsyncLockOptions {
            signal: None,
            on_compromised: Some(recording_hook(Arc::clone(&fired))),
        },
    );
    assert!(
        fired
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty(),
        "an io failure never fires the hook"
    );

    // A clean release fires nothing.
    let temp = temp_dir("pi-flock-hook-clean-");
    let dir = temp.path().join("target.lock");
    let fired = Arc::new(Mutex::new(Vec::<String>::new()));
    let guard = acquire_once(&dir, None).expect("acquire");
    release_with_hook(
        guard,
        &AsyncLockOptions {
            signal: None,
            on_compromised: Some(recording_hook(Arc::clone(&fired))),
        },
    );
    assert!(
        fired
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty(),
        "a clean release never fires the hook"
    );
}
