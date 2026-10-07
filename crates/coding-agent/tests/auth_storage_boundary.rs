//! Boundary tests binding the `auth_storage` branches the 1:1 suites leave
//! untested, at pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the wait helper reports a stalled acquire by panicking"
)]

mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::{empty_env, env_with};
use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::types::{
    ApiKeyCredential, AuthError, AuthOptions, Credential, CredentialModifyFn,
};
use pi_ai::types::BoxedFuture;
use pi_ai::utils::abort::AbortError;
use pi_coding_agent::auth_storage::{
    AuthStorage, AuthStorageBackend, AuthStorageData, FileAuthStorageBackend,
    InMemoryAuthStorageBackend, LockOutcome, ReadOnlyAuthStorage, read_stored_credential,
};
use pi_coding_agent::file_lock::{self, AsyncLockOptions, FileLock, LockError, lock_dir_for};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

/// The acquire result the [`FileLock`] doubles spell out.
type AcquireResult = Result<file_lock::FileLockGuard, Box<dyn std::error::Error + Send + Sync>>;

/// The async lock-operation closure's erased shape, the argument
/// `with_lock_async` takes with its output type pinned so the tail coerces.
type LockUpdate = Box<
    dyn FnOnce(Option<&str>) -> BoxedFuture<'static, Result<LockOutcome<()>, AuthError>> + Send,
>;

fn temp_dir(prefix: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("temp dir")
}

fn write_file(path: &Path, content: &str) {
    fs::write(path, content).expect("write the store file");
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

/// The boxed auth failure an operation callback reports, matching the error
/// type the trait objects carry.
fn fail_error(message: &str) -> AuthError {
    Box::new(std::io::Error::other(message.to_owned()))
}

/// Seed a one-entry in-memory store.
fn in_memory(entry: Value) -> AuthStorage<InMemoryAuthStorageBackend> {
    let mut data = AuthStorageData::new();
    data.insert("anthropic".to_owned(), entry);
    AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&data)
}

/// A file-backed store over `path`, with the shared read state the real
/// constructors join.
fn file_store(path: &Path) -> AuthStorage<FileAuthStorageBackend> {
    let path_str = path.to_string_lossy().into_owned();
    AuthStorage::create_with_env(&path_str, empty_env()).expect("store")
}

/// Assert the operation failed with the abort failure.
fn assert_aborted<T>(result: Result<T, AuthError>, what: &str) {
    let Err(error) = result else {
        panic!("{what} must abort");
    };
    assert!(
        error.downcast_ref::<AbortError>().is_some(),
        "{what} must fail with AbortError, got: {error}"
    );
}

/// Wait until the double has recorded `target` acquisitions.
async fn wait_for_calls(calls: &AtomicUsize, target: usize) {
    for _ in 0..500 {
        if calls.load(Ordering::SeqCst) >= target {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("the acquire never reached attempt {target}");
}

/// Fail the first two acquisitions with the held-lock failure so the contended
/// loop sits in its second backoff sleep when the test cancels the signal.
struct ContendedTwiceLock {
    calls: AtomicUsize,
}

impl FileLock for ContendedTwiceLock {
    fn lock<'a>(
        &'a self,
        lock_dir: &'a Path,
        options: &'a AsyncLockOptions,
    ) -> BoxedFuture<'a, AcquireResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let held: AcquireResult = Err(Box::new(LockError::Locked));
        Box::pin(async move {
            if self.calls.load(Ordering::SeqCst) <= 2 {
                return held;
            }
            file_lock::acquire(lock_dir, options)
        })
    }
}

// =============================================================================
// ReadOnlyAuthStorage
// =============================================================================

#[tokio::test]
async fn read_only_missing_file_reads_as_the_empty_record() {
    let temp = temp_dir("pi-auth-ro-missing-");
    let path = temp.path().join("auth.json");
    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");

    // Upstream's ENOENT branch loads the empty record, so a missing store
    // file reads as no credentials rather than a failure.
    let read = store.read("anthropic", None).await.expect("read");
    assert_eq!(read, None);
    let listed = store.list(None).await.expect("list");
    assert!(listed.is_empty());

    // The empty record is cached like any other load.
    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"late"}}"#);
    let cached = store.read("anthropic", None).await.expect("cached read");
    assert_eq!(cached, None, "the ENOENT load was cached");
}

#[tokio::test]
async fn read_only_rejects_a_non_object_root() {
    let temp = temp_dir("pi-auth-ro-root-");
    let path = temp.path().join("auth.json");
    for content in ["[1,2]", "null", "\"string\""] {
        write_file(&path, content);
        let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");
        let error = store
            .read("anthropic", None)
            .await
            .expect_err("the root must fail");
        assert_eq!(error.to_string(), "Invalid auth.json: expected an object");
    }
}

#[tokio::test]
async fn read_only_rejects_an_entry_that_is_not_an_object() {
    let temp = temp_dir("pi-auth-ro-entry-");
    let path = temp.path().join("auth.json");
    write_file(&path, r#"{"anthropic":"bogus"}"#);
    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");

    let error = store
        .read("anthropic", None)
        .await
        .expect_err("the entry must fail");
    assert_eq!(
        error.to_string(),
        "Invalid auth.json credential for provider \"anthropic\""
    );
}

#[tokio::test]
async fn read_only_rejects_an_api_key_with_a_non_string_key() {
    let temp = temp_dir("pi-auth-ro-keytype-");
    let path = temp.path().join("auth.json");
    write_file(&path, r#"{"anthropic":{"type":"api_key","key":123}}"#);
    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");

    let error = store
        .read("anthropic", None)
        .await
        .expect_err("the entry must fail");
    assert_eq!(
        error.to_string(),
        "Invalid auth.json credential for provider \"anthropic\""
    );
}

#[tokio::test]
async fn read_only_rejects_an_oauth_credential_with_non_string_tokens() {
    let temp = temp_dir("pi-auth-ro-oauth-");
    let path = temp.path().join("auth.json");
    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");

    for content in [
        r#"{"anthropic":{"type":"oauth","access":1,"refresh":"r","expires":1}}"#,
        r#"{"anthropic":{"type":"oauth","refresh":"r","access":"a"}}"#,
        r#"{"anthropic":{"type":"oauth","access":"a","refresh":"r","expires":"soon"}}"#,
    ] {
        write_file(&path, content);
        let error = store
            .read("anthropic", None)
            .await
            .expect_err("the entry must fail");
        assert_eq!(
            error.to_string(),
            "Invalid auth.json credential for provider \"anthropic\"",
            "row: {content}"
        );
    }
}

#[tokio::test]
async fn read_only_accepts_a_negative_expiry_and_a_keyless_api_key() {
    let temp = temp_dir("pi-auth-ro-accepts-");
    let path = temp.path().join("auth.json");
    write_file(
        &path,
        r#"{"anthropic":{"type":"api_key"},"openai":{"type":"oauth","access":"a","refresh":"r","expires":-5}}"#,
    );
    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");

    // serde accepts the shapes upstream's validation accepts: a keyless api
    // key, and any finite expiry (JSON cannot carry a non-finite number).
    let keyless = store.read("anthropic", None).await.expect("keyless read");
    assert_eq!(
        keyless,
        Some(Credential::ApiKey(ApiKeyCredential {
            key: None,
            env: None,
        }))
    );
    let oauth = store.read("openai", None).await.expect("oauth read");
    let expected = json!({"type":"oauth","access":"a","refresh":"r","expires":-5});
    assert_eq!(
        oauth.map(|credential| serde_json::to_value(&credential).expect("serialize")),
        Some(expected),
    );
}

#[tokio::test]
async fn read_only_caches_the_load() {
    let temp = temp_dir("pi-auth-ro-cache-");
    let path = temp.path().join("auth.json");
    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"first"}}"#);
    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");
    let first = store.read("anthropic", None).await.expect("first read");
    assert_eq!(first, api_key("first"));

    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"second"}}"#);
    let cached = store.read("anthropic", None).await.expect("cached read");
    assert_eq!(cached, api_key("first"), "the load happened once");
}

#[tokio::test]
async fn read_only_refuses_modify_and_delete_verbatim() {
    let temp = temp_dir("pi-auth-ro-refuse-");
    let store =
        ReadOnlyAuthStorage::new(&temp.path().join("auth.json").to_string_lossy()).expect("store");
    let update: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(None) }));

    let error = store
        .modify("anthropic", update, None)
        .await
        .expect_err("modify must refuse");
    assert_eq!(
        error.to_string(),
        "Read-only credential storage cannot modify auth.json"
    );
    let error = store
        .delete("anthropic", None)
        .await
        .expect_err("delete must refuse");
    assert_eq!(
        error.to_string(),
        "Read-only credential storage cannot modify auth.json"
    );
}

#[tokio::test]
async fn read_only_keeps_a_command_key_unresolved() {
    let temp = temp_dir("pi-auth-ro-command-");
    let path = temp.path().join("auth.json");
    write_file(
        &path,
        r#"{"anthropic":{"type":"api_key","key":"!printf 'x'"}}"#,
    );
    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");

    let read = store.read("anthropic", None).await.expect("read");
    assert_eq!(
        read,
        api_key("!printf 'x'"),
        "a command key passes through unexecuted"
    );
}

#[tokio::test]
async fn read_only_resolves_a_template_key_through_the_injected_env() {
    let temp = temp_dir("pi-auth-ro-template-");
    let path = temp.path().join("auth.json");
    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"$K"}}"#);
    let store =
        ReadOnlyAuthStorage::new_with_env(&path.to_string_lossy(), env_with(&[("K", "resolved")]))
            .expect("store");

    let read = store.read("anthropic", None).await.expect("read");
    assert_eq!(read, api_key("resolved"));
}

#[tokio::test]
async fn read_only_read_reports_a_pre_aborted_signal() {
    let temp = temp_dir("pi-auth-ro-abort-");
    let store =
        ReadOnlyAuthStorage::new(&temp.path().join("auth.json").to_string_lossy()).expect("store");
    let token = CancellationToken::new();
    token.cancel();
    let options = AuthOptions {
        signal: Some(token),
    };

    assert_aborted(store.read("anthropic", Some(&options)).await, "the read");
    assert_aborted(store.list(Some(&options)).await, "the list");
}

#[tokio::test]
async fn an_unseeded_in_memory_store_reads_as_the_empty_record() {
    // The unseeded backend reports no current content, and the missing
    // content parses as the empty record.
    let storage = AuthStorage::from_storage(InMemoryAuthStorageBackend::default());

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(read, None);
}

#[tokio::test]
async fn an_empty_store_file_reads_as_the_empty_record() {
    let temp = temp_dir("pi-auth-empty-file-");
    let path = temp.path().join("auth.json");
    write_file(&path, "");
    let storage = file_store(&path);

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(read, None, "empty content is the empty record");
}

#[tokio::test]
async fn read_only_reports_an_unreadable_store_file() {
    let temp = temp_dir("pi-auth-ro-noperm-");
    let path = temp.path().join("auth.json");
    write_file(&path, "{}");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o000))
        .expect("strip the file's permissions");

    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");
    let error = store
        .read("anthropic", None)
        .await
        .expect_err("the unreadable file must fail");
    assert!(
        error.to_string().contains("Failed to read auth.json"),
        "the read failure carries its message, got: {error}"
    );

    fs::set_permissions(&path, fs::Permissions::from_mode(0o644))
        .expect("restore the file's permissions");
}

#[tokio::test]
async fn read_only_debug_renders_the_path() {
    let temp = temp_dir("pi-auth-ro-debug-");
    let path = temp.path().join("auth.json");
    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");

    assert!(format!("{store:?}").contains("auth.json"));
}

// =============================================================================
// AuthStorage reads
// =============================================================================

#[tokio::test]
async fn list_skips_malformed_entries() {
    let mut data = AuthStorageData::new();
    data.insert("good".to_owned(), json!({"type": "api_key", "key": "k"}));
    data.insert("bad".to_owned(), json!("bogus"));
    let storage = AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&data);

    let listed = storage.list(None).await.expect("list");
    assert_eq!(
        listed
            .iter()
            .map(|info| info.provider_id.as_str())
            .collect::<Vec<_>>(),
        vec!["good"],
        "the malformed entry never lists"
    );
}

#[tokio::test]
async fn read_passes_a_keyless_api_key_through_unresolved() {
    // The keyless shape rides the passthrough arm the resolution path skips:
    // nothing to resolve, the stored credential returns verbatim.
    let mut data = AuthStorageData::new();
    data.insert("anthropic".to_owned(), json!({"type": "api_key"}));
    let storage = AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&data);

    let credential = storage.read("anthropic", None).await.expect("read");
    assert_eq!(
        credential,
        Some(Credential::ApiKey(ApiKeyCredential {
            key: None,
            env: None,
        }))
    );
}

#[tokio::test]
async fn read_only_lists_the_stored_credential_types() {
    let temp = temp_dir("pi-auth-ro-list-");
    let path = temp.path().join("auth.json");
    write_file(
        &path,
        r#"{"anthropic":{"type":"oauth","access":"a","refresh":"r","expires":1},"openai":{"type":"api_key","key":"sk"}}"#,
    );
    let store = ReadOnlyAuthStorage::new(&path.to_string_lossy()).expect("store");

    let listed = store.list(None).await.expect("list");

    assert_eq!(
        listed
            .iter()
            .map(|info| (info.provider_id.as_str(), info.auth_type.to_string()))
            .collect::<Vec<_>>(),
        vec![
            ("anthropic", "oauth".to_string()),
            ("openai", "api_key".to_string())
        ],
    );
}

#[tokio::test]
async fn read_returns_a_stored_null_entry_as_no_credential() {
    let storage = in_memory(Value::Null);

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(read, None, "a null entry is the falsy stored shape");
}

#[tokio::test]
async fn read_reports_a_malformed_entry() {
    let storage = in_memory(json!("bogus"));

    let error = storage
        .read("anthropic", None)
        .await
        .expect_err("the entry must fail");
    assert_eq!(
        error.to_string(),
        "Invalid auth.json credential for provider \"anthropic\""
    );
}

#[tokio::test]
async fn read_and_list_report_a_pre_aborted_signal() {
    let storage = in_memory(json!({"type": "api_key", "key": "k"}));
    let token = CancellationToken::new();
    token.cancel();
    let options = AuthOptions {
        signal: Some(token),
    };

    assert_aborted(storage.read("anthropic", Some(&options)).await, "the read");
    assert_aborted(storage.list(Some(&options)).await, "the list");
}

#[tokio::test]
async fn a_read_over_a_deleted_file_reloads_the_recreated_store() {
    let temp = temp_dir("pi-auth-deleted-");
    let path = temp.path().join("auth.json");
    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"k"}}"#);
    let storage = AuthStorage::create(&path.to_string_lossy()).expect("store");
    fs::remove_file(&path).expect("delete the store file");

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(
        read, None,
        "the missing revision reloads, recreating an empty store"
    );
    assert!(path.exists(), "the reload recreated the store file");
}

#[tokio::test]
async fn reload_preserves_the_snapshot_when_the_file_turns_malformed() {
    let temp = temp_dir("pi-auth-corrupt-");
    let path = temp.path().join("auth.json");
    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"stored"}}"#);
    let storage = file_store(&path);

    write_file(&path, "{invalid-json");
    storage.reload();

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(
        read,
        api_key("stored"),
        "every reload failure preserves the last valid snapshot"
    );
}

#[tokio::test]
async fn reload_preserves_the_snapshot_when_the_lock_cannot_be_acquired() {
    let temp = temp_dir("pi-auth-lockheld-");
    let path = temp.path().join("auth.json");
    let path_str = path.to_string_lossy().into_owned();
    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"stored"}}"#);
    let storage = file_store(&path);

    let held = file_lock::acquire(
        &lock_dir_for(&path_str),
        &AsyncLockOptions {
            signal: None,
            on_compromised: None,
        },
    )
    .expect("the test holds the lock");
    storage.reload();
    held.release().expect("release the held lock");

    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(
        read,
        api_key("stored"),
        "the failed reload kept the snapshot"
    );
}

#[tokio::test]
async fn reload_picks_up_the_new_snapshot_when_the_file_changes() {
    let temp = temp_dir("pi-auth-reread-");
    let path = temp.path().join("auth.json");
    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"old"}}"#);
    let storage = file_store(&path);

    write_file(&path, r#"{"openai":{"type":"api_key","key":"new"}}"#);
    storage.reload();

    let read = storage.read("openai", None).await.expect("read");
    assert_eq!(read, api_key("new"));
    let gone = storage.read("anthropic", None).await.expect("gone read");
    assert_eq!(gone, None);
}

#[tokio::test]
async fn a_backendless_store_falls_back_to_the_snapshot_when_the_reload_fails() {
    let temp = temp_dir("pi-auth-backendless-");
    let path = temp.path().join("auth.json");
    write_file(&path, "{invalid-json");
    let backend = FileAuthStorageBackend::new(&path.to_string_lossy()).expect("backend");
    let storage = AuthStorage::from_storage_with_env(backend, None, empty_env());

    // No path: the reload runs directly, and without a signal its failure
    // resolves to the current snapshot.
    let read = storage.read("anthropic", None).await.expect("read");
    assert_eq!(read, None);

    // With a signal, the failure propagates.
    let token = CancellationToken::new();
    let options = AuthOptions {
        signal: Some(token),
    };
    let error = storage
        .read("anthropic", Some(&options))
        .await
        .expect_err("the signal surfaces the failure");
    assert!(
        error.to_string().contains("Failed to read auth.json"),
        "got: {error}"
    );
}

#[tokio::test]
async fn a_file_backed_read_reports_the_reload_failure_when_a_signal_is_supplied() {
    let temp = temp_dir("pi-auth-signal-");
    let path = temp.path().join("auth.json");
    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"stored"}}"#);
    let storage = file_store(&path);
    write_file(&path, "{invalid-json");

    let token = CancellationToken::new();
    let options = AuthOptions {
        signal: Some(token),
    };
    let error = storage
        .read("anthropic", Some(&options))
        .await
        .expect_err("the reload failure propagates under a signal");
    assert!(
        error.to_string().contains("Failed to read auth.json"),
        "got: {error}"
    );
}

// =============================================================================
// Constructor and backend failures
// =============================================================================

#[test]
fn constructor_failures_surface_from_the_file_url_forms() {
    for error in [
        AuthStorage::create("file://host/share").expect_err("create must fail"),
        FileAuthStorageBackend::new("file://host/share").expect_err("backend must fail"),
        ReadOnlyAuthStorage::new("file://host/share").expect_err("read-only must fail"),
    ] {
        assert!(
            !error.to_string().is_empty(),
            "the normalize failure carries its message"
        );
    }
}

#[test]
fn the_store_creates_nested_parents_owner_only() {
    let temp = temp_dir("pi-auth-nested-");
    let path = temp.path().join("a/b/c/auth.json");
    file_store(&path);

    for dir in ["a", "a/b", "a/b/c"] {
        let mode = fs::metadata(temp.path().join(dir))
            .expect("dir metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700, "{dir} is owner-only");
    }
    let mode = fs::metadata(&path)
        .expect("store metadata")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600, "the store file is owner-only");
}
#[test]
fn with_lock_reports_the_parent_mkdir_failure() {
    let temp = temp_dir("pi-auth-mkdir-fail-");
    let blocker = temp.path().join("blocker");
    write_file(&blocker, "a file, not a directory");
    let backend = FileAuthStorageBackend::new(&blocker.join("sub/auth.json").to_string_lossy())
        .expect("backend");

    let error = backend
        .with_lock(|_current| LockOutcome {
            result: (),
            next: None,
        })
        .expect_err("the mkdir failure propagates");
    assert!(
        error.to_string().contains("File exists"),
        "the blocked component surfaces, got: {error}"
    );
}

#[test]
fn with_lock_reports_a_parentless_relative_path() {
    // A relative path's parent is the empty path, whose own parent is None —
    // the recursion's bottom branch, and a creation that cannot succeed.
    let backend = FileAuthStorageBackend::new("auth.json").expect("backend");

    let error = backend
        .with_lock(|_current| LockOutcome {
            result: (),
            next: None,
        })
        .expect_err("the empty-parent creation fails");
    assert!(
        !error.to_string().is_empty(),
        "the creation failure carries its message"
    );
}

#[test]
fn with_lock_reports_the_root_path_parentless_branch() {
    // The root path has no parent at all: ensureParentDir's bottom branch.
    let backend = FileAuthStorageBackend::new("/").expect("backend");

    let error = backend
        .with_lock(|_current| LockOutcome {
            result: (),
            next: None,
        })
        .expect_err("the root store cannot lock");
    assert!(
        !error.to_string().is_empty(),
        "the lock failure carries its message"
    );
}

#[test]
fn with_lock_reports_the_file_creation_failure() {
    let temp = temp_dir("pi-auth-create-fail-");
    let blocker = temp.path().join("blocker");
    write_file(&blocker, "a file, not a directory");
    let backend =
        FileAuthStorageBackend::new(&blocker.join("auth.json").to_string_lossy()).expect("backend");

    let error = backend
        .with_lock(|_current| LockOutcome {
            result: (),
            next: None,
        })
        .expect_err("the creation failure propagates");
    assert!(
        error.to_string().contains("Not a directory"),
        "the blocked file surfaces, got: {error}"
    );
}

#[test]
fn with_lock_writes_the_next_content_through_the_sync_path() {
    let temp = temp_dir("pi-auth-sync-write-");
    let path = temp.path().join("auth.json");
    write_file(&path, "{}");
    let backend = FileAuthStorageBackend::new(&path.to_string_lossy()).expect("backend");

    let outcome = backend.with_lock(|current| {
        assert_eq!(current, Some("{}"));
        LockOutcome {
            result: (),
            next: Some(r#"{"a":1}"#.to_owned()),
        }
    });
    outcome.expect("the write lands");
    assert_eq!(fs::read_to_string(&path).expect("store read"), r#"{"a":1}"#,);
}

#[test]
fn with_lock_reports_a_write_failure_when_the_store_file_is_a_directory() {
    let temp = temp_dir("pi-auth-dir-store-");
    let path = temp.path().join("auth.json");
    fs::create_dir(&path).expect("the store path is a directory");
    let backend = FileAuthStorageBackend::new(&path.to_string_lossy()).expect("backend");

    let error = backend
        .with_lock(|_current| LockOutcome {
            result: (),
            next: Some("{}".to_owned()),
        })
        .expect_err("the write failure propagates");
    assert!(
        error.to_string().contains("Is a directory"),
        "the blocked write surfaces, got: {error}"
    );
}

#[test]
fn with_lock_reports_a_compromised_lock_when_the_release_fails() {
    let temp = temp_dir("pi-auth-release-");
    let path = temp.path().join("auth.json");
    write_file(&path, "{}");
    let path_str = path.to_string_lossy().into_owned();
    let backend = FileAuthStorageBackend::new(&path_str).expect("backend");

    let error = backend
        .with_lock(|_current| {
            // The lock directory vanishes while the operation holds it, the
            // dependency's compromise condition.
            let _ = fs::remove_dir(lock_dir_for(&path_str));
            LockOutcome {
                result: (),
                next: None,
            }
        })
        .expect_err("the compromised release propagates");
    assert_eq!(error.to_string(), "lock was compromised");
}

// =============================================================================
// In-memory backend branches
// =============================================================================

#[test]
fn in_memory_backend_writes_through_the_sync_lock() {
    let backend = InMemoryAuthStorageBackend::default();

    let outcome = AuthStorageBackend::with_lock(&backend, |_current| LockOutcome {
        result: (),
        next: Some("{\"a\":1}".to_owned()),
    });
    outcome.expect("the seed lands");

    let seen = AuthStorageBackend::with_lock(&backend, |current| {
        assert_eq!(current, Some("{\"a\":1}"));
        LockOutcome {
            result: (),
            next: None,
        }
    });
    seen.expect("the follow-up read lands");
}

#[tokio::test]
async fn modify_reports_the_fn_failure_from_the_in_memory_backend() {
    let storage = in_memory(json!({"type": "api_key", "key": "stored"}));
    let failure: CredentialModifyFn = Box::new(|_current: Option<Credential>| {
        Box::pin(async move { Err(fail_error("refused")) })
    });

    let error = storage
        .modify("anthropic", failure, None)
        .await
        .expect_err("the callback failure propagates");
    assert!(
        error.to_string().contains("refused"),
        "the operation failure carries its message, got: {error}"
    );
    let stored = storage.read("anthropic", None).await.expect("stored read");
    assert_eq!(stored, api_key("stored"), "the failed modify never wrote");
}

#[tokio::test]
async fn modify_with_undefined_rereads_a_missing_provider() {
    let temp = temp_dir("pi-auth-modify-missing-");
    let path = temp.path().join("auth.json");
    write_file(&path, "{}");
    let storage = file_store(&path);
    let leave: CredentialModifyFn =
        Box::new(|_current: Option<Credential>| Box::pin(async move { Ok(None) }));

    let modified = storage.modify("openai", leave, None).await.expect("modify");
    assert_eq!(modified, None, "the re-read of a missing provider is None");
    assert_eq!(
        fs::read_to_string(&path).expect("store read"),
        "{}",
        "the undefined answer never writes"
    );
}

// =============================================================================
// Contended acquire branches
// =============================================================================

#[tokio::test]
async fn contended_acquire_aborts_during_backoff() {
    let temp = temp_dir("pi-auth-backoff-");
    let path = temp.path().join("auth.json");
    write_file(&path, "{}");
    let path_str = path.to_string_lossy().into_owned();
    let double = Arc::new(ContendedTwiceLock {
        calls: AtomicUsize::new(0),
    });
    let lock: Arc<dyn FileLock> = double.clone();
    let backend = FileAuthStorageBackend::with_lock_strategy(&path_str, lock);
    let update: LockUpdate = Box::new(|_current: Option<&str>| {
        Box::pin(async move {
            Ok(LockOutcome {
                result: (),
                next: None,
            })
        })
    });
    let token = CancellationToken::new();
    let options = AuthOptions {
        signal: Some(token.clone()),
    };

    let task = tokio::spawn({
        let backend = backend;
        async move { backend.with_lock_async(update, Some(&options)).await }
    });
    // Two held-lock failures have landed; the second backoff sleep (20-40ms)
    // is in flight. Cancelling now takes the select's abort arm.
    wait_for_calls(&double.calls, 2).await;
    token.cancel();

    let outcome = task.await.expect("join the acquire");
    assert_aborted(outcome, "the aborted backoff");
    assert_eq!(
        double.calls.load(Ordering::SeqCst),
        2,
        "the abort stops the retry loop"
    );
    assert!(
        !lock_dir_for(&path_str).exists(),
        "the double never acquired, so no lock leaks"
    );
}

// =============================================================================
// One-off read
// =============================================================================

#[test]
fn read_stored_credential_pins_the_one_off_read() {
    let temp = temp_dir("pi-auth-oneoff-");
    let path = temp.path().join("auth.json");
    let path_str = path.to_string_lossy().into_owned();

    write_file(&path, r#"{"anthropic":{"type":"api_key","key":"k"}}"#);
    assert_eq!(read_stored_credential("anthropic", &path_str), api_key("k"));
    assert_eq!(read_stored_credential("missing", &path_str), None);

    write_file(&path, "{invalid-json");
    assert_eq!(read_stored_credential("anthropic", &path_str), None);

    write_file(&path, r#"{"anthropic":"bogus"}"#);
    assert_eq!(read_stored_credential("anthropic", &path_str), None);

    let missing = temp.path().join("nope.json");
    assert_eq!(
        read_stored_credential("anthropic", &missing.to_string_lossy()),
        None,
        "a missing file reads as no credential"
    );
}

#[tokio::test]
async fn the_debug_forms_render_the_path() {
    let temp = temp_dir("pi-auth-debug-");
    let path = temp.path().join("auth.json");
    let path_str = path.to_string_lossy().into_owned();
    let backend = FileAuthStorageBackend::new(&path_str).expect("backend");
    assert!(format!("{backend:?}").contains("auth_path"));

    let storage = AuthStorage::create_with_env(&path_str, empty_env()).expect("store");
    assert!(format!("{storage:?}").contains(&path_str));
}
