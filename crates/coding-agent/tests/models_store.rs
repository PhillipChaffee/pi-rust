//! Upstream `packages/coding-agent/test/models-store.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::models_store` (#121).
//!
//! Porting restatements this suite records:
//!
//! - `vi.spyOn(lockfile, "lock")` restates as [`FileLock`] doubles injected
//!   through `FileModelsStore::with_lock_strategy`; the counting double
//!   delegates to the real directory lock the way the spy wrapped the real
//!   `lockfile.lock`, and the gated double stands in for
//!   `mockImplementation(async () => { await lockGranted; return release; })`.
//! - The `release` mock tally restates as the lock directory's absence: the
//!   port's guard release is the only remover and runs before the settled
//!   read returns, so the check is race-free.
//! - The process-wide shared read state is a single slot keyed on the first
//!   file-backed store's path. Upstream ran one shared path sequentially;
//!   the port gives every case its own temp path and pins the slot to the
//!   coalescing case's path through a claim store every case initializes
//!   first, so the coalescing case's interleaved stores share one snapshot
//!   while the other cases stay isolated and run in parallel.
//! - Upstream skips the mode case on win32; the port compiles it only on
//!   unix, where the mode assertions exist.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "a wrong abort outcome is reported by panicking"
)]

#[expect(
    dead_code,
    reason = "the fixture module is compiled into every test binary and this suite consumes only the model fixtures"
)]
mod common;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use common::model_layer::model;
use pi_ai::models_store::{ModelsStore, ModelsStoreEntry, ModelsStoreError, ModelsStoreOptions};
use pi_ai::types::{BoxedFuture, Model};
use pi_ai::utils::abort::AbortError;
use pi_coding_agent::file_lock::{self, AsyncLockOptions, FileLock, FileLockGuard, lock_dir_for};
use pi_coding_agent::models_store::FileModelsStore;
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The acquire result the doubles spell out, the trait's error type.
type AcquireResult = Result<FileLockGuard, Box<dyn std::error::Error + Send + Sync>>;

/// Count the async lock acquisitions and delegate to the real directory
/// lock, upstream's `vi.spyOn(lockfile, "lock")` call tally.
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

/// Gate the first acquire behind the test's signal, upstream's
/// `mockImplementation(async () => { await lockGranted; return release; })`.
struct GatedLock {
    gate: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
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

/// The slot-owning fixture every case initializes before building a store:
/// the process-wide shared read state is one slot keyed on the first
/// file-backed store's path, so the coalescing case's path must claim it
/// before any other case constructs a store (see the suite header).
struct SlotFixture {
    /// The shared scratch dir, upstream's `sharedTempDir`, kept alive for
    /// the process lifetime.
    dir: tempfile::TempDir,
    /// The slot path, upstream's `sharedModelsPath`.
    path: String,
    /// The lock-call tally the coalescing case's stores share.
    double: Arc<CountingLock>,
}

/// The one fixture, built on first use.
static SLOT_FIXTURE: OnceLock<SlotFixture> = OnceLock::new();

/// The fixture, claiming the shared read state's slot on first use: the
/// claim store is built before any case's own stores.
fn slot_fixture() -> &'static SlotFixture {
    SLOT_FIXTURE.get_or_init(|| {
        let dir = temp_dir("pi-models-store-shared-");
        let path = store_file(&dir);
        FileModelsStore::new(Some(&path)).expect("the shared read state's claim store");
        SlotFixture {
            dir,
            path,
            double: Arc::new(CountingLock {
                calls: AtomicUsize::new(0),
            }),
        }
    })
}

/// The temp dir one case owns, upstream's per-suite temp dir with one dir
/// per case (see the suite header).
fn temp_dir(prefix: &'static str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("temp dir")
}

/// The store file's path inside a case's temp dir, upstream's
/// `join(sharedTempDir, "models-store.json")` shape.
fn store_file(dir: &tempfile::TempDir) -> String {
    dir.path()
        .join("models-store.json")
        .to_string_lossy()
        .into_owned()
}

/// A file store over an injected lock double, the `vi.spyOn` cases' builder.
fn double_backed_store(path: &str, lock: Arc<dyn FileLock>) -> FileModelsStore {
    FileModelsStore::with_lock_strategy(path, lock).expect("store")
}

/// The store-API entry, upstream's `{ models: [...] }` and
/// `{ models: [...], checkedAt: 100 }` literals.
fn store_entry(models: &[Model], checked_at: Option<i64>) -> ModelsStoreEntry {
    ModelsStoreEntry {
        models: models.to_vec(),
        last_modified: None,
        checked_at,
        etag: None,
    }
}

/// One catalog entry's stored shape for files written straight to disk,
/// upstream's `{ models: [...] }` literal: the camelCase wire fields.
fn entry_value(models: &[Model]) -> serde_json::Value {
    json!({ "models": models })
}

/// Write the store file, upstream's `writeFileSync(sharedModelsPath, ...)`.
fn write_models_file(path: &str, value: &serde_json::Value) {
    std::fs::write(
        path,
        serde_json::to_string(value).expect("the store file serializes"),
    )
    .expect("write the store file");
}

/// Read the store file, the assertions' view of what a write persisted.
fn read_models_file(path: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read the store file"))
        .expect("the store file parses")
}

/// The entry's model ids, upstream's `.models.map((entry) => entry.id)`.
fn model_ids(entry: &ModelsStoreEntry) -> Vec<&str> {
    entry
        .models
        .iter()
        .map(|stored| stored.id.as_str())
        .collect()
}

/// Assert the operation failed with the abort failure, upstream's
/// `rejects.toMatchObject({ name: "AbortError" })`.
fn assert_aborted<T>(result: Result<T, ModelsStoreError>, what: &str) {
    let Err(error) = result else {
        panic!("{what} must abort");
    };
    assert!(
        error.downcast_ref::<AbortError>().is_some(),
        "{what} must fail with AbortError, got: {error}"
    );
}

#[tokio::test]
async fn persists_provider_catalogs_without_replacing_unrelated_providers() {
    slot_fixture();
    let dir = temp_dir("pi-models-store-persist-");
    let path = store_file(&dir);

    let store = FileModelsStore::new(Some(&path)).expect("store");
    store
        .write("one", store_entry(&[model("one", "m1")], Some(100)), None)
        .await
        .expect("write one");
    store
        .write("two", store_entry(&[model("two", "m2")], Some(200)), None)
        .await
        .expect("write two");

    let reloaded = FileModelsStore::new(Some(&path)).expect("reloaded store");
    let one = reloaded
        .read("one", None)
        .await
        .expect("read one")
        .expect("the first provider persists");
    assert_eq!(model_ids(&one), vec!["m1"]);
    let one_again = reloaded
        .read("one", None)
        .await
        .expect("reread one")
        .expect("the first provider persists");
    assert_eq!(one_again.checked_at, Some(100));
    let two = reloaded
        .read("two", None)
        .await
        .expect("read two")
        .expect("the second provider persists");
    assert_eq!(model_ids(&two), vec!["m2"]);

    reloaded.delete("one", None).await.expect("delete one");
    let deleted = reloaded
        .read("one", None)
        .await
        .expect("read the deleted provider");
    assert_eq!(deleted, None, "the deleted provider reads as none");
    let survivor = reloaded
        .read("two", None)
        .await
        .expect("read the surviving provider")
        .expect("the second provider survives the delete");
    assert_eq!(model_ids(&survivor), vec!["m2"]);
}

#[cfg(unix)]
#[tokio::test]
async fn preserves_the_mode_of_an_existing_models_file() {
    slot_fixture();
    let dir = temp_dir("pi-models-store-mode-");
    let path = dir
        .path()
        .join("managed-mode.json")
        .to_string_lossy()
        .into_owned();
    std::fs::write(&path, "{}").expect("seed the managed file");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o660))
        .expect("chmod the managed file");

    let store = FileModelsStore::new(Some(&path)).expect("store");
    store
        .write("one", store_entry(&[model("one", "m1")], Some(100)), None)
        .await
        .expect("write one");

    let mode = std::fs::metadata(&path)
        .expect("the store file exists")
        .permissions()
        .mode();
    assert_eq!(
        mode & 0o777,
        0o660,
        "the write preserved the administrator-managed mode"
    );
}

#[tokio::test]
async fn coalesces_file_reloads_across_concurrent_readers_and_interleaved_storage_instances() {
    let fixture = slot_fixture();
    let path = fixture.path.as_str();

    write_models_file(
        path,
        &json!({
            "one": entry_value(&[model("one", "old")]),
            "two": entry_value(&[model("two", "m2")]),
        }),
    );
    let first = double_backed_store(path, fixture.double.clone());
    let second = double_backed_store(path, fixture.double.clone());

    let first_options = ModelsStoreOptions {
        signal: Some(CancellationToken::new()),
    };
    let second_options = ModelsStoreOptions {
        signal: Some(CancellationToken::new()),
    };
    let missing_options = ModelsStoreOptions {
        signal: Some(CancellationToken::new()),
    };
    let (one, two, missing) = tokio::join!(
        first.read("one", Some(&first_options)),
        second.read("two", Some(&second_options)),
        first.read("missing", Some(&missing_options)),
    );
    let one = one
        .expect("read one")
        .expect("the coalesced reload resolves the first reader");
    assert_eq!(model_ids(&one), vec!["old"]);
    let two = two
        .expect("read two")
        .expect("the coalesced reload resolves the second reader");
    assert_eq!(model_ids(&two), vec!["m2"]);
    assert_eq!(
        missing.expect("read the missing provider"),
        None,
        "a missing provider reads as none"
    );
    assert_eq!(
        fixture.double.calls.load(Ordering::SeqCst),
        1,
        "three concurrent readers share one coalesced reload and one lock call"
    );

    let reread = second
        .read("one", None)
        .await
        .expect("reread one")
        .expect("the settled reload serves the follow-up read");
    assert_eq!(model_ids(&reread), vec!["old"]);
    assert_eq!(
        fixture.double.calls.load(Ordering::SeqCst),
        1,
        "the settled reload short-circuits the follow-up read"
    );

    let other_path = fixture
        .dir
        .path()
        .join("other-models-store.json")
        .to_string_lossy()
        .into_owned();
    std::fs::write(&other_path, "{}").expect("write the other store file");
    let other = double_backed_store(&other_path, fixture.double.clone());
    let other_one = other.read("one", None).await.expect("read the other store");
    assert_eq!(other_one, None, "a different path reads its own file");
    let other_again = other
        .read("one", None)
        .await
        .expect("reread the other store");
    assert_eq!(
        other_again, None,
        "the other path's revision caches the reread"
    );

    let third = double_backed_store(path, fixture.double.clone());
    write_models_file(
        path,
        &json!({ "one": entry_value(&[model("one", "newest-model")]) }),
    );
    let (first_reload, third_reload) =
        tokio::join!(first.read("one", None), third.read("one", None));
    let first_reload = first_reload
        .expect("the first store's reload")
        .expect("the first store reloads the rewritten file");
    assert_eq!(model_ids(&first_reload), vec!["newest-model"]);
    let third_reload = third_reload
        .expect("the third store's reload")
        .expect("the third store reloads the rewritten file");
    assert_eq!(model_ids(&third_reload), vec!["newest-model"]);
    assert_eq!(
        fixture.double.calls.load(Ordering::SeqCst),
        3,
        "the rewritten file coalesces the last two readers into one more lock call"
    );
}

#[tokio::test]
async fn keeps_a_coalesced_reload_alive_while_another_reader_is_still_waiting() {
    slot_fixture();
    let dir = temp_dir("pi-models-store-keepalive-");
    let path = store_file(&dir);
    write_models_file(
        &path,
        &json!({ "one": entry_value(&[model("one", "stored")]) }),
    );

    let (gate_tx, gate_rx) = tokio::sync::oneshot::channel::<()>();
    let double = Arc::new(GatedLock {
        gate: Mutex::new(Some(gate_rx)),
        calls: AtomicUsize::new(0),
    });
    let store = Arc::new(double_backed_store(&path, double.clone()));

    let first_token = CancellationToken::new();
    let second_token = CancellationToken::new();
    let first_options = ModelsStoreOptions {
        signal: Some(first_token.clone()),
    };
    let second_options = ModelsStoreOptions {
        signal: Some(second_token.clone()),
    };
    let first_task = tokio::spawn({
        let store = Arc::clone(&store);
        async move { store.read("one", Some(&first_options)).await }
    });
    let second_task = tokio::spawn({
        let store = Arc::clone(&store);
        async move { store.read("one", Some(&second_options)).await }
    });

    // Both readers have joined the coalesced reload and the reload is gated
    // on the lock. The first reader's abort must not kill the shared reload.
    tokio::time::sleep(Duration::from_millis(10)).await;
    first_token.cancel();
    assert_aborted(
        first_task.await.expect("join the first reader"),
        "the aborted reader",
    );

    let _ = gate_tx.send(());
    let second = second_task
        .await
        .expect("join the second reader")
        .expect("the second read")
        .expect("the second reader's reload resolves");
    assert_eq!(model_ids(&second), vec!["stored"]);
    assert_eq!(
        double.calls.load(Ordering::SeqCst),
        1,
        "one lock call covers both readers"
    );
    assert!(
        !lock_dir_for(&path).exists(),
        "the reload released its lock"
    );
}

#[tokio::test]
async fn cancels_a_catalog_write_waiting_for_a_held_file_lock_without_writing_later() {
    slot_fixture();
    let dir = temp_dir("pi-models-store-cancel-write-");
    let path = store_file(&dir);
    write_models_file(
        &path,
        &json!({ "one": entry_value(&[model("one", "existing")]) }),
    );

    let held = file_lock::acquire(
        &lock_dir_for(&path),
        &AsyncLockOptions {
            signal: None,
            on_compromised: None,
        },
    )
    .expect("the test holds the lock");
    let store = Arc::new(FileModelsStore::new(Some(&path)).expect("store"));
    let token = CancellationToken::new();
    let options = ModelsStoreOptions {
        signal: Some(token.clone()),
    };
    let pending_task = tokio::spawn({
        let store = Arc::clone(&store);
        async move {
            store
                .write(
                    "two",
                    store_entry(&[model("two", "cancelled")], None),
                    Some(&options),
                )
                .await
        }
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    token.cancel();
    assert_aborted(
        pending_task.await.expect("join the pending write"),
        "the cancelled write",
    );
    held.release().expect("release the held lock");
    // Upstream waits past the release for a hypothetical orphaned commit.
    tokio::time::sleep(Duration::from_millis(150)).await;

    let stored = read_models_file(&path);
    assert!(
        stored.get("one").is_some(),
        "the existing provider is intact"
    );
    assert!(
        stored.get("two").is_none(),
        "the cancelled write never landed"
    );
}
