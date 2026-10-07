//! Catalog persistence for dynamically refreshed provider catalogs,
//! upstream's `src/core/models-store.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: the file store rides the same locked-backend and
//! process-wide read-state machinery `auth_storage.rs` carries — the
//! `models-store.json` file has the auth store's shape, so the lock
//! semantics (mkdir locks, the 10×20ms sync ladder, async stale recovery)
//! carry over; the shared read state keys on one path the way the auth
//! file's single slot does.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use pi_ai::auth::types::{AuthError, AuthOptions};
use pi_ai::models_store::{ModelsStore, ModelsStoreEntry, ModelsStoreError, ModelsStoreOptions};
use pi_ai::types::BoxedFuture;
use pi_ai::utils::abort::AbortError;

use crate::auth_storage::{AuthStorageBackend, FileAuthStorageBackend, LockOutcome};
use crate::config::get_agent_dir;
use crate::file_lock::FileLock;
use crate::utils::abort::race_with_abort_signal;
use crate::utils::paths::{PathInputOptions, get_file_revision, normalize_path};
use crate::utils::text::strip_bom;

/// The stored catalogs keyed by provider id, upstream's `StoredModels`. The
/// JSON map keeps insertion order, the way upstream's object did.
type StoredModels = serde_json::Map<String, Value>;

fn signal_check(options: Option<&ModelsStoreOptions>) -> Result<(), ModelsStoreError> {
    if options.is_some_and(|options| {
        options
            .signal
            .as_ref()
            .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
    }) {
        return Err(Box::new(AbortError));
    }
    Ok(())
}

/// The in-memory store used by tests and non-file runtimes, upstream's
/// `InMemoryCodingAgentModelsStore`. Entries clone on the way in and out.
#[derive(Debug, Default)]
pub struct InMemoryCodingAgentModelsStore {
    entries: Mutex<BTreeMap<String, ModelsStoreEntry>>,
}

impl ModelsStore for InMemoryCodingAgentModelsStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<Option<ModelsStoreEntry>, ModelsStoreError>> {
        Box::pin(async move {
            signal_check(options)?;
            Ok(self
                .entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(provider_id)
                .cloned())
        })
    }

    fn write<'a>(
        &'a self,
        provider_id: &'a str,
        entry: ModelsStoreEntry,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), ModelsStoreError>> {
        Box::pin(async move {
            signal_check(options)?;
            self.entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(provider_id.to_owned(), entry);
            Ok(())
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), ModelsStoreError>> {
        Box::pin(async move {
            signal_check(options)?;
            self.entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(provider_id);
            Ok(())
        })
    }
}

/// One catalog entry's on-disk shape, upstream's inline `ModelsStoreEntry`
/// JSON. pi-ai's entry type does not serialize, so the file store carries
/// this mapping.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredEntry {
    models: Vec<pi_ai::types::Model>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_modified: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checked_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    etag: Option<String>,
}

impl StoredEntry {
    fn from_entry(entry: &ModelsStoreEntry) -> Self {
        Self {
            models: entry.models.clone(),
            last_modified: entry.last_modified,
            checked_at: entry.checked_at,
            etag: entry.etag.clone(),
        }
    }

    fn into_entry(self) -> ModelsStoreEntry {
        ModelsStoreEntry {
            models: self.models,
            last_modified: self.last_modified,
            checked_at: self.checked_at,
            etag: self.etag,
        }
    }
}

/// The per-path snapshot, upstream's `ModelsFileReadState`.
#[derive(Default)]
struct ModelsFileReadState {
    data: StoredModels,
    revision: Option<String>,
    reload: Option<Arc<ModelsFileReload>>,
}

/// One in-flight coalesced reload, upstream's `ModelsFileReload`: the
/// cancellation the last departing reader arms, the reader count, the
/// settled result, and the waiters' notification.
struct ModelsFileReload {
    controller: tokio_util::sync::CancellationToken,
    readers: std::sync::atomic::AtomicUsize,
    result: Mutex<Option<Result<StoredModels, String>>>,
    ready: tokio::sync::Notify,
}

impl ModelsFileReload {
    async fn settle(&self) -> Result<StoredModels, String> {
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
    /// current reload and arms its cancellation.
    fn depart(slot: &Arc<Self>, read_state: &Mutex<ModelsFileReadState>) {
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
                slot.controller.cancel();
            }
        }
    }
}

/// The process-wide slot's shape: the path it was taken under and the shared
/// snapshot.
type SharedReadStateSlot = Option<(String, Arc<Mutex<ModelsFileReadState>>)>;

/// The one process-wide slot, upstream's `sharedModelsFileReadState`: the
/// first file-backed store takes it, same-path stores share its snapshot.
static SHARED_MODELS_FILE_READ_STATE: LazyLock<Mutex<SharedReadStateSlot>> =
    LazyLock::new(|| Mutex::new(None));

fn shared_read_state_for(path: &str) -> Arc<Mutex<ModelsFileReadState>> {
    let mut shared = SHARED_MODELS_FILE_READ_STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((taken_path, state)) = shared.as_ref() {
        if taken_path == path {
            return Arc::clone(state);
        }
        return Arc::new(Mutex::new(ModelsFileReadState::default()));
    }
    let state = Arc::new(Mutex::new(ModelsFileReadState::default()));
    *shared = Some((path.to_string(), Arc::clone(&state)));
    state
}

/// Parse stored file content, upstream's `parse`: an absent file is empty,
/// invalid JSON is the reload's failure.
fn parse_stored_models(content: Option<&str>) -> Result<StoredModels, String> {
    let Some(content) = content else {
        return Ok(StoredModels::new());
    };
    let parsed: Value =
        serde_json::from_str(strip_bom(content).trim_end()).map_err(|error| error.to_string())?;
    let Some(entries) = parsed.as_object() else {
        return Err("Expected object".to_owned());
    };
    Ok(entries.clone().into_iter().collect())
}

/// Locked JSON-backed storage for dynamically refreshed provider catalogs,
/// upstream's `FileModelsStore`.
pub struct FileModelsStore {
    storage: Arc<FileAuthStorageBackend>,
    path: String,
    read_state: Arc<Mutex<ModelsFileReadState>>,
}

impl std::fmt::Debug for FileModelsStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileModelsStore")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl FileModelsStore {
    /// The store over `path`, defaulting to `models-store.json` beside the
    /// agent dir, upstream's constructor.
    ///
    /// # Errors
    /// A path that does not normalize.
    pub fn new(path: Option<&str>) -> Result<Self, ModelsStoreError> {
        let path = match path {
            Some(path) => normalize_path(path, &PathInputOptions::default())
                .map_err(|error| -> ModelsStoreError { Box::new(error) })?,
            None => get_agent_dir()
                .join("models-store.json")
                .display()
                .to_string(),
        };
        let storage = Arc::new(FileAuthStorageBackend::new(&path)?);
        let read_state = shared_read_state_for(&path);
        Ok(Self {
            storage,
            path,
            read_state,
        })
    }

    /// The store over `path` with a replaced lock strategy, the test seam
    /// standing in for upstream's `vi.spyOn(lockfile, "lock")`.
    ///
    /// # Errors
    /// A path that does not normalize.
    pub fn with_lock_strategy(
        path: &str,
        lock: Arc<dyn FileLock>,
    ) -> Result<Self, ModelsStoreError> {
        let path = normalize_path(path, &PathInputOptions::default())
            .map_err(|error| -> ModelsStoreError { Box::new(error) })?;
        let storage = Arc::new(FileAuthStorageBackend::with_lock_strategy(&path, lock));
        let read_state = shared_read_state_for(&path);
        Ok(Self {
            storage,
            path,
            read_state,
        })
    }

    fn update_read_state(
        read_state: &Mutex<ModelsFileReadState>,
        data: StoredModels,
        revision: Option<String>,
    ) {
        let mut state = read_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.data = data;
        if revision.is_some() {
            state.revision = revision;
        }
    }

    fn reload_from_storage(
        &self,
        read_state: Arc<Mutex<ModelsFileReadState>>,
        options: Option<&ModelsStoreOptions>,
    ) -> BoxedFuture<'static, Result<StoredModels, String>> {
        let storage = Arc::clone(&self.storage);
        let path = self.path.clone();
        let signal = options.and_then(|options| options.signal.clone());
        Box::pin(async move {
            let outcome = storage
                .with_lock_async(
                    move |content| {
                        let content = content.map(str::to_string);
                        Box::pin(async move {
                            let data = parse_stored_models(content.as_deref());
                            if let Ok(data) = &data {
                                Self::update_read_state(
                                    &read_state,
                                    data.clone(),
                                    get_file_revision(&path),
                                );
                            }
                            data.map(|data| LockOutcome {
                                result: data,
                                next: None,
                            })
                            .map_err(|error| -> AuthError {
                                Box::new(std::io::Error::other(error))
                            })
                        })
                    },
                    Some(&AuthOptions {
                        signal: signal.clone(),
                    }),
                )
                .await;
            outcome.map_err(|error| error.to_string())
        })
    }

    /// The coalescing read, upstream's `readLatest`: a matching revision
    /// serves the cached snapshot; otherwise one locked reload is shared by
    /// every waiting reader and cancelled by the last departure.
    #[expect(
        clippy::option_if_let_else,
        clippy::significant_drop_tightening,
        reason = "the coalescing reload reads clearest as an if-let beside the lock the slot is built under, and the guard lives as long as the snapshot it guards"
    )]
    async fn read_latest(
        &self,
        read_state: &Arc<Mutex<ModelsFileReadState>>,
        options: Option<&ModelsStoreOptions>,
    ) -> Result<StoredModels, ModelsStoreError> {
        signal_check(options)?;
        let revision = get_file_revision(&self.path);
        {
            let state = read_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if revision.is_some() && state.revision == revision {
                return Ok(state.data.clone());
            }
        }
        let slot = {
            let mut state = read_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(slot) = state.reload.clone() {
                slot
            } else {
                let slot = Arc::new(ModelsFileReload {
                    controller: tokio_util::sync::CancellationToken::new(),
                    readers: std::sync::atomic::AtomicUsize::new(0),
                    result: Mutex::new(None),
                    ready: tokio::sync::Notify::new(),
                });
                state.reload = Some(Arc::clone(&slot));
                let spawn_slot = Arc::clone(&slot);
                let read_state = Arc::clone(read_state);
                let reload = self.reload_from_storage(
                    Arc::clone(&read_state),
                    Some(&ModelsStoreOptions {
                        signal: Some(slot.controller.clone()),
                    }),
                );
                tokio::spawn(async move {
                    let settled = reload.await;
                    *spawn_slot
                        .result
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(settled);
                    spawn_slot.ready.notify_waiters();
                    // Upstream clears the slot on both settlement paths.
                    let should_clear = {
                        let mut state = read_state
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        let current_matches = state
                            .reload
                            .as_ref()
                            .is_some_and(|current| Arc::ptr_eq(current, &spawn_slot));
                        if current_matches {
                            state.reload = None;
                        }
                        current_matches
                    };
                    let _ = should_clear;
                });
                slot
            }
        };
        slot.readers
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let raced = race_with_abort_signal(
            slot.settle(),
            options.and_then(|options| options.signal.as_ref()),
        )
        .await;
        ModelsFileReload::depart(&slot, read_state);
        match raced {
            Ok(data) => Ok(data),
            Err(crate::utils::abort::RaceError::Aborted(_)) => Err(Box::new(AbortError)),
            Err(crate::utils::abort::RaceError::Operation(error)) => {
                Err(Box::<dyn std::error::Error + Send + Sync>::from(error))
            }
        }
    }
}

impl ModelsStore for FileModelsStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<Option<ModelsStoreEntry>, ModelsStoreError>> {
        Box::pin(async move {
            let entries = self.read_latest(&self.read_state, options).await?;
            signal_check(options)?;
            let entry = entries.get(provider_id).cloned();
            match entry {
                Some(value) => Ok(Some(
                    serde_json::from_value::<StoredEntry>(value)
                        .map_err(|error| -> ModelsStoreError { Box::new(error) })?
                        .into_entry(),
                )),
                None => Ok(None),
            }
        })
    }

    fn write<'a>(
        &'a self,
        provider_id: &'a str,
        entry: ModelsStoreEntry,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), ModelsStoreError>> {
        let write_entry = StoredEntry::from_entry(&entry);
        Box::pin(async move {
            let signal = options.and_then(|options| options.signal.clone());
            let provider_id = provider_id.to_owned();
            let outcome = self
                .storage
                .with_lock_async(
                    move |content| {
                        let content = content.map(str::to_string);
                        let write_entry = write_entry.clone();
                        let provider_id = provider_id;
                        Box::pin(async move {
                            let mut current = parse_stored_models(content.as_deref())?;
                            current.insert(
                                provider_id,
                                serde_json::to_value(&write_entry)
                                    .map_err(|error| error.to_string())?,
                            );
                            let next = serde_json::to_string_pretty(&current)
                                .map_err(|error| error.to_string())?;
                            Ok(LockOutcome {
                                result: current,
                                next: Some(next),
                            })
                        })
                    },
                    Some(&AuthOptions {
                        signal: signal.clone(),
                    }),
                )
                .await;
            match outcome {
                Ok(latest) => {
                    Self::update_read_state(&self.read_state, latest, None);
                    Ok(())
                }
                Err(error) => Err(error),
            }
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), ModelsStoreError>> {
        Box::pin(async move {
            let signal = options.and_then(|options| options.signal.clone());
            let provider_id = provider_id.to_owned();
            let outcome = self
                .storage
                .with_lock_async(
                    move |content| {
                        let content = content.map(str::to_string);
                        let provider_id = provider_id;
                        Box::pin(async move {
                            let mut current = parse_stored_models(content.as_deref())?;
                            current.remove(&provider_id);
                            let next = serde_json::to_string_pretty(&current)
                                .map_err(|error| error.to_string())?;
                            Ok(LockOutcome {
                                result: current,
                                next: Some(next),
                            })
                        })
                    },
                    Some(&AuthOptions {
                        signal: signal.clone(),
                    }),
                )
                .await;
            match outcome {
                Ok(latest) => {
                    Self::update_read_state(&self.read_state, latest, None);
                    Ok(())
                }
                Err(error) => Err(error),
            }
        })
    }
}
