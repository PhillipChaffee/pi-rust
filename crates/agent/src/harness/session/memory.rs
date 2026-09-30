//! The memory session backend, ported from upstream
//! `src/harness/session/memory.ts`.
//!
//! It carries the in-process [`MemoryStorage`], the admitted-operation
//! facade over the storage-backed session, and the
//! [`MemorySessionRepo`] lifecycle.
//!
#![expect(
    clippy::significant_drop_tightening,
    reason = "the lifecycle and tracker locks guard multi-step state transitions; early drops would race the admits close waits for"
)]
//!
//! Upstream serializes commits through a promise queue; the port applies
//! each memory operation synchronously at the call, so admission order is
//! call order and the queue machinery is gone. Upstream's insertion-ordered
//! session map restates as a `BTreeMap` keyed by session id: list order is
//! id-ascending, the order the suites' assertions sort to. The facade's
//! admitted-operation set restates as a counter with a drain signal, the
//! same `close`-waits-admitted-operations contract.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pi_ai::types::BoxedFuture;

use crate::harness::context::Context;
use crate::harness::session::facade;
use crate::harness::session::in_memory_storage_state::InMemoryStorageState;
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::types::{
    CommitResult, Entry, EntryScan, EntryStructure, ForkOptions,
    Session, SessionCreateOptions, SessionError, SessionMetadata, SessionRepo, SessionStats, Storage, StorageBranchScan,
    UsageRow, UsageScan,
};
use crate::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, Write,
};

/// The injected clock, upstream's `now?: () => number`.
pub type NowFn = Arc<dyn Fn() -> i64 + Send + Sync>;

/// The options a [`MemoryStorage`] accepts, upstream's `MemoryStorageOptions`.
#[derive(Clone, Default)]
pub struct MemoryStorageOptions {
    /// The injected clock; the wall clock when omitted, upstream's
    /// `Date.now` default.
    pub now: Option<NowFn>,
}

impl std::fmt::Debug for MemoryStorageOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryStorageOptions")
            .finish_non_exhaustive()
    }
}

/// The options a [`MemorySessionRepo`] accepts, upstream's
/// `MemorySessionRepoOptions`.
#[derive(Clone, Default)]
pub struct MemorySessionRepoOptions {
    /// The injected clock; the wall clock when omitted.
    pub now: Option<NowFn>,
}

impl std::fmt::Debug for MemorySessionRepoOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemorySessionRepoOptions")
            .finish_non_exhaustive()
    }
}

pub(crate) fn default_now() -> NowFn {
    Arc::new(pi_ai::auth::resolve::now_ms)
}

/// The memory lifecycle, upstream's `"open" | "closed"` for the storage:
/// memory operations complete at the call, so the closing state is
/// unobservable and collapses out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lifecycle {
    /// Accepting operations.
    Open,
    /// Closed.
    Closed,
}

/// The in-process storage, upstream's `MemoryStorage`.
///
/// Every operation applies synchronously at the call: the open-state check
/// and the state application are observable at the call, matching
/// upstream's enqueue-then-settle behavior where the check and queue slot
/// are taken at the call.
pub struct MemoryStorage {
    now: NowFn,
    state: Mutex<InMemoryStorageState>,
    lifecycle: Mutex<Lifecycle>,
}

impl std::fmt::Debug for MemoryStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryStorage").finish_non_exhaustive()
    }
}

impl MemoryStorage {
    /// A storage with the injected or wall clock, upstream's constructor.
    #[must_use]
    pub fn new(options: MemoryStorageOptions) -> Self {
        Self {
            now: options.now.unwrap_or_else(default_now),
            state: Mutex::new(InMemoryStorageState::new()),
            lifecycle: Mutex::new(Lifecycle::Open),
        }
    }

    fn assert_open(&self) -> Result<(), SessionError> {
        if *self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            == Lifecycle::Closed
        {
            return Err(SessionError::Message("MemoryStorage is closed".to_owned()));
        }
        Ok(())
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, InMemoryStorageState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Construct a destination storage at the current serialized boundary,
    /// upstream's `fork`.
    ///
    /// # Errors
    /// A `SessionError::Message` when the storage is closed or the fork
    /// scope is invalid for the source state.
    pub fn fork(&self, options: &ForkOptions) -> Result<Arc<Self>, SessionError> {
        self.assert_open()?;
        let destination = Self {
            now: self.now.clone(),
            state: Mutex::new(self.lock_state().create_fork(options)?),
            lifecycle: Mutex::new(Lifecycle::Open),
        };
        Ok(Arc::new(destination))
    }
}

impl Storage for MemoryStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        let result = (|| -> Result<CommitResult, SessionError> {
            self.assert_open()?;
            let mut storage_state = self.lock_state();
            let prepared = storage_state.prepare_commit(writes, (self.now)())?;
            let stats = storage_state.apply_validated(&prepared.writes);
            Ok(CommitResult {
                first_seq: prepared.result.first_seq,
                seqs: prepared.result.seqs,
                timestamp: prepared.result.timestamp,
                stats,
            })
        })();
        Box::pin(std::future::ready(result))
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>> {
        let result = self
            .assert_open()
            .map(|()| self.lock_state().get_entries(&ids));
        Box::pin(std::future::ready(result))
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        let result = self
            .assert_open()
            .map(|()| self.lock_state().get_value(address));
        Box::pin(std::future::ready(result))
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        let result = self
            .assert_open()
            .map(|()| self.lock_state().scan_values(prefix));
        Box::pin(std::future::ready(result))
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        let result = (|| -> Result<Vec<ListElement>, SessionError> {
            self.assert_open()?;
            self.lock_state().read_list(address, options)
        })();
        Box::pin(std::future::ready(result))
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        let result = (|| -> Result<Vec<Entry>, SessionError> {
            self.assert_open()?;
            self.lock_state().scan_branch(query)
        })();
        Box::pin(std::future::ready(result))
    }

    fn scan_branch_structure(
        &self,
        query: &StorageBranchScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<EntryStructure>, SessionError>> {
        let result = (|| -> Result<Vec<EntryStructure>, SessionError> {
            self.assert_open()?;
            self.lock_state().scan_branch_structure(query)
        })();
        Box::pin(std::future::ready(result))
    }

    fn scan_entries(
        &self,
        query: &EntryScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        let result = self
            .assert_open()
            .map(|()| self.lock_state().scan_entries(query));
        Box::pin(std::future::ready(result))
    }

    fn scan_usage(
        &self,
        query: &UsageScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<UsageRow>, SessionError>> {
        let result = self
            .assert_open()
            .map(|()| self.lock_state().scan_usage(query));
        Box::pin(std::future::ready(result))
    }

    fn get_stats(&self, _context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        let result = self.assert_open().map(|()| self.lock_state().get_stats());
        Box::pin(std::future::ready(result))
    }

    fn close(&self, _context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        *self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closed;
        Box::pin(std::future::ready(Ok(())))
    }
}

/// The one repo record, upstream's `MemorySessionRecord`; the clone shares
/// the storage and session handles, matching upstream's shared references.
#[derive(Clone)]
struct MemorySessionRecord {
    metadata: SessionMetadata,
    storage: Arc<MemoryStorage>,
    session: Arc<StorageBackedSession>,
    open: Arc<AtomicBool>,
}

/// The in-process session repository, upstream's `MemorySessionRepo`.
pub struct MemorySessionRepo {
    now: NowFn,
    sessions: Mutex<BTreeMap<String, MemorySessionRecord>>,
    pending_ids: Mutex<BTreeSet<String>>,
    closed: AtomicBool,
    close_cell: tokio::sync::OnceCell<Result<(), SessionError>>,
}

impl std::fmt::Debug for MemorySessionRepo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemorySessionRepo").finish_non_exhaustive()
    }
}

impl MemorySessionRepo {
    /// A repo with the injected or wall clock, upstream's constructor.
    #[must_use]
    pub fn new(options: MemorySessionRepoOptions) -> Self {
        Self {
            now: options.now.unwrap_or_else(default_now),
            sessions: Mutex::new(BTreeMap::new()),
            pending_ids: Mutex::new(BTreeSet::new()),
            closed: AtomicBool::new(false),
            close_cell: tokio::sync::OnceCell::new(),
        }
    }

    fn assert_open(&self) -> Result<(), SessionError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(SessionError::Message(
                "MemorySessionRepo is closed".to_owned(),
            ));
        }
        Ok(())
    }

    /// Reserve a destination id, upstream's `reserveId`.
    ///
    /// # Errors
    /// A `SessionError::Message` when the id is taken or pending.
    fn reserve_id(&self, id: &str) -> Result<(), SessionError> {
        let sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        let mut pending = self
            .pending_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if sessions.contains_key(id) || pending.contains(id) {
            return Err(SessionError::Message(format!(
                "Session already exists: {id}"
            )));
        }
        pending.insert(id.to_owned());
        Ok(())
    }

    fn release_pending_id(&self, id: &str) {
        self.pending_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(id);
    }

    fn open_record(record: &MemorySessionRecord) -> Box<dyn Session> {
        // The close flow runs the record's open flag flip, then the shared
        // core marks the facade closed; upstream's memory close marks the
        // facade closed before its on-close hook - the flip ordering is
        // unobservable (the hook is infallible and nothing reads the
        // lifecycle concurrently), and the port records the swap.
        let open = record.open.clone();
        let on_close: Arc<dyn Fn() + Send + Sync> =
            Arc::new(move || open.store(false, Ordering::Release));
        let core = facade::FacadeCore::new(
            record.session.clone(),
            record.metadata.clone(),
            Arc::new(move |_context| {
                (on_close.clone())();
                Box::pin(std::future::ready(Ok(())))
            }),
        );
        Box::new(facade::FacadeCore::clone(&core))
    }

    /// Build the record's session for one created or forked destination.
    fn build_record(
        id: String,
        created_at: i64,
        parent_session_id: Option<String>,
        storage: Arc<MemoryStorage>,
    ) -> MemorySessionRecord {
        let metadata = SessionMetadata {
            id,
            created_at,
            storage_version: MEMORY_STORAGE_VERSION,
            cwd: None,
            parent_session_id,
            legacy_parent_session_path: None,
        };
        let session = Arc::new(StorageBackedSession::new(
            metadata.clone(),
            storage.clone(),
            StorageBackedSessionOptions::default(),
        ));
        MemorySessionRecord {
            metadata,
            storage,
            session,
            open: Arc::new(AtomicBool::new(true)),
        }
    }
}

/// The memory backend's storage version, upstream's
/// `MEMORY_STORAGE_VERSION`.
pub const MEMORY_STORAGE_VERSION: u32 = 1;

impl SessionRepo for MemorySessionRepo {
    fn create(
        &self,
        options: SessionCreateOptions,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Session>, SessionError>> {
        Box::pin(async move {
            self.assert_open()?;
            let created_at = (self.now)();
            let id = match options.id {
                Some(id) => id,
                None => session_id(created_at)?,
            };
            self.reserve_id(&id)?;
            let record = Self::build_record(
                id.clone(),
                created_at,
                options.parent_session_id,
                Arc::new(MemoryStorage::new(MemoryStorageOptions {
                    now: Some(self.now.clone()),
                })),
            );
            let session = Self::open_record(&record);
            self.sessions
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(id.clone(), record);
            self.release_pending_id(&id);
            Ok(session)
        })
    }

    fn open(
        &self,
        metadata: &SessionMetadata,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Session>, SessionError>> {
        let id = metadata.id.clone();
        Box::pin(async move {
            self.assert_open()?;
            let sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
            let Some(record) = sessions.get(&id) else {
                return Err(SessionError::Message(format!("Unknown session: {id}")));
            };
            if record.open.load(Ordering::Acquire) {
                return Err(SessionError::Message(format!(
                    "Session is already open: {id}"
                )));
            }
            record.open.store(true, Ordering::Release);
            Ok(Self::open_record(record))
        })
    }

    fn list(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<SessionMetadata>, SessionError>> {
        Box::pin(std::future::ready(self.assert_open().map(|()| {
            self.sessions
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .values()
                .map(|record| record.metadata.clone())
                .collect()
        })))
    }

    fn delete(
        &self,
        metadata: &SessionMetadata,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        let id = metadata.id.clone();
        let context = context.clone();
        Box::pin(async move {
            self.assert_open()?;
            let record = self
                .sessions
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&id)
                .cloned()
                .ok_or_else(|| SessionError::Message(format!("Unknown session: {id}")))?;
            if record.open.load(Ordering::Acquire) {
                return Err(SessionError::Message(format!("Session is open: {id}")));
            }
            record.session.close(&context).await?;
            self.sessions
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&id);
            Ok(())
        })
    }

    fn fork(
        &self,
        source: &SessionMetadata,
        options: &ForkOptions,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Session>, SessionError>> {
        let options = options.clone();
        let source_id = source.id.clone();
        Box::pin(async move {
            self.assert_open()?;
            let source_record = self
                .sessions
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&source_id)
                .cloned()
                .ok_or_else(|| SessionError::Message(format!("Unknown session: {source_id}")))?;
            let created_at = (self.now)();
            let id = match options.id() {
                Some(id) => id.clone(),
                None => session_id(created_at)?,
            };
            self.reserve_id(&id)?;
            let storage = match source_record.storage.fork(&options) {
                Ok(storage) => storage,
                Err(error) => {
                    self.release_pending_id(&id);
                    return Err(error);
                }
            };
            let record = Self::build_record(
                id.clone(),
                created_at,
                Some(source_record.metadata.id),
                storage,
            );
            let session = Self::open_record(&record);
            self.sessions
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(id.clone(), record);
            self.release_pending_id(&id);
            Ok(session)
        })
    }
}

impl MemorySessionRepo {
    /// Close every session, upstream's `close` (memory-specific, outside
    /// the repository contract).
    ///
    /// # Errors
    /// The first session close failure; every session still closes.
    pub async fn close(&self, context: &Context) -> Result<(), SessionError> {
        self.closed.store(true, Ordering::Release);
        self.close_cell
            .get_or_init(|| async {
                let sessions: Vec<Arc<StorageBackedSession>> = {
                    let sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
                    sessions
                        .values()
                        .map(|record| record.session.clone())
                        .collect()
                };
                let mut first_error = None;
                for session in sessions {
                    if let Err(error) = session.close(context).await
                        && first_error.is_none()
                    {
                        first_error = Some(error);
                    }
                }
                first_error.map_or(Ok(()), Err)
            })
            .await
            .clone()
    }
}

/// The session id a generated identity takes, upstream's `uuidv7(createdAt)`.
///
/// # Errors
/// The uuidv7 range failures.
fn session_id(created_at: i64) -> Result<String, SessionError> {
    pi_ai::utils::uuid::uuidv7(u64::try_from(created_at).ok())
        .map_err(|error| SessionError::Message(error.to_string()))
}
