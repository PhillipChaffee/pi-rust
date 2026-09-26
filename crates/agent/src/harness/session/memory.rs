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

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pi_ai::types::BoxedFuture;

use crate::harness::context::Context;
use crate::harness::session::in_memory_storage_state::InMemoryStorageState;
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::types::{
    Branch, BranchScan, CommitResult, Entry, EntryQuery, EntryScan, EntryStructure, ForkOptions,
    IdGenerator, Session, SessionCreateOptions, SessionError, SessionMetadata, SessionMutation,
    SessionMutationCallback, SessionMutator, SessionReader, SessionRepo, SessionStats, Storage,
    StorageBranchScan, UsageRow, UsageScan,
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

fn default_now() -> NowFn {
    Arc::new(pi_ai::auth::resolve::now_ms)
}

/// The facade lifecycle, upstream's `"open" | "closing" | "closed"`.
///
/// Memory operations complete at the call, so the closing state is
/// unobservable and collapses out; only open and closed remain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lifecycle {
    /// Accepting operations.
    Open,
    /// Closing; operations not yet admitted reject.
    Closing,
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

/// The admitted-operation counter with its drain signal, upstream's
/// `admitted` promise set on the memory facade: registration increments,
/// settlement decrements, and close waits for the count to reach zero.
#[derive(Debug, Default)]
struct AdmissionTracker {
    open: Mutex<usize>,
    drained: tokio::sync::Notify,
}

impl AdmissionTracker {
    fn register(self: &Arc<Self>) -> Arc<AdmissionSlot> {
        *self.open.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        Arc::new(AdmissionSlot {
            tracker: self.clone(),
            done: AtomicBool::new(false),
        })
    }

    fn release(&self) {
        let mut open = self.open.lock().unwrap_or_else(PoisonError::into_inner);
        *open -= 1;
        if *open == 0 {
            self.drained.notify_waiters();
        }
    }

    /// Wait until every admitted operation has settled, upstream's
    /// `Promise.allSettled([...this.admitted])`.
    async fn drain(&self) {
        loop {
            if *self.open.lock().unwrap_or_else(PoisonError::into_inner) == 0 {
                return;
            }
            self.drained.notified().await;
        }
    }
}

/// One admitted operation's slot; settlement decrements the tracker, and a
/// dropped slot settles too so close never waits on an abandoned operation.
struct AdmissionSlot {
    tracker: Arc<AdmissionTracker>,
    done: AtomicBool,
}

impl AdmissionSlot {
    fn finish(&self) {
        if !self.done.swap(true, Ordering::AcqRel) {
            self.tracker.release();
        }
    }
}

impl Drop for AdmissionSlot {
    fn drop(&mut self) {
        self.finish();
    }
}

/// The facade's shared state: upstream's `MemorySessionFacade` fields, Arc'd
/// so the admitted branch objects and mutation callbacks outlive the
/// `Box<dyn Session>` handle they borrow from.
struct FacadeInner {
    session: Arc<StorageBackedSession>,
    metadata: SessionMetadata,
    id_generator: Arc<dyn IdGenerator>,
    lifecycle: Mutex<Lifecycle>,
    tracker: Arc<AdmissionTracker>,
    on_close: Arc<dyn Fn() + Send + Sync>,
    close_cell: tokio::sync::OnceCell<()>,
}

/// The admitted-operation facade over one storage-backed session, upstream's
/// `MemorySessionFacade`: every operation admits through the open-state
/// gate and registers with the tracker, and close waits for the admitted
/// operations before marking the session closed.
struct MemorySessionFacade(Arc<FacadeInner>);

impl MemorySessionFacade {
    fn new(session: Arc<StorageBackedSession>, on_close: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self(Arc::new(FacadeInner {
            metadata: session.metadata().clone(),
            id_generator: session.id_generator_arc(),
            session,
            lifecycle: Mutex::new(Lifecycle::Open),
            tracker: Arc::new(AdmissionTracker::default()),
            on_close,
            close_cell: tokio::sync::OnceCell::new(),
        }))
    }

    fn is_open(&self) -> bool {
        self.0.is_open()
    }

    fn closed_error() -> SessionError {
        SessionError::Message("Session is closed".to_owned())
    }

    fn wrap_branch(&self, branch: Box<dyn Branch>) -> Box<dyn Branch> {
        Box::new(AdmittedBranch {
            branch,
            inner: self.0.clone(),
        })
    }
}

/// The admitted-operation wrapper the facade's forwards share: the
/// open-state gate, the tracker slot around the operation's settlement,
/// upstream's `admit`.
fn admit_operation<'a, T: 'a>(
    inner: Arc<FacadeInner>,
    operation: BoxedFuture<'a, Result<T, SessionError>>,
) -> BoxedFuture<'a, Result<T, SessionError>> {
    Box::pin(async move {
        if !inner.is_open() {
            return Err(SessionError::Message("Session is closed".to_owned()));
        }
        let slot = inner.tracker.register();
        let outcome = operation.await;
        slot.finish();
        outcome
    })
}

impl FacadeInner {
    fn is_open(&self) -> bool {
        *self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            == Lifecycle::Open
    }
}

/// The branch surface admitted through the facade, upstream's
/// `wrapBranch`.
struct AdmittedBranch {
    branch: Box<dyn Branch>,
    inner: Arc<FacadeInner>,
}

impl Branch for AdmittedBranch {
    fn name(&self) -> &str {
        self.branch.name()
    }

    fn get_tip_id(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        admit_operation(self.inner.clone(), self.branch.get_tip_id(context))
    }

    fn find_entries(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        admit_operation(self.inner.clone(), self.branch.find_entries(query, context))
    }

    fn find_entry(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        admit_operation(self.inner.clone(), self.branch.find_entry(query, context))
    }

    fn append_message(
        &self,
        message: crate::types::AgentMessage,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, SessionError>> {
        admit_operation(
            self.inner.clone(),
            self.branch.append_message(message, context),
        )
    }

    fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, SessionError>> {
        admit_operation(
            self.inner.clone(),
            self.branch.append_custom_entry(custom_type, data, context),
        )
    }
}

impl SessionReader for MemorySessionFacade {
    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.get_entries(ids, context))
    }

    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.get_stats(context))
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.get_value(address, context))
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.scan_values(prefix, context))
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        admit_operation(
            self.0.clone(),
            self.0.session.read_list(address, options, context),
        )
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.scan_branch(query, context))
    }
}

/// The granted mutation wrapped with the facade's admission slot, upstream's
/// `beginMutation` return object.
struct AdmittedMutation {
    source: Box<dyn SessionMutation>,
    slot: Option<Arc<AdmissionSlot>>,
}

impl SessionReader for AdmittedMutation {
    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>> {
        self.source.get_entries(ids, context)
    }

    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        self.source.get_stats(context)
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        self.source.get_value(address, context)
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        self.source.scan_values(prefix, context)
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        self.source.read_list(address, options, context)
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.source.scan_branch(query, context)
    }
}

impl SessionMutator for AdmittedMutation {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        self.source.commit(writes, context)
    }
}

impl SessionMutation for AdmittedMutation {
    fn end(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        let slot = self.slot.clone();
        let context = context.clone();
        Box::pin(async move {
            let ended = self.source.end(&context).await;
            if let Some(slot) = &slot {
                slot.finish();
            }
            ended
        })
    }
}

impl Drop for AdmittedMutation {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.finish();
        }
    }
}

impl Session for MemorySessionFacade {
    fn metadata(&self) -> &SessionMetadata {
        &self.0.metadata
    }

    fn id_generator(&self) -> &dyn IdGenerator {
        &*self.0.id_generator
    }

    fn get_entry(
        &self,
        id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.get_entry(id, context))
    }

    fn get_name(&self, context: &Context) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.get_name(context))
    }

    fn get_label(
        &self,
        target_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.get_label(target_id, context))
    }

    fn find_entries(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.find_entries(query, context))
    }

    fn find_entry(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        admit_operation(self.0.clone(), self.0.session.find_entry(query, context))
    }

    fn branch(
        &self,
        name: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Box<dyn Branch>>, SessionError>> {
        let inner = self.0.clone();
        let name = name.to_owned();
        let context = context.clone();
        Box::pin(async move {
            let branch = admit_operation(inner.clone(), inner.session.branch(&name, &context))
                .await?
                .map(|branch| self.wrap_branch(branch));
            Ok(branch)
        })
    }

    fn create_branch(
        &self,
        name: &str,
        at: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Branch>, SessionError>> {
        let inner = self.0.clone();
        let name = name.to_owned();
        let context = context.clone();
        Box::pin(async move {
            let branch = admit_operation(
                inner.clone(),
                inner.session.create_branch(&name, at, &context),
            )
            .await?;
            Ok(self.wrap_branch(branch))
        })
    }

    fn begin_mutation(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn SessionMutation>, SessionError>> {
        let context = context.clone();
        Box::pin(async move {
            let slot = self.0.tracker.register();
            let source = match self.0.session.begin_mutation(&context).await {
                Ok(source) => source,
                Err(error) => {
                    slot.finish();
                    return Err(error);
                }
            };
            if !self.is_open() {
                source.end(&context).await?;
                slot.finish();
                return Err(Self::closed_error());
            }
            let granted: Box<dyn SessionMutation> = Box::new(AdmittedMutation {
                source,
                slot: Some(slot),
            });
            Ok(granted)
        })
    }

    fn mutate(
        &self,
        mutation: SessionMutationCallback,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>> {
        let inner = self.0.clone();
        let context = context.clone();
        Box::pin(admit_operation(
            self.0.clone(),
            Box::pin(async move {
                let callback_inner = inner.clone();
                inner
                    .session
                    .mutate(
                        Box::new(
                            move |mutator: &dyn SessionMutator, mutation_context: &Context| {
                                // Upstream re-checks the facade's open state when
                                // the callback body runs, so a callback queued
                                // before close still rejects.
                                let open = callback_inner.is_open();
                                if !open {
                                    return Box::pin(std::future::ready(Err(Self::closed_error())));
                                }
                                mutation(mutator, mutation_context)
                            },
                        ),
                        &context,
                    )
                    .await
            }),
        ))
    }

    fn set_value(
        &self,
        address: &ValueAddress,
        next: serde_json::Value,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        admit_operation(
            self.0.clone(),
            self.0.session.set_value(address, next, context),
        )
    }

    fn delete_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        admit_operation(
            self.0.clone(),
            self.0.session.delete_value(address, context),
        )
    }

    fn append_list(
        &self,
        address: &ListAddress,
        element: serde_json::Value,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        admit_operation(
            self.0.clone(),
            self.0.session.append_list(address, element, context),
        )
    }

    fn delete_list(
        &self,
        address: &ListAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        admit_operation(self.0.clone(), self.0.session.delete_list(address, context))
    }

    fn set_name(
        &self,
        name: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        admit_operation(self.0.clone(), self.0.session.set_name(name, context))
    }

    fn set_label(
        &self,
        target_id: &str,
        label: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        admit_operation(
            self.0.clone(),
            self.0.session.set_label(target_id, label, context),
        )
    }

    fn close(&self, _context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        // Upstream marks the facade closing synchronously in the close call;
        // operations not admitted before it reject from here on.
        if self.is_open() {
            *self
                .0
                .lifecycle
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closing;
        }
        let inner = self.0.clone();
        Box::pin(async move {
            inner
                .close_cell
                .get_or_init(|| async {
                    inner.tracker.drain().await;
                    *inner
                        .lifecycle
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closed;
                    (inner.on_close)();
                })
                .await;
            Ok(())
        })
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
        let open = record.open.clone();
        Box::new(MemorySessionFacade::new(
            record.session.clone(),
            Arc::new(move || open.store(false, Ordering::Release)),
        ))
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
