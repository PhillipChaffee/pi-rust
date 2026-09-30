//! The admitted-operation facade machinery shared by the backends that gate
//! one [`StorageBackedSession`] behind an admission tracker.
//!
//! Upstream implements the same pattern separately per backend (`memory.ts`,
//! `sqlite-node`'s `session.ts`) — an admitted promise set, an `admit` gate,
//! and wrapped branches and mutations; the port shares the machinery here so
//! the workspace's duplication budget stays green.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pi_ai::types::BoxedFuture;

use crate::harness::context::Context;
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::types::{
    Branch, BranchScan, CommitResult, Entry, EntryQuery, IdGenerator, Session, SessionError,
    SessionMetadata, SessionMutation, SessionMutationCallback, SessionMutator, SessionReader,
    SessionStats, StorageBranchScan,
};
use crate::harness::session::values::{ListAddress, ListElement, StoredValue, ValueAddress};

/// The admitted-operation counter with its drain signal, upstream's
/// `admitted` promise set on the backends' facades.
///
/// Registration increments, settlement decrements, and close waits for the
/// count to reach zero.
#[derive(Debug, Default)]
pub struct AdmissionTracker {
    open: Mutex<usize>,
    drained: tokio::sync::Notify,
}

impl AdmissionTracker {
    /// Registers one admitted operation; the returned slot settles the
    /// registration.
    pub fn register(self: &Arc<Self>) -> Arc<AdmissionSlot> {
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

    /// Waits until every admitted operation has settled, upstream's
    /// `Promise.allSettled([...this.admitted])`.
    pub async fn drain(&self) {
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
#[derive(Debug)]
pub struct AdmissionSlot {
    tracker: Arc<AdmissionTracker>,
    done: AtomicBool,
}

impl AdmissionSlot {
    /// Settles the slot, idempotent, upstream's promise removal.
    pub fn finish(&self) {
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

/// The open-state gate around one facade's admitted operations: the tracker
/// plus the facade's open check, upstream's `admit`.
#[derive(Clone)]
pub struct AdmissionGate {
    tracker: Arc<AdmissionTracker>,
    is_open: Arc<dyn Fn() -> bool + Send + Sync>,
}

impl std::fmt::Debug for AdmissionGate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AdmissionGate")
            .finish_non_exhaustive()
    }
}

impl AdmissionGate {
    /// A gate over the tracker and the facade's open check.
    pub fn new(
        tracker: Arc<AdmissionTracker>,
        is_open: Arc<dyn Fn() -> bool + Send + Sync>,
    ) -> Self {
        Self { tracker, is_open }
    }

    /// Whether the facade still admits operations.
    #[must_use]
    pub fn is_open(&self) -> bool {
        (self.is_open)()
    }

    /// Registers one admitted slot, upstream's admitted-set insertion.
    #[must_use]
    pub fn register(&self) -> Arc<AdmissionSlot> {
        self.tracker.register()
    }

    /// Waits for every admitted operation, upstream's
    /// `Promise.allSettled([...this.admitted])`.
    pub async fn drain(&self) {
        self.tracker.drain().await;
    }

    /// The admitted-operation wrapper the facade's forwards share: the
    /// open-state gate, the tracker slot around the operation's settlement,
    /// upstream's `admit`.
    #[must_use]
    pub fn admit<'a, T: 'a>(
        &self,
        operation: BoxedFuture<'a, Result<T, SessionError>>,
    ) -> BoxedFuture<'a, Result<T, SessionError>> {
        let gate = self.clone();
        Box::pin(async move {
            if !gate.is_open() {
                return Err(SessionError::Message("Session is closed".to_owned()));
            }
            let slot = gate.tracker.register();
            let outcome = operation.await;
            slot.finish();
            outcome
        })
    }
}

/// The branch surface admitted through a facade's gate, upstream's
/// `wrapBranch`: every method gates at call time, so a branch object
/// obtained while open rejects post-close.
pub struct AdmittedBranch {
    branch: Box<dyn Branch>,
    gate: AdmissionGate,
}

impl std::fmt::Debug for AdmittedBranch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AdmittedBranch")
            .finish_non_exhaustive()
    }
}

impl AdmittedBranch {
    /// Wraps one branch with the gate, upstream's `wrapBranch`.
    #[must_use]
    #[expect(
        clippy::new_ret_no_self,
        reason = "the constructor returns the erased Branch handle, upstream's wrapBranch"
    )]
    pub fn new(branch: Box<dyn Branch>, gate: AdmissionGate) -> Box<dyn Branch> {
        Box::new(Self { branch, gate })
    }
}

impl Branch for AdmittedBranch {
    fn name(&self) -> &str {
        self.branch.name()
    }

    fn get_tip_id(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        self.gate.admit(self.branch.get_tip_id(context))
    }

    fn find_entries(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.gate.admit(self.branch.find_entries(query, context))
    }

    fn find_entry(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        self.gate.admit(self.branch.find_entry(query, context))
    }

    fn append_message(
        &self,
        message: crate::types::AgentMessage,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, SessionError>> {
        self.gate
            .admit(self.branch.append_message(message, context))
    }

    fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, SessionError>> {
        self.gate
            .admit(self.branch.append_custom_entry(custom_type, data, context))
    }
}

/// The granted mutation wrapped with a facade's admission slot, upstream's
/// `beginMutation` return object: the reader surface passes through and
/// `end` releases the slot.
pub struct AdmittedMutation {
    /// The granted mutation.
    pub source: Box<dyn SessionMutation>,
    /// The admission slot, released on `end` or drop.
    pub slot: Option<Arc<AdmissionSlot>>,
}

impl std::fmt::Debug for AdmittedMutation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AdmittedMutation")
            .finish_non_exhaustive()
    }
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
        options: Option<crate::harness::session::values::ListReadOptions>,
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
        writes: Vec<crate::harness::session::values::Write>,
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

/// Grants one admitted mutation, upstream's `beginMutation` facade body.
///
/// The slot registers before admission, and a close that lands while
/// admission is in flight unwinds the granted mutation and rejects.
///
/// # Errors
/// The session's begin error, or the closed error when the facade closed
/// during admission.
pub async fn begin_admitted_mutation(
    gate: &AdmissionGate,
    session: &Arc<StorageBackedSession>,
    context: &Context,
) -> Result<Box<dyn SessionMutation>, SessionError> {
    let slot = gate.register();
    let source = match session.begin_mutation(context).await {
        Ok(source) => source,
        Err(error) => {
            slot.finish();
            return Err(error);
        }
    };
    if !gate.is_open() {
        source.end(context).await?;
        slot.finish();
        return Err(SessionError::Message("Session is closed".to_owned()));
    }
    Ok(Box::new(AdmittedMutation {
        source,
        slot: Some(slot),
    }))
}

/// Runs one mutation callback behind the gate, upstream's `mutate` facade
/// body: the callback body re-checks the open state when it runs, so a
/// callback queued before close still rejects.
#[must_use]
pub fn admitted_mutate(
    gate: &AdmissionGate,
    session: Arc<StorageBackedSession>,
    mutation: SessionMutationCallback,
    context: Context,
) -> BoxedFuture<'static, Result<Box<dyn Any + Send>, SessionError>> {
    let gate = gate.clone();
    let is_open = Arc::clone(&gate.is_open);
    Box::pin(gate.admit(Box::pin(async move {
        session
            .mutate(
                Box::new(
                    move |mutator: &dyn SessionMutator, mutation_context: &Context| {
                        let closed: BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>> =
                            Box::pin(std::future::ready(Err(SessionError::Message(
                                "Session is closed".to_owned(),
                            ))));
                        if !is_open() {
                            return closed;
                        }
                        mutation(mutator, mutation_context)
                    },
                ),
                &context,
            )
            .await
    })))
}

/// The backend close flow one facade runs after the drain: the wrapped
/// session's close plus whatever the backend owns (memory: the on-close
/// hook; sqlite: the database close and the deregistration).
pub type CloseFlow =
    Arc<dyn Fn(&Context) -> BoxedFuture<'static, Result<(), SessionError>> + Send + Sync>;

/// The shared facade lifecycle, upstream's `"open" | "closing" | "closed"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FacadeLifecycle {
    /// Accepting operations.
    Open,
    /// Closing; operations not yet admitted reject.
    Closing,
    /// Closed.
    Closed,
}

/// The facade's shared state over one storage-backed session, Arc'd so the
/// admitted branch objects and mutation callbacks outlive the
/// `Box<dyn Session>` handle they borrow from.
///
/// The backends' facades (`memory.ts`, sqlite-node's `session.ts`) carry the
/// same fields and the same operation forwards; the port shares this core
/// and the backends parameterize only their close flow. Cloning shares the
/// state, upstream's object aliasing.
#[derive(Clone)]
pub struct FacadeCore {
    session: Arc<StorageBackedSession>,
    metadata: SessionMetadata,
    id_generator: Arc<dyn IdGenerator>,
    lifecycle: Arc<Mutex<FacadeLifecycle>>,
    gate: AdmissionGate,
    close_cell: tokio::sync::OnceCell<Result<(), SessionError>>,
    /// The backend's close flow after the drain: the wrapped session's close
    /// plus whatever the backend owns (memory: the on-close hook; sqlite:
    /// the database close and the deregistration).
    close_flow: CloseFlow,
}

impl std::fmt::Debug for FacadeCore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("FacadeCore").finish_non_exhaustive()
    }
}

impl FacadeCore {
    /// Builds the core over one storage-backed session and the backend's
    /// close flow, upstream's facade constructors.
    pub fn new(
        session: Arc<StorageBackedSession>,
        metadata: SessionMetadata,
        close_flow: CloseFlow,
    ) -> Arc<Self> {
        let lifecycle = Arc::new(Mutex::new(FacadeLifecycle::Open));
        let gate_is_open = Arc::clone(&lifecycle);
        let gate = AdmissionGate::new(
            Arc::new(AdmissionTracker::default()),
            Arc::new(move || {
                *gate_is_open.lock().unwrap_or_else(PoisonError::into_inner)
                    == FacadeLifecycle::Open
            }),
        );
        Arc::new(Self {
            id_generator: session.id_generator_arc(),
            session,
            metadata,
            lifecycle,
            gate,
            close_cell: tokio::sync::OnceCell::new(),
            close_flow,
        })
    }

    /// Whether the facade still admits operations.
    #[must_use]
    pub fn is_open(&self) -> bool {
        *self.lifecycle.lock().unwrap_or_else(PoisonError::into_inner) == FacadeLifecycle::Open
    }

    /// Marks the closing state synchronously, upstream's
    /// `this.state = "closing"` inside the close call: operations not yet
    /// admitted reject from the call onward, before the close future polls.
    fn mark_closing(&self) {
        let mut lifecycle = self.lifecycle.lock().unwrap_or_else(PoisonError::into_inner);
        if *lifecycle == FacadeLifecycle::Open {
            *lifecycle = FacadeLifecycle::Closing;
        }
    }

    async fn close_impl(&self, context: &Context) -> Result<(), SessionError> {
        self.close_cell
            .get_or_init(|| async {
                self.gate.drain().await;
                let flow = Arc::clone(&self.close_flow);
                let context = context.clone();
                let close_result = flow(&context).await;
                *self.lifecycle.lock().unwrap_or_else(PoisonError::into_inner) =
                    FacadeLifecycle::Closed;
                close_result
            })
            .await
            .clone()
    }
}

impl SessionReader for FacadeCore {
    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>> {
        self.gate.admit(self.session.get_entries(ids, context))
    }

    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        self.gate.admit(self.session.get_stats(context))
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        self.gate.admit(self.session.get_value(address, context))
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        self.gate.admit(self.session.scan_values(prefix, context))
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<crate::harness::session::values::ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        self.gate.admit(self.session.read_list(address, options, context))
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.gate.admit(self.session.scan_branch(query, context))
    }
}

impl Session for FacadeCore {
    fn metadata(&self) -> &SessionMetadata {
        &self.metadata
    }

    fn id_generator(&self) -> &dyn IdGenerator {
        &*self.id_generator
    }

    fn get_entry(
        &self,
        id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        self.gate.admit(self.session.get_entry(id, context))
    }

    fn get_name(&self, context: &Context) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        self.gate.admit(self.session.get_name(context))
    }

    fn get_label(
        &self,
        target_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        self.gate.admit(self.session.get_label(target_id, context))
    }

    fn find_entries(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.gate.admit(self.session.find_entries(query, context))
    }

    fn find_entry(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        self.gate.admit(self.session.find_entry(query, context))
    }

    fn branch(
        &self,
        name: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Box<dyn Branch>>, SessionError>> {
        let gate = self.gate.clone();
        let name = name.to_owned();
        let context = context.clone();
        Box::pin(async move {
            let branch = gate.admit(self.session.branch(&name, &context)).await?;
            Ok(branch.map(|branch| AdmittedBranch::new(branch, gate)))
        })
    }

    fn create_branch(
        &self,
        name: &str,
        at: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Branch>, SessionError>> {
        let gate = self.gate.clone();
        let name = name.to_owned();
        let context = context.clone();
        Box::pin(async move {
            let branch = gate.admit(self.session.create_branch(&name, at, &context)).await?;
            Ok(AdmittedBranch::new(branch, gate))
        })
    }

    fn begin_mutation(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn SessionMutation>, SessionError>> {
        let context = context.clone();
        Box::pin(async move { begin_admitted_mutation(&self.gate, &self.session, &context).await })
    }

    fn mutate(
        &self,
        mutation: SessionMutationCallback,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>> {
        Box::pin(admitted_mutate(
            &self.gate,
            Arc::clone(&self.session),
            mutation,
            context.clone(),
        ))
    }

    fn set_value(
        &self,
        address: &ValueAddress,
        next: serde_json::Value,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.gate.admit(self.session.set_value(address, next, context))
    }

    fn delete_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.gate.admit(self.session.delete_value(address, context))
    }

    fn append_list(
        &self,
        address: &ListAddress,
        element: serde_json::Value,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.gate.admit(self.session.append_list(address, element, context))
    }

    fn delete_list(
        &self,
        address: &ListAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.gate.admit(self.session.delete_list(address, context))
    }

    fn set_name(
        &self,
        name: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.gate.admit(self.session.set_name(name, context))
    }

    fn set_label(
        &self,
        target_id: &str,
        label: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.gate.admit(self.session.set_label(target_id, label, context))
    }

    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.mark_closing();
        let context = context.clone();
        Box::pin(async move { self.close_impl(&context).await })
    }
}
