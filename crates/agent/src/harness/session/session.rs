//! The storage-backed session, ported from upstream
//! `src/harness/session/session.ts`.
//!
//! It carries the durable session over the
//! [`Storage`] contract — the
//! mutation barrier, the branch surface, and the writer helpers every
//! concrete session repository composes.
//!
#![expect(
    clippy::significant_drop_tightening,
    reason = "the lifecycle lock guards the close transition; the granted mutation's release is ordered against it"
)]
//!

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pi_ai::types::BoxedFuture;

use crate::harness::context::Context;
use crate::harness::session::commit::insert_entry;
use crate::harness::session::mutation_line::MutationLine;
use crate::harness::session::types::{
    Branch, BranchScan, BranchScanOrder, CommitResult, CustomEntryBody, Entry, EntryQuery,
    EntryScan, EntryScanOrder, IdGenerator, MessageEntry, NewEntry, Session, SessionError,
    SessionMetadata, SessionMutation, SessionMutationCallback, SessionMutator, SessionReader,
    SessionStats, Storage, StorageBranchScan,
};
use crate::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, Write, branch_tip,
    entry_label, session_name, set_value as set_value_write,
};
use crate::types::AgentMessage;

/// The session-wide entry query cursor bound upstream's
/// `Number.MAX_SAFE_INTEGER` restates.
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// The options a [`StorageBackedSession`] accepts, upstream's
/// `StorageBackedSessionOptions`.
#[derive(Clone, Default)]
pub struct StorageBackedSessionOptions {
    /// A pre-built mutation line to share, upstream's `mutationLine?`.
    pub mutation_line: Option<MutationLine>,
    /// The id generator to mint entry ids with, upstream's
    /// `idGenerator?`; defaults to the process uuidv7 generator.
    pub id_generator: Option<Arc<dyn IdGenerator>>,
    /// The close callback, upstream's `onClose?`.
    pub on_close: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl std::fmt::Debug for StorageBackedSessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageBackedSessionOptions")
            .finish_non_exhaustive()
    }
}

/// A branch append's payload, upstream's
/// `{ type: "message" | "custom", ... }` append shape.
#[derive(Clone, Debug)]
pub(crate) enum BranchAppend {
    /// A message entry.
    Message(Box<AgentMessage>),
    /// A custom entry.
    Custom {
        /// The custom type discriminator.
        custom_type: String,
        /// The application-defined payload.
        data: Option<serde_json::Value>,
    },
}

/// The session lifecycle, upstream's `"open" | "closing" | "closed"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lifecycle {
    /// Accepting operations.
    Open,
    /// Closing; operations not yet admitted reject.
    Closing,
    /// Closed.
    Closed,
}

/// The shared state one `StorageBackedSession` handle and its branches
/// read; upstream composes the same state through object references.
struct SessionCore {
    metadata: SessionMetadata,
    id_generator: Arc<dyn IdGenerator>,
    storage: Arc<dyn Storage>,
    mutation_line: MutationLine,
    on_close: Option<Arc<dyn Fn() + Send + Sync>>,
    lifecycle: Mutex<Lifecycle>,
    close_cell: tokio::sync::OnceCell<Result<(), SessionError>>,
}

/// The durable session over the Storage contract, upstream's
/// `StorageBackedSession`.
///
/// Upstream parameterizes the class over a metadata extension; the port
/// erases to [`SessionMetadata`] per the contract decision recorded on the
/// harness-foundations child. Upstream caches branch objects per name; the
/// port constructs fresh values per call — the branch surface carries no
/// object identity.
#[derive(Clone)]
pub struct StorageBackedSession {
    core: Arc<SessionCore>,
}

impl std::fmt::Debug for StorageBackedSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageBackedSession")
            .field("metadata", &self.core.metadata)
            .finish_non_exhaustive()
    }
}

/// The write whose entry carries a pending assistant message, upstream's
/// `SessionPendingAssistantMessageError` check.
fn pending_assistant_write_error(writes: &[Write]) -> Option<SessionError> {
    for write in writes {
        if let Write::Entry(entry_write) = write
            && let NewEntry::Message { body, .. } = &entry_write.entry
            && let AgentMessage::Standard(pi_ai::types::Message::Assistant(assistant)) =
                &body.message
            && assistant.stop_reason == pi_ai::types::StopReason::Pending
        {
            return Some(SessionError::PendingAssistantMessage);
        }
    }
    None
}

impl StorageBackedSession {
    /// A session over the storage, upstream's constructor.
    #[must_use]
    pub fn new(
        metadata: SessionMetadata,
        storage: Arc<dyn Storage>,
        options: StorageBackedSessionOptions,
    ) -> Self {
        Self {
            core: Arc::new(SessionCore {
                metadata,
                id_generator: options
                    .id_generator
                    .unwrap_or_else(|| Arc::new(UuidV7IdGenerator)),
                storage,
                mutation_line: options.mutation_line.unwrap_or_default(),
                on_close: options.on_close,
                lifecycle: Mutex::new(Lifecycle::Open),
                close_cell: tokio::sync::OnceCell::new(),
            }),
        }
    }

    fn lifecycle(&self) -> Lifecycle {
        *self
            .core
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The id generator as a shared handle, for facades re-exporting it,
    /// upstream's `session.idGenerator` reference.
    pub(crate) fn id_generator_arc(&self) -> Arc<dyn IdGenerator> {
        Arc::clone(&self.core.id_generator)
    }

    fn assert_open(&self) -> Result<(), SessionError> {
        if self.lifecycle() != Lifecycle::Open {
            return Err(SessionError::Message("Session is closed".to_owned()));
        }
        Ok(())
    }

    /// The branch tip's entry id, upstream's `getBranchTip`.
    ///
    /// # Errors
    /// A `SessionError::Invariant` when the branch is unknown; read
    /// failures from the storage.
    pub async fn get_branch_tip(
        &self,
        name: &str,
        context: &Context,
    ) -> Result<Option<String>, SessionError> {
        let Some(stored) = self
            .core
            .storage
            .get_value(&branch_tip(name).address, context)
            .await?
        else {
            return Err(SessionError::Invariant(format!("Unknown branch: {name}")));
        };
        Ok(stored.value.as_str().map(str::to_owned))
    }

    /// Append one message or custom entry at the named branch's tip,
    /// upstream's `appendToBranch`.
    ///
    /// # Errors
    /// A `SessionError::PendingAssistantMessage` for a pending assistant
    /// message, a `SessionError::Invariant` for an unknown branch, and
    /// the mutation/commit failures.
    pub(crate) async fn append_to_branch(
        &self,
        name: &str,
        entry: BranchAppend,
        context: &Context,
    ) -> Result<String, SessionError> {
        self.assert_open()?;
        if let BranchAppend::Message(message) = &entry
            && let AgentMessage::Standard(pi_ai::types::Message::Assistant(assistant)) = &**message
            && assistant.stop_reason == pi_ai::types::StopReason::Pending
        {
            return Err(SessionError::PendingAssistantMessage);
        }
        let id = self.core.id_generator.next(None);
        let name = name.to_owned();
        let append_id = id.clone();
        self.mutate(
            Box::new(
                move |mutator: &dyn SessionMutator,
                      context: &Context|
                -> BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>> {
                    let name = name.clone();
                    let entry = entry.clone();
                    let id = id.clone();
                    Box::pin(async move {
                        let Some(tip) = mutator
                            .get_value(&branch_tip(&name).address, context)
                            .await?
                        else {
                            return Err(SessionError::Invariant(format!(
                                "Unknown branch: {name}"
                            )));
                        };
                        let parent_id = tip.value.as_str().map(str::to_owned);
                        let new_entry = match entry {
                            BranchAppend::Message(message) => NewEntry::Message {
                                id: id.clone(),
                                parent_id,
                                body: Box::new(MessageEntry {
                                    message: *message,
                                    terminate: None,
                                }),
                            },
                            BranchAppend::Custom { custom_type, data } => NewEntry::Custom {
                                id: id.clone(),
                                parent_id,
                                body: CustomEntryBody { custom_type, data },
                            },
                        };
                        mutator
                            .commit(
                                vec![
                                    Write::Entry(Box::new(insert_entry(new_entry))),
                                    Write::ValueSet(set_value_write(
                                        &branch_tip(&name),
                                        Some(id.clone()),
                                    )?),
                                ],
                                context,
                            )
                            .await?;
                        let done: Box<dyn Any + Send> = Box::new(());
                        Ok(done)
                    })
                },
            ),
            context,
        )
        .await?;
        Ok(append_id)
    }

    /// One committed write through the exclusive mutator, the shape every
    /// single-write session method and the conformance suites share,
    /// upstream's `mutate((mutator) => mutator.commit([write]))` closures.
    #[must_use]
    pub fn commit_writes_callback(writes: Vec<Write>) -> SessionMutationCallback {
        Box::new(
            move |mutator: &dyn SessionMutator,
                  context: &Context|
                  -> BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>> {
                let writes = writes.clone();
                Box::pin(async move {
                    mutator.commit(writes, context).await.map(|result| {
                        let done: Box<dyn Any + Send> = Box::new(result);
                        done
                    })
                })
            },
        )
    }

    fn assert_valid_branch_name(name: &str) -> Result<(), SessionError> {
        if name.is_empty() {
            return Err(SessionError::InvalidBranch {
                branch: name.to_owned(),
                reason: "branch name must not be empty".to_owned(),
            });
        }
        if name.contains('\u{0}') {
            return Err(SessionError::InvalidBranch {
                branch: name.to_owned(),
                reason: "branch name must not contain \\u0000".to_owned(),
            });
        }
        Ok(())
    }
}

/// The default id generator over the process uuidv7 generator, upstream's
/// `{ next: uuidv7 }` object.
struct UuidV7IdGenerator;

impl IdGenerator for UuidV7IdGenerator {
    fn next(&self, timestamp_ms: Option<i64>) -> String {
        #[expect(
            clippy::panic,
            reason = "the IdGenerator contract is infallible like upstream's, whose uuidv7 throws for out-of-range timestamps; the same range failure surfaces as a panic here"
        )]
        match timestamp_ms.map(u64::try_from) {
            Some(Ok(timestamp)) => pi_ai::utils::uuid::uuidv7(Some(timestamp)),
            None => pi_ai::utils::uuid::uuidv7(None),
            Some(Err(_)) => Err(pi_ai::utils::uuid::UuidV7Error::TimestampOutOfRange),
        }
        .unwrap_or_else(|error| panic!("uuidv7 failed: {error}"))
    }
}

/// The granted exclusive mutation, upstream's
/// `StorageBackedSessionMutation`.
struct StorageBackedSessionMutation {
    storage: Arc<dyn Storage>,
    line_guard: Mutex<Option<crate::harness::session::mutation_line::MutationLineGuard>>,
    active: AtomicBool,
    attempted: AtomicBool,
    outcome: Mutex<Option<Result<CommitResult, SessionError>>>,
    settled: tokio::sync::Notify,
}

impl StorageBackedSessionMutation {
    fn assert_active(&self) -> Result<(), SessionError> {
        if !self.active.load(Ordering::Acquire) {
            return Err(SessionError::Message(
                "SessionMutator cannot be used outside its mutation callback".to_owned(),
            ));
        }
        Ok(())
    }

    fn set_outcome(&self, outcome: Result<CommitResult, SessionError>) {
        let mut slot = self.outcome.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(outcome);
            self.settled.notify_waiters();
        }
    }

    /// Wait for any commit attempt's settlement, upstream's `settle`: no
    /// attempt settles immediately.
    async fn settle(&self) {
        if !self.attempted.load(Ordering::Acquire) {
            return;
        }
        loop {
            if self
                .outcome
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_some()
            {
                return;
            }
            self.settled.notified().await;
        }
    }
}

impl SessionReader for StorageBackedSessionMutation {
    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
        if let Err(error) = self.assert_active() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.storage.get_entries(ids, context))
    }

    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        if let Err(error) = self.assert_active() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.storage.get_stats(context))
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        if let Err(error) = self.assert_active() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.storage.get_value(address, context))
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        if let Err(error) = self.assert_active() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.storage.scan_values(prefix, context))
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        if let Err(error) = self.assert_active() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.storage.read_list(address, options, context))
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        if let Err(error) = self.assert_active() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.storage.scan_branch(query, context))
    }
}

impl SessionMutator for StorageBackedSessionMutation {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        // Upstream runs these checks synchronously in the commit call and
        // records the attempt (including a failed one) immediately; the
        // guard consumption is observable after either failure path.
        if let Err(error) = self.assert_active() {
            return Box::pin(std::future::ready(Err(error)));
        }
        if self.attempted.swap(true, Ordering::AcqRel) {
            return Box::pin(std::future::ready(Err(SessionError::Message(
                "SessionMutator commit already attempted".to_owned(),
            ))));
        }
        if let Some(pending) = pending_assistant_write_error(&writes) {
            self.set_outcome(Err(pending.clone()));
            return Box::pin(std::future::ready(Err(pending)));
        }
        let storage = self.storage.clone();
        let context = context.clone();
        Box::pin(CommitAttemptFuture {
            inner: Box::pin(async move { storage.commit(writes, &context).await }),
            mutation: self,
            settled: false,
        })
    }
}

/// One commit attempt's future: the wrapper records the outcome (or the
/// abandonment) on the mutation, upstream's `commitResult` promise that
/// settles even when nobody polls it.
struct CommitAttemptFuture<'a> {
    inner: BoxedFuture<'a, Result<CommitResult, SessionError>>,
    mutation: &'a StorageBackedSessionMutation,
    settled: bool,
}

impl Future for CommitAttemptFuture<'_> {
    type Output = Result<CommitResult, SessionError>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        match this.inner.as_mut().poll(cx) {
            std::task::Poll::Ready(outcome) => {
                this.settled = true;
                this.mutation.set_outcome(outcome.clone());
                std::task::Poll::Ready(outcome)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl Drop for CommitAttemptFuture<'_> {
    fn drop(&mut self) {
        if !self.settled {
            self.mutation.set_outcome(Err(SessionError::Message(
                "commit future dropped before settling".to_owned(),
            )));
        }
    }
}

impl SessionMutation for StorageBackedSessionMutation {
    fn end(&self, _context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.active.store(false, Ordering::Release);
        Box::pin(async move {
            self.settle().await;
            drop(
                self.line_guard
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take(),
            );
            Ok(())
        })
    }
}

/// One named branch's write surface over the durable session, upstream's
/// `StorageBackedBranch`.
struct StorageBackedBranch {
    name: String,
    session: StorageBackedSession,
}

impl Branch for StorageBackedBranch {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_tip_id(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        let context = context.clone();
        Box::pin(async move { self.session.get_branch_tip(&self.name, &context).await })
    }

    fn find_entries(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        let query = query.cloned().unwrap_or_default();
        let session = self.session.clone();
        let name = self.name.clone();
        let context = context.clone();
        Box::pin(async move {
            let start = match query.start.clone() {
                Some(start) => Some(start),
                None => session.get_branch_tip(&name, &context).await?,
            };
            let Some(start) = start else {
                return Ok(Vec::new());
            };
            session
                .core
                .storage
                .scan_branch(
                    &StorageBranchScan {
                        start,
                        stop_at_type: query.stop_at_type,
                        stop_at_id: query.stop_at_id,
                        kind: query.kind,
                        custom_type: query.custom_type,
                        order: query.order.or(Some(BranchScanOrder::NewestFirst)),
                        limit: query.limit,
                        cursor: query.cursor,
                    },
                    &context,
                )
                .await
        })
    }

    fn find_entry(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        let query = query.cloned().unwrap_or_default();
        let context = context.clone();
        let query = BranchScan {
            limit: Some(query.limit.unwrap_or(1).min(1)),
            ..query
        };
        Box::pin(async move {
            Ok(self
                .find_entries(Some(&query), &context)
                .await?
                .into_iter()
                .next())
        })
    }

    fn append_message(
        &self,
        message: AgentMessage,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, SessionError>> {
        let session = self.session.clone();
        let name = self.name.clone();
        let context = context.clone();
        Box::pin(async move {
            session
                .append_to_branch(&name, BranchAppend::Message(Box::new(message)), &context)
                .await
        })
    }

    fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, SessionError>> {
        let session = self.session.clone();
        let name = self.name.clone();
        let context = context.clone();
        let custom_type = custom_type.to_owned();
        Box::pin(async move {
            session
                .append_to_branch(&name, BranchAppend::Custom { custom_type, data }, &context)
                .await
        })
    }
}

impl SessionReader for StorageBackedSession {
    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.core.storage.get_entries(ids, context))
    }

    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.core.storage.get_stats(context))
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.core.storage.get_value(address, context))
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.core.storage.scan_values(prefix, context))
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.core.storage.read_list(address, options, context))
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        Box::pin(self.core.storage.scan_branch(query, context))
    }
}

impl Session for StorageBackedSession {
    fn metadata(&self) -> &SessionMetadata {
        &self.core.metadata
    }

    fn id_generator(&self) -> &dyn IdGenerator {
        &*self.core.id_generator
    }

    fn get_entry(
        &self,
        id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        let id = id.to_owned();
        let context = context.clone();
        Box::pin(async move {
            Ok(self
                .get_entries(vec![id.clone()], &context)
                .await?
                .get(&id)
                .cloned())
        })
    }

    fn get_name(&self, context: &Context) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        let context = context.clone();
        Box::pin(async move {
            Ok(self
                .get_value(&session_name().address, &context)
                .await?
                .and_then(|stored| stored.value.as_str().map(str::to_owned)))
        })
    }

    fn get_label(
        &self,
        target_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        let address = entry_label(target_id).address;
        let context = context.clone();
        Box::pin(async move {
            Ok(self
                .get_value(&address, &context)
                .await?
                .and_then(|stored| stored.value.as_str().map(str::to_owned)))
        })
    }

    fn find_entries(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        let query = query.cloned().unwrap_or_default();
        let context = context.clone();
        Box::pin(async move {
            let order = query.order.unwrap_or(EntryScanOrder::Desc);
            if let Some(cursor) = query.cursor {
                if order == EntryScanOrder::Asc && cursor.seq == MAX_SAFE_INTEGER {
                    return Ok(Vec::new());
                }
                if order == EntryScanOrder::Desc && cursor.seq <= 1 {
                    return Ok(Vec::new());
                }
            }
            let (from_seq, to_seq) = query.cursor.map_or((None, None), |cursor| match order {
                EntryScanOrder::Asc => (Some(cursor.seq + 1), None),
                EntryScanOrder::Desc => (None, Some(cursor.seq - 1)),
            });
            self.core
                .storage
                .scan_entries(
                    &EntryScan {
                        kind: query.kind,
                        custom_type: query.custom_type,
                        from_seq,
                        to_seq,
                        order: Some(order),
                        limit: query.limit,
                    },
                    &context,
                )
                .await
        })
    }

    fn find_entry(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        let query = query.cloned().unwrap_or_default();
        let context = context.clone();
        Box::pin(async move {
            Ok(self
                .find_entries(Some(&query), &context)
                .await?
                .into_iter()
                .next())
        })
    }

    fn branch(
        &self,
        name: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Box<dyn Branch>>, SessionError>> {
        if let Err(error) = Self::assert_valid_branch_name(name) {
            return Box::pin(std::future::ready(Err(error)));
        }
        let name = name.to_owned();
        let context = context.clone();
        Box::pin(async move {
            if self
                .get_value(&branch_tip(&name).address, &context)
                .await?
                .is_none()
            {
                return Ok(None);
            }
            let branch: Box<dyn Branch> = Box::new(StorageBackedBranch {
                name,
                session: self.clone(),
            });
            Ok(Some(branch))
        })
    }

    fn create_branch(
        &self,
        name: &str,
        at: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Branch>, SessionError>> {
        if let Err(error) = self
            .assert_open()
            .and_then(|()| Self::assert_valid_branch_name(name))
        {
            return Box::pin(std::future::ready(Err(error)));
        }
        let name = name.to_owned();
        let context = context.clone();
        let branch_name = name.clone();
        Box::pin(async move {
            self.mutate(
                Box::new(
                    move |mutator: &dyn SessionMutator,
                          context: &Context|
                      -> BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>> {
                        let name = name.clone();
                        let at = at.clone();
                        Box::pin(async move {
                            if mutator
                                .get_value(&branch_tip(&name).address, context)
                                .await?
                                .is_some()
                            {
                                return Err(SessionError::BranchExists {
                                    branch: name.clone(),
                                });
                            }
                            if let Some(at) = &at
                                && !mutator.get_entries(vec![at.clone()], context).await?.contains_key(at)
                            {
                                return Err(SessionError::UnknownTarget {
                                    target_id: at.clone(),
                                });
                            }
                            mutator
                                .commit(
                                    vec![Write::ValueSet(set_value_write(
                                        &branch_tip(&name),
                                        at.clone(),
                                    )?)],
                                    context,
                                )
                                .await?;
                            let done: Box<dyn Any + Send> = Box::new(());
                            Ok(done)
                        })
                    },
                ),
                &context,
            )
            .await?;
            let branch: Box<dyn Branch> = Box::new(StorageBackedBranch {
                name: branch_name,
                session: self.clone(),
            });
            Ok(branch)
        })
    }

    fn begin_mutation(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn SessionMutation>, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        let storage = self.core.storage.clone();
        let line = self.core.mutation_line.clone();
        Box::pin(async move {
            let line_guard = line.acquire().await?;
            let granted: Box<dyn SessionMutation> = Box::new(StorageBackedSessionMutation {
                storage,
                line_guard: Mutex::new(Some(line_guard)),
                active: AtomicBool::new(true),
                attempted: AtomicBool::new(false),
                outcome: Mutex::new(None),
                settled: tokio::sync::Notify::new(),
            });
            Ok(granted)
        })
    }

    fn mutate(
        &self,
        mutation: SessionMutationCallback,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>> {
        if let Err(error) = self.assert_open() {
            return Box::pin(std::future::ready(Err(error)));
        }
        let context = context.clone();
        Box::pin(async move {
            let mutator = self.begin_mutation(&context).await?;
            let outcome = mutation(&*mutator, &context).await;
            let ended = mutator.end(&context).await;
            match outcome {
                Ok(value) => ended.map(|()| value),
                Err(error) => {
                    ended?;
                    Err(error)
                }
            }
        })
    }

    fn set_value(
        &self,
        address: &ValueAddress,
        next: serde_json::Value,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        let write = Write::ValueSet(crate::harness::session::values::ValueSetWrite {
            kind: "value".to_owned(),
            op: "set".to_owned(),
            namespace: address.namespace.clone(),
            key: address.key.clone(),
            value: next,
        });
        let context = context.clone();
        Box::pin(async move {
            self.mutate(Self::commit_writes_callback(vec![write]), &context)
                .await
                .map(|_| ())
        })
    }

    fn delete_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        let write = Write::ValueDelete(crate::harness::session::values::ValueDeleteWrite {
            kind: "value".to_owned(),
            op: "delete".to_owned(),
            namespace: address.namespace.clone(),
            key: address.key.clone(),
        });
        let context = context.clone();
        Box::pin(async move {
            self.mutate(Self::commit_writes_callback(vec![write]), &context)
                .await
                .map(|_| ())
        })
    }

    fn append_list(
        &self,
        address: &ListAddress,
        element: serde_json::Value,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        let write = Write::ListAppend(crate::harness::session::values::ListAppendWrite {
            kind: "list".to_owned(),
            op: "append".to_owned(),
            namespace: address.namespace.clone(),
            key: address.key.clone(),
            value: element,
        });
        let context = context.clone();
        Box::pin(async move {
            self.mutate(Self::commit_writes_callback(vec![write]), &context)
                .await
                .map(|_| ())
        })
    }

    fn delete_list(
        &self,
        address: &ListAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        let write = Write::ListDelete(crate::harness::session::values::ListDeleteWrite {
            kind: "list".to_owned(),
            op: "delete".to_owned(),
            namespace: address.namespace.clone(),
            key: address.key.clone(),
        });
        let context = context.clone();
        Box::pin(async move {
            self.mutate(Self::commit_writes_callback(vec![write]), &context)
                .await
                .map(|_| ())
        })
    }

    fn set_name(
        &self,
        name: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        let context = context.clone();
        Box::pin(async move {
            match name {
                None => self.delete_value(&session_name().address, &context).await,
                Some(name) => {
                    self.set_value(&session_name().address, name.into(), &context)
                        .await
                }
            }
        })
    }

    fn set_label(
        &self,
        target_id: &str,
        label: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        let address = entry_label(target_id).address;
        let context = context.clone();
        Box::pin(async move {
            match label {
                None => self.delete_value(&address, &context).await,
                Some(label) => self.set_value(&address, label.into(), &context).await,
            }
        })
    }

    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        let lifecycle = self.lifecycle();
        if lifecycle == Lifecycle::Open {
            *self
                .core
                .lifecycle
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closing;
        }
        self.core
            .mutation_line
            .seal(SessionError::Message("Session is closed".to_owned()));
        let core = self.core.clone();
        let context = context.clone();
        Box::pin(async move {
            core.close_cell
                .get_or_init(|| async {
                    core.mutation_line.drain().await;
                    let closed = core.storage.close(&context).await;
                    *core
                        .lifecycle
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closed;
                    if let Some(on_close) = &core.on_close {
                        on_close();
                    }
                    closed
                })
                .await
                .clone()
        })
    }
}
