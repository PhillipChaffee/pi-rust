//! The SQLite open-session facade, upstream's `src/sqlite/session.ts`.
//!
//! `SqliteOpenSession` wraps the storage-backed session with the admission
//! machinery from `pi-agent-core`'s facade module (upstream's `admitted`
//! set), and `close`
//! waits for every admitted operation, closes the wrapped session, then
//! closes the SQLite database — whose error supersedes the inner one,
//! upstream's `.finally` throw semantics — and always deregisters,
//! upstream's `finally` block.

use std::any::Any;
use std::sync::{Arc, Mutex, PoisonError};

use pi_agent_core::harness::context::Context;
use pi_agent_core::harness::session::facade::{
    AdmissionGate, AdmissionTracker, admitted_mutate, begin_admitted_mutation,
};
use pi_agent_core::harness::session::session::StorageBackedSession;
use pi_agent_core::harness::session::types::{
    Branch, Entry, EntryQuery, IdGenerator, Session, SessionError, SessionMetadata,
    SessionMutation, SessionMutationCallback, SessionReader, SessionStats, StorageBranchScan,
};
use pi_agent_core::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress,
};
use pi_agent_core::types::BoxedFuture;

use crate::sqlite::session_row::SqliteSessionMetadata;
use crate::sqlite::types::SqliteAdapterError;

/// The facade options, upstream's `SqliteOpenSessionOptions`.
pub struct SqliteOpenSessionOptions {
    /// Runs when the facade has closed the wrapped session and the SQLite
    /// database: the repository's deregistration, upstream's `onClose`.
    ///
    /// The database close rides the facade (upstream's repo callback closes
    /// the db there and its rejection propagates through `session.close`);
    /// this callback carries the cleanup that must always run after it.
    pub on_close: Arc<dyn Fn() + Send + Sync>,
    /// Closes the SQLite database, upstream's db close inside the repo
    /// callback; its error supersedes the wrapped session's close error.
    pub close_database: Arc<dyn Fn() -> Result<(), SqliteAdapterError> + Send + Sync>,
}

impl std::fmt::Debug for SqliteOpenSessionOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteOpenSessionOptions")
            .finish_non_exhaustive()
    }
}

/// The facade lifecycle, upstream's `"open" | "closing" | "closed"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lifecycle {
    /// Accepting operations.
    Open,
    /// Closing; operations not yet admitted reject.
    Closing,
    /// Closed.
    Closed,
}

/// The facade's shared state, Arc'd so the admitted branch objects and
/// mutation callbacks outlive the `Box<dyn Session>` handle they borrow
/// from, upstream's `SqliteOpenSession` fields.
struct FacadeInner {
    session: Arc<StorageBackedSession>,
    metadata: SqliteSessionMetadata,
    id_generator: Arc<dyn IdGenerator>,
    lifecycle: Arc<Mutex<Lifecycle>>,
    gate: AdmissionGate,
    options: SqliteOpenSessionOptions,
    close_cell: tokio::sync::OnceCell<Result<(), SessionError>>,
}

impl std::fmt::Debug for FacadeInner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FacadeInner")
            .finish_non_exhaustive()
    }
}

/// The SQLite open-session lifecycle wrapper, upstream's `SqliteOpenSession`.
///
/// Cloning shares the facade state, upstream's object aliasing between the
/// repository's registry and the caller's handle.
#[derive(Clone)]
pub struct SqliteOpenSession(Arc<FacadeInner>);

impl std::fmt::Debug for SqliteOpenSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteOpenSession")
            .finish_non_exhaustive()
    }
}

impl SqliteOpenSession {
    /// Wraps one storage-backed session, upstream's constructor: the typed
    /// metadata rides in from the repo because the port's
    /// `StorageBackedSession` erases it to the base shape.
    #[must_use]
    pub fn new(
        session: Arc<StorageBackedSession>,
        metadata: SqliteSessionMetadata,
        options: SqliteOpenSessionOptions,
    ) -> Self {
        let id_generator = session.id_generator_arc();
        let lifecycle = Arc::new(Mutex::new(Lifecycle::Open));
        let gate_is_open = Arc::clone(&lifecycle);
        let gate = AdmissionGate::new(
            Arc::new(AdmissionTracker::default()),
            Arc::new(move || {
                *gate_is_open.lock().unwrap_or_else(PoisonError::into_inner) == Lifecycle::Open
            }),
        );
        Self(Arc::new(FacadeInner {
            metadata,
            id_generator,
            gate,
            session,
            lifecycle,
            options,
            close_cell: tokio::sync::OnceCell::new(),
        }))
    }

    /// The session id, the deregistration's removal key.
    pub(crate) fn session_id(&self) -> &str {
        &self.0.metadata.base.id
    }

    async fn close_impl(&self, context: &Context) -> Result<(), SessionError> {
        let result = self
            .0
            .close_cell
            .get_or_init(|| async {
                *self
                    .0
                    .lifecycle
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closing;
                self.0.gate.drain().await;
                let inner_result = self.0.session.close(context).await;
                // The database close and the deregistration always run,
                // upstream's `.finally`; a database error supersedes the
                // wrapped session's.
                let close_result = (self.0.options.close_database)();
                (self.0.options.on_close)();
                *self
                    .0
                    .lifecycle
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closed;
                match close_result {
                    Err(database_error) => {
                        Err(SessionError::Message(database_error.message().to_owned()))
                    }
                    Ok(()) => inner_result,
                }
            })
            .await;
        result.clone()
    }
}

impl SessionReader for SqliteOpenSession {
    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
        self.0.gate.admit(self.0.session.get_entries(ids, context))
    }

    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        self.0.gate.admit(self.0.session.get_stats(context))
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        self.0
            .gate
            .admit(self.0.session.get_value(address, context))
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        self.0
            .gate
            .admit(self.0.session.scan_values(prefix, context))
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        self.0
            .gate
            .admit(self.0.session.read_list(address, options, context))
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.0
            .gate
            .admit(self.0.session.scan_branch(query, context))
    }
}

impl Session for SqliteOpenSession {
    fn metadata(&self) -> &SessionMetadata {
        &self.0.metadata.base
    }

    fn id_generator(&self) -> &dyn IdGenerator {
        &*self.0.id_generator
    }

    fn get_entry(
        &self,
        id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        self.0.gate.admit(self.0.session.get_entry(id, context))
    }

    fn get_name(&self, context: &Context) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        self.0.gate.admit(self.0.session.get_name(context))
    }

    fn get_label(
        &self,
        target_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        self.0
            .gate
            .admit(self.0.session.get_label(target_id, context))
    }

    fn find_entries(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.0
            .gate
            .admit(self.0.session.find_entries(query, context))
    }

    fn find_entry(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        self.0.gate.admit(self.0.session.find_entry(query, context))
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
            let branch = inner
                .gate
                .admit(inner.session.branch(&name, &context))
                .await?;
            Ok(branch.map(|branch| {
                pi_agent_core::harness::session::facade::AdmittedBranch::new(
                    branch,
                    inner.gate.clone(),
                )
            }))
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
            let branch = inner
                .gate
                .admit(inner.session.create_branch(&name, at, &context))
                .await?;
            Ok(
                pi_agent_core::harness::session::facade::AdmittedBranch::new(
                    branch,
                    inner.gate.clone(),
                ),
            )
        })
    }

    fn begin_mutation(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn SessionMutation>, SessionError>> {
        let inner = self.0.clone();
        let context = context.clone();
        Box::pin(
            async move { begin_admitted_mutation(&inner.gate, &inner.session, &context).await },
        )
    }

    fn mutate(
        &self,
        mutation: SessionMutationCallback,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>> {
        Box::pin(admitted_mutate(
            &self.0.gate,
            Arc::clone(&self.0.session),
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
        self.0
            .gate
            .admit(self.0.session.set_value(address, next, context))
    }

    fn delete_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.0
            .gate
            .admit(self.0.session.delete_value(address, context))
    }

    fn append_list(
        &self,
        address: &ListAddress,
        element: serde_json::Value,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.0
            .gate
            .admit(self.0.session.append_list(address, element, context))
    }

    fn delete_list(
        &self,
        address: &ListAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.0
            .gate
            .admit(self.0.session.delete_list(address, context))
    }

    fn set_name(
        &self,
        name: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.0.gate.admit(self.0.session.set_name(name, context))
    }

    fn set_label(
        &self,
        target_id: &str,
        label: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.0
            .gate
            .admit(self.0.session.set_label(target_id, label, context))
    }

    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        let context = context.clone();
        Box::pin(async move { self.close_impl(&context).await })
    }
}
