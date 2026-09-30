//! The SQLite open-session facade, upstream's `src/sqlite/session.ts`.
//!
//! `SqliteOpenSession` wraps the storage-backed session with the admission
//! machinery from `pi-agent-core`'s facade module (upstream's `admitted`
//! set), and `close` waits for every admitted operation, closes the wrapped
//! session, then closes the SQLite database — whose error supersedes the
//! inner one, upstream's `.finally` throw semantics — and always
//! deregisters, upstream's `finally` block.

use std::sync::Arc;

use pi_agent_core::harness::session::facade::FacadeCore;
use pi_agent_core::harness::session::session::StorageBackedSession;
use pi_agent_core::harness::session::types::{Session, SessionError};
use crate::sqlite::session_row::SqliteSessionMetadata;
use crate::sqlite::types::SqliteAdapterError;

/// The facade options, upstream's `SqliteOpenSessionOptions`.
#[derive(Clone)]
pub struct SqliteOpenSessionOptions {
    /// Closes the SQLite database, upstream's db close inside the repo
    /// callback; the facade runs it after the wrapped session's close, and
    /// its error supersedes the session's.
    pub close_database: Arc<dyn Fn() -> Result<(), SqliteAdapterError> + Send + Sync>,
    /// Runs after the database close, the repository's deregistration,
    /// upstream's `finally` block (it must run even when the close fails).
    pub on_close: Arc<dyn Fn() + Send + Sync>,
}

impl std::fmt::Debug for SqliteOpenSessionOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("SqliteOpenSessionOptions").finish_non_exhaustive()
    }
}

/// The SQLite open-session lifecycle wrapper, upstream's `SqliteOpenSession`.
///
/// Cloning shares the facade state, upstream's object aliasing between the
/// repository's registry and the caller's handle.
#[derive(Clone)]
pub struct SqliteOpenSession(Arc<FacadeCore>);

impl std::fmt::Debug for SqliteOpenSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("SqliteOpenSession").finish_non_exhaustive()
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
        let close_database = options.close_database.clone();
        let on_close = options.on_close.clone();
        let core = FacadeCore::new(
            Arc::clone(&session),
            metadata.base,
            Arc::new(move |context| {
                let session = Arc::clone(&session);
                let close_database = Arc::clone(&close_database);
                let on_close = Arc::clone(&on_close);
                let context = context.clone();
                Box::pin(async move {
                    let inner_result = session.close(&context).await;
                    // The database close and the deregistration always run,
                    // upstream's `.finally`; a database error supersedes the
                    // wrapped session's.
                    let close_result = (close_database)();
                    (on_close)();
                    match close_result {
                        Err(database_error) => {
                            Err(SessionError::Message(database_error.message().to_owned()))
                        }
                        Ok(()) => inner_result,
                    }
                })
            }),
        );
        SqliteOpenSession(core)
    }

    /// The session id, the deregistration's removal key.
    pub(crate) fn session_id(&self) -> &str {
        &self.0.metadata().id
    }

    /// The erased session handle over the shared core, the repository's
    /// `SessionRepo` adapter's return value.
    pub(crate) fn erased_session(&self) -> Box<dyn Session> {
        Box::new(FacadeCore::clone(&self.0))
    }
}

impl std::ops::Deref for SqliteOpenSession {
    type Target = FacadeCore;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}