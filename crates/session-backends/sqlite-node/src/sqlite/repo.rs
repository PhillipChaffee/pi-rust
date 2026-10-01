//! The SQLite session repository lifecycle, upstream's
//! `src/sqlite/repo.ts`.
//!
//! The typed lifecycle carries [`SqliteSessionMetadata`]; the erased
//! `SessionRepo` surface reconstructs the typed metadata by deriving the
//! candidate path from the repository's own layout, since the erased
//! metadata carries no path. The outside-this-repository guard is
//! typed-surface-only (a derived path is always inside the repository); the
//! erased fork canonicalizes its derived source path so a source open in
//! this repository still resolves through the queued-snapshot path.

use base64::Engine as _;
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pi_agent_core::harness::context::Context;
use pi_agent_core::harness::session::fork::{
    ForkDestinationSnapshot, ForkSourceSnapshot, create_fork_snapshot,
};
use pi_agent_core::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use pi_agent_core::harness::session::types::{
    Entry, EntryType, ForkOptions, Session, SessionCreateOptions, SessionError, SessionMetadata,
    SessionRepo,
};
use pi_agent_core::harness::session::values::StoredValue;
use pi_agent_core::types::BoxedFuture;

use crate::sql;
use crate::sqlite::branch_entries::append_entry_to_branch_index;
use crate::sqlite::entries::EntryRowWriter;
use crate::sqlite::migrations::apply_initial_schema;
use crate::sqlite::session::SqliteOpenSession;
use crate::sqlite::session_row::{
    SqliteSessionMetadata, delete_session_rows, has_session_row, insert_session_row,
    metadata_from_session_row, read_all_session_rows, read_session_row,
};
use crate::sqlite::storage::{NowFn, SqliteStorage, SqliteStorageOptions, SqliteStorageSnapshot};
use crate::sqlite::types::sql_integer;
use crate::sqlite::types::{
    SqliteAdapterError, SqliteDatabase, SqliteDatabaseFactory, SqliteTransactionOutcome,
};
use crate::sqlite::values::{read_all_scalar_value_rows, set_scalar_value_row};

/// The schema version this backend writes, upstream's
/// `SQLITE_STORAGE_VERSION`.
pub const SQLITE_STORAGE_VERSION: u32 = 1;

/// The container filename extension, upstream's `SQLITE_SESSION_EXTENSION`.
pub const SQLITE_SESSION_EXTENSION: &str = ".sqlite";

const FIRST_AVAILABLE_COMMIT_SEQ: u64 = 1;

/// The create options, upstream's `SqliteSessionCreateOptions` (an alias for
/// agent-core's `SessionCreateOptions`).
pub type SqliteSessionCreateOptions = SessionCreateOptions;

/// The repository options, upstream's `SqliteSessionRepoOptions`.
#[derive(Clone)]
pub struct SqliteSessionRepoOptions {
    /// The directory the per-session containers live under.
    pub directory: String,
    /// Optional single container path; defaults to one encoded `{id}.sqlite`
    /// file per session under the directory, upstream's `databasePath`.
    pub database_path: Option<String>,
    /// The connection factory, upstream's `databaseFactory`.
    pub database_factory: Arc<dyn SqliteDatabaseFactory>,
    /// The id/clock source; defaults to the wall clock, upstream's `now`.
    pub now: Option<NowFn>,
}

impl std::fmt::Debug for SqliteSessionRepoOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteSessionRepoOptions")
            .field("directory", &self.directory)
            .field("database_path", &self.database_path)
            .finish_non_exhaustive()
    }
}

/// The repository close error, upstream's rethrow semantics: a lone session
/// close failure rethrows that error alone, two or more wrap in
/// `AggregateError(errors, "Failed to close SQLite Sessions")`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SqliteRepoCloseError {
    /// One session's close failure, rethrown alone.
    Single(SessionError),
    /// Two or more failures, upstream's `AggregateError`.
    Aggregate(Vec<SessionError>),
}

impl std::fmt::Display for SqliteRepoCloseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(error) => write!(formatter, "{error}"),
            Self::Aggregate(_) => {
                write!(formatter, "Failed to close SQLite Sessions")
            }
        }
    }
}

impl std::error::Error for SqliteRepoCloseError {}

/// The fork snapshot one destination writes, upstream's `ForkSnapshot`.
struct ForkSnapshot {
    entries: Vec<Entry>,
    scalar_values: Vec<StoredValue>,
    message_count: u64,
    next_seq: u64,
}

/// The session id's filename safety gate, upstream's
/// `/^[A-Za-z0-9_-]+$/`: the regex requires at least one character, so the
/// empty id takes the encoded branch.
fn is_safe_session_file_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// The container filename for one session id, upstream's `sessionFileName`:
/// safe ids keep `{id}.sqlite`; every other id encodes as `~` plus a
/// base64url of the id's UTF-16 code units.
#[must_use]
pub fn session_file_name(id: &str) -> String {
    if is_safe_session_file_id(id) {
        return format!("{id}{SQLITE_SESSION_EXTENSION}");
    }
    let utf16le: Vec<u8> = id.encode_utf16().flat_map(u16::to_le_bytes).collect();
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(utf16le);
    format!("~{encoded}{SQLITE_SESSION_EXTENSION}")
}

/// The container path for one session id, upstream's `sessionPath`.
#[must_use]
pub fn session_path(directory: &str, id: &str) -> String {
    Path::new(directory)
        .join(session_file_name(id))
        .to_string_lossy()
        .into_owned()
}

/// The open-storage identity key, upstream's
/// `JSON.stringify([path, sessionId])`.
fn storage_identity(path: &str, session_id: &str) -> String {
    #[expect(
        clippy::expect_used,
        reason = "a two-element array of strings always serializes; there is no fallback shape to return"
    )]
    serde_json::to_string(&(path, session_id)).expect("storage identity serializes")
}

/// Removes one session's container plus its WAL/SHM sidecars, upstream's
/// `removeSessionFiles`: the main file's removal honors `force`, and when it
/// fails the sidecars are not touched.
fn remove_session_files(path: &str, force: bool) -> Result<(), SessionError> {
    let remove = |target: &str, ignore_missing: bool| -> Result<(), SessionError> {
        match std::fs::remove_file(Path::new(target)) {
            Ok(()) => Ok(()),
            Err(error) if ignore_missing && error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(SessionError::Message(error.to_string())),
        }
    };
    remove(path, force)?;
    remove(&format!("{path}-wal"), true)?;
    remove(&format!("{path}-shm"), true)
}

fn io_error(error: &std::io::Error) -> SessionError {
    SessionError::Message(error.to_string())
}

/// Wires one writable connection, upstream's `configureWritableConnection`.
fn configure_writable_connection(db: &dyn SqliteDatabase) -> Result<(), SessionError> {
    db.exec("PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;")
        .map_err(SessionError::from)
}

/// Wires one read-only connection, upstream's `configureReadOnlyConnection`.
fn configure_read_only_connection(db: &dyn SqliteDatabase) -> Result<(), SessionError> {
    db.exec("PRAGMA busy_timeout = 5000;")
        .map_err(SessionError::from)
}

/// Builds the fork snapshot from a source snapshot, upstream's
/// `buildForkSnapshot`: agent-core's fork projection, then the entries
/// sorted by sequence and the message count.
fn build_fork_snapshot(
    source: SqliteStorageSnapshot,
    options: &ForkOptions,
) -> Result<ForkSnapshot, SessionError> {
    let destination: ForkDestinationSnapshot = create_fork_snapshot(
        &ForkSourceSnapshot {
            entries: source.entries,
            scalar_values: source.scalar_values,
            entries_complete: Some(source.entries_complete),
        },
        options,
    )?;
    let mut entries: Vec<Entry> = destination.entries.into_values().collect();
    entries.sort_by_key(Entry::seq);
    let message_count = u64::try_from(
        entries
            .iter()
            .filter(|entry| entry.entry_type() == EntryType::Message)
            .count(),
    )
    .unwrap_or(u64::MAX);
    Ok(ForkSnapshot {
        entries,
        scalar_values: destination.scalar_values,
        message_count,
        next_seq: destination.next_seq,
    })
}

/// Reads one external fork source through one deferred WAL transaction,
/// upstream's `createSqliteForkSnapshot`: begin, gate, read, commit; an
/// uncommitted read rolls back with the rollback failure replacing the
/// original error, unlike the adapter transaction's swallowed rollback.
fn create_sqlite_fork_snapshot(
    source_db: &dyn SqliteDatabase,
    source: &SqliteSessionMetadata,
    options: &ForkOptions,
) -> Result<ForkSnapshot, SessionError> {
    source_db.exec("BEGIN").map_err(SessionError::from)?;
    let result = (|| -> Result<ForkSnapshot, SessionError> {
        metadata_from_session_row(
            &source.path,
            &read_session_row(source_db, &source.base.id)?,
            SQLITE_STORAGE_VERSION,
        )?;
        let scalar_values = read_all_scalar_value_rows(source_db, &source.base.id)?;
        let entries_complete = matches!(options, ForkOptions::Tree { .. });
        let entries = crate::sqlite::storage::read_fork_source_entries(
            source_db,
            &source.base.id,
            &scalar_values,
            options,
        )?;
        build_fork_snapshot(
            SqliteStorageSnapshot {
                entries,
                scalar_values,
                entries_complete,
            },
            options,
        )
    })();
    match result {
        Ok(snapshot) => {
            source_db.exec("COMMIT").map_err(SessionError::from)?;
            Ok(snapshot)
        }
        Err(error) => {
            source_db.exec("ROLLBACK").map_err(SessionError::from)?;
            Err(error)
        }
    }
}

/// Inserts one forked scalar value, upstream's `insertForkValue`.
fn insert_fork_value(
    db: &dyn SqliteDatabase,
    session_id: &str,
    stored: &StoredValue,
) -> Result<(), SessionError> {
    set_scalar_value_row(
        db,
        session_id,
        &stored.namespace,
        &stored.key,
        stored.seq,
        &stored.value,
    )
}

/// Sets the forked session's message count, upstream's
/// `updateForkSessionStats`.
fn update_fork_session_stats(
    db: &dyn SqliteDatabase,
    session_id: &str,
    message_count: u64,
) -> Result<(), SessionError> {
    sql!(
        "UPDATE sessions SET message_count = ? WHERE id = ?",
        sql_integer(message_count)?,
        session_id
    )
    .run(db)?;
    Ok(())
}

/// The SQLite session repository, upstream's `SqliteSessionRepo`.
pub struct SqliteSessionRepo {
    directory: String,
    database_path: Option<String>,
    database_factory: Arc<dyn SqliteDatabaseFactory>,
    now: NowFn,
    pending_ids: Arc<Mutex<HashSet<String>>>,
    open_storages: Arc<Mutex<HashMap<String, Arc<SqliteStorage>>>>,
    /// Insertion-ordered like upstream's `Set`: the close-failure
    /// aggregation order follows it, and deregistration removes by session
    /// id (ids are unique per repo while reserved).
    open_sessions: Arc<Mutex<Vec<Arc<SqliteOpenSession>>>>,
    closed: AtomicBool,
    close_cell: tokio::sync::OnceCell<Result<(), SqliteRepoCloseError>>,
}

impl std::fmt::Debug for SqliteSessionRepo {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteSessionRepo")
            .field("directory", &self.directory)
            .field("database_path", &self.database_path)
            .finish_non_exhaustive()
    }
}

impl SqliteSessionRepo {
    /// A repository over one directory, upstream's constructor.
    #[must_use]
    pub fn new(options: SqliteSessionRepoOptions) -> Self {
        Self {
            directory: options.directory,
            database_path: options.database_path,
            database_factory: options.database_factory,
            now: options
                .now
                .unwrap_or_else(crate::sqlite::storage::default_now),
            pending_ids: Arc::new(Mutex::new(HashSet::new())),
            open_storages: Arc::new(Mutex::new(HashMap::new())),
            open_sessions: Arc::new(Mutex::new(Vec::new())),
            closed: AtomicBool::new(false),
            close_cell: tokio::sync::OnceCell::new(),
        }
    }

    fn assert_open(&self) -> Result<(), SessionError> {
        if self.closed.load(Ordering::SeqCst) {
            Err(SessionError::Message(
                "SqliteSessionRepo is closed".to_owned(),
            ))
        } else {
            Ok(())
        }
    }

    /// Reserves an id against overlapping local create/open/fork/delete
    /// ownership, upstream's `reserveId`.
    fn reserve_id(&self, id: &str) -> Result<(), SessionError> {
        {
            let mut pending = self
                .pending_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if pending.contains(id) {
                return Err(SessionError::Message(format!(
                    "Session is already open: {id}"
                )));
            }
            pending.insert(id.to_owned());
        }
        Ok(())
    }

    fn release_pending_id(&self, id: &str) {
        self.pending_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(id);
    }

    fn path_for_session(&self, id: &str) -> String {
        self.database_path
            .clone()
            .unwrap_or_else(|| session_path(&self.directory, id))
    }

    const fn uses_shared_database(&self) -> bool {
        self.database_path.is_some()
    }

    fn generated_id(created_at: i64) -> Result<String, SessionError> {
        pi_ai::utils::uuid::uuidv7(u64::try_from(created_at).ok())
            .map_err(|error| SessionError::Message(error.to_string()))
    }

    fn repository_path_for_metadata(
        &self,
        metadata: &SqliteSessionMetadata,
    ) -> Result<String, SessionError> {
        let expected = std::fs::canonicalize(self.path_for_session(&metadata.base.id))
            .map_err(|error| io_error(&error))?;
        let actual = std::fs::canonicalize(&metadata.path).map_err(|error| io_error(&error))?;
        if expected != actual {
            return Err(SessionError::Message(format!(
                "SQLite session metadata path is outside this repository: {}",
                metadata.path
            )));
        }
        Ok(actual.to_string_lossy().into_owned())
    }

    fn typed_metadata(&self, metadata: &SessionMetadata) -> SqliteSessionMetadata {
        SqliteSessionMetadata {
            base: metadata.clone(),
            path: self.path_for_session(&metadata.id),
        }
    }

    /// The erased fork's source metadata: the derived path canonicalized so
    /// the open-storage lookup matches the registered (canonical) key.
    fn fork_source_metadata(&self, metadata: &SessionMetadata) -> SqliteSessionMetadata {
        let derived = self.path_for_session(&metadata.id);
        let path = std::fs::canonicalize(&derived)
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or(derived);
        SqliteSessionMetadata {
            base: metadata.clone(),
            path,
        }
    }

    /// Creates one session, upstream's `create`.
    ///
    /// # Errors
    /// `Session is already open: {id}` when the id is reserved, `SQLite
    /// session already exists: {id}` when the row exists, or a driver or
    /// filesystem failure; a failed per-file create removes its container.
    pub async fn create(
        &self,
        options: SqliteSessionCreateOptions,
        _context: &Context,
    ) -> Result<Arc<SqliteOpenSession>, SessionError> {
        self.assert_open()?;
        let created_at = (self.now)();
        let id = match &options.id {
            Some(id) => id.clone(),
            None => Self::generated_id(created_at)?,
        };
        self.reserve_id(&id)?;
        let path = self.path_for_session(&id);
        let failure_db: Mutex<Option<Arc<dyn SqliteDatabase>>> = Mutex::new(None);
        let reserved_file = AtomicBool::new(false);
        let initialized = AtomicBool::new(false);
        let outcome: Result<Arc<SqliteOpenSession>, SessionError> = async {
            self.reserve_destination(&path, &reserved_file)?;
            let active_db = self.open_destination(&path, &failure_db)?;
            let canonical_path = std::fs::canonicalize(&path)
                .map_err(|error| io_error(&error))?
                .to_string_lossy()
                .into_owned();
            let metadata = SqliteSessionMetadata {
                base: SessionMetadata {
                    id: id.clone(),
                    created_at,
                    storage_version: SQLITE_STORAGE_VERSION,
                    cwd: None,
                    parent_session_id: options.parent_session_id.clone(),
                    legacy_parent_session_path: None,
                },
                path: canonical_path,
            };
            let transaction_db = Arc::clone(&active_db);
            let transaction_metadata = metadata.clone();
            let transaction_id = id.clone();
            active_db
                .transaction(Box::new(move || {
                    if has_session_row(transaction_db.as_ref(), &transaction_id)? {
                        return Err(SqliteAdapterError::new(format!(
                            "SQLite session already exists: {transaction_id}"
                        )));
                    }
                    insert_session_row(
                        transaction_db.as_ref(),
                        &transaction_metadata,
                        SQLITE_STORAGE_VERSION,
                        FIRST_AVAILABLE_COMMIT_SEQ,
                    )?;
                    Ok(SqliteTransactionOutcome::Committed(Box::new(())))
                }))
                .map_err(SessionError::from)?;
            initialized.store(true, Ordering::SeqCst);
            Ok(self.open_storage_backed_session(metadata, &active_db))
        }
        .await;
        self.reserved_creation_outcome(
            outcome,
            &path,
            &id,
            &failure_db,
            &reserved_file,
            &initialized,
        )
    }

    /// The destination directory + file reservation shared by `create` and
    /// `fork`, upstream's mkdir + `wx` block.
    ///
    /// # Errors
    /// A filesystem failure; `reserved_file` records the reservation for the
    /// shared epilogue.
    fn reserve_destination(
        &self,
        path: &str,
        reserved_file: &AtomicBool,
    ) -> Result<(), SessionError> {
        let parent = Path::new(path)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).map_err(|error| io_error(&error))?;
        if !self.uses_shared_database() {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|error| io_error(&error))?;
            reserved_file.store(true, Ordering::SeqCst);
        }
        Ok(())
    }

    /// The writable-open + schema block shared by `create` and `fork`,
    /// upstream's open + configure + applyInitialSchema.
    ///
    /// # Errors
    /// A filesystem or driver failure; `failure_db` records the connection
    /// for the shared epilogue.
    fn open_destination(
        &self,
        path: &str,
        failure_db: &Mutex<Option<Arc<dyn SqliteDatabase>>>,
    ) -> Result<Arc<dyn SqliteDatabase>, SessionError> {
        let active_db: Arc<dyn SqliteDatabase> = Arc::from(self.database_factory.open(path)?);
        *failure_db.lock().unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(&active_db));
        configure_writable_connection(active_db.as_ref())?;
        apply_initial_schema(active_db.as_ref())?;
        Ok(active_db)
    }

    /// The create/fork shared epilogue, upstream's catch + finally: a failed
    /// reserved file is removed (its failure replacing the original), the
    /// connection closes (its failure replacing in turn), and the pending id
    /// releases even when the close throws.
    fn reserved_creation_outcome(
        &self,
        outcome: Result<Arc<SqliteOpenSession>, SessionError>,
        path: &str,
        id: &str,
        failure_db: &Mutex<Option<Arc<dyn SqliteDatabase>>>,
        reserved_file: &AtomicBool,
        initialized: &AtomicBool,
    ) -> Result<Arc<SqliteOpenSession>, SessionError> {
        match outcome {
            Ok(session) => Ok(session),
            Err(error) => {
                let error = if reserved_file.load(Ordering::SeqCst)
                    && !initialized.load(Ordering::SeqCst)
                {
                    match remove_session_files(path, true) {
                        Ok(()) => error,
                        Err(remove_error) => remove_error,
                    }
                } else {
                    error
                };
                let close_result = failure_db
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take()
                    .map_or(Ok(()), |db| db.close().map_err(SessionError::from));
                self.release_pending_id(id);
                close_result?;
                Err(error)
            }
        }
    }

    /// Opens one session by its metadata, upstream's `open`.
    ///
    /// # Errors
    /// `Session is already open: {id}` when the handle exists, `SQLite
    /// session metadata path is outside this repository: {path}` for foreign
    /// metadata, the storage-version gate's errors, and `Unknown SQLite
    /// session: {id}` when the row is missing.
    pub async fn open(
        &self,
        metadata: &SqliteSessionMetadata,
        _context: &Context,
    ) -> Result<Arc<SqliteOpenSession>, SessionError> {
        self.assert_open()?;
        self.reserve_id(&metadata.base.id)?;
        let failure_db: Mutex<Option<Arc<dyn SqliteDatabase>>> = Mutex::new(None);
        let outcome: Result<Arc<SqliteOpenSession>, SessionError> = async {
            let path = self.repository_path_for_metadata(metadata)?;
            let active_db: Arc<dyn SqliteDatabase> =
                Arc::from(self.database_factory.open_existing(&path)?);
            *failure_db.lock().unwrap_or_else(PoisonError::into_inner) =
                Some(Arc::clone(&active_db));
            configure_writable_connection(active_db.as_ref())?;
            let stored = metadata_from_session_row(
                &path,
                &read_session_row(active_db.as_ref(), &metadata.base.id)?,
                SQLITE_STORAGE_VERSION,
            )?;
            Ok(self.open_storage_backed_session(stored, &active_db))
        }
        .await;
        match outcome {
            Ok(session) => Ok(session),
            Err(error) => {
                let close_result = failure_db
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take()
                    .map_or(Ok(()), |db| db.close().map_err(SessionError::from));
                self.release_pending_id(&metadata.base.id);
                close_result?;
                Err(error)
            }
        }
    }

    /// Lists every session's metadata, upstream's `list`: read-only,
    /// best-effort discovery (corrupt files, incompatible versions, and
    /// unrelated `*.sqlite` files are reported when explicitly opened),
    /// newest first.
    ///
    /// # Errors
    /// `SqliteSessionRepo is closed` or a directory-read failure.
    pub fn list(&self, _context: &Context) -> Result<Vec<SqliteSessionMetadata>, SessionError> {
        self.assert_open()?;
        let paths: Vec<String> = if let Some(database_path) = &self.database_path {
            vec![database_path.clone()]
        } else {
            let entries = match std::fs::read_dir(&self.directory) {
                Ok(entries) => entries,
                Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
                Err(error) => return Err(io_error(&error)),
            };
            let mut paths = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|error| io_error(&error))?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.ends_with(SQLITE_SESSION_EXTENSION) {
                    paths.push(entry.path().to_string_lossy().into_owned());
                }
            }
            paths
        };
        let mut sessions = Vec::new();
        for path in paths {
            let read = (|| -> Result<Vec<SqliteSessionMetadata>, SessionError> {
                let canonical_path = std::fs::canonicalize(&path)
                    .map_err(|error| io_error(&error))?
                    .to_string_lossy()
                    .into_owned();
                let db = self.database_factory.open_read_only(&canonical_path)?;
                let body = (|| -> Result<Vec<SqliteSessionMetadata>, SessionError> {
                    configure_read_only_connection(db.as_ref())?;
                    read_all_session_rows(db.as_ref())?
                        .iter()
                        .map(|row| {
                            metadata_from_session_row(&canonical_path, row, SQLITE_STORAGE_VERSION)
                        })
                        .collect()
                })();
                let close_result = db.close().map_err(SessionError::from);
                match close_result {
                    Err(close_error) => Err(close_error),
                    Ok(()) => body,
                }
            })();
            // Discovery is best-effort: corrupt files, incompatible versions,
            // and unrelated *.sqlite files are reported when explicitly opened.
            if let Ok(rows) = read {
                sessions.extend(rows);
            }
        }
        sessions.sort_by_key(|metadata| std::cmp::Reverse(metadata.base.created_at));
        Ok(sessions)
    }

    /// Deletes one session, upstream's `delete`.
    ///
    /// # Errors
    /// `Session is already open: {id}` when the id is reserved, the
    /// outside-this-repository guard, the storage-version gate, `Expected to
    /// delete one SQLite session {id}, deleted {n}` when the final delete
    /// misses, or a filesystem failure.
    pub async fn delete(
        &self,
        metadata: &SqliteSessionMetadata,
        _context: &Context,
    ) -> Result<(), SessionError> {
        self.assert_open()?;
        self.reserve_id(&metadata.base.id)?;
        let outcome: Result<(), SessionError> = async {
            let path = self.repository_path_for_metadata(metadata)?;
            let db: Arc<dyn SqliteDatabase> =
                Arc::from(self.database_factory.open_existing(&path)?);
            let body = (|| -> Result<(), SessionError> {
                configure_writable_connection(db.as_ref())?;
                metadata_from_session_row(
                    &path,
                    &read_session_row(db.as_ref(), &metadata.base.id)?,
                    SQLITE_STORAGE_VERSION,
                )?;
                if self.uses_shared_database() {
                    // The shared container's row deletion is one transaction,
                    // upstream's `db.transaction(() => { ...; deleteSessionRows(...) })`.
                    let transaction_db = Arc::clone(&db);
                    let transaction_id = metadata.base.id.clone();
                    db.transaction(Box::new(move || {
                        delete_session_rows(transaction_db.as_ref(), &transaction_id)?;
                        Ok(SqliteTransactionOutcome::Committed(Box::new(())))
                    }))
                    .map_err(SessionError::from)?;
                }
                Ok(())
            })();
            let close_result = db.close().map_err(SessionError::from);
            match close_result {
                Err(close_error) => Err(close_error),
                Ok(()) => body,
            }?;
            if !self.uses_shared_database() {
                remove_session_files(&path, false)?;
            }
            Ok(())
        }
        .await;
        self.release_pending_id(&metadata.base.id);
        outcome
    }

    /// Forks one source, upstream's `fork`: an open source in this repository
    /// queues its snapshot on the source's commit line; any other source
    /// reads through one independent read-only connection and one deferred
    /// WAL transaction. Forking deliberately permits a foreign source
    /// metadata path.
    ///
    /// # Errors
    /// `Session is already open: {id}` when the destination id is reserved,
    /// `Unknown source branch: {branch}`, the source's storage-version gate,
    /// `SQLite session already exists: {id}`, or a driver or filesystem
    /// failure; a failed per-file fork removes its container.
    #[expect(
        clippy::too_many_lines,
        reason = "the fork pipeline mirrors upstream's fork body step for step; splitting it would scatter one atomic reservation flow"
    )]
    pub async fn fork(
        &self,
        source: &SqliteSessionMetadata,
        options: &ForkOptions,
        context: &Context,
    ) -> Result<Arc<SqliteOpenSession>, SessionError> {
        self.assert_open()?;
        let created_at = (self.now)();
        let id = match options.id() {
            Some(id) => id.clone(),
            None => Self::generated_id(created_at)?,
        };
        self.reserve_id(&id)?;
        let source_storage = self
            .open_storages
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&storage_identity(&source.path, &source.base.id))
            .cloned();
        // Upstream discards the racing snapshot's rejection via
        // `void .catch(() => undefined)`; a dropped future cannot panic here,
        // and the await below still propagates it.
        let active_source_snapshot = source_storage
            .as_ref()
            .map(|storage| storage.snapshot(options, context));
        let path = self.path_for_session(&id);
        let failure_db: Mutex<Option<Arc<dyn SqliteDatabase>>> = Mutex::new(None);
        let reserved_file = AtomicBool::new(false);
        let initialized = AtomicBool::new(false);
        let outcome: Result<Arc<SqliteOpenSession>, SessionError> = async {
            let parent = Path::new(&path)
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            std::fs::create_dir_all(parent).map_err(|error| io_error(&error))?;
            if !self.uses_shared_database() {
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(|error| io_error(&error))?;
                reserved_file.store(true, Ordering::SeqCst);
            }
            let snapshot = match active_source_snapshot {
                Some(active) => build_fork_snapshot(active.await?, options)?,
                None => self.create_fork_snapshot_from_external_source(source, options)?,
            };
            let active_db: Arc<dyn SqliteDatabase> = Arc::from(self.database_factory.open(&path)?);
            *failure_db.lock().unwrap_or_else(PoisonError::into_inner) =
                Some(Arc::clone(&active_db));
            configure_writable_connection(active_db.as_ref())?;
            apply_initial_schema(active_db.as_ref())?;
            let canonical_path = std::fs::canonicalize(&path)
                .map_err(|error| io_error(&error))?
                .to_string_lossy()
                .into_owned();
            let metadata = SqliteSessionMetadata {
                base: SessionMetadata {
                    id: id.clone(),
                    created_at,
                    storage_version: SQLITE_STORAGE_VERSION,
                    cwd: None,
                    parent_session_id: Some(source.base.id.clone()),
                    legacy_parent_session_path: None,
                },
                path: canonical_path,
            };
            let transaction_db = Arc::clone(&active_db);
            let transaction_metadata = metadata.clone();
            let transaction_id = id.clone();
            let snapshot_entries = snapshot.entries.clone();
            let snapshot_values = snapshot.scalar_values.clone();
            let snapshot_next_seq = snapshot.next_seq;
            let snapshot_message_count = snapshot.message_count;
            active_db
                .transaction(Box::new(move || {
                    if has_session_row(transaction_db.as_ref(), &transaction_id)? {
                        return Err(SqliteAdapterError::new(format!(
                            "SQLite session already exists: {transaction_id}"
                        )));
                    }
                    insert_session_row(
                        transaction_db.as_ref(),
                        &transaction_metadata,
                        SQLITE_STORAGE_VERSION,
                        snapshot_next_seq,
                    )?;
                    let entry_writer =
                        EntryRowWriter::new(Arc::clone(&transaction_db), transaction_id.clone());
                    for entry in &snapshot_entries {
                        entry_writer.insert(entry)?;
                        append_entry_to_branch_index(
                            transaction_db.as_ref(),
                            &transaction_id,
                            entry,
                        )?;
                    }
                    for stored in &snapshot_values {
                        insert_fork_value(transaction_db.as_ref(), &transaction_id, stored)?;
                    }
                    update_fork_session_stats(
                        transaction_db.as_ref(),
                        &transaction_id,
                        snapshot_message_count,
                    )?;
                    Ok(SqliteTransactionOutcome::Committed(Box::new(())))
                }))
                .map_err(SessionError::from)?;
            initialized.store(true, Ordering::SeqCst);
            Ok(self.open_storage_backed_session(metadata, &active_db))
        }
        .await;
        self.reserved_creation_outcome(
            outcome,
            &path,
            &id,
            &failure_db,
            &reserved_file,
            &initialized,
        )
    }

    /// Reads one external fork source, upstream's
    /// `createForkSnapshotFromExternalSource`: the source's exact existing
    /// container, read-only, through one deferred WAL transaction.
    fn create_fork_snapshot_from_external_source(
        &self,
        source: &SqliteSessionMetadata,
        options: &ForkOptions,
    ) -> Result<ForkSnapshot, SessionError> {
        let source_path = std::fs::canonicalize(&source.path)
            .map_err(|error| io_error(&error))?
            .to_string_lossy()
            .into_owned();
        let db = self.database_factory.open_read_only(&source_path)?;
        let body = (|| -> Result<ForkSnapshot, SessionError> {
            configure_read_only_connection(db.as_ref())?;
            create_sqlite_fork_snapshot(db.as_ref(), source, options)
        })();
        let close_result = db.close().map_err(SessionError::from);
        match close_result {
            Err(close_error) => Err(close_error),
            Ok(()) => body,
        }
    }

    /// Closes every open session, upstream's `close`: a lone failure rethrows
    /// alone, two or more aggregate.
    ///
    /// # Errors
    /// The sessions' close errors, aggregated per upstream's rethrow rules;
    /// memoized, so a second close reports the first result.
    pub async fn close(&self, context: &Context) -> Result<(), SqliteRepoCloseError> {
        let result = self
            .close_cell
            .get_or_init(|| async {
                self.closed.store(true, Ordering::SeqCst);
                let sessions: Vec<Arc<SqliteOpenSession>> = self
                    .open_sessions
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .drain(..)
                    .collect();
                let mut errors = Vec::new();
                for session in sessions {
                    if let Err(error) = session.close(context).await {
                        errors.push(error);
                    }
                }
                match errors.as_slice() {
                    [] => Ok(()),
                    [error] => Err(SqliteRepoCloseError::Single(error.clone())),
                    _ => Err(SqliteRepoCloseError::Aggregate(errors)),
                }
            })
            .await;
        result.clone()
    }

    /// Publishes one open session over its storage, upstream's
    /// `openStorageBackedSession`: the storage registers, the facade wraps
    /// the session, and the close callback closes the database and
    /// deregisters.
    fn open_storage_backed_session(
        &self,
        metadata: SqliteSessionMetadata,
        db: &Arc<dyn SqliteDatabase>,
    ) -> Arc<SqliteOpenSession> {
        let key = storage_identity(&metadata.path, &metadata.base.id);
        let storage = Arc::new(SqliteStorage::new(
            Arc::clone(db),
            &SqliteStorageOptions {
                session_id: metadata.base.id.clone(),
                now: Some(Arc::clone(&self.now)),
            },
        ));
        self.open_storages
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key.clone(), Arc::clone(&storage));
        let storage_contract: Arc<dyn pi_agent_core::harness::session::types::Storage> =
            storage.clone();
        let session = Arc::new(StorageBackedSession::new(
            metadata.base(),
            storage_contract,
            StorageBackedSessionOptions::default(),
        ));
        let identity_key = key;
        let identity_storage = Arc::clone(&storage);
        let open_storages = self.open_storages.clone();
        let open_sessions = self.open_sessions.clone();
        let pending_ids = self.pending_ids.clone();
        let session_id = metadata.base.id.clone();
        let on_close: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            let mut storages = open_storages.lock().unwrap_or_else(PoisonError::into_inner);
            if storages
                .get(&identity_key)
                .is_some_and(|current| Arc::ptr_eq(current, &identity_storage))
            {
                storages.remove(&identity_key);
            }
            drop(storages);
            open_sessions
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|open| open.session_id() != session_id);
            pending_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&session_id);
        });
        let close_database: Arc<dyn Fn() -> Result<(), SqliteAdapterError> + Send + Sync> =
            Arc::new({
                let db = Arc::clone(db);
                move || db.close()
            });
        let open_session = Arc::new(SqliteOpenSession::new(
            Arc::clone(&session),
            metadata,
            crate::sqlite::session::SqliteOpenSessionOptions {
                close_database,
                on_close,
            },
        ));
        self.open_sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::clone(&open_session));
        open_session
    }
}

impl SessionRepo for SqliteSessionRepo {
    fn create(
        &self,
        options: SessionCreateOptions,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Session>, SessionError>> {
        let context = context.clone();
        Box::pin(async move {
            let session = self.create(options, &context).await?;
            let boxed: Box<dyn Session> = session.erased_session();
            Ok(boxed)
        })
    }

    fn open(
        &self,
        metadata: &SessionMetadata,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Session>, SessionError>> {
        let typed = self.typed_metadata(metadata);
        let context = context.clone();
        Box::pin(async move {
            let session = self.open(&typed, &context).await?;
            let boxed: Box<dyn Session> = session.erased_session();
            Ok(boxed)
        })
    }

    fn list(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<SessionMetadata>, SessionError>> {
        let context = context.clone();
        Box::pin(async move {
            Ok(self
                .list(&context)?
                .iter()
                .map(SqliteSessionMetadata::base)
                .collect())
        })
    }

    fn delete(
        &self,
        metadata: &SessionMetadata,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        let typed = self.typed_metadata(metadata);
        let context = context.clone();
        Box::pin(async move { self.delete(&typed, &context).await })
    }

    fn fork(
        &self,
        source: &SessionMetadata,
        options: &ForkOptions,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Session>, SessionError>> {
        let typed = self.fork_source_metadata(source);
        let options = options.clone();
        let context = context.clone();
        Box::pin(async move {
            let session = self.fork(&typed, &options, &context).await?;
            let boxed: Box<dyn Session> = session.erased_session();
            Ok(boxed)
        })
    }
}
