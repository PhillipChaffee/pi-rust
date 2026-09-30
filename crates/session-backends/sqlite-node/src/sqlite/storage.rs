//! The SQLite `Storage` implementation, upstream's
//! `src/sqlite/storage.ts`.
//!
//! Upstream serializes commits through a promise queue and joins `snapshot`
//! and `close` onto the same tail; the port holds a FIFO lock across each
//! apply (the JSONL backend's restatement), so admission order is poll order
//! and the queue machinery is gone. Reads bypass the line, like upstream's
//! non-queued reads. Upstream's apply-time duplicate/parent validation is the
//! DDL's own triggers (`RAISE(ABORT, ...)`); the port runs no TS-side
//! preflight either, so a rejected commit carries the trigger's message.

use std::collections::BTreeMap;
use std::future::ready;
use std::sync::{Arc, Mutex, PoisonError};

use pi_agent_core::harness::context::Context;
use pi_agent_core::harness::session::commit::{CommittedWrite, prepare_storage_commit};
use pi_agent_core::harness::session::types::{
    BranchScanOrder, CommitResult, Entry, EntryScan, EntryStructure, EntryType, ForkOptions,
    SessionError, SessionStats, Storage, StorageBranchScan, UsageRow, UsageScan,
};
use pi_agent_core::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, Write, branch_tip,
};
use pi_ai::types::BoxedFuture;

use crate::sqlite::branch_entries::{
    append_entry_to_branch_index, scan_branch_entries, scan_branch_entry_structures,
};
use crate::sqlite::entries::{
    EntryRowWriter, decode_entry_row, decode_entry_rows, read_all_entry_rows, read_entry_rows,
    scan_entry_rows,
};
use crate::sqlite::session_sequences::{advance_next_seq, read_next_seq};
use crate::sqlite::session_stats::{
    add_usage_to_session_stats, increment_message_count, read_session_stats,
};
use crate::sqlite::types::SqliteDatabase;
use crate::sqlite::types::SqliteTransactionOutcome;
use crate::sqlite::usage_ledger::{
    UsageLedgerRowWriter, decode_usage_ledger_row, scan_usage_ledger_rows,
};
use crate::sqlite::values::{
    append_list_value_row, delete_list_value_rows, delete_scalar_value_row,
    read_all_scalar_value_rows, read_list_value_rows, read_scalar_value_row,
    scan_scalar_value_rows, set_scalar_value_row,
};

/// The commit-timestamp clock, upstream's `now?: () => number` (an alias, the
/// same shape every backend carries).
pub type NowFn = Arc<dyn Fn() -> i64 + Send + Sync>;

pub(crate) fn default_now() -> NowFn {
    Arc::new(pi_ai::auth::resolve::now_ms)
}

/// The storage options, upstream's `SqliteStorageOptions`.
#[derive(Clone)]
pub struct SqliteStorageOptions {
    /// The session the storage persists, upstream's `sessionId`.
    pub session_id: String,
    /// The commit-timestamp clock; defaults to the process wall clock,
    /// upstream's `now ?? Date.now`.
    pub now: Option<NowFn>,
}

impl std::fmt::Debug for SqliteStorageOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteStorageOptions")
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

/// The snapshot one fork reads from an open storage, upstream's
/// `SqliteStorageSnapshot`.
#[derive(Clone, Debug)]
pub struct SqliteStorageSnapshot {
    /// The fork's source entries.
    pub entries: Vec<Entry>,
    /// The fork's source values.
    pub scalar_values: Vec<StoredValue>,
    /// Whether the entries cover the whole tree, upstream's
    /// `entriesComplete` — true for tree scopes only.
    pub entries_complete: bool,
}

/// The storage lifecycle, upstream's `"open" | "closing" | "closed"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lifecycle {
    Open,
    Closing,
    Closed,
}

/// The SQLite-backed storage for one session, upstream's `SqliteStorage`.
pub struct SqliteStorage {
    db: Arc<dyn SqliteDatabase>,
    session_id: String,
    now: NowFn,
    entry_writer: EntryRowWriter,
    usage_writer: UsageLedgerRowWriter,
    /// The FIFO line every commit, snapshot, and close applies behind,
    /// upstream's `commitQueue`.
    commit_line: Arc<tokio::sync::Mutex<()>>,
    lifecycle: Mutex<Lifecycle>,
    close_cell: tokio::sync::OnceCell<Result<(), SessionError>>,
}

impl std::fmt::Debug for SqliteStorage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteStorage")
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

impl SqliteStorage {
    /// Opens the storage over an already-configured connection, upstream's
    /// constructor (the caller applies the schema and PRAGMAs).
    #[must_use]
    pub fn new(db: Arc<dyn SqliteDatabase>, options: &SqliteStorageOptions) -> Self {
        let now = options.now.clone().unwrap_or_else(default_now);
        Self {
            entry_writer: EntryRowWriter::new(Arc::clone(&db), options.session_id.clone()),
            usage_writer: UsageLedgerRowWriter::new(Arc::clone(&db), options.session_id.clone()),
            db,
            session_id: options.session_id.clone(),
            now,
            commit_line: Arc::new(tokio::sync::Mutex::new(())),
            lifecycle: Mutex::new(Lifecycle::Open),
            close_cell: tokio::sync::OnceCell::new(),
        }
    }

    /// The snapshot one fork reads from this open storage, upstream's
    /// `snapshot`: joined behind the commit line like a commit.
    ///
    /// # Errors
    /// `SqliteStorage is closed`, or `Unknown source branch: {branch}` for a
    /// branch scope whose tip value is missing.
    pub async fn snapshot(
        &self,
        options: &ForkOptions,
        _context: &Context,
    ) -> Result<SqliteStorageSnapshot, SessionError> {
        self.assert_open()?;
        let line = self.commit_line.lock().await;
        let snapshot = self.read_snapshot(options);
        drop(line);
        snapshot
    }

    fn read_snapshot(&self, options: &ForkOptions) -> Result<SqliteStorageSnapshot, SessionError> {
        let scalar_values = read_all_scalar_value_rows(self.db.as_ref(), &self.session_id)?;
        let entries_complete = matches!(options, ForkOptions::Tree { .. });
        let entries = match options {
            ForkOptions::Tree { .. } => {
                decode_entry_rows(&read_all_entry_rows(self.db.as_ref(), &self.session_id)?)?
            }
            ForkOptions::Branch { .. } => read_fork_source_entries(
                self.db.as_ref(),
                &self.session_id,
                &scalar_values,
                options,
            )?,
        };
        Ok(SqliteStorageSnapshot {
            entries,
            scalar_values,
            entries_complete,
        })
    }

    fn assert_open(&self) -> Result<(), SessionError> {
        let lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if *lifecycle == Lifecycle::Open {
            Ok(())
        } else {
            Err(SessionError::Message("SqliteStorage is closed".to_owned()))
        }
    }

    fn with_open<T>(
        &self,
        read: impl FnOnce() -> Result<T, SessionError>,
    ) -> Result<T, SessionError> {
        self.assert_open()?;
        read()
    }

    async fn commit_impl(&self, writes: Vec<Write>) -> Result<CommitResult, SessionError> {
        let line = self.commit_line.lock().await;
        let result = self.apply_commit(writes);
        drop(line);
        result
    }

    /// Applies one commit inside one write transaction, upstream's
    /// `applyCommit`: read the next sequence, prepare against it, apply every
    /// write, advance the sequence, and read the stats — all in one
    /// transaction.
    fn apply_commit(&self, writes: Vec<Write>) -> Result<CommitResult, SessionError> {
        let db = Arc::clone(&self.db);
        let entry_writer = self.entry_writer.clone();
        let usage_writer = self.usage_writer.clone();
        let session_id = self.session_id.clone();
        let now = (self.now)();
        let outcome = self.db.transaction(Box::new(move || {
            let result = apply_commit_body(
                db.as_ref(),
                &session_id,
                &entry_writer,
                &usage_writer,
                writes,
                now,
            )?;
            Ok(SqliteTransactionOutcome::Committed(Box::new(result)))
        }))?;
        outcome.downcast::<CommitResult>().map_or_else(
            |_| {
                Err(SessionError::Message(
                    "SQLite transaction returned an unexpected value".to_owned(),
                ))
            },
            |result| Ok(*result),
        )
    }

    async fn close_impl(&self) -> Result<(), SessionError> {
        // The synchronous "closing" latch happened in the trait method; this
        // body is the queue-tail drain that moves the state to "closed"
        // (upstream's `.then(() => { this.state = "closed"; })`).
        self.close_cell
            .get_or_init(|| async {
                let line = self.commit_line.lock().await;
                *self
                    .lifecycle
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closed;
                drop(line);
                Ok(())
            })
            .await
            .clone()
    }
}

/// Reads the fork source's entries for one scope, upstream's
/// `readForkSourceEntries` (repo.ts) — the same branch-tip selection
/// `readSnapshotEntries` (storage.ts) performs; the port shares the one
/// helper between the storage snapshot and the repo fork.
pub(crate) fn read_fork_source_entries(
    db: &dyn SqliteDatabase,
    session_id: &str,
    scalar_values: &[StoredValue],
    options: &ForkOptions,
) -> Result<Vec<Entry>, SessionError> {
    match options {
        ForkOptions::Tree { .. } => decode_entry_rows(&read_all_entry_rows(db, session_id)?),
        ForkOptions::Branch { branch, .. } => {
            let source_address = branch_tip(branch).address;
            let source_tip = scalar_values.iter().find(|stored| {
                stored.namespace == source_address.namespace && stored.key == source_address.key
            });
            let Some(source_tip) = source_tip else {
                return Err(SessionError::Message(format!(
                    "Unknown source branch: {branch}"
                )));
            };
            if source_tip.value.is_null() {
                return Ok(Vec::new());
            }
            let start: String = serde_json::from_value(source_tip.value.clone())
                .map_err(|error| SessionError::Message(error.to_string()))?;
            let query = StorageBranchScan {
                start,
                order: Some(BranchScanOrder::OldestFirst),
                ..StorageBranchScan::default()
            };
            scan_branch_entries(db, session_id, &query)
        }
    }
}

/// The transaction body, upstream's `applyCommit`'s transaction callback.
fn apply_commit_body(
    db: &dyn SqliteDatabase,
    session_id: &str,
    entry_writer: &EntryRowWriter,
    usage_writer: &UsageLedgerRowWriter,
    writes: Vec<Write>,
    now: i64,
) -> Result<CommitResult, SessionError> {
    let first_seq = read_next_seq(db, session_id)?;
    let prepared = prepare_storage_commit(writes, first_seq, now);
    for write in &prepared.writes {
        match write {
            CommittedWrite::Entry { entry } => {
                entry_writer.insert(entry)?;
                append_entry_to_branch_index(db, session_id, entry)?;
                if entry.entry_type() == EntryType::Message {
                    increment_message_count(db, session_id)?;
                }
            }
            CommittedWrite::Usage { row } => {
                usage_writer.insert(row)?;
                add_usage_to_session_stats(db, session_id, &row.usage)?;
            }
            CommittedWrite::ValueSet {
                seq,
                namespace,
                key,
                value,
            } => {
                set_scalar_value_row(db, session_id, namespace, key, *seq, value)?;
            }
            CommittedWrite::ValueDelete { namespace, key, .. } => {
                delete_scalar_value_row(db, session_id, namespace, key)?;
            }
            CommittedWrite::ListAppend {
                seq,
                namespace,
                key,
                value,
            } => {
                append_list_value_row(db, session_id, namespace, key, *seq, value)?;
            }
            CommittedWrite::ListDelete { namespace, key, .. } => {
                delete_list_value_rows(db, session_id, namespace, key)?;
            }
        }
    }
    advance_next_seq(
        db,
        session_id,
        first_seq.saturating_add(u64::try_from(prepared.writes.len()).unwrap_or(u64::MAX)),
    )?;
    let stats = read_session_stats(db, session_id)?;
    Ok(CommitResult {
        first_seq: prepared.result.first_seq,
        seqs: prepared.result.seqs.clone(),
        timestamp: prepared.result.timestamp,
        stats,
    })
}

impl Storage for SqliteStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        // Upstream admits the commit synchronously at the call
        // (storage.ts:77-80): the open gate and the queue-tail join run
        // before any await, so a commit called while open still applies once
        // close has latched "closing". The apply itself never re-checks the
        // state; the call-time gate is the only one.
        if let Err(error) = self.assert_open() {
            return Box::pin(ready(Err(error)));
        }
        Box::pin(self.commit_impl(writes))
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>> {
        Box::pin(ready(self.with_open(|| {
            let rows = read_entry_rows(self.db.as_ref(), &self.session_id, &ids)?;
            let mut entries = BTreeMap::new();
            for row in rows {
                let entry = decode_entry_row(&row)?;
                entries.insert(entry.id().to_owned(), entry);
            }
            Ok(entries)
        })))
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        Box::pin(ready(self.with_open(|| {
            read_scalar_value_row(self.db.as_ref(), &self.session_id, address)
        })))
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        Box::pin(ready(self.with_open(|| {
            scan_scalar_value_rows(self.db.as_ref(), &self.session_id, prefix)
        })))
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        Box::pin(ready(self.with_open(|| {
            read_list_value_rows(self.db.as_ref(), &self.session_id, address, options)
        })))
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        Box::pin(ready(self.with_open(|| {
            scan_branch_entries(self.db.as_ref(), &self.session_id, query)
        })))
    }

    fn scan_branch_structure(
        &self,
        query: &StorageBranchScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<EntryStructure>, SessionError>> {
        Box::pin(ready(self.with_open(|| {
            scan_branch_entry_structures(self.db.as_ref(), &self.session_id, query)
        })))
    }

    fn scan_entries(
        &self,
        query: &EntryScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        Box::pin(ready(self.with_open(|| {
            let rows = scan_entry_rows(self.db.as_ref(), &self.session_id, query)?;
            decode_entry_rows(&rows)
        })))
    }

    fn scan_usage(
        &self,
        query: &UsageScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<UsageRow>, SessionError>> {
        Box::pin(ready(self.with_open(|| {
            let rows = scan_usage_ledger_rows(self.db.as_ref(), &self.session_id, query)?;
            rows.iter().map(decode_usage_ledger_row).collect()
        })))
    }

    fn get_stats(&self, _context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        Box::pin(ready(self.with_open(|| {
            read_session_stats(self.db.as_ref(), &self.session_id)
        })))
    }

    fn close(&self, _context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        // Upstream's close is a plain method (storage.ts:202-209): the
        // memoized-promise check and the "closing" latch run synchronously at
        // the call, so reads and commits called afterwards reject while the
        // queue drain is still pending; a repeat call is the memoized one and
        // never regresses a completed close.
        if self.close_cell.get().is_none() {
            *self
                .lifecycle
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Lifecycle::Closing;
        }
        Box::pin(self.close_impl())
    }
}
