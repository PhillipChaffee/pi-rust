//! Shared test support for the sqlite-node suites: fixtures, raw seed
//! helpers, query-plan helpers, the conformance runner, and the seam doubles
//! upstream's tests build (transaction counting, gated opens, snapshot
//! interception, close tracking).

#![expect(
    dead_code,
    reason = "shared fixtures; each test binary uses the subset it needs"
)]
#![expect(clippy::expect_used, reason = "test fixtures assert on construction")]
#![expect(clippy::panic, reason = "test fixtures panic on misuse")]
#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pi_agent_core::harness::session::testing::ConformanceCase;
use pi_agent_core::harness::session::types::{CommitResult, SessionError};
use pi_agent_core::types::BoxedFuture;
use pi_session_backend_sqlite_node::sqlite::storage::NowFn;
use pi_session_backend_sqlite_node::sqlite::types::{
    SqliteDatabase, SqliteDatabaseFactory, SqliteParams, SqliteRow, SqliteRunResult,
    SqliteStatement,
};
use pi_session_backend_sqlite_node::{SqliteSessionRepoOptions, create_rusqlite_factory, sql};

/// The fixed clock the upstream suites inject, storage.test.ts's
/// `1_700_000_000_000`.
pub const NOW: i64 = 1_700_000_000_000;

/// The zero usage the seeded session rows carry, upstream's `EMPTY_USAGE`.
pub fn zero_usage_json() -> String {
    serde_json::json!({
        "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
        "totalTokens": 0,
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 },
    })
    .to_string()
}

/// A fixed clock, the suites' `now: () => <n>` injections.
pub fn fixed_clock(value: i64) -> NowFn {
    Arc::new(move || value)
}

/// The per-file repo options over one directory.
pub fn repo_options(directory: &str, now: NowFn) -> SqliteSessionRepoOptions {
    SqliteSessionRepoOptions {
        directory: directory.to_owned(),
        database_path: None,
        database_factory: Arc::new(create_rusqlite_factory()),
        now: Some(now),
    }
}

/// The shared-container repo options over one directory.
pub fn shared_repo_options(
    directory: &str,
    database_path: &str,
    now: NowFn,
) -> SqliteSessionRepoOptions {
    SqliteSessionRepoOptions {
        directory: directory.to_owned(),
        database_path: Some(database_path.to_owned()),
        database_factory: Arc::new(create_rusqlite_factory()),
        now: Some(now),
    }
}

/// Whether a path exists, the suites' `pathExists` helper.
pub fn path_exists(path: &str) -> bool {
    std::fs::exists(path).expect("path_exists")
}

/// Seeds the `sessions` row the storage suites commit against, upstream's
/// `insertCommitSessionRow`.
pub fn insert_commit_session_row(db: &dyn SqliteDatabase, next_seq: i64) {
    sql!(
        "INSERT INTO sessions
			(id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        "session",
        1i64,
        None::<String>,
        1i64,
        None::<String>,
        0i64,
        zero_usage_json(),
        next_seq,
    )
    .run(db)
    .map(|_| ())
    .expect("seed session row");
}

/// Seeds the conformance fixture's session row: created at NOW under the
/// crate's storage version, upstream's storage-conformance insert.
pub fn insert_conformance_session_row(db: &dyn SqliteDatabase) {
    sql!(
        "INSERT INTO sessions
			(id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        "session",
        NOW,
        None::<String>,
        i64::from(pi_session_backend_sqlite_node::SQLITE_STORAGE_VERSION),
        None::<String>,
        0i64,
        zero_usage_json(),
        1i64,
    )
    .run(db)
    .map(|_| ())
    .expect("seed conformance session row");
}

/// The plans one `EXPLAIN QUERY PLAN` read reports, upstream's
/// `explainQueryPlan`.
pub fn explain_query_plan(
    db: &dyn SqliteDatabase,
    query: &str,
    params: &SqliteParams,
) -> Vec<String> {
    db.prepare(&format!("EXPLAIN QUERY PLAN {query}"))
        .all(params)
        .expect("explain query plan")
        .iter()
        .map(|row| row.string("detail").expect("detail"))
        .collect()
}

/// The branch-plan gate, upstream's `expectBranchPlan`: the covering index
/// and primary-key searches, with no temp b-tree and no scan.
pub fn expect_branch_plan(plan: &[String]) {
    assert!(
        plan.iter()
            .any(|detail| detail.contains("SEARCH b USING COVERING INDEX ix_be_seq")),
        "missing covering-index search: {plan:?}"
    );
    assert!(
        plan.iter()
            .any(|detail| detail.contains("SEARCH e USING PRIMARY KEY")),
        "missing primary-key search: {plan:?}"
    );
    assert!(
        !plan.iter().any(|detail| detail.contains("USE TEMP B-TREE")),
        "temp b-tree in plan: {plan:?}"
    );
    assert!(
        !plan.iter().any(|detail| detail.contains("SCAN e")),
        "scan in plan: {plan:?}"
    );
}

/// Runs one conformance case by group and name, the suites' local
/// `registerConformance` runner.
pub async fn run_conformance_case(cases: &[ConformanceCase], group: &str, name: &str) {
    let case = cases
        .iter()
        .find(|case| case.group == group && case.name == name)
        .unwrap_or_else(|| panic!("missing conformance case {group}/{name}"));
    case.run().await;
}

// ---------------------------------------------------------------------------
// Seam doubles
// ---------------------------------------------------------------------------

/// Forwards everything, the base of the wrapping doubles, upstream's
/// `ForwardingDatabase`.
pub struct ForwardingDatabase {
    /// The wrapped handle.
    pub source: Box<dyn SqliteDatabase>,
}

impl SqliteDatabase for ForwardingDatabase {
    fn exec(
        &self,
        statement: &str,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.source.exec(statement)
    }

    fn prepare(&self, sql: &str) -> Box<dyn SqliteStatement> {
        self.source.prepare(sql)
    }

    fn transaction(
        &self,
        callback: pi_session_backend_sqlite_node::sqlite::types::SqliteTransactionCallback,
    ) -> Result<
        Box<dyn std::any::Any + Send>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.source.transaction(callback)
    }

    fn close(
        &self,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.source.close()
    }
}

/// Counts transaction invocations, upstream's `TransactionCountingDatabase`.
pub struct TransactionCountingDatabase {
    /// The wrapped handle.
    pub source: Box<dyn SqliteDatabase>,
    /// The transaction count the test asserts.
    pub transaction_count: AtomicUsize,
}

impl SqliteDatabase for TransactionCountingDatabase {
    fn exec(
        &self,
        statement: &str,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.source.exec(statement)
    }

    fn prepare(&self, sql: &str) -> Box<dyn SqliteStatement> {
        self.source.prepare(sql)
    }

    fn transaction(
        &self,
        callback: pi_session_backend_sqlite_node::sqlite::types::SqliteTransactionCallback,
    ) -> Result<
        Box<dyn std::any::Any + Send>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.transaction_count.fetch_add(1, Ordering::SeqCst);
        self.source.transaction(callback)
    }

    fn close(
        &self,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.source.close()
    }
}

/// Parks the delete's `open_existing` until released, upstream's
/// `GatedOpenExistingFactory`: the deferred-promise gate restates as a thread
/// gate — the worker signals entry over one channel and blocks on the other.
pub struct GatedOpenExistingFactory {
    source: Box<dyn SqliteDatabaseFactory>,
    gated: AtomicBool,
    entered_tx: std::sync::mpsc::Sender<()>,
    release_rx: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl GatedOpenExistingFactory {
    /// Builds the gate with its two channels.
    pub fn new() -> (
        Arc<Self>,
        std::sync::mpsc::Receiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let factory = Arc::new(Self {
            source: Box::new(create_rusqlite_factory()),
            gated: AtomicBool::new(false),
            entered_tx,
            release_rx: Mutex::new(release_rx),
        });
        (factory, entered_rx, release_tx)
    }

    /// Arms the gate for the next `open_existing`.
    pub fn arm(&self) {
        self.gated.store(true, Ordering::SeqCst);
    }
}

impl SqliteDatabaseFactory for GatedOpenExistingFactory {
    fn open(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.source.open(path)
    }

    fn open_existing(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        if self.gated.swap(false, Ordering::SeqCst) {
            self.entered_tx.send(()).expect("entered send");
            self.release_rx
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .recv()
                .expect("release recv");
        }
        self.source.open_existing(path)
    }

    fn open_read_only(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.source.open_read_only(path)
    }
}

/// Fires the snapshot-established hook after the first `FROM sessions` read,
/// upstream's `SnapshotBoundaryStatement`.
struct SnapshotBoundaryStatement {
    source: Box<dyn SqliteStatement>,
    after_snapshot_established: Arc<dyn Fn() + Send + Sync>,
    called: AtomicBool,
}

impl SqliteStatement for SnapshotBoundaryStatement {
    fn run(
        &self,
        params: &SqliteParams,
    ) -> Result<SqliteRunResult, pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError>
    {
        self.source.run(params)
    }

    fn get(
        &self,
        params: &SqliteParams,
    ) -> Result<Option<SqliteRow>, pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError>
    {
        // The hook fires after the underlying read completes: the concurrent
        // writer must commit once the WAL snapshot is established, not before.
        let row = self.source.get(params)?;
        if !self.called.swap(true, Ordering::SeqCst) {
            (self.after_snapshot_established)();
        }
        Ok(row)
    }

    fn all(
        &self,
        params: &SqliteParams,
    ) -> Result<Vec<SqliteRow>, pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError>
    {
        self.source.all(params)
    }

    fn iterate(
        &self,
        params: &SqliteParams,
    ) -> Result<Vec<SqliteRow>, pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError>
    {
        self.source.iterate(params)
    }
}

/// Intercepts `prepare` on the read-only connection, upstream's
/// `SnapshotBoundaryDatabase`.
pub struct SnapshotBoundaryDatabase {
    source: Box<dyn SqliteDatabase>,
    after_snapshot_established: Arc<dyn Fn() + Send + Sync>,
}

impl SqliteDatabase for SnapshotBoundaryDatabase {
    fn exec(
        &self,
        statement: &str,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.source.exec(statement)
    }

    fn prepare(&self, sql: &str) -> Box<dyn SqliteStatement> {
        let statement = self.source.prepare(sql);
        if sql.contains("FROM sessions") {
            Box::new(SnapshotBoundaryStatement {
                source: statement,
                after_snapshot_established: Arc::clone(&self.after_snapshot_established),
                called: AtomicBool::new(false),
            })
        } else {
            statement
        }
    }

    fn transaction(
        &self,
        callback: pi_session_backend_sqlite_node::sqlite::types::SqliteTransactionCallback,
    ) -> Result<
        Box<dyn std::any::Any + Send>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.source.transaction(callback)
    }

    fn close(
        &self,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.source.close()
    }
}

/// Counts read-only opens and wraps them with the boundary hook, upstream's
/// `SnapshotBoundaryFactory`.
pub struct SnapshotBoundaryFactory {
    source: Box<dyn SqliteDatabaseFactory>,
    /// Runs once the fork's snapshot is established.
    pub after_snapshot_established: Arc<dyn Fn() + Send + Sync>,
    /// The read-only open count the tests assert (1 per fork).
    pub read_only_open_count: AtomicUsize,
}

impl SnapshotBoundaryFactory {
    /// Builds the factory over the rusqlite factory.
    pub fn new(after_snapshot_established: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            source: Box::new(create_rusqlite_factory()),
            after_snapshot_established,
            read_only_open_count: AtomicUsize::new(0),
        }
    }
}

impl SqliteDatabaseFactory for SnapshotBoundaryFactory {
    fn open(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.source.open(path)
    }

    fn open_existing(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.source.open_existing(path)
    }

    fn open_read_only(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.read_only_open_count.fetch_add(1, Ordering::SeqCst);
        let source = self.source.open_read_only(path)?;
        Ok(Box::new(SnapshotBoundaryDatabase {
            source,
            after_snapshot_established: Arc::clone(&self.after_snapshot_established),
        }))
    }
}

/// One close-tracking connection, upstream's `CloseTrackingDatabase`: counts
/// close attempts and throws the injected error.
pub struct CloseTrackingDatabase {
    source: Box<dyn SqliteDatabase>,
    /// The close attempts the tests assert.
    pub close_attempts: AtomicUsize,
    /// The error close reports, when injected.
    pub close_error: Mutex<Option<SessionError>>,
}

impl SqliteDatabase for CloseTrackingDatabase {
    fn exec(
        &self,
        statement: &str,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.source.exec(statement)
    }

    fn prepare(&self, sql: &str) -> Box<dyn SqliteStatement> {
        self.source.prepare(sql)
    }

    fn transaction(
        &self,
        callback: pi_session_backend_sqlite_node::sqlite::types::SqliteTransactionCallback,
    ) -> Result<
        Box<dyn std::any::Any + Send>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.source.transaction(callback)
    }

    fn close(
        &self,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.close_attempts.fetch_add(1, Ordering::SeqCst);
        self.source.close()?;
        let injected = self
            .close_error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        injected.map_or(Ok(()), |error| {
            Err(
                pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError::new(
                    error.to_string(),
                ),
            )
        })
    }
}

/// Tracks every writable connection it opened, upstream's
/// `CloseTrackingFactory`.
pub struct CloseTrackingFactory {
    source: Box<dyn SqliteDatabaseFactory>,
    /// The writable connections, in open order.
    pub writable_connections: Mutex<Vec<Arc<CloseTrackingDatabase>>>,
}

impl CloseTrackingFactory {
    /// Builds the factory over the rusqlite factory.
    pub fn new() -> Self {
        Self {
            source: Box::new(create_rusqlite_factory()),
            writable_connections: Mutex::new(Vec::new()),
        }
    }

    /// The open-count shortcut the close-failure tests assert.
    pub fn close_attempts(&self) -> Vec<usize> {
        self.writable_connections
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|connection| connection.close_attempts.load(Ordering::SeqCst))
            .collect()
    }

    /// Injects one connection's close error by open order.
    pub fn fail_close(&self, index: usize, error: SessionError) {
        self.writable_connections
            .lock()
            .unwrap_or_else(PoisonError::into_inner)[index]
            .close_error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .replace(error);
    }
}

impl Default for CloseTrackingFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl SqliteDatabaseFactory for CloseTrackingFactory {
    fn open(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        let connection = Arc::new(CloseTrackingDatabase {
            source: self.source.open(path)?,
            close_attempts: AtomicUsize::new(0),
            close_error: Mutex::new(None),
        });
        self.writable_connections
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::clone(&connection));
        Ok(Box::new(HandleTrackingDatabase { inner: connection }))
    }

    fn open_existing(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        let connection = Arc::new(CloseTrackingDatabase {
            source: self.source.open_existing(path)?,
            close_attempts: AtomicUsize::new(0),
            close_error: Mutex::new(None),
        });
        self.writable_connections
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Arc::clone(&connection));
        Ok(Box::new(HandleTrackingDatabase { inner: connection }))
    }

    fn open_read_only(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.source.open_read_only(path)
    }
}

/// The `Box<dyn SqliteDatabase>` handle over one tracked connection; the
/// trait object owns a clone-counted inner so the factory can still reach it.
struct HandleTrackingDatabase {
    inner: Arc<CloseTrackingDatabase>,
}

impl SqliteDatabase for HandleTrackingDatabase {
    fn exec(
        &self,
        statement: &str,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.inner.exec(statement)
    }

    fn prepare(&self, sql: &str) -> Box<dyn SqliteStatement> {
        self.inner.prepare(sql)
    }

    fn transaction(
        &self,
        callback: pi_session_backend_sqlite_node::sqlite::types::SqliteTransactionCallback,
    ) -> Result<
        Box<dyn std::any::Any + Send>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        self.inner.transaction(callback)
    }

    fn close(
        &self,
    ) -> Result<(), pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError> {
        self.inner.close()
    }
}

/// The storage commit result the erased mutate callbacks return, the suites'
/// downcast helper.
pub fn downcast_commit_result(result: Box<dyn std::any::Any + Send>) -> CommitResult {
    result
        .downcast::<CommitResult>()
        .expect("commit result")
        .as_ref()
        .clone()
}

/// Boxes a future the conformance runner accepts, the fixture factories'
/// shape.
pub fn fixture_future<T: Send + 'static>(
    future: impl Future<Output = T> + Send + 'static,
) -> BoxedFuture<'static, T> {
    Box::pin(future)
}
