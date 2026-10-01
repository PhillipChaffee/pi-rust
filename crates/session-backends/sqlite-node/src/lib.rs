//! SQLite session backend for [`pi_agent_core`] sessions, the port of
//! `@earendil-works/pi-session-backend-sqlite-node` at upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The backend persists agent sessions as SQLite containers: one
//! `{sessionId}.sqlite` file per session by default, or one shared container
//! per [`crate::SqliteSessionRepoOptions`]. It implements the
//! `Storage`/`SessionRepo`/`Session` contracts from `pi-agent-core`'s harness
//! session layer; ids containing only ASCII letters, digits, `_`, and `-`
//! keep their `{id}.sqlite` filename and every other id is encoded as a
//! `~`-prefixed base64url of its UTF-16 code units.
//!
//! The driver seam ([`crate::sqlite::types`]) is structural: tests wrap it
//! with counting, gating, interception, and close-tracking doubles, exactly
//! like upstream; [`crate::rusqlite_adapter`] is the driver binding (upstream
//! wraps `node:sqlite`).

pub mod rusqlite_adapter;
pub mod sqlite;

pub use rusqlite_adapter::{
    RusqliteDatabase, RusqliteDatabaseFactory, RusqliteStatement, create_rusqlite_factory,
    wrap_rusqlite_connection,
};
pub use sqlite::migrations::{INITIAL_SCHEMA_SQL, apply_initial_schema};
pub use sqlite::repo::{
    SQLITE_SESSION_EXTENSION, SQLITE_STORAGE_VERSION, SqliteSessionCreateOptions,
    SqliteSessionRepo, SqliteSessionRepoOptions,
};
// Upstream's `sqlite/index.ts` barrel keeps the open-session facade and the
// schema helper internal; the Rust crate exports both because the typed
// lifecycle's return type and the tests need them (recorded barrel delta).
pub use sqlite::session::{SqliteOpenSession, SqliteOpenSessionOptions};
pub use sqlite::sql::{SqlQuery, join_sql_fragments};
pub use sqlite::storage::{SqliteStorage, SqliteStorageOptions, SqliteStorageSnapshot};
pub use sqlite::types::{
    SqliteAdapterError, SqliteDatabase, SqliteDatabaseFactory, SqliteParams, SqliteRow,
    SqliteRunResult, SqliteStatement, SqliteTransactionOutcome, SqliteValue,
};
