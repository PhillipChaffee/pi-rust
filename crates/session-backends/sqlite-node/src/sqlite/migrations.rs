//! The initial schema application, upstream's `src/sqlite/migrations.ts`.
//!
//! Upstream reads `migrations/001_initial.sql` from disk at runtime and a
//! build script copies the folder into `dist`; the port embeds the file at
//! compile time (`include_str!`), which a single Rust binary makes free.

use crate::sqlite::types::{SqliteAdapterError, SqliteDatabase};

/// The initial schema, upstream's `migrations/001_initial.sql` byte-verbatim.
pub const INITIAL_SCHEMA_SQL: &str = include_str!("migrations/001_initial.sql");

/// Applies the initial schema, upstream's `applyInitialSchema`.
///
/// The DDL is `CREATE ... IF NOT EXISTS` throughout, so re-applying is inert.
///
/// # Errors
/// A driver failure while executing the schema.
pub fn apply_initial_schema(db: &dyn SqliteDatabase) -> Result<(), SqliteAdapterError> {
    db.exec(INITIAL_SCHEMA_SQL)
}
