//! The SQLite session backend core, mirroring upstream's `src/sqlite/` tree:
//! the repo lifecycle, storage, the open-session facade, the row modules,
//! the query builder, and the structural seam types.

pub mod branch_entries;
pub mod entries;
pub mod migrations;
pub mod repo;
pub mod session;
pub mod session_row;
pub mod session_sequences;
pub mod session_stats;
pub mod sql;
pub mod storage;
pub mod types;
pub mod usage_ledger;
pub mod values;
