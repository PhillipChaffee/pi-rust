//! The session contract surface, ported from upstream
//! `src/harness/session/`.
//!
//! `types.ts` and `values.ts` land with the harness-foundations child —
//! the harness type surface (`AgentLane`, `HarnessEvent`, the snapshots)
//! is their direct consumer. The memory backend, the storage-backed
//! session, the commit/fork machinery, the context projection, and the
//! conformance testing surface ride the session-layer child; the JSONL
//! backend and its v3 migration ride theirs.

pub mod commit;
pub mod context;
pub mod fork;
pub mod fork_policy;
pub mod in_memory_storage_state;
pub mod memory;
pub mod mutation_line;
#[expect(
    clippy::module_inception,
    reason = "upstream's file is src/harness/session/session.ts; the port mirrors the name"
)]
pub mod session;
pub mod testing;
pub mod types;
pub mod values;
