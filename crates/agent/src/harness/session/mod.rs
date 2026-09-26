//! The session contract surface, ported from upstream
//! `src/harness/session/`.
//!
//! `types.ts` and `values.ts` land with the harness-foundations child —
//! the harness type surface (`AgentLane`, `HarnessEvent`, the snapshots)
//! is their direct consumer. The memory backend, the storage-backed
//! session, the commit/fork machinery, and the conformance testing surface
//! ride the session-layer child.
pub mod types;
pub mod values;
