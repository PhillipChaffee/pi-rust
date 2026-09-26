//! The session contract surface, ported from upstream
//! `src/harness/session/`.
//!
//! `types.ts` and `values.ts` land with the harness-foundations child —
//! the harness type surface (`AgentLane`, `HarnessEvent`, the snapshots)
//! is their direct consumer. The memory backend, the storage-backed
//! session, the commit/fork machinery, and the conformance testing surface
//! ride the session-layer child; `context.rs` (the context builder) is
//! carried for that child by the runtime-foundations child, which its
//! consumers need first.
pub mod context;
pub mod types;
pub mod values;
