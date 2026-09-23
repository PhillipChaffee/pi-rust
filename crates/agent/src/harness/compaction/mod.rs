//! The compaction domain, ported from upstream `src/harness/compaction/`.
//!
//! The shared type surface lands with the harness-foundations child (the
//! hook payloads, the config validators, and the harness options reference
//! it); `compaction.ts`'s logic, `branch-summarization.ts`'s machinery,
//! and the rest of `utils.ts` ride the compaction child.
pub mod types;
