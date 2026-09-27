//! The compaction domain, ported from upstream `src/harness/compaction/`.
//!
//! The type surface ([`types`], the harness-foundations child's shell
//! carried it), the file-operation belt and conversation serializer
//! ([`utils`]), the compaction logic ([`compaction`]), and branch
//! summarization ([`branch_summarization`]).
pub mod branch_summarization;
#[expect(
    clippy::module_inception,
    reason = "upstream's file is src/harness/compaction/compaction.ts; the port mirrors the name"
)]
pub mod compaction;
pub mod types;
pub mod utils;
