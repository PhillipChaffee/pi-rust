//! Compaction and summarization utilities, upstream
//! `src/core/compaction/` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

pub mod branch_summarization;
#[expect(
    clippy::module_inception,
    reason = "upstream's own core/compaction/compaction.ts layout"
)]
pub mod compaction;
pub mod utils;

pub use branch_summarization::{
    BranchPreparation, BranchSummaryDetails, BranchSummaryResult, CollectEntriesResult,
    GenerateBranchSummaryOptions, collect_entries_for_branch_summary, generate_branch_summary,
    prepare_branch_entries,
};
pub use compaction::{
    CompactionDetails, CompactionError, CompactionPreparation, CompactionResult,
    CompactionSettings, ContextUsageEstimate, CutPointResult, DEFAULT_COMPACTION_SETTINGS,
    SummaryWithUsage, calculate_context_tokens, compact, complete_summarization,
    estimate_context_tokens, estimate_tokens, find_cut_point, find_turn_start_index,
    generate_summary, generate_summary_with_usage, get_last_assistant_usage,
    get_summarization_failure, prepare_compaction, should_compact,
};
pub use utils::{
    FileLists, FileOperations, SUMMARIZATION_SYSTEM_PROMPT, compute_file_lists, create_file_ops,
    extract_file_ops_from_message, format_file_operations, serialize_conversation,
};
