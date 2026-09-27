//! The harness built-in tools, ported from upstream `src/harness/tools/`
//! (map child "pi-agent-core: built-in tools and output utilities").
//!
//! The `utils/` output belt landed with the execution and
//! harness-foundations children. Upstream's `index.ts` barrel restates as
//! the re-exports below.

pub mod bash;
pub mod edit;
pub mod edit_diff;
pub mod file_mutation_queue;
pub mod image;
pub mod path_utils;
pub mod read;
pub mod tool_context;
pub mod write;

pub use bash::{
    BashExecution, BashPrepare, BashToolDetails, BashToolInput, BashToolOptions, create_bash_tool,
};
pub use edit::{EditToolDetails, EditToolInput, create_edit_tool};
pub use read::{
    ReadImageProcessor, ReadImageProcessorOptions, ReadImageProcessorResult, ReadToolDetails,
    ReadToolInput, ReadToolOptions, create_read_tool,
};
pub use tool_context::{EnvToolContext, ExecutionToolContext};
pub use write::{WriteToolInput, create_write_tool};

#[cfg(test)]
mod tests;
