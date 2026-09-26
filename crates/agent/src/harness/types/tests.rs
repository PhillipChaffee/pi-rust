//! The harness capability and option types' unit tests: the error
//! surfaces' display/source chains and the option structs' debug shapes.
//! Upstream exercises them through the environment and runtime suites;
//! upstream has no dedicated unit file.

use std::sync::Arc;

use crate::harness::types::{
    AgentHarnessTool, AgentHarnessToolContextSource, BranchSummaryError, BranchSummaryErrorCode,
    CompactionError, CompactionErrorCode, ExecutionError, ExecutionErrorCode, FileContent,
    FileError, FileErrorCode, FileKind, ReadTextLinesOptions, RemoveOptions, ShellExecOptions,
    TempFileOptions, TextLine, ToolContext,
};

/// The file and execution errors render their messages and chain their
/// sources.
#[test]
fn the_errors_render_messages_and_chain_sources() {
    let file_error = FileError::new(
        FileErrorCode::NotFound,
        "missing",
        Some("/tmp/missing".to_owned()),
        Some(Box::new(std::io::Error::other("cause"))),
    );
    assert_eq!(file_error.to_string(), "missing");
    let source = std::error::Error::source(&file_error);
    assert!(source.is_some());
    assert_eq!(file_error.code, FileErrorCode::NotFound);

    let execution_error = ExecutionError::new(
        ExecutionErrorCode::Timeout,
        "timed out",
        Some(Box::new(std::io::Error::other("cause"))),
    );
    assert_eq!(execution_error.to_string(), "timed out");
    assert!(std::error::Error::source(&execution_error).is_some());
}

/// The option structs' debug impls surface their fields without the
/// callback noise.
#[test]
fn the_option_debug_impls_surface_their_fields() {
    let exec_options = ShellExecOptions {
        cwd: Some("/tmp".to_owned()),
        ..ShellExecOptions::default()
    };
    let debug = format!("{exec_options:?}");
    assert!(debug.contains("/tmp"));
    assert!(debug.contains("on_update"));
    let text_line = TextLine {
        text: "line".to_owned(),
        terminated: true,
    };
    let debug = format!("{text_line:?}");
    assert!(debug.contains("line"));
    let _ = (
        FileKind::File,
        ReadTextLinesOptions::default(),
        RemoveOptions::default(),
        TempFileOptions::default(),
    );
    assert!(ToolContext::None.is_none());
}

/// The harness tool renders its declarative surface without the executor
/// and names itself; the context source renders both arms opaquely.
#[test]
fn the_tool_surface_renders_and_names_itself() {
    let tool = AgentHarnessTool {
        tool: pi_ai::types::Tool {
            name: "test-tool".to_owned(),
            description: "does things".to_owned(),
            parameters: serde_json::json!({"type": "object"}),
            constrained_sampling: None,
        },
        label: "Test tool".to_owned(),
        prepare_arguments: None,
        execute: Arc::new(|_name, _args, _update, _tool_context, _invocation, _context| {
            Box::pin(async {
                Ok(crate::types::AgentToolResult {
                    content: vec![],
                    details: serde_json::json!({}),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
        replay: None,
        execution_mode: Some(crate::types::ToolExecutionMode::Parallel),
    };
    assert_eq!(tool.name(), "test-tool");
    let debug = format!("{tool:?}");
    assert!(debug.contains("AgentHarnessTool"), "{debug}");
    assert!(debug.contains("Test tool"), "{debug}");
    assert!(debug.contains("execute"), "{debug}");

    let static_source = AgentHarnessToolContextSource::Static(ToolContext::None);
    assert_eq!(
        format!("{static_source:?}"),
        "AgentHarnessToolContextSource::Static(..)"
    );
    let resolved_source =
        AgentHarnessToolContextSource::Resolved(Arc::new(|_context| Box::pin(async { None })));
    assert_eq!(
        format!("{resolved_source:?}"),
        "AgentHarnessToolContextSource::Resolved(..)"
    );
}

/// The compaction and branch-summary errors render their messages and
/// chain their sources through the typed error contract.
#[test]
fn the_compaction_and_branch_summary_errors_render_and_chain() {
    let compaction_error = CompactionError::new(
        CompactionErrorCode::SummarizationFailed,
        "summary failed",
        Some(Box::new(std::io::Error::other("cause"))),
    );
    assert_eq!(compaction_error.to_string(), "summary failed");
    assert_eq!(
        compaction_error.code,
        CompactionErrorCode::SummarizationFailed
    );
    let source = std::error::Error::source(&compaction_error);
    assert_eq!(
        source.map(ToString::to_string),
        Some("cause".to_owned())
    );

    let branch_error = BranchSummaryError::new(
        BranchSummaryErrorCode::Aborted,
        "aborted",
        Some(Box::new(std::io::Error::other("branch cause"))),
    );
    assert_eq!(branch_error.to_string(), "aborted");
    assert_eq!(branch_error.code, BranchSummaryErrorCode::Aborted);
    let source = std::error::Error::source(&branch_error);
    assert_eq!(
        source.map(ToString::to_string),
        Some("branch cause".to_owned())
    );

    let unchained_branch_error =
        BranchSummaryError::new(BranchSummaryErrorCode::SummarizationFailed, "failed", None);
    assert!(std::error::Error::source(&unchained_branch_error).is_none());
}

/// `FileContent` converts from borrowed text, owned text, and raw bytes.
#[test]
fn the_file_content_conversions_wrap_text_and_bytes() {
    assert_eq!(
        FileContent::from("borrowed"),
        FileContent::Text("borrowed".to_owned())
    );
    assert_eq!(
        FileContent::from(String::from("owned")),
        FileContent::Text("owned".to_owned())
    );
    assert_eq!(
        FileContent::from(vec![1_u8, 2]),
        FileContent::Bytes(vec![1, 2])
    );
}
