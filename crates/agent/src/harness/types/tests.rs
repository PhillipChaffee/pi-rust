//! The harness capability and option types' unit tests: the error
//! surfaces' display/source chains and the option structs' debug shapes.
//! Upstream exercises them through the environment and runtime suites;
//! upstream has no dedicated unit file.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use crate::harness::types::{
    ExecutionError, ExecutionErrorCode, FileError, FileErrorCode, FileKind, ReadTextLinesOptions,
    RemoveOptions, ShellExecOptions, TempFileOptions, TextLine, ToolContext,
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
    let _ = (FileKind::File, ReadTextLinesOptions::default(), RemoveOptions::default(), TempFileOptions::default());
    assert!(ToolContext::None.is_none());
}
