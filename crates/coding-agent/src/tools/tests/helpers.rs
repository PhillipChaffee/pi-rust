//! The shared test harness for the tools suites, the port of upstream's
//! direct `tool.execute(...)` calls: the stable invocation identity, the
//! runner over the wrapped harness tool, and the text extractor.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use pi_agent_core::harness::context::Context;
use pi_agent_core::harness::types::{
    AgentHarnessTool, AgentHarnessToolInvocation, AgentHarnessToolUpdateCallback, ToolContext,
};
use pi_agent_core::types::{AgentToolContent, AgentToolError, AgentToolResult};
use pi_ai::types::BoxedFuture;

/// The stable invocation identity, upstream's `invocation` literal.
#[derive(Debug)]
pub(super) struct TestInvocation;

impl AgentHarnessToolInvocation for TestInvocation {
    fn invocation_id(&self) -> &'static str {
        "test-result"
    }

    fn operation_id(&self) -> &'static str {
        "test-operation"
    }

    fn turn_id(&self) -> &'static str {
        "test-turn"
    }

    fn get_memo(
        &self,
        _name: &str,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<serde_json::Value>, String>> {
        Box::pin(async { Ok(None) })
    }

    fn set_memo(
        &self,
        _name: &str,
        _value: Option<serde_json::Value>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
}

/// Extract the text blocks, upstream's `getTextOutput` helper.
pub(super) fn text_output(result: &AgentToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|part| match part {
            AgentToolContent::Text(text) => Some(text.text.clone()),
            AgentToolContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Run a wrapped tool's execute, upstream's direct `tool.execute(...)`
/// calls. The coding-agent tools run with an empty context slot; the
/// extension context rides the definition-level executes in the suites
/// that need it.
pub(super) async fn run_tool(
    tool: &AgentHarnessTool,
    args: serde_json::Value,
    tool_context: ToolContext,
    on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
    context: &Context,
) -> Result<AgentToolResult, AgentToolError> {
    let invocation = TestInvocation;
    (tool.execute)(
        "test-call",
        &args,
        on_update,
        tool_context,
        &invocation,
        context,
    )
    .await
}

/// Drive a future on an all-enabled current-thread runtime — the child
/// processes and pipe IO the shell suites need, upstream's node event
/// loop.
pub(super) fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(future)
}
/// Applies a unified patch at fuzz factor 0, upstream's `applyPatch` from
/// the npm `diff` dependency standing in as the round-trip oracle: the
/// parser accepts the `FILE_HEADERS_ONLY` shape
/// [`pi_agent_core::harness::tools::edit_diff::generate_unified_patch`]
/// produces, and each hunk applies against the old content's line list at
/// its recorded start.
///
/// # Errors
/// When a hunk's context does not match, upstream's patch-application
/// failure.
pub(super) fn apply_unified_patch(old_content: &str, patch: &str) -> Result<String, String> {
    struct Hunk {
        old_start: usize,
        body: Vec<(char, String)>,
    }
    let mut hunks: Vec<Hunk> = Vec::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("@@ -") {
            let old_start = rest
                .split([',', ' '])
                .next()
                .and_then(|start| start.parse::<usize>().ok())
                .unwrap_or(1);
            hunks.push(Hunk {
                old_start,
                body: Vec::new(),
            });
        } else if !line.starts_with("--- ")
            && !line.starts_with("+++ ")
            && let Some(hunk) = hunks.last_mut()
        {
            let mut characters = line.chars();
            let marker = characters.next().unwrap_or(' ');
            hunk.body.push((marker, characters.as_str().to_owned()));
        }
    }

    let old_lines: Vec<&str> = old_content.split('\n').collect();
    let mut new_lines: Vec<String> = Vec::new();
    let mut consumed: usize = 0;
    for hunk in hunks {
        let anchor = hunk.old_start.saturating_sub(1);
        for line in &old_lines[consumed..anchor.min(old_lines.len())] {
            new_lines.push((*line).to_owned());
        }
        consumed = anchor;
        for (marker, text) in &hunk.body {
            let marker = *marker;
            match marker {
                ' ' | '-' => {
                    let current = old_lines.get(consumed).copied().unwrap_or("");
                    if current != text.as_str() {
                        return Err(format!(
                            "patch mismatch at old line {}: expected {text:?}, found {current:?}",
                            consumed + 1
                        ));
                    }
                    if marker == ' ' {
                        new_lines.push(text.clone());
                    }
                    consumed += 1;
                }
                '+' => new_lines.push(text.clone()),
                // The `\ No newline at end of file` sentinel, upstream's
                // EOFNL marker: jsdiff's apply consumes it as metadata.
                '\\' => {}
                _ => return Err(format!("unsupported hunk marker {marker:?}")),
            }
        }
    }
    for line in &old_lines[consumed.min(old_lines.len())..] {
        new_lines.push((*line).to_owned());
    }
    Ok(new_lines.join("\n"))
}
