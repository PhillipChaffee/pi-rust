//! The ls tool, ported from upstream `src/core/tools/ls.ts`.

use std::fmt::Write as _;
use std::sync::Arc;

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::harness::types::AgentHarnessTool;
use pi_agent_core::types::{AgentToolContent, AgentToolError, AgentToolResult};
use pi_ai::types::{BoxedFuture, TextContent};
use serde_json::{Value, json};

use crate::extensions::types::{ExtensionContext, ToolDefinition};

use super::bash::SystemPromptContribution;
use super::io_error;
use super::path_utils::{path_exists, resolve_to_cwd};
use super::tool_definition_wrapper::wrap_tool_definition;
use super::truncate::{DEFAULT_MAX_BYTES, TruncationOptions, truncate_head};

/// The default entry limit, upstream's `DEFAULT_LIMIT`.
pub const DEFAULT_LIMIT: usize = 500;

/// The ls tool's input, upstream's `LsToolInput`.
#[derive(Clone, Debug, PartialEq)]
pub struct LsToolInput {
    /// The directory to list.
    pub path: Option<String>,
    /// The maximum number of entries to return.
    pub limit: Option<f64>,
}

/// The ls tool's details, upstream's `LsToolDetails`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LsToolDetails {
    /// The truncation metadata, when the output truncated.
    pub truncation: Option<super::truncate::TruncationResult>,
    /// The entry limit that was reached, serialized as `entryLimitReached`.
    pub entry_limit_reached: Option<usize>,
}

impl LsToolDetails {
    /// The wire shape with absent members dropped.
    pub(crate) fn to_wire(&self) -> Value {
        let mut object = serde_json::Map::new();
        if let Some(truncation) = &self.truncation {
            object.insert(
                "truncation".to_owned(),
                serde_json::to_value(truncation).unwrap_or_default(),
            );
        }
        if let Some(entry_limit_reached) = self.entry_limit_reached {
            object.insert("entryLimitReached".to_owned(), json!(entry_limit_reached));
        }
        Value::Object(object)
    }
}

/// The pluggable operations for the ls tool, upstream's `LsOperations`.
/// Override to delegate directory listing to remote systems.
#[derive(Clone)]
pub struct LsOperations {
    /// Check the path exists, upstream's `exists`.
    pub exists: LsExistsFn,
    /// Get the path's stats, throwing when not found, upstream's `stat`.
    pub stat: LsStatFn,
    /// Read the directory entries, upstream's `readdir`.
    pub read_dir: LsReadDirFn,
}

/// The erased `exists` operation.
pub type LsExistsFn = Arc<dyn Fn(String) -> BoxedFuture<'static, bool> + Send + Sync>;
/// The erased `stat` operation: `None` models the throw upstream's
/// `isDirectory()` probe rides through.
pub type LsStatFn = Arc<dyn Fn(String) -> BoxedFuture<'static, Option<bool>> + Send + Sync>;
/// The erased `readdir` operation: the error carries the message upstream's
/// catch reports.
pub type LsReadDirFn =
    Arc<dyn Fn(String) -> BoxedFuture<'static, Result<Vec<String>, String>> + Send + Sync>;

impl std::fmt::Debug for LsOperations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LsOperations").finish_non_exhaustive()
    }
}

fn default_ls_operations() -> LsOperations {
    LsOperations {
        exists: Arc::new(|path| Box::pin(async move { path_exists(&path).await })),
        stat: Arc::new(|path| {
            Box::pin(async move {
                tokio::fs::metadata(&path)
                    .await
                    .ok()
                    .map(|metadata| metadata.is_dir())
            })
        }),
        read_dir: Arc::new(|path| {
            Box::pin(async move {
                let mut entries = tokio::fs::read_dir(&path)
                    .await
                    .map_err(|error| error.to_string())?;
                let mut names = Vec::new();
                while let Some(entry) = entries
                    .next_entry()
                    .await
                    .map_err(|error| error.to_string())?
                {
                    names.push(entry.file_name().to_string_lossy().into_owned());
                }
                Ok(names)
            })
        }),
    }
}

/// The ls tool's options, upstream's `LsToolOptions`.
#[derive(Clone, Default)]
pub struct LsToolOptions {
    /// Custom operations for directory listing; default, local filesystem.
    pub operations: Option<LsOperations>,
}

impl std::fmt::Debug for LsToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LsToolOptions")
            .field("operations", &self.operations.is_some())
            .finish()
    }
}

/// The ls tool's system-prompt contribution, upstream's
/// `lsToolSystemPromptContribution`.
pub const LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION: SystemPromptContribution = SystemPromptContribution {
    snippet: "List directory contents",
    guidelines: &[],
};

/// The ls tool's schema, upstream's `lsSchema`.
fn ls_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list (default: current directory)"
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of entries to return (default: 500)"
            }
        }
    })
}

fn parse_input(params: &Value) -> Result<LsToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        path: Option<String>,
        limit: Option<f64>,
    }
    serde_json::from_value::<RawInput>(params.clone())
        .map(|raw| LsToolInput {
            path: raw.path,
            limit: raw.limit,
        })
        .map_err(|error| io_error(error.to_string()))
}

/// The case-insensitive sort, upstream's
/// `a.toLowerCase().localeCompare(b.toLowerCase())` restated to the
/// lexicographic byte order of the lowercased names — no ICU collation.
fn sort_entries(entries: &mut [String]) {
    entries.sort_by(|a, b| {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        a_lower.cmp(&b_lower)
    });
}

/// The ls tool's execution body, upstream's `createLsToolDefinition`
/// execute.
#[expect(
    clippy::too_many_lines,
    reason = "the body mirrors upstream's single execute: the entry walk, the directory/file split, and the notice assembly"
)]
async fn execute_ls_tool(
    params: &Value,
    signal: Option<&AbortSignal>,
    ctx: Option<&dyn ExtensionContext>,
    ops: &LsOperations,
    cwd: &str,
) -> Result<AgentToolResult, AgentToolError> {
    let input = parse_input(params)?;
    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }
    let effective_cwd = ctx
        .map(ExtensionContext::cwd)
        .filter(|ctx_cwd| !ctx_cwd.is_empty())
        .unwrap_or(cwd);
    let dir_path = resolve_to_cwd(input.path.as_deref().unwrap_or("."), effective_cwd)
        .map_err(|error| io_error(error.to_string()))?;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a fractional limit floors to a whole entry count, upstream's Math.max(0, limit)"
    )]
    let effective_limit = input
        .limit
        .map_or(DEFAULT_LIMIT, |limit| limit.max(0.0) as usize)
        .max(1);

    // Check if path exists.
    if !(ops.exists)(dir_path.clone()).await {
        return Err(io_error(format!("Path not found: {dir_path}")));
    }

    // Check if path is a directory.
    let is_directory = (ops.stat)(dir_path.clone())
        .await
        .ok_or_else(|| io_error(format!("Not a directory: {dir_path}")))?;
    if !is_directory {
        return Err(io_error(format!("Not a directory: {dir_path}")));
    }

    // Read directory entries.
    let mut entries = (ops.read_dir)(dir_path.clone())
        .await
        .map_err(|error| io_error(format!("Cannot read directory: {error}")))?;

    // Sort alphabetically, case-insensitive.
    sort_entries(&mut entries);

    // Format entries with directory indicators.
    let mut results: Vec<String> = Vec::new();
    let mut entry_limit_reached = false;
    for entry in entries {
        if results.len() >= effective_limit {
            entry_limit_reached = true;
            break;
        }

        let full_path = std::path::Path::new(&dir_path)
            .join(&entry)
            .to_string_lossy()
            .into_owned();
        let suffix = match (ops.stat)(full_path).await {
            Some(is_directory) if is_directory => "/",
            Some(_) => "",
            // Skip entries we cannot stat.
            None => continue,
        };
        results.push(format!("{entry}{suffix}"));
    }

    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }

    if results.is_empty() {
        return Ok(AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: "(empty directory)".to_owned(),
                text_signature: None,
            })],
            details: Value::Null,
            usage: None,
            added_tool_names: None,
            terminate: None,
        });
    }

    let raw_output = results.join("\n");
    // Apply byte truncation. There is no separate line limit because entry
    // count is already capped.
    let truncation = truncate_head(
        &raw_output,
        TruncationOptions {
            max_lines: Some(usize::MAX),
            ..TruncationOptions::default()
        },
    );
    let mut output = truncation.content.clone();
    let mut details = LsToolDetails::default();
    // Build actionable notices for truncation and entry limits.
    let mut notices: Vec<String> = Vec::new();
    if entry_limit_reached {
        notices.push(format!(
            "{} entries limit reached. Use limit={} for more",
            effective_limit,
            effective_limit * 2
        ));
        details.entry_limit_reached = Some(effective_limit);
    }
    if truncation.truncated {
        notices.push(format!(
            "{} limit reached",
            super::truncate::format_size(DEFAULT_MAX_BYTES)
        ));
        details.truncation = Some(truncation);
    }
    if !notices.is_empty() {
        let _ = write!(output, "\n\n[{}]", notices.join(". "));
    }

    Ok(AgentToolResult {
        content: vec![AgentToolContent::Text(TextContent {
            text: output,
            text_signature: None,
        })],
        details: details.to_wire(),
        usage: None,
        added_tool_names: None,
        terminate: None,
    })
}

/// Build the ls tool definition, upstream's `createLsToolDefinition`.
#[must_use]
pub fn create_ls_tool_definition(cwd: &str, options: Option<LsToolOptions>) -> ToolDefinition {
    let options = options.unwrap_or_default();
    let ops = Arc::new(options.operations.unwrap_or_else(default_ls_operations));
    let cwd = Arc::new(cwd.to_owned());

    ToolDefinition {
        name: "ls".to_owned(),
        label: "ls".to_owned(),
        description: format!(
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to {DEFAULT_LIMIT} entries or {}KB (whichever is hit first).",
            DEFAULT_MAX_BYTES / 1024
        ),
        prompt_snippet: Some(LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet.to_owned()),
        prompt_guidelines: Some(
            LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION
                .guidelines
                .iter()
                .map(|guideline| (*guideline).to_owned())
                .collect(),
        ),
        parameters: ls_schema(),
        constrained_sampling: None,
        render_shell: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(
            move |_tool_call_id: &str, params: &Value, signal, _on_update, ctx| {
                let ops = Arc::clone(&ops);
                let cwd = Arc::clone(&cwd);
                Box::pin(async move { execute_ls_tool(params, signal, ctx, &ops, &cwd).await })
            },
        ),
    }
}

/// Build the ls tool, upstream's `createLsTool`.
#[must_use]
pub fn create_ls_tool(cwd: &str, options: Option<LsToolOptions>) -> AgentHarnessTool {
    let definition = create_ls_tool_definition(cwd, options);
    wrap_tool_definition::<crate::extensions::types::CwdContext>(definition, None)
}
