//! The write tool, ported from upstream `src/core/tools/write.ts`.
//!
//! The mutation runs under the file mutation queue; the abort checks ride
//! the queue's own contract — an abort listener rejection would release the
//! queue while an in-flight filesystem operation may still finish, so the
//! checks observe `signal.aborted` after each await instead, upstream's
//! documented `throwIfAborted` shape.

use std::sync::Arc;

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::harness::types::AgentHarnessTool;
use pi_agent_core::types::{AgentToolContent, AgentToolError, AgentToolResult};
use pi_ai::types::{BoxedFuture, TextContent};
use serde_json::{Value, json};

use crate::extensions::types::{ExtensionContext, ToolDefinition};

use super::bash::SystemPromptContribution;
use super::file_mutation_queue::{throw_if_aborted, with_file_mutation_queue};
use super::path_utils::resolve_to_cwd;
use super::tool_definition_wrapper::wrap_cwd_tool;
use super::{io_error, strict_sampling, tool_schema};

/// The write tool's input, upstream's `WriteToolInput`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteToolInput {
    /// The path to write.
    pub path: String,
    /// The content to write.
    pub content: String,
}

/// The pluggable operations for the write tool, upstream's
/// `WriteOperations`. Override to delegate file writing to remote systems.
#[derive(Clone)]
pub struct WriteOperations {
    /// Write content to a file, upstream's `writeFile`.
    pub write_file: WriteWriteFileFn,
    /// Create the directory recursively, upstream's `mkdir`.
    pub mkdir: WriteMkdirFn,
}

/// The erased `writeFile` operation.
pub type WriteWriteFileFn =
    Arc<dyn Fn(String, String) -> BoxedFuture<'static, Result<(), AgentToolError>> + Send + Sync>;
/// The erased `mkdir` operation.
pub type WriteMkdirFn =
    Arc<dyn Fn(String) -> BoxedFuture<'static, Result<(), AgentToolError>> + Send + Sync>;

impl std::fmt::Debug for WriteOperations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteOperations").finish_non_exhaustive()
    }
}

fn default_write_operations() -> WriteOperations {
    WriteOperations {
        write_file: Arc::new(|path, content| {
            Box::pin(async move {
                tokio::fs::write(&path, content.as_bytes())
                    .await
                    .map_err(AgentToolError::from)
            })
        }),
        mkdir: Arc::new(|dir| {
            Box::pin(async move {
                tokio::fs::create_dir_all(&dir)
                    .await
                    .map_err(AgentToolError::from)
            })
        }),
    }
}

/// The write tool's options, upstream's `WriteToolOptions`.
#[derive(Clone, Default)]
pub struct WriteToolOptions {
    /// Custom operations for file writing; default, local filesystem.
    pub operations: Option<WriteOperations>,
}

impl std::fmt::Debug for WriteToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteToolOptions")
            .field("operations", &self.operations.is_some())
            .finish()
    }
}

/// The write tool's system-prompt contribution, upstream's
/// `writeToolSystemPromptContribution`.
pub const WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION: SystemPromptContribution =
    SystemPromptContribution {
        snippet: "Create or overwrite files",
        guidelines: &["Use write only for new files or complete rewrites."],
    };

/// The write tool's schema, upstream's `writeSchema`.
fn write_schema() -> Value {
    tool_schema(
        &json!({
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative or absolute)"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        }),
        &["path", "content"],
    )
}

fn parse_input(params: &Value) -> Result<WriteToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        path: String,
        content: String,
    }
    serde_json::from_value::<RawInput>(params.clone())
        .map(|raw| WriteToolInput {
            path: raw.path,
            content: raw.content,
        })
        .map_err(|error| io_error(error.to_string()))
}

/// The write tool's execution body, upstream's `createWriteToolDefinition`
/// execute.
async fn execute_write_tool(
    params: &Value,
    signal: Option<&AbortSignal>,
    ctx: Option<&dyn ExtensionContext>,
    ops: &WriteOperations,
    cwd: &str,
) -> Result<AgentToolResult, AgentToolError> {
    let input = parse_input(params)?;
    let effective_cwd = ctx
        .map(ExtensionContext::cwd)
        .filter(|ctx_cwd| !ctx_cwd.is_empty())
        .unwrap_or(cwd);
    let absolute_path =
        resolve_to_cwd(&input.path, effective_cwd).map_err(|error| io_error(error.to_string()))?;
    let dir = std::path::Path::new(&absolute_path).parent().map_or_else(
        || absolute_path.clone(),
        |parent| parent.to_string_lossy().into_owned(),
    );

    with_file_mutation_queue(&absolute_path, async {
        throw_if_aborted(signal)?;
        // Create parent directories if needed.
        (ops.mkdir)(dir).await?;
        throw_if_aborted(signal)?;

        // Write the file contents.
        (ops.write_file)(absolute_path.clone(), input.content.clone()).await?;
        throw_if_aborted(signal)?;

        Ok(AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: format!("Successfully wrote to {}", input.path),
                text_signature: None,
            })],
            details: Value::Null,
            usage: None,
            added_tool_names: None,
            terminate: None,
        })
    })
    .await
    .and_then(std::convert::identity)
}

/// Build the write tool definition, upstream's `createWriteToolDefinition`.
#[must_use]
pub fn create_write_tool_definition(
    cwd: &str,
    options: Option<WriteToolOptions>,
) -> ToolDefinition {
    let options = options.unwrap_or_default();
    let ops = Arc::new(options.operations.unwrap_or_else(default_write_operations));
    let cwd = Arc::new(cwd.to_owned());

    ToolDefinition {
        name: "write".to_owned(),
        label: "write".to_owned(),
        description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories."
            .to_owned(),
        prompt_snippet: Some(WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet.to_owned()),
        prompt_guidelines: Some(
            WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION
                .guidelines
                .iter()
                .map(|guideline| (*guideline).to_owned())
                .collect(),
        ),
        parameters: write_schema(),
        constrained_sampling: Some(strict_sampling()),
        render_shell: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(move |_tool_call_id: &str, params: &Value, signal, _on_update, ctx| {
            let ops = Arc::clone(&ops);
            let cwd = Arc::clone(&cwd);
            Box::pin(async move { execute_write_tool(params, signal, ctx, &ops, &cwd).await })
        }),
    }
}

/// Build the write tool, upstream's `createWriteTool`.
#[must_use]
pub fn create_write_tool(cwd: &str, options: Option<WriteToolOptions>) -> AgentHarnessTool {
    let definition = create_write_tool_definition(cwd, options);
    wrap_cwd_tool(definition)
}
