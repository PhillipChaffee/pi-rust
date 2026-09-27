//! The write tool, ported from upstream `src/harness/tools/write.ts`.

use std::sync::Arc;

use pi_ai::types::{TextContent, Tool};
use serde_json::json;

use crate::harness::context::Context;
use crate::harness::types::{AgentHarnessTool, AgentHarnessToolExecuteFn, FileContent};
use crate::types::AgentToolError;
use crate::types::{AgentToolContent, AgentToolResult};

use super::file_mutation_queue::with_file_mutation_queue;
use super::path_utils::resolve_tool_path;
use super::tool_context::{
    ExecutionToolContext, aborted_error, execution_tool_context, io_error, is_aborted,
};

/// The write tool's input schema, upstream's `writeSchema` (the typebox
/// wire shape: `type`, `properties`, `required` in property order).
fn write_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative or absolute)"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        },
        "required": ["path", "content"]
    })
}

/// The parsed write-tool input, upstream's `WriteToolInput`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteToolInput {
    /// The file to write.
    pub path: String,
    /// The content to write.
    pub content: String,
}

/// Builds the write tool, upstream's `createWriteTool`.
#[must_use]
pub fn create_write_tool<C: ExecutionToolContext>() -> AgentHarnessTool {
    let execute: Arc<AgentHarnessToolExecuteFn> = Arc::new(
        |_tool_call_id: &str,
         args: &serde_json::Value,
         _on_update,
         tool_context,
         _invocation,
         context: &Context| {
            Box::pin(async move {
                let input: WriteToolInput = parse_input(args)?;
                let context_holder = execution_tool_context::<C>(&tool_context)?;
                let env = context_holder.env();
                let absolute_path = resolve_tool_path(env.as_ref(), &input.path, context).await?;
                let result = with_file_mutation_queue(
                    env,
                    &absolute_path,
                    async {
                        if is_aborted(context) {
                            return Err(aborted_error());
                        }
                        env.write_file(
                            &absolute_path,
                            FileContent::Text(input.content.clone()),
                            context,
                        )
                        .await
                        .map_err(Box::<crate::harness::types::FileError>::from)?;
                        if is_aborted(context) {
                            return Err(aborted_error());
                        }
                        Ok(AgentToolResult {
                            content: vec![AgentToolContent::Text(TextContent {
                                text: format!("Successfully wrote to {}", input.path),
                                text_signature: None,
                            })],
                            details: serde_json::Value::Null,
                            usage: None,
                            added_tool_names: None,
                            terminate: None,
                        })
                    },
                    context,
                )
                .await?;
                Ok(result)
            })
        },
    );
    AgentHarnessTool {
        tool: Tool {
            name: "write".to_owned(),
            description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.".to_owned(),
            parameters: write_schema(),
            constrained_sampling: None,
        },
        label: "write".to_owned(),
        prepare_arguments: None,
        execute,
        replay: None,
        execution_mode: None,
    }
}

fn parse_input(args: &serde_json::Value) -> Result<WriteToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        path: String,
        content: String,
    }
    serde_json::from_value::<RawInput>(args.clone())
        .map(|raw| WriteToolInput {
            path: raw.path,
            content: raw.content,
        })
        .map_err(|error| io_error(error.to_string()))
}
