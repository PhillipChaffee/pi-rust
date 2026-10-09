//! The edit tool, ported from upstream `src/core/tools/edit.ts`.
//!
//! The fuzzy ladder, line-ending preservation, and the unified patch ride
//! the shared edit-diff machinery; this module carries the tool shell: the
//! legacy-argument shim, the mutation-queue body, the access-error
//! classification, and the diff/patch details.

use std::sync::Arc;

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::harness::tools::edit_diff::{
    Edit, apply_edits_to_normalized_content, detect_line_ending, generate_diff_string,
    generate_unified_patch, normalize_to_lf, restore_line_endings,
};
use pi_agent_core::harness::types::AgentHarnessTool;
use pi_agent_core::types::{AgentToolContent, AgentToolError, AgentToolResult};
use pi_ai::types::{BoxedFuture, TextContent};
use serde_json::{Value, json};

use crate::extensions::types::{ExtensionContext, ToolDefinition};
use crate::utils::text::split_bom;

use super::bash::SystemPromptContribution;
use super::edit_diff::edit_access_error;
use super::file_mutation_queue::with_file_mutation_queue;
use super::io_error;
use super::path_utils::resolve_to_cwd;
use super::tool_definition_wrapper::wrap_tool_definition;

/// The edit tool's input, upstream's `EditToolInput`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditToolInput {
    /// The file to edit.
    pub path: String,
    /// The targeted replacements.
    pub edits: Vec<Edit>,
}

/// The edit tool's details, upstream's `EditToolDetails`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EditToolDetails {
    /// The display-oriented diff of the changes made.
    pub diff: String,
    /// The standard unified patch of the changes made.
    pub patch: String,
    /// The line number of the first change in the new file, for editor
    /// navigation, serialized as `firstChangedLine`.
    pub first_changed_line: Option<usize>,
}

impl EditToolDetails {
    /// The wire shape with the absent member dropped.
    pub(crate) fn to_wire(&self) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("diff".to_owned(), json!(self.diff));
        object.insert("patch".to_owned(), json!(self.patch));
        if let Some(first_changed_line) = self.first_changed_line {
            object.insert("firstChangedLine".to_owned(), json!(first_changed_line));
        }
        Value::Object(object)
    }
}

/// The pluggable operations for the edit tool, upstream's
/// `EditOperations`. Override to delegate file editing to remote systems.
#[derive(Clone)]
pub struct EditOperations {
    /// Read file contents, upstream's `readFile` returning a Buffer.
    pub read_file: EditReadFileFn,
    /// Write content to a file, upstream's `writeFile`.
    pub write_file: EditWriteFileFn,
    /// Check the file is readable and writable (throw if not), upstream's
    /// `access`.
    pub access: EditAccessFn,
}

/// The erased `readFile` operation.
pub type EditReadFileFn =
    Arc<dyn Fn(String) -> BoxedFuture<'static, Result<Vec<u8>, AgentToolError>> + Send + Sync>;
/// The erased `writeFile` operation.
pub type EditWriteFileFn =
    Arc<dyn Fn(String, String) -> BoxedFuture<'static, Result<(), AgentToolError>> + Send + Sync>;
/// The erased `access` operation.
pub type EditAccessFn =
    Arc<dyn Fn(String) -> BoxedFuture<'static, Result<(), AgentToolError>> + Send + Sync>;

impl std::fmt::Debug for EditOperations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditOperations").finish_non_exhaustive()
    }
}

fn default_edit_operations() -> EditOperations {
    EditOperations {
        read_file: Arc::new(|path| {
            Box::pin(async move { tokio::fs::read(&path).await.map_err(AgentToolError::from) })
        }),
        write_file: Arc::new(|path, content| {
            Box::pin(async move {
                tokio::fs::write(&path, content.as_bytes())
                    .await
                    .map_err(AgentToolError::from)
            })
        }),
        access: Arc::new(|path| {
            Box::pin(async move {
                // R_OK | W_OK, upstream's access mask.
                let metadata = tokio::fs::metadata(&path)
                    .await
                    .map_err(AgentToolError::from)?;
                if metadata.permissions().readonly() {
                    return Err(AgentToolError::from(std::io::Error::from(
                        std::io::ErrorKind::PermissionDenied,
                    )));
                }
                Ok(())
            })
        }),
    }
}

/// The edit tool's options, upstream's `EditToolOptions`.
#[derive(Clone, Default)]
pub struct EditToolOptions {
    /// Custom operations for file editing; default, local filesystem.
    pub operations: Option<EditOperations>,
}

impl std::fmt::Debug for EditToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditToolOptions")
            .field("operations", &self.operations.is_some())
            .finish()
    }
}

/// The edit tool's system-prompt contribution, upstream's
/// `editToolSystemPromptContribution`.
pub const EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION: SystemPromptContribution =
    SystemPromptContribution {
        snippet: "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
        guidelines: &[
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
        ],
    };

/// The edit tool's schema, upstream's `editSchema`.
fn edit_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative or absolute)"
            },
            "edits": {
                "type": "array",
                "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text for this targeted edit."
                        }
                    },
                    "required": ["oldText", "newText"]
                }
            }
        },
        "required": ["path", "edits"]
    })
}

/// Whether the value is a single-edit object, upstream's
/// `isSingleEditInput`.
fn is_single_edit_input(value: &Value) -> Option<(&str, &str)> {
    let object = value.as_object()?;
    let old_text = object.get("oldText")?.as_str()?;
    let new_text = object.get("newText")?.as_str()?;
    Some((old_text, new_text))
}

/// The legacy-argument shim, upstream's `prepareEditArguments`.
///
/// Folds the single-edit object and the top-level `oldText`/`newText` forms
/// into `edits`, and parses the JSON-string edits some models send. Invalid
/// shapes pass through untouched, upstream's falls through.
///
/// # Errors
/// The function is total over JSON values: a non-object input and every
/// unrecognized shape return the input unchanged rather than an error.
pub fn prepare_edit_arguments(input: &Value) -> Result<Value, AgentToolError> {
    let Some(object) = input.as_object() else {
        return Ok(input.clone());
    };
    let mut args = object.clone();

    // Some models (Opus 4.6, GLM-5.1) send edits as a JSON string instead of
    // an array. Others send a single edit object instead of a one-element
    // edits array.
    if let Some(edits) = args.get("edits") {
        if let Some(edits_string) = edits.as_str() {
            if let Ok(parsed) = serde_json::from_str::<Value>(edits_string) {
                if parsed.is_array() {
                    args.insert("edits".to_owned(), parsed);
                } else if let Some((old_text, new_text)) = is_single_edit_input(&parsed) {
                    args.insert(
                        "edits".to_owned(),
                        json!([{ "oldText": old_text, "newText": new_text }]),
                    );
                }
            }
        } else if let Some((old_text, new_text)) = is_single_edit_input(edits) {
            args.insert(
                "edits".to_owned(),
                json!([{ "oldText": old_text, "newText": new_text }]),
            );
        }
    }

    let Some(old_text) = args.get("oldText").and_then(Value::as_str) else {
        return Ok(Value::Object(args));
    };
    let Some(new_text) = args.get("newText").and_then(Value::as_str) else {
        return Ok(Value::Object(args));
    };

    let mut edits: Vec<Value> = args
        .get("edits")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    edits.push(json!({ "oldText": old_text, "newText": new_text }));
    args.insert("edits".to_owned(), Value::Array(edits));
    args.remove("oldText");
    args.remove("newText");
    Ok(Value::Object(args))
}

/// Validate the input, upstream's `validateEditInput`.
///
/// # Errors
/// The empty-edits rejection, upstream's throw.
pub(crate) fn validate_edit_input(
    input: &EditToolInput,
) -> Result<(&str, &[Edit]), AgentToolError> {
    if input.edits.is_empty() {
        return Err(io_error(
            "Edit tool input is invalid. edits must contain at least one replacement.",
        ));
    }
    Ok((&input.path, &input.edits))
}

fn parse_input(params: &Value) -> Result<EditToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        path: String,
        edits: Value,
    }
    let raw = serde_json::from_value::<RawInput>(params.clone())
        .map_err(|error| io_error(error.to_string()))?;
    let edits: Vec<Value> = if raw.edits.is_array() {
        serde_json::from_value(raw.edits).map_err(|error| io_error(error.to_string()))?
    } else {
        return Err(io_error("Edit tool input is invalid."));
    };
    let edits: Vec<Edit> = edits
        .into_iter()
        .map(|edit| {
            #[derive(serde::Deserialize)]
            struct RawEdit {
                #[serde(rename = "oldText")]
                old_text: String,
                #[serde(rename = "newText")]
                new_text: String,
            }
            serde_json::from_value::<RawEdit>(edit)
                .map(|raw| Edit {
                    old_text: raw.old_text,
                    new_text: raw.new_text,
                })
                .map_err(|error| io_error(error.to_string()))
        })
        .collect::<Result<Vec<_>, AgentToolError>>()?;
    Ok(EditToolInput {
        path: raw.path,
        edits,
    })
}

/// The edit tool's execution body, upstream's `createEditToolDefinition`
/// execute.
async fn execute_edit_tool(
    params: &Value,
    signal: Option<&AbortSignal>,
    ctx: Option<&dyn ExtensionContext>,
    ops: &EditOperations,
    cwd: &str,
) -> Result<AgentToolResult, AgentToolError> {
    let input = parse_input(params)?;
    let (path, edits) = validate_edit_input(&input)?;
    let path = path.to_owned();
    let edits = edits.to_vec();
    let effective_cwd = ctx
        .map(ExtensionContext::cwd)
        .filter(|ctx_cwd| !ctx_cwd.is_empty())
        .unwrap_or(cwd);
    let absolute_path =
        resolve_to_cwd(&path, effective_cwd).map_err(|error| io_error(error.to_string()))?;

    with_file_mutation_queue(&absolute_path, async {
        // Do not reject from an abort event listener here: that would
        // release the mutation queue while an in-flight filesystem
        // operation may still finish. Checking signal.aborted after each
        // await observes the same aborts while keeping the queue locked
        // until the current operation has settled.
        let throw_if_aborted = || -> Result<(), AgentToolError> {
            if signal.is_some_and(AbortSignal::aborted) {
                return Err(io_error("Operation aborted"));
            }
            Ok(())
        };

        throw_if_aborted()?;

        // Check if file exists.
        if let Err(error) = (ops.access)(absolute_path.clone()).await {
            throw_if_aborted()?;
            let message = error.downcast_ref::<std::io::Error>().map_or_else(
                || format!("Could not edit file: {path}. Error: {error}."),
                |io| edit_access_error(&path, io),
            );
            return Err(io_error(message));
        }
        throw_if_aborted()?;

        // Read the file.
        let buffer = (ops.read_file)(absolute_path.clone()).await?;
        let raw_content = String::from_utf8_lossy(&buffer).into_owned();
        throw_if_aborted()?;

        // Strip BOM before matching. The model will not include an
        // invisible BOM in oldText.
        let split = split_bom(&raw_content);
        let bom = split.bom;
        let content = split.text;
        let original_ending = detect_line_ending(&content);
        let normalized_content = normalize_to_lf(&content);
        let applied = apply_edits_to_normalized_content(&normalized_content, &edits, &path)
            .map_err(io_error)?;
        throw_if_aborted()?;

        let final_content = format!(
            "{bom}{}",
            restore_line_endings(&applied.new_content, original_ending)
        );
        (ops.write_file)(absolute_path.clone(), final_content).await?;
        throw_if_aborted()?;

        let diff_result = generate_diff_string(&applied.base_content, &applied.new_content, 4);
        let unified_patch =
            generate_unified_patch(&path, &applied.base_content, &applied.new_content, 4);
        Ok(AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: format!("Successfully replaced {} block(s) in {path}.", edits.len()),
                text_signature: None,
            })],
            details: EditToolDetails {
                diff: diff_result.diff,
                patch: unified_patch,
                first_changed_line: diff_result.first_changed_line,
            }
            .to_wire(),
            usage: None,
            added_tool_names: None,
            terminate: None,
        })
    })
    .await
    .and_then(std::convert::identity)
}

/// Build the edit tool definition, upstream's `createEditToolDefinition`.
#[must_use]
pub fn create_edit_tool_definition(cwd: &str, options: Option<EditToolOptions>) -> ToolDefinition {
    let options = options.unwrap_or_default();
    let ops = Arc::new(options.operations.unwrap_or_else(default_edit_operations));
    let cwd = Arc::new(cwd.to_owned());

    ToolDefinition {
        name: "edit".to_owned(),
        label: "edit".to_owned(),
        description: "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes."
            .to_owned(),
        prompt_snippet: Some(EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet.to_owned()),
        prompt_guidelines: Some(
            EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION
                .guidelines
                .iter()
                .map(|guideline| (*guideline).to_owned())
                .collect(),
        ),
        parameters: edit_schema(),
        constrained_sampling: Some(pi_ai::types::ConstrainedSamplingSetting::Config(
            pi_ai::types::ConstrainedSamplingConfig::JsonSchema {
                strict: pi_ai::types::Strictness::Prefer,
            },
        )),
        // The edit tool renders its own framing (the diff preview).
        render_shell: Some(crate::extensions::types::RenderShell::SelfRender),
        prepare_arguments: Some(Arc::new(prepare_edit_arguments)),
        execution_mode: None,
        execute: Arc::new(move |_tool_call_id: &str, params: &Value, signal, _on_update, ctx| {
            let ops = Arc::clone(&ops);
            let cwd = Arc::clone(&cwd);
            Box::pin(async move { execute_edit_tool(params, signal, ctx, &ops, &cwd).await })
        }),
    }
}

/// Build the edit tool, upstream's `createEditTool`.
#[must_use]
pub fn create_edit_tool(cwd: &str, options: Option<EditToolOptions>) -> AgentHarnessTool {
    let definition = create_edit_tool_definition(cwd, options);
    wrap_tool_definition::<crate::extensions::types::CwdContext>(definition, None)
}
