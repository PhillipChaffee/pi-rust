//! The edit tool, ported from upstream `src/harness/tools/edit.ts`.

use std::sync::Arc;

use pi_ai::types::{TextContent, Tool};
use serde_json::json;

use crate::harness::context::Context;
use crate::harness::types::{
    AgentHarnessTool, AgentHarnessToolExecuteFn, FileContent, FileError, FileErrorCode, FileKind,
};
use crate::types::AgentToolError;
use crate::types::{AgentToolContent, AgentToolResult};

use super::edit_diff::{
    Edit, apply_edits_to_normalized_content, detect_line_ending, generate_diff_string,
    generate_unified_patch, normalize_to_lf, restore_line_endings, strip_bom,
};
use super::file_mutation_queue::with_file_mutation_queue;
use super::path_utils::resolve_tool_path;
use super::tool_context::{
    ExecutionToolContext, aborted_error, execution_tool_context, io_error, is_aborted,
};

/// The edit tool's input schema, upstream's `editSchema`.
fn edit_schema() -> serde_json::Value {
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

/// The parsed edit-tool input, upstream's `EditToolInput`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditToolInput {
    /// The file to edit.
    pub path: String,
    /// The targeted replacements.
    pub edits: Vec<Edit>,
}

/// The edit tool's structured details, upstream's `EditToolDetails`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EditToolDetails {
    /// The display diff with line numbers.
    pub diff: String,
    /// The unified patch.
    pub patch: String,
    /// The first changed line number in the new file.
    pub first_changed_line: Option<usize>,
}

/// Whether one JSON value is the single-edit shape, upstream's
/// `isSingleEditInput`.
fn is_single_edit_input(value: &serde_json::Value) -> bool {
    value.as_object().is_some_and(|edit| {
        edit.get("oldText")
            .is_some_and(serde_json::Value::is_string)
            && edit
                .get("newText")
                .is_some_and(serde_json::Value::is_string)
    })
}

/// The pre-validation argument shim, upstream's `prepareEditArguments`: it
/// accepts `edits` as a JSON string, as a single-edit object, and the
/// legacy flat `oldText`/`newText` fields.
fn prepare_edit_arguments(input: &serde_json::Value) -> serde_json::Value {
    let Some(args) = input.as_object() else {
        return input.clone();
    };
    let mut args = args.clone();
    match args.get("edits") {
        Some(serde_json::Value::String(text)) => {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(text);
            match parsed {
                Ok(parsed @ serde_json::Value::Array(_)) => {
                    args.insert("edits".to_owned(), parsed);
                }
                Ok(parsed) if is_single_edit_input(&parsed) => {
                    args.insert("edits".to_owned(), json!([parsed]));
                }
                _ => {}
            }
        }
        Some(edits) if is_single_edit_input(edits) => {
            args.insert("edits".to_owned(), json!([edits]));
        }
        _ => {}
    }

    let legacy_old = args.get("oldText").and_then(serde_json::Value::as_str);
    let legacy_new = args.get("newText").and_then(serde_json::Value::as_str);
    if let (Some(old_text), Some(new_text)) = (legacy_old, legacy_new) {
        let mut edits = match args.get("edits") {
            Some(serde_json::Value::Array(existing)) => existing.clone(),
            _ => Vec::new(),
        };
        edits.push(json!({ "oldText": old_text, "newText": new_text }));
        args.insert("edits".to_owned(), serde_json::Value::Array(edits));
        args.remove("oldText");
        args.remove("newText");
    }
    serde_json::Value::Object(args)
}

/// Validates and returns the input, upstream's `validateEditInput`.
fn validate_edit_input(input: &EditToolInput) -> Result<(&str, &[Edit]), AgentToolError> {
    if input.edits.is_empty() {
        return Err(Box::new(std::io::Error::other(
            "Edit tool input is invalid. edits must contain at least one replacement.",
        )));
    }
    Ok((&input.path, &input.edits))
}

/// The file-access failure, upstream's `editAccessError` (`Error` with the
/// `FileError` as its `cause`).
#[derive(Debug)]
struct EditAccessError {
    message: String,
    source: FileError,
}

impl std::fmt::Display for EditAccessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EditAccessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

fn edit_access_error(path: &str, error: FileError) -> AgentToolError {
    Box::new(EditAccessError {
        message: format!(
            "Could not edit file: {path}. Error code: {}.",
            file_error_code(error.code)
        ),
        source: error,
    })
}

/// The error code's wire spelling, upstream's `error.code` string.
const fn file_error_code(code: FileErrorCode) -> &'static str {
    match code {
        FileErrorCode::Aborted => "aborted",
        FileErrorCode::NotFound => "not_found",
        FileErrorCode::PermissionDenied => "permission_denied",
        FileErrorCode::NotDirectory => "not_directory",
        FileErrorCode::IsDirectory => "is_directory",
        FileErrorCode::Invalid => "invalid",
        FileErrorCode::NotSupported => "not_supported",
        FileErrorCode::Unknown => "unknown",
    }
}

/// Builds the edit tool, upstream's `createEditTool`.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "the executor mirrors upstream's single createEditTool execute body: the validation, normalization, and diff steps read the same file state"
)]
pub fn create_edit_tool<C: ExecutionToolContext>() -> AgentHarnessTool {
    let execute: Arc<AgentHarnessToolExecuteFn> = Arc::new(
        |_tool_call_id: &str,
         args: &serde_json::Value,
         _on_update,
         tool_context,
         _invocation,
         context: &Context| {
            Box::pin(async move {
                let input: EditToolInput = parse_input(args)?;
                let (path, edits) = validate_edit_input(&input)?;
                let context_holder = execution_tool_context::<C>(&tool_context)?;
                let env = context_holder.env();
                let absolute_path = resolve_tool_path(env.as_ref(), path, context).await?;
                let outcome = with_file_mutation_queue(
                    env,
                    &absolute_path,
                    async {
                        if is_aborted(context) {
                            return Err(aborted_error());
                        }
                        let info = env
                            .file_info(&absolute_path, context)
                            .await
                            .map_err(|error| edit_access_error(path, error))?;
                        if info.kind != FileKind::File && info.kind != FileKind::Symlink {
                            return Err(io_error(format!(
                                "Could not edit file: {path}. Path is not a file."
                            )));
                        }

                        let read_result = env
                            .read_text_file(&absolute_path, context)
                            .await
                            .map_err(|error| edit_access_error(path, error))?;
                        if is_aborted(context) {
                            return Err(aborted_error());
                        }

                        let bom_split = strip_bom(&read_result);
                        let original_ending = detect_line_ending(&bom_split.text);
                        let normalized_content = normalize_to_lf(&bom_split.text);
                        let applied =
                            apply_edits_to_normalized_content(&normalized_content, edits, path)
                                .map_err(io_error)?;
                        if is_aborted(context) {
                            return Err(aborted_error());
                        }

                        let final_content = format!(
                            "{}{}",
                            bom_split.bom,
                            restore_line_endings(&applied.new_content, original_ending)
                        );
                        env.write_file(&absolute_path, FileContent::Text(final_content), context)
                            .await
                            .map_err(|error| edit_access_error(path, error))?;
                        if is_aborted(context) {
                            return Err(aborted_error());
                        }

                        let diff_result =
                            generate_diff_string(&applied.base_content, &applied.new_content, 4);
                        Ok(AgentToolResult {
                            content: vec![AgentToolContent::Text(TextContent {
                                text: format!(
                                    "Successfully replaced {} block(s) in {path}.",
                                    edits.len()
                                ),
                                text_signature: None,
                            })],
                            details: serialize_details(&EditToolDetails {
                                diff: diff_result.diff,
                                patch: generate_unified_patch(
                                    path,
                                    &applied.base_content,
                                    &applied.new_content,
                                    4,
                                ),
                                first_changed_line: diff_result.first_changed_line,
                            }),
                            usage: None,
                            added_tool_names: None,
                            terminate: None,
                        })
                    },
                    context,
                )
                .await?;
                Ok(outcome)
            })
        },
    );
    AgentHarnessTool {
        tool: Tool {
            name: "edit".to_owned(),
            description: "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.".to_owned(),
            parameters: edit_schema(),
            constrained_sampling: None,
        },
        label: "edit".to_owned(),
        prepare_arguments: Some(Arc::new(|input: &serde_json::Value| {
            Ok(prepare_edit_arguments(input))
        })),
        execute,
        replay: None,
        execution_mode: None,
    }
}

/// Serializes the edit details, upstream's `EditToolDetails`.
fn serialize_details(details: &EditToolDetails) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    object.insert("diff".to_owned(), json!(details.diff));
    object.insert("patch".to_owned(), json!(details.patch));
    if let Some(first_changed_line) = details.first_changed_line {
        object.insert("firstChangedLine".to_owned(), json!(first_changed_line));
    }
    serde_json::Value::Object(object)
}

fn parse_input(args: &serde_json::Value) -> Result<EditToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct RawEdit {
        old_text: String,
        new_text: String,
    }
    #[derive(serde::Deserialize)]
    struct RawInput {
        path: String,
        edits: serde_json::Value,
    }
    let raw: RawInput =
        serde_json::from_value(args.clone()).map_err(|error| io_error(error.to_string()))?;
    let edits = if let Some(items) = raw.edits.as_array() {
        items
            .iter()
            .map(|item| {
                serde_json::from_value::<RawEdit>(item.clone())
                    .map(|raw| Edit {
                        old_text: raw.old_text,
                        new_text: raw.new_text,
                    })
                    .map_err(|error| io_error(error.to_string()))
            })
            .collect::<Result<Vec<Edit>, AgentToolError>>()?
    } else {
        return Err(Box::new(std::io::Error::other(
            "Edit tool input is invalid. edits must contain at least one replacement.",
        )));
    };
    Ok(EditToolInput {
        path: raw.path,
        edits,
    })
}
