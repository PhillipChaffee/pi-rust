//! Shared conversion and vocabulary for the Google Generative AI and Google
//! Vertex wire APIs, ported from `packages/ai/src/api/google-shared.ts` at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//! - Upstream's Gemini wire types (`Content`, `Part`) and the
//!   `FinishReason`/`FunctionCallingConfigMode` enums come from `@google/genai`;
//!   here every Google wire object is a [`serde_json::Value`], and the enums
//!   are their wire strings compared literally.
//! - `sanitizeSurrogates` vanishes: Rust strings are valid UTF-8, so unpaired
//!   surrogates cannot exist (recorded on the utils-belt ticket).
//! - `retryGoogleRequest` collapses into the adapters calling
//!   [`crate::utils::provider_retry::retry_provider_request`] directly: the
//!   upstream normalization that stapled a missing `headers` property onto
//!   `@google/genai`'s `ApiError` has no counterpart, because
//!   [`ProviderRequestError`](crate::utils::provider_retry::ProviderRequestError)
//!   always carries headers.
//! - The exhaustive `never` check in `mapStopReason` becomes an `Err` carrying
//!   the same wording, which the adapters surface as a stream error.

use serde_json::{Value, json};

use crate::api::constrained_sampling::{
    get_json_schema_tool_parameters, resolve_json_schema_strict_sampling,
};
use crate::api::transform_messages::{ToolCallIdNormalizer, transform_messages};
use crate::types::{
    AssistantBlock, Context, ImageContent, Message, Model, ModelThinkingLevel,
    Modality, StopReason, Tool, ToolResultBlock, ToolResultMessage,
};

/// The wire spelling of Google's unspecified thinking level.
pub const GOOGLE_THINKING_LEVEL_UNSPECIFIED: &str = "THINKING_LEVEL_UNSPECIFIED";
/// The wire's `MINIMAL` thinking level.
pub const GOOGLE_THINKING_LEVEL_MINIMAL: &str = "MINIMAL";
/// The wire's `LOW` thinking level.
pub const GOOGLE_THINKING_LEVEL_LOW: &str = "LOW";
/// The wire's `MEDIUM` thinking level.
pub const GOOGLE_THINKING_LEVEL_MEDIUM: &str = "MEDIUM";
/// The wire's `HIGH` thinking level.
pub const GOOGLE_THINKING_LEVEL_HIGH: &str = "HIGH";

/// The Google API thinking levels a model's mapping may resolve to, upstream's
/// `ResolvedGoogleThinkingLevel = Exclude<ThinkingLevel, "xhigh" | "max">`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedGoogleThinkingLevel {
    /// Minimal thinking.
    Minimal,
    /// Low thinking.
    Low,
    /// Medium thinking.
    Medium,
    /// High thinking.
    High,
}

impl ResolvedGoogleThinkingLevel {
    /// The wire value Google's API expects, upstream's enum spelling.
    #[must_use]
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::Minimal => GOOGLE_THINKING_LEVEL_MINIMAL,
            Self::Low => GOOGLE_THINKING_LEVEL_LOW,
            Self::Medium => GOOGLE_THINKING_LEVEL_MEDIUM,
            Self::High => GOOGLE_THINKING_LEVEL_HIGH,
        }
    }

    /// The wire spelling as an owned string, the shape the raw wire carries.
    #[must_use]
    pub fn as_wire_string(self) -> String {
        String::from(self.as_wire())
    }
}

/// Resolve a supported pi level or model-specific Google mapping to a standard
/// Google level.
///
/// # Errors
/// When the model maps a level to an unrecognized value, with upstream's
/// `Unsupported Google thinking level mapping for ...` wording.
pub fn resolve_google_thinking_level(
    model: &Model,
    level: ModelThinkingLevel,
) -> Result<ResolvedGoogleThinkingLevel, String> {
    if level == ModelThinkingLevel::Off {
        return Ok(ResolvedGoogleThinkingLevel::High);
    }

    let mapped = model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&level));
    let resolved = match mapped {
        Some(Some(value)) => value.to_lowercase(),
        // An absent map or missing key resolves to the requested level itself.
        _ => level_wire_name(level).to_lowercase(),
    };
    match resolved.as_str() {
        "minimal" => Ok(ResolvedGoogleThinkingLevel::Minimal),
        "low" => Ok(ResolvedGoogleThinkingLevel::Low),
        "medium" => Ok(ResolvedGoogleThinkingLevel::Medium),
        "high" => Ok(ResolvedGoogleThinkingLevel::High),
        // Upstream stringifies the raw mapping value, so a missing entry reads
        // `undefined` and a present-but-null entry reads `null`.
        _ => Err(format!(
            "Unsupported Google thinking level mapping for {provider_}/{id}: {level} -> {mapped}",
            provider_ = model.provider.0,
            id = model.id,
            level = level_wire_name(level),
            mapped = mapped_display(mapped),
        )),
    }
}

/// The upstream spelling of a [`ModelThinkingLevel`] in error messages.
fn level_wire_name(level: ModelThinkingLevel) -> &'static str {
    match level {
        ModelThinkingLevel::Off => "off",
        ModelThinkingLevel::Minimal => "minimal",
        ModelThinkingLevel::Low => "low",
        ModelThinkingLevel::Medium => "medium",
        ModelThinkingLevel::High => "high",
        ModelThinkingLevel::Xhigh => "xhigh",
        ModelThinkingLevel::Max => "max",
    }
}

/// The `String(mapped)` rendering: `undefined` when no entry exists, `null`
/// when the map carries the wire's `null`.
fn mapped_display(mapped: Option<&Option<String>>) -> String {
    match mapped {
        None => "undefined".to_owned(),
        Some(None) => "null".to_owned(),
        Some(Some(value)) => value.clone(),
    }
}

/// Whether a streamed Gemini part is thinking content, upstream's
/// `isThinkingPart`.
///
/// `thought: true` is the definitive marker; `thoughtSignature` may ride on
/// any part type (text, functionCall, ...) and does not make the part
/// thinking. Signature-bearing parts are preserved as-is and never merged
/// across parts. See <https://ai.google.dev/gemini-api/docs/thought-signatures>.
#[must_use]
pub fn is_thinking_part(part: &Value) -> bool {
    part.get("thought").and_then(Value::as_bool) == Some(true)
}

/// Retain thought signatures during streaming, upstream's
/// `retainThoughtSignature`: some backends only send the signature on the
/// first delta of a block, so the last non-empty signature wins. This never
/// merges or moves signatures across distinct parts.
#[must_use]
pub fn retain_thought_signature(existing: Option<&str>, incoming: Option<&str>) -> Option<String> {
    match incoming {
        Some(value) if !value.is_empty() => Some(value.to_owned()),
        _ => existing.map(str::to_owned),
    }
}

/// Thought signatures must be base64 for Google APIs (`TYPE_BYTES`).
fn is_valid_thought_signature(signature: Option<&str>) -> bool {
    let Some(signature) = signature else {
        return false;
    };
    if signature.is_empty() || signature.len() % 4 != 0 {
        return false;
    }
    signature.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/' || byte == b'='
    })
}

/// Only keep signatures from the same provider/model and with valid base64.
fn resolve_thought_signature(
    is_same_provider_and_model: bool,
    signature: Option<&str>,
) -> Option<String> {
    if is_same_provider_and_model && is_valid_thought_signature(signature) {
        signature.map(str::to_owned)
    } else {
        None
    }
}

/// Models via Google APIs that require explicit tool call IDs in function
/// calls/responses, upstream's `requiresToolCallId`.
#[must_use]
pub fn requires_tool_call_id(model_id: &str) -> bool {
    let gemini_major_version = gemini_major_version(model_id);
    model_id.starts_with("claude-")
        || model_id.starts_with("gpt-oss-")
        || gemini_major_version.is_some_and(|version| version >= 3)
}

fn gemini_major_version(model_id: &str) -> Option<u32> {
    let lowered = model_id.to_lowercase();
    let rest = lowered
        .strip_prefix("gemini-live-")
        .or_else(|| lowered.strip_prefix("gemini-"))?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn supports_multimodal_function_response(model_id: &str) -> bool {
    match gemini_major_version(model_id) {
        Some(version) => version >= 3,
        None => true,
    }
}

/// Convert internal messages to Gemini `Content[]` format, upstream's
/// `convertMessages`.
#[must_use]
pub fn convert_messages(model: &Model, context: &Context) -> Vec<Value> {
    let mut contents: Vec<Value> = Vec::new();
    let normalize_tool_call_id: &ToolCallIdNormalizer<'_> =
        &|id, _model, _source| normalize_google_tool_call_id(id, &model.id);

    let transformed = transform_messages(context.messages.clone(), model, Some(normalize_tool_call_id));

    for msg in transformed {
        match msg {
            Message::User(user) => match &user.content {
                crate::types::UserContent::Text(text) => {
                    contents.push(json!({ "role": "user", "parts": [{ "text": text }] }));
                }
                crate::types::UserContent::Blocks(blocks) => {
                    let parts: Vec<Value> = blocks
                        .iter()
                        .map(|block| match block {
                            crate::types::UserBlock::Text(text) => json!({ "text": text.text }),
                            crate::types::UserBlock::Image(image) => json!({
                                "inlineData": {
                                    "mimeType": image.mime_type,
                                    "data": image.data,
                                }
                            }),
                        })
                        .collect();
                    if parts.is_empty() {
                        continue;
                    }
                    contents.push(json!({ "role": "user", "parts": parts }));
                }
            },
            Message::Assistant(assistant) => {
                let is_same_provider_and_model = same_provider_and_model(model, &assistant);
                let parts = convert_assistant_parts(model, &assistant.content, is_same_provider_and_model);
                if parts.is_empty() {
                    continue;
                }
                contents.push(json!({ "role": "model", "parts": parts }));
            }
            Message::ToolResult(result) => {
                append_tool_result_contents(model, &result, &mut contents);
            }
        }
    }

    contents
}

/// The module's own same-model check, upstream's
/// `msg.provider === model.provider && msg.model === model.id`; the shared
/// [`transform_messages`] gate additionally requires the same api.
fn same_provider_and_model(model: &Model, assistant: &crate::types::AssistantMessage) -> bool {
    assistant.provider == model.provider && assistant.model == model.id
}

fn convert_assistant_parts(
    model: &Model,
    content: &[AssistantBlock],
    is_same_provider_and_model: bool,
) -> Vec<Value> {
    let mut parts: Vec<Value> = Vec::new();
    for block in content {
        match block {
            AssistantBlock::Text(text) => {
                let thought_signature =
                    resolve_thought_signature(is_same_provider_and_model, text.text_signature.as_deref());
                // An empty text block is skipped unless it carries a thought
                // signature: Gemini can attach the signature to a part whose
                // visible text is empty and requires it echoed back; dropping
                // it breaks the reasoning chain and the model intermittently
                // ends mid-task turns with a thought-only STOP.
                if text.text.trim().is_empty() && thought_signature.is_none() {
                    continue;
                }
                let mut part = json!({ "text": text.text });
                if let Some(signature) = thought_signature {
                    part["thoughtSignature"] = json!(signature);
                }
                parts.push(part);
            }
            AssistantBlock::Thinking(thinking) => {
                if is_same_provider_and_model {
                    let thought_signature = resolve_thought_signature(
                        is_same_provider_and_model,
                        thinking.thinking_signature.as_deref(),
                    );
                    // Same rule as text blocks: an empty thinking block is
                    // dropped only when it carries no signature.
                    if thinking.thinking.trim().is_empty() && thought_signature.is_none() {
                        continue;
                    }
                    let mut part = json!({ "thought": true, "text": thinking.thinking });
                    if let Some(signature) = thought_signature {
                        part["thoughtSignature"] = json!(signature);
                    }
                    parts.push(part);
                } else {
                    // Cross-provider/model: the signature is unusable, empty
                    // blocks stay dropped.
                    if thinking.thinking.trim().is_empty() {
                        continue;
                    }
                    parts.push(json!({ "text": thinking.thinking }));
                }
            }
            AssistantBlock::ToolCall(call) => {
                let thought_signature =
                    resolve_thought_signature(is_same_provider_and_model, call.thought_signature.as_deref());
                let mut function_call = json!({
                    "name": call.name,
                    "args": Value::Object(call.arguments.clone()),
                });
                if requires_tool_call_id(&model.id) {
                    function_call["id"] = json!(call.id);
                }
                let mut part = json!({ "functionCall": function_call });
                if let Some(signature) = thought_signature {
                    part["thoughtSignature"] = json!(signature);
                }
                parts.push(part);
            }
        }
    }
    parts
}

/// The Google tool-call id rule, upstream's inline `normalizeToolCallId`.
fn normalize_google_tool_call_id(id: &str, model_id: &str) -> String {
    if !requires_tool_call_id(model_id) {
        return id.to_owned();
    }
    let replaced: String = id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    replaced.chars().take(64).collect()
}

fn append_tool_result_contents(
    model: &Model,
    result: &ToolResultMessage,
    contents: &mut Vec<Value>,
) {
    let text = result
        .content
        .iter()
        .filter_map(|block| match block {
            ToolResultBlock::Text(text) => Some(text.text.as_str()),
            ToolResultBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let image_content: Vec<&ImageContent> = if model.input.contains(&Modality::Image) {
        result
            .content
            .iter()
            .filter_map(|block| match block {
                ToolResultBlock::Image(image) => Some(image),
                ToolResultBlock::Text(_) => None,
            })
            .collect()
    } else {
        Vec::new()
    };

    let has_text = !text.is_empty();
    let has_images = !image_content.is_empty();
    // Gemini 3+ models support multimodal function responses with images
    // nested inside functionResponse.parts. Claude and other non-Gemini
    // models behind Cloud Code Assist / Gemini < 3 still need a separate
    // user image turn.
    let model_supports_multimodal_function_response =
        supports_multimodal_function_response(&model.id);

    // The "output" key marks success and the "error" key marks a failure, per
    // the Gemini SDK documentation's function-response shape.
    let response_value = if has_text {
        text
    } else if has_images {
        String::from("(see attached image)")
    } else {
        String::new()
    };

    let image_parts: Vec<Value> = image_content
        .iter()
        .map(|image| {
            json!({
                "inlineData": {
                    "mimeType": image.mime_type,
                    "data": image.data,
                }
            })
        })
        .collect();

    let mut function_response = json!({
        "name": result.tool_name,
        "response": if result.is_error {
            json!({ "error": response_value })
        } else {
            json!({ "output": response_value })
        },
    });
    if has_images && model_supports_multimodal_function_response {
        function_response["parts"] = json!(image_parts);
    }
    if requires_tool_call_id(&model.id) {
        function_response["id"] = json!(result.tool_call_id);
    }
    let function_response_part = json!({ "functionResponse": function_response });

    // Cloud Code Assist requires all function responses to be in a single
    // user turn; merge into the previous turn when it already carries
    // function responses.
    let is_function_response_turn = contents.last().is_some_and(|last| {
        last.get("role").and_then(Value::as_str) == Some("user")
            && last
                .get("parts")
                .and_then(Value::as_array)
                .is_some_and(|parts| {
                    parts.iter().any(|part| part.get("functionResponse").is_some())
                })
    });
    if is_function_response_turn {
        if let Some(parts) = contents
            .last_mut()
            .and_then(|last| last.get_mut("parts"))
            .and_then(Value::as_array_mut)
        {
            parts.push(function_response_part.clone());
        }
    } else {
        contents.push(json!({ "role": "user", "parts": [function_response_part] }));
    }

    // For Gemini < 3, images ride in a separate trailing user turn.
    if has_images && !model_supports_multimodal_function_response {
        let mut parts = vec![json!({ "text": "Tool result image:" })];
        parts.extend(image_parts);
        contents.push(json!({ "role": "user", "parts": parts }));
    }
}

const JSON_SCHEMA_META_DECLARATIONS: [&str; 8] = [
    "$schema",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$vocabulary",
    "$comment",
    "$defs",
    // Pre-draft-2019-09 equivalent of `$defs`.
    "definitions",
];

/// Strip meta-declarations from a schema object; only plain objects are
/// rewritten and arrays pass through untouched, matching upstream's
/// `sanitizeForOpenApi`.
#[must_use]
pub fn sanitize_for_open_api(schema: &Value) -> Value {
    let Some(object) = schema.as_object() else {
        return schema.clone();
    };
    let mut result = serde_json::Map::new();
    for (key, value) in object {
        if JSON_SCHEMA_META_DECLARATIONS.contains(&key.as_str()) {
            continue;
        }
        result.insert(key.clone(), sanitize_for_open_api(value));
    }
    Value::Object(result)
}

/// Convert tools to Gemini function declarations format, upstream's
/// `convertTools`.
///
/// `parametersJsonSchema` supports full JSON Schema (anyOf, oneOf, const, ...).
/// `use_parameters` selects the legacy `parameters` field instead (OpenAPI 3.03
/// schema), which Cloud Code Assist translates into Anthropic's `input_schema`
/// for Claude models behind Google APIs.
///
/// # Errors
/// When a tool requires strict JSON-schema sampling and its schema cannot be
/// made strict.
pub fn convert_tools(
    tools: &[Tool],
    use_parameters: bool,
    supports_strict_mode: bool,
) -> Result<Option<Value>, String> {
    if tools.is_empty() {
        return Ok(None);
    }
    let mut declarations = Vec::with_capacity(tools.len());
    for tool in tools {
        let strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)?;
        let parameters = get_json_schema_tool_parameters(tool, strict).map_err(|error| error.0)?;
        let declaration = if use_parameters {
            json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": sanitize_for_open_api(&parameters),
            })
        } else {
            json!({
                "name": tool.name,
                "description": tool.description,
                "parametersJsonSchema": parameters,
            })
        };
        declarations.push(declaration);
    }
    Ok(Some(json!([{ "functionDeclarations": declarations }])))
}

/// Gemini 3+ enforces required function parameters in validated tool-calling
/// modes, upstream's `supportsGoogleStrictToolSampling`.
#[must_use]
pub fn supports_google_strict_tool_sampling(model_id: &str) -> bool {
    gemini_major_version(model_id).is_some_and(|version| version >= 3)
}

/// Map a tool choice to the Gemini `FunctionCallingConfigMode` wire string,
/// upstream's `mapToolChoice`.
#[must_use]
pub fn map_tool_choice(choice: &str) -> &'static str {
    match choice {
        "auto" => "AUTO",
        "none" => "NONE",
        "any" => "ANY",
        _ => "AUTO",
    }
}

/// Resolve the function-calling mode for a request, upstream's
/// `resolveGoogleFunctionCallingMode`.
///
/// # Errors
/// When a tool requires strict sampling that cannot be resolved.
pub fn resolve_google_function_calling_mode(
    tools: &[Tool],
    tool_choice: Option<&str>,
    supports_strict_mode: bool,
) -> Result<Option<&'static str>, String> {
    // The strictness resolution propagates its error: a tool whose
    // `strict: "require"` cannot be honored fails the request, upstream's
    // thrown `UnsupportedStrictJsonSchemaError`.
    let use_strict_mode = tools
        .iter()
        .map(|tool| resolve_json_schema_strict_sampling(tool, supports_strict_mode))
        .collect::<Result<Vec<Option<bool>>, String>>()?
        .into_iter()
        .any(|strict| strict == Some(true));
    if matches!(tool_choice, Some("none") | Some("any")) {
        return Ok(Some(map_tool_choice(tool_choice.unwrap_or_default())));
    }
    if use_strict_mode {
        return Ok(Some("VALIDATED"));
    }
    Ok(tool_choice.map(map_tool_choice))
}

/// Map a Gemini `FinishReason` wire string to the pi stop reason, upstream's
/// `mapStopReason` with the enum values as their wire spellings.
///
/// # Errors
/// With upstream's `Unhandled stop reason: ...` wording when Google sends a
/// finish reason outside the mapped set.
pub fn map_stop_reason(reason: &str) -> Result<StopReason, String> {
    match reason {
        "STOP" => Ok(StopReason::Stop),
        "MAX_TOKENS" => Ok(StopReason::Length),
        "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "SAFETY" | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT" | "IMAGE_RECITATION" | "IMAGE_OTHER" | "RECITATION"
        | "FINISH_REASON_UNSPECIFIED" | "OTHER" | "LANGUAGE" | "MALFORMED_FUNCTION_CALL"
        | "UNEXPECTED_TOOL_CALL" | "TOO_MANY_TOOL_CALLS" | "NO_IMAGE" => Ok(StopReason::Error),
        _ => Err(format!("Unhandled stop reason: {reason}")),
    }
}

/// Map a raw string finish reason to the pi stop reason, upstream's
/// `mapStopReasonString`.
#[must_use]
pub fn map_stop_reason_string(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::Stop,
        "MAX_TOKENS" => StopReason::Length,
        _ => StopReason::Error,
    }
}

