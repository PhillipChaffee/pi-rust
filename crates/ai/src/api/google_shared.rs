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
    AssistantBlock, AssistantMessage, AssistantMessageEvent, Context, ImageContent, Message,
    Model, ModelThinkingLevel, Modality, StopReason, TextContent, Tool, ToolResultBlock,
    ToolResultMessage, ThinkingContent, ToolCall,
};
use crate::utils::error_body::safe_json_stringify;
use crate::utils::event_stream::AssistantMessageEventStream;

/// Counter for generating unique tool call IDs, upstream's module-level
/// `toolCallCounter` (one per Google adapter file, shared here because the
/// streaming loop it feeds is).
static TOOL_CALL_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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

/// The thinking control a caller can pin for the full-fidelity `stream`
/// entry, upstream's `GoogleOptions["thinking"]` and
/// `GoogleVertexOptions["thinking"]`.
#[derive(Clone, Debug, Default)]
pub struct GoogleThinkingControl {
    /// Whether thinking is enabled; `false` maps a reasoning model to its
    /// disabled-thinking config.
    pub enabled: bool,
    /// The thinking budget; `-1` means dynamic, `0` disables.
    pub budget_tokens: Option<i64>,
    /// The provider-native thinking level, upstream's `GoogleApiThinkingLevel`.
    pub level: Option<String>,
}

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
fn same_provider_and_model(model: &Model, assistant: &AssistantMessage) -> bool {
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

/// The ApiError message the pinned `@google/genai` SDK builds for a failed
/// response: the parsed JSON body stringified, or a synthesized
/// `{ error: { message, code, status } }` object when the body is not JSON.
/// Both Google adapters surface it verbatim, the `messageCarriesBody` pass
/// through.
#[must_use]
pub fn api_error_message(status: u16, body: &str) -> String {
    if let Ok(parsed) = serde_json::from_str::<Value>(body) {
        return safe_json_stringify(&parsed);
    }
    if body.trim().is_empty() {
        return "{}".to_owned();
    }
    json!({
        "error": {
            "message": body,
            "code": status,
            "status": "UNKNOWN",
        }
    })
    .to_string()
}

/// Whether the open block is thinking (`Some(true)`), text (`Some(false)`),
/// or closed (`None`).
type OpenBlock = Option<bool>;

/// The per-chunk state machine the Google adapters share, upstream's
/// `for await` loop over the SDK's `generateContentStream` iterator. Both
/// upstream files carry the identical loop; the port keeps one.
///
/// # Errors
/// With the adapter's error wording on aborts, parse failures, the SDK's
/// pre-stream error probe, and unmapped finish reasons.
pub(crate) async fn consume_google_stream(
    model: &Model,
    response: crate::http::client::HttpResponse,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let mut sse = crate::http::sse::SseStream::new(response.body);
    let mut current_block: OpenBlock = None;
    let mut first_event = true;
    while let Some(sse_event) = sse.next().await.map_err(|error| match error {
        crate::http::client::HttpError::Aborted => "Request was aborted".to_owned(),
        other => other.to_string(),
    })? {
        let value = crate::utils::json_parse::parse_json_with_repair(&sse_event.data)
            .map_err(|error| format!("Invalid Google streaming event: {error}"))?;
        if first_event {
            first_event = false;
            // The SDK's pre-loop probe: a 200 stream whose first JSON event
            // carries an error object with a 4xx/5xx code fails outright.
            if let Some(code) = value
                .get("error")
                .and_then(|error| error.get("code"))
                .and_then(Value::as_i64)
                && (400..600).contains(&code)
            {
                return Err(format!(
                    "got status: {code}. {}",
                    safe_json_stringify(&value)
                ));
            }
        }

        // `GenerateContentResponse.responseId` is an output-only field used
        // to identify each response; keep the first non-empty one.
        if let Some(id) = value.get("responseId").and_then(Value::as_str)
            && !id.is_empty()
            && output
                .response_id
                .as_ref()
                .is_none_or(String::is_empty)
        {
            output.response_id = Some(id.to_owned());
        }

        let candidate = value
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first());
        if let Some(parts) = candidate
            .and_then(|candidate| candidate.get("content"))
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    apply_text_part(
                        text,
                        is_thinking_part(part),
                        part.get("thoughtSignature").and_then(Value::as_str),
                        output,
                        &mut current_block,
                        events,
                    );
                }
                if let Some(function_call) = part.get("functionCall") {
                    close_current_block(&mut current_block, output, events);
                    apply_function_call(function_call, part, output);
                    push_tool_call_events(output, events);
                }
            }
        }

        if let Some(finish_reason) = candidate
            .and_then(|candidate| candidate.get("finishReason"))
            .and_then(Value::as_str)
        {
            output.raw_stop_reason = Some(finish_reason.to_owned());
            output.stop_reason = map_stop_reason(finish_reason)?;
            if output
                .content
                .iter()
                .any(|block| matches!(block, AssistantBlock::ToolCall(_)))
                && output.stop_reason == StopReason::Stop
            {
                output.stop_reason = StopReason::ToolUse;
            }
        }

        if let Some(usage_metadata) = value.get("usageMetadata") {
            let prompt = wire_u64(usage_metadata, "promptTokenCount");
            let cached = wire_u64(usage_metadata, "cachedContentTokenCount");
            let candidates = wire_u64(usage_metadata, "candidatesTokenCount");
            let thoughts = wire_u64(usage_metadata, "thoughtsTokenCount");
            let total = wire_u64(usage_metadata, "totalTokenCount");
            output.usage = crate::types::Usage {
                input: prompt.saturating_sub(cached),
                output: candidates + thoughts,
                cache_read: cached,
                cache_write: 0,
                cache_write_1h: None,
                reasoning: Some(thoughts),
                total_tokens: total,
                cost: crate::types::UsageCost::default(),
            };
            crate::models::calculate_cost(model, &mut output.usage);
        }
    }

    // A stream can settle without closing its last text/thinking block.
    close_current_block(&mut current_block, output, events);
    Ok(())
}

/// Apply one streamed text/thinking part, opening, switching, or growing the
/// current block, upstream's text-part handling.
fn apply_text_part(
    text: &str,
    is_thinking: bool,
    thought_signature: Option<&str>,
    output: &mut AssistantMessage,
    current_block: &mut OpenBlock,
    events: &AssistantMessageEventStream,
) {
    let switch = match current_block {
        None => true,
        Some(open_thinking) => *open_thinking != is_thinking,
    };
    if switch {
        close_current_block(current_block, output, events);
        if is_thinking {
            output.content.push(AssistantBlock::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: None,
                redacted: None,
            }));
            *current_block = Some(true);
            events.push(AssistantMessageEvent::ThinkingStart {
                content_index: output.content.len() as u64 - 1,
                partial: output.clone(),
            });
        } else {
            output.content.push(AssistantBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            }));
            *current_block = Some(false);
            events.push(AssistantMessageEvent::TextStart {
                content_index: output.content.len() as u64 - 1,
                partial: output.clone(),
            });
        }
    }
    let content_index = output.content.len() as u64 - 1;
    if is_thinking {
        let Some(AssistantBlock::Thinking(block)) = output.content.last_mut() else {
            return;
        };
        block.thinking.push_str(text);
        block.thinking_signature =
            retain_thought_signature(block.thinking_signature.as_deref(), thought_signature);
        events.push(AssistantMessageEvent::ThinkingDelta {
            content_index,
            delta: text.to_owned(),
            partial: output.clone(),
        });
    } else {
        let Some(AssistantBlock::Text(block)) = output.content.last_mut() else {
            return;
        };
        block.text.push_str(text);
        block.text_signature =
            retain_thought_signature(block.text_signature.as_deref(), thought_signature);
        events.push(AssistantMessageEvent::TextDelta {
            content_index,
            delta: text.to_owned(),
            partial: output.clone(),
        });
    }
}

/// Close the open text/thinking block, emitting its `*_end` event.
fn close_current_block(
    current_block: &mut OpenBlock,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) {
    if let Some(open_thinking) = *current_block {
        let content_index = output.content.len() as u64 - 1;
        if open_thinking {
            let Some(AssistantBlock::Thinking(block)) = output.content.last() else {
                return;
            };
            events.push(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content: block.thinking.clone(),
                partial: output.clone(),
            });
        } else {
            let Some(AssistantBlock::Text(block)) = output.content.last() else {
                return;
            };
            events.push(AssistantMessageEvent::TextEnd {
                content_index,
                content: block.text.clone(),
                partial: output.clone(),
            });
        }
        *current_block = None;
    }
}

/// Materialize one streamed function call, upstream's function-call part
/// handling: a synthesized id when none arrives or the id duplicates an
/// earlier block, then the three-event start/delta/end sequence.
fn apply_function_call(function_call: &Value, part: &Value, output: &mut AssistantMessage) {
    let provided_id = function_call.get("id").and_then(Value::as_str);
    let needs_new_id = provided_id.is_none_or(|id| id.is_empty())
        || output
            .content
            .iter()
            .any(|block| matches!(block, AssistantBlock::ToolCall(call) if Some(call.id.as_str()) == provided_id));
    let tool_call_id = if needs_new_id {
        let name = function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        format!(
            "{name}_{}_{counter}",
            crate::auth::resolve::now_ms(),
            counter = TOOL_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1,
        )
    } else {
        provided_id.unwrap_or_default().to_owned()
    };

    let arguments = function_call
        .get("args")
        .and_then(Value::as_object)
        .map(Clone::clone)
        .unwrap_or_default();
    let tool_call = ToolCall {
        id: tool_call_id,
        name: function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        arguments,
        thought_signature: part
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .map(str::to_owned),
        namespace: None,
    };
    output.content.push(AssistantBlock::ToolCall(tool_call));
}

/// The three-event function-call sequence, upstream's start/delta/end push.
fn push_tool_call_events(output: &AssistantMessage, events: &AssistantMessageEventStream) {
    let content_index = output.content.len() as u64 - 1;
    let Some(AssistantBlock::ToolCall(tool_call)) = output.content.last() else {
        return;
    };
    events.push(AssistantMessageEvent::ToolcallStart {
        content_index,
        partial: output.clone(),
    });
    events.push(AssistantMessageEvent::ToolcallDelta {
        content_index,
        delta: Value::Object(tool_call.arguments.clone()).to_string(),
        partial: output.clone(),
    });
    events.push(AssistantMessageEvent::ToolcallEnd {
        content_index,
        tool_call: tool_call.clone(),
        partial: output.clone(),
    });
}

fn wire_u64(value: &Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0) as u64
}

