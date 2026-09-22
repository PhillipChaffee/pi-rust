//! The Amazon Bedrock Converse Stream wire API, ported from
//! `packages/ai/src/api/bedrock-converse-stream.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream delegates signing (SigV4), the eventstream framing, the full
//! credential chain, and the SDK retries to the official client; this port
//! keeps that division of labor on the official Rust SDK, behind a thin
//! client seam the tests mock — the same boundary upstream's `vi.mock`
//! replaces.
//!
//! Porting restatements:
//! - The Converse Stream command input stays a JSON document shaped exactly
//!   like upstream's `commandInput` (camelCase Converse fields), which is
//!   what `onPayload` sees; the SDK adapter translates it into the SDK's
//!   typed input at the boundary.
//! - `bedrock-provider.ts` upstream is a re-export shim; the credential and
//!   proxy wiring the ticket names lives in the SDK upstream and in this
//!   module's config resolution here.
//! - The Smithy `build`-step custom-header middleware becomes the config's
//!   header list, applied before signing; the deserialize-step response
//!   middleware becomes the SDK adapter's response hook.
//! - The `Uint8Array` redacted-reasoning chunks carry over as base64 in
//!   `thinkingSignature`, upstream's `bytesToBase64` flush.

use serde_json::{Value, json};

use crate::api::constrained_sampling::{
    get_json_schema_tool_parameters, resolve_json_schema_strict_sampling,
};
use crate::api::transform_messages::{ToolCallIdNormalizer, transform_messages};
use crate::types::{
    AssistantBlock, Context, ImageContent, Message, Model, ModelThinkingLevel, StopReason, Tool,
    ToolResultBlock, ToolResultMessage,
};

/// Replaces blank/empty text in user messages and tool results: Bedrock
/// rejects empty content blocks, upstream's `EMPTY_TEXT_PLACEHOLDER`.
pub const EMPTY_TEXT_PLACEHOLDER: &str = "<empty>";
/// The placeholder redacted reasoning carries in `thinking`, upstream's
/// `REDACTED_THINKING_PLACEHOLDER`.
pub const REDACTED_THINKING_PLACEHOLDER: &str = "[Reasoning redacted]";
/// The docs link appended to data-retention-mode errors, upstream's
/// `BEDROCK_DATA_RETENTION_DOCS_URL`.
pub const BEDROCK_DATA_RETENTION_DOCS_URL: &str =
    "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html";
/// Over-long header-derived diagnostic values drop entirely, never truncate,
/// upstream's `MAX_BEDROCK_DIAGNOSTIC_VALUE_CHARS`.
pub const MAX_BEDROCK_DIAGNOSTIC_VALUE_CHARS: usize = 200;

/// The human-readable failure prefix, upstream's `BEDROCK_ERROR_PREFIXES`;
/// the retry and overflow classifiers match on the exact wording.
#[must_use]
pub fn bedrock_error_prefix(name: &str) -> Option<&'static str> {
    match name {
        "InternalServerException" => Some("Internal server error"),
        "ModelStreamErrorException" => Some("Model stream error"),
        "ValidationException" => Some("Validation error"),
        "ThrottlingException" => Some("Throttling error"),
        "ServiceUnavailableException" => Some("Service unavailable"),
        _ => None,
    }
}

/// Whether a model is Claude via Bedrock, upstream's `isAnthropicClaudeModel`:
/// id or name contains `anthropic.claude` or `anthropic/claude`, or the name
/// contains `claude`.
#[must_use]
pub fn is_anthropic_claude_model(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    let name = model.name.to_lowercase();
    id.contains("anthropic.claude")
        || id.contains("anthropic/claude")
        || name.contains("claude")
}

/// The model-candidate keys the catalog scans run over: id plus name, each
/// lowercased with separator runs collapsed, upstream's
/// `getModelMatchCandidates`.
#[must_use]
pub fn model_match_candidates(model: &Model) -> Vec<String> {
    let mut candidates = Vec::with_capacity(2);
    for value in [&model.id, &model.name] {
        let collapsed: String = value
            .to_lowercase()
            .split([' ', '_', '.', ':'])
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("-");
        if !candidates.contains(&collapsed) {
            candidates.push(collapsed);
        }
    }
    candidates
}

/// Whether the model's prompt caching rides on explicit cache points,
/// upstream's `supportsPromptCaching`: Nova has automatic caching (only
/// `AWS_BEDROCK_FORCE_CACHE=1` turns cache points on), and the Claude
/// 3.5 Haiku / 3.7 Sonnet / 4.x / 5.x name families cache. Application
/// inference profiles join the scan through `model.name` when the ARN lacks
/// the model name.
#[must_use]
pub fn supports_prompt_caching(model: &Model, env: Option<&crate::types::ProviderEnv>) -> bool {
    let candidates = model_match_candidates(model);
    if !candidates.iter().any(|candidate| candidate.contains("claude")) {
        return crate::utils::provider_env::get_provider_env_value("AWS_BEDROCK_FORCE_CACHE", env)
            .as_deref()
            == Some("1");
    }
    candidates.iter().any(|candidate| {
        candidate.contains("fable-5")
            || candidate.contains("opus-5")
            || candidate.contains("sonnet-5")
            || candidate.contains("-4-")
            || candidate.contains("claude-3-7-sonnet")
            || candidate.contains("claude-3-5-haiku")
    })
}

/// Whether the model supports native strict tool use, upstream's
/// `model.compat?.supportsStrictMode ?? false`: Bedrock's compat shape
/// defaults to false.
#[must_use]
pub fn supports_strict_mode(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_strict_mode)
        .unwrap_or(false)
}

/// The Bedrock stop-reason mapping, upstream's `mapStopReason` with the wire
/// spellings; the raw value rides on the message either way.
#[must_use]
pub fn map_stop_reason(reason: Option<&str>) -> (StopReason, Option<String>) {
    match reason {
        Some("end_turn") | Some("stop_sequence") => (StopReason::Stop, None),
        Some("max_tokens") | Some("model_context_window_exceeded") => (StopReason::Length, None),
        Some("tool_use") => (StopReason::ToolUse, None),
        Some(reason) => (
            StopReason::Error,
            Some(format!("Provider stopped with: {reason}")),
        ),
        None => (StopReason::Error, None),
    }
}

/// The image format a MIME type maps to, upstream's `ImageFormat` dispatch.
///
/// # Errors
/// With upstream's wording when the MIME type is unknown.
pub fn image_format(mime_type: &str) -> Result<&'static str, String> {
    match mime_type {
        "image/jpeg" | "image/jpg" => Ok("jpeg"),
        "image/png" => Ok("png"),
        "image/gif" => Ok("gif"),
        "image/webp" => Ok("webp"),
        _ => Err(format!("Unknown image type: {mime_type}")),
    }
}

/// Decode a base64 payload to bytes, upstream's `base64ToBytes`; a failed
/// decode drops the block, the caller's rule.
#[must_use]
pub fn base64_to_bytes(value: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(value).ok()
}

/// Encode bytes to base64, upstream's `bytesToBase64`.
#[must_use]
pub fn bytes_to_base64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// The image content block, upstream's `createImageBlock`: wire shape
/// `{ image: { source: { bytes }, format } }` with the data base64-decoded.
///
/// # Errors
/// When the MIME type is unknown or the data is not base64.
pub fn create_image_block(image: &ImageContent) -> Result<Value, String> {
    let format = image_format(&image.mime_type)?;
    let bytes = base64_to_bytes(&image.data)
        .ok_or_else(|| format!("Invalid base64 image data: {}", image.mime_type))?;
    Ok(json!({
        "image": {
            "source": { "bytes": bytes },
            "format": format,
        }
    }))
}

/// Blank text drops, upstream's `createNonBlankTextBlock`.
fn create_non_blank_text_block(text: &str) -> Option<Value> {
    if text.trim().is_empty() {
        None
    } else {
        Some(json!({ "text": text }))
    }
}

/// Blank text becomes the placeholder Bedrock requires, upstream's
/// `createRequiredTextBlock`.
fn create_required_text_block(text: &str) -> Value {
    create_non_blank_text_block(text).unwrap_or_else(|| json!({ "text": EMPTY_TEXT_PLACEHOLDER }))
}

/// Recursively drop empty-string object keys from replayed tool input,
/// upstream's `sanitizeBedrockDocument` — the streaming parser can emit `""`
/// keys Bedrock rejects. Arrays and nested objects recurse; primitives pass
/// through.
#[must_use]
pub fn sanitize_bedrock_document(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| !key.is_empty())
                .map(|(key, value)| (key.clone(), sanitize_bedrock_document(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(sanitize_bedrock_document).collect()),
        other => other.clone(),
    }
}

/// The tool-call id rule, upstream's `normalizeToolCallId`.
fn normalize_bedrock_tool_call_id(id: &str) -> String {
    let replaced: String = id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    replaced.chars().take(64).collect()
}

/// The decoded redacted-thinking payload, upstream's `decodeRedactedContent`;
/// a failed decode drops the block.
fn decode_redacted_content(signature: Option<&str>) -> Option<Vec<u8>> {
    base64_to_bytes(signature?)
}

/// Whether the model replays reasoning with signatures, upstream's
/// `supportsThinkingSignature`: Anthropic Claude models only.
fn supports_thinking_signature(model: &Model) -> bool {
    is_anthropic_claude_model(model)
}

/// The cache-point block, upstream's `cachePoint` literal: a `default` cache
/// point with the 1h TTL when long retention rides.
fn cache_point_block(long: bool) -> Value {
    if long {
        json!({ "cachePoint": { "type": "default", "ttl": "1h" } })
    } else {
        json!({ "cachePoint": { "type": "default" } })
    }
}

/// The Bedrock tool choice, upstream's `BedrockOptions["toolChoice"]`.
#[derive(Clone, Debug, PartialEq)]
pub enum BedrockToolChoice {
    /// Let the provider decide.
    Auto,
    /// The model must call a tool.
    Any,
    /// Never call tools.
    None,
    /// Force a specific tool, upstream's `{ type: "tool", name }`.
    Tool {
        /// The tool to force.
        name: String,
    },
}

/// Whether a model supports adaptive thinking, upstream's
/// `supportsAdaptiveThinking`: the Claude 5.x / Opus 4.6-4.8 / Sonnet 4.6+
/// name families.
#[must_use]
pub fn supports_adaptive_thinking(model: &Model) -> bool {
    model_match_candidates(model).iter().any(|candidate| {
        candidate.contains("opus-4-6")
            || candidate.contains("opus-4-7")
            || candidate.contains("opus-4-8")
            || candidate.contains("opus-5")
            || candidate.contains("sonnet-4-6")
            || candidate.contains("sonnet-5")
            || candidate.contains("fable-5")
    })
}

/// Whether the effort field carries the native `xhigh`, upstream's
/// `supportsNativeXhighEffort`.
fn supports_native_xhigh_effort(model: &Model) -> bool {
    model_match_candidates(model).iter().any(|candidate| {
        candidate.contains("opus-4-7")
            || candidate.contains("opus-4-8")
            || candidate.contains("opus-5")
            || candidate.contains("sonnet-5")
            || candidate.contains("fable-5")
    })
}

/// Map a pi thinking level to the Converse `output_config.effort` value,
/// upstream's `mapThinkingLevelToEffort`: the native `xhigh` only where the
/// model supports it, else the model's mapped value verbatim when it names
/// one, else the minimal/low pair collapsing to `low`.
fn map_thinking_level_to_effort(
    model: &Model,
    level: crate::types::ThinkingLevel,
    thinking_level_map: Option<&crate::types::ThinkingLevelMap>,
) -> String {
    if level == crate::types::ThinkingLevel::Xhigh && supports_native_xhigh_effort(model) {
        return "xhigh".to_owned();
    }
    let key = ModelThinkingLevel::from(level);
    if let Some(map) = thinking_level_map
        && let Some(Some(mapped)) = map.get(&key)
    {
        return mapped.clone();
    }
    match level {
        crate::types::ThinkingLevel::Minimal | crate::types::ThinkingLevel::Low => "low".to_owned(),
        crate::types::ThinkingLevel::Medium => "medium".to_owned(),
        crate::types::ThinkingLevel::High | crate::types::ThinkingLevel::Xhigh
        | crate::types::ThinkingLevel::Max => "high".to_owned(),
    }
}

/// Whether the target is GovCloud, upstream's `isGovCloudBedrockTarget`: the
/// region starts `us-gov-`, or the model id names a `us-gov.` or
/// `arn:aws-us-gov:` profile.
fn is_govcloud_bedrock_target(model: &Model, region: Option<&str>) -> bool {
    region.is_some_and(|region| region.starts_with("us-gov-"))
        || model.id.starts_with("us-gov.")
        || model.id.starts_with("arn:aws-us-gov:")
}

/// The additional-model-request-fields document for thinking, upstream's
/// `buildAdditionalModelRequestFields`: absent without a reasoning request or
/// a non-Claude model, adaptive for the 4.6+/5.x families, fixed-budget
/// otherwise, with the interleaved-thinking beta on non-adaptive Claude and
/// the `display` field dropped for GovCloud targets.
///
/// `level` is the clamped pi reasoning level; `display` carries the thinking
/// display option (default `summarized`).
#[must_use]
pub fn build_additional_model_request_fields(
    model: &Model,
    reasoning: Option<crate::types::ThinkingLevel>,
    thinking_budgets: Option<&crate::types::ThinkingBudgets>,
    interleaved_thinking: Option<bool>,
    thinking_display: Option<&str>,
    region: Option<&str>,
) -> Option<Value> {
    let reasoning = reasoning?;
    if !model.reasoning || !is_anthropic_claude_model(model) {
        return None;
    }
    let display = if is_govcloud_bedrock_target(model, region) {
        None
    } else {
        thinking_display
    };

    let mut additional = serde_json::Map::new();
    if supports_adaptive_thinking(model) {
        let mut thinking = serde_json::Map::new();
        thinking.insert("type".to_owned(), json!("adaptive"));
        if let Some(display) = display {
            thinking.insert("display".to_owned(), json!(display));
        }
        additional.insert("thinking".to_owned(), Value::Object(thinking));
        additional.insert(
            "output_config".to_owned(),
            json!({
                "effort": map_thinking_level_to_effort(
                    model,
                    reasoning,
                    model.thinking_level_map.as_ref(),
                ),
            }),
        );
        return Some(Value::Object(additional));
    }

    let level = match reasoning {
        crate::types::ThinkingLevel::Xhigh | crate::types::ThinkingLevel::Max => {
            crate::types::ThinkingLevel::High
        }
        other => other,
    };
    let budget = thinking_budget_for_level(level, thinking_budgets, reasoning);
    let mut thinking = serde_json::Map::new();
    thinking.insert("type".to_owned(), json!("enabled"));
    thinking.insert("budget_tokens".to_owned(), json!(budget));
    if let Some(display) = display {
        thinking.insert("display".to_owned(), json!(display));
    }
    additional.insert("thinking".to_owned(), Value::Object(thinking));
    if interleaved_thinking.unwrap_or(true) {
        additional.insert(
            "anthropic_beta".to_owned(),
            json!(["interleaved-thinking-2025-05-14"]),
        );
    }
    Some(Value::Object(additional))
}

/// The fixed thinking budget for the clamped level, upstream's default
/// budget table plus the caller's `thinkingBudgets` override.
fn thinking_budget_for_level(
    level: crate::types::ThinkingLevel,
    thinking_budgets: Option<&crate::types::ThinkingBudgets>,
    requested: crate::types::ThinkingLevel,
) -> u64 {
    let custom = thinking_budgets.and_then(|budgets| match level {
        crate::types::ThinkingLevel::Minimal => budgets.minimal,
        crate::types::ThinkingLevel::Low => budgets.low,
        crate::types::ThinkingLevel::Medium => budgets.medium,
        crate::types::ThinkingLevel::High => budgets.high,
        crate::types::ThinkingLevel::Xhigh | crate::types::ThinkingLevel::Max => None,
    });
    if let Some(budget) = custom {
        return budget;
    }
    // The extended levels clamp to the high budget, upstream's default table.
    match requested {
        crate::types::ThinkingLevel::Minimal => 1024,
        _ => match level {
            crate::types::ThinkingLevel::Minimal => 1024,
            crate::types::ThinkingLevel::Low => 2048,
            crate::types::ThinkingLevel::Medium => 8192,
            crate::types::ThinkingLevel::High | crate::types::ThinkingLevel::Xhigh
            | crate::types::ThinkingLevel::Max => 16384,
        },
    }
}

/// Convert internal messages to Bedrock's Message shape, upstream's
/// `convertMessages`: consecutive tool results merge into one user message,
/// assistant blocks degrade per the signature rules, and empty content drops
/// with the `<empty>` placeholder where Bedrock requires content.
///
/// # Errors
/// When an image MIME type is unknown or its data is not base64.
pub fn convert_messages(
    context: &Context,
    model: &Model,
    cache_retention: crate::types::CacheRetention,
    env: Option<&crate::types::ProviderEnv>,
) -> Result<Vec<Value>, String> {
    let normalize: &ToolCallIdNormalizer<'_> =
        &|id, _model, _source| normalize_bedrock_tool_call_id(id);
    let transformed = transform_messages(context.messages.clone(), model, Some(normalize));
    convert_transformed_messages(&transformed, model, cache_retention, env)
}

/// The per-message conversion over the already-transformed history, the
/// second half of upstream's `convertMessages`.
///
/// # Errors
/// When an image MIME type is unknown or its data is not base64.
fn convert_transformed_messages(
    messages: &[Message],
    model: &Model,
    cache_retention: crate::types::CacheRetention,
    env: Option<&crate::types::ProviderEnv>,
) -> Result<Vec<Value>, String> {
    let caching =
        cache_retention != crate::types::CacheRetention::None && supports_prompt_caching(model, env);
    let long = cache_retention == crate::types::CacheRetention::Long;

    let mut wire_messages: Vec<Value> = Vec::new();
    let mut pending_tool_results: Vec<Value> = Vec::new();

    for msg in messages {
        match msg {
            Message::User(user) => {
                flush_tool_results(&mut wire_messages, &mut pending_tool_results, caching, long);
                let content = match &user.content {
                    crate::types::UserContent::Text(text) => {
                        vec![create_required_text_block(text)]
                    }
                    crate::types::UserContent::Blocks(blocks) => {
                        let mut wire_blocks: Vec<Value> = Vec::new();
                        for block in blocks {
                            match block {
                                crate::types::UserBlock::Text(text) => {
                                    if let Some(block) = create_non_blank_text_block(&text.text) {
                                        wire_blocks.push(block);
                                    }
                                }
                                crate::types::UserBlock::Image(image) => {
                                    wire_blocks.push(create_image_block(image)?);
                                }
                            }
                        }
                        if wire_blocks.is_empty() {
                            wire_blocks.push(json!({ "text": EMPTY_TEXT_PLACEHOLDER }));
                        }
                        wire_blocks
                    }
                };
                wire_messages.push(json!({ "role": "user", "content": content }));
            }
            Message::Assistant(assistant) => {
                flush_tool_results(&mut wire_messages, &mut pending_tool_results, caching, long);
                let content = convert_assistant_content(assistant, model)?;
                if content.is_empty() {
                    // Aborted requests carry no content and skip entirely.
                    continue;
                }
                wire_messages.push(json!({ "role": "assistant", "content": content }));
            }
            Message::ToolResult(result) => {
                pending_tool_results.push(convert_tool_result(result)?);
            }
        }
    }
    flush_tool_results(&mut wire_messages, &mut pending_tool_results, caching, long);

    // When caching rides, the last user message carries the trailing cache
    // point, upstream's post-loop append. The merged tool-result turn above
    // already pushed one, so this only fires when the final user message is
    // an ordinary user turn.
    if caching
        && let Some(last) = wire_messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some("user")
        && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
        && !content.iter().any(|block| block.get("cachePoint").is_some())
    {
        content.push(cache_point_block(long));
    }

    Ok(wire_messages)
}

/// Merge the queued tool results into one user message — Bedrock requires
/// all tool results in one message — with the trailing cache point when
/// caching rides.
fn flush_tool_results(
    wire_messages: &mut Vec<Value>,
    pending: &mut Vec<Value>,
    caching: bool,
    long: bool,
) {
    if pending.is_empty() {
        return;
    }
    let mut content = std::mem::take(pending);
    if caching {
        content.push(cache_point_block(long));
    }
    wire_messages.push(json!({ "role": "user", "content": content }));
}

/// The assistant message's Bedrock content blocks, upstream's inline
/// assistant conversion: empty blocks skip, tool use sanitizes its input,
/// and thinking degrades per the signature rules.
fn convert_assistant_content(
    assistant: &crate::types::AssistantMessage,
    model: &Model,
) -> Result<Vec<Value>, String> {
    let mut content: Vec<Value> = Vec::new();
    for block in &assistant.content {
        match block {
            AssistantBlock::Text(text) => {
                if let Some(wire_block) = create_non_blank_text_block(&text.text) {
                    content.push(wire_block);
                }
            }
            AssistantBlock::ToolCall(call) => {
                content.push(json!({
                    "toolUse": {
                        "toolUseId": normalize_bedrock_tool_call_id(&call.id),
                        "name": call.name,
                        "input": sanitize_bedrock_document(&Value::Object(call.arguments.clone())),
                    }
                }));
            }
            AssistantBlock::Thinking(thinking) => {
                let redacted_payload = (thinking.redacted == Some(true))
                    .then(|| decode_redacted_content(thinking.thinking_signature.as_deref()))
                    .flatten();
                if let Some(redacted) = redacted_payload {
                    content.push(json!({
                        "reasoningContent": { "redactedContent": redacted }
                    }));
                    continue;
                }
                if thinking.thinking.trim().is_empty() {
                    continue;
                }
                if supports_thinking_signature(model) {
                    match thinking.thinking_signature.as_deref() {
                        Some(signature) if !signature.trim().is_empty() => {
                            content.push(json!({
                                "reasoningContent": {
                                    "reasoningText": {
                                        "text": thinking.thinking,
                                        "signature": signature,
                                    }
                                }
                            }));
                        }
                        _ => {
                            // Bedrock rejects signature-less replayed reasoning;
                            // fall back to a text block.
                            content.push(json!({ "text": thinking.thinking }));
                        }
                    }
                } else {
                    content.push(json!({
                        "reasoningContent": {
                            "reasoningText": { "text": thinking.thinking }
                        }
                    }));
                }
            }
        }
    }
    Ok(content)
}

/// One tool result as a Bedrock user-content block, the per-message
/// conversion inside the merged tool-result turn.
fn convert_tool_result(result: &ToolResultMessage) -> Result<Value, String> {
    let mut content: Vec<Value> = Vec::new();
    for block in &result.content {
        match block {
            ToolResultBlock::Image(image) => {
                content.push(create_image_block(image)?);
            }
            ToolResultBlock::Text(text) => {
                if let Some(wire_block) = create_non_blank_text_block(&text.text) {
                    content.push(wire_block);
                }
            }
        }
    }
    if content.is_empty() {
        content.push(json!({ "text": EMPTY_TEXT_PLACEHOLDER }));
    }
    Ok(json!({
        "toolResult": {
            "toolUseId": normalize_bedrock_tool_call_id(&result.tool_call_id),
            "content": content,
            "status": if result.is_error { "error" } else { "success" },
        }
    }))
}

/// The system prompt with its trailing cache point, upstream's
/// `buildSystemPrompt`.
#[must_use]
pub fn build_system_prompt(
    system_prompt: Option<&str>,
    model: &Model,
    cache_retention: crate::types::CacheRetention,
    env: Option<&crate::types::ProviderEnv>,
) -> Option<Vec<Value>> {
    let system_prompt = system_prompt?;
    let caching =
        cache_retention != crate::types::CacheRetention::None && supports_prompt_caching(model, env);
    let long = cache_retention == crate::types::CacheRetention::Long;
    let mut blocks = vec![json!({ "text": system_prompt })];
    if caching {
        blocks.push(cache_point_block(long));
    }
    Some(blocks)
}

/// The tool configuration, upstream's `convertToolConfig`: absent without
/// tools or a `none` choice, with the native `strict` flag riding only when
/// the model's compat admits it.
///
/// # Errors
/// When a tool requires strict sampling that cannot be resolved.
pub fn convert_tool_config(
    tools: &[Tool],
    tool_choice: Option<&BedrockToolChoice>,
    supports_strict: bool,
) -> Result<Option<Value>, String> {
    if tools.is_empty() || matches!(tool_choice, Some(BedrockToolChoice::None)) {
        return Ok(None);
    }
    let mut wire_tools = Vec::with_capacity(tools.len());
    for tool in tools {
        let strict = resolve_json_schema_strict_sampling(tool, supports_strict)?;
        let parameters = get_json_schema_tool_parameters(tool, strict).map_err(|error| error.0)?;
        let mut spec = json!({
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": { "json": parameters },
            }
        });
        if strict == Some(true) {
            spec["toolSpec"]["strict"] = json!(true);
        }
        wire_tools.push(spec);
    }
    let mut config = json!({ "tools": wire_tools });
    if let Some(choice) = tool_choice {
        match choice {
            BedrockToolChoice::Auto => {
                config["toolChoice"] = json!({ "auto": {} });
            }
            BedrockToolChoice::Any => {
                config["toolChoice"] = json!({ "any": {} });
            }
            BedrockToolChoice::None => {}
            BedrockToolChoice::Tool { name } => {
                config["toolChoice"] = json!({ "tool": { "name": name } });
            }
        }
    }
    Ok(Some(config))
}
