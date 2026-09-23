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
//! - The failure diagnostic reads a typed [`BedrockStreamFailure`] instead
//!   of duck-typed SDK error objects; the `instanceof
//!   BedrockRuntimeServiceException` prefix gate becomes a name lookup, which
//!   also keeps the SDK's `Unknown` placeholder from branding a gateway
//!   failure.
//! - The wire events the seam yields are JSON values in the Converse Stream
//!   frame shapes; the SDK adapter's `redactedContent` blobs ride as base64
//!   strings, and the replay path carries them as byte arrays the way the
//!   JSON command input does.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::StreamExt;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::api::bedrock_options::{self, BedrockClientConfig};
use crate::api::constrained_sampling::{
    get_json_schema_tool_parameters, resolve_json_schema_strict_sampling,
};
use crate::api::simple_options::{
    MIN_ANSWER_TOKENS, adjust_max_tokens_for_thinking, build_base_options,
    clamp_max_tokens_to_context, clamp_reasoning,
};
use crate::api::transform_messages::{ToolCallIdNormalizer, transform_messages};
use crate::api::wire_common::initial_output;
use crate::types::{
    AssistantBlock, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
    BoxedFuture, Context, ImageContent, Message, Model, ModelThinkingLevel, SimpleStreamOptions,
    StopReason, TextContent, ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolResultBlock,
    ToolResultMessage,
};
use crate::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use crate::utils::json_parse::parse_streaming_json;

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
    id.contains("anthropic.claude") || id.contains("anthropic/claude") || name.contains("claude")
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
/// upstream's `supportsPromptCaching`.
///
/// Nova has automatic caching (only `AWS_BEDROCK_FORCE_CACHE=1` turns cache
/// points on), and the Claude 3.5 Haiku / 3.7 Sonnet / 4.x / 5.x name
/// families cache. Application inference profiles join the scan through
/// `model.name` when the ARN lacks the model name.
#[must_use]
pub fn supports_prompt_caching(model: &Model, env: Option<&crate::types::ProviderEnv>) -> bool {
    let candidates = model_match_candidates(model);
    if !candidates
        .iter()
        .any(|candidate| candidate.contains("claude"))
    {
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
        Some("end_turn" | "stop_sequence") => (StopReason::Stop, None),
        Some("max_tokens" | "model_context_window_exceeded") => (StopReason::Length, None),
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
/// upstream's `sanitizeBedrockDocument`.
///
/// The streaming parser can emit `""` keys Bedrock rejects. Arrays and
/// nested objects recurse; primitives pass through.
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
    crate::api::google_shared::sanitize_tool_call_id(id)
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
    level: ThinkingLevel,
    thinking_level_map: Option<&crate::types::ThinkingLevelMap>,
) -> String {
    if level == ThinkingLevel::Xhigh && supports_native_xhigh_effort(model) {
        return "xhigh".to_owned();
    }
    let key = ModelThinkingLevel::from(level);
    if let Some(map) = thinking_level_map
        && let Some(Some(mapped)) = map.get(&key)
    {
        return mapped.clone();
    }
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low".to_owned(),
        ThinkingLevel::Medium => "medium".to_owned(),
        ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => "high".to_owned(),
    }
}

/// Whether the target is `GovCloud`, upstream's `isGovCloudBedrockTarget`: the
/// region starts `us-gov-`, or the model id names a `us-gov.` or
/// `arn:aws-us-gov:` profile.
fn is_govcloud_bedrock_target(model: &Model, region: Option<&str>) -> bool {
    region.is_some_and(|region| region.starts_with("us-gov-"))
        || model.id.starts_with("us-gov.")
        || model.id.starts_with("arn:aws-us-gov:")
}

/// The additional-model-request-fields document for thinking, upstream's
/// `buildAdditionalModelRequestFields`.
///
/// Absent without a reasoning request or a non-Claude model; adaptive for
/// the 4.6+/5.x families, fixed-budget otherwise, with the
/// interleaved-thinking beta on non-adaptive Claude and the `display` field
/// dropped for `GovCloud` targets.
///
/// `level` is the clamped pi reasoning level; `display` carries the thinking
/// display option (default `summarized`).
#[must_use]
pub fn build_additional_model_request_fields(
    model: &Model,
    reasoning: Option<ThinkingLevel>,
    thinking_budgets: Option<&crate::types::ThinkingBudgets>,
    interleaved_thinking: Option<bool>,
    thinking_display: Option<&str>,
    region: Option<&str>,
) -> Option<Value> {
    let reasoning = reasoning?;
    if !model.reasoning || !is_anthropic_claude_model(model) {
        return None;
    }
    // GovCloud Bedrock currently rejects the Claude thinking.display field;
    // omit it there until the GovCloud Converse schema catches up. Elsewhere
    // the pi default keeps the older-Claude behavior, upstream's
    // `thinkingDisplay ?? "summarized"`.
    let display = if is_govcloud_bedrock_target(model, region) {
        None
    } else {
        Some(thinking_display.unwrap_or("summarized"))
    };

    let mut additional = Map::new();
    if supports_adaptive_thinking(model) {
        let mut thinking = Map::new();
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
        ThinkingLevel::Xhigh | ThinkingLevel::Max => ThinkingLevel::High,
        other => other,
    };
    let budget = thinking_budget_for_level(level, thinking_budgets, reasoning);
    let mut thinking = Map::new();
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
    level: ThinkingLevel,
    thinking_budgets: Option<&crate::types::ThinkingBudgets>,
    requested: ThinkingLevel,
) -> u64 {
    let custom = thinking_budgets.and_then(|budgets| match level {
        ThinkingLevel::Minimal => budgets.minimal,
        ThinkingLevel::Low => budgets.low,
        ThinkingLevel::Medium => budgets.medium,
        ThinkingLevel::High => budgets.high,
        ThinkingLevel::Xhigh | ThinkingLevel::Max => None,
    });
    if let Some(budget) = custom {
        return budget;
    }
    // The extended levels clamp to the high budget, upstream's default table.
    match requested {
        ThinkingLevel::Minimal => 1024,
        _ => match level {
            ThinkingLevel::Minimal => 1024,
            ThinkingLevel::Low => 2048,
            ThinkingLevel::Medium => 8192,
            ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => 16384,
        },
    }
}

/// Convert internal messages to Bedrock's Message shape, upstream's
/// `convertMessages`.
///
/// Consecutive tool results merge into one user message, assistant blocks
/// degrade per the signature rules, and empty content drops with the
/// `<empty>` placeholder where Bedrock requires content.
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
    let caching = cache_retention != crate::types::CacheRetention::None
        && supports_prompt_caching(model, env);
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
                let content = convert_assistant_content(assistant, model);
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
        && !content
            .iter()
            .any(|block| block.get("cachePoint").is_some())
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
fn convert_assistant_content(assistant: &AssistantMessage, model: &Model) -> Vec<Value> {
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
    content
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
    let caching = cache_retention != crate::types::CacheRetention::None
        && supports_prompt_caching(model, env);
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

/// Format a Bedrock failure with a human-readable prefix, upstream's
/// `formatBedrockError`.
///
/// The raw HTTP body (with status) surfaces when the message does not
/// already carry it — what stops a gateway 403 from collapsing to
/// `Unknown: UnknownError` — a data-retention-mode failure points at the AWS
/// docs, and a recognized exception name rides as the stable prefix the
/// retry and overflow classifiers match on.
#[must_use]
pub fn format_bedrock_error(
    error_name: Option<&str>,
    message: &str,
    status: Option<u16>,
    body: Option<&str>,
) -> String {
    let carries_body = body.is_none_or(|body| message.contains(body));
    let core = match (status, body) {
        (Some(status), Some(body)) if !carries_body => format!("{status}: {body}"),
        _ => message.to_owned(),
    };
    let data_retention_hint = if core.to_lowercase().contains("data retention mode") {
        format!(" See {BEDROCK_DATA_RETENTION_DOCS_URL} for supported data retention modes.")
    } else {
        String::new()
    };
    error_name.and_then(bedrock_error_prefix).map_or_else(
        || format!("{core}{data_retention_hint}"),
        |prefix| format!("{prefix}: {core}{data_retention_hint}"),
    )
}

/// The trimmed header-derived diagnostic value within the length bound,
/// upstream's `normalizeDiagnosticValue`; over-long values drop rather than
/// truncate, because a truncated request id is not a request id.
fn normalize_diagnostic_value(value: Option<&str>) -> Option<String> {
    let value = value?;
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.chars().count() > MAX_BEDROCK_DIAGNOSTIC_VALUE_CHARS {
        return None;
    }
    Some(trimmed.to_owned())
}

/// Structured metadata alongside `errorMessage`, upstream's
/// `appendBedrockFailureDiagnostic`: `errorMessage` stays byte-identical
/// because the retry classifier matches against it, so the status, the
/// modeled error code, and the request id ride the diagnostic details.
/// Unknown fields are omitted, never guessed.
fn append_bedrock_failure_diagnostic(
    output: &mut AssistantMessage,
    error_name: Option<&str>,
    status: Option<u16>,
    request_id: Option<&str>,
    fallback_request_id: Option<&str>,
) {
    let mut details: BTreeMap<String, Value> = BTreeMap::new();

    if let Some(status) = status {
        details.insert("status".to_owned(), json!(status));
    }

    // Modeled Bedrock errors all end in `Exception`, unlike transport names
    // such as `TimeoutError` and the SDK's `Unknown` placeholder.
    if let Some(error_code) = error_name
        .filter(|name| name.ends_with("Exception"))
        .and_then(|name| normalize_diagnostic_value(Some(name)))
    {
        details.insert("errorCode".to_owned(), json!(error_code));
    }

    if let Some(request_id) = normalize_diagnostic_value(request_id)
        .or_else(|| normalize_diagnostic_value(fallback_request_id))
    {
        details.insert("requestId".to_owned(), json!(request_id));
    }

    if details.is_empty() {
        return;
    }

    crate::utils::diagnostics::append_assistant_message_diagnostic(
        output,
        AssistantMessageDiagnostic {
            kind: "bedrock_response_failure".to_owned(),
            timestamp: crate::auth::resolve::now_ms(),
            error: None,
            details: Some(details),
        },
    );
}

/// The boxed stream of wire events the runtime seam opens, upstream's
/// `response.stream`: one JSON value per Converse Stream frame, or a
/// mid-stream failure.
pub type BedrockEventStream =
    Pin<Box<dyn futures_core::Stream<Item = Result<Value, BedrockStreamFailure>> + Send>>;

/// The send-time response the runtime seam opens, upstream's
/// `client.send()` result reduced to the metadata fields the adapter reads.
pub struct BedrockStreamReply {
    /// The response's request id, upstream's `$metadata.requestId`.
    pub request_id: Option<String>,
    /// The HTTP status of the initial response, upstream's
    /// `$metadata.httpStatusCode`.
    pub status: Option<u16>,
    /// The event stream to consume.
    pub events: BedrockEventStream,
}

impl std::fmt::Debug for BedrockStreamReply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockStreamReply")
            .field("request_id", &self.request_id)
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

/// The failure a Converse Stream send or frame reports, upstream's SDK, upstream's SDK
/// exception reduced to the fields `formatBedrockError` and the failure
/// diagnostic read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BedrockStreamFailure {
    /// The exception name, upstream's `error.name`: modeled Bedrock errors
    /// end in `Exception`, transport names such as `TimeoutError` do not.
    pub name: Option<String>,
    /// The failure message, upstream's `error.message`.
    pub message: String,
    /// The HTTP status, upstream's `$metadata.httpStatusCode`.
    pub status: Option<u16>,
    /// The raw HTTP body, upstream's `$response.body`.
    pub body: Option<String>,
    /// The request id, upstream's `$metadata.requestId`.
    pub request_id: Option<String>,
}

impl BedrockStreamFailure {
    /// A failure carrying only its message, upstream's plain `Error` throws
    /// and the bare exception literals the unmarshaller can produce.
    #[must_use]
    pub const fn plain(message: String) -> Self {
        Self {
            name: None,
            message,
            status: None,
            body: None,
            request_id: None,
        }
    }
}

/// The runtime seam: one Converse Stream send, upstream's
/// `BedrockRuntimeClient.send(ConverseStreamCommand)` plus the event stream
/// it opens. The SDK adapter implements it; the tests mock it.
pub trait BedrockRuntime: std::fmt::Debug + Send + Sync {
    /// Send the command input and open the wire event stream.
    ///
    /// # Errors
    /// When the send fails before the event stream opens (credentials,
    /// endpoint, model rejection): the returned failure carries the HTTP
    /// status, body, and request id the response had. Mid-stream failures
    /// ride the returned event stream as `Err` items instead.
    fn converse_stream(
        &self,
        input: Value,
        config: BedrockClientConfig,
        signal: CancellationToken,
    ) -> BoxedFuture<'static, Result<BedrockStreamReply, BedrockStreamFailure>>;
}

/// The Bedrock Converse Stream streams, upstream's `bedrockConverseStreamApi()`.
#[derive(Debug, Default)]
pub struct BedrockStreams;

crate::api::wire_common::forward_provider_streams!(
    BedrockStreams,
    bedrock_options::BedrockStreamOptions
);

/// Stream an assistant response, upstream's `stream` export: the config
/// resolution, the command input, and the event adaptation run inside the
/// spawned task.
///
/// Every failure settles as a terminal error event, upstream's catch path.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&bedrock_options::BedrockStreamOptions>,
) -> AssistantMessageEventStream {
    let events = assistant_message_event_stream();
    let forward = events.clone();
    let model = model.clone();
    let context = context.clone();
    let options = options.cloned();
    tokio::spawn(async move {
        let options = options.unwrap_or_default();
        let runtime: Arc<dyn BedrockRuntime> = options
            .runtime
            .clone()
            .unwrap_or_else(|| Arc::new(crate::api::bedrock_sdk::SdkBedrockRuntime::new()));
        let mut output = initial_output(&model);
        let mut scratch = StreamScratch::default();
        match run_stream(
            &model,
            &context,
            &options,
            runtime.as_ref(),
            &mut output,
            &mut scratch,
            &forward,
        )
        .await
        {
            Ok(()) => {
                // The done event already settled the final result.
                forward.end(None);
            }
            Err(failure) => {
                // A stream can settle without stopping every block, so the
                // error path finalizes too, upstream's catch head.
                finalize_blocks(&mut output, &mut scratch.blocks);
                let aborted = options.transport_options.signal().is_cancelled();
                output.stop_reason = if aborted {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                output.error_message = Some(format_bedrock_error(
                    failure.name.as_deref(),
                    &failure.message,
                    failure.status,
                    failure.body.as_deref(),
                ));
                if output.stop_reason == StopReason::Error {
                    append_bedrock_failure_diagnostic(
                        &mut output,
                        failure.name.as_deref(),
                        failure.status,
                        failure.request_id.as_deref(),
                        scratch.response_request_id.as_deref(),
                    );
                }
                forward.push(AssistantMessageEvent::Error {
                    reason: output.stop_reason,
                    error: output.clone(),
                });
                forward.end(Some(&output));
            }
        }
    });
    events
}

/// Stream a simple assistant response, upstream's `streamSimple` export:
/// base options from the shared shaping, then the thinking level mapping.
///
/// The shared tool choice narrows to the two Bedrock choices, and the
/// thinking level maps per model family. Credential sources are ambient, so
/// no key is asserted here.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let mut base = bedrock_options::BedrockStreamOptions::from(build_base_options(
        model, context, options, None,
    ));
    base.tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(|choice| match choice {
            crate::types::ToolChoice::Auto => BedrockToolChoice::Auto,
            crate::types::ToolChoice::None => BedrockToolChoice::None,
        });
    let Some(reasoning) = options.and_then(|options| options.reasoning) else {
        return stream(model, context, Some(&base));
    };

    if is_anthropic_claude_model(model) {
        if supports_adaptive_thinking(model) {
            base.reasoning = Some(reasoning);
            base.thinking_budgets = options.and_then(|options| options.thinking_budgets);
            return stream(model, context, Some(&base));
        }

        // `None` means the caller did not request an output cap; the helper
        // then uses the model cap. Do not coerce to 0 here, or the thinking
        // budget would become the entire maxTokens value.
        let (max_tokens, thinking_budget) = adjust_max_tokens_for_thinking(
            base.max_tokens,
            model.max_tokens,
            reasoning,
            options.and_then(|options| options.thinking_budgets.as_ref()),
        );
        let max_tokens = clamp_max_tokens_to_context(model, context, max_tokens);
        base.max_tokens = Some(max_tokens);
        base.reasoning = Some(reasoning);
        let mut budgets = options
            .and_then(|options| options.thinking_budgets)
            .unwrap_or_default();
        let budget = thinking_budget.min(max_tokens.saturating_sub(MIN_ANSWER_TOKENS));
        match clamp_reasoning(Some(reasoning)).unwrap_or(ThinkingLevel::High) {
            ThinkingLevel::Minimal => budgets.minimal = Some(budget),
            ThinkingLevel::Low => budgets.low = Some(budget),
            ThinkingLevel::Medium => budgets.medium = Some(budget),
            ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => {
                budgets.high = Some(budget);
            }
        }
        base.thinking_budgets = Some(budgets);
        return stream(model, context, Some(&base));
    }

    base.reasoning = Some(reasoning);
    base.thinking_budgets = options.and_then(|options| options.thinking_budgets);
    stream(model, context, Some(&base))
}

/// Streaming scratch keyed by the wire's `contentBlockIndex`; the blocks on
/// the accumulator never carry it, upstream's `Block` index/partialJson/
/// redactedChunks fields.
#[derive(Default)]
struct BlockScratch {
    /// The live wire index -> accumulator position; the matching stop removes
    /// it, upstream's `block.index` delete.
    live: HashMap<u64, usize>,
    /// The accumulated raw tool-argument JSON per wire index, upstream's
    /// `block.partialJson`.
    partial_json: HashMap<u64, String>,
    /// The buffered redacted-reasoning bytes per wire index, upstream's
    /// `block.redactedChunks`.
    redacted_chunks: HashMap<u64, Vec<u8>>,
}

/// The cross-phase stream scratch: the block buffers plus the send-time
/// response request id, kept so the failure path can correlate a mid-stream
/// exception, upstream's `responseRequestId`.
#[derive(Default)]
struct StreamScratch {
    blocks: BlockScratch,
    response_request_id: Option<String>,
}

async fn run_stream(
    model: &Model,
    context: &Context,
    options: &bedrock_options::BedrockStreamOptions,
    runtime: &dyn BedrockRuntime,
    output: &mut AssistantMessage,
    scratch: &mut StreamScratch,
    events: &AssistantMessageEventStream,
) -> Result<(), BedrockStreamFailure> {
    let config = bedrock_options::resolve_client_config(model, options);
    let mut input = bedrock_options::build_command_input(
        model,
        context,
        options,
        bedrock_options::resolve_cache_retention(options),
    )
    .map_err(BedrockStreamFailure::plain)?;
    if let Some(hook) = &options.transport_options.on_payload {
        input = hook
            .call(input.clone(), model.clone())
            .await
            .unwrap_or(input);
    }

    let reply = runtime
        .converse_stream(input, config, options.transport_options.signal())
        .await;
    // Kept outside the error path so the catch can still correlate a
    // mid-stream failure: exceptions delivered as stream events carry no
    // HTTP metadata of their own.
    scratch.response_request_id = reply
        .as_ref()
        .ok()
        .and_then(|reply| normalize_diagnostic_value(reply.request_id.as_deref()));
    let reply = reply?;

    // The `$metadata` fallback for the response hook, upstream's
    // deserialize-step middleware collapsing into one fire after send.
    if let (Some(status), Some(hook)) =
        (reply.status, options.transport_options.on_response.as_ref())
    {
        let mut headers = BTreeMap::new();
        if let Some(request_id) = reply.request_id.clone() {
            headers.insert("x-amzn-requestid".to_owned(), request_id);
        }
        hook.call(
            crate::types::ProviderResponse { status, headers },
            model.clone(),
        )
        .await;
    }

    consume_chat_stream(model, reply, output, scratch, events).await?;
    finish_stream(options, output, scratch, events)?;
    Ok(())
}

/// The post-loop settlement, upstream's try-block tail: the aborted check
/// runs first, then the missing-stop and stop-reason failures, then every
/// block finalizes and the done event settles the stream.
fn finish_stream(
    options: &bedrock_options::BedrockStreamOptions,
    output: &mut AssistantMessage,
    scratch: &mut StreamScratch,
    events: &AssistantMessageEventStream,
) -> Result<(), BedrockStreamFailure> {
    if options.transport_options.signal().is_cancelled() {
        return Err(BedrockStreamFailure::plain(
            "Request was aborted".to_owned(),
        ));
    }
    if output.stop_reason == StopReason::Pending {
        return Err(BedrockStreamFailure::plain(
            "Bedrock stream ended without a stop reason".to_owned(),
        ));
    }
    if matches!(output.stop_reason, StopReason::Error | StopReason::Aborted) {
        return Err(BedrockStreamFailure::plain(
            output
                .error_message
                .clone()
                .unwrap_or_else(|| "An unknown error occurred".to_owned()),
        ));
    }
    finalize_blocks(output, &mut scratch.blocks);
    events.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output.clone(),
    });
    Ok(())
}

/// The event adaptation over the Converse Stream frames, upstream's
/// `for await (const item of response.stream)` loop: the message/block
/// handlers below, the raw stop reason on `messageStop`, the usage pricing
/// on `metadata`, and the five modeled exceptions thrown into the catch path.
async fn consume_chat_stream(
    model: &Model,
    reply: BedrockStreamReply,
    output: &mut AssistantMessage,
    scratch: &mut StreamScratch,
    events: &AssistantMessageEventStream,
) -> Result<(), BedrockStreamFailure> {
    let mut wire_events = reply.events;
    while let Some(item) = wire_events.next().await {
        let item = match item {
            Ok(item) => item,
            Err(failure) => return Err(failure),
        };
        if let Some(start) = item.get("messageStart") {
            if start.get("role").and_then(Value::as_str) != Some("assistant") {
                return Err(BedrockStreamFailure::plain(
                    "Unexpected assistant message start but got user message start instead"
                        .to_owned(),
                ));
            }
            events.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
        } else if let Some(event) = item.get("contentBlockStart") {
            handle_content_block_start(event, output, &mut scratch.blocks, events);
        } else if let Some(event) = item.get("contentBlockDelta") {
            handle_content_block_delta(event, output, &mut scratch.blocks, events);
        } else if let Some(event) = item.get("contentBlockStop") {
            handle_content_block_stop(event, output, &mut scratch.blocks, events);
        } else if let Some(stop) = item.get("messageStop") {
            output.raw_stop_reason = stop
                .get("stopReason")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let (stop_reason, error_message) = map_stop_reason(output.raw_stop_reason.as_deref());
            output.stop_reason = stop_reason;
            if let Some(error_message) = error_message {
                output.error_message = Some(error_message);
            }
        } else if let Some(metadata) = item.get("metadata") {
            handle_metadata(metadata, model, output);
        } else if let Some(exception) = modeled_stream_exception(&item) {
            return Err(failure_from_exception(&exception));
        }
    }
    Ok(())
}

/// The wire's content-block index a frame addresses, upstream's
/// `contentBlockIndex!` non-null assertion with a zero fallback.
fn wire_index(event: &Value) -> u64 {
    event
        .get("contentBlockIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// The wire's optional string field, upstream's `|| ""` defaulting.
fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// The token count the wire reports, upstream's `|| 0` defaulting.
fn wire_u64(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

/// Handle `contentBlockStart`: only a `toolUse` start opens a block, upstream's
/// `handleContentBlockStart` — text and reasoning blocks open on their first
/// delta instead.
fn handle_content_block_start(
    event: &Value,
    output: &mut AssistantMessage,
    scratch: &mut BlockScratch,
    events: &AssistantMessageEventStream,
) {
    let index = wire_index(event);
    let Some(tool_use) = event.get("start").and_then(|start| start.get("toolUse")) else {
        return;
    };
    output.content.push(AssistantBlock::ToolCall(ToolCall {
        id: string_field(tool_use, "toolUseId"),
        name: string_field(tool_use, "name"),
        arguments: Map::new(),
        thought_signature: None,
        namespace: None,
    }));
    let content_index = output.content.len() - 1;
    scratch.live.insert(index, content_index);
    scratch.partial_json.insert(index, String::new());
    events.push(AssistantMessageEvent::ToolcallStart {
        content_index: content_index as u64,
        partial: output.clone(),
    });
}

/// Handle `contentBlockDelta` for text, tool-use, and reasoning deltas,
/// upstream's `handleContentBlockDelta`.
fn handle_content_block_delta(
    event: &Value,
    output: &mut AssistantMessage,
    scratch: &mut BlockScratch,
    events: &AssistantMessageEventStream,
) {
    let index = wire_index(event);
    let delta = event.get("delta");
    let live = scratch.live.get(&index).copied();

    // If no text block exists yet, create one: `contentBlockStart` is not
    // sent for text blocks.
    if let Some(text) = delta
        .and_then(|delta| delta.get("text"))
        .and_then(Value::as_str)
    {
        let content_index = if let Some(position) = live {
            position
        } else {
            output.content.push(AssistantBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            }));
            let position = output.content.len() - 1;
            scratch.live.insert(index, position);
            events.push(AssistantMessageEvent::TextStart {
                content_index: position as u64,
                partial: output.clone(),
            });
            position
        };
        // A live non-text block at the wire index swallows the delta,
        // upstream's `block.type === "text"` guard.
        if let Some(AssistantBlock::Text(block)) = output.content.get_mut(content_index) {
            block.text.push_str(text);
            events.push(AssistantMessageEvent::TextDelta {
                content_index: content_index as u64,
                delta: text.to_owned(),
                partial: output.clone(),
            });
        }
    } else if let Some(tool_use) = delta.and_then(|delta| delta.get("toolUse")) {
        // A tool-use delta only applies to a live tool block, upstream's
        // `block?.type === "toolCall"` guard.
        if let Some(position) = live
            && let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(position)
        {
            let chunk = tool_use
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let partial = scratch.partial_json.entry(index).or_default();
            partial.push_str(chunk);
            block.arguments = parse_streaming_json_args(Some(partial.as_str()));
            events.push(AssistantMessageEvent::ToolcallDelta {
                content_index: position as u64,
                delta: chunk.to_owned(),
                partial: output.clone(),
            });
        }
    } else if let Some(reasoning) = delta.and_then(|delta| delta.get("reasoningContent")) {
        handle_reasoning_delta(reasoning, index, live, output, scratch, events);
    }
}

/// Handle a reasoning-content delta: the thinking text and signature ride
/// their own block, and encrypted `redactedContent` accumulates into the
/// scratch buffer, upstream's `handleContentBlockDelta` reasoning branch.
fn handle_reasoning_delta(
    reasoning: &Value,
    index: u64,
    live: Option<usize>,
    output: &mut AssistantMessage,
    scratch: &mut BlockScratch,
    events: &AssistantMessageEventStream,
) {
    let thinking_index = if let Some(position) = live {
        position
    } else {
        output
            .content
            .push(AssistantBlock::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(String::new()),
                redacted: None,
            }));
        let position = output.content.len() - 1;
        scratch.live.insert(index, position);
        events.push(AssistantMessageEvent::ThinkingStart {
            content_index: position as u64,
            partial: output.clone(),
        });
        position
    };
    let Some(AssistantBlock::Thinking(_)) = output.content.get(thinking_index) else {
        return;
    };
    let redacted = matches!(
        output.content.get(thinking_index),
        Some(AssistantBlock::Thinking(block)) if block.redacted == Some(true)
    );

    // Upstream's truthiness gates: empty text and empty signatures are no-ops.
    if let Some(text) = reasoning.get("text").and_then(Value::as_str)
        && !text.is_empty()
    {
        let Some(AssistantBlock::Thinking(block)) = output.content.get_mut(thinking_index) else {
            return;
        };
        block.thinking.push_str(text);
        events.push(AssistantMessageEvent::ThinkingDelta {
            content_index: thinking_index as u64,
            delta: text.to_owned(),
            partial: output.clone(),
        });
    }

    // `thinkingSignature` holds either an Anthropic signature or an opaque
    // redacted payload, never both: mixing them would corrupt whichever
    // arrived first.
    if !redacted
        && let Some(signature) = reasoning.get("signature").and_then(Value::as_str)
        && !signature.is_empty()
    {
        let Some(AssistantBlock::Thinking(block)) = output.content.get_mut(thinking_index) else {
            return;
        };
        let previous = block.thinking_signature.clone().unwrap_or_default();
        block.thinking_signature = Some(format!("{previous}{signature}"));
    }

    if let Some(bytes) = reasoning_bytes(reasoning.get("redactedContent"))
        && !bytes.is_empty()
    {
        // Encrypted reasoning from non-Anthropic models on Bedrock (e.g.
        // OpenAI GPT-5.6). The payload is opaque, so keep it verbatim in
        // `thinkingSignature` the way the Anthropic path stores redacted
        // thinking, and replay it on the next turn.
        if !redacted {
            let Some(AssistantBlock::Thinking(block)) = output.content.get_mut(thinking_index)
            else {
                return;
            };
            block.redacted = Some(true);
            block.thinking_signature = Some(String::new());
            block.thinking.push_str(REDACTED_THINKING_PLACEHOLDER);
            events.push(AssistantMessageEvent::ThinkingDelta {
                content_index: thinking_index as u64,
                delta: REDACTED_THINKING_PLACEHOLDER.to_owned(),
                partial: output.clone(),
            });
        }
        scratch
            .redacted_chunks
            .entry(index)
            .or_default()
            .extend_from_slice(&bytes);
    }
}

/// The wire's redacted-reasoning bytes: the SDK adapter encodes the blob as
/// base64, the JSON replay path as a byte array, upstream's `Uint8Array`.
fn reasoning_bytes(value: Option<&Value>) -> Option<Vec<u8>> {
    match value? {
        Value::String(base64) => base64_to_bytes(base64),
        Value::Array(items) => items
            .iter()
            .map(|item| item.as_u64().and_then(|byte| u8::try_from(byte).ok()))
            .collect::<Option<Vec<u8>>>(),
        _ => None,
    }
}

/// Handle `contentBlockStop`, upstream's `handleContentBlockStop`: the
/// terminal `*_end` events carry the authoritative block content, the tool
/// call re-parses its accumulated JSON, and redacted reasoning flushes.
fn handle_content_block_stop(
    event: &Value,
    output: &mut AssistantMessage,
    scratch: &mut BlockScratch,
    events: &AssistantMessageEventStream,
) {
    let index = wire_index(event);
    let Some(content_index) = scratch.live.remove(&index) else {
        return;
    };
    let partial_json = scratch.partial_json.remove(&index);
    match output.content.get_mut(content_index) {
        Some(AssistantBlock::Text(block)) => {
            events.push(AssistantMessageEvent::TextEnd {
                content_index: content_index as u64,
                content: block.text.clone(),
                partial: output.clone(),
            });
        }
        Some(AssistantBlock::Thinking(block)) => {
            if let Some(chunks) = scratch.redacted_chunks.remove(&index) {
                block.thinking_signature = Some(bytes_to_base64(&chunks));
            }
            events.push(AssistantMessageEvent::ThinkingEnd {
                content_index: content_index as u64,
                content: block.thinking.clone(),
                partial: output.clone(),
            });
        }
        Some(AssistantBlock::ToolCall(block)) => {
            // Finalize in-place and strip the scratch buffer so replay only
            // carries parsed arguments, upstream's `partialJson` delete.
            block.arguments = parse_streaming_json_args(partial_json.as_deref());
            events.push(AssistantMessageEvent::ToolcallEnd {
                content_index: content_index as u64,
                tool_call: block.clone(),
                partial: output.clone(),
            });
        }
        None => {}
    }
}

/// Strips every streaming scratch buffer. Runs from the terminal paths as
/// well as `contentBlockStop`, because a stream can settle without stopping
/// each block, upstream's `finalizeStreamingBlock`.
fn finalize_blocks(output: &mut AssistantMessage, scratch: &mut BlockScratch) {
    for (index, position) in scratch.live.drain() {
        if let Some(chunks) = scratch.redacted_chunks.remove(&index)
            && let Some(AssistantBlock::Thinking(block)) = output.content.get_mut(position)
        {
            // Encodes buffered encrypted reasoning into `thinkingSignature`
            // and drops the scratch buffer, which must never reach a
            // persisted message: raw bytes would serialize many times their
            // base64 size, upstream's `flushRedactedContent`.
            block.thinking_signature = Some(bytes_to_base64(&chunks));
        }
    }
    scratch.partial_json.clear();
    scratch.redacted_chunks.clear();
}

/// Handle the `metadata` frame's usage, upstream's `handleMetadata`: the
/// 1h cache details split out of the total cache write, the total falling
/// back to input + output, then the model pricing.
fn handle_metadata(event: &Value, model: &Model, output: &mut AssistantMessage) {
    let Some(usage) = event.get("usage") else {
        return;
    };
    output.usage.input = wire_u64(usage, "inputTokens");
    output.usage.output = wire_u64(usage, "outputTokens");
    output.usage.cache_read = wire_u64(usage, "cacheReadInputTokens");
    output.usage.cache_write = wire_u64(usage, "cacheWriteInputTokens");
    output.usage.cache_write_1h =
        usage
            .get("cacheDetails")
            .and_then(Value::as_array)
            .map(|details| {
                details
                    .iter()
                    .filter(|detail| detail.get("ttl").and_then(Value::as_str) == Some("1h"))
                    .map(|detail| wire_u64(detail, "inputTokens"))
                    .sum()
            });
    output.usage.total_tokens = match wire_u64(usage, "totalTokens") {
        0 => output.usage.input + output.usage.output,
        total => total,
    };
    crate::models::calculate_cost(model, &mut output.usage);
}

/// The five modeled stream exceptions, upstream's else-if throws.
const MODELED_STREAM_EXCEPTIONS: [&str; 5] = [
    "internalServerException",
    "modelStreamErrorException",
    "validationException",
    "throttlingException",
    "serviceUnavailableException",
];

/// The modeled exception frame the item carries, if any.
fn modeled_stream_exception(item: &Value) -> Option<Value> {
    MODELED_STREAM_EXCEPTIONS
        .iter()
        .find_map(|key| item.get(*key).cloned())
}

/// The mid-stream exception's failure shape: the typed members the frame
/// carries, upstream's thrown exception object.
fn failure_from_exception(exception: &Value) -> BedrockStreamFailure {
    BedrockStreamFailure {
        name: exception
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned),
        message: exception
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        status: None,
        body: None,
        request_id: None,
    }
}

/// The tool-call arguments re-parsed from the accumulated JSON, upstream's
/// `parseStreamingJson(block.partialJson)` shape.
fn parse_streaming_json_args(partial: Option<&str>) -> Map<String, Value> {
    parse_streaming_json(partial)
        .as_object()
        .cloned()
        .unwrap_or_default()
}
