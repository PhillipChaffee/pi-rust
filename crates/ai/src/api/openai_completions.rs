//! The OpenAI Chat Completions wire API, ported from
//! `packages/ai/src/api/openai-completions.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//!
//! - The `openai` SDK client collapses onto the [`crate::http::HttpClient`]
//!   seam. The module assembles the wire request the pinned SDK sent — POST
//!   `{baseUrl}/chat/completions`, `Authorization: Bearer {apiKey}`, the pi
//!   user agent, Copilot dynamic headers, and the session-affinity headers —
//!   and executes it through the injected or default client. Upstream's
//!   `options.client` injection has no separate shape: an injected
//!   [`crate::http::HttpClient`] replaces the transport the way `options.fetch`
//!   did, and request shaping always runs.
//! - The SDK's chat stream is the shared [`crate::http::sse`] decoder over the
//!   seam body: each `data:` frame is one `ChatCompletionChunk` JSON and the
//!   `data: [DONE]` sentinel ends the stream. The chunk fields the wire types
//!   loosely (`reasoning_content`, `reasoning`, `reasoning_text`,
//!   `reasoning_details`, `choice.usage`) are read straight off the JSON.
//! - `sanitizeSurrogates` disappears statically: a Rust [`String`] cannot hold
//!   the unpaired surrogates it stripped, and `serde_json` rejects
//!   lone-surrogate escapes when reading the wire.
//! - `streamSimple`'s synchronous auth throw becomes the setup-error stream:
//!   the Rust stream contract returns a stream synchronously, so a missing
//!   key surfaces as its `error` event.
//! - The streamed reasoning details serialize back into
//!   `thinkingSignature` byte-for-byte like `JSON.stringify` did: object key
//!   order is insertion order (`preserve_order`), so entries replay with the
//!   wire's original field layout.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use serde_json::{Map, Value, json};

use crate::api::constrained_sampling::{
    GrammarToolInputJsonBuffer, append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties, get_grammar_tool_input, get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling, resolve_json_schema_strict_sampling,
};
use crate::api::github_copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::simple_options::{
    build_base_options, clamp_thinking_budget_to_answer_room, thinking_budget_for_level,
};
use crate::api::transform_messages::{ToolCallIdNormalizer, transform_messages};
use crate::http::client::{HttpError, HttpMethod, HttpRequest, HttpResponse};
use crate::http::sse::{ServerSentEvent, SseStream};
use crate::models::{calculate_cost, clamp_thinking_level};
use crate::types::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, CacheControlFormat, CacheRetention,
    ChatTemplateKwargValue, Context, DeferredToolsMode, GrammarFormat, MaxTokensField, Message,
    Modality, Model, ModelThinkingLevel, OpenRouterRouting, ProviderEnv, ProviderHeaders,
    ProviderId, SessionAffinityFormat, SimpleStreamOptions, StopReason, StreamOptions, TextContent,
    ThinkingBudgets, ThinkingContent, ThinkingFormat, ThinkingLevel, ThinkingLevelMap,
    ThinkingTemplateVar, ThinkingTokenBudgetField, Tool, ToolCall, ToolChoice, ToolResultBlock,
    TransportOptions, Usage, UserBlock, UserContent, VercelGatewayRouting,
};
use crate::utils::error_body::{
    ErrorBody, SdkError, format_provider_error, normalize_provider_error,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::hash::short_hash;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

use crate::api::request_seam::{
    execute_checked_response, fire_response_hook, get_client_api_key, setup_error_stream,
    spawned_stream,
};

// ---------------------------------------------------------------------------
// Tool history / deferred tools
// ---------------------------------------------------------------------------

/// Whether the conversation carries tool calls or tool results, upstream's
/// `hasToolHistory`: proxies require the `tools` param when it does.
fn has_tool_history(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::ToolResult(_) => true,
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .any(|block| matches!(block, AssistantBlock::ToolCall(_))),
        Message::User(_) => false,
    })
}

/// The tool names tool results made available, upstream's
/// `getDeferredToolNames`.
fn get_deferred_tool_names(messages: &[Message]) -> BTreeSet<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(result) => result.added_tool_names.as_ref(),
            _ => None,
        })
        .flatten()
        .cloned()
        .collect()
}

/// The context tools named by `names`, upstream's `getToolsByName`.
fn get_tools_by_name(tools: Option<&[Tool]>, names: &BTreeSet<String>) -> Vec<Tool> {
    let Some(tools) = tools else {
        return Vec::new();
    };
    names
        .iter()
        .filter_map(|name| tools.iter().find(|tool| &tool.name == name))
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// Reasoning details (replay metadata carried in thinking signatures)
// ---------------------------------------------------------------------------

/// Whether the common detail fields are well-formed, upstream's
/// `hasValidCommonReasoningDetailFields`.
fn has_valid_common_reasoning_detail_fields(candidate: &Map<String, Value>) -> bool {
    let id_ok = candidate
        .get("id")
        .is_none_or(|id| id.is_null() || id.as_str().is_some());
    let format_ok = candidate
        .get("format")
        .is_none_or(|format| format.as_str().is_some());
    let index_ok = candidate.get("index").is_none_or(Value::is_number);
    id_ok && format_ok && index_ok
}

/// Validate one streamed or replayed reasoning-detail entry, upstream's
/// `isOpenAIReasoningDetail`. Unknown fields ride along unread; the wire's
/// `type` decides which body fields must hold strings.
fn is_openai_reasoning_detail(detail: &Value) -> bool {
    let Some(map) = detail.as_object() else {
        return false;
    };
    if !has_valid_common_reasoning_detail_fields(map) {
        return false;
    }
    match map.get("type").and_then(Value::as_str) {
        Some("reasoning.summary") => map.get("summary").is_some_and(Value::is_string),
        Some("reasoning.encrypted") => map.get("data").is_some_and(Value::is_string),
        Some("reasoning.text") => {
            map.get("text").is_some_and(Value::is_string)
                && map
                    .get("signature")
                    .is_none_or(|signature| signature.is_null() || signature.as_str().is_some())
        }
        _ => false,
    }
}

/// Parse the serialized reasoning details a thinking signature carries,
/// upstream's `parseOpenAIReasoningDetails`: a non-empty array of valid
/// entries, `None` otherwise.
fn parse_openai_reasoning_details(signature: Option<&str>) -> Option<Vec<Value>> {
    let signature = signature?;
    let parsed = serde_json::from_str::<Value>(signature).ok()?;
    let entries = parsed.as_array()?;
    if entries.is_empty() || !entries.iter().all(is_openai_reasoning_detail) {
        return None;
    }
    Some(entries.clone())
}

/// The legacy encrypted reasoning detail a tool call's `thoughtSignature`
/// carries, upstream's `parseLegacyEncryptedReasoningDetail`: a single
/// `reasoning.encrypted` entry with non-empty `id` and `data`.
fn parse_legacy_encrypted_reasoning_detail(signature: Option<&str>) -> Option<Value> {
    let signature = signature?;
    let parsed = serde_json::from_str::<Value>(signature).ok()?;
    if !is_openai_reasoning_detail(&parsed) {
        return None;
    }
    let map = parsed.as_object()?;
    if map.get("type").and_then(Value::as_str) != Some("reasoning.encrypted") {
        return None;
    }
    let id = map.get("id").and_then(Value::as_str)?;
    let data = map.get("data").and_then(Value::as_str)?;
    if id.is_empty() || data.is_empty() {
        return None;
    }
    Some(parsed)
}

/// The `??=`/`||=` common-field fill, upstream's
/// `fillMissingCommonReasoningDetailFields`: absent keys append at the end of
/// the wire object, existing keys keep their position.
fn fill_missing_common_reasoning_detail_fields(target: &mut Map<String, Value>, source: &Value) {
    let Some(source) = source.as_object() else {
        return;
    };
    if !target.contains_key("id")
        && let Some(id) = source.get("id").filter(|id| !id.is_null())
    {
        target.insert("id".to_owned(), id.clone());
    }
    let format_present = target
        .get("format")
        .is_some_and(|format| format.as_str().is_some_and(|format| !format.is_empty()));
    if !format_present
        && let Some(format) = source.get("format").filter(|format| format.is_string())
    {
        target.insert("format".to_owned(), format.clone());
    }
    if !target.contains_key("index")
        && let Some(index) = source.get("index").filter(|index| index.is_number())
    {
        target.insert("index".to_owned(), index.clone());
    }
}

/// Merge one streamed detail into the accumulator, upstream's
/// `appendOpenAIReasoningDetail`: consecutive text and summary entries merge
/// in place (wire key order preserved, new fields append), anything else
/// appends a copy.
fn append_openai_reasoning_detail(details: &mut Vec<Value>, detail: Value) {
    let Some(last) = details.last_mut() else {
        details.push(detail);
        return;
    };
    let kind = detail
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let last_kind = last
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();

    let (field, grows) = match (kind.as_str(), last_kind.as_str()) {
        ("reasoning.text", "reasoning.text") => ("text", true),
        ("reasoning.summary", "reasoning.summary") => ("summary", true),
        _ => ("", false),
    };
    if !grows {
        details.push(detail);
        return;
    }
    let payload = detail
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();

    let Some(entry) = last.as_object_mut() else {
        return;
    };
    entry
        .entry(field.to_owned())
        .and_modify(|existing| {
            if let Value::String(existing) = existing {
                existing.push_str(&payload);
            } else {
                *existing = Value::String(payload.clone());
            }
        })
        .or_insert_with(|| Value::String(payload));
    if kind == "reasoning.text" {
        let carried = entry
            .get("signature")
            .is_some_and(|signature| !signature.is_null());
        if !carried && let Some(signature) = detail.get("signature") {
            entry.insert("signature".to_owned(), signature.clone());
        }
    }
    fill_missing_common_reasoning_detail_fields(entry, &detail);
}

/// Whether a signature names one of the reasoning fields the adapter replays,
/// upstream's `isOpenAICompletionsReasoningField`.
fn is_openai_completions_reasoning_field(field: &str) -> bool {
    matches!(field, "reasoning" | "reasoning_content" | "reasoning_text")
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// An OpenAI tool choice, upstream's `ChatCompletionToolChoiceOption`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenAiToolChoice {
    /// `"none"`: never call tools.
    None,
    /// `"auto"`: let the model decide.
    Auto,
    /// `"required"`: the model must call a tool.
    Required,
    /// Force a named function tool, the wire's
    /// `{"type":"function","function":{"name":...}}`.
    Function {
        /// The tool to force.
        name: String,
    },
}

impl OpenAiToolChoice {
    /// The request's `tool_choice` value.
    #[must_use]
    pub fn to_wire(&self) -> Value {
        match self {
            Self::None => json!("none"),
            Self::Auto => json!("auto"),
            Self::Required => json!("required"),
            Self::Function { name } => json!({"type": "function", "function": {"name": name}}),
        }
    }
}

/// The OpenAI Completions options, upstream's `OpenAICompletionsOptions
/// extends StreamOptions`: the base fields this adapter reads plus the
/// tool-choice, effort, and budget extras.
///
/// Base fields the adapter never reads (`transport`,
/// `websocketConnectTimeoutMs`) are omitted, matching the sibling wire APIs'
/// option structs.
#[derive(Clone, Debug, Default)]
pub struct OpenAiCompletionsOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks.
    pub transport_options: TransportOptions,
    /// The API key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this request.
    pub telemetry_context: Option<pi_telemetry::TelemetryHandle>,
    /// Provider-scoped environment values.
    pub env: Option<ProviderEnv>,
    /// Custom HTTP headers merged over the client defaults; `None` values
    /// suppress a default header.
    pub headers: Option<ProviderHeaders>,
    /// HTTP request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts. Default: 0.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait for a server-requested retry.
    pub max_retry_delay_ms: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Arbitrary sampling parameters merged into the request body as-is,
    /// after the named request fields, so keys here override them.
    pub sampling_params: Option<BTreeMap<String, Value>>,
    /// Maximum output tokens.
    pub max_tokens: Option<u64>,
    /// Prompt cache retention preference. Default: `"short"`.
    pub cache_retention: Option<CacheRetention>,
    /// Session identifier for session-based caching and routing.
    pub session_id: Option<String>,
    /// Optional request metadata; the adapter does not read it.
    pub metadata: Option<BTreeMap<String, Value>>,
    /// The OpenAI tool choice; omitted by default.
    pub tool_choice: Option<OpenAiToolChoice>,
    /// The reasoning effort, upstream's `reasoningEffort`.
    pub reasoning_effort: Option<ThinkingLevel>,
    /// Token budgets per thinking level, spent when the compat field or the
    /// `thinking.budget` variable requests one.
    pub thinking_budgets: Option<ThinkingBudgets>,
}

crate::api::adapter_belt::impl_stream_options_from!(OpenAiCompletionsOptions from options {
    sampling_params: options.sampling_params,
    tool_choice: None,
    reasoning_effort: None,
    thinking_budgets: None,
});
// ---------------------------------------------------------------------------
// Compat resolution
// ---------------------------------------------------------------------------

/// The resolved compat flags the adapter acts on, upstream's
/// `ResolvedOpenAICompletionsCompat`: every auto-detectable flag carries its
/// detected default; the format/mode fields stay optional because absence is
/// meaningful.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the flag belt mirrors upstream's compat object; each flag is an independent provider admission"
)]
#[derive(Clone, Debug)]
struct CompletionsCompat {
    supports_store: bool,
    supports_developer_role: bool,
    supports_reasoning_effort: bool,
    supports_usage_in_streaming: bool,
    supports_finish_reason: bool,
    max_tokens_field: MaxTokensField,
    requires_tool_result_name: bool,
    requires_assistant_after_tool_result: bool,
    requires_thinking_as_text: bool,
    requires_reasoning_content_on_assistant_messages: bool,
    thinking_format: ThinkingFormat,
    open_router_routing: Option<OpenRouterRouting>,
    vercel_gateway_routing: Option<VercelGatewayRouting>,
    chat_template_kwargs: Option<BTreeMap<String, ChatTemplateKwargValue>>,
    chat_template_args: Option<BTreeMap<String, ChatTemplateKwargValue>>,
    zai_tool_stream: bool,
    supports_thinking_token_budget: Option<bool>,
    thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    supports_strict_mode: bool,
    supports_openai_grammar_tools: bool,
    cache_control_format: Option<CacheControlFormat>,
    send_session_affinity_headers: bool,
    deferred_tools_mode: Option<DeferredToolsMode>,
    session_affinity_format: SessionAffinityFormat,
    supports_long_cache_retention: bool,
    vllm_priority: Option<serde_json::Number>,
}

/// Unset caller flags fall back to the detected default, upstream's `??`.
fn field_or<T>(caller: Option<T>, detected: T) -> T {
    caller.unwrap_or(detected)
}

/// Auto-detect compatibility settings from provider name and baseUrl,
/// upstream's `detectCompat`. Explicit `model.compat` entries override these.
#[expect(
    clippy::too_many_lines,
    reason = "one flag per detected provider shape; the detection reads top to bottom like upstream"
)]
fn detect_compat(model: &Model) -> CompletionsCompat {
    let provider: &str = &model.provider;
    let base_url = &model.base_url;

    let is_zai = provider == "zai"
        || provider == "zai-coding-cn"
        || base_url.contains("api.z.ai")
        || base_url.contains("open.bigmodel.cn");
    let is_together = provider == "together"
        || base_url.contains("api.together.ai")
        || base_url.contains("api.together.xyz");
    let is_moonshot = provider == "moonshotai"
        || provider == "moonshotai-cn"
        || base_url.contains("api.moonshot.");
    let is_openrouter = provider == "openrouter" || base_url.contains("openrouter.ai");
    let is_cloudflare_workers_ai =
        provider == "cloudflare-workers-ai" || base_url.contains("api.cloudflare.com");
    let is_cloudflare_ai_gateway =
        provider == "cloudflare-ai-gateway" || base_url.contains("gateway.ai.cloudflare.com");
    let is_nvidia = provider == "nvidia" || base_url.contains("integrate.api.nvidia.com");
    let is_ant_ling = provider == "ant-ling" || base_url.contains("api.ant-ling.com");
    let is_deepseek = provider == "deepseek" || base_url.to_lowercase().contains("deepseek.com");

    let is_non_standard = is_nvidia
        || provider == "cerebras"
        || base_url.contains("cerebras.ai")
        || provider == "xai"
        || base_url.contains("api.x.ai")
        || is_together
        || base_url.contains("chutes.ai")
        || is_deepseek
        || is_zai
        || is_moonshot
        || provider == "opencode"
        || base_url.contains("opencode.ai")
        || is_cloudflare_workers_ai
        || is_cloudflare_ai_gateway
        || is_ant_ling;

    let use_max_tokens = base_url.contains("chutes.ai")
        || is_deepseek
        || is_moonshot
        || is_cloudflare_ai_gateway
        || is_together
        || is_nvidia
        || is_ant_ling
        || is_zai;

    let is_grok = provider == "xai" || base_url.contains("api.x.ai");
    let is_openrouter_developer_role_model =
        is_openrouter && (model.id.starts_with("anthropic/") || model.id.starts_with("openai/"));
    let cache_control_format = if provider == "openrouter" && model.id.starts_with("anthropic/") {
        Some(CacheControlFormat::Anthropic)
    } else {
        None
    };

    CompletionsCompat {
        supports_store: !is_non_standard,
        supports_developer_role: is_openrouter_developer_role_model
            || (!is_non_standard && !is_openrouter),
        supports_reasoning_effort: !is_grok
            && !is_zai
            && !is_moonshot
            && !is_together
            && !is_cloudflare_ai_gateway
            && !is_nvidia
            && !is_ant_ling,
        supports_usage_in_streaming: true,
        supports_finish_reason: true,
        max_tokens_field: if use_max_tokens {
            MaxTokensField::MaxTokens
        } else {
            MaxTokensField::MaxCompletionTokens
        },
        requires_tool_result_name: false,
        requires_assistant_after_tool_result: false,
        requires_thinking_as_text: false,
        requires_reasoning_content_on_assistant_messages: is_deepseek,
        thinking_format: if is_deepseek {
            ThinkingFormat::Deepseek
        } else if is_zai {
            ThinkingFormat::Zai
        } else if is_together {
            ThinkingFormat::Together
        } else if is_ant_ling {
            ThinkingFormat::AntLing
        } else if is_openrouter {
            ThinkingFormat::Openrouter
        } else {
            ThinkingFormat::Openai
        },
        open_router_routing: Some(OpenRouterRouting::default()),
        vercel_gateway_routing: Some(VercelGatewayRouting::default()),
        chat_template_kwargs: Some(BTreeMap::new()),
        chat_template_args: Some(BTreeMap::new()),
        zai_tool_stream: false,
        supports_thinking_token_budget: Some(false),
        thinking_token_budget_field: None,
        supports_strict_mode: !is_moonshot
            && !is_together
            && !is_cloudflare_ai_gateway
            && !is_nvidia,
        supports_openai_grammar_tools: false,
        cache_control_format,
        send_session_affinity_headers: is_openrouter,
        deferred_tools_mode: None,
        session_affinity_format: if is_openrouter {
            SessionAffinityFormat::Openrouter
        } else {
            SessionAffinityFormat::Openai
        },
        supports_long_cache_retention: !(is_together
            || is_cloudflare_workers_ai
            || is_cloudflare_ai_gateway
            || is_nvidia
            || is_ant_ling),
        vllm_priority: None,
    }
}

/// Resolve the compat settings for a model: detected defaults overridable
/// field by field with `model.compat`, upstream's `getCompat`.
fn get_compat(model: &Model) -> CompletionsCompat {
    let detected = detect_compat(model);
    let Some(compat) = model.compat.as_ref() else {
        return detected;
    };
    CompletionsCompat {
        supports_store: field_or(compat.supports_store, detected.supports_store),
        supports_developer_role: field_or(
            compat.supports_developer_role,
            detected.supports_developer_role,
        ),
        supports_reasoning_effort: field_or(
            compat.supports_reasoning_effort,
            detected.supports_reasoning_effort,
        ),
        supports_usage_in_streaming: field_or(
            compat.supports_usage_in_streaming,
            detected.supports_usage_in_streaming,
        ),
        supports_finish_reason: field_or(
            compat.supports_finish_reason,
            detected.supports_finish_reason,
        ),
        max_tokens_field: compat.max_tokens_field.unwrap_or(detected.max_tokens_field),
        requires_tool_result_name: field_or(
            compat.requires_tool_result_name,
            detected.requires_tool_result_name,
        ),
        requires_assistant_after_tool_result: field_or(
            compat.requires_assistant_after_tool_result,
            detected.requires_assistant_after_tool_result,
        ),
        requires_thinking_as_text: field_or(
            compat.requires_thinking_as_text,
            detected.requires_thinking_as_text,
        ),
        requires_reasoning_content_on_assistant_messages: field_or(
            compat.requires_reasoning_content_on_assistant_messages,
            detected.requires_reasoning_content_on_assistant_messages,
        ),
        thinking_format: compat.thinking_format.unwrap_or(detected.thinking_format),
        open_router_routing: compat
            .open_router_routing
            .clone()
            .or(detected.open_router_routing),
        vercel_gateway_routing: compat
            .vercel_gateway_routing
            .clone()
            .or(detected.vercel_gateway_routing),
        chat_template_kwargs: compat
            .chat_template_kwargs
            .clone()
            .or(detected.chat_template_kwargs),
        chat_template_args: compat
            .chat_template_args
            .clone()
            .or(detected.chat_template_args),
        zai_tool_stream: field_or(compat.zai_tool_stream, detected.zai_tool_stream),
        supports_thinking_token_budget: compat
            .supports_thinking_token_budget
            .or(detected.supports_thinking_token_budget),
        thinking_token_budget_field: compat
            .thinking_token_budget_field
            .or(detected.thinking_token_budget_field),
        supports_strict_mode: field_or(compat.supports_strict_mode, detected.supports_strict_mode),
        supports_openai_grammar_tools: field_or(
            compat.supports_openai_grammar_tools,
            detected.supports_openai_grammar_tools,
        ),
        cache_control_format: compat
            .cache_control_format
            .or(detected.cache_control_format),
        send_session_affinity_headers: field_or(
            compat.send_session_affinity_headers,
            detected.send_session_affinity_headers,
        ),
        deferred_tools_mode: compat.deferred_tools_mode.or(detected.deferred_tools_mode),
        session_affinity_format: compat
            .session_affinity_format
            .unwrap_or(detected.session_affinity_format),
        supports_long_cache_retention: field_or(
            compat.supports_long_cache_retention,
            detected.supports_long_cache_retention,
        ),
        vllm_priority: compat.vllm_priority.clone().or(detected.vllm_priority),
    }
}

// ---------------------------------------------------------------------------
// Cache retention
// ---------------------------------------------------------------------------

/// Resolve the cache retention preference, upstream's `resolveCacheRetention`:
/// the request's value, else `PI_CACHE_RETENTION`, else `short`.
fn resolve_cache_retention(
    cache_retention: Option<CacheRetention>,
    env: Option<&ProviderEnv>,
) -> CacheRetention {
    if let Some(cache_retention) = cache_retention {
        return cache_retention;
    }
    if get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
        return CacheRetention::Long;
    }
    CacheRetention::Short
}

/// The `cache_control` marker an Anthropic-format provider applies, upstream's
/// `OpenAICompatCacheControl`: `{"type":"ephemeral"}` with `ttl: "1h"` when
/// long retention is supported.
fn get_compat_cache_control(
    compat: &CompletionsCompat,
    cache_retention: CacheRetention,
) -> Option<Map<String, Value>> {
    if compat.cache_control_format != Some(CacheControlFormat::Anthropic)
        || cache_retention == CacheRetention::None
    {
        return None;
    }
    let mut control = Map::new();
    control.insert("type".to_owned(), json!("ephemeral"));
    if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
        control.insert("ttl".to_owned(), json!("1h"));
    }
    Some(control)
}

/// Attach the cache-control marker to the system prompt, the last tool
/// definition, and the last user/assistant/tool message, upstream's
/// `applyAnthropicCacheControl`.
fn apply_anthropic_cache_control(
    messages: &mut [Value],
    tools: Option<&mut Vec<Map<String, Value>>>,
    cache_control: &Map<String, Value>,
) {
    add_cache_control_to_system_prompt(messages, cache_control);
    if let Some(tools) = tools {
        add_cache_control_to_last_tool(tools, cache_control);
    }
    add_cache_control_to_last_conversation_message(messages, cache_control);
}

fn add_cache_control_to_system_prompt(messages: &mut [Value], cache_control: &Map<String, Value>) {
    for message in messages.iter_mut() {
        if matches!(
            message.get("role").and_then(Value::as_str),
            Some("system" | "developer")
        ) {
            add_cache_control_to_text_content(message, cache_control);
            return;
        }
    }
}

fn add_cache_control_to_last_conversation_message(
    messages: &mut [Value],
    cache_control: &Map<String, Value>,
) {
    for message in messages.iter_mut().rev() {
        if matches!(
            message.get("role").and_then(Value::as_str),
            Some("user" | "assistant" | "tool")
        ) && add_cache_control_to_text_content(message, cache_control)
        {
            return;
        }
    }
}

fn add_cache_control_to_last_tool(
    tools: &mut [Map<String, Value>],
    cache_control: &Map<String, Value>,
) {
    let Some(last_tool) = tools.last_mut() else {
        return;
    };
    last_tool.insert(
        "cache_control".to_owned(),
        Value::Object(cache_control.clone()),
    );
}

/// Stamp the marker onto the message's text content: a string becomes a one
/// part array, a content array marks its last text part, upstream's
/// `addCacheControlToTextContent`.
fn add_cache_control_to_text_content(
    message: &mut Value,
    cache_control: &Map<String, Value>,
) -> bool {
    let content = message.get_mut("content");
    match content {
        Some(Value::String(text)) if !text.is_empty() => {
            message["content"] = json!([
                {
                    "type": "text",
                    "text": text.clone(),
                    "cache_control": cache_control,
                }
            ]);
            true
        }
        Some(Value::Array(parts)) => {
            for part in parts.iter_mut().rev() {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(part) = part.as_object_mut() {
                        part.insert(
                            "cache_control".to_owned(),
                            Value::Object(cache_control.clone()),
                        );
                    }
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

/// Assemble the request headers a stream sends, the port of upstream's
/// `createClient`: the pi user agent, the model's headers, Copilot's dynamic
/// headers, the session-affinity headers when the compat flag admits them,
/// and the caller's headers merged last so they can override defaults.
#[must_use]
fn build_request_headers(
    model: &Model,
    context: &Context,
    compat: &CompletionsCompat,
    session_id: Option<&str>,
    api_key: &str,
    options_headers: Option<&ProviderHeaders>,
) -> Vec<(String, String)> {
    // The credential header participates in the caller-header merge, the
    // pinned SDK's `defaultHeaders` behavior: an explicit value overrides
    // the bearer placeholder and a `null` removes it entirely, so a
    // gateway-binding sentinel can own the auth without the placeholder
    // riding the wire.
    let mut headers: Vec<(String, String)> = Vec::new();
    let caller_authorization = options_headers.and_then(|headers| {
        headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.clone())
    });
    match caller_authorization {
        Some(value) => {
            if let Some(value) = value {
                headers.push(("Authorization".to_owned(), value));
            }
        }
        None => headers.push(("Authorization".to_owned(), format!("Bearer {api_key}"))),
    }
    headers.push(("User-Agent".to_owned(), get_pi_user_agent()));
    // The pinned OpenAI SDK's JSON content type; explicit headers
    // override it and a `None` option suppresses it.
    headers.push(("content-type".to_owned(), "application/json".to_owned()));
    if let Some(model_headers) = &model.headers {
        for (name, value) in model_headers {
            headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
            headers.push((name.clone(), value.clone()));
        }
    }
    if model.provider == ProviderId::from("github-copilot") {
        let has_images = has_copilot_vision_input(&context.messages);
        let copilot = build_copilot_dynamic_headers(&context.messages, has_images);
        for (name, value) in copilot {
            headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(&name));
            headers.push((name, value));
        }
    }

    if let Some(session_id) = session_id
        && compat.send_session_affinity_headers
    {
        match compat.session_affinity_format {
            SessionAffinityFormat::Openrouter => {
                headers.push(("x-session-id".to_owned(), session_id.to_owned()));
            }
            format => {
                if format == SessionAffinityFormat::Openai {
                    headers.push(("session_id".to_owned(), session_id.to_owned()));
                }
                headers.push(("x-client-request-id".to_owned(), session_id.to_owned()));
                headers.push(("x-session-affinity".to_owned(), session_id.to_owned()));
            }
        }
    }

    if let Some(options_headers) = options_headers {
        for (name, value) in options_headers {
            match value {
                Some(value) => {
                    headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
                    headers.push((name.clone(), value.clone()));
                }
                None => headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name)),
            }
        }
    }
    headers
}

/// The chat-completions endpoint, the path the pinned SDK appends to
/// `baseUrl`.
#[must_use]
fn completions_url(model: &Model) -> String {
    format!("{}/chat/completions", model.base_url.trim_end_matches('/'))
}

/// The per-conversion options, upstream's `ConvertCompletionsMessagesOptions`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConvertCompletionsMessagesOptions<'a> {
    /// The grammar tool-input property per tool name, the map a replayed
    /// custom tool call's arguments read through.
    pub grammar_tool_input_properties: Option<&'a BTreeMap<String, String>>,
}

/// Build the request payload, upstream's `buildParams`. The wire's `undefined`
/// fields are absent, so only present fields insert.
#[expect(
    clippy::too_many_lines,
    reason = "the thinking-format branches are the compat matrix; splitting them would bury the per-provider wire shapes"
)]
fn build_params(
    model: &Model,
    context: &Context,
    options: &OpenAiCompletionsOptions,
    compat: &CompletionsCompat,
    cache_retention: CacheRetention,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Map<String, Value>, String> {
    let mut params = Map::new();
    params.insert("model".to_owned(), json!(model.id));
    let messages = convert_messages(
        model,
        context,
        compat,
        Some(&ConvertCompletionsMessagesOptions {
            grammar_tool_input_properties: Some(grammar_tool_input_properties),
        }),
    )?;
    params.insert("messages".to_owned(), Value::Array(messages));
    params.insert("stream".to_owned(), json!(true));

    let wants_prompt_cache_key = (model.base_url.contains("api.openai.com")
        && cache_retention != CacheRetention::None)
        || (cache_retention == CacheRetention::Long && compat.supports_long_cache_retention);
    if wants_prompt_cache_key
        && let Some(key) = clamp_openai_prompt_cache_key(options.session_id.as_deref())
    {
        params.insert("prompt_cache_key".to_owned(), json!(key));
    }
    if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
        params.insert("prompt_cache_retention".to_owned(), json!("24h"));
    }

    if compat.supports_usage_in_streaming {
        params.insert("stream_options".to_owned(), json!({"include_usage": true}));
    }

    if compat.supports_store {
        params.insert("store".to_owned(), json!(false));
    }

    if let Some(max_tokens) = options.max_tokens {
        if compat.max_tokens_field == MaxTokensField::MaxTokens {
            params.insert("max_tokens".to_owned(), json!(max_tokens));
        } else {
            params.insert("max_completion_tokens".to_owned(), json!(max_tokens));
        }
    }

    if let Some(temperature) = options.temperature {
        params.insert("temperature".to_owned(), json!(temperature));
    }

    let deferred_tool_names = if compat.deferred_tools_mode == Some(DeferredToolsMode::Kimi) {
        get_deferred_tool_names(&context.messages)
    } else {
        BTreeSet::new()
    };
    let active_tools: Vec<Tool> = context
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|tool| !deferred_tool_names.contains(&tool.name))
        .cloned()
        .collect();
    if !active_tools.is_empty() {
        params.insert(
            "tools".to_owned(),
            Value::Array(convert_tools(&active_tools, compat)?),
        );
        if compat.zai_tool_stream {
            params.insert("tool_stream".to_owned(), json!(true));
        }
    } else if has_tool_history(&context.messages) {
        // Anthropic (via LiteLLM/proxy) requires the tools param when the
        // conversation carries tool calls or tool results.
        params.insert("tools".to_owned(), json!([]));
    }

    if let Some(cache_control) = get_compat_cache_control(compat, cache_retention) {
        let mut messages = params
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut tools_param: Option<Vec<Map<String, Value>>> = params
            .get("tools")
            .and_then(Value::as_array)
            .map(|tools| tools.iter().filter_map(Value::as_object).cloned().collect());
        apply_anthropic_cache_control(&mut messages, tools_param.as_mut(), &cache_control);
        params.insert("messages".to_owned(), Value::Array(messages));
        if let Some(tools) = tools_param {
            params.insert(
                "tools".to_owned(),
                Value::Array(tools.into_iter().map(Value::Object).collect()),
            );
        }
    }

    if let Some(tool_choice) = &options.tool_choice {
        params.insert("tool_choice".to_owned(), tool_choice.to_wire());
    }

    if let Some(priority) = &compat.vllm_priority {
        params.insert("priority".to_owned(), Value::Number(priority.clone()));
    }

    let thinking_token_budget_field = resolve_thinking_token_budget_field(compat);
    let thinking_budget = resolve_clamped_thinking_budget(model, options, &params);

    apply_thinking_format(
        model,
        options,
        compat,
        &mut params,
        thinking_budget,
        thinking_token_budget_field,
    );

    // OpenRouter provider routing preferences.
    if let Some(routing) = model
        .compat
        .as_ref()
        .and_then(|compat| compat.open_router_routing.as_ref())
        && let Ok(wire) = serde_json::to_value(routing)
    {
        params.insert("provider".to_owned(), wire);
    }

    // Vercel AI Gateway provider routing preferences.
    if let Some(routing) = model
        .compat
        .as_ref()
        .and_then(|compat| compat.vercel_gateway_routing.as_ref())
        && (routing.only.is_some() || routing.order.is_some())
    {
        let mut gateway_options = Map::new();
        if let Some(only) = &routing.only {
            gateway_options.insert("only".to_owned(), json!(only));
        }
        if let Some(order) = &routing.order {
            gateway_options.insert("order".to_owned(), json!(order));
        }
        params.insert(
            "providerOptions".to_owned(),
            json!({"gateway": gateway_options}),
        );
    }

    // Last so custom keys override the named request fields.
    if let Some(sampling_params) = &options.sampling_params {
        for (key, value) in sampling_params {
            params.insert(key.clone(), value.clone());
        }
    }

    Ok(params)
}

/// The pi level a budget-based provider acts on, in the model-level spelling
/// the `thinkingLevelMap` keys use.
const fn thinking_model_level(level: ThinkingLevel) -> ModelThinkingLevel {
    match level {
        ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
        ThinkingLevel::Low => ModelThinkingLevel::Low,
        ThinkingLevel::Medium => ModelThinkingLevel::Medium,
        ThinkingLevel::High => ModelThinkingLevel::High,
        ThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
        ThinkingLevel::Max => ModelThinkingLevel::Max,
    }
}

impl ThinkingLevel {
    /// The level as the wire spells it.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// The top-level budget field the compat selects, upstream's
/// `resolveThinkingTokenBudgetField`.
fn resolve_thinking_token_budget_field(
    compat: &CompletionsCompat,
) -> Option<ThinkingTokenBudgetField> {
    if let Some(field) = compat.thinking_token_budget_field {
        return Some(field);
    }
    if compat.supports_thinking_token_budget == Some(true) {
        return Some(ThinkingTokenBudgetField::ThinkingTokenBudget);
    }
    None
}

/// The clamped thinking budget a request spends, upstream's
/// `resolveClampedThinkingBudget`.
fn resolve_clamped_thinking_budget(
    model: &Model,
    options: &OpenAiCompletionsOptions,
    params: &Map<String, Value>,
) -> Option<u64> {
    let effort = options.reasoning_effort?;
    if !model.reasoning {
        return None;
    }
    let ceiling = params
        .get("max_tokens")
        .and_then(Value::as_u64)
        .or_else(|| params.get("max_completion_tokens").and_then(Value::as_u64))
        .unwrap_or(model.max_tokens);
    let budget = clamp_thinking_budget_to_answer_room(
        thinking_budget_for_level(effort, options.thinking_budgets.as_ref()),
        ceiling,
    );
    (budget > 0).then_some(budget)
}

/// Resolve the `$var` template values into wire scalars, upstream's
/// `buildChatTemplateValues`.
fn build_chat_template_values(
    model: &Model,
    options: &OpenAiCompletionsOptions,
    values: Option<&BTreeMap<String, ChatTemplateKwargValue>>,
    thinking_budget: Option<u64>,
) -> Option<Map<String, Value>> {
    let mut resolved = Map::new();
    for (key, value) in values.into_iter().flatten() {
        if let Some(resolved_value) =
            resolve_chat_template_kwarg_value(model, options, value, thinking_budget)
        {
            resolved.insert(key.clone(), resolved_value);
        }
    }
    (!resolved.is_empty()).then_some(resolved)
}

/// One resolved chat-template kwarg value, upstream's
/// `resolveChatTemplateKwargValue`: the `thinking.effort` variable maps
/// through the level map — the mapped spelling when the map names one, the
/// requested level when the map or key is missing, and nothing on the `null`
/// unsupported-level marker; with no requested effort the map's `off` entry
/// reads the same way.
fn resolve_chat_template_kwarg_value(
    model: &Model,
    options: &OpenAiCompletionsOptions,
    value: &ChatTemplateKwargValue,
    thinking_budget: Option<u64>,
) -> Option<Value> {
    let ChatTemplateKwargValue::Template(var) = value else {
        return Some(match value {
            ChatTemplateKwargValue::Str(text) => json!(text),
            ChatTemplateKwargValue::Number(number) => Value::Number(number.clone()),
            ChatTemplateKwargValue::Bool(flag) => json!(flag),
            ChatTemplateKwargValue::Null => Value::Null,
            ChatTemplateKwargValue::Template(_) => unreachable!("handled above"),
        });
    };

    let reasoning_effort = options.reasoning_effort;
    if reasoning_effort.is_none() && var.omit_when_off == Some(true) {
        return None;
    }
    match var.var {
        ThinkingTemplateVar::ThinkingEnabled => return Some(json!(reasoning_effort.is_some())),
        ThinkingTemplateVar::ThinkingBudget => {
            return thinking_budget.map(|budget| json!(budget));
        }
        ThinkingTemplateVar::ThinkingEffort => {}
    }
    let level_map = model.thinking_level_map.as_ref();
    let mapped = reasoning_effort.map_or_else(
        || lookup_effort(level_map, ModelThinkingLevel::Off),
        |effort| lookup_effort(level_map, thinking_model_level(effort)),
    );
    match mapped {
        // `mappedValue === undefined ? reasoningEffort : ...`: a missing map
        // or key falls back to the requested level, which with no request
        // drops the kwarg.
        MappedEffort::Absent => reasoning_effort.map(|effort| json!(effort.as_str())),
        MappedEffort::Mapped(mapped) => Some(json!(mapped)),
        // The null unsupported-level marker drops the kwarg.
        MappedEffort::Null => None,
    }
}
/// Apply the compat's thinking format, upstream's `buildParams` thinking
/// branches: every format maps the pi effort through the model's
/// `thinkingLevelMap`, with the off state spelled per format.
#[expect(
    clippy::too_many_lines,
    reason = "one branch per thinking-format value; the matrix reads top to bottom like upstream"
)]
fn apply_thinking_format(
    model: &Model,
    options: &OpenAiCompletionsOptions,
    compat: &CompletionsCompat,
    params: &mut Map<String, Value>,
    thinking_budget: Option<u64>,
    thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
) {
    let reasoning_effort = options.reasoning_effort;
    let level_map = model.thinking_level_map.as_ref();
    let off_state = level_map.and_then(|map| map.get(&ModelThinkingLevel::Off));

    if compat.thinking_format == ThinkingFormat::Zai && model.reasoning {
        let thinking = if reasoning_effort.is_some() {
            json!({"type": "enabled", "clear_thinking": false})
        } else {
            json!({"type": "disabled"})
        };
        params.insert("thinking".to_owned(), thinking);
        // `mappedEffort === undefined ? effort : mappedEffort` with the
        // string gate: a missing map or key sends the level spelling, the
        // null unsupported-level marker omits the field.
        if let Some(effort) = reasoning_effort
            && compat.supports_reasoning_effort
            && let Some(effort) = effort_or_level(level_map, effort)
        {
            params.insert("reasoning_effort".to_owned(), effort);
        }
    } else if compat.thinking_format == ThinkingFormat::Qwen && model.reasoning {
        params.insert(
            "enable_thinking".to_owned(),
            json!(reasoning_effort.is_some()),
        );
        if let Some(effort) = reasoning_effort
            && compat.supports_reasoning_effort
        {
            params.insert(
                "reasoning_effort".to_owned(),
                mapped_or_level(level_map, effort),
            );
        }
    } else if compat.thinking_format == ThinkingFormat::QwenChatTemplate && model.reasoning {
        params.insert(
            "chat_template_kwargs".to_owned(),
            json!({"enable_thinking": reasoning_effort.is_some(), "preserve_thinking": true}),
        );
    } else if compat.thinking_format == ThinkingFormat::ChatTemplate && model.reasoning {
        let kwargs = build_chat_template_values(
            model,
            options,
            compat.chat_template_kwargs.as_ref(),
            thinking_budget,
        );
        if let Some(kwargs) = kwargs {
            params.insert("chat_template_kwargs".to_owned(), Value::Object(kwargs));
        }
    } else if compat.thinking_format == ThinkingFormat::Baseten && model.reasoning {
        let args = build_chat_template_values(
            model,
            options,
            compat.chat_template_args.as_ref(),
            thinking_budget,
        );
        if let Some(args) = args {
            params.insert("chat_template_args".to_owned(), Value::Object(args));
        }
        if compat.supports_reasoning_effort {
            // `mappedEffort = requestedEffort ? map?.[requestedEffort] :
            // map?.off` with the `=== undefined` fallback: a null entry omits
            // the field, a missing map sends the requested level, and with no
            // request only the map's string `off` spelling sends.
            let effort = reasoning_effort.map_or_else(
                || {
                    off_state
                        .cloned()
                        .flatten()
                        .map_or(Value::Null, |off| json!(off))
                },
                |effort| effort_or_level(level_map, effort).unwrap_or(Value::Null),
            );
            if effort.is_string() {
                params.insert("reasoning_effort".to_owned(), effort);
            }
        }
    } else if compat.thinking_format == ThinkingFormat::Deepseek && model.reasoning {
        if reasoning_effort.is_some() {
            params.insert("thinking".to_owned(), json!({"type": "enabled"}));
        } else if off_state.is_none_or(Option::is_some) {
            // `map?.off !== null`: the disabled object rides whenever the off
            // entry is missing or a string; only the null marker drops it.
            params.insert("thinking".to_owned(), json!({"type": "disabled"}));
        }
        if let Some(effort) = reasoning_effort
            && compat.supports_reasoning_effort
        {
            params.insert(
                "reasoning_effort".to_owned(),
                mapped_or_level(level_map, effort),
            );
        }
    } else if compat.thinking_format == ThinkingFormat::Openrouter && model.reasoning {
        // OpenRouter normalizes reasoning across providers via a nested
        // reasoning object.
        if let Some(effort) = reasoning_effort {
            params.insert(
                "reasoning".to_owned(),
                json!({ "effort": mapped_or_level(level_map, effort) }),
            );
        } else if off_state.is_none_or(Option::is_some) {
            // `map?.off !== null` gates the object and `map?.off ?? "none"`
            // spells it: a missing map or key sends "none", the null marker
            // omits.
            let off = off_state
                .cloned()
                .flatten()
                .unwrap_or_else(|| "none".to_owned());
            params.insert("reasoning".to_owned(), json!({ "effort": off }));
        }
    } else if compat.thinking_format == ThinkingFormat::AntLing
        && model.reasoning
        && reasoning_effort.is_some()
    {
        // `typeof effort === "string"` over the raw entry: null and absent
        // entries both omit `reasoning` entirely.
        if let Some(effort) = reasoning_effort
            && let MappedEffort::Mapped(mapped) =
                lookup_effort(level_map, thinking_model_level(effort))
        {
            params.insert("reasoning".to_owned(), json!({ "effort": mapped }));
        }
    } else if compat.thinking_format == ThinkingFormat::Together && model.reasoning {
        params.insert(
            "reasoning".to_owned(),
            json!({"enabled": reasoning_effort.is_some()}),
        );
        if let Some(effort) = reasoning_effort
            && compat.supports_reasoning_effort
        {
            params.insert(
                "reasoning_effort".to_owned(),
                mapped_or_level(level_map, effort),
            );
        }
    } else if compat.thinking_format == ThinkingFormat::StringThinking && model.reasoning {
        if let Some(effort) = reasoning_effort {
            params.insert("thinking".to_owned(), mapped_or_level(level_map, effort));
        } else if off_state.is_none_or(Option::is_some) {
            // `map?.off !== null` gates the field and `map?.off ?? "none"`
            // spells it: a missing map or key sends "none", the null marker
            // omits.
            let off = off_state
                .cloned()
                .flatten()
                .unwrap_or_else(|| "none".to_owned());
            params.insert("thinking".to_owned(), json!(off));
        }
    } else if let Some(effort) = reasoning_effort {
        if model.reasoning && compat.supports_reasoning_effort {
            // OpenAI-style reasoning_effort
            params.insert(
                "reasoning_effort".to_owned(),
                mapped_or_level(level_map, effort),
            );
        }
    } else if model.reasoning
        && compat.supports_reasoning_effort
        && let Some(off) = off_state.cloned().flatten()
    {
        params.insert("reasoning_effort".to_owned(), json!(off));
    }

    // Cap reasoning with a top-level budget field, independent of the
    // thinking format: reasoning and the answer share max_tokens here, so an
    // uncapped reasoning phase can consume the whole response and leave no
    // answer and no tool call.
    if let (Some(field), Some(budget)) = (thinking_token_budget_field, thinking_budget) {
        let wire = match field {
            ThinkingTokenBudgetField::ThinkingTokenBudget => "thinking_token_budget",
            ThinkingTokenBudgetField::ThinkingBudget => "thinking_budget",
            ThinkingTokenBudgetField::ThinkingBudgetTokens => "thinking_budget_tokens",
        };
        params.insert(wire.to_owned(), json!(budget));
    }
}

/// The model's `thinkingLevelMap` entry for a level, upstream's
/// `model.thinkingLevelMap?.[level]`: JS keeps three states apart — the map's
/// effort spelling, the `null` unsupported-level marker, and a missing map or
/// key (`undefined`).
enum MappedEffort {
    /// No map on the model, or no entry for the level.
    Absent,
    /// The `null` unsupported-level marker.
    Null,
    /// The provider's effort spelling for the level.
    Mapped(String),
}

/// Read the map entry, preserving the `undefined`/`null` distinction the
/// optional-chain read makes.
fn lookup_effort(level_map: Option<&ThinkingLevelMap>, level: ModelThinkingLevel) -> MappedEffort {
    match level_map.and_then(|map| map.get(&level)) {
        Some(Some(mapped)) => MappedEffort::Mapped(mapped.clone()),
        Some(None) => MappedEffort::Null,
        None => MappedEffort::Absent,
    }
}

/// The model's mapped effort for a pi level, upstream's
/// `model.thinkingLevelMap?.[effort] ?? effort`: the map's spelling when it
/// names one, else the level itself. A `null` entry — the wire's
/// unsupported-level marker — falls through to the level like `??` does.
fn mapped_or_level(level_map: Option<&ThinkingLevelMap>, level: ThinkingLevel) -> Value {
    match lookup_effort(level_map, thinking_model_level(level)) {
        MappedEffort::Mapped(mapped) => json!(mapped),
        MappedEffort::Null | MappedEffort::Absent => json!(level.as_str()),
    }
}

/// The `=== undefined` effort resolution the zai and baseten branches share,
/// upstream's `mappedEffort === undefined ? requestedEffort : mappedEffort`:
/// the mapped spelling when the map names one, the level when the map or key
/// is missing, and nothing on the null unsupported-level marker — `None` is
/// the wire's failed `typeof effort === "string"` gate.
fn effort_or_level(level_map: Option<&ThinkingLevelMap>, level: ThinkingLevel) -> Option<Value> {
    match lookup_effort(level_map, thinking_model_level(level)) {
        MappedEffort::Mapped(mapped) => Some(json!(mapped)),
        MappedEffort::Absent => Some(json!(level.as_str())),
        MappedEffort::Null => None,
    }
}

// ---------------------------------------------------------------------------
// Message and tool conversion
// ---------------------------------------------------------------------------

/// The `image_url` content part the wire carries base64 images in, the
/// `data:{mimeType};base64,{data}` URL form.
fn image_url_part(data: &str, mime_type: &str) -> Value {
    json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{mime_type};base64,{data}") },
    })
}

/// Normalize a tool-call id for the Chat Completions wire, upstream's
/// `normalizeToolCallId`. Pipe-separated Responses ids —
/// `{call_id}|{item_id}`, the item half up to 400+ characters with special
/// characters — collapse to `{call_id}_{item_id}`, sanitized to
/// `[A-Za-z0-9_-]`; past the wire's 40-character limit a `shortHash` suffix
/// of the full id keeps distinct item ids distinct. The `openai` provider
/// truncates other ids to 40.
fn normalize_openai_tool_call_id(id: &str, provider: &str) -> String {
    let sanitize = |value: &str| -> String {
        value
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                    character
                } else {
                    '_'
                }
            })
            .collect()
    };
    if let Some(separator_index) = id.find('|') {
        let call_id = sanitize(&id[..separator_index]);
        let item_id = sanitize(&id[separator_index + 1..]);
        let hash: String = short_hash(id).chars().take(8).collect();
        let keep = 39usize.saturating_sub(hash.chars().count()).max(1);
        let prefix: String = call_id.chars().take(keep).collect();
        let combined = if item_id.is_empty() {
            call_id
        } else {
            format!("{call_id}_{item_id}")
        };
        if combined.chars().count() <= 40 {
            return combined;
        }
        return format!("{prefix}_{hash}");
    }
    if provider == "openai" {
        return id.chars().take(40).collect();
    }
    id.to_owned()
}

/// The pi level a clamped model level names; `None` for the off state.
const fn pi_thinking_level(level: ModelThinkingLevel) -> Option<ThinkingLevel> {
    match level {
        ModelThinkingLevel::Off => None,
        ModelThinkingLevel::Minimal => Some(ThinkingLevel::Minimal),
        ModelThinkingLevel::Low => Some(ThinkingLevel::Low),
        ModelThinkingLevel::Medium => Some(ThinkingLevel::Medium),
        ModelThinkingLevel::High => Some(ThinkingLevel::High),
        ModelThinkingLevel::Xhigh => Some(ThinkingLevel::Xhigh),
        ModelThinkingLevel::Max => Some(ThinkingLevel::Max),
    }
}

/// Convert the conversation to the Chat Completions request messages,
/// upstream's `convertMessages`.
///
/// # Errors
/// The grammar and strict-schema rejections a converted tool definition
/// raises; a failure aborts the request before it is sent.
fn convert_messages(
    model: &Model,
    context: &Context,
    compat: &CompletionsCompat,
    options: Option<&ConvertCompletionsMessagesOptions<'_>>,
) -> Result<Vec<Value>, String> {
    let provider: &str = &model.provider;
    let normalize_tool_call_id: &ToolCallIdNormalizer<'_> =
        &|id, _model, _source| normalize_openai_tool_call_id(id, provider);
    let transformed_messages = transform_messages(
        context.messages.clone(),
        model,
        Some(normalize_tool_call_id),
    );

    let mut params: Vec<Value> = Vec::new();
    if let Some(system_prompt) = context
        .system_prompt
        .as_deref()
        .filter(|prompt| !prompt.is_empty())
    {
        let role = if model.reasoning && compat.supports_developer_role {
            "developer"
        } else {
            "system"
        };
        params.push(json!({ "role": role, "content": system_prompt }));
    }

    let mut last_role: Option<&str> = None;
    let mut index = 0;
    while index < transformed_messages.len() {
        let message = &transformed_messages[index];
        // Some providers do not allow user messages directly after tool
        // results; a synthetic assistant turn bridges the gap.
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(message, Message::User(_))
        {
            params.push(json!({
                "role": "assistant",
                "content": "I have processed the tool results.",
            }));
        }
        match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    params.push(json!({ "role": "user", "content": text }));
                    last_role = Some("user");
                }
                UserContent::Blocks(blocks) => {
                    let parts: Vec<Value> = blocks
                        .iter()
                        .map(|block| match block {
                            UserBlock::Text(text) => {
                                json!({ "type": "text", "text": text.text })
                            }
                            UserBlock::Image(image) => {
                                image_url_part(&image.data, &image.mime_type)
                            }
                        })
                        .collect();
                    if parts.is_empty() {
                        index += 1;
                        continue;
                    }
                    params.push(json!({ "role": "user", "content": parts }));
                    last_role = Some("user");
                }
            },
            Message::Assistant(assistant) => {
                let assistant_msg = convert_assistant_message(
                    model,
                    assistant,
                    compat,
                    options.and_then(|options| options.grammar_tool_input_properties),
                )?;
                let Some(assistant_msg) = assistant_msg else {
                    index += 1;
                    continue;
                };
                params.push(assistant_msg);
                last_role = Some("assistant");
            }
            Message::ToolResult(_) => {
                convert_tool_result_run(
                    model,
                    context,
                    compat,
                    transformed_messages.as_slice(),
                    &mut index,
                    &mut params,
                    &mut last_role,
                )?;
                continue;
            }
        }
        index += 1;
    }
    Ok(params)
}

/// The assistant message a history entry converts to, upstream's
/// `msg.role === "assistant"` branch. `None` when the message carries no
/// content and no tool calls — providers require "either content or
/// `tool_calls`, but not none", so aborted responses with neither are dropped.
///
/// # Errors
/// The grammar tool-input rejection when a custom tool call's arguments do
/// not carry the string input its grammar streams through.
#[expect(
    clippy::too_many_lines,
    reason = "the thinking / reasoning-replay / tool-call branches are one wire shape per block kind"
)]
fn convert_assistant_message(
    model: &Model,
    assistant: &AssistantMessage,
    compat: &CompletionsCompat,
    grammar_tool_input_properties: Option<&BTreeMap<String, String>>,
) -> Result<Option<Value>, String> {
    let provider: &str = &model.provider;
    let mut assistant_msg = Map::new();
    // Some providers do not accept null content; the bridge format keeps an
    // empty string.
    assistant_msg.insert("role".to_owned(), json!("assistant"));
    assistant_msg.insert(
        "content".to_owned(),
        if compat.requires_assistant_after_tool_result {
            json!("")
        } else {
            Value::Null
        },
    );

    let assistant_texts: Vec<&str> = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Text(text) if !text.text.trim().is_empty() => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    let assistant_text = assistant_texts.join("");
    let assistant_text_parts: Vec<Value> = assistant_texts
        .iter()
        .map(|text| json!({ "type": "text", "text": text }))
        .collect();

    let thinking_blocks: Vec<&ThinkingContent> = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Thinking(thinking) => Some(thinking),
            _ => None,
        })
        .collect();
    let tool_calls: Vec<&ToolCall> = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .collect();

    let signed_reasoning_details = thinking_blocks
        .iter()
        .find_map(|block| parse_openai_reasoning_details(block.thinking_signature.as_deref()));
    let legacy_reasoning_details: Vec<Value> = tool_calls
        .iter()
        .filter_map(|tool_call| {
            parse_legacy_encrypted_reasoning_detail(tool_call.thought_signature.as_deref())
        })
        .collect();
    let preserved_reasoning_details = signed_reasoning_details
        .or_else(|| (!legacy_reasoning_details.is_empty()).then_some(legacy_reasoning_details));

    let non_empty_thinking_blocks: Vec<&&ThinkingContent> = thinking_blocks
        .iter()
        .filter(|block| !block.thinking.trim().is_empty())
        .collect();
    if !non_empty_thinking_blocks.is_empty() {
        if compat.requires_thinking_as_text {
            // Thinking rides as plain text; tags are avoided so the model
            // does not mimic them.
            let thinking_text = non_empty_thinking_blocks
                .iter()
                .map(|block| block.thinking.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let mut content = vec![json!({ "type": "text", "text": thinking_text })];
            content.extend(assistant_text_parts);
            assistant_msg.insert("content".to_owned(), Value::Array(content));
        } else {
            // Assistant content is a plain string (the Chat Completions
            // standard); a content-part array makes some models mirror the
            // block structure literally in their output.
            if !assistant_text.is_empty() {
                assistant_msg.insert("content".to_owned(), json!(assistant_text));
            }

            // reasoning_details is the structured alternative to a raw
            // reasoning field; a plain field only rides when no structured
            // replay exists.
            if preserved_reasoning_details.is_none() {
                let signature = non_empty_thinking_blocks[0].thinking_signature.as_deref();
                let signature = if provider == "opencode-go" && signature == Some("reasoning") {
                    Some("reasoning_content")
                } else {
                    signature
                };
                if let Some(field) =
                    signature.filter(|field| is_openai_completions_reasoning_field(field))
                {
                    let reasoning_text = non_empty_thinking_blocks
                        .iter()
                        .map(|block| block.thinking.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    assistant_msg.insert(field.to_owned(), json!(reasoning_text));
                }
            }
        }
    } else if !assistant_text.is_empty() {
        assistant_msg.insert("content".to_owned(), json!(assistant_text));
    }

    if !tool_calls.is_empty() {
        let mut converted_calls: Vec<Value> = Vec::new();
        for tool_call in &tool_calls {
            let custom_property = grammar_tool_input_properties
                .and_then(|properties| properties.get(&tool_call.name));
            let call = if let Some(property) = custom_property {
                let input =
                    get_grammar_tool_input(&tool_call.name, &tool_call.arguments, property)?;
                json!({
                    "id": tool_call.id,
                    "type": "custom",
                    "custom": { "name": tool_call.name, "input": input },
                })
            } else {
                json!({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": serde_json::to_string(&tool_call.arguments)
                            .unwrap_or_else(|_| "{}".to_owned()),
                    },
                })
            };
            converted_calls.push(call);
        }
        assistant_msg.insert("tool_calls".to_owned(), Value::Array(converted_calls));
    }

    if let Some(details) = preserved_reasoning_details {
        assistant_msg.insert("reasoning_details".to_owned(), Value::Array(details));
    }
    if compat.requires_reasoning_content_on_assistant_messages
        && model.reasoning
        && !assistant_msg.contains_key("reasoning_content")
    {
        assistant_msg.insert("reasoning_content".to_owned(), json!(""));
    }

    let has_content = match assistant_msg.get("content") {
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(parts)) => !parts.is_empty(),
        _ => false,
    };
    if !has_content && !assistant_msg.contains_key("tool_calls") {
        return Ok(None);
    }
    Ok(Some(Value::Object(assistant_msg)))
}

/// One tool-result run's wire messages, upstream's `msg.role ===
/// "toolResult"` branch: consecutive tool results each send their own `tool`
/// message, extracted images ride a following user message, and Kimi's
/// deferred tools re-enter through a system message that carries `tools`
/// without `content`.
///
/// # Errors
/// The grammar and strict-schema rejections a converted deferred tool
/// definition hits.
fn convert_tool_result_run(
    model: &Model,
    context: &Context,
    compat: &CompletionsCompat,
    transformed_messages: &[Message],
    index: &mut usize,
    params: &mut Vec<Value>,
    last_role: &mut Option<&str>,
) -> Result<(), String> {
    let mut image_blocks: Vec<Value> = Vec::new();
    let mut deferred_tool_names: BTreeSet<String> = BTreeSet::new();
    let mut run = *index;
    while let Some(Message::ToolResult(tool_msg)) = transformed_messages.get(run) {
        let text_result = tool_msg
            .content
            .iter()
            .filter_map(|block| match block {
                ToolResultBlock::Text(text) => Some(text.text.as_str()),
                ToolResultBlock::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let has_images = tool_msg
            .content
            .iter()
            .any(|block| matches!(block, ToolResultBlock::Image(_)));
        // Always send tool result with text; the placeholder keeps image-only
        // and empty results addressable.
        let tool_result_text = if !text_result.is_empty() {
            text_result
        } else if has_images {
            "(see attached image)".to_owned()
        } else {
            "(no tool output)".to_owned()
        };
        let mut tool_result_msg = json!({
            "role": "tool",
            "content": tool_result_text,
            "tool_call_id": tool_msg.tool_call_id,
        });
        if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
            tool_result_msg["name"] = json!(tool_msg.tool_name);
        }
        params.push(tool_result_msg);

        if compat.deferred_tools_mode == Some(DeferredToolsMode::Kimi) {
            for name in tool_msg.added_tool_names.iter().flatten() {
                deferred_tool_names.insert(name.clone());
            }
        }

        if has_images && model.input.contains(&Modality::Image) {
            for block in &tool_msg.content {
                if let ToolResultBlock::Image(image) = block {
                    image_blocks.push(image_url_part(&image.data, &image.mime_type));
                }
            }
        }
        run += 1;
    }
    *index = run;

    if image_blocks.is_empty() {
        *last_role = Some("toolResult");
    } else {
        if compat.requires_assistant_after_tool_result {
            params.push(json!({
                "role": "assistant",
                "content": "I have processed the tool results.",
            }));
        }
        let mut parts = vec![json!({
            "type": "text",
            "text": "Attached image(s) from tool result:",
        })];
        parts.extend(image_blocks);
        params.push(json!({ "role": "user", "content": parts }));
        *last_role = Some("user");
    }

    if !deferred_tool_names.is_empty() {
        let deferred_tools = get_tools_by_name(context.tools.as_deref(), &deferred_tool_names);
        if !deferred_tools.is_empty() {
            // Kimi accepts a system message with tools but omits the standard
            // content field.
            params.push(json!({
                "role": "system",
                "tools": convert_tools(&deferred_tools, compat)?,
            }));
        }
    }
    Ok(())
}

/// The request's tool definitions, upstream's `convertTools`: grammar
/// tools ride the wire's `custom` shape, everything else the `function`
/// shape with `strict` present only when the compat flag admits it.
///
/// # Errors
/// The grammar constrained-sampling rejection for a tool that opted in
/// without a usable variant or schema, and the strict-conversion rejection
/// for a tool that requires strict JSON-schema sampling.
fn convert_tools(tools: &[Tool], compat: &CompletionsCompat) -> Result<Vec<Value>, String> {
    let mut converted: Vec<Value> = Vec::new();
    for tool in tools {
        let grammar =
            resolve_grammar_constrained_sampling(tool, compat.supports_openai_grammar_tools)?;
        if let Some(grammar) = grammar {
            let syntax = match grammar.format {
                GrammarFormat::OpenaiLark => "lark",
                GrammarFormat::OpenaiRegex => "regex",
            };
            converted.push(json!({
                "type": "custom",
                "custom": {
                    "name": tool.name,
                    "description": tool.description,
                    "format": {
                        "type": "grammar",
                        "grammar": {
                            "syntax": syntax,
                            "definition": grammar.definition,
                        },
                    },
                },
            }));
            continue;
        }

        let strict = resolve_json_schema_strict_sampling(tool, compat.supports_strict_mode)?;
        let parameters = get_json_schema_tool_parameters(tool, strict).map_err(|error| error.0)?;
        let mut function = Map::new();
        function.insert("name".to_owned(), json!(tool.name));
        function.insert("description".to_owned(), json!(tool.description));
        function.insert("parameters".to_owned(), parameters);
        // Only include strict when the provider supports it; some reject
        // unknown fields.
        if compat.supports_strict_mode {
            function.insert("strict".to_owned(), json!(strict.unwrap_or(false)));
        }
        converted.push(json!({ "type": "function", "function": function }));
    }
    Ok(converted)
}

// ---------------------------------------------------------------------------
// Chunk parsing
// ---------------------------------------------------------------------------

/// The wire usage fields a chunk reports, priced into the message's
/// [`Usage`], upstream's `parseChunkUsage`. `prompt_tokens_details.cached_tokens`
/// is the cache read; DeepSeek names it `prompt_cache_hit_tokens` and Kimi
/// documents a top-level `cached_tokens`, first hit wins.
/// `prompt_tokens_details.cache_write_tokens` is a separate write count that
/// must not be subtracted from the read count. `completion_tokens` already
/// includes `completion_tokens_details.reasoning_tokens`.
fn parse_chunk_usage(raw_usage: &Value, model: &Model) -> Usage {
    let prompt_tokens = raw_usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let prompt_tokens_details = raw_usage.get("prompt_tokens_details");
    let cache_read_tokens = prompt_tokens_details
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            raw_usage
                .get("prompt_cache_hit_tokens")
                .and_then(Value::as_u64)
        })
        .or_else(|| raw_usage.get("cached_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let cache_write_tokens = prompt_tokens_details
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let input = prompt_tokens
        .saturating_sub(cache_read_tokens)
        .saturating_sub(cache_write_tokens);
    let output_tokens = raw_usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mut usage = Usage {
        input,
        output: output_tokens,
        cache_read: cache_read_tokens,
        cache_write: cache_write_tokens,
        reasoning: Some(
            raw_usage
                .get("completion_tokens_details")
                .and_then(|details| details.get("reasoning_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
        ),
        total_tokens: input
            .saturating_add(output_tokens)
            .saturating_add(cache_read_tokens)
            .saturating_add(cache_write_tokens),
        ..Usage::default()
    };
    calculate_cost(model, &mut usage);
    usage
}

/// The stop reason a wire `finish_reason` maps to, upstream's
/// `mapStopReason`; filtered and unknown reasons fail the stream with the
/// provider's spelling.
fn map_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        // Content flagged by safety filters.
        "content_filter" => (
            StopReason::Error,
            Some("Provider finish_reason: content_filter".to_owned()),
        ),
        "network_error" => (
            StopReason::Error,
            Some("Provider finish_reason: network_error".to_owned()),
        ),
        other => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {other}")),
        ),
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// One stream failure: its message plus the raw-metadata extra some
/// providers via OpenRouter carry in the error body's `metadata.raw` field.
struct StreamFailure {
    message: String,
    raw_metadata: Option<String>,
}

impl StreamFailure {
    /// The failure's display form, the catch block's `errorMessage`: the raw
    /// metadata appends after a newline only when the composed message does
    /// not already contain it, so it never prints twice.
    fn display(&self) -> String {
        if let Some(raw) = &self.raw_metadata
            && !self.message.contains(raw)
        {
            format!("{}\n{raw}", self.message)
        } else {
            self.message.clone()
        }
    }
}

/// Stream an assistant response over the OpenAI Chat Completions wire,
/// upstream's `stream`. The stream returns live; request setup, model, and
/// runtime failures arrive as its `error` event.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&OpenAiCompletionsOptions>,
) -> AssistantMessageEventStream {
    let options = options.cloned();
    let signal = options.as_ref().map_or_else(
        || TransportOptions::default().signal(),
        |options| options.transport_options.signal(),
    );
    spawned_stream(
        model,
        context,
        signal,
        initial_output(model),
        move |model, context, output, forward| {
            Box::pin(async move {
                let options = options.unwrap_or_default();
                run_stream(&model, &context, &options, output, &forward)
                    .await
                    .map_err(|failure| failure.display())
            })
        },
    )
}

/// The fresh accumulator a stream starts from, upstream's `output`: zeroed
/// usage and `stopReason: "pending"`.
fn initial_output(model: &Model) -> AssistantMessage {
    crate::api::request_seam::initial_output(model, model.api.clone(), None)
}

/// Run one stream to completion: resolve the credential, build and dispatch
/// the request, decode the chunks, and settle the final message. A failure
/// replays the streamed reasoning details onto every thinking block before
/// returning, the replay half of upstream's catch-block cleanup; the stop
/// reason and error message are set by [`stream`]'s error arm.
async fn run_stream(
    model: &Model,
    context: &Context,
    options: &OpenAiCompletionsOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), StreamFailure> {
    let mut streamed_reasoning_details: Option<Vec<Value>> = None;
    let result = async {
        let api_key = get_client_api_key(
            &model.provider,
            options.api_key.as_deref(),
            options.headers.as_ref(),
        )
        .map_err(|message| StreamFailure {
            message,
            raw_metadata: None,
        })?;
        let compat = get_compat(model);
        let grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools.as_deref(),
            compat.supports_openai_grammar_tools,
        );
        let cache_retention =
            resolve_cache_retention(options.cache_retention, options.env.as_ref());
        let cache_session_id = (cache_retention != CacheRetention::None)
            .then(|| options.session_id.clone())
            .flatten();
        let payload = build_params(
            model,
            context,
            options,
            &compat,
            cache_retention,
            &grammar_tool_input_properties,
        )
        .map_err(|message| StreamFailure {
            message,
            raw_metadata: None,
        })?;
        let response = dispatch_stream_request(
            model,
            context,
            &compat,
            options,
            &api_key,
            cache_session_id.as_deref(),
            payload,
        )
        .await?;
        events.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });

        let mut state = StreamState {
            tool_calls_by_index: HashMap::new(),
            tool_calls_by_id: HashMap::new(),
            tool_scratch: HashMap::new(),
            text_index: None,
            thinking_index: None,
            has_finish_reason: false,
        };
        iterate_chunks(
            response,
            model,
            &grammar_tool_input_properties,
            output,
            &mut state,
            &mut streamed_reasoning_details,
            events,
        )
        .await?;
        finish_stream(
            options,
            &compat,
            output,
            state,
            streamed_reasoning_details.as_deref(),
            events,
        )
    }
    .await;

    if result.is_err()
        && let Some(details) = streamed_reasoning_details.as_ref()
    {
        let signature = serde_json::to_string(details).unwrap_or_default();
        for block in &mut output.content {
            if let AssistantBlock::Thinking(thinking) = block {
                thinking.thinking_signature = Some(signature.clone());
            }
        }
    }
    result
}

/// Execute the stream request with retries and fire the response hook, the
/// seam-side port of upstream's `createClient` +
/// `retryProviderRequest(create(...).asResponse())`. A non-2xx body is
/// parsed before it is discarded so the failure can surface the OpenRouter
/// `metadata.raw` extra.
async fn dispatch_stream_request(
    model: &Model,
    context: &Context,
    compat: &CompletionsCompat,
    options: &OpenAiCompletionsOptions,
    api_key: &str,
    session_id: Option<&str>,
    payload: Map<String, Value>,
) -> Result<HttpResponse, StreamFailure> {
    let headers = build_request_headers(
        model,
        context,
        compat,
        session_id,
        api_key,
        options.headers.as_ref(),
    );

    let mut payload = payload;
    if let Some(hook) = &options.transport_options.on_payload
        && let Some(Value::Object(next)) = hook
            .call(Value::Object(payload.clone()), model.clone())
            .await
    {
        payload = next;
    }

    let http_client = options.transport_options.client();
    let signal = options.transport_options.signal();
    let request = HttpRequest {
        method: HttpMethod::Post,
        url: completions_url(model),
        headers,
        body: Some(Bytes::from(Value::Object(payload).to_string())),
        timeout_ms: options.timeout_ms,
        signal: signal.clone(),
    };
    let retry_options = ProviderRetryOptions {
        max_retries: options.max_retries.unwrap_or(0),
        max_retry_delay_ms: options.max_retry_delay_ms,
        signal: Some(signal.clone()),
        random: None,
    };
    // The parsed error body survives the retry loop so the failure message
    // and the OpenRouter `metadata.raw` extra can be composed from it.
    let parsed_error_body: Arc<Mutex<Option<Value>>> = Arc::default();
    let response = retry_provider_request(
        || {
            let client = Arc::clone(&http_client);
            let request = request.clone();
            let parsed_error_body = Arc::clone(&parsed_error_body);
            async move { execute_checked_response(client, request, Some(&parsed_error_body)).await }
        },
        &retry_options,
    )
    .await;

    let response = match response {
        Ok(response) => response,
        Err(error) => {
            let parsed_body = parsed_error_body
                .lock()
                .ok()
                .and_then(|mut slot| std::mem::take(&mut *slot));
            let raw_metadata = parsed_body
                .as_ref()
                .and_then(|body| body.get("metadata"))
                .and_then(|metadata| metadata.get("raw"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            let sdk_error = SdkError {
                message: error.message.clone(),
                status: error.status,
                error: parsed_body.map(ErrorBody::Parsed),
                ..SdkError::default()
            };
            return Err(StreamFailure {
                message: format_provider_error(&normalize_provider_error(sdk_error), None),
                raw_metadata,
            });
        }
    };

    fire_response_hook(
        &options.transport_options,
        response.status,
        &response.headers,
        model.clone(),
    )
    .await;

    Ok(response)
}

fn parse_chunk_error_message(detail: &str, sse_event: &ServerSentEvent) -> String {
    format!(
        "Could not parse OpenAI SSE chunk: {detail}; data={}; raw={}",
        sse_event.data,
        sse_event.raw.join("\\n")
    )
}

/// Stream a simple request, upstream's `streamSimple`: the pi reasoning
/// level clamps to the model's supported levels and spends an effort, with
/// the clamped-off state spending none.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    if let Err(message) = get_client_api_key(
        &model.provider,
        options.and_then(|options| options.api_key.as_deref()),
        options.and_then(|options| options.headers.as_ref()),
    ) {
        return setup_error_stream(model, &message);
    }

    let mut base = OpenAiCompletionsOptions::from(build_base_options(
        model,
        context,
        options,
        options.and_then(|options| options.api_key.as_deref()),
    ));
    base.tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(|choice| match choice {
            ToolChoice::Auto => OpenAiToolChoice::Auto,
            ToolChoice::None => OpenAiToolChoice::None,
        });
    base.reasoning_effort = options
        .and_then(|options| options.reasoning)
        .map(|reasoning| clamp_thinking_level(model, thinking_model_level(reasoning)))
        .and_then(pi_thinking_level);
    base.thinking_budgets = options.and_then(|options| options.thinking_budgets);
    stream(model, context, Some(&base))
}

/// The loop-carried state of a chunk stream: the open text and thinking
/// block positions, the tool-call lookup tables keyed by the wire's stream
/// index and call id, the per-block scratch buffers, and whether a finish
/// reason arrived.
struct StreamState {
    tool_calls_by_index: HashMap<i64, usize>,
    tool_calls_by_id: HashMap<String, usize>,
    tool_scratch: HashMap<usize, ToolScratch>,
    text_index: Option<usize>,
    thinking_index: Option<usize>,
    has_finish_reason: bool,
}

/// The streaming scratch of one tool-call block, upstream's `partialArgs` /
/// `customInput` / `streamIndex` fields on the block object. The scratch
/// never persists into the message; `finish_stream` drops it when the block
/// closes.
#[derive(Debug, Default)]
struct ToolScratch {
    partial_args: Option<String>,
    custom: Option<CustomInputState>,
    stream_index: Option<i64>,
}

/// The grammar buffer a custom tool call's raw input streams through,
/// upstream's `customInput`.
#[derive(Debug)]
struct CustomInputState {
    /// The arguments property the grammar's input feeds.
    property: String,
    /// The incremental `{"property":"...","}` wrapper state.
    json_buffer: GrammarToolInputJsonBuffer,
}

/// One `tool_calls` entry of a chunk choice delta, upstream's
/// `StreamingToolCallDelta`. Absent fields leave the tracked call alone.
struct ToolCallDelta<'a> {
    /// The wire's `index`: which tool call of the response this delta
    /// updates.
    stream_index: Option<i64>,
    /// The wire's `id`, carried by the delta that names the call.
    id: Option<&'a str>,
    /// The wire's `function` delta: the tool name and streamed JSON
    /// arguments.
    function: Option<FunctionDelta<'a>>,
    /// The wire's `custom` delta: the tool name and streamed raw input.
    custom: Option<CustomDelta<'a>>,
}

/// The `function` delta of a `tool_calls` entry, the wire's
/// `{ name?, arguments? }`.
struct FunctionDelta<'a> {
    name: Option<&'a str>,
    arguments: Option<&'a str>,
}

/// The `custom` delta of a `tool_calls` entry, the wire's
/// `{ name?, input? }`.
struct CustomDelta<'a> {
    name: Option<&'a str>,
    input: Option<&'a str>,
}

fn parse_tool_call_delta(entry: &Value) -> ToolCallDelta<'_> {
    let function = entry
        .get("function")
        .filter(|function| !function.is_null())
        .map(|function| FunctionDelta {
            name: function.get("name").and_then(Value::as_str),
            arguments: function.get("arguments").and_then(Value::as_str),
        });
    let custom = entry
        .get("custom")
        .filter(|custom| !custom.is_null())
        .map(|custom| CustomDelta {
            name: custom.get("name").and_then(Value::as_str),
            input: custom.get("input").and_then(Value::as_str),
        });
    ToolCallDelta {
        stream_index: entry.get("index").and_then(Value::as_i64),
        id: entry.get("id").and_then(Value::as_str),
        function,
        custom,
    }
}

/// JavaScript truthiness over a wire value, upstream's bare `if (value)`
/// checks: objects and arrays pass; `null`, `false`, `0`, and empty strings
/// fail.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|sample| sample != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// Decode the SSE chunk stream, upstream's
/// `for await (const chunk of openaiStream)`: the `data: [DONE]` sentinel
/// ends the stream and every other frame is one chunk JSON object.
async fn iterate_chunks(
    response: HttpResponse,
    model: &Model,
    grammar_tool_input_properties: &BTreeMap<String, String>,
    output: &mut AssistantMessage,
    state: &mut StreamState,
    streamed_reasoning_details: &mut Option<Vec<Value>>,
    events: &AssistantMessageEventStream,
) -> Result<(), StreamFailure> {
    let mut sse = SseStream::new(response.body);
    while let Some(sse_event) = sse.next().await.map_err(|error| match error {
        HttpError::Aborted => StreamFailure {
            message: "Request was aborted".to_owned(),
            raw_metadata: None,
        },
        other => StreamFailure {
            message: other.to_string(),
            raw_metadata: None,
        },
    })? {
        if sse_event.data.trim() == "[DONE]" {
            continue;
        }
        let chunk =
            serde_json::from_str::<Value>(&sse_event.data).map_err(|error| StreamFailure {
                message: parse_chunk_error_message(&error.to_string(), &sse_event),
                raw_metadata: None,
            })?;
        if !chunk.is_object() {
            continue;
        }
        apply_chunk(
            &chunk,
            model,
            grammar_tool_input_properties,
            output,
            state,
            streamed_reasoning_details,
            events,
        )?;
    }
    Ok(())
}

/// Apply one chunk to the accumulator, upstream's loop body: response
/// identity, usage, the choice's finish reason, and the delta's text,
/// reasoning, tool-call, and reasoning-details updates.
#[expect(
    clippy::too_many_lines,
    reason = "one branch per chunk field; splitting it would scatter the wire reads"
)]
fn apply_chunk(
    chunk: &Value,
    model: &Model,
    grammar_tool_input_properties: &BTreeMap<String, String>,
    output: &mut AssistantMessage,
    state: &mut StreamState,
    streamed_reasoning_details: &mut Option<Vec<Value>>,
    events: &AssistantMessageEventStream,
) -> Result<(), StreamFailure> {
    // OpenAI documents ChatCompletionChunk.id as the unique chat completion
    // identifier; each chunk of a streamed completion carries the same id.
    if output.response_id.as_deref().is_none_or(str::is_empty)
        && let Some(id) = chunk
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
    {
        output.response_id = Some(id.to_owned());
    }
    if let Some(serving_model) = chunk.get("model").and_then(Value::as_str)
        && !serving_model.is_empty()
        && serving_model != model.id
        && output.response_model.as_deref().is_none_or(str::is_empty)
    {
        output.response_model = Some(serving_model.to_owned());
    }
    let chunk_usage = chunk.get("usage").filter(|usage| is_truthy(usage));
    if let Some(usage) = chunk_usage {
        output.usage = parse_chunk_usage(usage, model);
    }

    let Some(choice) = chunk
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .filter(|choice| is_truthy(choice))
    else {
        return Ok(());
    };

    // Fallback: some providers (Moonshot) report usage per choice instead of
    // per chunk.
    if chunk_usage.is_none()
        && let Some(usage) = choice.get("usage").filter(|usage| is_truthy(usage))
    {
        output.usage = parse_chunk_usage(usage, model);
    }

    // The wire's null and empty finish reasons are skipped, JS truthiness.
    if let Some(reason) = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .filter(|reason| !reason.is_empty())
    {
        output.raw_stop_reason = Some(reason.to_owned());
        let (stop_reason, error_message) = map_stop_reason(reason);
        output.stop_reason = stop_reason;
        if let Some(error_message) = error_message {
            output.error_message = Some(error_message);
        }
        state.has_finish_reason = true;
    }

    let Some(delta) = choice.get("delta").filter(|delta| is_truthy(delta)) else {
        return Ok(());
    };

    if let Some(text) = delta
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        let content_index = ensure_text_block(output, state, events);
        if let AssistantBlock::Text(block) = &mut output.content[content_index] {
            block.text.push_str(text);
        }
        events.push(AssistantMessageEvent::TextDelta {
            content_index: content_index as u64,
            delta: text.to_owned(),
            partial: output.clone(),
        });
    }

    // Some endpoints return reasoning in reasoning_content (llama.cpp) or
    // reasoning (other OpenAI-compatible endpoints); the first non-empty
    // field wins so duplicated fields do not double the text.
    let provider: &str = &model.provider;
    let reasoning = ["reasoning_content", "reasoning", "reasoning_text"]
        .into_iter()
        .find_map(|field| {
            let text = delta.get(field)?.as_str()?;
            (!text.is_empty()).then_some((field, text))
        });
    if let Some((field, text)) = reasoning {
        let thinking_signature = if provider == "opencode-go" && field == "reasoning" {
            "reasoning_content"
        } else {
            field
        };
        let content_index = ensure_thinking_block(output, state, events, thinking_signature);
        if let AssistantBlock::Thinking(block) = &mut output.content[content_index] {
            block.thinking.push_str(text);
        }
        events.push(AssistantMessageEvent::ThinkingDelta {
            content_index: content_index as u64,
            delta: text.to_owned(),
            partial: output.clone(),
        });
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for entry in tool_calls {
            if !entry.is_object() {
                continue;
            }
            let tool_call = parse_tool_call_delta(entry);
            let content_index = ensure_tool_call_block(
                output,
                state,
                grammar_tool_input_properties,
                events,
                &tool_call,
            );
            let AssistantBlock::ToolCall(block) = &mut output.content[content_index] else {
                continue;
            };
            if block.id.is_empty()
                && let Some(id) = tool_call.id.filter(|id| !id.is_empty())
            {
                id.clone_into(&mut block.id);
                state.tool_calls_by_id.insert(id.to_owned(), content_index);
            }
            let mut delta_text = String::new();
            let function_arguments = tool_call
                .function
                .as_ref()
                .and_then(|function| function.arguments)
                .filter(|arguments| !arguments.is_empty());
            let custom_input = tool_call
                .custom
                .as_ref()
                .and_then(|custom| custom.input)
                .filter(|input| !input.is_empty());
            if let Some(arguments) = function_arguments {
                arguments.clone_into(&mut delta_text);
                let scratch = state.tool_scratch.entry(content_index).or_default();
                let partial_args = scratch.partial_args.get_or_insert_with(String::new);
                partial_args.push_str(arguments);
                block.arguments = parse_streaming_json(Some(partial_args))
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
            } else if let Some(input) = custom_input
                && let Some(scratch) = state.tool_scratch.get_mut(&content_index)
            {
                let next_input = format!("{}{input}", get_custom_tool_call_input(block, scratch));
                match append_custom_tool_call_input(block, scratch, &next_input, false) {
                    Ok(Some(appended)) => delta_text = appended,
                    Ok(None) => {}
                    Err(message) => {
                        return Err(StreamFailure {
                            message,
                            raw_metadata: None,
                        });
                    }
                }
            }
            events.push(AssistantMessageEvent::ToolcallDelta {
                content_index: content_index as u64,
                delta: delta_text,
                partial: output.clone(),
            });
        }
    }

    // The streamed reasoning details are replay metadata, not user-visible
    // deltas; they accumulate in memory and serialize into the thinking
    // signature when the block closes.
    if let Some(details) = delta.get("reasoning_details").and_then(Value::as_array) {
        for detail in details {
            if !is_openai_reasoning_detail(detail) {
                continue;
            }
            ensure_thinking_block(output, state, events, "");
            let accumulator = streamed_reasoning_details.get_or_insert_with(Vec::new);
            append_openai_reasoning_detail(accumulator, detail.clone());
        }
    }
    Ok(())
}

fn ensure_text_block(
    output: &mut AssistantMessage,
    state: &mut StreamState,
    events: &AssistantMessageEventStream,
) -> usize {
    if let Some(content_index) = state.text_index {
        return content_index;
    }
    output.content.push(AssistantBlock::Text(TextContent {
        text: String::new(),
        text_signature: None,
    }));
    let content_index = output.content.len() - 1;
    state.text_index = Some(content_index);
    events.push(AssistantMessageEvent::TextStart {
        content_index: content_index as u64,
        partial: output.clone(),
    });
    content_index
}

/// The signature names the wire field the thinking text replays through
/// (`reasoning_content`, `reasoning`, `reasoning_text`) and is fixed when
/// the block opens; streamed reasoning details overwrite it at close.
fn ensure_thinking_block(
    output: &mut AssistantMessage,
    state: &mut StreamState,
    events: &AssistantMessageEventStream,
    thinking_signature: &str,
) -> usize {
    if let Some(content_index) = state.thinking_index {
        return content_index;
    }
    output
        .content
        .push(AssistantBlock::Thinking(ThinkingContent {
            thinking: String::new(),
            thinking_signature: Some(thinking_signature.to_owned()),
            redacted: None,
        }));
    let content_index = output.content.len() - 1;
    state.thinking_index = Some(content_index);
    events.push(AssistantMessageEvent::ThinkingStart {
        content_index: content_index as u64,
        partial: output.clone(),
    });
    content_index
}

/// Track or create the tool-call block a delta updates, upstream's
/// `ensureToolCallBlock`: lookup by the wire's stream index, then by id;
/// creation stashes the scratch buffers beside the block. A later `custom`
/// delta on a function-shaped block retrofits the grammar input state,
/// reading the property from the grammar map by the block's name.
#[expect(
    clippy::too_many_lines,
    reason = "the lookup, creation, and retrofit arms are one state machine; splitting them scatters the tables"
)]
fn ensure_tool_call_block(
    output: &mut AssistantMessage,
    state: &mut StreamState,
    grammar_tool_input_properties: &BTreeMap<String, String>,
    events: &AssistantMessageEventStream,
    tool_call: &ToolCallDelta<'_>,
) -> usize {
    let name = tool_call
        .function
        .as_ref()
        .and_then(|function| function.name)
        .or_else(|| tool_call.custom.as_ref().and_then(|custom| custom.name))
        .unwrap_or_default();
    let id = tool_call.id.filter(|id| !id.is_empty());
    let mut found = tool_call
        .stream_index
        .and_then(|stream_index| state.tool_calls_by_index.get(&stream_index).copied());
    if found.is_none() {
        found = id.and_then(|id| state.tool_calls_by_id.get(id).copied());
    }
    let content_index = if let Some(content_index) = found {
        content_index
    } else {
        // The "input" fallback exists so a made-up tool still has a
        // place to stash its deltas.
        let custom_property = if tool_call.custom.is_some() && tool_call.function.is_none() {
            Some(
                grammar_tool_input_properties
                    .get(name)
                    .map_or("input", String::as_str)
                    .to_owned(),
            )
        } else {
            None
        };
        let mut arguments = Map::new();
        if let Some(property) = &custom_property {
            arguments.insert(property.clone(), json!(""));
        }
        output.content.push(AssistantBlock::ToolCall(ToolCall {
            id: tool_call.id.unwrap_or_default().to_owned(),
            name: name.to_owned(),
            arguments,
            thought_signature: None,
            namespace: None,
        }));
        let content_index = output.content.len() - 1;
        state.tool_scratch.insert(
            content_index,
            ToolScratch {
                partial_args: custom_property.is_none().then(String::new),
                custom: custom_property.map(|property| CustomInputState {
                    property,
                    json_buffer: GrammarToolInputJsonBuffer::default(),
                }),
                stream_index: tool_call.stream_index,
            },
        );
        if let Some(stream_index) = tool_call.stream_index {
            state
                .tool_calls_by_index
                .insert(stream_index, content_index);
        }
        if let Some(id) = id {
            state.tool_calls_by_id.insert(id.to_owned(), content_index);
        }
        events.push(AssistantMessageEvent::ToolcallStart {
            content_index: content_index as u64,
            partial: output.clone(),
        });
        content_index
    };
    if let Some(stream_index) = tool_call.stream_index
        && state
            .tool_scratch
            .get(&content_index)
            .and_then(|scratch| scratch.stream_index)
            .is_none()
        && let Some(scratch) = state.tool_scratch.get_mut(&content_index)
    {
        scratch.stream_index = Some(stream_index);
        state
            .tool_calls_by_index
            .insert(stream_index, content_index);
    }
    if let Some(id) = id {
        state.tool_calls_by_id.insert(id.to_owned(), content_index);
    }
    if !name.is_empty()
        && let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index)
        && block.name.is_empty()
    {
        name.clone_into(&mut block.name);
    }
    let retrofit = tool_call.custom.is_some()
        && tool_call.function.is_none()
        && state
            .tool_scratch
            .get(&content_index)
            .is_none_or(|scratch| scratch.custom.is_none());
    if retrofit
        && let Some(block_name) = output
            .content
            .get(content_index)
            .and_then(|block| match block {
                AssistantBlock::ToolCall(block) => Some(block.name.as_str()),
                _ => None,
            })
    {
        let property = grammar_tool_input_properties
            .get(block_name)
            .map_or("input", String::as_str)
            .to_owned();
        if let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index) {
            block.arguments = Map::from_iter([(property.clone(), json!(""))]);
        }
        if let Some(scratch) = state.tool_scratch.get_mut(&content_index) {
            scratch.custom = Some(CustomInputState {
                property,
                json_buffer: GrammarToolInputJsonBuffer::default(),
            });
            scratch.partial_args = None;
        }
    }
    content_index
}

/// The streamed input a custom tool call has accumulated, upstream's
/// `getCustomToolCallInput`: the arguments value under the property, `""`
/// when absent or not a string.
fn get_custom_tool_call_input(block: &ToolCall, scratch: &ToolScratch) -> String {
    let Some(custom) = &scratch.custom else {
        return String::new();
    };
    block
        .arguments
        .get(&custom.property)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// Stream the next custom-tool input through the grammar buffer and restage
/// the arguments object, upstream's `appendCustomToolCallInput`.
///
/// # Errors
/// The grammar buffer's rejection when the input changes non-monotonically
/// or after the property closed.
fn append_custom_tool_call_input(
    block: &mut ToolCall,
    scratch: &mut ToolScratch,
    next_input: &str,
    close: bool,
) -> Result<Option<String>, String> {
    let Some(custom) = &mut scratch.custom else {
        return Ok(None);
    };
    let delta = append_grammar_tool_input_json_delta(
        &mut custom.json_buffer,
        &custom.property,
        next_input,
        close,
    )?;
    block.arguments = Map::from_iter([(custom.property.clone(), json!(next_input))]);
    Ok(delta)
}

/// Close every block in content order and settle the stop reason, upstream's
/// `finishBlock` loop plus the post-stream checks. The `*_end` events are
/// the authoritative block closures; streaming never emits them mid-flight.
#[expect(
    clippy::too_many_lines,
    reason = "the block-closing branches and the stop-reason checks mirror upstream in one place"
)]
fn finish_stream(
    options: &OpenAiCompletionsOptions,
    compat: &CompletionsCompat,
    output: &mut AssistantMessage,
    mut state: StreamState,
    streamed_reasoning_details: Option<&[Value]>,
    events: &AssistantMessageEventStream,
) -> Result<(), StreamFailure> {
    for content_index in 0..output.content.len() {
        match &mut output.content[content_index] {
            AssistantBlock::Text(block) => {
                events.push(AssistantMessageEvent::TextEnd {
                    content_index: content_index as u64,
                    content: block.text.clone(),
                    partial: output.clone(),
                });
            }
            AssistantBlock::Thinking(block) => {
                if let Some(details) = streamed_reasoning_details {
                    block.thinking_signature =
                        Some(serde_json::to_string(details).unwrap_or_default());
                }
                events.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: content_index as u64,
                    content: block.thinking.clone(),
                    partial: output.clone(),
                });
            }
            AssistantBlock::ToolCall(block) => {
                let has_custom_input = state
                    .tool_scratch
                    .get(&content_index)
                    .is_some_and(|scratch| scratch.custom.is_some());
                let mut close_delta: Option<String> = None;
                if has_custom_input {
                    if let Some(scratch) = state.tool_scratch.get_mut(&content_index) {
                        let next_input = get_custom_tool_call_input(block, scratch);
                        match append_custom_tool_call_input(block, scratch, &next_input, true) {
                            Ok(Some(delta)) => close_delta = Some(delta),
                            Ok(None) => {}
                            Err(message) => {
                                return Err(StreamFailure {
                                    message,
                                    raw_metadata: None,
                                });
                            }
                        }
                    }
                } else {
                    let partial_args = state
                        .tool_scratch
                        .get(&content_index)
                        .and_then(|scratch| scratch.partial_args.clone());
                    block.arguments = parse_streaming_json(partial_args.as_deref())
                        .as_object()
                        .cloned()
                        .unwrap_or_default();
                }
                // The scratch buffers never persist into the replayed call.
                state.tool_scratch.remove(&content_index);
                let tool_call = block.clone();
                if let Some(delta) = close_delta {
                    events.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: content_index as u64,
                        delta,
                        partial: output.clone(),
                    });
                }
                events.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: content_index as u64,
                    tool_call,
                    partial: output.clone(),
                });
            }
        }
    }

    if options.transport_options.signal().is_cancelled() {
        return Err(StreamFailure {
            message: "Request was aborted".to_owned(),
            raw_metadata: None,
        });
    }
    if output.stop_reason == StopReason::Aborted {
        return Err(StreamFailure {
            message: "Request was aborted".to_owned(),
            raw_metadata: None,
        });
    }
    if !state.has_finish_reason && !compat.supports_finish_reason {
        output.stop_reason = if output
            .content
            .iter()
            .any(|block| matches!(block, AssistantBlock::ToolCall(_)))
        {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        };
    }
    if output.stop_reason == StopReason::Error {
        return Err(StreamFailure {
            message: output
                .error_message
                .clone()
                .unwrap_or_else(|| "Provider returned an error stop reason".to_owned()),
            raw_metadata: None,
        });
    }
    if (compat.supports_finish_reason && !state.has_finish_reason)
        || output.stop_reason == StopReason::Pending
    {
        return Err(StreamFailure {
            message: "Stream ended without finish_reason".to_owned(),
            raw_metadata: None,
        });
    }

    events.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output.clone(),
    });
    Ok(())
}

/// The OpenAI Completions [`crate::types::ProviderStreams`], upstream's
/// module-level `stream`/`streamSimple` exports behind the uniform dispatch.
#[derive(Debug, Default)]
pub struct OpenAiCompletionsStreams;

crate::api::adapter_belt::impl_provider_streams!(
    OpenAiCompletionsStreams,
    OpenAiCompletionsOptions
);
