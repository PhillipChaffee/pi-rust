//! The Anthropic Messages wire API, ported from
//! `packages/ai/src/api/anthropic-messages.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//!
//! - The `@anthropic-ai/sdk` client collapses onto the
//!   [`crate::http::HttpClient`] seam.
//!   The module assembles the wire request the pinned SDK sent — POST
//!   `{baseUrl}/v1/messages?beta=true`, `anthropic-version: 2023-06-01`,
//!   `X-Api-Key` or an `Authorization: Bearer` pair, the pi user agent, and
//!   `anthropic-beta` carrying the `betas` — and executes it through the
//!   injected or default client. Upstream's `options.client` injection has
//!   no separate shape here: an injected [`crate::http::HttpClient`] replaces the
//!   transport the way `options.fetch` did, and request shaping always runs.
//! - `sanitizeSurrogates` disappears statically: a Rust [`String`] cannot
//!   hold the unpaired surrogates it removed, and `serde_json` rejects
//!   lone-surrogate escapes when reading the wire.
//! - Upstream's hand-rolled SSE byte decoder is the shared
//!   [`crate::http::sse`] decoder; the per-API event parsing lives here.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::auth::resolve::now_ms;
use crate::http::client::{HttpError, HttpMethod, HttpRequest, HttpResponse};
use crate::http::sse::SseStream;
use crate::models::calculate_cost;
use crate::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
    CacheRetention, Context, ImageContent, Message, Model, ModelThinkingLevel, ProviderEnv,
    ProviderId, SimpleStreamOptions, StopReason, StreamOptions, TextContent, ThinkingContent, Tool,
    ToolCall, ToolResultBlock, ToolResultMessage, UserBlock, UserContent,
};
use crate::utils::deferred_tools::split_deferred_tools;
use crate::utils::diagnostics::append_assistant_message_diagnostic;
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::json_parse::{parse_json_with_repair, parse_streaming_json};
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};

use crate::api::constrained_sampling::{
    get_json_schema_tool_parameters, resolve_json_schema_strict_sampling,
};
use crate::api::github_copilot_headers::build_copilot_dynamic_headers;
use crate::api::request_seam::{
    execute_checked_response, fire_response_hook, has_header, setup_error_stream, spawned_stream,
};
use crate::api::simple_options::{
    adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context,
};
use crate::api::transform_messages::{ToolCallIdNormalizer, transform_messages};

/// Resolve the cache retention preference, upstream's
/// `resolveCacheRetention`: the request's value, else `PI_CACHE_RETENTION`,
/// else `short`.
fn resolve_cache_retention(
    cache_retention: Option<CacheRetention>,
    env: Option<&ProviderEnv>,
) -> CacheRetention {
    if let Some(retention) = cache_retention {
        return retention;
    }
    if get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
        return CacheRetention::Long;
    }
    CacheRetention::Short
}

/// The request's `cache_control` marker and the retention it encodes,
/// upstream's `getCacheControl` pair. `none` sends no marker; `long` carries
/// `ttl: "1h"` on models the compat flag admits.
fn get_cache_control(
    model: &Model,
    cache_retention: Option<CacheRetention>,
    env: Option<&ProviderEnv>,
) -> Option<Value> {
    let retention = resolve_cache_retention(cache_retention, env);
    if retention == CacheRetention::None {
        return None;
    }
    let ttl = (retention == CacheRetention::Long
        && get_anthropic_compat(model).supports_long_cache_retention)
        .then_some("1h");
    let cache_control = ttl.map_or_else(
        || json!({ "type": "ephemeral" }),
        |ttl| json!({ "type": "ephemeral", "ttl": ttl }),
    );
    Some(cache_control)
}

/// Stealth mode: mimic Claude Code's tool naming exactly. The version the
/// Claude Code 2.x client headers carry.
const CLAUDE_CODE_VERSION: &str = "2.1.251";

/// Claude Code 2.x tool names (canonical casing), upstream's `claudeCodeTools`.
const CLAUDE_CODE_TOOLS: [&str; 17] = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

/// Convert a tool name to CC canonical casing when it matches
/// case-insensitively, upstream's `toClaudeCodeName`.
fn to_claude_code_name(name: &str) -> String {
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|tool| tool.eq_ignore_ascii_case(name))
        .map_or_else(|| name.to_owned(), |tool| (*tool).to_owned())
}

/// Map an inbound CC-cased tool name back to the context's spelling,
/// upstream's `fromClaudeCodeName`. A case-insensitive lookup, not a rename:
/// a tool the context does not carry comes back unchanged.
fn from_claude_code_name(name: &str, tools: Option<&[Tool]>) -> String {
    let Some(tools) = tools else {
        return name.to_owned();
    };
    if tools.is_empty() {
        return name.to_owned();
    }
    tools
        .iter()
        .find(|tool| tool.name.eq_ignore_ascii_case(name))
        .map_or_else(|| name.to_owned(), |tool| tool.name.clone())
}

/// Convert user/tool-result content blocks to the Anthropic request shape: a
/// single joined string when text-only, otherwise a block array with a
/// placeholder text block ahead of image-only content, upstream's
/// `convertContentBlocks`.
fn convert_content_blocks(content: &[ToolResultBlock]) -> Value {
    let has_images = content
        .iter()
        .any(|block| matches!(block, ToolResultBlock::Image(_)));
    if !has_images {
        return Value::String(
            content
                .iter()
                .filter_map(|block| match block {
                    ToolResultBlock::Text(text) => Some(text.text.as_str()),
                    ToolResultBlock::Image(_) => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    let mut blocks: Vec<Value> = content
        .iter()
        .map(|block| match block {
            ToolResultBlock::Text(text) => json!({ "type": "text", "text": text.text }),
            ToolResultBlock::Image(image) => image_source_block(image),
        })
        .collect();
    let has_text = blocks
        .iter()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("text"));
    if !has_text {
        blocks.insert(0, json!({ "type": "text", "text": "(see attached image)" }));
    }
    Value::Array(blocks)
}

fn image_source_block(image: &ImageContent) -> Value {
    json!({
        "type": "image",
        "source": { "type": "base64", "media_type": image.mime_type, "data": image.data }
    })
}

/// An Anthropic effort level, the wire's `output_config.effort` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnthropicEffort {
    /// Minimal thinking, skips simple tasks.
    Low,
    /// Moderate thinking, may skip simple queries.
    Medium,
    /// Always thinks, deep reasoning.
    High,
    /// Highest reasoning level (Opus 4.7+, Sonnet 5, Fable 5).
    Xhigh,
    /// Always thinks with no constraints (Opus 4.6 only).
    Max,
}

impl AnthropicEffort {
    /// The wire spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl std::fmt::Display for AnthropicEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TryFrom<&str> for AnthropicEffort {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::Xhigh),
            "max" => Ok(Self::Max),
            other => Err(other.to_owned()),
        }
    }
}

/// How thinking content returns in API responses, the wire's
/// `thinking.display` values.
///
/// Anthropic's own API default for Claude Opus 4.7 and Claude Mythos
/// Preview is `"omitted"`; the port defaults to `"summarized"` to keep
/// behavior consistent with older Claude 4 models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnthropicThinkingDisplay {
    /// Thinking blocks carry summarized thinking text.
    #[default]
    Summarized,
    /// Thinking blocks return an empty thinking field; the encrypted
    /// signature still travels back for multi-turn continuity.
    Omitted,
}

impl AnthropicThinkingDisplay {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Summarized => "summarized",
            Self::Omitted => "omitted",
        }
    }
}

/// Anthropic tool choice, upstream's `AnthropicOptions.toolChoice`: string
/// values map to Anthropic's built-in choices; [`AnthropicToolChoice::Tool`]
/// forces a specific tool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnthropicToolChoice {
    /// The wire's `"auto"`.
    Auto,
    /// The wire's `"any"`.
    Any,
    /// The wire's `"none"`.
    None,
    /// The wire's `{ type: "tool", name }`.
    Tool {
        /// The forced tool's name.
        name: String,
    },
}

impl From<crate::types::ToolChoice> for AnthropicToolChoice {
    fn from(choice: crate::types::ToolChoice) -> Self {
        match choice {
            crate::types::ToolChoice::Auto => Self::Auto,
            crate::types::ToolChoice::None => Self::None,
        }
    }
}

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const SERVER_SIDE_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const MID_CONVERSATION_OUTPUT_CONFIG_BETA: &str = "mid-conversation-output-config-2026-07-01";
const THINKING_BINDING_CONTROLS_BETA: &str = "thinking-binding-controls-2026-08-01";

/// The Anthropic Messages options, upstream's `AnthropicOptions`.
///
/// The base request options this adapter reads plus the thinking and
/// tool-choice extras. Base fields upstream carries but this adapter never
/// reads (`samplingParams`, `transport`, `websocketConnectTimeoutMs`) are
/// omitted.
#[derive(Clone, Debug, Default)]
pub struct AnthropicStreamOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks.
    pub transport_options: crate::types::TransportOptions,
    /// The API key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this request.
    pub telemetry_context: Option<pi_telemetry::TelemetryHandle>,
    /// Provider-scoped environment values.
    pub env: Option<ProviderEnv>,
    /// Custom HTTP headers merged over the client defaults; `None` values
    /// suppress a default header.
    pub headers: Option<crate::types::ProviderHeaders>,
    /// HTTP request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts. Default: 0.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait for a server-requested retry.
    pub max_retry_delay_ms: Option<u64>,
    /// Sampling temperature. Incompatible with extended thinking and
    /// unsupported on Claude Opus 4.7+.
    pub temperature: Option<f64>,
    /// Maximum output tokens.
    pub max_tokens: Option<u64>,
    /// Prompt cache retention preference. Default: `"short"`.
    pub cache_retention: Option<CacheRetention>,
    /// Session identifier for session-based caching and routing.
    pub session_id: Option<String>,
    /// Optional request metadata; Anthropic reads `user_id`.
    pub metadata: Option<BTreeMap<String, Value>>,
    /// Enable extended thinking. For adaptive thinking models the model
    /// decides when and how much to think; for older models this drives
    /// budget-based thinking via [`Self::thinking_budget_tokens`]. Default:
    /// omitted unless [`stream_simple`] maps a reasoning level to it.
    pub thinking_enabled: Option<bool>,
    /// Token budget for extended thinking (older models only). Default:
    /// 1024 when thinking is enabled and no budget is provided.
    pub thinking_budget_tokens: Option<u64>,
    /// Effort level for adaptive thinking models; ignored for older models.
    pub effort: Option<AnthropicEffort>,
    /// How thinking content returns. Default: `summarized`.
    pub thinking_display: Option<AnthropicThinkingDisplay>,
    /// Whether to request the interleaved thinking beta header for
    /// non-adaptive thinking models. Default: true.
    pub interleaved_thinking: Option<bool>,
    /// Anthropic tool choice; omitted by default.
    pub tool_choice: Option<AnthropicToolChoice>,
}

crate::api::adapter_belt::impl_stream_options_from!(AnthropicStreamOptions from options {
    thinking_enabled: None,
    thinking_budget_tokens: None,
    effort: None,
    thinking_display: None,
    interleaved_thinking: None,
    tool_choice: None,
});
/// The compat flags the adapter reads, upstream's `getAnthropicCompat`.
/// Unset flags fall back to their defaults; OpenRouter endpoints
/// (provider id or `openrouter.ai` base URL) detect themselves for
/// session affinity.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the flag belt mirrors upstream's compat object; each flag is an independent provider admission"
)]
#[derive(Clone, Copy, Debug)]
struct AnthropicCompat {
    supports_eager_tool_input_streaming: bool,
    supports_long_cache_retention: bool,
    send_session_affinity_headers: bool,
    session_affinity_format: Option<crate::types::SessionAffinityFormat>,
    supports_cache_control_on_tools: bool,
    supports_temperature: bool,
    allow_empty_signature: bool,
    supports_strict_tools: bool,
    supports_mid_convo_effort: bool,
    supports_tool_references: bool,
}

fn get_anthropic_compat(model: &Model) -> AnthropicCompat {
    let is_open_router = model.provider == ProviderId::from("openrouter")
        || model.base_url.contains("openrouter.ai");
    let compat = model.compat.as_ref();
    AnthropicCompat {
        supports_eager_tool_input_streaming: compat
            .and_then(|compat| compat.supports_eager_tool_input_streaming)
            .unwrap_or(true),
        supports_long_cache_retention: compat
            .and_then(|compat| compat.supports_long_cache_retention)
            .unwrap_or(true),
        send_session_affinity_headers: compat
            .and_then(|compat| compat.send_session_affinity_headers)
            .unwrap_or(is_open_router),
        session_affinity_format: compat
            .and_then(|compat| compat.session_affinity_format)
            .or_else(|| is_open_router.then_some(crate::types::SessionAffinityFormat::Openrouter)),
        supports_cache_control_on_tools: compat
            .and_then(|compat| compat.supports_cache_control_on_tools)
            .unwrap_or(true),
        supports_temperature: compat
            .and_then(|compat| compat.supports_temperature)
            .unwrap_or(true),
        allow_empty_signature: compat
            .and_then(|compat| compat.allow_empty_signature)
            .unwrap_or(false),
        supports_strict_tools: compat
            .and_then(|compat| compat.supports_strict_tools)
            .unwrap_or(false),
        supports_mid_convo_effort: compat
            .and_then(|compat| compat.supports_mid_convo_effort)
            .unwrap_or(false),
        supports_tool_references: compat
            .and_then(|compat| compat.supports_tool_references)
            .unwrap_or_else(|| default_supports_tool_references(model)),
    }
}

/// Default `supportsToolReferences`: first-party Anthropic models except
/// Haiku, and models from Claude 4.5 on (which predate tool search by
/// version), upstream's `defaultSupportsToolReferences`.
fn default_supports_tool_references(model: &Model) -> bool {
    if model.provider != ProviderId::from("anthropic") || model.id.contains("haiku") {
        return false;
    }
    // `^claude-(?:opus|sonnet|fable)-(\d+)(?:-(\d+))?(?:-|$)`
    let Some(rest) = model.id.strip_prefix("claude-") else {
        return false;
    };
    for family in ["opus", "sonnet", "fable"] {
        let Some(after_family) = rest
            .strip_prefix(family)
            .and_then(|tail| tail.strip_prefix('-'))
        else {
            continue;
        };
        let (major_digits, tail) = split_digits(after_family);
        if major_digits.is_empty() {
            continue;
        }
        let Ok(major) = major_digits.parse::<u32>() else {
            continue;
        };
        // An 8+-digit run is a date suffix, not a minor version.
        let minor = tail
            .strip_prefix('-')
            .map(split_digits)
            .filter(|(digits, _)| !digits.is_empty() && digits.len() < 8)
            .and_then(|(digits, _)| digits.parse::<u32>().ok())
            .unwrap_or(0);
        return major > 4 || (major == 4 && minor >= 5);
    }
    false
}

fn split_digits(value: &str) -> (&str, &str) {
    let end = value
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(value.len());
    value.split_at(end)
}

/// Merge header sources in order; later sources override by name
/// case-insensitively and a `None` value suppresses the header, upstream's
/// `mergeHeaders` feeding the SDK's case-insensitive `Headers` merge.
fn merge_header_sources(sources: Vec<BTreeMap<String, Option<String>>>) -> Vec<(String, String)> {
    let mut merged: Vec<(String, String)> = Vec::new();
    for source in sources {
        for (name, value) in source {
            match value {
                Some(value) => {
                    if let Some(existing) = merged
                        .iter_mut()
                        .find(|(existing, _)| existing.eq_ignore_ascii_case(&name))
                    {
                        existing.1 = value;
                    } else {
                        merged.push((name, value));
                    }
                }
                None => {
                    merged.retain(|(existing, _)| !existing.eq_ignore_ascii_case(&name));
                }
            }
        }
    }
    merged
}

/// Require request auth: an API key, or a header-owned credential the
/// gateway accepts, upstream's `assertRequestAuth`.
fn assert_request_auth(
    provider: &str,
    api_key: Option<&str>,
    headers: Option<&crate::types::ProviderHeaders>,
) -> Result<(), String> {
    if api_key.is_some() {
        return Ok(());
    }
    if has_header(headers, "authorization")
        || has_header(headers, "x-api-key")
        || has_header(headers, "cf-aig-authorization")
    {
        return Ok(());
    }
    Err(format!("No API key for provider: {provider}"))
}

/// The stream event names the parser admits, upstream's
/// `ANTHROPIC_MESSAGE_EVENTS`.
const ANTHROPIC_MESSAGE_EVENTS: [&str; 6] = [
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
];

/// One decoded Anthropic stream event, the pinned SDK's
/// `BetaRawMessageStreamEvent`. Unknown wire fields pass through unread.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawMessageStreamEvent {
    /// The wire's `message_start`.
    MessageStart { message: StreamMessage },
    /// The wire's `message_delta`.
    MessageDelta {
        delta: MessageDeltaPayload,
        /// Partial usage update; omitted fields preserve the values
        /// `message_start` reported.
        #[serde(default)]
        usage: Option<DeltaUsage>,
        /// Server-reported input transformations, the wire's
        /// `input_transformations`.
        #[serde(default)]
        input_transformations: Option<Vec<InputTransformation>>,
    },
    /// The wire's `message_stop`.
    MessageStop {},
    /// The wire's `content_block_start`.
    ContentBlockStart {
        index: u64,
        content_block: ContentBlockStart,
    },
    /// The wire's `content_block_delta`.
    ContentBlockDelta {
        index: u64,
        delta: ContentBlockDelta,
    },
    /// The wire's `content_block_stop`.
    ContentBlockStop { index: u64 },
}

/// The `message` payload of `message_start`.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
struct StreamMessage {
    /// The response id, the wire's `id`.
    #[serde(default)]
    id: String,
    /// The serving model id when the wire reports one.
    #[serde(default)]
    model: Option<String>,
    /// The initial usage block.
    #[serde(default)]
    usage: StartUsage,
    /// Server-reported input transformations.
    #[serde(default)]
    input_transformations: Option<Vec<InputTransformation>>,
}

/// The `message_start` usage block.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
struct StartUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation: Option<CacheCreation>,
}

/// The `cache_creation` split of the cache-write tokens; only the 1h split
/// prices differently, and unknown fields pass through unread.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
struct CacheCreation {
    #[serde(default)]
    ephemeral_1h_input_tokens: Option<u64>,
}

/// One server-reported input transformation.
#[derive(Debug, Clone, Default, serde::Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct InputTransformation {
    /// The transformation kind, the wire's `type`.
    #[serde(rename = "type", default)]
    kind: Option<String>,
    /// The request path the transformation applied to.
    #[serde(default)]
    path: Option<String>,
    /// Why the transformation ran.
    #[serde(default)]
    reason: Option<String>,
}

/// The `delta` payload of `message_delta`.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
struct MessageDeltaPayload {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<StopDetails>,
}

/// The refusal details of a `refusal` stop reason.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
struct StopDetails {
    #[serde(default)]
    explanation: Option<String>,
}

/// The `message_delta` usage block: a partial update whose `null` and
/// missing fields preserve the values `message_start` reported.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
struct DeltaUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens_details: Option<OutputTokensDetails>,
}

/// The `output_tokens_details` breakdown; Anthropic reports reasoning
/// tokens as a subset of output tokens.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
struct OutputTokensDetails {
    #[serde(default)]
    thinking_tokens: Option<u64>,
}

/// One content block under construction, the wire's `content_block`
/// variants.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockStart {
    /// The wire's `text`.
    Text {
        /// The block's initial text.
        #[serde(default)]
        text: Option<String>,
    },
    /// The wire's `thinking`.
    Thinking {
        /// The block's initial thinking text.
        #[serde(default)]
        thinking: Option<String>,
        /// The signature.
        #[serde(default)]
        signature: Option<String>,
    },
    /// The wire's `redacted_thinking`; the opaque payload rides in `data`.
    RedactedThinking { data: String },
    /// The wire's `tool_use`.
    ToolUse {
        /// The tool call id, the wire's `id`.
        id: String,
        /// The tool name, the wire's `name`.
        name: String,
        /// The initial arguments object.
        #[serde(default)]
        input: Option<serde_json::Map<String, Value>>,
    },
    /// The wire's `fallback`: the server switched models before output
    /// began.
    Fallback {},
}

/// One delta, the wire's `delta` variants of `content_block_delta`.
#[expect(
    clippy::enum_variant_names,
    reason = "each variant spells its wire event verbatim (text_delta, thinking_delta, ...)"
)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockDelta {
    /// The wire's `text_delta`.
    TextDelta {
        /// The appended text.
        text: String,
    },
    /// The wire's `thinking_delta`.
    ThinkingDelta {
        /// The appended thinking text.
        thinking: String,
    },
    /// The wire's `input_json_delta`.
    InputJsonDelta {
        /// The appended partial JSON, the wire's `partial_json`.
        partial_json: String,
    },
    /// The wire's `signature_delta`.
    SignatureDelta {
        /// The appended signature.
        signature: String,
    },
}

/// A content block under construction: its position in the accumulator and
/// the tool-call JSON scratch buffer, the port of upstream's per-block
/// `index`/`partialJson` fields that never persist into the message.
struct BlockEntry {
    wire_index: u64,
    content_index: usize,
    partial_json: String,
}

/// Stream an assistant response over the Anthropic Messages wire, upstream's
/// `stream`. The stream returns live; request setup, model, and runtime
/// failures arrive as its `error` event.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&AnthropicStreamOptions>,
) -> AssistantMessageEventStream {
    let options = options.cloned();
    let signal = options.as_ref().map_or_else(
        || crate::types::TransportOptions::default().signal(),
        |options| options.transport_options.signal(),
    );
    let default_options = AnthropicStreamOptions::default();
    spawned_stream(
        model,
        context,
        signal,
        initial_output(model, options.as_ref().unwrap_or(&default_options)),
        move |model, context, output, forward| {
            Box::pin(async move {
                let options = options.unwrap_or_default();
                run_stream(&model, &context, &options, output, &forward).await
            })
        },
    )
}

/// The fresh accumulator a stream starts from, upstream's `output`: zeroed
/// usage, `stopReason: "pending"`, and the managed effort level when the
/// model supports mid-conversation effort.
fn initial_output(model: &Model, options: &AnthropicStreamOptions) -> AssistantMessage {
    let provider_thinking_level =
        (get_anthropic_compat(model).supports_mid_convo_effort).then(|| {
            options
                .effort
                .unwrap_or(AnthropicEffort::High)
                .as_str()
                .to_owned()
        });
    crate::api::request_seam::initial_output(model, model.api.clone(), provider_thinking_level)
}

async fn run_stream(
    model: &Model,
    context: &Context,
    options: &AnthropicStreamOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let (response, is_oauth) = dispatch_stream_request(model, context, options).await?;
    events.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });
    let mut state = StreamState {
        entries: Vec::new(),
        input_transformations: None,
        usage_model: model.clone(),
        saw_message_start: false,
        saw_message_end: false,
    };
    iterate_anthropic_events(
        response, output, &mut state, is_oauth, model, context, events,
    )
    .await?;
    finish_stream(options, output, events, state)
}

/// Execute the stream request with retries and fire the response hook,
/// the seam-side port of upstream's `createClient` +
/// `retryProviderRequest(create(...).asResponse())`.
async fn dispatch_stream_request(
    model: &Model,
    context: &Context,
    options: &AnthropicStreamOptions,
) -> Result<(HttpResponse, bool), String> {
    let api_key = options.api_key.clone();
    // An injected client owns its auth semantics (upstream's `options.client`
    // branch skips the assert); the constructed-client path requires a
    // credential.
    if options.transport_options.http_client.is_none() {
        assert_request_auth(
            &model.provider,
            api_key.as_deref(),
            options.headers.as_ref(),
        )?;
    }

    let copilot_headers = if model.provider == ProviderId::from("github-copilot") {
        let has_images =
            crate::api::github_copilot_headers::has_copilot_vision_input(&context.messages);
        Some(build_copilot_dynamic_headers(&context.messages, has_images))
    } else {
        None
    };

    let cache_retention = resolve_cache_retention(options.cache_retention, options.env.as_ref());
    let cache_session_id = (cache_retention != CacheRetention::None)
        .then(|| options.session_id.clone())
        .flatten();

    let (mut headers, is_oauth) = build_client_headers(
        model,
        api_key.as_deref(),
        options.headers.as_ref(),
        copilot_headers.as_ref(),
        cache_session_id.as_deref(),
    );

    let mut payload = build_params(model, context, is_oauth, options)?;
    if let Some(hook) = &options.transport_options.on_payload {
        payload = hook
            .call(payload.clone(), model.clone())
            .await
            .unwrap_or(payload);
    }
    if let Some(body) = payload.as_object_mut() {
        body.insert("stream".to_owned(), json!(true));
    }

    // The pinned SDK lifts `betas` out of the params into the
    // `anthropic-beta` header; the request body never carries them.
    let betas: Vec<String> = payload
        .get("betas")
        .and_then(Value::as_array)
        .map(|features| {
            features
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if !betas.is_empty() {
        headers.push(("anthropic-beta".to_owned(), betas.join(",")));
    }
    if let Some(body) = payload.as_object_mut() {
        body.remove("betas");
    }

    let http_client = options.transport_options.client();
    let signal = options.transport_options.signal();
    let request = HttpRequest {
        method: HttpMethod::Post,
        url: messages_url(model),
        headers,
        body: Some(Bytes::from(payload.to_string())),
        timeout_ms: options.timeout_ms,
        signal: signal.clone(),
    };
    let retry_options = ProviderRetryOptions {
        max_retries: options.max_retries.unwrap_or(0),
        max_retry_delay_ms: options.max_retry_delay_ms,
        signal: Some(signal.clone()),
        random: None,
    };
    let response = retry_provider_request(
        || {
            let client = Arc::clone(&http_client);
            let request = request.clone();
            async move { execute_checked_response(client, request, None).await }
        },
        &retry_options,
    )
    .await
    .map_err(|error| error.message)?;

    fire_response_hook(
        &options.transport_options,
        response.status,
        &response.headers,
        model.clone(),
    )
    .await;

    Ok((response, is_oauth))
}

async fn iterate_anthropic_events(
    response: HttpResponse,
    output: &mut AssistantMessage,
    state: &mut StreamState,
    is_oauth: bool,
    requested_model: &Model,
    context: &Context,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let mut sse = SseStream::new(response.body);
    while let Some(sse_event) = sse.next().await.map_err(|error| match error {
        HttpError::Aborted => "Request was aborted".to_owned(),
        other => other.to_string(),
    })? {
        if sse_event.event.as_deref() == Some("error") {
            return Err(sse_event.data);
        }
        let Some(event_name) = sse_event.event.as_deref() else {
            continue;
        };
        if !ANTHROPIC_MESSAGE_EVENTS.contains(&event_name) {
            continue;
        }
        let event: RawMessageStreamEvent = parse_json_with_repair(&sse_event.data)
            .map_err(|error| parse_error_message(event_name, &error.to_string(), &sse_event))
            .and_then(|value| {
                serde_json::from_value(value).map_err(|error| {
                    parse_error_message(event_name, &error.to_string(), &sse_event)
                })
            })?;

        match event {
            RawMessageStreamEvent::MessageStart { message } => {
                state.saw_message_start = true;
                apply_message_start(&message, output, state, requested_model);
            }
            RawMessageStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                apply_content_block_start(
                    index,
                    &content_block,
                    output,
                    state,
                    is_oauth,
                    context,
                    events,
                )?;
            }
            RawMessageStreamEvent::ContentBlockDelta { index, delta } => {
                apply_content_block_delta(index, &delta, output, state, events);
            }
            RawMessageStreamEvent::ContentBlockStop { index } => {
                apply_content_block_stop(index, output, state, events);
            }
            RawMessageStreamEvent::MessageDelta {
                delta,
                usage,
                input_transformations: transformations,
            } => {
                apply_message_delta(
                    &delta,
                    usage.as_ref(),
                    transformations.as_ref(),
                    output,
                    state,
                )?;
            }
            RawMessageStreamEvent::MessageStop {} => {
                state.saw_message_end = true;
            }
        }
    }

    if state.saw_message_start && !state.saw_message_end {
        return Err("Anthropic stream ended before message_stop".to_owned());
    }

    Ok(())
}

fn finish_stream(
    options: &AnthropicStreamOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
    state: StreamState,
) -> Result<(), String> {
    if options.transport_options.signal().is_cancelled() {
        return Err("Request was aborted".to_owned());
    }
    if output.stop_reason == StopReason::Pending {
        return Err("Anthropic stream ended without a stop reason".to_owned());
    }
    if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
        return Err(output
            .error_message
            .clone()
            .unwrap_or_else(|| "An unknown error occurred".to_owned()));
    }

    if let Some(transformations) = state
        .input_transformations
        .filter(|transformations| !transformations.is_empty())
    {
        append_input_transformations_diagnostic(output, &transformations);
    }

    events.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output.clone(),
    });
    Ok(())
}

/// Attach the server-reported input transformations to the message as a
/// redacted diagnostic, upstream's `appendAssistantMessageDiagnostic`
/// block.
fn append_input_transformations_diagnostic(
    output: &mut AssistantMessage,
    transformations: &[InputTransformation],
) {
    let mut details = BTreeMap::new();
    details.insert(
        "transformations".to_owned(),
        Value::Array(
            transformations
                .iter()
                .map(|transformation| {
                    // Absent fields drop their key, the wire's
                    // `?? undefined` serialization.
                    let mut entry = serde_json::Map::new();
                    if let Some(kind) = &transformation.kind {
                        entry.insert("type".to_owned(), json!(kind));
                    }
                    if let Some(path) = &transformation.path {
                        entry.insert("path".to_owned(), json!(path));
                    }
                    if let Some(reason) = &transformation.reason {
                        entry.insert("reason".to_owned(), json!(reason));
                    }
                    Value::Object(entry)
                })
                .collect(),
        ),
    );
    append_assistant_message_diagnostic(
        output,
        AssistantMessageDiagnostic {
            kind: "anthropic_input_transformations".to_owned(),
            timestamp: now_ms(),
            error: None,
            details: Some(details),
        },
    );
}

/// The stream's loop-carried state: the open content blocks with their
/// tool-call scratch buffers, the pricing model the fallback selected, and
/// the server-reported input transformations.
struct StreamState {
    entries: Vec<BlockEntry>,
    input_transformations: Option<Vec<InputTransformation>>,
    usage_model: Model,
    saw_message_start: bool,
    saw_message_end: bool,
}

/// Apply a `message_start` event: capture the serving model, its pricing,
/// and the input accounting, upstream's `event.type === "message_start"`
/// branch.
fn apply_message_start(
    message: &StreamMessage,
    output: &mut AssistantMessage,
    state: &mut StreamState,
    requested_model: &Model,
) {
    output.response_id = Some(message.id.clone());
    if let Some(transformations) = &message.input_transformations {
        state.input_transformations = Some(transformations.clone());
    }
    if let Some(serving_model) = &message.model {
        output.model.clone_from(serving_model);
    }
    // A fallback serving model prices the response with the fallback's
    // locally recorded rates.
    let fallback_cost = (output.model != requested_model.id).then(|| {
        requested_model
            .compat
            .as_ref()
            .and_then(|compat| compat.allowed_fallback_models.as_ref())
            .and_then(|fallbacks| {
                fallbacks
                    .iter()
                    .find(|fallback| {
                        fallback.provider == requested_model.provider
                            && fallback.model == output.model
                    })
                    .map(|fallback| fallback.cost.clone())
            })
    });
    if let Some(Some(cost)) = fallback_cost {
        state.usage_model = Model {
            id: output.model.clone(),
            cost,
            ..requested_model.clone()
        };
    }
    let usage = &message.usage;
    // message_start carries the request's input accounting; keep it even
    // when the stream aborts before message_delta.
    output.usage.input = usage.input_tokens.unwrap_or(0);
    output.usage.output = usage.output_tokens.unwrap_or(0);
    output.usage.cache_read = usage.cache_read_input_tokens.unwrap_or(0);
    output.usage.cache_write = usage.cache_creation_input_tokens.unwrap_or(0);
    output.usage.cache_write_1h = Some(
        usage
            .cache_creation
            .as_ref()
            .and_then(|creation| creation.ephemeral_1h_input_tokens)
            .unwrap_or(0),
    );
    // Anthropic does not report total_tokens; compute from the components.
    output.usage.total_tokens = output
        .usage
        .input
        .saturating_add(output.usage.output)
        .saturating_add(output.usage.cache_read)
        .saturating_add(output.usage.cache_write);
    calculate_cost(&state.usage_model, &mut output.usage);
}

/// Apply a `content_block_start` event, upstream's opening-block branches:
/// a fallback after output fails the stream, every other block opens a
/// tracked entry and its `*_start` stream event.
fn apply_content_block_start(
    index: u64,
    content_block: &ContentBlockStart,
    output: &mut AssistantMessage,
    state: &mut StreamState,
    is_oauth: bool,
    context: &Context,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    match content_block {
        ContentBlockStart::Fallback {} => {
            // A fallback after output began has no defined replay; fail so
            // the model restarts from the last valid state.
            if !output.content.is_empty() {
                return Err(
                    "Anthropic performed an unsupported mid-output model fallback".to_owned(),
                );
            }
            Ok(())
        }
        ContentBlockStart::Text { text } => {
            output.content.push(AssistantBlock::Text(TextContent {
                text: text.clone().unwrap_or_default(),
                text_signature: None,
            }));
            state.entries.push(BlockEntry {
                wire_index: index,
                content_index: output.content.len() - 1,
                partial_json: String::new(),
            });
            events.push(AssistantMessageEvent::TextStart {
                content_index: (output.content.len() - 1) as u64,
                partial: output.clone(),
            });
            Ok(())
        }
        ContentBlockStart::Thinking {
            thinking,
            signature,
        } => {
            output
                .content
                .push(AssistantBlock::Thinking(ThinkingContent {
                    thinking: thinking.clone().unwrap_or_default(),
                    thinking_signature: Some(signature.clone().unwrap_or_default()),
                    redacted: None,
                }));
            state.entries.push(BlockEntry {
                wire_index: index,
                content_index: output.content.len() - 1,
                partial_json: String::new(),
            });
            events.push(AssistantMessageEvent::ThinkingStart {
                content_index: (output.content.len() - 1) as u64,
                partial: output.clone(),
            });
            Ok(())
        }
        ContentBlockStart::RedactedThinking { data } => {
            output
                .content
                .push(AssistantBlock::Thinking(ThinkingContent {
                    thinking: "[Reasoning redacted]".to_owned(),
                    thinking_signature: Some(data.clone()),
                    redacted: Some(true),
                }));
            state.entries.push(BlockEntry {
                wire_index: index,
                content_index: output.content.len() - 1,
                partial_json: String::new(),
            });
            events.push(AssistantMessageEvent::ThinkingStart {
                content_index: (output.content.len() - 1) as u64,
                partial: output.clone(),
            });
            Ok(())
        }
        ContentBlockStart::ToolUse { id, name, input } => {
            let name = if is_oauth {
                from_claude_code_name(name, context.tools.as_deref())
            } else {
                name.clone()
            };
            output.content.push(AssistantBlock::ToolCall(ToolCall {
                id: id.clone(),
                name,
                arguments: input.clone().unwrap_or_default(),
                thought_signature: None,
                namespace: None,
            }));
            state.entries.push(BlockEntry {
                wire_index: index,
                content_index: output.content.len() - 1,
                partial_json: String::new(),
            });
            events.push(AssistantMessageEvent::ToolcallStart {
                content_index: (output.content.len() - 1) as u64,
                partial: output.clone(),
            });
            Ok(())
        }
    }
}

/// Apply a `content_block_delta` event against the block under
/// construction, upstream's `content_block_delta` branch.
fn apply_content_block_delta(
    index: u64,
    delta: &ContentBlockDelta,
    output: &mut AssistantMessage,
    state: &mut StreamState,
    events: &AssistantMessageEventStream,
) {
    let Some(entry) = state
        .entries
        .iter_mut()
        .find(|entry| entry.wire_index == index)
    else {
        return;
    };
    let content_index = entry.content_index;
    match (delta, &mut output.content[content_index]) {
        (ContentBlockDelta::TextDelta { text }, AssistantBlock::Text(block)) => {
            block.text += text;
            events.push(AssistantMessageEvent::TextDelta {
                content_index: content_index as u64,
                delta: text.clone(),
                partial: output.clone(),
            });
        }
        (ContentBlockDelta::ThinkingDelta { thinking }, AssistantBlock::Thinking(block)) => {
            block.thinking += thinking;
            events.push(AssistantMessageEvent::ThinkingDelta {
                content_index: content_index as u64,
                delta: thinking.clone(),
                partial: output.clone(),
            });
        }
        (ContentBlockDelta::InputJsonDelta { partial_json }, AssistantBlock::ToolCall(block)) => {
            entry.partial_json += partial_json;
            block.arguments = parsed_arguments(&entry.partial_json);
            events.push(AssistantMessageEvent::ToolcallDelta {
                content_index: content_index as u64,
                delta: partial_json.clone(),
                partial: output.clone(),
            });
        }
        (ContentBlockDelta::SignatureDelta { signature }, AssistantBlock::Thinking(block)) => {
            block
                .thinking_signature
                .get_or_insert_with(String::new)
                .push_str(signature);
        }
        _ => {}
    }
}

/// Apply a `content_block_stop` event: finalize the block, drop its scratch
/// buffer, and emit the authoritative `*_end` event, upstream's
/// `content_block_stop` branch.
fn apply_content_block_stop(
    index: u64,
    output: &mut AssistantMessage,
    state: &mut StreamState,
    events: &AssistantMessageEventStream,
) {
    let Some(position) = state
        .entries
        .iter()
        .position(|entry| entry.wire_index == index)
    else {
        return;
    };
    let entry = state.entries.remove(position);
    let content_index = entry.content_index;
    match &mut output.content[content_index] {
        AssistantBlock::Text(block) => {
            events.push(AssistantMessageEvent::TextEnd {
                content_index: content_index as u64,
                content: block.text.clone(),
                partial: output.clone(),
            });
        }
        AssistantBlock::Thinking(block) => {
            events.push(AssistantMessageEvent::ThinkingEnd {
                content_index: content_index as u64,
                content: block.thinking.clone(),
                partial: output.clone(),
            });
        }
        AssistantBlock::ToolCall(block) => {
            block.arguments = parsed_arguments(&entry.partial_json);
            events.push(AssistantMessageEvent::ToolcallEnd {
                content_index: content_index as u64,
                tool_call: block.clone(),
                partial: output.clone(),
            });
        }
    }
}

/// Apply a `message_delta` event: stop reason, partial usage, and pricing,
/// upstream's `message_delta` branch.
fn apply_message_delta(
    delta: &MessageDeltaPayload,
    usage: Option<&DeltaUsage>,
    transformations: Option<&Vec<InputTransformation>>,
    output: &mut AssistantMessage,
    state: &mut StreamState,
) -> Result<(), String> {
    if let Some(transformations) = transformations {
        state.input_transformations = Some(transformations.clone());
    }
    if let Some(reason) = delta.stop_reason.as_deref() {
        output.raw_stop_reason = Some(reason.to_owned());
        let (stop_reason, error_message) = map_stop_reason(reason, delta.stop_details.as_ref())?;
        output.stop_reason = stop_reason;
        if let Some(error_message) = error_message {
            output.error_message = Some(error_message);
        }
    }
    if let Some(usage) = usage {
        // Only present fields update; proxies that omit usage in
        // message_delta keep the message_start input counts.
        if let Some(input_tokens) = usage.input_tokens {
            output.usage.input = input_tokens;
        }
        if let Some(output_tokens) = usage.output_tokens {
            output.usage.output = output_tokens;
        }
        if let Some(cache_read) = usage.cache_read_input_tokens {
            output.usage.cache_read = cache_read;
        }
        if let Some(cache_write) = usage.cache_creation_input_tokens {
            output.usage.cache_write = cache_write;
        }
        if let Some(thinking_tokens) = usage
            .output_tokens_details
            .as_ref()
            .and_then(|details| details.thinking_tokens)
        {
            // Anthropic reports reasoning tokens as a subset of output
            // tokens.
            output.usage.reasoning = Some(thinking_tokens);
        }
    }
    output.usage.total_tokens = output
        .usage
        .input
        .saturating_add(output.usage.output)
        .saturating_add(output.usage.cache_read)
        .saturating_add(output.usage.cache_write);
    calculate_cost(&state.usage_model, &mut output.usage);
    Ok(())
}

fn parsed_arguments(partial_json: &str) -> serde_json::Map<String, Value> {
    parse_streaming_json(Some(partial_json))
        .as_object()
        .cloned()
        .unwrap_or_default()
}

fn parse_error_message(
    event_name: &str,
    detail: &str,
    sse_event: &crate::http::sse::ServerSentEvent,
) -> String {
    format!(
        "Could not parse Anthropic SSE event {event_name}: {detail}; data={}; raw={}",
        sse_event.data,
        sse_event.raw.join("\\n")
    )
}

/// Map a pi thinking level to an Anthropic effort for adaptive thinking,
/// upstream's `mapThinkingLevelToEffort`: the model's `thinkingLevelMap`
/// entry when it names an effort, else the level itself with `xhigh` and
/// `max` falling to `high`.
fn map_thinking_level_to_effort(
    model: &Model,
    level: crate::types::ThinkingLevel,
) -> AnthropicEffort {
    let model_level = match level {
        crate::types::ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
        crate::types::ThinkingLevel::Low => ModelThinkingLevel::Low,
        crate::types::ThinkingLevel::Medium => ModelThinkingLevel::Medium,
        crate::types::ThinkingLevel::High => ModelThinkingLevel::High,
        crate::types::ThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
        crate::types::ThinkingLevel::Max => ModelThinkingLevel::Max,
    };
    if let Some(mapped) = model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&model_level))
        .and_then(|mapped| mapped.as_deref())
        .and_then(|mapped| AnthropicEffort::try_from(mapped).ok())
    {
        return mapped;
    }
    match level {
        crate::types::ThinkingLevel::Minimal | crate::types::ThinkingLevel::Low => {
            AnthropicEffort::Low
        }
        crate::types::ThinkingLevel::Medium => AnthropicEffort::Medium,
        crate::types::ThinkingLevel::High
        | crate::types::ThinkingLevel::Xhigh
        | crate::types::ThinkingLevel::Max => AnthropicEffort::High,
    }
}

/// Stream a simple request, upstream's `streamSimple`: no reasoning means
/// thinking off; adaptive thinking models spend an effort level, older
/// models spend a token budget inside the response cap.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    if let Err(message) = assert_request_auth(
        &model.provider,
        options.and_then(|options| options.api_key.as_deref()),
        options.and_then(|options| options.headers.as_ref()),
    ) {
        return setup_error_stream(model, &message);
    }

    let mut base = AnthropicStreamOptions::from(build_base_options(
        model,
        context,
        options,
        options.and_then(|options| options.api_key.as_deref()),
    ));
    base.tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(AnthropicToolChoice::from);

    let Some(reasoning) = options.and_then(|options| options.reasoning) else {
        base.thinking_enabled = Some(false);
        return stream(model, context, Some(&base));
    };

    // Adaptive thinking models: the effort level decides the depth.
    if model
        .compat
        .as_ref()
        .and_then(|compat| compat.force_adaptive_thinking)
        == Some(true)
    {
        base.thinking_enabled = Some(true);
        base.effort = Some(map_thinking_level_to_effort(model, reasoning));
        return stream(model, context, Some(&base));
    }

    // Budget-based thinking for older models. The caller's absent cap lets
    // the model cap stand; the budget fits inside the room left for the
    // answer.
    let (adjusted_max_tokens, thinking_budget) = adjust_max_tokens_for_thinking(
        base.max_tokens,
        model.max_tokens,
        reasoning,
        options.and_then(|options| options.thinking_budgets.as_ref()),
    );
    let max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max_tokens);
    base.max_tokens = Some(max_tokens);
    base.thinking_enabled = Some(true);
    base.thinking_budget_tokens = Some(thinking_budget.min(max_tokens.saturating_sub(1024)));
    stream(model, context, Some(&base))
}

fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

/// Assemble the request headers a stream sends, the port of upstream's
/// `createClient`: bearer auth for Copilot and OAuth keys, `X-Api-Key`
/// otherwise, the pi user agent, the Claude Code identity headers for OAuth,
/// and session-affinity headers when the compat flag admits them. Returns
/// the merged pairs and whether the request runs in OAuth mode.
fn build_client_headers(
    model: &Model,
    api_key: Option<&str>,
    options_headers: Option<&crate::types::ProviderHeaders>,
    dynamic_headers: Option<&BTreeMap<String, String>>,
    session_id: Option<&str>,
) -> (Vec<(String, String)>, bool) {
    let is_oauth = api_key.is_some_and(is_oauth_token);
    let mut sources: Vec<BTreeMap<String, Option<String>>> = Vec::new();

    if model.provider == ProviderId::from("github-copilot") {
        // Copilot: bearer auth.
        if let Some(api_key) = api_key {
            sources.push(BTreeMap::from([(
                "authorization".to_owned(),
                Some(format!("Bearer {api_key}")),
            )]));
        }
        sources.push(base_browser_headers());
        sources.push(model_headers_source(model));
        if let Some(dynamic_headers) = dynamic_headers {
            sources.push(
                dynamic_headers
                    .iter()
                    .map(|(name, value)| (name.clone(), Some(value.clone())))
                    .collect(),
            );
        }
    } else if is_oauth {
        // OAuth: bearer auth, Claude Code identity headers.
        sources.push(BTreeMap::from([(
            "authorization".to_owned(),
            Some(format!("Bearer {}", api_key.unwrap_or_default())),
        )]));
        sources.push(base_browser_headers());
        sources.push(BTreeMap::from([
            (
                "user-agent".to_owned(),
                Some(format!("claude-cli/{CLAUDE_CODE_VERSION}")),
            ),
            ("x-app".to_owned(), Some("cli".to_owned())),
        ]));
        sources.push(model_headers_source(model));
    } else {
        // API key or header-owned auth.
        let compat = get_anthropic_compat(model);
        if let Some(api_key) = api_key {
            sources.push(BTreeMap::from([(
                "x-api-key".to_owned(),
                Some(api_key.to_owned()),
            )]));
        }
        if let Some(session_id) = session_id
            && compat.send_session_affinity_headers
        {
            let header = if compat.session_affinity_format
                == Some(crate::types::SessionAffinityFormat::Openrouter)
            {
                "x-session-id"
            } else {
                "x-session-affinity"
            };
            sources.push(BTreeMap::from([(
                header.to_owned(),
                Some(session_id.to_owned()),
            )]));
        }
        sources.push(base_browser_headers());
        sources.push(model_headers_source(model));
    }

    sources.push(options_headers.cloned().unwrap_or_default());
    sources.insert(
        0,
        BTreeMap::from([("User-Agent".to_owned(), Some(get_pi_user_agent()))]),
    );
    (merge_header_sources(sources), is_oauth)
}

/// The headers every request carries: the SDK's pinned `anthropic-version`,
/// the JSON accept, and the direct-browser-access flag upstream sets.
fn base_browser_headers() -> BTreeMap<String, Option<String>> {
    BTreeMap::from([
        (
            "anthropic-version".to_owned(),
            Some("2023-06-01".to_owned()),
        ),
        ("accept".to_owned(), Some("application/json".to_owned())),
        (
            "anthropic-dangerous-direct-browser-access".to_owned(),
            Some("true".to_owned()),
        ),
    ])
}

fn model_headers_source(model: &Model) -> BTreeMap<String, Option<String>> {
    model
        .headers
        .as_ref()
        .map(|headers| {
            headers
                .iter()
                .map(|(name, value)| (name.clone(), Some(value.clone())))
                .collect()
        })
        .unwrap_or_default()
}

/// The stream request URL, the pinned SDK's beta-messages path.
fn messages_url(model: &Model) -> String {
    let base = model.base_url.trim_end_matches('/');
    format!("{base}/v1/messages?beta=true")
}

/// The beta features the request carries, upstream's `getBetaFeatures`: an
/// explicit `anthropic-beta` header on the model or the request wins
/// (comma-split, deduplicated, `null` suppresses the field entirely);
/// otherwise the OAuth, tool-streaming, thinking, fallback, and
/// managed-effort features assemble in order.
fn get_beta_features(
    model: &Model,
    context: &Context,
    is_oauth_token: bool,
    options: &AnthropicStreamOptions,
) -> Vec<String> {
    let mut configured: Option<Option<String>> = None;
    for headers in [
        model_headers_source(model),
        options.headers.clone().unwrap_or_default(),
    ] {
        for (name, value) in headers {
            if name.eq_ignore_ascii_case("anthropic-beta") {
                configured = Some(value);
            }
        }
    }
    if let Some(configured) = configured {
        return configured.map_or_else(Vec::new, |features| {
            dedup(
                features
                    .split(',')
                    .map(str::trim)
                    .filter(|feature| !feature.is_empty())
                    .map(str::to_owned)
                    .collect(),
            )
        });
    }

    let mut features: Vec<String> = Vec::new();
    if is_oauth_token {
        features.push("claude-code-20250219".to_owned());
        features.push("oauth-2025-04-20".to_owned());
    }
    if should_use_fine_grained_tool_streaming_beta(model, context) {
        features.push(FINE_GRAINED_TOOL_STREAMING_BETA.to_owned());
    }
    if model.reasoning
        && options.thinking_enabled == Some(true)
        && options.interleaved_thinking.unwrap_or(true)
        && model
            .compat
            .as_ref()
            .and_then(|compat| compat.force_adaptive_thinking)
            != Some(true)
    {
        features.push(INTERLEAVED_THINKING_BETA.to_owned());
    }
    if model
        .compat
        .as_ref()
        .and_then(|compat| compat.allowed_fallback_models.as_ref())
        .is_some_and(|fallbacks| !fallbacks.is_empty())
    {
        features.push(SERVER_SIDE_FALLBACK_BETA.to_owned());
    }
    if get_anthropic_compat(model).supports_mid_convo_effort {
        features.push(MID_CONVERSATION_OUTPUT_CONFIG_BETA.to_owned());
        features.push(THINKING_BINDING_CONTROLS_BETA.to_owned());
    }
    dedup(features)
}

fn dedup(features: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    features
        .into_iter()
        .filter(|feature| seen.insert(feature.clone()))
        .collect()
}

/// The request payload, upstream's `buildParams` over
/// `MessageCreateParamsStreaming`: transformed messages, the system prompt
/// under cache control, tools, thinking parameters, and the managed-effort
/// markers. `betas` rides the `anthropic-beta` header, not the body.
#[expect(
    clippy::too_many_lines,
    reason = "the payload builder spells one wire field per branch; splitting it would scatter the upstream mapping"
)]
fn build_params(
    model: &Model,
    context: &Context,
    is_oauth_token: bool,
    options: &AnthropicStreamOptions,
) -> Result<Value, String> {
    let cache_control = get_cache_control(model, options.cache_retention, options.env.as_ref());
    let compat = get_anthropic_compat(model);
    let normalize_tool_call_id: &ToolCallIdNormalizer<'_> =
        &|id, _model, _source| normalize_anthropic_tool_call_id(id);
    let transformed_messages = transform_messages(
        context.messages.clone(),
        model,
        Some(&normalize_tool_call_id),
    );
    let normalize_tool_name = |name: &str| -> String {
        if is_oauth_token {
            to_claude_code_name(name)
        } else {
            name.to_owned()
        }
    };
    let tool_placement = split_deferred_tools(
        &Context {
            system_prompt: context.system_prompt.clone(),
            messages: transformed_messages.clone(),
            tools: context.tools.clone(),
        },
        compat.supports_tool_references,
        &normalize_tool_name,
    );
    let mut immediate_tools = tool_placement.immediate;
    let mut deferred_tools: Vec<Tool> = tool_placement
        .deferred
        .into_iter()
        .map(|(_, tool)| tool)
        .collect();
    if immediate_tools.is_empty() && !deferred_tools.is_empty() {
        immediate_tools = std::mem::take(&mut deferred_tools);
    }
    let deferred_tool_names: BTreeSet<String> = deferred_tools
        .iter()
        .map(|tool| normalize_tool_name(&tool.name))
        .collect();
    let converted = convert_messages(
        &transformed_messages,
        is_oauth_token,
        cache_control.as_ref(),
        compat.allow_empty_signature,
        &deferred_tool_names,
        &normalize_tool_name,
        compat.supports_mid_convo_effort.then_some(&model.provider),
    );
    let active_effort = options.effort.unwrap_or(AnthropicEffort::High);
    let beta_features = get_beta_features(model, context, is_oauth_token, options);

    let messages = if compat.supports_mid_convo_effort {
        insert_thinking_level_messages(&converted, active_effort)
    } else {
        converted.messages
    };

    let mut params = json!({
        "model": model.id,
        "messages": messages,
        "max_tokens": options.max_tokens.unwrap_or(model.max_tokens),
        "stream": true,
    });
    if !beta_features.is_empty() {
        params["betas"] = json!(beta_features);
    }

    // For OAuth tokens the Claude Code identity system block is required.
    let mut system_blocks: Vec<Value> = Vec::new();
    if is_oauth_token {
        system_blocks.push(with_cache_control(
            json!({ "type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude." }),
            cache_control.as_ref(),
        ));
        if let Some(system_prompt) = &context.system_prompt {
            system_blocks.push(with_cache_control(
                json!({ "type": "text", "text": system_prompt }),
                cache_control.as_ref(),
            ));
        }
    } else if let Some(system_prompt) = &context.system_prompt {
        system_blocks.push(with_cache_control(
            json!({ "type": "text", "text": system_prompt }),
            cache_control.as_ref(),
        ));
    }
    if !system_blocks.is_empty() {
        params["system"] = Value::Array(system_blocks);
    }

    // Temperature is incompatible with extended thinking and unsupported on
    // Claude Opus 4.7+.
    if options.temperature.is_some()
        && options.thinking_enabled != Some(true)
        && !compat.supports_mid_convo_effort
        && compat.supports_temperature
    {
        params["temperature"] = json!(options.temperature);
    }

    if !immediate_tools.is_empty() || !deferred_tools.is_empty() {
        let tools_cache_control = compat
            .supports_cache_control_on_tools
            .then_some(cache_control.as_ref())
            .flatten();
        let mut tools = convert_tools(
            &immediate_tools,
            is_oauth_token,
            compat.supports_eager_tool_input_streaming,
            compat.supports_strict_tools,
            tools_cache_control,
            false,
        )?;
        tools.extend(convert_tools(
            &deferred_tools,
            is_oauth_token,
            compat.supports_eager_tool_input_streaming,
            compat.supports_strict_tools,
            None,
            true,
        )?);
        params["tools"] = Value::Array(tools);
    }

    // Managed-effort models always use adaptive thinking so prefix
    // mismatches drop instead of surfacing as persistent 400 responses.
    if compat.supports_mid_convo_effort {
        params["thinking"] = json!({
            "type": "adaptive",
            "display": options.thinking_display.unwrap_or_default().as_str(),
            "block_binding": { "prefix_mismatch_behavior": "drop_block" },
        });
        params["output_config"] = json!({ "effort": "high" });
    } else if model.reasoning {
        if options.thinking_enabled == Some(true) {
            let display = options.thinking_display.unwrap_or_default().as_str();
            if model
                .compat
                .as_ref()
                .and_then(|compat| compat.force_adaptive_thinking)
                == Some(true)
            {
                // Adaptive thinking: Claude decides when and how much to
                // think.
                params["thinking"] = json!({ "type": "adaptive", "display": display });
                if let Some(effort) = options.effort {
                    params["output_config"] = json!({ "effort": effort.as_str() });
                }
            } else {
                params["thinking"] = json!({
                    "type": "enabled",
                    "budget_tokens": match options.thinking_budget_tokens {
                        Some(budget) if budget > 0 => budget,
                        _ => 1024,
                    },
                    "display": display,
                });
            }
        } else if options.thinking_enabled == Some(false) && thinking_level_off_supported(model) {
            params["thinking"] = json!({ "type": "disabled" });
        }
    }

    if let Some(user_id) = options
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("user_id"))
        .and_then(Value::as_str)
    {
        params["metadata"] = json!({ "user_id": user_id });
    }

    if let Some(tool_choice) = &options.tool_choice {
        params["tool_choice"] = match tool_choice {
            AnthropicToolChoice::Auto => json!({ "type": "auto" }),
            AnthropicToolChoice::Any => json!({ "type": "any" }),
            AnthropicToolChoice::None => json!({ "type": "none" }),
            AnthropicToolChoice::Tool { name } => json!({ "type": "tool", "name": name }),
        };
    }

    let allowed_fallback_models = model
        .compat
        .as_ref()
        .and_then(|compat| compat.allowed_fallback_models.as_ref())
        .filter(|fallbacks| !fallbacks.is_empty());
    if let Some(allowed_fallback_models) = allowed_fallback_models {
        params["fallbacks"] = json!(
            allowed_fallback_models
                .iter()
                .map(|fallback| json!({ "model": fallback.model }))
                .collect::<Vec<_>>()
        );
    }

    Ok(params)
}

fn with_cache_control(block: Value, cache_control: Option<&Value>) -> Value {
    let Some(cache_control) = cache_control else {
        return block;
    };
    let mut block = block;
    if let Some(object) = block.as_object_mut() {
        object.insert("cache_control".to_owned(), cache_control.clone());
    }
    block
}

/// Whether an explicit `thinking.type: "disabled"` is safe to send: models
/// without a `thinkingLevelMap` default to the off marker, and a map that
/// leaves `off` open (not `null`) admits it.
fn thinking_level_off_supported(model: &Model) -> bool {
    model
        .thinking_level_map
        .as_ref()
        .is_none_or(|map| map.get(&ModelThinkingLevel::Off) != Some(&None))
}

/// Normalize a tool-call id to Anthropic's required pattern and length,
/// upstream's `normalizeToolCallId`.
fn normalize_anthropic_tool_call_id(id: &str) -> String {
    let normalized: String = id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .collect();
    normalized.chars().take(64).collect()
}

fn convert_tool_result(
    message: &ToolResultMessage,
    is_oauth_token: bool,
    deferred_tool_names: &BTreeSet<String>,
    loaded_tool_names: &mut BTreeSet<String>,
    normalize_tool_name: &dyn Fn(&str) -> String,
) -> (Value, Vec<Value>) {
    let mut references: Vec<Value> = Vec::new();
    for name in message.added_tool_names.iter().flatten() {
        let normalized_name = normalize_tool_name(name);
        if !deferred_tool_names.contains(&normalized_name)
            || loaded_tool_names.contains(&normalized_name)
        {
            continue;
        }
        loaded_tool_names.insert(normalized_name);
        references.push(json!({
            "type": "tool_reference",
            "tool_name": if is_oauth_token {
                to_claude_code_name(name)
            } else {
                name.clone()
            },
        }));
    }
    let converted_content = convert_content_blocks(&message.content);
    // Anthropic rejects tool references mixed with ordinary tool-result
    // content; the ordinary content rides as sibling blocks.
    let has_references = !references.is_empty();
    let sibling_content = if !has_references {
        Vec::new()
    } else if converted_content.is_string() {
        vec![json!({ "type": "text", "text": converted_content })]
    } else {
        converted_content.as_array().cloned().unwrap_or_default()
    };
    let tool_result = json!({
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": if has_references {
            Value::Array(std::mem::take(&mut references))
        } else {
            converted_content
        },
        "is_error": message.is_error,
    });
    (tool_result, sibling_content)
}

/// The converted request messages plus the historical effort marker each
/// managed assistant message carries, upstream's `ConvertedAnthropicMessages`.
struct ConvertedAnthropicMessages {
    messages: Vec<Value>,
    assistant_levels: BTreeMap<usize, AnthropicEffort>,
}

#[expect(
    clippy::too_many_lines,
    reason = "each message role is one wire conversion; the run-splitting for tool results mirrors upstream"
)]
fn convert_messages(
    transformed_messages: &[Message],
    is_oauth_token: bool,
    cache_control: Option<&Value>,
    allow_empty_signature: bool,
    deferred_tool_names: &BTreeSet<String>,
    normalize_tool_name: &dyn Fn(&str) -> String,
    managed_provider: Option<&ProviderId>,
) -> ConvertedAnthropicMessages {
    let mut params: Vec<Value> = Vec::new();
    let mut assistant_levels = BTreeMap::new();
    let mut loaded_tool_names = BTreeSet::new();

    let mut index = 0;
    while index < transformed_messages.len() {
        match &transformed_messages[index] {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    if !text.trim().is_empty() {
                        params.push(json!({ "role": "user", "content": text }));
                    }
                }
                UserContent::Blocks(blocks) => {
                    let filtered_blocks: Vec<Value> = blocks
                        .iter()
                        .filter_map(|block| match block {
                            UserBlock::Text(text) => {
                                if text.text.trim().is_empty() {
                                    None
                                } else {
                                    Some(json!({ "type": "text", "text": text.text }))
                                }
                            }
                            UserBlock::Image(image) => Some(image_source_block(image)),
                        })
                        .collect();
                    if filtered_blocks.is_empty() {
                        index += 1;
                        continue;
                    }
                    params.push(json!({ "role": "user", "content": filtered_blocks }));
                }
            },
            Message::Assistant(assistant) => {
                let mut blocks: Vec<Value> = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantBlock::Text(text) => {
                            if text.text.trim().is_empty() {
                                continue;
                            }
                            blocks.push(json!({ "type": "text", "text": text.text }));
                        }
                        AssistantBlock::Thinking(thinking) => {
                            // Redacted thinking passes back as the opaque
                            // `redacted_thinking` payload.
                            if thinking.redacted.unwrap_or(false) {
                                blocks.push(json!({
                                    "type": "redacted_thinking",
                                    "data": thinking.thinking_signature.clone().unwrap_or_default(),
                                }));
                                continue;
                            }
                            let has_thinking_signature = thinking
                                .thinking_signature
                                .as_deref()
                                .is_some_and(|signature| !signature.trim().is_empty());
                            if thinking.thinking.trim().is_empty() && !has_thinking_signature {
                                continue;
                            }
                            if has_thinking_signature {
                                blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": thinking.thinking,
                                    "signature": thinking.thinking_signature,
                                }));
                            } else {
                                // A missing or empty signature (e.g. from an
                                // aborted stream) becomes plain text, unless
                                // the compat flag preserves the block.
                                blocks.push(if allow_empty_signature {
                                    json!({
                                        "type": "thinking",
                                        "thinking": thinking.thinking,
                                        "signature": "",
                                    })
                                } else {
                                    json!({ "type": "text", "text": thinking.thinking })
                                });
                            }
                        }
                        AssistantBlock::ToolCall(tool_call) => {
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": if is_oauth_token {
                                    to_claude_code_name(&tool_call.name)
                                } else {
                                    tool_call.name.clone()
                                },
                                "input": tool_call.arguments,
                            }));
                        }
                    }
                }
                if blocks.is_empty() {
                    index += 1;
                    continue;
                }
                let message_index = params.len();
                params.push(json!({ "role": "assistant", "content": blocks }));
                if let Some(managed_provider) = managed_provider
                    && assistant.api == Api::from("anthropic-messages")
                    && assistant.provider == *managed_provider
                    && let Some(level) = assistant
                        .provider_thinking_level
                        .as_deref()
                        .and_then(|level| AnthropicEffort::try_from(level).ok())
                {
                    assistant_levels.insert(message_index, level);
                }
            }
            Message::ToolResult(_) => {
                // Consecutive tool-result messages share one user turn,
                // needed for the z.ai Anthropic endpoint.
                let mut tool_results: Vec<Value> = Vec::new();
                let mut sibling_content: Vec<Value> = Vec::new();
                let mut run = index;
                while run < transformed_messages.len()
                    && let Message::ToolResult(tool_result) = &transformed_messages[run]
                {
                    let (tool_result_block, siblings) = convert_tool_result(
                        tool_result,
                        is_oauth_token,
                        deferred_tool_names,
                        &mut loaded_tool_names,
                        normalize_tool_name,
                    );
                    tool_results.push(tool_result_block);
                    sibling_content.extend(siblings);
                    run += 1;
                }
                // Skip the messages already processed.
                index = run - 1;

                // Displaced reference-bearing results follow every
                // tool_result block.
                let mut content = tool_results;
                content.extend(sibling_content);
                params.push(json!({ "role": "user", "content": content }));
            }
        }
        index += 1;
    }

    // Add cache_control to the last user message to cache the conversation
    // history.
    if let Some(cache_control) = cache_control
        && let Some(last_message) = params.last_mut()
        && last_message.get("role").and_then(Value::as_str) == Some("user")
    {
        match last_message.get_mut("content") {
            Some(Value::Array(blocks)) => {
                if let Some(last_block) = blocks.last_mut() {
                    let block_kind = last_block.get("type").and_then(Value::as_str);
                    if matches!(block_kind, Some("text" | "image" | "tool_result"))
                        && let Some(object) = last_block.as_object_mut()
                    {
                        object.insert("cache_control".to_owned(), cache_control.clone());
                    }
                }
            }
            Some(Value::String(text)) => {
                last_message["content"] = json!([{
                    "type": "text",
                    "text": text.clone(),
                    "cache_control": cache_control,
                }]);
            }
            _ => {}
        }
    }

    ConvertedAnthropicMessages {
        messages: params,
        assistant_levels,
    }
}

fn insert_thinking_level_messages(
    converted: &ConvertedAnthropicMessages,
    active_effort: AnthropicEffort,
) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();
    for (index, message) in converted.messages.iter().enumerate() {
        if let Some(historical_effort) = converted.assistant_levels.get(&index) {
            messages.push(json!({
                "role": "system",
                "content": [],
                "output_config": { "effort": historical_effort.as_str() },
            }));
        }
        messages.push(message.clone());
    }
    messages.push(json!({
        "role": "system",
        "content": [],
        "output_config": { "effort": active_effort.as_str() },
    }));
    messages
}

fn should_use_fine_grained_tool_streaming_beta(model: &Model, context: &Context) -> bool {
    context
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && !get_anthropic_compat(model).supports_eager_tool_input_streaming
}

#[expect(
    clippy::fn_params_excessive_bools,
    reason = "the flags mirror upstream's convertTools argument list verbatim"
)]
fn convert_tools(
    tools: &[Tool],
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: Option<&Value>,
    defer_loading: bool,
) -> Result<Vec<Value>, String> {
    let mut converted: Vec<Value> = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let strict = resolve_json_schema_strict_sampling(tool, supports_strict_tools)?;
        let parameters = get_json_schema_tool_parameters(tool, strict).map_err(|error| error.0)?;
        let legacy_input_schema = json!({
            "type": "object",
            "properties": parameters.get("properties").cloned().unwrap_or_else(|| json!({})),
            "required": parameters.get("required").cloned().unwrap_or_else(|| json!([])),
        });
        let input_schema = if strict == Some(true) {
            // The strict schema keeps its extra keys (title, ...) while the
            // legacy object's own keys win the merge.
            let mut merged = parameters;
            merged["type"] = json!("object");
            merged["properties"] = legacy_input_schema["properties"].clone();
            merged["required"] = legacy_input_schema["required"].clone();
            merged
        } else {
            legacy_input_schema
        };

        let mut block = json!({
            "name": if is_oauth_token {
                to_claude_code_name(&tool.name)
            } else {
                tool.name.clone()
            },
            "description": tool.description,
            "input_schema": input_schema,
        });
        if supports_eager_tool_input_streaming {
            block["eager_input_streaming"] = json!(true);
        }
        if strict == Some(true) {
            block["strict"] = json!(true);
        }
        if defer_loading {
            block["defer_loading"] = json!(true);
        }
        if index == tools.len() - 1
            && let Some(cache_control) = cache_control
        {
            block["cache_control"] = cache_control.clone();
        }
        converted.push(block);
    }
    Ok(converted)
}

/// The reason a stream stopped, upstream's `mapStopReason`; the refusal
/// explanation rides the stream error's message. Unknown reasons fail the
/// stream so the API can grow its vocabulary visibly.
fn map_stop_reason(
    reason: &str,
    stop_details: Option<&StopDetails>,
) -> Result<(StopReason, Option<String>), String> {
    match reason {
        "end_turn" | "pause_turn" | "stop_sequence" => Ok((StopReason::Stop, None)),
        "max_tokens" => Ok((StopReason::Length, None)),
        "tool_use" => Ok((StopReason::ToolUse, None)),
        "refusal" => Ok((
            StopReason::Error,
            Some(
                stop_details
                    .and_then(|details| details.explanation.clone())
                    .unwrap_or_else(|| "The model refused to complete the request".to_owned()),
            ),
        )),
        // Content flagged by safety filters.
        "sensitive" => Ok((
            StopReason::Error,
            Some("Provider stopped with: sensitive".to_owned()),
        )),
        other => Err(format!("Unhandled stop reason: {other}")),
    }
}

/// The Anthropic Messages [`crate::types::ProviderStreams`], upstream's
/// module-level `stream`/`streamSimple` exports behind the uniform dispatch.
#[derive(Debug, Default)]
pub struct AnthropicStreams;

crate::api::adapter_belt::impl_provider_streams!(AnthropicStreams, AnthropicStreamOptions);
