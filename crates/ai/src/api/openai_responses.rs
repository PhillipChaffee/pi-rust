//! The OpenAI Responses wire API, ported from
//! `packages/ai/src/api/openai-responses.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//!
//! - The `openai` SDK client collapses onto the [`crate::http::HttpClient`]
//!   seam. The module assembles the wire request the pinned SDK sent — POST
//!   `{baseUrl}/responses`, `Authorization: Bearer {apiKey}`, the pi user
//!   agent, Copilot dynamic headers, and the session-affinity headers — and
//!   executes it through the injected or default client. Upstream's
//!   `options.fetch` injection has no separate shape: an injected
//!   [`crate::http::HttpClient`] replaces the transport, and request shaping
//!   always runs.
//! - The SDK's response stream is the shared [`crate::http::sse`] decoder
//!   over the seam body: each `data:` frame is one `ResponseStreamEvent`
//!   JSON object and the `data: [DONE]` sentinel ends the stream. The event
//!   fields the wire types loosely are read straight off the JSON in
//!   [`crate::api::openai_responses_shared`].
//! - `sanitizeSurrogates` disappears statically: a Rust [`String`] cannot
//!   hold the unpaired surrogates it stripped, and `serde_json` rejects
//!   lone-surrogate escapes when reading the wire.
//! - `streamSimple`'s synchronous auth throw becomes the setup-error stream:
//!   the Rust stream contract returns a stream synchronously, so a missing
//!   key surfaces as its `error` event.

use std::collections::BTreeSet;
use std::sync::Arc;

use bytes::Bytes;
use serde_json::{Map, Value, json};

use crate::api::constrained_sampling::create_grammar_tool_input_properties;
use crate::api::github_copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::openai_responses_shared::{
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, OpenAiResponsesStreamOptions,
    ResponsesDeferredToolsMode, convert_responses_messages, convert_responses_tools, pricing_hook,
    process_responses_stream,
};
use crate::api::simple_options::build_base_options;
use crate::http::client::{HttpMethod, HttpRequest, HttpResponse};
use crate::models::clamp_thinking_level;
use crate::types::{
    AssistantMessage, AssistantMessageEvent, Context, Model, ProviderEnv, ProviderHeaders,
    ProviderId, SessionAffinityFormat, SimpleStreamOptions, StopReason, StreamOptions,
    ThinkingLevel, ThinkingLevelMap, Tool, ToolChoice, TransportOptions,
};
use crate::utils::deferred_tools::split_deferred_tools;
use crate::utils::error_body::{
    ErrorBody, SdkError, format_provider_error, normalize_provider_error,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The providers whose tool-call item ids carry OpenAI Responses pairing
/// history, upstream's `OPENAI_TOOL_CALL_PROVIDERS`.
fn openai_tool_call_providers() -> BTreeSet<String> {
    ["openai", "openai-codex", "opencode"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// OpenAI Responses rejects `max_output_tokens` below 16. See
/// <https://github.com/earendil-works/pi/issues/6265>.
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

use crate::api::request_seam::{
    execute_checked_response, fire_response_hook, get_client_api_key, setup_error_stream,
    spawned_stream,
};

// ---------------------------------------------------------------------------
// Compat
// ---------------------------------------------------------------------------

/// The resolved compat flags the adapter acts on, upstream's
/// `Required<OpenAIResponsesCompat>`: every flag carries its default after
/// the `model.compat` override.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the flag belt mirrors upstream's compat object; each flag is an independent provider admission"
)]
#[derive(Clone, Copy, Debug)]
struct ResponsesCompat {
    /// The `Required<OpenAIResponsesCompat>` belt mirrors the compat object's
    /// full field set; this flag rides for shape parity and is read by the
    /// shared message converter through `model.compat` instead.
    #[expect(
        dead_code,
        reason = "the compat belt mirrors upstream's Required<OpenAIResponsesCompat>; the shared message converter reads model.compat directly"
    )]
    supports_developer_role: bool,
    session_affinity_format: SessionAffinityFormat,
    supports_long_cache_retention: bool,
    supports_strict_mode: bool,
    supports_openai_grammar_tools: bool,
    supports_additional_tools: bool,
    supports_tool_search: bool,
    supports_explicit_prompt_cache_mode: bool,
    supports_max_output_tokens: bool,
}

/// The session-affinity header format a model defaults to, upstream's
/// `detectSessionAffinityFormat`: OpenRouter endpoints carry
/// `x-session-id`, everything else the OpenAI pair.
fn detect_session_affinity_format(model: &Model) -> SessionAffinityFormat {
    if model.provider == ProviderId::from("openrouter") || model.base_url.contains("openrouter.ai")
    {
        SessionAffinityFormat::Openrouter
    } else {
        SessionAffinityFormat::Openai
    }
}

/// Resolve the compat settings for a model: the defaults overridable field
/// by field with `model.compat`, upstream's `getCompat`.
fn get_compat(model: &Model) -> ResponsesCompat {
    let compat = model.compat.as_ref();
    ResponsesCompat {
        supports_developer_role: compat
            .and_then(|compat| compat.supports_developer_role)
            .unwrap_or(true),
        session_affinity_format: compat
            .and_then(|compat| compat.session_affinity_format)
            .unwrap_or_else(|| detect_session_affinity_format(model)),
        supports_long_cache_retention: compat
            .and_then(|compat| compat.supports_long_cache_retention)
            .unwrap_or(true),
        supports_strict_mode: compat
            .and_then(|compat| compat.supports_strict_mode)
            .unwrap_or(false),
        supports_openai_grammar_tools: compat
            .and_then(|compat| compat.supports_openai_grammar_tools)
            .unwrap_or(false),
        supports_additional_tools: compat
            .and_then(|compat| compat.supports_additional_tools)
            .unwrap_or(false),
        supports_tool_search: compat
            .and_then(|compat| compat.supports_tool_search)
            .unwrap_or(false),
        supports_explicit_prompt_cache_mode: compat
            .and_then(|compat| compat.supports_explicit_prompt_cache_mode)
            .unwrap_or(false),
        supports_max_output_tokens: compat
            .and_then(|compat| compat.supports_max_output_tokens)
            .unwrap_or(true),
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// The `reasoning.summary` request value, upstream's
/// `"auto" | "detailed" | "concise" | null`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningSummary {
    /// `"auto"`: the provider chooses.
    Auto,
    /// `"detailed"` summaries.
    Detailed,
    /// `"concise"` summaries.
    Concise,
}

impl ReasoningSummary {
    /// The value as the wire spells it.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Detailed => "detailed",
            Self::Concise => "concise",
        }
    }
}

/// The OpenAI Responses options, upstream's `OpenAIResponsesOptions extends
/// StreamOptions`: the base fields this adapter reads plus the reasoning,
/// service-tier, and tool-choice extras.
///
/// Base fields the adapter never reads (`transport`,
/// `websocketConnectTimeoutMs`) are omitted, matching the sibling wire APIs'
/// option structs.
#[derive(Clone, Debug, Default)]
pub struct OpenAiResponsesOptions {
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
    pub sampling_params: Option<std::collections::BTreeMap<String, Value>>,
    /// Maximum output tokens. The wire floor is 16.
    pub max_tokens: Option<u64>,
    /// Prompt cache retention preference. Default: `"short"`.
    pub cache_retention: Option<crate::types::CacheRetention>,
    /// Session identifier for session-based caching and routing.
    pub session_id: Option<String>,
    /// Optional request metadata; the adapter does not read it.
    pub metadata: Option<std::collections::BTreeMap<String, Value>>,
    /// The reasoning effort, upstream's
    /// `"minimal" | "low" | "medium" | "high" | "xhigh" | "max"`.
    pub reasoning_effort: Option<ThinkingLevel>,
    /// The reasoning summary detail, upstream's
    /// `"auto" | "detailed" | "concise" | null`.
    pub reasoning_summary: Option<ReasoningSummary>,
    /// The request's `service_tier` (`"auto"`, `"default"`, `"flex"`, or
    /// `"priority"`), priced into the final usage.
    pub service_tier: Option<String>,
    /// The Responses `tool_choice` value, forwarded verbatim: `"none"`,
    /// `"auto"`, `"required"`, or a provider tool-choice object.
    pub tool_choice: Option<Value>,
}

crate::api::adapter_belt::impl_stream_options_from!(OpenAiResponsesOptions from options {
    sampling_params: options.sampling_params,
    reasoning_effort: None,
    reasoning_summary: None,
    service_tier: None,
    tool_choice: None,
});
// ---------------------------------------------------------------------------
// Cache retention
// ---------------------------------------------------------------------------

/// Resolve the cache retention preference, upstream's `resolveCacheRetention`:
/// the request's value, else `PI_CACHE_RETENTION`, else `short`.
fn resolve_cache_retention(
    cache_retention: Option<crate::types::CacheRetention>,
    env: Option<&ProviderEnv>,
) -> crate::types::CacheRetention {
    if let Some(cache_retention) = cache_retention {
        return cache_retention;
    }
    if get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
        return crate::types::CacheRetention::Long;
    }
    crate::types::CacheRetention::Short
}

/// The `prompt_cache_retention` a request carries, upstream's
/// `getPromptCacheRetention`: `"24h"` under long retention on models the
/// compat flags admit and that do not take the explicit cache mode.
fn get_prompt_cache_retention(
    compat: &ResponsesCompat,
    cache_retention: crate::types::CacheRetention,
) -> Option<&'static str> {
    (cache_retention == crate::types::CacheRetention::Long
        && compat.supports_long_cache_retention
        && !compat.supports_explicit_prompt_cache_mode)
        .then_some("24h")
}

/// The `prompt_cache_options` a request carries, upstream's
/// `getPromptCacheOptions`: `{"mode":"explicit"}` under no-cache retention
/// and `{"ttl":"30m"}` under long retention on capable models.
fn get_prompt_cache_options(
    compat: &ResponsesCompat,
    cache_retention: crate::types::CacheRetention,
) -> Option<Map<String, Value>> {
    if !compat.supports_explicit_prompt_cache_mode {
        return None;
    }
    let mut options = Map::new();
    match cache_retention {
        crate::types::CacheRetention::None => {
            options.insert("mode".to_owned(), json!("explicit"));
        }
        crate::types::CacheRetention::Long if compat.supports_long_cache_retention => {
            options.insert("ttl".to_owned(), json!("30m"));
        }
        _ => return None,
    }
    Some(options)
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

/// Assemble the request headers a stream sends, the port of upstream's
/// `createClient`: the pi user agent, the model's headers, Copilot's dynamic
/// headers, the session-affinity headers when a session id rides, and the
/// caller's headers merged last so they can override defaults.
///
/// The Responses session-affinity format sends `x-session-id` (OpenRouter),
/// or `session_id` plus `x-client-request-id` (OpenAI), or only
/// `x-client-request-id` (OpenAI without the `session_id` header).
fn build_request_headers(
    model: &Model,
    context: &Context,
    compat: &ResponsesCompat,
    session_id: Option<&str>,
    api_key: &str,
    options_headers: Option<&ProviderHeaders>,
) -> Vec<(String, String)> {
    // The credential header participates in the caller-header merge, the
    // pinned SDK's `defaultHeaders` behavior, like the completions adapter's.
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
    // The pinned OpenAI SDK's JSON content type, overridable like the
    // completions adapter's.
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

    if let Some(session_id) = session_id {
        match compat.session_affinity_format {
            SessionAffinityFormat::Openrouter => {
                headers.push(("x-session-id".to_owned(), session_id.to_owned()));
            }
            SessionAffinityFormat::Openai => {
                headers.push(("session_id".to_owned(), session_id.to_owned()));
                headers.push(("x-client-request-id".to_owned(), session_id.to_owned()));
            }
            SessionAffinityFormat::OpenaiNosession => {
                headers.push(("x-client-request-id".to_owned(), session_id.to_owned()));
            }
        }
    }

    // Merge options headers last so they can override defaults
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

/// The Responses endpoint, the path the pinned SDK appends to `baseUrl`.
fn responses_url(model: &Model) -> String {
    format!("{}/responses", model.base_url.trim_end_matches('/'))
}

/// The pi level a mapped model level names; `None` for the off state.
pub(crate) const fn pi_thinking_level(
    level: crate::types::ModelThinkingLevel,
) -> Option<ThinkingLevel> {
    match level {
        crate::types::ModelThinkingLevel::Off => None,
        crate::types::ModelThinkingLevel::Minimal => Some(ThinkingLevel::Minimal),
        crate::types::ModelThinkingLevel::Low => Some(ThinkingLevel::Low),
        crate::types::ModelThinkingLevel::Medium => Some(ThinkingLevel::Medium),
        crate::types::ModelThinkingLevel::High => Some(ThinkingLevel::High),
        crate::types::ModelThinkingLevel::Xhigh => Some(ThinkingLevel::Xhigh),
        crate::types::ModelThinkingLevel::Max => Some(ThinkingLevel::Max),
    }
}

/// The pi level a budget-based provider acts on, in the model-level spelling
/// the `thinkingLevelMap` keys use.
pub(crate) const fn thinking_model_level(level: ThinkingLevel) -> crate::types::ModelThinkingLevel {
    match level {
        ThinkingLevel::Minimal => crate::types::ModelThinkingLevel::Minimal,
        ThinkingLevel::Low => crate::types::ModelThinkingLevel::Low,
        ThinkingLevel::Medium => crate::types::ModelThinkingLevel::Medium,
        ThinkingLevel::High => crate::types::ModelThinkingLevel::High,
        ThinkingLevel::Xhigh => crate::types::ModelThinkingLevel::Xhigh,
        ThinkingLevel::Max => crate::types::ModelThinkingLevel::Max,
    }
}

/// The level as the wire spells it; a module-local free function because the
/// sibling already owns the inherent `as_str`.
pub(crate) const fn thinking_level_str(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
}

/// The model's mapped effort for a pi level, upstream's
/// `model.thinkingLevelMap?.[effort] ?? effort`: the map's entry when it
/// names one, else the level itself. A `null` entry — the wire's
/// unsupported-level marker — falls through to the level like `??` does.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the Some wrapper keeps the call sites' unwrap_or_else chains reading like upstream's ?? expression"
)]
pub(crate) fn mapped_effort(
    level_map: Option<&ThinkingLevelMap>,
    level: ThinkingLevel,
) -> Option<Value> {
    let mapped = level_map
        .and_then(|map| map.get(&thinking_model_level(level)))
        .and_then(|entry| entry.as_deref());
    Some(mapped.map_or_else(|| json!(thinking_level_str(level)), |mapped| json!(mapped)))
}

/// Build the request payload, upstream's `buildParams`. The wire's
/// `undefined` fields are absent, so only present fields insert.
///
/// # Errors
/// The message- and tool-conversion rejections the input list hits.
#[expect(
    clippy::too_many_lines,
    reason = "the request fields mirror upstream's buildParams object literal and its conditional assignments in wire order"
)]
fn build_params(
    model: &Model,
    context: &Context,
    options: &OpenAiResponsesOptions,
    compat: &ResponsesCompat,
    grammar_tool_input_properties: &std::collections::BTreeMap<String, String>,
) -> Result<Map<String, Value>, String> {
    let deferred_tools_mode = if compat.supports_additional_tools {
        Some(ResponsesDeferredToolsMode::AdditionalTools)
    } else if compat.supports_tool_search {
        Some(ResponsesDeferredToolsMode::ToolSearch)
    } else {
        None
    };
    let normalize_tool_name = |name: &str| -> String { name.to_owned() };
    let tool_placement =
        split_deferred_tools(context, deferred_tools_mode.is_some(), &normalize_tool_name);
    let deferred_tools: std::collections::BTreeMap<String, Tool> =
        tool_placement.deferred.iter().cloned().collect();
    let messages = convert_responses_messages(
        model,
        context,
        &openai_tool_call_providers(),
        Some(&ConvertResponsesMessagesOptions {
            grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
            deferred_tools: Some(deferred_tools),
            deferred_tools_mode,
            tool_options: Some(ConvertResponsesToolsOptions {
                supports_strict_mode: Some(compat.supports_strict_mode),
                supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
                ..ConvertResponsesToolsOptions::default()
            }),
            ..ConvertResponsesMessagesOptions::default()
        }),
    )?;

    let cache_retention = resolve_cache_retention(options.cache_retention, options.env.as_ref());
    let mut params = Map::new();
    params.insert("model".to_owned(), json!(model.id));
    params.insert("input".to_owned(), Value::Array(messages));
    params.insert("stream".to_owned(), json!(true));
    if cache_retention != crate::types::CacheRetention::None
        && let Some(key) = clamp_openai_prompt_cache_key(options.session_id.as_deref())
    {
        params.insert("prompt_cache_key".to_owned(), json!(key));
    }
    if get_prompt_cache_retention(compat, cache_retention).is_some() {
        params.insert("prompt_cache_retention".to_owned(), json!("24h"));
    }
    if let Some(cache_options) = get_prompt_cache_options(compat, cache_retention) {
        params.insert(
            "prompt_cache_options".to_owned(),
            Value::Object(cache_options),
        );
    }
    params.insert("store".to_owned(), json!(false));

    if let Some(max_tokens) = options.max_tokens
        && compat.supports_max_output_tokens
    {
        params.insert(
            "max_output_tokens".to_owned(),
            json!(max_tokens.max(OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)),
        );
    }

    if let Some(temperature) = options.temperature {
        params.insert("temperature".to_owned(), json!(temperature));
    }

    if let Some(service_tier) = &options.service_tier {
        params.insert("service_tier".to_owned(), json!(service_tier));
    }

    if !tool_placement.immediate.is_empty() {
        params.insert(
            "tools".to_owned(),
            Value::Array(convert_responses_tools(
                &tool_placement.immediate,
                Some(&ConvertResponsesToolsOptions {
                    supports_strict_mode: Some(compat.supports_strict_mode),
                    supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
                    ..ConvertResponsesToolsOptions::default()
                }),
            )?),
        );
    }

    if let Some(tool_choice) = &options.tool_choice {
        params.insert("tool_choice".to_owned(), tool_choice.clone());
    }

    if model.reasoning {
        if options.reasoning_effort.is_some() || options.reasoning_summary.is_some() {
            let effort = options.reasoning_effort.map_or_else(
                || json!("medium"),
                |level| {
                    mapped_effort(model.thinking_level_map.as_ref(), level)
                        .unwrap_or_else(|| json!(thinking_level_str(level)))
                },
            );
            let summary = options
                .reasoning_summary
                .map_or("auto", ReasoningSummary::as_str);
            params.insert(
                "reasoning".to_owned(),
                json!({ "effort": effort, "summary": summary }),
            );
            params.insert("include".to_owned(), json!(["reasoning.encrypted_content"]));
        } else if model.provider != ProviderId::from("github-copilot")
            && model
                .thinking_level_map
                .as_ref()
                .and_then(|map| map.get(&crate::types::ModelThinkingLevel::Off))
                != Some(&None)
        {
            let effort = model
                .thinking_level_map
                .as_ref()
                .and_then(|map| map.get(&crate::types::ModelThinkingLevel::Off))
                .cloned()
                .flatten()
                .unwrap_or_else(|| "none".to_owned());
            params.insert("reasoning".to_owned(), json!({ "effort": effort }));
        }
        if model.provider == ProviderId::from("xai") {
            params.insert("include".to_owned(), json!(["reasoning.encrypted_content"]));
        }
    }

    // Last so custom keys override the named request fields.
    if let Some(sampling_params) = &options.sampling_params {
        for (key, value) in sampling_params {
            params.insert(key.clone(), value.clone());
        }
    }

    Ok(params)
}

/// The display prefix the provider's errors carry, upstream's catch-block
/// argument: the well-known provider spells `"OpenAI"`, others use their id.
fn provider_error_prefix(model: &Model) -> String {
    if model.provider == ProviderId::from("openai") {
        "OpenAI API error".to_owned()
    } else {
        format!("{} API error", model.provider.0)
    }
}

/// Execute the stream request with retries and fire the response hooks, the
/// seam-side port of upstream's `createClient` +
/// `retryProviderRequest(create(...).asResponse())`. A non-2xx body is
/// parsed before it is discarded so the failure message can surface it the
/// way the pinned SDK folds it into its error.
async fn dispatch_stream_request(
    model: &Model,
    context: &Context,
    compat: &ResponsesCompat,
    options: &OpenAiResponsesOptions,
    api_key: &str,
    session_id: Option<&str>,
    payload: Map<String, Value>,
) -> Result<HttpResponse, String> {
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
        url: responses_url(model),
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
    // can compose the way the pinned SDK's error object does.
    let parsed_error_body: Arc<std::sync::Mutex<Option<Value>>> = Arc::default();
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
            let sdk_error = SdkError {
                message: error.message.clone(),
                status: error.status,
                error: parsed_body.map(ErrorBody::Parsed),
                ..SdkError::default()
            };
            return Err(format_provider_error(
                &normalize_provider_error(sdk_error),
                Some(&provider_error_prefix(model)),
            ));
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

/// Stream an assistant response over the OpenAI Responses wire, upstream's
/// `stream`. The stream returns live; request setup, model, and runtime
/// failures arrive as its `error` event.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&OpenAiResponsesOptions>,
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
                run_stream(&model, &context, &options, output, &forward).await
            })
        },
    )
}

/// Run one stream to completion: resolve the credential, build and dispatch
/// the request, process the events, and settle the final message.
async fn run_stream(
    model: &Model,
    context: &Context,
    options: &OpenAiResponsesOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let api_key = get_client_api_key(
        &model.provider,
        options.api_key.as_deref(),
        options.headers.as_ref(),
    )?;
    let compat = get_compat(model);
    let grammar_tool_input_properties = create_grammar_tool_input_properties(
        context.tools.as_deref(),
        compat.supports_openai_grammar_tools,
    );
    let payload = build_params(
        model,
        context,
        options,
        &compat,
        &grammar_tool_input_properties,
    )?;
    let cache_retention = resolve_cache_retention(options.cache_retention, options.env.as_ref());
    let cache_session_id = (cache_retention != crate::types::CacheRetention::None)
        .then(|| options.session_id.clone())
        .flatten();
    let pricing_model = model.clone();
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

    let stream_options = OpenAiResponsesStreamOptions {
        service_tier: options.service_tier.clone(),
        grammar_tool_input_properties: Some(grammar_tool_input_properties),
        apply_service_tier_pricing: Some(pricing_hook(pricing_model)),
        ..OpenAiResponsesStreamOptions::default()
    };
    process_responses_stream(response, output, events, model, Some(&stream_options)).await?;

    if options.transport_options.signal().is_cancelled() {
        return Err("Request was aborted".to_owned());
    }
    if output.stop_reason == StopReason::Pending {
        return Err("OpenAI Responses stream ended without a stop reason".to_owned());
    }
    if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
        return Err(output
            .error_message
            .clone()
            .unwrap_or_else(|| "An unknown error occurred".to_owned()));
    }

    events.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output.clone(),
    });
    Ok(())
}

/// The fresh accumulator a stream starts from, upstream's `output`: zeroed
/// usage and `stopReason: "pending"`.
fn initial_output(model: &Model) -> AssistantMessage {
    crate::api::request_seam::initial_output(model, model.api.clone(), None)
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

    let mut base = OpenAiResponsesOptions::from(build_base_options(
        model,
        context,
        options,
        options.and_then(|options| options.api_key.as_deref()),
    ));
    base.tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(|choice| match choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::None => json!("none"),
        });
    base.reasoning_effort = options
        .and_then(|options| options.reasoning)
        .map(|reasoning| clamp_thinking_level(model, thinking_model_level(reasoning)))
        .and_then(pi_thinking_level);
    stream(model, context, Some(&base))
}

/// The OpenAI Responses [`crate::types::ProviderStreams`], upstream's
/// module-level `stream`/`streamSimple` exports behind the uniform dispatch.
#[derive(Debug, Default)]
pub struct OpenAiResponsesStreams;

crate::api::adapter_belt::impl_provider_streams!(OpenAiResponsesStreams, OpenAiResponsesOptions);
