//! The Google Generative AI (Gemini) wire API, ported from
//! `packages/ai/src/api/google-generative-ai.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//! - The `@google/genai` SDK surface collapses to one raw request: upstream's
//!   only SDK call is `client.models.generateContentStream(params)`, which is
//!   `POST {baseUrl}/models/{model}:streamGenerateContent?alt=sse` with
//!   `x-goog-api-key` auth and a JSON body of the SDK-style params flattened
//!   (`config` becomes top-level `generationConfig`; `systemInstruction`,
//!   `tools`, and `toolConfig` hoist to the body root). `onPayload` sees the
//!   SDK-style params, the hook's replacement included.
//! - The SDK's default headers drop with it: `x-goog-api-client` (SDK
//!   telemetry) is not reproduced, and `User-Agent` is pi's own value, which
//!   upstream overrides onto the SDK default anyway. `x-goog-api-key` keeps
//!   the SDK's only-when-absent rule, so a caller header can override it.
//! - The SDK's env fallbacks (`GOOGLE_API_KEY`/`GEMINI_API_KEY`,
//!   `GOOGLE_GENAI_USE_VERTEXAI`) never fire upstream because pi always passes
//!   an explicit key; the port requires the key outright.
//! - Upstream's `options.fetch` guard ("Custom fetch is not supported") has
//!   no counterpart: the transport seam is the supported injection point.
//! - Surrogate sanitization is statically upheld: Rust strings are valid
//!   UTF-8, so `sanitizeSurrogates` has no work.
//! - The streamed part index scratch field and its catch-path cleanup
//!   vanish; Rust owns blocks in the accumulator directly.
//! - The SDK's pre-stream error probe inspected the first decoded network
//!   chunk whole; the port probes the first SSE `data:` event, the same
//!   failure surface on every frame shape the API emits.
//! - A `usageMetadata` reporting more cached tokens than prompt tokens
//!   floors `usage.input` at zero (upstream's arithmetic can go negative;
//!   `Usage.input` is unsigned here).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use serde_json::{Value, json};

use crate::api::google_shared::{
    convert_messages, convert_tools, is_thinking_part, map_stop_reason,
    resolve_google_function_calling_mode, resolve_google_thinking_level, retain_thought_signature,
    supports_google_strict_tool_sampling, ResolvedGoogleThinkingLevel,
};
use crate::api::simple_options::build_base_options;
use crate::http::client::{HttpError, HttpMethod, HttpRequest, read_body_text};
use crate::types::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, Context, Model, ModelThinkingLevel,
    ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions, TextContent,
    ThinkingBudgets, ThinkingContent, ToolCall,
};
use crate::utils::error_body::safe_json_stringify;
use crate::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use crate::utils::headers::{headers_to_record, provider_headers_to_record};
use crate::utils::json_parse::parse_json_with_repair;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_retry::{
    ProviderRequestError, ProviderRetryOptions, retry_provider_request,
};

/// The model id the wire names when no custom base URL is set: the SDK's
/// default host plus its default `v1beta` version segment.
const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Counter for generating unique tool call IDs, upstream's module-level
/// `toolCallCounter`.
static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The adapter-facing options, upstream's `GoogleOptions extends
/// StreamOptions` plus the thinking control `streamSimple` resolves
/// upstream-side.
#[derive(Clone, Debug, Default)]
pub struct GoogleStreamOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks.
    pub transport_options: crate::types::TransportOptions,
    /// The API key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this logical request.
    pub telemetry_context: Option<pi_telemetry::TelemetryHandle>,
    /// Provider-scoped environment values; these take precedence over the
    /// process environment.
    pub env: Option<crate::types::ProviderEnv>,
    /// Custom HTTP headers merged with provider defaults.
    pub headers: Option<crate::types::ProviderHeaders>,
    /// HTTP request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts for client-side retries.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait for a retry when the server
    /// requests a long wait.
    pub max_retry_delay_ms: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Arbitrary sampling parameters merged into the request body as-is.
    pub sampling_params: Option<std::collections::BTreeMap<String, Value>>,
    /// Maximum output tokens.
    pub max_tokens: Option<u64>,
    /// Preferred transport; SSE is the only transport this adapter speaks.
    pub transport: Option<crate::types::Transport>,
    /// Prompt cache retention preference. Default: `"short"`.
    pub cache_retention: Option<crate::types::CacheRetention>,
    /// Optional session identifier for providers that support session-based
    /// caching.
    pub session_id: Option<String>,
    /// WebSocket connect timeout in milliseconds.
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Optional metadata to include in API requests.
    pub metadata: Option<std::collections::BTreeMap<String, Value>>,
    /// The tool choice, upstream's `"auto" | "none" | "any"`.
    pub tool_choice: Option<String>,
    /// The thinking control, upstream's
    /// `thinking?: { enabled, budgetTokens?, level? }`.
    pub thinking: Option<GoogleThinkingControl>,
}

/// The thinking control a caller can pin for the full-fidelity `stream`
/// entry, upstream's `GoogleOptions["thinking"]`.
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

impl From<StreamOptions> for GoogleStreamOptions {
    fn from(options: StreamOptions) -> Self {
        let StreamOptions {
            transport_options,
            api_key,
            telemetry_context,
            env,
            headers,
            timeout_ms,
            max_retries,
            max_retry_delay_ms,
            temperature,
            sampling_params,
            max_tokens,
            transport,
            cache_retention,
            session_id,
            websocket_connect_timeout_ms,
            metadata,
        } = options;
        Self {
            transport_options,
            api_key,
            telemetry_context,
            env,
            headers,
            timeout_ms,
            max_retries,
            max_retry_delay_ms,
            temperature,
            sampling_params,
            max_tokens,
            transport,
            cache_retention,
            session_id,
            websocket_connect_timeout_ms,
            metadata,
            tool_choice: None,
            thinking: None,
        }
    }
}

/// The Google Generative AI streams, upstream's `googleGenerativeAIApi()`.
#[derive(Debug, Default)]
pub struct GoogleStreams;

impl ProviderStreams for GoogleStreams {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let options = options.cloned().map(GoogleStreamOptions::from);
        stream(model, context, options.as_ref())
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        stream_simple(model, context, options)
    }
}

/// Stream an assistant response, upstream's `stream` export.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&GoogleStreamOptions>,
) -> AssistantMessageEventStream {
    let events = assistant_message_event_stream();
    let forward = events.clone();
    let model = model.clone();
    let context = context.clone();
    let options = options.cloned();
    tokio::spawn(async move {
        let options = options.unwrap_or_default();
        let mut output = initial_output(&model);
        match run_stream(&model, &context, &options, &mut output, &forward).await {
            Ok(()) => {
                // The done event already settled the final result.
                forward.end(None);
            }
            Err(message) => {
                let signal = options.transport_options.signal();
                output.stop_reason = if signal.is_cancelled() {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                output.error_message = Some(message);
                forward.push(AssistantMessageEvent::Error {
                    reason: output.stop_reason,
                    error: output.clone(),
                });
                forward.end(None);
            }
        }
    });
    events
}

/// Stream a simple assistant response, upstream's `streamSimple` export:
/// resolves the pi reasoning level to either a provider-native thinking
/// level (Gemini 3 / Gemma 4) or a token budget (Gemini 2.x), and disables
/// thinking outright when no level is requested.
///
/// Upstream throws synchronously when the key is missing; the port encodes
/// that failure as a settled error stream per the stream contract.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let Some(api_key) = options.as_ref().and_then(|options| options.api_key.as_deref()) else {
        return setup_error_stream(model, &format!("No API key for provider: {}", model.provider.0));
    };

    // The base options carry sampling_params upstream-side, but the Google
    // options shape has no samplingParams field: the merge is dropped.
    let base = build_base_options(model, context, options, Some(api_key));
    let tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(|choice| match choice {
            crate::types::ToolChoice::Auto => "auto",
            crate::types::ToolChoice::None => "none",
        });
    let mut base_options = GoogleStreamOptions::from(base);
    base_options.tool_choice = tool_choice.map(str::to_owned);

    let Some(reasoning) = options.and_then(|options| options.reasoning) else {
        return stream(
            model,
            context,
            Some(&GoogleStreamOptions {
                thinking: Some(GoogleThinkingControl {
                    enabled: false,
                    ..GoogleThinkingControl::default()
                }),
                ..base_options
            }),
        );
    };

    let clamped = crate::models::clamp_thinking_level(model, ModelThinkingLevel::from(reasoning));
    let resolved = match resolve_google_thinking_level(model, clamped) {
        Ok(level) => level,
        Err(message) => return setup_error_stream(model, &message),
    };

    if is_gemini3_pro_model(model) || is_gemini3_flash_model(model) || is_gemma4_model(model) {
        return stream(
            model,
            context,
            Some(&GoogleStreamOptions {
                thinking: Some(GoogleThinkingControl {
                    enabled: true,
                    level: Some(get_thinking_level(resolved, model).to_owned()),
                    budget_tokens: None,
                }),
                ..base_options
            }),
        );
    }

    stream(
        model,
        context,
        Some(&GoogleStreamOptions {
            thinking: Some(GoogleThinkingControl {
                enabled: true,
                budget_tokens: Some(get_google_budget(
                    model,
                    resolved,
                    options.and_then(|options| options.thinking_budgets.as_ref()),
                )),
                level: None,
            }),
            ..base_options
        }),
    )
}

/// The fresh accumulator a stream starts from, upstream's `output`: zeroed
/// usage and `stopReason: "pending"`.
fn initial_output(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: crate::types::Usage::default(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: crate::auth::resolve::now_ms(),
    }
}

async fn run_stream(
    model: &Model,
    context: &Context,
    options: &GoogleStreamOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    if options.api_key.as_deref().unwrap_or_default().is_empty() {
        return Err(format!("No API key for provider: {}", model.provider.0));
    }

    let mut params = build_params(model, context, options)?;
    if let Some(hook) = &options.transport_options.on_payload {
        params = hook
            .call(params.clone(), model.clone())
            .await
            .unwrap_or(params);
    }

    let response = dispatch_stream_request(model, &params, options).await?;
    events.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });
    consume_google_stream(model, response, output, events).await?;
    finish_stream(options, output, events)
}

/// Build and dispatch the stream request with the shared provider retry
/// policy, the seam-side port of upstream's `createClient` +
/// `retryGoogleRequest(() => client.models.generateContentStream(params))`.
async fn dispatch_stream_request(
    model: &Model,
    params: &Value,
    options: &GoogleStreamOptions,
) -> Result<crate::http::client::HttpResponse, String> {
    let http_client = options.transport_options.client();
    let signal = options.transport_options.signal();
    let request = HttpRequest {
        method: HttpMethod::Post,
        url: stream_generate_url(model),
        headers: build_request_headers(model, options),
        body: Some(Bytes::from(to_wire_body(params).to_string())),
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
            async move {
                let response = client
                    .execute(request)
                    .await
                    .map_err(provider_error_from_http)?;
                if !(200..300).contains(&response.status) {
                    // The pinned SDK folds the response body into the
                    // ApiError message, which the error formatter then
                    // passes through unchanged.
                    let body_text = read_body_text(response.body).await.unwrap_or_default();
                    return Err(ProviderRequestError::new(
                        Some(response.status),
                        Some(headers_to_record(
                            response
                                .headers
                                .iter()
                                .map(|(name, value)| (name.as_str(), value.as_str())),
                        )),
                        api_error_message(response.status, &body_text),
                    ));
                }
                Ok(response)
            }
        },
        &retry_options,
    )
    .await
    .map_err(|error| error.message)?;

    if let Some(hook) = &options.transport_options.on_response {
        hook.call(
            crate::types::ProviderResponse {
                status: response.status,
                headers: headers_to_record(
                    response
                        .headers
                        .iter()
                        .map(|(name, value)| (name.as_str(), value.as_str())),
                ),
            },
            model.clone(),
        )
        .await;
    }

    Ok(response)
}

/// The request URL, the SDK's URL construction: a custom `model.baseUrl` is
/// used verbatim (it already includes the version path; the SDK's empty
/// `apiVersion` skips the segment), otherwise the default host plus
/// `v1beta`.
fn stream_generate_url(model: &Model) -> String {
    let base = if model.base_url.trim().is_empty() {
        DEFAULT_GEMINI_BASE_URL.to_owned()
    } else {
        model.base_url.trim_end_matches('/').to_owned()
    };
    format!("{base}/models/{}:streamGenerateContent?alt=sse", model.id)
}

/// Assemble the request headers, upstream's `createClient` header merge: pi's
/// user agent, the model headers, then the caller headers (a `None` value
/// suppresses a default), with the credential header appended last and only
/// when the caller did not already set it.
fn build_request_headers(model: &Model, options: &GoogleStreamOptions) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = vec![
        ("User-Agent".to_owned(), get_pi_user_agent()),
        ("content-type".to_owned(), "application/json".to_owned()),
    ];
    if let Some(model_headers) = &model.headers {
        for (name, value) in model_headers {
            upsert_header(&mut headers, name, value);
        }
    }
    if let Some(record) = provider_headers_to_record(options.headers.as_ref()) {
        for (name, value) in record {
            upsert_header(&mut headers, &name, &value);
        }
    }
    if !headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("x-goog-api-key"))
    {
        headers.push((
            "x-goog-api-key".to_owned(),
            options.api_key.clone().unwrap_or_default(),
        ));
    }
    headers
}

fn upsert_header(headers: &mut Vec<(String, String)>, name: &str, value: &str) {
    if let Some(entry) = headers
        .iter_mut()
        .find(|(existing, _)| existing.eq_ignore_ascii_case(name))
    {
        entry.1 = value.to_owned();
    } else {
        headers.push((name.to_owned(), value.to_owned()));
    }
}

/// Build the SDK-style params, upstream's `buildParams`: `{ model, contents,
/// config }` with the camelCase config fields. The wire flattening happens in
/// [`to_wire_body`]; `onPayload` sees this shape.
///
/// # Errors
/// When a tool requires strict sampling that cannot be resolved, or when the
/// request is already aborted.
fn build_params(
    model: &Model,
    context: &Context,
    options: &GoogleStreamOptions,
) -> Result<Value, String> {
    let contents = convert_messages(model, context);

    let mut generation_config = serde_json::Map::new();
    if let Some(temperature) = options.temperature {
        generation_config.insert("temperature".to_owned(), json!(temperature));
    }
    if let Some(max_tokens) = options.max_tokens {
        generation_config.insert("maxOutputTokens".to_owned(), json!(max_tokens));
    }

    let supports_strict_mode = supports_google_strict_tool_sampling(&model.id);
    let tools = context.tools.as_deref().unwrap_or_default();
    let function_calling_mode = if !tools.is_empty() {
        Some(resolve_google_function_calling_mode(
            tools,
            options.tool_choice.as_deref(),
            supports_strict_mode,
        )?)
    } else {
        None
    };

    let mut config = serde_json::Map::new();
    for (key, value) in generation_config {
        config.insert(key, value);
    }
    if let Some(system_prompt) = &context.system_prompt {
        config.insert("systemInstruction".to_owned(), json!(system_prompt));
    }
    if let Some(tools) = &context.tools {
        if !tools.is_empty()
            && let Some(declarations) = convert_tools(tools, false, supports_strict_mode)?
        {
            config.insert("tools".to_owned(), declarations);
        }
    }
    if let Some(Some(mode)) = function_calling_mode {
        config.insert(
            "toolConfig".to_owned(),
            json!({ "functionCallingConfig": { "mode": mode } }),
        );
    }

    if options.thinking.as_ref().is_some_and(|thinking| thinking.enabled) && model.reasoning {
        let mut thinking_config = serde_json::Map::new();
        thinking_config.insert("includeThoughts".to_owned(), json!(true));
        if let Some(level) = options
            .thinking
            .as_ref()
            .and_then(|thinking| thinking.level.as_deref())
        {
            thinking_config.insert("thinkingLevel".to_owned(), json!(level));
        } else if let Some(budget) = options
            .thinking
            .as_ref()
            .and_then(|thinking| thinking.budget_tokens)
        {
            thinking_config.insert("thinkingBudget".to_owned(), json!(budget));
        }
        config.insert("thinkingConfig".to_owned(), Value::Object(thinking_config));
    } else if model.reasoning
        && let Some(thinking) = &options.thinking
        && !thinking.enabled
    {
        config.insert(
            "thinkingConfig".to_owned(),
            get_disabled_thinking_config(model),
        );
    }

    if options.transport_options.signal().is_cancelled() {
        return Err("Request aborted".to_owned());
    }

    Ok(json!({
        "model": model.id,
        "contents": contents,
        "config": Value::Object(config),
    }))
}

/// Flatten the SDK-style params into the request body the API expects, the
/// port of upstream's `generateContentParametersToMldev` serializer:
/// `contents` pass through verbatim, `config` becomes `generationConfig`, and
/// `systemInstruction`, `tools`, and `toolConfig` hoist to the body root.
/// The body never carries a `model` key; the model names the URL path.
fn to_wire_body(params: &Value) -> Value {
    let mut body = serde_json::Map::new();
    body.insert(
        "contents".to_owned(),
        params
            .get("contents")
            .cloned()
            .unwrap_or_else(|| json!([])),
    );

    if let Some(config) = params.get("config").and_then(Value::as_object) {
        let mut generation_config = serde_json::Map::new();
        for (key, value) in config {
            match key.as_str() {
                "systemInstruction" => {
                    body.insert("systemInstruction".to_owned(), content_from_text(value));
                }
                "tools" => {
                    body.insert("tools".to_owned(), value.clone());
                }
                "toolConfig" => {
                    body.insert("toolConfig".to_owned(), value.clone());
                }
                _ => {
                    generation_config.insert(key.clone(), value.clone());
                }
            }
        }
        if !generation_config.is_empty() {
            body.insert(
                "generationConfig".to_owned(),
                Value::Object(generation_config),
            );
        }
    }

    Value::Object(body)
}

/// The SDK's `tContent`: a string system instruction rides as a Content
/// object with role `user`; an already-shaped object passes through.
fn content_from_text(value: &Value) -> Value {
    if let Some(text) = value.as_str() {
        json!({ "role": "user", "parts": [{ "text": text }] })
    } else {
        value.clone()
    }
}

/// The disabled-thinking config, upstream's `getDisabledThinkingConfig`.
///
/// Google docs: Gemini 3.1 Pro cannot disable thinking, and Gemini 3 Flash /
/// Flash-Lite do not support full thinking-off either. For Gemini 3 models,
/// the lowest supported `thinkingLevel` rides without `includeThoughts` so
/// hidden thinking stays invisible to pi. Gemini 2.x disables via
/// `thinkingBudget: 0`.
fn get_disabled_thinking_config(model: &Model) -> Value {
    if is_gemini3_pro_model(model) {
        json!({ "thinkingLevel": "LOW" })
    } else if is_gemini3_flash_model(model) || is_gemma4_model(model) {
        json!({ "thinkingLevel": "MINIMAL" })
    } else {
        json!({ "thinkingBudget": 0 })
    }
}

/// `/gemma-?4/` over the lowercased id.
fn is_gemma4_model(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    id.contains("gemma-4") || id.contains("gemma4")
}

/// `/gemini-3(?:\.\d+)?-pro/` over the lowercased id.
fn is_gemini3_pro_model(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    id.split("gemini-3").skip(1).any(|rest| {
        let rest = rest.strip_prefix('.').map_or(rest, |after_dot| {
            let digits = after_dot.chars().take_while(|c| c.is_ascii_digit()).count();
            if digits == 0 {
                rest
            } else {
                &after_dot[digits..]
            }
        });
        rest.starts_with("-pro")
    })
}

/// `/gemini-3(?:\.\d+)?-flash/` over the lowercased id, plus the two
/// rolling-latest aliases.
fn is_gemini3_flash_model(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    id.split("gemini-3").skip(1).any(|rest| {
        let rest = rest.strip_prefix('.').map_or(rest, |after_dot| {
            let digits = after_dot.chars().take_while(|c| c.is_ascii_digit()).count();
            if digits == 0 {
                rest
            } else {
                &after_dot[digits..]
            }
        });
        rest.starts_with("-flash")
    }) || id == "gemini-flash-latest"
        || id == "gemini-flash-lite-latest"
}

/// Map a resolved pi level to the provider-native thinking level,
/// upstream's `getThinkingLevel`.
#[must_use]
pub fn get_thinking_level(effort: ResolvedGoogleThinkingLevel, model: &Model) -> &'static str {
    if is_gemini3_pro_model(model) {
        return match effort {
            ResolvedGoogleThinkingLevel::Minimal | ResolvedGoogleThinkingLevel::Low => "LOW",
            ResolvedGoogleThinkingLevel::Medium | ResolvedGoogleThinkingLevel::High => "HIGH",
        };
    }
    if is_gemma4_model(model) {
        return match effort {
            ResolvedGoogleThinkingLevel::Minimal | ResolvedGoogleThinkingLevel::Low => "MINIMAL",
            ResolvedGoogleThinkingLevel::Medium | ResolvedGoogleThinkingLevel::High => "HIGH",
        };
    }
    match effort {
        ResolvedGoogleThinkingLevel::Minimal => "MINIMAL",
        ResolvedGoogleThinkingLevel::Low => "LOW",
        ResolvedGoogleThinkingLevel::Medium => "MEDIUM",
        ResolvedGoogleThinkingLevel::High => "HIGH",
    }
}

/// The catalog's default thinking budgets, upstream's `getGoogleBudget`;
/// `-1` is the dynamic budget where the catalog has no entry.
#[must_use]
pub fn get_google_budget(
    model: &Model,
    level: ResolvedGoogleThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> i64 {
    if let Some(budgets) = custom_budgets {
        let custom = match level {
            ResolvedGoogleThinkingLevel::Minimal => budgets.minimal,
            ResolvedGoogleThinkingLevel::Low => budgets.low,
            ResolvedGoogleThinkingLevel::Medium => budgets.medium,
            ResolvedGoogleThinkingLevel::High => budgets.high,
        };
        if let Some(budget) = custom {
            return i64::try_from(budget).unwrap_or(-1);
        }
    }

    let (minimal, low, medium, high) = if model.id.contains("2.5-pro") {
        (128, 2048, 8192, 32768)
    } else if model.id.contains("2.5-flash-lite") {
        (512, 2048, 8192, 24576)
    } else if model.id.contains("2.5-flash") {
        (128, 2048, 8192, 24576)
    } else {
        return -1;
    };
    i64::from(match level {
        ResolvedGoogleThinkingLevel::Minimal => minimal,
        ResolvedGoogleThinkingLevel::Low => low,
        ResolvedGoogleThinkingLevel::Medium => medium,
        ResolvedGoogleThinkingLevel::High => high,
    })
}

/// The ApiError message the pinned SDK builds for a failed response: the
/// parsed JSON body stringified, or a synthesized `{ error: { message, code,
/// status } }` object when the body is not JSON.
fn api_error_message(status: u16, body: &str) -> String {
    if let Ok(parsed) = serde_json::from_str::<Value>(body) {
        return safe_json_stringify(&parsed);
    }
    if body.trim().is_empty() {
        return safe_json_stringify(&json!({}));
    }
    safe_json_stringify(&json!({
        "error": {
            "message": body,
            "code": status,
            "status": "UNKNOWN",
        }
    }))
}

fn provider_error_from_http(error: HttpError) -> ProviderRequestError {
    match error {
        HttpError::Aborted => ProviderRequestError::aborted(),
        HttpError::Timeout => ProviderRequestError::new(None, None, "request timed out"),
        HttpError::Transport(message) => ProviderRequestError::new(None, None, message),
        HttpError::InvalidUrl(url) => {
            ProviderRequestError::new(None, None, format!("invalid URL: {url}"))
        }
    }
}

/// The stream-setup failure as a settled error stream, upstream's
/// synchronous `streamSimple` throw encoded per the stream contract.
fn setup_error_stream(model: &Model, message: &str) -> AssistantMessageEventStream {
    let events = assistant_message_event_stream();
    let failure = std::io::Error::other(message.to_owned());
    let failing = crate::api::lazy::setup_error_message(model, &failure);
    events.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: failing.clone(),
    });
    events.end(Some(&failing));
    events
}

/// Whether the open block is thinking (`Some(true)`), text (`Some(false)`),
/// or none (`None`).
type OpenBlock = Option<bool>;

/// The per-chunk state machine, upstream's `for await` loop over the SDK's
/// `generateContentStream` iterator.
async fn consume_google_stream(
    model: &Model,
    response: crate::http::client::HttpResponse,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let mut sse = crate::http::sse::SseStream::new(response.body);
    let mut current_block: OpenBlock = None;
    let mut first_event = true;
    while let Some(sse_event) = sse.next().await.map_err(|error| match error {
        HttpError::Aborted => "Request was aborted".to_owned(),
        other => other.to_string(),
    })? {
        let value = parse_json_with_repair(&sse_event.data)
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
                return Err(format!("got status: {code}. {}", safe_json_stringify(&value)));
            }
        }

        // `GenerateContentResponse.responseId` is an output-only field used
        // to identify each response; keep the first non-empty one.
        if let Some(id) = value.get("responseId").and_then(Value::as_str)
            && !id.is_empty()
            && output.response_id.as_ref().is_none_or(String::is_empty)
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
    let block_index = output.content.len().saturating_sub(1) as u64;
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
    let _ = block_index;
    if is_thinking {
        let Some(AssistantBlock::Thinking(block)) = output.content.last_mut() else {
            return;
        };
        block.thinking.push_str(text);
        block.thinking_signature = retain_thought_signature(
            block.thinking_signature.as_deref(),
            thought_signature,
        );
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
        block.text_signature = retain_thought_signature(
            block.text_signature.as_deref(),
            thought_signature,
        );
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
fn apply_function_call(
    function_call: &Value,
    part: &Value,
    output: &mut AssistantMessage,
) {
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
            counter = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed) + 1,
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

/// The three-event function-call sequence, split from
/// [`apply_function_call`] so the partial snapshot carries the pushed call.
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

/// The post-loop settlement, upstream's tail of the try block: the signal
/// check, the pending and provider-stopped rejections, then the done event.
fn finish_stream(
    options: &GoogleStreamOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    if options.transport_options.signal().is_cancelled() {
        return Err("Request was aborted".to_owned());
    }
    if output.stop_reason == StopReason::Pending {
        return Err("Google stream ended without a finish reason".to_owned());
    }
    if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
        return Err(output
            .raw_stop_reason
            .as_ref()
            .map(|reason| format!("Provider stopped with: {reason}"))
            .unwrap_or_else(|| "An unknown error occurred".to_owned()));
    }

    events.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output.clone(),
    });
    Ok(())
}