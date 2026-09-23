//! The Azure OpenAI Responses wire API, ported from
//! `packages/ai/src/api/azure-openai-responses.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The module wraps the shared machinery of
//! [`crate::api::openai_responses_shared`] the way upstream re-exports it,
//! adding the Azure deployment and base-url resolution on top.
//!
//! Porting restatements:
//!
//! - The `AzureOpenAI` SDK client collapses onto the [`crate::http::HttpClient`]
//!   seam. The pinned SDK (openai 6.x) builds `responses.create` as POST
//!   `{baseURL}/responses`, appends the `defaultQuery` `api-version`
//!   parameter, and — because `/responses` is not one of the SDK's
//!   deployment-rewritten endpoints — leaves the deployment name to the
//!   body's `model` field. Its auth override sends `api-key: {apiKey}`
//!   instead of a bearer pair. The wire request is therefore
//!   `{baseUrl}/responses?api-version={apiVersion}` with an `api-key`
//!   header, and an existing query on a proxy base URL joins with `&`.
//! - The SDK stream is the shared SSE decoder (`data: [DONE]` sentinel) over
//!   the seam body; event decoding is shared with
//!   [`crate::api::openai_responses_shared`].
//! - `sanitizeSurrogates` disappears statically: a Rust [`String`] cannot
//!   hold the unpaired surrogates it stripped, and `serde_json` rejects
//!   lone-surrogate escapes when reading the wire.
//! - `streamSimple`'s synchronous auth throw becomes the setup-error stream:
//!   the Rust stream contract returns a stream synchronously, so a missing
//!   key surfaces as its `error` event.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::api::constrained_sampling::create_grammar_tool_input_properties;
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::openai_responses::{
    ReasoningSummary, mapped_effort, pi_thinking_level, thinking_level_str, thinking_model_level,
};
use crate::api::openai_responses_shared::{
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, convert_responses_messages,
    convert_responses_tools,
};
use crate::api::openai_responses_shared::{OpenAiResponsesStreamOptions, process_responses_stream};
use crate::api::request_seam::{
    execute_checked_response, fire_response_hook, setup_error_stream, spawned_stream,
};
use crate::api::simple_options::build_base_options;
use crate::types::{
    AssistantMessage, CacheRetention, Context, JsonValue, KnownApi, Model, ModelThinkingLevel,
    ProviderEnv, ProviderHeaders, SimpleStreamOptions, StopReason, StreamOptions, TransportOptions,
};
use crate::utils::error_body::{
    ErrorBody, SdkError, format_provider_error, normalize_provider_error,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};

/// The api-version an Azure deployment runs under when nothing else names
/// one, upstream's `DEFAULT_AZURE_API_VERSION`.
const DEFAULT_AZURE_API_VERSION: &str = "v1";

/// The providers whose tool-call item ids keep OpenAI Responses pairing
/// history, upstream's `AZURE_TOOL_CALL_PROVIDERS`.
fn azure_tool_call_providers() -> BTreeSet<String> {
    [
        "openai",
        "openai-codex",
        "opencode",
        "azure-openai-responses",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// OpenAI Responses rejects `max_output_tokens` below 16. See
/// <https://github.com/earendil-works/pi/issues/6265>.
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;

// ---------------------------------------------------------------------------
// Deployment resolution
// ---------------------------------------------------------------------------

/// Parse the `modelId=deploymentName,...` map the
/// `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` environment value carries, upstream's
/// `parseDeploymentNameMap`.
fn parse_deployment_name_map(value: Option<&str>) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let Some(value) = value else {
        return map;
    };
    for entry in value.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.splitn(2, '=');
        let (Some(model_id), Some(deployment_name)) = (parts.next(), parts.next()) else {
            continue;
        };
        if model_id.trim().is_empty() || deployment_name.trim().is_empty() {
            continue;
        }
        map.insert(
            model_id.trim().to_owned(),
            deployment_name.trim().to_owned(),
        );
    }
    map
}

/// The deployment name a request targets, upstream's `resolveDeploymentName`:
/// the explicit option, else the environment map's entry for the model id,
/// else the model id itself.
fn resolve_deployment_name(model: &Model, options: &AzureOpenAiResponsesOptions) -> String {
    if let Some(name) = options
        .azure_deployment_name
        .as_deref()
        .filter(|name| !name.is_empty())
    {
        return name.to_owned();
    }
    let mapped_deployment = parse_deployment_name_map(
        get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", options.env.as_ref()).as_deref(),
    )
    .get(&model.id)
    .cloned();
    mapped_deployment.unwrap_or_else(|| model.id.clone())
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// The Azure OpenAI Responses options, upstream's
/// `AzureOpenAIResponsesOptions extends StreamOptions`: the base fields this
/// adapter reads plus the reasoning, tool-choice, and Azure endpoint extras.
///
/// Base fields the adapter never reads (`transport`,
/// `websocketConnectTimeoutMs`) are omitted, matching the sibling wire APIs'
/// option structs.
#[derive(Clone, Debug, Default)]
pub struct AzureOpenAiResponsesOptions {
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
    pub sampling_params: Option<BTreeMap<String, JsonValue>>,
    /// Maximum output tokens. The wire floor is 16.
    pub max_tokens: Option<u64>,
    /// Prompt cache retention preference. Default: `"short"`.
    pub cache_retention: Option<CacheRetention>,
    /// Session identifier for session-based caching and routing.
    pub session_id: Option<String>,
    /// Optional request metadata; the adapter does not read it.
    pub metadata: Option<BTreeMap<String, JsonValue>>,
    /// The reasoning effort, upstream's
    /// `"minimal" | "low" | "medium" | "high" | "xhigh" | "max"`.
    pub reasoning_effort: Option<crate::types::ThinkingLevel>,
    /// The Responses `tool_choice` value, forwarded verbatim: `"none"`,
    /// `"auto"`, `"required"`, or a provider tool-choice object.
    pub tool_choice: Option<JsonValue>,
    /// The reasoning summary detail, upstream's
    /// `"auto" | "detailed" | "concise" | null`.
    pub reasoning_summary: Option<ReasoningSummary>,
    /// The `api-version` query parameter the request carries.
    pub azure_api_version: Option<String>,
    /// The Azure resource name the default base URL builds from.
    pub azure_resource_name: Option<String>,
    /// The Azure (or proxy) base URL, overriding the model's.
    pub azure_base_url: Option<String>,
    /// The deployment name the `model` field carries, overriding the
    /// environment map and the model id.
    pub azure_deployment_name: Option<String>,
}

crate::api::adapter_belt::impl_stream_options_from!(AzureOpenAiResponsesOptions from options {
    sampling_params: options.sampling_params,
    reasoning_effort: None,
    tool_choice: None,
    reasoning_summary: None,
    azure_api_version: None,
    azure_resource_name: None,
    azure_base_url: None,
    azure_deployment_name: None,
});
// ---------------------------------------------------------------------------
// Azure endpoint resolution
// ---------------------------------------------------------------------------

/// Normalize an Azure OpenAI base URL, upstream's `normalizeAzureBaseUrl`:
/// Azure hosts get `/openai/v1` as base path (the path the SDK-style request
/// appends to) and drop a stale query; proxy hosts keep their path and
/// query.
///
/// # Errors
/// A base URL that does not parse fails with the wire-meaningful message.
fn normalize_azure_base_url(base_url: &str) -> Result<String, String> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let mut url = url::Url::parse(trimmed)
        .map_err(|_| format!("Invalid Azure OpenAI base URL: {base_url}"))?;

    let hostname = url.host_str().unwrap_or_default().to_owned();
    let is_azure_host = hostname.ends_with(".openai.azure.com")
        || hostname.ends_with(".cognitiveservices.azure.com")
        || hostname.ends_with(".ai.azure.com");
    let normalized_path = url.path().trim_end_matches('/').to_owned();

    // Ensure Azure hosts have /openai/v1 as base path so the SDK-style
    // request appends /responses and ?api-version correctly.
    if is_azure_host
        && (normalized_path.is_empty()
            || normalized_path == "/"
            || normalized_path == "/openai"
            || normalized_path == "/openai/v1/responses")
    {
        url.set_path("/openai/v1");
        url.set_query(None);
    }

    Ok(url.to_string().trim_end_matches('/').to_owned())
}

/// The default base URL a resource name builds, upstream's
/// `buildDefaultBaseUrl`.
fn build_default_base_url(resource_name: &str) -> String {
    format!("https://{resource_name}.openai.azure.com/openai/v1")
}

/// Resolve the endpoint and api-version a request sends, upstream's
/// `resolveAzureConfig`: the options over the environment over the model's
/// base URL, with the resource name building the default URL.
///
/// # Errors
/// A base URL that does not parse, and the no-base-url failure when nothing
/// names one.
fn resolve_azure_config(
    model: &Model,
    options: &AzureOpenAiResponsesOptions,
) -> Result<(String, String), String> {
    let api_version = options
        .azure_api_version
        .as_deref()
        .filter(|version| !version.is_empty())
        .map_or_else(
            || {
                get_provider_env_value("AZURE_OPENAI_API_VERSION", options.env.as_ref())
                    .filter(|version| !version.is_empty())
                    .unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.to_owned())
            },
            str::to_owned,
        );

    let base_url = options
        .azure_base_url
        .as_deref()
        .map(str::trim)
        .filter(|base_url| !base_url.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            get_provider_env_value("AZURE_OPENAI_BASE_URL", options.env.as_ref())
                .map(|base_url| base_url.trim().to_owned())
                .filter(|base_url| !base_url.is_empty())
        });
    let resource_name = options
        .azure_resource_name
        .as_deref()
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            get_provider_env_value("AZURE_OPENAI_RESOURCE_NAME", options.env.as_ref())
                .filter(|name| !name.is_empty())
        });

    let resolved_base_url = base_url
        .or_else(|| resource_name.as_ref().map(|name| build_default_base_url(name)))
        .or_else(|| (!model.base_url.is_empty()).then(|| model.base_url.clone()))
        .ok_or_else(|| {
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl.".to_owned()
        })?;

    Ok((normalize_azure_base_url(&resolved_base_url)?, api_version))
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

/// Assemble the request headers a stream sends, the port of upstream's
/// `createClient`: the pi user agent, the model's headers, and the caller's
/// headers merged last so they can override defaults. Azure requests carry
/// no session-affinity headers.
fn build_request_headers(
    model: &Model,
    options_headers: Option<&ProviderHeaders>,
) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = vec![
        ("User-Agent".to_owned(), get_pi_user_agent()),
        // The pinned OpenAI SDK's JSON content type, overridable like the
        // completions adapter's.
        ("content-type".to_owned(), "application/json".to_owned()),
    ];
    if let Some(model_headers) = &model.headers {
        for (name, value) in model_headers {
            headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
            headers.push((name.clone(), value.clone()));
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

/// The responses endpoint of a resolved Azure base URL, the request the
/// pinned SDK builds: `/responses` joins the base path before any existing
/// proxy query, and the `defaultQuery` `api-version` joins the query with
/// `&`.
fn azure_responses_url(base_url: &str, api_version: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    match trimmed.split_once('?') {
        Some((path, query)) => {
            format!("{path}/responses?{query}&api-version={api_version}")
        }
        None => format!("{trimmed}/responses?api-version={api_version}"),
    }
}

/// Build the request payload, upstream's `buildParams`. The deployment name
/// rides the `model` field; Azure requests carry no deferred-tool split and
/// no prompt-cache retention extras.
///
/// # Errors
/// The message- and tool-conversion rejections the input list hits.
fn build_params(
    model: &Model,
    context: &Context,
    options: &AzureOpenAiResponsesOptions,
    deployment_name: &str,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Map<String, JsonValue>, String> {
    let messages = convert_responses_messages(
        model,
        context,
        &azure_tool_call_providers(),
        Some(&ConvertResponsesMessagesOptions {
            grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
            ..ConvertResponsesMessagesOptions::default()
        }),
    )?;

    let mut params = Map::new();
    params.insert("model".to_owned(), json!(deployment_name));
    params.insert("input".to_owned(), Value::Array(messages));
    params.insert("stream".to_owned(), json!(true));
    if let Some(key) = clamp_openai_prompt_cache_key(options.session_id.as_deref()) {
        params.insert("prompt_cache_key".to_owned(), json!(key));
    }
    params.insert("store".to_owned(), json!(false));

    if let Some(max_tokens) = options.max_tokens {
        params.insert(
            "max_output_tokens".to_owned(),
            json!(max_tokens.max(OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)),
        );
    }

    if let Some(temperature) = options.temperature {
        params.insert("temperature".to_owned(), json!(temperature));
    }

    if let Some(tools) = context.tools.as_deref().filter(|tools| !tools.is_empty()) {
        params.insert(
            "tools".to_owned(),
            Value::Array(convert_azure_tools(model, tools)?),
        );
    }
    if let Some(tool_choice) = &options.tool_choice {
        params.insert("tool_choice".to_owned(), tool_choice.clone());
    }

    if model.reasoning {
        let level_map = model.thinking_level_map.as_ref();
        if options.reasoning_effort.is_some() || options.reasoning_summary.is_some() {
            let effort = options.reasoning_effort.map_or_else(
                || json!("medium"),
                |level| {
                    mapped_effort(level_map, level)
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
        } else if level_map.and_then(|map| map.get(&ModelThinkingLevel::Off)) != Some(&None) {
            let effort = level_map
                .and_then(|map| map.get(&ModelThinkingLevel::Off))
                .cloned()
                .flatten()
                .unwrap_or_else(|| "none".to_owned());
            params.insert("reasoning".to_owned(), json!({ "effort": effort }));
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

/// The request's tool definitions, upstream's `convertResponsesTools` call
/// with the model's strict and grammar flags.
///
/// # Errors
/// The grammar and strict-conversion rejections a tool hits.
fn convert_azure_tools(
    model: &Model,
    tools: &[crate::types::Tool],
) -> Result<Vec<JsonValue>, String> {
    let compat = model.compat.as_ref();
    convert_responses_tools(
        tools,
        Some(&ConvertResponsesToolsOptions {
            supports_strict_mode: Some(
                compat
                    .and_then(|compat| compat.supports_strict_mode)
                    .unwrap_or(true),
            ),
            supports_openai_grammar_tools: Some(
                compat
                    .and_then(|compat| compat.supports_openai_grammar_tools)
                    .unwrap_or(false),
            ),
            ..ConvertResponsesToolsOptions::default()
        }),
    )
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// Stream an assistant response over the Azure OpenAI Responses wire,
/// upstream's `stream`. The stream returns live; request setup, model, and
/// runtime failures arrive as its `error` event.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&AzureOpenAiResponsesOptions>,
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
                let deployment_name = resolve_deployment_name(&model, &options);
                run_stream(
                    &model,
                    &context,
                    &options,
                    &deployment_name,
                    output,
                    &forward,
                )
                .await
            })
        },
    )
}

/// The fresh accumulator a stream starts from, upstream's `output`: the
/// Azure wire-api id, zeroed usage, and `stopReason: "pending"`.
fn initial_output(model: &Model) -> AssistantMessage {
    crate::api::request_seam::initial_output(
        model,
        crate::types::Api::from(KnownApi::AzureOpenaiResponses),
        None,
    )
}

/// Run one stream to completion: require the credential, build and dispatch
/// the request, process the events, and settle the final message.
async fn run_stream(
    model: &Model,
    context: &Context,
    options: &AzureOpenAiResponsesOptions,
    deployment_name: &str,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let api_key = options
        .api_key
        .as_deref()
        .filter(|api_key| !api_key.is_empty())
        .ok_or_else(|| format!("No API key for provider: {}", model.provider.0))?;
    let supports_grammar = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_openai_grammar_tools)
        .unwrap_or(false);
    let grammar_tool_input_properties =
        create_grammar_tool_input_properties(context.tools.as_deref(), supports_grammar);
    let payload = build_params(
        model,
        context,
        options,
        deployment_name,
        &grammar_tool_input_properties,
    )?;
    let response = dispatch_stream_request(model, options, api_key, payload).await?;
    events.push(crate::types::AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let stream_options = OpenAiResponsesStreamOptions {
        grammar_tool_input_properties: Some(grammar_tool_input_properties),
        ..OpenAiResponsesStreamOptions::default()
    };
    process_responses_stream(response, output, events, model, Some(&stream_options)).await?;

    if options.transport_options.signal().is_cancelled() {
        return Err("Request was aborted".to_owned());
    }
    if output.stop_reason == StopReason::Pending {
        return Err("Azure OpenAI Responses stream ended without a stop reason".to_owned());
    }
    if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
        return Err(output
            .error_message
            .clone()
            .unwrap_or_else(|| "An unknown error occurred".to_owned()));
    }

    events.push(crate::types::AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output.clone(),
    });
    Ok(())
}

/// The display prefix the provider's errors carry, upstream's
/// `formatAzureOpenAIError` argument.
const AZURE_ERROR_PREFIX: &str = "Azure OpenAI API error";

/// Execute the stream request with retries and fire the response hooks, the
/// seam-side port of upstream's `createClient` +
/// `retryProviderRequest(create(...).asResponse())`. A non-2xx body is
/// parsed before it is discarded so the failure message can surface it the
/// way the pinned SDK folds it into its error.
async fn dispatch_stream_request(
    model: &Model,
    options: &AzureOpenAiResponsesOptions,
    api_key: &str,
    payload: Map<String, JsonValue>,
) -> Result<crate::http::client::HttpResponse, String> {
    let (base_url, api_version) = resolve_azure_config(model, options)?;
    let url = azure_responses_url(&base_url, &api_version);
    let mut headers = vec![("api-key".to_owned(), api_key.to_owned())];
    headers.extend(build_request_headers(model, options.headers.as_ref()));

    let mut payload = payload;
    if let Some(hook) = &options.transport_options.on_payload
        && let Some(JsonValue::Object(next)) = hook
            .call(JsonValue::Object(payload.clone()), model.clone())
            .await
    {
        payload = next;
    }

    let http_client = options.transport_options.client();
    let signal = options.transport_options.signal();
    let request = crate::http::client::HttpRequest {
        method: crate::http::client::HttpMethod::Post,
        url,
        headers,
        body: Some(bytes::Bytes::from(JsonValue::Object(payload).to_string())),
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
    let parsed_error_body: Arc<std::sync::Mutex<Option<JsonValue>>> = Arc::default();
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
                Some(AZURE_ERROR_PREFIX),
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

/// Stream a simple request, upstream's `streamSimple`: the pi reasoning
/// level clamps to the model's supported levels and spends an effort, with
/// the clamped-off state spending none.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let api_key = options.and_then(|options| options.api_key.as_deref());
    if api_key.is_none_or(str::is_empty) {
        return setup_error_stream(
            model,
            &format!("No API key for provider: {}", model.provider.0),
        );
    }

    let mut base =
        AzureOpenAiResponsesOptions::from(build_base_options(model, context, options, api_key));
    base.tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(|choice| match choice {
            crate::types::ToolChoice::Auto => json!("auto"),
            crate::types::ToolChoice::None => json!("none"),
        });
    base.reasoning_effort = options
        .and_then(|options| options.reasoning)
        .map(|reasoning| {
            crate::models::clamp_thinking_level(model, thinking_model_level(reasoning))
        })
        .and_then(pi_thinking_level);
    stream(model, context, Some(&base))
}

/// The Azure OpenAI Responses [`crate::types::ProviderStreams`], upstream's
/// module-level `stream`/`streamSimple` exports behind the uniform dispatch.
#[derive(Debug, Default)]
pub struct AzureOpenAiResponsesStreams;

crate::api::adapter_belt::impl_provider_streams!(
    AzureOpenAiResponsesStreams,
    AzureOpenAiResponsesOptions
);
