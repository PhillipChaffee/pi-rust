//! The Bedrock adapter options, client-config resolution, and the
//! Converse Stream command-input builder, ported from
//! `packages/ai/src/api/bedrock-converse-stream.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. The config fields mirror the
//! SDK client construction upstream's tests capture from the mocked
//! constructor.

use serde_json::{Value, json};

use crate::api::bedrock_converse_stream::{
    BedrockToolChoice, build_additional_model_request_fields, build_system_prompt,
    convert_messages, convert_tool_config, is_anthropic_claude_model, supports_strict_mode,
};
use crate::types::{Context, Model};

/// The adapter-facing options, upstream's `BedrockOptions`.
#[derive(Clone, Debug, Default)]
pub struct BedrockStreamOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks.
    pub transport_options: crate::types::TransportOptions,
    /// The API key, doubling as a Bedrock bearer token, upstream's
    /// `options.apiKey`.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this logical request.
    pub telemetry_context: Option<pi_telemetry::TelemetryHandle>,
    /// Provider-scoped environment values; these take precedence over the
    /// process environment.
    pub env: Option<crate::types::ProviderEnv>,
    /// Custom HTTP headers merged into the signed request, upstream's
    /// custom-headers middleware.
    pub headers: Option<crate::types::ProviderHeaders>,
    /// Accepted upstream-side and never wired into the SDK client: the SDK's
    /// own retry strategy owns retries.
    pub timeout_ms: Option<u64>,
    /// Accepted upstream-side and never wired into the client config.
    pub max_retries: Option<u32>,
    /// Accepted upstream-side and never wired into the client config.
    pub max_retry_delay_ms: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Arbitrary sampling parameters; upstream's Bedrock options shape has
    /// no samplingParams field, so the merge is dropped.
    pub sampling_params: Option<std::collections::BTreeMap<String, Value>>,
    /// Maximum output tokens; Claude models fall back to the model cap.
    pub max_tokens: Option<u64>,
    /// Preferred transport; the SDK adapter owns the transport.
    pub transport: Option<crate::types::Transport>,
    /// Prompt cache retention preference. Default: `"short"`.
    pub cache_retention: Option<crate::types::CacheRetention>,
    /// Optional session identifier; Bedrock prompt caching rides on cache
    /// points rather than session affinity.
    pub session_id: Option<String>,
    /// Optional metadata to include in API requests.
    pub metadata: Option<std::collections::BTreeMap<String, Value>>,
    /// The AWS region, upstream's `options.region`.
    pub region: Option<String>,
    /// The AWS profile, upstream's `options.profile`.
    pub profile: Option<String>,
    /// The tool choice, upstream's Bedrock-specific union.
    pub tool_choice: Option<BedrockToolChoice>,
    /// The pi thinking level, upstream's `reasoning`.
    pub reasoning: Option<crate::types::ThinkingLevel>,
    /// Custom token budgets for thinking levels.
    pub thinking_budgets: Option<crate::types::ThinkingBudgets>,
    /// Adaptive thinking for the models that support it; default true,
    /// upstream's `interleavedThinking`.
    pub interleaved_thinking: Option<bool>,
    /// The thinking display, upstream's `thinkingDisplay`; default
    /// `summarized`, upstream's pi choice over the Anthropic API default.
    pub thinking_display: Option<String>,
    /// Bedrock cost-allocation tags, upstream's `requestMetadata`.
    pub request_metadata: Option<std::collections::BTreeMap<String, Value>>,
    /// The Bedrock API key, bypassing SigV4, upstream's `bearerToken`.
    pub bearer_token: Option<String>,
}

impl From<crate::types::StreamOptions> for BedrockStreamOptions {
    fn from(options: crate::types::StreamOptions) -> Self {
        let crate::types::StreamOptions {
            transport_options,
            api_key,
            telemetry_context,
            env,
            headers,
            timeout_ms,
            max_retries: _,
            max_retry_delay_ms: _,
            temperature,
            sampling_params: _,
            max_tokens,
            transport,
            cache_retention,
            session_id: _,
            websocket_connect_timeout_ms: _,
            metadata,
        } = options;
        Self {
            transport_options,
            api_key,
            telemetry_context,
            env,
            headers,
            timeout_ms,
            max_retries: None,
            max_retry_delay_ms: None,
            temperature,
            sampling_params: None,
            max_tokens,
            transport,
            cache_retention,
            session_id: None,
            metadata,
            region: None,
            profile: None,
            tool_choice: None,
            reasoning: None,
            thinking_budgets: None,
            interleaved_thinking: None,
            thinking_display: None,
            request_metadata: None,
            bearer_token: None,
        }
    }
}

/// The AWS credentials the config resolution applies, upstream's
/// `getConfiguredBedrockCredentials`.
#[derive(Clone, Debug, PartialEq)]
pub struct BedrockCredentials {
    /// The access key id.
    pub access_key_id: String,
    /// The secret access key.
    pub secret_access_key: String,
    /// The assumed-role session token.
    pub session_token: Option<String>,
}

/// The resolved client configuration the seam sends with, the shape the
/// upstream tests capture from the mocked client constructor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BedrockClientConfig {
    /// The resolved region.
    pub region: Option<String>,
    /// The pinned endpoint URL, when the model's base URL is pinned.
    pub endpoint: Option<String>,
    /// The resolved profile name.
    pub profile: Option<String>,
    /// The env credentials when the resolution applies them (no profile).
    pub credentials: Option<BedrockCredentials>,
    /// The dummy credentials `AWS_BEDROCK_SKIP_AUTH=1` installs.
    pub skip_auth: bool,
    /// The bearer token when the resolution applies one.
    pub token: Option<String>,
    /// The bearer auth scheme preference, upstream's `authSchemePreference`.
    pub auth_scheme_preference: Option<String>,
    /// The caller headers the pre-signing middleware injects.
    pub headers: Vec<(String, String)>,
    /// The resolved proxy URL, upstream's `resolveHttpProxyUrlForTarget`.
    pub proxy_url: Option<String>,
    /// Whether the client forces HTTP/1.1, upstream's
    /// `AWS_BEDROCK_FORCE_HTTP1`.
    pub force_http1: bool,
}

/// The standard Bedrock runtime hostname, upstream's
/// `getStandardBedrockEndpointRegion`:
/// `bedrock-runtime[-fips].<region>.amazonaws.com[.cn]`.
fn standard_endpoint_region(base_url: &str) -> Option<String> {
    let host = url::Url::parse(base_url).ok()?.host_str()?.to_lowercase();
    let host = host
        .strip_suffix(".amazonaws.com.cn")
        .or_else(|| host.strip_suffix(".amazonaws.com"))?;
    let rest = host
        .strip_prefix("bedrock-runtime-fips")
        .or_else(|| host.strip_prefix("bedrock-runtime"))?;
    let region = rest.strip_prefix('.')?;
    (!region.is_empty()
        && region
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'))
        .then(|| region.to_owned())
}

/// Whether the endpoint is pinned to the SDK, upstream's
/// `shouldUseExplicitBedrockEndpoint`: a non-standard host always pins; a
/// standard host pins only when neither a configured region nor an ambient
/// profile exists.
fn should_use_explicit_bedrock_endpoint(
    model: &Model,
    configured_region: Option<&str>,
    ambient_profile: bool,
) -> bool {
    standard_endpoint_region(&model.base_url)
        .is_none()
        || (configured_region.is_none() && !ambient_profile)
}

/// The ambient-profile flag, upstream's `hasAmbientConfiguredProfile`.
fn has_ambient_configured_profile(options: &BedrockStreamOptions) -> bool {
    crate::utils::provider_env::get_provider_env_value("AWS_PROFILE", options.env.as_ref()).is_some()
}

/// The configured region from the options or the environment, upstream's
/// `configuredRegion` chain.
fn configured_region(options: &BedrockStreamOptions) -> Option<String> {
    options
        .region
        .clone()
        .or_else(|| crate::utils::provider_env::get_provider_env_value("AWS_REGION", options.env.as_ref()))
        .or_else(|| {
            crate::utils::provider_env::get_provider_env_value("AWS_DEFAULT_REGION", options.env.as_ref())
        })
}

/// The region resolution, upstream's precedence table: the ARN-embedded
/// region, then the configured region, then the endpoint's region when the
/// endpoint is pinned, then `us-east-1` only when no ambient profile rides.
#[must_use]
pub fn resolve_region(model: &Model, options: &BedrockStreamOptions) -> String {
    if let Some(arn_region) = arn_region(&model.id) {
        return arn_region;
    }
    if let Some(region) = configured_region(options) {
        return region;
    }
    let ambient_profile = has_ambient_configured_profile(options);
    if should_use_explicit_bedrock_endpoint(model, None, ambient_profile)
        && let Some(region) = standard_endpoint_region(&model.base_url)
    {
        return region;
    }
    if ambient_profile {
        // No silent default when an ambient profile could redirect.
        String::new()
    } else {
        "us-east-1".to_owned()
    }
}

/// The region embedded in an inference-profile ARN, upstream's regex
/// `arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):`, which covers the
/// GovCloud partition spellings too.
fn arn_region(model_id: &str) -> Option<String> {
    let rest = model_id
        .strip_prefix("arn:aws-us-gov:bedrock:")
        .or_else(|| model_id.strip_prefix("arn:aws:bedrock:"))?;
    let region: String = rest
        .chars()
        .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-')
        .collect();
    (!region.is_empty()).then_some(region)
}

/// The pinned endpoint when the model's base URL should ride verbatim,
/// upstream's endpoint pinning decision.
#[must_use]
pub fn resolve_endpoint(model: &Model, options: &BedrockStreamOptions) -> Option<String> {
    let ambient_profile = has_ambient_configured_profile(options);
    if should_use_explicit_bedrock_endpoint(model, configured_region(options).as_deref(), ambient_profile)
    {
        Some(model.base_url.clone())
    } else {
        None
    }
}

/// The profile the client configures, upstream's `config.profile`.
#[must_use]
pub fn resolve_profile(options: &BedrockStreamOptions) -> Option<String> {
    options
        .profile
        .clone()
        .or_else(|| crate::utils::provider_env::get_provider_env_value("AWS_PROFILE", options.env.as_ref()))
}

/// The env access keys, upstream's `getConfiguredBedrockCredentials`: both
/// required, the session token optional.
fn env_credentials(options: &BedrockStreamOptions) -> Option<BedrockCredentials> {
    let access_key_id = crate::utils::provider_env::get_provider_env_value(
        "AWS_ACCESS_KEY_ID",
        options.env.as_ref(),
    )?;
    let secret_access_key = crate::utils::provider_env::get_provider_env_value(
        "AWS_SECRET_ACCESS_KEY",
        options.env.as_ref(),
    )?;
    Some(BedrockCredentials {
        access_key_id,
        secret_access_key,
        session_token: crate::utils::provider_env::get_provider_env_value(
            "AWS_SESSION_TOKEN",
            options.env.as_ref(),
        ),
    })
}

/// The bearer token the request carries, upstream's resolution: the explicit
/// option, then the shared API-key option, then the environment.
fn resolve_bearer_token(options: &BedrockStreamOptions) -> Option<String> {
    options
        .bearer_token
        .clone()
        .or_else(|| options.api_key.clone())
        .or_else(|| {
            crate::utils::provider_env::get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", options.env.as_ref())
        })
}

/// Resolve the client configuration, upstream's client-config construction:
/// profile precedence over ambient keys, env credentials only without a
/// profile, the skip-auth dummy pair, the bearer-token path, and the proxy.
#[must_use]
pub fn resolve_client_config(model: &Model, options: &BedrockStreamOptions) -> BedrockClientConfig {
    let skip_auth = crate::utils::provider_env::get_provider_env_value(
        "AWS_BEDROCK_SKIP_AUTH",
        options.env.as_ref(),
    )
    .as_deref()
        == Some("1");
    let profile = resolve_profile(options);

    let credentials = if skip_auth || profile.is_some() {
        // A configured profile must beat ambient access keys: the SDK default
        // chain would ignore profiles once credentials ride the config.
        None
    } else {
        env_credentials(options)
    };

    let token = if skip_auth {
        None
    } else {
        resolve_bearer_token(options).filter(|token| !token.is_empty())
    };
    let auth_scheme_preference = token.as_ref().map(|_| "httpBearerAuth".to_owned());

    let proxy_url = crate::utils::node_http_proxy::resolve_http_proxy_url_for_target(
        &model.base_url,
        options.env.as_ref(),
    )
    .ok()
    .flatten()
    .map(|url| url.to_string());
    let force_http1 = crate::utils::provider_env::get_provider_env_value(
        "AWS_BEDROCK_FORCE_HTTP1",
        options.env.as_ref(),
    )
    .as_deref()
        == Some("1");

    BedrockClientConfig {
        region: Some(resolve_region(model, options)).filter(|region| !region.is_empty()),
        endpoint: resolve_endpoint(model, options),
        profile,
        credentials,
        skip_auth,
        token,
        auth_scheme_preference,
        headers: custom_headers(options),
        proxy_url,
        force_http1,
    }
}

/// The caller headers the pre-signing middleware injects: reserved headers
/// (`authorization`, `host`, any `x-amz-*`) skip case-insensitively,
/// upstream's custom-headers middleware rule.
#[must_use]
pub fn custom_headers(options: &BedrockStreamOptions) -> Vec<(String, String)> {
    let Some(record) = crate::utils::headers::provider_headers_to_record(options.headers.as_ref()) else {
        return Vec::new();
    };
    record
        .into_iter()
        .filter(|(name, _)| !is_reserved_header(&name))
        .collect()
}

/// The reserved-header rule, upstream's `isReservedHeader`.
#[must_use]
pub fn is_reserved_header(name: &str) -> bool {
    let lowered = name.to_lowercase();
    lowered.starts_with("x-amz-") || lowered == "authorization" || lowered == "host"
}

/// The command input the SDK adapter sends, upstream's `commandInput`:
/// `modelId`, the converted messages and system, the inference config, the
/// tool config, the additional model request fields, and the cost-allocation
/// metadata.
///
/// # Errors
/// When the message or tool conversions fail (image types, strict sampling).
pub fn build_command_input(
    model: &Model,
    context: &Context,
    options: &BedrockStreamOptions,
    cache_retention: crate::types::CacheRetention,
) -> Result<Value, String> {
    let env = options.env.as_ref();
    let supports_strict = supports_strict_mode(model);
    let tool_choice = options.tool_choice.as_ref();

    let mut input = serde_json::Map::new();
    input.insert("modelId".to_owned(), json!(model.id));
    input.insert(
        "messages".to_owned(),
        json!(convert_messages(context, model, cache_retention, env)?),
    );
    if let Some(system) = build_system_prompt(context.system_prompt.as_deref(), model, cache_retention, env) {
        input.insert("system".to_owned(), json!(system));
    }
    let mut inference_config = serde_json::Map::new();
    let inference_max_tokens = options
        .max_tokens
        .or_else(|| is_anthropic_claude_model(model).then_some(model.max_tokens));
    if let Some(max_tokens) = inference_max_tokens {
        inference_config.insert("maxTokens".to_owned(), json!(max_tokens));
    }
    if let Some(temperature) = options.temperature {
        inference_config.insert("temperature".to_owned(), json!(temperature));
    }
    if !inference_config.is_empty() {
        input.insert("inferenceConfig".to_owned(), Value::Object(inference_config));
    }
    if let Some(tool_config) = convert_tool_config(context.tools.as_deref().unwrap_or_default(), tool_choice, supports_strict)? {
        input.insert("toolConfig".to_owned(), tool_config);
    }
    if let Some(additional) = build_additional_model_request_fields(
        model,
        options.reasoning,
        options.thinking_budgets.as_ref(),
        options.interleaved_thinking,
        options.thinking_display.as_deref(),
        configured_region(options).as_deref(),
    ) {
        input.insert("additionalModelRequestFields".to_owned(), additional);
    }
    if let Some(metadata) = &options.request_metadata {
        let metadata_object = metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<serde_json::Map<String, Value>>();
        input.insert("requestMetadata".to_owned(), Value::Object(metadata_object));
    }
    Ok(Value::Object(input))
}

/// The cache retention the request uses, upstream's `resolveCacheRetention`:
/// the explicit option wins, then the `PI_CACHE_RETENTION` env opt-in, then
/// the short default.
#[must_use]
pub fn resolve_cache_retention(
    options: &BedrockStreamOptions,
) -> crate::types::CacheRetention {
    if let Some(cache_retention) = options.cache_retention {
        return cache_retention;
    }
    if crate::utils::provider_env::get_provider_env_value("PI_CACHE_RETENTION", options.env.as_ref())
        .as_deref()
        == Some("long")
    {
        return crate::types::CacheRetention::Long;
    }
    crate::types::CacheRetention::Short
}