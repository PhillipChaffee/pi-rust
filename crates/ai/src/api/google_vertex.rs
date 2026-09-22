//! The Google Vertex AI wire API, ported from
//! `packages/ai/src/api/google-vertex.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//! - Upstream hands the whole transport to `@google/genai`'s Vertex client;
//!   here the request is raw wire: `POST {url}:streamGenerateContent?alt=sse`
//!   over the transport seam. Express-mode API keys ride the
//!   `x-goog-api-key` header exactly as the SDK sends them; ADC requests
//!   carry `Authorization: Bearer` plus `x-goog-user-project` when the
//!   credentials name a quota project.
//! - The SDK's ADC machinery (google-auth-library) is ported: a
//!   `GOOGLE_APPLICATION_CREDENTIALS` service account signs a scoped RS256
//!   JWT and exchanges it at `oauth2.googleapis.com/token`; an
//!   `authorized_user` file runs the refresh grant; with no credentials file
//!   the gcloud well-known ADC file is read, and with none of those the GCE
//!   metadata server is tried. Token caching collapses to acquisition per
//!   stream, matching the port's per-request client construction.
//! - `ResourceScope.COLLECTION` means one thing on the wire: with a custom
//!   base URL the `projects/{p}/locations/{l}` prefix is suppressed, and the
//!   version segment drops when the base URL already carries one.
//! - The generated catalog's `{location}` placeholder base URLs are never
//!   forwarded, upstream's `resolveCustomBaseUrl`.
//! - The `THINKING_LEVEL_MAP` indirection vanishes: the level value is the
//!   wire string already.
//! - The identical chunk loop the two upstream files carry lives in
//!   [`consume_google_stream`](crate::api::google_shared); each upstream
//!   adapter kept its own copy because the SDK owned the loop, and here the
//!   duplication gate pins one substrate.
//! - Surrogate sanitization is statically upheld: Rust strings are valid
//!   UTF-8, so `sanitizeSurrogates` has no work.

use std::sync::Arc;

use bytes::Bytes;
use serde_json::{Value, json};

use crate::api::google_generative_ai::to_wire_body;
use crate::api::google_shared::{
    GoogleThinkingControl, ResolvedGoogleThinkingLevel, api_error_message, convert_messages,
    convert_tools, resolve_google_function_calling_mode, resolve_google_thinking_level,
    supports_google_strict_tool_sampling,
};
use crate::api::simple_options::build_base_options;
use crate::http::client::{HttpMethod, HttpRequest, read_body_text};
use crate::types::{
    AssistantMessage, AssistantMessageEvent, Context, Model, ModelThinkingLevel, ProviderStreams,
    SimpleStreamOptions, StopReason, StreamOptions, ThinkingBudgets,
};
use crate::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use crate::utils::headers::{headers_to_record, provider_headers_to_record};
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::provider_retry::{
    ProviderRequestError, ProviderRetryOptions, retry_provider_request,
};

/// The Vertex REST version the SDK client pins.
const API_VERSION: &str = "v1";
/// The ambient-credential marker the auth layer stores instead of a key.
const GCP_VERTEX_CREDENTIALS_MARKER: &str = "gcp-vertex-credentials";
/// The ADC well-known file, upstream's `VERTEX_ADC_PATH` constant.
const VERTEX_ADC_PATH: &str = "~/.config/gcloud/application_default_credentials.json";
/// The cloud-platform scope the SDK pins for Vertex.
const VERTEX_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// The adapter-facing options, upstream's `GoogleVertexOptions`.
#[derive(Clone, Debug, Default)]
pub struct GoogleVertexStreamOptions {
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
    /// The GCP project id.
    pub project: Option<String>,
    /// The GCP region, e.g. `us-central1`.
    pub location: Option<String>,
}

impl From<StreamOptions> for GoogleVertexStreamOptions {
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
            project: None,
            location: None,
        }
    }
}

/// The Google Vertex streams, upstream's `googleVertexApi()`.
#[derive(Debug, Default)]
pub struct GoogleVertexStreams;

impl ProviderStreams for GoogleVertexStreams {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let options = options.cloned().map(GoogleVertexStreamOptions::from);
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
    options: Option<&GoogleVertexStreamOptions>,
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
/// resolves the pi reasoning level to either a provider-native level
/// (Gemini 3) or a token budget, and disables thinking outright when no
/// level is requested. Auth resolves inside the stream, upstream-side too.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let mut base = GoogleVertexStreamOptions::from(build_base_options(model, context, options, None));
    base.tool_choice = options.and_then(|options| options.tool_choice).map(|choice| match choice {
        crate::types::ToolChoice::Auto => "auto".to_owned(),
        crate::types::ToolChoice::None => "none".to_owned(),
    });

    let Some(reasoning) = options.and_then(|options| options.reasoning) else {
        return stream(
            model,
            context,
            Some(&GoogleVertexStreamOptions {
                thinking: Some(GoogleThinkingControl {
                    enabled: false,
                    ..GoogleThinkingControl::default()
                }),
                ..base
            }),
        );
    };

    let clamped = crate::models::clamp_thinking_level(model, ModelThinkingLevel::from(reasoning));
    let resolved = match resolve_google_thinking_level(model, clamped) {
        Ok(level) => level,
        Err(message) => return setup_error_stream(model, &message),
    };

    if is_gemini3_pro_model(model) || is_gemini3_flash_model(model) {
        return stream(
            model,
            context,
            Some(&GoogleVertexStreamOptions {
                thinking: Some(GoogleThinkingControl {
                    enabled: true,
                    level: Some(get_gemini3_thinking_level(resolved, model).to_owned()),
                    budget_tokens: None,
                }),
                ..base
            }),
        );
    }

    stream(
        model,
        context,
        Some(&GoogleVertexStreamOptions {
            thinking: Some(GoogleThinkingControl {
                enabled: true,
                budget_tokens: Some(get_google_budget(
                    model,
                    resolved,
                    options.and_then(|options| options.thinking_budgets.as_ref()),
                )),
                level: None,
            }),
            ..base
        }),
    )
}

/// The fresh accumulator a stream starts from, upstream's `output`.
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

/// The resolved request identity: an express API key, or an ADC credential
/// resolved to a bearer token, carrying the project/location the URL needs.
enum VertexAuth {
    ApiKey(String),
    Adc {
        project: String,
        location: String,
        token: String,
        /// The credentials' quota project, sent as `x-goog-user-project`.
        quota_project: Option<String>,
    },
}

/// Resolve the request identity, upstream's `resolveApiKey` + client
/// selection. The ADC token acquisition runs here too; upstream it happens
/// lazily inside the SDK's send.
async fn resolve_auth(_model: &Model, options: &GoogleVertexStreamOptions) -> Result<VertexAuth, String> {
    if let Some(api_key) = resolve_api_key(options) {
        return Ok(VertexAuth::ApiKey(api_key));
    }
    let project = resolve_project(options)?;
    let location = resolve_location(options)?;
    let (token, quota_project) = resolve_adc_token(options).await?;
    Ok(VertexAuth::Adc {
        project,
        location,
        token,
        quota_project,
    })
}

/// The API key when it is a real key: empty strings, the ambient-credential
/// marker, and `<placeholder>` keys fall through to ADC.
fn resolve_api_key(options: &GoogleVertexStreamOptions) -> Option<String> {
    let api_key = options.api_key.as_deref()?.trim();
    if api_key.is_empty()
        || api_key == GCP_VERTEX_CREDENTIALS_MARKER
        || is_placeholder_api_key(api_key)
    {
        return None;
    }
    Some(api_key.to_owned())
}

/// `/^<[^>]+>$/`: the ambient auth markers the credential layer store.
fn is_placeholder_api_key(api_key: &str) -> bool {
    api_key.len() >= 3
        && api_key.starts_with('<')
        && api_key.ends_with('>')
        && !api_key[1..api_key.len() - 1].contains('>')
}

fn resolve_project(options: &GoogleVertexStreamOptions) -> Result<String, String> {
    let project = options
        .project
        .clone()
        .filter(|project| !project.is_empty())
        .or_else(|| get_provider_env_value("GOOGLE_CLOUD_PROJECT", options.env.as_ref()))
        .or_else(|| get_provider_env_value("GCLOUD_PROJECT", options.env.as_ref()));
    project.ok_or_else(|| {
        "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
            .to_owned()
    })
}

fn resolve_location(options: &GoogleVertexStreamOptions) -> Result<String, String> {
    let location = options
        .location
        .clone()
        .filter(|location| !location.is_empty())
        .or_else(|| get_provider_env_value("GOOGLE_CLOUD_LOCATION", options.env.as_ref()));
    location.ok_or_else(|| {
        "Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options."
            .to_owned()
    })
}

/// Resolve the ADC bearer token, upstream's google-auth-library chain:
/// `GOOGLE_APPLICATION_CREDENTIALS` (service account or authorized user),
/// then the gcloud well-known ADC file, then the GCE metadata server.
/// Returns the token and the credentials' quota project when known.
async fn resolve_adc_token(
    options: &GoogleVertexStreamOptions,
) -> Result<(String, Option<String>), String> {
    let key_file = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", options.env.as_ref())
        .unwrap_or_else(|| VERTEX_ADC_PATH.to_owned());
    if let Some(path) = expand_home(&key_file)
        && let Ok(contents) = std::fs::read_to_string(&path)
        && let Ok(credentials) = serde_json::from_str::<Value>(&contents)
    {
        return acquire_file_credentials(options, &credentials).await;
    }
    acquire_metadata_server_token(options).await.map(|token| (token, None))
}

fn expand_home(path: &str) -> Option<std::path::PathBuf> {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = std::env::home_dir()?;
        return Some(home.join(rest));
    }
    Some(std::path::PathBuf::from(path))
}

/// Exchange a credentials-file entry for a bearer token, upstream's
/// google-auth-library `fromJSON` dispatch.
async fn acquire_file_credentials(
    options: &GoogleVertexStreamOptions,
    credentials: &Value,
) -> Result<(String, Option<String>), String> {
    match credentials.get("type").and_then(Value::as_str) {
        Some("service_account") => {
            let client_email = credentials
                .get("client_email")
                .and_then(Value::as_str)
                .ok_or("Google service-account credentials are missing client_email")?;
            let private_key = credentials
                .get("private_key")
                .and_then(Value::as_str)
                .ok_or("Google service-account credentials are missing private_key")?;
            let quota_project = credentials
                .get("quota_project_id")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let token = exchange_service_account_jwt(options, client_email, private_key).await?;
            Ok((token, quota_project))
        }
        Some("authorized_user") => {
            let client_id = credentials
                .get("client_id")
                .and_then(Value::as_str)
                .ok_or("Google authorized-user credentials are missing client_id")?;
            let client_secret = credentials
                .get("client_secret")
                .and_then(Value::as_str)
                .ok_or("Google authorized-user credentials are missing client_secret")?;
            let refresh_token = credentials
                .get("refresh_token")
                .and_then(Value::as_str)
                .ok_or("Google authorized-user credentials are missing refresh_token")?;
            let token = exchange_refresh_token(options, client_id, client_secret, refresh_token)
                .await?;
            Ok((token, None))
        }
        other => Err(format!(
            "Unsupported Google credentials type: {}",
            other.unwrap_or("unknown")
        )),
    }
}

/// Sign the scoped RS256 JWT and exchange it for an access token, upstream's
/// gtoken flow: `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer`.
async fn exchange_service_account_jwt(
    options: &GoogleVertexStreamOptions,
    client_email: &str,
    private_key: &str,
) -> Result<String, String> {
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};

    let now = crate::auth::resolve::now_ms() / 1000;
    let claims = json!({
        "iss": client_email,
        "scope": VERTEX_SCOPE,
        "aud": "https://oauth2.googleapis.com/token",
        "exp": now + 3600,
        "iat": now,
    });
    let key = EncodingKey::from_rsa_pem(private_key.as_bytes())
        .map_err(|error| format!("Google service-account private key is not usable: {error}"))?;
    let jwt = encode(&Header::new(Algorithm::RS256), &claims, &key)
        .map_err(|error| format!("Google service-account JWT signing failed: {error}"))?;
    let body = format!(
        "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion={jwt}"
    );
    token_request(options, &body)
        .await
        .map(|response| response.access_token)
}

/// The authorized-user refresh grant, upstream's oauth2client refresh path.
async fn exchange_refresh_token(
    options: &GoogleVertexStreamOptions,
    client_id: &str,
    client_secret: &str,
    refresh_token: &str,
) -> Result<String, String> {
    let body = format!(
        "grant_type=refresh_token&refresh_token={refresh}&client_id={id}&client_secret={secret}",
        refresh = urlencode(refresh_token),
        id = urlencode(client_id),
        secret = urlencode(client_secret),
    );
    token_request(options, &body)
        .await
        .map(|response| response.access_token)
}

/// The GCE metadata-server token endpoint, upstream's ComputeClient path.
async fn acquire_metadata_server_token(options: &GoogleVertexStreamOptions) -> Result<String, String> {
    let url = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token?scopes=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform";
    let http_client = options.transport_options.client();
    let request = HttpRequest {
        method: HttpMethod::Get,
        url: url.to_owned(),
        headers: vec![("Metadata-Flavor".to_owned(), "Google".to_owned())],
        body: None,
        timeout_ms: options.timeout_ms,
        signal: options.transport_options.signal(),
    };
    let response = http_client
        .execute(request)
        .await
        .map_err(|error| format!("Google metadata server token request failed: {error}"))?;
    if !(200..300).contains(&response.status) {
        return Err(format!(
            "Google metadata server returned status {}",
            response.status
        ));
    }
    let body = read_body_text(response.body)
        .await
        .map_err(|error| format!("Google metadata server token read failed: {error}"))?;
    let parsed: Value = serde_json::from_str(&body)
        .map_err(|error| format!("Google metadata server token is not JSON: {error}"))?;
    parsed
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "Google metadata server token response carries no access_token".to_owned())
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
}

/// POST the token endpoint with a form-encoded grant body, the exchange both
/// credential shapes share.
async fn token_request(
    options: &GoogleVertexStreamOptions,
    body: &str,
) -> Result<TokenResponse, String> {
    let http_client = options.transport_options.client();
    let request = HttpRequest {
        method: HttpMethod::Post,
        url: "https://oauth2.googleapis.com/token".to_owned(),
        headers: vec![
            (
                "content-type".to_owned(),
                "application/x-www-form-urlencoded".to_owned(),
            ),
            ("User-Agent".to_owned(), get_pi_user_agent()),
        ],
        body: Some(Bytes::from(body.to_owned())),
        timeout_ms: options.timeout_ms,
        signal: options.transport_options.signal(),
    };
    let response = http_client
        .execute(request)
        .await
        .map_err(|error| format!("Google token exchange failed: {error}"))?;
    if !(200..300).contains(&response.status) {
        let body_text = read_body_text(response.body).await.unwrap_or_default();
        return Err(format!(
            "Google token exchange returned status {}: {body_text}",
            response.status
        ));
    }
    let body = read_body_text(response.body)
        .await
        .map_err(|error| format!("Google token exchange read failed: {error}"))?;
    serde_json::from_str(&body)
        .map_err(|error| format!("Google token exchange response is not JSON: {error}"))
}

/// `%`-encode the characters a token-endpoint form body must not carry raw;
/// the form encoding upstream delegates to URLSearchParams.
fn urlencode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

async fn run_stream(
    model: &Model,
    context: &Context,
    options: &GoogleVertexStreamOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let auth = resolve_auth(model, options).await?;
    let mut params = build_params(model, context, options)?;
    if let Some(hook) = &options.transport_options.on_payload {
        params = hook
            .call(params.clone(), model.clone())
            .await
            .unwrap_or(params);
    }
    let response = dispatch_stream_request(model, &params, options, &auth).await?;
    events.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });
    crate::api::google_shared::consume_google_stream(model, response, output, events).await?;
    finish_stream(options, output, events)
}

/// Build and dispatch the stream request with the shared provider retry
/// policy, the seam-side port of upstream's `createClient` (express or ADC) +
/// `retryGoogleRequest(() => client.models.generateContentStream(params))`.
async fn dispatch_stream_request(
    model: &Model,
    params: &Value,
    options: &GoogleVertexStreamOptions,
    auth: &VertexAuth,
) -> Result<crate::http::client::HttpResponse, String> {
    let http_client = options.transport_options.client();
    let signal = options.transport_options.signal();
    let mut headers = build_request_headers(model, options, auth);
    if let VertexAuth::Adc {
        quota_project: Some(project),
        ..
    } = auth
    {
        headers.push(("x-goog-user-project".to_owned(), project.clone()));
    }
    let request = HttpRequest {
        method: HttpMethod::Post,
        url: stream_generate_url(model, auth),
        headers,
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

/// Assemble the request headers, upstream's `buildHttpOptions` header merge
/// plus the credential header appended only when the caller did not already
/// set it.
fn build_request_headers(
    model: &Model,
    options: &GoogleVertexStreamOptions,
    auth: &VertexAuth,
) -> Vec<(String, String)> {
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
    match auth {
        VertexAuth::ApiKey(api_key) => {
            if !headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("x-goog-api-key"))
            {
                headers.push(("x-goog-api-key".to_owned(), api_key.clone()));
            }
        }
        VertexAuth::Adc { token, .. } => {
            if !headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            {
                headers.push(("authorization".to_owned(), format!("Bearer {token}")));
            }
        }
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

/// The request URL, the SDK's URL construction for the Vertex backend: a
/// custom base URL suppresses the project/location prefix (the COLLECTION
/// resource scope) and drops the version segment when the URL already
/// carries one; otherwise the regional host plus `projects/.../locations/...`
/// rides for ADC and the plain host for API keys.
fn stream_generate_url(model: &Model, auth: &VertexAuth) -> String {
    let custom_base = resolve_custom_base_url(&model.base_url);
    let include_version = match &custom_base {
        Some(base) => !base_url_includes_api_version(base),
        None => true,
    };
    let has_custom_base = custom_base.is_some();
    let base = custom_base.unwrap_or_else(|| default_vertex_host(auth));
    let mut url = base;
    if include_version {
        url.push('/');
        url.push_str(API_VERSION);
    }
    url.push('/');
    // The COLLECTION resource scope suppresses the project/location prefix
    // whenever a custom base URL rides; only generated-URL requests carry it.
    if let VertexAuth::Adc { project, location, .. } = auth
        && !has_custom_base
    {
        url.push_str(&format!("projects/{project}/locations/{location}/"));
    }
    url.push_str(&t_model_vertex(&model.id));
    url.push_str(":streamGenerateContent?alt=sse");
    url
}

/// The default host for a generated-URL request, upstream's ApiClient
/// baseUrl selection: `us`/`eu` multi-regions use the `rep` host, `global`
/// and express keys use the plain host, everything else embeds the location.
fn default_vertex_host(auth: &VertexAuth) -> String {
    match auth {
        VertexAuth::ApiKey(_) => "https://aiplatform.googleapis.com".to_owned(),
        VertexAuth::Adc { location, .. } => match location.as_str() {
            "us" | "eu" => format!("https://aiplatform.{location}.rep.googleapis.com"),
            "global" => "https://aiplatform.googleapis.com".to_owned(),
            other => format!("https://{other}-aiplatform.googleapis.com"),
        },
    }
}

/// The SDK's Vertex `tModel`: bare ids ride under `publishers/google/`,
/// `publisher/model` ids under their publisher, already-prefixed ids pass
/// through.
fn t_model_vertex(id: &str) -> String {
    if id.starts_with("publishers/")
        || id.starts_with("projects/")
        || id.starts_with("models/")
    {
        return id.to_owned();
    }
    match id.split_once('/') {
        Some((publisher, model)) => format!("publishers/{publisher}/models/{model}"),
        None => format!("publishers/google/models/{id}"),
    }
}

/// A custom base URL the request may use, upstream's `resolveCustomBaseUrl`:
/// empty strings and the generated catalog's `{location}` placeholder are
/// not forwarded.
fn resolve_custom_base_url(base_url: &str) -> Option<String> {
    let trimmed = base_url.trim();
    if trimmed.is_empty() || trimmed.contains("{location}") {
        return None;
    }
    Some(trimmed.to_owned())
}

/// Whether the base URL already carries a `v<digits>[beta]` path segment,
/// upstream's `baseUrlIncludesApiVersion` (URL parse, regex fallback).
fn base_url_includes_api_version(base_url: &str) -> bool {
    if let Ok(url) = url::Url::parse(base_url) {
        return url.path().split('/').any(|part| {
            let Some(rest) = part.strip_prefix('v') else {
                return false;
            };
            let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
            if digits == 0 {
                return false;
            }
            rest[digits..]
                .strip_prefix("beta")
                .is_none_or(|tail| tail.chars().all(|c| c.is_ascii_digit()))
        });
    }
    base_url.split(|c: char| c == '/' || c == '?' || c == '#').any(|part| {
        let Some(rest) = part.strip_prefix('v') else {
            return false;
        };
        let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        digits > 0 && (rest.len() == digits + 1 || rest[digits..].starts_with("beta"))
    })
}

/// Build the SDK-style params, upstream's `buildParams`; identical to the
/// Gemini builder with the Vertex thinking-level vocabulary.
///
/// # Errors
/// When a tool requires strict sampling that cannot be resolved, or when the
/// request is already aborted.
fn build_params(
    model: &Model,
    context: &Context,
    options: &GoogleVertexStreamOptions,
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

/// The disabled-thinking config, upstream's Vertex `getDisabledThinkingConfig`
/// — no Gemma branch here: Vertex has no Gemma-4 catalog entries.
fn get_disabled_thinking_config(model: &Model) -> Value {
    if is_gemini3_pro_model(model) {
        json!({ "thinkingLevel": "LOW" })
    } else if is_gemini3_flash_model(model) {
        json!({ "thinkingLevel": "MINIMAL" })
    } else {
        json!({ "thinkingBudget": 0 })
    }
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

/// Map a resolved pi level to the provider-native thinking level, upstream's
/// `getGemini3ThinkingLevel` — the Gemini mapping without the Gemma branch.
#[must_use]
pub fn get_gemini3_thinking_level(
    effort: ResolvedGoogleThinkingLevel,
    model: &Model,
) -> &'static str {
    if is_gemini3_pro_model(model) {
        return match effort {
            ResolvedGoogleThinkingLevel::Minimal | ResolvedGoogleThinkingLevel::Low => "LOW",
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

/// The catalog's default thinking budgets, upstream's Vertex `getGoogleBudget`
/// — no flash-lite branch here.
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

fn provider_error_from_http(error: crate::http::client::HttpError) -> ProviderRequestError {
    match error {
        crate::http::client::HttpError::Aborted => ProviderRequestError::aborted(),
        crate::http::client::HttpError::Timeout => {
            ProviderRequestError::new(None, None, "request timed out")
        }
        crate::http::client::HttpError::Transport(message) => {
            ProviderRequestError::new(None, None, message)
        }
        crate::http::client::HttpError::InvalidUrl(url) => {
            ProviderRequestError::new(None, None, format!("invalid URL: {url}"))
        }
    }
}

/// The stream-setup failure as a settled error stream, upstream's
/// synchronous throw encoded per the stream contract.
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

/// The post-loop settlement, upstream's tail of the try block: the signal
/// check, the pending and provider-stopped rejections, then the done event.
fn finish_stream(
    options: &GoogleVertexStreamOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    if options.transport_options.signal().is_cancelled() {
        return Err("Request was aborted".to_owned());
    }
    if output.stop_reason == StopReason::Pending {
        return Err("Google Vertex stream ended without a finish reason".to_owned());
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