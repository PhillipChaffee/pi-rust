//! The OAuth flow loader seam and the per-provider flows, ported from
//! `packages/ai/src/auth/oauth/load.ts` and `packages/ai/src/auth/oauth/` at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! One flow per provider subscription login: PKCE-with-callback for
//! Anthropic, OpenAI Codex, OpenRouter, and Radius; device-code for xAI,
//! GitHub Copilot, and Kimi Code. The `HttpClient` seam replaces ambient
//! `fetch`, the epoch clock replaces `Date.now()`, and the loopback
//! callback servers share [`callback`].
//!
//! Porting restatements this module tree records:
//!
//! - `load.ts`'s bundler-opaque dynamic imports have no Rust counterpart —
//!   the flows are statically linked, so the loaders are direct constructors
//!   and the `registerBundledOAuthFlowLoaders` hook collapses with the
//!   providers' `lazyOAuth` wrappers.
//! - The per-flow `node:http` callback servers share [`callback`]'s
//!   hand-rolled responder; each flow keeps its routing and settle rules.
//! - `PI_OAUTH_CALLBACK_HOST` and `KIMI_CODE_OAUTH_HOST`/`KIMI_OAUTH_HOST`
//!   still resolve through [`crate::utils::provider_env`], like upstream's
//!   `getProviderEnvValue` reads.
//! - The flows build the closure-based [`OAuthAuth`](crate::auth::types::OAuthAuth)
//!   the merged auth core resolves through: each flow type keeps its
//!   `login`/`refresh`/`to_auth` logic as inherent methods and
//!   [`FlowType::auth`] wires them into the callback fields.

pub mod anthropic;
pub mod callback;
pub mod device_code;
pub mod github_copilot;
pub mod kimi_coding;
pub mod oauth_page;
pub mod openai_codex;
pub mod openrouter;
pub mod pkce;
pub mod radius;
pub mod xai;

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auth::clock::SystemClock;
use crate::auth::oauth::anthropic::AnthropicOAuth;
use crate::auth::oauth::github_copilot::{GitHubCopilotOAuth, KnownModels};
use crate::auth::oauth::kimi_coding::KimiCodingOAuth;
use crate::auth::oauth::openai_codex::OpenAICodexOAuth;
use crate::auth::oauth::openrouter::OpenRouterOAuth;
use crate::auth::oauth::radius::RadiusOAuth;
use crate::auth::oauth::xai::XaiOAuth;
use crate::auth::types::{AuthError, OAuthAuth, OAuthCredentials};
use crate::http::{HttpError, HttpMethod, HttpRequest};
use crate::types::{BoxedFuture, JsonValue};

/// The failure an OAuth flow step reports, upstream's thrown plain `Error`s:
/// the message is the contract the flows' rejections carry.
#[derive(Debug)]
pub(crate) struct FlowError(String);

impl std::fmt::Display for FlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FlowError {}

/// The auth error a flow step rejects with: a boxed message error whose
/// `Display` is the reason.
pub(crate) fn auth_error(message: impl Into<String>) -> AuthError {
    Box::new(FlowError(message.into()))
}

/// An OAuth credential with the three wire fields set, the shape every flow
/// mints and rotates.
#[must_use]
pub(crate) fn oauth_credentials(
    access: impl Into<String>,
    refresh: impl Into<String>,
    expires: i64,
) -> OAuthCredentials {
    OAuthCredentials {
        refresh: refresh.into(),
        access: access.into(),
        expires,
        extra: BTreeMap::new(),
    }
}

/// An extra field held as a JSON string, the shape flows store and read
/// `scope`, `enterpriseUrl`, and `accountId` with.
pub(crate) fn extra_string<'a>(
    extra: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> Option<&'a str> {
    extra.get(key).and_then(JsonValue::as_str)
}

/// The options of [`load_radius_oauth`], upstream's loadRadiusOAuth argument.
#[derive(Debug)]
pub struct RadiusOAuthOptions {
    /// Display name.
    pub name: String,
    /// Gateway URL the flow authenticates against.
    pub gateway: String,
}

/// Loads the Anthropic (Claude Pro/Max) OAuth flow, upstream's
/// `loadAnthropicOAuth`.
#[must_use]
pub fn load_anthropic_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move {
        AnthropicOAuth::new(crate::http::default_http_client(), Arc::new(SystemClock)).auth()
    })
}

/// Loads the OpenAI (`ChatGPT` Plus/Pro) OAuth flow, upstream's
/// `loadOpenAICodexOAuth`.
#[must_use]
pub fn load_openai_codex_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move {
        OpenAICodexOAuth::new(crate::http::default_http_client(), Arc::new(SystemClock)).auth()
    })
}

/// Loads the GitHub Copilot OAuth flow, upstream's `loadGitHubCopilotOAuth`,
/// with the known-model membership its policy updates check derived from the
/// committed `github-copilot` catalog.
#[must_use]
pub fn load_github_copilot_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move {
        GitHubCopilotOAuth::new(
            crate::http::default_http_client(),
            Arc::new(SystemClock),
            copilot_known_models(),
        )
        .auth()
    })
}

/// The known-model membership from the generated `github-copilot` catalog,
/// upstream's `Object.hasOwn(GITHUB_COPILOT_MODELS, id)` test: a model id is
/// known when the builtin catalog lists it under any API group.
fn copilot_known_models() -> KnownModels {
    Arc::new(|model_id: &str| {
        crate::providers::catalog::get_builtin_model("github-copilot", model_id).is_some()
    })
}

/// Loads the OpenRouter OAuth flow, upstream's `loadOpenRouterOAuth`.
#[must_use]
pub fn load_openrouter_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move {
        OpenRouterOAuth::new(crate::http::default_http_client()).auth()
    })
}

/// Loads the Kimi Code (subscription) OAuth flow, upstream's
/// `loadKimiCodingOAuth`.
#[must_use]
pub fn load_kimi_coding_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move { KimiCodingOAuth::new(crate::http::default_http_client()).auth() })
}

/// Loads the xAI (Grok/X subscription) OAuth flow, upstream's `loadXaiOAuth`.
#[must_use]
pub fn load_xai_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move {
        XaiOAuth::new(crate::http::default_http_client(), Arc::new(SystemClock)).auth()
    })
}

/// Loads the Radius OAuth flow, upstream's `loadRadiusOAuth`.
#[must_use]
pub fn load_radius_oauth(options: &RadiusOAuthOptions) -> BoxedFuture<'static, OAuthAuth> {
    let name = options.name.clone();
    let gateway = options.gateway.clone();
    Box::pin(async move {
        RadiusOAuth::new(name, gateway, crate::http::default_http_client(), Arc::new(SystemClock))
            .auth()
    })
}

/// Build a form-encoded POST request, upstream's
/// `fetch(url, { method: "POST", headers, body: new URLSearchParams(..) })`.
#[must_use]
pub(crate) fn form_post_request(
    url: &str,
    headers: &[(&str, &str)],
    body: &[(String, String)],
    signal: tokio_util::sync::CancellationToken,
    timeout_ms: Option<u64>,
) -> HttpRequest {
    let encoded = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(
            body.iter()
                .map(|(key, value)| (key.as_str(), value.as_str())),
        )
        .finish();
    let mut header_pairs: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    header_pairs.push((
        "Content-Type".to_owned(),
        "application/x-www-form-urlencoded".to_owned(),
    ));
    HttpRequest {
        method: HttpMethod::Post,
        url: url.to_owned(),
        headers: header_pairs,
        body: Some(bytes::Bytes::from(encoded)),
        timeout_ms,
        signal,
    }
}

/// Build a JSON POST request, upstream's `fetch(url, { body: JSON.stringify(..) })`.
#[must_use]
pub(crate) fn json_post_request(
    url: &str,
    headers: &[(&str, &str)],
    body: &serde_json::Value,
    signal: tokio_util::sync::CancellationToken,
    timeout_ms: Option<u64>,
) -> HttpRequest {
    let mut header_pairs: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    header_pairs.push(("Content-Type".to_owned(), "application/json".to_owned()));
    HttpRequest {
        method: HttpMethod::Post,
        url: url.to_owned(),
        headers: header_pairs,
        body: Some(serde_json::to_vec(body).unwrap_or_default().into()),
        timeout_ms,
        signal,
    }
}

/// Send one request through the seam, mapping transport failures to the
/// auth error the flows raise.
///
/// # Errors
/// The transport's own failure message; cancellation surfaces as
/// [`HttpError::Aborted`], timeouts as [`HttpError::Timeout`].
pub(crate) async fn execute(
    client: &Arc<dyn crate::http::HttpClient>,
    request: HttpRequest,
) -> Result<crate::http::HttpResponse, AuthError> {
    client.execute(request).await.map_err(|error| {
        auth_error(match &error {
            HttpError::Aborted => device_code::LOGIN_CANCELLED_MESSAGE.to_owned(),
            other => other.to_string(),
        })
    })
}

/// Read one response body to completion and parse it as JSON.
///
/// # Errors
/// Rejects when the body cannot be read or the JSON does not parse.
pub(crate) async fn read_json(
    response: &mut crate::http::HttpResponse,
) -> Result<serde_json::Value, AuthError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .body
        .next_chunk()
        .await
        .map_err(|error| auth_error(error.to_string()))?
    {
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| auth_error(format!("invalid JSON response: {error}")))
}

/// Read one response body to a string, swallowing read failures with an
/// empty body — upstream's `response.text().catch(() => "")`.
pub(crate) async fn read_body_lossy(response: &mut crate::http::HttpResponse) -> String {
    let mut bytes = Vec::new();
    while let Ok(Some(chunk)) = response.body.next_chunk().await {
        bytes.extend_from_slice(&chunk);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Parse one JSON body that must be a JSON object, upstream's
/// `json && typeof json === "object" && !Array.isArray(json)` guard.
///
/// # Errors
/// Rejects with `message` when the body is not an object.
pub(crate) fn json_object(
    value: serde_json::Value,
    message: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, AuthError> {
    match value {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(auth_error(message.to_owned())),
    }
}

/// A JSON object field that must be a non-empty string, upstream's
/// `requiredString`.
///
/// # Errors
/// Rejects with the field's invalid-field message.
pub(crate) fn required_string(
    body: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<String, AuthError> {
    match body.get(field) {
        Some(serde_json::Value::String(value)) if !value.is_empty() => Ok(value.clone()),
        _ => Err(auth_error(format!("Invalid OAuth response field: {field}"))),
    }
}

/// The wire's seconds count as a whole number, the float-to-int step the
/// flows' `expires_in`/`interval` arithmetic shares: every call site
/// validates the value is positive-finite immediately above the call, so
/// truncation drops only a fractional part and saturation only an extreme
/// the wire never sends.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "call sites validate the value positive-finite immediately above the cast, so truncation drops only a fractional part"
)]
#[must_use]
pub(crate) const fn wire_seconds_to_u64(value: f64) -> u64 {
    value as u64
}

/// The wire's `expires_in` seconds as a signed epoch offset, upstream's
/// `Date.now() + expiresIn * 1000` arithmetic: float-to-int `as` saturates,
/// so a malformed (NaN or extreme) wire value clamps at the epoch bounds
/// instead of corrupting the expiry.
#[expect(
    clippy::cast_possible_truncation,
    reason = "float-to-int `as` saturates NaN and out-of-range wire values at the epoch bounds, keeping the expiry arithmetic bounded"
)]
#[must_use]
pub(crate) const fn wire_seconds_to_i64(value: f64) -> i64 {
    value as i64
}

/// Parse a pasted authorization-code input into code and state, the shared
/// shape of `anthropic.ts`'s and `openai-codex.ts`'s
/// `parseAuthorizationInput`: a URL's query, then a `#` fragment, then a
/// bare `code=` form, then the input itself as the code.
#[must_use]
pub(crate) fn parse_authorization_code_state(input: &str) -> (Option<String>, Option<String>) {
    let value = input.trim();
    if value.is_empty() {
        return (None, None);
    }

    if let Ok(url) = url::Url::parse(value) {
        return (query_value(&url, "code"), query_value(&url, "state"));
    }

    if let Some((code, state)) = value.split_once('#') {
        return (Some(code.to_owned()), Some(state.to_owned()));
    }

    if value.contains("code=") {
        let pairs = url::form_urlencoded::parse(value.as_bytes()).collect::<Vec<_>>();
        let code = pairs
            .iter()
            .find(|(key, _)| key == "code")
            .map(|(_, value)| value.clone().into_owned());
        let state = pairs
            .iter()
            .find(|(key, _)| key == "state")
            .map(|(_, value)| value.clone().into_owned());
        return (code, state);
    }

    (Some(value.to_owned()), None)
}

fn query_value(url: &url::Url, name: &str) -> Option<String> {
    url.query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

/// A random RFC 4122 version-4 UUID, upstream's `crypto.randomUUID`.
pub(crate) fn random_uuid_v4() -> Result<String, AuthError> {
    let mut bytes = [0_u8; 16];
    getrandom::fill(&mut bytes).map_err(|error| auth_error(format!("getrandom failed: {error}")))?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Ok(format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    ))
}