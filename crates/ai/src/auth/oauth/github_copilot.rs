//! The GitHub Copilot OAuth flow, ported from
//! `packages/ai/src/auth/oauth/github-copilot.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Device-code login against github.com or a GitHub Enterprise domain, the
//! Copilot internal token exchange, model-catalog parsing with policy
//! fallback, best-effort policy enablement, and the per-token proxy endpoint
//! derivation `to_auth` carries.
//!
//! Porting restatements this module records:
//!
//! - The membership check against the generated `GITHUB_COPILOT_MODELS`
//!   catalog rides [`KnownModels`]: the registry ticket wires the real model
//!   set; this module only needs the predicate.
//! - `fetchWithRateLimitRetry`'s retry budget aborts the in-flight request
//!   upstream (`AbortSignal.timeout`); the port checks the budget between
//!   attempts, so a request already started is allowed to finish.

use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::auth::oauth::device_code::{PollOptions, PollOutcome, poll_oauth_device_code_flow};
use crate::auth::oauth::{
    auth_error, execute, form_post_request, json_post_request, oauth_credentials, read_body_lossy,
    read_json,
};
use crate::auth::types::{AuthError, AuthEvent, AuthPrompt, OAuthCredentials};
use crate::http::{HttpClient, HttpMethod, HttpRequest, HttpResponse};
use crate::auth::types::ModelAuth;
use crate::types::BoxedFuture;
use crate::utils::sleep::sleep;

const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const COPILOT_HEADERS: &[(&str, &str)] = &[
    ("User-Agent", "GitHubCopilotChat/0.35.0"),
    ("Editor-Version", "vscode/1.107.0"),
    ("Editor-Plugin-Version", "copilot-chat/0.35.0"),
    ("Copilot-Integration-Id", "vscode-chat"),
];
const COPILOT_API_VERSION: &str = "2026-06-01";
const UNTRUSTED_VERIFICATION_URI: &str = "Untrusted verification_uri in device code response";
/// The per-request timeout of [`fetch_with_rate_limit_retry`], upstream's
/// `AbortSignal.timeout(5000)` per attempt.
const PER_REQUEST_TIMEOUT_MS: u64 = 5000;
/// The rate-limit status the retry loop waits on.
const RATE_LIMIT_STATUS: u16 = 429;
/// The default backoff before the first 429 retry, upstream's `500 * 2**retry`.
const RETRY_BASE_DELAY_MS: u64 = 500;
/// The Copilot access token's expiry reads `expires_at` minus five minutes.
const ACCESS_TOKEN_SKEW_MS: i64 = 5 * 60 * 1000;

/// The known-model membership the policy updates check, the port of the
/// generated `GITHUB_COPILOT_MODELS` catalog's `Object.hasOwn` test; the
/// registry ticket wires the real set.
pub type KnownModels = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// The GitHub Copilot OAuth flow, wired with its [`HttpClient`] seam, epoch
/// clock, and the known-model membership check.
pub struct GitHubCopilotOAuth {
    client: Arc<dyn HttpClient>,
    clock: Arc<dyn crate::auth::clock::AuthClock>,
    known_models: KnownModels,
}

impl std::fmt::Debug for GitHubCopilotOAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GitHubCopilotOAuth")
    }
}

impl GitHubCopilotOAuth {
    /// The flow over the given client, clock, and known-model membership.
    #[must_use]
    pub fn new(
        client: Arc<dyn HttpClient>,
        clock: Arc<dyn crate::auth::clock::AuthClock>,
        known_models: KnownModels,
    ) -> Self {
        Self {
            client,
            clock,
            known_models,
        }
    }

    /// The flow wired into the merged auth core's callback-based
    /// [`OAuthAuth`](crate::auth::types::OAuthAuth): the login, refresh, and
    /// derivation closures drive this flow's own client, clock, and
    /// known-model membership. (#29)
    #[must_use]
    pub fn auth(&self) -> crate::auth::types::OAuthAuth {
        let login: crate::auth::types::OAuthLoginFn = {
            let client = Arc::clone(&self.client);
            let clock = Arc::clone(&self.clock);
            let known_models = Arc::clone(&self.known_models);
            Arc::new(move |interaction| {
                let flow = Self {
                    client: Arc::clone(&client),
                    clock: Arc::clone(&clock),
                    known_models: Arc::clone(&known_models),
                };
                flow.login(interaction)
            })
        };
        let refresh: crate::auth::types::OAuthRefreshFn = {
            let client = Arc::clone(&self.client);
            let clock = Arc::clone(&self.clock);
            let known_models = Arc::clone(&self.known_models);
            Arc::new(move |credential, signal| {
                let flow = Self {
                    client: Arc::clone(&client),
                    clock: Arc::clone(&clock),
                    known_models: Arc::clone(&known_models),
                };
                flow.refresh(credential, signal)
            })
        };
        let to_auth: crate::auth::types::OAuthToAuthFn = {
            let client = Arc::clone(&self.client);
            let clock = Arc::clone(&self.clock);
            let known_models = Arc::clone(&self.known_models);
            Arc::new(move |credential| {
                let flow = Self {
                    client: Arc::clone(&client),
                    clock: Arc::clone(&clock),
                    known_models: Arc::clone(&known_models),
                };
                let auth = flow.to_auth(&credential);
                Box::pin(async move { Ok(auth) })
            })
        };
        crate::auth::types::OAuthAuth {
            name: "GitHub Copilot".to_owned(),
            is_subscription: Some(true),
            login_label: None,
            login,
            refresh,
            to_auth,
        }
    }

    /// Run the interactive login flow.
    pub fn login(
        &self,
        interaction: crate::auth::types::ProviderAuthInteraction,
    ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>> {
        let client = Arc::clone(&self.client);
        let clock = Arc::clone(&self.clock);
        let known_models = Arc::clone(&self.known_models);
        Box::pin(
            async move { login_github_copilot(&client, &clock, &known_models, &interaction).await },
        )
    }

    /// Exchange the refresh token for a rotated credential.
    pub fn refresh(
        &self,
        credential: OAuthCredentials,
        signal: CancellationToken,
    ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>> {
        let client = Arc::clone(&self.client);
        let clock = Arc::clone(&self.clock);
        let known_models = Arc::clone(&self.known_models);
        let refresh_token = credential.refresh.clone();
        let enterprise_domain = copilot_enterprise_domain(&credential);
        Box::pin(async move {
            refresh_github_copilot_token(
                &client,
                clock.as_ref(),
                &known_models,
                &refresh_token,
                enterprise_domain.as_deref(),
                &signal,
            )
            .await
        })
    }

    /// Derive the request auth from a valid credential: the bearer token plus
    /// the per-token proxy endpoint.
    #[must_use]
    pub fn to_auth(&self, credential: &OAuthCredentials) -> ModelAuth {
        ModelAuth {
            api_key: Some(credential.access.clone()),
            headers: None,
            base_url: Some(get_github_copilot_base_url(
                Some(&credential.access),
                copilot_enterprise_domain(credential).as_deref(),
            )),
        }
    }
}

/// The wire's JSON number as whole seconds, upstream's bare numeric coercion:
/// fractional seconds floor and non-finite or negative values coerce to zero,
/// which the poll loop's one-second minimum then clamps.
#[must_use]
pub(crate) const fn wire_seconds(value: f64) -> u64 {
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "the wire's seconds are small non-negative JSON numbers, coerced like upstream's JS arithmetic"
    )]
    let coerced = value.max(0.0) as u64;
    coerced
}

/// The wire's JSON number as whole milliseconds of epoch time, upstream's
/// bare numeric coercion.
#[must_use]
pub(crate) const fn wire_milliseconds(value: f64) -> i64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the wire's times are small JSON numbers coerced like upstream's JS arithmetic"
    )]
    let coerced = value as i64;
    coerced
}

/// The GitHub host a flow talks to, trimmed and reduced to the hostname,
/// upstream's `normalizeDomain`.
#[must_use]
fn normalize_domain(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let candidate = if trimmed.contains("://") {
        trimmed.to_owned()
    } else {
        format!("https://{trimmed}")
    };
    url::Url::parse(&candidate)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
}

/// The endpoints one GitHub domain serves, upstream's `getUrls`.
struct DeviceFlowUrls {
    device_code: String,
    access_token: String,
    copilot_token: String,
}

fn get_urls(domain: &str) -> DeviceFlowUrls {
    DeviceFlowUrls {
        device_code: format!("https://{domain}/login/device/code"),
        access_token: format!("https://{domain}/login/oauth/access_token"),
        copilot_token: format!("https://api.{domain}/copilot_internal/v2/token"),
    }
}

/// Parse the proxy endpoint from a Copilot token, upstream's
/// `getBaseUrlFromToken`: the `proxy-ep=([^;]+)` capture with a leading
/// `proxy.` rewritten to `api.`.
fn get_base_url_from_token(token: &str) -> Option<String> {
    let capture_start = token.find("proxy-ep=")? + "proxy-ep=".len();
    let rest = &token[capture_start..];
    let capture_end = rest.find(';').unwrap_or(rest.len());
    if capture_end == 0 {
        return None;
    }
    let proxy_host = &rest[..capture_end];
    let api_host = proxy_host
        .strip_prefix("proxy.")
        .map_or_else(|| proxy_host.to_owned(), |host| format!("api.{host}"));
    Some(format!("https://{api_host}"))
}

/// The models API base for a credential, upstream's `getGitHubCopilotBaseUrl`:
/// the token's `proxy-ep` endpoint, else the enterprise fallback, else the
/// individual endpoint.
#[must_use]
fn get_github_copilot_base_url(token: Option<&str>, enterprise_domain: Option<&str>) -> String {
    if let Some(base_url) = token.and_then(get_base_url_from_token) {
        return base_url;
    }
    if let Some(enterprise_domain) = enterprise_domain {
        return format!("https://copilot-api.{enterprise_domain}");
    }
    "https://api.individual.githubcopilot.com".to_owned()
}

/// The parsed model catalog, upstream's `parseGitHubCopilotModelCatalog`
/// result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CopilotModelCatalog {
    /// The model ids requests may use, picker flags filtered by policy state.
    pub available_model_ids: Vec<String>,
    /// The known models still unconfigured, the IDs the policy updates enable.
    pub policy_model_ids: Vec<String>,
}

/// Parse the models response, upstream's `parseGitHubCopilotModelCatalog`:
/// entries with tool-call support survive, picker flags gate the available
/// IDs, and the policy fallback applies only when allowed.
///
/// # Errors
/// Rejects with `Invalid Copilot models response` when the data array is
/// missing.
pub fn parse_github_copilot_model_catalog(
    raw: &Value,
    allow_policy_fallback: bool,
    known_models: &KnownModels,
) -> Result<CopilotModelCatalog, AuthError> {
    let Some(data) = raw.get("data").and_then(Value::as_array) else {
        return Err(auth_error("Invalid Copilot models response".to_owned()));
    };

    // (id, picker_enabled, policy_state) — the state is any wire value; only
    // the string comparisons upstream makes survive `as_str`.
    let mut account_models: Vec<(String, bool, Option<&str>)> = Vec::new();
    for item in data {
        let Some(item) = item.as_object() else {
            continue;
        };
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            continue;
        };
        let tool_calls_disabled = item
            .get("capabilities")
            .and_then(|capabilities| capabilities.get("supports"))
            .and_then(|supports| supports.get("tool_calls"))
            == Some(&Value::Bool(false));
        if tool_calls_disabled {
            continue;
        }
        account_models.push((
            id.to_owned(),
            item.get("model_picker_enabled") == Some(&Value::Bool(true)),
            item.get("policy")
                .and_then(|policy| policy.get("state"))
                .and_then(Value::as_str),
        ));
    }

    let picker_model_ids: Vec<String> = account_models
        .iter()
        .filter(|(_, picker_enabled, policy_state)| {
            *picker_enabled && *policy_state != Some("disabled")
        })
        .map(|(id, _, _)| id.clone())
        .collect();
    let use_policy_fallback = allow_policy_fallback && picker_model_ids.is_empty();
    let available_model_ids = if !picker_model_ids.is_empty() || !allow_policy_fallback {
        picker_model_ids
    } else {
        account_models
            .iter()
            .filter(|(_, _, policy_state)| *policy_state == Some("enabled"))
            .map(|(id, _, _)| id.clone())
            .collect()
    };
    let policy_model_ids: Vec<String> = account_models
        .iter()
        .filter(|(id, picker_enabled, policy_state)| {
            *policy_state == Some("unconfigured")
                && known_models(id)
                && (*picker_enabled || use_policy_fallback)
        })
        .map(|(id, _, _)| id.clone())
        .collect();
    Ok(CopilotModelCatalog {
        available_model_ids,
        policy_model_ids,
    })
}

/// The rate-limit retry budget, upstream's `{ maxRetries, maxElapsedMs }`.
#[derive(Clone, Copy, Debug)]
struct RetryPolicy {
    max_retries: u32,
    max_elapsed_ms: u64,
}

/// Send one request, retrying 429s within the retry policy, upstream's
/// `fetchWithRateLimitRetry`. The request prototype carries the method, URL,
/// headers, and body; each attempt applies its own 5-second timeout.
///
/// # Errors
/// Rejects with the transport's failure, or the abort message when the signal
/// cancels during a backoff sleep.
async fn fetch_with_rate_limit_retry(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    request: HttpRequest,
    signal: &CancellationToken,
    retry_policy: &RetryPolicy,
) -> Result<HttpResponse, AuthError> {
    let retry_deadline = (retry_policy.max_retries > 0 && retry_policy.max_elapsed_ms > 0)
        .then(|| clock.now_ms() + i64::try_from(retry_policy.max_elapsed_ms).unwrap_or(i64::MAX));
    let mut retry = 0_u32;
    loop {
        let mut attempt = request.clone();
        attempt.timeout_ms = Some(PER_REQUEST_TIMEOUT_MS);
        let response = execute(client, attempt).await?;
        if response.status != RATE_LIMIT_STATUS || retry == retry_policy.max_retries {
            return Ok(response);
        }

        let retry_after = response
            .headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))
            .map(|(_, value)| value.clone());
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the 429 delay rides upstream's JS arithmetic: seconds or a date delta, floored to whole milliseconds"
        )]
        let delay_ms = match retry_after {
            Some(retry_after) => {
                let seconds = retry_after.trim().parse::<f64>().ok();
                let value = match seconds {
                    Some(seconds) => seconds * 1000.0,
                    None => match parse_http_date_epoch_ms(&retry_after) {
                        Some(epoch_ms) => (epoch_ms - clock.now_ms()) as f64,
                        None => return Ok(response),
                    },
                };
                if !value.is_finite() {
                    return Ok(response);
                }
                value.max(0.0) as u64
            }
            None => RETRY_BASE_DELAY_MS.saturating_pow(retry),
        };
        if let Some(deadline) = retry_deadline {
            let remaining = deadline - clock.now_ms();
            if i64::try_from(delay_ms).unwrap_or(i64::MAX) >= remaining {
                return Ok(response);
            }
        }
        drop(response);
        sleep(delay_ms, signal)
            .await
            .map_err(|_| auth_error(crate::utils::abort::AbortError::MESSAGE.to_owned()))?;
        retry += 1;
    }
}

/// Parse an HTTP-date (IMF-fixdate) into epoch milliseconds, the retry-after
/// form `Date.parse` handles upstream.
#[must_use]
fn parse_http_date_epoch_ms(text: &str) -> Option<i64> {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let rest = text.trim();
    let rest = rest.split_once(", ").map_or(rest, |(_, rest)| rest.trim());
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.len() != 5 || parts[4] != "GMT" {
        return None;
    }
    let day: i64 = parts[0].parse().ok()?;
    let month = i64::try_from(MONTHS.iter().position(|month| *month == parts[1])?).ok()? + 1;
    let year: i64 = parts[2].parse().ok()?;
    let mut time = parts[3].split(':');
    let hour: i64 = time.next()?.parse().ok()?;
    let minute: i64 = time.next()?.parse().ok()?;
    let second: i64 = time.next()?.parse().ok()?;
    Some((days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second) * 1000)
}

/// Days from 1970-01-01 to the given civil date, Howard Hinnant's algorithm.
#[must_use]
const fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The status text a failed Copilot call reports, the wire's `statusText`
/// for the statuses the Copilot APIs answer with; unknown statuses report an
/// empty reason.
#[must_use]
pub(crate) const fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "",
    }
}

/// The request headers a Copilot API call sends: the optional bearer
/// authorization, upstream's `COPILOT_HEADERS`, and the optional
/// `X-GitHub-Api-Version`, in wire order.
fn copilot_request_headers(
    authorization: Option<String>,
    api_version: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers = Vec::with_capacity(2 + COPILOT_HEADERS.len());
    if let Some(authorization) = authorization {
        headers.push(("Authorization".to_owned(), authorization));
    }
    headers.extend(
        COPILOT_HEADERS
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned())),
    );
    if let Some(api_version) = api_version {
        headers.push(("X-GitHub-Api-Version".to_owned(), api_version.to_owned()));
    }
    headers
}

/// Read one response body as JSON, upstream's `fetchJson`; non-2xx statuses
/// surface as `{status} {statusText}: {text}`.
///
/// # Errors
/// Rejects with the status message or the body's JSON parse error.
async fn fetch_json(
    client: &Arc<dyn HttpClient>,
    request: HttpRequest,
) -> Result<Value, AuthError> {
    let mut response = execute(client, request).await?;
    let status = response.status;
    if !(200..300).contains(&status) {
        let text = read_body_lossy(&mut response).await;
        return Err(auth_error(format!(
            "{status} {}: {text}",
            status_reason(status)
        )));
    }
    read_json(&mut response).await
}

/// Start the device flow, upstream's `startDeviceFlow`.
///
/// # Errors
/// Rejects with the invalid-response messages and the untrusted-URI message.
async fn start_device_flow(
    client: &Arc<dyn HttpClient>,
    domain: &str,
    signal: &CancellationToken,
) -> Result<DeviceCodeResponse, AuthError> {
    let urls = get_urls(domain);
    let request = form_post_request(
        &urls.device_code,
        &[
            ("Accept", "application/json"),
            ("User-Agent", "GitHubCopilotChat/0.35.0"),
        ],
        &[
            ("client_id".to_owned(), CLIENT_ID.to_owned()),
            ("scope".to_owned(), "read:user".to_owned()),
        ],
        signal.clone(),
        None,
    );
    let raw = fetch_json(client, request).await?;
    let map = match &raw {
        // An array response has no fields, upstream's `typeof data === "object"`
        // letting arrays pass to the field check.
        Value::Object(map) => map.clone(),
        Value::Array(_) => serde_json::Map::new(),
        _ => return Err(auth_error("Invalid device code response".to_owned())),
    };

    let device_code = map.get("device_code").and_then(Value::as_str);
    let user_code = map.get("user_code").and_then(Value::as_str);
    let verification_uri = map.get("verification_uri").and_then(Value::as_str);
    let interval_valid = map.get("interval").is_none_or(Value::is_number);
    let expires_in = map.get("expires_in").and_then(Value::as_f64);
    let (Some(device_code), Some(user_code), Some(verification_uri), true, Some(expires_in)) = (
        device_code,
        user_code,
        verification_uri,
        interval_valid,
        expires_in,
    ) else {
        return Err(auth_error("Invalid device code response fields".to_owned()));
    };

    // The verification URI is opened in the user's browser; a non-http(s)
    // value could make `open` launch something else.
    let parsed_uri = url::Url::parse(verification_uri)
        .map_err(|_| auth_error(UNTRUSTED_VERIFICATION_URI.to_owned()))?;
    if parsed_uri.scheme() != "https" && parsed_uri.scheme() != "http" {
        return Err(auth_error(UNTRUSTED_VERIFICATION_URI.to_owned()));
    }

    Ok(DeviceCodeResponse {
        device_code: device_code.to_owned(),
        user_code: user_code.to_owned(),
        verification_uri: parsed_uri.to_string(),
        interval_seconds: map
            .get("interval")
            .and_then(Value::as_f64)
            .map(wire_seconds),
        expires_in_seconds: wire_seconds(expires_in),
    })
}

/// One device-code start response, upstream's `DeviceCodeResponse`.
#[derive(Clone)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval_seconds: Option<u64>,
    expires_in_seconds: u64,
}

/// Poll for the GitHub access token, upstream's `pollForGitHubAccessToken`.
///
/// # Errors
/// Rejects with the flow's timeout messages and the poll's failed messages.
async fn poll_for_github_access_token(
    client: &Arc<dyn HttpClient>,
    domain: &str,
    device: DeviceCodeResponse,
    signal: &CancellationToken,
) -> Result<String, AuthError> {
    let urls = get_urls(domain);
    poll_oauth_device_code_flow::<String>(PollOptions {
        interval_seconds: device.interval_seconds,
        expires_in_seconds: Some(device.expires_in_seconds),
        wait_before_first_poll: true,
        signal: signal.clone(),
        poll: {
            let client = Arc::clone(client);
            let access_token_url = urls.access_token;
            let device = device.clone();
            let signal = signal.clone();
            Arc::new(move || {
                let client = Arc::clone(&client);
                let access_token_url = access_token_url.clone();
                let device = device.clone();
                let signal = signal.clone();
                Box::pin(async move {
                    let request = form_post_request(
                        &access_token_url,
                        &[
                            ("Accept", "application/json"),
                            ("User-Agent", "GitHubCopilotChat/0.35.0"),
                        ],
                        &[
                            ("client_id".to_owned(), CLIENT_ID.to_owned()),
                            ("device_code".to_owned(), device.device_code.clone()),
                            (
                                "grant_type".to_owned(),
                                "urn:ietf:params:oauth:grant-type:device_code".to_owned(),
                            ),
                        ],
                        signal.clone(),
                        None,
                    );
                    let raw = fetch_json(&client, request).await?;

                    if let Some(access_token) = raw.get("access_token").and_then(Value::as_str) {
                        return Ok(PollOutcome::Complete(access_token.to_owned()));
                    }

                    let error = raw.get("error").and_then(Value::as_str);
                    let Some(error) = error else {
                        return Ok(PollOutcome::Failed(
                            "Invalid device token response".to_owned(),
                        ));
                    };
                    if error == "authorization_pending" {
                        return Ok(PollOutcome::Pending);
                    }
                    if error == "slow_down" {
                        return Ok(PollOutcome::SlowDown {
                            interval_seconds: raw
                                .get("interval")
                                .and_then(Value::as_f64)
                                .map(wire_seconds),
                        });
                    }
                    let description_suffix = raw
                        .get("error_description")
                        .and_then(Value::as_str)
                        .map_or_else(String::new, |description| format!(": {description}"));
                    Ok(PollOutcome::Failed(format!(
                        "Device flow failed: {error}{description_suffix}"
                    )))
                })
            })
        },
    })
    .await
}

/// Exchange the GitHub access token for the Copilot credential, upstream's
/// `refreshGitHubCopilotAccessToken`.
///
/// # Errors
/// Rejects with the invalid-token-response messages.
async fn refresh_github_copilot_access_token(
    client: &Arc<dyn HttpClient>,
    refresh_token: &str,
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
) -> Result<OAuthCredentials, AuthError> {
    let domain = enterprise_domain.unwrap_or("github.com");
    let urls = get_urls(domain);
    let request = HttpRequest {
        method: HttpMethod::Get,
        url: urls.copilot_token,
        headers: copilot_request_headers(Some(format!("Bearer {refresh_token}")), None),
        body: None,
        timeout_ms: None,
        signal: signal.clone(),
    };
    let raw = fetch_json(client, request).await?;
    let map = raw
        .as_object()
        .ok_or_else(|| auth_error("Invalid Copilot token response".to_owned()))?;
    let token = map.get("token").and_then(Value::as_str);
    let expires_at = map.get("expires_at").and_then(Value::as_f64);
    let (Some(token), Some(expires_at)) = (token, expires_at) else {
        return Err(auth_error(
            "Invalid Copilot token response fields".to_owned(),
        ));
    };

    let mut credential = oauth_credentials(
        token,
        refresh_token,
        wire_milliseconds(expires_at).saturating_mul(1000) - ACCESS_TOKEN_SKEW_MS,
    );
    if let Some(enterprise_domain) = enterprise_domain {
        credential
            .extra
            .insert("enterpriseUrl".to_owned(), Value::String(enterprise_domain.to_owned()));
    }
    Ok(credential)
}

/// The Copilot credential plus its available models, upstream's
/// `refreshGitHubCopilotToken`.
///
/// # Errors
/// Rejects with the access-token exchange's and the models fetch's errors.
async fn refresh_github_copilot_token(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    known_models: &KnownModels,
    refresh_token: &str,
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
) -> Result<OAuthCredentials, AuthError> {
    let mut credentials =
        refresh_github_copilot_access_token(client, refresh_token, enterprise_domain, signal)
            .await?;
    let models = fetch_github_copilot_models(
        client,
        clock,
        known_models,
        &credentials.access,
        enterprise_domain,
        signal,
        RetryPolicy {
            max_retries: 0,
            max_elapsed_ms: 0,
        },
    )
    .await?;
    credentials.extra.insert(
        "availableModelIds".to_owned(),
        Value::Array(
            models
                .available_model_ids
                .into_iter()
                .map(Value::String)
                .collect(),
        ),
    );
    Ok(credentials)
}

/// Fetch the model catalog for a Copilot token, upstream's
/// `fetchGitHubCopilotModels`.
///
/// # Errors
/// Rejects with the rate-limited retry's errors and the
/// `{status} {statusText}: {text}` failure message.
async fn fetch_github_copilot_models(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    known_models: &KnownModels,
    copilot_token: &str,
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
    retry_policy: RetryPolicy,
) -> Result<CopilotModelCatalog, AuthError> {
    let base_url = get_github_copilot_base_url(Some(copilot_token), enterprise_domain);
    // Some Individual accounts return false for every picker flag despite
    // explicit enabled policies. Limit the fallback to that endpoint so other
    // account types keep strict picker semantics.
    let allow_policy_fallback = base_url == "https://api.individual.githubcopilot.com";
    let mut headers = vec![
        ("Accept".to_owned(), "application/json".to_owned()),
        (
            "Authorization".to_owned(),
            format!("Bearer {copilot_token}"),
        ),
    ];
    headers.extend(copilot_request_headers(None, Some(COPILOT_API_VERSION)));
    let request = HttpRequest {
        method: HttpMethod::Get,
        url: format!("{base_url}/models"),
        headers,
        body: None,
        timeout_ms: None,
        signal: signal.clone(),
    };
    let mut response =
        fetch_with_rate_limit_retry(client, clock, request, signal, &retry_policy).await?;
    if !(200..300).contains(&response.status) {
        let text = read_body_lossy(&mut response).await;
        return Err(auth_error(format!(
            "{} {}: {text}",
            response.status,
            status_reason(response.status)
        )));
    }
    let raw = read_json(&mut response).await?;
    parse_github_copilot_model_catalog(&raw, allow_policy_fallback, known_models)
}

/// Enable one model for the account, upstream's `enableGitHubCopilotModel`.
///
/// # Errors
/// Rejects on cancellation and when rate limiting exhausts the retry budget.
async fn enable_github_copilot_model(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    token: &str,
    model_id: &str,
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
) -> Result<bool, AuthError> {
    let base_url = get_github_copilot_base_url(Some(token), enterprise_domain);
    let url = format!("{base_url}/models/{model_id}/policy");
    let authorization = format!("Bearer {token}");
    let mut headers: Vec<(&str, &str)> = vec![("Authorization", authorization.as_str())];
    headers.extend(COPILOT_HEADERS.iter().copied());
    headers.extend([
        ("openai-intent", "chat-policy"),
        ("x-interaction-type", "chat-policy"),
    ]);
    let request = json_post_request(
        &url,
        &headers,
        &serde_json::json!({ "state": "enabled" }),
        signal.clone(),
        None,
    );
    let response = match fetch_with_rate_limit_retry(
        client,
        clock,
        request,
        signal,
        &RetryPolicy {
            max_retries: 2,
            max_elapsed_ms: 5000,
        },
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            if signal.is_cancelled() {
                return Err(error);
            }
            return Ok(false);
        }
    };
    if response.status == RATE_LIMIT_STATUS {
        let mut response = response;
        let text = read_body_lossy(&mut response).await;
        return Err(auth_error(format!(
            "{} {}: {text}",
            response.status,
            status_reason(response.status)
        )));
    }
    Ok((200..300).contains(&response.status))
}

/// Enable the policy models and return the successful IDs, upstream's
/// `enableGitHubCopilotModels`. Policy updates are best effort; exhausted
/// rate limiting stops the batch.
///
/// # Errors
/// Propagates cancellation; transport failures stop the batch without
/// failing.
async fn enable_github_copilot_models(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    token: &str,
    model_ids: &[String],
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
) -> Result<Vec<String>, AuthError> {
    let mut enabled_model_ids: Vec<String> = Vec::new();
    for model_id in model_ids {
        match enable_github_copilot_model(client, clock, token, model_id, enterprise_domain, signal)
            .await
        {
            Ok(true) => enabled_model_ids.push(model_id.clone()),
            Ok(false) => {}
            Err(error) => {
                if signal.is_cancelled() {
                    return Err(error);
                }
                break;
            }
        }
    }
    Ok(enabled_model_ids)
}

/// The enterprise domain a credential carries, upstream's
/// `copilotEnterpriseDomain`.
#[must_use]
fn copilot_enterprise_domain(credential: &OAuthCredentials) -> Option<String> {
    let enterprise_url = crate::auth::oauth::extra_string(&credential.extra, "enterpriseUrl")?;
    if enterprise_url.is_empty() {
        return None;
    }
    normalize_domain(enterprise_url)
}

/// The login flow, upstream's `loginGitHubCopilot`.
///
/// # Errors
/// Rejects with `Login cancelled`, the invalid-domain message, the device
/// flow's errors, and the model/policy steps' failures.
async fn login_github_copilot(
    client: &Arc<dyn HttpClient>,
    clock: &Arc<dyn crate::auth::clock::AuthClock>,
    known_models: &KnownModels,
    interaction: &crate::auth::types::ProviderAuthInteraction,
) -> Result<OAuthCredentials, AuthError> {
    let input = (interaction.prompt)(AuthPrompt {
        signal: None,
        kind: crate::auth::types::AuthPromptKind::Text {
            message: "GitHub Enterprise URL/domain (blank for github.com)".to_owned(),
            placeholder: Some("company.ghe.com".to_owned()),
        },
    })
    .await
    .map_err(AuthError::from)?;
    if interaction.signal.is_cancelled() {
        return Err(auth_error("Login cancelled".to_owned()));
    }

    let trimmed = input.trim();
    let enterprise_domain = normalize_domain(&input);
    if !trimmed.is_empty() && enterprise_domain.is_none() {
        return Err(auth_error("Invalid GitHub Enterprise URL/domain".to_owned()));
    }
    let domain = enterprise_domain
        .clone()
        .unwrap_or_else(|| "github.com".to_owned());

    let device = start_device_flow(client, &domain, &interaction.signal).await?;
    (interaction.notify)(AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: device.verification_uri.clone(),
        interval_seconds: device.interval_seconds,
        expires_in_seconds: Some(device.expires_in_seconds),
    });

    let github_access_token =
        poll_for_github_access_token(client, &domain, device, &interaction.signal).await?;
    let credentials = refresh_github_copilot_access_token(
        client,
        &github_access_token,
        enterprise_domain.as_deref(),
        &interaction.signal,
    )
    .await?;
    let models = fetch_github_copilot_models(
        client,
        clock.as_ref(),
        known_models,
        &credentials.access,
        enterprise_domain.as_deref(),
        &interaction.signal,
        RetryPolicy {
            max_retries: 2,
            max_elapsed_ms: 5000,
        },
    )
    .await?;

    let enabled_model_ids: Vec<String> = if models.policy_model_ids.is_empty() {
        Vec::new()
    } else {
        (interaction.notify)(AuthEvent::Progress {
            message: "Enabling models...".to_owned(),
        });
        enable_github_copilot_models(
            client,
            clock.as_ref(),
            &credentials.access,
            &models.policy_model_ids,
            enterprise_domain.as_deref(),
            &interaction.signal,
        )
        .await?
    };

    let mut available_model_ids = models.available_model_ids;
    for model_id in enabled_model_ids {
        if !available_model_ids.contains(&model_id) {
            available_model_ids.push(model_id);
        }
    }
    let mut credential = credentials;
    credential.extra.insert(
        "availableModelIds".to_owned(),
        Value::Array(available_model_ids.into_iter().map(Value::String).collect()),
    );
    Ok(credential)
}
