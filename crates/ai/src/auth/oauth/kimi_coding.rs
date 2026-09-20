//! The Kimi Code (subscription) OAuth device-code flow, ported from
//! `packages/ai/src/auth/oauth/kimi-coding.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::auth::oauth::device_code::{PollOptions, PollOutcome, poll_oauth_device_code_flow};
use crate::auth::oauth::{
    auth_error, execute, form_post_request, json_object, oauth_credentials, read_body_lossy,
    read_json, wire_seconds_to_i64, wire_seconds_to_u64,
};
use crate::auth::types::ModelAuth;
use crate::auth::types::{AuthError, AuthEvent, OAuthCredentials};
use crate::types::BoxedFuture;
use crate::utils::provider_env::get_provider_env_value;

const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516c098c0a1";
const DEFAULT_OAUTH_HOST: &str = "https://auth.kimi.com";
const DEVICE_CODE_TIMEOUT_SECONDS: u64 = 15 * 60;
const DEFAULT_POLL_INTERVAL_SECONDS: u64 = 5;
const REQUEST_TIMEOUT_MS: u64 = 30 * 1000;
const REFRESH_MAX_RETRIES: u32 = 3;

/// The Kimi Code OAuth device-code flow, wired with its [`HttpClient`]
/// seam.
pub struct KimiCodingOAuth {
    client: Arc<dyn crate::http::HttpClient>,
}

impl std::fmt::Debug for KimiCodingOAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("KimiCodingOAuth")
    }
}

impl KimiCodingOAuth {
    /// The flow over the given client.
    #[must_use]
    pub fn new(client: Arc<dyn crate::http::HttpClient>) -> Self {
        Self { client }
    }
}

fn get_oauth_host() -> String {
    let override_host = get_provider_env_value("KIMI_CODE_OAUTH_HOST", None)
        .or_else(|| get_provider_env_value("KIMI_OAUTH_HOST", None))
        .unwrap_or_else(|| DEFAULT_OAUTH_HOST.to_owned());
    override_host.trim_end_matches('/').to_owned()
}

/// The verification URI is opened in the user's browser; only http(s) URLs
/// are trusted.
fn trusted_http_url(value: &str) -> Option<String> {
    if value.is_empty() {
        return None;
    }
    let url = url::Url::parse(value).ok()?;
    if url.scheme() != "https" && url.scheme() != "http" {
        return None;
    }
    Some(url.to_string())
}

#[derive(Clone, Debug)]
struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri_complete: String,
    interval_seconds: u64,
    expires_in_seconds: u64,
}

#[derive(Clone, Debug)]
struct TokenResponse {
    access: String,
    refresh: String,
    expires: i64,
}

async fn start_device_authorization(
    client: &Arc<dyn crate::http::HttpClient>,
    oauth_host: &str,
    signal: &CancellationToken,
) -> Result<DeviceAuthorization, AuthError> {
    let request = form_post_request(
        &format!("{oauth_host}/api/oauth/device_authorization"),
        &[("Accept", "application/json")],
        &[("client_id".to_owned(), CLIENT_ID.to_owned())],
        signal.clone(),
        Some(REQUEST_TIMEOUT_MS),
    );
    let mut response = execute(client, request).await?;
    if !(200..300).contains(&response.status) {
        let text = read_body_lossy(&mut response).await;
        return Err(auth_error(format!(
            "Kimi Code device authorization failed with status {}{}",
            response.status,
            if text.is_empty() {
                String::new()
            } else {
                format!(": {text}")
            }
        )));
    }
    let json = json_object(
        read_json(&mut response).await?,
        "Invalid Kimi Code device authorization response",
    )?;
    let missing_field = || {
        auth_error(format!(
            "Invalid Kimi Code device authorization response: {}",
            serde_json::Value::Object(json.clone())
        ))
    };
    let device_code = json
        .get("device_code")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(missing_field)?;
    let user_code = json
        .get("user_code")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(missing_field)?;
    json.get("verification_uri")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(missing_field)?;
    let verification_uri_complete = json
        .get("verification_uri_complete")
        .and_then(serde_json::Value::as_str)
        .and_then(trusted_http_url)
        .ok_or_else(missing_field)?;
    let interval_seconds = json
        .get("interval")
        .and_then(serde_json::Value::as_f64)
        .filter(|interval| interval.is_finite() && *interval > 0.0)
        .map_or(DEFAULT_POLL_INTERVAL_SECONDS, wire_seconds_to_u64);
    let expires_in_seconds = json
        .get("expires_in")
        .and_then(serde_json::Value::as_f64)
        .filter(|expires| expires.is_finite() && *expires > 0.0)
        .map_or(DEVICE_CODE_TIMEOUT_SECONDS, wire_seconds_to_u64);
    Ok(DeviceAuthorization {
        device_code: device_code.to_owned(),
        user_code: user_code.to_owned(),
        verification_uri_complete,
        interval_seconds,
        expires_in_seconds,
    })
}

fn parse_token_response(
    json: &serde_json::Map<String, serde_json::Value>,
    clock: &dyn crate::auth::clock::AuthClock,
    operation: &str,
) -> Result<TokenResponse, AuthError> {
    let access = json.get("access_token").and_then(serde_json::Value::as_str);
    let refresh = json
        .get("refresh_token")
        .and_then(serde_json::Value::as_str);
    let expires_in = json.get("expires_in").and_then(serde_json::Value::as_f64);
    let (Some(access), Some(refresh), Some(expires_in)) = (
        access.filter(|value| !value.is_empty()),
        refresh.filter(|value| !value.is_empty()),
        expires_in.filter(|value| value.is_finite() && *value > 0.0),
    ) else {
        return Err(auth_error(format!(
            "Kimi Code token {operation} response missing fields: {}",
            serde_json::Value::Object(json.clone())
        )));
    };
    Ok(TokenResponse {
        access: access.to_owned(),
        refresh: refresh.to_owned(),
        expires: clock.now_ms() + wire_seconds_to_i64(expires_in).saturating_mul(1000),
    })
}

async fn poll_for_token(
    client: &Arc<dyn crate::http::HttpClient>,
    oauth_host: &str,
    device: DeviceAuthorization,
    clock: Arc<dyn crate::auth::clock::AuthClock>,
    signal: &CancellationToken,
) -> Result<TokenResponse, AuthError> {
    poll_oauth_device_code_flow::<TokenResponse>(PollOptions {
        interval_seconds: Some(device.interval_seconds),
        expires_in_seconds: Some(device.expires_in_seconds),
        wait_before_first_poll: true,
        signal: signal.clone(),
        poll: {
            let client = client.clone();
            let clock = clock.clone();
            let oauth_host = oauth_host.to_owned();
            let device = device.clone();
            let signal = signal.clone();
            Arc::new(move || {
                let client = client.clone();
                let clock = clock.clone();
                let oauth_host = oauth_host.clone();
                let device = device.clone();
                let signal = signal.clone();
                Box::pin(async move {
                    let request = form_post_request(
                        &format!("{oauth_host}/api/oauth/token"),
                        &[("Accept", "application/json")],
                        &[
                            ("client_id".to_owned(), CLIENT_ID.to_owned()),
                            ("device_code".to_owned(), device.device_code.clone()),
                            (
                                "grant_type".to_owned(),
                                "urn:ietf:params:oauth:grant-type:device_code".to_owned(),
                            ),
                        ],
                        signal.clone(),
                        Some(REQUEST_TIMEOUT_MS),
                    );
                    let mut response = execute(&client, request).await?;
                    let status = response.status;

                    if status >= 500 {
                        let text = read_body_lossy(&mut response).await;
                        return Ok(PollOutcome::Failed(format!(
                            "Kimi Code device token request failed with status {status}{text_suffix}",
                            text_suffix = if text.is_empty() { String::new() } else { format!(": {text}") }
                        )));
                    }

                    let json = match read_json(&mut response).await {
                        Ok(serde_json::Value::Object(map)) => Some(map),
                        _ => None,
                    };
                    if (200..300).contains(&status)
                        && let Some(json) = json
                            .as_ref()
                            .filter(|json| json.get("access_token").is_some_and(serde_json::Value::is_string))
                    {
                        return match parse_token_response(json, clock.as_ref(), "poll") {
                            Ok(token) => Ok(PollOutcome::Complete(token)),
                            Err(error) => Ok(PollOutcome::Failed(error.to_string())),
                        };
                    }

                    let error = json.as_ref().and_then(|json| json.get("error"));
                    let description = json
                        .as_ref()
                        .and_then(|json| json.get("error_description"))
                        .and_then(serde_json::Value::as_str);
                    match error.and_then(serde_json::Value::as_str) {
                        Some("authorization_pending") => Ok(PollOutcome::Pending),
                        Some("slow_down") => Ok(PollOutcome::SlowDown {
                            interval_seconds: json
                                .as_ref()
                                .and_then(|json| json.get("interval"))
                                .and_then(serde_json::Value::as_f64)
                                .filter(|interval| interval.is_finite() && *interval > 0.0)
                                .map(wire_seconds_to_u64),
                        }),
                        Some("expired_token") => Ok(PollOutcome::Failed(
                            "Kimi Code device authorization expired. Please restart login.".to_owned(),
                        )),
                        Some("access_denied") => Ok(PollOutcome::Failed("Kimi Code login was denied.".to_owned())),
                        _ => Ok(PollOutcome::Failed(format!(
                            "Kimi Code device token request failed (status {status}){error_suffix}",
                            error_suffix = error
                                .and_then(serde_json::Value::as_str)
                                .map(|error| {
                                    let description = description.map_or_else(String::new, |description| format!(": {description}"));
                                    format!(": {error}{description}")
                                })
                                .unwrap_or_default()
                        ))),
                    }
                })
            })
        },
    })
    .await
}

const fn is_retryable_refresh_failure(status: u16) -> bool {
    status == 429 || status >= 500
}

async fn refresh_token(
    client: &Arc<dyn crate::http::HttpClient>,
    oauth_host: &str,
    refresh_token_value: &str,
    clock: &Arc<dyn crate::auth::clock::AuthClock>,
    signal: &CancellationToken,
) -> Result<TokenResponse, AuthError> {
    let mut last_error: Option<AuthError> = None;
    for attempt in 0..=REFRESH_MAX_RETRIES {
        if attempt > 0 {
            let backoff_ms = 1000 * u64::from(2_u32).saturating_pow(attempt - 1);
            crate::utils::sleep::sleep(backoff_ms, signal)
                .await
                .map_err(|_| auth_error("Kimi Code token refresh aborted".to_owned()))?;
        }
        if signal.is_cancelled() {
            return Err(auth_error("Kimi Code token refresh aborted"));
        }

        let request = form_post_request(
            &format!("{oauth_host}/api/oauth/token"),
            &[("Accept", "application/json")],
            &[
                ("client_id".to_owned(), CLIENT_ID.to_owned()),
                ("grant_type".to_owned(), "refresh_token".to_owned()),
                ("refresh_token".to_owned(), refresh_token_value.to_owned()),
            ],
            signal.clone(),
            Some(REQUEST_TIMEOUT_MS),
        );
        let response = match execute(client, request).await {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(error);
                continue;
            }
        };
        let status = response.status;
        let mut response = response;
        let json = match read_json(&mut response).await {
            Ok(serde_json::Value::Object(map)) => Some(map),
            _ => None,
        };
        if (200..300).contains(&status) {
            let json = json.unwrap_or_default();
            return parse_token_response(&json, clock.as_ref(), "refresh");
        }

        // Unauthorized: the stored credential is dead; Models clears it and prompts re-login.
        let unauthorized = status == 401
            || status == 403
            || json.as_ref().is_some_and(|json| {
                json.get("error").and_then(serde_json::Value::as_str) == Some("invalid_grant")
            });
        if unauthorized {
            let description = json
                .as_ref()
                .and_then(|json| json.get("error_description"))
                .and_then(serde_json::Value::as_str)
                .map_or_else(String::new, |description| format!(": {description}"));
            return Err(auth_error(format!(
                "Kimi Code token refresh unauthorized (status {status}){description}"
            )));
        }

        if is_retryable_refresh_failure(status) && attempt < REFRESH_MAX_RETRIES {
            last_error = Some(auth_error(format!(
                "Kimi Code token refresh failed with status {status}"
            )));
            continue;
        }

        let text = json
            .map(|json| serde_json::Value::Object(json).to_string())
            .unwrap_or_default();
        let text_suffix = if text.is_empty() {
            String::new()
        } else {
            format!(": {text}")
        };
        return Err(auth_error(format!(
            "Kimi Code token refresh failed with status {status}{text_suffix}"
        )));
    }
    Err(last_error.unwrap_or_else(|| auth_error("Kimi Code token refresh failed".to_owned())))
}

impl KimiCodingOAuth {
    /// Run the interactive login flow: device-code authorization against the
    /// configured host.
    #[must_use]
    pub fn login(
        &self,
        interaction: crate::auth::types::ProviderAuthInteraction,
    ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>> {
        let client = self.client.clone();
        Box::pin(async move {
            let oauth_host = get_oauth_host();
            let clock: Arc<dyn crate::auth::clock::AuthClock> =
                Arc::new(crate::auth::clock::SystemClock);
            let device =
                start_device_authorization(&client, &oauth_host, &interaction.signal).await?;
            (interaction.notify)(AuthEvent::DeviceCode {
                user_code: device.user_code.clone(),
                verification_uri: device.verification_uri_complete.clone(),
                interval_seconds: Some(device.interval_seconds),
                expires_in_seconds: Some(device.expires_in_seconds),
            });
            let token =
                poll_for_token(&client, &oauth_host, device, clock, &interaction.signal).await?;
            Ok(oauth_credentials(
                token.access,
                token.refresh,
                token.expires,
            ))
        })
    }

    /// Exchange the refresh token for a rotated credential.
    #[must_use]
    pub fn refresh(
        &self,
        credential: OAuthCredentials,
        signal: CancellationToken,
    ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>> {
        let client = self.client.clone();
        Box::pin(async move {
            let oauth_host = get_oauth_host();
            let clock: Arc<dyn crate::auth::clock::AuthClock> =
                Arc::new(crate::auth::clock::SystemClock);
            let token =
                refresh_token(&client, &oauth_host, &credential.refresh, &clock, &signal).await?;
            Ok(oauth_credentials(
                token.access,
                token.refresh,
                token.expires,
            ))
        })
    }

    /// Derive the request auth from a valid credential: the bearer header,
    /// no api-key slot.
    #[must_use]
    pub fn to_auth(&self, credential: &OAuthCredentials) -> ModelAuth {
        ModelAuth {
            api_key: None,
            headers: Some({
                let mut headers = crate::types::ProviderHeaders::new();
                headers.insert(
                    "Authorization".to_owned(),
                    Some(format!("Bearer {}", credential.access)),
                );
                headers
            }),
            base_url: None,
        }
    }

    /// The flow wired into the merged auth core's callback-based
    /// [`OAuthAuth`](crate::auth::types::OAuthAuth): the login, refresh, and
    /// derivation closures drive this flow's own client. (#29)
    #[must_use]
    pub fn auth(&self) -> crate::auth::types::OAuthAuth {
        let login: crate::auth::types::OAuthLoginFn = {
            let client = Arc::clone(&self.client);
            Arc::new(move |interaction| {
                let flow = Self::new(Arc::clone(&client));
                flow.login(interaction)
            })
        };
        let refresh: crate::auth::types::OAuthRefreshFn = {
            let client = Arc::clone(&self.client);
            Arc::new(move |credential, signal| {
                let flow = Self::new(Arc::clone(&client));
                flow.refresh(credential, signal)
            })
        };
        let to_auth: crate::auth::types::OAuthToAuthFn = {
            Arc::new(|credential| {
                let auth = ModelAuth {
                    api_key: None,
                    headers: Some({
                        let mut headers = crate::types::ProviderHeaders::new();
                        headers.insert(
                            "Authorization".to_owned(),
                            Some(format!("Bearer {}", credential.access)),
                        );
                        headers
                    }),
                    base_url: None,
                };
                Box::pin(async move { Ok(auth) })
            })
        };
        crate::auth::types::OAuthAuth {
            name: "Kimi Code (subscription)".to_owned(),
            is_subscription: Some(true),
            login_label: Some("Sign in with Kimi Code".to_owned()),
            login,
            refresh,
            to_auth,
        }
    }
}
