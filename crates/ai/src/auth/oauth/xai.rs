//! The xAI OAuth device-code flow, ported from
//! `packages/ai/src/auth/oauth/xai.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::auth::oauth::device_code::{PollOptions, PollOutcome, poll_oauth_device_code_flow};
use crate::auth::oauth::{
    auth_error, execute, form_post_request, oauth_credentials, read_json, required_string,
    wire_seconds_to_u64,
};
use crate::auth::types::ModelAuth;
use crate::auth::types::{AuthError, AuthEvent, OAuthCredentials};
use crate::http::HttpClient;
use crate::types::BoxedFuture;

const XAI_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const XAI_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const XAI_DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
const XAI_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
// Refresh slightly before the reported expiry to avoid using a token that
// dies mid-request.
const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;
const DEFAULT_TOKEN_LIFETIME_SECONDS: u64 = 3600;

/// A parsed device-code start response, upstream's `XaiDeviceCode`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct XaiDeviceCode {
    pub(crate) device_code: String,
    pub(crate) user_code: String,
    pub(crate) verification_uri: String,
    pub(crate) verification_uri_complete: Option<String>,
    pub(crate) interval_seconds: Option<u64>,
    pub(crate) expires_in_seconds: u64,
}

/// The xAI OAuth device-code flow, wired with its [`HttpClient`] seam and
/// epoch clock.
pub struct XaiOAuth {
    client: Arc<dyn HttpClient>,
    clock: Arc<dyn crate::auth::clock::AuthClock>,
}

impl std::fmt::Debug for XaiOAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("XaiOAuth")
    }
}

impl XaiOAuth {
    /// The flow over the given client and clock.
    #[must_use]
    pub fn new(client: Arc<dyn HttpClient>, clock: Arc<dyn crate::auth::clock::AuthClock>) -> Self {
        Self { client, clock }
    }
}

/// A finite positive JSON number, upstream's `positiveNumber`.
fn positive_number(
    body: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<u64, AuthError> {
    match body.get(field).and_then(serde_json::Value::as_f64) {
        Some(value) if value.is_finite() && value > 0.0 && value.fract() == 0.0 => {
            Ok(wire_seconds_to_u64(value))
        }
        _ => Err(auth_error(format!(
            "Invalid xAI OAuth response field: {field}"
        ))),
    }
}

// The verification URI is opened in the user's browser; force it to be an
// https URL so a malicious response cannot make `open` launch something else.
fn validate_verification_uri(raw: &str) -> Result<String, AuthError> {
    let url = url::Url::parse(raw).map_err(|_| auth_error(UNTRUSTED_URI.to_owned()))?;
    if url.scheme() != "https" {
        return Err(auth_error(UNTRUSTED_URI.to_owned()));
    }
    Ok(url.to_string())
}

const UNTRUSTED_URI: &str = "Untrusted verification URI in xAI OAuth response";

async fn post_form(
    client: &Arc<dyn HttpClient>,
    url: &str,
    fields: &[(String, String)],
    signal: &CancellationToken,
) -> Result<(u16, serde_json::Map<String, serde_json::Value>), AuthError> {
    let request = form_post_request(
        url,
        &[("Accept", "application/json")],
        fields,
        signal.clone(),
        None,
    );
    let mut response = execute(client, request).await.map_err(|error| {
        if signal.is_cancelled() {
            return auth_error("Login cancelled".to_owned());
        }
        error
    })?;
    let status = response.status;
    let body = match read_json(&mut response).await {
        Ok(serde_json::Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    Ok((status, body))
}

fn request_failure(
    action: &str,
    status: u16,
    body: &serde_json::Map<String, serde_json::Value>,
) -> AuthError {
    let error = body.get("error").and_then(serde_json::Value::as_str);
    let description = body
        .get("error_description")
        .and_then(serde_json::Value::as_str);
    let detail = [error, description]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(": ");
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!(": {detail}")
    };
    auth_error(format!("xAI OAuth {action} failed (HTTP {status}){detail}"))
}

fn parse_device_code(
    body: &serde_json::Map<String, serde_json::Value>,
) -> Result<XaiDeviceCode, AuthError> {
    // RFC 8628 allows interval 0 (no minimum wait); fall back to the poller's
    // default instead of failing on non-positive or malformed values.
    let interval_seconds = body
        .get("interval")
        .and_then(serde_json::Value::as_f64)
        .filter(|interval| interval.is_finite() && *interval > 0.0)
        .map(wire_seconds_to_u64);
    let verification_uri_complete = body
        .get("verification_uri_complete")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(validate_verification_uri)
        .transpose()?;
    Ok(XaiDeviceCode {
        device_code: required_xai_string(body, "device_code")?,
        user_code: required_xai_string(body, "user_code")?,
        verification_uri: validate_verification_uri(&required_xai_string(
            body,
            "verification_uri",
        )?)?,
        verification_uri_complete,
        interval_seconds,
        expires_in_seconds: positive_number(body, "expires_in")?,
    })
}

fn required_xai_string(
    body: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<String, AuthError> {
    match body.get(field) {
        Some(serde_json::Value::String(value)) if !value.is_empty() => Ok(value.clone()),
        _ => Err(auth_error(format!(
            "Invalid xAI OAuth response field: {field}"
        ))),
    }
}

fn credentials_from_token_response(
    clock: &dyn crate::auth::clock::AuthClock,
    body: &serde_json::Map<String, serde_json::Value>,
    previous_refresh_token: Option<&str>,
) -> Result<OAuthCredentials, AuthError> {
    let access = required_string(body, "access_token")
        .map_err(|_| auth_error("Invalid xAI OAuth response field: access_token".to_owned()))?;
    // xAI may omit refresh_token on refresh when the token is not rotated.
    let refresh = if body.get("refresh_token").is_none() {
        match previous_refresh_token {
            Some(previous) => previous.to_owned(),
            None => required_string(body, "refresh_token").map_err(|_| {
                auth_error("Invalid xAI OAuth response field: refresh_token".to_owned())
            })?,
        }
    } else {
        required_string(body, "refresh_token")
            .map_err(|_| auth_error("Invalid xAI OAuth response field: refresh_token".to_owned()))?
    };
    let expires_in_seconds = match body.get("expires_in") {
        None | Some(serde_json::Value::Null) => DEFAULT_TOKEN_LIFETIME_SECONDS,
        Some(_) => positive_number(body, "expires_in")?,
    };
    let expires_ms = i64::try_from(expires_in_seconds.saturating_mul(1000)).unwrap_or(i64::MAX);
    Ok(oauth_credentials(
        access,
        refresh,
        clock.now_ms() + expires_ms - REFRESH_SKEW_MS,
    ))
}

async fn request_device_code(
    client: &Arc<dyn HttpClient>,
    signal: &CancellationToken,
) -> Result<XaiDeviceCode, AuthError> {
    let (status, body) = post_form(
        client,
        XAI_DEVICE_CODE_URL,
        &[
            ("client_id".to_owned(), XAI_CLIENT_ID.to_owned()),
            ("scope".to_owned(), XAI_SCOPE.to_owned()),
            ("referrer".to_owned(), "pi".to_owned()),
        ],
        signal,
    )
    .await?;
    if !(200..300).contains(&status) {
        return Err(request_failure("device authorization", status, &body));
    }
    parse_device_code(&body)
}

async fn poll_for_tokens(
    client: &Arc<dyn HttpClient>,
    device: XaiDeviceCode,
    clock: Arc<dyn crate::auth::clock::AuthClock>,
    signal: &CancellationToken,
) -> Result<OAuthCredentials, AuthError> {
    poll_oauth_device_code_flow::<OAuthCredentials>(PollOptions {
        interval_seconds: device.interval_seconds,
        expires_in_seconds: Some(device.expires_in_seconds),
        wait_before_first_poll: true,
        signal: signal.clone(),
        poll: {
            let client = client.clone();
            let clock = clock.clone();
            let device = device.clone();
            let signal = signal.clone();
            Arc::new(move || {
                let client = client.clone();
                let clock = clock.clone();
                let device = device.clone();
                let signal = signal.clone();
                Box::pin(async move {
                    let (status, body) = post_form(
                        &client,
                        XAI_TOKEN_URL,
                        &[
                            (
                                "grant_type".to_owned(),
                                "urn:ietf:params:oauth:grant-type:device_code".to_owned(),
                            ),
                            ("client_id".to_owned(), XAI_CLIENT_ID.to_owned()),
                            ("device_code".to_owned(), device.device_code.clone()),
                        ],
                        &signal,
                    )
                    .await?;

                    if (200..300).contains(&status) {
                        let credential =
                            credentials_from_token_response(clock.as_ref(), &body, None)?;
                        return Ok(PollOutcome::Complete(credential));
                    }

                    match body.get("error").and_then(serde_json::Value::as_str) {
                        Some("authorization_pending") => Ok(PollOutcome::Pending),
                        Some("slow_down") => Ok(PollOutcome::SlowDown {
                            interval_seconds: body
                                .get("interval")
                                .and_then(serde_json::Value::as_f64)
                                .filter(|interval| interval.is_finite() && *interval > 0.0)
                                .map(wire_seconds_to_u64),
                        }),
                        Some("access_denied" | "authorization_denied") => Ok(PollOutcome::Failed(
                            "xAI device authorization was denied".to_owned(),
                        )),
                        Some("expired_token") => {
                            Ok(PollOutcome::Failed("xAI device code expired".to_owned()))
                        }
                        _ => Ok(PollOutcome::Failed(
                            request_failure("device token polling", status, &body).to_string(),
                        )),
                    }
                })
            })
        },
    })
    .await
}

async fn refresh_xai_token(
    client: &Arc<dyn HttpClient>,
    clock: &Arc<dyn crate::auth::clock::AuthClock>,
    refresh_token: &str,
    signal: &CancellationToken,
) -> Result<OAuthCredentials, AuthError> {
    let (status, body) = post_form(
        client,
        XAI_TOKEN_URL,
        &[
            ("grant_type".to_owned(), "refresh_token".to_owned()),
            ("client_id".to_owned(), XAI_CLIENT_ID.to_owned()),
            ("refresh_token".to_owned(), refresh_token.to_owned()),
        ],
        signal,
    )
    .await?;
    if !(200..300).contains(&status) {
        return Err(request_failure("token refresh", status, &body));
    }
    credentials_from_token_response(clock.as_ref(), &body, Some(refresh_token))
}

impl XaiOAuth {
    /// Run the interactive login flow: device-code authorization.
    #[must_use]
    pub fn login(
        &self,
        interaction: crate::auth::types::ProviderAuthInteraction,
    ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>> {
        let client = self.client.clone();
        let clock = self.clock.clone();
        Box::pin(async move {
            let device = request_device_code(&client, &interaction.signal).await?;
            (interaction.notify)(AuthEvent::DeviceCode {
                user_code: device.user_code.clone(),
                verification_uri: device
                    .verification_uri_complete
                    .clone()
                    .unwrap_or_else(|| device.verification_uri.clone()),
                interval_seconds: device.interval_seconds,
                expires_in_seconds: Some(device.expires_in_seconds),
            });
            poll_for_tokens(&client, device, clock, &interaction.signal).await
        })
    }

    /// Exchange the refresh token for a rotated credential; an unrotated
    /// refresh token survives an absent wire field.
    #[must_use]
    pub fn refresh(
        &self,
        credential: OAuthCredentials,
        signal: CancellationToken,
    ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>> {
        let client = self.client.clone();
        let clock = self.clock.clone();
        Box::pin(
            async move { refresh_xai_token(&client, &clock, &credential.refresh, &signal).await },
        )
    }

    /// Derive the request auth from a valid credential.
    #[must_use]
    pub fn to_auth(&self, credential: &OAuthCredentials) -> ModelAuth {
        ModelAuth {
            api_key: Some(credential.access.clone()),
            ..ModelAuth::default()
        }
    }

    /// The flow wired into the merged auth core's callback-based
    /// [`OAuthAuth`](crate::auth::types::OAuthAuth): the login, refresh, and
    /// derivation closures drive this flow's own client and clock. (#29)
    #[must_use]
    pub fn auth(&self) -> crate::auth::types::OAuthAuth {
        let login: crate::auth::types::OAuthLoginFn = {
            let client = Arc::clone(&self.client);
            let clock = Arc::clone(&self.clock);
            Arc::new(move |interaction| {
                let flow = Self {
                    client: Arc::clone(&client),
                    clock: Arc::clone(&clock),
                };
                flow.login(interaction)
            })
        };
        let refresh: crate::auth::types::OAuthRefreshFn = {
            let client = Arc::clone(&self.client);
            let clock = Arc::clone(&self.clock);
            Arc::new(move |credential, signal| {
                let flow = Self {
                    client: Arc::clone(&client),
                    clock: Arc::clone(&clock),
                };
                flow.refresh(credential, signal)
            })
        };
        let to_auth: crate::auth::types::OAuthToAuthFn = {
            Arc::new(|credential| {
                let auth = ModelAuth {
                    api_key: Some(credential.access),
                    ..ModelAuth::default()
                };
                Box::pin(async move { Ok(auth) })
            })
        };
        crate::auth::types::OAuthAuth {
            name: "xAI (Grok/X subscription)".to_owned(),
            is_subscription: Some(true),
            login_label: Some("Sign in with SuperGrok or X Premium".to_owned()),
            login,
            refresh,
            to_auth,
        }
    }
}
