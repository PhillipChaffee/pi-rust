//! The Radius gateway OAuth flow, ported from
//! `packages/ai/src/auth/oauth/radius.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Radius is a pi-messages gateway. OAuth client APIs live on the configured
//! gateway; only the interactive browser authorization endpoint is discovered.
//! Login offers a browser path (PKCE over a loopback callback server) and a
//! device-code path; both share the token request reader whose OAuth error
//! responses the poll loop inspects.

use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::auth::oauth::callback::{
    CallbackHandler, CallbackRequest, CallbackResponse, OAuthCallbackServer, WaitCell,
};
use crate::auth::oauth::device_code::{PollOptions, PollOutcome, poll_oauth_device_code_flow};
use crate::auth::oauth::oauth_page::{oauth_error_html, oauth_success_html};
use crate::auth::oauth::pkce::generate_pkce;
use crate::auth::oauth::{execute, form_post_request, random_uuid_v4, read_body_lossy, read_json};
use crate::auth::types::{
    AuthError, AuthEvent, AuthInteraction, AuthPrompt, AuthPromptKind, AuthSelectOption,
    BoxAuthFuture, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction,
};
use crate::http::{HttpClient, HttpMethod, HttpRequest, HttpResponse};

const CALLBACK_HOST: &str = "127.0.0.1";
const CALLBACK_PORT: u16 = 1456;
const CALLBACK_PATH: &str = "/oauth/callback";
const REDIRECT_URI: &str = "http://127.0.0.1:1456/oauth/callback";
const TOKEN_EXPIRY_SKEW_MS: i64 = 60_000;
const LOGIN_METHOD_BROWSER: &str = "browser";
const LOGIN_METHOD_DEVICE_CODE: &str = "device-code";
const OAUTH_CLIENT_ID: &str = "pi-gateway";
const OAUTH_SCOPE: &str = "gateway offline_access";
const OAUTH_DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// Prepend `https://` to a bare gateway host and strip trailing slashes,
/// upstream's `normalizeRadiusGatewayUrl` (`radius-config.ts`).
#[must_use]
pub fn normalize_radius_gateway_url(value: &str) -> String {
    // Upstream tests the scheme case-insensitively (`/^https?:\/\//iu`).
    let scheme = value.to_ascii_lowercase();
    let with_scheme = if scheme.starts_with("http://") || scheme.starts_with("https://") {
        value.to_owned()
    } else {
        format!("https://{value}")
    };
    with_scheme.trim_end_matches('/').to_owned()
}

/// The Radius OAuth flow for one gateway, wired with its [`HttpClient`] seam
/// and epoch clock.
pub struct RadiusOAuth {
    name: String,
    gateway: String,
    client: Arc<dyn HttpClient>,
    clock: Arc<dyn crate::auth::clock::AuthClock>,
}

impl std::fmt::Debug for RadiusOAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RadiusOAuth")
    }
}

impl RadiusOAuth {
    /// The flow for a gateway under the given display name; the gateway URL is
    /// normalized per [`normalize_radius_gateway_url`].
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the loaders hand the flow owned name and gateway strings"
    )]
    pub fn new(
        name: String,
        gateway: String,
        client: Arc<dyn HttpClient>,
        clock: Arc<dyn crate::auth::clock::AuthClock>,
    ) -> Self {
        Self {
            name,
            gateway: normalize_radius_gateway_url(&gateway),
            client,
            clock,
        }
    }
}

/// Resolve a root-relative path against the gateway, upstream's
/// `new URL(path, gateway)`.
fn gateway_url(gateway: &str, path: &str) -> Result<String, AuthError> {
    let base = url::Url::parse(gateway)
        .map_err(|error| AuthError(format!("invalid Radius gateway URL {gateway}: {error}")))?;
    let joined = base
        .join(path)
        .map_err(|error| AuthError(format!("invalid Radius gateway path {path}: {error}")))?;
    Ok(joined.to_string())
}

/// The discovered browser authorization endpoint, upstream's
/// `RadiusOAuthDiscovery`.
async fn load_radius_oauth_discovery(
    client: &Arc<dyn HttpClient>,
    gateway: &str,
    signal: &CancellationToken,
) -> Result<String, AuthError> {
    let request = HttpRequest {
        method: HttpMethod::Get,
        url: gateway_url(gateway, "/v1/oauth")?,
        headers: vec![("Accept".to_owned(), "application/json".to_owned())],
        body: None,
        timeout_ms: None,
        signal: signal.clone(),
    };
    let mut response = execute(client, request).await?;
    let status = response.status;
    if !(200..300).contains(&status) {
        let text = read_body_lossy(&mut response).await;
        return Err(AuthError(format!(
            "Could not load Radius OAuth config from {gateway}: {status} {text}"
        )));
    }
    let discovery = read_json(&mut response).await?;
    discovery
        .get("authorizationEndpoint")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| AuthError(format!("Invalid Radius OAuth config from {gateway}")))
}

/// A structured OAuth error response, upstream's `OAuthResponseError`: the
/// message is `{message}: {oauth_error}: {description}`, the bare error, the
/// bare description, or the status text when the body carries neither.
struct OAuthResponseError {
    #[expect(
        dead_code,
        reason = "upstream carries the response status on the error type; the port reads it while building the message"
    )]
    status: u16,
    /// The OAuth `error` code, the poll loop's `authorization_pending` etc.
    oauth_error: Option<String>,
    message: AuthError,
}

impl std::fmt::Display for OAuthResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message.0)
    }
}

impl OAuthResponseError {
    fn new(
        status: u16,
        oauth_error: Option<String>,
        description: Option<String>,
        message: &str,
    ) -> Self {
        let detail = match &oauth_error {
            Some(oauth_error) => description.as_ref().map_or_else(
                || oauth_error.clone(),
                |description| format!("{oauth_error}: {description}"),
            ),
            None => description.unwrap_or_else(|| status.to_string()),
        };
        Self {
            status,
            oauth_error,
            message: AuthError(format!("{message}: {detail}")),
        }
    }
}

/// Read one failure response into an [`OAuthResponseError`], upstream's
/// `readOAuthResponseError`: a JSON body's `error`/`error_description`, else
/// the raw body as the description.
async fn read_oauth_response_error(
    response: &mut HttpResponse,
    message: &str,
) -> OAuthResponseError {
    let text = read_body_lossy(response).await;
    let (oauth_error, description) = if text.is_empty() {
        (None, None)
    } else {
        serde_json::from_str::<Value>(&text).map_or_else(
            |_| (None, Some(text)),
            |data| {
                (
                    data.get("error").and_then(Value::as_str).map(str::to_owned),
                    data.get("error_description")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                )
            },
        )
    };
    OAuthResponseError::new(response.status, oauth_error, description, message)
}

/// The failure modes of a Radius token request: a structured OAuth error the
/// poll loop maps onto outcomes, or any other rejection, upstream's thrown
/// `OAuthResponseError` vs. rethrown error.
enum RadiusTokenError {
    /// A structured OAuth error response, upstream's `OAuthResponseError`.
    Response(OAuthResponseError),
    /// Transport or parse failure, upstream's rethrown error.
    Other(AuthError),
}

/// POST one token request, upstream's `requestOAuthToken`.
///
/// # Errors
/// Rejects with the structured OAuth error on a failure status, and with the
/// transport's message otherwise.
async fn request_oauth_token(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    gateway: &str,
    fields: Vec<(String, String)>,
    signal: &CancellationToken,
) -> Result<OAuthCredential, RadiusTokenError> {
    let request = form_post_request(
        &gateway_url(gateway, "/v1/oauth/token").map_err(RadiusTokenError::Other)?,
        &[("Accept", "application/json")],
        &fields,
        signal.clone(),
        None,
    );
    let mut response = execute(client, request)
        .await
        .map_err(RadiusTokenError::Other)?;
    let status = response.status;
    if !(200..300).contains(&status) {
        return Err(RadiusTokenError::Response(
            read_oauth_response_error(&mut response, "Radius OAuth token request failed").await,
        ));
    }

    let data = read_json(&mut response)
        .await
        .map_err(RadiusTokenError::Other)?;
    let mut credential = OAuthCredential::new(
        data.get("access_token")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        data.get("refresh_token")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        clock.now_ms()
            + crate::auth::oauth::github_copilot::wire_milliseconds(
                data.get("expires_in")
                    .and_then(Value::as_f64)
                    .unwrap_or_default(),
            )
            .saturating_mul(1000)
            - TOKEN_EXPIRY_SKEW_MS,
    );
    data.get("scope")
        .and_then(Value::as_str)
        .inspect(|scope| credential.set_extra_string("scope", *scope));
    Ok(credential)
}

/// The loopback callback server and its wait pair, upstream's
/// `startOAuthCallbackServer`.
async fn start_oauth_callback_server(
    expected_state: &str,
) -> Result<
    (
        OAuthCallbackServer,
        WaitCell<String>,
        tokio::sync::oneshot::Receiver<Option<String>>,
    ),
    AuthError,
> {
    let (wait_cell, receiver) = WaitCell::<String>::new();
    let expected_state = expected_state.to_owned();
    let settle = wait_cell.clone();
    let handler: CallbackHandler = Arc::new(move |request: CallbackRequest| {
        let settle = settle.clone();
        let expected_state = expected_state.clone();
        Box::pin(async move {
            let (status, html) = if request.path != CALLBACK_PATH {
                (404, oauth_error_html("Callback route not found.", None))
            } else if request.query_value("state") != Some(expected_state.as_str()) {
                (400, oauth_error_html("OAuth state mismatch.", None))
            } else if let Some(error) = request.query_value("error") {
                let description = request
                    .query_value("error_description")
                    .unwrap_or(error)
                    .to_owned();
                settle.settle(None).await;
                (400, oauth_error_html(&description, None))
            } else {
                match request.query_value("code") {
                    None => (400, oauth_error_html("Missing authorization code.", None)),
                    Some(code) => {
                        settle.settle(Some(code.to_owned())).await;
                        (
                            200,
                            oauth_success_html("Signed in to Radius. You may now close this page."),
                        )
                    }
                }
            };
            CallbackResponse { status, html }
        })
    });
    let server = OAuthCallbackServer::bind(CALLBACK_HOST, CALLBACK_PORT, handler).await?;
    Ok((server, wait_cell, receiver))
}

/// The browser login path, upstream's `loginWithBrowser`.
///
/// # Errors
/// Rejects with `Login cancelled`, `OAuth callback did not complete.`, and
/// the token request's errors.
async fn login_with_browser(
    client: &Arc<dyn HttpClient>,
    clock: &Arc<dyn crate::auth::clock::AuthClock>,
    gateway: &str,
    authorization_endpoint: &str,
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential, AuthError> {
    let pkce = generate_pkce()?;
    let state = random_uuid_v4()?;
    let authorize_query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("response_type", "code")
        .append_pair("client_id", OAUTH_CLIENT_ID)
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", OAUTH_SCOPE)
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("handoff", "url")
        .append_pair("state", &state)
        .finish();
    let authorize_url = format!("{authorization_endpoint}?{authorize_query}");

    let (server, wait_cell, receiver) = start_oauth_callback_server(&state).await?;
    // Upstream's abort listener settles the wait with null; the watcher ends
    // with the wait.
    let abort_task = tokio::spawn({
        let wait_cell = wait_cell.clone();
        let signal = interaction.signal.clone();
        async move {
            signal.cancelled().await;
            wait_cell.settle(None).await;
        }
    });
    interaction.notify(AuthEvent::Progress {
        message: format!("Listening for OAuth callback on {REDIRECT_URI}"),
    });
    interaction.notify(AuthEvent::AuthUrl {
        url: authorize_url,
        instructions: Some("Continue in your browser.".to_owned()),
    });

    let code: Option<String> = receiver.await.unwrap_or(None);
    abort_task.abort();
    server.close();
    let Some(code) = code else {
        if interaction.signal.is_cancelled() {
            return Err(AuthError("Login cancelled".to_owned()));
        }
        return Err(AuthError("OAuth callback did not complete.".to_owned()));
    };

    request_oauth_token(
        client,
        clock.as_ref(),
        gateway,
        vec![
            ("grant_type".to_owned(), "authorization_code".to_owned()),
            ("client_id".to_owned(), OAUTH_CLIENT_ID.to_owned()),
            ("redirect_uri".to_owned(), REDIRECT_URI.to_owned()),
            ("code".to_owned(), code),
            ("code_verifier".to_owned(), pkce.verifier),
        ],
        &interaction.signal,
    )
    .await
    .map_err(|error| match error {
        RadiusTokenError::Response(response_error) => AuthError(response_error.to_string()),
        RadiusTokenError::Other(auth_error) => auth_error,
    })
}

/// One device authorization response, upstream's `DeviceAuthorizationResponse`.
#[derive(Clone)]
struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri: String,
    expires_in_seconds: u64,
    interval_seconds: Option<u64>,
}

/// POST the device authorization endpoint, upstream's
/// `requestDeviceAuthorization`.
///
/// # Errors
/// Rejects with the structured OAuth error's message on failure, and with
/// the missing-fields message when the response is incomplete.
async fn request_device_authorization(
    client: &Arc<dyn HttpClient>,
    gateway: &str,
    signal: &CancellationToken,
) -> Result<DeviceAuthorization, AuthError> {
    let request = form_post_request(
        &gateway_url(gateway, "/v1/oauth/device")?,
        &[("Accept", "application/json")],
        &[
            ("client_id".to_owned(), OAUTH_CLIENT_ID.to_owned()),
            ("scope".to_owned(), OAUTH_SCOPE.to_owned()),
        ],
        signal.clone(),
        None,
    );
    let mut response = execute(client, request).await?;
    let status = response.status;
    if !(200..300).contains(&status) {
        return Err(AuthError(
            read_oauth_response_error(&mut response, "Radius OAuth device authorization failed")
                .await
                .to_string(),
        ));
    }

    let data = read_json(&mut response).await?;
    let device_code = data
        .get("device_code")
        .and_then(Value::as_str)
        .filter(|code| !code.is_empty());
    let user_code = data
        .get("user_code")
        .and_then(Value::as_str)
        .filter(|code| !code.is_empty());
    let verification_uri = data
        .get("verification_uri")
        .and_then(Value::as_str)
        .filter(|uri| !uri.is_empty());
    let expires_in_seconds = data
        .get("expires_in")
        .and_then(Value::as_f64)
        .filter(|expires| *expires > 0.0);
    let interval_seconds = data
        .get("interval")
        .and_then(Value::as_f64)
        .filter(|interval| *interval > 0.0);
    let (Some(device_code), Some(user_code), Some(verification_uri), Some(expires_in_seconds)) =
        (device_code, user_code, verification_uri, expires_in_seconds)
    else {
        return Err(AuthError(
            "Radius OAuth device authorization response is missing required fields".to_owned(),
        ));
    };
    Ok(DeviceAuthorization {
        device_code: device_code.to_owned(),
        user_code: user_code.to_owned(),
        verification_uri: verification_uri.to_owned(),
        expires_in_seconds: crate::auth::oauth::github_copilot::wire_seconds(expires_in_seconds),
        interval_seconds: interval_seconds.map(crate::auth::oauth::github_copilot::wire_seconds),
    })
}

/// The device-code login path, upstream's `loginWithDeviceCode`.
///
/// # Errors
/// Rejects with the flow's timeout messages and the poll's failed messages.
async fn login_with_device_code(
    client: &Arc<dyn HttpClient>,
    clock: &Arc<dyn crate::auth::clock::AuthClock>,
    gateway: &str,
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential, AuthError> {
    let device = request_device_authorization(client, gateway, &interaction.signal).await?;
    interaction.notify(AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: device.verification_uri.clone(),
        interval_seconds: device.interval_seconds,
        expires_in_seconds: Some(device.expires_in_seconds),
    });

    poll_oauth_device_code_flow::<OAuthCredential>(PollOptions {
        interval_seconds: device.interval_seconds,
        expires_in_seconds: Some(device.expires_in_seconds),
        wait_before_first_poll: false,
        signal: interaction.signal.clone(),
        poll: {
            let client = Arc::clone(client);
            let clock = Arc::clone(clock);
            let gateway = gateway.to_owned();
            let device = device.clone();
            let signal = interaction.signal.clone();
            Arc::new(move || {
                let client = Arc::clone(&client);
                let clock = Arc::clone(&clock);
                let gateway = gateway.clone();
                let device = device.clone();
                let signal = signal.clone();
                Box::pin(async move {
                    match request_oauth_token(
                        &client,
                        clock.as_ref(),
                        &gateway,
                        vec![
                            (
                                "grant_type".to_owned(),
                                OAUTH_DEVICE_CODE_GRANT_TYPE.to_owned(),
                            ),
                            ("client_id".to_owned(), OAUTH_CLIENT_ID.to_owned()),
                            ("device_code".to_owned(), device.device_code.clone()),
                        ],
                        &signal,
                    )
                    .await
                    {
                        Ok(credentials) => Ok(PollOutcome::Complete(credentials)),
                        Err(RadiusTokenError::Response(error)) => {
                            match error.oauth_error.as_deref() {
                                Some("authorization_pending") => Ok(PollOutcome::Pending),
                                Some("slow_down") => Ok(PollOutcome::SlowDown {
                                    interval_seconds: None,
                                }),
                                Some("expired_token") => Ok(PollOutcome::Failed(
                                    "Device authorization expired.".to_owned(),
                                )),
                                Some("access_denied") => Ok(PollOutcome::Failed(
                                    "Device authorization was denied.".to_owned(),
                                )),
                                _ => Err(AuthError(error.to_string())),
                            }
                        }
                        Err(RadiusTokenError::Other(error)) => Err(error),
                    }
                })
            })
        },
    })
    .await
}

impl OAuthAuth for RadiusOAuth {
    fn name(&self) -> &str {
        &self.name
    }

    fn login(
        &self,
        interaction: ProviderAuthInteraction,
    ) -> BoxAuthFuture<Result<OAuthCredential, AuthError>> {
        let name = self.name.clone();
        let gateway = self.gateway.clone();
        let client = Arc::clone(&self.client);
        let clock = Arc::clone(&self.clock);
        Box::pin(async move {
            let login_method = interaction
                .prompt(AuthPrompt::new(
                    AuthPromptKind::Select(vec![
                        AuthSelectOption {
                            id: LOGIN_METHOD_BROWSER.to_owned(),
                            label: "Sign in with browser (recommended)".to_owned(),
                            description: None,
                        },
                        AuthSelectOption {
                            id: LOGIN_METHOD_DEVICE_CODE.to_owned(),
                            label: "Sign in with device code (when signing in from another device)"
                                .to_owned(),
                            description: None,
                        },
                    ]),
                    format!("Sign in to {name}:"),
                ))
                .await?;
            if login_method == LOGIN_METHOD_DEVICE_CODE {
                return login_with_device_code(&client, &clock, &gateway, &interaction).await;
            }
            if login_method == LOGIN_METHOD_BROWSER {
                let authorization_endpoint =
                    load_radius_oauth_discovery(&client, &gateway, &interaction.signal).await?;
                return login_with_browser(
                    &client,
                    &clock,
                    &gateway,
                    &authorization_endpoint,
                    &interaction,
                )
                .await;
            }
            Err(AuthError(format!(
                "Unknown {name} sign-in method: {login_method}"
            )))
        })
    }

    fn refresh(
        &self,
        credential: &OAuthCredential,
        signal: CancellationToken,
    ) -> BoxAuthFuture<Result<OAuthCredential, AuthError>> {
        let client = Arc::clone(&self.client);
        let clock = Arc::clone(&self.clock);
        let gateway = self.gateway.clone();
        let refresh_token = credential.refresh.clone();
        Box::pin(async move {
            request_oauth_token(
                &client,
                clock.as_ref(),
                &gateway,
                vec![
                    ("grant_type".to_owned(), "refresh_token".to_owned()),
                    ("client_id".to_owned(), OAUTH_CLIENT_ID.to_owned()),
                    ("refresh_token".to_owned(), refresh_token),
                ],
                &signal,
            )
            .await
            .map_err(|error| match error {
                RadiusTokenError::Response(response_error) => AuthError(response_error.to_string()),
                RadiusTokenError::Other(auth_error) => auth_error,
            })
        })
    }

    fn to_auth(&self, credential: &OAuthCredential) -> ModelAuth {
        ModelAuth::api_key(credential.access.clone())
    }
}
