//! The OpenAI Codex (`ChatGPT` OAuth) flow, ported from
//! `packages/ai/src/auth/oauth/openai-codex.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Login offers a browser path (PKCE over a loopback callback server raced
//! against a manual code prompt) and a device-code path; both end in the
//! shared token response reader, which also extracts the `ChatGPT` account id
//! from the access token's JWT.

use std::fmt::Write as _;
use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::auth::oauth::callback::{
    CallbackHandler, CallbackRequest, CallbackResponse, OAuthCallbackServer, WaitCell,
};
use crate::auth::oauth::device_code::{PollOptions, PollOutcome, poll_oauth_device_code_flow};
use crate::auth::oauth::github_copilot::status_reason;
use crate::auth::oauth::oauth_page::{oauth_error_html, oauth_success_html};
use crate::auth::oauth::pkce::{Pkce, decode_json_segment, generate_pkce};
use crate::auth::oauth::{
    execute, form_post_request, json_post_request, parse_authorization_code_state, read_body_lossy,
    read_json,
};
use crate::auth::types::{
    AuthError, AuthEvent, AuthInteraction, AuthPrompt, AuthPromptKind, AuthSelectOption,
    BoxAuthFuture, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction,
};
use crate::http::{HttpClient, HttpResponse};
use crate::utils::provider_env::get_provider_env_value;

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const DEVICE_USER_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const DEVICE_VERIFICATION_URI: &str = "https://auth.openai.com/codex/device";
const DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
const DEVICE_CODE_TIMEOUT_SECONDS: u64 = 15 * 60;
const OPENAI_CODEX_BROWSER_LOGIN_METHOD: &str = "browser";
const OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD: &str = "device_code";
const SCOPE: &str = "openid profile email offline_access";
const CALLBACK_PORT: u16 = 1455;
const CALLBACK_PATH: &str = "/auth/callback";
/// The JWT claim object holding the `ChatGPT` account ids, the wire's
/// `https://api.openai.com/auth` claim path.
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

/// A token triple, upstream's `OAuthToken`.
struct OAuthToken {
    access: String,
    refresh: String,
    expires: i64,
}

/// The fields a successful device-auth poll carries, upstream's
/// `DeviceTokenSuccess`.
struct DeviceTokenSuccess {
    authorization_code: String,
    code_verifier: String,
}

/// The OpenAI Codex OAuth flow, wired with its [`HttpClient`] seam and epoch
/// clock.
pub struct OpenAICodexOAuth {
    client: Arc<dyn HttpClient>,
    clock: Arc<dyn crate::auth::clock::AuthClock>,
}

impl std::fmt::Debug for OpenAICodexOAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpenAICodexOAuth")
    }
}

impl OpenAICodexOAuth {
    /// The flow over the given client and clock.
    #[must_use]
    pub fn new(client: Arc<dyn HttpClient>, clock: Arc<dyn crate::auth::clock::AuthClock>) -> Self {
        Self { client, clock }
    }
}

fn callback_host() -> String {
    get_provider_env_value("PI_OAUTH_CALLBACK_HOST", None).unwrap_or_else(|| "127.0.0.1".to_owned())
}

/// The 16-byte hex state, upstream's `createState`.
fn create_state() -> Result<String, AuthError> {
    let mut bytes = [0_u8; 16];
    getrandom::fill(&mut bytes).map_err(|error| AuthError(format!("getrandom failed: {error}")))?;
    let mut state = String::with_capacity(32);
    for byte in bytes {
        let _ = write!(state, "{byte:02x}");
    }
    Ok(state)
}

/// The authorize URL and its PKCE verifier plus state, upstream's
/// `createAuthorizationFlow`.
fn create_authorization_flow(pkce: &Pkce, state: &str) -> String {
    let query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("response_type", "code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPE)
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state)
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("originator", "pi")
        .finish();
    format!("{AUTHORIZE_URL}?{query}")
}

/// Read one token exchange or refresh response, upstream's `readTokenResponse`.
///
/// # Errors
/// Rejects with the wire's failure message on a non-2xx status and with the
/// missing-fields message when the token triple is incomplete.
async fn read_token_response(
    response: &mut HttpResponse,
    operation: &str,
    clock: &dyn crate::auth::clock::AuthClock,
) -> Result<OAuthToken, AuthError> {
    let status = response.status;
    if !(200..300).contains(&status) {
        let text = read_body_lossy(response).await;
        let detail = if text.is_empty() {
            status_reason(status).to_owned()
        } else {
            text
        };
        return Err(AuthError(format!(
            "OpenAI Codex token {operation} failed ({status}): {detail}"
        )));
    }
    let json = read_json(response).await?;
    let access = json
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|token| !token.is_empty());
    let refresh = json
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|token| !token.is_empty());
    let expires_in = json.get("expires_in").and_then(Value::as_f64);
    let (Some(access), Some(refresh), Some(expires_in)) = (access, refresh, expires_in) else {
        return Err(AuthError(format!(
            "OpenAI Codex token {operation} response missing fields: {json}"
        )));
    };
    Ok(OAuthToken {
        access: access.to_owned(),
        refresh: refresh.to_owned(),
        expires: clock.now_ms()
            + crate::auth::oauth::github_copilot::wire_milliseconds(expires_in)
                .saturating_mul(1000),
    })
}

/// Exchange one authorization code for tokens, upstream's
/// `exchangeAuthorizationCode`.
async fn exchange_authorization_code(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    signal: &CancellationToken,
) -> Result<OAuthToken, AuthError> {
    let request = form_post_request(
        TOKEN_URL,
        &[],
        &[
            ("grant_type".to_owned(), "authorization_code".to_owned()),
            ("client_id".to_owned(), CLIENT_ID.to_owned()),
            ("code".to_owned(), code.to_owned()),
            ("code_verifier".to_owned(), verifier.to_owned()),
            ("redirect_uri".to_owned(), redirect_uri.to_owned()),
        ],
        signal.clone(),
        None,
    );
    let mut response = execute(client, request).await?;
    read_token_response(&mut response, "exchange", clock).await
}

/// Refresh the OpenAI Codex token, upstream's `refreshAccessToken`; the
/// transport failure is re-reported with the flow's own prefix.
async fn refresh_access_token(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    refresh_token: &str,
    signal: &CancellationToken,
) -> Result<OAuthToken, AuthError> {
    let request = form_post_request(
        TOKEN_URL,
        &[],
        &[
            ("grant_type".to_owned(), "refresh_token".to_owned()),
            ("refresh_token".to_owned(), refresh_token.to_owned()),
            ("client_id".to_owned(), CLIENT_ID.to_owned()),
        ],
        signal.clone(),
        None,
    );
    let mut response = match execute(client, request).await {
        Ok(response) => response,
        Err(error) => {
            return Err(AuthError(format!(
                "OpenAI Codex token refresh error: {}",
                error.0
            )));
        }
    };
    read_token_response(&mut response, "refresh", clock).await
}

/// Start the device-code login, upstream's `startOpenAICodexDeviceAuth`.
///
/// # Errors
/// Rejects with the not-enabled message on 404, the status message on other
/// failures, and the invalid-response message when the fields are incomplete.
async fn start_openai_codex_device_auth(
    client: &Arc<dyn HttpClient>,
    signal: &CancellationToken,
) -> Result<DeviceAuthInfo, AuthError> {
    let request = json_post_request(
        DEVICE_USER_CODE_URL,
        &[],
        &serde_json::json!({ "client_id": CLIENT_ID }),
        signal.clone(),
        None,
    );
    let mut response = execute(client, request).await?;
    let status = response.status;
    if !(200..300).contains(&status) {
        if status == 404 {
            return Err(AuthError(
                "OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL."
                    .to_owned(),
            ));
        }
        let response_body = read_body_lossy(&mut response).await;
        return Err(AuthError(format!(
            "OpenAI Codex device code request failed with status {status}{}",
            if response_body.is_empty() {
                String::new()
            } else {
                format!(": {response_body}")
            }
        )));
    }

    let json = read_json(&mut response).await?;
    let interval_seconds = match json.get("interval") {
        Some(Value::String(text)) => text.trim().parse::<f64>().ok(),
        Some(Value::Number(number)) => number.as_f64(),
        _ => None,
    };
    let device_auth_id = json
        .get("device_auth_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty());
    let user_code = json
        .get("user_code")
        .and_then(Value::as_str)
        .filter(|code| !code.is_empty());
    let interval_seconds =
        interval_seconds.filter(|interval| interval.is_finite() && *interval >= 0.0);
    let (Some(device_auth_id), Some(user_code), Some(interval_seconds)) =
        (device_auth_id, user_code, interval_seconds)
    else {
        return Err(AuthError(format!(
            "Invalid OpenAI Codex device code response: {json}"
        )));
    };
    Ok(DeviceAuthInfo {
        device_auth_id: device_auth_id.to_owned(),
        user_code: user_code.to_owned(),
        interval_seconds: crate::auth::oauth::github_copilot::wire_seconds(interval_seconds),
    })
}

/// Poll the device auth endpoint, upstream's `pollOpenAICodexDeviceAuth`.
///
/// # Errors
/// Rejects with the flow's timeout messages, the poll's failed message, and
/// the poll step's own transport errors.
async fn poll_openai_codex_device_auth(
    client: &Arc<dyn HttpClient>,
    device: &DeviceAuthInfo,
    signal: &CancellationToken,
) -> Result<DeviceTokenSuccess, AuthError> {
    poll_oauth_device_code_flow::<DeviceTokenSuccess>(PollOptions {
        interval_seconds: Some(device.interval_seconds),
        expires_in_seconds: Some(DEVICE_CODE_TIMEOUT_SECONDS),
        wait_before_first_poll: false,
        signal: signal.clone(),
        poll: {
            let client = Arc::clone(client);
            let device = device.clone();
            let signal = signal.clone();
            Arc::new(move || {
                let client = Arc::clone(&client);
                let device = device.clone();
                let signal = signal.clone();
                Box::pin(async move {
                    let request = json_post_request(
                        DEVICE_TOKEN_URL,
                        &[],
                        &serde_json::json!({
                            "device_auth_id": device.device_auth_id,
                            "user_code": device.user_code,
                        }),
                        signal.clone(),
                        None,
                    );
                    let mut response = execute(&client, request).await?;
                    let status = response.status;

                    if (200..300).contains(&status) {
                        let json = read_json(&mut response).await?;
                        let authorization_code = json
                            .get("authorization_code")
                            .and_then(Value::as_str)
                            .filter(|code| !code.is_empty());
                        let code_verifier = json
                            .get("code_verifier")
                            .and_then(Value::as_str)
                            .filter(|verifier| !verifier.is_empty());
                        let (Some(authorization_code), Some(code_verifier)) =
                            (authorization_code, code_verifier)
                        else {
                            return Ok(PollOutcome::Failed(format!(
                                "Invalid OpenAI Codex device auth token response: {json}"
                            )));
                        };
                        return Ok(PollOutcome::Complete(DeviceTokenSuccess {
                            authorization_code: authorization_code.to_owned(),
                            code_verifier: code_verifier.to_owned(),
                        }));
                    }

                    if status == 403 || status == 404 {
                        return Ok(PollOutcome::Pending);
                    }

                    let response_body = read_body_lossy(&mut response).await;
                    let error_code =
                        serde_json::from_str::<Value>(&response_body)
                            .ok()
                            .and_then(|json| match json.get("error") {
                                Some(Value::Object(code_map)) => code_map.get("code").cloned(),
                                Some(Value::Null) | None => None,
                                Some(error) => Some(error.clone()),
                            });
                    let Some(Value::String(error_code)) = error_code else {
                        return Ok(PollOutcome::Failed(format!(
                            "OpenAI Codex device auth failed with status {status}{}",
                            if response_body.is_empty() {
                                String::new()
                            } else {
                                format!(": {response_body}")
                            }
                        )));
                    };
                    match error_code.as_str() {
                        "deviceauth_authorization_pending" => Ok(PollOutcome::Pending),
                        "slow_down" => Ok(PollOutcome::SlowDown {
                            interval_seconds: None,
                        }),
                        _ => Ok(PollOutcome::Failed(format!(
                            "OpenAI Codex device auth failed with status {status}{}",
                            if response_body.is_empty() {
                                String::new()
                            } else {
                                format!(": {response_body}")
                            }
                        ))),
                    }
                })
            })
        },
    })
    .await
}

/// The `ChatGPT` account id carried in the access token's auth claim, upstream's
/// `getAccountId`.
fn get_account_id(access_token: &str) -> Option<String> {
    let parts: Vec<&str> = access_token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let payload = decode_json_segment::<serde_json::Map<String, Value>>(parts[1]).ok()?;
    let account_id = payload
        .get(JWT_CLAIM_PATH)?
        .get("chatgpt_account_id")?
        .as_str()
        .filter(|account_id| !account_id.is_empty())?;
    Some(account_id.to_owned())
}

/// Derive the stored credential from a token, requiring the account id,
/// upstream's `credentialsFromToken`.
///
/// # Errors
/// Rejects with `Failed to extract accountId from token` when the JWT claim
/// is absent or empty.
fn credentials_from_token(token: OAuthToken) -> Result<OAuthCredential, AuthError> {
    let Some(account_id) = get_account_id(&token.access) else {
        return Err(AuthError(
            "Failed to extract accountId from token".to_owned(),
        ));
    };
    let mut credential = OAuthCredential::new(token.access, token.refresh, token.expires);
    credential.set_extra_string("accountId", account_id);
    Ok(credential)
}

/// Exchange an authorization code for the stored credential, upstream's
/// `exchangeAuthorizationCodeForCredentials`.
async fn exchange_authorization_code_for_credentials(
    client: &Arc<dyn HttpClient>,
    clock: &Arc<dyn crate::auth::clock::AuthClock>,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    signal: &CancellationToken,
) -> Result<OAuthCredential, AuthError> {
    let token =
        exchange_authorization_code(client, clock.as_ref(), code, verifier, redirect_uri, signal)
            .await?;
    credentials_from_token(token)
}

/// The loopback callback server, upstream's `startLocalOAuthServer`; the
/// receiver settles with the authorization code.
async fn start_callback_server(
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
                (400, oauth_error_html("State mismatch.", None))
            } else {
                match request.query_value("code") {
                    None => (400, oauth_error_html("Missing authorization code.", None)),
                    Some(code) => {
                        settle.settle(Some(code.to_owned())).await;
                        (
                            200,
                            oauth_success_html(
                                "OpenAI authentication completed. You can close this window.",
                            ),
                        )
                    }
                }
            };
            CallbackResponse { status, html }
        })
    });
    let server = OAuthCallbackServer::bind(&callback_host(), CALLBACK_PORT, handler).await?;
    Ok((server, wait_cell, receiver))
}

/// The browser login race, upstream's `loginOpenAICodex`.
async fn login_openai_codex_browser(
    client: &Arc<dyn HttpClient>,
    clock: &Arc<dyn crate::auth::clock::AuthClock>,
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential, AuthError> {
    let pkce = generate_pkce()?;
    let state = create_state()?;
    let authorize_url = create_authorization_flow(&pkce, &state);
    let (server, wait_cell, receiver) = start_callback_server(&state).await?;
    let manual_abort = CancellationToken::new();
    let manual_prompt = AuthPrompt {
        kind: AuthPromptKind::ManualCode,
        message:
            "Complete login in your browser, or paste the authorization code / redirect URL here:"
                .to_owned(),
        placeholder: Some(REDIRECT_URI.to_owned()),
        signal: Some(manual_abort.clone()),
    };

    interaction.notify(AuthEvent::AuthUrl {
        url: authorize_url,
        instructions: Some("A browser window should open. Complete login to finish.".to_owned()),
    });

    // The manual prompt runs detached, upstream's manualPromise: its
    // completion (input or rejection) cancels the callback wait, and so does
    // the interaction's own cancellation.
    let manual = tokio::spawn({
        let interaction = interaction.clone();
        let wait_cell = wait_cell.clone();
        async move {
            let result = interaction.prompt(manual_prompt).await;
            wait_cell.settle(None).await;
            result
        }
    });

    let mut code: Option<String> = None;
    let mut early_error: Option<AuthError> = None;
    match receiver.await.unwrap_or(None) {
        Some(callback_code) => code = Some(callback_code),
        None => match manual.await {
            Ok(Ok(input)) => {
                let (parsed_code, parsed_state) = parse_authorization_code_state(&input);
                if parsed_state.as_ref().is_some_and(|parsed| parsed != &state) {
                    manual_abort.cancel();
                    server.close();
                    return Err(AuthError("State mismatch".to_owned()));
                }
                code = parsed_code;
            }
            Ok(Err(error)) => early_error = Some(error),
            Err(join_error) => early_error = Some(AuthError(join_error.to_string())),
        },
    }

    let credential = if let Some(error) = early_error {
        Err(error)
    } else {
        match code {
            Some(code) => {
                exchange_authorization_code_for_credentials(
                    &client.clone(),
                    clock,
                    &code,
                    &pkce.verifier,
                    REDIRECT_URI,
                    &interaction.signal,
                )
                .await
            }
            None => Err(AuthError("Missing authorization code".to_owned())),
        }
    };
    manual_abort.cancel();
    server.close();
    credential
}

/// The device-code login path, upstream's `loginOpenAICodexDeviceCode`.
async fn login_openai_codex_device_code(
    client: &Arc<dyn HttpClient>,
    clock: &Arc<dyn crate::auth::clock::AuthClock>,
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential, AuthError> {
    let device = start_openai_codex_device_auth(client, &interaction.signal).await?;
    interaction.notify(AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: DEVICE_VERIFICATION_URI.to_owned(),
        interval_seconds: Some(device.interval_seconds),
        expires_in_seconds: Some(DEVICE_CODE_TIMEOUT_SECONDS),
    });
    let code = poll_openai_codex_device_auth(client, &device, &interaction.signal).await?;
    exchange_authorization_code_for_credentials(
        client,
        clock,
        &code.authorization_code,
        &code.code_verifier,
        DEVICE_REDIRECT_URI,
        &interaction.signal,
    )
    .await
}

/// One device authorization start response, upstream's `DeviceAuthInfo`.
#[derive(Clone)]
struct DeviceAuthInfo {
    device_auth_id: String,
    user_code: String,
    interval_seconds: u64,
}

impl OAuthAuth for OpenAICodexOAuth {
    #[expect(
        clippy::unnecessary_literal_bound,
        reason = "the trait signature ties the returned str to &self; the name is a constant"
    )]
    fn name(&self) -> &str {
        "OpenAI (ChatGPT Plus/Pro)"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    fn login(
        &self,
        interaction: ProviderAuthInteraction,
    ) -> BoxAuthFuture<Result<OAuthCredential, AuthError>> {
        let client = Arc::clone(&self.client);
        let clock = Arc::clone(&self.clock);
        Box::pin(async move {
            let method = interaction
                .prompt(AuthPrompt::new(
                    AuthPromptKind::Select(vec![
                        AuthSelectOption {
                            id: OPENAI_CODEX_BROWSER_LOGIN_METHOD.to_owned(),
                            label: "Browser login (default)".to_owned(),
                            description: None,
                        },
                        AuthSelectOption {
                            id: OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD.to_owned(),
                            label: "Device code login (headless)".to_owned(),
                            description: None,
                        },
                    ]),
                    "Select OpenAI Codex login method:",
                ))
                .await?;
            if method == OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD {
                return login_openai_codex_device_code(&client, &clock, &interaction).await;
            }
            if method != OPENAI_CODEX_BROWSER_LOGIN_METHOD {
                return Err(AuthError(format!(
                    "Unknown OpenAI Codex login method: {method}"
                )));
            }
            login_openai_codex_browser(&client, &clock, &interaction).await
        })
    }

    fn refresh(
        &self,
        credential: &OAuthCredential,
        signal: CancellationToken,
    ) -> BoxAuthFuture<Result<OAuthCredential, AuthError>> {
        let client = Arc::clone(&self.client);
        let clock = Arc::clone(&self.clock);
        let refresh_token = credential.refresh.clone();
        Box::pin(async move {
            let token =
                refresh_access_token(&client, clock.as_ref(), &refresh_token, &signal).await?;
            credentials_from_token(token)
        })
    }

    fn to_auth(&self, credential: &OAuthCredential) -> ModelAuth {
        ModelAuth::api_key(credential.access.clone())
    }
}
