//! The Anthropic OAuth flow (Claude Pro/Max), ported from
//! `packages/ai/src/auth/oauth/anthropic.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! PKCE login over a loopback callback server raced against a manual code
//! prompt; token exchange and refresh over the [`HttpClient`] seam.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::auth::oauth::callback::{
    CallbackHandler, CallbackRequest, CallbackResponse, OAuthCallbackServer, WaitCell,
};
use crate::auth::oauth::oauth_page::{oauth_error_html, oauth_success_html};
use crate::auth::oauth::pkce::{Pkce, generate_pkce};
use crate::auth::oauth::{
    auth_error, execute, json_post_request, oauth_credentials, parse_authorization_code_state,
    read_body_lossy, wire_seconds_to_i64,
};
use crate::auth::types::{AuthError, AuthEvent, AuthPrompt, AuthPromptKind, OAuthCredentials};
use crate::http::HttpClient;
use crate::auth::types::ModelAuth;
use crate::types::BoxedFuture;
use crate::utils::provider_env::get_provider_env_value;

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const CALLBACK_PORT: u16 = 53692;
const CALLBACK_PATH: &str = "/callback";
// The authorize URL carries localhost while the server binds the configured
// callback interface: the browser reaches localhost, the server listens on
// PI_OAUTH_CALLBACK_HOST.
const REDIRECT_URI: &str = "http://localhost:53692/callback";
const SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const EXCHANGE_TIMEOUT_MS: u64 = 30_000;
const EXPIRY_SKEW_MS: i64 = 5 * 60 * 1000;

/// The Anthropic OAuth flow, wired with its [`HttpClient`] seam and epoch
/// clock.
pub struct AnthropicOAuth {
    client: Arc<dyn HttpClient>,
    clock: Arc<dyn crate::auth::clock::AuthClock>,
}

impl std::fmt::Debug for AnthropicOAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AnthropicOAuth")
    }
}

impl AnthropicOAuth {
    /// The flow over the given client and clock.
    #[must_use]
    pub fn new(client: Arc<dyn HttpClient>, clock: Arc<dyn crate::auth::clock::AuthClock>) -> Self {
        Self { client, clock }
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
            let client = Arc::clone(&self.client);
            let clock = Arc::clone(&self.clock);
            Arc::new(move |credential| {
                let flow = Self {
                    client: Arc::clone(&client),
                    clock: Arc::clone(&clock),
                };
                let auth = flow.to_auth(&credential);
                Box::pin(async move { Ok(auth) })
            })
        };
        crate::auth::types::OAuthAuth {
            name: "Anthropic (Claude Pro/Max)".to_owned(),
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
        let client = self.client.clone();
        let clock = self.clock.clone();
        Box::pin(async move { login_anthropic(&client, &clock, &interaction).await })
    }

    /// Exchange the refresh token for a rotated credential.
    pub fn refresh(
        &self,
        credential: OAuthCredentials,
        signal: CancellationToken,
    ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>> {
        let client = self.client.clone();
        let clock = self.clock.clone();
        let refresh_token = credential.refresh.clone();
        Box::pin(async move {
            refresh_anthropic_token(&client, clock.as_ref(), &refresh_token, &signal).await
        })
    }

    /// Derive the request auth from a valid credential.
    #[must_use]
    pub fn to_auth(&self, credential: &OAuthCredentials) -> ModelAuth {
        ModelAuth {
            api_key: Some(credential.access.clone()),
            ..ModelAuth::default()
        }
    }
}

fn callback_host() -> String {
    get_provider_env_value("PI_OAUTH_CALLBACK_HOST", None).unwrap_or_else(|| "127.0.0.1".to_owned())
}

/// Build the authorize URL the browser opens.
#[must_use]
fn authorize_url(pkce: &Pkce) -> String {
    let query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("code", "true")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPES)
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", &pkce.verifier)
        .finish();
    format!("{AUTHORIZE_URL}?{query}")
}

/// Start the loopback callback server; the returned receiver settles with
/// `Some((code, state))` on a valid callback, `None` on cancel.
async fn start_callback_server(
    expected_state: &str,
) -> Result<
    (
        OAuthCallbackServer,
        WaitCell<(String, String)>,
        tokio::sync::oneshot::Receiver<Option<(String, String)>>,
    ),
    AuthError,
> {
    let (wait_cell, receiver) = WaitCell::<(String, String)>::new();
    let expected_state = expected_state.to_owned();
    let settle = wait_cell.clone();
    let handler: CallbackHandler = Arc::new(move |request: CallbackRequest| {
        let settle = settle.clone();
        let expected_state = expected_state.clone();
        Box::pin(async move {
            let (status, html) = if request.path != CALLBACK_PATH {
                (404, oauth_error_html("Callback route not found.", None))
            } else if let Some(error) = request.query_value("error") {
                (
                    400,
                    oauth_error_html(
                        "Anthropic authentication did not complete.",
                        Some(&format!("Error: {error}")),
                    ),
                )
            } else {
                let code = request.query_value("code");
                let state = request.query_value("state");
                match (code, state) {
                    (None, _) | (Some(_), None) => (
                        400,
                        oauth_error_html("Missing code or state parameter.", None),
                    ),
                    (Some(_), Some(state)) if state != expected_state => {
                        (400, oauth_error_html("State mismatch.", None))
                    }
                    (Some(code), Some(state)) => {
                        settle
                            .settle(Some((code.to_owned(), state.to_owned())))
                            .await;
                        (
                            200,
                            oauth_success_html(
                                "Anthropic authentication completed. You can close this window.",
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

/// POST one token request, returning the raw body string.
async fn post_token_json(
    client: &Arc<dyn HttpClient>,
    body: serde_json::Value,
    signal: &CancellationToken,
) -> Result<String, AuthError> {
    let request = json_post_request(
        TOKEN_URL,
        &[("Accept", "application/json")],
        &body,
        signal.clone(),
        Some(EXCHANGE_TIMEOUT_MS),
    );
    let mut response = execute(client, request).await?;
    let status = response.status;
    let body_text = read_body_lossy(&mut response).await;
    if !(200..300).contains(&status) {
        return Err(auth_error(format!(
            "HTTP request failed. status={status}; url={TOKEN_URL}; body={body_text}"
        )));
    }
    Ok(body_text)
}

/// Exchange one authorization code for tokens, upstream's
/// `exchangeAuthorizationCode`.
async fn exchange_authorization_code(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    code: &str,
    state: &str,
    verifier: &str,
    redirect_uri: &str,
    signal: &CancellationToken,
) -> Result<OAuthCredentials, AuthError> {
    let body = serde_json::json!({
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "state": state,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    });
    let response_body = match post_token_json(client, body, signal).await {
        Ok(body) => body,
        Err(error) => {
            return Err(auth_error(format!(
                "Token exchange request failed. url={TOKEN_URL}; redirect_uri={redirect_uri}; response_type=authorization_code; details={}",
                error
            )));
        }
    };
    let map = serde_json::from_str::<serde_json::Value>(&response_body).map_err(|error| {
        auth_error(format!(
            "Token exchange returned invalid JSON. url={TOKEN_URL}; body={response_body}; details={error}"
        ))
    })?;
    let map = crate::auth::oauth::json_object(map, "Token exchange returned invalid JSON")?;
    let access_token = crate::auth::oauth::required_string(&map, "access_token")?;
    let refresh_token = crate::auth::oauth::required_string(&map, "refresh_token")?;
    let expires_in = map
        .get("expires_in")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            auth_error("Token exchange returned invalid JSON: missing expires_in")
        })?;
    Ok(oauth_credentials(
        access_token,
        refresh_token,
        clock.now_ms() + wire_seconds_to_i64(expires_in).saturating_mul(1000) - EXPIRY_SKEW_MS,
    ))
}

/// Refresh the Anthropic token, upstream's `refreshAnthropicToken`.
async fn refresh_anthropic_token(
    client: &Arc<dyn HttpClient>,
    clock: &dyn crate::auth::clock::AuthClock,
    refresh_token: &str,
    signal: &CancellationToken,
) -> Result<OAuthCredentials, AuthError> {
    let body = serde_json::json!({
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    });
    let response_body = match post_token_json(client, body, signal).await {
        Ok(body) => body,
        Err(error) => {
            return Err(auth_error(format!(
                "Anthropic token refresh request failed. url={TOKEN_URL}; details={}",
                error
            )));
        }
    };
    let map = serde_json::from_str::<serde_json::Value>(&response_body)
        .map_err(|error| {
            auth_error(format!(
                "Anthropic token refresh returned invalid JSON. url={TOKEN_URL}; body={response_body}; details={error}"
            ))
        })?;
    let map =
        crate::auth::oauth::json_object(map, "Anthropic token refresh returned invalid JSON")?;
    let access = crate::auth::oauth::required_string(&map, "access_token")?;
    let refresh = crate::auth::oauth::required_string(&map, "refresh_token")?;
    let expires_in = map
        .get("expires_in")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            auth_error("Anthropic token refresh returned invalid JSON: missing expires_in")
        })?;
    Ok(oauth_credentials(
        access,
        refresh,
        clock.now_ms() + wire_seconds_to_i64(expires_in).saturating_mul(1000) - EXPIRY_SKEW_MS,
    ))
}

/// The login race, upstream's `loginAnthropic`: the manual prompt and the
/// interaction's cancellation settle the wait with `None` (handing the
/// login to manual entry); the callback settles it with the code.
async fn login_anthropic(
    client: &Arc<dyn HttpClient>,
    clock: &Arc<dyn crate::auth::clock::AuthClock>,
    interaction: &crate::auth::types::ProviderAuthInteraction,
) -> Result<OAuthCredentials, AuthError> {
    let pkce = generate_pkce()?;
    let (server, wait_cell, receiver) = start_callback_server(&pkce.verifier).await?;

    let manual_abort = CancellationToken::new();
    let manual_prompt = AuthPrompt {
        signal: Some(manual_abort.clone()),
        kind: AuthPromptKind::ManualCode {
            message:
                "Complete login in your browser, or paste the authorization code / redirect URL here:"
                    .to_owned(),
            placeholder: Some(REDIRECT_URI.to_owned()),
        },
    };

    (interaction.notify)(AuthEvent::AuthUrl {
        url: authorize_url(&pkce),
        instructions: Some(
            "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."
                .to_owned(),
        ),
    });

    // The manual prompt runs detached, upstream's manualPromise: its
    // completion (input or rejection) cancels the callback wait, and so
    // does the interaction's own cancellation.
    let manual = tokio::spawn({
        let interaction = interaction.clone();
        let wait_cell = wait_cell.clone();
        async move {
            let result = (interaction.prompt)(manual_prompt).await;
            wait_cell.settle(None).await;
            result
        }
    });

    let mut code: Option<String> = None;
    let mut state: Option<String> = None;
    let mut manual_error: Option<AuthError> = None;
    match receiver.await.unwrap_or(None) {
        Some((callback_code, callback_state)) => {
            code = Some(callback_code);
            state = Some(callback_state);
        }
        None => match manual.await {
            Ok(Ok(input)) => {
                let (parsed_code, parsed_state) = parse_authorization_code_state(&input);
                if parsed_state
                    .as_ref()
                    .is_some_and(|parsed| parsed != &pkce.verifier)
                {
                    manual_abort.cancel();
                    server.close();
                    return Err(auth_error("OAuth state mismatch"));
                }
                code = parsed_code;
                state = parsed_state.or_else(|| Some(pkce.verifier.clone()));
            }
            Ok(Err(error)) => manual_error = Some(AuthError::from(error)),
            Err(join_error) => manual_error = Some(auth_error(join_error.to_string())),
        },
    }

    let credential = match (code, state) {
        (Some(code), Some(state)) => {
            (interaction.notify)(AuthEvent::Progress {
                message: "Exchanging authorization code for tokens...".to_owned(),
            });
            exchange_authorization_code(
                &client,
                clock.as_ref(),
                &code,
                &state,
                &pkce.verifier,
                REDIRECT_URI,
                &interaction.signal,
            )
            .await
        }
        (None, _) => {
            Err(manual_error
                .unwrap_or_else(|| auth_error("Missing authorization code")))
        }
        (Some(_), None) => Err(auth_error("Missing OAuth state")),
    };
    manual_abort.cancel();
    server.close();
    credential
}