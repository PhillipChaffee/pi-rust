//! The OpenRouter OAuth PKCE flow, ported from
//! `packages/ai/src/auth/oauth/openrouter.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! OpenRouter exchanges an authorization code for a permanent, user-controlled
//! API key rather than an expiring access/refresh pair. The callback is a
//! one-shot loopback server on an ephemeral port, raced against a manual
//! prompt so headless sessions can paste the redirect URL.
//!
//! Porting restatement this module records: the shared callback server parses
//! only path and query, so the upstream method check (`request.method !==
//! "GET"`) collapses into the path check — the browser redirect it guards is
//! always a `GET`.

use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::auth::oauth::callback::{
    CallbackHandler, CallbackRequest, CallbackResponse, OAuthCallbackServer, WaitCell,
};
use crate::auth::oauth::oauth_page::{oauth_error_html, oauth_success_html};
use crate::auth::oauth::pkce::generate_pkce;
use crate::auth::oauth::{
    auth_error, execute, json_post_request, oauth_credentials, random_uuid_v4,
};
use crate::auth::types::{AuthError, AuthEvent, AuthPrompt, AuthPromptKind, ModelAuth, OAuthCredentials};
use crate::http::HttpClient;
use crate::types::BoxedFuture;
use crate::utils::provider_env::get_provider_env_value;

const AUTHORIZE_URL: &str = "https://openrouter.ai/auth";
const TOKEN_URL: &str = "https://openrouter.ai/api/v1/auth/keys";
const LOGIN_TIMEOUT_MS: u64 = 5 * 60 * 1000;
const TOKEN_EXCHANGE_TIMEOUT_MS: u64 = 30_000;
/// The expiry of a key credential, upstream's `Number.MAX_SAFE_INTEGER`.
const MAX_SAFE_INTEGER_MS: i64 = 9_007_199_254_740_991;
const CANCELLED_MESSAGE: &str = "Login cancelled";
const EXCHANGE_TIMEOUT_MESSAGE: &str = "OpenRouter OAuth token exchange timed out";

/// The OpenRouter OAuth PKCE flow, wired with its [`HttpClient`] seam.
pub struct OpenRouterOAuth {
    client: Arc<dyn HttpClient>,
}

impl std::fmt::Debug for OpenRouterOAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpenRouterOAuth")
    }
}

impl OpenRouterOAuth {
    /// The flow over the given client.
    #[must_use]
    pub fn new(client: Arc<dyn HttpClient>) -> Self {
        Self { client }
    }
}

fn callback_host() -> String {
    get_provider_env_value("PI_OAUTH_CALLBACK_HOST", None).unwrap_or_else(|| "127.0.0.1".to_owned())
}

/// A pasted authorization code, upstream's `parseAuthorizationInput`: a URL's
/// `code` query parameter, then a `code=` form, else the trimmed input.
fn parse_authorization_input(input: &str) -> Option<String> {
    let value = input.trim();
    if value.is_empty() {
        return None;
    }

    if let Ok(url) = url::Url::parse(value) {
        return url
            .query_pairs()
            .find(|(key, _)| key == "code")
            .map(|(_, value)| value.into_owned());
    }

    if value.contains("code=") {
        return url::form_urlencoded::parse(value.as_bytes())
            .find(|(key, _)| key == "code")
            .map(|(_, value)| value.into_owned());
    }

    Some(value.to_owned())
}

/// The error detail an OpenRouter body carries, upstream's `errorDetail`:
/// `error_description`, then `message`, then a string `error`, then an
/// object `error`'s `message`.
fn error_detail(body: &serde_json::Map<String, Value>) -> Option<String> {
    if let Some(description) = body.get("error_description").and_then(Value::as_str) {
        return Some(description.to_owned());
    }
    if let Some(message) = body.get("message").and_then(Value::as_str) {
        return Some(message.to_owned());
    }
    if let Some(message) = body
        .get("error")
        .and_then(Value::as_object)
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
    {
        return Some(message.to_owned());
    }
    body.get("error").and_then(Value::as_str).map(str::to_owned)
}

/// Exchange an authorization code for the API-key credential, upstream's
/// `exchangeAuthorizationCode`.
///
/// # Errors
/// Rejects with `Login cancelled` on cancellation, the timeout message when
/// the 30-second exchange expires, the `key exchange failed (HTTP …)` message
/// with the body's error detail, and the no-`key` message when the response
/// carries no key.
async fn exchange_authorization_code(
    client: &Arc<dyn HttpClient>,
    code: &str,
    verifier: &str,
    signal: &CancellationToken,
) -> Result<OAuthCredentials, AuthError> {
    if signal.is_cancelled() {
        return Err(auth_error(CANCELLED_MESSAGE.to_owned()));
    }
    let body = serde_json::json!({
        "code": code,
        "code_verifier": verifier,
        "code_challenge_method": "S256",
    });
    let request = json_post_request(
        TOKEN_URL,
        &[("accept", "application/json")],
        &body,
        signal.clone(),
        Some(TOKEN_EXCHANGE_TIMEOUT_MS),
    );
    let mut response = match execute(client, request).await {
        Ok(response) => response,
        Err(error) => {
            if signal.is_cancelled() {
                return Err(auth_error(CANCELLED_MESSAGE.to_owned()));
            }
            return Err(auth_error(
                if error.to_string() == crate::http::HttpError::Timeout.to_string() {
                    EXCHANGE_TIMEOUT_MESSAGE.to_owned()
                } else {
                    error.to_string()
                },
            ));
        }
    };
    let status = response.status;

    let parsed = crate::auth::oauth::read_json(&mut response).await;
    let body = match parsed {
        Ok(Value::Object(map)) => map,
        Ok(_) => serde_json::Map::new(),
        Err(_) => {
            if signal.is_cancelled() {
                return Err(auth_error(CANCELLED_MESSAGE.to_owned()));
            }
            if (200..300).contains(&status) {
                return Err(auth_error(
                    "OpenRouter OAuth returned invalid JSON".to_owned(),
                ));
            }
            serde_json::Map::new()
        }
    };

    if !(200..300).contains(&status) {
        let detail = error_detail(&body);
        return Err(auth_error(detail.map_or_else(
            || format!("OpenRouter OAuth key exchange failed (HTTP {status})"),
            |detail| format!("OpenRouter OAuth key exchange failed (HTTP {status}): {detail}"),
        )));
    }

    let Some(key) = body
        .get("key")
        .and_then(Value::as_str)
        .filter(|key| !key.is_empty())
    else {
        return Err(auth_error(
            "OpenRouter OAuth response carries no \"key\"".to_owned(),
        ));
    };
    Ok(oauth_credentials(key, "", MAX_SAFE_INTEGER_MS))
}

/// The shared handler state: whether a callback claimed the exchange and
/// whether the login settled at all, upstream's `claimed`/`settled`.
#[derive(Default)]
struct CallbackState {
    claimed: bool,
    settled: bool,
}

/// Settle the login wait unless a previous settle landed, upstream's
/// `finish`.
async fn finish(
    state: &tokio::sync::Mutex<CallbackState>,
    wait_cell: &WaitCell<Result<OAuthCredentials, AuthError>>,
    result: Option<Result<OAuthCredentials, AuthError>>,
) {
    let mut state = state.lock().await;
    if state.settled {
        return;
    }
    state.settled = true;
    drop(state);
    wait_cell.settle(result).await;
}

/// Mark the login settled without a value — the timeout and cancellation
/// paths, upstream's `finish` before the callback resolves the wait.
async fn mark_settled(state: &tokio::sync::Mutex<CallbackState>) {
    state.lock().await.settled = true;
}

/// The one-shot loopback server and its wait pair, upstream's
/// `startCallbackServer`; the tuple carries the bound server, the shared
/// state, the wait cell, the settle receiver, and the callback URL the
/// authorize URL points at.
async fn start_callback_server(
    client: &Arc<dyn HttpClient>,
    callback_path: &str,
    verifier: &str,
    signal: &CancellationToken,
) -> Result<
    (
        OAuthCallbackServer,
        Arc<tokio::sync::Mutex<CallbackState>>,
        WaitCell<Result<OAuthCredentials, AuthError>>,
        tokio::sync::oneshot::Receiver<Option<Result<OAuthCredentials, AuthError>>>,
        String,
    ),
    AuthError,
> {
    let callback_host = callback_host();
    let (wait_cell, receiver) = WaitCell::<Result<OAuthCredentials, AuthError>>::new();
    let state = Arc::new(tokio::sync::Mutex::new(CallbackState::default()));
    let callback_path = callback_path.to_owned();
    let verifier = verifier.to_owned();
    let handler_signal = signal.clone();
    let handler: CallbackHandler = Arc::new({
        let state = Arc::clone(&state);
        let wait_cell = wait_cell.clone();
        let client = Arc::clone(client);
        let signal = handler_signal;
        let callback_path = callback_path.clone();
        move |request: CallbackRequest| {
            let state = Arc::clone(&state);
            let wait_cell = wait_cell.clone();
            let client = Arc::clone(&client);
            let callback_path = callback_path.clone();
            let verifier = verifier.clone();
            let signal = signal.clone();
            Box::pin(async move {
                if request.path != callback_path {
                    return CallbackResponse {
                        status: 404,
                        html: oauth_error_html("OAuth callback route not found.", None),
                    };
                }

                let mut state = state.lock().await;
                if state.claimed || state.settled {
                    return CallbackResponse {
                        status: 409,
                        html: oauth_error_html("This OAuth callback has already been used.", None),
                    };
                }
                if let Some(oauth_error) = request.query_value("error") {
                    let description = request
                        .query_value("error_description")
                        .unwrap_or(oauth_error)
                        .to_owned();
                    state.settled = true;
                    drop(state);
                    wait_cell
                        .settle(Some(Err(auth_error(format!(
                            "OpenRouter authorization failed: {description}"
                        )))))
                        .await;
                    return CallbackResponse {
                        status: 400,
                        html: oauth_error_html(
                            "OpenRouter authorization was denied.",
                            Some(&description),
                        ),
                    };
                }
                let Some(code) = request.query_value("code").map(str::to_owned) else {
                    return CallbackResponse {
                        status: 400,
                        html: oauth_error_html("OpenRouter returned no authorization code.", None),
                    };
                };
                state.claimed = true;
                drop(state);

                match exchange_authorization_code(&client, &code, &verifier, &signal).await {
                    Ok(credential) => {
                        wait_cell.settle(Some(Ok(credential))).await;
                        CallbackResponse {
                            status: 200,
                            html: oauth_success_html(
                                "Signed in to OpenRouter. You may now close this page.",
                            ),
                        }
                    }
                    Err(error) => {
                        let message = error.to_string();
                        wait_cell.settle(Some(Err(error))).await;
                        CallbackResponse {
                            status: 502,
                            html: oauth_error_html(
                                "OpenRouter key exchange failed.",
                                Some(&message),
                            ),
                        }
                    }
                }
            })
        }
    });
    let server = OAuthCallbackServer::bind(&callback_host, 0, handler).await?;
    let port = server.local_addr()?.port();
    let callback_url = format!("http://{callback_host}:{port}{callback_path}");
    Ok((server, state, wait_cell, receiver, callback_url))
}

/// The login flow, upstream's `loginOpenRouter`: the callback server and the
/// manual prompt race, bounded by the five-minute login timeout.
///
/// # Errors
/// Rejects with `Login cancelled`, the login timeout message, the manual
/// prompt's rejection, `Missing authorization code`, and the exchange's own
/// errors.
async fn login_openrouter(
    client: &Arc<dyn HttpClient>,
    interaction: &crate::auth::types::ProviderAuthInteraction,
) -> Result<OAuthCredentials, AuthError> {
    if interaction.signal.is_cancelled() {
        return Err(auth_error(CANCELLED_MESSAGE.to_owned()));
    }
    let pkce = generate_pkce()?;
    let callback_path = format!("/oauth/callback/{}", random_uuid_v4()?);
    let (server, state, wait_cell, receiver, callback_url) =
        start_callback_server(client, &callback_path, &pkce.verifier, &interaction.signal).await?;

    let manual_abort = CancellationToken::new();
    let manual_prompt = AuthPrompt {
        signal: Some(manual_abort.clone()),
        kind: AuthPromptKind::ManualCode {
            message:
                "Complete sign-in in your browser, or paste the authorization code / redirect URL here:"
                    .to_owned(),
            placeholder: Some(callback_url.clone()),
        },
    };

    (interaction.notify)(AuthEvent::Progress {
        message: format!("Listening for OpenRouter OAuth callback on {callback_url}"),
    });
    let authorize_query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("callback_url", &callback_url)
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("code_challenge_method", "S256")
        .finish();
    (interaction.notify)(AuthEvent::AuthUrl {
        url: format!("{AUTHORIZE_URL}?{authorize_query}"),
        instructions: Some(
            "Complete sign-in in your browser. If the browser is on another machine, paste the final redirect URL here."
                .to_owned(),
        ),
    });

    // The manual prompt runs detached, upstream's manualPromise: its
    // completion (input or rejection) hands the login over to manual entry
    // unless a callback already claimed the exchange.
    let manual = tokio::spawn({
        let interaction = interaction.clone();
        let state = Arc::clone(&state);
        let wait_cell = wait_cell.clone();
        async move {
            let result = (interaction.prompt)(manual_prompt).await;
            if !state.lock().await.claimed {
                finish(&state, &wait_cell, None).await;
            }
            result
        }
    });

    let result: Result<OAuthCredentials, AuthError> = tokio::select! {
        () = interaction.signal.cancelled() => {
            mark_settled(&state).await;
            Err(auth_error(CANCELLED_MESSAGE.to_owned()))
        }
        () = tokio::time::sleep(Duration::from_millis(LOGIN_TIMEOUT_MS)) => {
            mark_settled(&state).await;
            Err(auth_error("OpenRouter OAuth login timed out".to_owned()))
        }
        callback_outcome = receiver => match callback_outcome.unwrap_or(None) {
            Some(outcome) => outcome,
            None => match manual.await {
                Ok(Ok(input)) => match parse_authorization_input(&input) {
                    Some(code) => {
                        (interaction.notify)(AuthEvent::Progress {
                            message: "Exchanging authorization code for an API key...".to_owned(),
                        });
                        exchange_authorization_code(client, &code, &pkce.verifier, &interaction.signal).await
                    }
                    None => Err(auth_error("Missing authorization code".to_owned())),
                },
                Ok(Err(error)) => Err(AuthError::from(error)),
                Err(join_error) => Err(auth_error(join_error.to_string())),
            },
        },
    };
    manual_abort.cancel();
    server.close();
    result
}

impl OpenRouterOAuth {
    /// Run the interactive login flow.
    pub fn login(
        &self,
        interaction: crate::auth::types::ProviderAuthInteraction,
    ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move { login_openrouter(&client, &interaction).await })
    }

    /// Refresh is a no-op: the credential is a permanent, user-controlled API
    /// key.
    pub fn refresh(
        &self,
        credential: OAuthCredentials,
        _signal: CancellationToken,
    ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>> {
        Box::pin(async move { Ok(credential) })
    }

    /// Derive the request auth from the permanent key.
    #[must_use]
    pub fn to_auth(&self, credential: &OAuthCredentials) -> ModelAuth {
        ModelAuth {
            api_key: Some(credential.access.clone()),
            ..ModelAuth::default()
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
        let refresh: crate::auth::types::OAuthRefreshFn =
            Arc::new(|credential, _signal| Box::pin(async move { Ok(credential) }));
        let to_auth: crate::auth::types::OAuthToAuthFn = Arc::new(|credential| {
            let auth = ModelAuth {
                api_key: Some(credential.access.clone()),
                ..ModelAuth::default()
            };
            Box::pin(async move { Ok(auth) })
        });
        crate::auth::types::OAuthAuth {
            name: "OpenRouter OAuth".to_owned(),
            is_subscription: None,
            login_label: Some("Sign in with OpenRouter".to_owned()),
            login,
            refresh,
            to_auth,
        }
    }
}
