//! Rust-native coverage of the browser-login flows the ported vitest suites
//! do not reach: the loopback callback side of the Anthropic login and the
//! manual-input-only paths of OpenRouter, OpenAI Codex, and Kimi Code.
//! Upstream drives these paths inline in its per-flow vitest suites
//! (`test/anthropic-oauth.test.ts`, `test/openrouter-oauth.test.ts`,
//! `test/openai-codex-oauth.test.ts`, `test/kimi-coding-oauth.test.ts`) at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Adaptations:
//!
//! - The anthropic manual-input-only path (a pasted URL with no callback
//!   ever arriving) cannot be driven hermetically — the flow binds a fixed
//!   port, and the paste only resolves when the callback server is
//!   unreachable — so it is skipped; the occupied-port branch surfaces the
//!   bind error instead, and the ephemeral-port manual path is covered for
//!   OpenRouter.
//! - The `Date.now()` reads go through the injected `AuthClock`; the flows'
//!   sleeps ride the paused tokio clock the fake-timer suites drive.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected shape panics by design"
)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::auth_fixtures::{RecordingInteraction, oauth_credentials};
use pi_ai::auth::clock::SteppedClock;
use pi_ai::auth::oauth::anthropic::AnthropicOAuth;
use pi_ai::auth::oauth::kimi_coding::KimiCodingOAuth;
use pi_ai::auth::oauth::openai_codex::OpenAICodexOAuth;
use pi_ai::auth::oauth::openrouter::OpenRouterOAuth;
use pi_ai::auth::types::{AuthError, AuthEvent, OAuthCredentials};
use pi_ai::http::{MockHttpClient, MockResponse, json_response};
use pi_ai::types::BoxedFuture;
use pi_ai::utils::abort::AbortError;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// The epoch the stepped clocks fix, so exchange arithmetic asserts exactly.
const EPOCH_MS: i64 = 1_758_000_000_000;

const ANTHROPIC_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const ANTHROPIC_CALLBACK_PORT: u16 = 53_692;
const OPENROUTER_TOKEN_URL: &str = "https://openrouter.ai/api/v1/auth/keys";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_USERCODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const CODEX_DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const KIMI_DEVICE_URL: &str = "https://auth.kimi.com/api/oauth/device_authorization";
const KIMI_TOKEN_URL: &str = "https://auth.kimi.com/api/oauth/token";

/// The host the flows bind their callback servers on, mirroring the flows'
/// `PI_OAUTH_CALLBACK_HOST` read.
fn callback_host() -> String {
    std::env::var("PI_OAUTH_CALLBACK_HOST").unwrap_or_else(|_| String::from("127.0.0.1"))
}

/// A clock fixed at [`EPOCH_MS`], the `Date.now()` the flows read.
fn stepped_clock() -> Arc<dyn pi_ai::auth::clock::AuthClock> {
    Arc::new(SteppedClock::new(EPOCH_MS))
}

/// A fresh uncancelled interaction bundle over `recording`.
fn provider_interaction(
    recording: &RecordingInteraction,
) -> pi_ai::auth::types::ProviderAuthInteraction {
    pi_ai::auth::types::ProviderAuthInteraction::from_interaction(
        recording.interaction(),
        CancellationToken::new(),
    )
}

/// Wait for a condition the login task reaches without clock dependence,
/// stepping the scheduler between probes like `advanceTimersByTimeAsync`
/// does, and return the probed value.
async fn wait_until<T>(condition: impl Fn() -> Option<T>, what: &str) -> T {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(value) = condition() {
            return value;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for {what}"
        );
        tokio::task::yield_now().await;
    }
}

/// Step the paused clock until `condition` holds, `advanceTimersByTimeAsync`'s
/// single-knob behaviour.
async fn advance_until(condition: impl Fn() -> bool, what: &str) {
    for _ in 0..2_000 {
        if condition() {
            return;
        }
        tokio::time::advance(Duration::from_millis(5)).await;
        tokio::task::yield_now().await;
    }
    panic!("timed out waiting for {what}");
}

/// Advance the paused clock in one-second steps until `condition` holds or
/// `budget_seconds` elapse — the long deadlines (the 30-second exchange
/// timeout, the five-minute login timeout) need coarse steps.
async fn advance_seconds_until(condition: impl Fn() -> bool, budget_seconds: u64, what: &str) {
    for _ in 0..budget_seconds {
        if condition() {
            return;
        }
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
    }
    if condition() {
        return;
    }
    panic!("timed out waiting for {what}");
}

/// Wait until `port` accepts a fresh listener — a finished login's callback
/// server closes asynchronously, and the next login's bind races the
/// teardown.
async fn wait_port_free(port: u16) {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if let Ok(listener) = TcpListener::bind((callback_host().as_str(), port)).await {
            drop(listener);
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the callback port {port} never freed"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A three-segment JWT whose auth claim carries the `ChatGPT` account id,
/// standard base64url without padding.
fn codex_jwt(account_id: &str) -> String {
    use base64::Engine as _;
    let encode = |bytes: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    let header = encode(br#"{"alg":"HS256","typ":"JWT"}"#);
    let payload =
        format!(r#"{{"https://api.openai.com/auth":{{"chatgpt_account_id":"{account_id}"}}}}"#);
    format!("{header}.{}.sig", encode(payload.as_bytes()))
}

/// The recorded request body of the first request matching `url`, from the
/// seam record.
fn recorded_body(mock: &MockHttpClient, url: &str) -> Vec<u8> {
    mock.recorded()
        .into_iter()
        .find(|request| request.url == url)
        .and_then(|request| request.body.map(|body| body.to_vec()))
        .unwrap_or_else(|| panic!("no request reached {url}"))
}

/// The form-encoded body of a request as pairs, the `URLSearchParams` shape
/// upstream asserts on.
fn form_pairs(mock: &MockHttpClient, url: &str) -> Vec<(String, String)> {
    url::form_urlencoded::parse(&recorded_body(mock, url))
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect()
}

fn json_body(mock: &MockHttpClient, url: &str) -> serde_json::Value {
    serde_json::from_slice(&recorded_body(mock, url)).expect("the recorded body is json")
}

/// The flow's device-code event, asserting one was reported.
fn device_code_event(recording: &RecordingInteraction) -> AuthEvent {
    recording
        .events()
        .into_iter()
        .find(|event| matches!(event, AuthEvent::DeviceCode { .. }))
        .expect("the flow reported the device code")
}

/// A query parameter of an event's URL, the authorize URL's carried params.
fn url_query_param(url: &str, name: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()?
        .query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

// ---------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------

/// The two port-53692 cases serialize: the login's callback server and the
/// dummy listener both claim the fixed port.
static ANTHROPIC_PORT: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The port-1455 cases serialize the same way for the Codex callback server.
static CODEX_PORT: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// A raw one-request-per-connection GET to `port`, the loopback callback the
/// browser sends; returns the raw response bytes as text.
async fn send_request(port: u16, request: impl AsRef<[u8]>) -> String {
    let mut stream = tokio::net::TcpStream::connect((callback_host().as_str(), port))
        .await
        .expect("the callback server accepts");
    stream
        .write_all(request.as_ref())
        .await
        .expect("the callback writes");
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .expect("the response reads to EOF");
    String::from_utf8(raw).expect("the response is utf-8")
}

#[tokio::test]
async fn anthropic_login_completes_through_a_real_callback_and_cancels_the_manual_prompt() {
    let _port = ANTHROPIC_PORT.lock().await;
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == ANTHROPIC_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            }),
        ));
    let recording = RecordingInteraction::new();
    let flow = AnthropicOAuth::new(Arc::new(mock.clone()), stepped_clock());
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };

    // The authorize URL the user's browser opens carries the verifier as
    // its state; the callback server is already bound on the fixed port.
    let auth_url = wait_until(|| recording.auth_url(), "the authorize URL").await;
    let state = url_query_param(&auth_url, "state").expect("the authorize URL carries the state");

    let response = send_callback(
        ANTHROPIC_CALLBACK_PORT,
        &format!("GET /callback?code=C&state={state} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "the browser gets the success page: {response:?}"
    );

    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    // now + expires_in - the five-minute expiry skew, on the stepped clock.
    assert_eq!(credential.expires, EPOCH_MS + 3_600_000 - 5 * 60_000);
    let exchange = json_body(&mock, ANTHROPIC_TOKEN_URL);
    assert_eq!(exchange["grant_type"], "authorization_code");
    assert_eq!(exchange["code"], "C");
    assert_eq!(exchange["state"], state);
    assert_eq!(exchange["redirect_uri"], "http://localhost:53692/callback");

    // The manual prompt was cancelled once the callback won the race, so
    // terminal UIs can dismiss it.
    wait_until(
        || recording.prompts().into_iter().next(),
        "the manual prompt",
    )
    .await;
    let manual_prompt = &recording.prompts()[0];
    assert!(
        manual_prompt
            .signal
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled),
        "the manual prompt's signal aborted: {manual_prompt:?}"
    );
}

/// The pasted-redirect one-shot the manual race uses, over a real loopback
/// connection to the flow's own callback server.
async fn send_callback(port: u16, request: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect((callback_host().as_str(), port))
        .await
        .expect("the callback server accepts");
    stream
        .write_all(request.as_bytes())
        .await
        .expect("the callback writes");
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .expect("the response reads to EOF");
    String::from_utf8(raw).expect("the response is utf-8")
}

#[tokio::test]
async fn an_occupied_callback_port_surfaces_the_bind_error() {
    let _port = ANTHROPIC_PORT.lock().await;
    let dummy = TcpListener::bind((callback_host().as_str(), ANTHROPIC_CALLBACK_PORT))
        .await
        .expect("the dummy listener occupies the callback port");

    let recording = RecordingInteraction::new();
    let outcome = AnthropicOAuth::new(Arc::new(MockHttpClient::new()), stepped_clock())
        .login(provider_interaction(&recording))
        .await;
    let error = outcome.expect_err("the occupied port fails login");
    assert!(
        error
            .to_string()
            .starts_with("could not bind the OAuth callback server"),
        "the bind error surfaces before login hands a URL to the user: {error:?}"
    );
    assert!(
        recording.prompts().is_empty(),
        "the flow fails before prompting"
    );

    drop(dummy);
}

// ---------------------------------------------------------------------------
// OpenRouter
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openrouter_manual_input_mints_the_key_without_a_callback() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "key": "sk-or-test" }),
        ));
    let recording = RecordingInteraction::new();
    let for_answer = Arc::new(recording.clone());
    recording.set_dynamic(
        move |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            // The paste is the captured callback URL plus the authorization
            // code, the headless path upstream's manual prompt takes.
            let callback_url = for_answer
                .events()
                .into_iter()
                .find_map(|event| match event {
                    AuthEvent::AuthUrl { url, .. } => Some(url),
                    _ => None,
                })
                .and_then(|url| url_query_param(&url, "callback_url"))
                .expect("the authorize URL carried the callback_url");
            Box::pin(std::future::ready(Ok(format!(
                "{callback_url}?code=authorization-code"
            ))))
        },
    );

    let credential = OpenRouterOAuth::new(Arc::new(mock.clone()))
        .login(provider_interaction(&recording))
        .await
        .expect("the pasted redirect mints the key");
    assert_eq!(credential.access, "sk-or-test");
    assert_eq!(credential.refresh, "", "the key credential is permanent");
    assert_eq!(
        credential.expires, 9_007_199_254_740_991,
        "upstream's MAX_SAFE_INTEGER"
    );
    let exchange = json_body(&mock, OPENROUTER_TOKEN_URL);
    assert_eq!(exchange["code"], "authorization-code");
    assert_eq!(exchange["code_challenge_method"], "S256");
}

#[tokio::test]
async fn a_cancelled_openrouter_manual_prompt_fails_login_without_exchanging() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "key": "sk-or-unexpected" }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Err(AbortError)]);

    let error = OpenRouterOAuth::new(Arc::new(mock.clone()))
        .login(provider_interaction(&recording))
        .await
        .expect_err("the cancelled prompt fails login");
    assert_eq!(error.to_string(), AbortError::MESSAGE);
    assert_eq!(mock.request_count(), 0, "no key exchange ran");
}

#[tokio::test]
async fn an_empty_openrouter_manual_input_rejects_without_exchanging() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "key": "sk-or-unexpected" }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("   "))]);

    let error = OpenRouterOAuth::new(Arc::new(mock.clone()))
        .login(provider_interaction(&recording))
        .await
        .expect_err("empty input fails login");
    assert_eq!(error.to_string(), "Missing authorization code");
    assert_eq!(mock.request_count(), 0, "no key exchange ran");
}

// ---------------------------------------------------------------------------
// OpenAI Codex
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_codex_browser_login_mints_the_account_from_manual_input() {
    let _port = CODEX_PORT.lock().await;
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": codex_jwt("acc"),
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    let for_answer = Arc::new(recording.clone());
    recording.set_dynamic(
        move |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            let state = for_answer
                .auth_url()
                .as_deref()
                .and_then(|url| url_query_param(url, "state"))
                .expect("the auth URL carries the state");
            Box::pin(std::future::ready(Ok(format!(
                "http://localhost:1455/auth/callback?code=oauth-code&state={state}"
            ))))
        },
    );

    let credential = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect("the manual paste mints the credential");
    assert_eq!(credential.access, codex_jwt("acc"));
    assert_eq!(
        credential
            .extra
            .get("accountId")
            .and_then(serde_json::Value::as_str),
        Some("acc")
    );
    assert_eq!(credential.refresh, "refresh-token");
    assert_eq!(credential.expires, EPOCH_MS + 3_600_000);

    let exchange_pairs = form_pairs(&mock, CODEX_TOKEN_URL);
    assert!(exchange_pairs.contains(&(
        String::from("grant_type"),
        String::from("authorization_code")
    )));
    assert!(exchange_pairs.contains(&(String::from("code"), String::from("oauth-code"))));
    assert!(exchange_pairs.contains(&(
        String::from("redirect_uri"),
        String::from("http://localhost:1455/auth/callback")
    )));
    assert!(
        exchange_pairs
            .iter()
            .any(|(key, value)| key == "code_verifier" && !value.is_empty()),
        "the browser path's own verifier exchanged the code"
    );
}

#[tokio::test(start_paused = true)]
async fn the_openai_codex_device_flow_logs_in_through_pending_polls() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_USERCODE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_auth_id": "device-auth-id",
                "user_code": "ABCD-1234",
                "interval": "5",
            }),
        ));
    mock.on(|request| request.url == CODEX_DEVICE_TOKEN_URL)
        .respond_sequence(vec![
            json_response(
                403,
                &serde_json::json!({ "error": { "code": "deviceauth_authorization_pending" } }),
            ),
            json_response(
                200,
                &serde_json::json!({
                    "authorization_code": "oauth-code",
                    "code_verifier": "device-code-verifier",
                }),
            ),
        ]);
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": codex_jwt("acc"),
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("device_code"))]);
    let flow = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock());
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };

    // The poller's first strike is immediate; the pending reply schedules
    // the server's five-second interval.
    advance_until(|| mock.request_count() >= 2, "the first device poll").await;
    tokio::time::advance(Duration::from_secs(5)).await;

    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, codex_jwt("acc"));
    assert_eq!(
        credential
            .extra
            .get("accountId")
            .and_then(serde_json::Value::as_str),
        Some("acc")
    );

    let AuthEvent::DeviceCode {
        user_code,
        verification_uri,
        interval_seconds,
        expires_in_seconds,
    } = device_code_event(&recording)
    else {
        panic!("the device code event carries the server's fields");
    };
    assert_eq!(user_code, "ABCD-1234");
    assert_eq!(verification_uri, "https://auth.openai.com/codex/device");
    assert_eq!(interval_seconds, Some(5));
    assert_eq!(expires_in_seconds, Some(900));

    let exchange_pairs = form_pairs(&mock, CODEX_TOKEN_URL);
    assert!(exchange_pairs.contains(&(
        String::from("code_verifier"),
        String::from("device-code-verifier")
    )));
    assert!(exchange_pairs.contains(&(
        String::from("redirect_uri"),
        String::from("https://auth.openai.com/deviceauth/callback")
    )));
    assert!(exchange_pairs.contains(&(String::from("code"), String::from("oauth-code"))));
}

#[tokio::test]
async fn a_disabled_openai_codex_device_auth_reports_the_not_enabled_error() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_USERCODE_URL)
        .respond(MockResponse::status(404));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("device_code"))]);

    let error = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the disabled server fails login");
    assert_eq!(
        error.to_string(),
        "OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL."
    );
}

// ---------------------------------------------------------------------------
// Kimi Code
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn the_kimi_device_flow_logs_in_and_notifies_the_complete_uri() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_DEVICE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_code": "device-code-123",
                "user_code": "ABCD-1234",
                "verification_uri": "https://www.kimi.com/code",
                "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
                "interval": 5,
                "expires_in": 600,
            }),
        ));
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond_sequence(vec![
            json_response(
                400,
                &serde_json::json!({ "error": "authorization_pending" }),
            ),
            json_response(
                200,
                &serde_json::json!({
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_in": 3600,
                }),
            ),
        ]);
    let recording = RecordingInteraction::new();
    let flow = KimiCodingOAuth::new(Arc::new(mock.clone()));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };

    // waitBeforeFirstPoll: the first poll lands after the server's interval.
    tokio::time::advance(Duration::from_secs(5)).await;
    advance_until(|| mock.request_count() >= 2, "the first token poll").await;
    tokio::time::advance(Duration::from_secs(5)).await;

    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    let AuthEvent::DeviceCode {
        user_code,
        verification_uri,
        interval_seconds,
        expires_in_seconds,
    } = device_code_event(&recording)
    else {
        panic!("the device code event carries the server's fields");
    };
    assert_eq!(user_code, "ABCD-1234");
    assert_eq!(
        verification_uri, "https://www.kimi.com/code?user_code=ABCD-1234",
        "the notification uses verification_uri_complete"
    );
    assert_eq!(interval_seconds, Some(5));
    assert_eq!(expires_in_seconds, Some(600));
}

#[tokio::test]
async fn an_unauthorized_kimi_refresh_reports_the_unauthorized_branch() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(json_response(401, &serde_json::json!({})));

    let error = KimiCodingOAuth::new(Arc::new(mock.clone()))
        .refresh(
            oauth_credentials("old-access", "old-refresh", 0),
            CancellationToken::new(),
        )
        .await
        .expect_err("the dead credential fails refresh");
    assert_eq!(
        error.to_string(),
        "Kimi Code token refresh unauthorized (status 401)"
    );
}

#[tokio::test(start_paused = true)]
async fn the_kimi_refresh_retries_backoffs_then_succeeds() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond_sequence(vec![
            json_response(500, &serde_json::json!({})),
            json_response(500, &serde_json::json!({})),
            json_response(
                200,
                &serde_json::json!({
                    "access_token": "a",
                    "refresh_token": "r",
                    "expires_in": 60,
                }),
            ),
        ]);
    let flow = KimiCodingOAuth::new(Arc::new(mock.clone()));
    let handle = tokio::spawn(async move {
        flow.refresh(
            oauth_credentials("old-access", "old-refresh", 0),
            CancellationToken::new(),
        )
        .await
    });

    advance_until(|| mock.request_count() >= 1, "the first refresh attempt").await;
    tokio::time::advance(Duration::from_secs(1)).await;
    advance_until(|| mock.request_count() >= 2, "the second refresh attempt").await;
    tokio::time::advance(Duration::from_secs(2)).await;

    let credential = handle
        .await
        .expect("the refresh task joins")
        .expect("refresh resolves");
    assert_eq!(credential.access, "a");
    assert_eq!(mock.request_count(), 3, "two 500s, then the success");
}

// ---------------------------------------------------------------------------
// OpenRouter: the loopback callback server paths
// ---------------------------------------------------------------------------

/// The loopback callback URL the login printed, from the progress event.
async fn wait_for_openrouter_callback_url(recording: &RecordingInteraction) -> String {
    wait_until(
        || {
            recording
                .events()
                .into_iter()
                .find_map(|event| match event {
                    AuthEvent::Progress { message }
                        if message.starts_with("Listening for OpenRouter OAuth callback on ") =>
                    {
                        Some(
                            message["Listening for OpenRouter OAuth callback on ".len()..]
                                .to_owned(),
                        )
                    }
                    _ => None,
                })
        },
        "the OpenRouter callback URL",
    )
    .await
}

#[tokio::test]
async fn the_openrouter_callback_route_rejects_unknown_paths() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(json_response(200, &serde_json::json!({ "key": "sk-or" })));
    let recording = RecordingInteraction::new();
    let flow = OpenRouterOAuth::new(Arc::new(mock.clone()));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    let callback_url = wait_for_openrouter_callback_url(&recording).await;
    let response = send_request(
        url::Url::parse(&callback_url)
            .expect("the callback url parses")
            .port()
            .expect("the callback url carries the port"),
        "GET /nope HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 404 Not Found")
            && response.contains("OAuth callback route not found."),
        "the unknown route rejects: {response:?}"
    );
    handle.abort();
}

#[tokio::test]
async fn the_openrouter_callback_error_page_fails_the_login_with_the_description() {
    let mock = MockHttpClient::new();
    let recording = RecordingInteraction::new();
    let flow = OpenRouterOAuth::new(Arc::new(mock.clone()));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    let callback_url = wait_for_openrouter_callback_url(&recording).await;
    let port = url::Url::parse(&callback_url)
        .expect("the callback url parses")
        .port()
        .expect("the callback url carries the port");
    let path = url::Url::parse(&callback_url)
        .expect("the callback url parses")
        .path()
        .to_owned();

    let response = send_request(
        port,
        &format!(
            "GET {path}?error=access_denied&error_description=Nope HTTP/1.1\r\nHost: x\r\n\r\n"
        ),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 400 Bad Request")
            && response.contains("OpenRouter authorization was denied.")
            && response.contains("Nope"),
        "the denial page carries the description: {response:?}"
    );
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the denial fails login");
    assert_eq!(error.to_string(), "OpenRouter authorization failed: Nope");
}

#[tokio::test]
async fn the_openrouter_callback_without_a_code_rejects_the_page_and_waits() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(json_response(200, &serde_json::json!({ "key": "sk-or" })));
    let recording = RecordingInteraction::new();
    let flow = OpenRouterOAuth::new(Arc::new(mock.clone()));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    let callback_url = wait_for_openrouter_callback_url(&recording).await;
    let parsed = url::Url::parse(&callback_url).expect("the callback url parses");
    let port = parsed.port().expect("the callback url carries the port");
    let path = parsed.path().to_owned();

    let response = send_request(
        port,
        &format!("GET {path}?state=s HTTP/1.1\r\nHost: x\r\n\r\n"),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 400 Bad Request")
            && response.contains("OpenRouter returned no authorization code."),
        "the missing-code page rejects: {response:?}"
    );
    // The page never settles the wait, so the login can still be cancelled.
    handle.abort();
}

#[tokio::test]
async fn a_second_openrouter_callback_reports_the_already_used_page() {
    let mock = MockHttpClient::new();
    let (exchange_started, exchange_started_rx) = tokio::sync::oneshot::channel::<()>();
    let exchange_started = Arc::new(tokio::sync::Mutex::new(Some(exchange_started)));
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond_fn(move |_request| {
            let started = Arc::clone(&exchange_started);
            async move {
                let sender = started.lock().await.take();
                if let Some(sender) = sender {
                    let _ = sender.send(());
                }
                // Hold the exchange long enough for the second callback.
                tokio::time::sleep(Duration::from_millis(500)).await;
                Ok(json_response(200, &serde_json::json!({ "key": "sk-or" })))
            }
        });
    let recording = RecordingInteraction::new();
    let flow = OpenRouterOAuth::new(Arc::new(mock.clone()));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    let callback_url = wait_for_openrouter_callback_url(&recording).await;
    let parsed = url::Url::parse(&callback_url).expect("the callback url parses");
    let port = parsed.port().expect("the callback url carries the port");
    let path = parsed.path().to_owned();

    let first = tokio::spawn(send_request(
        port,
        format!("GET {path}?code=C HTTP/1.1\r\nHost: x\r\n\r\n"),
    ));
    exchange_started_rx
        .await
        .expect("the first exchange started");

    let second = send_request(
        port,
        &format!("GET {path}?code=C2 HTTP/1.1\r\nHost: x\r\n\r\n"),
    )
    .await;
    assert!(
        second.starts_with("HTTP/1.1 409 Conflict")
            && second.contains("This OAuth callback has already been used."),
        "the second callback reports the reuse: {second:?}"
    );

    let first_response = first.await.expect("the first request joins");
    assert!(
        first_response.starts_with("HTTP/1.1 200 OK"),
        "the first callback succeeds: {first_response:?}"
    );
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, "sk-or");
}

#[tokio::test]
async fn a_failed_openrouter_exchange_answers_the_bad_gateway_page() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(json_response(
            400,
            &serde_json::json!({"error": "invalid_grant", "error_description": "bad"}),
        ));
    let recording = RecordingInteraction::new();
    let flow = OpenRouterOAuth::new(Arc::new(mock.clone()));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    let callback_url = wait_for_openrouter_callback_url(&recording).await;
    let parsed = url::Url::parse(&callback_url).expect("the callback url parses");
    let port = parsed.port().expect("the callback url carries the port");
    let path = parsed.path().to_owned();

    let response = send_request(
        port,
        &format!("GET {path}?code=C HTTP/1.1\r\nHost: x\r\n\r\n"),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 502 Bad Gateway")
            && response.contains("OpenRouter key exchange failed.")
            && response.contains("bad"),
        "the exchange failure page carries the detail: {response:?}"
    );
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the failed exchange fails login");
    assert_eq!(
        error.to_string(),
        "OpenRouter OAuth key exchange failed (HTTP 400): bad",
        "the error_description detail wins the precedence"
    );
}

#[tokio::test(start_paused = true)]
async fn an_openrouter_exchange_that_outlives_thirty_seconds_times_out() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond_fn(|_request| async move {
            // Park past the 30-second exchange timeout on the paused clock.
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(json_response(200, &serde_json::json!({ "key": "sk-or" })))
        });
    let recording = RecordingInteraction::with_answers(vec![]);
    let for_answer = Arc::new(recording.clone());
    recording.set_dynamic(
        move |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            let callback_url = for_answer
                .events()
                .into_iter()
                .find_map(|event| match event {
                    AuthEvent::Progress { message }
                        if message.starts_with("Listening for OpenRouter OAuth callback on ") =>
                    {
                        Some(
                            message["Listening for OpenRouter OAuth callback on ".len()..]
                                .to_owned(),
                        )
                    }
                    _ => None,
                })
                .expect("the callback url was announced");
            Box::pin(std::future::ready(Ok(format!("{callback_url}?code=C"))))
        },
    );
    let flow = OpenRouterOAuth::new(Arc::new(mock.clone()));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };

    advance_seconds_until(|| handle.is_finished(), 40, "the exchange timeout").await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the stuck exchange times out");
    assert_eq!(
        error.to_string(),
        "OpenRouter OAuth token exchange timed out"
    );
}

#[tokio::test(start_paused = true)]
async fn the_openrouter_login_times_out_after_five_minutes() {
    let recording = RecordingInteraction::new();
    let flow = OpenRouterOAuth::new(Arc::new(MockHttpClient::new()));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    advance_seconds_until(|| handle.is_finished(), 320, "the login timeout").await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the abandoned login times out");
    assert_eq!(error.to_string(), "OpenRouter OAuth login timed out");
}

#[tokio::test]
async fn an_already_cancelled_openrouter_login_rejects_before_binding() {
    let recording = RecordingInteraction::new();
    let signal = CancellationToken::new();
    signal.cancel();
    let interaction = pi_ai::auth::types::ProviderAuthInteraction::from_interaction(
        recording.interaction(),
        signal.clone(),
    );
    let error = OpenRouterOAuth::new(Arc::new(MockHttpClient::new()))
        .login(interaction)
        .await
        .expect_err("the cancelled login rejects");
    assert_eq!(error.to_string(), "Login cancelled");
    assert!(
        recording.events().is_empty(),
        "the flow fails before announcing anything"
    );
}

/// A one-paste login over `mock`, resolving with the paste answer.
async fn openrouter_login_with_paste(
    mock: MockHttpClient,
    paste: String,
) -> Result<OAuthCredentials, AuthError> {
    let recording = RecordingInteraction::new();
    recording.set_dynamic(
        move |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            Box::pin(std::future::ready(Ok(paste.clone())))
        },
    );
    OpenRouterOAuth::new(Arc::new(mock))
        .login(provider_interaction(&recording))
        .await
}

#[tokio::test(start_paused = true)]
async fn openrouter_exchange_failure_details_follow_the_wire_precedence() {
    for (body, expected) in [
        (
            serde_json::json!({"error_description": "desc", "message": "msg", "error": {"message": "obj"}}),
            "OpenRouter OAuth key exchange failed (HTTP 400): desc",
        ),
        (
            serde_json::json!({"message": "msg", "error": {"message": "obj"}}),
            "OpenRouter OAuth key exchange failed (HTTP 400): msg",
        ),
        (
            serde_json::json!({"error": {"message": "obj"}}),
            "OpenRouter OAuth key exchange failed (HTTP 400): obj",
        ),
        (
            serde_json::json!({"error": "str"}),
            "OpenRouter OAuth key exchange failed (HTTP 400): str",
        ),
        (
            serde_json::json!({}),
            "OpenRouter OAuth key exchange failed (HTTP 400)",
        ),
    ] {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
            .respond(json_response(400, &body));
        let error = openrouter_login_with_paste(mock, String::from("code"))
            .await
            .expect_err("the failed exchange rejects");
        assert_eq!(error.to_string(), expected, "{body:?}");
    }
}

#[tokio::test]
async fn openrouter_rejects_invalid_json_and_missing_keys_on_success_statuses() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(MockResponse::status(200).with_body("not json"));
    let error = openrouter_login_with_paste(mock, String::from("code"))
        .await
        .expect_err("the invalid JSON fails");
    assert_eq!(error.to_string(), "OpenRouter OAuth returned invalid JSON");

    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(json_response(200, &serde_json::json!({ "data": {} })));
    let error = openrouter_login_with_paste(mock, String::from("code"))
        .await
        .expect_err("the keyless response fails");
    assert_eq!(
        error.to_string(),
        "OpenRouter OAuth response carries no \"key\""
    );
}

#[test]
fn openrouter_debug_names_the_flow() {
    let flow = OpenRouterOAuth::new(Arc::new(MockHttpClient::new()));
    assert_eq!(format!("{flow:?}"), "OpenRouterOAuth");
    assert_eq!(flow.auth().name, "OpenRouter OAuth");
}

// ---------------------------------------------------------------------------
// OpenAI Codex: the real callback, refresh, and device-failure branches
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_codex_browser_login_completes_through_the_real_callback_port() {
    let _port = CODEX_PORT.lock().await;
    wait_port_free(1455).await;
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": codex_jwt("acc"),
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    let flow = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock());
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };

    let auth_url = wait_until(|| recording.auth_url(), "the authorize URL").await;
    let state = url_query_param(&auth_url, "state").expect("the authorize URL carries the state");

    let response = send_request(
        1455,
        &format!("GET /auth/callback?code=C&state={state} HTTP/1.1\r\nHost: x\r\n\r\n"),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK")
            && response.contains("OpenAI authentication completed."),
        "the callback page succeeds: {response:?}"
    );

    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, codex_jwt("acc"));
    assert_eq!(
        credential
            .extra
            .get("accountId")
            .and_then(serde_json::Value::as_str),
        Some("acc")
    );

    let exchange_pairs = form_pairs(&mock, CODEX_TOKEN_URL);
    assert!(exchange_pairs.contains(&(String::from("code"), String::from("C"))));
    assert_eq!(
        url_query_param(&auth_url, "redirect_uri").as_deref(),
        Some("http://localhost:1455/auth/callback")
    );
}

#[tokio::test]
async fn the_openai_codex_manual_paste_state_mismatch_rejects() {
    let _port = CODEX_PORT.lock().await;
    wait_port_free(1455).await;
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": codex_jwt("acc"),
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    recording.set_dynamic(
        |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            Box::pin(std::future::ready(Ok(String::from(
                "http://localhost:1455/auth/callback?code=C&state=wrong",
            ))))
        },
    );
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the mismatched state fails login");
    assert_eq!(error.to_string(), "State mismatch");
}

#[tokio::test]
async fn the_openai_codex_manual_paste_without_a_code_rejects() {
    let _port = CODEX_PORT.lock().await;
    wait_port_free(1455).await;
    let mock = MockHttpClient::new();
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    recording.set_dynamic(
        |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            Box::pin(std::future::ready(Ok(String::from("   "))))
        },
    );
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the empty paste fails login");
    assert_eq!(error.to_string(), "Missing authorization code");
}

#[tokio::test]
async fn the_openai_codex_paste_branch_table_drives_parse_authorization_input() {
    let _port = CODEX_PORT.lock().await;
    wait_port_free(1455).await;
    // A pasted URL with a matching state mints the credential.
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    let for_answer = Arc::new(recording.clone());
    recording.set_dynamic(
        move |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            let state = for_answer
                .auth_url()
                .as_deref()
                .and_then(|url| url_query_param(url, "state"))
                .expect("the auth URL carries the state");
            Box::pin(std::future::ready(Ok(format!(
                "http://localhost:1455/auth/callback?code=url-code&state={state}"
            ))))
        },
    );
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": codex_jwt("acc"),
                "refresh_token": "r",
                "expires_in": 60,
            }),
        ));
    let credential = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect("the pasted url mints the credential");
    wait_port_free(1455).await;
    assert_eq!(
        credential
            .extra
            .get("accountId")
            .and_then(serde_json::Value::as_str),
        Some("acc")
    );
    let pairs = form_pairs(&mock, CODEX_TOKEN_URL);
    assert!(pairs.contains(&(String::from("code"), String::from("url-code"))));

    // A `code=` form body parses as the code.
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    recording.set_dynamic(
        |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            Box::pin(std::future::ready(Ok(String::from("code=form-code"))))
        },
    );
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": codex_jwt("acc"),
                "refresh_token": "r",
                "expires_in": 60,
            }),
        ));
    let credential = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect("the form paste mints the credential");
    wait_port_free(1455).await;
    assert_eq!(
        credential
            .extra
            .get("accountId")
            .and_then(serde_json::Value::as_str),
        Some("acc")
    );
    let pairs = form_pairs(&mock, CODEX_TOKEN_URL);
    assert!(pairs.contains(&(String::from("code"), String::from("form-code"))));

    // A bare code falls back to itself with the flow's own state.
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    recording.set_dynamic(
        |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            Box::pin(std::future::ready(Ok(String::from("bare-code"))))
        },
    );
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": codex_jwt("acc"),
                "refresh_token": "r",
                "expires_in": 60,
            }),
        ));
    let credential = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect("the bare paste mints the credential");
    assert_eq!(
        credential
            .extra
            .get("accountId")
            .and_then(serde_json::Value::as_str),
        Some("acc")
    );
}

#[tokio::test]
async fn the_openai_codex_select_rejects_unknown_methods() {
    let mock = MockHttpClient::new();
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("telepathy"))]);
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the unknown method fails login");
    assert_eq!(
        error.to_string(),
        "Unknown OpenAI Codex login method: telepathy"
    );
}

#[test]
fn openai_codex_debug_names_the_flow_and_reports_the_subscription() {
    let flow = OpenAICodexOAuth::new(Arc::new(MockHttpClient::new()), stepped_clock());
    assert_eq!(format!("{flow:?}"), "OpenAICodexOAuth");
    assert_eq!(flow.auth().name, "OpenAI (ChatGPT Plus/Pro)");
    assert_eq!(flow.auth().is_subscription, Some(true));
}

#[tokio::test]
async fn the_openai_codex_refresh_reports_its_failure_shapes() {
    // Transport failure: the refresh error is re-reported with the prefix.
    let mock = MockHttpClient::new();
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the transport failure fails refresh");
    assert_eq!(
        error.to_string(),
        "OpenAI Codex token refresh error: no mock route matched POST https://auth.openai.com/oauth/token"
    );

    // Non-2xx with a body: the body is the detail.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(MockResponse::status(400).with_body("denied"));
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the failure status fails refresh");
    assert_eq!(
        error.to_string(),
        "OpenAI Codex token refresh failed (400): denied"
    );

    // Missing token fields: the response is echoed.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({"access_token": "a"}),
        ));
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the missing fields fail refresh");
    assert!(
        error
            .to_string()
            .starts_with("OpenAI Codex token refresh response missing fields: "),
        "the missing-fields message echoes the json: {error:?}"
    );
}

#[tokio::test]
async fn an_unextractable_account_id_rejects_the_credential() {
    let _port = CODEX_PORT.lock().await;
    wait_port_free(1455).await;
    // A two-segment token has no JWT payload to decode.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": "header.signature",
                "refresh_token": "r",
                "expires_in": 60,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    recording.set_dynamic(
        |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            Box::pin(std::future::ready(Ok(String::from("bare-code"))))
        },
    );
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the unextractable account id fails login");
    assert_eq!(error.to_string(), "Failed to extract accountId from token");
    wait_port_free(1455).await;

    // A three-segment token whose payload is not base64 JSON.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": "h.!!!not-base64!!!.s",
                "refresh_token": "r",
                "expires_in": 60,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    recording.set_dynamic(
        |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            Box::pin(std::future::ready(Ok(String::from("bare-code"))))
        },
    );
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the undecodable payload fails login");
    assert_eq!(error.to_string(), "Failed to extract accountId from token");
}

#[tokio::test]
async fn the_openai_codex_device_start_rejects_malformed_shapes() {
    // A failure status with a body carries the body; without one, just the
    // status.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_USERCODE_URL)
        .respond(MockResponse::status(500).with_body("kaput"));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("device_code"))]);
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the failed start fails login");
    assert_eq!(
        error.to_string(),
        "OpenAI Codex device code request failed with status 500: kaput"
    );

    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_USERCODE_URL)
        .respond(MockResponse::status(502));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("device_code"))]);
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the failed start fails login");
    assert_eq!(
        error.to_string(),
        "OpenAI Codex device code request failed with status 502",
        "an empty body leaves the message at the status"
    );

    // Missing fields: the response is echoed.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_USERCODE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({"device_auth_id": "d"}),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("device_code"))]);
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the incomplete response fails login");
    assert!(
        error
            .to_string()
            .starts_with("Invalid OpenAI Codex device code response: "),
        "the invalid-response message echoes the json: {error:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn the_openai_codex_device_flow_polls_failure_shapes() {
    for (responses, expected) in [
        (
            vec![
                json_response(500, &serde_json::json!({"error": {"code": "slow_down"}})),
                json_response(
                    200,
                    &serde_json::json!({
                        "authorization_code": "oauth-code",
                        "code_verifier": "device-code-verifier",
                    }),
                ),
            ],
            None,
        ),
        (
            vec![json_response(
                400,
                &serde_json::json!({"error": {"code": "boom"}}),
            )],
            Some("OpenAI Codex device auth failed with status 400"),
        ),
        (
            vec![MockResponse::status(400).with_body("oops")],
            Some("OpenAI Codex device auth failed with status 400: oops"),
        ),
        (
            vec![MockResponse::status(400)],
            Some("OpenAI Codex device auth failed with status 400"),
        ),
    ] {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url == CODEX_USERCODE_URL)
            .respond(json_response(
                200,
                &serde_json::json!({
                    "device_auth_id": "device-auth-id",
                    "user_code": "ABCD-1234",
                    "interval": 5,
                }),
            ));
        mock.on(|request| request.url == CODEX_DEVICE_TOKEN_URL)
            .respond_sequence(responses);
        if expected.is_none() {
            mock.on(|request| request.url == CODEX_TOKEN_URL)
                .respond(json_response(
                    200,
                    &serde_json::json!({
                        "access_token": codex_jwt("acc"),
                        "refresh_token": "r",
                        "expires_in": 60,
                    }),
                ));
        }
        let recording = RecordingInteraction::with_answers(vec![Ok(String::from("device_code"))]);
        let flow = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock());
        let handle = {
            let recording = recording.clone();
            tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
        };
        advance_until(|| mock.request_count() >= 2, "the first device poll").await;
        if let Some(prefix) = expected {
            advance_until(|| handle.is_finished(), "the failed poll").await;
            let error = handle
                .await
                .expect("the login task joins")
                .expect_err("the failed poll fails login");
            assert!(error.to_string().starts_with(prefix), "{error:?}");
        } else {
            // The slow-down bump adds five seconds to the server interval.
            tokio::time::advance(Duration::from_secs(15)).await;
            let credential = handle
                .await
                .expect("the login task joins")
                .expect("the slow-down poll resolves");
            assert_eq!(
                credential
                    .extra
                    .get("accountId")
                    .and_then(serde_json::Value::as_str),
                Some("acc")
            );
        }
    }
}

#[tokio::test(start_paused = true)]
async fn the_openai_codex_device_flow_rejects_a_token_response_without_fields() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_USERCODE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_auth_id": "device-auth-id",
                "user_code": "ABCD-1234",
                "interval": 5,
            }),
        ));
    mock.on(|request| request.url == CODEX_DEVICE_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({"authorization_code": ""}),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("device_code"))]);
    let flow = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock());
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    advance_until(|| handle.is_finished(), "the failed poll").await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the incomplete token response fails login");
    assert!(
        error
            .to_string()
            .starts_with("Invalid OpenAI Codex device auth token response: "),
        "{error:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn the_openai_codex_device_flow_accepts_a_numeric_interval() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_USERCODE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_auth_id": "device-auth-id",
                "user_code": "ABCD-1234",
                "interval": 5,
            }),
        ));
    mock.on(|request| request.url == CODEX_DEVICE_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "authorization_code": "oauth-code",
                "code_verifier": "device-code-verifier",
            }),
        ));
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": codex_jwt("acc"),
                "refresh_token": "r",
                "expires_in": 60,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("device_code"))]);
    let flow = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock());
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    advance_until(|| mock.request_count() >= 2, "the first device poll").await;
    advance_until(|| handle.is_finished(), "the numeric interval flow").await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("the numeric interval flow completes");
    assert_eq!(
        credential
            .extra
            .get("accountId")
            .and_then(serde_json::Value::as_str),
        Some("acc")
    );
}

// ---------------------------------------------------------------------------
// Kimi Code: poll failure shapes and refresh retry branches
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn the_kimi_poll_failure_shapes_reject_with_their_wire_messages() {
    for (responses, expected) in [
        (
            vec![json_response(
                500,
                &serde_json::json!({"error": "server_error"}),
            )],
            "Kimi Code device token request failed with status 500",
        ),
        (
            vec![json_response(
                400,
                &serde_json::json!({"error": "expired_token"}),
            )],
            "Kimi Code device authorization expired. Please restart login.",
        ),
        (
            vec![json_response(
                400,
                &serde_json::json!({"error": "access_denied"}),
            )],
            "Kimi Code login was denied.",
        ),
        (
            vec![json_response(
                400,
                &serde_json::json!({"error": "boom", "error_description": "bad"}),
            )],
            "Kimi Code device token request failed (status 400): boom: bad",
        ),
        (
            vec![MockResponse::status(400).with_body("oops")],
            "Kimi Code device token request failed (status 400)",
        ),
        (
            vec![MockResponse::status(200).with_body("[1]")],
            "Kimi Code device token request failed (status 200)",
        ),
    ] {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url == KIMI_DEVICE_URL)
            .respond(json_response(
                200,
                &serde_json::json!({
                    "device_code": "device-code-123",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://www.kimi.com/code",
                    "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
                    "interval": 5,
                    "expires_in": 600,
                }),
            ));
        mock.on(|request| request.url == KIMI_TOKEN_URL)
            .respond_sequence(responses);
        let recording = RecordingInteraction::new();
        let flow = KimiCodingOAuth::new(Arc::new(mock.clone()));
        let handle = {
            let recording = recording.clone();
            tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
        };
        // waitBeforeFirstPoll: the first poll lands after the interval.
        tokio::time::advance(Duration::from_secs(5)).await;
        advance_until(|| handle.is_finished(), "the failed poll").await;
        let error = handle
            .await
            .expect("the login task joins")
            .expect_err("the failed poll fails login");
        assert!(
            error.to_string().starts_with(expected),
            "expected {expected:?}: {error:?}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn the_kimi_poll_rejects_a_token_response_missing_fields() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_DEVICE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_code": "device-code-123",
                "user_code": "ABCD-1234",
                "verification_uri": "https://www.kimi.com/code",
                "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
                "interval": 5,
                "expires_in": 600,
            }),
        ));
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({"access_token": "a"}),
        ));
    let recording = RecordingInteraction::new();
    let flow = KimiCodingOAuth::new(Arc::new(mock.clone()));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    tokio::time::advance(Duration::from_secs(5)).await;
    advance_until(|| handle.is_finished(), "the failed poll").await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the missing fields fail the poll");
    assert!(
        error
            .to_string()
            .starts_with("Kimi Code token poll response missing fields: "),
        "{error:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn the_kimi_poll_slow_down_parks_until_cancelled() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_DEVICE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_code": "device-code-123",
                "user_code": "ABCD-1234",
                "verification_uri": "https://www.kimi.com/code",
                "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
                "interval": 5,
                "expires_in": 600,
            }),
        ));
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(json_response(
            400,
            &serde_json::json!({"error": "slow_down", "interval": 12}),
        ));
    let signal = CancellationToken::new();
    let recording = RecordingInteraction::new();
    let interaction = pi_ai::auth::types::ProviderAuthInteraction::from_interaction(
        recording.interaction(),
        signal.clone(),
    );
    let flow = KimiCodingOAuth::new(Arc::new(mock.clone()));
    let handle = tokio::spawn(async move { flow.login(interaction).await });
    tokio::time::advance(Duration::from_secs(5)).await;
    advance_until(|| mock.request_count() >= 2, "the slow-down poll").await;
    signal.cancel();
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the cancelled poll rejects");
    assert_eq!(error.to_string(), "Login cancelled");
}

#[tokio::test(start_paused = true)]
async fn the_kimi_refresh_rejects_after_the_retries_are_exhausted() {
    // A transport failure on every attempt leaves the last error; the
    // backoff sleeps ride the paused clock.
    let mock = MockHttpClient::new();
    let handle = {
        let mock = mock.clone();
        tokio::spawn(async move {
            KimiCodingOAuth::new(Arc::new(mock))
                .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
                .await
        })
    };
    for backoff in [1_u64, 2, 4] {
        advance_until(|| mock.request_count() >= 1, "the next refresh attempt").await;
        tokio::time::advance(Duration::from_secs(backoff)).await;
    }
    advance_until(|| handle.is_finished(), "the final refresh attempt").await;
    let error = handle
        .await
        .expect("the refresh task joins")
        .expect_err("the exhausted retries fail refresh");
    assert_eq!(
        error.to_string(),
        "no mock route matched POST https://auth.kimi.com/api/oauth/token",
        "the last transport error is the rejection"
    );

    // A final non-retryable failure carries the status and the body.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(json_response(
            400,
            &serde_json::json!({"error": "bad_request", "error_description": "nope"}),
        ));
    let error = KimiCodingOAuth::new(Arc::new(mock))
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the non-retryable failure fails refresh");
    assert_eq!(
        error.to_string(),
        "Kimi Code token refresh failed with status 400: {\"error\":\"bad_request\",\"error_description\":\"nope\"}"
    );
}

#[tokio::test(start_paused = true)]
async fn the_kimi_refresh_retries_a_429_then_succeeds() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond_sequence(vec![
            MockResponse::status(429).with_body("{}"),
            json_response(
                200,
                &serde_json::json!({
                    "access_token": "a",
                    "refresh_token": "r",
                    "expires_in": 60,
                }),
            ),
        ]);
    let handle = {
        let mock = mock.clone();
        tokio::spawn(async move {
            KimiCodingOAuth::new(Arc::new(mock))
                .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
                .await
        })
    };
    // The 429 must be served before the backoff starts; drive the clock.
    advance_until(
        || mock.request_count() >= 1 || handle.is_finished(),
        "the first refresh attempt",
    )
    .await;
    tokio::time::advance(Duration::from_secs(1)).await;
    let credential = handle
        .await
        .expect("the refresh task joins")
        .expect("the retried refresh resolves");
    assert_eq!(credential.access, "a");
    assert_eq!(credential.refresh, "r");
}

#[tokio::test(start_paused = true)]
async fn the_kimi_refresh_reports_the_unauthorized_shapes() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(json_response(403, &serde_json::json!({})));
    let error = KimiCodingOAuth::new(Arc::new(mock))
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the 403 fails refresh");
    assert_eq!(
        error.to_string(),
        "Kimi Code token refresh unauthorized (status 403)"
    );

    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(json_response(
            400,
            &serde_json::json!({"error": "invalid_grant", "error_description": "revoked"}),
        ));
    let error = KimiCodingOAuth::new(Arc::new(mock))
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the invalid grant fails refresh");
    assert_eq!(
        error.to_string(),
        "Kimi Code token refresh unauthorized (status 400): revoked"
    );
}

#[tokio::test]
async fn a_pre_cancelled_kimi_refresh_rejects_as_aborted() {
    let mock = MockHttpClient::new();
    let signal = CancellationToken::new();
    signal.cancel();
    let error = KimiCodingOAuth::new(Arc::new(mock))
        .refresh(oauth_credentials("a", "r", 0), signal)
        .await
        .expect_err("the cancelled refresh rejects");
    assert_eq!(error.to_string(), "Kimi Code token refresh aborted");
}

#[tokio::test(start_paused = true)]
async fn a_kimi_refresh_cancelled_during_the_backoff_rejects_as_aborted() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(json_response(500, &serde_json::json!({})));
    let signal = CancellationToken::new();
    let flow = KimiCodingOAuth::new(Arc::new(mock.clone()));
    let refresh_signal = signal.clone();
    let handle = tokio::spawn(async move {
        flow.refresh(oauth_credentials("a", "r", 0), refresh_signal)
            .await
    });

    advance_until(|| mock.request_count() >= 1, "the first refresh attempt").await;
    signal.cancel();
    let error = handle
        .await
        .expect("the refresh task joins")
        .expect_err("the cancelled backoff fails refresh");
    assert_eq!(error.to_string(), "Kimi Code token refresh aborted");
}

#[test]
fn the_kimi_flow_reports_its_name() {
    let flow = KimiCodingOAuth::new(Arc::new(MockHttpClient::new()));
    assert_eq!(flow.auth().name, "Kimi Code (subscription)");
}

// ---------------------------------------------------------------------------
// Anthropic: callback error pages, the manual paste race, exchange failures
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_anthropic_callback_pages_reject_without_derailing_the_login() {
    let _port = ANTHROPIC_PORT.lock().await;
    wait_port_free(ANTHROPIC_CALLBACK_PORT).await;
    let mock = MockHttpClient::new();
    mount_anthropic_token_route(&mock);
    let recording = RecordingInteraction::new();
    let flow = AnthropicOAuth::new(Arc::new(mock.clone()), stepped_clock());
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };

    let auth_url = wait_until(|| recording.auth_url(), "the authorize URL").await;
    let state = url_query_param(&auth_url, "state").expect("the authorize URL carries the state");

    // Route not found.
    let response = send_request(
        ANTHROPIC_CALLBACK_PORT,
        "GET /nope HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 404 Not Found")
            && response.contains("Callback route not found."),
        "the unknown route rejects: {response:?}"
    );

    // Provider error parameter.
    let response = send_request(
        ANTHROPIC_CALLBACK_PORT,
        "GET /callback?error=access_denied HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 400 Bad Request")
            && response.contains("Anthropic authentication did not complete.")
            && response.contains("Error: access_denied"),
        "the error page carries the reason: {response:?}"
    );

    // Missing code and missing state.
    let response = send_request(
        ANTHROPIC_CALLBACK_PORT,
        "GET /callback HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    .await;
    assert!(
        response.contains("Missing code or state parameter."),
        "the missing-parameters page rejects: {response:?}"
    );
    let response = send_request(
        ANTHROPIC_CALLBACK_PORT,
        "GET /callback?code=C HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    .await;
    assert!(
        response.contains("Missing code or state parameter."),
        "a code without a state is incomplete: {response:?}"
    );

    // State mismatch.
    let response = send_request(
        ANTHROPIC_CALLBACK_PORT,
        "GET /callback?code=C&state=wrong HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    .await;
    assert!(
        response.contains("State mismatch."),
        "the mismatch page rejects: {response:?}"
    );

    // The real callback still completes the login.
    let response = send_request(
        ANTHROPIC_CALLBACK_PORT,
        &format!("GET /callback?code=C&state={state} HTTP/1.1\r\nHost: x\r\n\r\n"),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "the late valid callback completes the login: {response:?}"
    );
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, "access-token");
    wait_port_free(ANTHROPIC_CALLBACK_PORT).await;
}

/// Mount the token route a browser-path login exchanges against.
fn mount_anthropic_token_route(mock: &MockHttpClient) {
    mock.on(|request| request.url == ANTHROPIC_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            }),
        ));
}

/// An anthropic login whose manual prompt answers `paste` (no callback).
async fn anthropic_login_with_paste(
    mock: MockHttpClient,
    paste: String,
) -> Result<OAuthCredentials, AuthError> {
    let _port = ANTHROPIC_PORT.lock().await;
    let recording = RecordingInteraction::new();
    recording.set_dynamic(move |_prompt| Box::pin(std::future::ready(Ok(paste.clone()))));
    let outcome = AnthropicOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await;
    wait_port_free(ANTHROPIC_CALLBACK_PORT).await;
    outcome
}

#[tokio::test]
async fn the_anthropic_login_closure_drives_the_flow_through_the_paste() {
    // The loader seam wires the merged core's login closure over this flow;
    // the paste path drives it under the fixed callback port.
    let _port = ANTHROPIC_PORT.lock().await;
    let mock = MockHttpClient::new();
    mount_anthropic_token_route(&mock);
    let recording = RecordingInteraction::new();
    recording.set_dynamic({
        let _ = &recording;
        move |_prompt| Box::pin(std::future::ready(Ok(String::from("bare-code"))))
    });
    let flow = AnthropicOAuth::new(Arc::new(mock.clone()), stepped_clock());
    let credential = ((flow.auth().login)(provider_interaction(&recording)))
        .await
        .expect("the login closure mints the credential");
    wait_port_free(ANTHROPIC_CALLBACK_PORT).await;
    assert_eq!(credential.access, "access-token");
    let exchange = json_body(&mock, ANTHROPIC_TOKEN_URL);
    assert_eq!(exchange["code"], "bare-code");
}

#[tokio::test]
async fn the_anthropic_manual_paste_state_mismatch_rejects() {
    let error = anthropic_login_with_paste(
        MockHttpClient::new(),
        String::from("http://localhost:53692/callback?code=C&state=wrong"),
    )
    .await
    .expect_err("the mismatched state fails login");
    assert_eq!(error.to_string(), "OAuth state mismatch");

    // A `code#state` fragment paste parses both halves; the mismatching
    // state rejects the paste, upstream's parseAuthorizationInput branch.
    let error = anthropic_login_with_paste(
        MockHttpClient::new(),
        String::from("some-code#not-the-state"),
    )
    .await
    .expect_err("the mismatched fragment fails login");
    assert_eq!(error.to_string(), "OAuth state mismatch");

    // A `code=...` form paste with a state parses both halves too.
    let error = anthropic_login_with_paste(
        MockHttpClient::new(),
        String::from("code=some-code&state=not-the-state"),
    )
    .await
    .expect_err("the mismatched form paste fails login");
    assert_eq!(error.to_string(), "OAuth state mismatch");
}

#[tokio::test]
async fn the_anthropic_manual_paste_without_a_state_uses_the_verifier() {
    let mock = MockHttpClient::new();
    mount_anthropic_token_route(&mock);
    let credential = anthropic_login_with_paste(mock.clone(), String::from("bare-code"))
        .await
        .expect("the bare paste mints the credential");
    assert_eq!(credential.access, "access-token");
    let exchange = json_body(&mock, ANTHROPIC_TOKEN_URL);
    assert_eq!(
        exchange["state"], exchange["code_verifier"],
        "the pasted code without a state exchanges against the flow's verifier"
    );
    assert_eq!(exchange["code"], "bare-code");
}

#[tokio::test]
async fn the_anthropic_manual_prompt_rejection_surfaces() {
    let _port = ANTHROPIC_PORT.lock().await;
    wait_port_free(ANTHROPIC_CALLBACK_PORT).await;
    let recording = RecordingInteraction::with_answers(vec![Err(AbortError)]);
    let error = AnthropicOAuth::new(Arc::new(MockHttpClient::new()), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the rejected prompt fails login");
    assert_eq!(error.to_string(), AbortError::MESSAGE);
}

#[tokio::test]
async fn the_anthropic_exchange_reports_its_failure_shapes() {
    // Transport failure.
    let error = anthropic_login_with_paste(MockHttpClient::new(), String::from("bare-code"))
        .await
        .expect_err("the transport failure fails login");
    assert_eq!(
        error.to_string(),
        "Token exchange request failed. url=https://platform.claude.com/v1/oauth/token; \
         redirect_uri=http://localhost:53692/callback; response_type=authorization_code; \
         details=no mock route matched POST https://platform.claude.com/v1/oauth/token"
    );

    // Non-2xx.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == ANTHROPIC_TOKEN_URL)
        .respond(MockResponse::status(400).with_body("denied"));
    let error = anthropic_login_with_paste(mock, String::from("bare-code"))
        .await
        .expect_err("the failure status fails login");
    assert_eq!(
        error.to_string(),
        "Token exchange request failed. url=https://platform.claude.com/v1/oauth/token; \
         redirect_uri=http://localhost:53692/callback; response_type=authorization_code; \
         details=HTTP request failed. status=400; url=https://platform.claude.com/v1/oauth/token; \
         body=denied"
    );

    // Invalid JSON.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == ANTHROPIC_TOKEN_URL)
        .respond(MockResponse::status(200).with_body("not json"));
    let error = anthropic_login_with_paste(mock, String::from("bare-code"))
        .await
        .expect_err("the invalid JSON fails login");
    assert_eq!(
        error.to_string(),
        "Token exchange returned invalid JSON. url=https://platform.claude.com/v1/oauth/token; \
         body=not json; details=expected ident at line 1 column 2"
    );

    // Non-object JSON.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == ANTHROPIC_TOKEN_URL)
        .respond(MockResponse::status(200).with_body("[1]"));
    let error = anthropic_login_with_paste(mock, String::from("bare-code"))
        .await
        .expect_err("the array body fails login");
    assert_eq!(error.to_string(), "Token exchange returned invalid JSON");

    // Missing expires_in.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == ANTHROPIC_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({"access_token": "a", "refresh_token": "r"}),
        ));
    let error = anthropic_login_with_paste(mock, String::from("bare-code"))
        .await
        .expect_err("the missing expiry fails login");
    assert_eq!(
        error.to_string(),
        "Token exchange returned invalid JSON: missing expires_in"
    );
}

// ---------------------------------------------------------------------------
// OpenAI Codex: the callback server's own error pages and refresh shapes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_openai_codex_callback_pages_reject_without_derailing_the_login() {
    let _port = CODEX_PORT.lock().await;
    wait_port_free(1455).await;
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": codex_jwt("acc"),
                "refresh_token": "refresh-token",
                "expires_in": 3600,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    let flow = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock());
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };

    let auth_url = wait_until(|| recording.auth_url(), "the authorize URL").await;
    let state = url_query_param(&auth_url, "state").expect("the authorize URL carries the state");

    // Route not found.
    let response = send_request(1455, "GET /nope HTTP/1.1\r\nHost: x\r\n\r\n").await;
    assert!(
        response.starts_with("HTTP/1.1 404 Not Found")
            && response.contains("Callback route not found."),
        "the unknown route rejects: {response:?}"
    );

    // State mismatch.
    let response = send_request(
        1455,
        "GET /auth/callback?code=C&state=wrong HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 400 Bad Request") && response.contains("State mismatch."),
        "the mismatch page rejects: {response:?}"
    );

    // Missing code (with a valid state).
    let response = send_request(
        1455,
        &format!("GET /auth/callback?state={state} HTTP/1.1\r\nHost: x\r\n\r\n"),
    )
    .await;
    assert!(
        response.contains("Missing authorization code."),
        "the missing-code page rejects: {response:?}"
    );

    // The real callback still completes the login.
    let response = send_request(
        1455,
        &format!("GET /auth/callback?code=C&state={state} HTTP/1.1\r\nHost: x\r\n\r\n"),
    )
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "the late valid callback completes the login: {response:?}"
    );
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(
        credential
            .extra
            .get("accountId")
            .and_then(serde_json::Value::as_str),
        Some("acc")
    );
    wait_port_free(1455).await;
}

#[tokio::test]
async fn the_openai_codex_manual_prompt_rejection_surfaces() {
    let _port = CODEX_PORT.lock().await;
    wait_port_free(1455).await;
    let recording =
        RecordingInteraction::with_answers(vec![Ok(String::from("browser")), Err(AbortError)]);
    let error = OpenAICodexOAuth::new(Arc::new(MockHttpClient::new()), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the rejected prompt fails login");
    assert_eq!(error.to_string(), AbortError::MESSAGE);
    wait_port_free(1455).await;
}

#[tokio::test]
async fn the_openai_codex_refresh_empty_failure_body_uses_the_status_reason() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(MockResponse::status(400));
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the empty-body failure fails refresh");
    assert!(
        error
            .to_string()
            .starts_with("OpenAI Codex token refresh failed (400): "),
        "the empty body falls back to the status reason: {error:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn the_openai_codex_poll_error_shapes_without_error_objects() {
    // An error that is a bare string: the code extraction falls through to
    // the generic failure, echoing the body.
    for (body, expected) in [
        (
            serde_json::json!({"error": "boom"}),
            "OpenAI Codex device auth failed with status 400",
        ),
        (
            serde_json::json!({}),
            "OpenAI Codex device auth failed with status 400",
        ),
    ] {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url == CODEX_USERCODE_URL)
            .respond(json_response(
                200,
                &serde_json::json!({
                    "device_auth_id": "device-auth-id",
                    "user_code": "ABCD-1234",
                    "interval": 5,
                }),
            ));
        mock.on(|request| request.url == CODEX_DEVICE_TOKEN_URL)
            .respond(json_response(400, &body));
        let recording = RecordingInteraction::with_answers(vec![Ok(String::from("device_code"))]);
        let flow = OpenAICodexOAuth::new(Arc::new(mock.clone()), stepped_clock());
        let handle = {
            let recording = recording.clone();
            tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
        };
        advance_until(|| handle.is_finished(), "the failed poll").await;
        let error = handle
            .await
            .expect("the login task joins")
            .expect_err("the failed poll fails login");
        assert!(
            error.to_string().starts_with(expected),
            "expected {expected:?}: {error:?}"
        );
        assert!(
            error.to_string().contains("boom") || error.to_string().contains("{}"),
            "the body is echoed: {error:?}"
        );
    }
}

#[tokio::test]
async fn a_jwt_with_an_unparseable_payload_rejects_the_credential() {
    use base64::Engine as _;
    let _port = CODEX_PORT.lock().await;
    wait_port_free(1455).await;
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"plain words, no json");
    let token = format!("h.{payload}.s");
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": token,
                "refresh_token": "r",
                "expires_in": 60,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    recording.set_dynamic(
        |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            Box::pin(std::future::ready(Ok(String::from("bare-code"))))
        },
    );
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the unparseable payload fails login");
    assert_eq!(error.to_string(), "Failed to extract accountId from token");
    wait_port_free(1455).await;
}

#[tokio::test]
async fn a_jwt_without_the_auth_claim_rejects_the_credential() {
    use base64::Engine as _;
    let _port = CODEX_PORT.lock().await;
    wait_port_free(1455).await;
    let payload = serde_json::json!({"other": "claim"}).to_string();
    let token = format!(
        "h.{}.s",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload.as_bytes())
    );
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == CODEX_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": token,
                "refresh_token": "r",
                "expires_in": 60,
            }),
        ));
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("browser"))]);
    recording.set_dynamic(
        |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            Box::pin(std::future::ready(Ok(String::from("bare-code"))))
        },
    );
    let error = OpenAICodexOAuth::new(Arc::new(mock), stepped_clock())
        .login(provider_interaction(&recording))
        .await
        .expect_err("the claimless token fails login");
    assert_eq!(error.to_string(), "Failed to extract accountId from token");
    wait_port_free(1455).await;
}

// ---------------------------------------------------------------------------
// OpenRouter: the remaining parse and exchange branches
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_openrouter_paste_branch_table_drives_parse_authorization_input() {
    // A pasted URL without a code parameter yields no code.
    let mock = MockHttpClient::new();
    let error = openrouter_login_with_paste(mock, String::from("http://cb.example/done"))
        .await
        .expect_err("the codeless URL fails login");
    assert_eq!(error.to_string(), "Missing authorization code");

    // A `code=` form body parses as the code.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(json_response(200, &serde_json::json!({ "key": "sk-form" })));
    let credential = openrouter_login_with_paste(mock, String::from("code=form-code"))
        .await
        .expect("the form paste mints the key");
    assert_eq!(credential.access, "sk-form");

    // A transport failure surfaces the seam's message.
    let mock = MockHttpClient::new();
    let error = openrouter_login_with_paste(mock, String::from("some-code"))
        .await
        .expect_err("the transport failure fails login");
    assert_eq!(
        error.to_string(),
        "no mock route matched POST https://openrouter.ai/api/v1/auth/keys"
    );

    // A success status carrying a non-object body has no key.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(MockResponse::status(200).with_body("[1]"));
    let error = openrouter_login_with_paste(mock, String::from("some-code"))
        .await
        .expect_err("the array body fails login");
    assert_eq!(
        error.to_string(),
        "OpenRouter OAuth response carries no \"key\""
    );

    // A failure status with an unparseable body reports the bare status.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == OPENROUTER_TOKEN_URL)
        .respond(MockResponse::status(400).with_body("not json"));
    let error = openrouter_login_with_paste(mock, String::from("some-code"))
        .await
        .expect_err("the unparseable failure fails login");
    assert_eq!(
        error.to_string(),
        "OpenRouter OAuth key exchange failed (HTTP 400)",
        "no detail survives an unparseable body"
    );
}

#[tokio::test]
async fn an_openrouter_answer_that_cancels_the_signal_fails_the_exchange() {
    let signal = CancellationToken::new();
    let recording = RecordingInteraction::new();
    recording.set_dynamic({
        let signal = signal.clone();
        move |_prompt| -> BoxedFuture<'static, Result<String, AbortError>> {
            let signal = signal.clone();
            Box::pin(async move {
                signal.cancel();
                Ok(String::from("http://cb.example?code=C"))
            })
        }
    });
    let interaction = pi_ai::auth::types::ProviderAuthInteraction::from_interaction(
        recording.interaction(),
        signal.clone(),
    );
    let error = OpenRouterOAuth::new(Arc::new(MockHttpClient::new()))
        .login(interaction)
        .await
        .expect_err("the cancelled exchange fails login");
    assert_eq!(error.to_string(), "Login cancelled");
}

#[tokio::test]
async fn an_openrouter_login_cancelled_mid_flight_reports_the_cancellation() {
    let recording = Arc::new(RecordingInteraction::new());
    let signal = CancellationToken::new();
    let interaction = pi_ai::auth::types::ProviderAuthInteraction::from_interaction(
        recording.interaction(),
        signal.clone(),
    );
    let handle = tokio::spawn(async move {
        OpenRouterOAuth::new(Arc::new(MockHttpClient::new()))
            .login(interaction)
            .await
    });
    wait_for_openrouter_callback_url(&recording).await;
    signal.cancel();
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the mid-flight cancel fails login");
    assert_eq!(error.to_string(), "Login cancelled");
}

#[tokio::test]
async fn a_pasted_fragment_code_parses_for_the_manual_race() {
    // `code#state` parses as the code with a state; a mismatching state
    // rejects the paste before any exchange.
    let error = openrouter_login_with_paste(
        MockHttpClient::new(),
        String::from("some-code#not-the-state"),
    )
    .await
    .expect_err("the fragment paste fails");
    assert!(
        error.to_string().contains("no mock route matched")
            || error.to_string() == "OpenRouter OAuth key exchange failed (HTTP 400)",
        "the fragment's code reached the exchange: {error:?}"
    );
}

// ---------------------------------------------------------------------------
// Kimi Code: start-response validation, poll transport, refresh parse shapes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_kimi_device_start_rejects_untrusted_or_missing_verification_uris() {
    for complete_uri in ["", "ftp://www.kimi.com/code"] {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url == KIMI_DEVICE_URL)
            .respond(json_response(
                200,
                &serde_json::json!({
                    "device_code": "device-code-123",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://www.kimi.com/code",
                    "verification_uri_complete": complete_uri,
                    "interval": 5,
                    "expires_in": 600,
                }),
            ));
        let recording = RecordingInteraction::new();
        let error = KimiCodingOAuth::new(Arc::new(mock))
            .login(provider_interaction(&recording))
            .await
            .expect_err("the untrusted uri fails login");
        assert!(
            error
                .to_string()
                .starts_with("Invalid Kimi Code device authorization response: "),
            "the untrusted uri falls into the missing-field error: {error:?}"
        );
        assert!(
            recording.events().is_empty(),
            "the flow fails before notifying"
        );
    }
}

#[tokio::test]
async fn the_kimi_device_start_failure_without_a_body_stops_at_the_status() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_DEVICE_URL)
        .respond(MockResponse::status(500));
    let recording = RecordingInteraction::new();
    let error = KimiCodingOAuth::new(Arc::new(mock))
        .login(provider_interaction(&recording))
        .await
        .expect_err("the empty-body failure fails login");
    assert_eq!(
        error.to_string(),
        "Kimi Code device authorization failed with status 500",
        "an empty body leaves the message at the status"
    );
}

#[tokio::test]
async fn the_kimi_device_start_rejects_each_missing_field() {
    let complete = serde_json::json!({
        "device_code": "device-code-123",
        "user_code": "ABCD-1234",
        "verification_uri": "https://www.kimi.com/code",
        "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
        "interval": 5,
        "expires_in": 600,
    });
    for _field in [
        "device_code",
        "user_code",
        "verification_uri",
        "verification_uri_complete",
    ] {
        let mut body = complete.clone();
        body["device_code"] = serde_json::json!(null);
        // Only the first miss matters; every miss reports the same message.
        let mock = MockHttpClient::new();
        mock.on(|request| request.url == KIMI_DEVICE_URL)
            .respond(json_response(200, &body));
        let recording = RecordingInteraction::new();
        let error = KimiCodingOAuth::new(Arc::new(mock))
            .login(provider_interaction(&recording))
            .await
            .expect_err("the missing field fails login");
        assert!(
            error
                .to_string()
                .starts_with("Invalid Kimi Code device authorization response: "),
            "{error:?}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn the_kimi_poll_transport_failure_propagates() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_DEVICE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_code": "device-code-123",
                "user_code": "ABCD-1234",
                "verification_uri": "https://www.kimi.com/code",
                "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
                "interval": 5,
                "expires_in": 600,
            }),
        ));
    let recording = RecordingInteraction::new();
    let flow = KimiCodingOAuth::new(Arc::new(mock));
    let handle = {
        let recording = recording.clone();
        tokio::spawn(async move { flow.login(provider_interaction(&recording)).await })
    };
    // waitBeforeFirstPoll parks the flow before the first token request.
    advance_seconds_until(|| handle.is_finished(), 10, "the transport failure").await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the unmatched token route fails login");
    assert_eq!(
        error.to_string(),
        "no mock route matched POST https://auth.kimi.com/api/oauth/token"
    );
}

#[tokio::test(start_paused = true)]
async fn the_kimi_login_cancels_during_the_first_poll_wait() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_DEVICE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_code": "device-code-123",
                "user_code": "ABCD-1234",
                "verification_uri": "https://www.kimi.com/code",
                "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
                "interval": 5,
                "expires_in": 600,
            }),
        ));
    let signal = CancellationToken::new();
    let recording = RecordingInteraction::new();
    let interaction = pi_ai::auth::types::ProviderAuthInteraction::from_interaction(
        recording.interaction(),
        signal.clone(),
    );
    let flow = KimiCodingOAuth::new(Arc::new(mock));
    let handle = tokio::spawn(async move { flow.login(interaction).await });

    // The device code event announces the first poll wait; cancelling then
    // ends the flow inside that wait, upstream's abort listener.
    wait_until(
        || (!recording.events().is_empty()).then_some(()),
        "the device code event",
    )
    .await;
    signal.cancel();
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the cancelled first wait fails login");
    assert_eq!(error.to_string(), "Login cancelled");
}

#[tokio::test(start_paused = true)]
async fn the_kimi_refresh_parse_and_body_shapes() {
    // A 200 whose body is not an object parses as an empty token response.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(MockResponse::status(200).with_body("[1]"));
    let error = KimiCodingOAuth::new(Arc::new(mock))
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the non-object body fails refresh");
    assert_eq!(
        error.to_string(),
        "Kimi Code token refresh response missing fields: {}",
        "the array body parses to an empty object: {error:?}"
    );

    // A non-retryable failure with an empty body stops at the status.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(MockResponse::status(400));
    let error = KimiCodingOAuth::new(Arc::new(mock))
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the empty failure fails refresh");
    assert_eq!(
        error.to_string(),
        "Kimi Code token refresh failed with status 400",
        "no body, no suffix"
    );

    // A 500 with an empty body repeats the retry-then-fail shape.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == KIMI_TOKEN_URL)
        .respond(MockResponse::status(400));
    let error = KimiCodingOAuth::new(Arc::new(mock))
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the bodyless 400 fails refresh");
    assert_eq!(
        error.to_string(),
        "Kimi Code token refresh failed with status 400"
    );
}
