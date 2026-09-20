//! The Radius gateway browser-login and failure paths, extending
//! `test/radius-oauth.test.ts` (whose device happy path, refresh, and
//! discovery-invalid cases live in `radius_oauth.rs`) at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The browser path drives the loopback callback server on the flow's fixed
//! port with a real `TcpStream` GET — the callback a browser would send —
//! and the failure branches pin the wire-shaped error messages upstream's
//! suites assert on.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::auth_interaction::{ScriptedAuthInteraction, provider_interaction};
use common::seam_forms::{form_field, form_fields};
use pi_ai::auth::clock::{AuthClock as _, FixedClock};
use pi_ai::auth::oauth::radius::{RadiusOAuth, normalize_radius_gateway_url};
use pi_ai::auth::types::{AuthEvent, ModelAuth, OAuthAuth as _, OAuthCredential};
use pi_ai::http::{MockHttpClient, MockResponse, json_response};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_util::sync::CancellationToken;

const GATEWAY: &str = "https://radius.example";
const DISCOVERY_URL: &str = "https://radius.example/v1/oauth";
const DEVICE_URL: &str = "https://radius.example/v1/oauth/device";
const TOKEN_URL: &str = "https://radius.example/v1/oauth/token";
const AUTHORIZE_ENDPOINT: &str = "https://radius-ui.example/authorize";
const CALLBACK_PORT: u16 = 1456;

/// `new Date("2026-07-24T00:00:00Z").getTime()`, upstream's pinned system
/// time.
const START: i64 = 1_784_851_200_000;

/// The two port-1456 cases serialize: the login's callback server claims the
/// fixed port for the whole login.
static CALLBACK_PORT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// `createRadiusOAuth({ name: "Radius", gateway: GATEWAY })` over the fixed
/// epoch.
fn radius_oauth(mock: &MockHttpClient, clock: &FixedClock) -> RadiusOAuth {
    RadiusOAuth::new(
        "Radius".to_owned(),
        GATEWAY.to_owned(),
        Arc::new(mock.clone()),
        Arc::new(FixedClock::new(clock.now_ms())),
    )
}

/// The scripted interaction answering `login_method`, upstream's
/// `interaction(loginMethod, events)` helper.
fn interaction(
    login_method: &str,
) -> (
    pi_ai::auth::types::ProviderAuthInteraction,
    Arc<ScriptedAuthInteraction>,
) {
    let scripted = Arc::new(ScriptedAuthInteraction::answering(login_method));
    (
        provider_interaction(Arc::clone(&scripted), CancellationToken::new()),
        scripted,
    )
}

/// Wait for a condition the login task reaches, stepping the scheduler
/// between probes.
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

/// The URL of the first reported authorize event, asserting one was reported.
async fn wait_for_auth_url(scripted: &Arc<ScriptedAuthInteraction>) -> String {
    wait_until(
        || {
            scripted.events().into_iter().find_map(|event| match event {
                AuthEvent::AuthUrl { url, .. } => Some(url),
                _ => None,
            })
        },
        "the authorize URL",
    )
    .await
}

/// A query parameter of a URL, the authorize URL's carried params.
fn url_query_param(url: &str, name: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()?
        .query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

/// The one-request-per-connection GET the browser sends to the flow's fixed
/// loopback port, returning the raw response.
async fn send_callback(request: &str) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", CALLBACK_PORT))
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

/// The discovery + token routes a browser login shares.
fn mount_browser_routes(mock: &MockHttpClient) {
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
        ));
    mock.on(|request| request.url == TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "scope": "gateway offline_access",
            }),
        ));
}

#[test]
fn normalize_radius_gateway_url_prepends_the_scheme_and_strips_trailing_slashes() {
    assert_eq!(
        normalize_radius_gateway_url("radius.example"),
        "https://radius.example",
        "a bare host gains the https scheme"
    );
    assert_eq!(
        normalize_radius_gateway_url("https://radius.example///"),
        "https://radius.example",
        "every trailing slash goes"
    );
    assert_eq!(
        normalize_radius_gateway_url("HTTP://Radius.Example/"),
        "HTTP://Radius.Example",
        "the scheme check is case-insensitive and the spelling survives"
    );
    assert_eq!(
        normalize_radius_gateway_url("http://radius.example"),
        "http://radius.example",
        "an http scheme is kept"
    );
}

#[tokio::test]
async fn browser_login_completes_through_a_real_loopback_callback() {
    let _port = CALLBACK_PORT_LOCK.lock().await;
    let mock = MockHttpClient::new();
    mount_browser_routes(&mock);
    let clock = FixedClock::new(START);
    let oauth = radius_oauth(&mock, &clock);
    let (interaction, scripted) = interaction("browser");
    let handle = tokio::spawn(async move { oauth.login(interaction).await });

    let auth_url = wait_for_auth_url(&scripted).await;
    assert!(
        auth_url.starts_with(&format!("{AUTHORIZE_ENDPOINT}?")),
        "the browser opens the discovered endpoint: {auth_url:?}"
    );
    let state = url_query_param(&auth_url, "state").expect("the authorize URL carries the state");
    assert_eq!(
        url_query_param(&auth_url, "client_id"),
        Some(String::from("pi-gateway"))
    );
    assert_eq!(
        url_query_param(&auth_url, "redirect_uri"),
        Some(String::from("http://127.0.0.1:1456/oauth/callback"))
    );
    assert_eq!(
        url_query_param(&auth_url, "scope"),
        Some(String::from("gateway offline_access"))
    );
    assert_eq!(
        url_query_param(&auth_url, "code_challenge_method").as_deref(),
        Some("S256")
    );
    assert!(
        !url_query_param(&auth_url, "code_challenge")
            .unwrap_or_default()
            .is_empty(),
        "the authorize URL carries the PKCE challenge"
    );

    let response = send_callback(&format!(
        "GET /oauth/callback?code=C&state={state} HTTP/1.1\r\nHost: localhost\r\n\r\n"
    ))
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "the browser gets the success page: {response:?}"
    );
    assert!(
        response.contains("Signed in to Radius"),
        "the success page names the gateway: {response:?}"
    );

    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    assert_eq!(
        credential.expires,
        START + 3_600_000 - 60_000,
        "the expiry carries the one-minute skew off the pinned clock"
    );
    assert_eq!(
        credential.extra_string("scope"),
        Some("gateway offline_access")
    );

    // The token request posts the authorization-code grant to the gateway.
    let token_fields = form_fields(
        &mock
            .recorded()
            .into_iter()
            .find(|request| request.url == TOKEN_URL)
            .expect("the token request ran"),
    );
    assert_eq!(
        form_field(&token_fields, "grant_type"),
        "authorization_code"
    );
    assert_eq!(form_field(&token_fields, "client_id"), "pi-gateway");
    assert_eq!(form_field(&token_fields, "code"), "C");
    assert_eq!(
        form_field(&token_fields, "redirect_uri"),
        "http://127.0.0.1:1456/oauth/callback"
    );
    assert!(
        !form_field(&token_fields, "code_verifier").is_empty(),
        "the exchange carries the PKCE verifier"
    );

    // The progress event names the loopback listener.
    assert!(
        scripted.events().iter().any(|event| matches!(
            event,
            AuthEvent::Progress { message }
                if message.contains("127.0.0.1:1456")
        )),
        "the flow announces the callback listener: {:?}",
        scripted.events()
    );
}

#[tokio::test]
async fn a_state_mismatched_callback_is_rejected_and_the_login_can_be_cancelled() {
    let _port = CALLBACK_PORT_LOCK.lock().await;
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
        ));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let signal = CancellationToken::new();
    let scripted = Arc::new(ScriptedAuthInteraction::answering("browser"));
    let interaction = provider_interaction(Arc::clone(&scripted), signal.clone());
    let handle = tokio::spawn(async move { oauth.login(interaction).await });

    let auth_url = wait_for_auth_url(&scripted).await;
    let _state = url_query_param(&auth_url, "state").expect("the state");

    let response =
        send_callback("GET /oauth/callback?code=C&state=wrong HTTP/1.1\r\nHost: x\r\n\r\n").await;
    assert!(
        response.starts_with("HTTP/1.1 400 Bad Request"),
        "the mismatch page rejects: {response:?}"
    );
    assert!(
        response.contains("OAuth state mismatch."),
        "the mismatch page names the reason: {response:?}"
    );

    // The mismatched page never settles the wait, so only the interaction's
    // cancellation ends the login — the abort watcher upstream wires.
    signal.cancel();
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the cancelled login rejects");
    assert_eq!(error.0, "Login cancelled");
}

#[tokio::test]
async fn the_callback_error_and_missing_code_pages_keep_the_login_waiting() {
    let _port = CALLBACK_PORT_LOCK.lock().await;
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
        ));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (interaction, scripted) = interaction("browser");
    let handle = tokio::spawn(async move { oauth.login(interaction).await });

    let auth_url = wait_for_auth_url(&scripted).await;
    let state = url_query_param(&auth_url, "state").expect("the state");

    // An OAuth error from the provider: the page carries the description and
    // the wait hands over as cancelled.
    let response = send_callback(&format!(
        "GET /oauth/callback?state={state}&error=access_denied&error_description=Nope HTTP/1.1\r\nHost: x\r\n\r\n"
    ))
    .await;
    assert!(
        response.starts_with("HTTP/1.1 400 Bad Request"),
        "the error page rejects: {response:?}"
    );
    assert!(
        response.contains("Nope"),
        "the error page carries the description: {response:?}"
    );

    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the error-param callback fails login");
    assert_eq!(
        error.0, "OAuth callback did not complete.",
        "a settled-but-empty wait without cancellation reports the incomplete callback"
    );
}

#[tokio::test]
async fn a_missing_code_page_keeps_the_login_waiting_until_cancelled() {
    let _port = CALLBACK_PORT_LOCK.lock().await;
    let mock = MockHttpClient::new();
    mount_browser_routes(&mock);
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (interaction, scripted) = interaction("browser");
    let handle = tokio::spawn(async move { oauth.login(interaction).await });

    let auth_url = wait_for_auth_url(&scripted).await;
    let state = url_query_param(&auth_url, "state").expect("the state");

    let response = send_callback(&format!(
        "GET /oauth/callback?state={state} HTTP/1.1\r\nHost: x\r\n\r\n"
    ))
    .await;
    assert!(
        response.starts_with("HTTP/1.1 400 Bad Request")
            && response.contains("Missing authorization code."),
        "the missing-code page rejects: {response:?}"
    );

    // A 404 route never settles the wait either.
    let response = send_callback("GET /nope HTTP/1.1\r\nHost: x\r\n\r\n").await;
    assert!(
        response.starts_with("HTTP/1.1 404 Not Found")
            && response.contains("Callback route not found."),
        "the route-not-found page rejects: {response:?}"
    );

    // Both pages leave the wait open, so the login still completes through
    // the real code callback.
    let response = send_callback(&format!(
        "GET /oauth/callback?code=C&state={state} HTTP/1.1\r\nHost: x\r\n\r\n"
    ))
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "the late valid callback still completes the login: {response:?}"
    );
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, "access-token");
}

#[tokio::test]
async fn an_aborted_transport_cancels_the_browser_login() {
    let _port = CALLBACK_PORT_LOCK.lock().await;
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
        ));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let signal = CancellationToken::new();
    let scripted = Arc::new(ScriptedAuthInteraction::answering("browser"));
    let interaction = provider_interaction(Arc::clone(&scripted), signal.clone());
    let handle = tokio::spawn(async move { oauth.login(interaction).await });

    wait_for_auth_url(&scripted).await;
    signal.cancel();
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the cancelled login rejects");
    assert_eq!(error.0, "Login cancelled", "the abort settles the wait");
}

#[tokio::test]
async fn a_failed_discovery_reports_the_gateway_status() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(MockResponse::status(500).with_body("boom"));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (interaction, _scripted) = interaction("browser");

    let error = oauth
        .login(interaction)
        .await
        .expect_err("the failed discovery fails login");
    assert_eq!(
        error.0,
        format!("Could not load Radius OAuth config from {GATEWAY}: 500 boom")
    );
}

#[tokio::test]
async fn a_failed_token_exchange_reports_the_structured_oauth_error() {
    let _port = CALLBACK_PORT_LOCK.lock().await;
    for (body, expected_detail) in [
        (
            serde_json::json!({"error": "invalid_grant", "error_description": "bad code"}),
            "invalid_grant: bad code",
        ),
        (serde_json::json!({"error": "slow_down"}), "slow_down"),
        (
            serde_json::json!({"error_description": "plain text"}),
            "plain text",
        ),
        (serde_json::json!({}), "400"),
    ] {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url == DISCOVERY_URL)
            .respond(json_response(
                200,
                &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
            ));
        mock.on(|request| request.url == TOKEN_URL)
            .respond(json_response(400, &body));
        let oauth = radius_oauth(&mock, &FixedClock::new(START));
        let (interaction, scripted) = interaction("browser");
        let handle = tokio::spawn(async move { oauth.login(interaction).await });
        let auth_url = wait_for_auth_url(&scripted).await;
        let state = url_query_param(&auth_url, "state").expect("the state");
        let response = send_callback(&format!(
            "GET /oauth/callback?code=C&state={state} HTTP/1.1\r\nHost: x\r\n\r\n"
        ))
        .await;
        assert!(
            response.starts_with("HTTP/1.1 200 OK"),
            "the callback succeeded before the exchange failed: {response:?}"
        );
        let error = handle
            .await
            .expect("the login task joins")
            .expect_err("the failed exchange fails login");
        assert_eq!(
            error.0,
            format!("Radius OAuth token request failed: {expected_detail}"),
            "{body:?}"
        );
    }
}

#[tokio::test]
async fn a_failed_device_authorization_reports_the_gateway_status() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(
            500,
            &serde_json::json!({"error": "server_error", "error_description": "kaput"}),
        ));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (interaction, _scripted) = interaction("device-code");
    let error = oauth
        .login(interaction)
        .await
        .expect_err("the failed device authorization fails login");
    assert_eq!(
        error.0,
        "Radius OAuth device authorization failed: server_error: kaput"
    );
}

#[tokio::test]
async fn an_incomplete_device_response_rejects_with_the_missing_fields_message() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_code": "device-code",
                "verification_uri": "https://radius-ui.example/pair",
                "expires_in": 600,
            }),
        ));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (interaction, _scripted) = interaction("device-code");
    let error = oauth
        .login(interaction)
        .await
        .expect_err("the incomplete response fails login");
    assert_eq!(
        error.0,
        "Radius OAuth device authorization response is missing required fields"
    );
}

#[tokio::test(start_paused = true)]
async fn an_unknown_sign_in_method_rejects_with_the_method_error() {
    let mock = MockHttpClient::new();
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (interaction, _scripted) = interaction("telepathy");
    let error = oauth
        .login(interaction)
        .await
        .expect_err("the unknown method fails login");
    assert_eq!(error.0, "Unknown Radius sign-in method: telepathy");
}

#[tokio::test]
async fn the_flow_reports_its_name_and_derives_the_api_key_auth() {
    let mock = MockHttpClient::new();
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    assert_eq!(oauth.name(), "Radius");
    assert_eq!(format!("{oauth:?}"), "RadiusOAuth");
    assert_eq!(
        oauth.to_auth(&OAuthCredential::new("tok", "r", 0)),
        ModelAuth::api_key("tok")
    );
}

#[tokio::test(start_paused = true)]
async fn refresh_failures_report_the_structured_oauth_error() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == TOKEN_URL)
        .respond(json_response(
            400,
            &serde_json::json!({"error": "invalid_grant", "error_description": "expired"}),
        ));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let error = oauth
        .refresh(
            &OAuthCredential::new("old", "old-refresh", 0),
            CancellationToken::new(),
        )
        .await
        .expect_err("the failed refresh rejects");
    assert_eq!(
        error.0,
        "Radius OAuth token request failed: invalid_grant: expired"
    );
}

// ---------------------------------------------------------------------------
// Device-code poll outcome branches
// ---------------------------------------------------------------------------

/// The device routes a poll branch case shares: the device authorization
/// succeeds with a one-second interval and a three-second lifetime, so the
/// poll loop reaches its deadline after three strikes.
fn mount_poll_routes(mock: &MockHttpClient, token_responses: Vec<MockResponse>) {
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_code": "device-code",
                "user_code": "ABCD-1234",
                "verification_uri": "https://radius-ui.example/pair",
                "expires_in": 3,
                "interval": 1,
            }),
        ));
    mock.on(|request| request.url == TOKEN_URL)
        .respond_sequence(token_responses);
}

#[tokio::test(start_paused = true)]
async fn the_device_poll_keeps_parking_on_pending_until_cancellation() {
    let mock = MockHttpClient::new();
    mount_poll_routes(
        &mock,
        vec![
            json_response(400, &serde_json::json!({"error": "authorization_pending"})),
            json_response(400, &serde_json::json!({"error": "authorization_pending"})),
        ],
    );
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let signal = CancellationToken::new();
    let scripted = Arc::new(ScriptedAuthInteraction::answering("device-code"));
    let interaction = provider_interaction(Arc::clone(&scripted), signal.clone());
    let handle = tokio::spawn(async move { oauth.login(interaction).await });

    // The pending poll schedules the server's interval; the flow parks in the
    // wait until the interaction cancels it.
    wait_until(
        || (mock.request_count() >= 2).then_some(()),
        "the first device poll",
    )
    .await;
    signal.cancel();
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the cancelled poll rejects");
    assert_eq!(error.0, "Login cancelled");
}

#[tokio::test(start_paused = true)]
async fn the_device_poll_parks_on_slow_down_until_cancellation() {
    let mock = MockHttpClient::new();
    mount_poll_routes(
        &mock,
        vec![json_response(
            400,
            &serde_json::json!({"error": "slow_down"}),
        )],
    );
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let signal = CancellationToken::new();
    let scripted = Arc::new(ScriptedAuthInteraction::answering("device-code"));
    let interaction = provider_interaction(Arc::clone(&scripted), signal.clone());
    let handle = tokio::spawn(async move { oauth.login(interaction).await });

    wait_until(
        || (mock.request_count() >= 2).then_some(()),
        "the slow-down poll",
    )
    .await;
    signal.cancel();
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the cancelled poll rejects");
    assert_eq!(error.0, "Login cancelled");
}

#[tokio::test(start_paused = true)]
async fn terminal_poll_errors_reject_with_their_wire_messages() {
    for (token_body, expected) in [
        (
            serde_json::json!({"error": "expired_token"}),
            "Device authorization expired.".to_owned(),
        ),
        (
            serde_json::json!({"error": "access_denied"}),
            "Device authorization was denied.".to_owned(),
        ),
        (
            serde_json::json!({"error": "unauthorized_client"}),
            "Radius OAuth token request failed: unauthorized_client".to_owned(),
        ),
    ] {
        let mock = MockHttpClient::new();
        mount_poll_routes(&mock, vec![json_response(400, &token_body)]);
        let oauth = radius_oauth(&mock, &FixedClock::new(START));
        let (interaction, _scripted) = interaction("device-code");
        let error = oauth
            .login(interaction)
            .await
            .expect_err("the terminal poll outcome fails login");
        assert_eq!(error.0, expected, "{token_body:?}");
    }

    // A transport failure propagates as its own message.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "device_code": "device-code",
                "user_code": "ABCD-1234",
                "verification_uri": "https://radius-ui.example/pair",
                "expires_in": 600,
                "interval": 5,
            }),
        ));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (interaction, _scripted) = interaction("device-code");
    let error = oauth
        .login(interaction)
        .await
        .expect_err("the unmatched token route fails login");
    assert!(
        error.0.contains("no mock route matched"),
        "the transport failure propagates: {error:?}"
    );
}
// ---------------------------------------------------------------------------
// Gateway URL, discovery, and device failure shapes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_unparseable_gateway_rejects_every_request_path() {
    let mock = MockHttpClient::new();
    let oauth = RadiusOAuth::new(
        "Radius".to_owned(),
        "ht tp://bad".to_owned(),
        Arc::new(mock.clone()),
        Arc::new(FixedClock::new(START)),
    );
    // The device path builds its URL first.
    let (interaction, _scripted) = interaction("device-code");
    let error = oauth
        .login(interaction)
        .await
        .expect_err("the unparseable gateway fails login");
    assert!(
        error
            .0
            .starts_with("invalid Radius gateway URL https://ht tp://bad"),
        "the gateway parse failure names the gateway: {error:?}"
    );

    // The refresh path reports the same parse failure.
    let error = oauth
        .refresh(&OAuthCredential::new("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the unparseable gateway fails refresh");
    assert!(
        error.0.starts_with("invalid Radius gateway URL"),
        "the refresh fails on the gateway: {error:?}"
    );
}

#[tokio::test]
async fn the_discovery_transport_and_json_failures_report_the_seam() {
    // No discovery route: the transport failure propagates.
    let mock = MockHttpClient::new();
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (browser_interaction, _scripted) = interaction("browser");
    let error = oauth
        .login(browser_interaction)
        .await
        .expect_err("the transport failure fails login");
    assert_eq!(
        error.0,
        "no mock route matched GET https://radius.example/v1/oauth"
    );

    // A discovery body that is not JSON fails the parse.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(MockResponse::status(200).with_body("not json"));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (browser_interaction, _scripted) = interaction("browser");
    let error = oauth
        .login(browser_interaction)
        .await
        .expect_err("the invalid discovery JSON fails login");
    assert!(
        error.0.starts_with("invalid JSON response: "),
        "the discovery JSON failure propagates: {error:?}"
    );
}

#[tokio::test]
async fn an_occupied_callback_port_fails_the_radius_browser_login_before_prompting() {
    let _port = CALLBACK_PORT_LOCK.lock().await;
    let holder = tokio::net::TcpListener::bind(("127.0.0.1", CALLBACK_PORT))
        .await
        .expect("the holder occupies the callback port");

    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
        ));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (interaction, scripted) = interaction("browser");
    let error = oauth
        .login(interaction)
        .await
        .expect_err("the occupied port fails login");
    assert!(
        error
            .0
            .starts_with("could not bind the OAuth callback server on 127.0.0.1"),
        "the bind error surfaces before the browser is sent anywhere: {error:?}"
    );
    assert!(
        scripted.events().is_empty(),
        "the flow fails before announcing the authorize URL"
    );
    drop(holder);
}

#[tokio::test]
async fn the_token_request_transport_failure_surfaces_through_the_browser_path() {
    let _port = CALLBACK_PORT_LOCK.lock().await;
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
        ));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (interaction, scripted) = interaction("browser");
    let handle = tokio::spawn(async move { oauth.login(interaction).await });

    let auth_url = wait_for_auth_url(&scripted).await;
    let state = url_query_param(&auth_url, "state").expect("the state");
    let response = send_callback(&format!(
        "GET /oauth/callback?code=C&state={state} HTTP/1.1\r\nHost: x\r\n\r\n"
    ))
    .await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "the callback succeeded before the exchange failed: {response:?}"
    );
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the token transport failure fails login");
    assert_eq!(
        error.0, "no mock route matched POST https://radius.example/v1/oauth/token",
        "the exchange's transport failure passes through: {error:?}"
    );
}

#[tokio::test]
async fn the_token_exchange_failure_body_shapes_reach_the_error_message() {
    let _port = CALLBACK_PORT_LOCK.lock().await;
    // A non-JSON body becomes the description.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
        ));
    mock.on(|request| request.url == TOKEN_URL)
        .respond(MockResponse::status(400).with_body("oops"));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (browser_interaction, scripted) = interaction("browser");
    let handle = tokio::spawn(async move { oauth.login(browser_interaction).await });
    let auth_url = wait_for_auth_url(&scripted).await;
    let state = url_query_param(&auth_url, "state").expect("the state");
    send_callback(&format!(
        "GET /oauth/callback?code=C&state={state} HTTP/1.1\r\nHost: x\r\n\r\n"
    ))
    .await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the non-JSON failure fails login");
    assert_eq!(error.0, "Radius OAuth token request failed: oops");

    // An empty body falls back to the status.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
        ));
    mock.on(|request| request.url == TOKEN_URL)
        .respond(MockResponse::status(400));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (browser_interaction, scripted) = interaction("browser");
    let handle = tokio::spawn(async move { oauth.login(browser_interaction).await });
    let auth_url = wait_for_auth_url(&scripted).await;
    let state = url_query_param(&auth_url, "state").expect("the state");
    send_callback(&format!(
        "GET /oauth/callback?code=C&state={state} HTTP/1.1\r\nHost: x\r\n\r\n"
    ))
    .await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the empty failure fails login");
    assert_eq!(
        error.0, "Radius OAuth token request failed: 400",
        "the empty body falls back to the status: {error:?}"
    );
}

#[tokio::test]
async fn the_device_request_transport_and_parse_failures_propagate() {
    // No device route: the transport failure propagates.
    let mock = MockHttpClient::new();
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (device_interaction, _scripted) = interaction("device-code");
    let error = oauth
        .login(device_interaction)
        .await
        .expect_err("the transport failure fails login");
    assert_eq!(
        error.0,
        "no mock route matched POST https://radius.example/v1/oauth/device"
    );

    // A non-JSON device body fails the parse.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_URL)
        .respond(MockResponse::status(200).with_body("not json"));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (device_interaction, _scripted) = interaction("device-code");
    let error = oauth
        .login(device_interaction)
        .await
        .expect_err("the invalid device JSON fails login");
    assert!(error.0.starts_with("invalid JSON response: "), "{error:?}");

    // A device failure without a body stops at the status.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_URL)
        .respond(MockResponse::status(500));
    let oauth = radius_oauth(&mock, &FixedClock::new(START));
    let (device_interaction, _scripted) = interaction("device-code");
    let error = oauth
        .login(device_interaction)
        .await
        .expect_err("the empty failure fails login");
    assert_eq!(
        error.0, "Radius OAuth device authorization failed: 500",
        "the empty body falls back to the status"
    );
}
