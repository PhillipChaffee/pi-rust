//! Radius-gateway test fixtures shared by the browser-path and device-path
//! suites, the shapes upstream's `test/radius-oauth.test.ts` builds inline
//! (the `createRadiusOAuth` factory, the `interaction(loginMethod, events)`
//! helper, and the discovery/device route stubs) at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use pi_ai::auth::oauth::radius::RadiusOAuth;
use pi_ai::auth::types::AuthEvent;
use pi_ai::http::{MockHttpClient, MockResponse};
use tokio_util::sync::CancellationToken;

use super::auth_interaction::{ScriptedAuthInteraction, provider_interaction};
use super::oauth_fixtures::{mount_json_route, wait_until};

/// The gateway host every Radius case points at.
pub const GATEWAY: &str = "https://radius.example";
/// The discovery endpoint the browser login fetches first.
pub const DISCOVERY_URL: &str = "https://radius.example/v1/oauth";
/// The device-authorization endpoint the device login posts.
pub const DEVICE_URL: &str = "https://radius.example/v1/oauth/device";
/// The token endpoint both login paths exchange against.
pub const TOKEN_URL: &str = "https://radius.example/v1/oauth/token";
/// The authorize endpoint discovery hands the browser.
pub const AUTHORIZE_ENDPOINT: &str = "https://radius-ui.example/authorize";
/// The fixed loopback port the flow's callback server claims.
pub const CALLBACK_PORT: u16 = 1456;

/// `new Date("2026-07-24T00:00:00Z").getTime()`, upstream's pinned system
/// time.
pub const START: i64 = 1_784_851_200_000;

/// `createRadiusOAuth({ name: "Radius", gateway: GATEWAY })` over a clock
/// pinned at `epoch_ms` — the stepped clock freezes the epoch while the
/// browser path spends real time on the loopback callback.
#[must_use]
pub fn radius_oauth(mock: &MockHttpClient, epoch_ms: i64) -> RadiusOAuth {
    RadiusOAuth::new(
        "Radius".to_owned(),
        GATEWAY.to_owned(),
        Arc::new(mock.clone()),
        Arc::new(pi_ai::auth::clock::SteppedClock::new(epoch_ms)),
    )
}

/// The scripted interaction answering `login_method`, upstream's
/// `interaction(loginMethod, events)` helper.
#[must_use]
pub fn interaction(
    login_method: &str,
) -> (
    pi_ai::auth::types::ProviderAuthInteraction,
    Arc<ScriptedAuthInteraction>,
) {
    let scripted = Arc::new(ScriptedAuthInteraction::answering(login_method));
    (
        provider_interaction(&scripted, CancellationToken::new()),
        scripted,
    )
}

/// The URL of the first reported authorize event, asserting one was reported.
pub async fn wait_for_auth_url(scripted: &Arc<ScriptedAuthInteraction>) -> String {
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

/// Mount the discovery route answering the browser authorize endpoint.
pub fn mount_discovery(mock: &MockHttpClient) {
    mount_json_route(
        mock,
        DISCOVERY_URL,
        200,
        &serde_json::json!({ "authorizationEndpoint": AUTHORIZE_ENDPOINT }),
    );
}

/// Mount the token route answering the standard rotated credential with the
/// gateway scope, the reply every successful exchange reads.
pub fn mount_token_success(mock: &MockHttpClient) {
    mount_json_route(
        mock,
        TOKEN_URL,
        200,
        &serde_json::json!({
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "scope": "gateway offline_access",
        }),
    );
}

/// Mount the discovery and token routes a browser login shares.
pub fn mount_browser_routes(mock: &MockHttpClient) {
    mount_discovery(mock);
    mount_token_success(mock);
}

/// Mount the device-start route succeeding with the given lifetime and
/// interval, the response every device-poll case starts from.
pub fn mount_device_start(mock: &MockHttpClient, expires_in: i64, interval: i64) {
    mount_json_route(
        mock,
        DEVICE_URL,
        200,
        &serde_json::json!({
            "device_code": "device-code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://radius-ui.example/pair",
            "expires_in": expires_in,
            "interval": interval,
        }),
    );
}

/// Mount the device routes a poll case shares: the device start plus the
/// token route answering the queued responses in order.
pub fn mount_device_routes(
    mock: &MockHttpClient,
    expires_in: i64,
    interval: i64,
    token_responses: Vec<MockResponse>,
) {
    mount_device_start(mock, expires_in, interval);
    mock.on(move |request| request.url == TOKEN_URL)
        .respond_sequence(token_responses);
}

/// The one-request-per-connection GET the browser sends to the flow's fixed
/// loopback port, returning the raw response.
pub async fn send_callback(request: impl AsRef<[u8]>) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", CALLBACK_PORT))
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

/// The valid-code callback GET the browser sends after authorizing, carrying
/// the authorize URL's `state` back.
#[must_use]
pub fn code_callback_get(state: &str) -> String {
    format!("GET /oauth/callback?code=C&state={state} HTTP/1.1\r\nHost: x\r\n\r\n")
}
