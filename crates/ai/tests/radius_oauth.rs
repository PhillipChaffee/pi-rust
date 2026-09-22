//! The Radius gateway OAuth flow, from `test/radius-oauth.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the seam mock replaces the
//! stubbed `globalThis.fetch`, and the paused clock pins `Date.now()` for
//! the device login's expiry arithmetic.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use pi_ai::auth::types::AuthEvent;
use pi_ai::http::{MockHttpClient, json_response};
use tokio_util::sync::CancellationToken;

use common::auth_fixtures::oauth_credentials;
use common::radius_fixtures::{
    DEVICE_URL, DISCOVERY_URL, GATEWAY, START, TOKEN_URL, interaction, mount_device_routes,
    radius_oauth,
};
use common::seam_forms::{form_field, form_fields};

/// `urn:ietf:params:oauth:grant-type:device_code`, the device grant the
/// exchange posts.
const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";

#[tokio::test(start_paused = true)]
async fn uses_gateway_endpoints_directly_for_device_login() {
    let mock = MockHttpClient::new();
    mount_device_routes(
        &mock,
        600,
        5,
        vec![json_response(
            200,
            &serde_json::json!({
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "scope": "gateway offline_access",
            }),
        )],
    );

    let oauth = radius_oauth(&mock, START);
    let (interaction, scripted) = interaction("device-code");
    let credential = oauth
        .login(interaction)
        .await
        .expect("device login completes");

    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    assert_eq!(
        credential.expires,
        START + 3_600_000 - 60_000,
        "the expiry carries the one-minute skew off the pinned clock"
    );
    assert_eq!(
        credential
            .extra
            .get("scope")
            .and_then(serde_json::Value::as_str),
        Some("gateway offline_access")
    );

    assert_eq!(
        scripted.events(),
        vec![AuthEvent::DeviceCode {
            user_code: "ABCD-1234".to_owned(),
            verification_uri: "https://radius-ui.example/pair".to_owned(),
            interval_seconds: Some(5),
            expires_in_seconds: Some(600),
        }],
    );

    let served = mock.recorded();
    assert_eq!(
        served
            .iter()
            .map(|request| request.url.as_str())
            .collect::<Vec<_>>(),
        vec![DEVICE_URL, TOKEN_URL],
    );
    let fields = form_fields(&served[0]);
    assert_eq!(form_field(&fields, "client_id"), "pi-gateway");
    assert_eq!(form_field(&fields, "scope"), "gateway offline_access");
    let fields = form_fields(&served[1]);
    assert_eq!(form_field(&fields, "grant_type"), DEVICE_GRANT);
    assert_eq!(form_field(&fields, "client_id"), "pi-gateway");
    assert_eq!(form_field(&fields, "device_code"), "device-code");
}

#[tokio::test]
async fn refreshes_directly_through_the_gateway_without_discovery() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }),
        ));

    let oauth = radius_oauth(&mock, 0);
    let refreshed = oauth
        .refresh(
            oauth_credentials("old-access", "old-refresh", 0),
            CancellationToken::new(),
        )
        .await
        .expect("refresh succeeds");

    assert_eq!(refreshed.access, "new-access");
    assert_eq!(refreshed.refresh, "new-refresh");
    assert_eq!(mock.request_count(), 1, "refresh skips discovery");

    let served = mock.recorded();
    let fields = form_fields(&served[0]);
    assert_eq!(form_field(&fields, "grant_type"), "refresh_token");
    assert_eq!(form_field(&fields, "client_id"), "pi-gateway");
    assert_eq!(form_field(&fields, "refresh_token"), "old-refresh");
}

#[tokio::test]
async fn discovers_only_the_interactive_browser_authorization_endpoint() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DISCOVERY_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "issuer": "https://radius-ui.example" }),
        ));

    let oauth = radius_oauth(&mock, 0);
    let (interaction, _scripted) = interaction("browser");
    let error = oauth
        .login(interaction)
        .await
        .expect_err("discovery without an authorization endpoint rejects");

    assert_eq!(
        error.to_string(),
        format!("Invalid Radius OAuth config from {GATEWAY}")
    );
    assert_eq!(mock.request_count(), 1);
}
