//! The `OAuthAuth` adapter suite, ported from the first
//! `describe("OAuthAuth adapters")` block of
//! `packages/ai/test/oauth-auth.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (this repo's upstream pin).
//!
//! Two upstream cases do not port and are skipped here:
//!
//! - The extension-barrel case (`src/oauth.ts` must not re-export built-in
//!   flow implementations) — the port has no extension barrel yet; the
//!   statically-linked flows live in [`pi_ai::auth::oauth`] and nothing
//!   re-exports them.
//! - The second describe, "OAuth through Models.getAuth" — the `Models`
//!   runtime and the provider registry land with their own ticket; the
//!   resolve-side substance it covers is pinned natively in
//!   `auth_core.rs` over [`pi_ai::auth::resolve_provider_auth`].
//!
//! Upstream stubs `globalThis.fetch` per case; the port routes the flows'
//! [`HttpClient`] seam through [`MockHttpClient`].

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected shape panics by design"
)]

mod common;

use common::auth_fixtures::{RecordingInteraction, oauth_credentials};

use std::sync::Arc;
use std::time::Duration;

use pi_ai::auth::clock::{FixedClock, SteppedClock};
use pi_ai::auth::oauth::anthropic::AnthropicOAuth;
use pi_ai::auth::oauth::github_copilot::GitHubCopilotOAuth;
use pi_ai::auth::oauth::kimi_coding::KimiCodingOAuth;
use pi_ai::auth::oauth::{
    RadiusOAuthOptions, load_anthropic_oauth, load_github_copilot_oauth, load_kimi_coding_oauth,
    load_openai_codex_oauth, load_openrouter_oauth, load_radius_oauth, load_xai_oauth,
};
use pi_ai::auth::oauth::openai_codex::OpenAICodexOAuth;
use pi_ai::auth::oauth::openrouter::OpenRouterOAuth;
use pi_ai::auth::oauth::xai::XaiOAuth;
use pi_ai::auth::types::{AuthError, OAuthCredentials};
use pi_ai::http::{HttpClient, MockHttpClient, MockResponse, json_response};
use pi_ai::auth::types::ModelAuth;
use tokio_util::sync::CancellationToken;

/// The epoch the stepped clocks fix, so refresh arithmetic asserts exactly.
const EPOCH_MS: i64 = 1_758_000_000_000;

/// Upstream's `Number.MAX_SAFE_INTEGER`, the expiry OpenRouter's permanent
/// key credentials carry.
const MAX_SAFE_INTEGER_MS: i64 = 9_007_199_254_740_991;

/// An empty seam mock: every request fails unless a case mounts a route,
/// upstream's `vi.stubGlobal("fetch", fetchMock)` per test.
fn mock() -> MockHttpClient {
    MockHttpClient::new()
}

/// A clock fixed at [`EPOCH_MS`], the `Date.now()` the flows read.
fn clock() -> Arc<dyn pi_ai::auth::clock::AuthClock> {
    Arc::new(SteppedClock::new(EPOCH_MS))
}

/// The never-aborted signal upstream's `neverAbortedSignal` carries.
fn never_aborted() -> CancellationToken {
    CancellationToken::new()
}

#[test]
fn identifies_only_subscription_backed_oauth_flows_as_subscriptions() {
    let client: Arc<dyn HttpClient> = Arc::new(mock());
    let anthropic = AnthropicOAuth::new(Arc::clone(&client), clock());
    let openai_codex = OpenAICodexOAuth::new(Arc::clone(&client), clock());
    let github_copilot =
        GitHubCopilotOAuth::new(Arc::clone(&client), clock(), Arc::new(|_: &str| true));
    let kimi_coding = KimiCodingOAuth::new(Arc::clone(&client));
    let xai = XaiOAuth::new(Arc::clone(&client), clock());
    for (is_subscription, flow_name) in [
        (anthropic.auth().is_subscription, "anthropic"),
        (openai_codex.auth().is_subscription, "openai-codex"),
        (github_copilot.auth().is_subscription, "github-copilot"),
        (kimi_coding.auth().is_subscription, "kimi-coding"),
        (xai.auth().is_subscription, "xai"),
    ] {
        assert_eq!(
            is_subscription,
            Some(true),
            "{flow_name}: expected a subscription-backed flow"
        );
    }
    assert_ne!(
        OpenRouterOAuth::new(Arc::clone(&client)).auth().is_subscription,
        Some(true),
        "openrouter's permanent key is not a subscription"
    );
}

#[test]
fn anthropic_to_auth_derives_the_api_key_from_the_access_token() {
    let flow = AnthropicOAuth::new(Arc::new(mock()), clock());
    let auth = flow.to_auth(&oauth_credentials("token", "r", 0));
    assert_eq!(
        auth,
        ModelAuth {
            api_key: Some("token".to_owned()),
            ..ModelAuth::default()
        }
    );
}

#[test]
fn openai_codex_to_auth_derives_the_api_key_from_the_access_token() {
    let flow = OpenAICodexOAuth::new(Arc::new(mock()), clock());
    let auth = flow.to_auth(&oauth_credentials("token", "r", 0));
    assert_eq!(
        auth,
        ModelAuth {
            api_key: Some("token".to_owned()),
            ..ModelAuth::default()
        }
    );
}

#[tokio::test]
async fn openrouter_keeps_the_permanent_credential_on_refresh() {
    let flow = OpenRouterOAuth::new(Arc::new(mock()));
    let credential = oauth_credentials("token", "", MAX_SAFE_INTEGER_MS);
    assert_eq!(
        flow.to_auth(&credential),
        ModelAuth {
            api_key: Some("token".to_owned()),
            ..ModelAuth::default()
        }
    );
    // Upstream asserts identity (`toBe`); the port's refresh clones, so the
    // adapter contract is value equality with the credential unchanged.
    let refreshed = flow
        .refresh(credential.clone(), never_aborted())
        .await
        .expect("openrouter refresh resolves");
    assert_eq!(refreshed, credential);
}

#[test]
fn xai_to_auth_derives_the_api_key_from_the_access_token() {
    let flow = XaiOAuth::new(Arc::new(mock()), clock());
    let auth = flow.to_auth(&oauth_credentials("token", "r", 0));
    assert_eq!(
        auth,
        ModelAuth {
            api_key: Some("token".to_owned()),
            ..ModelAuth::default()
        }
    );
}

#[test]
fn github_copilot_to_auth_derives_base_url_from_the_token_proxy_endpoint() {
    let flow = GitHubCopilotOAuth::new(Arc::new(mock()), clock(), Arc::new(|_: &str| true));
    let access = "tid=abc;exp=123;proxy-ep=proxy.enterprise.example;rest";
    let auth = flow.to_auth(&oauth_credentials(access, "r", 0));
    assert_eq!(
        auth,
        ModelAuth {
            api_key: Some(access.to_owned()),
            headers: None,
            base_url: Some("https://api.enterprise.example".to_owned()),
        }
    );
}

#[test]
fn github_copilot_to_auth_falls_back_to_the_enterprise_domain_then_the_individual_endpoint() {
    let flow = GitHubCopilotOAuth::new(Arc::new(mock()), clock(), Arc::new(|_: &str| true));
    let mut enterprise = oauth_credentials("no-proxy-ep", "r", 0);
    enterprise.extra.insert(
        "enterpriseUrl".to_owned(),
        serde_json::Value::String("https://company.ghe.com".to_owned()),
    );
    assert_eq!(
        flow.to_auth(&enterprise).base_url.as_deref(),
        Some("https://copilot-api.company.ghe.com")
    );
    assert_eq!(
        flow.to_auth(&oauth_credentials("no-proxy-ep", "r", 0))
            .base_url
            .as_deref(),
        Some("https://api.individual.githubcopilot.com")
    );
}

#[tokio::test]
async fn anthropic_refresh_exchanges_the_refresh_token_and_returns_a_typed_credential() {
    let client = mock();
    client
        .on(|request| request.url == "https://platform.claude.com/v1/oauth/token")
        .respond(json_response(
            200,
            &serde_json::json!({
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }),
        ));
    let flow = AnthropicOAuth::new(Arc::new(client), clock());

    let refreshed = flow
        .refresh(oauth_credentials("old", "old-r", 0), never_aborted())
        .await
        .expect("refresh resolves");
    let now = clock().now_ms();
    assert_eq!(refreshed.access, "new-access");
    assert_eq!(refreshed.refresh, "new-refresh");
    assert!(
        refreshed.expires > now,
        "the refreshed credential outlives the clock: {} > {now}",
        refreshed.expires
    );
}

#[tokio::test]
async fn github_copilot_refresh_preserves_the_enterprise_domain() {
    let client = mock();
    client
        .on(|request| request.url == "https://api.company.ghe.com/copilot_internal/v2/token")
        .respond(json_response(
            200,
            &serde_json::json!({ "token": "new-token", "expires_at": 9_999_999_999_u64 }),
        ));
    client
        .on(|request| request.url.ends_with("/models"))
        .respond(json_response(200, &serde_json::json!({ "data": [] })));
    let flow = GitHubCopilotOAuth::new(Arc::new(client.clone()), clock(), Arc::new(|_: &str| true));

    let mut credential = oauth_credentials("old", "gh-token", 0);
    credential.extra.insert(
        "enterpriseUrl".to_owned(),
        serde_json::Value::String("company.ghe.com".to_owned()),
    );
    let refreshed = flow
        .refresh(credential, never_aborted())
        .await
        .expect("refresh resolves");

    assert_eq!(refreshed.access, "new-token");
    assert_eq!(refreshed.extra.get("enterpriseUrl").and_then(serde_json::Value::as_str), Some("company.ghe.com"));
    let fetched_urls: Vec<String> = client
        .recorded()
        .iter()
        .map(|request| request.url.clone())
        .collect();
    assert!(
        fetched_urls[0].contains("api.company.ghe.com"),
        "the Copilot token exchange hits the enterprise endpoint: {fetched_urls:?}"
    );
}

// ---------------------------------------------------------------------------
// load.rs: the statically-linked flow constructors
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_flow_constructors_wire_the_process_defaults() {
    let anthropic = load_anthropic_oauth().await;
    assert_eq!(anthropic.name, "Anthropic (Claude Pro/Max)");
    assert_eq!(anthropic.is_subscription, Some(true));

    let openai_codex = load_openai_codex_oauth().await;
    assert_eq!(openai_codex.name, "OpenAI (ChatGPT Plus/Pro)");
    assert_eq!(openai_codex.is_subscription, Some(true));

    let github_copilot = load_github_copilot_oauth().await;
    assert_eq!(github_copilot.is_subscription, Some(true));

    let openrouter = load_openrouter_oauth().await;
    assert_eq!(openrouter.name, "OpenRouter OAuth");
    assert_ne!(openrouter.is_subscription, Some(true));
    assert_eq!(
        openrouter.login_label,
        Some("Sign in with OpenRouter".to_owned())
    );

    let kimi = load_kimi_coding_oauth().await;
    assert_eq!(kimi.name, "Kimi Code (subscription)");
    assert_eq!(kimi.login_label, Some("Sign in with Kimi Code".to_owned()));

    let xai = load_xai_oauth().await;
    assert_eq!(xai.name, "xAI (Grok/X subscription)");
    assert_eq!(
        xai.login_label,
        Some("Sign in with SuperGrok or X Premium".to_owned())
    );

    let radius = load_radius_oauth(&RadiusOAuthOptions {
        name: "Radius".to_owned(),
        gateway: "radius.example/".to_owned(),
    })
    .await;
    assert_eq!(radius.name, "Radius");
    assert_ne!(radius.is_subscription, Some(true));
}

// ---------------------------------------------------------------------------
// anthropic.rs: refresh failure branches
// ---------------------------------------------------------------------------

#[test]
fn anthropic_debug_names_the_flow() {
    let flow = AnthropicOAuth::new(Arc::new(mock()), clock());
    assert_eq!(flow.auth().name, "Anthropic (Claude Pro/Max)");
    assert_eq!(format!("{flow:?}"), "AnthropicOAuth");
}

#[tokio::test]
async fn anthropic_refresh_transport_failure_carries_the_refresh_prefix() {
    let flow = AnthropicOAuth::new(Arc::new(mock()), clock());
    let error = flow
        .refresh(oauth_credentials("a", "r", 0), never_aborted())
        .await
        .expect_err("the transport failure fails refresh");
    assert_eq!(
        error.to_string(),
        "Anthropic token refresh request failed. url=https://platform.claude.com/v1/oauth/token; \
         details=no mock route matched POST https://platform.claude.com/v1/oauth/token"
    );
}

#[tokio::test]
async fn anthropic_refresh_invalid_json_names_the_url_body_and_details() {
    let client = mock();
    client
        .on(|request| request.url == "https://platform.claude.com/v1/oauth/token")
        .respond(MockResponse::status(200).with_body("not json at all"));
    let flow = AnthropicOAuth::new(Arc::new(client), clock());
    let error = flow
        .refresh(oauth_credentials("a", "r", 0), never_aborted())
        .await
        .expect_err("the invalid JSON fails refresh");
    assert_eq!(
        error.to_string(),
        "Anthropic token refresh returned invalid JSON. \
         url=https://platform.claude.com/v1/oauth/token; body=not json at all; \
         details=expected ident at line 1 column 2"
    );
}

#[tokio::test]
async fn anthropic_refresh_rejects_a_non_object_body_and_missing_fields() {
    let client = mock();
    client
        .on(|request| request.url == "https://platform.claude.com/v1/oauth/token")
        .respond(MockResponse::status(200).with_body("[1,2]"));
    let flow = AnthropicOAuth::new(Arc::new(client), clock());
    let error = flow
        .refresh(oauth_credentials("a", "r", 0), never_aborted())
        .await
        .expect_err("the array body fails refresh");
    assert_eq!(
        error.to_string(), "Anthropic token refresh returned invalid JSON",
        "a non-object body rejects"
    );

    let client = mock();
    client
        .on(|request| request.url == "https://platform.claude.com/v1/oauth/token")
        .respond(json_response(
            200,
            &serde_json::json!({"access_token": "a", "refresh_token": "r"}),
        ));
    let flow = AnthropicOAuth::new(Arc::new(client), clock());
    let error = flow
        .refresh(oauth_credentials("a", "r", 0), never_aborted())
        .await
        .expect_err("the missing expiry fails refresh");
    assert_eq!(
        error.to_string(),
        "Anthropic token refresh returned invalid JSON: missing expires_in"
    );
}

#[tokio::test]
async fn anthropic_refresh_surfaces_the_http_failure_status() {
    let client = mock();
    client
        .on(|request| request.url == "https://platform.claude.com/v1/oauth/token")
        .respond(MockResponse::status(400).with_body("denied"));
    let flow = AnthropicOAuth::new(Arc::new(client), clock());
    let error = flow
        .refresh(oauth_credentials("a", "r", 0), never_aborted())
        .await
        .expect_err("the failure status fails refresh");
    assert_eq!(
        error.to_string(),
        "Anthropic token refresh request failed. \
         url=https://platform.claude.com/v1/oauth/token; details=HTTP request failed. status=400; \
         url=https://platform.claude.com/v1/oauth/token; body=denied"
    );
}

// ---------------------------------------------------------------------------
// kimi_coding.rs: adapter surface
// ---------------------------------------------------------------------------

#[test]
fn kimi_debug_names_the_flow() {
    let flow = KimiCodingOAuth::new(Arc::new(mock()));
    assert_eq!(format!("{flow:?}"), "KimiCodingOAuth");
}

#[test]
fn kimi_to_auth_carries_the_bearer_header() {
    let flow = KimiCodingOAuth::new(Arc::new(mock()));
    let auth = flow.to_auth(&oauth_credentials("tok", "r", 0));
    assert_eq!(auth.api_key, None, "kimi authenticates through headers");
    let headers = auth.headers.expect("the bearer header set");
    assert_eq!(
        headers.get("Authorization").and_then(Option::as_deref),
        Some("Bearer tok")
    );
}

// ---------------------------------------------------------------------------
// xai.rs: start-response validation and poll failure branches
// ---------------------------------------------------------------------------

const XAI_DEVICE_URL: &str = "https://auth.x.ai/oauth2/device/code";
const XAI_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";

/// The device-start response every poll case builds from.
fn xai_device_response() -> serde_json::Value {
    serde_json::json!({
        "device_code": "device-code",
        "user_code": "ABCD-1234",
        "verification_uri": "https://auth.x.ai/device",
        "interval": 5,
        "expires_in": 600,
    })
}

/// A login task over a fresh interaction bundle.
fn xai_login(
    flow: XaiOAuth,
) -> (
    tokio::task::JoinHandle<Result<OAuthCredentials, AuthError>>,
    Arc<RecordingInteraction>,
) {
    let recording: Arc<RecordingInteraction> = Arc::new(RecordingInteraction::new());
    let interaction =
        pi_ai::auth::types::ProviderAuthInteraction::from_interaction(
            recording.interaction(),
            never_aborted(),
        );
    let handle = tokio::spawn(async move { flow.login(interaction).await });
    (handle, recording)
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

#[test]
fn xai_debug_names_the_flow() {
    let flow = XaiOAuth::new(Arc::new(mock()), clock());
    assert_eq!(format!("{flow:?}"), "XaiOAuth");
    assert_eq!(flow.auth().name, "xAI (Grok/X subscription)");
    assert_eq!(
        flow.auth().login_label,
        Some("Sign in with SuperGrok or X Premium".to_owned())
    );
}

#[tokio::test]
async fn xai_login_with_a_cancelled_signal_reports_login_cancelled() {
    let flow = XaiOAuth::new(Arc::new(mock()), clock());
    let signal = CancellationToken::new();
    signal.cancel();
    let interaction = pi_ai::auth::types::ProviderAuthInteraction::from_interaction(
        RecordingInteraction::new().interaction(),
        signal.clone(),
    );
    let error = flow
        .login(interaction)
        .await
        .expect_err("the cancelled login rejects");
    assert_eq!(error.to_string(), "Login cancelled");
}

#[tokio::test]
async fn xai_device_start_failures_carry_the_status_and_error_detail() {
    let client = mock();
    client
        .on(|request| request.url == XAI_DEVICE_URL)
        .respond(json_response(
            400,
            &serde_json::json!({"error": "invalid_client", "error_description": "nope"}),
        ));
    let flow = XaiOAuth::new(Arc::new(client), clock());
    let error = xai_login(flow)
        .0
        .await
        .expect("the login task joins")
        .expect_err("the failed start fails login");
    assert_eq!(
        error.to_string(),
        "xAI OAuth device authorization failed (HTTP 400): invalid_client: nope"
    );

    // Without error fields the message stops at the status.
    let client = mock();
    client
        .on(|request| request.url == XAI_DEVICE_URL)
        .respond(MockResponse::status(400));
    let flow = XaiOAuth::new(Arc::new(client), clock());
    let error = xai_login(flow)
        .0
        .await
        .expect("the login task joins")
        .expect_err("the failed start fails login");
    assert_eq!(
        error.to_string(), "xAI OAuth device authorization failed (HTTP 400)",
        "the detail is empty"
    );
}

#[tokio::test]
async fn xai_device_start_rejects_malformed_fields() {
    let client = mock();
    client
        .on(|request| request.url == XAI_DEVICE_URL)
        .respond(MockResponse::status(200).with_body("[1,2]"));
    let flow = XaiOAuth::new(Arc::new(client), clock());
    let error = xai_login(flow)
        .0
        .await
        .expect("the login task joins")
        .expect_err("the non-object body fails login");
    assert_eq!(
        error.to_string(), "Invalid xAI OAuth response field: device_code",
        "a non-object body leaves every field missing"
    );

    // A non-positive expires_in rejects with the field's name.
    let client = mock();
    let mut body = xai_device_response();
    body["expires_in"] = serde_json::json!(0);
    client
        .on(|request| request.url == XAI_DEVICE_URL)
        .respond(json_response(200, &body));
    let flow = XaiOAuth::new(Arc::new(client), clock());
    let error = xai_login(flow)
        .0
        .await
        .expect("the login task joins")
        .expect_err("the non-positive expiry fails login");
    assert_eq!(error.to_string(), "Invalid xAI OAuth response field: expires_in");

    // A missing user_code rejects with its field's name.
    let client = mock();
    let mut body = xai_device_response();
    body["user_code"] = serde_json::json!("");
    client
        .on(|request| request.url == XAI_DEVICE_URL)
        .respond(json_response(200, &body));
    let flow = XaiOAuth::new(Arc::new(client), clock());
    let error = xai_login(flow)
        .0
        .await
        .expect("the login task joins")
        .expect_err("the empty user code fails login");
    assert_eq!(error.to_string(), "Invalid xAI OAuth response field: user_code");
}

#[tokio::test]
async fn xai_device_start_rejects_untrusted_verification_uris() {
    for field in ["verification_uri", "verification_uri_complete"] {
        let client = mock();
        let mut body = xai_device_response();
        body[field] = serde_json::json!("http://auth.x.ai/device");
        client
            .on(|request| request.url == XAI_DEVICE_URL)
            .respond(json_response(200, &body));
        let flow = XaiOAuth::new(Arc::new(client), clock());
        let error = xai_login(flow)
            .0
            .await
            .expect("the login task joins")
            .expect_err("the untrusted uri fails login");
        assert_eq!(
            error.to_string(), "Untrusted verification URI in xAI OAuth response",
            "{field} must be https: {error:?}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn xai_poll_failures_reject_with_the_wire_message() {
    for (error_body, expected) in [
        (
            serde_json::json!({"error": "access_denied"}),
            "xAI device authorization was denied".to_owned(),
        ),
        (
            serde_json::json!({"error": "expired_token"}),
            "xAI device code expired".to_owned(),
        ),
        (
            serde_json::json!({"error": "invalid_grant", "error_description": "nope"}),
            "xAI OAuth device token polling failed (HTTP 400): invalid_grant: nope".to_owned(),
        ),
    ] {
        let client = mock();
        client
            .on(|request| request.url == XAI_DEVICE_URL)
            .respond(json_response(200, &xai_device_response()));
        client
            .on(|request| request.url == XAI_TOKEN_URL)
            .respond(json_response(400, &error_body));
        let flow = XaiOAuth::new(Arc::new(client), clock());
        let (handle, _recording) = xai_login(flow);
        // waitBeforeFirstPoll: the first poll lands after the server interval.
        tokio::time::advance(Duration::from_secs(5)).await;
        advance_until(
            || handle.is_finished(),
            "the failed poll to finish the login",
        )
        .await;
        let error = handle
            .await
            .expect("the login task joins")
            .expect_err("the failed poll fails login");
        assert_eq!(error.to_string(), expected, "{error_body:?}");
    }
}

#[tokio::test(start_paused = true)]
async fn xai_slow_down_uses_the_server_interval_and_completes() {
    let client = mock();
    client
        .on(|request| request.url == XAI_DEVICE_URL)
        .respond(json_response(200, &xai_device_response()));
    client
        .on(|request| request.url == XAI_TOKEN_URL)
        .respond_sequence(vec![
            json_response(400, &serde_json::json!({"error": "authorization_pending"})),
            json_response(
                400,
                &serde_json::json!({"error": "slow_down", "interval": 7}),
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
    let clock: Arc<dyn pi_ai::auth::clock::AuthClock> = Arc::new(FixedClock::new(0));
    let flow = XaiOAuth::new(Arc::new(client.clone()), Arc::clone(&clock));
    let (handle, _recording) = xai_login(flow);

    tokio::time::advance(Duration::from_secs(5)).await;
    advance_until(|| client.request_count() >= 2, "the first poll").await;
    tokio::time::advance(Duration::from_secs(7)).await;
    advance_until(|| client.request_count() >= 3, "the slow-down poll").await;
    tokio::time::advance(Duration::from_secs(7)).await;

    let credential = handle
        .await
        .expect("the login task joins")
        .expect("the slow-down flow completes");
    assert_eq!(credential.access, "access-token");
    assert_eq!(client.request_count(), 4, "device start, then three polls");
}

#[tokio::test(start_paused = true)]
async fn xai_poll_rejects_a_token_response_without_a_refresh_token() {
    let client = mock();
    client
        .on(|request| request.url == XAI_DEVICE_URL)
        .respond(json_response(200, &xai_device_response()));
    client
        .on(|request| request.url == XAI_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({"access_token": "a"}),
        ));
    let clock: Arc<dyn pi_ai::auth::clock::AuthClock> = Arc::new(FixedClock::new(0));
    let flow = XaiOAuth::new(Arc::new(client), Arc::clone(&clock));
    let (handle, _recording) = xai_login(flow);

    tokio::time::advance(Duration::from_secs(5)).await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the missing refresh token fails login");
    assert_eq!(
        error.to_string(), "Invalid xAI OAuth response field: refresh_token",
        "a first poll without rotation carries no previous token to reuse"
    );
}

#[tokio::test]
async fn xai_refresh_preserves_the_previous_refresh_token_when_the_wire_omits_it() {
    let client = mock();
    client
        .on(|request| request.url == XAI_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({"access_token": "a2"}),
        ));
    let flow = XaiOAuth::new(Arc::new(client), clock());
    let refreshed = flow
        .refresh(oauth_credentials("a1", "prev-r", 0), never_aborted())
        .await
        .expect("refresh resolves");
    assert_eq!(refreshed.access, "a2");
    assert_eq!(
        refreshed.refresh, "prev-r",
        "xAI may omit refresh_token when the token is not rotated"
    );
}

#[tokio::test]
async fn xai_refresh_transport_failures_propagate_the_seam_message() {
    let flow = XaiOAuth::new(Arc::new(mock()), clock());
    let error = flow
        .refresh(oauth_credentials("a", "r", 0), never_aborted())
        .await
        .expect_err("the transport failure fails refresh");
    assert_eq!(
        error.to_string(),
        "no mock route matched POST https://auth.x.ai/oauth2/token"
    );
}

#[tokio::test(start_paused = true)]
async fn xai_poll_transport_failures_propagate_the_seam_message() {
    let client = mock();
    client
        .on(|request| request.url == XAI_DEVICE_URL)
        .respond(json_response(200, &xai_device_response()));
    let clock: Arc<dyn pi_ai::auth::clock::AuthClock> = Arc::new(FixedClock::new(0));
    let flow = XaiOAuth::new(Arc::new(client), Arc::clone(&clock));
    let (handle, _recording) = xai_login(flow);
    // waitBeforeFirstPoll parks the flow before the first token request.
    tokio::time::advance(Duration::from_secs(5)).await;
    advance_until(|| handle.is_finished(), "the transport failure").await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the unmatched token route fails login");
    assert_eq!(
        error.to_string(),
        "no mock route matched POST https://auth.x.ai/oauth2/token"
    );
}

#[tokio::test]
async fn xai_device_start_rejects_a_missing_verification_uri() {
    let client = mock();
    let mut body = xai_device_response();
    body["verification_uri"] = serde_json::json!(null);
    client
        .on(|request| request.url == XAI_DEVICE_URL)
        .respond(json_response(200, &body));
    let flow = XaiOAuth::new(Arc::new(client), clock());
    let error = xai_login(flow)
        .0
        .await
        .expect("the login task joins")
        .expect_err("the missing verification uri fails login");
    assert_eq!(
        error.to_string(),
        "Invalid xAI OAuth response field: verification_uri"
    );
}

#[tokio::test(start_paused = true)]
async fn xai_poll_rejects_an_empty_or_invalid_refresh_token_and_expiry() {
    for (body, expected) in [
        (
            serde_json::json!({"access_token": "a", "refresh_token": ""}),
            "Invalid xAI OAuth response field: refresh_token",
        ),
        (
            serde_json::json!({
                "access_token": "a",
                "refresh_token": "r",
                "expires_in": 0,
            }),
            "Invalid xAI OAuth response field: expires_in",
        ),
    ] {
        let client = mock();
        client
            .on(|request| request.url == XAI_DEVICE_URL)
            .respond(json_response(200, &xai_device_response()));
        client
            .on(|request| request.url == XAI_TOKEN_URL)
            .respond(json_response(200, &body));
        let clock: Arc<dyn pi_ai::auth::clock::AuthClock> = Arc::new(FixedClock::new(0));
        let flow = XaiOAuth::new(Arc::new(client), Arc::clone(&clock));
        let (handle, _recording) = xai_login(flow);
        tokio::time::advance(Duration::from_secs(5)).await;
        let error = handle
            .await
            .expect("the login task joins")
            .expect_err("the malformed token response fails login");
        assert_eq!(error.to_string(), expected, "{body:?}");
    }
}
