//! The GitHub Copilot OAuth flow, from `test/github-copilot-oauth.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the seam mock replaces
//! the stubbed `globalThis.fetch`, the paused clock with a
//! [`pi_ai::auth::clock::FixedClock`] replaces fake timers, and the scripted
//! interaction replaces the prompt/notify literals.

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

use common::auth_fixtures::oauth_credentials;
use common::auth_interaction::{ScriptedAuthInteraction, provider_interaction};
use common::seam_forms::{form_field, form_fields};
use pi_ai::auth::clock::SteppedClock;
use pi_ai::auth::oauth::github_copilot::{
    GitHubCopilotOAuth, KnownModels, parse_github_copilot_model_catalog,
};
use pi_ai::auth::oauth::load_github_copilot_oauth;
use pi_ai::auth::types::{AuthEvent, ModelAuth};
use pi_ai::http::{MockHttpClient, MockResponse, json_response};
use tokio_util::sync::CancellationToken;

const DEVICE_URL: &str = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const MODELS_URL: &str = "https://api.individual.githubcopilot.com/models";
const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const COPILOT_TOKEN: &str = "tid=abc;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com";

/// `new Date("2026-07-24T00:00:00Z").getTime()`, the epoch the flow's expiry
/// arithmetic pins against.
const START: i64 = 1_784_000_000_000;

/// The known-model membership the policy updates check, upstream's generated
/// `GITHUB_COPILOT_MODELS` catalog narrowed to the ids the cases use.
fn known_models() -> KnownModels {
    Arc::new(|model_id: &str| matches!(model_id, "claude-sonnet-4.6" | "gpt-5.4"))
}

/// The flow over the mock and a stepped clock (the epoch stays put while
/// real time passes, so expiry arithmetic asserts exactly).
fn flow(mock: &MockHttpClient) -> GitHubCopilotOAuth {
    GitHubCopilotOAuth::new(
        Arc::new(mock.clone()),
        Arc::new(SteppedClock::new(START)),
        known_models(),
    )
}

/// The device authorization response a login case starts from.
fn device_response() -> serde_json::Value {
    serde_json::json!({
        "device_code": "device-code",
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "interval": 1,
        "expires_in": 600,
    })
}

/// The Copilot internal token response, upstream's `expires_at` shape.
fn copilot_token_response() -> serde_json::Value {
    serde_json::json!({
        "token": COPILOT_TOKEN,
        "expires_at": 9_999_999_999_u64,
    })
}

/// One model entry with the given picker flag and policy state.
fn model_entry(id: &str, picker_enabled: bool, policy_state: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "model_picker_enabled": picker_enabled,
        "policy": { "state": policy_state },
        "capabilities": { "supports": { "tool_calls": true } },
    })
}

/// Mount the routes a device login runs: start, poll, Copilot token, and
/// `models` as the models GET's answer.
fn mount_login_routes(mock: &MockHttpClient, models: &serde_json::Value) {
    mount_login_route_builder(mock, |builder| builder.respond(json_response(200, models)));
}

/// Mount the device-login routes with a custom models responder, for the
/// rate-limit sequences.
fn mount_login_route_builder(
    mock: &MockHttpClient,
    mount_models: impl FnOnce(pi_ai::http::mock::MockRouteBuilder<'_>),
) {
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "access_token": "gh-token", "token_type": "bearer" }),
        ));
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    mount_models(mock.on(move |request| request.url == MODELS_URL));
}

#[tokio::test(start_paused = true)]
async fn login_completes_through_the_device_flow_and_merges_the_catalog() {
    let mock = MockHttpClient::new();
    mount_login_routes(&mock, &serde_json::json!({ "data": [] }));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    // waitBeforeFirstPoll: the first poll lands after the server interval.
    tokio::time::advance(Duration::from_secs(1)).await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");

    let AuthEvent::DeviceCode {
        user_code,
        verification_uri,
        interval_seconds,
        expires_in_seconds,
    } = &scripted.events()[0]
    else {
        panic!("the login announces the device code");
    };
    assert_eq!(user_code, "ABCD-1234");
    assert_eq!(verification_uri, "https://github.com/login/device");
    assert_eq!(*interval_seconds, Some(1));
    assert_eq!(*expires_in_seconds, Some(600));

    let served = mock.recorded();
    let device_fields = form_fields(&served[0]);
    assert_eq!(form_field(&device_fields, "client_id"), CLIENT_ID);
    assert_eq!(form_field(&device_fields, "scope"), "read:user");

    // The Copilot token exchange rides the internal endpoint with the
    // wire's client headers.
    let copilot_request = served
        .iter()
        .find(|request| request.url == COPILOT_TOKEN_URL)
        .expect("the Copilot token exchange ran");
    let headers = &copilot_request.headers;
    assert!(
        headers
            .iter()
            .any(|(name, value)| name == "Authorization" && value == "Bearer gh-token"),
        "the exchange carries the bearer token: {:?}",
        copilot_request.headers
    );
    assert!(
        headers
            .iter()
            .any(|(name, value)| name == "Copilot-Integration-Id" && value == "vscode-chat"),
        "the exchange carries the Copilot integration headers"
    );

    assert_eq!(credential.access, COPILOT_TOKEN);
    assert_eq!(
        credential.extra.get("availableModelIds"),
        Some(&serde_json::json!([])),
        "an empty picker list stores no model ids"
    );
}

#[tokio::test(start_paused = true)]
async fn login_enables_unconfigured_policy_models_and_merges_the_enabled_ids() {
    let mock = MockHttpClient::new();
    mount_login_routes(
        &mock,
        &serde_json::json!({ "data": [
            model_entry("claude-sonnet-4.6", false, "unconfigured"),
            model_entry("unknown-model", false, "unconfigured"),
        ]}),
    );
    mock.on(|request| {
        request.url == "https://api.individual.githubcopilot.com/models/claude-sonnet-4.6/policy"
    })
    .respond(json_response(
        200,
        &serde_json::json!({ "state": "enabled" }),
    ));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    tokio::time::advance(Duration::from_secs(1)).await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");

    assert_eq!(
        credential.extra.get("availableModelIds"),
        Some(&serde_json::json!(["claude-sonnet-4.6"])),
        "only the catalog-known unconfigured model enters the policy batch"
    );
    // The policy POST carried the chat-policy intent headers.
    let policy_request = mock
        .recorded()
        .into_iter()
        .find(|request| request.url.ends_with("/claude-sonnet-4.6/policy"))
        .expect("the policy request ran");
    assert!(
        policy_request
            .headers
            .iter()
            .any(|(name, value)| name == "openai-intent" && value == "chat-policy"),
        "the policy request carries the chat-policy intent: {:?}",
        policy_request.headers
    );
    assert!(
        scripted
            .events()
            .iter()
            .any(|event| matches!(event, AuthEvent::Progress { message } if message == "Enabling models...")),
        "the flow announces the enablement step: {:?}",
        scripted.events()
    );
}

#[tokio::test]
async fn login_rejects_an_invalid_enterprise_domain_before_prompting_the_device() {
    let mock = MockHttpClient::new();
    mount_login_routes(&mock, &serde_json::json!({ "data": [] }));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering("ht tp://bad"));
    let error = oauth
        .login(provider_interaction(&scripted, CancellationToken::new()))
        .await
        .expect_err("the invalid domain fails login");
    assert_eq!(error.to_string(), "Invalid GitHub Enterprise URL/domain");
    assert_eq!(mock.request_count(), 0, "no wire call ran");
}

#[tokio::test(start_paused = true)]
async fn login_routes_the_exchange_through_the_enterprise_domain() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == "https://company.ghe.com/login/device/code")
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == "https://company.ghe.com/login/oauth/access_token")
        .respond(json_response(
            200,
            &serde_json::json!({ "access_token": "gh-token" }),
        ));
    mock.on(|request| request.url == "https://api.company.ghe.com/copilot_internal/v2/token")
        .respond(json_response(200, &copilot_token_response()));
    // The exchanged token's proxy-ep endpoint wins the models fetch even
    // through the enterprise login, upstream's precedence.
    mock.on(|request| request.url == "https://api.individual.githubcopilot.com/models")
        .respond(json_response(200, &serde_json::json!({ "data": [] })));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering("company.ghe.com"));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    tokio::time::advance(Duration::from_secs(1)).await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(
        credential.extra.get("enterpriseUrl"),
        Some(&serde_json::json!("company.ghe.com")),
        "the enterprise domain rides the credential"
    );
    let served: Vec<String> = mock
        .recorded()
        .into_iter()
        .map(|request| request.url)
        .collect();
    assert!(
        served
            .iter()
            .any(|url| url == "https://company.ghe.com/login/device/code"),
        "the device flow runs against the enterprise host: {served:?}"
    );
}

#[tokio::test]
async fn login_rejects_an_untrusted_verification_uri() {
    let mock = MockHttpClient::new();
    let mut body = device_response();
    body["verification_uri"] = serde_json::json!("file:///etc/passwd");
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &body));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let error = oauth
        .login(provider_interaction(&scripted, CancellationToken::new()))
        .await
        .expect_err("the untrusted URI fails login");
    assert_eq!(
        error.to_string(),
        "Untrusted verification_uri in device code response"
    );
}

#[tokio::test]
async fn login_rejects_incomplete_device_responses() {
    for body in [
        serde_json::json!({ "user_code": "ABCD" }),
        serde_json::json!([1, 2]),
        serde_json::json!("nope"),
    ] {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url == DEVICE_URL)
            .respond(json_response(200, &body));
        let oauth = flow(&mock);
        let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
        let outcome = oauth
            .login(provider_interaction(&scripted, CancellationToken::new()))
            .await;
        let error = outcome.expect_err("the malformed start fails login");
        assert!(
            error
                .to_string()
                .starts_with("Invalid device code response"),
            "the start failure names the shape: {error}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn the_device_poll_reports_the_wire_error_and_description() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({
                "error": "expired_token",
                "error_description": "restart",
            }),
        ));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    tokio::time::advance(Duration::from_secs(1)).await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the failed poll fails login");
    assert_eq!(
        error.to_string(),
        "Device flow failed: expired_token: restart"
    );
}

#[tokio::test(start_paused = true)]
async fn the_device_poll_honours_the_server_slow_down_interval() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond_sequence(vec![
            json_response(
                200,
                &serde_json::json!({ "error": "slow_down", "interval": 4 }),
            ),
            json_response(200, &serde_json::json!({ "access_token": "gh-token" })),
        ]);
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    mock.on(|request| request.url == MODELS_URL)
        .respond(json_response(200, &serde_json::json!({ "data": [] })));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::time::advance(Duration::from_secs(4)).await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, COPILOT_TOKEN);
    assert_eq!(
        mock.request_count(),
        5,
        "start, slow poll, poll, exchange, models"
    );
}

#[tokio::test]
async fn a_pre_cancelled_login_fails_before_any_wire_call() {
    let mock = MockHttpClient::new();
    mount_login_routes(&mock, &serde_json::json!({ "data": [] }));
    let signal = CancellationToken::new();
    signal.cancel();

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let error = oauth
        .login(provider_interaction(&scripted, signal))
        .await
        .expect_err("the cancelled login rejects");
    assert_eq!(error.to_string(), "Login cancelled");
    assert_eq!(mock.request_count(), 0, "no wire call ran");
}

#[tokio::test]
async fn refresh_exchanges_the_github_token_again_and_keeps_the_enterprise_domain() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == "https://api.company.ghe.com/copilot_internal/v2/token")
        .respond(json_response(200, &copilot_token_response()));
    // The token's proxy-ep endpoint outranks the enterprise domain for the
    // models fetch, upstream's getGitHubCopilotBaseUrl precedence.
    mock.on(|request| request.url == "https://api.individual.githubcopilot.com/models")
        .respond(json_response(200, &serde_json::json!({ "data": [] })));

    let oauth = flow(&mock);
    let mut stored = oauth_credentials("old-access", "gh-refresh", 0);
    stored.extra.insert(
        "enterpriseUrl".to_owned(),
        serde_json::Value::String("company.ghe.com".to_owned()),
    );
    let refreshed = oauth
        .refresh(stored, CancellationToken::new())
        .await
        .expect("refresh resolves");

    assert_eq!(refreshed.access, COPILOT_TOKEN);
    assert_eq!(refreshed.refresh, "gh-refresh");
    assert_eq!(
        refreshed.extra.get("enterpriseUrl"),
        Some(&serde_json::json!("company.ghe.com")),
        "the enterprise domain survives the rotation"
    );
    assert_eq!(mock.request_count(), 2, "token exchange, then models");
    let served: Vec<String> = mock
        .recorded()
        .into_iter()
        .map(|request| request.url)
        .collect();
    assert!(
        served.iter().all(|url| !url.contains("github.com/login")),
        "refresh skips the device flow: {served:?}"
    );
    assert!(
        served
            .iter()
            .any(|url| url == "https://api.individual.githubcopilot.com/models"),
        "the token's proxy endpoint wins the models fetch: {served:?}"
    );
}

#[tokio::test]
async fn refresh_rejects_incomplete_copilot_token_responses() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &serde_json::json!({ "token": "t" })));

    let oauth = flow(&mock);
    let error = oauth
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the missing expiry fails refresh");
    assert_eq!(error.to_string(), "Invalid Copilot token response fields");
}

#[tokio::test]
async fn refresh_surfaces_the_models_failure_status() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    mock.on(|request| request.url.ends_with("/models"))
        .respond(MockResponse::status(403).with_body("forbidden"));

    let oauth = flow(&mock);
    let error = oauth
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the models failure fails refresh");
    assert_eq!(error.to_string(), "403 Forbidden: forbidden");
}

// ---------------------------------------------------------------------------
// The rate-limit retry: seconds, HTTP dates, and the retry budget
// ---------------------------------------------------------------------------

/// A models GET whose replies the caller queues; each poll time lands in the
/// shared sink so the paused-clock cases assert the backoff exactly.
#[tokio::test(start_paused = true)]
async fn the_login_models_fetch_retries_a_429_with_the_seconds_delay() {
    let mock = MockHttpClient::new();
    // The login models route carries the rate-limited pair; the login path's
    // models fetch carries the two-retry budget.
    mount_login_route_builder(&mock, |builder| {
        builder.respond_sequence(vec![
            MockResponse::status(429).with_header("Retry-After", "1"),
            json_response(200, &serde_json::json!({ "data": [] })),
        ]);
    });

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));
    // The wait-before-first-poll sleep and the one-second retry-after backoff
    // both ride the paused clock.
    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::task::yield_now().await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("the retried login resolves");
    assert_eq!(
        credential.extra.get("availableModelIds"),
        Some(&serde_json::json!([])),
        "the retried models fetch succeeds"
    );
    assert_eq!(
        mock.request_count(),
        5,
        "device start, poll, token exchange, 429, retried models"
    );
}

#[tokio::test(start_paused = true)]
async fn a_429_retry_after_http_date_waits_out_the_date_delta() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    // One hour after START, in IMF-fixdate form.
    let retry_after = "Tue, 14 Jul 2026 04:33:20 GMT";
    mock.on(|request| request.url.ends_with("/models"))
        .respond_sequence(vec![
            MockResponse::status(429).with_header("Retry-After", retry_after),
            json_response(200, &serde_json::json!({ "data": [] })),
        ]);

    let oauth = flow(&mock);
    let handle =
        tokio::spawn(oauth.refresh(oauth_credentials("a", "r", 0), CancellationToken::new()));
    // The dated retry-after is one hour out; the five-second retry budget
    // cannot cover it, so the retry stops and the 429 is the outcome.
    let error = handle
        .await
        .expect("the refresh task joins")
        .expect_err("the over-budget dated retry fails refresh");
    assert_eq!(error.to_string(), "429 Too Many Requests: ");
    assert_eq!(mock.request_count(), 2);
}

#[tokio::test]
async fn a_429_with_an_unparseable_retry_after_stops_the_retry() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    mock.on(|request| request.url.ends_with("/models"))
        .respond_sequence(vec![
            MockResponse::status(429).with_header("Retry-After", "not a date"),
            json_response(200, &serde_json::json!({ "data": [] })),
        ]);

    let oauth = flow(&mock);
    let error = oauth
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the unparseable retry-after keeps the 429, which fails refresh");
    assert_eq!(error.to_string(), "429 Too Many Requests: ");
    assert_eq!(mock.request_count(), 2, "no retry ran");
}

// ---------------------------------------------------------------------------
// The catalog parser's own surface
// ---------------------------------------------------------------------------

#[test]
fn the_catalog_filters_disabled_tool_calls_and_gates_by_picker() {
    let known = known_models();
    let raw = serde_json::json!({ "data": [
        model_entry("picker-on", true, "enabled"),
        model_entry("picker-off", false, "enabled"),
        {
            "id": "no-tools",
            "model_picker_enabled": true,
            "policy": { "state": "enabled" },
            "capabilities": { "supports": { "tool_calls": false } },
        },
        "not-an-object",
        3,
        {
            "model_picker_enabled": true,
        },
    ]});
    let catalog =
        parse_github_copilot_model_catalog(&raw, false, &known).expect("the catalog parses");
    assert_eq!(catalog.available_model_ids, vec!["picker-on"]);
    assert!(
        catalog.policy_model_ids.is_empty(),
        "no unconfigured models means no policy batch: {catalog:?}"
    );
}

#[test]
fn the_catalog_policy_fallback_applies_only_when_allowed() {
    let known = known_models();
    let raw = serde_json::json!({ "data": [
        model_entry("enabled-a", false, "enabled"),
        model_entry("disabled", false, "disabled"),
    ]});
    let strict =
        parse_github_copilot_model_catalog(&raw, false, &known).expect("the catalog parses");
    assert!(
        strict.available_model_ids.is_empty(),
        "picker-disabled models stay out without the fallback: {strict:?}"
    );

    let fallback =
        parse_github_copilot_model_catalog(&raw, true, &known).expect("the catalog parses");
    assert_eq!(
        fallback.available_model_ids,
        vec!["enabled-a"],
        "the fallback promotes picker-disabled but policy-enabled models"
    );
}

#[test]
fn the_catalog_policy_ids_need_the_known_models_gate() {
    let known = known_models();
    let raw = serde_json::json!({ "data": [
        model_entry("claude-sonnet-4.6", true, "unconfigured"),
        model_entry("unknown-model", true, "unconfigured"),
    ]});
    let catalog =
        parse_github_copilot_model_catalog(&raw, false, &known).expect("the catalog parses");
    assert_eq!(
        catalog.policy_model_ids,
        vec!["claude-sonnet-4.6"],
        "an unknown id never enters the policy batch: {catalog:?}"
    );

    let no_data = serde_json::json!({ "models": [] });
    let error = parse_github_copilot_model_catalog(&no_data, false, &known)
        .expect_err("the missing data array rejects");
    assert_eq!(error.to_string(), "Invalid Copilot models response");

    let not_objects = serde_json::json!({ "data": ["nope", 3] });
    let catalog = parse_github_copilot_model_catalog(&not_objects, false, &known)
        .expect("non-object entries skip");
    assert!(catalog.available_model_ids.is_empty());
}

// ---------------------------------------------------------------------------
// The adapter surface and the loader seam
// ---------------------------------------------------------------------------

#[test]
fn to_auth_derives_the_api_key_and_the_proxy_endpoint() {
    let mock = MockHttpClient::new();
    let oauth = flow(&mock);
    let auth = oauth.to_auth(&oauth_credentials(COPILOT_TOKEN, "r", 0));
    assert_eq!(
        auth,
        ModelAuth {
            api_key: Some(COPILOT_TOKEN.to_owned()),
            headers: None,
            base_url: Some("https://api.individual.githubcopilot.com".to_owned()),
        }
    );

    // The proxy-ep capture rewrites `proxy.` to `api.`.
    let proxy_token = "tid=abc;exp=1;proxy-ep=proxy.enterprise.example;rest";
    let auth = oauth.to_auth(&oauth_credentials(proxy_token, "r", 0));
    assert_eq!(
        auth.base_url.as_deref(),
        Some("https://api.enterprise.example"),
        "the per-token proxy endpoint becomes the base URL"
    );
}

#[test]
fn the_flow_debug_names_the_type() {
    let mock = MockHttpClient::new();
    let oauth = flow(&mock);
    assert_eq!(format!("{oauth:?}"), "GitHubCopilotOAuth");
    assert_eq!(oauth.auth().name, "GitHub Copilot");
}

#[tokio::test]
async fn the_loader_wires_the_login_and_derivation_closures() {
    // The loader wires the process-default seam, so the closure drives here
    // only take offline paths: a malformed enterprise answer rejects before
    // any wire call, and the derivation is side-effect-free.
    let oauth = load_github_copilot_oauth().await;
    assert_eq!(oauth.name, "GitHub Copilot");
    assert_eq!(oauth.is_subscription, Some(true));

    let scripted = Arc::new(ScriptedAuthInteraction::answering("ht tp://bad"));
    let error = ((oauth.login)(provider_interaction(&scripted, CancellationToken::new())))
        .await
        .expect_err("the malformed domain rejects before the wire");
    assert_eq!(error.to_string(), "Invalid GitHub Enterprise URL/domain");

    let auth = (oauth.to_auth)(oauth_credentials(COPILOT_TOKEN, "r", 0))
        .await
        .expect("the derivation resolves");
    assert_eq!(
        auth,
        ModelAuth {
            api_key: Some(COPILOT_TOKEN.to_owned()),
            headers: None,
            base_url: Some("https://api.individual.githubcopilot.com".to_owned()),
        }
    );
}

#[tokio::test]
async fn the_loader_wires_the_refresh_closure() {
    let _mock = MockHttpClient::new();
    let oauth = load_github_copilot_oauth().await;

    // A pre-cancelled signal aborts the exchange at the seam, offline: the
    // closure wiring, not the network, is what this pins.
    let signal = CancellationToken::new();
    signal.cancel();
    let error = (oauth.refresh)(oauth_credentials("a", "r", 0), signal)
        .await
        .expect_err("the aborted refresh rejects");
    assert_eq!(
        error.to_string(),
        "Login cancelled",
        "the aborted exchange surfaces as the cancelled login"
    );
}

#[tokio::test(start_paused = true)]
async fn an_enable_that_rate_limits_exhausting_the_budget_fails_the_login() {
    let mock = MockHttpClient::new();
    mount_login_routes(
        &mock,
        &serde_json::json!({ "data": [
            model_entry("claude-sonnet-4.6", false, "unconfigured"),
        ]}),
    );
    mock.on(|request| request.url.ends_with("/policy"))
        .respond(MockResponse::status(429).with_header("Retry-After", "30"));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    tokio::time::advance(Duration::from_secs(1)).await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("the rate-limited policy update is best effort");
    assert_eq!(
        credential.extra.get("availableModelIds"),
        Some(&serde_json::json!([])),
        "the exhausted policy batch contributes no ids"
    );
}

// ---------------------------------------------------------------------------
// The rate-limit retry machinery through the login path's budget
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn a_429_without_a_retry_after_uses_the_default_backoff() {
    let mock = MockHttpClient::new();
    mount_login_route_builder(&mock, |builder| {
        builder.respond_sequence(vec![
            MockResponse::status(429),
            json_response(200, &serde_json::json!({ "data": [] })),
        ]);
    });
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "access_token": "gh-token" }),
        ));
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));
    // The wait-before-first-poll sleep, then the 500 ms default backoff.
    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::time::advance(Duration::from_millis(500)).await;
    tokio::task::yield_now().await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("the backoff retried login resolves");
    assert_eq!(
        credential.extra.get("availableModelIds"),
        Some(&serde_json::json!([]))
    );
    assert_eq!(mock.request_count(), 5, "start, poll, token, 429, retried");
}

#[tokio::test]
async fn a_429_with_a_far_retry_after_date_exhausts_the_budget() {
    let mock = MockHttpClient::new();
    // The wait-before-first-poll sleep and the poll fire on the real clock:
    // no backoff sleep runs because the dated retry is out of budget.
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "access_token": "gh-token" }),
        ));
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    // One hour after START, in IMF-fixdate form.
    mount_login_route_builder(&mock, |builder| {
        builder.respond_sequence(vec![
            MockResponse::status(429).with_header("Retry-After", "Tue, 14 Jul 2026 04:33:20 GMT"),
            json_response(200, &serde_json::json!({ "data": [] })),
        ]);
    });

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));
    // The one-second wait-before-first-poll sleep runs on the real clock.
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the out-of-budget dated retry fails login");
    assert_eq!(error.to_string(), "429 Too Many Requests: ");
}

#[tokio::test(start_paused = true)]
async fn the_device_poll_fails_without_an_error_or_access_token_field() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond(json_response(200, &serde_json::json!({})));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    tokio::time::advance(Duration::from_secs(1)).await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the fieldless poll fails login");
    assert_eq!(error.to_string(), "Invalid device token response");
}

#[tokio::test(start_paused = true)]
async fn the_device_poll_parks_on_authorization_pending_then_completes() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond_sequence(vec![
            json_response(
                200,
                &serde_json::json!({ "error": "authorization_pending" }),
            ),
            json_response(200, &serde_json::json!({ "access_token": "gh-token" })),
        ]);
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    mock.on(|request| request.url == MODELS_URL)
        .respond(json_response(200, &serde_json::json!({ "data": [] })));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::time::advance(Duration::from_secs(1)).await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("login resolves");
    assert_eq!(credential.access, COPILOT_TOKEN);
    assert_eq!(
        mock.request_count(),
        5,
        "start, pending poll, complete poll, exchange, models"
    );
}

#[tokio::test(start_paused = true)]
async fn a_failing_policy_update_stops_the_batch_without_failing_the_login() {
    // The policy POST transport-fails: the batch stops, the login still
    // resolves with the picker's own ids.
    let mock = MockHttpClient::new();
    mount_login_routes(
        &mock,
        &serde_json::json!({ "data": [
            model_entry("claude-sonnet-4.6", false, "unconfigured"),
        ]}),
    );

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    tokio::time::advance(Duration::from_secs(1)).await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("the best-effort batch stop keeps the login");
    assert_eq!(
        credential.extra.get("availableModelIds"),
        Some(&serde_json::json!([]))
    );
}

#[tokio::test(start_paused = true)]
async fn a_failed_policy_status_counts_as_not_enabled() {
    // A non-429 policy failure answers false and the batch keeps going.
    let mock = MockHttpClient::new();
    mount_login_routes(
        &mock,
        &serde_json::json!({ "data": [
            model_entry("claude-sonnet-4.6", false, "unconfigured"),
            model_entry("gpt-5.4", false, "unconfigured"),
        ]}),
    );
    mock.on(|request| request.url.ends_with("/policy"))
        .respond(MockResponse::status(400).with_body("nope"));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));

    tokio::time::advance(Duration::from_secs(1)).await;
    let credential = handle
        .await
        .expect("the login task joins")
        .expect("the failed policy updates are not enabled");
    assert_eq!(
        credential.extra.get("availableModelIds"),
        Some(&serde_json::json!([]))
    );
}

#[tokio::test(start_paused = true)]
async fn a_cancelled_enable_fails_the_login() {
    // The policy responder cancels the login's own signal and aborts the
    // request; the enable step propagates the cancellation.
    let mock = MockHttpClient::new();
    mount_login_routes(
        &mock,
        &serde_json::json!({ "data": [
            model_entry("claude-sonnet-4.6", false, "unconfigured"),
        ]}),
    );
    let signal = CancellationToken::new();
    mock.on(|request| request.url.ends_with("/policy"))
        .respond_fn({
            let signal = signal.clone();
            move |_recorded| {
                let signal = signal.clone();
                async move {
                    signal.cancel();
                    Err(pi_ai::http::HttpError::Aborted)
                }
            }
        });

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle = tokio::spawn(oauth.login(provider_interaction(&scripted, signal)));

    tokio::time::advance(Duration::from_secs(1)).await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the cancelled enable fails login");
    assert_eq!(
        error.to_string(),
        "Login cancelled",
        "the aborted exchange surfaces as the cancelled login"
    );
}

#[tokio::test]
async fn refresh_with_an_empty_enterprise_url_drops_the_domain() {
    // An empty enterpriseUrl reads as "no domain", upstream's falsy-string
    // check, so the exchange runs against github.com and the rotated
    // credential drops the field.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    mock.on(|request| request.url == MODELS_URL)
        .respond(json_response(200, &serde_json::json!({ "data": [] })));

    let oauth = flow(&mock);
    let mut stored = oauth_credentials("a", "r", 0);
    stored.extra.insert(
        "enterpriseUrl".to_owned(),
        serde_json::Value::String(String::new()),
    );
    let refreshed = oauth
        .refresh(stored, CancellationToken::new())
        .await
        .expect("the refresh resolves");
    assert_eq!(
        refreshed.extra.get("enterpriseUrl"),
        None,
        "the empty domain drops, upstream's falsy-string semantics"
    );
}

#[tokio::test]
async fn a_token_without_a_proxy_capture_falls_back_to_the_domain() {
    // The proxy-ep capture stops at `;`; an empty capture leaves no endpoint,
    // so the base URL falls through.
    let mock = MockHttpClient::new();
    let oauth = flow(&mock);
    let auth = oauth.to_auth(&oauth_credentials("tid=1;proxy-ep=;rest", "r", 0));
    assert_eq!(
        auth.base_url.as_deref(),
        Some("https://api.individual.githubcopilot.com"),
        "an empty capture yields no endpoint"
    );
}

#[tokio::test]
async fn refresh_surfaces_the_exchange_status_reason() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(MockResponse::status(401).with_body("revoked"));

    let oauth = flow(&mock);
    let error = oauth
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the 401 fails refresh");
    assert_eq!(error.to_string(), "401 Unauthorized: revoked");
}

// ---------------------------------------------------------------------------
// The remaining retry and status-reason arms
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_429_with_an_unparseable_date_and_headerless_replies_stop_the_retry() {
    // A Retry-After that parses as neither seconds nor an HTTP date returns
    // the response; covered together with the date-shaped arm above it.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    mount_login_route_builder(&mock, |builder| {
        builder.respond_sequence(vec![
            MockResponse::status(429).with_header("Retry-After", "1"),
            MockResponse::status(429).with_header("Retry-After", "not a date"),
            json_response(200, &serde_json::json!({ "data": [] })),
        ]);
    });
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "access_token": "gh-token" }),
        ));
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let outcome = oauth.login(provider_interaction(&scripted, CancellationToken::new()));
    let error = outcome.await.expect_err("the unparseable date fails login");
    assert_eq!(error.to_string(), "429 Too Many Requests: ");
}

#[tokio::test(start_paused = true)]
async fn a_429_with_only_a_date_header_out_of_budget_exits_before_the_sleep() {
    // The dated retry-after is an hour out; the five-second budget rejects
    // the retry before the sleep, covering the budget-exit branch.
    let mock = MockHttpClient::new();
    mount_login_route_builder(&mock, |builder| {
        builder.respond_sequence(vec![
            MockResponse::status(429).with_header("Retry-After", "Tue, 14 Jul 2026 04:33:20 GMT"),
            json_response(200, &serde_json::json!({ "data": [] })),
        ]);
    });
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "access_token": "gh-token" }),
        ));
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));
    tokio::time::advance(Duration::from_secs(1)).await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the dated retry over the budget fails login");
    assert_eq!(error.to_string(), "429 Too Many Requests: ");
}

#[tokio::test(start_paused = true)]
async fn the_models_fetch_reports_each_status_reason() {
    // The status-reason match arms the Copilot APIs answer with, exercised
    // through the exchange failure message.
    for (status, reason) in [
        (404_u16, "Not Found"),
        (405, "Method Not Allowed"),
        (409, "Conflict"),
        (500, "Internal Server Error"),
        (502, "Bad Gateway"),
        (503, "Service Unavailable"),
        (504, "Gateway Timeout"),
    ] {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url == COPILOT_TOKEN_URL)
            .respond(MockResponse::status(status).with_body("kaput"));
        let oauth = flow(&mock);
        let error = oauth
            .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
            .await
            .expect_err("the status failure fails refresh");
        assert_eq!(
            error.to_string(),
            format!("{status} {reason}: kaput"),
            "reason phrase for {status}"
        );
    }
}

#[tokio::test]
async fn a_429_with_a_non_finite_retry_header_stops_the_retry() {
    // "nan" parses as a float but not a finite one: the response returns
    // without a retry, upstream's `Number.isFinite` guard.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));
    mock.on(|request| request.url.ends_with("/models"))
        .respond_sequence(vec![
            MockResponse::status(429).with_header("Retry-After", "nan"),
            json_response(200, &serde_json::json!({ "data": [] })),
        ]);

    let oauth = flow(&mock);
    let error = oauth
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the non-finite retry-after keeps the 429");
    assert_eq!(error.to_string(), "429 Too Many Requests: ");
    assert_eq!(mock.request_count(), 2, "no retry ran");
}

#[tokio::test]
async fn an_unknown_status_reports_the_empty_reason() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(MockResponse::status(418).with_body("teapot"));

    let oauth = flow(&mock);
    let error = oauth
        .refresh(oauth_credentials("a", "r", 0), CancellationToken::new())
        .await
        .expect_err("the unknown status fails refresh");
    assert_eq!(error.to_string(), "418 : teapot");
}

#[tokio::test(start_paused = true)]
async fn a_429_with_a_non_finite_retry_header_stops_the_login_retry() {
    // The login path's models fetch carries the two-retry budget, so the
    // Retry-After header is read: "nan" parses as a float but not a finite
    // one, and the response returns without a retry.
    let mock = MockHttpClient::new();
    mount_login_route_builder(&mock, |builder| {
        builder.respond_sequence(vec![
            MockResponse::status(429).with_header("Retry-After", "nan"),
            json_response(200, &serde_json::json!({ "data": [] })),
        ]);
    });
    mock.on(|request| request.url == DEVICE_URL)
        .respond(json_response(200, &device_response()));
    mock.on(|request| request.url == ACCESS_TOKEN_URL)
        .respond(json_response(
            200,
            &serde_json::json!({ "access_token": "gh-token" }),
        ));
    mock.on(|request| request.url == COPILOT_TOKEN_URL)
        .respond(json_response(200, &copilot_token_response()));

    let oauth = flow(&mock);
    let scripted = Arc::new(ScriptedAuthInteraction::answering(""));
    let handle =
        tokio::spawn(oauth.login(provider_interaction(&scripted, CancellationToken::new())));
    // The wait-before-first-poll sleep rides the paused clock.
    tokio::time::advance(Duration::from_secs(1)).await;
    let error = handle
        .await
        .expect("the login task joins")
        .expect_err("the non-finite retry-after fails login");
    assert_eq!(error.to_string(), "429 Too Many Requests: ");
    assert_eq!(
        mock.request_count(),
        4,
        "device start, poll, token exchange, 429"
    );
}
