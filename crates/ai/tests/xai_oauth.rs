//! The xAI OAuth device flow, from `test/xai-oauth.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the seam mock replaces the
//! stubbed `globalThis.fetch`, and the tokio paused clock with a
//! [`pi_ai::auth::clock::FixedClock`] replaces fake timers — one
//! `advance` moves both, the lockstep `advanceTimersByTimeAsync` plus
//! `setSystemTime` gave upstream. Where upstream spreads `undefined` onto a
//! token response field, the fixture drops the key, the wire shape
//! `JSON.stringify` produces.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_ai::auth::clock::{AuthClock as _, FixedClock};
use pi_ai::auth::oauth::xai::XaiOAuth;
use pi_ai::auth::types::{
    AuthError, AuthEvent, BoxAuthFuture, ModelAuth, OAuthAuth, OAuthCredential,
};
use pi_ai::http::{MockHttpClient, json_response};
use tokio_util::sync::CancellationToken;

use common::auth_interaction::{ScriptedAuthInteraction, provider_interaction};
use common::paused_clock::advance;
use common::seam_forms::{form_field, form_fields};

const DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
const TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
const XAI_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const XAI_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
const UNTRUSTED_URI: &str = "Untrusted verification URI in xAI OAuth response";

/// `new Date("2026-07-09T20:00:00Z").getTime()`, upstream's `startTime`.
const START: i64 = 1_783_627_200_000;

/// The device-code start response, upstream's `deviceCodeResponse` with the
/// given field overrides.
fn device_code_response(overrides: &[(&str, serde_json::Value)]) -> serde_json::Value {
    let mut body = serde_json::json!({
        "device_code": "device-code",
        "user_code": "ABCD-1234",
        "verification_uri": "https://accounts.x.ai/oauth2/device",
        "expires_in": 900,
        "interval": 5,
    });
    let fields = body.as_object_mut().expect("device response object");
    for (key, value) in overrides {
        fields.insert((*key).to_owned(), value.clone());
    }
    body
}

/// The token response, upstream's `tokenResponse`: `None` drops the field,
/// the port of `undefined` spreading onto the wire.
fn token_response(overrides: &[(&str, Option<serde_json::Value>)]) -> serde_json::Value {
    let mut body = serde_json::json!({
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 21_600,
        "token_type": "Bearer",
    });
    let fields = body.as_object_mut().expect("token response object");
    for (key, value) in overrides {
        match value {
            Some(value) => {
                fields.insert((*key).to_owned(), value.clone());
            }
            None => {
                fields.remove(*key);
            }
        }
    }
    body
}

/// Mount the token route answering the queued `(status, body)` replies in
/// order, recording each poll's epoch time — the `pollTimes.push(Date.now())`
/// inside upstream's fetch stub. Returns a receiver signalling each poll:
/// tokio fires timers at the next driver park, not inside `advance`, so the
/// suites park on the marker between clock moves the way
/// `advanceTimersByTimeAsync` flushes as it advances.
#[must_use]
fn mount_token_route(
    mock: &MockHttpClient,
    clock: &Arc<FixedClock>,
    poll_times: &Arc<Mutex<Vec<i64>>>,
    replies: Vec<(u16, serde_json::Value)>,
) -> tokio::sync::mpsc::UnboundedReceiver<()> {
    let (poll_marker, marker_receiver) = tokio::sync::mpsc::unbounded_channel();
    let clock = Arc::clone(clock);
    let poll_times = Arc::clone(poll_times);
    let replies = Arc::new(Mutex::new(VecDeque::from(replies)));
    mock.on(|request| request.url == TOKEN_URL)
        .respond_fn(move |_recorded| {
            let clock = Arc::clone(&clock);
            let poll_times = Arc::clone(&poll_times);
            let replies = Arc::clone(&replies);
            let poll_marker = poll_marker.clone();
            async move {
                poll_times.lock().expect("poll times").push(clock.now_ms());
                let _ = poll_marker.send(());
                let (status, body) = replies
                    .lock()
                    .expect("token replies")
                    .pop_front()
                    .expect("Unexpected token poll");
                Ok(json_response(status, &body))
            }
        });
    marker_receiver
}

/// `loginXaiForTest`: the flow's login over a scripted interaction whose
/// `prompt` rejects and whose events the caller reads through the returned
/// handle.
fn login_xai(
    oauth: &XaiOAuth,
    signal: CancellationToken,
    scripted: Arc<ScriptedAuthInteraction>,
) -> BoxAuthFuture<Result<OAuthCredential, AuthError>> {
    oauth.login(provider_interaction(scripted, signal))
}

/// The throwaway poll-time sink the cases that never assert poll times pass.
fn unwatched_poll_times() -> Arc<Mutex<Vec<i64>>> {
    Arc::new(Mutex::new(Vec::new()))
}

#[tokio::test(start_paused = true)]
async fn uses_the_device_grant_delays_polling_and_handles_pending_and_slow_down() {
    let clock = Arc::new(FixedClock::new(START));
    let poll_times = Arc::new(Mutex::new(Vec::new()));
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_CODE_URL)
        .respond(json_response(200, &device_code_response(&[])));
    let mut poll_marker = mount_token_route(
        &mock,
        &clock,
        &poll_times,
        vec![
            (400, serde_json::json!({ "error": "authorization_pending" })),
            (
                400,
                serde_json::json!({ "error": "slow_down", "interval": 10 }),
            ),
            (200, token_response(&[])),
        ],
    );

    let oauth = XaiOAuth::new(Arc::new(mock.clone()), clock.clone());
    let scripted = Arc::new(ScriptedAuthInteraction::rejecting_prompt(
        "Unexpected prompt",
    ));
    let runner = tokio::spawn(login_xai(
        &oauth,
        CancellationToken::new(),
        Arc::clone(&scripted),
    ));

    // The device authorization flushes immediately, upstream's
    // advanceTimersByTimeAsync(0).
    tokio::time::advance(Duration::ZERO).await;
    assert_eq!(
        scripted.events(),
        vec![AuthEvent::DeviceCode {
            user_code: "ABCD-1234".to_owned(),
            verification_uri: "https://accounts.x.ai/oauth2/device".to_owned(),
            interval_seconds: Some(5),
            expires_in_seconds: Some(900),
        }],
        "the device code is announced before the first poll"
    );
    assert_eq!(
        mock.recorded().len(),
        1,
        "the device-code request is the only wire call so far"
    );
    assert_eq!(
        *poll_times.lock().expect("poll times"),
        Vec::<i64>::new(),
        "the first poll waits out the reported interval"
    );

    advance(&clock, 5_000).await;
    poll_marker
        .recv()
        .await
        .expect("the poll marker channel stays open");
    assert_eq!(*poll_times.lock().expect("poll times"), vec![START + 5_000]);

    // slow_down raised the interval to 10 seconds.
    advance(&clock, 5_000).await;
    poll_marker
        .recv()
        .await
        .expect("the poll marker channel stays open");
    assert_eq!(
        *poll_times.lock().expect("poll times"),
        vec![START + 5_000, START + 10_000]
    );

    advance(&clock, 10_000).await;
    poll_marker
        .recv()
        .await
        .expect("the poll marker channel stays open");
    let credential = runner
        .await
        .expect("login task completes")
        .expect("login succeeds");
    assert_eq!(
        *poll_times.lock().expect("poll times"),
        vec![START + 5_000, START + 10_000, START + 20_000]
    );
    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    assert_eq!(
        credential.expires,
        START + 20_000 + 21_600_000 - 300_000,
        "the expiry is the poll time plus the wire lifetime less the refresh skew"
    );

    let served = mock.recorded();
    assert_eq!(served.len(), 4);
    let fields = form_fields(&served[0]);
    assert_eq!(form_field(&fields, "client_id"), XAI_CLIENT_ID);
    assert_eq!(form_field(&fields, "scope"), XAI_SCOPE);
    assert_eq!(form_field(&fields, "referrer"), "pi");
    for request in &served[1..] {
        assert_eq!(request.url, TOKEN_URL);
        let fields = form_fields(request);
        assert_eq!(form_field(&fields, "grant_type"), DEVICE_GRANT);
        assert_eq!(form_field(&fields, "client_id"), XAI_CLIENT_ID);
        assert_eq!(form_field(&fields, "device_code"), "device-code");
    }
}

#[tokio::test(start_paused = true)]
async fn falls_back_to_the_default_poll_interval_when_the_response_reports_interval_0() {
    let clock = Arc::new(FixedClock::new(START));
    let poll_times = Arc::new(Mutex::new(Vec::new()));
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_CODE_URL)
        .respond(json_response(
            200,
            &device_code_response(&[("interval", serde_json::json!(0))]),
        ));
    let _marker = mount_token_route(&mock, &clock, &poll_times, vec![(200, token_response(&[]))]);

    let oauth = XaiOAuth::new(Arc::new(mock), clock.clone());
    let runner = tokio::spawn(login_xai(
        &oauth,
        CancellationToken::new(),
        Arc::new(ScriptedAuthInteraction::rejecting_prompt(
            "Unexpected prompt",
        )),
    ));

    tokio::time::advance(Duration::ZERO).await;
    // RFC 8628 default interval is 5 seconds when the server does not
    // require a wait.
    advance(&clock, 5_000).await;
    runner
        .await
        .expect("login task completes")
        .expect("login succeeds");
    assert_eq!(*poll_times.lock().expect("poll times"), vec![START + 5_000]);
}

#[tokio::test(start_paused = true)]
async fn prefers_verification_uri_complete_when_the_server_provides_it() {
    let clock = Arc::new(FixedClock::new(START));
    let poll_times = unwatched_poll_times();
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_CODE_URL)
        .respond(json_response(
            200,
            &device_code_response(&[(
                "verification_uri_complete",
                serde_json::json!("https://accounts.x.ai/oauth2/device?user_code=ABCD-1234"),
            )]),
        ));
    let _marker = mount_token_route(&mock, &clock, &poll_times, vec![(200, token_response(&[]))]);

    let oauth = XaiOAuth::new(Arc::new(mock), clock.clone());
    let scripted = Arc::new(ScriptedAuthInteraction::rejecting_prompt(
        "Unexpected prompt",
    ));
    let runner = tokio::spawn(login_xai(
        &oauth,
        CancellationToken::new(),
        Arc::clone(&scripted),
    ));

    tokio::time::advance(Duration::ZERO).await;
    advance(&clock, 5_000).await;
    runner
        .await
        .expect("login task completes")
        .expect("login succeeds");

    assert_eq!(
        scripted.events(),
        vec![AuthEvent::DeviceCode {
            user_code: "ABCD-1234".to_owned(),
            verification_uri: "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234".to_owned(),
            interval_seconds: Some(5),
            expires_in_seconds: Some(900),
        }],
    );
}

#[tokio::test]
async fn rejects_a_non_https_verification_uri_complete() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_CODE_URL)
        .respond(json_response(
            200,
            &device_code_response(&[(
                "verification_uri_complete",
                serde_json::json!("http://accounts.x.ai/oauth2/device?user_code=ABCD-1234"),
            )]),
        ));

    let oauth = XaiOAuth::new(Arc::new(mock), Arc::new(FixedClock::new(START)));
    let scripted = Arc::new(ScriptedAuthInteraction::rejecting_prompt(
        "Unexpected prompt",
    ));
    let error = oauth
        .login(provider_interaction(scripted, CancellationToken::new()))
        .await
        .expect_err("the untrusted verification URI rejects");

    assert_eq!(error.to_string(), UNTRUSTED_URI);
}

/// Upstream `it("rejects a non-https verification URI: http://accounts.x.ai/oauth2/device")`.
#[tokio::test]
async fn rejects_a_non_https_verification_uri_http_url() {
    assert_eq!(
        login_with_verification_uri("http://accounts.x.ai/oauth2/device")
            .await
            .to_string(),
        UNTRUSTED_URI
    );
}

/// Upstream `it("rejects a non-https verification URI: file:///etc/passwd")`.
#[tokio::test]
async fn rejects_a_non_https_verification_uri_file_url() {
    assert_eq!(
        login_with_verification_uri("file:///etc/passwd")
            .await
            .to_string(),
        UNTRUSTED_URI
    );
}

/// Upstream `it("rejects a non-https verification URI: not a url")`.
#[tokio::test]
async fn rejects_a_non_https_verification_uri_garbage() {
    assert_eq!(
        login_with_verification_uri("not a url").await.to_string(),
        UNTRUSTED_URI
    );
}

/// One login whose device-code response carries the given verification URI,
/// the shape the three rejected-URI cases share.
async fn login_with_verification_uri(verification_uri: &str) -> AuthError {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_CODE_URL)
        .respond(json_response(
            200,
            &device_code_response(&[("verification_uri", serde_json::json!(verification_uri))]),
        ));

    let oauth = XaiOAuth::new(Arc::new(mock), Arc::new(FixedClock::new(START)));
    let scripted = Arc::new(ScriptedAuthInteraction::rejecting_prompt(
        "Unexpected prompt",
    ));
    oauth
        .login(provider_interaction(scripted, CancellationToken::new()))
        .await
        .expect_err("the untrusted verification URI rejects")
}

/// Upstream `it("fails when device authorization is denied: access_denied")`.
#[tokio::test(start_paused = true)]
async fn fails_when_device_authorization_is_denied_access_denied() {
    assert_eq!(
        login_denied("access_denied").await,
        "xAI device authorization was denied"
    );
}

/// Upstream `it("fails when device authorization is denied: authorization_denied")`.
#[tokio::test(start_paused = true)]
async fn fails_when_device_authorization_is_denied_authorization_denied() {
    assert_eq!(
        login_denied("authorization_denied").await,
        "xAI device authorization was denied"
    );
}

/// One login whose token poll answers the given OAuth error, the shape the
/// two denied cases share.
async fn login_denied(oauth_error: &str) -> String {
    let clock = Arc::new(FixedClock::new(START));
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_CODE_URL)
        .respond(json_response(
            200,
            &device_code_response(&[("interval", serde_json::json!(1))]),
        ));
    let _marker = mount_token_route(
        &mock,
        &clock,
        &unwatched_poll_times(),
        vec![(400, serde_json::json!({ "error": oauth_error }))],
    );

    let oauth = XaiOAuth::new(Arc::new(mock), clock.clone());
    let scripted = Arc::new(ScriptedAuthInteraction::rejecting_prompt(
        "Unexpected prompt",
    ));
    let runner =
        tokio::spawn(oauth.login(provider_interaction(scripted, CancellationToken::new())));

    tokio::time::advance(Duration::ZERO).await;
    advance(&clock, 1_000).await;
    runner
        .await
        .expect("login task completes")
        .expect_err("denied authorization rejects")
        .to_string()
}

#[tokio::test(start_paused = true)]
async fn cancels_while_waiting_for_the_first_token_poll() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == DEVICE_CODE_URL)
        .respond(json_response(200, &device_code_response(&[])));

    let clock = Arc::new(FixedClock::new(START));
    let oauth = XaiOAuth::new(Arc::new(mock.clone()), clock);
    let signal = CancellationToken::new();
    let scripted = Arc::new(
        ScriptedAuthInteraction::rejecting_prompt("Unexpected prompt")
            .aborting_on_device_code(signal.clone()),
    );
    let runner = tokio::spawn(login_xai(&oauth, signal, scripted));

    // The device-code event aborts the flow before its first poll.
    tokio::time::advance(Duration::ZERO).await;
    let error = runner
        .await
        .expect("login task completes")
        .expect_err("cancelled login rejects");

    assert_eq!(error.to_string(), "Login cancelled");
    assert_eq!(mock.request_count(), 1, "no token poll left the wire");
}

#[tokio::test]
async fn refreshes_tokens_and_preserves_an_unrotated_refresh_token() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == TOKEN_URL)
        .respond_sequence(vec![
            json_response(
                200,
                &token_response(&[
                    ("access_token", Some(serde_json::json!("new-access"))),
                    ("refresh_token", Some(serde_json::json!("new-refresh"))),
                ]),
            ),
            // `refresh_token: undefined` drops the key on the wire.
            json_response(
                200,
                &token_response(&[
                    ("access_token", Some(serde_json::json!("newer-access"))),
                    ("refresh_token", None),
                ]),
            ),
        ]);

    let oauth = XaiOAuth::new(Arc::new(mock.clone()), Arc::new(FixedClock::new(START)));
    let rotated = oauth
        .refresh(
            &OAuthCredential::new("old-access", "old-refresh", 0),
            CancellationToken::new(),
        )
        .await
        .expect("rotated refresh succeeds");
    let preserved = oauth
        .refresh(
            &OAuthCredential::new("old-access", "keep-refresh", 0),
            CancellationToken::new(),
        )
        .await
        .expect("preserved refresh succeeds");

    assert_eq!(rotated.access, "new-access");
    assert_eq!(rotated.refresh, "new-refresh");
    assert_eq!(preserved.access, "newer-access");
    assert_eq!(
        preserved.refresh, "keep-refresh",
        "the unrotated refresh token survives an absent wire field"
    );
    assert_eq!(OAuthAuth::name(&oauth), "xAI (Grok/X subscription)");
    assert_eq!(
        oauth.to_auth(&preserved),
        ModelAuth::api_key("newer-access")
    );

    let served = mock.recorded();
    assert_eq!(served.len(), 2);
    for (request, refresh_token) in served.iter().zip(["old-refresh", "keep-refresh"]) {
        assert_eq!(request.url, TOKEN_URL);
        let fields = form_fields(request);
        assert_eq!(form_field(&fields, "grant_type"), "refresh_token");
        assert_eq!(form_field(&fields, "client_id"), XAI_CLIENT_ID);
        assert_eq!(form_field(&fields, "refresh_token"), refresh_token);
    }
}

#[tokio::test(start_paused = true)]
async fn assumes_a_one_hour_lifetime_when_expires_in_is_missing() {
    let clock = Arc::new(FixedClock::new(START));
    let mock = MockHttpClient::new();
    // `expires_in: undefined` drops the key on the wire.
    mock.on(|request| request.url == TOKEN_URL)
        .respond(json_response(200, &token_response(&[("expires_in", None)])));

    let oauth = XaiOAuth::new(Arc::new(mock), clock);
    let credential = oauth
        .refresh(
            &OAuthCredential::new("old-access", "old-refresh", 0),
            CancellationToken::new(),
        )
        .await
        .expect("refresh succeeds");

    assert_eq!(
        credential.expires,
        START + 3_600_000 - 300_000,
        "the missing wire lifetime falls back to the default hour"
    );
}

#[tokio::test]
async fn rejects_token_responses_with_missing_fields() {
    let mock = MockHttpClient::new();
    // `access_token: undefined` drops the key on the wire.
    mock.on(|request| request.url == TOKEN_URL)
        .respond(json_response(
            200,
            &token_response(&[("access_token", None)]),
        ));

    let oauth = XaiOAuth::new(Arc::new(mock), Arc::new(FixedClock::new(START)));
    let error = oauth
        .refresh(
            &OAuthCredential::new("old-access", "old-refresh", 0),
            CancellationToken::new(),
        )
        .await
        .expect_err("missing access token rejects");

    assert_eq!(
        error.to_string(),
        "Invalid xAI OAuth response field: access_token"
    );
}

#[tokio::test]
async fn surfaces_the_upstream_error_code_and_description_on_refresh_failure() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == TOKEN_URL)
        .respond(json_response(
            400,
            &serde_json::json!({
                "error": "invalid_grant",
                "error_description": "refresh token revoked",
            }),
        ));

    let oauth = XaiOAuth::new(Arc::new(mock), Arc::new(FixedClock::new(START)));
    let error = oauth
        .refresh(
            &OAuthCredential::new("old-access", "old-refresh", 0),
            CancellationToken::new(),
        )
        .await
        .expect_err("revoked refresh rejects");

    assert_eq!(
        error.to_string(),
        "xAI OAuth token refresh failed (HTTP 400): invalid_grant: refresh token revoked"
    );
}
