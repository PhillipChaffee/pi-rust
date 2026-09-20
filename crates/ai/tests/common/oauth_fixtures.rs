//! Shared OAuth-flow test scaffolding: mock-route mounts over the seam mock,
//! the login-task spawn/join pair, the wait helpers, and the loopback
//! callback sender the flow suites drive. The shapes stand in for upstream's
//! vitest stubs (`vi.stubGlobal("fetch")`, fake timers, and the inline
//! spawn/join the suites drive) at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::panic,
    reason = "the wait helpers abort when the awaited condition never holds; the tests pin outcomes"
)]

use std::future::Future;
use std::time::Duration;

use pi_ai::auth::types::{AuthError, OAuthCredentials, ProviderAuthInteraction};
use pi_ai::http::{MockHttpClient, json_response};

use super::auth_fixtures::{RecordingInteraction, provider_interaction};

/// Mount a JSON route on the seam mock: when a request hits `url`, the
/// canned `(status, body)` reply answers it.
pub fn mount_json_route(
    mock: &MockHttpClient,
    url: impl Into<String>,
    status: u16,
    body: &serde_json::Value,
) {
    let url = url.into();
    mock.on(move |request| request.url == url)
        .respond(json_response(status, body));
}

/// The standard token-exchange success body the flows rotate into a
/// credential, with the three wire fields the flows read.
#[must_use]
pub fn token_success(access: &str, refresh: &str, expires_in: i64) -> serde_json::Value {
    serde_json::json!({
        "access_token": access,
        "refresh_token": refresh,
        "expires_in": expires_in,
    })
}

/// Spawn the flow's login over the fresh interaction copy the recording
/// double carries, the task the browser-flow cases drive and join.
pub fn spawn_login<F, Fut>(
    login: F,
    recording: &RecordingInteraction,
) -> tokio::task::JoinHandle<Result<OAuthCredentials, AuthError>>
where
    F: FnOnce(ProviderAuthInteraction) -> Fut,
    Fut: Future<Output = Result<OAuthCredentials, AuthError>> + Send + 'static,
{
    tokio::spawn(login(provider_interaction(recording)))
}

/// Join a login task, asserting the task itself completed.
///
/// # Panics
/// Panics when the task panicked, or when the login rejected where the
/// caller pins a credential.
pub async fn task_credential(
    handle: tokio::task::JoinHandle<Result<OAuthCredentials, AuthError>>,
    join_what: &str,
    what: &str,
) -> OAuthCredentials {
    handle.await.expect(join_what).expect(what)
}

/// Join a login task, asserting the login rejected with the error the
/// caller pins.
///
/// # Panics
/// Panics when the task itself fails to join, or when the login resolves
/// instead of rejecting.
pub async fn task_error(
    handle: tokio::task::JoinHandle<Result<OAuthCredentials, AuthError>>,
    join_what: &str,
    what: &str,
) -> AuthError {
    handle.await.expect(join_what).expect_err(what)
}

/// The first `name` query parameter of `url`, `None` when the URL carries
/// no such parameter or does not parse.
#[must_use]
pub fn url_query_param(url: &str, name: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()?
        .query_pairs()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
}

/// Wait for a condition the login task reaches without clock dependence,
/// stepping the scheduler between probes like `advanceTimersByTimeAsync`
/// does, and return the probed value.
///
/// # Panics
/// Panics when `what` does not hold within five seconds of wall time.
pub async fn wait_until<T>(condition: impl Fn() -> Option<T>, what: &str) -> T {
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
///
/// # Panics
/// Panics when `what` does not hold within two thousand five-millisecond
/// steps.
pub async fn advance_until(condition: impl Fn() -> bool, what: &str) {
    for _ in 0..2_000 {
        if condition() {
            return;
        }
        tokio::time::advance(Duration::from_millis(5)).await;
        tokio::task::yield_now().await;
    }
    panic!("timed out waiting for {what}");
}

/// A raw one-request-per-connection write to `host:port`, the loopback
/// callback the browser sends; reads to EOF and returns the response as text.
///
/// # Panics
/// Panics when the connection, the write, or the read fails — the tests pin
/// outcomes, so an unexpected seam failure panics by design.
pub async fn send_loopback(host: &str, port: u16, request: impl AsRef<[u8]>) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect((host, port))
        .await
        .expect("the loopback server accepts");
    stream
        .write_all(request.as_ref())
        .await
        .expect("the request writes");
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .expect("the response reads to EOF");
    String::from_utf8(raw).expect("the response is utf-8")
}
