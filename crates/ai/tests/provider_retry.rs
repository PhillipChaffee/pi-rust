//! The provider-request retry port, from `test/provider-retry.test.ts`,
//! on the tokio paused clock in place of fake timers.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use pi_ai::utils::provider_retry::{
    ProviderRequestError, ProviderRetryOptions, retry_provider_request,
};
use tokio_util::sync::CancellationToken;

fn provider_error(status: Option<u16>, headers: Option<&[(&str, &str)]>) -> ProviderRequestError {
    let header_map: Option<BTreeMap<String, String>> = headers.map(|pairs| {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    });
    ProviderRequestError::new(status, header_map, format!("Provider error: {status:?}"))
}

fn options() -> ProviderRetryOptions {
    ProviderRetryOptions {
        max_retries: 0,
        max_retry_delay_ms: None,
        signal: None,
        random: None,
    }
}

/// Poll the pinned future once with a noop waker, stepping the paused clock
/// between polls like `advanceTimersByTimeAsync` does.
fn poll<F: Future>(fut: &mut std::pin::Pin<&mut F>) -> Poll<F::Output> {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    fut.as_mut().poll(&mut context)
}

#[tokio::test(start_paused = true)]
async fn retries_retryable_provider_errors() {
    let mut options = options();
    options.max_retries = 1;
    let mut calls = 0usize;
    let fut = retry_provider_request(
        move || {
            calls += 1;
            let first_failure = calls == 1;
            async move {
                if first_failure {
                    Err(provider_error(
                        Some(429),
                        Some(&[("retry-after-ms", "1000")][..]),
                    ))
                } else {
                    Ok(String::from("ok"))
                }
            }
        },
        &options,
    );
    let mut fut = pin!(fut);

    // The first request fails and schedules the 1000 ms backoff.
    assert!(
        poll(&mut fut).is_pending(),
        "the retry is inside the backoff"
    );
    tokio::time::advance(std::time::Duration::from_millis(999)).await;
    assert!(
        poll(&mut fut).is_pending(),
        "still inside the server-requested wait"
    );
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    assert_eq![fut.await.as_deref(), Ok("ok")];
}

#[tokio::test]
async fn rejects_a_provider_requested_retry_delay_above_the_limit() {
    let mut options = options();
    options.max_retries = 1;
    options.max_retry_delay_ms = Some(1_000);
    let outcome = retry_provider_request(
        move || {
            let failure = provider_error(Some(429), Some(&[("retry-after", "277403")][..]));
            std::future::ready(Err::<String, _>(failure))
        },
        &options,
    )
    .await;

    let error = outcome.expect_err("the cap rejects the delay");
    assert_eq!(
        error.message,
        "Server requested 277403s retry delay (max: 1s). Provider error: Some(429)"
    );
}

#[tokio::test(start_paused = true)]
async fn allows_disabling_the_provider_requested_retry_delay_cap() {
    let mut options = options();
    options.max_retries = 1;
    options.max_retry_delay_ms = Some(0);
    let mut calls = 0usize;
    let fut = retry_provider_request(
        move || {
            calls += 1;
            let first_failure = calls == 1;
            async move {
                if first_failure {
                    Err(provider_error(Some(429), Some(&[("retry-after", "2")][..])))
                } else {
                    Ok(String::from("ok"))
                }
            }
        },
        &options,
    );
    let mut fut = pin!(fut);

    assert!(poll(&mut fut).is_pending());
    tokio::time::advance(std::time::Duration::from_millis(1_999)).await;
    assert!(
        poll(&mut fut).is_pending(),
        "still inside the two-second wait"
    );
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    assert_eq![fut.await.as_deref(), Ok("ok")];
}

#[tokio::test]
async fn does_not_retry_errors_the_provider_marks_as_non_retryable() {
    let mut options = options();
    options.max_retries = 2;
    let error = provider_error(Some(429), Some(&[("x-should-retry", "false")][..]));
    let error_for_request = error.clone();
    let outcome: Result<String, ProviderRequestError> = retry_provider_request(
        move || {
            let failure = error_for_request.clone();
            std::future::ready(Err::<String, _>(failure))
        },
        &options,
    )
    .await;
    assert_eq!(outcome.expect_err("the marked error surfaces"), error);
}

#[tokio::test(start_paused = true)]
async fn aborts_a_provider_requested_retry_delay() {
    let mut options = options();
    options.max_retries = 2;
    options.max_retry_delay_ms = Some(0);
    let token = CancellationToken::new();
    options.signal = Some(token.clone());
    let fut = retry_provider_request(
        move || {
            let failure = provider_error(Some(429), Some(&[("retry-after", "277403")][..]));
            std::future::ready(Err::<String, _>(failure))
        },
        &options,
    );
    let mut fut = pin!(fut);

    assert!(
        poll(&mut fut).is_pending(),
        "the backoff sleep holds a pending timer"
    );
    token.cancel();
    let error = fut.await.expect_err("the abort surfaces");
    assert!(
        error.is_abort(),
        "expected the AbortError shape, got {error:?}"
    );
}

#[test]
fn the_classifier_follows_the_pinned_sdk_policy() {
    use pi_ai::utils::provider_retry::is_retryable_provider_error;

    // A status-less failure is transient; 408, 409, 429, and 5xx retry.
    for status in [
        None,
        Some(408),
        Some(409),
        Some(429),
        Some(500),
        Some(502),
        Some(503),
        Some(504),
    ] {
        assert!(
            is_retryable_provider_error(&provider_error(status, None)),
            "status {status:?} must retry"
        );
    }
    // Any other status does not.
    for status in [Some(400), Some(401), Some(404), Some(418), Some(499)] {
        assert!(
            !is_retryable_provider_error(&provider_error(status, None)),
            "status {status:?} must not retry"
        );
    }
    // The explicit header wins over the status.
    assert!(
        is_retryable_provider_error(&provider_error(
            Some(404),
            Some(&[("x-should-retry", "true")][..])
        )),
        "the header forces a retry"
    );
    assert!(
        !is_retryable_provider_error(&provider_error(
            Some(500),
            Some(&[("X-Should-Retry", "false")][..])
        )),
        "the header blocks the retry case-insensitively"
    );
}

#[test]
fn headers_read_case_insensitively_and_errors_display_their_message() {
    use pi_ai::utils::provider_retry::ProviderRequestError as Error;

    let error = provider_error(Some(429), Some(&[("Retry-After-Ms", "10")][..]));
    assert_eq![error.header("retry-after-ms"), Some("10")];
    assert_eq![error.header("RETRY-AFTER-MS"), Some("10")];
    assert_eq![error.header("missing"), None];

    let bare = Error::new(None, None, "gateway gone");
    assert_eq![bare.header("x-anything"), None];
    assert_eq![bare.to_string(), "gateway gone"];
    assert![
        !bare.is_abort(),
        "a messageful error is not the abort shape"
    ];

    let abort = Error::aborted();
    assert![abort.is_abort()];
    assert_eq![abort.to_string(), "Request aborted"];
}

#[test]
fn the_options_debug_with_their_retry_budget() {
    let mut options = options();
    let debug = format!("{options:?}");
    assert![debug.starts_with("ProviderRetryOptions"), "{debug}"];
    assert![debug.contains("max_retries: 0"), "{debug}"];
    assert![debug.contains("max_retry_delay_ms: None"), "{debug}"];
    options.max_retries = 3;
    assert![format!("{options:?}").contains("max_retries: 3")];
}

#[tokio::test]
async fn a_pre_cancelled_signal_aborts_right_after_the_first_failure() {
    let mut options = options();
    options.max_retries = 3;
    let token = CancellationToken::new();
    token.cancel();
    options.signal = Some(token);

    let mut request_count = 0usize;
    let outcome = retry_provider_request(
        || {
            request_count += 1;
            std::future::ready(Err::<String, _>(provider_error(Some(500), None)))
        },
        &options,
    )
    .await;
    let error = outcome.expect_err("the cancelled request aborts");
    assert![error.is_abort(), "expected the abort shape, got {error:?}"];
    assert_eq![request_count, 1, "the abort check precedes any retry"];
}

#[tokio::test(start_paused = true)]
async fn exponential_backoff_uses_the_injected_jitter() {
    let mut options = options();
    options.max_retries = 1;
    // A zero sample means the full exponential delay survives the jitter.
    options.random = Some(std::sync::Arc::new(|| 0.0));

    let mut calls = 0usize;
    let fut = retry_provider_request(
        move || {
            calls += 1;
            let first_failure = calls == 1;
            async move {
                if first_failure {
                    Err(provider_error(Some(500), None))
                } else {
                    Ok(String::from("ok"))
                }
            }
        },
        &options,
    );
    let mut fut = pin!(fut);

    // attempt 1: 0.5 * 2^0 * 1000 ms at multiplier 1.0.
    assert!(poll(&mut fut).is_pending(), "the 500 ms backoff holds");
    tokio::time::advance(std::time::Duration::from_millis(499)).await;
    assert!(poll(&mut fut).is_pending());
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    assert_eq![fut.await.as_deref(), Ok("ok")];
}

#[tokio::test(start_paused = true)]
async fn the_default_jitter_source_sleeps_within_the_exponential_budget() {
    let mut options = options();
    options.max_retries = 1;

    let mut first_request = true;
    let fut = retry_provider_request(
        || {
            let first_failure = std::mem::take(&mut first_request);
            async move {
                if first_failure {
                    Err(provider_error(Some(500), None))
                } else {
                    Ok(String::from("ok"))
                }
            }
        },
        &options,
    );
    let fut = pin!(fut);

    // The multiplier is at most 0.75, so a full second covers any sample;
    // a zero-length sample completes before the advance.
    tokio::time::advance(std::time::Duration::from_millis(1_000)).await;
    assert_eq![fut.await.as_deref(), Ok("ok")];
}
