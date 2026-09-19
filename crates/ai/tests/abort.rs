//! Rust-native suites for the abort plumbing (`abort.ts`,
//! `abort-signals.ts`, `sleep.ts`): upstream's coverage rides the
//! credential-gated live-provider abort suites, so these pin the
//! `CancellationToken` semantics directly.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use pi_ai::utils::abort::{RaceError, operation_signal, race_with_abort_signal};
use pi_ai::utils::abort_signals::combine_abort_signals;
use pi_ai::utils::sleep::sleep;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn operation_signal_passes_the_caller_token_through() {
    let token = CancellationToken::new();
    let operation = operation_signal(Some(&token));
    assert_eq!(token.is_cancelled(), operation.is_cancelled());
    token.cancel();
    assert!(operation.is_cancelled());
}

#[tokio::test]
async fn operation_signal_mints_a_fresh_token_when_none_is_supplied() {
    let token = operation_signal(None);
    assert!(!token.is_cancelled());
}

#[tokio::test]
async fn a_race_rejects_immediately_when_the_signal_is_already_aborted() {
    let token = CancellationToken::new();
    token.cancel();
    let ran = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ran_for_operation = ran.clone();
    let outcome = race_with_abort_signal(
        async move {
            ran_for_operation.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok::<_, String>(1)
        },
        &token,
    )
    .await;
    assert_eq![
        outcome,
        Err(RaceError::Aborted(pi_ai::utils::abort::AbortError))
    ];
    assert!(
        !ran.load(std::sync::atomic::Ordering::SeqCst),
        "the abandoned operation is dropped, not abandoned-running"
    );
}

#[tokio::test]
async fn a_race_resolves_with_the_operation_value() {
    let token = CancellationToken::new();
    let outcome =
        race_with_abort_signal(async { Ok::<_, String>(String::from("value")) }, &token).await;
    assert_eq![outcome, Ok(String::from("value"))];
}

#[tokio::test]
async fn a_race_rejects_with_the_operation_error() {
    let token = CancellationToken::new();
    let outcome =
        race_with_abort_signal(async { Err::<u32, _>(String::from("failed")) }, &token).await;
    assert_eq![outcome, Err(RaceError::Operation(String::from("failed")))];
}

#[tokio::test]
async fn a_race_rejects_when_the_signal_aborts_first() {
    let token = CancellationToken::new();
    let released = std::sync::Arc::new(tokio::sync::Notify::new());
    let released_for_operation = released.clone();
    let outcome = race_with_abort_signal(
        async move {
            released_for_operation.notified().await;
            Ok::<_, String>(0u32)
        },
        &token,
    );
    tokio::pin!(outcome);

    // Let the race register, then abort; the operation never settles.
    tokio::task::yield_now().await;
    token.cancel();
    let error = outcome.await;
    assert_eq![
        error,
        Err(RaceError::Aborted(pi_ai::utils::abort::AbortError))
    ];
}

#[tokio::test]
async fn a_combination_cancels_when_any_source_cancels() {
    let first = CancellationToken::new();
    let second = CancellationToken::new();
    let combined = combine_abort_signals(&[Some(&first), Some(&second)]);
    assert![format!("{combined:?}").starts_with("CombinedAbortSignal")];
    let token = combined.signal.expect("two active sources combine");

    assert!(!token.is_cancelled());
    first.cancel();
    // The linking task needs one poll to forward the cancellation.
    tokio::task::yield_now().await;
    assert!(
        token.is_cancelled(),
        "the first source cancels the combination"
    );

    let third = CancellationToken::new();
    let combined_two = combine_abort_signals(&[Some(&third)]);
    // A single active source passes through unchanged.
    assert!(combined_two.signal.is_some());
    assert!(
        !combined_two
            .signal
            .as_ref()
            .expect("passed through")
            .is_cancelled()
    );

    let cleanup = combine_abort_signals(&[]);
    assert![cleanup.signal.is_none()];
    (cleanup.cleanup)();
}

#[tokio::test]
async fn a_combination_stops_observing_after_cleanup() {
    let source = CancellationToken::new();
    let other = CancellationToken::new();
    let combined = combine_abort_signals(&[Some(&source), Some(&other)]);
    let token = combined.signal.expect("two sources combine");
    (combined.cleanup)();
    source.cancel();
    tokio::task::yield_now().await;
    assert!(!token.is_cancelled(), "the cleanup detached the source");
}

#[tokio::test]
async fn the_abortable_sleep_completes_when_nothing_aborts() {
    let token = CancellationToken::new();
    sleep(1, &token).await.expect("the sleep completes");
}

#[tokio::test]
async fn the_sleep_stops_early_when_the_signal_aborts() {
    let token = CancellationToken::new();
    let sleeper = sleep(10_000, &token);
    tokio::pin!(sleeper);
    token.cancel();
    let error = sleeper.await;
    assert_eq![error, Err(pi_ai::utils::abort::AbortError)];
    assert_eq![pi_ai::utils::abort::AbortError::NAME, "AbortError"];
}

#[test]
fn the_abort_and_race_errors_display_like_the_wire_errors() {
    let abort = pi_ai::utils::abort::AbortError;
    assert_eq![abort.to_string(), pi_ai::utils::abort::AbortError::MESSAGE];
    assert_eq![abort.to_string(), "The operation was aborted"];

    let operation: RaceError<std::io::Error> = RaceError::Operation(std::io::Error::other("boom"));
    assert_eq![operation.to_string(), "boom"];
    assert_eq![
        std::error::Error::source(&operation).map(ToString::to_string),
        Some(String::from("boom"))
    ];

    let aborted: RaceError<std::io::Error> = RaceError::Aborted(pi_ai::utils::abort::AbortError);
    assert_eq![aborted.to_string(), "The operation was aborted"];
    assert_eq![
        std::error::Error::source(&aborted).map(ToString::to_string),
        Some(String::from("The operation was aborted"))
    ];
    assert_eq![format!("{aborted:?}"), "Aborted(AbortError)"];
}

#[tokio::test]
async fn a_single_source_combination_stops_observing_after_cleanup() {
    let source = CancellationToken::new();
    let combined = combine_abort_signals(&[Some(&source)]);
    // The pass-through signal shares the source's cancellation...
    let token = combined.signal.expect("one active source passes through");
    assert![!token.is_cancelled()];
    source.cancel();
    tokio::task::yield_now().await;
    assert![token.is_cancelled()];
    // ...and its cleanup is a no-op like the empty combination's.
    (combined.cleanup)();
}
