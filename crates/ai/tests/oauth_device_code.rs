//! The OAuth device-code poll loop, from `test/oauth-device-code.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the tokio paused
//! clock replaces fake timers, and [`pi_ai::auth::clock::FixedClock`] reads
//! the poll times as the epoch values `vi.setSystemTime` pinned upstream.
//! The flow is driven by hand — poll, advance, poll — the paused-clock
//! idiom the provider-retry suite established, so no advance can run past
//! an assertion point.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected shape panics by design"
)]

mod common;

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use pi_ai::auth::clock::{AuthClock as _, FixedClock};
use pi_ai::auth::oauth::device_code::{PollOptions, PollOutcome, poll_oauth_device_code_flow};
use pi_ai::auth::types::AuthError;
use pi_ai::types::BoxedFuture;
use tokio_util::sync::CancellationToken;

use common::paused_clock::advance;

/// `new Date("2026-03-09T00:00:00Z").getTime()`, upstream's pinned system
/// time.
const START: i64 = 1_773_014_400_000;

/// Poll the pinned flow once with a noop waker, stepping the paused clock
/// between polls like `advanceTimersByTimeAsync` does.
fn poll<F: Future>(flow: &mut std::pin::Pin<&mut F>) -> Poll<F::Output> {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    flow.as_mut().poll(&mut context)
}

/// A poll step recording the epoch time of each call — the
/// `pollTimes.push(Date.now())` upstream's stubs do — and answering from
/// `outcomes` in order.
fn recording_poll(
    clock: &Arc<FixedClock>,
    poll_times: &Arc<Mutex<Vec<i64>>>,
    outcomes: &[PollOutcome<String>],
) -> Arc<dyn Fn() -> BoxedFuture<'static, Result<PollOutcome<String>, AuthError>> + Send + Sync> {
    let clock = Arc::clone(clock);
    let poll_times = Arc::clone(poll_times);
    let outcomes = Arc::new(Mutex::new(outcomes.to_vec()));
    Arc::new(move || {
        let clock = Arc::clone(&clock);
        let poll_times = Arc::clone(&poll_times);
        let outcomes = Arc::clone(&outcomes);
        Box::pin(async move {
            poll_times.lock().expect("poll times").push(clock.now_ms());
            let mut queued = outcomes.lock().expect("queued outcomes");
            assert!(!queued.is_empty(), "Unexpected extra poll");
            Ok(queued.remove(0))
        })
    })
}

fn poll_times() -> Arc<Mutex<Vec<i64>>> {
    Arc::new(Mutex::new(Vec::new()))
}

/// The options a fixed-schedule case drives: the recorded poll over `clock`
/// answering `outcomes`, the interval and lifetime the server reports.
fn fixed_poll_options(
    clock: &Arc<FixedClock>,
    poll_times: &Arc<Mutex<Vec<i64>>>,
    outcomes: &[PollOutcome<String>],
    interval_seconds: Option<u64>,
    expires_in_seconds: Option<u64>,
    wait_before_first_poll: bool,
) -> PollOptions<String> {
    PollOptions {
        interval_seconds,
        expires_in_seconds,
        wait_before_first_poll,
        signal: CancellationToken::new(),
        poll: recording_poll(clock, poll_times, outcomes),
    }
}

#[tokio::test(start_paused = true)]
async fn polls_immediately_and_returns_the_completed_value() {
    let clock = Arc::new(FixedClock::new(START));
    let poll_times = poll_times();
    let flow = pin!(poll_oauth_device_code_flow(fixed_poll_options(
        &clock,
        &poll_times,
        &[
            PollOutcome::Pending,
            PollOutcome::Complete("token".to_owned()),
        ],
        Some(2),
        Some(30),
        false,
    )));
    let mut flow = flow;

    // The first poll runs immediately, upstream's advanceTimersByTimeAsync(0).
    assert!(
        poll(&mut flow).is_pending(),
        "the first pending answer leaves the flow waiting"
    );
    assert_eq!(*poll_times.lock().expect("poll times"), vec![START]);

    advance(&clock, 1_999).await;
    assert!(
        poll(&mut flow).is_pending(),
        "still inside the two-second wait"
    );
    assert_eq!(*poll_times.lock().expect("poll times"), vec![START]);

    advance(&clock, 1).await;
    let Poll::Ready(Ok(value)) = poll(&mut flow) else {
        panic!("the second poll completes the flow");
    };
    assert_eq!(value, "token");
    assert_eq!(
        *poll_times.lock().expect("poll times"),
        vec![START, START + 2_000]
    );
}

#[tokio::test(start_paused = true)]
async fn can_wait_before_the_first_poll() {
    let clock = Arc::new(FixedClock::new(START));
    let poll_times = poll_times();
    let flow = pin!(poll_oauth_device_code_flow(fixed_poll_options(
        &clock,
        &poll_times,
        &[PollOutcome::Complete("token".to_owned())],
        Some(2),
        Some(30),
        true,
    )));
    let mut flow = flow;

    assert!(
        poll(&mut flow).is_pending(),
        "the wait before the first poll holds"
    );
    advance(&clock, 1_999).await;
    assert!(
        poll(&mut flow).is_pending(),
        "still inside the initial wait"
    );
    assert_eq!(
        *poll_times.lock().expect("poll times"),
        Vec::<i64>::new(),
        "nothing polls before the wait elapses"
    );

    advance(&clock, 1).await;
    let Poll::Ready(Ok(value)) = poll(&mut flow) else {
        panic!("the first poll completes the flow");
    };
    assert_eq!(value, "token");
    assert_eq!(*poll_times.lock().expect("poll times"), vec![START + 2_000]);
}

#[tokio::test(start_paused = true)]
async fn increases_the_interval_by_5_seconds_after_slow_down_without_a_server_interval() {
    let clock = Arc::new(FixedClock::new(START));
    let poll_times = poll_times();
    let flow = pin!(poll_oauth_device_code_flow(fixed_poll_options(
        &clock,
        &poll_times,
        &[
            PollOutcome::SlowDown {
                interval_seconds: None
            },
            PollOutcome::Complete("token".to_owned()),
        ],
        Some(2),
        Some(900),
        false,
    )));
    let mut flow = flow;

    assert!(
        poll(&mut flow).is_pending(),
        "the slow_down leaves the flow waiting"
    );
    assert_eq!(*poll_times.lock().expect("poll times"), vec![START]);

    advance(&clock, 6_999).await;
    assert!(
        poll(&mut flow).is_pending(),
        "still inside the raised seven-second wait"
    );
    assert_eq!(*poll_times.lock().expect("poll times"), vec![START]);

    advance(&clock, 1).await;
    let Poll::Ready(Ok(value)) = poll(&mut flow) else {
        panic!("the second poll completes the flow");
    };
    assert_eq!(value, "token");
    assert_eq!(
        *poll_times.lock().expect("poll times"),
        vec![START, START + 7_000]
    );
}

#[tokio::test(start_paused = true)]
async fn honors_a_server_provided_slow_down_interval() {
    let clock = Arc::new(FixedClock::new(START));
    let poll_times = poll_times();
    let flow = pin!(poll_oauth_device_code_flow(fixed_poll_options(
        &clock,
        &poll_times,
        &[
            PollOutcome::SlowDown {
                interval_seconds: Some(30),
            },
            PollOutcome::Complete("token".to_owned()),
        ],
        Some(2),
        Some(900),
        false,
    )));
    let mut flow = flow;

    assert!(
        poll(&mut flow).is_pending(),
        "the slow_down leaves the flow waiting"
    );
    assert_eq!(*poll_times.lock().expect("poll times"), vec![START]);

    advance(&clock, 29_999).await;
    assert!(
        poll(&mut flow).is_pending(),
        "still inside the server-required thirty-second wait"
    );
    assert_eq!(*poll_times.lock().expect("poll times"), vec![START]);

    advance(&clock, 1).await;
    let Poll::Ready(Ok(value)) = poll(&mut flow) else {
        panic!("the second poll completes the flow");
    };
    assert_eq!(value, "token");
    assert_eq!(
        *poll_times.lock().expect("poll times"),
        vec![START, START + 30_000]
    );
}

#[tokio::test(start_paused = true)]
async fn cancels_an_in_flight_wait() {
    let clock = Arc::new(FixedClock::new(START));
    let poll_times = poll_times();
    let signal = CancellationToken::new();
    let flow = pin!(poll_oauth_device_code_flow(PollOptions {
        interval_seconds: Some(5),
        expires_in_seconds: Some(30),
        wait_before_first_poll: false,
        signal: signal.clone(),
        poll: recording_poll(&clock, &poll_times, &[PollOutcome::Pending]),
    }));
    let mut flow = flow;

    assert!(
        poll(&mut flow).is_pending(),
        "the flow parks inside its wait"
    );
    signal.cancel();
    let Poll::Ready(Err(error)) = poll(&mut flow) else {
        panic!("the cancel ends the flow");
    };
    assert_eq!(
        error.to_string(),
        "Login cancelled",
        "the abort surfaces as the cancelled login"
    );
}
