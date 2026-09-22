//! The assistant-call retry port, from `test/retry.test.ts`. The classifier
//! cases feed message shapes the faux provider builds; the local helper
//! mirrors them.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::sync::{Arc, Mutex};

use pi_ai::types::StopReason;
use pi_ai::utils::retry::{
    RetryCallbacks, RetryPolicy, is_retryable_assistant_error, retry_assistant_call, retry_delay_ms,
};
use tokio_util::sync::CancellationToken;

const OPENAI_EXPLICIT_RETRY_MESSAGE: &str = "An error occurred while processing your request. You can retry your request, or contact us through our help center at help.openai.com if the error persists. Please include the request ID req_******** in your message.";
const BEDROCK_EXPLICIT_RETRY_MESSAGE: &str = "{\"message\":\"The system encountered an unexpected error during processing. Try your request again.\"}";
type FinishedLog = Arc<Mutex<Vec<(bool, u32, Option<String>)>>>;

const NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE: &str =
    "ResourceExhausted: Worker local total request limit reached (288/48)";
const BUN_FETCH_SOCKET_CLOSED_MESSAGE: &str = "The socket connection was closed unexpectedly. For more information, pass `verbose: true` in the second argument to fetch()";
const OPENAI_RESPONSES_EARLY_EOF_MESSAGE: &str =
    "OpenAI Responses stream ended before a terminal response event";
const WRAPPED_DNS_LOOKUP_ERROR: &str = "The pending stream has been canceled (caused by: getaddrinfo ENOTFOUND bedrock-runtime.us-east-1.amazonaws.com)";

fn error_message(text: &str) -> pi_ai::types::AssistantMessage {
    common::assistant_message_with(text, StopReason::Error, Some(text.to_owned()))
}

/// The attempt counter the produce closures bump, the (counter, capture)
/// pair the assertions read through.
fn produce_counter() -> (Arc<Mutex<usize>>, Arc<Mutex<usize>>) {
    let produce = Arc::new(Mutex::new(0usize));
    (produce.clone(), produce)
}

/// The scheduled counter paired with the callbacks that bump it, the
/// on-retry-scheduled fixture.
fn scheduled_callbacks(scheduled: &Arc<Mutex<usize>>) -> RetryCallbacks {
    let scheduled_for_callbacks = scheduled.clone();
    RetryCallbacks {
        on_retry_scheduled: Some(Arc::new(move |_, _, _, _| {
            *scheduled_for_callbacks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        })),
        ..Default::default()
    }
}

/// The finished log paired with the callbacks that append to it, the
/// on-retry-finished fixture.
fn finished_callbacks(finished: &FinishedLog) -> RetryCallbacks {
    let finished_for_callbacks = finished.clone();
    RetryCallbacks {
        on_retry_finished: Some(Arc::new(move |success, attempt, error| {
            finished_for_callbacks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((success, attempt, error));
        })),
        ..Default::default()
    }
}

/// The retry harness a case wires: the produce counter, its capture twin,
/// the schedule log, the finished log, and the callbacks.
type RetryCase = (
    Arc<Mutex<usize>>,
    Arc<Mutex<usize>>,
    Arc<Mutex<usize>>,
    FinishedLog,
    RetryCallbacks,
);

/// The retry harness: the produce counter, the callbacks that record
/// schedules and finishes, the fixture every retry case wires.
fn retry_case() -> RetryCase {
    let (produce, produce_for_call) = produce_counter();
    let scheduled = Arc::new(Mutex::new(0usize));
    let finished: FinishedLog = FinishedLog::default();
    let callbacks = scheduled_and_finished(&scheduled, &finished);
    (produce, produce_for_call, scheduled, finished, callbacks)
}

/// The settled no-retry outcome: one produce attempt, no schedule, and an
/// empty finish log, the assertions the policy cases repeat.
/// The produce turn the retry-until-success cases replay: "terminated"
/// errors for the first two attempts, then the "recovered" message.
fn recovered_turn(turns: u32) -> pi_ai::types::AssistantMessage {
    if turns < 3 {
        common::assistant_message_with("", StopReason::Error, Some(String::from("terminated")))
    } else {
        common::assistant_message("recovered")
    }
}

/// The settled success outcome: the "recovered" text, the assertions the
/// retry-until-success cases repeat.
fn assert_recovered_text(response: &pi_ai::types::AssistantMessage) {
    assert_eq![
        response.content,
        vec![pi_ai::types::AssistantBlock::Text(common::text_block(
            "recovered"
        ))]
    ];
}

fn assert_no_retry(
    response: &pi_ai::types::AssistantMessage,
    produce: &Arc<Mutex<usize>>,
    scheduled: &Arc<Mutex<usize>>,
    finished: &FinishedLog,
) {
    assert_eq!(response.stop_reason, StopReason::Error);
    assert_eq!(
        *produce
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        1
    );
    assert_eq!(
        *scheduled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        0
    );
    assert!(
        finished
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    );
}

/// The produce-schedule callback pair: the callbacks that record schedules
/// and the finished log both set.
fn scheduled_and_finished(scheduled: &Arc<Mutex<usize>>, finished: &FinishedLog) -> RetryCallbacks {
    let scheduled_for_callbacks = scheduled.clone();
    let finished_for_callbacks = finished.clone();
    RetryCallbacks {
        on_retry_scheduled: Some(Arc::new(move |_, _, _, _| {
            *scheduled_for_callbacks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        })),
        on_retry_finished: Some(Arc::new(move |success, attempt, error| {
            finished_for_callbacks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((success, attempt, error));
        })),
        ..Default::default()
    }
}

#[test]
fn matches_explicit_provider_retry_guidance() {
    assert!(is_retryable_assistant_error(&error_message(
        OPENAI_EXPLICIT_RETRY_MESSAGE
    )));
    assert!(is_retryable_assistant_error(&error_message(
        BEDROCK_EXPLICIT_RETRY_MESSAGE
    )));
    assert!(is_retryable_assistant_error(&error_message(
        NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE
    )));
}

#[test]
fn matches_bun_fetch_socket_drop_wording() {
    assert!(is_retryable_assistant_error(&error_message(
        BUN_FETCH_SOCKET_CLOSED_MESSAGE
    )));
}

#[test]
fn matches_upstream_request_buffer_exhaustion_wording() {
    assert!(is_retryable_assistant_error(&error_message(
        "Error: exceeded request buffer limit while retrying upstream"
    )));
}

#[test]
fn matches_dns_transport_failure_wording() {
    for message in [
        WRAPPED_DNS_LOOKUP_ERROR,
        "connect ENOTFOUND api.example.com",
        "EAI_AGAIN api.example.com",
        "getaddrinfo failed for api.example.com",
    ] {
        assert!(
            is_retryable_assistant_error(&error_message(message)),
            "{message}"
        );
    }
}

#[test]
fn matches_openai_responses_streams_that_end_before_terminal_events() {
    assert!(is_retryable_assistant_error(&error_message(
        OPENAI_RESPONSES_EARLY_EOF_MESSAGE
    )));
}

#[test]
fn keeps_provider_limit_errors_non_retryable() {
    assert!(!is_retryable_assistant_error(&error_message(
        "429 quota exceeded"
    )));
}

#[test]
fn classifies_assistant_error_messages() {
    assert!(is_retryable_assistant_error(&error_message(
        "overloaded_error"
    )));
    assert!(is_retryable_assistant_error(&error_message(
        "524 status code (no body)"
    )));
    assert!(!is_retryable_assistant_error(&common::assistant_message(
        "not an error"
    )));
}

#[test]
fn caps_agent_retry_delay() {
    // Regression for pi #8826.
    assert_eq!(
        retry_delay_ms(
            &RetryPolicy {
                enabled: true,
                max_retries: 0,
                base_delay_ms: 2_000,
                max_agent_delay_ms: None,
            },
            6
        ),
        60_000
    );
    assert_eq!(
        retry_delay_ms(
            &RetryPolicy {
                enabled: true,
                max_retries: 0,
                base_delay_ms: 2_000,
                max_agent_delay_ms: Some(5_000),
            },
            5
        ),
        5_000
    );
    assert_eq!(
        retry_delay_ms(
            &RetryPolicy {
                enabled: true,
                max_retries: 0,
                base_delay_ms: 2_000,
                max_agent_delay_ms: Some(0),
            },
            5
        ),
        0
    );
}

const DISABLED: RetryPolicy = RetryPolicy {
    enabled: false,
    max_retries: 3,
    base_delay_ms: 0,
    max_agent_delay_ms: None,
};
const ENABLED: RetryPolicy = RetryPolicy {
    enabled: true,
    max_retries: 3,
    base_delay_ms: 0,
    max_agent_delay_ms: None,
};

#[tokio::test]
async fn returns_a_successful_response_immediately_without_retrying() {
    let (produce, produce_for_call) = produce_counter();
    let response = retry_assistant_call(
        move || {
            *produce_for_call
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
            std::future::ready(common::assistant_message("ok"))
        },
        Some(&ENABLED),
        None,
        None,
    )
    .await;
    assert_eq![
        response.content[0],
        pi_ai::types::AssistantBlock::Text(common::text_block("ok"))
    ];
    assert_eq!(
        *produce
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        1
    );
}

#[tokio::test]
async fn does_not_retry_an_aborted_message() {
    let (produce, produce_for_call) = produce_counter();
    let scheduled = Arc::new(Mutex::new(0usize));
    let callbacks = scheduled_callbacks(&scheduled);
    let response = retry_assistant_call(
        move || {
            *produce_for_call
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
            std::future::ready(common::assistant_message_with(
                "",
                StopReason::Aborted,
                None,
            ))
        },
        Some(&ENABLED),
        None,
        Some(&callbacks),
    )
    .await;
    assert_eq!(response.stop_reason, StopReason::Aborted);
    assert_eq!(
        *produce
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        1
    );
    assert_eq!(
        *scheduled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        0
    );
}

#[tokio::test]
async fn does_not_retry_a_non_retryable_error() {
    let (produce, produce_for_call, scheduled, finished, callbacks) = retry_case();
    let response = retry_assistant_call(
        move || {
            *produce_for_call
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
            std::future::ready(common::assistant_message_with(
                "",
                StopReason::Error,
                Some(String::from("insufficient_quota")),
            ))
        },
        Some(&ENABLED),
        None,
        Some(&callbacks),
    )
    .await;
    assert_no_retry(&response, &produce, &scheduled, &finished);
}

#[tokio::test]
async fn retries_a_transient_error_up_to_max_retries_then_returns_the_final_error() {
    let (produce, produce_for_call) = produce_counter();
    let finished: FinishedLog = FinishedLog::default();
    let callbacks = finished_callbacks(&finished);
    let response = retry_assistant_call(
        move || {
            *produce_for_call
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
            std::future::ready(common::assistant_message_with(
                "",
                StopReason::Error,
                Some(String::from("terminated")),
            ))
        },
        Some(&ENABLED),
        None,
        Some(&callbacks),
    )
    .await;
    assert_eq!(response.stop_reason, StopReason::Error);
    assert_eq!(
        *produce
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        4
    );
    let finished = finished
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        finished.as_slice(),
        [(false, 3, Some(String::from("terminated")))]
    );
}

#[tokio::test]
async fn reports_capped_retry_delays() {
    // Regression for pi #8826.
    let policy = RetryPolicy {
        enabled: true,
        max_retries: 4,
        base_delay_ms: 10,
        max_agent_delay_ms: Some(15),
    };
    let count = Arc::new(Mutex::new(0u32));
    let count_for_call = count.clone();
    let delays: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let delays_for_callbacks = delays.clone();
    let callbacks = RetryCallbacks {
        on_retry_scheduled: Some(Arc::new(move |_, _, delay, _| {
            delays_for_callbacks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(delay);
        })),
        ..Default::default()
    };

    retry_assistant_call(
        move || {
            let mut n = count_for_call
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *n += 1;
            let turns = *n;
            drop(n);
            let value = if turns < 5 {
                common::assistant_message_with(
                    "",
                    StopReason::Error,
                    Some(String::from("terminated")),
                )
            } else {
                common::assistant_message("recovered")
            };
            std::future::ready(value)
        },
        Some(&policy),
        None,
        Some(&callbacks),
    )
    .await;

    assert_eq![
        delays
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        [10, 15, 15, 15]
    ];
}

#[tokio::test]
async fn stops_retrying_once_a_call_succeeds() {
    let count = Arc::new(Mutex::new(0u32));
    let count_for_call = count.clone();
    let finished: FinishedLog = FinishedLog::default();
    let callbacks = finished_callbacks(&finished);
    let response = retry_assistant_call(
        move || {
            let mut n = count_for_call
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *n += 1;
            let turns = *n;
            drop(n);
            std::future::ready(recovered_turn(turns))
        },
        Some(&ENABLED),
        None,
        Some(&callbacks),
    )
    .await;
    assert_recovered_text(&response);
    assert_eq!(
        *count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        3
    );
    assert_eq![
        finished
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        [(true, 2, None)]
    ];
}

#[tokio::test]
async fn reports_an_aborted_retried_call_as_unsuccessful() {
    let count = Arc::new(Mutex::new(0u32));
    let count_for_call = count.clone();
    let finished: FinishedLog = FinishedLog::default();
    let callbacks = finished_callbacks(&finished);
    let response = retry_assistant_call(
        move || {
            let mut n = count_for_call
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *n += 1;
            let turns = *n;
            drop(n);
            std::future::ready(if turns == 1 {
                common::assistant_message_with(
                    "",
                    StopReason::Error,
                    Some(String::from("terminated")),
                )
            } else {
                common::assistant_message_with("", StopReason::Aborted, None)
            })
        },
        Some(&ENABLED),
        None,
        Some(&callbacks),
    )
    .await;
    assert_eq!(response.stop_reason, StopReason::Aborted);
    assert_eq!(
        *count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        2
    );
    assert_eq![
        finished
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        [(false, 1, None)]
    ];
}

#[tokio::test]
async fn does_not_retry_when_policy_is_disabled() {
    let (produce, produce_for_call, scheduled, finished, callbacks) = retry_case();
    let response = retry_assistant_call(
        move || {
            *produce_for_call
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
            std::future::ready(common::assistant_message_with(
                "",
                StopReason::Error,
                Some(String::from("terminated")),
            ))
        },
        Some(&DISABLED),
        None,
        Some(&callbacks),
    )
    .await;
    assert_no_retry(&response, &produce, &scheduled, &finished);
}

#[tokio::test]
async fn emits_on_retry_attempt_start_after_backoff_before_each_retried_call() {
    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let count = Arc::new(Mutex::new(0u32));
    let count_for_call = count.clone();
    let events_for_produce = events.clone();
    let events_for_scheduled = events.clone();
    let events_for_start = events.clone();
    let callbacks = RetryCallbacks {
        on_retry_scheduled: Some(Arc::new(move |attempt, _, _, _| {
            push_event(&events_for_scheduled, format!("retry:{attempt}"));
        })),
        on_retry_attempt_start: Some(Arc::new(move || {
            push_event(&events_for_start, String::from("attempt-start"));
        })),
        ..Default::default()
    };
    let response = retry_assistant_call(
        move || {
            let mut n = count_for_call
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let tag = format!("produce:{n}");
            *n += 1;
            let turns = *n;
            drop(n);
            push_event(&events_for_produce, tag);
            std::future::ready(recovered_turn(turns))
        },
        Some(&ENABLED),
        None,
        Some(&callbacks),
    )
    .await;
    assert_recovered_text(&response);
    assert_eq![
        events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        [
            "produce:0",
            "retry:1",
            "attempt-start",
            "produce:1",
            "retry:2",
            "attempt-start",
            "produce:2",
        ]
    ];
}

#[tokio::test]
async fn aborts_backoff_sleep_via_signal_and_returns_an_aborted_message() {
    let token = CancellationToken::new();
    let (produce, produce_for_call) = produce_counter();
    let _produce_wait = produce.clone();
    let policy = RetryPolicy {
        enabled: true,
        max_retries: 5,
        base_delay_ms: 10_000,
        max_agent_delay_ms: None,
    };
    let finished: FinishedLog = FinishedLog::default();
    let callbacks = finished_callbacks(&finished);
    let task = tokio::spawn({
        let token = token.clone();
        async move {
            retry_assistant_call(
                move || {
                    *produce_for_call
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                    std::future::ready(common::assistant_message_with(
                        "",
                        StopReason::Error,
                        Some(String::from("terminated")),
                    ))
                },
                Some(&policy),
                Some(&token),
                Some(&callbacks),
            )
            .await
        }
    });
    // Let one error call resolve and the first backoff sleep start, then abort.
    while *produce
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        == 0
    {
        tokio::task::yield_now().await;
    }
    token.cancel();
    let response = task.await.expect("the retry task");
    assert_eq!(response.stop_reason, StopReason::Aborted);
    assert_eq!(response.error_message, None);
    assert_eq!(
        *produce
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        1
    );
    assert_eq![
        finished
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_slice(),
        [(false, 1, Some(String::from("terminated")))]
    ];
}

fn push_event(events: &Mutex<Vec<String>>, entry: String) {
    events
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(entry);
}

#[test]
fn an_error_stop_without_a_message_is_not_retryable() {
    let mut message = common::assistant_message("failed");
    message.stop_reason = StopReason::Error;
    message.error_message = None;
    assert![!is_retryable_assistant_error(&message)];
}

#[test]
fn the_callbacks_debug_with_and_without_hooks() {
    let bare = RetryCallbacks::default();
    let debug = format!("{bare:?}");
    assert![debug.contains("on_retry_scheduled: false"), "{debug}"];
    assert![debug.contains("on_retry_attempt_start: false"), "{debug}"];
    assert![debug.contains("on_retry_finished: false"), "{debug}"];

    let callbacks = RetryCallbacks {
        on_retry_scheduled: Some(Arc::new(|_, _, _, _| {})),
        on_retry_attempt_start: Some(Arc::new(|| {})),
        on_retry_finished: Some(Arc::new(|_, _, _| {})),
    };
    let debug = format!("{callbacks:?}");
    assert![debug.contains("on_retry_scheduled: true"), "{debug}"];
    assert![debug.contains("on_retry_attempt_start: true"), "{debug}"];
    assert![debug.contains("on_retry_finished: true"), "{debug}"];
}

#[tokio::test]
async fn a_retry_backoff_completes_when_the_signal_stays_open() {
    let token = CancellationToken::new();
    let attempts = Arc::new(Mutex::new(0u32));
    let attempts_for_call = Arc::clone(&attempts);
    let policy = RetryPolicy {
        enabled: true,
        max_retries: 3,
        base_delay_ms: 1,
        max_agent_delay_ms: None,
    };
    let response = retry_assistant_call(
        move || {
            let turn = {
                let mut n = attempts_for_call
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                *n += 1;
                *n
            };
            std::future::ready(if turn < 2 {
                common::assistant_message_with(
                    "",
                    StopReason::Error,
                    Some(String::from("terminated")),
                )
            } else {
                common::assistant_message("recovered")
            })
        },
        Some(&policy),
        Some(&token),
        None,
    )
    .await;
    assert_eq![response.stop_reason, StopReason::Stop];
    assert_eq!(
        *attempts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        2
    );
}
