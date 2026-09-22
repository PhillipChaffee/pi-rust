//! OpenAI Completions provider retries, ported from
//! `packages/ai/test/openai-completions-retry.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: upstream's `vi.mock("openai")` client collapses onto
//! the [`MockHttpClient`] seam — each SDK call is one mock execute, and the
//! scripted failures are real non-2xx responses whose `retry-after-ms` and
//! `retry-after` headers the retry loop reads. The fake timers become the
//! tokio paused clock.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::openai_completions::{OpenAiCompletionsOptions, stream};
use pi_ai::http::{MockHttpClient, MockResponse};
use pi_ai::types::{Context, Message, StopReason, UserBlock, UserContent, UserMessage};
use serde_json::json;

mod common;
use common::{openai_completions_model, openai_mock, openai_sse_response, recorded_body_at};

/// The upstream context: empty system prompt, one text-block user turn, an
/// empty tool list.
fn retry_context() -> Context {
    Context {
        system_prompt: Some(String::new()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![UserBlock::Text(pi_ai::types::TextContent {
                text: "hi".to_owned(),
                text_signature: None,
            })]),
            timestamp: 0,
        })],
        tools: Some(Vec::new()),
    }
}

/// A non-2xx response carrying the provider text and a retry-delay header.
fn error_response(
    status: u16,
    header: (&'static str, &'static str),
    body: &'static str,
) -> MockResponse {
    MockResponse::status(status)
        .with_header(header.0, header.1)
        .with_body(body)
}

/// The successful SSE run the mock answers on the attempt that gets through.
fn done_chunks() -> Vec<serde_json::Value> {
    vec![
        json!({
            "id": "chatcmpl-test",
            "choices": [{ "index": 0, "delta": { "content": "ok" } }],
        }),
        json!({
            "id": "chatcmpl-test",
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
        }),
    ]
}

/// The keyed options with the retry budget a suite drives.
fn retry_options(
    mock: &MockHttpClient,
    max_retries: u32,
    max_retry_delay_ms: u64,
) -> OpenAiCompletionsOptions {
    OpenAiCompletionsOptions {
        transport_options: common::mock_transport(mock),
        api_key: Some("test".to_owned()),
        max_retries: Some(max_retries),
        max_retry_delay_ms: Some(max_retry_delay_ms),
        ..OpenAiCompletionsOptions::default()
    }
}

/// Spin until the mock has served `expected` attempts; the woken attempt
/// needs scheduling turns the advance does not spend.
async fn wait_for_attempts(mock: &MockHttpClient, expected: usize) {
    for _ in 0..256 {
        if mock.request_count() >= expected {
            return;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(mock.request_count(), expected);
}

/// Stream to settlement, the shape upstream's `consume` helper drains.
async fn consume(options: &OpenAiCompletionsOptions) -> pi_ai::types::AssistantMessage {
    let model = openai_completions_model();
    stream(&model, &retry_context(), Some(options))
        .result()
        .await
}

// ---------------------------------------------------------------------------
// SDK retries disabled
// ---------------------------------------------------------------------------

/// Upstream pins `maxRetries: 0` on the SDK call; the port's equivalent is
/// one mock execute per attempt, so the default options dispatch exactly one
/// request that carries the shared request body.
#[tokio::test]
async fn sdk_retries_are_disabled_by_default() {
    let mock = openai_mock(&done_chunks());
    let options = retry_options(&mock, 0, 0);

    let result = consume(&options).await;

    assert_eq!(mock.request_count(), 1);
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.error_message, None);
    let body = recorded_body_at(&mock, 0);
    assert_eq!(body["model"], json!("test-model"));
    assert_eq!(body["stream"], json!(true));
}

// ---------------------------------------------------------------------------
// Provider retries honored
// ---------------------------------------------------------------------------

/// The two provider-requested 100 ms waits space three attempts: at 99 ms the
/// second attempt has not fired, at 100 ms it has, and the same spacing holds
/// before the third. `vi.advanceTimersByTimeAsync` becomes the tokio paused
/// clock's `advance`.
#[tokio::test(start_paused = true)]
async fn honors_provider_retries_while_keeping_sdk_retries_disabled() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond_sequence(vec![
            error_response(429, ("retry-after-ms", "100"), "rate limited"),
            error_response(500, ("retry-after-ms", "100"), "server error"),
            openai_sse_response(&done_chunks()),
        ]);
    let options = retry_options(&mock, 2, 100);
    let stream = stream(
        &openai_completions_model(),
        &retry_context(),
        Some(&options),
    );

    // The result waiter joins the drive so it registers before the stream
    // task can settle and end. Advancing only wakes the due timer — the
    // woken attempt needs a few scheduling turns to run.
    let ((), result) = tokio::join!(
        async {
            // Drive the first attempt without moving the clock; the retry
            // sleep registers at the current (paused) instant.
            let mut spins = 0;
            while mock.request_count() == 0 {
                spins += 1;
                assert!(spins < 128, "the first attempt never dispatched");
                tokio::task::yield_now().await;
            }
            assert_eq!(mock.request_count(), 1);
            tokio::time::advance(std::time::Duration::from_millis(99)).await;
            assert_eq!(mock.request_count(), 1);
            tokio::time::advance(std::time::Duration::from_millis(1)).await;
            wait_for_attempts(&mock, 2).await;
            tokio::time::advance(std::time::Duration::from_millis(99)).await;
            assert_eq!(mock.request_count(), 2);
            tokio::time::advance(std::time::Duration::from_millis(1)).await;
            wait_for_attempts(&mock, 3).await;
        },
        stream.result(),
    );

    assert_eq!(mock.request_count(), 3);
    assert_eq!(result.stop_reason, StopReason::Stop);
    // Every attempt is one mock execute; the retried attempts carry the same
    // request the first one did.
    for index in 0..3 {
        let body = recorded_body_at(&mock, index);
        assert_eq!(body["model"], json!("test-model"));
    }
}

// ---------------------------------------------------------------------------
// Retry-delay cap
// ---------------------------------------------------------------------------

/// A server-requested delay above the cap fails immediately with the cap
/// message and the provider's original text, and spends only the first
/// attempt.
#[tokio::test]
async fn fails_immediately_when_a_provider_requested_retry_delay_exceeds_the_limit() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(error_response(
            429,
            ("retry-after", "277403"),
            "rate limited",
        ));
    let options = retry_options(&mock, 2, 1000);

    let result = consume(&options).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the cap failure message");
    assert!(
        message.contains("Server requested 277403s retry delay (max: 1s)"),
        "got: {message}"
    );
    assert!(message.contains("rate limited"), "got: {message}");
    assert_eq!(mock.request_count(), 1);
}
