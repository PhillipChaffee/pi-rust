//! Env-gated thinking-disable E2E suites, ported from the upstream
//! `google-thinking-disable.test.ts` probes — the `Anthropic thinking
//! disable E2E`, `Google thinking disable E2E`, `Google Vertex thinking
//! disable E2E`, `OpenAI thinking disable E2E`, and `OpenRouter thinking
//! disable E2E` describes — at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Like upstream's `describe.skipIf`, a probe without its provider credential
//! returns early. Upstream rides `streamSimple` with no `reasoning`, whose
//! per-adapter mapping pins thinking off; the port rides [`live::stream`]
//! with `reasoning` unset, and the Google family additionally carries the
//! disabled [`GoogleThinkingControl`] upstream's simple surface resolves,
//! config and leave Gemini's provider default (thinking on) in force.

#![expect(
    clippy::expect_used,
    reason = "the tests pin live outcomes; an unexpected shape panics the test by design"
)]
// binary and carries its own expects only for the lints its own code knows
// it trips; these cover the rest so the suite's gate runs clean without
// editing the shared file. Stale entries fail the build on their own.

use regex::Regex;

use pi_ai::api::google_shared::GoogleThinkingControl;
use pi_ai::types::{AssistantBlock, AssistantMessageEvent, Context, Model, StopReason};

mod common;
use common::live;

/// The probe result, upstream's `RunResult`.
struct RunResult {
    /// Thinking events the stream emitted.
    thinking_event_count: u64,
    /// Thinking characters the deltas carried.
    thinking_char_count: usize,
    /// The trimmed reply text.
    text: String,
    /// The response's output-token usage.
    output_tokens: u64,
    /// The final content's block types, upstream's `contentTypes`.
    content_types: Vec<&'static str>,
}

/// The per-probe expectations, upstream's `DisableExpectations`.
struct DisableExpectations {
    /// The request options the probe rides, upstream's `requestOptions`.
    request_options: live::LiveOptions,
    /// The pong floor, upstream's `minPongs`.
    min_pongs: usize,
    /// The output-token ceiling, upstream's `maxOutputTokens`.
    max_output_tokens: Option<u64>,
}

impl Default for DisableExpectations {
    fn default() -> Self {
        Self {
            request_options: live::LiveOptions::default(),
            min_pongs: 35,
            max_output_tokens: None,
        }
    }
}

/// The shared probe context, upstream's `makeContext`.
fn make_context() -> Context {
    Context {
        system_prompt: Some(
            "You are a precise assistant. Follow the requested output format exactly.".to_owned(),
        ),
        messages: vec![live::user_message(
            "Before replying, carefully solve 36863 * 5279 internally. Then reply with the word \
             pong repeated exactly 40 times, separated by single spaces. Do not add any other \
             text.",
        )],
        tools: None,
    }
}

/// The word-bounded pong counter, upstream's `countPongs`.
fn count_pongs(text: &str) -> usize {
    Regex::new(r"(?i)\bpong\b")
        .expect("the pong pattern compiles")
        .find_iter(text)
        .count()
}

/// The disabled thinking control, upstream's `streamSimple` mapping of an
/// absent `reasoning` to `thinking: { enabled: false }`.
fn thinking_off() -> GoogleThinkingControl {
    GoogleThinkingControl {
        enabled: false,
        ..GoogleThinkingControl::default()
    }
}

/// The shared probe run, upstream's `runWithoutReasoning`: stream with the
/// probe's request options and no reasoning level, count the thinking
/// events, and settle into the response shape.
async fn run_without_reasoning(model: &Model, options: &live::LiveOptions) -> RunResult {
    let stream = live::stream(model, &make_context(), options);

    let mut thinking_event_count = 0u64;
    let mut thinking_char_count = 0usize;
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::ThinkingStart { .. }
            | AssistantMessageEvent::ThinkingEnd { .. } => thinking_event_count += 1,
            AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                thinking_event_count += 1;
                thinking_char_count += delta.chars().count();
            }
            _ => {}
        }
    }

    let response = stream.result().await;
    assert_eq!(
        response.stop_reason,
        StopReason::Stop,
        "{}: {:?}",
        model.id,
        response.error_message
    );

    RunResult {
        thinking_event_count,
        thinking_char_count,
        text: live::response_text(&response).trim().to_owned(),
        output_tokens: response.usage.output,
        content_types: response
            .content
            .iter()
            .map(|block| match block {
                AssistantBlock::Text(_) => "text",
                AssistantBlock::Thinking(_) => "thinking",
                AssistantBlock::ToolCall(_) => "toolCall",
            })
            .collect(),
    }
}

/// The shared assertion, upstream's `expectThinkingDisabledE2E`: zero
/// thinking events, no thinking block in the final content, and at least the
/// expected pong floor.
async fn expect_thinking_disabled_e2e(model: &Model, expectations: &DisableExpectations) {
    let result = run_without_reasoning(model, &expectations.request_options).await;

    assert_eq!(result.thinking_event_count, 0, "no thinking events");
    assert_eq!(result.thinking_char_count, 0, "no thinking characters");
    assert!(
        !result.content_types.contains(&"thinking"),
        "no thinking block in the final content: {:?}",
        result.content_types
    );
    assert!(
        count_pongs(&result.text) >= expectations.min_pongs,
        "pong floor not reached: {:?}",
        result.text
    );
    if let Some(max_output_tokens) = expectations.max_output_tokens {
        assert!(
            result.output_tokens < max_output_tokens,
            "output tokens {} exceed the ceiling",
            result.output_tokens
        );
    }
}

/// Upstream describe `Anthropic thinking disable E2E`, it "disables thinking
/// for budget-based reasoning models": Claude Sonnet 4.5 with thinking off
/// emits no thinking events and answers with the pong floor. Upstream's
/// `maxTokens: 320, temperature: 0` requestOptions cannot ride
/// [`live::LiveOptions`], so the probe rides the model cap.
#[tokio::test]
async fn anthropic_disables_thinking_for_budget_based_reasoning_models() {
    let Some(api_key) = live::env_key("ANTHROPIC_API_KEY") else {
        return;
    };
    let expectations = DisableExpectations {
        request_options: live::LiveOptions::key(api_key),
        ..DisableExpectations::default()
    };
    expect_thinking_disabled_e2e(
        &live::model("anthropic", "claude-sonnet-4-5"),
        &expectations,
    )
    .await;
}

/// Upstream describe `Anthropic thinking disable E2E`, it "disables thinking
/// for adaptive reasoning models": Claude Sonnet 4.6, same expectations.
#[tokio::test]
async fn anthropic_disables_thinking_for_adaptive_reasoning_models() {
    let Some(api_key) = live::env_key("ANTHROPIC_API_KEY") else {
        return;
    };
    let expectations = DisableExpectations {
        request_options: live::LiveOptions::key(api_key),
        ..DisableExpectations::default()
    };
    expect_thinking_disabled_e2e(
        &live::model("anthropic", "claude-sonnet-4-6"),
        &expectations,
    )
    .await;
}

/// Upstream describe `Google thinking disable E2E`, it "disables thinking for
/// Gemini 2.5": the disabled thinking control rides the Google adapter,
/// upstream's `streamSimple` mapping of an absent `reasoning`.
#[tokio::test]
async fn google_disables_thinking_for_gemini_2_5() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let expectations = DisableExpectations {
        request_options: live::LiveOptions {
            api_key: Some(api_key),
            google_thinking: Some(thinking_off()),
            ..live::LiveOptions::default()
        },
        ..DisableExpectations::default()
    };
    expect_thinking_disabled_e2e(&live::model("google", "gemini-2.5-flash"), &expectations).await;
}

/// Upstream describe `Google thinking disable E2E`, it "disables thinking for
/// Gemini 3.x".
#[tokio::test]
async fn google_disables_thinking_for_gemini_3_x() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let expectations = DisableExpectations {
        request_options: live::LiveOptions {
            api_key: Some(api_key),
            google_thinking: Some(thinking_off()),
            ..live::LiveOptions::default()
        },
        ..DisableExpectations::default()
    };
    expect_thinking_disabled_e2e(
        &live::model("google", "gemini-3-flash-preview"),
        &expectations,
    )
    .await;
}

/// Upstream describe `Google thinking disable E2E`, it "does not error when
/// thinking is off for Gemini 3.1 Pro": Pro cannot fully disable thinking, so
/// the lowest level rides hidden and the pong floor drops to 20. Upstream's
/// `maxTokens: 512` cannot ride [`live::LiveOptions`].
#[tokio::test]
async fn google_does_not_error_when_thinking_is_off_for_gemini_3_1_pro() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let expectations = DisableExpectations {
        request_options: live::LiveOptions {
            api_key: Some(api_key),
            google_thinking: Some(thinking_off()),
            ..live::LiveOptions::default()
        },
        min_pongs: 20,
        ..DisableExpectations::default()
    };
    expect_thinking_disabled_e2e(
        &live::model("google", "gemini-3.1-pro-preview"),
        &expectations,
    )
    .await;
}

/// The Vertex credentials upstream's describe block resolves once, upstream's
/// `vertexOptions`: the dedicated api key, else project plus location, else
/// nothing, with the disabled thinking control riding either arm.
fn vertex_options() -> Option<live::LiveOptions> {
    let mut options = live::LiveOptions::default();
    if let Some(api_key) = live::env_key("GOOGLE_CLOUD_API_KEY") {
        options.api_key = Some(api_key);
    } else {
        let project =
            live::env_key("GOOGLE_CLOUD_PROJECT").or_else(|| live::env_key("GCLOUD_PROJECT"));
        let location = live::env_key("GOOGLE_CLOUD_LOCATION");
        match (project, location) {
            (Some(project), Some(location)) => {
                options.vertex_project = Some(project);
                options.vertex_location = Some(location);
            }
            _ => return None,
        }
    }
    options.google_thinking = Some(thinking_off());
    Some(options)
}

/// Upstream describe `Google Vertex thinking disable E2E`, it "disables
/// thinking for Gemini 2.5".
#[tokio::test]
async fn google_vertex_disables_thinking_for_gemini_2_5() {
    let Some(request_options) = vertex_options() else {
        return;
    };
    let expectations = DisableExpectations {
        request_options,
        ..DisableExpectations::default()
    };
    expect_thinking_disabled_e2e(
        &live::model("google-vertex", "gemini-2.5-flash"),
        &expectations,
    )
    .await;
}

/// Upstream describe `Google Vertex thinking disable E2E`, it "disables
/// thinking for Gemini 3.x".
#[tokio::test]
async fn google_vertex_disables_thinking_for_gemini_3_x() {
    let Some(request_options) = vertex_options() else {
        return;
    };
    let expectations = DisableExpectations {
        request_options,
        ..DisableExpectations::default()
    };
    expect_thinking_disabled_e2e(
        &live::model("google-vertex", "gemini-3-flash-preview"),
        &expectations,
    )
    .await;
}

/// Upstream describe `OpenAI thinking disable E2E`, it "disables thinking for
/// Responses reasoning models": upstream's `{ temperature: undefined }`
/// requestOptions only clears the shared temperature override, which the port
/// has no field for, so the probe rides the key alone.
#[tokio::test]
async fn openai_disables_thinking_for_responses_reasoning_models() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let expectations = DisableExpectations {
        request_options: live::LiveOptions::key(api_key),
        ..DisableExpectations::default()
    };
    expect_thinking_disabled_e2e(&live::model("openai", "gpt-5.4-mini"), &expectations).await;
}

/// Upstream describe `OpenRouter thinking disable E2E`, it "disables thinking
/// for Qwen 3.5 reasoning models", with the 100-output-token ceiling
/// upstream's `maxOutputTokens` expectation pins. Upstream's `maxTokens: 160`
/// request cap cannot ride [`live::LiveOptions`].
#[tokio::test]
async fn openrouter_disables_thinking_for_qwen_3_5_reasoning_models() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let expectations = DisableExpectations {
        request_options: live::LiveOptions::key(api_key),
        max_output_tokens: Some(100),
        ..DisableExpectations::default()
    };
    expect_thinking_disabled_e2e(
        &live::model("openrouter", "qwen/qwen3.5-plus-02-15"),
        &expectations,
    )
    .await;
}
