//! The branch-summarization suite, upstream `test/branch-summarization.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::{Arc, Mutex};

use pi_agent_core::types::StreamFn;
use pi_ai::providers::faux::{FauxAssistantMessageOptions, faux_assistant_message};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, KnownApi, Modality, Model,
    ModelCost, ModelCostRates, ProviderId, SimpleStreamOptions, StopReason, ToolCall,
};
use pi_ai::utils::event_stream::{
    AssistantMessageEventStream, create_assistant_message_event_stream,
};
use tokio_util::sync::CancellationToken;

use pi_coding_agent::compaction::{GenerateBranchSummaryOptions, generate_branch_summary};
use pi_coding_agent::session_manager::entries::{MessageEntry, SessionEntry, SessionEntryBase};

fn test_model() -> Model {
    Model {
        id: "test-model".to_owned(),
        name: "Test Model".to_owned(),
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("anthropic".to_owned()),
        base_url: "https://api.anthropic.com".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 200_000,
        max_tokens: 8192,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn entries() -> Vec<SessionEntry> {
    vec![SessionEntry::Message(MessageEntry {
        base: SessionEntryBase {
            id: Some("branch-user".to_owned()),
            parent_id: None,
            timestamp: "1970-01-01T00:00:00.001Z".to_owned(),
            extras: serde_json::Map::new(),
        },
        message: Some(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("Abandoned request".to_owned()),
                timestamp: 1,
            }),
        )),
        extras: serde_json::Map::new(),
    })]
}

/// The response the mocks settle with, upstream's `response(content)`: the
/// faux message restamped to the fixture model.
fn response(content: Vec<AssistantBlock>) -> AssistantMessage {
    let mut message = faux_assistant_message("", FauxAssistantMessageOptions::default());
    message.content = content;
    message.api = Api::from(KnownApi::AnthropicMessages);
    message.provider = ProviderId("anthropic".to_owned());
    "test-model".clone_into(&mut message.model);
    message
}

fn text_block(text: &str) -> AssistantBlock {
    AssistantBlock::Text(pi_ai::types::TextContent {
        text: text.to_owned(),
        text_signature: None,
    })
}

/// The streamFn mock upstream's inline closures build: capture the request
/// options and settle with the given message.
fn capture_stream(
    captured: Arc<Mutex<Option<SimpleStreamOptions>>>,
    message: AssistantMessage,
) -> StreamFn {
    Arc::new(
        move |_model, _context, options| -> AssistantMessageEventStream {
            *captured.lock().expect("captured lock") = options.cloned();
            let stream = create_assistant_message_event_stream();
            stream.push(AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message: message.clone(),
            });
            stream
        },
    )
}

fn tool_call_block() -> AssistantBlock {
    AssistantBlock::ToolCall(ToolCall {
        id: "tool-call-1".to_owned(),
        name: "read".to_owned(),
        namespace: None,
        arguments: {
            let mut args = serde_json::Map::new();
            args.insert(
                "path".to_owned(),
                serde_json::Value::String("README.md".to_owned()),
            );
            args
        },
        thought_signature: None,
    })
}

fn summary_options<'a>(
    model: &'a Model,
    signal: &'a CancellationToken,
    stream_fn: Option<&'a StreamFn>,
) -> GenerateBranchSummaryOptions<'a> {
    GenerateBranchSummaryOptions {
        model,
        api_key: None,
        headers: None,
        env: None,
        signal,
        custom_instructions: None,
        replace_instructions: false,
        reserve_tokens: None,
        stream_fn,
        retry: None,
        callbacks: None,
    }
}

#[tokio::test]
async fn does_not_override_tool_choice_for_branch_summaries() {
    let captured = Arc::new(Mutex::new(None));
    let stream_fn = capture_stream(Arc::clone(&captured), response(vec![text_block("summary")]));

    let model = test_model();
    let signal = CancellationToken::new();
    generate_branch_summary(
        &entries(),
        &summary_options(&model, &signal, Some(&stream_fn)),
    )
    .await;

    let request_options = captured
        .lock()
        .expect("captured lock")
        .clone()
        .expect("options captured");
    assert_eq!(request_options.max_tokens, Some(4096));
    assert!(request_options.tool_choice.is_none());
}

#[tokio::test]
async fn clamps_the_branch_summary_output_cap_to_the_model_limit() {
    let captured = Arc::new(Mutex::new(None));
    let stream_fn = capture_stream(Arc::clone(&captured), response(vec![text_block("summary")]));

    let mut model = test_model();
    model.max_tokens = 1024;
    let signal = CancellationToken::new();
    generate_branch_summary(
        &entries(),
        &summary_options(&model, &signal, Some(&stream_fn)),
    )
    .await;

    let request_options = captured
        .lock()
        .expect("captured lock")
        .clone()
        .expect("options captured");
    assert_eq!(request_options.max_tokens, Some(1024));
}

#[tokio::test]
async fn rejects_tool_calls_from_branch_summaries() {
    let stream_fn = capture_stream(
        Arc::new(Mutex::new(None)),
        response(vec![tool_call_block()]),
    );

    let model = test_model();
    let signal = CancellationToken::new();
    let result = generate_branch_summary(
        &entries(),
        &summary_options(&model, &signal, Some(&stream_fn)),
    )
    .await;

    assert_eq!(
        result.error.as_deref(),
        Some("Branch summarization attempted to call a tool")
    );
}

#[tokio::test]
async fn rejects_length_limited_branch_summaries() {
    let mut message = response(vec![text_block("partial")]);
    message.stop_reason = StopReason::Length;
    let stream_fn = capture_stream(Arc::new(Mutex::new(None)), message);

    let model = test_model();
    let signal = CancellationToken::new();
    let result = generate_branch_summary(
        &entries(),
        &summary_options(&model, &signal, Some(&stream_fn)),
    )
    .await;

    assert_eq!(
        result.error.as_deref(),
        Some(
            "Branch summarization failed: generation hit the token cap and the summary is incomplete"
        )
    );
}
