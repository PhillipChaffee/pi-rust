//! The summarization reasoning/options suite, upstream
//! `test/compaction-summary-reasoning.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.
//!
//! Upstream mocks `completeSimple`; the Rust port drives the same
//! option-transformation through the streamFn seam, which
//! [`complete_summarization`] prefers when supplied.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::{Arc, Mutex};

use pi_agent_core::types::StreamFn;
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, CacheRetention, Context,
    KnownApi, Modality, Model, ModelCompat, ModelCost, ModelCostRates, ProviderId,
    SimpleStreamOptions, StopReason, TextContent, ToolCall, Usage, UserContent,
};
use pi_ai::utils::event_stream::create_assistant_message_event_stream;

use pi_coding_agent::compaction::{
    CompactionPreparation, CompactionSettings, FileOperations, compact, complete_summarization,
    generate_summary, generate_summary_with_usage,
};

fn create_model(reasoning: bool, max_tokens: u64, compat: Option<ModelCompat>) -> Model {
    Model {
        id: if reasoning {
            "reasoning-model"
        } else {
            "non-reasoning-model"
        }
        .to_owned(),
        name: if reasoning {
            "Reasoning Model"
        } else {
            "Non-reasoning Model"
        }
        .to_owned(),
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("anthropic".to_owned()),
        base_url: "https://api.anthropic.com".to_owned(),
        reasoning,
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
        max_tokens,
        sampling_params: None,
        headers: None,
        compat,
    }
}

const fn mock_usage() -> Usage {
    Usage {
        input: 10,
        output: 10,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 20,
        cost: pi_ai::types::UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

fn summary_response() -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: "## Goal\nTest summary".to_owned(),
            text_signature: None,
        })],
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("anthropic".to_owned()),
        model: "claude-sonnet-4-5".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: mock_usage(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

fn tool_call_response() -> AssistantMessage {
    let mut message = summary_response();
    message.content = vec![AssistantBlock::ToolCall(ToolCall {
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
    })];
    message.stop_reason = StopReason::ToolUse;
    message
}

fn messages() -> Vec<pi_agent_core::types::AgentMessage> {
    vec![pi_agent_core::types::AgentMessage::Standard(
        pi_ai::types::Message::User(pi_ai::types::UserMessage {
            content: UserContent::Text("Summarize this.".to_owned()),
            timestamp: 0,
        }),
    )]
}

/// The mock upstream's `completeSimpleMock` restates: captures every request
/// option, serves queued one-shot responses, then the default.
struct CompleteMock {
    captured: Mutex<Vec<SimpleStreamOptions>>,
    queue: Mutex<Vec<AssistantMessage>>,
    default: AssistantMessage,
}

impl CompleteMock {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            captured: Mutex::new(Vec::new()),
            queue: Mutex::new(Vec::new()),
            default: summary_response(),
        })
    }

    fn once(&self, message: AssistantMessage) {
        self.queue.lock().expect("queue lock").push(message);
    }

    fn captured(&self) -> Vec<SimpleStreamOptions> {
        self.captured.lock().expect("captured lock").clone()
    }

    fn stream_fn(self: &Arc<Self>) -> StreamFn {
        let mock = Arc::clone(self);
        Arc::new(move |_model, _context, options| {
            mock.captured
                .lock()
                .expect("captured lock")
                .push(options.cloned().unwrap_or_default());
            let next = mock
                .queue
                .lock()
                .expect("queue lock")
                .pop()
                .unwrap_or_else(|| mock.default.clone());
            let stream = create_assistant_message_event_stream();
            stream.push(AssistantMessageEvent::Done {
                reason: next.stop_reason,
                message: next,
            });
            stream
        })
    }
}

#[tokio::test]
async fn uses_the_provided_thinking_level_for_reasoning_capable_models() {
    let mock = CompleteMock::new();
    let stream_fn = mock.stream_fn();

    let result = generate_summary_with_usage(
        &messages(),
        &create_model(true, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(pi_agent_core::types::ThinkingLevel::Medium),
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    assert_eq!(result.text, "## Goal\nTest summary");
    assert_eq!(result.usage, mock_usage());

    let calls = mock.captured();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].reasoning,
        Some(pi_ai::types::ThinkingLevel::Medium)
    );
    assert_eq!(calls[0].api_key.as_deref(), Some("test-key"));
}

#[tokio::test]
async fn preserves_the_string_result_from_generate_summary() {
    let mock = CompleteMock::new();
    let stream_fn = mock.stream_fn();

    let result = generate_summary(
        &messages(),
        &create_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    assert_eq!(result, "## Goal\nTest summary");
}

#[tokio::test]
async fn uses_fresh_routing_sessions_without_prompt_caching() {
    let mock = CompleteMock::new();
    let stream_fn = mock.stream_fn();

    for _ in 0..2 {
        generate_summary(
            &messages(),
            &create_model(false, 8192, None),
            2000,
            Some("test-key"),
            None,
            None,
            None,
            None,
            None,
            Some(&stream_fn),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("summary");
    }

    let options = mock.captured();
    assert_eq!(options.len(), 2);
    assert!(
        options
            .iter()
            .all(|options| options.cache_retention == Some(CacheRetention::None))
    );

    let session_ids: Vec<Option<&String>> = options
        .iter()
        .map(|options| options.session_id.as_ref())
        .collect();
    assert_ne!(session_ids[0], session_ids[1]);
}

#[tokio::test]
async fn honors_caller_supplied_routing_session_and_tool_choice_without_prompt_caching() {
    let mock = CompleteMock::new();
    let stream_fn = mock.stream_fn();

    complete_summarization(
        &create_model(false, 8192, None),
        &Context {
            system_prompt: Some("Summarize".to_owned()),
            messages: Vec::new(),
            tools: None,
        },
        &SimpleStreamOptions {
            session_id: Some("current-routing-session".to_owned()),
            cache_retention: Some(CacheRetention::Long),
            tool_choice: Some(pi_ai::types::ToolChoice::Auto),
            ..SimpleStreamOptions::default()
        },
        Some(&stream_fn),
        None,
        None,
    )
    .await;

    let calls = mock.captured();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].session_id.as_deref(),
        Some("current-routing-session")
    );
    assert_eq!(calls[0].cache_retention, Some(CacheRetention::None));
    assert_eq!(calls[0].tool_choice, Some(pi_ai::types::ToolChoice::Auto));
}

fn split_turn_preparation(
    turn_prefix: Vec<pi_agent_core::types::AgentMessage>,
) -> CompactionPreparation {
    CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_owned(),
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: turn_prefix,
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 2000,
            keep_recent_tokens: 20,
        },
    }
}

#[tokio::test]
async fn rejects_tool_calls_from_conversation_summaries() {
    let mock = CompleteMock::new();
    mock.once(tool_call_response());
    let stream_fn = mock.stream_fn();

    let error = generate_summary_with_usage(
        &messages(),
        &create_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("tool calls rejected");

    assert_eq!(error.message, "Summarization attempted to call a tool");
}

#[tokio::test]
async fn rejects_tool_calls_from_split_turn_summaries() {
    let mock = CompleteMock::new();
    mock.once(tool_call_response());
    let stream_fn = mock.stream_fn();

    let error = compact(
        split_turn_preparation(messages()),
        &create_model(false, 8192, None),
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("tool calls rejected");

    assert_eq!(
        error.message,
        "Turn prefix summarization attempted to call a tool"
    );
}

#[tokio::test]
async fn rejects_a_length_limited_history_summary() {
    let mock = CompleteMock::new();
    let mut length_limited = summary_response();
    length_limited.stop_reason = StopReason::Length;
    length_limited.content = vec![AssistantBlock::Text(TextContent {
        text: "partial".to_owned(),
        text_signature: None,
    })];
    mock.once(length_limited);
    let stream_fn = mock.stream_fn();

    let error = generate_summary_with_usage(
        &messages(),
        &create_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("length stops rejected");

    assert!(error.message.contains("generation hit the token cap"));
}

#[tokio::test]
async fn rejects_a_length_limited_split_turn_summary() {
    let mock = CompleteMock::new();
    let mut length_limited = summary_response();
    length_limited.stop_reason = StopReason::Length;
    length_limited.content = vec![AssistantBlock::Text(TextContent {
        text: "partial".to_owned(),
        text_signature: None,
    })];
    mock.once(length_limited);
    let stream_fn = mock.stream_fn();

    let error = compact(
        split_turn_preparation(messages()),
        &create_model(false, 8192, None),
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("length stops rejected");

    assert!(error.message.contains("generation hit the token cap"));
}

#[tokio::test]
async fn does_not_set_reasoning_when_thinking_is_off() {
    let mock = CompleteMock::new();
    let stream_fn = mock.stream_fn();

    generate_summary(
        &messages(),
        &create_model(true, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(pi_agent_core::types::ThinkingLevel::Off),
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    let calls = mock.captured();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].api_key.as_deref(), Some("test-key"));
    assert!(calls[0].reasoning.is_none());
}

#[tokio::test]
async fn does_not_set_reasoning_for_non_reasoning_models() {
    let mock = CompleteMock::new();
    let stream_fn = mock.stream_fn();

    generate_summary(
        &messages(),
        &create_model(false, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(pi_agent_core::types::ThinkingLevel::Medium),
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    let calls = mock.captured();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].api_key.as_deref(), Some("test-key"));
    assert!(calls[0].reasoning.is_none());
}

fn fallback_compat() -> ModelCompat {
    ModelCompat {
        allowed_fallback_models: Some(vec![pi_ai::types::AnthropicAllowedFallbackModel {
            provider: ProviderId("anthropic".to_owned()),
            model: "claude-opus-4-8".to_owned(),
            cost: ModelCost {
                rates: ModelCostRates {
                    input: 5.0,
                    output: 25.0,
                    cache_read: 0.5,
                    cache_write: 6.25,
                },
                tiers: None,
            },
        }]),
        ..ModelCompat::default()
    }
}

#[tokio::test]
async fn leaves_anthropic_refusal_fallback_handling_to_pi_ai_model_metadata() {
    let mock = CompleteMock::new();
    let stream_fn = mock.stream_fn();

    generate_summary(
        &messages(),
        &create_model(true, 8192, Some(fallback_compat())),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    let calls = mock.captured();
    assert_eq!(calls.len(), 1);
    // The pi-ai model metadata owns the fallback; the request options carry
    // only the fields the caller supplied — nothing fallback-shaped rides
    // the request.
    assert_eq!(calls[0].api_key.as_deref(), Some("test-key"));
    assert!(calls[0].reasoning.is_none());
    assert!(calls[0].tool_choice.is_none());
    assert!(calls[0].metadata.is_none());
}

#[tokio::test]
async fn does_not_set_anthropic_refusal_fallback_for_models_without_allowed_fallback_targets() {
    let mock = CompleteMock::new();
    let stream_fn = mock.stream_fn();

    generate_summary(
        &messages(),
        &create_model(true, 8192, None),
        2000,
        Some("test-key"),
        None,
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    let calls = mock.captured();
    assert_eq!(calls.len(), 1);
    // No allowed targets, no fallback fields on the request.
    assert!(calls[0].reasoning.is_none());
    assert!(calls[0].tool_choice.is_none());
    assert!(calls[0].metadata.is_none());
}

#[tokio::test]
async fn clamps_compaction_summary_max_tokens_to_the_model_output_cap() {
    let mock = CompleteMock::new();
    let stream_fn = mock.stream_fn();

    let preparation = CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_owned(),
        messages_to_summarize: messages(),
        turn_prefix_messages: messages(),
        is_split_turn: true,
        tokens_before: 600_000,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 500_000,
            keep_recent_tokens: 20000,
        },
    };

    let result = compact(
        preparation,
        &create_model(false, 128_000, None),
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("compaction");

    let expected_usage = Usage {
        input: 20,
        output: 20,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 40,
        cost: pi_ai::types::UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    };
    assert_eq!(result.usage, expected_usage);
    let max_tokens: Vec<Option<u64>> = mock
        .captured()
        .iter()
        .map(|options| options.max_tokens)
        .collect();
    assert_eq!(max_tokens, vec![Some(128_000), Some(128_000)]);
}

#[tokio::test]
async fn preserves_the_standalone_split_turn_summary_prompt() {
    // Upstream pins the split-turn prompt shape on the mocked context; the
    // Rust port captures the same context through the streamFn and asserts
    // the prompt text the summarization request carries.
    let mock = CompleteMock::new();
    let captured_context: Arc<Mutex<Vec<Context>>> = Arc::new(Mutex::new(Vec::new()));
    let contexts = Arc::clone(&captured_context);
    let mock_for_context = Arc::clone(&mock);
    let stream_fn: StreamFn = Arc::new(move |_model, context, options| {
        mock_for_context
            .captured
            .lock()
            .expect("captured lock")
            .push(options.cloned().unwrap_or_default());
        contexts.lock().expect("context lock").push(context.clone());
        let next = mock_for_context
            .queue
            .lock()
            .expect("queue lock")
            .pop()
            .unwrap_or_else(|| mock_for_context.default.clone());
        let stream = create_assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: next.stop_reason,
            message: next,
        });
        stream
    });

    compact(
        split_turn_preparation(messages()),
        &create_model(false, 8192, None),
        Some("test-key"),
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("compaction");

    let prompt = {
        let contexts = captured_context.lock().expect("context lock");
        serde_json::to_string(&contexts[0].messages).expect("serialize")
    };
    assert_eq!(captured_context.lock().expect("context lock").len(), 1);
    assert!(prompt.contains("This is the PREFIX of a turn that was too large to keep"));
    assert!(prompt.contains("<conversation>"));
}
