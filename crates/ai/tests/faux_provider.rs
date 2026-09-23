//! The faux provider suites, ported from upstream `faux-provider.test.ts`
//! and the `fauxProvider` block of `providers.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream drives the tests through compat's `registerFauxProvider` +
//! `stream`/`complete` over the api-registry; the Rust tests drive the same
//! core through its `ProviderStreams` entry points, identical semantics —
//! compat's dispatch is registry resolution plus env-key injection. The one
//! registry-behavior case, "unregisters the provider", pins the compat
//! dispatch error and rides with the compat child (#34).
//!
//! The `fauxProvider` block ports from its upstream home in
//! `providers.test.ts`: the Models-collection conformance for the standalone
//! provider, including the deferred submit/poll/redeem machinery.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures unwrap only the values the test just placed"
)]
#![expect(
    clippy::panic,
    reason = "test failures panic by design, mirroring expect's failure mode"
)]

mod common;

use common::event_type_name;
use std::sync::{Arc, Mutex};

use pi_ai::models::{
    ModelsDeferredFetchOptions, ModelsSimpleStreamOptions, Provider, create_models,
};
use pi_ai::providers::faux::{
    FauxAssistantMessageOptions, FauxProviderState, FauxResponseStep, FauxTokenSize,
    RegisterFauxProviderOptions, faux_assistant_message, faux_provider, faux_text, faux_thinking,
    faux_tool_call,
};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, CacheRetention, Context,
    DeferredFetchOptions, DeferredRequest, DeferredWindow, ImageContent, Message, Model,
    ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions, TextContent, Tool,
    ToolResultBlock, ToolResultMessage, TransportOptions, UserBlock, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use tokio_util::sync::CancellationToken;

fn user_message(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: pi_ai::auth::resolve::now_ms(),
    })
}

const fn context(messages: Vec<Message>) -> Context {
    Context {
        system_prompt: None,
        messages,
        tools: None,
    }
}

fn basic_context() -> Context {
    context(vec![user_message("hi")])
}

async fn complete(
    faux: &pi_ai::providers::faux::FauxCore,
    model: &Model,
    context: &Context,
    options: Option<StreamOptions>,
) -> AssistantMessage {
    faux.stream(model, context, options.as_ref()).result().await
}

/// The session-scoped complete, upstream's `complete(..., { sessionId, cacheRetention })`.
async fn complete_with_session(
    faux: &pi_ai::providers::faux::FauxCore,
    model: &Model,
    context: &Context,
    session: &str,
    retention: CacheRetention,
) -> AssistantMessage {
    complete(
        faux,
        model,
        context,
        Some(stream_options_with_session(session, retention)),
    )
    .await
}

async fn collect_events(stream: &AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

/// A scripted single-text response, the suite's most common queue entry.
fn response(text: &str) -> FauxResponseStep {
    message_step(faux_assistant_message(
        text,
        FauxAssistantMessageOptions::default(),
    ))
}

const fn message_step(message: AssistantMessage) -> FauxResponseStep {
    FauxResponseStep::Message(message)
}

fn factory<F>(f: F) -> FauxResponseStep
where
    F: for<'a> Fn(
            &'a Context,
            Option<&'a SimpleStreamOptions>,
            &'a FauxProviderState,
            &'a Model,
        )
            -> pi_ai::types::BoxedFuture<'a, Result<AssistantMessage, FauxFactoryError>>
        + Send
        + Sync
        + 'static,
{
    FauxResponseStep::Factory(Arc::new(f))
}

type FauxFactoryError = Box<dyn std::error::Error + Send + Sync>;

fn tool_arguments(arguments: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    match arguments {
        serde_json::Value::Object(map) => map,
        other => panic!("test tool-call arguments must be an object, got {other}"),
    }
}

fn event_types(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|event| match event {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        })
        .collect()
}

#[tokio::test]
async fn registers_a_custom_provider_and_estimates_usage() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("hello world")]);

    let request_context = Context {
        system_prompt: Some("Be concise.".to_owned()),
        messages: vec![user_message("hi there")],
        tools: None,
    };

    let response = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    assert_eq!(response.content, vec![faux_text("hello world")]);
    assert!(response.usage.input > 0);
    assert!(response.usage.output > 0);
    assert_eq!(
        response.usage.total_tokens,
        response.usage.input + response.usage.output
    );
    assert_eq!(faux.state().call_count(), 1);
}

#[tokio::test]
async fn supports_helper_blocks_for_text_thinking_and_tool_calls() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([message_step(faux_assistant_message(
        vec![
            faux_thinking("think"),
            faux_tool_call(
                "echo",
                tool_arguments(serde_json::json!({ "text": "hi" })),
                None,
            ),
            faux_text("done"),
        ],
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxAssistantMessageOptions::default()
        },
    ))]);

    let response = complete(faux.core(), &faux.first_model(), &basic_context(), None).await;

    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.content.len(), 3);
    assert_eq!(response.content[0], faux_thinking("think"));
    let AssistantBlock::ToolCall(tool_call) = &response.content[1] else {
        panic!("expected a tool-call block");
    };
    assert!(!tool_call.id.is_empty());
    assert_eq!(tool_call.name, "echo");
    assert_eq!(
        tool_call.arguments,
        tool_arguments(serde_json::json!({ "text": "hi" }))
    );
    assert_eq!(response.content[2], faux_text("done"));
}

#[tokio::test]
async fn supports_multiple_models_with_per_model_reasoning_and_model_aware_factories() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        models: vec![
            pi_ai::providers::faux::FauxModelDefinition {
                id: "faux-fast".to_owned(),
                name: Some("Faux Fast".to_owned()),
                reasoning: Some(false),
                ..pi_ai::providers::faux::FauxModelDefinition::default()
            },
            pi_ai::providers::faux::FauxModelDefinition {
                id: "faux-thinker".to_owned(),
                name: Some("Faux Thinker".to_owned()),
                reasoning: Some(true),
                ..pi_ai::providers::faux::FauxModelDefinition::default()
            },
        ],
        ..RegisterFauxProviderOptions::default()
    });
    let model_aware = factory(|_context, _options, _state, model| {
        let text = format!("{}:{}", model.id, model.reasoning);
        Box::pin(async move {
            Ok(faux_assistant_message(
                text,
                FauxAssistantMessageOptions::default(),
            ))
        })
    });
    faux.set_responses([model_aware.clone(), model_aware]);

    let model_ids: Vec<&str> = faux
        .models()
        .iter()
        .map(|model| model.id.as_str())
        .collect();
    assert_eq!(model_ids, vec!["faux-fast", "faux-thinker"]);
    assert_eq!(faux.first_model(), faux.models()[0]);
    assert!(!faux.model("faux-fast").expect("faux-fast").reasoning);
    assert!(faux.model("faux-thinker").expect("faux-thinker").reasoning);

    let fast = complete(
        faux.core(),
        &faux.model("faux-fast").unwrap(),
        &basic_context(),
        None,
    )
    .await;
    let thinker = complete(
        faux.core(),
        &faux.model("faux-thinker").unwrap(),
        &basic_context(),
        None,
    )
    .await;

    assert_eq!(fast.content, vec![faux_text("faux-fast:false")]);
    assert_eq!(thinker.content, vec![faux_text("faux-thinker:true")]);
}

#[tokio::test]
async fn rewrites_api_provider_and_model_on_returned_messages() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        api: Some("faux:test".to_owned()),
        provider: Some("faux-provider".to_owned()),
        models: vec![pi_ai::providers::faux::FauxModelDefinition {
            id: "faux-model".to_owned(),
            ..pi_ai::providers::faux::FauxModelDefinition::default()
        }],
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([response("hello")]);

    let response = complete(faux.core(), &faux.first_model(), &basic_context(), None).await;

    assert_eq!(response.api, Api::from("faux:test"));
    assert_eq!(
        response.provider,
        pi_ai::types::ProviderId::from("faux-provider")
    );
    assert_eq!(response.model, "faux-model");
}

#[tokio::test]
async fn consumes_queued_responses_in_order_and_errors_when_exhausted() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("first"), response("second")]);

    let request_context = basic_context();

    let first = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    let second = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    let exhausted = complete(faux.core(), &faux.first_model(), &request_context, None).await;

    assert_eq!(first.content, vec![faux_text("first")]);
    assert_eq!(second.content, vec![faux_text("second")]);
    assert_eq!(exhausted.stop_reason, StopReason::Error);
    assert_eq!(
        exhausted.error_message.as_deref(),
        Some("No more faux responses queued")
    );
    assert_eq!(faux.get_pending_response_count(), 0);
    assert_eq!(faux.state().call_count(), 3);
}

#[tokio::test]
async fn can_replace_and_append_queued_responses() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("first")]);

    let request_context = basic_context();

    let first = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    assert_eq!(first.content, vec![faux_text("first")]);
    assert_eq!(faux.get_pending_response_count(), 0);

    faux.set_responses([response("second")]);
    assert_eq!(faux.get_pending_response_count(), 1);
    let second = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    assert_eq!(second.content, vec![faux_text("second")]);

    faux.append_responses([response("third"), response("fourth")]);
    assert_eq!(faux.get_pending_response_count(), 2);
    let third = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    assert_eq!(third.content, vec![faux_text("third")]);
    let fourth = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    assert_eq!(fourth.content, vec![faux_text("fourth")]);
    assert_eq!(faux.get_pending_response_count(), 0);
}

#[tokio::test]
async fn supports_async_response_factories() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([factory(|context, _options, state, _model| {
        Box::pin(async move {
            let text = format!("{}:{}", context.messages.len(), state.call_count());
            Ok(faux_assistant_message(
                text,
                FauxAssistantMessageOptions::default(),
            ))
        })
    })]);

    let response = complete(faux.core(), &faux.first_model(), &basic_context(), None).await;

    assert_eq!(response.content, vec![faux_text("1:1")]);
}

#[tokio::test]
async fn emits_an_error_when_a_response_factory_throws() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([factory(|_context, _options, _state, _model| {
        Box::pin(async move { Err::<AssistantMessage, FauxFactoryError>("boom".into()) })
    })]);

    let stream = faux
        .core()
        .stream(&faux.first_model(), &basic_context(), None);
    let events = collect_events(&stream).await;

    assert_eq!(events.len(), 1);
    let AssistantMessageEvent::Error { reason, error } = &events[0] else {
        panic!("expected an error event, got {:?}", events[0]);
    };
    assert_eq!(*reason, StopReason::Error);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.error_message.as_deref(), Some("boom"));
}

#[tokio::test]
async fn rejects_a_queued_response_without_a_terminal_stop_reason() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([message_step(faux_assistant_message(
        "partial",
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::Pending),
            ..FauxAssistantMessageOptions::default()
        },
    ))]);

    let stream = faux
        .core()
        .stream(&faux.first_model(), &basic_context(), None);
    let events = collect_events(&stream).await;

    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::Done { .. }))
    );
    let AssistantMessageEvent::Error { error, .. } = events.last().expect("terminal event") else {
        panic!("expected an error event, got {:?}", events.last());
    };
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.error_message.as_deref(),
        Some("Faux response ended without a stop reason")
    );
}

#[tokio::test]
async fn estimates_prompt_and_output_tokens_from_serialized_context() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("done")]);

    let tool = Tool {
        name: "echo".to_owned(),
        description: "Echo back text".to_owned(),
        // The typebox `Type.Object({ text: Type.String() })` document.
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"],
        }),
        constrained_sampling: None,
    };
    let request_context = Context {
        system_prompt: Some("sys".to_owned()),
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Blocks(vec![
                    UserBlock::Text(TextContent {
                        text: "hello".to_owned(),
                        text_signature: None,
                    }),
                    UserBlock::Image(ImageContent {
                        data: "abcd".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ]),
                timestamp: 1,
            }),
            Message::Assistant(faux_assistant_message(
                "prior",
                FauxAssistantMessageOptions::default(),
            )),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "tool-1".to_owned(),
                tool_name: "echo".to_owned(),
                content: vec![ToolResultBlock::Text(TextContent {
                    text: "tool out".to_owned(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: 2,
            }),
        ],
        tools: Some(vec![tool]),
    };

    let response = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    let tools_json = serde_json::to_string(request_context.tools.as_ref().unwrap()).unwrap();
    let prompt_text = [
        "system:sys".to_owned(),
        "user:hello\n[image:image/png:4]".to_owned(),
        "assistant:prior".to_owned(),
        "toolResult:echo\ntool out".to_owned(),
        format!("tools:{tools_json}"),
    ]
    .join("\n\n");
    let expected_prompt_tokens = prompt_text.len().div_ceil(4) as u64;
    let expected_output_tokens = "done".len().div_ceil(4) as u64;

    assert_eq!(response.usage.input, expected_prompt_tokens);
    assert_eq!(response.usage.output, expected_output_tokens);
    assert_eq!(response.usage.cache_read, 0);
    assert_eq!(response.usage.cache_write, 0);
    assert_eq!(
        response.usage.total_tokens,
        expected_prompt_tokens + expected_output_tokens
    );
}

#[tokio::test]
async fn does_not_share_cache_across_sessions_or_requests_without_session_id() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("first"), response("second"), response("third")]);

    let mut request_context = context(vec![user_message("hello")]);

    let first = complete_with_session(
        faux.core(),
        &faux.first_model(),
        &request_context,
        "session-1",
        CacheRetention::Short,
    )
    .await;
    assert!(first.usage.cache_write > 0);
    request_context.messages.push(Message::Assistant(first));
    request_context.messages.push(Message::User(UserMessage {
        content: UserContent::Text("follow up".to_owned()),
        timestamp: pi_ai::auth::resolve::now_ms() + 1,
    }));

    let second = complete_with_session(
        faux.core(),
        &faux.first_model(),
        &request_context,
        "session-2",
        CacheRetention::Short,
    )
    .await;
    assert_eq!(second.usage.cache_read, 0);
    assert!(second.usage.cache_write > 0);

    let third = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    assert_eq!(third.usage.cache_read, 0);
    assert_eq!(third.usage.cache_write, 0);
}

#[tokio::test]
async fn simulates_prompt_caching_per_session_id() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("first"), response("second")]);

    let mut request_context = Context {
        system_prompt: Some("Be concise.".to_owned()),
        messages: vec![user_message("hello")],
        tools: None,
    };

    let first = complete_with_session(
        faux.core(),
        &faux.first_model(),
        &request_context,
        "session-1",
        CacheRetention::Short,
    )
    .await;
    assert_eq!(first.usage.cache_read, 0);
    assert!(first.usage.cache_write > 0);

    request_context.messages.push(Message::Assistant(first));
    request_context.messages.push(Message::User(UserMessage {
        content: UserContent::Text("follow up".to_owned()),
        timestamp: pi_ai::auth::resolve::now_ms() + 1,
    }));

    let second = complete_with_session(
        faux.core(),
        &faux.first_model(),
        &request_context,
        "session-1",
        CacheRetention::Short,
    )
    .await;
    assert!(second.usage.cache_read > 0);
    assert!(second.usage.input + second.usage.cache_read > second.usage.input);
}

#[tokio::test]
async fn does_not_simulate_caching_when_cache_retention_is_none() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("first"), response("second")]);

    let mut request_context = context(vec![user_message("hello")]);

    complete_with_session(
        faux.core(),
        &faux.first_model(),
        &request_context,
        "session-1",
        CacheRetention::None,
    )
    .await;
    request_context
        .messages
        .push(Message::Assistant(faux_assistant_message(
            "first",
            FauxAssistantMessageOptions::default(),
        )));
    request_context.messages.push(Message::User(UserMessage {
        content: UserContent::Text("follow up".to_owned()),
        timestamp: pi_ai::auth::resolve::now_ms() + 1,
    }));
    let second = complete_with_session(
        faux.core(),
        &faux.first_model(),
        &request_context,
        "session-1",
        CacheRetention::None,
    )
    .await;
    assert_eq!(second.usage.cache_read, 0);
    assert_eq!(second.usage.cache_write, 0);
}

fn stream_options_with_session(session_id: &str, retention: CacheRetention) -> StreamOptions {
    StreamOptions {
        session_id: Some(session_id.to_owned()),
        cache_retention: Some(retention),
        ..StreamOptions::default()
    }
}

#[tokio::test]
async fn streams_thinking_text_and_partial_tool_call_deltas() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([message_step(faux_assistant_message(
        vec![
            faux_thinking("thinking text"),
            faux_text("answer text"),
            faux_tool_call(
                "echo",
                tool_arguments(serde_json::json!({ "text": "hi", "count": 12 })),
                Some("tool-1".to_owned()),
            ),
        ],
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxAssistantMessageOptions::default()
        },
    ))]);

    let stream = faux
        .core()
        .stream(&faux.first_model(), &basic_context(), None);
    let events = collect_events(&stream).await;
    let names = event_type_names(&events);
    let tool_call_deltas = deltas(&events, "toolcall_delta");

    assert!(names.contains(&"thinking_start"));
    assert!(names.contains(&"thinking_delta"));
    assert!(names.contains(&"text_start"));
    assert!(names.contains(&"text_delta"));
    assert!(names.contains(&"toolcall_start"));
    assert!(names.contains(&"toolcall_delta"));
    assert!(names.contains(&"toolcall_end"));
    assert!(tool_call_deltas.len() > 1);
    let joined: String = tool_call_deltas.concat();
    let parsed: serde_json::Value = serde_json::from_str(&joined).expect("arguments JSON");
    assert_eq!(parsed, serde_json::json!({ "text": "hi", "count": 12 }));
}

#[tokio::test]
async fn streams_an_exact_event_order_for_fixed_size_chunks() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([message_step(faux_assistant_message(
        vec![
            faux_thinking("go"),
            faux_text("ok"),
            faux_tool_call(
                "echo",
                tool_arguments(serde_json::json!({})),
                Some("tool-1".to_owned()),
            ),
        ],
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxAssistantMessageOptions::default()
        },
    ))]);

    let stream = faux
        .core()
        .stream(&faux.first_model(), &basic_context(), None);
    let events = collect_events(&stream).await;

    let AssistantMessageEvent::Start { partial } = &events[0] else {
        panic!("expected a start event, got {:?}", events[0]);
    };
    assert_eq!(partial.stop_reason, StopReason::Pending);
    assert_eq!(
        event_types(&events),
        vec![
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
            "done",
        ]
    );
}

#[tokio::test]
async fn streams_multiple_tool_calls_in_one_message() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([message_step(faux_assistant_message(
        vec![
            faux_tool_call(
                "echo",
                tool_arguments(serde_json::json!({ "text": "one" })),
                Some("tool-1".to_owned()),
            ),
            faux_tool_call(
                "echo",
                tool_arguments(serde_json::json!({ "text": "two" })),
                Some("tool-2".to_owned()),
            ),
        ],
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxAssistantMessageOptions::default()
        },
    ))]);

    let stream = faux
        .core()
        .stream(&faux.first_model(), &basic_context(), None);
    let events = collect_events(&stream).await;

    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AssistantMessageEvent::ToolcallStart { .. }))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AssistantMessageEvent::ToolcallEnd { .. }))
            .count(),
        2
    );
}

#[tokio::test]
async fn streams_an_explicit_assistant_error_message_as_a_terminal_error() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        token_size: Some(FauxTokenSize {
            min: Some(2),
            max: Some(2),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([message_step({
        let mut message = faux_assistant_message("partial", FauxAssistantMessageOptions::default());
        message.stop_reason = StopReason::Error;
        message.error_message = Some("upstream failed".to_owned());
        message
    })]);

    let stream = faux
        .core()
        .stream(&faux.first_model(), &basic_context(), None);
    let events = collect_events(&stream).await;

    assert_eq!(
        event_types(&events),
        vec!["start", "text_start", "text_delta", "text_end", "error"]
    );
    let AssistantMessageEvent::Error { reason, error } = events.last().expect("terminal event")
    else {
        panic!("expected an error event, got {:?}", events.last());
    };
    assert_eq!(*reason, StopReason::Error);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.error_message.as_deref(), Some("upstream failed"));
}

#[tokio::test]
async fn streams_an_explicit_assistant_aborted_message_as_a_terminal_error() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        token_size: Some(FauxTokenSize {
            min: Some(2),
            max: Some(2),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([message_step({
        let mut message = faux_assistant_message("partial", FauxAssistantMessageOptions::default());
        message.stop_reason = StopReason::Aborted;
        message.error_message = Some("Request was aborted".to_owned());
        message
    })]);

    let stream = faux
        .core()
        .stream(&faux.first_model(), &basic_context(), None);
    let events = collect_events(&stream).await;

    assert_eq!(
        event_types(&events),
        vec!["start", "text_start", "text_delta", "text_end", "error"]
    );
    let AssistantMessageEvent::Error { reason, error } = events.last().expect("terminal event")
    else {
        panic!("expected an error event, got {:?}", events.last());
    };
    assert_eq!(*reason, StopReason::Aborted);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert_eq!(error.error_message.as_deref(), Some("Request was aborted"));
}

#[tokio::test]
async fn supports_aborting_before_the_first_chunk() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(50.0),
        token_size: Some(FauxTokenSize {
            min: Some(3),
            max: Some(3),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([response("abcdefghijklmnopqrstuvwxyz")]);

    let token = CancellationToken::new();
    token.cancel();
    let events = collect_events(&faux.core().stream(
        &faux.first_model(),
        &basic_context(),
        Some(&aborted_options_with_token(token)),
    ))
    .await;

    assert_eq!(events.len(), 1);
    let AssistantMessageEvent::Error { reason, error } = &events[0] else {
        panic!("expected an error event, got {:?}", events[0]);
    };
    assert_eq!(*reason, StopReason::Aborted);
    assert_eq!(error.stop_reason, StopReason::Aborted);
}

fn aborted_options_with_token(token: CancellationToken) -> StreamOptions {
    StreamOptions {
        transport_options: TransportOptions {
            signal: Some(token),
            ..TransportOptions::default()
        },
        ..StreamOptions::default()
    }
}

#[tokio::test]
async fn supports_aborting_mid_text_stream_when_paced() {
    let faux = paced_faux();
    faux.set_responses([response("abcdefghijklmnopqrstuvwxyz")]);

    let (event_names, text_delta_count) =
        drain_until_first_delta_cancelled(&faux, "text_delta").await;

    assert_eq!(text_delta_count, 1);
    assert!(event_names.contains(&"text_start"));
    assert!(event_names.contains(&"text_delta"));
    assert!(event_names.contains(&"error"));
    assert!(!event_names.contains(&"text_end"));
}

#[tokio::test]
async fn supports_aborting_mid_thinking_stream_when_paced() {
    let faux = paced_faux();
    faux.set_responses([message_step({
        let mut message = faux_assistant_message("ignored", FauxAssistantMessageOptions::default());
        message.content = vec![faux_thinking("abcdefghijklmnopqrstuvwxyz")];
        message
    })]);

    let (event_names, thinking_delta_count) =
        drain_until_first_delta_cancelled(&faux, "thinking_delta").await;

    assert_eq!(thinking_delta_count, 1);
    assert!(event_names.contains(&"thinking_start"));
    assert!(event_names.contains(&"thinking_delta"));
    assert!(event_names.contains(&"error"));
    assert!(!event_names.contains(&"thinking_end"));
}

#[tokio::test]
async fn supports_aborting_mid_toolcall_stream_when_paced() {
    let faux = paced_faux();
    faux.set_responses([message_step({
        let mut message = faux_assistant_message("done", FauxAssistantMessageOptions::default());
        message.content = vec![faux_tool_call(
            "echo",
            tool_arguments(serde_json::json!({
                "text": "abcdefghijklmnopqrstuvwxyz",
                "count": 123_456_789,
            })),
            Some("tool-1".to_owned()),
        )];
        message.stop_reason = StopReason::ToolUse;
        message
    })]);

    let (event_names, tool_call_delta_count) =
        drain_until_first_delta_cancelled(&faux, "toolcall_delta").await;

    assert_eq!(tool_call_delta_count, 1);
    assert!(event_names.contains(&"toolcall_start"));
    assert!(event_names.contains(&"toolcall_delta"));
    assert!(event_names.contains(&"error"));
    assert!(!event_names.contains(&"toolcall_end"));
}

/// The paced faux provider the mid-stream abort probes run against,
/// upstream's `{ tokensPerSecond: 100, tokenSize: { min: 3, max: 3 } }`.
fn paced_faux() -> pi_ai::providers::faux::FauxProviderHandle {
    faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(100.0),
        token_size: Some(FauxTokenSize {
            min: Some(3),
            max: Some(3),
        }),
        ..RegisterFauxProviderOptions::default()
    })
}

/// Drain a paced stream while cancelling after the first delta of `kind`,
/// upstream's abort loops. Returns the event names plus how many deltas of
/// `kind` arrived before the cancel took effect.
async fn drain_until_first_delta_cancelled(
    faux: &pi_ai::providers::faux::FauxProviderHandle,
    kind: &'static str,
) -> (Vec<&'static str>, usize) {
    let token = CancellationToken::new();
    let mut event_names: Vec<&'static str> = Vec::new();
    let mut delta_count = 0usize;
    let stream = faux.core().stream(
        &faux.first_model(),
        &basic_context(),
        Some(&aborted_options_with_token(token.clone())),
    );
    while let Some(event) = stream.next().await {
        event_names.push(event_type_name(&event));
        if event_type_name(&event) == kind {
            delta_count += 1;
            token.cancel();
        }
    }
    (event_names, delta_count)
}

/// The record helper's per-kind delta extraction, shared by the delta suites.
fn deltas(events: &[AssistantMessageEvent], kind: &'static str) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolcallDelta { delta, .. } if kind == "toolcall_delta" => {
                Some(delta.clone())
            }
            AssistantMessageEvent::TextDelta { delta, .. } if kind == "text_delta" => {
                Some(delta.clone())
            }
            AssistantMessageEvent::ThinkingDelta { delta, .. } if kind == "thinking_delta" => {
                Some(delta.clone())
            }
            _ => None,
        })
        .collect()
}

// The fauxProvider block, upstream `providers.test.ts`.

fn tool_use_options() -> ModelsSimpleStreamOptions {
    ModelsSimpleStreamOptions {
        options: SimpleStreamOptions {
            deferred: Some(DeferredRequest::Enabled(true)),
            ..SimpleStreamOptions::default()
        },
        transform_headers: None,
    }
}

#[tokio::test]
async fn streams_queued_responses_through_a_models_collection() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = create_models(None);
    models.set_provider(Arc::new(faux.provider.clone()));
    faux.set_responses([response("hello from faux")]);

    let model = models.models(Some(faux.provider.id())).remove(0);
    let result = models.complete_simple(&model, &basic_context(), None).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.content, vec![faux_text("hello from faux")]);
    assert_eq!(faux.state().call_count(), 1);
}

#[tokio::test]
async fn submits_polls_and_redeems_deferred_responses() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        deferred: Some(pi_ai::providers::faux::FauxDeferredOptions {
            pending_fetches: Some(1),
            poll_after_ms: Some(25),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    let models = create_models(None);
    models.set_provider(Arc::new(faux.provider.clone()));
    faux.set_responses([response("ready")]);
    let model = faux.first_model();

    let submission = models.stream_simple(
        &model,
        &basic_context(),
        Some(&ModelsSimpleStreamOptions {
            options: SimpleStreamOptions {
                deferred: Some(DeferredRequest::Windowed {
                    window: Some(DeferredWindow::H1),
                }),
                ..SimpleStreamOptions::default()
            },
            transform_headers: None,
        }),
    );
    let events = collect_events(&submission).await;
    assert_eq!(event_type_names(&events), vec!["start", "done"]);
    let deferred = submission.result().await;
    assert_eq!(deferred.stop_reason, StopReason::Deferred);
    assert!(deferred.content.is_empty());
    let handle = deferred.deferred.clone().expect("deferred handle");
    assert_eq!(handle.provider, model.provider.0);
    assert_eq!(handle.model_id, model.id);
    assert_eq!(handle.api, model.api.0);
    assert_eq!(handle.poll_after_ms, Some(25));

    let pending = models.fetch_deferred(&model, &handle, None).await;
    assert_eq!(pending.stop_reason, StopReason::Deferred);
    assert_eq!(pending.deferred.as_ref(), Some(&handle));

    let ready = models
        .fetch_deferred(
            &model,
            &handle,
            Some(&ModelsDeferredFetchOptions {
                options: DeferredFetchOptions {
                    wait: Some(0),
                    ..DeferredFetchOptions::default()
                },
                transform_headers: None,
            }),
        )
        .await;
    assert_eq!(ready.stop_reason, StopReason::Stop);
    assert_eq!(ready.content, vec![faux_text("ready")]);
    assert!(ready.usage.total_tokens > 0);
    assert_eq!(faux.state().call_count(), 1);
    assert_eq!(faux.state().deferred_fetch_count(), 2);
}

#[tokio::test]
async fn records_cancellation_and_returns_deferred_fetch_failures_in_band() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = create_models(None);
    models.set_provider(Arc::new(faux.provider.clone()));
    faux.set_responses([
        factory(|_context, _options, _state, _model| {
            Box::pin(
                async move { Err::<AssistantMessage, FauxFactoryError>("deferred failed".into()) },
            )
        }),
        response("cancelled"),
    ]);
    let model = faux.first_model();

    let failed_submission = models
        .complete_simple(&model, &basic_context(), Some(&tool_use_options()))
        .await;
    let failed_handle = failed_submission.deferred.clone().expect("deferred handle");
    let failed = models.fetch_deferred(&model, &failed_handle, None).await;
    assert_eq!(failed.stop_reason, StopReason::Error);
    assert_eq!(failed.error_message.as_deref(), Some("deferred failed"));

    let cancelled_submission = models
        .complete_simple(&model, &basic_context(), Some(&tool_use_options()))
        .await;
    let cancelled_handle = cancelled_submission
        .deferred
        .clone()
        .expect("deferred handle");
    models
        .cancel_deferred(&model, &cancelled_handle, None)
        .await
        .expect("cancel");
    assert_eq!(
        faux.state().cancelled_deferred(),
        vec![cancelled_handle.clone()]
    );
    let cancelled = models.fetch_deferred(&model, &cancelled_handle, None).await;
    assert_eq!(cancelled.stop_reason, StopReason::Error);
    assert!(
        cancelled
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("was cancelled")
    );
}

fn event_type_names(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
    events.iter().map(event_type_name).collect()
}

// The boundary suite: branches the upstream suite leaves untested.

#[tokio::test]
async fn the_string_content_form_builds_a_single_text_block() {
    let message = faux_assistant_message("hi".to_owned(), FauxAssistantMessageOptions::default());
    assert_eq!(message.content, vec![faux_text("hi")]);

    let step = FauxResponseStep::Message(message);
    assert!(format!("{step:?}").contains("Message("));
    let factory_step = factory(|_context, _options, _state, _model| {
        Box::pin(async move { Err::<AssistantMessage, FauxFactoryError>("boom".into()) })
    });
    assert_eq!(format!("{factory_step:?}"), "Factory(..)");
}

#[tokio::test]
async fn the_accessors_read_the_configuration() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        api: Some("faux:custom".to_owned()),
        provider: Some("faux-provider".to_owned()),
        ..RegisterFauxProviderOptions::default()
    });

    assert_eq!(faux.core().api(), "faux:custom");
    assert_eq!(faux.core().provider(), "faux-provider");
    assert_eq!(faux.api(), "faux:custom");
    assert_eq!(faux.models().len(), 1);
    let debug = format!("{:?}", faux.core());
    assert!(debug.contains("faux:custom"));
    assert!(debug.contains("faux-provider"));
}

#[tokio::test]
async fn estimates_tokens_for_tool_results_with_images() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("done")]);

    let request_context = Context {
        messages: vec![Message::ToolResult(ToolResultMessage {
            tool_call_id: "tool-1".to_owned(),
            tool_name: "echo".to_owned(),
            content: vec![ToolResultBlock::Image(ImageContent {
                data: "abcd".to_owned(),
                mime_type: "image/png".to_owned(),
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 2,
        })],
        ..Context::default()
    };

    let response = complete(faux.core(), &faux.first_model(), &request_context, None).await;
    let prompt_text = "toolResult:echo\n[image:image/png:4]";
    assert_eq!(response.usage.input, prompt_text.len().div_ceil(4) as u64);
}

#[tokio::test]
async fn prefix_cache_estimates_stop_at_the_divergence() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("first"), response("second")]);

    let mut request_context = Context {
        system_prompt: Some("first prompt".to_owned()),
        messages: vec![user_message("hello")],
        tools: None,
    };

    let first = complete_with_session(
        faux.core(),
        &faux.first_model(),
        &request_context,
        "session-1",
        CacheRetention::Short,
    )
    .await;
    assert!(first.usage.cache_write > 0);

    request_context.system_prompt = Some("second prompt".to_owned());
    let second = complete_with_session(
        faux.core(),
        &faux.first_model(),
        &request_context,
        "session-1",
        CacheRetention::Short,
    )
    .await;
    // The shared "system:" prefix still reads from cache; the divergence
    // stops the prefix estimate and the rest re-estimates as cache writes.
    assert!(second.usage.cache_read > 0);
    assert!(second.usage.cache_write > 0);
}

#[tokio::test]
async fn chunk_cuts_land_on_char_boundaries() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([response("héllo wörld")]);

    let response = complete(faux.core(), &faux.first_model(), &basic_context(), None).await;
    assert_eq!(response.content, vec![faux_text("héllo wörld")]);
}

#[tokio::test]
async fn on_response_hooks_fire_for_streams_and_deferred_operations() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([response("ok"), response("second")]);

    let fired: Arc<Mutex<Vec<u16>>> = Arc::new(Mutex::new(Vec::new()));
    let hook = {
        let fired = Arc::clone(&fired);
        pi_ai::types::OnResponse::new(move |response, _model| {
            let fired = Arc::clone(&fired);
            Box::pin(async move {
                fired
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(response.status);
            })
        })
    };

    let model = faux.first_model();
    let transport = || TransportOptions {
        on_response: Some(hook.clone()),
        ..TransportOptions::default()
    };

    // The plain stream, upstream's `streamOptions?.onResponse`.
    let stream = faux.core().stream(
        &model,
        &basic_context(),
        Some(&StreamOptions {
            transport_options: transport(),
            ..StreamOptions::default()
        }),
    );
    let _ = stream.result().await;
    // The deferred submission, pending fetch, and cancel hooks, upstream's
    // `fetchOptions?.onResponse` and `cancelOptions?.onResponse`.
    let submission = faux.core().stream_simple(
        &model,
        &basic_context(),
        Some(&SimpleStreamOptions {
            transport_options: transport(),
            deferred: Some(DeferredRequest::Enabled(true)),
            ..SimpleStreamOptions::default()
        }),
    );
    let handle = submission.result().await.deferred.expect("deferred handle");
    let fetched = faux.core().fetch_deferred(
        &model,
        &handle,
        Some(&DeferredFetchOptions {
            transport_options: transport(),
            ..DeferredFetchOptions::default()
        }),
    );
    let _ = fetched.expect("faux always fetches").result().await;
    let cancelled = faux
        .core()
        .cancel_deferred(
            &model,
            &handle,
            Some(&pi_ai::types::ProviderRequestOptions {
                transport_options: transport(),
                ..pi_ai::types::ProviderRequestOptions::default()
            }),
        )
        .await;
    assert!(cancelled.is_ok());

    let statuses = fired
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(statuses, vec![200, 200, 200, 200]);
}

#[tokio::test]
async fn unknown_deferred_handles_fail_in_band() {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let model = faux.first_model();

    let fabricated = pi_ai::types::DeferredHandle {
        provider: model.provider.0.clone(),
        model_id: model.id.clone(),
        api: model.api.0.clone(),
        id: "no-such-id".to_owned(),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    };
    let unknown = faux
        .core()
        .fetch_deferred(&model, &fabricated, None)
        .expect("faux always fetches")
        .result()
        .await;
    assert_eq!(unknown.stop_reason, StopReason::Error);
    assert_eq!(
        unknown.error_message.as_deref(),
        Some("Unknown faux deferred response: no-such-id")
    );

    // A submitted handle whose fields no longer match fails the same way,
    // upstream's provider/modelId/api guard.
    faux.set_responses([response("ok")]);
    let submission = faux
        .core()
        .stream_simple(
            &model,
            &basic_context(),
            Some(&SimpleStreamOptions {
                deferred: Some(DeferredRequest::Enabled(true)),
                ..SimpleStreamOptions::default()
            }),
        )
        .result()
        .await;
    let mut mismatched = submission.deferred.expect("deferred handle");
    mismatched.provider = "another-provider".to_owned();
    let mismatched_id = mismatched.id.clone();
    let mismatched = faux
        .core()
        .fetch_deferred(&model, &mismatched, None)
        .expect("faux always fetches")
        .result()
        .await;
    assert_eq!(
        mismatched.error_message.as_deref(),
        Some(format!("Unknown faux deferred response: {mismatched_id}").as_str())
    );
}
