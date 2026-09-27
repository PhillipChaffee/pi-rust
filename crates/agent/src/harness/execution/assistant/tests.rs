//! The assistant stream runner's suite, ported 1:1 from upstream
//! `test/harness/execution-assistant.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, with boundary tests binding
//! the branches the upstream suite leaves untested.
//!
//! Restatements the port carries: upstream's `queueMicrotask` pushes
//! restate as buffered pushes on the returned stream — `next` drains the
//! queue in push order, so the observable sequence is identical; the
//! "starts and updates are distinct objects" assertion restates as the
//! stored snapshots' contents diverging (the start snapshot keeps its
//! empty accumulator while the update carries the delta); and the mapped
//! `signal` restates behaviorally, the chord controller's abort cancelling
//! the token the options carry.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use pi_ai::providers::faux::{
    FauxAssistantMessageOptions, FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions,
    faux_assistant_message, faux_provider,
};
use pi_ai::types::ProviderStreams;
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, BoxedFuture, DeferredRequest,
    DeferredWindow, ProviderResponse, StopReason, TextContent, Transport, Usage, UsageCost,
};
use pi_ai::utils::event_stream::{
    AssistantMessageEventStream, create_assistant_message_event_stream,
};
use serde_json::Value as JsonValue;
use serde_json::json;

use crate::harness::context::{Context, background_context, with_abort_signal, with_cancel};
use crate::harness::execution::assistant::{
    AfterResponseHook, AssistantRequestContext, AssistantRequestHook, AssistantResponseMetadata,
    AssistantStreamError, AssistantStreamObserver, BeforePayloadHook, HarnessAssistantStreamConfig,
    ToProviderMessages, TransformContextHook, stream_harness_assistant,
};
use crate::harness::gate::AbortRequested;
use crate::types::{AgentMessage, ThinkingLevel};

fn usage() -> Usage {
    Usage {
        input: 1,
        output: 2,
        cache_read: 3,
        cache_write: 4,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 10,
        cost: UsageCost::default(),
    }
}

fn model() -> pi_ai::types::Model {
    pi_ai::types::Model {
        id: "model".to_owned(),
        name: "Model".to_owned(),
        api: Api::from("test"),
        provider: pi_ai::types::ProviderId::from("provider"),
        base_url: "https://example.invalid".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 128_000,
        max_tokens: 16_384,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn user(text: &str) -> AgentMessage {
    serde_json::from_value(json!({
        "role": "user",
        "content": text,
        "timestamp": 1
    }))
    .expect("user message")
}

fn assistant(text: &str, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: Api::from("test"),
        provider: pi_ai::types::ProviderId::from("provider"),
        model: "model".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage(),
        stop_reason,
        deferred: None,
        error_message: (stop_reason == StopReason::Error).then(|| text.to_owned()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    }
}

/// The live accumulator at `start`: upstream's `{ ...assistant(""),
/// stopReason: "pending" }` with empty content.
fn pending_partial() -> AssistantMessage {
    let mut partial = assistant("", StopReason::Pending);
    partial.content = Vec::new();
    partial
}

fn to_provider_messages() -> ToProviderMessages {
    Arc::new(|messages: &[AgentMessage], _context: &Context| {
        let converted = messages
            .iter()
            .filter_map(|message| match message {
                AgentMessage::Standard(message) => Some(message.clone()),
                AgentMessage::Custom(_) => None,
            })
            .collect::<Vec<_>>();
        Box::pin(async move { converted })
    })
}

fn wire_stop_reason(stop_reason: crate::harness::session::types::SettledStopReason) -> String {
    serde_json::to_value(stop_reason.stop_reason())
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// The lifecycle recorder, upstream's observer object literals.
#[derive(Debug, Default)]
struct RecordingObserver {
    order: Arc<Mutex<Vec<String>>>,
    starts: Arc<Mutex<Vec<AssistantMessage>>>,
    updates: Arc<Mutex<Vec<AssistantMessage>>>,
    ends: Arc<Mutex<Vec<AssistantMessage>>>,
    start_event_type: Arc<Mutex<Option<String>>>,
}

impl AssistantStreamObserver for RecordingObserver {
    fn start(
        &self,
        message: AssistantMessage,
        _event: &AssistantMessageEvent,
        _context: &Context,
    ) -> BoxedFuture<'_, ()> {
        self.order
            .lock()
            .expect("order lock")
            .push("observer_start".to_owned());
        self.starts.lock().expect("starts lock").push(message);
        *self.start_event_type.lock().expect("event lock") = Some("start".to_owned());
        Box::pin(async {})
    }

    fn update(
        &self,
        message: AssistantMessage,
        _event: &AssistantMessageEvent,
        _context: &Context,
    ) -> BoxedFuture<'_, ()> {
        self.order
            .lock()
            .expect("order lock")
            .push("observer_update".to_owned());
        self.updates.lock().expect("updates lock").push(message);
        Box::pin(async {})
    }

    fn end(
        &self,
        message: &crate::harness::session::types::SettledAssistantMessage,
        _context: &Context,
    ) -> BoxedFuture<'_, ()> {
        self.order
            .lock()
            .expect("order lock")
            .push("observer_end".to_owned());
        self.ends
            .lock()
            .expect("ends lock")
            .push(message.message.clone());
        Box::pin(async {})
    }
}

fn text_blocks(message: &AssistantMessage) -> Vec<String> {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

/// A request hook streaming the given events from a fresh stream — the
/// buffered restatement of the suite's `queueMicrotask` pushes, `next`
/// draining the queue in push order.
fn stream_of(events: Vec<AssistantMessageEvent>) -> AssistantRequestHook {
    Arc::new(
        move |_ai_context: &pi_ai::types::Context,
              _options: pi_ai::types::SimpleStreamOptions,
              _context: &Context| {
            let stream = create_assistant_message_event_stream();
            for event in &events {
                stream.push(event.clone());
            }
            Box::pin(async move { stream })
        },
    )
}

/// The boundary-config base: the fields the streaming cases share, with
/// each case's overrides carried by struct-update syntax.
fn boundary_config(
    request: AssistantRequestHook,
    observer: Arc<dyn AssistantStreamObserver>,
) -> HarnessAssistantStreamConfig {
    HarnessAssistantStreamConfig {
        model: model(),
        system_prompt: "system".to_owned(),
        tools: None,
        thinking_level: ThinkingLevel::Off,
        stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
        transform_context: None,
        to_provider_messages: to_provider_messages(),
        before_payload: None,
        after_response: None,
        request,
        observer,
    }
}

/// Maps curated options and runs the assistant lifecycle without mutating
/// the input.
#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the 1:1 port of upstream's lifecycle case asserts every mapped option and the full observer order in one flow"
)]
#[expect(
    clippy::significant_drop_tightening,
    reason = "the assertion tail reads each recording lock in turn; the guards span exactly the asserts that use them"
)]
async fn maps_curated_options_and_runs_the_assistant_lifecycle_without_mutating_input() {
    let input = vec![user("original")];
    let request_model = model();
    let (_derived, controller) = with_cancel(&background_context());
    let context = with_abort_signal(controller.signal().clone(), &background_context());
    let order = Arc::new(Mutex::new(Vec::<String>::new()));
    let starts = Arc::new(Mutex::new(Vec::<AssistantMessage>::new()));
    let updates = Arc::new(Mutex::new(Vec::<AssistantMessage>::new()));
    let ends = Arc::new(Mutex::new(Vec::<AssistantMessage>::new()));
    let converted = Arc::new(Mutex::new(Vec::<AgentMessage>::new()));
    let request_payload = Arc::new(Mutex::new(None::<JsonValue>));
    let received_context = Arc::new(Mutex::new(None::<pi_ai::types::Context>));
    let received_options = Arc::new(Mutex::new(None::<pi_ai::types::SimpleStreamOptions>));
    let response_metadata = Arc::new(Mutex::new(None::<AssistantResponseMetadata>));
    let seen_model_id = Arc::new(Mutex::new(None::<String>));

    let order_handle = Arc::clone(&order);
    let transform_context: TransformContextHook = Arc::new(
        move |mut request_context: AssistantRequestContext, _context: &Context| {
            order_handle
                .lock()
                .expect("order lock")
                .push("transform_context".to_owned());
            request_context.messages.push(user("injected"));
            let transformed = AssistantRequestContext {
                messages: request_context.messages,
                system_prompt: "transformed system".to_owned(),
            };
            Box::pin(async move { Ok(transformed) })
        },
    );

    let order_handle = Arc::clone(&order);
    let converted_handle = Arc::clone(&converted);
    let to_provider_messages: ToProviderMessages = {
        let base = to_provider_messages();
        Arc::new(move |messages: &[AgentMessage], context: &Context| {
            order_handle
                .lock()
                .expect("order lock")
                .push("to_provider_messages".to_owned());
            *converted_handle.lock().expect("converted lock") = messages.to_vec();
            base(messages, context)
        })
    };

    let order_handle = Arc::clone(&order);
    let payload_handle = Arc::clone(&request_payload);
    let model_handle = Arc::clone(&seen_model_id);
    let before_payload: BeforePayloadHook = Arc::new(
        move |payload: JsonValue, model: pi_ai::types::Model, _context: Context| {
            order_handle
                .lock()
                .expect("order lock")
                .push("before_payload".to_owned());
            *model_handle.lock().expect("model lock") = Some(model.id);
            *payload_handle.lock().expect("payload lock") = Some(payload);
            Box::pin(async move { Some(json!({ "replaced": true })) })
        },
    );

    let order_handle = Arc::clone(&order);
    let metadata_handle = Arc::clone(&response_metadata);
    let after_response: AfterResponseHook = Arc::new(
        move |mut message: crate::harness::session::types::SettledAssistantMessage,
              metadata: AssistantResponseMetadata,
              _context: Context| {
            order_handle
                .lock()
                .expect("order lock")
                .push("after_response".to_owned());
            *metadata_handle.lock().expect("metadata lock") = Some(metadata);
            message.message.content = vec![AssistantBlock::Text(TextContent {
                text: "transformed".to_owned(),
                text_signature: None,
            })];
            Box::pin(async move { Ok(message) })
        },
    );

    let order_handle = Arc::clone(&order);
    let context_handle = Arc::clone(&received_context);
    let options_handle = Arc::clone(&received_options);
    let stream_model = request_model.clone();
    let request: AssistantRequestHook = Arc::new(
        move |ai_context: &pi_ai::types::Context,
              options: pi_ai::types::SimpleStreamOptions,
              _context: &Context|
              -> BoxedFuture<'_, AssistantMessageEventStream> {
            order_handle
                .lock()
                .expect("order lock")
                .push("request".to_owned());
            *context_handle.lock().expect("context lock") = Some(ai_context.clone());
            *options_handle.lock().expect("options lock") = Some(options.clone());
            let mut resolved_model = stream_model.clone();
            resolved_model.id = "resolved".to_owned();
            let response_model = stream_model.clone();
            let stream = create_assistant_message_event_stream();
            Box::pin(async move {
                let replaced = options
                    .transport_options
                    .on_payload
                    .as_ref()
                    .expect("payload hook")
                    .call(json!({ "original": true }), resolved_model)
                    .await;
                assert_eq!(replaced, Some(json!({ "replaced": true })));
                options
                    .transport_options
                    .on_response
                    .as_ref()
                    .expect("response hook")
                    .call(
                        ProviderResponse {
                            status: 201,
                            headers: BTreeMap::from([("request-id".to_owned(), "r1".to_owned())]),
                        },
                        response_model,
                    )
                    .await;

                let initial = pending_partial();
                let mut partial = initial.clone();
                partial.content = vec![AssistantBlock::Text(TextContent {
                    text: "raw".to_owned(),
                    text_signature: None,
                })];
                stream.push(AssistantMessageEvent::Start { partial: initial });
                stream.push(AssistantMessageEvent::TextDelta {
                    content_index: 0,
                    delta: "raw".to_owned(),
                    partial,
                });
                stream.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message: assistant("raw", StopReason::Stop),
                });
                stream
            })
        },
    );

    let observer = RecordingObserver {
        order: Arc::clone(&order),
        starts: Arc::clone(&starts),
        updates: Arc::clone(&updates),
        ends: Arc::clone(&ends),
        start_event_type: Arc::new(Mutex::new(None)),
    };

    let result = stream_harness_assistant(
        &input,
        &HarnessAssistantStreamConfig {
            model: request_model.clone(),
            system_prompt: "system".to_owned(),
            tools: None,
            thinking_level: ThinkingLevel::High,
            stream_options: crate::harness::types::AgentHarnessStreamOptions {
                transport: Some(Transport::Websocket),
                timeout_ms: Some(123),
                max_retries: Some(2),
                max_retry_delay_ms: Some(456),
                headers: Some(BTreeMap::from([(
                    "authorization".to_owned(),
                    "test".to_owned(),
                )])),
                metadata: Some(BTreeMap::from([("tenant".to_owned(), json!("one"))])),
                cache_retention: Some(pi_ai::types::CacheRetention::Long),
                deferred: Some(DeferredRequest::Windowed {
                    window: Some(DeferredWindow::H1),
                }),
            },
            transform_context: Some(transform_context),
            to_provider_messages,
            before_payload: Some(before_payload),
            after_response: Some(after_response),
            request,
            observer: Arc::new(observer),
        },
        &context,
    )
    .await
    .expect("the stream settles");

    assert_eq!(input, vec![user("original")]);
    assert_eq!(
        *converted.lock().expect("converted lock"),
        vec![user("original"), user("injected")]
    );
    let received = received_context
        .lock()
        .expect("context lock")
        .clone()
        .expect("the request saw a context");
    assert_eq!(
        received.system_prompt.as_deref(),
        Some("transformed system")
    );
    let expected_messages: Vec<pi_ai::types::Message> = serde_json::from_value(json!([
        { "role": "user", "content": "original", "timestamp": 1 },
        { "role": "user", "content": "injected", "timestamp": 1 },
    ]))
    .expect("expected provider messages");
    assert_eq!(received.messages, expected_messages);
    assert_eq!(
        *seen_model_id.lock().expect("model lock"),
        Some("resolved".to_owned())
    );
    assert_eq!(
        *request_payload.lock().expect("payload lock"),
        Some(json!({ "original": true }))
    );
    let options = received_options
        .lock()
        .expect("options lock")
        .clone()
        .expect("the request saw options");
    assert_eq!(options.transport, Some(Transport::Websocket));
    assert_eq!(options.timeout_ms, Some(123));
    assert_eq!(options.max_retries, Some(2));
    assert_eq!(options.max_retry_delay_ms, Some(456));
    assert_eq!(
        options.headers,
        Some(BTreeMap::from([(
            "authorization".to_owned(),
            Some("test".to_owned())
        )]))
    );
    assert_eq!(
        options.metadata,
        Some(BTreeMap::from([("tenant".to_owned(), json!("one"))]))
    );
    assert_eq!(
        options.cache_retention,
        Some(pi_ai::types::CacheRetention::Long)
    );
    assert_eq!(
        options.deferred,
        Some(DeferredRequest::Windowed {
            window: Some(DeferredWindow::H1)
        })
    );
    assert_eq!(options.reasoning, Some(pi_ai::types::ThinkingLevel::High));
    assert!(options.telemetry_context.is_some());
    // The mapped signal restates behaviorally: the controller's abort
    // cancels the token the options carry; the link task runs on the
    // runtime, so give it a scheduling turn.
    let request_token = options
        .transport_options
        .signal
        .clone()
        .expect("a request token");
    assert!(!request_token.is_cancelled());
    controller.abort("done");
    tokio::task::yield_now().await;
    assert!(request_token.is_cancelled());

    assert_eq!(
        *response_metadata.lock().expect("metadata lock"),
        Some(AssistantResponseMetadata {
            status: Some(201),
            headers: Some(BTreeMap::from([("request-id".to_owned(), "r1".to_owned())])),
        })
    );
    {
        let starts = starts.lock().expect("starts lock");
        assert_eq!(starts.len(), 1);
        assert!(text_blocks(&starts[0]).is_empty());
    }
    {
        let updates = updates.lock().expect("updates lock");
        assert_eq!(updates.len(), 1);
        assert_eq!(text_blocks(&updates[0]), vec!["raw"]);
    }
    {
        let ends = ends.lock().expect("ends lock");
        assert_eq!(ends.len(), 1);
        assert_eq!(ends[0].content, result.message.content);
    }
    assert_eq!(text_blocks(&result.message), vec!["transformed"]);
    assert_eq!(
        *order.lock().expect("order lock"),
        vec![
            "transform_context",
            "to_provider_messages",
            "request",
            "before_payload",
            "observer_start",
            "observer_update",
            "after_response",
            "observer_end",
        ]
    );
}

/// Runs against the faux provider request boundary.
#[tokio::test]
#[expect(
    clippy::significant_drop_tightening,
    reason = "the assertion tail reads each recording lock in turn; the guards span exactly the asserts that use them"
)]
async fn runs_against_the_faux_provider_request_boundary() {
    let faux = faux_provider(RegisterFauxProviderOptions {
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([FauxResponseStep::Message(faux_assistant_message(
        "hello",
        FauxAssistantMessageOptions::default(),
    ))]);
    let lifecycle = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen_context = Arc::new(Mutex::new(Vec::<pi_ai::types::Message>::new()));
    let seen_options = Arc::new(Mutex::new(None::<pi_ai::types::SimpleStreamOptions>));

    let observer = RecordingObserver {
        order: Arc::clone(&lifecycle),
        starts: Arc::new(Mutex::new(Vec::new())),
        updates: Arc::new(Mutex::new(Vec::new())),
        ends: Arc::new(Mutex::new(Vec::new())),
        start_event_type: Arc::new(Mutex::new(None)),
    };

    let seen_handle = Arc::clone(&seen_context);
    let options_handle = Arc::clone(&seen_options);
    let request_model = faux.first_model();
    let faux_core = faux.core().clone();
    let request: AssistantRequestHook = Arc::new(
        move |ai_context: &pi_ai::types::Context,
              options: pi_ai::types::SimpleStreamOptions,
              _context: &Context|
              -> BoxedFuture<'_, AssistantMessageEventStream> {
            *seen_handle.lock().expect("seen lock") = ai_context.messages.clone();
            *options_handle.lock().expect("options lock") = Some(options.clone());
            let stream = ProviderStreams::stream_simple(
                &faux_core,
                &request_model,
                ai_context,
                Some(&options),
            );
            Box::pin(async move { stream })
        },
    );

    let result = stream_harness_assistant(
        &[user("prompt")],
        &HarnessAssistantStreamConfig {
            model: faux.first_model(),
            system_prompt: "system".to_owned(),
            tools: None,
            thinking_level: ThinkingLevel::Off,
            stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
            transform_context: None,
            to_provider_messages: to_provider_messages(),
            before_payload: None,
            after_response: None,
            request,
            observer: Arc::new(observer),
        },
        &background_context(),
    )
    .await
    .expect("the faux stream settles");

    assert_eq!(
        *seen_context.lock().expect("seen lock"),
        serde_json::from_value::<Vec<pi_ai::types::Message>>(json!([
            { "role": "user", "content": "prompt", "timestamp": 1 },
        ]))
        .expect("expected provider messages")
    );
    // A background context carries no abort signal, so the request options
    // carry no token.
    assert!(
        seen_options
            .lock()
            .expect("options lock")
            .as_ref()
            .expect("the options")
            .transport_options
            .signal
            .is_none()
    );
    assert_eq!(text_blocks(&result.message), vec!["hello"]);
    {
        let lifecycle = lifecycle.lock().expect("lifecycle lock");
        assert_eq!(lifecycle[0], "observer_start");
        assert_eq!(lifecycle[lifecycle.len() - 1], "observer_end");
        assert!(
            lifecycle
                .iter()
                .filter(|entry| entry.as_str() == "observer_update")
                .count()
                > 0
        );
    }
}

/// Rejects a successful terminal event before start, and the off thinking
/// level sends no reasoning field.
#[tokio::test]
async fn rejects_a_successful_terminal_event_before_start() {
    let final_message = assistant("complete", StopReason::Stop);
    let seen_reasoning = Arc::new(Mutex::new(None::<Option<pi_ai::types::ThinkingLevel>>));
    let reasoning_handle = Arc::clone(&seen_reasoning);
    let options_probe: AssistantRequestHook = Arc::new(
        move |_ai_context: &pi_ai::types::Context,
              options: pi_ai::types::SimpleStreamOptions,
              _context: &Context| {
            *reasoning_handle.lock().expect("reasoning lock") = Some(options.reasoning);
            let settled_final = final_message.clone();
            Box::pin(async move {
                let stream = create_assistant_message_event_stream();
                stream.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message: settled_final,
                });
                stream
            })
        },
    );

    let error = stream_harness_assistant(
        &[user("prompt")],
        &boundary_config(options_probe, Arc::new(RecordingObserver::default())),
        &background_context(),
    )
    .await
    .expect_err("a done before start rejects");

    assert!(matches!(&error, AssistantStreamError::Protocol(message)
            if message == "Assistant message stream emitted done before start"));
    assert_eq!(*seen_reasoning.lock().expect("reasoning lock"), Some(None));
}

/// Keeps the raw settlement when cancellation interrupts `after_response`.
#[tokio::test]
async fn keeps_the_raw_settlement_when_cancellation_interrupts_after_response() {
    let final_message = assistant("raw", StopReason::Stop);
    let ended = Arc::new(Mutex::new(Vec::<AssistantMessage>::new()));
    let settled_final = final_message.clone();
    let request = stream_of(vec![
        AssistantMessageEvent::Start {
            partial: {
                let mut partial = settled_final.clone();
                partial.content = Vec::new();
                partial.stop_reason = StopReason::Pending;
                partial
            },
        },
        AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: settled_final.clone(),
        },
    ]);
    let aborting_after_response: AfterResponseHook = Arc::new(
        |_message: crate::harness::session::types::SettledAssistantMessage,
         _metadata: AssistantResponseMetadata,
         _context: Context| {
            let (sender, receiver) = tokio::sync::watch::channel(());
            drop(sender);
            let error: Box<dyn std::error::Error + Send + Sync> = Box::new(AbortRequested {
                cancellation: receiver,
            });
            Box::pin(async move { Err(error) })
        },
    );

    let result = stream_harness_assistant(
        &[user("prompt")],
        &HarnessAssistantStreamConfig {
            after_response: Some(aborting_after_response),
            observer: Arc::new(RecordingObserver {
                order: Arc::new(Mutex::new(Vec::new())),
                starts: Arc::new(Mutex::new(Vec::new())),
                updates: Arc::new(Mutex::new(Vec::new())),
                ends: Arc::clone(&ended),
                start_event_type: Arc::new(Mutex::new(None)),
            }),
            ..boundary_config(request, Arc::new(RecordingObserver::default()))
        },
        &background_context(),
    )
    .await
    .expect("the raw settlement stands");

    assert_eq!(result.message, final_message);
    {
        let ended = ended.lock().expect("ends lock");
        assert_eq!(ended.len(), 1);
        assert_eq!(ended[0], final_message);
    }
}

/// Returns provider error settlements through the same lifecycle; a
/// pre-generation error must not synthesize start.
#[tokio::test]
async fn returns_provider_error_settlements_through_the_same_lifecycle() {
    let final_message = assistant("provider failed", StopReason::Error);
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let settled_final = final_message.clone();
    let request = stream_of(vec![AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: settled_final.clone(),
    }]);

    let result = stream_harness_assistant(
        &[user("prompt")],
        &boundary_config(
            request,
            Arc::new(StartPanickingObserver {
                events: Arc::clone(&events),
            }),
        ),
        &background_context(),
    )
    .await
    .expect("the error settlement returns");

    assert_eq!(result.message, final_message);
    assert_eq!(
        *events.lock().expect("events lock"),
        vec!["end:error".to_owned()]
    );
}

/// An observer whose `start` panics — the restatement of upstream's
/// throwing start ("pre-generation error must not synthesize start").
#[derive(Debug)]
struct StartPanickingObserver {
    events: Arc<Mutex<Vec<String>>>,
}

impl AssistantStreamObserver for StartPanickingObserver {
    fn start(
        &self,
        _message: AssistantMessage,
        _event: &AssistantMessageEvent,
        _context: &Context,
    ) -> BoxedFuture<'_, ()> {
        panic!("pre-generation error must not synthesize start");
    }

    fn update(
        &self,
        _message: AssistantMessage,
        _event: &AssistantMessageEvent,
        _context: &Context,
    ) -> BoxedFuture<'_, ()> {
        Box::pin(async {})
    }

    fn end(
        &self,
        message: &crate::harness::session::types::SettledAssistantMessage,
        _context: &Context,
    ) -> BoxedFuture<'_, ()> {
        self.events
            .lock()
            .expect("events lock")
            .push(format!("end:{}", wire_stop_reason(message.stop_reason)));
        Box::pin(async {})
    }
}

// --- boundary tests: the branches the upstream suite leaves untested ---

/// An update arriving before start rejects with the event's wire name.
#[tokio::test]
async fn an_update_before_start_rejects_with_the_events_wire_name() {
    let request = stream_of(vec![AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "raw".to_owned(),
        partial: pending_partial(),
    }]);

    let error = stream_harness_assistant(
        &[user("prompt")],
        &boundary_config(request, Arc::new(RecordingObserver::default())),
        &background_context(),
    )
    .await
    .expect_err("an update before start rejects");

    assert!(matches!(&error, AssistantStreamError::Protocol(message)
            if message == "Assistant message stream emitted text_delta before start"));
}

/// A second start event rejects with the upstream message.
#[tokio::test]
async fn a_second_start_event_rejects() {
    let request = stream_of(vec![
        AssistantMessageEvent::Start {
            partial: pending_partial(),
        },
        AssistantMessageEvent::Start {
            partial: pending_partial(),
        },
    ]);

    let error = stream_harness_assistant(
        &[user("prompt")],
        &boundary_config(request, Arc::new(RecordingObserver::default())),
        &background_context(),
    )
    .await
    .expect_err("a second start rejects");

    assert!(matches!(&error, AssistantStreamError::Protocol(message)
            if message == "Assistant message stream emitted more than one start event"));
}

/// A failing context-transform hook propagates as a hook error before any
/// request runs.
#[tokio::test]
async fn a_failing_transform_context_hook_propagates_as_a_hook_error() {
    let transform_context: TransformContextHook = Arc::new(
        |_request_context: AssistantRequestContext,
         _context: &Context|
         -> BoxedFuture<
            '_,
            Result<AssistantRequestContext, Box<dyn std::error::Error + Send + Sync>>,
        > {
            let error: Box<dyn std::error::Error + Send + Sync> = "transform failed".into();
            Box::pin(async move { Err(error) })
        },
    );
    let request: AssistantRequestHook = Arc::new(
        |_ai_context: &pi_ai::types::Context,
         _options: pi_ai::types::SimpleStreamOptions,
         _context: &Context|
         -> BoxedFuture<'_, AssistantMessageEventStream> {
            panic!("the request must not run")
        },
    );

    let error = stream_harness_assistant(
        &[user("prompt")],
        &HarnessAssistantStreamConfig {
            transform_context: Some(transform_context),
            ..boundary_config(request, Arc::new(RecordingObserver::default()))
        },
        &background_context(),
    )
    .await
    .expect_err("a transform failure propagates");

    assert!(
        matches!(&error, AssistantStreamError::Hook(error) if error.to_string() == "transform failed")
    );
}

/// A non-abort `after_response` failure propagates as a hook error and
/// skips the end observer.
#[tokio::test]
async fn a_non_abort_after_response_failure_propagates_and_skips_the_end_observer() {
    let final_message = assistant("raw", StopReason::Stop);
    let ended = Arc::new(Mutex::new(Vec::<AssistantMessage>::new()));
    let settled_final = final_message.clone();
    let request = stream_of(vec![
        AssistantMessageEvent::Start {
            partial: settled_final.clone(),
        },
        AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: settled_final.clone(),
        },
    ]);
    let failing_after_response: AfterResponseHook = Arc::new(
        |_message: crate::harness::session::types::SettledAssistantMessage,
         _metadata: AssistantResponseMetadata,
         _context: Context| {
            let error: Box<dyn std::error::Error + Send + Sync> = "post-processing failed".into();
            Box::pin(async move { Err(error) })
        },
    );

    let error = stream_harness_assistant(
        &[user("prompt")],
        &HarnessAssistantStreamConfig {
            after_response: Some(failing_after_response),
            observer: Arc::new(RecordingObserver {
                order: Arc::new(Mutex::new(Vec::new())),
                starts: Arc::new(Mutex::new(Vec::new())),
                updates: Arc::new(Mutex::new(Vec::new())),
                ends: Arc::clone(&ended),
                start_event_type: Arc::new(Mutex::new(None)),
            }),
            ..boundary_config(request, Arc::new(RecordingObserver::default()))
        },
        &background_context(),
    )
    .await
    .expect_err("a post-processing failure propagates");

    assert!(matches!(&error, AssistantStreamError::Hook(error)
            if error.to_string() == "post-processing failed"));
    assert!(ended.lock().expect("ends lock").is_empty());
}

/// An `after_response` hook with no captured response metadata receives
/// the default (both fields absent).
#[tokio::test]
async fn an_after_response_hook_without_captured_metadata_receives_the_default() {
    let seen_metadata = Arc::new(Mutex::new(None::<AssistantResponseMetadata>));
    let metadata_handle = Arc::clone(&seen_metadata);
    let final_message = assistant("raw", StopReason::Stop);
    let settled_final = final_message.clone();
    let request = stream_of(vec![
        AssistantMessageEvent::Start {
            partial: settled_final.clone(),
        },
        AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: settled_final.clone(),
        },
    ]);
    let recording_after_response: AfterResponseHook = Arc::new(
        move |message: crate::harness::session::types::SettledAssistantMessage,
              metadata: AssistantResponseMetadata,
              _context: Context| {
            *metadata_handle.lock().expect("metadata lock") = Some(metadata);
            Box::pin(async move { Ok(message) })
        },
    );

    let _ = stream_harness_assistant(
        &[user("prompt")],
        &HarnessAssistantStreamConfig {
            after_response: Some(recording_after_response),
            ..boundary_config(request, Arc::new(RecordingObserver::default()))
        },
        &background_context(),
    )
    .await
    .expect("the stream settles");

    assert_eq!(
        *seen_metadata.lock().expect("metadata lock"),
        Some(AssistantResponseMetadata::default())
    );
}

/// An error event after start flows through the lifecycle and settles with
/// the failing message; the update observer stays untouched.
#[tokio::test]
async fn an_error_event_after_start_settles_through_the_lifecycle() {
    let final_message = assistant("provider failed", StopReason::Error);
    let starts = Arc::new(AtomicU64::new(0));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let settled_final = final_message.clone();
    let request = stream_of(vec![
        AssistantMessageEvent::Start {
            partial: {
                let mut partial = settled_final.clone();
                partial.content = Vec::new();
                partial.stop_reason = StopReason::Pending;
                partial
            },
        },
        AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: settled_final.clone(),
        },
    ]);

    let result = stream_harness_assistant(
        &[user("prompt")],
        &boundary_config(
            request,
            Arc::new(CountingEventsObserver {
                starts: Arc::clone(&starts),
                events: Arc::clone(&events),
            }),
        ),
        &background_context(),
    )
    .await
    .expect("the error settlement returns");

    assert_eq!(
        result.stop_reason,
        crate::harness::session::types::SettledStopReason::Error
    );
    assert_eq!(starts.load(Ordering::Relaxed), 1);
    assert_eq!(
        *events.lock().expect("events lock"),
        vec!["end:error".to_owned()]
    );
}

/// An observer counting starts and formatting the end settlements; the
/// lifecycle-order probe for the error-settlement boundary cases.
#[derive(Debug)]
struct CountingEventsObserver {
    starts: Arc<AtomicU64>,
    events: Arc<Mutex<Vec<String>>>,
}

impl AssistantStreamObserver for CountingEventsObserver {
    fn start(
        &self,
        _message: AssistantMessage,
        _event: &AssistantMessageEvent,
        _context: &Context,
    ) -> BoxedFuture<'_, ()> {
        self.starts.fetch_add(1, Ordering::Relaxed);
        Box::pin(async {})
    }

    fn update(
        &self,
        _message: AssistantMessage,
        _event: &AssistantMessageEvent,
        _context: &Context,
    ) -> BoxedFuture<'_, ()> {
        Box::pin(async {})
    }

    fn end(
        &self,
        message: &crate::harness::session::types::SettledAssistantMessage,
        _context: &Context,
    ) -> BoxedFuture<'_, ()> {
        self.events
            .lock()
            .expect("events lock")
            .push(format!("end:{}", wire_stop_reason(message.stop_reason)));
        Box::pin(async {})
    }
}

/// The config and error surfaces render their shapes; the error's source
/// chain carries the hook failure.
#[test]
fn the_config_and_error_surfaces_render_their_shapes() {
    let request = stream_of(Vec::new());
    let config = boundary_config(request, Arc::new(RecordingObserver::default()));
    let rendered = format!("{config:?}");
    assert!(rendered.contains("HarnessAssistantStreamConfig"));
    assert!(rendered.contains("system_prompt: \"system\""));
    assert!(rendered.contains("after_response: false"));

    let protocol = AssistantStreamError::Protocol("done before start".to_owned());
    assert_eq!(protocol.to_string(), "done before start");
    assert!(std::error::Error::source(&protocol).is_none());

    let hook_error: Box<dyn std::error::Error + Send + Sync> = "hook failed".into();
    let hook = AssistantStreamError::Hook(hook_error);
    assert_eq!(hook.to_string(), "hook failed");
    let source = std::error::Error::source(&hook).expect("the hook error's source");
    assert_eq!(source.to_string(), "hook failed");
}
