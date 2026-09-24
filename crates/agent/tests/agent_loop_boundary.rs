//! Boundary tests for the agent-loop restatements.
//!
//! Upstream's `agent-loop.test.ts` covers the loop's observable behavior;
//! these tests bind what the port restates: the no-default stream-fn
//! degradation, the continue-entry error contracts, the update drain, the
//! join-error recovery, abort batch behavior, preflight failures, and the
//! streamed-partial transcript path. The rest bind loop-level behaviors the
//! upstream suites pin at other layers (`agent.test.ts`, `e2e.test.ts`) or
//! hook contracts upstream documents: the failed/aborted response run-end,
//! follow-up continuation, per-request api-key resolution, tool errors, the
//! argument-shim failures, the after-tool-call overrides, the full partial
//! event family, the start-less update, and the mid-flight stream end.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod common;
use common::*;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use pi_agent_core::AgentContext;
use pi_agent_core::AgentEvent;
use pi_agent_core::AgentLoopConfig;
use pi_agent_core::AgentLoopError;
use pi_agent_core::AgentMessage;
use pi_agent_core::AgentToolResult;
use pi_agent_core::ToolExecutionMode;
use pi_agent_core::agent_loop;
use pi_agent_core::run_agent_loop;
use pi_agent_core::run_agent_loop_continue;
use pi_agent_core::set_default_stream_fn;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use pi_ai::types::SimpleStreamOptions;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::ToolCall;
use pi_ai::types::ToolResultBlock;
use serde_json::Value;
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn never_stream_fn() -> pi_agent_core::StreamFn {
    Arc::new(
        |_model, _context, _options| -> pi_ai::utils::event_stream::AssistantMessageEventStream {
            panic!("Unexpected stream call");
        },
    )
}

fn sink() -> pi_agent_core::AgentEventSink {
    Arc::new(|event| {
        Box::pin(async move {
            let _ = event;
        })
    })
}

fn assistant_last_context() -> AgentContext {
    AgentContext {
        system_prompt: String::new(),
        messages: vec![AgentMessage::Standard(Message::Assistant(
            create_assistant_message(vec![text_block("hi")], StopReason::Stop),
        ))],
        tools: Some(Vec::new()),
    }
}

/// Run `agent_loop` over `prompts` with `tools`, `config`, and `stream_fn`;
/// collect the events and return them with the run's settled messages.
async fn run_prompt_turn(
    prompts: Vec<AgentMessage>,
    tools: Vec<pi_agent_core::AgentTool>,
    config: AgentLoopConfig,
    stream_fn: pi_agent_core::StreamFn,
) -> (Vec<AgentEvent>, Vec<AgentMessage>) {
    let stream = agent_loop(
        prompts,
        test_context(Vec::new(), tools),
        config,
        None,
        Some(stream_fn),
    );
    let events = collect_events(&stream).await;
    let messages = stream.result().await;
    (events, messages)
}

/// Start a prompt run over the empty context and return the raw stream, for
/// the tests that iterate it themselves instead of settling it.
fn prompt_stream(
    config: AgentLoopConfig,
    stream_fn: Option<pi_agent_core::StreamFn>,
) -> pi_agent_core::AgentEventStream {
    agent_loop(
        vec![create_user_message("Hello")],
        AgentContext {
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: Some(Vec::new()),
        },
        config,
        None,
        stream_fn,
    )
}

/// Run one tool-call turn over `stream_fn` with `config`, `tools`, and
/// `signal`; returns the collected events.
async fn run_tool_batch_turn(
    config: AgentLoopConfig,
    tools: Vec<pi_agent_core::AgentTool>,
    signal: Option<CancellationToken>,
    stream_fn: pi_agent_core::StreamFn,
) -> Vec<AgentEvent> {
    let stream = agent_loop(
        vec![create_user_message("go")],
        test_context(Vec::new(), tools),
        config,
        signal,
        Some(stream_fn),
    );
    collect_events(&stream).await
}

/// The run's settled assistant message, the final provider response.
fn settled_assistant(messages: &[AgentMessage]) -> &pi_ai::types::AssistantMessage {
    messages
        .iter()
        .find_map(|message| match message {
            AgentMessage::Standard(Message::Assistant(assistant)) => Some(assistant),
            _ => None,
        })
        .expect("the run's assistant message")
}

/// The run's settled tool-result message.
fn settled_tool_result(messages: &[AgentMessage]) -> &pi_ai::types::ToolResultMessage {
    messages
        .iter()
        .find_map(|message| match message {
            AgentMessage::Standard(Message::ToolResult(tool_result)) => Some(tool_result),
            _ => None,
        })
        .expect("the run's tool result message")
}

/// The tool calls whose executions started, in emission order.
fn tool_execution_start_ids(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect()
}

/// The tool calls whose executions finalized, in emission order.
fn tool_execution_end_ids(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn run_agent_loop_surfaces_the_no_default_stream_fn_error() {
    set_default_stream_fn(None);
    let result = Box::pin(run_agent_loop(
        vec![create_user_message("Hello")],
        AgentContext {
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: Some(Vec::new()),
        },
        test_config(identity_converter()),
        sink(),
        None,
        None,
    ))
    .await;
    let AgentLoopError::NoDefaultStreamFn(error) = result.expect_err("no default is configured")
    else {
        panic!("the resolution error surfaces");
    };
    assert_eq!(
        error.to_string(),
        "No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn()."
    );
}

#[tokio::test]
async fn agent_loop_without_a_default_leaves_the_stream_open_after_the_prompt_events() {
    tokio::time::pause();
    set_default_stream_fn(None);
    let stream = prompt_stream(test_config(identity_converter()), None);

    // The prompt events emit before the default resolution fails, matching
    // the upstream evaluation order.
    let mut seen: Vec<&'static str> = Vec::new();
    for _expected in ["agent_start", "turn_start", "message_start", "message_end"] {
        let event = tokio::time::timeout(Duration::from_millis(100), stream.next())
            .await
            .expect("the prompt events arrive")
            .expect("the stream carries the prompt events");
        seen.push(event_type_name(&event));
    }
    assert_eq!(
        seen,
        ["agent_start", "turn_start", "message_start", "message_end"]
    );

    // The stream never ends, upstream's rejected-promise statement.
    let hang = tokio::time::timeout(Duration::from_millis(100), stream.next()).await;
    assert!(hang.is_err(), "the stream stays open after the run dies");
}

#[tokio::test]
async fn run_agent_loop_continue_surfaces_the_continue_errors_verbatim() {
    let empty_error = Box::pin(run_agent_loop_continue(
        AgentContext {
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: Some(Vec::new()),
        },
        test_config(identity_converter()),
        sink(),
        None,
        Some(never_stream_fn()),
    ))
    .await
    .expect_err("an empty context cannot continue");
    assert_eq!(
        empty_error.to_string(),
        "Cannot continue: no messages in context"
    );

    let assistant_error = Box::pin(run_agent_loop_continue(
        assistant_last_context(),
        test_config(identity_converter()),
        sink(),
        None,
        Some(never_stream_fn()),
    ))
    .await
    .expect_err("an assistant-last context cannot continue");
    assert_eq!(
        assistant_error.to_string(),
        "Cannot continue from message role: assistant"
    );
}

#[tokio::test]
async fn run_agent_loop_continue_surfaces_the_no_default_stream_fn_error() {
    set_default_stream_fn(None);
    let result = Box::pin(run_agent_loop_continue(
        AgentContext {
            system_prompt: String::new(),
            messages: vec![create_user_message("Hello")],
            tools: Some(Vec::new()),
        },
        test_config(identity_converter()),
        sink(),
        None,
        None,
    ))
    .await;
    assert!(matches!(result, Err(AgentLoopError::NoDefaultStreamFn(_))));
}

#[tokio::test]
async fn streamed_partials_replace_the_transcript_tail_and_emit_updates() {
    // A stream that opens, grows a text block, then completes.
    let partial_at_start = create_assistant_message(vec![text_block("")], StopReason::Pending);
    let mut partial_mid = partial_at_start.clone();
    if let Some(AssistantBlock::Text(text)) = partial_mid.content.first_mut() {
        text.text = "hel".to_string();
    }
    let final_message = create_assistant_message(vec![text_block("hello")], StopReason::Stop);

    let stream_fn: pi_agent_core::StreamFn = Arc::new(move |_model, _context, _options| {
        let stream = mock_stream();
        let start = stream.clone();
        let delta = stream.clone();
        let done = stream.clone();
        let start_partial = partial_at_start.clone();
        let delta_partial = partial_mid.clone();
        let final_message = final_message.clone();
        tokio::spawn(async move {
            start.push(AssistantMessageEvent::Start {
                partial: start_partial,
            });
        });
        tokio::spawn(async move {
            delta.push(AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "hel".to_string(),
                partial: delta_partial,
            });
        });
        tokio::spawn(async move {
            done.push(done_event(final_message));
        });
        stream
    });

    let (events, messages) = run_prompt_turn(
        vec![create_user_message("Hello")],
        Vec::new(),
        test_config(identity_converter()),
        stream_fn,
    )
    .await;

    let updates: Vec<&AgentEvent> = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::MessageUpdate { .. }))
        .collect();
    assert_eq!(updates.len(), 1, "the text delta emits one message_update");
    let AgentEvent::MessageUpdate {
        message,
        assistant_message_event,
    } = updates[0]
    else {
        panic!("a message_update event");
    };
    assert!(matches!(
        assistant_message_event,
        AssistantMessageEvent::TextDelta { .. }
    ));
    assert_eq!(agent_message_role(message), "assistant");

    // The transcript's assistant entry is the final message, not a partial.
    assert_eq!(settled_assistant(&messages).stop_reason, StopReason::Stop);
}

/// Run one tool-call turn against `tool` with `config` and return the
/// collected events, the run's messages, and the finalized tool result
/// (call id, result, error flag).
async fn run_tool_turn(
    tools: Vec<pi_agent_core::AgentTool>,
    tool_call_name: &str,
    config: AgentLoopConfig,
    signal: Option<CancellationToken>,
) -> (
    Vec<AgentEvent>,
    Vec<AgentMessage>,
    (String, AgentToolResult, bool),
) {
    let stream_fn = two_call_stream_fn(
        create_assistant_message(
            vec![create_tool_call("tool-1", tool_call_name, json!({}))],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );
    let stream = agent_loop(
        vec![create_user_message("go")],
        test_context(Vec::new(), tools),
        config,
        signal,
        Some(stream_fn),
    );
    let events = collect_events(&stream).await;
    let messages = stream.result().await;
    let finalized = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                result,
                is_error,
                ..
            } => Some((tool_call_id.clone(), result.clone(), *is_error)),
            _ => None,
        })
        .expect("a tool_execution_end event");
    (events, messages, finalized)
}

/// The text of a tool result's first text block, if any.
fn result_text(result: &AgentToolResult) -> Option<String> {
    result.content.iter().find_map(|block| match block {
        ToolResultBlock::Text(text) => Some(text.text.clone()),
        ToolResultBlock::Image(_) => None,
    })
}

#[tokio::test]
async fn tool_updates_drain_in_emission_order_before_the_execution_settles() {
    let executed = Arc::new(AtomicBool::new(false));
    let executed_for_tool = Arc::clone(&executed);
    let tool = suite_tool(
        "progress",
        json!({ "type": "object", "properties": {} }),
        None,
        Arc::new(move |_tool_call_id, _args: &Value, _signal, on_update| {
            let executed = Arc::clone(&executed_for_tool);
            Box::pin(async move {
                if let Some(on_update) = on_update {
                    on_update(&empty_tool_result(json!({ "stage": "one" })));
                    on_update(&empty_tool_result(json!({ "stage": "two" })));
                }
                executed.store(true, Ordering::Relaxed);
                Ok(empty_tool_result(json!({ "stage": "done" })))
            })
        }),
    );

    let (events, _messages, _finalized) = run_tool_turn(
        vec![tool],
        "progress",
        test_config(identity_converter()),
        None,
    )
    .await;
    let update_details: Vec<Value> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionUpdate { partial_result, .. } => {
                Some(partial_result.details.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        update_details,
        vec![json!({ "stage": "one" }), json!({ "stage": "two" })]
    );

    // Updates precede the tool_execution_end, which carries the final result.
    let first_update = events
        .iter()
        .position(|event| matches!(event, AgentEvent::ToolExecutionUpdate { .. }))
        .expect("an update event");
    let end = events
        .iter()
        .position(|event| matches!(event, AgentEvent::ToolExecutionEnd { .. }))
        .expect("a tool_execution_end event");
    assert!(first_update < end);
    assert!(executed.load(Ordering::Relaxed));
}

#[tokio::test]
async fn a_panicking_parallel_tool_yields_an_error_tool_result() {
    let tool = suite_tool(
        "boom",
        json!({ "type": "object" }),
        None,
        Arc::new(|_tool_call_id, _args: &Value, _signal, _on_update| {
            Box::pin(async move {
                panic!("boom");
            })
        }),
    );

    let (_events, messages, (_id, result, is_error)) =
        run_tool_turn(vec![tool], "boom", test_config(identity_converter()), None).await;
    assert!(is_error, "the panicked tool's result is an error");
    let text = result_text(&result);
    assert!(text.is_some_and(|text| text.contains("panicked")));

    // The transcript still carries the tool result message.
    assert!(
        messages
            .iter()
            .any(|message| matches!(message, AgentMessage::Standard(Message::ToolResult(_))))
    );
}

#[tokio::test]
async fn an_unknown_tool_yields_an_immediate_error_result() {
    let (_events, _messages, (_id, result, is_error)) = run_tool_turn(
        Vec::new(),
        "missing",
        test_config(identity_converter()),
        None,
    )
    .await;
    assert!(is_error);
    assert_eq!(
        result_text(&result).as_deref(),
        Some("Tool missing not found")
    );
}

#[tokio::test]
async fn schema_invalid_arguments_fail_without_executing() {
    let executed = Arc::new(AtomicBool::new(false));
    let executed_for_tool = Arc::clone(&executed);
    let tool = suite_tool(
        "strict",
        json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
        }),
        None,
        Arc::new(move |_tool_call_id, _args: &Value, _signal, _on_update| {
            let executed = Arc::clone(&executed_for_tool);
            Box::pin(async move {
                executed.store(true, Ordering::Relaxed);
                Ok(empty_tool_result(json!({})))
            })
        }),
    );

    let (_events, _messages, (_id, result, is_error)) = run_tool_turn(
        vec![tool],
        "strict",
        test_config(identity_converter()),
        None,
    )
    .await;
    assert!(is_error);
    assert!(
        !result.content.is_empty(),
        "the validation error is reported as text"
    );
    assert!(!executed.load(Ordering::Relaxed));
}

#[tokio::test]
async fn a_blocked_call_without_a_reason_carries_the_default_message() {
    let tool = suite_tool(
        "gated",
        json!({ "type": "object" }),
        None,
        Arc::new(|_tool_call_id, _args: &Value, _signal, _on_update| {
            Box::pin(std::future::ready(Ok(empty_tool_result(json!({})))))
        }),
    );
    let blocked_gate: pi_agent_core::BeforeToolCall =
        Arc::new(|_context: pi_agent_core::BeforeToolCallContext, _signal| {
            Box::pin(std::future::ready(Some(
                pi_agent_core::BeforeToolCallResult {
                    block: Some(true),
                    reason: None,
                    terminate: None,
                },
            )))
        });
    let config = AgentLoopConfig {
        tool_execution: Some(ToolExecutionMode::Sequential),
        before_tool_call: Some(blocked_gate),
        ..test_config(identity_converter())
    };

    let stream_fn = two_call_stream_fn(
        create_assistant_message(
            vec![create_tool_call("tool-1", "gated", json!({}))],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );

    let events = run_tool_batch_turn(config, vec![tool], None, stream_fn).await;

    let (result, is_error) = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolExecutionEnd {
                result, is_error, ..
            } => Some((result, is_error)),
            _ => None,
        })
        .expect("a tool_execution_end event");
    assert!(is_error);
    let text = result.content.iter().find_map(|block| match block {
        ToolResultBlock::Text(text) => Some(text.text.clone()),
        ToolResultBlock::Image(_) => None,
    });
    assert_eq!(text.as_deref(), Some("Tool execution was blocked"));
}

#[tokio::test]
async fn an_abort_mid_batch_stops_preparing_the_remaining_calls() {
    let token = CancellationToken::new();
    let executed = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool("counting", Arc::clone(&executed));

    // The gate cancels the run's token during the first call's preparation,
    // so the prepare loop's post-hook abort check fires and the remaining
    // calls are never prepared.
    let aborting_gate: pi_agent_core::BeforeToolCall = Arc::new(
        |_context: pi_agent_core::BeforeToolCallContext, signal: Option<CancellationToken>| {
            Box::pin(async move {
                if let Some(signal) = signal {
                    signal.cancel();
                }
                None
            })
        },
    );

    let stream_fn = two_call_stream_fn(
        create_assistant_message(
            vec![
                create_tool_call("tool-1", "counting", json!({})),
                create_tool_call("tool-2", "counting", json!({})),
                create_tool_call("tool-3", "counting", json!({})),
            ],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );

    let events = run_tool_batch_turn(
        AgentLoopConfig {
            before_tool_call: Some(aborting_gate),
            ..test_config(identity_converter())
        },
        vec![tool],
        Some(token),
        stream_fn,
    )
    .await;

    assert_eq!(
        tool_execution_start_ids(&events).len(),
        1,
        "the remaining calls are never prepared"
    );

    assert_eq!(tool_execution_end_ids(&events), ["tool-1"]);
    assert_eq!(executed.load(Ordering::Relaxed), 0, "no tool runs");
}

#[tokio::test]
async fn a_cancelled_signal_before_the_batch_errors_every_prepared_call() {
    let token = CancellationToken::new();
    token.cancel();
    let executed = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool("counting", Arc::clone(&executed));

    let stream_fn = two_call_stream_fn(
        create_assistant_message(
            vec![
                create_tool_call("tool-1", "counting", json!({})),
                create_tool_call("tool-2", "counting", json!({})),
            ],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );

    let events = run_tool_batch_turn(
        test_config(identity_converter()),
        vec![tool],
        Some(token),
        stream_fn,
    )
    .await;

    let aborted_ends = events
        .iter()
        .filter(|event| match event {
            AgentEvent::ToolExecutionEnd {
                result, is_error, ..
            } => {
                *is_error
                    && result.content.iter().any(|block| match block {
                        ToolResultBlock::Text(text) => text.text == "Operation aborted",
                        ToolResultBlock::Image(_) => false,
                    })
            }
            _ => false,
        })
        .count();
    assert_eq!(
        aborted_ends, 1,
        "the first prepared call reports the abort; the loop breaks before preparing the rest"
    );
    assert_eq!(executed.load(Ordering::Relaxed), 0, "no tool runs");
}

#[tokio::test]
async fn a_failed_or_aborted_assistant_response_ends_the_run_without_tool_results() {
    for stop_reason in [StopReason::Error, StopReason::Aborted] {
        let executed = Arc::new(AtomicUsize::new(0));
        let tool = counting_tool("counting", Arc::clone(&executed));
        let message = create_assistant_message(
            vec![create_tool_call("tool-1", "counting", json!({}))],
            stop_reason,
        );

        let (events, messages) = run_prompt_turn(
            vec![create_user_message("go")],
            vec![tool],
            test_config(identity_converter()),
            single_response_stream_fn(message),
        )
        .await;

        let event_types: Vec<&str> = events.iter().map(event_type_name).collect();
        assert_eq!(
            event_types,
            [
                "agent_start",
                "turn_start",
                "message_start",
                "message_end",
                "message_start",
                "message_end",
                "turn_end",
                "agent_end",
            ],
            "the turn closes after the failed response"
        );
        let turn_end = events
            .iter()
            .find_map(|event| match event {
                AgentEvent::TurnEnd { tool_results, .. } => Some(tool_results.clone()),
                _ => None,
            })
            .expect("a turn_end event");
        assert!(
            turn_end.is_empty(),
            "a failed response carries no tool results"
        );
        assert_eq!(
            executed.load(Ordering::Relaxed),
            0,
            "no tool call from a failed response runs"
        );

        let roles: Vec<&str> = messages.iter().map(agent_message_role).collect();
        assert_eq!(roles, ["user", "assistant"]);
        assert_eq!(settled_assistant(&messages).stop_reason, stop_reason);
    }
}

#[tokio::test]
async fn follow_up_messages_keep_the_run_alive_and_inject_before_the_next_turn() {
    let polls = Arc::new(AtomicUsize::new(0));
    let polls_for_hook = Arc::clone(&polls);
    let get_follow_up_messages: pi_agent_core::GetFollowUpMessages = Arc::new(move || {
        let polls = Arc::clone(&polls_for_hook);
        Box::pin(async move {
            if polls.fetch_add(1, Ordering::Relaxed) == 0 {
                vec![create_user_message("the follow-up")]
            } else {
                Vec::new()
            }
        })
    });

    let llm_calls = Arc::new(AtomicUsize::new(0));
    let stream_fn = counted_two_call_stream_fn(
        &llm_calls,
        create_assistant_message(
            vec![create_tool_call("tool-1", "echo", json!({}))],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("after the follow-up")], StopReason::Stop),
    );

    let tool = counting_tool("echo", Arc::new(AtomicUsize::new(0)));
    let (events, messages) = run_prompt_turn(
        vec![create_user_message("go")],
        vec![tool],
        AgentLoopConfig {
            get_follow_up_messages: Some(get_follow_up_messages),
            ..test_config(identity_converter())
        },
        stream_fn,
    )
    .await;

    assert_eq!(
        llm_calls.load(Ordering::Relaxed),
        3,
        "the follow-up drives one more provider call"
    );
    assert_eq!(polls.load(Ordering::Relaxed), 2, "one poll per loop stop");
    let turn_starts = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::TurnStart))
        .count();
    assert_eq!(turn_starts, 3, "the follow-up starts another turn");
    let injected = events
        .iter()
        .filter(|event| {
            matches!(event, AgentEvent::MessageStart { message }
                if agent_message_role(message) == "user")
        })
        .count();
    assert_eq!(
        injected, 2,
        "the prompt and the follow-up each enter the transcript"
    );
    let roles: Vec<&str> = messages.iter().map(agent_message_role).collect();
    assert_eq!(
        roles,
        [
            "user",
            "assistant",
            "toolResult",
            "assistant",
            "user",
            "assistant",
        ],
        "the follow-up joins the transcript after the agent would have stopped"
    );
}

#[tokio::test]
async fn get_api_key_resolves_the_key_each_request_and_falls_back_when_it_answers_none() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_hook = Arc::clone(&calls);
    let get_api_key: pi_agent_core::GetApiKey = Arc::new(move |_provider: &str| {
        let calls = Arc::clone(&calls_for_hook);
        Box::pin(async move {
            if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                Some("fresh-key".to_string())
            } else {
                None
            }
        })
    });

    let keys = Arc::new(std::sync::Mutex::new(Vec::<Option<String>>::new()));
    let keys_for_stream = Arc::clone(&keys);
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let inner = counted_two_call_stream_fn(
        &llm_calls,
        create_assistant_message(
            vec![create_tool_call("tool-1", "echo", json!({}))],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );
    let stream_fn: pi_agent_core::StreamFn = Arc::new(
        move |model, context, options: Option<&SimpleStreamOptions>| {
            keys_for_stream
                .lock()
                .expect("keys lock")
                .push(options.and_then(|options| options.api_key.clone()));
            inner(model, context, options)
        },
    );

    let mut config = test_config(identity_converter());
    config.stream_options.api_key = Some("stale-key".to_string());
    config.get_api_key = Some(get_api_key);
    let executed = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool("echo", Arc::clone(&executed));

    let stream = agent_loop(
        vec![create_user_message("go")],
        test_context(Vec::new(), vec![tool]),
        config,
        None,
        Some(stream_fn),
    );
    let _ = collect_events(&stream).await;

    assert_eq!(llm_calls.load(Ordering::Relaxed), 2);
    let captured: Vec<Option<String>> = keys.lock().expect("keys lock").clone();
    assert_eq!(
        captured,
        [Some("fresh-key".to_string()), Some("stale-key".to_string()),],
        "the hook's key wins when present; the config key applies when it answers None"
    );
}

#[tokio::test]
async fn a_tool_error_becomes_an_error_tool_result() {
    let tool = suite_tool(
        "fallible",
        json!({ "type": "object" }),
        None,
        Arc::new(|_tool_call_id, _args: &Value, _signal, _on_update| {
            Box::pin(async move { Err("disk on fire".into()) })
        }),
    );

    let (_events, messages, (_id, result, is_error)) = run_tool_turn(
        vec![tool],
        "fallible",
        test_config(identity_converter()),
        None,
    )
    .await;
    assert!(is_error, "a tool error is an error result");
    assert_eq!(result_text(&result).as_deref(), Some("disk on fire"));
    assert!(settled_tool_result(&messages).is_error);
}

fn shim_tool(
    prepare_arguments: pi_agent_core::AgentToolPrepareArguments,
    executed: Arc<AtomicBool>,
) -> pi_agent_core::AgentTool {
    let mut tool = suite_tool(
        "shimmed",
        json!({ "type": "object" }),
        None,
        Arc::new(move |_tool_call_id, _args: &Value, _signal, _on_update| {
            let executed = Arc::clone(&executed);
            Box::pin(async move {
                executed.store(true, Ordering::Relaxed);
                Ok(empty_tool_result(json!({})))
            })
        }),
    );
    tool.prepare_arguments = Some(prepare_arguments);
    tool
}

/// Run one shimmed tool call; returns the finalized result's first text and
/// the loop's error flag.
async fn run_shim_case(
    prepare_arguments: pi_agent_core::AgentToolPrepareArguments,
    executed: Arc<AtomicBool>,
) -> (Option<String>, bool) {
    let (_events, _messages, (_id, result, is_error)) = run_tool_turn(
        vec![shim_tool(prepare_arguments, Arc::clone(&executed))],
        "shimmed",
        test_config(identity_converter()),
        None,
    )
    .await;
    (result_text(&result), is_error)
}

#[tokio::test]
async fn a_failing_prepare_arguments_shim_fails_the_call_without_executing() {
    let executed = Arc::new(AtomicBool::new(false));
    let prepare_arguments: pi_agent_core::AgentToolPrepareArguments =
        Arc::new(|_arguments: &Value| Err("shim exploded".into()));

    let (text, is_error) = run_shim_case(prepare_arguments, Arc::clone(&executed)).await;
    assert!(is_error);
    assert_eq!(text.as_deref(), Some("shim exploded"));
    assert!(!executed.load(Ordering::Relaxed), "the tool never runs");
}

#[tokio::test]
async fn a_prepare_arguments_shim_returning_a_non_object_fails_the_call() {
    let executed = Arc::new(AtomicBool::new(false));
    let prepare_arguments: pi_agent_core::AgentToolPrepareArguments =
        Arc::new(|_arguments: &Value| Ok(json!([1, 2, 3])));

    let (text, is_error) = run_shim_case(prepare_arguments, Arc::clone(&executed)).await;
    assert!(is_error);
    assert_eq!(
        text.as_deref(),
        Some("prepareArguments must return an object")
    );
    assert!(!executed.load(Ordering::Relaxed), "the tool never runs");
}

#[tokio::test]
async fn after_tool_call_overrides_content_details_and_the_error_flag() {
    let after_tool_call: pi_agent_core::AfterToolCall =
        Arc::new(|_context: pi_agent_core::AfterToolCallContext, _signal| {
            Box::pin(async move {
                Some(pi_agent_core::AfterToolCallResult {
                    content: Some(vec![ToolResultBlock::Text(TextContent {
                        text: "overridden".to_string(),
                        text_signature: None,
                    })]),
                    details: Some(json!({ "stage": "hook" })),
                    is_error: Some(true),
                    ..pi_agent_core::AfterToolCallResult::default()
                })
            })
        });
    let tool = suite_tool(
        "observed",
        json!({ "type": "object" }),
        None,
        Arc::new(|_tool_call_id, _args: &Value, _signal, _on_update| {
            Box::pin(async move { Ok(empty_tool_result(json!({ "stage": "executed" }))) })
        }),
    );

    let (_events, messages, (_id, result, is_error)) = run_tool_turn(
        vec![tool],
        "observed",
        AgentLoopConfig {
            after_tool_call: Some(after_tool_call),
            ..test_config(identity_converter())
        },
        None,
    )
    .await;
    assert!(is_error, "the override's error flag wins");
    assert_eq!(result_text(&result).as_deref(), Some("overridden"));
    assert_eq!(result.details, json!({ "stage": "hook" }));
    assert!(settled_tool_result(&messages).is_error);
}

#[tokio::test]
async fn the_no_default_stream_fn_error_carries_its_message_and_its_source() {
    set_default_stream_fn(None);
    let result = Box::pin(run_agent_loop(
        vec![create_user_message("Hello")],
        AgentContext {
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: Some(Vec::new()),
        },
        test_config(identity_converter()),
        sink(),
        None,
        None,
    ))
    .await
    .expect_err("no default is configured");
    assert_eq!(
        result.to_string(),
        "No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn().",
        "the wrapper displays the inner error's message"
    );
    let source = std::error::Error::source(&result).expect("the inner error is the source");
    assert_eq!(
        source.to_string(),
        "No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn()."
    );

    // The continue-entry errors carry no source.
    assert!(std::error::Error::source(&AgentLoopError::NoMessages).is_none());
    assert!(std::error::Error::source(&AgentLoopError::ContinueFromAssistant).is_none());
}

/// The full partial-event family upstream's protocol streams, pushed in
/// emission order: every partial-bearing event between `start` and the final
/// `done`.
fn push_partial_event_family(
    push: &pi_ai::utils::event_stream::AssistantMessageEventStream,
    partial: &pi_ai::types::AssistantMessage,
    tool_call: ToolCall,
    final_message: &pi_ai::types::AssistantMessage,
) {
    push.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });
    push.push(AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: partial.clone(),
    });
    push.push(AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "hello".to_string(),
        partial: partial.clone(),
    });
    push.push(AssistantMessageEvent::TextEnd {
        content_index: 0,
        content: "hello".to_string(),
        partial: partial.clone(),
    });
    push.push(AssistantMessageEvent::ThinkingStart {
        content_index: 1,
        partial: partial.clone(),
    });
    push.push(AssistantMessageEvent::ThinkingDelta {
        content_index: 1,
        delta: "hm".to_string(),
        partial: partial.clone(),
    });
    push.push(AssistantMessageEvent::ThinkingEnd {
        content_index: 1,
        content: "hm".to_string(),
        partial: partial.clone(),
    });
    push.push(AssistantMessageEvent::ToolcallStart {
        content_index: 2,
        partial: partial.clone(),
    });
    push.push(AssistantMessageEvent::ToolcallDelta {
        content_index: 2,
        delta: "{}".to_string(),
        partial: partial.clone(),
    });
    push.push(AssistantMessageEvent::ToolcallEnd {
        content_index: 2,
        tool_call,
        partial: partial.clone(),
    });
    push.push(done_event(final_message.clone()));
}

#[tokio::test]
async fn the_partial_event_family_replaces_the_transcript_tail_and_emits_updates() {
    let partial = create_assistant_message(vec![text_block("")], StopReason::Pending);
    let final_message = create_assistant_message(vec![text_block("done")], StopReason::Stop);
    let tool_call = ToolCall {
        id: "tool-1".to_string(),
        name: "echo".to_string(),
        arguments: serde_json::Map::new(),
        thought_signature: None,
        namespace: None,
    };

    let stream_fn: pi_agent_core::StreamFn = Arc::new(move |_model, _context, _options| {
        let stream = mock_stream();
        let push = stream.clone();
        let partial = partial.clone();
        let final_message = final_message.clone();
        let tool_call = tool_call.clone();
        tokio::spawn(async move {
            push_partial_event_family(&push, &partial, tool_call, &final_message);
        });
        stream
    });

    let (events, messages) = run_prompt_turn(
        vec![create_user_message("Hello")],
        Vec::new(),
        test_config(identity_converter()),
        stream_fn,
    )
    .await;

    let update_kinds: Vec<&'static str> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageUpdate {
                assistant_message_event,
                ..
            } => Some(match assistant_message_event {
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
            }),
            _ => None,
        })
        .collect();
    assert_eq!(
        update_kinds,
        [
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ],
        "every partial event replaces the transcript tail and emits one update"
    );

    // The transcript's assistant entry is the final message, not a partial.
    assert_eq!(settled_assistant(&messages).stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn an_update_without_a_start_event_does_not_reach_the_transcript_or_events() {
    let partial = create_assistant_message(vec![text_block("")], StopReason::Pending);
    let final_message = create_assistant_message(vec![text_block("done")], StopReason::Stop);
    let stream_fn: pi_agent_core::StreamFn = Arc::new(move |_model, _context, _options| {
        let stream = mock_stream();
        let delta = stream.clone();
        let partial = partial.clone();
        let final_message = final_message.clone();
        tokio::spawn(async move {
            delta.push(AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "orphan".to_string(),
                partial,
            });
        });
        push_done(&stream, final_message);
        stream
    });

    let (events, messages) = run_prompt_turn(
        vec![create_user_message("Hello")],
        Vec::new(),
        test_config(identity_converter()),
        stream_fn,
    )
    .await;

    let updates = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::MessageUpdate { .. }))
        .count();
    assert_eq!(
        updates, 0,
        "an update before the start event never reaches the events"
    );
    assert_eq!(
        settled_assistant(&messages).stop_reason,
        StopReason::Stop,
        "the transcript's assistant entry is the final message, not the orphan partial"
    );
}

#[tokio::test]
async fn a_stream_that_ends_mid_flight_leaves_the_agent_stream_open() {
    tokio::time::pause();
    let stream_fn: pi_agent_core::StreamFn = Arc::new(move |_model, _context, _options| {
        let stream = mock_stream();
        let closer = stream.clone();
        tokio::spawn(async move {
            closer.end(None);
        });
        stream
    });

    let stream = prompt_stream(test_config(identity_converter()), Some(stream_fn));

    // The prompt events emit before the provider stream ends mid-flight.
    for expected in ["agent_start", "turn_start", "message_start", "message_end"] {
        let event = tokio::time::timeout(Duration::from_millis(100), stream.next())
            .await
            .expect("the prompt events arrive")
            .expect("the stream carries the prompt events");
        assert_eq!(event_type_name(&event), expected);
    }

    // The run never settles, upstream's never-settling promise.
    let hang = tokio::time::timeout(Duration::from_millis(100), stream.next()).await;
    assert!(
        hang.is_err(),
        "the stream stays open when the provider stream ends mid-flight"
    );
}

#[tokio::test]
async fn a_cancelled_signal_mid_sequential_batch_stops_the_remaining_calls() {
    let token = CancellationToken::new();
    let executed = Arc::new(AtomicUsize::new(0));
    let executed_for_tool = Arc::clone(&executed);
    let aborting_tool = suite_tool(
        "solo",
        json!({ "type": "object" }),
        None,
        Arc::new(
            move |_tool_call_id, _args: &Value, signal: Option<&CancellationToken>, _on_update| {
                let executed = Arc::clone(&executed_for_tool);
                Box::pin(async move {
                    if let Some(signal) = signal {
                        signal.cancel();
                    }
                    executed.fetch_add(1, Ordering::Relaxed);
                    Ok(empty_tool_result(json!({})))
                })
            },
        ),
    );

    let stream_fn = two_call_stream_fn(
        create_assistant_message(
            vec![
                create_tool_call("tool-1", "solo", json!({})),
                create_tool_call("tool-2", "solo", json!({})),
            ],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );

    let events = run_tool_batch_turn(
        AgentLoopConfig {
            tool_execution: Some(ToolExecutionMode::Sequential),
            ..test_config(identity_converter())
        },
        vec![aborting_tool],
        Some(token),
        stream_fn,
    )
    .await;

    assert_eq!(
        tool_execution_start_ids(&events),
        ["tool-1"],
        "the loop breaks before the next call starts"
    );
    assert_eq!(tool_execution_end_ids(&events), ["tool-1"]);
    assert_eq!(executed.load(Ordering::Relaxed), 1, "one call runs");
}
