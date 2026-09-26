//! The `Agent`-class suite, ported 1:1 from upstream `test/agent.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! `MockAssistantStream` restates as `common::mock_stream()` — the same
//! terminal-event/extractor pair over pi-ai's `EventStream` — and the mock
//! pushes land on spawned tasks, the `queueMicrotask` analog. The abort
//! pollers (upstream `setTimeout(checkAbort, 5)`) rest as direct
//! `signal.cancelled()` awaits on the run's token; the deferred-promise
//! gates are tokio `Notify`/oneshot pairs; the `setTimeout` settles run on
//! the paused tokio clock, which advances whenever every task is idle.
//! Upstream's throwing closures restate as panicking stream fns and lazy
//! default resolution — both reach the same failure lifecycle.
//!
//! The two late-update cases (`agent.test.ts:301`, `agent.test.ts:366`)
//! restate structurally: the update callback's reference ends with
//! `execute`'s future, so a post-settlement push cannot exist — the loop's
//! accepting gate (the agent-loop child) pins the recorded equivalent, and
//! these tests pin the end-to-end update path and post-run event stability.
//! Upstream's `unhandledRejection` bookkeeping has no Rust counterpart.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod common;
use common::*;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use pi_agent_core::Agent;
use pi_agent_core::AgentError;
use pi_agent_core::AgentEvent;
use pi_agent_core::AgentInitialState;
use pi_agent_core::AgentListener;
use pi_agent_core::AgentLoopTurnUpdate;
use pi_agent_core::AgentMessage;
use pi_agent_core::AgentOptions;
use pi_agent_core::AgentTool;
use pi_agent_core::AgentToolResult;
use pi_agent_core::ShouldStopAfterTurnContext;
use pi_agent_core::StreamFn;
use pi_agent_core::ThinkingLevel;
use pi_agent_core::set_default_stream_fn;
use pi_ai::auth::resolve::now_ms;
use pi_ai::providers::catalog::get_builtin_model;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::UserBlock;
use pi_ai::types::UserContent;
use pi_ai::types::UserMessage;
use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

/// The stream fn most tests assert against: never called.
fn unused_stream_fn() -> StreamFn {
    Arc::new(|_model, _context, _options| panic!("Unexpected stream call"))
}

/// The stream fn that answers every call with one final "ok" message.
fn responding_stream_fn() -> StreamFn {
    single_response_stream_fn(assistant_text("ok"))
}

/// The upstream `createAssistantMessage(text)` shape.
fn assistant_text(text: &str) -> AssistantMessage {
    create_assistant_message(vec![text_block(text)], StopReason::Stop)
}

/// The upstream `createAssistantToolUseMessage(content)` shape.
fn assistant_tool_use(content: Vec<AssistantBlock>) -> AssistantMessage {
    create_assistant_message(content, StopReason::ToolUse)
}

/// The upstream user-message literal with block-array content.
fn user_message_blocks(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })]),
        timestamp: now_ms(),
    }))
}

/// A listener closure wrapped for `Agent::subscribe`.
fn sync_listener(
    f: impl Fn(&AgentEvent, &CancellationToken) + Send + Sync + 'static,
) -> AgentListener {
    Arc::new(move |event, token| {
        f(&event, &token);
        Box::pin(async {})
    })
}

/// The roles of the transcript, upstream's `.map(m => m.role)`.
fn transcript_roles(agent: &Agent) -> Vec<String> {
    let state = agent.state();
    state
        .messages
        .iter()
        .map(|message| agent_message_role(message).to_owned())
        .collect()
}

/// Every `tool_execution_update` event's recorded details.
fn update_details(events: &[AgentEvent]) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionUpdate { partial_result, .. } => {
                Some(partial_result.details.clone())
            }
            _ => None,
        })
        .collect()
}

/// The tool result whose `terminate: true` ends the batch, upstream's
/// `terminate: true` results.
fn terminating_tool_result(text: &str, details: serde_json::Value) -> AgentToolResult {
    AgentToolResult {
        terminate: Some(true),
        ..text_tool_result(text, details)
    }
}

/// The stream fn upstream's abort-polling mocks state: pushes a start
/// partial, then answers the run's abort with an `aborted` error event. The
/// 5 ms poll loop restates as a direct cancellation await on the run's
/// token, the same signal the options carry.
fn abort_aware_stream_fn() -> StreamFn {
    Arc::new(move |_model, _context, options| {
        let signal = options.and_then(|options| options.transport_options.signal.clone());
        let stream = mock_stream();
        let push = stream.clone();
        tokio::spawn(async move {
            push.push(AssistantMessageEvent::Start {
                partial: assistant_text(""),
            });
            match signal {
                Some(signal) => signal.cancelled().await,
                None => std::future::pending::<()>().await,
            }
            push.push(AssistantMessageEvent::Error {
                reason: StopReason::Aborted,
                error: assistant_text("Aborted"),
            });
        });
        stream
    })
}

/// The request-counting stream fn of the tool-call round trips: the first
/// call returns the tool-use message, later calls return the final text.
fn tool_then_final_stream_fn(request_count: &Arc<AtomicUsize>, final_text: &str) -> StreamFn {
    let request_count = Arc::clone(request_count);
    let final_text = final_text.to_owned();
    Arc::new(move |_model, _context, _options| {
        let index = request_count.fetch_add(1, Ordering::Relaxed);
        let stream = mock_stream();
        let push = stream.clone();
        let final_message = assistant_text(&final_text);
        tokio::spawn(async move {
            let message = if index == 0 {
                assistant_tool_use(vec![create_tool_call("tool-1", "noop", json!({}))])
            } else {
                final_message
            };
            push.push(done_event(message));
        });
        stream
    })
}

/// A no-op tool, upstream's noop fixture.
fn noop_tool(result_text: &str) -> AgentTool {
    let result_text = result_text.to_owned();
    suite_tool(
        "noop",
        json!({ "type": "object", "properties": {} }),
        None,
        Arc::new(
            move |_tool_call_id, _args: &serde_json::Value, _signal, _on_update| {
                let result = text_tool_result(&result_text, json!({}));
                Box::pin(async move { Ok(result) })
            },
        ),
    )
}

#[tokio::test]
async fn uses_the_configured_default_when_a_legacy_caller_omits_stream_fn() {
    tokio::time::pause();
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_stream = Arc::clone(&calls);
    set_default_stream_fn(Some(Arc::new(move |_model, _context, _options| {
        calls_for_stream.fetch_add(1, Ordering::Relaxed);
        let stream = mock_stream();
        push_done(&stream, assistant_text("fallback"));
        stream
    })));

    let agent = Agent::new(AgentOptions::default());
    agent.prompt("Hello").await.expect("the prompt runs");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    set_default_stream_fn(None);
}

#[tokio::test]
async fn creates_an_agent_instance_with_default_state() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..AgentOptions::default()
    });

    let state = agent.state();
    assert_eq!(state.system_prompt, "");
    assert_eq!(state.model.id, "unknown");
    assert_eq!(state.thinking_level, ThinkingLevel::Off);
    assert!(state.tools.is_empty());
    assert!(state.messages.is_empty());
    assert!(!state.is_streaming);
    assert!(state.streaming_message.is_none());
    assert!(state.pending_tool_calls.is_empty());
    assert!(state.error_message.is_none());
    drop(state);
}

#[tokio::test]
async fn creates_an_agent_instance_with_custom_initial_state() {
    let custom_model = get_builtin_model("openai", "gpt-4o-mini").expect("a catalog model");
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        initial_state: Some(AgentInitialState {
            system_prompt: "You are a helpful assistant.".to_owned(),
            model: Some(custom_model.clone()),
            thinking_level: Some(ThinkingLevel::Low),
            ..AgentInitialState::default()
        }),
        ..AgentOptions::default()
    });

    assert_eq!(agent.state().system_prompt, "You are a helpful assistant.");
    assert_eq!(agent.state().model, custom_model);
    assert_eq!(agent.state().thinking_level, ThinkingLevel::Low);
}

#[tokio::test]
async fn subscribes_to_events() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..AgentOptions::default()
    });

    let event_count = Arc::new(AtomicUsize::new(0));
    let count_for_listener = Arc::clone(&event_count);
    let unsubscribe = agent.subscribe(sync_listener(move |_event, _token| {
        count_for_listener.fetch_add(1, Ordering::Relaxed);
    }));

    // No initial event on subscribe.
    assert_eq!(event_count.load(Ordering::Relaxed), 0);

    // State mutators don't emit events.
    agent.set_system_prompt("Test prompt");
    assert_eq!(event_count.load(Ordering::Relaxed), 0);
    assert_eq!(agent.state().system_prompt, "Test prompt");

    // Unsubscribe should work.
    unsubscribe();
    agent.set_system_prompt("Another prompt");
    assert_eq!(event_count.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn emits_full_lifecycle_events_for_thrown_run_failures() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(Arc::new(|_model, _context, _options| {
            panic!("provider exploded")
        })),
        ..AgentOptions::default()
    });
    let events: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let events_for_listener = Arc::clone(&events);
    let _listener = agent.subscribe(sync_listener(move |event, _token| {
        events_for_listener
            .lock()
            .expect("events lock")
            .push(event_type_name(event));
    }));

    agent.prompt("hello").await.expect("the run settles");

    assert_eq!(
        *events.lock().expect("events lock"),
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
    let state = agent.state();
    let last = state.messages.last().expect("a final message");
    let AgentMessage::Standard(Message::Assistant(assistant)) = last else {
        panic!("the last message is an assistant message");
    };
    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert_eq!(
        assistant.error_message.as_deref(),
        Some("provider exploded")
    );
    assert_eq!(state.error_message.as_deref(), Some("provider exploded"));
}

#[tokio::test]
async fn awaits_async_subscribers_before_prompt_resolves() {
    tokio::time::pause();
    let (barrier_tx, barrier_rx) = tokio::sync::oneshot::channel::<()>();
    let barrier_rx = Arc::new(Mutex::new(Some(barrier_rx)));
    let listener_finished = Arc::new(AtomicBool::new(false));
    let finished_for_listener = Arc::clone(&listener_finished);
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });

    let _listener = agent.subscribe(Arc::new(move |event, _token| {
        let barrier_rx = Arc::clone(&barrier_rx);
        let finished = Arc::clone(&finished_for_listener);
        Box::pin(async move {
            if matches!(event, AgentEvent::AgentEnd { .. }) {
                let receiver = barrier_rx.lock().expect("barrier lock").take();
                if let Some(receiver) = receiver {
                    let _ = receiver.await;
                }
                finished.store(true, Ordering::Relaxed);
            }
        })
    }));

    let prompt_resolved = Arc::new(AtomicBool::new(false));
    let resolved_for_task = Arc::clone(&prompt_resolved);
    let run_agent = agent.clone();
    let prompt = tokio::spawn(async move {
        run_agent.prompt("hello").await.expect("the run settles");
        resolved_for_task.store(true, Ordering::Relaxed);
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!prompt_resolved.load(Ordering::Relaxed));
    assert!(!listener_finished.load(Ordering::Relaxed));
    assert!(agent.state().is_streaming);

    barrier_tx.send(()).expect("the barrier is open once");
    prompt.await.expect("the prompt task");
    assert!(listener_finished.load(Ordering::Relaxed));
    assert!(prompt_resolved.load(Ordering::Relaxed));
    assert!(!agent.state().is_streaming);
}

#[tokio::test]
async fn wait_for_idle_waits_for_async_subscribers() {
    tokio::time::pause();
    let (barrier_tx, barrier_rx) = tokio::sync::oneshot::channel::<()>();
    let barrier_rx = Arc::new(Mutex::new(Some(barrier_rx)));
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });

    let _listener = agent.subscribe(Arc::new(move |event, _token| {
        let barrier_rx = Arc::clone(&barrier_rx);
        Box::pin(async move {
            if let AgentEvent::MessageEnd {
                message: AgentMessage::Standard(Message::Assistant(_)),
            } = event
            {
                let receiver = barrier_rx.lock().expect("barrier lock").take();
                if let Some(receiver) = receiver {
                    let _ = receiver.await;
                }
            }
        })
    }));

    let run_agent = agent.clone();
    let prompt = tokio::spawn(async move {
        run_agent.prompt("hello").await.expect("the run settles");
    });
    let idle_agent = agent.clone();
    let idle_resolved = Arc::new(AtomicBool::new(false));
    let idle_flag = Arc::clone(&idle_resolved);
    let idle = tokio::spawn(async move {
        idle_agent.wait_for_idle().await;
        idle_flag.store(true, Ordering::Relaxed);
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!idle_resolved.load(Ordering::Relaxed));
    assert!(agent.state().is_streaming);

    barrier_tx.send(()).expect("the barrier is open");
    let (prompt_result, idle_result) = tokio::join!(prompt, idle);
    prompt_result.expect("the prompt task");
    idle_result.expect("the idle task");
    assert!(idle_resolved.load(Ordering::Relaxed));
    assert!(!agent.state().is_streaming);
}

#[tokio::test]
async fn passes_the_active_abort_signal_to_subscribers() {
    tokio::time::pause();
    let received: Arc<Mutex<Option<CancellationToken>>> = Arc::new(Mutex::new(None));
    let received_for_listener = Arc::clone(&received);
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(abort_aware_stream_fn()),
        ..AgentOptions::default()
    });

    let _listener = agent.subscribe(sync_listener(move |event, token| {
        if matches!(event, AgentEvent::AgentStart) {
            *received_for_listener.lock().expect("token lock") = Some(token.clone());
        }
    }));

    let run_agent = agent.clone();
    let prompt = tokio::spawn(async move { run_agent.prompt("hello").await });
    tokio::time::sleep(Duration::from_millis(10)).await;

    let token = received
        .lock()
        .expect("token lock")
        .clone()
        .expect("the agent_start listener saw the run's token");
    assert!(!token.is_cancelled());

    agent.abort();
    prompt
        .await
        .expect("the prompt task")
        .expect("the aborted run settles");
    assert!(token.is_cancelled());
}

#[tokio::test]
async fn records_tool_updates_and_stays_stable_after_settlement() {
    tokio::time::pause();
    let tool = suite_tool(
        "delayed_tool",
        json!({ "type": "object", "properties": {} }),
        None,
        Arc::new(
            move |_tool_call_id, _args: &serde_json::Value, _signal, on_update| {
                Box::pin(async move {
                    if let Some(on_update) = on_update {
                        on_update(&text_tool_result("running", json!({ "status": "running" })));
                    }
                    Ok(terminating_tool_result("ok", json!({ "status": "done" })))
                })
            },
        ),
    );
    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentInitialState {
            tools: vec![tool],
            ..AgentInitialState::default()
        }),
        stream_fn: Some(Arc::new(|_model, _context, _options| {
            let stream = mock_stream();
            push_done(
                &stream,
                assistant_tool_use(vec![create_tool_call("call-1", "delayed_tool", json!({}))]),
            );
            stream
        })),
        ..AgentOptions::default()
    });
    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_for_listener = Arc::clone(&events);
    let _listener = agent.subscribe(sync_listener(move |event, _token| {
        events_for_listener
            .lock()
            .expect("events lock")
            .push(event.clone());
    }));

    agent.prompt("run tool").await.expect("the run settles");
    let event_count_after_prompt = events.lock().expect("events lock").len();

    tokio::time::sleep(Duration::from_millis(10)).await;
    let recorded = events.lock().expect("events lock").clone();
    assert_eq!(recorded.len(), event_count_after_prompt);
    assert_eq!(
        update_details(&recorded),
        vec![json!({ "status": "running" })]
    );
}

#[tokio::test]
async fn a_settled_tool_cannot_reach_the_runtime_while_another_tool_is_still_running() {
    tokio::time::pause();
    let slow_started = Arc::new(Notify::new());
    let settled_ended = Arc::new(Notify::new());
    let release_slow = Arc::new(Notify::new());
    let started_for_slow = Arc::clone(&slow_started);
    let release_for_slow = Arc::clone(&release_slow);
    let settled_tool = suite_tool(
        "settled_tool",
        json!({ "type": "object", "properties": {} }),
        None,
        Arc::new(
            |_tool_call_id, _args: &serde_json::Value, _signal, _on_update| {
                Box::pin(
                    async move { Ok(terminating_tool_result("done", json!({ "status": "done" }))) },
                )
            },
        ),
    );
    let slow_tool = suite_tool(
        "slow_tool",
        json!({ "type": "object", "properties": {} }),
        None,
        Arc::new(
            move |_tool_call_id, _args: &serde_json::Value, _signal, _on_update| {
                let started = Arc::clone(&started_for_slow);
                let release = Arc::clone(&release_for_slow);
                Box::pin(async move {
                    started.notify_one();
                    release.notified().await;
                    Ok(terminating_tool_result("done", json!({ "status": "done" })))
                })
            },
        ),
    );
    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentInitialState {
            tools: vec![settled_tool, slow_tool],
            ..AgentInitialState::default()
        }),
        stream_fn: Some(Arc::new(|_model, _context, _options| {
            let stream = mock_stream();
            push_done(
                &stream,
                assistant_tool_use(vec![
                    create_tool_call("call-1", "settled_tool", json!({})),
                    create_tool_call("call-2", "slow_tool", json!({})),
                ]),
            );
            stream
        })),
        ..AgentOptions::default()
    });
    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_for_listener = Arc::clone(&events);
    let ended_for_listener = Arc::clone(&settled_ended);
    let _listener = agent.subscribe(sync_listener(move |event, _token| {
        events_for_listener
            .lock()
            .expect("events lock")
            .push(event.clone());
        if let AgentEvent::ToolExecutionEnd { tool_call_id, .. } = event
            && tool_call_id == "call-1"
        {
            ended_for_listener.notify_one();
        }
    }));

    let run_agent = agent.clone();
    let prompt = tokio::spawn(async move {
        run_agent
            .prompt("run tools")
            .await
            .expect("the run settles");
    });
    tokio::join!(slow_started.notified(), settled_ended.notified());
    let event_count_before = events.lock().expect("events lock").len();

    tokio::time::sleep(Duration::from_millis(10)).await;
    let recorded_before_release = events.lock().expect("events lock").clone();
    assert_eq!(recorded_before_release.len(), event_count_before);

    release_slow.notify_one();
    prompt.await.expect("the prompt task");
    let recorded = events.lock().expect("events lock").clone();
    assert!(update_details(&recorded).is_empty());
}

#[tokio::test]
async fn updates_state_with_mutators() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..AgentOptions::default()
    });

    // Test setSystemPrompt
    agent.set_system_prompt("Custom prompt");
    assert_eq!(agent.state().system_prompt, "Custom prompt");

    // Test setModel
    let new_model = get_builtin_model("google", "gemini-2.5-flash").expect("a catalog model");
    agent.set_model(new_model.clone());
    assert_eq!(agent.state().model, new_model);

    // Test setThinkingLevel
    agent.set_thinking_level(ThinkingLevel::High);
    assert_eq!(agent.state().thinking_level, ThinkingLevel::High);

    // Test setTools
    let tools = vec![suite_tool(
        "test",
        json!({ "type": "object", "properties": {} }),
        None,
        Arc::new(
            |_tool_call_id, _args: &serde_json::Value, _signal, _on_update| {
                Box::pin(async { Ok(empty_tool_result(json!({}))) })
            },
        ),
    )];
    agent.set_tools(&tools);
    assert_eq!(agent.state().tools.len(), 1);
    assert_eq!(agent.state().tools[0].name(), "test");

    // Test replaceMessages: the stored collection is a copy, the caller's
    // stays intact.
    let messages = vec![user_message_blocks("Hello")];
    agent.set_messages(&messages);
    assert_eq!(agent.state().messages, messages);

    // Test appendMessage
    let new_message = AgentMessage::Standard(Message::Assistant(assistant_text("Hi")));
    agent.state().messages.push(new_message.clone());
    assert_eq!(agent.state().messages.len(), 2);
    assert_eq!(agent.state().messages[1], new_message);

    // Test clearMessages
    agent.set_messages(&[]);
    assert!(agent.state().messages.is_empty());
}

#[tokio::test]
async fn supports_steering_message_queue() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..AgentOptions::default()
    });

    let message = create_user_message("Steering message");
    agent.steer(message.clone());

    // The message is queued but not yet in state.messages
    assert!(!agent.state().messages.contains(&message));
}

#[tokio::test]
async fn supports_follow_up_message_queue() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..AgentOptions::default()
    });

    let message = create_user_message("Follow-up message");
    agent.follow_up(message.clone());

    // The message is queued but not yet in state.messages
    assert!(!agent.state().messages.contains(&message));
}

#[tokio::test]
async fn handles_abort_controller() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(unused_stream_fn()),
        ..AgentOptions::default()
    });

    // Should not panic even if nothing is running
    agent.abort();
}

#[tokio::test]
async fn rejects_reset_while_processing_without_corrupting_the_transcript() {
    tokio::time::pause();
    let stream_started = Arc::new(Notify::new());
    let release_response = Arc::new(Notify::new());
    let started_for_stream = Arc::clone(&stream_started);
    let release_for_stream = Arc::clone(&release_response);
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(Arc::new(move |_model, _context, _options| {
            let stream = mock_stream();
            let push = stream.clone();
            let started = Arc::clone(&started_for_stream);
            let release = Arc::clone(&release_for_stream);
            tokio::spawn(async move {
                push.push(AssistantMessageEvent::Start {
                    partial: assistant_text(""),
                });
                started.notify_one();
                release.notified().await;
                push.push(done_event(assistant_text("Done")));
            });
            stream
        })),
        ..AgentOptions::default()
    });

    let run_agent = agent.clone();
    let prompt = tokio::spawn(async move {
        run_agent.prompt("Hello").await.expect("the run settles");
    });
    stream_started.notified().await;

    assert!(agent.state().is_streaming);
    assert_eq!(transcript_roles(&agent), vec!["user"]);
    let reset = agent.reset();
    assert!(matches!(reset, Err(AgentError::ResetWhileStreaming)));
    assert_eq!(
        reset.expect_err("reset rejects mid-run").to_string(),
        "Agent is already processing. Wait for completion before resetting."
    );
    assert!(agent.state().is_streaming);
    assert_eq!(transcript_roles(&agent), vec!["user"]);

    release_response.notify_one();
    prompt.await.expect("the prompt task");
    assert!(!agent.state().is_streaming);
    assert_eq!(transcript_roles(&agent), vec!["user", "assistant"]);
}

#[tokio::test]
async fn throws_when_prompt_called_while_streaming() {
    tokio::time::pause();
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(abort_aware_stream_fn()),
        ..AgentOptions::default()
    });

    // Start first prompt (don't await, it blocks until the abort)
    let run_agent = agent.clone();
    let first_prompt = tokio::spawn(async move { run_agent.prompt("First message").await });

    // Wait a tick for isStreaming to be set
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(agent.state().is_streaming);

    // Second prompt should reject
    let second = agent.prompt("Second message").await;
    assert!(matches!(second, Err(AgentError::PromptWhileStreaming)));
    assert_eq!(
        second.expect_err("the second prompt rejects").to_string(),
        "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
    );

    // Cleanup - abort to stop the stream
    agent.abort();
    first_prompt
        .await
        .expect("the first prompt task")
        .expect("the aborted run resolves");
}

#[tokio::test]
async fn throws_when_continue_called_while_streaming() {
    tokio::time::pause();
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(abort_aware_stream_fn()),
        ..AgentOptions::default()
    });

    // Start first prompt
    let run_agent = agent.clone();
    let first_prompt = tokio::spawn(async move { run_agent.prompt("First message").await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(agent.state().is_streaming);

    // continue() should reject
    let continued = agent.continue_run().await;
    assert!(matches!(continued, Err(AgentError::ContinueWhileStreaming)));
    assert_eq!(
        continued.expect_err("continue rejects mid-run").to_string(),
        "Agent is already processing. Wait for completion before continuing."
    );

    // Cleanup
    agent.abort();
    first_prompt
        .await
        .expect("the first prompt task")
        .expect("the aborted run resolves");
}

#[tokio::test]
async fn continue_processes_queued_follow_up_messages_after_an_assistant_turn() {
    tokio::time::pause();
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });

    agent.set_messages(&[
        user_message_blocks("Initial"),
        AgentMessage::Standard(Message::Assistant(assistant_text("Initial response"))),
    ]);

    agent.follow_up(user_message_blocks("Queued follow-up"));

    agent.continue_run().await.expect("the continue settles");

    let has_queued_follow_up = agent.state().messages.iter().any(|message| {
        matches!(
            message,
            AgentMessage::Standard(Message::User(user))
                if matches!(&user.content,
                    UserContent::Text(text) if text == "Queued follow-up"
                ) || matches!(&user.content, UserContent::Blocks(blocks)
                    if blocks.iter().any(|block| matches!(
                        block,
                        UserBlock::Text(text) if text.text == "Queued follow-up"
                    )))
        )
    });
    assert!(has_queued_follow_up);
    let state = agent.state();
    let last = state.messages.last().expect("a final message");
    assert_eq!(agent_message_role(last), "assistant");
    drop(state);
}

#[tokio::test]
async fn continue_keeps_one_at_a_time_steering_semantics_from_assistant_tail() {
    tokio::time::pause();
    let response_count = Arc::new(AtomicUsize::new(0));
    let count_for_stream = Arc::clone(&response_count);
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(Arc::new(move |_model, _context, _options| {
            let index = count_for_stream.fetch_add(1, Ordering::Relaxed);
            let stream = mock_stream();
            push_done(&stream, assistant_text(&format!("Processed {}", index + 1)));
            stream
        })),
        ..AgentOptions::default()
    });

    agent.set_messages(&[
        user_message_blocks("Initial"),
        AgentMessage::Standard(Message::Assistant(assistant_text("Initial response"))),
    ]);

    agent.steer(user_message_blocks("Steering 1"));
    agent.steer(user_message_blocks("Steering 2"));

    agent.continue_run().await.expect("the continue settles");

    let recent_roles: Vec<String> = {
        let state = agent.state();
        state
            .messages
            .iter()
            .map(|message| agent_message_role(message).to_owned())
            .collect()
    };
    assert_eq!(
        recent_roles[recent_roles.len() - 4..],
        ["user", "assistant", "user", "assistant"]
    );
    assert_eq!(response_count.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn keeps_legacy_prepare_next_turn_signal_callback_behavior() {
    tokio::time::pause();
    let saw_signal = Arc::new(AtomicBool::new(false));
    let saw_for_hook = Arc::clone(&saw_signal);
    let request_count = Arc::new(AtomicUsize::new(0));
    let count_for_stream = Arc::clone(&request_count);
    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentInitialState {
            tools: vec![noop_tool("ok")],
            ..AgentInitialState::default()
        }),
        prepare_next_turn: Some(Arc::new(move |signal: Option<CancellationToken>| {
            let saw = Arc::clone(&saw_for_hook);
            Box::pin(async move {
                saw.store(signal.is_some(), Ordering::Relaxed);
                None::<AgentLoopTurnUpdate>
            })
        })),
        stream_fn: Some(tool_then_final_stream_fn(&count_for_stream, "done")),
        ..AgentOptions::default()
    });

    agent.prompt("start").await.expect("the run settles");

    assert_eq!(request_count.load(Ordering::Relaxed), 2);
    assert!(saw_signal.load(Ordering::Relaxed));
}

#[tokio::test]
async fn forwards_should_stop_after_turn_through_agent_options() {
    tokio::time::pause();
    let saw_signal = Arc::new(AtomicBool::new(false));
    let saw_for_hook = Arc::clone(&saw_signal);
    let callback_roles: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let roles_for_hook = Arc::clone(&callback_roles);
    let request_count = Arc::new(AtomicUsize::new(0));
    let count_for_stream = Arc::clone(&request_count);
    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentInitialState {
            tools: vec![noop_tool("tool complete")],
            ..AgentInitialState::default()
        }),
        should_stop_after_turn: Some(Arc::new(
            move |context: ShouldStopAfterTurnContext, token: Option<CancellationToken>| {
                saw_for_hook.store(token.is_some(), Ordering::Relaxed);
                *roles_for_hook.lock().expect("roles lock") = context
                    .context
                    .messages
                    .iter()
                    .map(agent_message_role)
                    .map(str::to_owned)
                    .collect();
                Box::pin(async move { true })
            },
        )),
        stream_fn: Some(tool_then_final_stream_fn(
            &count_for_stream,
            "should not run",
        )),
        ..AgentOptions::default()
    });

    agent.prompt("start").await.expect("the run settles");

    assert_eq!(request_count.load(Ordering::Relaxed), 1);
    assert!(saw_signal.load(Ordering::Relaxed));
    assert_eq!(
        *callback_roles.lock().expect("roles lock"),
        vec!["user", "assistant", "toolResult"]
    );
}

#[tokio::test]
async fn forwards_session_id_to_stream_function_options() {
    tokio::time::pause();
    let received: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let received_for_stream = Arc::clone(&received);
    let agent = Agent::new(AgentOptions {
        session_id: Some("session-abc".to_owned()),
        stream_fn: Some(Arc::new(move |_model, _context, options| {
            *received_for_stream.lock().expect("session lock") =
                options.and_then(|options| options.session_id.clone());
            let stream = mock_stream();
            push_done(&stream, assistant_text("ok"));
            stream
        })),
        ..AgentOptions::default()
    });

    agent.prompt("hello").await.expect("the run settles");
    assert_eq!(
        received.lock().expect("session lock").as_deref(),
        Some("session-abc")
    );

    // Test setter
    agent.set_session_id(Some("session-def".to_owned()));
    assert_eq!(agent.session_id().as_deref(), Some("session-def"));

    agent.prompt("hello again").await.expect("the run settles");
    assert_eq!(
        received.lock().expect("session lock").as_deref(),
        Some("session-def")
    );
}
