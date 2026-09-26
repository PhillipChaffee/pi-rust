//! Boundary tests for the `Agent` class, binding the restated branches the
//! upstream `test/agent.test.ts` suite does not pin: the `continue` entry
//! points, the idle waiter's no-run path, queue clearing and modes, the
//! prompt-input normalization branches, the lazy default-stream resolution
//! failure, mid-run unsubscribe, and the converter contract's panic path.

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
use pi_agent_core::AgentEvent;
use pi_agent_core::AgentListener;
use pi_agent_core::AgentMessage;
use pi_agent_core::AgentOptions;
use pi_agent_core::BoxedFuture;
use pi_agent_core::QueueMode;
use pi_agent_core::ShouldStopAfterTurnContext;
use pi_agent_core::StreamFn;
use pi_agent_core::ThinkingLevel;
use pi_agent_core::set_default_stream_fn;
use pi_ai::auth::resolve::now_ms;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::ImageContent;
use pi_ai::types::Message;
use pi_ai::types::StopReason;
use pi_ai::types::UserContent;
use pi_ai::types::UserMessage;
use tokio_util::sync::CancellationToken;

/// The stream fn that answers every call with one final "ok" message.
fn responding_stream_fn() -> StreamFn {
    single_response_stream_fn(assistant_text("ok"))
}

fn assistant_text(text: &str) -> AssistantMessage {
    create_assistant_message(vec![text_block(text)], StopReason::Stop)
}

fn user_message_text(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: now_ms(),
    }))
}

fn transcript_roles(agent: &Agent) -> Vec<String> {
    let state = agent.state();
    state
        .messages
        .iter()
        .map(|message| agent_message_role(message).to_owned())
        .collect()
}

/// A listener future wrapped for `Agent::subscribe`.
fn async_listener(
    f: impl Fn(AgentEvent, CancellationToken) -> BoxedFuture<'static, ()> + Send + Sync + 'static,
) -> AgentListener {
    Arc::new(f)
}

#[tokio::test]
async fn continue_run_on_an_empty_transcript_reports_no_messages() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    let error = agent
        .continue_run()
        .await
        .expect_err("the transcript is empty");
    assert_eq!(error.to_string(), "No messages to continue from");
}

#[tokio::test]
async fn continue_run_from_an_assistant_tail_with_empty_queues_rejects() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    agent.set_messages(&[
        user_message_text("Initial"),
        AgentMessage::Standard(Message::Assistant(assistant_text("Initial response"))),
    ]);
    let error = agent
        .continue_run()
        .await
        .expect_err("the assistant tail has nothing to run");
    assert_eq!(
        error.to_string(),
        "Cannot continue from message role: assistant"
    );
}

#[tokio::test]
async fn continue_run_from_a_user_tail_runs_the_continuation() {
    tokio::time::pause();
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    agent.set_messages(&[user_message_text("Initial")]);

    agent.continue_run().await.expect("the continue settles");

    assert_eq!(transcript_roles(&agent), vec!["user", "assistant"]);
}

#[tokio::test]
async fn wait_for_idle_resolves_immediately_without_a_run() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    tokio::time::timeout(Duration::from_millis(10), agent.wait_for_idle())
        .await
        .expect("the waiter settles at once");
}

#[tokio::test]
async fn wait_for_idle_settles_when_the_active_run_finishes() {
    tokio::time::pause();
    let (gate_tx, gate_rx) = tokio::sync::oneshot::channel::<()>();
    let gate_rx = Arc::new(Mutex::new(Some(gate_rx)));
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(Arc::new(move |_model, _context, _options| {
            let stream = mock_stream();
            let push = stream.clone();
            let gate = Arc::clone(&gate_rx);
            tokio::spawn(async move {
                push.push(AssistantMessageEvent::Start {
                    partial: assistant_text(""),
                });
                let receiver = gate.lock().expect("gate lock").take();
                if let Some(receiver) = receiver {
                    let _ = receiver.await;
                }
                push.push(done_event(assistant_text("done")));
            });
            stream
        })),
        ..AgentOptions::default()
    });

    let run_agent = agent.clone();
    let prompt = tokio::spawn(async move { run_agent.prompt("hello").await });
    let settled = Arc::new(AtomicBool::new(false));
    let settled_for_waiter = Arc::clone(&settled);
    let idle_agent = agent.clone();
    let idle = tokio::spawn(async move {
        idle_agent.wait_for_idle().await;
        settled_for_waiter.store(true, Ordering::Relaxed);
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!settled.load(Ordering::Relaxed));

    gate_tx.send(()).expect("the gate is open");
    prompt
        .await
        .expect("the prompt task")
        .expect("the run settles");
    idle.await.expect("the idle task");
    assert!(settled.load(Ordering::Relaxed));
    assert!(!agent.state().is_streaming);
}

#[tokio::test]
async fn reset_clears_both_queues() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    agent.steer(user_message_text("steering"));
    agent.follow_up(user_message_text("follow-up"));
    assert!(agent.has_queued_messages());

    agent.reset().expect("the agent is idle");
    assert!(!agent.has_queued_messages());
    assert!(!agent.state().is_streaming);
}

#[tokio::test]
async fn clear_all_queues_empties_both_queues() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    agent.steer(user_message_text("steering"));
    agent.follow_up(user_message_text("follow-up"));
    assert!(agent.has_queued_messages());

    agent.clear_all_queues();
    assert!(!agent.has_queued_messages());
}

#[tokio::test]
async fn queue_modes_default_to_one_at_a_time_and_are_settable() {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    assert_eq!(agent.steering_mode(), QueueMode::OneAtATime);
    assert_eq!(agent.follow_up_mode(), QueueMode::OneAtATime);

    agent.set_steering_mode(QueueMode::All);
    agent.set_follow_up_mode(QueueMode::All);
    assert_eq!(agent.steering_mode(), QueueMode::All);
    assert_eq!(agent.follow_up_mode(), QueueMode::All);
}

#[tokio::test]
async fn prompt_normalizes_a_single_message_and_a_batch() {
    tokio::time::pause();
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });

    agent
        .prompt(AgentMessage::Custom(pi_agent_core::CustomAgentMessage {
            role: "status".to_owned(),
            timestamp: now_ms(),
            data: serde_json::Map::new(),
        }))
        .await
        .expect("the run settles");
    // The default converter drops the custom message from the LLM call, but
    // the transcript keeps it verbatim.
    assert_eq!(transcript_roles(&agent), vec!["status", "assistant"]);

    agent
        .prompt(vec![user_message_text("one"), user_message_text("two")])
        .await
        .expect("the run settles");
    assert_eq!(
        transcript_roles(&agent),
        vec!["status", "assistant", "user", "user", "assistant"]
    );
}

#[tokio::test]
async fn prompt_text_appends_image_blocks_only_when_images_are_present() {
    tokio::time::pause();
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });

    agent.prompt("no images").await.expect("the run settles");
    {
        // The state guard must not outlive the inspection: a let-else
        // scrutinee extends its temporaries to the end of the block, and a
        // held guard would deadlock the next prompt's registration.
        let state = agent.state();
        let AgentMessage::Standard(Message::User(user)) = &state.messages[0] else {
            panic!("a user message");
        };
        let UserContent::Blocks(blocks) = &user.content else {
            panic!("block content");
        };
        assert_eq!(blocks.len(), 1);
        drop(state);
    }

    agent
        .prompt(pi_agent_core::PromptInput::Text {
            text: "with images".to_owned(),
            images: vec![ImageContent {
                data: "aGVsbG8=".to_owned(),
                mime_type: "image/png".to_owned(),
            }],
        })
        .await
        .expect("the run settles");
    let state = agent.state();
    let AgentMessage::Standard(Message::User(user)) = &state.messages[2] else {
        panic!("a user message");
    };
    let UserContent::Blocks(blocks) = &user.content else {
        panic!("block content");
    };
    assert_eq!(blocks.len(), 2);
    drop(state);
}

#[tokio::test]
async fn a_missing_default_stream_fn_reaches_the_failure_lifecycle() {
    tokio::time::pause();
    set_default_stream_fn(None);
    let agent = Agent::new(AgentOptions::default());
    let events: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let events_for_listener = Arc::clone(&events);
    let _listener = agent.subscribe(async_listener(move |event, _token| {
        let events_for_listener = Arc::clone(&events_for_listener);
        Box::pin(async move {
            events_for_listener
                .lock()
                .expect("events lock")
                .push(event_type_name(&event));
        })
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
    let AgentMessage::Standard(Message::Assistant(assistant)) =
        state.messages.last().expect("a final message")
    else {
        panic!("an assistant message");
    };
    assert_eq!(
        assistant.error_message.as_deref(),
        Some(
            "No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn()."
        )
    );
    drop(state);
}

#[tokio::test]
async fn signal_tracks_the_active_run() {
    tokio::time::pause();
    let (gate_tx, gate_rx) = tokio::sync::oneshot::channel::<()>();
    let gate_rx = Arc::new(Mutex::new(Some(gate_rx)));
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(Arc::new(move |_model, _context, _options| {
            let stream = mock_stream();
            let push = stream.clone();
            let gate = Arc::clone(&gate_rx);
            tokio::spawn(async move {
                push.push(AssistantMessageEvent::Start {
                    partial: assistant_text(""),
                });
                let receiver = gate.lock().expect("gate lock").take();
                if let Some(receiver) = receiver {
                    let _ = receiver.await;
                }
                push.push(done_event(assistant_text("done")));
            });
            stream
        })),
        ..AgentOptions::default()
    });

    assert!(agent.signal().is_none());
    let run_agent = agent.clone();
    let prompt = tokio::spawn(async move { run_agent.prompt("hello").await });
    while agent.signal().is_none() {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let token = agent.signal().expect("the run is active");
    assert!(!token.is_cancelled());

    gate_tx.send(()).expect("the gate is open");
    prompt
        .await
        .expect("the prompt task")
        .expect("the run settles");
    assert!(agent.signal().is_none());
}

#[tokio::test]
async fn unsubscribe_mid_run_stops_delivery() {
    tokio::time::pause();
    let (gate_tx, gate_rx) = tokio::sync::oneshot::channel::<()>();
    let gate_rx = Arc::new(Mutex::new(Some(gate_rx)));
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(Arc::new(move |_model, _context, _options| {
            let stream = mock_stream();
            let push = stream.clone();
            let gate = Arc::clone(&gate_rx);
            tokio::spawn(async move {
                push.push(AssistantMessageEvent::Start {
                    partial: assistant_text(""),
                });
                let receiver = gate.lock().expect("gate lock").take();
                if let Some(receiver) = receiver {
                    let _ = receiver.await;
                }
                push.push(done_event(assistant_text("done")));
            });
            stream
        })),
        ..AgentOptions::default()
    });
    let unsubscribed_count = Arc::new(AtomicUsize::new(0));
    let remaining_count = Arc::new(AtomicUsize::new(0));
    let count_for_unsubscribed = Arc::clone(&unsubscribed_count);
    let unsubscribe_first: Box<dyn FnOnce() + Send> =
        Box::new(agent.subscribe(async_listener(move |_event, _token| {
            let count = Arc::clone(&count_for_unsubscribed);
            Box::pin(async move {
                count.fetch_add(1, Ordering::Relaxed);
            })
        })));
    let count_for_remaining = Arc::clone(&remaining_count);
    let _listener = agent.subscribe(async_listener(move |_event, _token| {
        let count = Arc::clone(&count_for_remaining);
        Box::pin(async move {
            count.fetch_add(1, Ordering::Relaxed);
        })
    }));

    let run_agent = agent.clone();
    let prompt = tokio::spawn(async move { run_agent.prompt("hello").await });
    // Unsubscribe the first listener once the run is underway and paused on
    // the gate; the second listener keeps counting to the run's end.
    while unsubscribed_count.load(Ordering::Relaxed) < 3 {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    unsubscribe_first();
    gate_tx.send(()).expect("the gate is open");
    prompt
        .await
        .expect("the prompt task")
        .expect("the run settles");

    // The unsubscribed listener froze at its pause-time count while the
    // remaining one saw the run through to agent_end.
    assert!(unsubscribed_count.load(Ordering::Relaxed) < remaining_count.load(Ordering::Relaxed));
}

#[tokio::test]
async fn a_panicking_converter_reaches_the_failure_lifecycle() {
    tokio::time::pause();
    let agent = Agent::new(AgentOptions {
        convert_to_llm: Some(Arc::new(|_messages: Vec<AgentMessage>| {
            panic!("converter exploded");
        })),
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    let events: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let events_for_listener = Arc::clone(&events);
    let _listener = agent.subscribe(async_listener(move |event, _token| {
        let events_for_listener = Arc::clone(&events_for_listener);
        Box::pin(async move {
            events_for_listener
                .lock()
                .expect("events lock")
                .push(event_type_name(&event));
        })
    }));

    agent.prompt("hello").await.expect("the run settles");

    let recorded = events.lock().expect("events lock").clone();
    assert_eq!(recorded.last().copied(), Some("agent_end"));
    let state = agent.state();
    let AgentMessage::Standard(Message::Assistant(assistant)) =
        state.messages.last().expect("a final message")
    else {
        panic!("an assistant message");
    };
    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert!(
        assistant
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("converter exploded"))
    );
    assert_eq!(
        state.error_message.as_deref(),
        assistant.error_message.as_deref()
    );
}

#[tokio::test]
async fn a_panicking_listener_rejects_the_prompt_and_settles_the_run() {
    tokio::time::pause();
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    let _listener = agent.subscribe(async_listener(move |_event, _token| {
        Box::pin(async { panic!("listener boom") })
    }));

    let outcome = agent.prompt("hello").await;

    let Err(error) = outcome else {
        panic!("the panicking listener rejects the prompt");
    };
    assert_eq!(error.to_string(), "listener boom");
    assert!(!agent.state().is_streaming);
    assert!(agent.signal().is_none());
}

#[tokio::test]
async fn prepare_next_turn_with_context_wins_and_receives_the_token() {
    tokio::time::pause();
    let context_saw_token = Arc::new(AtomicBool::new(false));
    let context_saw_roles = Arc::new(Mutex::new(Vec::<String>::new()));
    let signal_only_called = Arc::new(AtomicBool::new(false));
    let context_token = Arc::clone(&context_saw_token);
    let context_roles = Arc::clone(&context_saw_roles);
    let signal_only = Arc::clone(&signal_only_called);
    let request_count = Arc::new(AtomicUsize::new(0));
    let count_for_stream = Arc::clone(&request_count);
    let agent = Agent::new(AgentOptions {
        initial_state: Some(pi_agent_core::AgentInitialState {
            tools: vec![noop_tool("ok")],
            ..pi_agent_core::AgentInitialState::default()
        }),
        prepare_next_turn: Some(Arc::new(move |_token: Option<CancellationToken>| {
            let signal_only = Arc::clone(&signal_only);
            Box::pin(async move {
                signal_only.store(true, Ordering::Relaxed);
                None::<pi_agent_core::AgentLoopTurnUpdate>
            })
        })),
        prepare_next_turn_with_context: Some(Arc::new(
            move |context: ShouldStopAfterTurnContext, token: Option<CancellationToken>| {
                let saw = Arc::clone(&context_token);
                let roles = Arc::clone(&context_roles);
                saw.store(token.is_some(), Ordering::Relaxed);
                *roles.lock().expect("roles lock") = context
                    .context
                    .messages
                    .iter()
                    .map(|message| agent_message_role(message).to_owned())
                    .collect();
                Box::pin(async move { None::<pi_agent_core::AgentLoopTurnUpdate> })
            },
        )),
        stream_fn: Some(tool_then_final_stream_fn(&count_for_stream, "done")),
        ..AgentOptions::default()
    });

    agent.prompt("start").await.expect("the run settles");

    assert_eq!(request_count.load(Ordering::Relaxed), 2);
    assert!(context_saw_token.load(Ordering::Relaxed));
    assert_eq!(
        *context_saw_roles.lock().expect("roles lock"),
        vec!["user", "assistant", "toolResult"]
    );
    assert!(!signal_only_called.load(Ordering::Relaxed));
}

#[tokio::test]
async fn steering_mode_all_drains_every_queued_message_at_one_poll() {
    tokio::time::pause();
    let agent = Agent::new(AgentOptions {
        steering_mode: Some(QueueMode::All),
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });
    agent.steer(user_message_text("steering one"));
    agent.steer(user_message_text("steering two"));

    agent.prompt("start").await.expect("the run settles");

    let transcript = {
        let state = agent.state();
        state
            .messages
            .iter()
            .filter_map(|message| match message {
                AgentMessage::Standard(Message::User(user)) => match &user.content {
                    UserContent::Text(text) => Some(text.clone()),
                    UserContent::Blocks(_) => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert!(transcript.contains(&"steering one".to_owned()));
    assert!(transcript.contains(&"steering two".to_owned()));
    drop(transcript);
}

#[tokio::test]
async fn an_abort_before_a_run_failure_reports_the_aborted_stop_reason() {
    tokio::time::pause();
    // The context transform runs inside the executor on every LLM call and
    // carries the run's token: it parks until the abort, then panics — the
    // failure lands while the run is aborted, upstream's
    // `handleRunFailure(error, aborted)` with the aborted stop reason.
    let agent = Agent::new(AgentOptions {
        transform_context: Some(Arc::new(
            |_messages: Vec<AgentMessage>, signal: Option<CancellationToken>| {
                Box::pin(async move {
                    if let Some(signal) = signal {
                        signal.cancelled().await;
                    }
                    panic!("aborted boom");
                })
            },
        )),
        stream_fn: Some(responding_stream_fn()),
        ..AgentOptions::default()
    });

    let run_agent = agent.clone();
    let prompt = tokio::spawn(async move { run_agent.prompt("hello").await });
    while agent.signal().is_none() {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    agent.abort();
    prompt
        .await
        .expect("the prompt task")
        .expect("the failure resolves the prompt");

    let state = agent.state();
    let AgentMessage::Standard(Message::Assistant(assistant)) =
        state.messages.last().expect("a final message")
    else {
        panic!("an assistant message");
    };
    assert_eq!(assistant.stop_reason, StopReason::Aborted);
    assert_eq!(assistant.error_message.as_deref(), Some("aborted boom"));
    drop(state);
}

#[tokio::test]
async fn panic_payloads_reach_the_error_message_through_their_string_forms() {
    tokio::time::pause();
    let string_agent = Agent::new(AgentOptions {
        stream_fn: Some(Arc::new(|_model, _context, _options| {
            std::panic::panic_any("string payload".to_owned());
        })),
        ..AgentOptions::default()
    });
    string_agent.prompt("hello").await.expect("the run settles");
    {
        let state = string_agent.state();
        let AgentMessage::Standard(Message::Assistant(assistant)) =
            state.messages.last().expect("a final message")
        else {
            panic!("an assistant message");
        };
        assert_eq!(assistant.error_message.as_deref(), Some("string payload"));
        drop(state);
    }

    let non_string_agent = Agent::new(AgentOptions {
        stream_fn: Some(Arc::new(|_model, _context, _options| {
            std::panic::panic_any(7u32);
        })),
        ..AgentOptions::default()
    });
    non_string_agent
        .prompt("hello")
        .await
        .expect("the run settles");
    {
        let state = non_string_agent.state();
        let AgentMessage::Standard(Message::Assistant(assistant)) =
            state.messages.last().expect("a final message")
        else {
            panic!("an assistant message");
        };
        assert_eq!(assistant.error_message.as_deref(), Some("panicked"));
        drop(state);
    }
}

#[tokio::test]
async fn thinking_levels_map_onto_the_simple_request_reasoning_field() {
    tokio::time::pause();
    for (level, expected) in [
        (ThinkingLevel::Off, None),
        (
            ThinkingLevel::Minimal,
            Some(pi_ai::types::ThinkingLevel::Minimal),
        ),
        (
            ThinkingLevel::Xhigh,
            Some(pi_ai::types::ThinkingLevel::Xhigh),
        ),
        (ThinkingLevel::Max, Some(pi_ai::types::ThinkingLevel::Max)),
    ] {
        let received: Arc<Mutex<Option<pi_ai::types::ThinkingLevel>>> = Arc::new(Mutex::new(None));
        let received_for_stream = Arc::clone(&received);
        let agent = Agent::new(AgentOptions {
            initial_state: Some(pi_agent_core::AgentInitialState {
                thinking_level: Some(level),
                ..pi_agent_core::AgentInitialState::default()
            }),
            stream_fn: Some(Arc::new(move |_model, _context, options| {
                *received_for_stream.lock().expect("reasoning lock") =
                    options.and_then(|options| options.reasoning);
                let stream = mock_stream();
                push_done(&stream, assistant_text("ok"));
                stream
            })),
            ..AgentOptions::default()
        });
        agent.prompt("hello").await.expect("the run settles");
        assert_eq!(*received.lock().expect("reasoning lock"), expected);
    }
}

#[tokio::test]
async fn the_agent_and_options_debug_bodies_carry_their_shapes() {
    let agent = Agent::new(AgentOptions::default());
    assert!(format!("{agent:?}").contains("Agent"));
    assert!(format!("{:?}", AgentOptions::default()).contains("AgentOptions"));
}

#[test]
fn prompt_input_conversions_state_the_overload_union() {
    let message = user_message_text("hello");
    assert!(matches!(
        pi_agent_core::PromptInput::from("hello"),
        pi_agent_core::PromptInput::Text { text, images } if text == "hello" && images.is_empty()
    ));
    assert!(matches!(
        pi_agent_core::PromptInput::from("hello".to_owned()),
        pi_agent_core::PromptInput::Text { .. }
    ));
    assert!(matches!(
        pi_agent_core::PromptInput::from(message.clone()),
        pi_agent_core::PromptInput::One(_)
    ));
    assert!(matches!(
        pi_agent_core::PromptInput::from(vec![message]),
        pi_agent_core::PromptInput::Many(_)
    ));
}
