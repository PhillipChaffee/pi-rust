//! `test/agent-session-retry.test.ts` 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    clippy::type_complexity,
    reason = "the tests pin outcomes and name their collector shapes inline"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_agent_core::types::AgentTool;
use pi_agent_core::types::StreamFn;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::StopReason;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::settings_manager::Settings;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;
use common::agent_session::error_message;

type Session = Arc<AgentSession<InMemorySettingsStorage>>;

fn retry_settings(max_retries: i64, base_delay_ms: i64, max_agent_delay_ms: i64) -> Settings {
    serde_json::json!({
        "retry": {
            "enabled": true,
            "maxRetries": max_retries,
            "baseDelayMs": base_delay_ms,
            "maxAgentDelayMs": max_agent_delay_ms,
        }
    })
    .as_object()
    .expect("settings object")
    .clone()
}

fn stream_counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

fn failing_stream(fail_count: usize, counter: Arc<AtomicUsize>, failure: &'static str) -> StreamFn {
    Arc::new(move |_model, _context, _options| {
        let call = counter.fetch_add(1, Ordering::SeqCst) + 1;
        let stream: AssistantMessageEventStream = assistant_message_event_stream();
        let message = if call <= fail_count {
            error_message(failure, 0)
        } else {
            assistant_message("Success", 0)
        };
        stream.push(AssistantMessageEvent::Start {
            partial: AssistantMessage {
                content: Vec::new(),
                ..message.clone()
            },
        });
        if message.stop_reason == StopReason::Error {
            stream.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: message,
            });
        } else {
            stream.push(AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message,
            });
        }
        stream
    })
}

fn retry_events() -> (
    Arc<Mutex<Vec<String>>>,
    Arc<dyn Fn(AgentSessionEvent) + Send + Sync>,
) {
    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let listener: Arc<dyn Fn(AgentSessionEvent) + Send + Sync> =
        Arc::new(move |event| match event {
            AgentSessionEvent::AutoRetryStart { attempt, .. } => {
                sink.lock().expect("lock").push(format!("start:{attempt}"));
            }
            AgentSessionEvent::AutoRetryEnd { success, .. } => {
                sink.lock()
                    .expect("lock")
                    .push(format!("end:success={success}"));
            }
            _ => {}
        });
    (events, listener)
}

fn in_memory_manager() -> SessionManager {
    SessionManager::in_memory(Some("/tmp/pi-session-retry"), None, None).expect("session")
}

async fn retry_session(
    fail_count: usize,
    settings: Settings,
    extensions: Vec<InlineExtension>,
    bind: bool,
) -> (Session, Arc<AtomicUsize>) {
    let counter = stream_counter();
    let mut setup = SessionSetup::new("/tmp/pi-session-retry", in_memory_manager());
    setup.settings = settings;
    setup.stream = Some(failing_stream(
        fail_count,
        Arc::clone(&counter),
        "overloaded_error",
    ));
    setup.extensions = extensions;
    setup.bind_extensions = bind;
    let session = build(setup).await;
    authorize(&session).await;
    (session, counter)
}

#[tokio::test]
async fn retries_after_a_transient_error_and_succeeds() {
    let (session, counter) =
        retry_session(1, retry_settings(3, 1, 60_000), Vec::new(), false).await;
    let (events, listener) = retry_events();
    let _unsubscribe = session.subscribe(listener);

    session
        .prompt("Test".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(counter.load(Ordering::SeqCst), 2);
    assert_eq!(
        events.lock().expect("lock").clone(),
        vec!["start:1".to_owned(), "end:success=true".to_owned()]
    );
    assert!(!session.is_retrying());
}

#[tokio::test]
async fn exhausts_max_retries_and_emits_failure() {
    let (session, counter) =
        retry_session(99, retry_settings(2, 1, 60_000), Vec::new(), false).await;
    let (events, listener) = retry_events();
    let _unsubscribe = session.subscribe(listener);

    session
        .prompt("Test".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(counter.load(Ordering::SeqCst), 3);
    let events = events.lock().expect("lock").clone();
    assert!(events.contains(&"start:1".to_owned()));
    assert!(events.contains(&"start:2".to_owned()));
    assert!(events.contains(&"end:success=false".to_owned()));
    assert!(!session.is_retrying());
}

#[tokio::test]
async fn caps_the_agent_retry_delay() {
    let (session, _) = retry_session(4, retry_settings(5, 1, 5), Vec::new(), false).await;
    let delays: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&delays);
    let _unsubscribe = session.subscribe(Arc::new(move |event| {
        if let AgentSessionEvent::AutoRetryStart { delay_ms, .. } = event {
            sink.lock().expect("lock").push(delay_ms);
        }
    }));

    session
        .prompt("Test".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(delays.lock().expect("lock").clone(), vec![1, 2, 4, 5]);
}

fn slow_message_end_extension() -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(|api: Arc<ExtensionApi>| {
            Box::pin(async move {
                api.on(
                    "message_end",
                    Arc::new(|event: ExtensionEvent, _ctx| {
                        Box::pin(async move {
                            if let ExtensionEvent::MessageEnd { message } = &event
                                && matches!(
                                    message,
                                    pi_agent_core::types::AgentMessage::Standard(
                                        pi_ai::types::Message::Assistant(_)
                                    )
                                )
                            {
                                tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                            }
                            HandlerResult::None
                        })
                    }),
                )
                .expect("message_end handler");
                Ok(())
            })
        }),
    }
}

#[tokio::test]
async fn prompt_waits_for_retry_completion_when_message_end_handling_is_delayed() {
    let (session, counter) = retry_session(
        1,
        retry_settings(3, 1, 60_000),
        vec![slow_message_end_extension()],
        true,
    )
    .await;

    session
        .prompt("Test".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(counter.load(Ordering::SeqCst), 2);
    assert!(!session.is_retrying());
}

#[tokio::test]
async fn retries_provider_network_error_failures() {
    let counter = stream_counter();
    let mut setup = SessionSetup::new("/tmp/pi-session-retry-network", in_memory_manager());
    setup.settings = retry_settings(3, 1, 60_000);
    setup.stream = Some(failing_stream(
        1,
        Arc::clone(&counter),
        "Provider finish_reason: network_error",
    ));
    let session = build(setup).await;
    authorize(&session).await;
    let (events, listener) = retry_events();
    let _unsubscribe = session.subscribe(listener);

    session
        .prompt("Test".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(counter.load(Ordering::SeqCst), 2);
    assert_eq!(
        events.lock().expect("lock").clone(),
        vec!["start:1".to_owned(), "end:success=true".to_owned()]
    );
}

fn echo_tool(executed: Arc<AtomicUsize>) -> AgentTool {
    AgentTool {
        tool: pi_ai::types::Tool {
            name: "echo".to_owned(),
            description: "Echo text back".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
            }),
            constrained_sampling: None,
        },
        label: "Echo".to_owned(),
        prepare_arguments: None,
        execute: Arc::new(move |_id, _args, _signal, _on_update| {
            let executed = Arc::clone(&executed);
            Box::pin(async move {
                executed.fetch_add(1, Ordering::SeqCst);
                Ok(pi_agent_core::types::AgentToolResult {
                    content: vec![pi_ai::types::ToolResultBlock::Text(
                        pi_ai::types::TextContent {
                            text: "echoed".to_owned(),
                            text_signature: None,
                        },
                    )],
                    details: serde_json::Value::Null,
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
        replay: None,
        execution_mode: None,
    }
}

fn tool_call_message() -> AssistantMessage {
    AssistantMessage {
        content: vec![
            AssistantBlock::Text(pi_ai::types::TextContent {
                text: "Looking that up now.".to_owned(),
                text_signature: None,
            }),
            AssistantBlock::ToolCall(pi_ai::types::ToolCall {
                id: "call_1".to_owned(),
                name: "echo".to_owned(),
                arguments: serde_json::json!({ "text": "hello" })
                    .as_object()
                    .cloned()
                    .expect("object"),
                thought_signature: None,
                namespace: None,
            }),
        ],
        stop_reason: StopReason::ToolUse,
        ..assistant_message("Looking that up now.", 0)
    }
}

#[tokio::test]
async fn prompt_waits_for_the_full_agent_loop_when_the_retry_produces_tool_calls() {
    let counter = stream_counter();
    let executed = Arc::new(AtomicUsize::new(0));
    let stream_counter_ref = Arc::clone(&counter);
    let stream: StreamFn = Arc::new(move |_model, _context, _options| {
        let call = stream_counter_ref.fetch_add(1, Ordering::SeqCst) + 1;
        let stream = assistant_message_event_stream();
        let message = match call {
            1 => error_message("overloaded_error", 0),
            2 => tool_call_message(),
            _ => assistant_message("Final answer.", 0),
        };
        stream.push(AssistantMessageEvent::Start {
            partial: AssistantMessage {
                content: Vec::new(),
                ..message.clone()
            },
        });
        if message.stop_reason == StopReason::Error {
            stream.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: message,
            });
        } else {
            stream.push(AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message,
            });
        }
        stream
    });
    let mut setup = SessionSetup::new("/tmp/pi-session-retry-tools", in_memory_manager());
    setup.settings = retry_settings(3, 1, 60_000);
    setup.stream = Some(stream);
    setup.base_tools_override = Some(std::collections::BTreeMap::from([(
        "echo".to_owned(),
        echo_tool(Arc::clone(&executed)),
    )]));
    let session = build(setup).await;
    authorize(&session).await;

    session
        .prompt("Test".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(counter.load(Ordering::SeqCst), 3);
    assert_eq!(executed.load(Ordering::SeqCst), 1);
    assert!(!session.is_streaming());
    session
        .prompt("Follow-up".to_owned(), None)
        .await
        .expect("follow-up");
    assert_eq!(counter.load(Ordering::SeqCst), 4);
}
