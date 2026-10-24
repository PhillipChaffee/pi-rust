//! `test/agent-session-concurrent.test.ts` 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    clippy::type_complexity,
    reason = "the tests pin outcomes and name their collector shapes inline"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_agent_core::types::AgentTool;
use pi_agent_core::types::StreamFn;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use pi_ai::types::StopReason;
use pi_ai::types::UserBlock;
use pi_ai::types::UserContent;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::reasons::InputSource;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::runner::DeliverAs;
use pi_coding_agent::extensions::runner::SendUserMessageOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;

type Session = Arc<AgentSession<InMemorySettingsStorage>>;

fn start(message: AssistantMessage) -> AssistantMessageEvent {
    AssistantMessageEvent::Start {
        partial: AssistantMessage {
            content: Vec::new(),
            ..message
        },
    }
}

const fn done(message: AssistantMessage) -> AssistantMessageEvent {
    AssistantMessageEvent::Done {
        reason: message.stop_reason,
        message,
    }
}

const fn error(message: AssistantMessage) -> AssistantMessageEvent {
    AssistantMessageEvent::Error {
        reason: message.stop_reason,
        error: message,
    }
}

fn user_texts(context: &pi_ai::types::Context) -> Vec<String> {
    context
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(match &user.content {
                UserContent::Text(text) => text.clone(),
                UserContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        UserBlock::Text(text) => Some(text.text.clone()),
                        UserBlock::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            }),
            _ => None,
        })
        .collect()
}

fn abortable_stream() -> StreamFn {
    Arc::new(|_model, _context, options| {
        let stream: AssistantMessageEventStream = assistant_message_event_stream();
        stream.push(start(assistant_message("", 0)));
        let token = options.and_then(|options| options.transport_options.signal.clone());
        let sink = stream.clone();
        tokio::spawn(async move {
            if let Some(token) = token {
                token.cancelled().await;
            }
            sink.push(error(AssistantMessage {
                stop_reason: StopReason::Aborted,
                error_message: Some("Aborted".to_owned()),
                ..assistant_message("Aborted", 0)
            }));
        });
        stream
    })
}

fn immediate_stream() -> StreamFn {
    Arc::new(|_model, _context, _options| {
        let stream = assistant_message_event_stream();
        stream.push(start(assistant_message("", 0)));
        stream.push(done(assistant_message("Done", 0)));
        stream
    })
}

fn in_memory_manager(root: &str) -> SessionManager {
    SessionManager::in_memory(Some(root), None, None).expect("session")
}

async fn session_with(
    root: &str,
    stream: StreamFn,
    extensions: Vec<InlineExtension>,
    bind: bool,
) -> Session {
    let mut setup = SessionSetup::new(root, in_memory_manager(root));
    setup.stream = Some(stream);
    setup.extensions = extensions;
    setup.bind_extensions = bind;
    let session = build(setup).await;
    authorize(&session).await;
    session
}

async fn wait_for_streaming(session: &Session) {
    for _ in 0..100 {
        if session.is_streaming() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(session.is_streaming(), "the run never started");
}

/// Start the first prompt and wait until its run is streaming.
async fn start_abortable_prompt(
    root: &str,
) -> (Session, tokio::task::JoinHandle<Result<(), String>>) {
    let session = session_with(root, abortable_stream(), Vec::new(), false).await;
    let first = {
        let session = Arc::clone(&session);
        tokio::spawn(async move { session.prompt("First message".to_owned(), None).await })
    };
    wait_for_streaming(&session).await;
    (session, first)
}

#[tokio::test]
async fn throws_when_prompt_is_called_while_streaming() {
    let (session, first) = start_abortable_prompt("/tmp/pi-session-concurrent-guard").await;
    assert!(session.is_streaming());

    let error = session
        .prompt("Second message".to_owned(), None)
        .await
        .expect_err("second prompt rejects");
    assert_eq!(
        error,
        "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message."
    );

    session.abort().await;
    let _ = first.await.expect("join");
}

#[tokio::test]
async fn allows_steer_while_streaming() {
    let (session, first) = start_abortable_prompt("/tmp/pi-session-concurrent-steer").await;

    session
        .steer("Steering message".to_owned(), None, None)
        .await
        .expect("steer");
    assert_eq!(session.pending_message_count(), 1);

    session.abort().await;
    let _ = first.await.expect("join");
}

#[tokio::test]
async fn allows_follow_up_while_streaming() {
    let (session, first) = start_abortable_prompt("/tmp/pi-session-concurrent-follow").await;

    session
        .follow_up("Follow-up message".to_owned(), None, None)
        .await
        .expect("follow-up");
    assert_eq!(session.pending_message_count(), 1);

    session.abort().await;
    let _ = first.await.expect("join");
}

fn input_source_extension(sink: Arc<Mutex<Option<InputSource>>>) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let sink = Arc::clone(&sink);
            Box::pin(async move {
                api.on(
                    "input",
                    Arc::new(move |event: ExtensionEvent, _ctx| {
                        let sink = Arc::clone(&sink);
                        Box::pin(async move {
                            if let ExtensionEvent::Input { source, .. } = event {
                                *sink.lock().expect("lock") = Some(source);
                            }
                            HandlerResult::None
                        })
                    }),
                )
                .expect("input handler");
                Ok(())
            })
        }),
    }
}

fn steering_aware_stream(saw: Arc<AtomicUsize>) -> StreamFn {
    Arc::new(move |_model, context, options| {
        let stream = assistant_message_event_stream();
        let texts = user_texts(context);
        if texts.iter().any(|text| text == "Steer from extension") {
            saw.store(1, Ordering::SeqCst);
            stream.push(start(assistant_message("", 0)));
            stream.push(done(assistant_message("Steered", 0)));
            return stream;
        }
        stream.push(start(assistant_message("", 0)));
        let sink = stream.clone();
        let token = options.and_then(|options| options.transport_options.signal.clone());
        tokio::spawn(async move {
            if let Some(token) = token {
                token.cancelled().await;
            }
            sink.push(error(AssistantMessage {
                stop_reason: StopReason::Aborted,
                error_message: Some("Aborted".to_owned()),
                ..assistant_message("Aborted", 0)
            }));
        });
        stream
    })
}

#[tokio::test]
async fn queues_extension_origin_steering_messages_while_streaming() {
    let source: Arc<Mutex<Option<InputSource>>> = Arc::new(Mutex::new(None));
    let saw = Arc::new(AtomicUsize::new(0));
    let session = session_with(
        "/tmp/pi-session-concurrent-extension",
        steering_aware_stream(Arc::clone(&saw)),
        vec![input_source_extension(Arc::clone(&source))],
        true,
    )
    .await;
    let queue_events: Arc<Mutex<Vec<(Vec<String>, Vec<String>)>>> =
        Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&queue_events);
    let _unsubscribe = session.subscribe(Arc::new(move |event| {
        if let AgentSessionEvent::QueueUpdate {
            steering,
            follow_up,
        } = event
        {
            sink.lock().expect("lock").push((steering, follow_up));
        }
    }));

    let first = {
        let session = Arc::clone(&session);
        tokio::spawn(async move { session.prompt("First message".to_owned(), None).await })
    };
    wait_for_streaming(&session).await;
    assert!(session.is_streaming());

    let actions = session.extension_runner().bound_actions().expect("actions");
    (actions.send_user_message)(
        UserContent::Text("Steer from extension".to_owned()),
        Some(SendUserMessageOptions {
            deliver_as: Some(DeliverAs::Steer),
            expand_prompt_templates: None,
        }),
    );
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    assert_eq!(session.pending_message_count(), 1);
    assert!(
        session
            .steering_messages()
            .iter()
            .any(|text| text == "Steer from extension")
    );
    assert_eq!(*source.lock().expect("lock"), Some(InputSource::Extension));
    assert!(
        queue_events
            .lock()
            .expect("lock")
            .iter()
            .any(|(steering, _)| steering.iter().any(|text| text == "Steer from extension"))
    );

    session.abort().await;
    let _ = first.await.expect("join");
    assert_eq!(saw.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn allows_prompt_after_the_previous_one_completes() {
    let session = session_with(
        "/tmp/pi-session-concurrent-after",
        immediate_stream(),
        Vec::new(),
        false,
    )
    .await;
    session
        .prompt("First message".to_owned(), None)
        .await
        .expect("first prompt");
    assert!(!session.is_streaming());
    session
        .prompt("Second message".to_owned(), None)
        .await
        .expect("second prompt");
}

fn dummy_tool(executed: Arc<AtomicUsize>) -> AgentTool {
    common::agent_session::counting_tool(
        "dummy",
        executed,
        Arc::new(|args| {
            let q = args
                .get("q")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            format!("result:{q}")
        }),
    )
}

fn tool_call(id: &str, q: &str) -> AssistantBlock {
    common::agent_session::tool_call_block(id, "dummy", &serde_json::json!({ "q": q }))
}

fn has_tool_result(context: &pi_ai::types::Context) -> bool {
    context
        .messages
        .iter()
        .any(|message| matches!(message, Message::ToolResult(_)))
}

fn tool_call_stream(rounds: Arc<AtomicUsize>, with_text: bool) -> StreamFn {
    Arc::new(move |_model, context, _options| {
        if has_tool_result(context) {
            return common::agent_session::terminal_stream(assistant_message("done", 0));
        }
        let _ = rounds.fetch_add(1, Ordering::SeqCst);
        let content = if with_text {
            vec![
                AssistantBlock::Text(pi_ai::types::TextContent {
                    text: "calling tool".to_owned(),
                    text_signature: None,
                }),
                tool_call("toolu_1", "x"),
            ]
        } else {
            vec![tool_call("toolu_1", "x"), tool_call("toolu_2", "y")]
        };
        common::agent_session::terminal_stream(AssistantMessage {
            content,
            stop_reason: StopReason::ToolUse,
            ..assistant_message("", 0)
        })
    })
}

fn capture_extension(
    slot: Arc<Mutex<Option<Weak<AgentSession<InMemorySettingsStorage>>>>>,
    snapshots: Arc<Mutex<Vec<(usize, usize)>>>,
) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let slot = Arc::clone(&slot);
            let snapshots = Arc::clone(&snapshots);
            Box::pin(async move {
                api.on(
                    "tool_call",
                    Arc::new(move |_event: ExtensionEvent, _ctx| {
                        let slot = Arc::clone(&slot);
                        let snapshots = Arc::clone(&snapshots);
                        Box::pin(async move {
                            let session =
                                { slot.lock().expect("lock").as_ref().and_then(Weak::upgrade) };
                            if let Some(session) = session {
                                let stats = session.session_stats();
                                snapshots
                                    .lock()
                                    .expect("lock")
                                    .push((stats.user_messages, stats.assistant_messages));
                            }
                            HandlerResult::None
                        })
                    }),
                )
                .expect("tool_call handler");
                Ok(())
            })
        }),
    }
}

fn persisted_roles(path: &str) -> Vec<String> {
    let contents = std::fs::read_to_string(path).expect("export");
    contents
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|entry| entry.get("type").and_then(serde_json::Value::as_str) == Some("message"))
        .filter_map(|entry| {
            entry
                .get("message")
                .and_then(|message| message.get("role"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

#[tokio::test]
async fn waits_for_queued_agent_events_before_emitting_tool_call() {
    let rounds = Arc::new(AtomicUsize::new(0));
    let executed = Arc::new(AtomicUsize::new(0));
    let slot: Arc<Mutex<Option<Weak<AgentSession<InMemorySettingsStorage>>>>> =
        Arc::new(Mutex::new(None));
    let snapshots: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let root = "/tmp/pi-session-concurrent-tool-call";
    let mut setup = SessionSetup::new(root, in_memory_manager(root));
    setup.stream = Some(tool_call_stream(rounds, false));
    setup.extensions = vec![capture_extension(Arc::clone(&slot), Arc::clone(&snapshots))];
    setup.bind_extensions = true;
    setup.base_tools_override = Some(std::collections::BTreeMap::from([(
        "dummy".to_owned(),
        dummy_tool(Arc::clone(&executed)),
    )]));
    let session = build(setup).await;
    authorize(&session).await;
    *slot.lock().expect("lock") = Some(Arc::downgrade(&session));

    session.prompt("hi".to_owned(), None).await.expect("prompt");
    session.agent().wait_for_idle().await;

    assert_eq!(
        snapshots.lock().expect("lock").clone(),
        vec![(1, 1), (1, 1)]
    );
}

#[tokio::test]
async fn persists_message_end_events_in_order_with_slow_extension_handlers() {
    let rounds = Arc::new(AtomicUsize::new(0));
    let executed = Arc::new(AtomicUsize::new(0));
    let root = "/tmp/pi-session-concurrent-order";
    let mut setup = SessionSetup::new(root, in_memory_manager(root));
    setup.stream = Some(tool_call_stream(rounds, true));
    setup.extensions = vec![common::agent_session::slow_message_end_extension()];
    setup.bind_extensions = true;
    setup.base_tools_override = Some(std::collections::BTreeMap::from([(
        "dummy".to_owned(),
        dummy_tool(Arc::clone(&executed)),
    )]));
    let session = build(setup).await;
    authorize(&session).await;

    session.prompt("hi".to_owned(), None).await.expect("prompt");
    session.agent().wait_for_idle().await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let path = format!("{root}/export.jsonl");
    session.export_to_jsonl(Some(&path)).expect("export");
    assert_eq!(
        persisted_roles(&path),
        vec!["user", "assistant", "toolResult", "assistant"]
    );
}
