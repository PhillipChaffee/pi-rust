//! Boundary coverage for the ticket #149 surfaces: tools, compaction, tree
//! navigation, bash recording, stats, and export.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;

use pi_agent_core::Agent;
use pi_agent_core::AgentInitialState;
use pi_agent_core::AgentOptions;
use pi_agent_core::types::AgentMessage;
use pi_agent_core::types::StreamFn;
use pi_ai::types::Api;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use pi_ai::types::ProviderId;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::UserContent;
use pi_ai::types::UserMessage;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::core::agent_session::AgentSessionConfig;
use pi_coding_agent::core::agent_session::BashExecutionOptions;
use pi_coding_agent::core::agent_session::NavigateTreeOptions;
use pi_coding_agent::resource_loader::DefaultResourceLoader;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;

fn assistant(text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: Api::from("openai-completions"),
        provider: ProviderId("openai".to_owned()),
        model: "gpt-test".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: pi_ai::types::Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

fn scripted_stream(message: AssistantMessage) -> StreamFn {
    Arc::new(move |_model, _context, _options| {
        let stream = assistant_message_event_stream();
        let sink = stream.clone();
        let message = message.clone();
        tokio::spawn(async move {
            sink.push(AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message,
            });
        });
        stream
    })
}

async fn build_session(
    root: &str,
    manager: SessionManager,
) -> Arc<AgentSession<pi_coding_agent::settings_manager::InMemorySettingsStorage>> {
    let agent_dir = format!("{root}/agent");
    std::fs::create_dir_all(&agent_dir).expect("mkdir");
    let registry = common::model_layer::create_in_memory_model_registry(
        common::model_layer::empty_auth_storage(),
    )
    .await;
    let runtime = registry.runtime().clone();
    let model = common::model_layer::model("openai", "gpt-test");
    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentInitialState {
            system_prompt: String::new(),
            model: Some(model),
            thinking_level: None,
            tools: Vec::new(),
            messages: Vec::new(),
        }),
        ..AgentOptions::default()
    });
    let settings =
        SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default());
    let loader = DefaultResourceLoader::new(
        DefaultResourceLoaderOptions {
            cwd: root.to_owned(),
            agent_dir,
            ..DefaultResourceLoaderOptions::default()
        },
        settings,
    );
    AgentSession::new(AgentSessionConfig {
        agent,
        session_manager: manager,
        resource_loader: loader,
        cwd: root.to_owned(),
        scoped_models: None,
        model_runtime: runtime,
        custom_tools: None,
        initial_active_tool_names: None,
        allowed_tool_names: None,
        excluded_tool_names: None,
        base_tools_override: None,
        extension_runner_ref: None,
        session_start_event: None,
    })
}

#[tokio::test]
async fn tool_registry_exposes_and_filters_the_built_ins() {
    let root = "/tmp/pi-session-surfaces-tools";
    let manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    let session = build_session(root, manager).await;
    assert!(session.get_tool_definition("read").is_some());
    assert!(
        session
            .get_all_tools()
            .iter()
            .any(|tool| tool.name == "bash")
    );
    session.set_active_tools_by_name(&["read".to_owned()]);
    assert_eq!(session.active_tool_names(), ["read".to_owned()]);
    assert!(session.system_prompt().contains("Read file contents"));
    session.set_active_tools_by_name(&["write".to_owned(), "unknown".to_owned()]);
    assert_eq!(session.active_tool_names(), ["write".to_owned()]);
}

#[tokio::test]
async fn bash_execution_records_a_session_entry() {
    let root = "/tmp/pi-session-surfaces-bash";
    let manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    let session = build_session(root, manager).await;
    let result = session
        .execute_bash("printf hi", None, Some(BashExecutionOptions::default()))
        .await
        .expect("bash");
    assert_eq!(result.output, "hi");
    assert!(session.session_stats().total_messages >= 1);
}

#[tokio::test]
async fn tree_navigation_moves_the_leaf_and_restores_user_text() {
    let root = "/tmp/pi-session-surfaces-tree";
    let mut manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    let first = manager
        .append_message(AgentMessage::Standard(Message::User(UserMessage {
            content: UserContent::Text("first".to_owned()),
            timestamp: 0,
        })))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(assistant("ok"))))
        .expect("append");
    let session = build_session(root, manager).await;
    let outcome = session
        .navigate_tree(&first, NavigateTreeOptions::default())
        .await
        .expect("navigate");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.editor_text.as_deref(), Some("first"));
    assert!(session.leaf_id().is_none());
}

#[tokio::test]
async fn manual_compaction_summarizes_and_rebuilds_the_context() {
    let root = "/tmp/pi-session-surfaces-compact";
    let mut manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    for index in 0..60 {
        manager
            .append_message(AgentMessage::Standard(Message::User(UserMessage {
                content: UserContent::Text(format!("message {index} {}", "x".repeat(4000))),
                timestamp: index,
            })))
            .expect("append");
        manager
            .append_message(AgentMessage::Standard(Message::Assistant(assistant(
                &"y".repeat(4000),
            ))))
            .expect("append");
    }
    let session = build_session(root, manager).await;
    session
        .model_runtime()
        .set_runtime_api_key("openai", "test-key".to_owned(), None)
        .await
        .expect("runtime key");
    session
        .agent()
        .set_stream_function(Some(scripted_stream(assistant("summary"))));
    let result = session.compact(None).await.expect("compact");
    assert_eq!(result.summary, "summary");
}

#[tokio::test]
async fn persisted_sessions_export_jsonl() {
    let root = "/tmp/pi-session-surfaces-export";
    let session_dir = format!("{root}/sessions");
    let mut manager = SessionManager::create(root, Some(&session_dir), None).expect("session");
    manager
        .append_message(AgentMessage::Standard(Message::User(UserMessage {
            content: UserContent::Text("hello".to_owned()),
            timestamp: 0,
        })))
        .expect("append");
    let session = build_session(root, manager).await;
    let path = format!("{root}/export.jsonl");
    let written = session.export_to_jsonl(Some(&path)).expect("export");
    assert_eq!(written, path);
    assert!(std::path::Path::new(&path).exists());
}

fn tool_call_stream(path: &str) -> StreamFn {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let path = path.to_owned();
    Arc::new(move |_model, _context, _options| {
        let stream = assistant_message_event_stream();
        let sink = stream.clone();
        let path = path.clone();
        let first = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
        tokio::spawn(async move {
            let message = if first {
                let mut arguments = serde_json::Map::new();
                arguments.insert("path".to_owned(), serde_json::Value::String(path));
                AssistantMessage {
                    content: vec![AssistantBlock::ToolCall(pi_ai::types::ToolCall {
                        id: "call-1".to_owned(),
                        name: "read".to_owned(),
                        arguments,
                        thought_signature: None,
                        namespace: None,
                    })],
                    stop_reason: StopReason::ToolUse,
                    ..assistant("")
                }
            } else {
                assistant("done")
            };
            sink.push(AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message,
            });
        });
        stream
    })
}

#[tokio::test]
async fn prompt_runs_a_tool_call_end_to_end() {
    let root = "/tmp/pi-session-surfaces-prompt";
    let manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    let session = build_session(root, manager).await;
    let file = format!("{root}/note.txt");
    std::fs::write(&file, "hello").expect("write");
    session
        .agent()
        .set_stream_function(Some(tool_call_stream(&file)));
    let events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let _unsubscribe = session.subscribe(Arc::new(move |event| {
        let label = match event {
            pi_coding_agent::core::agent_session::AgentSessionEvent::AgentStart => "start",
            pi_coding_agent::core::agent_session::AgentSessionEvent::AgentSettled => "settled",
            pi_coding_agent::core::agent_session::AgentSessionEvent::ToolExecutionStart {
                ..
            } => "tool_start",
            pi_coding_agent::core::agent_session::AgentSessionEvent::ToolExecutionEnd {
                ..
            } => "tool_end",
            _ => "other",
        };
        sink.lock().expect("lock").push(label);
    }));
    session
        .prompt("read it".to_owned(), None)
        .await
        .expect("prompt");
    let labels = events.lock().expect("lock").clone();
    assert!(labels.contains(&"tool_start"));
    assert!(labels.contains(&"tool_end"));
    assert!(labels.contains(&"settled"));
    assert!(session.last_assistant_text().is_some());
}

#[tokio::test]
async fn custom_messages_append_and_trigger_turns() {
    let root = "/tmp/pi-session-surfaces-messages";
    let manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    let session = build_session(root, manager).await;
    let payload = pi_coding_agent::extensions::reasons::CustomMessagePayload {
        custom_type: "note".to_owned(),
        content: vec![pi_ai::types::UserBlock::Text(TextContent {
            text: "body".to_owned(),
            text_signature: None,
        })],
        display: true,
        details: None,
    };
    session
        .send_custom_message(payload, None)
        .await
        .expect("custom");
    assert!(
        session
            .agent()
            .state()
            .messages
            .iter()
            .any(|message| matches!(
                message,
                AgentMessage::Custom(custom) if custom.role == "custom"
            ))
    );
}
