//! Model, thinking, retry, stats, and tree-summary coverage for #149.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;

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
use pi_coding_agent::core::agent_session::ModelMutationOptions;
use pi_coding_agent::core::agent_session::NavigateTreeOptions;
use pi_coding_agent::resource_loader::DefaultResourceLoader;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;

fn assistant(text: &str, stop_reason: StopReason, error: Option<&str>) -> AssistantMessage {
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
        stop_reason,
        deferred: None,
        error_message: error.map(str::to_owned),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

fn stream_fn(messages: Vec<AssistantMessage>) -> StreamFn {
    let index = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    Arc::new(move |_model, _context, _options| {
        let stream = assistant_message_event_stream();
        let sink = stream.clone();
        let messages = messages.clone();
        let index = Arc::clone(&index);
        tokio::spawn(async move {
            let position = index.fetch_add(1, Ordering::SeqCst).min(messages.len() - 1);
            let message = messages[position].clone();
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
    settings: Settings,
) -> Arc<AgentSession<pi_coding_agent::settings_manager::InMemorySettingsStorage>> {
    let agent_dir = format!("{root}/agent");
    std::fs::create_dir_all(&agent_dir).expect("mkdir");
    let registry = common::model_layer::create_in_memory_model_registry(
        common::model_layer::empty_auth_storage(),
    )
    .await;
    let runtime = registry.runtime().clone();
    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentInitialState {
            system_prompt: String::new(),
            model: Some(common::model_layer::model("openai", "gpt-test")),
            thinking_level: None,
            tools: Vec::new(),
            messages: Vec::new(),
        }),
        ..AgentOptions::default()
    });
    let settings_manager =
        SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default());
    let loader = DefaultResourceLoader::new(
        DefaultResourceLoaderOptions {
            cwd: root.to_owned(),
            agent_dir,
            ..DefaultResourceLoaderOptions::default()
        },
        settings_manager,
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

fn default_settings() -> Settings {
    Settings::new()
}

#[tokio::test]
async fn model_and_thinking_mutations_apply() {
    let root = "/tmp/pi-session-model-mut";
    let manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    let session = build_session(root, manager, default_settings()).await;
    session
        .model_runtime()
        .set_runtime_api_key("openai", "k".to_owned(), None)
        .await
        .expect("key");
    session
        .set_model(
            common::model_layer::model("openai", "gpt-test"),
            ModelMutationOptions { persist: true },
        )
        .await
        .expect("set model");
    session.cycle_thinking_level(ModelMutationOptions::default());
    session.set_thinking_level(
        pi_agent_core::types::ThinkingLevel::Low,
        ModelMutationOptions { persist: true },
    );
    session.set_steering_mode(pi_agent_core::types::QueueMode::All);
    session.set_follow_up_mode(pi_agent_core::types::QueueMode::All);
    assert_eq!(
        session.steering_mode(),
        pi_agent_core::types::QueueMode::All
    );
    session.set_scoped_models(session.scoped_models());
    session.set_session_name("named");
    assert_eq!(session.session_name().as_deref(), Some("named"));
}

#[tokio::test]
async fn retry_continues_after_a_retryable_error() {
    let root = "/tmp/pi-session-model-retry";
    let manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    let settings: Settings =
        serde_json::from_str(r#"{"retry":{"enabled":true,"maxRetries":1,"baseDelayMs":0}}"#)
            .expect("settings");
    let session = build_session(root, manager, settings).await;
    session.agent().set_stream_function(Some(stream_fn(vec![
        assistant("", StopReason::Error, Some("Overloaded")),
        assistant("recovered", StopReason::Stop, None),
    ])));
    let events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let _unsubscribe = session.subscribe(Arc::new(move |event| {
        let label = match event {
            pi_coding_agent::core::agent_session::AgentSessionEvent::AutoRetryStart { .. } => {
                "start"
            }
            pi_coding_agent::core::agent_session::AgentSessionEvent::AutoRetryEnd { .. } => "end",
            _ => "other",
        };
        sink.lock().expect("lock").push(label);
    }));
    session.prompt("hi".to_owned(), None).await.expect("prompt");
    let labels = events.lock().expect("lock").clone();
    assert!(labels.contains(&"start"));
    assert!(labels.contains(&"end"));
}

#[tokio::test]
async fn stats_context_and_last_text_read_the_transcript() {
    let root = "/tmp/pi-session-model-stats";
    let manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    let session = build_session(root, manager, default_settings()).await;
    session
        .agent()
        .set_stream_function(Some(stream_fn(vec![assistant(
            "answer",
            StopReason::Stop,
            None,
        )])));
    session
        .prompt("question".to_owned(), None)
        .await
        .expect("prompt");
    assert_eq!(session.last_assistant_text().as_deref(), Some("answer"));
    let stats = session.session_stats();
    assert!(stats.total_messages >= 2);
    assert!(stats.user_messages >= 1);
    assert!(stats.assistant_messages >= 1);
    assert!(session.context_usage().is_some());
}

#[tokio::test]
async fn tree_navigation_summarizes_the_abandoned_branch() {
    let root = "/tmp/pi-session-model-tree";
    let mut manager = SessionManager::in_memory(Some(root), None, None).expect("session");
    let first = manager
        .append_message(AgentMessage::Standard(Message::User(UserMessage {
            content: UserContent::Text("first".to_owned()),
            timestamp: 0,
        })))
        .expect("append");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(assistant(
            "reply",
            StopReason::Stop,
            None,
        ))))
        .expect("append");
    let session = build_session(root, manager, default_settings()).await;
    session
        .model_runtime()
        .set_runtime_api_key("openai", "k".to_owned(), None)
        .await
        .expect("key");
    session
        .agent()
        .set_stream_function(Some(stream_fn(vec![assistant(
            "branch summary",
            StopReason::Stop,
            None,
        )])));
    let outcome = session
        .navigate_tree(
            &first,
            NavigateTreeOptions {
                summarize: true,
                ..NavigateTreeOptions::default()
            },
        )
        .await
        .expect("navigate");
    assert!(outcome.summary_entry.is_some());
    assert_eq!(outcome.editor_text.as_deref(), Some("first"));
}
