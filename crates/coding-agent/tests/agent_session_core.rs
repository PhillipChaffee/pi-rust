//! Boundary tests for the session event spine (ticket #149 slice 1).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::Agent;
use pi_agent_core::AgentInitialState;
use pi_agent_core::AgentOptions;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::core::agent_session::AgentSessionConfig;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::core::agent_session::parse_skill_block;
use pi_coding_agent::resource_loader::DefaultResourceLoader;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;

async fn build_session(
    root: &str,
) -> Arc<AgentSession<pi_coding_agent::settings_manager::InMemorySettingsStorage>> {
    let cwd = root.to_owned();
    let agent_dir = format!("{root}/agent");
    std::fs::create_dir_all(&agent_dir).expect("mkdir");
    let runtime = common::model_layer::create_in_memory_model_registry(
        common::model_layer::empty_auth_storage(),
    )
    .await
    .runtime()
    .clone();
    let model = common::model_layer::model("openai", "gpt-test");
    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentInitialState {
            system_prompt: "base".to_owned(),
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
            cwd: cwd.clone(),
            agent_dir,
            ..DefaultResourceLoaderOptions::default()
        },
        settings,
    );
    let manager = SessionManager::in_memory(Some(&cwd), None, None).expect("session");
    AgentSession::new(AgentSessionConfig {
        agent,
        session_manager: manager,
        resource_loader: pi_coding_agent::resource_loader::shared_resource_loader(loader),
        cwd,
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

#[test]
fn parses_a_skill_block_with_and_without_a_user_message() {
    let block = parse_skill_block(
        "<skill name=\"demo\" location=\"/skills/demo/SKILL.md\">\nbody\n</skill>\n\nuse it",
    )
    .expect("block");
    assert_eq!(block.name, "demo");
    assert_eq!(block.location, "/skills/demo/SKILL.md");
    assert_eq!(block.content, "body");
    assert_eq!(block.user_message.as_deref(), Some("use it"));

    let bare = parse_skill_block(
        "<skill name=\"demo\" location=\"/skills/demo/SKILL.md\">\nbody\n</skill>",
    )
    .expect("block");
    assert_eq!(bare.user_message, None);
}

#[test]
fn rejects_text_outside_the_anchored_skill_grammar() {
    assert!(parse_skill_block("plain text").is_none());
    assert!(parse_skill_block("<skill name=\"demo\">\nbody\n</skill>").is_none());
    assert!(
        parse_skill_block("<skill name=\"demo\" location=\"/x\">\nbody\n</skill>\ntrailing")
            .is_none()
    );
}

#[tokio::test]
async fn exposes_the_session_and_agent_state() {
    let session = build_session("/tmp/pi-session-spine").await;
    assert_eq!(session.cwd(), "/tmp/pi-session-spine");
    assert_eq!(session.model().expect("model").id, "gpt-test");
    assert_eq!(
        session.thinking_level(),
        pi_agent_core::types::ThinkingLevel::Off
    );
    assert!(session.system_prompt().contains("expert coding assistant"));
    assert!(session.is_idle());
    assert!(!session.is_streaming());
    assert_eq!(session.pending_message_count(), 0);
    assert!(session.session_file().is_none());
    assert!(!session.session_id().is_empty());
}

#[tokio::test]
async fn subscription_delivers_events_and_removal_stops_them() {
    let session = build_session("/tmp/pi-session-spine-sub").await;
    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let unsubscribe = session.subscribe(Arc::new(move |event: AgentSessionEvent| {
        let label = match event {
            AgentSessionEvent::SessionInfoChanged { .. } => "session_info",
            _ => "other",
        };
        sink.lock().expect("lock").push(label.to_owned());
    }));
    session.set_session_name("named");
    assert_eq!(session.session_name().as_deref(), Some("named"));
    assert_eq!(
        events.lock().expect("lock").as_slice(),
        ["session_info".to_string()]
    );

    unsubscribe();
    session.set_session_name("renamed");
    assert_eq!(
        events.lock().expect("lock").as_slice(),
        ["session_info".to_string()]
    );
}

#[tokio::test]
async fn queues_steering_and_follow_up_texts_and_clears_them() {
    let session = build_session("/tmp/pi-session-spine-queue").await;
    session
        .steer("steer me".to_string(), None, None)
        .await
        .expect("steer");
    session
        .follow_up("follow me".to_string(), None, None)
        .await
        .expect("follow up");
    assert_eq!(session.pending_message_count(), 2);
    assert_eq!(session.steering_messages(), ["steer me".to_string()]);
    assert_eq!(session.follow_up_messages(), ["follow me".to_string()]);

    let (steering, follow_up) = session.clear_queue();
    assert_eq!(steering, ["steer me".to_string()]);
    assert_eq!(follow_up, ["follow me".to_string()]);
    assert_eq!(session.pending_message_count(), 0);
}

#[tokio::test]
async fn dispose_disconnects_the_listener() {
    let session = build_session("/tmp/pi-session-spine-dispose").await;
    let events: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let sink = Arc::clone(&events);
    let _unsubscribe = session.subscribe(Arc::new(move |_event| {
        *sink.lock().expect("lock") += 1;
    }));
    session.dispose();
    session.set_session_name("after-dispose");
    assert_eq!(*events.lock().expect("lock"), 0);
}

#[tokio::test]
async fn clamps_thinking_levels_to_the_model_capabilities() {
    let session = build_session("/tmp/pi-session-spine-thinking").await;
    assert_eq!(
        session.available_thinking_levels(),
        [pi_agent_core::types::ThinkingLevel::Off]
    );
    assert!(!session.supports_thinking());
    assert_eq!(
        session.cycle_thinking_level(
            pi_coding_agent::core::agent_session::ModelMutationOptions::default()
        ),
        None
    );
    session.set_thinking_level(
        pi_agent_core::types::ThinkingLevel::High,
        pi_coding_agent::core::agent_session::ModelMutationOptions::default(),
    );
    assert_eq!(
        session.thinking_level(),
        pi_agent_core::types::ThinkingLevel::Off
    );
}

#[tokio::test]
async fn set_model_requires_configured_auth() {
    let session = build_session("/tmp/pi-session-spine-model").await;
    let error = session
        .set_model(
            common::model_layer::model("no-such-provider", "gpt-other"),
            pi_coding_agent::core::agent_session::ModelMutationOptions::default(),
        )
        .await
        .expect_err("no auth configured");
    assert!(error.contains("No API key for no-such-provider/gpt-other"));
}

#[tokio::test]
async fn cycling_a_single_scoped_model_is_a_no_op() {
    let session = build_session("/tmp/pi-session-spine-cycle").await;
    session.set_scoped_models(vec![pi_coding_agent::model_resolver::ScopedModel {
        model: common::model_layer::model("openai", "gpt-test"),
        thinking_level: None,
    }]);
    assert!(
        session
            .cycle_model(
                pi_coding_agent::core::agent_session::CycleDirection::default(),
                pi_coding_agent::core::agent_session::ModelMutationOptions::default(),
            )
            .is_none()
    );
}

#[tokio::test]
async fn retry_and_compaction_toggles_persist_to_settings() {
    let session = build_session("/tmp/pi-session-spine-toggles").await;
    session.set_auto_retry_enabled(false);
    assert!(!session.auto_retry_enabled());
    session.set_auto_compaction_enabled(false);
    assert!(!session.auto_compaction_enabled());
}
