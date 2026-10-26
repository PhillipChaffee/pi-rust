//! The SDK session factory's defaults, tool selection, and model restore,
//! upstream's `test/sdk-session-manager.test.ts`,
//! `test/default-tools-setting.test.ts`, the `test/suite/regressions/3592`
//! case, and the scoping block `test/model-resolver.test.ts` deferred from
//! #121, at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;

use pi_coding_agent::core::sdk::CreateAgentSessionOptions;
use pi_coding_agent::core::sdk::NoTools;
use pi_coding_agent::core::sdk::create_agent_session;
use pi_coding_agent::model_registry::ModelRegistry;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;
use serde_json::json;

async fn create(
    fixture: &common::sdk_fixture::Fixture,
    registry: &ModelRegistry,
    settings: Settings,
    session_manager: Option<SessionManager>,
    mutate: impl FnOnce(&mut CreateAgentSessionOptions<InMemorySettingsStorage>),
) -> pi_coding_agent::core::sdk::CreateAgentSessionResult<InMemorySettingsStorage> {
    let mut options = CreateAgentSessionOptions {
        cwd: Some(fixture.cwd.clone()),
        agent_dir: Some(fixture.agent_dir.clone()),
        model_runtime: Some(registry.runtime().clone()),
        model: Some(common::sdk_fixture::faux_model()),
        thinking_level: None,
        scoped_models: None,
        no_tools: None,
        tools: None,
        exclude_tools: None,
        custom_tools: None,
        resource_loader: None,
        session_manager,
        settings_manager: Some(SettingsManager::in_memory(
            &settings,
            SettingsManagerCreateOptions::default(),
        )),
        session_start_event: None,
    };
    mutate(&mut options);
    create_agent_session(options)
        .await
        .expect("session creates")
}

#[tokio::test]
async fn default_session_path_lives_under_the_agent_dir() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let result = create(&fixture, &registry, Settings::new(), None, |_| {}).await;
    let session_file = result.session.session_file().expect("persisted path");
    assert!(
        session_file.starts_with(&format!("{}/sessions/", fixture.agent_dir)),
        "{session_file}"
    );
}

#[tokio::test]
async fn explicit_session_manager_is_kept() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let manager =
        SessionManager::in_memory(Some(&fixture.cwd), None, None).expect("session manager");
    let result = create(&fixture, &registry, Settings::new(), Some(manager), |_| {}).await;
    assert_eq!(result.session.session_file(), None);
}

#[tokio::test]
async fn cwd_derives_from_an_explicit_session_manager() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let manager =
        SessionManager::in_memory(Some(&fixture.cwd), None, None).expect("session manager");
    let result = create(
        &fixture,
        &registry,
        Settings::new(),
        Some(manager),
        |options| {
            options.cwd = None;
        },
    )
    .await;
    assert_eq!(result.session.cwd(), fixture.cwd);
}

fn tool_names(
    result: &pi_coding_agent::core::sdk::CreateAgentSessionResult<InMemorySettingsStorage>,
) -> Vec<String> {
    result.session.active_tool_names()
}

#[tokio::test]
async fn default_tools_setting_selects_the_initial_builtins() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let settings = Settings::from_iter([("defaultTools".to_owned(), json!(["read", "edit"]))]);
    let result = create(&fixture, &registry, settings, None, |_| {}).await;
    assert_eq!(
        tool_names(&result),
        vec!["edit".to_owned(), "read".to_owned()]
    );
}

#[tokio::test]
async fn explicit_tools_override_the_setting() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let settings = Settings::from_iter([("defaultTools".to_owned(), json!(["read", "edit"]))]);
    let result = create(&fixture, &registry, settings, None, |options| {
        options.tools = Some(vec!["bash".to_owned()]);
    })
    .await;
    assert_eq!(tool_names(&result), vec!["bash".to_owned()]);
}

#[tokio::test]
async fn exclude_tools_applies_after_the_allowlist() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let result = create(&fixture, &registry, Settings::new(), None, |options| {
        options.tools = Some(vec!["read".to_owned(), "bash".to_owned()]);
        options.exclude_tools = Some(vec!["bash".to_owned()]);
    })
    .await;
    assert_eq!(tool_names(&result), vec!["read".to_owned()]);
}

#[tokio::test]
async fn no_tools_all_disables_every_builtin() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let result = create(&fixture, &registry, Settings::new(), None, |options| {
        options.no_tools = Some(NoTools::All);
    })
    .await;
    assert!(tool_names(&result).is_empty());
}

fn seed_restorable_session(registry: &ModelRegistry) -> SessionManager {
    let _ = registry;
    let mut manager =
        SessionManager::in_memory(Some("/tmp/pi-sdk-restore"), None, None).expect("session");
    manager
        .append_model_change("faux", "faux-1")
        .expect("model change");
    manager
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("hello".to_owned()),
                timestamp: 1,
            }),
        ))
        .expect("message");
    manager
}

#[tokio::test]
async fn restores_the_saved_model_and_thinking_level() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let manager = seed_restorable_session(&registry);
    let result = create(
        &fixture,
        &registry,
        Settings::new(),
        Some(manager),
        |options| {
            options.model = None;
        },
    )
    .await;
    let model = result.session.model().expect("model");
    assert_eq!(model.id, "faux-1");
    assert_eq!(result.model_fallback_message, None);
}

#[tokio::test]
async fn reports_the_unrestorable_saved_model() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let mut manager = SessionManager::in_memory(Some("/tmp/pi-sdk-restore-missing"), None, None)
        .expect("session");
    manager
        .append_model_change("missing", "gone")
        .expect("model change");
    manager
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("hello".to_owned()),
                timestamp: 1,
            }),
        ))
        .expect("message");
    let result = create(
        &fixture,
        &registry,
        Settings::new(),
        Some(manager),
        |options| {
            options.model = None;
        },
    )
    .await;
    let message = result.model_fallback_message.expect("fallback message");
    assert!(
        message.starts_with("Could not restore model missing/gone"),
        "{message}"
    );
}

async fn scoped_session(
    fixture: &common::sdk_fixture::Fixture,
    registry: &ModelRegistry,
    scoped: bool,
    persisted_scope: Option<&[String]>,
) -> pi_coding_agent::core::sdk::CreateAgentSessionResult<InMemorySettingsStorage> {
    let mut settings = Settings::new();
    if let Some(scope) = persisted_scope {
        settings.insert("enabledModels".to_owned(), json!(scope));
    }
    let sonnet = common::sdk_fixture::faux_model();
    create(fixture, registry, settings, None, |options| {
        if scoped {
            options.scoped_models = Some(vec![pi_coding_agent::model_resolver::ScopedModel {
                model: sonnet.clone(),
                thinking_level: None,
            }]);
        }
    })
    .await
}

fn scoped_names(
    result: &pi_coding_agent::core::sdk::CreateAgentSessionResult<InMemorySettingsStorage>,
) -> Vec<String> {
    result
        .session
        .scoped_models()
        .into_iter()
        .map(|scoped| format!("{}/{}", scoped.model.provider.0, scoped.model.id))
        .collect()
}

fn enabled_models(
    result: &pi_coding_agent::core::sdk::CreateAgentSessionResult<InMemorySettingsStorage>,
) -> Option<Vec<String>> {
    result
        .session
        .settings_manager()
        .lock()
        .expect("settings")
        .get_enabled_models()
}

#[tokio::test]
async fn persisted_default_model_joins_an_existing_scope() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let persisted = vec!["faux/faux-1".to_owned()];
    let result = scoped_session(&fixture, &registry, true, Some(&persisted)).await;
    let next = registry
        .runtime()
        .get_model("faux", "faux-2")
        .expect("model");
    result
        .session
        .set_model(
            next,
            pi_coding_agent::core::agent_session::ModelMutationOptions { persist: true },
        )
        .await
        .expect("model set");
    assert_eq!(
        scoped_names(&result),
        vec!["faux/faux-1".to_owned(), "faux/faux-2".to_owned()]
    );
    assert_eq!(
        enabled_models(&result),
        Some(vec!["faux/faux-1".to_owned(), "faux/faux-2".to_owned()])
    );
}

#[tokio::test]
async fn persisted_default_model_does_not_create_a_scope() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let result = scoped_session(&fixture, &registry, false, None).await;
    let next = registry
        .runtime()
        .get_model("faux", "faux-2")
        .expect("model");
    result
        .session
        .set_model(
            next,
            pi_coding_agent::core::agent_session::ModelMutationOptions { persist: true },
        )
        .await
        .expect("model set");
    assert!(scoped_names(&result).is_empty());
    assert_eq!(enabled_models(&result), None);
}

#[tokio::test]
async fn session_only_model_changes_stay_out_of_scope() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let persisted = vec!["faux/faux-1".to_owned()];
    let result = scoped_session(&fixture, &registry, true, Some(&persisted)).await;
    let next = registry
        .runtime()
        .get_model("faux", "faux-2")
        .expect("model");
    result
        .session
        .set_model(
            next,
            pi_coding_agent::core::agent_session::ModelMutationOptions { persist: false },
        )
        .await
        .expect("model set");
    assert_eq!(scoped_names(&result), vec!["faux/faux-1".to_owned()]);
    assert_eq!(
        enabled_models(&result),
        Some(vec!["faux/faux-1".to_owned()])
    );
}

#[tokio::test]
async fn restores_a_stored_thinking_level() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let mut manager =
        SessionManager::in_memory(Some("/tmp/pi-sdk-thinking"), None, None).expect("session");
    manager
        .append_thinking_level_change("high")
        .expect("thinking change");
    manager
        .append_model_change("faux", "faux-1")
        .expect("model change");
    manager
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("hello".to_owned()),
                timestamp: 1,
            }),
        ))
        .expect("message");
    let result = create(
        &fixture,
        &registry,
        Settings::new(),
        Some(manager),
        |options| {
            options.model = None;
        },
    )
    .await;
    assert_eq!(
        result.session.thinking_level(),
        pi_agent_core::types::ThinkingLevel::High
    );
}

#[tokio::test]
async fn applies_the_per_model_thinking_override() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let settings = Settings::from_iter([(
        "modelThinkingLevels".to_owned(),
        json!({"faux/faux-1": "low"}),
    )]);
    let result = create(&fixture, &registry, settings, None, |_| {}).await;
    assert_eq!(
        result.session.thinking_level(),
        pi_agent_core::types::ThinkingLevel::Low
    );
}

fn custom_tool(name: &str) -> pi_coding_agent::extensions::types::ToolDefinition {
    let name = name.to_owned();
    pi_coding_agent::extensions::types::ToolDefinition {
        name: name.clone(),
        label: name,
        description: "custom".to_owned(),
        prompt_snippet: None,
        prompt_guidelines: None,
        parameters: json!({"type": "object", "properties": {}}),
        constrained_sampling: None,
        render_shell: None,
        render_call: None,
        render_result: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(|_id, _args, _signal, _update, _ctx| {
            Box::pin(async {
                Ok(pi_agent_core::types::AgentToolResult {
                    content: Vec::new(),
                    details: serde_json::Value::Null,
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
    }
}

#[tokio::test]
async fn no_tools_builtin_keeps_custom_tools_enabled() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let result = create(&fixture, &registry, Settings::new(), None, |options| {
        options.no_tools = Some(NoTools::Builtin);
        options.custom_tools = Some(vec![custom_tool("custom_tool")]);
    })
    .await;
    assert_eq!(tool_names(&result), vec!["custom_tool".to_owned()]);
}

#[tokio::test]
async fn explicit_thinking_level_is_clamped_to_the_model() {
    let fixture = common::sdk_fixture::Fixture::new();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let result = create(&fixture, &registry, Settings::new(), None, |options| {
        options.thinking_level = Some(pi_agent_core::types::ThinkingLevel::Max);
    })
    .await;
    let level = result.session.thinking_level();
    assert_ne!(level, pi_agent_core::types::ThinkingLevel::Off);
}
