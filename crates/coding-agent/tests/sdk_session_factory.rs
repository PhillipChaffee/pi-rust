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

use pi_ai::types::Api;
use pi_ai::types::Modality;
use pi_ai::types::Model;
use pi_ai::types::ProviderId;
use pi_coding_agent::core::sdk::CreateAgentSessionOptions;
use pi_coding_agent::core::sdk::NoTools;
use pi_coding_agent::core::sdk::create_agent_session;
use pi_coding_agent::model_registry::ModelRegistry;
use pi_coding_agent::provider_composer::ProviderConfigInput;
use pi_coding_agent::provider_composer::ProviderModelInput;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;
use serde_json::json;
use tempfile::TempDir;

fn fixture_model() -> Model {
    Model {
        id: "faux-1".to_owned(),
        name: "Faux 1".to_owned(),
        api: Api::from("openai-completions"),
        provider: ProviderId("faux".to_owned()),
        base_url: "https://faux.invalid/v1".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text, Modality::Image],
        cost: common::model_layer::zero_cost(),
        context_window: 128_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// A registry with the fixture provider registered and its key configured.
async fn fixture_registry() -> ModelRegistry {
    let registry = common::model_layer::create_in_memory_model_registry(
        common::model_layer::empty_auth_storage(),
    )
    .await;
    registry
        .runtime()
        .register_provider(
            "faux",
            ProviderConfigInput {
                base_url: Some("https://faux.invalid/v1".to_owned()),
                api: Some(Api::from("openai-completions")),
                api_key: Some("faux-key".to_owned()),
                models: Some(vec![
                    ProviderModelInput {
                        id: "faux-1".to_owned(),
                        name: "Faux 1".to_owned(),
                        api: Some(Api::from("openai-completions")),
                        base_url: None,
                        reasoning: true,
                        thinking_level_map: None,
                        input: vec![Modality::Text, Modality::Image],
                        cost: common::model_layer::zero_cost(),
                        context_window: 128_000,
                        max_tokens: 4096,
                        sampling_params: None,
                        headers: None,
                        compat: None,
                    },
                    ProviderModelInput {
                        id: "faux-2".to_owned(),
                        name: "Faux 2".to_owned(),
                        api: Some(Api::from("openai-completions")),
                        base_url: None,
                        reasoning: true,
                        thinking_level_map: None,
                        input: vec![Modality::Text],
                        cost: common::model_layer::zero_cost(),
                        context_window: 128_000,
                        max_tokens: 4096,
                        sampling_params: None,
                        headers: None,
                        compat: None,
                    },
                ]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the provider registers");
    registry
}

struct Fixture {
    _temp: TempDir,
    cwd: String,
    agent_dir: String,
}

fn fixture() -> Fixture {
    let temp = TempDir::new().expect("tempdir");
    let cwd = temp.path().join("project");
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&cwd).expect("cwd");
    std::fs::create_dir_all(&agent_dir).expect("agent dir");
    Fixture {
        cwd: cwd.display().to_string(),
        agent_dir: agent_dir.display().to_string(),
        _temp: temp,
    }
}

async fn create(
    fixture: &Fixture,
    registry: &ModelRegistry,
    settings: Settings,
    session_manager: Option<SessionManager>,
    mutate: impl FnOnce(&mut CreateAgentSessionOptions<InMemorySettingsStorage>),
) -> pi_coding_agent::core::sdk::CreateAgentSessionResult<InMemorySettingsStorage> {
    let mut options = CreateAgentSessionOptions {
        cwd: Some(fixture.cwd.clone()),
        agent_dir: Some(fixture.agent_dir.clone()),
        model_runtime: Some(registry.runtime().clone()),
        model: Some(fixture_model()),
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
    let fixture = fixture();
    let registry = fixture_registry().await;
    let result = create(&fixture, &registry, Settings::new(), None, |_| {}).await;
    let session_file = result.session.session_file().expect("persisted path");
    assert!(
        session_file.starts_with(&format!("{}/sessions/", fixture.agent_dir)),
        "{session_file}"
    );
}

#[tokio::test]
async fn explicit_session_manager_is_kept() {
    let fixture = fixture();
    let registry = fixture_registry().await;
    let manager =
        SessionManager::in_memory(Some(&fixture.cwd), None, None).expect("session manager");
    let result = create(&fixture, &registry, Settings::new(), Some(manager), |_| {}).await;
    assert_eq!(result.session.session_file(), None);
}

#[tokio::test]
async fn cwd_derives_from_an_explicit_session_manager() {
    let fixture = fixture();
    let registry = fixture_registry().await;
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
    let fixture = fixture();
    let registry = fixture_registry().await;
    let settings = Settings::from_iter([("defaultTools".to_owned(), json!(["read", "edit"]))]);
    let result = create(&fixture, &registry, settings, None, |_| {}).await;
    assert_eq!(
        tool_names(&result),
        vec!["edit".to_owned(), "read".to_owned()]
    );
}

#[tokio::test]
async fn explicit_tools_override_the_setting() {
    let fixture = fixture();
    let registry = fixture_registry().await;
    let settings = Settings::from_iter([("defaultTools".to_owned(), json!(["read", "edit"]))]);
    let result = create(&fixture, &registry, settings, None, |options| {
        options.tools = Some(vec!["bash".to_owned()]);
    })
    .await;
    assert_eq!(tool_names(&result), vec!["bash".to_owned()]);
}

#[tokio::test]
async fn exclude_tools_applies_after_the_allowlist() {
    let fixture = fixture();
    let registry = fixture_registry().await;
    let result = create(&fixture, &registry, Settings::new(), None, |options| {
        options.tools = Some(vec!["read".to_owned(), "bash".to_owned()]);
        options.exclude_tools = Some(vec!["bash".to_owned()]);
    })
    .await;
    assert_eq!(tool_names(&result), vec!["read".to_owned()]);
}

#[tokio::test]
async fn no_tools_all_disables_every_builtin() {
    let fixture = fixture();
    let registry = fixture_registry().await;
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
    let fixture = fixture();
    let registry = fixture_registry().await;
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
    let fixture = fixture();
    let registry = fixture_registry().await;
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
    fixture: &Fixture,
    registry: &ModelRegistry,
    scoped: bool,
    persisted_scope: Option<&[String]>,
) -> pi_coding_agent::core::sdk::CreateAgentSessionResult<InMemorySettingsStorage> {
    let mut settings = Settings::new();
    if let Some(scope) = persisted_scope {
        settings.insert("enabledModels".to_owned(), json!(scope));
    }
    let sonnet = fixture_model();
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
    let fixture = fixture();
    let registry = fixture_registry().await;
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
    let fixture = fixture();
    let registry = fixture_registry().await;
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
    let fixture = fixture();
    let registry = fixture_registry().await;
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
