//! `test/suite/regressions/3592-no-builtin-tools-keeps-extension-tools.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use pi_coding_agent::core::agent_session_services::CreateAgentSessionFromServicesOptions;
use pi_coding_agent::core::agent_session_services::CreateAgentSessionServicesOptions;
use pi_coding_agent::core::agent_session_services::create_agent_session_from_services;
use pi_coding_agent::core::agent_session_services::create_agent_session_services;
use pi_coding_agent::core::sdk::NoTools;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;

use common::extension_regressions::manager;
use common::extension_regressions::sdk_session;
use common::extension_regressions::session_start_tool_extension;
use common::extension_regressions::sorted_tool_names;
use common::extension_regressions::tool_definition;

fn dynamic_tool_extension() -> pi_coding_agent::extensions::api::InlineExtension {
    session_start_tool_extension(vec![tool_definition(
        "dynamic_tool",
        "Tool registered from session_start",
        Some("Run dynamic test behavior"),
    )])
}

/// #3592: suppressing the built-ins keeps extension tools active.
#[tokio::test]
async fn no_builtin_tools_keeps_extension_tools_enabled() {
    let session = sdk_session(
        "/tmp/pi-ext-reg-no-builtin",
        dynamic_tool_extension(),
        None,
        None,
        Some(NoTools::Builtin),
    )
    .await;

    assert_eq!(
        sorted_tool_names(&session),
        [
            "bash".to_owned(),
            "dynamic_tool".to_owned(),
            "edit".to_owned(),
            "find".to_owned(),
            "grep".to_owned(),
            "ls".to_owned(),
            "powershell".to_owned(),
            "read".to_owned(),
            "write".to_owned(),
        ]
    );
    assert_eq!(session.active_tool_names(), ["dynamic_tool".to_owned()]);
    assert!(
        session
            .system_prompt()
            .contains("- dynamic_tool: Run dynamic test behavior")
    );
    assert!(!session.system_prompt().contains("- read:"));
    assert!(!session.system_prompt().contains("- bash:"));
}

/// #3592: `noTools: "all"` still disables every tool.
#[tokio::test]
async fn no_tools_all_still_disables_every_tool() {
    let session = sdk_session(
        "/tmp/pi-ext-reg-no-tools",
        dynamic_tool_extension(),
        None,
        None,
        Some(NoTools::All),
    )
    .await;

    assert!(session.get_all_tools().is_empty());
    assert!(session.active_tool_names().is_empty());
    assert!(session.system_prompt().contains("Available tools:\n(none)"));
}

/// #3592: the service-based factory propagates `noTools` too.
#[tokio::test]
async fn no_tools_propagates_through_service_session_creation() {
    let root = "/tmp/pi-ext-reg-no-tools-services";
    let agent_dir = format!("{root}/agent");
    std::fs::create_dir_all(&agent_dir).expect("mkdir");
    let registry = common::model_layer::create_in_memory_model_registry(
        common::model_layer::empty_auth_storage(),
    )
    .await;
    let settings_manager =
        SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default());
    let services = create_agent_session_services::<InMemorySettingsStorage>(
        CreateAgentSessionServicesOptions {
            cwd: root.to_owned(),
            agent_dir: Some(agent_dir.clone()),
            settings_manager: Some(settings_manager),
            model_runtime: Some(registry.runtime().clone()),
            model_runtime_signal: None,
            extension_flag_values: None,
            resource_loader_options: Some(DefaultResourceLoaderOptions {
                cwd: root.to_owned(),
                agent_dir: agent_dir.clone(),
                no_skills: true,
                no_prompt_templates: true,
                no_themes: true,
                no_context_files: true,
                ..DefaultResourceLoaderOptions::default()
            }),
            resource_loader_reload_options: None,
        },
    )
    .await
    .expect("services");
    let result = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services,
        session_manager: manager(root),
        session_start_event: None,
        model: Some(common::agent_session::anthropic_model()),
        thinking_level: None,
        scoped_models: None,
        tools: None,
        exclude_tools: None,
        no_tools: Some(NoTools::Builtin),
        custom_tools: None,
    })
    .await
    .expect("session");

    assert!(result.session.active_tool_names().is_empty());
    assert!(
        result
            .session
            .system_prompt()
            .contains("Available tools:\n(none)")
    );
    assert!(!result.session.system_prompt().contains("- read:"));
}
