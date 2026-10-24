//! The cwd-bound services bundle, upstream's `test/suite/agent-session-runtime.test.ts`
//! service-creation cases and `agent-session-services.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;

use pi_ai::types::Api;
use pi_ai::types::Modality;
use pi_coding_agent::core::agent_session_services::AgentSessionRuntimeDiagnosticKind;
use pi_coding_agent::core::agent_session_services::CreateAgentSessionServicesOptions;
use pi_coding_agent::core::agent_session_services::create_agent_session_services;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::runtime::FlagType;
use pi_coding_agent::extensions::runtime::FlagValue;
use pi_coding_agent::provider_composer::ProviderConfigInput;
use pi_coding_agent::provider_composer::ProviderModelInput;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use tempfile::TempDir;

fn fixture() -> (TempDir, String, String) {
    let temp = TempDir::new().expect("tempdir");
    let cwd = temp.path().join("project");
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&cwd).expect("cwd");
    std::fs::create_dir_all(&agent_dir).expect("agent dir");
    (
        temp,
        cwd.display().to_string(),
        agent_dir.display().to_string(),
    )
}

fn bare_loader_options() -> DefaultResourceLoaderOptions {
    DefaultResourceLoaderOptions {
        no_skills: true,
        no_prompt_templates: true,
        no_themes: true,
        no_context_files: true,
        ..DefaultResourceLoaderOptions::default()
    }
}

fn flag_factory() -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(|api: Arc<ExtensionApi>| {
            Box::pin(async move {
                api.register_flag("bool_flag".to_owned(), None, FlagType::Boolean, None)
                    .expect("boolean flag");
                api.register_flag("text_flag".to_owned(), None, FlagType::String, None)
                    .expect("string flag");
                Ok(())
            })
        }),
    }
}

#[tokio::test]
async fn applies_extension_flag_values_and_reports_bad_ones() {
    let (_temp, cwd, agent_dir) = fixture();
    let registry = common::model_layer::create_in_memory_model_registry(
        common::model_layer::empty_auth_storage(),
    )
    .await;
    let mut loader_options = bare_loader_options();
    loader_options.extension_factories = vec![flag_factory()];
    let services = create_agent_session_services::<InMemorySettingsStorage>(
        CreateAgentSessionServicesOptions {
            cwd,
            agent_dir: Some(agent_dir),
            settings_manager: None,
            model_runtime: Some(registry.runtime().clone()),
            model_runtime_signal: None,
            extension_flag_values: Some(std::collections::BTreeMap::from([
                ("bool_flag".to_owned(), FlagValue::Bool(false)),
                ("text_flag".to_owned(), FlagValue::Bool(true)),
                ("unknown".to_owned(), FlagValue::Text("x".to_owned())),
            ])),
            resource_loader_options: Some(loader_options),
            resource_loader_reload_options: None,
        },
    )
    .await
    .expect("services");

    let runtime = services
        .resource_loader
        .lock()
        .expect("loader")
        .as_ref()
        .expect("present")
        .get_extensions()
        .runtime;
    assert_eq!(runtime.flag_value("bool_flag"), Some(FlagValue::Bool(true)));
    assert_eq!(runtime.flag_value("text_flag"), None);
    assert_eq!(services.diagnostics.len(), 2);
    assert_eq!(
        services.diagnostics[0].kind,
        AgentSessionRuntimeDiagnosticKind::Error
    );
    assert!(services.diagnostics[0].message.contains("requires a value"));
    assert!(
        services.diagnostics[1]
            .message
            .starts_with("Unknown option")
    );
}

fn provider_factory() -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(|api: Arc<ExtensionApi>| {
            Box::pin(async move {
                api.register_provider(
                    "ext-prov".to_owned(),
                    ProviderConfigInput {
                        base_url: Some("https://ext.invalid/v1".to_owned()),
                        api: Some(Api::from("openai-completions")),
                        api_key: Some("ext-key".to_owned()),
                        models: Some(vec![ProviderModelInput {
                            id: "ext-model".to_owned(),
                            name: "Ext Model".to_owned(),
                            api: Some(Api::from("openai-completions")),
                            base_url: None,
                            reasoning: false,
                            thinking_level_map: None,
                            input: vec![Modality::Text],
                            cost: common::model_layer::zero_cost(),
                            context_window: 1000,
                            max_tokens: 100,
                            sampling_params: None,
                            headers: None,
                            compat: None,
                        }]),
                        ..ProviderConfigInput::default()
                    },
                )
                .expect("provider");
                Ok(())
            })
        }),
    }
}

#[tokio::test]
async fn flushes_queued_provider_registrations_into_the_runtime() {
    let (_temp, cwd, agent_dir) = fixture();
    let registry = common::model_layer::create_in_memory_model_registry(
        common::model_layer::empty_auth_storage(),
    )
    .await;
    let mut loader_options = bare_loader_options();
    loader_options.extension_factories = vec![provider_factory()];
    let services = create_agent_session_services::<InMemorySettingsStorage>(
        CreateAgentSessionServicesOptions {
            cwd,
            agent_dir: Some(agent_dir),
            settings_manager: None,
            model_runtime: Some(registry.runtime().clone()),
            model_runtime_signal: None,
            extension_flag_values: None,
            resource_loader_options: Some(loader_options),
            resource_loader_reload_options: None,
        },
    )
    .await
    .expect("services");

    assert!(
        services
            .model_runtime
            .get_model("ext-prov", "ext-model")
            .is_some()
    );
    assert!(services.diagnostics.is_empty());
}
