//! Extension binding and action coverage for #149.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_agent_core::Agent;
use pi_agent_core::AgentInitialState;
use pi_agent_core::AgentOptions;
use pi_ai::types::BoxedFuture;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::core::agent_session::AgentSessionConfig;
use pi_coding_agent::core::agent_session::ExtensionBindings;
use pi_coding_agent::extensions::reasons::ExtensionMode;
use pi_coding_agent::extensions::runner::CommandContextActions;
use pi_coding_agent::resource_loader::DefaultResourceLoader;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;

async fn build_session(
    root: &str,
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
    let manager = SessionManager::in_memory(Some(root), None, None).expect("session");
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

fn command_actions() -> CommandContextActions {
    let noop = Arc::new(|| -> BoxedFuture<'static, ()> { Box::pin(async {}) });
    let proceed =
        Arc::new(
            |_options: pi_coding_agent::extensions::runner::NewSessionOptions| -> BoxedFuture<
                'static,
                pi_coding_agent::extensions::runner::SessionReplaceOutcome,
            > {
                Box::pin(async {
                    pi_coding_agent::extensions::runner::SessionReplaceOutcome::proceeded()
                })
            },
        );
    let fork =
        Arc::new(
            |_id: String,
             _options: pi_coding_agent::extensions::runner::ForkOptions|
             -> BoxedFuture<
                'static,
                pi_coding_agent::extensions::runner::SessionReplaceOutcome,
            > {
                Box::pin(async {
                    pi_coding_agent::extensions::runner::SessionReplaceOutcome::proceeded()
                })
            },
        );
    let navigate =
        Arc::new(
            |_id: String,
             _options: pi_coding_agent::extensions::runner::NavigateTreeOptions|
             -> BoxedFuture<
                'static,
                pi_coding_agent::extensions::runner::SessionReplaceOutcome,
            > {
                Box::pin(async {
                    pi_coding_agent::extensions::runner::SessionReplaceOutcome::proceeded()
                })
            },
        );
    let switch =
        Arc::new(
            |_path: String,
             _options: pi_coding_agent::extensions::runner::SwitchSessionOptions|
             -> BoxedFuture<
                'static,
                pi_coding_agent::extensions::runner::SessionReplaceOutcome,
            > {
                Box::pin(async {
                    pi_coding_agent::extensions::runner::SessionReplaceOutcome::proceeded()
                })
            },
        );
    CommandContextActions {
        wait_for_idle: noop.clone(),
        new_session: proceed,
        fork,
        navigate_tree: navigate,
        switch_session: switch,
        reload: noop,
    }
}

#[tokio::test]
async fn bound_actions_drive_the_session() {
    let root = "/tmp/pi-session-ext-actions";
    let session = build_session(root).await;
    session
        .model_runtime()
        .set_runtime_api_key("openai", "k".to_owned(), None)
        .await
        .expect("key");
    let errors = Arc::new(AtomicUsize::new(0));
    let error_sink = Arc::clone(&errors);
    session
        .bind_extensions(ExtensionBindings {
            mode: Some(ExtensionMode::Rpc),
            command_context_actions: Some(command_actions()),
            abort_handler: Some(Arc::new(|| {})),
            shutdown_handler: Some(Arc::new(|| {})),
            on_error: Some(Arc::new(move |_error| {
                error_sink.fetch_add(1, Ordering::SeqCst);
            })),
        })
        .await
        .expect("bind");
    let runner = session.extension_runner();
    let actions = runner.bound_actions().expect("actions");
    assert!(!(actions.get_active_tools)().is_empty());
    assert!(!(actions.get_all_tools)().is_empty());
    let _ = (actions.get_commands)();
    (actions.set_session_name)("bound".to_owned());
    assert_eq!((actions.get_session_name)().as_deref(), Some("bound"));
    (actions.append_entry)("note".to_owned(), None);
    (actions.set_active_tools)(vec!["read".to_owned()]);
    assert_eq!((actions.get_active_tools)(), ["read".to_owned()]);
    (actions.refresh_tools)();
    (actions.set_thinking_level)(pi_agent_core::types::ThinkingLevel::Off);
    assert_eq!(
        (actions.get_thinking_level)(),
        pi_agent_core::types::ThinkingLevel::Off
    );
    assert!((actions.set_model)(common::model_layer::model("openai", "gpt-test")).await);
    (actions.send_message)(
        pi_coding_agent::extensions::reasons::CustomMessagePayload {
            custom_type: "note".to_owned(),
            content: vec![pi_coding_agent::extensions::reasons::text_block("body")],
            display: false,
            details: None,
        },
        None,
    );
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let context = runner.command_context();
    assert!(context.get_model().expect("model").is_some());
    assert!(context.get_scoped_models().expect("scoped").is_empty());
    assert!(context.is_idle().expect("idle"));
    let _ = context.is_project_trusted().expect("trust");
    assert!(context.get_signal().expect("signal").is_none());
    assert!(!context.has_pending_messages().expect("pending"));
    assert!(context.get_context_usage().expect("usage").is_some());
    assert!(
        context
            .get_system_prompt()
            .expect("prompt")
            .contains("expert")
    );
    assert_eq!(
        context.get_system_prompt_options().expect("options").cwd,
        root
    );
    let _ = context.compact(None);
    let _ = context.shutdown();
    assert_eq!(errors.load(Ordering::SeqCst), 0);
}
