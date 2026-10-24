//! The session runtime's replacement flows, upstream's
//! `test/suite/agent-session-runtime.test.ts` and
//! `test/suite/regressions/2860-replaced-session-context.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::Api;
use pi_ai::types::Modality;
use pi_ai::types::Model;
use pi_ai::types::ProviderId;
use pi_coding_agent::core::agent_session::ExtensionBindings;
use pi_coding_agent::core::agent_session_runtime::AgentSessionRuntimeError;
use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeFactory;
use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeOptions;
use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeResult;
use pi_coding_agent::core::agent_session_runtime::ForkOptions;
use pi_coding_agent::core::agent_session_runtime::NewSessionRequest;
use pi_coding_agent::core::agent_session_runtime::SwitchSessionOptions;
use pi_coding_agent::core::agent_session_runtime::create_agent_session_runtime;
use pi_coding_agent::core::agent_session_services::CreateAgentSessionFromServicesOptions;
use pi_coding_agent::core::agent_session_services::CreateAgentSessionServicesOptions;
use pi_coding_agent::core::agent_session_services::create_agent_session_from_services;
use pi_coding_agent::core::agent_session_services::create_agent_session_services;
use pi_coding_agent::core::sdk::CreateAgentSessionError;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::reasons::ForkPosition;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::results::SessionBeforeResult;
use pi_coding_agent::extensions::runtime::FlagType;
use pi_coding_agent::model_registry::ModelRegistry;
use pi_coding_agent::provider_composer::ProviderConfigInput;
use pi_coding_agent::provider_composer::ProviderModelInput;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
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
        input: vec![Modality::Text],
        cost: common::model_layer::zero_cost(),
        context_window: 128_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

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
                stream_simple: Some(Arc::new(|_model, _context, _options| {
                    let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
                    let message = pi_ai::types::AssistantMessage {
                        content: vec![pi_ai::types::AssistantBlock::Text(
                            pi_ai::types::TextContent {
                                text: "ok".to_owned(),
                                text_signature: None,
                            },
                        )],
                        api: Api::from("openai-completions"),
                        provider: ProviderId("faux".to_owned()),
                        model: "faux-1".to_owned(),
                        response_model: None,
                        response_id: None,
                        provider_thinking_level: None,
                        diagnostics: None,
                        usage: pi_ai::types::Usage::default(),
                        stop_reason: pi_ai::types::StopReason::Stop,
                        deferred: None,
                        error_message: None,
                        raw_stop_reason: None,
                        end_turn: None,
                        timestamp: 1,
                    };
                    stream.push(pi_ai::types::AssistantMessageEvent::Done {
                        reason: pi_ai::types::StopReason::Stop,
                        message,
                    });
                    stream
                })),
                models: Some(vec![ProviderModelInput {
                    id: "faux-1".to_owned(),
                    name: "Faux 1".to_owned(),
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
                }]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the provider registers");
    registry
}

#[derive(Default)]
struct Cancels {
    switch: AtomicBool,
    fork: AtomicBool,
}

fn event_factory(log: Arc<Mutex<Vec<String>>>, cancels: Arc<Cancels>) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let log = Arc::clone(&log);
            let cancels = Arc::clone(&cancels);
            Box::pin(async move {
                let switch_log = Arc::clone(&log);
                let switch_cancels = Arc::clone(&cancels);
                api.on(
                    "session_before_switch",
                    Arc::new(move |_event, _ctx| {
                        switch_log
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push("before_switch".to_owned());
                        let cancel = switch_cancels.switch.load(Ordering::SeqCst);
                        Box::pin(async move {
                            HandlerResult::SessionBefore(SessionBeforeResult::Switch { cancel })
                        })
                    }),
                )
                .expect("switch handler");
                let fork_log = Arc::clone(&log);
                let fork_cancels = Arc::clone(&cancels);
                api.on(
                    "session_before_fork",
                    Arc::new(move |_event, _ctx| {
                        fork_log
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push("before_fork".to_owned());
                        let cancel = fork_cancels.fork.load(Ordering::SeqCst);
                        Box::pin(async move {
                            HandlerResult::SessionBefore(SessionBeforeResult::Fork {
                                cancel,
                                skip_conversation_restore: false,
                            })
                        })
                    }),
                )
                .expect("fork handler");
                for channel in ["session_start", "session_shutdown"] {
                    let label = channel.to_owned();
                    let log = Arc::clone(&log);
                    api.on(
                        channel,
                        Arc::new(move |event: ExtensionEvent, _ctx| {
                            let name = match event {
                                ExtensionEvent::SessionStart { .. } => "start",
                                ExtensionEvent::SessionShutdown { .. } => "shutdown",
                                _ => "other",
                            };
                            log.lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push(name.to_owned());
                            let _ = &label;
                            Box::pin(async { HandlerResult::None })
                        }),
                    )
                    .expect("event handler");
                }
                api.register_flag("demo_flag".to_owned(), None, FlagType::Boolean, None)
                    .expect("flag");
                Ok(())
            })
        }),
    }
}

struct Harness {
    runtime:
        pi_coding_agent::core::agent_session_runtime::AgentSessionRuntime<InMemorySettingsStorage>,
    log: Arc<Mutex<Vec<String>>>,
    cancels: Arc<Cancels>,
    _temp: TempDir,
    cwd: String,
}

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

async fn harness_with(build: impl FnOnce(&str) -> SessionManager) -> Harness {
    let (temp, cwd, agent_dir) = fixture();
    let registry = fixture_registry().await;
    let model = fixture_model();
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let cancels = Arc::new(Cancels::default());
    let factory: CreateAgentSessionRuntimeFactory<InMemorySettingsStorage> = {
        let runtime = registry.runtime().clone();
        let model = model.clone();
        let log = Arc::clone(&log);
        let cancels = Arc::clone(&cancels);
        Arc::new(move |options: CreateAgentSessionRuntimeOptions| {
            let runtime = runtime.clone();
            let model = model.clone();
            let factory = event_factory(Arc::clone(&log), Arc::clone(&cancels));
            Box::pin(async move {
                let services = create_agent_session_services::<InMemorySettingsStorage>(
                    CreateAgentSessionServicesOptions {
                        cwd: options.cwd.clone(),
                        agent_dir: Some(options.agent_dir.clone()),
                        settings_manager: None,
                        model_runtime: Some(runtime),
                        model_runtime_signal: None,
                        extension_flag_values: None,
                        resource_loader_options: Some(DefaultResourceLoaderOptions {
                            cwd: options.cwd.clone(),
                            agent_dir: options.agent_dir.clone(),
                            extension_factories: vec![factory],
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
                .map_err(CreateAgentSessionError::ModelRuntime)?;
                let services_for_result = services.clone();
                let diagnostics = services.diagnostics.clone();
                let result =
                    create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
                        services,
                        session_manager: options.session_manager,
                        session_start_event: options.session_start_event,
                        model: Some(model),
                        thinking_level: None,
                        scoped_models: None,
                        tools: None,
                        exclude_tools: None,
                        no_tools: None,
                        custom_tools: None,
                    })
                    .await?;
                Ok(CreateAgentSessionRuntimeResult {
                    session: result.session,
                    extensions_result: result.extensions_result,
                    model_fallback_message: result.model_fallback_message,
                    services: services_for_result,
                    diagnostics,
                })
            })
        })
    };
    let runtime = create_agent_session_runtime(
        factory,
        CreateAgentSessionRuntimeOptions {
            cwd: cwd.clone(),
            agent_dir: agent_dir.clone(),
            session_manager: build(&cwd),
            session_start_event: None,
            project_trust_context: None,
        },
    )
    .await
    .expect("runtime");
    runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bindings");
    log.lock().unwrap_or_else(PoisonError::into_inner).clear();
    Harness {
        runtime,
        log,
        cancels,
        _temp: temp,
        cwd,
    }
}

async fn harness(session_manager: SessionManager) -> Harness {
    harness_with(move |_cwd| session_manager).await
}

async fn persisted_harness() -> Harness {
    harness_with(|cwd| SessionManager::create(cwd, None, None).expect("persisted manager")).await
}

fn entries_log(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    log.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

fn in_memory_manager(cwd: &str) -> SessionManager {
    SessionManager::in_memory(Some(cwd), None, None).expect("session manager")
}

fn assistant_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(pi_ai::types::Message::Assistant(
        pi_ai::types::AssistantMessage {
            content: vec![pi_ai::types::AssistantBlock::Text(
                pi_ai::types::TextContent {
                    text: text.to_owned(),
                    text_signature: None,
                },
            )],
            api: Api::from("openai-completions"),
            provider: ProviderId("faux".to_owned()),
            model: "faux-1".to_owned(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            diagnostics: None,
            usage: pi_ai::types::Usage::default(),
            stop_reason: pi_ai::types::StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1,
        },
    ))
}

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(pi_ai::types::Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Text(text.to_owned()),
        timestamp: 1,
    }))
}

fn persisted_manager(cwd: &str) -> SessionManager {
    SessionManager::create(cwd, None, None).expect("persisted manager")
}

#[tokio::test]
async fn new_session_replaces_the_session_and_emits_events() {
    let mut harness = persisted_harness().await;
    let original = harness.runtime.session().session_id();
    let outcome = harness
        .runtime
        .new_session(NewSessionRequest::default())
        .await
        .expect("new session");
    assert!(!outcome.cancelled);
    assert_ne!(original, harness.runtime.session().session_id());
    harness
        .runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bindings");
    assert_eq!(
        entries_log(&harness.log),
        vec![
            "before_switch".to_owned(),
            "shutdown".to_owned(),
            "start".to_owned()
        ]
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn new_session_honors_before_switch_cancellation() {
    let mut harness = persisted_harness().await;
    let original = harness.runtime.session().session_id();
    harness.cancels.switch.store(true, Ordering::SeqCst);
    let outcome = harness
        .runtime
        .new_session(NewSessionRequest::default())
        .await
        .expect("new session");
    assert!(outcome.cancelled);
    assert_eq!(original, harness.runtime.session().session_id());
    assert_eq!(entries_log(&harness.log), vec!["before_switch".to_owned()]);
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn new_session_runs_setup_before_loading_messages() {
    let mut harness = persisted_harness().await;
    let setup: pi_coding_agent::core::agent_session_runtime::NewSessionSetup =
        Box::new(|mut manager: SessionManager| {
            Box::pin(async move {
                let _ = manager.append_session_info("renamed");
                manager
            })
        });
    let outcome = harness
        .runtime
        .new_session(NewSessionRequest {
            parent_session: None,
            setup: Some(setup),
            with_session: None,
        })
        .await
        .expect("new session");
    assert!(!outcome.cancelled);
    assert_eq!(
        harness.runtime.session().session_name().as_deref(),
        Some("renamed")
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn fork_before_returns_the_user_text_and_at_does_not() {
    let cwd = "/tmp/pi-runtime-fork";
    let mut manager = in_memory_manager(cwd);
    let entry_id = manager
        .append_message(user_message("hello there"))
        .expect("message");
    let mut first = harness(manager).await;
    let outcome = first
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::Before),
                with_session: None,
            },
        )
        .await
        .expect("fork");
    assert!(!outcome.cancelled);
    assert_eq!(outcome.selected_text.as_deref(), Some("hello there"));
    assert_eq!(
        entries_log(&first.log),
        vec!["before_fork".to_owned(), "shutdown".to_owned()]
    );
    first.runtime.dispose().await;

    let mut manager = in_memory_manager(cwd);
    let entry_id = manager
        .append_message(user_message("hello there"))
        .expect("message");
    let mut second = harness(manager).await;
    let outcome = second
        .runtime
        .fork(
            &entry_id,
            ForkOptions {
                position: Some(ForkPosition::At),
                with_session: None,
            },
        )
        .await
        .expect("fork");
    assert_eq!(outcome.selected_text, None);
    second.runtime.dispose().await;
}

#[tokio::test]
async fn fork_rejects_an_invalid_entry_and_cancellation() {
    let mut harness = harness(in_memory_manager("/tmp/pi-runtime-fork-invalid")).await;
    let error = harness
        .runtime
        .fork("missing-entry", ForkOptions::default())
        .await
        .expect_err("invalid entry");
    assert!(error.to_string().contains("Invalid entry ID for forking"));
    harness.cancels.fork.store(true, Ordering::SeqCst);
    let outcome = harness
        .runtime
        .fork("missing-entry", ForkOptions::default())
        .await
        .expect("cancelled");
    assert!(outcome.cancelled);
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn switch_session_reopens_the_target() {
    let mut harness = harness(in_memory_manager("/tmp/pi-runtime-switch")).await;
    let mut target = persisted_manager(&harness.cwd);
    target
        .append_message(user_message("other"))
        .expect("target message");
    target
        .append_message(assistant_message("ok"))
        .expect("target flush");
    let target_id = target.session_id().to_owned();
    let target_file = target.session_file().expect("target file").to_owned();
    let outcome = harness
        .runtime
        .switch_session(&target_file, SwitchSessionOptions::default())
        .await
        .expect("switch");
    assert!(!outcome.cancelled);
    assert_eq!(harness.runtime.session().session_id(), target_id);
    harness
        .runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bindings");
    assert_eq!(
        entries_log(&harness.log),
        vec![
            "before_switch".to_owned(),
            "shutdown".to_owned(),
            "start".to_owned()
        ]
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn import_copies_the_file_and_reports_missing_sources() {
    let mut harness = persisted_harness().await;
    let missing = harness
        .runtime
        .import_from_jsonl("/tmp/definitely-not-a-session.jsonl", None)
        .await
        .expect_err("missing file");
    assert!(matches!(
        missing,
        AgentSessionRuntimeError::ImportFileNotFound(_)
    ));

    let mut source = persisted_manager(&harness.cwd);
    source
        .append_message(user_message("imported"))
        .expect("source message");
    source
        .append_message(assistant_message("ok"))
        .expect("source flush");
    let source_file = source.session_file().expect("source file").to_owned();
    let outcome = harness
        .runtime
        .import_from_jsonl(&source_file, None)
        .await
        .expect("import");
    assert!(!outcome.cancelled);
    let imported = harness
        .runtime
        .session()
        .session_file()
        .expect("imported file");
    assert!(imported.ends_with(&format!(
        "/{}",
        pi_coding_agent::utils::paths::basename_posix(&source_file)
    )));
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn replacement_callbacks_run_in_order() {
    let mut harness = harness(in_memory_manager("/tmp/pi-runtime-callbacks")).await;
    let log = Arc::clone(&harness.log);
    harness
        .runtime
        .set_rebind_session(Some(Arc::new(move |_session| {
            let log = Arc::clone(&log);
            Box::pin(async move {
                log.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push("rebind".to_owned());
            })
        })));
    let log = Arc::clone(&harness.log);
    harness
        .runtime
        .new_session(NewSessionRequest {
            parent_session: None,
            setup: None,
            with_session: Some(Box::new(move |_ctx| {
                let log = Arc::clone(&log);
                Box::pin(async move {
                    log.lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push("with".to_owned());
                })
            })),
        })
        .await
        .expect("new session");
    assert_eq!(
        entries_log(&harness.log),
        vec![
            "before_switch".to_owned(),
            "shutdown".to_owned(),
            "rebind".to_owned(),
            "with".to_owned()
        ]
    );
    harness.runtime.dispose().await;
}

#[tokio::test]
async fn dispose_emits_the_quit_shutdown() {
    let mut harness = harness(in_memory_manager("/tmp/pi-runtime-dispose")).await;
    harness.runtime.dispose().await;
    assert_eq!(entries_log(&harness.log), vec!["shutdown".to_owned()]);
}
