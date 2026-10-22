#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use super::*;

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use pi_agent_core::types::ThinkingLevel;
use pi_ai::auth::credential_store::InMemoryCredentialStore;

use crate::model_registry::ModelRegistry;
use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::session_manager::SessionManager;
use crate::source_info::{SourceInfo, SourceOrigin, SourceScope, SyntheticSourceOptions};
use crate::tools::bash::create_bash_tool_definition;

fn source_info() -> SourceInfo {
    crate::source_info::create_synthetic_source_info(
        "/ext/test.ts",
        &SyntheticSourceOptions {
            source: "extension:test".to_string(),
            scope: Some(SourceScope::Temporary),
            origin: Some(SourceOrigin::TopLevel),
            base_dir: None,
        },
    )
}

fn plain_manager() -> SessionManager {
    SessionManager::new("/tmp/project", "/tmp/sessions", None, false, None, None)
        .expect("the in-memory manager constructs")
}

async fn offline_registry() -> ModelRegistry {
    // In-crate constructor over a runtime facade; no disk or network.
    let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(Arc::new(InMemoryCredentialStore::default())),
        models_path: Some(None),
        allow_model_network: false,
        refresh_on_create: Some(false),
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .expect("the runtime constructs offline");
    ModelRegistry::new(runtime)
}

fn always_active() -> AssertActiveFn {
    Arc::new(|| Ok(()))
}

#[test]
fn extension_registers_handlers_tools_and_commands() {
    let extension = Extension::new(
        "/ext/test.ts".to_string(),
        "/ext/test.ts".to_string(),
        false,
        source_info(),
    );
    let handler: ExtensionHandler = Arc::new(|_event: ExtensionEvent, _ctx: HandlerContext| {
        Box::pin(async { HandlerResult::None })
    });
    extension.on("session_start", Arc::clone(&handler));
    extension.on("session_start", Arc::clone(&handler));
    extension.on("tool_call", handler);

    extension.register_tool(RegisteredTool {
        definition: create_bash_tool_definition(".", None),
        source_info: source_info(),
    });
    extension.register_command(ResolvedCommand {
        name: "ship".to_string(),
        invocation_name: "ship".to_string(),
        description: Some("ships it".to_string()),
        source_info: source_info(),
        handler: Arc::new(|_args: String, _ctx: HandlerContext| Box::pin(async {})),
    });

    let handlers = extension
        .handlers
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(handlers["session_start"].len(), 2);
    assert_eq!(handlers["tool_call"].len(), 1);
    drop(handlers);
    assert_eq!(
        extension
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        1
    );
    let commands = extension
        .commands
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(commands.len(), 1);
    assert_eq!(commands[0].invocation_name, "ship");
    drop(commands);

    let rendered = format!("{extension:?}");
    assert!(rendered.contains("session_start"), "{rendered}");
    assert!(rendered.contains("tools: 1"), "{rendered}");
    assert!(rendered.contains("commands: 1"), "{rendered}");
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the bound/unbound/stale ladder walks every accessor three ways to pin the default contracts"
)]
async fn context_accessors_bind_default_and_fail_stale() {
    let registry = offline_registry().await;
    let stale_registry = registry.clone();
    let manager = plain_manager();
    let unbound_manager = plain_manager();
    let stale_manager = plain_manager();

    let ran_shutdown = Arc::new(AtomicBool::new(false));
    let ran_abort = Arc::new(AtomicBool::new(false));
    let ran_compact = Arc::new(AtomicBool::new(false));
    let bound_actions = Arc::new(ContextActions {
        get_model: Some(Arc::new(|| {
            Some(pi_ai::types::Model {
                id: "gpt-test".to_string(),
                name: "GPT Test".to_string(),
                api: pi_ai::types::Api::from("openai"),
                provider: pi_ai::types::ProviderId::from("openai"),
                base_url: "https://example.invalid".to_string(),
                reasoning: false,
                thinking_level_map: None,
                input: Vec::new(),
                cost: pi_ai::types::ModelCost::default(),
                context_window: 0,
                max_tokens: 0,
                sampling_params: None,
                headers: None,
                compat: None,
            })
        })),
        get_scoped_models: Some(Arc::new(Vec::new)),
        get_thinking_level: Some(Arc::new(|| ThinkingLevel::High)),
        is_idle: Some(Arc::new(|| false)),
        is_project_trusted: Some(Arc::new(|| false)),
        get_signal: Some(Arc::new(|| None)),
        abort: Some({
            let flag = Arc::clone(&ran_abort);
            Arc::new(move || flag.store(true, Ordering::SeqCst))
        }),
        has_pending_messages: Some(Arc::new(|| true)),
        shutdown: Some({
            let flag = Arc::clone(&ran_shutdown);
            Arc::new(move || flag.store(true, Ordering::SeqCst))
        }),
        get_context_usage: Some(Arc::new(|| {
            Some(ContextUsage {
                tokens: Some(1_000),
                context_window: 200_000,
                percent: Some(0.5),
            })
        })),
        compact: Some({
            let flag = Arc::clone(&ran_compact);
            Arc::new(move |_instructions: Option<String>| {
                flag.store(true, Ordering::SeqCst);
            })
        }),
        get_system_prompt: Some(Arc::new(|| "bound prompt".to_string())),
        get_system_prompt_options: Some(Arc::new(|| {
            crate::system_prompt::BuildSystemPromptOptions {
                cwd: "/bound".to_string(),
                ..Default::default()
            }
        })),
    });

    let bound = HandlerContext::new(
        "/tmp/project".to_string(),
        crate::extensions::reasons::ExtensionMode::Rpc,
        Some(Arc::new("ui".to_string())),
        manager,
        registry.clone(),
        always_active(),
        Arc::clone(&bound_actions),
    );
    assert_eq!(bound.cwd().expect("cwd"), "/tmp/project");
    assert_eq!(
        bound.mode().expect("mode"),
        crate::extensions::reasons::ExtensionMode::Rpc
    );
    assert!(bound.has_ui().expect("has_ui"));
    assert!(bound.ui().expect("ui").is_some());
    assert!(!bound.session_manager().expect("sm").session_id.is_empty());
    bound.model_registry().expect("mr");
    assert_eq!(
        bound.get_model().expect("model").map(|model| model.id),
        Some("gpt-test".to_string())
    );
    assert!(bound.get_scoped_models().expect("scoped").is_empty());
    assert_eq!(
        bound.get_thinking_level().expect("level"),
        ThinkingLevel::High
    );
    assert!(!bound.is_idle().expect("idle"));
    assert!(!bound.is_project_trusted().expect("trusted"));
    assert!(bound.get_signal().expect("signal").is_none());
    bound.abort().expect("abort");
    assert!(ran_abort.load(Ordering::SeqCst));
    assert!(bound.has_pending_messages().expect("pending"));
    bound.shutdown().expect("shutdown");
    assert!(ran_shutdown.load(Ordering::SeqCst));
    assert_eq!(
        bound.get_context_usage().expect("usage"),
        Some(ContextUsage {
            tokens: Some(1_000),
            context_window: 200_000,
            percent: Some(0.5),
        })
    );
    bound
        .compact(Some("instructions".to_string()))
        .expect("compact");
    assert!(ran_compact.load(Ordering::SeqCst));
    assert_eq!(bound.get_system_prompt().expect("prompt"), "bound prompt");
    let options = bound.get_system_prompt_options().expect("options");
    assert_eq!(options.cwd, "/bound");
    assert!(format!("{bound:?}").contains("/tmp/project"));

    let tool = RegisteredTool {
        definition: create_bash_tool_definition(".", None),
        source_info: source_info(),
    };
    assert!(format!("{tool:?}").contains("RegisteredTool"));
    let command = ResolvedCommand {
        name: "ship".to_string(),
        invocation_name: "ship".to_string(),
        description: None,
        source_info: source_info(),
        handler: Arc::new(|_args: String, _ctx: HandlerContext| Box::pin(async {})),
    };
    assert!(format!("{command:?}").contains("ResolvedCommand"));
    assert!(format!("{:?}", Arc::clone(&bound_actions)).contains("bound_actions: 13"));

    let unbound = HandlerContext::new(
        "/tmp/project".to_string(),
        crate::extensions::reasons::ExtensionMode::Print,
        None,
        unbound_manager,
        registry,
        always_active(),
        Arc::new(ContextActions::default()),
    );
    assert!(!unbound.has_ui().expect("has_ui"));
    assert!(unbound.ui().expect("ui").is_none());
    assert!(unbound.get_model().expect("model").is_none());
    assert!(unbound.get_scoped_models().expect("scoped").is_empty());
    assert_eq!(
        unbound.get_thinking_level().expect("level"),
        ThinkingLevel::Off
    );
    assert!(unbound.is_idle().expect("idle"));
    assert!(unbound.is_project_trusted().expect("trusted"));
    assert!(unbound.get_signal().expect("signal").is_none());
    unbound.abort().expect("abort");
    assert!(!unbound.has_pending_messages().expect("pending"));
    unbound.shutdown().expect("shutdown");
    assert!(unbound.get_context_usage().expect("usage").is_none());
    unbound.compact(None).expect("compact");
    assert!(unbound.get_system_prompt().expect("prompt").is_empty());
    let options = unbound.get_system_prompt_options().expect("options");
    assert_eq!(options.cwd, "/tmp/project");

    let stale = HandlerContext::new(
        "/tmp/project".to_string(),
        crate::extensions::reasons::ExtensionMode::Print,
        None,
        stale_manager,
        stale_registry,
        Arc::new(|| Err("stale".to_string())),
        Arc::new(ContextActions::default()),
    );
    assert_eq!(stale.assert_active(), Err("stale".to_string()));
    assert!(stale.cwd().is_err());
    assert!(stale.mode().is_err());
    assert!(stale.has_ui().is_err());
    assert!(stale.ui().is_err());
    assert!(stale.session_manager().is_err());
    assert!(stale.model_registry().is_err());
    assert!(stale.get_model().is_err());
    assert!(stale.get_scoped_models().is_err());
    assert!(stale.get_thinking_level().is_err());
    assert!(stale.is_idle().is_err());
    assert!(stale.is_project_trusted().is_err());
    assert!(stale.get_signal().is_err());
    assert!(stale.abort().is_err());
    assert!(stale.has_pending_messages().is_err());
    assert!(stale.shutdown().is_err());
    assert!(stale.get_context_usage().is_err());
    assert!(stale.compact(None).is_err());
    assert!(stale.get_system_prompt().is_err());
    assert!(stale.get_system_prompt_options().is_err());
}

#[test]
fn flag_vocabulary_compares_and_debugs() {
    let flag = ExtensionFlag {
        name: "port".to_string(),
        description: Some("the port".to_string()),
        flag_type: FlagType::String,
        default: Some(FlagValue::Text("8080".to_string())),
        extension_path: "/ext/test.ts".to_string(),
    };
    let same = ExtensionFlag {
        flag_type: FlagType::String,
        default: Some(FlagValue::Text("8080".to_string())),
        ..ExtensionFlag {
            name: "port".to_string(),
            description: Some("the port".to_string()),
            flag_type: FlagType::String,
            default: Some(FlagValue::Bool(true)),
            extension_path: "/ext/test.ts".to_string(),
        }
    };
    assert_eq!(flag, same);
    assert_ne!(FlagType::Boolean, FlagType::String,);
    assert_eq!(FlagValue::Bool(true), FlagValue::Bool(true));
    assert_ne!(FlagValue::Text("1".to_string()), FlagValue::Bool(true));
    assert!(!format!("{flag:?}").is_empty());
    let error = ExtensionError {
        extension_path: "/ext/test.ts".to_string(),
        event: "tool_call".to_string(),
        error: "boom".to_string(),
        stack: Some("at line".to_string()),
    };
    assert_eq!(
        error,
        ExtensionError {
            extension_path: error.extension_path.clone(),
            event: error.event.clone(),
            error: error.error.clone(),
            stack: error.stack.clone(),
        }
    );
    assert!(!format!("{error:?}").is_empty());
}
