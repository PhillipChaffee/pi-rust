#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentToolError, AgentToolResult};

use super::{built_in_extensions, extension_factories};
use crate::extensions::api::{
    ExtensionApi, ExtensionFactory, ExtensionRuntime, InlineExtension, load_extension_from_factory,
};
use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::{ExtensionMode, SessionStartReason};
use crate::extensions::results::HandlerResult;
use crate::extensions::runner::ExtensionRunner;
use crate::extensions::runtime::{
    CommandHandler, ContextActions, ExtensionHandler, FlagType, FlagValue, HandlerContext,
};
use crate::extensions::types::ToolDefinition;
use crate::model_registry::ModelRegistry;
use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::session_manager::SessionManager;
use pi_ai::auth::credential_store::InMemoryCredentialStore;
use pi_ai::types::BoxedFuture;

type EventLog = Arc<Mutex<Vec<&'static str>>>;

fn noop_handler(log: &EventLog) -> ExtensionHandler {
    let log = Arc::clone(log);
    Arc::new(move |_event: ExtensionEvent, _ctx: HandlerContext| {
        let log = Arc::clone(&log);
        Box::pin(async move {
            log.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push("first-party");
            HandlerResult::None
        })
    })
}

fn noop_command() -> CommandHandler {
    Arc::new(|_args: String, _ctx: HandlerContext| Box::pin(async {}))
}

fn simple_tool() -> ToolDefinition {
    ToolDefinition {
        name: "party_tool".to_string(),
        label: "party_tool".to_string(),
        description: "test".to_string(),
        prompt_snippet: None,
        prompt_guidelines: None,
        parameters: serde_json::json!({ "type": "object" }),
        constrained_sampling: None,
        render_shell: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(
            |_call: &str,
             _params: &serde_json::Value,
             _signal: Option<&pi_agent_core::harness::context::AbortSignal>,
             _update: Option<pi_agent_core::harness::types::AgentHarnessToolUpdateCallback>,
             _context: Option<&dyn crate::extensions::ExtensionContext>| {
                Box::pin(async {
                    Ok::<AgentToolResult, AgentToolError>(AgentToolResult {
                        content: Vec::new(),
                        details: serde_json::json!({}),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    })
                })
            },
        ),
    }
}

/// The first-party factory the host list will carry once llama.cpp
/// compiles in: named, hidden, registering through `ExtensionApi` and
/// reading the seeded flag back with `getFlag`.
fn first_party_factory(log: &EventLog) -> ExtensionFactory {
    let log = Arc::clone(log);
    let factory = move |api: Arc<ExtensionApi>| -> BoxedFuture<'static, Result<(), String>> {
        let log = Arc::clone(&log);
        Box::pin(async move {
            api.on("session_start", noop_handler(&log))?;
            api.register_tool(simple_tool())?;
            api.register_command(
                "party".to_string(),
                Some("First-party".to_string()),
                noop_command(),
            )?;
            api.register_flag(
                "party_flag".to_string(),
                None,
                FlagType::Boolean,
                Some(FlagValue::Bool(true)),
            )?;
            assert_eq!(
                api.get_flag("party_flag")?,
                Some(FlagValue::Bool(true)),
                "the factory reads its own seeded flag"
            );
            Ok(())
        })
    };
    Arc::new(factory)
}

fn named_factory(name: &str, hidden: bool, factory: ExtensionFactory) -> InlineExtension {
    InlineExtension {
        name: Some(name.to_string()),
        hidden,
        factory,
    }
}

#[test]
fn caller_factories_pass_through_in_host_order() {
    let caller = vec![
        named_factory(
            "demo",
            true,
            Arc::new(|_api: Arc<ExtensionApi>| Box::pin(async { Ok(()) })),
        ),
        InlineExtension {
            name: None,
            hidden: false,
            factory: Arc::new(|_api: Arc<ExtensionApi>| Box::pin(async { Ok(()) })),
        },
    ];
    let merged = extension_factories(caller);
    assert_eq!(merged.len(), 2, "the empty built-in list adds nothing");
    assert_eq!(merged[0].name.as_deref(), Some("demo"));
    assert!(merged[1].name.is_none(), "the caller order is preserved");
    assert!(extension_factories(Vec::new()).is_empty());
}

#[test]
fn built_in_list_stays_empty_until_llama_compiles() {
    assert!(
        built_in_extensions().is_empty(),
        "upstream's single entry is the llama.cpp extension; its provider \
         registration rides the native-provider ticket and its /llama \
         command rides #132, so compiling it in lands later"
    );
}

fn plain_actions() -> Arc<ContextActions> {
    Arc::new(ContextActions::default())
}

fn plain_manager() -> SessionManager {
    SessionManager::new("/tmp/project", "/tmp/sessions", None, false, None, None)
        .expect("the in-memory manager constructs")
}

async fn offline_registry() -> ModelRegistry {
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

#[tokio::test]
async fn first_party_extension_registers_and_consumes_the_runner_surface() {
    let log = Arc::new(Mutex::new(Vec::new()));

    // Loader parity: a named+hidden factory loads exactly the way llama.cpp
    // will ride `built_in_extensions()`; the bare caller entry lands behind
    // it on the positional `<inline:N>` label.
    let merged = extension_factories(vec![
        named_factory("first-party", true, first_party_factory(&log)),
        InlineExtension {
            name: None,
            hidden: false,
            factory: Arc::new(|_api: Arc<ExtensionApi>| Box::pin(async { Ok(()) })),
        },
    ]);

    let runtime = ExtensionRuntime::new();
    let mut records = Vec::new();
    for (index, input) in merged.iter().enumerate() {
        let label = input.name.as_ref().map_or_else(
            || format!("<inline:{}>", index + 1),
            |name| format!("<inline:{name}>"),
        );
        records.push(
            load_extension_from_factory(&input.factory, &runtime, "/tmp/project", &label)
                .await
                .expect("the factory loads"),
        );
    }
    assert_eq!(records[0].path, "<inline:first-party>");
    assert_eq!(
        records[1].path, "<inline:2>",
        "the bare caller factory takes the positional label after the first-party entry"
    );

    let runner = ExtensionRunner::new(
        records,
        plain_actions(),
        Arc::new(|| Ok(())),
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        plain_manager(),
        offline_registry().await,
        runtime,
    );

    assert!(runner.has_handlers("session_start"));
    let tools = runner.get_all_registered_tools();
    assert_eq!(tools.len(), 1, "the first-party tool registers");
    assert_eq!(tools[0].definition.name, "party_tool");
    let command = runner
        .get_command("party")
        .expect("the first-party command registers");
    assert_eq!(command.description.as_deref(), Some("First-party"));
    let flags = runner.get_flags();
    assert_eq!(flags.len(), 1);
    assert_eq!(flags[0].name, "party_flag");
    assert_eq!(flags[0].flag_type, FlagType::Boolean);
    assert_eq!(
        runner.get_flag_values().get("party_flag"),
        Some(&FlagValue::Bool(true)),
        "the seeded default reads through the consumption surface"
    );

    runner.set_flag_value("party_flag", FlagValue::Bool(false));
    assert_eq!(
        runner.get_flag_values().get("party_flag"),
        Some(&FlagValue::Bool(false)),
        "the value roundtrips through the runtime"
    );

    runner
        .emit(ExtensionEvent::SessionStart {
            reason: SessionStartReason::Startup,
            previous_session_file: None,
        })
        .await;
    assert_eq!(
        *log.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec!["first-party"],
        "the compile-in handler dispatches"
    );
}
