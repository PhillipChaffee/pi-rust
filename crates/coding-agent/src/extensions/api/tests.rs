#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::{AgentToolError, AgentToolResult};

use super::{ExtensionApi, ExtensionFactory, ExtensionRuntime};
use crate::core::event_bus::EventBus;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::results::HandlerResult;
use crate::extensions::runtime::{CommandHandler, ExtensionHandler, FlagType, FlagValue};
use crate::extensions::types::ToolDefinition;
use crate::provider_composer::ProviderConfigInput;
use crate::source_info::SourceInfo;

fn source_info(path: &str) -> SourceInfo {
    crate::source_info::create_synthetic_source_info(
        path,
        &crate::source_info::SyntheticSourceOptions::default(),
    )
}

fn api_with(extension_path: &str, runtime: &ExtensionRuntime) -> Arc<ExtensionApi> {
    let extension = crate::extensions::runtime::Extension::new(
        extension_path.to_string(),
        extension_path.to_string(),
        false,
        source_info(extension_path),
    );
    Arc::new(ExtensionApi::new(
        Arc::new(extension),
        runtime.clone(),
        EventBus::default(),
        "/tmp/project".to_string(),
    ))
}

fn noop_handler() -> ExtensionHandler {
    Arc::new(|_event: ExtensionEvent, _ctx| Box::pin(async { HandlerResult::None }))
}

fn noop_command() -> CommandHandler {
    Arc::new(|_args: String, _ctx| Box::pin(async {}))
}

fn simple_tool(name: &str, parameters: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        label: name.to_string(),
        description: "test".to_string(),
        prompt_snippet: None,
        prompt_guidelines: None,
        parameters,
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
#[test]
fn flag_and_tool_registrations_write_the_record() {
    let runtime = ExtensionRuntime::new();
    let api = api_with("/ext/test.ts", &runtime);

    api.register_tool(simple_tool("find", serde_json::json!({ "type": "object" })))
        .expect("the tool registers");
    api.on("session_start", noop_handler())
        .expect("the handler registers");
    api.register_command(
        "ship".to_string(),
        Some("ships it".to_string()),
        noop_command(),
    )
    .expect("the command registers");
    api.register_flag(
        "port".to_string(),
        Some("the port".to_string()),
        FlagType::String,
        Some(FlagValue::Text("8080".to_string())),
    )
    .expect("the flag registers");

    // Pre-commit, the flag reads as the pending seed; the runtime is empty.
    assert_eq!(
        api.get_flag("port").expect("flag read"),
        Some(FlagValue::Text("8080".to_string()))
    );
    assert!(runtime.flag_value("port").is_none());
    let record = api.extension();
    assert_eq!(
        record
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        1
    );
    assert_eq!(
        record
            .commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        1
    );
    assert!(record.has_flag("port"));

    api.commit().expect("the commit flushes");
    // The flag seed moved to the runtime.
    assert_eq!(
        runtime.flag_value("port"),
        Some(FlagValue::Text("8080".to_string()))
    );
    // getFlag keeps reading after the commit.
    assert_eq!(
        api.get_flag("port").expect("flag read"),
        Some(FlagValue::Text("8080".to_string()))
    );
    // An unregistered name reads as none.
    assert_eq!(api.get_flag("unregistered").expect("flag read"), None);
}

#[test]
fn flag_default_type_mismatch_is_rejected() {
    let runtime = ExtensionRuntime::new();
    let api = api_with("/ext/test.ts", &runtime);
    let error = api
        .register_flag(
            "verbose".to_string(),
            None,
            FlagType::Boolean,
            Some(FlagValue::Text("yes".to_string())),
        )
        .expect_err("the mismatch rejects");
    assert_eq!(
        error,
        "Invalid default for flag \"verbose\": expected boolean, got string"
    );
}

#[test]
fn non_object_tool_schema_is_rejected() {
    let runtime = ExtensionRuntime::new();
    let api = api_with("/ext/test.ts", &runtime);
    let error = api
        .register_tool(simple_tool("broken", serde_json::json!(["array"])))
        .expect_err("the schema rejects");
    assert_eq!(
        error,
        "Tool \"broken\" registered by extension \"/ext/test.ts\" must define an object \
         parameter schema."
    );
}

#[test]
fn failed_load_discards_the_api() {
    let runtime = ExtensionRuntime::new();
    let api = api_with("/ext/test.ts", &runtime);
    api.discard();

    let error = api
        .on("session_start", noop_handler())
        .expect_err("the failed API rejects");
    assert_eq!(
        error,
        "Extension \"/ext/test.ts\" failed to load and its API is no longer active."
    );
    // The runtime staleness rides beneath the failed-API state.
    runtime.invalidate(None);
    assert!(runtime.assert_active().is_err());
    // Post-failure lifecycle calls are no-ops (upstream's state guard).
    api.commit().expect("the commit is a no-op");
    api.discard();

    let inline = super::InlineExtension {
        name: Some("demo".to_string()),
        hidden: true,
        factory: api_factory_noop(),
    };
    assert!(format!("{inline:?}").contains("demo"));
    assert!(format!("{runtime:?}").contains("stale: true"));
}

fn api_factory_noop() -> ExtensionFactory {
    Arc::new(|_api: Arc<ExtensionApi>| Box::pin(async { Ok(()) }))
}

#[tokio::test]
async fn registered_callables_execute_through_the_record() {
    let runtime = ExtensionRuntime::new();
    let api = api_with("/ext/dispatch.ts", &runtime);
    assert_eq!(api.path(), "/ext/dispatch.ts");
    assert_eq!(api.cwd(), "/tmp/project");
    api.on("session_start", noop_handler())
        .expect("the handler registers");
    api.register_command("demo".to_string(), None, noop_command())
        .expect("the command registers");
    api.register_tool(simple_tool(
        "probe",
        serde_json::json!({ "type": "object" }),
    ))
    .expect("the tool registers");
    let record = api.extension();

    let handler = record
        .handlers
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)["session_start"][0]
        .clone();
    let handler_outcome = (handler)(ExtensionEvent::AgentStart, dispatch_context().await).await;
    assert!(matches!(handler_outcome, HandlerResult::None));

    let command = record
        .commands
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)[0]
        .clone();
    (command.handler)("args".to_string(), dispatch_context().await).await;

    let tool = record
        .tools
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)[0]
        .clone();
    let executed = (tool.definition.execute)(
        "call-1",
        &serde_json::json!({}),
        None,
        None,
        Some(&crate::extensions::types::CwdContext {
            cwd: "/tmp".to_string(),
        }),
    )
    .await;
    assert!(executed.is_ok(), "the tool executes through its record");

    // The factory runs through its load pass lookalike.
    let factory = api_factory_noop();
    let fresh = api_with("/ext/fresh.ts", &runtime);
    (factory)(Arc::clone(&fresh))
        .await
        .expect("the factory runs");
}

#[test]
fn stale_runtime_marks_every_api_inactive() {
    let runtime = ExtensionRuntime::new();
    let api = api_with("/ext/test.ts", &runtime);
    runtime.invalidate(None);

    let error = api
        .on("session_start", noop_handler())
        .expect_err("the stale runtime rejects");
    assert!(error.contains("stale after session replacement"), "{error}");
    // The first message wins.
    runtime.invalidate(Some("custom".to_string()));
    let error = api
        .on("session_start", noop_handler())
        .expect_err("still stale");
    assert!(!error.contains("custom"));
}

async fn dispatch_context() -> crate::extensions::runtime::HandlerContext {
    use crate::extensions::runtime::{ContextActions, HandlerContext};
    use crate::model_registry::ModelRegistry;
    use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
    use crate::session_manager::SessionManager;
    use pi_agent_core::types::ThinkingLevel;
    use pi_ai::auth::credential_store::InMemoryCredentialStore;
    let manager = SessionManager::new("/tmp/project", "/tmp/sessions", None, false, None, None)
        .expect("the in-memory manager constructs");
    let registry_runtime = ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(Arc::new(InMemoryCredentialStore::default())),
        models_path: Some(None),
        allow_model_network: false,
        refresh_on_create: Some(false),
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .expect("the runtime constructs offline");
    HandlerContext::new(
        "/tmp/project".to_string(),
        crate::extensions::reasons::ExtensionMode::Print,
        None,
        manager,
        ModelRegistry::new(registry_runtime),
        Arc::new(|| Ok(())),
        Arc::new(ContextActions {
            get_thinking_level: Some(Arc::new(|| ThinkingLevel::Off)),
            ..ContextActions::default()
        }),
    )
}

#[test]
fn provider_queue_remembers_and_forgets_registrations() {
    let runtime = ExtensionRuntime::new();
    let api = api_with("/ext/test.ts", &runtime);
    api.register_provider("custom".to_string(), ProviderConfigInput::default())
        .expect("queued");
    api.register_provider("other".to_string(), ProviderConfigInput::default())
        .expect("queued");
    api.unregister_provider("custom").expect("dropped");
    assert!(
        runtime.take_pending_provider_registrations().is_empty(),
        "a loading API keeps its changes private"
    );
    api.commit().expect("the API commits");
    let queued = runtime.take_pending_provider_registrations();
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0].name, "other");
    assert_eq!(queued[0].extension_path, "/ext/test.ts");
    assert!(runtime.take_pending_provider_registrations().is_empty());

    // A second registration survives the earlier re-drain.
    api.register_provider("custom".to_string(), ProviderConfigInput::default())
        .expect("queued");
    assert_eq!(runtime.take_pending_provider_registrations().len(), 1);
    assert!(format!("{api:?}").contains("/ext/test.ts"));
}
#[tokio::test]
async fn load_extension_from_factory_runs_commit_and_lifecycle() {
    let runtime = ExtensionRuntime::new();
    let factory: ExtensionFactory = Arc::new(|api: Arc<ExtensionApi>| {
        Box::pin(async move {
            assert_eq!(api.path(), "<inline:demo>");
            api.register_flag(
                "demos".to_string(),
                None,
                FlagType::Boolean,
                Some(FlagValue::Bool(true)),
            )?;
            api.register_tool(simple_tool(
                "custom",
                serde_json::json!({ "type": "object", "properties": {} }),
            ))?;
            api.on("session_start", noop_handler())?;
            api.register_command("demo".to_string(), None, noop_command())?;
            Ok(())
        })
    });
    let extension = crate::extensions::api::load_extension_from_factory(
        &factory,
        &runtime,
        &EventBus::default(),
        "/tmp/project",
        "<inline:demo>",
    )
    .await
    .expect("the factory runs");
    assert!(
        !extension.hidden(),
        "the loader decides hidden, not initialize"
    );
    assert_eq!(extension.path, "<inline:demo>");
    assert!(
        extension
            .handlers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key("session_start")
    );
    assert_eq!(runtime.flag_value("demos"), Some(FlagValue::Bool(true)));

    // The hidden mark drives the inline loader, the factory does not.
    extension.set_hidden(false);
    assert!(!extension.hidden());
}

#[tokio::test]
async fn failing_factory_fails_the_load_and_discards_the_api() {
    let runtime = ExtensionRuntime::new();
    let factory: ExtensionFactory = Arc::new(|_api: Arc<ExtensionApi>| {
        Box::pin(async { Err("the factory refused".to_string()) })
    });
    let error = crate::extensions::api::load_extension_from_factory(
        &factory,
        &runtime,
        &EventBus::default(),
        "/tmp/project",
        "<inline:1>",
    )
    .await
    .expect_err("the load fails");
    assert_eq!(error, "the factory refused");
}

#[tokio::test]
async fn uninitialized_commit_paths_an_error_through_the_loader() {
    let runtime = ExtensionRuntime::new();
    runtime.invalidate(Some("stale".to_string()));
    let factory: ExtensionFactory = Arc::new(|_api: Arc<ExtensionApi>| Box::pin(async { Ok(()) }));
    let error = crate::extensions::api::load_extension_from_factory(
        &factory,
        &runtime,
        &EventBus::default(),
        "/tmp/project",
        "<inline:2>",
    )
    .await
    .expect_err("the stale runtime fails the commit");
    assert!(error.contains("stale"), "{error}");
}

#[tokio::test]
async fn post_bind_provider_registration_registers_directly() {
    let runtime = ExtensionRuntime::new();
    let extension = crate::extensions::runtime::Extension::new(
        "/ext/post-bind.ts".to_string(),
        "/ext/post-bind.ts".to_string(),
        false,
        source_info("/ext/post-bind.ts"),
    );

    // Pre-bind the registration queues privately until the API commits.
    let api = Arc::new(ExtensionApi::new(
        {
            let extension = Arc::new(extension);
            Arc::clone(&extension)
        },
        runtime.clone(),
        EventBus::default(),
        "/tmp/project".to_string(),
    ));
    api.register_provider("queued".to_string(), ProviderConfigInput::default())
        .expect("the queue accepts");
    assert!(
        runtime.take_pending_provider_registrations().is_empty(),
        "the loading API keeps its registration private"
    );
    api.commit().expect("the API commits");
    assert_eq!(
        runtime.take_pending_provider_registrations().len(),
        1,
        "the pre-bind path queues"
    );

    // Post-bind the registrations route through the armed actions and
    // return their failures; the queue stays drained.
    runtime.bind_provider_actions(Some(Arc::new(crate::extensions::runner::ProviderActions {
        register_provider: Arc::new(|name: String, _config: ProviderConfigInput| {
            Err(format!("cannot register {name}"))
        }),
        unregister_provider: Arc::new(|_name: String| Ok(())),
    })));
    assert_eq!(
        api.register_provider("bound".to_string(), ProviderConfigInput::default())
            .expect_err("the post-bind failure propagates"),
        "cannot register bound".to_string()
    );
    assert!(runtime.take_pending_provider_registrations().is_empty());
    assert_eq!(
        api.unregister_provider("queued")
            .expect("the post-bind drop"),
        ()
    );

    // Unarming re-queues: the reset returns the immediate path.
    runtime.bind_provider_actions(None);
    api.register_provider("after-reset".to_string(), ProviderConfigInput::default())
        .expect("the reset path queues");
    assert_eq!(runtime.take_pending_provider_registrations().len(), 1);
    assert!(
        format!("{runtime:?}").contains("bound_provider_actions: false"),
        "{}",
        format!("{runtime:?}")
    );
}

#[test]
fn flag_values_snapshot_reads_alphabetically() {
    let runtime = ExtensionRuntime::new();
    runtime.set_flag_value("zeta", FlagValue::Bool(true));
    runtime.set_flag_value("alpha", FlagValue::Text("1".to_string()));
    let values = runtime.flag_values();
    assert_eq!(
        values.keys().cloned().collect::<Vec<_>>(),
        vec!["alpha".to_string(), "zeta".to_string()]
    );
}

#[tokio::test]
async fn events_surface_emits_and_tracks_subscriptions() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    async fn settle() {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    let runtime = ExtensionRuntime::new();
    let api = api_with("/ext/events.ts", &runtime);
    let seen = Arc::new(AtomicUsize::new(0));
    let handler: crate::core::event_bus::EventHandler = Arc::new({
        let seen = Arc::clone(&seen);
        move |_data: serde_json::Value| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                seen.fetch_add(1, Ordering::SeqCst);
            })
        }
    });
    let unsubscribe = api.events_on("demo", handler).expect("subscribed");
    assert_eq!(runtime.event_bus_subscription_count(), 1);
    api.events_emit("demo", &serde_json::json!({ "ok": true }))
        .expect("emitted");
    settle().await;
    assert_eq!(seen.load(Ordering::SeqCst), 1);

    // A committed API tracks too; the runtime only drops what was asked.
    api.commit().expect("the API commits");
    let late = Arc::new(AtomicUsize::new(0));
    let late_unsubscribe = api
        .events_on(
            "demo",
            Arc::new({
                let late = Arc::clone(&late);
                move |_data: serde_json::Value| {
                    let late = Arc::clone(&late);
                    Box::pin(async move {
                        late.fetch_add(1, Ordering::SeqCst);
                    })
                }
            }),
        )
        .expect("subscribed after commit");
    assert_eq!(runtime.event_bus_subscription_count(), 2);
    api.events_emit("demo", &serde_json::Value::Null)
        .expect("emitted");
    settle().await;
    assert_eq!(late.load(Ordering::SeqCst), 1);

    unsubscribe();
    unsubscribe();
    assert_eq!(
        runtime.event_bus_subscription_count(),
        1,
        "the tracked unsubscribe is idempotent"
    );
    let seen_before = seen.load(Ordering::SeqCst);
    api.events_emit("demo", &serde_json::Value::Null)
        .expect("emitted");
    settle().await;
    assert_eq!(
        seen.load(Ordering::SeqCst),
        seen_before,
        "the listener is gone"
    );
    assert_eq!(
        late.load(Ordering::SeqCst),
        2,
        "the committed listener runs"
    );
    late_unsubscribe();
    assert_eq!(runtime.event_bus_subscription_count(), 0);
}

#[tokio::test]
async fn a_failed_load_drops_its_loading_subscriptions() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let runtime = ExtensionRuntime::new();
    let bus = EventBus::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let factory: ExtensionFactory = Arc::new({
        let calls = Arc::clone(&calls);
        move |api: Arc<ExtensionApi>| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                api.events_on(
                    "demo",
                    Arc::new(move |_data: serde_json::Value| {
                        let calls = Arc::clone(&calls);
                        Box::pin(async move {
                            calls.fetch_add(1, Ordering::SeqCst);
                        })
                    }),
                )?;
                Err("the factory refused".to_string())
            })
        }
    });
    super::load_extension_from_factory(&factory, &runtime, &bus, "/tmp/project", "<inline:events>")
        .await
        .expect_err("the load fails");
    assert_eq!(runtime.event_bus_subscription_count(), 0);
    bus.emit("demo", &serde_json::Value::Null);
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn invalidate_drops_every_tracked_subscription() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let runtime = ExtensionRuntime::new();
    let bus = EventBus::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let factory: ExtensionFactory = Arc::new({
        let calls = Arc::clone(&calls);
        move |api: Arc<ExtensionApi>| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                api.events_on(
                    "demo",
                    Arc::new(move |_data: serde_json::Value| {
                        let calls = Arc::clone(&calls);
                        Box::pin(async move {
                            calls.fetch_add(1, Ordering::SeqCst);
                        })
                    }),
                )?;
                Ok(())
            })
        }
    });
    super::load_extension_from_factory(&factory, &runtime, &bus, "/tmp/project", "<inline:events>")
        .await
        .expect("the factory loads");
    assert_eq!(runtime.event_bus_subscription_count(), 1);
    runtime.invalidate(None);
    assert_eq!(runtime.event_bus_subscription_count(), 0);
    bus.emit("demo", &serde_json::Value::Null);
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

fn bound_actions(log: &Arc<Mutex<Vec<String>>>) -> crate::extensions::runner::ExtensionActions {
    use crate::extensions::runner::ExtensionActions;
    ExtensionActions {
        send_message: Arc::new({
            let log = Arc::clone(log);
            move |message, options| {
                log.lock()
                    .expect("log")
                    .push(format!("message:{}:{options:?}", message.custom_type));
            }
        }),
        send_user_message: Arc::new({
            let log = Arc::clone(log);
            move |content, options| {
                log.lock()
                    .expect("log")
                    .push(format!("user:{content:?}:{options:?}"));
            }
        }),
        append_entry: Arc::new({
            let log = Arc::clone(log);
            move |custom_type, data| {
                log.lock()
                    .expect("log")
                    .push(format!("entry:{custom_type}:{data:?}"));
            }
        }),
        set_session_name: Arc::new(|_| {}),
        get_session_name: Arc::new(|| None),
        set_label: Arc::new(|_, _| {}),
        get_active_tools: Arc::new({
            let log = Arc::clone(log);
            move || {
                log.lock()
                    .expect("log")
                    .iter()
                    .filter_map(|line| line.strip_prefix("tools:").map(str::to_owned))
                    .next_back()
                    .map(|names| names.split(',').map(str::to_owned).collect())
                    .unwrap_or_default()
            }
        }),
        get_all_tools: Arc::new(Vec::new),
        set_active_tools: Arc::new({
            let log = Arc::clone(log);
            move |names| {
                log.lock()
                    .expect("log")
                    .push(format!("tools:{}", names.join(",")));
            }
        }),
        refresh_tools: Arc::new(|| {}),
        get_commands: Arc::new(Vec::new),
        set_model: Arc::new(|_model| Box::pin(async { true })),
        get_thinking_level: Arc::new(|| pi_agent_core::types::ThinkingLevel::Off),
        set_thinking_level: Arc::new(|_| {}),
    }
}

#[test]
fn action_methods_report_the_unbound_runtime() {
    let runtime = ExtensionRuntime::new();
    let api = api_with("<inline:unbound>", &runtime);
    let error = api
        .get_active_tools()
        .expect_err("the unbound runtime refuses action calls");
    assert_eq!(
        error,
        "Extension runtime not initialized. Action methods cannot be called during extension \
         loading."
    );
}

#[tokio::test]
async fn action_methods_delegate_to_the_bound_bundle() {
    use crate::extensions::reasons::{CustomMessagePayload, text_block};
    use crate::extensions::runner::{DeliverAs, SendMessageOptions, SendUserMessageOptions};

    let runtime = ExtensionRuntime::new();
    let api = api_with("<inline:bound>", &runtime);
    let log = Arc::new(Mutex::new(Vec::new()));
    runtime.bind_actions(Arc::new(bound_actions(&log)));

    api.send_message(
        CustomMessagePayload {
            custom_type: "plan".to_string(),
            content: vec![text_block("body")],
            display: true,
            details: None,
        },
        Some(SendMessageOptions {
            trigger_turn: Some(true),
            deliver_as: Some(DeliverAs::FollowUp),
        }),
    )
    .expect("send_message");
    api.send_user_message(
        pi_ai::types::UserContent::Text("refine".to_string()),
        Some(SendUserMessageOptions {
            deliver_as: Some(DeliverAs::FollowUp),
            ..SendUserMessageOptions::default()
        }),
    )
    .expect("send_user_message");
    api.append_entry(
        "plan-mode".to_string(),
        Some(serde_json::json!({ "enabled": true })),
    )
    .expect("append_entry");
    assert!(api.get_active_tools().expect("get_active_tools").is_empty());
    api.set_active_tools(vec!["read".to_string(), "bash".to_string()])
        .expect("set_active_tools");
    assert_eq!(
        api.get_active_tools().expect("get_active_tools"),
        ["read", "bash"]
    );

    let lines = log.lock().expect("log").clone();
    assert_eq!(lines.len(), 4);
    assert!(lines[0].starts_with("message:plan:"));
    assert!(lines[1].starts_with("user:Text(\"refine\"):"));
    assert!(lines[2].starts_with("entry:plan-mode:"));
    assert_eq!(lines[3], "tools:read,bash");
}
