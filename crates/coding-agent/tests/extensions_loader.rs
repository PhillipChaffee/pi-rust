//! The extension-loader suites, upstream's `suite/regressions/6260`, `8423`,
//! and `7193` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use pi_coding_agent::core::event_bus::EventBus;
use pi_coding_agent::extensions::api::{
    ExtensionApi, ExtensionFactory, ExtensionRuntime, InlineExtension, load_extension_from_factory,
};
use pi_coding_agent::extensions::runtime::{FlagType, FlagValue};
use pi_coding_agent::provider_composer::ProviderConfigInput;
use pi_coding_agent::resource_loader::DefaultResourceLoader;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;
use serde_json::Value;
use tempfile::TempDir;

fn noop() -> ExtensionFactory {
    Arc::new(|_api: Arc<ExtensionApi>| Box::pin(async { Ok(()) }))
}

fn fixture() -> TempDir {
    TempDir::new().expect("tempdir")
}

async fn loader_with(
    root: &TempDir,
    extensions: Vec<InlineExtension>,
) -> DefaultResourceLoader<InMemorySettingsStorage> {
    let cwd = root.path().join("project");
    let agent_dir = root.path().join("agent");
    std::fs::create_dir_all(&cwd).expect("cwd");
    std::fs::create_dir_all(&agent_dir).expect("agent dir");
    let settings =
        SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default());
    let mut loader = DefaultResourceLoader::new(
        DefaultResourceLoaderOptions {
            cwd: cwd.display().to_string(),
            agent_dir: agent_dir.display().to_string(),
            extension_factories: extensions,
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            ..DefaultResourceLoaderOptions::default()
        },
        settings,
    );
    loader.reload(None).await;
    loader
}

fn inline(name: Option<&str>, hidden: bool, factory: ExtensionFactory) -> InlineExtension {
    InlineExtension {
        name: name.map(str::to_string),
        hidden,
        factory,
    }
}

fn extension_paths(loader: &DefaultResourceLoader<InMemorySettingsStorage>) -> Vec<String> {
    loader
        .get_extensions()
        .extensions
        .iter()
        .map(|extension| extension.path.clone())
        .collect()
}

// =============================================================================
// #6260 inline extension naming
// =============================================================================

#[tokio::test]
async fn bare_factories_label_positionally() {
    let root = fixture();
    let loader = loader_with(
        &root,
        vec![inline(None, false, noop()), inline(None, false, noop())],
    )
    .await;
    assert_eq!(extension_paths(&loader), ["<inline:1>", "<inline:2>"]);
}

#[tokio::test]
async fn named_factories_label_by_name() {
    let root = fixture();
    let loader = loader_with(
        &root,
        vec![
            inline(Some("my-provider"), false, noop()),
            inline(Some("my-commands"), false, noop()),
        ],
    )
    .await;
    assert_eq!(
        extension_paths(&loader),
        ["<inline:my-provider>", "<inline:my-commands>"]
    );
}

#[tokio::test]
async fn named_factories_carry_the_hidden_mark() {
    let root = fixture();
    let loader = loader_with(&root, vec![inline(Some("built-in"), true, noop())]).await;
    let result = loader.get_extensions();
    assert_eq!(result.extensions.len(), 1);
    assert_eq!(result.extensions[0].path, "<inline:built-in>");
    assert!(result.extensions[0].hidden);
}

#[tokio::test]
async fn mixed_bare_and_named_factories_count_positionally() {
    let root = fixture();
    let loader = loader_with(
        &root,
        vec![
            inline(None, false, noop()),
            inline(Some("named-ext"), false, noop()),
            inline(None, false, noop()),
        ],
    )
    .await;
    assert_eq!(
        extension_paths(&loader),
        ["<inline:1>", "<inline:named-ext>", "<inline:3>"]
    );
}

// =============================================================================
// #8423 extension factory failure
// =============================================================================

fn working_provider_factory() -> ExtensionFactory {
    Arc::new(|api: Arc<ExtensionApi>| {
        Box::pin(async move {
            api.register_provider(
                "working-provider".to_string(),
                ProviderConfigInput::default(),
            )
        })
    })
}

async fn settle_bus() {
    // The bus spawns each handler future, so an emit returns before the
    // listeners run; upstream's `setImmediate` wait restates as a sleep
    // that outlives the scheduler hop.
    tokio::time::sleep(Duration::from_millis(10)).await;
}

fn queued_names(runtime: &ExtensionRuntime) -> Vec<String> {
    runtime
        .take_pending_provider_registrations()
        .iter()
        .map(|registration| registration.name.clone())
        .collect()
}

#[tokio::test]
async fn a_failed_factory_discards_its_runtime_changes_and_disables_its_api() {
    let runtime = ExtensionRuntime::new();
    let bus = EventBus::default();
    load_extension_from_factory(
        &working_provider_factory(),
        &runtime,
        &bus,
        "/tmp/project",
        "<working>",
    )
    .await
    .expect("the working factory loads");

    let captured: Arc<Mutex<Option<Arc<ExtensionApi>>>> = Arc::new(Mutex::new(None));
    let event_calls = Arc::new(AtomicUsize::new(0));
    let flag_during_load: Arc<Mutex<Option<FlagValue>>> = Arc::new(Mutex::new(None));
    let failing: ExtensionFactory = {
        let captured = Arc::clone(&captured);
        let event_calls = Arc::clone(&event_calls);
        let flag_during_load = Arc::clone(&flag_during_load);
        Arc::new(move |api: Arc<ExtensionApi>| {
            let captured = Arc::clone(&captured);
            let event_calls = Arc::clone(&event_calls);
            let flag_during_load = Arc::clone(&flag_during_load);
            Box::pin(async move {
                *captured
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&api));
                api.events_on(
                    "factory-failure",
                    Arc::new(move |_data: Value| {
                        let event_calls = Arc::clone(&event_calls);
                        Box::pin(async move {
                            event_calls.fetch_add(1, Ordering::SeqCst);
                        })
                    }),
                )?;
                api.register_flag(
                    "failed-flag".to_string(),
                    None,
                    FlagType::Boolean,
                    Some(FlagValue::Bool(true)),
                )?;
                *flag_during_load
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    api.get_flag("failed-flag")?;
                api.unregister_provider("working-provider")?;
                api.register_provider(
                    "failed-provider".to_string(),
                    ProviderConfigInput::default(),
                )?;
                Err("factory failed".to_string())
            })
        })
    };
    let error = load_extension_from_factory(&failing, &runtime, &bus, "/tmp/project", "<failing>")
        .await
        .expect_err("the failing factory fails the load");
    assert_eq!(error, "factory failed");

    bus.emit("factory-failure", &Value::Null);
    settle_bus().await;
    assert_eq!(
        flag_during_load
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone(),
        Some(FlagValue::Bool(true)),
        "the loading API reads its own pending seed"
    );
    assert!(
        runtime.flag_value("failed-flag").is_none(),
        "the failed seed never reaches the runtime"
    );
    assert_eq!(
        queued_names(&runtime),
        ["working-provider"],
        "the failing factory's registration and removal both drop"
    );
    assert_eq!(event_calls.load(Ordering::SeqCst), 0, "the listener drops");
    let captured = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("the factory captured its API");
    assert_eq!(
        captured
            .register_flag(
                "late-flag".to_string(),
                None,
                FlagType::Boolean,
                Some(FlagValue::Bool(true)),
            )
            .expect_err("the failed API rejects later calls"),
        "Extension \"<failing>\" failed to load and its API is no longer active."
    );
}

#[tokio::test]
async fn a_failing_load_keeps_a_concurrent_loads_provider() {
    let runtime = ExtensionRuntime::new();
    let bus = EventBus::default();
    let (started_send, started) = tokio::sync::oneshot::channel::<()>();
    let (release, wait) = tokio::sync::oneshot::channel::<()>();
    let wait = Arc::new(Mutex::new(Some(wait)));
    let started_send = Arc::new(Mutex::new(Some(started_send)));
    let failing: ExtensionFactory = {
        let wait = Arc::clone(&wait);
        let started_send = Arc::clone(&started_send);
        Arc::new(move |api: Arc<ExtensionApi>| {
            let wait = Arc::clone(&wait);
            let started_send = Arc::clone(&started_send);
            Box::pin(async move {
                api.register_provider(
                    "failed-provider".to_string(),
                    ProviderConfigInput::default(),
                )?;
                let send = started_send
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                if let Some(send) = send {
                    let _ = send.send(());
                }
                let wait = wait
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .expect("the gate is taken once");
                let _ = wait.await;
                Err("factory failed".to_string())
            })
        })
    };
    let failing_load = tokio::spawn({
        let runtime = runtime.clone();
        let bus = bus.clone();
        async move {
            load_extension_from_factory(&failing, &runtime, &bus, "/tmp/project", "<failing>").await
        }
    });
    started.await.expect("the failing factory started");
    let working = load_extension_from_factory(
        &working_provider_factory(),
        &runtime,
        &bus,
        "/tmp/project",
        "<working>",
    )
    .await;
    working.expect("the concurrent factory loads");
    release.send(()).expect("the gate is live");
    let error = failing_load
        .await
        .expect("the failing load task completes")
        .expect_err("the failing factory fails the load");
    assert_eq!(error, "factory failed");
    assert_eq!(queued_names(&runtime), ["working-provider"]);
}

async fn emit_counts(
    bus: &EventBus,
    extension: &AtomicUsize,
    host: &AtomicUsize,
) -> (usize, usize) {
    let extension_before = extension.load(Ordering::SeqCst);
    let host_before = host.load(Ordering::SeqCst);
    bus.emit("reload:test", &Value::Null);
    settle_bus().await;
    (
        extension.load(Ordering::SeqCst) - extension_before,
        host.load(Ordering::SeqCst) - host_before,
    )
}

// =============================================================================
// #7193 extension event-bus lifecycle
// =============================================================================

#[tokio::test]
async fn reload_and_dispose_drop_extension_owned_bus_listeners() {
    let root = fixture();
    let bus = EventBus::default();
    let extension_calls = Arc::new(AtomicUsize::new(0));
    let host_calls = Arc::new(AtomicUsize::new(0));
    let first_api: Arc<Mutex<Option<Arc<ExtensionApi>>>> = Arc::new(Mutex::new(None));
    let factory: ExtensionFactory = {
        let extension_calls = Arc::clone(&extension_calls);
        let first_api = Arc::clone(&first_api);
        Arc::new(move |api: Arc<ExtensionApi>| {
            let extension_calls = Arc::clone(&extension_calls);
            let first_api = Arc::clone(&first_api);
            Box::pin(async move {
                first_api
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_or_insert_with(|| Arc::clone(&api));
                api.events_on(
                    "reload:test",
                    Arc::new(move |_data: Value| {
                        let extension_calls = Arc::clone(&extension_calls);
                        Box::pin(async move {
                            extension_calls.fetch_add(1, Ordering::SeqCst);
                        })
                    }),
                )?;
                Ok(())
            })
        })
    };
    let _host_subscription = bus.on(
        "reload:test",
        Arc::new({
            let host_calls = Arc::clone(&host_calls);
            move |_data: Value| {
                let host_calls = Arc::clone(&host_calls);
                Box::pin(async move {
                    host_calls.fetch_add(1, Ordering::SeqCst);
                })
            }
        }),
    );

    let cwd = root.path().to_str().expect("utf8");
    let manager = SessionManager::in_memory(Some(cwd), None, None).expect("session");
    let mut setup = common::agent_session::SessionSetup::new(cwd, manager);
    setup.extensions = vec![inline(None, false, factory)];
    setup.event_bus = Some(bus.clone());
    setup.bind_extensions = true;
    let session = common::agent_session::build(setup).await;

    assert_eq!(
        emit_counts(&bus, &extension_calls, &host_calls).await,
        (1, 1)
    );

    session.reload(None).await;
    let stale = first_api
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("the factory captured its API");
    assert!(
        stale
            .get_flag("any")
            .expect_err("the replaced API reads stale")
            .contains("stale after session replacement or reload")
    );
    assert_eq!(
        emit_counts(&bus, &extension_calls, &host_calls).await,
        (1, 1)
    );

    session.reload(None).await;
    assert_eq!(
        emit_counts(&bus, &extension_calls, &host_calls).await,
        (1, 1)
    );

    session.dispose();
    assert_eq!(
        emit_counts(&bus, &extension_calls, &host_calls).await,
        (0, 1)
    );
}
