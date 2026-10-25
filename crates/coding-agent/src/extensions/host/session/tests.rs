#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::{Arc, Mutex};

use pi_ai::auth::credential_store::InMemoryCredentialStore;
use tokio::io::DuplexStream;
use tokio::task::JoinHandle;

use crate::extensions::reasons::ExtensionMode;
use crate::extensions::runtime::{
    ContextActions, ExtensionError, ExtensionHandler, HandlerContext,
};
use crate::model_registry::ModelRegistry;
use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::session_manager::SessionManager;
use crate::source_info::{SourceInfo, SourceOrigin, SourceScope, SyntheticSourceOptions};

use super::super::context::UiCallHandler;
use super::super::link::Link;
use super::super::{EXTENSION_PROTOCOL_VERSION, ExtensionIdentity, Frame, WireError};
use super::{HostConnectOptions, HostSession};

#[path = "tests/command/tests.rs"]
mod command;
#[path = "tests/context/tests.rs"]
mod context;
#[path = "tests/crash/tests.rs"]
mod crash;
#[path = "tests/event/tests.rs"]
mod event;
#[path = "tests/protocol/tests.rs"]
mod protocol;
#[path = "tests/provider/tests.rs"]
mod provider;
#[path = "tests/signal/tests.rs"]
mod signal;
#[path = "tests/support/tests.rs"]
mod support;
#[path = "tests/tool/tests.rs"]
mod tool;
#[path = "tests/ui/tests.rs"]
mod ui;

fn source_info() -> SourceInfo {
    crate::source_info::create_synthetic_source_info(
        "/ext/demo",
        &SyntheticSourceOptions {
            source: "extension:demo".to_string(),
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

async fn test_context(actions: ContextActions) -> HandlerContext {
    HandlerContext::new(
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        plain_manager(),
        offline_registry().await,
        Arc::new(|| Ok(())),
        Arc::new(actions),
    )
}

fn hello() -> Frame {
    Frame::Hello {
        protocol_version: EXTENSION_PROTOCOL_VERSION,
        extension: ExtensionIdentity {
            name: "demo".to_string(),
            version: "0.1.0".to_string(),
        },
    }
}

async fn open_session(
    actions: ContextActions,
    ui: Option<Arc<dyn UiCallHandler>>,
) -> (
    JoinHandle<Result<HostSession<DuplexStream>, WireError>>,
    Link<DuplexStream>,
    Arc<Mutex<Vec<ExtensionError>>>,
) {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let listener = {
        let errors = Arc::clone(&errors);
        Arc::new(move |error: &ExtensionError| {
            errors
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(error.clone());
        })
    };
    let mut options = support::connect_options(actions, ui).await;
    options.error_listener = Some(listener);
    let (host_io, child_io) = tokio::io::duplex(1 << 20);
    let task = tokio::spawn(async move { HostSession::connect(host_io, options).await });
    (task, Link::new(child_io).expect("child link"), errors)
}

async fn ready_session_with(
    registrations: Vec<Frame>,
    actions: ContextActions,
    ui: Option<Arc<dyn UiCallHandler>>,
) -> (
    HostSession<DuplexStream>,
    Link<DuplexStream>,
    Arc<Mutex<Vec<ExtensionError>>>,
) {
    let (task, mut child, errors) = open_session(actions, ui).await;
    child.write_frame(&hello()).await.expect("hello");
    assert!(matches!(
        child.read_frame().await.expect("welcome"),
        Some(Frame::Welcome { .. })
    ));
    for frame in registrations {
        child.write_frame(&frame).await.expect("registration");
    }
    child.write_frame(&Frame::Ready).await.expect("ready");
    let session = task.await.expect("join").expect("connect");
    (session, child, errors)
}

fn handler(session: &HostSession<DuplexStream>, channel: &str) -> ExtensionHandler {
    session
        .extension()
        .handlers
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(channel)
        .expect("the channel is registered")
        .first()
        .expect("the channel has a handler")
        .clone()
}

#[tokio::test]
async fn handshake_and_registration_bind_every_declaration() {
    let (mut session, _child, _errors) = ready_session_with(
        vec![
            Frame::RegisterEvent {
                channel: "agent_start".to_string(),
            },
            Frame::RegisterTool {
                tool: serde_json::json!({
                    "name": "demo_tool",
                    "label": "Demo",
                    "description": "Runs the demo",
                    "parameters": {"type": "object"},
                }),
            },
            Frame::RegisterCommand {
                command: serde_json::json!({"name": "demo", "description": "Run it"}),
            },
            Frame::RegisterFlag {
                flag: serde_json::json!({"name": "demo-flag", "type": "boolean", "default": true}),
            },
            Frame::RegisterProvider {
                provider: serde_json::json!({
                    "name": "demo-provider",
                    "config": {"name": "Demo", "api": "openai-completions"},
                }),
            },
        ],
        ContextActions::default(),
        None,
    )
    .await;
    assert_eq!(session.identity().name, "demo");
    let _ = format!("{session:?}");
    let extension = session.extension();
    assert!(
        extension
            .handlers
            .lock()
            .expect("handlers")
            .contains_key("agent_start")
    );
    assert_eq!(extension.tools.lock().expect("tools").len(), 1);
    assert_eq!(extension.commands.lock().expect("commands").len(), 1);
    assert!(extension.has_flag("demo-flag"));
    let providers = session.take_provider_registrations();
    assert_eq!(providers.len(), 1);
    assert_eq!(providers[0].name, "demo-provider");
    assert!(providers[0].config.stream_simple.is_some());
    assert!(session.take_provider_registrations().is_empty());
}
