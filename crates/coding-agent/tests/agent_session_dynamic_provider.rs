//! `test/agent-session-dynamic-provider.test.ts` 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatement: upstream's `registerProvider(nativeProvider)` object surface
//! is deferred with the composer's native-provider ticket; the two native
//! cases exercise the same active-model refresh through a provider
//! configuration carrying its own models and stream.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;

use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::StopReason;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::runtime::CommandHandler;
use pi_coding_agent::provider_composer::ProviderConfigInput;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;

type Session = Arc<AgentSession<InMemorySettingsStorage>>;

fn in_memory_manager(root: &str) -> SessionManager {
    SessionManager::in_memory(Some(root), None, None).expect("session")
}

fn base_url_config(base_url: &str) -> ProviderConfigInput {
    ProviderConfigInput {
        base_url: Some(base_url.to_owned()),
        ..ProviderConfigInput::default()
    }
}

async fn session_with_extension(root: &str, extension: InlineExtension, bind: bool) -> Session {
    let mut setup = SessionSetup::new(root, in_memory_manager(root));
    setup.extensions = vec![extension];
    setup.bind_extensions = bind;
    let session = build(setup).await;
    authorize(&session).await;
    session
}

fn provider_extension(config: ProviderConfigInput, on_session_start: bool) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let config = config.clone();
            let start_api = Arc::clone(&api);
            Box::pin(async move {
                if on_session_start {
                    let start_config = config.clone();
                    api.on(
                        "session_start",
                        Arc::new(move |_event: ExtensionEvent, _ctx| {
                            let config = start_config.clone();
                            let api = Arc::clone(&start_api);
                            Box::pin(async move {
                                api.register_provider("anthropic".to_owned(), config)
                                    .expect("provider registers");
                                HandlerResult::None
                            })
                        }),
                    )
                    .expect("session_start handler");
                } else {
                    api.register_provider("anthropic".to_owned(), config)
                        .expect("provider registers");
                }
                Ok(())
            })
        }),
    }
}

fn command_extension(command: &str, config: ProviderConfigInput) -> InlineExtension {
    let command = command.to_owned();
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let command = command.clone();
            let config = config.clone();
            let handler_api = Arc::clone(&api);
            Box::pin(async move {
                let handler: CommandHandler = Arc::new(move |_args, _ctx| {
                    let config = config.clone();
                    let api = Arc::clone(&handler_api);
                    Box::pin(async move {
                        api.register_provider("anthropic".to_owned(), config)
                            .expect("provider registers");
                    })
                });
                api.register_command(command.clone(), Some("Use proxy".to_owned()), handler)
                    .expect("command registers");
                Ok(())
            })
        }),
    }
}

async fn capture_prompt_base_url(session: &Session) -> Option<String> {
    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&captured);
    session
        .agent()
        .set_stream_function(Some(Arc::new(move |model, _context, _options| {
            *sink.lock().expect("lock") = Some(model.base_url.clone());
            let stream = assistant_message_event_stream();
            stream.push(AssistantMessageEvent::Done {
                reason: StopReason::Error,
                message: AssistantMessage {
                    stop_reason: StopReason::Error,
                    error_message: Some("stop".to_owned()),
                    ..assistant_message("", 0)
                },
            });
            stream
        })));
    session
        .prompt("hello".to_owned(), None)
        .await
        .expect("prompt");
    captured.lock().expect("lock").clone()
}

#[tokio::test]
async fn applies_top_level_register_provider_overrides_to_the_active_model() {
    let session = session_with_extension(
        "/tmp/pi-session-dynamic-provider-top",
        provider_extension(base_url_config("http://localhost:8080/top-level"), false),
        false,
    )
    .await;

    assert_eq!(
        session.model().expect("model").base_url,
        "http://localhost:8080/top-level"
    );
    assert_eq!(
        capture_prompt_base_url(&session).await.as_deref(),
        Some("http://localhost:8080/top-level")
    );
}

#[tokio::test]
async fn applies_session_start_register_provider_overrides_to_the_active_model() {
    let session = session_with_extension(
        "/tmp/pi-session-dynamic-provider-start",
        provider_extension(base_url_config("http://localhost:8080/session-start"), true),
        true,
    )
    .await;

    assert_eq!(
        session.model().expect("model").base_url,
        "http://localhost:8080/session-start"
    );
    assert_eq!(
        capture_prompt_base_url(&session).await.as_deref(),
        Some("http://localhost:8080/session-start")
    );
}

fn native_config(base_url: &str) -> ProviderConfigInput {
    ProviderConfigInput {
        base_url: Some(base_url.to_owned()),
        api: Some(pi_ai::types::Api::from("anthropic-messages")),
        models: Some(vec![common::sdk_fixture::provider_model(
            "claude-sonnet-4-5",
            true,
            vec![pi_ai::types::Modality::Text],
        )]),
        stream_simple: Some(Arc::new(|_model, _context, _options| {
            let stream = assistant_message_event_stream();
            stream.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: assistant_message("ok", 0),
            });
            stream
        })),
        ..ProviderConfigInput::default()
    }
}

#[tokio::test]
async fn registers_native_ai_providers_during_extension_loading() {
    let session = session_with_extension(
        "/tmp/pi-session-dynamic-provider-native",
        provider_extension(
            native_config("http://localhost:8080/native-top-level"),
            false,
        ),
        false,
    )
    .await;

    assert_eq!(
        session.model().expect("model").base_url,
        "http://localhost:8080/native-top-level"
    );
    assert_eq!(
        capture_prompt_base_url(&session).await.as_deref(),
        Some("http://localhost:8080/native-top-level")
    );
}

#[tokio::test]
async fn applies_command_time_register_provider_overrides_without_reload() {
    let session = session_with_extension(
        "/tmp/pi-session-dynamic-provider-command",
        command_extension(
            "use-proxy",
            base_url_config("http://localhost:8080/command"),
        ),
        true,
    )
    .await;

    session
        .prompt("/use-proxy".to_owned(), None)
        .await
        .expect("command");
    assert_eq!(
        session.model().expect("model").base_url,
        "http://localhost:8080/command"
    );
    assert_eq!(
        capture_prompt_base_url(&session).await.as_deref(),
        Some("http://localhost:8080/command")
    );
}

#[tokio::test]
async fn registers_native_ai_providers_at_command_time() {
    let session = session_with_extension(
        "/tmp/pi-session-dynamic-provider-native-command",
        command_extension(
            "use-native",
            native_config("http://localhost:8080/native-command"),
        ),
        true,
    )
    .await;

    session
        .prompt("/use-native".to_owned(), None)
        .await
        .expect("command");
    assert_eq!(
        session.model().expect("model").base_url,
        "http://localhost:8080/native-command"
    );
    assert_eq!(
        capture_prompt_base_url(&session).await.as_deref(),
        Some("http://localhost:8080/native-command")
    );
}
