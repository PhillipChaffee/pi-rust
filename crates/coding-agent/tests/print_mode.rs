//! Print mode's single-shot flows, upstream's `test/print-mode.test.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream fakes the runtime host with a duck-typed object; the port drives a
//! real in-memory runtime over the faux provider and reads the raw stdout
//! writer's captured chunks.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;

use pi_ai::types::Api;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::ImageContent;
use pi_ai::types::Modality;
use pi_ai::types::StopReason;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::agent_session_runtime::AgentSessionRuntime;
use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeOptions;
use pi_coding_agent::core::agent_session_runtime::create_agent_session_runtime;
use pi_coding_agent::core::output_guard::RawStdoutWriter;
use pi_coding_agent::core::output_guard::set_raw_writer_for_test;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::reasons::SessionShutdownReason;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::modes::print_mode::PrintModeKind;
use pi_coding_agent::modes::print_mode::PrintModeOptions;
use pi_coding_agent::modes::print_mode::run_print_mode;
use pi_coding_agent::provider_composer::ProviderConfigInput;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;

use common::agent_session::RuntimeFactoryOptions;
use common::agent_session::runtime_factory;
use common::sdk_fixture::Fixture;
use common::sdk_fixture::faux_model;
use common::sdk_fixture::provider_model;
use common::sdk_fixture::registry_with_faux;

/// The raw stdout chunks a run wrote, restored when the capture drops.
struct CapturedStdout {
    buffer: Arc<Mutex<Vec<u8>>>,
    previous: Option<RawStdoutWriter>,
}

impl CapturedStdout {
    fn new() -> Self {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&buffer);
        let previous = set_raw_writer_for_test(Some(Arc::new(move |text: &str| {
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend_from_slice(text.as_bytes());
            Ok(())
        })));
        Self { buffer, previous }
    }

    fn text(&self) -> String {
        String::from_utf8(
            self.buffer
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone(),
        )
        .expect("the raw stdout carries UTF-8")
    }
}

impl Drop for CapturedStdout {
    fn drop(&mut self) {
        set_raw_writer_for_test(self.previous.take());
    }
}

fn shutdown_extension(
    log: Arc<Mutex<Vec<SessionShutdownReason>>>,
) -> Arc<dyn Fn() -> Vec<InlineExtension> + Send + Sync> {
    Arc::new(move || {
        let log = Arc::clone(&log);
        vec![InlineExtension {
            name: None,
            hidden: false,
            factory: Arc::new(move |api: Arc<ExtensionApi>| {
                let log = Arc::clone(&log);
                Box::pin(async move {
                    api.on(
                        "session_shutdown",
                        Arc::new(move |event: ExtensionEvent, _ctx| {
                            if let ExtensionEvent::SessionShutdown { reason, .. } = event {
                                log.lock()
                                    .unwrap_or_else(PoisonError::into_inner)
                                    .push(reason);
                            }
                            Box::pin(async { HandlerResult::None })
                        }),
                    )
                    .expect("session_shutdown handler");
                    Ok(())
                })
            }),
        }]
    })
}

async fn build_runtime(
    fixture: &Fixture,
    extensions: Arc<dyn Fn() -> Vec<InlineExtension> + Send + Sync>,
    registry: pi_coding_agent::model_registry::ModelRegistry,
) -> AgentSessionRuntime<InMemorySettingsStorage> {
    let factory = runtime_factory(RuntimeFactoryOptions {
        runtime: registry.runtime().clone(),
        model: faux_model(),
        extensions,
        no_extensions: true,
        tools: None,
    });
    create_agent_session_runtime(
        factory,
        CreateAgentSessionRuntimeOptions {
            cwd: fixture.cwd.clone(),
            agent_dir: fixture.agent_dir.clone(),
            session_manager: SessionManager::in_memory(Some(&fixture.cwd), None, None)
                .expect("session manager"),
            session_start_event: None,
            project_trust_context: None,
        },
    )
    .await
    .expect("runtime")
}

async fn registry_with_error_reply() -> pi_coding_agent::model_registry::ModelRegistry {
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
                    let stream = assistant_message_event_stream();
                    stream.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: common::agent_session::error_message("provider failure", 1),
                    });
                    stream
                })),
                models: Some(vec![provider_model(
                    common::sdk_fixture::FAUX_MODEL_ID,
                    true,
                    vec![Modality::Text, Modality::Image],
                )]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the error provider registers");
    registry
}

#[tokio::test]
async fn text_mode_prompts_with_images_and_emits_shutdown() {
    let fixture = Fixture::new();
    let log: Arc<Mutex<Vec<SessionShutdownReason>>> = Arc::new(Mutex::new(Vec::new()));
    let mut runtime = build_runtime(
        &fixture,
        shutdown_extension(Arc::clone(&log)),
        registry_with_faux().await,
    )
    .await;
    let session = Arc::clone(runtime.session());
    let captured = CapturedStdout::new();
    let images = vec![ImageContent {
        data: "abc".to_owned(),
        mime_type: "image/png".to_owned(),
    }];

    let code = Box::pin(run_print_mode(
        &mut runtime,
        PrintModeOptions {
            mode: PrintModeKind::Text,
            initial_message: Some("Say done".to_owned()),
            initial_images: images.clone(),
            ..PrintModeOptions::default()
        },
    ))
    .await;

    assert_eq!(code, 0);
    assert_eq!(captured.text(), "ok\n");
    assert_eq!(
        *log.lock().unwrap_or_else(PoisonError::into_inner),
        vec![SessionShutdownReason::Quit]
    );

    let user = {
        let state = session.agent().state();
        state
            .messages
            .iter()
            .find_map(|message| match message {
                pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::User(user)) => {
                    Some(user.clone())
                }
                _ => None,
            })
            .expect("the prompt landed in the transcript")
    };
    let pi_ai::types::UserContent::Blocks(blocks) = &user.content else {
        panic!("the image attachment switches the content to blocks");
    };
    assert!(blocks.iter().any(
        |block| matches!(block, pi_ai::types::UserBlock::Image(image) if image == &images[0])
    ));
}

#[tokio::test]
async fn json_mode_streams_events_and_emits_shutdown() {
    let fixture = Fixture::new();
    let log: Arc<Mutex<Vec<SessionShutdownReason>>> = Arc::new(Mutex::new(Vec::new()));
    let mut runtime = build_runtime(
        &fixture,
        shutdown_extension(Arc::clone(&log)),
        registry_with_faux().await,
    )
    .await;
    let captured = CapturedStdout::new();

    let code = Box::pin(run_print_mode(
        &mut runtime,
        PrintModeOptions {
            mode: PrintModeKind::Json,
            messages: vec!["hello".to_owned()],
            ..PrintModeOptions::default()
        },
    ))
    .await;

    assert_eq!(code, 0);
    let output = captured.text();
    let lines: Vec<serde_json::Value> = output
        .lines()
        .map(|line| serde_json::from_str(line).expect("every stdout line is JSON"))
        .collect();
    assert!(lines.iter().any(|line| {
        line["type"] == "message_end"
            && line["message"]["role"] == "assistant"
            && line["message"]["content"][0]["text"] == "ok"
    }));
    assert_eq!(
        *log.lock().unwrap_or_else(PoisonError::into_inner),
        vec![SessionShutdownReason::Quit]
    );
}

#[tokio::test]
async fn text_mode_reports_an_assistant_error_and_emits_shutdown() {
    let fixture = Fixture::new();
    let log: Arc<Mutex<Vec<SessionShutdownReason>>> = Arc::new(Mutex::new(Vec::new()));
    let registry = registry_with_error_reply().await;
    let mut runtime = build_runtime(&fixture, shutdown_extension(Arc::clone(&log)), registry).await;
    let captured = CapturedStdout::new();

    let code = Box::pin(run_print_mode(
        &mut runtime,
        PrintModeOptions {
            mode: PrintModeKind::Text,
            initial_message: Some("fail".to_owned()),
            ..PrintModeOptions::default()
        },
    ))
    .await;

    assert_eq!(code, 1);
    assert_eq!(captured.text(), "");
    assert_eq!(
        *log.lock().unwrap_or_else(PoisonError::into_inner),
        vec![SessionShutdownReason::Quit]
    );
}
