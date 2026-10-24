//! The SDK stream-option wiring, upstream's `test/sdk-stream-options.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;

use pi_ai::types::Api;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Modality;
use pi_ai::types::Model;
use pi_ai::types::ProviderId;
use pi_ai::types::SimpleStreamOptions;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::sdk::CreateAgentSessionOptions;
use pi_coding_agent::core::sdk::create_agent_session;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::provider_composer::ProviderConfigInput;
use pi_coding_agent::provider_composer::ProviderModelInput;
use pi_coding_agent::resource_loader::DefaultResourceLoader;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;
use serde_json::json;
use tempfile::TempDir;

fn capture_model() -> Model {
    Model {
        id: "capture-model".to_owned(),
        name: "Capture Model".to_owned(),
        api: Api::from("openai-completions"),
        provider: ProviderId("capture-provider".to_owned()),
        base_url: "https://capture.invalid/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: common::model_layer::zero_cost(),
        context_window: 128_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: Some(std::collections::BTreeMap::from([(
            "x-model".to_owned(),
            "model".to_owned(),
        )])),
        compat: None,
    }
}

fn done_stream(api: &Api) -> AssistantMessageEventStream {
    let stream = assistant_message_event_stream();
    let message = AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: "ok".to_owned(),
            text_signature: None,
        })],
        api: api.clone(),
        provider: ProviderId("capture-provider".to_owned()),
        model: "capture-model".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: pi_ai::types::Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    };
    stream.push(AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message,
    });
    stream
}

fn header_hook() -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(|api: Arc<pi_coding_agent::extensions::api::ExtensionApi>| {
            Box::pin(async move {
                api.on(
                    "before_provider_headers",
                    Arc::new(|event: ExtensionEvent, _ctx| {
                        Box::pin(async move {
                            let ExtensionEvent::BeforeProviderHeaders { mut headers } = event
                            else {
                                return HandlerResult::None;
                            };
                            let value = ["x-provider", "x-model", "x-explicit"]
                                .iter()
                                .map(|key| {
                                    headers.get(*key).and_then(Clone::clone).unwrap_or_default()
                                })
                                .collect::<Vec<_>>()
                                .join(":");
                            headers.insert("x-hook".to_owned(), Some(value));
                            HandlerResult::BeforeProviderHeaders(headers)
                        })
                    }),
                )
                .expect("the handler registers");
                Ok(())
            })
        }),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the fixture assembles one runtime end to end"
)]
async fn capture_stream_options(
    settings: Settings,
    request: Option<SimpleStreamOptions>,
    with_hook: bool,
) -> Option<SimpleStreamOptions> {
    let temp = TempDir::new().expect("tempdir");
    let cwd = temp.path().join("project");
    let agent_dir = temp.path().join("agent");
    std::fs::create_dir_all(&cwd).expect("cwd");
    std::fs::create_dir_all(&agent_dir).expect("agent dir");

    let model = capture_model();
    let registry = common::model_layer::create_in_memory_model_registry(
        common::model_layer::empty_auth_storage(),
    )
    .await;
    let runtime = registry.runtime().clone();
    let captured: Arc<Mutex<Option<SimpleStreamOptions>>> = Arc::new(Mutex::new(None));
    let captured_cell = Arc::clone(&captured);
    runtime
        .register_provider(
            "capture-provider",
            ProviderConfigInput {
                base_url: Some("https://capture.invalid/v1".to_owned()),
                api: Some(Api::from("openai-completions")),
                api_key: Some("test-api-key".to_owned()),
                headers: Some(std::collections::BTreeMap::from([(
                    "x-provider".to_owned(),
                    "provider".to_owned(),
                )])),
                stream_simple: Some(Arc::new(move |_model, _context, options| {
                    *captured_cell.lock().unwrap_or_else(PoisonError::into_inner) =
                        options.cloned();
                    done_stream(&Api::from("openai-completions"))
                })),
                models: Some(vec![ProviderModelInput {
                    id: "capture-model".to_owned(),
                    name: "Capture Model".to_owned(),
                    api: Some(Api::from("openai-completions")),
                    base_url: None,
                    reasoning: false,
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

    let settings_manager =
        SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default());
    let mut loader = DefaultResourceLoader::new(
        DefaultResourceLoaderOptions {
            cwd: cwd.display().to_string(),
            agent_dir: agent_dir.display().to_string(),
            extension_factories: if with_hook {
                vec![header_hook()]
            } else {
                Vec::new()
            },
            ..DefaultResourceLoaderOptions::default()
        },
        settings_manager,
    );
    loader.reload(None).await;

    let result = create_agent_session::<InMemorySettingsStorage>(CreateAgentSessionOptions {
        cwd: Some(cwd.display().to_string()),
        agent_dir: Some(agent_dir.display().to_string()),
        model_runtime: Some(runtime),
        model: Some(model.clone()),
        thinking_level: None,
        scoped_models: None,
        no_tools: None,
        tools: None,
        exclude_tools: None,
        custom_tools: None,
        resource_loader: Some(pi_coding_agent::resource_loader::shared_resource_loader(
            loader,
        )),
        session_manager: Some(
            SessionManager::in_memory(Some(&cwd.display().to_string()), None, None)
                .expect("session"),
        ),
        settings_manager: None,
        session_start_event: None,
    })
    .await
    .expect("session creates");
    let session = result.session;

    let stream_fn = session.agent().stream_function().expect("stream fn");
    let stream = stream_fn(
        &model,
        &common::model_layer::empty_context(),
        request.as_ref(),
    );
    let _ = stream.result().await;
    let options = captured
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    session.dispose();
    options
}

#[tokio::test]
async fn forwards_http_idle_timeout_as_timeout() {
    let options = capture_stream_options(
        Settings::from_iter([("httpIdleTimeoutMs".to_owned(), json!(1234))]),
        None,
        false,
    )
    .await;
    assert_eq!(options.expect("captured").timeout_ms, Some(1234));
}

#[tokio::test]
async fn request_timeout_overrides_the_setting() {
    let options = capture_stream_options(
        Settings::from_iter([("httpIdleTimeoutMs".to_owned(), json!(1234))]),
        Some(SimpleStreamOptions {
            timeout_ms: Some(0),
            ..SimpleStreamOptions::default()
        }),
        false,
    )
    .await;
    assert_eq!(options.expect("captured").timeout_ms, Some(0));
}

#[tokio::test]
async fn forwards_the_websocket_connect_timeout() {
    let options = capture_stream_options(
        Settings::from_iter([("websocketConnectTimeoutMs".to_owned(), json!(1234))]),
        None,
        false,
    )
    .await;
    assert_eq!(
        options.expect("captured").websocket_connect_timeout_ms,
        Some(1234)
    );
}

#[tokio::test]
async fn forwards_provider_retry_settings() {
    let options = capture_stream_options(
        Settings::from_iter([(
            "retry".to_owned(),
            json!({"provider": {"maxRetries": 2, "maxRetryDelayMs": 3000}}),
        )]),
        None,
        false,
    )
    .await;
    let options = options.expect("captured");
    assert_eq!(options.max_retries, Some(2));
    assert_eq!(options.max_retry_delay_ms, Some(3000));
}

#[tokio::test]
async fn runs_before_provider_headers_on_assembled_headers() {
    let options = capture_stream_options(
        Settings::new(),
        Some(SimpleStreamOptions {
            headers: Some(std::collections::BTreeMap::from([(
                "x-explicit".to_owned(),
                Some("explicit".to_owned()),
            )])),
            ..SimpleStreamOptions::default()
        }),
        true,
    )
    .await;
    let headers = options.expect("captured").headers.expect("headers");
    assert_eq!(
        headers.get("x-provider"),
        Some(&Some("provider".to_owned()))
    );
    assert_eq!(headers.get("x-model"), Some(&Some("model".to_owned())));
    assert_eq!(
        headers.get("x-explicit"),
        Some(&Some("explicit".to_owned()))
    );
    assert_eq!(
        headers.get("x-hook"),
        Some(&Some("provider:model:explicit".to_owned()))
    );
}

#[test]
fn block_images_replaces_and_dedupes_placeholders() {
    let settings = Settings::from_iter([("images".to_owned(), json!({"blockImages": true}))]);
    let manager = SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default());
    let user = pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Blocks(vec![
            pi_ai::types::UserBlock::Image(pi_ai::types::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
            pi_ai::types::UserBlock::Image(pi_ai::types::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]),
        timestamp: 1,
    };
    let converted = pi_coding_agent::core::sdk::convert_to_llm_with_block_images(
        &[pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(user),
        )],
        &manager,
    );
    let pi_ai::types::Message::User(user) = &converted[0] else {
        panic!("user message");
    };
    let pi_ai::types::UserContent::Blocks(blocks) = &user.content else {
        panic!("blocks");
    };
    assert_eq!(blocks.len(), 1, "consecutive placeholders dedupe");
    assert!(matches!(
        &blocks[0],
        pi_ai::types::UserBlock::Text(text) if text.text == "Image reading is disabled."
    ));
}

#[test]
fn block_images_off_keeps_images() {
    let manager =
        SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default());
    let user = pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Blocks(vec![pi_ai::types::UserBlock::Image(
            pi_ai::types::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            },
        )]),
        timestamp: 1,
    };
    let converted = pi_coding_agent::core::sdk::convert_to_llm_with_block_images(
        &[pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(user),
        )],
        &manager,
    );
    let pi_ai::types::Message::User(user) = &converted[0] else {
        panic!("user message");
    };
    assert!(matches!(
        &user.content,
        pi_ai::types::UserContent::Blocks(blocks)
            if matches!(&blocks[0], pi_ai::types::UserBlock::Image(_))
    ));
}
