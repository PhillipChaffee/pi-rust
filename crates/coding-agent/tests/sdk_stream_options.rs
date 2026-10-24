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
use pi_ai::types::ProviderResponse;
use pi_ai::types::SimpleStreamOptions;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::agent_session::AgentSession;
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

fn done_message(api: &Api) -> AssistantMessage {
    AssistantMessage {
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
    }
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
    extensions: Vec<InlineExtension>,
) -> Capture {
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
                stream_simple: Some(Arc::new(move |model, _context, options| {
                    *captured_cell.lock().unwrap_or_else(PoisonError::into_inner) =
                        options.cloned();
                    let stream = assistant_message_event_stream();
                    let sink = stream.clone();
                    let model = model.clone();
                    let on_payload =
                        options.and_then(|options| options.transport_options.on_payload.clone());
                    let on_response =
                        options.and_then(|options| options.transport_options.on_response.clone());
                    tokio::spawn(async move {
                        if let Some(on_payload) = on_payload {
                            let _ = on_payload.call(json!({"probe": true}), model.clone()).await;
                        }
                        if let Some(on_response) = on_response {
                            on_response
                                .call(
                                    ProviderResponse {
                                        status: 200,
                                        headers: std::collections::BTreeMap::new(),
                                    },
                                    model,
                                )
                                .await;
                        }
                        sink.push(AssistantMessageEvent::Done {
                            reason: StopReason::Stop,
                            message: done_message(&Api::from("openai-completions")),
                        });
                    });
                    stream
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
            extension_factories: extensions,
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
    let result = stream.result().await;
    let options = captured
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    Capture {
        session,
        options,
        result,
    }
}

struct Capture {
    session: Arc<AgentSession<InMemorySettingsStorage>>,
    options: Option<SimpleStreamOptions>,
    result: AssistantMessage,
}

#[tokio::test]
async fn forwards_http_idle_timeout_as_timeout() {
    let options = capture_stream_options(
        Settings::from_iter([("httpIdleTimeoutMs".to_owned(), json!(1234))]),
        None,
        Vec::new(),
    )
    .await;
    let capture = options;
    assert_eq!(capture.options.expect("captured").timeout_ms, Some(1234));
    capture.session.dispose();
}

#[tokio::test]
async fn request_timeout_overrides_the_setting() {
    let options = capture_stream_options(
        Settings::from_iter([("httpIdleTimeoutMs".to_owned(), json!(1234))]),
        Some(SimpleStreamOptions {
            timeout_ms: Some(0),
            ..SimpleStreamOptions::default()
        }),
        Vec::new(),
    )
    .await;
    let capture = options;
    assert_eq!(capture.options.expect("captured").timeout_ms, Some(0));
    capture.session.dispose();
}

#[tokio::test]
async fn forwards_the_websocket_connect_timeout() {
    let options = capture_stream_options(
        Settings::from_iter([("websocketConnectTimeoutMs".to_owned(), json!(1234))]),
        None,
        Vec::new(),
    )
    .await;
    let capture = options;
    assert_eq!(
        capture
            .options
            .expect("captured")
            .websocket_connect_timeout_ms,
        Some(1234)
    );
    capture.session.dispose();
}

#[tokio::test]
async fn forwards_provider_retry_settings() {
    let options = capture_stream_options(
        Settings::from_iter([(
            "retry".to_owned(),
            json!({"provider": {"maxRetries": 2, "maxRetryDelayMs": 3000}}),
        )]),
        None,
        Vec::new(),
    )
    .await;
    let capture = options;
    let options = capture.options.expect("captured");
    assert_eq!(options.max_retries, Some(2));
    assert_eq!(options.max_retry_delay_ms, Some(3000));
    capture.session.dispose();
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
        vec![header_hook()],
    )
    .await;
    let capture = options;
    let headers = capture.options.expect("captured").headers.expect("headers");
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
    capture.session.dispose();
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

fn hooks_factory(log: Arc<Mutex<Vec<String>>>) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(
            move |api: Arc<pi_coding_agent::extensions::api::ExtensionApi>| {
                let log = Arc::clone(&log);
                Box::pin(async move {
                    let payload_log = Arc::clone(&log);
                    api.on(
                        "before_provider_request",
                        Arc::new(move |event, _ctx| {
                            payload_log
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push("payload".to_owned());
                            let ExtensionEvent::BeforeProviderRequest { payload } = event else {
                                return Box::pin(async { HandlerResult::None });
                            };
                            Box::pin(async move { HandlerResult::BeforeProviderRequest(payload) })
                        }),
                    )
                    .expect("payload handler");
                    let response_log = Arc::clone(&log);
                    api.on(
                        "after_provider_response",
                        Arc::new(move |_event, _ctx| {
                            response_log
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push("response".to_owned());
                            Box::pin(async { HandlerResult::None })
                        }),
                    )
                    .expect("response handler");
                    let context_log = Arc::clone(&log);
                    api.on(
                        "context",
                        Arc::new(move |event, _ctx| {
                            context_log
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push("context".to_owned());
                            let ExtensionEvent::Context { messages } = event else {
                                return Box::pin(async { HandlerResult::None });
                            };
                            Box::pin(async move {
                                HandlerResult::Context(
                                    pi_coding_agent::extensions::results::ContextResult {
                                        messages: Some(messages),
                                    },
                                )
                            })
                        }),
                    )
                    .expect("context handler");
                    Ok(())
                })
            },
        ),
    }
}

#[tokio::test]
async fn runs_the_payload_response_and_context_hooks() {
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let capture =
        capture_stream_options(Settings::new(), None, vec![hooks_factory(Arc::clone(&log))]).await;
    capture
        .session
        .prompt("hello".to_owned(), None)
        .await
        .expect("prompt");
    capture.session.wait_for_idle().await;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !log
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .any(|entry| entry == "response")
        && std::time::Instant::now() < deadline
    {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let entries = log.lock().unwrap_or_else(PoisonError::into_inner).clone();
    assert!(
        entries.iter().any(|entry| entry == "payload"),
        "{entries:?}"
    );
    assert!(
        entries.iter().any(|entry| entry == "response"),
        "{entries:?}"
    );
    assert!(
        entries.iter().any(|entry| entry == "context"),
        "{entries:?}"
    );
    capture.session.dispose();
}

#[tokio::test]
async fn invalid_timeout_settings_fail_the_stream() {
    let capture = capture_stream_options(
        Settings::from_iter([("httpIdleTimeoutMs".to_owned(), json!("bogus"))]),
        None,
        Vec::new(),
    )
    .await;
    assert_eq!(capture.result.stop_reason, StopReason::Error);
    assert!(
        capture
            .result
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("Invalid httpIdleTimeoutMs")
    );
    capture.session.dispose();

    let capture = capture_stream_options(
        Settings::from_iter([("websocketConnectTimeoutMs".to_owned(), json!("bogus"))]),
        None,
        Vec::new(),
    )
    .await;
    assert_eq!(capture.result.stop_reason, StopReason::Error);
    assert!(
        capture
            .result
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("Invalid websocketConnectTimeoutMs")
    );
    capture.session.dispose();
}

#[test]
fn block_images_filters_tool_result_images() {
    let settings = Settings::from_iter([("images".to_owned(), json!({"blockImages": true}))]);
    let manager = SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default());
    let message = pi_ai::types::ToolResultMessage {
        tool_call_id: "call".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![
            pi_ai::types::ToolResultBlock::Image(pi_ai::types::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
            pi_ai::types::ToolResultBlock::Image(pi_ai::types::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    };
    let converted = pi_coding_agent::core::sdk::convert_to_llm_with_block_images(
        &[pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::ToolResult(message),
        )],
        &manager,
    );
    let pi_ai::types::Message::ToolResult(message) = &converted[0] else {
        panic!("tool result message");
    };
    assert_eq!(message.content.len(), 1, "consecutive placeholders dedupe");
    assert!(matches!(
        &message.content[0],
        pi_ai::types::ToolResultBlock::Text(text)
            if text.text == "Image reading is disabled."
    ));
}

#[tokio::test]
async fn request_websocket_and_retry_values_override_settings() {
    let capture = capture_stream_options(
        Settings::from_iter([
            ("websocketConnectTimeoutMs".to_owned(), json!(1234)),
            (
                "retry".to_owned(),
                json!({"provider": {"maxRetries": 2, "maxRetryDelayMs": 3000}}),
            ),
        ]),
        Some(SimpleStreamOptions {
            websocket_connect_timeout_ms: Some(0),
            max_retries: Some(5),
            max_retry_delay_ms: Some(10),
            ..SimpleStreamOptions::default()
        }),
        Vec::new(),
    )
    .await;
    let options = capture.options.expect("captured");
    assert_eq!(options.websocket_connect_timeout_ms, Some(0));
    assert_eq!(options.max_retries, Some(5));
    assert_eq!(options.max_retry_delay_ms, Some(10));
    capture.session.dispose();
}

#[tokio::test]
async fn negative_retry_settings_fall_back_to_request_values() {
    let capture = capture_stream_options(
        Settings::from_iter([(
            "retry".to_owned(),
            json!({"provider": {"maxRetries": -1, "maxRetryDelayMs": -1, "timeoutMs": -1}}),
        )]),
        None,
        Vec::new(),
    )
    .await;
    let options = capture.options.expect("captured");
    assert_eq!(options.max_retries, None);
    capture.session.dispose();
}

#[test]
fn block_images_leaves_plain_text_content_alone() {
    let settings = Settings::from_iter([("images".to_owned(), json!({"blockImages": true}))]);
    let manager = SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default());
    let user = pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Text("plain".to_owned()),
        timestamp: 1,
    };
    let converted = pi_coding_agent::core::sdk::convert_to_llm_with_block_images(
        &[pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(user),
        )],
        &manager,
    );
    assert!(matches!(
        &converted[0],
        pi_ai::types::Message::User(user)
            if matches!(&user.content, pi_ai::types::UserContent::Text(text) if text == "plain")
    ));
}
