//! The OpenAI Completions seam edges the upstream suites reach only
//! implicitly: transport-error mapping, SDK error message shapes, malformed
//! SSE chunks, stream-lifecycle failures, credential setup, the payload
//! hook, and the typed tool-choice shape. Behaviors upstream exercises at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`; tests pinning port
//! seams the upstream suites reach only implicitly are marked in their doc
//! comments.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::openai_completions::{
    OpenAiCompletionsOptions, OpenAiToolChoice, stream, stream_simple,
};
use pi_ai::http::{
    HttpByteStream, HttpClient, HttpError, HttpRequest, HttpResponse, MockHttpClient,
};
use pi_ai::types::{
    AssistantBlock, CacheControlFormat, CacheRetention, ChatTemplateKwargValue, ChatTemplateVar,
    ConstrainedSamplingConfig, ConstrainedSamplingSetting, Context, DeferredToolsMode,
    GrammarFormat, Message, Model, ModelCompat, SimpleStreamOptions, StopReason, ThinkingContent,
    ThinkingFormat, ThinkingLevel, ThinkingLevelMap, ThinkingTemplateVar, Tool, ToolCall,
    ToolChoice, ToolResultBlock, ToolResultMessage, TransportOptions, UserBlock, UserContent,
    UserMessage,
};
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

mod common;
use common::{openai_done_chunk, openai_mock_with, recorded_body};

fn done_context() -> Context {
    Context {
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Hello".to_owned()),
            timestamp: 1,
        })],
        ..Context::default()
    }
}

fn keyed(mock: &MockHttpClient) -> OpenAiCompletionsOptions {
    common::keyed_openai_options(mock)
}

/// A canned-seam client: the request always resolves 200 and the body
/// streams the given chunks, the shape the SSE error mapping needs.
#[derive(Debug)]
struct SeamedClient {
    chunks: Vec<Result<bytes::Bytes, HttpError>>,
}

impl HttpClient for SeamedClient {
    fn execute(
        &self,
        _request: HttpRequest,
    ) -> pi_ai::http::BoxHttpFuture<Result<HttpResponse, HttpError>> {
        let response = HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: HttpByteStream::from_chunks(self.chunks.clone()),
        };
        Box::pin(async move { Ok(response) })
    }
}

fn seamed_options(chunks: Vec<Result<bytes::Bytes, HttpError>>) -> OpenAiCompletionsOptions {
    OpenAiCompletionsOptions {
        transport_options: TransportOptions {
            http_client: Some(std::sync::Arc::new(SeamedClient { chunks })),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..OpenAiCompletionsOptions::default()
    }
}

fn delta_frame(delta: &Value, finish_reason: Option<&str>) -> bytes::Bytes {
    let mut choice = json!({ "index": 0, "delta": delta });
    if let Some(finish) = finish_reason {
        choice["finish_reason"] = json!(finish);
    }
    bytes::Bytes::from(format!(
        "data: {}\n\n",
        json!({ "id": "chatcmpl-seamed", "choices": [choice] })
    ))
}

// ---------------------------------------------------------------------------
// Credential setup
// ---------------------------------------------------------------------------

/// Port-added: the wire `stream` without a credential settles through the
/// error event before dispatching — the async form of upstream
/// `streamSimple`'s synchronous auth throw.
#[tokio::test]
async fn stream_without_a_credential_fails_before_dispatch() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let model = common::openai_completions_model();
    let context = done_context();
    let options = OpenAiCompletionsOptions {
        transport_options: common::mock_transport(&mock),
        ..OpenAiCompletionsOptions::default()
    };

    let result = stream(&model, &context, Some(&options)).result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: opencode-go")
    );
    assert_eq!(mock.request_count(), 0);
}

/// Port-added: `stream_simple` without any credential fails through the
/// setup-error stream instead of dispatching.
#[tokio::test]
async fn stream_simple_without_a_credential_fails_before_dispatch() {
    let mock = MockHttpClient::new();
    let model = common::openai_completions_model();
    let context = done_context();

    let result = stream_simple(&model, &context, None).result().await;

    common::assert_setup_error_without_dispatch(&result, &mock);
}

/// Header-owned authorization stands in for a key: the request dispatches
/// with the caller's header pair.
#[tokio::test]
async fn header_owned_authorization_dispatches_without_a_key() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let model = common::openai_completions_model();
    let options = OpenAiCompletionsOptions {
        transport_options: common::mock_transport(&mock),
        headers: Some(
            std::iter::once((
                "Authorization".to_owned(),
                Some("Bearer caller-token".to_owned()),
            ))
            .collect(),
        ),
        ..OpenAiCompletionsOptions::default()
    };

    let result = stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert!(
        common::recorded_header_values(&mock, "Authorization")
            .iter()
            .any(|value| value == "Bearer caller-token")
    );
}

// ---------------------------------------------------------------------------
// Transport-error mapping
// ---------------------------------------------------------------------------

/// Port-added: the seam's `Timeout` and `InvalidUrl` arms keep their
/// transport wordings.
#[tokio::test]
async fn transport_error_arms_surface_their_seam_messages() {
    let cases: [(HttpError, &str); 2] = [
        (HttpError::Timeout, "request timed out"),
        (
            HttpError::InvalidUrl("not a url".to_owned()),
            "invalid URL: not a url",
        ),
    ];
    for (error, expected) in cases {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url.contains("/chat/completions"))
            .respond_fn(move |_request| {
                let error = error.clone();
                async move { Err(error) }
            });
        let model = common::openai_completions_model();
        let options = keyed(&mock);

        let result = stream(&model, &done_context(), Some(&options))
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some(expected));
    }
}

/// A pre-cancelled signal aborts before dispatch: the stream stops with
/// `Aborted`, upstream's `AbortError` mapping.
#[tokio::test]
async fn aborted_requests_set_the_aborted_stop_reason() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let token = CancellationToken::new();
    token.cancel();
    let model = common::openai_completions_model();
    let options = OpenAiCompletionsOptions {
        transport_options: TransportOptions {
            http_client: Some(std::sync::Arc::new(mock.clone())),
            signal: Some(token),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..OpenAiCompletionsOptions::default()
    };

    let result = stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("Request aborted"));
}

// ---------------------------------------------------------------------------
// SDK error message shapes
// ---------------------------------------------------------------------------

/// Port-added: the raw-body, parsed-body, and missing-body variants of the
/// SDK's `{status} {body}` error message.
#[tokio::test]
async fn sdk_error_messages_cover_raw_parsed_and_missing_bodies() {
    // A raw text body rides the message verbatim.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(pi_ai::http::MockResponse::status(502).with_body("Bad Gateway"));
    assert_error_message(&mock, "502 Bad Gateway").await;

    // A missing body falls back to the SDK's no-body form.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(pi_ai::http::MockResponse::status(503));
    assert_error_message(&mock, "503 status code (no body)").await;

    // A parsed JSON body serializes into the message.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(pi_ai::http::json_response(
            400,
            &json!({ "error": { "message": "bad shape" } }),
        ));
    assert_error_message(&mock, "400 {\"error\":{\"message\":\"bad shape\"}}").await;
}

/// Stream against the mounted mock and pin the settled error message.
async fn assert_error_message(mock: &MockHttpClient, expected: &str) {
    let model = common::openai_completions_model();
    let options = keyed(mock);

    let result = stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some(expected));
}

// ---------------------------------------------------------------------------
// Malformed SSE surfaces
// ---------------------------------------------------------------------------

/// A data frame whose JSON does not parse fails the stream with the parse
/// message naming the chunk data and raw lines.
#[tokio::test]
async fn malformed_sse_chunks_fail_with_the_parse_message() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(
            pi_ai::http::MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body("data: {not json}\n\ndata: [DONE]\n\n"),
        );
    let model = common::openai_completions_model();
    let options = keyed(&mock);

    let result = stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the parse error");
    assert!(
        message.starts_with("Could not parse OpenAI SSE chunk:"),
        "got: {message}"
    );
    assert!(message.contains("data={not json}"), "got: {message}");
    assert!(message.contains("raw="), "got: {message}");
}

// ---------------------------------------------------------------------------
// Mid-stream seam errors
// ---------------------------------------------------------------------------

/// Port-added: a transport failure mid-SSE carries the transport's wording,
/// and a mid-stream abort carries the abort wording.
#[tokio::test]
async fn mid_stream_errors_carry_their_wording() {
    let cases: [(HttpError, &str); 2] = [
        (
            HttpError::Transport("socket reset".to_owned()),
            "socket reset",
        ),
        (HttpError::Aborted, "Request was aborted"),
    ];
    for (error, expected) in cases {
        let options = seamed_options(vec![
            Ok(delta_frame(&json!({ "content": "partial" }), None)),
            Err(error),
        ]);
        let model = common::openai_completions_model();

        let result = stream(&model, &done_context(), Some(&options))
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some(expected));
    }
}

/// The non-2xx error body parses before the failure surfaces, so the
/// provider's JSON reason rides the message.
#[tokio::test]
async fn error_bodies_parse_before_the_failure_surfaces() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(pi_ai::http::json_response(
            429,
            &json!({ "error": { "message": "quota exhausted" } }),
        ));
    let model = common::openai_completions_model();
    let options = keyed(&mock);

    let result = stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the failure message");
    assert!(message.contains("quota exhausted"), "got: {message}");
    assert!(message.contains("429"), "got: {message}");
}

// ---------------------------------------------------------------------------
// Streamed reasoning details replay on failure
// ---------------------------------------------------------------------------

/// Port-added: a stream that fails after `reasoning_details` streamed replays
/// the accumulated details onto the thinking block's signature, the replay
/// half of upstream's catch-block cleanup.
#[tokio::test]
async fn streamed_reasoning_details_replay_onto_thinking_blocks_when_the_stream_fails() {
    let detail = json!({
        "type": "reasoning.encrypted",
        "id": "call_1",
        "data": "encrypted-signature",
    });
    let options = seamed_options(vec![
        Ok(delta_frame(&json!({ "reasoning_details": [detail] }), None)),
        Ok(bytes::Bytes::from_static(b"data: {not json}\n\n")),
    ]);
    let model = common::openai_completions_model();

    let result = stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let thinking = result.content.iter().find_map(|block| match block {
        AssistantBlock::Thinking(thinking) => Some(thinking),
        _ => None,
    });
    let thinking = thinking.expect("the thinking block");
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some(json!([detail]).to_string().as_str())
    );
}

// ---------------------------------------------------------------------------
// Payload shaping seams
// ---------------------------------------------------------------------------

/// Port-added: the on-payload hook can replace the request payload, the
/// seam-side port of upstream's `onPayload` mutation.
#[tokio::test]
async fn the_payload_hook_can_replace_the_request_payload() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let model = common::openai_completions_model();
    let mut options = keyed(&mock);
    options.transport_options.on_payload =
        Some(pi_ai::types::OnPayload::new(|_payload, _model| {
            Box::pin(async {
                Some(json!({
                    "model": "test-model",
                    "messages": [{ "role": "user", "content": "replaced" }],
                    "stream": true,
                }))
            })
        }));

    let _ = stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    let body = recorded_body(&mock);
    assert_eq!(body["messages"][0]["content"], json!("replaced"));
}

/// Port-added: the typed named-function tool choice maps to the wire's
/// `{"type":"function","function":{"name":...}}` shape.
#[tokio::test]
async fn function_tool_choice_maps_to_the_named_function_shape() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let model = common::openai_completions_model();
    let context = Context {
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("use the tool".to_owned()),
            timestamp: 1,
        })],
        tools: Some(vec![Tool {
            name: "lookup".to_owned(),
            description: "Look up.".to_owned(),
            parameters: json!({ "type": "object", "properties": {} }),
            constrained_sampling: None,
        }]),
        ..Context::default()
    };
    let options = OpenAiCompletionsOptions {
        api_key: Some("test".to_owned()),
        tool_choice: Some(OpenAiToolChoice::Function {
            name: "lookup".to_owned(),
        }),
        transport_options: common::mock_transport(&mock),
        ..OpenAiCompletionsOptions::default()
    };

    let _ = stream(&model, &context, Some(&options)).result().await;

    let body = recorded_body(&mock);
    assert_eq!(
        body["tool_choice"],
        json!({ "type": "function", "function": { "name": "lookup" } })
    );
    assert_eq!(body["tools"][0]["function"]["name"], json!("lookup"));
}

/// Port-added: the pi user agent rides every completions request, and the
/// model's own headers merge over the defaults.
#[tokio::test]
async fn the_pi_user_agent_and_model_headers_ride_the_request() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let model = Model {
        headers: Some(
            std::iter::once(("x-model-tenant".to_owned(), "tenant-1".to_owned())).collect(),
        ),
        ..common::openai_completions_model()
    };
    let options = keyed(&mock);

    let _ = stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    let agent = common::recorded_header(&mock, "User-Agent").expect("the user agent");
    assert!(agent.starts_with("pi "), "got: {agent}");
    assert_eq!(
        common::recorded_header(&mock, "x-model-tenant").as_deref(),
        Some("tenant-1")
    );
}

/// Port-added: the response hook observes the settled status.
#[tokio::test]
async fn the_response_hook_observes_the_response_status() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let slot = std::sync::Arc::clone(&observed);
    let model = common::openai_completions_model();
    let options = OpenAiCompletionsOptions {
        transport_options: TransportOptions {
            http_client: Some(std::sync::Arc::new(mock.clone())),
            on_response: Some(pi_ai::types::OnResponse::new(move |response, _model| {
                slot.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(response.status);
                Box::pin(async {})
            })),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..OpenAiCompletionsOptions::default()
    };

    let result = stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(
        *observed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        [200]
    );
}
// ---------------------------------------------------------------------------
// Thinking-format matrix edges
// ---------------------------------------------------------------------------

/// A reasoning-capable custom model whose compat overrides ride as given,
/// the fixture the thinking-format matrix suites drive.
fn compat_model(base_url: &str, compat: ModelCompat) -> Model {
    Model {
        id: "custom-model".to_owned(),
        name: "Custom Model".to_owned(),
        api: pi_ai::types::Api::from("openai-completions"),
        provider: pi_ai::types::ProviderId::from("custom"),
        base_url: base_url.to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 4000,
        max_tokens: 400,
        sampling_params: None,
        headers: None,
        compat: Some(compat),
    }
}

/// The bridged-provider context: one assistant lookup turn answering with a
/// "done" tool result and a follow-up user turn, the context the
/// bridged-provider cases replay.
fn bridged_context() -> Context {
    Context {
        messages: vec![
            Message::Assistant(common::tool_call_assistant_message(
                "openai-completions",
                "custom",
                "custom-model",
                ToolCall {
                    id: "call_1".to_owned(),
                    name: "lookup".to_owned(),
                    arguments: Map::new(),
                    thought_signature: None,
                    namespace: None,
                },
            )),
            common::tool_result_message(
                "call_1",
                vec![ToolResultBlock::Text(common::text_block("done"))],
                None,
            ),
            common::user_message_now("next"),
        ],
        ..Context::default()
    }
}

/// Stream the context against the canned done chunk and take the request's
/// payload, the matrix suites' capture helper.
async fn capture_compat(
    model: &Model,
    context: &Context,
    mut options: OpenAiCompletionsOptions,
) -> Value {
    if options.api_key.is_none() {
        options.api_key = Some("test-key".to_owned());
    }
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    options.transport_options = common::mock_transport(&mock);
    let _ = stream(model, context, Some(&options)).result().await;
    recorded_body(&mock)
}

/// Stream the detail-matrix chunks and take the thinking block, the shape
/// the streamed-signature cases read.
async fn streamed_thinking(model: &Model, chunks: &[Value], expect: &str) -> ThinkingContent {
    let message = stream_compat_chunks(
        model,
        &done_context(),
        keyed(&MockHttpClient::new()),
        chunks,
    )
    .await;
    message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::Thinking(thinking) => Some(thinking.clone()),
            _ => None,
        })
        .expect(expect)
}

/// Capture the request payload a replayed assistant tool-call signature
/// produces, the shape the reasoning-signature suites read.
async fn replayed_signature_payload(thought_signature: Option<String>) -> Value {
    let assistant = common::tool_call_assistant_message(
        "openai-completions",
        "custom",
        "custom-model",
        ToolCall {
            id: "call_1".to_owned(),
            name: "lookup".to_owned(),
            arguments: Map::new(),
            thought_signature,
            namespace: None,
        },
    );
    let context = Context {
        messages: vec![
            Message::Assistant(assistant),
            common::user_message_now("next"),
        ],
        ..Context::default()
    };
    capture_compat(
        &compat_model("https://custom.example/v1", ModelCompat::default()),
        &context,
        keyed(&MockHttpClient::new()),
    )
    .await
}

/// Stream the given chunks and settle the final message, the stream-shape
/// suites' helper.
async fn stream_compat_chunks(
    model: &Model,
    context: &Context,
    mut options: OpenAiCompletionsOptions,
    chunks: &[Value],
) -> pi_ai::types::AssistantMessage {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, chunks);
    options.transport_options = common::mock_transport(&mock);
    common::drain_and_settle(&stream(model, context, Some(&options))).await
}

/// Stream the chunks through the grammar harness and take the streamed
/// tool-call block, the shape the custom-tool cases share.
async fn streamed_custom_tool_call(chunks: &[Value], expect: &str) -> ToolCall {
    let message = stream_compat_chunks(
        &grammar_model(),
        &grammar_context(),
        keyed(&MockHttpClient::new()),
        chunks,
    )
    .await;
    message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .expect(expect)
}

/// Stream the detail-matrix chunks and take the last serialized reasoning
/// entry, the shape the merge-matrix cases read.
async fn merged_detail_entry(chunks: &[Value]) -> Value {
    let message = stream_compat_chunks(
        &compat_model("https://custom.example/v1", ModelCompat::default()),
        &done_context(),
        keyed(&MockHttpClient::new()),
        chunks,
    )
    .await;
    let signature = message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::Thinking(block) => Some(block.thinking_signature.clone()),
            _ => None,
        })
        .flatten()
        .expect("the merged details signature");
    let details: Value = serde_json::from_str(&signature).expect("the signature parses");
    details
        .as_array()
        .and_then(|entries| entries.last())
        .cloned()
        .expect("the merged entry")
}

/// A `thinkingLevelMap` from `(level, entry)` pairs, the fixtures the mapped
/// effort branches read.
fn level_map(entries: &[(pi_ai::types::ModelThinkingLevel, Option<&str>)]) -> ThinkingLevelMap {
    entries
        .iter()
        .map(|(level, entry)| (*level, entry.map(str::to_owned)))
        .collect()
}

/// Port-added: the Together thinking format spells `reasoning.enabled` and,
/// when the compat admits the effort field, the mapped `reasoning_effort`.
#[tokio::test]
async fn together_thinking_sends_the_enabled_flag_and_the_mapped_effort() {
    let base = ModelCompat {
        supports_reasoning_effort: Some(true),
        ..ModelCompat::default()
    };
    let model = compat_model("https://api.together.ai/v1", base.clone());
    let with_effort = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::High),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert_eq!(with_effort["reasoning"], json!({"enabled": true}));
    assert_eq!(with_effort["reasoning_effort"], json!("high"));

    let model = compat_model("https://api.together.ai/v1", base.clone());
    let without =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert_eq!(without["reasoning"], json!({"enabled": false}));
    assert![without.get("reasoning_effort").is_none()];
}

/// Port-added: the Deepseek thinking format sends `thinking` type objects
/// gated by the off entry, with the effort mapped into `reasoning_effort`.
#[tokio::test]
async fn deepseek_thinking_disabled_rides_when_the_off_entry_is_not_null() {
    let base = ModelCompat {
        thinking_format: Some(ThinkingFormat::Deepseek),
        supports_reasoning_effort: Some(true),
        ..ModelCompat::default()
    };
    let model = compat_model("https://custom.example/v1", base.clone());
    let enabled = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Medium),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert_eq!(enabled["thinking"], json!({"type": "enabled"}));
    assert_eq!(enabled["reasoning_effort"], json!("medium"));

    let model = compat_model("https://custom.example/v1", base.clone());
    let disabled =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert_eq!(disabled["thinking"], json!({"type": "disabled"}));

    let mut model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            thinking_format: Some(ThinkingFormat::Deepseek),
            ..ModelCompat::default()
        },
    );
    model.thinking_level_map = Some(level_map(&[(pi_ai::types::ModelThinkingLevel::Off, None)]));
    let null_off =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert![null_off.get("thinking").is_none()];
}

/// Port-added: the string-thinking format sends the level spelling, the
/// map's off entry, or the `none` fallback.
#[tokio::test]
async fn string_thinking_sends_the_level_the_off_entry_or_none() {
    let base = ModelCompat {
        thinking_format: Some(ThinkingFormat::StringThinking),
        ..ModelCompat::default()
    };
    let model = compat_model("https://custom.example/v1", base.clone());
    let effort = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Medium),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert_eq!(effort["thinking"], json!("medium"));

    let model = compat_model("https://custom.example/v1", base.clone());
    let none = capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert_eq!(none["thinking"], json!("none"));

    let mut model = compat_model("https://custom.example/v1", base.clone());
    model.thinking_level_map = Some(level_map(&[(
        pi_ai::types::ModelThinkingLevel::Off,
        Some("off-state"),
    )]));
    let mapped_off =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert_eq!(mapped_off["thinking"], json!("off-state"));
}

/// Port-added: chat-template kwargs ride scalars as given and drop on the
/// null unsupported-level marker.
#[tokio::test]
async fn chat_template_kwargs_ride_scalars_and_drop_null_mapped_efforts() {
    let kwargs = [
        (
            "static_str".to_owned(),
            ChatTemplateKwargValue::Str("s".to_owned()),
        ),
        ("num".to_owned(), ChatTemplateKwargValue::Number(2.into())),
        ("flag".to_owned(), ChatTemplateKwargValue::Bool(true)),
        ("nil".to_owned(), ChatTemplateKwargValue::Null),
        (
            "mapped".to_owned(),
            ChatTemplateKwargValue::Template(ChatTemplateVar {
                var: ThinkingTemplateVar::ThinkingEffort,
                omit_when_off: None,
            }),
        ),
        (
            "omitted".to_owned(),
            ChatTemplateKwargValue::Template(ChatTemplateVar {
                var: ThinkingTemplateVar::ThinkingEnabled,
                omit_when_off: Some(true),
            }),
        ),
    ];
    let compat = ModelCompat {
        thinking_format: Some(ThinkingFormat::ChatTemplate),
        chat_template_kwargs: Some(kwargs.iter().cloned().collect()),
        ..ModelCompat::default()
    };
    let model = compat_model("https://custom.example/v1", compat.clone());
    let payload = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Minimal),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    let sent = &payload["chat_template_kwargs"];
    assert_eq!(sent["static_str"], json!("s"));
    assert_eq!(sent["num"], json!(2));
    assert_eq!(sent["flag"], json!(true));
    assert_eq!(sent["nil"], Value::Null);
    assert_eq!(sent["mapped"], json!("minimal"));
    assert_eq!(sent["omitted"], json!(true));

    // With no requested effort the omit_when_off kwarg drops and the
    // unmapped `$var` falls back to nothing.
    let model = compat_model("https://custom.example/v1", compat.clone());
    let dropped =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    let sent = &dropped["chat_template_kwargs"];
    assert![sent.get("omitted").is_none()];
    assert![sent.get("mapped").is_none()];

    let mut model = compat_model("https://custom.example/v1", compat.clone());
    model.thinking_level_map = Some(level_map(&[(
        pi_ai::types::ModelThinkingLevel::Minimal,
        None,
    )]));
    let dropped = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Minimal),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert![dropped["chat_template_kwargs"].get("mapped").is_none()];
}

/// Port-added: the Baseten format resolves `chat_template_args`, reads the
/// map's off entry for the effort field, and omits on the null marker.
#[tokio::test]
async fn baseten_args_and_off_effort_spell_the_baseten_wire() {
    let args = [(
        "thinking".to_owned(),
        ChatTemplateKwargValue::Template(ChatTemplateVar {
            var: ThinkingTemplateVar::ThinkingEnabled,
            omit_when_off: None,
        }),
    )];
    let base = ModelCompat {
        thinking_format: Some(ThinkingFormat::Baseten),
        supports_reasoning_effort: Some(true),
        chat_template_args: Some(args.iter().cloned().collect()),
        ..ModelCompat::default()
    };
    let mut model = compat_model("https://custom.example/v1", base.clone());
    model.thinking_level_map = Some(level_map(&[(
        pi_ai::types::ModelThinkingLevel::Off,
        Some("off-value"),
    )]));
    let payload =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert_eq!(payload["chat_template_args"]["thinking"], json!(false));
    assert_eq!(payload["reasoning_effort"], json!("off-value"));

    let model = compat_model("https://custom.example/v1", base.clone());
    let without_map =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert![without_map.get("reasoning_effort").is_none()];

    let mut model = compat_model("https://custom.example/v1", base.clone());
    model.thinking_level_map = Some(level_map(&[(
        pi_ai::types::ModelThinkingLevel::Medium,
        None,
    )]));
    let null_marker = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Medium),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert![null_marker.get("reasoning_effort").is_none()];
}

/// Port-added: the zai effort field omits on the null unsupported-level
/// marker and falls back to the level spelling on a missing entry.
#[tokio::test]
async fn zai_effort_omits_on_the_null_marker_and_spells_missing_levels() {
    let base = ModelCompat {
        thinking_format: Some(ThinkingFormat::Zai),
        supports_reasoning_effort: Some(true),
        ..ModelCompat::default()
    };
    let mut model = compat_model("https://custom.example/v1", base.clone());
    model.thinking_level_map = Some(level_map(&[(
        pi_ai::types::ModelThinkingLevel::Minimal,
        None,
    )]));
    let null_marker = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Minimal),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert_eq!(
        null_marker["thinking"],
        json!({"type": "enabled", "clear_thinking": false})
    );
    assert![null_marker.get("reasoning_effort").is_none()];

    let model = compat_model("https://custom.example/v1", base.clone());
    let missing = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Xhigh),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert_eq!(missing["reasoning_effort"], json!("xhigh"));
}

/// Port-added: unrequested efforts on the OpenAI format send the map's off
/// entry as `reasoning_effort`.
#[tokio::test]
async fn the_off_entry_spells_the_openai_effort_when_nothing_is_requested() {
    let compat = ModelCompat {
        supports_reasoning_effort: Some(true),
        ..ModelCompat::default()
    };
    let mut model = compat_model("https://custom.example/v1", compat.clone());
    model.thinking_level_map = Some(level_map(&[(
        pi_ai::types::ModelThinkingLevel::Off,
        Some("low"),
    )]));
    let payload =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert_eq!(payload["reasoning_effort"], json!("low"));
}

#[tokio::test]
async fn stream_simple_spells_the_minimal_and_xhigh_levels() {
    // A level map admitting Minimal lets stream_simple pass the request
    // through unclamped.
    let mut model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            supports_reasoning_effort: Some(true),
            ..ModelCompat::default()
        },
    );
    model.thinking_level_map = Some(level_map(&[(
        pi_ai::types::ModelThinkingLevel::Minimal,
        Some("minimal"),
    )]));
    let minimal = capture_simple_payload(&model, Some(ThinkingLevel::Minimal)).await;
    assert_eq!(minimal["reasoning_effort"], json!("minimal"));

    // The raw stream takes the requested effort unclamped, so Xhigh spells
    // itself when no map renames it.
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            supports_reasoning_effort: Some(true),
            ..ModelCompat::default()
        },
    );
    let xhigh = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Xhigh),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert_eq!(xhigh["reasoning_effort"], json!("xhigh"));
}

async fn capture_simple_payload(model: &Model, reasoning: Option<ThinkingLevel>) -> Value {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = SimpleStreamOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        reasoning,
        ..SimpleStreamOptions::default()
    };
    let _ = stream_simple(model, &done_context(), Some(&options))
        .result()
        .await;
    recorded_body(&mock)
}

// ---------------------------------------------------------------------------
// Request-shape edges
// ---------------------------------------------------------------------------

/// Port-added: Vercel AI Gateway routing preferences ride the
/// `providerOptions.gateway` request field.
#[tokio::test]
async fn vercel_gateway_routing_rides_the_provider_options_field() {
    let compat = ModelCompat {
        vercel_gateway_routing: Some(pi_ai::types::VercelGatewayRouting {
            only: Some(vec!["bedrock".to_owned(), "anthropic".to_owned()]),
            order: Some(vec!["anthropic".to_owned()]),
        }),
        ..ModelCompat::default()
    };
    let model = compat_model("https://custom.example/v1", compat.clone());
    let payload =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert_eq!(
        payload["providerOptions"],
        json!({"gateway": {"only": ["bedrock", "anthropic"], "order": ["anthropic"]}})
    );
}

/// Port-added: the simple options' auto and none tool choices ride the
/// `tool_choice` field.
#[tokio::test]
async fn the_simple_tool_choices_ride_the_request() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = SimpleStreamOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        tool_choice: Some(ToolChoice::Auto),
        ..SimpleStreamOptions::default()
    };
    let _ = stream_simple(
        &compat_model("https://custom.example/v1", ModelCompat::default()),
        &done_context(),
        Some(&options),
    )
    .result()
    .await;
    assert_eq!(recorded_body(&mock)["tool_choice"], json!("auto"));
}

/// Port-added: unknown finish reasons fail the stream with the provider's
/// spelling; `length` maps to the length stop.
#[tokio::test]
async fn finish_reasons_map_to_their_stops_and_errors() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let context = done_context();
    let options = || {
        let mock = MockHttpClient::new();
        openai_mock_with(&mock, &[openai_done_chunk()]);
        keyed(&mock)
    };

    let length = stream_compat_chunks(
        &model,
        &context,
        options(),
        &[
            json!({"choices": [{"delta": {}, "finish_reason": "length"}]}),
            json!({"choices": [{"delta": {}}], "usage": null}),
        ],
    )
    .await;
    assert_eq![length.stop_reason, StopReason::Length];

    let unknown = stream_compat_chunks(
        &model,
        &context,
        options(),
        &[json!({"choices": [{"delta": {}, "finish_reason": "policy_violation"}]})],
    )
    .await;
    assert_eq![unknown.stop_reason, StopReason::Error];
    assert_eq![
        unknown.error_message.as_deref(),
        Some("Provider finish_reason: policy_violation")
    ];
}

/// Port-added: the OpenRouter `metadata.raw` extra appends after a newline
/// when the composed message does not already carry it.
#[tokio::test]
async fn openrouter_metadata_raw_appends_after_a_newline() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(
            pi_ai::http::MockResponse::status(500)
                .with_body(r#"{"error":{"message":"boom"},"metadata":{"raw":"line1\nline2"}}"#),
        );
    let result = stream(
        &compat_model("https://custom.example/v1", ModelCompat::default()),
        &done_context(),
        Some(&keyed(&mock)),
    )
    .result()
    .await;
    assert_eq![result.stop_reason, StopReason::Error];
    let message = result.error_message.expect("the error message");
    assert![message.contains("boom"), "{message}"];
    // The raw extra rides the body escaped, so the composed message does not
    // contain the unescaped spelling and the display appends it once.
    assert![message.contains("\nline1\nline2"), "{message}"];
    assert_eq!(
        message.matches("line1").count(),
        2,
        "the body copy and the appended extra"
    );
}

/// Port-added: a grammar rejection at request build time fails the stream
/// before anything is dispatched.
#[tokio::test]
async fn a_grammar_rejection_fails_before_dispatch() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            supports_openai_grammar_tools: Some(true),
            ..ModelCompat::default()
        },
    );
    let context = Context {
        messages: done_context().messages,
        tools: Some(vec![Tool {
            name: "ln".to_owned(),
            description: "Grammar tool".to_owned(),
            parameters: json!({"type": "object", "properties": {}, "required": []}),
            constrained_sampling: Some(ConstrainedSamplingSetting::Config(
                ConstrainedSamplingConfig::Grammar {
                    variants: std::collections::BTreeMap::new(),
                },
            )),
        }]),
        ..Context::default()
    };
    let result = stream(&model, &context, Some(&keyed(&mock))).result().await;
    assert_eq![result.stop_reason, StopReason::Error];
    let message = result.error_message.expect("the grammar rejection");
    assert![
        message.contains("cannot use grammar constrained sampling"),
        "{message}"
    ];
    assert_eq!(mock.request_count(), 0);
}

/// Port-added: chunk usage reads JS truthiness — `null`, `0`, `""`, and
/// `false` leave the message usage zeroed.
#[tokio::test]
async fn chunk_usage_truthiness_skips_the_falsy_shapes() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let context = done_context();
    let chunks = [
        json!({"choices": [{"delta": {}}], "usage": null}),
        json!({"choices": [{"delta": {}}], "usage": 0}),
        json!({"choices": [{"delta": {}}], "usage": ""}),
        json!({"choices": [{"delta": {}}], "usage": false}),
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ];
    let message =
        stream_compat_chunks(&model, &context, keyed(&MockHttpClient::new()), &chunks).await;
    assert_eq![message.stop_reason, StopReason::Stop];
    assert_eq!(message.usage.input, 0);
    assert_eq!(message.usage.output, 0);
}

/// Port-added: Moonshot-style per-choice usage fills the message usage when
/// the chunk carries no top-level usage.
#[tokio::test]
async fn moonshot_style_choice_usage_fills_the_message_usage() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let chunks = [
        json!({"choices": [{"delta": {"content": "Hi"}, "usage": {"prompt_tokens": 10, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 3}, "completion_tokens_details": {"reasoning_tokens": 2}}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ];
    let message = stream_compat_chunks(
        &model,
        &done_context(),
        keyed(&MockHttpClient::new()),
        &chunks,
    )
    .await;
    assert_eq!(message.usage.input, 7);
    assert_eq!(message.usage.output, 4);
    assert_eq!(message.usage.cache_read, 3);
    assert_eq!(message.usage.reasoning, Some(2));
}

/// Port-added: chunks without choices or without a delta leave the message
/// untouched and the stream still settles.
#[tokio::test]
async fn choiceless_and_deltaless_chunks_do_not_touch_the_message() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let chunks = [
        json!({"choices": []}),
        json!({"choices": [{"delta": null}]}),
        json!({"id": "chatcmpl-x", "choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]}),
    ];
    let message = stream_compat_chunks(
        &model,
        &done_context(),
        keyed(&MockHttpClient::new()),
        &chunks,
    )
    .await;
    assert_eq![message.stop_reason, StopReason::Stop];
    assert_eq!(
        message.content[0],
        AssistantBlock::Text(common::text_block("Hi"))
    );
}

/// Port-added: a tool call id that arrives after the block was created fills
/// the tracked block and its id table.
#[tokio::test]
async fn a_late_tool_call_id_updates_the_tracked_block() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let context = Context {
        messages: done_context().messages,
        tools: Some(vec![common::lookup_tool()]),
        ..Context::default()
    };
    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "lookup", "arguments": "{\"q\":"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_late", "function": {"arguments": "1}"}}]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let message =
        stream_compat_chunks(&model, &context, keyed(&MockHttpClient::new()), &chunks).await;
    let block = message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("the streamed tool call");
    assert_eq![block.id, "call_late"];
    assert_eq![
        block.arguments,
        json!({"q": 1}).as_object().expect("present").clone()
    ];
}

// ---------------------------------------------------------------------------
// Custom (grammar) tool streaming
// ---------------------------------------------------------------------------

/// The grammar-constrained tool the custom-tool suites stream through: one
/// required string property named `text`.
fn grammar_tool() -> Tool {
    Tool {
        name: "ln".to_owned(),
        description: "Grammar tool".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }),
        constrained_sampling: Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::Grammar {
                variants: std::iter::once((
                    GrammarFormat::OpenaiLark,
                    "start: /[a-z]+/".to_owned(),
                ))
                .collect(),
            },
        )),
    }
}

fn grammar_context() -> Context {
    Context {
        messages: done_context().messages,
        tools: Some(vec![grammar_tool()]),
        ..Context::default()
    }
}

fn grammar_model() -> Model {
    compat_model(
        "https://custom.example/v1",
        ModelCompat {
            supports_openai_grammar_tools: Some(true),
            ..ModelCompat::default()
        },
    )
}

/// Port-added: custom tool-call deltas stream through the grammar buffer and
/// close with a final `"}"` delta at finish.
#[tokio::test]
async fn custom_tool_input_deltas_stream_through_the_grammar_buffer() {
    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "custom": {"name": "ln", "input": "hel"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "custom": {"input": "lo"}}]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let block = streamed_custom_tool_call(&chunks, "the streamed custom tool call").await;
    assert_eq![block.id, "c1"];
    assert_eq![block.name, "ln"];
    let arguments = block.arguments.get("text").and_then(Value::as_str);
    assert_eq!(arguments, Some("hello"));
}

/// Port-added: a custom delta on a function-shaped block retrofits the
/// grammar input state and re-stages the arguments around the property.
#[tokio::test]
async fn a_custom_delta_retrofits_a_function_shaped_tool_block() {
    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "ln", "arguments": "{\"text\":\"ab"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "custom": {"input": "cd"}}]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let block = streamed_custom_tool_call(&chunks, "the retrofitted tool call").await;
    assert_eq![block.name, "ln"];
    let arguments = block.arguments.get("text").and_then(Value::as_str);
    assert_eq!(arguments, Some("cd"));
}

/// Port-added: the custom input buffer closes exactly once — the finish
/// close appends the `"}"` fragment and further input would fail, the
/// append-only contract the grammar tool contract pins.
#[tokio::test]
async fn the_custom_input_close_emits_the_final_delta() {
    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "custom": {"name": "ln", "input": "a\"b"}}]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let message = stream_compat_chunks(
        &grammar_model(),
        &grammar_context(),
        keyed(&MockHttpClient::new()),
        &chunks,
    )
    .await;
    assert_eq![message.stop_reason, StopReason::ToolUse];
    let block = message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("the streamed custom tool call");
    let arguments = block.arguments.get("text").and_then(Value::as_str);
    assert_eq!(arguments, Some("a\"b"));
}

/// Port-added: assistant grammar calls replay as the wire's custom shape,
/// and a non-string input under the property aborts the request.
#[tokio::test]
async fn assistant_grammar_calls_replay_as_custom_wire_calls() {
    let mut context = grammar_context();
    context.messages = vec![
        Message::Assistant(common::tool_call_assistant_message(
            "openai-completions",
            "custom",
            "custom-model",
            ToolCall {
                id: "call_1".to_owned(),
                name: "ln".to_owned(),
                arguments: Map::from_iter([("text".to_owned(), json!("hi"))]),
                thought_signature: None,
                namespace: None,
            },
        )),
        common::tool_result_message(
            "call_1",
            vec![ToolResultBlock::Text(common::text_block("done"))],
            None,
        ),
    ];
    let payload = capture_compat(&grammar_model(), &context, keyed(&MockHttpClient::new())).await;
    assert_eq!(
        payload["messages"][0]["tool_calls"][0],
        json!({"id": "call_1", "type": "custom", "custom": {"name": "ln", "input": "hi"}})
    );
}

/// Port-added: an assistant grammar call without a string input under the
/// grammar property aborts the request before dispatch.
#[tokio::test]
async fn an_assistant_grammar_call_without_a_string_input_aborts_the_request() {
    let mut context = grammar_context();
    context.messages = vec![Message::Assistant(common::tool_call_assistant_message(
        "openai-completions",
        "custom",
        "custom-model",
        ToolCall {
            id: "call_1".to_owned(),
            name: "ln".to_owned(),
            arguments: Map::from_iter([("text".to_owned(), json!(42))]),
            thought_signature: None,
            namespace: None,
        },
    ))];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let result = stream(&grammar_model(), &context, Some(&keyed(&mock)))
        .result()
        .await;
    assert_eq![result.stop_reason, StopReason::Error];
    let message = result.error_message.expect("the grammar rejection");
    assert![
        message.contains("requires argument \"text\" to be a string"),
        "{message}"
    ];
    assert_eq!(mock.request_count(), 0);
}

// ---------------------------------------------------------------------------
// Reasoning-details replay and streaming shapes
// ---------------------------------------------------------------------------

/// Port-added: malformed thinking signatures do not replay reasoning
/// details — the parse rejects non-JSON, non-arrays, empty arrays, and
/// arrays with invalid entries.
#[tokio::test]
async fn malformed_thinking_signatures_do_not_replay_reasoning_details() {
    let signatures = [
        "not json".to_owned(),
        "5".to_owned(),
        "\"json string\"".to_owned(),
        "[]".to_owned(),
        json!([{ "type": "reasoning.text", "text": 5 }]).to_string(),
        json!([{ "type": "reasoning.summary" }]).to_string(),
        json!([{ "type": "reasoning.encrypted", "data": 5 }]).to_string(),
    ];
    for signature in signatures {
        let assistant = common::thinking_assistant_message(
            "openai-completions",
            "custom",
            "custom-model",
            "deep thought",
            signature.as_str(),
        );
        let context = Context {
            messages: vec![
                Message::Assistant(assistant),
                common::user_message_now("next"),
            ],
            ..Context::default()
        };
        let payload = capture_compat(
            &compat_model("https://custom.example/v1", ModelCompat::default()),
            &context,
            keyed(&MockHttpClient::new()),
        )
        .await;
        assert![
            payload["messages"][0].get("reasoning_details").is_none(),
            "{signature}"
        ];
    }
}

/// Port-added: the legacy encrypted tool-call signature replays its detail;
/// wrong types or empty id/data do not.
#[tokio::test]
async fn legacy_encrypted_tool_signatures_replay_their_detail() {
    let detail = json!({
        "type": "reasoning.encrypted",
        "id": "rs_id",
        "data": "enc-payload",
        "format": "openai",
        "index": 0
    });
    let payload = replayed_signature_payload(Some(detail.to_string())).await;
    assert_eq!(payload["messages"][0]["reasoning_details"], json!([detail]));

    for signature in [
        json!({"type": "reasoning.text", "text": "s", "id": "rs_id", "data": "enc"}).to_string(),
        json!({"type": "reasoning.encrypted", "id": "", "data": "enc"}).to_string(),
        json!({"type": "reasoning.encrypted", "id": "rs_id", "data": ""}).to_string(),
        "not json".to_owned(),
    ] {
        let payload = replayed_signature_payload(Some(signature.clone())).await;
        assert![
            payload["messages"][0].get("reasoning_details").is_none(),
            "{signature}"
        ];
    }
}

/// Port-added: a thinking signature naming one of the reasoning fields
/// replays the text through that field; an unknown signature stays a plain
/// replay.
#[tokio::test]
async fn reasoning_field_signatures_replay_their_text() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    for (signature, field) in [
        ("reasoning_text", "reasoning_text"),
        ("reasoning", "reasoning"),
        ("bogus", ""),
    ] {
        let mut assistant = common::thinking_assistant_message(
            "openai-completions",
            "custom",
            "custom-model",
            "deep thought",
            signature,
        );
        // The text block keeps the replay alive; thinking-only replays with
        // no content and no tool calls are dropped.
        assistant
            .content
            .push(AssistantBlock::Text(common::text_block("answer")));
        let context = Context {
            messages: vec![
                Message::Assistant(assistant),
                common::user_message_now("next"),
            ],
            ..Context::default()
        };
        let payload = capture_compat(&model, &context, keyed(&MockHttpClient::new())).await;
        if field.is_empty() {
            assert![payload["messages"][0].get(field).is_none(), "{signature}"];
        } else {
            assert_eq!(
                payload["messages"][0][field],
                json!("deep thought"),
                "{signature}"
            );
        }
    }
}

/// Port-added: streamed reasoning details merge in place, fill missing
/// common fields, and serialize into the thinking signature at close; a
/// non-string text payload merges as its string form.
#[tokio::test]
async fn streamed_reasoning_details_merge_and_replay_into_signatures() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let chunks = [
        json!({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "text": "a"}
        ]}}]}),
        json!({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "text": "b", "id": "rs_1", "format": "v1", "index": 0},
            {"type": "reasoning.summary", "summary": "s", "id": "rs_2", "index": 1}
        ]}}]}),
        json!({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "text": 5, "id": "rs_3"}
        ]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ];
    let thinking = streamed_thinking(&model, &chunks, "the streamed thinking block").await;
    let signature = thinking
        .thinking_signature
        .as_deref()
        .expect("the details signature");
    let details: Vec<Value> = serde_json::from_str(signature).expect("the serialized details");
    assert_eq!(details.len(), 2);
    let merged = &details[0];
    assert_eq!(merged["text"], json!("ab"));
    assert_eq!(merged["id"], json!("rs_1"));
    assert_eq!(merged["format"], json!("v1"));
    assert_eq!(merged["index"], json!(0));
    assert_eq!(details[1]["summary"], json!("s"));
}

/// Port-added: streamed reasoning details replay onto every thinking block
/// when the stream fails mid-flight.
#[tokio::test]
async fn stream_failures_replay_the_streamed_reasoning_details() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let context = done_context();
    let mock = MockHttpClient::new();
    openai_mock_with(
        &mock,
        &[
            json!({"choices": [{"delta": {"reasoning_details": [
                {"type": "reasoning.text", "text": "a", "id": "rs_1"}
            ]}}]}),
            json!({"choices": [{"delta": {"content": "partial"}}]}),
        ],
    );
    let options = OpenAiCompletionsOptions {
        transport_options: TransportOptions {
            http_client: Some(std::sync::Arc::new(SeamedClient {
                chunks: vec![
                    Ok(delta_frame(
                        &json!({"reasoning_details": [{"type": "reasoning.text", "text": "a"}]}),
                        None,
                    )),
                    Err(HttpError::Transport("dead server".to_owned())),
                ],
            })),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..OpenAiCompletionsOptions::default()
    };
    let message = common::drain_and_settle(&stream(&model, &context, Some(&options))).await;
    assert_eq![message.stop_reason, StopReason::Error];
    let thinking = message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::Thinking(thinking) => Some(thinking),
            _ => None,
        })
        .expect("the thinking block");
    let signature = thinking
        .thinking_signature
        .as_deref()
        .expect("the replayed details");
    let details: Vec<Value> = serde_json::from_str(signature).expect("the serialized details");
    assert_eq![details[0]["text"], json!("a")];
}

// ---------------------------------------------------------------------------
// Message-conversion edges
// ---------------------------------------------------------------------------

/// Port-added: block-form user messages send `image_url` parts and empty
/// block lists are skipped.
#[tokio::test]
async fn block_user_messages_send_image_parts_and_empty_blocks_are_skipped() {
    let mut model = compat_model("https://custom.example/v1", ModelCompat::default());
    model.input = vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image];
    let context = Context {
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Blocks(vec![
                    UserBlock::Text(common::text_block("look")),
                    UserBlock::Image(pi_ai::types::ImageContent {
                        data: "aGk=".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ]),
                timestamp: 1,
            }),
            Message::User(UserMessage {
                content: UserContent::Blocks(vec![]),
                timestamp: 2,
            }),
        ],
        ..Context::default()
    };
    let payload = capture_compat(&model, &context, keyed(&MockHttpClient::new())).await;
    assert_eq!(payload["messages"].as_array().expect("present").len(), 1);
    assert_eq!(
        payload["messages"][0]["content"],
        json!([
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}
        ])
    );
}

/// Port-added: assistant messages without content and tool calls are dropped;
/// text-only replays ride the plain string.
#[tokio::test]
async fn empty_assistant_replays_are_dropped_and_text_rides_plain() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let empty = common::assistant_message_with_content(
        "openai-completions",
        "custom",
        "custom-model",
        vec![AssistantBlock::Text(common::text_block("   "))],
    );
    let texted = common::assistant_message_with_content(
        "openai-completions",
        "custom",
        "custom-model",
        vec![AssistantBlock::Text(common::text_block("answer"))],
    );
    let context = Context {
        messages: vec![
            Message::Assistant(empty),
            Message::Assistant(texted),
            common::user_message_now("next"),
        ],
        ..Context::default()
    };
    let payload = capture_compat(&model, &context, keyed(&MockHttpClient::new())).await;
    let messages = payload["messages"].as_array().expect("present");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["content"], json!("answer"));
}

/// Port-added: thinking replays with accompanying text ride the plain string
/// content, and thinking-as-text compat wraps both into one array.
#[tokio::test]
async fn thinking_and_text_replays_ride_the_plain_or_array_content() {
    let plain = compat_model("https://custom.example/v1", ModelCompat::default());
    let thinking_text = Context {
        messages: vec![
            Message::Assistant(common::assistant_message_with_content(
                "openai-completions",
                "custom",
                "custom-model",
                vec![
                    AssistantBlock::Thinking(ThinkingContent {
                        thinking: "hmm".to_owned(),
                        thinking_signature: None,
                        redacted: None,
                    }),
                    AssistantBlock::Text(common::text_block("answer")),
                ],
            )),
            common::user_message_now("next"),
        ],
        ..Context::default()
    };
    let payload = capture_compat(&plain, &thinking_text, keyed(&MockHttpClient::new())).await;
    assert_eq!(payload["messages"][0]["content"], json!("answer"));

    let as_text = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            requires_thinking_as_text: Some(true),
            ..ModelCompat::default()
        },
    );
    let payload = capture_compat(&as_text, &thinking_text, keyed(&MockHttpClient::new())).await;
    assert_eq!(
        payload["messages"][0]["content"],
        json!([
            {"type": "text", "text": "hmm"},
            {"type": "text", "text": "answer"}
        ])
    );
}

/// Port-added: providers that require the bridge get the synthetic assistant
/// turn after tool results, the empty content string on assistant replays,
/// and the tool-result name field.
#[tokio::test]
async fn bridged_providers_get_assistant_turns_and_tool_result_names() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            requires_assistant_after_tool_result: Some(true),
            requires_tool_result_name: Some(true),
            ..ModelCompat::default()
        },
    );
    let context = bridged_context();
    let payload = capture_compat(&model, &context, keyed(&MockHttpClient::new())).await;
    let messages = payload["messages"].as_array().expect("present");
    assert_eq!(messages[0]["role"], json!("assistant"));
    assert_eq!(messages[0]["content"], json!(""));
    assert_eq!(messages[1]["role"], json!("tool"));
    assert_eq!(messages[1]["name"], json!("tool"));
    assert_eq!(messages[2]["role"], json!("assistant"));
    assert_eq!(
        messages[2]["content"],
        json!("I have processed the tool results.")
    );
    assert_eq!(messages[3]["role"], json!("user"));
}

/// Port-added: image-carrying tool results send the placeholder text, ride
/// the following user message's image parts, and bridge with the synthetic
/// assistant turn when the compat requires it.
#[tokio::test]
async fn image_tool_results_send_the_placeholder_and_the_image_user_message() {
    let mut with_image = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            requires_assistant_after_tool_result: Some(true),
            ..ModelCompat::default()
        },
    );
    with_image.input = vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image];
    let image = pi_ai::types::ImageContent {
        data: "aGk=".to_owned(),
        mime_type: "image/png".to_owned(),
    };
    let context = Context {
        messages: vec![
            common::tool_result_message("call_1", vec![ToolResultBlock::Image(image)], None),
            common::user_message_now("next"),
        ],
        ..Context::default()
    };
    let payload = capture_compat(&with_image, &context, keyed(&MockHttpClient::new())).await;
    let messages = payload["messages"].as_array().expect("present");
    assert_eq!(messages[0]["content"], json!("(see attached image)"));
    assert_eq!(messages[1]["role"], json!("assistant"));
    assert_eq!(messages[2]["role"], json!("user"));
    assert_eq!(
        messages[2]["content"][0]["text"],
        json!("Attached image(s) from tool result:")
    );
    assert_eq!(
        messages[2]["content"][1]["image_url"]["url"],
        json!("data:image/png;base64,aGk=")
    );
}

/// Port-added: Kimi's deferred tools re-enter through a system message with
/// tools, and the deferred names stay out of the active tools list.
#[tokio::test]
async fn kimi_deferred_tools_reenter_through_a_system_message() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            deferred_tools_mode: Some(DeferredToolsMode::Kimi),
            ..ModelCompat::default()
        },
    );
    let mut context = bridged_context();
    context.messages[0] = Message::Assistant(common::tool_call_assistant_message(
        "openai-completions",
        "custom",
        "custom-model",
        ToolCall {
            id: "call_1".to_owned(),
            name: "lookup".to_owned(),
            arguments: Map::from_iter([("q".to_owned(), json!("x"))]),
            thought_signature: None,
            namespace: None,
        },
    ));
    context.messages[1] = common::tool_result_message(
        "call_1",
        vec![ToolResultBlock::Text(common::text_block("done"))],
        Some(vec!["late_tool".to_owned()]),
    );
    context.tools = Some(vec![common::lookup_tool(), common::late_tool()]);
    let payload = capture_compat(&model, &context, keyed(&MockHttpClient::new())).await;
    let messages = payload["messages"].as_array().expect("present");
    let system = messages
        .iter()
        .find(|message| message["role"] == json!("system"))
        .expect("the deferred-tools system message");
    assert![system.get("content").is_none()];
    assert_eq!(system["tools"][0]["function"]["name"], json!("late_tool"));
    // Only the non-deferred tool rides the request's tools list.
    assert_eq!(payload["tools"].as_array().expect("present").len(), 1);
}

// ---------------------------------------------------------------------------
// Cache-control surfaces
// ---------------------------------------------------------------------------

/// Port-added: the Anthropic-style cache-control marker hits the system
/// prompt, the last tool, and the last conversation message; a message whose
/// content carries image blocks gets the marker on its last text part; long
/// retention spells the ttl when the compat admits it.
#[tokio::test]
async fn anthropic_cache_control_hits_every_surface() {
    let mut model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            cache_control_format: Some(CacheControlFormat::Anthropic),
            supports_long_cache_retention: Some(true),
            ..ModelCompat::default()
        },
    );
    model.input = vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image];
    let context = Context {
        system_prompt: Some("System prompt.".to_owned()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                UserBlock::Text(common::text_block("look")),
                UserBlock::Image(pi_ai::types::ImageContent {
                    data: "aGk=".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
            ]),
            timestamp: 1,
        })],
        tools: Some(vec![common::lookup_tool()]),
    };
    let payload = capture_compat(
        &model,
        &context,
        OpenAiCompletionsOptions {
            cache_retention: Some(CacheRetention::Long),
            ..keyed(&MockHttpClient::new())
        },
    )
    .await;
    assert_eq!(
        payload["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );
    assert_eq!(
        payload["tools"][0]["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );
    let parts = payload["messages"][1]["content"]
        .as_array()
        .expect("present");
    let last_text = parts
        .iter()
        .rev()
        .find(|part| part["type"] == json!("text"))
        .expect("present");
    assert_eq!(
        last_text["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );
}

/// Port-added: without a system prompt the cache-control walk skips the
/// system arm; without any text-bearing conversation message it walks to the
/// end; and an empty tools list gets no marker.
#[tokio::test]
async fn cache_control_walks_past_its_empty_targets() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            cache_control_format: Some(CacheControlFormat::Anthropic),
            ..ModelCompat::default()
        },
    );
    // Tool history with no tools of its own: the wire carries an empty tools
    // array and the marker walk skips it.
    let context = Context {
        messages: vec![
            Message::Assistant(common::tool_call_assistant_message(
                "openai-completions",
                "custom",
                "custom-model",
                ToolCall {
                    id: "call_1".to_owned(),
                    name: "lookup".to_owned(),
                    arguments: Map::new(),
                    thought_signature: None,
                    namespace: None,
                },
            )),
            common::tool_result_message("call_1", vec![], None),
            common::user_message_now("next"),
        ],
        ..Context::default()
    };
    let payload = capture_compat(&model, &context, keyed(&MockHttpClient::new())).await;
    assert_eq!(payload["tools"], json!([]));
    let messages = payload["messages"].as_array().expect("present");
    // The tool message holds the placeholder, and the last user message
    // carries the marker.
    assert_eq!(messages[1]["content"], json!("(no tool output)"));
    assert_eq!(
        messages[2]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
}

// ---------------------------------------------------------------------------
// Copilot request headers
// ---------------------------------------------------------------------------

/// Port-added: github-copilot completions requests carry the dynamic
/// Copilot headers, with the vision flag on image-bearing input.
#[tokio::test]
async fn github_copilot_requests_carry_the_dynamic_headers() {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let mut model = compat_model("https://api.githubcopilot.com/", ModelCompat::default());
    model.provider = pi_ai::types::ProviderId::from("github-copilot");
    model.input = vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image];
    let context = Context {
        messages: vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                UserBlock::Text(common::text_block("look")),
                UserBlock::Image(pi_ai::types::ImageContent {
                    data: "aGk=".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
            ]),
            timestamp: 1,
        })],
        ..Context::default()
    };
    let _ = stream(&model, &context, Some(&keyed(&mock))).result().await;
    assert_eq!(
        common::recorded_header(&mock, "Openai-Intent").as_deref(),
        Some("conversation-edits")
    );
    assert_eq!(
        common::recorded_header(&mock, "Copilot-Vision-Request").as_deref(),
        Some("true")
    );
}

// ---------------------------------------------------------------------------
// Stream-lifecycle failures
// ---------------------------------------------------------------------------

/// Port-added: a cancelled signal at finish fails the stream with the
/// aborted failure instead of settling the message.
#[tokio::test]
async fn a_cancelled_signal_at_finish_fails_the_stream() {
    let token = CancellationToken::new();
    let cancel_token = token.clone();
    // The canned chunk client's body reads do not race the signal, so the
    // cancel lands after the chunk loop and the finish check sees it.
    let client = SeamedClient {
        chunks: vec![Ok(delta_frame(&json!({}), Some("stop")))],
    };
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let options = OpenAiCompletionsOptions {
        transport_options: TransportOptions {
            http_client: Some(std::sync::Arc::new(client)),
            signal: Some(token),
            on_response: Some(pi_ai::types::OnResponse::new(move |_response, _model| {
                cancel_token.cancel();
                Box::pin(async {})
            })),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..OpenAiCompletionsOptions::default()
    };
    let message = common::drain_and_settle(&stream(&model, &done_context(), Some(&options))).await;
    assert_eq![message.stop_reason, StopReason::Aborted];
    assert_eq!(
        message.error_message.as_deref(),
        Some("Request was aborted")
    );
}

/// Port-added: a stream that ends without a `finish_reason` the compat
/// supports fails with the lifecycle message.
#[tokio::test]
async fn a_stream_without_a_supported_finish_reason_fails() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let chunks = [json!({"choices": [{"delta": {"content": "Hi"}}]})];
    let message = stream_compat_chunks(
        &model,
        &done_context(),
        keyed(&MockHttpClient::new()),
        &chunks,
    )
    .await;
    assert_eq![message.stop_reason, StopReason::Error];
    assert_eq!(
        message.error_message.as_deref(),
        Some("Stream ended without finish_reason")
    );
}

/// Port-added: without finish-reason support the stop reason is inferred —
/// `toolUse` when tool calls streamed, `stop` otherwise.
#[tokio::test]
async fn finish_reason_free_streams_infer_their_stop_reasons() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            supports_finish_reason: Some(false),
            ..ModelCompat::default()
        },
    );
    let context = Context {
        messages: done_context().messages,
        tools: Some(vec![common::lookup_tool()]),
        ..Context::default()
    };
    let tool_chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "lookup", "arguments": "{}"}}]}}]}),
        json!({"choices": [{"delta": {}}]}),
    ];
    let message = stream_compat_chunks(
        &model,
        &context,
        keyed(&MockHttpClient::new()),
        &tool_chunks,
    )
    .await;
    assert_eq![message.stop_reason, StopReason::ToolUse];

    let text_chunks = [json!({"choices": [{"delta": {"content": "Hi"}}]})];
    let message = stream_compat_chunks(
        &model,
        &done_context(),
        keyed(&MockHttpClient::new()),
        &text_chunks,
    )
    .await;
    assert_eq![message.stop_reason, StopReason::Stop];
}
// ---------------------------------------------------------------------------
// Remaining seam edges: deferred tools, grammar syntaxes, and stream arms
// ---------------------------------------------------------------------------

/// Port-added: with no tools in the context the Kimi deferred names match
/// nothing — no system message re-enters and the walk ends empty.
#[tokio::test]
async fn kimi_deferred_names_without_tools_send_no_system_message() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            deferred_tools_mode: Some(DeferredToolsMode::Kimi),
            ..ModelCompat::default()
        },
    );
    let mut context = bridged_context();
    context.messages[1] = common::tool_result_message(
        "call_1",
        vec![ToolResultBlock::Text(common::text_block("done"))],
        Some(vec!["late_tool".to_owned()]),
    );

    let payload = capture_compat(&model, &context, keyed(&MockHttpClient::new())).await;
    let messages = payload["messages"].as_array().expect("present");
    assert![
        messages
            .iter()
            .all(|message| message["role"] != json!("system"))
    ];
    assert_eq!(payload["tools"], json!([]));
}

/// Port-added: a grammar rejection while converting the deferred tool set
/// fails the request with the tool's rejection message.
#[tokio::test]
async fn a_deferred_grammar_rejection_fails_the_request() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            deferred_tools_mode: Some(DeferredToolsMode::Kimi),
            supports_openai_grammar_tools: Some(true),
            ..ModelCompat::default()
        },
    );
    let mut context = bridged_context();
    context.messages[0] = Message::Assistant(common::tool_call_assistant_message(
        "openai-completions",
        "custom",
        "custom-model",
        ToolCall {
            id: "call_1".to_owned(),
            name: "lookup".to_owned(),
            arguments: Map::new(),
            thought_signature: None,
            namespace: None,
        },
    ));
    context.messages[1] = common::tool_result_message(
        "call_1",
        vec![ToolResultBlock::Text(common::text_block("done"))],
        Some(vec!["broken".to_owned()]),
    );
    context.tools = Some(vec![Tool {
        name: "broken".to_owned(),
        description: "Broken grammar".to_owned(),
        parameters: json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        constrained_sampling: Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::Grammar {
                variants: std::collections::BTreeMap::new(),
            },
        )),
    }]);
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let result = stream(&model, &context, Some(&keyed(&mock))).result().await;
    assert_eq![result.stop_reason, StopReason::Error];
    let message = result.error_message.expect("the grammar rejection");
    assert![
        message.contains("no supported grammar variant"),
        "{message}"
    ];
    assert_eq!(mock.request_count(), 0);
}

/// Port-added: the regex grammar variant spells the `regex` syntax.
#[tokio::test]
async fn the_regex_grammar_variant_spells_the_regex_syntax() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            supports_openai_grammar_tools: Some(true),
            ..ModelCompat::default()
        },
    );
    let tool = Tool {
        constrained_sampling: Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::Grammar {
                variants: std::iter::once((GrammarFormat::OpenaiRegex, "^[a-z]+$".to_owned()))
                    .collect(),
            },
        )),
        ..grammar_tool()
    };
    let context = Context {
        messages: done_context().messages,
        tools: Some(vec![tool]),
        ..Context::default()
    };
    let payload = capture_compat(&model, &context, keyed(&MockHttpClient::new())).await;
    assert_eq!(payload["tools"][0]["type"], json!("custom"));
    assert_eq!(
        payload["tools"][0]["custom"]["format"]["grammar"]["syntax"],
        json!("regex")
    );
    assert_eq!(
        payload["tools"][0]["custom"]["format"]["grammar"]["definition"],
        json!("^[a-z]+$")
    );
}

/// Port-added: a non-JSON-object signature string rejects the legacy
/// encrypted parse the same way invalid JSON does.
#[tokio::test]
async fn a_non_object_legacy_signature_does_not_replay() {
    let payload = replayed_signature_payload(Some("5".to_owned())).await;
    assert![payload["messages"][0].get("reasoning_details").is_none()];
}

/// Port-added: streamed reasoning-detail entries that fail validation —
/// non-objects, malformed common fields, unknown types — are skipped; the
/// valid entries still merge.
#[tokio::test]
async fn invalid_streamed_reasoning_details_are_skipped() {
    let model = compat_model("https://custom.example/v1", ModelCompat::default());
    let chunks = [
        json!({"choices": [{"delta": {"reasoning_details": [5, "text", {"type": "reasoning.other"}]}}]}),
        json!({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "text": "kept", "id": 5}
        ]}}]}),
        json!({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "text": "kept", "signature": 5}
        ]}}]}),
        json!({"choices": [{"delta": {"reasoning_details": [
            {"type": "reasoning.text", "text": "ok", "id": "rs_1"}
        ]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ];
    let thinking = streamed_thinking(&model, &chunks, "the thinking block").await;
    let signature = thinking.thinking_signature.as_deref().expect("the details");
    let details: Vec<Value> = serde_json::from_str(signature).expect("the serialized details");
    assert_eq![
        details,
        vec![json!({"type": "reasoning.text", "text": "ok", "id": "rs_1"})]
    ];
}

/// Port-added: the low and max efforts spell their wire strings when no map
/// renames them.
#[tokio::test]
async fn the_low_and_max_efforts_spell_their_wire_strings() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            thinking_format: Some(ThinkingFormat::Deepseek),
            supports_reasoning_effort: Some(true),
            ..ModelCompat::default()
        },
    );
    let low = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Low),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert_eq!(low["reasoning_effort"], json!("low"));

    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            thinking_format: Some(ThinkingFormat::Deepseek),
            supports_reasoning_effort: Some(true),
            ..ModelCompat::default()
        },
    );
    let max = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::Max),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert_eq!(max["reasoning_effort"], json!("max"));
}

/// Port-added: a requested effort on a non-reasoning model spends no
/// thinking budget.
#[tokio::test]
async fn an_effort_on_a_non_reasoning_model_spends_no_budget() {
    let mut model = compat_model("https://custom.example/v1", ModelCompat::default());
    model.reasoning = false;
    let payload = capture_compat(
        &model,
        &done_context(),
        OpenAiCompletionsOptions {
            reasoning_effort: Some(ThinkingLevel::High),
            ..OpenAiCompletionsOptions::default()
        },
    )
    .await;
    assert![payload.get("thinking_token_budget").is_none()];
    assert![payload.get("reasoning_effort").is_none()];
}

/// Port-added: without effort support the Baseten branch sends only the
/// chat-template args.
#[tokio::test]
async fn baseten_without_effort_support_sends_only_the_args() {
    let args = [(
        "thinking".to_owned(),
        ChatTemplateKwargValue::Template(ChatTemplateVar {
            var: ThinkingTemplateVar::ThinkingEnabled,
            omit_when_off: None,
        }),
    )];
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            thinking_format: Some(ThinkingFormat::Baseten),
            supports_reasoning_effort: Some(false),
            chat_template_args: Some(args.iter().cloned().collect()),
            ..ModelCompat::default()
        },
    );
    let payload =
        capture_compat(&model, &done_context(), OpenAiCompletionsOptions::default()).await;
    assert_eq!(payload["chat_template_args"]["thinking"], json!(false));
    assert![payload.get("reasoning_effort").is_none()];
}

/// Port-added: a non-object `tool_calls` entry is skipped; a delta carrying
/// both a function and a custom body feeds neither buffer.
#[tokio::test]
async fn mixed_function_and_custom_deltas_feed_neither_buffer() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            supports_openai_grammar_tools: Some(true),
            ..ModelCompat::default()
        },
    );
    let context = Context {
        messages: done_context().messages,
        tools: Some(vec![grammar_tool()]),
        ..Context::default()
    };
    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": ["nope"]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "ln"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"text\":\"x\""}, "custom": {"input": "zz"}}]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let message =
        stream_compat_chunks(&model, &context, keyed(&MockHttpClient::new()), &chunks).await;
    assert_eq![message.stop_reason, StopReason::ToolUse];
    let block = message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("the streamed tool call");
    assert_eq![
        block.arguments,
        json!({"text": "x"}).as_object().expect("present").clone()
    ];
}

// ---------------------------------------------------------------------------
// Port-added: the reasoning-detail matrix, legacy-signature and custom-delta
// error edges the upstream suites reach only implicitly
// ---------------------------------------------------------------------------

/// Port-added: the consecutive-detail merge matrix — a numeric payload
/// replaces the text field, an absent field inserts, carried signatures ride,
/// and the common-field fill keeps the entry's format while filling id and
/// index, upstream's `appendOpenAIReasoningDetail` arms.
#[tokio::test]
async fn the_reasoning_detail_matrix_merges_its_field_shapes() {
    let chunks = [
        json!({
            "choices": [{"delta": {"reasoning_details": [
                { "type": "reasoning.text", "text": 5, "format": "json" },
            ]}}],
        }),
        json!({
            "choices": [{"delta": {"reasoning_details": [
                { "type": "reasoning.text", "text": "b" },
            ]}}],
        }),
        json!({
            "choices": [{"delta": {"reasoning_details": [
                { "type": "reasoning.text", "text": "c", "id": "x", "index": 2 },
            ]}}],
        }),
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ];
    let last = merged_detail_entry(&chunks).await;
    assert_eq![last["type"], json!("reasoning.text")];
    assert_eq![last["text"], json!("bc")];
    assert_eq![last["id"], json!("x")];
    assert_eq![last["index"], json!(2)];
    // The entry's own "format" field survives the fill.
    assert![last.get("format").is_none()];
}

/// Port-added: a carried signature survives the merge, upstream's
/// `appendOpenAIReasoningDetail` signature-carried arm.
#[tokio::test]
async fn the_merged_detail_inserts_its_absent_field_and_carries_signatures() {
    let chunks = [
        json!({
            "choices": [{"delta": {"reasoning_details": [
                { "type": "reasoning.text", "text": "a", "signature": "s1" },
            ]}}],
        }),
        json!({
            "choices": [{"delta": {"reasoning_details": [
                { "type": "reasoning.text", "text": "b", "format": "json" },
            ]}}],
        }),
        json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ];
    let message = stream_compat_chunks(
        &compat_model("https://custom.example/v1", ModelCompat::default()),
        &done_context(),
        keyed(&MockHttpClient::new()),
        &chunks,
    )
    .await;
    let _signature = message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::Thinking(block) => Some(block.thinking_signature.clone()),
            _ => None,
        })
        .flatten()
        .expect("the merged details signature");
    let last = merged_detail_entry(&chunks).await;
    assert_eq![last["text"], json!("ab")];
    assert_eq![last["format"], json!("json")];
    assert_eq![last["signature"], json!("s1")];
}

/// Port-added: legacy encrypted tool signatures missing their id or data do
/// not replay, upstream's `parseLegacyEncryptedReasoningDetail` guards.
#[tokio::test]
async fn malformed_legacy_encrypted_signatures_do_not_replay() {
    for signature in [
        json!({ "type": "reasoning.encrypted" }).to_string(),
        json!({ "type": "reasoning.encrypted", "id": "x" }).to_string(),
        json!({ "type": "reasoning.encrypted", "id": "", "data": "d" }).to_string(),
        json!({ "type": "reasoning.encrypted", "id": "x", "data": "" }).to_string(),
    ] {
        let assistant = common::tool_call_assistant_message(
            "openai-completions",
            "custom",
            "custom-model",
            ToolCall {
                id: "call_1".to_owned(),
                name: "ln".to_owned(),
                arguments: Map::new(),
                thought_signature: Some(signature.clone()),
                namespace: None,
            },
        );
        let context = Context {
            messages: vec![Message::Assistant(assistant)],
            ..done_context()
        };
        let mock = MockHttpClient::new();
        openai_mock_with(&mock, &[openai_done_chunk()]);
        let result = stream(
            &compat_model("https://custom.example/v1", ModelCompat::default()),
            &context,
            Some(&keyed(&mock)),
        )
        .result()
        .await;
        assert_eq![result.stop_reason, StopReason::Stop];
    }
}

/// Port-added: a custom delta carrying a stream index backfills the tracked
/// block, upstream's `ensureToolCall` index fill.
#[tokio::test]
async fn a_custom_input_delta_backfills_its_stream_index() {
    let chunks = [
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "custom": {"name": "ln", "input": "a"}}]}}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 7, "id": "c1", "custom": {"input": "b"}}]}}]}),
        json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
    ];
    let block = streamed_custom_tool_call(&chunks, "the streamed custom tool call").await;
    assert_eq![block.id, "c1"];
    assert_eq![block.arguments.get("text"), Some(&json!("ab"))];
}

/// Port-added: a cache marker walks past conversation messages whose content
/// arrays hold no text parts and past non-array content, upstream's
/// `addCacheControlToTextContent` misses.
#[tokio::test]
async fn cache_control_walks_past_its_missing_text_targets() {
    let model = compat_model(
        "https://custom.example/v1",
        ModelCompat {
            cache_control_format: Some(CacheControlFormat::Anthropic),
            ..ModelCompat::default()
        },
    );
    let tool_result = Message::ToolResult(ToolResultMessage {
        tool_call_id: "call_1".to_owned(),
        tool_name: "tool".to_owned(),
        content: vec![ToolResultBlock::Image(pi_ai::types::ImageContent {
            mime_type: "image/png".to_owned(),
            data: "aGk=".to_owned(),
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 3,
    });
    let context = Context {
        system_prompt: None,
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Blocks(vec![UserBlock::Image(pi_ai::types::ImageContent {
                    mime_type: "image/png".to_owned(),
                    data: "aGk=".to_owned(),
                })]),
                timestamp: 1,
            }),
            Message::Assistant(common::bare_assistant_message()),
            tool_result,
        ],
        tools: None,
    };
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let mut options = keyed(&mock);
    options.cache_retention = Some(CacheRetention::Long);

    let _ = stream(&model, &context, Some(&options)).result().await;
    let body = recorded_body(&mock);
    // The marker rides no conversation target: the user turn's content holds
    // only an image part and the assistant turn carries no text content.
    let stamps = body["messages"]
        .as_array()
        .expect("the messages")
        .iter()
        .filter(|message| message.get("cache_control").is_some())
        .count();
    assert![stamps == 0, "no target carries the marker"];
}

// ---------------------------------------------------------------------------
// AI-gateway binding sentinel (upstream cloudflare-ai-binding.test.ts, the
// portable case)
// ---------------------------------------------------------------------------

/// The portable half of `test/cloudflare-ai-binding.test.ts`'s end-to-end
/// header contract: the gateway-binding sentinel in `cf-aig-authorization`
/// satisfies the request-auth check, and the explicit `null` caller headers
/// remove the adapter's own `Authorization` and any `x-api-key` before
/// dispatch, so neither rides the wire.
///
/// Restatement: the Workers-runtime `AiBinding` fetch wrapper upstream wraps
/// has no Rust counterpart; the custom seam client plays the binding's
/// canned 400, and the wire assertions read the recorded request.
#[tokio::test]
async fn the_gateway_binding_sentinel_keeps_placeholder_auth_off_the_wire() {
    use std::collections::BTreeMap;

    let mock = MockHttpClient::new();
    mock.on(|_request| true).respond(
        pi_ai::http::MockResponse::status(400)
            .with_body(r#"{"error":{"type":"bad_request","message":"stubbed"}}"#),
    );
    let mut model = common::openai_catalog_model("openai", "gpt-4o-mini");
    model.base_url = "https://workers-binding.ai/ai-gateway/gateways/my-gateway/openai".to_owned();

    let options = OpenAiCompletionsOptions {
        transport_options: TransportOptions {
            http_client: Some(std::sync::Arc::new(mock.clone())),
            ..TransportOptions::default()
        },
        headers: Some(BTreeMap::from([
            (
                "cf-aig-authorization".to_owned(),
                Some("Bearer cloudflare-gateway-binding".to_owned()),
            ),
            ("Authorization".to_owned(), None),
            ("x-api-key".to_owned(), None),
        ])),
        max_retries: Some(0),
        ..OpenAiCompletionsOptions::default()
    };

    let result = stream(&model, &done_context(), Some(&options))
        .result()
        .await;
    assert_eq!(result.stop_reason, StopReason::Error);

    let recorded = mock.recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(
        recorded[0].url,
        "https://workers-binding.ai/ai-gateway/gateways/my-gateway/openai/chat/completions"
    );
    let names: Vec<&String> = recorded[0].headers.iter().map(|(name, _)| name).collect();
    assert!(
        !names
            .iter()
            .any(|name| name.eq_ignore_ascii_case("authorization"))
    );
    assert!(
        !names
            .iter()
            .any(|name| name.eq_ignore_ascii_case("x-api-key"))
    );
}
