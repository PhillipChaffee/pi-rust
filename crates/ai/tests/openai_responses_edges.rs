//! The OpenAI Responses seam edges the upstream suites reach only
//! implicitly: transport-error mapping, SDK error message shapes, malformed
//! SSE, stream-lifecycle failures, credential setup, in-stream error forms,
//! and the payload/response hooks — for the openai-responses and
//! azure-openai-responses wires. Behaviors upstream exercises at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`; tests pinning port seams the
//! upstream suites reach only implicitly are marked in their doc comments.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use bytes::Bytes;
use pi_ai::api::azure_openai_responses::{self, AzureOpenAiResponsesOptions};
use pi_ai::api::openai_responses::{self, OpenAiResponsesOptions};
use pi_ai::http::{
    HttpByteStream, HttpClient, HttpError, HttpRequest, HttpResponse, MockHttpClient,
};
use pi_ai::types::{
    Context, Message, Modality, Model, SimpleStreamOptions, StopReason, ThinkingLevel, Tool,
    ToolChoice, TransportOptions, UserContent, UserMessage,
};
use serde_json::{Value, json};

mod common;
use common::{
    builtin_model, openai_responses_completed_event, openai_responses_mock_with, recorded_body,
};

/// The hello context every edges stream sends.
fn done_context() -> Context {
    Context {
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Hello".to_owned()),
            timestamp: 1,
        })],
        ..Context::default()
    }
}

/// The "see" tool result the image-shape cases replay, answering `call_1`.
fn tool_result_content(
    content: Vec<pi_ai::types::ToolResultBlock>,
    timestamp: i64,
) -> pi_ai::types::ToolResultMessage {
    pi_ai::types::ToolResultMessage {
        tool_call_id: "call_1".to_owned(),
        tool_name: "see".to_owned(),
        content,
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp,
    }
}

/// Stream the messages through the image-input model variant and read the
/// tool output the request carries, the shape the tool-output cases read.
async fn tool_result_output(input: Vec<Modality>, messages: Vec<Message>) -> Value {
    let model = builtin_model("openai", "gpt-5.4");
    let model = Model { input, ..model };
    let context = Context {
        system_prompt: None,
        messages,
        tools: None,
    };
    let mock = done_mock();
    let _ = openai_responses::stream(&model, &context, Some(&keyed(&mock)))
        .result()
        .await;
    recorded_body(&mock)["input"]
        .as_array()
        .and_then(|entries| entries.last())
        .and_then(|entry| entry.get("output"))
        .cloned()
        .expect("the tool output")
}

/// The keyed options against the given mock.
fn keyed(mock: &MockHttpClient) -> OpenAiResponsesOptions {
    common::keyed_openai_responses_options(mock)
}

/// The mounted completion run every settled edges stream answers with.
fn done_mock() -> MockHttpClient {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    mock
}

/// A canned-seam client for the mid-stream failure edges: the body opens
/// with one readable frame and then fails with the stored error, so the SSE
/// read fails in flight.
#[derive(Debug)]
struct InterruptedAfterFrame {
    first: Bytes,
    then: HttpError,
}

impl HttpClient for InterruptedAfterFrame {
    fn execute(
        &self,
        _request: HttpRequest,
    ) -> pi_ai::http::BoxHttpFuture<Result<HttpResponse, HttpError>> {
        let chunks = vec![Ok(self.first.clone()), Err(self.then.clone())];
        let response = HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: HttpByteStream::from_chunks(chunks),
        };
        Box::pin(async move { Ok(response) })
    }
}

fn interrupted_options(first: Bytes, then: HttpError) -> OpenAiResponsesOptions {
    OpenAiResponsesOptions {
        transport_options: TransportOptions {
            http_client: Some(std::sync::Arc::new(InterruptedAfterFrame { first, then })),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..OpenAiResponsesOptions::default()
    }
}

/// The one event a mid-stream body streams before the seam failure lands.
fn created_frame() -> Bytes {
    Bytes::from(format!(
        "data: {}\n\n",
        json!({ "type": "response.created", "response": { "id": "resp_seamed" } })
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
    let mock = done_mock();
    let model = builtin_model("openai", "gpt-5.4");
    let options = OpenAiResponsesOptions {
        transport_options: common::mock_transport(&mock),
        ..OpenAiResponsesOptions::default()
    };

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: openai")
    );
    assert_eq!(mock.request_count(), 0);
}

/// Port-added: `stream_simple` without any credential fails through the
/// setup-error stream instead of dispatching.
#[tokio::test]
async fn stream_simple_without_a_credential_fails_before_dispatch() {
    let mock = done_mock();
    let model = builtin_model("openai", "gpt-5.4");

    let result = openai_responses::stream_simple(&model, &done_context(), None)
        .result()
        .await;

    common::assert_setup_error_without_dispatch(&result, &mock);
}

/// Header-owned authorization stands in for a key: the request dispatches
/// with the caller's header pair.
#[tokio::test]
async fn header_owned_authorization_dispatches_without_a_key() {
    let mock = done_mock();
    let model = builtin_model("openai", "gpt-5.4");
    let options = OpenAiResponsesOptions {
        transport_options: common::mock_transport(&mock),
        headers: Some(
            std::iter::once((
                "Authorization".to_owned(),
                Some("Bearer caller-token".to_owned()),
            ))
            .collect(),
        ),
        ..OpenAiResponsesOptions::default()
    };

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert!(
        common::recorded_header_values(&mock, "Authorization")
            .iter()
            .any(|value| value == "Bearer caller-token")
    );
}

/// The Cloudflare gateway's credential header stands in for a key the same
/// way the Authorization pair does.
#[tokio::test]
async fn gateway_owned_authorization_dispatches_without_a_key() {
    let mock = done_mock();
    let model = builtin_model("openai", "gpt-5.4");
    let options = OpenAiResponsesOptions {
        transport_options: common::mock_transport(&mock),
        headers: Some(
            std::iter::once((
                "cf-aig-authorization".to_owned(),
                Some("gateway-token".to_owned()),
            ))
            .collect(),
        ),
        ..OpenAiResponsesOptions::default()
    };

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    let recorded = mock.recorded();
    assert!(
        recorded[0].headers.iter().any(|(name, value)| name
            .eq_ignore_ascii_case("cf-aig-authorization")
            && value == "gateway-token"),
        "got: {:?}",
        recorded[0].headers
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
        mock.on(|request| request.url.contains("/responses"))
            .respond_fn(move |_request| {
                let error = error.clone();
                async move { Err(error) }
            });
        let model = builtin_model("openai", "gpt-5.4");
        let options = keyed(&mock);

        let result = openai_responses::stream(&model, &done_context(), Some(&options))
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
    let mock = done_mock();
    let token = tokio_util::sync::CancellationToken::new();
    token.cancel();
    let model = builtin_model("openai", "gpt-5.4");
    let options = OpenAiResponsesOptions {
        transport_options: TransportOptions {
            http_client: Some(std::sync::Arc::new(mock.clone())),
            signal: Some(token),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..OpenAiResponsesOptions::default()
    };

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("Request aborted"));
}

// ---------------------------------------------------------------------------
// SDK error message shapes
// ---------------------------------------------------------------------------

/// Port-added: the raw-body, parsed-body, and missing-body variants of the
/// SDK's `{status} {body}` error message, under the responses wire's provider
/// prefix.
#[tokio::test]
async fn sdk_error_messages_cover_raw_parsed_and_missing_bodies() {
    // A raw text body rides the message verbatim.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/responses"))
        .respond(pi_ai::http::MockResponse::status(502).with_body("Bad Gateway"));
    assert_error_message(&mock, "OpenAI API error (502): 502 Bad Gateway").await;

    // A missing body falls back to the SDK's no-body form.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/responses"))
        .respond(pi_ai::http::MockResponse::status(503));
    assert_error_message(&mock, "OpenAI API error (503): 503 status code (no body)").await;

    // A parsed JSON body serializes into the message.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/responses"))
        .respond(pi_ai::http::json_response(
            400,
            &json!({ "error": { "message": "bad shape" } }),
        ));
    assert_error_message(
        &mock,
        "OpenAI API error (400): 400 {\"error\":{\"message\":\"bad shape\"}}",
    )
    .await;
}

/// Stream against the mounted mock and pin the settled error message.
async fn assert_error_message(mock: &MockHttpClient, expected: &str) {
    let model = builtin_model("openai", "gpt-5.4");
    let options = keyed(mock);

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some(expected));
}

// ---------------------------------------------------------------------------
// Malformed SSE surfaces
// ---------------------------------------------------------------------------

/// A data frame whose JSON does not parse fails the stream with the parse
/// message naming the event data and raw lines.
#[tokio::test]
async fn malformed_sse_frames_fail_with_the_parse_message() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/responses"))
        .respond(
            pi_ai::http::MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body("data: {not json}\n\ndata: [DONE]\n\n"),
        );
    let model = builtin_model("openai", "gpt-5.4");
    let options = keyed(&mock);

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the parse error");
    assert!(
        message.starts_with("Could not parse OpenAI Responses SSE event:"),
        "got: {message}"
    );
    assert!(message.contains("data={not json}"), "got: {message}");
    assert!(message.contains("raw="), "got: {message}");
}

// ---------------------------------------------------------------------------
// Mid-stream seam errors and in-stream error forms
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
        let options = interrupted_options(created_frame(), error);
        let model = builtin_model("openai", "gpt-5.4");

        let result = openai_responses::stream(&model, &done_context(), Some(&options))
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some(expected));
    }
}

/// The wire's in-stream `error` event fails the run with the code/message
/// composition, upstream's `Error Code` throw.
#[tokio::test]
async fn the_in_stream_error_event_fails_with_the_code_and_message() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/responses"))
        .respond(
            pi_ai::http::MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(format!(
                    "data: {}\n\n",
                    json!({
                        "type": "error",
                        "code": "insufficient_quota",
                        "message": "quota exceeded",
                    })
                )),
        );
    let model = builtin_model("openai", "gpt-5.4");
    let options = keyed(&mock);

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Error Code insufficient_quota: quota exceeded")
    );
}

/// A terminal status outside the provider's enum fails with the unhandled
/// stop reason, the exhaustive `never` branch.
#[tokio::test]
async fn an_unhandled_terminal_status_fails_the_stream() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/responses"))
        .respond(
            pi_ai::http::MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(format!(
                    "data: {}\n\n",
                    json!({
                        "type": "response.completed",
                        "response": { "id": "resp_bad", "status": "queued2" },
                    })
                )),
        );
    let model = builtin_model("openai", "gpt-5.4");
    let options = keyed(&mock);

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Unhandled stop reason: queued2")
    );
}

// ---------------------------------------------------------------------------
// Payload shaping seams
// ---------------------------------------------------------------------------

/// Port-added: the on-payload hook can replace the request payload, the
/// seam-side port of upstream's `onPayload` mutation.
#[tokio::test]
async fn the_payload_hook_can_replace_the_request_payload() {
    let mock = done_mock();
    let model = builtin_model("openai", "gpt-5.4");
    let mut options = keyed(&mock);
    options.transport_options.on_payload = Some(pi_ai::types::OnPayload::new(
        |_payload, _model| {
            Box::pin(async {
                Some(json!({
                    "model": "replaced-model",
                    "input": [{ "role": "user", "content": [{ "type": "input_text", "text": "replaced" }] }],
                    "stream": true,
                }))
            })
        },
    ));

    let _ = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    let body = recorded_body(&mock);
    assert_eq!(body["model"], json!("replaced-model"));
    assert_eq!(body["input"][0]["content"][0]["text"], json!("replaced"));
}

/// Port-added: the response hook observes the settled status.
#[tokio::test]
async fn the_response_hook_observes_the_response_status() {
    let mock = done_mock();
    let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let slot = std::sync::Arc::clone(&observed);
    let model = builtin_model("openai", "gpt-5.4");
    let options = OpenAiResponsesOptions {
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
        ..OpenAiResponsesOptions::default()
    };

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
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
// Azure OpenAI Responses edges
// ---------------------------------------------------------------------------

/// The keyed azure options an edges stream sends, with the base URL a
/// non-empty deployment base names.
fn azure_keyed(mock: &MockHttpClient, base_url: &str) -> AzureOpenAiResponsesOptions {
    AzureOpenAiResponsesOptions {
        transport_options: common::mock_transport(mock),
        api_key: Some("test-key".to_owned()),
        azure_base_url: Some(base_url.to_owned()),
        ..AzureOpenAiResponsesOptions::default()
    }
}

/// Port-added: the azure wire without a credential fails before dispatching.
#[tokio::test]
async fn azure_stream_without_a_credential_fails_before_dispatch() {
    let mock = done_mock();
    let model = builtin_model("azure-openai-responses", "gpt-4o-mini");
    let options = AzureOpenAiResponsesOptions {
        transport_options: common::mock_transport(&mock),
        ..AzureOpenAiResponsesOptions::default()
    };

    let result = azure_openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: azure-openai-responses")
    );
    assert_eq!(mock.request_count(), 0);
}

/// Port-added: with no base URL named anywhere, the run fails with the
/// base-url requirement before dispatching.
#[tokio::test]
async fn azure_stream_without_a_base_url_fails_before_dispatch() {
    let mock = done_mock();
    // The azure gpt-4o-mini catalog entry carries an empty base URL and the
    // options name none.
    let model = builtin_model("azure-openai-responses", "gpt-4o-mini");
    let options = AzureOpenAiResponsesOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        ..AzureOpenAiResponsesOptions::default()
    };

    let result = azure_openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the base-url error");
    assert!(
        message.contains("Azure OpenAI base URL is required"),
        "got: {message}"
    );
    assert_eq!(mock.request_count(), 0);
}

/// Port-added: a non-2xx body surfaces under the azure prefix the way the
/// openai wire's does.
#[tokio::test]
async fn azure_non_2xx_bodies_surface_under_the_azure_prefix() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/responses"))
        .respond(pi_ai::http::json_response(
            403,
            &json!({ "error": "blocked by gateway WAF" }),
        ));
    let model = builtin_model("azure-openai-responses", "gpt-4o-mini");

    let result = azure_openai_responses::stream(
        &model,
        &done_context(),
        Some(&azure_keyed(&mock, "https://my-resource.openai.azure.com")),
    )
    .result()
    .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the failure message");
    assert!(
        message.contains("Azure OpenAI API error (403)"),
        "got: {message}"
    );
    assert!(message.contains("blocked by gateway WAF"), "got: {message}");
}

// ---------------------------------------------------------------------------
// Port-added: the shared event-application edges the upstream suites reach
// only implicitly
// ---------------------------------------------------------------------------

/// Port-added: an in-stream error event without a code or message spells the
/// wire's "undefined" fallbacks, upstream's `applyEvents` error arm.
#[tokio::test]
async fn the_error_event_spells_undefined_for_absent_fields() {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[json!({ "type": "error" })]);
    let model = builtin_model("openai", "gpt-5.4");
    let options = keyed(&mock);

    let result = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Error Code undefined: undefined")
    );
}

/// Port-added: the failed terminal's error detail matrix — a truthy
/// non-object error names "unknown"/"no message", an empty reason names the
/// incomplete wording, and no details at all name the unknown error —
/// upstream's `applyEvents` response.failed arm over `isTruthy`.
#[tokio::test]
async fn the_failed_terminal_reports_its_truthy_matrix() {
    let model = builtin_model("openai", "gpt-5.4");
    let options = keyed(&MockHttpClient::new());
    let _ = options; // each phase builds its own options

    // A truthy non-object error: the code/message lookups fall back.
    let mock = MockHttpClient::new();
    openai_responses_mock_with(
        &mock,
        &[json!({
            "type": "response.failed",
            "response": { "id": "resp_1", "status": "failed", "error": 5 },
        })],
    );
    let result = openai_responses::stream(&model, &done_context(), Some(&keyed(&mock)))
        .result()
        .await;
    assert_eq![result.stop_reason, StopReason::Error];
    assert_eq![result.error_message.as_deref(), Some("unknown: no message")];

    // No error object, but an incomplete reason names the wording.
    let mock = MockHttpClient::new();
    openai_responses_mock_with(
        &mock,
        &[json!({
            "type": "response.failed",
            "response": {
                "id": "resp_2",
                "status": "incomplete",
                "incomplete_details": { "reason": "max_output_tokens" },
            },
        })],
    );
    let result = openai_responses::stream(&model, &done_context(), Some(&keyed(&mock)))
        .result()
        .await;
    assert_eq![result.stop_reason, StopReason::Error];
    assert_eq![
        result.error_message.as_deref(),
        Some("incomplete: max_output_tokens")
    ];

    // Neither error nor details: the unknown wording.
    let mock = MockHttpClient::new();
    openai_responses_mock_with(
        &mock,
        &[json!({
            "type": "response.failed",
            "response": { "id": "resp_3", "status": "failed" },
        })],
    );
    let result = openai_responses::stream(&model, &done_context(), Some(&keyed(&mock)))
        .result()
        .await;
    assert_eq![result.stop_reason, StopReason::Error];
    assert_eq![
        result.error_message.as_deref(),
        Some("Unknown error (no error details in response)")
    ];
}

/// Port-added: a custom tool call's raw input streams through the grammar
/// buffer under the grammar property, closes with the done event's input,
/// and settles as a tool call, upstream's `custom_tool_call_input` event
/// family.
#[tokio::test]
async fn the_custom_tool_input_streams_through_the_shared_buffer() {
    let grammar_tool = Tool {
        name: "ln".to_owned(),
        description: "List a file".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"],
        }),
        constrained_sampling: Some(pi_ai::types::ConstrainedSamplingSetting::Config(
            pi_ai::types::ConstrainedSamplingConfig::Grammar {
                variants: std::iter::once((
                    pi_ai::types::GrammarFormat::OpenaiLark,
                    "start: /[a-z]+/".to_owned(),
                ))
                .collect(),
            },
        )),
    };
    let model = builtin_model("openai", "gpt-5.4");
    let model = Model {
        compat: Some(pi_ai::types::ModelCompat {
            supports_openai_grammar_tools: Some(true),
            ..pi_ai::types::ModelCompat::default()
        }),
        ..model
    };
    let mut context = done_context();
    context.tools = Some(vec![grammar_tool]);

    let mock = MockHttpClient::new();
    openai_responses_mock_with(
        &mock,
        &[
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "custom_tool_call",
                    "id": "item_1",
                    "call_id": "call_1",
                    "name": "ln",
                    "input": "",
                },
            }),
            json!({
                "type": "response.custom_tool_call_input.delta",
                "output_index": 0,
                "delta": "he",
            }),
            json!({
                "type": "response.custom_tool_call_input.delta",
                "output_index": 0,
                "delta": "llo",
            }),
            json!({
                "type": "response.custom_tool_call_input.done",
                "output_index": 0,
                "input": "hello",
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "custom_tool_call",
                    "id": "item_1",
                    "call_id": "call_1",
                    "name": "ln",
                    "input": "hello",
                    "status": "completed",
                },
            }),
            openai_responses_completed_event(),
        ],
    );
    let options = keyed(&mock);

    let result = openai_responses::stream(&model, &context, Some(&options))
        .result()
        .await;

    // A run ending on an open tool call settles as tool use.
    assert_eq![result.stop_reason, StopReason::ToolUse];
    let call = result.content.iter().find_map(|block| match block {
        pi_ai::types::AssistantBlock::ToolCall(call) => Some(call),
        _ => None,
    });
    let call = call.expect("the custom tool call");
    assert_eq![call.name, "ln"];
    assert_eq![call.arguments.get("text"), Some(&json!("hello"))];
}

/// Port-added: `stream_simple` maps the `none` tool choice and the thinking
/// levels through `thinkingLevelMap`, upstream's `streamSimple` option
/// mapping. The map sends the minimal level to the wire's "low" and leaves
/// `max` mapped, so the clamp keeps it, upstream's catalog matrix.
#[tokio::test]
async fn the_responses_simple_options_ride_their_request_fields() {
    let model = builtin_model("openai", "gpt-5.4");
    let model = Model {
        id: "gpt-level-test".to_owned(),
        name: "Level Test".to_owned(),
        reasoning: true,
        thinking_level_map: Some(
            [
                (
                    pi_ai::types::ModelThinkingLevel::Minimal,
                    Some("low".to_owned()),
                ),
                (
                    pi_ai::types::ModelThinkingLevel::Max,
                    Some("xhigh".to_owned()),
                ),
            ]
            .into_iter()
            .collect(),
        ),
        ..model
    };

    for (choice, expected) in [(ToolChoice::Auto, "auto"), (ToolChoice::None, "none")] {
        let mock = done_mock();
        let options = SimpleStreamOptions {
            transport_options: TransportOptions {
                http_client: Some(std::sync::Arc::new(mock.clone())),
                ..TransportOptions::default()
            },
            api_key: Some("test-key".to_owned()),
            tool_choice: Some(choice),
            ..SimpleStreamOptions::default()
        };
        let _ = openai_responses::stream_simple(&model, &done_context(), Some(&options))
            .result()
            .await;
        assert_eq![recorded_body(&mock)["tool_choice"], json!(expected)];
    }

    for (level, expected) in [
        (ThinkingLevel::Minimal, "low"),
        (ThinkingLevel::Low, "low"),
        (ThinkingLevel::Medium, "medium"),
        (ThinkingLevel::High, "high"),
        (ThinkingLevel::Max, "xhigh"),
    ] {
        let mock = done_mock();
        let options = SimpleStreamOptions {
            transport_options: common::mock_transport(&mock),
            api_key: Some("test-key".to_owned()),
            reasoning: Some(level),
            ..SimpleStreamOptions::default()
        };
        let _ = openai_responses::stream_simple(&model, &done_context(), Some(&options))
            .result()
            .await;
        let body = recorded_body(&mock);
        assert_eq![
            body["reasoning"]["effort"],
            json!(expected),
            "level {level:?}"
        ];
    }
}

/// Port-added: an options header carrying `None` removes the default header,
/// upstream's header-merge `delete` arm.
#[tokio::test]
async fn a_responses_options_header_removal_empties_the_default() {
    let mock = done_mock();
    let model = builtin_model("openai", "gpt-5.4");
    let mut options = keyed(&mock);
    options.headers = Some(pi_ai::types::ProviderHeaders::from_iter([(
        "User-Agent".to_owned(),
        None,
    )]));

    let _ = openai_responses::stream(&model, &done_context(), Some(&options))
        .result()
        .await;

    let request = &mock.recorded()[0];
    let user_agent = request
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("user-agent"));
    assert![user_agent.is_none(), "the default header was removed"];
}

/// Port-added: tool results spell their image-capable shapes — `input_image`
/// parts on an image-capable model and the transform's omitted-image
/// placeholder without vision, upstream's `convertToolResultOutput` plus the
/// `NON_VISION_TOOL_IMAGE_PLACEHOLDER` rewrite.
#[tokio::test]
async fn the_responses_tool_result_output_spells_its_image_shapes() {
    let image = pi_ai::types::ImageContent {
        mime_type: "image/png".to_owned(),
        data: "aGk=".to_owned(),
    };
    // An image-seeing model sends the input_text part followed by the
    // input_image part.
    let assistant = common::tool_call_assistant_message(
        "openai-responses",
        "openai",
        "gpt-5.4",
        pi_ai::types::ToolCall {
            id: "call_1".to_owned(),
            name: "see".to_owned(),
            arguments: serde_json::Map::new(),
            thought_signature: None,
            namespace: None,
        },
    );
    let tool_result = tool_result_content(
        vec![
            pi_ai::types::ToolResultBlock::Text(pi_ai::types::TextContent {
                text: "summary".to_owned(),
                text_signature: None,
            }),
            pi_ai::types::ToolResultBlock::Image(image.clone()),
        ],
        2,
    );
    let output = tool_result_output(
        vec![Modality::Text, Modality::Image],
        vec![
            Message::User(UserMessage {
                content: UserContent::Text("look".to_owned()),
                timestamp: 1,
            }),
            Message::Assistant(assistant.clone()),
            Message::ToolResult(tool_result),
        ],
    )
    .await;
    let parts = output.as_array().expect("the image parts array");
    assert_eq![parts[0]["type"], json!("input_text")];
    assert_eq![parts[1]["type"], json!("input_image")];
    assert![
        parts[1]["image_url"]
            .as_str()
            .is_some_and(|url| url.starts_with("data:image/png;base64,"))
    ];

    // A text-only model sends the attached-image placeholder for an
    // image-only result.
    let image_only = tool_result_content(vec![pi_ai::types::ToolResultBlock::Image(image)], 2);
    let output = tool_result_output(
        vec![Modality::Text],
        vec![
            Message::User(UserMessage {
                content: UserContent::Text("look".to_owned()),
                timestamp: 1,
            }),
            Message::Assistant(assistant),
            Message::ToolResult(image_only),
        ],
    )
    .await;
    assert!(
        output.as_str().is_some_and(|text| {
            text.contains("(tool image omitted: model does not support images)")
        }),
        "got: {output}"
    );
}
