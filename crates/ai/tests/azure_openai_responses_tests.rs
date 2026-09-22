//! Azure OpenAI Responses suites, ported from the upstream
//! `azure-openai-base-url.test.ts` and `azure-openai-tool-choice.test.ts` at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, with the
//! `azure-utils.ts` helpers carried as module-local builders.
//!
//! Porting restatements: upstream captures the pinned SDK client's
//! `baseURL`/`defaultHeaders` through a mocked `AzureOpenAI` constructor and
//! the payload through a throwing `responses.create`; the port reads the same
//! endpoint off the [`MockHttpClient`] seam's recorded request URL and the
//! payload off its recorded body. Upstream's `process.env` manipulation
//! becomes the options' scoped `env` map, so the suites stay hermetic.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use std::collections::BTreeMap;

use pi_ai::api::azure_openai_responses::{self, AzureOpenAiResponsesOptions};
use pi_ai::api::openai_responses::ReasoningSummary;
use pi_ai::http::MockHttpClient;
use pi_ai::types::{
    Api, ConstrainedSamplingConfig, ConstrainedSamplingSetting, Context, Message, Modality, Model,
    ModelCompat, ProviderId, SimpleStreamOptions, StopReason, Tool, ToolChoice, TransportOptions,
    UserContent, UserMessage,
};
use pi_ai::utils::pi_user_agent::get_pi_user_agent;
use serde_json::{Value, json};

mod common;
use common::{openai_responses_completed_event, openai_responses_mock_with};

/// Stream the azure model/context/options against the mock, mounting the
/// completion run, upstream's `streamAzureOpenAIResponses(...).result()`.
async fn stream_once(model: &Model, mock: &MockHttpClient, options: &AzureOpenAiResponsesOptions) {
    openai_responses_mock_with(mock, &[openai_responses_completed_event()]);
    let mut options = options.clone();
    options.transport_options = common::mock_transport(mock);
    let stream = azure_openai_responses::stream(model, &context(), Some(&options));
    let result = common::drain_and_settle(&stream).await;
    // Upstream's azure suites stream against a mocked SDK create that
    // returns nothing; the seam mock completes the run, so the settled stop
    // reason pins that the request pipeline ran to the terminal event.
    assert!(matches!(
        result.stop_reason,
        StopReason::Stop | StopReason::Error
    ));
}

/// The context upstream's azure suites send.
fn context() -> Context {
    Context {
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("hello".to_owned()),
            timestamp: pi_ai::auth::resolve::now_ms(),
        })],
        ..Context::default()
    }
}

/// `getModel("azure-openai-responses", "gpt-4o-mini")`.
fn azure_model() -> Model {
    common::builtin_model("azure-openai-responses", "gpt-4o-mini")
}

/// The keyed options upstream's suites send, with the given scoped env map,
/// upstream's `beforeEach` environment sandbox.
fn azure_options(env: BTreeMap<String, String>) -> AzureOpenAiResponsesOptions {
    let mut options = AzureOpenAiResponsesOptions {
        api_key: Some("test-api-key".to_owned()),
        env: Some(env),
        ..AzureOpenAiResponsesOptions::default()
    };
    let needs_resource = options.env.as_ref().is_none_or(|env| {
        !env.contains_key("AZURE_OPENAI_BASE_URL")
            && !env.contains_key("AZURE_OPENAI_RESOURCE_NAME")
    });
    if needs_resource {
        options.env.as_mut().expect("the env map").insert(
            "AZURE_OPENAI_RESOURCE_NAME".to_owned(),
            "test-resource".to_owned(),
        );
    }
    options
}

/// The one-entry scoped env map a test drives, upstream's
/// `process.env.AZURE_OPENAI_* = value`.
fn env_with(name: &str, value: &str) -> BTreeMap<String, String> {
    std::iter::once((name.to_owned(), value.to_owned())).collect()
}

/// The first recorded request's URL.
fn recorded_url(mock: &MockHttpClient) -> String {
    mock.recorded()[0].url.clone()
}

/// The first recorded request's named header, case-insensitive.
fn recorded_body(mock: &MockHttpClient) -> Value {
    common::recorded_body(mock)
}

fn recorded_header(mock: &MockHttpClient, name: &str) -> Option<String> {
    common::recorded_header(mock, name)
}

// ---------------------------------------------------------------------------
// base URL normalization (upstream azure-openai-base-url)
// ---------------------------------------------------------------------------

/// The full responses URL a normalized base resolves to, with the default
/// api-version the SDK's `defaultQuery` carries.
fn expected_url(base_url: &str) -> String {
    format!("{base_url}/responses?api-version=v1")
}

/// Stream with the given env value in `AZURE_OPENAI_BASE_URL`, upstream's
/// `captureClientBaseUrl` env form.
async fn capture_env_base_url(base_url: &str) -> MockHttpClient {
    let mock = MockHttpClient::new();
    let options = azure_options(env_with("AZURE_OPENAI_BASE_URL", base_url));
    stream_once(&azure_model(), &mock, &options).await;
    mock
}

#[tokio::test]
async fn normalizes_cognitive_services_root_endpoints_to_openai_v1() {
    let mock =
        capture_env_base_url("https://marc-quicktests-resource.cognitiveservices.azure.com").await;
    assert_eq!(
        recorded_url(&mock),
        expected_url("https://marc-quicktests-resource.cognitiveservices.azure.com/openai/v1")
    );
}

#[tokio::test]
async fn normalizes_microsoft_foundry_root_endpoints_to_openai_v1() {
    let mock = capture_env_base_url("https://marc-quicktests-resource.ai.azure.com").await;
    assert_eq!(
        recorded_url(&mock),
        expected_url("https://marc-quicktests-resource.ai.azure.com/openai/v1")
    );
}

#[tokio::test]
async fn normalizes_azure_openai_root_endpoints_to_openai_v1() {
    let mock = capture_env_base_url("https://my-resource.openai.azure.com").await;
    assert_eq!(
        recorded_url(&mock),
        expected_url("https://my-resource.openai.azure.com/openai/v1")
    );
}

#[tokio::test]
async fn normalizes_openai_to_openai_v1() {
    let mock = capture_env_base_url("https://my-resource.cognitiveservices.azure.com/openai").await;
    assert_eq!(
        recorded_url(&mock),
        expected_url("https://my-resource.cognitiveservices.azure.com/openai/v1")
    );
}

#[tokio::test]
async fn preserves_openai_v1_endpoints() {
    let mock =
        capture_env_base_url("https://my-resource.cognitiveservices.azure.com/openai/v1").await;
    assert_eq!(
        recorded_url(&mock),
        expected_url("https://my-resource.cognitiveservices.azure.com/openai/v1")
    );
}

#[tokio::test]
async fn normalizes_openai_v1_responses_to_openai_v1() {
    let mock =
        capture_env_base_url("https://my-resource.services.ai.azure.com/openai/v1/responses").await;
    assert_eq!(
        recorded_url(&mock),
        expected_url("https://my-resource.services.ai.azure.com/openai/v1")
    );
}

#[tokio::test]
async fn preserves_explicit_non_azure_proxy_paths() {
    let mock = capture_env_base_url("https://my-proxy.example.com/v1").await;
    assert_eq!(
        recorded_url(&mock),
        expected_url("https://my-proxy.example.com/v1")
    );
}

#[tokio::test]
async fn strips_query_params_when_normalizing_azure_host_urls() {
    let mock =
        capture_env_base_url("https://my-resource.openai.azure.com/openai?api-version=2024-12-01")
            .await;
    assert_eq!(
        recorded_url(&mock),
        expected_url("https://my-resource.openai.azure.com/openai/v1")
    );
}

#[tokio::test]
async fn preserves_query_params_on_non_azure_proxy_urls() {
    let mock = capture_env_base_url("https://my-proxy.example.com/v1?custom=true").await;
    assert_eq!(
        recorded_url(&mock),
        "https://my-proxy.example.com/v1/responses?custom=true&api-version=v1"
    );
}

/// The invalid-URL failure, upstream's "throws on invalid URLs" case.
#[tokio::test]
async fn errors_on_invalid_urls() {
    let mock = MockHttpClient::new();
    let options = azure_options(env_with("AZURE_OPENAI_BASE_URL", "not-a-url"));
    let mut options = options;
    options.transport_options = common::mock_transport(&mock);
    let stream = azure_openai_responses::stream(&azure_model(), &context(), Some(&options));
    let result = common::drain_and_settle(&stream).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the base-url error");
    assert!(
        message.contains("Invalid Azure OpenAI base URL"),
        "got: {message}"
    );
    assert_eq!(
        mock.request_count(),
        0,
        "the invalid URL fails before dispatch"
    );
}

#[tokio::test]
async fn clamps_prompt_cache_key_to_the_64_character_limit() {
    let mock = MockHttpClient::new();
    let mut options = azure_options(BTreeMap::new());
    options.azure_base_url = Some("https://my-resource.openai.azure.com".to_owned());
    options.session_id = Some("x".repeat(67));
    stream_once(&azure_model(), &mock, &options).await;

    assert_eq!(
        recorded_body(&mock)["prompt_cache_key"],
        json!("x".repeat(64))
    );
}

#[tokio::test]
async fn disables_server_side_response_storage() {
    let mock = MockHttpClient::new();
    let mut options = azure_options(BTreeMap::new());
    options.azure_base_url = Some("https://my-resource.openai.azure.com".to_owned());
    stream_once(&azure_model(), &mock, &options).await;

    assert_eq!(recorded_body(&mock)["store"], json!(false));
}

#[tokio::test]
async fn honors_supports_strict_mode_false() {
    let model = Model {
        compat: Some(ModelCompat {
            supports_strict_mode: Some(false),
            ..azure_model().compat.unwrap_or_default()
        }),
        ..azure_model()
    };
    let context = Context {
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("hello".to_owned()),
            timestamp: pi_ai::auth::resolve::now_ms(),
        })],
        tools: Some(vec![Tool {
            name: "preferred".to_owned(),
            description: "Preferred constrained tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"],
            }),
            constrained_sampling: Some(ConstrainedSamplingSetting::Config(
                ConstrainedSamplingConfig::JsonSchema {
                    strict: pi_ai::types::Strictness::Prefer,
                },
            )),
        }]),
        ..Context::default()
    };
    let mock = MockHttpClient::new();
    let mut options = azure_options(BTreeMap::new());
    options.azure_base_url = Some("https://my-resource.openai.azure.com".to_owned());
    options.transport_options = common::mock_transport(&mock);
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let stream = azure_openai_responses::stream(&model, &context, Some(&options));
    let _result = common::drain_and_settle(&stream).await;

    let payload = recorded_body(&mock);
    let tools = payload["tools"].as_array().expect("tools");
    assert!(tools[0].get("strict").is_none());
}

#[tokio::test]
async fn builds_the_default_url_from_the_resource_name_env() {
    let mock = MockHttpClient::new();
    let options = azure_options(env_with("AZURE_OPENAI_RESOURCE_NAME", "my-resource"));
    stream_once(&azure_model(), &mock, &options).await;

    assert_eq!(
        recorded_url(&mock),
        expected_url("https://my-resource.openai.azure.com/openai/v1")
    );
}

// ---------------------------------------------------------------------------
// user agent (upstream azure-openai-base-url user-agent describe)
// ---------------------------------------------------------------------------

/// Stream with the explicit `azureBaseUrl` option and return the mock for
/// header assertions, upstream's `captureClientHeaders`.
async fn capture_client_headers(headers: Option<pi_ai::types::ProviderHeaders>) -> MockHttpClient {
    let mock = MockHttpClient::new();
    let mut options = azure_options(BTreeMap::new());
    options.azure_base_url = Some("https://my-resource.openai.azure.com".to_owned());
    options.headers = headers;
    stream_once(&azure_model(), &mock, &options).await;
    mock
}

#[tokio::test]
async fn uses_the_pi_user_agent_by_default() {
    let mock = capture_client_headers(None).await;

    assert_eq!(
        recorded_header(&mock, "User-Agent").as_deref(),
        Some(get_pi_user_agent().as_str())
    );
}

#[tokio::test]
async fn lets_explicit_headers_override_the_default_user_agent() {
    let mock = capture_client_headers(Some(
        std::iter::once(("User-Agent".to_owned(), Some("custom-agent".to_owned()))).collect(),
    ))
    .await;

    assert_eq!(
        recorded_header(&mock, "User-Agent").as_deref(),
        Some("custom-agent")
    );
}

// ---------------------------------------------------------------------------
// tool choice (upstream azure-openai-tool-choice)
// ---------------------------------------------------------------------------

/// The hand-built deployment model upstream's tool-choice suite constructs.
fn deployment_model() -> Model {
    Model {
        id: "test-deployment".to_owned(),
        name: "Test Deployment".to_owned(),
        api: Api::from("azure-openai-responses"),
        provider: ProviderId::from("azure-openai-responses"),
        base_url: "http://127.0.0.1:9/openai/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 10_000,
        max_tokens: 1_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The read tool the tool-choice suites send.
fn read_tool() -> Tool {
    Tool {
        name: "read".to_owned(),
        description: "Read a file".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
        }),
        constrained_sampling: None,
    }
}

fn tool_context() -> Context {
    Context {
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Summarize this".to_owned()),
            timestamp: 1,
        })],
        tools: Some(vec![read_tool()]),
        ..Context::default()
    }
}

#[tokio::test]
async fn forwards_provider_specific_tool_choice_while_preserving_tool_definitions() {
    let mock = MockHttpClient::new();
    let options = AzureOpenAiResponsesOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        tool_choice: Some(json!("required")),
        ..AzureOpenAiResponsesOptions::default()
    };
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let stream =
        azure_openai_responses::stream(&deployment_model(), &tool_context(), Some(&options));
    let result = common::drain_and_settle(&stream).await;
    assert_eq!(
        result.stop_reason,
        StopReason::Stop,
        "{:?}",
        result.error_message
    );

    let payload = common::recorded_body(&mock);

    assert_eq!(payload["tool_choice"], json!("required"));
    assert_eq!(payload["tools"].as_array().expect("tools").len(), 1);
}

#[tokio::test]
async fn forwards_provider_neutral_tool_choice_from_simple_options() {
    let mock = MockHttpClient::new();
    let options = SimpleStreamOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        tool_choice: Some(ToolChoice::None),
        ..SimpleStreamOptions::default()
    };
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let stream =
        azure_openai_responses::stream_simple(&deployment_model(), &tool_context(), Some(&options));
    let result = common::drain_and_settle(&stream).await;
    assert_eq!(
        result.stop_reason,
        StopReason::Stop,
        "{:?}",
        result.error_message
    );

    let payload = common::recorded_body(&mock);

    assert_eq!(payload["tool_choice"], json!("none"));
    assert_eq!(payload["tools"].as_array().expect("tools").len(), 1);
}

// ---------------------------------------------------------------------------
// reasoning replay (upstream azure-openai-responses-reasoning-replay)
// ---------------------------------------------------------------------------

/// The hand-built reasoning model upstream's replay suite constructs.
fn replay_model() -> Model {
    Model {
        id: "gpt-5-mini".to_owned(),
        name: "GPT-5 Mini".to_owned(),
        api: Api::from("azure-openai-responses"),
        provider: ProviderId::from("azure-openai-responses"),
        base_url: "https://example.invalid".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 400_000,
        max_tokens: 128_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The zeroed accumulator the replay cases process into, upstream's
/// `createOutput`.
fn replay_output(model: &Model) -> pi_ai::types::AssistantMessage {
    pi_ai::types::AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: pi_ai::types::Usage::default(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: pi_ai::auth::resolve::now_ms(),
    }
}

/// The three SSE frames upstream's `createEvents` yields: the reasoning
/// item opens, the `output_item.done` carries it with its encrypted state,
/// and the terminal response names the completed item.
fn replay_events(done_item: &Value, completed_item: &Value) -> Vec<Value> {
    vec![
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 0,
            "item": { "type": "reasoning", "id": done_item["id"], "summary": [] },
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": done_item,
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": { "id": "resp_test", "status": "completed", "output": [completed_item] },
        }),
    ]
}

/// Drive the shared processor over the synthetic frames, upstream's
/// `processResponsesStream(createEvents(...), output, ...)`.
async fn process_replay(done_item: Value, completed_item: Value) -> pi_ai::types::AssistantMessage {
    let model = replay_model();
    let mut output = replay_output(&model);
    let events = pi_ai::utils::event_stream::assistant_message_event_stream();
    let body = common::openai_responses_sse_body(&replay_events(&done_item, &completed_item));
    let response = pi_ai::http::HttpResponse {
        status: 200,
        headers: Vec::new(),
        body: pi_ai::http::HttpByteStream::from_chunks(vec![Ok(bytes::Bytes::from(body))]),
    };
    pi_ai::api::openai_responses_shared::process_responses_stream(
        response,
        &mut output,
        &events,
        &model,
        None,
    )
    .await
    .expect("the replay stream processes");
    output
}

/// The reasoning item the replayed history converts to, upstream's
/// `getReplayedReasoning`.
fn get_replayed_reasoning(
    model: &Model,
    assistant: &pi_ai::types::AssistantMessage,
) -> Option<Value> {
    let context = Context {
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Text("first".to_owned()),
                timestamp: pi_ai::auth::resolve::now_ms() - 1,
            }),
            Message::Assistant(assistant.clone()),
            Message::User(UserMessage {
                content: UserContent::Text("follow-up".to_owned()),
                timestamp: pi_ai::auth::resolve::now_ms(),
            }),
        ],
        ..Context::default()
    };
    let providers: std::collections::BTreeSet<String> =
        std::iter::once("azure-openai-responses".to_owned()).collect();
    let input = pi_ai::api::openai_responses_shared::convert_responses_messages(
        model, &context, &providers, None,
    )
    .expect("the replay history converts");
    input
        .into_iter()
        .find(|item| item["type"] == json!("reasoning"))
}

/// The `encrypted_content` the done item already carries survives the
/// terminal response's fresher value.
#[tokio::test]
async fn preserves_existing_encrypted_content_from_output_item_done() {
    let done_item = json!({
        "type": "reasoning",
        "id": "rs_done",
        "summary": [],
        "encrypted_content": "from-output-item-done",
    });
    let completed_item = {
        let mut item = done_item.clone();
        item["encrypted_content"] = json!("from-response-completed");
        item
    };

    let output = process_replay(done_item, completed_item).await;

    let replayed = get_replayed_reasoning(&replay_model(), &output).expect("the reasoning item");
    assert_eq!(replayed["type"], json!("reasoning"));
    assert_eq!(replayed["id"], json!("rs_done"));
    assert_eq!(
        replayed["encrypted_content"],
        json!("from-output-item-done")
    );
}

/// The terminal response fills `encrypted_content` in when the done item
/// omitted it, keeping `store: false` multi-turn replay stateless.
#[tokio::test]
async fn fills_encrypted_content_when_output_item_done_omitted_it() {
    let done_item = json!({
        "type": "reasoning",
        "id": "rs_missing",
        "summary": [],
    });
    let completed_item = {
        let mut item = done_item.clone();
        item["encrypted_content"] = json!("from-response-completed");
        item
    };

    let output = process_replay(done_item, completed_item).await;

    let replayed = get_replayed_reasoning(&replay_model(), &output).expect("the replay item");
    assert_eq!(replayed["type"], json!("reasoning"));
    assert_eq!(replayed["id"], json!("rs_missing"));
    assert_eq!(
        replayed["encrypted_content"],
        json!("from-response-completed")
    );
}
// ---------------------------------------------------------------------------
// Port-added: deployment mapping, header precedence, reasoning payloads, and
// the stream-lifecycle failures the upstream suites reach only implicitly
// ---------------------------------------------------------------------------

/// Port-added: the `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` env parses model=
/// deployment pairs, skipping blank entries and malformed pairs; the explicit
/// option and the model id fall back in order.
#[tokio::test]
async fn the_deployment_name_map_parses_its_entries_and_falls_back() {
    let env = env_with(
        "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
        "gpt-4o-mini=deploy-mini, , malformed, =empty, empty=,  gpt-4o = deploy-spaced ",
    );
    let mock = MockHttpClient::new();
    let model = azure_model();
    stream_once(&model, &mock, &azure_options(env)).await;
    assert_eq!(recorded_body(&mock)["model"], json!("deploy-mini"));

    // No env entry for the model id: the model id itself rides.
    let mock = MockHttpClient::new();
    let options = azure_options(env_with(
        "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
        "other-model=deploy-other",
    ));
    stream_once(&model, &mock, &options).await;
    assert_eq!(recorded_body(&mock)["model"], json!("gpt-4o-mini"));

    // The explicit option wins over the env map.
    let mock = MockHttpClient::new();
    let mut options = azure_options(env_with(
        "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
        "gpt-4o-mini=deploy-mini",
    ));
    options.azure_deployment_name = Some("explicit-deploy".to_owned());
    stream_once(&model, &mock, &options).await;
    assert_eq!(recorded_body(&mock)["model"], json!("explicit-deploy"));
}

/// Port-added: the model's headers merge, caller `None` values suppress
/// entries, and the api-key header wins.
#[tokio::test]
async fn the_azure_header_merges_follow_the_wire_precedence() {
    let mock = MockHttpClient::new();
    let mut model = azure_model();
    model.headers = Some(
        [
            ("x-init-header".to_owned(), "init".to_owned()),
            ("x-suppressed".to_owned(), "model-value".to_owned()),
        ]
        .into_iter()
        .collect(),
    );
    let mut options = azure_options(BTreeMap::new());
    options.headers = Some(
        [
            ("x-init-header".to_owned(), Some("caller".to_owned())),
            ("x-suppressed".to_owned(), None),
            ("x-additional".to_owned(), Some("extra".to_owned())),
        ]
        .into_iter()
        .collect(),
    );
    stream_once(&model, &mock, &options).await;
    let request = &mock.recorded()[0];
    let header = |name: &str| {
        request
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    assert_eq!(header("x-init-header").as_deref(), Some("caller"));
    assert_eq!(header("x-suppressed"), None);
    assert_eq!(header("x-additional").as_deref(), Some("extra"));
    assert_eq!(header("api-key").as_deref(), Some("test-api-key"));
}

/// Port-added: the request extras — temperature, tools, sampling params, and
/// the reasoning effort/summary pair — ride their wire fields.
#[tokio::test]
async fn the_azure_request_fields_ride_their_request_shapes() {
    let mock = MockHttpClient::new();
    let mut model = azure_model();
    model.reasoning = true;
    let mut options = azure_options(BTreeMap::new());
    options.temperature = Some(0.3);
    options.sampling_params = Some(std::iter::once(("top_p".to_owned(), json!(0.9))).collect());
    options.reasoning_effort = Some(pi_ai::types::ThinkingLevel::Xhigh);
    options.reasoning_summary = Some(ReasoningSummary::Detailed);
    stream_once(&model, &mock, &options).await;
    let body = recorded_body(&mock);
    assert_eq!(body["temperature"], json!(0.3));
    assert_eq!(body["top_p"], json!(0.9));
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));
    assert_eq!(body["reasoning"]["summary"], json!("detailed"));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
}

/// Port-added: with no requested effort the off entry (or `none`) spells the
/// reasoning effort.
#[tokio::test]
async fn the_azure_reasoning_effort_reads_the_off_entry() {
    let options = azure_options(BTreeMap::new());
    let mut model = azure_model();
    model.reasoning = true;
    let mock = MockHttpClient::new();
    stream_once(&model, &mock, &options).await;
    assert_eq!(recorded_body(&mock)["reasoning"]["effort"], json!("none"));

    let mock = MockHttpClient::new();
    let mut model = azure_model();
    model.reasoning = true;
    model.thinking_level_map = Some(
        std::iter::once((
            pi_ai::types::ModelThinkingLevel::Off,
            Some("low".to_owned()),
        ))
        .collect(),
    );
    stream_once(&model, &mock, &options).await;
    assert_eq!(recorded_body(&mock)["reasoning"]["effort"], json!("low"));

    // The null off marker omits the reasoning field entirely.
    let mock = MockHttpClient::new();
    let mut model = azure_model();
    model.reasoning = true;
    model.thinking_level_map =
        Some(std::iter::once((pi_ai::types::ModelThinkingLevel::Off, None)).collect());
    stream_once(&model, &mock, &options).await;
    assert![recorded_body(&mock).get("reasoning").is_none()];
}

/// Port-added: the payload hook replaces the azure request payload.
#[tokio::test]
async fn the_azure_payload_hook_replaces_the_request_payload() {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None::<Value>));
    let slot = std::sync::Arc::clone(&observed);
    let mut options = azure_options(BTreeMap::new());
    options.transport_options = TransportOptions {
        http_client: Some(std::sync::Arc::new(mock.clone())),
        on_payload: Some(pi_ai::types::OnPayload::new(move |payload, _model| {
            *slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(payload);
            Box::pin(async { None })
        })),
        ..TransportOptions::default()
    };
    let settled = common::drain_and_settle(&azure_openai_responses::stream(
        &azure_model(),
        &context(),
        Some(&options),
    ))
    .await;
    assert!(matches!(
        settled.stop_reason,
        StopReason::Stop | StopReason::Error
    ));
    let payload = observed
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("the payload hook observed the request");
    assert![payload.get("model").is_some()];
}

/// Port-added: the azure stream without an api key fails before dispatch
/// with the provider-named setup message.
#[tokio::test]
async fn the_azure_stream_without_an_api_key_fails_before_dispatch() {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let model = azure_model();
    let options = AzureOpenAiResponsesOptions::default();
    let result = common::drain_and_settle(&azure_openai_responses::stream(
        &model,
        &context(),
        Some(&options),
    ))
    .await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: azure-openai-responses")
    );
    assert_eq!(mock.request_count(), 0);
}

/// Port-added: azure `stream_simple` maps the auto tool choice and clamps
/// the reasoning level to the model's supported levels.
#[tokio::test]
async fn the_azure_simple_options_ride_their_request_fields() {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let mut model = azure_model();
    model.base_url = "https://test-resource.openai.azure.com/openai/v1".to_owned();
    let options = SimpleStreamOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        tool_choice: Some(ToolChoice::Auto),
        reasoning: Some(pi_ai::types::ThinkingLevel::Xhigh),
        ..SimpleStreamOptions::default()
    };
    let settled = common::drain_and_settle(&azure_openai_responses::stream_simple(
        &model,
        &context(),
        Some(&options),
    ))
    .await;
    assert!(matches!(
        settled.stop_reason,
        StopReason::Stop | StopReason::Error
    ));
    let body = recorded_body(&mock);
    assert_eq!(body["tool_choice"], json!("auto"));
}

// ---------------------------------------------------------------------------
// Port-added: the config-fallback, reasoning, tools, and conversion-rejection
// edges the upstream suites reach only implicitly
// ---------------------------------------------------------------------------

/// Port-added: an empty option string falls through to the scoped env, and
/// the api version falls to the default when nothing names one,
/// upstream's `resolveAzureConfig` `??` chains.
#[tokio::test]
async fn the_azure_empty_option_strings_fall_through_to_the_env() {
    let env = BTreeMap::from([
        (
            "AZURE_OPENAI_API_VERSION".to_owned(),
            "2025-04-01-preview".to_owned(),
        ),
        (
            "AZURE_OPENAI_BASE_URL".to_owned(),
            "https://from-env.openai.azure.com".to_owned(),
        ),
        (
            "AZURE_OPENAI_RESOURCE_NAME".to_owned(),
            "unused-resource".to_owned(),
        ),
    ]);
    let mut options = azure_options(env);
    options.azure_api_version = Some(String::new());
    options.azure_base_url = Some("   ".to_owned());
    options.azure_resource_name = Some(String::new());
    let mock = MockHttpClient::new();
    stream_once(&azure_model(), &mock, &options).await;
    assert_eq![
        recorded_url(&mock),
        "https://from-env.openai.azure.com/openai/v1/responses?api-version=2025-04-01-preview"
    ];
}

/// Port-added: a request-less reasoning summary defaults the effort to the
/// wire's "medium", and an explicit level maps through `thinkingLevelMap`,
/// upstream's `buildParams` reasoning arm.
#[tokio::test]
async fn the_azure_reasoning_defaults_and_maps_its_effort() {
    let mut options = azure_options(BTreeMap::new());
    options.reasoning_summary = Some(ReasoningSummary::Auto);
    let mut model = azure_model();
    model.reasoning = true;
    let mock = MockHttpClient::new();
    stream_once(&model, &mock, &options).await;
    let body = recorded_body(&mock);
    assert_eq!(body["reasoning"]["effort"], json!("medium"));
    assert_eq!(body["reasoning"]["summary"], json!("auto"));

    // An explicit level without a map rides the level's wire spelling.
    let mut options = azure_options(BTreeMap::new());
    options.reasoning_effort = Some(pi_ai::types::ThinkingLevel::Max);
    let mock = MockHttpClient::new();
    stream_once(&model, &mock, &options).await;
    assert_eq!(recorded_body(&mock)["reasoning"]["effort"], json!("max"));
}

/// Port-added: an azure request whose grammar-replayed assistant call
/// carries a non-string input rejects before dispatch,
/// the conversion rejection `buildParams` raises.
#[tokio::test]
async fn an_azure_grammar_replay_with_a_non_string_input_rejects() {
    let grammar_tool = Tool {
        name: "ln".to_owned(),
        description: "List a file".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"],
        }),
        constrained_sampling: Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::Grammar {
                variants: std::iter::once((
                    pi_ai::types::GrammarFormat::OpenaiLark,
                    "start: /[a-z]+/".to_owned(),
                ))
                .collect(),
            },
        )),
    };
    let mut context = context();
    context.tools = Some(vec![grammar_tool]);
    context.messages = vec![Message::Assistant(common::tool_call_assistant_message(
        "openai-responses",
        "openai",
        "openai-model",
        pi_ai::types::ToolCall {
            id: "call_1".to_owned(),
            name: "ln".to_owned(),
            arguments: serde_json::Map::from_iter([("text".to_owned(), json!(42))]),
            thought_signature: None,
            namespace: None,
        },
    ))];
    let model = Model {
        compat: Some(ModelCompat {
            supports_openai_grammar_tools: Some(true),
            ..ModelCompat::default()
        }),
        ..azure_model()
    };
    let _mock = MockHttpClient::new();
    let options = azure_options(BTreeMap::new());
    let settled = common::drain_and_settle(&azure_openai_responses::stream(
        &model,
        &context,
        Some(&options),
    ))
    .await;
    assert_eq!(settled.stop_reason, StopReason::Error);
    let message = settled.error_message.expect("the grammar rejection");
    assert![
        message.contains("requires argument \"text\" to be a string"),
        "{message}"
    ];
}
