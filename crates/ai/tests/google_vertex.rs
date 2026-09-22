//! The Google Vertex adapter suite, ported from
//! `packages/ai/test/google-vertex-api-key-resolution.test.ts`,
//! `packages/ai/test/google-raw-stop-reason.test.ts` (the Vertex sections),
//! and `packages/ai/test/google-thinking-level-map.test.ts` (the Vertex
//! sections) at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: upstream asserts on the mocked SDK client
//! construction; here the same decisions surface as the recorded request's
//! URL shape and headers over the transport seam, with the ADC metadata
//! server mocked so the fallback chain resolves hermetically. The capture
//! helpers assert the captured payload directly, without upstream's
//! throw-to-capture error-message check.

#![expect(
    clippy::expect_used,
    reason = "the tests pin adapter outcomes; an unexpected shape panics the test by design"
)]

use std::collections::BTreeMap;

use pi_ai::api::google_shared::{ResolvedGoogleThinkingLevel, resolve_google_thinking_level};
use pi_ai::api::google_vertex::{
    GoogleVertexStreamOptions, stream as stream_vertex, stream_simple,
};
use pi_ai::http::mock::{MockHttpClient, MockResponse};
use pi_ai::types::{
    Api, Context, Message, Modality, Model, ModelThinkingLevel, ProviderId, SimpleStreamOptions,
    ThinkingBudgets, ThinkingLevel, ThinkingLevelMap, UserContent, UserMessage,
};
use serde_json::json;

mod common;

/// The user-message-only context the suite runs.
fn context() -> Context {
    Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("hello".to_owned()),
            timestamp: 0,
        })],
        tools: None,
    }
}

/// The catalog gemini-3-flash-preview Vertex model, upstream's
/// `getModel("google-vertex", ...)`.
fn catalog_vertex() -> Model {
    common::builtin_model("google-vertex", "gemini-3-flash-preview")
}

/// The test Vertex model with a thinking map, upstream's `vertexModel`.
fn vertex_model(id: &str, thinking_level_map: ThinkingLevelMap) -> Model {
    Model {
        id: id.to_owned(),
        name: id.to_owned(),
        api: Api::from("google-vertex"),
        provider: ProviderId::from("test-vertex"),
        base_url: "https://example.invalid/v1".to_owned(),
        reasoning: true,
        thinking_level_map: Some(thinking_level_map),
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4_096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// One streamed chunk as a data-only SSE response body.
fn sse_data_response(status: u16, chunk: &str) -> MockResponse {
    MockResponse::status(status)
        .with_header("content-type", "text/event-stream")
        .with_body(format!("data: {chunk}\n\n"))
}

/// The terminal chunk the raw-stop-reason suite yields.
fn raw_stop_chunk(finish_reason: &str, include_function_call: bool) -> String {
    let mut candidate = json!({ "finishReason": finish_reason });
    if include_function_call {
        candidate["content"] = json!({
            "parts": [{
                "functionCall": {
                    "id": "call-1",
                    "name": "echo",
                    "args": { "value": "truncated" },
                }
            }]
        });
    }
    json!({
        "responseId": "vertex-response-id",
        "candidates": [candidate],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
    })
    .to_string()
}

fn mount_vertex_stream(mock: &MockHttpClient, chunk: &str) {
    let response = sse_data_response(200, chunk);
    mock.on(|request| request.url.contains(":streamGenerateContent"))
        .respond(response);
}

/// Mount the ADC metadata-server token endpoint so the fallback chain
/// resolves hermetically even on machines with real gcloud credentials.
fn mount_metadata_server(mock: &MockHttpClient) {
    mock.on(|request| request.url.contains("169.254.169.254"))
        .respond(
            MockResponse::status(200)
                .with_body(r#"{"access_token": "metadata-token", "expires_in": 3600}"#),
        );
}

/// Point `GOOGLE_APPLICATION_CREDENTIALS` at a path that does not exist so
/// the ADC chain reaches the (mocked) metadata server deterministically.
fn env_without_credentials_file() -> BTreeMap<String, String> {
    BTreeMap::from([(
        "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
        "/nonexistent/vertex-credentials.json".to_owned(),
    )])
}

fn express_options(mock: &MockHttpClient) -> GoogleVertexStreamOptions {
    GoogleVertexStreamOptions {
        transport_options: common::mock_transport(mock),
        api_key: Some("AIzaSyExampleRealisticLookingApiKey123456".to_owned()),
        env: Some(env_without_credentials_file()),
        ..GoogleVertexStreamOptions::default()
    }
}

fn adc_options(mock: &MockHttpClient) -> GoogleVertexStreamOptions {
    let mut options = GoogleVertexStreamOptions {
        transport_options: common::mock_transport(mock),
        env: Some(env_without_credentials_file()),
        ..GoogleVertexStreamOptions::default()
    };
    options.project = Some("test-project".to_owned());
    options.location = Some("us-central1".to_owned());
    options
}

/// The stream request the mock recorded, the URL-shape assertions the
/// SDK-client-config assertions became.
async fn settle_stream(
    model: &Model,
    context: &Context,
    options: &GoogleVertexStreamOptions,
) -> pi_ai::types::AssistantMessage {
    let stream = stream_vertex(model, context, Some(options));
    stream.result().await
}

// --- upstream google-vertex-api-key-resolution.test.ts ---

/// The header of the n-th recorded request; ADC flows record the token
/// request before the stream request, so the stream is not the first.
fn recorded_header_at(mock: &MockHttpClient, index: usize, name: &str) -> Option<String> {
    mock.recorded()[index]
        .headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}

/// A real API key rides the express path: the plain aiplatform host, the
/// `publishers/google` model path without a project prefix, and the API-key
/// header.
#[tokio::test]
async fn uses_the_api_key_client_for_real_api_keys() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_vertex();
    let options = express_options(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    let request = &mock.recorded()[0];
    assert_eq!(
        request.url,
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    );
    assert_eq!(
        common::recorded_header(&mock, "x-goog-api-key").as_deref(),
        Some("AIzaSyExampleRealisticLookingApiKey123456")
    );
    assert!(recorded_header_at(&mock, 0, "authorization").is_none());
}

/// The ambient-auth placeholder falls through to ADC: the request carries
/// the project/location prefix and a bearer token, no API-key header.
#[tokio::test]
async fn falls_back_to_adc_when_the_api_key_is_a_placeholder_marker() {
    let mock = MockHttpClient::new();
    mount_metadata_server(&mock);
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_vertex();
    let mut options = adc_options(&mock);
    options.api_key = Some("<authenticated>".to_owned());

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    let request = &mock.recorded()[1];
    assert_eq!(
        request.url,
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    );
    assert_eq!(
        recorded_header_at(&mock, 1, "authorization").as_deref(),
        Some("Bearer metadata-token")
    );
    assert!(common::recorded_header(&mock, "x-goog-api-key").is_none());
}

/// The `gcp-vertex-credentials` marker falls through to ADC the same way.
#[tokio::test]
async fn falls_back_to_adc_when_the_api_key_is_the_credentials_marker() {
    let mock = MockHttpClient::new();
    mount_metadata_server(&mock);
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_vertex();
    let mut options = adc_options(&mock);
    options.api_key = Some("gcp-vertex-credentials".to_owned());

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    assert!(common::recorded_header(&mock, "x-goog-api-key").is_none());
    assert_eq!(
        recorded_header_at(&mock, 1, "authorization").as_deref(),
        Some("Bearer metadata-token")
    );
}

/// A generated catalog base URL carrying the `{location}` placeholder is
/// never forwarded: the request uses the default host shape.
#[tokio::test]
async fn does_not_forward_generated_vertex_base_url_placeholders() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let mut model = catalog_vertex();
    model.base_url = "https://{location}-aiplatform.googleapis.com/v1".to_owned();
    let options = express_options(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    assert_eq!(
        mock.recorded()[0].url,
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    );
}

/// An explicit caller `User-Agent` overrides the default.
#[tokio::test]
async fn lets_explicit_headers_override_the_default_user_agent() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_vertex();
    let mut options = express_options(&mock);
    options.headers = Some(BTreeMap::from([(
        "User-Agent".to_owned(),
        Some("custom-agent".to_owned()),
    )]));

    let _ = settle_stream(&model, &context(), &options).await;

    assert_eq!(
        common::recorded_header(&mock, "User-Agent").as_deref(),
        Some("custom-agent")
    );
}

/// A custom base URL forwards verbatim with the COLLECTION scope: no project
/// prefix, and the version segment rides because the URL lacks one.
#[tokio::test]
async fn forwards_custom_base_urls_to_the_clients() {
    for build_options in [build_express, build_adc] {
        let mock = MockHttpClient::new();
        mount_metadata_server(&mock);
        mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
        let mut model = catalog_vertex();
        model.base_url = "https://proxy.example.com".to_owned();
        let options = build_options(&mock);

        let message = settle_stream(&model, &context(), &options).await;

        assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
        assert_eq!(
            mock.recorded().last().expect("request").url,
            "https://proxy.example.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
        );
    }
}

fn build_express(mock: &MockHttpClient) -> GoogleVertexStreamOptions {
    express_options(mock)
}

fn build_adc(mock: &MockHttpClient) -> GoogleVertexStreamOptions {
    let mut options = adc_options(mock);
    options.api_key = Some("<authenticated>".to_owned());
    options
}

/// When the custom base URL already carries a version segment, no `v1` is
/// inserted.
#[tokio::test]
async fn does_not_append_api_version_when_the_base_url_includes_one() {
    let mock = MockHttpClient::new();
    mount_metadata_server(&mock);
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let mut model = catalog_vertex();
    model.base_url =
        "https://proxy.example.com/v1/projects/test-project/locations/global".to_owned();
    let options = build_adc(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    assert_eq!(
        mock.recorded().last().expect("request").url,
        "https://proxy.example.com/v1/projects/test-project/locations/global/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    );
}

/// Missing project or location fail the stream with the upstream wording.
#[tokio::test]
async fn requires_a_project_and_location_for_adc() {
    let mock = MockHttpClient::new();
    let mut options = GoogleVertexStreamOptions {
        transport_options: common::mock_transport(&mock),
        env: Some(env_without_credentials_file()),
        ..GoogleVertexStreamOptions::default()
    };
    options.api_key = Some("gcp-vertex-credentials".to_owned());
    let model = catalog_vertex();

    let message = settle_stream(&model, &context(), &options).await;
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("Vertex AI requires a project ID")),
        "got: {:?}",
        message.error_message
    );

    options.project = Some("test-project".to_owned());
    let message = settle_stream(&model, &context(), &options).await;
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("Vertex AI requires a location")),
        "got: {:?}",
        message.error_message
    );
}

// --- upstream google-raw-stop-reason.test.ts (Vertex sections) ---

/// An unmapped finish reason surfaces as an error stop with the raw wire
/// value preserved and the exact provider-stopped wording.
#[tokio::test]
async fn preserves_raw_vertex_finish_reasons_for_errors() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("SAFETY", false));
    let model = catalog_vertex();
    let mut options = express_options(&mock);
    options.project = Some("test-project".to_owned());
    options.location = Some("us-central1".to_owned());

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("SAFETY"));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider stopped with: SAFETY")
    );
}

/// `MAX_TOKENS` with a tool call stays `length` on Vertex too.
#[tokio::test]
async fn preserves_max_tokens_with_a_tool_call_as_length_on_vertex() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("MAX_TOKENS", true));
    let model = catalog_vertex();
    let options = express_options(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Length);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("MAX_TOKENS"));
    assert!(
        message
            .content
            .iter()
            .any(|block| matches!(block, pi_ai::types::AssistantBlock::ToolCall(_)))
    );
}

/// `STOP` with a tool call upgrades to `toolUse` on Vertex too.
#[tokio::test]
async fn maps_stop_with_a_tool_call_to_tool_use_on_vertex() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", true));
    let model = catalog_vertex();
    let options = express_options(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::ToolUse);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("STOP"));
}

// --- upstream google-thinking-level-map.test.ts (Vertex sections) ---

async fn capture_vertex_payload(
    model: &Model,
    reasoning: ThinkingLevel,
    thinking_budgets: Option<ThinkingBudgets>,
) -> serde_json::Value {
    let mock = MockHttpClient::new();
    let (on_payload, captured) = common::payload_capture();
    let mut transport = common::mock_transport(&mock);
    transport.on_payload = Some(on_payload);
    let options = SimpleStreamOptions {
        transport_options: transport,
        api_key: Some("test".to_owned()),
        reasoning: Some(reasoning),
        thinking_budgets,
        ..SimpleStreamOptions::default()
    };

    let stream = stream_simple(model, &context(), Some(&options));
    let _message = stream.result().await;
    common::captured_payload(&captured)
}

/// `xhigh` maps through the model map to the provider-native level.
#[tokio::test]
async fn maps_vertex_extended_levels() {
    let map = ThinkingLevelMap::from([(ModelThinkingLevel::Xhigh, Some("high".to_owned()))]);
    let model = vertex_model("gemini-3.7-flash", map);
    let payload = capture_vertex_payload(&model, ThinkingLevel::Xhigh, None).await;

    assert_eq!(
        payload["config"]["thinkingConfig"]["thinkingLevel"],
        json!("HIGH")
    );
    assert_eq!(
        payload["config"]["thinkingConfig"]["includeThoughts"],
        json!(true)
    );
}

/// A custom budget for the mapped level rides as `thinkingBudget`.
#[tokio::test]
async fn uses_mapped_vertex_levels_for_token_budgets() {
    let map = ThinkingLevelMap::from([(ModelThinkingLevel::Max, Some("high".to_owned()))]);
    let model = vertex_model("gemini-2.5-flash", map);
    let payload = capture_vertex_payload(
        &model,
        ThinkingLevel::Max,
        Some(ThinkingBudgets {
            high: Some(4321),
            ..ThinkingBudgets::default()
        }),
    )
    .await;

    assert_eq!(
        payload["config"]["thinkingConfig"]["thinkingBudget"],
        json!(4321)
    );
}

/// The resolver accepts the mapped value set the Vertex suite exercises.
#[test]
fn resolves_vertex_thinking_levels() {
    let map = ThinkingLevelMap::from([(ModelThinkingLevel::Xhigh, Some("high".to_owned()))]);
    let model = vertex_model("gemini-3.7-flash", map);
    assert_eq!(
        resolve_google_thinking_level(&model, ModelThinkingLevel::Xhigh).expect("resolved"),
        ResolvedGoogleThinkingLevel::High
    );
}

// --- the ADC credential-file flows ---

/// One os-randomness-backed RNG, the generator the service-account fixture
/// signs with so no key material rides the repository.
struct TestRng;

impl rsa::rand_core::RngCore for TestRng {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0u8; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        getrandom::fill(dest).expect("os randomness");
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rsa::rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl rsa::rand_core::CryptoRng for TestRng {}

/// A throwaway RSA key pair as the PEM the service-account fixture carries.
fn generated_rsa_pem() -> String {
    use rsa::pkcs8::EncodePrivateKey;

    let mut rng = TestRng;
    let key = rsa::RsaPrivateKey::new(&mut rng, 2048).expect("the test key pair generates");
    key.to_pkcs8_pem(rsa::pkcs8::LineEnding::LF)
        .expect("the key serializes")
        .to_string()
}

/// The path of the ADC credentials file fixture, the temp-dir spelling the
/// other suites use, cleaned up when the test drops the guard.
fn credentials_file(name: &str, contents: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("pi-vertex-adc-{name}.json"));
    std::fs::write(&path, contents).expect("the fixture writes");
    path
}

fn env_with_credentials_file(path: &std::path::Path) -> BTreeMap<String, String> {
    BTreeMap::from([(
        "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
        path.to_string_lossy().into_owned(),
    )])
}

/// The full service-account JWT exchange: the credentials file signs a
/// scoped RS256 JWT and exchanges it for the bearer token the stream
/// carries, with the quota project riding the `x-goog-user-project` header.
#[tokio::test]
async fn exchanges_a_service_account_jwt_for_a_bearer_token() {
    let mock = MockHttpClient::new();
    let pem = generated_rsa_pem();
    let fixture = credentials_file(
        "service-account",
        &json!({
            "type": "service_account",
            "client_email": "sa@test-project.iam.gserviceaccount.com",
            "private_key": pem,
            "quota_project_id": "billing-project",
        })
        .to_string(),
    );
    mock.on(|request| request.url.contains("oauth2.googleapis.com/token"))
        .respond(MockResponse::status(200).with_body(r#"{"access_token": "sa-token"}"#));
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let mut options = adc_options(&mock);
    options.env = Some(env_with_credentials_file(&fixture));

    let message = settle_stream(&catalog_vertex(), &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    let token_request = &mock.recorded()[0];
    assert_eq!(token_request.url, "https://oauth2.googleapis.com/token");
    assert_eq!(
        recorded_header_at(&mock, 0, "content-type").as_deref(),
        Some("application/x-www-form-urlencoded")
    );
    let body = String::from_utf8(
        token_request
            .body
            .as_ref()
            .expect("the token body")
            .to_vec(),
    )
    .expect("the token body");
    let (grant, assertion) = body
        .split_once("&assertion=")
        .expect("the jwt-bearer grant body");
    assert_eq!(
        grant,
        "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"
    );
    let claims: Vec<&str> = assertion.split('.').collect();
    assert_eq!(claims.len(), 3, "the RS256 JWT carries three segments");
    let (header, payload) = (claims[0], claims[1]);
    let decode = |segment: &str| -> String {
        use base64::Engine;
        String::from_utf8(
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(segment)
                .expect("the JWT segment decodes"),
        )
        .expect("the JWT segment is text")
    };
    assert!(
        decode(header).contains("RS256"),
        "the header names the RS256 algorithm: {}",
        decode(header)
    );
    assert!(
        decode(payload).contains("https://www.googleapis.com/auth/cloud-platform"),
        "the claims carry the cloud-platform scope: {}",
        decode(payload)
    );

    assert_eq!(
        recorded_header_at(&mock, 1, "authorization").as_deref(),
        Some("Bearer sa-token")
    );
    assert_eq!(
        recorded_header_at(&mock, 1, "x-goog-user-project").as_deref(),
        Some("billing-project"),
        "the quota project rides the stream request"
    );
    let _ = std::fs::remove_file(&fixture);
}

/// The authorized-user refresh grant: the form-encoded refresh body carries
/// the `%`-encoded fields and the access token rides the stream request.
#[tokio::test]
async fn exchanges_the_authorized_user_refresh_grant_for_a_bearer_token() {
    let mock = MockHttpClient::new();
    let fixture = credentials_file(
        "authorized-user",
        &json!({
            "type": "authorized_user",
            "client_id": "apps.googleusercontent.com",
            "client_secret": "secret&x=1",
            "refresh_token": "refresh+token",
        })
        .to_string(),
    );
    mock.on(|request| request.url.contains("oauth2.googleapis.com/token"))
        .respond(
            MockResponse::status(200)
                .with_body(r#"{"access_token": "refreshed-token", "expires_in": 3600}"#),
        );
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let mut options = adc_options(&mock);
    options.env = Some(env_with_credentials_file(&fixture));

    let message = settle_stream(&catalog_vertex(), &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    let body = String::from_utf8(
        mock.recorded()[0]
            .body
            .as_ref()
            .expect("the token body")
            .to_vec(),
    )
    .expect("the token body is text");
    assert_eq!(
        body,
        "grant_type=refresh_token&refresh_token=refresh%2Btoken&client_id=apps.googleusercontent.com&client_secret=secret%26x%3D1"
    );
    assert_eq!(
        recorded_header_at(&mock, 1, "authorization").as_deref(),
        Some("Bearer refreshed-token")
    );
    let _ = std::fs::remove_file(&fixture);
}

/// The token-exchange failures surface with the upstream wording: non-2xx
/// with the body, non-JSON bodies, and credential-shape gates.
#[tokio::test]
async fn reports_the_token_exchange_failures() {
    let pem = generated_rsa_pem();

    for (name, contents, expected) in [
        (
            "unsupported-type",
            json!({ "type": "external_account" }).to_string(),
            "Unsupported Google credentials type: external_account",
        ),
        (
            "missing-email",
            json!({ "type": "service_account", "private_key": &pem }).to_string(),
            "Google service-account credentials are missing client_email",
        ),
        (
            "missing-key",
            json!({
                "type": "service_account",
                "client_email": "sa@test.iam.gserviceaccount.com",
            })
            .to_string(),
            "Google service-account credentials are missing private_key",
        ),
        (
            "bad-key",
            json!({
                "type": "service_account",
                "client_email": "sa@test.iam.gserviceaccount.com",
                "private_key": "not a pem",
            })
            .to_string(),
            "Google service-account private key is not usable",
        ),
        (
            "missing-client-id",
            json!({ "type": "authorized_user", "client_secret": "s", "refresh_token": "r" })
                .to_string(),
            "Google authorized-user credentials are missing client_id",
        ),
        (
            "missing-client-secret",
            json!({ "type": "authorized_user", "client_id": "i", "refresh_token": "r" })
                .to_string(),
            "Google authorized-user credentials are missing client_secret",
        ),
        (
            "missing-refresh-token",
            json!({ "type": "authorized_user", "client_id": "i", "client_secret": "s" })
                .to_string(),
            "Google authorized-user credentials are missing refresh_token",
        ),
    ] {
        let mock = MockHttpClient::new();
        let fixture = credentials_file(name, &contents);
        let mut options = adc_options(&mock);
        options.env = Some(env_with_credentials_file(&fixture));

        let message = settle_stream(&catalog_vertex(), &context(), &options).await;

        assert_eq!(message.stop_reason, pi_ai::types::StopReason::Error);
        assert!(
            message
                .error_message
                .as_deref()
                .is_some_and(|text| text.contains(expected)),
            "{name}: {:?}",
            message.error_message
        );
        let _ = std::fs::remove_file(&fixture);
    }
}

/// The gcloud well-known path expands `~` onto the home directory before
/// the file read, upstream's ADC default.
#[tokio::test]
async fn expands_the_tilde_in_the_credentials_path() {
    let mock = MockHttpClient::new();
    let fixture = credentials_file(
        "tilde",
        &json!({
            "type": "authorized_user",
            "client_id": "i",
            "client_secret": "s",
            "refresh_token": "r",
        })
        .to_string(),
    );
    let home = std::env::home_dir().expect("the home directory");
    let home_copy = fixture
        .file_name()
        .map(|name| home.join(name))
        .expect("the fixture has a name");
    std::fs::copy(&fixture, &home_copy).expect("the home copy writes");
    mock.on(|request| request.url.contains("oauth2.googleapis.com/token"))
        .respond(MockResponse::status(200).with_body(r#"{"access_token": "home-token"}"#));
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let mut options = adc_options(&mock);
    let tilde_path = format!(
        "~/{}",
        home_copy
            .file_name()
            .and_then(|n| n.to_str())
            .expect("the name")
    );
    options.env = Some(BTreeMap::from([(
        "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
        tilde_path,
    )]));

    let message = settle_stream(&catalog_vertex(), &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    assert_eq!(
        recorded_header_at(&mock, 1, "authorization").as_deref(),
        Some("Bearer home-token")
    );
    let _ = std::fs::remove_file(&fixture);
    let _ = std::fs::remove_file(&home_copy);
}

/// The metadata-server fallback failures surface with the phase wording.
#[tokio::test]
async fn reports_the_metadata_server_failures() {
    let model = catalog_vertex();

    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("169.254.169.254"))
        .respond(MockResponse::status(403));
    let message = settle_stream(&model, &context(), &adc_options(&mock)).await;
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("Google metadata server returned status 403")),
        "{:?}",
        message.error_message
    );

    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("169.254.169.254"))
        .respond(MockResponse::status(200).with_body("not json"));
    let message = settle_stream(&model, &context(), &adc_options(&mock)).await;
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("Google metadata server token is not JSON")),
        "{:?}",
        message.error_message
    );

    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("169.254.169.254"))
        .respond(MockResponse::status(200).with_body(r#"{"expires_in": 3600}"#));
    let message = settle_stream(&model, &context(), &adc_options(&mock)).await;
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("carries no access_token")),
        "{:?}",
        message.error_message
    );
}

/// The token exchange's non-2xx and non-JSON bodies surface with the phase
/// wording, the authorized-user fixture driving the request.
#[tokio::test]
async fn reports_the_token_endpoint_failures() {
    let model = catalog_vertex();
    let fixture = credentials_file(
        "failing-user",
        &json!({
            "type": "authorized_user",
            "client_id": "i",
            "client_secret": "s",
            "refresh_token": "r",
        })
        .to_string(),
    );

    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("oauth2.googleapis.com/token"))
        .respond(MockResponse::status(400).with_body("bad grant"));
    let mut options = adc_options(&mock);
    options.env = Some(env_with_credentials_file(&fixture));
    let message = settle_stream(&model, &context(), &options).await;
    assert!(
        message.error_message.as_deref().is_some_and(
            |text| text.contains("Google token exchange returned status 400: bad grant")
        ),
        "{:?}",
        message.error_message
    );

    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("oauth2.googleapis.com/token"))
        .respond(MockResponse::status(200).with_body("not json"));
    let mut options = adc_options(&mock);
    options.env = Some(env_with_credentials_file(&fixture));
    let message = settle_stream(&model, &context(), &options).await;
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("Google token exchange response is not JSON")),
        "{:?}",
        message.error_message
    );

    let _ = std::fs::remove_file(&fixture);
}

// --- the disabled-thinking configs and the Gemini 3 simple stream ---

/// The config payload a `streamSimple` capture sees, the thinking-shape
/// helper the level-mapping suites share.
async fn capture_simple_thinking_config(
    model: &Model,
    reasoning: Option<ThinkingLevel>,
) -> serde_json::Value {
    let mock = MockHttpClient::new();
    let (on_payload, captured) = common::payload_capture();
    let mut transport = common::mock_transport(&mock);
    transport.on_payload = Some(on_payload);
    let options = SimpleStreamOptions {
        transport_options: transport,
        api_key: Some("test".to_owned()),
        reasoning,
        ..SimpleStreamOptions::default()
    };

    let stream = stream_simple(model, &context(), Some(&options));
    let _message = stream.result().await;
    common::captured_payload(&captured)["config"]["thinkingConfig"].clone()
}

/// `streamSimple` without reasoning disables thinking with the model-native
/// shape: the Gemini 3 levels, the budget zero elsewhere.
#[tokio::test]
async fn disables_thinking_with_the_model_native_shape_without_reasoning() {
    let pro = common::builtin_model("google-vertex", "gemini-3.1-pro-preview");
    assert_eq!(
        capture_simple_thinking_config(&pro, None).await,
        json!({ "thinkingLevel": "LOW" })
    );

    let flash = common::builtin_model("google-vertex", "gemini-3-flash-preview");
    assert_eq!(
        capture_simple_thinking_config(&flash, None).await,
        json!({ "thinkingLevel": "MINIMAL" })
    );

    let other = vertex_model("gemini-2.5-flash", ThinkingLevelMap::default());
    assert_eq!(
        capture_simple_thinking_config(&other, None).await,
        json!({ "thinkingBudget": 0 })
    );

    // A level that resolves keeps the enabled-thinking shape.
    assert_eq!(
        capture_simple_thinking_config(&flash, Some(ThinkingLevel::High)).await,
        json!({ "thinkingLevel": "HIGH", "includeThoughts": true })
    );
}

/// The Gemini 3 simple stream maps the pi level onto the provider-native
/// level for the flash and pro families.
#[tokio::test]
async fn maps_gemini3_reasoning_levels_on_the_simple_stream() {
    let flash = common::builtin_model("google-vertex", "gemini-3-flash-preview");
    assert_eq!(
        capture_simple_thinking_config(&flash, Some(ThinkingLevel::Medium)).await["thinkingLevel"],
        json!("MEDIUM")
    );
    assert_eq!(
        capture_simple_thinking_config(&flash, Some(ThinkingLevel::Minimal)).await["thinkingLevel"],
        json!("MINIMAL")
    );

    let pro = common::builtin_model("google-vertex", "gemini-3.1-pro-preview");
    assert_eq!(
        capture_simple_thinking_config(&pro, Some(ThinkingLevel::Low)).await["thinkingLevel"],
        json!("LOW")
    );
    assert_eq!(
        capture_simple_thinking_config(&pro, Some(ThinkingLevel::High)).await["thinkingLevel"],
        json!("HIGH")
    );
}

/// The multi-region and global locations ride the SDK's host selection: the
/// `us`/`eu` rep hosts, the plain host for `global`, and the embedded host
/// for everything else.
#[tokio::test]
async fn selects_the_default_hosts_per_location() {
    for (location, expected_prefix) in [
        ("us", "https://aiplatform.us.rep.googleapis.com"),
        ("eu", "https://aiplatform.eu.rep.googleapis.com"),
        ("global", "https://aiplatform.googleapis.com"),
        (
            "asia-northeast1",
            "https://asia-northeast1-aiplatform.googleapis.com",
        ),
    ] {
        let mock = MockHttpClient::new();
        mount_metadata_server(&mock);
        mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
        let mut options = adc_options(&mock);
        options.location = Some(location.to_owned());

        let message = settle_stream(&catalog_vertex(), &context(), &options).await;

        assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
        let url = &mock.recorded()[1].url;
        assert!(
            url.starts_with(&format!("{expected_prefix}/v1/projects/")),
            "{location}: {url}"
        );
    }
}

/// A custom base URL carrying a `v1beta2` version segment suppresses the
/// version insertion too, upstream's `baseUrlIncludesApiVersion`.
#[tokio::test]
async fn does_not_append_the_version_when_the_custom_base_carries_beta() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let mut model = catalog_vertex();
    model.base_url = "https://proxy.example.com/v1beta2".to_owned();
    let options = express_options(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    assert_eq!(
        mock.recorded()[0].url,
        "https://proxy.example.com/v1beta2/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    );
}
