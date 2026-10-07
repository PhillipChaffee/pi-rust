//! Upstream `packages/coding-agent/test/model-runtime-cloudflare-compat.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::model_runtime` and `pi_coding_agent::model_registry`
//! (#121).
//!
//! Porting restatements this suite records (the ticket's recorded decision):
//!
//! - Upstream mocks the `openai` SDK constructor and asserts
//!   `clientOptions.baseURL`/`defaultHeaders`; the port injects
//!   `pi_ai::http::MockHttpClient` through the request options'
//!   `transport_options.http_client` and asserts the recorded request: the
//!   URL carries the materialized gateway endpoint (the completions adapter
//!   appends `/chat/completions` where upstream pinned the SDK's
//!   `baseURL`), and `cf-aig-authorization: Bearer test-token` rides the
//!   wire.
//! - The upstream `Authorization: null` and `"x-api-key": null` default
//!   headers mean the header is absent, not the string `"null"`:
//!   `ProviderHeaders` spells a suppressed header as a `None` value, and the
//!   wire record simply lacks the entry. The resolution assertions pin the
//!   `None` values; the wire assertions pin the absence.
//! - The registry's resolved auth restates into the completion options'
//!   `api_key`/`headers` slots, the fields upstream's compat `complete`
//!   consumed from the `auth` argument.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::types::{ApiKeyCredential, Credential, CredentialModifyFn};
use pi_ai::http::{MockHttpClient, MockResponse};
use pi_ai::models::{ModelsSimpleStreamOptions, ModelsStreamOptions};
use pi_ai::types::{SimpleStreamOptions, StopReason, StreamOptions, TransportOptions};
use pi_coding_agent::model_registry::{ModelRegistry, ResolvedRequestAuth};
use pi_coding_agent::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use serde_json::json;

/// The shared fixture module compiles whole into every test binary; the
/// config suites' env lookups stay unused in this one.
#[expect(
    dead_code,
    reason = "every test binary recompiles the shared fixture module and consumes only its own helpers"
)]
mod common;

use common::model_layer::{empty_auth_storage, empty_context};

/// The materialized gateway endpoint, upstream's asserted `baseURL`.
const GATEWAY_ENDPOINT: &str =
    "https://gateway.ai.cloudflare.com/v1/test-account/test-gateway/compat";
/// The gateway model id, upstream's `workers-ai/@cf/moonshotai/kimi-k2.6`.
const KIMI_MODEL_ID: &str = "workers-ai/@cf/moonshotai/kimi-k2.6";
/// The provider id the credential and model resolve under.
const PROVIDER_ID: &str = "cloudflare-ai-gateway";
/// The gateway key header's wire value, upstream's asserted default header.
const GATEWAY_TOKEN: &str = "Bearer test-token";

/// The SSE body the completions adapter parses, upstream's fake SDK stream:
/// one empty-delta `stop` chunk plus usage, closed by the `[DONE]` sentinel.
fn completions_sse_body() -> String {
    let chunk = json!({
        "choices": [{ "delta": {}, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1 },
    });
    format!("data: {chunk}\n\ndata: [DONE]\n\n")
}

/// The mock serving the completions endpoint with the body above.
fn completions_mock() -> MockHttpClient {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(
            MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(completions_sse_body()),
        );
    mock
}

/// The mock transport options the completions requests send, the injection
/// upstream's SDK-constructor mock stood in for.
fn transport(mock: &MockHttpClient) -> TransportOptions {
    TransportOptions {
        http_client: Some(Arc::new(mock.clone())),
        ..TransportOptions::default()
    }
}

/// The recorded request's header value, matched case-insensitively.
fn recorded_header(mock: &MockHttpClient, name: &str) -> Option<String> {
    mock.recorded()[0]
        .headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}

/// The recorded request materializes the gateway endpoint with the gateway
/// key header and no competing auth header, upstream's clientOptions
/// assertions restated onto the wire.
fn assert_gateway_request(mock: &MockHttpClient) {
    let request = &mock.recorded()[0];
    assert!(
        request.url.starts_with(GATEWAY_ENDPOINT),
        "got: {}",
        request.url,
    );
    assert_eq!(
        recorded_header(mock, "cf-aig-authorization").as_deref(),
        Some(GATEWAY_TOKEN),
    );
    assert!(
        recorded_header(mock, "Authorization").is_none(),
        "the suppressed Authorization header must not ride the wire",
    );
    assert!(
        recorded_header(mock, "x-api-key").is_none(),
        "the suppressed x-api-key header must not ride the wire",
    );
}

/// The runtime over the seeded gateway credential, upstream's
/// `createCloudflareRuntime`: the credential carries the key plus the
/// account/gateway env the endpoint placeholders materialize from.
async fn cloudflare_runtime() -> (ModelRuntime, ModelRegistry) {
    let storage = empty_auth_storage();
    let seed: CredentialModifyFn = Box::new(|_current| {
        Box::pin(async move {
            Ok(Some(Credential::ApiKey(ApiKeyCredential {
                key: Some("test-token".to_owned()),
                env: Some(BTreeMap::from([
                    (
                        "CLOUDFLARE_ACCOUNT_ID".to_owned(),
                        "test-account".to_owned(),
                    ),
                    (
                        "CLOUDFLARE_GATEWAY_ID".to_owned(),
                        "test-gateway".to_owned(),
                    ),
                ])),
            })))
        })
    });
    storage
        .modify(PROVIDER_ID, seed, None)
        .await
        .expect("the gateway credential seeds");
    let credentials: Arc<dyn CredentialStore> = storage.clone();
    let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(credentials),
        models_path: Some(None),
        allow_model_network: false,
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .expect("the runtime constructs");
    let registry = ModelRegistry::new(runtime.clone());
    (runtime, registry)
}

mod model_registry_cloudflare_compat_streaming {
    use super::*;

    /// Upstream "materializes the Cloudflare endpoint through ModelRuntime
    /// streaming": the simple completion resolves the credential, fills the
    /// endpoint placeholders, and binds the gateway key header.
    #[tokio::test]
    async fn materializes_the_cloudflare_endpoint_through_model_runtime_streaming() {
        let (runtime, _registry) = cloudflare_runtime().await;
        let model = runtime
            .get_model(PROVIDER_ID, KIMI_MODEL_ID)
            .expect("the gateway model is in the catalog");
        let mock = completions_mock();
        let options = ModelsSimpleStreamOptions {
            options: SimpleStreamOptions {
                transport_options: transport(&mock),
                ..SimpleStreamOptions::default()
            },
            transform_headers: None,
        };

        let result = runtime
            .complete_simple(&model, &empty_context(), Some(&options))
            .await;

        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_gateway_request(&mock);
    }

    /// Upstream "materializes the Cloudflare endpoint after extension-style
    /// auth resolution": the registry resolves the gateway headers (with the
    /// two auth headers suppressed, the nulls of upstream's assertion), and
    /// the completion those headers drive records the same request.
    #[tokio::test]
    async fn materializes_the_cloudflare_endpoint_after_extension_style_auth_resolution() {
        let (_runtime, registry) = cloudflare_runtime().await;
        let model = registry
            .find(PROVIDER_ID, KIMI_MODEL_ID)
            .expect("the gateway model is in the catalog");

        let auth = registry.get_api_key_and_headers(&model).await;
        assert!(
            auth.ok(),
            "the seeded gateway credential resolves: {auth:?}"
        );
        let resolution = match auth {
            ResolvedRequestAuth::Ok {
                api_key, headers, ..
            } => Ok((api_key, headers)),
            ResolvedRequestAuth::Err { error } => Err(error),
        };
        let (api_key, headers) = resolution.expect("the resolution carries its fields");
        assert_eq!(
            api_key, None,
            "the gateway key rides the header, not the key slot"
        );
        let headers = headers.expect("the gateway auth carries its header map");
        assert_eq!(
            headers
                .get("cf-aig-authorization")
                .map(|value| value.as_deref()),
            Some(Some(GATEWAY_TOKEN)),
        );
        assert_eq!(
            headers.get("Authorization").map(|value| value.as_deref()),
            Some(None),
            "the upstream null Authorization suppresses the header",
        );
        assert_eq!(
            headers.get("x-api-key").map(|value| value.as_deref()),
            Some(None),
            "the upstream null x-api-key suppresses the header",
        );

        let mock = completions_mock();
        let options = ModelsStreamOptions {
            options: StreamOptions {
                transport_options: transport(&mock),
                api_key,
                headers: Some(headers),
                ..StreamOptions::default()
            },
            transform_headers: None,
        };
        let result = registry
            .complete(&model, &empty_context(), Some(&options))
            .await;

        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_gateway_request(&mock);
    }
}
