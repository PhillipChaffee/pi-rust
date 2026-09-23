//! The `responseId` E2E suite, ported from `packages/ai/test/
//! responseid.test.ts` at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`:
//! every provider's settled completion must surface the response id the
//! provider sent, serialized as `responseId` on the assistant message.
//!
//! Like upstream's `it.skipIf`, a probe without its provider credential
//! returns early: the env-var gates read the process environment exactly as
//! upstream's `skipIf(!process.env.X)`, and the OAuth-backed probes resolve
//! through [`common::live::resolve_api_key`] — the pi credential store first,
//! then the provider's env vars.

#![expect(
    clippy::expect_used,
    reason = "the tests pin live outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::types::{Api, Context, Model, StopReason};

mod common;
use common::live;

/// The completion probe upstream's `expectResponseId`: one short reply whose
/// settled message must carry a truthy `responseId`.
async fn expect_response_id(model: &Model, options: &live::LiveOptions) {
    let context = Context {
        system_prompt: Some("You are a helpful assistant. Be concise.".to_owned()),
        messages: vec![live::user_message("Reply with exactly: response id test")],
        tools: None,
    };

    let response = live::complete(model, &context, options).await;

    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "error: {:?}",
        response.error_message
    );
    let response_id = response
        .response_id
        .expect("responseId set on the settled response");
    assert!(!response_id.is_empty(), "responseId truthy");
}

/// Upstream `describe.skipIf(!process.env.GEMINI_API_KEY)` "Google Provider" /
/// `it` "should expose responseId".
#[tokio::test]
async fn google_provider_should_expose_response_id() {
    if live::env_key("GEMINI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("google", "gemini-2.5-flash");
    expect_response_id(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe` "Google Vertex Provider" /
/// `it.skipIf(!isVertexConfigured)` "should expose responseId with ADC":
/// application-default credentials ride the project and location options.
#[tokio::test]
async fn google_vertex_provider_should_expose_response_id_with_adc() {
    let vertex_project =
        live::env_key("GOOGLE_CLOUD_PROJECT").or_else(|| live::env_key("GCLOUD_PROJECT"));
    let vertex_location = live::env_key("GOOGLE_CLOUD_LOCATION");
    let (Some(vertex_project), Some(vertex_location)) = (vertex_project, vertex_location) else {
        return;
    };
    let llm = live::model("google-vertex", "gemini-3-flash-preview");
    let options = live::LiveOptions {
        vertex_project: Some(vertex_project),
        vertex_location: Some(vertex_location),
        ..live::LiveOptions::default()
    };
    expect_response_id(&llm, &options).await;
}

/// Upstream `describe` "Google Vertex Provider" / `it.skipIf(!vertexApiKey)`
/// "should expose responseId with API key".
#[tokio::test]
async fn google_vertex_provider_should_expose_response_id_with_api_key() {
    let Some(vertex_api_key) = live::env_key("GOOGLE_CLOUD_API_KEY") else {
        return;
    };
    let llm = live::model("google-vertex", "gemini-3-flash-preview");
    expect_response_id(&llm, &live::LiveOptions::key(vertex_api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Completions
/// Provider" / `it` "should expose responseId": the catalog model retargeted
/// at the openai-completions wire, upstream's `{ ...getModel(...), api }`
/// spread.
#[tokio::test]
async fn openai_completions_provider_should_expose_response_id() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let llm = Model {
        api: Api::from("openai-completions"),
        ..live::model("openai", "gpt-4o-mini")
    };
    expect_response_id(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Responses
/// Provider" / `it` "should expose responseId".
#[tokio::test]
async fn openai_responses_provider_should_expose_response_id() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("openai", "gpt-5-mini");
    expect_response_id(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.ANTHROPIC_API_KEY)` "Anthropic
/// Provider" / `it` "should expose responseId".
#[tokio::test]
async fn anthropic_provider_should_expose_response_id() {
    if live::env_key("ANTHROPIC_API_KEY").is_none() {
        return;
    }
    let llm = live::model("anthropic", "claude-sonnet-4-5");
    expect_response_id(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasAzureOpenAICredentials())` "Azure OpenAI
/// Responses Provider" / `it` "should expose responseId": the deployment-name
/// map override rides when the catalog model has an entry, upstream's
/// `azureDeploymentName ? { azureDeploymentName } : {}`.
#[tokio::test]
async fn azure_openai_responses_provider_should_expose_response_id() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = live::LiveOptions {
        azure_deployment_name: live::azure_deployment_name(&llm.id),
        ..live::LiveOptions::default()
    };
    expect_response_id(&llm, &options).await;
}

/// Upstream `describe.skipIf(!process.env.MISTRAL_API_KEY)` "Mistral Provider"
/// / `it` "should expose responseId".
#[tokio::test]
async fn mistral_provider_should_expose_response_id() {
    if live::env_key("MISTRAL_API_KEY").is_none() {
        return;
    }
    let llm = live::model("mistral", "devstral-medium-latest");
    expect_response_id(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe` "GitHub Copilot Provider" /
/// `it.skipIf(!githubCopilotToken)` "OpenAI path should expose responseId".
#[tokio::test]
async fn github_copilot_provider_openai_path_should_expose_response_id() {
    let Some(github_copilot_token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "gpt-5.3-codex");
    expect_response_id(&llm, &live::LiveOptions::key(github_copilot_token)).await;
}

/// Upstream `describe` "GitHub Copilot Provider" /
/// `it.skipIf(!githubCopilotToken)` "Anthropic path should expose responseId".
#[tokio::test]
async fn github_copilot_provider_anthropic_path_should_expose_response_id() {
    let Some(github_copilot_token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    expect_response_id(&llm, &live::LiveOptions::key(github_copilot_token)).await;
}

/// Upstream `describe` "OpenAI Codex Provider" /
/// `it.skipIf(!openaiCodexToken)` "should expose responseId".
#[tokio::test]
async fn openai_codex_provider_should_expose_response_id() {
    let Some(openai_codex_token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    expect_response_id(&llm, &live::LiveOptions::key(openai_codex_token)).await;
}
