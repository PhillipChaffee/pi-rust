//! Direct-adapter authentication, ported from `test/pre-generation-error.test.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: every wire API's
//! `streamSimple` fails with the provider-named setup error before any
//! request dispatches when the options carry no key.
//!
//! Upstream throws synchronously; the port encodes the failure as a settled
//! error stream per the stream contract (each adapter documents the
//! restatement), so the assertion reads the settled message.

mod common;

use pi_ai::api::anthropic_messages as anthropic;
use pi_ai::api::azure_openai_responses as azure;
use pi_ai::api::google_generative_ai as google;
use pi_ai::api::mistral_conversations as mistral;
use pi_ai::api::openai_codex_responses as codex;
use pi_ai::api::openai_completions as completions;
use pi_ai::api::openai_responses as responses;
use pi_ai::types::{Api, Context, SimpleStreamOptions, StopReason};

fn model(api: &str) -> pi_ai::types::Model {
    let mut model = common::fixture_model();
    model.api = Api::from(api);
    model.provider = pi_ai::types::ProviderId::from("test-provider");
    model
}

fn expect_missing_auth(settled: &pi_ai::types::AssistantMessage) {
    assert_eq!(settled.stop_reason, StopReason::Error);
    assert_eq!(
        settled.error_message.as_deref(),
        Some("No API key for provider: test-provider")
    );
}

#[tokio::test]
async fn the_anthropic_stream_fails_without_auth() {
    let result = common::drain_and_settle(&anthropic::stream_simple(
        &model("anthropic-messages"),
        &Context::default(),
        Some(&SimpleStreamOptions::default()),
    ))
    .await;
    expect_missing_auth(&result);
}

#[tokio::test]
async fn the_azure_stream_fails_without_auth() {
    let result = common::drain_and_settle(&azure::stream_simple(
        &model("azure-openai-responses"),
        &Context::default(),
        Some(&SimpleStreamOptions::default()),
    ))
    .await;
    expect_missing_auth(&result);
}

#[tokio::test]
async fn the_google_stream_fails_without_auth() {
    let result = common::drain_and_settle(&google::stream_simple(
        &model("google-generative-ai"),
        &Context::default(),
        Some(&SimpleStreamOptions::default()),
    ))
    .await;
    expect_missing_auth(&result);
}

#[tokio::test]
async fn the_mistral_stream_fails_without_auth() {
    let result = common::drain_and_settle(&mistral::stream_simple(
        &model("mistral-conversations"),
        &Context::default(),
        Some(&SimpleStreamOptions::default()),
    ))
    .await;
    expect_missing_auth(&result);
}

#[tokio::test]
async fn the_codex_stream_fails_without_auth() {
    let result = common::drain_and_settle(&codex::stream_simple(
        &model("openai-codex-responses"),
        &Context::default(),
        Some(&SimpleStreamOptions::default()),
    ))
    .await;
    expect_missing_auth(&result);
}

#[tokio::test]
async fn the_completions_stream_fails_without_auth() {
    let result = common::drain_and_settle(&completions::stream_simple(
        &model("openai-completions"),
        &Context::default(),
        Some(&SimpleStreamOptions::default()),
    ))
    .await;
    expect_missing_auth(&result);
}

#[tokio::test]
async fn the_responses_stream_fails_without_auth() {
    let result = common::drain_and_settle(&responses::stream_simple(
        &model("openai-responses"),
        &Context::default(),
        Some(&SimpleStreamOptions::default()),
    ))
    .await;
    expect_missing_auth(&result);
}
