//! Env-gated OpenAI Responses cache-affinity E2E probe, ported from the
//! upstream `openai-responses-cache-affinity-e2e.test.ts` live probe at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Like upstream's `describe.skipIf`, a probe without its provider
//! credential returns early; the catalog-coverage assertion always runs.

use pi_ai::models::WithTransforms;
use pi_ai::types::{Context, StopReason, StreamOptions};

mod common;
use common::{builtin_model, models_runtime, user_message_now};

/// The catalog-coverage assertion the live probes open with: the affinity
/// probe's model rides the OpenAI Responses wire.
#[test]
fn openai_responses_catalog_carries_the_affinity_probe_model() {
    common::assert_catalog_api("openai", "gpt-5.4", "openai-responses");
    let model = builtin_model("openai", "gpt-5.4");
    assert_eq!(model.provider, pi_ai::types::ProviderId::from("openai"));
}

/// The aligned-identifier live probe, upstream's single case: a direct
/// OpenAI Responses request with the same session id rides the aligned
/// cache-affinity routing and answers with the requested text.
#[tokio::test]
async fn direct_openai_responses_requests_with_aligned_cache_affinity_identifiers() {
    let Some(api_key) = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|key| !key.trim().is_empty())
    else {
        return;
    };
    let model = builtin_model("openai", "gpt-5.4");
    let context = Context {
        system_prompt: Some("You are a helpful assistant. Reply exactly as requested.".to_owned()),
        messages: vec![user_message_now(
            "Reply with exactly: openai cache affinity e2e success",
        )],
        tools: None,
    };
    let options = WithTransforms {
        options: StreamOptions {
            api_key: Some(api_key),
            session_id: Some("0195d6e4-4cf9-7f44-a2d8-f8f7f49ee9d3".to_owned()),
            ..StreamOptions::default()
        },
        transform_headers: None,
    };

    let response = models_runtime()
        .complete(&model, &context, Some(&options))
        .await;

    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "{:?}",
        response.error_message
    );
    assert!(
        response.error_message.is_none(),
        "{:?}",
        response.error_message
    );
    common::assert_live_text_reply(&response, "openai cache affinity e2e success");
}
