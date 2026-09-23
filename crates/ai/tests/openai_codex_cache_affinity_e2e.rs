//! Env-gated OpenAI Codex cache-affinity E2E probe, ported from the
//! upstream `openai-codex-cache-affinity-e2e.test.ts` live probe at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatement: upstream gates the probe on the persisted OAuth
//! store's codex token (`resolveApiKey("openai-codex")` over
//! `it.skipIf`); the port's codex credential rides the same env-key seam
//! the other probes use, and the ChatGPT OAuth route is not ported yet, so
//! the probe runs only when that seam resolves a credential. The
//! catalog-coverage assertion always runs.

use pi_ai::api::openai_codex_responses::{OpenAiCodexResponsesOptions, stream as codex_stream};
use pi_ai::env_api_keys::get_env_api_key;
use pi_ai::types::{Context, StopReason};

mod common;
use common::{builtin_model, drain_and_settle};

/// The catalog-coverage assertion the live probes open with: the affinity
/// probe's model rides the codex wire.
#[test]
fn openai_codex_catalog_carries_the_affinity_probe_model() {
    let model = builtin_model("openai-codex", "gpt-5.5");
    assert_eq!(model.api, pi_ai::types::Api::from("openai-codex-responses"));
    assert_eq!(
        model.provider,
        pi_ai::types::ProviderId::from("openai-codex")
    );
}

/// The SSE live probe, upstream's single case: the forced-SSE transport
/// with the session id rides the aligned cache-affinity routing and answers
/// with the requested text.
#[tokio::test]
async fn sse_requests_with_aligned_cache_affinity_identifiers() {
    let Some(api_key) = get_env_api_key("openai-codex", None).filter(|key| !key.trim().is_empty())
    else {
        return;
    };
    let model = builtin_model("openai-codex", "gpt-5.5");
    let context = Context {
        system_prompt: Some("You are a helpful assistant. Reply exactly as requested.".to_owned()),
        messages: vec![common::user_message_now(
            "Reply with exactly: cache affinity e2e success",
        )],
        tools: None,
    };
    let options = OpenAiCodexResponsesOptions {
        api_key: Some(api_key),
        session_id: Some("0195d6e4-4cf9-7f44-a2d8-f8f7f49ee9d3".to_owned()),
        transport: Some(pi_ai::types::Transport::Sse),
        ..OpenAiCodexResponsesOptions::default()
    };

    let stream = codex_stream(&model, &context, Some(&options));
    let response = drain_and_settle(&stream).await;

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
    common::assert_live_text_reply(&response, "cache affinity e2e success");
}
