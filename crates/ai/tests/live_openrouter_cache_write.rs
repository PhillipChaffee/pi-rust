//! Env-gated OpenRouter cache-write repro E2E, ported from the upstream
//! `openrouter-cache-write-repro.test.ts` probe (the `OpenRouter cache_write
//! repro E2E` describe) at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Like upstream's `describe.skipIf`, a probe without `OPENROUTER_API_KEY`
//! returns early. The on-payload hook stamps the last user message's text
//! with an ephemeral `cache_control` marker, upstream's payload mutation.

// binary and carries its own expects only for the lints its own code knows
// it trips; these cover the rest so the suite's gate runs clean without
// editing the shared file. Stale entries fail the build on their own.

use std::sync::atomic::{AtomicU64, Ordering};

use pi_ai::auth::resolve::now_ms;
use pi_ai::types::{Context, OnPayload, StopReason};
use serde_json::{Value, json};

mod common;
use common::live;

/// The prompt-caching probe paragraph, upstream's repeated filler.
const PROBE_PARAGRAPH: &str = "Prompt-caching probe content. Keep this exact text stable \
     across requests so the provider can reuse prefix tokens and report cache read and \
     cache write usage.";

/// The process counter standing in for upstream's `Math.random()` suffix: the
/// nonce only keeps repeat calls off a shared cache prefix, so a
/// deterministic sequence preserves that purpose.
fn next_nonce() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The long system prompt, upstream's `createLongSystemPrompt`: the cache
/// nonce plus 80 copies of the stable probe paragraph joined by blank lines.
fn create_long_system_prompt() -> String {
    let nonce = format!("{}-{}", now_ms(), next_nonce());
    let filler = std::iter::repeat_n(PROBE_PARAGRAPH, 80)
        .collect::<Vec<_>>()
        .join("\n\n");
    format!("You are a concise assistant.\nCache nonce: {nonce}\n\n{filler}")
}

/// The cache-control stamping hook, upstream's `onPayload`: walk the messages
/// from the end, take the first user message, and mark its last text part
/// `cache_control: { type: "ephemeral" }`, replacing string content with a
/// one-part block array; a user message with neither string nor array content
/// falls through to the next older one.
fn cache_control_hook() -> OnPayload {
    OnPayload::new(|mut payload, _model| {
        let Some(messages) = payload.get_mut("messages").and_then(Value::as_array_mut) else {
            return Box::pin(async move { Some(payload) });
        };
        for message in messages.iter_mut().rev() {
            if message.get("role").and_then(Value::as_str) != Some("user") {
                continue;
            }
            if message.get("content").is_some_and(Value::is_string) {
                let text = message["content"].as_str().unwrap_or_default().to_owned();
                message["content"] = json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": { "type": "ephemeral" },
                }]);
                break;
            }
            let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts.iter_mut().rev() {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    part["cache_control"] = json!({ "type": "ephemeral" });
                    break;
                }
            }
            break;
        }
        Box::pin(async move { Some(payload) })
    })
}

/// Upstream it "regression: preserves `cache_write_tokens` on
/// `openai-completions` stream path": two identical calls over the
/// cache_control-marked prompt, and at least one reports `cache_write`
/// usage from the provider.
#[tokio::test]
async fn preserves_cache_write_tokens_on_openai_completions_stream_path() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let model = live::model("openrouter", "google/gemini-2.5-flash");
    let context = Context {
        system_prompt: Some(create_long_system_prompt()),
        messages: vec![live::user_message("Reply with exactly: OK")],
        tools: None,
    };
    let options = live::LiveOptions {
        api_key: Some(api_key),
        on_payload: Some(cache_control_hook()),
        ..live::LiveOptions::default()
    };

    let first = live::complete(&model, &context, &options).await;
    assert_eq!(
        first.stop_reason,
        StopReason::Stop,
        "Error: {:?}",
        first.error_message
    );

    let second = live::complete(&model, &context, &options).await;
    assert_eq!(
        second.stop_reason,
        StopReason::Stop,
        "Error: {:?}",
        second.error_message
    );

    assert!(
        first.usage.cache_write > 0 || second.usage.cache_write > 0,
        "the cache_control marker created cache on at least one call: \
         first cache_write {}, second cache_write {}",
        first.usage.cache_write,
        second.usage.cache_write
    );
}
