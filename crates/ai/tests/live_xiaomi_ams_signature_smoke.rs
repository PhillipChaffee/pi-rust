//! Env-gated Xiaomi Token Plan AMS Anthropic empty-signature smoke, ported
//! from the upstream
//! `xiaomi-token-plan-ams-anthropic-empty-signature-smoke.test.ts` probe —
//! the `Xiaomi Token Plan AMS Anthropic empty thinking signature smoke`
//! describe — at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Like upstream's `describe.skipIf`, a probe without the
//! `XIAOMI_TOKEN_PLAN_AMS_API_KEY` credential returns early.
//!
//! Porting restatement: upstream hand-builds the `mimo-v2.5-pro` Anthropic
//! smoke model; the catalog entry for the provider speaks openai-completions,
//! so the port resolves the catalog model and retargets the fields upstream
//! pins. Upstream's throwing `PayloadCaptured` hook stops the replay request
//! before the wire; a payload hook cannot abort in the port, so the replay
//! rides live and the probe asserts on the captured payload alone.

#![expect(
    clippy::expect_used,
    reason = "the tests pin live outcomes; an unexpected shape panics the test by design"
)]
// binary and carries its own expects only for the lints its own code knows
// it trips; these cover the rest so the suite's gate runs clean without
// editing the shared file. Stale entries fail the build on their own.

use std::sync::{Arc, Mutex};

use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, Context, Message, Model, ModelCompat, ModelCost,
    ModelCostRates, OnPayload, StopReason, ThinkingContent, ThinkingLevel,
};
use serde_json::{Value, json};

mod common;
use common::live;

/// The payload slot the capture hook fills, shared between the hook and the
/// assertion, upstream's `capturedPayload` local.
type CapturedPayload = Arc<Mutex<Option<Value>>>;

/// The smoke model, upstream's hand-built `Model<"anthropic-messages">`: the
/// catalog's `mimo-v2.5-pro` retargeted onto the provider's Anthropic
/// endpoint, with the name, token cap, and cost upstream pins and the
/// `allowEmptySignature` compat the endpoint needs.
fn smoke_model() -> Model {
    Model {
        name: String::from("MiMo-V2.5-Pro Anthropic smoke"),
        api: Api::from("anthropic-messages"),
        base_url: String::from("https://token-plan-ams.xiaomimimo.com/anthropic"),
        max_tokens: 1024,
        cost: ModelCost {
            rates: ModelCostRates {
                input: 1.0,
                output: 3.0,
                cache_read: 0.2,
                cache_write: 0.0,
            },
            tiers: None,
        },
        compat: Some(
            serde_json::from_value::<ModelCompat>(json!({ "allowEmptySignature": true }))
                .expect("compat map"),
        ),
        ..live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro")
    }
}

/// The first-turn context, upstream's `makeInitialContext`.
fn make_initial_context() -> Context {
    Context {
        system_prompt: Some(
            "You are concise. Follow the requested output format exactly.".to_owned(),
        ),
        messages: vec![live::user_message(
            "Think internally if you need to, then reply with exactly this text and nothing \
             else: first-ok",
        )],
        tools: None,
    }
}

/// The thinking blocks of a settled message, upstream's `getThinkingBlocks`.
fn thinking_blocks(message: &AssistantMessage) -> Vec<&ThinkingContent> {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Thinking(thinking) => Some(thinking),
            _ => None,
        })
        .collect()
}

/// The replay-payload capture hook, upstream's capture `onPayload`: record
/// the payload and keep it unchanged. Upstream throws `PayloadCaptured` to
/// stop the request before the wire; a control the port's payload hook
/// cannot express, so the replay rides live.
fn capture_hook(slot: CapturedPayload) -> OnPayload {
    OnPayload::new(move |payload, _model| {
        *slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(payload);
        Box::pin(async { None })
    })
}

/// The replay request whose outgoing payload the hook captures, upstream's
/// `captureReplayPayload`.
async fn capture_replay_payload(model: &Model, context: &Context, api_key: &str) -> Value {
    let captured: CapturedPayload = Arc::new(Mutex::new(None));
    let options = live::LiveOptions {
        api_key: Some(api_key.to_owned()),
        reasoning: Some(ThinkingLevel::High),
        on_payload: Some(capture_hook(Arc::clone(&captured))),
        ..live::LiveOptions::default()
    };
    live::complete(model, context, &options).await;

    captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("Expected payload capture before request")
}

/// Upstream it "reproduces empty thinking signatures and preserves them for
/// replay": the first turn emits an empty-signature thinking block, and the
/// replayed request carries it back as a thinking block with `signature: ""`
/// instead of converting the thinking to text.
#[tokio::test]
async fn reproduces_empty_thinking_signatures_and_preserves_them_for_replay() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    let model = smoke_model();

    let first_context = make_initial_context();
    let first_options = live::LiveOptions {
        api_key: Some(api_key.clone()),
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    let first = live::complete(&model, &first_context, &first_options).await;

    assert_eq!(
        first.stop_reason,
        StopReason::Stop,
        "Error: {:?}",
        first.error_message
    );

    let thinking_blocks = thinking_blocks(&first);
    assert!(!thinking_blocks.is_empty(), "thinking blocks arrived");
    assert!(
        thinking_blocks
            .iter()
            .any(|block| block.thinking_signature.as_deref() == Some("")),
        "an empty thinking signature rode the response"
    );

    let mut replay_messages = first_context.messages.clone();
    replay_messages.push(Message::Assistant(first.clone()));
    replay_messages.push(live::user_message(
        "Reply with exactly this text and nothing else: second-ok",
    ));
    let replay_context = Context {
        system_prompt: first_context.system_prompt.clone(),
        messages: replay_messages,
        tools: None,
    };

    let replay_payload = capture_replay_payload(&model, &replay_context, &api_key).await;
    let assistant_payload = replay_payload["messages"]
        .as_array()
        .and_then(|messages| {
            messages
                .iter()
                .find(|message| message["role"] == "assistant")
        })
        .expect("an assistant message in the replay payload");
    assert!(
        assistant_payload["content"].is_array(),
        "the replayed assistant content is an array: {assistant_payload}"
    );
    let content = assistant_payload["content"]
        .as_array()
        .expect("the content is an array");
    let replayed_thinking: Vec<Value> = content
        .iter()
        .filter(|block| block["type"] == json!("thinking"))
        .cloned()
        .collect();
    let replayed_text: Vec<Value> = content
        .iter()
        .filter(|block| block["type"] == json!("text"))
        .cloned()
        .collect();
    assert_eq!(
        replayed_thinking,
        vec![json!({
            "type": "thinking",
            "thinking": thinking_blocks[0].thinking,
            "signature": "",
        })],
        "the empty signature is preserved end to end"
    );
    assert!(
        !replayed_text
            .iter()
            .any(|block| block["text"].as_str() == Some(thinking_blocks[0].thinking.as_str())),
        "the thinking text is not replayed as a text block"
    );
}
