//! Env-gated Anthropic Messages E2E suites, ported from the upstream
//! `anthropic-eager-tool-input-e2e`, `anthropic-long-cache-retention-e2e`,
//! `anthropic-thinking-binding-e2e`, `anthropic-opus-4-8-smoke`, and
//! `anthropic-thinking-disable` live probes at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Like upstream's `it.skipIf`, a probe without its provider credential
//! returns early; the catalog-coverage assertions always run.

#![expect(
    clippy::expect_used,
    reason = "the tests pin live outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::env_api_keys::{ANTHROPIC_API_KEY_ENV, get_env_api_key};
use pi_ai::models::{ModelsSimpleStreamOptions, ModelsStreamOptions, WithTransforms};
use pi_ai::types::{
    AssistantBlock, AssistantMessageEvent, CacheRetention, Context, Message, Model, ModelCompat,
    SimpleStreamOptions, StreamOptions, Tool, UserContent, UserMessage,
};

mod common;
use common::builtin_model;

/// The Models runtime the live probes stream through, upstream's
/// `createModels` over the builtin provider registry.
fn models() -> pi_ai::models::Models {
    common::models_runtime()
}

fn user_message(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: pi_ai::auth::resolve::now_ms(),
    })
}

/// The probe priority upstream's E2E selection uses: prefer cheap current
/// Claude 4 routes, most negative first, then id order.
fn probe_priority(model: &Model) -> f64 {
    let model_id = model.id.to_lowercase();
    let mut priority = model.cost.rates.input + model.cost.rates.output;
    if model_id.contains("haiku") && (model_id.contains("4-5") || model_id.contains("4.5")) {
        priority -= 1000.0;
    } else if model_id.contains("sonnet") && (model_id.contains("4-") || model_id.contains("4.")) {
        priority -= 750.0;
    } else if model_id.contains("claude") && (model_id.contains("4-") || model_id.contains("4.")) {
        priority -= 500.0;
    }
    priority
}

/// One cheapest-per-provider probe of the anthropic-messages models, the
/// port of upstream's `selectOneCasePerProvider`. Copilot rides OAuth token
/// stores the test env does not carry, so it probes nothing here.
fn probe_cases() -> Vec<(String, String, Model)> {
    let mut by_provider: std::collections::BTreeMap<String, Vec<Model>> =
        std::collections::BTreeMap::new();
    for provider in pi_ai::providers::all::builtin_providers() {
        for model in provider.get_models().unwrap_or_default() {
            if model.api == pi_ai::types::Api::from("anthropic-messages") {
                by_provider
                    .entry(provider.id().to_owned())
                    .or_default()
                    .push(model);
            }
        }
    }
    by_provider
        .into_iter()
        .filter_map(|(provider, models)| {
            let api_key = get_env_api_key(&provider, None)?;
            let mut models = models;
            models.sort_by(|a, b| {
                probe_priority(a)
                    .partial_cmp(&probe_priority(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
            Some((provider, api_key, models.remove(0)))
        })
        .collect()
}

/// The coverage assertion upstream's E2E suites open with: the probes name
/// one anthropic-messages model per provider, and the catalog the suites
/// resolve actually carries the models the conformance suites use.
#[test]
fn anthropic_messages_catalog_covers_the_builtin_registry() {
    let providers = pi_ai::providers::all::builtin_providers();
    let with_anthropic_messages = providers
        .iter()
        .filter(|provider| {
            provider
                .get_models()
                .unwrap_or_default()
                .iter()
                .any(|model| model.api == pi_ai::types::Api::from("anthropic-messages"))
        })
        .count();
    assert!(
        with_anthropic_messages >= 3,
        "expected the anthropic-messages registry across several providers, got {with_anthropic_messages}"
    );
    for id in [
        "claude-fable-5-1",
        "claude-opus-4-8",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ] {
        assert!(
            builtin_model("anthropic", id).api == pi_ai::types::Api::from("anthropic-messages"),
            "{id} speaks anthropic-messages"
        );
    }
}

/// The eager tool input streaming probes, upstream
/// `anthropic-eager-tool-input-e2e.test.ts`: a tool-enabled request with
/// per-tool `eager_input_streaming: true` is accepted on each provider's
/// cheapest route.
#[tokio::test]
async fn eager_tool_input_probes_accept_configured_tool_streaming() {
    let echo = Tool {
        name: "echo_value".to_owned(),
        description: "Echo a string value".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string", "description": "The value to echo" } },
            "required": ["value"],
        }),
        constrained_sampling: None,
    };
    for (provider, api_key, model) in probe_cases() {
        let context = Context {
            system_prompt: Some("You are a concise assistant. Use tools when useful.".to_owned()),
            messages: vec![user_message(
                "Call echo_value with value set to eager-input-streaming-compat.",
            )],
            tools: Some(vec![echo.clone()]),
        };
        let options = WithTransforms {
            options: StreamOptions {
                api_key: Some(api_key),
                max_tokens: Some(128),
                ..StreamOptions::default()
            },
            transform_headers: None,
        };
        let response = models().complete(&model, &context, Some(&options)).await;
        assert!(
            response.error_message.is_none(),
            "{provider}/{}: {:?}",
            model.id,
            response.error_message
        );
        assert_ne!(
            response.stop_reason,
            pi_ai::types::StopReason::Error,
            "{provider}"
        );
    }
}

/// The forced per-tool `eager_input_streaming` probes, upstream's second
/// probe loop.
#[tokio::test]
async fn forced_eager_input_streaming_probes_accept_the_forced_flag() {
    let echo = Tool {
        name: "echo_value".to_owned(),
        description: "Echo a string value".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string", "description": "The value to echo" } },
            "required": ["value"],
        }),
        constrained_sampling: None,
    };
    for (provider, api_key, model) in probe_cases() {
        let mut model = model;
        model.compat = Some(
            serde_json::from_value::<ModelCompat>(serde_json::json!({
                "supportsEagerToolInputStreaming": true,
            }))
            .expect("compat map"),
        );
        let context = Context {
            system_prompt: Some("You are a concise assistant. Use tools when useful.".to_owned()),
            messages: vec![user_message(
                "Call echo_value with value set to eager-input-streaming-compat.",
            )],
            tools: Some(vec![echo.clone()]),
        };
        let options = WithTransforms {
            options: StreamOptions {
                api_key: Some(api_key),
                max_tokens: Some(128),
                ..StreamOptions::default()
            },
            transform_headers: None,
        };
        let response = models().complete(&model, &context, Some(&options)).await;
        assert!(
            response.error_message.is_none(),
            "{provider}: {:?}",
            response.error_message
        );
        assert_ne!(
            response.stop_reason,
            pi_ai::types::StopReason::Error,
            "{provider}"
        );
    }
}

/// The long cache retention probe, upstream
/// `anthropic-long-cache-retention-e2e.test.ts` for the first-party
/// anthropic provider.
#[tokio::test]
async fn long_cache_retention_probe_accepts_the_forced_retention() {
    let Some(api_key) = get_env_api_key("anthropic", None) else {
        return;
    };
    let mut model = builtin_model("anthropic", "claude-haiku-4-5");
    model.compat = Some(
        serde_json::from_value::<ModelCompat>(serde_json::json!({
            "supportsLongCacheRetention": true,
        }))
        .expect("compat map"),
    );
    let options = ModelsStreamOptions {
        options: StreamOptions {
            api_key: Some(api_key),
            cache_retention: Some(CacheRetention::Long),
            max_tokens: Some(128),
            ..StreamOptions::default()
        },
        transform_headers: None,
    };
    let context = Context {
        system_prompt: Some("You are a concise assistant.".to_owned()),
        messages: vec![user_message(
            "Reply with exactly: long cache retention accepted",
        )],
        tools: None,
    };

    let response = models().complete(&model, &context, Some(&options)).await;

    assert!(
        response.error_message.is_none(),
        "got: {:?}",
        response.error_message
    );
    assert_ne!(response.stop_reason, pi_ai::types::StopReason::Error);
}

/// The thinking binding conformance probe, upstream
/// `anthropic-thinking-binding-e2e.test.ts`: managed effort markers are
/// required by signed Fable thinking, and the replay fails without them.
#[tokio::test]
async fn thinking_binding_replays_managed_effort_markers() {
    let Some(api_key) = std::env::var(ANTHROPIC_API_KEY_ENV)
        .ok()
        .filter(|key| !key.trim().is_empty())
    else {
        return;
    };
    let model = builtin_model("anthropic", "claude-fable-5-1");

    // strictBinding upstream: replay under the strict block binding.
    let request = |context: Context| {
        let hook = pi_ai::types::OnPayload::new(|mut payload, _model| {
            if let Some(thinking) = payload.get_mut("thinking")
                && thinking
                    .get("block_binding")
                    .is_some_and(serde_json::Value::is_object)
            {
                thinking["block_binding"]["prefix_mismatch_behavior"] = serde_json::json!("error");
            }
            let replacement = payload;
            Box::pin(async move { Some(replacement) })
        });
        let options = pi_ai::api::anthropic_messages::AnthropicStreamOptions {
            transport_options: pi_ai::types::TransportOptions {
                on_payload: Some(hook),
                ..pi_ai::types::TransportOptions::default()
            },
            api_key: Some(api_key.clone()),
            cache_retention: Some(CacheRetention::None),
            max_tokens: Some(1536),
            thinking_enabled: Some(true),
            thinking_display: Some(
                pi_ai::api::anthropic_messages::AnthropicThinkingDisplay::Summarized,
            ),
            ..pi_ai::api::anthropic_messages::AnthropicStreamOptions::default()
        };
        let model = model.clone();
        async move {
            pi_ai::api::anthropic_messages::stream(&model, &context, Some(&options))
                .result()
                .await
        }
    };

    let first_user =
        user_message("Compute 982451653 multiplied by 961748941. Return only the integer.");
    let first = request(Context {
        system_prompt: None,
        messages: vec![first_user.clone()],
        tools: None,
    })
    .await;
    assert_eq!(
        first.stop_reason,
        pi_ai::types::StopReason::Stop,
        "got: {:?}",
        first.error_message
    );
    assert!(first.content.iter().any(|block| matches!(
        block,
        AssistantBlock::Thinking(thinking)
            if thinking
                .thinking_signature
                .as_deref()
                .is_some_and(|signature| !signature.is_empty())
    )));
    assert_eq!(first.provider_thinking_level.as_deref(), Some("low"));

    let second_user = user_message("Reply with exactly: ok");
    let replay = request(Context {
        system_prompt: None,
        messages: vec![
            first_user.clone(),
            Message::Assistant(first.clone()),
            second_user.clone(),
        ],
        tools: None,
    })
    .await;
    assert_eq!(replay.stop_reason, pi_ai::types::StopReason::Stop);

    let mut unmanaged_history = first.clone();
    unmanaged_history.provider_thinking_level = None;
    let missing_marker = request(Context {
        system_prompt: None,
        messages: vec![
            first_user.clone(),
            Message::Assistant(unmanaged_history),
            second_user.clone(),
        ],
        tools: None,
    })
    .await;
    assert_eq!(missing_marker.stop_reason, pi_ai::types::StopReason::Error);
    assert!(
        missing_marker
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("Invalid `signature`")),
        "got: {:?}",
        missing_marker.error_message
    );
}

/// The Opus 4.8 smoke probe, upstream `anthropic-opus-4-8-smoke.test.ts`.
#[tokio::test]
async fn opus_4_8_smoke_streams_with_reasoning_enabled() {
    let Some(api_key) = std::env::var(ANTHROPIC_API_KEY_ENV)
        .ok()
        .filter(|key| !key.trim().is_empty())
    else {
        return;
    };
    let model = builtin_model("anthropic", "claude-opus-4-8");
    let context = Context {
        system_prompt: Some("You are a precise assistant.".to_owned()),
        messages: vec![user_message(
            "Compute 48291 * 7317 and reply with exactly: sum=<sum>",
        )],
        tools: None,
    };
    let options = ModelsSimpleStreamOptions {
        options: SimpleStreamOptions {
            api_key: Some(api_key),
            reasoning: Some(pi_ai::types::ThinkingLevel::High),
            max_tokens: Some(1024),
            ..SimpleStreamOptions::default()
        },
        transform_headers: None,
    };

    let response = models()
        .complete_simple(&model, &context, Some(&options))
        .await;

    assert_eq!(response.stop_reason, pi_ai::types::StopReason::Stop);
    assert!(response.error_message.is_none());
    let thinking = response
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::Thinking(thinking) => Some(thinking),
            _ => None,
        })
        .expect("thinking block from Claude Opus 4.8");
    assert!(
        thinking
            .thinking_signature
            .as_deref()
            .is_some_and(|signature| !signature.is_empty()),
        "expected a thinking signature"
    );
}

/// The thinking disable E2E probe, upstream
/// `anthropic-thinking-disable.test.ts`: a reasoning model with thinking
/// off emits no thinking events.
#[tokio::test]
async fn thinking_disable_emits_no_thinking_events() {
    let Some(api_key) = std::env::var(ANTHROPIC_API_KEY_ENV)
        .ok()
        .filter(|key| !key.trim().is_empty())
    else {
        return;
    };
    let model = builtin_model("anthropic", "claude-sonnet-4-5");
    let context = Context {
        system_prompt: Some(
            "You are a precise assistant. Follow the requested output format exactly.".to_owned(),
        ),
        messages: vec![user_message(
            "Before replying, carefully solve 36863 * 5279 internally. Then reply with the word pong repeated exactly 40 times, separated by single spaces. Do not add any other text.",
        )],
        tools: None,
    };
    let options = ModelsSimpleStreamOptions {
        options: SimpleStreamOptions {
            api_key: Some(api_key),
            temperature: Some(0.0),
            max_tokens: Some(160),
            ..SimpleStreamOptions::default()
        },
        transform_headers: None,
    };
    let stream = models().stream_simple(&model, &context, Some(&options));

    let mut thinking_events = 0u64;
    let mut thinking_chars = 0u64;
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::ThinkingStart { .. }
            | AssistantMessageEvent::ThinkingEnd { .. } => thinking_events += 1,
            AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                thinking_events += 1;
                thinking_chars += delta.chars().count() as u64;
            }
            _ => {}
        }
    }
    let response = stream.result().await;

    assert_eq!(response.stop_reason, pi_ai::types::StopReason::Stop);
    assert_eq!(
        thinking_events, 0,
        "no thinking events when thinking is off"
    );
    assert_eq!(thinking_chars, 0);
    assert!(
        !response
            .content
            .iter()
            .any(|block| matches!(block, AssistantBlock::Thinking(_))),
        "no thinking blocks in the final content"
    );
}
