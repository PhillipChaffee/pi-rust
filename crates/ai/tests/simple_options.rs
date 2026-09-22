//! Simple-request option shaping, ported from the option-shaping parts of
//! `packages/ai/test/reasoning-options.test.ts` and
//! `packages/ai/test/sampling-options.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (upstream has no dedicated
//! simple-options suite; these pin the module's contract directly).

#![expect(
    clippy::expect_used,
    reason = "the tests pin shaping outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::simple_options::{
    adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context,
    clamp_reasoning, clamp_thinking_budget_to_answer_room, default_thinking_budgets,
    thinking_budget_for_level,
};
use pi_ai::types::{Context, Model, SimpleStreamOptions, ThinkingBudgets, ThinkingLevel};

fn fixture_model() -> Model {
    Model {
        id: "m".to_owned(),
        name: "m".to_owned(),
        api: pi_ai::types::Api::from("test-api"),
        provider: pi_ai::types::ProviderId::from("p"),
        base_url: "https://example.test/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 0,
        max_tokens: 100,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn windowed_model() -> Model {
    Model {
        context_window: 10_000,
        ..fixture_model()
    }
}

fn context() -> Context {
    Context {
        messages: vec![pi_ai::types::Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::UserContent::Text("hi".to_owned()),
            timestamp: 1,
        })],
        ..Context::default()
    }
}

#[test]
fn models_without_a_context_window_clamp_to_the_floor() {
    let model = fixture_model();
    assert_eq!(
        clamp_max_tokens_to_context(&model, &context(), 5_000),
        5_000
    );
    assert_eq!(
        clamp_max_tokens_to_context(&model, &context(), 0),
        pi_ai::api::simple_options::MIN_MAX_TOKENS,
    );
}

#[test]
fn the_context_window_minus_the_safety_margin_caps_max_tokens() {
    let model = windowed_model();
    let used = pi_ai::utils::estimate::estimate_context_tokens(&context())
        .tokens
        .saturating_add(pi_ai::api::simple_options::CONTEXT_SAFETY_TOKENS);
    let available = model.context_window.saturating_sub(used);
    assert_eq!(
        clamp_max_tokens_to_context(&model, &context(), 5_000),
        5_000.min(pi_ai::api::simple_options::MIN_MAX_TOKENS.max(available)),
    );
}

#[test]
fn base_options_without_a_request_clamp_the_model_cap() {
    let model = windowed_model();
    let options = build_base_options(&model, &context(), None, None);
    assert_eq!(
        options.max_tokens,
        Some(clamp_max_tokens_to_context(
            &model,
            &context(),
            model.max_tokens
        )),
    );
    assert_eq!(options.api_key, None);
}

#[test]
fn sampling_params_merge_model_defaults_under_request_overrides() {
    let mut model = windowed_model();
    model.sampling_params = Some(
        [
            ("temperature".to_owned(), serde_json::json!(0.2)),
            ("top_p".to_owned(), serde_json::json!(0.9)),
        ]
        .into_iter()
        .collect(),
    );
    let request = SimpleStreamOptions {
        sampling_params: Some(
            [
                ("temperature".to_owned(), serde_json::json!(0.7)),
                ("top_k".to_owned(), serde_json::json!(40)),
            ]
            .into_iter()
            .collect(),
        ),
        ..SimpleStreamOptions::default()
    };
    let options = build_base_options(&model, &context(), Some(&request), Some("key"));

    let merged = options.sampling_params.expect("merged params");
    assert_eq!(merged["temperature"], serde_json::json!(0.7));
    assert_eq!(merged["top_p"], serde_json::json!(0.9));
    assert_eq!(merged["top_k"], serde_json::json!(40));
    assert_eq!(options.api_key.as_deref(), Some("key"));
}

#[test]
fn sampling_params_survive_a_model_without_defaults() {
    let request = SimpleStreamOptions {
        sampling_params: Some(
            std::iter::once(("temperature".to_owned(), serde_json::json!(0.5))).collect(),
        ),
        ..SimpleStreamOptions::default()
    };
    let options = build_base_options(&fixture_model(), &context(), Some(&request), None);
    assert_eq!(
        options.sampling_params.expect("request params")["temperature"],
        serde_json::json!(0.5)
    );
}

#[test]
fn clamp_reasoning_clamps_xhigh_and_max_to_high() {
    for (level, expected) in [
        (Some(ThinkingLevel::Xhigh), Some(ThinkingLevel::High)),
        (Some(ThinkingLevel::Max), Some(ThinkingLevel::High)),
        (Some(ThinkingLevel::Medium), Some(ThinkingLevel::Medium)),
        (None, None),
    ] {
        assert_eq!(clamp_reasoning(level), expected);
    }
}

#[test]
fn thinking_budgets_follow_the_level_defaults() {
    let defaults = default_thinking_budgets();
    assert_eq!(
        thinking_budget_for_level(ThinkingLevel::Minimal, None),
        defaults.minimal.expect("the minimal default"),
    );
    assert_eq!(
        thinking_budget_for_level(ThinkingLevel::Low, None),
        defaults.low.expect("the low default"),
    );
    assert_eq!(
        thinking_budget_for_level(ThinkingLevel::Medium, None),
        defaults.medium.expect("the medium default"),
    );
    assert_eq!(
        thinking_budget_for_level(ThinkingLevel::High, None),
        defaults.high.expect("the high default"),
    );
    // xhigh and max clamp onto the high budget.
    assert_eq!(
        thinking_budget_for_level(ThinkingLevel::Max, None),
        defaults.high.expect("the high default"),
    );
}

#[test]
fn custom_budgets_override_only_their_levels() {
    let custom = ThinkingBudgets {
        medium: Some(123),
        ..ThinkingBudgets::default()
    };
    assert_eq!(
        thinking_budget_for_level(ThinkingLevel::Medium, Some(&custom)),
        123,
    );
    // Unset levels keep the defaults.
    assert_eq!(
        thinking_budget_for_level(ThinkingLevel::Low, Some(&custom)),
        default_thinking_budgets().low.expect("the low default"),
    );
}

#[test]
fn the_answer_room_caps_the_thinking_budget() {
    assert_eq!(clamp_thinking_budget_to_answer_room(2_048, 4_096), 2_048);
    assert_eq!(clamp_thinking_budget_to_answer_room(4_096, 4_096), 3_072);
    // A ceiling below the answer floor saturates the budget to zero.
    assert_eq!(clamp_thinking_budget_to_answer_room(8_192, 512), 0);
}

#[test]
fn adjust_max_tokens_fits_the_budget_under_the_cap() {
    // No explicit caller cap: the model cap stands and the budget fits.
    let (max_tokens, budget) =
        adjust_max_tokens_for_thinking(None, 20_000, ThinkingLevel::Medium, None);
    assert_eq!(max_tokens, 20_000);
    assert_eq!(
        budget,
        default_thinking_budgets()
            .medium
            .expect("the medium default")
    );

    // A caller cap adds the budget, clamped to the model cap.
    let (max_tokens, budget) =
        adjust_max_tokens_for_thinking(Some(1_000), 20_000, ThinkingLevel::Medium, None);
    assert_eq!(max_tokens, (1_000 + budget).min(20_000));

    // A cap below the budget clamps the budget to the answer room.
    let (max_tokens, budget) =
        adjust_max_tokens_for_thinking(Some(100), 2_000, ThinkingLevel::Medium, None);
    assert_eq!(max_tokens, 2_000);
    assert!(budget <= max_tokens - pi_ai::api::simple_options::MIN_ANSWER_TOKENS);
}

/// The metadata and session fields ride `build_base_options` verbatim.
#[test]
fn base_options_carry_the_request_plumbing() {
    let request = SimpleStreamOptions {
        transport_options: pi_ai::types::TransportOptions::default(),
        timeout_ms: Some(5_000),
        max_retries: Some(2),
        cache_retention: Some(pi_ai::types::CacheRetention::Long),
        session_id: Some("sess-1".to_owned()),
        metadata: Some(
            std::iter::once(("user_id".to_owned(), serde_json::json!("user_1"))).collect(),
        ),
        ..SimpleStreamOptions::default()
    };
    let options = build_base_options(&windowed_model(), &context(), Some(&request), None);
    assert_eq!(options.timeout_ms, Some(5_000));
    assert_eq!(options.max_retries, Some(2));
    assert_eq!(
        options.cache_retention,
        Some(pi_ai::types::CacheRetention::Long)
    );
    assert_eq!(options.session_id.as_deref(), Some("sess-1"));
    assert_eq!(
        options
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("user_id")),
        Some(&serde_json::json!("user_1")),
    );
}
