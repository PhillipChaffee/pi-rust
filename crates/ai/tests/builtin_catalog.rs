//! The builtin-catalog suites, ported 1:1 from the upstream `*-models.test.ts`
//! files whose subject is the generated catalog data (commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).
//!
//! The suites whose remaining tests stream through a wire API (baseten,
//! together, fireworks payloads, qwen sampling) land with their wire-API
//! tickets; the data-shape assertions port here. Files covered:
//! `model-catalog-types.test.ts`, `zai-coding-plan-models.test.ts`,
//! `xiaomi-models.test.ts`, `together-models.test.ts` (data parts),
//! `bedrock-models.test.ts`, `openrouter-cache-control-models.test.ts`,
//! `anthropic-adaptive-thinking-models.test.ts`, and the qwen allowlist
//! checks of `qwen-token-plan-models.test.ts`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "test failures panic by design, mirroring expect!'s failure mode"
)]

use std::collections::BTreeMap;

use pi_ai::env_api_keys::{find_env_keys, get_env_api_key};
use pi_ai::image_models::{get_image_model, get_image_models, get_image_providers};
use pi_ai::models::get_supported_thinking_levels;
use pi_ai::providers::all::{builtin_catalog_provider_ids, builtin_images_models, builtin_models};
use pi_ai::providers::catalog::{get_builtin_model, get_builtin_models};
use pi_ai::types::Model;

/// The model lookups upstream routes through `src/compat.ts`; the compat
/// layer lands with its own ticket, so the suite reads the builtin registry
/// directly.
fn get_model(provider: &str, id: &str) -> Option<Model> {
    get_builtin_model(provider, id)
}

fn get_models(provider: &str) -> Vec<Model> {
    get_builtin_models(provider)
}

#[test]
fn derives_model_api_id_and_provider_from_grouped_model_data() {
    let grok = get_model("xai", "grok-4.5").expect("grok-4.5 present");
    assert_eq!(
        grok.api.as_known(),
        Some(pi_ai::types::KnownApi::OpenaiResponses)
    );
    assert_eq!(grok.provider.0, "xai");
    for id in ["grok-4.5", "grok-4.6", "grok-4.3"] {
        let model = get_model("xai", id).unwrap_or_else(|| panic!("{id} present"));
        assert_eq!(
            model.api.as_known(),
            Some(pi_ai::types::KnownApi::OpenaiResponses)
        );
    }
}

#[test]
fn routes_github_copilot_grok_through_the_responses_api() {
    let model = get_model("github-copilot", "grok-4.5").expect("grok-4.5 present");
    assert_eq!(
        model.api.as_known(),
        Some(pi_ai::types::KnownApi::OpenaiResponses)
    );
}

#[test]
fn routes_all_github_copilot_gpt_models_through_the_responses_api() {
    let gpt_models: Vec<Model> = get_models("github-copilot")
        .into_iter()
        .filter(|model| model.id.starts_with("gpt-"))
        .collect();
    assert!(!gpt_models.is_empty());
    assert!(
        gpt_models
            .iter()
            .all(|model| model.api.as_known() == Some(pi_ai::types::KnownApi::OpenaiResponses)),
        "every Copilot GPT model uses the Responses API"
    );
    let gpt_6_astra = get_model("github-copilot", "gpt-6-astra").expect("gpt-6-astra present");
    assert_eq!(
        gpt_6_astra.api.as_known(),
        Some(pi_ai::types::KnownApi::OpenaiResponses)
    );
}

#[test]
fn exposes_the_glm_4_6v_on_the_china_coding_plan_catalog() {
    let model = get_builtin_model("zai-coding-cn", "glm-4.6v").expect("glm-4.6v present");
    assert_eq!(
        model.api.as_known(),
        Some(pi_ai::types::KnownApi::OpenaiCompletions)
    );
    assert_eq!(
        model.input,
        vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image]
    );
}

#[test]
fn keeps_zero_costs_for_coding_plan_models_without_a_matching_api_price() {
    // Coding Plan models without a matching API price carry zero rates, the
    // same degradation the upstream catalog records.
    let zero_cost = get_builtin_models("zai-coding-cn")
        .into_iter()
        .filter(|model| model.cost.rates.input == 0.0 && model.cost.rates.output == 0.0)
        .count();
    assert!(
        zero_cost > 0,
        "the catalog carries zero-cost Coding Plan models"
    );
}

#[test]
fn omits_deprecated_models_from_the_xiaomi_providers() {
    const DEPRECATED: [&str; 3] = ["mimo-v2-flash", "mimo-v2-omni", "mimo-v2-pro"];
    const REPLACEMENTS: [&str; 2] = ["mimo-v2.5", "mimo-v2.5-pro"];
    for provider in [
        "xiaomi",
        "xiaomi-token-plan-cn",
        "xiaomi-token-plan-ams",
        "xiaomi-token-plan-sgp",
    ] {
        let ids: Vec<String> = get_models(provider)
            .iter()
            .map(|model| model.id.clone())
            .collect();
        for deprecated in DEPRECATED {
            assert!(
                !ids.iter().any(|id| id == deprecated),
                "{provider} omits {deprecated}"
            );
        }
        for replacement in REPLACEMENTS {
            assert!(
                ids.iter().any(|id| id == replacement),
                "{provider} keeps {replacement}"
            );
        }
    }
}

#[test]
fn together_registers_the_default_kimi_via_chat_completions() {
    let model = get_model("together", "moonshotai/Kimi-K2.6").expect("Kimi K2.6 present");
    assert_eq!(
        model.api.as_known(),
        Some(pi_ai::types::KnownApi::OpenaiCompletions)
    );
    assert_eq!(model.provider.0, "together");
    assert_eq!(model.base_url, "https://api.together.ai/v1");
    assert!(model.reasoning);
    assert_eq!(
        model.thinking_level_map,
        Some(BTreeMap::from([
            (pi_ai::types::ModelThinkingLevel::Minimal, None),
            (pi_ai::types::ModelThinkingLevel::Low, None),
            (pi_ai::types::ModelThinkingLevel::Medium, None),
        ]))
    );
    assert_eq!(
        model.input,
        vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image]
    );
    assert_eq!(model.context_window, 262_144);
    assert_eq!(model.max_tokens, 131_000);
    assert_eq!(
        model.cost.rates,
        pi_ai::types::ModelCostRates {
            input: 1.2,
            output: 4.5,
            cache_read: 0.2,
            cache_write: 0.0,
        }
    );
    let compat = model.compat.expect("together compat");
    assert_eq!(compat.supports_store, Some(false));
    assert_eq!(compat.supports_developer_role, Some(false));
    assert_eq!(compat.supports_reasoning_effort, Some(false));
    assert_eq!(
        compat.max_tokens_field,
        Some(pi_ai::types::MaxTokensField::MaxTokens)
    );
    assert_eq!(
        compat.thinking_format,
        Some(pi_ai::types::ThinkingFormat::Together)
    );
    assert_eq!(compat.supports_strict_mode, Some(false));
    assert_eq!(compat.supports_long_cache_retention, Some(false));
}

#[test]
fn together_models_reasoning_controls_from_its_api_surface() {
    let gpt_oss = get_model("together", "openai/gpt-oss-120b").expect("gpt-oss present");
    let nulls =
        |levels: &[(pi_ai::types::ModelThinkingLevel, Option<&str>)]| -> BTreeMap<pi_ai::types::ModelThinkingLevel, Option<String>> {
            levels
                .iter()
                .map(|(level, mapped)| (*level, mapped.map(ToOwned::to_owned)))
                .collect()
        };
    assert_eq!(
        gpt_oss.thinking_level_map,
        Some(nulls(&[
            (pi_ai::types::ModelThinkingLevel::Off, None),
            (pi_ai::types::ModelThinkingLevel::Minimal, None),
            (pi_ai::types::ModelThinkingLevel::Low, Some("low")),
            (pi_ai::types::ModelThinkingLevel::Medium, Some("medium")),
            (pi_ai::types::ModelThinkingLevel::High, Some("high")),
            (pi_ai::types::ModelThinkingLevel::Max, None),
            (pi_ai::types::ModelThinkingLevel::Xhigh, None),
        ]))
    );
    let compat = gpt_oss.compat.expect("compat");
    assert_eq!(compat.supports_reasoning_effort, Some(true));
    assert_eq!(
        compat.thinking_format,
        Some(pi_ai::types::ThinkingFormat::Openai)
    );

    let deepseek = get_model("together", "deepseek-ai/DeepSeek-V4-Pro").expect("deepseek present");
    assert_eq!(
        deepseek.thinking_level_map,
        Some(nulls(&[
            (pi_ai::types::ModelThinkingLevel::Minimal, None),
            (pi_ai::types::ModelThinkingLevel::Low, None),
            (pi_ai::types::ModelThinkingLevel::Medium, None),
            (pi_ai::types::ModelThinkingLevel::High, Some("high")),
            (pi_ai::types::ModelThinkingLevel::Xhigh, None),
        ]))
    );
    let deepseek_compat = deepseek.compat.expect("compat");
    assert_eq!(deepseek_compat.supports_reasoning_effort, Some(true));
    assert_eq!(
        deepseek_compat.thinking_format,
        Some(pi_ai::types::ThinkingFormat::Together)
    );
}

#[test]
fn resolves_together_api_key_from_the_environment() {
    // The env overlay stands in for the process environment; upstream mutates
    // process.env, which the crate forbids touching under unsafe-code.
    let env: pi_ai::types::ProviderEnv = BTreeMap::from([(
        "TOGETHER_API_KEY".to_owned(),
        "test-together-key".to_owned(),
    )]);
    assert_eq!(
        find_env_keys("together", Some(&env)).map(|keys| keys.join(",")),
        Some("TOGETHER_API_KEY".to_owned())
    );
    assert_eq!(
        get_env_api_key("together", Some(&env)),
        Some("test-together-key".to_owned())
    );
}

#[test]
fn bedrock_exposes_all_available_models() {
    assert!(!get_models("amazon-bedrock").is_empty());
}

#[test]
fn bedrock_exposes_claude_opus_5_through_an_inference_profile_only() {
    let models = get_models("amazon-bedrock");
    assert!(
        models
            .iter()
            .any(|model| model.id == "global.anthropic.claude-opus-5")
    );
    assert!(
        !models
            .iter()
            .any(|model| model.id == "anthropic.claude-opus-5")
    );
}

#[test]
fn keeps_openrouter_anthropic_latest_aliases_on_completions_cache_control() {
    for model_id in [
        "~anthropic/claude-fable-latest",
        "~anthropic/claude-haiku-latest",
        "~anthropic/claude-opus-latest",
        "~anthropic/claude-sonnet-latest",
    ] {
        let model =
            get_model("openrouter", model_id).unwrap_or_else(|| panic!("{model_id} present"));
        assert_eq!(
            model.api.as_known(),
            Some(pi_ai::types::KnownApi::OpenaiCompletions)
        );
        let compat = model.compat.expect("cache-control compat");
        assert_eq!(
            compat.cache_control_format,
            Some(pi_ai::types::CacheControlFormat::Anthropic)
        );
    }
}

#[test]
fn marks_builtin_anthropic_messages_models_that_use_adaptive_thinking() {
    const EXPECTED_CURRENT: [&str; 16] = [
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "cloudflare-ai-gateway/claude-fable-5",
        "fireworks/accounts/fireworks/models/deepseek-v4-flash-0731",
        "fireworks/accounts/fireworks/models/gpt-oss-120b",
        "fireworks/accounts/fireworks/models/qwen3p8-max",
        "kimi-coding/kimi-for-coding",
        "kimi-coding/k3",
        "kimi-coding/kimi-for-coding-highspeed",
        "opencode/claude-opus-4-8",
        "opencode/claude-opus-5",
        "vercel-ai-gateway/anthropic/claude-opus-4.8",
        "vercel-ai-gateway/anthropic/claude-opus-5",
        "vercel-ai-gateway/anthropic/claude-sonnet-5",
    ];
    let providers = builtin_catalog_provider_ids();
    let flagged: Vec<String> = providers
        .iter()
        .flat_map(|provider| get_builtin_models(provider))
        .filter(|model| model.api.as_known() == Some(pi_ai::types::KnownApi::AnthropicMessages))
        .filter(|model| {
            model
                .compat
                .as_ref()
                .is_some_and(|compat| compat.force_adaptive_thinking == Some(true))
        })
        .map(|model| format!("{}/{}", model.provider.0, model.id))
        .collect();

    for expected in EXPECTED_CURRENT {
        assert!(
            flagged.iter().any(|id| id == expected),
            "adaptive-thinking metadata missing for {expected}"
        );
    }
    // Regression for pi #9323: Fireworks uses catalog effort metadata and
    // verified fallbacks, not a fixed set of adaptive model names.
    for id in &flagged {
        assert!(
            id.starts_with("fireworks/")
                || id.contains("opus-4.6")
                || id.contains("opus-4-6")
                || id.contains("opus-4.7")
                || id.contains("opus-4-7")
                || id.contains("opus-4.8")
                || id.contains("opus-4-8")
                || id.contains("opus-5")
                || id.contains("sonnet-4-6")
                || id.contains("sonnet-4.6")
                || id.contains("sonnet-5")
                || id.contains("fable-5")
                || id.contains("kimi-for-coding")
                || id.starts_with("kimi-coding/"),
            "unexpected adaptive-thinking flag on {id}"
        );
    }
}

#[test]
fn qwen_token_plan_exposes_exactly_the_documented_individual_text_models() {
    const INDIVIDUAL_TEXT_MODELS: [&str; 9] = [
        "deepseek-v4-flash-0731",
        "deepseek-v4-pro",
        "deepseek-v4-pro-0813",
        "glm-5.2",
        "qwen3.6-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.8-flash",
        "qwen3.8-max",
    ];
    let ids: Vec<String> = get_models("qwen-token-plan-individual")
        .iter()
        .map(|model| model.id.clone())
        .collect();
    assert_eq!(ids, INDIVIDUAL_TEXT_MODELS);
}

#[test]
fn qwen_token_plan_reuses_the_international_token_plan_env_var() {
    assert_eq!(
        find_env_keys("qwen-token-plan-individual", None),
        find_env_keys("qwen-token-plan", None)
    );
}

#[test]
fn image_registry_exposes_the_openrouter_catalog() {
    assert_eq!(get_image_providers(), ["openrouter"]);
    let models = get_image_models("openrouter");
    assert!(
        models
            .iter()
            .all(|model| model.api.0 == "openrouter-images")
    );
    let first = get_image_model("openrouter", &models[0].id).expect("lookup");
    assert_eq!(first.id, models[0].id);
}

#[test]
fn builtin_registry_matches_the_generated_catalog() {
    // The registry covers every catalog provider plus the purely dynamic
    // ones (radius has no static catalog entry), upstream's
    // `BuiltinProvider` comment.
    let catalog_ids: std::collections::BTreeSet<String> =
        pi_ai::providers::catalog::shard_provider_ids()
            .into_iter()
            .collect();
    let registry_ids: std::collections::BTreeSet<String> =
        pi_ai::providers::all::builtin_providers()
            .iter()
            .map(|provider| provider.id().to_owned())
            .collect();
    assert!(registry_ids.contains("radius"));
    assert_eq!(
        catalog_ids,
        registry_ids
            .difference(&std::collections::BTreeSet::from(["radius".to_owned()]))
            .cloned()
            .collect()
    );
    assert!(registry_ids.contains("anthropic"));
    assert!(registry_ids.contains("openrouter"));
}

#[test]
fn builtin_models_registers_every_provider_with_its_catalog() {
    let models = builtin_models(None);
    let providers = models.providers();
    assert_eq!(providers.len(), 40);
    for provider in providers {
        let listed = models.models(Some(provider.id()));
        let catalog = get_builtin_models(provider.id());
        // Radius is purely dynamic: its catalog is empty before a refresh.
        if provider.id() == "radius" {
            assert!(listed.is_empty(), "radius starts with no models");
            continue;
        }
        assert_eq!(listed.len(), catalog.len(), "provider {}", provider.id());
    }
}

#[test]
fn builtin_images_models_registers_the_openrouter_provider_with_its_catalog() {
    let models = builtin_images_models(Some(pi_ai::images_models::CreateImagesModelsOptions {
        auth_context: Some(Arc::new(FakeAuthContext(BTreeMap::from([(
            "OPENROUTER_API_KEY".to_owned(),
            "or-key".to_owned(),
        )])))),
        ..pi_ai::images_models::CreateImagesModelsOptions::default()
    }));
    let providers = models.providers();
    assert_eq!(
        providers
            .iter()
            .map(|provider| provider.id().to_owned())
            .collect::<Vec<_>>(),
        ["openrouter"]
    );

    let list = models.models(Some("openrouter"));
    assert!(!list.is_empty());
    assert!(list.iter().all(|model| model.api.0 == "openrouter-images"));
    let auth = futures_lite_block(models.get_auth_for_model(&list[0], None));
    assert_eq!(
        auth.expect("auth").and_then(|result| result.auth.api_key),
        Some("or-key".to_owned())
    );
}

fn futures_lite_block<F: ::std::future::Future>(future: F) -> F::Output {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    runtime.block_on(future)
}

use std::sync::Arc;

/// The fixture auth context the images test drives, upstream's
/// `fakeAuthContext`.
struct FakeAuthContext(BTreeMap<String, String>);

impl pi_ai::auth::types::AuthContext for FakeAuthContext {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

#[test]
fn image_model_registry_serves_every_provider() {
    for provider in get_image_providers() {
        let models = get_image_models(&provider);
        assert!(!models.is_empty(), "{provider} carries image models");
    }
    // The registry lookup round-trips a real entry.
    let first = get_image_models("openrouter")
        .first()
        .expect("entry")
        .clone();
    assert_eq!(
        get_image_model("openrouter", &first.id).map(|model| model.id),
        Some(first.id)
    );
}

#[test]
fn get_supported_thinking_levels_pins_the_off_default() {
    let mut model = get_model("baseten", "moonshotai/Kimi-K2.6").expect("Kimi K2.6 present");
    assert_eq!(
        get_supported_thinking_levels(&model),
        vec![
            pi_ai::types::ModelThinkingLevel::Off,
            pi_ai::types::ModelThinkingLevel::High
        ]
    );
    // A non-reasoning model supports only the off level.
    model.reasoning = false;
    assert_eq!(
        get_supported_thinking_levels(&model),
        vec![pi_ai::types::ModelThinkingLevel::Off]
    );
}
