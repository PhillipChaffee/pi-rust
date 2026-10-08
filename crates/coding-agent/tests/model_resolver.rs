//! Upstream `packages/coding-agent/test/model-resolver.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::model_resolver` (#121).
//!
//! Porting restatements this suite records:
//!
//! - The structural registry mocks (`{ getModels, getAvailable, getModel,
//!   hasConfiguredAuth, getAvailableSnapshot }`) port as `FakeRuntime`, a
//!   `ModelRuntimeView` implementation whose reads are optional; an unset
//!   read returns the empty list or `false` the runtime would.
//! - `parseModelPattern(pattern, allModels)` ports with an explicit `None`
//!   options argument; upstream's options object is optional.
//! - "returns scoped models and structured diagnostics without writing
//!   console warnings": the `console.warn` spy drops — the diagnostics
//!   surface is print-free by construction, so the port pins the
//!   structured-diagnostics equality alone.
//! - "resolveModelScope preserves CLI warning output": the `console.warn`
//!   spy drops, because the Rust surface prints through `eprintln!`, which
//!   tests cannot spy on; the port pins the same string through
//!   `format_scope_warning` and asserts the scoped list is empty.
//! - "built-in defaults exist in generated provider catalogs" reads
//!   `pi_ai::providers::all::{builtin_catalog_provider_ids,
//!   builtin_models_of}`, the port of `getBuiltinProviders`/`getBuiltinModels`.
//! - The "persisted default model scoping" describe is deferred to #125:
//!   it wires AgentSession/SessionManager/SettingsManager, not the resolver
//!   surface this suite pins.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use common::model_layer::model;
use pi_agent_core::types::ThinkingLevel;
use pi_ai::auth::resolve::ModelsFailure;
use pi_ai::auth::types::AuthOptions;
use pi_ai::providers::all::{builtin_catalog_provider_ids, builtin_models_of};
use pi_ai::types::{Api, BoxedFuture, Modality, Model, ModelCost, ModelCostRates, ProviderId};
use pi_coding_agent::model_resolver::{
    FindInitialModelOptions, ModelRuntimeView, ModelScopeDiagnostic, ModelScopeDiagnosticCode,
    ResolveCliModelOptions, ResolveCliModelResult, ResolveModelScopeResult,
    default_model_per_provider, find_initial_model, format_scope_warning, parse_model_pattern,
    resolve_cli_model, resolve_model_scope, resolve_model_scope_with_diagnostics,
};

/// The cost object upstream's `cost: { input, output, cacheRead, cacheWrite }`
/// spells.
const fn rates(input: f64, output: f64, cache_read: f64, cache_write: f64) -> ModelCost {
    ModelCost {
        rates: ModelCostRates {
            input,
            output,
            cache_read,
            cache_write,
        },
        tiers: None,
    }
}

/// The one-off mock model a case spells, upstream's full `Model` object
/// literal, built over the shared model-layer fixture.
#[expect(
    clippy::too_many_arguments,
    reason = "the object-literal restatement carries every field upstream sets"
)]
fn mock_model(
    provider: &str,
    id: &str,
    name: &str,
    reasoning: bool,
    input: &[Modality],
    cost: ModelCost,
    context_window: u64,
    max_tokens: u64,
    base_url: &str,
) -> Model {
    let mut fixture = model(provider, id);
    fixture.name = String::from(name);
    fixture.api = Api::from("anthropic-messages");
    fixture.base_url = String::from(base_url);
    fixture.reasoning = reasoning;
    fixture.input = input.to_vec();
    fixture.cost = cost;
    fixture.context_window = context_window;
    fixture.max_tokens = max_tokens;
    fixture
}

/// Upstream's `mockModels[0]`.
fn sonnet() -> Model {
    mock_model(
        "anthropic",
        "claude-sonnet-4-5",
        "Claude Sonnet 4.5",
        true,
        &[Modality::Text, Modality::Image],
        rates(3.0, 15.0, 0.3, 3.75),
        200_000,
        8192,
        "https://api.anthropic.com",
    )
}

/// Upstream's `mockModels[1]` ("using same type for simplicity").
fn gpt_4o() -> Model {
    mock_model(
        "openai",
        "gpt-4o",
        "GPT-4o",
        false,
        &[Modality::Text, Modality::Image],
        rates(5.0, 15.0, 0.5, 5.0),
        128_000,
        4096,
        "https://api.openai.com",
    )
}

/// Upstream's `mockOpenRouterModels[0]`, the colon-bearing id.
fn qwen_coder_exacto() -> Model {
    mock_model(
        "openrouter",
        "qwen/qwen3-coder:exacto",
        "Qwen3 Coder Exacto",
        true,
        &[Modality::Text],
        rates(1.0, 2.0, 0.1, 1.0),
        128_000,
        8192,
        "https://openrouter.ai/api/v1",
    )
}

/// Upstream's `mockOpenRouterModels[1]`, the id that starts with a provider
/// name.
fn gpt_4o_extended() -> Model {
    mock_model(
        "openrouter",
        "openai/gpt-4o:extended",
        "GPT-4o Extended",
        false,
        &[Modality::Text, Modality::Image],
        rates(5.0, 15.0, 0.5, 5.0),
        128_000,
        4096,
        "https://openrouter.ai/api/v1",
    )
}

/// Upstream's `allModels`: the two base mocks plus the OpenRouter colon-id
/// pair.
fn all_models() -> Vec<Model> {
    vec![sonnet(), gpt_4o(), qwen_coder_exacto(), gpt_4o_extended()]
}

/// Upstream's `bracketedModel`, the id whose brackets a glob would eat.
fn bracketed_model() -> Model {
    mock_model(
        "custom",
        "bracketed-model[1m]",
        "Bracketed Model",
        true,
        &[Modality::Text],
        rates(1.0, 2.0, 0.1, 1.0),
        128_000,
        8192,
        "https://example.invalid",
    )
}

/// Upstream's `azureModel`: the `gpt-4o` mock with the sol id, name, and
/// provider swapped in.
fn azure_sol() -> Model {
    let mut azure = gpt_4o();
    azure.id = String::from("gpt-5.6-sol");
    azure.name = String::from("GPT 5.6 Sol");
    azure.provider = ProviderId(String::from("azure-openai-responses"));
    azure
}

/// Upstream's `codexModel`: the `gpt-4o` mock with the sol id, name, and
/// provider swapped in.
fn codex_sol() -> Model {
    let mut codex = gpt_4o();
    codex.id = String::from("gpt-5.6-sol");
    codex.name = String::from("GPT 5.6 Sol");
    codex.provider = ProviderId(String::from("openai-codex"));
    codex
}

/// Upstream's `zaiModel`, the direct `zai` provider entry for `glm-5`.
fn zai_glm5() -> Model {
    mock_model(
        "zai",
        "glm-5",
        "GLM-5",
        true,
        &[Modality::Text],
        rates(1.0, 2.0, 0.1, 1.0),
        128_000,
        8192,
        "https://open.bigmodel.cn/api/paas/v4",
    )
}

/// Upstream's `gatewayModel`, the gateway entry whose id is `zai/glm-5`.
fn gateway_zai_glm5() -> Model {
    mock_model(
        "vercel-ai-gateway",
        "zai/glm-5",
        "GLM-5",
        true,
        &[Modality::Text],
        rates(1.0, 2.0, 0.1, 1.0),
        128_000,
        8192,
        "https://ai-gateway.vercel.sh",
    )
}

/// Upstream's `commandcodeModel`, whose id starts with a known provider name.
fn commandcode_mimo() -> Model {
    mock_model(
        "commandcode",
        "xiaomi/mimo-v2.5-pro",
        "Xiaomi MiMo via Commandcode",
        false,
        &[Modality::Text],
        rates(1.0, 2.0, 0.1, 1.0),
        128_000,
        8192,
        "https://example.invalid",
    )
}

/// Upstream's `xiaomiModel`, the direct `xiaomi` provider entry.
fn xiaomi_mimo() -> Model {
    mock_model(
        "xiaomi",
        "mimo-v2.5-pro",
        "Xiaomi MiMo",
        false,
        &[Modality::Text],
        rates(1.0, 2.0, 0.1, 1.0),
        128_000,
        8192,
        "https://api.xiaomimimo.com",
    )
}

/// Upstream's `savedDeepSeekModel`, the saved default the settings carry.
fn saved_deepseek() -> Model {
    mock_model(
        "deepseek",
        "deepseek-v4-flash",
        "DeepSeek V4 Flash",
        true,
        &[Modality::Text],
        rates(1.0, 2.0, 0.1, 1.0),
        128_000,
        8192,
        "https://api.deepseek.com",
    )
}

/// Upstream's `localDeepSeekModel`: the saved entry served by the
/// authenticated local provider instead.
fn local_deepseek() -> Model {
    let mut local = saved_deepseek();
    local.provider = ProviderId(String::from("spark-two"));
    local.base_url = String::from("http://spark-two:8000/v1");
    local
}

/// Upstream's `neuralwattModel`, the registered base model whose provider
/// lacks the requested id, triggering the fallback path.
fn neuralwatt_base() -> Model {
    mock_model(
        "neuralwatt",
        "some-base-model",
        "Some Base Model",
        false,
        &[Modality::Text],
        rates(1.0, 2.0, 0.1, 1.0),
        128_000,
        8192,
        "https://api.neuralwatt.com",
    )
}

/// Upstream's `aiGatewayModel`, the sole available gateway entry.
fn ai_gateway_opus() -> Model {
    mock_model(
        "vercel-ai-gateway",
        "anthropic/claude-opus-4-6",
        "Claude Opus 4.6",
        true,
        &[Modality::Text, Modality::Image],
        rates(5.0, 15.0, 0.5, 5.0),
        200_000,
        8192,
        "https://ai-gateway.vercel.sh",
    )
}

/// Upstream's `modelsWithNeuralwatt`, the fallback block's registry list.
fn models_with_neuralwatt() -> Vec<Model> {
    let mut models = all_models();
    models.push(neuralwatt_base());
    models
}

/// The valid thinking-level strings and the variants the assertions expect,
/// upstream's `["off", "minimal", ...]` iteration list.
fn all_thinking_levels() -> Vec<(&'static str, ThinkingLevel)> {
    vec![
        ("off", ThinkingLevel::Off),
        ("minimal", ThinkingLevel::Minimal),
        ("low", ThinkingLevel::Low),
        ("medium", ThinkingLevel::Medium),
        ("high", ThinkingLevel::High),
        ("xhigh", ThinkingLevel::Xhigh),
        ("max", ThinkingLevel::Max),
    ]
}

/// The pattern list a scope resolution drives, upstream's string-array
/// arguments.
fn patterns(patterns: &[&str]) -> Vec<String> {
    patterns.iter().copied().map(String::from).collect()
}

/// The invalid-thinking-suffix shape the warning cases pin: the pattern
/// resolves its prefix model with no thinking level, and carries the
/// warning the case asserts on.
fn invalid_suffix_result(pattern: &str) -> (Model, String) {
    let result = parse_model_pattern(pattern, &all_models(), None);
    let resolved = result.model.expect("the pattern resolves");
    assert!(result.thinking_level.is_none());
    let warning = result.warning.expect("the invalid suffix warns");
    (resolved, warning)
}

/// The scoped models' ids, the projection the diagnostics cases pin.
fn scoped_model_ids(result: &ResolveModelScopeResult) -> Vec<&str> {
    result
        .scoped_models
        .iter()
        .map(|scoped| scoped.model.id.as_str())
        .collect()
}

/// The CLI resolution over the all-models registry, upstream's
/// `resolveCliModel` with no thinking flag: the error-free result the
/// cases assert on.
fn resolved_cli(cli_provider: Option<&str>, cli_model: &str) -> ResolveCliModelResult {
    let runtime = FakeRuntime::models(all_models());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider,
        cli_model: Some(cli_model),
        cli_thinking: None,
        model_runtime: &runtime,
    });
    assert!(result.error.is_none());
    result
}

/// The fallback-path CLI resolution over the neuralwatt registry,
/// upstream's resolveCliModel calls: the error-free resolution with the
/// resolved provider and thinking level the cases assert on.
fn resolved_fallback(
    cli_provider: Option<&str>,
    cli_model: &str,
    cli_thinking: Option<ThinkingLevel>,
) -> (Model, Option<ThinkingLevel>) {
    let runtime = FakeRuntime::models(models_with_neuralwatt());
    let result = resolve_cli_model(ResolveCliModelOptions {
        cli_provider,
        cli_model: Some(cli_model),
        cli_thinking,
        model_runtime: &runtime,
    });
    assert!(result.error.is_none());
    let resolved = result.model.expect("the fallback model resolves");
    assert_eq!(resolved.provider.0, "neuralwatt");
    (resolved, result.thinking_level)
}

/// The `getModel` read's signature, upstream's `getModel(provider, modelId)`
/// closure.
type ModelLookup = Box<dyn Fn(&str, &str) -> Option<Model> + Send + Sync>;

/// The `hasConfiguredAuth` read's signature, upstream's
/// `hasConfiguredAuth(provider)` closure.
type AuthCheck = Box<dyn Fn(&str) -> bool + Send + Sync>;

/// The structural registry fake, upstream's
/// `{ getModels, getAvailable, getModel, hasConfiguredAuth,
/// getAvailableSnapshot }` object literals: each read is optional, and an
/// unset read returns the empty list or `false` the runtime would.
#[derive(Default)]
struct FakeRuntime {
    models: Option<Vec<Model>>,
    get_model: Option<ModelLookup>,
    auth: Option<AuthCheck>,
    available: Option<Vec<Model>>,
    available_snapshot: Option<Vec<Model>>,
}

impl FakeRuntime {
    /// The runtime whose `getModels` returns `models`, upstream's
    /// `{ getModels: () => models }`.
    fn models(models: Vec<Model>) -> Self {
        Self {
            models: Some(models),
            ..Self::default()
        }
    }

    /// The runtime whose `getAvailable` returns `models`, upstream's
    /// `{ getAvailable: () => models }`.
    fn available(models: Vec<Model>) -> Self {
        Self {
            available: Some(models),
            ..Self::default()
        }
    }

    /// The runtime whose `getAvailableSnapshot` returns `models`, upstream's
    /// `{ getAvailableSnapshot: () => models }`.
    fn available_snapshot(models: Vec<Model>) -> Self {
        Self {
            available_snapshot: Some(models),
            ..Self::default()
        }
    }

    /// Sets the `hasConfiguredAuth` read, upstream's inline
    /// `hasConfiguredAuth: (provider) => ...` closure.
    fn auth_for(self, auth: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        Self {
            auth: Some(Box::new(auth)),
            ..self
        }
    }

    /// Sets the `getModel` read, upstream's inline
    /// `getModel: (provider, modelId) => ...` closure.
    fn model_lookup(
        self,
        get_model: impl Fn(&str, &str) -> Option<Model> + Send + Sync + 'static,
    ) -> Self {
        Self {
            get_model: Some(Box::new(get_model)),
            ..self
        }
    }
}

impl ModelRuntimeView for FakeRuntime {
    fn get_models(&self) -> Vec<Model> {
        self.models.clone().unwrap_or_default()
    }

    fn get_model(&self, provider: &str, model_id: &str) -> Option<Model> {
        self.get_model
            .as_ref()
            .and_then(|get_model| get_model(provider, model_id))
    }

    fn has_configured_auth(&self, provider: &str) -> bool {
        self.auth.as_ref().is_some_and(|auth| auth(provider))
    }

    fn get_available_snapshot(&self) -> Vec<Model> {
        self.available_snapshot.clone().unwrap_or_default()
    }

    fn get_available(
        &self,
        _options: Option<&AuthOptions>,
    ) -> BoxedFuture<'_, Result<Vec<Model>, ModelsFailure>> {
        Box::pin(std::future::ready(Ok(self
            .available
            .clone()
            .unwrap_or_default())))
    }
}

/// Upstream's `parseModelPattern` describe block.
mod parse_model_pattern {
    use super::*;

    /// Upstream's "simple patterns without colons" block.
    mod simple_patterns_without_colons {
        use super::*;

        #[test]
        fn exact_match_returns_model_with_undefined_thinking_level() {
            let result = parse_model_pattern("claude-sonnet-4-5", &all_models(), None);
            let resolved = result.model.expect("the pattern resolves exactly");
            assert_eq!(resolved.id, "claude-sonnet-4-5");
            assert!(result.thinking_level.is_none());
            assert!(result.warning.is_none());
        }

        #[test]
        fn partial_match_returns_best_model_with_undefined_thinking_level() {
            let result = parse_model_pattern("sonnet", &all_models(), None);
            let resolved = result.model.expect("the partial pattern resolves");
            assert_eq!(resolved.id, "claude-sonnet-4-5");
            assert!(result.thinking_level.is_none());
            assert!(result.warning.is_none());
        }

        #[test]
        fn no_match_returns_undefined_model_and_thinking_level() {
            let result = parse_model_pattern("nonexistent", &all_models(), None);
            assert!(result.model.is_none());
            assert!(result.thinking_level.is_none());
            assert!(result.warning.is_none());
        }
    }

    /// Upstream's "patterns with valid thinking levels" block.
    mod patterns_with_valid_thinking_levels {
        use super::*;

        #[test]
        fn sonnet_high_returns_sonnet_with_high_thinking_level() {
            let result = parse_model_pattern("sonnet:high", &all_models(), None);
            let resolved = result.model.expect("the pattern resolves");
            assert_eq!(resolved.id, "claude-sonnet-4-5");
            assert_eq!(result.thinking_level, Some(ThinkingLevel::High));
            assert!(result.warning.is_none());
        }

        #[test]
        fn gpt_4o_medium_returns_gpt_4o_with_medium_thinking_level() {
            let result = parse_model_pattern("gpt-4o:medium", &all_models(), None);
            let resolved = result.model.expect("the pattern resolves");
            assert_eq!(resolved.id, "gpt-4o");
            assert_eq!(result.thinking_level, Some(ThinkingLevel::Medium));
            assert!(result.warning.is_none());
        }

        #[test]
        fn all_valid_thinking_levels_work() {
            for (level, expected) in all_thinking_levels() {
                let result = parse_model_pattern(&format!("sonnet:{level}"), &all_models(), None);
                let resolved = result.model.expect("the pattern resolves");
                assert_eq!(resolved.id, "claude-sonnet-4-5");
                assert_eq!(result.thinking_level, Some(expected));
                assert!(result.warning.is_none());
            }
        }
    }

    /// Upstream's "patterns with invalid thinking levels" block.
    mod patterns_with_invalid_thinking_levels {
        use super::*;

        #[test]
        fn sonnet_random_returns_sonnet_with_undefined_thinking_level_and_warning() {
            let (resolved, warning) = invalid_suffix_result("sonnet:random");
            assert_eq!(resolved.id, "claude-sonnet-4-5");
            assert!(warning.contains("Invalid thinking level"));
            assert!(warning.contains("random"));
        }

        #[test]
        fn gpt_4o_invalid_returns_gpt_4o_with_undefined_thinking_level_and_warning() {
            let (resolved, warning) = invalid_suffix_result("gpt-4o:invalid");
            assert_eq!(resolved.id, "gpt-4o");
            assert!(warning.contains("Invalid thinking level"));
        }
    }

    /// Upstream's "OpenRouter models with colons in IDs" block.
    mod openrouter_models_with_colons_in_ids {
        use super::*;

        #[test]
        fn qwen3_coder_exacto_matches_the_model_with_undefined_thinking_level() {
            let result = parse_model_pattern("qwen/qwen3-coder:exacto", &all_models(), None);
            let resolved = result.model.expect("the pattern resolves");
            assert_eq!(resolved.id, "qwen/qwen3-coder:exacto");
            assert!(result.thinking_level.is_none());
            assert!(result.warning.is_none());
        }

        #[test]
        fn openrouter_qwen_qwen3_coder_exacto_matches_with_provider_prefix() {
            let result =
                parse_model_pattern("openrouter/qwen/qwen3-coder:exacto", &all_models(), None);
            let resolved = result.model.expect("the pattern resolves");
            assert_eq!(resolved.id, "qwen/qwen3-coder:exacto");
            assert_eq!(resolved.provider.0, "openrouter");
            assert!(result.thinking_level.is_none());
            assert!(result.warning.is_none());
        }

        #[test]
        fn qwen3_coder_exacto_high_matches_model_with_high_thinking_level() {
            let result = parse_model_pattern("qwen/qwen3-coder:exacto:high", &all_models(), None);
            let resolved = result.model.expect("the pattern resolves");
            assert_eq!(resolved.id, "qwen/qwen3-coder:exacto");
            assert_eq!(result.thinking_level, Some(ThinkingLevel::High));
            assert!(result.warning.is_none());
        }

        #[test]
        fn openrouter_qwen_qwen3_coder_exacto_high_matches_with_provider_and_thinking_level() {
            let result = parse_model_pattern(
                "openrouter/qwen/qwen3-coder:exacto:high",
                &all_models(),
                None,
            );
            let resolved = result.model.expect("the pattern resolves");
            assert_eq!(resolved.id, "qwen/qwen3-coder:exacto");
            assert_eq!(resolved.provider.0, "openrouter");
            assert_eq!(result.thinking_level, Some(ThinkingLevel::High));
            assert!(result.warning.is_none());
        }

        #[test]
        fn gpt_4o_extended_matches_the_extended_model_with_undefined_thinking_level() {
            let result = parse_model_pattern("openai/gpt-4o:extended", &all_models(), None);
            let resolved = result.model.expect("the pattern resolves");
            assert_eq!(resolved.id, "openai/gpt-4o:extended");
            assert!(result.thinking_level.is_none());
            assert!(result.warning.is_none());
        }
    }

    /// Upstream's "invalid thinking levels with OpenRouter models" block.
    mod invalid_thinking_levels_with_openrouter_models {
        use super::*;

        #[test]
        fn qwen3_coder_exacto_random_returns_model_with_undefined_thinking_level_and_warning() {
            let (resolved, warning) = invalid_suffix_result("qwen/qwen3-coder:exacto:random");
            assert_eq!(resolved.id, "qwen/qwen3-coder:exacto");
            assert!(warning.contains("Invalid thinking level"));
            assert!(warning.contains("random"));
        }

        #[test]
        fn qwen3_coder_exacto_high_random_returns_model_with_undefined_thinking_level_and_warning()
        {
            let (resolved, warning) = invalid_suffix_result("qwen/qwen3-coder:exacto:high:random");
            assert_eq!(resolved.id, "qwen/qwen3-coder:exacto");
            assert!(warning.contains("Invalid thinking level"));
            assert!(warning.contains("random"));
        }
    }

    /// Upstream's "edge cases" block.
    mod edge_cases {
        use super::*;

        #[test]
        fn empty_pattern_matches_via_partial_matching() {
            // Empty string is included in all model IDs, so partial matching
            // finds a match.
            let result = parse_model_pattern("", &all_models(), None);
            assert!(result.model.is_some());
            assert!(result.thinking_level.is_none());
        }

        #[test]
        fn pattern_ending_with_colon_treats_empty_suffix_as_invalid() {
            let result = parse_model_pattern("sonnet:", &all_models(), None);
            let resolved = result.model.expect("the prefix still matches");
            assert_eq!(resolved.id, "claude-sonnet-4-5");
            let warning = result.warning.expect("the empty suffix warns");
            assert!(warning.contains("Invalid thinking level"));
        }
    }
}

/// Upstream's `resolveModelScopeWithDiagnostics` describe block.
mod resolve_model_scope_with_diagnostics {
    use super::*;

    #[tokio::test]
    async fn returns_scoped_models_and_structured_diagnostics() {
        let runtime = FakeRuntime::available(all_models());
        let patterns = patterns(&["sonnet:high", "gpt-4o:invalid", "missing"]);

        let result = resolve_model_scope_with_diagnostics(&patterns, &runtime, None).await;

        assert_eq!(scoped_model_ids(&result), ["claude-sonnet-4-5", "gpt-4o"]);
        assert_eq!(
            result.scoped_models[0].thinking_level,
            Some(ThinkingLevel::High)
        );
        assert!(result.scoped_models[1].thinking_level.is_none());
        assert_eq!(
            result.diagnostics,
            vec![
                ModelScopeDiagnostic {
                    code: ModelScopeDiagnosticCode::InvalidThinkingLevel,
                    message: String::from(
                        "Invalid thinking level \"invalid\" in pattern \"gpt-4o:invalid\". Using default instead.",
                    ),
                    pattern: String::from("gpt-4o:invalid"),
                },
                ModelScopeDiagnostic {
                    code: ModelScopeDiagnosticCode::NoMatch,
                    message: String::from("No models match pattern \"missing\""),
                    pattern: String::from("missing"),
                },
            ],
        );
    }

    #[tokio::test]
    async fn resolve_model_scope_preserves_cli_warning_output() {
        let runtime = FakeRuntime::available(all_models());
        let patterns = patterns(&["missing"]);

        let scoped_models = resolve_model_scope(&patterns, &runtime, None).await;
        assert!(scoped_models.is_empty());

        // The Rust surface prints through `eprintln!`, which tests cannot
        // spy on; the same string is pinned through `format_scope_warning`
        // on the diagnostic the scope resolution records.
        let diagnostics = resolve_model_scope_with_diagnostics(&patterns, &runtime, None)
            .await
            .diagnostics;
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            format_scope_warning(&diagnostics[0]),
            "Warning: No models match pattern \"missing\"",
        );
    }

    #[tokio::test]
    async fn resolves_bracketed_model_ids_as_exact_references_before_glob_matching() {
        let mut models = all_models();
        models.push(bracketed_model());
        let runtime = FakeRuntime::available(models);
        let patterns = patterns(&["custom/bracketed-model[1m]"]);

        let result = resolve_model_scope_with_diagnostics(&patterns, &runtime, None).await;

        assert_eq!(scoped_model_ids(&result), ["bracketed-model[1m]"]);
        assert!(result.diagnostics.is_empty());
    }

    #[tokio::test]
    async fn resolves_bracketed_model_ids_with_thinking_levels_as_exact_references_before_glob_matching()
     {
        let mut models = all_models();
        models.push(bracketed_model());
        let runtime = FakeRuntime::available(models);
        let patterns = patterns(&["custom/bracketed-model[1m]:high"]);

        let result = resolve_model_scope_with_diagnostics(&patterns, &runtime, None).await;

        assert_eq!(scoped_model_ids(&result), ["bracketed-model[1m]"]);
        assert_eq!(
            result.scoped_models[0].thinking_level,
            Some(ThinkingLevel::High)
        );
        assert!(result.diagnostics.is_empty());
    }
}

/// Upstream's `resolveCliModel` describe block.
mod resolve_cli_model {
    use super::*;

    #[test]
    fn resolves_model_provider_id_without_provider() {
        let resolved = resolved_cli(None, "openai/gpt-4o")
            .model
            .expect("the model resolves");
        assert_eq!(resolved.provider.0, "openai");
        assert_eq!(resolved.id, "gpt-4o");
    }

    #[test]
    fn resolves_fuzzy_patterns_within_an_explicit_provider() {
        let resolved = resolved_cli(Some("openai"), "4o")
            .model
            .expect("the model resolves");
        assert_eq!(resolved.provider.0, "openai");
        assert_eq!(resolved.id, "gpt-4o");
    }

    #[test]
    fn supports_model_pattern_with_thinking_without_explicit_thinking() {
        let result = resolved_cli(None, "sonnet:high");
        let resolved = result.model.expect("the model resolves");
        assert_eq!(resolved.id, "claude-sonnet-4-5");
        assert_eq!(result.thinking_level, Some(ThinkingLevel::High));
    }

    #[test]
    fn prefers_exact_model_id_match_over_provider_inference() {
        let resolved = resolved_cli(None, "openai/gpt-4o:extended")
            .model
            .expect("the model resolves");
        assert_eq!(resolved.provider.0, "openrouter");
        assert_eq!(resolved.id, "openai/gpt-4o:extended");
    }

    #[test]
    fn does_not_strip_invalid_suffix_as_thinking_level_in_model() {
        let resolved = resolved_cli(Some("openai"), "gpt-4o:extended")
            .model
            .expect("the model resolves");
        assert_eq!(resolved.provider.0, "openai");
        assert_eq!(resolved.id, "gpt-4o:extended");
    }

    #[test]
    fn allows_custom_model_ids_for_explicit_providers_without_double_prefixing() {
        let resolved = resolved_cli(Some("openrouter"), "openrouter/openai/ghost-model")
            .model
            .expect("the model resolves");
        assert_eq!(resolved.provider.0, "openrouter");
        assert_eq!(resolved.id, "openai/ghost-model");
    }

    #[test]
    fn returns_a_clear_error_when_there_are_no_models() {
        let runtime = FakeRuntime::models(Vec::new());

        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: Some("openai"),
            cli_model: Some("gpt-4o"),
            cli_thinking: None,
            model_runtime: &runtime,
        });

        assert!(result.model.is_none());
        let error = result.error.expect("the empty registry errors");
        assert!(error.contains("No models available"));
    }

    #[test]
    fn prefers_the_sole_authenticated_provider_for_an_ambiguous_bare_exact_model_id() {
        let runtime = FakeRuntime::models(vec![azure_sol(), codex_sol()])
            .auth_for(|provider: &str| provider == "openai-codex");

        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("gpt-5.6-sol"),
            cli_thinking: None,
            model_runtime: &runtime,
        });

        assert!(result.error.is_none());
        let resolved = result.model.expect("the model resolves");
        assert_eq!(resolved.provider.0, "openai-codex");
        assert_eq!(resolved.id, "gpt-5.6-sol");
    }

    #[test]
    fn requires_an_explicit_provider_for_an_ambiguous_bare_exact_model_id() {
        let runtime =
            FakeRuntime::models(vec![azure_sol(), codex_sol()]).auth_for(|_provider: &str| false);

        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("gpt-5.6-sol"),
            cli_thinking: None,
            model_runtime: &runtime,
        });

        assert!(result.model.is_none());
        let error = result.error.expect("the ambiguous id errors");
        assert!(error.contains("Model \"gpt-5.6-sol\" is ambiguous across providers"));
        assert!(error.contains("azure-openai-responses/gpt-5.6-sol"));
        assert!(error.contains("openai-codex/gpt-5.6-sol"));
        assert!(error.contains("Use --provider or provider/model"));
    }

    #[test]
    fn prefers_provider_model_split_over_gateway_model_with_matching_id() {
        let mut models = all_models();
        models.push(zai_glm5());
        models.push(gateway_zai_glm5());
        let runtime = FakeRuntime::models(models).auth_for(|_provider: &str| true);

        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("zai/glm-5"),
            cli_thinking: None,
            model_runtime: &runtime,
        });

        assert!(result.error.is_none());
        let resolved = result.model.expect("the model resolves");
        assert_eq!(resolved.provider.0, "zai");
        assert_eq!(resolved.id, "glm-5");
    }

    #[test]
    fn prefers_an_authenticated_exact_raw_model_id_over_an_unauthenticated_inferred_provider() {
        let mut models = all_models();
        models.push(commandcode_mimo());
        models.push(xiaomi_mimo());
        let runtime =
            FakeRuntime::models(models).auth_for(|provider: &str| provider == "commandcode");

        let result = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("xiaomi/mimo-v2.5-pro"),
            cli_thinking: None,
            model_runtime: &runtime,
        });

        assert!(result.error.is_none());
        let resolved = result.model.expect("the model resolves");
        assert_eq!(resolved.provider.0, "commandcode");
        assert_eq!(resolved.id, "xiaomi/mimo-v2.5-pro");
    }

    #[test]
    fn resolves_provider_prefixed_fuzzy_patterns() {
        let resolved = resolved_cli(None, "openrouter/qwen")
            .model
            .expect("the model resolves");
        assert_eq!(resolved.provider.0, "openrouter");
        assert_eq!(resolved.id, "qwen/qwen3-coder:exacto");
    }

    /// Upstream's "custom model fallback with :thinking suffix (upstream
    /// #5552)" block.
    mod custom_model_fallback_with_thinking_suffix {
        use super::*;

        #[test]
        fn strips_thinking_suffix_from_custom_model_id_in_fallback_path() {
            let (resolved, thinking) =
                resolved_fallback(None, "neuralwatt/zai-org/GLM-5.1-FP8:high", None);
            // The :high suffix must NOT leak into the model id sent to the API.
            assert_eq!(resolved.id, "zai-org/GLM-5.1-FP8");
            assert!(resolved.reasoning);
            assert_eq!(thinking, Some(ThinkingLevel::High));
        }

        #[test]
        fn custom_model_without_thinking_suffix_works_normally_in_fallback_path() {
            let (resolved, thinking) =
                resolved_fallback(None, "neuralwatt/zai-org/GLM-5.1-FP8", None);
            assert_eq!(resolved.id, "zai-org/GLM-5.1-FP8");
            assert!(thinking.is_none());
        }

        #[test]
        fn all_valid_thinking_levels_work_in_fallback_path() {
            for (level, expected) in all_thinking_levels() {
                let (resolved, thinking) = resolved_fallback(
                    None,
                    &format!("neuralwatt/zai-org/GLM-5.1-FP8:{level}"),
                    None,
                );
                assert_eq!(resolved.id, "zai-org/GLM-5.1-FP8");
                assert_eq!(thinking, Some(expected));
            }
        }

        #[test]
        fn invalid_thinking_suffix_on_custom_model_is_treated_as_part_of_model_id() {
            let (resolved, thinking) =
                resolved_fallback(None, "neuralwatt/zai-org/GLM-5.1-FP8:banana", None);
            // Invalid suffix stays in the id (it's not a thinking level).
            assert_eq!(resolved.id, "zai-org/GLM-5.1-FP8:banana");
            assert!(thinking.is_none());
        }

        #[test]
        fn explicit_provider_with_custom_model_thinking_strips_suffix_correctly() {
            let (resolved, thinking) =
                resolved_fallback(Some("neuralwatt"), "zai-org/GLM-5.1-FP8:high", None);
            assert_eq!(resolved.id, "zai-org/GLM-5.1-FP8");
            assert_eq!(thinking, Some(ThinkingLevel::High));
        }

        #[test]
        fn with_explicit_thinking_the_suffix_is_kept_as_part_of_model_id() {
            let (resolved, thinking) = resolved_fallback(
                None,
                "neuralwatt/zai-org/GLM-5.1-FP8:high",
                Some(ThinkingLevel::Medium),
            );
            // :high is kept as part of the model id since --thinking was explicit.
            assert_eq!(resolved.id, "zai-org/GLM-5.1-FP8:high");
            assert!(thinking.is_none());
        }
    }
}

/// Upstream's "default model selection" describe block.
mod default_model_selection {
    use super::*;

    #[test]
    fn openai_defaults_track_current_models() {
        assert_eq!(default_model_per_provider("openai"), Some("gpt-5.5"));
        assert_eq!(default_model_per_provider("openai-codex"), Some("gpt-5.5"));
    }

    #[test]
    fn zai_minimax_cerebras_and_ant_ling_defaults_track_current_models() {
        assert_eq!(default_model_per_provider("zai"), Some("glm-5.3"));
        assert_eq!(default_model_per_provider("zai-coding-cn"), Some("glm-5.3"));
        assert_eq!(default_model_per_provider("minimax"), Some("MiniMax-M2.7"));
        assert_eq!(
            default_model_per_provider("minimax-cn"),
            Some("MiniMax-M2.7")
        );
        assert_eq!(default_model_per_provider("cerebras"), Some("gpt-oss-120b"));
        assert_eq!(default_model_per_provider("ant-ling"), Some("Ring-2.6-1T"));
    }

    #[test]
    fn builtin_defaults_exist_in_generated_provider_catalogs() {
        for provider in builtin_catalog_provider_ids() {
            let default_id = default_model_per_provider(&provider)
                .expect("every generated-catalog provider carries a default model id");
            assert!(
                builtin_models_of(&provider)
                    .iter()
                    .any(|model| model.id == default_id),
                "{provider} default {default_id} should exist in its generated catalog",
            );
        }
    }

    #[test]
    fn ai_gateway_default_tracks_current_model() {
        assert_eq!(
            default_model_per_provider("vercel-ai-gateway"),
            Some("zai/glm-5.1"),
        );
    }

    #[test]
    fn xai_default_tracks_current_model() {
        assert_eq!(default_model_per_provider("xai"), Some("grok-4.6"));
    }

    #[test]
    fn qwen_token_plan_individual_default_tracks_current_model() {
        assert_eq!(
            default_model_per_provider("qwen-token-plan-individual"),
            Some("qwen3.8-max"),
        );
    }

    #[test]
    fn find_initial_model_accepts_explicit_provider_custom_model_ids() {
        let runtime = FakeRuntime::models(all_models());

        let result = find_initial_model(FindInitialModelOptions {
            cli_provider: Some("openrouter"),
            cli_model: Some("openrouter/openai/ghost-model"),
            scoped_models: &[],
            is_continuing: false,
            default_provider: None,
            default_model_id: None,
            default_thinking_level: None,
            model_thinking_levels: None,
            model_runtime: &runtime,
        })
        .expect("the CLI resolution succeeds");

        let resolved = result.model.expect("the initial model is found");
        assert_eq!(resolved.provider.0, "openrouter");
        assert_eq!(resolved.id, "openai/ghost-model");
    }

    #[test]
    fn find_initial_model_selects_ai_gateway_default_when_available() {
        let runtime = FakeRuntime::available_snapshot(vec![ai_gateway_opus()]);

        let result = find_initial_model(FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: &[],
            is_continuing: false,
            default_provider: None,
            default_model_id: None,
            default_thinking_level: None,
            model_thinking_levels: None,
            model_runtime: &runtime,
        })
        .expect("the selection succeeds");

        let resolved = result.model.expect("the initial model is found");
        assert_eq!(resolved.provider.0, "vercel-ai-gateway");
        assert_eq!(resolved.id, "anthropic/claude-opus-4-6");
    }

    #[test]
    fn find_initial_model_ignores_an_unauthenticated_saved_default() {
        let saved = saved_deepseek();
        let runtime = FakeRuntime::available_snapshot(vec![local_deepseek()])
            .auth_for(|provider: &str| provider == "spark-two")
            .model_lookup(move |provider: &str, model_id: &str| {
                (provider == "deepseek" && model_id == "deepseek-v4-flash").then(|| saved.clone())
            });

        let result = find_initial_model(FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: &[],
            is_continuing: false,
            default_provider: Some("deepseek"),
            default_model_id: Some("deepseek-v4-flash"),
            default_thinking_level: None,
            model_thinking_levels: None,
            model_runtime: &runtime,
        })
        .expect("the selection succeeds");

        let resolved = result.model.expect("the fallback model is found");
        assert_eq!(resolved.provider.0, "spark-two");
        assert_eq!(resolved.id, "deepseek-v4-flash");
    }
}
