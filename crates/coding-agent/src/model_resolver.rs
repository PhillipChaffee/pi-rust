//! Model resolution, scoping, and initial selection, upstream's
//! `src/core/model-resolver.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: the functions read the runtime through
//! [`ModelRuntimeView`] — the four reads and one async availability query
//! the resolver needs — so tests can drive fakes the way upstream's
//! structural mocks did. `findInitialModel`'s CLI-error path ports as a
//! returned error instead of `process.exit(1)`; the caller owns the exit.
//! The `console.warn`/`console.error`/`console.log` prints ride stderr and
//! stdout directly; the warning's text is pinned through
//! [`format_scope_warning`].

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_agent_core::types::ThinkingLevel;
use pi_ai::auth::resolve::ModelsFailure;
use pi_ai::auth::types::AuthOptions;
use pi_ai::models::models_are_equal;
use pi_ai::types::{BoxedFuture, Model};

use crate::model_runtime::ModelRuntimeCore;

use crate::defaults::DEFAULT_THINKING_LEVEL;
use crate::model_runtime::ModelRuntime;
use crate::utils::minimatch;

/// The runtime reads the resolver's functions go through, the structural
/// surface upstream's test fakes stand in for: the sync model reads plus the
/// async availability query.
pub trait ModelRuntimeView: Send + Sync {
    /// Every composed model, upstream's `getModels()`.
    fn get_models(&self) -> Vec<Model>;
    /// One model by provider and id, upstream's `getModel(provider, id)`.
    fn get_model(&self, provider: &str, model_id: &str) -> Option<Model>;
    /// Whether the provider has configured auth, upstream's
    /// `hasConfiguredAuth(provider)`.
    fn has_configured_auth(&self, provider: &str) -> bool;
    /// The last-known available list, upstream's `getAvailableSnapshot()`.
    fn get_available_snapshot(&self) -> Vec<Model>;
    /// The available models, upstream's `getAvailable(options)`.
    fn get_available(
        &self,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'_, Result<Vec<Model>, ModelsFailure>>;
}

/// Default model IDs for each known provider, upstream's
/// `defaultModelPerProvider`.
#[must_use]
pub fn default_model_per_provider(provider: &str) -> Option<&'static str> {
    match provider {
        "amazon-bedrock" => Some("us.anthropic.claude-opus-4-6-v1"),
        "ant-ling" => Some("Ring-2.6-1T"),
        "anthropic" => Some("claude-opus-4-8"),
        "openai" | "openai-codex" => Some("gpt-5.5"),
        "azure-openai-responses" | "github-copilot" => Some("gpt-5.4"),
        "radius" => Some("balanced"),
        "nvidia" => Some("nvidia/nemotron-3-super-120b-a12b"),
        "deepseek" => Some("deepseek-v4-pro"),
        "google" | "google-vertex" => Some("gemini-3.1-pro-preview"),
        "openrouter" => Some("moonshotai/kimi-k2.6"),
        "vercel-ai-gateway" => Some("zai/glm-5.1"),
        "moonshotai" | "moonshotai-cn" | "opencode" | "opencode-go" => Some("kimi-k2.6"),
        "xai" => Some("grok-4.6"),
        "groq" => Some("openai/gpt-oss-120b"),
        "cerebras" => Some("gpt-oss-120b"),
        "zai" | "zai-coding-cn" => Some("glm-5.3"),
        "mistral" => Some("devstral-medium-latest"),
        "minimax" | "minimax-cn" => Some("MiniMax-M2.7"),
        "huggingface" | "together" => Some("moonshotai/Kimi-K2.6"),
        "fireworks" => Some("accounts/fireworks/models/kimi-k2p6"),
        "baseten" => Some("zai-org/GLM-5.2"),
        "kimi-coding" => Some("kimi-for-coding"),
        "cloudflare-workers-ai" => Some("@cf/moonshotai/kimi-k2.6"),
        "cloudflare-ai-gateway" => Some("workers-ai/@cf/moonshotai/kimi-k2.6"),
        "qwen-token-plan" | "qwen-token-plan-cn" => Some("qwen3.7-max"),
        "qwen-token-plan-individual" => Some("qwen3.8-max"),
        "xiaomi" | "xiaomi-token-plan-cn" | "xiaomi-token-plan-ams" | "xiaomi-token-plan-sgp" => {
            Some("mimo-v2.5-pro")
        }
        _ => None,
    }
}

/// Whether the string is one of the pi thinking levels, upstream's
/// `isValidThinkingLevel` from `cli/args.ts`.
#[must_use]
pub fn is_valid_thinking_level(level: &str) -> bool {
    matches!(
        level,
        "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
    )
}

fn thinking_level_from_str(level: &str) -> Option<ThinkingLevel> {
    match level {
        "off" => Some(ThinkingLevel::Off),
        "minimal" => Some(ThinkingLevel::Minimal),
        "low" => Some(ThinkingLevel::Low),
        "medium" => Some(ThinkingLevel::Medium),
        "high" => Some(ThinkingLevel::High),
        "xhigh" => Some(ThinkingLevel::Xhigh),
        "max" => Some(ThinkingLevel::Max),
        _ => None,
    }
}

/// A pattern-resolved model with its optional thinking level, upstream's
/// `ScopedModel`.
#[derive(Clone, Debug)]
pub struct ScopedModel {
    /// The resolved model.
    pub model: Model,
    /// The thinking level a `pattern:level` suffix carried.
    pub thinking_level: Option<ThinkingLevel>,
}

/// Whether a model id looks like an alias (no date suffix), upstream's
/// `isAlias`.
fn is_alias(id: &str) -> bool {
    if id.ends_with("-latest") {
        return true;
    }
    // Dates are typically in format: -20241022 or -20250929
    let bytes = id.as_bytes();
    !(bytes.len() >= 9
        && bytes[bytes.len() - 9] == b'-'
        && bytes[bytes.len() - 8..].iter().all(u8::is_ascii_digit))
}

/// Find an exact model reference match, upstream's
/// `findExactModelReferenceMatch`: a bare id or a canonical
/// `provider/modelId` reference, with ambiguous bare-id matches rejected.
#[must_use]
pub fn find_exact_model_reference_match(
    model_reference: &str,
    available_models: &[Model],
) -> Option<Model> {
    let trimmed_reference = model_reference.trim();
    if trimmed_reference.is_empty() {
        return None;
    }
    let normalized_reference = trimmed_reference.to_lowercase();

    let canonical_matches: Vec<&Model> = available_models
        .iter()
        .filter(|model| {
            format!("{}/{}", model.provider.0, model.id).to_lowercase() == normalized_reference
        })
        .collect();
    if canonical_matches.len() == 1 {
        return Some(canonical_matches[0].clone());
    }
    if canonical_matches.len() > 1 {
        return None;
    }

    if let Some((provider, model_id)) = trimmed_reference.split_once('/')
        && !provider.trim().is_empty()
        && !model_id.trim().is_empty()
    {
        let provider_matches: Vec<&Model> = available_models
            .iter()
            .filter(|model| {
                model.provider.0.to_lowercase() == provider.trim().to_lowercase()
                    && model.id.to_lowercase() == model_id.trim().to_lowercase()
            })
            .collect();
        if provider_matches.len() == 1 {
            return Some(provider_matches[0].clone());
        }
        if provider_matches.len() > 1 {
            return None;
        }
    }

    let id_matches: Vec<&Model> = available_models
        .iter()
        .filter(|model| model.id.to_lowercase() == normalized_reference)
        .collect();
    (id_matches.len() == 1).then(|| id_matches[0].clone())
}

/// Try to match a pattern to a model, upstream's `tryMatchModel`: exact
/// first, then partial id/name matching preferring aliases over dated
/// versions.
fn try_match_model(model_pattern: &str, available_models: &[Model]) -> Option<Model> {
    if let Some(exact_match) = find_exact_model_reference_match(model_pattern, available_models) {
        return Some(exact_match);
    }

    let lower_pattern = model_pattern.to_lowercase();
    let matches: Vec<Model> = available_models
        .iter()
        .filter(|model| {
            model.id.to_lowercase().contains(&lower_pattern)
                || model.name.to_lowercase().contains(&lower_pattern)
        })
        .cloned()
        .collect();
    if matches.is_empty() {
        return None;
    }

    // Separate into aliases and dated versions; prefer an alias, and among
    // several pick the one that sorts highest.
    let mut aliases: Vec<Model> = matches
        .iter()
        .filter(|model| is_alias(&model.id))
        .cloned()
        .collect();
    if aliases.is_empty() {
        let mut dated_versions: Vec<Model> = matches
            .iter()
            .filter(|model| !is_alias(&model.id))
            .cloned()
            .collect();
        dated_versions.sort_by(|a, b| b.id.cmp(&a.id));
        dated_versions.first().cloned()
    } else {
        aliases.sort_by(|a, b| b.id.cmp(&a.id));
        Some(aliases.remove(0))
    }
}

/// The result of parsing one pattern, upstream's `ParsedModelResult`.
#[derive(Clone, Debug, Default)]
pub struct ParsedModelResult {
    /// The matched model.
    pub model: Option<Model>,
    /// The thinking level a valid `:level` suffix carried.
    pub thinking_level: Option<ThinkingLevel>,
    /// The warning an invalid suffix produced.
    pub warning: Option<String>,
}

/// The fallback model a custom id builds from a provider's catalog,
/// upstream's `buildFallbackModel`.
fn build_fallback_model(
    provider: &str,
    model_id: &str,
    available_models: &[Model],
) -> Option<Model> {
    let provider_models: Vec<Model> = available_models
        .iter()
        .filter(|model| model.provider.0 == provider)
        .cloned()
        .collect();
    if provider_models.is_empty() {
        return None;
    }
    let default_id = default_model_per_provider(provider);
    let base_model = default_id
        .and_then(|default_id| provider_models.iter().find(|model| model.id == default_id))
        .unwrap_or(&provider_models[0]);
    let mut model = base_model.clone();
    model.id = String::from(model_id);
    model.name = String::from(model_id);
    Some(model)
}

/// Parse a pattern to extract model and thinking level, upstream's
/// `parseModelPattern`.
///
/// Match the full pattern first, then progressively strip colon suffixes; a
/// valid suffix carries the level, an invalid one warns (or fails in strict
/// mode).
#[must_use]
pub fn parse_model_pattern(
    pattern: &str,
    available_models: &[Model],
    options: Option<ParseModelPatternOptions>,
) -> ParsedModelResult {
    if let Some(exact_match) = try_match_model(pattern, available_models) {
        return ParsedModelResult {
            model: Some(exact_match),
            thinking_level: None,
            warning: None,
        };
    }

    let Some((prefix, suffix)) = pattern.rsplit_once(':') else {
        return ParsedModelResult::default();
    };

    if is_valid_thinking_level(suffix) {
        let result = parse_model_pattern(prefix, available_models, options);
        if result.model.is_some() {
            // Only use this thinking level if no warning from the inner
            // recursion.
            return ParsedModelResult {
                model: result.model,
                thinking_level: if result.warning.is_some() {
                    None
                } else {
                    thinking_level_from_str(suffix)
                },
                warning: result.warning,
            };
        }
        return result;
    }

    let allow_fallback =
        options.is_none_or(|options| options.allow_invalid_thinking_level_fallback);
    if !allow_fallback {
        // In strict mode (CLI --model parsing), treat it as part of the
        // model id and fail. This avoids accidentally resolving to a
        // different model.
        return ParsedModelResult::default();
    }

    // Scope mode: recurse on prefix and warn.
    let result = parse_model_pattern(prefix, available_models, options);
    if result.model.is_some() {
        ParsedModelResult {
            model: result.model,
            thinking_level: None,
            warning: Some(format!(
                "Invalid thinking level \"{suffix}\" in pattern \"{pattern}\". Using default instead."
            )),
        }
    } else {
        result
    }
}

/// The options [`parse_model_pattern`] takes, upstream's inline
/// `allowInvalidThinkingLevelFallback`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParseModelPatternOptions {
    /// Whether an invalid suffix warns instead of failing. Default: true.
    pub allow_invalid_thinking_level_fallback: bool,
}

/// A scope-resolution diagnostic, upstream's `ModelScopeDiagnostic`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelScopeDiagnostic {
    /// The diagnostic code.
    pub code: ModelScopeDiagnosticCode,
    /// The user-facing message.
    pub message: String,
    /// The pattern that produced it.
    pub pattern: String,
}

/// The diagnostic codes, upstream's `"no-match" | "invalid-thinking-level"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelScopeDiagnosticCode {
    /// No model matched the pattern.
    NoMatch,
    /// The pattern carried an invalid thinking-level suffix.
    InvalidThinkingLevel,
}

/// The scope resolution result, upstream's `ResolveModelScopeResult`.
#[derive(Clone, Debug, Default)]
pub struct ResolveModelScopeResult {
    /// The resolved models in pattern order, duplicates skipped.
    pub scoped_models: Vec<ScopedModel>,
    /// The warnings the patterns produced.
    pub diagnostics: Vec<ModelScopeDiagnostic>,
}

/// The user-facing warning line a diagnostic prints, upstream's
/// `console.warn(chalk.yellow(...))` text.
#[must_use]
pub fn format_scope_warning(diagnostic: &ModelScopeDiagnostic) -> String {
    format!("Warning: {}", diagnostic.message)
}

/// Resolve model patterns to actual models with optional thinking levels,
/// upstream's `resolveModelScopeFromModels`.
///
/// Glob-carrying patterns match against `provider/modelId` or the bare id,
/// after an exact-reference attempt that keeps bracketed ids from reading as
/// character classes.
#[must_use]
pub fn resolve_model_scope_from_models(
    patterns: &[String],
    models: &[Model],
) -> ResolveModelScopeResult {
    let available_models = models.to_vec();
    let mut scoped_models: Vec<ScopedModel> = Vec::new();
    let mut diagnostics: Vec<ModelScopeDiagnostic> = Vec::new();

    for pattern in patterns {
        if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
            // Extract optional thinking level suffix (e.g., "provider/*:high")
            let mut glob_pattern = pattern.as_str();
            let mut thinking_level: Option<ThinkingLevel> = None;
            if let Some((prefix, suffix)) = pattern.rsplit_once(':')
                && is_valid_thinking_level(suffix)
            {
                thinking_level = thinking_level_from_str(suffix);
                glob_pattern = prefix;
            }

            if let Some(exact_match) =
                find_exact_model_reference_match(glob_pattern, &available_models)
            {
                if !scoped_models
                    .iter()
                    .any(|scoped| models_are_equal(Some(&scoped.model), Some(&exact_match)))
                {
                    scoped_models.push(ScopedModel {
                        model: exact_match,
                        thinking_level,
                    });
                }
                continue;
            }

            // Match against "provider/modelId" format OR just model ID. This
            // allows "*sonnet*" to match without requiring
            // "anthropic/*sonnet*".
            let matching_models: Vec<Model> = available_models
                .iter()
                .filter(|model| {
                    let full_id = format!("{}/{}", model.provider.0, model.id);
                    minimatch::matches(&full_id, glob_pattern, true)
                        || minimatch::matches(&model.id, glob_pattern, true)
                })
                .cloned()
                .collect();

            if matching_models.is_empty() {
                diagnostics.push(ModelScopeDiagnostic {
                    code: ModelScopeDiagnosticCode::NoMatch,
                    message: format!("No models match pattern \"{pattern}\""),
                    pattern: pattern.clone(),
                });
                continue;
            }

            for model in matching_models {
                if !scoped_models
                    .iter()
                    .any(|scoped| models_are_equal(Some(&scoped.model), Some(&model)))
                {
                    scoped_models.push(ScopedModel {
                        model,
                        thinking_level,
                    });
                }
            }
            continue;
        }

        let parsed = parse_model_pattern(pattern, &available_models, None);

        if let Some(warning) = &parsed.warning {
            diagnostics.push(ModelScopeDiagnostic {
                code: ModelScopeDiagnosticCode::InvalidThinkingLevel,
                message: warning.clone(),
                pattern: pattern.clone(),
            });
        }

        let Some(model) = parsed.model else {
            diagnostics.push(ModelScopeDiagnostic {
                code: ModelScopeDiagnosticCode::NoMatch,
                message: format!("No models match pattern \"{pattern}\""),
                pattern: pattern.clone(),
            });
            continue;
        };

        // Avoid duplicates
        if !scoped_models
            .iter()
            .any(|scoped| models_are_equal(Some(&scoped.model), Some(&model)))
        {
            scoped_models.push(ScopedModel {
                model,
                thinking_level: parsed.thinking_level,
            });
        }
    }

    ResolveModelScopeResult {
        scoped_models,
        diagnostics,
    }
}

/// Resolve scoped models against a runtime's available list, upstream's
/// `resolveModelScopeWithDiagnostics`.
pub fn resolve_model_scope_with_diagnostics<'a, V: ModelRuntimeView>(
    patterns: &'a [String],
    model_runtime: &'a V,
    options: Option<&'a AuthOptions>,
) -> BoxedFuture<'a, ResolveModelScopeResult> {
    let available = model_runtime.get_available(options);
    Box::pin(async move {
        // The scope surface degrades to an empty resolution on an
        // availability failure; the runtime records the error.
        let available: Vec<Model> = available.await.unwrap_or_default();
        resolve_model_scope_from_models(patterns, &available)
    })
}

/// Resolve scoped models and print the diagnostics as CLI warnings,
/// upstream's `resolveModelScope`.
#[expect(
    clippy::print_stderr,
    reason = "upstream prints scope warnings through console.warn on the CLI surface"
)]
pub async fn resolve_model_scope<V: ModelRuntimeView>(
    patterns: &[String],
    model_runtime: &V,
    options: Option<&AuthOptions>,
) -> Vec<ScopedModel> {
    let result = resolve_model_scope_with_diagnostics(patterns, model_runtime, options).await;
    for diagnostic in &result.diagnostics {
        eprintln!("{}", format_scope_warning(diagnostic));
    }
    result.scoped_models
}

/// The result of resolving a CLI model, upstream's `ResolveCliModelResult`.
#[derive(Clone, Debug, Default)]
pub struct ResolveCliModelResult {
    /// The resolved model.
    pub model: Option<Model>,
    /// The thinking level a valid `:level` suffix carried.
    pub thinking_level: Option<ThinkingLevel>,
    /// The warning the resolution produced.
    pub warning: Option<String>,
    /// The error message suitable for CLI display; when set, the model is
    /// `None`.
    pub error: Option<String>,
}

/// The CLI resolution inputs, upstream's `resolveCliModel` options.
pub struct ResolveCliModelOptions<'a, V: ModelRuntimeView> {
    /// The `--provider` flag value.
    pub cli_provider: Option<&'a str>,
    /// The `--model` flag value.
    pub cli_model: Option<&'a str>,
    /// The `--thinking` flag value.
    pub cli_thinking: Option<ThinkingLevel>,
    /// The runtime the resolution reads.
    pub model_runtime: &'a V,
}

/// Resolve a single model from CLI flags, upstream's `resolveCliModel`.
///
/// Supports `--provider <provider> --model <pattern>`, `--model
/// <provider>/<pattern>`, and fuzzy matching (exact id, then partial
/// id/name). The thinking level is parsed but not applied; the caller
/// applies it.
#[expect(
    clippy::too_many_lines,
    reason = "the 1:1 port of upstream's resolveCliModel carries the provider-inference ladder inline"
)]
#[must_use]
#[expect(
    clippy::needless_pass_by_value,
    reason = "the options struct carries references; upstream passes the options object by value"
)]
pub fn resolve_cli_model<V: ModelRuntimeView>(
    options: ResolveCliModelOptions<'_, V>,
) -> ResolveCliModelResult {
    let ResolveCliModelOptions {
        cli_provider,
        cli_model,
        cli_thinking,
        model_runtime,
    } = options;

    let Some(cli_model) = cli_model else {
        return ResolveCliModelResult::default();
    };

    // Important: use *all* models here, not just models with pre-configured
    // auth. This allows "--api-key" to be used for first-time setup.
    let available_models = model_runtime.get_models();
    if available_models.is_empty() {
        return ResolveCliModelResult {
            model: None,
            warning: None,
            thinking_level: None,
            error: Some(
                "No models available. Check your installation or add models to models.json."
                    .to_owned(),
            ),
        };
    }

    // Build canonical provider lookup (case-insensitive)
    let provider_map: BTreeMap<String, String> = available_models
        .iter()
        .map(|model| (model.provider.0.to_lowercase(), model.provider.0.clone()))
        .collect();

    let mut provider: Option<String> = cli_provider
        .and_then(|cli_provider| provider_map.get(&cli_provider.to_lowercase()).cloned());
    if cli_provider.is_some() && provider.is_none() {
        return ResolveCliModelResult {
            model: None,
            warning: None,
            thinking_level: None,
            error: Some(format!(
                "Unknown provider \"{}\". Use --list-models to see available providers/models.",
                cli_provider.unwrap_or_default()
            )),
        };
    }

    // If no explicit --provider, try to interpret "provider/model" format
    // first. When the prefix before the first slash matches a known
    // provider, prefer that interpretation over matching models whose IDs
    // literally contain slashes.
    let (mut pattern, inferred_provider) = if provider.is_none()
        && let Some((maybe_provider, rest)) = cli_model.split_once('/')
        && let Some(canonical) = provider_map.get(&maybe_provider.to_lowercase())
    {
        // Provider inference wins: prefer provider/model when the prefix
        // names a known provider.
        provider = Some(canonical.clone());
        (String::from(rest), true)
    } else {
        (cli_model.to_owned(), false)
    };

    // If no provider was inferred from the slash, try exact matches without
    // provider inference. This handles models whose IDs naturally contain
    // slashes (e.g. OpenRouter-style IDs). Prefer the sole authenticated
    // provider when there is one; otherwise require an explicit provider to
    // avoid silently selecting an unusable provider.
    if provider.is_none() {
        let lower = cli_model.to_lowercase();
        let exact_matches: Vec<Model> = available_models
            .iter()
            .filter(|model| {
                model.id.to_lowercase() == lower
                    || format!("{}/{}", model.provider.0, model.id).to_lowercase() == lower
            })
            .cloned()
            .collect();
        if exact_matches.len() == 1 {
            return ResolveCliModelResult {
                model: Some(exact_matches[0].clone()),
                warning: None,
                thinking_level: None,
                error: None,
            };
        }
        if exact_matches.len() > 1 {
            let authenticated_exact_matches: Vec<Model> = exact_matches
                .iter()
                .filter(|model| model_runtime.has_configured_auth(&model.provider.0))
                .cloned()
                .collect();
            if authenticated_exact_matches.len() == 1 {
                return ResolveCliModelResult {
                    model: Some(authenticated_exact_matches[0].clone()),
                    warning: None,
                    thinking_level: None,
                    error: None,
                };
            }
            let matches = {
                let mut references: Vec<String> = exact_matches
                    .iter()
                    .map(|model| format!("{}/{}", model.provider.0, model.id))
                    .collect();
                references.sort();
                references.join(", ")
            };
            let auth_hint = if authenticated_exact_matches.is_empty() {
                "No matching provider is authenticated."
            } else {
                "More than one matching provider is authenticated."
            };
            return ResolveCliModelResult {
                model: None,
                warning: None,
                thinking_level: None,
                error: Some(format!(
                    "Model \"{cli_model}\" is ambiguous across providers: {matches}. {auth_hint} Use --provider or provider/model."
                )),
            };
        }
    }

    if let Some(provider) = &provider {
        // If both were provided, tolerate --model <provider>/<pattern> by
        // stripping the provider prefix.
        let prefix = format!("{provider}/");
        if cli_model.to_lowercase().starts_with(&prefix.to_lowercase()) {
            pattern = String::from(&cli_model[prefix.len()..]);
        }
    }

    let candidates: Vec<Model> = provider.as_ref().map_or_else(
        || available_models.clone(),
        |provider| {
            available_models
                .iter()
                .filter(|model| &model.provider.0 == provider)
                .cloned()
                .collect()
        },
    );
    let parsed = parse_model_pattern(
        &pattern,
        &candidates,
        Some(ParseModelPatternOptions {
            allow_invalid_thinking_level_fallback: false,
        }),
    );

    if let Some(model) = &parsed.model {
        // If provider inference matched an unauthenticated provider/model
        // pair, prefer one exact raw model-id match that is authenticated.
        // This keeps "provider/model" syntax preferred when usable, but
        // handles models whose literal id starts with a known provider name
        // (for example commandcode model id "xiaomi/mimo-v2.5-pro").
        if inferred_provider && !model_runtime.has_configured_auth(&model.provider.0) {
            let raw_exact_matches: Vec<Model> = available_models
                .iter()
                .filter(|candidate| {
                    candidate.id.to_lowercase() == cli_model.to_lowercase()
                        && !models_are_equal(Some(candidate), Some(model))
                })
                .cloned()
                .collect();
            if !raw_exact_matches.is_empty() {
                let authenticated_raw_matches: Vec<Model> = raw_exact_matches
                    .iter()
                    .filter(|candidate| model_runtime.has_configured_auth(&candidate.provider.0))
                    .cloned()
                    .collect();
                if authenticated_raw_matches.len() == 1 {
                    return ResolveCliModelResult {
                        model: Some(authenticated_raw_matches[0].clone()),
                        thinking_level: None,
                        warning: None,
                        error: None,
                    };
                }
            }
        }
        return ResolveCliModelResult {
            model: Some(model.clone()),
            thinking_level: parsed.thinking_level,
            warning: parsed.warning,
            error: None,
        };
    }

    // If we inferred a provider from the slash but found no match within
    // that provider, fall back to matching the full input as a raw model id
    // across all models. This handles OpenRouter-style IDs like
    // "openai/gpt-4o:extended" where "openai" looks like a provider but the
    // full string is actually a model id on openrouter.
    if inferred_provider {
        let lower = cli_model.to_lowercase();
        if let Some(exact) = available_models
            .iter()
            .find(|model| {
                model.id.to_lowercase() == lower
                    || format!("{}/{}", model.provider.0, model.id).to_lowercase() == lower
            })
            .cloned()
        {
            return ResolveCliModelResult {
                model: Some(exact),
                warning: None,
                thinking_level: None,
                error: None,
            };
        }
        let fallback = parse_model_pattern(
            cli_model,
            &available_models,
            Some(ParseModelPatternOptions {
                allow_invalid_thinking_level_fallback: false,
            }),
        );
        if fallback.model.is_some() {
            return ResolveCliModelResult {
                model: fallback.model,
                thinking_level: fallback.thinking_level,
                warning: fallback.warning,
                error: None,
            };
        }
    }

    if let Some(provider) = &provider {
        // Parse thinking level suffix from the pattern before building the
        // fallback model, but only when --thinking is not explicitly
        // provided. e.g. "zai-org/GLM-5.1-FP8:high" →
        // modelId="zai-org/GLM-5.1-FP8", fallbackThinking="high"
        let (fallback_pattern, fallback_thinking) = if cli_thinking.is_none()
            && let Some((prefix, suffix)) = pattern.rsplit_once(':')
            && is_valid_thinking_level(suffix)
        {
            (String::from(prefix), thinking_level_from_str(suffix))
        } else {
            (pattern.clone(), None)
        };

        let fallback_model = build_fallback_model(provider, &fallback_pattern, &available_models);
        if let Some(fallback_model) = fallback_model {
            let requested_thinking = cli_thinking.or(fallback_thinking);
            let mut model = fallback_model;
            if requested_thinking.is_some_and(|level| level != ThinkingLevel::Off) {
                model.reasoning = true;
            }
            let fallback_warning = parsed.warning.map_or_else(
                || format!(
                    "Model \"{fallback_pattern}\" not found for provider \"{provider}\". Using custom model id."
                ),
                |warning| format!(
                    "{warning} Model \"{fallback_pattern}\" not found for provider \"{provider}\". Using custom model id."
                ),
            );
            return ResolveCliModelResult {
                model: Some(model),
                thinking_level: fallback_thinking,
                warning: Some(fallback_warning),
                error: None,
            };
        }
    }

    let display = provider.as_ref().map_or_else(
        || cli_model.to_owned(),
        |provider| format!("{provider}/{pattern}"),
    );
    ResolveCliModelResult {
        model: None,
        thinking_level: None,
        warning: parsed.warning,
        error: Some(format!(
            "Model \"{display}\" not found. Use --list-models to see available models."
        )),
    }
}

/// The initial-model result, upstream's `InitialModelResult`.
#[derive(Clone, Debug, Default)]
pub struct InitialModelResult {
    /// The model to start with.
    pub model: Option<Model>,
    /// The thinking level to start with.
    pub thinking_level: ThinkingLevel,
    /// The fallback message a restored-model substitution printed.
    pub fallback_message: Option<String>,
}

impl<V: ModelRuntimeView> std::fmt::Debug for ResolveCliModelOptions<'_, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolveCliModelOptions")
            .field("cli_provider", &self.cli_provider)
            .field("cli_model", &self.cli_model)
            .field("cli_thinking", &self.cli_thinking)
            .finish_non_exhaustive()
    }
}

impl<V: ModelRuntimeView> std::fmt::Debug for FindInitialModelOptions<'_, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FindInitialModelOptions")
            .field("cli_provider", &self.cli_provider)
            .field("cli_model", &self.cli_model)
            .field("scoped_models", &self.scoped_models)
            .field("is_continuing", &self.is_continuing)
            .field("default_provider", &self.default_provider)
            .field("default_model_id", &self.default_model_id)
            .field("default_thinking_level", &self.default_thinking_level)
            .field("model_thinking_levels", &self.model_thinking_levels)
            .finish_non_exhaustive()
    }
}

/// The CLI-resolution error [`find_initial_model`] reports where upstream
/// exits, the port of `console.error + process.exit(1)`.
#[derive(Clone, Debug)]
pub struct InitialModelError(pub String);

impl std::fmt::Display for InitialModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InitialModelError {}

/// The `findInitialModel` inputs, upstream's options object.
pub struct FindInitialModelOptions<'a, V: ModelRuntimeView> {
    /// The `--provider` flag value.
    pub cli_provider: Option<&'a str>,
    /// The `--model` flag value.
    pub cli_model: Option<&'a str>,
    /// The scoped models the session carries.
    pub scoped_models: &'a [ScopedModel],
    /// Whether the session is continuing or resuming.
    pub is_continuing: bool,
    /// The saved default provider from settings.
    pub default_provider: Option<&'a str>,
    /// The saved default model id from settings.
    pub default_model_id: Option<&'a str>,
    /// The saved default thinking level from settings.
    pub default_thinking_level: Option<ThinkingLevel>,
    /// The per-model thinking levels from settings.
    pub model_thinking_levels: Option<&'a BTreeMap<String, ThinkingLevel>>,
    /// The runtime the resolution reads.
    pub model_runtime: &'a V,
}

/// Find the initial model, upstream's `findInitialModel`: CLI args, then the
/// first scoped model, then the saved default, then the first available
/// known-provider default, then the first available model.
///
/// # Errors
/// The CLI resolution failure, where upstream prints and exits.
#[expect(
    clippy::print_stderr,
    reason = "upstream prints the CLI model error through console.error before exiting"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "the options struct carries references; upstream passes the options object by value"
)]
pub fn find_initial_model<V: ModelRuntimeView>(
    options: FindInitialModelOptions<'_, V>,
) -> Result<InitialModelResult, InitialModelError> {
    let FindInitialModelOptions {
        cli_provider,
        cli_model,
        scoped_models,
        is_continuing,
        default_provider,
        default_model_id,
        default_thinking_level,
        model_thinking_levels,
        model_runtime,
    } = options;

    // 1. CLI args take priority
    if let (Some(cli_provider), Some(cli_model)) = (cli_provider, cli_model) {
        let resolved = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: Some(cli_provider),
            cli_model: Some(cli_model),
            cli_thinking: None,
            model_runtime,
        });
        if let Some(error) = resolved.error {
            eprintln!("{error}");
            return Err(InitialModelError(error));
        }
        if let Some(model) = resolved.model {
            return Ok(InitialModelResult {
                model: Some(model),
                thinking_level: DEFAULT_THINKING_LEVEL,
                fallback_message: None,
            });
        }
    }

    // 2. Use first model from scoped models (skip if continuing/resuming)
    if !scoped_models.is_empty() && !is_continuing {
        let scoped_model = &scoped_models[0];
        let per_model = model_thinking_levels.and_then(|levels| {
            levels
                .get(&format!(
                    "{}/{}",
                    scoped_model.model.provider.0, scoped_model.model.id
                ))
                .copied()
        });
        return Ok(InitialModelResult {
            model: Some(scoped_model.model.clone()),
            thinking_level: scoped_model
                .thinking_level
                .or(per_model)
                .or(default_thinking_level)
                .unwrap_or(DEFAULT_THINKING_LEVEL),
            fallback_message: None,
        });
    }

    // 3. Try saved default from settings if auth is configured.
    if let (Some(default_provider), Some(default_model_id)) = (default_provider, default_model_id) {
        let found = model_runtime.get_model(default_provider, default_model_id);
        if let Some(found) = found
            && model_runtime.has_configured_auth(&found.provider.0)
        {
            let per_model = model_thinking_levels.and_then(|levels| {
                levels
                    .get(&format!("{default_provider}/{default_model_id}"))
                    .copied()
            });
            let thinking_level = per_model
                .or(default_thinking_level)
                .unwrap_or(DEFAULT_THINKING_LEVEL);
            return Ok(InitialModelResult {
                model: Some(found),
                thinking_level,
                fallback_message: None,
            });
        }
    }

    // 4. Try first available model with valid API key
    let available_models = model_runtime.get_available_snapshot();

    if !available_models.is_empty() {
        // Try to find a default model from known providers
        for provider in KNOWN_PROVIDER_DEFAULT_ORDER {
            let Some(default_id) = default_model_per_provider(provider) else {
                continue;
            };
            if let Some(matched) = available_models
                .iter()
                .find(|model| model.provider.0 == *provider && model.id == default_id)
            {
                return Ok(InitialModelResult {
                    model: Some(matched.clone()),
                    thinking_level: DEFAULT_THINKING_LEVEL,
                    fallback_message: None,
                });
            }
        }

        // If no default found, use first available
        return Ok(InitialModelResult {
            model: Some(available_models[0].clone()),
            thinking_level: DEFAULT_THINKING_LEVEL,
            fallback_message: None,
        });
    }

    // 5. No model found
    Ok(InitialModelResult {
        model: None,
        thinking_level: DEFAULT_THINKING_LEVEL,
        fallback_message: None,
    })
}

/// The provider ids a default model is configured for, upstream's
/// `Object.keys(defaultModelPerProvider)` iteration order.
pub const KNOWN_PROVIDER_DEFAULT_ORDER: &[&str] = &[
    "amazon-bedrock",
    "ant-ling",
    "anthropic",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "radius",
    "nvidia",
    "deepseek",
    "google",
    "google-vertex",
    "github-copilot",
    "openrouter",
    "vercel-ai-gateway",
    "xai",
    "groq",
    "cerebras",
    "zai",
    "zai-coding-cn",
    "mistral",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "huggingface",
    "fireworks",
    "together",
    "baseten",
    "opencode",
    "opencode-go",
    "kimi-coding",
    "cloudflare-workers-ai",
    "cloudflare-ai-gateway",
    "qwen-token-plan",
    "qwen-token-plan-cn",
    "qwen-token-plan-individual",
    "xiaomi",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-sgp",
];

/// The restore outcome, upstream's `restoreModelFromSession` return shape.
#[derive(Clone, Debug, Default)]
pub struct RestoreModelResult {
    /// The model to continue with.
    pub model: Option<Model>,
    /// The fallback message the substitution printed.
    pub fallback_message: Option<String>,
}

/// Restore a model from a session, with fallback to available models,
/// upstream's `restoreModelFromSession`.
#[expect(
    clippy::print_stdout,
    reason = "upstream echoes model restore status through console.log on the CLI surface"
)]
#[expect(
    clippy::print_stderr,
    reason = "upstream echoes restore failures through console.error on the CLI surface"
)]
pub fn restore_model_from_session<V: ModelRuntimeView>(
    saved_provider: &str,
    saved_model_id: &str,
    current_model: Option<&Model>,
    should_print_messages: bool,
    model_runtime: &V,
) -> RestoreModelResult {
    let restored_model = model_runtime.get_model(saved_provider, saved_model_id);

    // Check if restored model exists and still has auth configured
    let has_configured_auth = restored_model
        .as_ref()
        .is_some_and(|model| model_runtime.has_configured_auth(&model.provider.0));

    if let Some(restored_model) = &restored_model
        && has_configured_auth
    {
        if should_print_messages {
            println!("Restored model: {saved_provider}/{saved_model_id}");
        }
        return RestoreModelResult {
            model: Some(restored_model.clone()),
            fallback_message: None,
        };
    }

    // Model not found or no API key - fall back
    let reason = if restored_model.is_none() {
        "model no longer exists"
    } else {
        "no auth configured"
    };

    if should_print_messages {
        eprintln!("Warning: Could not restore model {saved_provider}/{saved_model_id} ({reason}).");
    }

    // If we already have a model, use it as fallback
    if let Some(current_model) = current_model {
        if should_print_messages {
            println!(
                "Falling back to: {}/{}",
                current_model.provider.0, current_model.id
            );
        }
        return RestoreModelResult {
            model: Some(current_model.clone()),
            fallback_message: Some(format!(
                "Could not restore model {saved_provider}/{saved_model_id} ({reason}). Using {}/{}.",
                current_model.provider.0, current_model.id
            )),
        };
    }

    // Try to find any available model
    let available_models = model_runtime.get_available_snapshot();

    if !available_models.is_empty() {
        // Try to find a default model from known providers
        let mut fallback_model: Option<Model> = None;
        for provider in KNOWN_PROVIDER_DEFAULT_ORDER {
            let Some(default_id) = default_model_per_provider(provider) else {
                continue;
            };
            if let Some(matched) = available_models
                .iter()
                .find(|model| model.provider.0 == *provider && model.id == default_id)
            {
                fallback_model = Some(matched.clone());
                break;
            }
        }

        // If no default found, use first available
        let fallback_model = fallback_model.unwrap_or_else(|| available_models[0].clone());

        if should_print_messages {
            println!(
                "Falling back to: {}/{}",
                fallback_model.provider.0, fallback_model.id
            );
        }

        return RestoreModelResult {
            fallback_message: Some(format!(
                "Could not restore model {saved_provider}/{saved_model_id} ({reason}). Using {}/{}.",
                fallback_model.provider.0, fallback_model.id
            )),
            model: Some(fallback_model),
        };
    }

    // No models available
    RestoreModelResult {
        model: None,
        fallback_message: None,
    }
}

/// The [`ModelRuntimeView`] implementation over the real runtime, the
/// adapter the resolver's functions read through.
impl ModelRuntimeView for ModelRuntime {
    fn get_models(&self) -> Vec<Model> {
        ModelRuntimeCore::get_models(self, None)
    }

    fn get_model(&self, provider: &str, model_id: &str) -> Option<Model> {
        ModelRuntimeCore::get_model(self, provider, model_id)
    }

    fn has_configured_auth(&self, provider: &str) -> bool {
        ModelRuntimeCore::has_configured_auth(self, provider)
    }

    fn get_available_snapshot(&self) -> Vec<Model> {
        ModelRuntimeCore::get_available_snapshot(self)
    }

    fn get_available(
        &self,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'_, Result<Vec<Model>, ModelsFailure>> {
        ModelRuntimeCore::get_available(self, None, options)
    }
}

/// Keep the `Arc`-shared core's view implementation alongside the wrapper's;
/// the resolver accepts either.
impl ModelRuntimeView for Arc<ModelRuntime> {
    fn get_models(&self) -> Vec<Model> {
        ModelRuntimeView::get_models(self.as_ref())
    }

    fn get_model(&self, provider: &str, model_id: &str) -> Option<Model> {
        ModelRuntimeView::get_model(self.as_ref(), provider, model_id)
    }

    fn has_configured_auth(&self, provider: &str) -> bool {
        ModelRuntimeView::has_configured_auth(self.as_ref(), provider)
    }

    fn get_available_snapshot(&self) -> Vec<Model> {
        ModelRuntimeView::get_available_snapshot(self.as_ref())
    }

    fn get_available(
        &self,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'_, Result<Vec<Model>, ModelsFailure>> {
        ModelRuntimeView::get_available(self.as_ref(), options)
    }
}
