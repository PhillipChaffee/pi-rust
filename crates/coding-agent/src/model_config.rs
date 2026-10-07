//! The immutable, credential-blind `models.json` snapshot, upstream's
//! `src/core/model-config.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatement: typebox's compile-time schema walk restates as a
//! hand-rolled walk over the parsed JSON (the same approach pi-protocol's
//! wire-schema validation took), producing the same `a.b.c` paths. The
//! per-API compat union restates as pi-ai's single [`pi_ai::types::ModelCompat`]
//! struct — the wire object carries no discriminator and the field sets never
//! conflict; a field whose wire type does not fit its slot fails validation
//! instead of riding through as junk the adapters would ignore.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use pi_ai::types::{Modality, ModelCompat, ModelCost, ModelCostTier, ThinkingLevelMap};

use crate::utils::json::strip_json_comments;
use crate::utils::paths::{PathInputOptions, normalize_path};
use crate::utils::text::strip_bom;

/// One custom model definition, upstream's `ModelDefinitionSchema`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsJsonModel {
    /// The model id.
    pub id: String,
    /// The display name; defaults to the id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The wire API; defaults to the provider's api, then a built-in
    /// sibling's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// The API base URL; defaults to the provider's, then a built-in
    /// sibling's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Whether the model supports reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    /// Maps pi thinking levels to provider/model-specific values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    /// The input modalities; defaults to `["text"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<Modality>>,
    /// The pricing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    /// The context window in tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// The maximum output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Default sampling parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<BTreeMap<String, Value>>,
    /// Custom HTTP headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// Compatibility overrides for OpenAI-compatible APIs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
}

/// The cost object of a model override, upstream's optional-member
/// `ModelOverrideSchema` cost shape.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsJsonOverrideCost {
    /// The input rate override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<f64>,
    /// The output rate override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,
    /// The cache-read rate override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    /// The cache-write rate override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
    /// The tier overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<ModelCostTier>>,
}

/// One per-model override, upstream's `ModelOverrideSchema`: every field is
/// optional and applies over the composed model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsJsonModelOverride {
    /// The display name override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The reasoning override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    /// The thinking-level map override, merged over the model's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    /// The input modalities override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<Modality>>,
    /// The pricing override, field by field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelsJsonOverrideCost>,
    /// The context window override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// The maximum output tokens override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// The sampling parameters override, merged over the model's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<BTreeMap<String, Value>>,
    /// The custom HTTP headers override, resolved at request time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// The compat override, merged over the model's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
}

/// One provider's custom configuration, upstream's `ProviderConfigSchema`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsJsonProvider {
    /// The display name; defaults to the provider id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The API base URL applied to every model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// A configured API key, a `$VAR`/`${VAR}` template, or a `!command`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// The wire API applied to models without their own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// `"radius"` routes logins through the Radius OAuth flow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth: Option<String>,
    /// Headers merged into every request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// Compatibility overrides applied to every model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
    /// Send the resolved API key as `Authorization: Bearer` instead of the
    /// provider's default key header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<bool>,
    /// Custom model definitions, upserted over the built-in catalog.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<ModelsJsonModel>>,
    /// Per-model overrides applied after custom-model upserts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_overrides: Option<BTreeMap<String, ModelsJsonModelOverride>>,
}

/// The full `models.json` shape, upstream's `ModelsConfigSchema`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsJson {
    /// Providers keyed by id.
    pub providers: BTreeMap<String, ModelsJsonProvider>,
}

// The `compat` object's nested shape: routing preferences carry their own
// object schemas, and the chat-template maps carry the `$var` union.
mod compat_schema {
    pub(super) const OPEN_ROUTER_ROUTING: &[(&str, super::Schema)] = &[
        ("allow_fallbacks", super::Schema::Bool),
        ("require_parameters", super::Schema::Bool),
        (
            "data_collection",
            super::Schema::StrLits(&["deny", "allow"]),
        ),
        ("zdr", super::Schema::Bool),
        ("enforce_distillable_text", super::Schema::Bool),
        ("order", super::Schema::Array(&super::Schema::Str)),
        ("only", super::Schema::Array(&super::Schema::Str)),
        ("ignore", super::Schema::Array(&super::Schema::Str)),
        ("quantizations", super::Schema::Array(&super::Schema::Str)),
        (
            "sort",
            super::Schema::Union(&[
                super::Schema::Str,
                super::Schema::Object(&[
                    ("by", super::Schema::Str),
                    ("partition", super::Schema::NullableStr),
                ]),
            ]),
        ),
        (
            "max_price",
            super::Schema::Object(&[
                ("prompt", super::Schema::Num),
                ("completion", super::Schema::Num),
                ("image", super::Schema::Num),
                ("audio", super::Schema::Num),
                ("request", super::Schema::Num),
            ]),
        ),
        (
            "preferred_min_throughput",
            super::Schema::Union(&[
                super::Schema::Num,
                super::Schema::Object(PERCENTILE_CUTOFFS),
            ]),
        ),
        (
            "preferred_max_latency",
            super::Schema::Union(&[
                super::Schema::Num,
                super::Schema::Object(PERCENTILE_CUTOFFS),
            ]),
        ),
    ];

    pub(super) const VERCEL_GATEWAY_ROUTING: &[(&str, super::Schema)] = &[
        ("only", super::Schema::Array(&super::Schema::Str)),
        ("order", super::Schema::Array(&super::Schema::Str)),
    ];

    pub(super) const CHAT_TEMPLATE_KWARG: super::Schema = super::Schema::Union(&[
        super::Schema::Str,
        super::Schema::Num,
        super::Schema::Bool,
        super::Schema::Nullable,
        super::Schema::Object(&[
            (
                "$var",
                super::Schema::StrLits(&["thinking.enabled", "thinking.effort"]),
            ),
            ("omitWhenOff", super::Schema::Bool),
        ]),
    ]);

    pub(super) const PERCENTILE_CUTOFFS: &[(&str, super::Schema)] = &[
        ("p50", super::Schema::Num),
        ("p75", super::Schema::Num),
        ("p90", super::Schema::Num),
        ("p99", super::Schema::Num),
    ];
}

/// The schema walk's node vocabulary, upstream's typebox builders.
enum Schema {
    /// An object whose listed properties are optional; unknown keys ride.
    Object(&'static [(&'static str, Self)]),
    /// A JSON record whose values all match, upstream's `Type.Record`.
    Record(&'static Self),
    /// An array of the element schema.
    Array(&'static Self),
    /// A string.
    Str,
    /// A string or the wire's `null`, upstream's `Union(String, Null)`.
    NullableStr,
    /// A number.
    Num,
    /// A boolean.
    Bool,
    /// The wire's `null`.
    Nullable,
    /// Any JSON value, upstream's `Type.Unknown()`.
    Any,
    /// One of these exact string values, upstream's `Type.Literal` union.
    StrLits(&'static [&'static str]),
    /// A non-empty string, upstream's `Type.String({ minLength: 1 })`.
    NonEmptyStr,
    /// Any of these variants, upstream's `Type.Union`.
    Union(&'static [Self]),
}

impl Schema {
    /// The typebox-style failure message for a value that does not match.
    fn message(&self) -> String {
        match self {
            Self::Object(_) | Self::Record(_) => "Expected object".to_owned(),
            Self::Array(_) => "Expected array".to_owned(),
            Self::Str | Self::NonEmptyStr => "Expected string".to_owned(),
            Self::NullableStr => "Expected string or null".to_owned(),
            Self::Num => "Expected number".to_owned(),
            Self::Bool => "Expected boolean".to_owned(),
            Self::Nullable => "Expected null".to_owned(),
            Self::Any => "Expected any value".to_owned(),
            Self::StrLits(literals) => format!(
                "Expected union value: {}",
                literals
                    .iter()
                    .map(|literal| format!("\"{literal}\""))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            Self::Union(_) => "Expected union value".to_owned(),
        }
    }

    /// Whether the value matches this node.
    fn check(&self, value: &Value) -> bool {
        match self {
            Self::Object(properties) => {
                let Some(map) = value.as_object() else {
                    return false;
                };
                map.iter().all(|(key, property)| {
                    properties
                        .iter()
                        .find(|(name, _)| name == key)
                        .is_none_or(|(_, schema)| schema.check(property))
                })
            }
            Self::Record(element) => value
                .as_object()
                .is_some_and(|map| map.values().all(|entry| element.check(entry))),
            Self::Array(element) => value
                .as_array()
                .is_some_and(|entries| entries.iter().all(|entry| element.check(entry))),
            Self::Str => value.is_string(),
            // A non-empty string, upstream's `Type.String({ minLength: 1 })`.
            Self::NonEmptyStr => value.as_str().is_some_and(|text| !text.is_empty()),
            Self::NullableStr => value.is_string() || value.is_null(),
            Self::Num => value.is_number(),
            Self::Bool => value.is_boolean(),
            Self::Nullable => value.is_null(),
            Self::Any => true,
            Self::StrLits(literals) => value.as_str().is_some_and(|text| literals.contains(&text)),
            Self::Union(variants) => variants.iter().any(|variant| variant.check(value)),
        }
    }
}

const THINKING_LEVEL_KEYS: &[&str] = &["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/// The `thinkingLevelMap` shape: the seven pi levels, each `string | null`.
fn check_thinking_level_map(value: &Value) -> bool {
    value.as_object().is_some_and(|map| {
        map.iter().all(|(key, entry)| {
            THINKING_LEVEL_KEYS.contains(&key.as_str()) && (entry.is_string() || entry.is_null())
        })
    })
}

/// The `ModelCost` shape's tier walk: every entry carries `inputTokensAbove`
/// and the four rates.
fn check_cost_tiers(value: &Value) -> bool {
    value.as_array().is_some_and(|entries| {
        entries.iter().all(|entry| {
            entry.as_object().is_some_and(|tier| {
                tier.get("inputTokensAbove").is_some_and(Value::is_number)
                    && ["input", "output", "cacheRead", "cacheWrite"]
                        .iter()
                        .all(|rate| tier.get(*rate).is_some_and(Value::is_number))
            })
        })
    })
}

/// The models.json `cost` shape: the four required rates plus optional tiers.
fn check_cost(value: &Value) -> bool {
    let Some(map) = value.as_object() else {
        return false;
    };
    for rate in ["input", "output", "cacheRead", "cacheWrite"] {
        if !map.get(rate).is_some_and(Value::is_number) {
            return false;
        }
    }
    map.get("tiers")
        .as_ref()
        .is_none_or(|tiers| check_cost_tiers(tiers))
}

/// The `modelOverrides` cost shape, upstream's `ModelOverrideSchema.cost`:
/// every rate optional, tiers optional.
fn check_override_cost(value: &Value) -> bool {
    let Some(map) = value.as_object() else {
        return false;
    };
    for rate in ["input", "output", "cacheRead", "cacheWrite"] {
        if !map.get(rate).is_none_or(Value::is_number) {
            return false;
        }
    }
    map.get("tiers")
        .as_ref()
        .is_none_or(|tiers| check_cost_tiers(tiers))
}

/// The models.json `compat` object's accepted surface, upstream's
/// `ProviderCompatSchema` union restated over [`ModelCompat`]'s fields.
fn check_compat(value: &Value) -> bool {
    let Some(map) = value.as_object() else {
        return false;
    };
    for (key, entry) in map {
        let ok = match key.as_str() {
            "supportsStore"
            | "supportsDeveloperRole"
            | "supportsReasoningEffort"
            | "supportsUsageInStreaming"
            | "supportsFinishReason"
            | "requiresToolResultName"
            | "requiresAssistantAfterToolResult"
            | "requiresThinkingAsText"
            | "requiresReasoningContentOnAssistantMessages"
            | "supportsOpenAIGrammarTools"
            | "supportsStrictMode"
            | "sendSessionAffinityHeaders"
            | "supportsLongCacheRetention"
            | "supportsAdditionalTools"
            | "supportsToolSearch"
            | "supportsMaxOutputTokens"
            | "supportsEagerToolInputStreaming"
            | "supportsCacheControlOnTools"
            | "supportsTemperature"
            | "forceAdaptiveThinking"
            | "allowEmptySignature"
            | "supportsStrictTools"
            | "supportsMidConvoEffort"
            | "supportsToolReferences"
            | "zaiToolStream"
            | "supportsThinkingTokenBudget"
            | "supportsExplicitPromptCacheMode"
            | "supportsBedrockStrictMode" => entry.is_boolean(),
            "maxTokensField" => {
                matches!(entry.as_str(), Some("max_completion_tokens" | "max_tokens"))
            }
            "thinkingFormat" => matches!(
                entry.as_str(),
                Some(
                    "openai"
                        | "openrouter"
                        | "together"
                        | "baseten"
                        | "deepseek"
                        | "zai"
                        | "qwen"
                        | "chat-template"
                        | "qwen-chat-template"
                        | "string-thinking"
                        | "ant-ling"
                )
            ),
            "chatTemplateKwargs" | "chatTemplateArgs" => {
                Schema::Record(&compat_schema::CHAT_TEMPLATE_KWARG).check(entry)
            }
            "cacheControlFormat" => matches!(entry.as_str(), Some("anthropic")),
            "openRouterRouting" => Schema::Object(compat_schema::OPEN_ROUTER_ROUTING).check(entry),
            "vercelGatewayRouting" => {
                Schema::Object(compat_schema::VERCEL_GATEWAY_ROUTING).check(entry)
            }
            "deferredToolsMode" => matches!(entry.as_str(), Some("kimi")),
            "sessionAffinityFormat" => matches!(
                entry.as_str(),
                Some("openai" | "openai-nosession" | "openrouter")
            ),
            "vllmPriority" => entry.is_number(),

            "thinkingTokenBudgetField" => matches!(
                entry.as_str(),
                Some("thinking_token_budget" | "thinking_budget" | "thinking_budget_tokens")
            ),
            _ => true,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Walk one entry of a model definition or override, upstream's shared
/// `ModelDefinitionSchema` members. `override_cost` selects the override
/// cost's all-optional rates, upstream's `ModelOverrideSchema.cost`.
fn check_model_fields(
    prefix: &str,
    map: &serde_json::Map<String, Value>,
    override_cost: bool,
    errors: &mut Vec<String>,
) {
    check_member(prefix, map, "name", &Schema::NonEmptyStr, errors);
    check_member(prefix, map, "api", &Schema::NonEmptyStr, errors);
    check_member(prefix, map, "baseUrl", &Schema::NonEmptyStr, errors);
    check_member(prefix, map, "reasoning", &Schema::Bool, errors);
    if let Some(value) = map.get("thinkingLevelMap")
        && !check_thinking_level_map(value)
    {
        errors.push(format!("{prefix}.thinkingLevelMap: Expected object"));
    }
    if let Some(value) = map.get("input")
        && !Schema::Array(&Schema::StrLits(&["text", "image"])).check(value)
    {
        errors.push(format!(
            "{prefix}.input: Expected union value: \"text\" | \"image\""
        ));
    }
    if let Some(value) = map.get("cost")
        && !(if override_cost {
            check_override_cost(value)
        } else {
            check_cost(value)
        })
    {
        errors.push(format!("{prefix}.cost: Expected object"));
    }
    check_member(prefix, map, "contextWindow", &Schema::Num, errors);
    check_member(prefix, map, "maxTokens", &Schema::Num, errors);
    if let Some(value) = map.get("samplingParams")
        && !Schema::Record(&Schema::Any).check(value)
    {
        errors.push(format!("{prefix}.samplingParams: Expected object"));
    }
    if let Some(value) = map.get("headers")
        && !Schema::Record(&Schema::Str).check(value)
    {
        errors.push(format!("{prefix}.headers: Expected object"));
    }
    if let Some(value) = map.get("compat")
        && !check_compat(value)
    {
        errors.push(format!("{prefix}.compat: Expected object"));
    }
}

/// Check one optional member, appending the dotted path on failure.
fn check_member(
    prefix: &str,
    map: &serde_json::Map<String, Value>,
    key: &str,
    schema: &Schema,
    errors: &mut Vec<String>,
) {
    if let Some(value) = map.get(key)
        && !schema.check(value)
    {
        errors.push(format!("{prefix}.{key}: {}", schema.message()));
    }
}

/// The models.json schema walk, upstream's `validateModelsConfig.Check`.
fn validate_models_config(value: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let Some(map) = value.as_object() else {
        return vec!["root: Expected object".to_owned()];
    };
    let Some(providers) = map.get("providers") else {
        errors.push("providers: Required property missing".to_owned());
        return errors;
    };
    let Some(provider_entries) = providers.as_object() else {
        errors.push("providers: Expected object".to_owned());
        return errors;
    };
    for (provider_id, provider_value) in provider_entries {
        let prefix = format!("providers.{provider_id}");
        let Some(provider) = provider_value.as_object() else {
            errors.push(format!("{prefix}: Expected object"));
            continue;
        };
        check_member(&prefix, provider, "name", &Schema::NonEmptyStr, &mut errors);
        check_member(
            &prefix,
            provider,
            "baseUrl",
            &Schema::NonEmptyStr,
            &mut errors,
        );
        check_member(
            &prefix,
            provider,
            "apiKey",
            &Schema::NonEmptyStr,
            &mut errors,
        );
        check_member(&prefix, provider, "api", &Schema::NonEmptyStr, &mut errors);
        check_member(
            &prefix,
            provider,
            "oauth",
            &Schema::StrLits(&["radius"]),
            &mut errors,
        );
        if let Some(value) = provider.get("headers")
            && !Schema::Record(&Schema::Str).check(value)
        {
            errors.push(format!("{prefix}.headers: Expected object"));
        }
        if let Some(value) = provider.get("compat")
            && !check_compat(value)
        {
            errors.push(format!("{prefix}.compat: Expected object"));
        }
        check_member(&prefix, provider, "authHeader", &Schema::Bool, &mut errors);
        if let Some(models) = provider.get("models") {
            match models.as_array() {
                Some(entries) => {
                    for (index, model_value) in entries.iter().enumerate() {
                        let model_prefix = format!("{prefix}.models.{index}");
                        let Some(model) = model_value.as_object() else {
                            errors.push(format!("{model_prefix}: Expected object"));
                            continue;
                        };
                        if model
                            .get("id")
                            .is_none_or(|id| id.as_str().is_none_or(str::is_empty))
                        {
                            errors.push(format!("{model_prefix}.id: Expected string"));
                        }
                        check_model_fields(&model_prefix, model, false, &mut errors);
                    }
                }
                None => errors.push(format!("{prefix}.models: Expected array")),
            }
        }
        if let Some(overrides) = provider.get("modelOverrides") {
            match overrides.as_object() {
                Some(entries) => {
                    for (model_id, override_value) in entries {
                        let override_prefix = format!("{prefix}.modelOverrides.{model_id}");
                        let Some(override_object) = override_value.as_object() else {
                            errors.push(format!("{override_prefix}: Expected object"));
                            continue;
                        };
                        check_model_fields(&override_prefix, override_object, true, &mut errors);
                    }
                }
                None => errors.push(format!("{prefix}.modelOverrides: Expected object")),
            }
        }
    }
    errors
}

/// One immutable load of `models.json`, upstream's `ModelConfig`.
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    providers: Arc<BTreeMap<String, ModelsJsonProvider>>,
    error: Option<String>,
}

impl ModelConfig {
    /// Load `models.json` from `models_json_path`; a missing path or a
    /// missing file yields an empty snapshot, while read/parse/schema
    /// failures yield an empty snapshot carrying the error text.
    ///
    /// # Errors
    /// A `models.json` path that does not normalize.
    pub fn load(models_json_path: Option<&str>) -> Result<Self, String> {
        let Some(models_json_path) = models_json_path else {
            return Ok(Self::default());
        };
        let path = match normalize_path(models_json_path, &PathInputOptions::default()) {
            Ok(path) => path,
            Err(error) => return Err(error.to_string()),
        };
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => {
                return Ok(Self {
                    providers: Arc::new(BTreeMap::new()),
                    error: Some(format!(
                        "Failed to load models.json: {error}\n\nFile: {path}"
                    )),
                });
            }
        };

        let parsed: Value =
            match serde_json::from_str(strip_json_comments(strip_bom(&content)).trim_end()) {
                Ok(parsed) => parsed,
                Err(error) => {
                    return Ok(Self {
                        providers: Arc::new(BTreeMap::new()),
                        error: Some(format!(
                            "Failed to parse models.json: {error}\n\nFile: {path}"
                        )),
                    });
                }
            };

        let errors = validate_models_config(&parsed);
        if !errors.is_empty() {
            let listed = errors
                .iter()
                .map(|error| format!("  - {error}"))
                .collect::<Vec<_>>()
                .join("\n");
            return Ok(Self {
                providers: Arc::new(BTreeMap::new()),
                error: Some(format!(
                    "Invalid models.json schema:\n{listed}\n\nFile: {path}"
                )),
            });
        }

        let config: ModelsJson = serde_json::from_value(parsed)
            .map_err(|error| format!("Failed to parse models.json: {error}\n\nFile: {path}"))?;
        Ok(Self {
            providers: Arc::new(config.providers),
            error: None,
        })
    }

    /// One provider's configuration.
    #[must_use]
    pub fn get_provider(&self, provider_id: &str) -> Option<&ModelsJsonProvider> {
        self.providers.get(provider_id)
    }

    /// Every configured provider id.
    #[must_use]
    pub fn get_provider_ids(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// The load error, when the file was unreadable, unparseable, or
    /// schema-invalid.
    #[must_use]
    pub fn get_error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}
