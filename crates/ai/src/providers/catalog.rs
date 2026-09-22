//! The builtin generated-catalog registry, ported from
//! `packages/ai/src/models.generated.ts` and the per-provider
//! `*.models.ts` shards at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Per [ADR 0006](docs/adr/0006-generated-model-catalogs-extracted.md) the
//! shards are committed JSON extracted from the pinned upstream build and
//! embedded here; the TypeScript `flattenModelCatalog` machinery becomes one
//! parse at first use. Porting restatement: model maps are
//! [`BTreeMap`]-ordered by model id where TypeScript preserved the generated
//! insertion order — no ported test depends on that order.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::types::Model;

/// The committed shard of one provider, keyed by the provider id the wire
/// carries.
macro_rules! shard {
    ($provider:literal) => {
        include_str!(concat!("data/", $provider, ".json"))
    };
}

/// Every committed provider shard, keyed by provider id, upstream's `MODELS`
/// aggregate.
pub const SHARDS: &[(&str, &str)] = &[
    ("amazon-bedrock", shard!("amazon-bedrock")),
    ("ant-ling", shard!("ant-ling")),
    ("anthropic", shard!("anthropic")),
    ("azure-openai-responses", shard!("azure-openai-responses")),
    ("baseten", shard!("baseten")),
    ("cerebras", shard!("cerebras")),
    ("cloudflare-ai-gateway", shard!("cloudflare-ai-gateway")),
    ("cloudflare-workers-ai", shard!("cloudflare-workers-ai")),
    ("deepseek", shard!("deepseek")),
    ("fireworks", shard!("fireworks")),
    ("github-copilot", shard!("github-copilot")),
    ("google", shard!("google")),
    ("google-vertex", shard!("google-vertex")),
    ("groq", shard!("groq")),
    ("huggingface", shard!("huggingface")),
    ("kimi-coding", shard!("kimi-coding")),
    ("minimax", shard!("minimax")),
    ("minimax-cn", shard!("minimax-cn")),
    ("mistral", shard!("mistral")),
    ("moonshotai", shard!("moonshotai")),
    ("moonshotai-cn", shard!("moonshotai-cn")),
    ("nvidia", shard!("nvidia")),
    ("openai", shard!("openai")),
    ("openai-codex", shard!("openai-codex")),
    ("opencode", shard!("opencode")),
    ("opencode-go", shard!("opencode-go")),
    ("openrouter", shard!("openrouter")),
    ("qwen-token-plan", shard!("qwen-token-plan")),
    ("qwen-token-plan-cn", shard!("qwen-token-plan-cn")),
    (
        "qwen-token-plan-individual",
        shard!("qwen-token-plan-individual"),
    ),
    ("together", shard!("together")),
    ("vercel-ai-gateway", shard!("vercel-ai-gateway")),
    ("xai", shard!("xai")),
    ("xiaomi", shard!("xiaomi")),
    ("xiaomi-token-plan-ams", shard!("xiaomi-token-plan-ams")),
    ("xiaomi-token-plan-cn", shard!("xiaomi-token-plan-cn")),
    ("xiaomi-token-plan-sgp", shard!("xiaomi-token-plan-sgp")),
    ("zai", shard!("zai")),
    ("zai-coding-cn", shard!("zai-coding-cn")),
];

/// The generated data manifest, upstream's `data/.manifest.json`.
const MANIFEST: &str = include_str!("data/.manifest.json");

/// The committed manifest, parsed once. `None` when the committed file
/// stopped matching its schema — the data-validation gate reports it.
fn manifest() -> Option<&'static crate::model_data::ModelDataManifest> {
    static PARSED: OnceLock<Option<crate::model_data::ModelDataManifest>> = OnceLock::new();
    PARSED
        .get_or_init(|| serde_json::from_str(MANIFEST).ok())
        .as_ref()
}

/// One provider's flattened catalog: model id to model, upstream's
/// `flattenModelCatalog("provider", values)` output.
pub type ProviderCatalog = BTreeMap<String, Model>;

/// Parse every shard, upstream's `MODELS` aggregate construction.
fn build_registry() -> BTreeMap<String, ProviderModels> {
    let mut registry: BTreeMap<String, ProviderModels> = BTreeMap::new();
    for (provider_id, shard) in SHARDS {
        registry.insert((*provider_id).to_owned(), parse_shard(shard));
    }
    registry
}

/// The full builtin catalog: provider id to (model id, model). `None` when a
/// committed shard stopped parsing — the data-validation gate reports it.
fn registry() -> &'static BTreeMap<String, ProviderModels> {
    static REGISTRY: OnceLock<BTreeMap<String, ProviderModels>> = OnceLock::new();
    REGISTRY.get_or_init(build_registry)
}

/// One provider's parsed shard: model id to model.
type ProviderModels = BTreeMap<String, Model>;

/// Parse a shard into its flattened model map.
fn parse_shard(shard: &str) -> ProviderModels {
    let grouped: BTreeMap<String, BTreeMap<String, Model>> =
        serde_json::from_str(shard).unwrap_or_default();
    let mut models = ProviderModels::new();
    for group in grouped.values() {
        for (model_id, model) in group {
            models.insert(model_id.clone(), model.clone());
        }
    }
    models
}

/// Providers present in the generated catalog, upstream's `BuiltinProvider`.
/// `KnownProvider` additionally includes purely dynamic providers (e.g.
/// "radius") that have no static catalog entry.
#[must_use]
pub fn builtin_provider_ids() -> Vec<String> {
    registry().keys().cloned().collect()
}

/// Typed read of the generated builtin catalog, upstream's
/// `getBuiltinModel(provider, modelId)`.
#[must_use]
pub fn get_builtin_model(provider: &str, model_id: &str) -> Option<Model> {
    registry()
        .get(provider)
        .and_then(|models| models.get(model_id))
        .cloned()
}

/// Generation timestamp shared by all builtin provider catalogs, upstream's
/// `getBuiltinModelDataGeneratedAt`.
#[must_use]
pub fn get_builtin_model_data_generated_at() -> Option<i64> {
    let manifest = manifest()?;
    parse_generated_at(&manifest.generated_at)
}

/// All builtin models of one provider, upstream's `getBuiltinModels(provider)`.
#[must_use]
pub fn get_builtin_models(provider: &str) -> Vec<Model> {
    registry()
        .get(provider)
        .map(|models| models.values().cloned().collect())
        .unwrap_or_default()
}

/// The provider id a shard's models carry, exposed for the data-validation
/// gate.
#[must_use]
pub fn shard_provider_ids() -> Vec<String> {
    SHARDS.iter().map(|(id, _)| (*id).to_owned()).collect()
}

/// Parse the manifest's ISO timestamp into unix milliseconds.
fn parse_generated_at(value: &str) -> Option<i64> {
    let bytes = value.as_bytes();
    if bytes.len() != 24 {
        return None;
    }
    let digit = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
    let char_at = |index: usize| bytes[index] as char;
    let valid = digit(0..4)
        && char_at(4) == '-'
        && digit(5..7)
        && char_at(7) == '-'
        && digit(8..10)
        && char_at(10) == 'T'
        && digit(11..13)
        && char_at(13) == ':'
        && digit(14..16)
        && char_at(16) == ':'
        && digit(17..19)
        && char_at(19) == '.'
        && digit(20..23)
        && char_at(23) == 'Z';
    valid.then_some(()).and_then(|()| {
        let year: i64 = value[..4].parse().ok()?;
        let month: i64 = value[5..7].parse().ok()?;
        let day: i64 = value[8..10].parse().ok()?;
        let hour: i64 = value[11..13].parse().ok()?;
        let minute: i64 = value[14..16].parse().ok()?;
        let second: i64 = value[17..19].parse().ok()?;
        let millis: i64 = value[20..23].parse().ok()?;
        // Days-per-month from the civil calendar; leap years included.
        let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let february = if leap { 29 } else { 28 };
        let month_len = if month == 2 {
            february
        } else {
            days_in_month
                .get(usize::try_from(month.saturating_sub(1)).unwrap_or(usize::MAX))
                .copied()
                .unwrap_or(0)
        };
        if !(1..=12).contains(&month) || !(1..=month_len).contains(&day) {
            return None;
        }
        // Days since the Unix epoch, matching Date.UTC.
        let mut days = 0;
        for year_value in 1970..year {
            days += if (year_value % 4 == 0 && year_value % 100 != 0) || year_value % 400 == 0 {
                366
            } else {
                365
            };
        }
        for month_value in 1..month {
            days += if month_value == 2 {
                february
            } else {
                days_in_month[usize::try_from(month_value - 1).unwrap_or(0)]
            };
        }
        days += day - 1;
        Some((((days * 24 + hour) * 60 + minute) * 60 + second) * 1000 + millis)
    })
}
