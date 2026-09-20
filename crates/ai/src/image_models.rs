//! The image-generation model registry, ported from
//! `packages/ai/src/image-models.ts` and `scripts/generate-image-models.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Per [ADR 0006](docs/adr/0006-generated-model-catalogs-extracted.md) the
//! generated `image-models.generated.ts` content is committed as one JSON
//! file and embedded here.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde_json::Value;

use crate::types::{ImagesApi, ImagesModel, Modality};

/// The committed image-model catalog, upstream's `image-models.generated.ts`.
const IMAGE_MODELS_JSON: &str = include_str!("image-models.generated.json");

/// The OpenRouter API base URL image models point at, upstream's
/// `OPENROUTER_BASE_URL`.
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// The image model registry: provider id to (model id, model), built once
/// from the committed catalog, upstream's module-level `imageModelRegistry`.
fn registry() -> &'static BTreeMap<String, BTreeMap<String, ImagesModel>> {
    static REGISTRY: OnceLock<BTreeMap<String, BTreeMap<String, ImagesModel>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        serde_json::from_str::<BTreeMap<String, BTreeMap<String, ImagesModel>>>(IMAGE_MODELS_JSON)
            .unwrap_or_default()
    })
}

/// Typed read of the generated image catalog, upstream's
/// `getImageModel(provider, modelId)`.
#[must_use]
pub fn get_image_model(provider: &str, model_id: &str) -> Option<ImagesModel> {
    registry()
        .get(provider)
        .and_then(|models| models.get(model_id))
        .cloned()
}

/// The image providers present in the generated catalog, upstream's
/// `getImageProviders()`.
#[must_use]
pub fn get_image_providers() -> Vec<String> {
    registry().keys().cloned().collect()
}

/// All image models of one provider, upstream's `getImageModels(provider)`.
#[must_use]
pub fn get_image_models(provider: &str) -> Vec<ImagesModel> {
    registry()
        .get(provider)
        .map(|models| models.values().cloned().collect())
        .unwrap_or_default()
}

/// Parse an OpenRouter model list into image models, upstream's
/// `parseOpenRouterImageModels`.
///
/// Keep models whose output modalities include `image`, default missing
/// input modalities to text, and scale the per-token prices to per-million
/// rates.
///
/// # Errors
/// In strict mode, a missing or empty list, or no usable image models.
pub fn parse_openrouter_image_models(
    payload: &Value,
    strict: bool,
) -> Result<Vec<ImagesModel>, Box<dyn std::error::Error + Send + Sync>> {
    let data = payload.as_object().and_then(|payload| payload.get("data"));
    let Some(data) = data
        .and_then(Value::as_array)
        .filter(|data| !data.is_empty())
    else {
        if strict {
            return Err(Box::new(std::io::Error::other(
                "OpenRouter API returned a missing or empty image model list",
            )));
        }
        return Ok(Vec::new());
    };

    let mut models = Vec::new();
    for entry in data {
        let modalities = |key: &str| -> Vec<Modality> {
            let mut seen: Vec<Modality> = Vec::new();
            for modality in entry
                .get("architecture")
                .and_then(|architecture| architecture.get(key))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let parsed = match modality.as_str() {
                    Some("text") => Some(Modality::Text),
                    Some("image") => Some(Modality::Image),
                    _ => None,
                };
                if let Some(parsed) = parsed
                    && !seen.contains(&parsed)
                {
                    seen.push(parsed);
                }
            }
            seen
        };
        let input = modalities("input_modalities");
        let output = modalities("output_modalities");

        if !output.contains(&Modality::Image) {
            continue;
        }
        let mut input = input;
        if input.is_empty() {
            input.push(Modality::Text);
        }

        let pricing = |key: &str| -> f64 {
            entry
                .get("pricing")
                .and_then(|pricing| pricing.get(key))
                .and_then(Value::as_str)
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.0)
                * 1_000_000.0
        };
        models.push(ImagesModel {
            id: entry
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            name: entry
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            api: ImagesApi::from(crate::types::KnownImagesApi::OpenrouterImages),
            provider: crate::types::ImagesProviderId::from("openrouter"),
            base_url: OPENROUTER_BASE_URL.to_owned(),
            thinking_level_map: None,
            input,
            output,
            cost: crate::types::ModelCost {
                rates: crate::types::ModelCostRates {
                    input: pricing("prompt"),
                    output: pricing("completion"),
                    cache_read: pricing("input_cache_read"),
                    cache_write: pricing("input_cache_write"),
                },
                tiers: None,
            },
            sampling_params: None,
            headers: None,
        });
    }

    if strict && models.is_empty() {
        return Err(Box::new(std::io::Error::other(
            "OpenRouter API returned no usable image models",
        )));
    }
    Ok(models)
}
