//! The builtin registry, ported from `packages/ai/src/providers/all.ts` at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream's side-effectful module registration becomes explicit
//! construction: every call builds fresh providers, the way upstream's
//! factory calls do.

use std::sync::Arc;

use crate::models::{CreateModelsOptions, Models, Provider, create_models};
use crate::providers::catalog::{builtin_provider_ids, get_builtin_models};
use crate::types::ProviderId;

/// All builtin providers, freshly constructed, upstream's
/// `builtinProviders()`. The order matches the pinned upstream registry.
#[must_use]
pub fn builtin_providers() -> Vec<Arc<dyn Provider>> {
    vec![
        crate::providers::amazon_bedrock::amazon_bedrock_provider(),
        crate::providers::ant_ling::ant_ling_provider(),
        crate::providers::anthropic::anthropic_provider(),
        crate::providers::azure_openai_responses::azure_openai_responses_provider(),
        crate::providers::baseten::baseten_provider(),
        crate::providers::cerebras::cerebras_provider(),
        crate::providers::cloudflare_ai_gateway::cloudflare_ai_gateway_provider(),
        crate::providers::cloudflare_workers_ai::cloudflare_workers_ai_provider(),
        crate::providers::deepseek::deepseek_provider(),
        crate::providers::fireworks::fireworks_provider(),
        crate::providers::github_copilot::github_copilot_provider(),
        crate::providers::google::google_provider(),
        crate::providers::google_vertex::google_vertex_provider(),
        crate::providers::groq::groq_provider(),
        crate::providers::huggingface::huggingface_provider(),
        crate::providers::kimi_coding::kimi_coding_provider(),
        crate::providers::minimax::minimax_provider(),
        crate::providers::minimax_cn::minimax_cn_provider(),
        crate::providers::mistral::mistral_provider(),
        crate::providers::moonshotai::moonshotai_provider(),
        crate::providers::moonshotai_cn::moonshotai_cn_provider(),
        crate::providers::nvidia::nvidia_provider(),
        crate::providers::openai::openai_provider(),
        crate::providers::openai_codex::openai_codex_provider(),
        crate::providers::opencode::opencode_provider(),
        crate::providers::opencode_go::opencode_go_provider(),
        crate::providers::openrouter::openrouter_provider(),
        crate::providers::qwen_token_plan::qwen_token_plan_provider(),
        crate::providers::qwen_token_plan_cn::qwen_token_plan_cn_provider(),
        crate::providers::qwen_token_plan_individual::qwen_token_plan_individual_provider(),
        crate::providers::radius::radius_provider(
            crate::providers::radius::RadiusProviderOptions::default(),
        ),
        crate::providers::together::together_provider(),
        crate::providers::vercel_ai_gateway::vercel_ai_gateway_provider(),
        crate::providers::xai::xai_provider(),
        crate::providers::xiaomi::xiaomi_provider(),
        crate::providers::xiaomi_token_plan_ams::xiaomi_token_plan_ams_provider(),
        crate::providers::xiaomi_token_plan_cn::xiaomi_token_plan_cn_provider(),
        crate::providers::xiaomi_token_plan_sgp::xiaomi_token_plan_sgp_provider(),
        crate::providers::zai::zai_provider(),
        crate::providers::zai_coding_cn::zai_coding_cn_provider(),
    ]
}

/// A `Models` collection with every builtin provider registered, upstream's
/// `builtinModels(options)`.
#[must_use]
pub fn builtin_models(options: Option<CreateModelsOptions>) -> Models {
    let models = create_models(options);
    for provider in builtin_providers() {
        models.set_provider(provider);
    }
    models
}

/// The provider ids present in the generated catalog, exposed so the
/// committed shards can be checked against the code-level aggregator,
/// upstream's `getBuiltinProviders()`.
#[must_use]
pub fn builtin_catalog_provider_ids() -> Vec<String> {
    builtin_provider_ids()
}

/// The builtin models of one provider by registry id, re-exported for the
/// catalog tests, upstream's `getBuiltinModels`.
#[must_use]
pub fn builtin_models_of(provider: &str) -> Vec<crate::types::Model> {
    get_builtin_models(provider)
}

/// The provider id a builtin registry entry carries as a [`ProviderId`].
#[must_use]
pub fn builtin_provider_id(id: &str) -> ProviderId {
    ProviderId::from(id)
}

/// All builtin image-generation providers, freshly constructed, upstream's
/// `builtinImagesProviders()`.
#[must_use]
pub fn builtin_images_providers() -> Vec<Arc<dyn crate::images_models::ImagesProvider>> {
    vec![Arc::new(
        crate::providers::openrouter_images::openrouter_images_provider(),
    )]
}

/// An `ImagesModels` collection with every builtin image-generation provider
/// registered, upstream's `builtinImagesModels(options)`.
#[must_use]
pub fn builtin_images_models(
    options: Option<crate::images_models::CreateImagesModelsOptions>,
) -> crate::images_models::ImagesModels {
    let models = crate::images_models::create_images_models(options);
    for provider in builtin_images_providers() {
        models.set_provider(provider);
    }
    models
}
