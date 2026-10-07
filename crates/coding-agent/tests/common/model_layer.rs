//! Shared fixtures for the model-layer suites (#121), upstream's
//! `model-runtime-test-utils.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The fixture module keeps the model/catalog helpers single-sourced: the
//! duplication gate's headroom is nearly spent, so every suite consumes these
//! instead of restating them. Suite-specific doubles (providers with scripted
//! auth, gate-backed credential stores) stay in their suites.

use std::sync::Arc;

use pi_ai::types::{Api, Context, Modality, Model, ModelCost, ModelCostRates, ProviderId};
use pi_coding_agent::auth_storage::{AuthStorage, AuthStorageData, InMemoryAuthStorageBackend};
use pi_coding_agent::model_registry::ModelRegistry;
use pi_coding_agent::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use pi_coding_agent::models_store::InMemoryCodingAgentModelsStore;

/// The zero-cost rates the fixture models carry, upstream's
/// `{ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }`.
#[must_use]
pub const fn zero_cost() -> ModelCost {
    ModelCost {
        rates: ModelCostRates {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        tiers: None,
    }
}

/// The dynamic-provider fixture model, upstream's shared `model(...)` helper:
/// an `openai-completions` model over an example base URL.
#[must_use]
pub fn model(provider: &str, id: &str) -> Model {
    Model {
        id: id.to_owned(),
        name: id.to_owned(),
        api: Api::from("openai-completions"),
        provider: ProviderId(provider.to_owned()),
        base_url: "https://example.test/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: zero_cost(),
        context_window: 1000,
        max_tokens: 100,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The no-message context upstream's `{ messages: [] }` spells.
#[must_use]
pub const fn empty_context() -> Context {
    Context {
        system_prompt: None,
        messages: Vec::new(),
        tools: None,
    }
}

/// An in-memory auth store seeded with raw credential JSON per provider id,
/// upstream's `AuthStorage.inMemory({...})`.
#[must_use]
pub fn in_memory_auth_storage(
    data: &AuthStorageData,
) -> Arc<AuthStorage<InMemoryAuthStorageBackend>> {
    Arc::new(AuthStorage::<InMemoryAuthStorageBackend>::in_memory(data))
}

/// The empty in-memory auth store, upstream's `AuthStorage.inMemory()`.
#[must_use]
pub fn empty_auth_storage() -> Arc<AuthStorage<InMemoryAuthStorageBackend>> {
    in_memory_auth_storage(&AuthStorageData::new())
}

/// The runtime over in-memory credentials and an optional models.json path,
/// upstream's `createModelRegistry`: the store keeps file-backed catalog
/// locks out of unit tests, and the network stays off.
///
/// # Panics
/// The runtime construction failure, which the fixture cannot satisfy.
pub async fn create_model_registry(
    credentials: Arc<dyn pi_ai::auth::credential_store::CredentialStore>,
    models_path: Option<&str>,
) -> ModelRegistry {
    ModelRegistry::new(
        ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(credentials),
            models_path: Some(models_path.map(str::to_owned)),
            models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
            allow_model_network: false,
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the fixture runtime constructs"),
    )
}

/// The runtime with models.json disabled entirely, upstream's
/// `createInMemoryModelRegistry`.
///
/// # Panics
/// The runtime construction failure, which the fixture cannot satisfy.
pub async fn create_in_memory_model_registry(
    credentials: Arc<dyn pi_ai::auth::credential_store::CredentialStore>,
) -> ModelRegistry {
    ModelRegistry::new(
        ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(credentials),
            models_path: Some(None),
            models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
            allow_model_network: false,
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the fixture runtime constructs"),
    )
}
