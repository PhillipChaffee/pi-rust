//! The `azure-openai-responses` provider factory, ported from
//! `packages/ai/src/providers/azure-openai-responses.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Azure OpenAI` provider, upstream's `azureOpenAIResponsesProvider()`.
    azure_openai_responses_provider,
    "azure-openai-responses",
    "Azure OpenAI",
    None,
    "Azure OpenAI API key",
    ["AZURE_OPENAI_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::azure_openai_responses()),
);
