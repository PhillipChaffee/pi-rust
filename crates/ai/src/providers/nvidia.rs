//! The `nvidia` provider factory, ported from
//! `packages/ai/src/providers/nvidia.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `NVIDIA` provider, upstream's `nvidiaProvider()`.
    nvidia_provider,
    "nvidia",
    "NVIDIA",
    Some("https://integrate.api.nvidia.com/v1"),
    "NVIDIA API key",
    ["NVIDIA_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);
