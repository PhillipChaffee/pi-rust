//! The `huggingface` provider factory, ported from
//! `packages/ai/src/providers/huggingface.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Hugging Face` provider, upstream's `huggingfaceProvider()`.
    huggingface_provider,
    "huggingface",
    "Hugging Face",
    Some("https://router.huggingface.co/v1"),
    "Hugging Face token",
    ["HF_TOKEN"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);
