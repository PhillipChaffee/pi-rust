//! The provider module of `packages/ai/src/providers/`, ported at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`catalog`] holds the committed generated shards behind the builtin
//! registry ([`all`]), and one file per provider mirrors upstream's factory
//! files. The shared helpers port next to their providers: [`cloudflare_auth`]
//! and [`cloudflare_stream`], [`opencode_headers`], and [`radius_config`].
//! [`faux`] is the in-memory fake provider, and [`images`] registers the
//! builtin image-API providers.

pub mod all;
pub mod amazon_bedrock;
pub mod ant_ling;
pub mod anthropic;
pub mod azure_openai_responses;
pub mod baseten;
pub mod catalog;
pub mod cerebras;
pub mod cloudflare_ai_gateway;
pub mod cloudflare_auth;
pub mod cloudflare_stream;
pub mod cloudflare_workers_ai;
pub mod deepseek;
pub mod factory;
pub mod faux;
pub mod fireworks;
pub mod github_copilot;
pub mod google;
pub mod google_vertex;
pub mod groq;
pub mod huggingface;
pub mod images;
pub mod kimi_coding;
pub mod minimax;
pub mod minimax_cn;
pub mod mistral;
pub mod moonshotai;
pub mod moonshotai_cn;
pub mod nvidia;
pub mod openai;
pub mod openai_codex;
pub mod opencode;
pub mod opencode_go;
pub mod opencode_headers;
pub mod openrouter;
pub mod openrouter_images;
pub mod qwen_token_plan;
pub mod qwen_token_plan_cn;
pub mod qwen_token_plan_individual;
pub mod radius;
pub mod radius_config;
pub mod together;
pub mod vercel_ai_gateway;
pub mod xai;
pub mod xiaomi;
pub mod xiaomi_token_plan_ams;
pub mod xiaomi_token_plan_cn;
pub mod xiaomi_token_plan_sgp;
pub mod zai;
pub mod zai_coding_cn;
