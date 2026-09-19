//! Provider environment lookup, ported from
//! `packages/ai/src/utils/provider-env.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Resolves a provider configuration value from scoped overrides first, then
//! the process environment. Upstream carries a third leg — a Bun-sandbox
//! fallback reading `/proc/self/environ` — which has no Rust counterpart: a
//! Rust process always sees its environment, so the fallback's condition is
//! statically false.

use crate::types::ProviderEnv;

/// Resolve a provider env value from scoped overrides, then the process environment.
///
/// An empty override or environment value falls through to the next source, upstream's
/// falsy-string semantics, and an unset name is [`None`].
#[must_use]
pub fn get_provider_env_value(name: &str, env: Option<&ProviderEnv>) -> Option<String> {
    get_provider_env_value_with_process_env(name, env, |name| std::env::var(name).ok())
}

/// The lookup path with the process environment injectable, the seam hermetic
/// tests use to read env values from their own source.
#[must_use]
pub fn get_provider_env_value_with_process_env(
    name: &str,
    env: Option<&ProviderEnv>,
    process_env: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    env.and_then(|env| env.get(name))
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| process_env(name).filter(|value| !value.is_empty()))
}
