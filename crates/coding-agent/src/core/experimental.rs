//! The experimental-feature gate, upstream's `src/core/experimental.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::config::{EnvLookup, default_env_lookup};

/// The opt-in variable, upstream's inline `PI_EXPERIMENTAL`.
const ENV_EXPERIMENTAL: &str = "PI_EXPERIMENTAL";

/// Whether experimental features are enabled, upstream's
/// `areExperimentalFeaturesEnabled`.
#[must_use]
pub fn are_experimental_features_enabled() -> bool {
    are_experimental_features_enabled_with(&default_env_lookup())
}

/// [`are_experimental_features_enabled`] over an injected environment
/// lookup, the test seam for the opt-in value.
#[must_use]
pub fn are_experimental_features_enabled_with(env: &EnvLookup) -> bool {
    env(ENV_EXPERIMENTAL).as_deref() == Some("1")
}

#[cfg(test)]
mod tests;
