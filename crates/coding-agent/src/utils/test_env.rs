//! The map-backed [`EnvLookup`] the in-src `#[cfg(test)]` suites share,
//! the same fixture shape `tests/common/mod.rs` provides the integration
//! binaries. Compiled only for the crate's own unit tests.

use crate::config::EnvLookup;

/// A map-backed [`EnvLookup`], the injected environment the `_with`
/// variants read in place of `process.env`.
#[must_use]
pub fn lookup_with(entries: &[(&str, &str)]) -> EnvLookup {
    let owned: Vec<(String, String)> = entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect();
    Box::new(move |key| {
        owned
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    })
}

/// An environment with no entries, the lookup for the fallback branches.
#[must_use]
pub fn empty_env() -> EnvLookup {
    Box::new(|_| None)
}
