//! Shared fixtures for the coding-agent config suites.

#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]

use pi_coding_agent::config::EnvLookup;

/// A map-backed [`EnvLookup`], the injected environment the `_with`
/// variants read in place of `process.env`.
pub fn env_with(entries: &[(&str, &str)]) -> EnvLookup {
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
pub fn empty_env() -> EnvLookup {
    Box::new(|_| None)
}
