//! The harness retry and validation defaults, ported from upstream
//! `src/harness/config.ts`.
//!
//! The validators restatement of upstream's `Number.isSafeInteger` probes:
//! the port carries the counts as `u64`/`u32`, so finiteness and
//! non-negativity are structural and only the JavaScript
//! `MAX_SAFE_INTEGER` ceiling remains checkable — including upstream's
//! explicit rejection of `maxRetries === Number.MAX_SAFE_INTEGER`, which is
//! vacuous over `u32` and recorded as such.

use std::collections::HashSet;

use pi_ai::utils::retry::{DEFAULT_MAX_AGENT_RETRY_DELAY_MS, RetryPolicy};

use crate::harness::compaction::types::CompactionSettings;

/// JavaScript's `Number.MAX_SAFE_INTEGER`, the ceiling the validators
/// restate.
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// The harness default retry policy, upstream's `DEFAULT_RETRY_POLICY`.
#[must_use]
pub const fn default_retry_policy() -> RetryPolicy {
    RetryPolicy {
        enabled: true,
        max_retries: 3,
        base_delay_ms: 1_000,
        max_agent_delay_ms: Some(DEFAULT_MAX_AGENT_RETRY_DELAY_MS),
    }
}

/// Rejects duplicate tool names, upstream's `validateToolNames`.
///
/// # Errors
/// A `TypeError`-shaped message when two tools share a name.
pub fn validate_tool_names(tools: &[impl AsRef<str>]) -> Result<(), String> {
    let mut names = HashSet::new();
    for tool in tools {
        let name = tool.as_ref();
        if !names.insert(name.to_owned()) {
            return Err(format!(
                "Duplicate tool name: {}",
                serde_json::to_string(name).unwrap_or_else(|_| format!("\"{name}\""))
            ));
        }
    }
    Ok(())
}

/// Validates the retry policy's values, upstream's `validateRetryPolicy`.
///
/// # Errors
/// A `RangeError`-shaped message when any value exceeds the JavaScript
/// safe-integer ceiling.
pub fn validate_retry_policy(policy: &RetryPolicy) -> Result<(), String> {
    if !within_safe_integer(policy.max_retries as u64)
        || !within_safe_integer(policy.base_delay_ms)
        || policy
            .max_agent_delay_ms
            .is_some_and(|max_agent_delay_ms| !within_safe_integer(max_agent_delay_ms))
    {
        return Err("Retry policy values must be finite non-negative safe integers".to_owned());
    }
    Ok(())
}

fn within_safe_integer(value: u64) -> bool {
    value <= MAX_SAFE_INTEGER
}

/// Validates the compaction settings' token counts, upstream's
/// `validateCompactionSettings`.
///
/// # Errors
/// A `RangeError`-shaped message when either token count exceeds the
/// JavaScript safe-integer ceiling.
pub fn validate_compaction_settings(settings: &CompactionSettings) -> Result<(), String> {
    if !within_safe_integer(settings.reserve_tokens)
        || !within_safe_integer(settings.keep_recent_tokens)
    {
        return Err("Compaction token counts must be finite non-negative safe integers".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests;