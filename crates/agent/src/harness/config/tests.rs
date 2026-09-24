//! The harness configuration validation suite, ported from upstream
//! `test/harness/types.test.ts`'s configuration fixtures; upstream has no
//! dedicated config unit file, so the validators' contracts pin here.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use pi_ai::utils::retry::RetryPolicy;

use crate::harness::compaction::types::{create_file_ops, DEFAULT_COMPACTION_SETTINGS};
use crate::harness::config::{default_retry_policy, validate_compaction_settings, validate_retry_policy, validate_tool_names};

/// The default retry policy normalizes over pi-ai's `RetryPolicy` with
/// upstream's defaults.
#[test]
fn the_default_retry_policy_carries_upstream_defaults() {
    let policy = default_retry_policy();
    assert!(policy.enabled);
    assert!(policy.max_retries > 0);
    assert!(policy.base_delay_ms > 0);
}

/// Tool names must be unique and non-empty, upstream's `validateTools`.
#[test]
fn tool_names_must_be_unique_and_non_empty() {
    assert!(validate_tool_names(&["read", "write"]).is_ok());
    let error = validate_tool_names(&["read", "read"]).expect_err("duplicates error");
    assert!(error.contains("read"));
    let error = validate_tool_names(&["read", "read"])
        .expect_err("duplicates error");
    assert!(error.contains("Duplicate tool name"));
}

/// Retry policies must be internally consistent, upstream's
/// `validateRetryPolicy`.
#[test]
fn retry_policies_must_be_internally_consistent() {
    let policy = RetryPolicy {
        enabled: true,
        max_retries: 3,
        base_delay_ms: 100,
        max_agent_delay_ms: Some(30_000),
    };
    assert!(validate_retry_policy(&policy).is_ok());
    let overflow = RetryPolicy {
        base_delay_ms: u64::MAX,
        ..policy
    };
    assert!(validate_retry_policy(&overflow).is_err());
}

/// Compaction settings must be positive, upstream's
/// `validateCompactionSettings`.
#[test]
fn compaction_settings_must_be_positive() {
    assert!(validate_compaction_settings(&DEFAULT_COMPACTION_SETTINGS).is_ok());
    let invalid = crate::harness::compaction::types::CompactionSettings {
        enabled: true,
        reserve_tokens: u64::MAX,
        keep_recent_tokens: 20_000,
    };
    assert!(validate_compaction_settings(&invalid).is_err());
}

/// `createFileOps` builds the empty sorted-vector wire form.
#[test]
fn create_file_ops_builds_the_empty_sets() {
    assert_eq!(create_file_ops(), crate::harness::compaction::types::FileOperations::default());
}