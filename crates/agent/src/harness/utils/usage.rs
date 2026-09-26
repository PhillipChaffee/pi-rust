//! Usage arithmetic for the harness's aggregate stats, ported from upstream
//! `src/harness/utils/usage.ts`.

use pi_ai::types::{Usage, UsageCost};

/// The zero usage row, upstream's `emptyUsage()`.
#[must_use]
pub fn empty_usage() -> Usage {
    Usage::default()
}

/// Sums two usage rows, upstream's `addUsage`: the optional `cacheWrite1h`
/// and `reasoning` components appear only when either side carries them.
#[must_use]
pub fn add_usage(left: Usage, right: Usage) -> Usage {
    Usage {
        input: left.input + right.input,
        output: left.output + right.output,
        cache_read: left.cache_read + right.cache_read,
        cache_write: left.cache_write + right.cache_write,
        cache_write_1h: optional_sum(left.cache_write_1h, right.cache_write_1h),
        reasoning: optional_sum(left.reasoning, right.reasoning),
        total_tokens: left.total_tokens + right.total_tokens,
        cost: UsageCost {
            input: left.cost.input + right.cost.input,
            output: left.cost.output + right.cost.output,
            cache_read: left.cost.cache_read + right.cost.cache_read,
            cache_write: left.cost.cache_write + right.cost.cache_write,
            total: left.cost.total + right.cost.total,
        },
    }
}

fn optional_sum(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    (left.is_some() || right.is_some())
        .then(|| left.unwrap_or_default() + right.unwrap_or_default())
}
