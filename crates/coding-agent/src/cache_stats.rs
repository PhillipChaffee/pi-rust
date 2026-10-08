//! Cache-waste accounting, upstream `src/core/cache-stats.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::HashMap;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::Message;

use crate::session_manager::entries::SessionEntry;

/// Prompt-cache TTL: idle gaps longer than this are worth mentioning as the
/// likely cause of a miss. Anthropic's default cache TTL is 5 minutes.
pub const CACHE_TTL_MS: i64 = 5 * 60 * 1000;

/// Per-turn misses at or below this are cache breakpoint granularity noise,
/// upstream's `NOISE_FLOOR_TOKENS`.
const NOISE_FLOOR_TOKENS: u64 = 1024;

/// A counted cache miss on a single assistant message, upstream's
/// `CacheMiss`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheMiss {
    /// Prompt tokens that were in the previous turn's prompt but not read
    /// from cache, serialized as `missedTokens`.
    pub missed_tokens: u64,
    /// Extra dollars paid vs. a full cache hit; 0 when pricing is unknown,
    /// serialized as `missedCost`.
    pub missed_cost: f64,
    /// Milliseconds since the previous request (which last refreshed the
    /// cache), serialized as `idleMs`.
    pub idle_ms: i64,
    /// True when the model changed relative to the previous request,
    /// serialized as `modelChanged`.
    pub model_changed: bool,
}

/// The cumulative cache waste across a session, upstream's
/// `CacheWasteTotals`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CacheWasteTotals {
    /// Prompt tokens re-billed that should have been cache reads.
    pub missed_tokens: u64,
    /// Dollars paid over the cache-read rate.
    pub missed_cost: f64,
    /// Number of counted misses (turns above the noise floor).
    pub miss_count: u64,
}

/// Minimal pricing lookup, upstream's `ModelPriceSource`, satisfied by
/// ModelRuntime. Costs are $/million tokens; the accessor restates
/// `getModel(provider, modelId).cost.cacheRead`.
pub trait ModelPriceSource {
    /// The model's cache-read price, `$`/million tokens.
    fn cache_read_price(&self, provider: &str, model_id: &str) -> Option<f64>;
}

/// The last request seen by the scan; everything in its prompt should be
/// cached, upstream's `PreviousRequest`.
#[derive(Debug, Clone)]
struct PreviousRequest {
    prompt_tokens: u64,
    model_key: String,
    timestamp: i64,
    /// Sticky: some earlier request in this scan segment reported cache
    /// activity. Distinguishes a total miss on a cache-read-only provider
    /// (OpenAI-style, writes unreported) from a provider that never reports
    /// caching at all.
    reported_cache: bool,
}

/// Compute the cache miss for one assistant message relative to the previous
/// request, upstream's `detectMiss`. Returns `None` when nothing is counted:
/// first turn, after a reset, no cache activity ever reported (provider
/// without cache support), or miss below the noise floor.
#[expect(
    clippy::cast_precision_loss,
    reason = "the per-token rates restate JS number division over token counts"
)]
fn detect_miss(
    prev: Option<&PreviousRequest>,
    message: &pi_ai::types::AssistantMessage,
    models: &dyn ModelPriceSource,
) -> Option<CacheMiss> {
    let usage = &message.usage;
    let prompt_tokens = usage.input + usage.cache_read + usage.cache_write;
    // A zero-cache turn only counts when cache activity was reported before:
    // on cache-read-only providers that is a total miss, while on providers
    // that never report caching it means nothing.
    let prev = prev?;
    if prompt_tokens == 0 || (usage.cache_read + usage.cache_write == 0 && !prev.reported_cache) {
        return None;
    }

    let missed_tokens = prev
        .prompt_tokens
        .min(prompt_tokens)
        .saturating_sub(usage.cache_read);
    if missed_tokens <= NOISE_FLOOR_TOKENS {
        return None;
    }

    // Extra cost = missed tokens billed at the actual paid rate
    // (input/cacheWrite, incl. write premium) instead of the cache-read
    // rate. Missed tokens can only land in the input or cacheWrite buckets,
    // so the paid rate comes straight from this message's own cost
    // breakdown.
    let paid_tokens = usage.input + usage.cache_write;
    let paid_per_token = if paid_tokens > 0 {
        (usage.cost.input + usage.cost.cache_write) / paid_tokens as f64
    } else {
        0.0
    };
    let read_per_token = if usage.cache_read > 0 {
        usage.cost.cache_read / usage.cache_read as f64
    } else {
        models
            .cache_read_price(&message.provider.0, &message.model)
            .unwrap_or(0.0)
            / 1_000_000.0
    };

    Some(CacheMiss {
        missed_tokens,
        missed_cost: missed_tokens as f64 * (paid_per_token - read_per_token).max(0.0),
        idle_ms: (message.timestamp - prev.timestamp).max(0),
        model_changed: format!("{}/{}", message.provider.0, message.model) != prev.model_key,
    })
}

/// The scan-state predecessor one counted message leaves behind, upstream's
/// `asPreviousRequest`.
fn as_previous_request(
    message: &pi_ai::types::AssistantMessage,
    reported_cache: bool,
) -> Option<PreviousRequest> {
    let usage = &message.usage;
    let prompt_tokens = usage.input + usage.cache_read + usage.cache_write;
    if prompt_tokens == 0 {
        return None;
    }
    Some(PreviousRequest {
        prompt_tokens,
        model_key: format!("{}/{}", message.provider.0, message.model),
        timestamp: message.timestamp,
        reported_cache: reported_cache || usage.cache_read + usage.cache_write > 0,
    })
}

/// The scan's outcome: the trailing request state, the cumulative totals, and
/// the counted misses keyed by the entries-index of the assistant message
/// that paid for them, upstream's `Map<AssistantMessage, CacheMiss>` restated
/// on entry indices (Rust clones have no reference identity).
struct Scan {
    prev: Option<PreviousRequest>,
    totals: CacheWasteTotals,
    misses: HashMap<usize, CacheMiss>,
}

fn scan(entries: &[SessionEntry], models: &dyn ModelPriceSource) -> Scan {
    let mut prev: Option<PreviousRequest> = None;
    let mut totals = CacheWasteTotals::default();
    let mut misses: HashMap<usize, CacheMiss> = HashMap::new();

    for (index, entry) in entries.iter().enumerate() {
        if matches!(
            entry,
            SessionEntry::Compaction(_) | SessionEntry::BranchSummary(_)
        ) {
            // The context legitimately changed; the next turn's prompt is new
            // content, not re-billed content. Model switches are NOT exempt:
            // they re-bill the full prompt and should be counted.
            prev = None;
            continue;
        }
        if let SessionEntry::Message(message_entry) = entry {
            let Some(AgentMessage::Standard(Message::Assistant(message))) = &message_entry.message
            else {
                continue;
            };
            if let Some(miss) = detect_miss(prev.as_ref(), message, models) {
                totals.missed_tokens += miss.missed_tokens;
                totals.missed_cost += miss.missed_cost;
                totals.miss_count += 1;
                misses.insert(index, miss);
            }
            prev = as_previous_request(message, prev.as_ref().is_some_and(|p| p.reported_cache))
                .or(prev);
        }
    }
    Scan {
        prev,
        totals,
        misses,
    }
}

/// Cumulative cache waste across a session: prompt tokens that should have
/// been cache reads (they were in the previous turn's prompt) but were
/// re-billed, upstream's `computeCacheWaste`.
#[must_use]
pub fn compute_cache_waste(
    entries: &[SessionEntry],
    models: &dyn ModelPriceSource,
) -> CacheWasteTotals {
    scan(entries, models).totals
}

/// All counted cache misses across a session, upstream's
/// `collectCacheMisses`.
///
/// Keyed by the entries-index of the assistant message that paid for them;
/// used to re-derive transcript notices when rebuilding the chat from entries
/// (resume, post-compaction rebuild).
#[must_use]
pub fn collect_cache_misses(
    entries: &[SessionEntry],
    models: &dyn ModelPriceSource,
) -> HashMap<usize, CacheMiss> {
    scan(entries, models).misses
}

/// Detect a cache miss on a just-completed assistant message, upstream's
/// `detectCacheMiss`. `entries` must not yet contain `message`
/// (`message_end` fires before persistence).
#[must_use]
pub fn detect_cache_miss(
    entries: &[SessionEntry],
    message: &pi_ai::types::AssistantMessage,
    models: &dyn ModelPriceSource,
) -> Option<CacheMiss> {
    let prev = scan(entries, models).prev;
    detect_miss(prev.as_ref(), message, models)
}
