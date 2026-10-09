//! Session usage totals, upstream `src/core/usage-totals.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{Message, Usage};

use crate::session_manager::entries::SessionEntry;

/// The running usage totals across a session, upstream's `UsageTotals`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UsageTotals {
    /// Prompt tokens, including cache reads and writes, serialized as
    /// `input`.
    pub input: u64,
    /// Completion tokens, serialized as `output`.
    pub output: u64,
    /// Prompt tokens served from cache, serialized as `cacheRead`.
    pub cache_read: u64,
    /// Prompt tokens written to cache, serialized as `cacheWrite`.
    pub cache_write: u64,
    /// Total cost in dollars.
    pub cost: f64,
}

/// Create zeroed usage totals, upstream's `createUsageTotals`.
#[must_use]
pub fn create_usage_totals() -> UsageTotals {
    UsageTotals::default()
}

/// Add one usage record into the totals, upstream's `addUsageToTotals`.
pub fn add_usage_to_totals(totals: &mut UsageTotals, usage: &Usage) {
    totals.input += usage.input;
    totals.output += usage.output;
    totals.cache_read += usage.cache_read;
    totals.cache_write += usage.cache_write;
    totals.cost += usage.cost.total;
}

/// One bucket of the cost breakdown, upstream's `UsageCostBreakdownEntry`.
#[derive(Debug, Clone, PartialEq)]
pub struct UsageCostBreakdownEntry {
    /// The bucket key: `provider/model` for attributable assistant usage,
    /// `Tools/summaries` for everything else.
    pub key: String,
    /// The bucket's total cost.
    pub cost: f64,
    /// The bucket's total token count (input + output + cacheRead +
    /// cacheWrite).
    pub tokens: u64,
}

/// Group attributable assistant usage by model and all other usage into a
/// separate bucket, upstream's `getUsageCostBreakdown`.
///
/// The result drops empty buckets and sorts by cost, most expensive first
/// (upstream's `b.cost - a.cost` compare).
#[must_use]
pub fn get_usage_cost_breakdown(entries: &[SessionEntry]) -> Vec<UsageCostBreakdownEntry> {
    let mut totals_by_key: indexmap::IndexMap<String, UsageTotals> = indexmap::IndexMap::new();

    for entry in entries {
        let mut key: Option<String> = None;
        let mut usage: Option<&Usage> = None;
        if let SessionEntry::Message(message_entry) = entry {
            match &message_entry.message {
                Some(AgentMessage::Standard(Message::Assistant(message))) => {
                    key = Some(format!(
                        "{}/{}",
                        message.provider.0,
                        message.response_model.as_deref().unwrap_or(&message.model)
                    ));
                    usage = Some(&message.usage);
                }
                Some(AgentMessage::Standard(Message::ToolResult(result))) => {
                    // Tool-result usage rides the entry when reported.
                    // `usage` on ToolResultMessage: absent when the tool ran
                    // free.
                    if let Some(tool_usage) = &result.usage {
                        key = Some("Tools/summaries".to_owned());
                        usage = Some(tool_usage);
                    }
                }
                _ => {}
            }
        }
        if key.is_none()
            && let Some(entry_usage) = summary_entry_usage(entry)
        {
            key = Some("Tools/summaries".to_owned());
            usage = Some(entry_usage);
        }
        let (Some(key), Some(usage)) = (key, usage) else {
            continue;
        };

        let totals = totals_by_key.entry(key).or_insert_with(create_usage_totals);
        add_usage_to_totals(totals, usage);
    }

    let mut breakdown: Vec<UsageCostBreakdownEntry> = totals_by_key
        .into_iter()
        .map(|(key, totals)| UsageCostBreakdownEntry {
            tokens: totals.input + totals.output + totals.cache_read + totals.cache_write,
            cost: totals.cost,
            key,
        })
        .filter(|entry| entry.cost > 0.0 || entry.tokens > 0)
        .collect();
    breakdown.sort_by(|a, b| {
        b.cost
            .partial_cmp(&a.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    breakdown
}

/// The usage a compaction or branch-summary entry carries, upstream's
/// `entry.usage` check.
const fn summary_entry_usage(entry: &SessionEntry) -> Option<&Usage> {
    match entry {
        SessionEntry::Compaction(compaction) => compaction.usage.as_ref(),
        SessionEntry::BranchSummary(branch) => branch.usage.as_ref(),
        _ => None,
    }
}
