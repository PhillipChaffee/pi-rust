//! The cache-stats suite, upstream `test/cache-stats.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::HashMap;

use pi_ai::types::{Api, AssistantMessage, KnownApi, ProviderId, StopReason, Usage, UsageCost};

use pi_coding_agent::cache_stats::{
    CacheMiss, ModelPriceSource, collect_cache_misses, compute_cache_waste, detect_cache_miss,
};
use pi_coding_agent::session_manager::entries::{MessageEntry, SessionEntry, SessionEntryBase};

/// The `$/million tokens` lookup, upstream's `models` fixture: used as the
/// cache-read price fallback on full-miss turns.
struct Models;

impl ModelPriceSource for Models {
    fn cache_read_price(&self, _provider: &str, _model_id: &str) -> Option<f64> {
        Some(0.3)
    }
}

fn assistant(options: AssistantOptions) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("test".to_owned()),
        model: options.model.unwrap_or_else(|| "test-model".to_owned()),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage {
            input: options.input.unwrap_or(0),
            output: 10,
            cache_read: options.cache_read.unwrap_or(0),
            cache_write: options.cache_write.unwrap_or(0),
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 0,
            cost: UsageCost {
                input: options.cost.input.unwrap_or(0.0),
                output: options.cost.output.unwrap_or(0.0),
                cache_read: options.cost.cache_read.unwrap_or(0.0),
                cache_write: options.cost.cache_write.unwrap_or(0.0),
                total: options.cost.total.unwrap_or(0.0),
            },
        },
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: options.timestamp.unwrap_or(0),
    }
}

struct AssistantOptions {
    input: Option<u64>,
    cache_read: Option<u64>,
    cache_write: Option<u64>,
    cost: PartialCost,
    model: Option<String>,
    timestamp: Option<i64>,
}

#[derive(Default)]
struct PartialCost {
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
    total: Option<f64>,
}

impl AssistantOptions {
    fn new() -> Self {
        Self {
            input: None,
            cache_read: None,
            cache_write: None,
            cost: PartialCost::default(),
            model: None,
            timestamp: None,
        }
    }
}

fn entry(message: AssistantMessage) -> SessionEntry {
    SessionEntry::Message(MessageEntry {
        base: SessionEntryBase {
            id: Some("x".to_owned()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        message: Some(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::Assistant(message),
        )),
        extras: serde_json::Map::new(),
    })
}

fn compaction_reset() -> SessionEntry {
    SessionEntry::Compaction(pi_coding_agent::session_manager::entries::CompactionEntry {
        base: SessionEntryBase {
            id: Some("c".to_owned()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        summary: String::new(),
        first_kept_entry_id: None,
        tokens_before: 0,
        details: None,
        usage: None,
        from_hook: None,
        extras: serde_json::Map::new(),
    })
}

fn options(
    input: Option<u64>,
    cache_read: Option<u64>,
    cache_write: Option<u64>,
    cost: PartialCost,
) -> AssistantOptions {
    let mut o = AssistantOptions::new();
    o.input = input;
    o.cache_read = cache_read;
    o.cache_write = cache_write;
    o.cost = cost;
    o
}

// Turn 1: fresh 100k cache write at $3.75/M.
fn turn1() -> AssistantMessage {
    let mut o = options(
        None,
        None,
        Some(100_000),
        PartialCost {
            cache_write: Some(0.375),
            ..Default::default()
        },
    );
    o.timestamp = Some(0);
    assistant(o)
}

// Turn 2: healthy, everything read back at $0.30/M.
fn turn2() -> AssistantMessage {
    let mut o = options(
        None,
        Some(100_000),
        Some(5_000),
        PartialCost {
            cache_read: Some(0.03),
            cache_write: Some(0.019),
            ..Default::default()
        },
    );
    o.timestamp = Some(60_000);
    assistant(o)
}

#[test]
fn accumulates_missed_tokens_and_cost_across_turns() {
    // Turn 3: full miss, previous 105k prompt re-billed at $3.75/M write.
    let mut o = options(
        None,
        None,
        Some(110_000),
        PartialCost {
            cache_write: Some(0.4125),
            ..Default::default()
        },
    );
    o.timestamp = Some(120_000);
    let turn3 = assistant(o);
    let totals = compute_cache_waste(&[entry(turn1()), entry(turn2()), entry(turn3)], &Models);
    assert_eq!(totals.missed_tokens, 105_000);
    // 105k at ($3.75 - $0.30)/M.
    assert!((totals.missed_cost - 0.36225).abs() < 1e-5);
}

#[test]
fn counts_nothing_for_healthy_sessions() {
    let totals = compute_cache_waste(&[entry(turn1()), entry(turn2())], &Models);
    assert_eq!(totals.missed_tokens, 0);
    assert!(totals.missed_cost.abs() < f64::EPSILON);
}

#[test]
fn skips_the_turn_after_a_compaction_reset() {
    let o = options(
        None,
        None,
        Some(20_000),
        PartialCost {
            cache_write: Some(0.075),
            ..Default::default()
        },
    );
    let after_reset = assistant(o);
    let totals = compute_cache_waste(
        &[entry(turn1()), compaction_reset(), entry(after_reset)],
        &Models,
    );
    assert_eq!(totals.missed_tokens, 0);
}

#[test]
fn counts_misses_caused_by_model_switches() {
    let mut o = options(
        None,
        None,
        Some(100_000),
        PartialCost {
            cache_write: Some(0.375),
            ..Default::default()
        },
    );
    o.model = Some("other-model".to_owned());
    let other_model = assistant(o);
    let totals = compute_cache_waste(&[entry(turn1()), entry(other_model)], &Models);
    assert_eq!(totals.missed_tokens, 100_000);
    assert_eq!(totals.miss_count, 1);
}

#[test]
fn skips_providers_that_report_no_cache_activity() {
    let a = assistant(options(Some(100_000), None, None, PartialCost::default()));
    let b = assistant(options(Some(110_000), None, None, PartialCost::default()));
    let totals = compute_cache_waste(&[entry(a), entry(b)], &Models);
    assert_eq!(totals.missed_tokens, 0);
}

#[test]
fn maps_counted_misses_to_their_assistant_messages() {
    let mut o = options(
        None,
        None,
        Some(110_000),
        PartialCost {
            cache_write: Some(0.4125),
            ..Default::default()
        },
    );
    o.timestamp = Some(120_000);
    let miss_turn = assistant(o);
    let entries = [entry(turn1()), entry(turn2()), entry(miss_turn)];
    let misses: HashMap<usize, CacheMiss> = collect_cache_misses(&entries, &Models);
    assert_eq!(misses.len(), 1);
    assert_eq!(misses.get(&2).map(|miss| miss.missed_tokens), Some(105_000));
}

#[test]
fn detects_a_miss_on_a_just_completed_message_with_idle_time() {
    let mut o = options(
        None,
        None,
        Some(110_000),
        PartialCost {
            cache_write: Some(0.4125),
            ..Default::default()
        },
    );
    o.timestamp = Some(600_000);
    let miss_message = assistant(o);
    let miss = detect_cache_miss(&[entry(turn1()), entry(turn2())], &miss_message, &Models)
        .expect("miss detected");
    assert_eq!(miss.missed_tokens, 105_000);
    assert!((miss.missed_cost - 0.36225).abs() < 1e-5);
    // 600s - 60s since the previous request.
    assert_eq!(miss.idle_ms, 540_000);
    assert!(!miss.model_changed);
}

#[test]
fn flags_model_switches_on_detected_misses() {
    let mut o = options(
        None,
        None,
        Some(110_000),
        PartialCost {
            cache_write: Some(0.4125),
            ..Default::default()
        },
    );
    o.model = Some("other-model".to_owned());
    o.timestamp = Some(120_000);
    let other_model = assistant(o);
    let miss = detect_cache_miss(&[entry(turn1()), entry(turn2())], &other_model, &Models)
        .expect("miss detected");
    assert_eq!(miss.missed_tokens, 105_000);
    assert!(miss.model_changed);
}

#[test]
fn returns_none_for_healthy_turns() {
    let mut o = options(
        None,
        Some(105_000),
        Some(2_000),
        PartialCost {
            cache_read: Some(0.0315),
            cache_write: Some(0.0075),
            ..Default::default()
        },
    );
    o.timestamp = Some(120_000);
    let healthy = assistant(o);
    let miss = detect_cache_miss(&[entry(turn1()), entry(turn2())], &healthy, &Models);
    assert!(miss.is_none());
}

#[test]
fn returns_none_for_the_first_turn_of_a_session() {
    assert!(detect_cache_miss(&[], &turn1(), &Models).is_none());
}
