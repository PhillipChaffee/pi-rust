//! The usage-totals, provider-attribution, and telemetry boundary suite:
//! branches the coding-agent 1:1 suites do not reach (upstream exercises
//! them through the interactive-mode and agent-session suites, which ride
//! their own tickets), pinned against upstream at
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::collections::BTreeMap;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{
    Api, AssistantMessage, KnownApi, Message, ProviderId, StopReason, TextContent, ToolResultBlock,
    ToolResultMessage, Usage, UsageCost,
};

use pi_coding_agent::provider_attribution::merge_provider_attribution_headers;
use pi_coding_agent::session_manager::entries::{
    BranchSummaryEntry, CompactionEntry, MessageEntry, SessionEntry, SessionEntryBase,
};
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, Settings, SettingsManager, SettingsManagerCreateOptions,
};
use pi_coding_agent::telemetry::{is_install_telemetry_enabled, is_install_telemetry_enabled_with};
use pi_coding_agent::usage_totals::{
    UsageCostBreakdownEntry, UsageTotals, add_usage_to_totals, create_usage_totals,
    get_usage_cost_breakdown,
};

const fn usage(input: u64, output: u64, cost_total: f64) -> Usage {
    Usage {
        input,
        output,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input + output,
        cost: UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: cost_total,
        },
    }
}

fn assistant_entry(
    provider: &str,
    model: &str,
    response_model: Option<&str>,
    cost: f64,
) -> SessionEntry {
    SessionEntry::Message(MessageEntry {
        base: SessionEntryBase::default(),
        message: Some(AgentMessage::Standard(Message::Assistant(
            AssistantMessage {
                content: Vec::new(),
                api: Api::from(KnownApi::AnthropicMessages),
                provider: ProviderId(provider.to_owned()),
                model: model.to_owned(),
                response_model: response_model.map(str::to_owned),
                response_id: None,
                provider_thinking_level: None,
                diagnostics: None,
                usage: usage(10, 10, cost),
                stop_reason: StopReason::Stop,
                deferred: None,
                error_message: None,
                raw_stop_reason: None,
                end_turn: None,
                timestamp: 0,
            },
        ))),
        extras: serde_json::Map::new(),
    })
}

fn tool_result_entry(cost: f64) -> SessionEntry {
    SessionEntry::Message(MessageEntry {
        base: SessionEntryBase::default(),
        message: Some(AgentMessage::Standard(Message::ToolResult(
            ToolResultMessage {
                tool_call_id: "tc".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![ToolResultBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                })],
                details: None,
                usage: Some(usage(5, 0, cost)),
                added_tool_names: None,
                is_error: false,
                timestamp: 0,
            },
        ))),
        extras: serde_json::Map::new(),
    })
}

fn compaction_entry(cost: f64) -> SessionEntry {
    SessionEntry::Compaction(CompactionEntry {
        base: SessionEntryBase::default(),
        summary: String::new(),
        first_kept_entry_id: None,
        tokens_before: 0,
        details: None,
        usage: Some(usage(20, 0, cost)),
        from_hook: None,
        extras: serde_json::Map::new(),
    })
}

fn branch_summary_entry(cost: f64) -> SessionEntry {
    SessionEntry::BranchSummary(BranchSummaryEntry {
        base: SessionEntryBase::default(),
        from_id: "x".to_owned(),
        summary: String::new(),
        details: None,
        usage: Some(usage(30, 0, cost)),
        from_hook: None,
        extras: serde_json::Map::new(),
    })
}

#[test]
fn add_usage_to_totals_sums_every_bucket() {
    let mut totals = create_usage_totals();
    add_usage_to_totals(
        &mut totals,
        &Usage {
            input: 10,
            output: 20,
            cache_read: 30,
            cache_write: 40,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 100,
            cost: UsageCost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 1.5,
            },
        },
    );
    assert_eq!(
        totals,
        UsageTotals {
            input: 10,
            output: 20,
            cache_read: 30,
            cache_write: 40,
            cost: 1.5,
        }
    );
}

#[test]
fn breakdown_groups_assistant_usage_by_provider_and_response_model() {
    let entries = vec![
        assistant_entry("anthropic", "claude-4", Some("claude-4-20250514"), 1.0),
        assistant_entry("anthropic", "claude-4", Some("claude-4-20250514"), 0.5),
        assistant_entry("openai", "gpt-5", None, 2.0),
        tool_result_entry(0.25),
    ];
    let breakdown = get_usage_cost_breakdown(&entries);
    assert_eq!(
        breakdown,
        vec![
            UsageCostBreakdownEntry {
                key: "openai/gpt-5".to_owned(),
                cost: 2.0,
                tokens: 20,
            },
            UsageCostBreakdownEntry {
                key: "anthropic/claude-4-20250514".to_owned(),
                cost: 1.5,
                tokens: 40,
            },
            UsageCostBreakdownEntry {
                key: "Tools/summaries".to_owned(),
                cost: 0.25,
                tokens: 5,
            },
        ]
    );
}

#[test]
fn breakdown_falls_back_to_the_requested_model_without_a_response_model() {
    let entries = vec![assistant_entry("anthropic", "claude-4", None, 1.0)];
    let breakdown = get_usage_cost_breakdown(&entries);
    assert_eq!(breakdown[0].key, "anthropic/claude-4");
}

#[test]
fn breakdown_buckets_compaction_and_branch_usage_into_tools_summaries() {
    let entries = vec![compaction_entry(0.4), branch_summary_entry(0.6)];
    let breakdown = get_usage_cost_breakdown(&entries);
    assert_eq!(breakdown.len(), 1);
    assert_eq!(breakdown[0].key, "Tools/summaries");
    assert!((breakdown[0].cost - 1.0).abs() < f64::EPSILON);
    assert_eq!(breakdown[0].tokens, 50);
}

#[test]
fn breakdown_drops_empty_buckets_and_keeps_cost_only_ones() {
    // A tool result with usage but zero tokens still reports (cost > 0
    // gate); an assistant message with no usage at all contributes nothing.
    let entries = vec![tool_result_entry(0.1)];
    let breakdown = get_usage_cost_breakdown(&entries);
    assert_eq!(breakdown.len(), 1);

    let empty = get_usage_cost_breakdown(&[]);
    assert!(empty.is_empty());
}

#[test]
fn breakdown_sorts_stably_by_cost() {
    let entries = vec![
        assistant_entry("a", "m1", None, 0.5),
        assistant_entry("b", "m2", None, 0.5),
        assistant_entry("c", "m3", None, 3.0),
    ];
    let breakdown = get_usage_cost_breakdown(&entries);
    assert_eq!(breakdown[0].key, "c/m3");
    // The two 0.5 buckets keep their insertion order, upstream's stable sort.
    assert_eq!(breakdown[1].key, "a/m1");
    assert_eq!(breakdown[2].key, "b/m2");
}

// ============================================================================
// Telemetry gating
// ============================================================================

fn settings_with(enable_install_telemetry: bool) -> SettingsManager<InMemorySettingsStorage> {
    let mut manager =
        SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default());
    let mut overrides = Settings::new();
    overrides.insert(
        "enableInstallTelemetry".to_owned(),
        serde_json::Value::Bool(enable_install_telemetry),
    );
    manager.apply_overrides(&overrides);
    manager
}

#[test]
fn telemetry_env_override_wins_over_the_setting() {
    let manager = settings_with(false);
    assert!(is_install_telemetry_enabled_with(&manager, Some("1")));
    assert!(is_install_telemetry_enabled_with(&manager, Some("TRUE")));
    assert!(is_install_telemetry_enabled_with(&manager, Some("yes")));
    assert!(!is_install_telemetry_enabled_with(&manager, Some("0")));
    assert!(!is_install_telemetry_enabled_with(&manager, Some("no")));
    assert!(!is_install_telemetry_enabled_with(&manager, Some("")));
    // Absent env falls back to the setting.
    assert!(!is_install_telemetry_enabled_with(&manager, None));
}

#[test]
fn telemetry_falls_back_to_the_setting_when_the_env_is_absent() {
    let enabled = settings_with(true);
    assert!(is_install_telemetry_enabled_with(&enabled, None));

    let live_lookup = is_install_telemetry_enabled(&enabled, &common::empty_env());
    assert!(live_lookup);
}

// ============================================================================
// Provider attribution
// ============================================================================

fn model(provider: &str, base_url: &str) -> pi_ai::types::Model {
    pi_ai::types::Model {
        id: "m".to_owned(),
        name: "m".to_owned(),
        api: Api::from(KnownApi::OpenaiCompletions),
        provider: ProviderId(provider.to_owned()),
        base_url: base_url.to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 200_000,
        max_tokens: 8192,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn attribution(
    model: &pi_ai::types::Model,
    enabled: bool,
) -> Option<BTreeMap<String, Option<String>>> {
    let settings = settings_with(enabled);
    merge_provider_attribution_headers(model, &settings, &common::empty_env(), None, &[])
}

#[test]
fn attribution_headers_gate_on_install_telemetry() {
    let openrouter = model("openrouter", "https://openrouter.ai/api/v1");
    assert!(attribution(&openrouter, false).is_none());
    let headers = attribution(&openrouter, true).expect("headers");
    assert_eq!(
        headers.get("HTTP-Referer").map(Option::as_deref),
        Some(Some("https://pi.dev"))
    );
    assert_eq!(
        headers.get("X-OpenRouter-Title").map(Option::as_deref),
        Some(Some("pi"))
    );
    assert_eq!(
        headers.get("X-OpenRouter-Categories").map(Option::as_deref),
        Some(Some("cli-agent"))
    );
}

#[test]
fn openrouter_attribution_matches_by_provider_id_or_base_url_host() {
    // A non-openrouter provider id still matches on the base URL host.
    let by_host = attribution(&model("custom", "https://openrouter.ai/api/v1"), true);
    assert!(by_host.is_some());
    // An unparseable base URL matches nothing.
    let unparseable = attribution(&model("custom", "not a url"), true);
    assert!(unparseable.is_none());
}

#[test]
fn nvidia_and_cloudflare_attribution_headers() {
    let nvidia = attribution(
        &model("nvidia", "https://integrate.api.nvidia.com/v1"),
        true,
    )
    .expect("nvidia headers");
    assert_eq!(
        nvidia.get("X-BILLING-INVOKE-ORIGIN").map(Option::as_deref),
        Some(Some("Pi"))
    );

    let cloudflare = attribution(
        &model(
            "cloudflare-workers-ai",
            "https://api.cloudflare.com/client/v4",
        ),
        true,
    )
    .expect("cloudflare headers");
    assert_eq!(
        cloudflare.get("User-Agent").map(Option::as_deref),
        Some(Some("pi-coding-agent"))
    );

    // The gateway host matches too.
    let gateway = attribution(
        &model("custom", "https://gateway.ai.cloudflare.com/v1"),
        true,
    );
    assert!(gateway.is_some());
}

#[test]
fn session_headers_only_apply_to_opencode() {
    let settings = settings_with(false);
    let opencode = model("opencode", "https://api.opencode.ai/v1");
    let headers = merge_provider_attribution_headers(
        &opencode,
        &settings,
        &common::empty_env(),
        Some("session-1"),
        &[],
    )
    .expect("session headers");
    assert_eq!(
        headers.get("x-opencode-session").map(Option::as_deref),
        Some(Some("session-1"))
    );
    assert_eq!(
        headers.get("x-opencode-client").map(Option::as_deref),
        Some(Some("pi"))
    );

    // The opencode-go provider id and the opencode host both qualify.
    let go = merge_provider_attribution_headers(
        &model("opencode-go", "https://other.example"),
        &settings,
        &common::empty_env(),
        Some("session-2"),
        &[],
    );
    assert!(go.is_some());
    let by_host = merge_provider_attribution_headers(
        &model("custom", "https://opencode.ai/anything"),
        &settings,
        &common::empty_env(),
        Some("session-3"),
        &[],
    );
    assert!(by_host.is_some());

    // Non-opencode models with a session id get nothing.
    let other = merge_provider_attribution_headers(
        &model("anthropic", "https://api.anthropic.com"),
        &settings,
        &common::empty_env(),
        Some("session-4"),
        &[],
    );
    assert!(other.is_none());
}

#[test]
fn merge_provider_attribution_headers_later_sources_win_and_none_when_empty() {
    let settings = settings_with(false);
    let model = model("anthropic", "https://api.anthropic.com");

    // No session, telemetry off, no sources: None.
    assert!(
        merge_provider_attribution_headers(&model, &settings, &common::empty_env(), None, &[])
            .is_none()
    );

    let mut source = BTreeMap::new();
    source.insert("X-Custom".to_owned(), Some("from-caller".to_owned()));
    let mut override_source = BTreeMap::new();
    override_source.insert("X-Custom".to_owned(), Some("from-override".to_owned()));

    let merged = merge_provider_attribution_headers(
        &model,
        &settings,
        &common::empty_env(),
        None,
        &[&source, &override_source],
    )
    .expect("merged");
    assert_eq!(
        merged.get("X-Custom").map(Option::as_deref),
        Some(Some("from-override"))
    );
}

// ============================================================================
// The breakdown's skip arms
// ============================================================================

fn tool_result_without_usage() -> SessionEntry {
    SessionEntry::Message(MessageEntry {
        base: SessionEntryBase::default(),
        message: Some(AgentMessage::Standard(Message::ToolResult(
            ToolResultMessage {
                tool_call_id: "tc".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![ToolResultBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: 0,
            },
        ))),
        extras: serde_json::Map::new(),
    })
}

fn assistant_entry_zero_usage(provider: &str, model: &str) -> SessionEntry {
    SessionEntry::Message(MessageEntry {
        base: SessionEntryBase::default(),
        message: Some(AgentMessage::Standard(Message::Assistant(
            AssistantMessage {
                content: Vec::new(),
                api: Api::from(KnownApi::AnthropicMessages),
                provider: ProviderId(provider.to_owned()),
                model: model.to_owned(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                diagnostics: None,
                usage: usage(0, 0, 0.0),
                stop_reason: StopReason::Stop,
                deferred: None,
                error_message: None,
                raw_stop_reason: None,
                end_turn: None,
                timestamp: 0,
            },
        ))),
        extras: serde_json::Map::new(),
    })
}

fn user_entry() -> SessionEntry {
    SessionEntry::Message(MessageEntry {
        base: SessionEntryBase::default(),
        message: Some(AgentMessage::Standard(Message::User(
            pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("hi".to_owned()),
                timestamp: 0,
            },
        ))),
        extras: serde_json::Map::new(),
    })
}

#[test]
fn breakdown_skips_tool_results_without_usage_and_zero_buckets() {
    let entries = vec![
        tool_result_without_usage(),
        assistant_entry_zero_usage("anthropic", "claude-4"),
        user_entry(),
    ];
    // The tool result carries no usage; the assistant bucket has zero cost
    // and zero tokens, so the cost>0 || tokens>0 filter drops it.
    assert!(get_usage_cost_breakdown(&entries).is_empty());
}

#[test]
fn breakdown_reads_the_summary_entries_optional_usage_gates() {
    // A compaction entry without usage contributes nothing.
    let mut entry = compaction_entry(0.5);
    if let SessionEntry::Compaction(compaction) = &mut entry {
        compaction.usage = None;
    }
    assert!(get_usage_cost_breakdown(&[entry]).is_empty());

    // Same for a branch summary without usage.
    let mut entry = branch_summary_entry(0.5);
    if let SessionEntry::BranchSummary(branch) = &mut entry {
        branch.usage = None;
    }
    assert!(get_usage_cost_breakdown(&[entry]).is_empty());
}
