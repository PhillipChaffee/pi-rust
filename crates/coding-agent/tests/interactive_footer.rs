//! The footer width/format suite, upstream's `test/footer-width.test.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, plus boundary cases.
//!
//! Restatements against upstream: the fake `AgentSession` becomes a
//! [`FooterSession`] implementation seeded with a small `FooterModel` and
//! session entries parsed from the same JSON shapes upstream builds by hand;
//! the fake provider is a [`FooterData`] implementation. The upstream
//! cache/compaction entries carry no `id`/`parentId` in the test, so the
//! helpers supply the tree fields the Rust parser requires.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the wire shape; a malformed fixture is a test bug"
)]

use std::rc::Rc;

use pi_agent_core::types::ThinkingLevel;
use pi_coding_agent::extensions::runtime::ContextUsage;
use pi_coding_agent::modes::interactive::components::footer::{
    FooterComponent, FooterData, FooterModel, FooterSession, format_cwd_for_footer, format_tokens,
};
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::session_manager::FileEntry;
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::Component;
use pi_tui::utils::visible_width;
use serde_json::{Value, json};

/// One fake session, upstream's `createSession` options object.
struct FakeSession {
    model: Option<FooterModel>,
    thinking_level: ThinkingLevel,
    context_usage: Option<ContextUsage>,
    cwd: String,
    session_name: Option<String>,
    using_subscription: bool,
    entries: Vec<FileEntry>,
}

impl FooterSession for FakeSession {
    fn model(&self) -> Option<FooterModel> {
        self.model.clone()
    }

    fn thinking_level(&self) -> ThinkingLevel {
        self.thinking_level
    }

    fn context_usage(&self) -> Option<ContextUsage> {
        self.context_usage.clone()
    }

    fn cwd(&self) -> String {
        self.cwd.clone()
    }

    fn session_name(&self) -> Option<String> {
        self.session_name.clone()
    }

    fn is_using_subscription(&self, _provider: &str) -> bool {
        self.using_subscription
    }

    fn for_each_entry(&self, visit: &mut dyn FnMut(&FileEntry)) {
        for entry in &self.entries {
            visit(entry);
        }
    }
}

/// One fake read-only provider.
struct FakeData {
    branch: Option<String>,
    statuses: Vec<(String, String)>,
    provider_count: usize,
}

impl FooterData for FakeData {
    fn get_git_branch(&self) -> Option<String> {
        self.branch.clone()
    }

    fn get_extension_statuses(&self) -> Vec<(String, String)> {
        self.statuses.clone()
    }

    fn get_available_provider_count(&self) -> usize {
        self.provider_count
    }
}

fn model(id: &str, provider: &str, reasoning: bool) -> FooterModel {
    FooterModel {
        id: id.to_string(),
        provider: provider.to_string(),
        context_window: 200_000,
        reasoning,
    }
}

fn session(name: &str) -> FakeSession {
    FakeSession {
        model: Some(model("test-model", "test", false)),
        thinking_level: ThinkingLevel::Off,
        context_usage: Some(ContextUsage {
            tokens: Some(24_600),
            context_window: 200_000,
            percent: Some(12.3),
        }),
        cwd: "/tmp/project".to_string(),
        session_name: (!name.is_empty()).then(|| name.to_string()),
        using_subscription: false,
        entries: Vec::new(),
    }
}

fn footer(session: FakeSession, provider_count: usize) -> FooterComponent {
    FooterComponent::new(
        Rc::new(session),
        Rc::new(FakeData {
            branch: Some("main".to_string()),
            statuses: Vec::new(),
            provider_count,
        }),
    )
}

fn parse(value: Value) -> FileEntry {
    serde_json::from_value(value).expect("fixture parses as a session entry")
}

fn usage_json(input: u64, output: u64, cache_read: u64, cache_write: u64, total: f64) -> Value {
    json!({
        "input": input,
        "output": output,
        "cacheRead": cache_read,
        "cacheWrite": cache_write,
        "totalTokens": input + output + cache_read + cache_write,
        "cost": {
            "input": 0.0,
            "output": 0.0,
            "cacheRead": 0.0,
            "cacheWrite": 0.0,
            "total": total
        }
    })
}

fn assistant_entry(
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    cost_total: f64,
) -> FileEntry {
    parse(json!({
        "type": "message",
        "id": "assistant-1",
        "parentId": null,
        "timestamp": "2025-01-01T00:00:00.000Z",
        "message": {
            "role": "assistant",
            "content": [],
            "api": "anthropic-messages",
            "provider": "test",
            "model": "test-model",
            "usage": usage_json(input, output, cache_read, cache_write, cost_total),
            "stopReason": "stop",
            "timestamp": 0
        }
    }))
}

fn branch_summary_entry(input: u64, output: u64, cost_total: f64) -> FileEntry {
    parse(json!({
        "type": "branch_summary",
        "id": "branch-1",
        "parentId": null,
        "timestamp": "2025-01-01T00:00:00.000Z",
        "fromId": "parent-1",
        "summary": "branch",
        "usage": usage_json(input, output, 0, 0, cost_total)
    }))
}

fn compaction_entry(input: u64, output: u64, cost_total: f64) -> FileEntry {
    parse(json!({
        "type": "compaction",
        "id": "compaction-1",
        "parentId": null,
        "timestamp": "2025-01-01T00:00:00.000Z",
        "summary": "compaction",
        "tokensBefore": 100,
        "usage": usage_json(input, output, 0, 0, cost_total)
    }))
}

fn tool_result_entry(input: u64, output: u64, cost_total: f64) -> FileEntry {
    parse(json!({
        "type": "message",
        "id": "tool-1",
        "parentId": null,
        "timestamp": "2025-01-01T00:00:00.000Z",
        "message": {
            "role": "toolResult",
            "toolCallId": "call-1",
            "toolName": "read",
            "content": [],
            "isError": false,
            "timestamp": 0,
            "usage": usage_json(input, output, 0, 0, cost_total)
        }
    }))
}

fn stats_line(component: &FooterComponent, width: usize) -> String {
    strip_ansi(&component.render(width)[1])
}

#[test]
fn format_cwd_keeps_sibling_paths_sharing_the_home_prefix() {
    assert_eq!(
        format_cwd_for_footer("/home/user2", Some("/home/user")),
        "/home/user2"
    );
}

#[test]
fn format_cwd_abbreviates_home_and_descendants() {
    assert_eq!(format_cwd_for_footer("/home/user", Some("/home/user")), "~");
    assert_eq!(
        format_cwd_for_footer("/home/user/project", Some("/home/user")),
        "~/project"
    );
    assert_eq!(
        format_cwd_for_footer("/home/user/project", None),
        "/home/user/project"
    );
}

#[test]
fn format_tokens_matches_the_compact_boundaries() {
    let cases = [
        (0, "0"),
        (999, "999"),
        (1_000, "1.0k"),
        (1_050, "1.1k"),
        (9_999, "10.0k"),
        (10_000, "10k"),
        (999_999, "1000k"),
        (1_000_000, "1.0M"),
        (9_999_999, "10.0M"),
        (10_000_000, "10M"),
        (12_345_678, "12M"),
    ];
    for (count, expected) in cases {
        assert_eq!(format_tokens(count), expected, "format_tokens({count})");
    }
}

#[test]
fn wide_session_names_keep_every_line_within_width() {
    init_theme(None, false);
    let width = 93;
    let mut state = session(&"한글".repeat(30));
    state.entries = vec![assistant_entry(100, 10, 0, 0, 0.001)];
    let component = footer(state, 1);

    for (index, line) in component.render(width).iter().enumerate() {
        let visible = visible_width(line);
        assert!(
            visible <= width,
            "line {index} visibleWidth={visible} > {width}"
        );
    }
}

#[test]
fn wide_model_and_provider_names_keep_the_stats_line_within_width() {
    init_theme(None, false);
    let width = 60;
    let mut state = session("");
    state.model = Some(FooterModel {
        id: "模".repeat(30),
        provider: "공급자".to_string(),
        context_window: 200_000,
        reasoning: true,
    });
    state.thinking_level = ThinkingLevel::High;
    state.entries = vec![assistant_entry(12_345, 6_789, 0, 0, 1.234)];
    let component = footer(state, 2);

    for (index, line) in component.render(width).iter().enumerate() {
        let visible = visible_width(line);
        assert!(
            visible <= width,
            "line {index} visibleWidth={visible} > {width}"
        );
    }
}

#[test]
fn summary_and_tool_result_usage_roll_into_cost() {
    init_theme(None, false);
    let mut state = session("");
    state.entries = vec![
        assistant_entry(100, 10, 0, 0, 0.5),
        branch_summary_entry(20, 5, 0.25),
        compaction_entry(5, 2, 0.125),
        tool_result_entry(15, 3, 0.375),
    ];
    let component = footer(state, 1);

    let line = stats_line(&component, 120);
    assert!(line.contains("$1.250"), "stats: {line}");
}

#[test]
fn latest_cache_hit_rate_renders() {
    init_theme(None, false);
    let mut state = session("");
    state.entries = vec![assistant_entry(100, 10, 50, 50, 0.001)];
    let component = footer(state, 1);

    assert!(stats_line(&component, 120).contains("CH25.0%"));
}

#[test]
fn kimi_coding_costs_are_subscription_estimates() {
    init_theme(None, false);
    let mut state = session("");
    state.model = Some(model("test-model", "kimi-coding", false));
    state.entries = vec![assistant_entry(100, 10, 0, 0, 1.234)];
    let component = footer(state, 1);

    assert!(stats_line(&component, 120).contains("$1.234 (sub)"));
}

#[test]
fn explicit_subscription_auth_marks_zero_cost() {
    init_theme(None, false);
    let mut state = session("");
    state.model = Some(model("test-model", "anthropic", false));
    state.using_subscription = true;
    let component = footer(state, 1);

    assert!(stats_line(&component, 120).contains("$0.000 (sub)"));
}

#[test]
fn generic_oauth_sign_in_is_not_a_subscription() {
    init_theme(None, false);
    let mut state = session("");
    state.model = Some(model("test-model", "openrouter", false));
    state.entries = vec![assistant_entry(100, 10, 0, 0, 1.234)];
    let component = footer(state, 1);

    let line = stats_line(&component, 120);
    assert!(line.contains("$1.234"));
    assert!(!line.contains("(sub)"));
}

#[test]
fn zero_and_narrow_widths_render_without_panicking() {
    init_theme(None, false);
    let mut state = session(&"wide".repeat(20));
    state.model = Some(FooterModel {
        id: "模".repeat(20),
        provider: "공급자".to_string(),
        context_window: 200_000,
        reasoning: true,
    });
    state.thinking_level = ThinkingLevel::High;
    state.entries = vec![assistant_entry(12_345, 6_789, 1_000, 2_000, 1.234)];
    let component = FooterComponent::new(
        Rc::new(state),
        Rc::new(FakeData {
            branch: Some("main".to_string()),
            statuses: vec![("a".to_string(), "status".to_string())],
            provider_count: 2,
        }),
    );

    for width in [0, 1, 2, 5] {
        for line in component.render(width) {
            assert!(
                visible_width(&line) <= width,
                "width {width}: {:?}",
                strip_ansi(&line)
            );
        }
    }
}

#[test]
fn unknown_context_percent_renders_a_question_mark() {
    init_theme(None, false);
    let mut state = session("");
    state.context_usage = Some(ContextUsage {
        tokens: None,
        context_window: 200_000,
        percent: None,
    });
    let component = footer(state, 1);

    assert!(stats_line(&component, 120).contains("?/200k (auto)"));
}

#[test]
fn extension_statuses_are_sanitized_sorted_and_width_bounded() {
    init_theme(None, false);
    let component = FooterComponent::new(
        Rc::new(session("")),
        Rc::new(FakeData {
            branch: None,
            statuses: vec![
                ("b".to_string(), "line\nwith\tspaces".to_string()),
                ("a".to_string(), "  alpha  ".to_string()),
            ],
            provider_count: 1,
        }),
    );

    let lines = component.render(120);
    assert_eq!(strip_ansi(&lines[2]), "alpha line with spaces");

    let narrow = component.render(10);
    assert!(visible_width(&narrow[2]) <= 10);
    assert!(strip_ansi(&narrow[2]).ends_with("..."));
}

#[test]
fn auto_compact_marker_toggles() {
    init_theme(None, false);
    let component = footer(session(""), 1);
    assert!(stats_line(&component, 120).contains("(auto)"));

    component.set_auto_compact_enabled(false);
    assert!(!stats_line(&component, 120).contains("(auto)"));
}

#[test]
fn set_session_swaps_the_rendered_session() {
    init_theme(None, false);
    let component = footer(session("first"), 1);
    assert!(strip_ansi(&component.render(120)[0]).contains("first"));

    component.set_session(Rc::new(session("second")));
    assert!(strip_ansi(&component.render(120)[0]).contains("second"));
}
