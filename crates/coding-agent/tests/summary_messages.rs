//! The branch- and compaction-summary rows, with the collapsed/expanded
//! contract and the token grouping boundary.

use pi_agent_core::harness::messages::{BranchSummaryMessage, CompactionSummaryMessage};
use pi_coding_agent::modes::interactive::components::branch_summary_message::BranchSummaryMessageComponent;
use pi_coding_agent::modes::interactive::components::compaction_summary_message::CompactionSummaryMessageComponent;
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::Component;

fn branch_message(summary: &str) -> BranchSummaryMessage {
    BranchSummaryMessage {
        role: "branchSummary".to_string(),
        summary: summary.to_string(),
        from_id: Some("entry-1".to_string()),
        timestamp: 0,
    }
}

fn compaction_message(summary: &str, tokens_before: i64) -> CompactionSummaryMessage {
    CompactionSummaryMessage {
        role: "compactionSummary".to_string(),
        summary: summary.to_string(),
        tokens_before,
        timestamp: 0,
    }
}

fn branch_text(component: &BranchSummaryMessageComponent) -> String {
    strip_ansi(&component.render(80).join("\n"))
}

fn compaction_text(component: &CompactionSummaryMessageComponent) -> String {
    strip_ansi(&component.render(80).join("\n"))
}

#[test]
fn branch_summary_starts_collapsed_with_the_expand_hint() {
    init_theme(Some("dark"), false);
    let component = BranchSummaryMessageComponent::new(branch_message("secret summary"), None);
    let rendered = branch_text(&component);
    assert!(rendered.contains("[branch]"));
    assert!(rendered.contains("Branch summary ("));
    assert!(rendered.contains("to expand)"));
    assert!(!rendered.contains("secret summary"));
    assert!(!component.is_expanded());
}

#[test]
fn branch_summary_expands_to_markdown() {
    init_theme(Some("dark"), false);
    let component = BranchSummaryMessageComponent::new(branch_message("secret summary"), None);
    component.set_expanded(true);
    let rendered = branch_text(&component);
    assert!(component.is_expanded());
    assert!(rendered.contains("Branch Summary"));
    assert!(rendered.contains("secret summary"));
}

#[test]
fn branch_summary_invalidate_keeps_the_expanded_state() {
    init_theme(Some("dark"), false);
    let component = BranchSummaryMessageComponent::new(branch_message("kept summary"), None);
    component.set_expanded(true);
    Component::invalidate(&component);
    assert!(branch_text(&component).contains("kept summary"));
}

#[test]
fn compaction_summary_groups_the_token_count() {
    init_theme(Some("dark"), false);
    let component = CompactionSummaryMessageComponent::new(compaction_message("body", 1234), None);
    let rendered = compaction_text(&component);
    assert!(rendered.contains("[compaction]"));
    assert!(rendered.contains("Compacted from 1,234 tokens ("));
    assert!(!rendered.contains("body"));
}

#[test]
fn compaction_summary_expands_to_markdown() {
    init_theme(Some("dark"), false);
    let component = CompactionSummaryMessageComponent::new(
        compaction_message("compacted body", 1_000_000),
        None,
    );
    component.set_expanded(true);
    let rendered = compaction_text(&component);
    assert!(component.is_expanded());
    assert!(rendered.contains("Compacted from 1,000,000 tokens"));
    assert!(rendered.contains("compacted body"));
}

#[test]
fn token_grouping_handles_zero_and_small_counts() {
    init_theme(Some("dark"), false);
    for (tokens, expected) in [(0, "0"), (1, "1"), (999, "999"), (1000, "1,000")] {
        let component =
            CompactionSummaryMessageComponent::new(compaction_message("body", tokens), None);
        assert!(
            compaction_text(&component).contains(&format!("Compacted from {expected} tokens")),
            "count {tokens} should render as {expected}"
        );
    }
}

#[test]
fn compaction_invalidate_keeps_the_expanded_state() {
    init_theme(Some("dark"), false);
    let component =
        CompactionSummaryMessageComponent::new(compaction_message("kept body", 42), None);
    component.set_expanded(true);
    Component::invalidate(&component);
    assert!(compaction_text(&component).contains("kept body"));
}
