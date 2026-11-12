//! The `status-indicator.test.ts` cases at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported against the
//! standalone loader rather than the editor, plus narrow-width and
//! indicator-replacement boundaries.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use pi_coding_agent::modes::interactive::components::keybinding_hints::key_text;
use pi_coding_agent::modes::interactive::components::status_indicator::{
    BranchSummaryStatusIndicator, CompactionStatusIndicator, CompactionStatusReason, IdleStatus,
    RetryStatusIndicator, StatusIndicatorKind, WorkingStatusIndicator,
};
use pi_coding_agent::modes::interactive::theme::theme::{init_theme, theme};
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::components::LoaderIndicatorOptions;
use pi_tui::tui::{Component, RenderRequest};
use pi_tui::utils::visible_width;

fn noop_request() -> RenderRequest {
    Arc::new(|| {})
}

fn line(indicator: &impl Component, width: usize) -> String {
    strip_ansi(&indicator.render(width).join("\n"))
}

#[test]
fn idle_status_keeps_two_blank_rows() {
    let lines = IdleStatus.render(20);
    assert_eq!(lines, vec![" ".repeat(20), " ".repeat(20)]);
}

#[test]
fn working_indicator_styles_spinner_and_message() {
    init_theme(Some("dark"), false);
    let indicator = WorkingStatusIndicator::new(noop_request(), "Working", None, None);
    let rendered = indicator
        .render(20)
        .get(1)
        .cloned()
        .expect("the loader renders a content line");
    assert!(rendered.contains(theme().get_fg_ansi("accent")));
    assert!(rendered.contains(theme().get_fg_ansi("muted")));
    indicator.dispose();
}

#[test]
fn render_in_border_drops_padding_and_trailing_spaces() {
    init_theme(Some("dark"), false);
    let indicator = WorkingStatusIndicator::new(noop_request(), "Working", None, None);
    let rendered = indicator.render_in_border(30);
    assert!(!rendered.starts_with(' '));
    assert!(strip_ansi(&rendered).ends_with("Working"));
    assert!(visible_width(&rendered) <= 30);
    indicator.dispose();
}

#[test]
fn render_spinner_in_border_keeps_only_the_indicator() {
    init_theme(Some("dark"), false);
    let indicator = WorkingStatusIndicator::new(noop_request(), "Working", None, None);
    let spinner = indicator.render_spinner_in_border(20);
    assert_eq!(visible_width(&spinner), 1);
    assert!(!strip_ansi(&spinner).contains("Working"));
    indicator.dispose();
}

#[test]
fn replacement_frames_render_verbatim() {
    init_theme(Some("dark"), false);
    let indicator = WorkingStatusIndicator::new(noop_request(), "Working", None, None);
    indicator.set_indicator(Some(LoaderIndicatorOptions {
        frames: Some(vec!["*".to_string()]),
        interval_ms: None,
    }));
    assert_eq!(strip_ansi(&indicator.render_spinner_in_border(20)), "*");
    assert!(strip_ansi(&indicator.render_in_border(30)).contains("* Working"));
    indicator.dispose();
}

#[test]
fn empty_frames_hide_the_spinner() {
    init_theme(Some("dark"), false);
    let indicator = WorkingStatusIndicator::new(noop_request(), "Working", None, None);
    indicator.set_indicator(Some(LoaderIndicatorOptions {
        frames: Some(Vec::new()),
        interval_ms: None,
    }));
    assert_eq!(indicator.render_spinner_in_border(20), "");
    assert!(strip_ansi(&indicator.render_in_border(30)).contains("Working"));
    indicator.dispose();
}

#[test]
fn compaction_labels_match_the_reason() {
    init_theme(Some("dark"), false);
    let cancel_hint = format!("({} to cancel)", key_text("app.interrupt"));
    let cases = [
        (
            CompactionStatusReason::Manual,
            format!("Compacting context... {cancel_hint}"),
        ),
        (
            CompactionStatusReason::Threshold,
            format!("Auto-compacting... {cancel_hint}"),
        ),
        (
            CompactionStatusReason::Overflow,
            format!("Context overflow detected, Auto-compacting... {cancel_hint}"),
        ),
    ];
    for (reason, expected) in cases {
        let indicator = CompactionStatusIndicator::new(noop_request(), reason);
        assert!(line(&indicator, 120).contains(&expected));
        assert_eq!(indicator.kind(), StatusIndicatorKind::Compaction);
        indicator.dispose();
    }
}

#[test]
fn branch_summary_label_names_the_interrupt_binding() {
    init_theme(Some("dark"), false);
    let indicator = BranchSummaryStatusIndicator::new(noop_request());
    let expected = format!(
        "Summarizing branch... ({} to cancel)",
        key_text("app.interrupt")
    );
    assert!(line(&indicator, 120).contains(&expected));
    assert_eq!(indicator.kind(), StatusIndicatorKind::BranchSummary);
    indicator.dispose();
}

#[test]
fn retry_counts_down_and_stops_on_dispose() {
    init_theme(Some("dark"), false);
    let indicator = RetryStatusIndicator::new(noop_request(), 1, 3, 3000);
    assert!(line(&indicator, 120).contains("Retrying (1/3) in 3s..."));
    let countdown = indicator.countdown().expect("the countdown is armed");
    countdown.tick();
    assert!(line(&indicator, 120).contains("Retrying (1/3) in 2s..."));

    indicator.dispose();
    assert!(indicator.countdown().is_none());
    assert!(!countdown.is_armed());

    let before = line(&indicator, 120);
    countdown.tick();
    assert_eq!(line(&indicator, 120), before);
}

#[test]
fn narrow_widths_never_overflow_the_border() {
    init_theme(Some("dark"), false);
    let indicator = WorkingStatusIndicator::new(noop_request(), "Working", None, None);
    assert_eq!(indicator.render_in_border(0), "");
    for width in [1, 2, 4, 10, 20] {
        assert!(visible_width(&indicator.render_in_border(width)) <= width);
        assert!(visible_width(&indicator.render_spinner_in_border(width)) <= width);
    }
    indicator.dispose();
}

#[test]
fn kind_wire_spellings_match_upstream() {
    assert_eq!(StatusIndicatorKind::Working.as_str(), "working");
    assert_eq!(StatusIndicatorKind::Retry.as_str(), "retry");
    assert_eq!(StatusIndicatorKind::Compaction.as_str(), "compaction");
    assert_eq!(StatusIndicatorKind::BranchSummary.as_str(), "branchSummary");
}
