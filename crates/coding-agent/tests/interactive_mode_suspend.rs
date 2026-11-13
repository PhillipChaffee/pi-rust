//! `test/interactive-mode-suspend.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Restatements: the port has no process-group signal seam, so
//! `handle_ctrl_z` reports the platform status instead of raising
//! `SIGTSTP` (`crates/coding-agent/src/modes/interactive/interactive_mode/input.rs`).
//! Upstream's three cases exercise the Windows branch, the SIGCONT restore,
//! and the failed-suspend cleanup; the portable assertions they share — one
//! status message, no TUI stop/start, no progress writes — are kept here and
//! the signal-specific ones are recorded rather than dropped.

#![expect(
    clippy::expect_used,
    reason = "the fixture frame is pinned; a failed mount is a test bug"
)]

mod common;

use common::interactive::{Harness, HarnessOptions};

#[tokio::test]
async fn shows_a_status_message_and_skips_suspend() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.tui().start();
    let starts = harness.terminal.start_count();
    let stops = harness.terminal.stop_count();

    harness.frame.handle_ctrl_z();

    let text = harness.chat_text();
    assert!(
        text.contains("Suspend to background is not supported"),
        "{text}"
    );
    assert_eq!(harness.terminal.start_count(), starts);
    assert_eq!(harness.terminal.stop_count(), stops);
    assert!(harness.terminal.progress().is_empty());
}

#[tokio::test]
async fn keeps_the_process_and_tui_untouched_instead_of_sending_sigstop() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.tui().start();

    harness.frame.handle_ctrl_z();

    // The TUI is still running; there is no SIGCONT handler to restore it and
    // no temporary SIGINT handler to remove.
    assert!(!harness.frame.tui().is_stopped());
    assert!(harness.chat_text().contains("Suspend to background"));
}

#[tokio::test]
async fn reports_the_build_restriction_rather_than_the_windows_branch() {
    let harness = Harness::new(HarnessOptions::default()).await;

    harness.frame.handle_ctrl_z();

    let text = harness.chat_text();
    assert!(text.contains("not supported in this build"), "{text}");
    assert!(!text.contains("not supported on Windows"), "{text}");
}
