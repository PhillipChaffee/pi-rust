//! Boundary coverage for the frame's slash-command dispatch (#131): every
//! command that settles without waiting on a dialog is submitted through the
//! same `submit_text` path the TUI drives, then its frame command is drained.
//!
//! Dialog commands (`/login`, `/logout`, `/resume`, `/tree`, `/session`
//! selector) and `/share`'s network legs are exercised by the dedicated
//! suites or need a real terminal; this file pins the dispatch-to-handler
//! wiring and the non-dialog handlers' completion.

#![allow(
    clippy::expect_used,
    clippy::large_futures,
    reason = "the shared fixture modules pin setup with expect, and each frame future is driven alone"
)]

mod common;

use common::agent_session::{assistant_message, user_message};
use common::interactive::{Harness, HarnessOptions};

async fn submit(harness: &Harness, line: &str) {
    harness.frame.submit_text(line).await;
    harness.frame.drain_commands().await;
    harness.frame.drain_events();
}

#[tokio::test]
async fn session_command_renders_session_information() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/session").await;
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn settings_command_mounts_the_settings_selector() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/settings").await;
}

#[tokio::test]
async fn model_command_mounts_the_model_selector() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/model").await;
    let _ = harness.frame.current_tui_mode();
}

#[tokio::test]
async fn thinking_command_cycles_the_level() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/thinking").await;
    submit(&harness, "/thinking high").await;
    assert_eq!(
        harness.frame.session().thinking_level(),
        pi_agent_core::types::ThinkingLevel::High
    );
}

#[tokio::test]
async fn trust_command_mounts_the_trust_selector() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/trust").await;
}

#[tokio::test]
async fn name_command_sets_the_session_name() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/name boundary-name").await;
    assert_eq!(
        harness.frame.session().session_name().as_deref(),
        Some("boundary-name")
    );
}

#[tokio::test]
async fn changelog_and_hotkeys_render_text() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/changelog").await;
    submit(&harness, "/hotkeys").await;
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn copy_command_without_a_message_reports_a_warning() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/copy").await;
}

#[tokio::test]
async fn compact_command_without_messages_reports_a_warning() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/compact").await;
}

#[tokio::test]
async fn export_command_writes_the_html_export() {
    let harness = Harness::new(HarnessOptions {
        persisted: true,
        messages: vec![
            user_message("hello", 1),
            pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(
                assistant_message("hi", 2),
            )),
        ],
        ..HarnessOptions::default()
    })
    .await;
    let target = harness.temp.path().join("boundary.html");
    submit(&harness, &format!("/export {}", target.display())).await;
    assert!(target.exists(), "{}", harness.chat_text());
}

#[tokio::test]
async fn import_command_without_an_argument_reports_usage() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/import").await;
}

#[tokio::test]
async fn reload_command_reloads_resources() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/reload").await;
}

#[tokio::test]
async fn new_command_starts_a_fresh_session() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let before = harness.frame.session();
    submit(&harness, "/new").await;
    let after = harness.frame.session();
    assert!(!std::sync::Arc::ptr_eq(&before, &after));
}

#[tokio::test]
async fn debug_command_renders_diagnostics() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/debug").await;
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn easter_egg_commands_render_their_panels() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/arminsayshi").await;
    submit(&harness, "/dementedelves").await;
}

#[tokio::test]
async fn fork_command_without_messages_reports_nothing_to_fork() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/fork").await;
}

#[tokio::test]
async fn scoped_models_command_mounts_the_picker() {
    let harness = Harness::new(HarnessOptions::default()).await;
    submit(&harness, "/scoped-models").await;
}
