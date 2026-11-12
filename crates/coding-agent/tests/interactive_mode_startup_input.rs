//! `test/interactive-mode-startup-input.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Restatements:
//! - Upstream's fake `this` object becomes the real frame; the assertions
//!   read the editor text and transcript the frame mounts.
//! - `pendingUserInputs`/`onInputCallback` are the port's editor submit
//!   channel: a submission handled before the main loop consumes it lands in
//!   the editor with the startup status, and the loop's `submit_text` is the
//!   drain. The "queued before a callback is installed" case therefore
//!   asserts the channel's startup branch, not an array mutation.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the session's messages; a missing message is a test bug"
)]
#![allow(
    clippy::large_futures,
    reason = "the acceptance test awaits one frame operation, upstream's direct suite shape"
)]

mod common;

use common::interactive::{Harness, HarnessOptions};
use pi_coding_agent::settings_manager::Settings;

/// The frame's editor text.
fn editor_text(harness: &Harness) -> String {
    harness.frame.editor().get_text()
}

#[tokio::test]
async fn restores_a_prompt_submitted_while_managed_tool_setup_is_running() {
    let harness = Harness::new(HarnessOptions::default()).await;

    harness.frame.handle_startup_submit("early prompt");

    assert_eq!(editor_text(&harness), "early prompt");
    assert!(harness.chat_text().contains("Startup is still in progress"));
}

#[tokio::test]
async fn queues_a_normal_prompt_submitted_before_the_input_callback_is_installed() {
    let harness = Harness::new(HarnessOptions {
        settings: Settings::new(),
        ..HarnessOptions::default()
    })
    .await;

    harness.frame.submit_text(" early prompt ").await;

    let session = harness.frame.session();
    let texts: Vec<String> = session
        .agent()
        .state()
        .messages
        .iter()
        .filter_map(|message| match message {
            pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::User(user)) => {
                Some(pi_ai::utils::text::content_text(&user.content))
            }
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["early prompt".to_owned()]);
    assert_eq!(editor_text(&harness), "");
}

#[tokio::test]
async fn returns_queued_startup_input_before_installing_a_new_input_callback() {
    let harness = Harness::new(HarnessOptions::default()).await;

    harness.frame.handle_startup_submit("queued prompt");

    assert_eq!(editor_text(&harness), "queued prompt");
    assert!(harness.chat_text().contains("Startup is still in progress"));
    assert!(harness.frame.session().agent().state().messages.is_empty());
}
