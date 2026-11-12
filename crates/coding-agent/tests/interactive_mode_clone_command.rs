//! `test/interactive-mode-clone-command.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Restatements: upstream's mock `runtimeHost.fork` becomes the real
//! `AgentSessionRuntime`; the assertions read the new session the frame
//! rebinds to and the editor/status the frame sets.
//! `renderCurrentSessionState`/`requestRender` are not called on the success
//! path upstream either; the port does not add a render there.

#![expect(
    clippy::expect_used,
    reason = "the fixture messages are pinned; a rejected append is a test bug"
)]
#![allow(
    clippy::large_futures,
    reason = "the acceptance test awaits one frame operation, upstream's direct suite shape"
)]

mod common;

use common::agent_session::{assistant_message, user_message};
use common::interactive::{Harness, HarnessOptions};

#[tokio::test]
async fn clones_the_current_leaf_into_a_new_session() {
    let harness = Harness::new(HarnessOptions {
        messages: vec![
            user_message("first", 1),
            pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(
                assistant_message("reply", 2),
            )),
        ],
        ..HarnessOptions::default()
    })
    .await;
    let before = harness.frame.session();
    assert!(before.leaf_id().is_some());

    harness.frame.handle_clone_command().await;

    let after = harness.frame.session();
    assert!(!std::sync::Arc::ptr_eq(&before, &after));
    assert!(harness.chat_text().contains("Cloned to new session"));
    assert_eq!(harness.frame.editor().get_text(), "");
    let messages = &after.agent().state().messages;
    assert_eq!(messages.len(), 2);
}

#[tokio::test]
async fn shows_a_status_message_when_there_is_nothing_to_clone() {
    let harness = Harness::new(HarnessOptions::default()).await;
    // A fresh session persists its initial model and thinking entries; the
    // leaf-less state upstream mocks is the explicitly reset pointer.
    harness.frame.session().reset_leaf();

    harness.frame.handle_clone_command().await;

    assert!(harness.chat_text().contains("Nothing to clone yet"));
    assert!(!harness.chat_text().contains("Cloned to new session"));
}
