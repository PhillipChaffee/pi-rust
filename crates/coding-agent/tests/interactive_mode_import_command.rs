//! `test/interactive-mode-import-command.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Restatements:
//! - The confirmation dialog is not ported (the frame's `/import` runs the
//!   named path directly), so the mocked `showExtensionConfirm` assertions
//!   collapse into "the import ran".
//! - Upstream's `importFromJsonl` mock becomes the real runtime over a
//!   persisted session store; the file-not-found assertion reads the real
//!   `SessionImportFileNotFoundError` text.

#![expect(
    clippy::expect_used,
    reason = "the fixture session files are pinned; a failed write is a test bug"
)]
#![allow(
    clippy::large_futures,
    reason = "the acceptance test awaits one frame operation, upstream's direct suite shape"
)]

mod common;

use common::agent_session::{assistant_message, user_message};
use common::interactive::{Harness, HarnessOptions};
use pi_coding_agent::session_manager::SessionManager;

fn app() -> HarnessOptions {
    HarnessOptions {
        persisted: true,
        ..HarnessOptions::default()
    }
}

#[tokio::test]
async fn strips_quotes_from_import_path_arguments() {
    let harness = Harness::new(app()).await;
    let frame = &harness.frame;
    assert_eq!(
        frame.get_path_command_argument("/import \"path/to/session.jsonl\"", "/import"),
        Some("path/to/session.jsonl".to_owned())
    );
    assert_eq!(
        frame.get_path_command_argument("/import \"path with spaces/session.jsonl\"", "/import"),
        Some("path with spaces/session.jsonl".to_owned())
    );
}

#[tokio::test]
async fn preserves_apostrophes_in_unquoted_import_path_arguments() {
    let harness = Harness::new(app()).await;
    assert_eq!(
        harness
            .frame
            .get_path_command_argument("/import john's/session.jsonl", "/import"),
        Some("john's/session.jsonl".to_owned())
    );
}

#[tokio::test]
async fn enforces_command_token_boundaries() {
    let harness = Harness::new(app()).await;
    let frame = &harness.frame;
    assert_eq!(
        frame.get_path_command_argument("/important /tmp/session.jsonl", "/import"),
        None
    );
    assert_eq!(
        frame.get_path_command_argument("/exporter out.html", "/export"),
        None
    );
    assert_eq!(
        frame.get_path_command_argument("/import /tmp/session.jsonl", "/import"),
        Some("/tmp/session.jsonl".to_owned())
    );
}

/// Write a source session with one exchange and return its file path.
fn source_session(root: &str) -> String {
    let mut manager =
        SessionManager::create(root, Some(&format!("{root}/sessions")), None).expect("manager");
    manager
        .append_message(user_message("imported", 1))
        .expect("user message");
    manager
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::Assistant(assistant_message("ok", 2)),
        ))
        .expect("assistant message");
    manager.session_file().expect("session file").to_owned()
}

#[tokio::test]
async fn passes_unquoted_path_to_the_runtime_import() {
    let harness = Harness::new(app()).await;
    let source = source_session(
        harness
            .temp
            .path()
            .join("source")
            .to_str()
            .expect("utf-8 temp path"),
    );
    let quoted = format!("/import \"{source}\"");

    harness.frame.handle_import_command(&quoted).await;

    assert!(
        harness
            .chat_text()
            .contains(&format!("Session imported from: {source}"))
    );
    assert!(!harness.chat_text().contains("Failed to import session"));
    assert_eq!(harness.frame.session().agent().state().messages.len(), 2);
}

#[tokio::test]
async fn passes_unquoted_apostrophe_path_to_the_runtime_import_unchanged() {
    let harness = Harness::new(app()).await;
    let source = source_session(
        harness
            .temp
            .path()
            .join("source")
            .to_str()
            .expect("utf-8 temp path"),
    );
    let apostrophe = harness.temp.path().join("john's-session.jsonl");
    std::fs::copy(&source, &apostrophe).expect("copy");

    harness
        .frame
        .handle_import_command(&format!("/import {}", apostrophe.display()))
        .await;

    assert!(
        harness
            .chat_text()
            .contains(&format!("Session imported from: {}", apostrophe.display()))
    );
    assert!(!harness.chat_text().contains("Failed to import session"));
}

#[tokio::test]
async fn shows_a_non_fatal_error_when_the_import_path_does_not_exist() {
    let harness = Harness::new(app()).await;
    let missing = harness.temp.path().join("missing-session.jsonl");

    harness
        .frame
        .handle_import_command(&format!("/import {}", missing.display()))
        .await;

    assert!(harness.chat_text().contains(&format!(
        "Failed to import session: File not found: {}",
        missing.display()
    )));
    assert!(!harness.chat_text().contains("Session imported from"));
}
