//! Coverage of the frame's input handlers: submit routing, bash, follow-up
//! and dequeue, clipboard paste, the external editor, and the Ctrl/Escape
//! ladders.
//!
//! The streaming cases use a gated faux stream, upstream's held promise. The
//! external editor is a real shell script fixture, so the hand-off is the
//! production spawn/read-back path.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the session and script shapes; a failure is a test bug"
)]
#![allow(
    clippy::large_futures,
    reason = "the acceptance suites await one frame operation at a time"
)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::interactive::{ClipboardGuard, Harness, HarnessOptions};
use pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery;
use pi_coding_agent::settings_manager::Settings;

fn double_escape(action: &str) -> Settings {
    let mut settings = Settings::new();
    settings.insert(
        "doubleEscapeAction".to_owned(),
        serde_json::Value::String(action.to_owned()),
    );
    settings
}

#[tokio::test]
async fn submit_routes_every_non_network_slash_command() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    for line in [
        "/session",
        "/settings",
        "/model",
        "/thinking",
        "/name demo",
        "/changelog",
        "/hotkeys",
        "/fork",
        "/clone",
        "/tree",
        "/trust",
        "/new",
        "/reload",
        "/arminsayshi",
        "/dementedelves",
        "/resume",
        "/compact",
        "/scoped-models",
    ] {
        common::interactive::submit(&harness, line).await;
        assert!(!harness.chat_text().is_empty(), "{line}");
    }
}

#[tokio::test]
async fn submit_runs_bash_in_both_context_modes() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    common::interactive::submit(&harness, "!echo hello").await;
    assert!(harness.chat_text().contains("hello"));
    common::interactive::submit(&harness, "!!echo excluded").await;
    assert!(harness.chat_text().contains("excluded"));
    common::interactive::submit(&harness, "!").await;
    common::interactive::submit(&harness, "   ").await;
}

#[tokio::test]
async fn submit_warns_when_bash_is_already_running() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    let session = harness.frame.session();
    let bash = tokio::spawn({
        let session = Arc::clone(&session);
        async move {
            session
                .execute_bash(
                    "sleep 1",
                    None,
                    Some(pi_coding_agent::core::agent_session::BashExecutionOptions::default()),
                )
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(session.is_bash_running());
    common::interactive::submit(&harness, "!echo later").await;
    assert!(harness.chat_text().contains("already running"));
    session.abort_bash();
    let _ = bash.await;
}

#[tokio::test]
async fn follow_up_handles_empty_idle_and_streaming_states() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    harness.frame.handle_follow_up().await;

    harness.frame.editor().set_text("idle follow up");
    harness.frame.handle_follow_up().await;
    assert!(harness.frame.editor().get_text().is_empty());

    let release = Arc::new(tokio::sync::Notify::new());
    let harness = common::interactive::harness_with(HarnessOptions {
        stream: Some(common::interactive::gated_stream(Arc::clone(&release))),
        ..HarnessOptions::default()
    })
    .await;
    let session = harness.frame.session();
    let prompt = tokio::spawn({
        let session = Arc::clone(&session);
        async move { session.prompt("first".to_owned(), None).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    harness.frame.editor().set_text("queued follow up");
    let mut follow_up = Box::pin(harness.frame.handle_follow_up());
    tokio::select! {
        () = &mut follow_up => {}
        () = tokio::time::sleep(Duration::from_millis(30)) => {
            release.notify_waiters();
            tokio::select! {
                () = &mut follow_up => {}
                () = tokio::time::sleep(Duration::from_millis(200)) => {
                    session.abort().await;
                }
            }
        }
    }
    release.notify_waiters();
    tokio::select! {
        _ = prompt => {}
        () = tokio::time::sleep(Duration::from_millis(200)) => {}
    }
}

#[tokio::test]
async fn dequeue_reports_empty_and_restores_queued_messages() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    harness.frame.handle_dequeue();
    assert!(
        harness
            .chat_text()
            .contains("No queued messages to restore")
    );

    harness
        .frame
        .queue_compaction_message("restore me", QueueDelivery::Steer);
    harness.frame.handle_dequeue();
    let text = harness.chat_text();
    assert!(text.contains("Restored 1 queued message"), "{text}");
    assert_eq!(harness.frame.editor().get_text(), "restore me");
}

#[tokio::test]
async fn clipboard_paste_inserts_text_and_image_paths() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    let clipboard = ClipboardGuard::new(Some("pasted text"));
    harness.frame.handle_clipboard_paste().await;
    assert_eq!(harness.frame.editor().get_text(), "pasted text");
    drop(clipboard);

    harness.frame.editor().set_text("");
    let png = {
        let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        bytes.extend_from_slice(&[0, 0, 0, 13]);
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&[0; 8]);
        bytes
    };
    let clipboard = ClipboardGuard::with_image(None, Some(png));
    harness.frame.handle_clipboard_paste().await;
    let text = harness.frame.editor().get_text();
    assert!(text.contains("pi-clipboard-"), "{text}");
    assert!(text.contains(".png"), "{text}");
    drop(clipboard);
}

#[tokio::test]
async fn external_editor_round_trips_the_editor_text() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let script = root.path().join("editor.sh");
    std::fs::write(&script, "#!/bin/sh\necho edited > \"$1\"\n").expect("script");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
            .expect("executable");
    }
    let mut settings = Settings::new();
    settings.insert(
        "externalEditor".to_owned(),
        serde_json::Value::String(script.display().to_string()),
    );
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;
    harness.start_tui();
    harness.frame.editor().set_text("original");
    harness.frame.handle_open_external_editor().await;
    assert_eq!(harness.frame.editor().get_text(), "edited");
}

#[tokio::test]
async fn escape_streaming_restores_queued_messages_and_aborts() {
    let release = Arc::new(tokio::sync::Notify::new());
    let harness = common::interactive::harness_with(HarnessOptions {
        stream: Some(common::interactive::gated_stream(Arc::clone(&release))),
        ..HarnessOptions::default()
    })
    .await;
    let session = harness.frame.session();
    let prompt = tokio::spawn({
        let session = Arc::clone(&session);
        async move { session.prompt("first".to_owned(), None).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    session
        .steer("queued".to_owned(), None, None)
        .await
        .expect("steer");
    harness.frame.handle_escape().await;
    assert_eq!(harness.frame.editor().get_text(), "queued");
    release.notify_waiters();
    let _ = prompt.await;
}

#[tokio::test]
async fn escape_bash_mode_and_idle_double_escape_ladders() {
    let harness = common::interactive::harness_with(HarnessOptions {
        settings: double_escape("tree"),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.handle_editor_changed("!ls");
    harness.frame.handle_escape().await;
    assert_eq!(harness.frame.editor().get_text(), "");
    harness.frame.handle_escape().await;
    harness.frame.handle_escape().await;

    let harness = common::interactive::harness_with(HarnessOptions {
        settings: double_escape("fork"),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.handle_escape().await;
    harness.frame.handle_escape().await;

    let harness = common::interactive::harness_with(HarnessOptions {
        settings: double_escape("none"),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.handle_escape().await;
    harness.frame.handle_escape().await;
    harness.frame.editor().set_text("not empty");
    harness.frame.handle_escape().await;
}

#[tokio::test]
async fn escape_aborts_a_bash_run() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    let session = harness.frame.session();
    let bash = tokio::spawn({
        let session = Arc::clone(&session);
        async move {
            session
                .execute_bash(
                    "sleep 1",
                    None,
                    Some(pi_coding_agent::core::agent_session::BashExecutionOptions::default()),
                )
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(session.is_bash_running());
    harness.frame.handle_escape().await;
    for _ in 0..50 {
        if !session.is_bash_running() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let _ = bash.await;
}

#[tokio::test]
async fn ctrl_c_clears_then_shuts_down_and_ctrl_d_shuts_down() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.editor().set_text("hello");
    harness.frame.handle_ctrl_c().await;
    assert!(harness.frame.editor().get_text().is_empty());

    let harness = Harness::new(HarnessOptions::default()).await;
    harness.start_tui();
    harness.frame.handle_ctrl_d().await;
    assert!(harness.frame.tui().is_stopped());

    harness.frame.handle_ctrl_z();
    assert!(harness.chat_text().contains("not supported"));
}

#[tokio::test]
async fn extension_command_detection_uses_the_registered_runner() {
    let fixtures = common::interactive::command_fixtures(&["demo"]);
    let harness = common::interactive::harness_with(HarnessOptions {
        fixtures: Some(fixtures),
        ..HarnessOptions::default()
    })
    .await;
    assert!(harness.frame.is_extension_command("/demo now"));
    assert!(!harness.frame.is_extension_command("demo"));
    assert!(!harness.frame.is_extension_command("/missing"));
}

#[tokio::test]
async fn deferred_bash_components_move_into_the_transcript() {
    let release = Arc::new(tokio::sync::Notify::new());
    let harness = common::interactive::harness_with(HarnessOptions {
        stream: Some(common::interactive::gated_stream(Arc::clone(&release))),
        ..HarnessOptions::default()
    })
    .await;
    let session = harness.frame.session();
    let prompt = tokio::spawn({
        let session = Arc::clone(&session);
        async move { session.prompt("first".to_owned(), None).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    harness
        .frame
        .handle_bash_command("echo deferred", false)
        .await;
    harness.frame.flush_pending_bash_components();
    assert!(harness.chat_text().contains("deferred"));
    release.notify_waiters();
    let _ = prompt.await;
}

#[tokio::test]
async fn compact_command_without_messages_is_a_no_op() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    harness.frame.handle_compact_command(None).await;
    harness
        .frame
        .handle_compact_command(Some("focus".to_owned()))
        .await;
}
