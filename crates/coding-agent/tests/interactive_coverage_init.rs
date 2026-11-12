//! Coverage of the frame lifecycle: `init`/`run`, startup diagnostics and
//! initial prompts, renderer switching with progress restore, runtime
//! settings application, terminal title/progress writes, and shutdown.
//!
//! `run` performs the managed-tool prefetch. The suite sets the frame's
//! `skip_managed_tool_setup` seam so the headless run stays offline.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the session and diagnostics shapes; a failure is a test bug"
)]
#![allow(
    clippy::large_futures,
    reason = "the acceptance suites await one lifecycle operation at a time"
)]

mod common;

use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use common::interactive::{Harness, HarnessOptions};
use pi_coding_agent::core::agent_session_services::{
    AgentSessionRuntimeDiagnostic, AgentSessionRuntimeDiagnosticKind,
};
use pi_coding_agent::modes::interactive::interactive_mode::InteractiveModeOptions;
use pi_coding_agent::settings_manager::Settings;

#[tokio::test]
async fn run_initializes_reports_startup_state_and_shuts_down_on_quit() {
    let diagnostics = vec![
        AgentSessionRuntimeDiagnostic {
            kind: AgentSessionRuntimeDiagnosticKind::Error,
            message: "startup error".to_owned(),
        },
        AgentSessionRuntimeDiagnostic {
            kind: AgentSessionRuntimeDiagnosticKind::Warning,
            message: "startup warning".to_owned(),
        },
        AgentSessionRuntimeDiagnostic {
            kind: AgentSessionRuntimeDiagnosticKind::Info,
            message: "startup info".to_owned(),
        },
    ];
    let harness = Harness::new(HarnessOptions {
        frame_options: InteractiveModeOptions {
            startup_diagnostics: diagnostics,
            migrated_providers: vec!["anthropic".to_owned()],
            model_fallback_message: Some("fell back".to_owned()),
            initial_message: Some("first prompt".to_owned()),
            initial_images: Vec::new(),
            initial_messages: vec!["second prompt".to_owned()],
            verbose: true,
            skip_managed_tool_setup: true,
            ..InteractiveModeOptions::default()
        },
        seed_manager: Some(Box::new(|manager| {
            manager.append_session_info("run-session").expect("name");
        })),
        ..HarnessOptions::default()
    })
    .await;
    let frame = Rc::clone(&harness.frame);
    let mut run = Box::pin(frame.run());
    let mut titled = false;
    for _ in 0..400 {
        tokio::select! {
            code = &mut run => panic!("run exited early with {code}"),
            () = tokio::time::sleep(Duration::from_millis(5)) => {}
        }
        if harness.terminal.writes().contains("\x1b]0;") {
            titled = true;
            break;
        }
    }
    assert!(titled, "init writes the terminal title");
    harness.send_input("/quit");
    harness.send_input("\r");
    let code = tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("the run loop exits after /quit");
    assert_eq!(code, 0);
    assert!(harness.terminal.stop_count() >= 1);
    let text = harness.chat_text();
    assert!(text.contains("startup error"), "{text}");
    assert!(text.contains("startup warning"), "{text}");
    assert!(text.contains("startup info"), "{text}");
    assert!(text.contains("Migrated credentials"), "{text}");
    assert!(text.contains("fell back"), "{text}");
}

#[tokio::test]
async fn switch_tui_mode_restores_progress_and_blocks_on_overlays() {
    let mut settings = Settings::new();
    settings.insert(
        "terminal".to_owned(),
        serde_json::json!({ "showTerminalProgress": true }),
    );
    let release = Arc::new(tokio::sync::Notify::new());
    let harness = Harness::new(HarnessOptions {
        settings,
        stream: Some(common::interactive::gated_stream(Arc::clone(&release))),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.tui().start();
    let session = harness.frame.session();
    let prompt = tokio::spawn({
        let session = Arc::clone(&session);
        async move { session.prompt("first".to_owned(), None).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;

    assert!(
        harness
            .frame
            .switch_tui_mode(pi_tui::tui::TuiMode::Fullscreen, true, true)
    );
    assert_eq!(
        harness.frame.current_tui_mode(),
        pi_tui::tui::TuiMode::Fullscreen
    );
    assert!(
        harness.terminal.writes().contains("\x1b]9;4;3\x07"),
        "the progress restore writes the active sequence"
    );
    assert!(
        harness
            .frame
            .switch_tui_mode(pi_tui::tui::TuiMode::Regular, true, true)
    );
    assert_eq!(
        harness.frame.current_tui_mode(),
        pi_tui::tui::TuiMode::Regular
    );

    let overlay: Rc<dyn pi_tui::tui::Component> =
        Rc::new(pi_tui::components::Text::with_padding("overlay", 0, 0));
    let _ = harness.frame.tui().show_overlay(overlay, None);
    assert!(
        !harness
            .frame
            .switch_tui_mode(pi_tui::tui::TuiMode::Fullscreen, false, false)
    );
    harness.frame.tui().hide_overlay();
    assert!(
        harness
            .frame
            .switch_tui_mode(pi_tui::tui::TuiMode::Regular, false, false)
    );
    release.notify_waiters();
    let _ = prompt.await;
}

#[tokio::test]
async fn apply_runtime_settings_rebinds_every_surface() {
    let harness = Harness::new(HarnessOptions::default()).await;
    {
        let settings = harness.frame.session().settings_manager();
        let mut guard = guard(&settings);
        guard.set_show_hardware_cursor(true);
        guard.set_clear_on_shrink(false);
        guard.set_hide_thinking_block(true);
        guard.set_output_pad(1);
        guard.set_editor_padding_x(2.0);
        guard.set_autocomplete_max_visible(9.0);
    }
    harness.frame.apply_runtime_settings();
    assert!(harness.frame.tui().get_show_hardware_cursor());
    assert!(!harness.frame.tui().get_clear_on_shrink());
    assert_eq!(harness.frame.status_container().children.borrow().len(), 0);

    let settings = harness.frame.session().settings_manager();
    guard(&settings).set_clear_on_shrink(true);
    harness.frame.apply_runtime_settings();
    assert!(harness.frame.tui().get_clear_on_shrink());
}

fn guard<T>(value: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    value
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[tokio::test]
async fn terminal_title_and_progress_write_the_osc_sequences() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.update_terminal_title();
    assert!(harness.terminal.writes().contains("\x1b]0;"));
    harness.frame.session().set_session_name("title test");
    harness.frame.update_terminal_title();
    assert!(harness.terminal.writes().contains("title test"));

    harness.frame.set_terminal_progress(true);
    assert!(harness.terminal.writes().contains("\x1b]9;4;3\x07"));
    harness.frame.set_terminal_progress(false);
    assert!(harness.terminal.writes().contains("\x1b]9;4;0\x07"));
}

#[tokio::test]
async fn changelog_for_display_skips_after_messages_and_without_new_entries() {
    let harness = Harness::new(HarnessOptions {
        messages: vec![common::agent_session::user_message("hello", 1)],
        ..HarnessOptions::default()
    })
    .await;
    assert!(harness.frame.changelog_for_display().is_none());

    let harness = Harness::new(HarnessOptions::default()).await;
    assert!(harness.frame.changelog_for_display().is_none());
    let settings = harness.frame.session().settings_manager();
    guard(&settings).set_last_changelog_version("0.0.1");
    assert!(harness.frame.changelog_for_display().is_none());
}

#[tokio::test]
async fn stop_releases_the_watcher_and_tolerates_a_second_stop() {
    use pi_coding_agent::settings_manager::FullscreenExitOutput;

    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.tui().start();
    harness.frame.stop(FullscreenExitOutput::Transcript);
    assert!(harness.frame.tui().is_stopped());
    harness.frame.stop(FullscreenExitOutput::ResumeHint);
    harness.frame.shutdown().await;
    assert!(harness.frame.tui().is_stopped());
}
