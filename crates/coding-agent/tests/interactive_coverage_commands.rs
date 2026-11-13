//! Coverage of the frame's command handlers and selectors: every
//! `FrameCommand` the slash router can produce, the settings callback bag,
//! model/thinking cycles, login/logout, export/import, session/tree/fork
//! selectors, and the project-trust surface.
//!
//! Selectors are driven the way the TUI drives them: the command future is
//! drained, keys go through the recording terminal to the focused selector,
//! and the resulting frame command is drained again. Dialog commands that
//! await their answer (`/tree` with the summary prompt) pin the drain future
//! and answer while it is in flight.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the session and settings shapes; a failure is a test bug"
)]
#![allow(
    clippy::large_futures,
    reason = "the acceptance suites await one frame operation at a time"
)]

mod common;

use std::time::Duration;

use common::interactive::{Harness, HarnessOptions};
use pi_ai::types::Transport;
use pi_coding_agent::settings_manager::{
    DefaultProjectTrust, FullscreenExitOutput, MermaidRenderingMode, QueueMode, Settings,
};
use pi_tui::components::scroll_view::ScrollViewScrollbar;
use pi_tui::tui::TuiMode;

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "the callback bag is driven as one scenario so the settings snapshot reads together"
)]
async fn settings_callbacks_write_every_setting_and_send_every_command() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    let callbacks = harness.frame.settings_callbacks();
    let settings = harness.frame.session().settings_manager();

    (callbacks.on_auto_compact_change)(true);
    assert!(harness.frame.session().auto_compaction_enabled());
    (callbacks.on_show_images_change)(true);
    (callbacks.on_image_width_cells_change)(48);
    (callbacks.on_auto_resize_images_change)(true);
    (callbacks.on_block_images_change)(true);
    (callbacks.on_enable_skill_commands_change)(true);
    (callbacks.on_steering_mode_change)(QueueMode::OneAtATime);
    (callbacks.on_follow_up_mode_change)(QueueMode::All);
    (callbacks.on_transport_change)(Transport::Websocket);
    (callbacks.on_http_idle_timeout_ms_change)(1500);
    {
        let guard = settings.lock().expect("settings");
        assert!(guard.get_show_images());
        assert_eq!(guard.get_image_width_cells(), 48);
        assert!(guard.get_image_auto_resize());
        assert!(guard.get_block_images());
        assert!(guard.get_enable_skill_commands());
        assert_eq!(guard.get_transport(), Transport::Websocket);
        assert_eq!(guard.get_http_idle_timeout_ms(), Ok(1500));
    }

    let model = harness.frame.session().model().expect("model");
    (callbacks.on_model_thinking_level_change)(
        &model.provider.0,
        &model.id,
        pi_agent_core::types::ThinkingLevel::High,
    );
    assert_eq!(
        harness.frame.session().thinking_level(),
        pi_agent_core::types::ThinkingLevel::High
    );
    (callbacks.on_model_thinking_level_remove)(&model.provider.0, &model.id);

    (callbacks.on_theme_change)("light");
    if let Some(preview) = &callbacks.on_theme_preview {
        preview("dark");
    }
    (callbacks.on_hide_thinking_block_change)(true);
    (callbacks.on_mermaid_rendering_mode_change)(MermaidRenderingMode::Off);
    (callbacks.on_show_cache_miss_notices_change)(true);
    (callbacks.on_collapse_changelog_change)(true);
    (callbacks.on_enable_install_telemetry_change)(true);
    (callbacks.on_double_escape_action_change)("tree");
    (callbacks.on_tree_filter_mode_change)("user-only");
    (callbacks.on_show_hardware_cursor_change)(true);
    (callbacks.on_editor_padding_x_change)(2);
    (callbacks.on_output_pad_change)(1);
    (callbacks.on_autocomplete_max_visible_change)(9);
    (callbacks.on_quiet_startup_change)(true);
    (callbacks.on_default_project_trust_change)(DefaultProjectTrust::Never);
    (callbacks.on_clear_on_shrink_change)(true);
    (callbacks.on_show_terminal_progress_change)(true);
    (callbacks.on_tui_mode_change)(TuiMode::Fullscreen);
    (callbacks.on_fullscreen_exit_output_change)(FullscreenExitOutput::Transcript);
    (callbacks.on_fullscreen_scrollbar_change)(ScrollViewScrollbar::Always);
    (callbacks.on_fullscreen_copy_on_select_change)(true);
    (callbacks.on_warnings_change)(
        pi_coding_agent::modes::interactive::components::settings_selector::WarningSettings {
            anthropic_extra_usage: false,
        },
    );
    (callbacks.on_cancel)();
    harness.frame.drain_commands().await;

    let guard = settings.lock().expect("settings");
    assert!(guard.get_hide_thinking_block());
    assert_eq!(
        guard.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Off
    );
    assert!(guard.get_show_cache_miss_notices());
    assert!(guard.get_collapse_changelog());
    assert!(guard.get_enable_install_telemetry());
    assert_eq!(guard.get_double_escape_action(), "tree");
    assert_eq!(guard.get_tree_filter_mode(), "user-only");
    assert!(guard.get_show_hardware_cursor());
    assert_eq!(guard.get_editor_padding_x(), 2);
    assert_eq!(guard.get_output_pad(), 1);
    assert_eq!(guard.get_autocomplete_max_visible(), 9);
    assert!(guard.get_quiet_startup());
    assert_eq!(
        guard.get_default_project_trust(),
        DefaultProjectTrust::Never
    );
    assert!(guard.get_clear_on_shrink());
    assert!(guard.get_show_terminal_progress());
    assert_eq!(
        guard.get_fullscreen_exit_output(),
        FullscreenExitOutput::Transcript
    );
    assert_eq!(
        guard.get_fullscreen_scrollbar(),
        ScrollViewScrollbar::Always
    );
    assert!(guard.get_fullscreen_copy_on_select());
    assert_eq!(
        guard
            .get_warnings()
            .get("anthropicExtraUsage")
            .and_then(serde_json::Value::as_bool),
        Some(false)
    );
    drop(guard);
    assert_eq!(harness.frame.current_tui_mode(), TuiMode::Fullscreen);
}

#[tokio::test]
async fn model_and_thinking_selectors_select_persist_and_cancel() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;

    common::interactive::submit(&harness, "/model").await;
    harness.send_input("\x1b");
    common::interactive::submit(&harness, "/model").await;
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    assert!(harness.chat_text().contains("Model:"));
    common::interactive::submit(&harness, "/model").await;
    harness.send_input("\x13");
    harness.frame.drain_commands().await;
    assert!(harness.chat_text().contains("Default model:"));

    common::interactive::submit(&harness, "/thinking").await;
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    common::interactive::submit(&harness, "/thinking").await;
    harness.send_input("\x13");
    harness.frame.drain_commands().await;
    common::interactive::submit(&harness, "/thinking").await;
    harness.send_input("\x1b");

    common::interactive::submit(&harness, "/thinking high").await;
    assert_eq!(
        harness.frame.session().thinking_level(),
        pi_agent_core::types::ThinkingLevel::High
    );
    common::interactive::submit(&harness, "/thinking bogus").await;
    assert!(harness.chat_text().contains("Unknown thinking level"));
}

#[tokio::test]
async fn model_search_arguments_match_exact_and_show_the_selector() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    let model = harness.frame.session().model().expect("model");
    let exact = format!("{}/{}", model.provider.0, model.id);

    common::interactive::submit(&harness, &format!("/model {exact}")).await;
    assert_eq!(harness.frame.session().model().expect("model").id, model.id);

    common::interactive::submit(&harness, "/model def-not-a-model").await;
    assert!(harness.frame.extension_selector().is_none());
    harness.send_input("\x1b");
}

#[tokio::test]
async fn thinking_and_model_cycles_and_daxnuts_row_are_reachable_directly() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;

    harness.frame.cycle_thinking_level();
    for direction in [
        pi_coding_agent::core::agent_session::CycleDirection::Forward,
        pi_coding_agent::core::agent_session::CycleDirection::Backward,
    ] {
        harness.frame.cycle_model(direction).await;
    }
    harness.frame.update_available_provider_count().await;

    let mut daxnuts = common::model_layer::model("opencode", "kimi-k2.5");
    daxnuts.name = "Kimi K2.5".to_owned();
    let before = harness.frame.chat_container().children.borrow().len();
    harness.frame.check_daxnuts_easter_egg(&daxnuts);
    assert!(harness.frame.chat_container().children.borrow().len() > before);
}

#[tokio::test]
async fn scoped_models_selector_toggles_enable_all_clear_all_and_persists() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;

    common::interactive::submit(&harness, "/scoped-models").await;
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    harness.send_input("\x01");
    harness.frame.drain_commands().await;
    harness.send_input("\x18");
    harness.frame.drain_commands().await;
    harness.send_input("\x13");
    harness.frame.drain_commands().await;
    harness.send_input("\x1b");
    common::interactive::submit(&harness, "/scoped-models").await;
    harness.send_input("\x1b");
}

#[tokio::test]
async fn scoped_model_application_and_persistence_cover_every_branch() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    let models = harness
        .frame
        .session()
        .model_runtime()
        .get_available_snapshot();
    assert!(models.len() >= 2, "the harness registers two faux models");
    let all: Vec<String> = models
        .iter()
        .map(|model| format!("{}/{}", model.provider.0, model.id))
        .collect();

    harness
        .frame
        .apply_scoped_models(Some(vec![all[0].clone()]));
    assert_eq!(harness.frame.session().scoped_models().len(), 1);
    harness.frame.apply_scoped_models(Some(all.clone()));
    assert!(harness.frame.session().scoped_models().is_empty());
    harness
        .frame
        .apply_scoped_models(Some(vec!["nope/nope".to_owned()]));
    assert!(harness.frame.session().scoped_models().is_empty());
    harness.frame.apply_scoped_models(None);

    harness
        .frame
        .persist_scoped_models(Some(vec![all[0].clone()]));
    assert!(harness.chat_text().contains("Model selection saved"));
    harness.frame.persist_scoped_models(Some(all));
    harness.frame.persist_scoped_models(None);
}

#[tokio::test]
async fn trust_selector_saves_a_decision_and_cancels() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;

    common::interactive::submit(&harness, "/trust").await;
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    assert!(harness.chat_text().contains("Saved trust decision"));

    common::interactive::submit(&harness, "/trust").await;
    harness.send_input("\x1b");
}

#[tokio::test]
async fn implicit_reload_trust_is_saved_once_for_the_armed_cwd() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().join("project").display().to_string();
    std::fs::create_dir_all(format!("{cwd}/.pi")).expect("project config");
    std::fs::write(format!("{cwd}/.pi/settings.json"), "{}").expect("project settings");

    let harness = Harness::new(HarnessOptions {
        cwd_override: Some(cwd.clone()),
        frame_options:
            pi_coding_agent::modes::interactive::interactive_mode::InteractiveModeOptions {
                auto_trust_on_reload_cwd: Some(cwd),
                ..Default::default()
            },
        ..HarnessOptions::default()
    })
    .await;
    assert!(
        harness
            .frame
            .maybe_save_implicit_project_trust_after_reload()
    );
    assert!(
        !harness
            .frame
            .maybe_save_implicit_project_trust_after_reload()
    );
}

#[tokio::test]
async fn implicit_reload_trust_skips_unarmed_or_untrusted_projects() {
    let harness = Harness::new(HarnessOptions {
        frame_options:
            pi_coding_agent::modes::interactive::interactive_mode::InteractiveModeOptions {
                auto_trust_on_reload_cwd: Some("/definitely/not/the/cwd".to_owned()),
                ..Default::default()
            },
        ..HarnessOptions::default()
    })
    .await;
    assert!(
        !harness
            .frame
            .maybe_save_implicit_project_trust_after_reload()
    );
    harness
        .frame
        .session()
        .settings_manager()
        .lock()
        .expect("settings")
        .set_project_trusted(false);
    assert!(
        !harness
            .frame
            .maybe_save_implicit_project_trust_after_reload()
    );
}

#[tokio::test]
async fn fullscreen_scrollbar_setting_applies_without_a_transcript() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.apply_fullscreen_scrollbar_setting();
}

#[tokio::test]
async fn tree_selector_navigates_to_an_entry_and_reports_the_leaf() {
    let harness = common::interactive::harness_with(HarnessOptions {
        messages: common::interactive::conversation_messages(),
        settings: {
            let mut settings = Settings::new();
            settings.insert(
                "branchSummary".to_owned(),
                serde_json::json!({ "skipPrompt": true }),
            );
            settings
        },
        ..HarnessOptions::default()
    })
    .await;

    common::interactive::submit(&harness, "/tree").await;
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    assert!(harness.chat_text().contains("Navigated to selected point"));

    let Some(leaf) = harness.frame.session().leaf_id() else {
        return;
    };
    harness.frame.navigate_tree(&leaf).await;
    assert!(harness.chat_text().contains("Already at this point"));
}

#[tokio::test]
async fn tree_selector_prompt_awaits_the_summary_choice() {
    let harness = common::interactive::harness_with(HarnessOptions {
        messages: common::interactive::conversation_messages(),
        ..HarnessOptions::default()
    })
    .await;

    common::interactive::submit(&harness, "/tree").await;
    harness.send_input("\r");
    let mut drain = Box::pin(harness.frame.drain_commands());
    tokio::select! {
        () = &mut drain => {}
        () = tokio::time::sleep(Duration::from_millis(40)) => {
            harness.send_input("\r");
            drain.await;
        }
    }
    assert!(
        !harness.chat_text().is_empty(),
        "the navigation outcome is reported"
    );

    common::interactive::submit(&harness, "/tree").await;
    harness.send_input("\r");
    let mut drain = Box::pin(harness.frame.drain_commands());
    tokio::select! {
        () = &mut drain => {}
        () = tokio::time::sleep(Duration::from_millis(40)) => {
            harness.send_input("\x1b");
            drain.await;
        }
    }
    harness.send_input("\x1b");

    common::interactive::submit(&harness, "/tree").await;
    harness.send_input("\r");
    let mut drain = Box::pin(harness.frame.drain_commands());
    tokio::select! {
        () = &mut drain => {}
        () = tokio::time::sleep(Duration::from_millis(40)) => {
            harness.send_input("\x1b[B");
            harness.send_input("\x1b[B");
            harness.send_input("\r");
            tokio::time::sleep(Duration::from_millis(40)).await;
            harness.send_input("notes");
            harness.send_input("\r");
            drain.await;
        }
    }
}

#[tokio::test]
async fn user_message_selector_forks_and_cancels() {
    let harness = common::interactive::harness_with(HarnessOptions {
        messages: common::interactive::conversation_messages(),
        ..HarnessOptions::default()
    })
    .await;

    common::interactive::submit(&harness, "/fork").await;
    harness.send_input("\x1b");
    common::interactive::submit(&harness, "/fork").await;
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    assert!(!harness.chat_text().contains("No messages to fork from"));
}

#[tokio::test]
async fn fork_command_without_messages_and_clone_at_the_leaf() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    common::interactive::submit(&harness, "/fork").await;
    assert!(harness.chat_text().contains("No messages to fork from"));
    common::interactive::submit(&harness, "/clone").await;
    assert!(!harness.chat_text().is_empty());

    let harness = common::interactive::harness_with(HarnessOptions {
        messages: common::interactive::conversation_messages(),
        ..HarnessOptions::default()
    })
    .await;
    common::interactive::submit(&harness, "/clone").await;
    assert!(harness.chat_text().contains("Cloned to new session"));
}

#[tokio::test]
async fn session_selector_switches_and_cancels() {
    let harness = common::interactive::harness_with(HarnessOptions {
        persisted: true,
        messages: common::interactive::conversation_messages(),
        ..HarnessOptions::default()
    })
    .await;

    common::interactive::submit(&harness, "/resume").await;
    harness.send_input("\x1b");
    common::interactive::submit(&harness, "/resume").await;
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn switching_to_a_missing_session_reports_an_error() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    let result = harness
        .frame
        .switch_session("/definitely/missing/session.jsonl")
        .await;
    assert!(result.is_err());
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn settings_selector_mounts_and_cancels() {
    let harness = common::interactive::harness_with(HarnessOptions::default()).await;
    common::interactive::submit(&harness, "/settings").await;
    harness.send_input("\x1b");
    assert!(harness.chat_text().is_empty());
}

#[tokio::test]
async fn user_message_text_and_queue_helpers_cover_both_message_shapes() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let user = common::agent_session::user_message("body", 1);
    let assistant =
        pi_ai::types::Message::Assistant(common::agent_session::assistant_message("reply", 2));
    let pi_agent_core::types::AgentMessage::Standard(user) = user else {
        return;
    };
    assert_eq!(harness.frame.get_user_message_text(&user), "body");
    assert_eq!(harness.frame.get_user_message_text(&assistant), "");
    assert_eq!(
        harness.frame.get_all_queued_messages(),
        (Vec::new(), Vec::new())
    );
    assert_eq!(harness.frame.clear_all_queues(), (Vec::new(), Vec::new()));
}

#[tokio::test]
async fn fatal_runtime_errors_report_and_stop_the_frame() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.tui().start();
    harness
        .frame
        .handle_fatal_runtime_error("Fatal", "boom")
        .await;
    assert!(harness.chat_text().contains("Fatal: boom"));
    assert!(harness.frame.tui().is_stopped());
    harness.frame.shutdown().await;
}

fn api_key_auth(key: &str) -> pi_coding_agent::auth_storage::AuthStorageData {
    let mut data = pi_coding_agent::auth_storage::AuthStorageData::new();
    data.insert(
        "anthropic".to_owned(),
        serde_json::json!({ "type": "api_key", "key": key }),
    );
    data
}

#[tokio::test]
async fn export_html_jsonl_and_failure_paths_report_status() {
    let harness = Harness::new(HarnessOptions {
        persisted: true,
        messages: common::interactive::conversation_messages(),
        ..HarnessOptions::default()
    })
    .await;
    let root = harness.temp.path().join("exports");
    std::fs::create_dir_all(&root).expect("exports dir");
    let html = root.join("session.html");
    let jsonl = root.join("session.jsonl");

    common::interactive::submit(&harness, &format!("/export {}", html.display())).await;
    assert!(harness.chat_text().contains("Session exported to:"));
    common::interactive::submit(&harness, &format!("/export {}", jsonl.display())).await;
    assert!(html.exists());
    assert!(jsonl.exists());

    common::interactive::submit(&harness, "/export /definitely/missing/out.html").await;
    assert!(harness.chat_text().contains("Failed to export session"));
}

#[tokio::test]
async fn name_command_reports_usage_sets_and_reads_the_name() {
    let harness = Harness::new(HarnessOptions::default()).await;
    common::interactive::submit(&harness, "/name").await;
    assert!(harness.chat_text().contains("Usage: /name <name>"));
    common::interactive::submit(&harness, "/name boundary").await;
    assert!(harness.chat_text().contains("Session name set: boundary"));
    common::interactive::submit(&harness, "/name").await;
    assert!(harness.chat_text().contains("Session name: boundary"));
}

#[tokio::test]
async fn session_info_renders_cost_breakdown_and_usage() {
    let usage = pi_ai::types::Usage {
        input: 10,
        output: 20,
        cache_read: 5,
        cache_write: 1,
        total_tokens: 36,
        cost: pi_ai::types::UsageCost {
            total: 0.25,
            ..pi_ai::types::UsageCost::default()
        },
        ..pi_ai::types::Usage::default()
    };
    let harness = Harness::new(HarnessOptions {
        messages: vec![
            common::agent_session::user_message("prompt", 1),
            pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(
                common::agent_session::assistant_with_usage("reply", usage, 2),
            )),
        ],
        ..HarnessOptions::default()
    })
    .await;
    common::interactive::submit(&harness, "/session").await;
    let text = harness.chat_text();
    assert!(harness.frame.session().session_stats().cost > 0.0, "{text}");
    assert!(text.contains("Session Info"));
}

#[tokio::test]
async fn copy_command_copies_the_last_assistant_message() {
    let harness = Harness::new(HarnessOptions {
        messages: common::interactive::conversation_messages(),
        ..HarnessOptions::default()
    })
    .await;
    let clipboard = common::interactive::ClipboardGuard::new(None);
    common::interactive::submit(&harness, "/copy").await;
    assert_eq!(clipboard.copied(), vec!["reply".to_owned()]);
    assert!(harness.chat_text().contains("Copied last agent message"));
}

#[tokio::test]
async fn login_and_logout_selectors_cover_selection_and_cancel() {
    let harness = Harness::new(HarnessOptions {
        auth: api_key_auth("sk-ant-api03-test"),
        ..HarnessOptions::default()
    })
    .await;
    harness.start_tui();

    common::interactive::submit(&harness, "/logout").await;
    harness.send_input("\x1b");
    common::interactive::submit(&harness, "/logout").await;
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    assert!(harness.chat_text().contains("Removed stored credential"));

    common::interactive::submit(&harness, "/login").await;
    harness.send_input("\x1b");
    common::interactive::submit(&harness, "/login not-a-provider").await;
    harness.send_input("\x1b");

    let empty = Harness::new(HarnessOptions::default()).await;
    common::interactive::submit(&empty, "/logout").await;
    assert!(
        harness
            .frame
            .get_login_provider_options(None)
            .iter()
            .any(|provider| provider.id == "faux"),
        "the fixture provider offers a login option"
    );
}

#[tokio::test]
async fn login_command_starts_a_provider_and_finishes_on_the_frame_command() {
    use pi_coding_agent::modes::interactive::components::oauth_selector::AuthSelectorProvider;

    let harness = Harness::new(HarnessOptions::default()).await;
    harness.start_tui();
    common::interactive::submit(&harness, "/login faux").await;
    harness.send_input("\x1b");
    harness
        .frame
        .start_provider_login(AuthSelectorProvider {
            id: "faux".to_owned(),
            name: "Faux".to_owned(),
            auth_type: pi_ai::auth::types::AuthType::ApiKey,
            method: None,
            status: None,
        })
        .await;
    assert!(harness.frame.login_dialog().is_some());
    tokio::time::sleep(Duration::from_millis(30)).await;
    harness.frame.drain_commands().await;
    harness
        .frame
        .finish_login("faux", pi_ai::auth::types::AuthType::OAuth, Ok(()))
        .await;
    assert!(harness.chat_text().contains("Logged in to faux"));
    harness
        .frame
        .finish_login(
            "faux",
            pi_ai::auth::types::AuthType::ApiKey,
            Err("denied".to_owned()),
        )
        .await;
    assert!(harness.chat_text().contains("failed: denied"));
}

#[tokio::test]
async fn logout_failure_reports_an_error() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.logout_provider("no-such-provider").await;
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn auth_prompts_round_trip_every_kind_through_the_login_dialog() {
    use pi_ai::auth::types::{AuthPrompt, AuthPromptKind, AuthPromptOption};
    use pi_coding_agent::modes::interactive::components::oauth_selector::AuthSelectorProvider;

    let harness = Harness::new(HarnessOptions::default()).await;
    harness.start_tui();
    let ui = harness.frame.ui_context();

    let (sender, receiver) = tokio::sync::oneshot::channel();
    ui.send_auth_prompt(
        AuthPrompt {
            signal: None,
            kind: AuthPromptKind::Text {
                message: "Token?".to_owned(),
                placeholder: None,
            },
        },
        sender,
    );
    harness.frame.drain_ui_requests();
    assert_eq!(
        receiver.await.expect("prompt"),
        Err("Login cancelled".to_owned())
    );

    harness
        .frame
        .start_provider_login(AuthSelectorProvider {
            id: "faux".to_owned(),
            name: "Faux".to_owned(),
            auth_type: pi_ai::auth::types::AuthType::ApiKey,
            method: None,
            status: None,
        })
        .await;
    assert!(harness.frame.extension_editor().is_none());

    for (kind, answer) in [
        (
            AuthPromptKind::Text {
                message: "Text".to_owned(),
                placeholder: None,
            },
            "typed",
        ),
        (
            AuthPromptKind::Secret {
                message: "Secret".to_owned(),
                placeholder: None,
            },
            "hunter2",
        ),
        (
            AuthPromptKind::ManualCode {
                message: "Code".to_owned(),
                placeholder: None,
            },
            "123456",
        ),
    ] {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        ui.send_auth_prompt(AuthPrompt { signal: None, kind }, sender);
        harness.frame.drain_ui_requests();
        harness.send_input(answer);
        harness.send_input("\r");
        assert_eq!(receiver.await.expect("prompt"), Ok(answer.to_owned()));
    }

    let (sender, receiver) = tokio::sync::oneshot::channel();
    ui.send_auth_prompt(
        AuthPrompt {
            signal: None,
            kind: AuthPromptKind::Select {
                message: "Pick".to_owned(),
                options: vec![AuthPromptOption {
                    id: "oauth".to_owned(),
                    label: "OAuth".to_owned(),
                    description: None,
                }],
            },
        },
        sender,
    );
    harness.frame.drain_ui_requests();
    harness.send_input("\r");
    assert_eq!(receiver.await.expect("prompt"), Ok("oauth".to_owned()));

    let cancelled = tokio_util::sync::CancellationToken::new();
    cancelled.cancel();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    ui.send_auth_prompt(
        AuthPrompt {
            signal: Some(cancelled),
            kind: AuthPromptKind::Text {
                message: "Cancelled".to_owned(),
                placeholder: None,
            },
        },
        sender,
    );
    harness.frame.drain_ui_requests();
    assert_eq!(
        receiver.await.expect("prompt"),
        Err("Login cancelled".to_owned())
    );
}

#[tokio::test]
async fn auth_notifications_render_every_event() {
    use pi_ai::auth::types::{AuthEvent, AuthInfoLink};
    use pi_coding_agent::modes::interactive::components::oauth_selector::AuthSelectorProvider;

    let harness = Harness::new(HarnessOptions::default()).await;
    harness.start_tui();
    harness
        .frame
        .start_provider_login(AuthSelectorProvider {
            id: "faux".to_owned(),
            name: "Faux".to_owned(),
            auth_type: pi_ai::auth::types::AuthType::ApiKey,
            method: None,
            status: None,
        })
        .await;
    let ui = harness.frame.ui_context();
    let events = [
        AuthEvent::AuthUrl {
            url: "https://example.test/auth".to_owned(),
            instructions: Some("visit".to_owned()),
        },
        AuthEvent::AuthUrl {
            url: "https://example.test/auth".to_owned(),
            instructions: None,
        },
        AuthEvent::DeviceCode {
            user_code: "ABCD".to_owned(),
            verification_uri: "https://example.test/device".to_owned(),
            interval_seconds: Some(5),
            expires_in_seconds: Some(60),
        },
        AuthEvent::Info {
            message: "info".to_owned(),
            links: Some(vec![AuthInfoLink {
                url: "https://example.test/info".to_owned(),
                label: Some("info".to_owned()),
            }]),
        },
        AuthEvent::Info {
            message: "plain".to_owned(),
            links: None,
        },
        AuthEvent::Progress {
            message: "working".to_owned(),
        },
    ];
    for event in events {
        ui.send_auth_notify(event);
        harness.frame.drain_ui_requests();
    }
    assert!(harness.frame.login_dialog().is_some());
}
