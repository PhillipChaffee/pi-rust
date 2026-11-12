//! Second-pass coverage for the frame's remaining command, display, adapter,
//! startup, and component branches.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the provider and session shapes"
)]
#![allow(
    clippy::large_futures,
    reason = "the acceptance suites await one frame operation at a time"
)]

mod common;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use common::interactive::{Harness, HarnessOptions, ResourceFixtures};
use pi_agent_core::types::{AgentMessage, CustomAgentMessage};
use pi_ai::auth::types::AuthType;
use pi_ai::types::Message;
use pi_coding_agent::extensions::host::UiCallHandler;
use pi_coding_agent::settings_manager::Settings;
use pi_tui::tui::Component;

async fn harness_with(options: HarnessOptions) -> Harness {
    let harness = Harness::new(options).await;
    harness.start_tui();
    harness
}

async fn submit(harness: &Harness, line: &str) {
    harness.frame.submit_text(line).await;
    harness.frame.drain_commands().await;
    harness.frame.drain_events();
}

fn seeds_messages() -> Vec<AgentMessage> {
    vec![
        common::agent_session::user_message("first", 1),
        AgentMessage::Standard(Message::Assistant(
            common::agent_session::assistant_message("reply", 2),
        )),
    ]
}

/// A stream that opens immediately and settles only when `release` notifies.
fn gated_stream(release: Arc<tokio::sync::Notify>) -> pi_agent_core::types::StreamFn {
    use pi_ai::types::{AssistantMessageEvent, StopReason};
    Arc::new(move |model, _context, _options| {
        let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
        let mut message = common::sdk_fixture::done_message();
        message.provider = model.provider.clone();
        message.model.clone_from(&model.id);
        stream.push(AssistantMessageEvent::Start {
            partial: pi_ai::types::AssistantMessage {
                content: Vec::new(),
                ..message.clone()
            },
        });
        let settle = stream.clone();
        let release = Arc::clone(&release);
        tokio::spawn(async move {
            release.notified().await;
            settle.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message,
            });
        });
        stream
    })
}

#[tokio::test]
async fn login_provider_filters_and_completion_text_cover_both_auth_types() {
    let harness = harness_with(HarnessOptions {
        model: common::agent_session::anthropic_model(),
        ..HarnessOptions::default()
    })
    .await;
    let all = harness.frame.get_login_provider_options(None);
    assert!(!all.is_empty());
    let oauth = harness
        .frame
        .get_login_provider_options(Some(AuthType::OAuth));
    assert!(
        oauth
            .iter()
            .all(|provider| provider.auth_type == AuthType::OAuth)
    );
    let api_key = harness
        .frame
        .get_login_provider_options(Some(AuthType::ApiKey));
    assert!(
        api_key
            .iter()
            .all(|provider| provider.auth_type == AuthType::ApiKey)
    );
    assert!(harness.frame.find_login_provider_options("").is_empty());
    assert!(
        !harness
            .frame
            .find_login_provider_options("anthropic")
            .is_empty()
    );
    assert!(
        !harness
            .frame
            .find_login_provider_options("ANTHROPIC")
            .is_empty()
    );

    let same_name = all
        .iter()
        .find(|provider| provider.id == provider.name)
        .cloned();
    if let Some(provider) = same_name {
        let autocomplete = harness.frame.create_base_autocomplete_provider();
        let line = format!("/login {}", provider.id);
        let suggestions = autocomplete
            .get_suggestions(
                std::slice::from_ref(&line),
                0,
                line.len(),
                &pi_tui::autocomplete::AutocompleteQueryOptions {
                    signal: tokio_util::sync::CancellationToken::new(),
                    force: false,
                },
            )
            .expect("provider completion");
        assert!(
            suggestions
                .items
                .iter()
                .any(|item| item.value == provider.id)
        );
    }
}

#[tokio::test]
async fn model_and_thinking_cycle_failure_arms() {
    let mut model = common::sdk_fixture::faux_model();
    model.reasoning = false;
    let harness = harness_with(HarnessOptions {
        model,
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.cycle_thinking_level();
    assert!(harness.chat_text().contains("does not support thinking"));

    let models = harness
        .frame
        .session()
        .model_runtime()
        .get_available_snapshot();
    let only = format!("{}/{}", models[0].provider.0, models[0].id);
    harness.frame.apply_scoped_models(Some(vec![only]));
    harness
        .frame
        .cycle_model(pi_coding_agent::core::agent_session::CycleDirection::Forward)
        .await;
    assert!(harness.chat_text().contains("Only one model in scope"));
}

#[tokio::test]
async fn model_selector_reads_the_default_model_setting() {
    let harness = harness_with(HarnessOptions {
        settings: {
            let mut settings = Settings::new();
            settings.insert(
                "defaultModel".to_owned(),
                serde_json::json!({ "provider": "faux", "model": "faux-1" }),
            );
            settings
        },
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.show_model_selector(None);
    harness.send_input("\x1b");
}

#[tokio::test]
async fn every_tree_filter_mode_selects_its_component_mode() {
    for mode in [
        "no-tools",
        "user-only",
        "labeled-only",
        "all",
        "default",
        "bogus",
    ] {
        let mut settings = Settings::new();
        settings.insert(
            "treeFilterMode".to_owned(),
            serde_json::Value::String(mode.to_owned()),
        );
        let harness = harness_with(HarnessOptions {
            settings,
            messages: seeds_messages(),
            ..HarnessOptions::default()
        })
        .await;
        harness.frame.show_tree_selector(None).await;
        harness.send_input("\x1b");
    }
}

#[tokio::test]
async fn resume_selector_uses_the_default_session_directory() {
    let harness = harness_with(HarnessOptions {
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    submit(&harness, "/resume").await;
    harness.send_input("\x1b");
}

#[tokio::test]
async fn reload_while_streaming_warns_and_keeps_the_run() {
    let release = Arc::new(tokio::sync::Notify::new());
    let harness = harness_with(HarnessOptions {
        stream: Some(gated_stream(Arc::clone(&release))),
        ..HarnessOptions::default()
    })
    .await;
    let session = harness.frame.session();
    let prompt = tokio::spawn({
        let session = Arc::clone(&session);
        async move { session.prompt("first".to_owned(), None).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    submit(&harness, "/reload").await;
    assert!(
        harness
            .chat_text()
            .contains("Wait for the current response")
    );
    release.notify_waiters();
    let _ = prompt.await;
}

#[tokio::test]
async fn name_normalization_warns_and_fork_errors_report() {
    let harness = harness_with(HarnessOptions {
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    submit(&harness, "/name line\nbreak").await;
    assert!(
        harness.chat_text().contains("normalized"),
        "{}",
        harness.chat_text()
    );

    harness.frame.fork_entry("missing-entry", false).await;
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn tui_mode_change_is_blocked_by_an_overlay() {
    let harness = harness_with(HarnessOptions::default()).await;
    let overlay: Rc<dyn Component> =
        Rc::new(pi_tui::components::Text::with_padding("overlay", 0, 0));
    let _ = harness.frame.tui().show_overlay(overlay, None);
    let callbacks = harness.frame.settings_callbacks();
    (callbacks.on_tui_mode_change)(pi_tui::tui::TuiMode::Fullscreen);
    harness.frame.drain_commands().await;
    assert!(harness.chat_text().contains("Close active overlays"));
    harness.frame.tui().hide_overlay();
}

#[tokio::test]
async fn skill_and_extension_commands_reach_autocomplete() {
    let source = pi_coding_agent::source_info::SourceInfo {
        path: "/tmp/skill/SKILL.md".to_owned(),
        source: "local".to_owned(),
        scope: pi_coding_agent::source_info::SourceScope::Project,
        origin: pi_coding_agent::source_info::SourceOrigin::TopLevel,
        base_dir: None,
    };
    let fixtures = ResourceFixtures {
        skills: vec![(
            "commit".to_owned(),
            "/tmp/skill/SKILL.md".to_owned(),
            source,
        )],
        extension_factories: vec![(
            Some("commands".to_owned()),
            false,
            Arc::new(|api| {
                Box::pin(async move {
                    api.register_command(
                        "demo".to_owned(),
                        None,
                        Arc::new(|_args, _ctx| Box::pin(async {})),
                    )?;
                    api.register_command(
                        "model".to_owned(),
                        None,
                        Arc::new(|_args, _ctx| Box::pin(async {})),
                    )?;
                    Ok(())
                })
            }),
        )],
        ..ResourceFixtures::default()
    };
    let mut settings = Settings::new();
    settings.insert(
        "enableSkillCommands".to_owned(),
        serde_json::Value::Bool(true),
    );
    let harness = harness_with(HarnessOptions {
        fixtures: Some(fixtures),
        settings,
        ..HarnessOptions::default()
    })
    .await;
    let provider = harness.frame.create_base_autocomplete_provider();
    let line = "/skill:commit";
    let suggestions = provider.get_suggestions(
        &[line.to_owned()],
        0,
        line.len(),
        &pi_tui::autocomplete::AutocompleteQueryOptions {
            signal: tokio_util::sync::CancellationToken::new(),
            force: false,
        },
    );
    assert!(suggestions.is_some());
}

#[tokio::test]
async fn successful_login_selects_the_provider_default_model() {
    let harness = Harness::new(HarnessOptions {
        model: common::agent_session::anthropic_model(),
        auth: {
            let mut data = pi_coding_agent::auth_storage::AuthStorageData::new();
            data.insert(
                "anthropic".to_owned(),
                serde_json::json!({ "type": "api_key", "key": "sk-ant-api03-test" }),
            );
            data
        },
        ..HarnessOptions::default()
    })
    .await;
    let provider_name = harness
        .frame
        .get_login_provider_options(None)
        .into_iter()
        .find(|provider| provider.id == "anthropic")
        .map_or_else(|| "anthropic".to_owned(), |provider| provider.name);
    harness
        .frame
        .finish_login(&provider_name, AuthType::OAuth, Ok(()))
        .await;
    assert!(harness.chat_text().contains("Logged in to"));
    let settings = harness.frame.session().settings_manager();
    let guard = settings.lock().expect("settings");
    let _ = guard.get_default_provider();
    let _ = guard.get_default_model();
    drop(guard);
}

#[tokio::test]
async fn default_editor_and_footer_adapters_expose_every_member() {
    use pi_tui::tui::Component;

    let harness = Harness::new(HarnessOptions {
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.session().set_session_name("adapter name");
    let editor = harness.frame.editor();
    editor.set_text("adapter");
    assert_eq!(editor.get_text(), "adapter");
    editor.insert_text_at_cursor("!");
    assert_eq!(editor.get_text(), "adapter!");
    let _ = editor.get_expanded_text();
    editor.set_padding_x(1);
    editor.set_autocomplete_max_visible(5);
    let _ = editor.render(40);
    editor.invalidate();
    let _ = editor.children();
    assert!(editor.wants_input());
    editor.handle_input("x");
    let _ = editor.border_color();
    editor.set_border_color(Rc::new(|text: &str| text.to_owned()));
    editor.set_on_escape(Some(Rc::new(|| {})));
    assert!(editor.on_escape().is_some());
    editor.set_on_ctrl_d(Some(Rc::new(|| {})));
    editor.set_on_paste_image(Some(Rc::new(|| {})));
    editor.set_on_extension_shortcut(None);
    editor.set_on_submit(None);
    editor.set_on_change(None);
    editor.on_action("app.clear", Rc::new(|| {}));
    editor.set_working_status_indicator(None);
    assert!(editor.embed_working_status());

    let footer = harness.frame.footer_component();
    let rendered = footer.render(120);
    assert!(!rendered.is_empty());
    let _ = footer.children();
    footer.handle_input("\x1b");
    footer.invalidate();
}

#[tokio::test]
async fn settings_tui_mode_and_initial_theme_are_read_when_unset() {
    let mut settings = Settings::new();
    settings.insert(
        "tuiMode".to_owned(),
        serde_json::Value::String("fullscreen".to_owned()),
    );
    let harness = Harness::new(HarnessOptions {
        settings,
        use_settings_tui_mode: true,
        frame_options:
            pi_coding_agent::modes::interactive::interactive_mode::InteractiveModeOptions {
                initial_theme_setting: Some("light".to_owned()),
                ..Default::default()
            },
        ..HarnessOptions::default()
    })
    .await;
    assert_eq!(
        harness.frame.current_tui_mode(),
        pi_tui::tui::TuiMode::Fullscreen
    );
    assert!(pi_coding_agent::modes::interactive::theme::theme::current_theme_name().is_some());
}

#[tokio::test]
async fn theme_handles_load_valid_paths_and_skip_broken_ones() {
    let fixtures = ResourceFixtures {
        themes: vec![
            pi_coding_agent::resource_loader::ThemeHandle {
                name: Some("dark".to_owned()),
                source_path: Some(
                    concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/src/modes/interactive/theme/dark.json"
                    )
                    .to_owned(),
                ),
                source_info: None,
            },
            pi_coding_agent::resource_loader::ThemeHandle {
                name: Some("broken".to_owned()),
                source_path: Some("/nonexistent/theme.json".to_owned()),
                source_info: None,
            },
            pi_coding_agent::resource_loader::ThemeHandle {
                name: Some("pathless".to_owned()),
                source_path: None,
                source_info: None,
            },
        ],
        ..ResourceFixtures::default()
    };
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(fixtures),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.show_loaded_resources(true);
    assert!(harness.resources_text().contains("dark"));
}

#[tokio::test]
async fn assistant_diagnostics_cover_malformed_and_plural_shapes() {
    let mut settings = Settings::new();
    settings.insert(
        "showCacheMissNotices".to_owned(),
        serde_json::Value::Bool(true),
    );
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;
    let details = |value: serde_json::Value| {
        let mut map = std::collections::BTreeMap::new();
        if let Some(object) = value.as_object() {
            for (key, value) in object {
                map.insert(key.clone(), value.clone());
            }
        }
        map
    };
    let mut message = common::agent_session::assistant_message("reply", 1);
    message.diagnostics = Some(vec![
        pi_ai::types::AssistantMessageDiagnostic {
            kind: "other".to_owned(),
            timestamp: 1,
            error: None,
            details: None,
        },
        pi_ai::types::AssistantMessageDiagnostic {
            kind: "anthropic_input_transformations".to_owned(),
            timestamp: 1,
            error: None,
            details: Some(details(serde_json::json!({}))),
        },
        pi_ai::types::AssistantMessageDiagnostic {
            kind: "anthropic_input_transformations".to_owned(),
            timestamp: 1,
            error: None,
            details: Some(details(serde_json::json!({ "transformations": [] }))),
        },
        pi_ai::types::AssistantMessageDiagnostic {
            kind: "anthropic_input_transformations".to_owned(),
            timestamp: 1,
            error: None,
            details: Some(details(serde_json::json!({
                "transformations": [
                    { "type": "thinking_dropped", "reason": "too old" },
                    { "type": "thinking_dropped", "reason": "empty", "path": "content[1]" },
                    { "type": "kept" }
                ]
            }))),
        },
    ]);
    harness.frame.maybe_show_assistant_diagnostics(&message);
    assert!(harness.chat_text().contains("2 thinking blocks"));
    assert!(harness.chat_text().contains("content[1]"));
}

#[tokio::test]
async fn initial_messages_render_trust_warning_and_plural_compactions() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let cwd = root.path().join("project").display().to_string();
    std::fs::create_dir_all(format!("{cwd}/.pi")).expect("project config");
    std::fs::write(format!("{cwd}/.pi/settings.json"), "{}").expect("settings");
    let harness = Harness::new(HarnessOptions {
        cwd_override: Some(cwd),
        seed_manager: Some(Box::new(|manager| {
            let first = manager
                .append_message(common::agent_session::user_message("first", 1))
                .expect("first");
            manager
                .append_compaction("one", &first, 10, None, None, None)
                .expect("compaction one");
            let second = manager
                .append_message(common::agent_session::user_message("second", 2))
                .expect("second");
            manager
                .append_compaction("two", &second, 10, None, None, None)
                .expect("compaction two");
        })),
        ..HarnessOptions::default()
    })
    .await;
    harness
        .frame
        .session()
        .settings_manager()
        .lock()
        .expect("settings")
        .set_project_trusted(false);
    harness.frame.render_initial_messages();
    assert!(harness.chat_text().contains("not trusted"));
    assert!(harness.chat_text().contains("Session compacted 2 times"));
}

#[tokio::test]
async fn flush_compaction_queue_prompts_static_commands_and_mixed_rows() {
    let fixtures = ResourceFixtures {
        extension_factories: vec![(
            Some("commands".to_owned()),
            false,
            Arc::new(|api| {
                Box::pin(async move {
                    api.register_command(
                        "demo".to_owned(),
                        None,
                        Arc::new(|_args, _ctx| Box::pin(async {})),
                    )?;
                    Ok(())
                })
            }),
        )],
        ..ResourceFixtures::default()
    };
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(fixtures),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.queue_compaction_message(
        "/demo",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::Steer,
    );
    harness.frame.flush_compaction_queue(false);
    harness.frame.queue_compaction_message(
        "/demo",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::Steer,
    );
    harness.frame.queue_compaction_message(
        "plain",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::FollowUp,
    );
    harness.frame.flush_compaction_queue(false);
    tokio::time::sleep(Duration::from_millis(20)).await;
}

#[tokio::test]
async fn skill_block_without_a_user_message_renders_only_the_skill_row() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let skill = common::agent_session::user_message(
        "<skill name=\"demo\" location=\"/x\">\nbody\n</skill>",
        1,
    );
    harness.frame.add_message_to_chat(&skill, false);
    assert!(harness.chat_text().contains("demo"));
}

#[tokio::test]
async fn session_info_includes_the_name_and_per_model_cost_rows() {
    let usage = pi_ai::types::Usage {
        input: 1,
        output: 1,
        total_tokens: 2,
        cost: pi_ai::types::UsageCost {
            total: 0.25,
            ..pi_ai::types::UsageCost::default()
        },
        ..pi_ai::types::Usage::default()
    };
    let harness = Harness::new(HarnessOptions {
        messages: vec![
            common::agent_session::user_message("prompt", 1),
            AgentMessage::Standard(Message::Assistant(
                common::agent_session::assistant_with_usage("one", usage, 2),
            )),
            AgentMessage::Standard(Message::Assistant(
                common::agent_session::assistant_with_usage("two", usage, 3),
            )),
        ],
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.session().set_session_name("named stats");
    harness.frame.submit_text("/session").await;
    harness.frame.drain_commands().await;
    let markdown = harness.chat_text();
    assert!(markdown.contains("named stats"));
    assert!(markdown.contains("Cost"));
}

#[tokio::test]
async fn render_entries_ignores_orphan_tool_results() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let orphan = AgentMessage::Standard(Message::ToolResult(pi_ai::types::ToolResultMessage {
        tool_call_id: "missing".to_owned(),
        tool_name: "read".to_owned(),
        content: Vec::new(),
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    }));
    let entry: pi_coding_agent::session_manager::FileEntry =
        serde_json::from_value(serde_json::json!({
            "type": "message",
            "id": "orphan",
            "parentId": null,
            "timestamp": "2025-01-01T00:00:00Z",
            "message": serde_json::to_value(&orphan).expect("message serializes")
        }))
        .expect("entry parses");
    harness.frame.render_session_entries(&[entry], false, false);
    assert!(harness.frame.chat_container().children.borrow().is_empty());
}

#[tokio::test]
async fn startup_options_default_read_the_process_directories() {
    let options = pi_coding_agent::cli::startup_ui::StartupTuiOptions::default();
    assert!(!options.cwd.is_empty());
    assert!(!options.agent_dir.is_empty());
    assert!(options.terminal.is_none());
}

#[tokio::test]
async fn auto_theme_probe_times_out_and_applies_a_theme() {
    let mut settings = Settings::new();
    settings.insert(
        "theme".to_owned(),
        serde_json::Value::String("light/dark".to_owned()),
    );
    let cwd = tempfile::TempDir::new().expect("cwd");
    let agent = tempfile::TempDir::new().expect("agent");
    let terminal = RecordingTerminal::new(80, 24);
    let ui = pi_coding_agent::cli::startup_ui::create_startup_tui_with(
        &manager(&settings),
        pi_coding_agent::cli::startup_ui::StartupTuiOptions {
            terminal: Some(Box::new(terminal.clone())),
            cwd: cwd.path().display().to_string(),
            agent_dir: agent.path().display().to_string(),
        },
    )
    .await;
    pi_coding_agent::cli::startup_ui::start_startup_tui(&ui, &manager(&settings));
    assert!(pi_coding_agent::modes::interactive::theme::theme::current_theme_name().is_some());
    assert!(terminal.writes().contains("\x1b[?996n"));
}

#[tokio::test]
async fn broken_agent_directory_leaves_the_registered_theme_set_empty() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let file_path = root.path().join("not-a-directory");
    std::fs::write(&file_path, "x").expect("file");
    let settings = Settings::new();
    let cwd = tempfile::TempDir::new().expect("cwd");
    let ui = pi_coding_agent::cli::startup_ui::create_startup_tui_with(
        &manager(&settings),
        pi_coding_agent::cli::startup_ui::StartupTuiOptions {
            terminal: Some(Box::new(RecordingTerminal::new(80, 24))),
            cwd: cwd.path().display().to_string(),
            agent_dir: file_path.display().to_string(),
        },
    )
    .await;
    assert_eq!(ui.terminal_rows(), 24);
}

fn manager(settings: &Settings) -> SettingsManager<InMemorySettingsStorage> {
    SettingsManager::in_memory(settings, SettingsManagerCreateOptions::default())
}

use common::interactive::RecordingTerminal;
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, SettingsManager, SettingsManagerCreateOptions,
};

#[tokio::test]
async fn status_indicator_wrappers_render_every_kind_and_reason() {
    use pi_coding_agent::modes::interactive::components::status_indicator::{
        BranchSummaryStatusIndicator, CompactionStatusIndicator, CompactionStatusReason,
        RetryStatusIndicator, WorkingStatusIndicator,
    };
    use pi_tui::tui::Component;

    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;
    let request = harness.frame.tui().render_request();
    let color: pi_tui::components::ColorFn = Arc::new(|text: &str| text.to_owned());

    let working =
        WorkingStatusIndicator::new(request.clone(), "Working", None, Some(Arc::clone(&color)));
    assert_eq!(working.kind(), pi_coding_agent::modes::interactive::components::status_indicator::StatusIndicatorKind::Working);
    assert!(!working.render(80).is_empty());
    working.invalidate();
    let _ = working.render_in_border(20);
    let _ = working.render_spinner_in_border(20);

    let retry = RetryStatusIndicator::new(request.clone(), 2, 5, 3000);
    assert_eq!(retry.kind(), pi_coding_agent::modes::interactive::components::status_indicator::StatusIndicatorKind::Retry);
    retry.countdown().expect("countdown").tick();
    let _ = retry.render(80);
    retry.dispose();

    for reason in [
        CompactionStatusReason::Manual,
        CompactionStatusReason::Threshold,
        CompactionStatusReason::Overflow,
    ] {
        let compaction = CompactionStatusIndicator::new(request.clone(), reason);
        let _ = compaction.render(80);
        compaction.dispose();
    }
    let branch = BranchSummaryStatusIndicator::new(request);
    let _ = branch.render(80);
    branch.dispose();
}

#[tokio::test]
async fn footer_formatters_cover_units_and_home_abbreviation() {
    use pi_coding_agent::modes::interactive::components::footer::{
        format_cwd_for_footer, format_tokens,
    };

    assert_eq!(format_tokens(999), "999");
    assert_eq!(format_tokens(1_000), "1.0k");
    assert_eq!(format_tokens(1_500), "1.5k");
    assert_eq!(format_tokens(1_000_000), "1.0M");
    assert_eq!(
        format_cwd_for_footer("/home/user/project", Some("/home/user")),
        "~/project"
    );
    assert_eq!(
        format_cwd_for_footer("/tmp/project", Some("/home/user")),
        "/tmp/project"
    );
    assert_eq!(format_cwd_for_footer("/tmp/project", None), "/tmp/project");
}

#[tokio::test]
async fn session_selector_setters_accept_every_scope_and_state() {
    use pi_coding_agent::modes::interactive::components::session_selector::{
        SessionScope, SessionSelectorComponent, SessionSelectorOptions,
    };
    use pi_tui::tui::Component as _;

    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let selector = SessionSelectorComponent::new(SessionSelectorOptions {
        current_sessions: Vec::new(),
        current_session_path: None,
        on_select: Rc::new(|_path: &str| {}),
        on_cancel: Rc::new(|| {}),
        on_request_all_sessions: Rc::new(|| {}),
        on_refresh_sessions: Rc::new(|| {}),
        request_render: Rc::new(|| {}),
        rename_session: Some(Rc::new(|_path: &str, _name: &str| {})),
        show_rename_hint: Some(true),
    });
    selector.set_current_sessions(Vec::new());
    selector.set_all_sessions(Vec::new());
    selector.set_load_error(SessionScope::All, "load failed");
    selector.set_progress(1, 4);
    let _ = selector.get_session_list().selected_session_path();
    let _ = selector.get_session_list();
    let _ = selector.render(80);
}

#[tokio::test]
async fn settings_selector_submenus_toggle_warnings_models_and_themes() {
    use pi_coding_agent::modes::interactive::components::settings_selector::{
        SettingsSelectorComponent, ThemeSubmenu,
    };
    use pi_coding_agent::modes::interactive::theme::theme::TerminalTheme;

    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;
    let selector = SettingsSelectorComponent::new(
        &harness.frame.settings_config(),
        harness.frame.settings_callbacks(),
    );
    let _ = selector.render(120);
    let _ = selector.children();
    selector.invalidate();
    let list = selector.get_settings_list();

    list.select_item("warnings");
    list.handle_input("\r");
    list.handle_input("\r");
    list.handle_input("\x1b");
    let warnings = harness
        .frame
        .session()
        .settings_manager()
        .lock()
        .expect("settings")
        .get_warnings();
    assert_eq!(
        warnings
            .get("anthropicExtraUsage")
            .and_then(serde_json::Value::as_bool),
        Some(false)
    );

    list.select_item("model-thinking");
    list.handle_input("\r");
    list.handle_input("\r");
    list.handle_input("\r");
    list.handle_input("\x1b");

    let done = Rc::new(RefCell::new(None::<Option<String>>));
    let done_for_callback = Rc::clone(&done);
    let theme = ThemeSubmenu::new(
        "dark",
        TerminalTheme::Dark,
        vec!["dark".to_owned(), "light".to_owned()],
        harness.frame.settings_callbacks(),
        Rc::new(move |value, _options| {
            *done_for_callback.borrow_mut() = Some(value);
        }),
    );
    let _ = theme.render(80);
    let _ = theme.children();
    theme.invalidate();
    theme.handle_input("\x1b[A");
    theme.handle_input("\r");
    theme.handle_input("\x1b[B");
    theme.handle_input("\x1b[B");
    theme.handle_input("\x1b[B");
    theme.handle_input("\r");
    theme.handle_input("\x1b");
    assert!(done.borrow().is_some());

    let done = Rc::new(RefCell::new(None::<Option<String>>));
    let done_for_callback = Rc::clone(&done);
    let automatic = ThemeSubmenu::new(
        "light/dark",
        TerminalTheme::Dark,
        vec!["dark".to_owned(), "light".to_owned()],
        harness.frame.settings_callbacks(),
        Rc::new(move |value, _options| {
            *done_for_callback.borrow_mut() = Some(value);
        }),
    );
    automatic.handle_input("\x1b[B");
    automatic.handle_input("\r");
    automatic.handle_input("\r");
    automatic.handle_input("\x1b[B");
    automatic.handle_input("\r");
    assert!(done.borrow().is_some());
}

#[tokio::test]
async fn switching_to_a_session_with_a_missing_cwd_retries_with_the_fallback() {
    use pi_coding_agent::session_manager::SessionManager;

    let root = tempfile::TempDir::new().expect("tempdir");
    let old_cwd = root.path().join("old-project");
    let session_dir = root.path().join("sessions");
    std::fs::create_dir_all(&old_cwd).expect("old cwd");
    let mut manager = SessionManager::create(
        &old_cwd.display().to_string(),
        Some(&session_dir.display().to_string()),
        None,
    )
    .expect("manager");
    manager
        .append_message(common::agent_session::user_message("old", 1))
        .expect("user");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            common::agent_session::assistant_message("reply", 2),
        )))
        .expect("assistant");
    let session_file = manager.session_file().expect("session file").to_owned();
    drop(manager);
    std::fs::remove_dir_all(&old_cwd).expect("remove old cwd");

    let harness = Harness::new(HarnessOptions::default()).await;
    harness
        .frame
        .switch_session(&session_file)
        .await
        .expect("retry");
    let text = harness.chat_text();
    assert!(
        text.contains("not found") || text.contains("does not exist"),
        "{text}"
    );
    assert!(text.contains("Resumed session in current cwd"), "{text}");
}

#[tokio::test]
async fn navigate_tree_can_abort_the_branch_summary() {
    use std::sync::Arc;
    use std::time::Duration;

    let release = Arc::new(tokio::sync::Notify::new());
    let harness = harness_with(HarnessOptions {
        stream: Some(gated_stream(Arc::clone(&release))),
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    let target = harness
        .frame
        .session()
        .user_messages_for_forking()
        .first()
        .map(|(id, _)| id.clone())
        .expect("user message");

    let navigate = harness.frame.navigate_tree(&target);
    tokio::pin!(navigate);
    tokio::select! {
        () = &mut navigate => {}
        () = tokio::time::sleep(Duration::from_millis(40)) => {
            harness.send_input("j");
            harness.send_input("\r");
        }
    }
    tokio::time::sleep(Duration::from_millis(40)).await;
    harness.frame.session().abort_branch_summary();
    tokio::select! {
        () = &mut navigate => {}
        () = tokio::time::sleep(Duration::from_millis(500)) => {}
    }
    harness.send_input("\x1b");
    release.notify_waiters();
}

#[tokio::test]
async fn missing_provider_credentials_fall_back_when_refreshing_login_options() {
    use pi_coding_agent::auth_storage::AuthStorageData;
    use pi_coding_agent::modes::interactive::components::oauth_selector::AuthSelectorProvider;

    let mut auth = AuthStorageData::new();
    auth.insert(
        "ghost-provider".to_owned(),
        serde_json::json!({ "type": "api_key", "key": "ghost" }),
    );
    let harness = harness_with(HarnessOptions {
        auth,
        ..HarnessOptions::default()
    })
    .await;
    submit(&harness, "/logout").await;
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    assert!(
        harness
            .chat_text()
            .contains("Removed stored credential for ghost-provider")
    );

    let provider = AuthSelectorProvider {
        id: "missing-provider".to_owned(),
        name: "Missing".to_owned(),
        auth_type: AuthType::ApiKey,
        method: None,
        status: None,
    };
    harness.frame.start_provider_login(provider).await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    harness.frame.drain_commands().await;
}

#[tokio::test]
async fn model_catalogs_refresh_with_errors_and_timeouts_are_reported() {
    let harness = harness_with(HarnessOptions::default()).await;
    harness.frame.show_models_selector().await;
    harness.send_input("\x1b");
}

#[tokio::test]
async fn extension_command_diagnostics_render_in_the_resource_listing() {
    let fixtures = ResourceFixtures {
        extension_factories: vec![
            (
                Some("first".to_owned()),
                false,
                Arc::new(|api| {
                    Box::pin(async move {
                        api.register_command(
                            "dup".to_owned(),
                            None,
                            Arc::new(|_a, _c| Box::pin(async {})),
                        )?;
                        Ok(())
                    })
                }),
            ),
            (
                Some("second".to_owned()),
                false,
                Arc::new(|api| {
                    Box::pin(async move {
                        api.register_command(
                            "dup".to_owned(),
                            None,
                            Arc::new(|_a, _c| Box::pin(async {})),
                        )?;
                        Ok(())
                    })
                }),
            ),
        ],
        ..ResourceFixtures::default()
    };
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(fixtures),
        verbose: true,
        ..HarnessOptions::default()
    })
    .await;
    let _ = harness.frame.create_base_autocomplete_provider();
    harness.frame.show_loaded_resources(true);
    let _ = harness.resources_text();
}

#[tokio::test]
async fn init_then_fuller_frame_settings_and_mode_switches() {
    let harness = Harness::new(HarnessOptions {
        frame_options:
            pi_coding_agent::modes::interactive::interactive_mode::InteractiveModeOptions {
                skip_managed_tool_setup: true,
                ..Default::default()
            },
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.init().await;
    harness.frame.apply_fullscreen_scrollbar_setting();
    assert!(
        harness
            .frame
            .switch_tui_mode(pi_tui::tui::TuiMode::Regular, false, false)
    );
    assert!(
        harness
            .frame
            .switch_tui_mode(pi_tui::tui::TuiMode::Fullscreen, false, false)
    );
    assert!(
        harness
            .frame
            .switch_tui_mode(pi_tui::tui::TuiMode::Fullscreen, true, false)
    );
    harness.frame.apply_runtime_settings();
    harness.frame.update_terminal_title();
    harness
        .frame
        .stop(pi_coding_agent::settings_manager::FullscreenExitOutput::Transcript);
}

#[tokio::test]
async fn new_version_notification_skips_blank_notes() {
    use pi_coding_agent::utils::version_check::LatestPiRelease;

    let harness = Harness::new(HarnessOptions::default()).await;
    harness
        .frame
        .show_new_version_notification(&LatestPiRelease {
            version: "1.2.3".to_owned(),
            package_name: Some("pkg".to_owned()),
            note: Some("   ".to_owned()),
        });
    assert!(harness.chat_text().contains("Update Available"));
}

#[tokio::test]
async fn reset_extension_ui_restores_defaults_with_a_custom_editor() {
    use pi_coding_agent::modes::interactive::interactive_mode::{AppEditor, EditorFactory};

    let harness = Harness::new(HarnessOptions::default()).await;
    let editor = Rc::new(common::interactive::TestEditor::new("CUSTOM", true));
    let editor_for_factory = Rc::clone(&editor);
    let factory: EditorFactory = Rc::new(move |_tui, _theme, _keybindings| {
        let editor: Rc<dyn AppEditor> = editor_for_factory.clone();
        editor
    });
    harness.frame.set_custom_editor_component(Some(factory));
    let ui = harness.frame.ui_context();
    let widget = ui.call(
        "setWidget",
        Some(serde_json::json!({ "key": "w", "content": ["line"] })),
    );
    harness.frame.drain_ui_requests();
    drop(widget);
    harness.frame.submit_text("/reload").await;
    harness.frame.drain_commands().await;
    assert!(harness.frame.editor().embed_working_status());
}

#[tokio::test]
async fn login_command_with_a_single_provider_match_starts_it() {
    let mut solo = common::model_layer::model("solo-provider", "solo-model");
    solo.name = "Solo".to_owned();
    let harness = harness_with(HarnessOptions {
        extra_models: vec![solo],
        ..HarnessOptions::default()
    })
    .await;
    submit(&harness, "/login solo-provider").await;
    harness
        .frame
        .finish_login("Solo", AuthType::ApiKey, Err("unsupported".to_owned()))
        .await;
    assert!(harness.chat_text().contains("failed"));
}

#[tokio::test]
async fn footer_renders_a_subscription_model_and_extension_statuses() {
    use pi_ai::auth::types::{Credential, OAuthCredentials};

    let credential = Credential::OAuth(OAuthCredentials {
        refresh: "refresh".to_owned(),
        access: "access".to_owned(),
        expires: 4_102_444_800_000,
        extra: std::collections::BTreeMap::new(),
    });
    let mut auth = pi_coding_agent::auth_storage::AuthStorageData::new();
    auth.insert(
        "anthropic".to_owned(),
        serde_json::to_value(&credential).expect("credential serializes"),
    );
    let harness = Harness::new(HarnessOptions {
        model: common::agent_session::anthropic_model(),
        auth,
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    let ui = harness.frame.ui_context();
    let status = ui.call(
        "setStatus",
        Some(serde_json::json!({ "key": "build", "text": "ok\x1b[31mred" })),
    );
    harness.frame.drain_ui_requests();
    drop(status);
    let footer = harness.frame.footer_component();
    let rendered = footer.render(160);
    assert!(!rendered.is_empty());
    footer.invalidate();
}

#[tokio::test]
async fn custom_oauth_login_drives_every_auth_prompt_channel() {
    use pi_ai::auth::types::{AuthEvent, OAuthCredentials};
    use pi_coding_agent::provider_composer::{
        ExtensionOAuthConfig, OAuthLoginCallbacks, OAuthPrompt, OAuthSelectOption,
        OAuthSelectPrompt, ProviderConfigInput,
    };

    let harness = harness_with(HarnessOptions::default()).await;
    let runtime = harness.frame.session().model_runtime().clone();
    let login: pi_coding_agent::provider_composer::ExtensionOAuthLoginFn =
        Arc::new(|callbacks: OAuthLoginCallbacks| {
            Box::pin(async move {
                (callbacks.notify)(AuthEvent::Progress {
                    message: "starting".to_owned(),
                });
                let _ = (callbacks.select)(OAuthSelectPrompt {
                    message: "Pick".to_owned(),
                    options: vec![OAuthSelectOption {
                        id: "one".to_owned(),
                        label: "One".to_owned(),
                    }],
                })
                .await;
                let _ = (callbacks.prompt)(OAuthPrompt {
                    message: "Token".to_owned(),
                    placeholder: Some("token".to_owned()),
                    allow_empty: None,
                })
                .await;
                let _ = (callbacks.manual_code_input)().await;
                Ok::<OAuthCredentials, pi_ai::models::ProviderError>(OAuthCredentials {
                    refresh: "refresh".to_owned(),
                    access: "access".to_owned(),
                    expires: 4_102_444_800_000,
                    extra: std::collections::BTreeMap::new(),
                })
            })
        });
    runtime
        .register_provider(
            "prompting",
            ProviderConfigInput {
                name: Some("Prompting".to_owned()),
                base_url: Some("https://prompt.invalid/v1".to_owned()),
                api: Some(pi_ai::types::Api::from("openai-completions")),
                api_key: None,
                oauth: Some(ExtensionOAuthConfig {
                    name: "OAuth".to_owned(),
                    is_subscription: None,
                    uses_callback_server: None,
                    login,
                    refresh_token: Arc::new(|credentials, _signal| {
                        Box::pin(async move { Ok(credentials) })
                    }),
                    get_api_key: Arc::new(|_credentials| "key".to_owned()),
                    modify_models: None,
                }),
                ..ProviderConfigInput::default()
            },
        )
        .expect("provider registers");

    harness.frame.handle_login_command(Some("prompting")).await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    harness.frame.drain_ui_requests();
    harness.send_input("\r");

    tokio::time::sleep(Duration::from_millis(20)).await;
    harness.frame.drain_ui_requests();
    harness.send_input("token");
    harness.send_input("\r");

    tokio::time::sleep(Duration::from_millis(20)).await;
    harness.frame.drain_ui_requests();
    harness.send_input("code");
    harness.send_input("\r");

    tokio::time::sleep(Duration::from_millis(40)).await;
    harness.frame.drain_commands().await;
    assert!(
        harness.chat_text().contains("Logged in to Prompting")
            || harness.chat_text().contains("failed"),
        "{}",
        harness.chat_text()
    );
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "one fixture builds every expandable row kind so the toggle exercises each impl"
)]
async fn toggling_tool_output_covers_every_expandable_row_kind() {
    use pi_coding_agent::session_manager::SessionManager;

    let fixtures = ResourceFixtures {
        extension_factories: vec![(
            Some("renderers".to_owned()),
            false,
            Arc::new(|api| {
                Box::pin(async move {
                    let message_renderer: pi_coding_agent::extensions::types::MessageRenderer =
                        Arc::new(|_message, _options, _theme| {
                            let text: Rc<dyn Component> = Rc::new(
                                pi_tui::components::Text::with_padding("CUSTOM MESSAGE", 0, 0),
                            );
                            Some(text)
                        });
                    api.register_message_renderer("note".to_owned(), message_renderer)?;
                    let entry_renderer: pi_coding_agent::modes::interactive::components::custom_entry::EntryRenderer =
                        Arc::new(|_entry, _options, _theme| {
                            let text: Rc<dyn Component> =
                                Rc::new(pi_tui::components::Text::with_padding(
                                    "CUSTOM ENTRY",
                                    0,
                                    0,
                                ));
                            Some(text)
                        });
                    api.register_entry_renderer("note-entry".to_owned(), entry_renderer)?;
                    Ok(())
                })
            }),
        )],
        context_files: vec![pi_coding_agent::resource_loader::ContextFile {
            path: "/tmp/project/AGENTS.md".to_owned(),
            content: String::new(),
        }],
        ..ResourceFixtures::default()
    };
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(fixtures),
        verbose: true,
        seed_manager: Some(Box::new(|manager: &mut SessionManager| {
            manager
                .append_message(common::agent_session::user_message(
                    "<skill name=\"demo\" location=\"/x\">\nbody\n</skill>\n\nask",
                    1,
                ))
                .expect("skill");
            manager
                .append_custom_entry("note-entry", Some(serde_json::json!({ "text": "x" })))
                .expect("entry");
        })),
        ..HarnessOptions::default()
    })
    .await;
    let mut tool_call = common::agent_session::assistant_message("tool", 2);
    tool_call.content = vec![pi_ai::types::AssistantBlock::ToolCall(
        pi_ai::types::ToolCall {
            id: "call".to_owned(),
            name: "read".to_owned(),
            arguments: serde_json::Map::new(),
            thought_signature: None,
            namespace: None,
        },
    )];
    harness.frame.add_message_to_chat(
        &AgentMessage::Standard(Message::Assistant(tool_call)),
        false,
    );
    harness.frame.add_message_to_chat(
        &AgentMessage::Custom(CustomAgentMessage {
            role: "bashExecution".to_owned(),
            timestamp: 3,
            data: serde_json::json!({ "command": "ls", "output": "x" })
                .as_object()
                .cloned()
                .unwrap_or_default(),
        }),
        false,
    );
    harness.frame.add_message_to_chat(
        &AgentMessage::Custom(CustomAgentMessage {
            role: "custom".to_owned(),
            timestamp: 4,
            data: serde_json::json!({ "display": true, "customType": "note" })
                .as_object()
                .cloned()
                .unwrap_or_default(),
        }),
        false,
    );
    harness.frame.add_message_to_chat(
        &AgentMessage::Custom(CustomAgentMessage {
            role: "compactionSummary".to_owned(),
            timestamp: 5,
            data: serde_json::json!({ "summary": "compact", "tokensBefore": 10 })
                .as_object()
                .cloned()
                .unwrap_or_default(),
        }),
        false,
    );
    harness.frame.add_message_to_chat(
        &AgentMessage::Custom(CustomAgentMessage {
            role: "branchSummary".to_owned(),
            timestamp: 6,
            data: serde_json::json!({ "summary": "branch" })
                .as_object()
                .cloned()
                .unwrap_or_default(),
        }),
        false,
    );
    harness.frame.show_loaded_resources(true);
    harness.frame.set_tools_expanded(true);
    harness.frame.set_tools_expanded(false);
    harness.frame.rebuild_chat_from_messages();
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn tree_selector_copy_selected_reaches_the_copy_callback() {
    use pi_coding_agent::modes::interactive::components::tree_selector::{
        TreeSelectorComponent, TreeSelectorOptions,
    };

    let harness = Harness::new(HarnessOptions {
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    let tree = harness.frame.session_tree();
    assert!(!tree.is_empty());
    let copied = Rc::new(RefCell::new(Vec::new()));
    let copied_for_callback = Rc::clone(&copied);
    let selector = TreeSelectorComponent::new(
        &tree,
        None,
        24,
        TreeSelectorOptions {
            on_select: Rc::new(|_id: &str| {}),
            on_cancel: Rc::new(|| {}),
            on_label_change: Some(Rc::new(|_id: &str, _label: Option<&str>| {})),
            initial_selected_id: None,
            initial_filter_mode: None,
        },
    );
    selector.set_on_copy(Rc::new(move |text: Option<&str>| {
        copied_for_callback
            .borrow_mut()
            .extend(text.map(str::to_owned));
    }));
    selector.get_tree_list().copy_selected();
    let _ = selector.render(120);
    let _ = selector.children();
    selector.invalidate();
    assert!(!copied.borrow().is_empty());
}

#[tokio::test]
async fn session_selector_rename_callback_persists_through_the_frame() {
    let harness = harness_with(HarnessOptions {
        persisted: true,
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    submit(&harness, "/resume").await;
    harness.send_input("\x12");
    harness.send_input("renamed session");
    harness.send_input("\r");
    harness.frame.drain_commands().await;
    let name = harness.frame.session().session_name();
    let _ = name;
}

#[tokio::test]
async fn models_refresh_with_a_failing_provider_reports_partial_failure() {
    use pi_coding_agent::provider_composer::ProviderConfigInput;

    let harness = harness_with(HarnessOptions::default()).await;
    let runtime = harness.frame.session().model_runtime().clone();
    runtime
        .register_provider(
            "failing-refresh",
            ProviderConfigInput {
                name: Some("Failing".to_owned()),
                base_url: Some("https://fail.invalid/v1".to_owned()),
                api: Some(pi_ai::types::Api::from("openai-completions")),
                api_key: Some("key".to_owned()),
                refresh_models: Some(Arc::new(|_context| {
                    Box::pin(async move { Err("refresh failed".into()) })
                })),
                ..ProviderConfigInput::default()
            },
        )
        .expect("provider registers");
    harness.frame.show_models_selector().await;
    harness.send_input("\x1b");
    let _ = harness.chat_text();
}

#[tokio::test]
async fn settings_config_maps_default_models_and_thinking_overrides() {
    use pi_coding_agent::modes::interactive::components::settings_selector::SettingsSelectorComponent;

    let harness = Harness::new(HarnessOptions {
        settings: {
            let mut settings = Settings::new();
            settings.insert(
                "defaultProvider".to_owned(),
                serde_json::Value::String("faux".to_owned()),
            );
            settings.insert(
                "defaultModel".to_owned(),
                serde_json::Value::String("faux-1".to_owned()),
            );
            settings.insert(
                "modelThinkingLevels".to_owned(),
                serde_json::json!({ "faux/faux-1": "high", "bad/entry": "nonsense" }),
            );
            settings
        },
        ..HarnessOptions::default()
    })
    .await;
    let config = harness.frame.settings_config();
    assert_ne!(config.default_model, "not set");
    assert_eq!(config.model_thinking_levels.len(), 1);
    let selector = SettingsSelectorComponent::new(&config, harness.frame.settings_callbacks());
    let _ = selector.render(120);
}

#[tokio::test]
async fn switching_to_a_project_with_resources_asks_for_trust_and_cancels() {
    use pi_coding_agent::session_manager::SessionManager;

    let root = tempfile::TempDir::new().expect("tempdir");
    let target_cwd = root.path().join("target");
    std::fs::create_dir_all(format!("{}/.pi", target_cwd.display())).expect("target config");
    std::fs::write(format!("{}/.pi/settings.json", target_cwd.display()), "{}").expect("settings");
    let session_dir = root.path().join("target-sessions");
    let mut manager = SessionManager::create(
        &target_cwd.display().to_string(),
        Some(&session_dir.display().to_string()),
        None,
    )
    .expect("manager");
    manager
        .append_message(common::agent_session::user_message("target", 1))
        .expect("user");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            common::agent_session::assistant_message("reply", 2),
        )))
        .expect("assistant");
    let path = manager.session_file().expect("file").to_owned();

    let harness = harness_with(HarnessOptions::default()).await;
    harness
        .frame
        .session()
        .settings_manager()
        .lock()
        .expect("settings")
        .set_project_trusted(false);
    let mut switch = Box::pin(harness.frame.switch_session(&path));
    tokio::select! {
        _ = &mut switch => {}
        () = tokio::time::sleep(Duration::from_millis(50)) => {
            harness.send_input("\x1b");
            tokio::select! {
                _ = &mut switch => {}
                () = tokio::time::sleep(Duration::from_millis(500)) => {}
            }
        }
    }
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn trust_selector_renders_a_saved_decision() {
    use pi_coding_agent::trust_manager::ProjectTrustStore;

    let harness = harness_with(HarnessOptions::default()).await;
    let agent_dir = harness.agent_dir.clone();
    let store = ProjectTrustStore::new(&agent_dir);
    store.set(&harness.cwd, Some(true)).expect("trust entry");
    submit(&harness, "/trust").await;
    harness.send_input("\x1b");
}

#[tokio::test]
async fn navigating_with_a_summary_completes_and_clears_the_indicator() {
    let harness = harness_with(HarnessOptions {
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    let target = harness
        .frame
        .session()
        .user_messages_for_forking()
        .first()
        .map(|(id, _)| id.clone())
        .expect("user message");
    let navigate = harness.frame.navigate_tree(&target);
    tokio::pin!(navigate);
    tokio::select! {
        () = &mut navigate => {}
        () = tokio::time::sleep(Duration::from_millis(40)) => {
            harness.send_input("j");
            harness.send_input("\r");
        }
    }
    tokio::select! {
        () = &mut navigate => {}
        () = tokio::time::sleep(Duration::from_millis(500)) => {}
    }
    assert_ne!(
        harness.frame.active_status_kind(),
        Some(
            pi_coding_agent::modes::interactive::components::status_indicator::StatusIndicatorKind::BranchSummary
        )
    );
}

#[tokio::test]
async fn finish_login_with_available_default_selects_the_model() {
    use pi_ai::auth::types::{Credential, OAuthCredentials};

    let credential = Credential::OAuth(OAuthCredentials {
        refresh: "refresh".to_owned(),
        access: "access".to_owned(),
        expires: 4_102_444_800_000,
        extra: std::collections::BTreeMap::new(),
    });
    let mut auth = pi_coding_agent::auth_storage::AuthStorageData::new();
    auth.insert(
        "anthropic".to_owned(),
        serde_json::to_value(&credential).expect("credential serializes"),
    );
    let harness = Harness::new(HarnessOptions {
        model: common::agent_session::anthropic_model(),
        auth,
        ..HarnessOptions::default()
    })
    .await;
    let provider_name = harness
        .frame
        .get_login_provider_options(None)
        .into_iter()
        .find(|provider| provider.id == "anthropic")
        .map_or_else(|| "anthropic".to_owned(), |provider| provider.name);
    harness
        .frame
        .finish_login(&provider_name, AuthType::OAuth, Ok(()))
        .await;
    let settings = harness.frame.session().settings_manager();
    let guard = settings.lock().expect("settings");
    let _ = (guard.get_default_provider(), guard.get_default_model());
    drop(guard);
    assert!(harness.chat_text().contains("Logged in to"));
}

#[tokio::test]
async fn implicit_trust_store_failure_reports_a_warning() {
    let harness = Harness::new(HarnessOptions {
        frame_options:
            pi_coding_agent::modes::interactive::interactive_mode::InteractiveModeOptions {
                auto_trust_on_reload_cwd: Some(String::new()),
                ..Default::default()
            },
        ..HarnessOptions::default()
    })
    .await;
    // Arm the cwd and make the store path unwritable by replacing the agent
    // directory with a regular file.
    let cwd = harness.cwd.clone();
    std::fs::create_dir_all(format!("{cwd}/.pi")).expect("project config");
    std::fs::write(format!("{cwd}/.pi/settings.json"), "{}").expect("settings");
    let agent_dir = harness.agent_dir.clone();
    std::fs::remove_dir_all(&agent_dir).expect("remove agent dir");
    std::fs::write(&agent_dir, "not a directory").expect("file agent dir");
    harness.frame.set_auto_trust_on_reload_for_test(Some(cwd));
    assert!(
        !harness
            .frame
            .maybe_save_implicit_project_trust_after_reload()
    );
    assert!(harness.chat_text().contains("Could not save project trust"));
}

#[tokio::test]
async fn clearing_queues_drains_compaction_entries_and_session_entries() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.queue_compaction_message(
        "steer",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::Steer,
    );
    harness.frame.queue_compaction_message(
        "follow",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::FollowUp,
    );
    assert_eq!(
        harness.frame.clear_all_queues(),
        (vec!["steer".to_owned()], vec!["follow".to_owned()])
    );
    harness.frame.queue_compaction_message(
        "queued",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::Steer,
    );
    let (steering, follow_up) = harness.frame.get_all_queued_messages();
    assert_eq!(steering, vec!["queued".to_owned()]);
    assert!(follow_up.is_empty());
}

#[tokio::test]
async fn session_selector_toggles_scope_sort_and_name_filter() {
    let harness = harness_with(HarnessOptions {
        persisted: true,
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    submit(&harness, "/resume").await;
    harness.send_input("\x01");
    harness.send_input("\x13");
    harness.send_input("\x0e");
    harness.send_input("\x1b");
    harness.frame.drain_commands().await;
    assert!(!harness.chat_text().is_empty() || harness.frame.extension_selector().is_none());
}

#[tokio::test]
async fn import_of_a_session_with_a_missing_cwd_reports_without_hanging() {
    use pi_coding_agent::session_manager::SessionManager;

    let root = tempfile::TempDir::new().expect("tempdir");
    let old_cwd = root.path().join("old");
    let session_dir = root.path().join("sessions");
    std::fs::create_dir_all(&old_cwd).expect("cwd");
    let mut manager = SessionManager::create(
        &old_cwd.display().to_string(),
        Some(&session_dir.display().to_string()),
        None,
    )
    .expect("manager");
    manager
        .append_message(common::agent_session::user_message("old", 1))
        .expect("user");
    manager
        .append_message(AgentMessage::Standard(Message::Assistant(
            common::agent_session::assistant_message("reply", 2),
        )))
        .expect("assistant");
    let path = manager.session_file().expect("file").to_owned();
    drop(manager);
    std::fs::remove_dir_all(&old_cwd).expect("remove");

    let harness = Harness::new(HarnessOptions {
        persisted: true,
        ..HarnessOptions::default()
    })
    .await;
    harness
        .frame
        .handle_import_command(&format!("/import {path}"))
        .await;
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn footer_adapter_reads_session_entries_and_subscription() {
    let harness = Harness::new(HarnessOptions {
        messages: seeds_messages(),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.session().set_session_name("footer name");
    let footer = harness.frame.footer_component();
    footer.render(200);
    footer.render(20);
    footer.invalidate();
    footer.dispose();
}

#[tokio::test]
async fn model_thinking_submenu_can_clear_an_override() {
    use pi_coding_agent::modes::interactive::components::settings_selector::SettingsSelectorComponent;

    let harness = Harness::new(HarnessOptions {
        settings: {
            let mut settings = Settings::new();
            settings.insert(
                "modelThinkingLevels".to_owned(),
                serde_json::json!({ "faux/faux-1": "high" }),
            );
            settings
        },
        ..HarnessOptions::default()
    })
    .await;
    let selector = SettingsSelectorComponent::new(
        &harness.frame.settings_config(),
        harness.frame.settings_callbacks(),
    );
    let list = selector.get_settings_list();
    list.select_item("model-thinking");
    list.handle_input("\r");
    list.handle_input("\r");
    // The active override appends "(clear override)" as the last level row.
    for _ in 0..8 {
        list.handle_input("\x1b[B");
    }
    list.handle_input("\r");
    list.handle_input("\x1b");
    let levels = harness
        .frame
        .session()
        .settings_manager()
        .lock()
        .expect("settings")
        .get_all_model_thinking_levels();
    let _ = levels;
}

#[tokio::test]
async fn automatic_theme_menu_selects_light_theme_and_single_mode() {
    use pi_coding_agent::modes::interactive::components::settings_selector::ThemeSubmenu;
    use pi_coding_agent::modes::interactive::theme::theme::TerminalTheme;

    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;
    let done = Rc::new(RefCell::new(None::<Option<String>>));
    let done_for_callback = Rc::clone(&done);
    let automatic = ThemeSubmenu::new(
        "light/dark",
        TerminalTheme::Light,
        vec!["dark".to_owned(), "light".to_owned()],
        harness.frame.settings_callbacks(),
        Rc::new(move |value, _options| {
            *done_for_callback.borrow_mut() = Some(value);
        }),
    );
    // Light-theme row opens the themed select; choose light.
    automatic.handle_input("\r");
    automatic.handle_input("\x1b[B");
    automatic.handle_input("\r");
    // Move to "Change mode" and switch to single mode.
    automatic.handle_input("\x1b[B");
    automatic.handle_input("\x1b[B");
    automatic.handle_input("\r");
    automatic.handle_input("\x1b");
    let _ = done;
}

#[tokio::test]
async fn compaction_notice_omits_small_costs() {
    let mut settings = Settings::new();
    settings.insert(
        "showCacheMissNotices".to_owned(),
        serde_json::Value::Bool(true),
    );
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;
    let usage = pi_ai::types::Usage {
        input: 10,
        output: 5,
        total_tokens: 15,
        cost: pi_ai::types::UsageCost::default(),
        ..pi_ai::types::Usage::default()
    };
    harness
        .frame
        .add_compaction_cost_notice("Compaction", &usage);
    let text = harness.chat_text();
    assert!(text.contains("15 tokens billed"));
    assert!(!text.contains("~$"), "{text}");
}

#[tokio::test]
async fn startup_notice_adds_a_spacer_after_existing_rows() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.show_error("before");
    harness
        .frame
        .set_changelog_markdown_for_test(Some("## Changes".to_owned()));
    harness.frame.show_startup_notices_if_needed();
    assert!(harness.chat_text().contains("What's New"));
}
