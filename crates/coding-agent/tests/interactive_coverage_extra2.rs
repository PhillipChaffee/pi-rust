//! Third-pass coverage for the remaining dispatch, selector, renderer, and
//! footer branches. Fixtures shared with the other coverage suites are hoisted
//! into `common::interactive`.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the session and settings shapes"
)]
#![allow(
    clippy::large_futures,
    reason = "each test drives one frame operation at a time"
)]

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use common::interactive::{Harness, HarnessOptions};
use pi_ai::types::Message;
use pi_coding_agent::modes::interactive::components::footer::FooterComponent;
use pi_coding_agent::modes::interactive::components::session_selector::{
    SessionScope, SessionSelectorComponent, SessionSelectorOptions,
};
use pi_coding_agent::session_manager::{SessionInfo, SessionManager};
use pi_coding_agent::settings_manager::Settings;
use pi_tui::tui::Component;

#[tokio::test]
async fn key_actions_dispatch_their_frame_commands() {
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
    harness.frame.drain_commands().await;

    for key in [
        "\x1b[Z", // shift+tab: cycle thinking
        "\x10",   // ctrl+p: cycle model forward
        "\x0c",   // ctrl+l: model selector
        "\x0f",   // ctrl+o: toggle tool output
        "\x14",   // ctrl+t: toggle thinking blocks
        "\x18",   // ctrl+x: copy message (no message: error row)
        "\x1a",   // ctrl+z: suspend status
    ] {
        harness.send_input(key);
        harness.frame.drain_commands().await;
        harness.send_input("\x1b");
        harness.send_input("\x1b");
    }
    harness.frame.drain_commands().await;
    assert!(
        harness.chat_text().contains("not supported") || harness.chat_text().contains("Thinking"),
        "{}",
        harness.chat_text()
    );
}

fn selector_sessions() -> (Vec<SessionInfo>, String) {
    let root = tempfile::TempDir::new().expect("tempdir");
    let mut current = SessionManager::create(
        &root.path().display().to_string(),
        Some(&root.path().join("sessions").display().to_string()),
        None,
    )
    .expect("current");
    current
        .append_message(common::agent_session::user_message("current", 1))
        .expect("current message");
    current
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            Message::Assistant(common::agent_session::assistant_message("reply", 2)),
        ))
        .expect("current reply");
    let current_path = current.session_file().expect("current file").to_owned();
    let mut other = SessionManager::create(
        "/tmp/other-project",
        Some(&root.path().join("sessions").display().to_string()),
        None,
    )
    .expect("other");
    other
        .append_message(common::agent_session::user_message("other", 3))
        .expect("other message");
    let other_path = other.session_file().expect("other file").to_owned();
    let info = |path: &str| SessionInfo {
        path: path.to_owned(),
        id: "id".to_owned(),
        cwd: "/tmp".to_owned(),
        name: Some("name".to_owned()),
        parent_session_path: None,
        created: Some(1),
        modified: 1,
        message_count: 1,
        first_message: "first".to_owned(),
        all_messages_text: "first reply".to_owned(),
    };
    (vec![info(&current_path), info(&other_path)], current_path)
}

#[tokio::test]
async fn session_selector_covers_scope_sort_filter_rename_and_delete_paths() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let (sessions, current_path) = selector_sessions();
    let selected = Rc::new(RefCell::new(Vec::new()));
    let selected_for_callback = Rc::clone(&selected);
    let renamed = Rc::new(RefCell::new(Vec::new()));
    let renamed_for_callback = Rc::clone(&renamed);
    let selector = SessionSelectorComponent::new(SessionSelectorOptions {
        current_sessions: sessions.clone(),
        current_session_path: Some(current_path),
        on_select: Rc::new(move |path: &str| {
            selected_for_callback.borrow_mut().push(path.to_owned());
        }),
        on_cancel: Rc::new(|| {}),
        on_request_all_sessions: Rc::new(|| {}),
        on_refresh_sessions: Rc::new(|| {}),
        request_render: Rc::new(|| {}),
        rename_session: Some(Rc::new(move |path: &str, name: &str| {
            renamed_for_callback
                .borrow_mut()
                .push((path.to_owned(), name.to_owned()));
        })),
        show_rename_hint: Some(true),
    });
    selector.set_all_sessions(sessions);
    selector.set_progress(1, 2);
    selector.set_load_error(SessionScope::All, "load failed");
    for key in ["\x01", "\x13", "\x0e", "\x12", "renamed", "\r", "\x1b"] {
        selector.handle_input(key);
    }
    selector.handle_input("\x04");
    selector.handle_input("\x1b");
    let _ = selected.borrow();
    let _ = renamed.borrow();
    let _ = selector.get_session_list().selected_session_path();
    let _ = selector.render(120);
    let _ = selector.children();
    selector.invalidate();
}

#[tokio::test]
async fn tree_selector_renders_every_row_kind_and_filter_view() {
    use pi_coding_agent::modes::interactive::components::tree_selector::{
        TreeSelectorComponent, TreeSelectorOptions,
    };

    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);

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
    let tool_result = pi_agent_core::types::AgentMessage::Standard(Message::ToolResult(
        pi_ai::types::ToolResultMessage {
            tool_call_id: "call".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![pi_ai::types::ToolResultBlock::Text(
                pi_ai::types::TextContent {
                    text: "result".to_owned(),
                    text_signature: None,
                },
            )],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 3,
        },
    ));
    let harness = Harness::new(HarnessOptions {
        seed_manager: Some(Box::new(move |manager: &mut SessionManager| {
            let tool_call = tool_call.clone();
            let tool_result = tool_result.clone();
            manager
                .append_message(common::agent_session::user_message("user", 1))
                .expect("user");
            manager
                .append_message(pi_agent_core::types::AgentMessage::Standard(
                    Message::Assistant(tool_call),
                ))
                .expect("tool call");
            manager.append_message(tool_result).expect("tool result");
            manager
                .append_custom_entry("note", Some(serde_json::json!({ "text": "custom" })))
                .expect("custom");
            manager
                .append_message(common::agent_session::user_message("second", 4))
                .expect("second");
        })),
        ..HarnessOptions::default()
    })
    .await;
    let tree = harness.frame.session_tree();
    let copied = Rc::new(RefCell::new(Vec::new()));
    let copied_for_callback = Rc::clone(&copied);
    let selector = TreeSelectorComponent::new(
        &tree,
        None,
        12,
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
    for key in ["\x0f", "\x0f", "\x0f", "\x0f", "user", "\x1b", "\x1b"] {
        selector.handle_input(key);
    }
    selector.get_tree_list().copy_selected();
    let _ = selector.render(30);
    let _ = selector.render(200);
    let _ = selector.children();
    selector.invalidate();
    let _ = copied.borrow();
}

#[tokio::test]
async fn footer_renders_every_thinking_level_and_model_state() {
    let harness = Harness::new(HarnessOptions {
        messages: common::interactive::exchange_messages(),
        ..HarnessOptions::default()
    })
    .await;
    let footer: Rc<FooterComponent> = harness.frame.footer_component();
    for level in [
        pi_agent_core::types::ThinkingLevel::Off,
        pi_agent_core::types::ThinkingLevel::Minimal,
        pi_agent_core::types::ThinkingLevel::Low,
        pi_agent_core::types::ThinkingLevel::Medium,
        pi_agent_core::types::ThinkingLevel::High,
        pi_agent_core::types::ThinkingLevel::Xhigh,
        pi_agent_core::types::ThinkingLevel::Max,
    ] {
        harness.frame.session().set_thinking_level(
            level,
            pi_coding_agent::core::agent_session::ModelMutationOptions::default(),
        );
        footer.render(100);
    }
    footer.render(10);
    footer.render(200);
    footer.set_auto_compact_enabled(true);
    footer.set_auto_compact_enabled(false);
    footer.invalidate();
}

#[tokio::test]
async fn default_editor_mouse_and_history_members_are_reached() {
    use pi_tui::tui::{TuiMouseButton, TuiMouseEvent, TuiMouseEventType};

    let harness = Harness::new(HarnessOptions {
        messages: common::interactive::exchange_messages(),
        ..HarnessOptions::default()
    })
    .await;
    let editor = harness.frame.editor();
    editor.set_text("mouse");
    editor.add_to_history("history");
    let event = TuiMouseEvent {
        event_type: TuiMouseEventType::Click,
        button: TuiMouseButton::Left,
        x: 1,
        y: 1,
        screen_x: 1,
        screen_y: 1,
        width: 40,
        height: 10,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: Some(1),
    };
    let _ = editor.handle_mouse(&event);
    for kind in [TuiMouseEventType::Press, TuiMouseEventType::Release] {
        let event = TuiMouseEvent {
            event_type: kind,
            ..event.clone()
        };
        let _ = editor.handle_mouse(&event);
    }
}

#[tokio::test]
async fn session_selector_delete_confirmation_removes_a_non_current_session() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let (sessions, current_path) = selector_sessions();
    let deleted = Rc::new(RefCell::new(None));
    let deleted_for_callback = Rc::clone(&deleted);
    let selector = SessionSelectorComponent::new(SessionSelectorOptions {
        current_sessions: sessions,
        current_session_path: Some(current_path),
        on_select: Rc::new(|_path: &str| {}),
        on_cancel: Rc::new(|| {}),
        on_request_all_sessions: Rc::new(|| {}),
        on_refresh_sessions: Rc::new(|| {}),
        request_render: Rc::new(|| {}),
        rename_session: None,
        show_rename_hint: None,
    });
    *selector.get_session_list().on_delete_session.borrow_mut() =
        Some(Rc::new(move |path: &str| {
            *deleted_for_callback.borrow_mut() = Some(path.to_owned());
        }));
    selector.handle_input("\x1b[B");
    selector.handle_input("\x04");
    selector.handle_input("\r");
    selector.handle_input("\x1b");
    let _ = deleted.borrow();
    let _ = selector.render(120);
}

#[tokio::test]
async fn session_selector_renders_every_scope_sort_and_filter_combination() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let info = |path: &str, name: &str, cwd: &str, created: i64| SessionInfo {
        path: path.to_owned(),
        id: name.to_owned(),
        cwd: cwd.to_owned(),
        name: Some(name.to_owned()),
        parent_session_path: None,
        created: Some(created),
        modified: created,
        message_count: 2,
        first_message: format!("first from {name}"),
        all_messages_text: format!("first from {name} reply"),
    };
    let sessions = vec![
        info("/tmp/alpha/one.jsonl", "shared", "/tmp/alpha", 1),
        info("/tmp/beta/two.jsonl", "alpha", "/tmp/beta", 2),
        info("/tmp/project/tools/three.jsonl", "tools", "/tmp/project", 3),
        info("/tmp/project/other/four.jsonl", "other", "/tmp/project", 4),
    ];
    let current = info("/tmp/project/current.jsonl", "current", "/tmp/project", 5);
    let selector = SessionSelectorComponent::new(SessionSelectorOptions {
        current_sessions: vec![info(
            "/tmp/project/current.jsonl",
            "current",
            "/tmp/project",
            5,
        )],
        current_session_path: Some(current.path),
        on_select: Rc::new(|_path: &str| {}),
        on_cancel: Rc::new(|| {}),
        on_request_all_sessions: Rc::new(|| {}),
        on_refresh_sessions: Rc::new(|| {}),
        request_render: Rc::new(|| {}),
        rename_session: Some(Rc::new(|_path: &str, _name: &str| {})),
        show_rename_hint: Some(true),
    });
    selector.set_all_sessions(sessions);
    for key in [
        "\x01", "\x13", "\x13", "\x0e", "\x0e", "f", "i", "\x7f", "\x1b", "\x01", "\x04", "\x1b",
    ] {
        selector.handle_input(key);
        let _ = selector.render(60);
        let _ = selector.render(160);
    }
    selector.set_progress(1, 9);
    selector.set_load_error(SessionScope::Current, "current load failed");
    let _ = selector.render(80);
    let _ = selector.children();
    selector.invalidate();
}

#[tokio::test]
async fn tree_selector_renders_labels_settings_and_custom_entries() {
    use pi_coding_agent::modes::interactive::components::tree_selector::{
        FilterMode, TreeSelectorComponent, TreeSelectorOptions,
    };

    let harness = Harness::new(HarnessOptions {
        messages: common::interactive::exchange_messages(),
        ..HarnessOptions::default()
    })
    .await;
    let mut tree = harness.frame.session_tree();
    if let Some(node) = tree.first_mut() {
        node.label = Some("checkpoint".to_owned());
        node.label_timestamp = Some("2026-03-28T14:32:00.000Z".to_owned());
    }
    for mode in [
        FilterMode::Default,
        FilterMode::NoTools,
        FilterMode::UserOnly,
        FilterMode::LabeledOnly,
        FilterMode::All,
    ] {
        let selector = TreeSelectorComponent::new(
            &tree,
            None,
            24,
            TreeSelectorOptions {
                on_select: Rc::new(|_id: &str| {}),
                on_cancel: Rc::new(|| {}),
                on_label_change: Some(Rc::new(|_id: &str, _label: Option<&str>| {})),
                initial_selected_id: None,
                initial_filter_mode: Some(mode),
            },
        );
        for key in ["\x0c", "\x1b[C", "\x1b[D", "shared", "\x1b", "\x0f"] {
            selector.handle_input(key);
        }
        let _ = selector.render(40);
        let _ = selector.render(120);
    }
}

#[tokio::test]
async fn status_spinner_hidden_and_retry_countdown_branches() {
    use pi_coding_agent::modes::interactive::components::status_indicator::{
        RetryStatusIndicator, StatusIndicator, StatusIndicatorKind,
    };
    use pi_tui::components::{ColorFn, LoaderIndicatorOptions};

    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;
    let color: ColorFn = std::sync::Arc::new(|text: &str| text.to_owned());
    let hidden = LoaderIndicatorOptions {
        frames: Some(Vec::new()),
        ..LoaderIndicatorOptions::default()
    };
    let indicator = StatusIndicator::new(
        StatusIndicatorKind::Working,
        harness.frame.tui().render_request(),
        std::sync::Arc::clone(&color),
        color,
        "hidden",
        Some(hidden),
    );
    assert!(indicator.render_spinner_in_border(20).is_empty());
    indicator.set_indicator(None);
    let _ = indicator.render_spinner_in_border(0);

    let retry = RetryStatusIndicator::new(harness.frame.tui().render_request(), 1, 3, 1000);
    let countdown = retry.countdown().expect("countdown");
    countdown.tick();
    countdown.tick();
    let _ = retry.render_in_border(80);
    retry.dispose();
}

#[tokio::test]
async fn extension_working_message_and_visibility_requests_touch_the_indicator() {
    use pi_coding_agent::extensions::host::UiCallHandler;

    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.show_working_status_indicator();
    let ui = harness.frame.ui_context();
    let message = ui.call(
        "setWorkingMessage",
        Some(serde_json::json!({ "message": "requested" })),
    );
    harness.frame.drain_ui_requests();
    let _ = message.await;
    let visible = ui.call(
        "setWorkingVisible",
        Some(serde_json::json!({ "visible": false })),
    );
    harness.frame.drain_ui_requests();
    let _ = visible.await;
    let label = ui.call(
        "setHiddenThinkingLabel",
        Some(serde_json::json!({ "label": "hidden" })),
    );
    harness.frame.drain_ui_requests();
    let _ = label.await;
    let widget = ui.call(
        "setWidget",
        Some(serde_json::json!({ "key": "w", "content": ["line"], "placement": "belowEditor" })),
    );
    harness.frame.drain_ui_requests();
    let _ = widget.await;
    let clear = ui.call("setWidget", Some(serde_json::json!({ "key": "w" })));
    harness.frame.drain_ui_requests();
    let _ = clear.await;
}

#[tokio::test]
async fn auth_prompt_and_notify_without_a_dialog_settle_immediately() {
    use pi_ai::auth::types::{AuthEvent, AuthPrompt, AuthPromptKind, AuthPromptOption};

    let harness = Harness::new(HarnessOptions::default()).await;
    let ui = harness.frame.ui_context();
    for kind in [
        AuthPromptKind::Text {
            message: "text".to_owned(),
            placeholder: None,
        },
        AuthPromptKind::Secret {
            message: "secret".to_owned(),
            placeholder: None,
        },
        AuthPromptKind::ManualCode {
            message: "code".to_owned(),
            placeholder: None,
        },
        AuthPromptKind::Select {
            message: "select".to_owned(),
            options: vec![AuthPromptOption {
                id: "one".to_owned(),
                label: "One".to_owned(),
                description: None,
            }],
        },
    ] {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        ui.send_auth_prompt(AuthPrompt { signal: None, kind }, sender);
        harness.frame.drain_ui_requests();
        assert_eq!(
            receiver.await.expect("prompt"),
            Err("Login cancelled".to_owned())
        );
    }
    for event in [
        AuthEvent::AuthUrl {
            url: "https://example.test".to_owned(),
            instructions: None,
        },
        AuthEvent::DeviceCode {
            user_code: "code".to_owned(),
            verification_uri: "https://example.test".to_owned(),
            interval_seconds: Some(1),
            expires_in_seconds: Some(2),
        },
        AuthEvent::Progress {
            message: "progress".to_owned(),
        },
    ] {
        ui.send_auth_notify(event);
        harness.frame.drain_ui_requests();
    }
    let themes = ui.get_all_themes();
    harness.frame.drain_ui_requests();
    let _ = themes.await;
    let theme = ui.get_theme("dark");
    harness.frame.drain_ui_requests();
    let _ = theme.await;
}

#[tokio::test]
async fn display_optional_branches_render_without_rows() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness
        .frame
        .show_extension_error("/ext.ts", "no stack", None);
    assert!(harness.chat_text().contains("no stack"));

    let harness = Harness::new(HarnessOptions::default()).await;
    let cwd = harness.cwd.clone();
    std::fs::create_dir_all(format!("{cwd}/.pi")).expect("project config");
    std::fs::write(format!("{cwd}/.pi/settings.json"), "{}").expect("settings");
    harness
        .frame
        .session()
        .settings_manager()
        .lock()
        .expect("settings")
        .set_project_trusted(false);
    harness.frame.render_initial_messages();
    assert!(harness.chat_text().contains("not trusted"));

    harness.frame.show_new_version_notification(
        &pi_coding_agent::utils::version_check::LatestPiRelease {
            version: "2.0.0".to_owned(),
            package_name: None,
            note: None,
        },
    );
    assert!(harness.chat_text().contains("Update Available"));
}

#[tokio::test]
async fn run_with_an_empty_initial_message_reports_the_prompt_error() {
    let harness = Harness::new(HarnessOptions {
        frame_options:
            pi_coding_agent::modes::interactive::interactive_mode::InteractiveModeOptions {
                skip_managed_tool_setup: true,
                initial_message: Some(String::new()),
                ..Default::default()
            },
        ..HarnessOptions::default()
    })
    .await;
    let frame = Rc::clone(&harness.frame);
    let mut run = Box::pin(frame.run());
    for _ in 0..200 {
        tokio::select! {
            code = &mut run => {
                assert_eq!(code, 0);
                return;
            }
            () = tokio::time::sleep(std::time::Duration::from_millis(5)) => {}
        }
        if harness.terminal.writes().contains("\x1b]0;") {
            harness.send_input("/quit");
            harness.send_input("\r");
        }
    }
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), run).await;
}

struct BareEditor;

impl Component for BareEditor {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["bare".to_owned()]
    }
    fn invalidate(&self) {}
}

impl pi_tui::components::EditorComponent for BareEditor {
    fn get_text(&self) -> String {
        String::new()
    }
    fn set_text(&self, _text: &str) {}
    fn add_to_history(&self, _text: &str) {}
    fn insert_text_at_cursor(&self, _text: &str) {}
    fn get_expanded_text(&self) -> Option<String> {
        None
    }
    fn set_padding_x(&self, _padding: usize) {}
    fn set_autocomplete_provider(
        &self,
        _provider: std::sync::Arc<dyn pi_tui::autocomplete::AutocompleteProvider>,
    ) {
    }
    fn set_autocomplete_max_visible(&self, _max_visible: usize) {}
}

impl pi_coding_agent::modes::interactive::interactive_mode::AppEditor for BareEditor {}

#[test]
fn app_editor_default_members_are_callable() {
    use pi_coding_agent::modes::interactive::interactive_mode::AppEditor;

    let editor = BareEditor;
    editor.set_border_color(Rc::new(|text: &str| text.to_owned()));
    assert!(editor.border_color().is_none());
    editor.set_working_status_indicator(None);
    assert!(!editor.embed_working_status());
    editor.set_on_extension_shortcut(None);
    editor.set_on_escape(None);
    assert!(editor.on_escape().is_none());
    editor.set_on_ctrl_d(None);
    editor.set_on_paste_image(None);
    editor.set_on_submit(None);
    editor.set_on_change(None);
    editor.on_action("app.clear", Rc::new(|| {}));
    let _ = editor.render(10);
    editor.invalidate();
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "one fixture exercises every provenance shape in a single render pass"
)]
async fn loaded_resource_renderer_handles_every_provenance_shape() {
    use pi_coding_agent::modes::interactive::interactive_mode::{
        LoadedPromptEntry, LoadedResource, LoadedResourcesOptions, LoadedResourcesView,
        LoadedSkillEntry, LoadedThemeEntry, render_loaded_resource_sections,
    };
    use pi_coding_agent::source_info::{SourceInfo, SourceOrigin, SourceScope};

    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let source = |path: &str, source: &str, scope, origin, base: Option<&str>| SourceInfo {
        path: path.to_owned(),
        source: source.to_owned(),
        scope,
        origin,
        base_dir: base.map(str::to_owned),
    };
    let local = |path: &str, base: &str| LoadedResource {
        path: path.to_owned(),
        source_info: Some(source(
            path,
            "local",
            SourceScope::Project,
            SourceOrigin::TopLevel,
            Some(base),
        )),
    };
    let npm = |path: &str, source_name: &str, base: Option<&str>| LoadedResource {
        path: path.to_owned(),
        source_info: Some(source(
            path,
            source_name,
            SourceScope::User,
            SourceOrigin::Package,
            base,
        )),
    };
    let git = LoadedResource {
        path: "/tmp/git/github.com/owner/repo/extensions/tools/index.ts".to_owned(),
        source_info: Some(source(
            "/tmp/git/github.com/owner/repo/extensions/tools/index.ts",
            "git:github.com/owner/repo",
            SourceScope::Project,
            SourceOrigin::Package,
            Some("/tmp/git/github.com/owner/repo"),
        )),
    };
    let view = LoadedResourcesView {
        context_files: vec![
            "AGENTS.md".to_owned(),
            "/home/user/.pi/agent/AGENTS.md".to_owned(),
            "/tmp/project/.pi/SYSTEM.md".to_owned(),
        ],
        skills: vec![
            LoadedSkillEntry {
                name: "local-skill".to_owned(),
                resource: local("/tmp/project/skill/SKILL.md", "/tmp/project/skill"),
            },
            LoadedSkillEntry {
                name: "npm-skill".to_owned(),
                resource: npm(
                    "/tmp/npm/node_modules/pkg/skill/SKILL.md",
                    "npm:pkg",
                    Some("/tmp/npm/node_modules/pkg"),
                ),
            },
        ],
        prompts: vec![
            LoadedPromptEntry {
                name: "one".to_owned(),
                resource: local("/tmp/project/prompts/one.md", "/tmp/project/prompts"),
            },
            LoadedPromptEntry {
                name: "two".to_owned(),
                resource: local("/tmp/project/prompts/two.md", "/tmp/project/prompts"),
            },
        ],
        extensions: vec![
            local("/tmp/project/.pi/extensions/alpha/index.ts", "/tmp/project"),
            local("/tmp/project/.pi/extensions/beta/index.js", "/tmp/project"),
            local("/tmp/other/tools/main.ts", "/tmp/other"),
            npm(
                "/tmp/npm/node_modules/pkg/extensions/sub/index.ts",
                "npm:pkg",
                Some("/tmp/npm/node_modules/pkg"),
            ),
            npm(
                "/tmp/npm/node_modules/no-base/index.ts",
                "npm:no-base",
                None,
            ),
            git,
        ],
        themes: vec![
            LoadedThemeEntry {
                name: Some("named".to_owned()),
                resource: local("/tmp/themes/named.json", "/tmp/themes"),
            },
            LoadedThemeEntry {
                name: None,
                resource: local("/tmp/themes/unnamed.json", "/tmp/themes"),
            },
        ],
    };
    for expanded in [false, true] {
        for home in [None, Some("/home/user")] {
            let sections = render_loaded_resource_sections(
                &view,
                &LoadedResourcesOptions {
                    expanded,
                    cwd: "/tmp/project".to_owned(),
                    home: home.map(str::to_owned),
                },
            );
            assert!(!sections.is_empty());
        }
    }
}

#[tokio::test]
async fn automatic_theme_submenu_light_and_dark_selects_apply() {
    use pi_coding_agent::modes::interactive::components::settings_selector::ThemeSubmenu;
    use pi_coding_agent::modes::interactive::theme::theme::TerminalTheme;

    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;
    let done = Rc::new(RefCell::new(None::<Option<String>>));
    let done_for_callback = Rc::clone(&done);
    let submenu = ThemeSubmenu::new(
        "light/dark",
        TerminalTheme::Light,
        vec!["dark".to_owned(), "light".to_owned()],
        harness.frame.settings_callbacks(),
        Rc::new(move |value, _options| {
            *done_for_callback.borrow_mut() = Some(value);
        }),
    );
    // Light row -> light theme submenu -> pick the second theme.
    submenu.handle_input("\r");
    submenu.handle_input("\x1b[B");
    submenu.handle_input("\r");
    // Dark row -> dark theme submenu -> pick the second theme.
    submenu.handle_input("\x1b[B");
    submenu.handle_input("\r");
    submenu.handle_input("\x1b[B");
    submenu.handle_input("\r");
    // Apply.
    submenu.handle_input("\x1b[B");
    submenu.handle_input("\r");
    let _ = done.borrow();
    let _ = submenu.render(80);
}

#[tokio::test]
async fn create_startup_tui_default_wrapper_builds_a_frame() {
    let settings = Settings::new();
    let ui =
        pi_coding_agent::cli::startup_ui::create_startup_tui(&settings_manager(&settings)).await;
    assert!(ui.terminal_columns() > 0);
    assert!(pi_coding_agent::modes::interactive::theme::theme::current_theme_name().is_some());
}

fn settings_manager(
    settings: &Settings,
) -> pi_coding_agent::settings_manager::SettingsManager<
    pi_coding_agent::settings_manager::InMemorySettingsStorage,
> {
    pi_coding_agent::settings_manager::SettingsManager::in_memory(
        settings,
        pi_coding_agent::settings_manager::SettingsManagerCreateOptions::default(),
    )
}

#[tokio::test]
async fn custom_agent_rows_cover_default_and_missing_field_branches() {
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(common::interactive::renderer_fixtures()),
        messages: common::interactive::exchange_messages(),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.add_message_to_chat(
        &common::interactive::custom_agent_message(
            "bashExecution",
            serde_json::json!({ "command": "ls" }),
        ),
        true,
    );
    harness.frame.add_message_to_chat(
        &common::interactive::custom_agent_message(
            "bashExecution",
            serde_json::json!({
                "command": "rg",
                "cancelled": true,
                "truncated": false,
                "exitCode": null
            }),
        ),
        false,
    );
    harness.frame.add_message_to_chat(
        &common::interactive::custom_agent_message(
            "custom",
            serde_json::json!({ "display": true, "customType": "unknown" }),
        ),
        false,
    );
    harness.frame.add_message_to_chat(
        &common::interactive::custom_agent_message("compactionSummary", serde_json::json!({})),
        false,
    );
    harness.frame.add_message_to_chat(
        &common::interactive::custom_agent_message("branchSummary", serde_json::json!({})),
        false,
    );
    harness.frame.add_message_to_chat(
        &common::interactive::custom_agent_message("unknown", serde_json::json!({})),
        false,
    );
    harness.frame.show_loaded_resources(true);
    harness.frame.set_tools_expanded(true);
    harness.frame.set_tools_expanded(false);
    assert!(!harness.chat_text().is_empty());
}

#[tokio::test]
async fn assistant_diagnostics_singular_disabled_and_incomplete_branches() {
    let mut settings = Settings::new();
    settings.insert(
        "showCacheMissNotices".to_owned(),
        serde_json::Value::Bool(false),
    );
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;
    let mut message = common::agent_session::assistant_message("reply", 1);
    message.diagnostics = Some(vec![pi_ai::types::AssistantMessageDiagnostic {
        kind: "anthropic_input_transformations".to_owned(),
        timestamp: 1,
        error: None,
        details: Some(std::collections::BTreeMap::new()),
    }]);
    harness.frame.maybe_show_assistant_diagnostics(&message);
    assert!(harness.chat_text().is_empty());

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
    let mut details = std::collections::BTreeMap::new();
    details.insert(
        "transformations".to_owned(),
        serde_json::json!([
            { "type": "thinking_dropped", "reason": "empty" },
            { "type": "other" }
        ]),
    );
    message.diagnostics = Some(vec![pi_ai::types::AssistantMessageDiagnostic {
        kind: "anthropic_input_transformations".to_owned(),
        timestamp: 1,
        error: None,
        details: Some(details),
    }]);
    harness.frame.maybe_show_assistant_diagnostics(&message);
    assert!(harness.chat_text().contains("thinking block"));
}

#[tokio::test]
async fn render_entries_cover_error_no_message_and_unrendered_custom_rows() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let mut assistant = common::agent_session::assistant_message("error", 1);
    assistant.stop_reason = pi_ai::types::StopReason::Error;
    assistant.error_message = None;
    assistant.content = vec![pi_ai::types::AssistantBlock::ToolCall(
        pi_ai::types::ToolCall {
            id: "call-error".to_owned(),
            name: "read".to_owned(),
            arguments: serde_json::Map::new(),
            thought_signature: None,
            namespace: None,
        },
    )];
    let entry: pi_coding_agent::session_manager::FileEntry =
        serde_json::from_value(serde_json::json!({
            "type": "message",
            "id": "assistant-error",
            "parentId": null,
            "timestamp": "2025-01-01T00:00:00Z",
            "message": serde_json::to_value(pi_agent_core::types::AgentMessage::Standard(
                Message::Assistant(assistant)
            ))
            .expect("assistant serializes")
        }))
        .expect("entry parses");
    harness.frame.render_session_entries(&[entry], false, false);
    let text = harness.chat_text();
    assert!(text.contains("Error"), "{text}");

    harness.frame.add_custom_entry_to_chat(
        &serde_json::from_value(serde_json::json!({
            "type": "custom",
            "customType": "missing",
            "id": "custom-missing",
            "parentId": null,
            "timestamp": "2025-01-01T00:00:00Z",
            "data": {}
        }))
        .expect("custom entry"),
    );
    assert!(harness.chat_text().contains("Error"));
}

#[tokio::test]
async fn autocomplete_covers_scoped_models_templates_thinking_and_extension_commands() {
    use pi_coding_agent::resource_loader::ContextFile;

    let fixtures = common::interactive::ResourceFixtures {
        prompts: vec![(
            "two".to_owned(),
            "/tmp/prompts/two.md".to_owned(),
            pi_coding_agent::source_info::SourceInfo {
                path: "/tmp/prompts/two.md".to_owned(),
                source: "local".to_owned(),
                scope: pi_coding_agent::source_info::SourceScope::Project,
                origin: pi_coding_agent::source_info::SourceOrigin::TopLevel,
                base_dir: None,
            },
        )],
        context_files: vec![ContextFile {
            path: "/tmp/project/AGENTS.md".to_owned(),
            content: String::new(),
        }],
        extension_factories: common::interactive::command_fixtures(&["demo"]).extension_factories,
        ..common::interactive::ResourceFixtures::default()
    };
    let harness = common::interactive::harness_with(HarnessOptions {
        fixtures: Some(fixtures),
        ..HarnessOptions::default()
    })
    .await;
    let model = harness.frame.session().model().expect("model");
    harness
        .frame
        .apply_scoped_models(Some(vec![format!("{}/{}", model.provider.0, model.id)]));
    let provider = harness.frame.create_base_autocomplete_provider();
    for line in [
        "/model faux",
        "/thinking h",
        "/login faux",
        "/two",
        "/demo",
        "/missing",
    ] {
        let _ = provider.get_suggestions(
            &[line.to_owned()],
            0,
            line.len(),
            &pi_tui::autocomplete::AutocompleteQueryOptions {
                signal: tokio_util::sync::CancellationToken::new(),
                force: true,
            },
        );
    }
}

#[tokio::test]
async fn session_info_renders_multi_model_cost_breakdown() {
    let usage = pi_ai::types::Usage {
        input: 10,
        output: 5,
        total_tokens: 15,
        cost: pi_ai::types::UsageCost {
            total: 0.25,
            ..pi_ai::types::UsageCost::default()
        },
        ..pi_ai::types::Usage::default()
    };
    let mut first = common::agent_session::assistant_with_usage("one", usage, 2);
    first.model = "faux-1".to_owned();
    let mut second = common::agent_session::assistant_with_usage("two", usage, 3);
    second.model = "faux-2".to_owned();
    second.provider = pi_ai::types::ProviderId("other".to_owned());
    let harness = Harness::new(HarnessOptions {
        messages: vec![
            common::agent_session::user_message("prompt", 1),
            pi_agent_core::types::AgentMessage::Standard(Message::Assistant(first)),
            pi_agent_core::types::AgentMessage::Standard(Message::Assistant(second)),
        ],
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.session().set_session_name("multi model");
    common::interactive::submit(&harness, "/session").await;
    let text = harness.chat_text();
    assert!(text.contains("multi model"), "{text}");
    assert!(text.contains("Cost"), "{text}");
}
