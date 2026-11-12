//! Coverage of the frame's transcript rendering, notifications, queued
//! messages, status machine, extension widgets/footer/header, terminal
//! listeners, cache-miss and billing notices, and startup notices.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the message and settings shapes; a failure is a test bug"
)]
#![allow(
    clippy::large_futures,
    reason = "the acceptance suites await one frame operation at a time"
)]

mod common;

use std::rc::Rc;
use std::sync::Arc;

use common::interactive::{Harness, HarnessOptions, ResourceFixtures};
use pi_agent_core::types::{AgentMessage, CustomAgentMessage};
use pi_ai::types::{
    AssistantBlock, AssistantMessage, Message, TextContent, ToolCall, ToolResultBlock,
    ToolResultMessage, Usage, UsageCost,
};
use pi_coding_agent::extensions::host::UiCallHandler;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::source_info::{SourceInfo, SourceOrigin, SourceScope};
use pi_tui::tui::{Component, TuiInputListener};

fn source(path: &str, source: &str, scope: SourceScope) -> SourceInfo {
    SourceInfo {
        path: path.to_owned(),
        source: source.to_owned(),
        scope,
        origin: SourceOrigin::TopLevel,
        base_dir: None,
    }
}

fn renderer_fixtures() -> ResourceFixtures {
    ResourceFixtures {
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
                    let transformer: pi_coding_agent::extensions::types::MarkdownTransformer =
                        Arc::new(|text, _context| text.replace("@@", "!"));
                    api.register_markdown_transformer(transformer)?;
                    Ok(())
                })
            }),
        )],
        ..ResourceFixtures::default()
    }
}

fn assistant_with_tool_call(stop: pi_ai::types::StopReason) -> AssistantMessage {
    let mut message = common::agent_session::assistant_message("thinking", 1);
    message.content = vec![AssistantBlock::ToolCall(ToolCall {
        id: "call-1".to_owned(),
        name: "read".to_owned(),
        arguments: serde_json::Map::new(),
        thought_signature: None,
        namespace: None,
    })];
    message.stop_reason = stop;
    message.error_message = Some("tool failed".to_owned());
    message
}

fn tool_result(tool_call_id: &str) -> AgentMessage {
    AgentMessage::Standard(Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call_id.to_owned(),
        tool_name: "read".to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: "file body".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 2,
    }))
}

fn settings_with(key: &str, value: serde_json::Value) -> Settings {
    let mut settings = Settings::new();
    settings.insert(key.to_owned(), value);
    settings
}

#[tokio::test]
async fn add_message_to_chat_covers_user_skill_assistant_and_tool_results() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let empty = AgentMessage::Standard(Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Blocks(Vec::new()),
        timestamp: 0,
    }));
    harness.frame.add_message_to_chat(&empty, false);
    assert!(harness.frame.chat_container().children.borrow().is_empty());

    let user = common::agent_session::user_message("plain", 1);
    harness.frame.add_message_to_chat(&user, true);
    assert!(harness.chat_text().contains("plain"));

    let skill = common::agent_session::user_message(
        "<skill name=\"demo\" location=\"/x\">\nbody\n</skill>\n\nask",
        2,
    );
    harness.frame.add_message_to_chat(&skill, true);
    assert!(harness.chat_text().contains("skill"));

    let assistant = AgentMessage::Standard(Message::Assistant(
        common::agent_session::assistant_message("reply", 3),
    ));
    harness.frame.add_message_to_chat(&assistant, false);
    harness
        .frame
        .add_message_to_chat(&tool_result("call-1"), false);
    assert!(harness.chat_text().contains("reply"));
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "call sites read as the JSON literal they pass"
)]
fn custom(role: &str, data: serde_json::Value) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage {
        role: role.to_owned(),
        timestamp: 1,
        data: data.as_object().cloned().unwrap_or_default(),
    })
}

#[tokio::test]
async fn custom_agent_messages_cover_every_branch() {
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(renderer_fixtures()),
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.set_tools_expanded(true);

    harness.frame.add_message_to_chat(
        &custom(
            "bashExecution",
            serde_json::json!({
                "command": "ls",
                "output": "a\nb",
                "exitCode": 0,
                "cancelled": false,
                "truncated": true,
                "fullOutputPath": "/tmp/full.log"
            }),
        ),
        false,
    );
    assert!(harness.chat_text().contains("ls"));

    let before = harness.frame.chat_container().children.borrow().len();
    harness.frame.add_message_to_chat(
        &custom(
            "custom",
            serde_json::json!({ "display": false, "customType": "note" }),
        ),
        false,
    );
    assert_eq!(
        harness.frame.chat_container().children.borrow().len(),
        before
    );

    harness.frame.add_message_to_chat(
        &custom(
            "custom",
            serde_json::json!({ "display": true, "customType": "note" }),
        ),
        false,
    );
    assert!(harness.chat_text().contains("CUSTOM MESSAGE"));

    harness.frame.add_message_to_chat(
        &custom(
            "compactionSummary",
            serde_json::json!({ "summary": "compact body", "tokensBefore": 99 }),
        ),
        false,
    );
    harness.frame.add_message_to_chat(
        &custom(
            "branchSummary",
            serde_json::json!({ "summary": "branch body", "fromId": "abc" }),
        ),
        false,
    );
    harness
        .frame
        .add_message_to_chat(&custom("unknown", serde_json::json!({})), false);
    let text = harness.chat_text();
    assert!(text.contains("branch body"), "{text}");
}

#[tokio::test]
async fn render_session_entries_renders_tools_results_and_custom_entries() {
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(renderer_fixtures()),
        settings: settings_with("showCacheMissNotices", serde_json::Value::Bool(true)),
        seed_manager: Some(Box::new(|manager: &mut SessionManager| {
            manager
                .append_message(common::agent_session::user_message("prompt", 1))
                .expect("user");
            let aborted = AgentMessage::Standard(Message::Assistant(assistant_with_tool_call(
                pi_ai::types::StopReason::Aborted,
            )));
            manager.append_message(aborted).expect("aborted tool call");
            let call = AgentMessage::Standard(Message::Assistant(assistant_with_tool_call(
                pi_ai::types::StopReason::Stop,
            )));
            manager.append_message(call).expect("tool call");
            manager
                .append_message(tool_result("call-1"))
                .expect("tool result");
            manager
                .append_custom_entry("note-entry", Some(serde_json::json!({ "text": "x" })))
                .expect("custom entry");
            manager
                .append_custom_entry("unrendered", None)
                .expect("custom entry without renderer");
            let first_kept = manager
                .append_message(common::agent_session::user_message("kept", 3))
                .expect("kept");
            manager
                .append_compaction(
                    "summary",
                    &first_kept,
                    123,
                    None,
                    None,
                    Some(Usage {
                        input: 10,
                        output: 5,
                        total_tokens: 15,
                        cost: UsageCost {
                            total: 0.05,
                            ..UsageCost::default()
                        },
                        ..Usage::default()
                    }),
                )
                .expect("compaction");
        })),
        ..HarnessOptions::default()
    })
    .await;
    let entries = harness.frame.session_entries();
    harness.frame.render_session_entries(&entries, true, true);
    let text = harness.chat_text();
    assert!(text.contains("CUSTOM ENTRY"), "{text}");
    assert!(text.contains("Compaction: 15 tokens billed"), "{text}");
}

fn custom_entry(custom_type: &str) -> pi_coding_agent::session_manager::SessionEntry {
    serde_json::from_value(serde_json::json!({
        "type": "custom",
        "customType": custom_type,
        "id": "custom-1",
        "parentId": null,
        "timestamp": "2025-01-01T00:00:00Z",
        "data": { "text": "x" }
    }))
    .expect("custom entry parses")
}

#[tokio::test]
async fn custom_entries_render_only_with_a_renderer_and_content() {
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(renderer_fixtures()),
        ..HarnessOptions::default()
    })
    .await;
    harness
        .frame
        .add_custom_entry_to_chat(&custom_entry("unrendered"));
    assert!(harness.chat_text().is_empty());
    harness
        .frame
        .add_custom_entry_to_chat(&custom_entry("note-entry"));
    assert!(harness.chat_text().contains("CUSTOM ENTRY"));
}

#[tokio::test]
async fn queue_rows_and_restore_cover_compaction_and_session_queues() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.queue_compaction_message(
        "steer me",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::Steer,
    );
    harness.frame.queue_compaction_message(
        "follow me",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::FollowUp,
    );
    harness.frame.update_pending_messages_display();
    let (steering, follow_up) = harness.frame.get_all_queued_messages();
    assert_eq!(steering, vec!["steer me".to_owned()]);
    assert_eq!(follow_up, vec!["follow me".to_owned()]);
    assert!(harness.frame.editor().get_text().is_empty());

    harness.frame.editor().set_text("current");
    assert_eq!(
        harness.frame.restore_queued_messages_to_editor(false, None),
        2
    );
    let text = harness.frame.editor().get_text();
    assert!(text.contains("steer me"), "{text}");
    assert!(text.contains("follow me"), "{text}");
    assert!(text.contains("current"), "{text}");
    assert_eq!(
        harness.frame.get_all_queued_messages(),
        (Vec::new(), Vec::new())
    );
    assert_eq!(
        harness.frame.restore_queued_messages_to_editor(true, None),
        0
    );
    assert_eq!(harness.frame.clear_all_queues(), (Vec::new(), Vec::new()));
}

#[tokio::test]
async fn flushing_compaction_queue_handles_static_and_prompt_messages() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.queue_compaction_message(
        "plain prompt",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::FollowUp,
    );
    harness.frame.flush_compaction_queue(true);
    assert_eq!(
        harness.frame.get_all_queued_messages(),
        (Vec::new(), Vec::new())
    );

    harness.frame.queue_compaction_message(
        "steered",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::Steer,
    );
    harness.frame.queue_compaction_message(
        "followed",
        pi_coding_agent::modes::interactive::interactive_mode::QueueDelivery::FollowUp,
    );
    harness.frame.flush_compaction_queue(false);
    assert_eq!(
        harness.frame.get_all_queued_messages(),
        (Vec::new(), Vec::new())
    );
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    harness.frame.flush_compaction_queue(false);
}

#[tokio::test]
async fn status_machine_covers_working_visibility_thinking_and_tools() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.add_message_to_chat(
        &AgentMessage::Standard(Message::Assistant(
            common::agent_session::assistant_message("reply", 1),
        )),
        false,
    );

    harness.frame.show_working_status_indicator();
    assert_eq!(
        harness.frame.active_status_kind(),
        Some(
            pi_coding_agent::modes::interactive::components::status_indicator::StatusIndicatorKind::Working
        )
    );
    harness.frame.set_working_visible(false);
    assert_eq!(harness.frame.active_status_kind(), None);
    harness.frame.set_working_visible(true);
    harness.frame.set_working_indicator(None);
    harness.frame.set_hidden_thinking_label(Some("Pondering"));
    harness.frame.update_thinking_block_visibility();
    harness.frame.toggle_thinking_block_visibility();
    assert!(harness.chat_text().contains("Thinking blocks:"));
    harness.frame.set_tools_expanded(false);
    harness.frame.toggle_tool_output_expansion();
    assert!(harness.chat_text().contains("Tool output: expanded"));
    harness.frame.toggle_tool_output_expansion();
    harness.frame.clear_editor();
    assert!(harness.frame.editor().get_text().is_empty());
}

#[tokio::test]
async fn status_helpers_build_every_kind_and_clear_selectively() {
    use pi_coding_agent::modes::interactive::components::status_indicator::StatusIndicatorKind;

    let harness = Harness::new(HarnessOptions::default()).await;
    for (kind, message) in [
        (StatusIndicatorKind::Working, "working"),
        (StatusIndicatorKind::Compaction, "compact"),
        (StatusIndicatorKind::BranchSummary, "branch"),
        (StatusIndicatorKind::Retry, "retry"),
    ] {
        let indicator = harness.frame.build_status(kind, message.to_owned(), None);
        harness.frame.show_status_indicator(indicator);
        assert_eq!(harness.frame.active_status_kind(), Some(kind));
    }
    harness
        .frame
        .clear_status_indicator(Some(StatusIndicatorKind::Retry));
    assert_eq!(harness.frame.active_status_kind(), None);
    harness.frame.show_working_status_indicator();
    harness
        .frame
        .clear_status_indicator(Some(StatusIndicatorKind::Retry));
    assert_eq!(
        harness.frame.active_status_kind(),
        Some(StatusIndicatorKind::Working)
    );
}

#[tokio::test]
async fn extension_widgets_footer_and_terminal_listeners_are_wired() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let ui = harness.frame.ui_context();

    let above = ui.call(
        "setWidget",
        Some(serde_json::json!({ "key": "above", "content": ["a"], "placement": "aboveEditor" })),
    );
    harness.frame.drain_ui_requests();
    let _ = above.await;
    let below = ui.call(
        "setWidget",
        Some(serde_json::json!({ "key": "below", "content": ["b"], "placement": "belowEditor" })),
    );
    harness.frame.drain_ui_requests();
    let _ = below.await;
    let clear = ui.call("setWidget", Some(serde_json::json!({ "key": "above" })));
    harness.frame.drain_ui_requests();
    let _ = clear.await;
    harness.frame.clear_extension_widgets();
    harness.frame.render_widgets();

    let footer: Rc<dyn Component> = Rc::new(pi_tui::components::Text::with_padding(
        "custom footer",
        0,
        0,
    ));
    harness.frame.set_extension_footer(Some(footer));
    harness.frame.set_extension_footer(None);
    let header: Rc<dyn Component> = Rc::new(pi_tui::components::Text::with_padding(
        "custom header",
        0,
        0,
    ));
    harness.frame.set_extension_header(Some(header));
    harness.frame.set_extension_header(None);

    let listener: TuiInputListener = Rc::new(|_data| None);
    let unsubscribe = harness
        .frame
        .add_extension_terminal_input_listener(Rc::clone(&listener));
    harness.frame.rebind_extension_terminal_input_listeners();
    harness
        .frame
        .add_extension_terminal_input_listener(listener);
    unsubscribe();
    harness.frame.clear_extension_terminal_input_listeners();
}

#[tokio::test]
async fn cache_miss_notices_cover_threshold_model_switch_and_idle_labels() {
    let mut previous = common::agent_session::assistant_message("previous", 1_000);
    previous.usage.cache_read = 30_000;
    let current = |cache_read: u64, timestamp: i64, model: &str| {
        let mut message = common::agent_session::assistant_message("current", timestamp);
        message.model = model.to_owned();
        message.usage.input = 30_000;
        message.usage.cache_read = cache_read;
        message.usage.total_tokens = 60_000;
        message
    };

    let harness = Harness::new(HarnessOptions {
        settings: settings_with("showCacheMissNotices", serde_json::Value::Bool(true)),
        seed_manager: Some(Box::new(move |manager: &mut SessionManager| {
            let previous = previous.clone();
            manager
                .append_message(AgentMessage::Standard(Message::Assistant(previous)))
                .expect("previous");
        })),
        ..HarnessOptions::default()
    })
    .await;

    harness
        .frame
        .maybe_show_cache_miss_notice(&current(0, 2_000, "faux-model"));
    harness
        .frame
        .maybe_show_cache_miss_notice(&current(0, 2_000, "other-model"));
    harness
        .frame
        .maybe_show_cache_miss_notice(&current(0, 30 * 60_000, "faux-model"));
    let text = harness.chat_text();
    assert!(text.contains("Cache miss"), "{text}");
    assert!(text.contains("Cache miss after"), "{text}");

    harness
        .frame
        .maybe_show_cache_miss_notice(&current(30_000, 2_000, "faux-model"));
}

#[tokio::test]
async fn cache_miss_notice_is_silent_when_disabled() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let message = common::agent_session::assistant_message("current", 2_000);
    harness.frame.maybe_show_cache_miss_notice(&message);
    assert!(harness.chat_text().is_empty());
}

#[tokio::test]
async fn startup_notices_render_collapsed_expanded_and_once() {
    let mut collapsed = Settings::new();
    collapsed.insert(
        "collapseChangelog".to_owned(),
        serde_json::Value::Bool(true),
    );
    let harness = Harness::new(HarnessOptions {
        settings: collapsed,
        ..HarnessOptions::default()
    })
    .await;
    harness
        .frame
        .set_changelog_markdown_for_test(Some("## Changes".to_owned()));
    harness.frame.show_startup_notices_if_needed();
    assert!(harness.chat_text().contains("Updated to v"));
    let count = harness.frame.chat_container().children.borrow().len();
    harness.frame.show_startup_notices_if_needed();
    assert_eq!(
        harness.frame.chat_container().children.borrow().len(),
        count
    );

    let harness = Harness::new(HarnessOptions::default()).await;
    harness
        .frame
        .set_changelog_markdown_for_test(Some("## Changes".to_owned()));
    harness.frame.show_startup_notices_if_needed();
    let text = harness.chat_text();
    assert!(text.contains("What's New"), "{text}");
    assert!(text.contains("Changes"), "{text}");

    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.show_startup_notices_if_needed();
    assert!(harness.chat_text().is_empty());
}

#[tokio::test]
async fn extension_version_and_package_notifications_render() {
    use pi_coding_agent::utils::version_check::LatestPiRelease;

    let harness = Harness::new(HarnessOptions::default()).await;
    harness
        .frame
        .show_extension_error("/ext.ts", "boom", Some("Error: boom\n  at one\n  at two"));
    harness
        .frame
        .show_extension_error("/ext.ts", "quiet", Some(""));
    let text = harness.chat_text();
    assert!(text.contains("Extension \"/ext.ts\" error: boom"), "{text}");

    harness
        .frame
        .show_new_version_notification(&LatestPiRelease {
            version: "9.9.9".to_owned(),
            package_name: None,
            note: Some("release note".to_owned()),
        });
    harness
        .frame
        .show_new_version_notification(&LatestPiRelease {
            version: "9.9.9".to_owned(),
            package_name: None,
            note: None,
        });
    assert!(harness.chat_text().contains("release note"));
    harness
        .frame
        .show_package_update_notification(&["pkg-a".to_owned(), "pkg-b".to_owned()]);
    assert!(harness.chat_text().contains("Package Updates Available"));
}

#[tokio::test]
async fn loaded_resources_render_every_section_and_diagnostic() {
    let fixtures = ResourceFixtures {
        context_files: vec![pi_coding_agent::resource_loader::ContextFile {
            path: "/tmp/project/AGENTS.md".to_owned(),
            content: String::new(),
        }],
        skills: vec![
            (
                "commit".to_owned(),
                "/tmp/skill/SKILL.md".to_owned(),
                source("/tmp/skill/SKILL.md", "local", SourceScope::Project),
            ),
            (
                "review".to_owned(),
                "/tmp/skill2/SKILL.md".to_owned(),
                source("/tmp/skill2/SKILL.md", "npm:pkg", SourceScope::User),
            ),
        ],
        prompts: vec![(
            "review".to_owned(),
            "/tmp/prompts/review.md".to_owned(),
            source("/tmp/prompts/review.md", "local", SourceScope::Project),
        )],
        extensions: vec![
            (
                "/tmp/ext/one.ts".to_owned(),
                Some(source("/tmp/ext/one.ts", "local", SourceScope::Project)),
            ),
            ("/tmp/ext/hidden.ts".to_owned(), None),
        ],
        themes: vec![pi_coding_agent::resource_loader::ThemeHandle {
            name: Some("midnight".to_owned()),
            source_path: Some("/tmp/themes/midnight.json".to_owned()),
            source_info: None,
        }],
        skill_diagnostics: vec![pi_coding_agent::diagnostics::ResourceDiagnostic {
            kind: pi_coding_agent::diagnostics::ResourceDiagnosticKind::Warning,
            message: "duplicate".to_owned(),
            path: Some("/tmp/skill/SKILL.md".to_owned()),
            collision: None,
        }],
        ..ResourceFixtures::default()
    };
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(fixtures),
        verbose: true,
        ..HarnessOptions::default()
    })
    .await;
    harness.frame.show_loaded_resources(true);
    let text = harness.resources_text();
    for expected in [
        "[Context]",
        "[Skills]",
        "[Prompts]",
        "[Extensions]",
        "[Themes]",
    ] {
        assert!(text.contains(expected), "{expected}: {text}");
    }
    assert!(text.contains("[Skill conflicts]"), "{text}");

    harness.frame.add_loaded_diagnostics(
        "Manual",
        &[
            pi_coding_agent::diagnostics::ResourceDiagnostic {
                kind: pi_coding_agent::diagnostics::ResourceDiagnosticKind::Error,
                message: "bad".to_owned(),
                path: None,
                collision: None,
            },
            pi_coding_agent::diagnostics::ResourceDiagnostic {
                kind: pi_coding_agent::diagnostics::ResourceDiagnosticKind::Warning,
                message: "warn".to_owned(),
                path: Some("/tmp/x".to_owned()),
                collision: None,
            },
        ],
    );
    assert!(harness.resources_text().contains("[Manual]"));
}

#[tokio::test]
async fn tool_options_and_registered_tool_definitions_read_the_session() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let options = harness.frame.tool_options();
    assert!(options.show_images.is_some());
    assert!(options.image_width_cells.is_some());
    let _ = harness.frame.registered_tool_definition("read");
    let _ = harness.frame.registered_tool_definition("no-such-tool");
    let assistant = harness.frame.assistant_options();
    let _ = assistant.output_pad;
    let theme = harness.frame.markdown_theme_with_settings();
    let _ = theme.code_block_indent;
    let transformers = harness.frame.markdown_transformers();
    assert!(!transformers.is_empty());
}
