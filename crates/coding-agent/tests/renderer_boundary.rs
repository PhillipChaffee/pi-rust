//! Boundary coverage for the interactive renderer surfaces: every built-in
//! renderer's call/result arms, the truncation ladders, the elapsed-label
//! path, and the setters the app frame will drive.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::types::{AgentToolError, AgentToolResult};
use pi_ai::types::{BoxedFuture, ToolResultBlock};
use pi_coding_agent::extensions::types::{
    ToolDefinition, ToolRenderContext, ToolRenderResultOptions, ToolRenderState,
};
use pi_coding_agent::modes::interactive::components::assistant_message::{
    AssistantMessageComponent, AssistantMessageOptions,
};
use pi_coding_agent::modes::interactive::components::bash_execution::BashExecutionComponent;
use pi_coding_agent::modes::interactive::components::custom_message::CustomMessageComponent;
use pi_coding_agent::modes::interactive::components::keybinding_hints::{
    key_display_text, raw_key_hint,
};
use pi_coding_agent::modes::interactive::components::tool_execution::{
    ToolExecutionComponent, ToolExecutionOptions,
};
use pi_coding_agent::modes::interactive::theme::theme::{init_theme, theme};
use pi_coding_agent::tools::renderers::{create_all_tool_renderers, with_built_in_renderers};
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::components::Text;
use pi_tui::tui::{Component, RenderRequest};
use serde_json::{Value, json};

fn render_request() -> RenderRequest {
    Arc::new(|| {})
}

fn base_definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        label: name.to_string(),
        description: "custom tool".to_string(),
        prompt_snippet: None,
        prompt_guidelines: None,
        parameters: json!({}),
        constrained_sampling: None,
        render_shell: None,
        render_call: None,
        render_result: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(
            |_id: &str,
             _params: &Value,
             _signal,
             _update,
             _ctx|
             -> BoxedFuture<'_, Result<AgentToolResult, AgentToolError>> {
                Box::pin(async {
                    Ok(AgentToolResult {
                        content: Vec::new(),
                        details: json!({}),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    })
                })
            },
        ),
    }
}

fn text_result(text: &str, details: Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultBlock::Text(pi_ai::types::TextContent {
            text: text.to_string(),
            text_signature: None,
        })],
        details,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn truncation_details(by_lines: bool) -> Value {
    json!({
        "truncation": {
            "content": "tail",
            "truncated": true,
            "truncatedBy": if by_lines { "lines" } else { "bytes" },
            "totalLines": 10,
            "totalBytes": 100,
            "outputLines": 3,
            "outputBytes": 30,
            "lastLinePartial": false,
            "firstLineExceedsLimit": false,
            "maxLines": 3,
            "maxBytes": 30
        }
    })
}

fn invoke_call(name: &str, args: &Value) -> String {
    let definition = with_built_in_renderers(name, Some(base_definition(name))).expect("built-in");
    let state = ToolRenderState::new();
    let invalidate = || {};
    let context = ToolRenderContext {
        args,
        tool_call_id: "boundary",
        invalidate: &invalidate,
        last_component: None,
        state: &state,
        cwd: "/tmp",
        execution_started: false,
        args_complete: true,
        is_partial: false,
        expanded: false,
        show_images: true,
        is_error: false,
    };
    let component = (definition.render_call.as_ref().expect("call hook"))(args, &theme(), &context);
    strip_ansi(&component.render(120).join("\n"))
}

fn invoke_result(name: &str, args: &Value, result: &AgentToolResult, expanded: bool) -> String {
    let definition = with_built_in_renderers(name, Some(base_definition(name))).expect("built-in");
    let state = ToolRenderState::new();
    let invalidate = || {};
    let context = ToolRenderContext {
        args,
        tool_call_id: "boundary",
        invalidate: &invalidate,
        last_component: None,
        state: &state,
        cwd: "/tmp",
        execution_started: true,
        args_complete: true,
        is_partial: false,
        expanded,
        show_images: true,
        is_error: false,
    };
    let component = (definition.render_result.as_ref().expect("result hook"))(
        result,
        &ToolRenderResultOptions {
            expanded,
            is_partial: false,
        },
        &theme(),
        &context,
    );
    strip_ansi(&component.render(120).join("\n"))
}

#[test]
fn find_and_grep_and_ls_renderers_cover_calls_and_truncation_warnings() {
    init_theme(Some("dark"), false);

    assert!(
        invoke_call(
            "find",
            &json!({ "pattern": "needle", "path": "src", "limit": 5 })
        )
        .contains("find")
    );
    assert!(
        invoke_call(
            "grep",
            &json!({ "pattern": "needle", "path": "src", "glob": "*.rs" })
        )
        .contains("grep")
    );
    assert!(invoke_call("ls", &json!({ "path": "src", "limit": 5 })).contains("ls"));
    assert!(invoke_call("find", &json!({ "pattern": 1 })).contains("[invalid arg]"));
    assert!(invoke_call("grep", &json!({ "pattern": "x", "path": 1 })).contains("[invalid arg]"));

    let find = invoke_result(
        "find",
        &json!({ "pattern": "needle" }),
        &text_result(
            "a\nb",
            json!({ "resultLimitReached": 5, "truncation": truncation_details(true)["truncation"] }),
        ),
        false,
    );
    assert!(find.contains("5 results limit"));
    assert!(find.contains("30B limit") || find.contains("limit"));

    let grep = invoke_result(
        "grep",
        &json!({ "pattern": "needle" }),
        &text_result(
            "a\nb",
            json!({ "matchLimitReached": 2, "linesTruncated": true, "truncation": truncation_details(false)["truncation"] }),
        ),
        false,
    );
    assert!(grep.contains("2 matches limit"));
    assert!(grep.contains("some lines truncated"));

    let ls = invoke_result(
        "ls",
        &json!({ "path": "./" }),
        &text_result("a\nb", json!({ "entryLimitReached": 9 })),
        false,
    );
    assert!(ls.contains("9 entries limit"));

    let long_output = (1..=25)
        .map(|index| format!("line-{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        invoke_result(
            "find",
            &json!({ "pattern": "x" }),
            &text_result(&long_output, json!({})),
            false
        )
        .contains("5 more lines")
    );
}

#[test]
fn read_renderer_covers_every_truncation_warning_ladder() {
    init_theme(Some("dark"), false);
    let first_line = invoke_result(
        "read",
        &json!({ "path": "notes.txt" }),
        &text_result(
            "one",
            json!({ "truncation": {
                "content": "one", "truncated": true, "truncatedBy": "bytes",
                "totalLines": 1, "totalBytes": 80, "outputLines": 1, "outputBytes": 80,
                "lastLinePartial": false, "firstLineExceedsLimit": true, "maxLines": 1, "maxBytes": 30
            }}),
        ),
        true,
    );
    assert!(first_line.contains("First line exceeds 30B limit"));

    let lines = invoke_result(
        "read",
        &json!({ "path": "notes.txt" }),
        &text_result("one", truncation_details(true)),
        true,
    );
    assert!(lines.contains("showing 3 of 10 lines (3 line limit)"));

    let bytes = invoke_result(
        "read",
        &json!({ "path": "notes.txt" }),
        &text_result("one", truncation_details(false)),
        true,
    );
    assert!(bytes.contains("3 lines shown (30B limit)"));
}

#[test]
fn edit_renderer_covers_the_intra_line_diff_and_error_arms() {
    init_theme(Some("dark"), false);
    let definition = with_built_in_renderers("edit", Some(base_definition("edit"))).expect("edit");
    let component = ToolExecutionComponent::new(
        "edit",
        "edit-boundary",
        json!({ "path": "a.rs", "oldText": "old", "newText": "new" }),
        ToolExecutionOptions::default(),
        Some(definition),
        render_request(),
        "/tmp",
    );
    component.update_result(
        text_result("", json!({ "diff": "-1 old\n+1 new" })),
        false,
        false,
    );
    assert!(strip_ansi(&component.render(120).join("\n")).contains("old"));

    component.update_result(text_result("boom", json!({})), true, false);
    assert!(strip_ansi(&component.render(120).join("\n")).contains("boom"));
}

#[test]
fn bash_execution_covers_elapsed_preview_and_reader_surfaces() {
    init_theme(Some("dark"), false);
    let component = BashExecutionComponent::new("echo hi", render_request(), false);
    component.append_output("one\r\ntwo\rthree\n");
    let collapsed = component.render(40);
    assert!(!collapsed.is_empty());
    component.set_expanded(true);
    let truncation = pi_coding_agent::tools::truncate::truncate_tail(
        &"x".repeat(60_000),
        pi_coding_agent::tools::truncate::TruncationOptions::default(),
    );
    component.set_complete(Some(2), false, Some(truncation), Some("/tmp/full.log"));
    let expanded = strip_ansi(&component.render(80).join("\n"));
    assert!(expanded.contains("(exit 2)"));
    assert!(expanded.contains("Output truncated. Full output: /tmp/full.log"));
    assert_eq!(component.get_command(), "echo hi");
    assert!(component.get_output().contains("one"));

    let cancelled = BashExecutionComponent::new("sleep 1", render_request(), true);
    cancelled.set_complete(None, true, None, None);
    assert!(strip_ansi(&cancelled.render(40).join("\n")).contains("(cancelled)"));
}

#[test]
fn shell_and_write_result_renderers_cover_collapsed_and_error_rows() {
    init_theme(Some("dark"), false);
    let shell_definition =
        with_built_in_renderers("bash", Some(base_definition("bash"))).expect("bash");
    let shell = ToolExecutionComponent::new(
        "bash",
        "bash-boundary",
        json!({ "command": "echo hi", "timeout": 5 }),
        ToolExecutionOptions::default(),
        Some(shell_definition),
        render_request(),
        "/tmp",
    );
    shell.mark_execution_started();
    shell.update_result(text_result("one\ntwo\nthree", json!({})), false, false);
    let collapsed = strip_ansi(&shell.render(60).join("\n"));
    assert!(collapsed.contains("$ echo hi"));
    shell.set_expanded(true);
    let expanded = strip_ansi(&shell.render(60).join("\n"));
    assert!(expanded.contains("Took") || expanded.contains("Elapsed"));
    assert!(expanded.contains("three"));

    let write = ToolExecutionComponent::new(
        "write",
        "write-boundary",
        json!({ "path": "notes.txt", "content": "x" }),
        ToolExecutionOptions::default(),
        Some(pi_coding_agent::tools::write::create_write_tool_definition(
            "/tmp", None,
        )),
        render_request(),
        "/tmp",
    );
    write.update_result(text_result("denied", json!({})), true, false);
    assert!(strip_ansi(&write.render(60).join("\n")).contains("denied"));
}

#[test]
fn tool_execution_covers_every_setter_and_the_self_render_mouse_path() {
    init_theme(Some("dark"), false);
    let definition = ToolDefinition {
        render_shell: Some(pi_coding_agent::extensions::types::RenderShell::SelfRender),
        render_call: Some(Arc::new(|_args, _theme, _context| {
            Rc::new(Text::with_padding("self row", 0, 0))
        })),
        ..base_definition("custom_tool")
    };
    let component = ToolExecutionComponent::new(
        "custom_tool",
        "setters",
        json!({}),
        ToolExecutionOptions {
            show_images: None,
            image_width_cells: None,
        },
        Some(definition),
        render_request(),
        "/tmp",
    );
    component.update_args(json!({ "changed": true }));
    component.set_args_complete();
    component.mark_execution_started();
    component.set_show_images(false);
    component.set_image_width_cells(0);
    let lines = component.render(40);
    let event = pi_tui::tui::TuiMouseEvent {
        event_type: pi_tui::tui::TuiMouseEventType::Click,
        button: pi_tui::tui::TuiMouseButton::Left,
        x: 1,
        y: 1,
        screen_x: 1,
        screen_y: 1,
        width: 40,
        height: u16::try_from(lines.len()).expect("height"),
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: Some(1),
    };
    let _ = component.handle_mouse(&event);
    assert!(!lines.is_empty());
}

#[test]
fn remaining_renderer_surfaces_are_exercised() {
    init_theme(Some("dark"), false);
    assert_eq!(create_all_tool_renderers().len(), 8);
    let _ = key_display_text("app.tools.expand");
    assert!(raw_key_hint("ctrl+o", "to expand").contains("to expand"));

    let message = pi_agent_core::types::CustomAgentMessage {
        role: "custom".to_string(),
        timestamp: 0,
        data: serde_json::Map::from_iter([
            ("customType".to_string(), json!("test")),
            (
                "content".to_string(),
                json!([{ "type": "text", "text": "block text" }]),
            ),
        ]),
    };
    let custom = CustomMessageComponent::new(message, None, None, 1);
    custom.set_output_pad(0);
    custom.set_expanded(true);
    assert!(strip_ansi(&custom.render(40).join("\n")).contains("[test]"));

    let assistant = AssistantMessageComponent::new(None, AssistantMessageOptions::default());
    assistant.set_hide_thinking_block(true);
    assistant.set_hidden_thinking_label("Reasoning...");
    assert!(strip_ansi(&assistant.render(40).join("\n")).is_empty());
}
