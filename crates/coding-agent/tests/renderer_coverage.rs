//! Coverage for the renderer surfaces the ported suites do not drive:
//! component trait methods, debug implementations, the write highlighter's
//! incremental path, image rows, the diff branches, and the assistant
//! notice arms.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::types::{AgentToolResult, CustomAgentMessage};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, KnownApi, ProviderId, StopReason, TextContent,
    ToolResultBlock, Usage, UsageCost,
};
use pi_coding_agent::extensions::types::{
    MarkdownMessageType, MessageRenderOptions, ToolRenderContext, ToolRenderResultOptions,
    ToolRenderState,
};
use pi_coding_agent::modes::interactive::components::assistant_message::{
    AssistantMessageComponent, AssistantMessageOptions,
};
use pi_coding_agent::modes::interactive::components::bash_execution::BashExecutionComponent;
use pi_coding_agent::modes::interactive::components::custom_message::CustomMessageComponent;
use pi_coding_agent::modes::interactive::components::mermaid::{MermaidArt, Span};
use pi_coding_agent::modes::interactive::components::tool_execution::{
    ToolExecutionComponent, ToolExecutionOptions,
};
use pi_coding_agent::modes::interactive::components::user_message::UserMessageComponent;
use pi_coding_agent::modes::interactive::components::visual_truncate::{
    VisualTruncateResult, truncate_to_visual_lines,
};
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::tools::index::ToolName;
use pi_coding_agent::tools::renderers::{ToolRenderers, create_all_tool_renderers};
use pi_coding_agent::tools::truncate::{TruncationOptions, truncate_tail};
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::{Component, RenderRequest, TuiMouseButton, TuiMouseEvent, TuiMouseEventType};
use serde_json::{Value, json};

const PNG_1X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

fn render_request() -> RenderRequest {
    Arc::new(|| {})
}

const fn event(y: u16, height: u16) -> TuiMouseEvent {
    TuiMouseEvent {
        event_type: TuiMouseEventType::Click,
        button: TuiMouseButton::Left,
        x: 1,
        y,
        screen_x: 1,
        screen_y: y,
        width: 80,
        height,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: Some(1),
    }
}

fn text_result(text: &str, details: Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultBlock::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        })],
        details,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn image_result() -> AgentToolResult {
    AgentToolResult {
        content: vec![
            ToolResultBlock::Text(TextContent {
                text: "caption".to_string(),
                text_signature: None,
            }),
            ToolResultBlock::Image(pi_ai::types::ImageContent {
                data: PNG_1X1.to_string(),
                mime_type: "image/png".to_string(),
            }),
            ToolResultBlock::Image(pi_ai::types::ImageContent {
                data: "not-base64".to_string(),
                mime_type: "image/jpeg".to_string(),
            }),
        ],
        details: json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn assistant_message(content: Vec<AssistantBlock>, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content,
        api: Api::from(KnownApi::OpenaiResponses),
        provider: ProviderId("openai".to_string()),
        model: "m".to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 0,
            cost: UsageCost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

fn read_component() -> ToolExecutionComponent {
    ToolExecutionComponent::new(
        "read",
        "coverage",
        json!({ "path": "notes.txt" }),
        ToolExecutionOptions::default(),
        Some(pi_coding_agent::tools::read::create_read_tool_definition(
            "/tmp", None,
        )),
        render_request(),
        "/tmp",
    )
}

#[test]
fn component_surface_methods_and_debug_implementations() {
    init_theme(Some("dark"), false);

    let assistant = AssistantMessageComponent::new(
        Some(assistant_message(
            vec![AssistantBlock::Text(TextContent {
                text: "hello".to_string(),
                text_signature: None,
            })],
            StopReason::Stop,
        )),
        AssistantMessageOptions::default(),
    );
    let _ = format!("{assistant:?}");
    assert!(assistant.handle_mouse(&event(99, 1)).is_none());
    assistant.invalidate();
    assert!(!assistant.children().is_empty());

    let user = UserMessageComponent::new("hello", None, 1, Arc::new(Vec::new()));
    assert!(!user.children().is_empty());
    user.invalidate();
    let _ = user.handle_mouse(&event(99, 1));
    let _ = format!("{user:?}");

    let custom = CustomMessageComponent::new(
        CustomAgentMessage {
            role: "custom".to_string(),
            timestamp: 0,
            data: serde_json::Map::from_iter([
                ("customType".to_string(), json!("t")),
                ("content".to_string(), json!("body")),
            ]),
        },
        None,
        None,
        1,
    );
    assert!(!custom.children().is_empty());
    custom.invalidate();
    let _ = custom.handle_mouse(&event(99, 1));
    let _ = format!("{custom:?}");

    let shell = BashExecutionComponent::new("echo hi", render_request(), false);
    assert!(!shell.render(40).is_empty());
    assert!(!shell.children().is_empty());
    shell.invalidate();
    let _ = shell.handle_mouse(&event(0, 1));
    let _ = format!("{shell:?}");

    let tool = read_component();
    assert!(!tool.children().is_empty());
    tool.invalidate();
    let _ = tool.handle_mouse(&event(99, 1));
    let _ = format!("{tool:?}");

    let _ = format!("{:?}", ToolRenderState::default());
    let _ = ToolRenderResultOptions::default();
    let _ = MessageRenderOptions::default();
    let _ = ToolExecutionOptions::default();
    let _ = VisualTruncateResult::default();
    assert_eq!(
        truncate_to_visual_lines("", 3, 40, 0),
        VisualTruncateResult::default()
    );
    let _ = format!(
        "{:?}",
        MermaidArt {
            plain: vec!["x".to_string()],
            styled: vec![vec![Span {
                cls: "none",
                text: "x".to_string(),
            }]],
            width: 1,
            warnings: vec!["w".to_string()],
        }
    );
    let _ = MarkdownMessageType::User.as_str();

    let renderers: ToolRenderers = create_all_tool_renderers()
        .remove(&ToolName::Read)
        .expect("read renderers");
    let _ = format!("{renderers:?}");
    let args = json!({});
    let invalidate = || {};
    let state = ToolRenderState::new();
    let context = ToolRenderContext {
        args: &args,
        tool_call_id: "debug",
        invalidate: &invalidate,
        last_component: None,
        state: &state,
        cwd: "/tmp",
        execution_started: false,
        args_complete: false,
        is_partial: false,
        expanded: false,
        show_images: false,
        is_error: false,
    };
    let _ = format!("{context:?}");
}

#[test]
fn write_highlighter_covers_the_incremental_append_path() {
    init_theme(Some("dark"), false);
    let write = ToolExecutionComponent::new(
        "write",
        "write-incremental",
        json!({ "path": "a.rs", "content": "fn a() {}" }),
        ToolExecutionOptions::default(),
        Some(pi_coding_agent::tools::write::create_write_tool_definition(
            "/tmp", None,
        )),
        render_request(),
        "/tmp",
    );
    write.update_args(json!({ "path": "a.rs", "content": "fn a() {}\nfn b() {}" }));
    write.update_args(json!({ "path": "a.rs", "content": "fn a() {}\nfn b() {}\nfn c() {}" }));
    write.set_args_complete();
    let rendered = strip_ansi(&write.render(80).join("\n"));
    assert!(rendered.contains("fn a"));
    assert!(rendered.contains("fn c"));
}

#[test]
fn image_rows_cover_protocols_and_fallbacks() {
    use pi_tui::terminal_image::{
        ImageProtocol, TerminalCapabilities, reset_capabilities_cache, set_capabilities,
    };

    init_theme(Some("dark"), false);
    set_capabilities(TerminalCapabilities {
        images: Some(ImageProtocol::Iterm2),
        true_color: true,
        hyperlinks: true,
    });
    let tool = read_component();
    tool.update_result(image_result(), false, false);
    assert!(tool.render(80).len() > 1);
    tool.set_show_images(false);
    tool.set_expanded(true);
    let fallback = strip_ansi(&tool.render(80).join("\n"));
    assert!(fallback.contains("[image/png]"));

    let self_render = ToolExecutionComponent::new(
        "read",
        "self-image",
        json!({ "path": "notes.txt" }),
        ToolExecutionOptions::default(),
        {
            let mut definition =
                pi_coding_agent::tools::read::create_read_tool_definition("/tmp", None);
            definition.render_shell =
                Some(pi_coding_agent::extensions::types::RenderShell::SelfRender);
            Some(definition)
        },
        render_request(),
        "/tmp",
    );
    self_render.update_result(image_result(), false, false);
    assert!(!self_render.render(80).is_empty());

    set_capabilities(TerminalCapabilities {
        images: Some(ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: false,
    });
    let kitty = read_component();
    kitty.update_result(image_result(), false, false);
    let _ = kitty.render(80);
    reset_capabilities_cache();
}

#[test]
fn diff_renderer_covers_every_row_branch() {
    init_theme(Some("dark"), false);
    let edit = ToolExecutionComponent::new(
        "edit",
        "diff-branches",
        json!({ "path": "a.rs", "oldText": "a", "newText": "b" }),
        ToolExecutionOptions::default(),
        Some(pi_coding_agent::tools::edit::create_edit_tool_definition(
            "/tmp", None,
        )),
        render_request(),
        "/tmp",
    );
    edit.update_result(
        text_result(
            "",
            json!({ "diff": " 1 context\n-2 old one\n-3 old two\n+2 new one\n+3 new two\n+4 added\n? stray" }),
        ),
        false,
        false,
    );
    let rendered = strip_ansi(&edit.render(120).join("\n"));
    assert!(rendered.contains("context"));
    assert!(rendered.contains("added"));
}

#[test]
fn assistant_notices_cover_abort_and_error_wording() {
    init_theme(Some("dark"), false);
    let mut aborted = assistant_message(Vec::new(), StopReason::Aborted);
    aborted.error_message = Some("Request was aborted".to_string());
    let aborted_component =
        AssistantMessageComponent::new(Some(aborted.clone()), AssistantMessageOptions::default());
    assert!(strip_ansi(&aborted_component.render(80).join("\n")).contains("Operation aborted"));

    aborted.error_message = Some("custom abort".to_string());
    let custom_component =
        AssistantMessageComponent::new(Some(aborted), AssistantMessageOptions::default());
    assert!(strip_ansi(&custom_component.render(80).join("\n")).contains("custom abort"));

    let mut failed = assistant_message(Vec::new(), StopReason::Error);
    failed.error_message = Some("boom".to_string());
    let failed_component =
        AssistantMessageComponent::new(Some(failed), AssistantMessageOptions::default());
    assert!(strip_ansi(&failed_component.render(80).join("\n")).contains("Error: boom"));
}

#[test]
fn find_result_covers_the_bytes_truncation_warning() {
    init_theme(Some("dark"), false);
    let find = ToolExecutionComponent::new(
        "find",
        "find-bytes",
        json!({ "pattern": "x" }),
        ToolExecutionOptions::default(),
        Some(pi_coding_agent::tools::find::create_find_tool_definition(
            "/tmp", None,
        )),
        render_request(),
        "/tmp",
    );
    find.update_result(
        text_result(
            "a",
            json!({ "truncation": {
                "content": "a", "truncated": true, "truncatedBy": "bytes",
                "totalLines": 1, "totalBytes": 80, "outputLines": 1, "outputBytes": 20,
                "lastLinePartial": false, "firstLineExceedsLimit": false,
                "maxLines": 10, "maxBytes": 20
            }}),
        ),
        false,
        false,
    );
    assert!(strip_ansi(&find.render(80).join("\n")).contains("20B limit"));
}

fn base_definition(name: &str) -> pi_coding_agent::extensions::types::ToolDefinition {
    pi_coding_agent::extensions::types::ToolDefinition {
        name: name.to_string(),
        label: name.to_string(),
        description: "custom".to_string(),
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
             -> pi_ai::types::BoxedFuture<
                '_,
                Result<AgentToolResult, pi_agent_core::types::AgentToolError>,
            > {
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

#[test]
#[expect(
    clippy::panic,
    reason = "the cases pin the renderer-throw fallback contract"
)]
fn invalid_arguments_and_throwing_renderers_fall_back() {
    init_theme(Some("dark"), false);
    let read = ToolExecutionComponent::new(
        "read",
        "invalid-read",
        json!({ "path": 1 }),
        ToolExecutionOptions::default(),
        Some(pi_coding_agent::tools::read::create_read_tool_definition(
            "/tmp", None,
        )),
        render_request(),
        "/tmp",
    );
    assert!(strip_ansi(&read.render(80).join("\n")).contains("[invalid arg]"));

    let write = ToolExecutionComponent::new(
        "write",
        "invalid-write",
        json!({ "path": "a", "content": 1 }),
        ToolExecutionOptions::default(),
        Some(pi_coding_agent::tools::write::create_write_tool_definition(
            "/tmp", None,
        )),
        render_request(),
        "/tmp",
    );
    assert!(strip_ansi(&write.render(80).join("\n")).contains("invalid content arg"));

    let find = ToolExecutionComponent::new(
        "find",
        "invalid-find",
        json!({ "pattern": "x", "path": 1 }),
        ToolExecutionOptions::default(),
        Some(pi_coding_agent::tools::find::create_find_tool_definition(
            "/tmp", None,
        )),
        render_request(),
        "/tmp",
    );
    assert!(strip_ansi(&find.render(80).join("\n")).contains("[invalid arg]"));

    let ls = ToolExecutionComponent::new(
        "ls",
        "invalid-ls",
        json!({ "path": 1 }),
        ToolExecutionOptions::default(),
        Some(pi_coding_agent::tools::ls::create_ls_tool_definition(
            "/tmp", None,
        )),
        render_request(),
        "/tmp",
    );
    assert!(strip_ansi(&ls.render(80).join("\n")).contains("[invalid arg]"));

    let throwing = pi_coding_agent::extensions::types::ToolDefinition {
        render_call: Some(Arc::new(|_args, _theme, _context| -> Rc<dyn Component> {
            panic!("broken call renderer")
        })),
        render_result: Some(Arc::new(
            |_result, _options, _theme, _context| -> Rc<dyn Component> {
                panic!("broken result renderer")
            },
        )),
        ..base_definition("throwing")
    };
    let tool = ToolExecutionComponent::new(
        "throwing",
        "throwing",
        json!({}),
        ToolExecutionOptions::default(),
        Some(throwing),
        render_request(),
        "/tmp",
    );
    assert!(strip_ansi(&tool.render(80).join("\n")).contains("throwing"));
    tool.update_result(text_result("ok", json!({})), false, false);
    assert!(strip_ansi(&tool.render(80).join("\n")).contains("ok"));
}

#[test]
fn custom_renderers_declining_or_throwing_fall_back_to_the_default_box() {
    #[expect(
        clippy::panic,
        reason = "the case pins the renderer-throw fallback contract"
    )]
    fn throwing(
        _message: &CustomAgentMessage,
        _options: &MessageRenderOptions,
        _theme: &pi_coding_agent::modes::interactive::theme::theme::Theme,
    ) -> Option<Rc<dyn Component>> {
        panic!("broken custom renderer")
    }

    init_theme(Some("dark"), false);
    let message = CustomAgentMessage {
        role: "custom".to_string(),
        timestamp: 0,
        data: serde_json::Map::from_iter([
            ("customType".to_string(), json!("t")),
            ("content".to_string(), json!("body")),
        ]),
    };
    let declining: pi_coding_agent::extensions::types::MessageRenderer =
        Arc::new(|_message, _options, _theme| None);
    let declined = CustomMessageComponent::new(message.clone(), Some(declining), None, 1);
    assert!(strip_ansi(&declined.render(80).join("\n")).contains("[t]"));

    let thrown = CustomMessageComponent::new(message, Some(Arc::new(throwing)), None, 1);
    assert!(strip_ansi(&thrown.render(80).join("\n")).contains("[t]"));
}

#[test]
fn bash_preview_invalidate_and_hint_formatting_are_exercised() {
    init_theme(Some("dark"), false);
    let shell = BashExecutionComponent::new("echo hi", render_request(), false);
    shell.append_output("one\ntwo\n");
    shell.set_complete(Some(0), false, None, None);
    assert!(!shell.render(40).is_empty());
    shell.invalidate();

    assert!(
        pi_coding_agent::modes::interactive::components::keybinding_hints::format_key_text(
            "alt+enter",
            true
        )
        .contains("Enter")
    );
    let _ = ToolRenderers::default();
    let _ = pi_coding_agent::tools::renderers::bash::BashResultRenderComponent::default();
    let truncation = truncate_tail("x", TruncationOptions::default());
    assert!(!truncation.truncated);
}
