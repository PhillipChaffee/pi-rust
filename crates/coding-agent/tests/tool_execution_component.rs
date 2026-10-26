//! The `tool-execution-component.test.ts` suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::future::Future;
use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::types::{AgentToolError, AgentToolResult};
use pi_ai::types::{BoxedFuture, ToolResultBlock};
use pi_coding_agent::config::get_readme_path;
use pi_coding_agent::extensions::types::{RenderShell, ToolDefinition};
use pi_coding_agent::modes::interactive::components::tool_execution::{
    ToolExecutionComponent, ToolExecutionOptions,
};
use pi_coding_agent::modes::interactive::theme::theme::{init_theme, theme};
use pi_coding_agent::tools::bash::{
    BashExecOptions, BashExecOutcome, BashOperations, BashToolOptions, create_bash_tool_definition,
};
use pi_coding_agent::tools::read::create_read_tool_definition;
use pi_coding_agent::tools::renderers::with_built_in_renderers;
use pi_coding_agent::tools::write::create_write_tool_definition;
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::components::Text;
use pi_tui::tui::{Component, RenderRequest, TuiMouseButton, TuiMouseEvent, TuiMouseEventType};
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
                        content: vec![ToolResultBlock::Text(pi_ai::types::TextContent {
                            text: "ok".to_string(),
                            text_signature: None,
                        })],
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

fn result_with_text(text: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultBlock::Text(pi_ai::types::TextContent {
            text: text.to_string(),
            text_signature: None,
        })],
        details: json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn cwd() -> String {
    std::env::current_dir()
        .expect("the test has a working directory")
        .to_string_lossy()
        .into_owned()
}

fn component(
    name: &str,
    id: &str,
    args: Value,
    definition: Option<ToolDefinition>,
) -> ToolExecutionComponent {
    ToolExecutionComponent::new(
        name,
        id,
        args,
        ToolExecutionOptions::default(),
        definition,
        render_request(),
        &cwd(),
    )
}

#[test]
fn stacks_custom_call_and_result_renderers_like_the_old_implementation() {
    init_theme(Some("dark"), false);
    let definition = ToolDefinition {
        render_call: Some(Arc::new(|_args, _theme, _context| {
            Rc::new(Text::with_padding("custom call", 0, 0))
        })),
        render_result: Some(Arc::new(|_result, _options, _theme, _context| {
            Rc::new(Text::with_padding("custom result", 0, 0))
        })),
        ..base_definition("custom_tool")
    };
    let component = component("custom_tool", "tool-1", json!({}), Some(definition));
    assert!(strip_ansi(&component.render(120).join("\n")).contains("custom call"));

    component.update_result(result_with_text("done"), false, false);

    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert!(rendered.contains("custom call"));
    assert!(rendered.contains("custom result"));
}

#[test]
fn self_rendered_empty_tool_rows_take_no_layout_space() {
    init_theme(Some("dark"), false);
    let definition = ToolDefinition {
        render_shell: Some(RenderShell::SelfRender),
        render_call: Some(Arc::new(|_args, _theme, _context| {
            Rc::new(Text::with_padding("", 0, 0))
        })),
        render_result: Some(Arc::new(|_result, _options, _theme, _context| {
            Rc::new(Text::with_padding("", 0, 0))
        })),
        ..base_definition("custom_tool")
    };
    let component = component(
        "custom_tool",
        "tool-empty-self-render",
        json!({}),
        Some(definition),
    );
    assert_eq!(component.render(120), Vec::<String>::new());

    component.update_result(
        AgentToolResult {
            content: Vec::new(),
            details: json!({}),
            usage: None,
            added_tool_names: None,
            terminate: None,
        },
        false,
        false,
    );

    assert_eq!(component.render(120), Vec::<String>::new());
}

#[test]
fn uses_built_in_rendering_for_built_in_overrides_without_custom_renderers() {
    init_theme(Some("dark"), false);
    let definition = with_built_in_renderers("edit", Some(base_definition("edit")))
        .expect("edit has built-in renderers");
    let component = component(
        "edit",
        "tool-2",
        json!({ "path": "README.md", "oldText": "before", "newText": "after" }),
        Some(definition),
    );
    component.update_result(
        AgentToolResult {
            content: Vec::new(),
            details: json!({ "diff": "+1 after", "firstChangedLine": 1 }),
            usage: None,
            added_tool_names: None,
            terminate: None,
        },
        false,
        false,
    );
    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert!(rendered.contains("edit"));
    assert!(rendered.contains("README.md"));
    assert!(!rendered.contains(":1"));
}

#[test]
fn preserves_legacy_file_path_rendering_compatibility_for_built_in_tools() {
    init_theme(Some("dark"), false);
    let component = component("read", "tool-3", json!({ "file_path": "README.md" }), None);
    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert!(rendered.contains("read"));
    assert!(rendered.contains("README.md"));
}

fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(future)
}

fn scripted_operations(
    body: impl Fn(&Arc<dyn Fn(&[u8]) + Send + Sync>) -> Result<BashExecOutcome, AgentToolError>
    + Send
    + Sync
    + 'static,
) -> BashOperations {
    let body = Arc::new(body);
    BashOperations {
        exec: Arc::new(
            move |_command: &str, _cwd: &str, options: BashExecOptions| {
                let body = Arc::clone(&body);
                Box::pin(async move { body(&options.on_data) })
            },
        ),
    }
}

#[test]
fn bash_execute_emits_an_initial_empty_partial_update_before_output_arrives() {
    let definition = create_bash_tool_definition(
        &cwd(),
        Some(BashToolOptions {
            operations: Some(scripted_operations(|_on_data| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                Ok(BashExecOutcome { exit_code: Some(0) })
            })),
            expose_session_environment: Some(false),
            ..BashToolOptions::default()
        }),
    );
    let updates = Arc::new(std::sync::Mutex::new(Vec::<AgentToolResult>::new()));
    let seen = Arc::clone(&updates);
    let on_update = move |result: &AgentToolResult,
                          _options: Option<
        pi_agent_core::harness::types::AgentHarnessToolUpdateOptions,
    >| {
        seen.lock().expect("lock").push(result.clone());
    };
    block_on((definition.execute)(
        "tool-bash-1",
        &json!({ "command": "sleep 10" }),
        None,
        Some(&on_update),
        None,
    ))
    .expect("the scripted command settles");

    let updates = updates.lock().expect("lock");
    assert_eq!(updates.len(), 1);
    assert!(updates[0].content.is_empty());
    assert_eq!(updates[0].details, Value::Null);
}

#[test]
fn bash_renderer_does_not_duplicate_final_full_output_truncation_details() {
    init_theme(Some("dark"), false);
    let definition = create_bash_tool_definition(
        &cwd(),
        Some(BashToolOptions {
            operations: Some(scripted_operations(|on_data| {
                for index in 1..=4000 {
                    on_data(format!("line-{index:04}\n").as_bytes());
                }
                Ok(BashExecOutcome { exit_code: Some(0) })
            })),
            expose_session_environment: Some(false),
            ..BashToolOptions::default()
        }),
    );
    let result = block_on((definition.execute)(
        "tool-bash-1b",
        &json!({ "command": "generate output" }),
        None,
        None,
        None,
    ))
    .expect("the scripted command settles");
    let component = component(
        "bash",
        "tool-bash-1b",
        json!({ "command": "generate output" }),
        Some(definition),
    );
    component.set_expanded(true);
    component.update_result(result, false, false);

    let rendered = strip_ansi(&component.render(200).join("\n"));
    assert_eq!(rendered.matches("Full output:").count(), 1);
    assert!(rendered.contains("Truncated: showing 2000 of 4000 lines"));
    assert!(!rendered.contains("[Showing lines 2001-4000 of 4000. Full output:"));
}

#[test]
fn does_not_duplicate_built_in_headers_when_passed_the_active_built_in_definition() {
    init_theme(Some("dark"), false);
    let component = component(
        "read",
        "tool-4",
        json!({ "path": "README.md" }),
        Some(create_read_tool_definition(&cwd(), None)),
    );
    component.update_result(result_with_text("hello"), false, false);
    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert_eq!(rendered.matches("read").count(), 1);
}

#[test]
fn inherits_missing_built_in_result_renderer_slot_from_the_built_in_tool() {
    init_theme(Some("dark"), false);
    let definition = ToolDefinition {
        render_call: Some(Arc::new(|_args, _theme, _context| {
            Rc::new(Text::with_padding("override call", 0, 0))
        })),
        ..base_definition("read")
    };
    let definition = with_built_in_renderers("read", Some(definition)).expect("merged");
    let component = component(
        "read",
        "tool-4b",
        json!({ "path": "notes.txt" }),
        Some(definition),
    );
    component.update_result(result_with_text("hello"), false, false);
    component.set_expanded(true);
    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert!(rendered.contains("override call"));
    assert!(rendered.contains("hello"));
}

#[test]
fn inherits_missing_built_in_call_renderer_slot_from_the_built_in_tool() {
    init_theme(Some("dark"), false);
    let definition = ToolDefinition {
        render_result: Some(Arc::new(|_result, _options, _theme, _context| {
            Rc::new(Text::with_padding("override result", 0, 0))
        })),
        ..base_definition("read")
    };
    let definition = with_built_in_renderers("read", Some(definition)).expect("merged");
    let component = component(
        "read",
        "tool-4c",
        json!({ "path": "README.md" }),
        Some(definition),
    );
    component.update_result(result_with_text("hello"), false, false);
    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert!(rendered.contains("read"));
    assert!(rendered.contains("README.md"));
    assert!(rendered.contains("override result"));
}

#[test]
fn uses_custom_renderers_for_built_in_overrides_that_reuse_built_in_definition_parameters() {
    init_theme(Some("dark"), false);
    let definition = ToolDefinition {
        render_call: Some(Arc::new(|_args, _theme, _context| {
            Rc::new(Text::with_padding("override call", 0, 0))
        })),
        render_result: Some(Arc::new(|_result, _options, _theme, _context| {
            Rc::new(Text::with_padding("override result", 0, 0))
        })),
        ..create_read_tool_definition(&cwd(), None)
    };
    let component = component(
        "read",
        "tool-4d",
        json!({ "path": "README.md" }),
        Some(definition),
    );
    component.update_result(result_with_text("hello"), false, false);
    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert!(rendered.contains("override call"));
    assert!(rendered.contains("override result"));
    assert!(!rendered.contains("read README.md"));
}

#[derive(Default)]
struct SharedRenderState {
    token: Option<String>,
}

#[test]
fn shares_renderer_state_across_custom_call_and_result_slots() {
    init_theme(Some("dark"), false);
    let definition = ToolDefinition {
        render_call: Some(Arc::new(|_args, _theme, context| {
            let body = context
                .state
                .with_state_mut(SharedRenderState::default, |state| {
                    state
                        .token
                        .get_or_insert_with(|| "shared-token".to_string())
                        .clone()
                })
                .unwrap_or_default();
            Rc::new(Text::with_padding(format!("custom call {body}"), 0, 0))
        })),
        render_result: Some(Arc::new(|_result, _options, _theme, context| {
            let body = context
                .state
                .with_state_mut(SharedRenderState::default, |state| {
                    state.token.clone().unwrap_or_default()
                })
                .unwrap_or_default();
            Rc::new(Text::with_padding(format!("custom result {body}"), 0, 0))
        })),
        ..base_definition("custom_tool")
    };
    let component = component("custom_tool", "tool-5", json!({}), Some(definition));
    component.update_result(result_with_text("done"), false, false);
    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert!(rendered.contains("custom call shared-token"));
    assert!(rendered.contains("custom result shared-token"));
}

#[test]
fn exposes_args_in_render_result_context() {
    init_theme(Some("dark"), false);
    let definition = ToolDefinition {
        render_call: Some(Arc::new(|_args, _theme, _context| {
            Rc::new(Text::with_padding("call", 0, 0))
        })),
        render_result: Some(Arc::new(|_result, _options, _theme, context| {
            Rc::new(Text::with_padding(
                format!("arg:{}", context.args["foo"].as_str().unwrap_or_default()),
                0,
                0,
            ))
        })),
        ..base_definition("custom_tool")
    };
    let component = component(
        "custom_tool",
        "tool-5b",
        json!({ "foo": "bar" }),
        Some(definition),
    );
    component.update_result(result_with_text("done"), false, false);
    assert!(strip_ansi(&component.render(120).join("\n")).contains("arg:bar"));
}

#[test]
fn collapses_fallback_results_until_expanded() {
    init_theme(Some("dark"), false);
    let component = component(
        "custom_tool",
        "tool-6",
        json!({ "foo": "bar" }),
        Some(base_definition("custom_tool")),
    );
    let output = (1..=15)
        .map(|index| format!("line-{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    component.update_result(result_with_text(&output), false, false);

    let collapsed = strip_ansi(&component.render(120).join("\n"));
    assert!(collapsed.contains("custom_tool"));
    assert!(collapsed.contains("line-10"));
    assert!(!collapsed.contains("line-11"));
    assert!(collapsed.contains("5 more lines"));
    assert!(collapsed.contains("to expand"));

    component.set_expanded(true);
    let expanded = strip_ansi(&component.render(120).join("\n"));
    assert!(expanded.contains("line-15"));
    assert!(!expanded.contains("more lines"));
}

#[test]
fn trims_trailing_blank_display_lines_from_write_previews() {
    init_theme(Some("dark"), false);
    let component = component(
        "write",
        "tool-7",
        json!({ "path": "README.md", "content": "one\ntwo\n" }),
        Some(create_write_tool_definition(&cwd(), None)),
    );
    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert!(rendered.contains("one"));
    assert!(rendered.contains("two"));
    assert!(!rendered.contains("two\n\n"));
}

#[test]
fn trims_trailing_blank_display_lines_from_read_results() {
    init_theme(Some("dark"), false);
    let component = component(
        "read",
        "tool-8",
        json!({ "path": "notes.txt" }),
        Some(create_read_tool_definition(&cwd(), None)),
    );
    component.update_result(result_with_text("one\ntwo\n"), false, false);
    component.set_expanded(true);
    let rendered = strip_ansi(&component.render(120).join("\n"));
    assert!(rendered.contains("one"));
    assert!(rendered.contains("two"));
    assert!(!rendered.contains("two\n\n"));
}

#[test]
fn does_not_syntax_highlight_read_errors_based_on_the_requested_file_path() {
    init_theme(Some("dark"), false);
    let component = component(
        "read",
        "tool-read-error-highlighting",
        json!({ "path": "config.exs", "offset": 120, "limit": 130 }),
        Some(create_read_tool_definition(&cwd(), None)),
    );
    let error = "Offset 120 is beyond end of file (96 lines total)";
    component.update_result(result_with_text(error), true, false);

    let rendered = component.render(120).join("\n");
    assert!(strip_ansi(&rendered).contains(error));
    assert!(rendered.contains(&theme().fg("toolOutput", error)));
}

fn click_row(component: &ToolExecutionComponent, width: usize, needle: &str) -> TuiMouseEvent {
    let lines = component.render(width);
    let row = lines
        .iter()
        .position(|line| strip_ansi(line).contains(needle))
        .expect("the requested row renders");
    TuiMouseEvent {
        event_type: TuiMouseEventType::Click,
        button: TuiMouseButton::Left,
        x: 2,
        y: u16::try_from(row).expect("row fits"),
        screen_x: 2,
        screen_y: u16::try_from(row).expect("row fits"),
        width: u16::try_from(width).expect("width fits"),
        height: u16::try_from(lines.len()).expect("height fits"),
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: Some(1),
    }
}

#[test]
fn expands_a_collapsed_tool_result_when_clicked() {
    init_theme(Some("dark"), false);
    let component = component(
        "read",
        "tool-click-expand",
        json!({ "path": "notes.txt" }),
        Some(create_read_tool_definition(&cwd(), None)),
    );
    component.update_result(result_with_text("hidden content"), false, false);
    let width = 120;
    let event = click_row(&component, width, "notes.txt");
    assert!(
        component
            .handle_mouse(&event)
            .is_some_and(|result| result.handled)
    );
    assert!(strip_ansi(&component.render(width).join("\n")).contains("hidden content"));
}

#[test]
fn collapses_ordinary_read_results_until_expanded() {
    init_theme(Some("dark"), false);
    let component = component(
        "read",
        "tool-ordinary-read-collapsed",
        json!({ "path": "notes.txt" }),
        Some(create_read_tool_definition(&cwd(), None)),
    );
    component.update_result(result_with_text("hidden content"), false, false);

    let collapsed = strip_ansi(&component.render(120).join("\n"));
    assert!(collapsed.contains("read"));
    assert!(collapsed.contains("notes.txt"));
    assert!(!collapsed.contains("hidden content"));

    component.set_expanded(true);
    assert!(strip_ansi(&component.render(120).join("\n")).contains("hidden content"));
}

fn compact_scenarios() -> Vec<(String, String, String, String, Option<String>)> {
    let cwd = cwd();
    let parent = std::path::Path::new(&cwd)
        .parent()
        .expect("the crate directory has a parent")
        .join("AGENTS.md");
    vec![
        (
            "SKILL.md".to_string(),
            std::path::Path::new(&cwd)
                .join("attio")
                .join("SKILL.md")
                .to_string_lossy()
                .into_owned(),
            "---\nname: attio\ndescription: CRM helper\n---\n\n# Hidden skill instructions"
                .to_string(),
            "[skill] attio".to_string(),
            Some("read skill attio".to_string()),
        ),
        (
            "AGENTS.md".to_string(),
            std::path::Path::new(&cwd)
                .join(".pi")
                .join("AGENTS.md")
                .to_string_lossy()
                .into_owned(),
            "Hidden resource instructions".to_string(),
            "read resource .pi/AGENTS.md".to_string(),
            None,
        ),
        (
            "AGENTS.override.md".to_string(),
            std::path::Path::new(&cwd)
                .join(".pi")
                .join("AGENTS.override.md")
                .to_string_lossy()
                .into_owned(),
            "Hidden override instructions".to_string(),
            "read resource .pi/AGENTS.override.md".to_string(),
            None,
        ),
        (
            "outside AGENTS.md".to_string(),
            parent.to_string_lossy().replace('\\', "/"),
            "Hidden outside resource instructions".to_string(),
            format!(
                "read resource {}",
                parent.to_string_lossy().replace('\\', "/")
            ),
            None,
        ),
        (
            "Pi documentation".to_string(),
            get_readme_path(),
            "Hidden docs content".to_string(),
            "read docs README.md".to_string(),
            None,
        ),
    ]
}

#[test]
fn renders_compact_read_results_until_expanded() {
    init_theme(Some("dark"), false);
    for (title, path, content, compact, absent) in compact_scenarios() {
        let component = component(
            "read",
            &format!("tool-compact-{title}"),
            json!({ "path": path }),
            Some(create_read_tool_definition(&cwd(), None)),
        );
        component.update_result(result_with_text(&content), false, false);

        let collapsed = strip_ansi(&component.render(120).join("\n"));
        assert!(collapsed.contains(compact.as_str()), "{title}: {collapsed}");
        assert!(!collapsed.contains("Hidden"));
        if let Some(absent) = absent {
            assert!(!collapsed.contains(absent.as_str()), "{title}: {collapsed}");
        }

        component.set_expanded(true);
        assert!(
            strip_ansi(&component.render(120).join("\n")).contains("Hidden"),
            "{title}"
        );
    }
}

#[test]
fn shows_the_read_line_range_in_compact_reads_before_the_expand_hint() {
    init_theme(Some("dark"), false);
    for (title, path, compact) in [
        (
            "SKILL.md".to_string(),
            std::path::Path::new(&cwd())
                .join("attio")
                .join("SKILL.md")
                .to_string_lossy()
                .into_owned(),
            "[skill] attio:120-329".to_string(),
        ),
        (
            "Pi documentation".to_string(),
            get_readme_path(),
            "read docs README.md:120-329".to_string(),
        ),
    ] {
        let component = component(
            "read",
            &format!("tool-compact-range-{title}"),
            json!({ "path": path, "offset": 120, "limit": 210 }),
            Some(create_read_tool_definition(&cwd(), None)),
        );

        let collapsed = strip_ansi(&component.render(120).join("\n"));
        assert!(collapsed.contains(compact.as_str()), "{title}: {collapsed}");
        assert!(
            collapsed.find(":120-329").expect("range") < collapsed.find("to expand").expect("hint"),
            "{title}"
        );
    }
}
