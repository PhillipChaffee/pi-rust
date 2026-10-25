//! The `assistant-message.test.ts` suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1 plus the user
//! message padding case the upstream file carries.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, KnownApi, ProviderId, StopReason, TextContent,
    ThinkingContent, Usage, UsageCost,
};
use pi_coding_agent::extensions::types::{MarkdownTransformContext, MarkdownTransformer};
use pi_coding_agent::modes::interactive::components::assistant_message::{
    AssistantMessageComponent, AssistantMessageOptions,
};
use pi_coding_agent::modes::interactive::components::user_message::UserMessageComponent;
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::{Component, TuiMouseButton, TuiMouseEvent, TuiMouseEventType};

const OSC133_ZONE_START: &str = "\u{1b}]133;A\u{7}";
const OSC133_ZONE_END: &str = "\u{1b}]133;B\u{7}";
const OSC133_ZONE_FINAL: &str = "\u{1b}]133;C\u{7}";

const fn usage() -> Usage {
    Usage {
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
    }
}

fn text(value: &str) -> AssistantBlock {
    AssistantBlock::Text(TextContent {
        text: value.to_string(),
        text_signature: None,
    })
}

fn thinking(value: &str) -> AssistantBlock {
    AssistantBlock::Thinking(ThinkingContent {
        thinking: value.to_string(),
        thinking_signature: None,
        redacted: None,
    })
}

fn message(content: Vec<AssistantBlock>, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content,
        api: Api::from(KnownApi::OpenaiResponses),
        provider: ProviderId("openai".to_string()),
        model: "gpt-4o-mini".to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage(),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

macro_rules! transformers {
    ($($transformer:expr),* $(,)?) => {{
        let list: Vec<MarkdownTransformer> = vec![$(Arc::new($transformer)),*];
        Arc::new(list)
    }};
}

const fn click(y: u16, height: u16, width: u16) -> TuiMouseEvent {
    TuiMouseEvent {
        event_type: TuiMouseEventType::Click,
        button: TuiMouseButton::Left,
        x: 1,
        y,
        screen_x: 1,
        screen_y: y,
        width,
        height,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: Some(1),
    }
}

#[test]
fn adds_osc_133_zone_markers_to_assistant_messages_without_tool_calls() {
    init_theme(Some("dark"), false);
    let component = AssistantMessageComponent::new(
        Some(message(vec![text("hello")], StopReason::Stop)),
        AssistantMessageOptions::default(),
    );
    let lines = component.render(40);
    assert!(!lines.is_empty());
    assert!(lines[0].contains(OSC133_ZONE_START));
    assert!(lines[lines.len() - 1].starts_with(&format!("{OSC133_ZONE_END}{OSC133_ZONE_FINAL}")));
}

#[test]
fn does_not_add_osc_133_zone_markers_when_assistant_message_contains_tool_calls() {
    init_theme(Some("dark"), false);
    let component = AssistantMessageComponent::new(
        Some(message(
            vec![
                text("calling tool"),
                AssistantBlock::ToolCall(pi_ai::types::ToolCall {
                    id: "tool-1".to_string(),
                    name: "read".to_string(),
                    arguments: serde_json::Map::from_iter([(
                        "path".to_string(),
                        serde_json::json!("file.txt"),
                    )]),
                    thought_signature: None,
                    namespace: None,
                }),
            ],
            StopReason::ToolUse,
        )),
        AssistantMessageOptions::default(),
    );
    let rendered = component.render(60).join("\n");
    assert!(!rendered.contains(OSC133_ZONE_START));
    assert!(!rendered.contains(OSC133_ZONE_END));
    assert!(!rendered.contains(OSC133_ZONE_FINAL));
}

#[test]
fn renders_length_stops_with_neutral_truncation_wording() {
    init_theme(Some("dark"), false);
    let component = AssistantMessageComponent::new(
        Some(message(
            vec![thinking("private reasoning")],
            StopReason::Length,
        )),
        AssistantMessageOptions {
            hide_thinking_block: true,
            ..AssistantMessageOptions::default()
        },
    );
    let rendered = component.render(80).join("\n");
    assert!(rendered.contains("Thinking..."));
    assert!(rendered.contains("Response was truncated before completion."));
}

#[test]
fn coalesces_adjacent_thinking_blocks_into_one_hidden_thinking_label() {
    init_theme(Some("dark"), false);
    let component = AssistantMessageComponent::new(
        Some(message(
            vec![
                thinking("first thought"),
                thinking(""),
                thinking("second thought"),
                text("answer"),
            ],
            StopReason::Stop,
        )),
        AssistantMessageOptions {
            hide_thinking_block: true,
            ..AssistantMessageOptions::default()
        },
    );
    let rendered = strip_ansi(&component.render(80).join("\n"));
    assert_eq!(rendered.matches("Thinking...").count(), 1);
    assert!(rendered.contains("answer"));
}

#[test]
fn collapses_individual_thinking_runs_when_clicked() {
    init_theme(Some("dark"), false);
    let component = AssistantMessageComponent::new(
        Some(message(
            vec![
                thinking("first reasoning"),
                text("answer"),
                thinking("second reasoning"),
            ],
            StopReason::Stop,
        )),
        AssistantMessageOptions::default(),
    );
    let width = 80;
    let lines = component.render(width);
    let first_row = lines
        .iter()
        .position(|line| strip_ansi(line).contains("first reasoning"))
        .expect("the first thinking run renders");
    let event = click(
        u16::try_from(first_row).expect("row fits"),
        u16::try_from(lines.len()).expect("height fits"),
        u16::try_from(width).expect("width fits"),
    );
    assert!(
        component
            .handle_mouse(&event)
            .is_some_and(|result| result.handled)
    );

    let collapsed = strip_ansi(&component.render(width).join("\n"));
    assert!(!collapsed.contains("first reasoning"));
    assert!(collapsed.contains("Thinking..."));
    assert!(collapsed.contains("second reasoning"));
}

#[test]
fn uses_configured_output_padding_for_text_and_thinking() {
    init_theme(Some("dark"), false);
    let component = AssistantMessageComponent::new(
        Some(message(
            vec![text("hello"), thinking("reasoning")],
            StopReason::Stop,
        )),
        AssistantMessageOptions::default(),
    );
    let lines = component
        .render(80)
        .into_iter()
        .map(|line| strip_ansi(&line))
        .collect::<Vec<_>>();
    assert!(lines.iter().any(|line| line.contains(" hello")));
    assert!(lines.iter().any(|line| line.contains(" reasoning")));

    component.set_output_pad(0);
    let updated = component
        .render(80)
        .into_iter()
        .map(|line| strip_ansi(&line))
        .collect::<Vec<_>>();
    assert!(updated.iter().any(|line| line.starts_with("hello")));
    assert!(updated.iter().any(|line| line.starts_with("reasoning")));
}

#[test]
fn chains_markdown_transformers_in_registration_order() {
    init_theme(Some("dark"), false);
    let calls = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen = Arc::clone(&calls);
    let component = AssistantMessageComponent::new(
        Some(message(
            vec![text("The result is $x^2$.")],
            StopReason::Stop,
        )),
        AssistantMessageOptions {
            markdown_transformers: transformers![
                move |markdown: &str, context: &MarkdownTransformContext| {
                    seen.lock().expect("lock").push("formula".to_string());
                    assert_eq!(context.message_type.as_str(), "assistant");
                    assert!(!context.is_streaming);
                    assert_eq!(context.available_width, 78);
                    markdown.replace("$x^2$", "x²")
                },
                |markdown: &str, _context: &MarkdownTransformContext| format!("{markdown} Done."),
            ],
            ..AssistantMessageOptions::default()
        },
    );
    let rendered = strip_ansi(&component.render(80).join("\n"));
    assert!(rendered.contains("The result is x². Done."));
    assert_eq!(*calls.lock().expect("lock"), vec!["formula".to_string()]);
}

#[test]
fn identifies_partial_assistant_markdown_as_streaming() {
    init_theme(Some("dark"), false);
    let streaming_states = Arc::new(std::sync::Mutex::new(Vec::<bool>::new()));
    let seen = Arc::clone(&streaming_states);
    let component = AssistantMessageComponent::new(
        None,
        AssistantMessageOptions {
            markdown_transformers: transformers![
                move |markdown: &str, context: &MarkdownTransformContext| {
                    seen.lock().expect("lock").push(context.is_streaming);
                    if context.is_streaming {
                        markdown.to_string()
                    } else {
                        format!("{markdown} transformed")
                    }
                },
            ],
            ..AssistantMessageOptions::default()
        },
    );
    let partial = message(vec![text("partial")], StopReason::Stop);
    component.update_content(&partial, true);
    assert!(!strip_ansi(&component.render(80).join("\n")).contains("transformed"));

    component.update_content(&partial, false);
    assert!(strip_ansi(&component.render(80).join("\n")).contains("partial transformed"));
    assert_eq!(*streaming_states.lock().expect("lock"), vec![true, false]);
}

#[test]
fn reapplies_markdown_transformers_when_available_width_changes() {
    init_theme(Some("dark"), false);
    let widths = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
    let seen = Arc::clone(&widths);
    let component = AssistantMessageComponent::new(
        Some(message(vec![text("answer")], StopReason::Stop)),
        AssistantMessageOptions {
            markdown_transformers: transformers![
                move |markdown: &str, context: &MarkdownTransformContext| {
                    seen.lock().expect("lock").push(context.available_width);
                    format!("{markdown} ({})", context.available_width)
                },
            ],
            ..AssistantMessageOptions::default()
        },
    );
    assert!(strip_ansi(&component.render(80).join("\n")).contains("answer (78)"));
    let _ = component.render(80);
    assert!(strip_ansi(&component.render(60).join("\n")).contains("answer (58)"));
    assert_eq!(*widths.lock().expect("lock"), vec![78, 58]);
}

#[test]
#[expect(
    clippy::panic,
    reason = "the case pins the chain's throw-isolation contract"
)]
fn continues_the_markdown_transformer_chain_when_a_transformer_throws() {
    init_theme(Some("dark"), false);
    let calls = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let first = Arc::clone(&calls);
    let throw = Arc::clone(&calls);
    let last = Arc::clone(&calls);
    let component = AssistantMessageComponent::new(
        Some(message(vec![text("still visible")], StopReason::Stop)),
        AssistantMessageOptions {
            markdown_transformers: transformers![
                move |markdown: &str, _context: &MarkdownTransformContext| {
                    first.lock().expect("lock").push("first".to_string());
                    markdown.replace("still", "remains")
                },
                move |_markdown: &str, _context: &MarkdownTransformContext| -> String {
                    throw.lock().expect("lock").push("throw".to_string());
                    panic!("broken transformer");
                },
                move |markdown: &str, _context: &MarkdownTransformContext| {
                    last.lock().expect("lock").push("last".to_string());
                    format!("{markdown} after error")
                },
            ],
            ..AssistantMessageOptions::default()
        },
    );
    let rendered = strip_ansi(&component.render(80).join("\n"));
    assert!(rendered.contains("remains visible after error"));
    assert_eq!(
        *calls.lock().expect("lock"),
        vec!["first".to_string(), "throw".to_string(), "last".to_string()]
    );
}

#[test]
fn transforms_text_and_thinking_markdown_without_mutating_the_original_message() {
    init_theme(Some("dark"), false);
    let original = message(
        vec![text("answer"), thinking("reasoning")],
        StopReason::Stop,
    );
    let snapshot = original.clone();
    let component = AssistantMessageComponent::new(
        Some(original.clone()),
        AssistantMessageOptions {
            markdown_transformers: transformers![
                |markdown: &str, context: &MarkdownTransformContext| {
                    format!("{}:{markdown}", context.message_type.as_str())
                },
            ],
            ..AssistantMessageOptions::default()
        },
    );
    let rendered = strip_ansi(&component.render(80).join("\n"));
    assert!(rendered.contains("assistant:answer"));
    assert!(rendered.contains("assistant-thinking:reasoning"));
    assert_eq!(original, snapshot);
}

#[test]
fn uses_configured_output_padding_for_user_messages() {
    init_theme(Some("dark"), false);
    let padded = UserMessageComponent::new("hello", None, 1, Arc::new(Vec::new()));
    let padded_lines = padded
        .render(40)
        .into_iter()
        .map(|line| strip_ansi(&line))
        .collect::<Vec<_>>();
    assert!(padded_lines.iter().any(|line| line.starts_with(" hello")));

    let unpadded = UserMessageComponent::new("hello", None, 0, Arc::new(Vec::new()));
    let unpadded_lines = unpadded
        .render(40)
        .into_iter()
        .map(|line| strip_ansi(&line))
        .collect::<Vec<_>>();
    assert!(unpadded_lines.iter().any(|line| line.starts_with("hello")));
}
