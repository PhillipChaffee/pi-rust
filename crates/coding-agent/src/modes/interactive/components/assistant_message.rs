//! The assistant message block, upstream
//! `src/modes/interactive/components/assistant-message.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Adjacent thinking blocks coalesce into one clickable run; a click toggles
//! that run only, and `set_hide_thinking_block` clears every override.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use pi_ai::types::{AssistantBlock, AssistantMessage, StopReason};
use pi_tui::components::markdown::DefaultTextStyle;
use pi_tui::components::{
    ColorFn, Markdown, MarkdownOptions, MarkdownTheme, MouseRegion, Spacer, Text,
};
use pi_tui::tui::{
    Component, Container, TuiMouseButton, TuiMouseEvent, TuiMouseEventResult, TuiMouseEventType,
};

use super::markdown_transform::create_markdown_transform;
use crate::extensions::types::{MarkdownMessageType, MarkdownTransformer};
use crate::modes::interactive::theme::theme::{get_markdown_theme, italic, theme};

const OSC133_ZONE_START: &str = "\u{1b}]133;A\u{7}";
const OSC133_ZONE_END: &str = "\u{1b}]133;B\u{7}";
const OSC133_ZONE_FINAL: &str = "\u{1b}]133;C\u{7}";

/// The assistant row's constructor options.
#[derive(Clone)]
pub struct AssistantMessageOptions {
    /// Whether thinking runs start hidden.
    pub hide_thinking_block: bool,
    /// Markdown theme override; the active theme's when absent.
    pub markdown_theme: Option<MarkdownTheme>,
    /// The label shown for a hidden thinking run.
    pub hidden_thinking_label: String,
    /// Horizontal padding for text and thinking blocks.
    pub output_pad: usize,
    /// Extension Markdown transformers, applied in registration order.
    pub markdown_transformers: Arc<Vec<MarkdownTransformer>>,
}

impl std::fmt::Debug for AssistantMessageOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssistantMessageOptions")
            .field("hide_thinking_block", &self.hide_thinking_block)
            .field("markdown_theme", &self.markdown_theme.is_some())
            .field("hidden_thinking_label", &self.hidden_thinking_label)
            .field("output_pad", &self.output_pad)
            .field("markdown_transformers", &self.markdown_transformers.len())
            .finish()
    }
}

impl Default for AssistantMessageOptions {
    fn default() -> Self {
        Self {
            hide_thinking_block: false,
            markdown_theme: None,
            hidden_thinking_label: "Thinking...".to_string(),
            output_pad: 1,
            markdown_transformers: Arc::new(Vec::new()),
        }
    }
}

/// The per-run visibility state the mouse regions share with the row.
#[derive(Default)]
struct ThinkingToggleState {
    overrides: RefCell<HashMap<usize, bool>>,
    dirty: Cell<bool>,
}

/// The component that renders a complete assistant message.
pub struct AssistantMessageComponent {
    content_container: Container,
    hide_thinking_block: Cell<bool>,
    markdown_theme: MarkdownTheme,
    hidden_thinking_label: RefCell<String>,
    output_pad: Cell<usize>,
    markdown_transformers: Arc<Vec<MarkdownTransformer>>,
    last_message: RefCell<Option<AssistantMessage>>,
    has_tool_calls: Cell<bool>,
    is_streaming: Cell<bool>,
    thinking: Rc<ThinkingToggleState>,
}

impl AssistantMessageComponent {
    /// Build the row, upstream's constructor.
    #[must_use]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the constructor stores its own copy; callers keep the message they pass"
    )]
    pub fn new(message: Option<AssistantMessage>, options: AssistantMessageOptions) -> Self {
        let markdown_theme = options.markdown_theme.unwrap_or_else(get_markdown_theme);
        let component = Self {
            content_container: Container::default(),
            hide_thinking_block: Cell::new(options.hide_thinking_block),
            markdown_theme,
            hidden_thinking_label: RefCell::new(options.hidden_thinking_label),
            output_pad: Cell::new(options.output_pad),
            markdown_transformers: options.markdown_transformers,
            last_message: RefCell::new(message.clone()),
            has_tool_calls: Cell::new(false),
            is_streaming: Cell::new(false),
            thinking: Rc::new(ThinkingToggleState::default()),
        };
        if let Some(message) = &message {
            component.update_content(message, false);
        }
        component
    }

    /// Set whether thinking runs start hidden, clearing per-run overrides.
    pub fn set_hide_thinking_block(&self, hide: bool) {
        self.hide_thinking_block.set(hide);
        self.thinking.overrides.borrow_mut().clear();
        self.refresh();
    }

    /// Set the label shown for hidden thinking runs.
    pub fn set_hidden_thinking_label(&self, label: &str) {
        *self.hidden_thinking_label.borrow_mut() = label.to_string();
        self.refresh();
    }

    /// Set the horizontal output padding.
    pub fn set_output_pad(&self, padding: usize) {
        self.output_pad.set(padding);
        self.refresh();
    }

    /// Re-render the last message with the current settings.
    fn refresh(&self) {
        let message = self.last_message.borrow().clone();
        if let Some(message) = message {
            self.update_content(&message, self.is_streaming.get());
        }
    }

    /// Rebuild the content from `message`, upstream's `updateContent`.
    pub fn update_content(&self, message: &AssistantMessage, is_streaming: bool) {
        *self.last_message.borrow_mut() = Some(message.clone());
        self.is_streaming.set(is_streaming);
        self.thinking.dirty.set(false);
        self.content_container.clear();
        self.rebuild_content(message, is_streaming);
    }

    /// The content walk, shared by `update_content` and the deferred click
    /// rebuild.
    fn rebuild_content(&self, message: &AssistantMessage, is_streaming: bool) {
        let has_visible_content = message.content.iter().any(visible_block);
        if has_visible_content {
            self.content_container.add_child(Rc::new(Spacer::new()));
        }

        let mut thinking_run_index = 0;
        let mut index = 0;
        while index < message.content.len() {
            match &message.content[index] {
                AssistantBlock::Text(text) if !text.text.trim().is_empty() => {
                    let transform = create_markdown_transform(
                        MarkdownMessageType::Assistant,
                        is_streaming,
                        Arc::clone(&self.markdown_transformers),
                    );
                    self.content_container
                        .add_child(Rc::new(Markdown::with_options(
                            text.text.trim(),
                            self.output_pad.get(),
                            0,
                            self.markdown_theme.clone(),
                            None,
                            MarkdownOptions {
                                transform: Some(transform),
                                ..MarkdownOptions::default()
                            },
                        )));
                    index += 1;
                }
                AssistantBlock::Thinking(_) => {
                    let mut thinking_blocks = Vec::new();
                    while index < message.content.len() {
                        let AssistantBlock::Thinking(thinking) = &message.content[index] else {
                            break;
                        };
                        let thinking = thinking.thinking.trim();
                        if !thinking.is_empty() {
                            thinking_blocks.push(thinking.to_string());
                        }
                        index += 1;
                    }
                    let has_visible_after = message.content[index..].iter().any(visible_block);
                    if thinking_blocks.is_empty() {
                        continue;
                    }
                    let run_index = thinking_run_index;
                    thinking_run_index += 1;
                    self.add_thinking_run(
                        run_index,
                        &thinking_blocks,
                        is_streaming,
                        has_visible_after,
                    );
                }
                _ => index += 1,
            }
        }

        let has_tool_calls = message
            .content
            .iter()
            .any(|block| matches!(block, AssistantBlock::ToolCall(_)));
        self.has_tool_calls.set(has_tool_calls);
        self.add_stop_notice(message, has_tool_calls);
    }
}

/// Whether a block carries visible content, upstream's predicate.
fn visible_block(block: &AssistantBlock) -> bool {
    match block {
        AssistantBlock::Text(text) => !text.text.trim().is_empty(),
        AssistantBlock::Thinking(thinking) => !thinking.thinking.trim().is_empty(),
        AssistantBlock::ToolCall(_) => false,
    }
}

impl AssistantMessageComponent {
    /// Add one coalesced thinking run and its click region, upstream's
    /// thinking branch.
    fn add_thinking_run(
        &self,
        run_index: usize,
        thinking_blocks: &[String],
        is_streaming: bool,
        has_visible_after: bool,
    ) {
        let hidden = self
            .thinking
            .overrides
            .borrow()
            .get(&run_index)
            .copied()
            .unwrap_or_else(|| self.hide_thinking_block.get());
        let thinking_component: Rc<dyn Component> = if hidden {
            Rc::new(Text::with_padding(
                italic(&theme().fg("thinkingText", &self.hidden_thinking_label.borrow())),
                self.output_pad.get(),
                0,
            ))
        } else {
            let transform = create_markdown_transform(
                MarkdownMessageType::AssistantThinking,
                is_streaming,
                Arc::clone(&self.markdown_transformers),
            );
            let active = theme();
            let color: ColorFn = Arc::new(move |text| active.fg("thinkingText", text));
            Rc::new(Markdown::with_options(
                thinking_blocks.join("\n\n"),
                self.output_pad.get(),
                0,
                self.markdown_theme.clone(),
                Some(DefaultTextStyle {
                    color: Some(color),
                    italic: true,
                    ..DefaultTextStyle::default()
                }),
                MarkdownOptions {
                    transform: Some(transform),
                    ..MarkdownOptions::default()
                },
            ))
        };
        let state = Rc::clone(&self.thinking);
        let default_hidden = self.hide_thinking_block.get();
        let handler = move |event: &TuiMouseEvent| {
            if event.event_type != TuiMouseEventType::Click || event.button != TuiMouseButton::Left
            {
                return None;
            }
            let mut overrides = state.overrides.borrow_mut();
            let hidden = overrides.get(&run_index).copied().unwrap_or(default_hidden);
            overrides.insert(run_index, !hidden);
            drop(overrides);
            state.dirty.set(true);
            Some(TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            })
        };
        self.content_container
            .add_child(Rc::new(MouseRegion::new(thinking_component, handler)));
        if has_visible_after {
            self.content_container.add_child(Rc::new(Spacer::new()));
        }
    }

    /// Add the truncated/aborted/error notice, upstream's stop-reason ladder.
    fn add_stop_notice(&self, message: &AssistantMessage, has_tool_calls: bool) {
        let padding = self.output_pad.get();
        if message.stop_reason == StopReason::Length {
            self.content_container.add_child(Rc::new(Spacer::new()));
            self.content_container.add_child(Rc::new(Text::with_padding(
                theme().fg("error", "Response was truncated before completion."),
                padding,
                0,
            )));
            return;
        }
        if has_tool_calls {
            return;
        }
        let notice = match message.stop_reason {
            StopReason::Aborted => Some(
                message
                    .error_message
                    .as_deref()
                    .filter(|message| *message != "Request was aborted")
                    .unwrap_or("Operation aborted")
                    .to_string(),
            ),
            StopReason::Error => Some(format!(
                "Error: {}",
                message.error_message.as_deref().unwrap_or("Unknown error")
            )),
            _ => None,
        };
        if let Some(notice) = notice {
            self.content_container.add_child(Rc::new(Spacer::new()));
            self.content_container.add_child(Rc::new(Text::with_padding(
                theme().fg("error", &notice),
                padding,
                0,
            )));
        }
    }
}

impl std::fmt::Debug for AssistantMessageComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssistantMessageComponent")
            .finish_non_exhaustive()
    }
}

impl Component for AssistantMessageComponent {
    fn render(&self, width: usize) -> Vec<String> {
        let dirty = self.thinking.dirty.replace(false);
        let message = self.last_message.borrow().clone();
        if dirty && let Some(message) = message {
            self.content_container.clear();
            self.rebuild_content(&message, self.is_streaming.get());
        }
        let mut lines = self.content_container.render(width);
        if self.has_tool_calls.get() || lines.is_empty() {
            return lines;
        }
        lines[0] = format!("{OSC133_ZONE_START}{}", lines[0]);
        let last = lines.len() - 1;
        lines[last] = format!("{OSC133_ZONE_END}{OSC133_ZONE_FINAL}{}", lines[last]);
        lines
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.content_container.handle_mouse(event)
    }

    fn invalidate(&self) {
        self.refresh();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.content_container.children()
    }
}
