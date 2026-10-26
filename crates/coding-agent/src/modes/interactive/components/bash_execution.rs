//! The streaming bash execution block, upstream
//! `src/modes/interactive/components/bash-execution.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Output accumulates through [`BashExecutionComponent::append_output`] and is
//! context-truncated before the preview, so the collapsed row shows the tail
//! the model sees.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::{Loader, Spacer, Text};
use pi_tui::tui::Container;
use pi_tui::tui::{Component, RenderRequest};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::{key_hint, key_text};
use super::visual_truncate::truncate_to_visual_lines;
use crate::modes::interactive::theme::theme::theme;
use crate::tools::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationOptions, TruncationResult, truncate_tail,
};
use crate::utils::ansi::strip_ansi;

/// The collapsed preview line count.
const PREVIEW_LINES: usize = 20;

/// The block's lifecycle state, upstream's status union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BashStatus {
    Running,
    Complete,
    Cancelled,
    Error,
}

/// The width-keyed preview cache, upstream's render closure fields.
#[derive(Default)]
struct PreviewState {
    width: Option<usize>,
    lines: Option<Vec<String>>,
}

/// The collapsed preview block.
struct PreviewComponent {
    styled_input: String,
    state: RefCell<PreviewState>,
}

impl Component for PreviewComponent {
    fn render(&self, width: usize) -> Vec<String> {
        let mut state = self.state.borrow_mut();
        if state.lines.is_none() || state.width != Some(width) {
            let result = truncate_to_visual_lines(&self.styled_input, PREVIEW_LINES, width, 1);
            state.lines = Some(result.visual_lines);
            state.width = Some(width);
        }
        state.lines.clone().unwrap_or_default()
    }

    fn invalidate(&self) {
        *self.state.borrow_mut() = PreviewState::default();
    }
}

/// One bash execution block.
pub struct BashExecutionComponent {
    command: String,
    output_lines: RefCell<Vec<String>>,
    status: Cell<BashStatus>,
    exit_code: Cell<Option<i32>>,
    loader: Rc<Loader>,
    truncation_result: RefCell<Option<TruncationResult>>,
    full_output_path: RefCell<Option<String>>,
    expanded: Cell<bool>,
    content_container: Rc<Container>,
    root: Container,
}

impl BashExecutionComponent {
    /// Build the block, upstream's constructor.
    #[must_use]
    pub fn new(command: &str, render_request: RenderRequest, exclude_from_context: bool) -> Self {
        let color_key = if exclude_from_context {
            "dim"
        } else {
            "bashMode"
        };
        let border_color: Rc<dyn Fn(&str) -> String> = {
            let color_key = color_key.to_string();
            Rc::new(move |text| theme().fg(&color_key, text))
        };
        let spinner_key = color_key.to_string();
        let spinner_color: pi_tui::components::ColorFn =
            Arc::new(move |text| theme().fg(&spinner_key, text));
        let message_color: pi_tui::components::ColorFn = Arc::new(|text| theme().fg("muted", text));
        let loader = Rc::new(Loader::new(
            render_request,
            spinner_color,
            message_color,
            format!("Running... ({} to cancel)", key_text("tui.select.cancel")),
            None,
        ));
        loader.start();

        let content_container = Rc::new(Container::default());
        content_container.add_child(Rc::new(Text::with_padding(
            theme().fg(
                color_key,
                &super::super::theme::theme::bold(&format!("$ {command}")),
            ),
            1,
            0,
        )));
        let loader_component: Rc<dyn Component> = loader.clone();
        content_container.add_child(loader_component);

        let root = Container::default();
        root.add_child(Rc::new(Spacer::new()));
        root.add_child(Rc::new(DynamicBorder::new(Rc::clone(&border_color))));
        let content_component: Rc<dyn Component> = content_container.clone();
        root.add_child(content_component);
        root.add_child(Rc::new(DynamicBorder::new(border_color)));

        Self {
            command: command.to_string(),
            output_lines: RefCell::new(Vec::new()),
            status: Cell::new(BashStatus::Running),
            exit_code: Cell::new(None),
            loader,
            truncation_result: RefCell::new(None),
            full_output_path: RefCell::new(None),
            expanded: Cell::new(false),
            content_container,
            root,
        }
    }

    /// Set whether the output is expanded and repaint.
    pub fn set_expanded(&self, expanded: bool) {
        self.expanded.set(expanded);
        self.update_display();
    }

    /// Append a chunk, stripping ANSI and normalizing line endings, upstream's
    /// `appendOutput`.
    pub fn append_output(&self, chunk: &str) {
        let clean = strip_ansi(chunk).replace("\r\n", "\n").replace('\r', "\n");
        let new_lines: Vec<&str> = clean.split('\n').collect();
        let mut output_lines = self.output_lines.borrow_mut();
        if !output_lines.is_empty() && !new_lines.is_empty() {
            let last = output_lines.len() - 1;
            output_lines[last].push_str(new_lines[0]);
            output_lines.extend(new_lines[1..].iter().map(|line| (*line).to_string()));
        } else {
            output_lines.extend(new_lines.iter().map(|line| (*line).to_string()));
        }
        drop(output_lines);
        self.update_display();
    }

    /// Mark the command settled, upstream's `setComplete`.
    pub fn set_complete(
        &self,
        exit_code: Option<i32>,
        cancelled: bool,
        truncation_result: Option<TruncationResult>,
        full_output_path: Option<&str>,
    ) {
        self.exit_code.set(exit_code);
        let status = if cancelled {
            BashStatus::Cancelled
        } else if exit_code.is_some_and(|code| code != 0) {
            BashStatus::Error
        } else {
            BashStatus::Complete
        };
        self.status.set(status);
        *self.truncation_result.borrow_mut() = truncation_result;
        *self.full_output_path.borrow_mut() = full_output_path.map(str::to_string);
        self.loader.stop();
        self.update_display();
    }

    /// The raw output, upstream's `getOutput`.
    #[must_use]
    pub fn get_output(&self) -> String {
        self.output_lines.borrow().join("\n")
    }

    /// The executed command, upstream's `getCommand`.
    #[must_use]
    pub fn get_command(&self) -> &str {
        &self.command
    }

    /// Rebuild the block, upstream's `updateDisplay`.
    #[expect(
        clippy::too_many_lines,
        reason = "the truncation, preview, loader, and status ladder are one upstream method"
    )]
    #[expect(
        clippy::option_if_let_else,
        reason = "the match keeps the error label's two arms explicit; the closure form adds function instances the coverage budget counts"
    )]
    fn update_display(&self) {
        let theme = theme();
        let full_output = self.output_lines.borrow().join("\n");
        let context_truncation = truncate_tail(
            &full_output,
            TruncationOptions {
                max_lines: Some(DEFAULT_MAX_LINES),
                max_bytes: Some(DEFAULT_MAX_BYTES),
            },
        );
        let available_lines: Vec<&str> = if context_truncation.content.is_empty() {
            Vec::new()
        } else {
            context_truncation.content.split('\n').collect()
        };
        let preview_logical_lines: Vec<&str> = available_lines
            .iter()
            .skip(available_lines.len().saturating_sub(PREVIEW_LINES))
            .copied()
            .collect();
        let hidden_line_count = available_lines.len() - preview_logical_lines.len();

        let content = Rc::clone(&self.content_container);
        content.clear();
        content.add_child(Rc::new(Text::with_padding(
            theme.fg(
                "bashMode",
                &super::super::theme::theme::bold(&format!("$ {}", self.command)),
            ),
            1,
            0,
        )));

        if !available_lines.is_empty() {
            if self.expanded.get() {
                let display_text = available_lines
                    .iter()
                    .map(|line| theme.fg("muted", line))
                    .collect::<Vec<_>>()
                    .join("\n");
                content.add_child(Rc::new(Text::with_padding(
                    format!("\n{display_text}"),
                    1,
                    0,
                )));
            } else {
                let styled_output = preview_logical_lines
                    .iter()
                    .map(|line| theme.fg("muted", line))
                    .collect::<Vec<_>>()
                    .join("\n");
                content.add_child(Rc::new(PreviewComponent {
                    styled_input: format!("\n{styled_output}"),
                    state: RefCell::new(PreviewState::default()),
                }));
            }
        }

        if self.status.get() == BashStatus::Running {
            let loader_component: Rc<dyn Component> = self.loader.clone();
            content.add_child(loader_component);
            return;
        }

        let mut status_parts = Vec::new();
        if hidden_line_count > 0 {
            if self.expanded.get() {
                status_parts.push(format!(
                    "{}{}{}",
                    theme.fg("muted", "("),
                    key_hint("app.tools.expand", "to collapse"),
                    theme.fg("muted", ")")
                ));
            } else {
                status_parts.push(format!(
                    "{}{}{}",
                    theme.fg("muted", &format!("... {hidden_line_count} more lines (")),
                    key_hint("app.tools.expand", "to expand"),
                    theme.fg("muted", ")")
                ));
            }
        }
        match self.status.get() {
            BashStatus::Cancelled => status_parts.push(theme.fg("warning", "(cancelled)")),
            BashStatus::Error => {
                let exit = match self.exit_code.get() {
                    Some(code) => code.to_string(),
                    None => "unknown".to_string(),
                };
                status_parts.push(theme.fg("error", &format!("(exit {exit})")));
            }
            _ => {}
        }
        let was_truncated = self
            .truncation_result
            .borrow()
            .as_ref()
            .is_some_and(|result| result.truncated)
            || context_truncation.truncated;
        if was_truncated && let Some(full_output_path) = self.full_output_path.borrow().as_deref() {
            status_parts.push(theme.fg(
                "warning",
                &format!("Output truncated. Full output: {full_output_path}"),
            ));
        }
        if !status_parts.is_empty() {
            content.add_child(Rc::new(Text::with_padding(
                format!("\n{}", status_parts.join("\n")),
                1,
                0,
            )));
        }
    }
}

impl std::fmt::Debug for BashExecutionComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashExecutionComponent")
            .field("command", &self.command)
            .finish_non_exhaustive()
    }
}

impl Component for BashExecutionComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.root.render(width)
    }

    fn handle_mouse(
        &self,
        event: &pi_tui::tui::TuiMouseEvent,
    ) -> Option<pi_tui::tui::TuiMouseEventResult> {
        self.root.handle_mouse(event)
    }

    fn invalidate(&self) {
        self.root.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.root.children()
    }
}
