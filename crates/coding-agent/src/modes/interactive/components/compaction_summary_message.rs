//! The compaction-summary block, upstream
//! `src/modes/interactive/components/compaction-summary-message.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Collapsed, the row shows the pre-compaction token count and the expand
//! binding; expanded, it renders the summary as Markdown under a bold header.
//! Restatement: upstream's `tokensBefore.toLocaleString()` becomes
//! `format_tokens`, which groups digits with commas as the en-US locale
//! does. The count itself comes from the compaction entry; nothing here
//! estimates it.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::harness::messages::CompactionSummaryMessage;
use pi_tui::components::markdown::DefaultTextStyle;
use pi_tui::components::{Box, ColorFn, Markdown, MarkdownTheme, Spacer, Text};
use pi_tui::tui::Component;

use super::keybinding_hints::key_text;
use crate::modes::interactive::theme::theme::{get_markdown_theme, theme};

/// Component that renders a compaction message with collapsed/expanded state.
pub struct CompactionSummaryMessageComponent {
    expanded: Cell<bool>,
    message: CompactionSummaryMessage,
    markdown_theme: MarkdownTheme,
    box_component: Box,
}

impl std::fmt::Debug for CompactionSummaryMessageComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactionSummaryMessageComponent")
            .field("expanded", &self.expanded.get())
            .field("tokens_before", &self.message.tokens_before)
            .finish_non_exhaustive()
    }
}

impl CompactionSummaryMessageComponent {
    /// Build the block, upstream's constructor.
    #[must_use]
    pub fn new(message: CompactionSummaryMessage, markdown_theme: Option<MarkdownTheme>) -> Self {
        let active = theme();
        let background: ColorFn = {
            let active = Arc::clone(&active);
            Arc::new(move |text| active.bg("customMessageBg", text))
        };
        let component = Self {
            expanded: Cell::new(false),
            message,
            markdown_theme: markdown_theme.unwrap_or_else(get_markdown_theme),
            box_component: Box::with_padding(1, 1, Some(background)),
        };
        component.update_display();
        component
    }

    /// Set the expanded state and repaint.
    pub fn set_expanded(&self, expanded: bool) {
        self.expanded.set(expanded);
        self.update_display();
    }

    /// Whether the summary is expanded.
    #[must_use]
    pub const fn is_expanded(&self) -> bool {
        self.expanded.get()
    }

    /// Rebuild the box contents, upstream's `updateDisplay`.
    fn update_display(&self) {
        self.box_component.clear();
        let active = theme();
        let token_str = format_tokens(self.message.tokens_before);
        let label = active.fg("customMessageLabel", "\u{1b}[1m[compaction]\u{1b}[22m");
        self.box_component
            .add_child(Rc::new(Text::with_padding(label, 0, 0)));
        self.box_component.add_child(Rc::new(Spacer::new()));

        if self.expanded.get() {
            let content = format!(
                "**Compacted from {token_str} tokens**\n\n{}",
                self.message.summary
            );
            let color: ColorFn = {
                let active = Arc::clone(&active);
                Arc::new(move |text| active.fg("customMessageText", text))
            };
            self.box_component.add_child(Rc::new(Markdown::with_style(
                content,
                0,
                0,
                self.markdown_theme.clone(),
                DefaultTextStyle {
                    color: Some(color),
                    ..DefaultTextStyle::default()
                },
            )));
        } else {
            let line = active.fg(
                "customMessageText",
                &format!("Compacted from {token_str} tokens ("),
            ) + &active.fg("dim", &key_text("app.tools.expand"))
                + &active.fg("customMessageText", " to expand)");
            self.box_component
                .add_child(Rc::new(Text::with_padding(line, 0, 0)));
        }
    }
}

impl Component for CompactionSummaryMessageComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.box_component.render(width)
    }

    fn handle_mouse(
        &self,
        event: &pi_tui::tui::TuiMouseEvent,
    ) -> Option<pi_tui::tui::TuiMouseEventResult> {
        self.box_component.handle_mouse(event)
    }

    fn invalidate(&self) {
        self.update_display();
    }
}

/// Group a token count with commas, upstream's default-locale
/// `toLocaleString()`.
fn format_tokens(value: i64) -> String {
    let digits = value.unsigned_abs().to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    if value < 0 {
        format!("-{grouped}")
    } else {
        grouped
    }
}
