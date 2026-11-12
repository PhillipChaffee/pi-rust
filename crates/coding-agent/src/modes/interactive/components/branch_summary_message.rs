//! The branch-summary block, upstream
//! `src/modes/interactive/components/branch-summary-message.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Collapsed, the row names the summary and the expand binding; expanded, it
//! renders the summary as Markdown under a bold header. The box shares the
//! custom-message background so summaries read as transcript annotations.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::harness::messages::BranchSummaryMessage;
use pi_tui::components::markdown::DefaultTextStyle;
use pi_tui::components::{Box, ColorFn, Markdown, MarkdownTheme, Spacer, Text};
use pi_tui::tui::Component;

use super::keybinding_hints::key_text;
use crate::modes::interactive::theme::theme::{get_markdown_theme, theme};

/// Component that renders a branch summary message with collapsed/expanded
/// state.
pub struct BranchSummaryMessageComponent {
    expanded: Cell<bool>,
    message: BranchSummaryMessage,
    markdown_theme: MarkdownTheme,
    box_component: Box,
}

impl std::fmt::Debug for BranchSummaryMessageComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BranchSummaryMessageComponent")
            .field("expanded", &self.expanded.get())
            .finish_non_exhaustive()
    }
}

impl BranchSummaryMessageComponent {
    /// Build the block, upstream's constructor.
    #[must_use]
    pub fn new(message: BranchSummaryMessage, markdown_theme: Option<MarkdownTheme>) -> Self {
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
        let label = active.fg("customMessageLabel", "\u{1b}[1m[branch]\u{1b}[22m");
        self.box_component
            .add_child(Rc::new(Text::with_padding(label, 0, 0)));
        self.box_component.add_child(Rc::new(Spacer::new()));

        if self.expanded.get() {
            let content = format!("**Branch Summary**\n\n{}", self.message.summary);
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
            let line = active.fg("customMessageText", "Branch summary (")
                + &active.fg("dim", &key_text("app.tools.expand"))
                + &active.fg("customMessageText", " to expand)");
            self.box_component
                .add_child(Rc::new(Text::with_padding(line, 0, 0)));
        }
    }
}

impl Component for BranchSummaryMessageComponent {
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
