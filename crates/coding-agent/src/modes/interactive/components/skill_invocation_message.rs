//! The skill-invocation block, upstream
//! `src/modes/interactive/components/skill-invocation-message.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The block renders only the skill request itself; the user message that
//! follows a skill invocation is a separate row. Collapsed, it shows the
//! `[skill]` tag, the skill name, and the expand binding; expanded, it adds
//! the skill body as Markdown.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::markdown::DefaultTextStyle;
use pi_tui::components::{Box, ColorFn, Markdown, MarkdownTheme, Text};
use pi_tui::tui::Component;

use super::keybinding_hints::key_text;
use crate::core::agent_session::ParsedSkillBlock;
use crate::modes::interactive::theme::theme::{get_markdown_theme, theme};

/// Component that renders a skill invocation message with collapsed/expanded
/// state.
pub struct SkillInvocationMessageComponent {
    expanded: Cell<bool>,
    skill_block: ParsedSkillBlock,
    markdown_theme: MarkdownTheme,
    box_component: Box,
}

impl std::fmt::Debug for SkillInvocationMessageComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillInvocationMessageComponent")
            .field("expanded", &self.expanded.get())
            .field("name", &self.skill_block.name)
            .finish_non_exhaustive()
    }
}

impl SkillInvocationMessageComponent {
    /// Build the block, upstream's constructor.
    #[must_use]
    pub fn new(skill_block: ParsedSkillBlock, markdown_theme: Option<MarkdownTheme>) -> Self {
        let active = theme();
        let background: ColorFn = {
            let active = Arc::clone(&active);
            Arc::new(move |text| active.bg("customMessageBg", text))
        };
        let component = Self {
            expanded: Cell::new(false),
            skill_block,
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

    /// Whether the skill body is expanded.
    #[must_use]
    pub const fn is_expanded(&self) -> bool {
        self.expanded.get()
    }

    /// Rebuild the box contents, upstream's `updateDisplay`.
    fn update_display(&self) {
        self.box_component.clear();
        let active = theme();

        if self.expanded.get() {
            let label = active.fg("customMessageLabel", "\u{1b}[1m[skill]\u{1b}[22m");
            self.box_component
                .add_child(Rc::new(Text::with_padding(label, 0, 0)));
            let content = format!(
                "**{}**\n\n{}",
                self.skill_block.name, self.skill_block.content
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
            let line = active.fg("customMessageLabel", "\u{1b}[1m[skill]\u{1b}[22m ")
                + &active.fg("customMessageText", &self.skill_block.name)
                + &active.fg(
                    "dim",
                    &format!(" ({} to expand)", key_text("app.tools.expand")),
                );
            self.box_component
                .add_child(Rc::new(Text::with_padding(line, 0, 0)));
        }
    }
}

impl Component for SkillInvocationMessageComponent {
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
