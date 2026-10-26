//! The user message block, upstream
//! `src/modes/interactive/components/user-message.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The row paints the user-message background across the whole box and keeps
//! the closing OSC 133 markers off the last content line so the zone stays
//! height-stable.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::markdown::DefaultTextStyle;
use pi_tui::components::{Box, ColorFn, Markdown, MarkdownOptions, MarkdownTheme};
use pi_tui::tui::{Component, Container};

use super::markdown_transform::create_markdown_transform;
use crate::extensions::types::{MarkdownMessageType, MarkdownTransformer};
use crate::modes::interactive::theme::theme::{get_markdown_theme, theme};

const OSC133_ZONE_START: &str = "\u{1b}]133;A\u{7}";
const OSC133_ZONE_END: &str = "\u{1b}]133;B\u{7}";
const OSC133_ZONE_FINAL: &str = "\u{1b}]133;C\u{7}";

/// The component that renders a user message.
pub struct UserMessageComponent {
    text: String,
    markdown_theme: MarkdownTheme,
    output_pad: Cell<usize>,
    markdown_transformers: Arc<Vec<MarkdownTransformer>>,
    container: Container,
}

impl UserMessageComponent {
    /// Build the row, upstream's constructor.
    #[must_use]
    pub fn new(
        text: &str,
        markdown_theme: Option<MarkdownTheme>,
        output_pad: usize,
        markdown_transformers: Arc<Vec<MarkdownTransformer>>,
    ) -> Self {
        let component = Self {
            text: text.to_string(),
            markdown_theme: markdown_theme.unwrap_or_else(get_markdown_theme),
            output_pad: Cell::new(output_pad),
            markdown_transformers,
            container: Container::default(),
        };
        component.rebuild();
        component
    }

    /// Set the horizontal output padding and repaint.
    pub fn set_output_pad(&self, padding: usize) {
        self.output_pad.set(padding);
        self.rebuild();
    }

    /// Rebuild the background box and its markdown child, upstream's
    /// `rebuild`.
    fn rebuild(&self) {
        self.container.clear();
        let active = theme();
        let background: ColorFn = {
            let active = Arc::clone(&active);
            Arc::new(move |text| active.bg("userMessageBg", text))
        };
        let content_box = Box::with_padding(
            u16::try_from(self.output_pad.get()).unwrap_or(u16::MAX),
            1,
            Some(background),
        );
        let color: ColorFn = Arc::new(move |text| active.fg("userMessageText", text));
        content_box.add_child(Rc::new(Markdown::with_options(
            &self.text,
            0,
            0,
            self.markdown_theme.clone(),
            Some(DefaultTextStyle {
                color: Some(color),
                ..DefaultTextStyle::default()
            }),
            MarkdownOptions {
                preserve_ordered_list_markers: true,
                preserve_backslash_escapes: true,
                transform: Some(create_markdown_transform(
                    MarkdownMessageType::User,
                    false,
                    Arc::clone(&self.markdown_transformers),
                )),
                ..MarkdownOptions::default()
            },
        )));
        self.container.add_child(Rc::new(content_box));
    }
}

impl std::fmt::Debug for UserMessageComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserMessageComponent")
            .finish_non_exhaustive()
    }
}

impl Component for UserMessageComponent {
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines = self.container.render(width);
        if lines.is_empty() {
            return lines;
        }
        lines[0] = format!("{OSC133_ZONE_START}{}", lines[0]);
        let last = lines.len() - 1;
        lines[last] = format!("{OSC133_ZONE_END}{OSC133_ZONE_FINAL}{}", lines[last]);
        lines
    }

    fn handle_mouse(
        &self,
        event: &pi_tui::tui::TuiMouseEvent,
    ) -> Option<pi_tui::tui::TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }

    fn invalidate(&self) {
        self.rebuild();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }
}
