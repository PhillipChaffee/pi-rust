//! The custom-message block, upstream
//! `src/modes/interactive/components/custom-message.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! An extension renderer draws its own styling; when it declines or throws,
//! the default purple box shows the custom type and its text content.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::types::CustomAgentMessage;
use pi_tui::components::markdown::DefaultTextStyle;
use pi_tui::components::{Box, ColorFn, Markdown, MarkdownTheme, Spacer, Text};
use pi_tui::tui::{Component, Container};

use crate::extensions::types::{MessageRenderOptions, MessageRenderer};
use crate::modes::interactive::theme::theme::{get_markdown_theme, theme};

/// The component that renders a custom message entry from extensions.
pub struct CustomMessageComponent {
    message: CustomAgentMessage,
    custom_renderer: Option<MessageRenderer>,
    box_component: Rc<Box>,
    custom_component: RefCell<Option<Rc<dyn Component>>>,
    markdown_theme: MarkdownTheme,
    expanded: Cell<bool>,
    output_pad: Cell<usize>,
    container: Container,
}

impl CustomMessageComponent {
    /// Build the row, upstream's constructor.
    #[must_use]
    pub fn new(
        message: CustomAgentMessage,
        custom_renderer: Option<MessageRenderer>,
        markdown_theme: Option<MarkdownTheme>,
        output_pad: usize,
    ) -> Self {
        let component = Self {
            message,
            custom_renderer,
            box_component: Rc::new(Box::with_padding(1, 1, None)),
            custom_component: RefCell::new(None),
            markdown_theme: markdown_theme.unwrap_or_else(get_markdown_theme),
            expanded: Cell::new(false),
            output_pad: Cell::new(output_pad),
            container: Container::default(),
        };
        component.container.add_child(Rc::new(Spacer::new()));
        component.rebuild();
        component
    }

    /// Set whether the message is expanded and repaint.
    pub fn set_expanded(&self, expanded: bool) {
        if self.expanded.get() != expanded {
            self.expanded.set(expanded);
            self.rebuild();
        }
    }

    /// Set the horizontal output padding and repaint.
    pub fn set_output_pad(&self, output_pad: usize) {
        if self.output_pad.get() != output_pad {
            self.output_pad.set(output_pad);
            self.rebuild();
        }
    }

    /// Rebuild the box or custom child, upstream's `rebuild`.
    fn rebuild(&self) {
        if let Some(custom_component) = self.custom_component.borrow_mut().take() {
            self.container.remove_child(&custom_component);
        }
        let box_component: Rc<dyn Component> = self.box_component.clone();
        self.container.remove_child(&box_component);

        if let Some(renderer) = &self.custom_renderer {
            let options = MessageRenderOptions {
                expanded: self.expanded.get(),
                output_pad: self.output_pad.get(),
            };
            let active = theme();
            if let Some(component) = renderer(&self.message, &options, &active) {
                self.container.add_child(Rc::clone(&component));
                *self.custom_component.borrow_mut() = Some(component);
                return;
            }
        }

        let active = theme();
        let background: ColorFn = {
            let active = Arc::clone(&active);
            Arc::new(move |text| active.bg("customMessageBg", text))
        };
        self.box_component.set_bg_fn(Some(background));
        self.container.add_child(box_component);
        self.box_component.clear();
        let label = active.fg(
            "customMessageLabel",
            &format!("\u{1b}[1m[{}]\u{1b}[22m", self.custom_type()),
        );
        self.box_component
            .add_child(Rc::new(Text::with_padding(label, 0, 0)));
        self.box_component.add_child(Rc::new(Spacer::new()));
        let color: ColorFn = Arc::new(move |text| active.fg("customMessageText", text));
        self.box_component.add_child(Rc::new(Markdown::with_style(
            self.text_content(),
            0,
            0,
            self.markdown_theme.clone(),
            DefaultTextStyle {
                color: Some(color),
                ..DefaultTextStyle::default()
            },
        )));
    }

    /// The custom type tag, upstream's `message.customType`.
    fn custom_type(&self) -> String {
        self.message
            .field("customType")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string()
    }

    /// The message's text, upstream's string-or-text-blocks extraction.
    fn text_content(&self) -> String {
        match self.message.field("content") {
            Some(serde_json::Value::String(text)) => text.clone(),
            Some(serde_json::Value::Array(blocks)) => blocks
                .iter()
                .filter(|block| {
                    block.get("type").and_then(serde_json::Value::as_str) == Some("text")
                })
                .filter_map(|block| block.get("text").and_then(serde_json::Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }
}

impl std::fmt::Debug for CustomMessageComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomMessageComponent")
            .finish_non_exhaustive()
    }
}

impl Component for CustomMessageComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
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
