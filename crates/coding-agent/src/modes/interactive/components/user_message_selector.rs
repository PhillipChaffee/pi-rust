//! The fork-from-message picker, upstream's
//! `src/modes/interactive/components/user-message-selector.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#210).
//!
//! The list takes the session's user-message snapshot and reports the chosen
//! entry id; forking and persistence stay with the interactive frame.
//!
//! Restatements: upstream's never-read optional `timestamp` field is absent,
//! and an empty picker ignores navigation (JavaScript would move the index to
//! `-1`) — the host cancels before mounting, upstream's 100 ms auto-cancel.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container};
use pi_tui::utils::truncate_to_width;

use super::dynamic_border::DynamicBorder;
use crate::modes::interactive::theme::theme::{bold, theme};

/// Rows visible before the list scrolls, upstream's `maxVisible`.
const MAX_VISIBLE: usize = 10;

/// One fork point, upstream's `UserMessageItem`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserMessageItem {
    /// The session entry id, upstream's `id`.
    pub id: String,
    /// The message text, upstream's `text`.
    pub text: String,
}

/// Selection callback, upstream's `onSelect`.
pub type UserMessageSelectCallback = Rc<dyn Fn(&str)>;

/// Cancellation callback, upstream's `onCancel`.
pub type UserMessageCancelCallback = Rc<dyn Fn()>;

/// The scrollable, keyboard-navigable message list.
pub struct UserMessageList {
    messages: Vec<UserMessageItem>,
    selected_index: Cell<usize>,
    /// Set by the host, upstream's public `onSelect`.
    pub on_select: RefCell<Option<UserMessageSelectCallback>>,
    /// Set by the host, upstream's public `onCancel`.
    pub on_cancel: RefCell<Option<UserMessageCancelCallback>>,
    parser: KeyParser,
}

impl std::fmt::Debug for UserMessageList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserMessageList")
            .field("messages", &self.messages.len())
            .field("selected_index", &self.selected_index.get())
            .finish_non_exhaustive()
    }
}

impl UserMessageList {
    /// Preselects the matching entry id, otherwise the newest message.
    #[must_use]
    pub fn new(messages: Vec<UserMessageItem>, initial_selected_id: Option<&str>) -> Self {
        let selected_index = initial_selected_id
            .and_then(|id| messages.iter().position(|message| message.id == id))
            .unwrap_or_else(|| messages.len().saturating_sub(1));
        Self {
            messages,
            selected_index: Cell::new(selected_index),
            on_select: RefCell::new(None),
            on_cancel: RefCell::new(None),
            parser: KeyParser::new(),
        }
    }

    /// Whether there is nothing to fork from.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    fn handle_key(&self, action: &str, data: &str) -> bool {
        get_keybindings().matches(&self.parser, data, action)
    }
}

impl Component for UserMessageList {
    fn render(&self, width: usize) -> Vec<String> {
        if self.messages.is_empty() {
            return vec![theme().fg("muted", "  No user messages found")];
        }

        let total = self.messages.len();
        let selected = self.selected_index.get();
        let start = selected
            .saturating_sub(MAX_VISIBLE / 2)
            .min(total.saturating_sub(MAX_VISIBLE));
        let end = (start + MAX_VISIBLE).min(total);

        let mut lines = Vec::new();
        for index in start..end {
            let message = &self.messages[index];
            let is_selected = index == selected;
            let normalized = message.text.replace('\n', " ").trim().to_string();
            let cursor = if is_selected {
                theme().fg("accent", "› ")
            } else {
                "  ".to_string()
            };
            let truncated = truncate_to_width(&normalized, width.saturating_sub(2), "...", false);
            lines.push(if is_selected {
                format!("{cursor}{}", bold(&truncated))
            } else {
                format!("{cursor}{truncated}")
            });
            lines.push(theme().fg("muted", &format!("  Message {} of {total}", index + 1)));
            lines.push(String::new());
        }

        if start > 0 || end < total {
            lines.push(theme().fg("muted", &format!("  ({}/{total})", selected + 1)));
        }

        lines
    }

    fn handle_input(&self, data: &str) {
        if self.messages.is_empty() {
            if self.handle_key("tui.select.cancel", data)
                && let Some(on_cancel) = self.on_cancel.borrow().as_ref()
            {
                on_cancel();
            }
            return;
        }

        if self.handle_key("tui.select.up", data) {
            let index = self.selected_index.get();
            self.selected_index.set(if index == 0 {
                self.messages.len() - 1
            } else {
                index - 1
            });
        } else if self.handle_key("tui.select.down", data) {
            let index = self.selected_index.get();
            self.selected_index
                .set(if index == self.messages.len() - 1 {
                    0
                } else {
                    index + 1
                });
        } else if self.handle_key("tui.select.confirm", data) {
            let selected = self
                .messages
                .get(self.selected_index.get())
                .map(|message| message.id.clone());
            if let (Some(id), Some(on_select)) = (selected, self.on_select.borrow().as_ref()) {
                on_select(&id);
            }
        } else if self.handle_key("tui.select.cancel", data)
            && let Some(on_cancel) = self.on_cancel.borrow().as_ref()
        {
            on_cancel();
        }
    }

    fn wants_input(&self) -> bool {
        true
    }
}

/// The bordered fork-from-message dialog, upstream's
/// `UserMessageSelectorComponent`.
pub struct UserMessageSelectorComponent {
    container: Container,
    message_list: Rc<UserMessageList>,
}

impl UserMessageSelectorComponent {
    /// Builds the dialog over the session's user-message snapshot.
    #[must_use]
    pub fn new(
        messages: Vec<UserMessageItem>,
        on_select: UserMessageSelectCallback,
        on_cancel: UserMessageCancelCallback,
        initial_selected_id: Option<&str>,
    ) -> Self {
        let message_list = Rc::new(UserMessageList::new(messages, initial_selected_id));
        *message_list.on_select.borrow_mut() = Some(on_select);
        *message_list.on_cancel.borrow_mut() = Some(on_cancel);

        let container = Container::default();
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding(bold("Fork from Message"), 1, 0)));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg(
                "muted",
                "Select a user message to copy the active path up to that point into a new session",
            ),
            1,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));
        let list_component: Rc<dyn Component> = message_list.clone();
        container.add_child(list_component);
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));

        Self {
            container,
            message_list,
        }
    }

    /// The focusable list inside the dialog, upstream's `getMessageList`.
    #[must_use]
    pub fn get_message_list(&self) -> Rc<UserMessageList> {
        Rc::clone(&self.message_list)
    }

    /// Whether the host must skip the dialog, upstream's empty auto-cancel.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.message_list.is_empty()
    }
}

impl Component for UserMessageSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }
}

impl std::fmt::Debug for UserMessageSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserMessageSelectorComponent")
            .finish_non_exhaustive()
    }
}
