//! Selection list component, ported from
//! `packages/tui/src/components/select-list.ts` in earendil-works/pi at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#49).
//!
//! A filterable, scrollable list with a two-column layout (primary value
//! plus optional single-line description) and arrow-key wrap-around
//! selection. The editor uses it as the autocomplete dropdown; standalone
//! consumers set the selection callbacks directly.
//!
//! Restatements against upstream:
//!
//! - The theme and layout callbacks — upstream `(text: string) => string`
//!   fields — become [`Rc`]-shared function handles
//!   ([`SelectListColorFn`], [`SelectListTruncatePrimaryFn`]), the same
//!   shape the editor's border color already uses.
//! - `onSelect`/`onCancel`/`onSelectionChange` are consumer-assigned
//!   callback fields; they live behind `RefCell`s so they can be assigned
//!   after construction, upstream's public field assignment.
//! - Keyboard matching owns a [`KeyParser`]: upstream calls the
//!   module-global `matchesKey` through the keybindings registry, the port
//!   holds a parser instance.
//! - `localeCompare` restates to lexicographic byte comparison; the sort is
//!   stable, matching JS `Array.prototype.sort`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::keybindings::get_keybindings;
use crate::keys::KeyParser;
use crate::tui::{
    Component, TuiMouseButton, TuiMouseEvent, TuiMouseEventResult, TuiMouseEventType,
};
use crate::utils::{truncate_to_width, visible_width};

const DEFAULT_PRIMARY_COLUMN_WIDTH: usize = 32;
const PRIMARY_COLUMN_GAP: usize = 2;
const MIN_DESCRIPTION_WIDTH: usize = 10;

/// Collapse every run of carriage returns / newlines into one space and
/// trim, upstream `normalizeToSingleLine`.
fn normalize_to_single_line(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut in_break = false;
    for character in text.chars() {
        if character == '\r' || character == '\n' {
            if !in_break {
                normalized.push(' ');
                in_break = true;
            }
        } else {
            normalized.push(character);
            in_break = false;
        }
    }
    normalized.trim().to_string()
}

/// One selectable row, upstream `SelectItem`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelectItem {
    /// Value applied on selection, upstream `value`.
    pub value: String,
    /// Display label; falls back to `value` when empty, upstream `label`.
    pub label: String,
    /// Optional description rendered in the second column, upstream
    /// `description`.
    pub description: Option<String>,
}

/// A text-decoration callback, upstream `(text: string) => string`.
pub type SelectListColorFn = Rc<dyn Fn(&str) -> String>;

/// The select-list theme, upstream `SelectListTheme`.
#[derive(Clone)]
pub struct SelectListTheme {
    /// Prefix decoration for the selected row, upstream `selectedPrefix`.
    pub selected_prefix: SelectListColorFn,
    /// Full-line decoration for the selected row, upstream `selectedText`.
    pub selected_text: SelectListColorFn,
    /// Description-column decoration, upstream `description`.
    pub description: SelectListColorFn,
    /// Scroll-indicator decoration, upstream `scrollInfo`.
    pub scroll_info: SelectListColorFn,
    /// No-match message decoration, upstream `noMatch`.
    pub no_match: SelectListColorFn,
}

/// Context handed to the primary-column truncation override, upstream
/// `SelectListTruncatePrimaryContext`.
#[derive(Debug)]
pub struct SelectListTruncatePrimaryContext<'a> {
    /// The display value about to be truncated, upstream `text`.
    pub text: &'a str,
    /// Maximum visible width for the value, upstream `maxWidth`.
    pub max_width: usize,
    /// The column budget the value sits in, upstream `columnWidth`.
    pub column_width: usize,
    /// The row being rendered, upstream `item`.
    pub item: &'a SelectItem,
    /// Whether the row is the selected one, upstream `isSelected`.
    pub is_selected: bool,
}

/// Primary-column truncation override, upstream
/// `SelectListLayoutOptions.truncatePrimary`. The result is re-truncated to
/// `max_width`, so an override cannot overflow the column.
pub type SelectListTruncatePrimaryFn = Rc<dyn Fn(&SelectListTruncatePrimaryContext<'_>) -> String>;

/// Primary-column layout tuning, upstream `SelectListLayoutOptions`.
#[derive(Clone, Default)]
pub struct SelectListLayoutOptions {
    /// Lower bound for the primary column, upstream
    /// `minPrimaryColumnWidth`. When only one bound is configured it fixes
    /// both ends at that value.
    pub min_primary_column_width: Option<usize>,
    /// Upper bound for the primary column, upstream
    /// `maxPrimaryColumnWidth`.
    pub max_primary_column_width: Option<usize>,
    /// Truncation override, upstream `truncatePrimary`.
    pub truncate_primary: Option<SelectListTruncatePrimaryFn>,
}

/// The selection callback, upstream `onSelect` / `onSelectionChange`.
pub type SelectListCallback = Rc<dyn Fn(&SelectItem)>;

/// The cancel callback, upstream `onCancel`.
pub type SelectListCancelCallback = Rc<dyn Fn()>;

/// The selectable list, upstream `SelectList`.
pub struct SelectList {
    items: RefCell<Vec<SelectItem>>,
    filtered_items: RefCell<Vec<SelectItem>>,
    selected_index: Cell<usize>,
    mouse_pressed_index: Cell<Option<usize>>,
    max_visible: usize,
    theme: SelectListTheme,
    layout: SelectListLayoutOptions,

    /// Called when a row is confirmed, upstream `onSelect`.
    pub on_select: RefCell<Option<SelectListCallback>>,
    /// Called when the list is dismissed, upstream `onCancel`.
    pub on_cancel: RefCell<Option<SelectListCancelCallback>>,
    /// Called whenever the selection changes, upstream
    /// `onSelectionChange`.
    pub on_selection_change: RefCell<Option<SelectListCallback>>,

    parser: KeyParser,
}

struct VisibleRange {
    start_index: usize,
    end_index: usize,
}

struct ColumnBounds {
    min: usize,
    max: usize,
}

/// The visible scroll window centered on the selection, upstream
/// `getVisibleRange` (shared by the select list and the settings list).
pub(crate) fn centered_visible_range(
    selected_index: usize,
    max_visible: usize,
    len: usize,
) -> (usize, usize) {
    let selected = i64::try_from(selected_index).unwrap_or(i64::MAX);
    let max_visible = i64::try_from(max_visible).unwrap_or(i64::MAX);
    let len = i64::try_from(len).unwrap_or(i64::MAX);
    let start = (selected - max_visible / 2)
        .clamp(0, (len - max_visible).max(0))
        .max(0);
    let start_index = usize::try_from(start).unwrap_or_default();
    let end_index = (start + max_visible).min(len);
    (start_index, usize::try_from(end_index).unwrap_or_default())
}

impl SelectList {
    /// Upstream's `new SelectList(items, maxVisible, theme)` — default
    /// layout options.
    #[must_use]
    pub fn new(items: Vec<SelectItem>, max_visible: usize, theme: SelectListTheme) -> Self {
        Self::with_layout(
            items,
            max_visible,
            theme,
            SelectListLayoutOptions::default(),
        )
    }

    /// Upstream's `new SelectList(items, maxVisible, theme, layout)`.
    #[must_use]
    pub fn with_layout(
        items: Vec<SelectItem>,
        max_visible: usize,
        theme: SelectListTheme,
        layout: SelectListLayoutOptions,
    ) -> Self {
        Self {
            items: RefCell::new(items.clone()),
            filtered_items: RefCell::new(items),
            selected_index: Cell::new(0),
            mouse_pressed_index: Cell::new(None),
            max_visible,
            theme,
            layout,
            on_select: RefCell::new(None),
            on_cancel: RefCell::new(None),
            on_selection_change: RefCell::new(None),
            parser: KeyParser::new(),
        }
    }

    /// Keep only rows whose value starts with the filter
    /// (case-insensitive) and reset the selection, upstream `setFilter`.
    /// The filter runs over the full item list every time, so repeated
    /// filters compose like upstream's fresh array.
    pub fn set_filter(&self, filter: &str) {
        let lowered = filter.to_lowercase();
        *self.filtered_items.borrow_mut() = self
            .items
            .borrow()
            .iter()
            .filter(|item| item.value.to_lowercase().starts_with(&lowered))
            .cloned()
            .collect();
        // Reset selection when filter changes
        self.selected_index.set(0);
    }

    /// Select a row, clamped into the filtered range, upstream
    /// `setSelectedIndex`. An empty list clamps to zero.
    pub fn set_selected_index(&self, index: usize) {
        let len = self.filtered_items.borrow().len();
        self.selected_index.set(index.min(len.saturating_sub(1)));
    }

    /// The selected row, upstream `getSelectedItem`.
    #[must_use]
    pub fn get_selected_item(&self) -> Option<SelectItem> {
        self.filtered_items
            .borrow()
            .get(self.selected_index.get())
            .cloned()
    }

    /// The mouse dispatch, upstream `handleMouse`. Hover never changes the
    /// selection — the visible range is centered on it.
    fn handle_mouse_impl(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        if self.filtered_items.borrow().is_empty() {
            return None;
        }
        if event.event_type == TuiMouseEventType::Wheel
            && let Some(wheel_delta) = event.wheel_delta
            && wheel_delta != 0
        {
            let delta: i64 = if wheel_delta < 0 { -1 } else { 1 };
            let len = self.filtered_items.borrow().len();
            let previous_index = self.selected_index.get();
            let moved = i64::try_from(self.selected_index.get()).unwrap_or(i64::MAX) + delta;
            let max = i64::try_from(len.saturating_sub(1)).unwrap_or(i64::MAX);
            let next = usize::try_from(moved.clamp(0, max).max(0)).unwrap_or_default();
            self.selected_index.set(next);
            let changed = self.selected_index.get() != previous_index;
            if changed {
                self.notify_selection_change();
            }
            return Some(TuiMouseEventResult {
                handled: true,
                render: Some(changed),
                ..TuiMouseEventResult::default()
            });
        }
        // Hover must not change selection: the visible range is centered on
        // it.
        if event.button != TuiMouseButton::Left
            || (event.event_type != TuiMouseEventType::Press
                && event.event_type != TuiMouseEventType::Click)
        {
            return None;
        }
        let visible_range = self.visible_range();
        let item_index = visible_range.start_index + usize::from(event.y);
        if item_index < visible_range.start_index || item_index >= visible_range.end_index {
            return None;
        }

        if event.event_type == TuiMouseEventType::Press {
            self.mouse_pressed_index.set(Some(item_index));
            if self.selected_index.get() != item_index {
                self.selected_index.set(item_index);
                self.notify_selection_change();
            }
            return Some(TuiMouseEventResult {
                handled: true,
                focus: true,
                ..TuiMouseEventResult::default()
            });
        }
        if event.event_type == TuiMouseEventType::Click {
            let clicked_index = self.mouse_pressed_index.get().unwrap_or(item_index);
            self.mouse_pressed_index.set(None);
            let changed = self.selected_index.get() != clicked_index;
            self.selected_index.set(clicked_index);
            if changed {
                self.notify_selection_change();
            }
            let selected_item = self.get_selected_item();
            if let Some(selected_item) = selected_item {
                let callback = self.on_select.borrow().clone();
                if let Some(on_select) = callback {
                    on_select(&selected_item);
                }
            }
            return Some(TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            });
        }
        None
    }

    /// The raw input dispatch, upstream `handleInput`: arrow keys wrap at
    /// the edges, Enter confirms, Escape/Ctrl+C cancels.
    fn handle_input_impl(&self, data: &str) {
        // Up arrow - wrap to bottom when at top
        if get_keybindings().matches(&self.parser, data, "tui.select.up") {
            let len = self.filtered_items.borrow().len();
            self.selected_index.set(if self.selected_index.get() == 0 {
                len.saturating_sub(1)
            } else {
                self.selected_index.get() - 1
            });
            self.notify_selection_change();
        }
        // Down arrow - wrap to top when at bottom
        else if get_keybindings().matches(&self.parser, data, "tui.select.down") {
            let len = self.filtered_items.borrow().len();
            self.selected_index
                .set(if self.selected_index.get() == len.saturating_sub(1) {
                    0
                } else {
                    self.selected_index.get() + 1
                });
            self.notify_selection_change();
        }
        // Enter
        else if get_keybindings().matches(&self.parser, data, "tui.select.confirm") {
            let selected_item = self.get_selected_item();
            if let Some(selected_item) = selected_item {
                let callback = self.on_select.borrow().clone();
                if let Some(on_select) = callback {
                    on_select(&selected_item);
                }
            }
        }
        // Escape or Ctrl+C
        else if get_keybindings().matches(&self.parser, data, "tui.select.cancel") {
            let callback = self.on_cancel.borrow().clone();
            if let Some(on_cancel) = callback {
                on_cancel();
            }
        }
    }

    /// The visible scroll window centered on the selection, upstream
    /// `getVisibleRange`.
    fn visible_range(&self) -> VisibleRange {
        let (start_index, end_index) = centered_visible_range(
            self.selected_index.get(),
            self.max_visible,
            self.filtered_items.borrow().len(),
        );
        VisibleRange {
            start_index,
            end_index,
        }
    }

    /// One row, upstream `renderItem`: prefix arrow, the primary column,
    /// and the description column when the terminal is wide enough.
    fn render_item(
        &self,
        item: &SelectItem,
        is_selected: bool,
        width: usize,
        description_single_line: Option<&str>,
        primary_column_width: usize,
    ) -> String {
        let prefix = if is_selected { "→ " } else { "  " };
        let prefix_width = visible_width(prefix);

        if let Some(description_single_line) = description_single_line
            && width > 40
        {
            let effective_primary_column_width =
                1.max(primary_column_width.min(width.saturating_sub(prefix_width + 4)));
            let max_primary_width =
                1.max(effective_primary_column_width.saturating_sub(PRIMARY_COLUMN_GAP));
            let truncated_value = self.truncate_primary(
                item,
                is_selected,
                max_primary_width,
                effective_primary_column_width,
            );
            let truncated_value_width = visible_width(&truncated_value);
            let spacing = " ".repeat(
                1.max(effective_primary_column_width.saturating_sub(truncated_value_width)),
            );
            let description_start = prefix_width + truncated_value_width + spacing.len();
            // -2 for safety
            let remaining_width = width.saturating_sub(description_start + 2);

            if remaining_width > MIN_DESCRIPTION_WIDTH {
                let truncated_desc =
                    truncate_to_width(description_single_line, remaining_width, "", false);
                if is_selected {
                    let line = format!("{prefix}{truncated_value}{spacing}{truncated_desc}");
                    return (self.theme.selected_text)(&line);
                }

                let desc_text = (self.theme.description)(&format!("{spacing}{truncated_desc}"));
                return format!("{prefix}{truncated_value}{desc_text}");
            }
        }

        let max_width = width.saturating_sub(prefix_width + 2);
        let truncated_value = self.truncate_primary(item, is_selected, max_width, max_width);
        if is_selected {
            let line = format!("{prefix}{truncated_value}");
            return (self.theme.selected_text)(&line);
        }

        format!("{prefix}{truncated_value}")
    }

    /// The primary column budget, upstream `getPrimaryColumnWidth`: the
    /// widest display value, clamped into the configured bounds.
    fn primary_column_width(&self) -> usize {
        let bounds = self.primary_column_bounds();
        let widest = self
            .filtered_items
            .borrow()
            .iter()
            .map(|item| visible_width(self.display_value(item)) + PRIMARY_COLUMN_GAP)
            .max()
            .unwrap_or(0);
        widest.clamp(bounds.min, bounds.max)
    }

    /// The primary-column bounds, upstream `getPrimaryColumnBounds`.
    fn primary_column_bounds(&self) -> ColumnBounds {
        let raw_min = self
            .layout
            .min_primary_column_width
            .or(self.layout.max_primary_column_width)
            .unwrap_or(DEFAULT_PRIMARY_COLUMN_WIDTH);
        let raw_max = self
            .layout
            .max_primary_column_width
            .or(self.layout.min_primary_column_width)
            .unwrap_or(DEFAULT_PRIMARY_COLUMN_WIDTH);
        ColumnBounds {
            min: 1.max(raw_min.min(raw_max)),
            max: 1.max(raw_min.max(raw_max)),
        }
    }

    /// The primary column value, upstream `truncatePrimary`: the consumer
    /// override when configured, then always re-truncated to `max_width`.
    fn truncate_primary(
        &self,
        item: &SelectItem,
        is_selected: bool,
        max_width: usize,
        column_width: usize,
    ) -> String {
        let display_value = self.display_value(item);
        #[expect(
            clippy::option_if_let_else,
            reason = "mirrors upstream's truncatePrimary override branch; map_or_else would bury the override contract in a closure"
        )]
        let truncated_value = if let Some(truncate_primary) = &self.layout.truncate_primary {
            truncate_primary(&SelectListTruncatePrimaryContext {
                text: display_value,
                max_width,
                column_width,
                item,
                is_selected,
            })
        } else {
            truncate_to_width(display_value, max_width, "", false)
        };
        truncate_to_width(&truncated_value, max_width, "", false)
    }

    /// The label when non-empty, else the value, upstream `getDisplayValue`.
    #[expect(
        clippy::unused_self,
        reason = "upstream's getDisplayValue is a list method; the port keeps the shape"
    )]
    fn display_value<'a>(&self, item: &'a SelectItem) -> &'a str {
        if item.label.is_empty() {
            &item.value
        } else {
            &item.label
        }
    }

    /// Fire `onSelectionChange` for the current selection, upstream
    /// `notifySelectionChange`.
    fn notify_selection_change(&self) {
        let selected_item = self.get_selected_item();
        if let Some(selected_item) = selected_item {
            let callback = self.on_selection_change.borrow().clone();
            if let Some(on_selection_change) = callback {
                on_selection_change(&selected_item);
            }
        }
    }
}

impl Component for SelectList {
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();

        // If no items match filter, show message
        if self.filtered_items.borrow().is_empty() {
            lines.push((self.theme.no_match)("  No matching commands"));
            return lines;
        }

        let primary_column_width = self.primary_column_width();

        // Calculate visible range with scrolling
        let visible_range = self.visible_range();

        // Render visible items
        for index in visible_range.start_index..visible_range.end_index {
            let Some(item) = self.filtered_items.borrow().get(index).cloned() else {
                continue;
            };
            let is_selected = index == self.selected_index.get();
            let description_single_line = item
                .description
                .as_ref()
                .map(|description| normalize_to_single_line(description));
            lines.push(self.render_item(
                &item,
                is_selected,
                width,
                description_single_line.as_deref(),
                primary_column_width,
            ));
        }

        // Add scroll indicators if needed
        if visible_range.start_index > 0
            || visible_range.end_index < self.filtered_items.borrow().len()
        {
            let scroll_text = format!(
                "  ({}/{})",
                self.selected_index.get() + 1,
                self.filtered_items.borrow().len()
            );
            // Truncate if too long for terminal
            lines.push((self.theme.scroll_info)(&truncate_to_width(
                &scroll_text,
                width.saturating_sub(2),
                "",
                false,
            )));
        }

        lines
    }

    fn handle_input(&self, data: &str) {
        self.handle_input_impl(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.handle_mouse_impl(event)
    }
}

impl std::fmt::Debug for SelectListLayoutOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectListLayoutOptions")
            .field("min_primary_column_width", &self.min_primary_column_width)
            .field("max_primary_column_width", &self.max_primary_column_width)
            .field("truncate_primary", &self.truncate_primary.is_some())
            .finish()
    }
}

impl std::fmt::Debug for SelectListTheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectListTheme").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for SelectList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectList")
            .field("max_visible", &self.max_visible)
            .field("selected_index", &self.selected_index.get())
            .field("filtered_items", &self.filtered_items.borrow().len())
            .finish_non_exhaustive()
    }
}
