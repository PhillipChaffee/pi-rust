//! Settings list component, ported from
//! `packages/tui/src/components/settings-list.ts` in earendil-works/pi at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#49).
//!
//! A scrollable list of settings with value cycling, optional submenu
//! delegation, an optional fuzzy-search row, and a description footer for
//! the selected row.
//!
//! Restatements against upstream:
//!
//! - `filteredItems` holds references to the same item objects upstream, so
//!   `updateValue` mutations show through both views; the port filters by
//!   index into the owned item vec, which preserves the same read path.
//! - The submenu's `done` callback captures the list upstream; the port
//!   hands the submenu factory a done callback holding the list core behind
//!   a `Weak`, so the submenu Rc never cycles back into the list and a
//!   submenu closed after its list dies cannot resurrect it (same shape as
//!   the editor's `Weak<Tui>`).
//! - The theme and callback functions — upstream `(text) => string` fields
//!   and the `(id, newValue) => void` handlers — become [`Rc`]-shared
//!   function handles.
//! - Keyboard matching owns a [`KeyParser`], upstream's module-global
//!   `matchesKey` through the keybindings registry.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::components::input::Input;
use crate::components::select_list::centered_visible_range;
use crate::fuzzy::fuzzy_filter;
use crate::keybindings::get_keybindings;
use crate::keys::KeyParser;
use crate::tui::{
    Component, TuiMouseButton, TuiMouseEvent, TuiMouseEventResult, TuiMouseEventType,
};
use crate::utils::{truncate_to_width, visible_width, wrap_text_with_ansi};

/// One configurable setting, upstream `SettingItem`.
#[derive(Clone)]
pub struct SettingItem {
    /// Unique identifier for this setting, upstream `id`.
    pub id: String,
    /// Display label (left side), upstream `label`.
    pub label: String,
    /// Optional description shown when selected, upstream `description`.
    pub description: Option<String>,
    /// Current value to display (right side), upstream `currentValue`.
    pub current_value: String,
    /// If provided, Enter/Space cycles through these values, upstream
    /// `values`.
    pub values: Option<Vec<String>>,
    /// If provided, Enter opens this submenu; it receives the current value
    /// and a done callback, upstream `submenu`. `done` accepts an optional
    /// selected value and an optional `navigateTo` id to move the cursor
    /// after close.
    pub submenu: Option<SettingsSubmenuFn>,
}

impl std::fmt::Debug for SettingItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettingItem")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("description", &self.description)
            .field("current_value", &self.current_value)
            .field("values", &self.values)
            .field("submenu", &self.submenu.is_some())
            .finish()
    }
}

/// A submenu factory, upstream `SettingItem.submenu`: receives the current
/// value and the done callback, returns the component to render while the
/// submenu is open.
pub type SettingsSubmenuFn = Rc<dyn Fn(&str, SettingsSubmenuDone) -> Rc<dyn Component>>;

/// The submenu close callback, upstream `done(selectedValue?, options?)`.
pub type SettingsSubmenuDone = Rc<dyn Fn(Option<String>, Option<SettingsSubmenuDoneOptions>)>;

/// Options for the submenu close callback, upstream `done`'s `options`.
#[derive(Debug, Clone, Default)]
pub struct SettingsSubmenuDoneOptions {
    /// Item id to select (and open) after the submenu closes, upstream
    /// `navigateTo`.
    pub navigate_to: Option<String>,
}

/// A settings text-decoration callback, upstream `(text: string) => string`.
pub type SettingsColorFn = Rc<dyn Fn(&str) -> String>;

/// The label/value decoration callback carrying the selected flag, upstream
/// `(text: string, selected: boolean) => string`.
pub type SettingsSelectedColorFn = Rc<dyn Fn(&str, bool) -> String>;

/// The settings-list theme, upstream `SettingsListTheme`.
#[derive(Clone)]
pub struct SettingsListTheme {
    /// Label column decoration, upstream `label`.
    pub label: SettingsSelectedColorFn,
    /// Value column decoration, upstream `value`.
    pub value: SettingsSelectedColorFn,
    /// Description footer decoration, upstream `description`.
    pub description: SettingsColorFn,
    /// The selected-row cursor prefix, upstream `cursor`.
    pub cursor: String,
    /// Hint-line decoration, upstream `hint`.
    pub hint: SettingsColorFn,
}

/// The change callback, upstream `onChange(id, newValue)`.
pub type SettingsChangeCallback = Rc<dyn Fn(&str, &str)>;

/// The cancel callback, upstream `onCancel`.
pub type SettingsCancelCallback = Rc<dyn Fn()>;

/// Construction options, upstream `SettingsListOptions`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SettingsListOptions {
    /// Whether the list carries a fuzzy-search row, upstream
    /// `enableSearch`.
    pub enable_search: bool,
}

/// Shared list state; `Rc`-held so the submenu `done` callback can reach it
/// through a `Weak` without a cycle.
struct SettingsCore {
    items: RefCell<Vec<SettingItem>>,
    /// Item indices matching the search query, upstream `filteredItems`
    /// (upstream holds the item objects themselves; the port holds indexes
    /// into `items`, so value mutations show through both views the same
    /// way).
    filtered_indices: RefCell<Vec<usize>>,
    theme: SettingsListTheme,
    selected_index: Cell<usize>,
    mouse_pressed_index: Cell<Option<usize>>,
    max_visible: usize,
    on_change: SettingsChangeCallback,
    on_cancel: SettingsCancelCallback,
    search_input: Option<Input>,
    search_enabled: bool,

    // Submenu state
    submenu_component: RefCell<Option<Rc<dyn Component>>>,
    submenu_item_index: Cell<Option<usize>>,
    navigate_after_close: RefCell<Option<String>>,
}

/// The settings list, upstream `SettingsList`.
pub struct SettingsList {
    core: Rc<SettingsCore>,
    parser: KeyParser,
}

impl SettingsList {
    /// Upstream's `new SettingsList(items, maxVisible, theme, onChange,
    /// onCancel)` — search disabled.
    #[must_use]
    pub fn new(
        items: Vec<SettingItem>,
        max_visible: usize,
        theme: SettingsListTheme,
        on_change: SettingsChangeCallback,
        on_cancel: SettingsCancelCallback,
    ) -> Self {
        Self::with_options(
            items,
            max_visible,
            theme,
            on_change,
            on_cancel,
            SettingsListOptions::default(),
        )
    }

    /// Upstream's `new SettingsList(items, maxVisible, theme, onChange,
    /// onCancel, options)`.
    #[must_use]
    pub fn with_options(
        items: Vec<SettingItem>,
        max_visible: usize,
        theme: SettingsListTheme,
        on_change: SettingsChangeCallback,
        on_cancel: SettingsCancelCallback,
        options: SettingsListOptions,
    ) -> Self {
        let search_enabled = options.enable_search;
        let core = Rc::new(SettingsCore {
            filtered_indices: RefCell::new((0..items.len()).collect()),
            items: RefCell::new(items),
            theme,
            selected_index: Cell::new(0),
            mouse_pressed_index: Cell::new(None),
            max_visible,
            on_change,
            on_cancel,
            search_input: search_enabled.then(Input::new),
            search_enabled,
            submenu_component: RefCell::new(None),
            submenu_item_index: Cell::new(None),
            navigate_after_close: RefCell::new(None),
        });
        Self {
            core,
            parser: KeyParser::new(),
        }
    }

    /// Update an item's `currentValue`, upstream `updateValue`.
    pub fn update_value(&self, id: &str, new_value: &str) {
        self.core.set_item_value(id, new_value);
    }

    /// Move selection to the item with the given id (no-op if not found),
    /// upstream `selectItem`. In search mode the position resolves against
    /// the filtered view.
    pub fn select_item(&self, id: &str) {
        self.core.select_item_by_id(id);
    }

    /// The mouse dispatch, upstream `handleMouse`.
    fn handle_mouse_impl(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        // Bind first: the dispatch may close the submenu through its done
        // callback, and an if-let scrutinee would hold the borrow through
        // the body.
        let submenu = self.core.submenu_component.borrow().clone();
        if let Some(submenu) = submenu {
            let result = submenu.handle_mouse(event);
            return result.map(|result| TuiMouseEventResult {
                focus: true,
                ..result
            });
        }

        if self.core.search_enabled
            && let Some(search_input) = &self.core.search_input
        {
            if event.y == 0 {
                let result = Component::handle_mouse(search_input, event);
                return result.map(|result| TuiMouseEventResult {
                    focus: true,
                    ..result
                });
            }
            if event.y == 1 {
                return None;
            }
        }

        let display_items = self.core.display_indices();
        if display_items.is_empty() {
            return None;
        }
        if event.event_type == TuiMouseEventType::Wheel
            && let Some(wheel_delta) = event.wheel_delta
            && wheel_delta != 0
        {
            let delta: i64 = if wheel_delta < 0 { -1 } else { 1 };
            let len = i64::try_from(display_items.len()).unwrap_or(i64::MAX);
            let previous_index = self.core.selected_index.get();
            let moved = i64::try_from(self.core.selected_index.get()).unwrap_or(i64::MAX) + delta;
            let next =
                usize::try_from(moved.clamp(0, len.saturating_sub(1)).max(0)).unwrap_or_default();
            self.core.selected_index.set(next);
            return Some(TuiMouseEventResult {
                handled: true,
                render: Some(self.core.selected_index.get() != previous_index),
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

        let row_offset = usize::from(self.core.search_enabled) * 2;
        let (start_index, end_index) = self.core.visible_range(display_items.len());
        let item_position = i64::try_from(start_index).unwrap_or(i64::MAX) + i64::from(event.y)
            - i64::try_from(row_offset).unwrap_or(i64::MAX);
        if item_position < i64::try_from(start_index).unwrap_or(i64::MAX)
            || item_position >= i64::try_from(end_index).unwrap_or(i64::MAX)
        {
            return None;
        }
        let item_index = usize::try_from(item_position).unwrap_or_default();
        if event.event_type == TuiMouseEventType::Press {
            self.core.mouse_pressed_index.set(Some(item_index));
            self.core.selected_index.set(item_index);
            return Some(TuiMouseEventResult {
                handled: true,
                focus: true,
                ..TuiMouseEventResult::default()
            });
        }
        if event.event_type == TuiMouseEventType::Click {
            let clicked = self.core.mouse_pressed_index.get().unwrap_or(item_index);
            self.core.mouse_pressed_index.set(None);
            self.core.selected_index.set(clicked);
            SettingsCore::activate_item(&self.core);
            return Some(TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            });
        }
        None
    }

    /// The raw input dispatch, upstream `handleInput`. While a submenu is
    /// open every key delegates to it — its Escape path calls `done()`,
    /// which closes it.
    fn handle_input_impl(&self, data: &str) {
        // Bind first: the dispatch may close the submenu through its done
        // callback, and an if-let scrutinee would hold the borrow.
        let submenu = self.core.submenu_component.borrow().clone();
        if let Some(submenu) = submenu {
            submenu.handle_input(data);
            return;
        }

        // Main list input handling
        let display_items = self.core.display_indices();
        if get_keybindings().matches(&self.parser, data, "tui.select.up") {
            if display_items.is_empty() {
                return;
            }
            let len = display_items.len();
            let selected = self.core.selected_index.get();
            self.core
                .selected_index
                .set(if selected == 0 { len - 1 } else { selected - 1 });
        } else if get_keybindings().matches(&self.parser, data, "tui.select.down") {
            if display_items.is_empty() {
                return;
            }
            let len = display_items.len();
            let selected = self.core.selected_index.get();
            self.core
                .selected_index
                .set(if selected == len.saturating_sub(1) {
                    0
                } else {
                    selected + 1
                });
        } else if get_keybindings().matches(&self.parser, data, "tui.select.confirm")
            || (data == " "
                && (!self.core.search_enabled
                    || self
                        .core
                        .search_input
                        .as_ref()
                        .is_some_and(|input| input.get_value().is_empty())))
        {
            SettingsCore::activate_item(&self.core);
        } else if get_keybindings().matches(&self.parser, data, "tui.select.cancel") {
            (self.core.on_cancel)();
        } else if self.core.search_enabled
            && let Some(search_input) = &self.core.search_input
        {
            search_input.handle_input(data);
            let query = search_input.get_value();
            self.core.apply_filter(&query);
        }
    }

    /// The main-list render, upstream `renderMainList`.
    fn render_main_list(&self, width: usize) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();

        if self.core.search_enabled
            && let Some(search_input) = &self.core.search_input
        {
            lines.extend(Component::render(search_input, width));
            lines.push(String::new());
        }

        if self.core.items.borrow().is_empty() {
            lines.push((self.core.theme.hint)("  No settings available"));
            if self.core.search_enabled {
                self.add_hint_line(&mut lines, width);
            }
            return lines;
        }

        let display_items = self.core.display_indices();
        if display_items.is_empty() {
            lines.push(truncate_to_width(
                &(self.core.theme.hint)("  No matching settings"),
                width,
                "",
                false,
            ));
            self.add_hint_line(&mut lines, width);
            return lines;
        }

        // Calculate visible range with scrolling
        let (start_index, end_index) = self.core.visible_range(display_items.len());

        // Calculate max label width for alignment
        let max_label_width = 36.min(
            self.core
                .items
                .borrow()
                .iter()
                .map(|item| visible_width(&item.label))
                .max()
                .unwrap_or(0),
        );

        // Render visible items
        for position in start_index..end_index {
            let Some(&item_index) = display_items.get(position) else {
                continue;
            };
            let item = self.core.items.borrow()[item_index].clone();
            let is_selected = position == self.core.selected_index.get();
            let prefix = if is_selected {
                self.core.theme.cursor.clone()
            } else {
                "  ".to_string()
            };
            let prefix_width = visible_width(&prefix);

            // Pad label to align values
            let label_padded = format!(
                "{}{}",
                item.label,
                " ".repeat(max_label_width.saturating_sub(visible_width(&item.label)))
            );
            let label_text = (self.core.theme.label)(&label_padded, is_selected);

            // Calculate space for value
            let separator = "  ";
            let used_width = prefix_width + max_label_width + visible_width(separator);
            let value_max_width = width.saturating_sub(used_width + 2);

            let value_text = (self.core.theme.value)(
                &truncate_to_width(&item.current_value, value_max_width, "", false),
                is_selected,
            );

            let line = format!("{prefix}{label_text}{separator}{value_text}");
            lines.push(truncate_to_width(&line, width, "", false));
        }

        // Add scroll indicator if needed
        if start_index > 0 || end_index < display_items.len() {
            let scroll_text = format!(
                "  ({}/{})",
                self.core.selected_index.get() + 1,
                display_items.len()
            );
            lines.push((self.core.theme.hint)(&truncate_to_width(
                &scroll_text,
                width.saturating_sub(2),
                "",
                false,
            )));
        }

        // Add description for selected item
        let selected_item = self
            .core
            .display_items()
            .get(self.core.selected_index.get())
            .cloned();
        if let Some(description) = selected_item.and_then(|item| item.description) {
            lines.push(String::new());
            let wrapped = wrap_text_with_ansi(&description, width.saturating_sub(4));
            for line in wrapped {
                lines.push((self.core.theme.description)(&format!("  {line}")));
            }
        }

        // Add hint
        self.add_hint_line(&mut lines, width);

        lines
    }

    /// The hint footer, upstream `addHintLine`.
    fn add_hint_line(&self, lines: &mut Vec<String>, width: usize) {
        lines.push(String::new());
        let hint = if self.core.search_enabled {
            "  Type to search · Enter/Space to change · Esc to cancel"
        } else {
            "  Enter/Space to change · Esc to cancel"
        };
        lines.push(truncate_to_width(
            &(self.core.theme.hint)(hint),
            width,
            "",
            false,
        ));
    }
}

impl SettingsCore {
    /// The items of the active view, upstream `getDisplayItems`: the
    /// filtered set in search mode, all items otherwise.
    fn display_indices(&self) -> Vec<usize> {
        if self.search_enabled {
            self.filtered_indices.borrow().clone()
        } else {
            (0..self.items.borrow().len()).collect()
        }
    }

    /// Convenience for single-item reads, upstream `displayItems[i]`.
    fn display_items(&self) -> Vec<SettingItem> {
        self.display_indices()
            .into_iter()
            .filter_map(|index| self.items.borrow().get(index).cloned())
            .collect()
    }

    fn visible_range(&self, len: usize) -> (usize, usize) {
        centered_visible_range(self.selected_index.get(), self.max_visible, len)
    }

    /// Mutate one item's `currentValue` by id, upstream's direct
    /// `item.currentValue = …` writes.
    fn set_item_value(&self, id: &str, new_value: &str) {
        if let Some(item) = self
            .items
            .borrow_mut()
            .iter_mut()
            .find(|item| item.id == id)
        {
            item.current_value = new_value.to_string();
        }
    }

    /// Fuzzy-filter the item view, upstream `applyFilter`.
    fn apply_filter(&self, query: &str) {
        let indexed: Vec<(usize, SettingItem)> =
            self.items.borrow().iter().cloned().enumerate().collect();
        let matched = fuzzy_filter(&indexed, query, |(_, item)| item.label.clone());
        *self.filtered_indices.borrow_mut() = matched.iter().map(|(index, _)| *index).collect();
        self.selected_index.set(0);
    }

    /// Close the open submenu and restore or retarget the selection,
    /// upstream `closeSubmenu`.
    fn close_submenu(core_rc: &Rc<Self>) {
        let this = core_rc;
        *this.submenu_component.borrow_mut() = None;
        let id = this.navigate_after_close.borrow_mut().take();
        if let Some(id) = id {
            this.submenu_item_index.set(None);
            this.select_item_by_id(&id);
            // Open the target item's submenu automatically
            Self::activate_item(core_rc);
        } else if let Some(index) = this.submenu_item_index.take() {
            // Restore selection to the item that opened the submenu
            this.selected_index.set(index);
        }
    }

    /// `selectItem(id)` (no-op if not found) against the active view.
    fn select_item_by_id(&self, id: &str) {
        let display = self.display_indices();
        let position = display
            .iter()
            .position(|index| self.items.borrow()[*index].id == id);
        if let Some(position) = position {
            self.selected_index.set(position);
        }
    }

    /// Open the selected item's submenu or cycle its values, upstream
    /// `activateItem`.
    fn activate_item(core_rc: &Rc<Self>) {
        let this = core_rc;
        let Some(item) = this.display_items().get(this.selected_index.get()).cloned() else {
            return;
        };

        if let Some(submenu) = &item.submenu {
            // Open submenu, passing current value so it can pre-select
            // correctly
            this.submenu_item_index.set(Some(this.selected_index.get()));
            let done = Self::make_done(core_rc, &item.id);
            *this.submenu_component.borrow_mut() = Some(submenu(&item.current_value, done));
        } else if item
            .values
            .as_ref()
            .is_some_and(|values| !values.is_empty())
        {
            // Cycle through values
            #[expect(
                clippy::expect_used,
                reason = "the non-empty check is the branch condition; the unwrap cannot miss"
            )]
            let values = item.values.as_ref().expect("checked non-empty above");
            let next_index = values
                .iter()
                .position(|value| *value == item.current_value)
                .map_or(0, |index| (index + 1) % values.len());
            let new_value = values[next_index].clone();
            this.set_item_value(&item.id, &new_value);
            (this.on_change)(&item.id, &new_value);
        }
    }

    /// Build the submenu close callback, upstream's `done` closure inside
    /// `activateItem`.
    fn make_done(core_rc: &Rc<Self>, item_id: &str) -> SettingsSubmenuDone {
        let core = Rc::downgrade(core_rc);
        let item_id = item_id.to_string();
        Rc::new(move |selected_value, options| {
            let Some(core) = core.upgrade() else {
                return;
            };
            if let Some(selected_value) = selected_value {
                core.set_item_value(&item_id, &selected_value);
                (core.on_change)(&item_id, &selected_value);
            }
            if let Some(options) = options
                && let Some(navigate_to) = options.navigate_to
            {
                *core.navigate_after_close.borrow_mut() = Some(navigate_to);
            }
            Self::close_submenu(&core);
        })
    }
}

impl Component for SettingsList {
    fn render(&self, width: usize) -> Vec<String> {
        // If submenu is active, render it instead
        let submenu = self.core.submenu_component.borrow().clone();
        if let Some(submenu) = submenu {
            return submenu.render(width);
        }

        self.render_main_list(width)
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

    fn invalidate(&self) {
        let submenu = self.core.submenu_component.borrow().clone();
        if let Some(submenu) = submenu {
            submenu.invalidate();
        }
    }
}

impl std::fmt::Debug for SettingsListTheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettingsListTheme").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for SettingsList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettingsList")
            .field("max_visible", &self.core.max_visible)
            .field("selected_index", &self.core.selected_index.get())
            .field("search_enabled", &self.core.search_enabled)
            .finish_non_exhaustive()
    }
}
