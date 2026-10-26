//! The ordinary model picker at upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#200).
//!
//! The component renders the catalog snapshot its host supplies. Catalog
//! refresh and cancellation stay with the interactive frame, which updates
//! the picker through [`ModelSelectorComponent::update_models`] and
//! [`ModelSelectorComponent::set_refresh_status`].

use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

use pi_ai::types::Model;
use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::fuzzy::fuzzy_filter;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, Focusable, TuiMouseEvent, TuiMouseEventResult};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::{key_display_text, key_hint};
use crate::modes::interactive::model_search::get_model_selector_search_text;
use crate::modes::interactive::theme::theme::theme;

/// The rows shown before scrolling, upstream's `maxVisible`.
const MAX_VISIBLE: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelScope {
    All,
    Scoped,
}

/// Catalog snapshot and session state the picker renders.
#[derive(Debug)]
pub struct ModelSelectorConfig {
    /// The session's current model; marked with `✓` while browsing.
    pub current_model: Option<Model>,
    /// Every available model, in catalog order.
    pub models: Vec<Model>,
    /// The session's scoped models; empty leaves the picker in the all scope.
    pub scoped_models: Vec<Model>,
    /// The settings default; marked `· default` and ranked second.
    pub default_model: Option<Model>,
    /// Applied through the same path as typed search, upstream's
    /// `initialSearchInput`.
    pub initial_search: Option<String>,
    /// A refresh line shown until the host reports an outcome.
    pub refresh_status: Option<String>,
}

/// Callbacks receive owned models; persistence belongs to the host.
pub struct ModelSelectorCallbacks {
    /// Confirm selects and closes.
    pub on_select: Rc<dyn Fn(Model)>,
    /// The save binding; `None` hides the footer hint and ignores the key.
    pub on_select_as_default: Option<Rc<dyn Fn(Model)>>,
    /// Escape or the cancel binding closes without selecting.
    pub on_cancel: Rc<dyn Fn()>,
}

impl std::fmt::Debug for ModelSelectorCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSelectorCallbacks")
            .finish_non_exhaustive()
    }
}

struct ModelState {
    all_models: Vec<Model>,
    scoped_models: Vec<Model>,
    filtered: Vec<Model>,
    selected_index: usize,
    scope: ModelScope,
}

impl ModelState {
    fn active(&self) -> &[Model] {
        match self.scope {
            ModelScope::All => &self.all_models,
            ModelScope::Scoped => &self.scoped_models,
        }
    }

    fn selected(&self) -> Option<&Model> {
        self.filtered.get(self.selected_index)
    }
}

/// Orders the current model first, the settings default second, then by
/// provider id; equal providers keep catalog order through the stable sort.
fn sort_models(models: &mut [Model], current: Option<&Model>, default: Option<&Model>) {
    let is_current = |model: &Model| current.is_some_and(|current| same_model(current, model));
    let is_default = |model: &Model| default.is_some_and(|default| same_model(default, model));
    models.sort_by(|a, b| {
        match (is_current(a), is_current(b)) {
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            _ => {}
        }
        match (is_default(a), is_default(b)) {
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            _ => {}
        }
        a.provider.0.cmp(&b.provider.0)
    });
}

fn same_model(a: &Model, b: &Model) -> bool {
    a.provider == b.provider && a.id == b.id
}

/// Upstream's `isDefaultSearch`: every prefix of `default`, case-folded.
fn is_default_search(query: &str) -> bool {
    let normalized = query.trim().to_lowercase();
    !normalized.is_empty() && "default".starts_with(&normalized)
}

/// Filters the active catalog, hoisting default models when the query
/// prefixes `default` (#7209: a filtered query selects the top row).
fn filter_models(active: &[Model], query: &str, default: Option<&Model>) -> Vec<Model> {
    if query.is_empty() {
        return active.to_vec();
    }
    let is_default = |model: &Model| default.is_some_and(|default| same_model(default, model));
    let filtered = fuzzy_filter(active, query, |model| {
        let default_text = if is_default(model) { " default" } else { "" };
        format!(
            "{}{default_text}",
            get_model_selector_search_text(&model.id, &model.provider.0, Some(&model.name))
        )
    });
    if !is_default_search(query) {
        return filtered.into_iter().cloned().collect();
    }
    let defaults: Vec<Model> = active
        .iter()
        .filter(|model| is_default(model))
        .cloned()
        .collect();
    let rest: Vec<Model> = filtered
        .into_iter()
        .filter(|model| !defaults.iter().any(|default| same_model(default, model)))
        .cloned()
        .collect();
    let mut models = defaults;
    models.extend(rest);
    models
}

fn scope_text(scope: ModelScope) -> String {
    let all = if scope == ModelScope::All {
        theme().fg("accent", "all")
    } else {
        theme().fg("muted", "all")
    };
    let scoped = if scope == ModelScope::Scoped {
        theme().fg("accent", "scoped")
    } else {
        theme().fg("muted", "scoped")
    };
    format!(
        "{} {all} {} {scoped}",
        theme().fg("muted", "Scope:"),
        theme().fg("muted", "|")
    )
}

fn scope_hint_text() -> String {
    key_hint("tui.input.tab", "scope") + &theme().fg("muted", " (all/scoped)")
}

/// Picker state plus the list/search/footer children it renders.
pub struct ModelSelectorComponent {
    container: Container,
    list_container: Rc<Container>,
    search_input: Rc<Input>,
    scope_text: Option<Rc<Text>>,
    scope_hint_text: Option<Rc<Text>>,
    current_model: Option<Model>,
    default_model: Option<Model>,
    state: RefCell<ModelState>,
    error_message: RefCell<Option<String>>,
    refresh_status: RefCell<Option<(String, bool)>>,
    callbacks: ModelSelectorCallbacks,
    parser: KeyParser,
}

impl std::fmt::Debug for ModelSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSelectorComponent")
            .field("current_model", &self.current_model)
            .field("scope", &self.state.borrow().scope)
            .finish_non_exhaustive()
    }
}

impl ModelSelectorComponent {
    /// Builds the picker around the supplied snapshot. Scoped models select
    /// the scoped scope initially, upstream's `scopedModels.length > 0`.
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's single constructor branch for branch; splitting it would detach the port from the method it mirrors"
    )]
    pub fn new(config: ModelSelectorConfig, callbacks: ModelSelectorCallbacks) -> Self {
        let scope = if config.scoped_models.is_empty() {
            ModelScope::All
        } else {
            ModelScope::Scoped
        };
        let mut all_models = config.models;
        sort_models(
            &mut all_models,
            config.current_model.as_ref(),
            config.default_model.as_ref(),
        );
        let mut state = ModelState {
            all_models,
            scoped_models: config.scoped_models,
            filtered: Vec::new(),
            selected_index: 0,
            scope,
        };
        let active = state.active().to_vec();
        state.selected_index = active
            .iter()
            .position(|model| {
                config
                    .current_model
                    .as_ref()
                    .is_some_and(|current| same_model(current, model))
            })
            .unwrap_or(0);
        state.filtered = active;

        let container = Container::default();
        let search_input = Rc::new(Input::new());
        if let Some(query) = config
            .initial_search
            .as_ref()
            .filter(|query| !query.is_empty())
        {
            search_input.set_value(query.clone());
        }
        let list_container = Rc::new(Container::default());
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));
        let scope_text = if state.scoped_models.is_empty() {
            container.add_child(Rc::new(Text::with_padding(
                theme().fg(
                    "warning",
                    "Only showing models from configured providers. Use /login to add providers.",
                ),
                0,
                0,
            )));
            None
        } else {
            let text = Rc::new(Text::with_padding(scope_text(state.scope), 0, 0));
            container.add_child(text.clone());
            Some(text)
        };
        let scope_hint_text = if state.scoped_models.is_empty() {
            None
        } else {
            let text = Rc::new(Text::with_padding(scope_hint_text(), 0, 0));
            container.add_child(text.clone());
            Some(text)
        };
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(search_input.clone());
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(list_container.clone());
        container.add_child(Rc::new(Spacer::new()));
        if callbacks.on_select_as_default.is_some() {
            container.add_child(Rc::new(Text::with_padding(
                theme().fg(
                    "dim",
                    &format!(
                        "  {} to select · {} to set as default · {} to cancel",
                        key_display_text("tui.select.confirm"),
                        key_display_text("app.models.save"),
                        key_display_text("tui.select.cancel")
                    ),
                ),
                0,
                0,
            )));
        }
        container.add_child(Rc::new(DynamicBorder::themed()));
        let component = Self {
            container,
            list_container,
            search_input,
            scope_text,
            scope_hint_text,
            current_model: config.current_model,
            default_model: config.default_model,
            state: RefCell::new(state),
            error_message: RefCell::new(None),
            refresh_status: RefCell::new(
                config
                    .refresh_status
                    .filter(|message| !message.is_empty())
                    .map(|message| (message, false)),
            ),
            callbacks,
            parser: KeyParser::new(),
        };
        let query = component.search_input.get_value();
        component.filter(&query);
        component
    }

    /// Replaces the catalog and scoped list; the current model wins the
    /// selection, otherwise the previous index clamps to the new list.
    pub fn update_models(&self, models: &[Model], scoped_models: &[Model]) {
        let mut all_models = models.to_vec();
        sort_models(
            &mut all_models,
            self.current_model.as_ref(),
            self.default_model.as_ref(),
        );
        {
            let mut state = self.state.borrow_mut();
            state.all_models = all_models;
            state.scoped_models = scoped_models.to_vec();
        }
        let query = self.search_input.get_value();
        let active = self.state.borrow().active().to_vec();
        let filtered = filter_models(&active, &query, self.default_model.as_ref());
        {
            let mut state = self.state.borrow_mut();
            state.selected_index = if query.is_empty() {
                filtered
                    .iter()
                    .position(|model| self.is_current(model))
                    .unwrap_or_else(|| state.selected_index.min(filtered.len().saturating_sub(1)))
            } else {
                0
            };
            state.filtered = filtered;
        }
        self.sync_scope_text();
        self.update_list();
    }

    /// The host reports the refresh outcome; an empty message clears the line.
    /// `success` selects the success color, otherwise the muted color.
    pub fn set_refresh_status(&self, message: &str, success: bool) {
        *self.refresh_status.borrow_mut() = if message.is_empty() {
            None
        } else {
            Some((message.to_string(), success))
        };
        self.update_list();
    }

    /// Model catalog failures render above the list, upstream's `errorMessage`.
    pub fn set_error_message(&self, message: Option<&str>) {
        *self.error_message.borrow_mut() = message
            .filter(|message| !message.is_empty())
            .map(str::to_string);
        self.update_list();
    }

    /// The search field receives focus and stays mounted across catalog updates.
    #[must_use]
    pub fn get_search_input(&self) -> Rc<Input> {
        Rc::clone(&self.search_input)
    }

    fn is_default(&self, model: &Model) -> bool {
        self.default_model
            .as_ref()
            .is_some_and(|default| same_model(default, model))
    }

    fn is_current(&self, model: &Model) -> bool {
        self.current_model
            .as_ref()
            .is_some_and(|current| same_model(current, model))
    }

    fn set_scope(&self, scope: ModelScope) {
        if self.state.borrow().scope == scope {
            return;
        }
        self.state.borrow_mut().scope = scope;
        let query = self.search_input.get_value();
        let active = self.state.borrow().active().to_vec();
        let filtered = filter_models(&active, &query, self.default_model.as_ref());
        {
            let mut state = self.state.borrow_mut();
            state.selected_index = if query.is_empty() {
                filtered
                    .iter()
                    .position(|model| self.is_current(model))
                    .unwrap_or(0)
            } else {
                0
            };
            state.filtered = filtered;
        }
        self.sync_scope_text();
        self.update_list();
    }

    fn sync_scope_text(&self) {
        let scope = self.state.borrow().scope;
        if let Some(text) = &self.scope_text {
            text.set_text(&scope_text(scope));
        }
        if let Some(text) = &self.scope_hint_text {
            text.set_text(&scope_hint_text());
        }
    }

    fn filter(&self, query: &str) {
        let active = self.state.borrow().active().to_vec();
        let filtered = filter_models(&active, query, self.default_model.as_ref());
        {
            let mut state = self.state.borrow_mut();
            state.selected_index = if query.is_empty() {
                state.selected_index.min(filtered.len().saturating_sub(1))
            } else {
                0
            };
            state.filtered = filtered;
        }
        self.update_list();
    }

    fn update_list(&self) {
        self.list_container.clear();
        let state = self.state.borrow();
        let len = state.filtered.len();
        if len > 0 {
            let start = state
                .selected_index
                .saturating_sub(MAX_VISIBLE / 2)
                .min(len.saturating_sub(MAX_VISIBLE));
            let end = (start + MAX_VISIBLE).min(len);
            for (index, model) in state.filtered.iter().enumerate().take(end).skip(start) {
                let selected = index == state.selected_index;
                let cursor = if selected {
                    theme().fg("accent", "→ ")
                } else {
                    "  ".to_string()
                };
                let marker = if self.is_current(model) {
                    theme().fg("accent", "✓ ")
                } else {
                    "  ".to_string()
                };
                let label = if selected {
                    theme().fg("accent", &model.id)
                } else {
                    model.id.clone()
                };
                let provider = theme().fg("muted", &format!("[{}]", model.provider.0));
                let default = if self.is_default(model) {
                    theme().fg("muted", " · default")
                } else {
                    String::new()
                };
                self.add_line(format!("{cursor}{marker}{label} {provider}{default}"));
            }
            if start > 0 || end < len {
                self.add_line(
                    theme().fg("muted", &format!("  ({}/{len})", state.selected_index + 1)),
                );
            }
        }
        if let Some(error) = self.error_message.borrow().as_ref() {
            for line in error.split('\n') {
                self.add_line(theme().fg("error", line));
            }
        } else if len == 0 {
            self.add_line(theme().fg("muted", "  No matching models"));
        } else {
            let name = state
                .selected()
                .map_or_else(String::new, |model| model.name.clone());
            self.list_container.add_child(Rc::new(Spacer::new()));
            self.add_line(theme().fg("muted", &format!("  Model Name: {name}")));
        }
        if let Some((message, success)) = self.refresh_status.borrow().as_ref() {
            self.list_container.add_child(Rc::new(Spacer::new()));
            let kind = if *success { "success" } else { "muted" };
            self.add_line(theme().fg(kind, &format!("  {message}")));
        }
    }

    fn add_line(&self, line: String) {
        self.list_container
            .add_child(Rc::new(Text::with_padding(line, 0, 0)));
    }
}

impl Component for ModelSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        let matches = |action| get_keybindings().matches(&self.parser, data, action);
        if matches("tui.input.tab") {
            if self.state.borrow().scoped_models.is_empty() {
                return;
            }
            let next = match self.state.borrow().scope {
                ModelScope::All => ModelScope::Scoped,
                ModelScope::Scoped => ModelScope::All,
            };
            self.set_scope(next);
            return;
        }
        if matches("tui.select.up") || matches("tui.select.down") {
            let mut state = self.state.borrow_mut();
            let len = state.filtered.len();
            if len == 0 {
                return;
            }
            state.selected_index = if matches("tui.select.up") {
                (state.selected_index + len - 1) % len
            } else {
                (state.selected_index + 1) % len
            };
            drop(state);
            self.update_list();
        } else if matches("tui.select.confirm") {
            if let Some(model) = self.state.borrow().selected().cloned() {
                (self.callbacks.on_select)(model);
            }
        } else if matches("tui.select.cancel") {
            (self.callbacks.on_cancel)();
        } else if matches("app.models.save")
            && let Some(on_select_as_default) = &self.callbacks.on_select_as_default
        {
            if let Some(model) = self.state.borrow().selected().cloned() {
                on_select_as_default(model);
            }
        } else {
            self.search_input.handle_input(data);
            let query = self.search_input.get_value();
            self.filter(&query);
        }
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }
}

impl Focusable for ModelSelectorComponent {
    fn set_focused(&self, focused: bool) {
        self.search_input.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.search_input.is_focused()
    }
}
