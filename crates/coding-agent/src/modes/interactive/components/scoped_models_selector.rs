//! Session-only model cycling configuration at upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. Persistence belongs to the caller.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;

use indexmap::IndexMap;
use pi_ai::types::Model;
use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::fuzzy::fuzzy_filter;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, Focusable, TuiMouseEvent, TuiMouseEventResult};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::key_display_text;
use crate::modes::interactive::model_search::get_model_search_text;
use crate::modes::interactive::theme::theme::{bold, strikethrough, theme};

/// `None` enables every available model; `Some` preserves an explicit cycling order,
/// including unavailable ids. An empty explicit list enables nothing.
pub type EnabledIds = Option<Vec<String>>;

/// The catalog snapshot and session selection need not contain the same ids.
#[derive(Debug, Clone)]
pub struct ModelsConfig {
    /// Available models, in catalog order.
    pub all_models: Vec<Model>,
    /// Full `provider/id` strings; unavailable entries remain editable.
    pub enabled_model_ids: EnabledIds,
    /// If initially nonempty, reserves a line for subsequent refresh messages.
    pub refresh_status: Option<String>,
}

/// Callbacks receive owned snapshots, never a borrow of the component state.
pub struct ModelsCallbacks {
    /// Session-only changes, including reorder operations; does not persist.
    pub on_change: Rc<dyn Fn(EnabledIds)>,
    /// Explicit save; asynchronous work, if needed, is scheduled by the caller.
    pub on_persist: Rc<dyn Fn(EnabledIds)>,
    /// Dismissal does not revert session-only changes.
    pub on_cancel: Rc<dyn Fn()>,
}

impl std::fmt::Debug for ModelsCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelsCallbacks").finish_non_exhaustive()
    }
}

#[derive(Default)]
struct ModelState {
    models_by_id: IndexMap<String, Model>,
    all_ids: Vec<String>,
    enabled_ids: EnabledIds,
    filtered_ids: Vec<String>,
    selected_index: usize,
}

impl ModelState {
    fn set_models(&mut self, models: &[Model]) {
        self.models_by_id.clear();
        self.all_ids.clear();
        for model in models {
            let id = format!("{}/{}", model.provider.0, model.id);
            self.all_ids.push(id.clone());
            self.models_by_id.insert(id, model.clone());
        }
    }

    fn is_enabled(&self, id: &str) -> bool {
        self.enabled_ids
            .as_ref()
            .is_none_or(|ids| ids.iter().any(|enabled| enabled == id))
    }

    fn normalize_enabled(&mut self) {
        if self.enabled_ids.as_ref().is_some_and(|ids| {
            ids.len() == self.all_ids.len() && ids.iter().all(|id| self.all_ids.contains(id))
        }) {
            self.enabled_ids = None;
        }
    }

    fn toggle(&mut self, id: &str) {
        if let Some(ids) = &mut self.enabled_ids {
            if let Some(index) = ids.iter().position(|enabled| enabled == id) {
                ids.remove(index);
            } else {
                ids.push(id.to_string());
                self.normalize_enabled();
            }
        } else {
            self.enabled_ids = Some(
                self.all_ids
                    .iter()
                    .filter(|other| *other != id)
                    .cloned()
                    .collect(),
            );
        }
    }

    fn enable_all(&mut self, targets: &[String]) {
        if let Some(ids) = &mut self.enabled_ids {
            for id in targets {
                if !ids.contains(id) {
                    ids.push(id.clone());
                }
            }
            self.normalize_enabled();
        }
    }

    fn clear_all(&mut self, targets: &[String]) {
        let ids = self.enabled_ids.get_or_insert_with(|| self.all_ids.clone());
        ids.retain(|id| !targets.contains(id));
    }

    fn filter(&mut self, query: &str) {
        let mut ids = self
            .enabled_ids
            .clone()
            .unwrap_or_else(|| self.all_ids.clone());
        if let Some(enabled) = &self.enabled_ids {
            let enabled: HashSet<&String> = enabled.iter().collect();
            ids.extend(
                self.all_ids
                    .iter()
                    .filter(|id| !enabled.contains(id))
                    .cloned(),
            );
        }
        self.filtered_ids = if query.is_empty() {
            ids
        } else {
            fuzzy_filter(&ids, query, |id| {
                self.models_by_id.get(id).map_or_else(
                    || id.clone(),
                    |model| get_model_search_text(&model.id, &model.provider.0, Some(&model.name)),
                )
            })
            .into_iter()
            .cloned()
            .collect()
        };
        self.selected_index = self
            .selected_index
            .min(self.filtered_ids.len().saturating_sub(1));
    }

    fn selected_id(&self) -> Option<&str> {
        self.filtered_ids
            .get(self.selected_index)
            .map(String::as_str)
    }

    fn reorder(&mut self, up: bool) -> bool {
        let Some(id) = self.selected_id().map(str::to_string) else {
            return false;
        };
        let Some(ids) = &mut self.enabled_ids else {
            return false;
        };
        let Some(index) = ids.iter().position(|enabled| enabled == &id) else {
            return false;
        };
        let next = if up {
            index.checked_sub(1)
        } else {
            index.checked_add(1).filter(|next| *next < ids.len())
        };
        let Some(next) = next else {
            return false;
        };
        ids.swap(index, next);
        // A filtered first row can still move up in the full cycling order.
        self.selected_index = if up {
            self.selected_index.saturating_sub(1)
        } else {
            self.selected_index + 1
        };
        true
    }
}

/// Enabled rows precede disabled rows, while unavailable enabled ids remain removable.
pub struct ScopedModelsSelectorComponent {
    container: Container,
    list_container: Rc<Container>,
    search_input: Rc<Input>,
    footer: Rc<Text>,
    refresh_status: Option<Rc<Text>>,
    state: RefCell<ModelState>,
    dirty: Cell<bool>,
    callbacks: ModelsCallbacks,
    parser: KeyParser,
}

impl std::fmt::Debug for ScopedModelsSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopedModelsSelectorComponent")
            .field("enabled_ids", &self.state.borrow().enabled_ids)
            .field("dirty", &self.dirty.get())
            .finish_non_exhaustive()
    }
}

impl ScopedModelsSelectorComponent {
    /// Catalog order is the initial order only when no explicit selection is supplied.
    #[must_use]
    pub fn new(config: ModelsConfig, callbacks: ModelsCallbacks) -> Self {
        let mut state = ModelState {
            enabled_ids: config.enabled_model_ids,
            ..ModelState::default()
        };
        state.set_models(&config.all_models);
        state.filter("");
        let container = Container::default();
        let search_input = Rc::new(Input::new());
        let list_container = Rc::new(Container::default());
        let footer = Rc::new(Text::with_padding("", 0, 0));
        let refresh_status = config
            .refresh_status
            .filter(|message| !message.is_empty())
            .map(|message| {
                Rc::new(Text::with_padding(
                    theme().fg("muted", &format!("  {message}")),
                    0,
                    0,
                ))
            });
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg("accent", &bold("Model Configuration")),
            0,
            0,
        )));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg(
                "muted",
                &format!(
                    "Session-only. {} to save to settings.",
                    key_display_text("app.models.save")
                ),
            ),
            0,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(search_input.clone());
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(list_container.clone());
        container.add_child(Rc::new(Spacer::new()));
        if let Some(status) = &refresh_status {
            container.add_child(status.clone());
        }
        container.add_child(footer.clone());
        container.add_child(Rc::new(DynamicBorder::themed()));
        let selector = Self {
            container,
            list_container,
            search_input,
            footer,
            refresh_status,
            state: RefCell::new(state),
            dirty: Cell::new(false),
            callbacks,
            parser: KeyParser::new(),
        };
        selector.update_list();
        selector.update_footer();
        selector
    }

    /// Preserves the selected id when still visible. `None` keeps the session selection;
    /// `Some(None)` explicitly switches to all enabled.
    pub fn update_models(&self, models: &[Model], enabled_model_ids: Option<EnabledIds>) {
        let mut state = self.state.borrow_mut();
        let selected = state.selected_id().map(str::to_string);
        if let Some(ids) = enabled_model_ids {
            state.enabled_ids = ids;
        }
        state.set_models(models);
        state.filter(&self.search_input.get_value());
        if let Some(index) = state
            .filtered_ids
            .iter()
            .position(|id| Some(id.as_str()) == selected.as_deref())
        {
            state.selected_index = index;
        }
        drop(state);
        self.update_list();
        self.update_footer();
    }

    /// Does nothing unless construction reserved a refresh line. `kind` is a theme
    /// token, normally `muted`, `success`, or `warning`.
    pub fn set_refresh_status(&self, message: &str, kind: &str) {
        if let Some(status) = &self.refresh_status {
            status.set_text(&theme().fg(kind, &format!("  {message}")));
        }
    }

    /// The same input handle receives focus and remains mounted across catalog updates.
    #[must_use]
    pub fn get_search_input(&self) -> Rc<Input> {
        Rc::clone(&self.search_input)
    }

    fn refresh(&self) {
        self.state
            .borrow_mut()
            .filter(&self.search_input.get_value());
        self.update_list();
        self.update_footer();
    }

    fn notify_change(&self) {
        self.dirty.set(true);
        self.refresh();
        let ids = self.state.borrow().enabled_ids.clone();
        (self.callbacks.on_change)(ids);
    }

    fn update_footer(&self) {
        let state = self.state.borrow();
        let count = state.enabled_ids.as_ref().map_or_else(
            || "all enabled".to_string(),
            |ids| {
                let enabled = ids
                    .iter()
                    .filter(|id| state.models_by_id.contains_key(*id))
                    .count();
                let unavailable = ids.len() - enabled;
                let suffix = if unavailable == 0 {
                    String::new()
                } else {
                    format!(" · {unavailable} unavailable")
                };
                format!("{enabled}/{} enabled{suffix}", state.all_ids.len())
            },
        );
        let text = format!(
            "  {} toggle · {} all · {} clear · {} provider · {}/{} reorder · {} save · {count}",
            key_display_text("tui.select.confirm"),
            key_display_text("app.models.enableAll"),
            key_display_text("app.models.clearAll"),
            key_display_text("app.models.toggleProvider"),
            key_display_text("app.models.reorderUp"),
            key_display_text("app.models.reorderDown"),
            key_display_text("app.models.save")
        );
        self.footer.set_text(&if self.dirty.get() {
            theme().fg("dim", &format!("{text} ")) + &theme().fg("warning", "(unsaved)")
        } else {
            theme().fg("dim", &text)
        });
    }

    fn update_list(&self) {
        self.list_container.clear();
        let state = self.state.borrow();
        if state.filtered_ids.is_empty() {
            self.add_line(theme().fg("muted", "  No matching models"));
            return;
        }
        let max_visible = 8;
        let start = state
            .selected_index
            .saturating_sub(max_visible / 2)
            .min(state.filtered_ids.len().saturating_sub(max_visible));
        let end = (start + max_visible).min(state.filtered_ids.len());
        for (index, id) in state.filtered_ids.iter().enumerate().take(end).skip(start) {
            let model = state.models_by_id.get(id);
            let selected = index == state.selected_index;
            let prefix = if selected {
                theme().fg("accent", "→ ")
            } else {
                "  ".to_string()
            };
            let label = model.map_or_else(|| strikethrough(id), |model| model.id.clone());
            let label = if selected {
                theme().fg("accent", &label)
            } else {
                label
            };
            let provider = model.map_or_else(
                || " [unavailable]".to_string(),
                |model| format!(" [{}]", model.provider.0),
            );
            let status = if model.is_some() && state.is_enabled(id) {
                theme().fg("accent", "✓ ")
            } else {
                "  ".to_string()
            };
            self.add_line(format!(
                "{prefix}{status}{label}{}",
                theme().fg("muted", &provider)
            ));
        }
        if start > 0 || end < state.filtered_ids.len() {
            self.add_line(theme().fg(
                "muted",
                &format!(
                    "  ({}/{})",
                    state.selected_index + 1,
                    state.filtered_ids.len()
                ),
            ));
        }
        if let Some(id) = state.selected_id() {
            let name = state.models_by_id.get(id).map_or_else(
                || "Model unavailable".to_string(),
                |model| format!("Model Name: {}", model.name),
            );
            self.list_container.add_child(Rc::new(Spacer::new()));
            self.add_line(theme().fg("muted", &format!("  {name}")));
        }
    }

    fn add_line(&self, line: String) {
        self.list_container
            .add_child(Rc::new(Text::with_padding(line, 0, 0)));
    }

    fn toggle_provider(&self) {
        let mut state = self.state.borrow_mut();
        let Some(model) = state
            .selected_id()
            .and_then(|id| state.models_by_id.get(id))
        else {
            return;
        };
        let ids: Vec<String> = state
            .all_ids
            .iter()
            .filter(|id| {
                state
                    .models_by_id
                    .get(*id)
                    .is_some_and(|other| other.provider == model.provider)
            })
            .cloned()
            .collect();
        if ids.iter().all(|id| state.is_enabled(id)) {
            state.clear_all(&ids);
        } else {
            state.enable_all(&ids);
        }
        drop(state);
        self.notify_change();
    }
}

impl Component for ScopedModelsSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        let matches = |action| get_keybindings().matches(&self.parser, data, action);
        if matches("tui.select.up") || matches("tui.select.down") {
            let mut state = self.state.borrow_mut();
            let len = state.filtered_ids.len();
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
        } else if matches("app.models.reorderUp") || matches("app.models.reorderDown") {
            let changed = self
                .state
                .borrow_mut()
                .reorder(matches("app.models.reorderUp"));
            if changed {
                self.notify_change();
            }
        } else if matches("tui.select.confirm") {
            let mut state = self.state.borrow_mut();
            if let Some(id) = state.selected_id().map(str::to_string) {
                state.toggle(&id);
                drop(state);
                self.notify_change();
            }
        } else if matches("app.models.enableAll") || matches("app.models.clearAll") {
            let mut state = self.state.borrow_mut();
            let filtered = !self.search_input.get_value().is_empty();
            let targets = if filtered {
                state.filtered_ids.clone()
            } else if matches("app.models.clearAll") {
                state
                    .enabled_ids
                    .clone()
                    .unwrap_or_else(|| state.all_ids.clone())
            } else {
                state.all_ids.clone()
            };
            if matches("app.models.enableAll") {
                state.enable_all(&targets);
            } else {
                state.clear_all(&targets);
            }
            drop(state);
            self.notify_change();
        } else if matches("app.models.toggleProvider") {
            self.toggle_provider();
        } else if matches("app.models.save") {
            let ids = self.state.borrow().enabled_ids.clone();
            (self.callbacks.on_persist)(ids);
            self.dirty.set(false);
            self.update_footer();
        } else if self.parser.matches_key(data, "ctrl+c") {
            if self.search_input.get_value().is_empty() {
                (self.callbacks.on_cancel)();
            } else {
                self.search_input.set_value(String::new());
                self.refresh();
            }
        } else if self.parser.matches_key(data, "escape") {
            (self.callbacks.on_cancel)();
        } else {
            self.search_input.handle_input(data);
            self.refresh();
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

impl Focusable for ScopedModelsSelectorComponent {
    fn set_focused(&self, focused: bool) {
        self.search_input.set_focused(focused);
    }
    fn is_focused(&self) -> bool {
        self.search_input.is_focused()
    }
}
