//! Reusable selection submenus, upstream's `settings-submenu.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#206).
//!
//! Restatements against upstream:
//!
//! - Callbacks and step option sources are [`Rc`]-shared function handles.
//! - Filtering replaces the select list at its container index, mirroring
//!   upstream's `children[listChildIndex] = newList`.
//! - A stepped submenu swaps its active child behind a [`RefCell`]; step
//!   callbacks hold the shared state behind an [`Rc`] so the step components
//!   never cycle back into the submenu.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use pi_tui::components::input::Input;
use pi_tui::components::select_list::{SelectItem, SelectList, SelectListLayoutOptions};
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::fuzzy::fuzzy_filter;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, TuiMouseEvent, TuiMouseEventResult};

use crate::modes::interactive::theme::theme::{bold, get_select_list_theme, theme};

/// The default primary-column bounds, upstream's
/// `SUBMENU_SELECT_LIST_LAYOUT`.
fn submenu_layout() -> SelectListLayoutOptions {
    SelectListLayoutOptions {
        min_primary_column_width: Some(12),
        max_primary_column_width: Some(32),
        truncate_primary: None,
    }
}

/// Construction options, upstream's `SelectSubmenuOptions`.
#[derive(Debug, Clone, Default)]
pub struct SelectSubmenuOptions {
    /// Enables type-to-search fuzzy filtering.
    pub searchable: bool,
    /// Overrides the select-list column bounds.
    pub layout: Option<SelectListLayoutOptions>,
}

/// An optional callback carrying one selected value.
pub type SelectSubmenuValueCallback = Option<Rc<dyn Fn(&str)>>;

/// A borrowed optional selection-change callback.
type SelectSubmenuValueCallbackRef<'a> = Option<&'a Rc<dyn Fn(&str)>>;

/// A single-step titled select list, upstream `SelectSubmenu`.
pub struct SelectSubmenu {
    container: Container,
    select_list: Rc<RefCell<Rc<SelectList>>>,
    list_child_index: usize,
    all_options: Vec<SelectItem>,
    list_layout: SelectListLayoutOptions,
    search_input: Option<Rc<Input>>,
    on_select_cb: Rc<dyn Fn(&str)>,
    on_cancel_cb: Rc<dyn Fn()>,
    on_selection_change_cb: SelectSubmenuValueCallback,
    parser: KeyParser,
}

impl std::fmt::Debug for SelectSubmenu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectSubmenu")
            .field("all_options", &self.all_options)
            .finish_non_exhaustive()
    }
}

impl SelectSubmenu {
    /// Builds the titled list; `on_select` receives the chosen value.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors upstream's constructor argument list; callbacks stay separate parameters like the source"
    )]
    pub fn new(
        title: &str,
        description: &str,
        options: Vec<SelectItem>,
        current_value: &str,
        on_select: Rc<dyn Fn(&str)>,
        on_cancel: Rc<dyn Fn()>,
        on_selection_change: SelectSubmenuValueCallback,
        submenu_options: Option<&SelectSubmenuOptions>,
    ) -> Self {
        let all_options = options;
        let list_layout = submenu_options
            .and_then(|options| options.layout.clone())
            .unwrap_or_else(submenu_layout);
        let searchable = submenu_options.is_some_and(|options| options.searchable);

        let container = Container::default();
        container.add_child(Rc::new(Text::with_padding(
            bold(&theme().fg("accent", title)),
            0,
            0,
        )));
        if !description.is_empty() {
            container.add_child(Rc::new(Spacer::new()));
            container.add_child(Rc::new(Text::with_padding(
                theme().fg("muted", description),
                0,
                0,
            )));
        }
        let search_input = if searchable {
            container.add_child(Rc::new(Spacer::new()));
            let input = Rc::new(Input::new());
            let input_component: Rc<dyn Component> = input.clone();
            container.add_child(input_component);
            Some(input)
        } else {
            None
        };

        container.add_child(Rc::new(Spacer::new()));
        let select_list = Rc::new(RefCell::new(Self::build_select_list(
            &all_options,
            current_value,
            &on_select,
            &on_cancel,
            on_selection_change.as_ref(),
            &list_layout,
        )));
        let list_child_index = container.children.borrow().len();
        let list_rc = select_list.borrow().clone();
        let list_component: Rc<dyn Component> = list_rc;
        container.add_child(list_component);

        container.add_child(Rc::new(Spacer::new()));
        let hint = if searchable {
            "  Type to filter · Enter to select · Esc to go back"
        } else {
            "  Enter to select · Esc to go back"
        };
        container.add_child(Rc::new(Text::with_padding(theme().fg("dim", hint), 0, 0)));

        if let Some(search_input) = &search_input {
            let list = Rc::clone(&select_list);
            *search_input.on_submit.borrow_mut() = Some(Rc::new(move |_| {
                list.borrow().handle_input("\r");
            }));
        }

        Self {
            container,
            select_list,
            list_child_index,
            all_options,
            list_layout,
            search_input,
            on_select_cb: on_select,
            on_cancel_cb: on_cancel,
            on_selection_change_cb: on_selection_change,
            parser: KeyParser::new(),
        }
    }

    fn build_select_list(
        options: &[SelectItem],
        preselect: &str,
        on_select: &Rc<dyn Fn(&str)>,
        on_cancel: &Rc<dyn Fn()>,
        on_selection_change: SelectSubmenuValueCallbackRef<'_>,
        layout: &SelectListLayoutOptions,
    ) -> Rc<SelectList> {
        let max_visible = options.len().min(10);
        let list = Rc::new(SelectList::with_layout(
            options.to_vec(),
            max_visible,
            get_select_list_theme(),
            layout.clone(),
        ));

        if let Some(index) = options.iter().position(|option| option.value == preselect) {
            list.set_selected_index(index);
        }

        let on_select = Rc::clone(on_select);
        *list.on_select.borrow_mut() = Some(Rc::new(move |item| on_select(&item.value)));
        let on_cancel = Rc::clone(on_cancel);
        *list.on_cancel.borrow_mut() = Some(Rc::new(move || on_cancel()));
        if let Some(on_selection_change) = on_selection_change {
            let on_selection_change = Rc::clone(on_selection_change);
            *list.on_selection_change.borrow_mut() =
                Some(Rc::new(move |item| on_selection_change(&item.value)));
        }
        list
    }

    /// Replace the list with the fuzzy-filtered options, upstream
    /// `applyFilter`.
    fn apply_filter(&self, query: &str) {
        let filtered: Vec<SelectItem> = if query.is_empty() {
            self.all_options.clone()
        } else {
            fuzzy_filter(&self.all_options, query, |item| {
                format!(
                    "{} {}",
                    item.label,
                    item.description.as_deref().unwrap_or_default()
                )
            })
            .into_iter()
            .cloned()
            .collect()
        };

        let new_list = Self::build_select_list(
            &filtered,
            "",
            &self.on_select_cb,
            &self.on_cancel_cb,
            self.on_selection_change_cb.as_ref(),
            &self.list_layout,
        );
        *self.select_list.borrow_mut() = Rc::clone(&new_list);
        let children = &self.container.children;
        if let Some(child) = children.borrow_mut().get_mut(self.list_child_index) {
            *child = new_list;
        }
    }
}

impl Component for SelectSubmenu {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        if let Some(search_input) = &self.search_input {
            let matches = |action| get_keybindings().matches(&self.parser, data, action);
            let is_nav = matches("tui.select.up")
                || matches("tui.select.down")
                || matches("tui.select.confirm")
                || matches("tui.select.cancel");
            if is_nav {
                self.select_list.borrow().handle_input(data);
            } else {
                search_input.handle_input(data);
                self.apply_filter(&search_input.get_value());
            }
        } else {
            self.select_list.borrow().handle_input(data);
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

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }
}

/// One stepped-submenu selection context, keyed by step id.
pub type SteppedSubmenuContext = BTreeMap<String, String>;

/// Static or context-dependent step text, upstream's
/// `string | ((context) => string)`.
#[derive(Clone)]
pub enum SteppedSubmenuText {
    /// A fixed string.
    Static(String),
    /// Computed from prior selections.
    Dynamic(Rc<dyn Fn(&SteppedSubmenuContext) -> String>),
}

impl SteppedSubmenuText {
    fn resolve(&self, context: &SteppedSubmenuContext) -> String {
        match self {
            Self::Static(text) => text.clone(),
            Self::Dynamic(text) => text(context),
        }
    }
}

impl std::fmt::Debug for SteppedSubmenuText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(text) => f.debug_tuple("Static").field(text).finish(),
            Self::Dynamic(_) => f.write_str("Dynamic(..)"),
        }
    }
}

/// Builds a step's rows from prior selections.
pub type SteppedSubmenuOptionsFn = Rc<dyn Fn(&SteppedSubmenuContext) -> Vec<SelectItem>>;

/// Pre-selects a step's value from prior selections.
pub type SteppedSubmenuPreselectFn = Rc<dyn Fn(&SteppedSubmenuContext) -> Option<String>>;

/// One step in a [`SteppedSubmenu`], upstream `SteppedSubmenuStep`.
#[derive(Clone)]
pub struct SteppedSubmenuStep {
    /// Result key for this step's selected value.
    pub key: String,
    /// Title shown at the top of the step.
    pub title: SteppedSubmenuText,
    /// Description shown below the title.
    pub description: SteppedSubmenuText,
    /// Builds this step's rows on every entry.
    pub options: SteppedSubmenuOptionsFn,
    /// Value to pre-select when entering the step.
    pub preselect: Option<SteppedSubmenuPreselectFn>,
    /// Enables type-to-search fuzzy filtering for this step.
    pub searchable: bool,
    /// Overrides the select-list column bounds for this step.
    pub layout: Option<SelectListLayoutOptions>,
}

impl std::fmt::Debug for SteppedSubmenuStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SteppedSubmenuStep")
            .field("key", &self.key)
            .field("title", &self.title)
            .finish_non_exhaustive()
    }
}

/// Construction options, upstream `SteppedSubmenuOptions`.
#[derive(Debug, Clone, Default)]
pub struct SteppedSubmenuOptions {
    /// First step to show, skipping earlier ones.
    pub start_at_step: Option<usize>,
    /// Pre-fills the selections of skipped steps.
    pub initial_context: SteppedSubmenuContext,
    /// Returns to step 0 after the last step instead of closing.
    pub loop_: bool,
}

struct SteppedShared {
    steps: Vec<SteppedSubmenuStep>,
    on_complete: Rc<dyn Fn(SteppedSubmenuContext)>,
    on_cancel: Rc<dyn Fn()>,
    loop_: bool,
    context: RefCell<SteppedSubmenuContext>,
    active: RefCell<Option<Rc<dyn Component>>>,
}

/// A multi-step selector over [`SelectSubmenu`], upstream `SteppedSubmenu`.
pub struct SteppedSubmenu {
    shared: Rc<SteppedShared>,
}

impl std::fmt::Debug for SteppedSubmenu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SteppedSubmenu")
            .field("steps", &self.shared.steps.len())
            .finish_non_exhaustive()
    }
}

impl SteppedSubmenu {
    /// Builds the submenu at `start_at_step` (0 when unset).
    #[must_use]
    pub fn new(
        steps: Vec<SteppedSubmenuStep>,
        on_complete: Rc<dyn Fn(SteppedSubmenuContext)>,
        on_cancel: Rc<dyn Fn()>,
        options: SteppedSubmenuOptions,
    ) -> Self {
        let shared = Rc::new(SteppedShared {
            steps,
            on_complete,
            on_cancel,
            loop_: options.loop_,
            context: RefCell::new(options.initial_context),
            active: RefCell::new(None),
        });
        let start = options.start_at_step.unwrap_or(0);
        let active = Self::build_step(&shared, start);
        *shared.active.borrow_mut() = active;
        Self { shared }
    }

    fn build_step(shared: &Rc<SteppedShared>, step_index: usize) -> Option<Rc<dyn Component>> {
        let step = shared.steps.get(step_index)?;
        let total = shared.steps.len();
        let step_label = if total > 1 {
            format!("Step {}/{total} · ", step_index + 1)
        } else {
            String::new()
        };
        let context = shared.context.borrow().clone();
        let title = step.title.resolve(&context);
        let description = format!("{step_label}{}", step.description.resolve(&context));
        let items = (step.options)(&context);
        let preselect = step
            .preselect
            .as_ref()
            .and_then(|preselect| preselect(&context))
            .unwrap_or_default();
        let key = step.key.clone();
        let searchable = step.searchable;
        let layout = step.layout.clone();
        let submenu_options =
            (searchable || layout.is_some()).then_some(SelectSubmenuOptions { searchable, layout });

        let select_shared = Rc::downgrade(shared);
        let select_key = key.clone();
        let on_select: Rc<dyn Fn(&str)> = Rc::new(move |value: &str| {
            let Some(shared) = select_shared.upgrade() else {
                return;
            };
            shared
                .context
                .borrow_mut()
                .insert(select_key.clone(), value.to_string());
            if step_index < total - 1 {
                let next = Self::build_step(&shared, step_index + 1);
                *shared.active.borrow_mut() = next;
            } else {
                (shared.on_complete)(shared.context.borrow().clone());
                if shared.loop_ {
                    shared.context.borrow_mut().clear();
                    let next = Self::build_step(&shared, 0);
                    *shared.active.borrow_mut() = next;
                } else {
                    (shared.on_cancel)();
                }
            }
        });

        let cancel_shared = Rc::downgrade(shared);
        let cancel_key = key;
        let on_cancel: Rc<dyn Fn()> = Rc::new(move || {
            let Some(shared) = cancel_shared.upgrade() else {
                return;
            };
            if step_index > 0 {
                shared.context.borrow_mut().remove(&cancel_key);
                let previous = Self::build_step(&shared, step_index - 1);
                *shared.active.borrow_mut() = previous;
            } else {
                (shared.on_cancel)();
            }
        });

        Some(Rc::new(SelectSubmenu::new(
            &title,
            &description,
            items,
            &preselect,
            on_select,
            on_cancel,
            None,
            submenu_options.as_ref(),
        )))
    }

    fn active(&self) -> Option<Rc<dyn Component>> {
        self.shared.active.borrow().clone()
    }
}

impl Component for SteppedSubmenu {
    fn render(&self, width: usize) -> Vec<String> {
        self.active()
            .map_or_else(Vec::new, |active| active.render(width))
    }

    fn handle_input(&self, data: &str) {
        if let Some(active) = self.active() {
            active.handle_input(data);
        }
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {
        if let Some(active) = self.active() {
            active.invalidate();
        }
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.active().map_or_else(Vec::new, |active| vec![active])
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.active().and_then(|active| active.handle_mouse(event))
    }
}
