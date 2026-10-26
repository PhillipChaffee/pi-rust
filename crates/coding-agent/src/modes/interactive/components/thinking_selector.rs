//! Thinking picker at upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//! Fixed token-budget estimates are omitted: they are not provider metadata.

use std::cell::RefCell;
use std::rc::Rc;

use pi_agent_core::types::ThinkingLevel;
use pi_tui::components::input::Input;
use pi_tui::components::select_list::{
    SelectItem, SelectList, SelectListCallback, SelectListLayoutOptions,
};
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::fuzzy::fuzzy_filter;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, Focusable, TuiMouseEvent, TuiMouseEventResult};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::key_display_text;
use crate::modes::interactive::theme::theme::{get_select_list_theme, theme};

/// Reasoning descriptions without guessed provider budgets (#195).
pub(crate) const fn level_details(level: ThinkingLevel) -> (&'static str, &'static str) {
    match level {
        ThinkingLevel::Off => ("off", "No reasoning"),
        ThinkingLevel::Minimal => ("minimal", "Very brief reasoning"),
        ThinkingLevel::Low => ("low", "Light reasoning"),
        ThinkingLevel::Medium => ("medium", "Moderate reasoning"),
        ThinkingLevel::High => ("high", "Deep reasoning"),
        ThinkingLevel::Xhigh => ("xhigh", "Extra-high reasoning"),
        ThinkingLevel::Max => ("max", "Maximum reasoning"),
    }
}

/// The current marker stays fixed while browsing and filtering.
pub struct ThinkingSelectorComponent {
    container: Container,
    search_input: Rc<Input>,
    select_list: Rc<RefCell<Rc<SelectList>>>,
    select_list_child_index: usize,
    all_items: Vec<SelectItem>,
    on_select: SelectListCallback,
    on_cancel: Rc<dyn Fn()>,
    on_select_as_default: Option<SelectListCallback>,
    parser: KeyParser,
}

impl std::fmt::Debug for ThinkingSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThinkingSelectorComponent")
            .field("all_items", &self.all_items)
            .finish_non_exhaustive()
    }
}

impl ThinkingSelectorComponent {
    /// Offers only `available_levels`; saving a default is optional and does not persist here.
    #[must_use]
    pub fn new(
        current_level: ThinkingLevel,
        available_levels: &[ThinkingLevel],
        on_select: Rc<dyn Fn(ThinkingLevel)>,
        on_cancel: Rc<dyn Fn()>,
        on_select_as_default: Option<Rc<dyn Fn(ThinkingLevel)>>,
        default_thinking_level: Option<ThinkingLevel>,
    ) -> Self {
        let all_items: Vec<SelectItem> = available_levels
            .iter()
            .map(|&level| {
                let (value, description) = level_details(level);
                SelectItem {
                    value: value.to_string(),
                    label: format!(
                        "{}{value}",
                        if level == current_level { "✓ " } else { "  " }
                    ),
                    description: Some(if Some(level) == default_thinking_level {
                        format!("{description} · default")
                    } else {
                        description.to_string()
                    }),
                }
            })
            .collect();
        let bind = |callback: Rc<dyn Fn(ThinkingLevel)>| -> SelectListCallback {
            let levels = available_levels.to_vec();
            Rc::new(move |item| {
                if let Some(level) = levels
                    .iter()
                    .find(|&&level| level_details(level).0 == item.value)
                {
                    callback(*level);
                }
            })
        };
        let on_select = bind(on_select);
        let on_select_as_default = on_select_as_default.map(bind);
        let select_list = Rc::new(RefCell::new(build_select_list(
            all_items.clone(),
            Some(level_details(current_level).0),
            &on_select,
            &on_cancel,
        )));
        let search_input = Rc::new(Input::new());
        *search_input.on_submit.borrow_mut() = Some({
            let select_list = Rc::clone(&select_list);
            Rc::new(move |_| select_list.borrow().handle_input("\r"))
        });
        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding("Thinking Level", 0, 0)));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding(
            format!(
                "{} cycles thinking levels in-session",
                key_display_text("app.thinking.cycle")
            ),
            0,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(search_input.clone());
        container.add_child(Rc::new(Spacer::new()));
        let select_list_child_index = container.children.borrow().len();
        container.add_child(select_list.borrow().clone());
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg(
                "dim",
                &format!(
                    "  {} to select · {} to set as default · {} to cancel",
                    key_display_text("tui.select.confirm"),
                    key_display_text("app.thinking.save"),
                    key_display_text("tui.select.cancel"),
                ),
            ),
            0,
            0,
        )));
        container.add_child(Rc::new(DynamicBorder::themed()));
        Self {
            container,
            search_input,
            select_list,
            select_list_child_index,
            all_items,
            on_select,
            on_cancel,
            on_select_as_default,
            parser: KeyParser::new(),
        }
    }

    /// The active filtered list; a filter change replaces this handle, as upstream does.
    #[must_use]
    pub fn get_select_list(&self) -> Rc<SelectList> {
        self.select_list.borrow().clone()
    }

    fn apply_filter(&self, query: &str) {
        let filtered = fuzzy_filter(&self.all_items, query, |item| {
            format!(
                "{} {}",
                item.value,
                item.description.as_deref().unwrap_or_default()
            )
        })
        .into_iter()
        .cloned()
        .collect();
        let selected = self.get_select_list().get_selected_item();
        let list = build_select_list(
            filtered,
            selected.as_ref().map(|item| item.value.as_str()),
            &self.on_select,
            &self.on_cancel,
        );
        self.container.children.borrow_mut()[self.select_list_child_index] = list.clone();
        *self.select_list.borrow_mut() = list;
    }
}

fn build_select_list(
    items: Vec<SelectItem>,
    preselect: Option<&str>,
    on_select: &SelectListCallback,
    on_cancel: &Rc<dyn Fn()>,
) -> Rc<SelectList> {
    let index = items
        .iter()
        .position(|item| Some(item.value.as_str()) == preselect);
    let len = items.len().max(1);
    let list = SelectList::with_layout(
        items,
        len,
        get_select_list_theme(),
        SelectListLayoutOptions {
            min_primary_column_width: Some(12),
            max_primary_column_width: Some(32),
            ..SelectListLayoutOptions::default()
        },
    );
    if let Some(index) = index {
        list.set_selected_index(index);
    }
    *list.on_select.borrow_mut() = Some(Rc::clone(on_select));
    *list.on_cancel.borrow_mut() = Some(Rc::clone(on_cancel));
    Rc::new(list)
}

impl Component for ThinkingSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        let matches = |action| get_keybindings().matches(&self.parser, data, action);
        if matches("app.thinking.save")
            && let Some(callback) = &self.on_select_as_default
        {
            if let Some(item) = self.get_select_list().get_selected_item() {
                callback(&item);
            }
            return;
        }
        if [
            "tui.select.up",
            "tui.select.down",
            "tui.select.confirm",
            "tui.select.cancel",
        ]
        .iter()
        .any(|action| matches(action))
        {
            self.get_select_list().handle_input(data);
            return;
        }
        self.search_input.handle_input(data);
        self.apply_filter(&self.search_input.get_value());
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

impl Focusable for ThinkingSelectorComponent {
    fn set_focused(&self, focused: bool) {
        self.search_input.set_focused(focused);
    }
    fn is_focused(&self) -> bool {
        self.search_input.is_focused()
    }
}
