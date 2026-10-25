//! The theme picker, upstream's
//! `src/modes/interactive/components/theme-selector.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The list is bracketed by dynamic borders; moving the selection previews
//! the theme through `on_preview`, selecting confirms it, and cancelling
//! restores the caller's theme.

use std::rc::Rc;

use pi_tui::components::select_list::{SelectItem, SelectList, SelectListLayoutOptions};
use pi_tui::tui::{Component, Container};

use super::dynamic_border::DynamicBorder;
use crate::modes::interactive::theme::theme::{get_available_themes, get_select_list_theme};

/// The primary-column bounds, upstream's `THEME_SELECT_LIST_LAYOUT`.
fn theme_select_list_layout() -> SelectListLayoutOptions {
    SelectListLayoutOptions {
        min_primary_column_width: Some(12),
        max_primary_column_width: Some(32),
        ..SelectListLayoutOptions::default()
    }
}

/// The theme selector component, upstream's `ThemeSelectorComponent`.
pub struct ThemeSelectorComponent {
    container: Container,
    select_list: Rc<SelectList>,
}

impl ThemeSelectorComponent {
    /// Build the selector over the available themes.
    ///
    /// `on_select` takes the confirmed theme name, `on_cancel` runs on
    /// dismissal, and `on_preview` receives every selection change.
    #[must_use]
    pub fn new(
        current_theme: &str,
        on_select: Rc<dyn Fn(&str)>,
        on_cancel: Rc<dyn Fn()>,
        on_preview: Rc<dyn Fn(&str)>,
    ) -> Self {
        let themes = get_available_themes();
        let items: Vec<SelectItem> = themes
            .iter()
            .map(|name| SelectItem {
                value: name.clone(),
                label: name.clone(),
                description: (name == current_theme).then(|| "(current)".to_string()),
            })
            .collect();
        let select_list = SelectList::with_layout(
            items,
            10,
            get_select_list_theme(),
            theme_select_list_layout(),
        );
        if let Some(index) = themes.iter().position(|name| name == current_theme) {
            select_list.set_selected_index(index);
        }
        {
            *select_list.on_select.borrow_mut() =
                Some(Rc::new(move |item: &SelectItem| on_select(&item.value)));
            *select_list.on_cancel.borrow_mut() = Some(on_cancel);
            *select_list.on_selection_change.borrow_mut() =
                Some(Rc::new(move |item: &SelectItem| on_preview(&item.value)));
        }
        let select_list = Rc::new(select_list);
        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::themed()));
        let select_component: Rc<dyn Component> = select_list.clone();
        container.add_child(select_component);
        container.add_child(Rc::new(DynamicBorder::themed()));
        Self {
            container,
            select_list,
        }
    }

    /// The focusable list inside the selector, upstream's `getSelectList`.
    #[must_use]
    pub fn get_select_list(&self) -> Rc<SelectList> {
        Rc::clone(&self.select_list)
    }
}

impl Component for ThemeSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }
}

impl std::fmt::Debug for ThemeSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThemeSelectorComponent")
            .finish_non_exhaustive()
    }
}
