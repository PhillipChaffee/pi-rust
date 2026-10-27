//! The show-images picker, upstream's
//! `src/modes/interactive/components/show-images-selector.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#210).
//!
//! The list takes the current setting and reports the new boolean; persisting
//! it stays with the interactive frame.

use std::rc::Rc;

use pi_tui::components::select_list::{SelectItem, SelectList, SelectListLayoutOptions};
use pi_tui::tui::{Component, Container};

use super::dynamic_border::DynamicBorder;
use crate::modes::interactive::theme::theme::get_select_list_theme;

/// The primary-column bounds, upstream's `SHOW_IMAGES_SELECT_LIST_LAYOUT`.
fn show_images_select_list_layout() -> SelectListLayoutOptions {
    SelectListLayoutOptions {
        min_primary_column_width: Some(12),
        max_primary_column_width: Some(32),
        ..SelectListLayoutOptions::default()
    }
}

/// The bordered Yes/No dialog, upstream's `ShowImagesSelectorComponent`.
pub struct ShowImagesSelectorComponent {
    container: Container,
    select_list: Rc<SelectList>,
}

impl ShowImagesSelectorComponent {
    /// Builds the dialog with the current setting preselected.
    #[must_use]
    pub fn new(current_value: bool, on_select: Rc<dyn Fn(bool)>, on_cancel: Rc<dyn Fn()>) -> Self {
        let items = vec![
            SelectItem {
                value: "yes".to_string(),
                label: "Yes".to_string(),
                description: Some("Show images inline in terminal".to_string()),
            },
            SelectItem {
                value: "no".to_string(),
                label: "No".to_string(),
                description: Some("Show text placeholder instead".to_string()),
            },
        ];
        let select_list = SelectList::with_layout(
            items,
            5,
            get_select_list_theme(),
            show_images_select_list_layout(),
        );
        select_list.set_selected_index(usize::from(!current_value));
        *select_list.on_select.borrow_mut() = Some(Rc::new(move |item: &SelectItem| {
            on_select(item.value == "yes");
        }));
        *select_list.on_cancel.borrow_mut() = Some(on_cancel);

        let select_list = Rc::new(select_list);
        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::themed()));
        let list_component: Rc<dyn Component> = select_list.clone();
        container.add_child(list_component);
        container.add_child(Rc::new(DynamicBorder::themed()));

        Self {
            container,
            select_list,
        }
    }

    /// The focusable list inside the dialog, upstream's `getSelectList`.
    #[must_use]
    pub fn get_select_list(&self) -> Rc<SelectList> {
        Rc::clone(&self.select_list)
    }
}

impl Component for ShowImagesSelectorComponent {
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

impl std::fmt::Debug for ShowImagesSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShowImagesSelectorComponent")
            .finish_non_exhaustive()
    }
}
