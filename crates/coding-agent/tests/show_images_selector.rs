//! Boundary tests for the show-images picker at pin `60e7e76` (#210);
//! upstream ships no component test for it.

#[path = "common/selectors.rs"]
mod selectors;

use std::rc::Rc;

use pi_coding_agent::modes::interactive::components::show_images_selector::ShowImagesSelectorComponent;
use pi_tui::tui::Component;

use selectors::{cancellation, output, recording, setup};

fn dialog(
    current: bool,
) -> (
    ShowImagesSelectorComponent,
    Rc<std::cell::RefCell<Vec<bool>>>,
) {
    setup(&[]);
    let (selected, on_select) = recording::<bool>();
    (
        ShowImagesSelectorComponent::new(current, on_select, Rc::new(|| {})),
        selected,
    )
}

#[test]
fn preselects_yes_when_images_are_shown() {
    let (component, _) = dialog(true);
    assert_eq!(
        component
            .get_select_list()
            .get_selected_item()
            .map(|item| item.value),
        Some("yes".to_string())
    );
}

#[test]
fn preselects_no_when_images_are_hidden() {
    let (component, _) = dialog(false);
    assert_eq!(
        component
            .get_select_list()
            .get_selected_item()
            .map(|item| item.value),
        Some("no".to_string())
    );
}

#[test]
fn renders_both_choices_with_their_descriptions() {
    let (component, _) = dialog(true);
    let rendered = output(&component, 120);
    assert!(rendered.contains("Yes"));
    assert!(rendered.contains("Show images inline in terminal"));
    assert!(rendered.contains("No"));
    assert!(rendered.contains("Show text placeholder instead"));
}

#[test]
fn confirming_yes_selects_showing_images() {
    let (component, selected) = dialog(true);
    component.get_select_list().handle_input("\r");
    assert_eq!(*selected.borrow(), vec![true]);
}

#[test]
fn moving_down_and_confirming_selects_hiding_images() {
    let (component, selected) = dialog(true);
    component.get_select_list().handle_input("\x1b[B");
    component.get_select_list().handle_input("\r");
    assert_eq!(*selected.borrow(), vec![false]);
}

#[test]
fn navigation_wraps_between_the_two_choices() {
    let (component, selected) = dialog(false);
    component.get_select_list().handle_input("\x1b[A");
    component.get_select_list().handle_input("\r");
    assert_eq!(*selected.borrow(), vec![true]);
}

#[test]
fn cancel_reports_without_selecting() {
    setup(&[]);
    let (selected, on_select) = recording::<bool>();
    let (cancelled, cancel) = cancellation();
    let component = ShowImagesSelectorComponent::new(true, on_select, cancel);
    component.get_select_list().handle_input("\x1b");
    assert_eq!(cancelled.get(), 1);
    assert!(selected.borrow().is_empty());
}

#[test]
fn borders_bracket_the_list() {
    let (component, _) = dialog(true);
    let lines: Vec<String> = output(&component, 120)
        .lines()
        .map(ToString::to_string)
        .collect();
    assert!(lines.first().is_some_and(|line| line.contains('─')));
    assert!(lines.last().is_some_and(|line| line.contains('─')));
    assert!(!component.children().is_empty());
}

#[test]
fn debug_and_invalidate_reach_the_composed_parts() {
    let (component, _) = dialog(true);
    component.invalidate();
    assert!(format!("{component:?}").contains("ShowImagesSelectorComponent"));
}
