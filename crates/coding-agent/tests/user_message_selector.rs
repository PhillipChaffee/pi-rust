//! Boundary tests for the fork-from-message picker at pin `60e7e76` (#210);
//! upstream ships no component test for it.

#[path = "common/selectors.rs"]
mod selectors;

use std::cell::RefCell;
use std::rc::Rc;

use pi_coding_agent::modes::interactive::components::user_message_selector::{
    UserMessageItem, UserMessageSelectorComponent,
};
use pi_tui::tui::Component;
use pi_tui::utils::visible_width;

use selectors::{cancellation, output, setup};

fn messages(texts: &[&str]) -> Vec<UserMessageItem> {
    texts
        .iter()
        .enumerate()
        .map(|(index, text)| UserMessageItem {
            id: format!("entry-{index}"),
            text: (*text).to_string(),
        })
        .collect()
}

/// Record chosen entry ids without a settings or trust store.
type IdLog = Rc<RefCell<Vec<String>>>;
type IdCallback = Rc<dyn Fn(&str)>;

fn id_recorder() -> (IdLog, IdCallback) {
    let values = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&values);
    (
        values,
        Rc::new(move |id: &str| captured.borrow_mut().push(id.to_string())),
    )
}

fn dialog(items: Vec<UserMessageItem>, initial: Option<&str>) -> UserMessageSelectorComponent {
    setup(&[]);
    UserMessageSelectorComponent::new(items, Rc::new(|_| {}), Rc::new(|| {}), initial)
}

#[test]
fn explains_the_fork_in_the_header() {
    let component = dialog(messages(&["hello"]), None);
    let rendered = output(&component, 120);
    assert!(rendered.contains("Fork from Message"));
    assert!(rendered.contains(
        "Select a user message to copy the active path up to that point into a new session"
    ));
}

#[test]
fn empty_picker_reports_empty_and_renders_the_placeholder() {
    setup(&[]);
    let (cancelled, cancel) = cancellation();
    let component = UserMessageSelectorComponent::new(Vec::new(), Rc::new(|_| {}), cancel, None);
    assert!(component.is_empty());
    assert!(output(&component, 120).contains("No user messages found"));
    let list = component.get_message_list();
    list.handle_input("\r");
    assert_eq!(cancelled.get(), 0);
    list.handle_input("\x1b");
    assert_eq!(cancelled.get(), 1);
}

#[test]
fn newest_message_is_selected_by_default() {
    let component = dialog(messages(&["oldest", "middle", "newest"]), None);
    let rendered = output(&component, 120);
    assert!(rendered.contains("› newest"));
    assert!(!rendered.contains("› oldest"));
}

#[test]
fn initial_entry_id_selects_that_message() {
    let component = dialog(messages(&["oldest", "middle", "newest"]), Some("entry-1"));
    assert!(output(&component, 120).contains("› middle"));
}

#[test]
fn unknown_initial_entry_falls_back_to_the_newest() {
    let component = dialog(messages(&["oldest", "newest"]), Some("missing"));
    assert!(output(&component, 120).contains("› newest"));
}

#[test]
fn newlines_collapse_to_spaces_in_the_preview() {
    let component = dialog(messages(&["first\nsecond\nthird"]), None);
    let rendered = output(&component, 120);
    assert!(rendered.contains("first second third"));
    assert!(!rendered.contains("second\nthird"));
}

#[test]
fn rows_carry_their_position_in_history() {
    let component = dialog(messages(&["one", "two", "three"]), None);
    let rendered = output(&component, 120);
    assert!(rendered.contains("Message 1 of 3"));
    assert!(rendered.contains("Message 2 of 3"));
    assert!(rendered.contains("Message 3 of 3"));
}

#[test]
fn long_messages_truncate_to_the_row_width() {
    let component = dialog(
        messages(&["a very long message that cannot fit on one narrow row"]),
        None,
    );
    let rendered = output(&*component.get_message_list(), 20);
    assert!(rendered.contains("..."));
    for line in rendered.lines() {
        assert!(
            visible_width(line) <= 20,
            "line wider than the row: {line:?}"
        );
    }
}

#[test]
fn scrolls_the_ten_row_window_around_the_selection() {
    let texts: Vec<String> = (1..=12).map(|index| format!("message {index}")).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let component = dialog(messages(&refs), None);
    let list = component.get_message_list();
    let rendered = output(&*list, 120);
    assert!(rendered.contains("Message 3 of 12"));
    assert!(!rendered.contains("Message 2 of 12"));
    assert!(rendered.contains("(12/12)"));
    for _ in 0..6 {
        list.handle_input("\x1b[A");
    }
    let rendered = output(&*list, 120);
    assert!(rendered.contains("Message 1 of 12"));
    assert!(!rendered.contains("Message 11 of 12"));
    assert!(rendered.contains("(6/12)"));
}

#[test]
fn navigation_wraps_at_both_ends() {
    let component = dialog(messages(&["oldest", "middle", "newest"]), None);
    let list = component.get_message_list();
    list.handle_input("\x1b[A");
    assert!(output(&*list, 120).contains("› middle"));
    list.handle_input("\x1b[A");
    list.handle_input("\x1b[A");
    assert!(output(&*list, 120).contains("› newest"));
    list.handle_input("\x1b[B");
    assert!(output(&*list, 120).contains("› oldest"));
}

#[test]
fn confirm_emits_the_selected_entry_id() {
    setup(&[]);
    let (selected, on_select) = id_recorder();
    let component = UserMessageSelectorComponent::new(
        messages(&["oldest", "newest"]),
        on_select,
        Rc::new(|| {}),
        None,
    );
    let list = component.get_message_list();
    list.handle_input("\x1b[A");
    list.handle_input("\r");
    assert_eq!(*selected.borrow(), vec!["entry-0".to_string()]);
}

#[test]
fn cancel_reports_without_selecting() {
    setup(&[]);
    let (selected, on_select) = id_recorder();
    let (cancelled, cancel) = cancellation();
    let component = UserMessageSelectorComponent::new(messages(&["only"]), on_select, cancel, None);
    component.get_message_list().handle_input("\x1b");
    assert_eq!(cancelled.get(), 1);
    assert!(selected.borrow().is_empty());
}

#[test]
fn single_message_stays_selected_under_navigation() {
    let component = dialog(messages(&["only"]), None);
    let list = component.get_message_list();
    list.handle_input("\x1b[A");
    list.handle_input("\x1b[B");
    assert!(output(&*list, 120).contains("› only"));
}

#[test]
fn unrelated_keys_leave_the_selection_alone() {
    let component = dialog(messages(&["oldest", "newest"]), None);
    let list = component.get_message_list();
    list.handle_input("x");
    assert!(output(&*list, 120).contains("› newest"));
}

#[test]
fn the_list_takes_input_and_the_dialog_forwards_children() {
    let component = dialog(messages(&["only"]), None);
    assert!(component.get_message_list().wants_input());
    assert!(!component.children().is_empty());
}

#[test]
fn debug_and_invalidate_reach_the_composed_parts() {
    let component = dialog(messages(&["only"]), None);
    component.invalidate();
    assert!(format!("{component:?}").contains("UserMessageSelectorComponent"));
    let list = component.get_message_list();
    list.invalidate();
    assert!(format!("{list:?}").contains("UserMessageList"));
}
