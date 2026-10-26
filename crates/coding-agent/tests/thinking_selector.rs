//! `thinking-selector.test.ts` at pin `60e7e76`, ported 1:1, plus boundaries.

#![expect(
    clippy::expect_used,
    reason = "assertions require the selected rows to exist"
)]

#[path = "common/selectors.rs"]
mod selectors;

use std::rc::Rc;

use pi_agent_core::types::ThinkingLevel::{self, High, Medium};
use pi_coding_agent::modes::interactive::components::thinking_selector::ThinkingSelectorComponent;
use pi_tui::components::input::Input;
use pi_tui::components::select_list::SelectItem;
use pi_tui::tui::{CURSOR_MARKER, Component, Focusable};

use selectors::{cancellation, click, output, recording, setup};

#[test]
fn keeps_the_current_thinking_level_marked_while_browsing() {
    setup(&[]);
    let selector = ThinkingSelectorComponent::new(
        Medium,
        &[Medium, High],
        Rc::new(|_| {}),
        Rc::new(|| {}),
        None,
        None,
    );
    let list = selector.get_select_list();
    assert_eq!(list.get_selected_item().expect("current").label, "✓ medium");
    assert!(
        output(list.as_ref(), 80)
            .lines()
            .any(|line| line.starts_with("→ ✓ medium"))
    );
    selector.handle_input("\x1b[B");
    let rendered = output(list.as_ref(), 80);
    assert!(rendered.lines().any(|line| line.starts_with("  ✓ medium")));
    assert!(rendered.lines().any(|line| line.starts_with("→   high")));
}

#[test]
fn uses_the_configured_save_binding() {
    setup(&[("app.thinking.save", &["ctrl+r"])]);
    let (saved, save) = recording();
    let selector = ThinkingSelectorComponent::new(
        Medium,
        &[Medium, High],
        Rc::new(|_| {}),
        Rc::new(|| {}),
        Some(save),
        None,
    );
    assert!(output(&selector, 80).contains("Ctrl+R to set as default"));
    selector.handle_input("\x13");
    assert!(saved.borrow().is_empty());
    selector.handle_input("\x12");
    assert_eq!(*saved.borrow(), vec![Medium]);
}

#[test]
fn filters_descriptions_preserves_selection_and_submits_the_rebuilt_list() {
    setup(&[]);
    let (selected, select) = recording();
    let (saved, save) = recording();
    let (cancelled, cancel) = cancellation();
    let selector = ThinkingSelectorComponent::new(
        Medium,
        &[Medium, High],
        select,
        cancel,
        Some(save),
        Some(High),
    );
    selector.handle_input("\x1b[B");
    let old = selector.get_select_list();
    selector.handle_input("reasoning");
    assert!(!Rc::ptr_eq(&old, &selector.get_select_list()));
    assert_eq!(
        selector
            .get_select_list()
            .get_selected_item()
            .expect("high")
            .value,
        "high"
    );
    selector.handle_input("\x15");
    selector.handle_input("default");
    assert!(!output(&selector, 100).contains("✓ medium"));
    assert!(output(&selector, 100).contains("Deep reasoning · default"));
    selector.handle_input("\x13");
    selector.handle_input("\n");
    selector.handle_input("\r");
    assert_eq!(*saved.borrow(), vec![High]);
    assert_eq!(*selected.borrow(), vec![High, High]);
    selector.handle_input("\x1b");
    assert_eq!(cancelled.get(), 1);
    selector.handle_input("\x15");
    assert!(output(&selector, 100).contains("✓ medium"));
}

#[test]
fn empty_filters_and_empty_available_levels_never_select_or_save() {
    setup(&[]);
    for levels in [&[Medium][..], &[][..]] {
        let (selected, select) = recording();
        let (saved, save) = recording();
        let (cancelled, cancel) = cancellation();
        let selector =
            ThinkingSelectorComponent::new(Medium, levels, select, cancel, Some(save), None);
        selector.handle_input("zzzzzz");
        assert!(selector.get_select_list().get_selected_item().is_none());
        for key in ["\x13", "\r", "\n", "\x1b[A", "\x1b[B"] {
            selector.handle_input(key);
        }
        assert!(selected.borrow().is_empty());
        assert!(saved.borrow().is_empty());
        selector.handle_input("\x1b");
        assert_eq!(cancelled.get(), 1);
    }
}

#[test]
fn all_level_names_descriptions_and_callbacks_are_typed_without_budget_estimates() {
    use ThinkingLevel::{Low, Max, Minimal, Off, Xhigh};
    setup(&[]);
    let levels = [Off, Minimal, Low, Medium, High, Xhigh, Max];
    let (selected, select) = recording();
    let selector =
        ThinkingSelectorComponent::new(Medium, &levels, select, Rc::new(|| {}), None, Some(Off));
    let rendered = output(&selector, 160);
    for description in [
        "No reasoning · default",
        "Very brief reasoning",
        "Light reasoning",
        "Moderate reasoning",
        "Deep reasoning",
        "Extra-high reasoning",
        "Maximum reasoning",
    ] {
        assert!(rendered.contains(description), "{description}: {rendered}");
    }
    assert!(!rendered.contains("tokens"));
    let list = selector.get_select_list();
    for (index, level) in levels.iter().enumerate() {
        list.set_selected_index(index);
        selector.handle_input("\r");
        assert_eq!(selected.borrow().last(), Some(level));
    }
    list.on_select.borrow().as_ref().expect("callback")(&SelectItem {
        value: "unknown".to_string(),
        ..SelectItem::default()
    });
    assert_eq!(selected.borrow().len(), levels.len());
    selector.handle_input("\x13");
    assert_eq!(selected.borrow().len(), levels.len());
}

#[test]
fn absent_current_level_and_remapped_keys_select_only_available_levels() {
    setup(&[
        ("tui.select.up", &["ctrl+p"]),
        ("tui.select.down", &["ctrl+n"]),
        ("tui.select.confirm", &["ctrl+y"]),
        ("tui.select.cancel", &["ctrl+x"]),
    ]);
    let (selected, select) = recording();
    let (cancelled, cancel) = cancellation();
    let selector = ThinkingSelectorComponent::new(
        ThinkingLevel::Max,
        &[Medium, High],
        select,
        cancel,
        None,
        None,
    );
    assert!(!output(&selector, 100).contains('✓'));
    assert_eq!(
        selector
            .get_select_list()
            .get_selected_item()
            .expect("first")
            .value,
        "medium"
    );
    selector.handle_input("\x0e");
    selector.handle_input("\x19");
    selector.handle_input("\x10");
    selector.handle_input("\x19");
    selector.handle_input("\x18");
    assert_eq!(*selected.borrow(), vec![High, Medium]);
    assert_eq!(cancelled.get(), 1);
}

#[test]
fn delegates_focus_children_invalidation_and_mouse_selection() {
    setup(&[]);
    let selector = ThinkingSelectorComponent::new(
        Medium,
        &[Medium, High],
        Rc::new(|_| {}),
        Rc::new(|| {}),
        None,
        None,
    );
    assert!(selector.wants_input());
    assert!(format!("{selector:?}").contains("ThinkingSelectorComponent"));
    assert!(!selector.is_focused());
    selector
        .as_focusable()
        .expect("focusable")
        .set_focused(true);
    assert!(selector.is_focused());
    assert!(selector.render(80).join("\n").contains(CURSOR_MARKER));
    let children = selector.children();
    let input = children
        .iter()
        .find_map(|child| {
            let any: &dyn std::any::Any = child.as_ref();
            any.downcast_ref::<Input>()
        })
        .expect("input child");
    assert!(input.is_focused());
    selector.set_focused(false);
    assert!(!input.is_focused());
    let lines = selector.render(80);
    let high_y = lines
        .iter()
        .position(|line| line.contains("high"))
        .expect("high row");
    let event = click(
        u16::try_from(high_y).expect("row"),
        u16::try_from(lines.len()).expect("height"),
    );
    assert!(
        selector
            .handle_mouse(&event)
            .is_some_and(|result| result.handled)
    );
    assert_eq!(
        selector
            .get_select_list()
            .get_selected_item()
            .expect("high")
            .value,
        "high"
    );
    selector.invalidate();
    assert_eq!(selector.render(80).len(), lines.len());
}
