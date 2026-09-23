//! Select list tests, ported 1:1 from
//! `packages/tui/test/select-list.test.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#49), plus the
//! select-list block of `test/mouse-components.test.ts`, which the
//! survey deferred to this ticket.

#![expect(
    clippy::expect_used,
    reason = "the suite asserts on finds like upstream's assert.ok(...); expecting keeps the failure modes readable"
)]

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::RefCell;
use std::rc::Rc;

use pi_tui::components::{SelectItem, SelectList, SelectListTheme};
use pi_tui::tui::{Component, TuiMouseEventResult, TuiMouseEventType};
use pi_tui::utils::visible_width;
use tui_support::{mouse_event, mouse_move, mouse_wheel};

const ITEM_COUNT_FOR_SCROLL: usize = 12;

/// The identity theme, upstream `testTheme`.
fn test_theme() -> SelectListTheme {
    let identity: Rc<dyn Fn(&str) -> String> = Rc::new(ToString::to_string);
    SelectListTheme {
        selected_prefix: Rc::clone(&identity),
        selected_text: Rc::clone(&identity),
        description: Rc::clone(&identity),
        scroll_info: Rc::clone(&identity),
        no_match: identity,
    }
}

fn items(pairs: &[(&str, &str)]) -> Vec<SelectItem> {
    pairs
        .iter()
        .map(|(value, label)| SelectItem {
            value: (*value).to_string(),
            label: (*label).to_string(),
            description: None,
        })
        .collect()
}

fn item(value: &str, label: &str, description: &str) -> SelectItem {
    SelectItem {
        value: value.to_string(),
        label: label.to_string(),
        description: Some(description.to_string()),
    }
}

/// The visible column of `text` inside `line`, upstream `visibleIndexOf`.
fn visible_index_of(line: &str, text: &str) -> usize {
    let index = line.find(text).expect("text present in rendered line");
    visible_width(&line[..index])
}

fn press_result(result: Option<TuiMouseEventResult>) -> bool {
    result.is_some_and(|result| result.handled)
}

// --- select-list.test.ts ------------------------------------------------

#[test]
fn normalizes_multiline_descriptions_to_single_line() {
    let list = SelectList::new(
        vec![item("test", "test", "Line one\nLine two\nLine three")],
        5,
        test_theme(),
    );
    let rendered = list.render(100);

    assert!(!rendered.is_empty());
    assert!(!rendered[0].contains('\n'));
    assert!(rendered[0].contains("Line one Line two Line three"));
}

#[test]
fn keeps_descriptions_aligned_when_the_primary_text_is_truncated() {
    let list = SelectList::new(
        vec![
            item("short", "short", "short description"),
            item(
                "very-long-command-name-that-needs-truncation",
                "very-long-command-name-that-needs-truncation",
                "long description",
            ),
        ],
        5,
        test_theme(),
    );
    let rendered = list.render(80);

    assert_eq!(
        visible_index_of(&rendered[0], "short description"),
        visible_index_of(&rendered[1], "long description")
    );
}

#[test]
fn uses_the_configured_minimum_primary_column_width() {
    let list = SelectList::with_layout(
        vec![item("a", "a", "first"), item("bb", "bb", "second")],
        5,
        test_theme(),
        pi_tui::components::SelectListLayoutOptions {
            min_primary_column_width: Some(12),
            max_primary_column_width: Some(20),
            ..Default::default()
        },
    );
    let rendered = list.render(80);

    // Upstream asserts a UTF-16 `indexOf`; the port's byte offsets would
    // split the selected row's "→ " prefix (4 bytes, 2 columns) from the
    // unselected one, so both rows assert the visible column instead —
    // the same 14 the UTF-16 index reports for the ASCII data.
    assert_eq!(visible_index_of(&rendered[0], "first"), 14);
    assert_eq!(visible_index_of(&rendered[1], "second"), 14);
}

#[test]
fn uses_the_configured_maximum_primary_column_width() {
    let list = SelectList::with_layout(
        vec![
            item(
                "very-long-command-name-that-needs-truncation",
                "very-long-command-name-that-needs-truncation",
                "first",
            ),
            item("short", "short", "second"),
        ],
        5,
        test_theme(),
        pi_tui::components::SelectListLayoutOptions {
            min_primary_column_width: Some(12),
            max_primary_column_width: Some(20),
            ..Default::default()
        },
    );
    let rendered = list.render(80);

    assert_eq!(visible_index_of(&rendered[0], "first"), 22);
    assert_eq!(visible_index_of(&rendered[1], "second"), 22);
}

#[test]
fn allows_overriding_primary_truncation_while_preserving_description_alignment() {
    let list = SelectList::with_layout(
        vec![
            item(
                "very-long-command-name-that-needs-truncation",
                "very-long-command-name-that-needs-truncation",
                "first",
            ),
            item("short", "short", "second"),
        ],
        5,
        test_theme(),
        pi_tui::components::SelectListLayoutOptions {
            min_primary_column_width: Some(12),
            max_primary_column_width: Some(12),
            truncate_primary: Some(Rc::new(|context| {
                if context.text.len() <= context.max_width {
                    return context.text.to_string();
                }
                let cut = context.max_width.saturating_sub(1);
                format!("{}\u{2026}", &context.text[..cut])
            })),
        },
    );
    let rendered = list.render(80);

    assert!(rendered[0].contains('\u{2026}'));
    assert_eq!(
        visible_index_of(&rendered[0], "first"),
        visible_index_of(&rendered[1], "second")
    );
}

// --- mouse-components.test.ts, select-list block -------------------------

#[test]
fn selects_and_activates_list_rows() {
    let list = SelectList::new(
        items(&[("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"), ("e", "E")]),
        3,
        test_theme(),
    );
    let selected = Rc::new(RefCell::new(None::<String>));
    let selected_sink = Rc::clone(&selected);
    *list.on_select.borrow_mut() = Some(Rc::new(move |item| {
        selected_sink.borrow_mut().replace(item.value.clone());
    }));

    assert!(press_result(Component::handle_mouse(
        &list,
        &mouse_event(TuiMouseEventType::Press, 1, 2, 40, 3),
    )));
    assert_eq!(
        list.get_selected_item().map(|item| item.value),
        Some("c".to_string())
    );
    assert!(press_result(Component::handle_mouse(
        &list,
        &mouse_event(TuiMouseEventType::Click, 1, 2, 40, 3),
    )));
    assert_eq!(selected.borrow().as_deref(), Some("c"));
}

#[test]
fn ignores_hover_and_clicks_visible_rows_after_scrolling_row_0() {
    ignores_hover_and_clicks_visible_rows_after_scrolling(0);
}

#[test]
fn ignores_hover_and_clicks_visible_rows_after_scrolling_row_4() {
    ignores_hover_and_clicks_visible_rows_after_scrolling(4);
}

fn ignores_hover_and_clicks_visible_rows_after_scrolling(row: u16) {
    let list = SelectList::new(
        (0..ITEM_COUNT_FOR_SCROLL)
            .map(|index| SelectItem {
                value: format!("item-{index}"),
                label: format!("Item {index}"),
                description: None,
            })
            .collect(),
        5,
        test_theme(),
    );
    let changes = Rc::new(RefCell::new(Vec::<String>::new()));
    let selected = Rc::new(RefCell::new(None::<String>));
    {
        let changes = Rc::clone(&changes);
        *list.on_selection_change.borrow_mut() = Some(Rc::new(move |item| {
            changes.borrow_mut().push(item.value.clone());
        }));
    }
    {
        let selected_sink = Rc::clone(&selected);
        *list.on_select.borrow_mut() = Some(Rc::new(move |item| {
            selected_sink.borrow_mut().replace(item.value.clone());
        }));
    }
    list.set_selected_index(5);

    assert!(press_result(Component::handle_mouse(
        &list,
        &mouse_wheel(1, row, 1),
    )));
    assert_eq!(
        list.get_selected_item().map(|item| item.value),
        Some("item-6".to_string())
    );
    assert_eq!(changes.borrow().clone(), vec!["item-6".to_string()]);
    let before = list.render(80);
    assert!(
        before[usize::from(row)].ends_with(&format!("Item {}", 4 + row)),
        "row {row} should end with Item {}: {:?}",
        4 + row,
        before[usize::from(row)]
    );

    for y in [0u16, 1, 2, 3, 4, row] {
        assert!(Component::handle_mouse(&list, &mouse_move(1, y)).is_none());
        assert_eq!(list.render(80), before);
    }
    assert_eq!(
        list.get_selected_item().map(|item| item.value),
        Some("item-6".to_string())
    );
    assert_eq!(changes.borrow().clone(), vec!["item-6".to_string()]);
    assert!(selected.borrow().is_none(), "no selection before click");

    Component::handle_mouse(
        &list,
        &mouse_event(TuiMouseEventType::Press, 1, row, 80, 10),
    );
    list.render(80);
    Component::handle_mouse(
        &list,
        &mouse_event(TuiMouseEventType::Click, 1, row, 80, 10),
    );
    assert_eq!(
        selected.borrow().as_deref(),
        Some(format!("item-{}", 4 + row).as_str())
    );
    assert_eq!(
        changes.borrow().clone(),
        vec!["item-6".to_string(), format!("item-{}", 4 + row)]
    );
}
