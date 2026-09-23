//! Settings list tests, ported 1:1 from
//! `packages/tui/test/settings-list.test.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#49), plus the settings-list
//! block of `test/mouse-components.test.ts`, which the survey deferred to
//! this ticket, and a boundary suite binding the branches upstream's two
//! suites leave untested (submenu delegation, navigation, the empty views,
//! and the search row's mouse routing).

#![expect(
    clippy::expect_used,
    reason = "the suite asserts on finds like upstream's assert.ok(...); expecting keeps the failure modes readable"
)]

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_tui::components::{
    SettingItem, SettingsList, SettingsListOptions, SettingsListTheme, SettingsSelectedColorFn,
    SettingsSubmenuDone, SettingsSubmenuDoneOptions,
};
use pi_tui::tui::{Component, TuiMouseEventType};
use tui_support::{mouse_event, mouse_move, mouse_wheel};

/// The identity theme with the `> ` cursor, upstream `testTheme`.
fn test_theme() -> SettingsListTheme {
    let identity: Rc<dyn Fn(&str) -> String> = Rc::new(ToString::to_string);
    let identity_selected: SettingsSelectedColorFn =
        Rc::new(|text: &str, _selected: bool| text.to_string());
    SettingsListTheme {
        label: Rc::clone(&identity_selected),
        value: identity_selected,
        description: Rc::clone(&identity),
        cursor: "> ".to_string(),
        hint: identity,
    }
}

fn items() -> Vec<SettingItem> {
    vec![SettingItem {
        id: "tui-mode".to_string(),
        label: "TUI mode".to_string(),
        description: None,
        current_value: "regular".to_string(),
        values: Some(vec!["regular".to_string(), "fullscreen".to_string()]),
        submenu: None,
    }]
}

fn setting_item(id: &str, label: &str, current_value: &str) -> SettingItem {
    SettingItem {
        id: id.to_string(),
        label: label.to_string(),
        description: Some(format!("Description {id}")),
        current_value: current_value.to_string(),
        values: Some(vec!["off".to_string(), "on".to_string()]),
        submenu: None,
    }
}

// --- settings-list.test.ts ----------------------------------------------

#[test]
fn includes_spaces_in_an_active_search_instead_of_changing_the_selected_setting() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let sink = Rc::clone(&changes);
    let list = SettingsList::with_options(
        items(),
        10,
        test_theme(),
        Rc::new(move |id, value| sink.borrow_mut().push((id.to_string(), value.to_string()))),
        Rc::new(|| {}),
        SettingsListOptions {
            enable_search: true,
        },
    );

    for character in "TUI mode".chars() {
        Component::handle_input(&list, &character.to_string());
    }

    assert!(changes.borrow().is_empty());
    assert!(
        tui_support::strip_ansi(&list.render(80)[0]).contains("TUI mode"),
        "the input row should show the typed query: {:?}",
        list.render(80)[0]
    );

    Component::handle_input(&list, "\r");
    assert_eq!(
        changes.borrow().clone(),
        vec![("tui-mode".to_string(), "fullscreen".to_string())]
    );
}

#[test]
fn keeps_space_as_a_change_shortcut_before_a_search_query_is_entered() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let sink = Rc::clone(&changes);
    let list = SettingsList::with_options(
        items(),
        10,
        test_theme(),
        Rc::new(move |id, value| sink.borrow_mut().push((id.to_string(), value.to_string()))),
        Rc::new(|| {}),
        SettingsListOptions {
            enable_search: true,
        },
    );

    Component::handle_input(&list, " ");

    assert_eq!(
        changes.borrow().clone(),
        vec![("tui-mode".to_string(), "fullscreen".to_string())]
    );
}

// --- mouse-components.test.ts, settings-list block -----------------------

#[test]
fn activates_settings_rows() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let sink = Rc::clone(&changes);
    let list_items = vec![
        SettingItem {
            id: "mode".to_string(),
            label: "Mode".to_string(),
            description: None,
            current_value: "one".to_string(),
            values: Some(vec!["one".to_string(), "two".to_string()]),
            submenu: None,
        },
        SettingItem {
            id: "other".to_string(),
            label: "Other".to_string(),
            description: None,
            current_value: "off".to_string(),
            values: Some(vec!["off".to_string(), "on".to_string()]),
            submenu: None,
        },
        SettingItem {
            id: "third".to_string(),
            label: "Third".to_string(),
            description: None,
            current_value: "low".to_string(),
            values: Some(vec!["low".to_string(), "high".to_string()]),
            submenu: None,
        },
        SettingItem {
            id: "fourth".to_string(),
            label: "Fourth".to_string(),
            description: None,
            current_value: "x".to_string(),
            values: Some(vec!["x".to_string(), "y".to_string()]),
            submenu: None,
        },
    ];
    let list = SettingsList::new(
        list_items,
        3,
        test_theme(),
        Rc::new(move |id, value| sink.borrow_mut().push((id.to_string(), value.to_string()))),
        Rc::new(|| {}),
    );

    Component::handle_mouse(&list, &mouse_event(TuiMouseEventType::Press, 1, 2, 40, 5));
    Component::handle_mouse(&list, &mouse_event(TuiMouseEventType::Click, 1, 2, 40, 5));
    assert_eq!(
        changes.borrow().clone(),
        vec![("third".to_string(), "high".to_string())]
    );
}

#[test]
fn ignores_hover_and_clicks_visible_settings_rows_after_scrolling_row_0() {
    ignores_hover_and_clicks_visible_settings_rows_after_scrolling(0);
}

#[test]
fn ignores_hover_and_clicks_visible_settings_rows_after_scrolling_row_4() {
    ignores_hover_and_clicks_visible_settings_rows_after_scrolling(4);
}

fn ignores_hover_and_clicks_visible_settings_rows_after_scrolling(row: u16) {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let sink = Rc::clone(&changes);
    let list = SettingsList::with_options(
        (0..12)
            .map(|index| setting_item(&format!("item-{index}"), &format!("Item {index}"), "off"))
            .collect(),
        5,
        test_theme(),
        Rc::new(move |id, value| sink.borrow_mut().push((id.to_string(), value.to_string()))),
        Rc::new(|| {}),
        SettingsListOptions {
            enable_search: true,
        },
    );
    list.select_item("item-5");
    Component::handle_mouse(&list, &mouse_wheel(1, row + 2, 1));
    let before = list.render(80);
    assert!(
        strip(&before[4]).starts_with("> Item 6"),
        "row 4 should show the selected item: {:?}",
        strip(&before[4])
    );
    assert!(
        strip(&before[usize::from(row) + 2]).contains(&format!("Item {} ", 4 + row)),
        "row {} should show item-{}: {:?}",
        row + 2,
        4 + row,
        strip(&before[usize::from(row) + 2])
    );

    for y in [0u16, 1, 2, 3, 4, row] {
        assert!(Component::handle_mouse(&list, &mouse_move(1, y + 2)).is_none());
        assert_eq!(list.render(80), before);
    }
    assert!(changes.borrow().is_empty());

    Component::handle_mouse(
        &list,
        &mouse_event(TuiMouseEventType::Press, 1, row + 2, 80, 10),
    );
    list.render(80);
    Component::handle_mouse(
        &list,
        &mouse_event(TuiMouseEventType::Click, 1, row + 2, 80, 10),
    );
    assert_eq!(
        changes.borrow().clone(),
        vec![(format!("item-{}", 4 + row), "on".to_string())]
    );
}

fn strip(line: &str) -> String {
    tui_support::strip_ansi(line).trim_end().to_string()
}

// --- boundary suite: branches upstream's two suites leave untested -------

/// A submenu component capturing its `done` callback, driving the
/// delegation paths. Typing `a` applies `changed`; typing `n` navigates to
/// `target`; Escape closes with neither.
struct TestSubmenu {
    done: RefCell<Option<SettingsSubmenuDone>>,
    handled_input: Rc<RefCell<Vec<String>>>,
}

impl Component for TestSubmenu {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["SUBMENU".to_string()]
    }

    fn handle_input(&self, data: &str) {
        self.handled_input.borrow_mut().push(data.to_string());
        if data == "a" {
            if let Some(done) = self.done.borrow_mut().take() {
                done(Some("changed".to_string()), None);
            }
        } else if data == "n" {
            if let Some(done) = self.done.borrow_mut().take() {
                done(
                    Some("changed".to_string()),
                    Some(SettingsSubmenuDoneOptions {
                        navigate_to: Some("target".to_string()),
                    }),
                );
            }
        } else if data == "\x1b"
            && let Some(done) = self.done.borrow_mut().take()
        {
            done(None, None);
        }
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {}
}

/// Build a list whose `mode` item opens the test submenu and whose
/// `target` item opens a plain one.
fn submenu_list(
    changes: &Rc<RefCell<Vec<(String, String)>>>,
    cancelled: &Rc<Cell<bool>>,
    open_count: &Rc<Cell<usize>>,
) -> SettingsList {
    let open_count = Rc::clone(open_count);
    let items = vec![
        SettingItem {
            id: "mode".to_string(),
            label: "Mode".to_string(),
            description: Some("The mode".to_string()),
            current_value: "a".to_string(),
            values: None,
            submenu: Some({
                let open_count = Rc::clone(&open_count);
                Rc::new(move |_current_value, done| {
                    open_count.set(open_count.get() + 1);
                    let component = TestSubmenu {
                        done: RefCell::new(Some(done)),
                        handled_input: Rc::new(RefCell::new(Vec::new())),
                    };
                    Rc::new(component)
                })
            }),
        },
        SettingItem {
            id: "target".to_string(),
            label: "Target".to_string(),
            description: None,
            current_value: "x".to_string(),
            values: None,
            submenu: Some({
                let open_count = Rc::clone(&open_count);
                Rc::new(move |_current_value, done| {
                    open_count.set(open_count.get() + 1);
                    Rc::new(TestSubmenu {
                        done: RefCell::new(Some(done)),
                        handled_input: Rc::new(RefCell::new(Vec::new())),
                    })
                })
            }),
        },
        SettingItem {
            id: "plain".to_string(),
            label: "Plain".to_string(),
            description: None,
            current_value: "off".to_string(),
            values: Some(vec!["off".to_string(), "on".to_string()]),
            submenu: None,
        },
    ];
    let sink = Rc::clone(changes);
    let cancelled_sink = Rc::clone(cancelled);
    SettingsList::new(
        items,
        10,
        test_theme(),
        Rc::new(move |id, value| sink.borrow_mut().push((id.to_string(), value.to_string()))),
        Rc::new(move || cancelled_sink.set(true)),
    )
}

#[test]
fn opens_a_submenu_and_applies_its_done_value() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let cancelled = Rc::new(Cell::new(false));
    let open_count = Rc::new(Cell::new(0));
    let list = submenu_list(&changes, &cancelled, &open_count);

    // Enter opens the submenu for the selected item.
    Component::handle_input(&list, "\r");
    assert_eq!(open_count.get(), 1);

    // While open, render delegates to it and keys flow to it.
    assert!(strip(&list.render(80)[0]).contains("SUBMENU"));
    let cancel_sink = Rc::clone(&cancelled);
    let _ = cancel_sink;

    // `a` applies the done value: change recorded, submenu closed,
    // selection restored to the item that opened it.
    Component::handle_input(&list, "a");
    assert_eq!(
        changes.borrow().clone(),
        vec![("mode".to_string(), "changed".to_string())]
    );
    assert!(
        strip(&list.render(80)[0]).contains("Mode"),
        "submenu closed"
    );

    // Escape on the main list hits the list's own cancel.
    Component::handle_input(&list, "\x1b");
    assert!(cancelled.get());
}

#[test]
fn submenu_navigation_opens_the_target_item() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let cancelled = Rc::new(Cell::new(false));
    let open_count = Rc::new(Cell::new(0));
    let list = submenu_list(&changes, &cancelled, &open_count);

    Component::handle_input(&list, "\r"); // open mode submenu
    Component::handle_input(&list, "n"); // done with navigateTo target

    // The target item's submenu opens automatically.
    assert_eq!(open_count.get(), 2);
    assert!(strip(&list.render(80)[0]).contains("SUBMENU"));
    assert_eq!(
        changes.borrow().clone(),
        vec![("mode".to_string(), "changed".to_string())]
    );
}

#[test]
fn submenu_escape_restores_the_opener_selection_without_changes() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let cancelled = Rc::new(Cell::new(false));
    let open_count = Rc::new(Cell::new(0));
    let list = submenu_list(&changes, &cancelled, &open_count);

    Component::handle_input(&list, "\r");
    Component::handle_input(&list, "\x1b"); // done with neither

    assert!(changes.borrow().is_empty());
    assert!(
        strip(&list.render(80)[0]).contains("Mode"),
        "submenu closed, main list restored"
    );
}

#[test]
fn cycles_values_and_reports_changes() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let sink = Rc::clone(&changes);
    let list = SettingsList::new(
        vec![SettingItem {
            id: "toggle".to_string(),
            label: "Toggle".to_string(),
            description: None,
            current_value: "b".to_string(),
            values: Some(vec!["a".to_string(), "b".to_string()]),
            submenu: None,
        }],
        5,
        test_theme(),
        Rc::new(move |id, value| sink.borrow_mut().push((id.to_string(), value.to_string()))),
        Rc::new(|| {}),
    );

    Component::handle_input(&list, "\r");
    assert_eq!(
        changes.borrow().clone(),
        vec![("toggle".to_string(), "a".to_string())]
    );
    Component::handle_input(&list, "\r");
    assert_eq!(
        changes.borrow().clone(),
        vec![
            ("toggle".to_string(), "a".to_string()),
            ("toggle".to_string(), "b".to_string())
        ]
    );

    // A value outside the cycle starts at the first entry, upstream's
    // `indexOf` miss mapping to index 0.
    Component::handle_input(&list, "\r");
    assert_eq!(changes.borrow().last().expect("a change recorded").1, "a");
}

#[test]
fn update_value_and_select_item_rewrite_the_view() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let sink = Rc::clone(&changes);
    let list = SettingsList::new(
        vec![setting_item("a", "A", "off"), setting_item("b", "B", "off")],
        5,
        test_theme(),
        Rc::new(move |id, value| sink.borrow_mut().push((id.to_string(), value.to_string()))),
        Rc::new(|| {}),
    );

    list.update_value("b", "on");
    let rendered = strip(&list.render(80)[1]);
    assert!(rendered.contains("on"), "updated value renders: {rendered}");

    list.select_item("b");
    let rendered = strip(&list.render(80)[1]);
    assert!(
        rendered.starts_with("> "),
        "selection follows the id: {rendered}"
    );

    // Unknown ids are no-ops.
    list.select_item("missing");
    let rendered = strip(&list.render(80)[1]);
    assert!(rendered.starts_with("> "));
}

#[test]
fn shows_the_empty_views_and_the_description_footer() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let sink = Rc::clone(&changes);
    let empty = SettingsList::new(
        Vec::new(),
        5,
        test_theme(),
        Rc::new(move |id, value| sink.borrow_mut().push((id.to_string(), value.to_string()))),
        Rc::new(|| {}),
    );
    let rendered = empty.render(80);
    // The non-search empty view renders the hint alone, upstream returns
    // before the hint footer without search enabled.
    assert!(strip(&rendered[0]).contains("No settings available"));
    assert_eq!(rendered.len(), 1);

    let described = SettingsList::new(
        vec![SettingItem {
            id: "a".to_string(),
            label: "A".to_string(),
            description: Some("long description text".to_string()),
            current_value: "x".to_string(),
            values: None,
            submenu: None,
        }],
        5,
        test_theme(),
        Rc::new(|_id, _value| {}),
        Rc::new(|| {}),
    );
    let rendered = described.render(80);
    assert!(
        strip(&rendered[rendered.len() - 3]).contains("long description text"),
        "the selected item's description renders in the footer: {rendered:?}"
    );
    assert!(strip(&rendered[rendered.len() - 1]).contains("Enter/Space to change"));
}

#[test]
fn search_filters_through_the_fuzzy_view_and_renders_the_scroll_indicator() {
    let changes = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let sink = Rc::clone(&changes);
    let list = SettingsList::with_options(
        (0..12)
            .map(|index| setting_item(&format!("item-{index}"), &format!("Item {index}"), "off"))
            .collect(),
        3,
        test_theme(),
        Rc::new(move |id, value| sink.borrow_mut().push((id.to_string(), value.to_string()))),
        Rc::new(|| {}),
        SettingsListOptions {
            enable_search: true,
        },
    );

    // Narrow the view, then confirm the no-match path.
    Component::handle_input(&list, "1");
    // An impossible query shows the no-match line.
    for character in "zzz".chars() {
        Component::handle_input(&list, &character.to_string());
    }
    let rendered = list.render(80);
    assert!(strip(&rendered[2]).contains("No matching settings"));
    // The hint footer still renders.
    assert!(strip(&rendered[rendered.len() - 1]).contains("Type to search"));
}
