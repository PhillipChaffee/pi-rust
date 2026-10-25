//! Dynamic border and theme selector components (#187).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::cell::RefCell;
use std::rc::Rc;

use pi_coding_agent::modes::interactive::components::dynamic_border::DynamicBorder;
use pi_coding_agent::modes::interactive::components::theme_selector::ThemeSelectorComponent;
use pi_coding_agent::modes::interactive::theme::theme::{init_theme, set_theme_dirs_override};
use pi_tui::tui::Component;
use tempfile::TempDir;

fn empty_dirs() -> TempDir {
    let root = TempDir::new().expect("temp");
    let builtin = root.path().join("builtin");
    let custom = root.path().join("custom");
    std::fs::create_dir_all(&builtin).expect("builtin");
    std::fs::create_dir_all(&custom).expect("custom");
    set_theme_dirs_override(Some((builtin, custom)));
    root
}

#[test]
fn dynamic_border_fills_the_width_with_an_explicit_color() {
    let border = DynamicBorder::new(Rc::new(|text: &str| format!("<{text}>")));
    assert!(format!("{border:?}").contains("DynamicBorder"));
    let lines = border.render(4);
    assert_eq!(lines, vec!["<────>".to_string()]);
    assert_eq!(border.render(0), vec!["<─>".to_string()]);
}

#[test]
fn themed_border_uses_the_border_token() {
    let _root = empty_dirs();
    init_theme(Some("dark"), false);
    let lines = DynamicBorder::themed().render(3);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("───"));
    assert!(lines[0].starts_with("\x1b[38;2;95;135;255m"));
    set_theme_dirs_override(None);
}

#[test]
fn theme_selector_lists_themes_and_previews_selection() {
    let _root = empty_dirs();
    init_theme(Some("dark"), false);
    let selected = Rc::new(RefCell::new(Vec::<String>::new()));
    let cancelled = Rc::new(RefCell::new(false));
    let previewed = Rc::new(RefCell::new(Vec::<String>::new()));
    let selector = ThemeSelectorComponent::new(
        "dark",
        {
            let selected = Rc::clone(&selected);
            Rc::new(move |name: &str| selected.borrow_mut().push(name.to_string()))
        },
        {
            let cancelled = Rc::clone(&cancelled);
            Rc::new(move || *cancelled.borrow_mut() = true)
        },
        {
            let previewed = Rc::clone(&previewed);
            Rc::new(move |name: &str| previewed.borrow_mut().push(name.to_string()))
        },
    );
    assert!(format!("{selector:?}").contains("ThemeSelectorComponent"));
    let lines = selector.render(30);
    assert!(lines.len() >= 3, "border, list, border: {lines:?}");
    assert!(lines[0].contains('─'));
    assert!(lines.last().expect("last").contains('─'));

    let list = selector.get_select_list();
    let current = list.get_selected_item().expect("current");
    assert_eq!(current.value, "dark");
    assert_eq!(current.description.as_deref(), Some("(current)"));

    let light = {
        list.set_selected_index(1);
        list.get_selected_item().expect("light listed")
    };
    let on_preview = list
        .on_selection_change
        .borrow()
        .as_ref()
        .map(Rc::clone)
        .expect("preview callback");
    on_preview(&light);
    assert_eq!(*previewed.borrow(), vec!["light".to_string()]);

    let on_select = list
        .on_select
        .borrow()
        .as_ref()
        .map(Rc::clone)
        .expect("select callback");
    on_select(&light);
    assert_eq!(*selected.borrow(), vec!["light".to_string()]);

    let on_cancel = list
        .on_cancel
        .borrow()
        .as_ref()
        .map(Rc::clone)
        .expect("cancel callback");
    on_cancel();
    assert!(*cancelled.borrow());

    selector.invalidate();
    set_theme_dirs_override(None);
}
