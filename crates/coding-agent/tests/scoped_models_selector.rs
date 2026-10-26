//! The four scoped-model selector cases at pin `60e7e76`, the component
//! regressions from #3217/#6949, and catalog-update/input boundaries.

#![expect(
    clippy::expect_used,
    reason = "a missing rendered row fails the assertion"
)]

mod common;
#[path = "common/selectors.rs"]
mod selectors;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_ai::types::Model;
use pi_coding_agent::modes::interactive::components::scoped_models_selector::{
    EnabledIds, ModelsCallbacks, ModelsConfig, ScopedModelsSelectorComponent,
};
use pi_coding_agent::modes::interactive::model_search::get_model_search_text;
use pi_coding_agent::modes::interactive::theme::theme::strikethrough;
use pi_tui::tui::{CURSOR_MARKER, Component, Focusable};

use common::model_layer::model;
use selectors::{cancellation, click, output, recording, setup};

struct Picker {
    component: ScopedModelsSelectorComponent,
    changes: Rc<RefCell<Vec<EnabledIds>>>,
    saved: Rc<RefCell<Vec<EnabledIds>>>,
    cancelled: Rc<Cell<usize>>,
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "explicit-list fixtures match the nullable selection contract; all-enabled fixtures use None"
)]
fn ids(values: &[&str]) -> EnabledIds {
    Some(values.iter().map(|id| format!("faux/{id}")).collect())
}

fn catalog() -> Vec<Model> {
    ["model-a", "model-b", "model-c"]
        .map(|id| model("faux", id))
        .to_vec()
}

fn picker(models: Vec<Model>, enabled: EnabledIds, status: Option<&str>) -> Picker {
    let (changes, on_change) = recording();
    let (saved, on_persist) = recording();
    let (cancelled, on_cancel) = cancellation();
    Picker {
        component: ScopedModelsSelectorComponent::new(
            ModelsConfig {
                all_models: models,
                enabled_model_ids: enabled,
                refresh_status: status.map(str::to_string),
            },
            ModelsCallbacks {
                on_change,
                on_persist,
                on_cancel,
            },
        ),
        changes,
        saved,
        cancelled,
    }
}

fn markers(component: &ScopedModelsSelectorComponent) -> Vec<bool> {
    let text = output(component, 200);
    ["model-a", "model-b", "model-c"]
        .iter()
        .map(|id| {
            let row = text
                .lines()
                .find(|line| line.contains(&format!("{id} [")))
                .expect("model row");
            row.chars().skip(2).collect::<String>().starts_with("✓ ")
        })
        .collect()
}

#[test]
fn marks_every_model_after_enabling_all() {
    setup(&[]);
    let p = picker(catalog(), ids(&["model-a"]), None);
    p.component.handle_input("\x01");
    assert_eq!(*p.changes.borrow(), vec![None]);
    assert_eq!(markers(&p.component), vec![true, true, true]);
    assert!(output(&p.component, 200).contains("all enabled"));
}

#[test]
fn disables_only_the_selected_model_after_enabling_all() {
    setup(&[]);
    let p = picker(catalog(), ids(&["model-a"]), None);
    p.component.handle_input("\x01");
    p.component.handle_input("\r");
    assert_eq!(
        p.changes.borrow().last(),
        Some(&ids(&["model-b", "model-c"]))
    );
    assert_eq!(markers(&p.component), vec![false, true, true]);
}

#[test]
fn enables_only_the_selected_model_after_clearing_all() {
    setup(&[]);
    let p = picker(catalog(), ids(&["model-a", "model-b", "model-c"]), None);
    p.component.handle_input("\x18");
    assert_eq!(p.changes.borrow().last(), Some(&ids(&[])));
    assert_eq!(markers(&p.component), vec![false, false, false]);
    p.component.handle_input("\r");
    assert_eq!(p.changes.borrow().last(), Some(&ids(&["model-a"])));
    assert_eq!(markers(&p.component), vec![true, false, false]);
}

#[test]
fn restores_all_enabled_after_reenabling_the_last_disabled_model() {
    setup(&[]);
    let p = picker(catalog(), ids(&["model-a"]), None);
    for key in ["\x01", "\r", "\x1b[B", "\x1b[B", "\r"] {
        p.component.handle_input(key);
    }
    assert_eq!(p.changes.borrow().last(), Some(&None));
    assert_eq!(markers(&p.component), vec![true, true, true]);
    assert!(output(&p.component, 200).contains("all enabled"));
}

#[test]
fn propagates_reordered_models_as_an_owned_session_snapshot() {
    setup(&[]);
    let p = picker(catalog(), ids(&["model-a", "model-b", "model-c"]), None);
    p.component.handle_input("\x1b[1;3B");
    assert_eq!(
        *p.changes.borrow(),
        vec![ids(&["model-b", "model-a", "model-c"])]
    );
    assert!(output(&p.component, 200).contains("→ ✓ model-a [faux]"));
    p.component.handle_input("\x1b[1;3A");
    assert_eq!(
        p.changes.borrow().last(),
        Some(&ids(&["model-a", "model-b", "model-c"]))
    );
    assert_eq!(
        p.changes.borrow()[0],
        ids(&["model-b", "model-a", "model-c"])
    );
}

#[test]
fn shows_and_removes_an_enabled_model_without_a_catalog_entry() {
    setup(&[]);
    let p = picker(
        vec![model("faux", "available")],
        ids(&["unavailable", "available"]),
        None,
    );
    let raw = p.component.render(200).join("\n");
    assert!(raw.contains(&strikethrough("faux/unavailable")));
    let text = output(&p.component, 200);
    assert!(text.contains("faux/unavailable [unavailable]"));
    assert!(text.contains("1/1 enabled · 1 unavailable"));
    assert!(text.contains("Model unavailable"));
    p.component.handle_input("\x10");
    assert!(p.changes.borrow().is_empty());
    p.component.handle_input("\r");
    assert_eq!(*p.changes.borrow(), vec![ids(&["available"])]);
    p.component.handle_input("\x13");
    assert_eq!(*p.saved.borrow(), vec![ids(&["available"])]);
    assert!(!output(&p.component, 200).contains("(unsaved)"));
}

#[test]
fn renders_cached_models_and_accepts_catalog_updates_without_losing_selection() {
    setup(&[]);
    let models = catalog();
    let p = picker(
        models[..2].to_vec(),
        None,
        Some("Refreshing model catalogs…"),
    );
    assert!(!output(&p.component, 200).contains("model-c"));
    assert!(output(&p.component, 200).contains("Refreshing model catalogs…"));
    p.component.handle_input("\x1b[B");
    p.component.update_models(
        &[models[2].clone(), models[0].clone(), models[1].clone()],
        None,
    );
    p.component
        .set_refresh_status("Model catalogs refreshed.", "success");
    let text = output(&p.component, 200);
    assert!(text.contains("model-c"));
    assert!(text.contains("→ ✓ model-b [faux]"));
    assert!(text.contains("Model catalogs refreshed."));
    p.component
        .update_models(&models[..1], Some(ids(&["model-a", "missing"])));
    assert!(output(&p.component, 200).contains("→   faux/missing [unavailable]"));
    assert!(output(&p.component, 200).contains("faux/missing [unavailable]"));
    p.component.update_models(&models, Some(None));
    assert!(output(&p.component, 200).contains("all enabled"));
    assert!(p.changes.borrow().is_empty());
}

#[test]
fn searches_model_names_providers_and_unavailable_ids_and_bulk_edits_only_matches() {
    setup(&[]);
    let mut models = catalog();
    models[1].name = "Special Display Name".to_string();
    let p = picker(models, None, None);
    p.component.handle_input("Special");
    let text = output(&p.component, 200);
    assert!(text.contains("model-b [faux]"));
    assert!(!text.contains("model-a [faux]"));
    assert!(text.contains("Model Name: Special Display Name"));
    p.component.handle_input("\x18");
    assert_eq!(
        p.changes.borrow().last(),
        Some(&ids(&["model-a", "model-c"]))
    );
    p.component.handle_input("\x01");
    assert_eq!(p.changes.borrow().last(), Some(&None));
    p.component.handle_input("\x03");
    p.component.handle_input("faux/model-a");
    assert!(output(&p.component, 200).contains("→ ✓ model-a [faux]"));
    p.component.update_models(&[], Some(ids(&["unknown"])));
    p.component.handle_input("\x03");
    p.component.handle_input("unknown");
    assert!(output(&p.component, 200).contains("faux/unknown [unavailable]"));
    p.component.handle_input("\x18");
    assert_eq!(p.changes.borrow().last(), Some(&ids(&[])));
    p.component.handle_input("\x01");
    assert_eq!(p.changes.borrow().last(), Some(&None));
}

#[test]
fn provider_toggle_edits_the_whole_provider_even_when_search_hides_its_other_models() {
    setup(&[]);
    let p = picker(
        vec![
            model("faux", "one"),
            model("faux", "two"),
            model("other", "one"),
        ],
        None,
        None,
    );
    p.component.handle_input("faux two");
    p.component.handle_input("\x10");
    assert_eq!(
        *p.changes.borrow(),
        vec![Some(vec!["other/one".to_string()])]
    );
    p.component.handle_input("\x10");
    assert_eq!(p.changes.borrow().last(), Some(&None));
    let mut models = catalog();
    models[0].name = "UniqueDisplay".to_string();
    p.component
        .update_models(&models, Some(ids(&["model-a", "missing"])));
    p.component.handle_input("\x03");
    p.component.handle_input("UniqueDisplay");
    assert!(!output(&p.component, 200).contains("model-b [faux]"));
    p.component.handle_input("\x10");
    assert_eq!(
        p.changes.borrow().last(),
        Some(&ids(&["model-a", "missing", "model-b", "model-c"]))
    );
    p.component.handle_input("\x18");
    assert_eq!(
        p.changes.borrow().last(),
        Some(&ids(&["missing", "model-b", "model-c"]))
    );
    p.component.handle_input("\x03");
    p.component.handle_input("\x18");
    assert_eq!(p.changes.borrow().last(), Some(&ids(&[])));
}

#[test]
fn reorders_only_explicit_enabled_rows_within_the_full_order() {
    setup(&[]);
    let p = picker(catalog(), None, None);
    for key in ["\x1b[1;3A", "\x1b[1;3B"] {
        p.component.handle_input(key);
    }
    assert!(p.changes.borrow().is_empty());
    p.component
        .update_models(&catalog(), Some(ids(&["model-a", "model-b"])));
    p.component.handle_input("\x1b[1;3A");
    p.component.handle_input("\x1b[B");
    p.component.handle_input("\x1b[1;3B");
    p.component.handle_input("\x1b[B");
    for key in ["\x1b[1;3A", "\x1b[1;3B"] {
        p.component.handle_input(key);
    }
    assert!(p.changes.borrow().is_empty());
    p.component.handle_input("model-b");
    p.component.handle_input("\x1b[1;3A");
    assert_eq!(*p.changes.borrow(), vec![ids(&["model-b", "model-a"])]);
    assert!(output(&p.component, 200).contains("→ ✓ model-b [faux]"));
    p.component.handle_input("\x03");
    p.component
        .update_models(&catalog(), Some(ids(&["missing", "model-a"])));
    p.component.handle_input("missing");
    p.component.handle_input("\x1b[1;3B");
    assert_eq!(
        p.changes.borrow().last(),
        Some(&ids(&["model-a", "missing"]))
    );
}

#[test]
fn catalog_update_preserves_search_dirty_state_and_unavailable_selections() {
    setup(&[]);
    let p = picker(catalog(), ids(&["model-a"]), Some("Refreshing"));
    p.component.handle_input("model-b");
    p.component.handle_input("\r");
    assert_eq!(
        p.changes.borrow().last(),
        Some(&ids(&["model-a", "model-b"]))
    );
    p.component.update_models(&catalog()[..1], None);
    let text = output(&p.component, 200);
    assert!(text.contains("faux/model-b [unavailable]"));
    assert!(text.contains("(unsaved)"));
    assert_eq!(p.component.get_search_input().get_value(), "model-b");
    p.component.update_models(&catalog(), Some(ids(&[])));
    assert!(output(&p.component, 200).contains("→   model-b [faux]"));
    p.component
        .set_refresh_status("Using cached models.", "warning");
    assert!(output(&p.component, 200).contains("Using cached models."));
    assert_eq!(p.changes.borrow().len(), 1);
}

#[test]
fn empty_catalog_and_empty_search_results_are_safe_for_every_action() {
    setup(&[]);
    for models in [vec![], catalog()] {
        let p = picker(models, None, Some(""));
        p.component.handle_input("zzzzzzzz");
        assert!(output(&p.component, 200).contains("No matching models"));
        for key in ["\x1b[A", "\x1b[B", "\r", "\x1b[1;3A", "\x1b[1;3B", "\x10"] {
            p.component.handle_input(key);
        }
        assert!(p.changes.borrow().is_empty());
        p.component.set_refresh_status("not mounted", "success");
        assert!(!output(&p.component, 200).contains("not mounted"));
        p.component.handle_input("\x01");
        p.component.handle_input("\x18");
        p.component.handle_input("\x13");
        assert_eq!(p.changes.borrow().len(), 2);
        assert_eq!(p.saved.borrow().len(), 1);
        p.component.handle_input("\x03");
        p.component.update_models(&[], Some(ids(&[])));
        p.component.handle_input("\x01");
        assert_eq!(p.changes.borrow().last(), Some(&None));
    }
}

#[test]
fn configured_actions_drive_navigation_toggles_and_save_without_implicit_persistence() {
    setup(&[
        ("app.models.save", &["ctrl+r"]),
        ("app.models.enableAll", &["ctrl+e"]),
        ("app.models.clearAll", &["ctrl+d"]),
        ("app.models.toggleProvider", &["ctrl+t"]),
        ("app.models.reorderUp", &["alt+k"]),
        ("app.models.reorderDown", &["alt+j"]),
        ("tui.select.up", &["ctrl+p"]),
        ("tui.select.down", &["ctrl+n"]),
        ("tui.select.confirm", &["ctrl+y"]),
    ]);
    let p = picker(catalog(), ids(&["model-a", "model-b"]), None);
    let text = output(&p.component, 300);
    for hint in [
        "Ctrl+R to save to settings.",
        "Ctrl+Y toggle",
        "Ctrl+E all",
        "Ctrl+D clear",
        "Ctrl+T provider",
        "Alt+K/Alt+J reorder",
        "Ctrl+R save",
    ] {
        assert!(text.contains(hint), "{hint}: {text}");
    }
    p.component.handle_input("\x13");
    assert!(p.saved.borrow().is_empty());
    p.component.handle_input("\x1bj");
    p.component.handle_input("\x1bk");
    p.component.handle_input("\x0e");
    p.component.handle_input("\x10");
    p.component.handle_input("\x19");
    assert!(p.saved.borrow().is_empty());
    assert!(output(&p.component, 300).contains("(unsaved)"));
    p.component.handle_input("\x12");
    assert_eq!(*p.saved.borrow(), vec![ids(&["model-b"])]);
    assert!(!output(&p.component, 300).contains("(unsaved)"));
    p.component.handle_input("\x05");
    p.component.handle_input("\x04");
    p.component.handle_input("\x14");
    assert_eq!(p.changes.borrow().last(), Some(&None));
    assert_eq!(p.saved.borrow().len(), 1);
}

#[test]
fn ctrl_c_clears_search_before_cancelling_and_escape_always_cancels() {
    setup(&[]);
    let p = picker(catalog(), None, None);
    p.component.handle_input("model-a");
    p.component.handle_input("\x03");
    assert!(p.component.get_search_input().get_value().is_empty());
    assert_eq!(p.cancelled.get(), 0);
    p.component.handle_input("\x03");
    p.component.handle_input("query");
    p.component.handle_input("\x1b");
    assert_eq!(p.cancelled.get(), 2);
    assert_eq!(p.component.get_search_input().get_value(), "query");
    assert!(p.changes.borrow().is_empty());
    assert!(p.saved.borrow().is_empty());
}

#[test]
fn navigation_wraps_and_limits_visible_rows_to_eight() {
    setup(&[]);
    let models = (0..12)
        .map(|i| model("faux", &format!("choice-{i:02}")))
        .collect();
    let p = picker(models, None, None);
    let text = output(&p.component, 200);
    assert!(text.contains("(1/12)"));
    assert!(!text.contains("choice-08 [faux]"));
    p.component.handle_input("\x1b[A");
    let text = output(&p.component, 200);
    assert!(text.contains("→ ✓ choice-11 [faux]"));
    assert!(text.contains("(12/12)"));
    assert!(!text.contains("choice-03 [faux]"));
    p.component.handle_input("\x1b[B");
    assert!(output(&p.component, 200).contains("→ ✓ choice-00 [faux]"));
}

#[test]
fn forwards_focus_children_invalidation_and_mouse_to_the_mounted_input() {
    setup(&[]);
    let p = picker(catalog(), None, None);
    assert!(p.component.wants_input());
    assert!(format!("{:?}", p.component).contains("ScopedModelsSelectorComponent"));
    assert!(!p.component.is_focused());
    p.component
        .as_focusable()
        .expect("focusable")
        .set_focused(true);
    assert!(p.component.get_search_input().is_focused());
    assert!(p.component.render(80).join("\n").contains(CURSOR_MARKER));
    p.component.set_focused(false);
    assert!(!p.component.is_focused());
    assert!(!p.component.get_search_input().is_focused());
    assert_eq!(p.component.children().len(), 11);
    p.component.invalidate();
    assert!(!p.component.render(80).is_empty());
    assert!(p.component.handle_mouse(&click(0, 1)).is_none());
}

#[test]
fn callbacks_can_reenter_catalog_updates_without_a_state_borrow_panic() {
    setup(&[]);
    let owner = Rc::new(RefCell::new(
        std::rc::Weak::<ScopedModelsSelectorComponent>::new(),
    ));
    let updated = Rc::clone(&owner);
    let persisted = Rc::clone(&owner);
    let callbacks = ModelsCallbacks {
        on_change: Rc::new(move |ids| {
            updated
                .borrow()
                .upgrade()
                .expect("mounted")
                .update_models(&catalog(), Some(ids));
        }),
        on_persist: Rc::new(move |ids| {
            persisted
                .borrow()
                .upgrade()
                .expect("mounted")
                .update_models(&catalog(), Some(ids));
        }),
        on_cancel: Rc::new(|| {}),
    };
    assert!(format!("{callbacks:?}").contains("ModelsCallbacks"));
    let config = ModelsConfig {
        all_models: catalog(),
        enabled_model_ids: None,
        refresh_status: None,
    };
    assert!(format!("{config:?}").contains("ModelsConfig"));
    let component = Rc::new(ScopedModelsSelectorComponent::new(
        config.clone(),
        callbacks,
    ));
    *owner.borrow_mut() = Rc::downgrade(&component);
    component.handle_input("\r");
    component.handle_input("\x13");
    assert!(!output(component.as_ref(), 200).contains("(unsaved)"));
    assert!(config.enabled_model_ids.is_none());
}

#[test]
fn bulk_enable_retains_unavailable_ids_without_mistaking_equal_counts_for_full_coverage() {
    setup(&[]);
    let p = picker(catalog(), ids(&["missing", "model-a"]), None);
    p.component.handle_input("model-b");
    p.component.handle_input("\x01");
    assert_eq!(
        p.changes.borrow().last(),
        Some(&ids(&["missing", "model-a", "model-b"]))
    );
    p.component.handle_input("\x03");
    p.component.handle_input("\x01");
    assert_eq!(
        p.changes.borrow().last(),
        Some(&ids(&["missing", "model-a", "model-b", "model-c"]))
    );
    p.component.update_models(&catalog(), Some(None));
    p.component.handle_input("\x18");
    assert_eq!(p.changes.borrow().last(), Some(&ids(&[])));
}

#[test]
fn search_text_preserves_upstreams_repeated_spellings_and_optional_name() {
    assert_eq!(
        get_model_search_text("id", "provider", Some("Display Name")),
        "id provider provider/id provider id Display Name"
    );
    for name in [None, Some("")] {
        assert_eq!(
            get_model_search_text("nested/id", "provider", name),
            "nested/id provider provider/nested/id provider nested/id"
        );
    }
}
