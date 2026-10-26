//! The ordinary model picker at pin `60e7e76`: the two upstream
//! `model-selector.test.ts` cases that do not need catalog refresh, the
//! #7209 filter-reset regression, and host-update/state boundaries.

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
use pi_coding_agent::modes::interactive::components::model_selector::{
    ModelSelectorCallbacks, ModelSelectorComponent, ModelSelectorConfig,
};
use pi_coding_agent::modes::interactive::model_search::get_model_selector_search_text;
use pi_tui::tui::{Component, Focusable};

use common::model_layer::model;
use selectors::{cancellation, click, output, recording, setup};

struct Picker {
    component: ModelSelectorComponent,
    selected: Rc<RefCell<Vec<Model>>>,
    defaults: Rc<RefCell<Vec<Model>>>,
    cancelled: Rc<Cell<usize>>,
}

fn picker_with(config: ModelSelectorConfig, with_default: bool) -> Picker {
    let (selected, on_select) = recording();
    let (defaults, on_default) = recording();
    let (cancelled, on_cancel) = cancellation();
    Picker {
        component: ModelSelectorComponent::new(
            config,
            ModelSelectorCallbacks {
                on_select,
                on_select_as_default: with_default.then_some(on_default),
                on_cancel,
            },
        ),
        selected,
        defaults,
        cancelled,
    }
}

fn picker(config: ModelSelectorConfig) -> Picker {
    picker_with(config, true)
}

fn catalog(ids: &[&str]) -> Vec<Model> {
    ids.iter().map(|id| model("faux", id)).collect()
}

const fn config(models: Vec<Model>) -> ModelSelectorConfig {
    ModelSelectorConfig {
        current_model: None,
        models,
        scoped_models: Vec::new(),
        default_model: None,
        initial_search: None,
        refresh_status: None,
    }
}

fn row(text: &str, id: &str) -> String {
    text.lines()
        .find(|line| line.contains(&format!("{id} [")))
        .expect("model row")
        .trim_end()
        .to_string()
}

#[test]
fn keeps_the_current_model_marked_while_browsing() {
    setup(&[]);
    let models = catalog(&["current-model", "browsed-model"]);
    let current = models[0].clone();
    let p = picker(ModelSelectorConfig {
        current_model: Some(current.clone()),
        ..config(models)
    });
    let text = output(&p.component, 120);
    assert_eq!(
        row(&text, "current-model"),
        format!("→ ✓ current-model [{}]", current.provider.0)
    );

    p.component.handle_input("\x1b[B");
    let text = output(&p.component, 120);
    assert_eq!(
        row(&text, "current-model"),
        format!("  ✓ current-model [{}]", current.provider.0)
    );
    assert_eq!(
        row(&text, "browsed-model"),
        format!("→   browsed-model [{}]", current.provider.0)
    );
}

#[test]
fn uses_the_configured_save_binding() {
    setup(&[("app.models.save", &["ctrl+r"])]);
    let models = catalog(&["current-model"]);
    let current = models[0].clone();
    let p = picker(ModelSelectorConfig {
        current_model: Some(current.clone()),
        ..config(models)
    });

    assert!(output(&p.component, 120).contains("Ctrl+R to set as default"));
    p.component.handle_input("\x13");
    assert!(p.defaults.borrow().is_empty());
    p.component.handle_input("\x12");
    assert_eq!(*p.defaults.borrow(), vec![current]);
}

#[test]
fn filters_by_id_provider_and_name_and_restores_the_clamped_selection() {
    setup(&[]);
    let p = picker(config(catalog(&["current-model", "browsed-model"])));
    for character in "browsed".chars() {
        p.component.handle_input(&character.to_string());
    }
    let text = output(&p.component, 120);
    assert!(text.contains("browsed-model [faux]"));
    assert!(!text.contains("current-model [faux]"));

    for _ in 0.."browsed".len() {
        p.component.handle_input("\x7f");
    }
    let text = output(&p.component, 120);
    assert!(text.contains("current-model [faux]"));
    assert!(text.contains("browsed-model [faux]"));
    assert_eq!(row(&text, "current-model"), "→   current-model [faux]");
}

#[test]
fn a_default_query_hoists_default_models_above_better_matches() {
    setup(&[]);
    let models = catalog(&["d-model", "d-other"]);
    let default = models[1].clone();
    let p = picker(ModelSelectorConfig {
        default_model: Some(default),
        initial_search: Some("DeFault".to_string()),
        ..config(models.clone())
    });
    let text = output(&p.component, 200);
    assert_eq!(row(&text, "d-other"), "→   d-other [faux] · default");

    let p = picker(ModelSelectorConfig {
        default_model: Some(models[1].clone()),
        initial_search: Some(" d ".to_string()),
        ..config(models)
    });
    let text = output(&p.component, 200);
    assert_eq!(row(&text, "d-other"), "→   d-other [faux] · default");
}

#[test]
fn orders_current_then_default_then_provider() {
    setup(&[]);
    let current = model("beta", "current");
    let default = model("alpha", "default");
    let models = vec![
        model("zeta", "last"),
        model("alpha", "other"),
        current.clone(),
        default.clone(),
    ];
    let p = picker(ModelSelectorConfig {
        current_model: Some(current),
        default_model: Some(default),
        ..config(models)
    });
    let text = output(&p.component, 200);
    let rows: Vec<&str> = text.lines().filter(|line| line.contains(" [")).collect();
    assert!(rows[0].contains("current"));
    assert!(rows[1].contains("default"));
    assert!(rows[2].contains("other"));
    assert!(rows[3].contains("last"));
}

#[test]
fn tab_toggles_between_all_and_scoped_models_and_rewrites_the_scope_line() {
    setup(&[]);
    let models = catalog(&["model-a", "model-b"]);
    let p = picker(ModelSelectorConfig {
        scoped_models: vec![models[0].clone()],
        ..config(models)
    });
    let text = output(&p.component, 120);
    assert!(text.contains("Scope: all | scoped"));
    assert!(text.contains("model-a [faux]"));
    assert!(!text.contains("model-b [faux]"));
    assert!(text.contains("tab scope (all/scoped)"));

    p.component.handle_input("\t");
    let text = output(&p.component, 120);
    assert!(text.contains("model-b [faux]"));

    p.component.handle_input("\t");
    let text = output(&p.component, 120);
    assert!(!text.contains("model-b [faux]"));
}

#[test]
fn shows_no_matches_error_lines_and_refresh_status() {
    setup(&[]);
    let p = picker(ModelSelectorConfig {
        initial_search: Some("zzz".to_string()),
        refresh_status: Some("Refreshing model catalogs…".to_string()),
        ..config(catalog(&["model-a"]))
    });
    let text = output(&p.component, 120);
    assert!(text.contains("No matching models"));
    assert!(text.contains("Refreshing model catalogs…"));

    p.component.set_error_message(Some("boom\nbang"));
    let text = output(&p.component, 120);
    assert!(text.contains("boom"));
    assert!(text.contains("bang"));
    assert!(!text.contains("No matching models"));

    p.component.set_error_message(None);
    assert!(output(&p.component, 120).contains("No matching models"));

    p.component
        .set_refresh_status("Model catalogs refreshed.", true);
    assert!(output(&p.component, 120).contains("Model catalogs refreshed."));
    p.component.set_refresh_status("", true);
    assert!(!output(&p.component, 120).contains("Model catalogs refreshed."));
}

#[test]
fn warns_when_there_are_no_scoped_models_and_tab_does_nothing() {
    setup(&[]);
    let p = picker(config(catalog(&["model-a"])));
    let text = output(&p.component, 200);
    assert!(
        text.contains(
            "Only showing models from configured providers. Use /login to add providers."
        )
    );
    assert!(!text.contains("Scope:"));

    p.component.handle_input("\t");
    let text = output(&p.component, 200);
    assert!(!text.contains("Scope:"));
    assert!(text.contains("model-a [faux]"));
}

#[test]
fn scrolls_a_ten_row_window_and_names_the_selected_model() {
    setup(&[]);
    let ids: Vec<String> = (0..12).map(|index| format!("model-{index}")).collect();
    let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
    let p = picker(config(catalog(&ids)));
    let text = output(&p.component, 120);
    assert!(text.contains("(1/12)"));
    assert!(text.contains("Model Name: model-0"));

    for _ in 0..6 {
        p.component.handle_input("\x1b[B");
    }
    let text = output(&p.component, 120);
    assert!(text.contains("(7/12)"));
    assert!(text.contains("Model Name: model-6"));
}

#[test]
fn confirm_selects_and_cancel_cancels_without_selecting() {
    setup(&[]);
    let p = picker(config(catalog(&["model-a", "model-b"])));
    let first = p.component.render(120);
    assert!(!first.is_empty());

    p.component.handle_input("\x1b[B");
    p.component.handle_input("\r");
    assert_eq!(*p.selected.borrow(), vec![model("faux", "model-b")]);

    p.component.handle_input("\x1b");
    assert_eq!(p.cancelled.get(), 1);
    assert_eq!(*p.selected.borrow(), vec![model("faux", "model-b")]);
}

#[test]
fn initial_search_filters_immediately() {
    setup(&[]);
    let p = picker(ModelSelectorConfig {
        initial_search: Some("browsed".to_string()),
        ..config(catalog(&["current-model", "browsed-model"]))
    });
    let text = output(&p.component, 120);
    assert!(text.contains("browsed-model [faux]"));
    assert!(!text.contains("current-model [faux]"));
    assert_eq!(p.component.get_search_input().get_value(), "browsed");
}

#[test]
fn host_catalog_updates_resort_and_keep_the_current_model_selected() {
    setup(&[]);
    let current = model("beta", "current");
    let p = picker(ModelSelectorConfig {
        current_model: Some(current.clone()),
        ..config(vec![model("alpha", "old")])
    });
    p.component.update_models(
        &[model("zeta", "new"), current.clone()],
        std::slice::from_ref(&current),
    );
    let text = output(&p.component, 200);
    assert_eq!(row(&text, "current"), "→ ✓ current [beta]");
    assert!(text.contains("new [zeta]"));

    p.component.update_models(&[model("gamma", "only")], &[]);
    let text = output(&p.component, 200);
    assert!(text.contains("only [gamma]"));

    let scoped = picker(ModelSelectorConfig {
        scoped_models: vec![model("faux", "model-a")],
        ..config(catalog(&["model-a", "model-b"]))
    });
    assert!(output(&scoped.component, 120).contains("Scope: all | scoped"));
    scoped.component.update_models(
        &catalog(&["model-a", "model-b"]),
        &[model("faux", "model-b")],
    );
    let text = output(&scoped.component, 120);
    assert!(text.contains("Scope: all | scoped"));
    assert!(text.contains("model-b [faux]"));
    assert!(!text.contains("model-a [faux]"));
}

#[test]
fn empty_catalogs_ignore_navigation_and_selection() {
    setup(&[]);
    let p = picker(config(Vec::new()));
    let text = output(&p.component, 120);
    assert!(text.contains("No matching models"));

    p.component.handle_input("\x1b[A");
    p.component.handle_input("\x1b[B");
    p.component.handle_input("\r");
    p.component.handle_input("\x12");
    assert!(p.selected.borrow().is_empty());
    assert!(p.defaults.borrow().is_empty());

    p.component.handle_input("\x1b");
    assert_eq!(p.cancelled.get(), 1);
}

#[test]
fn focus_children_invalidation_and_mouse_route_through_the_component() {
    setup(&[]);
    let p = picker(config(catalog(&["model-a"])));
    assert!(!p.component.children().is_empty());
    assert!(p.component.wants_input());
    p.component.invalidate();
    p.component.set_focused(true);
    assert!(p.component.is_focused());
    let _ = p.component.handle_mouse(&click(0, 20));
}

#[test]
fn search_text_keeps_the_provider_first_and_the_name_optional() {
    assert_eq!(
        get_model_selector_search_text("gpt-5", "openai", Some("GPT 5")),
        "openai openai/gpt-5 openai gpt-5 GPT 5"
    );
    assert_eq!(
        get_model_selector_search_text("gpt-5", "openai", None),
        "openai openai/gpt-5 openai gpt-5"
    );
    assert_eq!(
        get_model_selector_search_text("gpt-5", "openai", Some("")),
        "openai openai/gpt-5 openai gpt-5"
    );
}

#[test]
fn a_current_model_outside_the_catalog_leaves_the_top_row_selected() {
    setup(&[]);
    let p = picker(ModelSelectorConfig {
        current_model: Some(model("absent", "gone")),
        ..config(catalog(&["model-a", "model-b"]))
    });
    let text = output(&p.component, 120);
    assert_eq!(row(&text, "model-a"), "→   model-a [faux]");
    assert!(!text.contains("✓ "));
}

#[test]
fn an_active_query_selects_the_top_row_across_scope_and_catalog_changes() {
    setup(&[]);
    let models = catalog(&["model-a", "model-b"]);
    let p = picker(ModelSelectorConfig {
        scoped_models: models.clone(),
        ..config(models.clone())
    });
    for character in "model".chars() {
        p.component.handle_input(&character.to_string());
    }
    p.component.handle_input("\x1b[B");
    p.component.handle_input("\t");
    let text = output(&p.component, 120);
    assert_eq!(row(&text, "model-a"), "→   model-a [faux]");

    p.component.update_models(&models, &models);
    let text = output(&p.component, 120);
    assert_eq!(row(&text, "model-a"), "→   model-a [faux]");
    assert!(row(&text, "model-b").starts_with("    "));
}

#[test]
fn empty_host_messages_clear_their_lines_and_the_hint_is_optional() {
    setup(&[]);
    let p = picker_with(config(catalog(&["model-a"])), false);
    p.component.set_error_message(Some(""));
    assert!(output(&p.component, 120).contains("Model Name: model-a"));
    assert!(!output(&p.component, 120).contains("set as default"));

    let initial = picker(ModelSelectorConfig {
        initial_search: Some(String::new()),
        ..config(catalog(&["model-a"]))
    });
    assert!(output(&initial.component, 120).contains("model-a [faux]"));
    assert!(format!("{:?}", initial.component).contains("current_model"));
}
