//! The custom-entry row: renderer output, the expanded hand-off, the panic
//! fallback, and the no-content case.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use pi_coding_agent::modes::interactive::components::custom_entry::{
    CustomEntryComponent, EntryRenderOptions, EntryRenderer,
};
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::session_manager::entries::{CustomEntry, SessionEntryBase};
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::components::Text;
use pi_tui::tui::Component;

fn entry(custom_type: &str) -> CustomEntry {
    CustomEntry {
        custom_type: custom_type.to_string(),
        data: None,
        base: SessionEntryBase::default(),
        extras: serde_json::Map::new(),
    }
}

fn text(component: &CustomEntryComponent) -> String {
    strip_ansi(&component.render(60).join("\n"))
}

#[test]
fn renderer_output_is_the_only_content() {
    init_theme(Some("dark"), false);
    let renderer: EntryRenderer = Arc::new(|_entry, _options: &EntryRenderOptions, _theme| {
        Some(Rc::new(Text::with_padding("rendered entry", 0, 0)))
    });
    let component = CustomEntryComponent::new(entry("demo"), renderer);
    assert!(component.has_content());
    assert!(text(&component).contains("rendered entry"));
}

#[test]
fn declined_renderer_adds_no_row() {
    init_theme(Some("dark"), false);
    let renderer: EntryRenderer = Arc::new(|_entry, _options: &EntryRenderOptions, _theme| None);
    let component = CustomEntryComponent::new(entry("demo"), renderer);
    assert!(!component.has_content());
    assert!(component.render(60).is_empty());
}

#[test]
fn expanded_state_reaches_the_renderer_and_repaints() {
    init_theme(Some("dark"), false);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let renderer: EntryRenderer = {
        let seen = Arc::clone(&seen);
        Arc::new(move |_entry, options: &EntryRenderOptions, _theme| {
            seen.lock().expect("lock").push(options.expanded);
            let label = if options.expanded {
                "expanded"
            } else {
                "collapsed"
            };
            Some(Rc::new(Text::with_padding(label, 0, 0)))
        })
    };
    let component = CustomEntryComponent::new(entry("demo"), renderer);
    assert!(text(&component).contains("collapsed"));

    component.set_expanded(true);
    assert!(component.is_expanded());
    assert!(text(&component).contains("expanded"));
    assert_eq!(*seen.lock().expect("lock"), vec![false, true]);

    component.set_expanded(true);
    assert_eq!(*seen.lock().expect("lock"), vec![false, true]);
}

#[test]
fn panicking_renderer_degrades_to_the_error_box() {
    init_theme(Some("dark"), false);
    let renderer: EntryRenderer = Arc::new(
        |_entry, _options: &EntryRenderOptions, _theme| -> Option<Rc<dyn Component>> {
            std::panic::resume_unwind(Box::new("boom"))
        },
    );
    let component = CustomEntryComponent::new(entry("demo-type"), renderer);
    assert!(component.has_content());
    let output = text(&component);
    assert!(output.contains("[demo-type] renderer failed: boom"));
}

#[test]
fn invalidate_rebuilds_from_the_renderer() {
    init_theme(Some("dark"), false);
    let calls = Arc::new(Mutex::new(0_usize));
    let renderer: EntryRenderer = {
        let calls = Arc::clone(&calls);
        Arc::new(move |_entry, _options: &EntryRenderOptions, _theme| {
            *calls.lock().expect("lock") += 1;
            Some(Rc::new(Text::with_padding("entry", 0, 0)))
        })
    };
    let component = CustomEntryComponent::new(entry("demo"), renderer);
    Component::invalidate(&component);
    assert_eq!(*calls.lock().expect("lock"), 2);
}
