//! The first-run setup suite: upstream's `first-time-setup.ts` at pin
//! `60e7e76` (#131), plus boundary coverage for the detected-theme start,
//! previews, clamping, both step transitions, and dismissal.
//!
//! The predicate cases from `first-time-setup.test.ts` live beside the
//! predicate in `cli/startup_ui/tests.rs`; the analytics block lives in
//! `settings_boundary.rs`, where the settings slice ported it.

#[path = "common/selectors.rs"]
mod selectors;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_coding_agent::modes::interactive::components::first_time_setup::{
    FirstTimeSetupComponent, FirstTimeSetupOptions, FirstTimeSetupResult,
};
use pi_coding_agent::modes::interactive::theme::theme::TerminalTheme;
use selectors::{cancellation, output, setup};

struct Fixture {
    previews: Rc<RefCell<Vec<TerminalTheme>>>,
    submissions: Rc<RefCell<Vec<FirstTimeSetupResult>>>,
    cancel_count: Rc<Cell<usize>>,
    options: FirstTimeSetupOptions,
}

fn fixture(detected_theme: TerminalTheme) -> Fixture {
    let previews = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&previews);
    let on_theme_preview: Rc<dyn Fn(TerminalTheme)> =
        Rc::new(move |theme| captured.borrow_mut().push(theme));
    let submissions = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&submissions);
    let on_submit: Rc<dyn Fn(FirstTimeSetupResult)> =
        Rc::new(move |result| captured.borrow_mut().push(result));
    let (cancel_count, on_cancel) = cancellation();
    Fixture {
        previews,
        submissions,
        cancel_count,
        options: FirstTimeSetupOptions {
            detected_theme,
            on_theme_preview,
            on_submit,
            on_cancel,
        },
    }
}

#[test]
fn opens_on_the_detected_theme() {
    setup(&[]);
    let fixture = fixture(TerminalTheme::Light);
    let component = FirstTimeSetupComponent::new(fixture.options);

    let rendered = output(&component, 80);
    assert!(
        rendered.contains("Welcome to pi, the minimal coding agent."),
        "{rendered}"
    );
    assert!(rendered.contains("Pick a theme."), "{rendered}");
    assert!(
        rendered.contains("Detected system appearance: light"),
        "{rendered}"
    );
    assert!(rendered.contains("→ Light"), "{rendered}");
    assert!(rendered.contains("  Dark"), "{rendered}");
    assert!(rendered.contains("continue"), "{rendered}");
    assert!(rendered.contains("skip setup"), "{rendered}");
    assert!(rendered.contains("██████"), "{rendered}");
}

#[test]
fn moving_the_selection_previews_and_clamps_at_the_top() {
    setup(&[]);
    let fixture = fixture(TerminalTheme::Light);
    let component = FirstTimeSetupComponent::new(fixture.options);

    component.handle_input("k");
    component.handle_input("k");
    assert_eq!(*fixture.previews.borrow(), vec![TerminalTheme::Dark]);
    assert!(output(&component, 80).contains("→ Dark"));

    component.handle_input("j");
    component.handle_input("j");
    assert_eq!(
        *fixture.previews.borrow(),
        vec![TerminalTheme::Dark, TerminalTheme::Light]
    );
    assert!(output(&component, 80).contains("→ Light"));
}

#[test]
fn arrow_keys_navigate_like_j_and_k() {
    setup(&[]);
    let fixture = fixture(TerminalTheme::Dark);
    let component = FirstTimeSetupComponent::new(fixture.options);

    component.handle_input("\x1b[B");
    assert_eq!(*fixture.previews.borrow(), vec![TerminalTheme::Light]);
    component.handle_input("\x1b[A");
    assert_eq!(
        *fixture.previews.borrow(),
        vec![TerminalTheme::Light, TerminalTheme::Dark]
    );
}

#[test]
fn confirm_advances_to_analytics_then_submits_the_decision() {
    setup(&[]);
    let fixture = fixture(TerminalTheme::Light);
    let component = FirstTimeSetupComponent::new(fixture.options);

    component.handle_input("\r");
    let rendered = output(&component, 80);
    assert!(
        rendered.contains("Opt-in to anonymous usage data sharing?"),
        "{rendered}"
    );
    assert!(rendered.contains("finish"), "{rendered}");
    assert!(
        rendered.contains("→ Share anonymous usage data"),
        "{rendered}"
    );

    component.handle_input("\n");
    assert_eq!(
        *fixture.submissions.borrow(),
        vec![FirstTimeSetupResult {
            theme: TerminalTheme::Light,
            share_analytics: true,
        }]
    );
}

#[test]
fn analytics_selection_is_clamped_and_submitted() {
    setup(&[]);
    let fixture = fixture(TerminalTheme::Dark);
    let component = FirstTimeSetupComponent::new(fixture.options);

    component.handle_input("\r");
    component.handle_input("j");
    component.handle_input("j");
    assert!(
        output(&component, 80).contains("→ Don't share"),
        "the analytics row clamps at the last option"
    );

    component.handle_input("\r");
    assert_eq!(
        *fixture.submissions.borrow(),
        vec![FirstTimeSetupResult {
            theme: TerminalTheme::Dark,
            share_analytics: false,
        }]
    );
}

#[test]
fn cancel_skips_setup_from_either_step() {
    setup(&[]);
    let fixture = fixture(TerminalTheme::Dark);
    let component = FirstTimeSetupComponent::new(fixture.options);

    component.handle_input("\x1b");
    component.handle_input("\r");
    component.handle_input("\x1b");
    assert_eq!(fixture.cancel_count.get(), 2);
    assert!(fixture.submissions.borrow().is_empty());
}

#[test]
fn custom_confirm_binding_replaces_enter() {
    setup(&[("tui.select.confirm", &["space"])]);
    let fixture = fixture(TerminalTheme::Dark);
    let component = FirstTimeSetupComponent::new(fixture.options);

    component.handle_input(" ");
    assert!(
        output(&component, 80).contains("Opt-in to anonymous usage data sharing?"),
        "space advances with the override installed"
    );
}
