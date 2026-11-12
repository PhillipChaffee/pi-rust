//! The extension text prompt suite: upstream's `extension-input.ts` at pin
//! `60e7e76` (#131), plus boundary coverage for submit, cancel, the countdown
//! suffix, expiry, and focus propagation.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

#[path = "common/selectors.rs"]
mod selectors;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_coding_agent::modes::interactive::components::extension_input::{
    ExtensionInputComponent, ExtensionInputOptions,
};
use pi_tui::tui::{Component, Focusable};
use selectors::{cancellation, output, setup};

struct Fixture {
    submissions: Rc<RefCell<Vec<String>>>,
    cancel_count: Rc<Cell<usize>>,
    on_submit: Rc<dyn Fn(String)>,
    on_cancel: Rc<dyn Fn()>,
}

fn fixture() -> Fixture {
    let submissions = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&submissions);
    let on_submit: Rc<dyn Fn(String)> = Rc::new(move |value| captured.borrow_mut().push(value));
    let (cancel_count, on_cancel) = cancellation();
    Fixture {
        submissions,
        cancel_count,
        on_submit,
        on_cancel,
    }
}

fn component(fixture: &Fixture, opts: ExtensionInputOptions) -> ExtensionInputComponent {
    ExtensionInputComponent::new(
        "Project name",
        Some("ignored placeholder"),
        Rc::clone(&fixture.on_submit),
        Rc::clone(&fixture.on_cancel),
        opts,
    )
}

#[test]
fn renders_the_title_and_key_hints() {
    setup(&[]);
    let fixture = fixture();
    let input = component(&fixture, ExtensionInputOptions::default());

    let rendered = output(&input, 80);
    assert!(rendered.contains("Project name"), "{rendered}");
    assert!(rendered.contains("submit"), "{rendered}");
    assert!(rendered.contains("cancel"), "{rendered}");
    assert!(rendered.contains("─"), "{rendered}");
}

#[test]
fn submit_returns_the_typed_value() {
    setup(&[]);
    let fixture = fixture();
    let input = component(&fixture, ExtensionInputOptions::default());

    input.handle_input("hello world");
    input.handle_input("\r");
    assert_eq!(*fixture.submissions.borrow(), vec!["hello world"]);
}

#[test]
fn cancel_fires_without_a_value() {
    setup(&[]);
    let fixture = fixture();
    let input = component(&fixture, ExtensionInputOptions::default());

    input.handle_input("discard me");
    input.handle_input("\x1b");
    assert_eq!(fixture.cancel_count.get(), 1);
    assert!(fixture.submissions.borrow().is_empty());
}

#[test]
fn countdown_appends_the_remaining_seconds() {
    setup(&[]);
    let fixture = fixture();
    let input = component(
        &fixture,
        ExtensionInputOptions {
            request_render: Some(Rc::new(|| {})),
            timeout_ms: Some(2_000),
        },
    );

    let countdown = input.countdown().expect("armed countdown");
    assert!(output(input.title_text().as_ref(), 80).contains("Project name (2s)"));
    countdown.tick();
    assert!(output(input.title_text().as_ref(), 80).contains("Project name (1s)"));
    countdown.tick();
    assert!(output(input.title_text().as_ref(), 80).contains("Project name (0s)"));
    assert_eq!(fixture.cancel_count.get(), 1);
    assert!(!countdown.is_armed());
}

#[test]
fn timeout_without_a_render_request_leaves_the_countdown_off() {
    setup(&[]);
    let fixture = fixture();
    let input = component(
        &fixture,
        ExtensionInputOptions {
            request_render: None,
            timeout_ms: Some(2_000),
        },
    );

    assert!(input.countdown().is_none());
    assert!(!output(&input, 80).contains("(2s)"));
}

#[test]
fn dispose_disarms_the_countdown() {
    setup(&[]);
    let fixture = fixture();
    let input = component(
        &fixture,
        ExtensionInputOptions {
            request_render: Some(Rc::new(|| {})),
            timeout_ms: Some(1_000),
        },
    );

    let countdown = input.countdown().expect("armed countdown");
    input.dispose();
    assert!(!countdown.is_armed());
    countdown.tick();
    assert_eq!(fixture.cancel_count.get(), 0);
}

#[test]
fn focus_reaches_the_focusable_contract() {
    setup(&[]);
    let fixture = fixture();
    let input = component(&fixture, ExtensionInputOptions::default());

    assert!(!input.is_focused());
    input.set_focused(true);
    assert!(input.is_focused());
    assert!(input.as_focusable().is_some());
    input.set_focused(false);
    assert!(!input.is_focused());
}
