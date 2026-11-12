//! The extension-selector suite: upstream's `extension-selector.ts` and
//! `countdown-timer.ts` at pin `60e7e76` (#213), plus boundary coverage for
//! navigation, callbacks, the expand binding, and the countdown seam.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

#[path = "common/selectors.rs"]
mod selectors;

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_coding_agent::modes::interactive::components::extension_selector::{
    ExtensionSelectorComponent, ExtensionSelectorOptions,
};
use pi_tui::tui::Component;
use selectors::{cancellation, output, setup};

struct Fixture {
    on_select: Rc<dyn Fn(&str)>,
    on_cancel: Rc<dyn Fn()>,
    selected: Rc<RefCell<Vec<String>>>,
    cancel_count: Rc<Cell<usize>>,
}

fn callbacks() -> Fixture {
    let selected = Rc::new(RefCell::new(Vec::<String>::new()));
    let captured = Rc::clone(&selected);
    let on_select: Rc<dyn Fn(&str)> =
        Rc::new(move |value| captured.borrow_mut().push(value.to_string()));
    let (cancel_count, on_cancel) = cancellation();
    Fixture {
        on_select,
        on_cancel,
        selected,
        cancel_count,
    }
}

fn component(
    options: Vec<&str>,
    fixture: &Fixture,
    opts: ExtensionSelectorOptions,
) -> ExtensionSelectorComponent {
    ExtensionSelectorComponent::new(
        "Choose an option",
        options.into_iter().map(str::to_string).collect(),
        Rc::clone(&fixture.on_select),
        Rc::clone(&fixture.on_cancel),
        opts,
    )
}

#[test]
fn renders_title_options_and_hints() {
    setup(&[]);
    let fixture = callbacks();
    let selector = component(
        vec!["Yes", "No"],
        &fixture,
        ExtensionSelectorOptions::default(),
    );

    let rendered = output(&selector, 80);
    assert!(rendered.contains("Choose an option"), "{rendered}");
    assert!(rendered.contains("→ Yes"), "{rendered}");
    assert!(rendered.contains("  No"), "{rendered}");
    assert!(rendered.contains("↑↓ navigate"), "{rendered}");
    assert!(rendered.contains("select"), "{rendered}");
    assert!(rendered.contains("cancel"), "{rendered}");
    assert!(rendered.contains("─"), "{rendered}");
}

#[test]
fn navigation_moves_selection_and_clamps_at_the_ends() {
    setup(&[]);
    let fixture = callbacks();
    let selector = component(
        vec!["One", "Two", "Three"],
        &fixture,
        ExtensionSelectorOptions::default(),
    );

    selector.handle_input("k");
    assert!(output(&selector, 80).contains("→ One"));

    selector.handle_input("j");
    assert!(output(&selector, 80).contains("→ Two"));
    selector.handle_input("j");
    selector.handle_input("j");
    assert!(output(&selector, 80).contains("→ Three"));

    selector.handle_input("\x1b[A");
    assert!(output(&selector, 80).contains("→ Two"));
    selector.handle_input("\x1b[B");
    assert!(output(&selector, 80).contains("→ Three"));
}

#[test]
fn confirm_selects_the_highlighted_option() {
    setup(&[]);
    let fixture = callbacks();
    let selector = component(
        vec!["Alpha", "Beta"],
        &fixture,
        ExtensionSelectorOptions::default(),
    );

    selector.handle_input("\r");
    selector.handle_input("j");
    selector.handle_input("\n");
    assert_eq!(
        *fixture.selected.borrow(),
        vec!["Alpha".to_string(), "Beta".to_string()]
    );
}

#[test]
fn cancel_fires_on_escape() {
    setup(&[]);
    let fixture = callbacks();
    let selector = component(vec!["Alpha"], &fixture, ExtensionSelectorOptions::default());

    selector.handle_input("\x1b");
    assert_eq!(fixture.cancel_count.get(), 1);
}

#[test]
fn expand_binding_fires_the_toggle() {
    setup(&[]);
    let fixture = callbacks();
    let toggles = Rc::new(Cell::new(0));
    let captured = Rc::clone(&toggles);
    let selector = component(
        vec!["Alpha"],
        &fixture,
        ExtensionSelectorOptions {
            on_toggle_tools_expanded: Some(Rc::new(move || {
                captured.set(captured.get() + 1);
            })),
            ..ExtensionSelectorOptions::default()
        },
    );

    selector.handle_input("\x0f");
    assert_eq!(toggles.get(), 1);
}

#[test]
fn countdown_ticks_and_expires_into_cancel() {
    setup(&[]);
    let fixture = callbacks();
    let renders = Rc::new(Cell::new(0));
    let captured = Rc::clone(&renders);
    let selector = component(
        vec!["Alpha"],
        &fixture,
        ExtensionSelectorOptions {
            request_render: Some(Rc::new(move || captured.set(captured.get() + 1))),
            timeout_ms: Some(1500),
            ..ExtensionSelectorOptions::default()
        },
    );

    assert!(output(&selector, 80).contains("(2s)"));
    let countdown = selector.countdown().expect("countdown armed");
    countdown.tick();
    assert!(output(&selector, 80).contains("(1s)"));
    countdown.tick();
    assert!(output(&selector, 80).contains("(0s)"));
    assert_eq!(renders.get(), 2);
    assert_eq!(fixture.cancel_count.get(), 1);
    assert!(!countdown.is_armed());

    countdown.tick();
    assert_eq!(renders.get(), 2);
    assert_eq!(fixture.cancel_count.get(), 1);
}

#[test]
fn timeout_without_request_render_leaves_the_countdown_unarmed() {
    setup(&[]);
    let fixture = callbacks();
    let selector = component(
        vec!["Alpha"],
        &fixture,
        ExtensionSelectorOptions {
            timeout_ms: Some(1000),
            ..ExtensionSelectorOptions::default()
        },
    );
    assert!(selector.countdown().is_none());

    let fixture = callbacks();
    let zero_timeout = component(
        vec!["Alpha"],
        &fixture,
        ExtensionSelectorOptions {
            request_render: Some(Rc::new(|| {})),
            timeout_ms: Some(0),
            ..ExtensionSelectorOptions::default()
        },
    );
    assert!(zero_timeout.countdown().is_none());
}

#[test]
fn dispose_stops_the_countdown() {
    setup(&[]);
    let fixture = callbacks();
    let selector = component(
        vec!["Alpha"],
        &fixture,
        ExtensionSelectorOptions {
            request_render: Some(Rc::new(|| {})),
            timeout_ms: Some(3000),
            ..ExtensionSelectorOptions::default()
        },
    );
    let countdown = selector.countdown().expect("countdown armed");

    selector.dispose();
    assert!(!countdown.is_armed());
    countdown.tick();
    assert_eq!(fixture.cancel_count.get(), 0);
}

#[test]
fn empty_options_ignore_navigation_and_confirm() {
    setup(&[]);
    let fixture = callbacks();
    let selector = component(Vec::new(), &fixture, ExtensionSelectorOptions::default());

    selector.handle_input("j");
    selector.handle_input("\r");
    let rendered = output(&selector, 80);
    assert!(rendered.contains("Choose an option"), "{rendered}");
    assert!(fixture.selected.borrow().is_empty());
}

#[test]
fn component_surface_answers_contract_queries() {
    setup(&[]);
    let fixture = callbacks();
    let selector = component(
        vec!["Alpha"],
        &fixture,
        ExtensionSelectorOptions {
            request_render: Some(Rc::new(|| {})),
            timeout_ms: Some(2000),
            ..ExtensionSelectorOptions::default()
        },
    );

    assert!(selector.wants_input());
    assert_eq!(selector.children().len(), 9);
    selector.invalidate();
    let as_component: &dyn Component = &selector;
    as_component.handle_input("j");
    assert!(output(&selector, 80).contains("→ Alpha"));
    assert!(format!("{selector:?}").contains("options"));
    assert!(format!("{:?}", ExtensionSelectorOptions::default()).contains("timeout_ms"));
    let countdown = selector.countdown().expect("countdown armed");
    assert!(format!("{countdown:?}").contains("remaining_seconds"));
    assert_eq!(countdown.remaining_seconds(), 2);
    countdown.tick();
    assert_eq!(countdown.remaining_seconds(), 1);
}
