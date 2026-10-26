//! Headless selector fixtures; callbacks record decisions without persisting them.

#![expect(
    unreachable_pub,
    reason = "the shared fixture module is private in each selector suite"
)]
#![allow(
    dead_code,
    reason = "each selector suite uses only its own subset of fixtures"
)]

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_coding_agent::core::keybindings::keybindings;
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::keybindings::{KeybindingsConfig, KeybindingsManager, set_keybindings};
use pi_tui::tui::{Component, TuiMouseButton, TuiMouseEvent, TuiMouseEventType};

/// Install app bindings with optional overrides, upstream's beforeEach.
pub fn setup(overrides: &[(&str, &[&str])]) {
    init_theme(Some("dark"), false);
    let mut config = KeybindingsConfig::new();
    for (action, keys) in overrides {
        config = config.bind(*action, keys.iter().copied());
    }
    set_keybindings(KeybindingsManager::with_user_bindings(
        keybindings(),
        config,
    ));
}

/// Render without ANSI decoration, like upstream's stripAnsi assertions.
pub fn output(component: &dyn Component, width: usize) -> String {
    strip_ansi(&component.render(width).join("\n"))
}

/// The callback and its owned-value log.
pub type Recording<T> = (Rc<RefCell<Vec<T>>>, Rc<dyn Fn(T)>);

/// Record typed selections without a settings or trust store.
pub fn recording<T: 'static>() -> Recording<T> {
    let values = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&values);
    (
        values,
        Rc::new(move |value| captured.borrow_mut().push(value)),
    )
}

/// Count cancel callbacks.
pub fn cancellation() -> (Rc<Cell<usize>>, Rc<dyn Fn()>) {
    let count = Rc::new(Cell::new(0));
    let captured = Rc::clone(&count);
    (count, Rc::new(move || captured.set(captured.get() + 1)))
}

/// A left-button click in a component with the given rendered dimensions.
pub const fn click(y: u16, height: u16) -> TuiMouseEvent {
    TuiMouseEvent {
        event_type: TuiMouseEventType::Click,
        button: TuiMouseButton::Left,
        x: 0,
        y,
        screen_x: 0,
        screen_y: y,
        width: 80,
        height,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: Some(1),
    }
}
