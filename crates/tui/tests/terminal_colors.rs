//! Port of `packages/tui/test/terminal-colors.test.ts` — 1:1 against
//! upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#41): the parser
//! suites and the `TUI.queryTerminalBackgroundColor` describe block, which
//! drives `Tui::query_terminal_background_color` through the shared harness.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use pi_tui::terminal_colors::{
    RgbColor, TerminalColorScheme, is_osc11_background_color_response,
    parse_osc11_background_color, parse_terminal_color_scheme_report,
};
use pi_tui::tui::{Component, Tui, TuiConfig};

use tui_support::{FocusableOverlay, VirtualTerminal};

#[test]
fn parses_16_bit_osc_11_rgb_responses() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:0000/8000/ffff\x07"),
        Some(RgbColor {
            r: 0,
            g: 128,
            b: 255
        })
    );
}

#[test]
fn parses_osc_11_hex_responses() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;#ffffff\x1b\\"),
        Some(RgbColor {
            r: 255,
            g: 255,
            b: 255
        })
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;#000000\x07"),
        Some(RgbColor { r: 0, g: 0, b: 0 })
    );
}

#[test]
fn rejects_non_strict_osc_11_responses() {
    assert_eq!(parse_osc11_background_color("x\x1b]11;#ffffff\x07"), None);
    assert_eq!(parse_osc11_background_color("\x1b]10;#ffffff\x07"), None);
    assert_eq!(parse_osc11_background_color("\x1b]11;#ffffff\x07x"), None);
}

#[test]
fn parses_color_scheme_reports() {
    assert_eq!(
        parse_terminal_color_scheme_report("\x1b[?997;1n"),
        Some(TerminalColorScheme::Dark)
    );
    assert_eq!(
        parse_terminal_color_scheme_report("\x1b[?997;2n"),
        Some(TerminalColorScheme::Light)
    );
    assert_eq!(
        parse_terminal_color_scheme_report("\x1b[?997;2n\x1b[?997;1n\x1b[?997;1n"),
        Some(TerminalColorScheme::Dark)
    );
    assert_eq!(
        parse_terminal_color_scheme_report("\x1b[?997;1n\x1b[?997;2n\x1b[?997;2n"),
        Some(TerminalColorScheme::Light)
    );
    assert_eq!(parse_terminal_color_scheme_report("\x1b[?997;3n"), None);
    assert_eq!(parse_terminal_color_scheme_report("\x1b[?996n"), None);
    assert_eq!(parse_terminal_color_scheme_report("x\x1b[?997;1n"), None);
}

/// Boundary: the strict-response predicate and the parser agree, and
/// unparsable replies are rejected without tripping the trim.
#[test]
fn osc11_strictness_matches_the_response_gate() {
    assert!(is_osc11_background_color_response("\x1b]11;#ffffff\x1b\\"));
    assert!(is_osc11_background_color_response(
        "\x1b]11;  #ffffff  \x07"
    ));
    assert!(!is_osc11_background_color_response("\x1b]11;#ffffff"));
    assert!(!is_osc11_background_color_response("\x1b]11;\x07x"));

    assert_eq!(
        parse_osc11_background_color("\x1b]11;not-a-color\x07"),
        None,
        "a strict response with an unparsable value resolves nothing"
    );
}

/// Boundary: the `XParseColor` form scales each channel from its own radix.
#[test]
fn parses_osc_11_rgb_channels_of_mixed_widths() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:ff/80/0\x07"),
        Some(RgbColor {
            r: 255,
            g: 128,
            b: 0
        })
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:zz/80/ff\x07"),
        None,
        "non-hex channel is rejected"
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:ff/80\x07"),
        None,
        "two channels are not a color"
    );
}

/// Boundary: `rgba:` prefixes are stripped exactly as upstream's anchored
/// replace, uppercase included.
#[test]
fn strips_rgba_prefixes_case_insensitively() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;RGB:ff/00/ff\x07"),
        Some(RgbColor {
            r: 255,
            g: 0,
            b: 255
        })
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;RGBA:ff/00/ff\x07"),
        Some(RgbColor {
            r: 255,
            g: 0,
            b: 255
        })
    );
}

/// Boundary: a 12-digit hex response scales each 16-bit channel, and a
/// non-hex tail after `#` rejects.
#[test]
fn parses_osc_11_sixteen_bit_hex_and_rejects_bad_hex() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;#ffff8000ffff\x1b\\"),
        Some(RgbColor {
            r: 255,
            g: 128,
            b: 255
        })
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;#ffffff0\x07"),
        None,
        "seven digits are neither the 6 nor the 12 digit form"
    );
    assert_eq!(parse_osc11_background_color("\x1b]11;#\x07"), None);
}

/// Boundary: absurdly long channels exceed the radix the port scales from
/// and reject, where upstream's `parseInt` would answer with float garbage.
#[test]
fn overlong_hex_channels_are_rejected() {
    let long_channel = "f".repeat(17);
    assert_eq!(
        parse_osc11_background_color(&format!("\x1b]11;rgb:{long_channel}/0/0\x07")),
        None
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:ffff8000ffff/0/0\x07"),
        Some(RgbColor { r: 255, g: 0, b: 0 }),
        "a 12-digit channel is a valid 16-bit value; upstream ignores any fourth channel"
    );
}

// === the OSC 11 query block, upstream's `TUI.queryTerminalBackgroundColor` ===

/// The OSC 11 sinks: the focused `FocusableOverlay` recorder child and the
/// recording input listener, the wiring every query test performs.
fn wire_query_sinks(tui: &Tui) -> (Rc<FocusableOverlay>, Rc<RefCell<Vec<String>>>) {
    let recorder = FocusableOverlay::new(&[]);
    let focus_target: Rc<dyn Component> = recorder.clone();
    tui.add_child(Rc::clone(&focus_target));
    tui.set_focus(Some(focus_target));
    let listener_inputs = Rc::new(RefCell::new(Vec::<String>::new()));
    let sink = Rc::clone(&listener_inputs);
    tui.add_input_listener(Rc::new(move |data| {
        sink.borrow_mut().push(data.to_string());
        None
    }));
    (recorder, listener_inputs)
}

/// A wired harness started against the default clock, the shape upstream's
/// listener-and-recorder tests construct.
struct QueryHarness {
    terminal: VirtualTerminal,
    tui: Rc<Tui>,
    recorder: Rc<FocusableOverlay>,
    listener_inputs: Rc<RefCell<Vec<String>>>,
}

fn query_harness() -> QueryHarness {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let (recorder, listener_inputs) = wire_query_sinks(&tui);
    tui.start();
    QueryHarness {
        terminal,
        tui,
        recorder,
        listener_inputs,
    }
}

/// The query tests' settled read: the reply path sends synchronously, so an
/// empty channel is a port regression that must panic loudly.
#[expect(
    clippy::expect_used,
    reason = "the reply is sent synchronously before the receive runs; an empty channel is a port regression that must panic loudly"
)]
fn settled(query: &std::sync::mpsc::Receiver<Option<RgbColor>>) -> Option<RgbColor> {
    query.try_recv().expect("the reply settles the query")
}

/// No listener or focused-component dispatch happened, upstream's
/// `deepStrictEqual(listenerInputs, [])` / `deepStrictEqual(component.inputs, [])` pair.
fn assert_no_dispatch(recorder: &FocusableOverlay, listener_inputs: &[String]) {
    assert!(listener_inputs.is_empty());
    assert!(recorder.inputs().is_empty());
}

/// The white-reply half of the plain-query tests: the `#ffffff` reply and
/// its settled expectation, shared by the write-probe and the
/// non-matching-dispatch tests.
fn send_white_reply(
    terminal: &VirtualTerminal,
    query: &std::sync::mpsc::Receiver<Option<RgbColor>>,
) {
    terminal.send_input("\x1b]11;#ffffff\x07");
    assert_eq!(
        settled(query),
        Some(RgbColor {
            r: 255,
            g: 255,
            b: 255
        })
    );
}

#[test]
fn writes_osc_11_query_and_resolves_with_the_parsed_rgb_reply() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.start();

    let query = tui.query_terminal_background_color(1000);
    assert!(terminal.write_log().contains("\x1b]11;?\x07"));

    send_white_reply(&terminal, &query);

    tui_support::stop(&tui);
}

#[test]
fn consumes_osc_11_replies_before_input_listeners_and_focused_component_dispatch() {
    let QueryHarness {
        terminal,
        tui,
        recorder,
        listener_inputs,
    } = query_harness();

    let query = tui.query_terminal_background_color(1000);
    terminal.send_input("\x1b]11;#000000\x07");

    assert_eq!(settled(&query), Some(RgbColor { r: 0, g: 0, b: 0 }));
    assert_no_dispatch(&recorder, &listener_inputs.borrow());

    tui_support::stop(&tui);
}

#[test]
fn consumes_unparsable_strict_osc_11_replies_and_resolves_undefined() {
    let QueryHarness {
        terminal,
        tui,
        recorder,
        listener_inputs,
    } = query_harness();

    let query = tui.query_terminal_background_color(1000);
    terminal.send_input("\x1b]11;not-a-color\x07");

    assert_eq!(settled(&query), None);
    assert_no_dispatch(&recorder, &listener_inputs.borrow());

    tui_support::stop(&tui);
}

#[test]
fn dispatches_non_matching_input_normally_while_waiting_for_an_osc_11_reply() {
    let QueryHarness {
        terminal,
        tui,
        recorder,
        listener_inputs,
    } = query_harness();

    let query = tui.query_terminal_background_color(1000);
    terminal.send_input("x");
    assert!(
        query.try_recv().is_err(),
        "non-matching input must not settle the query"
    );
    assert_eq!(*listener_inputs.borrow(), vec!["x".to_string()]);
    assert_eq!(recorder.inputs(), vec!["x".to_string()]);

    send_white_reply(&terminal, &query);

    tui_support::stop(&tui);
}

#[test]
fn keeps_consuming_a_late_osc_11_reply_after_timeout() {
    let terminal = VirtualTerminal::new(80, 24);
    let now = Rc::new(Cell::new(Instant::now()));
    let clock_now = Rc::clone(&now);
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal.clone())),
        renderer: Some(tui_support::test_renderer()),
        clock: Some(Box::new(move || clock_now.get())),
        ..TuiConfig::default()
    });
    let (recorder, listener_inputs) = wire_query_sinks(&tui);
    tui.start();

    let query = tui.query_terminal_background_color(1);
    now.set(now.get() + Duration::from_millis(5));
    tui_support::render_and_flush(&tui);
    assert_eq!(settled(&query), None);

    terminal.send_input("\x1b]11;#ffffff\x07");
    assert_no_dispatch(&recorder, &listener_inputs.borrow());

    tui_support::stop(&tui);
}
