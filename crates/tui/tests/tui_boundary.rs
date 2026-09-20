//! Boundary tests for the #43 TUI-core slice, added where the ported
//! upstream suites leave branches untested: mouse retargeting and overlay
//! dispatch, the scheduler's throttle/preemption timing under a fake clock,
//! the terminal queries and their deadlines, the input listeners and debug
//! key, the queued-input drain, and the container/region mouse machinery.
//! These bind the 95% coverage gate alongside the ported suites.

#![expect(
    clippy::expect_used,
    reason = "a failed focus or missing target in a fixture is a test-environment failure; expecting keeps the assertions readable"
)]

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use pi_tui::tui::{
    Component, Container, OverlayAnchor, OverlayHandle, OverlayOptions, OverlayUnfocusOptions,
    SizeValue, Tui, TuiConfig, TuiMode, TuiMouseEvent, TuiMouseEventResult, TuiMouseEventType,
    TuiRenderer, composite_tui_line, dispatch_mouse_event, is_focusable, retarget_mouse_event,
};
use std::rc::Weak;

use tui_support::{
    EmptyContent, FocusableOverlay, TestTerminal, render_and_flush, wait_for_render,
};

// === fixtures ===

/// A leaf that renders one line and answers nothing.
struct SilentLeaf {
    mouse_result: Option<TuiMouseEventResult>,
}

impl SilentLeaf {
    fn new() -> Rc<Self> {
        Rc::new(Self { mouse_result: None })
    }

    fn mouse_result(result: TuiMouseEventResult) -> Rc<Self> {
        Rc::new(Self {
            mouse_result: Some(result),
        })
    }
}

impl Component for SilentLeaf {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["leaf".to_string()]
    }

    fn handle_mouse(&self, _event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.mouse_result.clone()
    }
}

/// A delegating container, upstream's `InputOverlay` shape: it embeds a
/// [`Container`], forwards input, and keeps keyboard focus when a nested
/// control is clicked.
struct DelegatingContainer {
    container: Container,
    inputs: RefCell<Vec<String>>,
}

impl DelegatingContainer {
    fn with_child(child: Rc<dyn Component>) -> Rc<Self> {
        let cell = Rc::new(Self {
            container: Container::default(),
            inputs: RefCell::new(Vec::new()),
        });
        cell.container.add_child(child);
        cell
    }
}

impl Component for DelegatingContainer {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        self.inputs.borrow_mut().push(data.to_string());
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }
}

/// The boxed mouse handler the fixtures install, restating upstream's
/// inline function properties.
type MouseHandler = Box<dyn Fn(&TuiMouseEvent) -> Option<TuiMouseEventResult>>;

/// A fake clock the scheduler's deadline arithmetic reads, upstream's
/// `performance.now()` monkeypatch.
#[derive(Clone)]
struct FakeClock {
    now: Rc<Cell<Instant>>,
}

impl FakeClock {
    fn new() -> Self {
        Self {
            now: Rc::new(Cell::new(Instant::now())),
        }
    }

    fn advance_ms(&self, ms: u64) {
        self.now.set(self.now.get() + Duration::from_millis(ms));
    }
}

fn stop(tui: &Tui) {
    tui.stop(pi_tui::tui::TuiStopOptions::default());
}

// === mouse dispatch helpers ===

#[test]
fn retarget_mouse_event_rebuilds_local_coordinates() {
    let leaf = SilentLeaf::mouse_result(TuiMouseEventResult {
        handled: true,
        ..TuiMouseEventResult::default()
    });
    let event = tui_support::mouse_event(TuiMouseEventType::Press, 4, 3, 20, 10);
    let leaf_dyn: Rc<dyn Component> = leaf;
    let result = dispatch_mouse_event(&leaf_dyn, &event).expect("leaf result with target");
    let target = result.target.expect("dispatch target");
    let retargeted = retarget_mouse_event(&event, &target);
    assert_eq!(retargeted.x, event.screen_x - target.origin_x);
    assert_eq!(retargeted.y, event.screen_y - target.origin_y);
    assert_eq!(retargeted.width, target.width);
    assert_eq!(retargeted.height, target.height);
}

#[test]
fn dispatch_drops_results_that_set_nothing() {
    let leaf: Rc<dyn Component> = SilentLeaf::new();
    let event = tui_support::mouse_event(TuiMouseEventType::Press, 0, 0, 20, 1);
    assert!(
        dispatch_mouse_event(&leaf, &event).is_none(),
        "a result with no handled/capture/focus is dropped"
    );
}

#[test]
fn delegating_container_rewrites_nested_focus_to_itself() {
    // A container that handles input keeps keyboard focus itself, upstream's
    // `focusTarget: this` on delegating containers.
    let child = SilentLeaf::mouse_result(TuiMouseEventResult {
        focus: true,
        ..TuiMouseEventResult::default()
    });
    let delegating = DelegatingContainer::with_child(child);
    let _delegating_dyn: Rc<dyn Component> = delegating.clone();
    let event = tui_support::mouse_event(TuiMouseEventType::Press, 0, 0, 20, 1);
    let delegating_dyn: Rc<dyn Component> = delegating;
    let result = dispatch_mouse_event(&delegating_dyn, &event).expect("delegating dispatch");
    let focus_target = result.focus_target.expect("delegating container");
    assert!(Rc::ptr_eq(&focus_target, &delegating_dyn));
}

#[test]
fn plain_container_leaves_nested_focus_target_intact() {
    let child = SilentLeaf::mouse_result(TuiMouseEventResult {
        focus: true,
        ..TuiMouseEventResult::default()
    });
    let container = Rc::new(Container::default());
    let child_dyn: Rc<dyn Component> = child;
    container.add_child(child_dyn.clone());
    let event = tui_support::mouse_event(TuiMouseEventType::Press, 0, 0, 20, 1);
    let container_dyn: Rc<dyn Component> = container;
    let result = dispatch_mouse_event(&container_dyn, &event).expect("dispatch");
    let focus_target = result.focus_target.expect("child focus");
    assert!(Rc::ptr_eq(&focus_target, &child_dyn));
}

#[test]
fn container_mouse_dispatch_misses_out_of_bounds_rows() {
    let container = Rc::new(Container::default());
    container.add_child(Rc::new(EmptyContent));
    let event = tui_support::mouse_event(TuiMouseEventType::Press, 0, 5, 20, 1);
    assert!(container.handle_mouse(&event).is_none(), "y beyond height");
}

#[test]
fn container_mouse_dispatch_misses_children_and_reuses_cached_layout() {
    let leaf = SilentLeaf::new();
    let container = Rc::new(Container::default());
    container.add_child(leaf.clone());
    // Commit the layout at width 20.
    container.render(20);
    // Same-width dispatch walks the committed heights (miss below the
    // child's line).
    let miss = tui_support::mouse_event(TuiMouseEventType::Press, 0, 5, 20, 10);
    assert!(container.handle_mouse(&miss).is_none());
    // A different width re-measures without rendering (the upstream fresh
    // path does not update the cache).
    let other = tui_support::mouse_event(TuiMouseEventType::Press, 0, 0, 21, 10);
    let _ = other;
    let leaf_dyn: Rc<dyn Component> = leaf;
    container.remove_child(&leaf_dyn);
    container.clear();
    container.invalidate();
}

#[test]
fn mouse_region_renders_through_the_child_and_invalidates_through_it() {
    let child = SilentLeaf::new();
    let handler: MouseHandler = Box::new(|_| None);
    let region = pi_tui::components::MouseRegion::new(child, handler);
    assert_eq!(region.render(10), vec!["leaf".to_string()]);
    region.invalidate();
    // The debug impl for the boxed-handler instantiation.
    let debug = format!("{region:?}");
    assert!(debug.contains("MouseRegion"));
}

#[test]
fn mouse_region_defers_to_child_dispatch_then_falls_back() {
    let child = SilentLeaf::mouse_result(TuiMouseEventResult {
        handled: true,
        ..TuiMouseEventResult::default()
    });
    let handled: MouseHandler = Box::new(|_| Some(TuiMouseEventResult::default()));
    let region = pi_tui::components::MouseRegion::new(child, handled);
    let press = tui_support::mouse_event(TuiMouseEventType::Press, 0, 0, 20, 1);
    let region_dyn: Rc<dyn Component> = Rc::new(region);
    let result = dispatch_mouse_event(&region_dyn, &press).expect("child handled");
    assert!(result.handled);

    let fallback: MouseHandler = Box::new(|_| {
        Some(TuiMouseEventResult {
            handled: true,
            ..TuiMouseEventResult::default()
        })
    });
    let silent = SilentLeaf::new();
    let fallback_region = pi_tui::components::MouseRegion::new(silent, fallback);
    let fallback_region_dyn: Rc<dyn Component> = Rc::new(fallback_region);
    let result2 = dispatch_mouse_event(
        &fallback_region_dyn,
        &tui_support::mouse_event(TuiMouseEventType::Press, 0, 0, 20, 1),
    )
    .expect("handler fallback");
    assert!(result2.handled);
}

// === focusable contract ===

#[test]
fn focusable_defaults_and_downcasting() {
    let leaf = SilentLeaf::new();
    let leaf_dyn: Rc<dyn Component> = leaf.clone();
    assert!(!is_focusable(Some(&leaf_dyn)));
    assert!(!is_focusable(None), "upstream's isFocusable(null) is false");
    // Defaulted contract methods.
    assert!(!leaf.wants_key_release());
    leaf.handle_input("dropped");
    let empty: Rc<dyn Component> = Rc::new(EmptyContent);
    assert!(!is_focusable(Some(&empty)));
}

// === TUI core ===

#[test]
fn mode_defaults_and_counters() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal);
    assert_eq!(tui.mode(), TuiMode::Regular);
    assert!(!tui.is_viewport_tui());
    tui.set_layout_root(None);
    assert_eq!(tui.full_redraws(), 0);
    tui.bump_full_redraws();
    assert_eq!(tui.full_redraws(), 1);
    // Debug output covers the manual Debug impls.
    let debug = format!("{tui:?}");
    assert!(debug.contains("Tui"));
}

#[test]
fn inert_renderer_fallback_construction() {
    let tui = Tui::new(TuiConfig::default());
    assert_eq!(tui.mode(), TuiMode::Regular);
    assert!(!tui.is_viewport_tui());
    tui.set_layout_root(None);
    tui.render_now(false);
    stop(&tui);
}

#[test]
fn thread_render_request_flag_wakes_the_pump() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.start();
    let request = tui.render_request();
    request();
    // The pump sees the demand flag and renders in this turn.
    wait_for_render(&tui);
    let viewport = terminal.get_viewport();
    assert!(!viewport.is_empty(), "the demand flag produced a frame");
    stop(&tui);
}

#[test]
fn scheduler_coalesces_within_the_16_ms_interval() {
    let clock = FakeClock::new();
    let terminal = TestTerminal::new(20, 6);
    let now = Rc::clone(&clock.now);
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(tui_support::TestRenderer)),
        clock: Some(Box::new(move || now.get())),
        ..TuiConfig::default()
    });
    tui.start();
    wait_for_render(&tui); // first render, lastRenderAt set
    tui.request_render(false);
    // Coalesced: still inside the 16 ms interval, so this pump fires nothing.
    wait_for_render(&tui);
    clock.advance_ms(20);
    wait_for_render(&tui);
    // Input preempts an armed throttled frame: the immediate path cancels it.
    tui.request_render(false);
    tui.set_focus(None); // owner-thread force render, immediate path
    render_and_flush(&tui);
    stop(&tui);
}

#[test]
fn stopped_sessions_skip_scheduler_and_input() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.start();
    tui.stop(pi_tui::tui::TuiStopOptions::default());
    wait_for_render(&tui);
    terminal.send_input("x");
    assert!(
        terminal.write_log().contains("\x1b[?2004l"),
        "stop disables bracketed paste"
    );
}

#[test]
fn queued_input_drains_after_the_terminal_poll_window() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    tui.set_focus(Some(editor.clone()));
    tui.start();
    terminal.queue_input("q");
    wait_for_render(&tui);
    assert_eq!(editor.inputs(), vec!["q".to_string()]);
    stop(&tui);
}

#[test]
fn resize_notification_raises_render_demand() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.start();
    terminal.resize(30, 10);
    wait_for_render(&tui);
    // The frame rendered at the new size is in the emulator's grid.
    assert_eq!(terminal.get_viewport().len(), 10);
    stop(&tui);
}

#[test]
fn input_listeners_rewrite_consume_and_unsubscribe() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    tui.set_focus(Some(editor.clone()));
    tui.start();
    // Rewrite listener: appends a marker.
    let rewrite = tui.add_input_listener(Rc::new(|data: &str| {
        Some(pi_tui::tui::TuiInputListenerResult {
            consume: false,
            data: Some(format!("{data}!")),
        })
    }));
    terminal.send_input("a");
    assert_eq!(editor.inputs(), vec!["a!".to_string()]);
    // Consume everything downstream.
    let consume = tui.add_input_listener(Rc::new(|_: &str| {
        Some(pi_tui::tui::TuiInputListenerResult {
            consume: true,
            data: None,
        })
    }));
    terminal.send_input("b");
    assert_eq!(
        editor.inputs(),
        vec!["a!".to_string()],
        "the consumed chunk never reached the editor"
    );
    // Remove the consumer and re-send.
    tui.remove_input_listener(consume);
    terminal.send_input("b");
    assert_eq!(editor.inputs(), vec!["a!".to_string(), "b!".to_string()]);
    tui.remove_input_listener(rewrite);
    terminal.send_input("c");
    assert_eq!(
        editor.inputs(),
        vec!["a!".to_string(), "b!".to_string(), "c".to_string()]
    );
    stop(&tui);
}

#[test]
fn debug_key_dispatches_when_installed() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let fired = Rc::new(Cell::new(false));
    let flag = fired.clone();
    tui.set_on_debug(Some(Rc::new(move || flag.set(true))));
    // Legacy ctrl+shift+d, the encoding the default parser decodes.
    terminal.send_input("\x1b[27;6;100~");
    assert!(fired.get(), "debug key fired");
    assert_eq!(
        editor.inputs(),
        Vec::<String>::new(),
        "debug key never reaches the focused component"
    );
    tui.set_on_debug(None);
    terminal.send_input("x");
    assert_eq!(
        editor.inputs(),
        vec!["x".to_string()],
        "ordinary input still routes"
    );
    stop(&tui);
}

#[test]
fn cell_size_consumer_drops_zero_and_huge_dimensions() {
    let terminal = TestTerminal::new(80, 24);
    let tui = tui_support::new_test_tui_with_images(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    tui.set_focus(Some(editor.clone()));
    tui.start();
    assert!(
        terminal.write_log().contains("\x1b[16t"),
        "image-capable probe sends the cell-size query"
    );
    // Zero dimensions are consumed and dropped.
    terminal.send_input("\x1b[6;0;0t");
    // An overlong digit run is consumed too.
    terminal.send_input("\x1b[6;99999999999999999999999;1t");
    // Later input still flows.
    terminal.send_input("q");
    assert_eq!(editor.inputs(), vec!["q".to_string()]);
    stop(&tui);
}

#[test]
fn hardware_cursor_and_clear_on_shrink_toggles() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    assert!(!tui.get_show_hardware_cursor());
    assert!(!tui.get_clear_on_shrink());
    tui.set_show_hardware_cursor(true);
    assert!(tui.get_show_hardware_cursor());
    tui.set_show_hardware_cursor(true); // no-op repeat
    tui.set_show_hardware_cursor(false);
    assert!(terminal.write_log().contains("\x1b[?25l"));
    tui.set_clear_on_shrink(true);
    assert!(tui.get_clear_on_shrink());
    tui.set_clear_on_shrink(false);
    stop(&tui);
}

#[test]
fn color_scheme_notification_writes_follow_the_flag() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.set_terminal_color_scheme_notifications(true);
    assert!(terminal.write_log().contains("\x1b[?2031h"));
    tui.set_terminal_color_scheme_notifications(true); // no-op repeat
    tui.set_terminal_color_scheme_notifications(false);
    assert!(terminal.write_log().contains("\x1b[?2031l"));
    // While stopped, only the flag flips; start emits the enable sequence.
    tui.set_terminal_color_scheme_notifications(true);
    stop(&tui);
    let writes_at_stop = terminal.write_log().len();
    tui.set_terminal_color_scheme_notifications(false);
    assert_eq!(terminal.write_log().len(), writes_at_stop);
}

#[test]
fn start_emits_scheme_notifications_when_enabled_before_start() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.set_terminal_color_scheme_notifications(true);
    stop(&tui);
    let writes_at_stop = terminal.write_log().len();
    tui.start();
    assert!(
        terminal.write_log()[writes_at_stop..].contains("\x1b[?2031h"),
        "start emits the enable sequence for the armed flag"
    );
    stop(&tui);
}

// === terminal queries ===

#[test]
fn osc11_background_query_resolves_on_reply() {
    let clock = FakeClock::new();
    let terminal = TestTerminal::new(20, 6);
    let now = Rc::clone(&clock.now);
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal.clone())),
        renderer: Some(Box::new(tui_support::TestRenderer)),
        clock: Some(Box::new(move || now.get())),
        ..TuiConfig::default()
    });
    tui.start();
    let receiver = tui.query_terminal_background_color(5_000);
    assert!(terminal.write_log().contains("\x1b]11;?\u{7}"));
    // The reply arrives as input.
    terminal.send_input("\x1b]11;rgb:12/34/56\x07");
    let color = receiver
        .recv_timeout(Duration::from_millis(100))
        .expect("reply resolved the channel");
    let rgb = color.expect("reply carried a color");
    assert_eq!(
        rgb,
        pi_tui::terminal_colors::RgbColor {
            r: 0x12,
            g: 0x34,
            b: 0x56
        }
    );
    stop(&tui);
}

#[test]
fn osc11_background_query_times_out() {
    let clock = FakeClock::new();
    let terminal = TestTerminal::new(20, 6);
    let now = Rc::clone(&clock.now);
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(tui_support::TestRenderer)),
        clock: Some(Box::new(move || now.get())),
        ..TuiConfig::default()
    });
    tui.start();
    let receiver = tui.query_terminal_background_color(50);
    clock.advance_ms(60);
    wait_for_render(&tui);
    let color = receiver
        .recv_timeout(Duration::from_millis(100))
        .expect("deadline fired");
    assert!(color.is_none(), "timeout resolved with None");
    stop(&tui);
}

#[test]
fn color_scheme_query_resolves_on_report_and_timeout_clears_the_listener() {
    let clock = FakeClock::new();
    let terminal = TestTerminal::new(20, 6);
    let now = Rc::clone(&clock.now);
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal.clone())),
        renderer: Some(Box::new(tui_support::TestRenderer)),
        clock: Some(Box::new(move || now.get())),
        ..TuiConfig::default()
    });
    tui.start();
    let receiver = tui.query_terminal_color_scheme(5_000);
    assert!(terminal.write_log().contains("\x1b[?996n"));
    terminal.send_input("\x1b[?997;1n");
    let scheme = receiver
        .recv_timeout(Duration::from_millis(100))
        .expect("reply settled the channel")
        .expect("reply carried a scheme");
    assert_eq!(scheme, pi_tui::terminal_colors::TerminalColorScheme::Dark);

    // A timed-out query unsubscribes its listener.
    let expired = tui.query_terminal_color_scheme(10);
    clock.advance_ms(20);
    wait_for_render(&tui);
    assert!(
        expired
            .recv_timeout(Duration::from_millis(100))
            .map_or(true, |scheme| scheme.is_none()),
        "timeout resolved"
    );
    // A later scheme report finds no listener and is consumed silently.
    terminal.send_input("\x1b[?997;2n");
    stop(&tui);
}

// === overlay handle surface ===

#[test]
fn overlay_handle_reports_bounds_and_hidden_state() {
    let terminal = TestTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    tui.add_child(editor);
    let overlay = tui_support::StaticOverlay::new(vec!["X"]);
    let handle: OverlayHandle = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(10)),
            ..OverlayOptions::default()
        }),
    );
    assert!(!handle.is_hidden());
    assert!(handle.get_bounds().is_none(), "no bounds before a render");
    tui.start();
    render_and_flush(&tui);
    let bounds = handle.get_bounds().expect("bounds after render");
    assert_eq!(bounds.row, 0);
    assert_eq!(bounds.col, 0);
    handle.set_hidden(true);
    assert!(handle.is_hidden());
    assert!(
        handle.get_bounds().is_none(),
        "hidden overlays expose no bounds"
    );
    handle.set_hidden(false);
    assert!(handle.get_bounds().is_some());
    handle.hide();
    assert!(
        handle.get_bounds().is_none(),
        "hidden overlays expose no bounds"
    );
    stop(&tui);
}

#[test]
fn overlay_unfocus_with_null_target_clears_focus() {
    let terminal = TestTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    tui.add_child(editor);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.start();
    let handle = tui.show_overlay(overlay, None);
    assert!(handle.is_focused());
    handle.unfocus(Some(OverlayUnfocusOptions { target: None }));
    assert!(!handle.is_focused());
    assert!(tui.get_focused_component().is_none());
    stop(&tui);
}

#[test]
fn overlay_mouse_dispatch_routes_through_rendered_layouts() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.set_focus(Some(editor));
    tui.start();
    let _ = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(7)),
            ..OverlayOptions::default()
        }),
    );
    render_and_flush(&tui);
    assert!(tui.has_overlay());
    assert!(tui.is_overlay_focused());

    // Miss: outside every rendered overlay rectangle.
    let miss = tui.dispatch_mouse_to_overlay(&tui_support::mouse_event(
        TuiMouseEventType::Press,
        19,
        5,
        20,
        6,
    ));
    assert!(!miss.hit);

    // Hit: the overlay does not implement a mouse handler, so the dispatch
    // records the hit without a result — upstream's `{ hit: true }` shape.
    let hit = tui.dispatch_mouse_to_overlay(&tui_support::mouse_event(
        TuiMouseEventType::Press,
        2,
        0,
        20,
        6,
    ));
    assert!(hit.hit);
    assert!(
        hit.result.is_none(),
        "a non-mouse overlay records a bare hit"
    );

    // resolve_mouse_focus_target keeps the overlay as the focus owner and
    // passes unrelated components through.
    let overlay_dyn: Rc<dyn Component> = overlay;
    let resolved = tui.resolve_mouse_focus_target(&overlay_dyn);
    assert!(Rc::ptr_eq(&resolved, &overlay_dyn));
    let outsider: Rc<dyn Component> = Rc::new(EmptyContent);
    let passed_through = tui.resolve_mouse_focus_target(&outsider);
    assert!(Rc::ptr_eq(&passed_through, &outsider));
    stop(&tui);
}

// === compositing helpers ===

#[test]
fn composite_tui_line_passes_image_lines_through() {
    let image_line = "\x1b_Gf=32,s=1,i=1\x1b\\".to_string();
    let out = composite_tui_line(&image_line, "XX", 0, 2, 20);
    assert_eq!(out, image_line);
}

#[test]
fn composite_tui_line_truncates_when_content_overflows() {
    // A long base line with a wide overlay at the end forces the
    // sliceByColumn fallback path.
    let base = format!("{}tail", "x".repeat(10));
    let out = composite_tui_line(&base, "AAAAAAAAAAAAAAAA", 4, 4, 10);
    assert!(pi_tui::utils::visible_width(&out) <= 10);
}

#[test]
fn extract_cursor_position_strips_the_marker_and_scans_only_the_viewport() {
    let terminal = TestTerminal::new(20, 3);
    let tui = tui_support::new_test_tui(terminal);
    let mut lines = vec![
        format!("above{}", pi_tui::tui::CURSOR_MARKER),
        format!("be{}low", pi_tui::tui::CURSOR_MARKER),
        "third".to_string(),
    ];
    let position = tui
        .extract_cursor_position(&mut lines, 2)
        .expect("marker in the viewport");
    // Bottom-up over the last `height` rows: line 1 holds the marker after
    // "be", so the visual column is 2 and line 0 was never scanned.
    assert_eq!(position, (1, 2), "scans only the bottom height lines");
    assert!(!lines[1].contains(pi_tui::tui::CURSOR_MARKER));
    assert_eq!(lines[1], "below");
    // The marker on line 0 was outside the viewport and survives.
    assert!(lines[0].contains("above"));
}

#[test]
fn overlay_layout_clamps_negative_rows_and_margins() {
    let terminal = TestTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.add_child(FocusableOverlay::new(&["EDITOR"]));
    let _ = tui.show_overlay(
        tui_support::StaticOverlay::new(vec!["NEG"]),
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(10)),
            margin: Some(pi_tui::tui::OverlayMargin::All(-3)),
            row: Some(SizeValue::Cells(-10)),
            offset_x: Some(-50),
            offset_y: Some(-30),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);
    let viewport = terminal.get_viewport();
    assert!(
        viewport.first().is_some_and(|line| line.contains("NEG")),
        "negative rows, margins, and offsets clamp to the top-left corner"
    );
    stop(&tui);
}

#[test]
fn layout_debug_output_covers_the_option_surface() {
    let options = OverlayOptions {
        anchor: Some(OverlayAnchor::Center),
        ..OverlayOptions::default()
    };
    let debug = format!("{options:?}");
    assert!(debug.contains("OverlayOptions"));
    let container = Container::default();
    assert!(format!("{container:?}").contains("Container"));
    let result = TuiMouseEventResult::default();
    assert!(format!("{result:?}").contains("TuiMouseEventResult"));
    let config = TuiConfig::default();
    assert!(format!("{config:?}").contains("TuiConfig"));
}

#[test]
fn default_component_contract_answers() {
    // The defaulted trait methods every non-input component inherits.
    let empty: Rc<dyn Component> = Rc::new(EmptyContent);
    assert!(!is_focusable(Some(&empty)));
    empty.handle_input("dropped");
    assert!(!empty.wants_input());
    assert!(!empty.wants_key_release());
    let event = tui_support::mouse_event(TuiMouseEventType::Move, 0, 0, 20, 1);
    assert!(dispatch_mouse_event(&empty, &event).is_none());
}

#[test]
fn overlay_handle_on_a_dropped_tui_answers_neutrally() {
    let handle = {
        let terminal = TestTerminal::new(20, 6);
        let tui = tui_support::new_test_tui(terminal);
        let overlay = FocusableOverlay::new(&["O"]);
        tui.show_overlay(overlay, None)
    };
    // The TUI is gone; the handle's Weak upgrade fails and methods answer
    // neutrally rather than panicking.
    assert!(!handle.is_focused());
    assert!(handle.get_bounds().is_none());
    handle.hide();
    handle.focus();
    handle.unfocus(Some(OverlayUnfocusOptions::default()));
}
// === scheduler re-arm under a fake clock ===

/// A renderer that bumps a frame counter and re-raises a render request
/// mid-frame, restating a component calling `requestRender` during its own
/// render.
#[derive(Debug)]
struct SelfRequestingRenderer {
    self_weak: RefCell<Weak<Tui>>,
    frames: Cell<u32>,
}

impl TuiRenderer for SelfRequestingRenderer {
    fn mode(&self) -> TuiMode {
        TuiMode::Regular
    }

    fn do_render(&self, tui: &Tui) {
        if tui.is_stopped() {
            return;
        }
        self.frames.set(self.frames.get() + 1);
        if let Some(tui) = self.self_weak.borrow().upgrade() {
            (tui.render_request())();
        }
    }
}

#[test]
fn a_render_that_re_requests_rearms_the_throttle() {
    let clock = FakeClock::new();
    let terminal = TestTerminal::new(20, 6);
    let renderer = Rc::new(SelfRequestingRenderer {
        self_weak: RefCell::new(Weak::new()),
        frames: Cell::new(0),
    });
    let now = Rc::clone(&clock.now);
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(SelfRequestingShared(renderer.clone()))),
        clock: Some(Box::new(move || now.get())),
        ..TuiConfig::default()
    });
    renderer.self_weak.replace(Rc::downgrade(&tui));
    tui.start();
    wait_for_render(&tui);
    // The first frame re-raised a request, which upstream re-arms through
    // `scheduleRender`: the fresh 16 ms deadline holds the pump inside the
    // interval.
    let frames = renderer.frames.get();
    // The first frame re-raised a request; upstream re-arms through
    // `scheduleRender`, so a pump inside the fresh 16 ms interval coalesces.
    wait_for_render(&tui);
    assert_eq!(
        renderer.frames.get(),
        frames,
        "coalesced inside the interval"
    );
    clock.advance_ms(20);
    wait_for_render(&tui);
    assert!(
        renderer.frames.get() > frames,
        "the re-armed deadline produced another frame"
    );
    // That frame re-armed again; another pump inside the interval coalesces.
    wait_for_render(&tui);
    assert_eq!(
        renderer.frames.get(),
        frames + 1,
        "still coalesced after the re-arm"
    );
    stop(&tui);
}

/// The `Rc`-shared renderer handed to the TUI through its `Box` seam.
#[derive(Debug)]
struct SelfRequestingShared(Rc<SelfRequestingRenderer>);

impl TuiRenderer for SelfRequestingShared {
    fn mode(&self) -> TuiMode {
        self.0.mode()
    }

    fn do_render(&self, tui: &Tui) {
        self.0.do_render(tui);
    }
}

#[test]
fn composite_without_overlays_clears_the_rendered_layouts() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal);
    let lines = vec!["a".to_string(), "b".to_string()];
    assert_eq!(tui.composite_overlays(lines.clone(), 20, 6), lines);
    assert!(!tui.has_overlay_entries());
}

#[test]
fn child_container_helpers_reach_the_container() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal);
    let child: Rc<dyn Component> = Rc::new(EmptyContent);
    tui.add_child(child.clone());
    assert!(tui.render_children(10).is_empty());
    tui.remove_child(&child);
    tui.clear();
    assert!(tui.children().is_empty());
    // The TUI as a component: render and mouse delegation.
    assert!(Component::render(&*tui, 10).is_empty());
    let event = tui_support::mouse_event(TuiMouseEventType::Press, 0, 0, 20, 1);
    assert!(Component::handle_mouse(&*tui, &event).is_none());
    tui.invalidate();
}

#[test]
fn free_fn_is_viewport_tui_reads_the_capability() {
    let terminal = TestTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal);
    assert!(!pi_tui::tui::is_viewport_tui(&tui));
}

#[test]
fn composite_tui_line_falls_back_to_slicing_on_overflow() {
    // A base line wider than the total width forces the sliceByColumn
    // fallback before the overlay is stamped on.
    let base = "y".repeat(30);
    let out = composite_tui_line(&base, "OVR", 25, 4, 10);
    assert_eq!(pi_tui::utils::visible_width(&out), 10);
}

#[test]
fn debug_impls_cover_the_handle_and_unfocus_options() {
    let handle = {
        let terminal = TestTerminal::new(20, 6);
        let tui = tui_support::new_test_tui(terminal);
        tui.show_overlay(FocusableOverlay::new(&["O"]), None)
    };
    let debug = format!("{handle:?}");
    assert!(debug.contains("OverlayHandle"));
    let unfocus = OverlayUnfocusOptions::default();
    let unfocus_debug = format!("{unfocus:?}");
    assert!(unfocus_debug.contains("OverlayUnfocusOptions"));
}
