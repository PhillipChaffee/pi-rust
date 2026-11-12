//! The interactive TUI composition root and reference at pin `60e7e76`:
//! renderer selection, the stable handle across a re-point, and the captured
//! render request routing (upstream's #7731 regression).

#![expect(
    clippy::expect_used,
    reason = "a missing renderer handle or focus target fails the assertion by design"
)]

use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_coding_agent::modes::interactive::tui_renderer::{
    InteractiveTuiOptions, create_interactive_tui, create_interactive_tui_reference,
};
use pi_tui::terminal::{ProcessTerminal, Terminal, WriteSink};
use pi_tui::tui::{Component, Container, Tui, TuiConfig, TuiMode, TuiRenderer};

#[derive(Debug)]
struct CountingRenderer {
    renders: Arc<AtomicUsize>,
}

impl TuiRenderer for CountingRenderer {
    fn mode(&self) -> TuiMode {
        TuiMode::Regular
    }

    fn do_render(&self, _tui: &Tui) {
        self.renders.fetch_add(1, Ordering::SeqCst);
    }
}

fn headless_terminal() -> Box<dyn Terminal> {
    let sink: WriteSink = Arc::new(Mutex::new(Box::new(|_| {})));
    Box::new(ProcessTerminal::headless(sink))
}

fn counting_tui(renders: Arc<AtomicUsize>) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(headless_terminal()),
        renderer: Some(Box::new(CountingRenderer { renders })),
        ..TuiConfig::default()
    })
}

fn options(tui_mode: TuiMode) -> InteractiveTuiOptions {
    InteractiveTuiOptions {
        tui_mode,
        show_hardware_cursor: true,
        log_directory: "/tmp/pi-interactive-tui-test".to_string(),
        terminal: Some(headless_terminal()),
        on_right_click_paste: None,
        fullscreen_copy_on_select: Some(true),
    }
}

#[test]
fn selects_the_alternate_screen_renderer_only_when_requested() {
    let regular = create_interactive_tui(options(TuiMode::Regular));
    assert_eq!(regular.tui().mode(), TuiMode::Regular);
    assert!(!regular.tui().is_viewport_tui());
    assert!(regular.main_screen().is_some());
    assert!(regular.viewport().is_none());
    assert_eq!(
        regular
            .main_screen()
            .expect("main renderer")
            .capture_render_state()
            .previous_width,
        0,
        "the regular renderer keeps its restorable state"
    );

    let fullscreen = create_interactive_tui(options(TuiMode::Fullscreen));
    assert_eq!(fullscreen.tui().mode(), TuiMode::Fullscreen);
    assert!(fullscreen.tui().is_viewport_tui());
    assert!(fullscreen.main_screen().is_none());
    assert!(fullscreen.viewport().is_some());
    assert!(
        fullscreen
            .viewport()
            .expect("viewport")
            .get_copy_on_select()
    );
    assert!(fullscreen.tui().get_show_hardware_cursor());
    assert_eq!(
        fullscreen.tui().log_directory.as_deref(),
        Some(Path::new("/tmp/pi-interactive-tui-test"))
    );
}

#[test]
fn reference_delegates_and_follows_a_re_point() {
    let first_tui = counting_tui(Arc::new(AtomicUsize::new(0)));
    let second_tui = counting_tui(Arc::new(AtomicUsize::new(0)));
    let reference = create_interactive_tui_reference(Rc::clone(&first_tui));
    let first_component: Rc<dyn Component> = Rc::new(Container::default());
    let second_component: Rc<dyn Component> = Rc::new(Container::default());

    reference.set_focus(Some(Rc::clone(&first_component)));
    assert!(Rc::ptr_eq(
        &reference.get_focused_component().expect("first focus"),
        &first_component
    ));

    reference.repoint(Rc::clone(&second_tui));
    assert!(Rc::ptr_eq(&reference.current(), &second_tui));
    // The proxy forwards to the replacement TUI, which starts unfocused; the
    // frame re-applies focus after mounting, as upstream re-points the proxy.
    assert!(reference.get_focused_component().is_none());

    reference.set_focus(Some(Rc::clone(&second_component)));
    assert!(Rc::ptr_eq(
        &second_tui.get_focused_component().expect("second focus"),
        &second_component
    ));
    assert!(Rc::ptr_eq(
        &first_tui.get_focused_component().expect("first focus"),
        &first_component
    ));
    assert_eq!(reference.mode(), TuiMode::Regular);
}

#[test]
fn a_captured_render_request_repaints_the_tui_that_minted_it() {
    let renders = Arc::new(AtomicUsize::new(0));
    let tui = counting_tui(Arc::clone(&renders));
    let reference = create_interactive_tui_reference(Rc::clone(&tui));
    let request = reference.render_request();

    request();
    tui.poll(Duration::ZERO);

    assert_eq!(renders.load(Ordering::SeqCst), 1);
}

#[test]
fn a_captured_render_request_routes_to_a_replacement_tui() {
    let first_renders = Arc::new(AtomicUsize::new(0));
    let second_renders = Arc::new(AtomicUsize::new(0));
    let first_tui = counting_tui(Arc::clone(&first_renders));
    let second_tui = counting_tui(Arc::clone(&second_renders));
    let reference = create_interactive_tui_reference(Rc::clone(&first_tui));
    let request = reference.render_request();

    reference.repoint(Rc::clone(&second_tui));
    request();
    second_tui.poll(Duration::ZERO);
    first_tui.poll(Duration::ZERO);

    assert_eq!(
        second_renders.load(Ordering::SeqCst),
        1,
        "the captured request reaches the replacement"
    );
    assert_eq!(first_renders.load(Ordering::SeqCst), 0);
}

#[test]
fn reference_mounts_and_clears_on_the_current_tui() {
    let tui = counting_tui(Arc::new(AtomicUsize::new(0)));
    let reference = create_interactive_tui_reference(Rc::clone(&tui));
    let child: Rc<dyn Component> = Rc::new(Container::default());

    reference.add_child(Rc::clone(&child));
    assert_eq!(reference.current().children().len(), 1);
    assert!(Rc::ptr_eq(&reference.current().children()[0], &child));

    reference.remove_child(&child);
    assert!(reference.current().children().is_empty());

    reference.set_layout_root(Some(Rc::clone(&child)));
    reference.set_clear_on_shrink(true);
    assert_eq!(reference.with_current(Tui::mode), TuiMode::Regular);
}
