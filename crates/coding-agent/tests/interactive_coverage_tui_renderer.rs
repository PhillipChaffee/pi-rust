//! Coverage of the interactive TUI composition root and its stable reference:
//! both renderer branches, every delegated reference method across a re-point,
//! and the fullscreen callback surface the frame wires.

#![expect(
    clippy::expect_used,
    reason = "the renderer handles are pinned by the fixture"
)]

mod common;

use std::rc::Rc;

use common::interactive::RecordingTerminal;
use pi_coding_agent::modes::interactive::tui_renderer::{
    InteractiveTuiOptions, create_interactive_tui, create_interactive_tui_reference,
};
use pi_tui::components::Text;
use pi_tui::components::scroll_view::{FollowMode, ScrollView, ScrollViewOptions};
use pi_tui::tui::{Component, Container, TuiMode, TuiStopOptions};

fn options(mode: TuiMode, terminal: &RecordingTerminal) -> InteractiveTuiOptions {
    InteractiveTuiOptions {
        tui_mode: mode,
        show_hardware_cursor: true,
        log_directory: "/tmp/pi-coverage-tui".to_owned(),
        terminal: Some(Box::new(terminal.clone())),
        on_right_click_paste: Some(Rc::new(|| {})),
        fullscreen_copy_on_select: Some(false),
    }
}

#[test]
fn both_renderer_branches_expose_their_handles() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let regular_terminal = RecordingTerminal::new(80, 24);
    let regular = create_interactive_tui(options(TuiMode::Regular, &regular_terminal));
    assert!(regular.main_screen().is_some());
    assert!(regular.viewport().is_none());
    assert_eq!(regular.tui().mode(), TuiMode::Regular);

    let fullscreen_terminal = RecordingTerminal::new(80, 24);
    let fullscreen = create_interactive_tui(options(TuiMode::Fullscreen, &fullscreen_terminal));
    assert!(fullscreen.viewport().is_some());
    assert!(fullscreen.main_screen().is_none());
    assert_eq!(fullscreen.tui().mode(), TuiMode::Fullscreen);
    fullscreen.tui().start();
    fullscreen.tui().render_now(true);
    assert!(fullscreen_terminal.writes().contains("\x1b[?1049h"));
    fullscreen.tui().stop(TuiStopOptions::default());
}

#[test]
fn reference_delegates_every_method_across_a_repoint() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let first = create_interactive_tui(options(TuiMode::Regular, &RecordingTerminal::new(80, 24)));
    let second = create_interactive_tui(options(
        TuiMode::Fullscreen,
        &RecordingTerminal::new(80, 24),
    ));
    let reference = create_interactive_tui_reference(first.tui());
    assert_eq!(reference.mode(), TuiMode::Regular);
    assert!(!reference.is_viewport_tui());

    let child: Rc<dyn Component> = Rc::new(Text::with_padding("child", 0, 0));
    reference.add_child(Rc::clone(&child));
    reference.set_focus(Some(Rc::clone(&child)));
    assert!(Rc::ptr_eq(
        &reference.get_focused_component().expect("focus"),
        &child
    ));
    reference.set_clear_on_shrink(true);
    reference.request_render(false);
    let request = reference.render_request();
    request();
    reference.set_layout_root(Some(Rc::clone(&child)));
    reference.remove_child(&child);
    reference.clear();

    reference.repoint(second.tui());
    assert_eq!(reference.mode(), TuiMode::Fullscreen);
    assert!(reference.is_viewport_tui());
    let replacement: Rc<dyn Component> = Rc::new(Text::with_padding("replacement", 0, 0));
    reference.add_child(Rc::clone(&replacement));
    reference.set_focus(Some(Rc::clone(&replacement)));
    assert!(reference.get_focused_component().is_some());
    let _ = reference.with_current(|tui| tui.children().len());
    request();
    reference.start();
    reference.stop(TuiStopOptions::default());
    assert!(Rc::ptr_eq(&reference.current(), &second.tui()));
}

#[test]
fn fullscreen_scroll_callbacks_are_wired() {
    pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
    let terminal = RecordingTerminal::new(60, 6);
    let ui = create_interactive_tui(options(TuiMode::Fullscreen, &terminal));
    let transcript = ScrollView::new(
        Rc::new(Text::with_padding(
            (1..=20)
                .map(|line| format!("line {line}"))
                .collect::<Vec<_>>()
                .join("\n"),
            0,
            0,
        )),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    let root: Rc<dyn Component> = transcript.clone();
    ui.tui().set_layout_root(Some(root));
    ui.tui().start();
    ui.tui().render_now(true);
    let _ = transcript.scroll_by(-3);
    ui.tui().render_now(true);
    assert!(terminal.writes().contains("Jump to latest message"));
    let container: Rc<dyn Component> = Rc::new(Container::default());
    ui.tui().add_child(container);
    ui.tui().render_now(true);
    ui.tui().stop(TuiStopOptions::default());
}
