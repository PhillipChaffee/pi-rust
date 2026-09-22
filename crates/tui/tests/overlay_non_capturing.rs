//! `overlay-non-capturing.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): the non-capturing overlay
//! contract — focus management, no-op guards, focus-cycle prevention, and
//! rendering order through the focus-order sort.
//!
//! The suites render through [`tui_support::new_test_tui`]'s `TuiMainScreen`;
//! see the support module.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::Cell;
use std::rc::Rc;

use pi_tui::tui::{Container, OverlayOptions, OverlayUnfocusOptions, SizeValue};

use tui_support::{
    EmptyContent, FocusableOverlay, StaticOverlay, VirtualTerminal, render_and_flush,
};

// === focus management ===

#[test]
fn non_capturing_overlay_preserves_focus_on_creation() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let _ = &handle;
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!tui_support::is_focused(&overlay));
    tui_support::stop(&tui);
}

#[test]
fn focus_transfers_focus_to_the_overlay() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.focus();
    render_and_flush(&tui);
    assert!(!tui_support::is_focused(&editor));
    assert!(tui_support::is_focused(&overlay));
    assert!(handle.is_focused());
    tui_support::stop(&tui);
}

#[test]
fn unfocus_restores_previous_focus() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.focus();
    handle.unfocus(None);
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!tui_support::is_focused(&overlay));
    assert!(!handle.is_focused());
    tui_support::stop(&tui);
}

#[test]
fn set_hidden_false_on_non_capturing_overlay_does_not_auto_focus() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.set_hidden(true);
    handle.set_hidden(false);
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!tui_support::is_focused(&overlay));
    tui_support::stop(&tui);
}

#[test]
fn hide_when_overlay_is_not_focused_does_not_change_focus() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.hide();
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    tui_support::stop(&tui);
}

#[test]
fn hide_when_focused_restores_focus_correctly() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.focus();
    handle.hide();
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!tui_support::is_focused(&overlay));
    tui_support::stop(&tui);
}

#[test]
fn capturing_overlay_removed_with_non_capturing_below_restores_focus_to_editor() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let non_capturing = FocusableOverlay::new(&["NC"]);
    let capturing = FocusableOverlay::new(&["CAP"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let _ = tui.show_overlay(
        non_capturing.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let handle = tui.show_overlay(capturing.clone(), None);
    assert!(tui_support::is_focused(&capturing));
    handle.hide();
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!tui_support::is_focused(&non_capturing));
    tui_support::stop(&tui);
}

#[test]
fn sub_overlay_cleanup_then_hide_overlay_restores_focus_and_input_to_editor() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let timer = FocusableOverlay::new(&["TIMER"]);
    let controller = FocusableOverlay::new(&["CTRL"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let timer_handle = tui.show_overlay(
        timer.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let _ = tui.show_overlay(controller.clone(), None);
    assert!(tui_support::is_focused(&controller));
    assert!(!tui_support::is_focused(&editor));
    timer_handle.hide();
    tui.hide_overlay();
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!tui_support::is_focused(&controller));
    assert!(!tui_support::is_focused(&timer));
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(editor.inputs(), vec!["x".to_string()]);
    assert_eq!(controller.inputs(), Vec::<String>::new());
    assert_eq!(timer.inputs(), Vec::<String>::new());
    tui_support::stop(&tui);
}

#[test]
fn removed_focused_child_overlay_does_not_become_parent_overlay_fallback() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let child = FocusableOverlay::new(&["CHILD"]);
    let parent = FocusableOverlay::new(&["PARENT"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let child_handle = tui.show_overlay(
        child.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    child_handle.focus();
    let parent_handle = tui.show_overlay(parent.clone(), None);
    assert!(tui_support::is_focused(&parent));

    child_handle.hide();
    parent_handle.hide();
    terminal.send_input("x");
    render_and_flush(&tui);

    assert_eq!(editor.inputs(), vec!["x".to_string()]);
    assert_eq!(child.inputs(), Vec::<String>::new());
    assert_eq!(parent.inputs(), Vec::<String>::new());
    assert!(tui_support::is_focused(&editor));
    tui_support::stop(&tui);
}

#[test]
fn microtask_deferred_sub_overlay_pattern_restores_focus() {
    // Simulates showExtensionCustom: the timer overlay is created
    // synchronously, then the controller lands in a follow-up turn.
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let timer = FocusableOverlay::new(&["TIMER"]);
    let controller = FocusableOverlay::new(&["CTRL"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let timer_handle = tui.show_overlay(
        timer.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    // `.then()` runs as a follow-up task — the port's follow-up turn is the
    // next pump.
    let _ = tui.show_overlay(controller.clone(), None);
    render_and_flush(&tui);

    assert!(tui_support::is_focused(&controller));
    assert!(!tui_support::is_focused(&editor));

    // Simulate Esc: cleanup + close (from inside handleInput).
    timer_handle.hide();
    tui.hide_overlay();
    render_and_flush(&tui);

    assert!(
        tui_support::is_focused(&editor),
        "editor should regain focus"
    );
    assert!(!tui_support::is_focused(&controller));
    assert!(!tui_support::is_focused(&timer));

    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(
        editor.inputs(),
        vec!["x".to_string()],
        "editor should receive input after close"
    );
    assert_eq!(controller.inputs(), Vec::<String>::new());
    tui_support::stop(&tui);
}

#[test]
fn handle_input_redirection_skips_non_capturing_overlays_when_focused_overlay_becomes_invisible() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let fallback_capturing = FocusableOverlay::new(&["FALLBACK"]);
    let non_capturing = FocusableOverlay::new(&["NC"]);
    let primary = FocusableOverlay::new(&["PRIMARY"]);
    let visible = Rc::new(Cell::new(true));
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor));
    tui.start();
    let _ = tui.show_overlay(fallback_capturing.clone(), None);
    let _ = tui.show_overlay(
        non_capturing.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let visible_check = visible.clone();
    let _ = tui.show_overlay(
        primary.clone(),
        Some(OverlayOptions {
            visible: Some(Box::new(move |_, _| visible_check.get())),
            ..OverlayOptions::default()
        }),
    );
    assert!(tui_support::is_focused(&primary));
    visible.set(false);
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(primary.inputs(), Vec::<String>::new());
    assert_eq!(non_capturing.inputs(), Vec::<String>::new());
    assert_eq!(fallback_capturing.inputs(), vec!["x".to_string()]);
    assert!(tui_support::is_focused(&fallback_capturing));
    tui_support::stop(&tui);
}

#[test]
fn active_base_focus_replacement_receives_close_input_before_overlay_restore() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let replacement = FocusableOverlay::new(&["REPLACEMENT"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    {
        let tui_ref = tui.clone();
        let replacement_ref = replacement.clone();
        overlay.set_on_input(move |data| {
            if data == "b" {
                tui_ref.set_focus(Some(replacement_ref.clone()));
            }
        });
    }
    {
        let tui_ref = tui.clone();
        let editor_ref = editor.clone();
        replacement.set_on_input(move |data| {
            if data == "\r" {
                tui_ref.set_focus(Some(editor_ref.clone()));
            }
        });
    }
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor));
    tui.start();
    let _ = tui.show_overlay(overlay.clone(), None);
    assert!(tui_support::is_focused(&overlay));
    terminal.send_input("b");
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&replacement));

    terminal.send_input("\r");
    render_and_flush(&tui);
    assert_eq!(replacement.inputs(), vec!["\r".to_string()]);
    assert_eq!(overlay.inputs().as_slice(), ["b"]);
    assert!(tui_support::is_focused(&overlay));

    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(overlay.inputs().as_slice(), ["b", "x"]);
    tui_support::stop(&tui);
}

#[test]
fn active_replacement_still_receives_input_when_it_is_another_overlay_pre_focus() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let replacement = FocusableOverlay::new(&["REPLACEMENT"]);
    let passive = FocusableOverlay::new(&["PASSIVE"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    {
        let tui_ref = tui.clone();
        let replacement_ref = replacement.clone();
        overlay.set_on_input(move |data| {
            if data == "b" {
                tui_ref.set_focus(Some(replacement_ref.clone()));
            }
        });
    }
    {
        let tui_ref = tui.clone();
        let editor_ref = editor.clone();
        replacement.set_on_input(move |data| {
            if data == "\r" {
                tui_ref.set_focus(Some(editor_ref.clone()));
            }
        });
    }
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    tui.set_focus(Some(replacement.clone()));
    let _ = tui.show_overlay(
        passive,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    tui.set_focus(Some(editor));
    let _ = tui.show_overlay(overlay.clone(), None);
    terminal.send_input("b");
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&replacement));

    terminal.send_input("1");
    terminal.send_input("\r");
    render_and_flush(&tui);
    assert_eq!(
        replacement.inputs(),
        vec!["1".to_string(), "\r".to_string()]
    );
    assert_eq!(overlay.inputs().as_slice(), ["b"]);
    assert!(tui_support::is_focused(&overlay));
    tui_support::stop(&tui);
}

#[test]
fn blocked_replacement_can_move_focus_internally_before_overlay_restore() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let base = Rc::new(Container::default());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let first_replacement = FocusableOverlay::new(&["FIRST"]);
    let second_replacement = FocusableOverlay::new(&["SECOND"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    {
        let tui_ref = tui.clone();
        let first_ref = first_replacement.clone();
        overlay.set_on_input(move |data| {
            if data == "b" {
                tui_ref.set_focus(Some(first_ref.clone()));
            }
        });
    }
    {
        let tui_ref = tui.clone();
        let second_ref = second_replacement.clone();
        first_replacement.set_on_input(move |data| {
            if data == "n" {
                tui_ref.set_focus(Some(second_ref.clone()));
            }
        });
    }
    {
        let tui_ref = tui.clone();
        let base_ref = base.clone();
        let editor_ref = editor.clone();
        second_replacement.set_on_input(move |data| {
            if data == "\r" {
                base_ref.clear();
                base_ref.add_child(editor_ref.clone());
                tui_ref.set_focus(Some(editor_ref.clone()));
            }
        });
    }
    base.add_child(editor.clone());
    base.add_child(first_replacement.clone());
    base.add_child(second_replacement.clone());
    tui.add_child(base);
    tui.set_focus(Some(editor));
    tui.start();
    let _ = tui.show_overlay(overlay.clone(), None);
    terminal.send_input("b");
    render_and_flush(&tui);
    terminal.send_input("n");
    render_and_flush(&tui);
    terminal.send_input("2");
    terminal.send_input("\r");
    render_and_flush(&tui);

    assert_eq!(overlay.inputs().as_slice(), ["b"]);
    assert_eq!(first_replacement.inputs(), vec!["n".to_string()]);
    assert_eq!(
        second_replacement.inputs(),
        vec!["2".to_string(), "\r".to_string()]
    );
    assert!(tui_support::is_focused(&overlay));
    tui_support::stop(&tui);
}

#[test]
fn removed_replacement_restores_overlay_even_when_overlay_pre_focus_differs_from_next_focus() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let base = Rc::new(Container::default());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let palette = FocusableOverlay::new(&["PALETTE"]);
    let replacement = FocusableOverlay::new(&["REPLACEMENT"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    {
        let tui_ref = tui.clone();
        let replacement_ref = replacement.clone();
        overlay.set_on_input(move |data| {
            if data == "b" {
                tui_ref.set_focus(Some(replacement_ref.clone()));
            }
        });
    }
    {
        let tui_ref = tui.clone();
        let base_ref = base.clone();
        let editor_ref = editor.clone();
        replacement.set_on_input(move |data| {
            if data == "\r" {
                base_ref.clear();
                base_ref.add_child(editor_ref.clone());
                tui_ref.set_focus(Some(editor_ref.clone()));
            }
        });
    }
    base.add_child(editor.clone());
    base.add_child(palette.clone());
    base.add_child(replacement.clone());
    tui.add_child(base);
    tui.set_focus(Some(palette));
    tui.start();
    let _ = tui.show_overlay(overlay.clone(), None);
    terminal.send_input("b");
    render_and_flush(&tui);
    terminal.send_input("\r");
    terminal.send_input("x");
    render_and_flush(&tui);

    assert_eq!(overlay.inputs().as_slice(), ["b", "x"]);
    assert_eq!(replacement.inputs(), vec!["\r".to_string()]);
    assert_eq!(editor.inputs(), Vec::<String>::new());
    assert!(tui_support::is_focused(&overlay));
    tui_support::stop(&tui);
}

#[test]
fn unfocus_target_releases_a_blocked_overlay_while_replacement_remains_focused() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let fallback = FocusableOverlay::new(&["FALLBACK"]);
    let target = FocusableOverlay::new(&["TARGET"]);
    let replacement = FocusableOverlay::new(&["REPLACEMENT"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    {
        let tui_ref = tui.clone();
        let fallback_ref = fallback.clone();
        replacement.set_on_input(move |data| {
            if data == "\r" {
                tui_ref.set_focus(Some(fallback_ref.clone()));
            }
        });
    }
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let overlay_handle = tui.show_overlay(overlay.clone(), None);
    {
        let tui_ref = tui.clone();
        let replacement_ref = replacement.clone();
        let target_ref = target.clone();
        let handle = overlay_handle;
        overlay.set_on_input(move |data| {
            if data == "b" {
                tui_ref.set_focus(Some(replacement_ref.clone()));
                handle.unfocus(Some(OverlayUnfocusOptions {
                    target: Some(target_ref.clone()),
                }));
            }
        });
    }

    terminal.send_input("b");
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&replacement));
    terminal.send_input("\r");
    terminal.send_input("x");
    render_and_flush(&tui);

    assert_eq!(overlay.inputs().as_slice(), ["b"]);
    assert_eq!(replacement.inputs(), vec!["\r".to_string()]);
    assert_eq!(fallback.inputs(), Vec::<String>::new());
    assert_eq!(target.inputs(), vec!["x".to_string()]);
    tui_support::stop(&tui);
}

#[test]
fn handle_input_restores_focus_to_a_visible_focused_overlay_after_base_focus_steal() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let replacement = FocusableOverlay::new(&["REPLACEMENT"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let _ = tui.show_overlay(overlay.clone(), None);
    assert!(tui_support::is_focused(&overlay));
    tui.set_focus(Some(replacement));
    tui.set_focus(Some(editor.clone()));
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(overlay.inputs(), vec!["x".to_string()]);
    assert_eq!(editor.inputs(), Vec::<String>::new());
    assert!(tui_support::is_focused(&overlay));
    tui_support::stop(&tui);
}

#[test]
fn handle_input_restores_focus_to_explicitly_focused_raw_sub_overlay_after_base_focus_steal() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let controller = FocusableOverlay::new(&["CONTROLLER"]);
    let sub_overlay = FocusableOverlay::new(&["SUB"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let _ = tui.show_overlay(controller.clone(), None);
    let sub_handle = tui.show_overlay(
        sub_overlay.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    sub_handle.focus();
    tui.set_focus(Some(editor.clone()));
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(sub_overlay.inputs(), vec!["x".to_string()]);
    assert_eq!(controller.inputs(), Vec::<String>::new());
    assert_eq!(editor.inputs(), Vec::<String>::new());
    tui_support::stop(&tui);
}

#[test]
fn passive_non_capturing_overlay_does_not_regain_input_after_base_focus() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let passive = FocusableOverlay::new(&["PASSIVE"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let _ = tui.show_overlay(
        passive.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(editor.inputs(), vec!["x".to_string()]);
    assert_eq!(passive.inputs(), Vec::<String>::new());
    assert!(tui_support::is_focused(&editor));
    tui_support::stop(&tui);
}

#[test]
fn explicitly_focused_non_capturing_overlay_regains_input_after_base_focus_steal() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["NC"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.focus();
    tui.set_focus(Some(editor.clone()));
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(overlay.inputs(), vec!["x".to_string()]);
    assert_eq!(editor.inputs(), Vec::<String>::new());
    tui_support::stop(&tui);
}

#[test]
fn unfocus_prevents_visible_overlay_from_regaining_input() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(overlay.clone(), None);
    handle.unfocus(None);
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(editor.inputs(), vec!["x".to_string()]);
    assert_eq!(overlay.inputs(), Vec::<String>::new());
    assert!(tui_support::is_focused(&editor));
    tui_support::stop(&tui);
}

#[test]
fn set_focus_null_explicitly_clears_visible_overlay_restore() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let _ = tui.show_overlay(overlay.clone(), None);
    tui.set_focus(None);
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(overlay.inputs(), Vec::<String>::new());
    assert!(!tui_support::is_focused(&overlay));
    tui_support::stop(&tui);
}

#[test]
fn blocked_replacement_set_focus_null_resumes_the_visible_overlay() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let replacement = FocusableOverlay::new(&["REPLACEMENT"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    {
        let tui_ref = tui.clone();
        replacement.set_on_input(move |data| {
            if data == "\r" {
                tui_ref.set_focus(None);
            }
        });
    }
    {
        let tui_ref = tui.clone();
        let replacement_ref = replacement.clone();
        overlay.set_on_input(move |data| {
            if data == "b" {
                tui_ref.set_focus(Some(replacement_ref.clone()));
            }
        });
    }
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let _ = tui.show_overlay(overlay.clone(), None);
    terminal.send_input("b");
    render_and_flush(&tui);
    terminal.send_input("\r");
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(replacement.inputs(), vec!["\r".to_string()]);
    assert_eq!(overlay.inputs().as_slice(), ["b", "x"]);
    assert!(tui_support::is_focused(&overlay));
    tui_support::stop(&tui);
}

#[test]
fn temporarily_invisible_focused_overlay_falls_back_without_losing_restore_eligibility() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    let visible = Rc::new(Cell::new(true));
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let visible_check = visible.clone();
    let _ = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            visible: Some(Box::new(move |_, _| visible_check.get())),
            ..OverlayOptions::default()
        }),
    );
    tui.set_focus(Some(editor.clone()));
    visible.set(false);
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(editor.inputs(), vec!["x".to_string()]);
    assert_eq!(overlay.inputs(), Vec::<String>::new());
    visible.set(true);
    terminal.send_input("y");
    render_and_flush(&tui);
    assert_eq!(editor.inputs(), vec!["x".to_string()]);
    assert_eq!(overlay.inputs(), vec!["y".to_string()]);
    tui_support::stop(&tui);
}

#[test]
fn temporarily_invisible_focused_overlay_with_null_pre_focus_restores_when_visible_again() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    let visible = Rc::new(Cell::new(true));
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let visible_check = visible.clone();
    let _ = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            visible: Some(Box::new(move |_, _| visible_check.get())),
            ..OverlayOptions::default()
        }),
    );
    visible.set(false);
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(overlay.inputs(), Vec::<String>::new());
    visible.set(true);
    terminal.send_input("y");
    render_and_flush(&tui);
    assert_eq!(overlay.inputs(), vec!["y".to_string()]);
    tui_support::stop(&tui);
}

#[test]
fn cyclic_overlay_pre_focus_ancestry_does_not_hang_focus_changes() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(overlay.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.focus();
    tui.set_focus(Some(editor.clone()));
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(editor.inputs(), vec!["x".to_string()]);
    assert_eq!(editor.inputs(), vec!["x".to_string()]);
    tui_support::stop(&tui);
}

#[test]
fn handle_input_restores_the_focus_order_top_overlay_after_base_focus_steal() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let lower = FocusableOverlay::new(&["LOWER"]);
    let upper = FocusableOverlay::new(&["UPPER"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let lower_handle = tui.show_overlay(lower.clone(), None);
    let _ = tui.show_overlay(upper.clone(), None);
    lower_handle.focus();
    tui.set_focus(Some(editor.clone()));
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(lower.inputs(), vec!["x".to_string()]);
    assert_eq!(upper.inputs(), Vec::<String>::new());
    assert_eq!(editor.inputs(), Vec::<String>::new());
    tui_support::stop(&tui);
}

#[test]
fn hide_overlay_does_not_reassign_focus_when_topmost_overlay_is_non_capturing() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let capturing = FocusableOverlay::new(&["CAP"]);
    let non_capturing = FocusableOverlay::new(&["NC"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor));
    tui.start();
    let _ = tui.show_overlay(capturing.clone(), None);
    let _ = tui.show_overlay(
        non_capturing,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    assert!(tui_support::is_focused(&capturing));
    tui.hide_overlay();
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&capturing));
    tui_support::stop(&tui);
}

#[test]
fn multiple_capturing_and_non_capturing_overlays_restore_focus_through_removals() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let c1 = FocusableOverlay::new(&["C1"]);
    let n1 = FocusableOverlay::new(&["N1"]);
    let c2 = FocusableOverlay::new(&["C2"]);
    let n2 = FocusableOverlay::new(&["N2"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let c1_handle = tui.show_overlay(c1.clone(), None);
    let _ = tui.show_overlay(
        n1,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let c2_handle = tui.show_overlay(c2.clone(), None);
    let _ = tui.show_overlay(
        n2,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    assert!(tui_support::is_focused(&c2));
    c2_handle.hide();
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&c1));
    c1_handle.hide();
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    tui_support::stop(&tui);
}

#[test]
fn capturing_overlay_unfocus_on_topmost_capturing_overlay_falls_back_to_pre_focus() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let capturing = FocusableOverlay::new(&["CAP"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(capturing.clone(), None);
    assert!(tui_support::is_focused(&capturing));
    handle.unfocus(None);
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!tui_support::is_focused(&capturing));
    tui_support::stop(&tui);
}

// === no-op guards ===

#[test]
fn focus_on_hidden_overlay_is_a_no_op() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.set_hidden(true);
    handle.focus();
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!handle.is_focused());
    tui_support::stop(&tui);
}

#[test]
fn focus_after_hide_is_a_no_op() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.hide();
    handle.focus();
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!handle.is_focused());
    tui_support::stop(&tui);
}

#[test]
fn unfocus_when_overlay_does_not_have_focus_is_a_no_op() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let handle = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    handle.unfocus(None);
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!handle.is_focused());
    tui_support::stop(&tui);
}

#[test]
fn unfocus_with_null_pre_focus_clears_focus_and_does_not_route_input_back_to_overlay() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let handle = tui.show_overlay(overlay.clone(), None);
    assert!(tui_support::is_focused(&overlay));
    handle.unfocus(None);
    assert!(!tui_support::is_focused(&overlay));
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(overlay.inputs(), Vec::<String>::new());
    assert!(!handle.is_focused());
    tui_support::stop(&tui);
}

// === focus cycle prevention ===

#[test]
fn toggle_focus_between_non_capturing_overlays_then_unfocus_returns_to_editor() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal);
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let a = FocusableOverlay::new(&["A"]);
    let b = FocusableOverlay::new(&["B"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let a_handle = tui.show_overlay(
        a,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let b_handle = tui.show_overlay(
        b,
        Some(OverlayOptions {
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    a_handle.focus();
    b_handle.focus();
    a_handle.focus();
    a_handle.unfocus(None);
    render_and_flush(&tui);
    assert!(tui_support::is_focused(&editor));
    assert!(!a_handle.is_focused());
    assert!(!b_handle.is_focused());
    tui_support::stop(&tui);
}

#[test]
fn explicit_unfocus_target_supports_cycling_between_three_overlays_and_editor() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let a = FocusableOverlay::new(&["A"]);
    let b = FocusableOverlay::new(&["B"]);
    let c = FocusableOverlay::new(&["C"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor.clone()));
    tui.start();
    let a_handle = tui.show_overlay(a.clone(), None);
    let b_handle = tui.show_overlay(b.clone(), None);
    let c_handle = tui.show_overlay(c.clone(), None);

    a_handle.focus();
    terminal.send_input("a");
    render_and_flush(&tui);
    b_handle.focus();
    terminal.send_input("b");
    render_and_flush(&tui);
    c_handle.focus();
    terminal.send_input("c");
    render_and_flush(&tui);
    c_handle.unfocus(Some(OverlayUnfocusOptions {
        target: Some(editor.clone()),
    }));
    terminal.send_input("e");
    render_and_flush(&tui);
    a_handle.focus();
    terminal.send_input("A");
    render_and_flush(&tui);
    a_handle.unfocus(Some(OverlayUnfocusOptions {
        target: Some(editor.clone()),
    }));
    terminal.send_input("E");
    render_and_flush(&tui);

    assert_eq!(a.inputs(), vec!["a".to_string(), "A".to_string()]);
    assert_eq!(b.inputs(), vec!["b".to_string()]);
    assert_eq!(c.inputs(), vec!["c".to_string()]);
    assert_eq!(editor.inputs(), vec!["e".to_string(), "E".to_string()]);
    assert!(tui_support::is_focused(&editor));
    tui_support::stop(&tui);
}

#[test]
fn explicit_null_unfocus_target_clears_focus_without_restoring_overlays() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = FocusableOverlay::new(&["OVERLAY"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let handle = tui.show_overlay(overlay.clone(), None);
    handle.unfocus(Some(OverlayUnfocusOptions { target: None }));
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(overlay.inputs(), Vec::<String>::new());
    assert!(!handle.is_focused());
    tui_support::stop(&tui);
}

#[test]
fn hiding_focused_overlay_falls_back_to_next_visual_frontmost_overlay() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    let a = FocusableOverlay::new(&["A"]);
    let b = FocusableOverlay::new(&["B"]);
    let c = FocusableOverlay::new(&["C"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor));
    tui.start();
    let a_handle = tui.show_overlay(a.clone(), None);
    let b_handle = tui.show_overlay(b, None);
    let _ = tui.show_overlay(c.clone(), None);
    a_handle.focus();
    b_handle.focus();
    b_handle.set_hidden(true);
    terminal.send_input("x");
    render_and_flush(&tui);
    assert_eq!(a.inputs(), vec!["x".to_string()]);
    assert_eq!(c.inputs(), Vec::<String>::new());
    assert!(tui_support::is_focused(&a));
    tui_support::stop(&tui);
}

// === rendering order ===

#[test]
fn focus_on_already_focused_overlay_bumps_visual_order() {
    let terminal = VirtualTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor));
    tui.start();
    let a_handle = tui.show_overlay(
        StaticOverlay::new(vec!["A"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let _ = tui.show_overlay(
        StaticOverlay::new(vec!["B"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    a_handle.focus();
    let _ = tui.show_overlay(
        StaticOverlay::new(vec!["C"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'C'
    );
    a_handle.focus();
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'A'
    );
    assert!(a_handle.is_focused());
    tui_support::stop(&tui);
}

#[test]
fn default_rendering_order_for_overlapping_overlays_follows_creation_order() {
    let terminal = VirtualTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let _ = tui.show_overlay(
        StaticOverlay::new(vec!["A"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let _ = tui.show_overlay(
        StaticOverlay::new(vec!["B"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'B'
    );
    tui_support::stop(&tui);
}

#[test]
fn focus_on_lower_overlay_renders_it_on_top() {
    let terminal = VirtualTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let lower = tui.show_overlay(
        StaticOverlay::new(vec!["A"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let _ = tui.show_overlay(
        StaticOverlay::new(vec!["B"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'B'
    );
    lower.focus();
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'A'
    );
    tui_support::stop(&tui);
}

#[test]
fn focusing_middle_overlay_places_it_on_top_while_preserving_others_relative_order() {
    let terminal = VirtualTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let _ = tui.show_overlay(
        StaticOverlay::new(vec!["A"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let middle = tui.show_overlay(
        StaticOverlay::new(vec!["B"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let top = tui.show_overlay(
        StaticOverlay::new(vec!["C"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'C'
    );
    middle.focus();
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'B'
    );
    middle.hide();
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'C'
    );
    top.hide();
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'A'
    );
    tui_support::stop(&tui);
}

#[test]
fn capturing_overlay_hidden_and_shown_again_renders_on_top_after_unhide() {
    let terminal = VirtualTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.add_child(Rc::new(EmptyContent));
    tui.start();
    let _ = tui.show_overlay(
        StaticOverlay::new(vec!["A"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let capturing = tui.show_overlay(
        StaticOverlay::new(vec!["B"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            ..OverlayOptions::default()
        }),
    );
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'B'
    );
    capturing.set_hidden(true);
    let _ = tui.show_overlay(
        StaticOverlay::new(vec!["C"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'C'
    );
    capturing.set_hidden(false);
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'B'
    );
    tui_support::stop(&tui);
}

#[test]
fn unfocus_does_not_change_visual_order_until_another_overlay_is_focused() {
    let terminal = VirtualTerminal::new(20, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    let editor = FocusableOverlay::new(&["EDITOR"]);
    tui.add_child(Rc::new(EmptyContent));
    tui.set_focus(Some(editor));
    tui.start();
    let a = tui.show_overlay(
        StaticOverlay::new(vec!["A"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    let b = tui.show_overlay(
        StaticOverlay::new(vec!["B"]),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(0)),
            width: Some(SizeValue::Cells(1)),
            non_capturing: true,
            ..OverlayOptions::default()
        }),
    );
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'B'
    );
    a.focus();
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'A'
    );
    a.unfocus(None);
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'A'
    );
    b.focus();
    render_and_flush(&tui);
    assert_eq!(
        terminal
            .get_viewport()
            .first()
            .map_or(' ', |line| line.chars().next().unwrap_or(' ')),
        'B'
    );
    tui_support::stop(&tui);
}
