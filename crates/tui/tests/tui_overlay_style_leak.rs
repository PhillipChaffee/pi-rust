//! `tui-overlay-style-leak.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): compositing must not leak
//! italic styling into the lines below an overlay, both when a trailing
//! reset sits beyond the last visible column and when overlay slicing drops
//! trailing SGR resets.
//!
//! Upstream read the attribute off xterm's buffer cells; the port reads the
//! emulator's per-cell italic through [`tui_support::TestTerminal::is_italic`].

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::rc::Rc;

use pi_tui::tui::{Component, OverlayOptions, SizeValue};

use tui_support::{TestTerminal, render_and_flush};

/// The suite's `StaticLines`.
struct StaticLines {
    lines: Vec<String>,
}

impl Component for StaticLines {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines.clone()
    }

    fn invalidate(&self) {}
}

/// The suite's `StaticOverlay`.
struct StaticOverlayLine {
    line: String,
}

impl Component for StaticOverlayLine {
    fn render(&self, _width: usize) -> Vec<String> {
        vec![self.line.clone()]
    }

    fn invalidate(&self) {}
}

#[test]
fn does_not_leak_styles_when_a_trailing_reset_sits_beyond_the_last_visible_column_no_overlay() {
    let width: u16 = 20;
    let base_line = format!("\x1b[3m{}\x1b[23m", "X".repeat(usize::from(width)));

    let terminal = TestTerminal::new(width, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.add_child(Rc::new(StaticLines {
        lines: vec![base_line, "INPUT".to_string()],
    }));
    tui.start();
    render_and_flush(&tui);
    assert!(!terminal.is_italic(1, 0), "line 1 col 0 must not be italic");
    tui_support::stop(&tui);
}

#[test]
fn does_not_leak_styles_when_overlay_slicing_drops_trailing_sgr_resets() {
    let width: u16 = 20;
    let base_line = format!("\x1b[3m{}\x1b[23m", "X".repeat(usize::from(width)));

    let terminal = TestTerminal::new(width, 6);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.add_child(Rc::new(StaticLines {
        lines: vec![base_line, "INPUT".to_string()],
    }));

    let _ = tui.show_overlay(
        Rc::new(StaticOverlayLine {
            line: "OVR".to_string(),
        }),
        Some(OverlayOptions {
            row: Some(SizeValue::Cells(0)),
            col: Some(SizeValue::Cells(5)),
            width: Some(SizeValue::Cells(3)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    assert!(!terminal.is_italic(1, 0), "line 1 col 0 must not be italic");
    tui_support::stop(&tui);
}
