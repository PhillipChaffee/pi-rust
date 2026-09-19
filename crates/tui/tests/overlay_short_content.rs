//! `overlay-short-content.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): an overlay renders even when
//! the base content is shorter than the terminal height — the composite
//! pads to the terminal height so overlays have screen-relative rows.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::rc::Rc;

use pi_tui::tui::Component;

use tui_support::{TestTerminal, wait_for_render};

/// The suite's `SimpleContent`.
struct SimpleContent {
    lines: Vec<String>,
}

impl Component for SimpleContent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines.clone()
    }

    fn invalidate(&self) {}
}

/// The suite's `SimpleOverlay`.
struct SimpleOverlay;

impl Component for SimpleOverlay {
    fn render(&self, _width: usize) -> Vec<String> {
        vec![
            "OVERLAY_TOP".to_string(),
            "OVERLAY_MID".to_string(),
            "OVERLAY_BOT".to_string(),
        ]
    }

    fn invalidate(&self) {}
}

#[test]
fn renders_overlay_when_content_is_shorter_than_terminal_height() {
    // Terminal has 24 rows, but content only has 3 lines.
    let terminal = TestTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());

    // Only 3 lines of content.
    tui.add_child(Rc::new(SimpleContent {
        lines: vec![
            "Line 1".to_string(),
            "Line 2".to_string(),
            "Line 3".to_string(),
        ],
    }));

    // Show overlay centered - should be around row 10 in a 24-row terminal.
    let _ = tui.show_overlay(Rc::new(SimpleOverlay), None);

    // Trigger render (upstream relies on the request `start()` made).
    tui.start();
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport.iter().any(|line| line.contains("OVERLAY")),
        "Overlay should be visible when content is shorter than terminal"
    );

    tui_support::stop(&tui);
}
