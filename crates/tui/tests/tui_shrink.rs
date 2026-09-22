//! `tui-shrink.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): shrinking content clears the
//! rendered lines, through the differential renderer's deleted-lines branch.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::rc::Rc;

use pi_tui::tui::{Component, TuiStopOptions};

use tui_support::{VirtualTerminal, wait_for_render};

/// The suite's `Lines`.
struct Lines {
    lines: Vec<String>,
}

impl Component for Lines {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines.clone()
    }

    fn invalidate(&self) {}
}

fn stop(tui: &pi_tui::tui::Tui) {
    tui.stop(TuiStopOptions::default());
}

#[test]
fn clears_all_rendered_lines_when_content_shrinks_to_zero() {
    let terminal = VirtualTerminal::new(40, 10);
    let tui = tui_support::new_test_tui(terminal.clone());
    let content = Rc::new(Lines {
        lines: vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string(),
        ],
    });
    tui.add_child(content);
    tui.start();
    wait_for_render(&tui);

    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("first"))
            && terminal
                .get_viewport()
                .iter()
                .any(|line| line.contains("second"))
            && terminal
                .get_viewport()
                .iter()
                .any(|line| line.contains("third")),
        "initial content rendered: {:?}",
        terminal.get_viewport()
    );

    tui.clear();
    tui.request_render(false);
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        !viewport.iter().any(|line| line.contains("first")),
        "first line should be cleared"
    );
    assert!(
        !viewport.iter().any(|line| line.contains("second")),
        "second line should be cleared"
    );
    assert!(
        !viewport.iter().any(|line| line.contains("third")),
        "third line should be cleared"
    );

    stop(&tui);
}
