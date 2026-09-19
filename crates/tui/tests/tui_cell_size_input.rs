//! `tui-cell-size-input.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): bare escapes and later user
//! input survive the cell-size response consumer, even when a cell-size
//! query was sent at startup.
//!
//! Restatement: upstream's `withImageTerminal` manipulated `process.env` to
//! make `getCapabilities().images` truthy; the port's `queryCellSize` gate
//! is an injected probe (the capability matrix lands with #51), so the
//! helper injects an image-capable probe instead.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::RefCell;
use std::rc::Rc;

use pi_tui::terminal_image::{CellDimensions, get_cell_dimensions, set_cell_dimensions};
use pi_tui::tui::{Component, Tui};

use tui_support::TestTerminal;

/// The cell-size suite's `InputRecorder`.
struct InputRecorder {
    inputs: RefCell<Vec<String>>,
}

impl Component for InputRecorder {
    fn render(&self, _width: usize) -> Vec<String> {
        vec![String::new()]
    }

    fn handle_input(&self, data: &str) {
        self.inputs.borrow_mut().push(data.to_string());
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {}
}

fn with_image_terminal(setup: impl FnOnce()) {
    // Upstream reset the capability cache around the suite; the port's probe
    // is constructed per TUI, so the injection is the whole story.
    setup();
}

#[test]
fn forwards_bare_escape_even_when_a_cell_size_query_was_sent_at_startup() {
    with_image_terminal(|| {
        let terminal = TestTerminal::new(80, 24);
        let tui: Rc<Tui> = tui_support::new_test_tui_with_images(terminal.clone());
        let recorder = Rc::new(InputRecorder {
            inputs: RefCell::new(Vec::new()),
        });
        let focus_target: Rc<dyn Component> = recorder.clone();
        tui.set_focus(Some(focus_target));
        tui.start();

        terminal.send_input("\x1b");

        assert_eq!(recorder.inputs.borrow().len(), 1);
        assert_eq!(
            recorder.inputs.borrow().first().map(String::as_str),
            Some("\x1b")
        );
        tui.stop(pi_tui::tui::TuiStopOptions::default());
    });
}

#[test]
fn consumes_cell_size_responses_and_still_forwards_later_user_input() {
    with_image_terminal(|| {
        set_cell_dimensions(CellDimensions {
            width_px: 9,
            height_px: 18,
        });

        let terminal = TestTerminal::new(80, 24);
        let tui: Rc<Tui> = tui_support::new_test_tui_with_images(terminal.clone());
        let recorder = Rc::new(InputRecorder {
            inputs: RefCell::new(Vec::new()),
        });
        let focus_target: Rc<dyn Component> = recorder.clone();
        tui.set_focus(Some(focus_target));
        tui.start();

        terminal.send_input("\x1b[6;20;10t");
        assert!(
            recorder.inputs.borrow().is_empty(),
            "the cell-size response never reaches the focused component"
        );
        assert_eq!(
            get_cell_dimensions(),
            CellDimensions {
                width_px: 10,
                height_px: 20
            }
        );

        terminal.send_input("q");
        assert_eq!(recorder.inputs.borrow().as_slice(), ["q"]);
        tui.stop(pi_tui::tui::TuiStopOptions::default());
    });
}
