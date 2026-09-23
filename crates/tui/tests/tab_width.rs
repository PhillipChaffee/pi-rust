//! 1:1 port of `packages/tui/test/tab-width.test.ts` (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): the pure-utils tests and
//! the tab-containing overlay test through the shared harness.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::rc::Rc;

use pi_tui::tui::{Component, OverlayOptions, SizeValue};
use pi_tui::utils::{extract_segments, normalize_terminal_output, slice_with_width, visible_width};

use tui_support::VirtualTerminal;

/// The fourth test's full-viewport content, upstream `FullViewportContent`:
/// three lines padded to the full width so the overlay row stands out.
struct FullViewportContent;

impl Component for FullViewportContent {
    fn render(&self, width: usize) -> Vec<String> {
        ["base 0", "base 1", "base 2"]
            .iter()
            .map(|line| format!("{line:<width$}"))
            .collect()
    }

    fn invalidate(&self) {}
}

/// The fourth test's status overlay, upstream `TabStatusOverlay`: one line
/// whose tab must never reach the terminal as a raw `\t`.
struct TabStatusOverlay;

impl Component for TabStatusOverlay {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["\tX".to_string()]
    }

    fn invalidate(&self) {}
}

#[test]
fn keeps_slice_helper_widths_consistent_with_visible_width() {
    let text = "out 192M\t.pi/skill-tests/results-ha";
    let slice = slice_with_width(text, 0, 10, true);

    assert_eq!(slice.text, "out 192M");
    assert_eq!(slice.width, 8);
    assert_eq!(visible_width(&slice.text), slice.width);
}

#[test]
fn keeps_overlay_segment_widths_consistent_with_visible_width() {
    let text = "out 192M\t.pi/skill-tests/results-ha";
    let segments = extract_segments(text, 10, 13, 10, true);

    assert_eq!(segments.before, "out 192M");
    assert_eq!(segments.before_width, 8);
    assert_eq!(visible_width(&segments.before), segments.before_width);

    let tab_fits = extract_segments(text, 11, 13, 10, true);
    assert_eq!(tab_fits.before, "out 192M\t");
    assert_eq!(tab_fits.before_width, 11);
    assert_eq!(visible_width(&tab_fits.before), tab_fits.before_width);
}

#[test]
fn keeps_tabs_inside_terminal_control_sequences_byte_identical() {
    let control_sequences = [
        "\x1b]8;;https://example.test/a\tb\x07",
        "\x1b]0;window\ttitle\x1b\\",
        "\x1b_payload\tdata\x1b\\",
    ];

    for control_sequence in control_sequences {
        assert_eq!(
            normalize_terminal_output(&format!("{control_sequence}label\ttext")),
            format!("{control_sequence}label   text")
        );
    }
}

#[test]
fn keeps_tab_containing_overlays_on_one_physical_terminal_row() {
    let terminal = VirtualTerminal::new(16, 3);
    let tui = tui_support::new_test_tui(terminal.clone());
    tui.add_child(Rc::new(FullViewportContent));
    let _ = tui.show_overlay(
        Rc::new(TabStatusOverlay),
        Some(OverlayOptions {
            width: Some(SizeValue::Cells(4)),
            row: Some(SizeValue::Cells(1)),
            col: Some(SizeValue::Cells(4)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();

    tui_support::wait_for_render(&tui);
    assert_eq!(
        terminal.get_viewport(),
        ["base 0          ", "base   X        ", "base 2          "]
    );
    assert!(!terminal.write_log().contains('\t'));

    tui_support::stop(&tui);
}
