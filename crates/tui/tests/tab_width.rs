//! 1:1 port of the pure-utils tests from
//! `packages/tui/test/tab-width.test.ts` (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).
//!
//! The fourth upstream test ("keeps tab-containing overlays on one physical
//! terminal row") needs `TuiMainScreen`, the overlay API, and the headless
//! virtual terminal; it rides with the main-screen renderer port ticket.

use pi_tui::utils::{extract_segments, normalize_terminal_output, slice_with_width, visible_width};

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
