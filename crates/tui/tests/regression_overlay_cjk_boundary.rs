//! `regression-overlay-cjk-boundary.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): an overlay that starts
//! inside a wide grapheme excludes it from the before-segment and composites
//! at the requested column.

use pi_tui::tui::composite_tui_line;
use pi_tui::utils::{extract_segments, slice_by_column, visible_width};

#[test]
fn excludes_a_wide_grapheme_from_before_when_overlay_starts_inside_it() {
    let segments = extract_segments("abcd让EFGH", 5, 9, 11, true);

    assert_eq!(segments.before, "abcd");
    assert_eq!(segments.before_width, 4);
    assert_eq!(visible_width(&segments.before), segments.before_width);
    assert_eq!(segments.after, "H");
    assert_eq!(segments.after_width, 1);
}

#[test]
fn keeps_ascii_before_segment_behavior_at_the_same_boundary() {
    let segments = extract_segments("abcdG EFGH", 5, 9, 11, true);

    assert_eq!(segments.before, "abcdG");
    assert_eq!(segments.before_width, 5);
    assert_eq!(visible_width(&segments.before), segments.before_width);
}

#[test]
fn composites_an_overlay_at_the_requested_column_when_it_starts_inside_a_wide_grapheme() {
    let out = composite_tui_line("abcd让EFGH", "│XX│", 5, 4, 20);
    let prefix = slice_by_column(&out, 0, 5, true);
    let overlay = slice_by_column(&out, 5, 4, true);

    assert!(!out.contains("让"));
    assert_eq!(visible_width(&out), 20);
    assert_eq!(visible_width(&prefix), 5);
    assert_eq!(visible_width(&overlay), 4);
    assert!(overlay.contains("│XX│"));
}

#[test]
fn composites_an_overlay_when_it_starts_at_a_wide_grapheme_boundary() {
    let out = composite_tui_line("abcd让EFGH", "│XX│", 4, 4, 20);
    let overlay = slice_by_column(&out, 4, 4, true);

    assert!(!out.contains("让"));
    assert_eq!(visible_width(&out), 20);
    assert_eq!(visible_width(&overlay), 4);
    assert!(overlay.contains("│XX│"));
}
