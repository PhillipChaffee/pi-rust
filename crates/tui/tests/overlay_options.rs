//! `overlay-options.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): overlay width overflow
//! protection, width percentages, anchor positioning, margins, offsets,
//! percentage positioning, maxHeight, absolute positioning, and stacked
//! overlays, all through the TUI core's compositing.
//!
//! The suites render through [`tui_support::new_test_tui`]'s `TuiMainScreen`;
//! see the support module.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::rc::Rc;

use pi_tui::tui::{OverlayAnchor, OverlayMargin, OverlayMarginSides, OverlayOptions, SizeValue};

use tui_support::{EmptyContent, StaticOverlay, VirtualTerminal, render_and_flush};

// === width overflow protection ===

#[test]
fn truncates_overlay_lines_that_exceed_declared_width() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    // Overlay declares width 20 but renders lines much wider.
    let overlay = StaticOverlay::new(vec![&"X".repeat(100)]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            width: Some(SizeValue::Cells(20)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    // Should not crash, and every row is a string (upstream asserted
    // `line !== undefined`); line length is only a rough check.
    let viewport = terminal.get_viewport();
    for line in &viewport {
        let _ = line;
    }
    tui_support::stop(&tui);
}

#[test]
fn handles_overlay_with_complex_ansi_sequences_without_crashing() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    // Simulate complex ANSI content like the crash log showed.
    let complex_line =
        "\x1b[48;2;40;50;40m \x1b[38;2;128;128;128mSome styled content\x1b[39m\x1b[49m".to_string()
            + "\x1b]8;;http://example.com\u{7}link\x1b]8;;\u{7}"
            + &" more content ".repeat(10);
    let overlay = StaticOverlay::new(vec![&complex_line, &complex_line, &complex_line]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            width: Some(SizeValue::Cells(60)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    // Should not crash.
    let viewport = terminal.get_viewport();
    assert!(!viewport.is_empty());
    tui_support::stop(&tui);
}

#[test]
fn handles_overlay_composited_on_styled_base_content() {
    struct StyledContent;
    impl pi_tui::tui::Component for StyledContent {
        fn render(&self, width: usize) -> Vec<String> {
            let styled_line = format!("\x1b[1m\x1b[38;2;255;0;0m{}\x1b[0m", "X".repeat(width));
            vec![styled_line.clone(), styled_line.clone(), styled_line]
        }
        fn invalidate(&self) {}
    }

    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());

    let overlay = StaticOverlay::new(vec!["OVERLAY"]);

    tui.add_child(Rc::new(StyledContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            width: Some(SizeValue::Cells(20)),
            anchor: Some(OverlayAnchor::Center),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    // Should not crash and overlay should be visible.
    let viewport = terminal.get_viewport();
    assert!(
        viewport.iter().any(|line| line.contains("OVERLAY")),
        "Overlay should be visible"
    );
    tui_support::stop(&tui);
}

#[test]
fn handles_wide_characters_at_overlay_boundary() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    // Wide chars (each takes 2 columns) at the edge of declared width.
    let wide_char_line = "中文日本語한글テスト漢字"; // Mix of CJK chars
    let overlay = StaticOverlay::new(vec![wide_char_line]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            width: Some(SizeValue::Cells(15)), // Odd width to potentially hit boundary
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    // Should not crash.
    let viewport = terminal.get_viewport();
    assert!(!viewport.is_empty());
    tui_support::stop(&tui);
}

#[test]
fn handles_overlay_positioned_at_terminal_edge() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    // Overlay positioned at right edge with content that exceeds declared width.
    let overlay = StaticOverlay::new(vec![&"X".repeat(50)]);

    tui.add_child(Rc::new(EmptyContent));
    // Position at col 60 with width 20 - should fit exactly at right edge.
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            col: Some(SizeValue::Cells(60)),
            width: Some(SizeValue::Cells(20)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    // Should not crash.
    let viewport = terminal.get_viewport();
    assert!(!viewport.is_empty());
    tui_support::stop(&tui);
}

#[test]
fn handles_overlay_on_base_content_with_osc_sequences() {
    struct HyperlinkContent;
    impl pi_tui::tui::Component for HyperlinkContent {
        fn render(&self, width: usize) -> Vec<String> {
            let hyperlink = "\x1b]8;;file:///path/to/file.ts\u{7}file.ts\x1b]8;;\u{7}";
            let line = format!(
                "See {hyperlink} for details {}",
                "X".repeat(width.saturating_sub(30))
            );
            vec![line.clone(), line.clone(), line]
        }
        fn invalidate(&self) {}
    }

    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());

    let overlay = StaticOverlay::new(vec!["OVERLAY-TEXT"]);

    tui.add_child(Rc::new(HyperlinkContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::Center),
            width: Some(SizeValue::Cells(20)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    // Should not crash - this was the original bug scenario.
    let viewport = terminal.get_viewport();
    assert!(!viewport.is_empty());
    tui_support::stop(&tui);
}

// === width percentage ===

#[test]
fn renders_overlay_at_percentage_of_terminal_width() {
    let terminal = VirtualTerminal::new(100, 24);
    let tui = tui_support::new_test_tui(terminal);
    let overlay = StaticOverlay::new(vec!["test"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            width: Some(SizeValue::Percent(50.0)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    assert_eq!(overlay.requested_width(), Some(50));
    tui_support::stop(&tui);
}

#[test]
fn respects_min_width_when_width_percent_results_in_smaller_width() {
    let terminal = VirtualTerminal::new(100, 24);
    let tui = tui_support::new_test_tui(terminal);
    let overlay = StaticOverlay::new(vec!["test"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay.clone(),
        Some(OverlayOptions {
            width: Some(SizeValue::Percent(10.0)),
            min_width: Some(30),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    assert_eq!(overlay.requested_width(), Some(30));
    tui_support::stop(&tui);
}

// === anchor positioning ===

#[test]
fn positions_overlay_at_top_left() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["TOP-LEFT"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(10)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport
            .first()
            .is_some_and(|line| line.starts_with("TOP-LEFT")),
        "Expected TOP-LEFT at start, got: {:?}",
        viewport.first()
    );
    tui_support::stop(&tui);
}

#[test]
fn positions_overlay_at_bottom_right() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["BTM-RIGHT"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::BottomRight),
            width: Some(SizeValue::Cells(10)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    // Should be on last row, ending at last column.
    let viewport = terminal.get_viewport();
    let last_row = viewport.get(23).map(String::as_str);
    assert!(
        last_row.is_some_and(|line| line.contains("BTM-RIGHT")),
        "Expected BTM-RIGHT on last row, got: {last_row:?}"
    );
    assert!(
        last_row.is_some_and(|line| line.trim_end().ends_with("BTM-RIGHT")),
        "Expected BTM-RIGHT at end, got: {last_row:?}"
    );
    tui_support::stop(&tui);
}

#[test]
fn positions_overlay_at_top_center() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["CENTERED"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopCenter),
            width: Some(SizeValue::Cells(10)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    // Should be on first row, centered horizontally.
    let viewport = terminal.get_viewport();
    let first_row = viewport.first().map(String::as_str);
    assert!(
        first_row.is_some_and(|line| line.contains("CENTERED")),
        "Expected CENTERED on first row, got: {first_row:?}"
    );
    // Check it's roughly centered (col 35 for width 10 in 80 col terminal).
    let col_index = first_row.map_or(-1, |line| {
        i64::try_from(line.find("CENTERED").unwrap_or(0)).unwrap_or(-1)
    });
    assert!(
        (30..=40).contains(&col_index),
        "Expected centered, got col {col_index}"
    );
    tui_support::stop(&tui);
}

// === margin ===

#[test]
fn clamps_negative_margins_to_zero() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["NEG-MARGIN"]);

    tui.add_child(Rc::new(EmptyContent));
    // Negative margins should be treated as 0.
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(12)),
            margin: Some(OverlayMargin::Sides(OverlayMarginSides {
                top: -5,
                left: -10,
                right: 0,
                bottom: 0,
            })),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    // Should be at row 0, col 0 (negative margins clamped to 0).
    assert!(
        viewport
            .first()
            .is_some_and(|line| line.starts_with("NEG-MARGIN")),
        "Expected NEG-MARGIN at start of row 0, got: {:?}",
        viewport.first()
    );
    tui_support::stop(&tui);
}

#[test]
fn respects_margin_as_number() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["MARGIN"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(10)),
            margin: Some(OverlayMargin::All(5)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    // Should be on row 5 (not 0) due to margin.
    assert!(
        !viewport.first().is_some_and(|line| line.contains("MARGIN")),
        "Should not be on row 0"
    );
    assert!(
        !viewport.get(4).is_some_and(|line| line.contains("MARGIN")),
        "Should not be on row 4"
    );
    assert!(
        viewport.get(5).is_some_and(|line| line.contains("MARGIN")),
        "Expected MARGIN on row 5, got: {:?}",
        viewport.get(5)
    );
    // Should start at col 5 (not 0).
    let col_index = viewport.get(5).map_or(-1, |line| {
        i64::try_from(line.find("MARGIN").unwrap_or(0)).unwrap_or(-1)
    });
    assert_eq!(col_index, 5, "Expected col 5, got {col_index}");
    tui_support::stop(&tui);
}

#[test]
fn respects_margin_object() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["MARGIN"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(10)),
            margin: Some(OverlayMargin::Sides(OverlayMarginSides {
                top: 2,
                left: 3,
                right: 0,
                bottom: 0,
            })),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport.get(2).is_some_and(|line| line.contains("MARGIN")),
        "Expected MARGIN on row 2, got: {:?}",
        viewport.get(2)
    );
    let col_index = viewport.get(2).map_or(-1, |line| {
        i64::try_from(line.find("MARGIN").unwrap_or(0)).unwrap_or(-1)
    });
    assert_eq!(col_index, 3, "Expected col 3, got {col_index}");
    tui_support::stop(&tui);
}

// === offset ===

#[test]
fn applies_offset_x_and_offset_y_from_anchor_position() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["OFFSET"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(10)),
            offset_x: Some(10),
            offset_y: Some(5),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport.get(5).is_some_and(|line| line.contains("OFFSET")),
        "Expected OFFSET on row 5, got: {:?}",
        viewport.get(5)
    );
    let col_index = viewport.get(5).map_or(-1, |line| {
        i64::try_from(line.find("OFFSET").unwrap_or(0)).unwrap_or(-1)
    });
    assert_eq!(col_index, 10, "Expected col 10, got {col_index}");
    tui_support::stop(&tui);
}

// === percentage positioning ===

#[test]
fn positions_with_row_percent_and_col_percent() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["PCT"]);

    tui.add_child(Rc::new(EmptyContent));
    // 50% should center both ways.
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            width: Some(SizeValue::Cells(10)),
            row: Some(SizeValue::Percent(50.0)),
            col: Some(SizeValue::Percent(50.0)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    // Find the row with PCT.
    let found_row = viewport
        .iter()
        .position(|line| line.contains("PCT"))
        .map_or(-1, |row| i64::try_from(row).unwrap_or(-1));
    // Should be roughly centered vertically (row ~11-12 for 24 row terminal).
    assert!(
        (10..=13).contains(&found_row),
        "Expected centered row, got {found_row}"
    );
    tui_support::stop(&tui);
}

#[test]
fn row_percent_zero_positions_at_top() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["TOP"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            width: Some(SizeValue::Cells(10)),
            row: Some(SizeValue::Percent(0.0)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport.first().is_some_and(|line| line.contains("TOP")),
        "Expected TOP on row 0, got: {:?}",
        viewport.first()
    );
    tui_support::stop(&tui);
}

#[test]
fn row_percent_hundred_positions_at_bottom() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["BOTTOM"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            width: Some(SizeValue::Cells(10)),
            row: Some(SizeValue::Percent(100.0)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport.get(23).is_some_and(|line| line.contains("BOTTOM")),
        "Expected BOTTOM on last row, got: {:?}",
        viewport.get(23)
    );
    tui_support::stop(&tui);
}

// === maxHeight ===

#[test]
fn truncates_overlay_to_max_height() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            max_height: Some(SizeValue::Cells(3)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    let content = viewport.join("\n");
    assert!(content.contains("Line 1"), "Should include Line 1");
    assert!(content.contains("Line 2"), "Should include Line 2");
    assert!(content.contains("Line 3"), "Should include Line 3");
    assert!(!content.contains("Line 4"), "Should NOT include Line 4");
    assert!(!content.contains("Line 5"), "Should NOT include Line 5");
    tui_support::stop(&tui);
}

#[test]
fn truncates_overlay_to_max_height_percent() {
    let terminal = VirtualTerminal::new(80, 10);
    let tui = tui_support::new_test_tui(terminal.clone());
    // 10 lines in a 10 row terminal with 50% maxHeight should show 5 lines.
    let overlay = StaticOverlay::new(vec![
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10",
    ]);

    tui.add_child(Rc::new(EmptyContent));
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            max_height: Some(SizeValue::Percent(50.0)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    let content = viewport.join("\n");
    assert!(content.contains("L1"), "Should include L1");
    assert!(content.contains("L5"), "Should include L5");
    assert!(!content.contains("L6"), "Should NOT include L6");
    tui_support::stop(&tui);
}

// === absolute positioning ===

#[test]
fn row_and_col_override_anchor() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());
    let overlay = StaticOverlay::new(vec!["ABSOLUTE"]);

    tui.add_child(Rc::new(EmptyContent));
    // Even with bottom-right anchor, row/col should win.
    let _ = tui.show_overlay(
        overlay,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::BottomRight),
            row: Some(SizeValue::Cells(3)),
            col: Some(SizeValue::Cells(5)),
            width: Some(SizeValue::Cells(10)),
            ..OverlayOptions::default()
        }),
    );
    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport
            .get(3)
            .is_some_and(|line| line.contains("ABSOLUTE")),
        "Expected ABSOLUTE on row 3, got: {:?}",
        viewport.get(3)
    );
    let col_index = viewport.get(3).map_or(-1, |line| {
        i64::try_from(line.find("ABSOLUTE").unwrap_or(0)).unwrap_or(-1)
    });
    assert_eq!(col_index, 5, "Expected col 5, got {col_index}");
    tui_support::stop(&tui);
}

// === stacked overlays ===

#[test]
fn renders_multiple_overlays_with_later_ones_on_top() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());

    tui.add_child(Rc::new(EmptyContent));

    // First overlay at top-left.
    let overlay1 = StaticOverlay::new(vec!["FIRST-OVERLAY"]);
    let _ = tui.show_overlay(
        overlay1,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(20)),
            ..OverlayOptions::default()
        }),
    );

    // Second overlay at top-left (should cover part of first).
    let overlay2 = StaticOverlay::new(vec!["SECOND"]);
    let _ = tui.show_overlay(
        overlay2,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(10)),
            ..OverlayOptions::default()
        }),
    );

    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    // Second overlay should be visible (on top).
    assert!(
        viewport.first().is_some_and(|line| line.contains("SECOND")),
        "Expected SECOND on row 0, got: {:?}",
        viewport.first()
    );
    // Part of first overlay might still be visible after SECOND:
    // FIRST-OVERLAY is 13 chars, SECOND is 6 chars, so "OVERLAY" might show.
    tui_support::stop(&tui);
}

#[test]
fn handles_overlays_at_different_positions_without_interference() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());

    tui.add_child(Rc::new(EmptyContent));

    // Overlay at top-left.
    let overlay1 = StaticOverlay::new(vec!["TOP-LEFT"]);
    let _ = tui.show_overlay(
        overlay1,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(15)),
            ..OverlayOptions::default()
        }),
    );

    // Overlay at bottom-right.
    let overlay2 = StaticOverlay::new(vec!["BTM-RIGHT"]);
    let _ = tui.show_overlay(
        overlay2,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::BottomRight),
            width: Some(SizeValue::Cells(15)),
            ..OverlayOptions::default()
        }),
    );

    tui.start();
    render_and_flush(&tui);

    let viewport = terminal.get_viewport();
    // Both should be visible.
    assert!(
        viewport
            .first()
            .is_some_and(|line| line.contains("TOP-LEFT")),
        "Expected TOP-LEFT on row 0, got: {:?}",
        viewport.first()
    );
    assert!(
        viewport
            .get(23)
            .is_some_and(|line| line.contains("BTM-RIGHT")),
        "Expected BTM-RIGHT on row 23, got: {:?}",
        viewport.get(23)
    );
    tui_support::stop(&tui);
}

#[test]
fn properly_hides_overlays_in_stack_order() {
    let terminal = VirtualTerminal::new(80, 24);
    let tui = tui_support::new_test_tui(terminal.clone());

    tui.add_child(Rc::new(EmptyContent));

    // Show two overlays.
    let overlay1 = StaticOverlay::new(vec!["FIRST"]);
    let _ = tui.show_overlay(
        overlay1,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(10)),
            ..OverlayOptions::default()
        }),
    );

    let overlay2 = StaticOverlay::new(vec!["SECOND"]);
    let _ = tui.show_overlay(
        overlay2,
        Some(OverlayOptions {
            anchor: Some(OverlayAnchor::TopLeft),
            width: Some(SizeValue::Cells(10)),
            ..OverlayOptions::default()
        }),
    );

    tui.start();
    render_and_flush(&tui);

    // Second should be visible.
    let viewport = terminal.get_viewport();
    assert!(
        viewport.first().is_some_and(|line| line.contains("SECOND")),
        "SECOND should be visible initially"
    );

    // Hide top overlay.
    tui.hide_overlay();
    render_and_flush(&tui);

    // First should now be visible.
    let viewport = terminal.get_viewport();
    assert!(
        viewport.first().is_some_and(|line| line.contains("FIRST")),
        "FIRST should be visible after hiding SECOND"
    );

    tui_support::stop(&tui);
}
