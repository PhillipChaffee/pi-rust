//! 1:1 port of `packages/tui/test/truncated-text.test.ts` (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).
//!
//! Upstream's `chalk` styles become literal SGR escapes so the ANSI
//! assertions stay deterministic.

use pi_tui::components::TruncatedText;
use pi_tui::tui::Component;
use pi_tui::utils::{strip_terminal_sequences, visible_width};

mod truncated_text_component {
    use super::*;

    #[test]
    fn pads_output_lines_to_exactly_match_width() {
        let text = TruncatedText::with_padding("Hello world", 1, 0);
        let lines = text.render(50);

        // Should have exactly one content line (no vertical padding)
        assert_eq!(lines.len(), 1);

        // Line should be exactly 50 visible characters
        let visible_len = visible_width(&lines[0]);
        assert_eq!(visible_len, 50);
    }

    #[test]
    fn pads_output_with_vertical_padding_lines_to_width() {
        let text = TruncatedText::with_padding("Hello", 0, 2);
        let lines = text.render(40);

        // Should have 2 padding lines + 1 content line + 2 padding lines = 5 total
        assert_eq!(lines.len(), 5);

        // All lines should be exactly 40 characters
        for line in &lines {
            assert_eq!(visible_width(line), 40);
        }
    }

    #[test]
    fn truncates_long_text_and_pads_to_width() {
        let long_text =
            "This is a very long piece of text that will definitely exceed the available width";
        let text = TruncatedText::with_padding(long_text, 1, 0);
        let lines = text.render(30);

        assert_eq!(lines.len(), 1);

        // Should be exactly 30 characters
        assert_eq!(visible_width(&lines[0]), 30);

        // Should contain ellipsis
        let stripped = strip_terminal_sequences(&lines[0]);
        assert!(stripped.contains("..."));
    }

    #[test]
    fn preserves_ansi_codes_in_output_and_pads_correctly() {
        let styled_text = "\x1b[31mHello\x1b[0m \x1b[34mworld\x1b[0m";
        let text = TruncatedText::with_padding(styled_text, 1, 0);
        let lines = text.render(40);

        assert_eq!(lines.len(), 1);

        // Should be exactly 40 visible characters (ANSI codes don't count)
        assert_eq!(visible_width(&lines[0]), 40);

        // Should preserve the color codes
        assert!(lines[0].contains("\x1b["));
    }

    #[test]
    fn truncates_styled_text_and_adds_reset_code_before_ellipsis() {
        let long_styled_text = "\x1b[31mThis is a very long red text that will be truncated\x1b[0m";
        let text = TruncatedText::with_padding(long_styled_text, 1, 0);
        let lines = text.render(20);

        assert_eq!(lines.len(), 1);

        // Should be exactly 20 visible characters
        assert_eq!(visible_width(&lines[0]), 20);

        // Should contain reset code before ellipsis
        assert!(lines[0].contains("\x1b[0m..."));
    }

    #[test]
    fn handles_text_that_fits_exactly() {
        // With paddingX=1, available width is 30-2=28
        // "Hello world" is 11 chars, fits comfortably
        let text = TruncatedText::with_padding("Hello world", 1, 0);
        let lines = text.render(30);

        assert_eq!(lines.len(), 1);
        assert_eq!(visible_width(&lines[0]), 30);

        // Should NOT contain ellipsis
        let stripped = strip_terminal_sequences(&lines[0]);
        assert!(!stripped.contains("..."));
    }

    #[test]
    fn handles_empty_text() {
        let text = TruncatedText::with_padding("", 1, 0);
        let lines = text.render(30);

        assert_eq!(lines.len(), 1);
        assert_eq!(visible_width(&lines[0]), 30);
    }

    #[test]
    fn stops_at_newline_and_only_shows_first_line() {
        let multiline_text = "First line\nSecond line\nThird line";
        let text = TruncatedText::with_padding(multiline_text, 1, 0);
        let lines = text.render(40);

        assert_eq!(lines.len(), 1);
        assert_eq!(visible_width(&lines[0]), 40);

        // Should only contain "First line"
        let stripped = strip_terminal_sequences(&lines[0]);
        assert!(stripped.trim().contains("First line"));
        assert!(!stripped.contains("Second line"));
        assert!(!stripped.contains("Third line"));
    }

    #[test]
    fn truncates_first_line_even_with_newlines_in_text() {
        let long_multiline_text =
            "This is a very long first line that needs truncation\nSecond line";
        let text = TruncatedText::with_padding(long_multiline_text, 1, 0);
        let lines = text.render(25);

        assert_eq!(lines.len(), 1);
        assert_eq!(visible_width(&lines[0]), 25);

        // Should contain ellipsis and not second line
        let stripped = strip_terminal_sequences(&lines[0]);
        assert!(stripped.contains("..."));
        assert!(!stripped.contains("Second line"));
    }
}
