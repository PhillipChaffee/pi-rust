//! 1:1 port of `packages/tui/test/truncate-to-width.test.ts` (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).

use pi_tui::utils::{normalize_terminal_output, truncate_to_width, visible_width};

mod truncate_to_width {
    use super::*;

    #[test]
    fn keeps_output_within_width_for_very_large_unicode_input() {
        let text = "🙂界".repeat(100_000);
        let truncated = truncate_to_width(&text, 40, "…", false);

        assert!(visible_width(&truncated) <= 40);
        assert!(truncated.ends_with("…\x1b[0m"));
    }

    #[test]
    fn preserves_ansi_styling_for_kept_text_and_resets_before_and_after_ellipsis() {
        let text = format!("\x1b[31m{}\x1b[0m", "hello ".repeat(1000));
        let truncated = truncate_to_width(&text, 20, "…", false);

        assert!(visible_width(&truncated) <= 20);
        assert!(truncated.contains("\x1b[31m"));
        assert!(truncated.ends_with("\x1b[0m…\x1b[0m"));
    }

    #[test]
    fn closes_a_bel_terminated_osc8_link_when_truncating_its_label() {
        let open = "\x1b]8;;https://example.com\x07";
        let close = "\x1b]8;;\x07";
        let text = format!("{open}some-longer-label-here{close}");

        assert_eq!(
            truncate_to_width(&text, 15, "...", false),
            format!("{open}some-longer-{close}\x1b[0m...\x1b[0m")
        );
    }

    #[test]
    fn handles_malformed_ansi_escape_prefixes_without_hanging() {
        let text = format!("abc\x1bnot-ansi {}", "🙂".repeat(1000));
        let truncated = truncate_to_width(&text, 20, "…", false);

        assert!(visible_width(&truncated) <= 20);
    }

    #[test]
    fn clips_wide_ellipsis_safely_and_brackets_it_with_resets() {
        assert_eq!(truncate_to_width("abcdef", 1, "🙂", false), "");
        assert_eq!(
            truncate_to_width("abcdef", 2, "🙂", false),
            "\x1b[0m🙂\x1b[0m"
        );
        assert!(visible_width(&truncate_to_width("abcdef", 2, "🙂", false)) <= 2);
    }

    #[test]
    fn returns_the_original_text_when_it_already_fits_even_if_ellipsis_is_too_wide() {
        assert_eq!(truncate_to_width("a", 2, "🙂", false), "a");
        assert_eq!(truncate_to_width("界", 2, "🙂", false), "界");
    }

    #[test]
    fn pads_truncated_output_to_requested_width() {
        let truncated = truncate_to_width("🙂界🙂界🙂界", 8, "…", true);
        assert_eq!(visible_width(&truncated), 8);
    }

    #[test]
    fn adds_a_trailing_reset_when_truncating_without_an_ellipsis() {
        let truncated =
            truncate_to_width(&format!("\x1b[31m{}", "hello".repeat(100)), 10, "", false);
        assert!(visible_width(&truncated) <= 10);
        assert!(truncated.ends_with("\x1b[0m"));
    }

    #[test]
    fn keeps_a_contiguous_prefix_instead_of_skipping_a_wide_grapheme_and_resuming_later() {
        let truncated = truncate_to_width("🙂\t界 \x1b_abc\x07", 7, "…", true);
        assert_eq!(truncated, "🙂\t\x1b[0m…\x1b[0m ");
    }
}

mod visible_width {
    use super::*;

    #[test]
    fn counts_tabs_inline_and_skips_ansi_inline() {
        assert_eq!(visible_width("\t\x1b[31m界\x1b[0m"), 5);
    }

    #[test]
    fn counts_indic_conjunct_spacing_code_points_within_grapheme_clusters() {
        assert_eq!(visible_width("र्क"), 2);
        assert_eq!(visible_width("नेटवर्क"), 5);
        assert_eq!(visible_width("सर्वाधिकार सुरक्षित। ऑर्डर पर क्लिक करें"), 33);
        assert_eq!(visible_width("র্ক"), 2);
        assert_eq!(visible_width("ર્ક"), 2);
        assert_eq!(visible_width("ର୍କ"), 2);
        assert_eq!(visible_width("ర్క"), 2);
        assert_eq!(visible_width("ര്‍ക"), 2);
    }

    #[test]
    fn keeps_ordinary_combining_marks_zero_width() {
        assert_eq!(visible_width("e\u{0301}"), 1);
        assert_eq!(visible_width("čřžůú"), 5);
        assert_eq!(visible_width("שָׁ"), 1);
        assert_eq!(visible_width("بّ"), 1);
        assert_eq!(visible_width("རྐ"), 1);
        assert_eq!(visible_width("ᜠ᜴"), 1);
        assert_eq!(visible_width("가〮"), 2);
        assert_eq!(visible_width("가〯"), 2);
    }

    #[test]
    fn keeps_cjk_and_japanese_width_accounting_unchanged() {
        assert_eq!(visible_width("网络"), 4);
        assert_eq!(visible_width("ネットワーク"), 12);
        assert_eq!(visible_width("が"), 2);
        assert_eq!(visible_width("か\u{3099}"), 2);
    }

    #[test]
    fn counts_myanmar_marks_that_terminals_allocate_cells_for() {
        assert_eq!(visible_width("ကာ"), 2);
        assert_eq!(visible_width("ကေ"), 2);
        assert_eq!(visible_width("က်"), 2);
        assert_eq!(visible_width("ကျ"), 2);
        assert_eq!(visible_width("ကြ"), 2);
        assert_eq!(visible_width("ကဳ"), 2);
        assert_eq!(visible_width("ကဴ"), 2);
        assert_eq!(visible_width("ကဵ"), 2);
        assert_eq!(visible_width("ကး"), 2);
        assert_eq!(visible_width("ကို"), 1);
        assert_eq!(visible_width("က္"), 1);
    }

    #[test]
    fn keeps_thai_and_lao_am_clusters_at_their_normal_cell_width() {
        assert_eq!(visible_width("ำ"), 1);
        assert_eq!(visible_width("ຳ"), 1);
        assert_eq!(visible_width("กำ"), 2);
        assert_eq!(visible_width("ກຳ"), 2);
    }

    #[test]
    fn normalizes_thai_and_lao_am_vowels_only_for_terminal_output() {
        assert_eq!(normalize_terminal_output("ำ"), "ํา");
        assert_eq!(normalize_terminal_output("ຳ"), "ໍາ");
        assert_eq!(
            visible_width(&normalize_terminal_output("ำabc")),
            visible_width("ำabc")
        );
        assert_eq!(
            visible_width(&normalize_terminal_output("ຳabc")),
            visible_width("ຳabc")
        );
    }
}
