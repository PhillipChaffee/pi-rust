//! Boundary tests for the ANSI string-width utilities: the public surface the
//! upstream golden files exercise only indirectly (or exercise later, when
//! their consumer components land) gets its own coverage here so the crate
//! stands on the coverage gate from the first slice.
#![expect(
    clippy::expect_used,
    reason = "a missing fixture range is the test environment failing; expecting keeps the ported assertions readable"
)]

use pi_tui::utils::{
    apply_background_to_line, extract_segments, get_active_background_ansi,
    get_grapheme_cell_range, get_osc8_link_at_column, grapheme_segments, normalize_terminal_output,
    slice_by_column, strip_terminal_sequences, truncate_to_width, visible_width, word_segments,
    wrap_text_with_ansi,
};

#[test]
fn strip_terminal_sequences_preserves_visible_text() {
    assert_eq!(strip_terminal_sequences("plain"), "plain");
    assert_eq!(
        strip_terminal_sequences(
            "\x1b[31mred\x1b[0m\x1b]8;;https://x.test\x07link\x1b]8;;\x07\x1b_apc\x1b\\tail"
        ),
        "redlinktail"
    );
}

#[test]
fn grapheme_cell_ranges_cover_wide_clusters_and_skip_ansi() {
    let line = "\x1b[31ma界b\x1b[0m";
    let wide = get_grapheme_cell_range(line, 1).expect("range at col 1");
    assert_eq!((wide.start, wide.end), (1, 3));
    let inside = get_grapheme_cell_range(line, 2).expect("range at col 2");
    assert_eq!((inside.start, inside.end), (1, 3));
    let b = get_grapheme_cell_range(line, 3).expect("range at col 3");
    assert_eq!((b.start, b.end), (3, 4));
    assert!(get_grapheme_cell_range(line, 9).is_none());
}

#[test]
fn grapheme_cell_range_ignores_zero_width_clusters() {
    // ZWJ-only input has no visible cells, so no cluster covers any column.
    assert!(get_grapheme_cell_range("\u{200d}", 0).is_none());
}

#[test]
fn osc8_link_at_column_resolves_only_inside_links() {
    let url = "https://example.com";
    let line = format!("\x1b]8;;{url}\x1b\\abc\x1b]8;;\x1b\\ def");
    assert_eq!(get_osc8_link_at_column(&line, 0), Some(url));
    assert_eq!(get_osc8_link_at_column(&line, 2), Some(url));
    // The close ends the link: the space at col 3 is outside it.
    assert_eq!(get_osc8_link_at_column(&line, 3), None);
    assert_eq!(get_osc8_link_at_column(&line, 5), None);
    assert_eq!(get_osc8_link_at_column(&line, 50), None);

    // A BEL-terminated open keeps resolving too.
    let bel = format!("\x1b]8;;{url}\x07ab\x1b]8;;\x07");
    assert_eq!(get_osc8_link_at_column(&bel, 1), Some(url));

    // An empty-URL open is a close, never a link.
    let empty = "\x1b]8;;\x1b\\ab";
    assert_eq!(get_osc8_link_at_column(empty, 0), None);

    // A tab spans three cells.
    let with_tab = format!("\x1b]8;;{url}\x07a\tb\x1b]8;;\x07");
    assert_eq!(get_osc8_link_at_column(&with_tab, 2), Some(url));
}

#[test]
fn active_background_ansi_reports_the_last_background() {
    assert_eq!(get_active_background_ansi("\x1b[44mtext"), "\x1b[44m");
    assert_eq!(get_active_background_ansi("\x1b[41;4mtext\x1b[49m"), "");
    assert_eq!(
        get_active_background_ansi("\x1b[48;2;1;2;3mx"),
        "\x1b[48;2;1;2;3m"
    );
    assert_eq!(get_active_background_ansi("\x1b[48;5;9mx"), "\x1b[48;5;9m");
    assert_eq!(get_active_background_ansi("\x1b[41m\x1b[42mx"), "\x1b[42m");
    assert_eq!(get_active_background_ansi("plain"), "");
    assert_eq!(get_active_background_ansi("\x1b[0mx"), "");
}

#[test]
fn apply_background_to_line_pads_to_full_width() {
    let padded = apply_background_to_line("ab", 6, |text| format!("\x1b[44m{text}\x1b[0m"));
    assert_eq!(padded, "\x1b[44mab    \x1b[0m");
    let exact = apply_background_to_line("abcd", 4, |text| format!("<{text}>"));
    assert_eq!(exact, "<abcd>");
}

#[test]
fn slice_by_column_strict_drops_boundary_wide_graphemes() {
    let line = "a界b";
    assert_eq!(slice_by_column(line, 1, 1, true), "");
    assert_eq!(slice_by_column(line, 1, 1, false), "界");
    assert_eq!(slice_by_column(line, 1, 2, true), "界");
    assert_eq!(slice_by_column(line, 0, 10, true), "a界b");
    assert_eq!(slice_by_column(line, 0, 0, false), "");
}

#[test]
fn extract_segments_after_region_uses_style_state_at_boundary() {
    // The reset before the after-region is tracked but not re-emitted, so the
    // after segment carries no stale styling.
    let line = "\x1b[31mABC\x1b[0mDEF";
    let segments = extract_segments(line, 3, 3, 3, false);
    assert_eq!(segments.before, "\x1b[31mABC");
    assert_eq!(segments.before_width, 3);
    assert_eq!(segments.after, "DEF");
    assert_eq!(segments.after_width, 3);

    // Styling active before the boundary re-opens on the after segment.
    let styled = "\x1b[4mab\x1b[0mcd";
    let segments = extract_segments(styled, 2, 2, 2, false);
    assert_eq!(segments.before, "\x1b[4mab");
    assert_eq!(segments.after_width, 2);
}

#[test]
fn extract_segments_strict_after_excludes_boundary_wide_graphemes() {
    let line = "a界b";
    let segments = extract_segments(line, 1, 1, 1, true);
    assert_eq!(segments.before, "a");
    assert_eq!(segments.after, "");
    assert_eq!(segments.after_width, 0);

    let loose = extract_segments(line, 1, 1, 1, false);
    assert_eq!(loose.after, "界");
    assert_eq!(loose.after_width, 2);
}

#[test]
fn extract_segments_with_zero_after_length_stops_at_before_end() {
    let line = "abcdef";
    let segments = extract_segments(line, 3, 9, 0, false);
    assert_eq!(segments.before, "abc");
    assert_eq!(segments.after, "");
}

#[test]
fn character_classification_matches_the_wrapping_sets() {
    assert!(pi_tui::utils::is_whitespace_char(' '));
    assert!(pi_tui::utils::is_whitespace_char('\t'));
    assert!(!pi_tui::utils::is_whitespace_char('a'));

    for ch in "()[]{}<>,.;:'\"!?+-=*/\\|&%^$#@~`".chars() {
        assert!(
            pi_tui::utils::is_punctuation_char(ch),
            "{ch} should be punctuation"
        );
    }
    assert!(!pi_tui::utils::is_punctuation_char('a'));
}

#[test]
fn segment_accessors_mirror_the_upstream_iterators() {
    assert_eq!(
        grapheme_segments("a🇨🇳b").collect::<Vec<_>>(),
        vec!["a", "🇨🇳", "b"]
    );
    assert_eq!(
        word_segments("hello world").collect::<Vec<_>>(),
        vec!["hello", " ", "world"]
    );
    assert_eq!(word_segments("don't").collect::<Vec<_>>(), vec!["don't"]);
}

#[test]
fn width_cache_stays_consistent_across_eviction() {
    // Push past the cache bound; every width must stay correct afterwards.
    for i in 0..1200 {
        let text = format!("界{i}");
        assert_eq!(visible_width(&text), 2 + i.to_string().len());
    }
    assert_eq!(visible_width("网络"), 4);
}

#[test]
fn truncate_to_width_edges() {
    assert_eq!(truncate_to_width("", 5, "...", true), "     ");
    assert_eq!(truncate_to_width("hello", 0, "...", false), "");
    // An empty ellipsis still brackets the kept prefix with a reset.
    assert_eq!(
        truncate_to_width("hello world", 5, "", false),
        "hello\x1b[0m"
    );
    // A text that fits is returned unchanged even with an empty ellipsis.
    assert_eq!(truncate_to_width("hi", 5, "", true), "hi   ");
}

#[test]
fn wrap_text_with_ansi_edges() {
    assert_eq!(wrap_text_with_ansi("", 10), vec![""]);
    assert_eq!(wrap_text_with_ansi("\n", 10), vec!["", ""]);
    // An empty active-code prefix starts the first broken line, matching
    // upstream's breakLongWord shape.
    assert_eq!(wrap_text_with_ansi("ab", 0), vec!["", "a", "b"]);
}

#[test]
fn extract_ansi_code_unit_cases() {
    use pi_tui::utils::extract_ansi_code;

    assert!(extract_ansi_code("", 0).is_none());
    assert!(extract_ansi_code("abc", 1).is_none());
    assert!(extract_ansi_code("\x1b", 0).is_none());
    assert!(
        extract_ansi_code("\x1b[31", 0).is_none(),
        "unterminated CSI"
    );
    assert!(
        extract_ansi_code("\x1b]8;;url", 0).is_none(),
        "unterminated OSC"
    );
    assert!(extract_ansi_code("\x1b_x", 0).is_none(), "unterminated APC");
    assert!(
        extract_ansi_code("\x1bx", 0).is_none(),
        "not a sequence family"
    );

    let code = extract_ansi_code("a\x1b[1;2G", 1).expect("CSI with G terminator");
    assert_eq!(code.code, "\x1b[1;2G");
    assert_eq!(code.length, 6);

    let osc = extract_ansi_code("\x1b]8;;u\x1b\\!", 0).expect("ST-terminated OSC");
    assert_eq!(osc.code, "\x1b]8;;u\x1b\\");
    assert_eq!(osc.length, 8);

    let apc = extract_ansi_code("\x1b_marker\x07!", 0).expect("BEL-terminated APC");
    assert_eq!(apc.code, "\x1b_marker\x07");
    assert_eq!(apc.length, 9);
}

#[test]
fn normalize_terminal_output_without_tabs_still_decomposes_am() {
    assert_eq!(normalize_terminal_output("กำ"), "ก\u{0e4d}\u{0e32}");
    assert_eq!(normalize_terminal_output("\x1b[31mຳ"), "\x1b[31mໍາ");
}

#[test]
fn wrap_breaks_long_tokens_with_active_styles() {
    // A long unbreakable token with styles still splits across lines.
    let wrapped = wrap_text_with_ansi("\x1b[4mabcdefgh\x1b[0m", 3);
    assert_eq!(wrapped.len(), 3);
    assert_eq!(visible_width(wrapped.last().unwrap_or(&String::new())), 2);
}

#[test]
fn visible_width_of_punctuated_ascii_is_its_length() {
    assert_eq!(visible_width("a,b.c"), 5);
}

#[test]
fn wrap_carries_all_sgr_attributes_onto_continuation_lines() {
    let wrapped = wrap_text_with_ansi("\x1b[1;2;3;5;7;8;9mabcd efgh", 4);
    assert_eq!(wrapped[1], "\x1b[1;2;3;5;7;8;9mefgh");
}

#[test]
fn wrap_reemits_color_attributes_with_their_full_parameter_lists() {
    // 38;5;240 sets the foreground, 48;2 sets an RGB background, then 90
    // replaces the foreground and 100 the background.
    let wrapped = wrap_text_with_ansi("\x1b[38;5;240;48;2;1;2;3;90;100mabcd ef", 6);
    assert_eq!(wrapped[1], "\x1b[90;100mef");
}

#[test]
fn active_background_tracks_every_sgr_reset_family() {
    assert_eq!(
        get_active_background_ansi("\x1b[1;2;3;4;5;7;8;9;21;22;23;24;25;27;28;29mx"),
        ""
    );
    assert_eq!(get_active_background_ansi("\x1b[41m\x1b[49mx"), "");
    assert_eq!(get_active_background_ansi("\x1b[31m\x1b[39mx"), "");
    // A bright foreground replaces an earlier one; backgrounds stay.
    assert_eq!(get_active_background_ansi("\x1b[31m\x1b[90mx"), "");
    assert_eq!(
        get_active_background_ansi("\x1b[41m\x1b[100mx"),
        "\x1b[100m"
    );
    // An empty SGR parameter is skipped like upstream's NaN parseInt result.
    assert_eq!(get_active_background_ansi("\x1b[;41mx"), "\x1b[41m");
    // Non-SGR sequences pass through the tracker without state changes.
    assert_eq!(get_active_background_ansi("\x1b[K"), "");
    assert_eq!(get_active_background_ansi("\x1b]8;;\x1b\\"), "");
    // An OSC 8 with no parameter field is not a hyperlink at all.
    assert_eq!(get_active_background_ansi("\x1b]8;abc\x07"), "");
}

#[test]
fn truncate_reopens_and_closes_links_around_the_reset() {
    // The close inside the kept prefix clears the active link, so the final
    // reset carries no hyperlink re-open.
    assert_eq!(
        truncate_to_width("\x1b]8;;u\x07ab\x1b]8;;\x07cdef", 5, "", false),
        "\x1b]8;;u\x07ab\x1b]8;;\x07cde\x1b[0m"
    );
    // A plain SGR inside the prefix is not a hyperlink and leaves no close.
    assert_eq!(
        truncate_to_width("\x1b]8;;u\x07ab\x1b[31mcdefgh", 5, "", false),
        "\x1b]8;;u\x07ab\x1b[31mcde\x1b]8;;\x07\x1b[0m"
    );
}

#[test]
fn format_only_graphemes_measure_zero_width() {
    // U+0600 is a Format code point that is not default-ignorable: it is
    // stripped as a leading non-printing run, leaving no base to measure.
    assert_eq!(visible_width("\u{0600}"), 0);
}

#[test]
fn truncate_flushes_pending_ansi_before_tabs_and_clears_it_past_the_prefix() {
    assert_eq!(
        truncate_to_width("\x1b[31mab\x1b[4m\tcd\x1b[0m", 6, "…", false),
        "\x1b[31mab\x1b[4m\t\x1b[0m…\x1b[0m"
    );
}

#[test]
fn truncate_returns_the_unchanged_text_with_padding_when_it_fits() {
    assert_eq!(
        truncate_to_width("\x1b[31mab\x1b[0m", 10, "…", true),
        "\x1b[31mab\x1b[0m        "
    );
}

#[test]
fn truncate_clips_wide_ellipses_of_every_shape() {
    // ASCII ellipsis clips to the target width.
    assert_eq!(
        truncate_to_width("🙂界🙂界🙂界", 2, "abcd", false),
        "\x1b[0mab\x1b[0m"
    );
    // A multi-cluster wide ellipsis keeps its first fitting cluster.
    assert_eq!(
        truncate_to_width("🙂🙂🙂🙂", 2, "🙂🙂", false),
        "\x1b[0m🙂\x1b[0m"
    );
    // An ANSI+tab ellipsis carries its styling and drops the tab that would
    // overflow.
    assert_eq!(
        truncate_to_width("🙂🙂🙂🙂", 4, "\x1b[31mab\tcd\x1b[0m", false),
        "\x1b[0m\x1b[31mab\x1b[0m"
    );
    // A wide grapheme inside the ellipsis still fits at the exact boundary.
    assert_eq!(
        truncate_to_width("🙂🙂🙂🙂", 3, "\x1b[31ma界cd\x1b[0m", false),
        "\x1b[0m\x1b[31ma界\x1b[0m"
    );
    // Past the boundary the clip stops before the wide grapheme.
    assert_eq!(
        truncate_to_width("🙂🙂🙂🙂", 2, "\x1b[31ma界cd\x1b[0m", false),
        "\x1b[0m\x1b[31ma\x1b[0m"
    );
    // An ellipsis ending inside the budget clips to its full width; the kept
    // prefix still carries its first wide grapheme.
    assert_eq!(
        truncate_to_width("🙂🙂🙂🙂", 5, "\x1b[31ma b", false),
        "🙂\x1b[0m\x1b[31ma b\x1b[0m"
    );
}

#[test]
fn slice_with_width_carries_ansi_codes_into_the_range() {
    // Codes before the range attach to the first kept grapheme.
    assert_eq!(
        slice_by_column("\x1b[31mabc\x1b[0mdef", 1, 2, false),
        "\x1b[31mbc"
    );
    // Codes at the range start column are kept; a code seen before the range
    // re-attaches to the first kept grapheme.
    assert_eq!(
        slice_by_column("\x1b[31mabc\x1b[0mdef", 3, 3, false),
        "\x1b[0m\x1b[31mdef"
    );
}

#[test]
fn extract_segments_carries_ansi_codes_inside_the_after_region() {
    let segments = extract_segments("\x1b[31ma\x1b[4mb\x1b[0mc", 1, 1, 2, false);
    assert_eq!(segments.before, "\x1b[31ma");
    assert_eq!(segments.after, "\x1b[4;31mb\x1b[0mc");
    assert_eq!(segments.after_width, 2);
}
