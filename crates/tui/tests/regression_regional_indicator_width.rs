//! 1:1 port of `packages/tui/test/regression-regional-indicator-width.test.ts`
//! (upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).

use pi_tui::utils::{visible_width, wrap_text_with_ansi};

#[test]
fn treats_partial_flag_grapheme_as_full_width_to_avoid_streaming_render_drift() {
    // During streaming, "🇨🇳" often appears as an intermediate "🇨" first.
    // If "🇨" is measured as width 1 while the terminal renders it as width 2,
    // differential rendering can drift and leave stale characters on screen.
    let partial_flag = "🇨";
    let list_line = "      - 🇨";

    assert_eq!(visible_width(partial_flag), 2);
    assert_eq!(visible_width(list_line), 10);
}

#[test]
fn wraps_intermediate_partial_flag_list_line_before_overflow() {
    // Width 9 cannot fit "      - 🇨" if 🇨 is width 2 (8 + 2 = 10).
    let wrapped = wrap_text_with_ansi("      - 🇨", 9);

    assert_eq!(wrapped.len(), 2);
    assert_eq!(visible_width(wrapped.first().unwrap_or(&String::new())), 7);
    assert_eq!(visible_width(wrapped.last().unwrap_or(&String::new())), 2);
}

#[test]
fn treats_all_regional_indicator_singleton_graphemes_as_width_2() {
    for cp in 0x1_f1e6_u32..=0x1_f1ff {
        let regional_indicator = char::from_u32(cp).unwrap_or('\0');
        assert_eq!(
            visible_width(&regional_indicator.to_string()),
            2,
            "Expected {regional_indicator} (U+{cp:X}) to be width 2"
        );
    }
}

#[test]
fn keeps_full_flag_pairs_at_width_2() {
    let samples = ["🇯🇵", "🇺🇸", "🇬🇧", "🇨🇳", "🇩🇪", "🇫🇷"];
    for flag in samples {
        assert_eq!(visible_width(flag), 2, "Expected {flag} to be width 2");
    }
}

#[test]
fn keeps_common_streaming_emoji_intermediates_at_stable_width() {
    let samples = ["👍", "👍🏻", "✅", "⚡", "⚡️", "👨", "👨‍💻", "🏳️‍🌈"];
    for sample in samples {
        assert_eq!(visible_width(sample), 2, "Expected {sample} to be width 2");
    }
}
