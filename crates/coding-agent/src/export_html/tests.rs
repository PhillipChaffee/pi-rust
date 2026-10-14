#![expect(
    clippy::expect_used,
    reason = "tests pin outcomes; an unexpected result panics the test by design"
)]

use super::*;

#[test]
fn js_replace_reproduces_the_five_substitution_families() {
    // `$$` collapses to a literal `$`, `$&` rewrites to the match, and the
    // context escapes substitute the surrounding text.
    assert_eq!(js_replace("aXb", "X", "$$"), "a$b");
    assert_eq!(js_replace("aXb", "X", "[$&]"), "a[X]b");
    assert_eq!(
        js_replace("aXb", "X", "[$`]"),
        "a[a]b",
        "the backtick escape substitutes the prefix"
    );
    assert_eq!(
        js_replace("aXb", "X", "[$']"),
        "a[b]b",
        "the quote escape substitutes the suffix"
    );
    assert_eq!(
        js_replace("aXb", "X", "$1$<name>$n"),
        "a$1$<name>$nb",
        "numbered and named groups stay literal for a string pattern"
    );
    assert_eq!(
        js_replace("aXb", "X", "$"),
        "a$b",
        "a trailing dollar rides as text"
    );
    assert_eq!(
        js_replace("abc", "X", "[hit]"),
        "abc",
        "an absent match leaves the haystack alone"
    );
    assert_eq!(
        js_replace("aXYb", "XY", "z"),
        "azb",
        "only the first occurrence replaces"
    );
}

#[test]
fn parse_color_reads_hex_and_rgb_and_rejects_the_rest() {
    assert_eq!(parse_color("#343541"), Some((0x34, 0x35, 0x41)));
    assert_eq!(parse_color("rgb(12, 34, 56)"), Some((12, 34, 56)));
    assert_eq!(parse_color("rgb(12,34,56)"), Some((12, 34, 56)));
    assert_eq!(
        parse_color("rgb (12, 34, 56)"),
        Some((12, 34, 56)),
        "upstream's regex allows the space after rgb"
    );
    assert_eq!(
        parse_color("#343541extra"),
        None,
        "upstream's hex regex is a full match"
    );
    assert_eq!(
        parse_color("rgb(300, 0, 0)"),
        None,
        "the out-of-range degradation"
    );
    assert_eq!(parse_color("#34354"), None, "a short hex rejects");
    assert_eq!(parse_color("#34354G"), None, "a non-hex digit rejects");
    assert_eq!(parse_color("rgb(1,2)"), None, "two channels reject");
    assert_eq!(
        parse_color("rgb(1,2,300)"),
        None,
        "an out-of-range channel rejects"
    );
    assert_eq!(
        parse_color("rgb(1,2,x)"),
        None,
        "a non-numeric channel rejects"
    );
    assert_eq!(parse_color("hsl(1,2,3)"), None, "another function rejects");
    assert_eq!(parse_color(""), None);
}

#[test]
fn adjust_brightness_clamps_and_passes_unparseable_colors_through() {
    assert_eq!(
        adjust_brightness("#ffffff", 2.0),
        "rgb(255, 255, 255)",
        "the ceiling clamps"
    );
    assert_eq!(
        adjust_brightness("#000000", 0.0),
        "rgb(0, 0, 0)",
        "the floor clamps"
    );
    assert_eq!(
        adjust_brightness("not-a-color", 2.0),
        "not-a-color",
        "an unparseable color rides through"
    );
}

#[test]
fn the_serde_failure_converts_into_the_io_arm() {
    let error: ExportError =
        serde_json::from_str::<crate::session_manager::SessionHeader>("not json")
            .expect_err("not json")
            .into();
    assert!(
        error.to_string().contains("line 1"),
        "the serde message rides through: {error}"
    );
}

#[test]
fn the_export_error_conversions_reprint_their_sources() {
    let io: ExportError = std::io::Error::other("boom").into();
    assert_eq!(io.to_string(), "boom");
    let session: ExportError = SessionManagerError::Io("gone".to_owned()).into();
    assert_eq!(
        session.to_string(),
        "gone",
        "the session arm reprints the manager error"
    );
}
