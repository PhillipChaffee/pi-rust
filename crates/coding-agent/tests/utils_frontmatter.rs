//! The frontmatter suite, upstream's `test/frontmatter.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The invalid-YAML case asserts the failure family the port reports —
//! upstream asserts the npm `yaml` package's `at line 1, column 10`
//! message, the port surfaces yaml-rust2's scanner message for the same
//! malformed input (recorded with the ticket).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_coding_agent::utils::frontmatter::{parse_frontmatter, strip_frontmatter};
use pi_coding_agent::utils::text::split_bom;
use yaml_rust2::Yaml;

fn yaml_str<'a>(frontmatter: &'a Yaml, key: &str) -> Option<&'a str> {
    frontmatter
        .as_hash()?
        .get(&Yaml::String(key.to_string()))
        .and_then(Yaml::as_str)
}

// === parseFrontmatter =======================================================

#[test]
fn parses_keys_strips_quotes_and_returns_body() {
    let input =
        "---\nname: \"skill-name\"\ndescription: 'A desc'\nfoo-bar: value\n---\n\nBody text";
    let parsed = parse_frontmatter(input).expect("parse");
    assert_eq!(yaml_str(&parsed.frontmatter, "name"), Some("skill-name"));
    assert_eq!(yaml_str(&parsed.frontmatter, "description"), Some("A desc"));
    assert_eq!(yaml_str(&parsed.frontmatter, "foo-bar"), Some("value"));
    assert_eq!(parsed.body, "Body text");
}

#[test]
fn normalizes_newlines_and_handles_crlf() {
    let input = "---\r\nname: test\r\n---\r\nLine one\r\nLine two";
    let parsed = parse_frontmatter(input).expect("parse");
    assert_eq!(parsed.body, "Line one\nLine two");
}

#[test]
fn fails_on_invalid_yaml_frontmatter() {
    let input = "---\nfoo: [bar\n---\nBody";
    let error = parse_frontmatter(input).expect_err("parse error");
    // The malformed-input family the `yaml` parse throws on: an unterminated
    // flow sequence, reported at its scanner position.
    assert!(
        error.0.contains("flow sequence") && error.0.contains("expected ',' or ']'"),
        "{error:?}"
    );
}

#[test]
fn parses_multiline_yaml_syntax() {
    let input = "---\ndescription: |\n  Line one\n  Line two\n---\n\nBody";
    let parsed = parse_frontmatter(input).expect("parse");
    assert_eq!(
        yaml_str(&parsed.frontmatter, "description"),
        Some("Line one\nLine two\n")
    );
    assert_eq!(parsed.body, "Body");
}

#[test]
fn returns_original_content_when_frontmatter_is_missing_or_unterminated() {
    let no_frontmatter = "Just text\nsecond line";
    let missing_end = "---\nname: test\nBody without terminator";
    let result_no_frontmatter = parse_frontmatter(no_frontmatter).expect("parse");
    let result_missing_end = parse_frontmatter(missing_end).expect("parse");
    assert_eq!(result_no_frontmatter.body, "Just text\nsecond line");
    assert_eq!(
        result_missing_end.body,
        "---\nname: test\nBody without terminator"
    );
}

#[test]
fn returns_empty_object_for_empty_or_comment_only_frontmatter() {
    let input = "---\n# just a comment\n---\nBody";
    let parsed = parse_frontmatter(input).expect("parse");
    assert_eq!(
        parsed.frontmatter,
        Yaml::Hash(yaml_rust2::yaml::Hash::new())
    );
}

// === stripFrontmatter =======================================================

#[test]
fn removes_frontmatter_and_trims_body() {
    let input = "---\nkey: value\n---\n\nBody\n";
    assert_eq!(strip_frontmatter(input).expect("strip"), "Body");
}

#[test]
fn returns_body_when_no_frontmatter_present() {
    let input = "\n  No frontmatter body  \n";
    assert_eq!(
        strip_frontmatter(input).expect("strip"),
        "\n  No frontmatter body  \n"
    );
}

// === the BOM strip this belt's copy carries (upstream's #8337 regression
// exercises it through the settings loader; the belt-level contract is the
// BOM split) ==============================================================

#[test]
fn split_bom_carries_the_mark_and_the_remainder() {
    let with_bom = split_bom("\u{FEFF}name: test\n---");
    assert_eq!(with_bom.bom, "\u{FEFF}");
    assert_eq!(with_bom.text, "name: test\n---");

    let parsed = parse_frontmatter("\u{FEFF}---\nname: bom\n---\nBody").expect("parse");
    assert_eq!(yaml_str(&parsed.frontmatter, "name"), Some("bom"));
    assert_eq!(parsed.body, "Body");

    let without_bom = split_bom("plain");
    assert_eq!(without_bom.bom, "");
    assert_eq!(without_bom.text, "plain");

    assert_eq!(pi_coding_agent::utils::text::strip_bom("\u{FEFF}x"), "x");
    assert_eq!(pi_coding_agent::utils::text::strip_bom("x"), "x");
}
