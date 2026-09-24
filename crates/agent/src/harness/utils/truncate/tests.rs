//! The truncate utility suite, ported from upstream
//! `test/harness/truncate.test.ts`.
//!
//! The buffer-tail equivalence harness restates over Rust bytes: upstream
//! compares against Node `Buffer` semantics for lone-surrogate inputs,
//! which cannot exist in a Rust `str` — those cases collapse, and the
//! exhaustive fuzz walks the same alphabet minus the lone surrogates.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
#![expect(clippy::unwrap_used, reason = "tests unwrap the pinned outcomes")]

use crate::harness::types::TruncatedBy;
use crate::harness::utils::truncate::{truncate_head, truncate_tail, utf8_byte_length, TruncationOptions};

fn options(max_bytes: u64, max_lines: u64) -> TruncationOptions {
    TruncationOptions {
        max_lines: Some(max_lines),
        max_bytes: Some(max_bytes),
    }
}

/// UTF-8 byte counts come straight from the string, upstream's
/// `Buffer.byteLength` probe.
#[test]
fn counts_utf8_bytes_without_node_buffer() {
    let content = "aé🙂\nb";
    let result = truncate_head(content, options(100, 10));
    assert!(!result.truncated);
    assert_eq!(result.metadata.total_bytes, utf8_byte_length(content));
    assert_eq!(result.metadata.total_bytes, 9);
}

/// A trailing newline is not a final empty line.
#[test]
fn does_not_count_a_trailing_newline_as_an_extra_line() {
    let content = "line\nline\nline\n";
    let head = truncate_head(content, options(100, 3));
    let tail = truncate_tail(content, options(100, 3));
    assert!(!head.truncated);
    assert_eq!(head.metadata.total_lines, 3);
    assert_eq!(head.metadata.output_lines, 3);
    assert_eq!(tail.metadata.total_lines, 3);
    assert_eq!(tail.metadata.output_lines, 3);
}

/// Head truncation never returns partial lines.
#[test]
fn truncates_head_on_utf8_byte_limits_without_partial_lines() {
    let result = truncate_head("éé\nabc", options(4, 10));
    assert_eq!(result.content, "éé");
    assert!(result.truncated);
    assert_eq!(result.metadata.truncated_by, Some(TruncatedBy::Bytes));
    assert_eq!(result.metadata.output_bytes, 4);
    assert!(!result.metadata.first_line_exceeds_limit);
}

/// A first line larger than the whole byte limit truncates to empty with
/// the dedicated flag.
#[test]
fn reports_head_truncation_when_the_first_line_exceeds_the_byte_limit() {
    let result = truncate_head("éé\nabc", options(3, 10));
    assert_eq!(result.content, "");
    assert!(result.truncated);
    assert_eq!(result.metadata.truncated_by, Some(TruncatedBy::Bytes));
    assert!(result.metadata.first_line_exceeds_limit);
}

/// Tail truncation lands on UTF-8 boundaries when only a partial last
/// line fits.
#[test]
fn truncates_tail_on_utf8_boundaries() {
    let result = truncate_tail("aé🙂b", options(5, 10));
    assert_eq!(result.content, "🙂b");
    assert!(result.truncated);
    assert_eq!(result.metadata.truncated_by, Some(TruncatedBy::Bytes));
    assert!(result.metadata.last_line_partial);
    assert_eq!(result.metadata.output_bytes, 5);
}

/// An oversized single line with a trailing newline truncates to the
/// byte budget.
#[test]
fn truncates_an_oversized_single_line_with_a_trailing_newline() {
    let input = format!("{}\n", "X".repeat(300_000));
    let result = truncate_tail(&input, options(1024, 100));
    assert_eq!(result.content, "X".repeat(1024));
    assert_eq!(result.metadata.output_bytes, 1024);
    assert_eq!(result.metadata.output_lines, 1);
    assert!(result.metadata.last_line_partial);
    assert_eq!(result.metadata.truncated_by, Some(TruncatedBy::Bytes));
}

/// A trailing character that cannot fit in the tail byte limit drops.
#[test]
fn drops_an_oversized_trailing_character() {
    let result = truncate_tail("abc🙂", options(3, 10));
    assert_eq!(result.content, "");
    assert!(result.truncated);
    assert_eq!(result.metadata.truncated_by, Some(TruncatedBy::Bytes));
    assert!(result.metadata.last_line_partial);
    assert_eq!(result.metadata.output_bytes, 0);
}
/// The tail truncation stays byte-exact against a reference tail walk for
/// every prefix length, over multi-byte alphabets (upstream's Buffer
/// equivalence harness; the lone-surrogate cases cannot exist in Rust).
#[test]
fn matches_the_reference_tail_semantics_over_multi_byte_inputs() {
    let alphabet = ["a", "\u{7f}", "\u{80}", "é", "\u{7ff}", "\u{800}", "中", "🙂", "\u{e000}", "\u{ffff}"];
    fn reference_tail(content: &str, max_bytes: u64) -> String {
        let bytes = content.as_bytes();
        if bytes.len() as u64 <= max_bytes {
            return content.to_owned();
        }
        let mut start = bytes.len() - usize::try_from(max_bytes).unwrap_or(bytes.len());
        while start < bytes.len() && (bytes[start] & 0xc0) == 0x80 {
            start += 1;
        }
        String::from_utf8_lossy(&bytes[start..]).into_owned()
    }
    fn check(alphabet: &[&str], prefix: String, depth: usize) {
        let total = utf8_byte_length(&prefix);
        for limit in 0..=total + 5 {
            let result = truncate_tail(&prefix, options(limit, 10));
            let expected = reference_tail(&prefix, limit);
            assert_eq!(result.content, expected, "input={prefix:?} maxBytes={limit}");
            assert!(
                utf8_byte_length(&result.content) <= limit,
                "tail exceeded the limit: input={prefix:?} maxBytes={limit}"
            );
        }
        if depth == 0 {
            return;
        }
        for character in alphabet {
            check(alphabet, format!("{prefix}{character}"), depth - 1);
        }
    }
    check(&alphabet, String::new(), 2);
}
