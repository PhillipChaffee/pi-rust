//! Boundary tests for the truncation belt, upstream's `truncate.ts`
//! (the read/bash suites exercise the tool-level surfaces; these bind the
//! limits and the edge cases directly).

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use crate::tools::truncate::*;

#[test]
fn within_limits_passes_through() {
    let content = "one\ntwo\nthree";
    let result = truncate_head(content, TruncationOptions::default());
    assert_eq!(
        result,
        TruncationResult {
            content: content.to_owned(),
            truncated: false,
            truncated_by: None,
            total_lines: 3,
            total_bytes: content.len(),
            output_lines: 3,
            output_bytes: content.len(),
            last_line_partial: false,
            first_line_exceeds_limit: false,
            max_lines: 2000,
            max_bytes: 50 * 1024,
        }
    );
    let tail = truncate_tail(content, TruncationOptions::default());
    assert_eq!(tail, result);
}

#[test]
fn trailing_newline_does_not_open_a_line() {
    let result = truncate_head("a\nb\n", TruncationOptions::default());
    assert_eq!(result.total_lines, 2);
}

#[test]
fn head_keeps_the_first_lines() {
    let content = (1..=2500usize)
        .map(|i| format!("Line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let result = truncate_head(&content, TruncationOptions::default());
    assert!(result.truncated);
    assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
    assert_eq!(result.total_lines, 2500);
    assert_eq!(result.output_lines, 2000);
    assert_eq!(result.output_lines, result.content.split('\n').count());
    assert!(!result.last_line_partial);
    assert!(!result.first_line_exceeds_limit);
}

#[test]
fn head_byte_limit_reports_bytes() {
    // Five hundred lines of ~208 bytes each: the line budget never fires,
    // the 50KB byte budget does.
    let content = (1..=500usize)
        .map(|i| format!("Line {i}: {}", "x".repeat(200)))
        .collect::<Vec<_>>()
        .join("\n");
    let result = truncate_head(&content, TruncationOptions::default());
    assert!(result.truncated);
    assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    assert_eq!(result.total_lines, 500);
    assert!(result.output_lines < 500);
}

#[test]
fn head_first_line_exceeds_byte_limit() {
    let long_line = "x".repeat(60 * 1024);
    let content = format!("{long_line}\nsecond\n");
    let result = truncate_head(&content, TruncationOptions::default());
    assert!(result.truncated);
    assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    assert_eq!(result.content, "");
    assert_eq!(result.output_lines, 0);
    assert_eq!(result.output_bytes, 0);
    assert!(result.first_line_exceeds_limit);
    assert_eq!(result.total_lines, 2);
}

#[test]
fn tail_keeps_the_last_lines() {
    let content = (1..=4000usize)
        .map(|i| format!("line-{i:0>4}"))
        .collect::<Vec<_>>()
        .join("\n");
    let result = truncate_tail(&content, TruncationOptions::default());
    assert!(result.truncated);
    assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
    assert_eq!(result.total_lines, 4000);
    assert_eq!(result.output_lines, 2000);
    assert!(result.content.starts_with("line-2001\n"));
    assert!(result.content.ends_with("line-4000"));
}

#[test]
fn tail_byte_limit_reports_bytes() {
    let content = (1..=500usize)
        .map(|i| format!("Line {i}: {}", "x".repeat(200)))
        .collect::<Vec<_>>()
        .join("\n");
    let result = truncate_tail(&content, TruncationOptions::default());
    assert!(result.truncated);
    assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    assert_eq!(result.total_lines, 500);
}

#[test]
fn tail_oversized_last_line_takes_the_partial_end() {
    // The last line alone exceeds the byte limit: the tail takes its end.
    let long_line = "y".repeat(60 * 1024);
    let content = format!("first\n{long_line}");
    let result = truncate_tail(&content, TruncationOptions::default());
    assert!(result.truncated);
    assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    assert_eq!(result.output_lines, 1);
    assert!(result.last_line_partial);
    assert_eq!(result.output_bytes, 50 * 1024);
    assert!(result.content.ends_with("yyyy"));
    assert!(!result.content.contains("first"));
}

#[test]
fn tail_partial_cut_lands_on_a_utf8_boundary() {
    // € is three bytes; a byte-level cut into it must skip to the boundary
    // rather than produce invalid UTF-8.
    let long_line = "€".repeat((50 * 1024 / 3) + 10);
    let content = format!("first\n{long_line}");
    let result = truncate_tail(&content, TruncationOptions::default());
    assert!(result.last_line_partial);
    // At most two continuation bytes were skipped past the byte budget.
    assert!(result.output_bytes <= 50 * 1024);
    assert!(result.output_bytes > 50 * 1024 - 3);
    assert!(result.content.chars().all(|c| c == '€'));
}

#[test]
fn tail_line_limit_beats_byte_headroom() {
    // Exactly maxLines lines that jointly fit the byte budget: the line
    // limit is the reported cause.
    let content = (1..=2000usize)
        .map(|i| format!("{i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let result = truncate_tail(&content, TruncationOptions::default());
    // 2000 lines at ≤4 bytes each fit well under 50KB, so no truncation.
    assert!(!result.truncated);
}

#[test]
fn format_size_buckets() {
    assert_eq!(format_size(0), "0B");
    assert_eq!(format_size(1023), "1023B");
    assert_eq!(format_size(1024), "1.0KB");
    assert_eq!(format_size(1536), "1.5KB");
    assert_eq!(format_size(1024 * 1024), "1.0MB");
    assert_eq!(format_size(2 * 1024 * 1024 + 512 * 1024), "2.5MB");
}

#[test]
fn truncate_line_appends_the_suffix() {
    let short = truncate_line_default("short line");
    assert_eq!(short.text, "short line");
    assert!(!short.was_truncated);

    let long = "z".repeat(600);
    let truncated = truncate_line_default(&long);
    assert!(truncated.was_truncated);
    assert_eq!(truncated.text.len(), 500 + "... [truncated]".len());
    assert!(truncated.text.starts_with(&"z".repeat(500)));
    assert!(truncated.text.ends_with("... [truncated]"));
}

#[test]
fn truncate_line_counts_utf16_units() {
    // € is one UTF-16 unit; a 500-char cut keeps it whole.
    let line = "€".repeat(600);
    let truncated = truncate_line_default(&line);
    assert!(truncated.was_truncated);
    let head = truncated.text.strip_suffix("... [truncated]").unwrap();
    assert_eq!(head.chars().count(), 500);

    // An astral character is two UTF-16 units: a 500-unit cut keeps whole
    // characters (upstream's slice could split the pair; Rust cannot).
    let emoji = "🐛".repeat(300); // 600 UTF-16 units
    let truncated = truncate_line_default(&emoji);
    assert!(truncated.was_truncated);
    let head = truncated.text.strip_suffix("... [truncated]").unwrap();
    assert_eq!(head.chars().count(), 250);
}
