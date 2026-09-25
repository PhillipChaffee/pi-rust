//! Shared truncation utilities for tool outputs, ported from upstream
//! `src/harness/utils/truncate.ts`.
//!
//! Truncation is based on two independent limits — whichever is hit first
//! wins: a line limit (default 2000 lines) and a byte limit (default 50
//! KiB). Truncation never returns partial lines, except the bash tail
//! truncation edge case where the last retained line alone exceeds the
//! byte limit.
//!
//! Upstream's UTF-16 surrogate repair paths (`replaceUnpairedSurrogates`
//! and the unpaired-surrogate branch of the tail truncation) have no
//! counterpart: Rust strings are UTF-8 and cannot hold unpaired surrogates,
//! the same statically-upheld invariant the pi-ai port records for
//! `sanitize-unicode.ts`. Byte lengths are Rust `str::len()` — UTF-8 byte
//! counts by construction, replacing upstream's `Buffer.byteLength` probe.

use crate::harness::types::{ShellOutputTruncation, TruncatedBy};

/// Default maximum number of lines, upstream's `DEFAULT_MAX_LINES`.
pub const DEFAULT_MAX_LINES: u64 = 2000;
/// Default maximum size in bytes, upstream's `DEFAULT_MAX_BYTES` (50 KiB).
pub const DEFAULT_MAX_BYTES: u64 = 50 * 1024;
/// Maximum characters per grep match line, upstream's `GREP_MAX_LINE_LENGTH`.
pub const GREP_MAX_LINE_LENGTH: usize = 500;

/// Options for the truncation functions, upstream's `TruncationOptions`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TruncationOptions {
    /// Maximum number of lines. Defaults to [`DEFAULT_MAX_LINES`].
    pub max_lines: Option<u64>,
    /// Maximum number of bytes. Defaults to [`DEFAULT_MAX_BYTES`].
    pub max_bytes: Option<u64>,
}

/// The full truncation result, upstream's `TruncationResult`: the retained
/// content plus the [`ShellOutputTruncation`] metadata (derefs to it, and
/// serializes flat like upstream's single object).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TruncationResult {
    /// The truncated content.
    pub content: String,
    /// The truncation metadata.
    pub metadata: ShellOutputTruncation,
}

impl std::ops::Deref for TruncationResult {
    type Target = ShellOutputTruncation;

    fn deref(&self) -> &Self::Target {
        &self.metadata
    }
}

impl TruncationResult {
    /// The retained content, consuming the result.
    #[must_use]
    pub fn into_content(self) -> String {
        self.content
    }
}

/// Counts the UTF-8 byte length of `content`; Rust strings carry the byte
/// length directly, replacing upstream's `Buffer.byteLength` probe.
#[must_use]
pub fn utf8_byte_length(content: &str) -> u64 {
    u64::try_from(content.len()).unwrap_or(u64::MAX)
}

/// Splits into lines for counting: a trailing newline does not contribute
/// a final empty line, and an empty string has no lines.
fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// Formats bytes as a human-readable size, upstream's `formatSize`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "the label is human-readable; precision beyond 2^53 bytes is invisible at KB/MB scale"
)]
pub fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Truncates content from the head, keeping the first N lines and bytes.
/// Suitable for file reads where the beginning matters.
///
/// Never returns partial lines. If the first line exceeds the byte limit,
/// returns empty content with `first_line_exceeds_limit` set.
#[must_use]
pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
    let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);

    let total_bytes = utf8_byte_length(content);
    let lines = split_lines_for_counting(content);
    let total_lines = u64::try_from(lines.len()).unwrap_or(u64::MAX);

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncationResult {
            content: content.to_owned(),
            metadata: ShellOutputTruncation {
                truncated: false,
                truncated_by: None,
                total_lines,
                total_bytes,
                output_lines: total_lines,
                output_bytes: total_bytes,
                last_line_partial: false,
                first_line_exceeds_limit: false,
                max_lines,
                max_bytes,
            },
        };
    }

    // Check if the first line alone exceeds the byte limit.
    let first_line_bytes = lines.first().map_or(0, |line| utf8_byte_length(line));
    if first_line_bytes > max_bytes {
        return TruncationResult {
            content: String::new(),
            metadata: ShellOutputTruncation {
                truncated: true,
                truncated_by: Some(TruncatedBy::Bytes),
                total_lines,
                total_bytes,
                output_lines: 0,
                output_bytes: 0,
                last_line_partial: false,
                first_line_exceeds_limit: true,
                max_lines,
                max_bytes,
            },
        };
    }

    // Collect complete lines that fit; `line_bytes` folds the separating
    // newline into every line after the first, upstream's `+1` accounting.
    let mut output_lines_arr: Vec<&str> = Vec::new();
    let mut output_bytes_count: u64 = 0;
    let mut truncated_by = TruncatedBy::Lines;

    for (index, line) in lines.iter().enumerate() {
        if u64::try_from(index).unwrap_or(u64::MAX) >= max_lines {
            break;
        }
        let line_bytes = utf8_byte_length(line) + u64::from(index > 0);

        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }

        output_lines_arr.push(line);
        output_bytes_count += line_bytes;
    }

    // If we exited due to the line limit.
    let kept = u64::try_from(output_lines_arr.len()).unwrap_or(u64::MAX);
    if kept >= max_lines && output_bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output_lines_arr.join("\n");
    let final_output_bytes = utf8_byte_length(&output_content);

    TruncationResult {
        content: output_content,
        metadata: ShellOutputTruncation {
            truncated: true,
            truncated_by: Some(truncated_by),
            total_lines,
            total_bytes,
            output_lines: kept,
            output_bytes: final_output_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
            max_lines,
            max_bytes,
        },
    }
}

/// Truncates content from the tail, keeping the last N lines and bytes.
/// Suitable for bash output where the end (errors, final results) matters.
///
/// May return a partial first line if the last line of the original
/// content exceeds the byte limit.
#[must_use]
pub fn truncate_tail(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
    let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);

    let total_bytes = utf8_byte_length(content);
    let lines = split_lines_for_counting(content);
    let total_lines = u64::try_from(lines.len()).unwrap_or(u64::MAX);

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncationResult {
            content: content.to_owned(),
            metadata: ShellOutputTruncation {
                truncated: false,
                truncated_by: None,
                total_lines,
                total_bytes,
                output_lines: total_lines,
                output_bytes: total_bytes,
                last_line_partial: false,
                first_line_exceeds_limit: false,
                max_lines,
                max_bytes,
            },
        };
    }

    // Work backwards from the end.
    let mut output_lines_arr: Vec<String> = Vec::new();
    let mut output_bytes_count: u64 = 0;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;

    for line in lines.iter().rev() {
        if u64::try_from(output_lines_arr.len()).unwrap_or(u64::MAX) >= max_lines {
            break;
        }
        let line_bytes = utf8_byte_length(line) + u64::from(!output_lines_arr.is_empty()); // +1 for the newline

        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            // Edge case: no line kept yet and this line alone exceeds the
            // byte limit — take the end of the line (partial).
            if output_lines_arr.is_empty() {
                let truncated_line = truncate_string_to_bytes_from_end(line, max_bytes);
                output_bytes_count = utf8_byte_length(&truncated_line);
                output_lines_arr.insert(0, truncated_line);
                last_line_partial = true;
            }
            break;
        }

        output_lines_arr.insert(0, (*line).to_owned());
        output_bytes_count += line_bytes;
    }

    // If we exited due to the line limit.
    let kept = u64::try_from(output_lines_arr.len()).unwrap_or(u64::MAX);
    if kept >= max_lines && output_bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output_lines_arr.join("\n");
    let final_output_bytes = utf8_byte_length(&output_content);

    TruncationResult {
        content: output_content,
        metadata: ShellOutputTruncation {
            truncated: true,
            truncated_by: Some(truncated_by),
            total_lines,
            total_bytes,
            output_lines: kept,
            output_bytes: final_output_bytes,
            last_line_partial,
            first_line_exceeds_limit: false,
            max_lines,
            max_bytes,
        },
    }
}

/// Truncates a string to fit within a byte limit, measured from the end.
/// Handles multi-byte UTF-8 characters correctly by backing up to char
/// boundaries; upstream's unpaired-surrogate replacement path has no Rust
/// counterpart (see the module docs).
#[must_use]
fn truncate_string_to_bytes_from_end(s: &str, max_bytes: u64) -> String {
    if max_bytes == 0 {
        return String::new();
    }
    let bytes = s.as_bytes();
    let limit = usize::try_from(max_bytes).unwrap_or(bytes.len());
    if bytes.len() <= limit {
        return s.to_owned();
    }
    let mut start = bytes.len() - limit;
    while start < bytes.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    s[start..].to_owned()
}

/// Truncates a single line to `max_chars` characters, adding a
/// `[truncated]` suffix. Used for grep match lines.
///
/// Upstream measures UTF-16 code units; the port counts chars, the closest
/// Rust analog for the user-visible "character" budget.
#[must_use]
pub fn truncate_line(line: &str, max_chars: usize) -> TruncatedLine {
    if line.chars().count() <= max_chars {
        return TruncatedLine {
            text: line.to_owned(),
            was_truncated: false,
        };
    }
    let cut: String = line.chars().take(max_chars).collect();
    TruncatedLine {
        text: format!("{cut}... [truncated]"),
        was_truncated: true,
    }
}

/// The result of [`truncate_line`], upstream's
/// `{ text, wasTruncated }` pair.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TruncatedLine {
    /// The line, possibly cut with a `[truncated]` suffix.
    pub text: String,
    /// Whether the line was cut.
    pub was_truncated: bool,
}

#[cfg(test)]
mod tests;
