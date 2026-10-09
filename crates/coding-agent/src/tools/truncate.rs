//! Shared truncation utilities for tool outputs, ported from upstream
//! `src/core/tools/truncate.ts`.
//!
//! Truncation is based on two independent limits — whichever is hit first
//! wins: the line limit (default 2000 lines) and the byte limit (default
//! 50KB). Neither ever returns partial lines, except the bash tail-truncation
//! edge case that takes the end of an oversized line.
//!
//! JS string lengths restate as UTF-8 byte counts for the limits (upstream's
//! `Buffer.byteLength`) and as UTF-16 code units for [`truncate_line`]
//! (upstream's `String.length`/`slice`); a UTF-16 cut that would split a
//! surrogate pair lands before the whole character, which Rust strings
//! cannot split.

/// The default line limit, upstream's `DEFAULT_MAX_LINES`.
pub const DEFAULT_MAX_LINES: usize = 2000;

/// The default byte limit, upstream's `DEFAULT_MAX_BYTES` (50KB).
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// The max chars per grep match line, upstream's `GREP_MAX_LINE_LENGTH`.
pub const GREP_MAX_LINE_LENGTH: usize = 500;

/// The truncation outcome, upstream's `TruncationResult`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TruncationResult {
    /// The truncated content.
    pub content: String,
    /// Whether truncation occurred.
    pub truncated: bool,
    /// Which limit was hit, absent when not truncated.
    pub truncated_by: Option<TruncatedBy>,
    /// Total number of lines in the original content.
    pub total_lines: usize,
    /// Total number of bytes in the original content.
    pub total_bytes: usize,
    /// Number of complete lines in the truncated output.
    pub output_lines: usize,
    /// Number of bytes in the truncated output.
    pub output_bytes: usize,
    /// Whether the last line was partially truncated (only for the tail
    /// truncation edge case).
    pub last_line_partial: bool,
    /// Whether the first line exceeded the byte limit (for head truncation).
    pub first_line_exceeds_limit: bool,
    /// The max lines limit that was applied.
    pub max_lines: usize,
    /// The max bytes limit that was applied.
    pub max_bytes: usize,
}

/// Which limit fired, upstream's `"lines" | "bytes"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TruncatedBy {
    /// The line limit fired.
    Lines,
    /// The byte limit fired.
    Bytes,
}

/// The truncation limits, upstream's `TruncationOptions`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TruncationOptions {
    /// Maximum number of lines (default: 2000).
    pub max_lines: Option<usize>,
    /// Maximum number of bytes (default: 50KB).
    pub max_bytes: Option<usize>,
}

impl TruncationOptions {
    fn max_lines(&self) -> usize {
        self.max_lines.unwrap_or(DEFAULT_MAX_LINES)
    }

    fn max_bytes(&self) -> usize {
        self.max_bytes.unwrap_or(DEFAULT_MAX_BYTES)
    }
}

/// The untruncated outcome both directions share.
fn no_truncation(
    content: &str,
    lines: usize,
    max_lines: usize,
    max_bytes: usize,
) -> TruncationResult {
    TruncationResult {
        content: content.to_owned(),
        truncated: false,
        truncated_by: None,
        total_lines: lines,
        total_bytes: content.len(),
        output_lines: lines,
        output_bytes: content.len(),
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Split content into lines the way the line count runs, upstream's
/// `splitLinesForCounting`: a trailing newline does not open an extra line.
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

/// Format bytes as human-readable size, upstream's `formatSize`.
#[must_use]
pub fn format_size(bytes: usize) -> String {
    #[expect(
        clippy::cast_precision_loss,
        reason = "the KB/MB divisors only ever shrink the count; upstream prints the f64"
    )]
    let scaled = |divisor: usize| -> String {
        let value = bytes as f64 / divisor as f64;
        format!("{value:.1}")
    };
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{}KB", scaled(1024))
    } else {
        format!("{}MB", scaled(1024 * 1024))
    }
}

/// Truncate content from the head (keep first N lines/bytes), upstream's
/// `truncateHead`. Suitable for file reads where the beginning matters.
///
/// Never returns partial lines. If the first line exceeds the byte limit,
/// returns empty content with `first_line_exceeds_limit` set.
#[must_use]
pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines();
    let max_bytes = options.max_bytes();

    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return no_truncation(content, total_lines, max_lines, max_bytes);
    }

    // Check if first line alone exceeds byte limit
    let first_line_bytes = lines.first().map_or(0, |line| line.len());
    if first_line_bytes > max_bytes {
        return TruncationResult {
            content: String::new(),
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
        };
    }

    // Collect complete lines that fit
    let mut output_lines: Vec<&str> = Vec::new();
    let mut output_bytes_count = 0usize;
    let mut truncated_by = TruncatedBy::Lines;

    for (index, line) in lines.iter().enumerate().take(max_lines) {
        // +1 for the newline joining the line into the output
        let line_bytes = line.len() + usize::from(index > 0);
        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        output_lines.push(line);
        output_bytes_count += line_bytes;
    }

    // If we exited due to line limit
    if output_lines.len() >= max_lines && output_bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output_lines.join("\n");
    let final_output_bytes = output_content.len();

    TruncationResult {
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output_lines.len(),
        output_bytes: final_output_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Truncate content from the tail (keep last N lines/bytes), upstream's
/// `truncateTail`. Suitable for bash output where the end (errors, final
/// results) matters.
///
/// May return a partial first line when the last line of the original
/// content exceeds the byte limit.
#[must_use]
pub fn truncate_tail(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines();
    let max_bytes = options.max_bytes();

    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return no_truncation(content, total_lines, max_lines, max_bytes);
    }

    // Work backwards from the end
    let mut output_lines: Vec<String> = Vec::new();
    let mut output_bytes_count = 0usize;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;

    for line in lines.iter().rev() {
        if output_lines.len() >= max_lines {
            break;
        }
        // +1 for the newline joining the line into the output
        let line_bytes = line.len() + usize::from(!output_lines.is_empty());
        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            // Edge case: if we haven't added ANY lines yet and this line
            // exceeds maxBytes, take the end of the line (partial)
            if output_lines.is_empty() {
                let truncated_line = truncate_string_to_bytes_from_end(line, max_bytes);
                output_bytes_count = truncated_line.len();
                output_lines.insert(0, truncated_line);
                last_line_partial = true;
            }
            break;
        }
        output_bytes_count += line_bytes;
        output_lines.insert(0, (*line).to_owned());
    }

    // If we exited due to line limit
    if output_lines.len() >= max_lines && output_bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output_lines.join("\n");
    let final_output_bytes = output_content.len();

    TruncationResult {
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output_lines.len(),
        output_bytes: final_output_bytes,
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Truncate a string to fit within a byte limit from the end, upstream's
/// `truncateStringToBytesFromEnd`. Handles multi-byte UTF-8 characters.
fn truncate_string_to_bytes_from_end(string: &str, max_bytes: usize) -> String {
    let bytes = string.as_bytes();
    if bytes.len() <= max_bytes {
        return string.to_owned();
    }

    // Start from the end, skip maxBytes back
    let mut start = bytes.len() - max_bytes;

    // Find a valid UTF-8 boundary (start of a character)
    while start < bytes.len() && (bytes[start] & 0xc0) == 0x80 {
        start += 1;
    }

    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

/// The single-line truncation outcome, upstream's `truncateLine` return.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TruncatedLine {
    /// The possibly truncated text.
    pub text: String,
    /// Whether the suffix was appended.
    pub was_truncated: bool,
}

/// Truncate a single line to max characters, adding the `[truncated]`
/// suffix, upstream's `truncateLine`. Used for grep match lines.
///
/// The cut runs over UTF-16 code units, upstream's `String.length`; a cut
/// that would split a surrogate pair lands before the whole character.
#[must_use]
pub fn truncate_line(line: &str, max_chars: usize) -> TruncatedLine {
    let utf16_len: usize = line.chars().map(char::len_utf16).sum();
    if utf16_len <= max_chars {
        return TruncatedLine {
            text: line.to_owned(),
            was_truncated: false,
        };
    }
    let mut units = 0usize;
    let mut head = String::new();
    for character in line.chars() {
        if units + character.len_utf16() > max_chars {
            break;
        }
        units += character.len_utf16();
        head.push(character);
    }
    TruncatedLine {
        text: format!("{head}... [truncated]"),
        was_truncated: true,
    }
}

/// The default-characters form of [`truncate_line`], upstream's default
/// `maxChars`.
#[must_use]
pub fn truncate_line_default(line: &str) -> TruncatedLine {
    truncate_line(line, GREP_MAX_LINE_LENGTH)
}
