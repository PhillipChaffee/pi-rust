//! TruncatedText component, ported from
//! `packages/tui/src/components/truncated-text.ts` in earendil-works/pi at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#42).
//!
//! A single-line text that truncates to fit the viewport width.

use crate::tui::Component;
use crate::utils::{truncate_to_width, visible_width};

/// Text component that truncates to fit viewport width.
#[derive(Debug)]
pub struct TruncatedText {
    text: String,
    padding_x: usize,
    padding_y: usize,
}

impl TruncatedText {
    /// Upstream's default-argument constructor: no padding.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self::with_padding(text, 0, 0)
    }

    /// Upstream's `new TruncatedText(text, paddingX, paddingY)`.
    #[must_use]
    pub fn with_padding(text: impl Into<String>, padding_x: usize, padding_y: usize) -> Self {
        Self {
            text: text.into(),
            padding_x,
            padding_y,
        }
    }
}

impl Component for TruncatedText {
    // No cached state to invalidate currently, upstream's empty `invalidate`.
    fn render(&self, width: usize) -> Vec<String> {
        let mut result: Vec<String> = Vec::new();

        // Empty line padded to width
        let empty_line = " ".repeat(width);

        // Add vertical padding above
        for _ in 0..self.padding_y {
            result.push(empty_line.clone());
        }

        // Calculate available width after horizontal padding
        let available_width = width.saturating_sub(self.padding_x * 2).max(1);

        // Take only the first line (stop at newline)
        let single_line_text = self
            .text
            .find('\n')
            .map_or(self.text.as_str(), |newline_index| {
                &self.text[..newline_index]
            });

        // Truncate text if needed (accounting for ANSI codes)
        let display_text = truncate_to_width(single_line_text, available_width, "...", false);

        // Add horizontal padding
        let left_padding = " ".repeat(self.padding_x);
        let right_padding = " ".repeat(self.padding_x);
        let line_with_padding = format!("{left_padding}{display_text}{right_padding}");

        // Pad line to exactly width characters
        let line_visible_width = visible_width(&line_with_padding);
        let padding_needed = width.saturating_sub(line_visible_width);
        result.push(line_with_padding + " ".repeat(padding_needed).as_str());

        // Add vertical padding below
        for _ in 0..self.padding_y {
            result.push(empty_line.clone());
        }

        result
    }
}
