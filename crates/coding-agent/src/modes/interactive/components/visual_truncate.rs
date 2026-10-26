//! Visual-line truncation shared by the tool and bash result renderers,
//! upstream `src/modes/interactive/components/visual-truncate.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Wrapping goes through [`Text`], the same component the surrounding rows
//! render with, so the cut matches what the terminal displays.

use pi_tui::components::Text;
use pi_tui::tui::Component;

/// The truncated window, upstream's `VisualTruncateResult`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VisualTruncateResult {
    /// The visual lines to display.
    pub visual_lines: Vec<String>,
    /// Number of visual lines that were skipped (hidden).
    pub skipped_count: usize,
}

/// Keep the last `max_visual_lines` visual lines of `text` at `width`
/// columns, upstream's `truncateToVisualLines`.
#[must_use]
pub fn truncate_to_visual_lines(
    text: &str,
    max_visual_lines: usize,
    width: usize,
    padding_x: usize,
) -> VisualTruncateResult {
    if text.is_empty() {
        return VisualTruncateResult::default();
    }

    let all_visual_lines = Text::with_padding(text, padding_x, 0).render(width);
    if all_visual_lines.len() <= max_visual_lines {
        return VisualTruncateResult {
            visual_lines: all_visual_lines,
            skipped_count: 0,
        };
    }

    let start = all_visual_lines.len() - max_visual_lines;
    VisualTruncateResult {
        visual_lines: all_visual_lines[start..].to_vec(),
        skipped_count: start,
    }
}
