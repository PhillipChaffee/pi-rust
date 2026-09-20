//! Text component, ported from `packages/tui/src/components/text.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#42).
//!
//! Multi-line text with word wrapping, margins, and an optional background.
//!
//! The component's whole state lives behind a `Mutex` rather than the
//! sibling components' `RefCell`s: [`crate::components::Loader`] embeds a
//! `Text` and its animation thread writes the display string, so the state
//! must be shareable across threads.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::components::ColorFn;
use crate::tui::Component;
use crate::utils::{apply_background_to_line, visible_width, wrap_text_with_ansi};

struct TextState {
    text: String,
    /// Left/right padding.
    padding_x: usize,
    /// Top/bottom padding.
    padding_y: usize,
    custom_bg_fn: Option<ColorFn>,
    cached_text: Option<String>,
    cached_width: Option<usize>,
    cached_lines: Option<Vec<String>>,
}

/// Text component that displays multi-line text with word wrapping.
pub struct Text {
    state: Mutex<TextState>,
}

impl std::fmt::Debug for Text {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Text").finish_non_exhaustive()
    }
}

impl Text {
    /// Upstream's default-argument constructor: one cell of padding on every
    /// side, no background.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self::with_background(text, 1, 1, None)
    }

    /// Upstream's `new Text(text, paddingX, paddingY)`.
    #[must_use]
    pub fn with_padding(text: impl Into<String>, padding_x: usize, padding_y: usize) -> Self {
        Self::with_background(text, padding_x, padding_y, None)
    }

    /// Upstream's `new Text(text, paddingX, paddingY, customBgFn)`.
    #[must_use]
    pub fn with_background(
        text: impl Into<String>,
        padding_x: usize,
        padding_y: usize,
        custom_bg_fn: Option<ColorFn>,
    ) -> Self {
        Self {
            state: Mutex::new(TextState {
                text: text.into(),
                padding_x,
                padding_y,
                custom_bg_fn,
                cached_text: None,
                cached_width: None,
                cached_lines: None,
            }),
        }
    }

    /// Upstream `setText`.
    pub fn set_text(&self, text: &str) {
        let mut state = self.lock();
        state.text = text.to_string();
        state.cached_text = None;
        state.cached_width = None;
        state.cached_lines = None;
    }

    /// Upstream `setCustomBgFn`.
    pub fn set_custom_bg_fn(&self, custom_bg_fn: Option<ColorFn>) {
        let mut state = self.lock();
        state.custom_bg_fn = custom_bg_fn;
        state.cached_text = None;
        state.cached_width = None;
        state.cached_lines = None;
    }

    fn lock(&self) -> MutexGuard<'_, TextState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Component for Text {
    fn render(&self, width: usize) -> Vec<String> {
        let mut state = self.lock();

        // Check cache
        if state.cached_lines.is_some()
            && state.cached_text.as_deref() == Some(state.text.as_str())
            && state.cached_width == Some(width)
        {
            return state.cached_lines.clone().unwrap_or_default();
        }

        // Don't render anything if there's no actual text
        if state.text.is_empty() || state.text.trim().is_empty() {
            state.cached_text = Some(state.text.clone());
            state.cached_width = Some(width);
            state.cached_lines = Some(Vec::new());
            return Vec::new();
        }

        // Replace tabs with 3 spaces
        let normalized_text = state.text.replace('\t', "   ");

        // Reduce margins when necessary so content and padding fit within
        // the available width.
        let padding_x = state.padding_x.min(width.saturating_sub(1) / 2);
        let content_width = width.saturating_sub(padding_x * 2).max(1);

        // Wrap text (this preserves ANSI codes but does NOT pad)
        let wrapped_lines = wrap_text_with_ansi(&normalized_text, content_width);

        // Add margins and background to each line
        let left_margin = " ".repeat(padding_x);
        let right_margin = " ".repeat(padding_x);
        let mut content_lines: Vec<String> = Vec::new();

        for line in wrapped_lines {
            // Add margins
            let line_with_margins = format!("{left_margin}{line}{right_margin}");

            // Apply background if specified (this also pads to full width)
            let styled = state.custom_bg_fn.as_ref().map_or_else(
                || {
                    // No background - just pad to width with spaces
                    let visible_len = visible_width(&line_with_margins);
                    let padding_needed = width.saturating_sub(visible_len);
                    format!("{line_with_margins}{}", " ".repeat(padding_needed))
                },
                |custom_bg_fn| {
                    apply_background_to_line(&line_with_margins, width, |text| custom_bg_fn(text))
                },
            );
            content_lines.push(styled);
        }

        // Add top/bottom padding (empty lines)
        let empty_line = " ".repeat(width);
        let blank = state.custom_bg_fn.as_ref().map_or_else(
            || empty_line.clone(),
            |custom_bg_fn| apply_background_to_line(&empty_line, width, |text| custom_bg_fn(text)),
        );
        let empty_lines: Vec<String> = (0..state.padding_y).map(|_| blank.clone()).collect();

        let mut result = empty_lines.clone();
        result.extend(content_lines);
        result.extend(empty_lines);

        // Update cache
        state.cached_text = Some(state.text.clone());
        state.cached_width = Some(width);
        state.cached_lines = Some(result.clone());
        drop(state);

        if result.is_empty() {
            vec![String::new()]
        } else {
            result
        }
    }

    fn invalidate(&self) {
        let mut state = self.lock();
        state.cached_text = None;
        state.cached_width = None;
        state.cached_lines = None;
    }
}
