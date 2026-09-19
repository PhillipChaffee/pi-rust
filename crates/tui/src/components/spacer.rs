//! Spacer component, ported from `packages/tui/src/components/spacer.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#42).
//!
//! Renders empty lines.

use std::cell::Cell;

use crate::tui::Component;

/// Spacer component that renders empty lines.
#[derive(Debug)]
pub struct Spacer {
    lines: Cell<usize>,
}

impl Spacer {
    /// Upstream's default-argument constructor: one blank line.
    #[must_use]
    pub const fn new() -> Self {
        Self::with_lines(1)
    }

    /// Upstream's `new Spacer(lines)`.
    #[must_use]
    pub const fn with_lines(lines: usize) -> Self {
        Self {
            lines: Cell::new(lines),
        }
    }

    /// Upstream `setLines`.
    pub fn set_lines(&self, lines: usize) {
        self.lines.set(lines);
    }
}

impl Default for Spacer {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for Spacer {
    fn render(&self, _width: usize) -> Vec<String> {
        vec![String::new(); self.lines.get()]
    }
}
