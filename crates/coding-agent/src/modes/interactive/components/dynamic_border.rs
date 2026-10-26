//! The width-filling border, upstream's
//! `src/modes/interactive/components/dynamic-border.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream warns that extension-loaded module instances may not see the
//! global theme, so callers pass their own color function there. The port
//! keeps the same optional-by-constructor shape: [`DynamicBorder::themed`]
//! reads the process-wide theme, [`DynamicBorder::new`] takes an explicit
//! decorator.

use std::rc::Rc;

use pi_tui::tui::Component;

use crate::modes::interactive::theme::theme::theme;

/// A horizontal rule sized to the viewport, upstream's `DynamicBorder`.
pub struct DynamicBorder {
    color: Rc<dyn Fn(&str) -> String>,
}

impl DynamicBorder {
    /// Build a border with an explicit decorator.
    #[must_use]
    pub fn new(color: Rc<dyn Fn(&str) -> String>) -> Self {
        Self { color }
    }

    /// Build a border in the theme's `border` color.
    #[must_use]
    pub fn themed() -> Self {
        Self::new(Rc::new(|text: &str| theme().fg("border", text)))
    }
}

impl Component for DynamicBorder {
    fn render(&self, width: usize) -> Vec<String> {
        let rule = "─".repeat(width.max(1));
        vec![(self.color)(&rule)]
    }
}

impl std::fmt::Debug for DynamicBorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicBorder").finish_non_exhaustive()
    }
}
