//! Interface for custom editor components, ported from
//! `packages/tui/src/editor-component.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#47).
//!
//! This allows extensions to provide their own editor implementation
//! (e.g. vim mode, emacs mode, custom keybindings) while maintaining
//! compatibility with the core application.
//!
//! Restatements against upstream:
//!
//! - Upstream's optional interface members are properties and methods the
//!   consumer checks for; traits restate them as defaulted methods the
//!   implementation may override.
//! - `onSubmit`/`onChange` are consumer-assigned callback fields rather
//!   than behavioral members; they live on the concrete editor and carry
//!   no trait method here.
//! - `setAutocompleteProvider` and `setAutocompleteMaxVisible` ship with
//!   the autocomplete child
//!   ([#49](https://github.com/PhillipChaffee/pi-rust/issues/49)).
//! - `borderColor` is a consumer-assigned field on [`crate::components::Editor`],
//!   not a trait member.

use crate::tui::Component;

/// The custom-editor contract, upstream `EditorComponent`.
pub trait EditorComponent: Component {
    /// Get the current text content, upstream `getText`.
    fn get_text(&self) -> String;

    /// Set the text content, upstream `setText`.
    fn set_text(&self, text: &str);

    /// Add text to history for up/down navigation, upstream
    /// `addToHistory?`. Default: unsupported.
    fn add_to_history(&self, text: &str) {
        let _ = text;
    }

    /// Insert text at the current cursor position, upstream
    /// `insertTextAtCursor`. Default: unsupported.
    fn insert_text_at_cursor(&self, text: &str) {
        let _ = text;
    }

    /// Get text with any markers expanded (e.g. paste markers), upstream
    /// `getExpandedText`. `None` mirrors upstream's absent member: callers
    /// fall back to [`EditorComponent::get_text`].
    fn get_expanded_text(&self) -> Option<String> {
        None
    }

    /// Set the horizontal padding, upstream `setPaddingX`. Default: no-op.
    fn set_padding_x(&self, padding: usize) {
        let _ = padding;
    }
}
