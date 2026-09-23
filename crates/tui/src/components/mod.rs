//! The leaf components of `packages/tui/src/components` in earendil-works/pi
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#42).
//!
//! This directory carries the padding-and-background container, the wrapping
//! text display, the truncated title line, and the blank-line spacer, and
//! the loader family. The stacks, scroll view, and layout engine landed with
//! #44; the editor machinery (`Editor`, `Input`) landed with #47, and the
//! selection machinery (select list, settings list) landed with #49.
//!
//! Restatements against upstream:
//!
//! - `class X extends Y` becomes composition: [`Loader`] embeds a
//!   [`Text`], and [`CancellableLoader`] embeds a [`Loader`].
//! - Background/color functions — upstream `(text: string) => string`
//!   callbacks — become [`ColorFn`], a shared `Arc` handle: the loader's
//!   animation thread calls them.
//! - [`CancellableLoader`]'s `AbortSignal` becomes
//!   `tokio_util::sync::CancellationToken` per the stack decision (map
//!   ticket "Decide the Rust stack").

pub mod alt_screen_flash;
pub mod box_component;
pub mod cancellable_loader;
pub mod editor;
pub mod editor_component;
pub mod h_stack;
pub mod image;
pub mod input;
pub mod loader;
pub mod markdown;
pub mod mouse_region;
pub mod scroll_view;
pub mod select_list;
pub mod settings_list;
pub mod spacer;
pub mod stack;
pub mod text;
pub mod truncated_text;
pub mod v_stack;

use std::sync::Arc;

/// A text-decoration callback, upstream `(text: string) => string`. Shared
/// because the loader's animation thread invokes its color functions.
pub type ColorFn = Arc<dyn Fn(&str) -> String + Send + Sync>;

pub use alt_screen_flash::AltScreenFlashContainer;
pub use box_component::Box;
pub use cancellable_loader::CancellableLoader;
pub use editor::{
    CursorPosition, Editor, EditorColorFn, EditorOptions, EditorTheme, TextChunk, word_wrap_line,
};
pub use editor_component::EditorComponent;
pub use h_stack::HStack;
pub use input::{Input, InputOptions, InputStyleFn};
pub use loader::{Loader, LoaderIndicatorOptions};
pub use markdown::{DefaultTextStyle, Markdown, MarkdownOptions, MarkdownTheme};
pub use mouse_region::MouseRegion;
pub use scroll_view::{
    FollowMode, ScrollView, ScrollViewOptions, ScrollViewScrollToOptions, ScrollViewScrollbar,
};
pub use select_list::{
    SelectItem, SelectList, SelectListCallback, SelectListCancelCallback, SelectListColorFn,
    SelectListLayoutOptions, SelectListTheme, SelectListTruncatePrimaryContext,
    SelectListTruncatePrimaryFn,
};
pub use settings_list::{
    SettingItem, SettingsCancelCallback, SettingsChangeCallback, SettingsColorFn, SettingsList,
    SettingsListOptions, SettingsListTheme, SettingsSelectedColorFn, SettingsSubmenuDone,
    SettingsSubmenuDoneOptions, SettingsSubmenuFn,
};
pub use spacer::Spacer;
pub use stack::{StackChild, StackEntry, StackEntryOptions, StackOptions, allocate_stack_sizes};
pub use text::Text;
pub use truncated_text::TruncatedText;
pub use v_stack::VStack;
