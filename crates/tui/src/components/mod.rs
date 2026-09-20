//! The leaf components of `packages/tui/src/components` in earendil-works/pi
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#42).
//!
//! This directory carries the padding-and-background container, the wrapping
//! text display, the truncated title line, the blank-line spacer, and the
//! loader family. The stacks, scroll view, and layout engine land with the
//! layout ticket (#44); the editor, input, and selection machinery carry
//! their own tickets.
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

pub mod box_component;
pub mod cancellable_loader;
pub mod h_stack;
pub mod loader;
pub mod mouse_region;
pub mod scroll_view;
pub mod spacer;
pub mod stack;
pub mod text;
pub mod truncated_text;
pub mod v_stack;

use std::sync::Arc;

/// A text-decoration callback, upstream `(text: string) => string`. Shared
/// because the loader's animation thread invokes its color functions.
pub type ColorFn = Arc<dyn Fn(&str) -> String + Send + Sync>;

pub use box_component::Box;
pub use cancellable_loader::CancellableLoader;
pub use h_stack::HStack;
pub use loader::{Loader, LoaderIndicatorOptions};
pub use mouse_region::MouseRegion;
pub use scroll_view::{
    FollowMode, ScrollView, ScrollViewOptions, ScrollViewScrollToOptions, ScrollViewScrollbar,
};
pub use spacer::Spacer;
pub use stack::{StackChild, StackEntry, StackEntryOptions, StackOptions, allocate_stack_sizes};
pub use text::Text;
pub use truncated_text::TruncatedText;
pub use v_stack::VStack;
