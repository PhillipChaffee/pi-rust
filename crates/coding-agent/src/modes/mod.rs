//! The app modes, upstream's `src/modes/`.
//!
//! Interactive mode lives with its TUI tickets (#131, #132); this module
//! carries the non-interactive modes and the event wire they share.

pub mod interactive;
pub mod json_event;
pub mod print_mode;
