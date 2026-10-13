//! The extension-system type surface, upstream's `src/core/extensions/`.
//!
//! The tool definitions and context the built-in tools execute over land
//! with the tools slice; the out-of-process loader (#128) and the
//! interactive surfaces ride their tickets.

pub mod api;
pub mod events;
pub mod reasons;
pub mod results;
pub mod runner;
pub mod runtime;
pub mod types;
pub mod wrapper;

pub use types::{
    CwdContext, ExtensionContext, RenderShell, ToolDefinition, ToolExecuteFn, extension_context,
};
