//! The extension-system type surface, upstream's `src/core/extensions/`.
//!
//! The tool definitions and context the built-in tools execute over land
//! with the tools slice; the extension loading, event dispatch, and
//! registration machinery ride the extension-system ticket.

pub mod types;

pub use types::{
    CwdContext, ExtensionContext, RenderShell, ToolDefinition, ToolExecuteFn, extension_context,
};
