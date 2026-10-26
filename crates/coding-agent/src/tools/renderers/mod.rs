//! Built-in tool renderers, upstream `src/core/tools/renderers/index.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Renderers stay separate from the tool implementations so a presentation
//! can draw rows without pulling in the execution path. A definition that
//! supplies no renderer inherits the built-in slot for its name.

pub mod bash;
pub mod edit;
pub mod find;
pub mod grep;
pub mod ls;
pub mod read;
pub mod write;

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::extensions::types::{ToolDefinition, ToolRenderCallFn, ToolRenderResultFn};
use crate::tools::index::ToolName;

use self::bash::create_shell_renderers;
use self::edit::edit_renderers;
use self::find::find_renderers;
use self::grep::grep_renderers;
use self::ls::ls_renderers;
use self::read::read_renderers;
use self::write::write_renderers;

/// The pair of hooks a tool definition may supply, upstream's `ToolRenderers`.
#[derive(Clone, Default)]
pub struct ToolRenderers {
    /// Custom rendering for the tool call display.
    pub render_call: Option<Arc<ToolRenderCallFn>>,
    /// Custom rendering for the tool result display.
    pub render_result: Option<Arc<ToolRenderResultFn>>,
}

impl std::fmt::Debug for ToolRenderers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRenderers")
            .field("render_call", &self.render_call.is_some())
            .field("render_result", &self.render_result.is_some())
            .finish()
    }
}

/// Renderers for every built-in tool, keyed by tool name, upstream's
/// `createAllToolRenderers`.
#[must_use]
pub fn create_all_tool_renderers() -> BTreeMap<ToolName, ToolRenderers> {
    BTreeMap::from([
        (ToolName::Read, read_renderers()),
        (ToolName::Bash, create_shell_renderers("$")),
        (ToolName::Powershell, create_shell_renderers("PS>")),
        (ToolName::Edit, edit_renderers()),
        (ToolName::Write, write_renderers()),
        (ToolName::Grep, grep_renderers()),
        (ToolName::Find, find_renderers()),
        (ToolName::Ls, ls_renderers()),
    ])
}

/// Merge the built-in renderers into a definition that does not supply its
/// own, upstream's `withBuiltInRenderers`.
#[must_use]
pub fn with_built_in_renderers(
    tool_name: &str,
    definition: Option<ToolDefinition>,
) -> Option<ToolDefinition> {
    let built_in = ToolName::parse(tool_name)
        .ok()
        .and_then(|name| create_all_tool_renderers().remove(&name));
    let Some(built_in) = built_in else {
        return definition;
    };
    let mut definition = definition?;
    if definition.render_call.is_none() {
        definition.render_call = built_in.render_call;
    }
    if definition.render_result.is_none() {
        definition.render_result = built_in.render_result;
    }
    Some(definition)
}
