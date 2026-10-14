//! The built-in coding tools barrel, ported from upstream
//! `src/core/tools/index.ts`: the `ToolName` vocabulary, the per-tool
//! option bundle, and the definition/tool factory family.

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_agent_core::harness::types::AgentHarnessTool;
use serde_json::Value;

use crate::extensions::types::ToolDefinition;

use super::bash::{BashToolOptions, create_bash_tool, create_bash_tool_definition};
use super::edit::{EditToolOptions, create_edit_tool, create_edit_tool_definition};
use super::find::{FindToolOptions, create_find_tool, create_find_tool_definition};
use super::grep::{GrepToolOptions, create_grep_tool, create_grep_tool_definition};
use super::ls::{LsToolOptions, create_ls_tool, create_ls_tool_definition};
use super::powershell::{
    PowerShellToolOptions, create_power_shell_tool, create_power_shell_tool_definition,
};
use super::read::{ReadToolOptions, create_read_tool, create_read_tool_definition};
use super::write::{WriteToolOptions, create_write_tool, create_write_tool_definition};

/// The built-in tool names, upstream's `ToolName` union.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ToolName {
    /// The read tool.
    Read,
    /// The bash tool.
    Bash,
    /// The powershell tool.
    Powershell,
    /// The edit tool.
    Edit,
    /// The write tool.
    Write,
    /// The grep tool.
    Grep,
    /// The find tool.
    Find,
    /// The ls tool.
    Ls,
}

impl ToolName {
    /// The wire name, upstream's union's string values.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Bash => "bash",
            Self::Powershell => "powershell",
            Self::Edit => "edit",
            Self::Write => "write",
            Self::Grep => "grep",
            Self::Find => "find",
            Self::Ls => "ls",
        }
    }

    /// Parse the wire name, upstream's `allToolNames` membership check.
    ///
    /// # Errors
    /// The unknown-name rejection, upstream's `throw new Error`.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "read" => Ok(Self::Read),
            "bash" => Ok(Self::Bash),
            "powershell" => Ok(Self::Powershell),
            "edit" => Ok(Self::Edit),
            "write" => Ok(Self::Write),
            "grep" => Ok(Self::Grep),
            "find" => Ok(Self::Find),
            "ls" => Ok(Self::Ls),
            _ => Err(format!("Unknown tool name: {name}")),
        }
    }

    /// The strict-sampling tools, upstream's `strictToolNames`.
    #[must_use]
    pub const fn is_strict_sampling(self) -> bool {
        matches!(
            self,
            Self::Read | Self::Bash | Self::Powershell | Self::Edit | Self::Write
        )
    }
}

/// The per-tool options bundle, upstream's `ToolsOptions`.
#[derive(Clone, Default)]
pub struct ToolsOptions {
    /// The read tool's options.
    pub read: Option<ReadToolOptions>,
    /// The bash tool's options.
    pub bash: Option<BashToolOptions>,
    /// The powershell tool's options.
    pub powershell: Option<PowerShellToolOptions>,
    /// The write tool's options.
    pub write: Option<WriteToolOptions>,
    /// The edit tool's options.
    pub edit: Option<EditToolOptions>,
    /// The grep tool's options.
    pub grep: Option<GrepToolOptions>,
    /// The find tool's options.
    pub find: Option<FindToolOptions>,
    /// The ls tool's options.
    pub ls: Option<LsToolOptions>,
}

impl std::fmt::Debug for ToolsOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolsOptions")
            .field("read", &self.read.is_some())
            .field("bash", &self.bash.is_some())
            .field("powershell", &self.powershell.is_some())
            .field("write", &self.write.is_some())
            .field("edit", &self.edit.is_some())
            .field("grep", &self.grep.is_some())
            .field("find", &self.find.is_some())
            .field("ls", &self.ls.is_some())
            .finish()
    }
}

/// Build the named tool's definition, upstream's `createToolDefinition`.
///
/// # Errors
/// The unknown-name rejection, upstream's throw.
pub fn create_tool_definition(
    tool_name: ToolName,
    cwd: &str,
    options: Option<&ToolsOptions>,
) -> Result<ToolDefinition, String> {
    let options = options.cloned().unwrap_or_default();
    Ok(match tool_name {
        ToolName::Read => create_read_tool_definition(cwd, options.read),
        ToolName::Bash => create_bash_tool_definition(cwd, options.bash),
        ToolName::Powershell => create_power_shell_tool_definition(cwd, options.powershell),
        ToolName::Edit => create_edit_tool_definition(cwd, options.edit),
        ToolName::Write => create_write_tool_definition(cwd, options.write),
        ToolName::Grep => create_grep_tool_definition(cwd, options.grep),
        ToolName::Find => create_find_tool_definition(cwd, options.find),
        ToolName::Ls => create_ls_tool_definition(cwd, options.ls),
    })
}

/// Build the named tool, upstream's `createTool`.
///
/// # Errors
/// The unknown-name rejection, upstream's throw.
pub fn create_tool(
    tool_name: ToolName,
    cwd: &str,
    options: Option<&ToolsOptions>,
) -> Result<AgentHarnessTool, String> {
    let options = options.cloned().unwrap_or_default();
    Ok(match tool_name {
        ToolName::Read => create_read_tool(cwd, options.read),
        ToolName::Bash => create_bash_tool(cwd, options.bash),
        ToolName::Powershell => create_power_shell_tool(cwd, options.powershell),
        ToolName::Edit => create_edit_tool(cwd, options.edit),
        ToolName::Write => create_write_tool(cwd, options.write),
        ToolName::Grep => create_grep_tool(cwd, options.grep),
        ToolName::Find => create_find_tool(cwd, options.find),
        ToolName::Ls => create_ls_tool(cwd, options.ls),
    })
}

/// The four coding tools' definitions, upstream's
/// `createCodingToolDefinitions`.
#[must_use]
pub fn create_coding_tool_definitions(
    cwd: &str,
    options: Option<&ToolsOptions>,
) -> Vec<ToolDefinition> {
    let options = options.cloned().unwrap_or_default();
    vec![
        create_read_tool_definition(cwd, options.read),
        create_bash_tool_definition(cwd, options.bash),
        create_edit_tool_definition(cwd, options.edit),
        create_write_tool_definition(cwd, options.write),
    ]
}

/// The four read-only tools' definitions, upstream's
/// `createReadOnlyToolDefinitions`.
#[must_use]
pub fn create_read_only_tool_definitions(
    cwd: &str,
    options: Option<&ToolsOptions>,
) -> Vec<ToolDefinition> {
    let options = options.cloned().unwrap_or_default();
    vec![
        create_read_tool_definition(cwd, options.read),
        create_grep_tool_definition(cwd, options.grep),
        create_find_tool_definition(cwd, options.find),
        create_ls_tool_definition(cwd, options.ls),
    ]
}

/// Every tool's definition keyed by name, upstream's
/// `createAllToolDefinitions`.
#[must_use]
pub fn create_all_tool_definitions(
    cwd: &str,
    options: Option<&ToolsOptions>,
) -> BTreeMap<ToolName, ToolDefinition> {
    let options = options.cloned().unwrap_or_default();
    BTreeMap::from([
        (
            ToolName::Read,
            create_read_tool_definition(cwd, options.read),
        ),
        (
            ToolName::Bash,
            create_bash_tool_definition(cwd, options.bash),
        ),
        (
            ToolName::Powershell,
            create_power_shell_tool_definition(cwd, options.powershell),
        ),
        (
            ToolName::Edit,
            create_edit_tool_definition(cwd, options.edit),
        ),
        (
            ToolName::Write,
            create_write_tool_definition(cwd, options.write),
        ),
        (
            ToolName::Grep,
            create_grep_tool_definition(cwd, options.grep),
        ),
        (
            ToolName::Find,
            create_find_tool_definition(cwd, options.find),
        ),
        (ToolName::Ls, create_ls_tool_definition(cwd, options.ls)),
    ])
}

/// The four coding tools, upstream's `createCodingTools`.
#[must_use]
pub fn create_coding_tools(cwd: &str, options: Option<&ToolsOptions>) -> Vec<AgentHarnessTool> {
    let options = options.cloned().unwrap_or_default();
    vec![
        create_read_tool(cwd, options.read),
        create_bash_tool(cwd, options.bash),
        create_edit_tool(cwd, options.edit),
        create_write_tool(cwd, options.write),
    ]
}

/// The four read-only tools, upstream's `createReadOnlyTools`.
#[must_use]
pub fn create_read_only_tools(cwd: &str, options: Option<&ToolsOptions>) -> Vec<AgentHarnessTool> {
    let options = options.cloned().unwrap_or_default();
    vec![
        create_read_tool(cwd, options.read),
        create_grep_tool(cwd, options.grep),
        create_find_tool(cwd, options.find),
        create_ls_tool(cwd, options.ls),
    ]
}

/// Every tool keyed by name, upstream's `createAllTools`.
#[must_use]
pub fn create_all_tools(
    cwd: &str,
    options: Option<&ToolsOptions>,
) -> BTreeMap<ToolName, AgentHarnessTool> {
    let options = options.cloned().unwrap_or_default();
    BTreeMap::from([
        (ToolName::Read, create_read_tool(cwd, options.read)),
        (ToolName::Bash, create_bash_tool(cwd, options.bash)),
        (
            ToolName::Powershell,
            create_power_shell_tool(cwd, options.powershell),
        ),
        (ToolName::Edit, create_edit_tool(cwd, options.edit)),
        (ToolName::Write, create_write_tool(cwd, options.write)),
        (ToolName::Grep, create_grep_tool(cwd, options.grep)),
        (ToolName::Find, create_find_tool(cwd, options.find)),
        (ToolName::Ls, create_ls_tool(cwd, options.ls)),
    ])
}

/// The renderer registry upstream's `renderers/index.ts` carries rides the
/// theme ticket; the barrel keeps the option-bundle types the definitions
/// share.
///
/// The `Arc<Value>` alias documents the schema carrier the strict-mode
/// suite reads.
pub type ToolParameters = Arc<Value>;
