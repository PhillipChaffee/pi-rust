//! The powershell tool, ported from upstream `src/core/tools/powershell.ts`
//! over its posix path.
//!
//! The Windows pwsh discovery, the win32 branches of the shell resolver,
//! and the win32-gated suite row ride the map's Windows ruling ("Decide the
//! Rust stack"): the reachable contract everywhere this effort targets is
//! the resolver's non-Windows throw, and the constant argv the tool
//! declares pins its invocation shape.

use std::sync::Arc;

use pi_agent_core::harness::types::AgentHarnessTool;

use crate::extensions::types::ToolDefinition;
use crate::utils::shell::get_power_shell_config;

use super::bash::{
    BashExecOptions, BashOperations, BashSpawnContext, BashSpawnHook, BashToolDetails,
    BashToolInput, BashToolOptions, ShellToolConfig, SystemPromptContribution,
    create_shell_tool_definition,
};
use super::tool_definition_wrapper::wrap_cwd_tool;

/// The UTF-8 output preamble every powershell command prepends, upstream's
/// `UTF8_OUTPUT_PREFIX`.
const UTF8_OUTPUT_PREFIX: &str =
    "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n";

/// The powershell tool's system-prompt contribution, upstream's
/// `powershellToolSystemPromptContribution`.
pub const POWERSHELL_TOOL_SYSTEM_PROMPT_CONTRIBUTION: SystemPromptContribution =
    SystemPromptContribution {
        snippet: "Execute PowerShell commands",
        guidelines: &[
            "You can inspect PI_* environment variables for current model and session details.",
        ],
    };

/// The powershell operations reuse the bash seam's shapes, upstream's type
/// aliases.
pub type PowerShellOperations = BashOperations;
/// The spawn context the powershell hook adjusts, upstream's alias.
pub type PowerShellSpawnContext = BashSpawnContext;
/// The spawn hook, upstream's alias.
pub type PowerShellSpawnHook = BashSpawnHook;
/// The structured details, upstream's alias.
pub type PowerShellToolDetails = BashToolDetails;
/// The tool input, upstream's alias.
pub type PowerShellToolInput = BashToolInput;

/// The powershell tool's options, upstream's `PowerShellToolOptions`
/// (the `Pick<BashToolOptions, ...>` slice).
#[derive(Clone, Default)]
pub struct PowerShellToolOptions {
    /// Custom operations; default, the local PowerShell resolver.
    pub operations: Option<BashOperations>,
    /// Expose the current session's `PI_*` metadata; default true.
    pub expose_session_environment: Option<bool>,
    /// The hook adjusting the spawn before execution.
    pub spawn_hook: Option<BashSpawnHook>,
}

impl std::fmt::Debug for PowerShellToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PowerShellToolOptions")
            .field("operations", &self.operations.is_some())
            .field(
                "expose_session_environment",
                &self.expose_session_environment,
            )
            .field("spawn_hook", &self.spawn_hook.is_some())
            .finish()
    }
}

/// The local PowerShell operations, upstream's
/// `createLocalPowerShellOperations`: the shared shell body over the
/// PowerShell resolver, with the UTF-8 preamble prepended to every
/// command.
#[must_use]
pub fn create_local_power_shell_operations() -> PowerShellOperations {
    let operations =
        super::bash::create_local_shell_operations("PowerShell", get_power_shell_config);
    let exec = operations.exec.clone();
    PowerShellOperations {
        exec: Arc::new(move |command: &str, cwd: &str, options: BashExecOptions| {
            let exec = Arc::clone(&exec);
            let prefixed = format!("{UTF8_OUTPUT_PREFIX}{command}");
            let cwd = cwd.to_owned();
            Box::pin(async move { (exec)(&prefixed, &cwd, options).await })
        }),
    }
}

const POWERSHELL_TOOL_CONFIG: ShellToolConfig = ShellToolConfig {
    name: "powershell",
    label: "powershell",
    shell_name: "PowerShell",
    prompt: "PS>",
    prompt_snippet: POWERSHELL_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet,
    prompt_guidelines: POWERSHELL_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines,
    temp_file_prefix: "pi-powershell",
};

/// Build the powershell tool definition, upstream's
/// `createPowerShellToolDefinition`.
#[must_use]
pub fn create_power_shell_tool_definition(
    cwd: &str,
    options: Option<PowerShellToolOptions>,
) -> ToolDefinition {
    let options = options.unwrap_or_default();
    create_shell_tool_definition(
        cwd,
        POWERSHELL_TOOL_CONFIG,
        Some(BashToolOptions {
            operations: Some(
                options
                    .operations
                    .unwrap_or_else(create_local_power_shell_operations),
            ),
            expose_session_environment: options.expose_session_environment,
            spawn_hook: options.spawn_hook,
            ..BashToolOptions::default()
        }),
    )
}

/// Build the powershell tool, upstream's `createPowerShellTool`.
#[must_use]
pub fn create_power_shell_tool(
    cwd: &str,
    options: Option<PowerShellToolOptions>,
) -> AgentHarnessTool {
    let definition = create_power_shell_tool_definition(cwd, options);
    wrap_cwd_tool(definition)
}
