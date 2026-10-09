//! The `tool-system-prompt-contributions`, `builtin-tool-strict-mode`
//! (the portable describes), and `powershell-tool` suites.
//!
//! The strict-mode suite's third describe — the extension re-registration
//! loop over `DefaultResourceLoader` / `createAgentSession` — rides the
//! extension-system and AgentSession tickets (map children "pi-coding-agent:
//! extension system" and "pi-coding-agent: AgentSession core"); its
//! observable contract (definitions carry `constrainedSampling`, the wrap
//! preserves explicit opt-outs) binds here. The powershell suite's win32
//! row rides the map's Windows ruling.

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::sync::Arc;

use pi_ai::types::{ConstrainedSamplingConfig, ConstrainedSamplingSetting, Strictness};
use serde_json::json;

use crate::extensions::types::{CwdContext, ToolDefinition};
use crate::tools::bash::{
    BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION, BashExecOptions, create_bash_tool_definition,
};
use crate::tools::edit::{EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_edit_tool_definition};
use crate::tools::find::{FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_find_tool_definition};
use crate::tools::grep::{GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_grep_tool_definition};
use crate::tools::index::{ToolName, create_all_tool_definitions, create_all_tools};
use crate::tools::ls::{LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_ls_tool_definition};
use crate::tools::powershell::create_local_power_shell_operations;
use crate::tools::powershell::{
    POWERSHELL_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_power_shell_tool_definition,
};

use super::helpers::block_on;
use crate::tools::read::{READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_read_tool_definition};
use crate::tools::tool_definition_wrapper::wrap_tool_definition;
use crate::tools::write::{WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_write_tool_definition};
use crate::utils::shell::POWERSHELL_ARGS;

/// The `{ type: "json_schema", strict: "prefer" }` setting the strict
/// tools carry.
fn strict_prefer() -> ConstrainedSamplingSetting {
    ConstrainedSamplingSetting::Config(ConstrainedSamplingConfig::JsonSchema {
        strict: Strictness::Prefer,
    })
}

// ---------------------------------------------------------------------------
// tool-system-prompt-contributions
// ---------------------------------------------------------------------------

#[test]
fn keeps_the_read_tool_definition_aligned_with_its_contribution() {
    let definition = create_read_tool_definition("/workspace", None);
    assert_eq!(
        definition.prompt_snippet.as_deref(),
        Some(READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet)
    );
    assert_eq!(
        definition.prompt_guidelines.as_deref().unwrap_or_default(),
        READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines
    );
}

#[test]
fn keeps_the_bash_tool_definition_aligned_with_its_contribution() {
    let definition = create_bash_tool_definition("/workspace", None);
    assert_eq!(
        definition.prompt_snippet.as_deref(),
        Some(BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet)
    );
    assert_eq!(
        definition.prompt_guidelines.as_deref().unwrap_or_default(),
        BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines
    );
}

#[test]
fn keeps_the_powershell_tool_definition_aligned_with_its_contribution() {
    let definition = create_power_shell_tool_definition("/workspace", None);
    assert_eq!(
        definition.prompt_snippet.as_deref(),
        Some(POWERSHELL_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet)
    );
    assert_eq!(
        definition.prompt_guidelines.as_deref().unwrap_or_default(),
        POWERSHELL_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines
    );
}

#[test]
fn keeps_the_edit_tool_definition_aligned_with_its_contribution() {
    let definition = create_edit_tool_definition("/workspace", None);
    assert_eq!(
        definition.prompt_snippet.as_deref(),
        Some(EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet)
    );
    assert_eq!(
        definition.prompt_guidelines.as_deref().unwrap_or_default(),
        EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines
    );
}

#[test]
fn keeps_the_write_tool_definition_aligned_with_its_contribution() {
    let definition = create_write_tool_definition("/workspace", None);
    assert_eq!(
        definition.prompt_snippet.as_deref(),
        Some(WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet)
    );
    assert_eq!(
        definition.prompt_guidelines.as_deref().unwrap_or_default(),
        WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines
    );
}

#[test]
fn keeps_the_grep_tool_definition_aligned_with_its_contribution() {
    let definition = create_grep_tool_definition("/workspace", None);
    assert_eq!(
        definition.prompt_snippet.as_deref(),
        Some(GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet)
    );
    assert_eq!(
        definition.prompt_guidelines.as_deref().unwrap_or_default(),
        GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines
    );
}

#[test]
fn keeps_the_find_tool_definition_aligned_with_its_contribution() {
    let definition = create_find_tool_definition("/workspace", None);
    assert_eq!(
        definition.prompt_snippet.as_deref(),
        Some(FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet)
    );
    assert_eq!(
        definition.prompt_guidelines.as_deref().unwrap_or_default(),
        FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines
    );
}

#[test]
fn keeps_the_ls_tool_definition_aligned_with_its_contribution() {
    let definition = create_ls_tool_definition("/workspace", None);
    assert_eq!(
        definition.prompt_snippet.as_deref(),
        Some(LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet)
    );
    assert_eq!(
        definition.prompt_guidelines.as_deref().unwrap_or_default(),
        LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines
    );
}

#[test]
fn keeps_bash_session_environment_guidance_conditional() {
    let definition = create_bash_tool_definition(
        "/workspace",
        Some(crate::tools::bash::BashToolOptions {
            expose_session_environment: Some(false),
            ..crate::tools::bash::BashToolOptions::default()
        }),
    );
    assert!(definition.prompt_guidelines.is_none());
}

#[test]
fn keeps_powershell_session_environment_guidance_conditional() {
    let definition = create_power_shell_tool_definition(
        "/workspace",
        Some(crate::tools::powershell::PowerShellToolOptions {
            expose_session_environment: Some(false),
            ..crate::tools::powershell::PowerShellToolOptions::default()
        }),
    );
    assert!(definition.prompt_guidelines.is_none());
}

// ---------------------------------------------------------------------------
// builtin-tool-strict-mode (portable describes)
// ---------------------------------------------------------------------------

#[test]
fn prefers_strict_sampling_regardless_of_the_experimental_flag() {
    // Upstream parametrizes over PI_EXPERIMENTAL=undefined|0|1; the
    // definitions never read the env, so the strict preference pins once.
    let definitions = create_all_tool_definitions(".", None);
    let tools = create_all_tools(".", None);
    for name in [
        ToolName::Read,
        ToolName::Bash,
        ToolName::Powershell,
        ToolName::Edit,
        ToolName::Write,
    ] {
        let expected = Some(strict_prefer());
        assert_eq!(
            definitions[&name].constrained_sampling, expected,
            "{name:?}"
        );
        assert_eq!(tools[&name].tool.constrained_sampling, expected, "{name:?}");
    }
    for name in [ToolName::Grep, ToolName::Find, ToolName::Ls] {
        assert_eq!(definitions[&name].constrained_sampling, None, "{name:?}");
    }
    // Strictness is a provider-side conversion, not a change to the
    // execution schema.
    assert_eq!(
        definitions[&ToolName::Read].parameters["required"],
        json!(["path"])
    );
    assert_eq!(
        definitions[&ToolName::Bash].parameters["required"],
        json!(["command"])
    );
}

#[test]
fn preserves_explicit_opt_outs_when_wrapping_definitions_for_execution() {
    let definitions = create_all_tool_definitions(".", None);
    for name in [
        ToolName::Read,
        ToolName::Bash,
        ToolName::Powershell,
        ToolName::Edit,
        ToolName::Write,
    ] {
        let definition = &definitions[&name];
        let mut override_definition = definition.clone();
        override_definition.constrained_sampling =
            Some(ConstrainedSamplingSetting::Disabled(false));
        let wrapped = wrap_tool_definition::<CwdContext>(override_definition.clone(), None);
        assert_eq!(
            wrapped.tool.constrained_sampling,
            Some(ConstrainedSamplingSetting::Disabled(false))
        );
        // The override carries the definition's own fields, upstream's
        // reference-identity spread assertions.
        assert!(Arc::ptr_eq(
            &override_definition.execute,
            &definition.execute
        ));
        assert_eq!(override_definition.parameters, definition.parameters);
        assert_eq!(
            override_definition.prompt_guidelines,
            definition.prompt_guidelines
        );
        assert_eq!(definition.constrained_sampling, Some(strict_prefer()));
    }
}

// ---------------------------------------------------------------------------
// powershell-tool
// ---------------------------------------------------------------------------

#[test]
fn uses_process_local_execution_policy_bypass() {
    assert_eq!(
        POWERSHELL_ARGS,
        [
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command"
        ]
    );
}

/// The definition the opt-out test clones; the alias keeps the helper
/// imports honest.
#[expect(
    dead_code,
    reason = "the strict-mode suite's re-registration describe rides the extension-system ticket"
)]
fn sample_definition() -> ToolDefinition {
    create_bash_tool_definition(".", None)
}

#[test]
fn the_local_powershell_operations_throw_off_windows() {
    block_on(async {
        let operations = create_local_power_shell_operations();
        let outcome = (operations.exec)(
            "Get-ChildItem",
            ".",
            BashExecOptions {
                on_data: Arc::new(|_data: &[u8]| {}),
                signal: None,
                timeout: None,
                env: None,
            },
        )
        .await;
        let error = outcome.unwrap_err();
        assert!(
            error.to_string().contains("only available on Windows"),
            "{error}"
        );
    });
}
