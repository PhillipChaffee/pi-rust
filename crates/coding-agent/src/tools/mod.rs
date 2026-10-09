//! The built-in coding tools, ported from upstream
//! `src/core/tools/` plus `core/exec.ts` and `core/bash-executor.ts`
//! (map child "pi-coding-agent: built-in coding tools").
//!
//! The tools execute over the extension-tool contract the
//! [`crate::extensions`] seam carries: a [`crate::extensions::ToolDefinition`]
//! wraps into a `pi_agent_core` harness tool through
//! [`tool_definition_wrapper::wrap_tool_definition`]. Upstream's renderer
//! slices (`core/tools/renderers/`, `render-utils.ts`) ride the theme
//! ticket's landing (map child "pi-coding-agent: interactive renderers and
//! theme system") — their `ToolDefinition` fields land there.
//!
//! The grep and find tools restate their backing binaries' semantics
//! natively: find's `fd` walk on the `ignore` crate's hierarchical
//! gitignore handling with fd's basename/full-path glob split, grep's `rg`
//! scan on the same walk plus the `regex` crate (ripgrep's own engine), the
//! restatement pinned by the `3302-find-path-glob` and
//! `3303-find-nested-gitignore` regressions.

pub mod bash;
pub mod edit;
pub mod edit_diff;
pub mod file_mutation_queue;
pub mod find;
pub mod grep;
pub mod index;
pub mod ls;
pub mod output_accumulator;
pub mod path_utils;
pub mod powershell;
pub mod read;
pub mod tool_definition_wrapper;
pub mod truncate;
pub mod write;

#[cfg(test)]
mod tests;

/// The boxed error the tools surface, upstream's `throw new Error(message)`.
pub(crate) fn io_error(message: impl Into<String>) -> pi_agent_core::types::AgentToolError {
    Box::new(std::io::Error::other(message.into()))
}

/// The strict-JSON-schema sampling setting every built-in tool declares,
/// upstream's `constrainedSampling` config.
pub(crate) const fn strict_sampling() -> pi_ai::types::ConstrainedSamplingSetting {
    pi_ai::types::ConstrainedSamplingSetting::Config(
        pi_ai::types::ConstrainedSamplingConfig::JsonSchema {
            strict: pi_ai::types::Strictness::Prefer,
        },
    )
}

/// The tool input schema envelope every built-in tool declares, upstream's
/// `bashSchema`-family wrappers (`{ type: "object", properties, required }`).
pub(crate) fn tool_schema(properties: &serde_json::Value, required: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

/// The search tools' shared path resolution, upstream's effective-cwd
/// fallback: the context cwd wins when non-empty, and the requested path
/// (defaulting to the cwd itself) resolves against it.
pub(crate) fn resolve_search_path(
    ctx: Option<&dyn crate::extensions::ExtensionContext>,
    cwd: &str,
    path: Option<&str>,
) -> Result<String, pi_agent_core::types::AgentToolError> {
    let effective_cwd = ctx
        .map(crate::extensions::ExtensionContext::cwd)
        .filter(|ctx_cwd| !ctx_cwd.is_empty())
        .unwrap_or(cwd);
    path_utils::resolve_to_cwd(path.unwrap_or("."), effective_cwd)
        .map_err(|error| io_error(error.to_string()))
}

/// The text-block tool result the tools settle with, upstream's
/// `{ content: [{ type: "text", text }], details }`.
pub(crate) fn text_result(
    output: String,
    details: serde_json::Value,
) -> pi_agent_core::types::AgentToolResult {
    pi_agent_core::types::AgentToolResult {
        content: vec![pi_agent_core::types::AgentToolContent::Text(
            pi_ai::types::TextContent {
                text: output,
                text_signature: None,
            },
        )],
        details,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// The hex form of `len` random bytes, upstream's
/// `randomBytes(len).toString("hex")` at the temp-file and extract-dir
/// naming sites.
pub(crate) fn random_bytes_hex(len: usize) -> String {
    use rand::RngCore;
    use std::fmt::Write as _;

    let mut bytes = vec![0u8; len];
    rand::rng().fill_bytes(&mut bytes);
    let mut hex = String::with_capacity(len * 2);
    for byte in bytes {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}
