//! The HTML generation and export entry points, upstream's
//! `src/core/export-html/index.ts` core at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::fs;
use std::path::Path;

use serde::Serialize;
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;

use base64::Engine as _;

use pi_agent_core::types::AgentState;
use pi_ai::types::{AssistantBlock, Message};

use crate::config::APP_NAME;
use crate::session_manager::{FileEntry, SessionHeader, SessionManager};

use super::{
    ExportError, ExportOptions, ThemeColorsSource, assets, derive_export_colors,
    generate_theme_vars, js_replace,
};

/// HTML a custom tool renderer produced for one tool call, upstream's
/// `RenderedToolHtml`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderedToolHtml {
    /// The call-side HTML.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_html: Option<String>,
    /// The collapsed result HTML.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_html_collapsed: Option<String>,
    /// The expanded result HTML.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_html_expanded: Option<String>,
}

/// One exported tool spec, upstream's `Pick<ToolDefinition, "name" |
/// "description" | "parameters">`.
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "the parameter schema rides serde_json::Value, which is PartialEq-only"
)]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportedTool {
    /// The tool name.
    pub name: String,
    /// The tool description.
    pub description: String,
    /// The parameter JSON Schema document.
    pub parameters: JsonValue,
}

/// The session payload the client template renders, upstream's `SessionData`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionData<'a> {
    /// The session header.
    pub header: Option<&'a SessionHeader>,
    /// All session entries.
    pub entries: Vec<&'a FileEntry>,
    /// The current leaf id.
    pub leaf_id: Option<&'a str>,
    /// The system prompt, when the exporting state carries one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<&'a str>,
    /// The exported tool specs, when the exporting state carries tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ExportedTool>>,
    /// Pre-rendered HTML for custom tool calls/results, keyed by tool call id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rendered_tools: Option<BTreeMap<String, RenderedToolHtml>>,
}

/// Tools rendered directly by the HTML template (not pre-rendered via the
/// TUI→ANSI→HTML pipeline), upstream's `TEMPLATE_RENDERED_TOOLS`.
const TEMPLATE_RENDERED_TOOLS: [&str; 5] = ["bash", "read", "write", "edit", "ls"];

/// Pre-render custom tools to HTML using their TUI renderers, upstream's
/// `preRenderCustomTools`.
#[must_use]
pub fn pre_render_custom_tools(
    entries: &[&FileEntry],
    tool_renderer: &mut dyn super::ToolHtmlRenderer,
) -> BTreeMap<String, RenderedToolHtml> {
    use crate::session_manager::SessionEntry;

    let mut rendered_tools: BTreeMap<String, RenderedToolHtml> = BTreeMap::new();

    for entry in entries {
        let FileEntry::Entry(SessionEntry::Message(message_entry)) = entry else {
            continue;
        };
        match message_entry.message.as_ref() {
            Some(pi_agent_core::types::AgentMessage::Standard(Message::Assistant(assistant))) => {
                // Find tool calls in assistant messages.
                for block in &assistant.content {
                    if let AssistantBlock::ToolCall(tool_call) = block {
                        if TEMPLATE_RENDERED_TOOLS.contains(&tool_call.name.as_str()) {
                            continue;
                        }
                        if let Some(call_html) = tool_renderer.render_call(
                            &tool_call.id,
                            &tool_call.name,
                            &tool_call.arguments,
                        ) {
                            rendered_tools.insert(
                                tool_call.id.clone(),
                                RenderedToolHtml {
                                    call_html: Some(call_html),
                                    ..RenderedToolHtml::default()
                                },
                            );
                        }
                    }
                }
            }
            Some(pi_agent_core::types::AgentMessage::Standard(Message::ToolResult(result))) => {
                // Only render if we have a pre-rendered call OR it's not
                // template-rendered.
                let tool_call_id = &result.tool_call_id;
                let tool_name = result.tool_name.as_str();
                let existing = rendered_tools.get(tool_call_id);
                if (existing.is_some() || !TEMPLATE_RENDERED_TOOLS.contains(&tool_name))
                    && let Some(rendered) = tool_renderer.render_result(
                        tool_call_id,
                        tool_name,
                        &result.content,
                        result.details.as_ref(),
                        result.is_error,
                    )
                {
                    let mut merged = existing.cloned().unwrap_or_default();
                    merged.result_html_collapsed = rendered.collapsed;
                    merged.result_html_expanded = rendered.expanded;
                    rendered_tools.insert(tool_call_id.clone(), merged);
                }
            }
            _ => {}
        }
    }

    rendered_tools
}

/// Core HTML generation logic shared by both export functions, upstream's
/// `generateHtml`.
///
/// # Errors
/// [`ExportError::Io`] when a template asset fails to embed (impossible for
/// the include-carrying build) — the upstream readFileSync surface, kept for
/// the signature parity the callers lean on.
pub fn generate_html(
    session_data: &SessionData<'_>,
    theme_name: Option<&str>,
    theme_source: &dyn ThemeColorsSource,
) -> Result<String, ExportError> {
    let template = assets::TEMPLATE_HTML;
    let template_css = assets::TEMPLATE_CSS;
    let template_js = assets::TEMPLATE_JS;
    let marked_js = assets::MARKED_JS;
    let hljs_js = assets::HIGHLIGHT_JS;

    let theme_vars = generate_theme_vars(theme_name, theme_source);
    let colors = theme_source.resolved_colors(theme_name);
    let theme_export = theme_source.export_colors(theme_name);
    let derived_export_colors = derive_export_colors(
        colors
            .get("userMessageBg")
            .map_or("#343541", String::as_str),
    );
    let body_bg = theme_export
        .page_bg
        .unwrap_or_else(|| derived_export_colors.page_bg.clone());
    let container_bg = theme_export
        .card_bg
        .unwrap_or_else(|| derived_export_colors.card_bg.clone());
    let info_bg = theme_export
        .info_bg
        .unwrap_or_else(|| derived_export_colors.info_bg.clone());

    // Base64 encode session data to avoid escaping issues.
    let session_json = serde_json::to_string(session_data).map_err(ExportError::from)?;
    let session_data_base64 =
        base64::engine::general_purpose::STANDARD.encode(session_json.as_bytes());

    // Build the CSS with theme variables injected.
    let css = js_replace(template_css, "{{THEME_VARS}}", &theme_vars);
    let css = js_replace(&css, "{{BODY_BG}}", &body_bg);
    let css = js_replace(&css, "{{CONTAINER_BG}}", &container_bg);
    let css = js_replace(&css, "{{INFO_BG}}", &info_bg);

    let html = js_replace(template, "{{CSS}}", &css);
    let html = js_replace(&html, "{{JS}}", template_js);
    let html = js_replace(&html, "{{SESSION_DATA}}", &session_data_base64);
    let html = js_replace(&html, "{{MARKED_JS}}", marked_js);
    let html = js_replace(&html, "{{HIGHLIGHT_JS}}", hljs_js);
    Ok(html)
}

/// The default export output path for a session file, upstream's
/// `pi-session-${basename}.html` default.
fn default_output_path(session_file: &str) -> String {
    let basename = Path::new(session_file).file_name().map_or_else(
        || session_file.to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    let stem = basename.strip_suffix(".jsonl").unwrap_or(&basename);
    format!("{APP_NAME}-session-{stem}.html")
}

/// Write the generated HTML, upstream's `writeFileSync` tail. Returns the
/// output path.
///
/// # Errors
/// [`ExportError::Io`] when the file cannot be written.
pub fn write_export(html: &str, output_path: &str) -> Result<String, ExportError> {
    fs::write(output_path, html)?;
    Ok(output_path.to_owned())
}

/// Export session to HTML using SessionManager and AgentState, upstream's
/// `exportSessionToHtml` (the TUI's `/export` command entry).
///
/// # Errors
/// [`ExportError::Message`] with upstream's throws for an in-memory session
/// ("`Cannot export in-memory session to HTML`") and for a session that has
/// nothing persisted yet ("`Nothing to export yet - start a conversation
/// first`").
pub fn export_session_to_html(
    session_manager: &SessionManager,
    state: Option<&AgentState>,
    options: ExportOptions,
    theme_source: &dyn ThemeColorsSource,
) -> Result<String, ExportError> {
    let Some(session_file) = session_manager.session_file() else {
        return Err(ExportError::Message(
            "Cannot export in-memory session to HTML".to_owned(),
        ));
    };
    if !Path::new(session_file).exists() {
        return Err(ExportError::Message(
            "Nothing to export yet - start a conversation first".to_owned(),
        ));
    }

    let entries = session_manager.entries();

    // Pre-render custom tools if a tool renderer is provided.
    let mut rendered_tools: Option<BTreeMap<String, RenderedToolHtml>> = None;
    if let Some(mut tool_renderer) = options.tool_renderer {
        let rendered = pre_render_custom_tools(&entries, tool_renderer.as_mut());
        if !rendered.is_empty() {
            rendered_tools = Some(rendered);
        }
    }

    let session_data = SessionData {
        header: session_manager.get_header(),
        entries,
        leaf_id: session_manager.get_leaf_id(),
        system_prompt: state.map(|state| state.system_prompt.as_str()),
        tools: state.map(|state| {
            state
                .tools
                .iter()
                .map(|tool| ExportedTool {
                    name: tool.tool.name.clone(),
                    description: tool.tool.description.clone(),
                    parameters: tool.tool.parameters.clone(),
                })
                .collect()
        }),
        rendered_tools,
    };

    let html = generate_html(&session_data, options.theme_name.as_deref(), theme_source)?;

    let output_path = options
        .output_path
        .unwrap_or_else(|| default_output_path(session_file));
    write_export(&html, &output_path)
}

/// Export session file to HTML (standalone, without AgentState), upstream's
/// `exportFromFile` (the CLI entry for arbitrary session files).
///
/// # Errors
/// [`ExportError::Message`] with upstream's throw for a missing input file
/// ("`File not found: <path>`").
pub fn export_from_file(
    input_path: &str,
    options: ExportOptions,
    theme_source: &dyn ThemeColorsSource,
) -> Result<String, ExportError> {
    let resolved_input_path = crate::session_manager::resolve_here(input_path);
    if !Path::new(&resolved_input_path).exists() {
        return Err(ExportError::Message(format!(
            "File not found: {resolved_input_path}"
        )));
    }

    let session_manager = SessionManager::open(&resolved_input_path, None, None)?;

    let session_data = SessionData {
        header: session_manager.get_header(),
        entries: session_manager.entries(),
        leaf_id: session_manager.get_leaf_id(),
        system_prompt: None,
        tools: None,
        rendered_tools: None,
    };

    let html = generate_html(&session_data, options.theme_name.as_deref(), theme_source)?;

    let output_path = options
        .output_path
        .unwrap_or_else(|| default_output_path(&resolved_input_path));
    write_export(&html, &output_path)
}
