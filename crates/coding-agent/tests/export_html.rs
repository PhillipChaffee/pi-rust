//! The export-HTML suites at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`:
//! `export-html-xss.test.ts` and `export-html-skill-block.test.ts` port 1:1
//! as static-source assertions over the embedded template assets, and
//! `export-html-whitespace.test.ts` ports its behavior checks over the ANSI
//! converter and the tool renderer.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::fs;
use std::sync::Arc;

use pi_tui::tui::Component;
use serde_json::json;

use pi_ai::types::{
    AssistantBlock, AssistantMessage, KnownApi, Message, ProviderId, StopReason, Usage, UsageCost,
};
use pi_coding_agent::export_html::assets::{TEMPLATE_CSS, TEMPLATE_JS};
use pi_coding_agent::export_html::{
    CreatedToolHtmlRenderer, ExportedTool, NoThemeSource, SessionData, ToolHtmlRenderer,
    ToolHtmlRendererDeps, ToolRenderDefinition, ansi_lines_to_html, export_from_file,
    export_session_to_html, generate_html,
};
use pi_coding_agent::session_manager::SessionManager;

fn matches(pattern: &str, haystack: &str) -> bool {
    regex::Regex::new(pattern)
        .expect("pattern")
        .is_match(haystack)
}

fn matches_case_insensitive(pattern: &str, haystack: &str) -> bool {
    regex::Regex::new(&format!("(?i){pattern}"))
        .expect("pattern")
        .is_match(haystack)
}

// ---------------------------------------------------------------------------
// export-html-xss.test.ts: markdown link sanitization.
// ---------------------------------------------------------------------------

#[test]
fn overrides_the_marked_link_renderer_to_use_scheme_allow_list_sanitization() {
    assert!(matches(r"link\s*\(\s*token\s*\)", TEMPLATE_JS));
    assert!(matches(r"sanitizeMarkdownUrl\(token\.href\)", TEMPLATE_JS));
    assert!(matches(r"\^\(https\?\|mailto\|tel\|ftp\)", TEMPLATE_JS));
}

#[test]
fn overrides_the_marked_image_renderer_to_use_scheme_allow_list_sanitization() {
    assert!(matches(r"image\s*\(\s*token\s*\)", TEMPLATE_JS));
    assert!(matches(r"sanitizeMarkdownUrl\(token\.href\)", TEMPLATE_JS));
}

#[test]
fn strips_c0_controls_before_checking_and_emitting_markdown_urls() {
    assert!(TEMPLATE_JS.contains(r"replace(/[\x00-\x1f\x7f]/g, '')"));
    assert!(!matches_case_insensitive(
        r"\^\\s\*\(javascript\|vbscript\|data\):",
        TEMPLATE_JS
    ));
}

#[test]
fn escapes_href_attributes_in_the_custom_link_renderer() {
    assert!(matches(r"escapeHtml\(href\)", TEMPLATE_JS));
}

#[test]
fn escapes_image_mime_type_and_data_attributes() {
    assert!(!matches(r"\$\{img\.mimeType\}", TEMPLATE_JS));
    assert!(matches(r"escapeHtml\(img\.mimeType", TEMPLATE_JS));
    assert!(!matches(r#";base64,\$\{img\.data\}""#, TEMPLATE_JS));
    assert!(matches(
        r#";base64,\$\{escapeHtml\(img\.data \|\| (?:''|"")\)\}""#,
        TEMPLATE_JS
    ));
}

#[test]
fn escapes_entry_ids_before_inserting_them_into_attributes() {
    assert!(!matches(r#"id="\$\{entryId\}""#, TEMPLATE_JS));
    assert!(!matches(r#"data-entry-id="\$\{entryId\}""#, TEMPLATE_JS));
    assert!(matches(r"entry-\$\{escapeHtml\(entry\.id\)\}", TEMPLATE_JS));
    assert!(matches(
        r#"data-entry-id="\$\{escapeHtml\(entryId\)\}""#,
        TEMPLATE_JS
    ));
}

#[test]
fn escapes_tree_metadata_rendered_from_session_fields() {
    assert!(!matches(
        r"\[\$\{msg\.toolName \|\| 'tool'\}\]",
        TEMPLATE_JS
    ));
    assert!(!matches(r"\[\$\{msg\.role\}\]", TEMPLATE_JS));
    assert!(!matches(r"\[model: \$\{entry\.modelId\}\]", TEMPLATE_JS));
    assert!(!matches(
        r"\[thinking: \$\{entry\.thinkingLevel\}\]",
        TEMPLATE_JS
    ));
    assert!(!matches(r"\[\$\{entry\.type\}\]", TEMPLATE_JS));
    assert!(matches(
        r"\$\{escapeHtml\(msg\.toolName \|\| 'tool'\)\}",
        TEMPLATE_JS
    ));
    assert!(matches(r"\$\{escapeHtml\(msg\.role\)\}", TEMPLATE_JS));
    assert!(matches(r"\$\{escapeHtml\(entry\.modelId\)\}", TEMPLATE_JS));
    assert!(matches(
        r"\$\{escapeHtml\(entry\.thinkingLevel\)\}",
        TEMPLATE_JS
    ));
    assert!(matches(r"\$\{escapeHtml\(entry\.type\)\}", TEMPLATE_JS));
}

#[test]
fn escapes_model_names_in_the_exported_header() {
    assert!(!matches(
        r"\$\{globalStats\.models\.join\(', '\) \|\| 'unknown'\}",
        TEMPLATE_JS
    ));
    assert!(matches(
        r"\$\{escapeHtml\(globalStats\.models\.join\(', '\) \|\| 'unknown'\)\}",
        TEMPLATE_JS
    ));
}

// ---------------------------------------------------------------------------
// export-html-skill-block.test.ts: skill wrapper rendering.
// ---------------------------------------------------------------------------

#[test]
fn strips_skill_wrapper_xml_from_user_message_rendering() {
    assert!(matches(r"parseSkillBlock", TEMPLATE_JS));
    assert!(matches(r"skillBlock\.userMessage", TEMPLATE_JS));
}

#[test]
fn renders_skill_invocation_and_user_message_as_separate_sibling_blocks() {
    assert!(matches(r"skill-invocation", TEMPLATE_JS));
    assert!(matches(r"hasUserContent", TEMPLATE_JS));
}

#[test]
fn renders_skill_content_as_markdown_not_raw_text() {
    assert!(matches(
        r"safeMarkedParse\(skillBlock\.content\)",
        TEMPLATE_JS
    ));
}

#[test]
fn shows_skill_name_and_user_message_in_the_sidebar_tree() {
    assert!(matches(r"tree-role-skill", TEMPLATE_JS));
}

// ---------------------------------------------------------------------------
// export-html-whitespace.test.ts: whitespace preservation.
// ---------------------------------------------------------------------------

#[test]
fn preserves_whitespace_for_plain_text_tool_output_lines() {
    assert!(matches(
        r"\.output-preview > div:not\(\.expand-hint\),\s*\.output-full > div:not\(\.expand-hint\) \{(?s).*?white-space:\s*pre-wrap;",
        TEMPLATE_CSS
    ));
    assert!(matches(
        r"\.ansi-line\s*\{(?s).*?white-space:\s*pre;",
        TEMPLATE_CSS
    ));
    assert!(!matches(
        r"\.output-preview,\s*\.output-full\s*\{(?s).*?white-space:\s*pre-wrap;",
        TEMPLATE_CSS
    ));
}

#[test]
fn does_not_insert_source_whitespace_between_ansi_rendered_lines() {
    assert_eq!(
        ansi_lines_to_html(&["one".to_owned(), "two".to_owned()]),
        "<div class=\"ansi-line\">one</div><div class=\"ansi-line\">two</div>"
    );
}

/// The plain component the trim test drives, upstream's `{ render: () =>
/// [...], invalidate: () => {} }` literal.
struct FixedComponent(Vec<String>);

impl Component for FixedComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.0.clone()
    }
}

/// The plain renderer the trim test drives: every lookup yields a result
/// hook returning the fixed component.
fn fixed_renderer(lines: Vec<String>) -> CreatedToolHtmlRenderer<'static> {
    let lines: &'static Vec<String> = Box::leak(Box::new(lines));
    let lookup: &'static (dyn Fn(&str) -> Option<ToolRenderDefinition> + 'static) =
        &*Box::leak(Box::new(move |name: &str| {
            if name == "custom" {
                let lines: &'static Vec<String> = lines;
                Some(ToolRenderDefinition {
                    render_call: None,
                    render_result: Some(Box::new(move |_result, _options, _context| {
                        Box::new(FixedComponent(lines.clone()))
                    })),
                })
            } else {
                None
            }
        }));
    CreatedToolHtmlRenderer::new(ToolHtmlRendererDeps {
        get_tool_definition: lookup,
        theme: Arc::new(()),
        cwd: "/tmp".to_owned(),
        width: 100,
    })
}

#[test]
fn trims_tui_spacing_lines_from_custom_tool_result_html() {
    let mut tool_renderer = fixed_renderer(vec![
        String::new(),
        "\u{1b}[31mred\u{1b}[0m".to_owned(),
        "two".to_owned(),
        String::new(),
    ]);
    let rendered = tool_renderer
        .render_result("id", "custom", &[], None, false)
        .expect("rendered");
    assert_eq!(
        rendered.expanded.expect("expanded"),
        "<div class=\"ansi-line\"><span style=\"color:#800000\">red</span></div><div class=\"ansi-line\">two</div>"
    );
    assert_eq!(
        rendered.collapsed, None,
        "collapsed equals expanded and drops"
    );
}

// ---------------------------------------------------------------------------
// generateHtml boundaries: template substitution and theme fallbacks.
// ---------------------------------------------------------------------------

fn minimal_session_data(session: &SessionManager) -> SessionData<'_> {
    SessionData {
        header: session.get_header(),
        entries: session.entries(),
        leaf_id: session.get_leaf_id(),
        system_prompt: None,
        tools: Some(vec![ExportedTool {
            name: "demo".to_owned(),
            description: "demo tool".to_owned(),
            parameters: json!({"type": "object"}),
        }]),
        rendered_tools: None,
    }
}

#[test]
fn the_placeholder_pipeline_reproduces_the_js_replace_artifacts() {
    let session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let html = generate_html(&minimal_session_data(&session), None, &NoThemeSource).expect("html");

    // highlight.min.js's `$&` rewrites to the matched placeholder text
    // inside the operator char class (upstream's actual emitted bytes).
    assert!(
        html.contains("[-+*\\/?!{{HIGHLIGHT_JS}}|:<=>@^~]"),
        "the $& substitution artifact is present"
    );
    // highlight.min.js's `\\$${ze}+` collapses to `\\${ze}+`.
    assert!(
        html.contains("\\\\${ze}+"),
        "the $$ collapse inside hljs: {html}"
    );
    // template.js's `$${totalCost...` collapses to `${totalCost...`.
    assert!(html.contains("${totalCost.toFixed(3)"));
    assert!(
        !html.contains("$${totalCost"),
        "the literal $$ does not survive"
    );
}

#[test]
fn the_no_theme_source_drives_the_derived_fallback_colors() {
    let session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let html = generate_html(&minimal_session_data(&session), None, &NoThemeSource).expect("html");

    // userMessageBg falls back to #343541; the dark-derived exports are
    // upstream's parse/luminance math at that base.
    assert!(html.contains("--exportPageBg: rgb(36, 37, 46);"), "{html}");
    assert!(html.contains("--exportCardBg: rgb(44, 45, 55);"));
    assert!(html.contains("--exportInfoBg: rgb(72, 68, 65);"));
}

#[test]
fn the_session_payload_rides_base64_in_the_script_tag() {
    let session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let html = generate_html(&minimal_session_data(&session), None, &NoThemeSource).expect("html");
    let marker = "<script id=\"session-data\" type=\"application/json\">";
    let start = html.find(marker).expect("script tag") + marker.len();
    let end = html[start..].find("</script>").expect("close") + start;
    let payload = &html[start..end];
    let decoded = base64_decode(payload);
    let data: serde_json::Value = serde_json::from_slice(&decoded).expect("json");
    assert_eq!(data["header"]["type"], "session");
    assert_eq!(data["tools"][0]["name"], "demo");
    assert_eq!(data["leafId"], serde_json::Value::Null);
}

fn base64_decode(payload: &str) -> Vec<u8> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .expect("base64")
}

// ---------------------------------------------------------------------------
// The export entry points.
// ---------------------------------------------------------------------------

#[test]
fn export_from_file_writes_the_default_named_output() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let session_path = format!("{temp}/2026-01-01T00-00-00-000Z_abc.jsonl");
    fs::write(
        &session_path,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"export-me\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"{temp}\"}}\n"
        ),
    )
    .expect("write");

    let output = export_from_file(
        &session_path,
        pi_coding_agent::export_html::ExportOptions::default(),
        &NoThemeSource,
    )
    .expect("export");
    // Upstream writes the relative default name into the process cwd.
    assert_eq!(output, "pi-session-2026-01-01T00-00-00-000Z_abc.html");
    let html = fs::read_to_string(&output).expect("read");
    assert!(html.contains("<!DOCTYPE html>"));
    fs::remove_file(&output).expect("cleanup");
}

#[test]
fn export_session_to_html_requires_a_persisted_session() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let in_memory = SessionManager::in_memory(None, None, None).expect("in-memory");
    let error = export_session_to_html(
        &in_memory,
        None,
        pi_coding_agent::export_html::ExportOptions::for_path(&temp),
        &NoThemeSource,
    )
    .expect_err("in-memory export fails");
    assert_eq!(error.to_string(), "Cannot export in-memory session to HTML");

    let created = SessionManager::create(&temp, Some(&temp), None).expect("create");
    let error = export_session_to_html(
        &created,
        None,
        pi_coding_agent::export_html::ExportOptions::for_path(&temp),
        &NoThemeSource,
    )
    .expect_err("nothing persisted yet");
    assert_eq!(
        error.to_string(),
        "Nothing to export yet - start a conversation first"
    );
}

#[test]
fn export_session_to_html_writes_the_state_and_rendered_tools() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    session
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            Message::Assistant(AssistantMessage {
                content: vec![AssistantBlock::ToolCall(pi_ai::types::ToolCall {
                    id: "call-1".to_owned(),
                    name: "custom-thing".to_owned(),
                    arguments: serde_json::Map::new(),
                    thought_signature: None,
                    namespace: None,
                })],
                api: KnownApi::AnthropicMessages.into(),
                provider: ProviderId("anthropic".to_owned()),
                model: "test".to_owned(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                diagnostics: None,
                usage: Usage {
                    input: 0,
                    output: 0,
                    cache_read: 0,
                    cache_write: 0,
                    cache_write_1h: None,
                    reasoning: None,
                    total_tokens: 0,
                    cost: UsageCost {
                        input: 0.0,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                        total: 0.0,
                    },
                },
                stop_reason: StopReason::ToolUse,
                deferred: None,
                error_message: None,
                raw_stop_reason: None,
                end_turn: None,
                timestamp: 0,
            }),
        ))
        .expect("append");
    let session_file = session.session_file().expect("file").to_owned();

    // A custom renderer for the non-template tool.
    let lookup: &'static (dyn Fn(&str) -> Option<ToolRenderDefinition> + 'static) =
        &*Box::leak(Box::new(move |_name: &str| {
            Some(ToolRenderDefinition {
                render_call: Some(Box::new(|_context| {
                    Box::new(FixedComponent(vec![
                        "\u{1b}[32mrendered\u{1b}[0m".to_owned(),
                    ]))
                })),
                render_result: None,
            })
        }));
    let options = pi_coding_agent::export_html::ExportOptions {
        output_path: Some(format!("{temp}/out.html")),
        theme_name: None,
        tool_renderer: Some(Box::new(CreatedToolHtmlRenderer::new(
            ToolHtmlRendererDeps {
                get_tool_definition: lookup,
                theme: Arc::new(()),
                cwd: temp.clone(),
                width: 100,
            },
        ))),
    };
    let output = export_session_to_html(&session, None, options, &NoThemeSource).expect("export");
    assert_eq!(output, format!("{temp}/out.html"));
    let html = fs::read_to_string(&output).expect("read");
    let marker = "<script id=\"session-data\" type=\"application/json\">";
    let start = html.find(marker).expect("script tag") + marker.len();
    let end = html[start..].find("</script>").expect("close") + start;
    let decoded = base64_decode(&html[start..end]);
    let data: serde_json::Value = serde_json::from_slice(&decoded).expect("json");
    assert_eq!(
        data["renderedTools"]["call-1"]["callHtml"],
        "<div class=\"ansi-line\"><span style=\"color:#008000\">rendered</span></div>",
        "the custom tool pre-renders through its TUI renderer"
    );
    let _ = session_file;
}
