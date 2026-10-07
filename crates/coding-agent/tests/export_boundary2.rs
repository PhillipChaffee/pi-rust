//! The export boundary suite, second pass at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the pre-render arms the
//! 1:1 export suite does not reach, the entry-point option seams, the
//! renderer/debug surfaces, and the ANSI tail arms.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::fs;
use std::sync::Arc;

use pi_agent_core::types::{
    AgentMessage, AgentState, AgentTool, AgentToolError, AgentToolResult, BoxedFuture,
    ThinkingLevel,
};
use pi_ai::types::{
    AssistantBlock, ImageContent, KnownApi, Message, Modality, Model, ModelCost, ProviderId,
    TextContent, Tool, ToolCall, ToolResultBlock, ToolResultMessage, UserBlock, UserContent,
    UserMessage,
};
use serde_json::Value as JsonValue;
use tokio_util::sync::CancellationToken;

use pi_coding_agent::export_html::{
    ExportOptions, ExportedTool, NoThemeSource, RenderedToolResult, ToolHtmlRenderer,
    ToolHtmlRendererDeps, ToolRenderContext, ToolRenderDefinition, ansi_to_html, export_from_file,
    export_session_to_html, pre_render_custom_tools,
};
use pi_coding_agent::session_manager::SessionManager;

fn assistant_with_tool_calls(calls: &[(&str, &str)]) -> Message {
    Message::Assistant(pi_ai::types::AssistantMessage {
        content: calls
            .iter()
            .map(|(id, name)| {
                AssistantBlock::ToolCall(ToolCall {
                    id: (*id).to_owned(),
                    name: (*name).to_owned(),
                    arguments: serde_json::Map::new(),
                    thought_signature: None,
                    namespace: None,
                })
            })
            .collect(),
        api: KnownApi::AnthropicMessages.into(),
        provider: ProviderId("anthropic".to_owned()),
        model: "test".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: pi_ai::types::Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 0,
            cost: pi_ai::types::UsageCost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason: pi_ai::types::StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 42,
    })
}
fn tool_result(call_id: &str, name: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: call_id.to_owned(),
        tool_name: name.to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: "done".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 42,
    })
}
/// The stub renderer: every id renders except the `void` ids, upstream's
/// renderer-absent degradation.
struct StubRenderer;
impl ToolHtmlRenderer for StubRenderer {
    fn render_call(
        &mut self,
        tool_call_id: &str,
        _tool_name: &str,
        _args: &serde_json::Map<String, JsonValue>,
    ) -> Option<String> {
        (!tool_call_id.starts_with("void")).then(|| format!(r#"<call id="{tool_call_id}">"#))
    }
    fn render_result(
        &mut self,
        tool_call_id: &str,
        _tool_name: &str,
        _result: &[ToolResultBlock],
        _details: Option<&JsonValue>,
        _is_error: bool,
    ) -> Option<RenderedToolResult> {
        (!tool_call_id.starts_with("void")).then(|| RenderedToolResult {
            collapsed: Some(format!(r"<result-{tool_call_id}>")),
            expanded: Some(format!(r"<expanded-{tool_call_id}>")),
        })
    }
}

struct EmptyRenderer;
impl ToolHtmlRenderer for EmptyRenderer {
    fn render_call(
        &mut self,
        _: &str,
        _: &str,
        _: &serde_json::Map<String, JsonValue>,
    ) -> Option<String> {
        None
    }
    fn render_result(
        &mut self,
        _: &str,
        _: &str,
        _: &[ToolResultBlock],
        _: Option<&JsonValue>,
        _: bool,
    ) -> Option<RenderedToolResult> {
        None
    }
}

fn stub_tool(name: &str) -> AgentTool {
    AgentTool {
        tool: Tool {
            name: name.to_owned(),
            description: "a stub".to_owned(),
            parameters: serde_json::json!({"type": "object"}),
            constrained_sampling: None,
        },
        label: name.to_owned(),
        prepare_arguments: None,
        execute: Arc::new(
            |_id: &str,
             _args: &JsonValue,
             _cancel: Option<&CancellationToken>,
             _update: Option<&(dyn Fn(&AgentToolResult) + Send + Sync)>| {
                never_execute()
            },
        ),
        replay: None,
        execution_mode: None,
    }
}
fn never_execute() -> BoxedFuture<'static, Result<AgentToolResult, AgentToolError>> {
    Box::pin(async { unreachable!("the export never executes tools") })
}
fn stub_state() -> AgentState {
    AgentState {
        system_prompt: "the system prompt".to_owned(),
        model: Model {
            id: "m".to_owned(),
            name: "M".to_owned(),
            api: KnownApi::AnthropicMessages.into(),
            provider: ProviderId("anthropic".to_owned()),
            base_url: "https://example.invalid".to_owned(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![Modality::Text],
            cost: ModelCost::default(),
            context_window: 1000,
            max_tokens: 100,
            sampling_params: None,
            headers: None,
            compat: None,
        },
        thinking_level: ThinkingLevel::Off,
        tools: vec![stub_tool("rich")],
        messages: Vec::new(),
        is_streaming: false,
        streaming_message: None,
        pending_tool_calls: std::collections::BTreeSet::default(),
        error_message: None,
    }
}
fn user_with_blocks() -> Message {
    Message::User(UserMessage {
        content: UserContent::Blocks(vec![
            UserBlock::Text(TextContent {
                text: "hello".to_owned(),
                text_signature: None,
            }),
            UserBlock::Image(ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]),
        timestamp: 42,
    })
}
#[test]
fn pre_render_skips_template_tools_and_merges_result_side_html() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::in_memory(Some(&temp), None, None).expect("in-memory");
    session
        .append_message(AgentMessage::Standard(user_with_blocks()))
        .expect("user");
    session
        .append_message(AgentMessage::Standard(assistant_with_tool_calls(&[
            ("bash-1", "bash"),
            ("custom-1", "custom"),
            ("void-1", "void"),
        ])))
        .expect("assistant");
    session
        .append_custom_message_entry("ext.note", UserContent::Text("note".to_owned()), true, None)
        .expect("custom message");
    session
        .append_message(AgentMessage::Standard(tool_result("custom-1", "custom")))
        .expect("result");
    session
        .append_message(AgentMessage::Standard(tool_result("bash-1", "bash")))
        .expect("result");
    session
        .append_message(AgentMessage::Standard(tool_result("custom-2", "custom2")))
        .expect("result");
    session
        .append_message(AgentMessage::Standard(tool_result("void-2", "void")))
        .expect("result");
    let entries = session.entries();
    let rendered = pre_render_custom_tools(&entries, &mut StubRenderer);
    let keys: Vec<&str> = rendered.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        vec!["custom-1", "custom-2"],
        "template-rendered bash skips, the absent renderer degrades, a result-only call renders"
    );
    let merged = &rendered["custom-1"];
    assert_eq!(merged.call_html.as_deref(), Some(r#"<call id="custom-1">"#));
    assert_eq!(
        merged.result_html_collapsed.as_deref(),
        Some("<result-custom-1>")
    );
    let result_only = &rendered["custom-2"];
    assert!(
        result_only.call_html.is_none(),
        "a result without a rendered call renders alone"
    );
    assert_eq!(
        result_only.result_html_collapsed.as_deref(),
        Some("<result-custom-2>")
    );
}
#[test]
fn the_tui_export_entry_renders_custom_tools_and_carries_the_state() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    session
        .append_message(AgentMessage::Standard(assistant_with_tool_calls(&[(
            "custom-1", "custom",
        )])))
        .expect("assistant");
    session
        .append_message(AgentMessage::Standard(tool_result("custom-1", "custom")))
        .expect("result");
    let state = stub_state();
    let output = format!("{temp}/with-tools.html");
    let options = ExportOptions {
        output_path: Some(output.clone()),
        tool_renderer: Some(Box::new(StubRenderer)),
        ..ExportOptions::default()
    };
    let written =
        export_session_to_html(&session, Some(&state), options, &NoThemeSource).expect("export");
    assert_eq!(written, output);
    let html = fs::read_to_string(&output).expect("read export");
    assert!(html.contains("</html>"), "the template skeleton renders");
    assert!(
        html.len() > 100_000,
        "the vendored assets embed: {}",
        html.len()
    ); // An empty render map degrades to no renderedTools payload, upstream's
    // `Object.keys(renderedTools).length` gate.
    let options = ExportOptions {
        output_path: Some(format!("{temp}/empty-tools.html")),
        tool_renderer: Some(Box::new(EmptyRenderer)),
        ..ExportOptions::default()
    };
    export_session_to_html(&session, None, options, &NoThemeSource)
        .expect("the empty render still exports");
}
#[test]
fn the_standalone_export_reports_upstreams_missing_file_message() {
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = format!("{}/missing.jsonl", dir.path().display());
    let options = ExportOptions::for_path(format!("{}/out.html", dir.path().display()));
    let error = export_from_file(&missing, options, &NoThemeSource).expect_err("missing input");
    assert!(error.to_string().starts_with("File not found: "), "{error}");
}
#[test]
fn the_render_surfaces_debug_as_declarative_summaries() {
    let options = ExportOptions::for_path("out.html");
    let debug = format!("{options:?}");
    assert!(debug.contains("Some(\"out.html\")"), "{debug}");
    assert!(debug.contains("tool_renderer: false"), "{debug}");
    let theme: pi_coding_agent::export_html::ThemeHandle = Arc::new(42u8);
    let mut state = serde_json::Map::new();
    let context = ToolRenderContext {
        args: None,
        tool_call_id: "call-1",
        last_component: None,
        state: &mut state,
        cwd: "/tmp",
        execution_started: true,
        args_complete: true,
        is_partial: false,
        expanded: false,
        show_images: false,
        is_error: false,
        theme: &theme,
    };
    let debug = format!("{context:?}");
    assert!(debug.contains(r#"tool_call_id: "call-1""#), "{debug}");
    let deps = ToolHtmlRendererDeps {
        get_tool_definition: &|_| {
            Some(ToolRenderDefinition {
                render_call: None,
                render_result: None,
            })
        },
        theme: theme.clone(),
        cwd: "/tmp".to_owned(),
        width: 100,
    };
    let debug = format!("{deps:?}");
    assert!(debug.contains(r#"cwd: "/tmp""#), "{debug}");
}
#[test]
fn a_standard_256_index_maps_the_palette_and_an_open_span_closes() {
    assert_eq!(
        ansi_to_html("\u{1b}[38;5;1mred\u{1b}[0m"),
        r#"<span style="color:#800000">red</span>"#,
        "the standard palette indexes answer before the cube"
    );
    assert_eq!(
        ansi_to_html("\u{1b}[31mred"),
        r#"<span style="color:#800000">red</span>"#,
        "text ending inside a span closes it"
    );
} // ---------------------------------------------------------------------------
// The env-dependent arms run as probe children: `set_var` is forbidden in
// this workspace, so the parent composes the child's environment (and its
// working directory, for the default output paths) at spawn time.
// ---------------------------------------------------------------------------/// The probe key the child-process runs key their scenario on.
const EXPORT_PROBE: &str = "PI_CODING_AGENT_EXPORT_PROBE";
/// The fixture root the parent hands the probe child.
const EXPORT_TEMP: &str = "PI_CODING_AGENT_EXPORT_TEMP";
/// Run this suite's own binary as a child whose working directory is the
/// fixture root — the default export paths resolve against it.
fn export_probe(mode: &str, temp: &str) {
    let mut command =
        std::process::Command::new(std::env::current_exe().expect("the test binary path"));
    command
        .args([
            "--exact",
            "the_export_probe",
            "--nocapture",
            "--test-threads=1",
        ])
        .current_dir(temp)
        .env(EXPORT_PROBE, mode)
        .env(EXPORT_TEMP, temp);
    if mode == "share-tmpdir" {
        // A TMPDIR that cannot hold the share temp directory, upstream's
        // mkdtempSync throw.
        command.env("TMPDIR", format!("{temp}/does-not-exist"));
    }
    let output = command.output().expect("the probe child runs");
    assert!(
        output.status.success(),
        "the {mode:?} probe child passes: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
#[test]
fn the_default_export_paths_and_the_share_temp_directory_behave() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    export_probe("export-default-tui", &temp);
    export_probe("export-default-cli", &temp);
    export_probe("share-gist-ok", &temp);
    export_probe("share-tmpdir", &temp);
}
/// The probe child: each mode runs one env-dependent scenario from the
/// fixture root as the process cwd.
#[test]
fn the_export_probe() {
    let Ok(mode) = std::env::var(EXPORT_PROBE) else {
        return;
    };
    let temp = std::env::var(EXPORT_TEMP).expect("the fixture root");
    match mode.as_str() {
        "export-default-tui" => {
            let session_file = format!("{temp}/session.jsonl");
            fs::write(
                &session_file,
                r#"{"type":"session","version":3,"id":"def","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#,
            )
            .expect("write session");
            let session = SessionManager::open(&session_file, None, None).expect("open");
            export_session_to_html(&session, None, ExportOptions::default(), &NoThemeSource)
                .expect("the default path exports");
            assert!(
                fs::exists("pi-session-session.html").unwrap_or(false),
                "the TUI default path lands beside the process cwd"
            );
        }
        "export-default-cli" => {
            let session_file = format!("{temp}/transcript.jsonl");
            fs::write(
                &session_file,
                r#"{"type":"session","version":3,"id":"cli","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#,
            )
            .expect("write session");
            export_from_file(&session_file, ExportOptions::default(), &NoThemeSource)
                .expect("the default path exports");
            assert!(
                fs::exists("pi-session-transcript.html").unwrap_or(false),
                "the CLI default path lands beside the process cwd"
            );
        }
        "share-gist-ok" => {
            let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
            session
                .append_message(AgentMessage::Standard(Message::User(UserMessage {
                    content: UserContent::Text("hello".to_owned()),
                    timestamp: 42,
                })))
                .expect("append");
            let mut source = ProbeSource { session };
            let mut ui = ProbeUi::default();
            let mut runner = ProbeRunner::gist_ok();
            let mut http = ProbeHttp;
            tokio_block_on(pi_coding_agent::session_share::share_session(
                &mut source,
                &mut ui,
                &mut runner,
                &mut http,
            ))
            .expect("the gist flow shares");
            assert!(
                ui.statuses
                    .iter()
                    .any(|status| status.starts_with("Share URL: ")),
                "the share URL status lands: {:?}",
                ui.statuses
            );
        }
        "share-tmpdir" => {
            let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
            session
                .append_message(AgentMessage::Standard(Message::User(UserMessage {
                    content: UserContent::Text("hello".to_owned()),
                    timestamp: 42,
                })))
                .expect("append");
            let mut source = ProbeSource { session };
            let mut ui = ProbeUi::default();
            let mut runner = ProbeRunner::gist_ok();
            let mut http = ProbeHttp;
            let error = tokio_block_on(pi_coding_agent::session_share::share_session(
                &mut source,
                &mut ui,
                &mut runner,
                &mut http,
            ))
            .expect_err("the temp directory cannot exist under the broken TMPDIR");
            assert!(
                error.contains("No such file or directory"),
                "the mkdtemp io failure surfaces outward: {error}"
            );
        }
        other => panic!("unknown probe mode: {other}"),
    }
}
/// The share source stand-in for the probe, upstream's session reads.
struct ProbeSource {
    session: SessionManager,
}
impl pi_coding_agent::session_share::ShareSessionSource for ProbeSource {
    fn session_manager(&self) -> &SessionManager {
        &self.session
    }
    fn system_prompt(&self) -> Option<String> {
        None
    }
    fn tools(&self) -> Vec<ExportedTool> {
        Vec::new()
    }
    fn export_to_html(&mut self, _file_path: &str) -> BoxedFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
    fn has_radius_provider(&self) -> bool {
        false
    }
    fn radius_token(&mut self) -> BoxedFuture<'_, Option<String>> {
        Box::pin(async { None })
    }
}
#[derive(Default)]
struct ProbeUi {
    statuses: Vec<String>,
    errors: Vec<String>,
}
impl pi_coding_agent::session_share::ShareUserInterface for ProbeUi {
    fn show_status(&mut self, message: &str) {
        self.statuses.push(message.to_owned());
    }
    fn show_error(&mut self, message: &str) {
        self.errors.push(message.to_owned());
    }
}
struct ProbeRunner {
    gist_ok: bool,
}
impl ProbeRunner {
    const fn gist_ok() -> Self {
        Self { gist_ok: true }
    }
}
impl pi_coding_agent::session_share::ShareProcessRunner for ProbeRunner {
    fn auth_status(&mut self) -> Option<i32> {
        Some(0)
    }
    fn create_gist(
        &mut self,
        _file_path: &str,
    ) -> BoxedFuture<'_, pi_coding_agent::session_share::GistOutcome> {
        let gist_ok = self.gist_ok;
        Box::pin(async move {
            if gist_ok {
                pi_coding_agent::session_share::GistOutcome {
                    stdout: "https://gist.github.com/test/abc\n".to_owned(),
                    stderr: String::new(),
                    code: Some(0),
                }
            } else {
                pi_coding_agent::session_share::GistOutcome::default()
            }
        })
    }
}
struct ProbeHttp;
impl pi_coding_agent::session_share::ShareHttpClient for ProbeHttp {
    fn upload_artifact(
        &mut self,
        _token: &str,
        _body: &[u8],
    ) -> BoxedFuture<'_, pi_coding_agent::session_share::RadiusUploadOutcome> {
        Box::pin(async { pi_coding_agent::session_share::RadiusUploadOutcome::Aborted })
    }
}
/// The single-threaded runtime the async share call needs.
fn tokio_block_on<T>(future: impl Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(future)
}

#[test]
fn an_in_memory_session_rejects_the_tui_html_export() {
    let session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let error = export_session_to_html(&session, None, ExportOptions::default(), &NoThemeSource)
        .expect_err("in-memory export");
    assert_eq!(error.to_string(), "Cannot export in-memory session to HTML");
}

#[test]
fn writing_the_export_into_a_directory_reports_the_io_arm() {
    let dir = tempfile::tempdir().expect("temp dir");
    let error = pi_coding_agent::export_html::export::write_export(
        "<html>",
        dir.path().display().to_string().as_str(),
    )
    .expect_err("a directory is not a writable export path");
    assert!(
        matches!(error, pi_coding_agent::export_html::ExportError::Io(_)),
        "{error}"
    );
}

#[test]
fn an_html_export_into_a_directory_reports_the_write_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let session_file = format!("{temp}/session.jsonl");
    fs::write(
        &session_file,
        r#"{"type":"session","version":3,"id":"dir","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#,
    )
    .expect("write session");
    let session = SessionManager::open(&session_file, None, None).expect("open");

    // The session-file export takes the output path verbatim.
    let error = export_session_to_html(
        &session,
        None,
        ExportOptions::for_path(temp.clone()),
        &NoThemeSource,
    )
    .expect_err("the output path is a directory");
    assert!(
        matches!(error, pi_coding_agent::export_html::ExportError::Io(_)),
        "{error}"
    );

    // The standalone export takes it the same way.
    let error = export_from_file(
        &session_file,
        ExportOptions::for_path(temp.clone()),
        &NoThemeSource,
    )
    .expect_err("the output path is a directory");
    assert!(
        matches!(error, pi_coding_agent::export_html::ExportError::Io(_)),
        "{error}"
    );
}
