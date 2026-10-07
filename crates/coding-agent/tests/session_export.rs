//! The JSONL export suite at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`:
//! the branch re-chaining and default naming of `exportSessionToJsonl`, plus
//! the `export-jsonl-share.test.ts` shape assertions restated over a stub
//! share source (the full `AgentSession` wiring rides #125).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::fs;
use std::sync::{Arc, Mutex};

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{
    AssistantBlock, AssistantMessage, KnownApi, Message, ProviderId, StopReason, TextContent,
    ToolCall, ToolResultBlock, ToolResultMessage, Usage, UsageCost, UserContent, UserMessage,
};

use pi_coding_agent::export_html::ExportedTool;
use pi_coding_agent::session_export::export_session_to_jsonl;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::session_share::{ShareSessionSource, export_session_for_share};

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_millis()
        .try_into()
        .expect("epoch millis fit i64")
}

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: now_millis(),
    }))
}

const fn usage() -> Usage {
    Usage {
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
    }
}

fn tool_call_assistant() -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: vec![AssistantBlock::ToolCall(ToolCall {
            id: "call-1".to_owned(),
            name: "share_tool".to_owned(),
            arguments: serde_json::json!({"value": "example"})
                .as_object()
                .expect("object")
                .clone(),
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
        usage: usage(),
        stop_reason: StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_millis(),
    }))
}

fn tool_result() -> AgentMessage {
    AgentMessage::Standard(Message::ToolResult(ToolResultMessage {
        tool_call_id: "call-1".to_owned(),
        tool_name: "share_tool".to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: "done".to_owned(),
            text_signature: None,
        })],
        details: Some(serde_json::json!({})),
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: now_millis(),
    }))
}

fn read_records(path: &str) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .expect("read export")
        .lines()
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect()
}

#[test]
fn export_rechains_the_branch_into_a_linear_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::in_memory(Some(&temp), None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    let second = session
        .append_message(user_message("again"))
        .expect("append");
    session.branch(&first).expect("branch");
    let third = session
        .append_message(user_message("side"))
        .expect("append");

    let output = export_session_to_jsonl(&session, None, None).expect("export");
    // Upstream's `resolvePath` returns the default name under the process cwd.
    assert!(
        std::path::Path::new(&output)
            .extension()
            .is_some_and(|extension| extension == "jsonl")
            && output.contains("session-"),
        "the default name: {output}"
    );
    assert!(
        output.starts_with(
            std::env::current_dir()
                .expect("cwd")
                .display()
                .to_string()
                .as_str()
        )
    );
    assert!(fs::exists(&output).unwrap_or(false));
    let output_name = output.clone();

    let records = read_records(&output);
    assert_eq!(records.len(), 3, "header + the leaf path (two entries)");
    assert_eq!(records[0]["type"], "session");
    assert_eq!(records[0]["id"], session.session_id());
    assert_eq!(records[0]["cwd"], temp);
    assert!(
        records[0].get("parentSession").is_none(),
        "the export carries no parent link"
    );
    assert_eq!(records[1]["id"], first);
    assert_eq!(
        records[1]["parentId"],
        serde_json::Value::Null,
        "the path re-chains from the root"
    );
    assert_eq!(
        records[2]["id"], third,
        "the side branch rides the export; the old leaf does not"
    );
    assert_eq!(records[2]["parentId"], first);
    let _ = second;
    fs::remove_file(&output_name).expect("cleanup");
}

#[test]
fn trailing_entries_append_after_the_branch_with_the_tail_parent() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::in_memory(Some(&temp), None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    let output_path = format!("{temp}/nested/out.jsonl");

    let output = export_session_to_jsonl(
        &session,
        Some(&output_path),
        Some(&|parent_id, timestamp| {
            assert!(!timestamp.is_empty());
            vec![serde_json::json!({
                "type": "custom",
                "customType": "trailing",
                "parentId": parent_id,
            })]
        }),
    )
    .expect("export");
    assert_eq!(output, output_path, "the explicit path wins");
    let records = read_records(&output);
    assert_eq!(records.len(), 3);
    assert_eq!(records[2]["customType"], "trailing");
    assert_eq!(
        records[2]["parentId"], first,
        "the trailing entry parents onto the branch tail"
    );
    assert!(
        fs::exists(format!("{temp}/nested")).unwrap_or(false),
        "the parent directory is created"
    );
}

/// The share source stand-in, upstream's `session.state` + manager reads.
struct StubSource {
    session: SessionManager,
    system_prompt: Option<String>,
    tools: Vec<ExportedTool>,
}

impl ShareSessionSource for StubSource {
    fn session_manager(&self) -> &SessionManager {
        &self.session
    }

    fn system_prompt(&self) -> Option<String> {
        self.system_prompt.clone()
    }

    fn tools(&self) -> Vec<ExportedTool> {
        self.tools.clone()
    }

    fn export_to_html(
        &mut self,
        _file_path: &str,
    ) -> pi_agent_core::types::BoxedFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }

    fn has_radius_provider(&self) -> bool {
        false
    }

    fn radius_token(&mut self) -> pi_agent_core::types::BoxedFuture<'_, Option<String>> {
        Box::pin(async { None })
    }
}

#[test]
fn the_share_export_adds_presentation_data_without_changing_conversation_links() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session_manager =
        SessionManager::in_memory(Some(&temp), None, None).expect("in-memory");
    let user_id = session_manager
        .append_message(user_message("hello"))
        .expect("append");
    let assistant_id = session_manager
        .append_message(tool_call_assistant())
        .expect("append");
    let result_id = session_manager
        .append_message(tool_result())
        .expect("append");
    let original_entry_ids: Vec<String> = session_manager
        .get_branch(None)
        .iter()
        .filter_map(|entry| entry.entry_id())
        .map(str::to_owned)
        .collect();
    assert_eq!(
        original_entry_ids,
        vec![user_id.clone(), assistant_id.clone(), result_id.clone()]
    );

    let source = StubSource {
        session: session_manager,
        system_prompt: Some("the system prompt".to_owned()),
        tools: vec![ExportedTool {
            name: "share_tool".to_owned(),
            description: "Render a value for sharing".to_owned(),
            parameters: serde_json::json!({"type": "object"}),
        }],
    };

    // The plain export carries no pi.share entry.
    let normal_path = format!("{temp}/normal.jsonl");
    export_session_to_jsonl(&source.session, Some(&normal_path), None).expect("export");
    let normal_records = read_records(&normal_path);
    assert!(
        !normal_records
            .iter()
            .any(|record| record["type"] == "custom" && record["customType"] == "pi.share")
    );

    let share_path = format!("{temp}/share.jsonl");
    export_session_for_share(&share_path, &source).expect("share export");
    let records = read_records(&share_path);
    let conversation_records = &records[1..records.len() - 1];
    let ids: Vec<String> = conversation_records
        .iter()
        .map(|record| record["id"].as_str().expect("id").to_owned())
        .collect();
    assert_eq!(ids, original_entry_ids, "conversation ids are unchanged");
    let parent_ids: Vec<serde_json::Value> = conversation_records
        .iter()
        .map(|record| record["parentId"].clone())
        .collect();
    let expected_parents: Vec<serde_json::Value> = std::iter::once(serde_json::Value::Null)
        .chain(
            original_entry_ids[..original_entry_ids.len() - 1]
                .iter()
                .map(|id| serde_json::json!(id)),
        )
        .collect();
    assert_eq!(parent_ids, expected_parents);
    let last_three = &conversation_records[conversation_records.len() - 3..];
    assert_eq!(
        last_three
            .iter()
            .map(|record| record["id"].as_str().expect("id"))
            .collect::<Vec<_>>(),
        vec![user_id.as_str(), assistant_id.as_str(), result_id.as_str()],
        "upstream's slice(-3) keeps branch order"
    );

    let share_entry = records.last().expect("share entry");
    assert_eq!(share_entry["type"], "custom");
    assert_eq!(share_entry["customType"], "pi.share");
    assert_eq!(share_entry["parentId"], result_id);
    assert!(share_entry["timestamp"].is_string());
    assert_eq!(share_entry["data"]["systemPrompt"], "the system prompt");
    assert_eq!(share_entry["data"]["tools"][0]["name"], "share_tool");
    assert_eq!(
        share_entry["data"]["tools"][0]["description"],
        "Render a value for sharing"
    );
    assert!(share_entry["data"].get("renderedTools").is_none());
    assert!(share_entry["data"].get("theme").is_none());
    assert!(share_entry["data"].get("version").is_none());

    // The share file re-imports as a working session.
    let imported = SessionManager::open(&share_path, None, None).expect("import");
    let leaf_id = imported.get_leaf_id().expect("leaf").to_owned();
    assert_eq!(leaf_id, share_entry["id"].as_str().expect("id"));
    let roles: Vec<String> = imported
        .build_session_context()
        .messages
        .iter()
        .map(|message| match message {
            AgentMessage::Standard(Message::User(_)) => "user".to_owned(),
            AgentMessage::Standard(Message::Assistant(_)) => "assistant".to_owned(),
            AgentMessage::Standard(Message::ToolResult(_)) => "toolResult".to_owned(),
            AgentMessage::Custom(custom) => custom.role.clone(),
        })
        .collect();
    assert_eq!(roles, vec!["user", "assistant", "toolResult"]);
}

#[test]
fn a_system_prompt_absent_from_the_state_omits_the_field() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session_manager =
        SessionManager::in_memory(Some(&temp), None, None).expect("in-memory");
    session_manager
        .append_message(user_message("hello"))
        .expect("append");
    let source = StubSource {
        session: session_manager,
        system_prompt: None,
        tools: Vec::new(),
    };
    let share_path = format!("{temp}/share.jsonl");
    export_session_for_share(&share_path, &source).expect("share export");
    let records = read_records(&share_path);
    let share_entry = records.last().expect("share entry");
    assert!(
        share_entry["data"].get("systemPrompt").is_none(),
        "absent prompts drop the key"
    );
    assert_eq!(share_entry["data"]["tools"], serde_json::json!([]));
}

fn make_runner(uploads: &Uploads, auth_status: Option<i32>) -> MockRunner {
    MockRunner {
        uploads: Arc::clone(uploads),
        auth_status,
        failing_gist: false,
    }
}

/// Shared state the mock runner captures, upstream's `uploads` list.
type Uploads = Arc<Mutex<Vec<String>>>;

/// The mock gist runner, upstream's `vi.mock("node:child_process")`.
struct MockRunner {
    uploads: Uploads,
    auth_status: Option<i32>,
    /// When set, the gist child fails with an empty stderr.
    failing_gist: bool,
}

impl pi_coding_agent::session_share::ShareProcessRunner for MockRunner {
    fn auth_status(&mut self) -> Option<i32> {
        self.auth_status
    }

    fn create_gist(
        &mut self,
        file_path: &str,
    ) -> pi_agent_core::types::BoxedFuture<'_, pi_coding_agent::session_share::GistOutcome> {
        let uploads = Arc::clone(&self.uploads);
        let file_path = file_path.to_owned();
        let failing_gist = self.failing_gist;
        Box::pin(async move {
            uploads
                .lock()
                .expect("uploads")
                .push(fs::read_to_string(&file_path).expect("the html file"));
            if failing_gist {
                return pi_coding_agent::session_share::GistOutcome {
                    stdout: String::new(),
                    stderr: String::new(),
                    code: Some(1),
                };
            }
            pi_coding_agent::session_share::GistOutcome {
                stdout: format!(
                    "https://gist.github.com/test/{}\n",
                    uploads.lock().expect("uploads").len()
                ),
                stderr: String::new(),
                code: Some(0),
            }
        })
    }
}

struct NoHttp;

impl pi_coding_agent::session_share::ShareHttpClient for NoHttp {
    fn upload_artifact(
        &mut self,
        _token: &str,
        _body: &[u8],
    ) -> pi_agent_core::types::BoxedFuture<'_, pi_coding_agent::session_share::RadiusUploadOutcome>
    {
        Box::pin(async {
            pi_coding_agent::session_share::RadiusUploadOutcome::Failed("unused".to_owned())
        })
    }
}

/// The UI mock capturing statuses and errors, upstream's `showStatus` /
/// `showError` context methods.
#[derive(Default)]
struct CapturingUi {
    statuses: Vec<String>,
    errors: Vec<String>,
}

impl pi_coding_agent::session_share::ShareUserInterface for CapturingUi {
    fn show_status(&mut self, message: &str) {
        self.statuses.push(message.to_owned());
    }

    fn show_error(&mut self, message: &str) {
        self.errors.push(message.to_owned());
    }
}

/// The session mock whose HTML export writes its name, upstream's
/// `exportToHtml: async (filePath) => writeFileSync(filePath, name)`.
struct NamedSource {
    session: SessionManager,
    name: &'static str,
}

impl ShareSessionSource for NamedSource {
    fn session_manager(&self) -> &SessionManager {
        &self.session
    }

    fn system_prompt(&self) -> Option<String> {
        Some(self.name.to_owned())
    }

    fn tools(&self) -> Vec<ExportedTool> {
        Vec::new()
    }

    fn export_to_html(
        &mut self,
        file_path: &str,
    ) -> pi_agent_core::types::BoxedFuture<'_, Result<(), String>> {
        let name = self.name;
        let file_path = file_path.to_owned();
        Box::pin(async move {
            fs::write(file_path, name).expect("write html");
            Ok(())
        })
    }

    fn has_radius_provider(&self) -> bool {
        false
    }

    fn radius_token(&mut self) -> pi_agent_core::types::BoxedFuture<'_, Option<String>> {
        Box::pin(async { None })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_session_exports_stay_isolated() {
    let uploads: Uploads = Arc::new(Mutex::new(Vec::new()));
    let mut source_a = NamedSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        name: "A",
    };
    let mut source_b = NamedSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        name: "B",
    };
    let mut ui_a = CapturingUi::default();
    let mut ui_b = CapturingUi::default();
    let mut runner_a = make_runner(&uploads, Some(0));
    let mut runner_b = make_runner(&uploads, Some(0));
    let mut http_a = NoHttp;
    let mut http_b = NoHttp;

    // Both shares run concurrently on one runtime; each exports into its own
    // exclusive temp directory, so the two gist uploads carry their own
    // session's HTML.
    let (a, b) = tokio::join!(
        pi_coding_agent::session_share::share_session(
            &mut source_a,
            &mut ui_a,
            &mut runner_a,
            &mut http_a
        ),
        pi_coding_agent::session_share::share_session(
            &mut source_b,
            &mut ui_b,
            &mut runner_b,
            &mut http_b
        ),
    );
    a.expect("share a");
    b.expect("share b");

    let uploads = uploads.lock().expect("uploads").clone();
    assert_eq!(uploads, vec!["A", "B"], "each upload reads its own export");
    assert!(
        ui_a.errors.is_empty() && ui_b.errors.is_empty(),
        "no errors: {:?}",
        (&ui_a.errors, &ui_b.errors)
    );
    assert_eq!(ui_a.statuses.len(), 1, "the share URL status lands");
    assert!(ui_a.statuses[0].starts_with("Share URL: "));
}

#[tokio::test(flavor = "current_thread")]
async fn a_missing_github_cli_reports_upstreams_messages() {
    let uploads: Uploads = Arc::new(Mutex::new(Vec::new()));
    let mut source = NamedSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        name: "A",
    };
    let mut ui = CapturingUi::default();
    let mut runner_missing = make_runner(&uploads, None);
    let mut http = NoHttp;
    pi_coding_agent::session_share::share_session(
        &mut source,
        &mut ui,
        &mut runner_missing,
        &mut http,
    )
    .await
    .expect("share");
    assert_eq!(
        ui.errors,
        vec!["GitHub CLI (gh) is not installed. Install it from https://cli.github.com/"]
    );

    let mut ui = CapturingUi::default();
    let mut runner_logged_out = make_runner(&uploads, Some(1));
    pi_coding_agent::session_share::share_session(
        &mut source,
        &mut ui,
        &mut runner_logged_out,
        &mut http,
    )
    .await
    .expect("share");
    assert_eq!(
        ui.errors,
        vec!["GitHub CLI is not logged in. Run 'gh auth login' first."]
    );
    assert!(
        uploads.lock().expect("uploads").is_empty(),
        "no gist without auth"
    );
}

/// The Radius-configured source whose token read comes back absent, upstream's
/// `getAuthCredential` returning nothing (the credential surface rides
/// #121/#129).
struct RadiusNoTokenSource {
    session: SessionManager,
}

impl ShareSessionSource for RadiusNoTokenSource {
    fn session_manager(&self) -> &SessionManager {
        &self.session
    }

    fn system_prompt(&self) -> Option<String> {
        Some("A".to_owned())
    }

    fn tools(&self) -> Vec<ExportedTool> {
        Vec::new()
    }

    fn export_to_html(
        &mut self,
        file_path: &str,
    ) -> pi_agent_core::types::BoxedFuture<'_, Result<(), String>> {
        let file_path = file_path.to_owned();
        Box::pin(async move {
            fs::write(file_path, "A").expect("write html");
            Ok(())
        })
    }

    fn has_radius_provider(&self) -> bool {
        true
    }

    fn radius_token(&mut self) -> pi_agent_core::types::BoxedFuture<'_, Option<String>> {
        Box::pin(async { None })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn an_absent_radius_token_falls_through_to_the_gist() {
    let uploads: Uploads = Arc::new(Mutex::new(Vec::new()));
    let mut source = RadiusNoTokenSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
    };
    let mut ui = CapturingUi::default();
    let mut runner = make_runner(&uploads, Some(0));
    let mut http = NoHttp;
    pi_coding_agent::session_share::share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");
    assert_eq!(
        uploads.lock().expect("uploads").len(),
        1,
        "the gist fallback runs"
    );
    assert!(ui.errors.is_empty(), "{:?}", ui.errors);
    assert_eq!(ui.statuses.len(), 1, "the share URL status lands");
}

#[test]
fn raw_entries_re_chain_their_parent_links_in_the_export() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let path = format!("{temp}/raw.jsonl");
    fs::write(
        &path,
        concat!(
            r#"{"type":"session","version":3,"id":"raw","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#, "\n",
            r#"{"type":"custom","id":"e1","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","customType":"root"}"#, "\n",
            r#"{"type":"mystery","id":"m1","parentId":"e1","timestamp":"2026-01-01T00:00:01.000Z","value":1}"#, "\n",
        ),
    )
    .expect("write raw file");
    let session = SessionManager::open(&path, None, None).expect("open");
    let output_path = format!("{temp}/export.jsonl");
    export_session_to_jsonl(&session, Some(&output_path), None).expect("export");

    let records = read_records(&output_path);
    assert_eq!(records.len(), 3, "header + the typed entry + the raw value");
    assert_eq!(
        records[2]["type"], "mystery",
        "the raw value rides the export"
    );
    assert_eq!(
        records[2]["parentId"], "e1",
        "the raw value re-chains onto the branch"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_gist_failure_without_stderr_reports_upstreams_unknown_error() {
    let uploads: Uploads = Arc::new(Mutex::new(Vec::new()));
    let mut source = NamedSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
        name: "A",
    };
    let mut ui = CapturingUi::default();
    let mut runner = make_runner(&uploads, Some(0));
    runner.failing_gist = true;
    let mut http = NoHttp;
    pi_coding_agent::session_share::share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");
    assert_eq!(
        ui.errors,
        vec!["Failed to create gist: Unknown error"],
        "an empty stderr names the unknown error"
    );
}

#[test]
fn an_export_under_a_file_path_reports_the_directory_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let blocker = format!("{temp}/blocker");
    fs::write(&blocker, "a file, not a directory").expect("write blocker");
    let mut session = SessionManager::in_memory(Some(&temp), None, None).expect("in-memory");
    session
        .append_message(user_message("hello"))
        .expect("append");

    let error = export_session_to_jsonl(&session, Some(&format!("{blocker}/out.jsonl")), None)
        .expect_err("the parent is a file");
    assert!(
        matches!(
            error,
            pi_coding_agent::session_manager::SessionManagerError::Io(_)
        ),
        "{error}"
    );
}

/// The Radius-configured source whose token resolves, upstream's
/// `getAuthCredential` returning a token.
struct RadiusTokenSource {
    session: SessionManager,
}

impl ShareSessionSource for RadiusTokenSource {
    fn session_manager(&self) -> &SessionManager {
        &self.session
    }

    fn system_prompt(&self) -> Option<String> {
        Some("A".to_owned())
    }

    fn tools(&self) -> Vec<ExportedTool> {
        Vec::new()
    }

    fn export_to_html(
        &mut self,
        file_path: &str,
    ) -> pi_agent_core::types::BoxedFuture<'_, Result<(), String>> {
        let file_path = file_path.to_owned();
        Box::pin(async move {
            fs::write(file_path, "A").expect("write html");
            Ok(())
        })
    }

    fn has_radius_provider(&self) -> bool {
        true
    }

    fn radius_token(&mut self) -> pi_agent_core::types::BoxedFuture<'_, Option<String>> {
        Box::pin(async { Some("token".to_owned()) })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn a_failing_radius_upload_reports_upstreams_message() {
    let uploads: Uploads = Arc::new(Mutex::new(Vec::new()));
    let mut source = RadiusTokenSource {
        session: SessionManager::in_memory(None, None, None).expect("in-memory"),
    };
    let mut ui = CapturingUi::default();
    let mut runner = make_runner(&uploads, Some(0));
    let mut http = NoHttp;
    pi_coding_agent::session_share::share_session(&mut source, &mut ui, &mut runner, &mut http)
        .await
        .expect("share");
    assert_eq!(
        ui.errors,
        vec!["Failed to upload Radius artifact: unused"],
        "the failed upload settles the flow and skips the gist"
    );
    assert!(
        uploads.lock().expect("uploads").is_empty(),
        "no gist upload after the radius failure"
    );
}
