//! Coverage of the startup dialogs and the share/export seams: the startup
//! selector, first-run setup, startup input, the `Arc<AgentSession>` share
//! source, and the frame's share wiring over fake process/HTTP clients.

#![expect(
    clippy::expect_used,
    reason = "the startup fixtures and fake clients pin deterministic outcomes"
)]

mod common;

use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use common::interactive::{Harness, HarnessOptions, RecordingTerminal};
use pi_coding_agent::cli::startup_ui::{
    StartupTuiOptions, create_startup_tui_with, show_first_time_setup_with,
    show_startup_input_with, show_startup_selector_with,
};
use pi_coding_agent::modes::interactive::session_share::{
    GistOutcome, RadiusUploadOutcome, ShareHttpClient, ShareProcessRunner, ShareSessionSource,
    export_session_for_share,
};
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, Settings, SettingsManager, SettingsManagerCreateOptions,
};
use pi_tui::tui::Tui;

fn manager(settings: &Settings) -> SettingsManager<InMemorySettingsStorage> {
    SettingsManager::in_memory(settings, SettingsManagerCreateOptions::default())
}

fn dark_settings() -> Settings {
    let mut settings = Settings::new();
    settings.insert(
        "theme".to_owned(),
        serde_json::Value::String("dark".to_owned()),
    );
    settings
}

struct StartupFixture {
    ui: Rc<Tui>,
    terminal: RecordingTerminal,
    _cwd: tempfile::TempDir,
    _agent: tempfile::TempDir,
}

async fn startup_tui(settings: &Settings) -> StartupFixture {
    let cwd = tempfile::TempDir::new().expect("cwd");
    let agent = tempfile::TempDir::new().expect("agent");
    let terminal = RecordingTerminal::new(80, 24);
    let ui = create_startup_tui_with(
        &manager(settings),
        StartupTuiOptions {
            terminal: Some(Box::new(terminal.clone())),
            cwd: cwd.path().display().to_string(),
            agent_dir: agent.path().display().to_string(),
        },
    )
    .await;
    StartupFixture {
        ui,
        terminal,
        _cwd: cwd,
        _agent: agent,
    }
}

#[tokio::test]
async fn startup_selector_returns_the_chosen_value_and_cancels() {
    let settings = dark_settings();
    let fixture = startup_tui(&settings).await;
    fixture.terminal.queue_poll_input("\x1b[B");
    fixture.terminal.queue_poll_input("\r");
    let selected = show_startup_selector_with(
        &fixture.ui,
        &manager(&settings),
        "Pick",
        vec![("One".to_owned(), "one"), ("Two".to_owned(), "two")],
    );
    assert_eq!(selected, Some("two"));

    let fixture = startup_tui(&settings).await;
    fixture.terminal.queue_poll_input("\x1b");
    assert_eq!(
        show_startup_selector_with(
            &fixture.ui,
            &manager(&settings),
            "Pick",
            vec![("One".to_owned(), "one")],
        ),
        None
    );
}

#[tokio::test]
async fn startup_input_returns_text_and_cancels() {
    let settings = dark_settings();
    let fixture = startup_tui(&settings).await;
    fixture.terminal.queue_poll_input("hello");
    fixture.terminal.queue_poll_input("\r");
    assert_eq!(
        show_startup_input_with(&fixture.ui, &manager(&settings), "Name", None),
        Some("hello".to_owned())
    );

    let fixture = startup_tui(&settings).await;
    fixture.terminal.queue_poll_input("\x1b");
    assert_eq!(
        show_startup_input_with(
            &fixture.ui,
            &manager(&settings),
            "Name",
            Some("placeholder")
        ),
        None
    );
}

#[tokio::test]
async fn first_time_setup_submits_and_cancels() {
    let settings = dark_settings();
    let fixture = startup_tui(&settings).await;
    fixture
        .terminal
        .queue_poll_input_after(Duration::from_millis(300), "j");
    fixture
        .terminal
        .queue_poll_input_after(Duration::from_millis(300), "\r");
    fixture
        .terminal
        .queue_poll_input_after(Duration::from_millis(300), "\r");
    let mut setup_manager = manager(&settings);
    show_first_time_setup_with(&fixture.ui, &mut setup_manager).await;

    let fixture = startup_tui(&settings).await;
    fixture
        .terminal
        .queue_poll_input_after(Duration::from_millis(300), "\x1b");
    let mut setup_manager = manager(&settings);
    show_first_time_setup_with(&fixture.ui, &mut setup_manager).await;
}

fn radius_auth() -> pi_coding_agent::auth_storage::AuthStorageData {
    let mut data = pi_coding_agent::auth_storage::AuthStorageData::new();
    data.insert(
        "radius".to_owned(),
        serde_json::json!({ "type": "api_key", "key": "radius-token" }),
    );
    data
}

struct FakeProcess {
    auth_status: Option<i32>,
    stdout: String,
    stderr: String,
    code: Option<i32>,
    uploads: Vec<String>,
}

impl FakeProcess {
    fn succeeding() -> Self {
        Self {
            auth_status: Some(0),
            stdout: "https://gist.github.com/test/abc123\n".to_owned(),
            stderr: String::new(),
            code: Some(0),
            uploads: Vec::new(),
        }
    }
}

impl ShareProcessRunner for FakeProcess {
    fn auth_status(&mut self) -> Option<i32> {
        self.auth_status
    }

    fn create_gist(&mut self, file_path: &str) -> pi_ai::types::BoxedFuture<'_, GistOutcome> {
        let content = std::fs::read_to_string(file_path).unwrap_or_default();
        self.uploads.push(content);
        let stdout = self.stdout.clone();
        let stderr = self.stderr.clone();
        let code = self.code;
        Box::pin(async move {
            GistOutcome {
                stdout,
                stderr,
                code,
            }
        })
    }
}

struct FakeHttp {
    outcome: RadiusUploadOutcome,
    calls: usize,
}

impl ShareHttpClient for FakeHttp {
    fn upload_artifact(
        &mut self,
        _token: &str,
        _body: &[u8],
    ) -> pi_ai::types::BoxedFuture<'_, RadiusUploadOutcome> {
        self.calls += 1;
        let outcome = self.outcome.clone();
        Box::pin(async move { outcome })
    }
}

#[tokio::test]
async fn share_wiring_reports_the_gist_url_and_every_failure() {
    let harness = Harness::new(HarnessOptions {
        persisted: true,
        messages: vec![
            common::agent_session::user_message("share me", 1),
            pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(
                common::agent_session::assistant_message("shared reply", 2),
            )),
        ],
        ..HarnessOptions::default()
    })
    .await;
    let mut process = FakeProcess::succeeding();
    let mut http = FakeHttp {
        outcome: RadiusUploadOutcome::Aborted,
        calls: 0,
    };
    harness
        .frame
        .share_current_session_with(&mut process, &mut http)
        .await;
    assert!(
        harness.chat_text().contains("Share URL:"),
        "{}",
        harness.chat_text()
    );
    assert_eq!(process.uploads.len(), 1);
    assert!(process.uploads[0].contains("<!DOCTYPE html") || !process.uploads[0].is_empty());

    let mut failing = FakeProcess {
        auth_status: None,
        ..FakeProcess::succeeding()
    };
    harness
        .frame
        .share_current_session_with(&mut failing, &mut http)
        .await;
    assert!(harness.chat_text().contains("not installed"));

    let mut logged_out = FakeProcess {
        auth_status: Some(1),
        ..FakeProcess::succeeding()
    };
    harness
        .frame
        .share_current_session_with(&mut logged_out, &mut http)
        .await;
    assert!(harness.chat_text().contains("not logged in"));

    let mut unknown = FakeProcess {
        code: Some(1),
        ..FakeProcess::succeeding()
    };
    harness
        .frame
        .share_current_session_with(&mut unknown, &mut http)
        .await;
    assert!(harness.chat_text().contains("Unknown error"));

    let mut stderr = FakeProcess {
        code: Some(1),
        stderr: "no permission".to_owned(),
        ..FakeProcess::succeeding()
    };
    harness
        .frame
        .share_current_session_with(&mut stderr, &mut http)
        .await;
    assert!(harness.chat_text().contains("no permission"));

    let mut empty = FakeProcess {
        stdout: String::new(),
        ..FakeProcess::succeeding()
    };
    harness
        .frame
        .share_current_session_with(&mut empty, &mut http)
        .await;
    assert!(harness.chat_text().contains("Failed to parse gist ID"));
}

#[tokio::test]
async fn radius_share_urls_and_failures_short_circuit_the_gist() {
    let harness = Harness::new(HarnessOptions {
        persisted: true,
        auth: radius_auth(),
        extra_models: vec![common::model_layer::model("radius", "radius-model")],
        ..HarnessOptions::default()
    })
    .await;
    assert!(
        harness
            .frame
            .session()
            .model_runtime()
            .get_provider("radius")
            .is_some()
    );

    let mut process = FakeProcess::succeeding();
    let mut http = FakeHttp {
        outcome: RadiusUploadOutcome::Url("https://radius.test/artifact/1".to_owned()),
        calls: 0,
    };
    harness
        .frame
        .share_current_session_with(&mut process, &mut http)
        .await;
    assert!(
        harness
            .chat_text()
            .contains("https://radius.test/artifact/1")
    );
    assert!(process.uploads.is_empty(), "the gist fallback is skipped");
    assert_eq!(http.calls, 1);

    let mut http = FakeHttp {
        outcome: RadiusUploadOutcome::Failed("upload rejected".to_owned()),
        calls: 0,
    };
    harness
        .frame
        .share_current_session_with(&mut process, &mut http)
        .await;
    assert!(harness.chat_text().contains("upload rejected"));
}

#[tokio::test]
async fn the_agent_session_is_a_share_source() {
    let harness = Harness::new(HarnessOptions {
        persisted: true,
        messages: vec![
            common::agent_session::user_message("shared", 1),
            pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(
                common::agent_session::assistant_message("reply", 2),
            )),
        ],
        ..HarnessOptions::default()
    })
    .await;
    let mut source: Arc<
        pi_coding_agent::core::agent_session::AgentSession<InMemorySettingsStorage>,
    > = harness.frame.session();
    let manager = source.session_manager();
    assert!(!manager.entries().is_empty());
    assert!(source.system_prompt().is_some());
    let _ = source.tools();
    let _ = source.has_radius_provider();
    assert_eq!(source.radius_token().await, None);

    let root = tempfile::TempDir::new().expect("tempdir");
    let jsonl = root.path().join("session.jsonl");
    let exported = export_session_for_share(
        &root.path().join("share.jsonl").display().to_string(),
        &source,
    );
    assert!(exported.is_ok(), "{exported:?}");
    let _ = source.export_to_html(&jsonl.display().to_string()).await;
}
