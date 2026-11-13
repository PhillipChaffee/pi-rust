//! The interactive session-share surface at pin `60e7e76`: the upstream
//! concurrent-export isolation case plus the Radius/gist failure boundaries.

#![expect(
    clippy::expect_used,
    reason = "temp-dir and channel fixtures are deterministic; an unexpected failure should fail the test"
)]

use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pi_agent_core::types::BoxedFuture;
use pi_coding_agent::export_html::ExportedTool;
use pi_coding_agent::modes::interactive::session_share::{
    GistOutcome, RadiusUploadOutcome, ShareHttpClient, ShareProcessRunner, ShareSessionSource,
    ShareUserInterface, share_session,
};
use pi_coding_agent::session_manager::SessionManager;
use tokio::sync::oneshot;

fn lock<T>(value: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    value.lock().unwrap_or_else(PoisonError::into_inner)
}

struct FakeSession {
    manager: SessionManager,
    name: String,
    radius_token: Option<String>,
    written: Option<oneshot::Sender<()>>,
    wait_for: Option<oneshot::Receiver<()>>,
}

impl FakeSession {
    fn new(
        name: &str,
        radius_token: Option<&str>,
        written: Option<oneshot::Sender<()>>,
        wait_for: Option<oneshot::Receiver<()>>,
    ) -> Self {
        Self {
            manager: SessionManager::in_memory(Some("/tmp"), None, None)
                .expect("in-memory session"),
            name: name.to_string(),
            radius_token: radius_token.map(str::to_owned),
            written,
            wait_for,
        }
    }
}

impl ShareSessionSource for FakeSession {
    fn session_manager(&self) -> SessionManager {
        self.manager.clone()
    }

    fn system_prompt(&self) -> Option<String> {
        Some(self.name.clone())
    }

    fn tools(&self) -> Vec<ExportedTool> {
        Vec::new()
    }

    fn export_to_html(&mut self, file_path: &str) -> BoxedFuture<'_, Result<(), String>> {
        let name = self.name.clone();
        let file_path = file_path.to_string();
        let written = self.written.take();
        let wait_for = self.wait_for.take();
        Box::pin(async move {
            fs::write(&file_path, name).map_err(|error| error.to_string())?;
            if let Some(sender) = written {
                let _ = sender.send(());
            }
            if let Some(receiver) = wait_for {
                let _ = receiver.await;
            }
            Ok(())
        })
    }

    fn has_radius_provider(&self) -> bool {
        self.radius_token.is_some()
    }

    fn radius_token(&mut self) -> BoxedFuture<'_, Option<String>> {
        let token = self.radius_token.clone();
        Box::pin(async move { token })
    }
}

#[derive(Default)]
struct RecordingUi {
    statuses: Vec<String>,
    errors: Vec<String>,
}

impl ShareUserInterface for RecordingUi {
    fn show_status(&mut self, message: &str) {
        self.statuses.push(message.to_string());
    }

    fn show_error(&mut self, message: &str) {
        self.errors.push(message.to_string());
    }
}

struct FakeProcess {
    uploads: Arc<Mutex<Vec<String>>>,
    created: Arc<AtomicUsize>,
    auth_status: Option<i32>,
    stdout: Option<String>,
    stderr: String,
    code: Option<i32>,
}

impl FakeProcess {
    fn succeeding() -> Self {
        Self {
            uploads: Arc::new(Mutex::new(Vec::new())),
            created: Arc::new(AtomicUsize::new(0)),
            auth_status: Some(0),
            stdout: None,
            stderr: String::new(),
            code: Some(0),
        }
    }
}

impl ShareProcessRunner for FakeProcess {
    fn auth_status(&mut self) -> Option<i32> {
        self.auth_status
    }

    fn create_gist(&mut self, file_path: &str) -> BoxedFuture<'_, GistOutcome> {
        let content = fs::read_to_string(file_path).unwrap_or_default();
        lock(&self.uploads).push(content);
        let number = self.created.fetch_add(1, Ordering::SeqCst) + 1;
        let stdout = self
            .stdout
            .clone()
            .unwrap_or_else(|| format!("https://gist.github.com/test/{number}\n"));
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

struct StubHttp {
    outcome: RadiusUploadOutcome,
}

impl ShareHttpClient for StubHttp {
    fn upload_artifact(
        &mut self,
        _token: &str,
        _body: &[u8],
    ) -> BoxedFuture<'_, RadiusUploadOutcome> {
        let outcome = self.outcome.clone();
        Box::pin(async move { outcome })
    }
}

const fn stub_http(outcome: RadiusUploadOutcome) -> StubHttp {
    StubHttp { outcome }
}

#[tokio::test]
async fn keeps_concurrent_session_exports_isolated() {
    let mut process_a = FakeProcess::succeeding();
    let uploads = Arc::clone(&process_a.uploads);
    let created = Arc::clone(&process_a.created);
    let process_b = FakeProcess {
        uploads: Arc::clone(&uploads),
        created: Arc::clone(&created),
        auth_status: Some(0),
        stdout: None,
        stderr: String::new(),
        code: Some(0),
    };

    let (a_written_sender, a_written) = oneshot::channel();
    let (b_written_sender, b_written) = oneshot::channel();
    let (release_b_sender, release_b) = oneshot::channel();
    let (a_done_sender, a_done) = oneshot::channel();

    let mut session_a = FakeSession::new("A", None, Some(a_written_sender), Some(b_written));
    let mut session_b = FakeSession::new("B", None, Some(b_written_sender), Some(release_b));
    let mut ui_a = RecordingUi::default();
    let mut ui_b = RecordingUi::default();
    let mut process_b = process_b;
    let mut http_a = stub_http(RadiusUploadOutcome::Aborted);
    let mut http_b = stub_http(RadiusUploadOutcome::Aborted);

    let a_run = async {
        let result = share_session(&mut session_a, &mut ui_a, &mut process_a, &mut http_a).await;
        let _ = a_done_sender.send(());
        result
    };
    let b_run = share_session(&mut session_b, &mut ui_b, &mut process_b, &mut http_b);
    let gate = async {
        let _ = a_written.await;
        let _ = a_done.await;
        let _ = release_b_sender.send(());
    };

    let (a_result, b_result, ()) = tokio::join!(a_run, b_run, gate);

    assert!(a_result.is_ok());
    assert!(b_result.is_ok());
    assert_eq!(*lock(&uploads), ["A", "B"]);
    assert_eq!(created.load(Ordering::SeqCst), 2);
    assert!(ui_a.errors.is_empty(), "{:?}", ui_a.errors);
    assert!(ui_b.errors.is_empty(), "{:?}", ui_b.errors);
}

#[tokio::test]
async fn a_radius_upload_short_circuits_the_gist_fallback() {
    let mut session = FakeSession::new("A", Some("radius-token"), None, None);
    let mut ui = RecordingUi::default();
    let mut process = FakeProcess::succeeding();
    let mut http = stub_http(RadiusUploadOutcome::Url(
        "https://radius.test/artifact/1".to_string(),
    ));

    share_session(&mut session, &mut ui, &mut process, &mut http)
        .await
        .expect("share flow");

    assert_eq!(process.created.load(Ordering::SeqCst), 0);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("https://radius.test/artifact/1")),
        "{:?}",
        ui.statuses
    );
    assert!(ui.errors.is_empty());
}

#[tokio::test]
async fn a_failed_radius_upload_reports_and_skips_the_gist() {
    let mut session = FakeSession::new("A", Some("radius-token"), None, None);
    let mut ui = RecordingUi::default();
    let mut process = FakeProcess::succeeding();
    let mut http = stub_http(RadiusUploadOutcome::Failed("boom".to_string()));

    share_session(&mut session, &mut ui, &mut process, &mut http)
        .await
        .expect("share flow");

    assert_eq!(ui.errors, ["Failed to upload Radius artifact: boom"]);
    assert_eq!(process.created.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn missing_or_logged_out_github_cli_reports_the_install_or_login_hint() {
    let cases = [
        (
            None,
            "GitHub CLI (gh) is not installed. Install it from https://cli.github.com/",
        ),
        (
            Some(1),
            "GitHub CLI is not logged in. Run 'gh auth login' first.",
        ),
    ];
    for (auth_status, expected) in cases {
        let mut session = FakeSession::new("A", None, None, None);
        let mut ui = RecordingUi::default();
        let mut process = FakeProcess {
            auth_status,
            ..FakeProcess::succeeding()
        };
        let mut http = stub_http(RadiusUploadOutcome::Aborted);

        share_session(&mut session, &mut ui, &mut process, &mut http)
            .await
            .expect("share flow");

        assert_eq!(ui.errors, [expected]);
        assert_eq!(process.created.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn gist_outcomes_map_to_the_upstream_errors_and_urls() {
    let cases = [
        (None, "boom\n", Some(1), "Failed to create gist: boom"),
        (
            Some(String::new()),
            "",
            Some(0),
            "Failed to parse gist ID from gh output",
        ),
    ];
    for (stdout, stderr, code, expected) in cases {
        let mut session = FakeSession::new("A", None, None, None);
        let mut ui = RecordingUi::default();
        let mut process = FakeProcess {
            stdout,
            stderr: stderr.to_string(),
            code,
            ..FakeProcess::succeeding()
        };
        let mut http = stub_http(RadiusUploadOutcome::Aborted);

        share_session(&mut session, &mut ui, &mut process, &mut http)
            .await
            .expect("share flow");

        assert_eq!(ui.errors, [expected]);
    }

    let mut session = FakeSession::new("A", None, None, None);
    let mut ui = RecordingUi::default();
    let mut process = FakeProcess {
        stdout: Some("https://gist.github.com/test/9\n".to_string()),
        ..FakeProcess::succeeding()
    };
    let mut http = stub_http(RadiusUploadOutcome::Aborted);

    share_session(&mut session, &mut ui, &mut process, &mut http)
        .await
        .expect("share flow");

    let status = ui.statuses.first().expect("share status");
    assert!(status.contains("Share URL:"));
    assert!(status.contains("https://gist.github.com/test/9"));
}
