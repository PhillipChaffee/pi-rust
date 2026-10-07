//! The session share surface, upstream's
//! `src/modes/interactive/session-share.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The portable substance is the export-for-share shape and the share flow:
//! export → Radius artifact upload when a Radius provider and token exist →
//! GitHub CLI fallback (auth check, HTML export, private gist).
//!
//! The loader UI (`BorderedLoader`, editor swap) and its abort signal ride the
//! interactive-mode slice (#131); the model-runtime/auth reads the runtime
//! slice wires (#121/#125); until then the session is a trait source and the
//! child-process/HTTP legs are injectable seams, upstream's `vi.mock` and
//! `fetch` boundaries.

use std::fs;
use std::path::PathBuf;

use serde_json::{Value as JsonValue, json};

use pi_agent_core::types::BoxedFuture;
use pi_tui::terminal_image::hyperlink;

use crate::config::get_share_viewer_url;
use crate::export_html::ExportedTool;
use crate::session_export::export_session_to_jsonl;
use crate::session_manager::{SessionManager, SessionManagerError};

/// The shareable state a session exposes, upstream's `session.sessionManager`
/// and `session.state`/`session.modelRuntime` reads the flow consumes.
pub trait ShareSessionSource {
    /// The session manager the share export walks, upstream's
    /// `session.sessionManager`.
    fn session_manager(&self) -> &SessionManager;

    /// The system prompt, upstream's `session.state.systemPrompt`.
    fn system_prompt(&self) -> Option<String>;

    /// The active tools, upstream's `session.state.tools` mapped to
    /// `{name, description, parameters}`.
    fn tools(&self) -> Vec<ExportedTool>;

    /// HTML export, upstream's `session.exportToHtml` — the AgentSession
    /// method rides its own slice (#125).
    fn export_to_html(&mut self, file_path: &str) -> BoxedFuture<'_, Result<(), String>>;

    /// Whether a Radius provider is configured, upstream's
    /// `modelRuntime.getProvider("radius")` presence — rides #121.
    fn has_radius_provider(&self) -> bool;

    /// The Radius auth token, upstream's
    /// `getAuthCredential(await modelRuntime.getAuth("radius", ...))` —
    /// rides #121/#129.
    fn radius_token(&mut self) -> BoxedFuture<'_, Option<String>>;
}

/// The share flow's user-visible output, upstream's `showStatus`/`showError`
/// context methods. The loader/editor container machinery rides #131.
pub trait ShareUserInterface {
    /// A transient status line, upstream's `showStatus`.
    fn show_status(&mut self, message: &str);

    /// An error line, upstream's `showError`.
    fn show_error(&mut self, message: &str);
}

/// One child-process outcome, upstream's `(stdout, stderr, code)` close event.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GistOutcome {
    /// The child's stdout.
    pub stdout: String,
    /// The child's stderr.
    pub stderr: String,
    /// The exit code; absent when the child was killed by a signal.
    pub code: Option<i32>,
}

/// The child-process seam, upstream's `node:child_process`
/// `spawnSync`/`spawn` of `gh` (the boundary the suite mocks).
pub trait ShareProcessRunner {
    /// `gh auth status`'s exit status; absent when the program cannot spawn
    /// (upstream's throw).
    fn auth_status(&mut self) -> Option<i32>;

    /// `gh gist create --public=false <file>`'s close event.
    fn create_gist(&mut self, file_path: &str) -> BoxedFuture<'_, GistOutcome>;
}

/// Why a Radius artifact upload settled, upstream's fetch/json/abort ladder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RadiusUploadOutcome {
    /// The artifact's canonical URL.
    Url(String),
    /// The upload failed with the composed upstream message.
    Failed(String),
    /// The caller aborted, upstream's `loader.signal.aborted` checks.
    Aborted,
}

/// The HTTP seam for the Radius artifact upload, upstream's `fetch` boundary.
pub trait ShareHttpClient {
    /// POST `/v1/artifacts` with the session JSONL body.
    fn upload_artifact(&mut self, token: &str, body: &[u8])
    -> BoxedFuture<'_, RadiusUploadOutcome>;
}

/// Export the current branch with presentation metadata for Radius, upstream's
/// `exportSessionForShare`.
///
/// The JSONL export plus one trailing `pi.share` custom entry carries the
/// system prompt and tool specs.
///
/// # Errors
/// [`SessionManagerError`] when the export write fails.
pub fn export_session_for_share(
    file_path: &str,
    session: &dyn ShareSessionSource,
) -> Result<String, SessionManagerError> {
    let system_prompt = session.system_prompt();
    let tools = session.tools();
    export_session_to_jsonl(
        session.session_manager(),
        Some(file_path),
        Some(&|parent_id, timestamp| {
            let mut data = serde_json::Map::new();
            if let Some(system_prompt) = system_prompt.clone() {
                data.insert("systemPrompt".to_owned(), JsonValue::String(system_prompt));
            }
            data.insert(
                "tools".to_owned(),
                serde_json::to_value(&tools).unwrap_or_default(),
            );
            vec![json!({
                "type": "custom",
                "customType": "pi.share",
                "id": short_share_id(),
                "parentId": parent_id.map_or(JsonValue::Null, |id| JsonValue::String(id.to_owned())),
                "timestamp": timestamp,
                "data": data,
            })]
        }),
    )
}

/// An 8-hex id, upstream's `crypto.randomUUID().slice(0, 8)`.
fn short_share_id() -> String {
    let mut simple = uuid::Uuid::new_v4().simple().to_string();
    simple.truncate(8);
    simple
}

/// The mkdtemp prefix, upstream's `path.join(os.tmpdir(), "pi-share-")`.
const SHARE_TEMP_PREFIX: &str = "pi-share-";

/// Create the exclusive per-share temp directory, upstream's
/// `fs.mkdtempSync`.
fn make_share_temp_dir() -> Result<PathBuf, String> {
    for _ in 0..100 {
        let mut simple = uuid::Uuid::new_v4().simple().to_string();
        simple.truncate(8);
        let dir = std::env::temp_dir().join(format!("{SHARE_TEMP_PREFIX}{simple}"));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    Err("could not create a share temp directory".to_owned())
}

/// Try the Radius upload leg; true when the flow is settled (handled or
/// failed), false when no Radius provider/token exists, upstream's
/// `tryShareViaRadius`.
async fn try_share_via_radius(
    jsonl_file: &str,
    session: &mut dyn ShareSessionSource,
    ui: &mut dyn ShareUserInterface,
    http: &mut dyn ShareHttpClient,
) -> bool {
    if !session.has_radius_provider() {
        return false;
    }
    let Some(token) = session.radius_token().await else {
        return false;
    };

    let Ok(body) = fs::read(jsonl_file) else {
        // Upstream's readFileSync throw lands in the catch block.
        ui.show_error("Failed to upload Radius artifact: could not read the exported session");
        return true;
    };
    match http.upload_artifact(&token, &body).await {
        RadiusUploadOutcome::Aborted => true,
        RadiusUploadOutcome::Failed(message) => {
            ui.show_error(&format!("Failed to upload Radius artifact: {message}"));
            true
        }
        RadiusUploadOutcome::Url(share_url) => {
            ui.show_status(&format!("Share URL: {}", hyperlink(&share_url, &share_url)));
            true
        }
    }
}

/// The gist fallback leg, upstream's `shareViaGist`.
async fn share_via_gist(
    html_file: &str,
    ui: &mut dyn ShareUserInterface,
    process: &mut dyn ShareProcessRunner,
) {
    let result = process.create_gist(html_file).await;

    if result.code != Some(0) {
        let message = if result.stderr.trim().is_empty() {
            "Unknown error"
        } else {
            result.stderr.trim()
        };
        ui.show_error(&format!("Failed to create gist: {message}"));
        return;
    }

    let gist_url = result.stdout.trim();
    let Some(gist_id) = gist_url.split('/').next_back().filter(|id| !id.is_empty()) else {
        ui.show_error("Failed to parse gist ID from gh output");
        return;
    };

    let preview_url = get_share_viewer_url(gist_id);
    ui.show_status(&format!(
        "Share URL: {}\nGist: {}",
        hyperlink(&preview_url, &preview_url),
        hyperlink(gist_url, gist_url)
    ));
}

/// Share the current session through Radius, falling back to a private gist,
/// upstream's `shareSession`. Each call exports into its own exclusive temp
/// directory, so concurrent shares stay isolated.
///
/// # Errors
/// Only the temp-directory creation fails outward (upstream's
/// `mkdtempSync` throw); every session/export leg reports through
/// [`ShareUserInterface::show_error`] and returns.
pub async fn share_session(
    session: &mut dyn ShareSessionSource,
    ui: &mut dyn ShareUserInterface,
    process: &mut dyn ShareProcessRunner,
    http: &mut dyn ShareHttpClient,
) -> Result<(), String> {
    let temp_dir = make_share_temp_dir()?;
    let jsonl_file = temp_dir.join("session.jsonl").display().to_string();
    let html_file = temp_dir.join("session.html").display().to_string();

    let settled = async {
        if let Err(error) = export_session_for_share(&jsonl_file, session) {
            ui.show_error(&format!("Failed to export session: {error}"));
            return;
        }
        if try_share_via_radius(&jsonl_file, session, ui, http).await {
            return;
        }

        match process.auth_status() {
            None => {
                ui.show_error(
                    "GitHub CLI (gh) is not installed. Install it from https://cli.github.com/",
                );
            }
            Some(status) if status != 0 => {
                ui.show_error("GitHub CLI is not logged in. Run 'gh auth login' first.");
            }
            Some(_) => {
                if let Err(error) = session.export_to_html(&html_file).await {
                    ui.show_error(&format!("Failed to export session: {error}"));
                    return;
                }
                share_via_gist(&html_file, ui, process).await;
            }
        }
    }
    .await;

    // finally: remove the temp dir, ignoring cleanup errors.
    let _ = fs::remove_dir_all(&temp_dir);
    Ok(settled)
}
