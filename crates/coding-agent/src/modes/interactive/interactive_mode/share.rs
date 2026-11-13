//! The frame's `/share` wiring, upstream's `shareSession` call site in
//! `src/modes/interactive/interactive-mode.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! The export → Radius → gist ladder lives in [`crate::session_share`]; this
//! module supplies its three concrete seams and reports through the frame's
//! status surface.
//!
//! Restatement: upstream swaps a `BorderedLoader` into the editor container
//! for the duration of the upload. The port runs the flow inline and reports
//! through `show_status`/`show_error`; the loader's abort signal has no
//! counterpart on the seam, so an upload is never cancelled by Escape.

use std::sync::Arc;

use pi_ai::http::client::read_body_text;
use pi_ai::http::{HttpMethod, HttpRequest, default_http_client, status_reason};
use pi_ai::providers::radius_config::DEFAULT_RADIUS_GATEWAY;
use pi_ai::types::BoxedFuture;
use serde_json::Value;

use super::InteractiveMode;
use crate::core::agent_session::AgentSession;
use crate::session_share::{
    GistOutcome, RadiusUploadOutcome, ShareHttpClient, ShareProcessRunner, ShareUserInterface,
    share_session,
};
use crate::settings_manager::SettingsStorageKind;

/// The frame's status sink for the share flow.
struct FrameShareUi<'a, S: SettingsStorageKind> {
    frame: &'a InteractiveMode<S>,
}

impl<S: SettingsStorageKind> ShareUserInterface for FrameShareUi<'_, S> {
    fn show_status(&mut self, message: &str) {
        self.frame.show_status(message);
    }

    fn show_error(&mut self, message: &str) {
        self.frame.show_error(message);
    }
}

/// The `gh` child-process seam, upstream's `spawnSync`/`spawn` of the GitHub
/// CLI.
struct GhProcessRunner;

impl ShareProcessRunner for GhProcessRunner {
    fn auth_status(&mut self) -> Option<i32> {
        std::process::Command::new("gh")
            .args(["auth", "status"])
            .output()
            .ok()
            .and_then(|output| output.status.code())
    }

    fn create_gist(&mut self, file_path: &str) -> BoxedFuture<'_, GistOutcome> {
        let file_path = file_path.to_owned();
        Box::pin(async move {
            match tokio::process::Command::new("gh")
                .args(["gist", "create", "--public=false", &file_path])
                .output()
                .await
            {
                Ok(output) => GistOutcome {
                    stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                    code: output.status.code(),
                },
                Err(_) => GistOutcome::default(),
            }
        })
    }
}

/// The Radius artifact upload over the process-default HTTP client.
struct RadiusHttpClient;

impl ShareHttpClient for RadiusHttpClient {
    fn upload_artifact(
        &mut self,
        token: &str,
        body: &[u8],
    ) -> BoxedFuture<'_, RadiusUploadOutcome> {
        let token = token.to_owned();
        let body = body.to_vec();
        Box::pin(async move {
            let response = default_http_client()
                .execute(HttpRequest {
                    method: HttpMethod::Post,
                    url: format!(
                        "{DEFAULT_RADIUS_GATEWAY}/v1/artifacts?visibility=organization&title=Pi%20session"
                    ),
                    headers: vec![
                        ("Authorization".to_owned(), format!("Bearer {token}")),
                        (
                            "Content-Type".to_owned(),
                            "application/x-ndjson".to_owned(),
                        ),
                    ],
                    body: Some(body.into()),
                    timeout_ms: None,
                    signal: tokio_util::sync::CancellationToken::new(),
                })
                .await;
            let response = match response {
                Ok(response) => response,
                Err(error) => return RadiusUploadOutcome::Failed(error.to_string()),
            };
            let status = response.status;
            let text = read_body_text(response.body).await.unwrap_or_default();
            let json: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            if status < 400
                && let Some(url) = json
                    .get("artifact")
                    .and_then(|artifact| artifact.get("canonical_url"))
                    .and_then(Value::as_str)
            {
                return RadiusUploadOutcome::Url(url.to_owned());
            }
            let message = json.get("error").and_then(Value::as_str).map_or_else(
                || {
                    let reason = status_reason(status);
                    if reason.is_empty() {
                        status.to_string()
                    } else {
                        format!("{status} {reason}")
                    }
                },
                str::to_owned,
            );
            RadiusUploadOutcome::Failed(message)
        })
    }
}

/// Share the current session, upstream `handleShareCommand`'s call into
/// `shareSession`.
pub(crate) async fn share_current_session<S: SettingsStorageKind>(frame: &InteractiveMode<S>) {
    let mut process = GhProcessRunner;
    let mut http = RadiusHttpClient;
    frame
        .share_current_session_with(&mut process, &mut http)
        .await;
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Share the current session over caller-supplied process and HTTP seams,
    /// the acceptance suite's stand-in for the real `gh` and Radius clients.
    #[doc(hidden)]
    pub async fn share_current_session_with(
        &self,
        process: &mut dyn ShareProcessRunner,
        http: &mut dyn ShareHttpClient,
    ) {
        let mut session: Arc<AgentSession<S>> = self.session();
        let mut ui = FrameShareUi { frame: self };
        if let Err(error) = share_session(&mut session, &mut ui, process, http).await {
            self.show_error(&format!("Failed to share session: {error}"));
        }
        self.tui().request_render(false);
    }
}
