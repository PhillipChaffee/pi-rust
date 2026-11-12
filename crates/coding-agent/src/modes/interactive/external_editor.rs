//! The external-editor hand-off, upstream
//! `src/modes/interactive/external-editor.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! The prompt is written into a private `pi-editor-*` directory, the
//! configured editor inherits the terminal until it exits, and a successful
//! run reads the file back — BOM stripped, one trailing newline removed.
//! Every path removes the directory again.
//!
//! Port restatements:
//!
//! - Upstream spawns the editor asynchronously precisely so a Windows
//!   `spawnSync` cannot keep libuv's console input read active while
//!   vim/nvim own the console. The port keeps the asynchronous spawn
//!   ([`tokio::process::Command`]) and drops the `shell: win32` branch;
//!   Windows is out of scope for this effort, so the command string splits
//!   on spaces exactly as upstream's non-shell branch does.
//! - The announcement goes through [`crate::core::output_guard::write_stdout`],
//!   the port's monkey-patched `process.stdout.write`: while the TUI owns the
//!   terminal the notice lands on stderr, which is what upstream's output
//!   guard does too.
//! - `mkdtempSync` becomes a retrying [`std::fs::DirBuilder`] with mode
//!   `0700` on unix, keeping the private-directory contract the suite pins.

use std::fs;
use std::io::ErrorKind;
use std::path::PathBuf;

use crate::core::output_guard::write_stdout;
use crate::utils::text::strip_bom;

/// The hand-off options, upstream `ExternalEditorOptions`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalEditorOptions {
    /// The editor command line, e.g. `vim -f`; split on single spaces.
    pub command: String,
    /// The prompt text written before the editor starts.
    pub content: String,
}

/// The hand-off outcome, upstream `ExternalEditorResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalEditorResult {
    /// The editor exited zero; `content` is the file read back.
    Complete {
        /// The edited prompt, BOM stripped and one trailing newline removed.
        content: String,
    },
    /// The editor failed to spawn, exited non-zero, or died on a signal.
    Failed,
}

/// The edited file name inside the private directory, upstream
/// `prompt.md`.
const PROMPT_FILE_NAME: &str = "prompt.md";

/// The editor command, upstream's inline
/// `explicit || VISUAL || EDITOR || platform default`.
#[must_use]
pub fn resolve_external_editor_command(explicit: Option<&str>) -> String {
    resolve_external_editor_command_with(explicit, &|key| std::env::var(key).ok(), cfg!(windows))
}

/// [`resolve_external_editor_command`] over an injected environment lookup
/// and platform flag.
#[must_use]
pub fn resolve_external_editor_command_with(
    explicit: Option<&str>,
    env: &dyn Fn(&str) -> Option<String>,
    is_windows: bool,
) -> String {
    if let Some(explicit) = explicit.filter(|command| !command.is_empty()) {
        return explicit.to_string();
    }
    for key in ["VISUAL", "EDITOR"] {
        if let Some(command) = env(key).filter(|command| !command.is_empty()) {
            return command;
        }
    }
    if is_windows {
        "notepad".to_string()
    } else {
        "nano".to_string()
    }
}

/// The temp-directory prefix, upstream `pi-editor-`.
const TEMP_DIRECTORY_PREFIX: &str = "pi-editor-";

/// The temp-directory suffix length, upstream `mkdtempSync`'s six-random-
/// character tail reduced to the first eight hex characters of a UUID.
const TEMP_DIRECTORY_SUFFIX_LENGTH: usize = 8;

/// Create the exclusive per-edit temp directory with upstream
/// `mkdtempSync`'s private mode.
fn create_editor_temp_directory() -> std::io::Result<PathBuf> {
    let parent = std::env::temp_dir();
    for _ in 0..100 {
        let mut suffix = uuid::Uuid::new_v4().simple().to_string();
        suffix.truncate(TEMP_DIRECTORY_SUFFIX_LENGTH);
        let directory = parent.join(format!("{TEMP_DIRECTORY_PREFIX}{suffix}"));
        #[cfg(unix)]
        let created = {
            use std::os::unix::fs::DirBuilderExt;
            fs::DirBuilder::new().mode(0o700).create(&directory)
        };
        #[cfg(not(unix))]
        let created = fs::DirBuilder::new().create(&directory);
        match created {
            Ok(()) => return Ok(directory),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        ErrorKind::AlreadyExists,
        "could not create an external-editor temp directory",
    ))
}

/// Open the prompt in the configured editor and read the result back,
/// upstream `editInExternalEditor`.
///
/// # Errors
/// The temp-directory creation, the initial write, or the read-back can
/// fail; upstream lets the same `fs` throws reject the promise.
pub async fn edit_in_external_editor(
    options: ExternalEditorOptions,
) -> std::io::Result<ExternalEditorResult> {
    let directory = create_editor_temp_directory()?;
    let file_path = directory.join(PROMPT_FILE_NAME);

    let outcome = async {
        fs::write(&file_path, &options.content)?;
        let mut parts = options.command.split(' ');
        // `split` always yields at least one item, so the editor is present
        // even when the command string is empty; the spawn then fails the
        // way upstream's empty-program spawn raises its error event.
        let editor = parts.next().unwrap_or_default();
        let editor_args: Vec<&str> = parts.collect();

        write_stdout(&format!(
            "Launching external editor: {}\nPi will resume when the editor exits.\n",
            options.command
        ));

        let mut command = tokio::process::Command::new(editor);
        command.args(&editor_args).arg(&file_path);
        let succeeded = command.status().await.is_ok_and(|status| status.success());
        if !succeeded {
            return Ok(ExternalEditorResult::Failed);
        }

        let bytes = fs::read(&file_path)?;
        let content = String::from_utf8_lossy(&bytes);
        let content = strip_bom(&content);
        let content = content.strip_suffix('\n').unwrap_or(content);
        Ok(ExternalEditorResult::Complete {
            content: content.to_string(),
        })
    }
    .await;

    // Cleanup is best effort, upstream's `finally` swallowing `rmSync`.
    let _ = fs::remove_dir_all(&directory);
    outcome
}

/// Edit `content` in the external editor and block until it exits.
///
/// This is upstream `editInExternalEditor` for callers that cannot await:
/// the extension prompt's Ctrl+G handler runs inside the synchronous input
/// dispatch, and the TUI is stopped for the editor's whole lifetime, so
/// blocking is observationally the async form.
#[must_use]
pub fn edit_in_external_editor_blocking(command: &str, content: &str) -> ExternalEditorResult {
    let Ok(directory) = create_editor_temp_directory() else {
        return ExternalEditorResult::Failed;
    };
    let file_path = directory.join(PROMPT_FILE_NAME);
    let outcome = (|| {
        if fs::write(&file_path, content).is_err() {
            return ExternalEditorResult::Failed;
        }
        write_stdout(&format!(
            "Launching external editor: {command}\nPi will resume when the editor exits.\n"
        ));
        let mut parts = command.split(' ');
        // `split` always yields at least one item; an empty command fails
        // the spawn the way upstream's empty-program spawn raises.
        let editor = parts.next().unwrap_or_default();
        let status = std::process::Command::new(editor)
            .args(parts)
            .arg(&file_path)
            .status();
        if !status.is_ok_and(|status| status.success()) {
            return ExternalEditorResult::Failed;
        }
        let Ok(bytes) = fs::read(&file_path) else {
            return ExternalEditorResult::Failed;
        };
        let content = String::from_utf8_lossy(&bytes);
        let content = strip_bom(&content);
        ExternalEditorResult::Complete {
            content: content.strip_suffix('\n').unwrap_or(content).to_string(),
        }
    })();

    // Cleanup is best effort, upstream's `finally` swallowing `rmSync`.
    let _ = fs::remove_dir_all(&directory);
    outcome
}
