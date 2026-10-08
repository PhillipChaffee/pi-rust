//! The write plan: the filesystem operations the legs propose and the
//! runner applies once every leg has inventoried its artifacts.
//!
//! `--dry-run` skips the apply, so the report's outcomes and the writes
//! always come from one pass, and an apply failure downgrades the report
//! item the op was planned for instead of leaving it claiming success.

use std::path::PathBuf;

use pi_coding_agent::auth_storage::{AuthStorageBackend, FileAuthStorageBackend, LockOutcome};

/// The message form of one filesystem error, the shared `map_err` mapper:
/// one function instance for every error arm keeps the never-firing arms
/// from carrying their own coverage weight.
#[expect(
    clippy::needless_pass_by_value,
    reason = "map_err supplies the error by value; the shared mapper keeps one function instance for every error arm"
)]
fn io_message(error: std::io::Error) -> String {
    error.to_string()
}

/// The report item an operation was planned for, downgraded to a failure
/// when the operation's apply fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpTarget {
    /// The sessions leg's item at this index.
    Session(usize),
    /// The credentials leg's item at this index.
    Credential(usize),
    /// The settings leg's item at this index.
    Settings(usize),
    /// The trust leg's item.
    Trust,
}

/// One proposed filesystem operation.
#[derive(Debug)]
pub enum PlannedOp {
    /// Write a file's text, creating its parent directory first. The mode
    /// rides the house store rules when set (auth.json's 0600); otherwise
    /// the default umask applies, upstream's plain `writeFileSync`.
    WriteFile {
        /// The file to write.
        path: PathBuf,
        /// The text to write.
        content: String,
        /// The file mode when the store's rules set one.
        mode: Option<u32>,
        /// The report items the operation's failure downgrades.
        targets: Vec<OpTarget>,
    },
    /// Byte-copy a file, creating its parent directory first; the sessions
    /// leg's cross-dir carrier.
    CopyFile {
        /// The source file.
        from: PathBuf,
        /// The destination file.
        to: PathBuf,
        /// The report items the operation's failure downgrades.
        targets: Vec<OpTarget>,
    },
    /// Move a file within the source tree, creating the destination's
    /// parent first; the agent-root stray's in-place placement, upstream's
    /// `renameSync`.
    Rename {
        /// The file to move.
        from: PathBuf,
        /// The destination file.
        to: PathBuf,
        /// The report items the operation's failure downgrades.
        targets: Vec<OpTarget>,
    },
    /// Replace the auth store's content through the store's own locked
    /// write path — parent directory at 0700, the lock discipline, the
    /// 0600 file mode — with the merged map the leg computed.
    AuthWrite {
        /// The store backend the write goes through.
        backend: FileAuthStorageBackend,
        /// The next file content, the merged map pretty-serialized.
        next: String,
        /// The report items the operation's failure downgrades.
        targets: Vec<OpTarget>,
    },
}

impl PlannedOp {
    /// The report items this operation's failure downgrades.
    #[must_use]
    pub fn targets(&self) -> &[OpTarget] {
        match self {
            Self::WriteFile { targets, .. }
            | Self::CopyFile { targets, .. }
            | Self::Rename { targets, .. }
            | Self::AuthWrite { targets, .. } => targets,
        }
    }

    /// Apply the operation, `--dry-run`'s no-write branch included.
    ///
    /// # Errors
    /// The filesystem failure the operation raised, message-only for the
    /// report.
    pub fn apply(&self, dry_run: bool) -> Result<(), String> {
        if dry_run {
            return Ok(());
        }
        match self {
            Self::WriteFile {
                path,
                content,
                mode,
                ..
            } => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(io_message)?;
                }
                write_file(path, content, *mode)
            }
            Self::CopyFile { from, to, .. } => {
                if let Some(parent) = to.parent() {
                    std::fs::create_dir_all(parent).map_err(io_message)?;
                }
                std::fs::copy(from, to).map_err(io_message)?;
                Ok(())
            }
            Self::Rename { from, to, .. } => {
                if let Some(parent) = to.parent() {
                    std::fs::create_dir_all(parent).map_err(io_message)?;
                }
                std::fs::rename(from, to).map_err(io_message)
            }
            Self::AuthWrite { backend, next, .. } => backend
                .with_lock(|_| LockOutcome {
                    result: (),
                    next: Some(next.clone()),
                })
                .map(drop)
                .map_err(|error| error.to_string()),
        }
    }
}

/// Write a file's text with the optional mode, upstream's `writeFileSync`
/// options object.
fn write_file(path: &std::path::Path, content: &str, mode: Option<u32>) -> Result<(), String> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let Some(mode) = mode else {
        return std::fs::write(path, content).map_err(io_message);
    };
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(mode)
        .open(path)
        .map_err(io_message)?;
    file.write_all(content.as_bytes()).map_err(io_message)
}
