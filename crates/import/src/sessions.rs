//! The sessions leg: TS pi's tree-structured JSONL session files.
//!
//! The store lives under `<source>/sessions/<encoded-cwd>/`, plus the
//! agent-root strays the v0.30.0 bug left beside the agent dir, upstream's
//! `migrateSessionsFromAgentRoot`. The Rust coding-agent's session store is
//! the same v3 tree format the TS pi writes (the session-manager port rides
//! the same header codec), so the leg validates each file through the
//! store's own parser and copies bytes verbatim — placement derives from
//! the header's cwd through the Rust encoding, upstream's
//! `getDefaultSessionDirPath`.
//!
//! A file whose header cannot be read fails; a file without a session
//! header skips, upstream's discovery semantics. In-place runs validate
//! only: the files already sit where the Rust pi reads them.

use std::path::{Path, PathBuf};

use pi_coding_agent::config::default_session_dir_path;
use pi_coding_agent::session_manager::{SessionHeader, read_session_header};

use crate::discovery::Discovery;
use crate::plan::{OpTarget, PlannedOp};
use crate::report::{ImportReport, SessionItem, SessionOutcome};

/// Run the sessions leg against the discovery, appending items and
/// operations.
pub fn run_sessions(discovery: &Discovery, report: &mut ImportReport) -> Vec<PlannedOp> {
    let mut ops = Vec::new();
    run_sessions_dir(discovery, report, &mut ops);
    run_agent_root_strays(discovery, report, &mut ops);
    ops
}

/// The sessions tree, `<source>/sessions/<encoded-cwd>/*.jsonl`.
fn run_sessions_dir(discovery: &Discovery, report: &mut ImportReport, ops: &mut Vec<PlannedOp>) {
    let sessions_dir = discovery.source.join("sessions");
    if !sessions_dir.is_dir() {
        return;
    }
    for subdir in list_sorted_dirs(&sessions_dir) {
        let subdir_name = subdir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        for file in list_sorted_jsonl(&subdir) {
            let index = report.sessions.len();
            let relative = relative_path(&discovery.source, &file);
            report.sessions.push(SessionItem {
                path: relative.clone(),
                version: None,
                outcome: SessionOutcome::Present,
            });
            match read_session_header(&file) {
                Err(error) => {
                    let reason = error.to_string();
                    report.sessions[index].outcome = SessionOutcome::Failed {
                        reason: reason.clone(),
                    };
                    report.push_failure(relative, reason);
                }
                Ok(None) => {
                    report.sessions[index].outcome = SessionOutcome::Skipped {
                        reason: "no session header".to_string(),
                    };
                }
                Ok(Some(header)) => {
                    report.sessions[index].version = header.version;
                    if discovery.in_place {
                        continue;
                    }
                    let dest_dir = placement(&discovery.target, &header)
                        .unwrap_or_else(|| fallback_placement(&discovery.target, &subdir_name));
                    let placed_to = dest_dir.join(file.file_name().unwrap_or_default());
                    if placed_to.exists() {
                        continue;
                    }
                    report.sessions[index].outcome = SessionOutcome::Copied {
                        placed_to: placed_to.display().to_string(),
                    };
                    ops.push(PlannedOp::CopyFile {
                        from: file,
                        to: placed_to,
                        targets: vec![OpTarget::Session(index)],
                    });
                }
            }
        }
    }
}

/// The agent-root strays, upstream's v0.30.0 bug: `*.jsonl` files beside
/// the agent dir, placed by the cwd in their header. In-place runs move
/// them (upstream's `renameSync`); cross-dir runs copy, leaving the source
/// untouched.
fn run_agent_root_strays(
    discovery: &Discovery,
    report: &mut ImportReport,
    ops: &mut Vec<PlannedOp>,
) {
    for stray in list_sorted_jsonl(&discovery.source) {
        let index = report.sessions.len();
        let relative = relative_path(&discovery.source, &stray);
        report.sessions.push(SessionItem {
            path: relative.clone(),
            version: None,
            outcome: SessionOutcome::Present,
        });
        let header = match read_session_header(&stray) {
            Err(error) => {
                let reason = error.to_string();
                report.sessions[index].outcome = SessionOutcome::Failed {
                    reason: reason.clone(),
                };
                report.push_failure(relative, reason);
                continue;
            }
            Ok(None) => {
                report.sessions[index].outcome = SessionOutcome::Skipped {
                    reason: "no session header".to_string(),
                };
                continue;
            }
            Ok(Some(header)) => header,
        };
        report.sessions[index].version = header.version;
        let Some(cwd) = header.cwd() else {
            // Upstream's sweep skips headers without a cwd.
            report.sessions[index].outcome = SessionOutcome::Skipped {
                reason: "session header has no cwd".to_string(),
            };
            continue;
        };
        let placed_to = default_session_dir_path(cwd, &discovery.target.display().to_string())
            .join(stray.file_name().unwrap_or_default());
        if placed_to.exists() {
            continue;
        }
        report.sessions[index].outcome = SessionOutcome::Copied {
            placed_to: placed_to.display().to_string(),
        };
        ops.push(if discovery.in_place {
            PlannedOp::Rename {
                from: stray,
                to: placed_to,
                targets: vec![OpTarget::Session(index)],
            }
        } else {
            PlannedOp::CopyFile {
                from: stray,
                to: placed_to,
                targets: vec![OpTarget::Session(index)],
            }
        });
    }
}

/// The target session directory for a header, upstream's
/// `getDefaultSessionDirPath` over the header's cwd; a header without a cwd
/// falls back to the source directory it already sits in, the placement the
/// TS runtime made.
fn placement(target: &Path, header: &SessionHeader) -> Option<PathBuf> {
    let cwd = header.cwd()?;
    Some(default_session_dir_path(cwd, &target.display().to_string()))
}

/// The fallback placement directory: the same encoded directory name the
/// file sits in at the source.
fn fallback_placement(target: &Path, subdir_name: &str) -> PathBuf {
    target.join("sessions").join(subdir_name)
}

/// The sorted immediate subdirectories of a directory; unreadable or
/// non-directory entries skip, discovery's semantics.
fn list_sorted_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return dirs;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();
    dirs
}

/// The sorted `*.jsonl` files of a directory.
fn list_sorted_jsonl(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension == "jsonl")
        {
            files.push(path);
        }
    }
    files.sort();
    files
}

/// A path relative to the agent dir, the report's path form; paths outside
/// the dir carry their full display form.
#[doc(hidden)]
#[must_use]
pub fn relative_path(base: &Path, path: &Path) -> String {
    path.strip_prefix(base).map_or_else(
        |_| path.display().to_string(),
        |relative| relative.display().to_string(),
    )
}
