//! The manager constructors and discovery entry points, upstream's
//! `SessionManager.create/open/continueRecent/inMemory/forkFrom/list/listAll`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::fs;
use std::io::Write as _;
use std::path::Path;

use pi_ai::utils::uuid::uuidv7;

use super::discovery::{
    SessionInfo, SessionListProgress, get_default_session_dir, list_sessions_from_dir,
};
use super::file_entry::{FileEntry, SessionManagerError};
use super::manager::SessionManager;
use super::parse::{load_entries_from_file, read_session_header};
use super::types::{CURRENT_SESSION_VERSION, NewSessionOptions, SessionHeader};
use super::{
    assert_valid_session_id, file_timestamp, normalize_here, now_iso8601, path_exists, resolve_here,
};

impl SessionManager {
    /// Create a persisted session, upstream's `SessionManager.create`.
    ///
    /// # Errors
    /// [`SessionManagerError::InvalidSessionId`] for an invalid explicit id;
    /// [`SessionManagerError::Io`] when the default directory cannot be
    /// created.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "create takes the upstream options object by value; the parity signature outranks the borrow"
    )]
    pub fn create(
        cwd: &str,
        session_dir: Option<&str>,
        options: Option<NewSessionOptions>,
    ) -> Result<Self, SessionManagerError> {
        let dir = match session_dir {
            Some(session_dir) => normalize_here(session_dir)?,
            None => get_default_session_dir(cwd)?.display().to_string(),
        };
        Self::new(cwd, &dir, None, true, options.as_ref(), None)
    }

    /// Open a specific session file, upstream's `SessionManager.open`.
    ///
    /// # Errors
    /// [`SessionManagerError::InvalidSessionFile`] when the file exists but
    /// is not a pi session. The bounded header scan is only a discovery
    /// optimization: a scan-limit failure falls back to the full load, which
    /// remains authoritative for legacy files with very large headers or
    /// prefixes.
    pub fn open(
        path: &str,
        session_dir: Option<&str>,
        cwd_override: Option<&str>,
    ) -> Result<Self, SessionManagerError> {
        let resolved_path = resolve_here(path);
        let mut header: Option<SessionHeader> = None;
        let mut preloaded_file_entries: Option<Vec<FileEntry>> = None;
        if cwd_override.is_none() && path_exists(&resolved_path) {
            match read_session_header(Path::new(&resolved_path)) {
                Ok(read) => header = read,
                Err(super::file_entry::ReadSessionHeaderError::ScanLimit(_)) => {
                    preloaded_file_entries = Some(load_entries_from_file(&resolved_path));
                    header = match preloaded_file_entries
                        .as_ref()
                        .and_then(|entries| entries.first())
                    {
                        Some(FileEntry::Session(header)) => Some(header.clone()),
                        _ => None,
                    };
                }
                Err(super::file_entry::ReadSessionHeaderError::Io(error)) => {
                    return Err(SessionManagerError::Io(error));
                }
            }
        }
        let cwd = cwd_override
            .map(str::to_owned)
            .or_else(|| header.as_ref().and_then(|header| header.cwd.clone()))
            .unwrap_or_else(crate::config::process_cwd);
        // If no sessionDir provided, derive from the file's parent directory.
        let dir = match session_dir {
            Some(session_dir) => normalize_here(session_dir)?,
            None => Path::new(&resolved_path)
                .parent()
                .map_or_else(|| ".".to_owned(), |parent| parent.display().to_string()),
        };
        Self::new(
            &cwd,
            &dir,
            Some(&resolved_path),
            true,
            None,
            preloaded_file_entries,
        )
    }

    /// Continue the most recent session, or create new if none, upstream's
    /// `SessionManager.continueRecent`.
    ///
    /// # Errors
    /// [`SessionManagerError::Io`] when the default directory cannot be
    /// created.
    pub fn continue_recent(
        cwd: &str,
        session_dir: Option<&str>,
    ) -> Result<Self, SessionManagerError> {
        let dir = match session_dir {
            Some(session_dir) => normalize_here(session_dir)?,
            None => get_default_session_dir(cwd)?.display().to_string(),
        };
        let filter_cwd = Self::discovery_filters_cwd(session_dir, &dir, cwd);
        let most_recent =
            super::discovery::find_most_recent_session(&dir, filter_cwd.then_some(cwd));
        most_recent.map_or_else(
            || Self::new(cwd, &dir, None, true, None, None),
            |recent| Self::new(cwd, &dir, Some(&recent), true, None, None),
        )
    }

    /// Create an in-memory session (no file persistence), optionally from
    /// entries held outside the filesystem, upstream's
    /// `SessionManager.inMemory`.
    ///
    /// # Errors
    /// [`SessionManagerError::InvalidSessionId`] for an invalid explicit id.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "inMemory takes the upstream options object by value; the parity signature outranks the borrow"
    )]
    pub fn in_memory(
        cwd: Option<&str>,
        options: Option<NewSessionOptions>,
        entries: Option<Vec<FileEntry>>,
    ) -> Result<Self, SessionManagerError> {
        let cwd = cwd.map_or_else(crate::config::process_cwd, str::to_owned);
        Self::new(&cwd, "", None, false, options.as_ref(), entries)
    }

    /// Fork a session from another project directory into the current
    /// project, upstream's `SessionManager.forkFrom`: creates a new session
    /// in the target cwd with the full history from the source session.
    ///
    /// # Errors
    /// [`SessionManagerError::ForkSourceInvalid`] or
    /// [`SessionManagerError::ForkSourceHeaderless`] for an unusable source;
    /// [`SessionManagerError::InvalidSessionId`] for an invalid explicit id.
    pub fn fork_from(
        source_path: &str,
        target_cwd: &str,
        session_dir: Option<&str>,
        options: Option<NewSessionOptions>,
    ) -> Result<Self, SessionManagerError> {
        let resolved_source_path = resolve_here(source_path);
        let resolved_target_cwd = resolve_here(target_cwd);
        let source_entries = load_entries_from_file(&resolved_source_path);
        if source_entries.is_empty() {
            return Err(SessionManagerError::ForkSourceInvalid(resolved_source_path));
        }
        if !source_entries
            .iter()
            .any(|entry| matches!(entry, FileEntry::Session(_)))
        {
            return Err(SessionManagerError::ForkSourceHeaderless(
                resolved_source_path,
            ));
        }

        let dir = match session_dir {
            Some(session_dir) => normalize_here(session_dir)?,
            None => get_default_session_dir(&resolved_target_cwd)?
                .display()
                .to_string(),
        };
        if !path_exists(&dir) {
            fs::create_dir_all(&dir)?;
        }

        // Create new session file with new ID but forked content.
        if let Some(id) = options.as_ref().and_then(|options| options.id.as_deref()) {
            assert_valid_session_id(id)?;
        }
        let new_session_id = match options.and_then(|options| options.id) {
            Some(id) => id,
            None => uuidv7(None).map_err(|error| SessionManagerError::Io(error.to_string()))?,
        };
        let timestamp = now_iso8601();
        let new_session_file = Path::new(&dir)
            .join(format!(
                "{}_{}.jsonl",
                file_timestamp(&timestamp),
                new_session_id
            ))
            .display()
            .to_string();

        // Write new header pointing to source as parent, with updated cwd.
        let new_header = SessionHeader {
            version: Some(CURRENT_SESSION_VERSION),
            id: new_session_id,
            timestamp,
            cwd: Some(resolved_target_cwd.clone()),
            parent_session: Some(resolved_source_path),
            extras: serde_json::Map::default(),
        };
        let header_line = serde_json::to_string(&FileEntry::Session(new_header))
            .map_err(|error| SessionManagerError::Io(error.to_string()))?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&new_session_file)?;
        writeln!(file, "{header_line}")?;

        // Copy all non-header entries from source.
        for entry in &source_entries {
            if matches!(entry, FileEntry::Session(_)) {
                continue;
            }
            let line = serde_json::to_string(entry)
                .map_err(|error| SessionManagerError::Io(error.to_string()))?;
            writeln!(file, "{line}")?;
        }
        drop(file);

        Self::new(
            &resolved_target_cwd,
            &dir,
            Some(&new_session_file),
            true,
            None,
            None,
        )
    }

    /// List all sessions for a directory, upstream's `SessionManager.list`.
    /// `session_dir` overrides the directory; when it differs from the
    /// default for `cwd`, sessions filter to the cwd they started in.
    ///
    /// # Errors
    /// [`SessionManagerError::Io`] when the default directory cannot be
    /// created, upstream's `getDefaultSessionDir` throw.
    ///
    /// The signature is async like upstream's `list` even though the
    /// sequential load never awaits; the runtime slices await these.
    pub async fn list(
        cwd: &str,
        session_dir: Option<&str>,
        on_progress: Option<SessionListProgress<'_>>,
    ) -> Result<Vec<SessionInfo>, SessionManagerError> {
        let dir = match session_dir {
            Some(session_dir) => normalize_here(session_dir)?,
            None => get_default_session_dir(cwd)?.display().to_string(),
        };
        let filter_cwd = Self::discovery_filters_cwd(session_dir, &dir, cwd);
        let resolved_cwd = resolve_here(cwd);
        let mut sessions = list_sessions_from_dir(&dir, on_progress, 0, None).await;
        if filter_cwd {
            sessions.retain(|session| Self::cwd_matches(&session.cwd, &resolved_cwd));
        }
        sessions.sort_by_key(|session| std::cmp::Reverse(session.modified));
        Ok(sessions)
    }

    /// List all sessions across all project directories, upstream's
    /// `SessionManager.listAll` (no directory override). Symlinked and
    /// broken project directories are tolerated per upstream's discovery.
    #[expect(
        clippy::unused_async,
        clippy::unused_async_trait_impl,
        reason = "upstream's list/listAll are async; the port keeps the signatures the runtime slices await"
    )]
    #[must_use]
    pub async fn list_all(on_progress: Option<SessionListProgress<'_>>) -> Vec<SessionInfo> {
        let sessions_dir = crate::config::get_sessions_dir();

        if !sessions_dir.exists() {
            return Vec::new();
        }
        let Ok(dir_entries) = fs::read_dir(&sessions_dir) else {
            return Vec::new();
        };
        let dirs: Vec<std::path::PathBuf> = dir_entries
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_type()
                    .is_ok_and(|file_type| file_type.is_dir() || file_type.is_symlink())
            })
            .map(|entry| sessions_dir.join(entry.file_name()))
            .collect();

        // Count total files first for accurate progress.
        let mut total_files: u64 = 0;
        let mut dir_files: Vec<Vec<String>> = Vec::new();
        for dir in dirs {
            match fs::read_dir(&dir) {
                Ok(entries) => {
                    let files: Vec<String> = entries
                        .filter_map(Result::ok)
                        .filter(|entry| {
                            entry
                                .path()
                                .extension()
                                .is_some_and(|extension| extension == "jsonl")
                        })
                        .map(|entry| dir.join(entry.file_name()).display().to_string())
                        .collect();
                    total_files += files.len() as u64;
                    dir_files.push(files);
                }
                Err(_) => dir_files.push(Vec::new()),
            }
        }

        // Process all files with progress tracking.
        let mut loaded: u64 = 0;
        let mut sessions: Vec<SessionInfo> = Vec::new();
        for file in dir_files.into_iter().flatten() {
            if let Some(info) = super::discovery::build_session_info(&file) {
                sessions.push(info);
            }
            loaded += 1;
            if let Some(on_progress) = on_progress {
                on_progress(loaded, total_files);
            }
        }

        sessions.sort_by_key(|session| std::cmp::Reverse(session.modified));
        sessions
    }

    /// List all sessions in one directory, upstream's `SessionManager.listAll`
    /// (directory override form).
    ///
    /// # Errors
    /// [`SessionManagerError::PathNormalize`] when the directory does not
    /// normalize.
    ///
    /// The signature is async like upstream's `listAll` even though the
    /// sequential load never awaits; the runtime slices await these.
    pub async fn list_all_from_dir(
        session_dir: &str,
        on_progress: Option<SessionListProgress<'_>>,
    ) -> Result<Vec<SessionInfo>, SessionManagerError> {
        let dir = normalize_here(session_dir)?;
        let mut sessions = list_sessions_from_dir(&dir, on_progress, 0, None).await;
        sessions.sort_by_key(|session| std::cmp::Reverse(session.modified));
        Ok(sessions)
    }
}

/// The `SessionCwdSource` implementation, upstream's `SessionManager`
/// satisfying the `session-cwd` seam (the real source the #118 stub
/// deferred to).
impl crate::session_cwd::SessionCwdSource for SessionManager {
    fn cwd(&self) -> &str {
        &self.cwd
    }

    fn session_file(&self) -> Option<&str> {
        self.session_file.as_deref()
    }
}
