//! The file-backed format-4 session repository lifecycle, ported from
//! upstream `src/harness/session/jsonl/repo.ts`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, PoisonError};

use crate::harness::context::Context;
use crate::harness::session::jsonl::codec::{JsonlParsedSessionHeader, parse_jsonl_session_header};
use crate::harness::session::jsonl::fork::{
    JsonlForkInput, JsonlForkRun, JsonlForkSourceMetadata, run_jsonl_fork,
};
use crate::harness::session::jsonl::io::file_value;
use crate::harness::session::jsonl::legacy_v3::{
    LegacyV3Source, jsonl_metadata_from_base, metadata_from_legacy_v3_header,
};
use crate::harness::session::jsonl::storage::JsonlStorage;
use crate::harness::session::jsonl::types::{
    JSONL_FORMAT_VERSION, JSONL_STORAGE_VERSION, JsonlSessionCreateOptions,
    JsonlSessionListOptions, JsonlSessionMetadata, JsonlSessionRepoOptions, JsonlStorageHeader,
    JsonlStorageOptions,
};
use crate::harness::session::memory::default_now;
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::types::{ForkOptions, Session, SessionError, Storage};
use crate::harness::types::{CreateDirOptions, FileSystem, ReadTextLinesOptions};

/// Builds the typed metadata one format-4 header carries, upstream's
/// `metadataFromHeader`.
#[must_use]
pub fn metadata_from_header(
    header: &JsonlStorageHeader,
    path: String,
    modified_at: i64,
) -> JsonlSessionMetadata {
    JsonlSessionMetadata {
        id: header.id.clone(),
        created_at: header.created_at,
        storage_version: header.storage_version,
        cwd: header.cwd.clone(),
        path,
        modified_at,
        parent_session_id: header.parent_session_id.clone(),
        legacy_parent_session_path: header.legacy_parent_session_path.clone(),
    }
}

/// The encoded directory name one cwd's sessions live under, upstream's
/// `sessionDirectoryName`.
///
/// `--` + the cwd with its leading separator and separators/colons replaced
/// by dashes + `--`. The encoding is lossy: `/a/b` and `/a-b` both map to
/// `--a-b--`.
#[must_use]
pub fn session_directory_name(cwd: &str) -> String {
    let trimmed = cwd.trim_start_matches(['/', '\\']);
    let encoded: String = trimmed
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' => '-',
            other => other,
        })
        .collect();
    format!("--{encoded}--")
}

/// The percent-encoding one JS `encodeURIComponent` produces; the session
/// filename's id segment.
fn js_uri_component(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        let keep = byte.is_ascii_alphanumeric()
            || matches!(
                byte,
                b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')'
            );
        if keep {
            encoded.push(byte as char);
        } else {
            let _ = std::fmt::Write::write_fmt(&mut encoded, format_args!("%{byte:02X}"));
        }
    }
    encoded
}

/// The session file's name, upstream's `sessionFileName`: the ISO timestamp
/// with its separators replaced by dashes, the uri-encoded id.
///
/// # Panics
/// Never: the timestamp falls back to the epoch when it does not parse.
#[must_use]
pub fn session_file_name(created_at: i64, id: &str) -> String {
    let timestamp =
        crate::harness::session::jsonl::codec::format_iso8601(created_at).replace([':', '.'], "-");
    format!("{timestamp}_{}.jsonl", js_uri_component(id))
}

/// File-backed format-4 session repository lifecycle, upstream's
/// `JsonlSessionRepo`.
///
/// The repository's typed lifecycle carries [`JsonlSessionMetadata`]; the
/// erased session contract rides the base fields through
/// [`StorageBackedSession`]. Repository close marks the repo closed only:
/// whether repository close should close session handles is an open
/// ownership question recorded upstream; the port keeps the same
/// hands-off behavior.
pub struct JsonlSessionRepo {
    file_system: Arc<dyn FileSystem>,
    sessions_root_input: String,
    now: crate::harness::session::memory::NowFn,
    open_sessions: Arc<Mutex<BTreeMap<String, Arc<JsonlStorage>>>>,
    pending_creates: Mutex<BTreeSet<String>>,
    closed: std::sync::atomic::AtomicBool,
    close_cell: tokio::sync::OnceCell<()>,
}

impl std::fmt::Debug for JsonlSessionRepo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonlSessionRepo")
            .field("sessions_root", &self.sessions_root_input)
            .finish_non_exhaustive()
    }
}

/// The destination identity one fork publishes under, folded so the fork
/// path stays under the arity gate.
struct ForkDestination {
    /// The forked session's working directory.
    cwd: String,
    /// The forked session's id.
    id: String,
    /// The fork's creation timestamp.
    created_at: i64,
    /// The open-registry key.
    key: String,
}

impl JsonlSessionRepo {
    /// A repo over the filesystem capability, upstream's constructor.
    #[must_use]
    pub fn new(options: JsonlSessionRepoOptions) -> Self {
        Self {
            file_system: options.file_system,
            sessions_root_input: options.sessions_root,
            now: options.now.unwrap_or_else(default_now),
            open_sessions: Arc::new(Mutex::new(BTreeMap::new())),
            pending_creates: Mutex::new(BTreeSet::new()),
            closed: std::sync::atomic::AtomicBool::new(false),
            close_cell: tokio::sync::OnceCell::new(),
        }
    }

    fn assert_open(&self) -> Result<(), SessionError> {
        if self.closed.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SessionError::Message(
                "JsonlSessionRepo is closed".to_owned(),
            ));
        }
        Ok(())
    }

    fn session_key(cwd: &str, id: &str) -> String {
        format!("{cwd}\u{0}{id}")
    }

    /// Create a session, upstream's `create`.
    ///
    /// # Errors
    /// A `SessionError::Message` when the repo is closed, the id is taken,
    /// or the file operations fail.
    pub async fn create(
        &self,
        options: JsonlSessionCreateOptions,
        context: &Context,
    ) -> Result<(Box<dyn Session>, JsonlSessionMetadata), SessionError> {
        self.assert_open()?;
        let created_at = (self.now)();
        let destination_id = options
            .id
            .clone()
            .map_or_else(|| Self::generated_id(created_at), Ok)?;
        let cwd = file_value(
            self.file_system.absolute_path(&options.cwd, context).await,
            &format!("Failed to resolve session cwd {}", options.cwd),
        )
        .map_err(SessionError::from)?;
        let id = destination_id;
        let key = Self::session_key(&cwd, &id);
        self.assert_key_available(&key, &id)?;
        self.pending_creates
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key.clone());
        let result = self
            .create_inner(
                &cwd,
                &id,
                created_at,
                options.parent_session_id.clone(),
                context,
            )
            .await;
        self.pending_creates
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&key);
        result
    }

    async fn create_inner(
        &self,
        cwd: &str,
        id: &str,
        created_at: i64,
        parent_session_id: Option<String>,
        context: &Context,
    ) -> Result<(Box<dyn Session>, JsonlSessionMetadata), SessionError> {
        let key = Self::session_key(cwd, id);
        let path = self
            .resolve_new_session_path(cwd, created_at, id, context)
            .await?;
        let header = JsonlStorageHeader {
            v: JSONL_FORMAT_VERSION,
            kind: "header".to_owned(),
            id: id.to_owned(),
            storage_version: JSONL_STORAGE_VERSION,
            created_at,
            cwd: cwd.to_owned(),
            parent_session_id,
            legacy_parent_session_path: None,
            next_seq: None,
        };
        let storage_result = JsonlStorage::create(
            &self.storage_options(&path),
            header.clone(),
            Vec::new(),
            context,
        )
        .await;
        let storage = match storage_result {
            Ok(storage) => storage,
            Err(error) => return Err(self.discard_partial_session(&path, context, error).await),
        };
        let info = file_value(
            self.file_system.file_info(&path, context).await,
            &format!("Failed to read session {path}"),
        )
        .map_err(SessionError::from)?;
        let metadata = metadata_from_header(&header, path.clone(), info.mtime_ms);
        let storage = Arc::new(storage);
        let session = self.publish_open_session(&metadata, storage, &key)?;
        Ok((session, metadata))
    }

    /// Open a previously created session, upstream's `open`.
    ///
    /// # Errors
    /// A `SessionError::Message` when the repo is closed, the session is
    /// already open, the file is missing, the identity mismatches, or the
    /// storage version is unsupported.
    pub async fn open(
        &self,
        metadata: &JsonlSessionMetadata,
        context: &Context,
    ) -> Result<(Box<dyn Session>, JsonlSessionMetadata), SessionError> {
        self.assert_open()?;
        let key = Self::session_key(&metadata.cwd, &metadata.id);
        if self
            .open_sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains_key(&key)
        {
            return Err(SessionError::Message(format!(
                "Session is already open: {}",
                metadata.id
            )));
        }
        let storage = self.load_storage(metadata, context).await?;
        let storage = Arc::new(storage);
        let session = self.publish_open_session(metadata, storage, &key)?;
        Ok((session, metadata.clone()))
    }

    /// List the sessions, upstream's `list`: the cwd-filtered or global
    /// discovery, newest first.
    ///
    /// # Errors
    /// A `SessionError::Message` when the repo is closed or the filesystem
    /// discovery fails.
    pub async fn list(
        &self,
        options: Option<JsonlSessionListOptions>,
        context: &Context,
    ) -> Result<Vec<JsonlSessionMetadata>, SessionError> {
        self.assert_open()?;
        let cwd = match options.unwrap_or_default().cwd {
            None => None,
            Some(cwd) => Some(
                file_value(
                    self.file_system.absolute_path(&cwd, context).await,
                    &format!("Failed to resolve session cwd {cwd}"),
                )
                .map_err(SessionError::from)?,
            ),
        };
        let root = self.root(context).await?;
        if !file_value(
            self.file_system.exists(&root, context).await,
            &format!("Failed to check sessions root {root}"),
        )
        .map_err(SessionError::from)?
        {
            return Ok(Vec::new());
        }
        let directories = match &cwd {
            None => self.session_directories(&root, context).await?,
            Some(cwd) => vec![self.session_directory(cwd, context).await?],
        };
        let mut metadata: Vec<JsonlSessionMetadata> = Vec::new();
        for directory in &directories {
            metadata.extend(
                self.list_directory(directory, cwd.as_deref(), context)
                    .await?,
            );
        }
        metadata.sort_by(|left, right| {
            right
                .created_at
                .cmp(&left.created_at)
                .then_with(|| left.id.cmp(&right.id))
                .then_with(|| left.cwd.cmp(&right.cwd))
        });
        Ok(metadata)
    }

    /// Delete a closed session's file, upstream's `delete`.
    ///
    /// # Errors
    /// A `SessionError::Message` when the repo is closed, the session is
    /// open, or the file does not exist.
    pub async fn delete(
        &self,
        metadata: &JsonlSessionMetadata,
        context: &Context,
    ) -> Result<(), SessionError> {
        self.assert_open()?;
        let key = Self::session_key(&metadata.cwd, &metadata.id);
        if self
            .open_sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains_key(&key)
        {
            return Err(SessionError::Message(format!(
                "Session is open: {}",
                metadata.id
            )));
        }
        if !file_value(
            self.file_system.exists(&metadata.path, context).await,
            &format!("Failed to check session {}", metadata.path),
        )
        .map_err(SessionError::from)?
        {
            return Err(SessionError::Message(format!(
                "Session file does not exist: {}",
                metadata.path
            )));
        }
        file_value(
            self.file_system.remove(&metadata.path, None, context).await,
            &format!("Failed to delete session {}", metadata.path),
        )
        .map_err(SessionError::from)
    }

    /// Fork a source session into a new file, upstream's `fork`.
    ///
    /// # Errors
    /// A `SessionError::Message` when the repo is closed, the destination id
    /// is taken, the source is an open legacy v3 session, or the fork
    /// machinery fails.
    pub async fn fork(
        &self,
        source: &JsonlSessionMetadata,
        options: &ForkOptions,
        context: &Context,
    ) -> Result<(Box<dyn Session>, JsonlSessionMetadata), SessionError> {
        self.assert_open()?;
        let created_at = (self.now)();
        let cwd = source.cwd.clone();
        let id = options
            .id()
            .cloned()
            .map_or_else(|| Self::generated_id(created_at), Ok)?;
        let destination_key = Self::session_key(&cwd, &id);
        self.assert_key_available(&destination_key, &id)?;
        self.pending_creates
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(destination_key.clone());
        let destination = ForkDestination {
            cwd: cwd.clone(),
            id: id.clone(),
            created_at,
            key: destination_key.clone(),
        };
        let result = self
            .fork_inner(source, options, &destination, context)
            .await;
        self.pending_creates
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&destination_key);
        result
    }

    async fn fork_inner(
        &self,
        source: &JsonlSessionMetadata,
        options: &ForkOptions,
        destination: &ForkDestination,
        context: &Context,
    ) -> Result<(Box<dyn Session>, JsonlSessionMetadata), SessionError> {
        let source_storage = self
            .open_sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&Self::session_key(&source.cwd, &source.id))
            .cloned();
        let input = self
            .resolve_fork_input(source, source_storage.as_ref(), context)
            .await?;
        let path = self
            .resolve_new_session_path(
                &destination.cwd,
                destination.created_at,
                &destination.id,
                context,
            )
            .await?;
        let header = JsonlStorageHeader {
            v: JSONL_FORMAT_VERSION,
            kind: "header".to_owned(),
            id: destination.id.clone(),
            storage_version: JSONL_STORAGE_VERSION,
            created_at: destination.created_at,
            cwd: destination.cwd.clone(),
            parent_session_id: Some(source.id.clone()),
            legacy_parent_session_path: None,
            next_seq: None,
        };
        let published = async {
            run_jsonl_fork(
                JsonlForkRun {
                    input,
                    file_system: self.file_system.clone(),
                    destination_path: path.clone(),
                    destination_header: header.clone(),
                    fork: options.clone(),
                },
                context,
            )
            .await
            .map_err(SessionError::from)?;
            let storage = JsonlStorage::open(&self.storage_options(&path), context).await?;
            let info = file_value(
                self.file_system.file_info(&path, context).await,
                &format!("Failed to read session {path}"),
            )
            .map_err(SessionError::from)?;
            Ok::<(JsonlStorage, i64), SessionError>((storage, info.mtime_ms))
        }
        .await;
        let (storage, mtime_ms) = match published {
            Ok(published) => published,
            Err(error) => return Err(self.discard_partial_session(&path, context, error).await),
        };
        let storage = Arc::new(storage);
        let metadata = metadata_from_header(&header, path.clone(), mtime_ms);
        let session = self.publish_open_session(&metadata, storage, &destination.key)?;
        Ok((session, metadata))
    }

    /// Mark the repo closed, upstream's `close`.
    ///
    /// # Errors
    /// Never.
    pub async fn close(&self, _context: &Context) -> Result<(), SessionError> {
        if self.closed.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return Ok(());
        }
        self.close_cell.get_or_init(|| async {}).await;
        Ok(())
    }

    /// Rejects a create/fork whose cwd+id key is already open or reserved,
    /// upstream's `assertSessionIdAvailable`'s in-memory half.
    ///
    /// # Errors
    /// The `Session already exists` message.
    fn assert_key_available(&self, key: &str, id: &str) -> Result<(), SessionError> {
        let taken = {
            let open = self
                .open_sessions
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            open.contains_key(key)
        };
        let reserved = {
            let pending = self
                .pending_creates
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            pending.contains(key)
        };
        if taken || reserved {
            return Err(SessionError::Message(format!(
                "Session already exists: {id}"
            )));
        }
        Ok(())
    }

    /// Force-removes a session file whose create/fork failed partway,
    /// upstream's catch's `remove(..., { force: true })`.
    async fn discard_partial_session(
        &self,
        path: &str,
        context: &Context,
        error: SessionError,
    ) -> SessionError {
        let _ = self
            .file_system
            .remove(
                path,
                Some(crate::harness::types::RemoveOptions {
                    force: Some(true),
                    ..Default::default()
                }),
                context,
            )
            .await;
        error
    }

    /// The storage options one path's backend reads through.
    fn storage_options(&self, path: &str) -> JsonlStorageOptions {
        JsonlStorageOptions {
            file_system: self.file_system.clone(),
            path: path.to_owned(),
            now: Some(self.now.clone()),
        }
    }

    async fn root(&self, context: &Context) -> Result<String, SessionError> {
        file_value(
            self.file_system
                .absolute_path(&self.sessions_root_input, context)
                .await,
            &format!(
                "Failed to resolve sessions root {}",
                self.sessions_root_input
            ),
        )
        .map_err(SessionError::from)
    }

    async fn session_directories(
        &self,
        root: &str,
        context: &Context,
    ) -> Result<Vec<String>, SessionError> {
        let entries = file_value(
            self.file_system.list_dir(root, context).await,
            &format!("Failed to list sessions root {root}"),
        )
        .map_err(SessionError::from)?;
        Ok(entries
            .into_iter()
            .filter(|entry| entry.kind == crate::harness::types::FileKind::Directory)
            .map(|entry| entry.path)
            .collect())
    }

    async fn session_directory(
        &self,
        cwd: &str,
        context: &Context,
    ) -> Result<String, SessionError> {
        let root = self.root(context).await?;
        file_value(
            self.file_system
                .join_path(&[root, session_directory_name(cwd)], context)
                .await,
            &format!("Failed to resolve sessions directory for {cwd}"),
        )
        .map_err(SessionError::from)
    }

    async fn resolve_new_session_path(
        &self,
        cwd: &str,
        created_at: i64,
        id: &str,
        context: &Context,
    ) -> Result<String, SessionError> {
        let directory = self.session_directory(cwd, context).await?;
        self.assert_session_id_available(&directory, id, context)
            .await?;
        file_value(
            self.file_system
                .create_dir(&directory, Some(CreateDirOptions::default()), context)
                .await,
            &format!("Failed to create sessions directory {directory}"),
        )
        .map_err(SessionError::from)?;
        file_value(
            self.file_system
                .join_path(&[directory, session_file_name(created_at, id)], context)
                .await,
            &format!("Failed to resolve path for session {id}"),
        )
        .map_err(SessionError::from)
    }

    async fn assert_session_id_available(
        &self,
        directory: &str,
        id: &str,
        context: &Context,
    ) -> Result<(), SessionError> {
        if !file_value(
            self.file_system.exists(directory, context).await,
            &format!("Failed to check sessions directory {directory}"),
        )
        .map_err(SessionError::from)?
        {
            return Ok(());
        }
        let suffix = format!("_{}.jsonl", js_uri_component(id));
        let entries = file_value(
            self.file_system.list_dir(directory, context).await,
            &format!("Failed to list sessions directory {directory}"),
        )
        .map_err(SessionError::from)?;
        if entries.iter().any(|entry| {
            entry.kind != crate::harness::types::FileKind::Directory
                && entry.name.ends_with(&suffix)
        }) {
            return Err(SessionError::Message(format!(
                "Session already exists: {id}"
            )));
        }
        Ok(())
    }

    #[expect(
        clippy::case_sensitive_file_extension_comparisons,
        reason = "upstream matches endsWith('.jsonl') case-sensitively"
    )]
    async fn list_directory(
        &self,
        directory: &str,
        cwd: Option<&str>,
        context: &Context,
    ) -> Result<Vec<JsonlSessionMetadata>, SessionError> {
        if !file_value(
            self.file_system.exists(directory, context).await,
            &format!("Failed to check sessions directory {directory}"),
        )
        .map_err(SessionError::from)?
        {
            return Ok(Vec::new());
        }
        let files = file_value(
            self.file_system.list_dir(directory, context).await,
            &format!("Failed to list sessions directory {directory}"),
        )
        .map_err(SessionError::from)?
        .into_iter()
        .filter(|file| {
            file.kind != crate::harness::types::FileKind::Directory
                // Upstream matches `endsWith(".jsonl")` case-sensitively.
                && file.name.ends_with(".jsonl")
        })
        .collect::<Vec<_>>();
        let mut metadata: Vec<JsonlSessionMetadata> = Vec::new();
        for file in files {
            let Some(discovered) = self.read_session_metadata(&file, context).await? else {
                continue;
            };
            // Directory encoding is lossy: /a/b and /a-b both map to --a-b--.
            if cwd.is_none_or(|cwd| discovered.cwd == cwd) {
                metadata.push(discovered);
            }
        }
        Ok(metadata)
    }

    async fn read_session_metadata(
        &self,
        file: &crate::harness::types::FileInfo,
        context: &Context,
    ) -> Result<Option<JsonlSessionMetadata>, SessionError> {
        let lines = file_value(
            self.file_system
                .read_text_lines(
                    &file.path,
                    Some(ReadTextLinesOptions { max_lines: Some(1) }),
                    context,
                )
                .await,
            &format!("Failed to read session header {}", file.path),
        )
        .map_err(SessionError::from)?;
        let Some(first_line) = lines.first() else {
            return Ok(None);
        };
        let Ok(parsed) = parse_jsonl_session_header(first_line) else {
            return Ok(None);
        };
        match parsed {
            JsonlParsedSessionHeader::V3Legacy(header) => {
                let base =
                    metadata_from_legacy_v3_header(self.file_system.as_ref(), &header, context)
                        .await;
                Ok(Some(jsonl_metadata_from_base(
                    base,
                    file.path.clone(),
                    file.mtime_ms,
                )))
            }
            JsonlParsedSessionHeader::V4(header) => Ok(Some(metadata_from_header(
                &header,
                file.path.clone(),
                file.mtime_ms,
            ))),
        }
    }

    async fn resolve_fork_input(
        &self,
        source: &JsonlSessionMetadata,
        storage: Option<&Arc<JsonlStorage>>,
        context: &Context,
    ) -> Result<JsonlForkInput, SessionError> {
        if let Some(storage) = storage {
            if storage.is_legacy_v3() {
                return Err(SessionError::Message(
                    "Cannot fork an open legacy v3 JSONL session; commit a non-empty transaction to upgrade it to format 4 first"
                        .to_owned(),
                ));
            }
            let next_seq = storage.capture_fork_next_seq().await?;
            return Ok(JsonlForkInput::Open {
                metadata: JsonlForkSourceMetadata {
                    id: source.id.clone(),
                    cwd: source.cwd.clone(),
                    path: source.path.clone(),
                },
                next_seq,
            });
        }
        if self.is_legacy_v3_fork_source(source, context).await? {
            let normalized =
                LegacyV3Source::read(self.file_system.clone(), &source.path, context).await?;
            if normalized.header.id != source.id || normalized.header.cwd != source.cwd {
                return Err(SessionError::Message(format!(
                    "Session identity does not match header: {}",
                    source.id
                )));
            }
            return Ok(JsonlForkInput::LegacyV3(Arc::new(normalized)));
        }
        Ok(JsonlForkInput::Closed {
            metadata: JsonlForkSourceMetadata {
                id: source.id.clone(),
                cwd: source.cwd.clone(),
                path: source.path.clone(),
            },
        })
    }

    async fn is_legacy_v3_fork_source(
        &self,
        source: &JsonlSessionMetadata,
        context: &Context,
    ) -> Result<bool, SessionError> {
        let lines = file_value(
            self.file_system
                .read_text_lines(
                    &source.path,
                    Some(ReadTextLinesOptions { max_lines: Some(1) }),
                    context,
                )
                .await,
            &format!("Failed to read session header {}", source.path),
        )
        .map_err(SessionError::from)?;
        let Some(first_line) = lines.first() else {
            return Ok(false);
        };
        Ok(matches!(
            parse_jsonl_session_header(first_line),
            Ok(JsonlParsedSessionHeader::V3Legacy(_))
        ))
    }

    fn publish_open_session(
        &self,
        metadata: &JsonlSessionMetadata,
        storage: Arc<JsonlStorage>,
        key: &str,
    ) -> Result<Box<dyn Session>, SessionError> {
        {
            let open = self
                .open_sessions
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if open.contains_key(key) {
                return Err(SessionError::Message(format!(
                    "Session is already open: {}",
                    metadata.id
                )));
            }
        }
        let open_sessions = self.open_sessions.clone();
        let key_for_close = key.to_owned();
        let session_storage = storage.clone();
        let session = StorageBackedSession::new(
            metadata.base(),
            storage.clone(),
            StorageBackedSessionOptions {
                on_close: Some(Arc::new(move || {
                    let mut open = open_sessions.lock().unwrap_or_else(PoisonError::into_inner);
                    if open
                        .get(&key_for_close)
                        .is_some_and(|current| Arc::ptr_eq(current, &session_storage))
                    {
                        open.remove(&key_for_close);
                    }
                })),
                ..Default::default()
            },
        );
        self.open_sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key.to_owned(), storage);
        Ok(Box::new(session))
    }

    async fn load_storage(
        &self,
        metadata: &JsonlSessionMetadata,
        context: &Context,
    ) -> Result<JsonlStorage, SessionError> {
        if !file_value(
            self.file_system.exists(&metadata.path, context).await,
            &format!("Failed to check session {}", metadata.path),
        )
        .map_err(SessionError::from)?
        {
            return Err(SessionError::Message(format!(
                "Session file does not exist: {}",
                metadata.path
            )));
        }
        let storage = JsonlStorage::open(
            &JsonlStorageOptions {
                file_system: self.file_system.clone(),
                path: metadata.path.clone(),
                now: Some(self.now.clone()),
            },
            context,
        )
        .await?;
        if storage.header.id != metadata.id || storage.header.cwd != metadata.cwd {
            storage.close(context).await?;
            return Err(SessionError::Message(format!(
                "Session identity does not match header: {}",
                metadata.id
            )));
        }
        if storage.header.storage_version != JSONL_STORAGE_VERSION {
            let version = storage.header.storage_version;
            storage.close(context).await?;
            return Err(SessionError::Message(format!(
                "Session {} uses unsupported storage version {version}",
                metadata.id
            )));
        }
        Ok(storage)
    }

    fn generated_id(created_at: i64) -> Result<String, SessionError> {
        pi_ai::utils::uuid::uuidv7(u64::try_from(created_at).ok())
            .map_err(|error| SessionError::Message(error.to_string()))
    }
}
