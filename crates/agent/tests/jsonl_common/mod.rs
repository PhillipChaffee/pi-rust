//! Shared fixtures for the JSONL suites, mirroring the shapes upstream's
//! `test/harness/*.test.ts` build: the temp-root filesystem, the
//! publication/line-read observation wrapper, and the cwd-scoped
//! conformance adapter.

#![expect(
    dead_code,
    reason = "shared fixtures; each test binary uses the subset it needs"
)]
#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]
#![expect(
    clippy::expect_used,
    reason = "the fixtures pin shapes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use pi_agent_core::harness::context::Context;
use pi_agent_core::harness::session::jsonl::repo::JsonlSessionRepo;
use pi_agent_core::harness::session::jsonl::types::{
    JsonlSessionCreateOptions, JsonlSessionListOptions, JsonlSessionMetadata,
    JsonlSessionRepoOptions,
};
use pi_agent_core::harness::session::types::{
    ForkOptions, Session, SessionCreateOptions, SessionError, SessionMetadata, SessionRepo,
};
use pi_agent_core::harness::types::{
    CreateDirOptions, FileContent, FileError, FileInfo, FileSystem, ReadTextLinesOptions,
    RemoveOptions, TempFileOptions, TextLine, TextLineReader,
};
use pi_ai::types::BoxedFuture;

/// The clock value the suites pin, upstream's `NOW`.
pub const NOW: i64 = 1_700_000_000_000;

/// The conformance cwd the repo adapter scopes to, upstream's
/// `CONFORMANCE_CWD`.
pub const CONFORMANCE_CWD: &str = "/workspace";

/// The temp root the suites' fixtures live under, upstream's
/// `createTempDir()`; the caller keeps the guard for the test's duration.
pub struct TempRoot {
    _guard: tempfile::TempDir,
    path: String,
}

impl TempRoot {
    #[must_use]
    pub fn new() -> Self {
        let guard = tempfile::tempdir().expect("temp root");
        let path = guard.path().to_string_lossy().to_string();
        Self {
            _guard: guard,
            path,
        }
    }

    /// The root's absolute path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// The publication record the atomic-rename wrapper captures, upstream's
/// `AtomicPublicationNodeExecutionEnv.publication`.
#[derive(Clone, Debug)]
pub struct Publication {
    /// The staged file's path.
    pub source_path: String,
    /// The renamed-to path.
    pub destination_path: String,
    /// Whether the destination existed at rename time.
    pub destination_existed: bool,
    /// The staged content at rename time.
    pub staged_content: String,
}

/// The execution-environment wrapper the suites drive, upstream's
/// `FailableRenameNodeExecutionEnv` + `AtomicPublicationNodeExecutionEnv` +
/// `ObservedEnv` combined: the rename failure flag, the publication record,
/// and the open/line-read/close counters.
#[derive(Debug, Default)]
struct IoCounters {
    line_reads: AtomicU64,
    opens: AtomicU64,
    closes: AtomicU64,
}

pub struct WrappedEnv {
    base: NodeEnv,
    fail_rename: AtomicBool,
    fail_next: Mutex<Option<&'static str>>,
    publication: Mutex<Option<Publication>>,
    counters: Arc<IoCounters>,
}

type NodeEnv = pi_agent_core::harness::env::nodejs::NodeExecutionEnv;

impl std::fmt::Debug for WrappedEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WrappedEnv")
            .field(
                "line_reads",
                &self.counters.line_reads.load(Ordering::Relaxed),
            )
            .field("opens", &self.counters.opens.load(Ordering::Relaxed))
            .field("closes", &self.counters.closes.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl WrappedEnv {
    /// A wrapper over the node environment rooted at `cwd`.
    #[must_use]
    pub fn new(cwd: String) -> Arc<Self> {
        Arc::new(Self {
            base: NodeEnv::new(cwd, None, None),
            fail_rename: AtomicBool::new(false),
            fail_next: Mutex::new(None),
            publication: Mutex::new(None),
            counters: Arc::new(IoCounters::default()),
        })
    }

    /// The rename failure flag, upstream's `failRename`.
    pub fn set_fail_rename(&self, fail: bool) {
        self.fail_rename.store(fail, Ordering::Release);
    }

    /// Inject one failure into the next named call, upstream's
    /// `vi.spyOn(fileSystem, method).mockResolvedValueOnce(err(failure))`.
    pub fn fail_next(&self, method: &'static str) {
        *self.fail_next.lock().expect("fail-next lock") = Some(method);
    }

    /// The captured publication, upstream's `fileSystem.publication`.
    #[must_use]
    pub fn publication(&self) -> Option<Publication> {
        self.publication.lock().expect("publication lock").clone()
    }

    /// The line-read counter, upstream's `reader.lineReads`.
    #[must_use]
    pub fn line_reads(&self) -> u64 {
        self.counters.line_reads.load(Ordering::Acquire)
    }

    /// Reset the line-read counter.
    pub fn reset_line_reads(&self) {
        self.counters.line_reads.store(0, Ordering::Release);
    }

    /// The open counter.
    #[must_use]
    pub fn opens(&self) -> u64 {
        self.counters.opens.load(Ordering::Acquire)
    }

    /// The close counter.
    #[must_use]
    pub fn closes(&self) -> u64 {
        self.counters.closes.load(Ordering::Acquire)
    }

    fn take_failure(&self, method: &'static str) -> Option<FileError> {
        let mut fail_next = self.fail_next.lock().expect("fail-next lock");
        if fail_next.is_some_and(|queued| queued == method) {
            *fail_next = None;
            drop(fail_next);
            return Some(FileError {
                code: pi_agent_core::harness::types::FileErrorCode::Unknown,
                message: "injected I/O failure".to_owned(),
                path: None,
                source: None,
            });
        }
        None
    }
}

impl FileSystem for WrappedEnv {
    fn cwd(&self) -> &str {
        self.base.cwd()
    }

    fn absolute_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        self.base.absolute_path(path, context)
    }

    fn join_path<'a>(
        &'a self,
        parts: &'a [String],
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        self.base.join_path(parts, context)
    }

    fn read_text_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        self.base.read_text_file(path, context)
    }

    fn open_text_line_reader<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Box<dyn TextLineReader>, FileError>> {
        Box::pin(async move {
            if let Some(failure) = self.take_failure("openTextLineReader") {
                return Err(failure);
            }
            let opened = self.base.open_text_line_reader(path, context).await?;
            self.counters.opens.fetch_add(1, Ordering::AcqRel);
            let observed: Box<dyn TextLineReader> = Box::new(ObservedReader {
                inner: opened,
                counters: self.counters.clone(),
            });
            Ok(observed)
        })
    }

    fn read_text_lines<'a>(
        &'a self,
        path: &'a str,
        options: Option<ReadTextLinesOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<String>, FileError>> {
        self.base.read_text_lines(path, options, context)
    }

    fn read_binary_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<u8>, FileError>> {
        self.base.read_binary_file(path, context)
    }

    fn write_file<'a>(
        &'a self,
        path: &'a str,
        payload: FileContent,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        Box::pin(async move {
            if let Some(failure) = self.take_failure("writeFile") {
                return Err(failure);
            }
            self.base.write_file(path, payload, context).await
        })
    }

    fn append_file<'a>(
        &'a self,
        path: &'a str,
        payload: FileContent,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        Box::pin(async move {
            if let Some(failure) = self.take_failure("appendFile") {
                return Err(failure);
            }
            self.base.append_file(path, payload, context).await
        })
    }

    fn rename_file<'a>(
        &'a self,
        source_path: &'a str,
        destination_path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        Box::pin(async move {
            if self.fail_rename.load(Ordering::Acquire) {
                return Err(FileError {
                    code: pi_agent_core::harness::types::FileErrorCode::Unknown,
                    message: "Injected rename failure".to_owned(),
                    path: Some(source_path.to_owned()),
                    source: None,
                });
            }
            if let Some(failure) = self.take_failure("renameFile") {
                return Err(failure);
            }
            let destination_existed = self
                .base
                .exists(destination_path, context)
                .await
                .unwrap_or(false);
            if let Ok(staged_content) = self.base.read_text_file(source_path, context).await {
                *self.publication.lock().expect("publication lock") = Some(Publication {
                    source_path: source_path.to_owned(),
                    destination_path: destination_path.to_owned(),
                    destination_existed,
                    staged_content,
                });
            }
            self.base
                .rename_file(source_path, destination_path, context)
                .await
        })
    }

    fn file_info<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<FileInfo, FileError>> {
        self.base.file_info(path, context)
    }

    fn list_dir<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<FileInfo>, FileError>> {
        self.base.list_dir(path, context)
    }

    fn canonical_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        self.base.canonical_path(path, context)
    }

    fn exists<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<bool, FileError>> {
        self.base.exists(path, context)
    }

    fn create_dir<'a>(
        &'a self,
        path: &'a str,
        options: Option<CreateDirOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        self.base.create_dir(path, options, context)
    }

    fn remove<'a>(
        &'a self,
        path: &'a str,
        options: Option<RemoveOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        self.base.remove(path, options, context)
    }

    fn create_temp_dir<'a>(
        &'a self,
        prefix: Option<&'a str>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        self.base.create_temp_dir(prefix, context)
    }

    fn create_temp_file<'a>(
        &'a self,
        options: Option<TempFileOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        self.base.create_temp_file(options, context)
    }

    fn cleanup<'a>(&'a self, context: &'a Context) -> BoxedFuture<'a, ()> {
        self.base.cleanup(context)
    }
}

/// The line-reader wrapper counting reads and closes, upstream's
/// `ObservedEnv`'s wrapped reader.
struct ObservedReader {
    inner: Box<dyn TextLineReader>,
    counters: Arc<IoCounters>,
}

impl TextLineReader for ObservedReader {
    fn read_line<'a>(
        &'a mut self,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Option<TextLine>, FileError>> {
        self.counters.line_reads.fetch_add(1, Ordering::AcqRel);
        self.inner.read_line(context)
    }

    fn close<'a>(&'a mut self, context: &'a Context) -> BoxedFuture<'a, ()> {
        self.counters.closes.fetch_add(1, Ordering::AcqRel);
        self.inner.close(context)
    }
}

/// The repo the jsonl suites build, upstream's
/// `new JsonlSessionRepo({ fileSystem, sessionsRoot: "sessions", now })`.
pub fn jsonl_repo(file_system: Arc<dyn FileSystem>) -> JsonlSessionRepo {
    JsonlSessionRepo::new(JsonlSessionRepoOptions {
        file_system,
        sessions_root: "sessions".to_owned(),
        now: Some(Arc::new(move || NOW)),
    })
}

/// The create options one conformance call carries, upstream's
/// `{ ...options, cwd: CONFORMANCE_CWD }`.
#[must_use]
pub fn conformance_create_options(
    id: Option<&str>,
    parent_session_id: Option<&str>,
) -> SessionCreateOptions {
    SessionCreateOptions {
        id: id.map(str::to_owned),
        parent_session_id: parent_session_id.map(str::to_owned),
    }
}

/// The cwd-scoped adapter the repo conformance runs through, upstream's
/// `createConformanceRepo`: the typed repo's lifecycle with the conformance
/// cwd applied and the typed metadata tracked for the erased calls.
pub struct CwdScopedRepo {
    pub repo: JsonlSessionRepo,
    sessions: Mutex<BTreeMap<String, JsonlSessionMetadata>>,
}

impl SessionRepo for CwdScopedRepo {
    fn create(
        &self,
        options: SessionCreateOptions,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Session>, SessionError>> {
        let context = context.clone();
        Box::pin(async move {
            let (session, metadata) = self
                .repo
                .create(
                    JsonlSessionCreateOptions {
                        id: options.id,
                        parent_session_id: options.parent_session_id,
                        cwd: CONFORMANCE_CWD.to_owned(),
                    },
                    &context,
                )
                .await?;
            self.sessions
                .lock()
                .expect("tracked sessions")
                .insert(metadata.id.clone(), metadata);
            Ok(session)
        })
    }

    fn open(
        &self,
        metadata: &SessionMetadata,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Session>, SessionError>> {
        let context = context.clone();
        let id = metadata.id.clone();
        Box::pin(async move {
            let typed = self
                .sessions
                .lock()
                .expect("tracked sessions")
                .get(&id)
                .cloned()
                .ok_or_else(|| SessionError::Message(format!("Unknown session: {id}")))?;
            let (session, _) = self.repo.open(&typed, &context).await?;
            Ok(session)
        })
    }

    fn list(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<SessionMetadata>, SessionError>> {
        let context = context.clone();
        Box::pin(async move {
            let typed = self
                .repo
                .list(
                    Some(JsonlSessionListOptions {
                        cwd: Some(CONFORMANCE_CWD.to_owned()),
                    }),
                    &context,
                )
                .await?;
            Ok(typed
                .into_iter()
                .map(|metadata| {
                    self.sessions
                        .lock()
                        .expect("tracked sessions")
                        .insert(metadata.id.clone(), metadata.clone());
                    metadata.base()
                })
                .collect())
        })
    }

    fn delete(
        &self,
        metadata: &SessionMetadata,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        let context = context.clone();
        let id = metadata.id.clone();
        Box::pin(async move {
            let typed = self
                .sessions
                .lock()
                .expect("tracked sessions")
                .get(&id)
                .cloned()
                .ok_or_else(|| SessionError::Message(format!("Unknown session: {id}")))?;
            self.repo.delete(&typed, &context).await
        })
    }

    fn fork(
        &self,
        source: &SessionMetadata,
        options: &ForkOptions,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Session>, SessionError>> {
        let context = context.clone();
        let id = source.id.clone();
        let options = options.clone();
        Box::pin(async move {
            let typed = self
                .sessions
                .lock()
                .expect("tracked sessions")
                .get(&id)
                .cloned()
                .ok_or_else(|| SessionError::Message(format!("Unknown session: {id}")))?;
            let (session, metadata) = self.repo.fork(&typed, &options, &context).await?;
            self.sessions
                .lock()
                .expect("tracked sessions")
                .insert(metadata.id.clone(), metadata);
            Ok(session)
        })
    }
}

impl CwdScopedRepo {
    /// The adapter over one repo with the conformance cwd.
    #[must_use]
    pub const fn new(repo: JsonlSessionRepo) -> Self {
        Self {
            repo,
            sessions: Mutex::new(BTreeMap::new()),
        }
    }

    /// Close the wrapped repo, upstream's `jsonlRepo.close` onClose hook.
    pub async fn close(&self, context: &Context) {
        let _ = self.repo.close(context).await;
    }
}
