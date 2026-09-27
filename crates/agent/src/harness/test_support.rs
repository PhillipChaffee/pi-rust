//! Shared fixtures for the harness loader suites.
//!
//! Upstream's `test/harness/session-test-utils.ts` hands the suites a temp
//! dir and vitest supplies the rest; this module carries the same plumbing
//! for the ported suites: the nodejs execution environment over a tempdir
//! root, the file and directory writers, the absolute-path formatter the
//! path assertions use, and the provenance shape the sourced tests carry.

#![allow(
    dead_code,
    reason = "these helpers serve the ported test files; only the modules implemented so far reference them"
)]
#![expect(
    clippy::expect_used,
    reason = "the fixtures pin outcomes; an unexpected result panics the test by design"
)]

use std::path::Path;

use crate::harness::context::Context;
use crate::harness::env::nodejs::NodeExecutionEnv;
use crate::harness::types::{
    CreateDirOptions, ExecutionEnv, FileContent, FileError, FileErrorCode, FileInfo, FileSystem,
    ReadTextLinesOptions, RemoveOptions, Shell, ShellExecOptions, ShellExecResult, TempFileOptions,
    TextLineReader,
};

/// The provenance shape the sourced tests carry, upstream's
/// `{ type: "user" as const }` / `{ type: "project" as const }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TestSource {
    /// A user-level source directory.
    User,
    /// A project-level source directory.
    Project,
}

/// The execution environment over one tempdir root, upstream's
/// `new NodeExecutionEnv({ cwd: root })`.
pub(crate) fn env_for(root: &Path) -> NodeExecutionEnv {
    NodeExecutionEnv::new(root.to_string_lossy().into_owned(), None, None)
}

/// Write a text file through the environment, upstream's `env.writeFile`.
pub(crate) async fn write(env: &NodeExecutionEnv, path: &str, text: &str, context: &Context) {
    env.write_file(path, FileContent::from(text), context)
        .await
        .expect("write");
}

/// Create a directory through the environment, upstream's
/// `env.createDir(path, { recursive: true })`.
pub(crate) async fn mkdir(env: &NodeExecutionEnv, path: &str, context: &Context) {
    env.create_dir(path, None, context).await.expect("mkdir");
}

/// The absolute path of a root-relative fixture, upstream's `join(root, ...)` —
/// the environment resolves relative paths lexically, so the raw tempdir
/// prefix matches what the loaders report.
pub(crate) fn file_path(root: &Path, relative: &str) -> String {
    format!("{}/{relative}", root.display())
}

/// One injected failure: a fixed code and message, applied to the method's
/// calls whose path contains `path_contains` (an empty needle matches every
/// call).
pub(crate) struct Fault {
    /// The path substring that triggers the failure.
    pub(crate) path_contains: &'static str,
    /// The backend-independent error code the failure reports.
    pub(crate) code: FileErrorCode,
    /// The failure message.
    pub(crate) message: &'static str,
}

/// A nodejs execution environment wrapper that faults individual
/// [`FileSystem`] methods, so the loaders' error-diagnostic branches bind
/// deterministically. Unfaulted methods delegate to the real environment;
/// `exec` always delegates.
pub(crate) struct FaultEnv {
    inner: NodeExecutionEnv,
    pub(crate) file_info_fault: Option<Fault>,
    pub(crate) join_path_fault: Option<Fault>,
    pub(crate) read_text_file_fault: Option<Fault>,
    pub(crate) list_dir_fault: Option<Fault>,
    pub(crate) canonical_path_fault: Option<Fault>,
}

impl FaultEnv {
    pub(crate) fn new(root: &Path) -> Self {
        Self {
            inner: env_for(root),
            file_info_fault: None,
            join_path_fault: None,
            read_text_file_fault: None,
            list_dir_fault: None,
            canonical_path_fault: None,
        }
    }

    fn fault_or(fault: Option<&Fault>, path: &str) -> Option<FileError> {
        let fault = fault?;
        if !path.contains(fault.path_contains) {
            return None;
        }
        Some(FileError::new(
            fault.code,
            fault.message,
            Some(path.to_owned()),
            None,
        ))
    }
}

impl FileSystem for FaultEnv {
    fn cwd(&self) -> &str {
        self.inner.cwd()
    }

    fn absolute_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<String, FileError>> {
        FileSystem::absolute_path(&self.inner, path, context)
    }

    fn join_path<'a>(
        &'a self,
        parts: &'a [String],
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<String, FileError>> {
        if let Some(error) = Self::fault_or(self.join_path_fault.as_ref(), parts.join("/").as_str())
        {
            return Box::pin(async move { Err(error) });
        }
        FileSystem::join_path(&self.inner, parts, context)
    }

    fn read_text_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<String, FileError>> {
        if let Some(error) = Self::fault_or(self.read_text_file_fault.as_ref(), path) {
            return Box::pin(async move { Err(error) });
        }
        FileSystem::read_text_file(&self.inner, path, context)
    }

    fn open_text_line_reader<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<Box<dyn TextLineReader>, FileError>> {
        FileSystem::open_text_line_reader(&self.inner, path, context)
    }

    fn read_text_lines<'a>(
        &'a self,
        path: &'a str,
        options: Option<ReadTextLinesOptions>,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<Vec<String>, FileError>> {
        FileSystem::read_text_lines(&self.inner, path, options, context)
    }

    fn read_binary_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<Vec<u8>, FileError>> {
        FileSystem::read_binary_file(&self.inner, path, context)
    }

    fn write_file<'a>(
        &'a self,
        path: &'a str,
        payload: FileContent,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<(), FileError>> {
        FileSystem::write_file(&self.inner, path, payload, context)
    }

    fn append_file<'a>(
        &'a self,
        path: &'a str,
        payload: FileContent,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<(), FileError>> {
        FileSystem::append_file(&self.inner, path, payload, context)
    }

    fn rename_file<'a>(
        &'a self,
        source_path: &'a str,
        destination_path: &'a str,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<(), FileError>> {
        FileSystem::rename_file(&self.inner, source_path, destination_path, context)
    }

    fn file_info<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<FileInfo, FileError>> {
        if let Some(error) = Self::fault_or(self.file_info_fault.as_ref(), path) {
            return Box::pin(async move { Err(error) });
        }
        FileSystem::file_info(&self.inner, path, context)
    }

    fn list_dir<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<Vec<FileInfo>, FileError>> {
        if let Some(error) = Self::fault_or(self.list_dir_fault.as_ref(), path) {
            return Box::pin(async move { Err(error) });
        }
        FileSystem::list_dir(&self.inner, path, context)
    }

    fn canonical_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<String, FileError>> {
        if let Some(error) = Self::fault_or(self.canonical_path_fault.as_ref(), path) {
            return Box::pin(async move { Err(error) });
        }
        FileSystem::canonical_path(&self.inner, path, context)
    }

    fn exists<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<bool, FileError>> {
        FileSystem::exists(&self.inner, path, context)
    }

    fn create_dir<'a>(
        &'a self,
        path: &'a str,
        options: Option<CreateDirOptions>,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<(), FileError>> {
        FileSystem::create_dir(&self.inner, path, options, context)
    }

    fn remove<'a>(
        &'a self,
        path: &'a str,
        options: Option<RemoveOptions>,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<(), FileError>> {
        FileSystem::remove(&self.inner, path, options, context)
    }

    fn create_temp_dir<'a>(
        &'a self,
        prefix: Option<&'a str>,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<String, FileError>> {
        FileSystem::create_temp_dir(&self.inner, prefix, context)
    }

    fn create_temp_file<'a>(
        &'a self,
        options: Option<TempFileOptions>,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<String, FileError>> {
        FileSystem::create_temp_file(&self.inner, options, context)
    }

    fn cleanup<'a>(&'a self, context: &'a Context) -> pi_ai::types::BoxedFuture<'a, ()> {
        FileSystem::cleanup(&self.inner, context)
    }
}

impl Shell for FaultEnv {
    fn exec<'a>(
        &'a self,
        command: &'a str,
        options: Option<ShellExecOptions>,
        context: &'a Context,
    ) -> pi_ai::types::BoxedFuture<'a, Result<ShellExecResult, crate::harness::types::ExecutionError>>
    {
        Shell::exec(&self.inner, command, options, context)
    }

    fn cleanup<'a>(&'a self, context: &'a Context) -> pi_ai::types::BoxedFuture<'a, ()> {
        Shell::cleanup(&self.inner, context)
    }
}

impl ExecutionEnv for FaultEnv {}
