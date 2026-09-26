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
use crate::harness::types::{FileContent, FileSystem as _};

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
