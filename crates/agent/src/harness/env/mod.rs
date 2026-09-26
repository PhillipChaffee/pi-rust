//! The nodejs execution environment, ported from upstream
//! `src/harness/env/nodejs.ts`.
//!
//! The module carries the real-process real-filesystem
//! [`nodejs::NodeExecutionEnv`]:
//! tokio process + fs for upstream's `node:child_process` and
//! `node:fs/promises`, process-group `SIGKILL` for upstream's
//! `killProcessTree`, and the same non-throwing contract — every operation
//! encodes its failures, including unexpected backend failures, in the
//! returned `Result::Err`, and `cleanup` is best-effort and never fails.
//!
//! Upstream's win32 branches (Git-for-Windows discovery, `taskkill`, the
//! detached-descendant stdio grace timer) ride the map's win32 ticket; the
//! unix branches carry the whole surface here.

pub mod nodejs;
