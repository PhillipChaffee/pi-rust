//! The built-in tools suite, ported 1:1 from upstream
//! `test/harness/tools.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, with boundary tests binding
//! the restated branches.
//!
//! Restatements the port carries: the `NodeExecutionEnv` test subclasses
//! (slow reads, blocked writes, late and paced output) restate as one
//! hookable wrapper delegating to the real environment; the JS
//! `AbortController`/`Promise.all` interleavings restate as chord contexts
//! with `tokio::join!`; upstream's fake-timer checkpoint test drives
//! tokio's pausable clock; and the `applyPatch` oracle is the port of
//! jsdiff's parse-and-apply core (the fuzz-0 path) the suite exercises.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
// The applyPatch oracle ports jsdiff's signed edit-graph positions; the
// isize/usize casts are the port's arithmetic, bounds-checked at the sites.
#![expect(
    clippy::cast_sign_loss,
    reason = "the applyPatch oracle ports jsdiff's signed edit-graph positions; the casts are the port's arithmetic"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "the applyPatch oracle ports jsdiff's signed edit-graph positions; the casts are the port's arithmetic"
)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use base64::Engine as _;

use pi_ai::types::BoxedFuture;
use serde_json::json;

use crate::harness::context::{Context, background_context, with_cancel};
use crate::harness::env::nodejs::NodeExecutionEnv;
use crate::harness::tools::edit::create_edit_tool;
use crate::harness::tools::edit_diff::{DiffPart, Edit, apply_edits_to_normalized_content};
use crate::harness::tools::image::{detect_supported_image_mime_type, encode_base64};
use crate::harness::tools::read::{
    ReadImageProcessorOptions, ReadImageProcessorResult, ReadToolOptions, create_read_tool,
};
use crate::harness::tools::tool_context::{EnvToolContext, ExecutionToolContext};
use crate::harness::tools::write::create_write_tool;
use crate::harness::tools::{BashExecution, BashToolOptions, create_bash_tool};
use crate::harness::types::{
    AgentHarnessTool, AgentHarnessToolInvocation, AgentHarnessToolUpdateCallback, CreateDirOptions,
    ExecutionEnv, ExecutionError, ExecutionErrorCode, FileContent, FileError, FileInfo, FileSystem,
    ReadTextLinesOptions, Shell, ShellExecOptions, ShellExecResult, ShellOutputLimits,
    ShellOutputMetadata, ShellOutputRetention, ShellOutputUpdate, ShellOutputView, TempFileOptions,
    TextLineReader,
};
use crate::harness::utils::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationOptions, truncate_tail,
};
use crate::types::{AgentToolContent, AgentToolError, AgentToolResult};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The stable invocation identity, upstream's `invocation` literal.
#[derive(Debug)]
struct TestInvocation;

impl AgentHarnessToolInvocation for TestInvocation {
    fn invocation_id(&self) -> &'static str {
        "test-result"
    }

    fn operation_id(&self) -> &'static str {
        "test-operation"
    }

    fn turn_id(&self) -> &'static str {
        "test-turn"
    }

    fn get_memo(
        &self,
        _name: &str,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<serde_json::Value>, String>> {
        Box::pin(async { Ok(None) })
    }

    fn set_memo(
        &self,
        _name: &str,
        _value: Option<serde_json::Value>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
}

fn text_output(result: &AgentToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|part| match part {
            AgentToolContent::Text(text) => Some(text.text.clone()),
            AgentToolContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The environment-over-tempdir context, upstream's `createContext()`.
struct ToolTestContext {
    root: tempfile::TempDir,
    env: Arc<NodeExecutionEnv>,
    tool_context: crate::harness::types::ToolContext,
}

fn create_context() -> ToolTestContext {
    let root = tempfile::tempdir().expect("tempdir");
    let env = Arc::new(NodeExecutionEnv::new(
        root.path().to_string_lossy().into_owned(),
        None,
        None,
    ));
    let tool_context: crate::harness::types::ToolContext =
        Some(Arc::new(EnvToolContext { env: env.clone() }));
    ToolTestContext {
        root,
        env,
        tool_context,
    }
}

/// Runs a tool's execute, upstream's direct `tool.execute(...)` calls.
async fn run_tool(
    tool: &AgentHarnessTool,
    args: serde_json::Value,
    tool_context: crate::harness::types::ToolContext,
    on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
    context: &Context,
) -> Result<AgentToolResult, AgentToolError> {
    let invocation = TestInvocation;
    (tool.execute)(
        "test-call",
        &args,
        on_update,
        tool_context,
        &invocation,
        context,
    )
    .await
}

async fn delay(ms: u64) {
    tokio::time::sleep(Duration::from_millis(ms)).await;
}

// ---------------------------------------------------------------------------
// The hookable environment wrapper, the test subclasses of NodeExecutionEnv
// ---------------------------------------------------------------------------

type ReadTextHook = Arc<
    dyn for<'a> Fn(&'a str, &'a Context) -> BoxedFuture<'a, Result<String, FileError>>
        + Send
        + Sync,
>;
type WriteFileHook = Arc<
    dyn for<'a> Fn(&'a str, FileContent, &'a Context) -> BoxedFuture<'a, Result<(), FileError>>
        + Send
        + Sync,
>;
type ExecHook = Arc<
    dyn for<'a> Fn(
            &'a str,
            Option<ShellExecOptions>,
            &'a Context,
        ) -> BoxedFuture<'a, Result<ShellExecResult, ExecutionError>>
        + Send
        + Sync,
>;

/// The `NodeExecutionEnv` subclass stand-in: overridden methods run the
/// hook; everything else delegates to the real environment.
struct HookedEnv {
    inner: Arc<NodeExecutionEnv>,
    read_text_file_hook: Option<ReadTextHook>,
    write_file_hook: Option<WriteFileHook>,
    exec_hook: Option<ExecHook>,
}

#[allow(
    clippy::similar_names,
    reason = "the hook signatures mirror the FileSystem trait's `content`/`context` parameter names"
)]
impl FileSystem for HookedEnv {
    fn cwd(&self) -> &str {
        self.inner.cwd()
    }

    fn absolute_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        FileSystem::absolute_path(self.inner.as_ref(), path, context)
    }

    fn join_path<'a>(
        &'a self,
        parts: &'a [String],
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        FileSystem::join_path(self.inner.as_ref(), parts, context)
    }

    fn read_text_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        self.read_text_file_hook.as_ref().map_or_else(
            || FileSystem::read_text_file(self.inner.as_ref(), path, context),
            |hook| hook(path, context),
        )
    }

    fn open_text_line_reader<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Box<dyn TextLineReader>, FileError>> {
        FileSystem::open_text_line_reader(self.inner.as_ref(), path, context)
    }

    fn read_text_lines<'a>(
        &'a self,
        path: &'a str,
        options: Option<ReadTextLinesOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<String>, FileError>> {
        FileSystem::read_text_lines(self.inner.as_ref(), path, options, context)
    }

    fn read_binary_file<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<u8>, FileError>> {
        FileSystem::read_binary_file(self.inner.as_ref(), path, context)
    }

    fn write_file<'a>(
        &'a self,
        path: &'a str,
        content: FileContent,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        self.write_file_hook.as_ref().map_or_else(
            || FileSystem::write_file(self.inner.as_ref(), path, content.clone(), context),
            |hook| hook(path, content.clone(), context),
        )
    }

    fn append_file<'a>(
        &'a self,
        path: &'a str,
        content: FileContent,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        FileSystem::append_file(self.inner.as_ref(), path, content, context)
    }

    fn rename_file<'a>(
        &'a self,
        source_path: &'a str,
        destination_path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        FileSystem::rename_file(self.inner.as_ref(), source_path, destination_path, context)
    }

    fn file_info<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<FileInfo, FileError>> {
        FileSystem::file_info(self.inner.as_ref(), path, context)
    }

    fn list_dir<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<Vec<FileInfo>, FileError>> {
        FileSystem::list_dir(self.inner.as_ref(), path, context)
    }

    fn canonical_path<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        FileSystem::canonical_path(self.inner.as_ref(), path, context)
    }

    fn exists<'a>(
        &'a self,
        path: &'a str,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<bool, FileError>> {
        FileSystem::exists(self.inner.as_ref(), path, context)
    }

    fn create_dir<'a>(
        &'a self,
        path: &'a str,
        options: Option<CreateDirOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        FileSystem::create_dir(self.inner.as_ref(), path, options, context)
    }

    fn remove<'a>(
        &'a self,
        path: &'a str,
        options: Option<crate::harness::types::RemoveOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<(), FileError>> {
        FileSystem::remove(self.inner.as_ref(), path, options, context)
    }

    fn create_temp_dir<'a>(
        &'a self,
        prefix: Option<&'a str>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        FileSystem::create_temp_dir(self.inner.as_ref(), prefix, context)
    }

    fn create_temp_file<'a>(
        &'a self,
        options: Option<TempFileOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<String, FileError>> {
        FileSystem::create_temp_file(self.inner.as_ref(), options, context)
    }

    fn cleanup<'a>(&'a self, context: &'a Context) -> BoxedFuture<'a, ()> {
        FileSystem::cleanup(self.inner.as_ref(), context)
    }
}

impl Shell for HookedEnv {
    fn exec<'a>(
        &'a self,
        command: &'a str,
        options: Option<ShellExecOptions>,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<ShellExecResult, ExecutionError>> {
        self.exec_hook.as_ref().map_or_else(
            || Shell::exec(self.inner.as_ref(), command, options.clone(), context),
            |hook| hook(command, options.clone(), context),
        )
    }

    fn cleanup<'a>(&'a self, context: &'a Context) -> BoxedFuture<'a, ()> {
        Shell::cleanup(self.inner.as_ref(), context)
    }
}

impl ExecutionEnv for HookedEnv {}

/// A tool context over the hookable wrapper; the erased context is
/// Option-typed, so the always-present value rides the same shape.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the erased ToolContext is Option-typed; the fixture returns the populated shape"
)]
fn hooked_context(inner: Arc<HookedEnv>) -> crate::harness::types::ToolContext {
    Some(Arc::new(EnvToolContext { env: inner }))
}

/// The `SlowReadExecutionEnv`: every text read stalls 20 ms first.
fn slow_read_env() -> (tempfile::TempDir, Arc<HookedEnv>) {
    let root = tempfile::tempdir().expect("tempdir");
    let inner = Arc::new(NodeExecutionEnv::new(
        root.path().to_string_lossy().into_owned(),
        None,
        None,
    ));
    let hook_inner = Arc::clone(&inner);
    let env = Arc::new(HookedEnv {
        inner,
        read_text_file_hook: Some(Arc::new(move |path: &str, context: &Context| {
            let inner = Arc::clone(&hook_inner);
            Box::pin(async move {
                delay(20).await;
                FileSystem::read_text_file(inner.as_ref(), path, context).await
            })
        })),
        write_file_hook: None,
        exec_hook: None,
    });
    (root, env)
}

/// The `BlockingWriteExecutionEnv`: the first write signals and blocks; the
/// second write records it started.
struct BlockingWriteFixture {
    _root: tempfile::TempDir,
    env: Arc<HookedEnv>,
    first_write_started: Arc<tokio::sync::Notify>,
    finish_first_write: Arc<tokio::sync::Notify>,
    second_write_started: Arc<AtomicBool>,
}

fn blocking_write_env() -> BlockingWriteFixture {
    let root = tempfile::tempdir().expect("tempdir");
    let inner = Arc::new(NodeExecutionEnv::new(
        root.path().to_string_lossy().into_owned(),
        None,
        None,
    ));
    let first_write_started = Arc::new(tokio::sync::Notify::new());
    let finish_first_write = Arc::new(tokio::sync::Notify::new());
    let second_write_started = Arc::new(AtomicBool::new(false));
    let hook_inner = Arc::clone(&inner);
    let started = Arc::clone(&first_write_started);
    let finish = Arc::clone(&finish_first_write);
    let second_started = Arc::clone(&second_write_started);
    let env = Arc::new(HookedEnv {
        inner,
        read_text_file_hook: None,
        write_file_hook: Some(Arc::new(
            move |path: &str, content: FileContent, context: &Context| {
                let inner = Arc::clone(&hook_inner);
                let started = Arc::clone(&started);
                let finish = Arc::clone(&finish);
                let second_started = Arc::clone(&second_started);
                Box::pin(async move {
                    let FileContent::Text(text) = &content else {
                        return FileSystem::write_file(inner.as_ref(), path, content, context)
                            .await;
                    };
                    if text == "first\n" {
                        started.notify_one();
                        finish.notified().await;
                    } else if text == "second\n" {
                        second_started.store(true, Ordering::Relaxed);
                    }
                    FileSystem::write_file(inner.as_ref(), path, content, context).await
                })
            },
        )),
        exec_hook: None,
    });
    BlockingWriteFixture {
        _root: root,
        env,
        first_write_started,
        finish_first_write,
        second_write_started,
    }
}

/// The `BlockingEditExecutionEnv`: the first edit's write signals and
/// blocks, settles through the background context, then records; the second
/// edit's write records it started.
struct BlockingEditFixture {
    _root: tempfile::TempDir,
    env: Arc<HookedEnv>,
    first_edit_write_started: Arc<tokio::sync::Notify>,
    finish_first_edit_write: Arc<tokio::sync::Notify>,
    first_edit_write_settled: Arc<AtomicBool>,
    second_edit_write_started: Arc<AtomicBool>,
}

fn blocking_edit_env() -> BlockingEditFixture {
    let root = tempfile::tempdir().expect("tempdir");
    let inner = Arc::new(NodeExecutionEnv::new(
        root.path().to_string_lossy().into_owned(),
        None,
        None,
    ));
    let first_edit_write_started = Arc::new(tokio::sync::Notify::new());
    let finish_first_edit_write = Arc::new(tokio::sync::Notify::new());
    let first_edit_write_settled = Arc::new(AtomicBool::new(false));
    let second_edit_write_started = Arc::new(AtomicBool::new(false));
    let hook_inner = Arc::clone(&inner);
    let started = Arc::clone(&first_edit_write_started);
    let finish = Arc::clone(&finish_first_edit_write);
    let settled = Arc::clone(&first_edit_write_settled);
    let second_started = Arc::clone(&second_edit_write_started);
    let env = Arc::new(HookedEnv {
        inner,
        read_text_file_hook: None,
        write_file_hook: Some(Arc::new(
            move |path: &str, content: FileContent, context: &Context| {
                let inner = Arc::clone(&hook_inner);
                let started = Arc::clone(&started);
                let finish = Arc::clone(&finish);
                let settled = Arc::clone(&settled);
                let second_started = Arc::clone(&second_started);
                Box::pin(async move {
                    let FileContent::Text(text) = &content else {
                        return FileSystem::write_file(inner.as_ref(), path, content, context)
                            .await;
                    };
                    if text == "ALPHA\nbeta\n" {
                        started.notify_one();
                        finish.notified().await;
                        // The blocked write settles through the background
                        // context so the aborted call context cannot fail it
                        // before the flag records, upstream's
                        // `super.writeFile(path, content, BACKGROUND_CONTEXT)`.
                        let result = FileSystem::write_file(
                            inner.as_ref(),
                            path,
                            content,
                            &background_context(),
                        )
                        .await;
                        settled.store(true, Ordering::Relaxed);
                        return result;
                    }
                    if text == "ALPHA\nBETA\n" || text == "alpha\nBETA\n" {
                        second_started.store(true, Ordering::Relaxed);
                    }
                    FileSystem::write_file(inner.as_ref(), path, content, context).await
                })
            },
        )),
        exec_hook: None,
    });
    BlockingEditFixture {
        _root: root,
        env,
        first_edit_write_started,
        finish_first_edit_write,
        first_edit_write_settled,
        second_edit_write_started,
    }
}

/// The bounded-view fixture, upstream's `fakeShellOutput`: it truncates the
/// text against the options' limits, fires one replace update, and returns
/// the exec result.
fn fake_shell_output(
    text: &str,
    options: Option<&ShellExecOptions>,
    spill_path: Option<&str>,
) -> ShellExecResult {
    let limits = options.and_then(|options| options.capture.as_ref()).map_or(
        ShellOutputLimits {
            max_bytes: DEFAULT_MAX_BYTES,
            max_lines: DEFAULT_MAX_LINES,
            retain: Some(ShellOutputRetention::Tail),
        },
        |capture| capture.limits,
    );
    let truncated = truncate_tail(
        text,
        TruncationOptions {
            max_lines: Some(limits.max_lines),
            max_bytes: Some(limits.max_bytes),
        },
    );
    if let Some(on_update) = options.and_then(|options| options.on_update.as_ref()) {
        on_update(
            &ShellOutputUpdate::Replace {
                output: ShellOutputView {
                    text: truncated.content.clone(),
                    metadata: ShellOutputMetadata {
                        truncation: truncated.metadata.clone(),
                        spill_path: spill_path.map(str::to_owned),
                        last_line_bytes: None,
                    },
                },
            },
            &background_context(),
        );
    }
    ShellExecResult {
        exit_code: 0,
        truncation: truncated.metadata,
        spill_path: spill_path.map(str::to_owned),
        last_line_bytes: None,
    }
}

/// The `LateOutputExecutionEnv`: it returns `before\n` and then publishes a
/// replacement carrying `late` after the execution settled.
fn late_output_env() -> (tempfile::TempDir, Arc<HookedEnv>) {
    let root = tempfile::tempdir().expect("tempdir");
    let inner = Arc::new(NodeExecutionEnv::new(
        root.path().to_string_lossy().into_owned(),
        None,
        None,
    ));
    let env = Arc::new(HookedEnv {
        inner,
        read_text_file_hook: None,
        write_file_hook: None,
        exec_hook: Some(Arc::new(
            |_command: &str, options: Option<ShellExecOptions>, _context: &Context| {
                let options = options;
                Box::pin(async move {
                    let result = fake_shell_output("before\n", options.as_ref(), None);
                    if let Some(options) = options {
                        tokio::spawn(async move {
                            tokio::task::yield_now().await;
                            let limits = options.capture.as_ref().map_or(
                                ShellOutputLimits {
                                    max_bytes: DEFAULT_MAX_BYTES,
                                    max_lines: DEFAULT_MAX_LINES,
                                    retain: Some(ShellOutputRetention::Tail),
                                },
                                |capture| capture.limits,
                            );
                            let truncation = truncate_tail(
                                "before\nlate\n",
                                TruncationOptions {
                                    max_lines: Some(limits.max_lines),
                                    max_bytes: Some(limits.max_bytes),
                                },
                            );
                            if let Some(on_update) = options.on_update.as_ref() {
                                on_update(
                                    &ShellOutputUpdate::Replace {
                                        output: ShellOutputView {
                                            text: truncation.content,
                                            metadata: ShellOutputMetadata {
                                                truncation: truncation.metadata,
                                                spill_path: None,
                                                last_line_bytes: None,
                                            },
                                        },
                                    },
                                    &background_context(),
                                );
                            }
                        });
                    }
                    Ok(result)
                })
            },
        )),
    });
    (root, env)
}

/// The `CheckpointOutputExecutionEnv`: paced replace updates at 0, 2100,
/// 2200, and 4200 ms, the last one the returned value.
fn checkpoint_output_env() -> (tempfile::TempDir, Arc<HookedEnv>) {
    let root = tempfile::tempdir().expect("tempdir");
    let inner = Arc::new(NodeExecutionEnv::new(
        root.path().to_string_lossy().into_owned(),
        None,
        None,
    ));
    let env = Arc::new(HookedEnv {
        inner,
        read_text_file_hook: None,
        write_file_hook: None,
        exec_hook: Some(Arc::new(
            |_command: &str, options: Option<ShellExecOptions>, _context: &Context| {
                let options = options;
                Box::pin(async move {
                    fake_shell_output("one\n", options.as_ref(), None);
                    delay(2_100).await;
                    fake_shell_output("one\ntwo\n", options.as_ref(), None);
                    delay(100).await;
                    fake_shell_output("one\ntwo\nthree\n", options.as_ref(), None);
                    delay(2_000).await;
                    Ok(fake_shell_output(
                        "one\ntwo\nthree\nfour\n",
                        options.as_ref(),
                        None,
                    ))
                })
            },
        )),
    });
    (root, env)
}

/// The `TimeoutOutputExecutionEnv`: it spills 2001 lines to a temp file,
/// publishes the truncated view with the spill path, and fails with the
/// timeout error.
fn timeout_output_env() -> (tempfile::TempDir, Arc<HookedEnv>) {
    let root = tempfile::tempdir().expect("tempdir");
    let inner = Arc::new(NodeExecutionEnv::new(
        root.path().to_string_lossy().into_owned(),
        None,
        None,
    ));
    let truncated_lines = DEFAULT_MAX_LINES + 1;
    let hook_inner = Arc::clone(&inner);
    let env = Arc::new(HookedEnv {
        inner,
        read_text_file_hook: None,
        write_file_hook: None,
        exec_hook: Some(Arc::new(
            move |_command: &str, options: Option<ShellExecOptions>, context: &Context| {
                let hook_inner = Arc::clone(&hook_inner);
                let options = options;
                Box::pin(async move {
                    let output = (1..=truncated_lines)
                        .map(|index| format!("line-{index}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                        + "\n";
                    let spill_path = FileSystem::create_temp_file(
                        hook_inner.as_ref(),
                        Some(TempFileOptions {
                            prefix: Some("timeout-".to_owned()),
                            suffix: Some(".log".to_owned()),
                        }),
                        context,
                    )
                    .await
                    .expect("the spill file");
                    FileSystem::write_file(
                        hook_inner.as_ref(),
                        &spill_path,
                        FileContent::Text(output.clone()),
                        context,
                    )
                    .await
                    .expect("the spill write");
                    fake_shell_output(&output, options.as_ref(), Some(&spill_path));
                    Err(ExecutionError::new(
                        ExecutionErrorCode::Timeout,
                        format!(
                            "timeout:{}",
                            options
                                .as_ref()
                                .and_then(|options| options.timeout)
                                .map_or_else(
                                    || "undefined".to_owned(),
                                    |seconds| seconds.to_string()
                                )
                        ),
                        None,
                    ))
                })
            },
        )),
    });
    (root, env)
}

/// A 58-byte 1x1 24-bit BMP, upstream's `createTinyBmp`.
fn create_tiny_bmp() -> Vec<u8> {
    let mut bytes = vec![0_u8; 58];
    bytes[0] = 0x42;
    bytes[1] = 0x4d;
    bytes[2..6].copy_from_slice(&58_u32.to_le_bytes());
    bytes[10..14].copy_from_slice(&54_u32.to_le_bytes());
    bytes[14..18].copy_from_slice(&40_u32.to_le_bytes());
    bytes[18..22].copy_from_slice(&1_i32.to_le_bytes());
    bytes[22..26].copy_from_slice(&1_i32.to_le_bytes());
    bytes[26..28].copy_from_slice(&1_u16.to_le_bytes());
    bytes[28..30].copy_from_slice(&24_u16.to_le_bytes());
    bytes[34..38].copy_from_slice(&4_u32.to_le_bytes());
    bytes
}

// ---------------------------------------------------------------------------
// The applyPatch oracle: jsdiff's parse-and-apply core (the fuzz-0 path the
// suite exercises)
// ---------------------------------------------------------------------------

/// A parsed hunk, jsdiff's parseHunk output (post-quirk starts).
struct ParsedHunk {
    old_start: usize,
    old_lines: usize,
    new_start: usize,
    new_lines: usize,
    lines: Vec<String>,
}

/// Parses the suite's generated patches, jsdiff's `parsePatch` slice:
/// metadata and file headers skipped, hunks with the count-0 start quirk.
fn parse_patch(patch: &str) -> Vec<ParsedHunk> {
    let diffstr: Vec<&str> = patch.split('\n').collect();
    let mut index = 0;
    while index < diffstr.len()
        && !(diffstr[index].starts_with("--- ")
            || diffstr[index].starts_with("+++ ")
            || diffstr[index].starts_with("@@ "))
    {
        index += 1;
    }
    if index < diffstr.len() && diffstr[index].starts_with("--- ") {
        index += 1;
    }
    if index < diffstr.len() && diffstr[index].starts_with("+++ ") {
        index += 1;
    }
    let mut hunks = Vec::new();
    while index < diffstr.len() {
        if diffstr[index].starts_with("@@") {
            hunks.push(parse_hunk(&diffstr, &mut index));
        } else if diffstr[index].is_empty() {
            index += 1;
        } else {
            break;
        }
    }
    hunks
}

/// Parses one hunk at `index`, jsdiff's `parseHunk`.
fn parse_hunk(diffstr: &[&str], index: &mut usize) -> ParsedHunk {
    let header = diffstr[*index];
    *index += 1;
    let header = header
        .strip_prefix("@@ -")
        .expect("the hunk header opens with @@ -");
    let (old_start_text, rest) = split_number(header);
    let (old_lines_text, rest) = rest.strip_prefix(',').map_or(("1", rest), split_number);
    let rest = rest
        .strip_prefix(" +")
        .expect("the hunk header's new range");
    let (new_start_text, rest) = split_number(rest);
    let (new_lines_text, _rest) = rest.strip_prefix(',').map_or(("1", rest), split_number);
    let mut hunk = ParsedHunk {
        old_start: old_start_text.parse().expect("the old start"),
        old_lines: old_lines_text.parse().expect("the old count"),
        new_start: new_start_text.parse().expect("the new start"),
        new_lines: new_lines_text.parse().expect("the new count"),
        lines: Vec::new(),
    };
    if hunk.old_lines == 0 {
        hunk.old_start += 1;
    }
    if hunk.new_lines == 0 {
        hunk.new_start += 1;
    }
    let mut add_count = 0_usize;
    let mut remove_count = 0_usize;
    while *index < diffstr.len()
        && (remove_count < hunk.old_lines
            || add_count < hunk.new_lines
            || diffstr[*index].starts_with('\\'))
    {
        let line = diffstr[*index];
        let operation = if line.is_empty() && *index != diffstr.len() - 1 {
            Some(' ')
        } else {
            line.chars().next()
        };
        if operation.is_some_and(|operation| {
            operation == '+' || operation == '-' || operation == ' ' || operation == '\\'
        }) {
            hunk.lines.push(line.to_owned());
            match operation {
                Some('+') => add_count += 1,
                Some('-') => remove_count += 1,
                // Context lines advance both counts, jsdiff's parse.
                Some(' ') => {
                    add_count += 1;
                    remove_count += 1;
                }
                _ => {}
            }
        }
        *index += 1;
    }
    // The empty-block count case: a countless side whose header read 1
    // collapses to 0, jsdiff's post-loop normalization.
    if add_count == 0 && hunk.new_lines == 1 {
        hunk.new_lines = 0;
    }
    if remove_count == 0 && hunk.old_lines == 1 {
        hunk.old_lines = 0;
    }
    assert_eq!(
        add_count, hunk.new_lines,
        "the hunk's added count matches its header"
    );
    assert_eq!(
        remove_count, hunk.old_lines,
        "the hunk's removed count matches its header"
    );
    hunk
}

/// Splits a leading decimal run off the front, returning it and the rest.
fn split_number(text: &str) -> (&str, &str) {
    let end = text
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(text.len());
    (&text[..end], &text[end..])
}

/// Applies the patch at fuzz factor 0, jsdiff's `applyPatch`: the EOFNL
/// pass, the distance-iterator search, and the hunk fitter.
fn apply_patch(source: &str, patch: &str) -> Option<String> {
    let hunks = parse_patch(patch);
    let mut lines: Vec<String> = source.split('\n').map(str::to_owned).collect();
    let mut min_line = 0_isize;

    // The EOFNL pass over the final hunk's marker lines.
    let mut previous_line = String::new();
    let mut remove_eofnl = false;
    let mut add_eofnl = false;
    if let Some(last) = hunks.last() {
        for line in &last.lines {
            if line.starts_with('\\') {
                if previous_line.starts_with('+') {
                    remove_eofnl = true;
                } else if previous_line.starts_with('-') {
                    add_eofnl = true;
                }
            }
            previous_line.clone_from(line);
        }
    }
    if remove_eofnl {
        if add_eofnl {
            if lines.last().is_some_and(String::is_empty) {
                return None;
            }
        } else if lines.last().is_some_and(String::is_empty) {
            lines.pop();
        } else {
            return None;
        }
    } else if add_eofnl {
        if lines.last().is_some_and(String::is_empty) {
            return None;
        }
        lines.push(String::new());
    }

    let mut result_lines: Vec<String> = Vec::new();
    let mut previous_hunk_offset: isize = 0;
    for hunk in &hunks {
        let max_line = lines.len() as isize - hunk.old_lines as isize;
        let start_to_pos = hunk.old_start as isize + previous_hunk_offset - 1;
        let mut fitted: Option<(Vec<String>, isize)> = None;
        let mut applied_at = 0_isize;
        for to_pos in distance_positions(start_to_pos, min_line, max_line) {
            if to_pos < 0 {
                continue;
            }
            if let Some(fit) = apply_hunk(&hunk.lines, to_pos as usize, &lines) {
                fitted = Some(fit);
                applied_at = to_pos;
                break;
            }
        }
        let (patched_lines, old_line_last_i) = fitted?;
        // Copy everything from the end of the previous hunk to where this
        // one started, then the hunk's patched lines.
        for line in &lines[min_line as usize..applied_at as usize] {
            result_lines.push(line.clone());
        }
        for line in patched_lines {
            result_lines.push(line);
        }
        min_line = old_line_last_i + 1;
        previous_hunk_offset = applied_at + 1 - hunk.old_start as isize;
    }
    for line in &lines[min_line as usize..] {
        result_lines.push(line.clone());
    }
    Some(result_lines.join("\n"))
}

/// The hunk fitter at `maxErrors = 0`, jsdiff's `applyHunk`: context must
/// match exactly, removals must be present, and trailing context trims so
/// later hunks can start inside it.
fn apply_hunk(
    hunk_lines: &[String],
    mut to_pos: usize,
    lines: &[String],
) -> Option<(Vec<String>, isize)> {
    let mut patched_lines: Vec<String> = Vec::new();
    let mut consecutive_old_context_lines = 0_usize;
    for hunk_line in hunk_lines {
        if hunk_line.starts_with('\\') {
            continue;
        }
        let (operation, content) = if hunk_line.is_empty() {
            (' ', "")
        } else {
            (
                hunk_line.chars().next().expect("a non-empty line"),
                &hunk_line[1..],
            )
        };
        match operation {
            '-' => {
                if lines.get(to_pos).is_some_and(|line| line == content) {
                    to_pos += 1;
                    consecutive_old_context_lines = 0;
                } else {
                    return None;
                }
            }
            '+' => {
                patched_lines.push(content.to_owned());
                consecutive_old_context_lines = 0;
            }
            _ => {
                consecutive_old_context_lines += 1;
                if lines.get(to_pos).is_some_and(|line| line == content) {
                    patched_lines.push(content.to_owned());
                    to_pos += 1;
                } else {
                    return None;
                }
            }
        }
    }
    patched_lines.truncate(patched_lines.len() - consecutive_old_context_lines);
    Some((
        patched_lines,
        to_pos as isize - 1 - consecutive_old_context_lines as isize,
    ))
}

/// jsdiff's `distanceIterator`: the initial position, then the alternating
/// forward/backward offsets, bounded to the search window.
fn distance_positions(start: isize, min_line: isize, max_line: isize) -> Vec<isize> {
    let mut positions = vec![start];
    let mut local_offset = 1_isize;
    let mut want_forward = true;
    let mut backward_exhausted = false;
    let mut forward_exhausted = false;
    loop {
        if want_forward && !forward_exhausted {
            if backward_exhausted {
                local_offset += 1;
            } else {
                want_forward = false;
            }
            if start + local_offset <= max_line {
                positions.push(start + local_offset);
                continue;
            }
            forward_exhausted = true;
        }
        if !backward_exhausted {
            if !forward_exhausted {
                want_forward = true;
            }
            if min_line <= start - local_offset {
                positions.push(start - local_offset);
                local_offset += 1;
                continue;
            }
            backward_exhausted = true;
            continue;
        }
        break;
    }
    positions
}
// ---------------------------------------------------------------------------
// The 1:1 port of tools.test.ts
// ---------------------------------------------------------------------------

/// The PNG 1x1 red pixel, upstream's base64 fixture.
const PNG_FIXTURE: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg==";

#[tokio::test]
async fn reads_text_with_offsets_limits_and_continuation_notices() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "test.txt",
        FileContent::Text(
            (1..=100)
                .map(|index| format!("Line {index}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let tool = create_read_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "path": "test.txt", "offset": 41, "limit": 20 }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the read");
    let output = text_output(&result);

    assert!(!output.contains("Line 40"));
    assert!(output.contains("Line 41"));
    assert!(output.contains("Line 60"));
    assert!(!output.contains("Line 61"));
    assert!(output.contains("[40 more lines in file. Use offset=61 to continue.]"));
}

#[tokio::test]
async fn truncates_large_text_by_line_count() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "large.txt",
        FileContent::Text(
            (1..=2500)
                .map(|index| format!("Line {index}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let tool = create_read_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "path": "large.txt" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the read");

    assert!(
        text_output(&result)
            .contains("[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]")
    );
    let truncation = &result.details["truncation"];
    assert_eq!(truncation["truncated"], json!(true));
    assert_eq!(truncation["truncatedBy"], json!("lines"));
    assert_eq!(truncation["totalLines"], json!(2500));
    assert_eq!(truncation["outputLines"], json!(2000));
}

#[tokio::test]
async fn does_not_count_a_trailing_newline_as_an_extra_line_at_the_truncation_limit() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "exact.txt",
        FileContent::Text(format!("{}\n", "x\n".repeat(2000).trim_end_matches('\n'))),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let tool = create_read_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "path": "exact.txt" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the read");

    assert!(result.details.is_null());
    assert!(!text_output(&result).contains("Use offset="));
}

#[tokio::test]
async fn rejects_offsets_beyond_the_file() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "short.txt",
        FileContent::Text("one\ntwo\nthree".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let tool = create_read_tool::<EnvToolContext>(None);
    let error = run_tool(
        &tool,
        json!({ "path": "short.txt", "offset": 100 }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the offset rejection");
    assert!(
        error
            .to_string()
            .contains("Offset 100 is beyond end of file (3 lines total)")
    );
}

#[tokio::test]
async fn detects_supported_images_by_content() {
    let test = create_context();
    let png = base64::engine::general_purpose::STANDARD
        .decode(PNG_FIXTURE)
        .expect("the fixture decodes");
    FileSystem::write_file(
        test.env.as_ref(),
        "image.txt",
        FileContent::Bytes(png.clone()),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let tool = create_read_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "path": "image.txt" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the read");

    assert!(text_output(&result).contains("Read image file [image/png]"));
    assert!(
        result
            .content
            .contains(&AgentToolContent::Image(pi_ai::types::ImageContent {
                data: encode_base64(&png),
                mime_type: "image/png".to_owned(),
            }))
    );
}

#[tokio::test]
async fn delegates_image_conversion_and_resizing_to_an_injected_processor() {
    /// The recorded processor call, upstream's `received` object.
    #[derive(Clone)]
    struct Received {
        bytes: Vec<u8>,
        mime_type: String,
        auto_resize_images: bool,
    }
    let test = create_context();
    let bmp = create_tiny_bmp();
    FileSystem::write_file(
        test.env.as_ref(),
        "image.bmp",
        FileContent::Bytes(bmp.clone()),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    let received: Arc<Mutex<Option<Received>>> = Arc::new(Mutex::new(None));
    let received_sink = Arc::clone(&received);
    let tool = create_read_tool::<EnvToolContext>(Some(ReadToolOptions {
        auto_resize_images: Some(false),
        image_processor: Some(Arc::new(
            move |bytes: &[u8],
                  mime_type: &str,
                  options: &ReadImageProcessorOptions,
                  _context: &Context| {
                let received = Arc::clone(&received_sink);
                let mime_type = mime_type.to_owned();
                let bytes = bytes.to_vec();
                let auto_resize_images = options.auto_resize_images;
                Box::pin(async move {
                    *received
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Received {
                        bytes,
                        mime_type,
                        auto_resize_images,
                    });
                    Ok(ReadImageProcessorResult::Ok {
                        data: "converted".to_owned(),
                        mime_type: "image/png".to_owned(),
                        hints: vec!["[Image converted from image/bmp to image/png.]".to_owned()],
                    })
                })
            },
        )),
    }));

    let result = run_tool(
        &tool,
        json!({ "path": "image.bmp" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the read");

    let received = received
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("the processor ran");
    assert_eq!(received.mime_type, "image/bmp");
    assert!(!received.auto_resize_images);
    assert_eq!(received.bytes, bmp);
    assert!(text_output(&result).contains("[Image converted from image/bmp to image/png.]"));
    assert!(
        result
            .content
            .contains(&AgentToolContent::Image(pi_ai::types::ImageContent {
                data: "converted".to_owned(),
                mime_type: "image/png".to_owned(),
            }))
    );
}

#[tokio::test]
async fn writes_files_and_creates_parent_directories() {
    let test = create_context();
    let tool = create_write_tool::<EnvToolContext>();
    let result = run_tool(
        &tool,
        json!({ "path": "nested/dir/file.txt", "content": "hello" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the write");

    assert_eq!(
        text_output(&result),
        "Successfully wrote to nested/dir/file.txt"
    );
    let content = FileSystem::read_text_file(
        test.env.as_ref(),
        "nested/dir/file.txt",
        &background_context(),
    )
    .await
    .expect("the read back");
    assert_eq!(content, "hello");
}

#[tokio::test]
async fn keeps_the_mutation_queue_locked_until_an_aborted_write_settles() {
    let fixture = blocking_write_env();
    let tool = create_write_tool::<EnvToolContext>();
    let (context, controller) = with_cancel(&background_context());
    let first = run_tool(
        &tool,
        json!({ "path": "file.txt", "content": "first\n" }),
        hooked_context(Arc::clone(&fixture.env)),
        None,
        &context,
    );
    let settled = background_context();
    let second = async {
        fixture.first_write_started.notified().await;
        controller.abort_without_reason();
        let second = run_tool(
            &tool,
            json!({ "path": "file.txt", "content": "second\n" }),
            hooked_context(Arc::clone(&fixture.env)),
            None,
            &settled,
        );
        delay(20).await;
        assert!(!fixture.second_write_started.load(Ordering::Relaxed));
        fixture.finish_first_write.notify_one();
        second.await
    };
    let (first_result, second_result) = tokio::join!(first, second);

    assert!(first_result.is_err());
    second_result.expect("the second write");
    let written = FileSystem::read_text_file(
        fixture.env.inner.as_ref(),
        "file.txt",
        &background_context(),
    )
    .await
    .expect("the read back");
    assert_eq!(written, "second\n");
}

#[tokio::test]
async fn applies_disjoint_edits_and_returns_both_diff_formats() {
    let test = create_context();
    let original = "alpha\nbeta\ngamma\ndelta\n";
    FileSystem::write_file(
        test.env.as_ref(),
        "edit.txt",
        FileContent::Text(original.to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let tool = create_edit_tool::<EnvToolContext>();
    let result = run_tool(
        &tool,
        json!({
            "path": "edit.txt",
            "edits": [
                { "oldText": "alpha\n", "newText": "ALPHA\n" },
                { "oldText": "gamma\n", "newText": "GAMMA\n" }
            ]
        }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the edit");

    assert_eq!(
        text_output(&result),
        "Successfully replaced 2 block(s) in edit.txt."
    );
    assert!(
        result.details["diff"]
            .as_str()
            .expect("a diff string")
            .contains("ALPHA")
    );
    assert!(
        result.details["diff"]
            .as_str()
            .expect("a diff string")
            .contains("GAMMA")
    );
    assert_eq!(
        apply_patch(
            original,
            result.details["patch"].as_str().expect("a patch string")
        )
        .expect("the patch applies"),
        "ALPHA\nbeta\nGAMMA\ndelta\n"
    );
    let content = FileSystem::read_text_file(test.env.as_ref(), "edit.txt", &background_context())
        .await
        .expect("the read back");
    assert_eq!(content, "ALPHA\nbeta\nGAMMA\ndelta\n");
}

#[tokio::test]
async fn matches_all_edits_against_the_original_and_rejects_overlaps() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "edit.txt",
        FileContent::Text("one\ntwo\nthree\n".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let tool = create_edit_tool::<EnvToolContext>();
    let error = run_tool(
        &tool,
        json!({
            "path": "edit.txt",
            "edits": [
                { "oldText": "one\ntwo\n", "newText": "ONE\nTWO\n" },
                { "oldText": "two\nthree\n", "newText": "TWO\nTHREE\n" }
            ]
        }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the overlap rejection");
    assert!(error.to_string().contains("overlap"));
    let content = FileSystem::read_text_file(test.env.as_ref(), "edit.txt", &background_context())
        .await
        .expect("the read back");
    assert_eq!(content, "one\ntwo\nthree\n");
}

#[tokio::test]
async fn rejects_missing_and_duplicate_target_text() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "edit.txt",
        FileContent::Text("foo foo foo".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    let tool = create_edit_tool::<EnvToolContext>();

    let missing = run_tool(
        &tool,
        json!({ "path": "edit.txt", "edits": [{ "oldText": "bar", "newText": "baz" }] }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the missing rejection");
    assert!(
        missing
            .to_string()
            .contains("Could not find the exact text")
    );

    let duplicate = run_tool(
        &tool,
        json!({ "path": "edit.txt", "edits": [{ "oldText": "foo", "newText": "bar" }] }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the duplicate rejection");
    assert!(duplicate.to_string().contains("Found 3 occurrences"));
}

#[tokio::test]
async fn keeps_the_mutation_queue_locked_until_an_aborted_edit_write_settles() {
    let fixture = blocking_edit_env();
    FileSystem::write_file(
        fixture.env.inner.as_ref(),
        "file.txt",
        FileContent::Text("alpha\nbeta\n".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    let tool = create_edit_tool::<EnvToolContext>();
    let (context, controller) = with_cancel(&background_context());
    let first = run_tool(
        &tool,
        json!({ "path": "file.txt", "edits": [{ "oldText": "alpha", "newText": "ALPHA" }] }),
        hooked_context(Arc::clone(&fixture.env)),
        None,
        &context,
    );
    let settled = background_context();
    let second = async {
        fixture.first_edit_write_started.notified().await;
        controller.abort_without_reason();
        let second = run_tool(
            &tool,
            json!({ "path": "file.txt", "edits": [{ "oldText": "beta", "newText": "BETA" }] }),
            hooked_context(Arc::clone(&fixture.env)),
            None,
            &settled,
        );
        delay(20).await;
        assert!(!fixture.second_edit_write_started.load(Ordering::Relaxed));
        fixture.finish_first_edit_write.notify_one();
        second.await
    };
    let (first_result, second_result) = tokio::join!(first, second);

    first_result.expect_err("the first edit aborts");
    second_result.expect("the second edit");
    assert!(fixture.first_edit_write_settled.load(Ordering::Relaxed));
    let written = FileSystem::read_text_file(
        fixture.env.inner.as_ref(),
        "file.txt",
        &background_context(),
    )
    .await
    .expect("the read back");
    assert_eq!(written, "ALPHA\nBETA\n");
}

#[tokio::test]
async fn serializes_concurrent_edits_through_canonical_and_symlink_paths() {
    let (root, env) = slow_read_env();
    FileSystem::write_file(
        env.inner.as_ref(),
        "target.txt",
        FileContent::Text("alpha\nbeta\ngamma\n".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    std::os::unix::fs::symlink(
        "target.txt",
        format!("{}/link.txt", FileSystem::cwd(env.as_ref())),
    )
    .expect("the symlink");
    let tool = create_edit_tool::<EnvToolContext>();

    let shared = background_context();
    let (target_result, link_result) = tokio::join!(
        run_tool(
            &tool,
            json!({ "path": "target.txt", "edits": [{ "oldText": "alpha", "newText": "ALPHA" }] }),
            hooked_context(Arc::clone(&env)),
            None,
            &shared,
        ),
        run_tool(
            &tool,
            json!({ "path": "link.txt", "edits": [{ "oldText": "beta", "newText": "BETA" }] }),
            hooked_context(Arc::clone(&env)),
            None,
            &shared,
        ),
    );
    target_result.expect("the target edit");
    link_result.expect("the link edit");
    let content =
        FileSystem::read_text_file(env.inner.as_ref(), "target.txt", &background_context())
            .await
            .expect("the read back");
    assert_eq!(content, "ALPHA\nBETA\ngamma\n");
    let _ = root;
}

#[tokio::test]
async fn edits_regular_files_through_symlinks() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "target.txt",
        FileContent::Text("before\n".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    std::os::unix::fs::symlink("target.txt", format!("{}/link.txt", test.env.cwd()))
        .expect("the symlink");

    let tool = create_edit_tool::<EnvToolContext>();
    run_tool(
        &tool,
        json!({ "path": "link.txt", "edits": [{ "oldText": "before", "newText": "after" }] }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the edit");

    let content =
        FileSystem::read_text_file(test.env.as_ref(), "target.txt", &background_context())
            .await
            .expect("the read back");
    assert_eq!(content, "after\n");
}

#[tokio::test]
async fn preserves_bom_and_crlf_line_endings() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "edit.txt",
        FileContent::Text("\u{FEFF}one\r\ntwo\r\n".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let tool = create_edit_tool::<EnvToolContext>();
    run_tool(
        &tool,
        json!({ "path": "edit.txt", "edits": [{ "oldText": "two", "newText": "TWO" }] }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the edit");

    let content = FileSystem::read_text_file(test.env.as_ref(), "edit.txt", &background_context())
        .await
        .expect("the read back");
    assert_eq!(content, "\u{FEFF}one\r\nTWO\r\n");
}

#[tokio::test]
async fn executes_commands_and_combines_stdout_and_stderr() {
    let test = create_context();
    let tool = create_bash_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "command": "printf out; printf err >&2" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the command");

    assert!(text_output(&result).contains("out"));
    assert!(text_output(&result).contains("err"));
}

#[tokio::test]
async fn reports_nonzero_exits_and_timeouts() {
    let test = create_context();
    let tool = create_bash_tool::<EnvToolContext>(None);

    let exit_error = run_tool(
        &tool,
        json!({ "command": "printf failed; exit 7" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the nonzero exit");
    let message = exit_error.to_string();
    let failed_at = message.find("failed").expect("the command's output");
    let exit_at = message
        .find("Command exited with code 7")
        .expect("the exit notice");
    assert!(failed_at < exit_at);

    let timeout_error = run_tool(
        &tool,
        json!({ "command": "sleep 2", "timeout": 0.01 }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the timeout");
    assert!(
        timeout_error
            .to_string()
            .contains("Command timed out after 0.01 seconds")
    );
}

#[tokio::test]
async fn preserves_truncated_output_when_a_command_times_out() {
    let test = create_context();
    let tool = create_bash_tool::<EnvToolContext>(None);
    let timeout_context = ToolTestContext {
        root: test.root,
        env: test.env,
        tool_context: hooked_context(timeout_output_env().1),
    };
    let error = run_tool(
        &tool,
        json!({ "command": "emit-output-then-time-out", "timeout": 0.05 }),
        timeout_context.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the timeout");

    let message = error.to_string();
    assert!(message.contains("Command timed out after 0.05 seconds"));
    let marker = "Full output: ";
    let start = message.find(marker).expect("the spill notice") + marker.len();
    let rest = &message[start..];
    let end = rest.find(['\n', ']']).expect("the notice's end");
    let full_output_path = &rest[..end];
    let full_output = FileSystem::read_text_file(
        timeout_context.env.as_ref(),
        full_output_path,
        &background_context(),
    )
    .await
    .expect("the spill read");
    assert!(full_output.contains("line-1\nline-2"));
    assert!(full_output.contains(&format!(
        "line-{DEFAULT_MAX_LINES}\nline-{}",
        DEFAULT_MAX_LINES + 1
    )));
}

#[tokio::test]
async fn ignores_output_callbacks_after_execution_settles() {
    let (_root, env) = late_output_env();
    let updates: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_sink = Arc::clone(&updates);
    let tool = create_bash_tool::<EnvToolContext>(None);
    let on_update = move |update: &AgentToolResult, _options| {
        updates_sink
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(text_output(update));
    };
    let result = run_tool(
        &tool,
        json!({ "command": "late" }),
        hooked_context(Arc::clone(&env)),
        Some(&on_update),
        &background_context(),
    )
    .await
    .expect("the command");
    delay(20).await;

    assert_eq!(text_output(&result), "before\n");
    assert!(
        !updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .any(|update| update.contains("late"))
    );
}

#[tokio::test]
async fn reports_the_total_size_of_an_oversized_final_line() {
    let test = create_context();
    let tool = create_bash_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "command": "printf '%060000d' 0" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the command");

    assert!(
        text_output(&result)
            .contains("[Showing last 50.0KB of line 1 (line is 58.6KB). Full output:")
    );
}

/// The richer tool context the prepare test carries, upstream's
/// `{ env, workspace }` context type.
struct PrepareToolContext {
    env: Arc<dyn ExecutionEnv>,
    workspace: String,
}

impl ExecutionToolContext for PrepareToolContext {
    fn env(&self) -> &Arc<dyn ExecutionEnv> {
        &self.env
    }
}

#[tokio::test]
async fn prepares_command_cwd_and_an_explicit_environment_with_the_turn_context() {
    let root = tempfile::tempdir().expect("tempdir");
    let env = Arc::new(NodeExecutionEnv::new(
        root.path().to_string_lossy().into_owned(),
        None,
        Some(BTreeMap::from([(
            "PI_BASH_PREPARE_INHERITED".to_owned(),
            "inherited".to_owned(),
        )])),
    ));
    FileSystem::create_dir(env.as_ref(), "workspace", None, &background_context())
        .await
        .expect("the workspace dir");
    let workspace = format!("{}/workspace", env.cwd());
    let holder = Arc::new(PrepareToolContext {
        env: env.clone(),
        workspace: workspace.clone(),
    });
    let holder_address = Arc::as_ptr(&holder).cast::<()>() as usize;
    let received_address: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(None));
    let received_workspace: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let received_signal: Arc<Mutex<Option<crate::harness::context::AbortSignal>>> =
        Arc::new(Mutex::new(None));
    let received_address_sink = Arc::clone(&received_address);
    let received_workspace_sink = Arc::clone(&received_workspace);
    let received_signal_sink = Arc::clone(&received_signal);
    let (call_context, controller) = with_cancel(&background_context());
    let tool = create_bash_tool::<PrepareToolContext>(Some(BashToolOptions {
        command_prefix: Some("prefix=ready".to_owned()),
        prepare: Some(Arc::new(
            move |execution: &mut BashExecution,
                  tool_context: &PrepareToolContext,
                  call_context: &Context| {
                *received_address_sink
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    Some(std::ptr::from_ref(tool_context).cast::<()>() as usize);
                *received_workspace_sink
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    Some(tool_context.workspace.clone());
                *received_signal_sink
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    call_context.abort_signal();
                execution.cwd = tool_context.workspace.clone();
                execution.env = BTreeMap::from([(
                    "PI_BASH_PREPARE_EXPLICIT".to_owned(),
                    "explicit".to_owned(),
                )]);
                execution.inherit_env = false;
                execution.command += "\nprintf '%s:%s:%s:%s' \"$prefix\" \"${PI_BASH_PREPARE_INHERITED-}\" \"$PI_BASH_PREPARE_EXPLICIT\" \"$PWD\"";
                Box::pin(async {})
            },
        )),
    }));

    let tool_context: crate::harness::types::ToolContext = Some(holder);
    let result = run_tool(
        &tool,
        json!({ "command": ":" }),
        tool_context.clone(),
        None,
        &call_context,
    )
    .await
    .expect("the command");

    assert_eq!(
        received_address
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .expect("the context arrived"),
        holder_address
    );
    assert_eq!(
        received_workspace
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .expect("the workspace arrived"),
        workspace
    );
    let received_signal = received_signal
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("the signal arrived");
    controller.abort("test");
    assert!(received_signal.aborted());
    let expected_cwd = FileSystem::canonical_path(env.as_ref(), &workspace, &background_context())
        .await
        .expect("the canonical path");
    assert_eq!(
        text_output(&result),
        format!("ready::explicit:{expected_cwd}")
    );
    let _ = root;
}

#[tokio::test]
async fn supports_command_prefixes() {
    let test = create_context();
    let tool = create_bash_tool::<EnvToolContext>(Some(BashToolOptions {
        command_prefix: Some("value=hello".to_owned()),
        prepare: None,
    }));
    let result = run_tool(
        &tool,
        json!({ "command": "printf $value" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the command");

    assert_eq!(text_output(&result), "hello");
}

#[tokio::test(start_paused = true)]
async fn requests_distinct_bounded_checkpoints_at_most_every_two_seconds() {
    let (_root, env) = checkpoint_output_env();
    let checkpoints: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let checkpoints_sink = Arc::clone(&checkpoints);
    let tool = create_bash_tool::<EnvToolContext>(None);
    let on_update =
        move |update: &AgentToolResult,
              options: Option<crate::harness::types::AgentHarnessToolUpdateOptions>| {
            if options.is_some_and(|options| options.checkpoint) {
                checkpoints_sink
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(text_output(update));
            }
        };

    // The clock driver advances the paused clock so the environment's paced
    // delays settle while the execution future awaits, upstream's
    // `advanceTimersByTimeAsync`.
    let stop = Arc::new(AtomicBool::new(false));
    let stop_driver = Arc::clone(&stop);
    let driver = tokio::spawn(async move {
        while !stop_driver.load(Ordering::Relaxed) {
            tokio::time::advance(Duration::from_millis(25)).await;
        }
    });
    let result = run_tool(
        &tool,
        json!({ "command": "controlled" }),
        hooked_context(Arc::clone(&env)),
        Some(&on_update),
        &background_context(),
    )
    .await
    .expect("the command");
    stop.store(true, Ordering::Relaxed);
    driver.await.expect("the clock driver");

    assert_eq!(
        checkpoints
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone(),
        vec![
            "one\ntwo\n".to_owned(),
            "one\ntwo\nthree\nfour\n".to_owned()
        ]
    );
    assert!(text_output(&result).contains("four"));
}

#[tokio::test]
async fn coalesces_updates_and_persists_truncated_full_output() {
    let test = create_context();
    let updates: Arc<Mutex<Vec<AgentToolResult>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_sink = Arc::clone(&updates);
    let tool = create_bash_tool::<EnvToolContext>(None);
    let on_update = move |update: &AgentToolResult, _options| {
        updates_sink
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(update.clone());
    };
    let result = run_tool(
        &tool,
        json!({ "command": "i=1; while [ $i -le 3000 ]; do echo line-$i; i=$((i + 1)); done" }),
        test.tool_context.clone(),
        Some(&on_update),
        &background_context(),
    )
    .await
    .expect("the command");

    assert!(
        updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
            < 25
    );
    let details = &result.details;
    assert_eq!(details["truncation"]["truncated"], json!(true));
    assert_eq!(details["truncation"]["truncatedBy"], json!("lines"));
    assert_eq!(details["truncation"]["totalLines"], json!(3000));
    assert_eq!(details["truncation"]["outputLines"], json!(2000));
    assert!(text_output(&result).contains("line-3000"));
    let full_output_path = details["fullOutputPath"]
        .as_str()
        .expect("the spill path")
        .to_owned();
    let final_update = updates
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .last()
        .cloned()
        .expect("updates were published");
    assert!(text_output(&final_update).contains("line-3000"));
    assert_eq!(
        final_update.details["truncation"]["totalLines"],
        json!(3000)
    );
    assert_eq!(
        final_update.details["fullOutputPath"],
        json!(full_output_path)
    );
    let full_output =
        FileSystem::read_text_file(test.env.as_ref(), &full_output_path, &background_context())
            .await
            .expect("the spill read");
    assert!(full_output.contains("line-1\nline-2"));
    assert!(full_output.contains("line-2999\nline-3000"));
}

// ---------------------------------------------------------------------------
// Boundary tests binding the restated branches
// ---------------------------------------------------------------------------

/// The unified patch's wire shape, goldens computed from npm `diff` 8.0.4
/// at the pin: the count-0 `@@` decrement quirk, the `\ No newline` markers,
/// and the multi-hunk context split.
#[test]
fn the_unified_patch_matches_the_npm_diff_goldens() {
    use crate::harness::tools::edit_diff::generate_unified_patch;
    let cases: &[(&str, &str, usize, &str)] = &[
        (
            "one\ntwo\nthree\n",
            "one\nTWO\nthree\n",
            2,
            "--- f\n+++ f\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n",
        ),
        (
            "one\ntwo\nthree",
            "one\ntwo\nthree\n",
            2,
            "--- f\n+++ f\n@@ -1,3 +1,3 @@\n one\n two\n-three\n\\ No newline at end of file\n+three\n",
        ),
        (
            "one\ntwo",
            "one\nTWO",
            2,
            "--- f\n+++ f\n@@ -1,2 +1,2 @@\n one\n-two\n\\ No newline at end of file\n+TWO\n\\ No newline at end of file\n",
        ),
        (
            "a\nc\n",
            "a\nb\nc\n",
            0,
            "--- f\n+++ f\n@@ -1,0 +2,1 @@\n+b\n",
        ),
        (
            "a\nb\nc\n",
            "a\nc\n",
            0,
            "--- f\n+++ f\n@@ -2,1 +1,0 @@\n-b\n",
        ),
        ("", "x\n", 0, "--- f\n+++ f\n@@ -0,0 +1,1 @@\n+x\n"),
        ("x\n", "", 0, "--- f\n+++ f\n@@ -1,1 +0,0 @@\n-x\n"),
        (
            "one\n",
            "one\ntwo\n",
            4,
            "--- f\n+++ f\n@@ -1,1 +1,2 @@\n one\n+two\n",
        ),
        ("a\n", "a\n", 4, "--- f\n+++ f\n"),
        (
            "1\nX\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\nY\n15\n",
            "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n",
            4,
            "--- f\n+++ f\n@@ -1,6 +1,6 @@\n 1\n-X\n+2\n 3\n 4\n 5\n 6\n@@ -10,6 +10,6 @@\n 10\n 11\n 12\n 13\n-Y\n+14\n 15\n",
        ),
    ];
    for (old, new, context, expected) in cases {
        assert_eq!(&generate_unified_patch("f", old, new, *context), expected);
    }
}

/// The display diff's goldens, computed from npm `diffLines` run through
/// upstream's formatter: the context elision and the first changed line.
#[test]
fn the_display_diff_matches_the_npm_diff_goldens() {
    use crate::harness::tools::edit_diff::generate_diff_string;
    let result = generate_diff_string(
        "alpha\nbeta\ngamma\ndelta\n",
        "ALPHA\nbeta\nGAMMA\ndelta\n",
        4,
    );
    assert_eq!(
        result.diff,
        "-1 alpha\n+1 ALPHA\n 2 beta\n-3 gamma\n+3 GAMMA\n 4 delta"
    );
    assert_eq!(result.first_changed_line, Some(1));

    let result = generate_diff_string(
        "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12\n",
        "l1\nX2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nX10\nl11\nl12\n",
        2,
    );
    assert_eq!(
        result.diff,
        "  1 l1\n- 2 l2\n+ 2 X2\n  3 l3\n  4 l4\n    ...\n  8 l8\n  9 l9\n-10 l10\n+10 X10\n 11 l11\n 12 l12"
    );
    assert_eq!(result.first_changed_line, Some(2));

    // The leading-only elision clamps to the run's length (upstream's
    // `slice(0, contextLines)`).
    let result = generate_diff_string("one\ntwo\nthree\n", "one\nTWO\nthree\n", 4);
    assert_eq!(result.diff, " 1 one\n-2 two\n+2 TWO\n 3 three");
    assert_eq!(result.first_changed_line, Some(2));

    let result = generate_diff_string("a\nb\n", "a\nb\n", 4);
    assert_eq!(result.diff, "");
    assert_eq!(result.first_changed_line, None);
}

/// Fuzzy matching: the trailing-whitespace and smart-quote normalizations
/// find and replace in normalized space, overlaying the touched lines onto
/// the original so untouched lines keep their bytes.
#[test]
fn fuzzy_matching_normalizes_whitespace_and_quotes() {
    use crate::harness::tools::edit_diff::fuzzy_find_text;

    // Trailing whitespace: the exact match misses, the fuzzy one hits.
    let content = "alpha\nbeta  \ngamma\n";
    let found = fuzzy_find_text(content, "beta\n");
    assert!(found.found);
    assert!(found.used_fuzzy_match);

    let applied = apply_edits_to_normalized_content(
        content,
        &[Edit {
            old_text: "beta\n".to_owned(),
            new_text: "BETA\n".to_owned(),
        }],
        "fuzzy.txt",
    )
    .expect("the fuzzy edit");
    assert_eq!(applied.new_content, "alpha\nBETA\ngamma\n");

    // Smart quotes normalize to ASCII in both directions.
    let quoted = "say \u{201C}hello\u{201D}\n";
    let applied = apply_edits_to_normalized_content(
        quoted,
        &[Edit {
            old_text: "say \"hello\"\n".to_owned(),
            new_text: "OK\n".to_owned(),
        }],
        "quotes.txt",
    )
    .expect("the quoted edit");
    assert_eq!(applied.new_content, "OK\n");

    // The exact match wins when one exists.
    let exact = fuzzy_find_text("alpha\n", "alpha");
    assert!(exact.found);
    assert!(!exact.used_fuzzy_match);
    assert!(!fuzzy_find_text("alpha\n", "omega").found);
}

/// The edit rejections' messages, single and multiple edits, upstream's
/// thrown texts verbatim.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the case table mirrors upstream's rejection messages; a data-driven table would repeat the shape per message arity"
)]
fn the_edit_rejections_carry_the_upstream_messages() {
    let single = |old_text: &str| {
        vec![Edit {
            old_text: old_text.to_owned(),
            new_text: "x".to_owned(),
        }]
    };
    let empty_single = apply_edits_to_normalized_content("a\n", &single(""), "f.txt")
        .expect_err("the empty rejection");
    assert_eq!(empty_single, "oldText must not be empty in f.txt.");
    let empty_multi = apply_edits_to_normalized_content(
        "a\n",
        &[
            Edit {
                old_text: "a".to_owned(),
                new_text: "b".to_owned(),
            },
            Edit {
                old_text: String::new(),
                new_text: "b".to_owned(),
            },
        ],
        "f.txt",
    )
    .expect_err("the empty rejection");
    assert_eq!(empty_multi, "edits[1].oldText must not be empty in f.txt.");

    let not_found = apply_edits_to_normalized_content("a\n", &single("zzz"), "f.txt")
        .expect_err("the not-found rejection");
    assert_eq!(
        not_found,
        "Could not find the exact text in f.txt. The old text must match exactly including all whitespace and newlines."
    );
    let not_found_multi = apply_edits_to_normalized_content(
        "a\n",
        &[
            Edit {
                old_text: "a".to_owned(),
                new_text: "b".to_owned(),
            },
            Edit {
                old_text: "zzz".to_owned(),
                new_text: "b".to_owned(),
            },
        ],
        "f.txt",
    )
    .expect_err("the not-found rejection");
    assert_eq!(
        not_found_multi,
        "Could not find edits[1] in f.txt. The oldText must match exactly including all whitespace and newlines."
    );

    let duplicate = apply_edits_to_normalized_content("a\na\n", &single("a"), "f.txt")
        .expect_err("the duplicate rejection");
    assert_eq!(
        duplicate,
        "Found 2 occurrences of the text in f.txt. The text must be unique. Please provide more context to make it unique."
    );
    let duplicate_multi = apply_edits_to_normalized_content(
        "a\na\n",
        &[
            Edit {
                old_text: "a".to_owned(),
                new_text: "b".to_owned(),
            },
            Edit {
                old_text: "a".to_owned(),
                new_text: "c".to_owned(),
            },
        ],
        "f.txt",
    )
    .expect_err("the duplicate rejection");
    assert_eq!(
        duplicate_multi,
        "Found 2 occurrences of edits[0] in f.txt. Each oldText must be unique. Please provide more context to make it unique."
    );

    let no_change = apply_edits_to_normalized_content(
        "a\n",
        &[Edit {
            old_text: "a".to_owned(),
            new_text: "a".to_owned(),
        }],
        "f.txt",
    )
    .expect_err("the no-change rejection");
    assert_eq!(
        no_change,
        "No changes made to f.txt. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
    );
    let no_change_multi = apply_edits_to_normalized_content(
        "ab\n",
        &[
            Edit {
                old_text: "a".to_owned(),
                new_text: "a".to_owned(),
            },
            Edit {
                old_text: "b".to_owned(),
                new_text: "b".to_owned(),
            },
        ],
        "f.txt",
    )
    .expect_err("the no-change rejection");
    assert_eq!(
        no_change_multi,
        "No changes made to f.txt. The replacements produced identical content."
    );

    let overlap = apply_edits_to_normalized_content(
        "abcd\n",
        &[
            Edit {
                old_text: "ab".to_owned(),
                new_text: "X".to_owned(),
            },
            Edit {
                old_text: "bc".to_owned(),
                new_text: "Y".to_owned(),
            },
        ],
        "f.txt",
    )
    .expect_err("the overlap rejection");
    assert_eq!(
        overlap,
        "edits[0] and edits[1] overlap in f.txt. Merge them into one edit or target disjoint regions."
    );
}

/// The line tokenizer, jsdiff's `tokenize`: separators merge into the
/// preceding token, CRLF is one separator, lone CR is content.
#[test]
fn the_line_tokenizer_splits_on_lf_and_crlf() {
    use crate::harness::tools::edit_diff::line_diff;

    let parts = line_diff("a\r\nb\nc", "a\r\nB\nc");
    assert_eq!(
        parts,
        vec![
            DiffPart {
                added: false,
                removed: false,
                value: "a\r\n".to_owned()
            },
            DiffPart {
                added: false,
                removed: true,
                value: "b\n".to_owned()
            },
            DiffPart {
                added: true,
                removed: false,
                value: "B\n".to_owned()
            },
            DiffPart {
                added: false,
                removed: false,
                value: "c".to_owned()
            },
        ]
    );
}

/// The image sniffer's boundary cases: exotic JPEG markers, animated PNG,
/// malformed BMPs, and the base64 padding ladder.
#[test]
fn the_image_sniffer_rejects_malformed_variants() {
    assert_eq!(
        detect_supported_image_mime_type(&[0xff, 0xd8, 0xff, 0xe0]),
        Some("image/jpeg")
    );
    assert_eq!(
        detect_supported_image_mime_type(&[0xff, 0xd8, 0xff, 0xf7]),
        None
    );
    assert_eq!(
        detect_supported_image_mime_type(&[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, b'I', b'H', b'D', b'R'
        ]),
        Some("image/png")
    );
    // Animated PNG: an acTL chunk before IDAT.
    let mut apng = vec![0x89_u8, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    apng.extend_from_slice(&8_u32.to_be_bytes());
    apng.extend_from_slice(b"acTL");
    apng.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
    apng.extend_from_slice(&0_u32.to_be_bytes());
    apng.extend_from_slice(b"IDAT");
    assert_eq!(detect_supported_image_mime_type(&apng), None);
    // A PNG whose IHDR length disagrees is not a readable PNG.
    assert_eq!(
        detect_supported_image_mime_type(&[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 9, b'I', b'H', b'D', b'R'
        ]),
        None
    );
    assert_eq!(
        detect_supported_image_mime_type(b"GIF89a"),
        Some("image/gif")
    );
    let mut webp = b"RIFF".to_vec();
    webp.extend_from_slice(&12_u32.to_le_bytes());
    webp.extend_from_slice(b"WEBPVP8 ");
    assert_eq!(detect_supported_image_mime_type(&webp), Some("image/webp"));
    assert_eq!(
        detect_supported_image_mime_type(&create_tiny_bmp()),
        Some("image/bmp")
    );
    // BMP with the wrong plane count is not a readable BMP.
    let mut bad_bmp = create_tiny_bmp();
    bad_bmp[26] = 2;
    assert_eq!(detect_supported_image_mime_type(&bad_bmp), None);
    assert_eq!(detect_supported_image_mime_type(b""), None);
    assert_eq!(detect_supported_image_mime_type(b"BM"), None);

    assert_eq!(encode_base64(b"x"), "eA==");
    assert_eq!(encode_base64(b"xy"), "eHk=");
    assert_eq!(encode_base64(b"xyz"), "eHl6");
}

/// The path resolution's variant ladder through the read tool: the narrow
/// no-break space before `AM.`, the NFD decomposition, and the typed
/// apostrophe all resolve to the on-disk names.
#[tokio::test]
async fn the_read_path_variants_resolve_ondisk_names() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "10\u{202F}AM.txt",
        FileContent::Text("am".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    FileSystem::write_file(
        test.env.as_ref(),
        "nai\u{0308}ve.txt",
        FileContent::Text("naive".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    FileSystem::write_file(
        test.env.as_ref(),
        "it\u{2019}s.txt",
        FileContent::Text("apos".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let tool = create_read_tool::<EnvToolContext>(None);
    let ampm = run_tool(
        &tool,
        json!({ "path": "10 AM.txt" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the narrow-space variant");
    assert!(text_output(&ampm).contains("am"));
    let nfd = run_tool(
        &tool,
        json!({ "path": "na\u{00EF}ve.txt" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the NFD variant");
    assert!(text_output(&nfd).contains("naive"));
    let apostrophe = run_tool(
        &tool,
        json!({ "path": "it's.txt" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the apostrophe variant");
    assert!(text_output(&apostrophe).contains("apos"));
}

/// The tool-path normalization through the write tool: the `@` prefix and
/// the Unicode spaces normalize away.
#[tokio::test]
async fn the_tool_path_normalizes_at_prefix_and_unicode_spaces() {
    let test = create_context();
    let tool = create_write_tool::<EnvToolContext>();
    run_tool(
        &tool,
        json!({ "path": "@nested/dir/file.txt", "content": "hello" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the write");
    let content = FileSystem::read_text_file(
        test.env.as_ref(),
        "nested/dir/file.txt",
        &background_context(),
    )
    .await
    .expect("the read back");
    assert_eq!(content, "hello");

    run_tool(
        &tool,
        json!({ "path": "a\u{00A0}b\u{202F}c.txt", "content": "spaced" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the write");
    let content = FileSystem::read_text_file(test.env.as_ref(), "a b c.txt", &background_context())
        .await
        .expect("the read back");
    assert_eq!(content, "spaced");
}

/// The read tool's BMP fallback without a processor and the oversized
/// first-line notice.
#[tokio::test]
async fn the_read_tool_reports_bmp_and_oversized_first_lines() {
    let test = create_context();
    FileSystem::write_file(
        test.env.as_ref(),
        "image.bmp",
        FileContent::Bytes(create_tiny_bmp()),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    let tool = create_read_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "path": "image.bmp" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the read");
    assert_eq!(
        text_output(&result),
        "Read image file [image/bmp]\n[Image omitted: configure an imageProcessor to convert BMP images.]"
    );

    // A first line over the byte limit reports the sed hint instead.
    let big_line = "x".repeat(60_000);
    FileSystem::write_file(
        test.env.as_ref(),
        "big.txt",
        FileContent::Text(format!("{big_line}\ntail\n")),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    let result = run_tool(
        &tool,
        json!({ "path": "big.txt" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the read");
    assert_eq!(
        text_output(&result),
        "[Line 1 is 58.6KB, exceeds 50.0KB limit. Use bash: sed -n '1p' big.txt | head -c 51200]"
    );
}

/// The byte-limit truncation's continuation notice.
#[tokio::test]
async fn the_read_tool_reports_the_byte_limit_continuation() {
    let test = create_context();
    let line = "y".repeat(600);
    FileSystem::write_file(
        test.env.as_ref(),
        "wide.txt",
        FileContent::Text(
            (1..=100)
                .map(|index| format!("w{index} {line}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    let tool = create_read_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "path": "wide.txt" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the read");
    let output = text_output(&result);
    assert!(output.contains("(50.0KB limit). Use offset="));
    let truncation = &result.details["truncation"];
    assert_eq!(truncation["truncatedBy"], json!("bytes"));
}

/// The context downcast failures, upstream's type-level
/// `TContext extends ExecutionToolContext` constraint surfacing as errors.
#[tokio::test]
async fn the_builtin_tools_reject_missing_and_mistyped_contexts() {
    let test = create_context();
    let tool = create_write_tool::<EnvToolContext>();

    let missing = run_tool(
        &tool,
        json!({ "path": "x", "content": "y" }),
        None,
        None,
        &background_context(),
    )
    .await
    .expect_err("the missing context");
    assert!(
        missing
            .to_string()
            .contains("without its execution tool context")
    );

    let mistyped_tool = create_write_tool::<OtherContext>();
    let mistyped = run_tool(
        &mistyped_tool,
        json!({ "path": "x", "content": "y" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the mistyped context");
    assert!(
        mistyped
            .to_string()
            .contains("expected execution tool context type")
    );
}

/// The mutation queue's canonicalization fallback: when canonical paths are
/// unsupported, the absolute path keys the queue and concurrent edits still
/// serialize.
#[tokio::test]
async fn the_mutation_queue_falls_back_to_absolute_paths() {
    use crate::harness::test_support::FaultEnv;

    let root = tempfile::tempdir().expect("tempdir");
    let mut env = FaultEnv::new(root.path());
    env.canonical_path_fault = Some(crate::harness::test_support::Fault {
        path_contains: "",
        code: crate::harness::types::FileErrorCode::NotSupported,
        message: "canonical paths are not supported",
    });
    FileSystem::write_file(
        &env,
        "file.txt",
        FileContent::Text("alpha\nbeta\ngamma\n".to_owned()),
        &background_context(),
    )
    .await
    .expect("the fixture write");

    let env: Arc<dyn ExecutionEnv> = Arc::new(env);
    let tool = create_edit_tool::<EnvToolContext>();
    let tool_context: crate::harness::types::ToolContext = Some(Arc::new(EnvToolContext {
        env: Arc::clone(&env),
    }));
    let shared = background_context();
    let (first, second) = tokio::join!(
        run_tool(
            &tool,
            json!({ "path": "file.txt", "edits": [{ "oldText": "alpha", "newText": "ALPHA" }] }),
            tool_context.clone(),
            None,
            &shared,
        ),
        run_tool(
            &tool,
            json!({ "path": "./file.txt", "edits": [{ "oldText": "beta", "newText": "BETA" }] }),
            tool_context.clone(),
            None,
            &shared,
        ),
    );
    first.expect("the first edit");
    second.expect("the second edit");
    let content = FileSystem::read_text_file(env.as_ref(), "file.txt", &background_context())
        .await
        .expect("the read back");
    assert_eq!(content, "ALPHA\nBETA\ngamma\n");
}

/// The mistyped-context probe, upstream's richer `TContext` types that do
/// not match the creator's.
struct OtherContext;

impl ExecutionToolContext for OtherContext {
    fn env(&self) -> &Arc<dyn ExecutionEnv> {
        unreachable!("never resolved")
    }
}

/// The bash timeout validation's rejections.
#[tokio::test]
async fn the_bash_tool_rejects_invalid_timeouts() {
    let test = create_context();
    let tool = create_bash_tool::<EnvToolContext>(None);

    let negative = run_tool(
        &tool,
        json!({ "command": ":", "timeout": -1 }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the negative timeout");
    assert!(
        negative
            .to_string()
            .contains("Invalid timeout: must be a finite number of seconds")
    );

    let overflow = run_tool(
        &tool,
        json!({ "command": ":", "timeout": 2_147_483.648 }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the oversized timeout");
    assert!(
        overflow
            .to_string()
            .contains("Invalid timeout: maximum is 2147483.647 seconds")
    );
}

/// A command with no output reports the `(no output)` placeholder, and a
/// command aborted before it runs reports the abort status.
#[tokio::test]
async fn the_bash_tool_reports_no_output_and_aborts() {
    let test = create_context();
    let tool = create_bash_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "command": "true" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the command");
    assert_eq!(text_output(&result), "(no output)");

    let (context, controller) = with_cancel(&background_context());
    controller.abort("done");
    let error = run_tool(
        &tool,
        json!({ "command": "printf out" }),
        test.tool_context.clone(),
        None,
        &context,
    )
    .await
    .expect_err("the abort");
    assert!(error.to_string().contains("Command aborted"));
}

/// The edit tool's pre-validation shim, upstream's `prepareEditArguments`:
/// `edits` as a JSON string, as a single-edit object, and the legacy flat
/// `oldText`/`newText` fields all restate into the array shape.
#[test]
fn the_edit_argument_shim_restates_the_legacy_shapes() {
    let tool = create_edit_tool::<EnvToolContext>();
    let prepare = tool.prepare_arguments.as_ref().expect("the edit shim");

    // edits as a JSON string
    let shimmed =
        prepare(&json!({ "path": "f", "edits": "[{\"oldText\":\"a\",\"newText\":\"b\"}]" }))
            .expect("the string form");
    assert_eq!(
        shimmed,
        json!({ "path": "f", "edits": [{ "oldText": "a", "newText": "b" }] })
    );

    // edits as a single-edit object
    let shimmed = prepare(&json!({ "path": "f", "edits": { "oldText": "a", "newText": "b" } }))
        .expect("the single-edit form");
    assert_eq!(
        shimmed,
        json!({ "path": "f", "edits": [{ "oldText": "a", "newText": "b" }] })
    );

    // edits as a JSON string holding a single-edit object
    let shimmed =
        prepare(&json!({ "path": "f", "edits": "{\"oldText\":\"a\",\"newText\":\"b\"}" }))
            .expect("the string single-edit form");
    assert_eq!(
        shimmed,
        json!({ "path": "f", "edits": [{ "oldText": "a", "newText": "b" }] })
    );

    // the legacy flat fields append to the existing edits
    let shimmed = prepare(&json!({
        "path": "f",
        "oldText": "a",
        "newText": "b",
        "edits": [{ "oldText": "c", "newText": "d" }]
    }))
    .expect("the legacy form");
    assert_eq!(
        shimmed,
        json!({
            "path": "f",
            "edits": [
                { "oldText": "c", "newText": "d" },
                { "oldText": "a", "newText": "b" }
            ]
        })
    );

    // the legacy flat fields without existing edits
    let shimmed = prepare(&json!({ "path": "f", "oldText": "a", "newText": "b" }))
        .expect("the bare legacy form");
    assert_eq!(
        shimmed,
        json!({ "path": "f", "edits": [{ "oldText": "a", "newText": "b" }] })
    );

    // malformed JSON strings and non-edit objects pass through untouched
    let passthrough = prepare(&json!({ "path": "f", "edits": "{not json" }))
        .expect("the malformed string passes through");
    assert_eq!(passthrough, json!({ "path": "f", "edits": "{not json" }));
    let passthrough = prepare(&json!({ "path": "f", "edits": { "other": 1 } }))
        .expect("the non-edit object passes through");
    assert_eq!(passthrough, json!({ "path": "f", "edits": { "other": 1 } }));
}

/// The edit tool's file-access failures: a missing file and a directory
/// target, upstream's `editAccessError` messages.
#[tokio::test]
async fn the_edit_tool_reports_file_access_errors() {
    let test = create_context();
    FileSystem::create_dir(test.env.as_ref(), "subdir", None, &background_context())
        .await
        .expect("the dir");
    let tool = create_edit_tool::<EnvToolContext>();

    let missing = run_tool(
        &tool,
        json!({ "path": "missing.txt", "edits": [{ "oldText": "a", "newText": "b" }] }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the missing file");
    assert_eq!(
        missing.to_string(),
        "Could not edit file: missing.txt. Error code: not_found."
    );

    let directory = run_tool(
        &tool,
        json!({ "path": "subdir", "edits": [{ "oldText": "a", "newText": "b" }] }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect_err("the directory target");
    assert_eq!(
        directory.to_string(),
        "Could not edit file: subdir. Path is not a file."
    );
}

/// The read tool's processor-failure branch renders the message as text.
#[tokio::test]
async fn the_read_tool_reports_a_failed_processor() {
    let test = create_context();
    let png = base64::engine::general_purpose::STANDARD
        .decode(PNG_FIXTURE)
        .expect("the fixture decodes");
    FileSystem::write_file(
        test.env.as_ref(),
        "image.png",
        FileContent::Bytes(png),
        &background_context(),
    )
    .await
    .expect("the fixture write");
    let tool = create_read_tool::<EnvToolContext>(Some(ReadToolOptions {
        auto_resize_images: None,
        image_processor: Some(Arc::new(
            |_bytes: &[u8],
             _mime_type: &str,
             _options: &ReadImageProcessorOptions,
             _context: &Context| {
                Box::pin(async {
                    Ok(ReadImageProcessorResult::Failed {
                        message: "the decoder refused the bytes".to_owned(),
                    })
                })
            },
        )),
    }));

    let result = run_tool(
        &tool,
        json!({ "path": "image.png" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the read");
    assert_eq!(
        text_output(&result),
        "Read image file [image/png]\nthe decoder refused the bytes"
    );
}

/// The bash tool's byte-limit suffix: output truncated by bytes across
/// complete lines.
#[tokio::test]
async fn the_bash_tool_reports_the_byte_limit_suffix() {
    let test = create_context();
    let tool = create_bash_tool::<EnvToolContext>(None);
    let result = run_tool(
        &tool,
        json!({ "command": "yes 0123456789012345678901234567890123456789012345678901234567890 | head -n 1000" }),
        test.tool_context.clone(),
        None,
        &background_context(),
    )
    .await
    .expect("the command");

    let output = text_output(&result);
    assert!(output.contains("(50.0KB limit). Full output:"));
}

/// The generated patch round-trips through the ported applier over random
/// line arrays: apply(generate(old, new)) == new. The property walks the
/// Myers search's diagonals the fixed cases never reach.
#[test]
#[expect(
    clippy::panic,
    reason = "a failed round trip is the test's failure mode; the panic carries the diverging case"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "the u32 seed loses nothing to f64's 52-bit mantissa; the fraction drives the fixture sizes"
)]
fn the_patch_round_trips_over_random_line_arrays() {
    use crate::harness::tools::edit_diff::generate_unified_patch;

    let mut seed: u64 = 0x1234_5678;
    let mut random = || {
        seed = seed.wrapping_mul(16_645_225).wrapping_add(1_013_904_223) & 0xffff_ffff;
        (seed as f64) / 4_294_967_296.0
    };
    let alphabet = ["a\n", "b\n", "c\n", "d\n", "e\n"];
    for _case in 0..300 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the fixture sizes are the LCG fraction's integral part"
        )]
        let old_len = (random() * 14.0) as usize;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the fixture sizes are the LCG fraction's integral part"
        )]
        let new_len = (random() * 14.0) as usize;
        let old: String = (0..old_len)
            .map(|_| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the alphabet index is the LCG fraction's integral part"
                )]
                alphabet[(random() * 5.0) as usize]
            })
            .collect();
        let new: String = (0..new_len)
            .map(|_| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the alphabet index is the LCG fraction's integral part"
                )]
                alphabet[(random() * 5.0) as usize]
            })
            .collect();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the context count is the LCG fraction's integral part"
        )]
        let context_lines = (random() * 4.0) as usize;
        let patch = generate_unified_patch("f", &old, &new, context_lines);
        let applied = apply_patch(&old, &patch).unwrap_or_else(|| {
            panic!("the patch failed to apply: old={old:?} new={new:?} ctx={context_lines}")
        });
        assert_eq!(
            applied, new,
            "the applied patch diverged: old={old:?} new={new:?} ctx={context_lines} patch={patch:?}"
        );
    }
}
