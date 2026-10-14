//! The bash and powershell tools, ported from upstream
//! `src/core/tools/bash.ts` and `powershell.ts`.
//!
//! The shell tools share one definition body: the pluggable
//! [`BashOperations`] seam (local spawn, or a remote delegate), the spawn
//! context the session environment rides, and the throttled streaming
//! update loop over the [`super::output_accumulator::OutputAccumulator`].
//! The renderer slice upstream spreads onto the definition
//! (`createShellRenderers`, `BASH_UPDATE_THROTTLE_MS`'s home) rides the
//! theme ticket; the throttle constant lands here with the loop that uses
//! it. Upstream's `Object.assign(tool, { promptSnippet, promptGuidelines })`
//! has no harness-tool counterpart — the system-prompt contributions ride
//! the definitions the session registry carries.
//!
//! The throttled pump restates upstream's `updateTimer` set: an immediate
//! emit when the spacing has elapsed (the initial `lastUpdateAt = 0` makes
//! the first update immediate), otherwise one trailing timer the `??=` arm
//! only arms once; the pump loop is the execute future's body, so the
//! update-callback borrow lives exactly as long as the execution.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::harness::types::{AgentHarnessTool, AgentHarnessToolUpdateCallback};
use pi_agent_core::types::{AgentToolContent, AgentToolError, AgentToolResult};
use pi_ai::types::{BoxedFuture, TextContent};
use serde_json::{Value, json};

use crate::extensions::types::{ExtensionContext, ToolDefinition};
use crate::utils::shell::{
    CommandTransport, ShellConfig, ShellError, get_shell_config, get_shell_env,
};

use super::output_accumulator::{OutputAccumulator, OutputAccumulatorOptions};
use super::strict_sampling;
use super::tool_schema;
use super::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationResult, format_size,
};

/// The maximum accepted timeout in milliseconds, upstream's
/// `MAX_TIMEOUT_MS`.
const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;

/// The spacing between streaming updates, upstream's
/// `BASH_UPDATE_THROTTLE_MS` (its renderer-module home rides the theme
/// ticket).
const BASH_UPDATE_THROTTLE_MS: u64 = 100;

/// The system-prompt contribution shape every built-in tool carries,
/// upstream's `*ToolSystemPromptContribution` constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemPromptContribution {
    /// The Available tools section snippet.
    pub snippet: &'static str,
    /// The Guidelines section bullets.
    pub guidelines: &'static [&'static str],
}

/// The bash tool's system-prompt contribution, upstream's
/// `bashToolSystemPromptContribution`.
pub const BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION: SystemPromptContribution =
    SystemPromptContribution {
        snippet: "Execute bash commands (ls, grep, find, etc.)",
        guidelines: &[
            "You can inspect PI_* environment variables for current model and session details.",
        ],
    };

/// The tool input, upstream's `BashToolInput` (`{ command, timeout? }`).
#[derive(Clone, Debug, PartialEq)]
pub struct BashToolInput {
    /// The command to execute.
    pub command: String,
    /// The timeout in seconds.
    pub timeout: Option<f64>,
}

/// The tool's structured details, upstream's `BashToolDetails`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BashToolDetails {
    /// The truncation metadata, when the output truncated.
    pub truncation: Option<TruncationResult>,
    /// The temp file preserving the full output.
    pub full_output_path: Option<String>,
}

impl BashToolDetails {
    /// The wire shape with absent members dropped, upstream's object with
    /// `undefined` fields omitted.
    pub(crate) fn to_wire(&self) -> Value {
        let mut object = serde_json::Map::new();
        if let Some(truncation) = &self.truncation {
            object.insert(
                "truncation".to_owned(),
                serde_json::to_value(truncation).unwrap_or_default(),
            );
        }
        if let Some(full_output_path) = &self.full_output_path {
            object.insert("fullOutputPath".to_owned(), json!(full_output_path));
        }
        Value::Object(object)
    }
}

/// Validate and convert the timeout to milliseconds, upstream's
/// `resolveTimeoutMs`.
///
/// # Errors
/// The two rejection messages, upstream's throws.
pub fn resolve_timeout_ms(timeout: Option<f64>) -> Result<Option<Duration>, AgentToolError> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    if !timeout.is_finite() || timeout <= 0.0 {
        return Err(AgentToolError::from(std::io::Error::other(
            "Invalid timeout: must be a finite number of seconds",
        )));
    }
    let timeout_ms = timeout * 1000.0;
    if timeout_ms > MAX_TIMEOUT_MS {
        return Err(AgentToolError::from(std::io::Error::other(format!(
            "Invalid timeout: maximum is {} seconds",
            format_seconds(MAX_TIMEOUT_MS / 1000.0)
        ))));
    }
    Ok(Some(Duration::from_secs_f64(timeout_ms / 1000.0)))
}

/// The JS `String(number)` rendering for the timeout messages.
pub(crate) fn format_seconds(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "integral f64 values print bare like JS String(number)"
        )]
        let integer = value as i64;
        format!("{integer}")
    } else {
        format!("{value}")
    }
}

// ============================================================================
// Operations seam
// ============================================================================

/// The chunk listener both pipes feed, upstream's shared `onData`.
pub type OnDataListener = Arc<dyn Fn(&[u8]) + Send + Sync>;

/// The execution options one operations call receives, upstream's
/// `{ onData, signal, timeout, env }`.
pub struct BashExecOptions {
    /// The chunk listener both pipes feed, upstream's `onData`.
    pub on_data: OnDataListener,
    /// The abort signal cancelling the command.
    pub signal: Option<AbortSignal>,
    /// The timeout in seconds, upstream's `timeout`.
    pub timeout: Option<f64>,
    /// The explicit environment; absent, the shell environment runs,
    /// upstream's `env ?? getShellEnv()`.
    pub env: Option<BTreeMap<String, String>>,
}

impl std::fmt::Debug for BashExecOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashExecOptions")
            .field("on_data", &"<callback>")
            .field("signal", &self.signal)
            .field("timeout", &self.timeout)
            .field("env", &self.env)
            .finish()
    }
}

/// The execution outcome, upstream's `{ exitCode }` with its `null` for a
/// signal death.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BashExecOutcome {
    /// The exit code, absent when a signal killed the process.
    pub exit_code: Option<i32>,
}

/// The pluggable operations for the shell tools, upstream's
/// `BashOperations`. Override to delegate command execution to remote
/// systems (for example SSH).
#[derive(Clone)]
pub struct BashOperations {
    /// Execute a command and stream output, upstream's `exec`. Errors carry
    /// the `aborted` and `timeout:<seconds>` sentinels the tool maps.
    pub exec: BashExecFn,
}

/// The erased executor behind [`BashOperations::exec`].
pub type BashExecFn = Arc<
    dyn for<'a> Fn(
            &'a str,
            &'a str,
            BashExecOptions,
        ) -> BoxedFuture<'a, Result<BashExecOutcome, AgentToolError>>
        + Send
        + Sync,
>;

impl std::fmt::Debug for BashOperations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashOperations").finish_non_exhaustive()
    }
}

/// The io error message carrying the node-style code the spawn-failure
/// tests regex match, upstream's `Error.code` surface (`spawn <shell>
/// ENOENT`).
pub(crate) fn io_error_message(error: &std::io::Error) -> String {
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => "ENOENT",
        std::io::ErrorKind::PermissionDenied => "EACCES",
        std::io::ErrorKind::AlreadyExists => "EEXIST",
        std::io::ErrorKind::DirectoryNotEmpty => "ENOTEMPTY",
        std::io::ErrorKind::NotADirectory => "ENOTDIR",
        std::io::ErrorKind::IsADirectory => "EISDIR",
        _ => return error.to_string(),
    };
    code.to_owned()
}

/// Shared process execution used by the built-in shell tools, upstream's
/// `createLocalShellOperations`.
#[expect(
    clippy::too_many_lines,
    reason = "the exec body mirrors upstream's single closure: spawn, stream, abort, timeout, and the settle checks"
)]
pub fn create_local_shell_operations(
    shell_name: &'static str,
    resolve_shell_config: impl Fn() -> Result<ShellConfig, ShellError> + Send + Sync + 'static,
) -> BashOperations {
    let resolver = Arc::new(resolve_shell_config);
    let exec: BashExecFn = Arc::new(move |command: &str, cwd: &str, options: BashExecOptions| {
        let resolver = Arc::clone(&resolver);
        let shell_name = shell_name;
        Box::pin(async move {
            let timeout = resolve_timeout_ms(options.timeout)?;
            if options.signal.as_ref().is_some_and(AbortSignal::aborted) {
                return Err(AgentToolError::from(std::io::Error::other("aborted")));
            }
            let shell_config =
                resolver().map_err(|error| AgentToolError::from(std::io::Error::other(error.0)))?;
            if !std::path::Path::new(cwd).exists() {
                return Err(AgentToolError::from(std::io::Error::other(format!(
                    "Working directory does not exist: {cwd}\nCannot execute {shell_name} commands."
                ))));
            }

            let command_from_stdin =
                shell_config.command_transport == Some(CommandTransport::Stdin);
            let mut spawn = tokio::process::Command::new(&shell_config.shell);
            if command_from_stdin {
                spawn.args(&shell_config.args);
                spawn.stdin(std::process::Stdio::piped());
            } else {
                if !shell_config.args.is_empty() {
                    spawn.args(shell_config.args.iter().map(String::as_str));
                }
                spawn.arg(command);
                spawn.stdin(std::process::Stdio::null());
            }
            spawn
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .current_dir(cwd);
            // detached: true, upstream's new process group.
            #[cfg(unix)]
            spawn.process_group(0);
            spawn.env_clear();
            if let Some(env) = &options.env {
                spawn.envs(
                    env.iter()
                        .map(|(key, value)| (key.as_str(), value.as_str())),
                );
            } else {
                spawn.envs(get_shell_env());
            }
            let mut child = spawn.spawn().map_err(|error| {
                AgentToolError::from(std::io::Error::other(format!(
                    "spawn {} {}",
                    shell_config.shell,
                    io_error_message(&error)
                )))
            })?;

            if command_from_stdin && let Some(mut stdin) = child.stdin.take() {
                use tokio::io::AsyncWriteExt;
                // The stdin error upstream swallows (`on("error", () => {})
                // `) is the write failing when the child died first.
                let _ignored = stdin.write_all(command.as_bytes()).await;
                let _closed = stdin.shutdown().await;
            }
            let pid = child.id();
            if let Some(pid) = pid {
                crate::utils::shell::track_detached_child_pid(pid);
            }
            let timed_out = Arc::new(std::sync::atomic::AtomicBool::new(false));

            // The abort arm, upstream's abort listener killing the tree.
            let abort_task = options.signal.as_ref().map(|signal| {
                let signal = signal.clone();
                tokio::spawn(async move {
                    if !signal.aborted() {
                        signal.wait().await;
                    }
                    if let Some(pid) = pid {
                        crate::utils::shell::kill_process_tree(pid);
                    }
                })
            });

            // The timeout arm, upstream's setTimeout arming `timedOut` and
            // killing the tree.
            let timeout_task = timeout.map(|timeout| {
                let timed_out = Arc::clone(&timed_out);
                tokio::spawn(async move {
                    tokio::time::sleep(timeout).await;
                    timed_out.store(true, std::sync::atomic::Ordering::SeqCst);
                    if let Some(pid) = pid {
                        crate::utils::shell::kill_process_tree(pid);
                    }
                })
            });

            let on_data = Arc::clone(&options.on_data);
            let code = crate::utils::child_process::wait_for_child_process_streaming(
                child,
                Arc::new(move |chunk: &[u8], _is_stderr: bool| on_data(chunk)),
            )
            .await;

            if let Some(task) = abort_task {
                task.abort();
            }
            if let Some(task) = timeout_task {
                task.abort();
            }
            if let Some(pid) = pid {
                crate::utils::shell::untrack_detached_child_pid(pid);
            }

            if options.signal.as_ref().is_some_and(AbortSignal::aborted) {
                return Err(AgentToolError::from(std::io::Error::other("aborted")));
            }
            if timed_out.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(AgentToolError::from(std::io::Error::other(format!(
                    "timeout:{}",
                    format_seconds(options.timeout.unwrap_or_default())
                ))));
            }
            Ok(BashExecOutcome { exit_code: code })
        })
    });
    BashOperations { exec }
}

/// Create bash operations using pi's built-in local shell execution
/// backend, upstream's `createLocalBashOperations`.
///
/// Useful for extensions that intercept `user_bash` and still want pi's
/// standard local shell behavior while wrapping or rewriting commands.
#[must_use]
pub fn create_local_bash_operations(shell_path: Option<&str>) -> BashOperations {
    let shell_path = shell_path.map(str::to_owned);
    create_local_shell_operations("bash", move || get_shell_config(shell_path.as_deref()))
}

// ============================================================================
// Spawn context
// ============================================================================

/// The spawn the hook may adjust, upstream's `BashSpawnContext`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BashSpawnContext {
    /// The command to run.
    pub command: String,
    /// The working directory.
    pub cwd: String,
    /// The environment.
    pub env: BTreeMap<String, String>,
}

/// The hook adjusting command, cwd, or env before execution, upstream's
/// `BashSpawnHook`.
pub type BashSpawnHook = Arc<dyn Fn(BashSpawnContext) -> BashSpawnContext + Send + Sync>;

/// The session-scoped `PI_*` names the spawn environment clears before the
/// session environment re-adds its own, upstream's five `delete`s.
const SESSION_ENV_KEYS: [&str; 5] = [
    "PI_SESSION_ID",
    "PI_SESSION_FILE",
    "PI_PROVIDER",
    "PI_MODEL",
    "PI_REASONING_LEVEL",
];

/// Resolve the spawn context, upstream's `resolveSpawnContext`.
#[must_use]
pub fn resolve_spawn_context(
    command: &str,
    cwd: &str,
    spawn_hook: Option<&BashSpawnHook>,
    expose_session_environment: bool,
    ctx: Option<&dyn ExtensionContext>,
) -> BashSpawnContext {
    let mut env = get_shell_env();
    for key in SESSION_ENV_KEYS {
        env.remove(key);
    }
    if expose_session_environment && let Some(ctx) = ctx {
        if let Some(session_id) = ctx.session_id() {
            env.insert("PI_SESSION_ID".to_owned(), session_id);
        }
        if let Some(session_file) = ctx.session_file() {
            env.insert("PI_SESSION_FILE".to_owned(), session_file);
        }
        if let Some(model) = ctx.model() {
            env.insert("PI_PROVIDER".to_owned(), model.provider.to_string());
            env.insert("PI_MODEL".to_owned(), model.id.clone());
        }
        if let Some(level) = ctx.thinking_level()
            && let Some(rendered) = serde_json::to_value(level)
                .ok()
                .and_then(|value| value.as_str().map(str::to_owned))
        {
            env.insert("PI_REASONING_LEVEL".to_owned(), rendered);
        }
    }
    let base = BashSpawnContext {
        command: command.to_owned(),
        cwd: cwd.to_owned(),
        env,
    };
    spawn_hook.map_or_else(|| base.clone(), |hook| hook(base.clone()))
}

// ============================================================================
// Tool options and definition
// ============================================================================

/// The shell tool's options, upstream's `BashToolOptions`.
#[derive(Clone, Default)]
pub struct BashToolOptions {
    /// Custom operations for command execution; default, local shell.
    pub operations: Option<BashOperations>,
    /// Command prefix prepended to every command.
    pub command_prefix: Option<String>,
    /// Optional explicit shell path from settings.
    pub shell_path: Option<String>,
    /// Expose the current session's `PI_*` metadata; default true.
    pub expose_session_environment: Option<bool>,
    /// The hook adjusting the spawn before execution.
    pub spawn_hook: Option<BashSpawnHook>,
}

impl std::fmt::Debug for BashToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashToolOptions")
            .field("operations", &self.operations.is_some())
            .field("command_prefix", &self.command_prefix)
            .field("shell_path", &self.shell_path)
            .field(
                "expose_session_environment",
                &self.expose_session_environment,
            )
            .field("spawn_hook", &self.spawn_hook.is_some())
            .finish()
    }
}

/// The shell tool's configuration, upstream's `ShellToolConfig`.
#[derive(Clone, Copy, Debug)]
pub struct ShellToolConfig {
    /// Tool name.
    pub name: &'static str,
    /// Human-readable label.
    pub label: &'static str,
    /// The shell the description names.
    pub shell_name: &'static str,
    /// The prompt the renderer displays.
    pub prompt: &'static str,
    /// The prompt snippet.
    pub prompt_snippet: &'static str,
    /// The prompt guidelines.
    pub prompt_guidelines: &'static [&'static str],
    /// The temp file prefix.
    pub temp_file_prefix: &'static str,
}

/// The shell tools' input schema, upstream's `bashSchema`.
fn bash_schema() -> Value {
    tool_schema(
        &json!({
            "command": {
                "type": "string",
                "description": "Shell command to execute"
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, no default timeout)"
            }
        }),
        &["command"],
    )
}

fn parse_input(params: &Value) -> Result<BashToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        command: String,
        timeout: Option<f64>,
    }
    serde_json::from_value::<RawInput>(params.clone())
        .map(|raw| BashToolInput {
            command: raw.command,
            timeout: raw.timeout,
        })
        .map_err(|error| AgentToolError::from(std::io::Error::other(error.to_string())))
}

/// The status line join, upstream's `appendStatus`:
/// `` `${text ? `${text}\n\n` : ""}${status}` ``.
fn append_status(text: &str, status: &str) -> String {
    if text.is_empty() {
        status.to_owned()
    } else {
        format!("{text}\n\n{status}")
    }
}

/// The truncation footer formatting, upstream's `formatOutput`.
fn format_output(
    last_line_bytes: usize,
    snapshot: &super::output_accumulator::OutputSnapshot,
    empty_text: &str,
) -> (String, Option<BashToolDetails>) {
    let truncation = &snapshot.truncation;
    let mut text = if snapshot.content.is_empty() {
        empty_text.to_owned()
    } else {
        snapshot.content.clone()
    };
    let mut details = None;
    if truncation.truncated {
        details = Some(BashToolDetails {
            truncation: Some(truncation.clone()),
            full_output_path: snapshot.full_output_path.clone(),
        });
        let start_line = truncation.total_lines - truncation.output_lines + 1;
        let end_line = truncation.total_lines;
        let full_output_path = snapshot.full_output_path.as_deref().unwrap_or_default();
        if truncation.last_line_partial {
            let last_line_size = format_size(last_line_bytes);
            let _ = write!(
                text,
                "\n\n[Showing last {} of line {end_line} (line is {last_line_size}). Full output: {full_output_path}]",
                format_size(truncation.output_bytes)
            );
        } else if truncation.truncated_by == Some(TruncatedBy::Lines) {
            let _ = write!(
                text,
                "\n\n[Showing lines {start_line}-{end_line} of {}. Full output: {full_output_path}]",
                truncation.total_lines
            );
        } else {
            let _ = write!(
                text,
                "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit). Full output: {full_output_path}]",
                truncation.total_lines,
                format_size(DEFAULT_MAX_BYTES)
            );
        }
    }
    (text, details)
}

/// The streaming-update throttle state, upstream's `updateDirty` /
/// `lastUpdateAt` / `updateTimer` trio.
struct Throttle {
    dirty: std::sync::atomic::AtomicBool,
    last_update_at: Mutex<Option<tokio::time::Instant>>,
}

/// The shared execute state the data handler and the pump race on.
struct ExecuteState {
    output: OutputAccumulator,
    throttle: Throttle,
    accepting_output: std::sync::atomic::AtomicBool,
}

impl ExecuteState {
    /// Emit the pending snapshot, upstream's `emitOutputUpdate`: skip when
    /// no callback is bound or nothing went dirty since the last emit.
    fn emit(&self, on_update: Option<AgentHarnessToolUpdateCallback<'_>>) {
        let Some(on_update) = on_update else { return };
        if !self
            .throttle
            .dirty
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        *self
            .throttle
            .last_update_at
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(tokio::time::Instant::now());
        let snapshot = self.output.snapshot(true);
        let details = BashToolDetails {
            truncation: if snapshot.truncation.truncated {
                Some(snapshot.truncation.clone())
            } else {
                None
            },
            full_output_path: snapshot.full_output_path.clone(),
        };
        let update = AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: if snapshot.content.is_empty() {
                    String::new()
                } else {
                    snapshot.content
                },
                text_signature: None,
            })],
            details: details.to_wire(),
            usage: None,
            added_tool_names: None,
            terminate: None,
        };
        on_update(&update, None);
    }

    /// Schedule the next snapshot emit, upstream's `scheduleOutputUpdate`:
    /// an immediate emit when the spacing has elapsed, else arm the one
    /// trailing timer (`??=`).
    fn schedule(
        &self,
        on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
        timer_deadline: &mut Option<tokio::time::Instant>,
    ) {
        if on_update.is_none() {
            return;
        }
        self.throttle
            .dirty
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let now = tokio::time::Instant::now();
        let elapsed_since_last = self
            .throttle
            .last_update_at
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .map_or(u128::from(BASH_UPDATE_THROTTLE_MS) + 1, |last| {
                now.duration_since(last).as_millis()
            });
        let delay = i128::from(BASH_UPDATE_THROTTLE_MS)
            - i128::try_from(elapsed_since_last).unwrap_or(i128::MAX);
        if delay <= 0 {
            // The immediate emit, upstream's clearUpdateTimer + emit.
            *timer_deadline = None;
            self.emit(on_update);
        } else if timer_deadline.is_none() {
            *timer_deadline =
                Some(now + Duration::from_millis(u64::try_from(delay).unwrap_or_default()));
        }
    }

    /// Complete the stream, upstream's `finishOutput`: latch the accepting
    /// flag (the late-output guard), flush, emit the final snapshot, and
    /// close the temp file.
    async fn finish(
        &self,
        on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
    ) -> super::output_accumulator::OutputSnapshot {
        self.accepting_output
            .store(false, std::sync::atomic::Ordering::SeqCst);
        self.output.finish();
        self.emit(on_update);
        let snapshot = self.output.snapshot(true);
        let _closed = self.output.close_temp_file().await;
        snapshot
    }
}

/// The shell definition's captured runtime, the constants the execute
/// closure clones per invocation (upstream's closure-captured locals).
struct ShellToolRuntime {
    ops: BashOperations,
    command_prefix: Option<String>,
    expose_session_environment: bool,
    spawn_hook: Option<BashSpawnHook>,
    cwd: String,
    temp_file_prefix: &'static str,
}

/// The shell tool's execution body, upstream's `createShellToolDefinition`
/// execute.
#[expect(
    clippy::too_many_lines,
    reason = "the body mirrors upstream's single execute: the spawn, the throttled pump, the finish arms, and the two settle mappings"
)]
async fn execute_shell_tool(
    params: &Value,
    signal: Option<&AbortSignal>,
    on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
    ctx: Option<&dyn ExtensionContext>,
    runtime: &ShellToolRuntime,
) -> Result<AgentToolResult, AgentToolError> {
    let input = parse_input(params)?;
    let resolved_command = runtime.command_prefix.as_deref().map_or_else(
        || input.command.clone(),
        |prefix| format!("{prefix}\n{}", input.command),
    );
    let effective_cwd = ctx
        .map(ExtensionContext::cwd)
        .filter(|ctx_cwd| !ctx_cwd.is_empty())
        .unwrap_or(runtime.cwd.as_str());
    let spawn_context = resolve_spawn_context(
        &resolved_command,
        effective_cwd,
        runtime.spawn_hook.as_ref(),
        runtime.expose_session_environment,
        ctx,
    );

    let state = Arc::new(ExecuteState {
        output: OutputAccumulator::new(OutputAccumulatorOptions {
            temp_file_prefix: Some(runtime.temp_file_prefix.to_owned()),
            ..OutputAccumulatorOptions::default()
        }),
        throttle: Throttle {
            dirty: std::sync::atomic::AtomicBool::new(false),
            last_update_at: Mutex::new(None),
        },
        accepting_output: std::sync::atomic::AtomicBool::new(true),
    });

    if let Some(on_update) = on_update {
        on_update(
            &AgentToolResult {
                content: Vec::new(),
                details: Value::Null,
                usage: None,
                added_tool_names: None,
                terminate: None,
            },
            None,
        );
    }

    // The chunk handler, upstream's `handleData`: latch-gated append and
    // schedule.
    let (dirty_tx, mut dirty_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let handle_data: OnDataListener = {
        let state = Arc::clone(&state);
        let dirty_tx = dirty_tx.clone();
        Arc::new(move |data: &[u8]| {
            if !state
                .accepting_output
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                return;
            }
            state.output.append(data);
            state
                .throttle
                .dirty
                .store(true, std::sync::atomic::Ordering::SeqCst);
            let _scheduled = dirty_tx.send(());
        })
    };

    let (settle_tx, mut settle_rx) = tokio::sync::oneshot::channel::<()>();
    let exec = {
        let ops = runtime.ops.clone();
        let command = spawn_context.command.clone();
        let spawn_cwd = spawn_context.cwd.clone();
        let env = spawn_context.env.clone();
        async move {
            let outcome = (ops.exec)(
                &command,
                &spawn_cwd,
                BashExecOptions {
                    on_data: handle_data,
                    signal: signal.cloned(),
                    timeout: input.timeout,
                    env: Some(env),
                },
            )
            .await;
            let _settled = settle_tx.send(());
            outcome
        }
    };

    // The throttle pump, upstream's updateTimer: drains the dirty signals,
    // emits when the spacing has elapsed, and fires the one trailing timer.
    let pump_state = Arc::clone(&state);
    let pump = async {
        let mut timer_deadline: Option<tokio::time::Instant> = None;
        loop {
            tokio::select! {
                _ = &mut settle_rx => break,
                _ = dirty_rx.recv() => {
                    pump_state.schedule(on_update, &mut timer_deadline);
                }
                () = tokio::time::sleep_until(timer_deadline.unwrap_or_else(tokio::time::Instant::now)),
                    if timer_deadline.is_some() => {
                    timer_deadline = None;
                    pump_state.emit(on_update);
                }
            }
        }
    };
    let (result, ()) = tokio::join!(exec, pump);

    match result {
        Err(error) => {
            let snapshot = state.finish(on_update).await;
            let (text, _) = format_output(state.output.get_last_line_bytes(), &snapshot, "");
            let message = error.to_string();
            if message == "aborted" {
                return Err(AgentToolError::from(std::io::Error::other(append_status(
                    &text,
                    "Command aborted",
                ))));
            }
            if let Some(timeout_secs) = message.strip_prefix("timeout:") {
                return Err(AgentToolError::from(std::io::Error::other(append_status(
                    &text,
                    &format!("Command timed out after {timeout_secs} seconds"),
                ))));
            }
            Err(error)
        }
        Ok(settled) => {
            let snapshot = state.finish(on_update).await;
            let (output_text, details) =
                format_output(state.output.get_last_line_bytes(), &snapshot, "(no output)");
            if settled.exit_code.is_some_and(|exit_code| exit_code != 0) {
                return Err(AgentToolError::from(std::io::Error::other(append_status(
                    &output_text,
                    &format!(
                        "Command exited with code {}",
                        settled.exit_code.unwrap_or_default()
                    ),
                ))));
            }
            Ok(AgentToolResult {
                content: vec![AgentToolContent::Text(TextContent {
                    text: output_text,
                    text_signature: None,
                })],
                details: details.map_or(Value::Null, |details| details.to_wire()),
                usage: None,
                added_tool_names: None,
                terminate: None,
            })
        }
    }
}

/// Build the shell tool definition, upstream's `createShellToolDefinition`.
#[must_use]
pub fn create_shell_tool_definition(
    cwd: &str,
    config: ShellToolConfig,
    options: Option<BashToolOptions>,
) -> ToolDefinition {
    let options = options.unwrap_or_default();
    let ops = options
        .operations
        .unwrap_or_else(|| create_local_bash_operations(options.shell_path.as_deref()));
    let runtime = Arc::new(ShellToolRuntime {
        ops,
        command_prefix: options.command_prefix.clone(),
        expose_session_environment: options.expose_session_environment.unwrap_or(true),
        spawn_hook: options.spawn_hook.clone(),
        cwd: cwd.to_owned(),
        temp_file_prefix: config.temp_file_prefix,
    });

    ToolDefinition {
        name: config.name.to_owned(),
        label: config.label.to_owned(),
        description: format!(
            "Execute a {} command in the current working directory. Returns stdout and stderr. Output is truncated to last {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
            config.shell_name,
            DEFAULT_MAX_BYTES / 1024
        ),
        prompt_snippet: Some(config.prompt_snippet.to_owned()),
        prompt_guidelines: if runtime.expose_session_environment
            && !config.prompt_guidelines.is_empty()
        {
            Some(
                config
                    .prompt_guidelines
                    .iter()
                    .map(|guideline| (*guideline).to_owned())
                    .collect(),
            )
        } else {
            None
        },
        parameters: bash_schema(),
        constrained_sampling: Some(strict_sampling()),
        render_shell: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(
            move |_tool_call_id: &str, params: &Value, signal, on_update, ctx| {
                // The extension-facing body ignores the wrapper's call id: the
                // shell tool's updates and errors carry no per-call state.
                let runtime = Arc::clone(&runtime);
                Box::pin(async move {
                    execute_shell_tool(params, signal, on_update, ctx, &runtime).await
                })
            },
        ),
    }
}

const BASH_TOOL_CONFIG: ShellToolConfig = ShellToolConfig {
    name: "bash",
    label: "bash",
    shell_name: "bash",
    prompt: "$",
    prompt_snippet: BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet,
    prompt_guidelines: BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION.guidelines,
    temp_file_prefix: "pi-bash",
};

/// Build the bash tool definition, upstream's `createBashToolDefinition`.
#[must_use]
pub fn create_bash_tool_definition(cwd: &str, options: Option<BashToolOptions>) -> ToolDefinition {
    create_shell_tool_definition(cwd, BASH_TOOL_CONFIG, options)
}

/// Build the bash tool, upstream's `createBashTool` (the wrapped
/// definition; upstream's prompt-metadata `Object.assign` has no harness
/// tool counterpart).
#[must_use]
pub fn create_bash_tool(cwd: &str, options: Option<BashToolOptions>) -> AgentHarnessTool {
    let definition = create_bash_tool_definition(cwd, options);
    crate::tools::tool_definition_wrapper::wrap_cwd_tool(definition)
}
