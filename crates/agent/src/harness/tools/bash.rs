//! The bash tool, ported from upstream `src/harness/tools/bash.ts`.
//!
//! Upstream's `onUpdate` wrapper runs inside the environment's exec call
//! and forwards bounded-view snapshots to the tool's update callback with
//! the checkpoint flag; the callback reference is a borrow of the execute
//! scope while [`crate::harness::types::ShellExecOptions::on_update`] is an
//! owned `'static` handle, so the port bridges the two with an unbounded
//! channel drained inside the execute future — `tokio::join!` keeps the
//! drain alive exactly as long as the callback borrow, and the oneshot
//! `done` signal plus the final `try_recv` sweep forward every update the
//! environment published during execution while post-settle updates (the
//! `acceptingUpdates` latch's target) go unread.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::sync::Arc;
use std::time::Duration;

use pi_ai::types::{BoxedFuture, TextContent, Tool};
use serde_json::json;

use crate::harness::context::Context;
use crate::harness::types::{
    AgentHarnessTool, AgentHarnessToolExecuteFn, AgentHarnessToolUpdateCallback,
    AgentHarnessToolUpdateOptions, ExecutionError, ExecutionErrorCode, OnShellOutputUpdate,
    ShellExecOptions, ShellOutputCaptureOptions, ShellOutputLimits, ShellOutputRetention,
    ShellOutputTruncation, ShellOutputUpdate, ShellOutputView,
};
use crate::harness::utils::output_capture::apply_shell_output_update;
use crate::harness::utils::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size};
use crate::types::AgentToolError;
use crate::types::{AgentToolContent, AgentToolResult};

use super::tool_context::{ExecutionToolContext, execution_tool_context, io_error};

/// The maximum accepted timeout, upstream's `MAX_TIMEOUT_SECONDS`
/// (`2147483647 / 1000`, the int32 millisecond bound in seconds).
const MAX_TIMEOUT_SECONDS: f64 = 2_147_483_647.0 / 1000.0;

/// The minimum spacing between checkpoint-flagged updates, upstream's
/// `BASH_CHECKPOINT_INTERVAL_MS`.
const BASH_CHECKPOINT_INTERVAL_MS: u64 = 2_000;

/// The bash tool's input schema, upstream's `bashSchema`.
fn bash_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Bash command to execute"
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, no default timeout)"
            }
        },
        "required": ["command"]
    })
}

/// The parsed bash-tool input, upstream's `BashToolInput`.
#[derive(Clone, Debug, PartialEq)]
pub struct BashToolInput {
    /// The command to run.
    pub command: String,
    /// The timeout in seconds.
    pub timeout: Option<f64>,
}

/// The execution the prepare hook mutates, upstream's `BashExecution`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BashExecution {
    /// The command to run, prefixed when the tool carries a prefix.
    pub command: String,
    /// The working directory, seeded from the environment's cwd.
    pub cwd: String,
    /// The explicit environment variables, empty by default.
    pub env: BTreeMap<String, String>,
    /// Whether the environment's default variables are inherited.
    pub inherit_env: bool,
}

/// The prepare hook, upstream's `BashPrepare<TContext>`: it mutates the
/// execution with the resolved tool context and the call context.
pub type BashPrepare<C> = Arc<
    dyn for<'a> Fn(&'a mut BashExecution, &'a C, &'a Context) -> BoxedFuture<'a, ()> + Send + Sync,
>;

/// The bash tool's options, upstream's `BashToolOptions<TContext>`.
pub struct BashToolOptions<C> {
    /// A command prefix joined with `\n` ahead of the call's command.
    pub command_prefix: Option<String>,
    /// The prepare hook mutating the execution before it runs.
    pub prepare: Option<BashPrepare<C>>,
}

impl<C> Clone for BashToolOptions<C> {
    fn clone(&self) -> Self {
        Self {
            command_prefix: self.command_prefix.clone(),
            prepare: self.prepare.clone(),
        }
    }
}

impl<C> fmt::Debug for BashToolOptions<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BashToolOptions")
            .field("command_prefix", &self.command_prefix)
            .field("prepare", &self.prepare.is_some())
            .finish()
    }
}

/// The bash tool's structured details, upstream's `BashToolDetails`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BashToolDetails {
    /// The truncation metadata, when the output truncated.
    pub truncation: Option<ShellOutputTruncation>,
    /// The spill file preserving complete output.
    pub full_output_path: Option<String>,
}

/// The bash tool's failure, upstream's `throw new Error(message, { cause })`.
#[derive(Debug)]
struct BashToolError {
    message: String,
    source: Option<ExecutionError>,
}

impl fmt::Display for BashToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BashToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|source| {
            let coerced: &(dyn std::error::Error + 'static) = source;
            coerced
        })
    }
}

fn validate_timeout(timeout: Option<f64>) -> Result<(), AgentToolError> {
    let Some(timeout) = timeout else {
        return Ok(());
    };
    if !timeout.is_finite() || timeout <= 0.0 {
        return Err(Box::new(std::io::Error::other(
            "Invalid timeout: must be a finite number of seconds",
        )));
    }
    if timeout > MAX_TIMEOUT_SECONDS {
        return Err(Box::new(std::io::Error::other(format!(
            "Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"
        ))));
    }
    Ok(())
}

/// Builds the bash tool, upstream's `createBashTool`.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "the executor mirrors upstream's single createBashTool execute body: the drain bridge, the checkpoint pacing, and the settle formatting read the same state"
)]
pub fn create_bash_tool<C: ExecutionToolContext>(
    options: Option<BashToolOptions<C>>,
) -> AgentHarnessTool {
    let options = options.unwrap_or(BashToolOptions {
        command_prefix: None,
        prepare: None,
    });
    let execute: Arc<AgentHarnessToolExecuteFn> = Arc::new(
        move |_tool_call_id: &str,
              args: &serde_json::Value,
              on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
              tool_context,
              _invocation,
              context: &Context| {
            let options = options.clone();
            Box::pin(async move {
                let input: BashToolInput = parse_input(args)?;
                validate_timeout(input.timeout)?;
                let context_holder = execution_tool_context::<C>(&tool_context)?;
                let env = context_holder.env();
                let mut execution = BashExecution {
                    command: options.command_prefix.as_ref().map_or_else(
                        || input.command.clone(),
                        |prefix| format!("{prefix}\n{}", input.command),
                    ),
                    cwd: env.cwd().to_owned(),
                    env: BTreeMap::new(),
                    inherit_env: true,
                };
                if let Some(prepare) = options.prepare.as_ref() {
                    prepare(&mut execution, context_holder.as_ref(), context).await;
                }

                if let Some(outer) = on_update {
                    outer(
                        &AgentToolResult {
                            content: Vec::new(),
                            details: serde_json::Value::Null,
                            usage: None,
                            added_tool_names: None,
                            terminate: None,
                        },
                        None,
                    );
                }

                let (update_tx, mut update_rx) = tokio::sync::mpsc::unbounded_channel();
                let listener: Arc<OnShellOutputUpdate> = Arc::new({
                    let update_tx = update_tx.clone();
                    move |update: &ShellOutputUpdate, _context: &Context| {
                        let _ = update_tx.send(update.clone());
                    }
                });
                drop(update_tx);
                let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();

                let view: Arc<std::sync::Mutex<Option<ShellOutputView>>> =
                    Arc::new(std::sync::Mutex::new(None));
                let drain_view = Arc::clone(&view);
                let drain = async move {
                    let mut last_checkpoint_at = tokio::time::Instant::now();
                    let mut last_checkpoint: Option<String> = None;
                    let mut done_rx = done_rx;
                    // Every forwarded update runs the same snapshot-and-
                    // checkpoint step, upstream's single onUpdate wrapper.
                    let mut forward = |update: &ShellOutputUpdate| {
                        let snapshot = {
                            let mut slot = drain_view
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            let next = apply_shell_output_update(slot.as_ref(), update);
                            let snapshot = snapshot_from_view(&next);
                            *slot = Some(next);
                            snapshot
                        };
                        let encoded = serde_json::to_string(&snapshot).unwrap_or_default();
                        let now = tokio::time::Instant::now();
                        let checkpoint = now.duration_since(last_checkpoint_at)
                            >= Duration::from_millis(BASH_CHECKPOINT_INTERVAL_MS)
                            && last_checkpoint.as_ref() != Some(&encoded);
                        if let Some(outer) = on_update {
                            outer(
                                &snapshot,
                                checkpoint
                                    .then_some(AgentHarnessToolUpdateOptions { checkpoint: true }),
                            );
                        }
                        if checkpoint {
                            last_checkpoint_at = now;
                            last_checkpoint = Some(encoded);
                        }
                    };
                    loop {
                        tokio::select! {
                            update = update_rx.recv() => {
                                let Some(update) = update else { break };
                                forward(&update);
                            }
                            _ = &mut done_rx => {
                                // The execution settled: forward everything
                                // already queued, then stop reading.
                                while let Ok(update) = update_rx.try_recv() {
                                    forward(&update);
                                }
                                break;
                            }
                        }
                    }
                };
                let exec = async {
                    let exec_options = ShellExecOptions {
                        cwd: Some(execution.cwd.clone()),
                        env: Some(execution.env.clone()),
                        inherit_env: Some(execution.inherit_env),
                        timeout: input.timeout,
                        capture: Some(ShellOutputCaptureOptions {
                            limits: ShellOutputLimits {
                                max_bytes: DEFAULT_MAX_BYTES,
                                max_lines: DEFAULT_MAX_LINES,
                                retain: Some(ShellOutputRetention::Tail),
                            },
                            spill: true,
                        }),
                        on_update: Some(listener),
                    };
                    let result = env
                        .exec(&execution.command, Some(exec_options), context)
                        .await;
                    let _ = done_tx.send(());
                    result
                };
                let (result, ()) = tokio::join!(exec, drain);

                let output_text = view
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .as_ref()
                    .map_or_else(String::new, |view| view.text.clone());
                let (truncation, spill_path, last_line_bytes) = result.as_ref().ok().map_or_else(
                    || {
                        let guard = view
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        guard.as_ref().map_or_else(
                            || (ShellOutputTruncation::default(), None, None),
                            |view| {
                                (
                                    view.metadata.truncation.clone(),
                                    view.metadata.spill_path.clone(),
                                    view.metadata.last_line_bytes,
                                )
                            },
                        )
                    },
                    |result| {
                        (
                            result.truncation.clone(),
                            result.spill_path.clone(),
                            result.last_line_bytes,
                        )
                    },
                );

                let mut output_text = output_text;
                let mut details = BashToolDetails::default();
                if truncation.truncated {
                    details = BashToolDetails {
                        truncation: Some(truncation.clone()),
                        full_output_path: spill_path.clone(),
                    };
                    let start_line = truncation.total_lines - truncation.output_lines + 1;
                    let end_line = truncation.total_lines;
                    if truncation.last_line_partial {
                        let last_line_size =
                            format_size(last_line_bytes.unwrap_or(truncation.output_bytes));
                        let _ = write!(
                            output_text,
                            "\n\n[Showing last {} of line {} (line is {last_line_size}). Full output: {}]",
                            format_size(truncation.output_bytes),
                            end_line,
                            spill_path.as_deref().unwrap_or_default()
                        );
                    } else if truncation.truncated_by
                        == Some(crate::harness::types::TruncatedBy::Lines)
                    {
                        let _ = write!(
                            output_text,
                            "\n\n[Showing lines {start_line}-{end_line} of {}. Full output: {}]",
                            truncation.total_lines,
                            spill_path.as_deref().unwrap_or_default()
                        );
                    } else {
                        let _ = write!(
                            output_text,
                            "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit). Full output: {}]",
                            truncation.total_lines,
                            format_size(DEFAULT_MAX_BYTES),
                            spill_path.as_deref().unwrap_or_default()
                        );
                    }
                }

                match result {
                    Err(error) => {
                        let status = match error.code {
                            ExecutionErrorCode::Timeout => format!(
                                "Command timed out after {} seconds",
                                format_seconds(input.timeout.unwrap_or_default())
                            ),
                            ExecutionErrorCode::Aborted => "Command aborted".to_owned(),
                            _ => error.message.clone(),
                        };
                        let error: AgentToolError = Box::new(BashToolError {
                            message: if output_text.is_empty() {
                                status
                            } else {
                                format!("{output_text}\n\n{status}")
                            },
                            source: Some(error),
                        });
                        Err(error)
                    }
                    Ok(result) if result.exit_code != 0 => Err(io_error(format!(
                        "{}Command exited with code {}",
                        if output_text.is_empty() {
                            String::new()
                        } else {
                            format!("{output_text}\n\n")
                        },
                        result.exit_code
                    ))),
                    Ok(_) => Ok(AgentToolResult {
                        content: vec![AgentToolContent::Text(TextContent {
                            text: if output_text.is_empty() {
                                "(no output)".to_owned()
                            } else {
                                output_text
                            },
                            text_signature: None,
                        })],
                        details: serialize_details(&details),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    }),
                }
            })
        },
    );
    AgentHarnessTool {
        tool: Tool {
            name: "bash".to_owned(),
            description: format!(
                "Execute a bash command in the current working directory. Returns combined stdout and stderr. Output is truncated to last {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
                DEFAULT_MAX_BYTES / 1024
            ),
            parameters: bash_schema(),
            constrained_sampling: None,
        },
        label: "bash".to_owned(),
        prepare_arguments: None,
        execute,
        replay: None,
        execution_mode: None,
    }
}

/// The snapshot update the bash tool forwards, upstream's
/// `{ content: [{ type: "text", text }], details }`.
fn snapshot_from_view(view: &ShellOutputView) -> AgentToolResult {
    let mut details = serde_json::Map::new();
    if view.metadata.truncation.truncated {
        details.insert(
            "truncation".to_owned(),
            serde_json::to_value(&view.metadata.truncation).unwrap_or_default(),
        );
    }
    if let Some(spill_path) = &view.metadata.spill_path {
        details.insert("fullOutputPath".to_owned(), json!(spill_path));
    }
    AgentToolResult {
        content: vec![AgentToolContent::Text(TextContent {
            text: view.text.clone(),
            text_signature: None,
        })],
        details: serde_json::Value::Object(details),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// Serializes the final details, upstream's `BashToolDetails` with absent
/// members dropped from the JSON object.
fn serialize_details(details: &BashToolDetails) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    if let Some(truncation) = &details.truncation {
        object.insert(
            "truncation".to_owned(),
            serde_json::to_value(truncation).unwrap_or_default(),
        );
    }
    if let Some(full_output_path) = &details.full_output_path {
        object.insert("fullOutputPath".to_owned(), json!(full_output_path));
    }
    serde_json::Value::Object(object)
}

/// The JS `String(number)` seconds rendering for the timeout status.
fn format_seconds(value: f64) -> String {
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

fn parse_input(args: &serde_json::Value) -> Result<BashToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        command: String,
        timeout: Option<f64>,
    }
    serde_json::from_value::<RawInput>(args.clone())
        .map(|raw| BashToolInput {
            command: raw.command,
            timeout: raw.timeout,
        })
        .map_err(|error| io_error(error.to_string()))
}
